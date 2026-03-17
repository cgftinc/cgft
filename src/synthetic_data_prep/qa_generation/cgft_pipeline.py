"""Unified Cgft QA generation pipeline."""

from __future__ import annotations

import inspect
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from tqdm.auto import tqdm as _tqdm

from synthetic_data_prep.corpus.corpora.source import CorporaChunkSource
from synthetic_data_prep.qa_generation.cgft_models import (
    CgftContext,
    CgftPipelineConfig,
    CgftRunStats,
    EntityExtractionConfig,
    GenerationTask,
    build_generation_tasks,
    load_cgft_config,
)
from synthetic_data_prep.qa_generation.filters import (
    DeterministicGuardsFilter,
    GroundingLLMFilter,
    RetrievalLLMFilter,
)
from synthetic_data_prep.qa_generation.formatters import TrainEvalFormatter
from synthetic_data_prep.qa_generation.generated_qa import FilterVerdict, GeneratedQA
from synthetic_data_prep.qa_generation.generators import DirectLLMGenerator, EnvRolloutGenerator
from synthetic_data_prep.qa_generation.helpers import render_template
from synthetic_data_prep.qa_generation.linkers import (
    RELATED_CHUNK_SYSTEM_PROMPT,
    RELATED_CHUNK_USER_TEMPLATE,
    LLMGuidedChunkLinker,
    StructuralChunkLinker,
)
from synthetic_data_prep.qa_generation.protocols import (
    ChunkLinker,
    EvaluatorFilter,
    QuestionGenerator,
    QuestionTransformer,
)
from synthetic_data_prep.qa_generation.response_parsers import parse_corpus_summary_response
from synthetic_data_prep.qa_generation.transformers import LLMStyleTransformer
from synthetic_data_prep.trainer.client import RolloutClient

logger = logging.getLogger(__name__)

_ENTITY_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert at analyzing documentation corpora to identify linkable entities, "
    "code patterns, and domain terminology that can be used for keyword search."
)

_ENTITY_EXTRACTION_USER_TEMPLATE = """\
Analyze these sample chunks from a documentation corpus and identify patterns for \
finding related content via keyword (BM25) search.

<samples>
{chunk_samples}
</samples>

Identify the following:
1. Named entities: product names, tools, services, APIs, brand names that appear across chunks
2. Code patterns: regex patterns for extracting function calls, file paths, config keys, properties
3. Domain terminology: technical terms, acronyms, and jargon specific to this corpus
4. Query templates: search phrase templates using {{entity}} as a placeholder

Return JSON only:
{{
  "entity_names": ["Name1", "Name2"],
  "code_patterns": {{
    "category_name": "<regex_pattern>"
  }},
  "domain_terms": ["term1", "term2"],
  "query_templates": ["{{entity}} setup", "configure {{entity}}", "{{entity}} guide"],
  "confidence": "high"
}}"""

_RETRIEVAL_TOO_EASY_FILTER_STAGE = "retrieval_too_easy_llm"
_GROUNDING_FILTER_STAGE = "grounding_llm"
_SUPPORTED_FILTER_STAGES = (
    _GROUNDING_FILTER_STAGE,
    _RETRIEVAL_TOO_EASY_FILTER_STAGE,
)


def _build_openai_client(*, api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def _is_retryable_openai_error(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    # BadRequestError is a subclass of APIError — check it first so the specific
    # logic below takes precedence over the broad APIError fallthrough.
    if isinstance(exc, BadRequestError):
        code = str(getattr(exc, "code", "") or "").strip().lower()
        body = str(getattr(exc, "body", "") or "").strip().lower()
        message = str(exc).strip().lower()
        return code == "upstream_error" or "internal error" in body or "internal error" in message
    if isinstance(exc, APIError):
        return True
    return False


def _chat_completion_with_retry(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, str] | None = None,
    temperature: float | None = None,
    max_attempts: int = 3,
    initial_delay_seconds: float = 0.75,
) -> Any:
    delay = max(0.1, float(initial_delay_seconds))
    last_exc: Exception | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": messages}
            if response_format is not None:
                kwargs["response_format"] = response_format
            if temperature is not None:
                kwargs["temperature"] = temperature
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_attempts or not _is_retryable_openai_error(exc):
                raise
            logger.warning(
                "Transient LLM error on attempt %d/%d; retrying in %.2fs: %s",
                attempt,
                max_attempts,
                delay,
                exc,
            )
            time.sleep(delay)
            delay *= 2.0
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Completion retry loop exited unexpectedly.")


def _load_source(cfg: CgftPipelineConfig) -> CorporaChunkSource:
    source = CorporaChunkSource(
        api_key=cfg.platform.api_key,
        corpus_name=cfg.corpus.corpus_name,
        base_url=cfg.platform.base_url,
    )
    if cfg.corpus.docs_path:
        source.populate_from_folder(cfg.corpus.docs_path, show_summary=cfg.corpus.show_summary)
        return source
    if cfg.corpus.corpus_id:
        source.populate_from_existing_corpus(cfg.corpus.corpus_id, show_summary=cfg.corpus.show_summary)
        return source
    source.populate_from_existing_corpus_name(
        cfg.corpus.corpus_name,
        show_summary=cfg.corpus.show_summary,
    )
    return source


def _build_rollout_client(cfg: CgftPipelineConfig) -> RolloutClient:
    return RolloutClient(api_key=cfg.platform.api_key)


def _build_linker(cfg: CgftPipelineConfig, source: Any) -> ChunkLinker:
    if cfg.linker.type == "llm_guided":
        linker_cfg = cfg.linker.llm_guided
        linker_client = _build_openai_client(api_key=linker_cfg.api_key, base_url=linker_cfg.base_url)
        return LLMGuidedChunkLinker(
            source=source,
            client=linker_client,
            model=linker_cfg.model,
            system_prompt=linker_cfg.system_prompt or RELATED_CHUNK_SYSTEM_PROMPT,
            user_template=linker_cfg.user_template or RELATED_CHUNK_USER_TEMPLATE,
            top_k_bm25=linker_cfg.top_k_bm25,
            top_related_chunks=linker_cfg.top_related_chunks,
            context_preview_chars=linker_cfg.context_preview_chars,
        )

    structural_cfg = cfg.linker.structural
    return StructuralChunkLinker(
        source=source,
        type_distribution=structural_cfg.type_distribution,
        target_hop_counts=structural_cfg.target_hop_counts,
        bm25_enrichment_queries=structural_cfg.bm25_enrichment_queries,
        bm25_enrichment_top_k=structural_cfg.bm25_enrichment_top_k,
        max_related_refs=structural_cfg.max_related_refs,
        entity_extraction=cfg.corpus_context.entity_extraction,
    )


def _build_generator(
    cfg: CgftPipelineConfig,
    *,
    linker: ChunkLinker,
    rollout_client_factory: Callable[[CgftPipelineConfig], RolloutClient] | None = None,
) -> QuestionGenerator:
    if cfg.generation.mode == "llm_env":
        rollout_client = (
            rollout_client_factory(cfg) if rollout_client_factory else _build_rollout_client(cfg)
        )
        return EnvRolloutGenerator(
            rollout_client=rollout_client,
            linker=linker,
            cfg=cfg.generation.llm_env,
        )

    generation_cfg = cfg.generation.llm_direct
    generation_client = _build_openai_client(
        api_key=generation_cfg.api_key,
        base_url=generation_cfg.base_url,
    )
    return DirectLLMGenerator(
        client=generation_client,
        linker=linker,
        cfg=generation_cfg,
    )


def _build_transformer(cfg: CgftPipelineConfig) -> QuestionTransformer:
    return LLMStyleTransformer(cfg.transformation)


def _build_filter_from_stage_name(
    stage_name: str,
    cfg: CgftPipelineConfig,
    *,
    source: Any,
) -> EvaluatorFilter:
    stage = str(stage_name or "").strip().lower()
    if stage == _GROUNDING_FILTER_STAGE:
        return GroundingLLMFilter(
            chunk_source=source,
            cfg=cfg.filtering.grounding_llm,
        )
    if stage == _RETRIEVAL_TOO_EASY_FILTER_STAGE:
        return RetrievalLLMFilter(
            chunk_source=source,
            cfg=cfg.filtering.retrieval_llm,
        )
    raise ValueError(
        f"Unknown filter stage '{stage_name}'. "
        f"Supported filter stage names: {', '.join(_SUPPORTED_FILTER_STAGES)}."
    )


def _build_filter_chain(
    cfg: CgftPipelineConfig,
    *,
    source: Any,
) -> tuple[list[str], list[EvaluatorFilter]]:
    chain_names = [
        str(name).strip().lower()
        for name in (cfg.filtering.filters or [])
        if str(name).strip()
    ]
    filters = [
        _build_filter_from_stage_name(
            stage_name,
            cfg,
            source=source,
        )
        for stage_name in chain_names
    ]
    return chain_names, filters


def _mark_rejected(items: list[GeneratedQA], *, reason: str, reason_code: str) -> list[GeneratedQA]:
    for item in items:
        item.filter_verdict = FilterVerdict(
            status="rejected",
            reason=reason,
            reasoning=reason,
            metadata={
                "filter_mode": "pipeline",
                "reason_code": reason_code,
                "confidence": 1.0,
                "retrieval_query": str(item.qa.get("question", "")).strip(),
                "ref_overlap_ratio": None,
                "feedback_type": None,
                "refinement_hint": None,
            },
        )
    return items


def _serialize_qa_with_filter_details(item: GeneratedQA) -> dict[str, Any]:
    row = dict(item.qa)
    verdict = item.filter_verdict
    if verdict is None:
        return row
    row["filter_status"] = verdict.status
    row["filter_reason"] = verdict.reason
    row["filter_reasoning"] = verdict.reasoning
    row["filter_metadata"] = dict(verdict.metadata) if isinstance(verdict.metadata, dict) else {}
    return row


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _short_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _normalize_failure_type_for_regeneration(
    *,
    metadata: dict[str, Any],
    reason_code: str = "",
    verdict_reason: str = "",
    verdict_reasoning: str = "",
) -> str:
    raw = str(metadata.get("failure_type", "")).strip().lower()
    if raw in {"too_easy", "unsupported"}:
        return raw

    text = " ".join(
        part.lower()
        for part in (reason_code, verdict_reason, verdict_reasoning)
        if str(part).strip()
    )
    if "unsupported" in text or "unanswerable" in text:
        return "unsupported"
    if "too_easy" in text or "naive_retrieval_sufficient" in text:
        return "too_easy"
    return "unknown"


def _expected_action_for_failure_type(failure_type: str) -> str:
    if failure_type == "too_easy":
        return "increase_difficulty_keep_answer_locked"
    if failure_type == "unsupported":
        return "reanchor_and_revise_answer_using_new_evidence"
    return "address_feedback_and_remain_grounded"


def _select_regeneration_seed(
    *,
    current_seed_chunk_id: str,
    seed_ids: list[str],
    item_index: int,
    current_refinement_count: int,
    same_seed_failures: int,
    max_same_seed_attempts_before_reanchor: int,
    force_reanchor: bool = False,
) -> tuple[str, bool]:
    if force_reanchor:
        candidates = [seed_id for seed_id in seed_ids if seed_id and seed_id != current_seed_chunk_id]
        if not candidates:
            return current_seed_chunk_id, False
        next_seed = candidates[(item_index + current_refinement_count) % len(candidates)]
        return next_seed, True

    if max_same_seed_attempts_before_reanchor <= 0:
        return current_seed_chunk_id, False
    if same_seed_failures < max_same_seed_attempts_before_reanchor:
        return current_seed_chunk_id, False

    candidates = [seed_id for seed_id in seed_ids if seed_id and seed_id != current_seed_chunk_id]
    if not candidates:
        return current_seed_chunk_id, False

    next_seed = candidates[(item_index + current_refinement_count) % len(candidates)]
    return next_seed, True


def _build_regeneration_prompt(
    item: GeneratedQA,
    *,
    regeneration_attempt: int,
    source_task_id: str,
    old_seed_chunk_id: str,
    new_seed_chunk_id: str,
    switched_seed: bool,
) -> str:
    verdict = item.filter_verdict
    metadata = dict(verdict.metadata) if verdict is not None else {}
    reason_code = str(metadata.get("reason_code", "")).strip() or (verdict.reason if verdict else "unknown")
    reasoning = str(verdict.reasoning if verdict is not None else "").strip()
    hint = str(metadata.get("refinement_hint", "")).strip()
    judge_reasoning = str(metadata.get("judge_reasoning", "")).strip()
    failure_type = _normalize_failure_type_for_regeneration(
        metadata=metadata,
        reason_code=reason_code,
        verdict_reason=verdict.reason if verdict is not None else "",
        verdict_reasoning=reasoning,
    )
    judge_reason_tag = str(metadata.get("judge_reason_tag", "")).strip() or "unknown"
    overlap_triggered = bool(metadata.get("overlap_triggered", False))
    question = _short_text(item.qa.get("question", ""), max_chars=420)
    answer = _short_text(item.qa.get("answer", ""), max_chars=520)
    expected_action = _expected_action_for_failure_type(failure_type)

    lines = [
        "Regeneration instructions:",
        f"- source_task_id: {source_task_id}",
        f"- attempt: {regeneration_attempt}",
        f"- previous_filter_reason_code: {reason_code or 'unknown'}",
        f"- previous_failure_type: {failure_type}",
        f"- previous_judge_reason_tag: {judge_reason_tag}",
        f"- overlap_triggered: {str(overlap_triggered).lower()}",
        f"- expected_action: {expected_action}",
    ]
    if reasoning:
        lines.append(f"- previous_filter_reasoning: {reasoning}")
    if hint:
        lines.append(f"- previous_refinement_hint: {hint}")
    if judge_reasoning:
        lines.append(f"- previous_judge_reasoning: {judge_reasoning}")
    if switched_seed:
        lines.append(
            "- reanchor_action: switched seed chunk after repeated failures "
            f"({old_seed_chunk_id} -> {new_seed_chunk_id})"
        )
    else:
        lines.append(f"- reanchor_action: keep same seed chunk ({new_seed_chunk_id})")
    lines.extend(
        [
            "Previous QA to improve:",
            f"- question: {question}",
            f"- answer: {answer}",
            (
                "Rewrite the QA to address the failure reason while staying grounded in the provided "
                "primary/secondary chunks for this attempt."
            ),
        ]
    )
    return "\n".join(lines)


def _create_generation_task(task_kwargs: dict[str, Any]) -> GenerationTask:
    """Instantiate GenerationTask while tolerating older class signatures."""
    try:
        return GenerationTask(**task_kwargs)
    except TypeError:
        try:
            supported_keys = set(inspect.signature(GenerationTask).parameters)
        except (TypeError, ValueError):
            supported_keys = set()
        if not supported_keys:
            raise

        compatible_kwargs = {key: value for key, value in task_kwargs.items() if key in supported_keys}
        if len(compatible_kwargs) == len(task_kwargs):
            raise

        task = GenerationTask(**compatible_kwargs)

        # Backfill dropped fields for downstream template rendering when possible.
        for key, value in task_kwargs.items():
            if key in compatible_kwargs or hasattr(task, key):
                continue
            try:
                setattr(task, key, value)
            except Exception:
                continue
        return task


def _build_regeneration_tasks(
    items: list[GeneratedQA],
    *,
    context: CgftContext,
) -> tuple[list[GenerationTask], dict[str, GeneratedQA]]:
    seed_lookup = context.get("seed_chunk_lookup", {}) or {}
    fallback_seed_ids = [str(seed_id) for seed_id in seed_lookup if str(seed_id)]

    tasks: list[GenerationTask] = []
    item_by_task_id: dict[str, GeneratedQA] = {}
    seen_task_ids: set[str] = set()

    max_same_seed_attempts_before_reanchor = max(
        0,
        _int_or(
            context.config.refinement.max_same_seed_attempts_before_reanchor,
            0,
        ),
    )

    for idx, item in enumerate(items):
        meta = dict(item.generation_metadata)
        verdict_meta = dict(item.filter_verdict.metadata) if item.filter_verdict is not None else {}
        verdict_reason = str(item.filter_verdict.reason if item.filter_verdict is not None else "").strip()
        verdict_reasoning = str(
            item.filter_verdict.reasoning if item.filter_verdict is not None else ""
        ).strip()
        reason_code = str(verdict_meta.get("reason_code", "")).strip() or verdict_reason
        failure_type = _normalize_failure_type_for_regeneration(
            metadata=verdict_meta,
            reason_code=reason_code,
            verdict_reason=verdict_reason,
            verdict_reasoning=verdict_reasoning,
        )
        judge_reason_tag = str(verdict_meta.get("judge_reason_tag", "")).strip() or "unknown"
        overlap_triggered = bool(verdict_meta.get("overlap_triggered", False))
        expected_action = _expected_action_for_failure_type(failure_type)
        current_count = max(0, _int_or(meta.get("refinement_count", 0), 0))
        same_seed_failures = max(0, _int_or(meta.get("same_seed_refinement_count", 0), 0))
        force_reanchor = bool(verdict_meta.get("force_reanchor", False))

        qa_type = str(meta.get("qa_type_target", "")).strip() or str(item.qa.get("qa_type", "")).strip()
        if not qa_type:
            qa_type = "lookup"

        target_hop_count = max(
            1,
            _int_or(meta.get("target_hop_count", item.qa.get("min_hop_count", 1)), 1),
        )
        if failure_type == "too_easy":
            # Too-easy retries should increase composition depth by default.
            target_hop_count = min(max(target_hop_count + 1, 2), 6)

        seed_chunk_id = str(meta.get("seed_chunk_id", "")).strip()
        if not seed_chunk_id:
            reference_chunks = list(item.qa.get("reference_chunks", []) or [])
            if reference_chunks:
                seed_chunk_id = str((reference_chunks[0] or {}).get("id", "")).strip()
        if not seed_chunk_id and fallback_seed_ids:
            seed_chunk_id = fallback_seed_ids[idx % len(fallback_seed_ids)]
        old_seed_chunk_id = seed_chunk_id
        seed_chunk_id, switched_seed = _select_regeneration_seed(
            current_seed_chunk_id=seed_chunk_id,
            seed_ids=fallback_seed_ids,
            item_index=idx,
            current_refinement_count=current_count,
            same_seed_failures=same_seed_failures,
            max_same_seed_attempts_before_reanchor=max_same_seed_attempts_before_reanchor,
            force_reanchor=force_reanchor,
        )

        source_task_id = str(meta.get("task_id", "")).strip() or f"task_refine_{idx:05d}"
        regeneration_attempt = current_count + 1
        task_id = f"{source_task_id}__regen_{current_count + 1:02d}"
        suffix = 1
        while task_id in seen_task_ids:
            suffix += 1
            task_id = f"{source_task_id}__regen_{current_count + 1:02d}_{suffix}"
        seen_task_ids.add(task_id)

        task_kwargs: dict[str, Any] = {
            "task_id": task_id,
            "qa_type": qa_type,
            "target_hop_count": target_hop_count,
            "seed_chunk_id": seed_chunk_id,
            "regeneration_prompt": _build_regeneration_prompt(
                item,
                regeneration_attempt=regeneration_attempt,
                source_task_id=source_task_id,
                old_seed_chunk_id=old_seed_chunk_id,
                new_seed_chunk_id=seed_chunk_id,
                switched_seed=switched_seed,
            ),
            "regeneration_attempt": regeneration_attempt,
            "source_task_id": source_task_id,
            "previous_failure_type": failure_type,
            "previous_judge_reason_tag": judge_reason_tag,
            "overlap_triggered": overlap_triggered,
            "expected_action": expected_action,
            "failed_question": str(item.qa.get("question", "")).strip(),
            "failed_answer": str(item.qa.get("answer", "")).strip(),
        }
        task = _create_generation_task(task_kwargs)
        tasks.append(task)
        item_by_task_id[task.task_id] = item

    return tasks, item_by_task_id


def _regenerate_with_generator(
    items: list[GeneratedQA],
    *,
    generator: QuestionGenerator,
    context: CgftContext,
) -> tuple[list[GeneratedQA], list[GeneratedQA]]:
    tasks, originals_by_task_id = _build_regeneration_tasks(items, context=context)
    if not tasks:
        return [], []

    regenerated_items = generator.generate(tasks, context)
    regenerated_by_task_id: dict[str, GeneratedQA] = {}
    for regenerated in regenerated_items:
        task_id = str(regenerated.generation_metadata.get("task_id", "")).strip()
        if not task_id:
            logger.warning("Dropping regenerated item with missing task_id metadata.")
            continue
        if task_id not in originals_by_task_id:
            logger.warning("Dropping regenerated item for unknown task_id '%s'.", task_id)
            continue
        if task_id in regenerated_by_task_id:
            logger.warning("Dropping duplicate regenerated item for task_id '%s'.", task_id)
            continue
        regenerated_by_task_id[task_id] = regenerated

    regenerated_for_retry: list[GeneratedQA] = []
    failed_to_regenerate: list[GeneratedQA] = []
    for task in tasks:
        source_item = originals_by_task_id[task.task_id]
        regenerated = regenerated_by_task_id.get(task.task_id)
        if regenerated is None:
            failed_to_regenerate.append(source_item)
            continue

        source_meta = dict(source_item.generation_metadata)
        current_count = max(0, _int_or(source_meta.get("refinement_count", 0), 0))
        next_count = current_count + 1
        previous_seed_chunk_id = str(source_meta.get("seed_chunk_id", "")).strip() or task.seed_chunk_id
        same_seed_failures = max(0, _int_or(source_meta.get("same_seed_refinement_count", 0), 0))
        next_same_seed_failures = same_seed_failures + 1
        if task.seed_chunk_id != previous_seed_chunk_id:
            next_same_seed_failures = 1

        merged_meta = {**source_meta, **dict(regenerated.generation_metadata)}
        merged_meta["task_id"] = task.task_id
        merged_meta["seed_chunk_id"] = task.seed_chunk_id
        merged_meta["refinement_count"] = next_count
        merged_meta["same_seed_refinement_count"] = next_same_seed_failures
        merged_meta["source_task_id"] = task.source_task_id or str(source_meta.get("task_id", "")).strip()
        merged_meta["regeneration_attempt"] = task.regeneration_attempt

        regenerated.generation_metadata = merged_meta
        regenerated.filter_verdict = None
        regenerated.regeneration_history = source_item.regeneration_history + [
            {
                "type": "generator_retry",
                "round": next_count,
                "source_task_id": str(source_meta.get("task_id", "")).strip() or task.task_id,
                "seed_chunk_id": task.seed_chunk_id,
                "reanchored": task.seed_chunk_id != previous_seed_chunk_id,
            }
        ]
        regenerated_for_retry.append(regenerated)

    return regenerated_for_retry, failed_to_regenerate


def _generate_entity_extraction_patterns(
    cfg: CgftPipelineConfig,
    source: Any,
    context: CgftContext,
) -> EntityExtractionConfig | None:
    """Generate entity extraction patterns from corpus samples via LLM."""
    profile_cfg = cfg.corpus_context
    samples: list[Any] = []

    corpus_pool: list[Any] = context.get("corpus_pool", []) or []
    if corpus_pool:
        rng = context.rng
        n = min(8, len(corpus_pool))
        samples = rng.sample(corpus_pool, n)

    if not samples:
        return None

    chunk_samples_text = "\n\n---\n\n".join(
        chunk.chunk_str() if hasattr(chunk, "chunk_str") else str(chunk)
        for chunk in samples
    )
    user_prompt = _ENTITY_EXTRACTION_USER_TEMPLATE.format(chunk_samples=chunk_samples_text)

    try:
        entity_llm_cfg = profile_cfg.entity_extraction_llm
        client = _build_openai_client(
            api_key=entity_llm_cfg.api_key,
            base_url=entity_llm_cfg.base_url,
        )
        completion = _chat_completion_with_retry(
            client=client,
            model=entity_llm_cfg.model,
            messages=[
                {"role": "system", "content": _ENTITY_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = completion.choices[0].message.content or ""
        data = json.loads(raw)

        valid_patterns: dict[str, str] = {}
        for name, pattern in (data.get("code_patterns") or {}).items():
            try:
                re.compile(str(pattern))
                valid_patterns[str(name)] = str(pattern)
            except re.error:
                logger.warning("Skipping invalid entity extraction regex '%s': %s", name, pattern)

        return EntityExtractionConfig(
            entity_names=[
                str(e).strip() for e in (data.get("entity_names") or []) if str(e).strip()
            ],
            code_patterns=valid_patterns,
            domain_terms=[
                str(t).strip() for t in (data.get("domain_terms") or []) if str(t).strip()
            ],
            query_templates=[
                str(q).strip() for q in (data.get("query_templates") or []) if str(q).strip()
            ],
            confidence=str(data.get("confidence", "low")).strip().lower(),
        )
    except Exception as exc:
        logger.warning(
            "Entity extraction pattern generation failed; BM25 will use metadata fallback. reason=%s",
            exc,
        )
        return None


def _build_corpus_profile(cfg: CgftPipelineConfig, source: Any, context: CgftContext) -> None:
    """Generate corpus summary/example queries from description and samples."""
    profile_cfg = cfg.corpus_context
    default_summary = profile_cfg.description
    default_queries = list(profile_cfg.example_queries)
    default_queries_bulleted = "\n".join(f"- {query}" for query in default_queries)

    # Always set context keys so generators can safely reference these placeholders.
    context["corpus_summary"] = default_summary
    context["corpus_queries"] = default_queries_bulleted
    context["corpus_example_queries"] = default_queries
    context["corpus_description"] = profile_cfg.description

    has_user_context = bool(profile_cfg.description.strip() or default_queries)
    if not profile_cfg.enabled or not has_user_context:
        context["corpus_profile"] = {
            "enabled": profile_cfg.enabled,
            "used": False,
            "summary": default_summary,
            "example_queries": default_queries,
            "raw_response": "",
        }
        return

    top_level_chunks = []
    if hasattr(source, "get_top_level_chunks"):
        try:
            top_level_chunks = list(source.get_top_level_chunks() or [])
        except Exception:
            logger.exception("Failed to fetch top-level chunks for corpus profiling.")
            top_level_chunks = []

    sampled_top_level = []
    if top_level_chunks and profile_cfg.num_top_level_samples > 0:
        sample_count = min(profile_cfg.num_top_level_samples, len(top_level_chunks))
        sampled_top_level = context.rng.sample(top_level_chunks, sample_count)

    sampled_random = []
    if profile_cfg.num_random_samples > 0:
        try:
            sampled_random = source.sample_chunks(
                profile_cfg.num_random_samples,
                min_chars=profile_cfg.min_chunk_chars,
            )
        except Exception:
            logger.exception("Failed to fetch random chunks for corpus profiling.")
            sampled_random = []

    variables = {
        "user_context": (
            f"Description: {profile_cfg.description}\n"
            f"Example queries provided by user: {', '.join(default_queries)}"
        ),
        "top_level_content": "\n\n".join(
            (
                chunk.chunk_str()
                if hasattr(chunk, "chunk_str")
                else str(chunk)
            )
            for chunk in sampled_top_level
        ),
        "random_content": "\n\n".join(
            (
                chunk.chunk_str()
                if hasattr(chunk, "chunk_str")
                else str(chunk)
            )
            for chunk in sampled_random
        ),
    }

    try:
        user_prompt = render_template(profile_cfg.user_template, variables)
        client = _build_openai_client(
            api_key=profile_cfg.api_key,
            base_url=profile_cfg.base_url,
        )
        completion = _chat_completion_with_retry(
            client=client,
            model=profile_cfg.model,
            messages=[
                {"role": "system", "content": profile_cfg.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = completion.choices[0].message.content or ""
        summary, queries = parse_corpus_summary_response(raw)
    except Exception as exc:
        logger.warning(
            "Corpus profiling failed; using user-provided description/example queries. reason=%s",
            exc,
        )
        summary, queries, raw, user_prompt = "", [], "", ""

    summary = summary.strip() or default_summary
    queries = [str(query).strip() for query in queries if str(query).strip()] or default_queries

    context["corpus_summary"] = summary
    context["corpus_queries"] = "\n".join(f"- {query}" for query in queries)
    context["corpus_example_queries"] = queries
    context["corpus_profile"] = {
        "enabled": profile_cfg.enabled,
        "used": bool(raw),
        "summary": summary,
        "example_queries": queries,
        "user_prompt": user_prompt,
        "raw_response": raw,
    }


def _print_progress(message: str, *, verbose: bool) -> None:
    if verbose:
        _tqdm.write(message)


class CgftPipeline:
    """Orchestrates Cgft generation, filtering, refinement, and formatting."""

    def __init__(
        self,
        cfg: CgftPipelineConfig,
        *,
        source_factory: Callable[[CgftPipelineConfig], Any] | None = None,
        rollout_client_factory: Callable[[CgftPipelineConfig], RolloutClient] | None = None,
    ) -> None:
        self.cfg = cfg
        self.source_factory = source_factory or _load_source
        self.rollout_client_factory = rollout_client_factory

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        cfg.resolve_api_keys()

        source = self.source_factory(cfg)
        rng = random.Random(cfg.random_seed)
        context = CgftContext(config=cfg, source=source, rng=rng)

        _print_progress("[1/8] Loading chunks from corpus...", verbose=cfg.verbose)
        seed_chunks = source.sample_chunks(cfg.targets.total_samples, min_chars=cfg.corpus.min_chunk_chars)
        if not seed_chunks and getattr(source, "collection", None) is not None:
            seed_chunks = list(source.collection)[: cfg.targets.total_samples]
        if not seed_chunks:
            raise RuntimeError("No eligible chunks were found for CgftPipeline generation.")
        _print_progress(
            f"[1/8] Loaded {len(seed_chunks)} seed chunks from corpus", verbose=cfg.verbose
        )

        pool_size = max(cfg.linker.structural.corpus_pool_size, len(seed_chunks))
        corpus_pool = source.sample_chunks(pool_size, min_chars=cfg.corpus.min_chunk_chars)
        if not corpus_pool:
            corpus_pool = list(seed_chunks)

        context["seed_chunk_lookup"] = {chunk.hash: chunk for chunk in seed_chunks}
        context["corpus_pool"] = corpus_pool
        context["seed_chunks"] = seed_chunks
        _print_progress("[2/8] Building corpus profile...", verbose=cfg.verbose)
        _build_corpus_profile(cfg, source, context)
        _print_progress("[2/8] Built corpus profile", verbose=cfg.verbose)

        if cfg.corpus_context.generate_entity_patterns:
            _print_progress("[3/8] Generating entity patterns...", verbose=cfg.verbose)
            extraction = _generate_entity_extraction_patterns(cfg, source, context)
            if extraction is not None:
                cfg.corpus_context.entity_extraction = extraction
                logger.info(
                    "Entity extraction patterns generated: %d entities, %d code patterns, "
                    "%d domain terms (confidence=%s)",
                    len(extraction.entity_names),
                    len(extraction.code_patterns),
                    len(extraction.domain_terms),
                    extraction.confidence,
                )
                n_entities = len(extraction.entity_names)
                n_patterns = len(extraction.code_patterns) + len(extraction.domain_terms)
                _print_progress(
                    f"[3/8] Generated entity patterns: {n_entities} entities,"
                    f" {n_patterns} patterns",
                    verbose=cfg.verbose,
                )
        else:
            _print_progress("[3/8] Skipped entity pattern generation", verbose=cfg.verbose)

        tasks = build_generation_tasks(
            cfg,
            seed_chunk_ids=[chunk.hash for chunk in seed_chunks],
        )
        context["tasks"] = tasks
        _print_progress(f"[4/8] Created {len(tasks)} generation tasks", verbose=cfg.verbose)

        linker = _build_linker(cfg, source)
        generator = _build_generator(
            cfg,
            linker=linker,
            rollout_client_factory=self.rollout_client_factory,
        )
        guard_filter = DeterministicGuardsFilter(cfg.filtering.deterministic_guards)
        filter_stage_names, filter_chain = _build_filter_chain(
            cfg,
            source=source,
        )
        context["filter_chain"] = list(filter_stage_names)
        transformer = _build_transformer(cfg)
        formatter = TrainEvalFormatter(output_cfg=cfg.output, split_cfg=cfg.split)

        _print_progress(f"[5/8] Generating {len(tasks)} QA candidates...", verbose=cfg.verbose)
        raw_items = generator.generate(tasks, context)
        context["raw_candidates"] = [item.qa for item in raw_items]
        context["raw_count"] = len(raw_items)
        _failed_to_parse = len(tasks) - len(raw_items)
        _parse_note = f" ({_failed_to_parse} failed to parse)" if _failed_to_parse else ""
        _print_progress(
            f"[5/8] Generated {len(raw_items)} QA candidates{_parse_note}", verbose=cfg.verbose
        )

        final_passed: list[GeneratedQA] = []
        final_rejected: list[GeneratedQA] = []
        active_items = list(raw_items)
        total_regens = 0

        _print_progress(
            f"[6/8] Starting filtering (max {cfg.refinement.max_rounds} rounds)...",
            verbose=cfg.verbose,
        )
        for round_idx in range(cfg.refinement.max_rounds + 1):
            if not active_items:
                break
            for item in active_items:
                item.filter_verdict = None

            _print_progress(
                f"  Round {round_idx + 1}/{cfg.refinement.max_rounds + 1}:"
                f" {len(active_items)} items to evaluate",
                verbose=cfg.verbose,
            )

            active_items = guard_filter.evaluate(active_items, context)
            _passed = sum(1 for i in active_items if i.is_passed)
            _refine = sum(1 for i in active_items if i.needs_refinement)
            _rejected = sum(1 for i in active_items if i.is_rejected)
            _print_progress(
                f"    deterministic_guards: {_passed} passed, {_refine} need refinement,"
                f" {_rejected} rejected",
                verbose=cfg.verbose,
            )

            for stage_name, stage_filter in zip(filter_stage_names, filter_chain):
                _n_to_eval = sum(1 for i in active_items if not i.is_passed and not i.is_rejected)
                _print_progress(
                    f"    {stage_name}: evaluating {_n_to_eval} items...", verbose=cfg.verbose
                )
                active_items = stage_filter.evaluate(active_items, context)
                _passed = sum(1 for i in active_items if i.is_passed)
                _refine = sum(1 for i in active_items if i.needs_refinement)
                _rejected = sum(1 for i in active_items if i.is_rejected)
                _print_progress(
                    f"    {stage_name}: {_passed} passed, {_refine} need refinement,"
                    f" {_rejected} rejected",
                    verbose=cfg.verbose,
                )

            passed = [item for item in active_items if item.is_passed]
            needs_refinement = [item for item in active_items if item.needs_refinement]
            rejected = [item for item in active_items if item.is_rejected]
            final_passed.extend(passed)
            final_rejected.extend(rejected)

            if not needs_refinement:
                active_items = []
                break

            if not cfg.refinement.enabled or round_idx >= cfg.refinement.max_rounds:
                final_rejected.extend(
                    _mark_rejected(
                        needs_refinement,
                        reason="max_rounds_reached",
                        reason_code="max_rounds_reached",
                    )
                )
                active_items = []
                break

            remaining_budget = cfg.refinement.max_total_regenerations - total_regens
            if remaining_budget <= 0:
                final_rejected.extend(
                    _mark_rejected(
                        needs_refinement,
                        reason="global_regeneration_budget_exhausted",
                        reason_code="global_regeneration_budget_exhausted",
                    )
                )
                active_items = []
                break

            to_refine = needs_refinement[:remaining_budget]
            overflow = needs_refinement[remaining_budget:]
            if overflow:
                final_rejected.extend(
                    _mark_rejected(
                        overflow,
                        reason="global_regeneration_budget_exhausted",
                        reason_code="global_regeneration_budget_exhausted",
                    )
                )

            _print_progress(
                f"    Regenerating {len(to_refine)} items...", verbose=cfg.verbose
            )
            refined_items, regen_failures = _regenerate_with_generator(
                to_refine,
                generator=generator,
                context=context,
            )
            if regen_failures:
                final_rejected.extend(
                    _mark_rejected(
                        regen_failures,
                        reason="generator_regeneration_failed",
                        reason_code="generator_regeneration_failed",
                    )
                )
            total_regens += len(to_refine)
            _regen_fail_note = f" ({len(regen_failures)} failed)" if regen_failures else ""
            _print_progress(
                f"    Regenerated {len(to_refine)} items{_regen_fail_note}", verbose=cfg.verbose
            )
            next_active: list[GeneratedQA] = []
            for item in refined_items:
                if item.filter_verdict is None:
                    next_active.append(item)
                elif item.is_rejected:
                    final_rejected.append(item)
                elif item.is_passed:
                    final_passed.append(item)
                else:
                    item.filter_verdict = None
                    next_active.append(item)
            active_items = next_active

        if active_items:
            final_rejected.extend(
                _mark_rejected(
                    active_items,
                    reason="pipeline_terminated_with_unresolved_items",
                    reason_code="unresolved_items",
                )
            )

        final_passed = transformer.transform(final_passed, context)
        _print_progress(
            f"[7/8] Transformed {len(final_passed)} passed items", verbose=cfg.verbose
        )
        context["rejected_items"] = [_serialize_qa_with_filter_details(item) for item in final_rejected]
        context["passed_items"] = [item.qa for item in final_passed]
        context["total_regenerations"] = total_regens

        result = formatter.format(final_passed, context)
        _train = result.get("stats", {}).get("train", 0)
        _eval = result.get("stats", {}).get("eval", 0)
        _print_progress(
            f"[8/8] Formatted output: {_train} train, {_eval} eval", verbose=cfg.verbose
        )
        _print_progress(
            f"\nPipeline complete: {len(final_passed)} passed, {len(final_rejected)} rejected"
            f" ({total_regens} regenerations)",
            verbose=cfg.verbose,
        )
        result["raw_candidates"] = [item.qa for item in raw_items]
        result["filtered_dataset"] = [item.qa for item in final_passed]
        result["rejected_dataset"] = [_serialize_qa_with_filter_details(item) for item in final_rejected]
        run_stats = CgftRunStats(
            raw_candidates_total=len(raw_items),
            passed_total=len(final_passed),
            rejected_total=len(final_rejected),
            regenerated_total=total_regens,
            round_limit=cfg.refinement.max_rounds,
        )
        result["stats"] = {
            **result.get("stats", {}),
            **run_stats.__dict__,
            "filter_chain": list(filter_stage_names),
        }
        guard_stats = context.get("deterministic_guard_stats")
        if isinstance(guard_stats, dict) and guard_stats:
            result["stats"]["deterministic_guards"] = dict(guard_stats)

        retrieval_too_easy_stats = context.get("retrieval_too_easy_filter_stats")
        if isinstance(retrieval_too_easy_stats, dict) and retrieval_too_easy_stats:
            result["stats"]["retrieval_too_easy_filter"] = dict(retrieval_too_easy_stats)

        grounding_stats = context.get("grounding_filter_stats")
        if isinstance(grounding_stats, dict) and grounding_stats:
            result["stats"]["grounding_filter"] = dict(grounding_stats)

        transformation_stats = context.get("transformation_stats")
        if isinstance(transformation_stats, dict) and transformation_stats:
            result["stats"]["transformation"] = dict(transformation_stats)

        return result


def run_cgft_pipeline(
    cfg: CgftPipelineConfig,
    *,
    source_factory: Callable[[CgftPipelineConfig], Any] | None = None,
    rollout_client_factory: Callable[[CgftPipelineConfig], RolloutClient] | None = None,
) -> dict[str, Any]:
    """Run CgftPipeline with a fully constructed config."""
    pipeline = CgftPipeline(
        cfg,
        source_factory=source_factory,
        rollout_client_factory=rollout_client_factory,
    )
    return pipeline.run()


def run_cgft_pipeline_from_config(
    config_path: str | Path,
    *,
    source_factory: Callable[[CgftPipelineConfig], Any] | None = None,
    rollout_client_factory: Callable[[CgftPipelineConfig], RolloutClient] | None = None,
) -> dict[str, Any]:
    """Load config YAML and run CgftPipeline."""
    cfg = load_cgft_config(config_path)
    return run_cgft_pipeline(
        cfg,
        source_factory=source_factory,
        rollout_client_factory=rollout_client_factory,
    )
