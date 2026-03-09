"""Unified Cgft QA generation pipeline."""

from __future__ import annotations

import inspect
import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from synthetic_data_prep.corpus.corpora.source import CorporaChunkSource
from synthetic_data_prep.qa_generation.cgft_models import (
    CgftContext,
    CgftPipelineConfig,
    CgftRunStats,
    GenerationTask,
    build_generation_tasks,
    load_cgft_config,
)
from synthetic_data_prep.qa_generation.filters import (
    DeterministicGuardsFilter,
    EnvRolloutFilter,
    GroundingLLMFilter,
    RetrievalLLMFilter,
)
from synthetic_data_prep.qa_generation.formatters import TrainEvalFormatter
from synthetic_data_prep.qa_generation.generated_qa import FilterVerdict, GeneratedQA
from synthetic_data_prep.qa_generation.generators import DirectLLMGenerator, EnvRolloutGenerator
from synthetic_data_prep.qa_generation.helpers import render_template
from synthetic_data_prep.qa_generation.linkers import (
    LLMGuidedChunkLinker,
    RELATED_CHUNK_SYSTEM_PROMPT,
    RELATED_CHUNK_USER_TEMPLATE,
    StructuralChunkLinker,
)
from synthetic_data_prep.qa_generation.protocols import ChunkLinker, EvaluatorFilter, QuestionGenerator
from synthetic_data_prep.qa_generation.regenerators import FeedbackRefiner
from synthetic_data_prep.qa_generation.response_parsers import parse_corpus_summary_response
from synthetic_data_prep.trainer.client import RolloutClient

logger = logging.getLogger(__name__)

_FRESH_GENERATION_STRATEGIES = {"fresh_generation", "generator_retry"}
_GENERATOR_REFINEMENT_STRATEGIES = {"regenerate_with_generator"}
_FEEDBACK_REFINEMENT_STRATEGIES = {"same_anchor_feedback", "feedback"}
_RETRIEVAL_FILTER_STAGE = "retrieval_llm"
_RETRIEVAL_TOO_EASY_FILTER_STAGE = "retrieval_too_easy_llm"
_GROUNDING_FILTER_STAGE = "grounding_llm"
_ENV_FILTER_STAGE = "llm_env"
_SUPPORTED_FILTER_STAGES = (
    _RETRIEVAL_TOO_EASY_FILTER_STAGE,
    _GROUNDING_FILTER_STAGE,
    _RETRIEVAL_FILTER_STAGE,
    _ENV_FILTER_STAGE,
)


def _build_openai_client(*, api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


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


def _build_active_filter(
    cfg: CgftPipelineConfig,
    *,
    source: Any,
    rollout_client_factory: Callable[[CgftPipelineConfig], RolloutClient] | None = None,
) -> EvaluatorFilter:
    mode = str(cfg.filtering.mode or "").strip().lower() or _RETRIEVAL_FILTER_STAGE
    if mode == _ENV_FILTER_STAGE:
        rollout_client = (
            rollout_client_factory(cfg) if rollout_client_factory else _build_rollout_client(cfg)
        )
        return EnvRolloutFilter(
            rollout_client=rollout_client,
            cfg=cfg.filtering.llm_env,
        )

    return RetrievalLLMFilter(
        chunk_source=source,
        cfg=cfg.filtering.retrieval_llm,
    )


def _build_filter_from_stage_name(
    stage_name: str,
    cfg: CgftPipelineConfig,
    *,
    source: Any,
    rollout_client_factory: Callable[[CgftPipelineConfig], RolloutClient] | None = None,
) -> EvaluatorFilter:
    stage = str(stage_name or "").strip().lower()
    if stage == _RETRIEVAL_TOO_EASY_FILTER_STAGE:
        stage_cfg = replace(
            cfg.filtering.retrieval_llm,
            route_unsupported_to_failure=False,
            stats_key="retrieval_too_easy_filter_stats",
        )
        return RetrievalLLMFilter(
            chunk_source=source,
            cfg=stage_cfg,
        )
    if stage == _GROUNDING_FILTER_STAGE:
        return GroundingLLMFilter(
            chunk_source=source,
            cfg=cfg.filtering.grounding_llm,
        )
    if stage == _RETRIEVAL_FILTER_STAGE:
        return RetrievalLLMFilter(
            chunk_source=source,
            cfg=cfg.filtering.retrieval_llm,
        )
    if stage == _ENV_FILTER_STAGE:
        rollout_client = (
            rollout_client_factory(cfg) if rollout_client_factory else _build_rollout_client(cfg)
        )
        return EnvRolloutFilter(
            rollout_client=rollout_client,
            cfg=cfg.filtering.llm_env,
        )
    raise ValueError(
        f"Unknown filter stage '{stage_name}'. "
        f"Supported filter stage names: {', '.join(_SUPPORTED_FILTER_STAGES)}."
    )


def _build_filter_chain(
    cfg: CgftPipelineConfig,
    *,
    source: Any,
    rollout_client_factory: Callable[[CgftPipelineConfig], RolloutClient] | None = None,
) -> tuple[list[str], list[EvaluatorFilter]]:
    chain_names = [
        str(name).strip().lower()
        for name in (cfg.filtering.filters or [])
        if str(name).strip()
    ]
    if not chain_names:
        mode = str(cfg.filtering.mode or "").strip().lower() or _RETRIEVAL_FILTER_STAGE
        return [mode], [
            _build_active_filter(
                cfg,
                source=source,
                rollout_client_factory=rollout_client_factory,
            )
        ]
    filters = [
        _build_filter_from_stage_name(
            stage_name,
            cfg,
            source=source,
            rollout_client_factory=rollout_client_factory,
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


def _resolve_refinement_strategy(strategy: str) -> str:
    normalized = str(strategy or "").strip().lower()
    if not normalized:
        return "fresh_generation"
    if normalized in _FRESH_GENERATION_STRATEGIES:
        return "fresh_generation"
    if normalized in _GENERATOR_REFINEMENT_STRATEGIES:
        return "regenerate_with_generator"
    if normalized in _FEEDBACK_REFINEMENT_STRATEGIES:
        return "same_anchor_feedback"
    logger.warning(
        "Unknown refinement strategy '%s'; defaulting to fresh_generation.",
        strategy,
    )
    return "fresh_generation"


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

        style_target = str(meta.get("style_target", "")).strip() or str(
            (item.qa.get("eval_scores", {}) or {}).get("query_style_target", "")
        ).strip()
        if not style_target:
            style_target = "natural"

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
            "style_target": style_target,
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


def _regenerate_fresh(
    items: list[GeneratedQA],
    *,
    generator: QuestionGenerator,
    context: CgftContext,
) -> tuple[list[GeneratedQA], list[GeneratedQA]]:
    """Discard failed items and generate a fresh batch with new random seeds.

    Mirrors SAGE's ``GenerationRetryRegenerator`` approach: no feedback prompt,
    no failed Q/A — just fresh tasks with different seed assignments.
    """
    cfg = context.config
    seed_lookup = context.get("seed_chunk_lookup", {}) or {}
    seed_chunk_ids = [str(sid) for sid in seed_lookup if str(sid)]
    if not seed_chunk_ids:
        return [], list(items)

    # Bump the random seed so new tasks get different chunk assignments.
    fresh_seed = cfg.random_seed + len(items) + int(context.get("total_regenerations", 0) or 0) + 1
    fresh_cfg = CgftPipelineConfig(
        platform=cfg.platform,
        targets=cfg.targets,
        linker=cfg.linker,
        random_seed=fresh_seed,
    )
    fresh_tasks = build_generation_tasks(fresh_cfg, seed_chunk_ids=seed_chunk_ids)
    # Only need as many replacements as there are failed items.
    fresh_tasks = fresh_tasks[: len(items)]
    if not fresh_tasks:
        return [], list(items)

    fresh_items = generator.generate(fresh_tasks, context)
    # Mark them as having come through a fresh-generation retry.
    for item in fresh_items:
        item.filter_verdict = None
        meta = dict(item.generation_metadata)
        prev_count = max(0, _int_or(meta.get("refinement_count", 0), 0))
        meta["refinement_count"] = prev_count + 1
        item.generation_metadata = meta
        item.regeneration_history = item.regeneration_history + [
            {"type": "fresh_generation", "round": prev_count + 1}
        ]

    failed = items[len(fresh_items) :]
    return list(fresh_items), list(failed)


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
        completion = client.chat.completions.create(
            model=profile_cfg.model,
            messages=[
                {"role": "system", "content": profile_cfg.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or ""
        summary, queries = parse_corpus_summary_response(raw)
    except Exception:
        logger.exception("Corpus profiling failed; using user-provided description/example queries.")
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

        seed_chunks = source.sample_chunks(cfg.targets.total_samples, min_chars=cfg.corpus.min_chunk_chars)
        if not seed_chunks and getattr(source, "collection", None) is not None:
            seed_chunks = list(source.collection)[: cfg.targets.total_samples]
        if not seed_chunks:
            raise RuntimeError("No eligible chunks were found for CgftPipeline generation.")

        pool_size = max(cfg.linker.structural.corpus_pool_size, len(seed_chunks))
        corpus_pool = source.sample_chunks(pool_size, min_chars=cfg.corpus.min_chunk_chars)
        if not corpus_pool:
            corpus_pool = list(seed_chunks)

        context["seed_chunk_lookup"] = {chunk.hash: chunk for chunk in seed_chunks}
        context["corpus_pool"] = corpus_pool
        context["seed_chunks"] = seed_chunks
        _build_corpus_profile(cfg, source, context)

        tasks = build_generation_tasks(
            cfg,
            seed_chunk_ids=[chunk.hash for chunk in seed_chunks],
        )
        context["tasks"] = tasks

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
            rollout_client_factory=self.rollout_client_factory,
        )
        context["filter_chain"] = list(filter_stage_names)
        refinement_strategy = _resolve_refinement_strategy(cfg.refinement.strategy)
        refiner: FeedbackRefiner | None = None
        if refinement_strategy == "same_anchor_feedback":
            refiner = FeedbackRefiner(cfg.refinement)
        formatter = TrainEvalFormatter(output_cfg=cfg.output, split_cfg=cfg.split)

        raw_items = generator.generate(tasks, context)
        context["raw_candidates"] = [item.qa for item in raw_items]
        context["raw_count"] = len(raw_items)

        final_passed: list[GeneratedQA] = []
        final_rejected: list[GeneratedQA] = []
        active_items = list(raw_items)
        total_regens = 0

        for round_idx in range(cfg.refinement.max_rounds + 1):
            if not active_items:
                break
            for item in active_items:
                item.filter_verdict = None

            active_items = guard_filter.evaluate(active_items, context)
            for stage_filter in filter_chain:
                active_items = stage_filter.evaluate(active_items, context)

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

            if refinement_strategy == "fresh_generation":
                refined_items, regen_failures = _regenerate_fresh(
                    to_refine,
                    generator=generator,
                    context=context,
                )
                if regen_failures:
                    final_rejected.extend(
                        _mark_rejected(
                            regen_failures,
                            reason="fresh_generation_failed",
                            reason_code="fresh_generation_failed",
                        )
                    )
            elif refinement_strategy == "regenerate_with_generator":
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
            else:
                if refiner is None:
                    raise RuntimeError("Refiner is not configured for feedback refinement strategy.")
                refined_items = refiner.refine(to_refine, context)
            total_regens += len(to_refine)
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

        context["rejected_items"] = [_serialize_qa_with_filter_details(item) for item in final_rejected]
        context["passed_items"] = [item.qa for item in final_passed]
        context["total_regenerations"] = total_regens

        result = formatter.format(final_passed, context)
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

        retrieval_stats = context.get("retrieval_filter_stats")
        if isinstance(retrieval_stats, dict) and retrieval_stats:
            result["stats"]["retrieval_filter"] = dict(retrieval_stats)

        retrieval_too_easy_stats = context.get("retrieval_too_easy_filter_stats")
        if isinstance(retrieval_too_easy_stats, dict) and retrieval_too_easy_stats:
            result["stats"]["retrieval_too_easy_filter"] = dict(retrieval_too_easy_stats)

        grounding_stats = context.get("grounding_filter_stats")
        if isinstance(grounding_stats, dict) and grounding_stats:
            result["stats"]["grounding_filter"] = dict(grounding_stats)

        env_stats = context.get("env_filter_stats")
        if isinstance(env_stats, dict) and env_stats:
            result["stats"]["env_filter"] = dict(env_stats)
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
