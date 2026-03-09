"""Hybrid QA pipeline combining SAGE anchors with notebook generation and filtering."""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

import yaml
from openai import OpenAI
from tqdm.auto import tqdm

from synthetic_data_prep.corpus.corpora.source import CorporaChunkSource
from synthetic_data_prep.envs.query_rewriter import heuristic_query_rewrite
from synthetic_data_prep.qa_generation.helpers import generate_single_hop_batch, render_template
from synthetic_data_prep.qa_generation.models import QADataPoint, ReferenceChunk
from synthetic_data_prep.qa_generation.response_parsers import (
    parse_corpus_summary_response,
    parse_multi_hop_validation_response,
    parse_single_hop_response,
)
from synthetic_data_prep.qa_generation.storage import (
    save_jsonl_rows,
    save_qa_dataset,
    save_qa_dataset_jsonl,
)

from .anchor_utils import (
    extract_anchor_ref_ids as _shared_extract_anchor_ref_ids,
    generate_bm25_queries as _shared_generate_bm25_queries,
    select_anchor_bundle_with_enrichment as _shared_select_anchor_bundle_with_enrichment,
)
from .anchor_selector import AnchorBundle, AnchorSelector
from .corpus_capabilities import CorpusCapabilities
from .hybrid_prompts import (
    CORPUS_SYSTEM_PROMPT,
    CORPUS_USER_TEMPLATE,
    EQUIVALENCE_JUDGE_SYSTEM_PROMPT,
    MULTI_HOP_SYSTEM_TEMPLATE,
    MULTI_HOP_USER_TEMPLATE,
    NO_CONTEXT_FILTER_SYSTEM_PROMPT,
    QUESTION_STYLE_INSTRUCTION_TEMPLATE,
    SINGLE_HOP_SYSTEM_TEMPLATE,
    SINGLE_HOP_USER_TEMPLATE,
)
from .style_controls import (
    DEFAULT_QUERY_STYLE_DISTRIBUTION,
    QUERY_STYLE_EXPERT,
    QUERY_STYLE_KEYS,
    QUERY_STYLE_KEYWORD,
    QUERY_STYLE_NATURAL,
    allocate_largest_remainder as _shared_allocate_largest_remainder,
    classify_query_style as _shared_classify_query_style,
    normalize_style_distribution as _shared_normalize_style_distribution,
    style_sequence_from_counts as _shared_style_sequence_from_counts,
)

MAX_STYLE_TOPUP_PER_ROUND = 30


@dataclass
class PlatformConfig:
    """Top-level platform settings shared by corpus and model clients."""

    api_key: str
    base_url: str = "https://app.cgft.io"


@dataclass
class CorpusConfig:
    """Corpus loading configuration."""

    docs_path: str = ""
    corpus_id: str = ""
    corpus_name: str = "hybrid-corpus"
    show_summary: bool = True


@dataclass
class ModelConfig:
    """LLM endpoint configuration for one role."""

    model: str
    api_key: str = ""
    base_url: str = "https://app.cgft.io/api/llm"


@dataclass
class ContextConfig:
    """Corpus context generation controls."""

    description: str = ""
    example_queries: list[str] = field(default_factory=list)
    num_top_level_samples: int = 4
    num_random_samples: int = 4
    min_chunk_chars: int = 400


@dataclass
class SingleHopConfig:
    """Single-hop generation controls."""

    num_samples: int = 40
    min_chunk_chars: int = 400
    context_preview_chars: int = 200
    max_questions_per_chunk: int = 1
    max_concurrent: int = 10
    max_tokens: int = 1000
    timeout: float = 120.0


@dataclass
class MultiHopAnchorConfig:
    """Anchor selector options used by multi-hop generation."""

    type_distribution: dict[str, float] | None = None
    target_hop_counts: dict[str, int] | None = None
    corpus_pool_size: int = 200


@dataclass
class MultiHopConfig:
    """Multi-hop generation controls."""

    num_samples: int = 20
    min_chunk_chars: int = 400
    context_preview_chars: int = 200
    max_questions_per_pair: int = 1
    top_related_chunks: int = 3
    bm25_enrichment_queries: int = 3
    bm25_enrichment_top_k: int = 5
    max_bm25_related_refs: int = 3
    max_tokens: int = 2000
    timeout: float = 120.0
    anchor: MultiHopAnchorConfig = field(default_factory=MultiHopAnchorConfig)


@dataclass
class FilterConfig:
    """Deterministic and model-based filtering controls."""

    enabled: bool = True
    min_question_chars: int = 12
    min_answer_chars: int = 24
    min_reference_chunks: int = 1


@dataclass
class QuestionStyleMixConfig:
    """Style-mix controls for generated questions."""

    enabled: bool = True
    distribution: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_QUERY_STYLE_DISTRIBUTION))
    tolerance: float = 0.10
    apply_to_single_hop: bool = True
    apply_to_multi_hop: bool = True
    max_topup_rounds: int = 2
    enforce_style_mismatch_guard: bool = False


@dataclass
class QueryRewriteConfig:
    """Configuration for generating retrieval-friendly rewrite fields."""

    enabled: bool = True
    source: str = "heuristic"
    max_terms: int = 16
    max_chars: int = 140
    apply_to_all_rows: bool = True
    write_split_datasets: bool = True


@dataclass
class HybridPromptConfig:
    """Optional prompt overrides for hybrid generation/filtering."""

    corpus_system: str = ""
    corpus_user: str = ""
    single_hop_system: str = ""
    single_hop_user: str = ""
    multi_hop_system: str = ""
    multi_hop_user: str = ""
    no_context_filter_system: str = ""
    equivalence_judge_system: str = ""
    question_style_instruction: str = ""

    def resolve(self) -> None:
        """Resolve prompt overrides; supports ``file:path/to/prompt.txt`` values."""
        for field_name in (
            "corpus_system",
            "corpus_user",
            "single_hop_system",
            "single_hop_user",
            "multi_hop_system",
            "multi_hop_user",
            "no_context_filter_system",
            "equivalence_judge_system",
            "question_style_instruction",
        ):
            value = getattr(self, field_name)
            if value:
                setattr(self, field_name, _load_prompt(value))

    def get_corpus_system(self) -> str:
        return self.corpus_system or CORPUS_SYSTEM_PROMPT

    def get_corpus_user(self) -> str:
        return self.corpus_user or CORPUS_USER_TEMPLATE

    def get_single_hop_system(self) -> str:
        return self.single_hop_system or SINGLE_HOP_SYSTEM_TEMPLATE

    def get_single_hop_user(self) -> str:
        return self.single_hop_user or SINGLE_HOP_USER_TEMPLATE

    def get_multi_hop_system(self) -> str:
        return self.multi_hop_system or MULTI_HOP_SYSTEM_TEMPLATE

    def get_multi_hop_user(self) -> str:
        return self.multi_hop_user or MULTI_HOP_USER_TEMPLATE

    def get_no_context_filter_system(self) -> str:
        return self.no_context_filter_system or NO_CONTEXT_FILTER_SYSTEM_PROMPT

    def get_equivalence_judge_system(self) -> str:
        return self.equivalence_judge_system or EQUIVALENCE_JUDGE_SYSTEM_PROMPT

    def get_question_style_instruction(self) -> str:
        return self.question_style_instruction or QUESTION_STYLE_INSTRUCTION_TEMPLATE


@dataclass
class OutputConfig:
    """Output artifact locations and filenames."""

    dir: str = "outputs/hybrid"
    raw_yaml: str = "raw_candidates.yaml"
    raw_jsonl: str = "raw_candidates.jsonl"
    filtered_yaml: str = "filtered_dataset.yaml"
    filtered_jsonl: str = "filtered_dataset.jsonl"
    filter_stats_json: str = "filter_stats.json"
    rewriter_jsonl: str = "rewriter_dataset.jsonl"
    retriever_jsonl: str = "retriever_dataset.jsonl"
    agent_jsonl: str = "agent_dataset.jsonl"


@dataclass
class HybridPipelineConfig:
    """Full configuration for the hybrid QA generation pipeline."""

    platform: PlatformConfig
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    generator_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(model="gpt-5-mini")
    )
    filter_target_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(model="gpt-5-nano")
    )
    filter_judge_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(model="gpt-5-mini")
    )
    context: ContextConfig = field(default_factory=ContextConfig)
    single_hop: SingleHopConfig = field(default_factory=SingleHopConfig)
    multi_hop: MultiHopConfig = field(default_factory=MultiHopConfig)
    question_style_mix: QuestionStyleMixConfig = field(default_factory=QuestionStyleMixConfig)
    query_rewrite: QueryRewriteConfig = field(default_factory=QueryRewriteConfig)
    prompts: HybridPromptConfig = field(default_factory=HybridPromptConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def resolve_api_keys(self) -> None:
        """Populate missing model API keys from the platform key."""
        if not self.generator_model.api_key:
            self.generator_model.api_key = self.platform.api_key
        if not self.filter_target_model.api_key:
            self.filter_target_model.api_key = self.platform.api_key
        if not self.filter_judge_model.api_key:
            self.filter_judge_model.api_key = self.platform.api_key


def _parse_example_queries(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(v).strip() for v in raw if str(v).strip()]
    if isinstance(raw, str):
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) > 1:
            return [ln.lstrip("- ").strip() for ln in lines if ln.lstrip("- ").strip()]
        return [v.strip() for v in raw.split(",") if v.strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _load_prompt(value: str) -> str:
    if value.startswith("file:"):
        return Path(value[5:].strip()).read_text(encoding="utf-8")
    return value


def _parse_model_config(raw: Any, default_model: str) -> ModelConfig:
    if isinstance(raw, str):
        return ModelConfig(model=raw)
    if not isinstance(raw, dict):
        return ModelConfig(model=default_model)
    return ModelConfig(
        model=str(raw.get("model", default_model)),
        api_key=str(raw.get("api_key", "")),
        base_url=str(raw.get("base_url", "https://app.cgft.io/api/llm")),
    )


def _normalize_style_distribution(raw_distribution: Any) -> dict[str, float]:
    return _shared_normalize_style_distribution(raw_distribution)


def load_hybrid_config(path: str | Path) -> HybridPipelineConfig:
    """Load hybrid pipeline config from YAML."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    platform_raw = raw.get("platform", {}) or {}
    api_key = str(platform_raw.get("api_key", "")).strip()
    if not api_key:
        raise ValueError("Missing required config value: platform.api_key")
    platform = PlatformConfig(
        api_key=api_key,
        base_url=str(platform_raw.get("base_url", "https://app.cgft.io")).strip(),
    )

    corpus_raw = raw.get("corpus", {}) or {}
    corpus = CorpusConfig(
        docs_path=str(corpus_raw.get("docs_path", "")).strip(),
        corpus_id=str(corpus_raw.get("corpus_id", "")).strip(),
        corpus_name=str(corpus_raw.get("corpus_name", "hybrid-corpus")).strip(),
        show_summary=bool(corpus_raw.get("show_summary", True)),
    )

    models_raw = raw.get("models", {}) or {}
    generator_model = _parse_model_config(
        models_raw.get("generator", {"model": "gpt-5-mini"}),
        default_model="gpt-5-mini",
    )
    filter_target_model = _parse_model_config(
        models_raw.get("filter_target", {"model": "gpt-5-nano"}),
        default_model="gpt-5-nano",
    )
    filter_judge_model = _parse_model_config(
        models_raw.get("filter_judge", {"model": "gpt-5-mini"}),
        default_model="gpt-5-mini",
    )

    context_raw = raw.get("context", {}) or {}
    context = ContextConfig(
        description=str(context_raw.get("description", "")).strip(),
        example_queries=_parse_example_queries(context_raw.get("example_queries", [])),
        num_top_level_samples=int(context_raw.get("num_top_level_samples", 4)),
        num_random_samples=int(context_raw.get("num_random_samples", 4)),
        min_chunk_chars=int(context_raw.get("min_chunk_chars", 400)),
    )

    single_raw = raw.get("single_hop", {}) or {}
    single_hop = SingleHopConfig(
        num_samples=int(single_raw.get("num_samples", 40)),
        min_chunk_chars=int(single_raw.get("min_chunk_chars", 400)),
        context_preview_chars=int(single_raw.get("context_preview_chars", 200)),
        max_questions_per_chunk=int(single_raw.get("max_questions_per_chunk", 1)),
        max_concurrent=int(single_raw.get("max_concurrent", 10)),
        max_tokens=int(single_raw.get("max_tokens", 1000)),
        timeout=float(single_raw.get("timeout", 120.0)),
    )

    multi_raw = raw.get("multi_hop", {}) or {}
    multi_anchor_raw = multi_raw.get("anchor", {}) or {}
    multi_hop = MultiHopConfig(
        num_samples=int(multi_raw.get("num_samples", 20)),
        min_chunk_chars=int(multi_raw.get("min_chunk_chars", 400)),
        context_preview_chars=int(multi_raw.get("context_preview_chars", 200)),
        max_questions_per_pair=int(multi_raw.get("max_questions_per_pair", 1)),
        top_related_chunks=int(multi_raw.get("top_related_chunks", 3)),
        bm25_enrichment_queries=int(multi_raw.get("bm25_enrichment_queries", 3)),
        bm25_enrichment_top_k=int(multi_raw.get("bm25_enrichment_top_k", 5)),
        max_bm25_related_refs=int(multi_raw.get("max_bm25_related_refs", 3)),
        max_tokens=int(multi_raw.get("max_tokens", 2000)),
        timeout=float(multi_raw.get("timeout", 120.0)),
        anchor=MultiHopAnchorConfig(
            type_distribution=multi_anchor_raw.get("type_distribution"),
            target_hop_counts=multi_anchor_raw.get("target_hop_counts"),
            corpus_pool_size=int(multi_anchor_raw.get("corpus_pool_size", 200)),
        ),
    )

    filter_raw = raw.get("filter", {}) or {}
    filter_cfg = FilterConfig(
        enabled=bool(filter_raw.get("enabled", True)),
        min_question_chars=int(filter_raw.get("min_question_chars", 12)),
        min_answer_chars=int(filter_raw.get("min_answer_chars", 24)),
        min_reference_chunks=int(filter_raw.get("min_reference_chunks", 1)),
    )

    style_raw = raw.get("question_style_mix", {}) or {}
    style_mix = QuestionStyleMixConfig(
        enabled=bool(style_raw.get("enabled", True)),
        distribution=_normalize_style_distribution(style_raw.get("distribution")),
        tolerance=max(0.0, min(0.5, float(style_raw.get("tolerance", 0.10)))),
        apply_to_single_hop=bool(style_raw.get("apply_to_single_hop", True)),
        apply_to_multi_hop=bool(style_raw.get("apply_to_multi_hop", True)),
        max_topup_rounds=max(0, int(style_raw.get("max_topup_rounds", 2))),
        enforce_style_mismatch_guard=bool(style_raw.get("enforce_style_mismatch_guard", False)),
    )

    rewrite_raw = raw.get("query_rewrite", {}) or {}
    query_rewrite = QueryRewriteConfig(
        enabled=bool(rewrite_raw.get("enabled", True)),
        source=str(rewrite_raw.get("source", "heuristic")).strip().lower() or "heuristic",
        max_terms=max(1, int(rewrite_raw.get("max_terms", 16))),
        max_chars=max(20, int(rewrite_raw.get("max_chars", 140))),
        apply_to_all_rows=bool(rewrite_raw.get("apply_to_all_rows", True)),
        write_split_datasets=bool(rewrite_raw.get("write_split_datasets", True)),
    )

    prompts_raw = raw.get("prompts", {}) or {}
    prompts = HybridPromptConfig(
        corpus_system=str(prompts_raw.get("corpus_system", "")),
        corpus_user=str(prompts_raw.get("corpus_user", "")),
        single_hop_system=str(prompts_raw.get("single_hop_system", "")),
        single_hop_user=str(prompts_raw.get("single_hop_user", "")),
        multi_hop_system=str(prompts_raw.get("multi_hop_system", "")),
        multi_hop_user=str(prompts_raw.get("multi_hop_user", "")),
        no_context_filter_system=str(prompts_raw.get("no_context_filter_system", "")),
        equivalence_judge_system=str(prompts_raw.get("equivalence_judge_system", "")),
        question_style_instruction=str(prompts_raw.get("question_style_instruction", "")),
    )

    output_raw = raw.get("output", {}) or {}
    output = OutputConfig(
        dir=str(output_raw.get("dir", "outputs/hybrid")).strip(),
        raw_yaml=str(output_raw.get("raw_yaml", "raw_candidates.yaml")).strip(),
        raw_jsonl=str(output_raw.get("raw_jsonl", "raw_candidates.jsonl")).strip(),
        filtered_yaml=str(output_raw.get("filtered_yaml", "filtered_dataset.yaml")).strip(),
        filtered_jsonl=str(output_raw.get("filtered_jsonl", "filtered_dataset.jsonl")).strip(),
        filter_stats_json=str(output_raw.get("filter_stats_json", "filter_stats.json")).strip(),
        rewriter_jsonl=str(output_raw.get("rewriter_jsonl", "rewriter_dataset.jsonl")).strip(),
        retriever_jsonl=str(output_raw.get("retriever_jsonl", "retriever_dataset.jsonl")).strip(),
        agent_jsonl=str(output_raw.get("agent_jsonl", "agent_dataset.jsonl")).strip(),
    )

    cfg = HybridPipelineConfig(
        platform=platform,
        corpus=corpus,
        generator_model=generator_model,
        filter_target_model=filter_target_model,
        filter_judge_model=filter_judge_model,
        context=context,
        single_hop=single_hop,
        multi_hop=multi_hop,
        question_style_mix=style_mix,
        query_rewrite=query_rewrite,
        prompts=prompts,
        filter=filter_cfg,
        output=output,
    )
    cfg.resolve_api_keys()
    cfg.prompts.resolve()
    return cfg


def _build_openai_client(model_cfg: ModelConfig) -> Any:
    return OpenAI(base_url=model_cfg.base_url, api_key=model_cfg.api_key)


def _load_source(cfg: HybridPipelineConfig) -> CorporaChunkSource:
    source = CorporaChunkSource(
        api_key=cfg.platform.api_key,
        corpus_name=cfg.corpus.corpus_name,
        base_url=cfg.platform.base_url,
    )

    if cfg.corpus.docs_path:
        source.populate_from_folder(cfg.corpus.docs_path, show_summary=cfg.corpus.show_summary)
        return source

    if cfg.corpus.corpus_id:
        source.populate_from_existing_corpus(
            cfg.corpus.corpus_id, show_summary=cfg.corpus.show_summary
        )
        return source

    source.populate_from_existing_corpus_name(
        cfg.corpus.corpus_name, show_summary=cfg.corpus.show_summary
    )
    return source


def _create_chat_completion_with_fallback(client: Any, **kwargs: Any) -> Any:
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        retry_kwargs = dict(kwargs)
        if "unsupported parameter" in msg and "max_tokens" in msg:
            retry_kwargs.pop("max_tokens", None)
            return client.chat.completions.create(**retry_kwargs)
        if "unsupported parameter" in msg and "max_completion_tokens" in msg:
            retry_kwargs.pop("max_completion_tokens", None)
            return client.chat.completions.create(**retry_kwargs)
        raise


def _chunk_to_reference_chunk(chunk: Any) -> ReferenceChunk:
    metadata: dict[str, Any] = {}
    if hasattr(chunk, "metadata_dict"):
        metadata = dict(chunk.metadata_dict)
    elif hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        metadata = dict(chunk.metadata)
    elif isinstance(chunk, dict):
        metadata = dict(chunk.get("metadata", {}) or {})

    chunk_id = (
        getattr(chunk, "hash", None)
        or metadata.get("id")
        or metadata.get("file")
        or metadata.get("file_name")
        or metadata.get("document_id")
        or str(chunk)[:80]
    )
    content = chunk.content if hasattr(chunk, "content") else str(chunk)

    return {
        "id": str(chunk_id),
        "metadata": metadata,
        "content": str(content),
    }


def _reference_chunk_ids(reference_chunks: list[ReferenceChunk]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ref in reference_chunks:
        rid = str(ref.get("id", "")).strip()
        if rid and rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def _detect_co_located(reference_chunks: list[ReferenceChunk]) -> bool | None:
    if not reference_chunks:
        return None
    doc_ids: list[str] = []
    for ref in reference_chunks:
        meta = ref.get("metadata", {}) or {}
        if not isinstance(meta, dict):
            return None
        doc_id = (
            meta.get("document_id")
            or meta.get("file_name")
            or meta.get("file")
            or meta.get("source")
            or meta.get("doc_id")
        )
        if not doc_id:
            return None
        doc_ids.append(str(doc_id))
    return len(set(doc_ids)) == 1


def _ensure_qapoint_shape(item: dict[str, Any] | QADataPoint) -> QADataPoint:
    point = dict(item)
    point.setdefault("question", "")
    point.setdefault("user_question", "")
    point.setdefault("retrieval_query", "")
    point.setdefault("answer", "")
    point.setdefault("reference_chunks", [])
    point.setdefault("qa_type", "unknown")
    point.setdefault("min_hop_count", None)
    point.setdefault("is_co_located", None)
    point.setdefault("filter_status", None)
    point.setdefault("filter_reasoning", None)
    point.setdefault("no_context_answer", None)
    if not isinstance(point.get("eval_scores"), dict):
        point["eval_scores"] = {}
    return cast(QADataPoint, point)


def _normalize_question(question: str) -> str:
    collapsed = re.sub(r"\s+", " ", question.strip().lower())
    return re.sub(r"[^a-z0-9\s]", "", collapsed).strip()


def _mark_eval_scores(point: QADataPoint, **kwargs: Any) -> QADataPoint:
    eval_scores = dict(point.get("eval_scores") or {})
    eval_scores.update(kwargs)
    point["eval_scores"] = eval_scores
    return point


def _allocate_largest_remainder(total: int, distribution: dict[str, float]) -> dict[str, int]:
    return _shared_allocate_largest_remainder(total, distribution)


def _style_minimum_counts(
    total: int,
    distribution: dict[str, float],
    tolerance: float,
) -> dict[str, int]:
    normalized = _normalize_style_distribution(distribution)
    bounded_tolerance = max(0.0, min(0.5, tolerance))
    minimums: dict[str, int] = {}
    for style in QUERY_STYLE_KEYS:
        lower_ratio = max(0.0, normalized[style] - bounded_tolerance)
        minimums[style] = int(math.ceil(total * lower_ratio))
    return minimums


def _style_sequence_from_counts(style_counts: dict[str, int]) -> list[str]:
    styles = _shared_style_sequence_from_counts(style_counts)
    random.shuffle(styles)
    return styles


def _render_style_instruction(
    style_target: str | None,
    prompt_cfg: HybridPromptConfig | None = None,
) -> str:
    target = style_target if style_target in QUERY_STYLE_KEYS else "mixed"
    template = (
        prompt_cfg.get_question_style_instruction()
        if prompt_cfg is not None
        else QUESTION_STYLE_INSTRUCTION_TEMPLATE
    )
    return template.format(query_style_target=target)


def _classify_query_style(question: str) -> str:
    return _shared_classify_query_style(question)


def _contains_unresolved_component(text: str) -> bool:
    # Detect unresolved MDX/JSX component tags while allowing lowercase placeholders.
    return bool(re.search(r"<[A-Z][A-Za-z0-9]*(?:\s[^>]*)?/?>", text))


def _annotate_query_style(point: QADataPoint) -> QADataPoint:
    eval_scores = dict(point.get("eval_scores") or {})
    observed = _classify_query_style(str(point.get("question", "")))
    eval_scores["query_style_observed"] = observed
    point["eval_scores"] = eval_scores
    return point


def _annotate_user_and_retrieval_query(
    point: QADataPoint,
    rewrite_cfg: QueryRewriteConfig,
) -> QADataPoint:
    user_question = str(point.get("user_question", "")).strip() or str(point.get("question", "")).strip()
    point["user_question"] = user_question

    if not user_question:
        point["retrieval_query"] = ""
        return point

    should_rewrite = rewrite_cfg.enabled
    if should_rewrite and not rewrite_cfg.apply_to_all_rows:
        observed = str((point.get("eval_scores") or {}).get("query_style_observed", "")).strip()
        if observed not in QUERY_STYLE_KEYS:
            observed = _classify_query_style(user_question)
        should_rewrite = observed in {QUERY_STYLE_NATURAL, QUERY_STYLE_EXPERT}

    if should_rewrite:
        if rewrite_cfg.source == "heuristic":
            rewritten = heuristic_query_rewrite(
                user_question,
                max_terms=rewrite_cfg.max_terms,
                max_chars=rewrite_cfg.max_chars,
            )
        else:
            rewritten = heuristic_query_rewrite(
                user_question,
                max_terms=rewrite_cfg.max_terms,
                max_chars=rewrite_cfg.max_chars,
            )
        point["retrieval_query"] = rewritten or user_question
    else:
        point["retrieval_query"] = user_question

    return point


def _rewrite_counters(rows: list[QADataPoint]) -> dict[str, int]:
    rewritten_total = 0
    unchanged_total = 0
    for row in rows:
        point = _ensure_qapoint_shape(row)
        user_question = str(point.get("user_question", "")).strip() or str(point.get("question", "")).strip()
        retrieval_query = str(point.get("retrieval_query", "")).strip()
        if not user_question and not retrieval_query:
            unchanged_total += 1
            continue
        if retrieval_query and retrieval_query != user_question:
            rewritten_total += 1
        else:
            unchanged_total += 1
    return {
        "rewritten_total": rewritten_total,
        "unchanged_total": unchanged_total,
    }


def _style_distribution(rows: list[QADataPoint]) -> dict[str, int]:
    counts = {key: 0 for key in QUERY_STYLE_KEYS}
    for row in rows:
        point = _ensure_qapoint_shape(row)
        eval_scores = point.get("eval_scores", {}) or {}
        observed = str(eval_scores.get("query_style_observed", "")).strip()
        if observed not in QUERY_STYLE_KEYS:
            observed = _classify_query_style(str(point.get("question", "")))
        counts[observed] += 1
    return counts


def _style_shortfalls(
    *,
    expected_counts: dict[str, int],
    observed_counts: dict[str, int],
) -> dict[str, int]:
    return {
        style: max(0, expected_counts.get(style, 0) - observed_counts.get(style, 0))
        for style in QUERY_STYLE_KEYS
    }


def _cap_shortfalls(shortfalls: dict[str, int], max_total: int) -> dict[str, int]:
    total_shortfall = sum(shortfalls.values())
    if total_shortfall <= max_total:
        return dict(shortfalls)
    if total_shortfall <= 0 or max_total <= 0:
        return {style: 0 for style in QUERY_STYLE_KEYS}

    weights = {style: shortfalls[style] / total_shortfall for style in QUERY_STYLE_KEYS}
    return _allocate_largest_remainder(max_total, weights)


def _print_style_distribution(stage: str, rows: list[QADataPoint]) -> None:
    counts = _style_distribution(rows)
    total = len(rows)
    print(
        f"[style::{stage}] total={total}, "
        f"keyword={counts[QUERY_STYLE_KEYWORD]}, "
        f"natural={counts[QUERY_STYLE_NATURAL]}, "
        f"expert={counts[QUERY_STYLE_EXPERT]}"
    )


def _render_corpus_user_context(context_cfg: ContextConfig) -> str:
    provided_queries = ", ".join(context_cfg.example_queries) if context_cfg.example_queries else ""
    return (
        f"Description: {context_cfg.description}\n"
        f"Example queries provided by user: {provided_queries}"
    )


def _generate_corpus_context(
    source: CorporaChunkSource,
    client: Any,
    model_cfg: ModelConfig,
    context_cfg: ContextConfig,
    prompt_cfg: HybridPromptConfig | None = None,
) -> dict[str, Any]:
    top_level = source.get_top_level_chunks()
    sampled_top_level = random.sample(top_level, min(context_cfg.num_top_level_samples, len(top_level)))
    sampled_random = source.sample_chunks(
        context_cfg.num_random_samples, min_chars=context_cfg.min_chunk_chars
    )

    variables = {
        "user_context": _render_corpus_user_context(context_cfg),
        "top_level_content": "\n\n".join([chunk.chunk_str() for chunk in sampled_top_level]),
        "random_content": "\n\n".join([chunk.chunk_str() for chunk in sampled_random]),
    }
    user_template = prompt_cfg.get_corpus_user() if prompt_cfg is not None else CORPUS_USER_TEMPLATE
    system_prompt = prompt_cfg.get_corpus_system() if prompt_cfg is not None else CORPUS_SYSTEM_PROMPT
    user_prompt = render_template(user_template, variables)

    response = _create_chat_completion_with_fallback(
        client,
        model=model_cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1.0 if "gpt" in model_cfg.model.lower() else 0.2,
        max_completion_tokens=2000,
    )
    response_text = response.choices[0].message.content or ""
    summary, queries = parse_corpus_summary_response(response_text)

    if not summary.strip() or not queries:
        raise ValueError(
            "Corpus context generation failed: expected non-empty summary and example_queries. "
            "Adjust context settings or model prompt and retry."
        )

    query_lines = "\n".join([f"- {q}" for q in queries])
    return {
        "corpus_summary": summary.strip(),
        "corpus_queries": query_lines,
        "example_queries": queries,
        "raw_response": response_text,
    }


def _generate_single_hop_candidates(
    source: CorporaChunkSource,
    client: Any,
    model_cfg: ModelConfig,
    context: dict[str, Any],
    cfg: SingleHopConfig,
    style_mix_cfg: QuestionStyleMixConfig | None = None,
    style_sample_overrides: dict[str, int] | None = None,
    prompt_cfg: HybridPromptConfig | None = None,
) -> list[QADataPoint]:
    style_mix_enabled = bool(
        style_mix_cfg
        and style_mix_cfg.enabled
        and style_mix_cfg.apply_to_single_hop
    )

    if style_mix_enabled:
        if style_sample_overrides is not None:
            style_counts = {
                style: max(0, int(style_sample_overrides.get(style, 0)))
                for style in QUERY_STYLE_KEYS
            }
        else:
            style_counts = _allocate_largest_remainder(cfg.num_samples, style_mix_cfg.distribution)
        generation_plan: list[tuple[str | None, int]] = [
            (style, count) for style, count in style_counts.items() if count > 0
        ]
    else:
        generation_plan = [(None, cfg.num_samples)]

    out: list[QADataPoint] = []
    for style_target, sample_count in generation_plan:
        if sample_count <= 0:
            continue
        prompt_context = {
            **context,
            "style_instruction": _render_style_instruction(style_target, prompt_cfg),
        }
        system_template = (
            prompt_cfg.get_single_hop_system() if prompt_cfg is not None else SINGLE_HOP_SYSTEM_TEMPLATE
        )
        user_template = (
            prompt_cfg.get_single_hop_user() if prompt_cfg is not None else SINGLE_HOP_USER_TEMPLATE
        )
        system_prompt = render_template(system_template, prompt_context)
        dataset = generate_single_hop_batch(
            source=source,
            client=client,
            model=model_cfg.model,
            system_prompt=system_prompt,
            user_template=user_template,
            num_samples=sample_count,
            response_parser=parse_single_hop_response,
            min_chunk_chars=cfg.min_chunk_chars,
            context_preview_chars=cfg.context_preview_chars,
            max_concurrent=cfg.max_concurrent,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            max_questions=cfg.max_questions_per_chunk,
        )

        for item in dataset:
            point = _ensure_qapoint_shape(item)
            point["min_hop_count"] = 1
            point["is_co_located"] = _detect_co_located(point["reference_chunks"])
            point = _mark_eval_scores(
                point,
                qa_subtype="single_hop",
                target_hop_count=1,
                anchor_ref_ids=_reference_chunk_ids(point["reference_chunks"]),
                query_style_target=style_target,
            )
            point = _annotate_query_style(point)
            out.append(point)
    return out


def _generate_bm25_queries(chunk: Any, n: int) -> list[str]:
    return _shared_generate_bm25_queries(chunk, n=n)


def _extract_anchor_ref_ids(bundle: AnchorBundle) -> list[str]:
    return _shared_extract_anchor_ref_ids(
        bundle,
        ref_id_fn=lambda c: _chunk_to_reference_chunk(c)["id"],
    )


def _build_multi_hop_candidates(
    bundle: AnchorBundle,
    bm25_results: list[dict[str, Any]],
    anchor_queries: list[str],
    top_related_chunks: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    primary_id = _chunk_to_reference_chunk(bundle.primary_chunk)["id"]

    for secondary in bundle.secondary_chunks:
        sid = _chunk_to_reference_chunk(secondary)["id"]
        if not sid or sid == primary_id or sid in seen:
            continue
        seen.add(sid)
        candidates.append(
            {
                "chunk": secondary,
                "queries": anchor_queries,
                "source_kind": "anchor_secondary",
            }
        )

    for result in bm25_results[:top_related_chunks]:
        candidate_chunk = result.get("chunk")
        if candidate_chunk is None:
            continue
        sid = _chunk_to_reference_chunk(candidate_chunk)["id"]
        if not sid or sid == primary_id or sid in seen:
            continue
        seen.add(sid)
        candidate_queries = result.get("queries", [])
        candidates.append(
            {
                "chunk": candidate_chunk,
                "queries": list(candidate_queries) if isinstance(candidate_queries, list) else [],
                "source_kind": "bm25_related",
            }
        )
    return candidates


def _build_reference_chunks(
    primary_chunk: Any,
    selected_chunk: Any,
    bm25_related_chunks: list[Any],
) -> list[ReferenceChunk]:
    refs = [
        _chunk_to_reference_chunk(primary_chunk),
        _chunk_to_reference_chunk(selected_chunk),
    ]
    for chunk in bm25_related_chunks:
        refs.append(_chunk_to_reference_chunk(chunk))

    out: list[ReferenceChunk] = []
    seen: set[str] = set()
    for ref in refs:
        rid = str(ref["id"])
        if rid not in seen:
            seen.add(rid)
            out.append(ref)
    return out


def _generate_multi_hop_candidates(
    source: CorporaChunkSource,
    client: Any,
    model_cfg: ModelConfig,
    context: dict[str, Any],
    cfg: MultiHopConfig,
    style_mix_cfg: QuestionStyleMixConfig | None = None,
    style_sample_overrides: dict[str, int] | None = None,
    prompt_cfg: HybridPromptConfig | None = None,
) -> list[QADataPoint]:
    sample_count = cfg.num_samples
    if style_sample_overrides is not None:
        sample_count = max(1, sum(max(0, int(v)) for v in style_sample_overrides.values()))

    sampled_chunks = source.sample_chunks(sample_count, min_chars=cfg.min_chunk_chars)
    if not sampled_chunks:
        return []

    capabilities = CorpusCapabilities.detect(sampled_chunks)
    selector = AnchorSelector(
        capabilities,
        type_distribution=cfg.anchor.type_distribution,
        target_hop_counts=cfg.anchor.target_hop_counts,
    )

    collection_size = len(source.collection) if source.collection is not None else len(sampled_chunks)
    pool_size = min(cfg.anchor.corpus_pool_size, collection_size)
    corpus_pool = (
        source.sample_chunks(pool_size, min_chars=cfg.min_chunk_chars)
        if pool_size > 0
        else list(sampled_chunks)
    )
    if not corpus_pool:
        corpus_pool = list(sampled_chunks)

    dataset: list[QADataPoint] = []
    candidate_jobs: list[tuple[Any, AnchorBundle, list[Any], dict[str, Any]]] = []

    primary_progress = tqdm(
        sampled_chunks,
        desc="Multi-hop: assemble candidates",
        unit="chunk",
    )
    for primary in primary_progress:
        enriched = _shared_select_anchor_bundle_with_enrichment(
            selector=selector,
            primary_chunk=primary,
            corpus_pool=corpus_pool,
            source=source,
            bm25_enrichment_queries=cfg.bm25_enrichment_queries,
            bm25_enrichment_top_k=cfg.bm25_enrichment_top_k,
            max_related_refs=cfg.max_bm25_related_refs,
            include_search_payload=True,
        )
        if isinstance(enriched, tuple):
            bundle, enrichment_queries, bm25_results = enriched
        else:
            bundle = enriched
            enrichment_queries = _generate_bm25_queries(primary, cfg.bm25_enrichment_queries)
            bm25_results = source.search_related(
                primary,
                enrichment_queries,
                top_k=cfg.bm25_enrichment_top_k,
            )
        bm25_related_chunks = list(bundle.structural_hints.get("bm25_related", []))

        candidates = _build_multi_hop_candidates(
            bundle=bundle,
            bm25_results=bm25_results,
            anchor_queries=enrichment_queries,
            top_related_chunks=cfg.top_related_chunks,
        )

        for candidate in candidates:
            candidate_jobs.append((primary, bundle, bm25_related_chunks, candidate))
        primary_progress.set_postfix(total_pairs=len(candidate_jobs))

    if not candidate_jobs:
        return dataset

    style_mix_enabled = bool(style_mix_cfg and style_mix_cfg.enabled and style_mix_cfg.apply_to_multi_hop)
    style_distribution = (
        style_mix_cfg.distribution if style_mix_cfg is not None else DEFAULT_QUERY_STYLE_DISTRIBUTION
    )
    if style_sample_overrides is not None and sum(style_sample_overrides.values()) > 0:
        style_distribution = _normalize_style_distribution(style_sample_overrides)
    if style_mix_enabled:
        style_counts = _allocate_largest_remainder(len(candidate_jobs), style_distribution)
        style_targets = _style_sequence_from_counts(style_counts)
    else:
        style_targets = [None] * len(candidate_jobs)

    system_prompt_cache: dict[str, str] = {}
    validation_progress = tqdm(
        enumerate(candidate_jobs),
        desc="Multi-hop: validate pairs",
        unit="pair",
    )
    for idx, (primary, bundle, bm25_related_chunks, candidate) in validation_progress:
        secondary = candidate["chunk"]
        candidate_queries = candidate.get("queries", [])
        if not isinstance(candidate_queries, list):
            candidate_queries = []
        style_target = style_targets[idx] if idx < len(style_targets) else None

        cache_key = style_target or "mixed"
        if cache_key not in system_prompt_cache:
            prompt_context = {
                **context,
                "style_instruction": _render_style_instruction(style_target, prompt_cfg),
            }
            system_template = (
                prompt_cfg.get_multi_hop_system()
                if prompt_cfg is not None
                else MULTI_HOP_SYSTEM_TEMPLATE
            )
            system_prompt_cache[cache_key] = render_template(
                system_template, prompt_context
            )
        system_prompt = system_prompt_cache[cache_key]

        ctx_a = source.get_chunk_with_context(primary, max_chars=cfg.context_preview_chars)
        ctx_b = source.get_chunk_with_context(secondary, max_chars=cfg.context_preview_chars)
        user_template = (
            prompt_cfg.get_multi_hop_user() if prompt_cfg is not None else MULTI_HOP_USER_TEMPLATE
        )
        user_prompt = render_template(
            user_template,
            {
                "connecting_queries": ", ".join(candidate_queries),
                "chunk_a": ctx_a["chunk_content"],
                "chunk_a_context_before": ctx_a["prev_chunk_preview"],
                "chunk_a_context_after": ctx_a["next_chunk_preview"],
                "chunk_b": ctx_b["chunk_content"],
                "chunk_b_context_before": ctx_b["prev_chunk_preview"],
                "chunk_b_context_after": ctx_b["next_chunk_preview"],
            },
        )

        completion = _create_chat_completion_with_fallback(
            client,
            model=model_cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0 if "gpt" in model_cfg.model.lower() else 0.2,
            max_completion_tokens=cfg.max_tokens,
        )
        response_text = completion.choices[0].message.content or ""
        qa_pairs = parse_multi_hop_validation_response(response_text)
        if not qa_pairs:
            continue

        anchor_ref_ids = _extract_anchor_ref_ids(bundle)
        for qa in qa_pairs[: cfg.max_questions_per_pair]:
            refs = _build_reference_chunks(primary, secondary, bm25_related_chunks)
            point = _ensure_qapoint_shape(
                {
                    "question": str(qa.get("question", "")).strip(),
                    "answer": str(qa.get("answer", "")).strip(),
                    "reference_chunks": refs,
                    "qa_type": "multi_hop",
                    "min_hop_count": None,
                    "is_co_located": _detect_co_located(refs),
                    "filter_status": None,
                    "filter_reasoning": None,
                    "no_context_answer": None,
                    "eval_scores": {
                        "qa_subtype": bundle.target_qa_type,
                        "target_hop_count": bundle.target_hop_count,
                        "anchor_ref_ids": anchor_ref_ids,
                        "candidate_source": candidate.get("source_kind"),
                        "connecting_queries": candidate_queries,
                        "query_style_target": style_target,
                    },
                }
            )
            point = _annotate_query_style(point)
            dataset.append(point)
        validation_progress.set_postfix(generated=len(dataset))
    return dataset


def _apply_deterministic_guards(
    qa_pairs: list[QADataPoint],
    cfg: FilterConfig,
    style_mix_cfg: QuestionStyleMixConfig | None = None,
    rewrite_cfg: QueryRewriteConfig | None = None,
) -> tuple[list[QADataPoint], list[QADataPoint], dict[str, Any]]:
    passed: list[QADataPoint] = []
    rejected: list[QADataPoint] = []
    seen_questions: set[str] = set()
    rejection_counter: Counter[str] = Counter()
    style_mismatch_observed = 0
    style_mismatch_rejected = 0

    for pair in qa_pairs:
        point = _ensure_qapoint_shape(pair)
        point = _annotate_query_style(point)
        if rewrite_cfg is not None:
            point = _annotate_user_and_retrieval_query(point, rewrite_cfg)
        question = str(point.get("question", "")).strip()
        answer = str(point.get("answer", "")).strip()
        references = point.get("reference_chunks", [])
        eval_scores = point.get("eval_scores", {}) or {}
        query_style_target = str(eval_scores.get("query_style_target", "")).strip()
        query_style_observed = str(eval_scores.get("query_style_observed", "")).strip()
        has_style_mismatch = (
            style_mix_cfg is not None
            and style_mix_cfg.enabled
            and query_style_target in QUERY_STYLE_KEYS
            and query_style_observed in QUERY_STYLE_KEYS
            and query_style_target != query_style_observed
        )
        if has_style_mismatch:
            style_mismatch_observed += 1

        reason: str | None = None
        if not question or not answer:
            reason = "empty_question_or_answer"
        elif _contains_unresolved_component(question) or _contains_unresolved_component(answer):
            reason = "contains_unresolved_component"
        elif len(question) < cfg.min_question_chars:
            reason = "question_too_short"
        elif len(answer) < cfg.min_answer_chars:
            reason = "answer_too_short"
        elif len(references) < cfg.min_reference_chunks:
            reason = "insufficient_reference_chunks"
        elif has_style_mismatch and bool(style_mix_cfg and style_mix_cfg.enforce_style_mismatch_guard):
            reason = "style_mismatch"
            style_mismatch_rejected += 1
        else:
            norm_question = _normalize_question(question)
            if norm_question in seen_questions:
                reason = "duplicate_question"
            else:
                seen_questions.add(norm_question)

        if reason:
            point["filter_status"] = "rejected_guard"
            point["filter_reasoning"] = reason
            point["no_context_answer"] = None
            rejected.append(point)
            rejection_counter[reason] += 1
            continue

        passed.append(point)

    stats = {
        "total": len(qa_pairs),
        "passed": len(passed),
        "rejected": len(rejected),
        "rejection_reasons": dict(rejection_counter),
        "style_guard_rejections": {
            "style_mismatch": int(rejection_counter.get("style_mismatch", 0)),
            "contains_unresolved_component": int(
                rejection_counter.get("contains_unresolved_component", 0)
            ),
        },
        "style_mismatch_observed": style_mismatch_observed,
        "style_mismatch_rejected": style_mismatch_rejected,
    }
    return passed, rejected, stats


def _run_style_topup_rounds(
    *,
    source: CorporaChunkSource,
    client: Any,
    model_cfg: ModelConfig,
    context: dict[str, Any],
    single_cfg: SingleHopConfig,
    multi_cfg: MultiHopConfig,
    filter_cfg: FilterConfig,
    style_mix_cfg: QuestionStyleMixConfig,
    rewrite_cfg: QueryRewriteConfig,
    prompt_cfg: HybridPromptConfig | None,
    expected_style_counts: dict[str, int],
    guard_passed: list[QADataPoint],
    guard_rejected: list[QADataPoint],
) -> tuple[list[QADataPoint], list[QADataPoint], dict[str, Any]]:
    if not style_mix_cfg.enabled or style_mix_cfg.max_topup_rounds <= 0:
        return guard_passed, guard_rejected, {"enabled": False, "rounds": []}
    if not style_mix_cfg.apply_to_single_hop and not style_mix_cfg.apply_to_multi_hop:
        return guard_passed, guard_rejected, {"enabled": False, "rounds": []}

    passed_rows = list(guard_passed)
    rejected_rows = list(guard_rejected)
    round_stats: list[dict[str, Any]] = []

    for round_idx in range(1, style_mix_cfg.max_topup_rounds + 1):
        before_counts = _style_distribution(passed_rows)
        shortfalls = _style_shortfalls(
            expected_counts=expected_style_counts,
            observed_counts=before_counts,
        )
        if sum(shortfalls.values()) <= 0:
            break

        capped_shortfalls = _cap_shortfalls(shortfalls, MAX_STYLE_TOPUP_PER_ROUND)
        if sum(capped_shortfalls.values()) <= 0:
            break

        round_generated: list[QADataPoint] = []
        if style_mix_cfg.apply_to_single_hop:
            round_generated.extend(
                _generate_single_hop_candidates(
                    source=source,
                    client=client,
                    model_cfg=model_cfg,
                    context=context,
                    cfg=single_cfg,
                    style_mix_cfg=style_mix_cfg,
                    style_sample_overrides=capped_shortfalls,
                    prompt_cfg=prompt_cfg,
                )
            )

        remaining_shortfalls = _style_shortfalls(
            expected_counts=expected_style_counts,
            observed_counts=_style_distribution(passed_rows + round_generated),
        )
        if style_mix_cfg.apply_to_multi_hop and sum(remaining_shortfalls.values()) > 0:
            round_generated.extend(
                _generate_multi_hop_candidates(
                    source=source,
                    client=client,
                    model_cfg=model_cfg,
                    context=context,
                    cfg=multi_cfg,
                    style_mix_cfg=style_mix_cfg,
                    style_sample_overrides=remaining_shortfalls,
                    prompt_cfg=prompt_cfg,
                )
            )

        if not round_generated:
            round_stats.append(
                {
                    "round": round_idx,
                    "requested_shortfalls": shortfalls,
                    "capped_shortfalls": capped_shortfalls,
                    "generated": 0,
                    "passed": 0,
                    "rejected": 0,
                    "after_counts": before_counts,
                }
            )
            break

        round_passed, round_rejected, _ = _apply_deterministic_guards(
            round_generated,
            filter_cfg,
            style_mix_cfg,
            rewrite_cfg,
        )
        passed_rows.extend(round_passed)
        rejected_rows.extend(round_rejected)
        after_counts = _style_distribution(passed_rows)
        round_stats.append(
            {
                "round": round_idx,
                "requested_shortfalls": shortfalls,
                "capped_shortfalls": capped_shortfalls,
                "generated": len(round_generated),
                "passed": len(round_passed),
                "rejected": len(round_rejected),
                "after_counts": after_counts,
            }
        )

        if _style_shortfalls(expected_counts=expected_style_counts, observed_counts=after_counts) == {
            QUERY_STYLE_KEYWORD: 0,
            QUERY_STYLE_NATURAL: 0,
            QUERY_STYLE_EXPERT: 0,
        }:
            break

    payload = {
        "enabled": True,
        "rounds": round_stats,
        "total_generated": sum(int(row.get("generated", 0)) for row in round_stats),
        "total_passed": sum(int(row.get("passed", 0)) for row in round_stats),
        "total_rejected": sum(int(row.get("rejected", 0)) for row in round_stats),
    }
    return passed_rows, rejected_rows, payload


def _run_filter_agent(
    qa_pair: QADataPoint,
    *,
    target_client: Any,
    judge_client: Any,
    target_model: str,
    judge_model: str,
    prompt_cfg: HybridPromptConfig | None = None,
) -> QADataPoint:
    point = _ensure_qapoint_shape(qa_pair)
    no_context_filter_system = (
        prompt_cfg.get_no_context_filter_system()
        if prompt_cfg is not None
        else NO_CONTEXT_FILTER_SYSTEM_PROMPT
    )
    equivalence_judge_system = (
        prompt_cfg.get_equivalence_judge_system()
        if prompt_cfg is not None
        else EQUIVALENCE_JUDGE_SYSTEM_PROMPT
    )

    no_context_response = _create_chat_completion_with_fallback(
        target_client,
        model=target_model,
        messages=[
            {"role": "system", "content": no_context_filter_system},
            {"role": "user", "content": point["question"]},
        ],
        temperature=1.0 if "gpt" in target_model.lower() else 0.2,
        max_completion_tokens=800,
    )
    attempted_answer = no_context_response.choices[0].message.content or ""

    judge_response = _create_chat_completion_with_fallback(
        judge_client,
        model=judge_model,
        messages=[
            {"role": "system", "content": equivalence_judge_system},
            {
                "role": "user",
                "content": (
                    f"Reference answer: {point['answer']}\n\n"
                    f"Attempted answer: {attempted_answer}\n\n"
                    "Are these equivalent?"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=1.0 if "gpt" in judge_model.lower() else 0.2,
        max_completion_tokens=600,
    )
    judge_text = judge_response.choices[0].message.content or "{}"

    parse_error = False
    try:
        judgment = json.loads(judge_text)
        is_equivalent = bool(judgment.get("is_equivalent", False))
        reasoning = str(judgment.get("reasoning", "")).strip() or "No reasoning provided."
    except json.JSONDecodeError:
        parse_error = True
        is_equivalent = False
        reasoning = "judge_response_parse_error"

    if parse_error:
        status = "rejected"
    else:
        status = "rejected" if is_equivalent else "passed"

    point["filter_status"] = status
    point["filter_reasoning"] = reasoning
    point["no_context_answer"] = attempted_answer
    return point


def _run_filter_pipeline(
    qa_pairs: list[QADataPoint],
    *,
    target_client: Any,
    judge_client: Any,
    target_model: str,
    judge_model: str,
    prompt_cfg: HybridPromptConfig | None = None,
) -> tuple[list[QADataPoint], list[QADataPoint], dict[str, Any]]:
    results = [
        _run_filter_agent(
            pair,
            target_client=target_client,
            judge_client=judge_client,
            target_model=target_model,
            judge_model=judge_model,
            prompt_cfg=prompt_cfg,
        )
        for pair in qa_pairs
    ]

    passed = [row for row in results if row.get("filter_status") == "passed"]
    rejected = [row for row in results if row.get("filter_status") != "passed"]
    rejection_reasons = Counter([str(r.get("filter_reasoning", "")) for r in rejected])

    stats = {
        "total": len(results),
        "passed": len(passed),
        "rejected": len(rejected),
        "pass_rate": (len(passed) / len(results)) if results else 0.0,
        "rejection_reasons": dict(rejection_reasons),
    }
    return passed, rejected, stats


def _print_stage(name: str, elapsed: float, details: str) -> None:
    print(f"[{name}] {details} ({elapsed:.2f}s)")


def _build_split_datasets(
    filtered_dataset: list[QADataPoint],
) -> dict[str, list[dict[str, Any]]]:
    rewriter_rows: list[dict[str, Any]] = []
    retriever_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []

    for row in filtered_dataset:
        point = _ensure_qapoint_shape(row)
        eval_scores = dict(point.get("eval_scores", {}) or {})
        user_question = str(point.get("user_question", "")).strip() or str(point.get("question", "")).strip()
        retrieval_query = str(point.get("retrieval_query", "")).strip() or user_question
        reference_chunk_ids = _reference_chunk_ids(point.get("reference_chunks", []))

        rewriter_rows.append(
            {
                "user_question": user_question,
                "retrieval_query": retrieval_query,
                "qa_type": point.get("qa_type"),
                "query_style_observed": eval_scores.get("query_style_observed"),
                "reference_chunk_ids": reference_chunk_ids,
            }
        )

        retriever_row = dict(point)
        retriever_row["question"] = retrieval_query
        retriever_row["user_question"] = user_question
        retriever_row["retrieval_query"] = retrieval_query
        retriever_rows.append(retriever_row)

        agent_rows.append(dict(point))

    return {
        "rewriter": rewriter_rows,
        "retriever": retriever_rows,
        "agent": agent_rows,
    }


def run_hybrid_pipeline(
    cfg: HybridPipelineConfig,
    *,
    source_factory: Callable[[HybridPipelineConfig], CorporaChunkSource] | None = None,
    client_factory: Callable[[ModelConfig], Any] | None = None,
) -> dict[str, Any]:
    """Run the full hybrid pipeline and persist output artifacts."""
    cfg.resolve_api_keys()
    source_loader = source_factory or _load_source
    llm_client_factory = client_factory or _build_openai_client

    source = source_loader(cfg)
    generator_client = llm_client_factory(cfg.generator_model)
    filter_target_client = llm_client_factory(cfg.filter_target_model)
    filter_judge_client = llm_client_factory(cfg.filter_judge_model)

    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "raw_yaml": output_dir / cfg.output.raw_yaml,
        "raw_jsonl": output_dir / cfg.output.raw_jsonl,
        "filtered_yaml": output_dir / cfg.output.filtered_yaml,
        "filtered_jsonl": output_dir / cfg.output.filtered_jsonl,
        "filter_stats_json": output_dir / cfg.output.filter_stats_json,
        "rewriter_jsonl": output_dir / cfg.output.rewriter_jsonl,
        "retriever_jsonl": output_dir / cfg.output.retriever_jsonl,
        "agent_jsonl": output_dir / cfg.output.agent_jsonl,
    }

    stage_durations: dict[str, float] = {}

    stage_start = time.perf_counter()
    context = _generate_corpus_context(
        source,
        generator_client,
        cfg.generator_model,
        cfg.context,
        cfg.prompts,
    )
    stage_durations["context"] = time.perf_counter() - stage_start
    _print_stage(
        "context",
        stage_durations["context"],
        f"summary generated, example_queries={len(context['example_queries'])}",
    )

    stage_start = time.perf_counter()
    single_hop_rows = _generate_single_hop_candidates(
        source=source,
        client=generator_client,
        model_cfg=cfg.generator_model,
        context=context,
        cfg=cfg.single_hop,
        style_mix_cfg=cfg.question_style_mix,
        prompt_cfg=cfg.prompts,
    )
    stage_durations["single_hop_gen"] = time.perf_counter() - stage_start
    _print_stage(
        "single_hop_gen",
        stage_durations["single_hop_gen"],
        f"generated={len(single_hop_rows)}",
    )

    stage_start = time.perf_counter()
    multi_hop_rows = _generate_multi_hop_candidates(
        source=source,
        client=generator_client,
        model_cfg=cfg.generator_model,
        context=context,
        cfg=cfg.multi_hop,
        style_mix_cfg=cfg.question_style_mix,
        prompt_cfg=cfg.prompts,
    )
    stage_durations["multi_hop_gen"] = time.perf_counter() - stage_start
    _print_stage(
        "multi_hop_gen",
        stage_durations["multi_hop_gen"],
        f"generated={len(multi_hop_rows)}",
    )

    raw_candidates = [
        _annotate_user_and_retrieval_query(
            _annotate_query_style(_ensure_qapoint_shape(r)),
            cfg.query_rewrite,
        )
        for r in single_hop_rows + multi_hop_rows
    ]
    style_distribution_raw = _style_distribution(raw_candidates)
    rewrite_counts_raw = _rewrite_counters(raw_candidates)
    _print_style_distribution("raw", raw_candidates)
    save_qa_dataset(raw_candidates, output_paths["raw_yaml"])
    save_qa_dataset_jsonl(raw_candidates, output_paths["raw_jsonl"])

    expected_style_counts = (
        _allocate_largest_remainder(
            len(raw_candidates),
            cfg.question_style_mix.distribution,
        )
        if cfg.question_style_mix.enabled
        else {style: 0 for style in QUERY_STYLE_KEYS}
    )
    style_minimum_counts = (
        _style_minimum_counts(
            len(raw_candidates),
            cfg.question_style_mix.distribution,
            cfg.question_style_mix.tolerance,
        )
        if cfg.question_style_mix.enabled
        else {style: 0 for style in QUERY_STYLE_KEYS}
    )

    stage_start = time.perf_counter()
    guard_passed, guard_rejected, guard_stats = _apply_deterministic_guards(
        raw_candidates,
        cfg.filter,
        cfg.question_style_mix,
        cfg.query_rewrite,
    )
    stage_durations["deterministic_guards"] = time.perf_counter() - stage_start
    _print_stage(
        "deterministic_guards",
        stage_durations["deterministic_guards"],
        f"passed={len(guard_passed)}, rejected={len(guard_rejected)}",
    )
    style_topup_stats = {"enabled": False, "rounds": []}

    stage_start = time.perf_counter()
    if cfg.question_style_mix.enabled and style_minimum_counts:
        guard_passed, guard_rejected, style_topup_stats = _run_style_topup_rounds(
            source=source,
            client=generator_client,
            model_cfg=cfg.generator_model,
            context=context,
            single_cfg=cfg.single_hop,
            multi_cfg=cfg.multi_hop,
            filter_cfg=cfg.filter,
            style_mix_cfg=cfg.question_style_mix,
            rewrite_cfg=cfg.query_rewrite,
            prompt_cfg=cfg.prompts,
            expected_style_counts=style_minimum_counts,
            guard_passed=guard_passed,
            guard_rejected=guard_rejected,
        )
    stage_durations["style_topup"] = time.perf_counter() - stage_start

    guard_rejection_reasons = Counter([str(r.get("filter_reasoning", "")) for r in guard_rejected])
    style_mismatch_observed = sum(
        1
        for row in (guard_passed + guard_rejected)
        if str((row.get("eval_scores", {}) or {}).get("query_style_target", "")).strip() in QUERY_STYLE_KEYS
        and str((row.get("eval_scores", {}) or {}).get("query_style_observed", "")).strip() in QUERY_STYLE_KEYS
        and str((row.get("eval_scores", {}) or {}).get("query_style_target", "")).strip()
        != str((row.get("eval_scores", {}) or {}).get("query_style_observed", "")).strip()
    )
    style_mismatch_rejected = int(guard_rejection_reasons.get("style_mismatch", 0))
    guard_stats = {
        "total": len(guard_passed) + len(guard_rejected),
        "passed": len(guard_passed),
        "rejected": len(guard_rejected),
        "rejection_reasons": dict(guard_rejection_reasons),
        "style_guard_rejections": {
            "style_mismatch": int(guard_rejection_reasons.get("style_mismatch", 0)),
            "contains_unresolved_component": int(
                guard_rejection_reasons.get("contains_unresolved_component", 0)
            ),
        },
        "style_mismatch_observed": style_mismatch_observed,
        "style_mismatch_rejected": style_mismatch_rejected,
    }
    style_distribution_after_guards = _style_distribution(guard_passed)
    _print_style_distribution("after_guards", guard_passed)

    stage_start = time.perf_counter()
    if cfg.filter.enabled:
        filtered_dataset, model_rejected, model_filter_stats = _run_filter_pipeline(
            guard_passed,
            target_client=filter_target_client,
            judge_client=filter_judge_client,
            target_model=cfg.filter_target_model.model,
            judge_model=cfg.filter_judge_model.model,
            prompt_cfg=cfg.prompts,
        )
    else:
        filtered_dataset = []
        model_rejected = []
        for row in guard_passed:
            row["filter_status"] = "passed"
            row["filter_reasoning"] = "model_filter_disabled"
            row["no_context_answer"] = None
            filtered_dataset.append(row)
        model_filter_stats = {
            "total": len(guard_passed),
            "passed": len(filtered_dataset),
            "rejected": 0,
            "pass_rate": 1.0 if guard_passed else 0.0,
            "rejection_reasons": {},
        }
    stage_durations["llm_filter"] = time.perf_counter() - stage_start
    filtered_dataset = [
        _annotate_user_and_retrieval_query(
            _annotate_query_style(_ensure_qapoint_shape(r)),
            cfg.query_rewrite,
        )
        for r in filtered_dataset
    ]
    _print_stage(
        "llm_filter",
        stage_durations["llm_filter"],
        f"passed={len(filtered_dataset)}, rejected={len(model_rejected)}",
    )
    style_distribution_filtered = _style_distribution(filtered_dataset)
    rewrite_counts_filtered = _rewrite_counters(filtered_dataset)
    _print_style_distribution("filtered", filtered_dataset)

    save_qa_dataset(filtered_dataset, output_paths["filtered_yaml"])
    save_qa_dataset_jsonl(filtered_dataset, output_paths["filtered_jsonl"])

    split_datasets = _build_split_datasets(filtered_dataset)
    split_output_paths: dict[str, str] = {}
    if cfg.query_rewrite.write_split_datasets:
        save_jsonl_rows(split_datasets["rewriter"], output_paths["rewriter_jsonl"])
        save_jsonl_rows(split_datasets["retriever"], output_paths["retriever_jsonl"])
        save_jsonl_rows(split_datasets["agent"], output_paths["agent_jsonl"])
        split_output_paths = {
            "rewriter_jsonl": str(output_paths["rewriter_jsonl"]),
            "retriever_jsonl": str(output_paths["retriever_jsonl"]),
            "agent_jsonl": str(output_paths["agent_jsonl"]),
        }

    all_rejected = guard_rejected + model_rejected
    filter_stats_payload = {
        "raw_candidates_total": len(raw_candidates),
        "single_hop_generated": len(single_hop_rows),
        "multi_hop_generated": len(multi_hop_rows),
        "deterministic_guards": guard_stats,
        "style_topup": style_topup_stats,
        "llm_filter": model_filter_stats,
        "style_distribution_raw": style_distribution_raw,
        "style_distribution_after_guards": style_distribution_after_guards,
        "style_distribution_filtered": style_distribution_filtered,
        "style_guard_rejections": guard_stats["style_guard_rejections"],
        "style_mismatch_observed": style_mismatch_observed,
        "style_mismatch_rejected": style_mismatch_rejected,
        "style_expected_counts": expected_style_counts,
        "style_minimum_counts_for_topup": style_minimum_counts,
        "query_rewrite": {
            "enabled": cfg.query_rewrite.enabled,
            "source": cfg.query_rewrite.source,
            "max_terms": cfg.query_rewrite.max_terms,
            "max_chars": cfg.query_rewrite.max_chars,
            "apply_to_all_rows": cfg.query_rewrite.apply_to_all_rows,
            "write_split_datasets": cfg.query_rewrite.write_split_datasets,
        },
        "rewritten_total": int(rewrite_counts_filtered.get("rewritten_total", 0)),
        "unchanged_total": int(rewrite_counts_filtered.get("unchanged_total", 0)),
        "rewrite_counts_raw": rewrite_counts_raw,
        "rewrite_counts_filtered": rewrite_counts_filtered,
        "rewriter_rows": len(split_datasets["rewriter"]),
        "retriever_rows": len(split_datasets["retriever"]),
        "agent_rows": len(split_datasets["agent"]),
        "split_output_paths": split_output_paths,
        "passed_total": len(filtered_dataset),
        "rejected_total": len(all_rejected),
        "stage_durations_seconds": stage_durations,
        "output_paths": {k: str(v) for k, v in output_paths.items()},
    }
    output_paths["filter_stats_json"].write_text(
        json.dumps(filter_stats_payload, indent=2), encoding="utf-8"
    )

    print(
        "[output] "
        f"raw={len(raw_candidates)}, filtered={len(filtered_dataset)}, "
        f"stats={output_paths['filter_stats_json']}"
    )

    return {
        "context": context,
        "raw_candidates": raw_candidates,
        "filtered_dataset": filtered_dataset,
        "split_datasets": split_datasets,
        "rejected_dataset": all_rejected,
        "stats": filter_stats_payload,
        "output_paths": output_paths,
    }


def run_hybrid_pipeline_from_config(
    config_path: str | Path,
    *,
    source_factory: Callable[[HybridPipelineConfig], CorporaChunkSource] | None = None,
    client_factory: Callable[[ModelConfig], Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper to load config and run the hybrid pipeline."""
    cfg = load_hybrid_config(config_path)
    return run_hybrid_pipeline(cfg, source_factory=source_factory, client_factory=client_factory)
