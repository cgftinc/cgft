"""Shared helpers for SAGE generator implementations."""

from __future__ import annotations

import logging
from typing import Any

from synthetic_data_prep.qa_generation.anchor_selector import AnchorBundle
from synthetic_data_prep.qa_generation.generated_qa import GeneratedQA
from synthetic_data_prep.qa_generation.models import QADataPoint, ReferenceChunk
from synthetic_data_prep.qa_generation.sage_utils import (
    QuestionGenEnv,
    SagePipelineConfig,
    SearchAgentEnv,
    augment_prompt_with_corpus_context,
    bundle_environment,
    generate_corpus_auto_profile,
    render_corpus_context,
    render_merged_corpus_context,
)

logger = logging.getLogger(__name__)


def sample_seed_chunks(
    source: Any,
    cfg: SagePipelineConfig,
) -> tuple[list[Any], list[str]]:
    """Sample seed chunks and return both chunk objects and string views."""
    sample_chunks = source.sample_chunks(cfg.num_samples, min_chars=cfg.min_chunk_chars)
    return sample_chunks, [str(c) for c in sample_chunks]


def build_generation_context(
    *,
    source: Any,
    client: Any,
    cfg: SagePipelineConfig,
) -> tuple[str, dict[str, Any] | None]:
    """Build merged corpus context prompt text and optional auto-profile metadata."""
    manual_context = render_corpus_context(cfg.corpus_context)
    if not cfg.corpus_context.auto_profile.enabled:
        return manual_context, None

    try:
        profile = generate_corpus_auto_profile(
            source=source,
            client=client,
            model_cfg=cfg.question_generator,
            corpus_context=cfg.corpus_context,
        )
        merged_context = render_merged_corpus_context(cfg.corpus_context, profile)
        return merged_context or manual_context, profile
    except Exception as exc:
        logger.warning(
            "Corpus auto-profile generation failed; continuing with manual corpus context only: %s",
            exc,
        )
        return manual_context, None


def configure_environment_prompts(cfg: SagePipelineConfig, corpus_context_text: str) -> None:
    """Configure shared environment system prompts."""
    QuestionGenEnv.system_prompt = augment_prompt_with_corpus_context(
        cfg.prompts.get_question_generation(),
        corpus_context_text,
    )
    SearchAgentEnv.system_prompt = cfg.prompts.get_search_agent()


def build_environment_constructor_args(
    source: Any,
    cfg: SagePipelineConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build constructor args for question/search environments."""
    base_constructor_args: dict[str, Any] = {
        "api_key": cfg.api_key,
        "corpus_id": source.corpus_id,
        "base_url": cfg.base_url,
    }
    search_constructor_args: dict[str, Any] = dict(base_constructor_args)
    if cfg.query_rewrite.enabled:
        search_constructor_args.update(
            {
                "query_rewriter_enabled": True,
                "query_rewriter_strategy": cfg.query_rewrite.strategy,
                "query_rewriter_model": cfg.query_rewrite.model.model,
                "query_rewriter_api_key": cfg.query_rewrite.model.api_key,
                "query_rewriter_base_url": cfg.query_rewrite.model.base_url,
                "query_rewriter_max_terms": cfg.query_rewrite.max_terms,
                "query_rewriter_max_chars": cfg.query_rewrite.max_chars,
                "query_rewriter_log": cfg.query_rewrite.log_rewrites,
            }
        )
    return base_constructor_args, search_constructor_args


def bundle_question_generation_environment(constructor_args: dict[str, Any]) -> tuple[bytes, bytes]:
    """Bundle rollout-based question generation environment."""
    return bundle_environment(
        QuestionGenEnv,
        constructor_args,
        pip_dependencies=["aiohttp"],
    )


def bundle_search_environment(
    cfg: SagePipelineConfig,
    search_constructor_args: dict[str, Any],
) -> tuple[bytes, bytes]:
    """Bundle search-agent environment used by SageFilter."""
    search_env_pip_deps = ["aiohttp"]
    if cfg.query_rewrite.enabled and cfg.query_rewrite.strategy == "llm":
        search_env_pip_deps.append("openai")
    return bundle_environment(
        SearchAgentEnv,
        search_constructor_args,
        pip_dependencies=search_env_pip_deps,
    )


def _chunk_to_reference_chunk(chunk: Any) -> ReferenceChunk:
    """Convert a Chunk (or chunk-like object) to a ReferenceChunk dict."""
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
        or str(chunk)[:80]
    )
    content = chunk.content if hasattr(chunk, "content") else str(chunk)
    return {"id": str(chunk_id), "metadata": metadata, "content": str(content)}


def _build_reference_chunks_from_anchor(
    anchor: AnchorBundle,
) -> list[ReferenceChunk]:
    """Extract reference chunks from an AnchorBundle, deduplicated by id."""
    refs = [_chunk_to_reference_chunk(anchor.primary_chunk)]
    refs.extend(_chunk_to_reference_chunk(c) for c in anchor.secondary_chunks)
    refs.extend(
        _chunk_to_reference_chunk(c) for c in anchor.structural_hints.get("bm25_related", [])
    )
    seen: set[str] = set()
    deduped: list[ReferenceChunk] = []
    for ref in refs:
        if ref["id"] not in seen:
            seen.add(ref["id"])
            deduped.append(ref)
    return deduped


def build_generated_item(
    *,
    qa: dict[str, Any],
    chunk_idx: int,
    chunk_text: str,
    target_steps: int,
    anchor: AnchorBundle | None,
    seed_chunk: Any | None,
) -> GeneratedQA:
    """Convert generated QA payload into a GeneratedQA item."""
    qa_data_point: QADataPoint = {
        "question": qa["question"],
        "answer": qa["answer"],
        "reference_chunks": (_build_reference_chunks_from_anchor(anchor) if anchor else []),
        "qa_type": anchor.target_qa_type if anchor else "unknown",
        "min_hop_count": target_steps,
        "is_co_located": None,
        "filter_status": None,
        "filter_reasoning": None,
        "no_context_answer": None,
        "eval_scores": {},
    }
    generation_metadata: dict[str, Any] = {
        "chunk_idx": chunk_idx,
        "chunk_text": chunk_text[:500],
        "target_steps": target_steps,
        "qa_raw": qa,
    }
    if anchor is not None:
        generation_metadata["anchor_qa_type"] = anchor.target_qa_type
        generation_metadata["anchor_hop_count"] = anchor.target_hop_count
    if seed_chunk is not None:
        generation_metadata["seed_chunk"] = seed_chunk
    return GeneratedQA(qa=qa_data_point, generation_metadata=generation_metadata)


def store_shared_context(
    context: dict[str, Any],
    *,
    anchors: list[AnchorBundle | None],
    sample_chunks: list[Any],
    corpus_context_text: str,
    corpus_profile: dict[str, Any] | None,
    search_cls_bytes: bytes,
    search_meta_bytes: bytes,
    corpus_pool: list[Any],
    selector: Any,
    source: Any,
) -> None:
    """Persist shared context keys consumed by SAGE filter/regenerator/formatter."""
    context["anchors"] = anchors
    context["sample_chunks"] = sample_chunks
    context["corpus_context_text"] = corpus_context_text
    context["corpus_profile"] = corpus_profile
    context["search_cls_bytes"] = search_cls_bytes
    context["search_meta_bytes"] = search_meta_bytes
    context["corpus_pool"] = corpus_pool
    context["selector"] = selector
    context["source"] = source
