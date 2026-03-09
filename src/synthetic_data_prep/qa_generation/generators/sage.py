"""SageGenerator — rollout-based or direct LLM generation with anchor bundles."""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from synthetic_data_prep.corpus.corpora.source import CorporaChunkSource
from synthetic_data_prep.qa_generation.anchor_selector import (
    AnchorBundle,
    AnchorSelector,
)
from synthetic_data_prep.qa_generation.corpus_capabilities import CorpusCapabilities
from synthetic_data_prep.qa_generation.generated_qa import GeneratedQA
from synthetic_data_prep.qa_generation.models import QADataPoint
from synthetic_data_prep.qa_generation.sage_utils import (
    QuestionGenEnv,
    SagePipelineConfig,
    SearchAgentEnv,
    _select_anchor_bundle_with_enrichment,
    augment_prompt_with_corpus_context,
    bundle_environment,
    render_corpus_context,
    run_question_generation,
    run_question_generation_direct,
)
from synthetic_data_prep.trainer.client import RolloutClient

logger = logging.getLogger(__name__)


class SageGenerator:
    """Generates QA candidates using the SAGE pipeline approach.

    For anchored generation, uses direct LLM calls with pre-selected reference
    chunks. For non-anchored generation, uses rollout-based generation with a
    bundled QuestionGenEnv.

    Stores shared state (anchors, env bundles, corpus pool, selector) in the
    pipeline ``context`` dict for downstream steps (SageFilter,
    SageFeedbackRegenerator).
    """

    def __init__(
        self,
        source: CorporaChunkSource,
        rollout_client: RolloutClient,
        qgen_client: OpenAI,
        cfg: SagePipelineConfig,
    ) -> None:
        self.source = source
        self.rollout_client = rollout_client
        self.qgen_client = qgen_client
        self.cfg = cfg

    def generate(self, context: dict[str, Any]) -> list[GeneratedQA]:
        """Produce initial QA items via SAGE generation."""
        cfg = self.cfg

        # 1. Sample chunks
        sample_chunks = self.source.sample_chunks(
            cfg.num_samples, min_chars=cfg.min_chunk_chars
        )
        chunk_texts = [str(c) for c in sample_chunks]
        if not chunk_texts:
            logger.warning("No eligible chunks found in corpus.")
            return []

        # 2. Render corpus context
        corpus_context_text = render_corpus_context(cfg.corpus_context)

        # 3. Detect corpus capabilities and create AnchorSelector
        anchors: list[AnchorBundle | None] = [None] * len(chunk_texts)
        selector: AnchorSelector | None = None
        corpus_pool: list[Any] = []

        if cfg.anchor.enabled:
            capabilities = CorpusCapabilities.detect(sample_chunks)
            logger.info("Corpus capabilities: %s", capabilities.describe())
            selector = AnchorSelector(
                capabilities,
                type_distribution=cfg.anchor.type_distribution,
                target_hop_counts=cfg.anchor.target_hop_counts,
            )
            corpus_pool = self.source.sample_chunks(
                min(200, len(self.source.collection)),
                min_chars=cfg.min_chunk_chars,
            )

            # 4. Select anchor bundles with BM25 enrichment
            for i, chunk in enumerate(sample_chunks):
                bundle = _select_anchor_bundle_with_enrichment(
                    selector=selector,
                    primary_chunk=chunk,
                    corpus_pool=corpus_pool,
                    source=self.source,
                    cfg=cfg,
                )
                anchors[i] = bundle
                n_refs = len(bundle.secondary_chunks) + len(
                    bundle.structural_hints.get("bm25_related", [])
                )
                logger.info(
                    "Chunk %d: type=%s, hops=%d, refs=%d",
                    i + 1,
                    bundle.target_qa_type,
                    bundle.target_hop_count,
                    n_refs,
                )

        # 5. Bundle environments
        corpus_id = self.source.corpus_id
        constructor_args = {
            "api_key": cfg.api_key,
            "corpus_id": corpus_id,
            "base_url": cfg.base_url,
        }
        search_constructor_args = dict(constructor_args)
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

        QuestionGenEnv.system_prompt = augment_prompt_with_corpus_context(
            cfg.prompts.get_question_generation(),
            corpus_context_text,
        )
        SearchAgentEnv.system_prompt = cfg.prompts.get_search_agent()

        qgen_cls_bytes, qgen_meta_bytes = None, None
        if not cfg.anchor.enabled:
            logger.info("Bundling QuestionGenEnv...")
            qgen_cls_bytes, qgen_meta_bytes = bundle_environment(
                QuestionGenEnv,
                constructor_args,
                pip_dependencies=["aiohttp"],
            )

        logger.info("Bundling SearchAgentEnv...")
        search_env_pip_deps = ["aiohttp"]
        if cfg.query_rewrite.enabled and cfg.query_rewrite.strategy == "llm":
            search_env_pip_deps.append("openai")
        search_cls_bytes, search_meta_bytes = bundle_environment(
            SearchAgentEnv,
            search_constructor_args,
            pip_dependencies=search_env_pip_deps,
        )

        # 6. Generate Q/A for each chunk
        items: list[GeneratedQA] = []
        for chunk_idx, chunk_text in enumerate(chunk_texts):
            anchor = anchors[chunk_idx]
            target_steps = (
                anchor.target_hop_count if anchor else cfg.n_search_steps
            )

            if anchor:
                qa = run_question_generation_direct(
                    anchor=anchor,
                    client=self.qgen_client,
                    model_cfg=cfg.question_generator,
                    prompt_template=cfg.prompts.get_question_generation_with_refs(),
                    corpus_context=corpus_context_text,
                )
            else:
                if qgen_cls_bytes is None or qgen_meta_bytes is None:
                    logger.warning(
                        "Skipping chunk %d: QuestionGenEnv not bundled.",
                        chunk_idx,
                    )
                    continue
                qa = run_question_generation(
                    chunk_text=chunk_text,
                    n_search_steps=target_steps,
                    corpus_context=corpus_context_text,
                    rollout_client=self.rollout_client,
                    env_cls_bytes=qgen_cls_bytes,
                    env_metadata_bytes=qgen_meta_bytes,
                    model_cfg=cfg.question_generator,
                    rollout_log_full_messages=cfg.rollout_log_full_messages,
                    rollout_log_event_meta=cfg.rollout_log_event_meta,
                    max_retries=cfg.max_retries,
                    retry_delay=cfg.retry_delay,
                )

            if not qa.get("question") or not qa.get("answer"):
                logger.info(
                    "Chunk %d: could not parse Q/A from generator output.",
                    chunk_idx + 1,
                )
                continue

            qa["original_question"] = qa["question"]
            qa["original_answer"] = qa["answer"]

            logger.info("Chunk %d Q: %s", chunk_idx + 1, qa["question"])

            qa_data_point: QADataPoint = {
                "question": qa["question"],
                "answer": qa["answer"],
                "reference_chunks": [],
                "qa_type": anchor.target_qa_type if anchor else "unknown",
                "min_hop_count": target_steps,
                "is_co_located": None,
                "filter_status": None,
                "filter_reasoning": None,
                "no_context_answer": None,
                "eval_scores": {},
            }

            gen_meta: dict[str, Any] = {
                "chunk_idx": chunk_idx,
                "chunk_text": chunk_text[:500],
                "target_steps": target_steps,
                "qa_raw": qa,
            }
            if anchor:
                gen_meta["anchor_qa_type"] = anchor.target_qa_type
                gen_meta["anchor_hop_count"] = anchor.target_hop_count
            seed_chunk = (
                sample_chunks[chunk_idx]
                if chunk_idx < len(sample_chunks)
                else None
            )
            if seed_chunk is not None:
                gen_meta["seed_chunk"] = seed_chunk

            items.append(
                GeneratedQA(qa=qa_data_point, generation_metadata=gen_meta)
            )

        # 7. Store shared state in context for downstream steps
        context["anchors"] = anchors
        context["sample_chunks"] = sample_chunks
        context["corpus_context_text"] = corpus_context_text
        context["search_cls_bytes"] = search_cls_bytes
        context["search_meta_bytes"] = search_meta_bytes
        context["corpus_pool"] = corpus_pool
        context["selector"] = selector
        context["source"] = self.source

        logger.info("SageGenerator produced %d items", len(items))
        return items
