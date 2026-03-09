"""SAGE anchored generator (direct LLM calls with anchor bundles)."""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from synthetic_data_prep.corpus.corpora.source import CorporaChunkSource
from synthetic_data_prep.qa_generation.anchor_selector import AnchorBundle, AnchorSelector
from synthetic_data_prep.qa_generation.corpus_capabilities import CorpusCapabilities
from synthetic_data_prep.qa_generation.generated_qa import GeneratedQA
from synthetic_data_prep.qa_generation.protocols import ChunkLinker
from synthetic_data_prep.qa_generation.sage_utils import (
    SagePipelineConfig,
    _select_anchor_bundle_with_enrichment,
    run_question_generation_direct,
)

from .sage_common import (
    build_environment_constructor_args,
    build_generated_item,
    build_generation_context,
    bundle_search_environment,
    configure_environment_prompts,
    sample_seed_chunks,
    store_shared_context,
)

logger = logging.getLogger(__name__)


class SageAnchoredGenerator:
    """Generates QA using anchor bundles + direct LLM calls.

    Accepts an optional ``linker`` to override the default anchor selection
    strategy.  When provided, ``linker.link()`` is used instead of the
    built-in ``AnchorSelector`` + BM25 enrichment flow.
    """

    def __init__(
        self,
        source: CorporaChunkSource,
        qgen_client: OpenAI,
        cfg: SagePipelineConfig,
        linker: ChunkLinker | None = None,
    ) -> None:
        self.source = source
        self.qgen_client = qgen_client
        self.cfg = cfg
        self.linker = linker

    def generate(self, context: dict[str, Any]) -> list[GeneratedQA]:
        cfg = self.cfg
        sample_chunks, chunk_texts = sample_seed_chunks(self.source, cfg)
        if not chunk_texts:
            logger.warning("No eligible chunks found in corpus.")
            return []

        corpus_context_text, corpus_profile = build_generation_context(
            source=self.source,
            client=self.qgen_client,
            cfg=cfg,
        )
        configure_environment_prompts(cfg, corpus_context_text)

        capabilities = CorpusCapabilities.detect(sample_chunks)
        logger.info("Corpus capabilities: %s", capabilities.describe())
        selector: AnchorSelector | None = AnchorSelector(
            capabilities,
            type_distribution=cfg.anchor.type_distribution,
            target_hop_counts=cfg.anchor.target_hop_counts,
        )

        collection = self.source.collection
        pool_size = min(200, len(collection)) if collection else 200
        corpus_pool = self.source.sample_chunks(pool_size, min_chars=cfg.min_chunk_chars)

        anchors: list[AnchorBundle | None] = [None] * len(chunk_texts)
        for i, chunk in enumerate(sample_chunks):
            if self.linker is not None:
                bundle = self.linker.link(chunk, corpus_pool=corpus_pool)
            else:
                bundle = _select_anchor_bundle_with_enrichment(
                    selector=selector,
                    primary_chunk=chunk,
                    corpus_pool=corpus_pool,
                    source=self.source,
                    cfg=cfg,
                )
            anchors[i] = bundle
            n_refs = (
                len(bundle.secondary_chunks)
                + len(bundle.structural_hints.get("bm25_related", []))
            )
            logger.info(
                "Chunk %d: type=%s, hops=%d, refs=%d",
                i + 1,
                bundle.target_qa_type,
                bundle.target_hop_count,
                n_refs,
            )

        search_constructor_args = build_environment_constructor_args(self.source, cfg)[1]
        logger.info("Bundling SearchAgentEnv...")
        search_cls_bytes, search_meta_bytes = bundle_search_environment(
            cfg, search_constructor_args
        )

        items: list[GeneratedQA] = []
        for chunk_idx, chunk_text in enumerate(chunk_texts):
            anchor = anchors[chunk_idx]
            if anchor is None:
                logger.warning("Skipping chunk %d: anchor bundle unavailable.", chunk_idx)
                continue

            qa = run_question_generation_direct(
                anchor=anchor,
                client=self.qgen_client,
                model_cfg=cfg.question_generator,
                prompt_template=cfg.prompts.get_question_generation_with_refs(),
                corpus_context=corpus_context_text,
            )

            if not qa.get("question") or not qa.get("answer"):
                logger.info(
                    "Chunk %d: could not parse Q/A from generator output.", chunk_idx + 1
                )
                continue

            qa["original_question"] = qa["question"]
            qa["original_answer"] = qa["answer"]
            logger.info("Chunk %d Q: %s", chunk_idx + 1, qa["question"])

            seed_chunk = sample_chunks[chunk_idx] if chunk_idx < len(sample_chunks) else None
            items.append(
                build_generated_item(
                    qa=qa,
                    chunk_idx=chunk_idx,
                    chunk_text=chunk_text,
                    target_steps=anchor.target_hop_count,
                    anchor=anchor,
                    seed_chunk=seed_chunk,
                )
            )

        store_shared_context(
            context,
            anchors=anchors,
            sample_chunks=sample_chunks,
            corpus_context_text=corpus_context_text,
            corpus_profile=corpus_profile,
            search_cls_bytes=search_cls_bytes,
            search_meta_bytes=search_meta_bytes,
            corpus_pool=corpus_pool,
            selector=selector,
            source=self.source,
        )
        logger.info("SageAnchoredGenerator produced %d items", len(items))
        return items
