"""HybridGenerator — wraps hybrid_pipeline generation functions."""

from __future__ import annotations

import logging
from typing import Any

from synthetic_data_prep.corpus.corpora.source import CorporaChunkSource
from synthetic_data_prep.qa_generation.generated_qa import GeneratedQA
from synthetic_data_prep.qa_generation.hybrid_pipeline import (
    ContextConfig,
    ModelConfig,
    MultiHopConfig,
    QuestionStyleMixConfig,
    SingleHopConfig,
    _allocate_largest_remainder,
    _generate_corpus_context,
    _generate_multi_hop_candidates,
    _generate_single_hop_candidates,
)

logger = logging.getLogger(__name__)


class HybridGenerator:
    """Generates QA candidates using hybrid single-hop and multi-hop strategies.

    Wraps ``_generate_corpus_context``, ``_generate_single_hop_candidates``,
    and ``_generate_multi_hop_candidates`` from ``hybrid_pipeline``.
    """

    def __init__(
        self,
        source: CorporaChunkSource,
        client: Any,
        model_cfg: ModelConfig,
        single_hop_cfg: SingleHopConfig,
        multi_hop_cfg: MultiHopConfig,
        context_cfg: ContextConfig,
        style_mix_cfg: QuestionStyleMixConfig | None = None,
    ) -> None:
        self.source = source
        self.client = client
        self.model_cfg = model_cfg
        self.single_hop_cfg = single_hop_cfg
        self.multi_hop_cfg = multi_hop_cfg
        self.context_cfg = context_cfg
        self.style_mix_cfg = style_mix_cfg

    def generate(self, context: dict[str, Any]) -> list[GeneratedQA]:
        """Produce initial QA items via hybrid generation."""
        if "corpus_summary" not in context:
            corpus_context = _generate_corpus_context(
                self.source, self.client, self.model_cfg, self.context_cfg
            )
            context.update(corpus_context)
            logger.info(
                "Corpus context generated, example_queries=%d",
                len(corpus_context.get("example_queries", [])),
            )

        single_hop_rows = _generate_single_hop_candidates(
            source=self.source,
            client=self.client,
            model_cfg=self.model_cfg,
            context=context,
            cfg=self.single_hop_cfg,
            style_mix_cfg=self.style_mix_cfg,
        )
        logger.info("Single-hop generated=%d", len(single_hop_rows))

        multi_hop_rows = _generate_multi_hop_candidates(
            source=self.source,
            client=self.client,
            model_cfg=self.model_cfg,
            context=context,
            cfg=self.multi_hop_cfg,
            style_mix_cfg=self.style_mix_cfg,
        )
        logger.info("Multi-hop generated=%d", len(multi_hop_rows))

        all_rows = single_hop_rows + multi_hop_rows

        # Store expected style counts for downstream style shortfall detection
        if self.style_mix_cfg and self.style_mix_cfg.enabled:
            context["expected_style_counts"] = _allocate_largest_remainder(
                len(all_rows), self.style_mix_cfg.distribution
            )

        items = [GeneratedQA(qa=row) for row in all_rows]
        return items
