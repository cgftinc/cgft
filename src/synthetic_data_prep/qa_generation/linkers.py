"""ChunkLinker implementations for finding related chunks in multi-hop QA generation.

Three standalone linkers, each usable independently:

- ``StructuralChunkLinker`` — metadata-driven selection + deterministic BM25 enrichment
- ``LLMGuidedChunkLinker`` — LLM generates BM25 queries, searches corpus
- ``AdaptiveChunkLinker``  — composes multiple linkers, picks by corpus capabilities
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .anchor_selector import AnchorBundle, AnchorSelector
from .anchor_utils import (
    generate_bm25_queries,
    generate_bm25_queries_from_extraction,
    select_anchor_bundle_with_enrichment,
)
from .corpus_capabilities import CorpusCapabilities
from .helpers import render_template
from .response_parsers import parse_related_queries_response

if TYPE_CHECKING:
    from .cgft_models import EntityExtractionConfig
    from .protocols import ChunkLinker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default prompt for LLM-guided chunk linking
# ---------------------------------------------------------------------------

RELATED_CHUNK_SYSTEM_PROMPT = """\
You are generating BM25 search queries to find chunks that have meaningful \
relationships with the given chunk.

## BM25 Behavior
BM25 matches exact keywords, weighted by rarity. This means:
- Specific/rare terms (product names, technical terms, unique phrases) are powerful
- Common corpus terms (e.g., "API", "data", "system") barely help
- BM25 won't match synonyms: "k8s" won't find "Kubernetes"
- Shorter, focused queries often outperform long ones

## Query Strategies

**1. Entity-focused**: Target specific named things that might appear elsewhere
  - Product/tool names: "Redis", "Workday", "Stripe"
  - Internal terms: "Project Atlas", "Q3 planning", "customer churn analysis"

**2. Reference-chasing**: If this chunk mentions other docs/sections, query for them
  - "see the deployment guide" → query: "deployment guide"

**3. Inverse references**: Query for terms that other chunks might use to reference \
this one
  - If this is the Redis setup guide → query: "Redis setup", "Redis configuration"

**5. Synonym/variant expansion**: Generate alternate phrasings for key concepts
  - "Kubernetes" + "k8s"
  - "authentication" + "auth" + "login"

## Query Format
- Prefer specific terms over generic ones
- Each query should target a *different* potential related chunk
- If the chunk is boilerplate, set confidence to "low"

Return JSON with:
- keywords: Distinctive terms from this chunk likely to appear in related chunks
- confidence: "low" | "mid" | "high"
- queries: ["q1", "q2", ...] - diverse queries targeting different relationships
"""

RELATED_CHUNK_USER_TEMPLATE = """\
Generate search queries based on this chunk to find other relevant chunks:

<context_before>
{prev_chunk_preview}
</context_before>

<main_chunk>
{chunk_content}
</main_chunk>

<context_after>
{next_chunk_preview}
</context_after>

The before/after context is provided only as additional context. \
Queries should target content from the main chunk only.

Return JSON with keys: keywords, confidence, queries
"""


# ---------------------------------------------------------------------------
# StructuralChunkLinker
# ---------------------------------------------------------------------------


class StructuralChunkLinker:
    """Chunk linking via structural metadata + deterministic BM25 enrichment.

    Uses ``CorpusCapabilities`` to detect available metadata, then
    ``AnchorSelector`` to pick secondaries based on document structure
    (co-location, cross-document, sequential).  BM25 enrichment queries
    are generated deterministically from chunk metadata.
    """

    def __init__(
        self,
        source: Any,
        capabilities: CorpusCapabilities | None = None,
        type_distribution: dict[str, float] | None = None,
        target_hop_counts: dict[str, int] | None = None,
        bm25_enrichment_queries: int = 3,
        bm25_enrichment_top_k: int = 5,
        max_related_refs: int = 3,
        entity_extraction: EntityExtractionConfig | None = None,
    ) -> None:
        self.source = source
        self.capabilities = capabilities
        self.type_distribution = type_distribution
        self.target_hop_counts = target_hop_counts
        self.bm25_enrichment_queries = bm25_enrichment_queries
        self.bm25_enrichment_top_k = bm25_enrichment_top_k
        self.max_related_refs = max_related_refs
        self.entity_extraction = entity_extraction
        self._selector: AnchorSelector | None = None

    def _ensure_selector(self, corpus_pool: list[Any]) -> AnchorSelector:
        if self._selector is None:
            caps = self.capabilities or CorpusCapabilities.detect(corpus_pool)
            self._selector = AnchorSelector(
                caps,
                type_distribution=self.type_distribution,
                target_hop_counts=self.target_hop_counts,
            )
        return self._selector

    def link(
        self,
        primary_chunk: Any,
        *,
        target_qa_type: str | None = None,
        target_hop_count: int | None = None,
        corpus_pool: list[Any] | None = None,
    ) -> AnchorBundle:
        pool = corpus_pool or []
        selector = self._ensure_selector(pool)
        if self.entity_extraction is not None:
            queries = generate_bm25_queries_from_extraction(
                primary_chunk, self.entity_extraction, self.bm25_enrichment_queries
            ) or generate_bm25_queries(primary_chunk, self.bm25_enrichment_queries)
        else:
            queries = generate_bm25_queries(primary_chunk, self.bm25_enrichment_queries)
        bundle = select_anchor_bundle_with_enrichment(
            selector=selector,
            primary_chunk=primary_chunk,
            corpus_pool=pool,
            source=self.source,
            bm25_enrichment_queries=self.bm25_enrichment_queries,
            bm25_enrichment_top_k=self.bm25_enrichment_top_k,
            max_related_refs=self.max_related_refs,
            qa_type=target_qa_type,
            prebuilt_queries=queries,
        )
        if isinstance(bundle, tuple):
            bundle = bundle[0]
        if target_hop_count is not None and target_hop_count > 0:
            bundle.target_hop_count = int(target_hop_count)
        bundle.connecting_queries = queries
        return bundle


# ---------------------------------------------------------------------------
# LLMGuidedChunkLinker
# ---------------------------------------------------------------------------


class LLMGuidedChunkLinker:
    """Chunk linking via LLM-generated BM25 queries.

    An LLM call generates diverse search queries from the primary chunk.
    Those queries are run through ``source.search_related()`` to find
    related chunks.  Results are packaged as an ``AnchorBundle``.
    """

    def __init__(
        self,
        source: Any,
        client: Any,
        model: str,
        system_prompt: str = RELATED_CHUNK_SYSTEM_PROMPT,
        user_template: str = RELATED_CHUNK_USER_TEMPLATE,
        response_parser: Callable[[str], tuple[str, list[str]]] = parse_related_queries_response,
        top_k_bm25: int = 5,
        top_related_chunks: int = 3,
        context_preview_chars: int = 200,
    ) -> None:
        self.source = source
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.response_parser = response_parser
        self.top_k_bm25 = top_k_bm25
        self.top_related_chunks = top_related_chunks
        self.context_preview_chars = context_preview_chars

    def link(
        self,
        primary_chunk: Any,
        *,
        target_qa_type: str | None = None,
        target_hop_count: int | None = None,
        corpus_pool: list[Any] | None = None,
    ) -> AnchorBundle:
        ctx = self.source.get_chunk_with_context(
            primary_chunk, max_chars=self.context_preview_chars
        )
        prompt = render_template(self.user_template, ctx)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        response_text = completion.choices[0].message.content or ""
        confidence, queries = self.response_parser(response_text)

        if confidence.lower() == "low" or not queries:
            return AnchorBundle(
                primary_chunk=primary_chunk,
                secondary_chunks=[],
                target_qa_type=target_qa_type or "multi_hop",
                target_hop_count=target_hop_count or 2,
                structural_hints={"llm_confidence": confidence},
                connecting_queries=queries,
            )

        search_results = self.source.search_related(primary_chunk, queries, top_k=self.top_k_bm25)
        secondary_chunks = [r["chunk"] for r in search_results[: self.top_related_chunks]]

        return AnchorBundle(
            primary_chunk=primary_chunk,
            secondary_chunks=secondary_chunks,
            target_qa_type=target_qa_type or "multi_hop",
            target_hop_count=target_hop_count or 2,
            structural_hints={
                "llm_confidence": confidence,
                "bm25_related": secondary_chunks,
            },
            connecting_queries=queries,
        )


# ---------------------------------------------------------------------------
# AdaptiveChunkLinker
# ---------------------------------------------------------------------------


class AdaptiveChunkLinker:
    """Composes multiple ``ChunkLinker`` implementations, selecting by condition.

    Each entry is a ``(condition, linker)`` pair.  On ``link()``, the first
    linker whose condition returns ``True`` for the detected
    ``CorpusCapabilities`` is used.

    Use the ``default()`` factory to get a sensible configuration that picks
    ``StructuralChunkLinker`` when rich metadata is available and falls back
    to ``LLMGuidedChunkLinker`` otherwise.
    """

    def __init__(
        self,
        linkers: list[tuple[Callable[[CorpusCapabilities], bool], ChunkLinker]],
        capabilities: CorpusCapabilities | None = None,
    ) -> None:
        if not linkers:
            raise ValueError("At least one (condition, linker) pair is required.")
        self.linkers = linkers
        self._capabilities = capabilities

    def _detect_capabilities(self, corpus_pool: list[Any] | None) -> CorpusCapabilities:
        if self._capabilities is None:
            self._capabilities = CorpusCapabilities.detect(corpus_pool or [])
            logger.info("AdaptiveChunkLinker: %s", self._capabilities.describe())
        return self._capabilities

    def link(
        self,
        primary_chunk: Any,
        *,
        target_qa_type: str | None = None,
        target_hop_count: int | None = None,
        corpus_pool: list[Any] | None = None,
    ) -> AnchorBundle:
        caps = self._detect_capabilities(corpus_pool)
        for condition, linker in self.linkers:
            if condition(caps):
                return linker.link(
                    primary_chunk,
                    target_qa_type=target_qa_type,
                    target_hop_count=target_hop_count,
                    corpus_pool=corpus_pool,
                )
        # Fallback to last linker if no condition matched
        return self.linkers[-1][1].link(
            primary_chunk,
            target_qa_type=target_qa_type,
            target_hop_count=target_hop_count,
            corpus_pool=corpus_pool,
        )

    @classmethod
    def default(
        cls,
        source: Any,
        client: Any | None = None,
        model: str = "",
        *,
        system_prompt: str = RELATED_CHUNK_SYSTEM_PROMPT,
        user_template: str = RELATED_CHUNK_USER_TEMPLATE,
        capabilities: CorpusCapabilities | None = None,
        type_distribution: dict[str, float] | None = None,
        target_hop_counts: dict[str, int] | None = None,
        bm25_enrichment_queries: int = 3,
        bm25_enrichment_top_k: int = 5,
        top_k_bm25: int = 5,
        top_related_chunks: int = 3,
        context_preview_chars: int = 200,
    ) -> AdaptiveChunkLinker:
        """Build a default adaptive linker.

        Uses ``StructuralChunkLinker`` when corpus has document IDs,
        falls back to ``LLMGuidedChunkLinker`` (if client provided)
        or deterministic BM25 otherwise.
        """
        entries: list[tuple[Callable[[CorpusCapabilities], bool], ChunkLinker]] = []

        structural = StructuralChunkLinker(
            source=source,
            capabilities=capabilities,
            type_distribution=type_distribution,
            target_hop_counts=target_hop_counts,
            bm25_enrichment_queries=bm25_enrichment_queries,
            bm25_enrichment_top_k=bm25_enrichment_top_k,
        )
        entries.append((lambda caps: caps.has_document_ids, structural))

        if client is not None and model:
            llm_guided = LLMGuidedChunkLinker(
                source=source,
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_template=user_template,
                top_k_bm25=top_k_bm25,
                top_related_chunks=top_related_chunks,
                context_preview_chars=context_preview_chars,
            )
            entries.append((lambda _caps: True, llm_guided))
        else:
            # No LLM client — structural linker is the only option
            entries.append((lambda _caps: True, structural))

        return cls(linkers=entries, capabilities=capabilities)
