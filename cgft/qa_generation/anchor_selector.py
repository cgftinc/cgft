from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .corpus_capabilities import CorpusCapabilities

if TYPE_CHECKING:
    from cgft.chunkers.models import Chunk


DEFAULT_TARGET_HOP_COUNTS: dict[str, int] = {
    "lookup": 1,
    "co_located_multi_hop": 2,
    "cross_document_multi_hop": 3,
    "multi_hop": 3,
    "sequential_reasoning": 2,
    "synthesis": 4,
}

DEFAULT_TYPE_DISTRIBUTION: dict[str, float] = {
    "lookup": 0.25,
    "co_located_multi_hop": 0.15,
    "cross_document_multi_hop": 0.25,
    "sequential_reasoning": 0.10,
    "synthesis": 0.25,
}

REDUCED_TYPE_DISTRIBUTION: dict[str, float] = {
    "lookup": 0.30,
    "multi_hop": 0.45,
    "synthesis": 0.25,
}


@dataclass
class AnchorBundle:
    """Selection intent used to drive multi-hop generation."""

    primary_chunk: "Chunk"
    secondary_chunks: list["Chunk"]
    target_qa_type: str
    target_hop_count: int
    structural_hints: dict = field(default_factory=dict)
    connecting_queries: list[str] = field(default_factory=list)


class AnchorSelector:
    """Select anchor chunks for each QA type based on corpus capabilities.

    Note:
        ``synthesis`` is intentionally anchor-first (no explicit secondary chunk).
        It works best when search-agent filtering is enabled downstream to enforce
        true multi-hop behavior. If search-agent filtering is disabled, reduce or
        remove ``synthesis`` from type distribution to avoid lookup-like outputs.
    """

    def __init__(
        self,
        capabilities: CorpusCapabilities,
        type_distribution: dict[str, float] | None = None,
        target_hop_counts: dict[str, int] | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.target_hop_counts = {
            **DEFAULT_TARGET_HOP_COUNTS,
            **(target_hop_counts or {}),
        }

        available = set(capabilities.available_qa_types)
        if type_distribution:
            dist = {key: value for key, value in type_distribution.items() if key in available}
        elif capabilities.has_document_ids:
            dist = {
                key: value for key, value in DEFAULT_TYPE_DISTRIBUTION.items() if key in available
            }
        else:
            dist = {
                key: value for key, value in REDUCED_TYPE_DISTRIBUTION.items() if key in available
            }

        if not dist:
            raise ValueError(
                "Anchor type distribution is empty after filtering by available corpus capabilities."
            )

        total = sum(dist.values())
        self.type_distribution = {key: value / total for key, value in dist.items()}

    def sample_qa_type(self) -> str:
        """Sample a question type according to configured distribution."""
        qa_types = list(self.type_distribution.keys())
        weights = [self.type_distribution[qa_type] for qa_type in qa_types]
        return random.choices(qa_types, weights=weights, k=1)[0]

    def select(
        self,
        primary_chunk: "Chunk",
        corpus_sample: list["Chunk"],
        qa_type: str | None = None,
    ) -> AnchorBundle:
        target_type = qa_type or self.sample_qa_type()
        selector_fn = {
            "lookup": self._select_lookup,
            "co_located_multi_hop": self._select_co_located,
            "cross_document_multi_hop": self._select_cross_document,
            "sequential_reasoning": self._select_sequential,
            "synthesis": self._select_synthesis,
            "multi_hop": self._select_multi_hop_fallback,
        }.get(target_type)

        if selector_fn is None:
            raise ValueError(
                f"Unknown qa_type '{target_type}'. "
                f"Available: {list(self.capabilities.available_qa_types)}"
            )

        return selector_fn(primary_chunk, corpus_sample)

    def _select_lookup(self, primary: "Chunk", corpus_sample: list["Chunk"]) -> AnchorBundle:
        return AnchorBundle(
            primary_chunk=primary,
            secondary_chunks=[],
            target_qa_type="lookup",
            target_hop_count=self.target_hop_counts["lookup"],
            structural_hints={},
        )

    def _select_co_located(self, primary: "Chunk", corpus_sample: list["Chunk"]) -> AnchorBundle:
        primary_doc_id = self._get_document_id(primary)
        candidates = [
            chunk
            for chunk in corpus_sample
            if self._get_document_id(chunk) == primary_doc_id and chunk.hash != primary.hash
        ]

        if not candidates:
            return self._select_multi_hop_fallback(primary, corpus_sample)

        if self.capabilities.has_section_headers:
            primary_section = self._get_section_header(primary)
            different_section = [
                chunk for chunk in candidates if self._get_section_header(chunk) != primary_section
            ]
            if different_section:
                candidates = different_section

        secondary = random.choice(candidates)
        return AnchorBundle(
            primary_chunk=primary,
            secondary_chunks=[secondary],
            target_qa_type="co_located_multi_hop",
            target_hop_count=self.target_hop_counts["co_located_multi_hop"],
            structural_hints={
                "same_document": True,
                "primary_section": self._get_section_header(primary),
                "secondary_section": self._get_section_header(secondary),
            },
        )

    def _select_cross_document(
        self, primary: "Chunk", corpus_sample: list["Chunk"]
    ) -> AnchorBundle:
        primary_doc_id = self._get_document_id(primary)
        candidates = [
            chunk
            for chunk in corpus_sample
            if self._get_document_id(chunk) != primary_doc_id and chunk.hash != primary.hash
        ]

        if not candidates:
            return self._select_multi_hop_fallback(primary, corpus_sample)

        secondary = random.choice(candidates)
        return AnchorBundle(
            primary_chunk=primary,
            secondary_chunks=[secondary],
            target_qa_type="cross_document_multi_hop",
            target_hop_count=self.target_hop_counts["cross_document_multi_hop"],
            structural_hints={
                "cross_boundary": True,
                "primary_doc": primary_doc_id,
                "secondary_doc": self._get_document_id(secondary),
            },
        )

    def _select_sequential(self, primary: "Chunk", corpus_sample: list["Chunk"]) -> AnchorBundle:
        meta = primary.metadata_dict if hasattr(primary, "metadata_dict") else primary.metadata
        next_id = meta.get("next_chunk_id") or meta.get("next_id")
        prev_id = meta.get("prev_chunk_id") or meta.get("prev_id")

        target_id = next_id or prev_id
        if target_id:
            adjacent = next((chunk for chunk in corpus_sample if chunk.hash == target_id), None)
            if adjacent:
                return AnchorBundle(
                    primary_chunk=primary,
                    secondary_chunks=[adjacent],
                    target_qa_type="sequential_reasoning",
                    target_hop_count=self.target_hop_counts["sequential_reasoning"],
                    structural_hints={
                        "sequential": True,
                        "direction": "next" if target_id == next_id else "prev",
                    },
                )

        return self._select_co_located(primary, corpus_sample)

    def _select_synthesis(self, primary: "Chunk", corpus_sample: list["Chunk"]) -> AnchorBundle:
        """Create a synthesis anchor.

        Synthesis starts with a single anchor and expects downstream retrieval and
        filtering to enforce bridge discovery + hop depth. Without search-agent
        filtering, synthesis can degrade toward lookup-style questions.
        """
        hints: dict[str, str | None] = {}
        if self.capabilities.has_section_headers:
            hints["anchor_section"] = self._get_section_header(primary)

        return AnchorBundle(
            primary_chunk=primary,
            secondary_chunks=[],
            target_qa_type="synthesis",
            target_hop_count=self.target_hop_counts["synthesis"],
            structural_hints=hints,
        )

    def _select_multi_hop_fallback(
        self, primary: "Chunk", corpus_sample: list["Chunk"]
    ) -> AnchorBundle:
        candidates = [chunk for chunk in corpus_sample if chunk.hash != primary.hash]
        secondary_chunks = [random.choice(candidates)] if candidates else []

        return AnchorBundle(
            primary_chunk=primary,
            secondary_chunks=secondary_chunks,
            target_qa_type="multi_hop",
            target_hop_count=self.target_hop_counts["multi_hop"],
            structural_hints={"fallback": True},
        )

    def _get_document_id(self, chunk: "Chunk") -> str | None:
        if not hasattr(chunk, "metadata"):
            return None
        meta = chunk.metadata_dict if hasattr(chunk, "metadata_dict") else chunk.metadata
        return (
            meta.get("document_id")
            or meta.get("file_name")
            or meta.get("file")
            or meta.get("source")
            or meta.get("doc_id")
        )

    def _get_section_header(self, chunk: "Chunk") -> str | None:
        if not hasattr(chunk, "metadata"):
            return None
        meta = chunk.metadata_dict if hasattr(chunk, "metadata_dict") else chunk.metadata
        return (
            meta.get("section_header")
            or meta.get("header")
            or meta.get("h1")
            or meta.get("h2")
            or meta.get("h3")
            or meta.get("title")
            or meta.get("heading")
        )
