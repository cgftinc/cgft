from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CorpusCapabilities:
    """
    Describes which metadata fields are available in the corpus.
    Detected once at pipeline initialization and threaded through
    downstream components to enable graceful degradation.
    """

    has_document_ids: bool
    has_section_headers: bool
    has_sequential_links: bool

    @property
    def can_detect_document_boundaries(self) -> bool:
        """Enables cross-document multi-hop targeting and co-location detection."""
        return self.has_document_ids

    @property
    def can_target_sequential_reasoning(self) -> bool:
        """Enables procedural/cause-effect chain questions."""
        return self.has_sequential_links

    @property
    def can_target_section_synthesis(self) -> bool:
        """Enables synthesis questions that span named sections."""
        return self.has_section_headers

    @property
    def available_qa_types(self) -> list[str]:
        """
        Returns question types this corpus can reliably support.
        Types are removed when required metadata is unavailable.
        """
        types = ["lookup", "synthesis"]
        if self.has_document_ids:
            types.append("co_located_multi_hop")
            types.append("cross_document_multi_hop")
        else:
            types.append("multi_hop")

        if self.has_sequential_links:
            types.append("sequential_reasoning")
        return types

    @classmethod
    def detect(cls, sample_chunks: list[Any]) -> "CorpusCapabilities":
        """Auto-detect capabilities from a sample of chunks."""
        if not sample_chunks:
            return cls(
                has_document_ids=False,
                has_section_headers=False,
                has_sequential_links=False,
            )

        metadata: dict[str, Any] = {}
        for chunk in sample_chunks:
            if hasattr(chunk, "metadata_dict"):
                md = chunk.metadata_dict
                if md:
                    metadata = md
                    break
            elif hasattr(chunk, "metadata") and chunk.metadata:
                metadata = chunk.metadata
                break
            elif isinstance(chunk, dict) and chunk.get("metadata"):
                metadata = chunk["metadata"]
                break

        has_document_ids = any(
            key in metadata for key in ("document_id", "file_name", "file", "source", "doc_id")
        )
        has_section_headers = any(
            key in metadata for key in ("section_header", "header", "h1", "h2", "h3", "title", "heading")
        )
        has_sequential_links = any(
            key in metadata for key in ("prev_chunk_id", "next_chunk_id", "prev_id", "next_id")
        )

        return cls(
            has_document_ids=has_document_ids,
            has_section_headers=has_section_headers,
            has_sequential_links=has_sequential_links,
        )

    def describe(self) -> str:
        """Human-readable summary for logging."""
        available = self.available_qa_types
        return (
            f"CorpusCapabilities("
            f"doc_ids={self.has_document_ids}, "
            f"headers={self.has_section_headers}, "
            f"sequential={self.has_sequential_links}) "
            f"→ available types: {available}"
        )
