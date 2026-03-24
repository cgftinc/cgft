"""ChunkSource protocol — the interface all corpus backends must implement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from cgft.corpus.search_schema.search_types import (
    FilterPredicate,
    HybridOptions,
    SearchCapabilities,
    SearchMode,
    SearchSpec,
)

if TYPE_CHECKING:
    from cgft.chunkers.models import Chunk, ChunkCollection



@runtime_checkable
class ChunkSource(Protocol):
    """Protocol for corpus backends that supply chunks for QA generation.

    Abstracts away how chunks are stored and searched, allowing pipeline code
    to generalize across local storage + Corpora, Turbopuffer, or any other backend.

    Implementations:
        CorporaChunkSource: chunks stored locally + uploaded to Corpora API for BM25 search
        TpufChunkSource: chunks stored and searched in Turbopuffer
    """

    def populate_from_folder(
        self,
        docs_path: str,
        min_chars: int = 1024,
        max_chars: int = 2048,
        overlap_chars: int = 128,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Chunk documents in a folder and upload to the backend corpus."""
        ...

    def populate_from_chunks(
        self,
        collection: "ChunkCollection",
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Upload a pre-built ChunkCollection to the Corpora API.
        """


    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        """Return n randomly sampled chunks, filtered by minimum character length."""
        ...

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        """Return a chunk with neighboring context as formatted strings.

        Returns:
            Dict with keys: chunk_content, prev_chunk_preview, next_chunk_preview
        """
        ...

    def get_top_level_chunks(self) -> list[Chunk]:
        """Return chunks from files at the top level of the document directory.

        Returns an empty list if file-structure awareness is not supported.
        """
        ...

    def search_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode: SearchMode | None = None,
        hybrid: HybridOptions | None = None,
    ) -> list[dict]:
        """Search for chunks related to source using BM25 queries.

        Returns:
            List of dicts with keys: chunk, queries, same_file, max_score.
            Sorted by relevance descending.
        """
        ...

    def search(self, spec: SearchSpec) -> list[Chunk]:
        """Search for chunks using a structured search spec."""
        ...

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter: FilterPredicate | None = None,
    ) -> list[Chunk]:
        """Search for chunks with a text query and optional filter."""
        ...

    def get_search_capabilities(self) -> SearchCapabilities:
        """Return search capabilities for this backend."""
        ...
