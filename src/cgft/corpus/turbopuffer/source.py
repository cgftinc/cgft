"""TpufChunkSource — ChunkSource implementation backed by Turbopuffer."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from tqdm.auto import tqdm

from .filter_mapper import to_turbopuffer_filters
from cgft.chunkers.models import Chunk, ChunkCollection
from cgft.corpus.search_schema.search_exceptions import (
    InvalidSearchSpecError,
    UnsupportedSearchModeError,
)
from cgft.corpus.search_schema.search_types import (
    FilterPredicate,
    HybridOptions,
    SearchCapabilities,
    SearchMode,
    SearchSpec,
    validate_search_spec_shape,
)

from .namespace import TpufNamespace
from .files import FileAwareness
from .filter_mapper import to_turbopuffer_filters


class TpufChunkSource:
    """ChunkSource backed by a Turbopuffer namespace.

    Chunks are stored in Turbopuffer with vector embeddings and BM25-enabled
    full-text search.

    File-structure awareness (neighboring chunk context, top-level chunk
    retrieval, adjacent-chunk filtering in search_related) is supported when
    the namespace contains ``file_path`` and ``chunk_index`` attributes.
    These are written automatically by ``populate_from_folder`` /
    ``populate_from_chunks``.  For pre-existing namespaces that lack these
    fields, all file-aware methods gracefully degrade to their previous
    stub behavior (empty context, empty top-level list, no neighbor
    filtering).

    Args:
        api_key: Turbopuffer API key
        namespace: Turbopuffer namespace name
        region: Turbopuffer region (default "aws-us-east-1")
        content_attr: List of Turbopuffer attribute names to use as the chunk's
            searchable text content. Defaults to ["content"]. For pre-existing
            namespaces, supply the BM25-indexed field(s), e.g. ["description"]
            or ["title", "content"].

    Example:
        >>> # Managed docs — populate from folder
        >>> source = TpufChunkSource(api_key="tpuf_...", namespace="my-docs")
        >>> source.populate_from_folder("./docs", embed_fn=my_embed_fn)
        >>> chunks = source.sample_chunks(n=10, min_chars=400)

        >>> # Pre-existing namespace with known BM25-indexed fields
        >>> source = TpufChunkSource(
        ...     api_key="tpuf_...",
        ...     namespace="product-catalog",
        ...     content_attr=["description"],
        ... )
    """

    def __init__(
        self,
        api_key: str,
        namespace: str,
        region: str = "aws-us-east-1",
        content_attr: list[str] | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        distance_metric: str = "cosine_distance",
    ) -> None:
        self._client = TpufNamespace(
            api_key=api_key,
            namespace=namespace,
            region=region,
            content_attr=content_attr,
            embed_fn=embed_fn,
            distance_metric=distance_metric,
        )
        self._files = FileAwareness(self._client)

        modes: set[SearchMode] = {"lexical"}
        ranking: set[str] = {"bm25"}
        if embed_fn is not None:
            modes |= {"vector", "hybrid"}
            ranking.add("cosine")

        self._search_capabilities: SearchCapabilities = {
            "backend": "turbopuffer",
            "modes": modes,
            "filter_ops": {
                "field": {"eq", "in", "gte", "lte"},
                "logical": {"and", "or", "not"},
            },
            "ranking": ranking,
            "constraints": {"max_top_k": 10000, "vector_dimensions": None},
            "graph_expansion": False,
        }

    # ------------------------------------------------------------------
    # Populate
    # ------------------------------------------------------------------

    def populate_from_folder(
        self,
        docs_path: str,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        min_chars: int = 1024,
        max_chars: int = 2048,
        overlap_chars: int = 128,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Chunk documents in a folder and upload them to Turbopuffer.

        Args:
            docs_path: Path to the folder containing documents to chunk
            embed_fn: Optional embedding function for vector/hybrid search
            min_chars: Minimum characters per chunk (default 1024)
            max_chars: Maximum characters per chunk (default 2048)
            overlap_chars: Character overlap between adjacent chunks
            file_extensions: File types to process (default [".md", ".mdx"])
            batch_size: Number of chunks per upload batch (default 100)
            show_summary: Print chunking summary and upload progress
        """
        from cgft.chunkers.inspector import ChunkInspector
        from cgft.chunkers.markdown import MarkdownChunker

        if file_extensions is None:
            file_extensions = [".md", ".mdx"]

        if show_summary:
            print(f"Chunking documents from {docs_path}...")

        chunker = MarkdownChunker(
            min_char=min_chars,
            max_char=max_chars,
            chunk_overlap=overlap_chars,
        )
        collection = chunker.chunk_folder(
            docs_path, file_extensions=file_extensions
        )

        if show_summary:
            inspector = ChunkInspector(collection)
            inspector.summary(max_depth=3, max_files_per_folder=4)

        self.populate_from_chunks(
            collection=collection,
            embed_fn=embed_fn,
            batch_size=batch_size,
            show_summary=show_summary,
        )
        self._files.invalidate()

    def populate_from_chunks(
        self,
        collection: ChunkCollection,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Upload a pre-built ChunkCollection to Turbopuffer.

        Args:
            collection: ChunkCollection produced by any chunker.
            embed_fn: Optional embedding function for vector/hybrid search.
            batch_size: Number of chunks per upload batch (default 100).
            show_summary: Print upload progress (default True).
        """
        self._client.upsert_chunks(
            collection,
            embed_fn=embed_fn,
            batch_size=batch_size,
            show_summary=show_summary,
        )
        self._files.invalidate()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        """Return n randomly sampled chunks, optionally filtered by
        minimum length.
        """
        all_ids = self._client.paginate_all_ids()
        if not all_ids:
            return []

        if min_chars == 0:
            sampled_ids = random.sample(all_ids, min(n, len(all_ids)))
            rows = self._client.query_rows_by_ids(sampled_ids)
            return [self._client.row_to_chunk(row) for row in rows]

        # With min_chars, fetch in batches until n chunks pass
        pool = list(all_ids)
        random.shuffle(pool)
        collected: list[Chunk] = []
        batch_size = max(n * 2, 50)
        pos = 0

        while pos < len(pool) and len(collected) < n:
            batch_ids = pool[pos : pos + batch_size]
            pos += batch_size
            rows = self._client.query_rows_by_ids(batch_ids)

            for row in rows:
                chunk = self._client.row_to_chunk(row)
                if len(chunk.content) >= min_chars:
                    collected.append(chunk)
                    if len(collected) == n:
                        break

        return collected

    # ------------------------------------------------------------------
    # Context & structure
    # ------------------------------------------------------------------

    def get_chunk_with_context(
        self, chunk: Chunk, max_chars: int = 200
    ) -> dict:
        """Return chunk with neighboring context from the same file.

        When ``file_path`` and ``chunk_index`` attributes exist, previous
        and next chunk previews are fetched.  Otherwise returns empty
        strings.
        """
        fallback = {
            "chunk_content": chunk.chunk_str(),
            "prev_chunk_preview": "",
            "next_chunk_preview": "",
        }

        if not self._files.check():
            return fallback

        file_path = self._files.chunk_file_path(chunk)
        idx = self._files.chunk_index(chunk)
        if not file_path or idx is None:
            return fallback

        file_chunks = self._files.get_file_chunks(file_path)
        if not file_chunks:
            return fallback

        position = None
        for i, fc in enumerate(file_chunks):
            if self._files.chunk_index(fc) == idx:
                position = i
                break

        if position is None:
            return fallback

        prev_preview = "(No previous chunk)"
        if position > 0:
            prev_preview = file_chunks[position - 1].chunk_str(
                max_chars=max_chars, truncate="leading"
            )

        next_preview = "(No next chunk)"
        if position < len(file_chunks) - 1:
            next_preview = file_chunks[position + 1].chunk_str(
                max_chars=max_chars, truncate="trailing"
            )

        return {
            "chunk_content": chunk.chunk_str(),
            "prev_chunk_preview": prev_preview,
            "next_chunk_preview": next_preview,
        }

    def get_top_level_chunks(self) -> list[Chunk]:
        """Return chunks from files at the shallowest directory depth.

        Returns an empty list when file-structure metadata is unavailable.
        """
        if not self._files.check():
            return []

        all_paths = self._files.get_all_file_paths()
        if not all_paths:
            return []

        path_depths = {p: p.count("/") for p in all_paths}
        min_depth = min(path_depths.values())
        top_level_paths = [
            p for p, d in path_depths.items() if d == min_depth
        ]

        chunks: list[Chunk] = []
        for path in top_level_paths:
            chunks.extend(self._files.get_file_chunks(path))
        return chunks

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_related(
        self, source: Chunk, queries: list[str], top_k: int = 5
    ) -> list[dict]:
        """Search for chunks related to *source* using BM25 queries.

        Deduplicates by row ID, skips the source chunk and adjacent
        neighbors when file-structure metadata is available.

        Returns:
            List of dicts sorted by relevance, each containing:
            chunk, queries, same_file, max_score.
        """
        source_tpuf_id = source.get_metadata("_tpuf_id")
        related_map: dict[int, dict] = {}

        file_aware = self._files.check()
        source_file = (
            self._files.chunk_file_path(source) if file_aware else None
        )
        source_idx = (
            self._files.chunk_index(source) if file_aware else None
        )

        # Batch-embed all queries when embed_fn is available
        vectors: list[list[float]] | None = None
        if self._client.embed_fn is not None:
            vectors = self._client.embed_fn(queries)

        for i, query in enumerate(queries):
            if vectors is not None:
                rank_by = self._client.build_hybrid_rank_by(
                    query, vectors[i], None
                )
            else:
                rank_by = self._client.build_bm25_rank_by(query)

            result = self._client.ns.query(
                rank_by=rank_by,
                top_k=top_k,
                include_attributes=True,
            )

            for row in result.rows:
                # Skip source chunk
                if source_tpuf_id is not None:
                    if row.id == source_tpuf_id:
                        continue
                else:
                    if self._client.row_content(row) == source.content:
                        continue

                result_chunk = self._client.row_to_chunk(row)

                # File-awareness: detect same-file, skip adjacent
                is_same_file = False
                if source_file is not None:
                    result_file = self._files.chunk_file_path(result_chunk)
                    is_same_file = result_file == source_file

                    if is_same_file and source_idx is not None:
                        result_idx = self._files.chunk_index(result_chunk)
                        if (
                            result_idx is not None
                            and abs(result_idx - source_idx) <= 1
                        ):
                            continue

                score = getattr(row, "$dist", 0.0) or 0.0

                if row.id not in related_map:
                    related_map[row.id] = {
                        "chunk": result_chunk,
                        "queries": [],
                        "same_file": is_same_file,
                        "max_score": score,
                    }
                else:
                    related_map[row.id]["max_score"] = max(
                        related_map[row.id]["max_score"], score
                    )

                related_map[row.id]["queries"].append(query)

        return sorted(
            related_map.values(),
            key=lambda x: (
                len(x["queries"]),
                not x["same_file"],
                x["max_score"],
            ),
            reverse=True,
        )

    def _build_query_kwargs(self, spec: SearchSpec) -> dict[str, Any]:
        """Validate spec and build turbopuffer query kwargs."""
        mode = spec.get("mode")
        supported_modes = set(self._search_capabilities.get("modes", set()))
        if mode not in supported_modes:
            raise UnsupportedSearchModeError(
                backend=str(self._search_capabilities.get("backend", "unknown")),
                mode=str(mode),
                supported_modes={str(m) for m in supported_modes},
            )

        shape_errors = validate_search_spec_shape(spec)
        if shape_errors:
            raise InvalidSearchSpecError(
                backend=str(self._search_capabilities.get("backend", "unknown")),
                message="; ".join(shape_errors),
                spec=spec,
            )

        text_query = str(spec.get("text_query") or "")
        vector_query: list[float] | None = spec.get("vector_query")
        hybrid_opts: HybridOptions | None = spec.get("hybrid")

        if mode == "lexical":
            rank_by = self._client.build_bm25_rank_by(text_query)
        elif mode == "vector":
            rank_by = self._client.build_vector_rank_by(vector_query)  # type: ignore[arg-type]
        else:  # hybrid
            rank_by = self._client.build_hybrid_rank_by(text_query, vector_query, hybrid_opts)  # type: ignore[arg-type]

        query_kwargs: dict[str, Any] = {
            "rank_by": rank_by,
            "top_k": int(spec.get("top_k", 10)),
            "include_attributes": True,
        }
        translated_filters = to_turbopuffer_filters(
            spec.get("filter"), self._search_capabilities
        )
        if translated_filters is not None:
            query_kwargs["filters"] = translated_filters

        return query_kwargs

    def search(self, spec: SearchSpec) -> list[Chunk]:
        """Search chunks using a structured search spec."""
        query_kwargs = self._build_query_kwargs(spec)
        result = self._client.ns.query(**query_kwargs)
        return [self._client.row_to_chunk(row) for row in result.rows]

    def search_content(self, spec: SearchSpec) -> list[str]:
        """Search and return content strings without Chunk construction.

        Cloudpickle-safe alternative to ``search()`` for use in remote envs
        where Pydantic models don't survive pickle roundtripping.
        """
        query_kwargs = self._build_query_kwargs(spec)
        result = self._client.ns.query(**query_kwargs)
        return [self._client.row_content(row) for row in result.rows]

    def embed_query(self, text: str) -> list[float] | None:
        """Return an embedding vector for *text*, or ``None``."""
        if self._client.embed_fn is None:
            return None
        return self._client.embed_fn([text])[0]

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter: FilterPredicate | None = None,
    ) -> list[Chunk]:
        """Search chunks with a text query and optional filter."""
        return self.search(
            SearchSpec(
                mode="lexical",
                text_query=text_query,
                top_k=top_k,
                filter=filter,
            )
        )

    def get_search_capabilities(self) -> SearchCapabilities:
        """Return search capabilities for Turbopuffer backend."""
        return self._search_capabilities
