"""TpufChunkSource — ChunkSource implementation backed by Turbopuffer."""

from __future__ import annotations

import json
import random
from typing import Any, Callable

from tqdm.auto import tqdm

from .filter_mapper import to_turbopuffer_filters
from cgft.chunkers.inspector import ChunkInspector
from cgft.chunkers.markdown import MarkdownChunker
from cgft.chunkers.models import Chunk, ChunkCollection
from cgft.corpus.search_schema.search_exceptions import (
    InvalidSearchSpecError,
    UnsupportedSearchModeError,
)
from cgft.corpus.search_schema.search_types import (
    FilterPredicate,
    SearchCapabilities,
    SearchSpec,
    validate_search_spec_shape,
)


class TpufChunkSource:
    """ChunkSource backed by a Turbopuffer namespace.

    Chunks are stored in Turbopuffer with vector embeddings and BM25-enabled
    full-text search.

    Currently, there is no file-structure awareness: get_top_level_chunks()
    always returns [] and get_chunk_with_context() returns empty strings for
    prev/next context.

    Args:
        api_key: Turbopuffer API key
        namespace: Turbopuffer namespace name
        region: Turbopuffer region (default "aws-us-east-1")
        content_attr: List of Turbopuffer attribute names to use as the chunk's
            searchable text content. Defaults to ["content"]. For pre-existing namespaces,
            supply the BM25-indexed field(s), e.g. ["description"] or
            ["title", "content"].

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
    ) -> None:
        import turbopuffer

        self._ns = turbopuffer.Turbopuffer(api_key=api_key, region=region).namespace(namespace)
        self._fields: list[str] = content_attr if content_attr is not None else ["content"]
        self._search_capabilities: SearchCapabilities = {
            "backend": "turbopuffer",
            "modes": {"lexical"},
            "filter_ops": {
                "field": {"eq", "in", "gte", "lte"},
                "logical": {"and", "or", "not"},
            },
            "ranking": {"bm25"},
            "constraints": {"max_top_k": 10000, "vector_dimensions": None},
            "graph_expansion": False,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_rank_by(self, query: str) -> Any:
        """Build the rank_by argument for a BM25 Turbopuffer query."""
        if len(self._fields) == 1:
            return (self._fields[0], "BM25", query)
        return (
            "Sum",
            tuple(("Product", 1, (f, "BM25", query)) for f in self._fields),
        )

    def _row_content(self, row: Any) -> str:
        """Return the text content for a row, drawn from self._fields."""
        if len(self._fields) == 1:
            return str(getattr(row, self._fields[0], ""))
        return json.dumps({f: getattr(row, f, "") for f in self._fields}, default=str)

    def _row_to_chunk(self, row: Any) -> Chunk:
        """Convert a Turbopuffer row to a Chunk.

        Stores the Turbopuffer row ID in metadata as '_tpuf_id' so that
        search_related can reliably identify and skip the source chunk.
        All non-vector, non-internal attributes are stored in metadata when
        accessible; otherwise falls back to just the content fields.
        """
        content = self._row_content(row)

        try:
            raw = vars(row)
            attrs: dict = {
                k: v for k, v in raw.items()
                if not k.startswith("$") and k not in ("id", "vector")
            }
        except TypeError:
            attrs = {f: getattr(row, f, "") for f in self._fields}

        attrs["_tpuf_id"] = row.id
        metadata = tuple(attrs.items())
        return Chunk(content=content, metadata=metadata)

    # ------------------------------------------------------------------
    # ChunkSource protocol
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
        """Chunk documents in a folder and upload them to Turbopuffer. Searchable document contents
        are stored in a "content" attribute, and other metadata will be stored as well.

        Args:
            docs_path: Path to the folder containing documents to chunk
            embed_fn: Optional callable that maps list[str] → list[list[float]].
                      If provided, vector embeddings are stored alongside the text
                      (enables vector/hybrid search). If omitted, only BM25
                      full-text search is available.
            min_chars: Minimum characters per chunk (default 1024)
            max_chars: Maximum characters per chunk (default 2048)
            overlap_chars: Character overlap between adjacent chunks (default 128)
            file_extensions: File types to process (default [".md", ".mdx"])
            batch_size: Number of chunks per upload batch (default 100)
            show_summary: Print chunking summary and upload progress (default True)
        """
        if file_extensions is None:
            file_extensions = [".md", ".mdx"]

        if show_summary:
            print(f"Chunking documents from {docs_path}...")

        chunker = MarkdownChunker(min_char=min_chars, max_char=max_chars, chunk_overlap=overlap_chars)
        collection = chunker.chunk_folder(docs_path, file_extensions=file_extensions)

        if show_summary:
            inspector = ChunkInspector(collection)
            inspector.summary(max_depth=3, max_files_per_folder=4)
        self.populate_from_chunks(
            collection=collection,
            embed_fn=embed_fn,
            batch_size=batch_size,
            show_summary=show_summary,
        )

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
            embed_fn: Optional callable that maps list[str] -> list[list[float]].
                      If provided, vector embeddings are stored alongside text.
            batch_size: Number of chunks per upload batch (default 100).
            show_summary: Print upload progress (default True).
        """
        all_chunks = list(collection)

        if show_summary:
            print(f"\nUploading {len(all_chunks)} chunks to Turbopuffer namespace...")

        for batch_start in tqdm(
            range(0, len(all_chunks), batch_size),
            desc="Uploading batches",
            disable=not show_summary,
        ):
            batch = all_chunks[batch_start : batch_start + batch_size]

            # Build rows without vectors first; "content" is the BM25-indexed field
            upsert_rows = [
                {
                    "id": batch_start + i + 1,  # 1-indexed, stable across re-runs
                    "content": chunk.content,
                    "file_path": chunk.get_metadata("file", ""),
                    "h1": chunk.get_metadata("h1", ""),
                    "h2": chunk.get_metadata("h2", ""),
                    "h3": chunk.get_metadata("h3", ""),
                    "chunk_index": chunk.get_metadata("index", 0),
                    "chunk_hash": chunk.hash,
                    "char_count": len(chunk),
                }
                for i, chunk in enumerate(batch)
            ]

            write_kwargs: dict = {
                "upsert_rows": upsert_rows,
                "schema": {"content": {"type": "string", "full_text_search": True}},
            }

            # Inject vectors and add distance metric if an embedding function was provided
            if embed_fn:
                for row, v in zip(upsert_rows, embed_fn([chunk.content for chunk in batch])):
                    row["vector"] = v
                write_kwargs["distance_metric"] = "cosine_distance"
            self._ns.write(**write_kwargs)

        if show_summary:
            print(f"\nUpload complete! {len(all_chunks)} chunks written to namespace.")

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        """Return n randomly sampled chunks, optionally filtered by minimum length.
        """
        
        # Turbopuffer has no native random sampling, so we paginate all IDs first,
        # then sample locally and fetch only the selected rows.
        page_size = 1000
        all_ids: list[int] = []
        last_id = 0

        while True:
            result = self._ns.query(
                rank_by=["id", "asc"],
                filters=["id", "Gt", last_id],
                top_k=page_size,
            )
            rows = result.rows
            if not rows:
                break
            all_ids.extend(row.id for row in rows)
            last_id = rows[-1].id
            if len(rows) < page_size:
                break

        if not all_ids:
            return []

        # if no min_chars filter, we can sample directly from all IDs and fetch just those rows
        if min_chars == 0:
            sampled_ids = random.sample(all_ids, min(n, len(all_ids)))
            result = self._ns.query(
                rank_by=["id", "asc"],
                filters=["id", "In", sampled_ids],
                top_k=len(sampled_ids),
                include_attributes=True,
            )
            return [self._row_to_chunk(row) for row in result.rows]

        # if min chars is provided, fetch in batches until n chunks pass
        pool = list(all_ids)
        random.shuffle(pool)
        collected: list[Chunk] = []
        batch_size = max(n * 2, 50)
        pos = 0

        while pos < len(pool) and len(collected) < n:
            batch_ids = pool[pos : pos + batch_size]
            pos += batch_size

            result = self._ns.query(
                rank_by=["id", "asc"],
                filters=["id", "In", batch_ids],
                top_k=len(batch_ids),
                include_attributes=True,
            )

            for row in result.rows:
                chunk = self._row_to_chunk(row)
                if len(chunk.content) >= min_chars:
                    collected.append(chunk)
                    if len(collected) == n:
                        break

        return collected

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        """Return chunk with empty prev/next context (no file-structure awareness).

        Returns:
            Dict with keys: chunk_content, prev_chunk_preview, next_chunk_preview.
            prev/next are always empty strings for Turbopuffer.
        """
        return {
            "chunk_content": chunk.chunk_str(),
            "prev_chunk_preview": "",
            "next_chunk_preview": "",
        }

    def get_top_level_chunks(self) -> list[Chunk]:
        """Returns empty list — Turbopuffer has no file-structure awareness."""
        return []

    def search_related(self, source: Chunk, queries: list[str], top_k: int = 5) -> list[dict]:
        """Search for chunks related to source using BM25 queries via Turbopuffer.

        Runs each query, deduplicates results by row ID, skips the source chunk
        (matched by its '_tpuf_id' metadata if available, otherwise by content),
        and aggregates scores across queries.

        No adjacent-chunk filtering — Turbopuffer has no guaranteed file-structure
        attributes at this stage.

        Returns:
            List of dicts sorted by relevance (num matching queries DESC,
            max BM25 score DESC), each containing: chunk, queries, same_file, max_score
        """
        source_tpuf_id = source.get_metadata("_tpuf_id")
        related_map: dict[int, dict] = {}

        for query in queries:
            result = self._ns.query(
                rank_by=self._build_rank_by(query),
                top_k=top_k,
                include_attributes=True,
            )
            for row in result.rows:
                # Skip source chunk — prefer ID comparison, fall back to content match
                if source_tpuf_id is not None:
                    if row.id == source_tpuf_id:
                        continue
                else:
                    if self._row_content(row) == source.content:
                        continue

                score = getattr(row, "$dist", 0.0) or 0.0

                if row.id not in related_map:
                    related_map[row.id] = {
                        "chunk": self._row_to_chunk(row),
                        "queries": [],
                        "same_file": False,  # no file-structure awareness
                        "max_score": score,
                    }
                else:
                    related_map[row.id]["max_score"] = max(
                        related_map[row.id]["max_score"], score
                    )

                related_map[row.id]["queries"].append(query)

        return sorted(
            related_map.values(),
            key=lambda x: (len(x["queries"]), x["max_score"]),
            reverse=True,
        )

    def search(self, spec: SearchSpec) -> list[Chunk]:
        """Search chunks using a structured search spec."""
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

        query_kwargs: dict[str, Any] = {
            "rank_by": self._build_rank_by(str(spec.get("text_query") or "")),
            "top_k": int(spec.get("top_k", 10)),
            "include_attributes": True,
        }
        translated_filters = to_turbopuffer_filters(spec.get("filter"), self._search_capabilities)
        if translated_filters is not None:
            query_kwargs["filters"] = translated_filters

        result = self._ns.query(**query_kwargs)
        return [self._row_to_chunk(row) for row in result.rows]

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter: FilterPredicate | None = None,
    ) -> list[Chunk]:
        """Search chunks with a text query and optional filter."""
        return self.search(
            SearchSpec(mode="lexical", text_query=text_query, top_k=top_k, filter=filter)
        )

    def get_search_capabilities(self) -> SearchCapabilities:
        """Return search capabilities for Turbopuffer backend."""
        return self._search_capabilities
