"""ChromaChunkSource — ChunkSource implementation backed by ChromaDB.

Supports vector, lexical (BM25), and hybrid (RRF) search modes via
Chroma's Search API.  When the Search API is unavailable (older chromadb
or OSS-only builds), falls back to the legacy ``query()`` API with
vector-only search.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from tqdm.auto import tqdm

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

from .files import FileAwareness
from .filter_mapper import to_chroma_filters

# Sparse-key name used when setting up BM25 schema
_BM25_KEY = "bm25_embedding"

# Default RRF parameters — tunable via rrf_k / rrf_oversample constructor args
_DEFAULT_RRF_K = 60
_DEFAULT_RRF_OVERSAMPLE = 20
_DEFAULT_RRF_MAX_CANDIDATES = 200


def _has_search_api() -> bool:
    """Return True when the chromadb package exposes the Search API."""
    try:
        from chromadb import Knn, Search  # noqa: F401

        return True
    except ImportError:
        return False


class ChromaChunkSource:
    """ChunkSource backed by a ChromaDB collection.

    Supports three retrieval modes:

    * **vector** — dense embedding similarity (always available).
    * **lexical** — native BM25 via Chroma's sparse-vector index.
    * **hybrid** — Reciprocal Rank Fusion of dense + sparse rankings.

    Lexical and hybrid modes require ``chromadb ≥ 1.0`` with the
    ``Search`` API and a client-server connection (sparse indexing is
    not available on local/in-memory Chroma).  When unavailable the
    source gracefully degrades to vector-only.

    The client is lazily initialized on first use so that the source
    survives ``pickle`` round-trips (only connection parameters are
    stored).  For remote training environments, only **client-server**
    mode (``host``/``port``) is safe — local/in-memory collections
    cannot be transferred to a different process.

    Args:
        collection_name: Name of the Chroma collection.
        host: Chroma server hostname for client-server mode.
        port: Chroma server port (default 8000).
        path: Directory for persistent-local mode (mutually exclusive
            with ``host``).  When neither ``host`` nor ``path`` is given,
            an ephemeral in-memory client is created.
        embed_fn: Optional embedding callable ``list[str] → list[list[float]]``.
            When provided, Chroma's default dense embedding function is
            replaced.
        content_attr: List of metadata field names to treat as searchable
            text content.  Defaults to ``["content"]``.  Useful for
            pre-existing collections where the text lives in a different
            field (analogous to ``content_attr`` on ``TpufChunkSource``).
        distance_metric: Distance function for the dense index.  One of
            ``"cosine"``, ``"l2"``, ``"ip"`` (default ``"cosine"``).
            Applied both with and without BM25 — controls the HNSW index.
        enable_bm25: Create a sparse BM25 index on the collection for
            lexical / hybrid search (default ``True``).  Only takes
            effect when client-server mode and the Search API are both
            available.
        rrf_k: RRF smoothing constant for hybrid search (default 60).
            Higher values flatten rank differences.
        rrf_oversample: Multiplier for candidate pool in each RRF leg,
            capped at ``rrf_max_candidates`` (default 20×).
        rrf_max_candidates: Hard cap on per-leg candidate count
            (default 200).

    Example:
        >>> source = ChromaChunkSource(
        ...     collection_name="my-docs",
        ...     host="chroma.internal",
        ... )
        >>> source.populate_from_folder("./docs")
        >>> chunks = source.sample_chunks(n=10, min_chars=400)
    """

    def __init__(
        self,
        collection_name: str,
        host: str | None = None,
        port: int = 8000,
        path: str | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        content_attr: list[str] | None = None,
        distance_metric: str = "cosine",
        enable_bm25: bool = True,
        rrf_k: int = _DEFAULT_RRF_K,
        rrf_oversample: int = _DEFAULT_RRF_OVERSAMPLE,
        rrf_max_candidates: int = _DEFAULT_RRF_MAX_CANDIDATES,
    ) -> None:
        self._collection_name = collection_name
        self._host = host
        self._port = port
        self._path = path
        self._embed_fn = embed_fn
        self._content_attr: list[str] = content_attr if content_attr is not None else ["content"]
        self._distance_metric = distance_metric
        self._enable_bm25 = enable_bm25
        self._rrf_k = rrf_k
        self._rrf_oversample = rrf_oversample
        self._rrf_max_candidates = rrf_max_candidates

        # Lazily initialized
        self._client: Any = None
        self._collection: Any = None
        self._total_count: int | None = None
        self._embed_dim: int | None = None  # validated on first embed call

        # Detect Search API availability
        self._search_api = _has_search_api()

        self._files = FileAwareness(self._get_collection)

        # BM25/hybrid require Search API + client-server (sparse indexing
        # is not available on local/in-memory Chroma).
        modes: set[SearchMode] = {"vector"}
        ranking: set[str] = {"cosine"}
        if self._search_api and enable_bm25 and (host is not None):
            modes |= {"lexical", "hybrid"}
            ranking.add("bm25")

        self._search_capabilities: SearchCapabilities = {
            "backend": "chroma",
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
    # Lazy client / collection
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return (and cache) the ChromaDB client."""
        if self._client is not None:
            return self._client

        import chromadb

        if self._host is not None:
            self._client = chromadb.HttpClient(host=self._host, port=self._port)
        elif self._path is not None:
            self._client = chromadb.PersistentClient(path=self._path)
        else:
            self._client = chromadb.Client()

        return self._client

    def _get_collection(self) -> Any:
        """Return (and cache) the Chroma collection, creating if needed."""
        if self._collection is not None:
            return self._collection

        client = self._get_client()

        kwargs: dict[str, Any] = {"name": self._collection_name}
        if self._embed_fn is not None:
            kwargs["embedding_function"] = _WrapEmbedFn(self._embed_fn)

        # BM25 schema requires client-server mode and a server that
        # supports sparse indexing.  Chroma does not allow schema +
        # metadata simultaneously, so distance_metric is only set via
        # metadata when no schema is used.
        created_with_schema = False
        if self._search_api and self._enable_bm25 and self.is_client_server:
            try:
                schema_kwargs = {**kwargs, "schema": self._build_schema()}
                self._collection = client.get_or_create_collection(
                    **schema_kwargs
                )
                created_with_schema = True
            except Exception:
                # Server doesn't support sparse indexing (e.g. older
                # version).  Downgrade to vector-only and retry below.
                self._search_capabilities["modes"] = {"vector"}
                self._search_capabilities["ranking"] = {"cosine"}

        if not created_with_schema:
            metadata: dict[str, str] = {}
            if self._distance_metric:
                metadata["hnsw:space"] = self._distance_metric
            if metadata:
                kwargs["metadata"] = metadata
            self._collection = client.get_or_create_collection(**kwargs)

        return self._collection

    def _build_schema(self) -> Any:
        """Build a Chroma Schema with a BM25 sparse-vector index."""
        from chromadb import Schema, SparseVectorIndexConfig
        from chromadb.utils.embedding_functions import (
            ChromaBm25EmbeddingFunction,
        )

        schema = Schema()
        schema = schema.create_index(
            key=_BM25_KEY,
            config=SparseVectorIndexConfig(
                embedding_function=ChromaBm25EmbeddingFunction(),
                source_key="#document",
                bm25=True,
            ),
        )
        return schema

    @property
    def is_client_server(self) -> bool:
        """True when configured for HTTP client-server mode."""
        return self._host is not None

    def __getstate__(self) -> dict:
        """Strip live client/collection for pickle safety.

        Connection params, config, and search capabilities are preserved.
        The client, collection, and file-awareness caches are rebuilt
        lazily after unpickling — this is safe because training envs
        require client-server mode where data persists on the server.
        """
        state = self.__dict__.copy()
        state["_client"] = None
        state["_collection"] = None
        state["_files"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # FileAwareness caches (file_cache, all_file_paths, awareness
        # flag) are rebuilt lazily on next access from the remote server.
        self._files = FileAwareness(self._get_collection)

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
        """Chunk documents in a folder and upload them to Chroma.

        Args:
            docs_path: Path to the folder containing documents to chunk.
            embed_fn: Ignored (kept for API parity). Use the constructor
                ``embed_fn`` or Chroma's default embedding.
            min_chars: Minimum characters per chunk (default 1024).
            max_chars: Maximum characters per chunk (default 2048).
            overlap_chars: Character overlap between adjacent chunks.
            file_extensions: File types to process (default [".md", ".mdx"]).
            batch_size: Number of chunks per upload batch (default 100).
            show_summary: Print chunking summary and upload progress.
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
        collection = chunker.chunk_folder(docs_path, file_extensions=file_extensions)

        if show_summary:
            inspector = ChunkInspector(collection)
            inspector.summary(max_depth=3, max_files_per_folder=4)

        self.populate_from_chunks(
            collection=collection,
            batch_size=batch_size,
            show_summary=show_summary,
        )

    def populate_from_chunks(
        self,
        collection: ChunkCollection,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        """Upload a pre-built ChunkCollection to ChromaDB.

        Args:
            collection: ChunkCollection produced by any chunker.
            batch_size: Number of chunks per upload batch (default 100).
            show_summary: Print upload progress (default True).
        """
        col = self._get_collection()
        all_chunks = list(collection)

        if show_summary:
            print(
                f"\nUploading {len(all_chunks)} chunks to Chroma"
                f" collection '{self._collection_name}'..."
            )

        for batch_start in tqdm(
            range(0, len(all_chunks), batch_size),
            desc="Uploading batches",
            disable=not show_summary,
        ):
            batch = all_chunks[batch_start : batch_start + batch_size]

            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []

            for chunk in batch:
                ids.append(chunk.hash)
                documents.append(chunk.content)
                metadatas.append(
                    {
                        "file_path": chunk.get_metadata("file", ""),
                        "h1": chunk.get_metadata("h1", ""),
                        "h2": chunk.get_metadata("h2", ""),
                        "h3": chunk.get_metadata("h3", ""),
                        "chunk_index": chunk.get_metadata("index", 0),
                        "chunk_hash": chunk.hash,
                        "char_count": len(chunk),
                    }
                )

            col.upsert(ids=ids, documents=documents, metadatas=metadatas)

        self._total_count = len(all_chunks)
        self._files.invalidate()

        if show_summary:
            print(
                f"\nUpload complete! {len(all_chunks)} chunks written to"
                f" collection '{self._collection_name}'."
            )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _get_total_count(self) -> int:
        """Return the total document count, caching after first call."""
        if self._total_count is None:
            self._total_count = self._get_collection().count()
        return self._total_count

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        """Return n randomly sampled chunks, optionally filtered by length."""
        total = self._get_total_count()
        if total == 0:
            return []

        col = self._get_collection()

        if min_chars == 0:
            sample_size = min(n, total)
            if sample_size == total:
                result = col.get(include=["documents", "metadatas"], limit=total)
                return _results_to_chunks(result, self._content_attr)

            indices = sorted(random.sample(range(total), sample_size))
            chunks: list[Chunk] = []
            for idx in indices:
                result = col.get(
                    include=["documents", "metadatas"],
                    limit=1,
                    offset=idx,
                )
                chunks.extend(_results_to_chunks(result, self._content_attr))
            return chunks

        # With min_chars: fetch in batches, filter locally
        collected: list[Chunk] = []
        all_offsets = list(range(total))
        random.shuffle(all_offsets)
        fetch_size = max(n * 2, 50)
        pos = 0

        while pos < len(all_offsets) and len(collected) < n:
            batch_offsets = sorted(all_offsets[pos : pos + fetch_size])
            pos += fetch_size

            for offset in batch_offsets:
                result = col.get(
                    include=["documents", "metadatas"],
                    limit=1,
                    offset=offset,
                )
                for chunk in _results_to_chunks(result, self._content_attr):
                    if len(chunk.content) >= min_chars:
                        collected.append(chunk)
                        if len(collected) == n:
                            return collected

        return collected

    # ------------------------------------------------------------------
    # Context & structure
    # ------------------------------------------------------------------

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        """Return chunk with neighboring context from the same file."""
        fallback: dict[str, str] = {
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
        """Return chunks from files at the shallowest directory depth."""
        if not self._files.check():
            return []

        all_paths = self._files.get_all_file_paths()
        if not all_paths:
            return []

        path_depths = {p: p.count("/") for p in all_paths}
        min_depth = min(path_depths.values())
        top_level_paths = [p for p, d in path_depths.items() if d == min_depth]

        chunks: list[Chunk] = []
        for path in top_level_paths:
            chunks.extend(self._files.get_file_chunks(path))
        return chunks

    # ------------------------------------------------------------------
    # Search — new Search API (lexical / vector / hybrid)
    # ------------------------------------------------------------------

    def _build_search_expr(self, spec: SearchSpec) -> Any:
        """Build a ``chromadb.Search`` expression from a SearchSpec.

        Uses ``Knn`` for dense and sparse rankings, ``Rrf`` for hybrid
        fusion.  Only called when the Search API is available.
        """
        from chromadb import K, Knn, Search

        mode = spec.get("mode")
        text_query = spec.get("text_query") or ""
        vector_query: list[float] | None = spec.get("vector_query")
        top_k = spec.get("top_k", 10)
        hybrid_opts: HybridOptions | None = spec.get("hybrid")

        search = Search()

        # Metadata filter — Search.where() accepts dicts directly
        where_filter = to_chroma_filters(spec.get("filter"), self._search_capabilities)
        if where_filter is not None:
            search = search.where(where_filter)

        # Build ranking expression
        if mode == "lexical":
            rank = Knn(query=text_query, key=_BM25_KEY)
        elif mode == "vector":
            if vector_query is not None:
                rank = Knn(query=vector_query)
            else:
                rank = Knn(query=text_query)
        else:  # hybrid
            from chromadb import Rrf

            opts = hybrid_opts or {}
            vector_weight = opts.get("vector_weight", 1.0)
            lexical_weight = opts.get("lexical_weight", 1.0)
            candidates = min(top_k * self._rrf_oversample, self._rrf_max_candidates)

            dense_rank = Knn(
                query=(vector_query if vector_query is not None else text_query),
                return_rank=True,
                limit=candidates,
            )
            sparse_rank = Knn(
                query=text_query,
                key=_BM25_KEY,
                return_rank=True,
                limit=candidates,
            )
            rank = Rrf(
                ranks=[dense_rank, sparse_rank],
                weights=[vector_weight, lexical_weight],
                k=self._rrf_k,
            )

        search = search.rank(rank).limit(top_k).select(K.DOCUMENT, K.METADATA, K.SCORE)
        return search

    def _search_via_api(self, spec: SearchSpec) -> Any:
        """Execute a search using the new Search API and return raw result."""
        col = self._get_collection()
        search_expr = self._build_search_expr(spec)
        return col.search(search_expr)

    @staticmethod
    def _search_rows(result: Any) -> list[dict]:
        """Extract rows from a Search API result.

        ``result.rows()`` returns ``list[list[SearchResultRow]]`` — one
        inner list per search payload.  We always pass a single search,
        so we take element ``[0]``.
        """
        if not hasattr(result, "rows"):
            return []
        payloads = result.rows()
        if not payloads:
            return []
        return list(payloads[0])

    def _search_results_to_chunks(self, result: Any) -> list[Chunk]:
        """Convert Search API results to Chunks."""
        fields = self._content_attr
        chunks: list[Chunk] = []
        for row in self._search_rows(result):
            doc = row.get("document", "")
            meta = row.get("metadata") or {}
            if isinstance(meta, dict):
                content = _extract_content(doc, meta, fields)
                chunks.append(Chunk(content=content, metadata=_clean_metadata(meta)))
            else:
                chunks.append(Chunk(content=doc))
        return chunks

    def _search_results_to_strings(self, result: Any) -> list[str]:
        """Convert Search API results to content strings."""
        fields = self._content_attr
        results: list[str] = []
        for row in self._search_rows(result):
            doc = row.get("document", "")
            meta = row.get("metadata") or {}
            if isinstance(meta, dict):
                results.append(_extract_content(doc, meta, fields))
            else:
                results.append(doc)
        return results

    def _search_results_to_scored_chunks(self, result: Any) -> list[tuple[Chunk, float]]:
        """Convert Search API results to (Chunk, score) pairs."""
        fields = self._content_attr
        pairs: list[tuple[Chunk, float]] = []
        for row in self._search_rows(result):
            doc = row.get("document", "")
            meta = row.get("metadata") or {}
            score = row.get("score", 0.0) or 0.0
            if isinstance(meta, dict):
                content = _extract_content(doc, meta, fields)
                chunk = Chunk(content=content, metadata=_clean_metadata(meta))
            else:
                chunk = Chunk(content=doc)
            pairs.append((chunk, float(score)))
        return pairs

    # ------------------------------------------------------------------
    # Search — legacy query() API (vector-only fallback)
    # ------------------------------------------------------------------

    def _build_query_kwargs(self, spec: SearchSpec) -> dict[str, Any]:
        """Build kwargs for the legacy ``collection.query()`` API."""
        vector_query: list[float] | None = spec.get("vector_query")
        top_k = spec.get("top_k", 10)

        query_kwargs: dict[str, Any] = {
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if vector_query is not None:
            query_kwargs["query_embeddings"] = [vector_query]
        else:
            query_kwargs["query_texts"] = [spec.get("text_query") or ""]

        where = to_chroma_filters(spec.get("filter"), self._search_capabilities)
        if where is not None:
            query_kwargs["where"] = where

        return query_kwargs

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> list[float] | None:
        """Return a dense embedding vector for *text*, or ``None``.

        Uses the custom ``embed_fn`` if provided.  Otherwise returns
        ``None`` — Chroma will auto-embed via ``query_texts``.

        On the first call with ``embed_fn``, validates that the returned
        dimension is consistent.  Raises ``ValueError`` on mismatch.
        """
        if self._embed_fn is None:
            return None
        vec = self._embed_fn([text])[0]
        self._validate_embed_dim(vec)
        return vec

    def _validate_embed_dim(self, vec: list[float]) -> None:
        """Check embedding dimension consistency on first call."""
        dim = len(vec)
        if self._embed_dim is None:
            self._embed_dim = dim
        elif dim != self._embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embed_dim}, "
                f"got {dim}. Check that your embed_fn produces vectors of "
                f"consistent size."
            )

    # ------------------------------------------------------------------
    # Public search interface
    # ------------------------------------------------------------------

    def _validate_spec(self, spec: SearchSpec) -> None:
        """Validate mode and shape, raising on error.

        Chroma can auto-embed text queries in vector mode, so we relax
        the standard shape validation that would require ``vector_query``
        — a ``text_query`` is sufficient.
        """
        mode = spec.get("mode")
        supported_modes = self._search_capabilities["modes"]
        if mode not in supported_modes:
            raise UnsupportedSearchModeError(
                backend="chroma",
                mode=mode or "None",
                supported_modes={m for m in supported_modes},
            )

        shape_errors = validate_search_spec_shape(spec)

        # Chroma auto-embeds text queries via its built-in embedding
        # function, so vector_query is never strictly required — a
        # text_query is sufficient for vector and hybrid modes.
        if mode in ("vector", "hybrid") and shape_errors:
            text_query = spec.get("text_query")
            has_text = isinstance(text_query, str) and bool(text_query.strip())
            shape_errors = [e for e in shape_errors if not (has_text and "vector_query" in e)]

        if shape_errors:
            raise InvalidSearchSpecError(
                backend="chroma",
                message="; ".join(shape_errors),
                spec=spec,
            )

    def search(self, spec: SearchSpec) -> list[Chunk]:
        """Search chunks using a structured search spec."""
        self._validate_spec(spec)
        mode = spec.get("mode")

        if self._search_api and mode in ("lexical", "hybrid"):
            result = self._search_via_api(spec)
            return self._search_results_to_chunks(result)

        # Vector mode (or fallback when Search API unavailable)
        col = self._get_collection()
        query_kwargs = self._build_query_kwargs(spec)
        result = col.query(**query_kwargs)
        return _query_results_to_chunks(result, self._content_attr)

    def search_content(self, spec: SearchSpec) -> list[str]:
        """Search and return content strings without Chunk construction.

        Cloudpickle-safe alternative to ``search()`` for use in remote
        envs where Pydantic models don't survive pickle roundtripping.
        """
        self._validate_spec(spec)
        mode = spec.get("mode")

        if self._search_api and mode in ("lexical", "hybrid"):
            result = self._search_via_api(spec)
            return self._search_results_to_strings(result)

        col = self._get_collection()
        query_kwargs = self._build_query_kwargs(spec)
        result = col.query(**query_kwargs)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        return [
            _extract_content(doc, meta, self._content_attr)
            for doc, meta in zip(docs, metas)
        ]

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter: FilterPredicate | None = None,
    ) -> list[Chunk]:
        """Search chunks with a text query and optional filter.

        Picks the best available mode: hybrid > lexical > vector.
        """
        modes = self._search_capabilities["modes"]
        if "hybrid" in modes:
            vector = self.embed_query(text_query)
            return self.search(
                SearchSpec(
                    mode="hybrid",
                    top_k=top_k,
                    text_query=text_query,
                    vector_query=vector,
                    filter=filter,
                )
            )
        if "lexical" in modes:
            return self.search(
                SearchSpec(
                    mode="lexical",
                    top_k=top_k,
                    text_query=text_query,
                    filter=filter,
                )
            )
        # Vector-only fallback
        vector = self.embed_query(text_query)
        return self.search(
            SearchSpec(
                mode="vector",
                top_k=top_k,
                text_query=text_query,
                vector_query=vector,
                filter=filter,
            )
        )

    def search_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode: str | None = None,
        hybrid: HybridOptions | None = None,
    ) -> list[dict]:
        """Search for chunks related to *source*.

        Uses the best available mode.  When the Search API is available,
        actual distance scores from the backend are used for ranking.
        Otherwise falls back to the legacy query API with Chroma distance
        scores (lower = closer, converted to higher = better).

        Deduplicates by document ID, skips the source chunk and adjacent
        neighbors when file-structure metadata is available.

        Returns:
            List of dicts sorted by relevance, each containing:
            chunk, queries, same_file, max_score.
        """
        source_hash = source.hash

        file_aware = self._files.check()
        source_file = self._files.chunk_file_path(source) if file_aware else None
        source_idx = self._files.chunk_index(source) if file_aware else None

        # Pick best mode
        modes = self._search_capabilities["modes"]
        use_hybrid = "hybrid" in modes
        use_lexical = "lexical" in modes

        # Batch-embed all queries when embed_fn available and vectors needed
        vectors: list[list[float]] | None = None
        if self._embed_fn is not None and (use_hybrid or not use_lexical):
            vectors = self._embed_fn(queries)

        related_map: dict[str, dict] = {}

        for i, query in enumerate(queries):
            if use_hybrid and vectors is not None:
                spec = SearchSpec(
                    mode="hybrid",
                    top_k=top_k,
                    text_query=query,
                    vector_query=vectors[i],
                )
            elif use_lexical:
                spec = SearchSpec(
                    mode="lexical",
                    top_k=top_k,
                    text_query=query,
                )
            else:
                spec = SearchSpec(
                    mode="vector",
                    top_k=top_k,
                    text_query=query,
                    vector_query=vectors[i] if vectors else None,
                )

            # Get scored results for proper ranking
            scored = self._search_with_scores(spec)

            for result_chunk, score in scored:
                doc_id = result_chunk.hash
                if doc_id == source_hash:
                    continue
                if result_chunk.content == source.content:
                    continue

                is_same_file = False
                if source_file is not None:
                    result_file = self._files.chunk_file_path(result_chunk)
                    is_same_file = result_file == source_file

                    if is_same_file and source_idx is not None:
                        result_idx = self._files.chunk_index(result_chunk)
                        if result_idx is not None and abs(result_idx - source_idx) <= 1:
                            continue

                if doc_id not in related_map:
                    related_map[doc_id] = {
                        "chunk": result_chunk,
                        "queries": [],
                        "same_file": is_same_file,
                        "max_score": score,
                    }
                else:
                    related_map[doc_id]["max_score"] = max(related_map[doc_id]["max_score"], score)

                related_map[doc_id]["queries"].append(query)

        return sorted(
            related_map.values(),
            key=lambda x: (
                len(x["queries"]),
                not x["same_file"],
                x["max_score"],
            ),
            reverse=True,
        )

    def _search_with_scores(self, spec: SearchSpec) -> list[tuple[Chunk, float]]:
        """Search and return (Chunk, score) pairs.

        Uses actual backend scores — Search API scores for lexical/hybrid,
        Chroma distances (converted: higher = better) for vector/legacy.
        """
        self._validate_spec(spec)
        mode = spec.get("mode")

        if self._search_api and mode in ("lexical", "hybrid"):
            result = self._search_via_api(spec)
            return self._search_results_to_scored_chunks(result)

        # Legacy query() API — distances (lower = closer)
        col = self._get_collection()
        query_kwargs = self._build_query_kwargs(spec)
        result = col.query(**query_kwargs)
        return _query_results_to_scored_chunks(result, self._content_attr)

    def get_search_capabilities(self) -> SearchCapabilities:
        """Return search capabilities for ChromaDB backend.

        When BM25 is requested on a client-server connection, triggers
        lazy collection initialization so that capabilities reflect what
        the server actually supports (sparse indexing may be unavailable
        on older servers, causing a downgrade to vector-only).
        """
        if (
            self._collection is None
            and self._enable_bm25
            and self.is_client_server
            and "lexical" in self._search_capabilities["modes"]
        ):
            self._get_collection()
        return self._search_capabilities


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _WrapEmbedFn:
    """Adapt a plain ``Callable[[list[str]], list[list[float]]]`` to the
    chromadb ``EmbeddingFunction`` protocol.

    Chromadb 1.5+ requires ``name()``, ``embed_query()``, and other
    methods for collection configuration and query embedding.
    """

    def __init__(self, fn: Callable[[list[str]], list[list[float]]]) -> None:
        self._fn = fn

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: N802
        return self._fn(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: N802
        return self._fn(input)

    @staticmethod
    def name() -> str:
        return "default"

    @staticmethod
    def is_legacy() -> bool:
        return False


def _clean_metadata(meta: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Filter metadata to only JSON-serializable scalar values.

    Chroma Cloud may include non-serializable objects in metadata
    (e.g. SparseVector from BM25 indexes).  We keep only str, int,
    float, bool, and None values.
    """
    return tuple(
        (k, v)
        for k, v in meta.items()
        if isinstance(v, (str, int, float, bool, type(None)))
    )


def _extract_content(
    doc: str | None,
    meta: dict[str, Any],
    content_fields: list[str],
) -> str:
    """Return the chunk text, reading from metadata when content_fields != ["content"].

    For the default ``["content"]`` case the Chroma ``document`` field is
    used.  For custom content_attr (pre-existing collections) we pull
    from the named metadata field(s) and JSON-join when multiple.
    """
    if content_fields == ["content"]:
        return doc or ""
    if len(content_fields) == 1:
        return str(meta.get(content_fields[0], doc or ""))
    import json as _json

    return _json.dumps(
        {f: meta.get(f, "") for f in content_fields}, default=str
    )


def _results_to_chunks(
    result: dict[str, Any],
    content_fields: list[str] | None = None,
) -> list[Chunk]:
    """Convert a Chroma ``get()`` result dict to a list of Chunks."""
    fields = content_fields or ["content"]
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    chunks: list[Chunk] = []
    for doc, meta in zip(docs, metas):
        content = _extract_content(doc, meta, fields)
        chunks.append(Chunk(content=content, metadata=_clean_metadata(meta)))
    return chunks


def _query_results_to_chunks(
    result: dict[str, Any],
    content_fields: list[str] | None = None,
) -> list[Chunk]:
    """Convert a Chroma ``query()`` result dict to a list of Chunks.

    Query results are nested one level deeper than get results
    (list-of-lists since multiple query texts are supported).
    """
    fields = content_fields or ["content"]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    chunks: list[Chunk] = []
    for doc, meta in zip(docs, metas):
        content = _extract_content(doc, meta, fields)
        chunks.append(Chunk(content=content, metadata=_clean_metadata(meta)))
    return chunks


def _query_results_to_scored_chunks(
    result: dict[str, Any],
    content_fields: list[str] | None = None,
) -> list[tuple[Chunk, float]]:
    """Convert a Chroma ``query()`` result to (Chunk, score) pairs.

    Chroma distances are lower-is-closer; we convert to higher-is-better
    via ``1 / (1 + distance)`` for consistent ranking with the Search API.
    """
    fields = content_fields or ["content"]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    pairs: list[tuple[Chunk, float]] = []
    for doc, meta, dist in zip(docs, metas, distances):
        content = _extract_content(doc, meta, fields)
        chunk = Chunk(content=content, metadata=_clean_metadata(meta))
        score = 1.0 / (1.0 + dist) if dist >= 0 else 0.0
        pairs.append((chunk, score))
    return pairs
