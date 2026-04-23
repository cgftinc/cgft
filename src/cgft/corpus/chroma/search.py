"""ChromaSearch — pickle-safe search client for RL environments.

Implements :class:`SearchClient` using the shared
:class:`ChromaClient`.  No Chunk or Pydantic dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Eager import so `from cgft.corpus.chroma.search import ChromaSearch`
# pulls `.client` into sys.modules. Without this, cloudpickle's
# `register_pickle_by_value(cgft)` doesn't know to inline the client
# submodule — the rollout-server worker then fails the lazy `from .client
# import ChromaClient` inside _get_client with `ModuleNotFoundError: cgft`.
# Matches the CorporaSearch pattern (corpus/corpora/search.py:10).
from .client import ChromaClient


class ChromaSearch:
    """Pickle-safe Chroma search client for RL environments.

    Stores only serializable connection parameters.  The Chroma client
    is created lazily on first search call (including after unpickle).

    Pass **either** Cloud credentials (``api_key`` + ``tenant`` +
    ``database``) **or** a self-hosted ``host``. Cloud takes precedence
    if both are set; wizard-emitted env classes always use Cloud.

    Args:
        collection_name: Name of the Chroma collection.
        api_key: Chroma Cloud API key (``ck-...``).
        tenant: Chroma Cloud tenant ID.
        database: Chroma Cloud database name.
        host: Self-hosted server hostname.
        port: Self-hosted server port (default 8000).
        embed_fn: Custom embedding function. When ``None``, Chroma's
            built-in embeddings are used.
        enable_bm25: Enable BM25 for lexical/hybrid modes.
        content_attr: Metadata fields to treat as content.
    """

    def __init__(
        self,
        collection_name: str,
        api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        host: str | None = None,
        port: int = 8000,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        enable_bm25: bool = True,
        content_attr: list[str] | None = None,
    ) -> None:
        if not any([api_key and tenant and database, host]):
            raise ValueError(
                "ChromaSearch requires either Cloud credentials "
                "(api_key + tenant + database) or a self-hosted host."
            )
        self._collection_name = collection_name
        self._api_key = api_key
        self._tenant = tenant
        self._database = database
        self._host = host
        self._port = port
        self._embed_fn = embed_fn
        self._enable_bm25 = enable_bm25
        self._content_attr = content_attr
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = ChromaClient(
                collection_name=self._collection_name,
                api_key=self._api_key,
                tenant=self._tenant,
                database=self._database,
                host=self._host,
                port=self._port,
                embed_fn=self._embed_fn,
                enable_bm25=self._enable_bm25,
                content_attr=self._content_attr,
            )
        return self._client

    def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search and return structured results."""
        client = self._get_client()

        if mode == "auto":
            modes = client.modes
            if "hybrid" in modes:
                mode = "hybrid"
            elif "lexical" in modes:
                mode = "lexical"
            else:
                mode = "vector"
        elif mode not in client.modes:
            raise ValueError(
                f"ChromaSearch does not support mode '{mode}'. "
                f"Available modes: {sorted(client.modes)}"
            )

        if client.search_api and mode in ("lexical", "hybrid"):
            vec = client.embed(query) if mode == "hybrid" else None
            rows = client.search_api_raw(
                text_query=query,
                vector_query=vec,
                mode=mode,
                top_k=top_k,
            )
        else:
            vec = client.embed(query)
            rows = client.query_raw(
                text_query=query,
                vector_query=vec,
                top_k=top_k,
            )

        return [
            {
                "content": client.extract_content(r["content"], r["metadata"]),
                "source": str(r["metadata"].get("file") or r["metadata"].get("file_path") or ""),
                "metadata": r["metadata"],
                "score": float(r.get("score", 0.0) or 0.0),
            }
            for r in rows
        ]

    def embed(self, text: str) -> list[float] | None:
        """Return embedding vector, or None for auto-embed."""
        return self._get_client().embed(text)

    @property
    def available_modes(self) -> list[str]:
        return sorted(self._get_client().modes)

    def get_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "backend": "chroma",
            "collection_name": self._collection_name,
            "enable_bm25": self._enable_bm25,
        }
        if self._api_key is not None:
            # Redact the key — this dict is printed in diagnostics and logs.
            params.update(
                mode="cloud",
                tenant=self._tenant,
                database=self._database,
                api_key=(self._api_key[:8] + "...") if self._api_key else None,
            )
        else:
            params.update(mode="self_hosted", host=self._host, port=self._port)
        return params

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None
