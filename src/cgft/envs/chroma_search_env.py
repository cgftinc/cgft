"""ChromaSearchEnv — backward-compatible wrapper.

Delegates to :class:`SearchClientEnv` with a :class:`ChromaSearch` client.
New code should use ``SearchClientEnv`` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cgft.corpus.chroma.search import ChromaSearch

from .search_client_env import SearchClientEnv


class ChromaSearchEnv(SearchClientEnv):
    """Search environment backed by ChromaDB.

    Thin wrapper that constructs a :class:`ChromaSearch` and delegates
    to :class:`SearchClientEnv`.  Provided for backward compatibility —
    new code should use ``SearchClientEnv(search=ChromaSearch(...))``
    directly.
    """

    def __init__(
        self,
        collection_name: str,
        host: str,
        port: int = 8000,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        enable_bm25: bool = True,
        judge_base_url: str = "",
        judge_api_key: str = "",
        judge_model: str = "",
        judge_timeout: float = 30.0,
        w_correctness: float = 1.0,
        **kwargs: Any,
    ):
        if not host:
            raise ValueError(
                "ChromaSearchEnv requires client-server mode (host must be "
                "set). Local/in-memory Chroma clients cannot survive pickle "
                "for remote training environments."
            )

        search = ChromaSearch(
            collection_name=collection_name,
            host=host,
            port=port,
            embed_fn=embed_fn,
            enable_bm25=enable_bm25,
        )
        super().__init__(
            search=search,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
            judge_model=judge_model,
            judge_timeout=judge_timeout,
            w_correctness=w_correctness,
            **kwargs,
        )
