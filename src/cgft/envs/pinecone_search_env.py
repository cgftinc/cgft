"""PineconeSearchEnv — backward-compatible wrapper.

Delegates to :class:`SearchClientEnv` with a :class:`PineconeSearch` client.
New code should use ``SearchClientEnv`` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cgft.corpus.pinecone.search import PineconeSearch

from .search_client_env import SearchClientEnv


class PineconeSearchEnv(SearchClientEnv):
    """Search environment backed by Pinecone.

    Thin wrapper that constructs a :class:`PineconeSearch` and delegates
    to :class:`SearchClientEnv`.  Provided for backward compatibility —
    new code should use ``SearchClientEnv(search=PineconeSearch(...))``
    directly.
    """

    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        embed_model: str = "multilingual-e5-large",
        index_host: str | None = None,
        namespace: str = "",
        field_mapping: dict[str, str] | None = None,
        judge_base_url: str = "",
        judge_api_key: str = "",
        judge_model: str = "",
        judge_timeout: float = 30.0,
        w_correctness: float = 1.0,
        **kwargs: Any,
    ):
        search = PineconeSearch(
            api_key=pinecone_api_key,
            index_name=index_name,
            index_host=index_host,
            namespace=namespace,
            embed_fn=embed_fn,
            embed_model=embed_model,
            field_mapping=field_mapping,
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
