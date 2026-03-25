"""TpufSearchEnv — backward-compatible wrapper.

Delegates to :class:`SearchClientEnv` with a :class:`TpufSearch` client.
New code should use ``SearchClientEnv`` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cgft.corpus.turbopuffer.search import TpufSearch

from .search_client_env import SearchClientEnv


class TpufSearchEnv(SearchClientEnv):
    """Search environment backed by Turbopuffer.

    Thin wrapper that constructs a :class:`TpufSearch` and delegates
    to :class:`SearchClientEnv`.  Provided for backward compatibility —
    new code should use ``SearchClientEnv(search=TpufSearch(...))``
    directly.
    """

    def __init__(
        self,
        turbopuffer_api_key: str,
        namespace: str,
        region: str = "aws-us-east-1",
        content_attr: list[str] | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        judge_base_url: str = "",
        judge_api_key: str = "",
        judge_model: str = "",
        judge_timeout: float = 30.0,
        w_correctness: float = 1.0,
        **kwargs: Any,
    ):
        search = TpufSearch(
            api_key=turbopuffer_api_key,
            namespace=namespace,
            region=region,
            content_attr=content_attr,
            embed_fn=embed_fn,
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
