"""TpufSearchEnv — SearchEnv backed by Turbopuffer search (BM25, vector, or hybrid)."""

from __future__ import annotations

from collections.abc import Callable

from benchmax.envs.types import ToolDefinition

from synthetic_data_prep.corpus.search_schema.search_types import SearchSpec
from synthetic_data_prep.corpus.turbopuffer.source import TpufChunkSource

from .search_env import SearchEnv


class TpufSearchEnv(SearchEnv):
    """Search environment backed by a Turbopuffer namespace.

    Supports lexical (BM25), vector (ANN), and hybrid search. When ``embed_fn``
    is provided, vector and hybrid modes are enabled and the tool schema exposes a
    ``mode`` parameter so the RL agent can choose between them. Without ``embed_fn``
    only lexical search is available.

    Args:
        turbopuffer_api_key: Turbopuffer API key
        namespace: Turbopuffer namespace name
        region: Turbopuffer region (default ``"aws-us-east-1"``)
        content_attr: List of attribute names to search over via BM25.
            Defaults to ["content"].
        embed_fn: Optional callable that maps list[str] → list[list[float]].
            When provided, enables vector and hybrid search modes.
    """

    def __init__(
        self,
        turbopuffer_api_key: str,
        namespace: str,
        region: str = "aws-us-east-1",
        content_attr: list[str] | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        **kwargs,
    ):
        self._source = TpufChunkSource(
            api_key=turbopuffer_api_key,
            namespace=namespace,
            region=region,
            content_attr=content_attr,
            embed_fn=embed_fn,
        )
        supported_modes = self._source.get_search_capabilities()["modes"]

        if "hybrid" in supported_modes:
            mode_enum = ["lexical", "vector", "hybrid"]
            self._default_mode = "hybrid"
        elif "vector" in supported_modes:
            mode_enum = ["lexical", "vector"]
            self._default_mode = "vector"
        else:
            mode_enum = ["lexical"]
            self._default_mode = "lexical"

        tool_properties: dict = {
            "query": {
                "type": "string",
                "description": "Search query string.",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of results to return (default 10).",
            },
        }
        if len(mode_enum) > 1:
            tool_properties["mode"] = {
                "type": "string",
                "enum": mode_enum,
                "description": f'Search mode: {", ".join(mode_enum)}. Default: "{self._default_mode}".',
            }

        search_tool_definition = ToolDefinition(
            name="search",
            description="Search the corpus using full-text, vector, or hybrid search.",
            input_schema={
                "type": "object",
                "properties": tool_properties,
                "required": ["query"],
            },
        )

        self._tools: dict[str, tuple[ToolDefinition, Callable]] = {
            search_tool_definition.name: (search_tool_definition, self._search_tool)
        }
        self._experiment_id = kwargs.get("experiment_id")
        self._rollout_api_key = kwargs.get("api_key")

    async def _search_tool(self, query: str, mode: str | None = None, limit: int = 10, **kwargs) -> str:
        """Search the Turbopuffer namespace.

        Args:
            query: Search query string
            mode: Search mode (lexical/vector/hybrid). Defaults to best available.
            limit: Maximum number of results

        Returns:
            Formatted search results or error message
        """
        if not query:
            return "Error: Missing required parameter: 'query'"

        effective_mode = mode or self._default_mode
        vector_query = None
        if effective_mode in ("vector", "hybrid"):
            vector_query = self._source.embed_query(query)
            if vector_query is None:
                effective_mode = "lexical"  # graceful fallback if embed_fn not set

        spec = SearchSpec(
            mode=effective_mode,  # type: ignore[arg-type]
            top_k=limit,
            text_query=query,
            vector_query=vector_query,
        )
        try:
            chunks = self._source.search(spec)
        except Exception as e:
            return f"Error: {str(e)}"

        if not chunks:
            return "No results found."

        lines = [f"{i}.\n   Content: {chunk.content}" for i, chunk in enumerate(chunks, 1)]
        return self._truncate_tool_output("\n".join(lines))
