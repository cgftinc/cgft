"""TpufSearchEnv — SearchEnv backed by Turbopuffer BM25 search."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from benchmax.envs.types import ToolDefinition

from .search_env import SearchEnv


class TpufSearchEnv(SearchEnv):
    """Search environment backed by a Turbopuffer namespace.

    Args:
        turbopuffer_api_key: Turbopuffer API key
        namespace: Turbopuffer namespace name
        dataset_path: Path to dataset JSONL file
        region: Turbopuffer region (default ``"aws-us-east-1"``)
        content_attr: List of attribute names to search over via BM25.
            Defaults to ["content"].
    """

    def __init__(
        self,
        turbopuffer_api_key: str,
        namespace: str,
        dataset_path: str,
        region: str = "aws-us-east-1",
        content_attr: list[str] | None = None,
        **kwargs,
    ):
        import turbopuffer

        self._ns = turbopuffer.Turbopuffer(api_key=turbopuffer_api_key, region=region).namespace(namespace)
        self._fields: list[str] = content_attr if content_attr is not None else ["content"]
        self._dataset_path = dataset_path

        search_tool_definition = ToolDefinition(
            name="search_corpus",
            description="Search the corpus using BM25 full-text search.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        )

        self._tools: dict[str, tuple[ToolDefinition, Callable]] = {
            search_tool_definition.name: (search_tool_definition, self._search_corpus_tool)
        }

    def _build_rank_by(self, query: str) -> Any:
        """Build the rank_by argument for a BM25 Turbopuffer query."""
        if len(self._fields) == 1:
            return (self._fields[0], "BM25", query)
        return (
            "Sum",
            tuple(("Product", 1, (f, "BM25", query)) for f in self._fields),
        )

    async def _search_corpus_tool(self, query: str, limit: int = 10, **kwargs) -> str:
        """Search the Turbopuffer namespace using BM25.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            Formatted search results or error message
        """
        if not query:
            return "Error: Missing required parameter: 'query'"

        try:
            result = self._ns.query(
                rank_by=self._build_rank_by(query),
                top_k=limit,
                include_attributes=True,
            )
        except Exception as e:
            return f"Error: {str(e)}"

        if not result.rows:
            return "No results found."

        lines = []
        for i, row in enumerate(result.rows, start=1):
            score = getattr(row, "$dist", None)
            score_str = f"(score: {score:.4f})" if score is not None else ""
            if len(self._fields) == 1:
                content = str(getattr(row, self._fields[0], ""))
            else:
                content = json.dumps(
                    {f: getattr(row, f, "") for f in self._fields}, default=str
                )
            lines.append(f"{i}. {score_str}\n   Content: {content}")

        return "\n".join(lines)
