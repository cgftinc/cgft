"""CgftSearchEnv — SearchEnv backed by the CGFT HTTP API using BM25 search."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import aiohttp
from benchmax.envs.types import ToolDefinition

from .search_env import SearchEnv


class CgftSearchEnv(SearchEnv):
    """Search environment backed by the CGFT HTTP API using BM25 search.

    Args:
        api_key: API key for the search service
        corpus_id: ID of the corpus to search
        base_url: Base URL for the API
    """

    def __init__(
        self,
        api_key: str,
        corpus_id: str,
        base_url: str,
        **kwargs,
    ):
        self._api_key = api_key
        self._corpus_id = corpus_id
        self._base_url = base_url.rstrip("/")

        search_tool_definition = ToolDefinition(
            name="search",
            description="Search using BM25 with optional metadata and filename filtering.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata filters (e.g., {'ticker': 'DDOG', 'year': 2024}).",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename filter. Simple string for substring match (e.g., 'config') or regex pattern (e.g., '.*\\.json$').",
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
            search_tool_definition.name: (search_tool_definition, self._search_tool)
        }

        super().__init__(
            experiment_id=kwargs.get("experiment_id"),
            api_key=kwargs.get("api_key"),
            **{k: v for k, v in kwargs.items() if k not in ("experiment_id", "api_key")},
        )

    async def _search_tool(
        self,
        query: str,
        metadata: dict[str, Any] | None = None,
        filename: str | None = None,
        limit: int = 10,
        **kwargs,
    ) -> str:
        """Search using BM25.

        Args:
            query: Search query string
            metadata: Optional metadata filters
            filename: Optional filename filter (substring or regex)
            limit: Maximum number of results

        Returns:
            Formatted search results or error message
        """
        if not query:
            return "Error: Missing required parameter: 'query'"

        request_body = {"query": query, "limit": limit}
        if metadata:
            request_body["metadata"] = metadata
        if filename:
            request_body["filename"] = filename

        url = f"{self._base_url}/api/corpora/{self._corpus_id}/search"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=request_body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return f"Error: API request failed with status {resp.status}: {error_text}"

                    data = await resp.json()

            results = data.get("results", [])
            total = data.get("total", 0)

            if not results:
                return "No results found."

            lines = []
            for i, item in enumerate(results, start=1):
                filename_val = item.get("filename", "—")
                score = item.get("score")
                score_str = f"(score: {score:.2f})" if score is not None else "(filtered)"
                content = item.get("content", "")
                metadata_val = item.get("metadata", {})

                lines.append(f"{i}. {filename_val} {score_str}")
                lines.append(f"   Content: {content}")
                if metadata_val:
                    lines.append(f"   Metadata: {metadata_val}")

            lines.append(f"\nTotal: {total} results")
            return "\n".join(lines)

        except aiohttp.ClientError as e:
            return f"Error: Network error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
