"""CorporaSearchEnv — SearchEnv backed by the Corpora HTTP API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import aiohttp
from benchmax.envs.types import ToolDefinition

from .search_env import SearchEnv


class CorporaSearchEnv(SearchEnv):
    """Search environment backed by the Corpora HTTP corpus API.

    Args:
        api_key: API key for the corpus service
        corpus_id: ID of the corpus to search
        base_url: Base URL for the corpus API
        dataset_path: Path to dataset JSONL file
    """

    def __init__(
        self,
        api_key: str,
        corpus_id: str,
        base_url: str,
        dataset_path: str,
        **kwargs,
    ):
        self._api_key = api_key
        self._corpus_id = corpus_id
        self._base_url = base_url.rstrip("/")
        self._dataset_path = dataset_path

        search_tool_definition = ToolDefinition(
            name="search_corpus",
            description="Search the corpus using BM25 with optional metadata and filename filtering.",
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
            search_tool_definition.name: (search_tool_definition, self._search_corpus_tool)
        }

    async def _search_corpus_tool(
        self,
        query: str,
        metadata: dict[str, Any] | None = None,
        filename: str | None = None,
        limit: int = 10,
        **kwargs,
    ) -> str:
        """Search the corpus using BM25.

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

        # Build request body
        request_body = {"query": query, "limit": limit}
        if metadata:
            request_body["metadata"] = metadata
        if filename:
            request_body["filename"] = filename

        # Build URL
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

            # Format results
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
