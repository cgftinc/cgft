"""LinkerEnv — lightweight search environment for LLM-driven chunk linking.

Used by SearchAgentLinker to let an LLM search a corpus and find
related chunks for multi-hop QA generation. Much simpler than SearchEnv:
no reward components, no citation tracking, no answer correctness.
"""

from __future__ import annotations

import traceback
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import StandardizedExample, ToolDefinition

from cgft.corpus.search_client import SearchClient

MAX_TOOL_OUTPUT_CHARS = 8000
TOOL_OUTPUT_TRUNCATION_SUFFIX = "\n...[truncated]"

_SYSTEM_PROMPT = """\
You are finding related chunks for multi-hop question generation.

Given a primary chunk, search the corpus to find diverse related chunks. \
Issue 2-3 search queries exploring different aspects of the primary chunk.

After searching, list the most promising chunks you found. Focus on chunks that:
- Cover different but related topics to the primary chunk
- Could be combined with the primary chunk to form a multi-hop question
- Are substantive enough to contain useful information

Return your final selection inside <selected>...</selected> tags as a numbered list \
of the best chunk contents you found.\
"""


class LinkerEnv(BaseEnv):
    """Search environment for LLM-driven chunk linking.

    Exposes a single ``search`` tool backed by a :class:`SearchClient`.
    The LLM generates queries to find related chunks.

    Args:
        search: A pickle-safe :class:`SearchClient` instance.
        max_search_calls: Maximum number of search calls allowed.
    """

    def __init__(
        self,
        search: SearchClient,
        *,
        max_search_calls: int = 3,
        **kwargs: Any,
    ) -> None:
        self._search = search
        self._max_search_calls = max_search_calls

        modes = sorted(search.available_modes)
        if "hybrid" in modes:
            self._default_mode = "hybrid"
        elif "lexical" in modes:
            self._default_mode = "lexical"
        elif modes:
            self._default_mode = modes[0]
        else:
            self._default_mode = "lexical"

        self._has_metadata_search = hasattr(search, "search_with_metadata")

        search_tool = ToolDefinition(
            name="search",
            description="Search the corpus for related content.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        )
        self._tools: dict[str, tuple[ToolDefinition, Any]] = {
            "search": (search_tool, self._search_tool),
        }

        self.system_prompt = _SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[ToolDefinition]:
        return [self._tools[k][0] for k in sorted(self._tools)]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        if tool_name not in self._tools:
            return f"Error: Unknown tool '{tool_name}'"
        _, tool_fn = self._tools[tool_name]
        return await tool_fn(**tool_args)

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs: Any) -> StandardizedExample:
        target_n = example.get("target_n", 2)
        prompt = (
            f"Find {target_n} diverse related chunk(s) for the following "
            f"primary chunk.\n\nPrimary chunk:\n{example.get('prompt', '')}"
        )
        return StandardizedExample(
            prompt=prompt,
            ground_truth=None,
            init_rollout_args={},
        )

    async def compute_reward(
        self,
        rollout_id: str,
        completion: str | list[dict[str, Any]],
        ground_truth: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        return {"linking": 1.0}

    # ------------------------------------------------------------------
    # Search tool
    # ------------------------------------------------------------------

    async def _search_tool(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> str:
        if not query:
            return "Error: Missing required parameter: 'query'"
        try:
            if self._has_metadata_search:
                results = self._search.search_with_metadata(
                    query=query, mode=self._default_mode, top_k=limit
                )
                return self._format_metadata_results(results)
            results = self._search.search(query=query, mode=self._default_mode, top_k=limit)
            return self._format_simple_results(results)
        except Exception:
            return f"Error:\n{traceback.format_exc()}"

    def _format_metadata_results(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "No results found."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.get("source", "")
            score = r.get("score", 0.0)
            content = r.get("content", "")
            header = f"{i}."
            if source:
                header += f" [source: {source}]"
            if score:
                header += f" (score: {score:.2f})"
            lines.append(f"{header}\n   {content}")
        return _truncate("\n".join(lines))

    @staticmethod
    def _format_simple_results(results: list[str]) -> str:
        if not results:
            return "No results found."
        lines = [f"{i}. {c}" for i, c in enumerate(results, 1)]
        return _truncate("\n".join(lines))


def _truncate(
    text: str,
    max_chars: int = MAX_TOOL_OUTPUT_CHARS,
    suffix: str = TOOL_OUTPUT_TRUNCATION_SUFFIX,
) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(suffix))
    return text[:keep].rstrip() + suffix
