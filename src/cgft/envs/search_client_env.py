"""SearchClientEnv — unified search environment for any backend.

Uses :class:`SearchClient` instead of per-backend subclasses.
Pickle-safe: only the SearchClient (which stores serializable connection
params) is serialized.  No Pydantic/Chunk in the pickle graph.
"""

from __future__ import annotations

import re
import traceback
from collections.abc import Callable
from typing import Any

from benchmax.envs.tracking import log_env
from benchmax.envs.types import ToolDefinition

from cgft.corpus.search_client import SearchClient

from .search_env import SearchEnv

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


class SearchClientEnv(SearchEnv):
    """Backend-agnostic search environment using SearchClient.

    Replaces per-backend envs (PineconeSearchEnv, ChromaSearchEnv, etc.)
    with a single composable environment.  Any backend that implements
    :class:`SearchClient` works.

    Args:
        search: A :class:`SearchClient` instance (pickle-safe).
        judge_base_url: Base URL for the LLM judge API (optional).
        judge_api_key: API key for the LLM judge (optional).
        judge_model: Model name for the LLM judge (optional).
        judge_timeout: Timeout for judge API calls.
        w_correctness: Weight for correctness reward component.

    Example::

        from cgft.corpus.pinecone.search import PineconeSearch
        from cgft.envs.search_client_env import SearchClientEnv

        search = PineconeSearch(api_key="...", index_name="my-docs")
        env = SearchClientEnv(search=search)

        # Or with judge reward:
        env = SearchClientEnv(
            search=search,
            judge_base_url="https://...",
            judge_api_key="...",
            judge_model="gpt-4o",
        )
    """

    def __init__(
        self,
        search: SearchClient,
        *,
        judge_base_url: str = "",
        judge_api_key: str = "",
        judge_model: str = "",
        judge_timeout: float = 30.0,
        w_correctness: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self._search = search
        self._judge_base_url = judge_base_url
        self._judge_api_key = judge_api_key
        self._judge_model = judge_model
        self._judge_timeout = judge_timeout
        self._w_correctness = w_correctness
        self._experiment_id = kwargs.get("experiment_id")
        self._rollout_api_key = kwargs.get("api_key")

        modes = search.available_modes

        # Pick default mode: hybrid > lexical > vector
        if "hybrid" in modes:
            self._default_mode = "hybrid"
        elif "lexical" in modes:
            self._default_mode = "lexical"
        else:
            self._default_mode = "vector"

        # Build tool schema from available modes
        properties: dict[str, Any] = {
            "query": {
                "type": "string",
                "description": "Search query string.",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of results to return (default 10).",
            },
        }
        if len(modes) > 1:
            properties["mode"] = {
                "type": "string",
                "enum": modes,
                "description": (
                    f"Search mode: {', '.join(modes)}. "
                    f'Default: "{self._default_mode}".'
                ),
            }

        search_tool = ToolDefinition(
            name="search",
            description="Search the corpus.",
            input_schema={
                "type": "object",
                "properties": properties,
                "required": ["query"],
            },
        )
        self._tools: dict[str, tuple[ToolDefinition, Callable]] = {
            search_tool.name: (search_tool, self._search_tool)
        }

    async def _search_tool(
        self,
        query: str,
        mode: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> str:
        """Execute search via the SearchClient."""
        if not query:
            return "Error: Missing required parameter: 'query'"

        effective_mode = mode or self._default_mode
        try:
            results = self._search.search(
                query=query,
                mode=effective_mode,
                top_k=limit,
            )
        except Exception:
            return f"Error:\n{traceback.format_exc()}"

        if not results:
            return "No results found."

        lines = [
            f"{i}.\n   Content: {content}"
            for i, content in enumerate(results, 1)
        ]
        return self._truncate_tool_output("\n".join(lines))

    async def compute_reward(
        self,
        rollout_id: str,
        completion: str | list[dict[str, Any]],
        ground_truth: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute reward — uses judge if configured, else overlap."""
        # If no judge configured, fall back to base class overlap reward
        if not self._judge_base_url or not self._judge_api_key:
            return await super().compute_reward(
                rollout_id, completion, ground_truth, **kwargs
            )

        # Judge-based reward
        zeros = {"correctness": 0.0}
        try:
            text = _extract_completion_text(completion)
            if not text.strip():
                return zeros

            answer = _extract_answer_block(text)
            prompt = str(
                kwargs.get("prompt") or kwargs.get("question") or ""
            )
            gt_str = str(ground_truth or "")

            log_env(
                rollout_id,
                f"[SearchClientEnv] Q: {prompt[:200]}\n"
                f"  GT: {gt_str[:200]}\n"
                f"  A: {answer[:200]}",
            )

            from cgft.rubrics.rubric import Rubric, evaluate_single_rubric

            rubric = Rubric(
                title="Answer correctness",
                description=(
                    "Response correctly answers the question and is "
                    "factually consistent with the reference answer."
                ),
                type="positive",
                score_map={
                    0: "Provided answer is missing or incorrect.",
                    0.5: "Partially correct — captures some facts.",
                    1: "Fully correct and factually consistent.",
                },
            )
            result = await evaluate_single_rubric(
                rubric=rubric,
                question=prompt,
                ground_truth=gt_str,
                response=answer,
                model_name=self._judge_model,
                base_url=self._judge_base_url,
                api_key=self._judge_api_key,
                timeout=self._judge_timeout,
            )
            score = max(0.0, min(1.0, float(result.get("score", 0.0))))
            log_env(rollout_id, f"[SearchClientEnv] correctness={score:.2f}")
            return {"correctness": self._w_correctness * score}

        except Exception as exc:
            log_env(
                rollout_id,
                f"[SearchClientEnv] compute_reward failed: {exc}",
            )
            return zeros


def _extract_completion_text(completion: str | list[dict[str, Any]]) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return ""
    parts: list[str] = []
    for msg in completion:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content)
    return "\n".join(parts)


def _extract_answer_block(text: str) -> str:
    match = _ANSWER_TAG_RE.search(text or "")
    return (match.group(1) if match else text).strip()
