"""SearchEnv — unified search environment for any backend.

Uses :class:`SearchClient` instead of per-backend subclasses.
Pickle-safe: only the SearchClient (which stores serializable connection
params) is serialized.  No Pydantic/Chunk in the pickle graph.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.tracking import log_env
from benchmax.envs.types import StandardizedExample, ToolDefinition

from cgft.corpus.search_client import SearchClient
from cgft.envs.reward_helpers import (
    extract_answer_block,
    extract_completion_text,
    overlap_reward,
)

SYSTEM_PROMPT = """Please use the search tool provided to find relevant information from the corpus.
Formulate effective search queries to retrieve the most relevant chunks.
You can filter by metadata or filename to narrow your search.
Write your complete answer on the final line only as a concise entity, within the xml tags <answer></answer>.
"""
MAX_TOOL_OUTPUT_CHARS = 10000
TOOL_OUTPUT_TRUNCATION_SUFFIX = "\n...[truncated due to character limit]"


class SearchEnv(BaseEnv):
    """Backend-agnostic search environment using SearchClient.

    Subclasses ``BaseEnv`` directly — no Chunk/Pydantic in the pickle
    graph.  Any backend that implements :class:`SearchClient` works.

    Args:
        search: A :class:`SearchClient` instance (pickle-safe).
        judge_base_url: Base URL for the LLM judge API (optional).
        judge_api_key: API key for the LLM judge (optional).
        judge_model: Model name for the LLM judge (optional).
        judge_timeout: Timeout for judge API calls.
        w_correctness: Weight for correctness reward component.

    Example::

        from cgft.corpus.pinecone.search import PineconeSearch
        from cgft.envs.search_env import SearchEnv

        search = PineconeSearch(api_key="...", index_name="my-docs")
        env = SearchEnv(search=search)

        # Or with judge reward:
        env = SearchEnv(
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

    system_prompt: str = SYSTEM_PROMPT

    @staticmethod
    def _truncate_tool_output(
        text: str,
        max_chars: int = MAX_TOOL_OUTPUT_CHARS,
        suffix: str = TOOL_OUTPUT_TRUNCATION_SUFFIX,
    ) -> str:
        if len(text) <= max_chars:
            return text
        keep = max(0, max_chars - len(suffix))
        return f"{text[:keep].rstrip()}{suffix}"

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs: Any) -> StandardizedExample:
        return StandardizedExample(
            prompt=example.get("question", ""),
            ground_truth=example.get("answer", None),
            init_rollout_args={},
        )

    async def list_tools(self) -> list[ToolDefinition]:
        return [self._tools[k][0] for k in sorted(self._tools)]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        _, tool_function = self._tools[tool_name]
        return await tool_function(**tool_args)

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
        if not self._judge_base_url or not self._judge_api_key:
            return {
                "chunk_overlap_reward_function": overlap_reward(
                    completion, ground_truth, **kwargs
                )
            }

        zeros = {"correctness": 0.0}
        try:
            text = extract_completion_text(completion)
            if not text.strip():
                return zeros

            answer = extract_answer_block(text)
            prompt = str(
                kwargs.get("prompt") or kwargs.get("question") or ""
            )
            gt_str = str(ground_truth or "")

            log_env(
                rollout_id,
                f"[SearchEnv] Q: {prompt[:200]}\n"
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
            log_env(rollout_id, f"[SearchEnv] correctness={score:.2f}")
            return {"correctness": self._w_correctness * score}

        except Exception as exc:
            log_env(
                rollout_id,
                f"[SearchEnv] compute_reward failed: {exc}",
            )
            return zeros
