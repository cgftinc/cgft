"""TpufSearchEnv — SearchEnv backed by Turbopuffer search (BM25, vector, or hybrid)."""

from __future__ import annotations

import re
import traceback
from collections.abc import Callable
from typing import Any

from benchmax.envs.tracking import log_env
from benchmax.envs.types import ToolDefinition

from cgft.corpus.search_schema.search_types import SearchSpec
from cgft.corpus.turbopuffer.source import TpufChunkSource
from cgft.rubrics.rubric import Rubric, evaluate_single_rubric

from .search_env import SearchEnv

_CORRECTNESS_RUBRIC = Rubric(
    title="Answer correctness",
    description=(
        "Response correctly answers the question and is factually consistent "
        "with the reference answer."
    ),
    type="positive",
    score_map={
        0: "Provided answer is missing or incorrect.",
        0.5: (
            "Response captures some facts from the reference answer, "
            "but is missing key facts or has an incorrect conclusion."
        ),
        1: (
            "Response correctly answers the question and is factually "
            "consistent with the reference answer."
        ),
    },
)

DEFAULT_W_CORRECTNESS = 1.0
DEFAULT_JUDGE_TIMEOUT = 30.0

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


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
        judge_base_url: Base URL for the LLM judge API.
        judge_api_key: API key for the LLM judge.
        judge_model: Model name for the LLM judge.
        w_correctness: Weight for the correctness reward component.
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
        judge_timeout: float | None = DEFAULT_JUDGE_TIMEOUT,
        w_correctness: float = DEFAULT_W_CORRECTNESS,
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
        self._judge_base_url = judge_base_url
        self._judge_api_key = judge_api_key
        self._judge_model = judge_model
        self._judge_timeout = judge_timeout
        self._w_correctness = w_correctness
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
                effective_mode = "lexical"

        spec = SearchSpec(
            mode=effective_mode,  # type: ignore[arg-type]
            top_k=limit,
            text_query=query,
            vector_query=vector_query,
        )
        try:
            results = self._source.search_content(spec)
        except Exception:
            return f"Error:\n{traceback.format_exc()}"

        if not results:
            return "No results found."

        lines = [f"{i}.\n   Content: {content}" for i, content in enumerate(results, 1)]
        return self._truncate_tool_output("\n".join(lines))

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    async def compute_reward(
        self,
        rollout_id: str,
        completion: str | list[dict[str, Any]],
        ground_truth: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Return correctness reward via LLM judge rubric."""
        zeros = {"correctness": 0.0}

        try:
            completion_text = _extract_completion_text(completion)
            if not completion_text.strip():
                return zeros

            answer_block = _extract_answer_block(completion_text)
            prompt = str(kwargs.get("prompt") or kwargs.get("question") or "")
            gt_str = str(ground_truth or "")

            log_env(
                rollout_id,
                f"[TpufSearchEnv] Question: {prompt[:200]}\n"
                f"  Ground truth: {gt_str[:200]}\n"
                f"  Answer: {answer_block[:200]}",
            )

            if not self._judge_base_url or not self._judge_api_key:
                log_env(rollout_id, "[TpufSearchEnv] Judge disabled: missing credentials")
                return zeros

            correctness_result = await evaluate_single_rubric(
                rubric=_CORRECTNESS_RUBRIC,
                question=prompt,
                ground_truth=gt_str,
                response=answer_block,
                model_name=self._judge_model,
                base_url=self._judge_base_url,
                api_key=self._judge_api_key,
                timeout=self._judge_timeout,
            )
            correctness_raw = _clip01(correctness_result.get("score", 0.0))

            log_env(rollout_id, f"[TpufSearchEnv] correctness={correctness_raw:.2f}")

            return {"correctness": self._w_correctness * correctness_raw}
        except Exception as exc:
            log_env(rollout_id, f"[TpufSearchEnv] compute_reward failed: {exc}")
            return zeros


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _clip01(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


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
