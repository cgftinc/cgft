"""PineconeSearchEnv — SearchEnv backed by Pinecone vector search."""

from __future__ import annotations

import re
import traceback
from collections.abc import Callable
from typing import Any

from benchmax.envs.tracking import log_env
from benchmax.envs.types import ToolDefinition

from cgft.corpus.pinecone.source import PineconeChunkSource
from cgft.corpus.search_schema.search_types import SearchSpec
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


class PineconeSearchEnv(SearchEnv):
    """Search environment backed by a Pinecone index.

    Only vector (ANN) search is supported.  The RL agent's search tool
    accepts a text query which is embedded automatically via ``embed_fn``.

    Args:
        pinecone_api_key: Pinecone API key.
        index_name: Name of the Pinecone index.
        index_host: Optional host URL (bypasses index name lookup).
        namespace: Pinecone namespace within the index (default ``""``).
        embed_fn: Custom embedding function ``list[str] → list[list[float]]``.
            When ``None``, Pinecone's Inference API is used with
            ``embed_model``.
        embed_model: Pinecone hosted embedding model name.  Ignored when
            ``embed_fn`` is provided.  Defaults to
            ``"multilingual-e5-large"``.
        field_mapping: Maps Pinecone metadata field names to internal names.
        judge_base_url: Base URL for the LLM judge API.
        judge_api_key: API key for the LLM judge.
        judge_model: Model name for the LLM judge.
        w_correctness: Weight for the correctness reward component.
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
        judge_timeout: float | None = DEFAULT_JUDGE_TIMEOUT,
        w_correctness: float = DEFAULT_W_CORRECTNESS,
        **kwargs,
    ):
        self._source = PineconeChunkSource(
            api_key=pinecone_api_key,
            index_name=index_name,
            index_host=index_host,
            namespace=namespace,
            embed_fn=embed_fn,
            embed_model=embed_model,
            field_mapping=field_mapping,
        )

        search_tool_definition = ToolDefinition(
            name="search",
            description="Search the corpus using vector similarity search.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": ("Max number of results to return (default 10)."),
                    },
                },
                "required": ["query"],
            },
        )

        self._tools: dict[str, tuple[ToolDefinition, Callable]] = {
            search_tool_definition.name: (
                search_tool_definition,
                self._search_tool,
            )
        }
        self._judge_base_url = judge_base_url
        self._judge_api_key = judge_api_key
        self._judge_model = judge_model
        self._judge_timeout = judge_timeout
        self._w_correctness = w_correctness
        self._experiment_id = kwargs.get("experiment_id")
        self._rollout_api_key = kwargs.get("api_key")

    async def _search_tool(self, query: str, limit: int = 10, **kwargs) -> str:
        """Search the Pinecone index.

        Args:
            query: Search query string (auto-embedded).
            limit: Maximum number of results.

        Returns:
            Formatted search results or error message.
        """
        if not query:
            return "Error: Missing required parameter: 'query'"

        vector_query = self._source.embed_query(query)

        spec = SearchSpec(
            mode="vector",
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
                f"[PineconeSearchEnv] Question: {prompt[:200]}\n"
                f"  Ground truth: {gt_str[:200]}\n"
                f"  Answer: {answer_block[:200]}",
            )

            if not self._judge_base_url or not self._judge_api_key:
                log_env(
                    rollout_id,
                    "[PineconeSearchEnv] Judge disabled: missing credentials",
                )
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

            log_env(
                rollout_id,
                f"[PineconeSearchEnv] correctness={correctness_raw:.2f}",
            )

            return {"correctness": self._w_correctness * correctness_raw}
        except Exception as exc:
            log_env(
                rollout_id,
                f"[PineconeSearchEnv] compute_reward failed: {exc}",
            )
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


def _extract_completion_text(
    completion: str | list[dict[str, Any]],
) -> str:
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
