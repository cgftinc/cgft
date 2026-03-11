"""Base search environment with shared logic for corpus-backed RL training."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import StandardizedExample, ToolDefinition

if TYPE_CHECKING:
    from collections.abc import Callable

SYSTEM_PROMPT = """Please use the search tool provided to find relevant information from the corpus.
Formulate effective search queries to retrieve the most relevant chunks.
You can filter by metadata or filename to narrow your search.
Write your complete answer on the final line only as a concise entity, within the xml tags <answer></answer>.
"""
MAX_TOOL_OUTPUT_CHARS = 10000
TOOL_OUTPUT_TRUNCATION_SUFFIX = "\n...[truncated due to character limit]"


def percent_of_text_a_in_text_b(text_a: str, text_b: str) -> float:
    """Calculate percentage of text_a that appears in text_b."""
    if not text_a:
        return 0.0

    matcher = SequenceMatcher(None, text_a, text_b)
    matched_chars = sum(size for _, _, size in matcher.get_matching_blocks())
    return matched_chars / len(text_a)


async def chunk_overlap_reward_function(completion: str, ground_truth: str, **kwargs: Any) -> float:
    """Reward function based on chunk overlap.

    Computes the percentage of overlapping text between the completion
    and the reference chunks.

    Args:
        completion: The model's generated text
        ground_truth: The reference text (not used directly)
        **kwargs: Must include reference_chunks

    Returns:
        float: A score between 0.0 and 1.0 representing the overlap percentage
    """
    reference_chunks = kwargs.get("reference_chunks", [])
    reference_string = " ".join(
        [
            chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in reference_chunks
        ]
    )

    completion_str = completion if isinstance(completion, str) else ""
    if isinstance(completion, list):
        completion_str = " ".join(
            [
                c.get("content", "")
                for c in completion
                if isinstance(c, dict) and c.get("role", "") != "assistant"
            ]
        )
        # Penalize excessive tool calls
        for msg in completion:
            if not isinstance(msg, dict):
                continue
            if msg.get("role", "") != "assistant":
                continue
            msg_content = msg.get("content", "")
            if msg_content.count("<tool_call>") >= 4:
                return 0.0

    if reference_string:
        overlap_score = percent_of_text_a_in_text_b(reference_string, completion_str)
        if overlap_score >= 0.25:
            return overlap_score
    return 0.0


class SearchEnv(BaseEnv):
    """Base search environment with shared logic for corpus-backed RL training.

    Subclasses must set ``self._tools`` in their ``__init__`` as a dict mapping
    tool name to a tuple of ``(ToolDefinition, Callable)``, e.g.::

        self._tools = {
            "search": (tool_definition, self._search_tool)
        }

    See ``CgftSearchEnv`` and ``TpufSearchEnv`` for concrete implementations.
    """

    system_prompt: str = SYSTEM_PROMPT
    _tools: dict[str, tuple[ToolDefinition, Callable]] = {}

    @staticmethod
    def _truncate_tool_output(
        text: str,
        max_chars: int = MAX_TOOL_OUTPUT_CHARS,
        suffix: str = TOOL_OUTPUT_TRUNCATION_SUFFIX,
    ) -> str:
        """Clamp tool output length to avoid overlong prompts in later turns."""
        if len(text) <= max_chars:
            return text

        keep = max(0, max_chars - len(suffix))
        truncated = text[:keep].rstrip()
        return f"{truncated}{suffix}"

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> StandardizedExample:
        """Preprocess dataset example into standardized format."""
        return StandardizedExample(
            prompt=example.get("question", ""),
            ground_truth=example.get("answer", None),
            init_rollout_args={},
        )

    async def list_tools(self) -> list[ToolDefinition]:
        """List available tools."""
        return [self._tools[k][0] for k in sorted(self._tools)]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        """Execute a tool.

        Args:
            rollout_id: Identifier for current rollout
            tool_name: Name of the tool
            **tool_args: Arguments for the tool function

        Returns:
            Tool execution result or error message
        """
        _, tool_function = self._tools[tool_name]
        return await tool_function(**tool_args)

    async def compute_reward(
        self, rollout_id: str, completion: str, ground_truth: Any, **kwargs: Any
    ) -> dict[str, float]:
        """Compute rewards using the chunk overlap reward function."""
        return {
            "chunk_overlap_reward_function": await chunk_overlap_reward_function(
                completion, ground_truth, **kwargs
            )
        }
