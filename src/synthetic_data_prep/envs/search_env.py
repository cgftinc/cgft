"""Base search environment with shared logic for corpus-backed RL training."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import StandardizedExample, ToolDefinition

SYSTEM_PROMPT = """Please use the search tool provided to find relevant information from the corpus.
Formulate effective search queries to retrieve the most relevant chunks.
You can filter by metadata or filename to narrow your search.
Write your complete answer on the final line only as a concise entity, within the xml tags <answer></answer>.
"""


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

    Subclasses must set ``self._dataset_path`` and ``self._tools`` in their
    ``__init__``, then implement ``_search_corpus_tool``.

    See ``CorporaSearchEnv`` and ``TpufSearchEnv`` for concrete implementations.
    """

    system_prompt: str = SYSTEM_PROMPT
    reward_funcs = [chunk_overlap_reward_function]

    def get_train_val_split(self, train_ratio: float = 0.7, seed: int = 42, **kwargs):
        """Load the dataset and return stratified train/validation splits by qa_type.

        Args:
            train_ratio: Fraction of data to use for training (default 0.7)
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to load_dataset

        Returns:
            Tuple of (train_dataset, val_dataset) with stratified splits
        """
        from datasets import concatenate_datasets
        from datasets import load_dataset as hf_load_dataset

        ds = hf_load_dataset(
            "json", data_files=str(Path(self._dataset_path).expanduser().absolute())
        )
        dataset = ds["train"]

        # Get unique qa_types
        qa_types = set(dataset["qa_type"])

        train_splits = []
        val_splits = []

        for qa_type in qa_types:
            # Filter dataset by qa_type
            type_indices = [i for i, t in enumerate(dataset["qa_type"]) if t == qa_type]
            type_subset = dataset.select(type_indices)

            # Shuffle and split
            type_subset = type_subset.shuffle(seed=seed)
            split_idx = int(len(type_subset) * train_ratio)

            # Handle edge cases
            if split_idx == 0 and len(type_subset) > 0:
                split_idx = 1
            elif split_idx == len(type_subset) and len(type_subset) > 1:
                split_idx = len(type_subset) - 1

            train_splits.append(type_subset.select(range(split_idx)))
            val_splits.append(type_subset.select(range(split_idx, len(type_subset))))

        # Concatenate all splits
        train_dataset = concatenate_datasets(train_splits)
        val_dataset = concatenate_datasets(val_splits)

        # Shuffle to mix qa_types
        train_dataset = train_dataset.shuffle(seed=seed)
        val_dataset = val_dataset.shuffle(seed=seed)

        return train_dataset, val_dataset

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

    async def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        """Initialize rollout (no-op for stateless environment)."""
        pass

    async def release_rollout(self, rollout_id: str) -> None:
        """Release rollout (no-op for stateless environment)."""
        pass

    async def copy_to_workspace(
        self, rollout_id: str, src_path: Path, dst_filename: str | None = None
    ) -> None:
        """Not needed for this environment."""
        pass

    async def copy_content_to_workspace(
        self, rollout_id: str, src_content: str | bytes, dst_filename: str
    ) -> None:
        """Not needed for this environment."""
        pass

    async def copy_from_workspace(self, rollout_id: str, src_filename: str, dst_path: Path) -> None:
        """Not needed for this environment."""
        pass

    async def shutdown(self):
        """Cleanup (no-op for this environment)."""
        pass
