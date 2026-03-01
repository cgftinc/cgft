"""Data models for QA generation datasets."""

from typing import Any, TypedDict


class ReferenceChunk(TypedDict):
    """A reference chunk that contains the answer to a question."""

    id: str
    metadata: dict[str, Any]
    content: str


class QADataPoint(TypedDict):
    """A single question-answer data point with reference chunks."""

    question: str
    answer: str
    reference_chunks: list[ReferenceChunk]
    qa_type: str  # "single_hop" or "multi_hop"
