"""Data models for QA generation datasets."""

from typing import Any, Optional, TypedDict

from typing_extensions import NotRequired


class ReferenceChunk(TypedDict):
    """A reference chunk that contains the answer to a question."""

    id: str
    metadata: dict[str, Any]
    content: str


class QADataPoint(TypedDict):
    """A single question-answer data point with reference chunks."""

    question: str
    task_id: NotRequired[str]
    user_question: NotRequired[str]
    retrieval_query: NotRequired[str]
    style_target: NotRequired[str]
    style_observed: NotRequired[str]
    answer: str
    reference_chunks: list[ReferenceChunk]
    qa_type: str  # "single_hop" or "multi_hop"
    min_hop_count: Optional[int]
    is_co_located: Optional[bool]

    # Populated by pipeline steps
    filter_status: Optional[str]
    filter_reasoning: Optional[str]
    no_context_answer: Optional[str]
    eval_scores: dict

    # Populated by grounding filter: subset of reference_chunks verified as supporting the answer
    verified_reference_chunks: NotRequired[list[ReferenceChunk]]

    # @model_validator(mode='after')
    # def derive_additional_fields(self):
    #     super()
    #     hop_map = {"single_hop": 1, "multi_hop": 2}

    #     # Co-location detection — extract document IDs from metadata if available
    #     doc_ids = [
    #         c.metadata.get("file") 
    #         for c in self.reference_chunks
    #         if c.metadata.get("file") is not None
    #     ]
    #     is_co_located = (
    #         len(set(doc_ids)) == 1 
    #         if len(doc_ids) == len(self.reference_chunks)  # only if all chunks have doc IDs
    #         else None
    #     )

    #     self.min_hop_count=hop_map.get(self.qa_type)
    #     self.is_co_located=is_co_located
    #     return self
