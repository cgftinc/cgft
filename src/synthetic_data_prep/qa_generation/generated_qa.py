"""Data model for QA items as they flow through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synthetic_data_prep.qa_generation.models import QADataPoint


@dataclass
class FilterVerdict:
    """Result of a filter step applied to a GeneratedQA item.

    This is the boundary type between Filter and Regenerator — it carries
    rich metadata that the Regenerator needs to decide how to refine an item.

    Attributes:
        status: One of "passed", "rejected", or "needs_refinement".
        reason: Machine-readable code (e.g. "too_easy", "incorrect_answer").
        reasoning: Human-readable explanation of why this verdict was given.
        metadata: Pipeline-specific data (for example, refinement hints and
            retrieval diagnostics).
    """

    status: str  # "passed" | "rejected" | "needs_refinement"
    reason: str
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedQA:
    """A QA item flowing through the pipeline, accumulating metadata at each step.

    Wraps an existing ``QADataPoint`` and carries filter verdicts and
    regeneration history so that downstream steps can make informed decisions.

    Attributes:
        qa: The underlying question-answer data point.
        generation_metadata: Arbitrary metadata from the generation step.
        filter_verdict: Set by a Filter; read (and cleared) by a Regenerator.
        regeneration_history: Append-only log of past regeneration attempts.
    """

    qa: QADataPoint
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    filter_verdict: FilterVerdict | None = None
    regeneration_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_passed(self) -> bool:
        """True when the item has been explicitly marked as passed by a filter."""
        return self.filter_verdict is not None and self.filter_verdict.status == "passed"

    @property
    def is_rejected(self) -> bool:
        """True when the item has been explicitly rejected by a filter."""
        return self.filter_verdict is not None and self.filter_verdict.status == "rejected"

    @property
    def needs_refinement(self) -> bool:
        """True when a filter has flagged this item for regeneration."""
        return self.filter_verdict is not None and self.filter_verdict.status == "needs_refinement"

    def to_qa_data_point(self) -> QADataPoint:
        """Return the underlying QADataPoint."""
        return self.qa
