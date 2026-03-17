"""Cgft-first protocol definitions for QA generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cgft.qa_generation.anchor_selector import AnchorBundle
    from cgft.qa_generation.cgft_models import CgftContext, GenerationTask
    from cgft.qa_generation.generated_qa import GeneratedQA


@runtime_checkable
class ChunkLinker(Protocol):
    """Selects related chunks and returns an anchor bundle for one task."""

    def link(
        self,
        primary_chunk: Any,
        *,
        target_qa_type: str | None = None,
        target_hop_count: int | None = None,
        corpus_pool: list[Any] | None = None,
    ) -> AnchorBundle: ...


@runtime_checkable
class QuestionGenerator(Protocol):
    """Generates initial QA candidates from task intents."""

    def generate(self, tasks: list[GenerationTask], context: CgftContext) -> list[GeneratedQA]: ...


@runtime_checkable
class LLMSupportedGenerator(QuestionGenerator, Protocol):
    """Direct-LLM generator specialization."""

    def generate(self, tasks: list[GenerationTask], context: CgftContext) -> list[GeneratedQA]: ...


@runtime_checkable
class LLMEnvSupportedGenerator(QuestionGenerator, Protocol):
    """Environment-backed rollout generator specialization."""

    def generate(self, tasks: list[GenerationTask], context: CgftContext) -> list[GeneratedQA]: ...


@runtime_checkable
class EvaluatorFilter(Protocol):
    """Evaluates QA candidates and annotates each with a FilterVerdict."""

    def evaluate(self, items: list[GeneratedQA], context: CgftContext) -> list[GeneratedQA]: ...


@runtime_checkable
class LLMBasedFilter(EvaluatorFilter, Protocol):
    """Direct-LLM evaluation filter specialization."""

    def evaluate(self, items: list[GeneratedQA], context: CgftContext) -> list[GeneratedQA]: ...


@runtime_checkable
class LLMEnvBasedFilter(EvaluatorFilter, Protocol):
    """Environment-backed rollout evaluation filter specialization."""

    def evaluate(self, items: list[GeneratedQA], context: CgftContext) -> list[GeneratedQA]: ...


@runtime_checkable
class Refiner(Protocol):
    """Refines items marked `needs_refinement` and returns updated items."""

    def refine(self, items: list[GeneratedQA], context: CgftContext) -> list[GeneratedQA]: ...


@runtime_checkable
class QuestionTransformer(Protocol):
    """Transforms question text without changing answers or reference chunks."""

    def transform(self, items: list[GeneratedQA], context: CgftContext) -> list[GeneratedQA]: ...


@runtime_checkable
class Formatter(Protocol):
    """Formats final pipeline output artifacts."""

    def format(self, items: list[GeneratedQA], context: CgftContext) -> dict[str, Any]: ...
