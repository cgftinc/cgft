"""GenerationRetryRegenerator — replace failed items with fresh generator output."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synthetic_data_prep.qa_generation.protocols import Generator

from synthetic_data_prep.qa_generation.generated_qa import GeneratedQA

logger = logging.getLogger(__name__)


class GenerationRetryRegenerator:
    """Discards ``needs_refinement`` items and replaces them with fresh generator output.

    Unlike ``SageFeedbackRegenerator`` which rewrites questions based on execution
    feedback, this regenerator simply calls the original generator again and takes
    replacement items from the fresh batch.  Fresh items re-enter the filter with
    ``filter_verdict=None`` in the next round.

    Args:
        generator: The same ``Generator`` instance used by the pipeline.
    """

    def __init__(self, generator: Generator) -> None:
        self.generator = generator

    def regenerate(
        self, items: list[GeneratedQA], context: dict[str, Any]
    ) -> list[GeneratedQA]:
        failed = [i for i in items if i.needs_refinement]
        unchanged = [i for i in items if not i.needs_refinement]

        if not failed:
            return items

        n_needed = len(failed)
        logger.info("Regenerating %d items via fresh generation", n_needed)

        fresh_items = self.generator.generate(context)

        replacements = fresh_items[:n_needed]
        for replacement in replacements:
            refinements = replacement.generation_metadata.get("question_refinements", 0)
            replacement.generation_metadata["question_refinements"] = refinements + 1

        logger.info(
            "Got %d replacements for %d failed items", len(replacements), n_needed
        )

        return unchanged + replacements
