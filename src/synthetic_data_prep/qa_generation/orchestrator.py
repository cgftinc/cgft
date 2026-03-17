"""Pipeline orchestrator for modular QA generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synthetic_data_prep.qa_generation.protocols import (
        Filter,
        Formatter,
        Generator,
        Regenerator,
    )

from synthetic_data_prep.qa_generation.generated_qa import GeneratedQA

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates generate -> filter -> regenerate -> format.

    Supports two modes controlled by ``max_rounds``:

    - **Batch** (``max_rounds=0``, default): generate all items, run one
      filter pass, one regenerate pass on items needing refinement, then
      re-filter the regenerated items and format.
    - **Iterative** (``max_rounds > 0``): loop filter -> regenerate up to
      ``max_rounds`` times, allowing items to be refined multiple times
      (used by SAGE-style pipelines).

    Args:
        generator: Produces initial QA items.
        filter: Annotates items with verdicts (passed / rejected / needs_refinement).
        regenerator: Refines items flagged as needs_refinement.
        formatter: Converts passed items into a final output bundle.
        max_rounds: Number of filter-regenerate cycles. 0 means batch mode.
    """

    def __init__(
        self,
        generator: Generator,
        filter: Filter,
        regenerator: Regenerator,
        formatter: Formatter,
        *,
        max_rounds: int = 0,
    ) -> None:
        self.generator = generator
        self.filter = filter
        self.regenerator = regenerator
        self.formatter = formatter
        self.max_rounds = max_rounds

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the full pipeline and return formatted output."""
        items = self.generator.generate(context)
        logger.info("Generated %d items", len(items))

        if self.max_rounds == 0:
            items = self._run_batch(items, context)
        else:
            items = self._run_iterative(items, context)

        passed = [i for i in items if i.is_passed]
        logger.info(
            "Pipeline complete: %d passed, %d rejected, %d unresolved",
            len(passed),
            sum(1 for i in items if i.is_rejected),
            sum(1 for i in items if i.needs_refinement),
        )
        return self.formatter.format(passed, context)

    def _run_batch(
        self, items: list[GeneratedQA], context: dict[str, Any]
    ) -> list[GeneratedQA]:
        """Batch mode: filter -> regenerate needs_refinement -> re-filter regenerated."""
        items = self.filter.filter(items, context)

        to_refine = [i for i in items if i.needs_refinement]
        if to_refine:
            logger.info("Regenerating %d items", len(to_refine))
            refined = self.regenerator.regenerate(to_refine, context)
            refined = self.filter.filter(refined, context)

            # Replace refined items back into the main list
            refined_ids = {id(original) for original in to_refine}
            items = [i for i in items if id(i) not in refined_ids] + refined

        return items

    def _run_iterative(
        self, items: list[GeneratedQA], context: dict[str, Any]
    ) -> list[GeneratedQA]:
        """Iterative mode: loop filter -> regenerate up to max_rounds."""
        for round_num in range(1, self.max_rounds + 1):
            items = self.filter.filter(items, context)

            to_refine = [i for i in items if i.needs_refinement]
            if not to_refine:
                logger.info("All items resolved after %d rounds", round_num)
                break

            logger.info("Round %d: regenerating %d items", round_num, len(to_refine))
            refined = self.regenerator.regenerate(to_refine, context)

            # Replace refined items back into the main list
            refined_ids = {id(original) for original in to_refine}
            items = [i for i in items if id(i) not in refined_ids] + refined
        else:
            # Final filter pass after all rounds exhausted
            items = self.filter.filter(items, context)

        return items
