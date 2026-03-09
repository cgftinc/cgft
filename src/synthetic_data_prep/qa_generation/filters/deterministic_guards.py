"""Deterministic quality guards for CgftPipeline."""

from __future__ import annotations

from typing import Any

from synthetic_data_prep.qa_generation.cgft_models import CgftContext, DeterministicGuardsConfig
from synthetic_data_prep.qa_generation.generated_qa import FilterVerdict, GeneratedQA
from synthetic_data_prep.qa_generation.style_controls import classify_query_style


class DeterministicGuardsFilter:
    """Applies deterministic validation checks before expensive evaluation."""

    def __init__(self, cfg: DeterministicGuardsConfig) -> None:
        self.cfg = cfg

    def evaluate(self, items: list[GeneratedQA], context: CgftContext) -> list[GeneratedQA]:
        if not self.cfg.enabled:
            return items

        stats = context.setdefault("deterministic_guard_stats", {"passed": 0, "rejected": 0})
        for item in items:
            if item.filter_verdict is not None:
                continue

            question = str(item.qa.get("question", "")).strip()
            answer = str(item.qa.get("answer", "")).strip()
            reference_chunks = list(item.qa.get("reference_chunks", []) or [])
            target_style = str(item.generation_metadata.get("style_target", "")).strip()
            observed_style = classify_query_style(question) if question else ""

            reason = self._validate(
                question=question,
                answer=answer,
                reference_chunks=reference_chunks,
                target_style=target_style,
                observed_style=observed_style,
            )
            if reason is None:
                item.filter_verdict = FilterVerdict(
                    status="passed",
                    reason="deterministic_passed",
                    reasoning="Passed deterministic guards.",
                    metadata={
                        "filter_mode": "deterministic_guards",
                        "reason_code": "deterministic_passed",
                        "confidence": 1.0,
                        "retrieval_query": str(item.qa.get("retrieval_query", "")).strip() or question,
                        "ref_overlap_ratio": None,
                        "feedback_type": None,
                        "refinement_hint": None,
                    },
                )
                stats["passed"] = int(stats.get("passed", 0)) + 1
                continue

            item.filter_verdict = FilterVerdict(
                status="rejected",
                reason="deterministic_rejected",
                reasoning=reason,
                metadata={
                    "filter_mode": "deterministic_guards",
                    "reason_code": "deterministic_rejected",
                    "confidence": 1.0,
                    "retrieval_query": str(item.qa.get("retrieval_query", "")).strip() or question,
                    "ref_overlap_ratio": None,
                    "feedback_type": None,
                    "refinement_hint": reason,
                },
            )
            stats["rejected"] = int(stats.get("rejected", 0)) + 1
        return items

    def _validate(
        self,
        *,
        question: str,
        answer: str,
        reference_chunks: list[Any],
        target_style: str,
        observed_style: str,
    ) -> str | None:
        if len(question) < self.cfg.min_question_chars:
            return "question_too_short"
        if len(answer) < self.cfg.min_answer_chars:
            return "answer_too_short"
        if len(reference_chunks) < self.cfg.min_reference_chunks:
            return "insufficient_reference_chunks"
        if (
            self.cfg.enforce_style_mismatch_guard
            and target_style
            and observed_style
            and target_style != observed_style
        ):
            return "style_mismatch"
        return None
