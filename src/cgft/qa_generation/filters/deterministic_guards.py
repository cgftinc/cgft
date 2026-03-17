"""Deterministic quality guards for CgftPipeline."""

from __future__ import annotations

import re
from typing import Any

from cgft.qa_generation.cgft_models import CgftContext, DeterministicGuardsConfig
from cgft.qa_generation.generated_qa import FilterVerdict, GeneratedQA

# Patterns indicating a meta-question about generation failure rather than a real domain question
_META_QUESTION_PATTERNS = [
    re.compile(
        r"\b(can'?t|cannot|unable to|not possible to)\s+(create|generate|form|construct)", re.I
    ),
    re.compile(r"\bno valid\s+(multi[- ]?hop\s+)?question", re.I),
    re.compile(r"\b(why\s+)?(can'?t|cannot)\s+I\s+(create|generate|form)", re.I),
    re.compile(r"\bchunks?\s+(do not|don'?t|lack|cannot)\s+(support|provide|contain)", re.I),
    re.compile(r"\binsufficient\s+(interconnected\s+)?(evidence|information|data|chunks)", re.I),
    re.compile(r"\black(s|ing)?\s+(realistic\s+)?(connection|link|evidence)", re.I),
    re.compile(r"\bnot possible (to generate|with (the\s+)?provided)", re.I),
]

_META_ANSWER_PATTERNS = [
    re.compile(r"^no valid (answer|question|response)", re.I),
    re.compile(r"^(chunks?|evidence)\s+(do not|don'?t)\s+support", re.I),
    re.compile(r"^insufficient\b", re.I),
    re.compile(r"^cannot (be )?(determined|answered|generated)", re.I),
    re.compile(r"^not possible\b", re.I),
    re.compile(r"\bmulti[- ]?hop chaining\b", re.I),
    re.compile(r"^(no|lacking)\s+(valid\s+)?(evidence|connection|link)", re.I),
]


def _is_meta_question(question: str, answer: str) -> bool:
    """Check if the QA is a meta-statement about generation failure."""
    for pattern in _META_QUESTION_PATTERNS:
        if pattern.search(question):
            return True
    for pattern in _META_ANSWER_PATTERNS:
        if pattern.search(answer):
            return True
    return False


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

            reason = self._validate(
                question=question,
                answer=answer,
                reference_chunks=reference_chunks,
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
    ) -> str | None:
        if len(question) < self.cfg.min_question_chars:
            return "question_too_short"
        if len(answer) < self.cfg.min_answer_chars:
            return "answer_too_short"
        if len(reference_chunks) < self.cfg.min_reference_chunks:
            return "insufficient_reference_chunks"
        if _is_meta_question(question, answer):
            return "meta_question_about_generation"
        return None
