"""Feedback-based same-anchor refiner for CgftPipeline."""

from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from synthetic_data_prep.qa_generation.cgft_models import CgftContext, RefinementConfig
from synthetic_data_prep.qa_generation.generated_qa import FilterVerdict, GeneratedQA
from synthetic_data_prep.qa_generation.helpers import render_template
from synthetic_data_prep.qa_generation.models import QADataPoint
from synthetic_data_prep.qa_generation.style_controls import classify_query_style

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_refined_qa(raw_text: str) -> tuple[str, str]:
    text = (raw_text or "").strip()
    if not text:
        return "", ""
    for candidate in (text, *_JSON_BLOCK_RE.findall(text)):
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if question and answer:
            return question, answer
    return "", ""


class FeedbackRefiner:
    """Refines needs-refinement items while preserving anchor and target intent."""

    def __init__(self, cfg: RefinementConfig) -> None:
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def refine(self, items: list[GeneratedQA], context: CgftContext) -> list[GeneratedQA]:
        if not self.cfg.enabled:
            return items

        refined: list[GeneratedQA] = []
        for item in items:
            if not item.needs_refinement:
                refined.append(item)
                continue
            refined.append(self._refine_one(item))
        return refined

    def _refine_one(self, item: GeneratedQA) -> GeneratedQA:
        meta = dict(item.generation_metadata)
        current_count = int(meta.get("refinement_count", 0))
        verdict = item.filter_verdict
        feedback_reason = verdict.reasoning if verdict is not None else ""
        style_target = str(meta.get("style_target", "")).strip() or str(
            (item.qa.get("eval_scores", {}) or {}).get("query_style_target", "")
        ).strip()
        qa_type = str(meta.get("qa_type_target", "")).strip() or str(item.qa.get("qa_type", "")).strip()

        if current_count >= self.cfg.max_refinements_per_item:
            item.filter_verdict = FilterVerdict(
                status="rejected",
                reason="refinement_budget_exhausted",
                reasoning=(
                    f"Reached max refinements per item ({self.cfg.max_refinements_per_item})."
                ),
                metadata={
                    "filter_mode": "refiner",
                    "reason_code": "refinement_budget_exhausted",
                    "confidence": 1.0,
                    "retrieval_query": str(item.qa.get("question", "")).strip(),
                    "ref_overlap_ratio": None,
                    "feedback_type": None,
                    "refinement_hint": None,
                },
            )
            return item

        prompt = render_template(
            self.cfg.prompt_template,
            {
                "qa_type": qa_type or "unknown",
                "style_target": style_target or "natural",
                "feedback": feedback_reason,
                "question": str(item.qa.get("question", "")).strip(),
                "answer": str(item.qa.get("answer", "")).strip(),
            },
        )
        response = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": "You improve QA pairs for retrieval training."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_completion_tokens=900,
        )
        raw = response.choices[0].message.content or ""
        question, answer = _parse_refined_qa(raw)
        if not question or not answer:
            item.filter_verdict = FilterVerdict(
                status="rejected",
                reason="refinement_parse_failed",
                reasoning="Refinement response could not be parsed.",
                metadata={
                    "filter_mode": "refiner",
                    "reason_code": "refinement_parse_failed",
                    "confidence": 0.0,
                    "retrieval_query": str(item.qa.get("question", "")).strip(),
                    "ref_overlap_ratio": None,
                    "feedback_type": None,
                    "refinement_hint": None,
                },
            )
            return item

        updated_eval_scores = dict(item.qa.get("eval_scores", {}) or {})
        if style_target:
            updated_eval_scores["query_style_target"] = style_target
        updated_eval_scores["query_style_observed"] = classify_query_style(question)

        refined_qa: QADataPoint = {
            "question": question,
            "answer": answer,
            "reference_chunks": list(item.qa.get("reference_chunks", [])),
            "qa_type": str(item.qa.get("qa_type", qa_type or "unknown")),
            "min_hop_count": item.qa.get("min_hop_count"),
            "is_co_located": item.qa.get("is_co_located"),
            "filter_status": None,
            "filter_reasoning": None,
            "no_context_answer": None,
            "eval_scores": updated_eval_scores,
        }
        meta["refinement_count"] = current_count + 1
        refined_item = GeneratedQA(
            qa=refined_qa,
            generation_metadata=meta,
            filter_verdict=None,
            regeneration_history=item.regeneration_history
            + [{"type": "same_anchor_feedback", "round": current_count + 1}],
        )
        return refined_item
