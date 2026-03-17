"""Base question transformer with shared metadata/stats handling."""

from __future__ import annotations

import logging
from typing import Any

from cgft.qa_generation.cgft_models import CgftContext, TransformationConfig
from cgft.qa_generation.generated_qa import GeneratedQA
from cgft.qa_generation.style_controls import classify_query_style
from cgft.qa_generation.transformers.validator import LLMSanityValidator

logger = logging.getLogger(__name__)


class BaseQuestionTransformer:
    """Shared transform flow: mutate, validate, annotate style, and record metadata."""

    stats_mode = "base"

    def __init__(self, cfg: TransformationConfig, *, enable_validation: bool = True) -> None:
        self.cfg = cfg
        self.validator: LLMSanityValidator | None = None
        if enable_validation and cfg.validation_enabled:
            self.validator = LLMSanityValidator(cfg)

    def transform(self, items: list[GeneratedQA], context: CgftContext) -> list[GeneratedQA]:
        stats = self._stats_bucket(context)
        for item in items:
            stats["processed"] = int(stats.get("processed", 0)) + 1
            original_question = str(item.qa.get("question", "")).strip()
            if self.cfg.preserve_original_in_metadata and original_question:
                item.generation_metadata.setdefault("original_question", original_question)

            try:
                candidate_question, step_meta = self._transform_question(
                    item,
                    context=context,
                    original_question=original_question,
                )
            except Exception:
                logger.exception("Question transformation failed for mode '%s'.", self.stats_mode)
                stats["errors"] = int(stats.get("errors", 0)) + 1
                candidate_question = original_question
                step_meta = {"error": "transform_exception"}

            candidate_question = str(candidate_question or "").strip() or original_question
            changed = bool(candidate_question and candidate_question != original_question)
            if changed:
                self._set_question_fields(item, candidate_question)

            validation_reason = "not_run"
            if changed and self.validator is not None:
                is_valid, validation_reason = self.validator.is_valid(item, context)
                if not is_valid:
                    self._set_question_fields(item, original_question)
                    changed = False
                    stats["validation_reverted"] = int(stats.get("validation_reverted", 0)) + 1

            if changed:
                stats["mutated"] = int(stats.get("mutated", 0)) + 1
            else:
                stats["unchanged"] = int(stats.get("unchanged", 0)) + 1

            final_question = str(item.qa.get("question", "")).strip()
            observed_style = self._annotate_observed_style(item, final_question)
            self._record_step_metadata(
                item,
                changed=changed,
                style_observed=observed_style,
                validation_reason=validation_reason,
                original_question=original_question,
                final_question=final_question,
                step_meta=step_meta,
            )
        return items

    def _transform_question(
        self,
        item: GeneratedQA,
        *,
        context: CgftContext,
        original_question: str,
    ) -> tuple[str, dict[str, Any]]:
        del item, context
        return original_question, {}

    def _stats_bucket(self, context: CgftContext) -> dict[str, Any]:
        stats = context.setdefault("transformation_stats", {})
        if not isinstance(stats, dict):
            stats = {}
            context["transformation_stats"] = stats
        mode_stats = stats.setdefault(self.stats_mode, {})
        if not isinstance(mode_stats, dict):
            mode_stats = {}
            stats[self.stats_mode] = mode_stats
        mode_stats.setdefault("processed", 0)
        mode_stats.setdefault("mutated", 0)
        mode_stats.setdefault("unchanged", 0)
        mode_stats.setdefault("validation_reverted", 0)
        mode_stats.setdefault("errors", 0)
        return mode_stats

    @staticmethod
    def _set_question_fields(item: GeneratedQA, question: str) -> None:
        item.qa["question"] = question
        item.qa["user_question"] = question

    @staticmethod
    def _annotate_observed_style(item: GeneratedQA, question: str) -> str:
        observed = classify_query_style(question)
        eval_scores = dict(item.qa.get("eval_scores", {}) or {})
        eval_scores.pop("query_style_target", None)
        eval_scores["query_style_observed"] = observed
        item.qa["eval_scores"] = eval_scores
        item.qa["style_observed"] = observed
        return observed

    def _record_step_metadata(
        self,
        item: GeneratedQA,
        *,
        changed: bool,
        style_observed: str,
        validation_reason: str,
        original_question: str,
        final_question: str,
        step_meta: dict[str, Any],
    ) -> None:
        transform_meta = dict(item.generation_metadata.get("transformation", {}) or {})
        steps = list(transform_meta.get("steps", []) or [])
        entry: dict[str, Any] = {
            "mode": self.stats_mode,
            "changed": changed,
            "style_observed": style_observed,
            "validation_reason": validation_reason,
        }
        if step_meta:
            entry.update(step_meta)
        if original_question:
            entry["original_question"] = original_question
        if final_question:
            entry["final_question"] = final_question
        steps.append(entry)
        transform_meta["steps"] = steps
        transform_meta["style_observed"] = style_observed
        transform_meta["changed"] = any(bool(step.get("changed")) for step in steps)
        if "target_style" in entry and entry["target_style"]:
            transform_meta["target_style"] = entry["target_style"]
        item.generation_metadata["transformation"] = transform_meta
