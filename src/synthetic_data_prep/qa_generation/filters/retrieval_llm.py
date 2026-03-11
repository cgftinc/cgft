"""Retrieval + LLM-judge too-easy filter for CgftPipeline."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from synthetic_data_prep.chunkers.models import Chunk
from synthetic_data_prep.qa_generation.cgft_models import (
    DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT,
    DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE,
    CgftContext,
    RetrievalLLMFilterConfig,
)
from synthetic_data_prep.qa_generation.generated_qa import FilterVerdict, GeneratedQA
from synthetic_data_prep.qa_generation.retrieval_query import QueryRewriteConfig, resolve_retrieval_query

logger = logging.getLogger(__name__)

_DUMMY_CHUNK = Chunk(content="", metadata=(("_dummy", True),))

_FILTER_MODE = "retrieval_too_easy_llm"
_JUDGE_SYSTEM_PROMPT = DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT
_JUDGE_USER_TEMPLATE = DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE
_JUDGE_TAG_TOO_EASY = "too_easy_lexical"
_JUDGE_TAG_UNSUPPORTED = "unsupported"
_JUDGE_TAG_PASS = "challenging_answerable_pass"
_JUDGE_TAG_UNKNOWN = "unknown"
_FAILURE_TYPE_TOO_EASY = "too_easy"
_FAILURE_TYPE_NONE = "none"


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class RetrievalLLMFilter:
    """Flags QA pairs that appear too easy for naive retrieval."""

    def __init__(
        self,
        *,
        chunk_source: Any,
        cfg: RetrievalLLMFilterConfig,
    ) -> None:
        self.chunk_source = chunk_source
        self.cfg = cfg
        self.judge_client = OpenAI(
            api_key=cfg.judge_api_key,
            base_url=cfg.judge_base_url,
        )

    def evaluate(self, items: list[GeneratedQA], context: CgftContext) -> list[GeneratedQA]:
        if not self.cfg.enabled:
            return items

        max_refinements = context.config.refinement.max_refinements_per_item
        stats_key = str(self.cfg.stats_key or "").strip() or "retrieval_too_easy_filter_stats"
        stats = context.setdefault(
            stats_key,
            {
                "passed": 0,
                "needs_refinement": 0,
                "rejected": 0,
                "errors": 0,
                "judge_calls": 0,
                "judge_answerable_true": 0,
                "judge_answerable_false": 0,
                "judge_reason_tags": {
                    _JUDGE_TAG_TOO_EASY: 0,
                    _JUDGE_TAG_UNSUPPORTED: 0,
                    _JUDGE_TAG_PASS: 0,
                    _JUDGE_TAG_UNKNOWN: 0,
                },
                "too_easy_total": 0,
                "too_easy_due_to_overlap_pre_gate": 0,
                "too_easy_due_to_judge": 0,
                "overlap_threshold_triggered": 0,
                "too_easy_overlap_threshold_triggered": 0,
            },
        )
        self._ensure_stats_shape(stats)
        rewrite_cfg = context.config.filtering.query_rewrite

        for item in items:
            if item.filter_verdict is not None and not item.is_passed:
                continue

            try:
                verdict = self._evaluate_item(
                    item,
                    max_refinements=max_refinements,
                    rewrite_cfg=rewrite_cfg,
                )
            except Exception:
                logger.exception("RetrievalLLMFilter failed for one item")
                query = resolve_retrieval_query(item.qa, rewrite_cfg=rewrite_cfg)
                verdict = FilterVerdict(
                    status="passed",
                    reason="retrieval_filter_error",
                    reasoning="Filter error; passed by default.",
                    metadata={
                        "filter_mode": _FILTER_MODE,
                        "reason_code": "filter_error",
                        "confidence": 0.0,
                        "retrieval_query": query,
                        "ref_overlap_ratio": 0.0,
                        "failure_type": _FAILURE_TYPE_NONE,
                        "force_reanchor": False,
                        "feedback_type": None,
                        "refinement_hint": None,
                        "judge_called": False,
                    },
                )
                stats["errors"] = int(stats.get("errors", 0)) + 1

            item.filter_verdict = verdict
            if verdict.status == "passed":
                stats["passed"] = int(stats.get("passed", 0)) + 1
            elif verdict.status == "needs_refinement":
                stats["needs_refinement"] = int(stats.get("needs_refinement", 0)) + 1
            else:
                stats["rejected"] = int(stats.get("rejected", 0)) + 1
            self._record_diagnostics(stats, verdict)

        return items

    def _evaluate_item(
        self,
        item: GeneratedQA,
        *,
        max_refinements: int,
        rewrite_cfg: QueryRewriteConfig,
    ) -> FilterVerdict:
        query = resolve_retrieval_query(item.qa, rewrite_cfg=rewrite_cfg)
        if not query:
            return FilterVerdict(
                status="rejected",
                reason="retrieval_filter_rejected",
                reasoning="Missing question/retrieval_query.",
                metadata={
                    "filter_mode": _FILTER_MODE,
                    "reason_code": "missing_query",
                    "confidence": 1.0,
                    "retrieval_query": "",
                    "ref_overlap_ratio": 0.0,
                    "failure_type": _FAILURE_TYPE_NONE,
                    "force_reanchor": False,
                    "feedback_type": None,
                    "refinement_hint": None,
                    "judge_called": False,
                },
            )

        search_results = self.chunk_source.search_related(
            source=_DUMMY_CHUNK,
            queries=[query],
            top_k=self.cfg.top_k,
        )
        retrieved_chunks: list[Chunk] = [row["chunk"] for row in search_results if "chunk" in row]
        ref_chunks = list(item.qa.get("reference_chunks", []) or [])
        overlap_ratio, matched_reference_ids = self._compute_overlap(ref_chunks, retrieved_chunks)
        overlap_triggered = overlap_ratio >= self.cfg.overlap_threshold
        too_easy_overlap_triggered = overlap_ratio >= self.cfg.too_easy_overlap_threshold
        refinements = int(item.generation_metadata.get("refinement_count", 0))

        shared_metadata = {
            "filter_mode": _FILTER_MODE,
            "retrieval_query": query,
            "ref_overlap_ratio": overlap_ratio,
            "overlap_threshold": self.cfg.overlap_threshold,
            "overlap_triggered": overlap_triggered,
            "too_easy_overlap_threshold": self.cfg.too_easy_overlap_threshold,
            "too_easy_overlap_triggered": too_easy_overlap_triggered,
            "too_easy_confidence_threshold": self.cfg.too_easy_confidence_threshold,
            "force_reanchor": False,
            "matched_reference_ids": matched_reference_ids,
            "matched_reference_chunks": len(matched_reference_ids),
            "retrieved_chunk_count": len(retrieved_chunks),
            "top_score": search_results[0].get("max_score", 0.0) if search_results else 0.0,
        }

        if too_easy_overlap_triggered:
            metadata = {
                **shared_metadata,
                "reason_code": "naive_retrieval_sufficient",
                "confidence": 1.0,
                "failure_type": _FAILURE_TYPE_TOO_EASY,
                "feedback_type": "same_anchor_feedback",
                "refinement_hint": "Increase retrieval difficulty and avoid directly retrievable terms.",
                "judge_called": False,
                "judge_answerable": None,
                "judge_reasoning": "",
                "judge_reason_tag": _JUDGE_TAG_UNKNOWN,
                "too_easy_source": "overlap_pre_gate",
            }
            return self._too_easy_verdict(
                metadata=metadata,
                overlap_ratio=overlap_ratio,
                refinements=refinements,
                max_refinements=max_refinements,
            )

        judge_result = self._run_judge(item, retrieved_chunks)
        answerable = bool(judge_result.get("answerable", False))
        confidence = _safe_float(judge_result.get("confidence", 0.0), default=0.0)
        judge_reasoning = str(judge_result.get("reasoning", "")).strip()
        judge_reason_tag = self._extract_judge_reason_tag(
            reason_tag=judge_result.get("reason_tag", ""),
            reasoning=judge_reasoning,
        )
        too_easy_high_confidence = confidence >= self.cfg.too_easy_confidence_threshold
        too_easy_due_to_judge = (
            judge_reason_tag == _JUDGE_TAG_TOO_EASY and too_easy_high_confidence
        )
        too_easy_calibrated_out = (
            judge_reason_tag == _JUDGE_TAG_TOO_EASY and not too_easy_due_to_judge
        )

        if too_easy_due_to_judge:
            metadata = {
                **shared_metadata,
                "reason_code": "naive_retrieval_sufficient",
                "confidence": confidence,
                "failure_type": _FAILURE_TYPE_TOO_EASY,
                "feedback_type": "same_anchor_feedback",
                "refinement_hint": "Increase retrieval difficulty and avoid directly retrievable terms.",
                "judge_called": True,
                "judge_answerable": answerable,
                "judge_reasoning": judge_reasoning,
                "judge_reason_tag": judge_reason_tag,
                "too_easy_high_confidence": too_easy_high_confidence,
                "too_easy_calibrated_out": False,
                "too_easy_source": "judge",
            }
            return self._too_easy_verdict(
                metadata=metadata,
                overlap_ratio=overlap_ratio,
                refinements=refinements,
                max_refinements=max_refinements,
            )

        return FilterVerdict(
            status="passed",
            reason="retrieval_filter_passed",
            reasoning=(
                "Borderline too-easy signal did not meet confidence threshold. "
                if too_easy_calibrated_out
                else "Not classified as too easy by overlap gate or judge. "
            )
            + f"(overlap={overlap_ratio:.2f}, confidence={confidence:.2f}).",
            metadata={
                **shared_metadata,
                "reason_code": "retrieval_difficulty_passed",
                "confidence": confidence,
                "failure_type": _FAILURE_TYPE_NONE,
                "feedback_type": None,
                "refinement_hint": None,
                "judge_called": True,
                "judge_answerable": answerable,
                "judge_reasoning": judge_reasoning,
                "judge_reason_tag": judge_reason_tag,
                "too_easy_high_confidence": too_easy_high_confidence,
                "too_easy_calibrated_out": too_easy_calibrated_out,
                "too_easy_source": "none",
            },
        )

    @staticmethod
    def _too_easy_verdict(
        *,
        metadata: dict[str, Any],
        overlap_ratio: float,
        refinements: int,
        max_refinements: int,
    ) -> FilterVerdict:
        if refinements < max_refinements:
            return FilterVerdict(
                status="needs_refinement",
                reason="retrieval_filter_needs_refinement",
                reasoning=(
                    "Naive retrieval appears sufficient "
                    f"(overlap={overlap_ratio:.2f}, source={metadata.get('too_easy_source', 'unknown')})."
                ),
                metadata=metadata,
            )

        return FilterVerdict(
            status="rejected",
            reason="retrieval_filter_rejected",
            reasoning=(
                "Refinement budget exhausted for failure_type=too_easy "
                f"({refinements}/{max_refinements})."
            ),
            metadata=metadata,
        )

    @staticmethod
    def _compute_overlap(
        ref_chunks: list[dict[str, Any]],
        retrieved_chunks: list[Chunk],
    ) -> tuple[float, list[str]]:
        if not ref_chunks:
            return 0.0, []
        retrieved_hashes = {chunk.hash for chunk in retrieved_chunks}
        matched: list[str] = []
        for ref in ref_chunks:
            ref_id = str(ref.get("id", "")).strip()
            if ref_id and ref_id in retrieved_hashes:
                matched.append(ref_id)
        return len(matched) / max(1, len(ref_chunks)), matched

    def _run_judge(self, item: GeneratedQA, retrieved_chunks: list[Chunk]) -> dict[str, Any]:
        if not retrieved_chunks:
            return {"answerable": False, "confidence": 1.0, "reasoning": "No chunks retrieved."}

        chunks_text = "\n---\n".join(
            f"[Chunk {idx + 1}]\n{chunk.content}" for idx, chunk in enumerate(retrieved_chunks)
        )
        prompt_vars = {
            "question": item.qa.get("question", ""),
            "answer": item.qa.get("answer", ""),
            "chunks_text": chunks_text,
        }
        user_template = str(self.cfg.judge_user_template or "").strip() or _JUDGE_USER_TEMPLATE
        try:
            user_prompt = user_template.format(**prompt_vars)
        except KeyError:
            logger.warning(
                "Retrieval judge user template requires unknown placeholders; "
                "falling back to default template."
            )
            user_prompt = _JUDGE_USER_TEMPLATE.format(**prompt_vars)

        system_prompt = self.cfg.judge_system_prompt or _JUDGE_SYSTEM_PROMPT
        response = self.judge_client.chat.completions.create(
            model=self.cfg.judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return {}
            payload["reason_tag"] = self._extract_judge_reason_tag(
                reason_tag=payload.get("reason_tag", ""),
                reasoning=payload.get("reasoning", ""),
            )
            return payload
        except json.JSONDecodeError:
            logger.warning("Judge response parse failure: %s", raw[:240])
            return {"answerable": False, "confidence": 0.0, "reasoning": "parse_error"}

    @staticmethod
    def _extract_judge_reason_tag(*, reason_tag: Any = "", reasoning: Any = "") -> str:
        explicit = str(reason_tag or "").strip().lower()
        if explicit in {_JUDGE_TAG_TOO_EASY, _JUDGE_TAG_UNSUPPORTED, _JUDGE_TAG_PASS}:
            return explicit
        if explicit == "too_easy":
            return _JUDGE_TAG_TOO_EASY
        if explicit == "pass":
            return _JUDGE_TAG_PASS

        # When no reason_tag was provided (neutral prompt), return unknown immediately
        # without scanning reasoning text — avoids false substring matches like
        # "unsupported" appearing in natural reasoning prose.
        if not explicit:
            return _JUDGE_TAG_UNKNOWN

        # An explicit but unrecognized tag was given; try bracket-prefixed reasoning.
        text = str(reasoning or "").strip().lower()
        if text.startswith("[") and "]" in text:
            candidate = text[1 : text.find("]")].strip()
            if candidate in {_JUDGE_TAG_TOO_EASY, _JUDGE_TAG_UNSUPPORTED, _JUDGE_TAG_PASS}:
                return candidate

        if "too_easy_lexical" in text:
            return _JUDGE_TAG_TOO_EASY
        if "challenging_answerable_pass" in text:
            return _JUDGE_TAG_PASS
        if "unsupported" in text or "unanswerable" in text:
            return _JUDGE_TAG_UNSUPPORTED
        return _JUDGE_TAG_UNKNOWN

    @staticmethod
    def _ensure_stats_shape(stats: dict[str, Any]) -> None:
        stats.setdefault("passed", 0)
        stats.setdefault("needs_refinement", 0)
        stats.setdefault("rejected", 0)
        stats.setdefault("errors", 0)
        stats.setdefault("judge_calls", 0)
        stats.setdefault("judge_answerable_true", 0)
        stats.setdefault("judge_answerable_false", 0)
        stats.setdefault("too_easy_total", 0)
        stats.setdefault("too_easy_due_to_overlap_pre_gate", 0)
        stats.setdefault("too_easy_due_to_judge", 0)
        stats.setdefault("overlap_threshold_triggered", 0)
        stats.setdefault("too_easy_overlap_threshold_triggered", 0)

        tags = stats.setdefault("judge_reason_tags", {})
        if not isinstance(tags, dict):
            stats["judge_reason_tags"] = {}
            tags = stats["judge_reason_tags"]
        tags.setdefault(_JUDGE_TAG_TOO_EASY, 0)
        tags.setdefault(_JUDGE_TAG_UNSUPPORTED, 0)
        tags.setdefault(_JUDGE_TAG_PASS, 0)
        tags.setdefault(_JUDGE_TAG_UNKNOWN, 0)

    @staticmethod
    def _record_diagnostics(stats: dict[str, Any], verdict: FilterVerdict) -> None:
        metadata = verdict.metadata if isinstance(verdict.metadata, dict) else {}

        overlap_triggered = bool(metadata.get("overlap_triggered", False))
        if overlap_triggered:
            stats["overlap_threshold_triggered"] = int(stats.get("overlap_threshold_triggered", 0)) + 1
        too_easy_overlap_triggered = bool(metadata.get("too_easy_overlap_triggered", False))
        if too_easy_overlap_triggered:
            stats["too_easy_overlap_threshold_triggered"] = int(
                stats.get("too_easy_overlap_threshold_triggered", 0)
            ) + 1

        judge_called = bool(metadata.get("judge_called", False))
        if judge_called:
            stats["judge_calls"] = int(stats.get("judge_calls", 0)) + 1
            judge_answerable = metadata.get("judge_answerable")
            if isinstance(judge_answerable, bool):
                if judge_answerable:
                    stats["judge_answerable_true"] = int(stats.get("judge_answerable_true", 0)) + 1
                else:
                    stats["judge_answerable_false"] = int(stats.get("judge_answerable_false", 0)) + 1

            judge_reason_tag = (
                str(metadata.get("judge_reason_tag", "")).strip().lower() or _JUDGE_TAG_UNKNOWN
            )
            tags = stats.get("judge_reason_tags", {})
            if isinstance(tags, dict):
                if judge_reason_tag not in tags:
                    tags[judge_reason_tag] = 0
                tags[judge_reason_tag] = int(tags.get(judge_reason_tag, 0)) + 1

        failure_type = str(metadata.get("failure_type", "")).strip()
        if failure_type == _FAILURE_TYPE_TOO_EASY:
            stats["too_easy_total"] = int(stats.get("too_easy_total", 0)) + 1
            source = str(metadata.get("too_easy_source", "")).strip()
            if source == "overlap_pre_gate":
                stats["too_easy_due_to_overlap_pre_gate"] = int(
                    stats.get("too_easy_due_to_overlap_pre_gate", 0)
                ) + 1
            elif source == "judge":
                stats["too_easy_due_to_judge"] = int(
                    stats.get("too_easy_due_to_judge", 0)
                ) + 1
