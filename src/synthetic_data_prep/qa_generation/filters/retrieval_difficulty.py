"""RetrievalDifficultyFilter — rejects questions answerable by a single naive BM25 search."""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from synthetic_data_prep.chunkers.models import Chunk
from synthetic_data_prep.qa_generation.generated_qa import FilterVerdict, GeneratedQA

logger = logging.getLogger(__name__)

_DUMMY_CHUNK = Chunk(content="", metadata=(("_dummy", True),))

_JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating whether retrieved text chunks contain ALL information \
needed to fully answer a question.

Rules:
- The answer must be FULLY supported by the retrieved chunks. Partial information is NOT enough.
- For multi-hop questions, ALL intermediate facts must be present in the chunks.
- Do not use your own knowledge — only consider what is explicitly in the chunks.

Respond with JSON only:
{"answerable": <bool>, "confidence": <float 0-1>, "reasoning": "<brief explanation>"}"""

_JUDGE_USER_TEMPLATE = """\
Question: {question}

Gold answer: {answer}

Retrieved chunks:
{chunks_text}

Can the question be fully answered using ONLY the retrieved chunks above?"""


class RetrievalDifficultyFilter:
    """Rejects questions that a single naive BM25 search can answer.

    Fills the gap between the zero-context ``LLMEquivalenceFilter`` and the
    full-rollout ``SageFilter``.  For each item, runs a BM25 search with the
    question, then asks an LLM judge whether the top-k results suffice.
    """

    def __init__(
        self,
        chunk_source: Any,
        judge_client: Any,
        judge_model: str,
        top_k: int = 5,
        overlap_threshold: float = 0.5,
        max_refinements: int | None = None,
        enabled: bool = True,
    ) -> None:
        self.chunk_source = chunk_source
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.top_k = top_k
        self.overlap_threshold = overlap_threshold
        self.max_refinements = max_refinements
        self.enabled = enabled

    def filter(
        self, items: list[GeneratedQA], context: dict[str, Any]
    ) -> list[GeneratedQA]:
        if not self.enabled:
            for item in items:
                if item.filter_verdict is None or item.is_passed:
                    item.filter_verdict = FilterVerdict(
                        status="passed",
                        reason="retrieval_difficulty_disabled",
                        reasoning="Retrieval difficulty filter is disabled.",
                    )
            return items

        stats: dict[str, int] = {
            "total": 0,
            "passed": 0,
            "rejected": 0,
            "errors": 0,
        }

        for item in items:
            # Skip items already rejected by a prior filter.
            # Items with no verdict yet (filter_verdict is None) are treated
            # as candidates so this filter can be used as the sole filter.
            if item.is_rejected:
                continue

            stats["total"] += 1

            try:
                verdict = self._evaluate_item(item)
                item.filter_verdict = verdict
                if verdict.status == "passed":
                    stats["passed"] += 1
                else:
                    stats["rejected"] += 1
            except Exception:
                logger.exception(
                    "Error evaluating item in RetrievalDifficultyFilter"
                )
                stats["errors"] += 1
                item.filter_verdict = FilterVerdict(
                    status="passed",
                    reason="retrieval_difficulty_error",
                    reasoning=(
                        "Error during retrieval difficulty evaluation; "
                        "passing by default."
                    ),
                )

        context["retrieval_difficulty_stats"] = stats
        logger.info(
            "Retrieval difficulty filter: passed=%d, rejected=%d, errors=%d",
            stats["passed"],
            stats["rejected"],
            stats["errors"],
        )
        return items

    def _evaluate_item(self, item: GeneratedQA) -> FilterVerdict:
        query = str(item.qa.get("retrieval_query") or item.qa["question"])

        search_results = self.chunk_source.search_related(
            source=_DUMMY_CHUNK, queries=[query], top_k=self.top_k
        )

        retrieved_chunks: list[Chunk] = [
            r["chunk"] for r in search_results if "chunk" in r
        ]
        top_score = (
            search_results[0]["max_score"] if search_results else 0.0
        )

        ref_chunks = cast(
            list[dict[str, Any]], item.qa.get("reference_chunks", [])
        )
        overlap_ratio, matched_ref_ids = self._compute_overlap(
            ref_chunks, retrieved_chunks
        )

        judge_result = self._run_judge(item, retrieved_chunks)
        judge_answerable = judge_result.get("answerable", False)
        judge_confidence = judge_result.get("confidence", 0.0)
        judge_reasoning = judge_result.get("reasoning", "")

        too_easy = (
            judge_answerable or overlap_ratio >= self.overlap_threshold
        )

        metadata = {
            "search_query": query,
            "retrieved_chunk_count": len(retrieved_chunks),
            "top_score": top_score,
            "judge_answerable": judge_answerable,
            "judge_confidence": judge_confidence,
            "judge_reasoning": judge_reasoning,
            "ref_overlap_ratio": overlap_ratio,
            "matched_ref_ids": matched_ref_ids,
        }

        if too_easy:
            return FilterVerdict(
                status="rejected",
                reason="naive_retrieval_sufficient",
                reasoning=(
                    f"Single BM25 search sufficient: "
                    f"judge_answerable={judge_answerable}, "
                    f"overlap={overlap_ratio:.2f} "
                    f"(threshold={self.overlap_threshold})"
                ),
                metadata=metadata,
            )

        return FilterVerdict(
            status="passed",
            reason="retrieval_difficulty_passed",
            reasoning=(
                f"Not answerable by naive retrieval: "
                f"judge_answerable={judge_answerable}, "
                f"overlap={overlap_ratio:.2f}"
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

        retrieved_hashes = {c.hash for c in retrieved_chunks}
        retrieved_contents = [c.content for c in retrieved_chunks]

        matched_ids: list[str] = []
        for ref in ref_chunks:
            ref_id = ref.get("id", "")
            if ref_id in retrieved_hashes:
                matched_ids.append(ref_id)
                continue
            # Content-substring fallback
            ref_content = ref.get("content", "")
            if ref_content and any(
                ref_content in rc for rc in retrieved_contents
            ):
                matched_ids.append(ref_id)

        return len(matched_ids) / len(ref_chunks), matched_ids

    def _run_judge(
        self, item: GeneratedQA, retrieved_chunks: list[Chunk]
    ) -> dict[str, Any]:
        if not retrieved_chunks:
            return {
                "answerable": False,
                "confidence": 1.0,
                "reasoning": "No chunks retrieved.",
            }

        chunks_text = "\n---\n".join(
            f"[Chunk {i + 1}]\n{c.content}"
            for i, c in enumerate(retrieved_chunks)
        )

        user_msg = _JUDGE_USER_TEMPLATE.format(
            question=item.qa["question"],
            answer=item.qa["answer"],
            chunks_text=chunks_text,
        )

        response = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        try:
            result: dict[str, Any] = json.loads(raw)
            return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse judge response: %s", raw)
            return {
                "answerable": False,
                "confidence": 0.0,
                "reasoning": f"Parse error: {raw}",
            }
