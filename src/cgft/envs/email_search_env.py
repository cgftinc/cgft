"""EmailSearchEnv — email-specific prompt and judge-based rewards.

Extends :class:`SearchEnv` with a 5-component reward:
correctness, conciseness, citation recall, citation precision,
and search efficiency.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from benchmax.envs.tracking import log_env
from benchmax.envs.types import ToolDefinition

from cgft.corpus.corpora.search import CorporaSearch
from cgft.rubrics.rubric import Rubric, evaluate_single_rubric

from .search_env import SearchEnv

EMAIL_SYSTEM_PROMPT = """Answer the given question by using a search engine over both an email corpus and wikipedia entries.

You will fist reason about the question inside <think> and </think>. For instance you may want
to rephrase the question or break down the question into multiple sub-questions that you will search for.

You must call the search engine with <tool_call> and it will return the top searched results.
After receiving the information, you must reason about it inside <think> and </think> before either
(1) issuing a new query with <tool_call>
(2) providing the final answer inside <answer> and </answer> tags.

Each of your reasoning steps should be grounded in the retrieved information.

You can search up to 4 times. Try to break down the question for each search query and gather comprehensive
information.

Recommended approach:
1. If initial results do not contain the answer, try to re-query with broadened or rephrased language.
2. Reference retrieved chunks to formulate more specific follow-up queries (e.g. using keywords in chunk content or using metadata filters like sender, recipient, date, or keywords)

If you have gathered enough information to answer the question,
return your final answer inside <answer>...</answer> and cite supporting threads as [Source: <thread_id>].
DO NOT cite using the chunk_id."""

_ANSWER_RUBRIC_POSITIVE = Rubric(
    title="Answer correctness",
    description=(
        "Response correctly answers the question and is factually "
        "consistent with the reference answer."
    ),
    type="positive",
    score_map={
        0: "Provided answer is missing or incorrect.",
        0.5: (
            "Response captures some facts from the reference answer, "
            "but is missing key facts or has an incorrect conclusion."
        ),
        1: (
            "Response correctly answers the question and is factually "
            "consistent with the reference answer."
        ),
    },
)
_ANSWER_RUBRIC_CONCISENESS = Rubric(
    title="Answer conciseness",
    description=(
        "Response is concise and avoids unnecessary verbosity while "
        "still directly answering the question."
    ),
    type="positive",
)
DEFAULT_W_CORRECTNESS = 1.0
DEFAULT_W_CONCISENESS = 0.5
DEFAULT_W_RECALL = 0.5
DEFAULT_W_PRECISION = 0.5
DEFAULT_W_SEARCH_EFFICIENCY = 0.1
DEFAULT_JUDGE_TIMEOUT = 30.0

_ANSWER_TAG_RE = re.compile(
    r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
)
_CITATION_PATTERN = re.compile(
    r"\[(?:Source|Thread)\s*:\s*([^\]]+)\]", re.IGNORECASE
)


class EmailSearchEnv(SearchEnv):
    """Email-focused search env with judge + citation rewards."""

    system_prompt: str = EMAIL_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        api_key: str,
        corpus_id: str,
        base_url: str,
        judge_base_url: str = "",
        judge_api_key: str = "",
        judge_model: str = "",
        judge_timeout: float = DEFAULT_JUDGE_TIMEOUT,
        w_correctness: float = DEFAULT_W_CORRECTNESS,
        w_conciseness: float = DEFAULT_W_CONCISENESS,
        w_recall: float = DEFAULT_W_RECALL,
        w_precision: float = DEFAULT_W_PRECISION,
        w_search_efficiency: float = DEFAULT_W_SEARCH_EFFICIENCY,
        **kwargs: Any,
    ):
        search = CorporaSearch(
            api_key=api_key,
            corpus_name="",
            base_url=base_url,
            corpus_id=corpus_id,
        )
        super().__init__(
            search=search,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
            judge_model=judge_model,
            judge_timeout=judge_timeout,
            w_correctness=w_correctness,
            **kwargs,
        )
        self._w_conciseness = w_conciseness
        self._w_recall = w_recall
        self._w_precision = w_precision
        self._w_search_efficiency = w_search_efficiency

        # Override tool schema to use simpler search tool for emails
        search_tool_definition = ToolDefinition(
            name="search",
            description="Search with BM25.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Max number of results to return (default 10)."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )
        self._tools = {"search": (search_tool_definition, self._search_tool)}

    async def compute_reward(
        self,
        rollout_id: str,
        completion: str | list[dict[str, Any]],
        ground_truth: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Return 5 weighted reward components."""
        zeros = {
            "answer_correctness": 0.0,
            "conciseness": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "search_efficiency": 0.0,
        }

        try:
            search_calls = _count_search_calls(completion)
            search_efficiency_score = (
                self._w_search_efficiency
                * _search_efficiency_raw(search_calls)
            )
            completion_text = _extract_completion_text(completion)
            if not completion_text.strip():
                return zeros

            answer_block = _extract_answer_block(completion_text)
            cited_ids = {
                _canonicalize_source_id(x)
                for x in _parse_citations_from_text(answer_block)
                if x
            }

            ref_ids = {
                _canonicalize_source_id(thread_id)
                for thread_id in _extract_reference_thread_ids(
                    kwargs.get("reference_chunks", [])
                )
                if thread_id
            }

            overlap = cited_ids & ref_ids
            citation_recall = (
                (len(overlap) / len(ref_ids)) if ref_ids else 0.0
            )
            citation_precision = (
                (len(overlap) / len(cited_ids)) if cited_ids else 0.0
            )

            prompt = str(
                kwargs.get("prompt") or kwargs.get("question") or ""
            )
            gt_str = str(ground_truth or "")

            log_env(
                rollout_id,
                f"[EmailSearchEnv] Ground Truth Answer:\n{gt_str}\n\n"
                f"[EmailSearchEnv] Detected Answer (from <answer> tag):"
                f"\n{answer_block}\n\n"
                f"[EmailSearchEnv] Question:\n{prompt}\n\n"
                f"[EmailSearchEnv] Citations - "
                f"predicted={sorted(cited_ids)}, "
                f"ground_truth={sorted(ref_ids)}, "
                f"overlap={sorted(overlap)}",
            )

            correctness_raw, conciseness_raw = (
                await self._judge_answer_quality(
                    question=prompt,
                    ground_truth=gt_str,
                    response=answer_block,
                )
            )

            correctness_ok = correctness_raw > 0
            precision_ok = citation_precision > 0
            recall_ok = citation_recall > 0

            return {
                "answer_correctness": (
                    self._w_correctness * correctness_raw
                ),
                "conciseness": (
                    self._w_conciseness * conciseness_raw
                    if correctness_ok
                    else 0.0
                ),
                "recall": self._w_recall * citation_recall,
                "precision": self._w_precision * citation_precision,
                "search_efficiency": (
                    search_efficiency_score
                    if (correctness_ok and precision_ok and recall_ok)
                    else 0.0
                ),
            }
        except Exception:
            return zeros

    async def _judge_answer_quality(
        self,
        question: str,
        ground_truth: str,
        response: str,
    ) -> tuple[float, float]:
        if not response.strip():
            return (0.0, 0.0)
        if not self._judge_base_url or not self._judge_api_key:
            return (0.0, 0.0)

        try:
            pos_task = evaluate_single_rubric(
                rubric=_ANSWER_RUBRIC_POSITIVE,
                question=question,
                ground_truth=ground_truth,
                response=response,
                model_name=self._judge_model,
                base_url=self._judge_base_url,
                api_key=self._judge_api_key,
                timeout=self._judge_timeout,
            )
            concise_task = evaluate_single_rubric(
                rubric=_ANSWER_RUBRIC_CONCISENESS,
                question=question,
                ground_truth=ground_truth,
                response=response,
                model_name=self._judge_model,
                base_url=self._judge_base_url,
                api_key=self._judge_api_key,
                timeout=self._judge_timeout,
            )

            pos_result, concise_result = await asyncio.gather(
                pos_task, concise_task
            )
            return (_clip01(pos_result.get("score", 0.0)),
                    _clip01(concise_result.get("score", 0.0)))
        except Exception:
            return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip01(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def _search_efficiency_raw(calls: int) -> float:
    if calls <= 2:
        return 1.0
    if calls == 3:
        return 0.5
    return 0.0


def _count_search_calls(completion: str | list[dict[str, Any]]) -> int:
    if not isinstance(completion, list):
        return 0
    total = 0
    for msg in completion:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total += content.count("<tool_call>")
    return total


def _extract_completion_text(
    completion: str | list[dict[str, Any]],
) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return ""
    parts: list[str] = []
    for msg in completion:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
    return "\n".join(parts) if parts else ""


def _extract_answer_block(text: str) -> str:
    match = _ANSWER_TAG_RE.search(text or "")
    return (match.group(1) if match else text).strip()


def _parse_citations_from_text(text: str) -> set[str]:
    return {
        m.group(1).strip()
        for m in _CITATION_PATTERN.finditer(text or "")
        if m.group(1).strip()
    }


def _extract_reference_thread_ids(
    reference_chunks: Any,
) -> set[str]:
    ids: set[str] = set()
    if not isinstance(reference_chunks, list):
        return ids
    for chunk in reference_chunks:
        if not isinstance(chunk, dict):
            continue
        md = chunk.get("metadata", {})
        if not isinstance(md, dict):
            continue
        thread_id = str(md.get("thread_id") or "").strip()
        if thread_id:
            ids.add(thread_id)
    return ids


def _canonicalize_source_id(source_id: str) -> str:
    text = str(source_id or "").strip()
    if not text:
        return ""
    return re.sub(r"\.txt_\d+$", "", text, flags=re.IGNORECASE)
