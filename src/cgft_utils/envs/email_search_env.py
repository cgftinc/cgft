"""EmailSearchEnv — CGFT search env with email-specific prompt and judge-based rewards."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from benchmax.envs.tracking import log_env
from benchmax.envs.types import ToolDefinition
from cgft_utils.rubrics.rubric import Rubric, evaluate_single_rubric

from .cgft_search_env import CgftSearchEnv


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
        "Response correctly answers the question and is factually consistent with the reference answer."
    ),
    type="positive",
    score_map={
        0: "Provided answer is missing or incorrect.",
        0.5: "Response captures some facts from the reference answer, but is missing key facts or has an incorrect conclusion.",
        1: "Response correctly answers the question and is factually consistent with the reference answer.",
    },
)
_ANSWER_RUBRIC_CONCISENESS = Rubric(
    title="Answer conciseness",
    description=(
        "Response is concise and avoids unnecessary verbosity while still directly answering the question."
    ),
    type="positive",
)
DEFAULT_W_CORRECTNESS = 1.0
DEFAULT_W_CONCISENESS = 0.5
DEFAULT_W_RECALL = 0.5
DEFAULT_W_PRECISION = 0.5
DEFAULT_W_SEARCH_EFFICIENCY = 0.1
DEFAULT_JUDGE_TIMEOUT = 30.0

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_CITATION_PATTERN = re.compile(r"\[(?:Source|Thread)\s*:\s*([^\]]+)\]", re.IGNORECASE)


class EmailSearchEnv(CgftSearchEnv):
    """Email-focused search env with judge + citation rewards."""

    system_prompt: str = EMAIL_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        judge_base_url: str = "",
        judge_api_key: str = "",
        judge_model: str = "",
        judge_timeout: float | None = DEFAULT_JUDGE_TIMEOUT,
        w_correctness: float = DEFAULT_W_CORRECTNESS,
        w_conciseness: float = DEFAULT_W_CONCISENESS,
        w_recall: float = DEFAULT_W_RECALL,
        w_precision: float = DEFAULT_W_PRECISION,
        w_search_efficiency: float = DEFAULT_W_SEARCH_EFFICIENCY,
        debug_reward: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._judge_base_url = judge_base_url
        self._judge_api_key = judge_api_key
        self._judge_model = judge_model
        self._judge_timeout = judge_timeout
        self._w_correctness = w_correctness
        self._w_conciseness = w_conciseness
        self._w_recall = w_recall
        self._w_precision = w_precision
        self._w_search_efficiency = w_search_efficiency
        self._debug_reward = bool(debug_reward)

        # Keep the same tool name/behavior, but update schema text for agent clarity.
        search_tool_definition = ToolDefinition(
            name="search",
            description=(
                "Search with BM25."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results to return (default 10).",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )
        self._tools = {"search": (search_tool_definition, self._search_tool)}

    async def compute_reward(
        self, rollout_id: str, completion: str | list[dict[str, Any]], ground_truth: Any, **kwargs: Any
    ) -> dict[str, float]:
        """Return 5 weighted reward components: correctness, conciseness, recall, precision, search_efficiency."""
        def _search_efficiency_raw(calls: int) -> float:
            if calls <= 2:
                return 1.0
            if calls == 3:
                return 0.5
            return 0.0

        log_lines: list[str] = []

        try:
            search_calls = _count_search_calls(completion)
            search_efficiency_score = self._w_search_efficiency * _search_efficiency_raw(search_calls)
            completion_text = _extract_completion_text(completion)
            if not completion_text.strip():
                if self._debug_reward:
                    print(f"[EmailSearchEnv] empty completion, returning zeros")
                # All gated rewards are 0 since correctness/precision/recall are 0
                return {
                    "answer_correctness": 0.0,
                    "conciseness": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "search_efficiency": 0.0,
                }

            answer_block = _extract_answer_block(completion_text)
            cited_ids = {_canonicalize_source_id(x) for x in _parse_citations_from_text(answer_block) if x}

            ref_ids = {
                _canonicalize_source_id(thread_id)
                for thread_id in _extract_reference_thread_ids(kwargs.get("reference_chunks", []))
                if thread_id
            }

            overlap = cited_ids & ref_ids
            citation_recall = (len(overlap) / len(ref_ids)) if ref_ids else 0.0
            citation_precision = (len(overlap) / len(cited_ids)) if cited_ids else 0.0

            prompt = str(kwargs.get("prompt") or kwargs.get("question") or "")
            gt_str = str(ground_truth or "")

            # Log ground truth and detected answer for comparison
            log_env(
                rollout_id,
                f"[EmailSearchEnv] Ground Truth Answer:\n{gt_str}\n\n"
                f"[EmailSearchEnv] Detected Answer (from <answer> tag):\n{answer_block}\n\n"
                f"[EmailSearchEnv] Question:\n{prompt}\n\n"
                f"[EmailSearchEnv] Citations - predicted={sorted(cited_ids)}, ground_truth={sorted(ref_ids)}, overlap={sorted(overlap)}"
            )
            if self._debug_reward:
                log_lines.append(f"question={prompt}")
                log_lines.append(f"ground_truth_answer={str(ground_truth or '')}")
                log_lines.append(f"response_answer={answer_block}")
                log_lines.append("citation comparison:")
                log_lines.append(f"  predicted_citations={sorted(cited_ids)}")
                log_lines.append(f"  ground_truth_citations={sorted(ref_ids)}")
                log_lines.append(f"  overlap={sorted(overlap)}")
                log_lines.append(f"  citation_recall_raw={citation_recall:.4f} citation_precision_raw={citation_precision:.4f}")

            correctness_raw, conciseness_raw, judge_ok = await self._judge_answer_quality(
                rollout_id=rollout_id,
                question=prompt,
                ground_truth=gt_str,
                response=answer_block,
                log_lines=log_lines,
            )
            if self._debug_reward:
                log_lines.append(f"judge_ok={judge_ok}, correctness_raw={correctness_raw}, conciseness_raw={conciseness_raw}")

            # Gate rewards: conciseness requires correctness; search_efficiency requires all three
            correctness_ok = correctness_raw > 0
            precision_ok = citation_precision > 0
            recall_ok = citation_recall > 0

            rewards = {
                "answer_correctness": self._w_correctness * correctness_raw,
                "conciseness": self._w_conciseness * conciseness_raw if correctness_ok else 0.0,
                "recall": self._w_recall * citation_recall,
                "precision": self._w_precision * citation_precision,
                "search_efficiency": search_efficiency_score if (correctness_ok and precision_ok and recall_ok) else 0.0,
            }
            if self._debug_reward:
                log_lines.append(f"final rewards={rewards}")
                print("[EmailSearchEnv]\n" + "\n".join(log_lines))
            return rewards
        except Exception as exc:
            if self._debug_reward:
                log_lines.append(f"compute_reward failed: {exc}")
                print("[EmailSearchEnv]\n" + "\n".join(log_lines))
            # All gated rewards are 0 since correctness/precision/recall are 0
            return {
                "answer_correctness": 0.0,
                "conciseness": 0.0,
                "recall": 0.0,
                "precision": 0.0,
                "search_efficiency": 0.0,
            }

    async def _judge_answer_quality(
        self,
        rollout_id: str,
        question: str,
        ground_truth: str,
        response: str,
        log_lines: list[str] | None = None,
    ) -> tuple[float, float, bool]:
        if not response.strip():
            return (0.0, 0.0, False)

        if not self._judge_base_url or not self._judge_api_key:
            if self._debug_reward and log_lines is not None:
                log_lines.append("Judge disabled: missing judge_base_url or judge_api_key")
            return (0.0, 0.0, False)

        # Log what we're sending to the judge
        log_env(
            rollout_id,
            f"[EmailSearchEnv] Judge Input:\n"
            f"  model: {self._judge_model}\n"
            f"  base_url: {self._judge_base_url}\n"
            f"  question: {question[:200]}{'...' if len(question) > 200 else ''}\n"
            f"  ground_truth: {ground_truth[:200]}{'...' if len(ground_truth) > 200 else ''}\n"
            f"  response: {response[:200]}{'...' if len(response) > 200 else ''}\n"
            f"  correctness_rubric: {_ANSWER_RUBRIC_POSITIVE.description}\n"
            f"  conciseness_rubric: {_ANSWER_RUBRIC_CONCISENESS.description}"
        )

        try:
            if self._debug_reward and log_lines is not None:
                log_lines.append(f"Judge request starting (model={self._judge_model}, base_url={self._judge_base_url})")
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

            pos_result, concise_result = await asyncio.gather(pos_task, concise_task)
            pos_raw = _clip01(pos_result.get("score", 0.0))
            concise_raw = _clip01(concise_result.get("score", 0.0))

            # Log judge results
            log_env(
                rollout_id,
                f"[EmailSearchEnv] Judge Results:\n"
                f"  correctness_score: {pos_raw:.4f}\n"
                f"  conciseness_score: {concise_raw:.4f}\n"
                f"  correctness_raw: {pos_result}\n"
                f"  conciseness_raw: {concise_result}"
            )

            if self._debug_reward and log_lines is not None:
                log_lines.append(f"Judge completed: correctness={pos_raw:.4f}, conciseness={concise_raw:.4f}")
                log_lines.append(f"Judge raw responses: pos={pos_result}, concise={concise_result}")
            return (pos_raw, concise_raw, True)
        except Exception as exc:
            log_env(rollout_id, f"[EmailSearchEnv] Judge request failed: {exc}")
            if self._debug_reward and log_lines is not None:
                log_lines.append(f"Judge request failed: {exc}")
            return (0.0, 0.0, False)

def _clip01(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


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


# Backward-compat shim for older bundled compute_reward implementations that
# still reference this module-level helper.
def _search_call_score(search_calls: int) -> float:
    if search_calls <= 2:
        return 1.0
    if search_calls == 3:
        return 0.5
    return 0.0


def _extract_completion_text(completion: str | list[dict[str, Any]]) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return ""

    assistant_contents: list[str] = []
    for msg in completion:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            assistant_contents.append(content)

    if not assistant_contents:
        return ""
    return "\n".join(assistant_contents)


def _extract_answer_block(text: str) -> str:
    match = _ANSWER_TAG_RE.search(text or "")
    return (match.group(1) if match else text).strip()


def _parse_citations_from_text(text: str) -> set[str]:
    return {m.group(1).strip() for m in _CITATION_PATTERN.finditer(text or "") if m.group(1).strip()}




def _extract_reference_thread_ids(reference_chunks: Any) -> set[str]:
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
    # Normalize common chunk/file suffix variants to thread id shape.
    return re.sub(r"\.txt_\d+$", "", text, flags=re.IGNORECASE)
