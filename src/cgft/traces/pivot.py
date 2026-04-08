"""Pivot filtering — LLM-based turn importance classification.

Identifies critical decision points in agentic traces using counterfactual
reasoning.  Based on PivotRL (arXiv:2603.21383): ~71% of turns produce zero
gradient during RL training.  Filtering to pivots yields +5.37 over SFT
with 4x fewer rollout turns.

Call strategy: 1 LLM call per turn (independent per-turn estimation), with
the full trace as context.  Uses gpt-4o-mini by default — this is a filtering
step, not a reward signal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Literal

from cgft.traces.adapter import NormalizedTrace, TraceMessage
from cgft.traces.processing import MIN_TRAIN_SAMPLES, TrainingExample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = {"pivot", "downstream", "trivial"}


@dataclass
class PivotRating:
    """Per-turn importance rating from counterfactual analysis."""

    trace_id: str
    turn_index: int
    importance_score: float  # 0.0–1.0
    category: Literal["pivot", "downstream", "trivial"]
    reasoning: str

    def __post_init__(self) -> None:
        if self.category not in _VALID_CATEGORIES:
            self.category = "downstream"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "turn_index": self.turn_index,
            "importance_score": self.importance_score,
            "category": self.category,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PivotRating:
        return cls(
            trace_id=d["trace_id"],
            turn_index=d["turn_index"],
            importance_score=float(d["importance_score"]),
            category=d["category"],
            reasoning=d["reasoning"],
        )


@dataclass
class PivotFilterResult:
    """Result of pivot filtering."""

    kept: list[TrainingExample]
    dropped: list[TrainingExample]
    ratings: list[PivotRating]

    @property
    def summary(self) -> dict[str, Any]:
        cats = {"pivot": 0, "downstream": 0, "trivial": 0}
        for r in self.ratings:
            cats[r.category] = cats.get(r.category, 0) + 1
        total = len(self.kept) + len(self.dropped)
        return {
            "categories": cats,
            "kept": len(self.kept),
            "dropped": len(self.dropped),
            "retention_rate": len(self.kept) / total if total else 0,
        }


# ---------------------------------------------------------------------------
# Heuristic fast path
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|welcome|greetings)\b",
    re.IGNORECASE,
)

_AUTH_ASK_RE = re.compile(
    r"\b(provide|share|give|tell)\b.*\b(name|email|order\s*(id|number)|zip|account)\b",
    re.IGNORECASE,
)


def _is_heuristic_trivial(
    ex: TrainingExample,
    turn_index_in_trace: int,
) -> bool:
    """Fast-path: detect obviously trivial turns without an LLM call.

    Returns True for:
    - First assistant turn that's a greeting with no tool calls
    - Turns that just ask for identity/auth info with no tool calls
    - Single lookup tool call to a get_*/find_* function
    """
    if not ex.completion_messages:
        return True

    first_completion = ex.completion_messages[0]

    # Greeting as first turn
    if turn_index_in_trace == 0 and not first_completion.tool_calls:
        if _GREETING_RE.search(first_completion.content):
            return True

    # Auth verification ask (no tool calls)
    if not first_completion.tool_calls and _AUTH_ASK_RE.search(first_completion.content):
        if len(first_completion.content) < 200:
            return True

    # Single lookup tool call (get_*, find_*)
    if first_completion.tool_calls and len(first_completion.tool_calls) == 1:
        tc = first_completion.tool_calls[0]
        if tc.name.startswith(("get_", "find_", "lookup_", "fetch_")):
            # Only if content is empty or very short (just calling the tool)
            if len(first_completion.content) < 50:
                return True

    return False


# ---------------------------------------------------------------------------
# LLM counterfactual analysis
# ---------------------------------------------------------------------------

_PIVOT_PROMPT = """\
You are analyzing an agentic trace to identify critical decision points.

TASK SUMMARY: {outcome_summary}

FULL TRACE:
{trace_messages}

CURRENT TURN (index {turn_index}):
Previous message: {previous_message}
Agent action: {current_turn}

Question: If the agent had made a DIFFERENT decision at this specific turn, \
how likely is it that the final outcome would have changed?

Consider:
- Is this a routine/formulaic action (greeting, auth lookup, confirmation)?
- Does this turn involve a judgment call (which tool to use, what parameters, \
how to interpret ambiguous information)?
- Is this turn mechanically executing a decision already made in a prior turn?
- Could a reasonable alternative action at this point lead to a different outcome?

Respond with JSON only:
{{"importance_score": <float 0.0-1.0>, "category": "pivot" | "downstream" | "trivial", \
"reasoning": "<one sentence>"}}"""


def _format_trace_messages(messages: list[TraceMessage], max_chars: int = 6000) -> str:
    """Render trace messages for the LLM prompt, truncating if needed."""
    parts: list[str] = []
    for m in messages:
        role = m.role.upper()
        content = m.content[:300] if len(m.content) > 300 else m.content
        line = f"[{role}] {content}"
        if m.tool_calls:
            for tc in m.tool_calls:
                line += f"\n  → {tc.name}({tc.arguments[:100]})"
        parts.append(line)

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


def _build_outcome_summary(trace: NormalizedTrace) -> str:
    """Generate a 1-sentence outcome summary from scores and final messages."""
    parts: list[str] = []

    # Use provider scores if available
    task_success = trace.scores.get("task_success")
    if task_success is not None:
        outcome = "SUCCESS" if task_success >= 0.5 else "FAILED"
        parts.append(f"Outcome: {outcome} (task_success={task_success})")

    # Add last assistant message as context
    last_assistant = None
    for msg in reversed(trace.messages):
        if msg.role == "assistant" and msg.content:
            last_assistant = msg.content[:150]
            break
    if last_assistant:
        parts.append(f"Final response: {last_assistant}")

    return " | ".join(parts) if parts else "Outcome unknown"


async def _rate_single_turn(
    client: Any,
    model: str,
    trace: NormalizedTrace,
    ex: TrainingExample,
    semaphore: asyncio.Semaphore,
) -> PivotRating:
    """Rate a single turn using LLM counterfactual analysis."""
    # Build the prompt
    trace_text = _format_trace_messages(trace.messages)
    outcome_summary = _build_outcome_summary(trace)

    # Previous message context
    prev_msg = "(start of conversation)"
    if ex.prompt_messages:
        last = ex.prompt_messages[-1]
        prev_msg = f"[{last.role.upper()}] {last.content[:200]}"

    # Current turn
    current = ""
    if ex.completion_messages:
        c = ex.completion_messages[0]
        current = f"[{c.role.upper()}] {c.content[:300]}"
        if c.tool_calls:
            for tc in c.tool_calls:
                current += f"\n  → {tc.name}({tc.arguments[:100]})"

    prompt = _PIVOT_PROMPT.format(
        outcome_summary=outcome_summary,
        trace_messages=trace_text,
        turn_index=ex.turn_index,
        previous_message=prev_msg,
        current_turn=current,
    )

    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=150,
                )
                text = str(response.choices[0].message.content).strip()
                # Parse JSON from response (handle markdown fences)
                text = re.sub(r"^```json\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                data = json.loads(text)

                # Validate category
                cat = data.get("category", "downstream")
                if cat not in _VALID_CATEGORIES:
                    cat = "downstream"

                return PivotRating(
                    trace_id=ex.trace_id,
                    turn_index=ex.turn_index,
                    importance_score=max(0.0, min(1.0, float(data["importance_score"]))),
                    category=cat,
                    reasoning=data.get("reasoning", ""),
                )
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                # Parse/validation errors — don't retry, return fallback
                return PivotRating(
                    trace_id=ex.trace_id,
                    turn_index=ex.turn_index,
                    importance_score=0.5,
                    category="downstream",
                    reasoning=f"LLM response parse error: {e}",
                )
            except Exception as e:
                # Transient/rate-limit errors — retry with jitter
                if attempt == 4:
                    logger.warning(
                        "Pivot rating failed for trace=%s turn=%d: %s",
                        ex.trace_id,
                        ex.turn_index,
                        e,
                    )
                    return PivotRating(
                        trace_id=ex.trace_id,
                        turn_index=ex.turn_index,
                        importance_score=0.5,
                        category="downstream",
                        reasoning=f"LLM rating failed: {e}",
                    )
                is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
                base_wait = (2**attempt) * (3 if is_rate_limit else 1)
                jitter = base_wait * (0.5 + random.random())
                await asyncio.sleep(jitter)

    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def _rate_all_turns(
    client: Any,
    model: str,
    examples: list[TrainingExample],
    traces: list[NormalizedTrace],
    max_concurrency: int,
) -> list[PivotRating]:
    """Rate all non-heuristic-trivial turns via LLM, in batches."""
    trace_map = {t.id: t for t in traces}
    semaphore = asyncio.Semaphore(max_concurrency)

    # Build work items
    work: list[tuple[TrainingExample, NormalizedTrace]] = []
    for ex in examples:
        trace = trace_map.get(ex.trace_id)
        if trace is not None:
            work.append((ex, trace))

    total = len(work)
    print(f"  Pivot rating: 0/{total} turns to rate via LLM", flush=True)

    # Process in batches of max_concurrency for visible progress
    results: list[PivotRating] = []
    batch_size = max_concurrency
    for i in range(0, total, batch_size):
        batch = work[i : i + batch_size]
        batch_results = await asyncio.gather(
            *[_rate_single_turn(client, model, trace, ex, semaphore) for ex, trace in batch]
        )
        results.extend(batch_results)
        print(f"  Pivot rating: {len(results)}/{total} turns rated", flush=True)

    return results


def apply_pivot_filter(
    examples: list[TrainingExample],
    traces: list[NormalizedTrace],
    *,
    llm_client: Any = None,
    model: str = "gpt-5.4-nano",
    threshold: float = 0.6,
    anchor_fraction: float = 0.1,
    ratings: list[PivotRating] | None = None,
    seed: int = 42,
    max_concurrency: int = 3,
) -> PivotFilterResult:
    """Filter training examples to high-signal pivot turns.

    Sync wrapper around ``apply_pivot_filter_async``.  Creates a new event
    loop to avoid monkey-patching the global loop.  If you are already in
    an async context, call ``apply_pivot_filter_async`` directly.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            apply_pivot_filter_async(
                examples,
                traces,
                llm_client=llm_client,
                model=model,
                threshold=threshold,
                anchor_fraction=anchor_fraction,
                ratings=ratings,
                seed=seed,
                max_concurrency=max_concurrency,
            )
        )
    finally:
        loop.close()


async def apply_pivot_filter_async(
    examples: list[TrainingExample],
    traces: list[NormalizedTrace],
    *,
    llm_client: Any = None,
    model: str = "gpt-5.4-nano",
    threshold: float = 0.6,
    anchor_fraction: float = 0.1,
    ratings: list[PivotRating] | None = None,
    seed: int = 42,
    max_concurrency: int = 3,
) -> PivotFilterResult:
    """Filter training examples to high-signal pivot turns (async version).

    Two-phase approach:
    1. Heuristic fast path tags obviously trivial turns (no LLM cost)
    2. LLM counterfactual analysis rates remaining turns

    Args:
        examples: output of ``build_training_examples()``
        traces: the ``NormalizedTrace`` objects (needed for full-trace context)
        llm_client: AsyncOpenAI-compatible client. Required if ``ratings``
            is not provided.
        model: LLM model for counterfactual analysis
        threshold: importance score cutoff (default 0.6)
        anchor_fraction: fraction of below-threshold turns to keep (default 0.1)
        ratings: pre-computed ratings — skips LLM if provided.
        seed: random seed for anchor sampling reproducibility
        max_concurrency: max concurrent LLM requests

    Returns:
        ``PivotFilterResult`` with kept, dropped, and all ratings.

    Raises:
        ``ValueError`` if fewer than ``MIN_TRAIN_SAMPLES`` examples remain.
    """
    if ratings is not None:
        all_ratings = list(ratings)
    else:
        if llm_client is None:
            raise ValueError("Either llm_client or ratings must be provided")
        all_ratings = await _compute_ratings_async(
            examples, traces, llm_client, model, max_concurrency
        )

    # Build lookup: (trace_id, turn_index) → PivotRating
    rating_map: dict[tuple[str, int], PivotRating] = {
        (r.trace_id, r.turn_index): r for r in all_ratings
    }

    # Split into kept and dropped
    above: list[TrainingExample] = []
    below: list[TrainingExample] = []

    for ex in examples:
        key = (ex.trace_id, ex.turn_index)
        rating = rating_map.get(key)
        if rating is None:
            below.append(ex)
            continue
        if rating.importance_score >= threshold:
            above.append(ex)
        else:
            below.append(ex)

    # Anchor sampling: keep a fraction of below-threshold turns
    rng = random.Random(seed)
    if anchor_fraction > 0 and below:
        anchor_count = max(1, int(len(below) * anchor_fraction))
        anchors = rng.sample(below, min(anchor_count, len(below)))
    else:
        anchors = []

    kept = above + anchors
    dropped = [ex for ex in below if ex not in anchors]

    if len(kept) < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Pivot filtering produced {len(kept)} examples, need at least "
            f"{MIN_TRAIN_SAMPLES}. Lower the threshold (currently {threshold}) "
            f"or add more traces."
        )

    return PivotFilterResult(kept=kept, dropped=dropped, ratings=all_ratings)


async def _compute_ratings_async(
    examples: list[TrainingExample],
    traces: list[NormalizedTrace],
    llm_client: Any,
    model: str,
    max_concurrency: int,
) -> list[PivotRating]:
    """Compute pivot ratings: heuristic fast path + LLM for the rest."""
    heuristic_ratings: list[PivotRating] = []
    llm_examples: list[TrainingExample] = []

    for ex in examples:
        if _is_heuristic_trivial(ex, ex.turn_index):
            heuristic_ratings.append(
                PivotRating(
                    trace_id=ex.trace_id,
                    turn_index=ex.turn_index,
                    importance_score=0.0,
                    category="trivial",
                    reasoning="Heuristic: trivial pattern detected",
                )
            )
        else:
            llm_examples.append(ex)

    logger.info(
        "Pivot filtering: %d heuristic trivial, %d sent to LLM",
        len(heuristic_ratings),
        len(llm_examples),
    )

    if llm_examples:
        llm_ratings = await _rate_all_turns(
            llm_client, model, llm_examples, traces, max_concurrency
        )
    else:
        llm_ratings = []

    return heuristic_ratings + list(llm_ratings)
