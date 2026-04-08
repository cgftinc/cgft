"""Generic trace processing pipeline — provider-agnostic.

Operates on ``NormalizedTrace`` / ``TraceMessage`` objects produced by any
``TraceAdapter`` implementation.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from cgft.traces.adapter import (
    DetectedSystemPrompt,
    DetectedTool,
    DetectedTools,
    NormalizedTrace,
    TraceMessage,
)

MIN_TRAIN_SAMPLES = 16

VALID_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_AUTH_PATTERNS = re.compile(
    r"\b(authenticate|login|sign[_\s]?in|bearer|oauth|refresh[_\s]?token)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def detect_system_prompt(
    traces: list[NormalizedTrace],
) -> DetectedSystemPrompt | None:
    """Scan *all* traces for ``role=="system"`` messages.

    Returns the most common system prompt.  If multiple distinct prompts are
    found, ``variants`` contains the alternatives so the wizard can surface a
    warning.
    """
    prompts: list[str] = []
    for trace in traces:
        for msg in trace.messages:
            if msg.role == "system":
                prompts.append(msg.content)
                break  # only first system msg per trace

    if not prompts:
        return None

    counts = Counter(prompts)
    most_common, count = counts.most_common(1)[0]
    variants = [p for p, _ in counts.most_common() if p != most_common]

    return DetectedSystemPrompt(
        prompt=most_common,
        count=count,
        total_traces=len(traces),
        variants=variants,
    )


def detect_tools(traces: list[NormalizedTrace]) -> DetectedTools:
    """Scan all traces for ``tool_calls`` in assistant messages.

    Tool names that are not valid Python identifiers are silently dropped
    (they cannot be used in env codegen and could be injection vectors).
    """
    tool_data: dict[str, dict[str, Any]] = {}  # name → {count, args, keys}

    for trace in traces:
        for msg in trace.messages:
            if msg.role != "assistant" or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if not VALID_IDENTIFIER_RE.match(tc.name):
                    continue
                entry = tool_data.setdefault(
                    tc.name,
                    {"count": 0, "sample_args": [], "param_keys": set()},
                )
                entry["count"] += 1
                args = tc.arguments_dict()
                # validate argument keys
                safe_args = {k: v for k, v in args.items() if VALID_IDENTIFIER_RE.match(k)}
                if safe_args and len(entry["sample_args"]) < 3:
                    entry["sample_args"].append(safe_args)
                entry["param_keys"].update(safe_args.keys())

    tools = [
        DetectedTool(
            name=name,
            call_count=data["count"],
            sample_args=data["sample_args"],
            param_keys=data["param_keys"],
        )
        for name, data in sorted(tool_data.items())
    ]
    return DetectedTools(tools=tools)


# ---------------------------------------------------------------------------
# Windowing — traces → training examples
# ---------------------------------------------------------------------------


@dataclass
class TrainingExample:
    """One training example derived from a trace turn.

    Dual representation:
    - ``prompt_messages`` / ``completion_messages`` — structured
      ``TraceMessage`` lists.  These are the authoritative training data,
      used by ``to_jsonl_dict()`` and consumed by ``dataset_preprocess``.
    - ``prompt`` / ``ground_truth`` — human-readable summaries for display
      in the wizard preview.  These are lossy (tool call arguments are
      flattened to text) and should NOT be used for training.
    """

    prompt_messages: list[TraceMessage]
    completion_messages: list[TraceMessage]
    prompt: str
    ground_truth: str
    trace_id: str
    turn_index: int
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Serialise to the JSONL schema consumed by ``dataset_preprocess``.

        The trainer expects structured message dicts (not human-readable
        strings).  ``prompt`` is a list of message dicts, ``ground_truth``
        is the completion message dict.  Metadata goes inside
        ``init_rollout_args`` so it's available in ``compute_reward()``.

        Schema::

            {
                "prompt": list[dict],        # chat messages before this turn
                "ground_truth": dict,         # the assistant completion
                "init_rollout_args": {
                    "trace_id": str,
                    "turn_index": int,
                    "total_messages": int,
                    "scores": dict[str, float],
                    "raw_prompt": str,        # human-readable for judge context
                },
            }
        """
        gt = self.completion_messages[0].to_dict() if self.completion_messages else {}
        return {
            "prompt": [m.to_dict() for m in self.prompt_messages],
            "ground_truth": gt,
            "init_rollout_args": {
                "trace_id": self.trace_id,
                "turn_index": self.turn_index,
                "total_messages": len(self.prompt_messages) + len(self.completion_messages),
                "scores": self.scores,
                "raw_prompt": self.prompt,
            },
        }


def _serialize_messages(messages: list[TraceMessage]) -> str:
    """Render a list of messages as a human-readable string."""
    parts: list[str] = []
    for m in messages:
        prefix = m.role.upper()
        if m.name:
            prefix += f" ({m.name})"
        parts.append(f"[{prefix}] {m.content}")
        if m.tool_calls:
            for tc in m.tool_calls:
                parts.append(f"  → tool_call: {tc.name}({tc.arguments})")
    return "\n".join(parts)


def build_training_examples(
    traces: list[NormalizedTrace],
    *,
    include_system_prompt: bool = False,
) -> list[TrainingExample]:
    """Window each trace's messages into prompt/completion pairs.

    For each assistant turn in the conversation:
    - prompt = all messages before this assistant turn
    - completion = this assistant turn + any immediately following tool
      response messages (up to the next user or assistant message)

    Traces with errors are skipped.
    """
    examples: list[TrainingExample] = []

    for trace in traces:
        if trace.errors:
            continue

        msgs = trace.messages
        if not msgs:
            continue

        # Find indices of assistant messages
        turn_index = 0
        i = 0
        while i < len(msgs):
            if msgs[i].role != "assistant":
                i += 1
                continue

            # prompt = everything before this assistant turn
            prompt_start = 0
            if not include_system_prompt and msgs[0].role == "system":
                prompt_start = 1
            prompt_msgs = list(msgs[prompt_start:i])

            # completion = this assistant msg + following tool responses
            completion_msgs: list[TraceMessage] = [msgs[i]]
            j = i + 1
            while j < len(msgs) and msgs[j].role == "tool":
                completion_msgs.append(msgs[j])
                j += 1

            if prompt_msgs:
                examples.append(
                    TrainingExample(
                        prompt_messages=prompt_msgs,
                        completion_messages=completion_msgs,
                        prompt=_serialize_messages(prompt_msgs),
                        ground_truth=_serialize_messages(completion_msgs),
                        trace_id=trace.id,
                        turn_index=turn_index,
                        scores=dict(trace.scores),
                        metadata=dict(trace.metadata),
                    )
                )

            turn_index += 1
            i = j

    return examples


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Result of applying heuristic filters."""

    kept: list[TrainingExample]
    dropped: list[tuple[TrainingExample, str]]

    @property
    def summary(self) -> dict[str, int]:
        """Count of drops per reason."""
        return dict(Counter(reason for _, reason in self.dropped))


def apply_heuristic_filters(
    examples: list[TrainingExample],
    *,
    min_completion_chars: int = 50,
    drop_auth_lookups: bool = False,
) -> FilterResult:
    """Filter examples using deterministic heuristics.

    Returns ``FilterResult`` with both kept and dropped examples (with
    reasons) so the wizard can show what was filtered and why.

    ``drop_auth_lookups`` defaults to ``False`` — users must opt in.
    """
    kept: list[TrainingExample] = []
    dropped: list[tuple[TrainingExample, str]] = []

    for ex in examples:
        # Check actual message content, not the rendered string (which
        # includes [ASSISTANT] prefixes and tool_call formatting).
        actual_content = ex.completion_messages[0].content if ex.completion_messages else ""
        if len(actual_content) < min_completion_chars:
            dropped.append((ex, "too_short"))
            continue
        if drop_auth_lookups and _AUTH_PATTERNS.search(actual_content):
            dropped.append((ex, "auth_lookup"))
            continue
        kept.append(ex)

    return FilterResult(kept=kept, dropped=dropped)


def apply_score_filter(
    examples: list[TrainingExample],
    score_name: str,
    *,
    min_score: float | None = None,
    max_score: float | None = None,
) -> list[TrainingExample]:
    """Filter examples by provider-reported scores."""
    result: list[TrainingExample] = []
    for ex in examples:
        score = ex.scores.get(score_name)
        if score is None:
            continue
        if min_score is not None and score < min_score:
            continue
        if max_score is not None and score > max_score:
            continue
        result.append(ex)
    return result


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_dataset(
    examples: list[TrainingExample],
    train_count: int,
    eval_count: int,
    *,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into train/eval and serialise to JSONL-ready dicts.

    Shuffles with a fixed seed before splitting to avoid temporal data
    leakage (traces are fetched chronologically).

    Raises ``ValueError`` if ``train_count < MIN_TRAIN_SAMPLES`` or if there
    are not enough examples.
    """
    import random

    if train_count < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TRAIN_SAMPLES} training examples, got train_count={train_count}"
        )
    total = train_count + eval_count
    if len(examples) < total:
        raise ValueError(
            f"Not enough examples: need {total} (train={train_count} + "
            f"eval={eval_count}), have {len(examples)}"
        )

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    train = [ex.to_jsonl_dict() for ex in shuffled[:train_count]]
    eval_ = [ex.to_jsonl_dict() for ex in shuffled[train_count : train_count + eval_count]]
    return train, eval_
