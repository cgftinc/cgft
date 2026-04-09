"""Tests for pivot filtering."""

import json

import pytest

from cgft.traces.adapter import NormalizedTrace, ToolCall, TraceMessage
from cgft.traces.pivot import (
    PivotCheckpointManager,
    PivotFilterResult,
    PivotRating,
    _build_outcome_summary,
    _is_heuristic_trivial,
    apply_pivot_filter,
)
from cgft.traces.processing import TrainingExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(role: str, content: str, **kwargs) -> TraceMessage:
    return TraceMessage(role=role, content=content, **kwargs)


def _example(
    turn_index: int = 0,
    trace_id: str = "t1",
    completion_content: str = "response",
    completion_tool_calls: list[ToolCall] | None = None,
    prompt_content: str = "question",
) -> TrainingExample:
    return TrainingExample(
        prompt_messages=[_msg("user", prompt_content)],
        completion_messages=[
            _msg("assistant", completion_content, tool_calls=completion_tool_calls)
        ],
        prompt=f"[USER] {prompt_content}",
        ground_truth=f"[ASSISTANT] {completion_content}",
        trace_id=trace_id,
        turn_index=turn_index,
    )


def _trace(trace_id: str = "t1", scores: dict | None = None) -> NormalizedTrace:
    return NormalizedTrace(
        id=trace_id,
        messages=[
            _msg("user", "Hello"),
            _msg("assistant", "Hi, how can I help?"),
            _msg("user", "I want to return my order"),
            _msg(
                "assistant",
                "",
                tool_calls=[ToolCall(name="get_order_details", arguments='{"order_id": "#123"}')],
            ),
            _msg("tool", "Order #123: laptop, delivered", name="get_order_details"),
            _msg("assistant", "I can process that return for you."),
        ],
        scores=scores or {},
    )


# ---------------------------------------------------------------------------
# Heuristic fast path
# ---------------------------------------------------------------------------


class TestHeuristicTrivial:
    def test_greeting_first_turn(self):
        ex = _example(turn_index=0, completion_content="Hello! How can I help you today?")
        assert _is_heuristic_trivial(ex, 0) is True

    def test_greeting_not_first_turn(self):
        ex = _example(turn_index=2, completion_content="Hello! How can I help you today?")
        assert _is_heuristic_trivial(ex, 2) is False

    def test_auth_ask(self):
        ex = _example(
            turn_index=1,
            completion_content="Could you please provide your name and zip code?",
        )
        assert _is_heuristic_trivial(ex, 1) is True

    def test_auth_ask_long_content_not_trivial(self):
        ex = _example(
            turn_index=1,
            completion_content="Could you provide your name? " + "x" * 200,
        )
        assert _is_heuristic_trivial(ex, 1) is False

    def test_single_lookup_tool(self):
        ex = _example(
            turn_index=2,
            completion_content="",
            completion_tool_calls=[
                ToolCall(name="get_user_details", arguments='{"user_id": "u1"}')
            ],
        )
        assert _is_heuristic_trivial(ex, 2) is True

    def test_single_non_lookup_tool_not_trivial(self):
        ex = _example(
            turn_index=2,
            completion_content="",
            completion_tool_calls=[
                ToolCall(name="cancel_order", arguments='{"order_id": "#123"}')
            ],
        )
        assert _is_heuristic_trivial(ex, 2) is False

    def test_multiple_tools_not_trivial(self):
        ex = _example(
            turn_index=2,
            completion_content="",
            completion_tool_calls=[
                ToolCall(name="get_user_details", arguments='{}'),
                ToolCall(name="get_order_details", arguments='{}'),
            ],
        )
        assert _is_heuristic_trivial(ex, 2) is False

    def test_substantive_response_not_trivial(self):
        ex = _example(
            turn_index=3,
            completion_content="Based on the order details, I can process a return. "
            "The refund of $199.99 will be applied to your credit card within 5-7 business days.",
        )
        assert _is_heuristic_trivial(ex, 3) is False

    def test_empty_completion_trivial(self):
        ex = TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[],
            prompt="Q",
            ground_truth="",
            trace_id="t1",
            turn_index=0,
        )
        assert _is_heuristic_trivial(ex, 0) is True


# ---------------------------------------------------------------------------
# Outcome summary
# ---------------------------------------------------------------------------


class TestOutcomeSummary:
    def test_with_scores(self):
        trace = _trace(scores={"task_success": 1.0})
        summary = _build_outcome_summary(trace)
        assert "SUCCESS" in summary

    def test_with_failure(self):
        trace = _trace(scores={"task_success": 0.0})
        summary = _build_outcome_summary(trace)
        assert "FAILED" in summary

    def test_without_scores(self):
        trace = _trace(scores={})
        summary = _build_outcome_summary(trace)
        # Should still produce something from last assistant message
        assert len(summary) > 0

    def test_empty_trace(self):
        trace = NormalizedTrace(id="empty", messages=[])
        summary = _build_outcome_summary(trace)
        assert "unknown" in summary.lower()


# ---------------------------------------------------------------------------
# PivotRating serialization
# ---------------------------------------------------------------------------


class TestPivotRating:
    def test_roundtrip(self):
        rating = PivotRating(
            trace_id="t1",
            turn_index=3,
            importance_score=0.85,
            category="pivot",
            reasoning="Agent made a judgment call about return policy",
        )
        d = rating.to_dict()
        restored = PivotRating.from_dict(d)
        assert restored.trace_id == "t1"
        assert restored.turn_index == 3
        assert restored.importance_score == 0.85
        assert restored.category == "pivot"


# ---------------------------------------------------------------------------
# apply_pivot_filter with pre-computed ratings
# ---------------------------------------------------------------------------


class TestApplyPivotFilter:
    def _make_examples(self, n: int) -> list[TrainingExample]:
        return [
            _example(turn_index=i, trace_id=f"t{i // 10}", completion_content="A" * 100)
            for i in range(n)
        ]

    def _make_traces(self, n: int) -> list[NormalizedTrace]:
        return [_trace(trace_id=f"t{i}") for i in range(n)]

    def test_filters_by_threshold(self):
        examples = self._make_examples(40)
        traces = self._make_traces(4)

        # Rate half as pivots, half as trivial
        ratings = [
            PivotRating(
                trace_id=ex.trace_id,
                turn_index=ex.turn_index,
                importance_score=0.9 if i % 2 == 0 else 0.1,
                category="pivot" if i % 2 == 0 else "trivial",
                reasoning="test",
            )
            for i, ex in enumerate(examples)
        ]

        result = apply_pivot_filter(
            examples,
            traces,
            ratings=ratings,
            threshold=0.6,
            anchor_fraction=0.0,  # no anchors for clean test
        )

        assert len(result.kept) == 20
        assert len(result.dropped) == 20

    def test_anchor_sampling(self):
        examples = self._make_examples(40)
        traces = self._make_traces(4)

        # All trivial
        ratings = [
            PivotRating(
                trace_id=ex.trace_id,
                turn_index=ex.turn_index,
                importance_score=0.1,
                category="trivial",
                reasoning="test",
            )
            for ex in examples
        ]

        result = apply_pivot_filter(
            examples,
            traces,
            ratings=ratings,
            threshold=0.6,
            anchor_fraction=0.5,  # keep 50% as anchors
        )

        # 0 above threshold + 50% of 40 anchors = 20
        assert len(result.kept) == 20

    def test_anchor_sampling_deterministic(self):
        examples = self._make_examples(40)
        traces = self._make_traces(4)
        ratings = [
            PivotRating(
                trace_id=ex.trace_id,
                turn_index=ex.turn_index,
                importance_score=0.1,
                category="trivial",
                reasoning="test",
            )
            for ex in examples
        ]

        r1 = apply_pivot_filter(
            examples, traces, ratings=ratings, threshold=0.6, seed=42, anchor_fraction=0.5
        )
        r2 = apply_pivot_filter(
            examples, traces, ratings=ratings, threshold=0.6, seed=42, anchor_fraction=0.5
        )
        assert [ex.turn_index for ex in r1.kept] == [ex.turn_index for ex in r2.kept]

    def test_raises_when_too_few_kept(self):
        examples = self._make_examples(40)
        traces = self._make_traces(4)
        ratings = [
            PivotRating(
                trace_id=ex.trace_id,
                turn_index=ex.turn_index,
                importance_score=0.01,
                category="trivial",
                reasoning="test",
            )
            for ex in examples
        ]

        with pytest.raises(ValueError, match="Pivot filtering produced"):
            apply_pivot_filter(
                examples,
                traces,
                ratings=ratings,
                threshold=0.99,
                anchor_fraction=0.0,
            )

    def test_requires_llm_client_or_ratings(self):
        examples = self._make_examples(40)
        traces = self._make_traces(4)

        with pytest.raises(ValueError, match="Either llm_client or ratings"):
            apply_pivot_filter(examples, traces)

    def test_summary(self):
        examples = self._make_examples(40)
        traces = self._make_traces(4)
        ratings = [
            PivotRating(
                trace_id=ex.trace_id,
                turn_index=ex.turn_index,
                importance_score=0.9 if i < 20 else 0.1,
                category="pivot" if i < 20 else "trivial",
                reasoning="test",
            )
            for i, ex in enumerate(examples)
        ]

        result = apply_pivot_filter(
            examples, traces, ratings=ratings, threshold=0.6, anchor_fraction=0.0
        )
        s = result.summary
        assert s["kept"] == 20
        assert s["dropped"] == 20
        assert s["categories"]["pivot"] == 20
        assert s["categories"]["trivial"] == 20
        assert 0.49 < s["retention_rate"] < 0.51

    def test_matches_on_trace_id_and_turn_index(self):
        """Ratings must match on (trace_id, turn_index), not turn_index alone."""
        # Create enough examples so we have 16+ after filtering
        examples = []
        ratings = []
        for i in range(20):
            ex = _example(turn_index=i, trace_id="trace_a", completion_content="A" * 100)
            examples.append(ex)
            ratings.append(PivotRating("trace_a", i, 0.9, "pivot", "important"))

        # Add one from trace_b that should be dropped
        ex_b = _example(turn_index=0, trace_id="trace_b", completion_content="B" * 100)
        examples.append(ex_b)
        ratings.append(PivotRating("trace_b", 0, 0.1, "trivial", "routine"))

        traces = [_trace(trace_id="trace_a"), _trace(trace_id="trace_b")]

        result = apply_pivot_filter(
            examples,
            traces,
            ratings=ratings,
            threshold=0.6,
            anchor_fraction=0.0,
        )

        assert len(result.kept) == 20
        assert all(ex.trace_id == "trace_a" for ex in result.kept)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class TestPivotCheckpointManager:
    def test_save_and_resume(self, tmp_path):
        cp = PivotCheckpointManager(tmp_path / "ckpt", model="gpt-5.4-nano")
        ratings = [
            PivotRating("t1", 0, 0.9, "pivot", "important"),
            PivotRating("t1", 1, 0.2, "trivial", "routine"),
        ]
        cp.save_batch(ratings)

        loaded = cp.resume()
        assert len(loaded) == 2
        assert loaded[("t1", 0)].importance_score == 0.9
        assert loaded[("t1", 1)].category == "trivial"

    def test_resume_empty(self, tmp_path):
        cp = PivotCheckpointManager(tmp_path / "ckpt", model="gpt-5.4-nano")
        assert cp.resume() == {}

    def test_model_change_invalidates(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        cp1 = PivotCheckpointManager(ckpt_dir, model="gpt-5.4-nano")
        cp1.save_batch([PivotRating("t1", 0, 0.9, "pivot", "test")])

        cp2 = PivotCheckpointManager(ckpt_dir, model="gpt-4o-mini")
        loaded = cp2.resume()
        assert loaded == {}
        assert not ckpt_dir.exists()  # cleaned up

    def test_truncated_jsonl_recovery(self, tmp_path):
        cp = PivotCheckpointManager(tmp_path / "ckpt", model="gpt-5.4-nano")
        cp.save_batch([
            PivotRating("t1", 0, 0.9, "pivot", "good"),
            PivotRating("t1", 1, 0.3, "trivial", "good"),
        ])
        # Simulate crash: append truncated line
        with (cp.checkpoint_dir / cp._RATINGS_FILE).open("a") as fh:
            fh.write('{"trace_id": "t1", "turn_index": 2, "importance_sc')

        loaded = cp.resume()
        # Should recover the 2 good lines, skip the truncated one
        assert len(loaded) == 2
        assert ("t1", 0) in loaded
        assert ("t1", 1) in loaded

    def test_incremental_append(self, tmp_path):
        cp = PivotCheckpointManager(tmp_path / "ckpt", model="gpt-5.4-nano")
        cp.save_batch([PivotRating("t1", 0, 0.9, "pivot", "batch1")])
        cp.save_batch([PivotRating("t1", 1, 0.3, "downstream", "batch2")])

        loaded = cp.resume()
        assert len(loaded) == 2

    def test_cleanup(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        cp = PivotCheckpointManager(ckpt_dir, model="gpt-5.4-nano")
        cp.save_batch([PivotRating("t1", 0, 0.9, "pivot", "test")])
        assert ckpt_dir.exists()

        cp.cleanup()
        assert not ckpt_dir.exists()

    def test_manifest_is_atomic(self, tmp_path):
        cp = PivotCheckpointManager(tmp_path / "ckpt", model="gpt-5.4-nano")
        cp.save_batch([PivotRating("t1", 0, 0.9, "pivot", "test")])

        with cp.manifest_path.open() as f:
            manifest = json.load(f)
        assert manifest["model"] == "gpt-5.4-nano"
        # No .tmp file left behind
        assert not cp.manifest_path.with_suffix(".tmp").exists()

    def test_jsonl_format(self, tmp_path):
        cp = PivotCheckpointManager(tmp_path / "ckpt", model="gpt-5.4-nano")
        cp.save_batch([
            PivotRating("t1", 0, 0.9, "pivot", "test1"),
            PivotRating("t2", 1, 0.3, "trivial", "test2"),
        ])

        lines = (cp.checkpoint_dir / cp._RATINGS_FILE).read_text().strip().split("\n")
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert row["trace_id"] == "t1"
        assert row["importance_score"] == 0.9


class TestProgressCallback:
    def test_callback_receives_progress(self):
        """Verify apply_pivot_filter calls progress_callback with pre-computed ratings."""
        examples = [
            _example(turn_index=i, trace_id=f"t{i}", completion_content="A" * 100)
            for i in range(20)
        ]
        traces = [_trace(trace_id=f"t{i}") for i in range(20)]
        ratings = [
            PivotRating(f"t{i}", i, 0.9, "pivot", "test") for i in range(20)
        ]

        # With pre-computed ratings, no LLM calls happen and no progress
        # callback is invoked (ratings skip the LLM path entirely).
        calls: list[tuple[int, int]] = []
        result = apply_pivot_filter(
            examples, traces, ratings=ratings, threshold=0.6,
            progress_callback=lambda done, total: calls.append((done, total)),
        )
        assert len(result.kept) == 20
        # Pre-computed ratings bypass the callback (no LLM work to report)
        assert calls == []
