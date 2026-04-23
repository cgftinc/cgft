"""Tests for the user-facing TracesPipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cgft.traces.adapter import NormalizedTrace, TraceMessage
from cgft.traces.pipeline import TracesPipeline
from cgft.traces.processing import MIN_TRAIN_SAMPLES


def _msg(role: str, content: str) -> TraceMessage:
    return TraceMessage(role=role, content=content)


def _trace(idx: int, system: str = "be helpful") -> NormalizedTrace:
    # Each trace yields 2 training examples (2 assistant turns).
    return NormalizedTrace(
        id=f"t{idx}",
        messages=[
            _msg("system", system),
            _msg("user", f"question {idx} part 1"),
            _msg("assistant", f"answer {idx} part 1 — a long answer well over forty chars so it survives the heuristic filter"),
            _msg("user", f"question {idx} part 2"),
            _msg("assistant", f"answer {idx} part 2 — another long answer also well over forty chars for the heuristic"),
        ],
        scores={},
    )


class TestTracesPipelineSmoke:
    def test_minimum_viable_run(self, tmp_path: Path) -> None:
        """Pipeline runs end-to-end on a small set and produces stats + jsonl."""
        traces = [_trace(i) for i in range(40)]
        result = TracesPipeline(
            traces=traces,
            target_examples=60,
            train_eval_split=0.9,
            output_dir=tmp_path,
            # dedup off — our synthetic traces are intentionally similar
            dedup=None,
        ).run()

        assert "train_dataset" in result
        assert "eval_dataset" in result
        assert "stats" in result
        assert len(result["train_dataset"]) >= MIN_TRAIN_SAMPLES
        assert len(result["eval_dataset"]) >= 1

        # Stats look right
        stats = result["stats"]
        assert stats["total_traces"] == 40
        assert stats["examples_built"] >= 40  # at least one per trace
        assert stats["train_count"] == len(result["train_dataset"])
        assert stats["eval_count"] == len(result["eval_dataset"])
        assert stats["detected_system_prompt"]["prompt"] == "be helpful"

        # Files written
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "eval.jsonl").exists()
        assert (tmp_path / "metadata.json").exists()

        # JSONL contents are actual JSON
        with (tmp_path / "train.jsonl").open() as f:
            first_row = json.loads(f.readline())
            assert "prompt" in first_row or "prompt_messages" in first_row

    def test_disabling_filters_via_none(self) -> None:
        traces = [_trace(i) for i in range(20)]
        result = TracesPipeline(
            traces=traces,
            min_completion_chars=None,
            tool_relay=None,
            dedup=None,
            target_examples=100,
        ).run()
        # With all heuristics off, every built example survives
        assert result["stats"]["final_kept"] == result["stats"]["examples_built"]

    def test_raises_when_too_few_examples(self) -> None:
        # Only 2 traces → 4 examples — below MIN_TRAIN_SAMPLES of 16
        traces = [_trace(i) for i in range(2)]
        with pytest.raises(ValueError, match="need at least"):
            TracesPipeline(traces=traces, dedup=None).run()

    def test_explicit_system_prompt_overrides_detection(self) -> None:
        traces = [_trace(i, system="auto-detected prompt") for i in range(20)]
        result = TracesPipeline(
            traces=traces,
            system_prompt="explicit override",
            dedup=None,
        ).run()
        # When user passes explicit system_prompt, detection is skipped.
        assert result["system_prompt"] == "explicit override"
        assert result["stats"]["detected_system_prompt"] is None
