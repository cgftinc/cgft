"""End-to-end agentic-trace pipeline against live Braintrust.

Smoke test for the full ``BraintrustTraceAdapter.fetch_traces()`` →
``TracesPipeline.run()`` flow, tuned for minimum cost + time:

  * Fetch a small page of traces (default 50)
  * No LLM pivot filter (pure heuristic path)
  * dedup disabled by default — adjacent Braintrust traces often differ
    only in arguments, which tanks the dataset on a Jaccard threshold

Runs in ~5-15 seconds depending on Braintrust latency.  Zero LLM tokens.

Env vars:
    BT_API_KEY — Braintrust API key
    BT_PROJECT_ID — Braintrust project ID with ≥ 20 traces

Run manually::

    BT_API_KEY=... BT_PROJECT_ID=... pytest -m e2e tests/e2e/test_traces_pipeline.py -v
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from cgft.traces import PivotConfig, TracesPipeline
from cgft.traces.braintrust.adapter import BraintrustTraceAdapter
from cgft.traces.processing import MIN_TRAIN_SAMPLES

pytestmark = pytest.mark.e2e


BT_API_KEY = os.environ.get("BT_API_KEY", "")
BT_PROJECT_ID = os.environ.get("BT_PROJECT_ID", "")

# Enough traces for MIN_TRAIN_SAMPLES=16 to survive the 0.9/0.1 split even
# after heuristic drops.  Each trace yields ≥ 1 training example.
MIN_TRACES_REQUIRED = 30


def _step(message: str) -> None:
    print(f"  [braintrust-traces] → {message}", flush=True)
    sys.stdout.flush()


def _require_creds() -> None:
    if not BT_API_KEY or not BT_PROJECT_ID:
        pytest.skip("BT_API_KEY and BT_PROJECT_ID required for e2e trace tests")


def _fetch_traces(limit: int = 50):
    adapter = BraintrustTraceAdapter(api_key=BT_API_KEY)
    t0 = time.monotonic()
    _step(f"fetching up to {limit} traces via BTQL…")
    traces, _ = adapter.fetch_traces(BT_PROJECT_ID, limit=limit)
    _step(f"fetched {len(traces)} traces  ({time.monotonic() - t0:.1f}s)")

    if len(traces) < MIN_TRACES_REQUIRED:
        pytest.skip(
            f"Project has only {len(traces)} traces; need ≥ {MIN_TRACES_REQUIRED} "
            f"to exercise the pipeline (split requires ≥ {MIN_TRAIN_SAMPLES} train)."
        )
    return traces


def _assert_jsonl_row_shape(row: dict) -> None:
    """Match TrainingExample.to_jsonl_dict() contract (see processing.py:167)."""
    assert "prompt" in row, f"Missing 'prompt' in {row.keys()}"
    assert "ground_truth" in row, f"Missing 'ground_truth' in {row.keys()}"
    assert "init_rollout_args" in row, f"Missing 'init_rollout_args' in {row.keys()}"
    assert isinstance(row["prompt"], list), "prompt must be a list of message dicts"
    assert isinstance(row["ground_truth"], dict), "ground_truth must be a dict"
    # init_rollout_args carries trace_id + turn_index so compute_reward() can
    # cross-reference the original trace.
    rargs = row["init_rollout_args"]
    assert "trace_id" in rargs
    assert "turn_index" in rargs


# ── Pipeline smoke ────────────────────────────────────────────────────────


class TestBraintrustTracesPipelineE2E:
    def test_fetch_then_pipeline_run(self, tmp_path: Path) -> None:
        """Full path: fetch → build → heuristic filter → split → JSONL."""
        _require_creds()
        traces = _fetch_traces(limit=50)

        t0 = time.monotonic()
        _step("TracesPipeline.run() — heuristic filters only")
        result = TracesPipeline(
            traces=traces,
            # Real Braintrust projects often have near-duplicate trace
            # content (same underlying agent); dedup at 0.85 can drop too
            # many.  Turn it off for the smoke path.
            dedup=None,
            target_examples=50,
            output_dir=tmp_path,
        ).run()
        _step(f"pipeline returned  ({time.monotonic() - t0:.1f}s)")

        train = result["train_dataset"]
        eval_ = result["eval_dataset"]
        stats = result["stats"]

        _step(
            f"built={stats['examples_built']}  "
            f"kept_after_filters={stats['kept_after_filters']}  "
            f"final_kept={stats['final_kept']}  "
            f"train={stats['train_count']}  eval={stats['eval_count']}"
        )

        # Shape assertions
        assert len(train) >= MIN_TRAIN_SAMPLES, f"Only {len(train)} train examples"
        assert len(eval_) >= 1, "Need at least 1 eval example"
        assert stats["total_traces"] == len(traces)
        assert stats["examples_built"] > 0

        _assert_jsonl_row_shape(train[0])
        _assert_jsonl_row_shape(eval_[0])

        # JSONL artifacts written to output_dir
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "eval.jsonl").exists()
        assert (tmp_path / "metadata.json").exists()
        train_lines = (tmp_path / "train.jsonl").read_text().splitlines()
        assert len(train_lines) == len(train)

    def test_detection_identifies_system_prompt(self) -> None:
        """Auto-detection should surface either a system prompt or None, not crash."""
        _require_creds()
        traces = _fetch_traces(limit=30)
        result = TracesPipeline(
            traces=traces,
            dedup=None,
            target_examples=40,
        ).run()

        sp = result["stats"]["detected_system_prompt"]
        # Either the project has a detectable system prompt, or it doesn't —
        # both are valid real-world cases.  What matters is the pipeline
        # didn't blow up on the detection step.
        if sp is not None:
            assert sp["prompt"]
            assert sp["count"] >= 1
            _step(f"detected system prompt (len={len(sp['prompt'])}, count={sp['count']})")
        else:
            _step("no system prompt in project — detection returned None (expected)")

    def test_disabling_heuristics_keeps_more_examples(self) -> None:
        """Turning filters off produces ≥ the count we got with them on."""
        _require_creds()
        traces = _fetch_traces(limit=30)

        with_filters = TracesPipeline(
            traces=traces,
            min_completion_chars=40,
            tool_relay=0.8,
            dedup=None,
            target_examples=200,  # high enough that cap doesn't bind
        ).run()
        without_filters = TracesPipeline(
            traces=traces,
            min_completion_chars=None,
            tool_relay=None,
            dedup=None,
            target_examples=200,
        ).run()

        _step(
            f"with filters: {with_filters['stats']['final_kept']}, "
            f"without: {without_filters['stats']['final_kept']}"
        )
        assert (
            without_filters["stats"]["final_kept"] >= with_filters["stats"]["final_kept"]
        ), "Disabling filters should not reduce example count"


# ── Pivot (optional, gated on OPENAI/LLM creds) ──────────────────────────


class TestBraintrustTracesPipelinePivotE2E:
    """Gated separately since pivot costs LLM tokens."""

    def test_pivot_filter_runs_end_to_end(self, tmp_path: Path) -> None:
        _require_creds()
        llm_api_key = os.environ.get("CGFT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        llm_base_url = os.environ.get(
            "LLM_BASE_URL",
            "https://llm.cgft.io/v1" if os.environ.get("CGFT_API_KEY") else None,
        )
        if not llm_api_key:
            pytest.skip("CGFT_API_KEY or OPENAI_API_KEY required for pivot e2e")

        from openai import AsyncOpenAI

        client_kwargs = {"api_key": llm_api_key}
        if llm_base_url:
            client_kwargs["base_url"] = llm_base_url
        llm_client = AsyncOpenAI(**client_kwargs)

        traces = _fetch_traces(limit=40)

        t0 = time.monotonic()
        _step("TracesPipeline.run() with PivotConfig enabled")
        result = TracesPipeline(
            traces=traces,
            dedup=None,
            pivot=PivotConfig(
                llm_client=llm_client,
                model=os.environ.get("LLM_MODEL", "gpt-5.4-nano"),
                threshold=0.5,
                max_concurrency=8,
            ),
            target_examples=50,
            output_dir=tmp_path,
        ).run()
        _step(f"pivot pipeline returned  ({time.monotonic() - t0:.1f}s)")

        stats = result["stats"]
        assert stats["pivot"] is not None, "pivot stats must be populated when pivot is set"
        assert stats["pivot"]["pre_pivot"] >= stats["pivot"]["post_pivot"]
        # Final kept ≥ MIN_TRAIN_SAMPLES for the split to have succeeded
        assert stats["final_kept"] >= MIN_TRAIN_SAMPLES
        _step(
            f"pivot pre={stats['pivot']['pre_pivot']} "
            f"post={stats['pivot']['post_pivot']} "
            f"threshold={stats['pivot']['threshold']}"
        )
