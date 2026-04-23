"""End-to-end agentic-trace pipeline against live Braintrust.

Two scenarios:

1. **Full fetch** — pull every trace in the project and verify the count
   matches what BTQL ``count_traces`` reports.  This catches the bug where
   transient BTQL errors silently drop a page mid-pagination (symptom:
   "fetches 500 when the project has 509").

2. **All filters including importance_filter** — on a much smaller slice
   (10 traces), run the whole ``TracesPipeline`` with
   ``min_completion_chars`` + ``max_tool_output_overlap`` +
   ``max_example_similarity`` + ``importance_filter`` all enabled.
   Proves every filter stage actually executes on real data, and that
   the LLM-backed importance path works end-to-end.  The 10-trace slice
   keeps per-turn LLM calls bounded so the test finishes fast.

Env vars:
    BT_API_KEY          — Braintrust API key
    BT_PROJECT_ID       — Braintrust project ID with ≥ 500 traces
    CGFT_API_KEY        — (importance filter only) CGFT platform key for LLM relay
    OPENAI_API_KEY      — (importance filter only) fallback if CGFT key missing
    LLM_BASE_URL        — (optional) override LLM endpoint
    LLM_MODEL           — (optional, default gpt-5.4-nano)

Run manually::

    BT_API_KEY=... BT_PROJECT_ID=... CGFT_API_KEY=... \\
        pytest -m e2e tests/e2e/test_traces_pipeline.py -v -s
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from cgft.traces import ImportanceFilterConfig, TracesPipeline
from cgft.traces.braintrust.adapter import BraintrustTraceAdapter
from cgft.traces.processing import MIN_TRAIN_SAMPLES

pytestmark = pytest.mark.e2e


BT_API_KEY = os.environ.get("BT_API_KEY", "")
BT_PROJECT_ID = os.environ.get("BT_PROJECT_ID", "")


def _step(message: str) -> None:
    print(f"  [traces-e2e] → {message}", flush=True)
    sys.stdout.flush()


def _require_bt_creds() -> None:
    if not BT_API_KEY or not BT_PROJECT_ID:
        pytest.skip("BT_API_KEY and BT_PROJECT_ID required for trace e2e tests")


# ── 1. Full-fetch count parity ───────────────────────────────────────────


class TestBraintrustFetchAll:
    """Pull ALL traces and assert the fetched count matches the BTQL count.

    This is the regression guard for the "fetched 500 when the project has
    509" failure mode: a transient BTQL error (429/500/502/...) mid-
    pagination used to bail out the whole loop, silently returning fewer
    traces.  With retry-on-500 landed in ``cgft/traces/http.py`` and the
    break-with-partial logic in ``_fetch_via_btql``, this test should
    pass reliably.
    """

    def test_fetch_all_matches_count_traces(self) -> None:
        _require_bt_creds()
        adapter = BraintrustTraceAdapter(api_key=BT_API_KEY)

        _step("count_traces(project_id) — paginates BTQL SELECT count(*)…")
        t0 = time.monotonic()
        count = adapter.count_traces(BT_PROJECT_ID)
        _step(f"count_traces = {count}  ({time.monotonic()-t0:.1f}s)")

        if count < 500:
            pytest.skip(
                f"Project has only {count} traces; need ≥ 500 to exercise "
                "multi-page BTQL fetch."
            )

        _step(f"fetch_traces(limit={count}) — full paginated pull…")
        t0 = time.monotonic()
        traces, _ = adapter.fetch_traces(BT_PROJECT_ID, limit=count)
        elapsed = time.monotonic() - t0
        fetched = len(traces)
        _step(f"fetched = {fetched}  ({elapsed:.1f}s)")

        # Very tight delta: a handful of new traces may legitimately
        # arrive between count and fetch, but losing 9 out of 509 is the
        # exact failure mode we're guarding against.  Previous test used
        # 2% which masked that.
        delta = count - fetched
        assert -3 <= delta <= 3, (
            f"count_traces={count} but fetch_traces returned {fetched} "
            f"(missing {delta}).  Likely partial-fetch bug — BTQL 500/502 "
            "during pagination silently dropped a page."
        )

        # No duplicates
        ids = [t.id for t in traces]
        assert len(ids) == len(set(ids)), (
            f"Duplicate trace IDs in fetch: {len(ids) - len(set(ids))} dupes"
        )

        # Every trace has at least 1 message — sanity check on the
        # span-grouping + message-extraction path
        empty = [t.id for t in traces if not t.messages]
        assert not empty, f"{len(empty)} traces came back with no messages (e.g. {empty[:3]})"

        _step(
            f"parity OK: count={count} fetched={fetched} delta={delta}  "
            f"throughput={fetched/elapsed:.0f} traces/sec"
        )


# ── 2. Small-scale run with ALL filters + importance filter ──────────────


def _fetch_small_slice(limit: int = 10):
    adapter = BraintrustTraceAdapter(api_key=BT_API_KEY)
    t0 = time.monotonic()
    _step(f"fetching {limit} traces for small-scale test…")
    traces, _ = adapter.fetch_traces(BT_PROJECT_ID, limit=limit)
    _step(f"fetched {len(traces)} traces  ({time.monotonic()-t0:.1f}s)")
    if len(traces) < limit:
        pytest.skip(f"Project has only {len(traces)} traces; need ≥ {limit}")
    return traces


def _build_llm_client():
    """Build an AsyncOpenAI client against CGFT's relay or OpenAI direct."""
    llm_api_key = os.environ.get("CGFT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not llm_api_key:
        pytest.skip(
            "CGFT_API_KEY or OPENAI_API_KEY required for importance filter "
            "(hits LLM relay for per-turn importance ratings)"
        )

    base_url = os.environ.get("LLM_BASE_URL")
    # Default to CGFT's LLM relay when a CGFT_API_KEY was used.
    if not base_url and os.environ.get("CGFT_API_KEY"):
        base_url = "https://llm.cgft.io/v1"

    from openai import AsyncOpenAI

    kwargs: dict = {"api_key": llm_api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


class TestBraintrustPipelineAllFilters:
    """Smaller run, every filter turned on — including the LLM importance filter.

    The full chain: detection → build_training_examples → min_completion_chars
    → max_tool_output_overlap → max_example_similarity → importance_filter
    → max_examples cap → split → JSONL.  Asserts each stage actually did
    work (not a silent no-op) and that the final JSONL shape is correct.
    """

    def test_all_filters_including_importance(self, tmp_path: Path) -> None:
        _require_bt_creds()
        llm_client = _build_llm_client()
        traces = _fetch_small_slice(limit=10)

        t0 = time.monotonic()
        _step(
            "TracesPipeline.run() with min_completion_chars + "
            "max_tool_output_overlap + max_example_similarity + "
            "importance_filter ALL on"
        )
        result = TracesPipeline(
            traces=traces,
            min_completion_chars=40,
            max_tool_output_overlap=0.8,
            max_example_similarity=0.85,
            importance_filter=ImportanceFilterConfig(
                llm_client=llm_client,
                model=os.environ.get("LLM_MODEL", "gpt-5.4-nano"),
                min_importance_score=0.5,
                anchor_fraction=0.1,
                max_concurrency=8,
            ),
            max_examples=60,
            train_fraction=0.9,
            output_dir=tmp_path,
        ).run()
        elapsed = time.monotonic() - t0

        stats = result["stats"]
        _step(
            f"built={stats['examples_built']}  "
            f"after_heuristic_chain={stats['kept_after_filters']}  "
            f"after_importance={stats['importance_filter']['post']}  "
            f"final={stats['final_kept']}  "
            f"train={stats['train_count']}  eval={stats['eval_count']}  "
            f"total={elapsed:.1f}s"
        )

        # Every filter had the opportunity to drop.  If kept_after_filters
        # equals examples_built, the heuristic chain silently no-opped —
        # a regression we care about.
        assert stats["examples_built"] > stats["kept_after_filters"], (
            f"Filter chain dropped nothing: {stats['examples_built']} → "
            f"{stats['kept_after_filters']}.  One of min_completion_chars/"
            "max_tool_output_overlap/max_example_similarity silently broke."
        )

        # Importance filter ran and produced stats
        assert stats["importance_filter"] is not None, (
            "stats['importance_filter'] is None — the LLM stage was skipped"
        )
        assert stats["importance_filter"]["pre"] >= stats["importance_filter"]["post"], (
            "importance_filter's post count exceeds pre — stage is wrong-way-round"
        )

        # System-prompt detection fired
        assert stats["detected_system_prompt"] is not None, (
            "No system prompt detected on a 30-trace slice — detection broken "
            "or project has no system prompts (re-pick project)"
        )

        # MIN_TRAIN_SAMPLES guarantee holds
        train = result["train_dataset"]
        eval_ = result["eval_dataset"]
        assert len(train) >= MIN_TRAIN_SAMPLES, (
            f"Only {len(train)} train examples survived full filtering; "
            f"need ≥ {MIN_TRAIN_SAMPLES}.  min_importance_score too aggressive?"
        )
        assert len(eval_) >= 1

        # JSONL row shape (matches TrainingExample.to_jsonl_dict())
        row = train[0]
        for key in ("prompt", "ground_truth", "init_rollout_args"):
            assert key in row, f"missing {key} in train row: {list(row.keys())}"
        assert isinstance(row["prompt"], list)
        assert isinstance(row["ground_truth"], dict)
        rargs = row["init_rollout_args"]
        for key in ("trace_id", "turn_index"):
            assert key in rargs, f"missing {key} in init_rollout_args"

        # Artifacts on disk
        for name in ("train.jsonl", "eval.jsonl", "metadata.json"):
            assert (tmp_path / name).exists(), f"missing {name}"
        train_lines = (tmp_path / "train.jsonl").read_text().splitlines()
        assert len(train_lines) == len(train), (
            f"train.jsonl has {len(train_lines)} lines, dataset has {len(train)}"
        )
