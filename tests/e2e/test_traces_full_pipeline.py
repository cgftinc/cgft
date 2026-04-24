"""End-to-end agentic-trace pipeline against live services.

Replicates the exact sequence the wizard runs when a user clicks
Link → Generate dataset → Build env → Launch:

    1. BraintrustTraceAdapter.fetch_traces(project_id)      [Modal: trace_fetch]
    2. TracesPipeline(traces=...).run()                     [Modal: trace_generate]
    3. train(env_class=CustomTraceEnv, ..., dry_run=True)   [Modal: build-env + submit]
       — bundles the env via cloudpickle, uploads the dataset, runs
       rollout-server validation remotely, then stops before SkyPilot.

Mirrors ``tests/e2e/test_rag_providers_env.py`` for RAG providers.

Env vars:
    BT_API_KEY, BT_PROJECT_ID  — Braintrust (project needs ≥ 30 traces)
    CGFT_API_KEY               — platform + LLM relay

Run manually::

    BT_API_KEY=... BT_PROJECT_ID=... CGFT_API_KEY=... \\
        pytest -m e2e tests/e2e/test_traces_full_pipeline.py -v -s
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import StandardizedExample, ToolDefinition

import cgft
from cgft.traces import ImportanceFilterConfig, TracesPipeline
from cgft.traces.braintrust.adapter import BraintrustTraceAdapter
from cgft.trainer.pipeline import train

pytestmark = pytest.mark.e2e


# ── Config ────────────────────────────────────────────────────────────────

BT_API_KEY = os.environ.get("BT_API_KEY", "")
BT_PROJECT_ID = os.environ.get("BT_PROJECT_ID", "")
CGFT_API_KEY = os.environ.get("CGFT_API_KEY", "")
CGFT_BASE_URL = os.environ.get("CGFT_BASE_URL", "https://app.cgft.io")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://llm.cgft.io/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-nano")


def _step(message: str) -> None:
    print(f"  [traces-full] → {message}", flush=True)
    sys.stdout.flush()


def _require(var: str, value: str) -> str:
    if not value:
        pytest.skip(f"{var} required for full trace e2e")
    return value


# ── CustomTraceEnv — mirrors the wizard codegen verbatim ─────────────────

_SYSTEM_PROMPT_PLACEHOLDER = "you are a helpful assistant."


class CustomTraceEnv(BaseEnv):
    """Mirrors lib/codegen/traces.ts::generateEnvDefinition exactly.

    ``_SYSTEM_PROMPT`` is set by the test at runtime from the detected
    prompt so the pickled env carries the same text the wizard would
    emit.  A class-level attribute (not instance-level) so cloudpickle
    captures it without needing to pass through ``env_args``.
    """

    _SYSTEM_PROMPT: str = _SYSTEM_PROMPT_PLACEHOLDER

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.system_prompt = self._SYSTEM_PROMPT

    async def list_tools(self) -> list[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        return ""

    async def compute_reward(
        self,
        rollout_id: str,
        completion: str | list[dict[str, Any]],
        ground_truth: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Stub reward. Wizard users replace this with rubric / judge code;
        the e2e just needs a valid dict so remote validation succeeds."""
        return {"stub_match": 0.0}

    @classmethod
    def dataset_preprocess(cls, example: dict, **kwargs: Any) -> StandardizedExample:
        """Same shape as TRACE_DATASET_PREPROCESS in the codegen."""
        prompt_messages = example.get("prompt_messages") or example.get("prompt") or []
        completion_messages = example.get("completion_messages") or []
        if isinstance(prompt_messages, list) and prompt_messages and isinstance(
            prompt_messages[0], dict
        ):
            prompt_text = "\n".join(
                m.get("content", "") for m in prompt_messages if m.get("content")
            )
        else:
            prompt_text = str(prompt_messages) if prompt_messages else ""

        ground_truth_raw = example.get("ground_truth")
        if isinstance(ground_truth_raw, dict):
            ground_truth_text = ground_truth_raw.get("content", "") or ""
        elif isinstance(ground_truth_raw, list):
            ground_truth_text = "\n".join(
                m.get("content", "") for m in ground_truth_raw if isinstance(m, dict)
            )
        else:
            ground_truth_text = str(ground_truth_raw or "")

        return StandardizedExample(
            prompt=prompt_text,
            ground_truth=ground_truth_text,
            init_rollout_args={
                "prompt_messages": prompt_messages,
                "completion_messages": completion_messages,
                "trace_id": example.get("trace_id")
                or example.get("init_rollout_args", {}).get("trace_id"),
                "turn_index": example.get("turn_index")
                or example.get("init_rollout_args", {}).get("turn_index"),
            },
        )


# ── The full-pipeline test ───────────────────────────────────────────────


class TestBraintrustTraceFullPipeline:
    """One test, four stages, live APIs."""

    def test_full_wizard_path(self, tmp_path: Path) -> None:
        _require("BT_API_KEY", BT_API_KEY)
        _require("BT_PROJECT_ID", BT_PROJECT_ID)
        _require("CGFT_API_KEY", CGFT_API_KEY)

        # ── Stage 1: fetch ───────────────────────────────────────────────
        t_fetch = time.monotonic()
        _step("Stage 1/4: BraintrustTraceAdapter.fetch_traces()")
        adapter = BraintrustTraceAdapter(api_key=BT_API_KEY)
        traces, _ = adapter.fetch_traces(BT_PROJECT_ID, limit=30, verbose=False)
        _step(f"Stage 1/4: fetched {len(traces)} traces ({time.monotonic()-t_fetch:.1f}s)")
        if len(traces) < 30:
            pytest.skip(f"project has only {len(traces)} traces; need ≥ 30")

        # ── Stage 2: pipeline ────────────────────────────────────────────
        t_pipe = time.monotonic()
        _step("Stage 2/4: TracesPipeline.run() with all filters + importance")
        from openai import AsyncOpenAI

        llm_client = AsyncOpenAI(api_key=CGFT_API_KEY, base_url=LLM_BASE_URL)
        pipeline_result = TracesPipeline(
            traces=traces,
            min_completion_chars=40,
            max_tool_output_overlap=0.8,
            max_example_similarity=0.85,
            importance_filter=ImportanceFilterConfig(
                llm_client=llm_client,
                model=LLM_MODEL,
                min_importance_score=0.5,
                max_concurrency=8,
            ),
            max_examples=60,
            train_fraction=0.9,
            output_dir=tmp_path,
            verbose=False,
        ).run()
        train_data = pipeline_result["train_dataset"]
        eval_data = pipeline_result["eval_dataset"]
        system_prompt = pipeline_result["system_prompt"]
        _step(
            f"Stage 2/4: {len(train_data)} train, {len(eval_data)} eval "
            f"({time.monotonic()-t_pipe:.1f}s)"
        )

        # Artifacts landed — this is what trace_generate writes to blob storage
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "eval.jsonl").exists()
        assert (tmp_path / "metadata.json").exists()

        # ── Stage 3: JSONL round-trip ────────────────────────────────────
        # The wizard runs fetch + generate in separate Modal jobs, so the
        # train.jsonl written by one is reloaded by the next.  Exercise
        # that seam: reload and feed reloaded rows into train().
        _step("Stage 3/4: reloading train/eval JSONL from disk (round-trip)")
        import json as _json

        with (tmp_path / "train.jsonl").open() as f:
            reloaded_train = [_json.loads(line) for line in f if line.strip()]
        with (tmp_path / "eval.jsonl").open() as f:
            reloaded_eval = [_json.loads(line) for line in f if line.strip()]
        assert len(reloaded_train) == len(train_data), "train row count drifted in JSONL round-trip"
        assert len(reloaded_eval) == len(eval_data), "eval row count drifted in JSONL round-trip"

        # ── Stage 4: env bundle + remote validate ───────────────────────
        _step("Stage 4/4: train(dry_run=True, validate_env_remotely=True)")
        CustomTraceEnv._SYSTEM_PROMPT = system_prompt or _SYSTEM_PROMPT_PLACEHOLDER

        t_train = time.monotonic()
        result = train(
            env_class=CustomTraceEnv,
            env_args={},
            train_dataset=reloaded_train,
            eval_dataset=reloaded_eval,
            prefix="e2e-traces",
            api_key=CGFT_API_KEY,
            base_url=CGFT_BASE_URL,
            local_modules=[cgft],
            experiment_name=f"e2e-traces-{int(time.time())}",
            pip_dependencies=["openai"],
            validate_env=False,
            validate_env_remotely=True,
            validation_model=LLM_MODEL,
            dry_run=True,
            show_summary=True,
        )
        _step(
            f"Stage 4/4: status={result.get('status')!r}  "
            f"({time.monotonic()-t_train:.1f}s)"
        )

        assert isinstance(result, dict)
        assert result.get("status") == "validated", (
            f"Expected 'validated' from train(dry_run=True), got: {result}"
        )
