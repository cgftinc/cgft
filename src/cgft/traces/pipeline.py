"""User-facing traces pipeline — normalised traces → training examples.

Mirrors the ergonomics of ``cgft.qa_generation.cgft_pipeline.CgftPipeline``:
configure once, call ``.run()``, get ``{"train_dataset", "eval_dataset",
"stats"}`` back.

Scope: takes already-normalised traces as input.  Fetching from providers
(Braintrust, Langfuse, etc.) is a separate concern handled by the
``TraceAdapter`` implementations in ``cgft.traces.<provider>``.

Design notes:
- Scalar/bool knobs for simple filters (``min_completion_chars=40``) —
  wrapping each in a one-knob dataclass is bloat.
- Dataclass configs only where a stage has 3+ knobs or makes LLM calls
  (currently just pivot).
- Named stages in fixed order.  ``apply_filters``'s list-of-tuples API
  stays available for power users who need custom ordering or new
  filters, but most users should hit this pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cgft.traces.adapter import NormalizedTrace
from cgft.traces.processing import (
    MIN_TRAIN_SAMPLES,
    apply_filters,
    build_training_examples,
    detect_system_prompt,
    detect_tools,
    split_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class PivotConfig:
    """LLM-driven importance classification — earns a dataclass because
    it bundles an LLM client, model, threshold, and concurrency knob.

    Off by default in ``TracesPipeline`` because it costs per-example LLM
    calls. Set ``pipeline.pivot = PivotConfig(llm_client=..., model=...)``
    to enable.
    """

    llm_client: Any
    """AsyncOpenAI-compatible client."""
    model: str = "gpt-5.4-nano"
    threshold: float = 0.6
    anchor_fraction: float = 0.1
    max_concurrency: int = 20
    checkpoint_dir: Path | None = None


@dataclass
class TracesPipeline:
    """Convert normalised traces into train/eval training examples.

    Example
    -------
    >>> from cgft.traces.pipeline import TracesPipeline
    >>> pipeline = TracesPipeline(
    ...     traces=traces,
    ...     min_completion_chars=40,
    ...     tool_relay=0.8,
    ...     dedup=0.85,
    ...     target_examples=1000,
    ...     output_dir="outputs/agentic",
    ... )
    >>> result = pipeline.run()
    >>> train, eval_ = result["train_dataset"], result["eval_dataset"]

    Stages, in order:
        1. Detect system prompt + tool schemas (skippable if provided).
        2. Build training examples by windowing each trace's messages.
        3. Apply heuristic filters (``min_completion_chars``, ``tool_relay``,
           ``excluded_tools``, ``dedup``).
        4. Optional LLM pivot filter (``pivot=PivotConfig(...)``).
        5. Cap total kept examples at ``target_examples``.
        6. Train/eval split (``train_eval_split`` fraction).
        7. Write ``train.jsonl`` / ``eval.jsonl`` / ``metadata.json`` if
           ``output_dir`` is set.

    Parameters
    ----------
    traces
        Already-fetched + normalised traces (from any source).
    system_prompt
        System prompt baked into each example.  Defaults to the
        auto-detected majority prompt from ``traces``.  Pass ``""`` to
        skip detection and train without a system prompt.
    tools
        Tool schemas exposed to the model.  Defaults to auto-detected from
        observed tool calls.
    min_completion_chars
        Drop examples whose completion is shorter than this.  ``None`` to
        disable.
    tool_relay
        Drop examples where the assistant just relays tool output (token
        overlap above threshold).  ``None`` to disable.
    excluded_tools
        Skip examples whose last assistant step calls one of these tools.
    dedup
        Drop near-duplicate examples (Jaccard similarity above threshold).
        ``None`` to disable.  Must run last among filters (dataset-level).
    pivot
        Optional LLM importance filter.  ``None`` to skip.
    target_examples
        Cap on total kept examples *after* filtering.  Downstream split is
        computed from ``train_eval_split``.
    train_eval_split
        Fraction of kept examples that go to train; the rest go to eval.
        Default 0.9 matches QA-gen defaults.
    random_seed
        For reproducible capping + splitting.
    output_dir
        If set, writes JSONL + metadata here.  Otherwise pipeline only
        returns the in-memory result.
    """

    traces: list[NormalizedTrace]
    system_prompt: str | None = None
    tools: list[dict[str, Any]] | None = None
    min_completion_chars: int | None = 40
    tool_relay: float | None = 0.8
    excluded_tools: list[str] = field(default_factory=list)
    dedup: float | None = 0.85
    pivot: PivotConfig | None = None
    target_examples: int = 1000
    train_eval_split: float = 0.9
    random_seed: int = 42
    output_dir: str | Path | None = None

    def run(self) -> dict[str, Any]:
        """Execute the pipeline.  Returns ``{train_dataset, eval_dataset, stats}``."""
        # 1. Detection ------------------------------------------------------
        detected_sp = detect_system_prompt(self.traces) if self.system_prompt is None else None
        sp_text = (
            self.system_prompt
            if self.system_prompt is not None
            else (detected_sp.prompt if detected_sp else "")
        )
        detected_tools = detect_tools(self.traces) if self.tools is None else None

        # 2. Build examples -------------------------------------------------
        # ``include_system_prompt=False`` strips the leading system message
        # from each prompt so we don't duplicate it — the trainer appends
        # sp_text separately via env config.
        examples = build_training_examples(self.traces, include_system_prompt=False)
        logger.info("Built %d training examples from %d traces", len(examples), len(self.traces))

        # 3. Heuristic + dataset-level filters ------------------------------
        stages: list[tuple[str, dict[str, Any]]] = []
        if self.min_completion_chars is not None:
            stages.append(("heuristic", {"min_completion_chars": self.min_completion_chars}))
        if self.tool_relay is not None:
            stages.append(("tool_relay", {"overlap_threshold": self.tool_relay}))
        if self.excluded_tools:
            stages.append(("tool_calls", {"exclude_tools": list(self.excluded_tools)}))
        if self.dedup is not None:
            stages.append(("dedup", {"similarity_threshold": self.dedup}))

        filter_result = apply_filters(examples, stages)
        kept = filter_result.kept
        logger.info("After filters: %d kept, %d dropped", len(kept), len(filter_result.dropped))

        # 4. Optional LLM pivot filter --------------------------------------
        pivot_stats: dict[str, Any] | None = None
        if self.pivot is not None:
            # Local import — avoid pulling openai into sys.modules when pivot
            # is not used (e.g. tests, CLI dry-runs).
            from cgft.traces.pivot import apply_pivot_filter_async

            pre_pivot = len(kept)
            pivot_result = asyncio.run(
                apply_pivot_filter_async(
                    examples=kept,
                    traces=self.traces,
                    llm_client=self.pivot.llm_client,
                    model=self.pivot.model,
                    threshold=self.pivot.threshold,
                    anchor_fraction=self.pivot.anchor_fraction,
                    max_concurrency=self.pivot.max_concurrency,
                    checkpoint_dir=self.pivot.checkpoint_dir,
                )
            )
            kept = pivot_result.kept
            pivot_stats = {
                "pre_pivot": pre_pivot,
                "post_pivot": len(kept),
                "threshold": self.pivot.threshold,
            }
            logger.info(
                "After pivot: %d kept (threshold=%.2f)", len(kept), self.pivot.threshold
            )

        # 5. Cap to target --------------------------------------------------
        if self.target_examples and len(kept) > self.target_examples:
            import random

            rng = random.Random(self.random_seed)
            kept = rng.sample(kept, self.target_examples)

        # 6. Split ----------------------------------------------------------
        train_count = max(int(round(len(kept) * self.train_eval_split)), MIN_TRAIN_SAMPLES)
        eval_count = len(kept) - train_count
        if len(kept) < MIN_TRAIN_SAMPLES:
            raise ValueError(
                f"Only {len(kept)} examples survived filtering; need at least "
                f"{MIN_TRAIN_SAMPLES}.  Loosen filters, or lower thresholds, "
                f"or fetch more traces."
            )
        train_data, eval_data = split_dataset(
            kept, train_count, eval_count, seed=self.random_seed
        )

        # 7. Stats + optional JSONL write -----------------------------------
        stats = {
            "total_traces": len(self.traces),
            "examples_built": len(examples),
            "kept_after_filters": len(filter_result.kept),
            "final_kept": len(kept),
            "train_count": len(train_data),
            "eval_count": len(eval_data),
            "detected_system_prompt": (
                {
                    "prompt": detected_sp.prompt,
                    "count": detected_sp.count,
                    "total_traces": detected_sp.total_traces,
                    "n_variants": len(detected_sp.variants),
                }
                if detected_sp
                else None
            ),
            "detected_tools": (
                [
                    {"name": t.name, "call_count": t.call_count}
                    for t in detected_tools.tools
                ]
                if detected_tools
                else None
            ),
            "pivot": pivot_stats,
        }

        if self.output_dir:
            out = Path(self.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            _write_jsonl(out / "train.jsonl", train_data)
            _write_jsonl(out / "eval.jsonl", eval_data)
            with (out / "metadata.json").open("w") as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info("Wrote %d train + %d eval examples to %s", len(train_data), len(eval_data), out)

        return {
            "train_dataset": train_data,
            "eval_dataset": eval_data,
            "stats": stats,
            "system_prompt": sp_text,
            "tools": (
                self.tools
                if self.tools is not None
                else [
                    {"name": t.name, "param_keys": sorted(list(t.param_keys))}
                    for t in (detected_tools.tools if detected_tools else [])
                ]
            ),
        }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
