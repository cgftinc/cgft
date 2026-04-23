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
  (currently just the importance filter).
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

from tqdm.auto import tqdm

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


def _print_progress(message: str, *, verbose: bool) -> None:
    """Stage marker printer — uses ``tqdm.write`` so it plays nicely with
    active progress bars (e.g. the importance-filter LLM bar)."""
    if verbose:
        tqdm.write(message)


@dataclass
class ImportanceFilterConfig:
    """LLM-driven importance classification — earns a dataclass because
    it bundles an LLM client, model, threshold, and concurrency knob.

    Off by default in ``TracesPipeline`` because it costs per-example LLM
    calls.  Set ``pipeline.importance_filter = ImportanceFilterConfig(...)``
    to enable.

    The filter rates each assistant turn's "pivotal-ness": does this turn
    drive the conversation forward, or is it filler?  Turns scoring below
    ``min_importance_score`` are dropped — except for an ``anchor_fraction``
    kept below the line so the dataset doesn't collapse to only
    high-signal turns and lose calibration.
    """

    llm_client: Any
    """AsyncOpenAI-compatible client."""
    model: str = "gpt-5.4-nano"
    min_importance_score: float = 0.6
    """Keep turns scoring at least this (0-1).  Below-threshold turns are
    dropped unless sampled as anchors."""
    anchor_fraction: float = 0.1
    """Fraction of *below-threshold* turns retained as anchors to preserve
    calibration.  Prevents the dataset from over-filtering to only
    high-scoring turns."""
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
    ...     max_tool_output_overlap=0.8,
    ...     max_example_similarity=0.85,
    ...     max_examples=1000,
    ...     output_dir="outputs/agentic",
    ... )
    >>> result = pipeline.run()
    >>> train, eval_ = result["train_dataset"], result["eval_dataset"]

    Stages, in order:
        1. Detect system prompt + tool schemas (skippable if provided).
        2. Build training examples by windowing each trace's messages.
        3. Apply heuristic filters (``min_completion_chars``,
           ``max_tool_output_overlap``, ``excluded_tools``,
           ``max_example_similarity``).
        4. Optional LLM importance filter
           (``importance_filter=ImportanceFilterConfig(...)``).
        5. Cap total kept examples at ``max_examples``.
        6. Train/eval split (``train_fraction``).
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
    max_tool_output_overlap
        Drop examples where the assistant's completion overlaps (token
        Jaccard) with preceding tool output above this threshold — i.e.
        the assistant is parroting tool output instead of adding value.
        ``None`` to disable.
    excluded_tools
        Skip examples whose last assistant step calls one of these tools.
    max_example_similarity
        Drop near-duplicate examples whose Jaccard similarity exceeds this
        threshold.  ``None`` to disable.  Dataset-level filter; must run
        last among filters.
    importance_filter
        Optional LLM-driven importance classifier.  ``None`` to skip.
    max_examples
        Cap on total kept examples *after* filtering.  If fewer survive
        filtering you get fewer — no up-sampling.
    train_fraction
        Fraction of kept examples that go to train; the rest go to eval.
        Default 0.9 matches QA-gen defaults.
    random_seed
        For reproducible capping + splitting.
    output_dir
        If set, writes JSONL + metadata here.  Otherwise pipeline only
        returns the in-memory result.
    verbose
        Print ``[N/6]`` stage markers + a tqdm progress bar for the
        importance-filter LLM calls.  Default ``True`` — matches the
        ``CgftPipeline`` QA-gen default for interactive-notebook use.
        Set ``False`` for clean library/CI output.  Progress markers
        use ``tqdm.write`` so they interleave correctly with active bars.
    """

    traces: list[NormalizedTrace]
    system_prompt: str | None = None
    tools: list[dict[str, Any]] | None = None
    min_completion_chars: int | None = 40
    max_tool_output_overlap: float | None = 0.8
    excluded_tools: list[str] = field(default_factory=list)
    max_example_similarity: float | None = 0.85
    importance_filter: ImportanceFilterConfig | None = None
    max_examples: int = 1000
    train_fraction: float = 0.9
    random_seed: int = 42
    output_dir: str | Path | None = None
    verbose: bool = True

    def run(self) -> dict[str, Any]:
        """Execute the pipeline.  Returns ``{train_dataset, eval_dataset, stats}``."""
        v = self.verbose

        # 1. Detection ------------------------------------------------------
        _print_progress(
            f"[1/6] Detecting system prompt + tools across {len(self.traces)} traces...",
            verbose=v,
        )
        detected_sp = detect_system_prompt(self.traces) if self.system_prompt is None else None
        sp_text = (
            self.system_prompt
            if self.system_prompt is not None
            else (detected_sp.prompt if detected_sp else "")
        )
        detected_tools = detect_tools(self.traces) if self.tools is None else None
        if detected_sp is not None:
            _print_progress(
                f"[1/6] Detected system prompt in {detected_sp.count}/{detected_sp.total_traces} "
                f"traces ({len(detected_sp.variants)} variants)",
                verbose=v,
            )
        elif self.system_prompt is not None:
            _print_progress(
                f"[1/6] Using explicit system_prompt (len={len(self.system_prompt)})",
                verbose=v,
            )
        if detected_tools is not None:
            tool_names = [t.name for t in detected_tools.tools]
            _print_progress(
                f"[1/6] Detected {len(tool_names)} tool(s): {', '.join(tool_names) or '(none)'}",
                verbose=v,
            )

        # 2. Build examples -------------------------------------------------
        _print_progress("[2/6] Building training examples from traces...", verbose=v)
        # ``include_system_prompt=False`` strips the leading system message
        # from each prompt so we don't duplicate it — the trainer appends
        # sp_text separately via env config.
        examples = build_training_examples(self.traces, include_system_prompt=False)
        _print_progress(
            f"[2/6] Built {len(examples)} training examples "
            f"(avg {len(examples)/max(len(self.traces),1):.1f} per trace)",
            verbose=v,
        )
        logger.info("Built %d training examples from %d traces", len(examples), len(self.traces))

        # 3. Heuristic + dataset-level filters ------------------------------
        stages: list[tuple[str, dict[str, Any]]] = []
        if self.min_completion_chars is not None:
            stages.append(("heuristic", {"min_completion_chars": self.min_completion_chars}))
        if self.max_tool_output_overlap is not None:
            stages.append(("tool_relay", {"overlap_threshold": self.max_tool_output_overlap}))
        if self.excluded_tools:
            stages.append(("tool_calls", {"exclude_tools": list(self.excluded_tools)}))
        if self.max_example_similarity is not None:
            stages.append(("dedup", {"similarity_threshold": self.max_example_similarity}))

        _print_progress(
            f"[3/6] Applying {len(stages)} filter stage(s): "
            f"{', '.join(name for name, _ in stages) or '(none)'}",
            verbose=v,
        )
        filter_result = apply_filters(examples, stages)
        kept = filter_result.kept
        _print_progress(
            f"[3/6] Filters kept {len(kept)}/{len(examples)} examples "
            f"({len(filter_result.dropped)} dropped)",
            verbose=v,
        )
        logger.info("After filters: %d kept, %d dropped", len(kept), len(filter_result.dropped))

        # 4. Optional LLM importance filter ---------------------------------
        importance_stats: dict[str, Any] | None = None
        if self.importance_filter is not None:
            # Local import — avoid pulling openai into sys.modules when the
            # importance filter is not used (e.g. tests, CLI dry-runs).
            from cgft.traces.pivot import apply_pivot_filter_async

            pre = len(kept)
            cfg = self.importance_filter
            _print_progress(
                f"[4/6] Running importance filter on {pre} examples "
                f"(model={cfg.model}, min_score={cfg.min_importance_score}, "
                f"concurrency={cfg.max_concurrency})...",
                verbose=v,
            )

            # Wire a tqdm bar through pivot.py's progress_callback so the
            # user sees per-batch LLM progress in real time.
            pbar = tqdm(
                total=pre,
                desc="importance filter",
                disable=not v,
                unit="turn",
            )

            def _on_progress(completed: int, total: int) -> None:
                pbar.total = total  # may change due to heuristic trivial-skip
                pbar.n = completed
                pbar.refresh()

            try:
                result = asyncio.run(
                    apply_pivot_filter_async(
                        examples=kept,
                        traces=self.traces,
                        llm_client=cfg.llm_client,
                        model=cfg.model,
                        threshold=cfg.min_importance_score,
                        anchor_fraction=cfg.anchor_fraction,
                        max_concurrency=cfg.max_concurrency,
                        checkpoint_dir=cfg.checkpoint_dir,
                        progress_callback=_on_progress,
                    )
                )
            finally:
                pbar.close()

            kept = result.kept
            importance_stats = {
                "pre": pre,
                "post": len(kept),
                "min_importance_score": cfg.min_importance_score,
            }
            _print_progress(
                f"[4/6] Importance filter kept {len(kept)}/{pre} examples",
                verbose=v,
            )
            logger.info(
                "After importance filter: %d kept (min_score=%.2f)",
                len(kept),
                cfg.min_importance_score,
            )
        else:
            _print_progress("[4/6] Skipping importance filter (not configured)", verbose=v)

        # 5. Cap ------------------------------------------------------------
        pre_cap = len(kept)
        if self.max_examples and len(kept) > self.max_examples:
            import random

            rng = random.Random(self.random_seed)
            kept = rng.sample(kept, self.max_examples)
            _print_progress(
                f"[5/6] Capped to max_examples={self.max_examples} "
                f"(sampled from {pre_cap})",
                verbose=v,
            )
        else:
            _print_progress(
                f"[5/6] No cap needed ({pre_cap} ≤ max_examples={self.max_examples})",
                verbose=v,
            )

        # 6. Split ----------------------------------------------------------
        if len(kept) < MIN_TRAIN_SAMPLES:
            raise ValueError(
                f"Only {len(kept)} examples survived filtering; need at least "
                f"{MIN_TRAIN_SAMPLES}.  Loosen filters, or lower thresholds, "
                f"or fetch more traces."
            )
        train_count = max(int(round(len(kept) * self.train_fraction)), MIN_TRAIN_SAMPLES)
        eval_count = len(kept) - train_count
        train_data, eval_data = split_dataset(
            kept, train_count, eval_count, seed=self.random_seed
        )
        _print_progress(
            f"[6/6] Split complete: {len(train_data)} train, {len(eval_data)} eval",
            verbose=v,
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
            "importance_filter": importance_stats,
        }

        if self.output_dir:
            out = Path(self.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            _write_jsonl(out / "train.jsonl", train_data)
            _write_jsonl(out / "eval.jsonl", eval_data)
            with (out / "metadata.json").open("w") as f:
                json.dump(stats, f, indent=2, default=str)
            _print_progress(
                f"Wrote {len(train_data)} train + {len(eval_data)} eval to {out}",
                verbose=v,
            )
            logger.info(
                "Wrote %d train + %d eval examples to %s",
                len(train_data),
                len(eval_data),
                out,
            )

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
