"""QA quality report runner.

Computes all applicable metrics over a JSONL dataset and produces
a structured report.  Can be used standalone (``run_eval_report``)
or as a post-pipeline step.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cgft.qa_generation.evals.metrics import (
    AnswerDiversityMetric,
    AnswerLengthMetric,
    ChunkConcentrationMetric,
    DiversityMetric,
    HopCountDistributionMetric,
    LexicalShortcutMetric,
    Metric,
    MetricResult,
    NaturalnessMetric,
    NearDuplicateMetric,
    ReferenceCoherenceMetric,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Runs a set of metrics and collects results."""

    metrics: list[Metric] = field(default_factory=list)

    @classmethod
    def default(cls) -> QualityReport:
        """Create a report with all JSONL-only metrics (no LLM/corpus deps)."""
        return cls(
            metrics=[
                DiversityMetric(),
                AnswerDiversityMetric(),
                LexicalShortcutMetric(),
                NaturalnessMetric(),
                HopCountDistributionMetric(),
                AnswerLengthMetric(),
                ReferenceCoherenceMetric(),
                ChunkConcentrationMetric(),
                NearDuplicateMetric(),
            ]
        )

    def run(self, items: list[dict[str, Any]]) -> list[MetricResult]:
        """Compute all metrics and return results."""
        results: list[MetricResult] = []
        for metric in self.metrics:
            try:
                result = metric.compute(items)
                results.append(result)
            except Exception:
                logger.exception("Metric %s failed", getattr(metric, "name", metric))
                results.append(
                    MetricResult(
                        name=getattr(metric, "name", str(metric)),
                        value=0.0,
                        details={"error": "metric computation failed"},
                    )
                )
        return results

    def run_and_format(self, items: list[dict[str, Any]]) -> str:
        """Compute metrics and return a human-readable summary."""
        results = self.run(items)
        return format_report(results, len(items))


def format_report(results: list[MetricResult], n_items: int) -> str:
    """Format metric results as a readable text report."""
    lines = [
        "=" * 60,
        "QA Quality Report",
        f"Items evaluated: {n_items}",
        "=" * 60,
    ]

    for result in results:
        lines.append("")
        lines.append(f"--- {result.name} ---")
        lines.append(f"  Score: {result.value}")
        for key, val in result.details.items():
            if key == "examples":
                lines.append(f"  {key}: ({len(val)} shown)")
                for ex in val:
                    lines.append(f"    - {ex}")
            elif isinstance(val, dict):
                lines.append(f"  {key}:")
                for k, v in val.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {val}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load items from a JSONL file."""
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_eval_report(
    jsonl_path: str | Path,
    *,
    metrics: list[Metric] | None = None,
    print_report: bool = True,
) -> list[MetricResult]:
    """Standalone entry point: load a JSONL file and run quality metrics.

    Args:
        jsonl_path: Path to the JSONL dataset file.
        metrics: Optional list of metrics to run. Defaults to all JSONL-only metrics.
        print_report: Whether to print the formatted report to stdout.

    Returns:
        List of MetricResult objects.
    """
    items = load_jsonl(jsonl_path)
    if not items:
        logger.warning("No items loaded from %s", jsonl_path)
        return []

    report = QualityReport(metrics=metrics) if metrics else QualityReport.default()
    results = report.run(items)

    if print_report:
        formatted = format_report(results, len(items))
        print(formatted)  # noqa: T201

    return results
