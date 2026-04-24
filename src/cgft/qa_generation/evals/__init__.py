"""QA quality evaluation harness for measuring dataset fidelity."""

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
from cgft.qa_generation.evals.quality_report import QualityReport, run_eval_report

__all__ = [
    "AnswerDiversityMetric",
    "AnswerLengthMetric",
    "ChunkConcentrationMetric",
    "DiversityMetric",
    "HopCountDistributionMetric",
    "LexicalShortcutMetric",
    "Metric",
    "MetricResult",
    "NaturalnessMetric",
    "NearDuplicateMetric",
    "QualityReport",
    "ReferenceCoherenceMetric",
    "run_eval_report",
]
