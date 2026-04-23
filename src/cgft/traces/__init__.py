"""Trace processing — normalised traces → training examples.

Top-level surface:
    ``TracesPipeline`` — the ergonomic entrypoint for users.
    ``PivotConfig`` — optional LLM-importance filter config.
    ``NormalizedTrace`` / ``TraceMessage`` — the data model.

Lower-level primitives live in ``cgft.traces.processing`` / ``cgft.traces.pivot``
for power users who need custom filter ordering or to compose their own
pipelines.
"""

from cgft.traces.adapter import (
    NormalizedTrace,
    TraceCredentials,
    TraceMessage,
    ToolCall,
)
from cgft.traces.pipeline import PivotConfig, TracesPipeline

__all__ = [
    "NormalizedTrace",
    "PivotConfig",
    "ToolCall",
    "TraceCredentials",
    "TraceMessage",
    "TracesPipeline",
]
