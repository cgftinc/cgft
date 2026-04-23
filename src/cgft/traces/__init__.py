"""Trace processing — normalised traces → training examples.

Top-level surface:
    ``TracesPipeline`` — the ergonomic entrypoint for users.
    ``ImportanceFilterConfig`` — optional LLM-importance filter config.
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
from cgft.traces.pipeline import ImportanceFilterConfig, TracesPipeline

__all__ = [
    "ImportanceFilterConfig",
    "NormalizedTrace",
    "ToolCall",
    "TraceCredentials",
    "TraceMessage",
    "TracesPipeline",
]
