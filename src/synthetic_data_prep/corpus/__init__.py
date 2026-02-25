"""Corpus module for document chunking and corpus management."""

from .source import ChunkSource
from .corpora import CorporaChunkSource

__all__ = [
    "ChunkSource",
    "CorporaChunkSource",
]
