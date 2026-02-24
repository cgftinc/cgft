"""Corpus module for document chunking and corpus management."""

from .corpora.client import CorpusClient
from .corpora.models import Corpus
from .corpora.pipeline import prepare_corpus

__all__ = [
    "CorpusClient",
    "Corpus",
    "prepare_corpus",
]
