"""Tests for mode-aware query generation in anchor_utils."""

from __future__ import annotations

from cgft.chunkers.models import Chunk
from cgft.qa_generation.anchor_utils import (
    best_search_mode,
    generate_search_queries,
    generate_vector_queries,
)


class _FakeSource:
    def __init__(self, modes: set[str]):
        self._modes = modes

    def get_search_capabilities(self):
        return {"modes": self._modes, "backend": "test"}


class TestBestSearchMode:
    def test_no_capabilities(self):
        assert best_search_mode(object()) == "lexical"

    def test_lexical_only(self):
        assert best_search_mode(_FakeSource({"lexical"})) == "lexical"

    def test_vector_only(self):
        assert best_search_mode(_FakeSource({"vector"})) == "vector"

    def test_hybrid_preferred(self):
        assert best_search_mode(_FakeSource({"lexical", "vector", "hybrid"})) == "hybrid"

    def test_none_source(self):
        assert best_search_mode(None) == "lexical"


class TestGenerateSearchQueries:
    def test_delegates_to_bm25_for_lexical(self):
        chunk = Chunk(content="Hello world. Some text here.", metadata=(("h1", "Title"),))
        source = _FakeSource({"lexical"})
        queries = generate_search_queries(chunk, 3, source=source)
        # Should use BM25 strategy (metadata headers first)
        assert any("Title" in q for q in queries)

    def test_delegates_to_vector_for_vector_only(self):
        chunk = Chunk(
            content="Kubernetes cluster management guide. Deploy pods easily.",
            metadata=(("h1", "K8s Guide"),),
        )
        source = _FakeSource({"vector"})
        queries = generate_search_queries(chunk, 3, source=source)
        assert len(queries) > 0

    def test_no_source_defaults_to_bm25(self):
        chunk = Chunk(content="Some content", metadata=(("h1", "Header"),))
        queries = generate_search_queries(chunk, 2)
        assert len(queries) > 0


class TestGenerateVectorQueries:
    def test_uses_headers(self):
        chunk = Chunk(
            content="Details about deployment.",
            metadata=(("h1", "Deployment"), ("h2", "Configuration")),
        )
        queries = generate_vector_queries(chunk, 3)
        assert any("Deployment" in q for q in queries)

    def test_uses_content_sentences(self):
        chunk = Chunk(
            content="Redis is used for caching. It improves performance significantly.",
            metadata=(),
        )
        queries = generate_vector_queries(chunk, 3)
        assert len(queries) > 0
        # Should contain natural language from content
        assert any("Redis" in q for q in queries)

    def test_respects_n(self):
        chunk = Chunk(content="Short.", metadata=())
        queries = generate_vector_queries(chunk, 1)
        assert len(queries) <= 1

    def test_zero_n(self):
        chunk = Chunk(content="Content", metadata=())
        assert generate_vector_queries(chunk, 0) == []
