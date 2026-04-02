"""Tests for the CorpusProfile module."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from cgft.qa_generation.corpus_profile import (
    CorpusProfile,
    EntityPattern,
    build_entity_patterns_from_extraction,
    compute_entity_document_frequency,
    detect_search_capabilities,
    diverse_profile_sample,
)


@dataclass(frozen=True)
class FakeChunk:
    content: str
    metadata: tuple[tuple[str, Any], ...] = ()

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata)


def _make_chunks() -> list[FakeChunk]:
    return [
        FakeChunk(
            content="Redis supports TTL-based key expiration.",
            metadata=(("file", "redis.md"),),
        ),
        FakeChunk(
            content="Redis is used in the caching layer with 5m TTL.",
            metadata=(("file", "redis.md"),),
        ),
        FakeChunk(
            content="The API uses JWT tokens for auth.",
            metadata=(("file", "auth.md"),),
        ),
        FakeChunk(
            content="Deploy using Kubernetes on Azure AKS.",
            metadata=(("file", "deploy.md"),),
        ),
        FakeChunk(
            content="PostHog analytics tracks user events.",
            metadata=(("file", "analytics.md"),),
        ),
    ]


class TestEntityPattern:
    def test_basic(self):
        p = EntityPattern(name="Redis", type="entity")
        assert p.name == "Redis"
        assert p.document_frequency == 0.0
        assert p.idf_weight == 0.0


class TestComputeEntityDocumentFrequency:
    def test_computes_df(self):
        chunks = _make_chunks()
        patterns = [
            EntityPattern(name="Redis", type="entity"),
            EntityPattern(name="JWT", type="entity"),
            EntityPattern(name="Kubernetes", type="entity"),
        ]
        compute_entity_document_frequency(patterns, chunks)

        # Redis appears in 2/5 chunks
        assert patterns[0].document_frequency == 0.4
        assert patterns[0].idf_weight > 0

        # JWT appears in 1/5
        assert patterns[1].document_frequency == 0.2

        # Kubernetes in 1/5
        assert patterns[2].document_frequency == 0.2

    def test_ubiquitous_entity(self):
        chunks = [FakeChunk(content=f"PostHog docs chunk {i}") for i in range(10)]
        patterns = [EntityPattern(name="PostHog", type="entity")]
        compute_entity_document_frequency(patterns, chunks)
        assert patterns[0].document_frequency == 1.0

    def test_empty_chunks(self):
        patterns = [EntityPattern(name="Redis", type="entity")]
        compute_entity_document_frequency(patterns, [])
        assert patterns[0].document_frequency == 0.0


class TestCorpusProfile:
    def test_get_discriminative_entities(self):
        profile = CorpusProfile(
            entity_patterns=[
                EntityPattern(name="PostHog", type="entity", document_frequency=0.95),
                EntityPattern(name="Redis", type="entity", document_frequency=0.3),
                EntityPattern(name="TTL", type="domain_term", document_frequency=0.2),
            ],
            ubiquitous_df_threshold=0.80,
        )
        disc = profile.get_discriminative_entities()
        assert len(disc) == 2
        assert all(e.name != "PostHog" for e in disc)

    def test_get_entity_names(self):
        profile = CorpusProfile(
            entity_patterns=[
                EntityPattern(name="PostHog", type="entity", document_frequency=0.95),
                EntityPattern(name="Redis", type="entity", document_frequency=0.3),
            ],
            ubiquitous_df_threshold=0.80,
        )
        names = profile.get_entity_names(discriminative_only=True)
        assert names == ["Redis"]

        all_names = profile.get_entity_names(discriminative_only=False)
        assert len(all_names) == 2

    def test_get_domain_terms(self):
        profile = CorpusProfile(
            entity_patterns=[
                EntityPattern(name="Redis", type="entity", document_frequency=0.3),
                EntityPattern(name="TTL", type="domain_term", document_frequency=0.2),
                EntityPattern(name="API", type="domain_term", document_frequency=0.9),
            ],
            ubiquitous_df_threshold=0.80,
        )
        terms = profile.get_domain_terms(discriminative_only=True)
        assert terms == ["TTL"]

    def test_corpus_queries_bulleted(self):
        profile = CorpusProfile(corpus_queries=["how to X", "what is Y"])
        assert profile.corpus_queries_bulleted == "- how to X\n- what is Y"

    def test_enrichment_methods(self):
        profile = CorpusProfile()

        profile.add_discovered_entities(["Redis", "TTL"])
        assert profile.discovered_entities == ["Redis", "TTL"]

        # No duplicates
        profile.add_discovered_entities(["Redis", "JWT"])
        assert profile.discovered_entities == ["Redis", "TTL", "JWT"]

        profile.record_query_effectiveness("Redis setup", 5)
        assert "Redis setup" in profile.effective_query_templates

        profile.record_query_effectiveness("empty query", 0)
        assert "empty query" not in profile.effective_query_templates

        profile.update_connection_stats("cross_document")
        profile.update_connection_stats("cross_document")
        profile.update_connection_stats("co_located")
        assert profile.connection_distribution == {
            "cross_document": 2,
            "co_located": 1,
        }


class TestBuildEntityPatternsFromExtraction:
    def test_builds_patterns(self):
        patterns = build_entity_patterns_from_extraction(
            entity_names=["Redis", "PostHog"],
            code_patterns={"func": r"def \w+\("},
            domain_terms=["TTL", "cache"],
        )
        assert len(patterns) == 5
        types = [p.type for p in patterns]
        assert types.count("entity") == 2
        assert types.count("code_pattern") == 1
        assert types.count("domain_term") == 2

    def test_skips_empty(self):
        patterns = build_entity_patterns_from_extraction(
            entity_names=["", "Redis", "  "],
            code_patterns={},
            domain_terms=[""],
        )
        assert len(patterns) == 1
        assert patterns[0].name == "Redis"


class TestDiverseProfileSample:
    def test_stratified_sampling(self):
        source = MagicMock()
        chunks = _make_chunks()
        source.sample_chunks.return_value = chunks

        result = diverse_profile_sample(source, sample_size=4, min_chars=0, rng=random.Random(42))
        assert len(result) == 4

        # Should have chunks from multiple files
        files = {dict(c.metadata).get("file") for c in result}
        assert len(files) >= 2

    def test_empty_source(self):
        source = MagicMock()
        source.sample_chunks.return_value = []
        result = diverse_profile_sample(source, sample_size=10, min_chars=0, rng=random.Random(42))
        assert result == []

    def test_fewer_chunks_than_requested(self):
        source = MagicMock()
        chunks = _make_chunks()[:2]
        source.sample_chunks.return_value = chunks
        result = diverse_profile_sample(source, sample_size=10, min_chars=0, rng=random.Random(42))
        assert len(result) == 2


class TestDetectSearchCapabilities:
    def test_hybrid_mode(self):
        source = MagicMock()
        source.get_search_capabilities.return_value = {
            "modes": {"lexical", "vector", "hybrid"},
        }
        modes, best = detect_search_capabilities(source)
        assert "hybrid" in modes
        assert best == "hybrid"

    def test_vector_mode(self):
        source = MagicMock()
        source.get_search_capabilities.return_value = {
            "modes": {"lexical", "vector"},
        }
        modes, best = detect_search_capabilities(source)
        assert best == "vector"

    def test_lexical_only(self):
        source = MagicMock()
        source.get_search_capabilities.return_value = {
            "modes": {"lexical"},
        }
        modes, best = detect_search_capabilities(source)
        assert best == "lexical"

    def test_fallback_on_error(self):
        source = MagicMock()
        source.get_search_capabilities.side_effect = AttributeError
        modes, best = detect_search_capabilities(source)
        assert modes == {"lexical"}
        assert best == "lexical"
