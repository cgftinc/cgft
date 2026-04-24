"""Tests for the QA quality eval harness."""

import json
import tempfile
from pathlib import Path

from cgft.qa_generation.evals import (
    AnswerDiversityMetric,
    AnswerLengthMetric,
    ChunkConcentrationMetric,
    DiversityMetric,
    HopCountDistributionMetric,
    LexicalShortcutMetric,
    NaturalnessMetric,
    NearDuplicateMetric,
    QualityReport,
    ReferenceCoherenceMetric,
    run_eval_report,
)

_CHUNK_GEOGRAPHY = (
    "The capital of France is Paris, a major European city known for the Eiffel Tower."
)
_ANSWER_GEOGRAPHY = (
    "The capital of France is Paris, a major European city known for the Eiffel Tower"
)
_ANSWER_REDIS = "Redis uses TTL-based expiration with a 5-minute default in the caching layer."
_ANSWER_AUTH = "V1 used session-based auth while v2 migrated to JWT tokens for stateless operation."
_CHUNK_AUTH_V1 = "Version 1 of the API uses session-based authentication with server-side storage."


def _make_items() -> list[dict]:
    return [
        {
            "question": "What is the capital of France?",
            "answer": _ANSWER_GEOGRAPHY,
            "qa_type": "lookup",
            "min_hop_count": 1,
            "reference_chunks": [
                {
                    "id": "chunk_1",
                    "metadata": {"file": "geography.md"},
                    "content": _CHUNK_GEOGRAPHY,
                }
            ],
        },
        {
            "question": "How do Redis TTL and caching interact?",
            "answer": _ANSWER_REDIS,
            "qa_type": "multi_hop",
            "min_hop_count": 2,
            "reference_chunks": [
                {
                    "id": "chunk_2",
                    "metadata": {"file": "redis.md"},
                    "content": "Redis supports TTL-based key expiration.",
                },
                {
                    "id": "chunk_3",
                    "metadata": {"file": "caching.md"},
                    "content": "The caching layer uses Redis with 5m TTL.",
                },
            ],
        },
        {
            "question": "Compare auth methods in v1 and v2.",
            "answer": _ANSWER_AUTH,
            "qa_type": "multi_hop",
            "min_hop_count": 2,
            "reference_chunks": [
                {
                    "id": "chunk_4",
                    "metadata": {"file": "auth_v1.md"},
                    "content": _CHUNK_AUTH_V1,
                },
                {
                    "id": "chunk_5",
                    "metadata": {"file": "auth_v2.md"},
                    "content": "V2 migrated to JWT for stateless auth.",
                },
            ],
        },
    ]


class TestDiversityMetric:
    def test_diverse_questions(self):
        items = _make_items()
        result = DiversityMetric().compute(items)
        assert result.name == "diversity"
        assert 0.0 <= result.value <= 1.0
        assert result.details["num_items"] == 3

    def test_identical_questions(self):
        item = _make_items()[0]
        items = [item, item.copy(), item.copy()]
        result = DiversityMetric().compute(items)
        assert result.value <= 0.5  # Low diversity for identical questions

    def test_empty_items(self):
        result = DiversityMetric().compute([])
        assert result.value == 0.0

    def test_has_entropy(self):
        items = _make_items()
        result = DiversityMetric().compute(items)
        assert result.details["qa_type_entropy"] > 0


class TestLexicalShortcutMetric:
    def test_detects_verbatim_answer(self):
        items = _make_items()
        result = LexicalShortcutMetric().compute(items)
        assert result.name == "lexical_shortcut"
        # First item has answer verbatim in chunk
        assert result.details["shortcut_count"] >= 1

    def test_no_shortcuts(self):
        items = [
            {
                "question": "What is X?",
                "answer": "X is a completely unique answer not in any chunk whatsoever.",
                "qa_type": "lookup",
                "reference_chunks": [
                    {"id": "c1", "metadata": {}, "content": "Some unrelated content here."}
                ],
            }
        ]
        result = LexicalShortcutMetric().compute(items)
        assert result.details["shortcut_count"] == 0

    def test_empty_items(self):
        result = LexicalShortcutMetric().compute([])
        assert result.value == 0.0

    def test_short_answers_skipped(self):
        items = [
            {
                "question": "What?",
                "answer": "Yes",
                "qa_type": "lookup",
                "reference_chunks": [{"id": "c1", "metadata": {}, "content": "Yes this is true."}],
            }
        ]
        result = LexicalShortcutMetric(min_answer_tokens=3).compute(items)
        assert result.details["shortcut_count"] == 0


class TestHopCountDistributionMetric:
    def test_distribution(self):
        items = _make_items()
        result = HopCountDistributionMetric().compute(items)
        assert result.name == "hop_count_distribution"
        assert result.details["total_items"] == 3
        dist = result.details["hop_count_distribution"]
        assert dist.get("1", 0) == 1  # lookup
        assert dist.get("2", 0) == 2  # multi_hop

    def test_empty_items(self):
        result = HopCountDistributionMetric().compute([])
        assert result.value == 0.0


class TestAnswerLengthMetric:
    def test_lengths(self):
        items = _make_items()
        result = AnswerLengthMetric().compute(items)
        assert result.name == "answer_length"
        assert result.value > 0
        assert result.details["min_chars"] > 0
        assert result.details["max_chars"] >= result.details["min_chars"]

    def test_empty_items(self):
        result = AnswerLengthMetric().compute([])
        assert result.value == 0.0


class TestNaturalnessMetric:
    def test_clean_keyword_passes(self):
        items = [
            {
                "question": "PostHog reverse proxy setup Remix",
                "style_observed": "keyword",
                "answer": "Configure a reverse proxy.",
                "qa_type": "lookup",
                "reference_chunks": [],
            }
        ]
        result = NaturalnessMetric().compute(items)
        assert result.name == "naturalness"
        assert result.details["degenerate_count"] == 0

    def test_degenerate_keyword_flagged(self):
        items = [
            {
                "question": (
                    "PostHog experiments traffic split change variant reassignment analysis bias"
                ),
                "style_observed": "keyword",
                "answer": "Adjusting the traffic split...",
                "qa_type": "multi_hop",
                "reference_chunks": [],
            }
        ]
        result = NaturalnessMetric().compute(items)
        assert result.details["degenerate_count"] >= 1
        assert result.value > 0

    def test_natural_not_evaluated(self):
        items = [
            {
                "question": "How do I configure PostHog with React?",
                "style_observed": "natural",
                "answer": "Install the React SDK.",
                "qa_type": "lookup",
                "reference_chunks": [],
            }
        ]
        result = NaturalnessMetric().compute(items)
        assert result.details["keyword_count"] == 0
        assert result.value == 0.0

    def test_short_keyword_is_fine(self):
        items = [
            {
                "question": "redis TTL config",
                "style_observed": "keyword",
                "answer": "Set TTL via config.",
                "qa_type": "lookup",
                "reference_chunks": [],
            }
        ]
        result = NaturalnessMetric().compute(items)
        assert result.details["degenerate_count"] == 0

    def test_heuristic_fallback(self):
        """Without style_observed, heuristic detects keyword-style."""
        items = [
            {
                "question": (
                    "PostHog SDK Unity events missing OnApplicationQuit Shutdown ingestion pipeline"
                ),
                "answer": "Check debug logs.",
                "qa_type": "multi_hop",
                "reference_chunks": [],
            }
        ]
        result = NaturalnessMetric().compute(items)
        assert result.details["keyword_count"] >= 1

    def test_empty_items(self):
        result = NaturalnessMetric().compute([])
        assert result.value == 0.0


class TestAnswerDiversityMetric:
    def test_diverse_answers(self):
        items = _make_items()
        result = AnswerDiversityMetric().compute(items)
        assert result.name == "answer_diversity"
        assert 0.0 <= result.value <= 1.0

    def test_similar_answers_detected(self):
        """Answers that are near-identical but not exact should show high similarity."""
        answer_base = "Check the PostHog app to confirm events appear in your activity feed."
        answer_variant = (
            "Check the PostHog app to confirm events appear"
            " in your activity feed. If errors, troubleshoot."
        )
        answer_different = (
            "Redis uses TTL-based expiration with a configurable timeout for caching."
        )
        items = [
            {
                "question": "Q1",
                "answer": answer_base,
                "qa_type": "lookup",
                "reference_chunks": [],
            },
            {
                "question": "Q2",
                "answer": answer_variant,
                "qa_type": "lookup",
                "reference_chunks": [],
            },
            {
                "question": "Q3",
                "answer": answer_different,
                "qa_type": "multi_hop",
                "reference_chunks": [],
            },
        ]
        result = AnswerDiversityMetric(similarity_threshold=0.5).compute(items)
        assert result.details["high_similarity_pairs"] >= 1

    def test_empty_items(self):
        result = AnswerDiversityMetric().compute([])
        assert result.value == 0.0


class TestReferenceCoherenceMetric:
    def test_coherent_multi_hop(self):
        items = _make_items()
        result = ReferenceCoherenceMetric().compute(items)
        assert result.name == "reference_coherence"
        assert result.value > 0

    def test_incoherent_chunks_flagged(self):
        items = [
            {
                "question": "Compare Redis and Shakespeare?",
                "answer": "They are unrelated.",
                "qa_type": "multi_hop",
                "min_hop_count": 2,
                "reference_chunks": [
                    {
                        "id": "c1",
                        "metadata": {},
                        "content": "Redis is an in-memory data store for caching.",
                    },
                    {
                        "id": "c2",
                        "metadata": {},
                        "content": "Shakespeare wrote Hamlet in the early 1600s.",
                    },
                ],
            }
        ]
        result = ReferenceCoherenceMetric(incoherence_threshold=0.1).compute(items)
        assert result.details["incoherent_count"] >= 1

    def test_single_hop_skipped(self):
        items = [_make_items()[0]]  # single-chunk lookup
        result = ReferenceCoherenceMetric().compute(items)
        assert result.details.get("note") == "no multi-hop items"
        assert result.value == 1.0

    def test_empty_items(self):
        result = ReferenceCoherenceMetric().compute([])
        assert result.value == 0.0


class TestChunkConcentrationMetric:
    def test_balanced_usage(self):
        items = _make_items()
        result = ChunkConcentrationMetric().compute(items)
        assert result.name == "chunk_concentration"
        assert 0.0 <= result.value <= 1.0
        assert result.details["unique_chunks"] == 5

    def test_concentrated_usage(self):
        """One chunk used many times alongside unique chunks -> high Gini."""
        dominant = {"id": "dominant", "metadata": {}, "content": "Common."}

        def _unique(i: int) -> dict:
            return {"id": f"u{i}", "metadata": {}, "content": f"Unique {i}."}

        items = [
            {
                "question": f"Q{i}",
                "answer": "A",
                "qa_type": "lookup",
                "reference_chunks": [dominant, _unique(i)],
            }
            for i in range(10)
        ]
        result = ChunkConcentrationMetric().compute(items)
        assert result.value > 0.3  # Gini should be elevated

    def test_empty_items(self):
        result = ChunkConcentrationMetric().compute([])
        assert result.value == 0.0


class TestNearDuplicateMetric:
    def test_no_duplicates(self):
        items = _make_items()
        result = NearDuplicateMetric().compute(items)
        assert result.name == "near_duplicate"
        assert result.details["duplicate_pair_count"] == 0

    def test_detects_near_duplicates(self):
        """Near-duplicate questions should be flagged."""
        q_base = (
            "After setting up the reverse proxy for PostHog, how do I verify events are captured?"
        )
        q_variant = (
            "After setting up the reverse proxy for PostHog,"
            " how do I verify events are being captured?"
        )
        items = [
            {
                "question": q_base,
                "answer": "Check the activity feed.",
                "qa_type": "lookup",
                "reference_chunks": [],
            },
            {
                "question": q_variant,
                "answer": "Check the activity feed.",
                "qa_type": "lookup",
                "reference_chunks": [],
            },
            {
                "question": "What auth method does v2 use?",
                "answer": "JWT.",
                "qa_type": "lookup",
                "reference_chunks": [],
            },
        ]
        result = NearDuplicateMetric(similarity_threshold=0.4).compute(items)
        assert result.details["duplicate_pair_count"] >= 1
        assert result.value > 0

    def test_empty_items(self):
        result = NearDuplicateMetric().compute([])
        assert result.value == 0.0


class TestQualityReport:
    def test_default_report(self):
        items = _make_items()
        report = QualityReport.default()
        results = report.run(items)
        assert len(results) == 9
        names = {r.name for r in results}
        assert "diversity" in names
        assert "answer_diversity" in names
        assert "lexical_shortcut" in names
        assert "naturalness" in names
        assert "hop_count_distribution" in names
        assert "answer_length" in names
        assert "reference_coherence" in names
        assert "chunk_concentration" in names
        assert "near_duplicate" in names

    def test_format(self):
        items = _make_items()
        report = QualityReport.default()
        text = report.run_and_format(items)
        assert "QA Quality Report" in text
        assert "diversity" in text


class TestRunEvalReport:
    def test_from_jsonl(self):
        items = _make_items()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
            path = f.name

        results = run_eval_report(path, print_report=False)
        assert len(results) == 9
        Path(path).unlink()

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        results = run_eval_report(path, print_report=False)
        assert results == []
        Path(path).unlink()
