"""Tests for shared reward helpers."""

from __future__ import annotations

from cgft.envs.reward_helpers import (
    citation_score,
    extract_answer_block,
    extract_completion_text,
    overlap_reward,
    tool_call_efficiency,
)


class TestExtractCompletionText:
    def test_string(self):
        assert extract_completion_text("hello") == "hello"

    def test_message_list(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "answer one"},
            {"role": "assistant", "content": "answer two"},
        ]
        result = extract_completion_text(msgs)
        assert "answer one" in result
        assert "answer two" in result

    def test_empty_list(self):
        assert extract_completion_text([]) == ""

    def test_non_list_non_string(self):
        assert extract_completion_text(42) == ""  # type: ignore[arg-type]


class TestExtractAnswerBlock:
    def test_with_tags(self):
        assert extract_answer_block("before <answer>the answer</answer> after") == "the answer"

    def test_without_tags(self):
        assert extract_answer_block("no tags here") == "no tags here"

    def test_empty(self):
        assert extract_answer_block("") == ""


class TestOverlapReward:
    def test_matching_text(self):
        score = overlap_reward("the reference text is here", "unused", reference_chunks=[
            {"content": "the reference text is here"}
        ])
        assert score > 0.25

    def test_no_match(self):
        score = overlap_reward("completely different", "unused", reference_chunks=[
            {"content": "nothing in common xyz abc"}
        ])
        assert score == 0.0

    def test_below_threshold(self):
        score = overlap_reward("a tiny bit matches", "unused", reference_chunks=[
            {"content": "a" * 1000}
        ])
        assert score == 0.0

    def test_falls_back_to_ground_truth(self):
        score = overlap_reward("the ground truth text", "the ground truth text")
        assert score > 0.25

    def test_tool_call_penalty(self):
        msgs = [
            {"role": "assistant", "content": "<tool_call>" * 4 + "text"},
        ]
        score = overlap_reward(msgs, "", reference_chunks=[{"content": "text"}])
        assert score == 0.0

    def test_empty_reference(self):
        assert overlap_reward("text", "", reference_chunks=[]) == 0.0


class TestCitationScore:
    def test_perfect_match(self):
        completion = "According to [Source: doc1] and [Source: doc2], the answer is yes."
        chunks = [
            {"metadata": {"source_id": "doc1"}},
            {"metadata": {"source_id": "doc2"}},
        ]
        result = citation_score(completion, chunks)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_partial_recall(self):
        completion = "According to [Source: doc1], the answer is yes."
        chunks = [
            {"metadata": {"source_id": "doc1"}},
            {"metadata": {"source_id": "doc2"}},
        ]
        result = citation_score(completion, chunks)
        assert result["precision"] == 1.0
        assert result["recall"] == 0.5

    def test_partial_precision(self):
        completion = "According to [Source: doc1] and [Source: doc99], yes."
        chunks = [{"metadata": {"source_id": "doc1"}}]
        result = citation_score(completion, chunks)
        assert result["precision"] == 0.5
        assert result["recall"] == 1.0

    def test_no_citations(self):
        result = citation_score("No citations here.", [{"metadata": {"source_id": "doc1"}}])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_no_reference_chunks(self):
        result = citation_score("See [Source: doc1].", [])
        assert result["precision"] == 1.0
        assert result["recall"] == 0.0

    def test_empty_both(self):
        result = citation_score("No citations.", [])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_custom_source_field(self):
        completion = "See [Source: abc]."
        chunks = [{"metadata": {"doc_id": "abc"}}]
        result = citation_score(completion, chunks, source_field="doc_id")
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_message_list_input(self):
        msgs = [{"role": "assistant", "content": "See [Source: doc1]."}]
        chunks = [{"metadata": {"source_id": "doc1"}}]
        result = citation_score(msgs, chunks)
        assert result["precision"] == 1.0

    def test_missing_metadata(self):
        completion = "See [Source: doc1]."
        chunks = [{"content": "no metadata key"}]
        result = citation_score(completion, chunks)
        assert result["precision"] == 1.0
        assert result["recall"] == 0.0

    def test_case_insensitive(self):
        # Model emits lowercase [source: ...] (mirroring search tool output);
        # regex must still match so citations count.
        completion = "See [source: doc1] and [SOURCE: doc2]."
        chunks = [
            {"metadata": {"source_id": "doc1"}},
            {"metadata": {"source_id": "doc2"}},
        ]
        result = citation_score(completion, chunks)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_source_field_list(self):
        # Heterogeneous chunks: some expose `file`, others `file_path`.
        # First-non-empty-key-wins semantics — chunks without either are ignored.
        completion = "See [Source: a.md] and [Source: b.md]."
        chunks = [
            {"metadata": {"file": "a.md"}},
            {"metadata": {"file_path": "b.md"}},
            {"metadata": {"other": "c.md"}},
        ]
        result = citation_score(
            completion, chunks, source_field=["file", "file_path"]
        )
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_canonicalize_hook(self):
        # Subclass-style canonicalizer (e.g. strip extension + lowercase)
        # applied symmetrically to cited + reference IDs before intersection.
        completion = "See [Source: Doc1.MD]."
        chunks = [{"metadata": {"source_id": "doc1.md"}}]
        result = citation_score(
            completion,
            chunks,
            canonicalize=lambda s: s.strip().lower().removesuffix(".md"),
        )
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0


class TestToolCallEfficiency:
    def test_zero_calls(self):
        assert tool_call_efficiency("no tool calls here") == 0.0

    def test_default_ranges_low(self):
        completion = "<tool_call>" * 2
        assert tool_call_efficiency(completion) == 1.0

    def test_default_ranges_mid(self):
        completion = "<tool_call>" * 5
        assert tool_call_efficiency(completion) == 0.5

    def test_default_ranges_high(self):
        completion = "<tool_call>" * 10
        assert tool_call_efficiency(completion) == 0.0

    def test_boundary_1(self):
        assert tool_call_efficiency("<tool_call>" * 1) == 1.0

    def test_boundary_3(self):
        assert tool_call_efficiency("<tool_call>" * 3) == 1.0

    def test_boundary_4(self):
        assert tool_call_efficiency("<tool_call>" * 4) == 0.5

    def test_boundary_6(self):
        assert tool_call_efficiency("<tool_call>" * 6) == 0.5

    def test_boundary_7(self):
        assert tool_call_efficiency("<tool_call>" * 7) == 0.0

    def test_custom_ranges(self):
        ranges = [(1, 2, 1.0), (3, 5, 0.3)]
        assert tool_call_efficiency("<tool_call>" * 2, ranges=ranges) == 1.0
        assert tool_call_efficiency("<tool_call>" * 4, ranges=ranges) == 0.3
        assert tool_call_efficiency("<tool_call>" * 6, ranges=ranges) == 0.0

    def test_message_list_input(self):
        msgs = [{"role": "assistant", "content": "<tool_call>" * 2}]
        assert tool_call_efficiency(msgs) == 1.0
