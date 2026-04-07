"""Tests for the generic trace processing pipeline."""

import pytest

from cgft.traces.adapter import (
    NormalizedTrace,
    ToolCall,
    TraceMessage,
)
from cgft.traces.processing import (
    MIN_TRAIN_SAMPLES,
    VALID_IDENTIFIER_RE,
    FilterResult,
    TrainingExample,
    apply_heuristic_filters,
    apply_score_filter,
    build_training_examples,
    detect_system_prompt,
    detect_tools,
    split_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trace(
    messages: list[TraceMessage],
    trace_id: str = "t1",
    scores: dict | None = None,
    errors: list[str] | None = None,
) -> NormalizedTrace:
    return NormalizedTrace(
        id=trace_id,
        messages=messages,
        scores=scores or {},
        errors=errors,
    )


def _msg(role: str, content: str, **kwargs) -> TraceMessage:
    return TraceMessage(role=role, content=content, **kwargs)


# ---------------------------------------------------------------------------
# detect_system_prompt
# ---------------------------------------------------------------------------


class TestDetectSystemPrompt:
    def test_finds_most_common(self):
        traces = [
            _trace([_msg("system", "Be helpful"), _msg("user", "Q1")], trace_id="t1"),
            _trace([_msg("system", "Be helpful"), _msg("user", "Q2")], trace_id="t2"),
            _trace([_msg("system", "Be concise"), _msg("user", "Q3")], trace_id="t3"),
        ]
        result = detect_system_prompt(traces)
        assert result is not None
        assert result.prompt == "Be helpful"
        assert result.count == 2
        assert result.total_traces == 3
        assert result.variants == ["Be concise"]

    def test_returns_none_when_no_system(self):
        traces = [_trace([_msg("user", "Hello")])]
        assert detect_system_prompt(traces) is None

    def test_single_trace_no_variants(self):
        traces = [_trace([_msg("system", "Sys"), _msg("user", "Q")])]
        result = detect_system_prompt(traces)
        assert result is not None
        assert result.variants == []

    def test_scans_all_traces_not_just_first(self):
        traces = [
            _trace([_msg("user", "Q1")], trace_id="t1"),  # no system
            _trace([_msg("system", "Found it"), _msg("user", "Q2")], trace_id="t2"),
        ]
        result = detect_system_prompt(traces)
        assert result is not None
        assert result.prompt == "Found it"


# ---------------------------------------------------------------------------
# detect_tools
# ---------------------------------------------------------------------------


class TestDetectTools:
    def test_detects_valid_tools(self):
        traces = [
            _trace([
                _msg("user", "Q"),
                _msg(
                    "assistant",
                    "",
                    tool_calls=[
                        ToolCall(name="search", arguments='{"q": "test"}'),
                        ToolCall(name="search", arguments='{"q": "other"}'),
                    ],
                ),
                _msg(
                    "assistant",
                    "",
                    tool_calls=[ToolCall(name="lookup", arguments='{"id": "1"}')],
                ),
            ])
        ]
        result = detect_tools(traces)
        assert len(result.tools) == 2
        tool_names = {t.name for t in result.tools}
        assert tool_names == {"search", "lookup"}

        search_tool = next(t for t in result.tools if t.name == "search")
        assert search_tool.call_count == 2
        assert "q" in search_tool.param_keys

    def test_drops_invalid_tool_names(self):
        traces = [
            _trace([
                _msg("user", "Q"),
                _msg(
                    "assistant",
                    "",
                    tool_calls=[
                        ToolCall(name="valid_tool", arguments="{}"),
                        ToolCall(name="invalid name!", arguments="{}"),
                        ToolCall(name="import os\nos.system('bad')", arguments="{}"),
                    ],
                ),
            ])
        ]
        result = detect_tools(traces)
        assert len(result.tools) == 1
        assert result.tools[0].name == "valid_tool"

    def test_filters_invalid_argument_keys(self):
        traces = [
            _trace([
                _msg("user", "Q"),
                _msg(
                    "assistant",
                    "",
                    tool_calls=[
                        ToolCall(
                            name="search",
                            arguments='{"valid_key": 1, "bad key!": 2, "also-bad": 3}',
                        ),
                    ],
                ),
            ])
        ]
        result = detect_tools(traces)
        tool = result.tools[0]
        assert tool.param_keys == {"valid_key"}
        assert tool.sample_args == [{"valid_key": 1}]

    def test_empty_traces(self):
        result = detect_tools([])
        assert result.tools == []

    def test_limits_sample_args_to_3(self):
        traces = [
            _trace([
                _msg("user", "Q"),
                _msg(
                    "assistant", "",
                    tool_calls=[ToolCall(name="search", arguments=f'{{"q": "test{i}"}}')]
                ),
            ], trace_id=f"t{i}")
            for i in range(10)
        ]
        result = detect_tools(traces)
        search = result.tools[0]
        assert len(search.sample_args) <= 3


# ---------------------------------------------------------------------------
# build_training_examples
# ---------------------------------------------------------------------------


class TestBuildTrainingExamples:
    def test_simple_conversation(self):
        traces = [
            _trace([
                _msg("system", "Be helpful"),
                _msg("user", "What is 2+2?"),
                _msg("assistant", "4"),
            ])
        ]
        # Default: include_system_prompt=False (system prompt goes on BaseEnv)
        examples = build_training_examples(traces)
        assert len(examples) == 1
        ex = examples[0]
        assert ex.trace_id == "t1"
        assert ex.turn_index == 0
        assert len(ex.prompt_messages) == 1  # user only (system excluded by default)
        assert ex.prompt_messages[0].role == "user"
        assert len(ex.completion_messages) == 1  # assistant

    def test_includes_system_prompt_when_requested(self):
        traces = [
            _trace([
                _msg("system", "Be helpful"),
                _msg("user", "Q"),
                _msg("assistant", "A"),
            ])
        ]
        examples = build_training_examples(traces, include_system_prompt=True)
        assert len(examples) == 1
        assert len(examples[0].prompt_messages) == 2  # system + user
        assert examples[0].prompt_messages[0].role == "system"

    def test_multi_turn_cumulative_context(self):
        traces = [
            _trace([
                _msg("user", "Q1"),
                _msg("assistant", "A1"),
                _msg("user", "Q2"),
                _msg("assistant", "A2"),
            ])
        ]
        examples = build_training_examples(traces)
        assert len(examples) == 2
        # First turn: prompt=[user Q1]
        assert len(examples[0].prompt_messages) == 1
        # Second turn: prompt=[user Q1, assistant A1, user Q2]
        assert len(examples[1].prompt_messages) == 3

    def test_tool_calls_bundled_with_assistant(self):
        traces = [
            _trace([
                _msg("user", "Search for X"),
                _msg(
                    "assistant",
                    "",
                    tool_calls=[ToolCall(name="search", arguments='{"q": "X"}')],
                ),
                _msg("tool", "Found X", name="search", tool_call_id="tc1"),
                _msg("assistant", "Here is what I found about X."),
            ])
        ]
        examples = build_training_examples(traces)
        assert len(examples) == 2
        # First completion: assistant + tool response
        assert len(examples[0].completion_messages) == 2
        assert examples[0].completion_messages[0].role == "assistant"
        assert examples[0].completion_messages[1].role == "tool"

    def test_skips_traces_with_errors(self):
        traces = [
            _trace(
                [_msg("user", "Q"), _msg("assistant", "A")],
                trace_id="good",
            ),
            _trace(
                [_msg("user", "Q")],
                trace_id="bad",
                errors=["extraction failed"],
            ),
        ]
        examples = build_training_examples(traces)
        assert len(examples) == 1
        assert examples[0].trace_id == "good"

    def test_skips_empty_traces(self):
        traces = [_trace([], trace_id="empty")]
        examples = build_training_examples(traces)
        assert len(examples) == 0

    def test_no_example_when_no_prompt_before_assistant(self):
        # Edge case: assistant message is the first message
        traces = [_trace([_msg("assistant", "Unprompted response")])]
        examples = build_training_examples(traces)
        assert len(examples) == 0

    def test_inherits_scores_and_metadata(self):
        traces = [
            NormalizedTrace(
                id="t1",
                messages=[_msg("user", "Q"), _msg("assistant", "A")],
                scores={"quality": 0.9},
                metadata={"source": "test"},
            )
        ]
        examples = build_training_examples(traces)
        assert examples[0].scores == {"quality": 0.9}
        assert examples[0].metadata == {"source": "test"}


class TestTrainingExampleSerialization:
    def test_to_jsonl_dict(self):
        ex = TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", "A")],
            prompt="[USER] Q",
            ground_truth="[ASSISTANT] A",
            trace_id="t1",
            turn_index=0,
            scores={"quality": 0.9},
        )
        d = ex.to_jsonl_dict()
        # prompt is structured message dicts
        assert isinstance(d["prompt"], list)
        assert d["prompt"][0]["role"] == "user"
        # ground_truth is the completion message dict
        assert isinstance(d["ground_truth"], dict)
        assert d["ground_truth"]["role"] == "assistant"
        # metadata goes in init_rollout_args
        assert d["init_rollout_args"]["trace_id"] == "t1"
        assert d["init_rollout_args"]["turn_index"] == 0
        assert d["init_rollout_args"]["scores"] == {"quality": 0.9}
        assert d["init_rollout_args"]["raw_prompt"] == "[USER] Q"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestHeuristicFilters:
    def _make_example(self, ground_truth: str) -> TrainingExample:
        return TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", ground_truth)],
            prompt="[USER] Q",
            ground_truth=ground_truth,
            trace_id="t1",
            turn_index=0,
        )

    def test_drops_short_completions(self):
        examples = [
            self._make_example("OK"),
            self._make_example("A" * 100),
        ]
        result = apply_heuristic_filters(examples, min_completion_chars=50)
        assert len(result.kept) == 1
        assert len(result.dropped) == 1
        assert result.dropped[0][1] == "too_short"

    def test_auth_filter_off_by_default(self):
        examples = [self._make_example("Please authenticate with OAuth bearer token to proceed.")]
        result = apply_heuristic_filters(examples)
        assert len(result.kept) == 1  # not dropped

    def test_auth_filter_when_enabled(self):
        examples = [self._make_example("Please authenticate with OAuth bearer token to proceed.")]
        result = apply_heuristic_filters(examples, drop_auth_lookups=True)
        assert len(result.kept) == 0
        assert result.dropped[0][1] == "auth_lookup"

    def test_summary(self):
        examples = [
            self._make_example("X"),
            self._make_example("Y"),
            self._make_example("A" * 100),
        ]
        result = apply_heuristic_filters(examples, min_completion_chars=50)
        assert result.summary == {"too_short": 2}


class TestScoreFilter:
    def _make_scored_example(self, score: float) -> TrainingExample:
        return TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", "A")],
            prompt="Q",
            ground_truth="A",
            trace_id="t1",
            turn_index=0,
            scores={"quality": score},
        )

    def test_min_score(self):
        examples = [self._make_scored_example(s) for s in [0.2, 0.5, 0.8]]
        result = apply_score_filter(examples, "quality", min_score=0.5)
        assert len(result) == 2

    def test_max_score(self):
        examples = [self._make_scored_example(s) for s in [0.2, 0.5, 0.8]]
        result = apply_score_filter(examples, "quality", max_score=0.5)
        assert len(result) == 2

    def test_range(self):
        examples = [self._make_scored_example(s) for s in [0.2, 0.5, 0.8]]
        result = apply_score_filter(examples, "quality", min_score=0.3, max_score=0.6)
        assert len(result) == 1
        assert result[0].scores["quality"] == 0.5

    def test_boundary_min_score_inclusive(self):
        examples = [self._make_scored_example(0.5)]
        result = apply_score_filter(examples, "quality", min_score=0.5)
        assert len(result) == 1  # exactly min_score is kept

    def test_boundary_max_score_inclusive(self):
        examples = [self._make_scored_example(0.5)]
        result = apply_score_filter(examples, "quality", max_score=0.5)
        assert len(result) == 1  # exactly max_score is kept

    def test_missing_score_excluded(self):
        ex = TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", "A")],
            prompt="Q",
            ground_truth="A",
            trace_id="t1",
            turn_index=0,
            scores={},  # no quality score
        )
        result = apply_score_filter([ex], "quality", min_score=0.0)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


class TestSplitDataset:
    def _make_examples(self, n: int) -> list[TrainingExample]:
        return [
            TrainingExample(
                prompt_messages=[_msg("user", f"Q{i}")],
                completion_messages=[_msg("assistant", f"A{i}")],
                prompt=f"Q{i}",
                ground_truth=f"A{i}",
                trace_id=f"t{i}",
                turn_index=0,
            )
            for i in range(n)
        ]

    def test_basic_split(self):
        examples = self._make_examples(20)
        train, eval_ = split_dataset(examples, train_count=16, eval_count=4)
        assert len(train) == 16
        assert len(eval_) == 4

    def test_shuffle_is_deterministic(self):
        examples = self._make_examples(20)
        train1, eval1 = split_dataset(examples, train_count=16, eval_count=4, seed=42)
        train2, eval2 = split_dataset(examples, train_count=16, eval_count=4, seed=42)
        assert train1 == train2
        assert eval1 == eval2

    def test_shuffle_mixes_order(self):
        examples = self._make_examples(20)
        train, _ = split_dataset(examples, train_count=16, eval_count=4)
        # With shuffle, first train example should NOT always be t0
        trace_ids = [row["init_rollout_args"]["trace_id"] for row in train]
        assert trace_ids != [f"t{i}" for i in range(16)]  # not in original order

    def test_min_train_samples_enforced(self):
        examples = self._make_examples(20)
        with pytest.raises(ValueError, match="at least 16"):
            split_dataset(examples, train_count=10, eval_count=5)

    def test_not_enough_examples(self):
        examples = self._make_examples(10)
        with pytest.raises(ValueError, match="Not enough"):
            split_dataset(examples, train_count=16, eval_count=4)

    def test_jsonl_dict_schema(self):
        examples = self._make_examples(20)
        train, _ = split_dataset(examples, train_count=16, eval_count=4)
        row = train[0]
        assert "prompt" in row
        assert isinstance(row["prompt"], list)  # structured message dicts
        assert "ground_truth" in row
        assert isinstance(row["ground_truth"], dict)  # completion message dict
        assert "init_rollout_args" in row
        assert "trace_id" in row["init_rollout_args"]
        assert "turn_index" in row["init_rollout_args"]


class TestValidIdentifierRegex:
    def test_valid_names(self):
        for name in ["search", "lookup_tool", "_private", "Tool1"]:
            assert VALID_IDENTIFIER_RE.match(name), f"{name} should be valid"

    def test_invalid_names(self):
        for name in ["123start", "has space", "has-dash", "import\nos", ""]:
            assert not VALID_IDENTIFIER_RE.match(name), f"{name!r} should be invalid"
