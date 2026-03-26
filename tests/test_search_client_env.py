"""Tests for SearchClientEnv — unified search environment."""

from __future__ import annotations

import asyncio
import pickle
from unittest.mock import AsyncMock, patch

import cloudpickle
import pytest

from cgft.corpus.search_client import SearchClient
from cgft.envs.search_client_env import SearchClientEnv


class StubSearch:
    """Minimal SearchClient for testing."""

    def __init__(self, modes=None, results=None):
        self._modes = modes or ["vector"]
        self._results = results if results is not None else ["result one", "result two"]

    def search(self, query, mode="auto", top_k=10):
        return self._results[:top_k]

    def embed(self, text):
        return [0.1, 0.2, 0.3]

    @property
    def available_modes(self):
        return self._modes

    def get_params(self):
        return {"backend": "stub"}


class TestInit:
    def test_isinstance_search_client(self):
        assert isinstance(StubSearch(), SearchClient)

    def test_tool_schema_has_query(self):
        env = SearchClientEnv(search=StubSearch())
        tool_def = env._tools["search"][0]
        assert "query" in tool_def.input_schema["properties"]
        assert "query" in tool_def.input_schema["required"]

    def test_no_mode_property_with_single_mode(self):
        env = SearchClientEnv(search=StubSearch(modes=["vector"]))
        tool_def = env._tools["search"][0]
        assert "mode" not in tool_def.input_schema["properties"]

    def test_mode_property_with_multiple_modes(self):
        env = SearchClientEnv(
            search=StubSearch(modes=["lexical", "vector", "hybrid"])
        )
        tool_def = env._tools["search"][0]
        assert "mode" in tool_def.input_schema["properties"]
        assert "hybrid" in tool_def.input_schema["properties"]["mode"]["enum"]

    def test_default_mode_hybrid_preferred(self):
        env = SearchClientEnv(
            search=StubSearch(modes=["lexical", "vector", "hybrid"])
        )
        assert env._default_mode == "hybrid"

    def test_default_mode_lexical_when_no_hybrid(self):
        env = SearchClientEnv(
            search=StubSearch(modes=["lexical", "vector"])
        )
        assert env._default_mode == "lexical"

    def test_default_mode_vector_fallback(self):
        env = SearchClientEnv(search=StubSearch(modes=["vector"]))
        assert env._default_mode == "vector"


class TestSearchTool:
    def test_empty_query_returns_error(self):
        env = SearchClientEnv(search=StubSearch())
        result = asyncio.run(env._search_tool(query=""))
        assert result.startswith("Error")

    def test_returns_formatted_results(self):
        env = SearchClientEnv(search=StubSearch(results=["foo", "bar"]))
        result = asyncio.run(env._search_tool(query="test"))
        assert "foo" in result
        assert "bar" in result

    def test_no_results(self):
        env = SearchClientEnv(search=StubSearch(results=[]))
        result = asyncio.run(env._search_tool(query="test"))
        assert result == "No results found."

    def test_delegates_to_search_client(self):
        calls = []

        class TrackingSearch(StubSearch):
            def search(self, query, mode="auto", top_k=10):
                calls.append({"query": query, "mode": mode, "top_k": top_k})
                return ["result"]

        env = SearchClientEnv(search=TrackingSearch())
        asyncio.run(env._search_tool(query="test query", limit=5))
        assert len(calls) == 1
        assert calls[0]["query"] == "test query"
        assert calls[0]["top_k"] == 5


class TestComputeReward:
    def test_overlap_fallback_without_judge(self):
        env = SearchClientEnv(search=StubSearch())
        result = asyncio.run(env.compute_reward(
            rollout_id="r1",
            completion="the reference text is here",
            ground_truth="",
            reference_chunks=[{"content": "the reference text is here"}],
        ))
        assert "chunk_overlap_reward_function" in result
        assert result["chunk_overlap_reward_function"] > 0.25

    def test_overlap_zero_no_match(self):
        env = SearchClientEnv(search=StubSearch())
        result = asyncio.run(env.compute_reward(
            rollout_id="r1",
            completion="completely different",
            ground_truth="",
            reference_chunks=[{"content": "nothing in common xyz"}],
        ))
        assert result["chunk_overlap_reward_function"] == 0.0

    @patch("cgft.rubrics.rubric.evaluate_single_rubric", new_callable=AsyncMock)
    def test_judge_reward_path(self, mock_eval):
        mock_eval.return_value = {"score": 0.8}
        env = SearchClientEnv(
            search=StubSearch(),
            judge_base_url="http://judge.test/v1",
            judge_api_key="test-key",
            judge_model="gpt-4o",
            w_correctness=0.5,
        )
        result = asyncio.run(env.compute_reward(
            rollout_id="r1",
            completion="The answer is <answer>42</answer>",
            ground_truth="42",
            prompt="What is the answer?",
        ))
        assert "correctness" in result
        assert result["correctness"] == pytest.approx(0.4)  # 0.8 * 0.5
        mock_eval.assert_awaited_once()

    def test_judge_empty_completion_returns_zero(self):
        env = SearchClientEnv(
            search=StubSearch(),
            judge_base_url="http://judge.test/v1",
            judge_api_key="test-key",
            judge_model="gpt-4o",
        )
        result = asyncio.run(env.compute_reward(
            rollout_id="r1",
            completion="   ",
            ground_truth="42",
        ))
        assert result["correctness"] == 0.0


class TestDatasetPreprocess:
    def test_extracts_question_answer(self):
        result = SearchClientEnv.dataset_preprocess(
            {"question": "What is X?", "answer": "Y"}
        )
        assert result["prompt"] == "What is X?"
        assert result["ground_truth"] == "Y"

    def test_missing_question_returns_empty(self):
        result = SearchClientEnv.dataset_preprocess({"answer": "Y"})
        assert result["prompt"] == ""

    def test_missing_answer_returns_none(self):
        result = SearchClientEnv.dataset_preprocess({"question": "What?"})
        assert result["ground_truth"] is None


class TestListTools:
    def test_returns_tool_definitions(self):
        env = SearchClientEnv(search=StubSearch())
        tools = asyncio.run(env.list_tools())
        assert len(tools) == 1
        assert tools[0].name == "search"


class TestPickle:
    def test_class_pickle(self):
        data = cloudpickle.dumps(SearchClientEnv)
        restored = pickle.loads(data)
        assert restored.__name__ == "SearchClientEnv"

    def test_instance_pickle_roundtrip(self):
        search = StubSearch(modes=["lexical", "vector"])
        env = SearchClientEnv(search=search)
        data = cloudpickle.dumps(env)
        restored = pickle.loads(data)
        assert isinstance(restored, SearchClientEnv)
        assert restored._default_mode == "lexical"
        assert restored._search._modes == ["lexical", "vector"]
        # Verify the restored env still works
        result = asyncio.run(restored._search_tool(query="test"))
        assert "result one" in result
