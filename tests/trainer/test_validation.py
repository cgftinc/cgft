"""Tests for validate_env_full — local environment validation."""

from __future__ import annotations

from cgft.envs.search_env import SearchEnv
from cgft.trainer.validation import validate_env_full

# ── Stubs ────────────────────────────────────────────────────────────

class StubSearch:
    """Minimal SearchClient for testing."""

    def search(self, query, mode="auto", top_k=10):
        return ["result one", "result two"]

    def embed(self, text):
        return [0.1, 0.2, 0.3]

    @property
    def available_modes(self):
        return ["vector"]

    def get_params(self):
        return {"backend": "stub"}


SAMPLE_DATA = [
    {"question": "What is X?", "answer": "Y", "reference_chunks": [{"content": "Y is X"}]},
    {"question": "What is A?", "answer": "B", "reference_chunks": [{"content": "B is A"}]},
]


# ── Happy path ───────────────────────────────────────────────────────

class TestHappyPath:
    def test_search_env_passes_all_checks(self, capsys):
        result = validate_env_full(
            env_class=SearchEnv,
            env_args={"search": StubSearch()},
            train_dataset=SAMPLE_DATA,
        )
        assert result is True
        out = capsys.readouterr().out
        assert "All" in out
        assert "passed" in out
        assert "\u2717" not in out

    def test_returns_true_on_success(self):
        result = validate_env_full(
            env_class=SearchEnv,
            env_args={"search": StubSearch()},
            train_dataset=SAMPLE_DATA,
        )
        assert result is True


# ── Failure: bad prompt type ─────────────────────────────────────────

class BadPromptEnv(SearchEnv):
    @classmethod
    def dataset_preprocess(cls, example, **kwargs):
        from benchmax.envs.types import StandardizedExample
        return StandardizedExample(
            prompt=["message1", "message2"],  # list, not string!
            ground_truth=example.get("answer", ""),
            init_rollout_args={},
        )


class TestBadPrompt:
    def test_list_prompt_fails_hashability(self, capsys):
        result = validate_env_full(
            env_class=BadPromptEnv,
            env_args={"search": StubSearch()},
            train_dataset=SAMPLE_DATA,
        )
        assert result is False
        out = capsys.readouterr().out
        assert "NOT hashable" in out or "not a string" in out


# ── Failure: broken run_tool ─────────────────────────────────────────

class BrokenToolEnv(SearchEnv):
    async def run_tool(self, rollout_id, tool_name, **tool_args):
        raise NotImplementedError("Not implemented")


class TestBrokenTool:
    def test_run_tool_error_caught(self, capsys):
        result = validate_env_full(
            env_class=BrokenToolEnv,
            env_args={"search": StubSearch()},
            train_dataset=SAMPLE_DATA,
        )
        assert result is False
        out = capsys.readouterr().out
        assert "run_tool raised NotImplementedError" in out


# ── Failure: bad reward return type ──────────────────────────────────

class BadRewardEnv(SearchEnv):
    async def compute_reward(self, rollout_id, completion, ground_truth=None, **kwargs):
        return {"score": "not_a_float"}


class TestBadReward:
    def test_non_float_reward_values(self, capsys):
        result = validate_env_full(
            env_class=BadRewardEnv,
            env_args={"search": StubSearch()},
            train_dataset=SAMPLE_DATA,
        )
        assert result is False
        out = capsys.readouterr().out
        assert "non-float" in out


# ── Failure: missing system_prompt ───────────────────────────────────

class NoPromptEnv(SearchEnv):
    system_prompt = ""


class TestNoSystemPrompt:
    def test_empty_system_prompt_fails(self, capsys):
        result = validate_env_full(
            env_class=NoPromptEnv,
            env_args={"search": StubSearch()},
            train_dataset=SAMPLE_DATA,
        )
        assert result is False
        out = capsys.readouterr().out
        assert "system_prompt" in out


# ── Edge case: empty dataset ─────────────────────────────────────────

class TestEmptyDataset:
    def test_empty_train_dataset_fails(self, capsys):
        result = validate_env_full(
            env_class=SearchEnv,
            env_args={"search": StubSearch()},
            train_dataset=[],
        )
        assert result is False


# ── Failure: broken dataset_preprocess ───────────────────────────────

class BrokenPreprocessEnv(SearchEnv):
    @classmethod
    def dataset_preprocess(cls, example, **kwargs):
        raise ValueError("bad data")


class TestBrokenPreprocess:
    def test_preprocess_exception_caught(self, capsys):
        result = validate_env_full(
            env_class=BrokenPreprocessEnv,
            env_args={"search": StubSearch()},
            train_dataset=SAMPLE_DATA,
        )
        assert result is False
        out = capsys.readouterr().out
        assert "dataset_preprocess raised ValueError" in out


# ── Simulated rollout ────────────────────────────────────────────────

class TestSimulatedRollout:
    def test_rollout_runs_with_reference_chunks(self, capsys):
        result = validate_env_full(
            env_class=SearchEnv,
            env_args={"search": StubSearch()},
            train_dataset=SAMPLE_DATA,
        )
        assert result is True
        out = capsys.readouterr().out
        assert "simulated rollout OK" in out
        assert "2 tool calls" in out

    def test_rollout_catches_broken_tool(self, capsys):
        result = validate_env_full(
            env_class=BrokenToolEnv,
            env_args={"search": StubSearch()},
            train_dataset=SAMPLE_DATA,
        )
        assert result is False
        out = capsys.readouterr().out
        # run_tool fails in check 4, so simulated rollout also fails
        assert "run_tool raised" in out


# ── Failure: NaN reward ───────────────────────────────────────────────

class NanRewardEnv(SearchEnv):
    async def compute_reward(self, rollout_id, completion, ground_truth=None, **kwargs):
        return {"score": float("nan")}


class TestNanReward:
    def test_nan_reward_fails(self, capsys):
        result = validate_env_full(
            env_class=NanRewardEnv,
            env_args={"search": StubSearch()},
            train_dataset=SAMPLE_DATA,
        )
        assert result is False
        out = capsys.readouterr().out
        assert "NaN/Inf" in out


# ── Failure: unpicklable env_args ────────────────────────────────────

class TestUnpicklableEnvArgs:
    def test_lambda_in_env_args_fails(self, capsys):
        import pickle as _pickle

        class Unpicklable:
            def __reduce__(self):
                raise _pickle.PicklingError("nope")

        result = validate_env_full(
            env_class=SearchEnv,
            env_args={"search": StubSearch(), "bad": Unpicklable()},
            train_dataset=SAMPLE_DATA,
        )
        assert result is False
        out = capsys.readouterr().out
        assert "env_args pickle failed" in out


# ── Verify output format ─────────────────────────────────────────────

class TestOutputFormat:
    def test_prints_all_check_names(self, capsys):
        validate_env_full(
            env_class=SearchEnv,
            env_args={"search": StubSearch()},
            train_dataset=SAMPLE_DATA,
        )
        out = capsys.readouterr().out
        assert "dataset_preprocess" in out
        assert "prompt" in out
        assert "load_dataset" in out
        assert "list_tools" in out
        assert "run_tool" in out
        assert "compute_reward" in out
        assert "simulated rollout" in out
        assert "pickle" in out
        assert "env_args pickle" in out
        assert "system_prompt" in out
