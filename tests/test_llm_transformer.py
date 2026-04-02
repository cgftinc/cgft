"""Tests for LLMStyleTransformer — including the keyword post-hoc length check."""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock, patch

from cgft.qa_generation.cgft_models import TransformationConfig
from cgft.qa_generation.generated_qa import GeneratedQA
from cgft.qa_generation.transformers.llm_transformer import (
    LLMStyleTransformer,
    _MAX_KEYWORD_TOKENS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> MagicMock:
    """Return a minimal mock context with a seeded RNG.

    Mirrors the pattern used in test_hop_count_validity.py and test_dedup.py:
    use a MagicMock so we don't need to satisfy CgftPipelineConfig's required fields,
    but attach a real random.Random so style sampling works correctly.
    """
    ctx = MagicMock()
    ctx.rng = random.Random(42)
    ctx.setdefault = lambda key, default: default
    return ctx


def _make_item(
    *,
    question: str = "How do I configure feature flags?",
    answer: str = "Use the PostHog SDK init call with your API key and a flags configuration.",
    qa_type: str = "lookup",
) -> GeneratedQA:
    return GeneratedQA(
        qa={
            "question": question,
            "answer": answer,
            "qa_type": qa_type,
            "reference_chunks": [{"id": "c1", "metadata": {}, "content": "Some content."}],
            "min_hop_count": 1,
            "is_co_located": None,
            "filter_status": None,
            "filter_reasoning": None,
            "no_context_answer": None,
            "eval_scores": {},
        },
        generation_metadata={"qa_type_target": qa_type, "refinement_count": 0},
    )


def _make_transformer(*, style: str = "keyword") -> LLMStyleTransformer:
    """Return an LLMStyleTransformer pinned to the given style, with validation disabled."""
    cfg = TransformationConfig(
        model="test-model",
        api_key="test-key",
        base_url="http://test",
        style_distribution={style: 1.0},
        validation_enabled=False,
    )
    return LLMStyleTransformer(cfg, enable_validation=False)


def _make_openai_response(question_text: str) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response returning the given question JSON."""
    payload = json.dumps({"question": question_text})
    choice = MagicMock()
    choice.message.content = payload
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKeywordPostHocLengthCheck:
    """The transformer must fall back when the LLM returns a keyword that is too long."""

    def test_keyword_too_long_falls_back_to_original(self):
        """When the LLM returns a keyword with > _MAX_KEYWORD_TOKENS tokens, the
        transformer should fall back to the original question and record
        error metadata 'keyword_too_long'."""
        assert _MAX_KEYWORD_TOKENS == 7, (
            "Test assumes _MAX_KEYWORD_TOKENS=7; update if the constant changes."
        )

        # Build a keyword that is definitely over the limit (10 space-separated tokens).
        oversized_keyword = " ".join(f"tok{i}" for i in range(10))
        assert len(oversized_keyword.split()) == 10

        transformer = _make_transformer(style="keyword")
        item = _make_item()
        original_question = item.qa["question"]

        with patch.object(transformer.client, "chat") as mock_chat:
            mock_chat.completions.create.return_value = _make_openai_response(oversized_keyword)
            ctx = _make_context()
            transformer.transform([item], ctx)

        # The question must be unchanged (fallback to original).
        assert item.qa["question"] == original_question

        # The transformation metadata must contain the error code.
        steps = item.generation_metadata.get("transformation", {}).get("steps", [])
        assert steps, "Expected at least one transformation step recorded."
        last_step = steps[-1]
        assert last_step.get("error") == "keyword_too_long", (
            f"Expected error='keyword_too_long' in step metadata, got: {last_step}"
        )
        assert last_step.get("keyword_tokens") == 10

    def test_short_keyword_is_accepted(self):
        """A keyword within the token limit is accepted and replaces the original question."""
        short_keyword = "posthog feature flags"  # 3 tokens, well within limit
        assert len(short_keyword.split()) <= _MAX_KEYWORD_TOKENS

        transformer = _make_transformer(style="keyword")
        item = _make_item()

        with patch.object(transformer.client, "chat") as mock_chat:
            mock_chat.completions.create.return_value = _make_openai_response(short_keyword)
            ctx = _make_context()
            transformer.transform([item], ctx)

        assert item.qa["question"] == short_keyword

    def test_keyword_exactly_at_limit_is_accepted(self):
        """A keyword with exactly _MAX_KEYWORD_TOKENS tokens must be accepted."""
        keyword_at_limit = " ".join(f"word{i}" for i in range(_MAX_KEYWORD_TOKENS))
        assert len(keyword_at_limit.split()) == _MAX_KEYWORD_TOKENS

        transformer = _make_transformer(style="keyword")
        item = _make_item()

        with patch.object(transformer.client, "chat") as mock_chat:
            mock_chat.completions.create.return_value = _make_openai_response(keyword_at_limit)
            ctx = _make_context()
            transformer.transform([item], ctx)

        assert item.qa["question"] == keyword_at_limit
