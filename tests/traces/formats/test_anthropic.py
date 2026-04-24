"""Tests for Anthropic message format normalization.

Test data modeled on ZClawBench dataset format.
"""

from cgft.traces.formats.anthropic import normalize_anthropic_message


class TestNormalizeAnthropicMessage:
    def test_text_only_assistant(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "I can help with that."}],
        }
        result = normalize_anthropic_message(msg)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == "I can help with that."
        assert result[0].tool_calls is None

    def test_assistant_with_tool_use(self):
        """ZClawBench format: tool_use block in assistant content."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search for that."},
                {
                    "type": "tool_use",
                    "id": "toolu_vrtx_abc123",
                    "name": "search",
                    "input": {"query": "revenue Q4"},
                },
            ],
        }
        result = normalize_anthropic_message(msg)
        assert len(result) == 1
        assert result[0].content == "Let me search for that."
        assert result[0].tool_calls is not None
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].name == "search"
        assert result[0].tool_calls[0].id == "toolu_vrtx_abc123"
        assert '"query"' in result[0].tool_calls[0].arguments

    def test_user_with_tool_result(self):
        """Anthropic embeds tool results inside user messages."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_vrtx_abc123",
                    "content": "Revenue was $1.2M in Q4.",
                },
            ],
        }
        result = normalize_anthropic_message(msg)
        assert len(result) == 1
        assert result[0].role == "tool"
        assert result[0].content == "Revenue was $1.2M in Q4."
        assert result[0].tool_call_id == "toolu_vrtx_abc123"

    def test_user_with_tool_result_and_text(self):
        """User message with both tool_result and text blocks."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "search result here",
                },
                {"type": "text", "text": "Now also check the database."},
            ],
        }
        result = normalize_anthropic_message(msg)
        # tool_result becomes tool message, text becomes user message
        assert len(result) == 2
        assert result[0].role == "tool"
        assert result[0].content == "search result here"
        assert result[1].role == "user"
        assert result[1].content == "Now also check the database."

    def test_multiple_tool_use_blocks(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "/a"}},
                {"type": "tool_use", "id": "t2", "name": "read", "input": {"path": "/b"}},
            ],
        }
        result = normalize_anthropic_message(msg)
        assert len(result) == 1
        assert len(result[0].tool_calls) == 2
        assert result[0].tool_calls[0].name == "read"
        assert result[0].tool_calls[1].name == "read"
        assert result[0].tool_calls[0].id == "t1"
        assert result[0].tool_calls[1].id == "t2"

    def test_thinking_blocks_stripped(self):
        """ZClawBench includes thinking blocks — should be stripped."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I should search first..."},
                {"type": "text", "text": "Let me look that up."},
                {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
            ],
        }
        result = normalize_anthropic_message(msg)
        assert len(result) == 1
        assert result[0].content == "Let me look that up."
        assert "thinking" not in result[0].content
        assert result[0].tool_calls is not None

    def test_tool_result_with_nested_content_blocks(self):
        """tool_result content can be a list of blocks."""
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [
                        {"type": "text", "text": "Line 1"},
                        {"type": "text", "text": "Line 2"},
                    ],
                },
            ],
        }
        result = normalize_anthropic_message(msg)
        assert result[0].role == "tool"
        assert result[0].content == "Line 1\nLine 2"

    def test_multiple_tool_results_in_user_message(self):
        """User message with multiple tool results (parallel tool calls)."""
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "result 1"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "result 2"},
            ],
        }
        result = normalize_anthropic_message(msg)
        assert len(result) == 2
        assert result[0].tool_call_id == "t1"
        assert result[1].tool_call_id == "t2"

    def test_string_content_fallback(self):
        msg = {"role": "user", "content": "plain text"}
        result = normalize_anthropic_message(msg)
        assert len(result) == 1
        assert result[0].content == "plain text"

    def test_empty_content_array(self):
        msg = {"role": "assistant", "content": []}
        result = normalize_anthropic_message(msg)
        assert len(result) == 1
        assert result[0].content == ""

    def test_tool_use_with_string_input(self):
        """Edge case: input as string instead of dict."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "exec", "input": '{"cmd": "ls"}'},
            ],
        }
        result = normalize_anthropic_message(msg)
        assert result[0].tool_calls[0].arguments == '{"cmd": "ls"}'

    def test_assistant_tool_call_only_no_text(self):
        """Assistant message with only tool_use, no text."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "/x"}},
            ],
        }
        result = normalize_anthropic_message(msg)
        assert result[0].content == ""
        assert result[0].tool_calls is not None
        assert result[0].tool_calls[0].name == "read"
