"""Tests for message format auto-detection and dispatch."""

from cgft.traces.formats.detect import (
    detect_message_format,
    normalize_message,
    _normalize_openai_message,
)


class TestDetectMessageFormat:
    def test_string_content_is_openai(self):
        assert detect_message_format({"role": "user", "content": "hello"}) == "openai"

    def test_none_content_is_openai(self):
        assert detect_message_format({"role": "assistant", "content": None}) == "openai"

    def test_missing_content_is_openai(self):
        assert detect_message_format({"role": "assistant"}) == "openai"

    def test_array_with_toolcall_is_openclaw(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "toolCall", "name": "exec", "arguments": {}},
            ],
        }
        assert detect_message_format(msg) == "openclaw"

    def test_array_with_toolresult_is_openclaw(self):
        msg = {
            "role": "toolResult",
            "content": [{"type": "toolResult", "text": "output"}],
        }
        assert detect_message_format(msg) == "openclaw"

    def test_array_with_tool_use_is_anthropic(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Searching..."},
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}},
            ],
        }
        assert detect_message_format(msg) == "anthropic"

    def test_array_with_tool_result_is_anthropic(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "found it"},
            ],
        }
        assert detect_message_format(msg) == "anthropic"

    def test_multimodal_openai_is_openai(self):
        """OpenAI multi-modal: content is array but only text/image_url blocks."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://..."}},
            ],
        }
        assert detect_message_format(msg) == "openai"

    def test_text_only_array_is_openai(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        assert detect_message_format(msg) == "openai"

    def test_empty_array_is_openai(self):
        msg = {"role": "user", "content": []}
        assert detect_message_format(msg) == "openai"

    def test_image_only_array_is_openai(self):
        msg = {"role": "user", "content": [{"type": "image", "source": {}}]}
        assert detect_message_format(msg) == "openai"


class TestNormalizeMessage:
    def test_dispatches_openai(self):
        msg = {"role": "user", "content": "hello"}
        result = normalize_message(msg)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "hello"

    def test_dispatches_openclaw(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {"type": "toolCall", "name": "exec", "arguments": {"cmd": "ls"}},
            ],
        }
        result = normalize_message(msg)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == "Checking."
        assert result[0].tool_calls is not None
        assert result[0].tool_calls[0].name == "exec"

    def test_dispatches_anthropic(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "test"}},
            ],
        }
        result = normalize_message(msg)
        assert len(result) == 1
        assert result[0].tool_calls is not None
        assert result[0].tool_calls[0].name == "search"


class TestNormalizeOpenaiMessage:
    def test_basic_message(self):
        msg = {"role": "user", "content": "hello"}
        result = _normalize_openai_message(msg)
        assert result.role == "user"
        assert result.content == "hello"
        assert result.tool_calls is None

    def test_none_content(self):
        msg = {"role": "assistant", "content": None}
        result = _normalize_openai_message(msg)
        assert result.content == ""

    def test_tool_calls_nested_function(self):
        """OpenAI nested function format: tool_calls[].function.{name, arguments}"""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                    "id": "call_1",
                    "type": "function",
                }
            ],
        }
        result = _normalize_openai_message(msg)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].id == "call_1"

    def test_tool_calls_direct_format(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "lookup", "arguments": "{}", "id": "tc2"}
            ],
        }
        result = _normalize_openai_message(msg)
        assert result.tool_calls[0].name == "lookup"

    def test_old_function_format(self):
        msg = {
            "role": "assistant",
            "content": "calling",
            "function": {"name": "old_tool", "arguments": "{}"},
        }
        result = _normalize_openai_message(msg)
        assert result.tool_calls[0].name == "old_tool"

    def test_tool_response(self):
        msg = {
            "role": "tool",
            "content": "result data",
            "tool_call_id": "call_1",
            "name": "search",
        }
        result = _normalize_openai_message(msg)
        assert result.role == "tool"
        assert result.tool_call_id == "call_1"
        assert result.name == "search"

    def test_dict_arguments_serialized(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "search", "arguments": {"query": "test"}, "id": "tc1"}
            ],
        }
        result = _normalize_openai_message(msg)
        assert result.tool_calls[0].arguments == '{"query": "test"}'
