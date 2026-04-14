"""Tests for OpenClaw message format normalization."""

import json
import tempfile
from pathlib import Path

import pytest

from cgft.traces.formats.openclaw import (
    normalize_openclaw_message,
    parse_openclaw_session,
)


class TestNormalizeOpenclawMessage:
    def test_text_only_assistant(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello, how can I help?"}],
        }
        result = normalize_openclaw_message(msg)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == "Hello, how can I help?"
        assert result[0].tool_calls is None

    def test_assistant_with_tool_call(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check that."},
                {
                    "type": "toolCall",
                    "name": "exec",
                    "id": "call_abc",
                    "arguments": {"command": "skys status gpu4"},
                },
            ],
        }
        result = normalize_openclaw_message(msg)
        assert len(result) == 1
        assert result[0].content == "Let me check that."
        assert result[0].tool_calls is not None
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].name == "exec"
        assert result[0].tool_calls[0].id == "call_abc"
        assert '"command"' in result[0].tool_calls[0].arguments

    def test_multiple_tool_calls(self):
        """OpenClaw commonly has 2+ tool calls in one message."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking both."},
                {"type": "toolCall", "name": "exec", "arguments": {"command": "ls"}},
                {"type": "toolCall", "name": "read", "arguments": {"path": "/tmp/x"}},
            ],
        }
        result = normalize_openclaw_message(msg)
        assert len(result[0].tool_calls) == 2
        assert result[0].tool_calls[0].name == "exec"
        assert result[0].tool_calls[1].name == "read"

    def test_toolresult_role_mapped_to_tool(self):
        msg = {
            "role": "toolResult",
            "toolCallId": "call_abc",
            "toolName": "exec",
            "content": [{"type": "text", "text": "gpu4: healthy"}],
        }
        result = normalize_openclaw_message(msg)
        assert len(result) == 1
        assert result[0].role == "tool"
        assert result[0].content == "gpu4: healthy"
        assert result[0].tool_call_id == "call_abc"
        assert result[0].name == "exec"

    def test_image_blocks_stripped(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here's the image."},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        }
        result = normalize_openclaw_message(msg)
        assert result[0].content == "Here's the image."
        assert result[0].tool_calls is None

    def test_arguments_as_dict(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "name": "exec", "arguments": {"command": "pwd"}},
            ],
        }
        result = normalize_openclaw_message(msg)
        assert result[0].tool_calls[0].arguments == '{"command": "pwd"}'

    def test_arguments_as_string(self):
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "toolCall",
                    "name": "exec",
                    "arguments": '{"command": "pwd"}',
                },
            ],
        }
        result = normalize_openclaw_message(msg)
        assert result[0].tool_calls[0].arguments == '{"command": "pwd"}'

    def test_embedded_toolresult_block(self):
        """toolResult blocks can appear inside assistant content arrays."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here's what I found:"},
                {"type": "toolResult", "text": "file contents here"},
            ],
        }
        result = normalize_openclaw_message(msg)
        assert len(result) == 2
        assert result[0].role == "assistant"
        assert result[0].content == "Here's what I found:"
        assert result[1].role == "tool"
        assert result[1].content == "file contents here"

    def test_string_content_fallback(self):
        """Rare but possible: OpenClaw message with plain string content."""
        msg = {"role": "user", "content": "hello"}
        result = normalize_openclaw_message(msg)
        assert len(result) == 1
        assert result[0].content == "hello"

    def test_empty_content_array(self):
        msg = {"role": "assistant", "content": []}
        result = normalize_openclaw_message(msg)
        assert len(result) == 1
        assert result[0].content == ""

    def test_user_message(self):
        msg = {
            "role": "user",
            "content": [{"type": "text", "text": "Check GPU status"}],
        }
        result = normalize_openclaw_message(msg)
        assert result[0].role == "user"
        assert result[0].content == "Check GPU status"

    def test_tool_call_without_text(self):
        """Assistant message with only tool calls, no text."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "name": "exec", "arguments": {"command": "ls"}},
            ],
        }
        result = normalize_openclaw_message(msg)
        assert result[0].content == ""
        assert result[0].tool_calls is not None


class TestParseOpenclawSession:
    def _write_session(self, messages: list[dict], tmpdir: Path) -> Path:
        """Write a session JSONL file with message events."""
        path = tmpdir / "test_session.jsonl"
        lines = []
        for msg in messages:
            event = {"type": "message", "id": "evt_1", "message": msg}
            lines.append(json.dumps(event))
        path.write_text("\n".join(lines))
        return path

    def test_basic_session(self, tmp_path):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
            },
        ]
        path = self._write_session(messages, tmp_path)
        traces = parse_openclaw_session(path)
        assert len(traces) == 1
        assert traces[0].id == "test_session"
        assert len(traces[0].messages) == 2
        assert traces[0].messages[0].role == "user"
        assert traces[0].messages[1].role == "assistant"

    def test_filters_non_message_events(self, tmp_path):
        path = tmp_path / "session.jsonl"
        lines = [
            json.dumps({"type": "session", "id": "sess_1"}),
            json.dumps({"type": "model_change", "model": "opus"}),
            json.dumps({
                "type": "message",
                "message": {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            }),
            json.dumps({"type": "thinking_level_change", "level": "high"}),
            json.dumps({
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                },
            }),
        ]
        path.write_text("\n".join(lines))
        traces = parse_openclaw_session(path)
        assert len(traces) == 1
        assert len(traces[0].messages) == 2

    def test_with_tool_calls(self, tmp_path):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Check status"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "toolCall", "name": "exec", "arguments": {"command": "status"}},
                ],
            },
            {
                "role": "toolResult",
                "toolCallId": "call_1",
                "toolName": "exec",
                "content": [{"type": "text", "text": "all good"}],
            },
        ]
        path = self._write_session(messages, tmp_path)
        traces = parse_openclaw_session(path)
        msgs = traces[0].messages
        assert len(msgs) == 3
        assert msgs[1].tool_calls is not None
        assert msgs[1].tool_calls[0].name == "exec"
        assert msgs[2].role == "tool"
        assert msgs[2].tool_call_id == "call_1"

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        traces = parse_openclaw_session(path)
        assert traces == []

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_openclaw_session(tmp_path / "nonexistent.jsonl")

    def test_malformed_json_skipped(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        lines = [
            '{"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "Hi"}]}}',
            "this is not json",
            '{"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}}',
        ]
        path.write_text("\n".join(lines))
        traces = parse_openclaw_session(path)
        assert len(traces[0].messages) == 2

    def test_metadata_includes_source(self, tmp_path):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        ]
        path = self._write_session(messages, tmp_path)
        traces = parse_openclaw_session(path)
        assert traces[0].metadata["source"] == "openclaw"
