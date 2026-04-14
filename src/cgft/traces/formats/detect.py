"""Auto-detect and normalize messages in any supported format.

Supports three message formats:
- **OpenAI**: ``content`` is a string, tool calls in ``tool_calls`` field
- **Anthropic**: ``content`` is an array with ``tool_use``/``tool_result`` blocks
- **OpenClaw**: ``content`` is an array with ``toolCall``/``toolResult`` blocks (camelCase)

Detection is per-message based on content structure and block type names.
"""

from __future__ import annotations

from typing import Any

from cgft.traces.adapter import ToolCall, TraceMessage


def detect_message_format(msg: dict[str, Any]) -> str:
    """Detect the message format from content structure.

    Returns ``"openai"``, ``"anthropic"``, ``"openclaw"``, or ``"unknown"``.
    """
    content = msg.get("content")
    if content is None or isinstance(content, str):
        return "openai"
    if isinstance(content, list):
        block_types: set[str] = set()
        for block in content:
            if isinstance(block, dict):
                block_types.add(block.get("type", ""))
        if "toolCall" in block_types or "toolResult" in block_types:
            return "openclaw"
        if "tool_use" in block_types or "tool_result" in block_types:
            return "anthropic"
        # OpenAI multi-modal: only text/image_url/image blocks
        return "openai"
    return "unknown"


def normalize_message(msg: dict[str, Any]) -> list[TraceMessage]:
    """Auto-detect format and normalize a single message to TraceMessage(s).

    Returns a list because some formats (Anthropic) embed tool results
    inside user messages, which need to be split into separate messages.

    For OpenAI format, delegates to ``_normalize_openai_message``.
    For Anthropic, delegates to ``cgft.traces.formats.anthropic``.
    For OpenClaw, delegates to ``cgft.traces.formats.openclaw``.
    """
    fmt = detect_message_format(msg)
    if fmt == "openclaw":
        from cgft.traces.formats.openclaw import normalize_openclaw_message

        return normalize_openclaw_message(msg)
    if fmt == "anthropic":
        from cgft.traces.formats.anthropic import normalize_anthropic_message

        return normalize_anthropic_message(msg)
    return [_normalize_openai_message(msg)]


def _normalize_openai_message(msg: dict[str, Any]) -> TraceMessage:
    """Normalize an OpenAI-format message to TraceMessage.

    Handles tool_calls in both nested function format
    (``{function: {name, arguments}, id}``) and direct format
    (``{name, arguments, id}``).
    """
    role = msg.get("role", "assistant")
    content = msg.get("content")
    if content is None:
        content = ""
    content = str(content)

    tool_calls = _parse_openai_tool_calls(msg)
    tool_call_id = msg.get("tool_call_id")
    name = msg.get("name")

    return TraceMessage(
        role=role,
        content=content,
        tool_calls=tool_calls if tool_calls else None,
        tool_call_id=str(tool_call_id) if tool_call_id else None,
        name=str(name) if name else None,
    )


def _parse_openai_tool_calls(msg: dict[str, Any]) -> list[ToolCall]:
    """Extract tool calls from an OpenAI-format message."""
    raw_calls = msg.get("tool_calls")
    if not raw_calls:
        # Try older "function" format
        func = msg.get("function")
        if func and isinstance(func, dict):
            return [
                ToolCall(
                    name=func.get("name", ""),
                    arguments=_ensure_json_string(func.get("arguments", "{}")),
                    id=msg.get("id"),
                )
            ]
        return []

    if not isinstance(raw_calls, list):
        return []

    result: list[ToolCall] = []
    for tc in raw_calls:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", {})
        if isinstance(func, dict) and "name" in func:
            result.append(
                ToolCall(
                    name=func["name"],
                    arguments=_ensure_json_string(func.get("arguments", "{}")),
                    id=tc.get("id"),
                )
            )
        elif "name" in tc:
            result.append(
                ToolCall(
                    name=tc["name"],
                    arguments=_ensure_json_string(tc.get("arguments", "{}")),
                    id=tc.get("id"),
                )
            )
    return result


def _ensure_json_string(value: Any) -> str:
    """Ensure value is a JSON string. If it's a dict, serialise it."""
    import json

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)
