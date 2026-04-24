"""OpenClaw message format normalization.

Converts OpenClaw's native message format (content-as-array with camelCase
fields) into ``TraceMessage`` objects.

OpenClaw format specifics:
- Roles: ``user``, ``assistant``, ``toolResult`` (mapped to ``tool``)
- Content is always an array of typed blocks: ``text``, ``toolCall``,
  ``toolResult``, ``image``
- Tool calls are inline content blocks, not a separate ``tool_calls`` field
- ``toolResult`` messages use camelCase: ``toolCallId``, ``toolName``
- ``arguments`` can be a dict or a pre-stringified JSON string
- Image blocks are stripped (can't train on binary data)
- Non-message events (``session``, ``model_change``) are filtered out

Session parsing: ``parse_openclaw_session()`` reads a raw session JSONL file,
filters to message events, and produces ``NormalizedTrace`` objects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cgft.traces.adapter import NormalizedTrace, ToolCall, TraceMessage

logger = logging.getLogger(__name__)


def normalize_openclaw_message(msg: dict[str, Any]) -> list[TraceMessage]:
    """Convert an OpenClaw-format message to TraceMessage(s).

    Returns a list (usually length 1). An assistant message with mixed
    text + toolCall blocks produces one TraceMessage with both content
    and tool_calls populated.

    Handles:
    - Content as array of typed blocks → joined text + extracted ToolCalls
    - ``toolResult`` role → ``tool`` role
    - camelCase fields → snake_case
    - Image block stripping
    - ``toolResult`` blocks embedded inside content arrays
    - ``arguments`` as dict or pre-stringified JSON string
    """
    role = msg.get("role", "assistant")
    content = msg.get("content", "")

    # Map OpenClaw roles to standard roles
    if role == "toolResult":
        role = "tool"

    # Simple string content (rare in OpenClaw but possible)
    if isinstance(content, str):
        return [
            TraceMessage(
                role=role,
                content=content,
                tool_call_id=str(msg.get("toolCallId", "")) or None,
                name=str(msg.get("toolName", "")) or None,
            )
        ]

    if not isinstance(content, list):
        return [TraceMessage(role=role, content=str(content))]

    # Parse content block array
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_result_messages: list[TraceMessage] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")

        if btype == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(str(text))

        elif btype == "toolCall":
            args = block.get("arguments", "{}")
            if isinstance(args, dict):
                args = json.dumps(args)
            tool_calls.append(
                ToolCall(
                    name=block.get("name", ""),
                    arguments=str(args),
                    id=block.get("id"),
                )
            )

        elif btype == "toolResult":
            # toolResult blocks embedded inside content arrays
            result_text = block.get("text", "")
            if not result_text:
                # Try nested content
                nested = block.get("content")
                if isinstance(nested, list):
                    result_text = " ".join(b.get("text", "") for b in nested if isinstance(b, dict))
                elif isinstance(nested, str):
                    result_text = nested
            tool_result_messages.append(
                TraceMessage(
                    role="tool",
                    content=str(result_text),
                    tool_call_id=str(block.get("toolCallId", block.get("tool_use_id", ""))) or None,
                    name=str(block.get("toolName", "")) or None,
                )
            )

        # Skip image blocks and unknown types silently

    joined_text = "\n".join(text_parts) if text_parts else ""

    # Build the primary message
    primary = TraceMessage(
        role=role,
        content=joined_text,
        tool_calls=tool_calls if tool_calls else None,
        tool_call_id=str(msg.get("toolCallId", "")) or None,
        name=str(msg.get("toolName", "")) or None,
    )

    # Return primary + any embedded tool result messages
    result = [primary]
    result.extend(tool_result_messages)
    return result


def parse_openclaw_session(
    path: str | Path,
) -> list[NormalizedTrace]:
    """Parse an OpenClaw session JSONL file into NormalizedTrace objects.

    Reads the JSONL file, filters to ``type: "message"`` events, and
    groups messages into a single NormalizedTrace per session file.
    Each contiguous conversation becomes one trace.

    Does NOT segment multi-topic conversations — that's a separate
    preprocessing step the caller handles.

    Args:
        path: Path to an OpenClaw session JSONL file.

    Returns:
        List of NormalizedTrace objects (typically one per file, but
        splits on large time gaps > 30 minutes if detected).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OpenClaw session file not found: {path}")

    raw_messages: list[dict[str, Any]] = []
    for line_num, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON at %s:%d", path, line_num)
            continue

        # Filter to message events only
        if not isinstance(event, dict):
            continue
        if event.get("type") != "message":
            continue

        msg = event.get("message", event)
        if not isinstance(msg, dict):
            continue
        if "role" not in msg:
            continue

        raw_messages.append(msg)

    if not raw_messages:
        return []

    # Normalize all messages
    normalized_messages: list[TraceMessage] = []
    for msg in raw_messages:
        normalized_messages.extend(normalize_openclaw_message(msg))

    # Create a single trace from the session
    trace_id = path.stem
    return [
        NormalizedTrace(
            id=trace_id,
            messages=normalized_messages,
            metadata={"source": "openclaw", "file": str(path.name)},
        )
    ]
