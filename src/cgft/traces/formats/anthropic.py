"""Anthropic message format normalization.

Converts Anthropic's native message format (content blocks with
``tool_use``/``tool_result``) into ``TraceMessage`` objects.

Anthropic format specifics:
- Roles: ``user``, ``assistant`` (no separate ``tool`` role)
- Content is always an array of typed blocks: ``text``, ``tool_use``,
  ``tool_result``, ``thinking``
- ``tool_use`` blocks in assistant messages contain the tool call
- ``tool_result`` blocks in user messages contain the tool response
- Tool result blocks reference ``tool_use_id`` (snake_case)
- ``input`` field on tool_use is always a dict
- ``thinking`` blocks are stripped (model reasoning, not trainable)

Key difference from OpenClaw: Anthropic embeds tool results as blocks
inside ``user`` messages. These are split out into separate ``tool``
role messages during normalization.
"""

from __future__ import annotations

import json
from typing import Any

from cgft.traces.adapter import ToolCall, TraceMessage


def normalize_anthropic_message(msg: dict[str, Any]) -> list[TraceMessage]:
    """Convert an Anthropic-format message to TraceMessage(s).

    Returns a list because ``user`` messages may contain ``tool_result``
    blocks that need to be split into separate ``tool`` role messages.
    An assistant message with ``tool_use`` blocks produces one
    TraceMessage with tool_calls populated.

    Handles:
    - Content as array of typed blocks → joined text + extracted ToolCalls
    - ``tool_use`` blocks → ``ToolCall`` objects
    - ``tool_result`` blocks in user messages → separate ``tool`` role messages
    - ``thinking`` block stripping
    - ``input`` as dict → JSON string for arguments
    """
    role = msg.get("role", "assistant")
    content = msg.get("content", "")

    # Simple string content (not typical for Anthropic but handle gracefully)
    if isinstance(content, str):
        return [TraceMessage(role=role, content=content)]

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

        elif btype == "tool_use":
            # Assistant's tool call
            tool_input = block.get("input", {})
            if isinstance(tool_input, dict):
                args = json.dumps(tool_input)
            elif isinstance(tool_input, str):
                args = tool_input
            else:
                args = str(tool_input)
            tool_calls.append(
                ToolCall(
                    name=block.get("name", ""),
                    arguments=args,
                    id=block.get("id"),
                )
            )

        elif btype == "tool_result":
            # Tool result embedded in user message — split into separate message
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                # Nested content blocks within tool_result
                parts = []
                for nested in result_content:
                    if isinstance(nested, dict) and nested.get("type") == "text":
                        parts.append(nested.get("text", ""))
                result_content = "\n".join(parts) if parts else ""
            elif not isinstance(result_content, str):
                result_content = str(result_content)

            tool_result_messages.append(
                TraceMessage(
                    role="tool",
                    content=str(result_content),
                    tool_call_id=str(block.get("tool_use_id", "")) or None,
                    name=None,  # Anthropic doesn't include tool name on results
                )
            )

        # Skip "thinking" blocks and unknown types silently

    joined_text = "\n".join(text_parts) if text_parts else ""

    # Build messages list
    messages: list[TraceMessage] = []

    # If we have tool results (from a user message), emit them BEFORE
    # any user text to maintain correct conversation ordering
    if tool_result_messages:
        messages.extend(tool_result_messages)
        # If there's also user text, add it as a separate message
        if joined_text:
            messages.append(TraceMessage(role=role, content=joined_text))
    else:
        # Normal case: one message with text + optional tool calls
        messages.append(
            TraceMessage(
                role=role,
                content=joined_text,
                tool_calls=tool_calls if tool_calls else None,
            )
        )

    return messages
