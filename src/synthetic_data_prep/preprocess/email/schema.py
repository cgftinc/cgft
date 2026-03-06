"""Canonical email schema validation helpers.

Canonical input schema (JSONL rows):
- One object per parsed email message.
- Required fields:
  - "id": stable message id.
  - "thread_id": conversation/thread identifier.
  - "date": sortable timestamp string.
  - "subject": subject text (can be empty).
  - "body": body text (can be empty).
- "from": address object.
- Required linkage fields:
  - "reply_to": parent message id (empty for root messages).
- Optional fields (used when present):
  - "to": list of address objects.
  - "cc": list of address objects.
  - other metadata fields are preserved in message-level output.

Expected usage:
- Parsers produce canonical message rows according to the schema.
- Dedupe/chunking call `validate_rows(...)` to collect non-fatal warnings.

Validation is non-fatal by design: this module surfaces warnings instead of
raising, so pipelines can continue on large corpora with sparse bad rows.
"""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS = ("id", "thread_id", "date", "subject", "body", "from", "reply_to")


def as_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def validate_rows(rows: list[dict[str, Any]], max_warnings: int = 200) -> list[str]:
    """Collect schema warnings for canonical parsed-email rows.

    Returns:
    - list of warning strings (empty when no issues detected)
    """

    warnings: list[str] = []
    for idx, row in enumerate(rows):
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            warnings.append(
                f"row={idx} missing required fields={missing} (id={as_str(row.get('id'))!r}, "
                f"thread_id={as_str(row.get('thread_id'))!r})"
            )
            if len(warnings) >= max_warnings:
                break
            continue

        if not as_str(row.get("id")):
            warnings.append(
                f"row={idx} has empty 'id' (thread_id={as_str(row.get('thread_id'))!r})"
            )
        if not as_str(row.get("thread_id")):
            warnings.append(
                f"row={idx} has empty 'thread_id' (id={as_str(row.get('id'))!r})"
            )
        if not isinstance(row.get("from"), dict):
            warnings.append(
                f"row={idx} has invalid 'from' type={type(row.get('from')).__name__}; expected object"
            )
        for opt_list in ("to", "cc"):
            if opt_list in row and not isinstance(row.get(opt_list), list):
                warnings.append(
                    f"row={idx} has invalid '{opt_list}' type={type(row.get(opt_list)).__name__}; "
                    "expected list"
                )
        if len(warnings) >= max_warnings:
            break

    return warnings
