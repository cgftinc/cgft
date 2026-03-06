"""Parse .mbox files into canonical email JSONL rows.

Expected usage:
- Run once per mailbox export to produce canonical JSONL.
- Optionally run dedupe next, then pass resulting JSONL to `EmailChunker`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mailbox
import re
from collections import Counter
from datetime import datetime
from email.header import decode_header, make_header
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Any


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return str(value).strip()


def _normalize_message_id(value: str | None) -> str:
    text = _decode_header_value(value).strip()
    if not text:
        return ""
    m = re.search(r"<([^>]+)>", text)
    if m:
        return m.group(1).strip()
    return text.strip("<>").strip()


def _parse_addr_header(value: str | None) -> list[dict[str, str]]:
    def to_addr_dict(name: str, email: str) -> dict[str, str]:
        return {"name": _decode_header_value(name).strip(), "email": (email or "").strip()}

    out: list[dict[str, str]] = []
    if not value:
        return out
    for name, email in getaddresses([value]):
        out.append(to_addr_dict(name, email))
    # Stable de-dup.
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for a in out:
        key = (a.get("name", ""), a.get("email", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(a)
    return unique


def _extract_body_text(msg: Message) -> str:
    if msg.is_multipart():
        plain_parts: list[str] = []
        html_parts: list[str] = []
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if (part.get("Content-Disposition") or "").lower().startswith("attachment"):
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                text = payload.decode(charset, errors="replace")
            except Exception:
                text = payload.decode("utf-8", errors="replace")
            ctype = (part.get_content_type() or "").lower()
            if ctype == "text/plain":
                plain_parts.append(text)
            elif ctype == "text/html":
                html_parts.append(text)
        body = "\n\n".join(plain_parts) if plain_parts else "\n\n".join(html_parts)
    else:
        payload = msg.get_payload(decode=True)
        if payload is None:
            body = str(msg.get_payload() or "")
        else:
            charset = msg.get_content_charset() or "utf-8"
            try:
                body = payload.decode(charset, errors="replace")
            except Exception:
                body = payload.decode("utf-8", errors="replace")

    # Lightweight cleanup for HTML fallback and whitespace.
    body = re.sub(r"(?i)<br\s*/?>", "\n", body)
    body = re.sub(r"(?i)</p\s*>", "\n", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"\s+", " ", body or "").strip()
    return body


def _parse_date_iso(raw_date: str | None) -> str:
    if not raw_date:
        return ""
    try:
        dt = parsedate_to_datetime(raw_date)
        if dt is None:
            return ""
        return dt.isoformat()
    except Exception:
        return ""


def _thread_id_for_message(message_id: str, reply_to: str, refs: list[str], subject: str) -> str:
    subject_key = re.sub(r"^(re|fw|fwd)\s*:\s*", "", subject.lower()).strip()
    anchor = refs[0] if refs else (reply_to or subject_key or message_id)
    digest = hashlib.sha1(anchor.encode("utf-8")).hexdigest()[:16]
    return f"thread_{digest}"


def parse_mbox_messages(input_path: str | Path) -> list[dict[str, Any]]:
    """Parse a .mbox file into canonical parsed-email rows.

    Returns message-level rows expected by preprocess/dedupe/chunking stages.
    """

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    mbox = mailbox.mbox(path)
    rows: list[dict[str, Any]] = []

    for idx, msg in enumerate(mbox):
        raw_mid = _normalize_message_id(msg.get("Message-ID"))
        if not raw_mid:
            raw_mid = f"mbox_{hashlib.sha1(msg.as_bytes()).hexdigest()[:24]}"

        reply_to = _normalize_message_id(msg.get("In-Reply-To"))
        refs = [_normalize_message_id(r) for r in re.findall(r"<[^>]+>", msg.get("References", ""))]
        refs = [r for r in refs if r]

        subject = _decode_header_value(msg.get("Subject"))
        date_iso = _parse_date_iso(msg.get("Date"))
        from_list = _parse_addr_header(msg.get("From"))
        from_addr = from_list[0] if from_list else {"name": "", "email": ""}

        to_addrs = _parse_addr_header(msg.get("To"))
        cc_addrs = _parse_addr_header(msg.get("Cc"))
        body = _extract_body_text(msg)

        thread_id = _thread_id_for_message(raw_mid, reply_to, refs, subject)

        rows.append(
            {
                "id": raw_mid,
                "thread_id": thread_id,
                "date": date_iso,
                "subject": subject,
                "from": from_addr,
                "to": to_addrs,
                "cc": cc_addrs,
                "body": body,
                "attachments": [],
                "quoted_chain": [],
                "reply_to": reply_to,
                "_source_mbox_file": path.name,
                "_source_index": idx,
            }
        )

    # Ensure reply_to field is always present for downstream thread reconstruction.
    for row in rows:
        row["reply_to"] = row.get("reply_to") or ""

    rows.sort(key=lambda r: (r.get("thread_id") or "", r.get("date") or "", r.get("id") or ""))
    return rows


def parse_mbox_to_jsonl(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Convenience wrapper: parse .mbox and persist canonical JSONL output."""
    rows = parse_mbox_messages(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    thread_counts = Counter(r.get("thread_id") or "" for r in rows)
    return {
        "source": str(input_path),
        "output_path": str(out),
        "messages": len(rows),
        "threads": len(thread_counts),
        "top_threads": thread_counts.most_common(10),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse .mbox file into canonical email JSONL.")
    p.add_argument("--input", required=True, help="Path to .mbox file")
    p.add_argument("--output", default="emails_mbox_parsed.jsonl", help="Output canonical JSONL path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats = parse_mbox_to_jsonl(args.input, args.output)
    print("Mbox parse complete")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
