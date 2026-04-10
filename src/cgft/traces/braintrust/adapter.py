"""Braintrust TraceAdapter implementation."""

from __future__ import annotations

import logging
from typing import Any

from cgft.traces.adapter import (
    NormalizedTrace,
    TraceCredentials,
    TraceProject,
)
from cgft.traces.braintrust.message_extraction import extract_messages, extract_scores
from cgft.traces.http import request_with_retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.braintrust.dev/v1"
_BTQL_URL = "https://api.braintrust.dev/btql"
_MAX_TRACES_PER_FETCH = 1000
_BTQL_COLUMNS = (
    "id, span_id, root_span_id, input, output, scores, "
    "metadata, span_attributes, created, name"
)


class BraintrustTraceAdapter:
    """Adapter for fetching and normalising traces from Braintrust."""

    def connect(self, credentials: TraceCredentials) -> dict[str, Any]:
        """Validate the API key by listing projects."""
        headers = credentials.to_headers()
        resp = request_with_retry(
            "GET", f"{_BASE_URL}/project", headers=headers, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        objects = data.get("objects", []) if isinstance(data, dict) else []
        return {"status": "ok", "projects": len(objects)}

    def list_projects(self, credentials: TraceCredentials) -> list[TraceProject]:
        """List projects visible to the API key."""
        headers = credentials.to_headers()
        resp = request_with_retry(
            "GET", f"{_BASE_URL}/project", headers=headers, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        objects = data.get("objects", data) if isinstance(data, dict) else data
        if not isinstance(objects, list):
            return []
        return [
            TraceProject(id=str(p["id"]), name=str(p.get("name", p["id"])))
            for p in objects
            if isinstance(p, dict) and "id" in p
        ]

    def count_traces(
        self,
        credentials: TraceCredentials,
        project_id: str,
    ) -> int:
        """Return total root-span count for a project using BTQL."""
        headers = {**credentials.to_headers(), "Content-Type": "application/json"}
        query = (
            f"SELECT count(*) AS total FROM project_logs('{project_id}') "
            f"WHERE span_id = root_span_id"
        )
        resp = request_with_retry(
            "POST",
            _BTQL_URL,
            headers=headers,
            json={"query": query, "fmt": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        # BTQL returns { data: [...] } or just an array
        rows = data.get("data", data) if isinstance(data, dict) else data
        if rows and isinstance(rows, list) and isinstance(rows[0], dict):
            return int(rows[0].get("total", 0))
        return 0

    def fetch_traces(
        self,
        credentials: TraceCredentials,
        project_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> tuple[list[NormalizedTrace], str | None]:
        """Fetch spans from Braintrust, group by root_span_id, and normalise.

        Uses BTQL (higher rate limits, larger page sizes) with automatic
        fallback to the REST API if BTQL fails.

        Returns ``(traces, next_cursor)``.
        """
        max_traces = min(limit, _MAX_TRACES_PER_FETCH) if limit else _MAX_TRACES_PER_FETCH

        try:
            events = self._fetch_via_btql(credentials, project_id, max_traces)
        except Exception as e:
            logger.warning("BTQL fetch failed, falling back to REST: %s", e)
            events, cursor = self._fetch_via_rest(
                credentials, project_id, max_traces, cursor
            )

        trace_trees = _group_into_traces(events)

        traces: list[NormalizedTrace] = []
        for trace_id, tree in trace_trees.items():
            traces.append(_normalize_trace(trace_id, tree))
            if len(traces) >= max_traces:
                break

        return traces, cursor

    def _fetch_via_btql(
        self,
        credentials: TraceCredentials,
        project_id: str,
        max_traces: int,
    ) -> list[dict[str, Any]]:
        """Fetch spans using BTQL. Higher rate limits, 1000 rows/page."""
        headers = {**credentials.to_headers(), "Content-Type": "application/json"}
        all_events: list[dict[str, Any]] = []
        offset = 0
        # Each trace has ~10-20 child spans, so fetch 20x the desired trace count
        span_limit = max_traces * 20

        while True:
            page_size = min(1000, span_limit - offset)
            if page_size <= 0:
                break

            query = (
                f"SELECT {_BTQL_COLUMNS} "
                f"FROM project_logs('{project_id}') "
                f"ORDER BY created DESC "
                f"LIMIT {page_size} OFFSET {offset}"
            )
            resp = request_with_retry(
                "POST",
                _BTQL_URL,
                headers=headers,
                json={"query": query, "fmt": "json"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            rows = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(rows, list) or not rows:
                break

            all_events.extend(rows)
            offset += len(rows)

            # Check if we have enough traces
            trace_trees = _group_into_traces(all_events)
            if len(trace_trees) >= max_traces:
                break

            # If we got fewer rows than requested, no more data
            if len(rows) < page_size:
                break

        logger.info(
            "BTQL fetch: %d spans, %d traces",
            len(all_events),
            len(_group_into_traces(all_events)),
        )
        return all_events

    def _fetch_via_rest(
        self,
        credentials: TraceCredentials,
        project_id: str,
        max_traces: int,
        cursor: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch spans using REST API. Fallback for BTQL failures."""
        headers = {**credentials.to_headers(), "Content-Type": "application/json"}
        all_events: list[dict[str, Any]] = []
        page_cursor = cursor

        while True:
            body: dict[str, Any] = {"limit": 100}
            if page_cursor is not None:
                body["cursor"] = page_cursor

            resp = request_with_retry(
                "POST",
                f"{_BASE_URL}/project_logs/{project_id}/fetch",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

            events = data.get("events", [])
            if not events:
                break

            all_events.extend(events)
            page_cursor = data.get("cursor")

            trace_trees = _group_into_traces(all_events)
            if len(trace_trees) >= max_traces:
                break
            if not page_cursor:
                break

        return all_events, page_cursor


def _group_into_traces(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Group raw span events by ``root_span_id`` into trace trees.

    Returns ``{trace_id: {"root": root_span, "children": [child_spans]}}``.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        root_id = event.get("root_span_id", event.get("id", ""))
        groups.setdefault(root_id, []).append(event)

    traces: dict[str, dict[str, Any]] = {}
    for trace_id, spans in groups.items():
        root = None
        children: list[dict[str, Any]] = []
        for span in spans:
            if span.get("span_id") == span.get("root_span_id") or span.get("id") == trace_id:
                root = span
            else:
                children.append(span)
        if root is None and spans:
            root = spans[0]
            children = spans[1:]
        if root is not None:
            traces[trace_id] = {"root": root, "children": children}

    return traces


def _normalize_trace(trace_id: str, tree: dict[str, Any]) -> NormalizedTrace:
    """Convert a grouped trace tree into a ``NormalizedTrace``."""
    root = tree["root"]
    children = tree.get("children", [])

    trace_dict = {
        "input": root.get("input"),
        "output": root.get("output"),
        "scores": root.get("scores"),
        "children": [
            {
                "input": c.get("input"),
                "output": c.get("output"),
                "span_attributes": c.get("span_attributes"),
                "created": c.get("created", ""),
                "name": c.get("name", ""),
                "id": c.get("id", ""),
                "span_id": c.get("span_id", ""),
            }
            for c in children
        ],
    }

    errors: list[str] = []
    try:
        messages = extract_messages(trace_dict)
    except Exception as e:
        messages = []
        errors.append(f"Message extraction failed: {e}")

    scores = extract_scores(trace_dict)

    metadata: dict[str, Any] = {}
    if root.get("metadata"):
        metadata = dict(root["metadata"]) if isinstance(root["metadata"], dict) else {}

    return NormalizedTrace(
        id=trace_id,
        messages=messages,
        scores=scores,
        metadata=metadata,
        timestamp=root.get("created"),
        errors=errors if errors else None,
    )
