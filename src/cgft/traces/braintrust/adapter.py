"""Braintrust TraceAdapter implementation."""

from __future__ import annotations

import time
from typing import Any

import httpx

from cgft.traces.adapter import (
    NormalizedTrace,
    TraceCredentials,
    TraceProject,
)
from cgft.traces.braintrust.message_extraction import extract_messages, extract_scores

_BASE_URL = "https://api.braintrust.dev/v1"
_MAX_TRACES_PER_FETCH = 1000


class BraintrustTraceAdapter:
    """Adapter for fetching and normalising traces from Braintrust."""

    def connect(self, credentials: TraceCredentials) -> dict[str, Any]:
        """Validate the API key by listing projects."""
        headers = credentials.to_headers()
        resp = httpx.get(f"{_BASE_URL}/project", headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        objects = data.get("objects", []) if isinstance(data, dict) else []
        return {"status": "ok", "projects": len(objects)}

    def list_projects(self, credentials: TraceCredentials) -> list[TraceProject]:
        """List projects visible to the API key."""
        headers = credentials.to_headers()
        resp = httpx.get(f"{_BASE_URL}/project", headers=headers, timeout=15)
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

    def fetch_traces(
        self,
        credentials: TraceCredentials,
        project_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> tuple[list[NormalizedTrace], str | None]:
        """Fetch spans from Braintrust, group by root_span_id, and normalise.

        Paginates automatically using cursor-based pagination until all
        spans are fetched (or *limit* traces are reached).  Uses POST
        ``/v1/project_logs/{project_id}/fetch`` with JSON body.  Retries
        with exponential backoff on 429.

        Returns ``(traces, next_cursor)``.
        """
        headers = {**credentials.to_headers(), "Content-Type": "application/json"}
        max_traces = min(limit, _MAX_TRACES_PER_FETCH) if limit else _MAX_TRACES_PER_FETCH

        all_events: list[dict[str, Any]] = []
        page_cursor = cursor

        while True:
            body: dict[str, Any] = {"limit": 100}
            if page_cursor is not None:
                body["cursor"] = page_cursor

            resp = self._post_with_retry(
                f"{_BASE_URL}/project_logs/{project_id}/fetch",
                headers=headers,
                body=body,
            )
            resp.raise_for_status()
            data = resp.json()

            events = data.get("events", [])
            if not events:
                break

            all_events.extend(events)
            page_cursor = data.get("cursor")

            # Group what we have so far to check trace count
            trace_trees = _group_into_traces(all_events)
            if len(trace_trees) >= max_traces:
                break
            if not page_cursor:
                break

        # Final grouping and normalisation
        trace_trees = _group_into_traces(all_events)

        traces: list[NormalizedTrace] = []
        for trace_id, tree in trace_trees.items():
            traces.append(_normalize_trace(trace_id, tree))
            if len(traces) >= max_traces:
                break

        return traces, page_cursor

    @staticmethod
    def _post_with_retry(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        *,
        max_retries: int = 5,
        timeout: float = 60,
    ) -> httpx.Response:
        """POST with exponential backoff on 429."""
        resp: httpx.Response | None = None
        for attempt in range(max_retries):
            resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
            if resp.status_code == 429:
                wait = 2**attempt
                time.sleep(wait)
                continue
            break
        if resp is None:
            raise RuntimeError(f"No response after {max_retries} retries: {url}")
        return resp


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
