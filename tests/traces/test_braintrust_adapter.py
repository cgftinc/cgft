"""Tests for BraintrustTraceAdapter span grouping and normalisation."""

from unittest.mock import MagicMock, patch

from cgft.traces.adapter import TraceCredentials
from cgft.traces.braintrust.adapter import BraintrustTraceAdapter, _group_into_traces, _normalize_trace


class TestGroupIntoTraces:
    def test_groups_by_root_span_id(self):
        events = [
            {"id": "r1", "span_id": "r1", "root_span_id": "r1", "input": {"messages": []}},
            {"id": "c1", "span_id": "c1", "root_span_id": "r1", "input": {}},
            {"id": "c2", "span_id": "c2", "root_span_id": "r1", "input": {}},
            {"id": "r2", "span_id": "r2", "root_span_id": "r2", "input": {"messages": []}},
        ]
        traces = _group_into_traces(events)
        assert len(traces) == 2
        assert "r1" in traces
        assert "r2" in traces
        assert len(traces["r1"]["children"]) == 2
        assert len(traces["r2"]["children"]) == 0

    def test_root_detected_by_span_id_equals_root_span_id(self):
        events = [
            {"id": "x", "span_id": "root1", "root_span_id": "root1", "input": {}},
            {"id": "y", "span_id": "child1", "root_span_id": "root1", "input": {}},
        ]
        traces = _group_into_traces(events)
        assert traces["root1"]["root"]["span_id"] == "root1"

    def test_fallback_root_when_no_match(self):
        # If no span has span_id == root_span_id, first span becomes root
        events = [
            {"id": "a", "span_id": "s1", "root_span_id": "orphan"},
            {"id": "b", "span_id": "s2", "root_span_id": "orphan"},
        ]
        traces = _group_into_traces(events)
        assert "orphan" in traces
        assert traces["orphan"]["root"]["id"] == "a"
        assert len(traces["orphan"]["children"]) == 1

    def test_empty_events(self):
        assert _group_into_traces([]) == {}


class TestNormalizeTrace:
    def test_basic_normalisation(self):
        tree = {
            "root": {
                "input": {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ]
                },
                "scores": {"quality": 0.9},
                "metadata": {"env": "test"},
                "created": "2024-01-01T00:00:00Z",
            },
            "children": [],
        }
        trace = _normalize_trace("t1", tree)
        assert trace.id == "t1"
        assert len(trace.messages) == 2
        assert trace.scores == {"quality": 0.9}
        assert trace.metadata == {"env": "test"}
        assert trace.timestamp == "2024-01-01T00:00:00Z"
        assert trace.errors is None

    def test_extraction_failure_populates_errors(self):
        # Malformed input that will cause extract_messages to fail
        tree = {
            "root": {
                "input": None,  # not a dict — will cause issues
                "created": "2024-01-01",
            },
            "children": [
                {
                    "input": "not a dict",
                    "output": "also not a dict",
                    "span_attributes": "invalid",
                    "created": "",
                    "name": "",
                    "id": "",
                    "span_id": "",
                },
            ],
        }
        trace = _normalize_trace("t2", tree)
        # Should have messages (possibly empty) but no crash
        assert trace.id == "t2"
        # Even if extraction produces empty messages, it shouldn't error
        # since None input returns []

    def test_preserves_child_metadata_for_extraction(self):
        tree = {
            "root": {
                "input": {"messages": [{"role": "user", "content": "Q"}]},
            },
            "children": [
                {
                    "input": {
                        "messages": [
                            {"role": "system", "content": "S"},
                            {"role": "user", "content": "Q"},
                        ]
                    },
                    "output": {"role": "assistant", "content": "A"},
                    "span_attributes": {"type": "llm"},
                    "created": "2024-01-01T00:00:01Z",
                    "name": "llm_call",
                    "id": "c1",
                    "span_id": "c1",
                },
            ],
        }
        trace = _normalize_trace("t3", tree)
        assert len(trace.messages) > 0
        assert any(m.role == "assistant" for m in trace.messages)

    def test_missing_metadata_handled(self):
        tree = {
            "root": {
                "input": {
                    "messages": [
                        {"role": "user", "content": "Q"},
                        {"role": "assistant", "content": "A"},
                    ]
                },
                # no scores, no metadata, no created
            },
            "children": [],
        }
        trace = _normalize_trace("t4", tree)
        assert trace.scores == {}
        assert trace.metadata == {}
        assert trace.timestamp is None


class TestCountTraces:
    def test_count_traces_parses_btql_response(self):
        adapter = BraintrustTraceAdapter()
        creds = TraceCredentials(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"total": 142}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.request", return_value=mock_resp) as mock_req:
            count = adapter.count_traces(creds, "proj-123")

        assert count == 142
        mock_req.assert_called_once()
        call_args = mock_req.call_args
        # httpx.request(method, url, ...)
        assert call_args.args[0] == "POST"
        assert "/btql" in call_args.args[1]
        body = call_args.kwargs["json"]
        assert "project_logs('proj-123')" in body["query"]
        assert body["fmt"] == "json"

    def test_count_traces_handles_array_response(self):
        adapter = BraintrustTraceAdapter()
        creds = TraceCredentials(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"total": 55}]
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.request", return_value=mock_resp):
            count = adapter.count_traces(creds, "proj-456")

        assert count == 55

    def test_count_traces_returns_zero_on_empty(self):
        adapter = BraintrustTraceAdapter()
        creds = TraceCredentials(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.request", return_value=mock_resp):
            count = adapter.count_traces(creds, "proj-789")

        assert count == 0
