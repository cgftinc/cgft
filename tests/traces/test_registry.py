"""Tests for the trace adapter registry."""

import pytest

from cgft.traces.registry import get_adapter


class TestGetAdapter:
    def test_braintrust_returns_adapter(self):
        adapter = get_adapter("braintrust")
        assert hasattr(adapter, "connect")
        assert hasattr(adapter, "list_projects")
        assert hasattr(adapter, "fetch_traces")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown trace provider"):
            get_adapter("nonexistent")

    def test_error_lists_available_providers(self):
        with pytest.raises(ValueError, match="braintrust"):
            get_adapter("nonexistent")
