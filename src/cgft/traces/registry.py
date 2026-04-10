"""Adapter registry — maps provider names to TraceAdapter implementations."""

from __future__ import annotations

from cgft.traces.adapter import TraceAdapter

_ADAPTERS: dict[str, type[TraceAdapter]] = {}


def _register_builtins() -> None:
    from cgft.traces.braintrust.adapter import BraintrustTraceAdapter

    _ADAPTERS["braintrust"] = BraintrustTraceAdapter


def get_adapter(provider: str) -> TraceAdapter:
    """Return a fresh adapter instance for *provider*.

    Raises ``ValueError`` for unknown providers.
    """
    if not _ADAPTERS:
        _register_builtins()
    cls = _ADAPTERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown trace provider: {provider!r}. Available: {sorted(_ADAPTERS)}")
    return cls()
