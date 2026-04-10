"""Root conftest — loads .env.test for integration test credentials."""

from __future__ import annotations

import os
from pathlib import Path


def pytest_configure(config):
    """Load .env.test into os.environ before test collection."""
    env_file = Path(__file__).parent / ".env.test"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value:
            os.environ.setdefault(key, value)
