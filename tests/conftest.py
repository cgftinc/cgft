"""Test harness setup: tests/ on sys.path, .env.test loaded for integration runs."""

import os
import sys
from pathlib import Path

_tests_dir = str(Path(__file__).parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

# Load .env.test for integration runs. Kept dependency-free (no python-dotenv)
# because tests/ shouldn't pull extra deps into the default install. Existing
# env vars win — CI / shell exports aren't clobbered by the file.
_env_file = Path(__file__).parent.parent / ".env.test"
if _env_file.is_file():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
