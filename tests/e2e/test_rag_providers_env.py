"""End-to-end env validation against live RAG providers.

Mirrors the proven notebook pattern — `env_class=SearchEnv` directly,
`env_args` carries the SearchClient instance + judge config, and
`local_modules=[cgft]` inlines cgft into the env class pickle. Calls
`train(dry_run=True)` to go through dataset upload + env bundling +
remote rollout-server validation, stopping before SkyPilot launch.

Run via `pytest -m e2e`. Skipped when creds are missing.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

import cgft
from cgft.envs.search_env import SearchEnv
from cgft.trainer.pipeline import train

pytestmark = pytest.mark.e2e


# ── Shared platform creds ────────────────────────────────────────────────

CGFT_API_KEY = os.environ.get("CGFT_API_KEY", "")
CGFT_BASE_URL = os.environ.get("CGFT_BASE_URL", "https://app.cgft.io")
LLM_BASE_URL = "https://llm.cgft.io/v1"
JUDGE_MODEL = "gpt-5.4-nano"  # same as cgft's _VALIDATION_MODEL


def _step(backend: str, message: str) -> None:
    """Flushed progress marker — visible under `pytest -s`."""
    print(f"  [{backend}] → {message}", flush=True)
    sys.stdout.flush()


def _make_dummy_qa(n: int = 16) -> list[dict]:
    """train() requires >= 16 train rows. Remote validation uses only the
    first 2 eval rows — the rest is a gate satisfier."""
    return [
        {
            "question": f"Stub question {i} — how do I get started?",
            "answer": f"Stub answer {i}",
            "reference_chunks": [
                {
                    "content": "Install with pip, then import the library.",
                    "metadata": {"file": f"stub_{i}.md", "file_path": f"stub_{i}.md"},
                }
            ],
        }
        for i in range(n)
    ]


def _run_train(backend: str, search_client, pip_deps: list[str]) -> dict:
    """Mirror the coworker's proven-working shape: SearchEnv + search in env_args."""
    t0 = time.monotonic()
    _step(backend, "building env_args")
    env_args = {
        "search": search_client,
        "judge_base_url": LLM_BASE_URL,
        "judge_api_key": CGFT_API_KEY,
        "judge_model": JUDGE_MODEL,
        "corpus_description": "E2E test corpus",
    }

    _step(backend, f"calling train(dry_run=True) pip_deps={pip_deps}")
    result = train(
        env_class=SearchEnv,
        env_args=env_args,
        train_dataset=_make_dummy_qa(16),
        eval_dataset=_make_dummy_qa(16)[:4],
        prefix=f"e2e-{backend}",
        api_key=CGFT_API_KEY,
        base_url=CGFT_BASE_URL,
        local_modules=[cgft],
        experiment_name=f"e2e-{backend}-{int(time.time())}",
        pip_dependencies=pip_deps,
        validate_env=False,  # skip local isolated venv — can't pip install cgft
        validate_env_remotely=True,
        validation_model=JUDGE_MODEL,
        dry_run=True,
        show_summary=True,
    )
    _step(backend, f"status={result.get('status')!r}  ({time.monotonic() - t0:.1f}s)")
    return result


def _assert_validated(result: dict) -> None:
    assert isinstance(result, dict)
    assert result.get("status") == "validated", f"Expected 'validated', got: {result}"


def _require(var: str) -> str:
    v = os.environ.get(var, "")
    if not v:
        pytest.skip(f"{var} missing — set in .env.test")
    return v


# ── Chroma Cloud ─────────────────────────────────────────────────────────


class TestChromaCloudEnvValidation:
    def test_dry_run_validates(self):
        from cgft.corpus.chroma.search import ChromaSearch

        search = ChromaSearch(
            api_key=_require("CHROMA_CLOUD_API_KEY"),
            tenant=_require("CHROMA_CLOUD_TENANT"),
            database=_require("CHROMA_CLOUD_DATABASE"),
            collection_name=_require("CHROMA_CLOUD_COLLECTION"),
        )
        result = _run_train(
            "chroma-cloud",
            search,
            pip_deps=["chromadb>=1.0.0", "snowballstemmer>=2.2.0", "openai"],
        )
        _assert_validated(result)


# ── Turbopuffer ──────────────────────────────────────────────────────────


class TestTurbopufferEnvValidation:
    def test_dry_run_validates(self):
        from cgft.corpus.turbopuffer.search import TpufSearch

        search = TpufSearch(
            api_key=_require("TPUF_API_KEY"),
            namespace=_require("TPUF_NAMESPACE"),
            region=os.environ.get("TPUF_REGION", "aws-us-east-1"),
        )
        result = _run_train(
            "turbopuffer",
            search,
            pip_deps=["turbopuffer", "openai"],
        )
        _assert_validated(result)


# ── Pinecone ─────────────────────────────────────────────────────────────


class TestPineconeEnvValidation:
    def test_dry_run_validates(self):
        from cgft.corpus.pinecone.search import PineconeSearch

        search = PineconeSearch(
            api_key=_require("PINECONE_API_KEY"),
            index_name=_require("PINECONE_INDEX_NAME"),
        )
        result = _run_train(
            "pinecone",
            search,
            pip_deps=["pinecone>=5.0.0", "openai"],
        )
        _assert_validated(result)


# ── CGFT Corpora (BM25) ──────────────────────────────────────────────────


class TestCorporaEnvValidation:
    def test_dry_run_validates(self):
        from cgft.corpus.corpora.search import CorporaSearch

        search = CorporaSearch(
            api_key=_require("CGFT_API_KEY"),
            corpus_name=_require("CGFT_CORPUS_NAME"),
            base_url=CGFT_BASE_URL,
        )
        result = _run_train(
            "cgft-corpora",
            search,
            pip_deps=["openai"],
        )
        _assert_validated(result)
