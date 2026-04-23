"""End-to-end QA generation against live RAG providers.

Smoke test for the ``CgftPipeline`` flow each provider's ChunkSource
supports, tuned for minimum cost + time:

  * 1 sample per run
  * multi-hop only (exercises the MetadataChunkLinker's pairing logic)
  * 2-hop distribution (minimum for multi-hop)
  * No filters, no refinement, no dedup
  * gpt-5.4-nano for generation

Costs ~1-2 LLM calls per test (~$0.001), runs in ~10-30 seconds each.

Run via ``pytest -m e2e``. Skipped when backend creds are missing.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from cgft.qa_generation.cgft_models import (
    CgftPipelineConfig,
    CorpusConfig,
    CorpusContextConfig,
    DedupConfig,
    FilteringConfig,
    GenerationConfig,
    LLMDirectGenerationConfig,
    MicroBatchConfig,
    OutputConfig,
    PlatformConfig,
    RefinementConfig,
    TargetsConfig,
)
from cgft.qa_generation.cgft_pipeline import CgftPipeline

pytestmark = pytest.mark.e2e


CGFT_API_KEY = os.environ.get("CGFT_API_KEY", "")
CGFT_BASE_URL = os.environ.get("CGFT_BASE_URL", "https://app.cgft.io")
LLM_BASE_URL = "https://llm.cgft.io/v1"
TEST_LLM_MODEL = "gpt-5.4-nano"


def _step(backend: str, message: str) -> None:
    print(f"  [{backend}] → {message}", flush=True)
    sys.stdout.flush()


def _require(var: str) -> str:
    v = os.environ.get(var, "")
    if not v:
        pytest.skip(f"{var} missing — set in .env.test")
    return v


def _minimum_config(corpus_name: str, output_dir: Path) -> CgftPipelineConfig:
    """Smallest viable CgftPipelineConfig — 1 multi-hop, no retries."""
    return CgftPipelineConfig(
        platform=PlatformConfig(
            api_key=CGFT_API_KEY,
            base_url=CGFT_BASE_URL,
            llm_api_key=CGFT_API_KEY,
            llm_base_url=LLM_BASE_URL,
        ),
        corpus=CorpusConfig(corpus_name=corpus_name),
        corpus_context=CorpusContextConfig(
            description="Test corpus for e2e QA generation",
            example_queries=["how do I get started"],
            num_top_level_samples=0,
            generate_entity_patterns=False,
        ),
        targets=TargetsConfig(
            total_samples=1,
            primary_type_distribution={"multi_hop": 1.0},
            hop_distribution={2: 1.0},
        ),
        generation=GenerationConfig(
            mode="llm_direct",
            llm_direct=LLMDirectGenerationConfig(
                model=TEST_LLM_MODEL,
                api_key=CGFT_API_KEY,
                base_url=LLM_BASE_URL,
            ),
        ),
        filtering=FilteringConfig(filters=[]),
        refinement=RefinementConfig(enabled=False),
        dedup=DedupConfig(enabled=False),
        micro_batch=MicroBatchConfig(
            batch_size=1,
            max_iterations=1,
            resume=False,
            keep_checkpoints=False,
            checkpoint_dir=str(output_dir / ".checkpoints"),
        ),
        output=OutputConfig(dir=str(output_dir)),
    )


def _run_pipeline(backend: str, source, corpus_name: str, tmp_path: Path) -> list[dict]:
    if not CGFT_API_KEY:
        pytest.skip("CGFT_API_KEY missing — QA gen requires platform + LLM access")

    t0 = time.monotonic()
    _step(backend, "fetching chunk count to warm source")
    try:
        _step(backend, f"chunk count: {source.get_chunk_count()}")
    except Exception as e:
        _step(backend, f"get_chunk_count failed (non-fatal): {e}")

    _step(backend, "CgftPipeline.run() — generating 1 QA pair")
    cfg = _minimum_config(corpus_name=corpus_name, output_dir=tmp_path)
    pipeline = CgftPipeline(cfg, source_factory=lambda _: source)
    result = pipeline.run()

    rows = list(result.get("filtered_dataset", []))
    _step(backend, f"pipeline returned {len(rows)} row(s)  ({time.monotonic() - t0:.1f}s)")
    return rows


def _assert_row_shape(row: dict) -> None:
    assert row.get("question"), f"Missing question: {row}"
    assert row.get("answer"), f"Missing answer: {row}"
    assert len(row.get("reference_chunks", [])) >= 1, (
        f"Expected ≥1 reference_chunk, got: {row.get('reference_chunks')}"
    )


# ── Chroma Cloud ─────────────────────────────────────────────────────────


class TestChromaCloudQaGen:
    def test_generates_one_multihop_question(self, tmp_path):
        from cgft.corpus.chroma.source import ChromaChunkSource

        source = ChromaChunkSource(
            api_key=_require("CHROMA_CLOUD_API_KEY"),
            tenant=_require("CHROMA_CLOUD_TENANT"),
            database=_require("CHROMA_CLOUD_DATABASE"),
            collection_name=_require("CHROMA_CLOUD_COLLECTION"),
        )
        rows = _run_pipeline("chroma-cloud", source, corpus_name="chroma-cloud", tmp_path=tmp_path)
        assert len(rows) >= 1
        _assert_row_shape(rows[0])


# ── Turbopuffer ──────────────────────────────────────────────────────────


class TestTurbopufferQaGen:
    def test_generates_one_multihop_question(self, tmp_path):
        from cgft.corpus.turbopuffer.source import TpufChunkSource

        source = TpufChunkSource(
            api_key=_require("TPUF_API_KEY"),
            namespace=_require("TPUF_NAMESPACE"),
            region=os.environ.get("TPUF_REGION", "aws-us-east-1"),
        )
        rows = _run_pipeline("turbopuffer", source, corpus_name="turbopuffer", tmp_path=tmp_path)
        assert len(rows) >= 1
        _assert_row_shape(rows[0])


# ── Pinecone ─────────────────────────────────────────────────────────────


class TestPineconeQaGen:
    def test_generates_one_multihop_question(self, tmp_path):
        from cgft.corpus.pinecone.source import PineconeChunkSource

        source = PineconeChunkSource(
            api_key=_require("PINECONE_API_KEY"),
            index_name=_require("PINECONE_INDEX_NAME"),
        )
        rows = _run_pipeline("pinecone", source, corpus_name="pinecone", tmp_path=tmp_path)
        assert len(rows) >= 1
        _assert_row_shape(rows[0])


# ── CGFT Corpora (BM25) ──────────────────────────────────────────────────


class TestCorporaQaGen:
    def test_generates_one_multihop_question(self, tmp_path):
        from cgft.corpus.corpora.source import CorporaChunkSource

        source = CorporaChunkSource(
            api_key=_require("CGFT_API_KEY"),
            corpus_name=_require("CGFT_CORPUS_NAME"),
            base_url=CGFT_BASE_URL,
        )
        source.populate_from_existing_corpus_name(os.environ["CGFT_CORPUS_NAME"])
        rows = _run_pipeline("cgft-corpora", source, corpus_name="cgft-corpora", tmp_path=tmp_path)
        assert len(rows) >= 1
        _assert_row_shape(rows[0])
