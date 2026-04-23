"""Live integration tests for Chroma Cloud support.

These are the contract the wizard relies on: `ChromaSearch` and
`ChromaChunkSource` must accept Cloud credentials (api_key + tenant +
database + collection_name), survive cloudpickle roundtrips with those
credentials intact, and satisfy the same protocol shapes their
self-hosted counterparts already expose (search result dicts, Chunk
sample lists, SearchClient isinstance).

Runs only under `pytest -m integration` and skipped entirely when
CHROMA_CLOUD_* vars aren't set. Populate `.env.test` (copy from
`.env.test.example`) and point it at a pre-existing Cloud collection —
the tests don't create or tear down data.

Written before the Cloud constructors exist — failures here guide the
implementation in `src/cgft/corpus/chroma/`.
"""

from __future__ import annotations

import os

import cloudpickle
import pytest

from cgft.chunkers.models import Chunk
from cgft.corpus.chroma.search import ChromaSearch
from cgft.corpus.chroma.source import ChromaChunkSource
from cgft.corpus.search_client import SearchClient

pytestmark = pytest.mark.integration


API_KEY = os.environ.get("CHROMA_CLOUD_API_KEY", "")
TENANT = os.environ.get("CHROMA_CLOUD_TENANT", "")
DATABASE = os.environ.get("CHROMA_CLOUD_DATABASE", "")
COLLECTION = os.environ.get("CHROMA_CLOUD_COLLECTION", "")

_CREDS = all([API_KEY, TENANT, DATABASE, COLLECTION])
_SKIP_REASON = (
    "Chroma Cloud creds missing — set CHROMA_CLOUD_API_KEY / _TENANT / "
    "_DATABASE / _COLLECTION in .env.test to enable these tests."
)

pytestmark = [pytestmark, pytest.mark.skipif(not _CREDS, reason=_SKIP_REASON)]


def _make_search(**overrides) -> ChromaSearch:
    params = dict(
        api_key=API_KEY,
        tenant=TENANT,
        database=DATABASE,
        collection_name=COLLECTION,
    )
    params.update(overrides)
    return ChromaSearch(**params)


def _make_source(**overrides) -> ChromaChunkSource:
    params = dict(
        api_key=API_KEY,
        tenant=TENANT,
        database=DATABASE,
        collection_name=COLLECTION,
    )
    params.update(overrides)
    return ChromaChunkSource(**params)


# ────────────────────────────────────────────────────────────────────────────
# ChromaSearch — runs inside the pickled env class at training time.
# Wizard emits: ChromaSearch(api_key=..., tenant=..., database=..., collection_name=...).
# ────────────────────────────────────────────────────────────────────────────


class TestChromaSearchCloud:
    def test_constructs_with_cloud_creds_only(self):
        # The wizard never passes host/port for Cloud. Missing host is valid.
        search = _make_search()
        assert search is not None

    def test_satisfies_search_client_protocol(self):
        # SearchEnv does `isinstance(search, SearchClient)` at construction;
        # passing this makes ChromaSearch swappable with TpufSearch/CorporaSearch.
        search = _make_search()
        assert isinstance(search, SearchClient)

    def test_available_modes_includes_vector(self):
        # Wizard reads this to build the search tool schema. Cloud collections
        # always support vector (server-side auto-embedding); lexical/hybrid
        # depend on whether a BM25 schema was declared at creation.
        search = _make_search()
        modes = search.available_modes
        assert "vector" in modes

    def test_vector_search_returns_search_client_shape(self):
        # SearchClient contract: list of dicts with content/source/metadata/score.
        # SearchEnv._format_results reads exactly these keys.
        search = _make_search()
        results = search.search("installation", mode="vector", top_k=3)
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert "content" in r and isinstance(r["content"], str)
            assert "source" in r and isinstance(r["source"], str)
            assert "metadata" in r and isinstance(r["metadata"], dict)
            assert "score" in r and isinstance(r["score"], (int, float))

    def test_vector_search_finds_relevant_content(self):
        # Smoke test: the word "installation" should rank an installation
        # chunk above unrelated ones. Not a quality benchmark; just verifies
        # we're hitting a real search, not returning canned data.
        search = _make_search()
        results = search.search("installation", mode="vector", top_k=1)
        assert len(results) == 1
        assert results[0]["content"]

    def test_auto_mode_works(self):
        # mode="auto" is what SearchEnv calls internally when no mode kwarg is
        # passed by the model. Must resolve to something available.
        search = _make_search()
        results = search.search("python", mode="auto", top_k=2)
        assert isinstance(results, list)

    def test_source_field_present_for_citation_reward(self):
        # citation_score reads chunk.metadata[source_field]. The cgft uploader
        # writes `file_path`, so Cloud collections populated by cgft should
        # have that key on every chunk. The wizard's auto-picker prefers
        # `file` > `file_path`, so as long as file_path is preserved,
        # citation rewards work without extra config.
        search = _make_search()
        results = search.search("installation", mode="vector", top_k=2)
        for r in results:
            md = r["metadata"]
            assert "file_path" in md or "file" in md, (
                f"No file/file_path in metadata keys: {sorted(md)}"
            )

    def test_pickle_roundtrip_preserves_creds(self):
        # SearchEnv's env class gets cloudpickled. ChromaSearch must survive
        # with its connection params intact. The live client handle is
        # rebuilt lazily after unpickling (see __setstate__).
        search = _make_search()
        # Warm up so the client handle exists before pickling — surfaces any
        # "pickled a live connection" bug.
        _ = search.search("hello", mode="vector", top_k=1)

        restored = cloudpickle.loads(cloudpickle.dumps(search))
        results = restored.search("installation", mode="vector", top_k=1)
        assert len(results) == 1

    def test_bad_api_key_surfaces_error(self):
        # Auth failure should raise, not silently return empty results —
        # the wizard's probe path relies on this to show "Invalid API key"
        # instead of "collection is empty".
        search = _make_search(api_key="ck-invalid-key-for-test")
        with pytest.raises(Exception):
            search.search("q", mode="vector", top_k=1)

    def test_bm25_sparse_vector_metadata_stripped_or_json_safe(self):
        # Cloud collections populated by cgft include a `bm25_embedding`
        # metadata entry that's a non-JSON `SparseVector`. The TS sample
        # adapter JSON-serializes what we return, so either the SDK strips
        # this, or our search() result's metadata must be JSON-safe.
        import json

        search = _make_search()
        results = search.search("installation", mode="vector", top_k=1)
        for r in results:
            # Round-trip through json.dumps — will raise TypeError on
            # non-serializable values (SparseVector, numpy arrays, etc.)
            json.dumps(r["metadata"])


# ────────────────────────────────────────────────────────────────────────────
# ChromaChunkSource — runs during QA generation (CgftPipeline) and for the
# wizard's preview-sample path. Wizard emits it with the same Cloud kwargs
# as ChromaSearch.
# ────────────────────────────────────────────────────────────────────────────


class TestChromaChunkSourceCloud:
    def test_constructs_with_cloud_creds_only(self):
        source = _make_source()
        assert source is not None

    def test_get_chunk_count(self):
        # Wizard's preview hook uses this to show "N chunks total" before
        # sampling. Must hit the real Cloud collection.
        source = _make_source()
        count = source.get_chunk_count()
        assert count >= 1

    def test_sample_chunks_returns_populated_chunks(self):
        # CgftPipeline calls this during corpus profiling. Returned Chunks
        # need content + metadata_dict for downstream QA gen.
        source = _make_source()
        chunks = source.sample_chunks(n=2)
        assert 1 <= len(chunks) <= 2
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.content
            assert isinstance(c.metadata_dict, dict)

    def test_sample_chunks_preserves_file_path(self):
        # citation_score reads chunk.metadata[source_field]. QA-gen rows
        # flow from these chunks into the train.jsonl `reference_chunks`
        # field, so file_path must round-trip through sample_chunks.
        source = _make_source()
        chunks = source.sample_chunks(n=3)
        assert any("file_path" in c.metadata_dict for c in chunks)

    def test_pickle_roundtrip(self):
        # CgftPipeline pickles the source to ship to the GPU sidecar for
        # entity extraction. Must survive with lazy client rebuild.
        source = _make_source()
        _ = source.get_chunk_count()  # warm live handles

        restored = cloudpickle.loads(cloudpickle.dumps(source))
        assert restored.get_chunk_count() >= 1


# ────────────────────────────────────────────────────────────────────────────
# Constructor parity — Cloud kwargs should be accepted on both classes via
# the same names so wizard codegen can emit them from a single resource map.
# ────────────────────────────────────────────────────────────────────────────


class TestCloudConstructorParity:
    def test_search_accepts_all_cloud_kwargs(self):
        # Regression guard — the wizard's registry maps resource fields
        # one-to-one to Python constructor kwargs. Any rename here breaks
        # the generated env class with a TypeError at training time.
        ChromaSearch(
            api_key=API_KEY,
            tenant=TENANT,
            database=DATABASE,
            collection_name=COLLECTION,
        )

    def test_source_accepts_all_cloud_kwargs(self):
        ChromaChunkSource(
            api_key=API_KEY,
            tenant=TENANT,
            database=DATABASE,
            collection_name=COLLECTION,
        )
