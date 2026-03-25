"""Protocol shape conformance tests for ChunkSource."""

from __future__ import annotations

from cgft.chunkers.models import Chunk
from cgft.corpus.search_schema.search_types import SearchCapabilities, SearchSpec
from cgft.corpus.source import ChunkSource

# ---------------------------------------------------------------------------
# Minimal stub — returns canned data, implements protocol structurally
# ---------------------------------------------------------------------------


class StubChunkSource:
    """Minimal ChunkSource for shape testing. No real logic."""

    _CHUNK = Chunk(content="stub chunk", metadata=(("file", "stub.md"), ("index", 0)))

    def populate_from_folder(
        self,
        docs_path: str,
        min_chars: int = 1024,
        max_chars: int = 2048,
        overlap_chars: int = 128,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        pass

    def populate_from_chunks(self, collection, batch_size=100, show_summary=True):
        pass

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        return [self._CHUNK] * min(n, 1)

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        return {
            "chunk_content": chunk.chunk_str(),
            "prev_chunk_preview": "(No previous chunk)",
            "next_chunk_preview": "(No next chunk)",
        }

    def get_top_level_chunks(self) -> list[Chunk]:
        return [self._CHUNK]

    def search_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode=None,
        hybrid=None,
    ) -> list[dict]:
        return [
            {
                "chunk": self._CHUNK,
                "queries": queries[:1],
                "same_file": False,
                "max_score": 0.5,
            }
        ] if queries else []

    def search(self, spec: SearchSpec) -> list[Chunk]:
        return [self._CHUNK]

    def search_content(self, spec: SearchSpec) -> list[str]:
        return ["stub content"]

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter=None,
    ) -> list[Chunk]:
        return [self._CHUNK]

    def get_search_capabilities(self) -> SearchCapabilities:
        return {
            "backend": "stub",
            "modes": {"lexical"},
            "filter_ops": {"field": {"eq"}, "logical": set()},
            "ranking": {"bm25"},
            "constraints": {},
            "graph_expansion": False,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


_stub = StubChunkSource()


class TestProtocolConformance:
    def test_isinstance_check(self):
        assert isinstance(_stub, ChunkSource)

    def test_capabilities_keys(self):
        caps = _stub.get_search_capabilities()
        assert "backend" in caps
        assert "modes" in caps
        assert "filter_ops" in caps
        assert "ranking" in caps
        assert "constraints" in caps
        assert "graph_expansion" in caps

    def test_search_related_shape(self):
        result = _stub.search_related(
            StubChunkSource._CHUNK, ["test query"], top_k=3
        )
        assert isinstance(result, list)
        for item in result:
            assert "chunk" in item
            assert "queries" in item
            assert "same_file" in item
            assert "max_score" in item
            assert isinstance(item["chunk"], Chunk)

    def test_get_chunk_with_context_shape(self):
        ctx = _stub.get_chunk_with_context(StubChunkSource._CHUNK)
        assert "chunk_content" in ctx
        assert "prev_chunk_preview" in ctx
        assert "next_chunk_preview" in ctx

    def test_sample_chunks_returns_chunks(self):
        result = _stub.sample_chunks(1)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_search_returns_chunks(self):
        spec = SearchSpec(mode="lexical", top_k=5, text_query="test")
        result = _stub.search(spec)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_search_text_returns_chunks(self):
        result = _stub.search_text("test", top_k=5)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_get_top_level_returns_chunks(self):
        result = _stub.get_top_level_chunks()
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)
