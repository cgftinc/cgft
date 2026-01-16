"""Tests for the markdown chunker."""

from synthetic_data_prep import Chunk, MarkdownChunker


class TestChunk:
    """Tests for the Chunk model."""

    def test_chunk_length(self):
        chunk = Chunk(content="hello world", metadata={})
        assert len(chunk) == 11

    def test_chunk_to_dict(self):
        chunk = Chunk(content="test", metadata={"h1": "Title"})
        result = chunk.to_dict()
        assert result == {"content": "test", "metadata": {"h1": "Title"}}


class TestMarkdownChunker:
    """Tests for the MarkdownChunker."""

    def test_empty_content(self):
        chunker = MarkdownChunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_whitespace_only(self):
        chunker = MarkdownChunker()
        chunks = chunker.chunk("   \n\n   ")
        assert chunks == []

    def test_content_without_headers(self):
        chunker = MarkdownChunker(min_char=10, max_char=500, chunk_overlap=50)
        content = "This is plain text without any headers."
        chunks = chunker.chunk(content)
        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].metadata["index"] == 0

    def test_simple_header_split(self):
        chunker = MarkdownChunker(min_char=10, max_char=1000)
        content = """# Title

First section content.

## Subtitle

Second section content."""

        chunks = chunker.chunk(content)
        assert len(chunks) >= 1
        assert any("h1" in c.metadata for c in chunks)

    def test_section_fusion(self):
        """Short sections should be fused together."""
        chunker = MarkdownChunker(min_char=100, max_char=500)
        content = """# A

Short.

# B

Also short.

# C

Still short."""

        chunks = chunker.chunk(content)
        # With min_char=100, these tiny sections should get fused
        assert len(chunks) < 3

    def test_large_section_split(self):
        """Large sections should be split."""
        chunker = MarkdownChunker(min_char=10, max_char=100, chunk_overlap=10)
        long_content = "# Title\n\n" + "word " * 100  # ~500 chars

        chunks = chunker.chunk(long_content)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_metadata_preservation(self):
        """Header metadata should be preserved in chunks."""
        chunker = MarkdownChunker(min_char=10, max_char=1000)
        content = """# Main Title

## Section One

Content under section one.

### Subsection

More detailed content."""

        chunks = chunker.chunk(content)
        # Check that header metadata exists
        all_metadata_keys = set()
        for chunk in chunks:
            all_metadata_keys.update(chunk.metadata.keys())

        assert "index" in all_metadata_keys

    def test_chunk_indices_are_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        chunker = MarkdownChunker(min_char=10, max_char=1000)
        content = """# One

First.

# Two

Second.

# Three

Third."""

        chunks = chunker.chunk(content)
        indices = [c.metadata["index"] for c in chunks]
        assert indices == list(range(len(chunks)))
