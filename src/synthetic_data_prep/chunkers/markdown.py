"""Markdown document chunker with header-aware splitting."""

from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from synthetic_data_prep.models import Chunk


class MarkdownChunker:
    """Chunker for markdown documents with header-aware splitting.

    This chunker implements a 3-stage pipeline:
    1. Split by markdown headers (H1, H2, H3), preserving hierarchy in metadata
    2. Fuse adjacent short sections to avoid over-fragmentation
    3. Split remaining large sections using recursive character splitting

    Example:
        >>> chunker = MarkdownChunker(min_char=1024, max_char=2048)
        >>> chunks = chunker.chunk(markdown_text)
        >>> for chunk in chunks:
        ...     print(chunk.content, chunk.metadata)
    """

    def __init__(
        self,
        min_char: int = 1024,
        max_char: int = 2048,
        chunk_overlap: int = 128,
    ) -> None:
        """Initialize the markdown chunker.

        Args:
            min_char: Minimum characters per chunk. Sections shorter than this
                will be fused with adjacent sections.
            max_char: Maximum characters per chunk. Sections longer than this
                will be split using recursive character splitting.
            chunk_overlap: Number of overlapping characters between chunks
                when splitting large sections.
        """
        self.min_char = min_char
        self.max_char = max_char
        self.chunk_overlap = chunk_overlap

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_char,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, content: str) -> list[Chunk]:
        """Chunk a markdown document.

        Args:
            content: Raw markdown text.

        Returns:
            List of Chunk objects with content and metadata.
            Metadata includes header hierarchy (h1, h2, h3) and index.
        """
        sections = self._split_by_headers(content)

        if not sections:
            if content.strip():
                sections = [Chunk(content=content, metadata={})]
            else:
                return []

        fused = self._fuse_short_sections(sections)
        chunks = self._split_large_sections(fused)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["index"] = idx

        return chunks

    def chunk_file(self, file_path: str | Path) -> list[Chunk]:
        """Chunk a markdown file.

        Args:
            file_path: Path to the markdown file.

        Returns:
            List of Chunk objects with 'file' added to metadata.
        """
        file_path = Path(file_path)
        content = file_path.read_text(encoding="utf-8")
        chunks = self.chunk(content)

        for chunk in chunks:
            chunk.metadata["file"] = str(file_path.name)

        return chunks

    def chunk_folder(
        self,
        folder_path: str | Path,
        file_extension: str = ".md",
    ) -> list[Chunk]:
        """Chunk all markdown files in a folder recursively.

        Args:
            folder_path: Path to the folder to process.
            file_extension: File extension to filter (default: .md).

        Returns:
            List of all chunks from all files, with relative 'file' path
            in metadata.
        """
        folder_path = Path(folder_path).resolve()
        all_chunks: list[Chunk] = []

        files = list(folder_path.rglob(f"*{file_extension}"))

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                relative_path = str(file_path.relative_to(folder_path))
                chunks = self.chunk(content)

                for chunk in chunks:
                    chunk.metadata["file"] = relative_path

                all_chunks.extend(chunks)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return all_chunks

    def _split_by_headers(self, content: str) -> list[Chunk]:
        """Split content by markdown headers."""
        splits = self._header_splitter.split_text(content)
        return [
            Chunk(content=doc.page_content, metadata=dict(doc.metadata))
            for doc in splits
        ]

    def _fuse_short_sections(self, sections: list[Chunk]) -> list[Chunk]:
        """Fuse adjacent short sections until they reach min_char threshold."""
        if not sections:
            return []

        def single_pass(secs: list[Chunk]) -> list[Chunk]:
            if not secs:
                return []

            fused: list[Chunk] = []
            current = Chunk(
                content=secs[0].content,
                metadata=secs[0].metadata.copy(),
            )

            for i in range(1, len(secs)):
                next_section = secs[i]
                combined_length = len(current) + len(next_section) + 2

                if len(current) < self.min_char and combined_length <= self.max_char:
                    current.content = current.content + "\n\n" + next_section.content
                    for key, value in next_section.metadata.items():
                        if key not in current.metadata:
                            current.metadata[key] = value
                else:
                    fused.append(current)
                    current = Chunk(
                        content=next_section.content,
                        metadata=next_section.metadata.copy(),
                    )

            fused.append(current)
            return fused

        result = sections
        while True:
            fused = single_pass(result)
            if len(fused) == len(result):
                break
            result = fused

        return result

    def _split_large_sections(self, sections: list[Chunk]) -> list[Chunk]:
        """Split sections that exceed max_char."""
        result: list[Chunk] = []

        for section in sections:
            sub_chunks = self._text_splitter.split_text(section.content)
            for sub_chunk in sub_chunks:
                result.append(
                    Chunk(content=sub_chunk, metadata=section.metadata.copy())
                )

        return result
