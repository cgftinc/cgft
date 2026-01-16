"""Utilities for inspecting and visualizing chunks."""

from collections import defaultdict
from typing import Callable

from synthetic_data_prep.models import Chunk


class ChunkInspector:
    """Inspector for analyzing and displaying chunks.

    Groups chunks by file and provides methods for selective printing
    and statistical analysis.

    Example:
        >>> inspector = ChunkInspector(chunks)
        >>> inspector.summary()
        >>> inspector.print_file("README.md")
        >>> inspector.print_chunks(max_chunks=10)
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """Initialize the inspector with a list of chunks.

        Args:
            chunks: List of Chunk objects to inspect.
        """
        self.chunks = chunks
        self._by_file: dict[str, list[Chunk]] = defaultdict(list)

        for chunk in chunks:
            file = chunk.metadata.get("file", "unknown")
            self._by_file[file].append(chunk)

    @property
    def files(self) -> list[str]:
        """List of unique files in the chunks."""
        return list(self._by_file.keys())

    def summary(self) -> None:
        """Print summary statistics about the chunks."""
        if not self.chunks:
            print("No chunks to inspect.")
            return

        # Find min/max chunks with their files
        min_chunk = min(self.chunks, key=len)
        max_chunk = max(self.chunks, key=len)
        min_file = min_chunk.metadata.get("file", "unknown")
        max_file = max_chunk.metadata.get("file", "unknown")
        total_chars = sum(len(c) for c in self.chunks)

        print(f"Total chunks: {len(self.chunks)}")
        print(f"Unique files: {len(self._by_file)}")
        print(f"\nChunk length stats:")
        print(f"  Min: {len(min_chunk)} chars ({min_file})")
        print(f"  Max: {len(max_chunk)} chars ({max_file})")
        print(f"  Avg: {total_chars / len(self.chunks):.0f} chars")

        print(f"\nChunks per file:")
        for file, file_chunks in sorted(
            self._by_file.items(), key=lambda x: -len(x[1])
        )[:10]:
            print(f"  {file}: {len(file_chunks)} chunks")

        if len(self._by_file) > 10:
            print(f"  ... and {len(self._by_file) - 10} more files")

    def print_file(
        self,
        file: str,
        max_chunks: int | None = None,
        show_content: bool = True,
    ) -> None:
        """Print all chunks from a specific file.

        Args:
            file: The file path to filter by.
            max_chunks: Maximum number of chunks to print (None for all).
            show_content: Whether to print the full content.
        """
        if file not in self._by_file:
            print(f"File not found: {file}")
            print(f"Available files: {', '.join(self.files[:5])}...")
            return

        file_chunks = self._by_file[file]
        self._print_file_header(file, len(file_chunks))

        chunks_to_print = file_chunks[:max_chunks] if max_chunks else file_chunks

        for chunk in chunks_to_print:
            self._print_chunk(chunk, show_content=show_content)

        if max_chunks and len(file_chunks) > max_chunks:
            print(f"\n... {len(file_chunks) - max_chunks} more chunks in this file")

    def print_chunks(
        self,
        max_chunks: int | None = None,
        max_files: int | None = None,
        show_content: bool = True,
        filter_fn: Callable[[Chunk], bool] | None = None,
    ) -> None:
        """Print chunks grouped by file.

        Args:
            max_chunks: Maximum total number of chunks to print.
                Cannot be used with max_files.
            max_files: Maximum number of files to show (prints all chunks
                from each file). Cannot be used with max_chunks.
            show_content: Whether to print the full content.
            filter_fn: Optional function to filter chunks.

        Raises:
            ValueError: If both max_chunks and max_files are provided.
        """
        if max_chunks is not None and max_files is not None:
            raise ValueError("Cannot specify both max_chunks and max_files")

        # Default to max_chunks=10 if neither specified
        if max_chunks is None and max_files is None:
            max_chunks = 10

        chunks_to_inspect = self.chunks
        if filter_fn:
            chunks_to_inspect = [c for c in chunks_to_inspect if filter_fn(c)]

        if max_files is not None:
            self._print_by_files(chunks_to_inspect, max_files, show_content)
        else:
            # max_chunks is guaranteed to be int here (defaulted to 10 above)
            assert max_chunks is not None
            self._print_by_chunks(chunks_to_inspect, max_chunks, show_content)

    def _print_by_chunks(
        self,
        chunks: list[Chunk],
        max_chunks: int,
        show_content: bool,
    ) -> None:
        """Print up to max_chunks chunks."""
        current_file = None
        printed = 0

        for chunk in chunks:
            if printed >= max_chunks:
                break

            file = chunk.metadata.get("file", "unknown")

            if file != current_file:
                current_file = file
                file_chunk_count = len(self._by_file.get(file, []))
                self._print_file_header(file, file_chunk_count)

            self._print_chunk(chunk, show_content=show_content)
            printed += 1

        remaining = len(chunks) - printed
        if remaining > 0:
            print(f"\n... {remaining} more chunks not shown")

    def _print_by_files(
        self,
        chunks: list[Chunk],
        max_files: int,
        show_content: bool,
    ) -> None:
        """Print all chunks from up to max_files files."""
        # Group filtered chunks by file
        by_file: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            file = chunk.metadata.get("file", "unknown")
            by_file[file].append(chunk)

        files_printed = 0
        total_chunks_printed = 0

        for file, file_chunks in by_file.items():
            if files_printed >= max_files:
                break

            total_chunks_printed += self._print_file_chunks(
                file,
                file_chunks,
                show_content=show_content,
            )

            files_printed += 1

        remaining_files = len(by_file) - files_printed
        if remaining_files > 0:
            print(f"\n... {remaining_files} more files not shown")

    def _print_file_header(self, file: str, chunk_count: int) -> None:
        """Print a standardized file header."""
        print(f"\n{'=' * 80}")
        print(f"FILE: {file} ({chunk_count} chunks)")
        print("=" * 80)

    def _print_file_chunks(
        self,
        file: str,
        chunks: list[Chunk],
        show_content: bool,
    ) -> int:
        """Print all chunks for a file and return the count printed."""
        self._print_file_header(file, len(chunks))
        for chunk in chunks:
            self._print_chunk(chunk, show_content=show_content)
        return len(chunks)

    def _print_chunk(self, chunk: Chunk, show_content: bool = True) -> None:
        """Print a single chunk with formatting."""
        other_metadata = {
            k: v for k, v in chunk.metadata.items() if k not in ["file", "index"]
        }
        index = chunk.metadata.get("index", "?")

        print(f"\n==== Chunk {index} ({len(chunk)} chars) {other_metadata}")

        if show_content:
            print(chunk.content)

    def get_file_chunks(self, file: str) -> list[Chunk]:
        """Get all chunks from a specific file.

        Args:
            file: The file path to filter by.

        Returns:
            List of chunks from that file.
        """
        return self._by_file.get(file, [])

    def filter(self, fn: Callable[[Chunk], bool]) -> "ChunkInspector":
        """Create a new inspector with filtered chunks.

        Args:
            fn: Filter function that returns True for chunks to keep.

        Returns:
            New ChunkInspector with only matching chunks.
        """
        return ChunkInspector([c for c in self.chunks if fn(c)])

    def search(self, query: str, case_sensitive: bool = False) -> "ChunkInspector":
        """Search for chunks containing a query string.

        Args:
            query: String to search for in chunk content.
            case_sensitive: Whether to do case-sensitive matching.

        Returns:
            New ChunkInspector with only matching chunks.
        """
        if case_sensitive:
            return self.filter(lambda c: query in c.content)
        else:
            query_lower = query.lower()
            return self.filter(lambda c: query_lower in c.content.lower())
