"""High-level pipeline functions for corpus preparation."""

from pathlib import Path

from synthetic_data_prep.chunkers.inspector import ChunkInspector
from synthetic_data_prep.chunkers.markdown import MarkdownChunker
from synthetic_data_prep.chunkers.models import ChunkCollection
from synthetic_data_prep.corpus.corpora.client import CorpusClient
from synthetic_data_prep.corpus.corpora.models import Corpus


def prepare_corpus(
    docs_path: str,
    corpus_name: str,
    api_key: str,
    base_url: str = "https://app.cgft.io",
    min_chars: int = 1024,
    max_chars: int = 2048,
    overlap_chars: int = 128,
    file_extensions: list[str] | None = None,
    show_summary: bool = True,
) -> tuple[Corpus, ChunkCollection, CorpusClient]:
    """Chunk documents and upload to corpus API in one step.

    This function handles the complete corpus preparation pipeline:
    1. Chunk markdown documents with configurable size
    2. Create or get existing corpus
    3. Upload all chunks to the corpus

    Args:
        docs_path: Path to documentation folder
        corpus_name: Name for the corpus
        api_key: API key for corpus service
        base_url: Base URL for corpus API (default: https://app.cgft.io)
        min_chars: Minimum characters per chunk (default: 1024)
        max_chars: Maximum characters per chunk (default: 2048)
        overlap_chars: Character overlap between chunks (default: 128)
        file_extensions: List of file extensions to process (default: [".md", ".mdx"])
        show_summary: Whether to print summary information (default: True)

    Returns:
        Tuple of (Corpus, ChunkCollection, CorpusClient)

    Example:
        >>> corpus, collection, client = prepare_corpus(
        ...     docs_path="./samples/posthog",
        ...     corpus_name="my-docs",
        ...     api_key="your-api-key"
        ... )
        Chunking documents from ./samples/posthog...
        Using corpus: my-docs (ID: abc123)
        Uploading 245 chunks to corpus...
        Upload complete! Inserted: 245
    """
    if file_extensions is None:
        file_extensions = [".md", ".mdx"]

    # Step 1: Chunk documents
    if show_summary:
        print(f"Chunking documents from {docs_path}...")

    chunker = MarkdownChunker(min_char=min_chars, max_char=max_chars, chunk_overlap=overlap_chars)
    collection = chunker.chunk_folder(docs_path, file_extensions=file_extensions)

    if show_summary:
        inspector = ChunkInspector(collection)
        inspector.summary(max_depth=3, max_files_per_folder=4)

    # Step 2: Create/get corpus
    corpus_client = CorpusClient(api_key=api_key, base_url=base_url)
    corpus = corpus_client.get_or_create_corpus(corpus_name, on_limit="prompt")

    if show_summary:
        print(f"\nUsing corpus: {corpus.name} (ID: {corpus.id})")

    # Step 3: Upload chunks
    if show_summary:
        print(f"Uploading {len(collection)} chunks to corpus...")

    upload_result = corpus_client.upload_chunks(
        corpus_id=corpus.id,
        collection=collection,
        batch_size=100,
        show_progress=show_summary,
    )

    if show_summary:
        print(f"\nUpload complete! Inserted: {upload_result.inserted_count}")

    return corpus, collection, corpus_client
