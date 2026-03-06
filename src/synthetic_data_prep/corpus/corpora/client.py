"""Corpus API client for uploading chunks and searching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from tqdm.auto import tqdm

from .exceptions import (
    AuthenticationError,
    ChunkLimitError,
    CorpusAPIError,
    CorpusLimitError,
    CorpusNotFoundError,
)
from .models import Corpus, CorpusChunk, SearchResult, UploadResult

if TYPE_CHECKING:
    from synthetic_data_prep.chunkers.models import Chunk, ChunkCollection


@dataclass
class CorpusClient:
    """Client for interacting with the Corpora API.

    Example:
        >>> client = CorpusClient(api_key="sk_...", base_url="http://localhost:3000")
        >>> corpus = client.create_corpus("my-docs")
        >>> result = client.upload_chunks(corpus.id, collection)
        >>> print(f"Uploaded {result.inserted_count} chunks")
    """

    api_key: str
    base_url: str = "http://localhost:3000"
    timeout: float = 30.0
    _http_client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize HTTP client with auth headers."""
        self._http_client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

    def __enter__(self) -> "CorpusClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._http_client.close()

    def _handle_response_errors(self, response: httpx.Response) -> None:
        """Convert HTTP errors to appropriate exceptions."""
        if response.status_code in (200, 201):
            return

        try:
            error_data = response.json()
            message = error_data.get("error", response.text)
        except Exception:
            message = response.text

        if response.status_code == 401:
            raise AuthenticationError(message)

        if response.status_code == 400:
            if "Maximum of 5 corpora" in message:
                raise CorpusLimitError()
            if "Chunk limit exceeded" in message:
                raise CorpusAPIError(message, 400)
            raise CorpusAPIError(message, 400)

        if response.status_code == 404:
            raise CorpusNotFoundError(message)

        raise CorpusAPIError(message, response.status_code)

    # === Corpus Management ===

    def create_corpus(self, name: str) -> Corpus:
        """Create a new corpus.

        Args:
            name: Name for the corpus

        Returns:
            Corpus object with id, name, timestamps

        Raises:
            CorpusLimitError: If max 5 corpora limit reached
            AuthenticationError: If API key is invalid
        """
        response = self._http_client.post("/api/corpora", json={"name": name})
        self._handle_response_errors(response)
        return Corpus.from_api_response(response.json())

    def list_corpora(self) -> list[Corpus]:
        """List all corpora for the authenticated user.

        Returns:
            List of Corpus objects
        """
        response = self._http_client.get("/api/corpora")
        self._handle_response_errors(response)
        return [Corpus.from_api_response(c) for c in response.json()]

    def delete_corpus(self, corpus_id: str) -> bool:
        """Delete a corpus and all its chunks.

        Args:
            corpus_id: ID of the corpus to delete

        Returns:
            True if deletion was successful
        """
        response = self._http_client.delete(f"/api/corpora/{corpus_id}")
        self._handle_response_errors(response)
        return response.json().get("success", False)

    def get_or_create_corpus(self, name: str, on_limit: str = "prompt") -> Corpus:
        """Get existing corpus by name or create new one.

        Args:
            name: Corpus name
            on_limit: Behavior when limit reached:
                - "prompt": Interactive prompt to delete (notebook)
                - "error": Raise CorpusLimitError
                - "oldest": Auto-delete oldest corpus

        Returns:
            Corpus object
        """
        # First, try to find existing corpus by name
        existing = self.list_corpora()
        for corpus in existing:
            if corpus.name == name:
                return corpus

        # Try to create new corpus
        try:
            return self.create_corpus(name)
        except CorpusLimitError as e:
            e.existing_corpora = existing

            if on_limit == "error":
                raise

            if on_limit == "oldest":
                oldest = min(existing, key=lambda c: c.created_at)
                print(f"Deleting oldest corpus: {oldest.name} (created {oldest.created_at})")
                self.delete_corpus(oldest.id)
                return self.create_corpus(name)

            if on_limit == "prompt":
                return self._interactive_corpus_selection(name, existing)

            raise ValueError(f"Unknown on_limit strategy: {on_limit}")

    def _interactive_corpus_selection(self, new_name: str, existing: list[Corpus]) -> Corpus:
        """Interactive corpus selection for Jupyter notebooks."""
        print("\n" + "=" * 60)
        print("CORPUS LIMIT REACHED (max 5)")
        print("=" * 60)
        print("\nExisting corpora:")

        for i, corpus in enumerate(existing, 1):
            print(f"  {i}. {corpus.name}")
            print(f"     ID: {corpus.id}")
            print(f"     Created: {corpus.created_at}")

        print(f"\n  0. Cancel operation")
        print()

        while True:
            try:
                choice = input(f"Enter number to delete (1-{len(existing)}) or 0 to cancel: ")
                choice_int = int(choice)

                if choice_int == 0:
                    raise CorpusLimitError(existing)

                if 1 <= choice_int <= len(existing):
                    to_delete = existing[choice_int - 1]
                    confirm = input(f"Delete '{to_delete.name}'? (y/N): ")

                    if confirm.lower() == "y":
                        self.delete_corpus(to_delete.id)
                        print(f"Deleted corpus: {to_delete.name}")
                        return self.create_corpus(new_name)
                    print("Cancelled.")
                    continue

                print(f"Please enter a number between 0 and {len(existing)}")

            except ValueError:
                print("Please enter a valid number")

    # === Chunk Upload ===

    def upload_chunks(
        self,
        corpus_id: str,
        collection: ChunkCollection,
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> UploadResult:
        """Upload a ChunkCollection to a corpus.

        Args:
            corpus_id: Target corpus ID
            collection: ChunkCollection to upload
            batch_size: Number of chunks per batch (default 100)
            show_progress: Show tqdm progress bar

        Returns:
            UploadResult with counts and ID list

        Raises:
            ChunkLimitError: If would exceed 10,000 chunk limit
        """
        chunks_list = list(collection)
        total_chunks = len(chunks_list)
        all_chunk_ids: list[str] = []
        total_inserted = 0

        # Create batches
        batches = [chunks_list[i : i + batch_size] for i in range(0, total_chunks, batch_size)]

        pbar = tqdm(batches, desc="Uploading chunks", disable=not show_progress, unit="batch")

        for batch in pbar:
            # Prepare batch payload - include hash in metadata for later matching
            payload = {
                "chunks": [
                    {
                        "id": chunk.hash,
                        "content": chunk.content,
                        "metadata": {**chunk.metadata_dict, "_local_hash": chunk.hash},
                    }
                    for chunk in batch
                ]
            }

            response = self._http_client.post(f"/api/corpora/{corpus_id}/chunks", json=payload)
            self._handle_response_errors(response)

            data = response.json()
            total_inserted += data.get("insertedCount", 0)
            chunk_ids = data.get("chunkIds", [])
            all_chunk_ids.extend(chunk_ids)

            pbar.set_postfix({"inserted": total_inserted})

        return UploadResult(
            success=True,
            inserted_count=total_inserted,
            chunk_ids=all_chunk_ids,
        )

    # === Search ===

    def search(
        self,
        corpus_id: str,
        query: str,
        limit: int = 10,
        offset: int = 0,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> SearchResult:
        """BM25 search within a corpus.

        Args:
            corpus_id: Corpus to search
            query: Search query string
            limit: Maximum results to return
            offset: Pagination offset
            metadata: Optional metadata filter
            filters: Optional structured filter DSL.
                Example:
                {
                    "and": [
                        {"field": "date_start", "op": "gte", "value": "2017-01-01"},
                        {"field": "participants", "op": "contains_any", "value": ["angel"]},
                    ]
                }

        Returns:
            SearchResult with results and total count
        """
        payload: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        if metadata:
            payload["metadata"] = metadata
        if filters:
            payload["filters"] = filters

        response = self._http_client.post(f"/api/corpora/{corpus_id}/search", json=payload)
        self._handle_response_errors(response)

        data = response.json()
        results = [
            CorpusChunk(
                id=r["id"],
                content=r["content"],
                metadata=r.get("metadata") or {},
                score=r.get("score"),
            )
            for r in data.get("results", [])
        ]

        return SearchResult(results=results, total=data.get("total", 0), query=query)

    def search_with_chunks(
        self,
        corpus_id: str,
        query: str,
        collection: ChunkCollection,
        limit: int = 10,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search and return matching local Chunk objects with scores.

        Requires chunks to have been uploaded (uses hash for matching).

        Args:
            corpus_id: Corpus to search
            query: Search query
            collection: Local ChunkCollection
            limit: Max results
            metadata: Optional exact-match metadata filter
            filters: Optional structured filter DSL

        Returns:
            List of (Chunk, score) tuples
        """
        result = self.search(
            corpus_id=corpus_id,
            query=query,
            limit=limit,
            metadata=metadata,
            filters=filters,
        )

        matched: list[tuple[Chunk, float]] = []
        for corpus_chunk in result.results:
            # The chunk ID is the hash, so we can look up directly
            local_chunk = collection.get_chunk_by_hash(corpus_chunk.id)
            if local_chunk:
                matched.append((local_chunk, corpus_chunk.score or 0.0))

        return matched
