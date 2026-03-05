"""Helper functions for QA generation and corpus analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import QADataPoint, ReferenceChunk

if TYPE_CHECKING:
    from ..chunkers.models import Chunk, ChunkCollection
    from ..corpus.source import ChunkSource

# ============================================================================
# Template Rendering and Parsing
# ============================================================================


def render_template(template: str, variables: dict[str, Any]) -> str:
    """
    Render a template string with variables.

    Args:
        template: Template string with {variable} placeholders
        variables: Dictionary of variable values

    Returns:
        Rendered template string
    """
    return template.format(**variables)


# ============================================================================
# Chunk Filtering and Sampling
# ============================================================================


def filter_chunks_by_length(
    chunk_list: list[Chunk],
    min_chars: int = 0,
    max_chars: int | None = None,
) -> list[Chunk]:
    """Filter chunks by character length.

    This is a simple filter that works for both single-hop and multi-hop
    chunk selection. For more complex sampling (with max_chunks and
    sampling strategies), use select_eligible_chunks().

    Args:
        chunk_list: List of chunks to filter
        min_chars: Minimum character length (inclusive, default 0)
        max_chars: Maximum character length (inclusive, None for no limit)

    Returns:
        List of chunks meeting the length criteria

    Example:
        >>> # Filter for chunks with meaningful content (>500 chars)
        >>> eligible = filter_chunks_by_length(collection, min_chars=500)
        >>>
        >>> # Filter for medium-sized chunks
        >>> medium = filter_chunks_by_length(collection, min_chars=500, max_chars=2000)
    """
    filtered = []
    for chunk in chunk_list:
        length = len(chunk)
        if length >= min_chars:
            if max_chars is None or length <= max_chars:
                filtered.append(chunk)

    return filtered


# ============================================================================
# Corpus Search Helpers
# ============================================================================


def search_related_chunks(
    source_chunk: Chunk,
    queries: list[str],
    corpus_client,
    corpus,
    collection: ChunkCollection,
    top_k: int = 5,
) -> list[dict]:
    """
    Search corpus for chunks related to source_chunk using BM25 queries.

    Runs each query through the corpus client's BM25 search and aggregates
    results, filtering out the source chunk and adjacent chunks in the same file.

    Args:
        source_chunk: The chunk we're finding related chunks for
        queries: List of search queries to run
        corpus_client: CorpusClient instance with search_with_chunks method
        corpus: Corpus object with .id attribute
        collection: ChunkCollection to match results against
        top_k: Number of results to fetch per query

    Returns:
        List of dicts sorted by relevance, each containing:
        - chunk: Chunk object
        - queries: list of queries that matched this chunk
        - same_file: bool indicating if chunk is from same file as source
        - max_score: float (highest BM25 score across queries)

        Sorted by: (num matching queries DESC, different file first, max_score DESC)
    """
    related_chunks_map: dict[str, dict] = {}

    for query in queries:
        matched_chunks = corpus_client.search_with_chunks(
            corpus_id=corpus.id, query=query, collection=collection, limit=top_k
        )

        for result_chunk, score in matched_chunks[:top_k]:
            # Skip the source chunk itself
            if result_chunk.hash == source_chunk.hash:
                continue

            # Skip adjacent chunks in the same file
            is_same_file = result_chunk.get_metadata("file") == source_chunk.get_metadata("file")
            if is_same_file:
                index_diff = abs(
                    result_chunk.get_metadata("index", 0) - source_chunk.get_metadata("index", 0)
                )
                if index_diff <= 1:
                    continue

            # Aggregate results
            if result_chunk.hash not in related_chunks_map:
                related_chunks_map[result_chunk.hash] = {
                    "chunk": result_chunk,
                    "queries": [],
                    "same_file": is_same_file,
                    "max_score": score,
                }
            else:
                related_chunks_map[result_chunk.hash]["max_score"] = max(
                    related_chunks_map[result_chunk.hash]["max_score"], score
                )

            related_chunks_map[result_chunk.hash]["queries"].append(query)

    # Sort by: num matching queries (desc), different file first, max_score (desc)
    return sorted(
        related_chunks_map.values(),
        key=lambda x: (len(x["queries"]), not x["same_file"], x["max_score"]),
        reverse=True,
    )


# ============================================================================
# Display and Formatting Helpers
# ============================================================================


def print_sections(*sections: tuple[str, str]) -> None:
    """
    Print multiple titled sections with boxed headers for readability.

    Each section consists of a title (centered in a box) and content below it.

    Args:
        *sections: Variable number of (title, content) tuples

    Example:
        >>> print_sections(
        ...     ("SYSTEM PROMPT", "You are a helpful assistant."),
        ...     ("USER PROMPT", "What is Python?"),
        ... )
        ┌─────────────────────────────────────┐
        │          SYSTEM PROMPT              │
        └─────────────────────────────────────┘
        You are a helpful assistant.

        ┌─────────────────────────────────────┐
        │          USER PROMPT                │
        └─────────────────────────────────────┘
        What is Python?
    """
    for i, (title, content) in enumerate(sections):
        if i > 0:
            print()  # Add blank line between sections

        # Calculate box width based on title length (minimum 120, add padding)
        title_width = max(120, len(title) + 10)

        # Create box lines
        top_line = "┌" + "─" * (title_width - 2) + "┐"
        bottom_line = "└" + "─" * (title_width - 2) + "┘"

        # Center the title within the box
        title_padding = title_width - 2 - len(title)
        left_pad = title_padding // 2
        right_pad = title_padding - left_pad
        centered_title = " " * left_pad + title + " " * right_pad

        # Print the boxed title
        print(top_line)
        print(f"│{centered_title}│")
        print(bottom_line)

        # Print the content
        print(content)


# ============================================================================
# Batch Generation Helpers
# ============================================================================


def generate_single_hop_batch(
    source: ChunkSource,
    client: Any,
    model: str,
    system_prompt: str,
    user_template: str,
    num_samples: int,
    response_parser: Any,
    min_chunk_chars: int = 400,
    context_preview_chars: int = 200,
    max_concurrent: int = 10,
    max_tokens: int = 1000,
    timeout: float = 120.0,
    show_progress: bool = True,
    max_questions: int | None = None,
) -> list[QADataPoint]:
    """Generate single-hop QA pairs in batch using parallel LLM calls.

    Args:
        source: ChunkSource backend to sample chunks from
        client: OpenAI client instance
        model: Model name to use
        system_prompt: System prompt for single-hop generation
        user_template: User prompt template with {chunk_content}, {prev_chunk_preview},
            {next_chunk_preview} placeholders
        num_samples: Number of chunks to sample and process
        response_parser: Function to parse LLM response, returns (confidence, qa_pairs)
        min_chunk_chars: Minimum chunk length to consider (default 400)
        context_preview_chars: Max chars for context previews (default 200)
        max_concurrent: Maximum concurrent LLM calls (default 10)
        max_tokens: Maximum tokens per response (default 1000)
        timeout: Request timeout in seconds (default 120.0)
        show_progress: Whether to show progress bar (default True)
        max_questions: Maximum number of questions to keep per chunk (default None for no limit)

    Returns:
        List of QADataPoints with generated single-hop QA pairs
    """
    from .batch_processor import batch_process_sync

    # 1. Sample chunks
    sampled_chunks = source.sample_chunks(num_samples, min_chars=min_chunk_chars)

    # 2. Build prompts for each chunk
    prompts = []
    for chunk in sampled_chunks:
        ctx = source.get_chunk_with_context(chunk, max_chars=context_preview_chars)
        prompt = render_template(user_template, ctx)
        prompts.append(prompt)

    # 3. Batch process all prompts
    result = batch_process_sync(
        client=client,
        model=model,
        prompts=prompts,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        max_concurrent=max_concurrent,
        show_progress=show_progress,
    )

    # 4. Parse responses and collect QA pairs
    dataset: list[QADataPoint] = []
    for i, response in enumerate(result.responses):
        if response is None:
            continue

        chunk = sampled_chunks[i]
        confidence, qa_pairs = response_parser(response.answer)

        if confidence.lower() == "low":
            continue

        if max_questions is not None:
            qa_pairs = qa_pairs[:max_questions]

        for qa in qa_pairs:
            dataset.append(
                QADataPoint(
                    question=qa["query"],
                    answer=qa["answer"],
                    reference_chunks=[
                        ReferenceChunk(
                            id=chunk.hash,
                            metadata=chunk.metadata_dict,
                            content=chunk.content,
                        )
                    ],
                    qa_type="single_hop",
                )
            )

    return dataset


def generate_multi_hop_batch(
    source: ChunkSource,
    client: Any,
    model: str,
    related_query_system_prompt: str,
    related_query_user_template: str,
    multi_hop_system_prompt: str,
    multi_hop_user_template: str,
    num_samples: int,
    related_query_parser: Any,
    multi_hop_parser: Any,
    min_chunk_chars: int = 400,
    context_preview_chars: int = 200,
    top_k_bm25: int = 5,
    top_related_chunks: int = 3,
    max_concurrent: int = 10,
    max_tokens: int = 1000,
    timeout: float = 120.0,
    show_progress: bool = True,
    max_questions: int | None = None,
) -> list[QADataPoint]:
    """Generate multi-hop QA pairs in batch using parallel LLM calls.

    Multi-hop generation is a two-step process:
    1. Generate related queries for sampled chunks (parallelized)
    2. For each chunk with high-confidence queries, search for related chunks
       and validate connections to generate QA pairs (parallelized)

    Args:
        source: ChunkSource backend to sample chunks from and search related chunks
        client: OpenAI client instance
        model: Model name to use
        related_query_system_prompt: System prompt for generating related queries
        related_query_user_template: User template for related query generation
        multi_hop_system_prompt: System prompt for multi-hop QA validation
        multi_hop_user_template: User template for multi-hop QA validation
        num_samples: Number of chunks to sample and process
        related_query_parser: Function to parse related query response,
            returns (confidence, queries)
        multi_hop_parser: Function to parse multi-hop response, returns list of qa_pairs
        min_chunk_chars: Minimum chunk length to consider (default 400)
        context_preview_chars: Max chars for context previews (default 200)
        top_k_bm25: Number of BM25 results per query (default 5)
        top_related_chunks: Number of related chunks to validate per source (default 3)
        max_concurrent: Maximum concurrent LLM calls (default 10)
        max_tokens: Maximum tokens per response (default 1000)
        timeout: Request timeout in seconds (default 120.0)
        show_progress: Whether to show progress bar (default True)
        max_questions: Maximum number of questions to keep per chunk pair (default None for no limit)

    Returns:
        List of QADataPoints with generated multi-hop QA pairs
    """
    from .batch_processor import batch_process_sync

    # 1. Sample chunks
    sampled_chunks = source.sample_chunks(num_samples, min_chars=min_chunk_chars)

    if show_progress:
        print(f"Step 1: Generating related queries for {len(sampled_chunks)} chunks...")

    # 2. Build prompts for related query generation
    related_prompts = []
    for chunk in sampled_chunks:
        ctx = source.get_chunk_with_context(chunk, max_chars=context_preview_chars)
        prompt = render_template(related_query_user_template, ctx)
        related_prompts.append(prompt)

    # 3. Batch process related query generation
    related_result = batch_process_sync(
        client=client,
        model=model,
        prompts=related_prompts,
        system_prompt=related_query_system_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        max_concurrent=max_concurrent,
        show_progress=show_progress,
    )

    # 4. Parse responses and search for related chunks
    chunk_pairs = []  # List of (chunk_a, chunk_b, connecting_queries)

    for i, response in enumerate(related_result.responses):
        if response is None:
            continue

        chunk_a = sampled_chunks[i]
        confidence, queries = related_query_parser(response.answer)

        if confidence.lower() == "low" or not queries:
            continue

        search_results = source.search_related(chunk_a, queries, top_k=top_k_bm25)

        for result in search_results[:top_related_chunks]:
            chunk_b = result["chunk"]
            connecting_queries = result["queries"]
            chunk_pairs.append((chunk_a, chunk_b, connecting_queries))

    if not chunk_pairs:
        if show_progress:
            print("No valid chunk pairs found for multi-hop generation.")
        return []

    if show_progress:
        print(f"\nStep 2: Validating {len(chunk_pairs)} chunk pairs for multi-hop QA...")

    # 5. Build prompts for multi-hop validation
    multi_hop_prompts = []
    for chunk_a, chunk_b, connecting_queries in chunk_pairs:
        ctx_a = source.get_chunk_with_context(chunk_a, max_chars=context_preview_chars)
        ctx_b = source.get_chunk_with_context(chunk_b, max_chars=context_preview_chars)

        prompt = render_template(
            multi_hop_user_template,
            {
                "connecting_queries": connecting_queries,
                "chunk_a": ctx_a["chunk_content"],
                "chunk_a_context_before": ctx_a["prev_chunk_preview"],
                "chunk_a_context_after": ctx_a["next_chunk_preview"],
                "chunk_b": ctx_b["chunk_content"],
                "chunk_b_context_before": ctx_b["prev_chunk_preview"],
                "chunk_b_context_after": ctx_b["next_chunk_preview"],
            },
        )
        multi_hop_prompts.append(prompt)

    # 6. Batch process multi-hop validation
    multi_hop_result = batch_process_sync(
        client=client,
        model=model,
        prompts=multi_hop_prompts,
        system_prompt=multi_hop_system_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        max_concurrent=max_concurrent,
        show_progress=show_progress,
    )

    # 7. Parse responses and collect QA pairs
    dataset: list[QADataPoint] = []
    for i, response in enumerate(multi_hop_result.responses):
        if response is None:
            continue

        chunk_a, chunk_b, _ = chunk_pairs[i]
        qa_pairs = multi_hop_parser(response.answer)

        if not qa_pairs:
            continue

        if max_questions is not None:
            qa_pairs = qa_pairs[:max_questions]

        for qa in qa_pairs:
            dataset.append(
                QADataPoint(
                    question=qa["question"],
                    answer=qa["answer"],
                    reference_chunks=[
                        ReferenceChunk(
                            id=chunk_a.hash,
                            metadata=chunk_a.metadata_dict,
                            content=chunk_a.content,
                        ),
                        ReferenceChunk(
                            id=chunk_b.hash,
                            metadata=chunk_b.metadata_dict,
                            content=chunk_b.content,
                        ),
                    ],
                    qa_type="multi_hop",
                )
            )

    return dataset
