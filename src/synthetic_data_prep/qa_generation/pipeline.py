"""High-level pipeline functions for QA dataset generation."""

import random
from pathlib import Path

from openai import OpenAI

from synthetic_data_prep.chunkers.models import ChunkCollection
from synthetic_data_prep.corpus.client import CorpusClient
from synthetic_data_prep.corpus.models import Corpus
from synthetic_data_prep.qa_generation.helpers import (
    filter_chunks_by_length,
    generate_multi_hop_batch,
    generate_single_hop_batch,
    render_template,
)
from synthetic_data_prep.qa_generation.models import QADataset
from synthetic_data_prep.qa_generation.response_parsers import (
    parse_corpus_summary_response,
    parse_multi_hop_validation_response,
    parse_related_queries_response,
    parse_single_hop_response,
)
from synthetic_data_prep.qa_generation.storage import save_qa_dataset, save_qa_dataset_jsonl


def generate_dataset(
    corpus: Corpus,
    collection: ChunkCollection,
    corpus_client: CorpusClient,
    api_key: str,
    corpus_description: str,
    example_queries: list[str],
    num_single_hop: int = 40,
    num_multi_hop: int = 5,
    model: str = "grok-4-fast-non-reasoning",
    model_endpoint: str = "http://localhost:3000/api/llm",
    output_dir: str = "outputs",
    max_questions_per_chunk: int = 2,
    show_summary: bool = True,
) -> QADataset:
    """Generate synthetic QA dataset from corpus in one step.

    This function handles the complete QA generation pipeline:
    1. Generate corpus context (summary and example queries)
    2. Generate single-hop QA pairs (one chunk answers question)
    3. Generate multi-hop QA pairs (multiple chunks needed)
    4. Merge and save datasets

    Args:
        corpus: Corpus object from prepare_corpus
        collection: ChunkCollection from prepare_corpus
        corpus_client: CorpusClient from prepare_corpus
        api_key: API key for LLM service
        corpus_description: Short description of corpus (e.g., "Posthog documentation")
        example_queries: Example queries users might search for
        num_single_hop: Number of single-hop samples to generate (default: 40)
        num_multi_hop: Number of multi-hop samples to generate (default: 5)
        model: Model name for generation (default: "grok-4-fast-non-reasoning")
        model_endpoint: LLM API endpoint (default: "http://localhost:3000/api/llm")
        output_dir: Directory to save outputs (default: "outputs")
        max_questions_per_chunk: Max questions per chunk for diversity (default: 2)
        show_summary: Whether to print summary information (default: True)

    Returns:
        Combined QADataset with single-hop and multi-hop questions

    Example:
        >>> dataset = generate_dataset(
        ...     corpus=corpus,
        ...     collection=collection,
        ...     corpus_client=corpus_client,
        ...     api_key="your-api-key",
        ...     corpus_description="Posthog documentation",
        ...     example_queries=["how to feature flag", "setup reverse proxy"],
        ...     num_single_hop=40,
        ...     num_multi_hop=5
        ... )
        Generating corpus summary and example queries...
        Generating 40 single-hop QA pairs...
        QADataset: 67 total data points
        Generating 5 multi-hop QA pairs...
        QADataset: 13 total data points
        Combined dataset: 80 total data points
    """
    # Setup LLM client
    client = OpenAI(base_url=model_endpoint, api_key=api_key)

    # Generate corpus context
    if show_summary:
        print("Generating corpus summary and example queries...")

    corpus_context = _generate_corpus_context(
        collection, corpus_description, example_queries, client, model
    )

    # Generate single-hop QA
    if show_summary:
        print(f"\nGenerating {num_single_hop} single-hop QA pairs...")

    single_hop_dataset = _generate_single_hop(
        collection, client, model, corpus_context, num_single_hop, max_questions_per_chunk
    )

    if show_summary:
        print(single_hop_dataset.summary())

    # Generate multi-hop QA
    if show_summary:
        print(f"\nGenerating {num_multi_hop} multi-hop QA pairs...")

    multi_hop_dataset = _generate_multi_hop(
        collection,
        corpus_client,
        corpus,
        client,
        model,
        corpus_context,
        num_multi_hop,
        max_questions_per_chunk,
    )

    if show_summary:
        print(multi_hop_dataset.summary())

    # Combine datasets
    combined = single_hop_dataset.merge(multi_hop_dataset)

    if show_summary:
        print(f"\nCombined dataset: {combined.summary()}")

    # Save to files
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    yaml_path = f"{output_dir}/qa_dataset.yaml"
    jsonl_path = f"{output_dir}/qa_dataset.jsonl"

    save_qa_dataset(combined, yaml_path)
    save_qa_dataset_jsonl(combined, jsonl_path)

    if show_summary:
        print(f"\nSaved datasets:\n  YAML: {yaml_path}\n  JSONL: {jsonl_path}")

    return combined


# Internal helper functions


def _generate_corpus_context(
    collection: ChunkCollection,
    description: str,
    example_queries: list[str],
    client: OpenAI,
    model: str,
) -> dict[str, str]:
    """Generate corpus-level context summary."""
    corpus_system_prompt = """You are a technical analyst specializing in document corpus analysis.
Your goal is to understand the overall themes, content type, and typical query that a user might search for.

Based on the context and thoughts, form an understanding of the corpus, and then provide a short summary and example search queries. Weigh the user's input if provided.

For the summary:
- First line - corpus themes (documentation, tutorials, reference, etc.)
- Second line - content domain (technical, business, scientific, etc.)
- Third line - user persona and purpose (likely developer looking up API documentation)

For the example queries:
- Provide 5-10 realistic example queries a user might search for in this corpus.

Return JSON with: thoughts, summary, example_queries
"""

    corpus_user_template = """Analyze the following document corpus:

<user_context>
{user_context}
</user_context>

<top_level_chunks>
{top_level_content}
</top_level_chunks>

<random_sampled_chunks>
{random_content}
</random_sampled_chunks>

Return your analysis as JSON with keys: thoughts, summary, example_queries
"""

    all_chunks = filter_chunks_by_length(collection.chunks, min_chars=400)
    top_level = collection.get_top_level_chunks()
    sampled_top_level = random.sample(top_level, min(4, len(top_level)))
    sampled_random = random.sample(all_chunks, min(4, len(all_chunks)))

    variables = {
        "user_context": f"Description: {description}\nExample queries: {', '.join(example_queries)}",
        "top_level_content": "\n\n".join([chunk.chunk_str() for chunk in sampled_top_level]),
        "random_content": "\n\n".join([chunk.chunk_str() for chunk in sampled_random]),
    }

    user_prompt = render_template(corpus_user_template, variables)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": corpus_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    response = completion.choices[0].message.content or ""
    corpus_summary, queries = parse_corpus_summary_response(response)
    corpus_queries = "\n".join([f"- {q}" for q in queries])

    return {"corpus_summary": corpus_summary, "corpus_queries": corpus_queries}


def _generate_single_hop(
    collection: ChunkCollection,
    client: OpenAI,
    model: str,
    corpus_context: dict[str, str],
    num_samples: int,
    max_questions: int,
) -> QADataset:
    """Generate single-hop QA pairs."""
    single_hop_system_template = """You are generating realistic search queries for a RAG system.

Corpus summary:
{corpus_summary}

Example queries:
{corpus_queries}

Guide:
- Keep queries terse, keyword-heavy like real users would search
- Queries/answers don't need to encompass the whole chunk
- Query should be answerable from the provided chunk alone
- Paraphrase keywords to make queries natural and varied

Return JSON with: keywords, confidence, qa_pairs
"""

    single_hop_user_template = """Generate single-hop search q&a pairs based on the following chunk:

<context_before>
{prev_chunk_preview}
</context_before>

<main_chunk>
{chunk_content}
</main_chunk>

<context_after>
{next_chunk_preview}
</context_after>

Return JSON with keys: keywords, confidence, qa_pairs
"""

    single_hop_system_prompt = render_template(single_hop_system_template, corpus_context)

    return generate_single_hop_batch(
        collection=collection,
        client=client,
        model=model,
        system_prompt=single_hop_system_prompt,
        user_template=single_hop_user_template,
        num_samples=num_samples,
        response_parser=parse_single_hop_response,
        context_preview_chars=200,
        max_questions=max_questions,
    )


def _generate_multi_hop(
    collection: ChunkCollection,
    corpus_client: CorpusClient,
    corpus: Corpus,
    client: OpenAI,
    model: str,
    corpus_context: dict[str, str],
    num_samples: int,
    max_questions: int,
) -> QADataset:
    """Generate multi-hop QA pairs."""
    related_chunk_system_template = """You are generating BM25 search queries to find chunks that have meaningful relationships with the given chunk.

Corpus summary:
{corpus_summary}

Generate queries to find related chunks using distinctive keywords.

Return JSON with: keywords, confidence, queries
"""

    multi_hop_system_template = """You are analyzing whether two chunks have a meaningful dependency.

Corpus summary:
{corpus_summary}

Example queries:
{corpus_queries}

Analyze if there's a meaningful relationship and generate multi-hop questions.

Return JSON with: thoughts, relationship_type, direction, linking_info, qa_pairs
"""

    related_chunk_user_template = """Generate search queries based on this chunk to find other relevant chunks:

<context_before>
{prev_chunk_preview}
</context_before>

<main_chunk>
{chunk_content}
</main_chunk>

<context_after>
{next_chunk_preview}
</context_after>

Return JSON with keys: keywords, confidence, queries
"""

    multi_hop_user_template = """Analyze the connection between these chunks.

Connecting Queries: {connecting_queries}

<chunk_a>
{chunk_a}
</chunk_a>

<chunk_b>
{chunk_b}
</chunk_b>

Return JSON with keys: thoughts, relationship_type, direction, linking_info, qa_pairs
"""

    related_system = render_template(related_chunk_system_template, corpus_context)
    multi_hop_system = render_template(multi_hop_system_template, corpus_context)

    return generate_multi_hop_batch(
        collection=collection,
        client=client,
        model=model,
        related_query_system_prompt=related_system,
        related_query_user_template=related_chunk_user_template,
        multi_hop_system_prompt=multi_hop_system,
        multi_hop_user_template=multi_hop_user_template,
        corpus_client=corpus_client,
        corpus=corpus,
        num_samples=num_samples,
        related_query_parser=parse_related_queries_response,
        multi_hop_parser=parse_multi_hop_validation_response,
        context_preview_chars=200,
        top_k_bm25=5,
        top_related_chunks=3,
        max_questions=max_questions,
    )
