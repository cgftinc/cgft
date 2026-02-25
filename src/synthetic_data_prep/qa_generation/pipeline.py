"""High-level pipeline functions for QA dataset generation."""

import random
from pathlib import Path

from openai import OpenAI

from synthetic_data_prep.corpus.source import ChunkSource
from synthetic_data_prep.qa_generation.helpers import (
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
    source: ChunkSource,
    api_key: str,
    corpus_description: str,
    example_queries: list[str],
    num_single_hop: int = 40,
    num_multi_hop: int = 5,
    model: str = "grok-4-fast-non-reasoning",
    model_endpoint: str = "http://app.cgft.io/api/llm",
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
        source: ChunkSource backend (e.g. CorporaChunkSource, TpufChunkSource)
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
        ...     source=source,
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
        source, corpus_description, example_queries, client, model
    )

    # Generate single-hop QA
    if show_summary:
        print(f"\nGenerating {num_single_hop} single-hop QA pairs...")

    single_hop_dataset = _generate_single_hop(
        source, client, model, corpus_context, num_single_hop, max_questions_per_chunk
    )

    if show_summary:
        print(single_hop_dataset.summary())

    # Generate multi-hop QA
    if show_summary:
        print(f"\nGenerating {num_multi_hop} multi-hop QA pairs...")

    multi_hop_dataset = _generate_multi_hop(
        source,
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
    source: ChunkSource,
    description: str,
    example_queries: list[str],
    client: OpenAI,
    model: str,
) -> dict[str, str]:
    """Generate corpus-level context summary."""
    corpus_system_prompt = """You are a technical analyst specializing in document corpus analysis.
Your goal is to understand the overall themes, content type, and typical query that a user might search for.

Based on the context and thoughts, form an understanding of the corpus, and then provide a short summary and example search queries. Weigh the user's input if provided.

Guidelines:
- Try to generalize and identify the overall theme of the corpus.
- Use your context of the theme, domain, etc. to guess the persona and purpose of searching this corpus
- i.e.
    - student finding their learning notes
    - developer looking up API documentation
    - journalist researching a topic
    - business analyst gathering market research
- Use that persona to guess the types of queries they might perform.

For the summary:
- First line - corpus themes (documentation, tutorials, reference, etc.)
- Second line - content domain (technical, business, scientific, etc.)
- Third line - user persona and purpose (likely developer looking up API documentation)
- Do NOT cite specific chunk content in the summary.

For the example queries:
- Provide 5-10 realistic example queries a user might search for in this corpus.
- Use the inferred user persona and purpose to guide the query style.
- Queries can have incomplete information, as often users do not remember full context.

Return JSON with:
- thoughts: Your analysis and reasoning here
- summary: our summary here (3 lines as described above)
- example_queries: List of example queries in the form of ["query1", "query2", ...]

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

    top_level = source.get_top_level_chunks()
    sampled_top_level = random.sample(top_level, min(4, len(top_level)))
    sampled_random = source.sample_chunks(4, min_chars=400)

    variables = {
        "user_context": f"Description: {description}\nExample queries provided by user: {', '.join(example_queries)}",
        "top_level_content": "\n\n".join([chunk.chunk_str() for chunk in sampled_top_level]),
        "random_content": "\n\n".join([chunk.chunk_str() for chunk in sampled_random]),
    }

    corpus_user_prompt = render_template(corpus_user_template, variables)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": corpus_system_prompt},
            {"role": "user", "content": corpus_user_prompt},
        ],
    )

    response = completion.choices[0].message.content or ""
    corpus_summary, queries = parse_corpus_summary_response(response)
    corpus_queries = "\n".join([f"- {q}" for q in queries])

    return {"corpus_summary": corpus_summary, "corpus_queries": corpus_queries}


def _generate_single_hop(
    source: ChunkSource,
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
- When generating queries, keep them terse, keyword-heavy like how real users would search. i.e.:
    - "k8s pod memory limits configuration"
    - "python async await best practices"
    - "quarterly revenue breakdown Q3"
- Queries / answers does not need to encompass the whole chunk. Query just need to target specific piece in the chunk that a user would likely want to know
- Query does not have to completely target all keywords in the chunk since users often only have partial recollection of the information, which is why they are searching
- The query should be answerable from the provided chunk alone
- Paraphrase the keywords / use synonyms in the query to make it natural and varied
- Rank and place your best question first if multiple q&a pairs are generated

Return JSON with:
- keywords: Relevant keywords that a user might search for in the chunk
- confidence: "low" | "mid" | "high";  use "low" if chunk has no meaningful information (too generic, boilerplate, or navigation-only)
- qa_pairs: List of query and answer pairs in the form of [{{"query": "q1", "answer": "a1"}}, ...]
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

Context before and after are only provided as additional context. Q&A should only target main chunk content.

Return JSON with keys: keywords, confidence, qa_pairs
"""

    single_hop_system_prompt = render_template(single_hop_system_template, corpus_context)

    return generate_single_hop_batch(
        source=source,
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
    source: ChunkSource,
    client: OpenAI,
    model: str,
    corpus_context: dict[str, str],
    num_samples: int,
    max_questions: int,
) -> QADataset:
    """Generate multi-hop QA pairs."""

    # Prompt for finding related chunks
    related_chunk_system_template = """You are generating BM25 search queries to find chunks that have meaningful relationships with the given chunk.

Corpus summary:
{corpus_summary}

## BM25 Behavior
BM25 matches exact keywords, weighted by rarity. This means:
- Specific/rare terms (product names, technical terms, unique phrases) are powerful
- Common corpus terms (e.g., "API", "data", "system") barely help
- BM25 won't match synonyms: "k8s" won't find "Kubernetes"
- Shorter, focused queries often outperform long ones

## Query Strategies

**1. Entity-focused**: Target specific named things that might appear elsewhere
  - Product/tool names: "Redis", "Workday", "Stripe"
  - Internal terms: "Project Atlas", "Q3 planning", "customer churn analysis"
  - Document names: "migration guide", "onboarding checklist"

**2. Reference-chasing**: If this chunk mentions other docs/sections, query for them
  - "see the deployment guide" → query: "deployment guide"
  - "as discussed in Q2 review" → query: "Q2 review"

**3. Inverse references**: Query for terms that other chunks might use to reference this one
  - If this is the Redis setup guide → query: "Redis setup", "Redis configuration"
  - If this covers auth flow → query: "authentication", "OAuth implementation"

**5. Synonym/variant expansion**: Generate alternate phrasings for key concepts
  - "Kubernetes" + "k8s"
  - "authentication" + "auth" + "login"
  - "configuration" + "config" + "setup"

## Query Format
- Prefer specific terms over generic ones
- Include both the canonical term and common variants
- Each query should target a *different* potential related chunk
- If the chunk is boilerplate (e.g., empty template, generic footer), set confidence to "low" and generate few/no queries

Return JSON with:
- keywords: Distinctive terms from this chunk likely to appear in related chunks
- confidence: "low" | "mid" | "high" - based on how much unique, linkable content exists
- queries: ["q1", "q2", ...] - diverse queries targeting different potential relationships
"""

    # Prompt for validating chunk connections and generating QA
    multi_hop_system_template = """You are analyzing whether two chunks have a meaningful dependency.

Corpus summary:
{corpus_summary}

Example queries from corpus:
{corpus_queries}

## Task

Analyze two chunks to determine if a meaningful relationship exists, then generate multi-hop questions that exploit that relationship.

**Terminology:**
- **Source chunk**: Contains the reference, pointer, or entry point
- **Target chunk**: Contains the referenced content, details, or destination

## Overall Notes:
- Queries should be terse, keyword-heavy like real user searches: "k8s pod memory limits configuration", "quarterly revenue breakdown Q3"
- Connecting queries come from BM25 (keyword matching)—shared terms don't guarantee meaningful relationships. Scrutinize whether matched terms have the same meaning in both chunks.
- Same high-level entity ≠ valid connection. Two chunks mentioning "Q3 revenue" or "the API" need actual dependency, not just shared subject matter.
- **When in doubt, choose "No Valid Relationship."** Weak relationships produce bad questions.
- **Hard requirements for every question:**
1. Question vocabulary matches source chunk better than target
2. Source chunk contains explicit or inferrable path to target
3. Answer cannot be complete without target chunk's information
4. Target chunk's distinctive terms must be paraphrased (use hypernyms, describe function/outcome, genericize proper nouns)
- Rank and place your best question first if multiple q&a pairs are generated

## Step 1: Identify relationship type

Classify into exactly one category:

### Explicit Reference
One chunk mentions, names, or links to content that the other chunk *is*.

Signals: Direct references ("see X", "refer to X"), matching document/section names, links, citations, phrases like "as explained in..."

Example: Source says "Q3 planning doc references the customer research findings" → Target is the customer research report

If found: Note direction (A→B means A is source, B is target)

### Abstraction Levels
Same core information at different granularity. One summarizes/claims, the other details/proves.

Signals: Claim + evidence, code + rationale, concept + procedure, summary + full content

Example: Source says "Mediterranean diet has strong evidence for heart health" → Target has study details with participant data and outcomes

Ensure both chunks refer to the same topic/sub-topic, not something tangentially related. **Bidirectional questions possible**—either chunk can serve as source.

### No Valid Relationship

Choose this if:
- Chunks are unrelated or connection is superficial
- Connection requires excessive inference (more than one logical step / only tangentially related)
- Near-duplicates (multi-hop adds no value)
- Content is independently complete on similar subjects

## Step 2: Generate Questions

Skip if relationship type is "No Valid Relationship."

### Question Strategies

**Explicit Reference:** Frame question around the *context* in source where reference appears. Ask for what target provides using paraphrased terms.

Source: "Trip itinerary mentions the restaurant recommendations doc"
Target: [Restaurant list: Tsuta for ramen, Sukiyabashi for sushi...]
✓ "good food japan trip plan" — matches source context, needs target for specifics, no target keywords
✗ "Tsuta ramen Sukiyabashi sushi" — retrieves target directly, bypasses source
✗ "japan trip plan" — matches source, but has no relevance to the target

Source: "Customer onboarding improvements are detailed in the Q3 ops review"
Target: [Ops review data: automated welcome emails, support ticket reduction 60%, average onboarding time 3 days → 4 hours...]
✓ "what changed in customer onboarding process" — matches source context, needs target for specifics, no target keywords
✗ "welcome email automation support ticket reduction" — retrieves target directly, bypasses source
✗ "when was Q3 ops review published" — retrieves source, but asks about metadata, not onboarding details

**Abstraction Levels:** Use vocabulary from source chunk, require precision only target provides. Generate questions in both directions.

General: "The org restructure significantly improved cross-team collaboration"
Specific: [Survey data: cross-team project completion up 40%, meeting conflicts down 25%...]
✓ General→Specific: "measurable impact of reorg on team collaboration"
✓ Specific→General: "leadership claims about collaboration survey results"
✗ "cross-team project completion meeting conflicts" — retrieves specific directly, bypasses general
✗ "org restructure announcement" — retrieves general, but no indication that survey data is needed

General: "Our new caching layer reduced API latency dramatically"
Specific: [Redis implementation: cache hit rates 94%, p99 latency down from 800ms to 120ms, eviction policies...]
✓ General→Specific: "performance metrics for the API speed improvements"
✓ Specific→General: "engineering summary of Redis cache results"
✗ "Redis cache hit rate p99 latency" — retrieves specific directly, bypasses general
✗ "new caching layer" — retrieves general, but no indication that implementation details are needed

Return JSON with:
- thoughts: Analysis of the relationship. State the relationship type found (or why none exists). If relationship exists, identify source vs target and the linking mechanism.
- relationship_type: "explicit_reference" | "abstraction_levels" | "none"
- direction: "A_to_B" | "B_to_A" | "bidirectional" | null
- linking_info: Object describing the connection, or null if none. Structure depends on relationship_type:
    - For explicit_reference: {{ reference_text, source ("A"|"B"), target ("A"|"B") }}
    - For abstraction_levels: {{ general_chunk ("A"|"B"), specific_chunk ("A"|"B"), abstraction_link }}
- qa_pairs: List of QA pairs, or null if no valid multi-hop questions can be formed. Each pair contains:
    - question: Terse, keyword-focused multi-hop question
    - answer: Answer requiring synthesis of both chunks
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

The before/after context is provided only as additional context. Queries should target content from the main chunk only.

Return JSON with keys: keywords, confidence, queries
"""

    multi_hop_user_template = """Analyze the connection between these chunks.

Connecting Queries: {connecting_queries}

<chunk_a_context_before>
{chunk_a_context_before}
</chunk_a_context_before>

<chunk_a>
{chunk_a}
</chunk_a>

<chunk_a_context_after>
{chunk_a_context_after}
</chunk_a_context_after>

---

<chunk_b_context_before>
{chunk_b_context_before}
</chunk_b_context_before>

<chunk_b>
{chunk_b}
</chunk_b>

<chunk_b_context_after>
{chunk_b_context_after}
</chunk_b_context_after>

Analyze whether there is a meaningful relationship between the chunks and whether multi-hop questions can be formed.

Return JSON with keys: thoughts, relationship_type, direction, linking_info, qa_pairs
"""

    related_chunk_system_prompt = render_template(related_chunk_system_template, corpus_context)
    multi_hop_system_prompt = render_template(multi_hop_system_template, corpus_context)

    return generate_multi_hop_batch(
        source=source,
        client=client,
        model=model,
        related_query_system_prompt=related_chunk_system_prompt,
        related_query_user_template=related_chunk_user_template,
        multi_hop_system_prompt=multi_hop_system_prompt,
        multi_hop_user_template=multi_hop_user_template,
        num_samples=num_samples,
        related_query_parser=parse_related_queries_response,
        multi_hop_parser=parse_multi_hop_validation_response,
        context_preview_chars=200,
        top_k_bm25=5,
        top_related_chunks=3,
        max_questions=max_questions,
    )
