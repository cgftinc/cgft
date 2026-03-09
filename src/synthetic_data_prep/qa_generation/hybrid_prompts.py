"""Prompt templates used by the hybrid QA generation pipeline."""

from __future__ import annotations

QUESTION_STYLE_INSTRUCTION_TEMPLATE = """Question style target: {query_style_target}

Style definitions:
- keyword: 2-7 token search phrase, no punctuation, dense with domain terms.
- natural: complete user-style question in plain language, usually with a question mark.
- expert: troubleshooting/comparison/decision-style question with concrete technical constraints.

Hard requirements:
- Follow the target style exactly.
- Do not output unresolved MDX/JSX component tags like <MyComponent/> in questions or answers.
- Keep technical terms and entities from the source content accurate.
"""

CORPUS_SYSTEM_PROMPT = """You are a technical analyst specializing in document corpus analysis.
Your goal is to understand the overall themes, content type, and typical query that a user might search for.

Based on the context and thoughts, form an understanding of the corpus, and then provide a short summary and example search queries. Weigh the user's input if provided.

Guideline:
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

CORPUS_USER_TEMPLATE = """Analyze the following document corpus:

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

SINGLE_HOP_SYSTEM_TEMPLATE = """You are generating realistic search queries for a RAG system.

Corpus summary:
{corpus_summary}

Example queries:
{corpus_queries}

Style guidance:
{style_instruction}

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

SINGLE_HOP_USER_TEMPLATE = """Generate a single-hop search q&a pairs based on the following chunk:

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

MULTI_HOP_SYSTEM_TEMPLATE = """You are analyzing whether two chunks have a meaningful dependency.

Corpus summary:
{corpus_summary}

Example queries from corpus:
{corpus_queries}

Style guidance:
{style_instruction}

## Task

Analyze two chunks to determine if a meaningful relationship exists, then generate multi-hop questions that exploit that relationship.

**Terminology:**
- **Source chunk**: Contains the reference, pointer, or entry point
- **Target chunk**: Contains the referenced content, details, or destination

## Overall Notes:
- Queries should be terse, keyword-heavy like real user searches: "k8s pod memory limits configuration", "quarterly revenue breakdown Q3"
- Connecting queries come from BM25 (keyword matching)-shared terms don't guarantee meaningful relationships. Scrutinize whether matched terms have the same meaning in both chunks.
- Same high-level entity != valid connection. Two chunks mentioning "Q3 revenue" or "the API" need actual dependency, not just shared subject matter.
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

If found: Note direction (A->B means A is source, B is target)

### Abstraction Levels
Same core information at different granularity. One summarizes/claims, the other details/proves.

Signals: Claim + evidence, code + rationale, concept + procedure, summary + full content

Ensure both chunks refer to the same topic/sub-topic, not something tangentially related. **Bidirectional questions possible**-either chunk can serve as source.

### No Valid Relationship
Choose this if:
- Chunks are unrelated or connection is superficial
- Connection requires excessive inference (more than one logical step / only tangentially related)
- Near-duplicates (multi-hop adds no value)
- Content is independently complete on similar subjects

## Step 2: Generate Questions

Skip if relationship type is "No Valid Relationship."

Return JSON with:
- thoughts: Analysis of the relationship. State the relationship type found (or why none exists). If relationship exists, identify source vs target and the linking mechanism.
- relationship_type: "explicit_reference" | "abstraction_levels" | "none"
- direction: "A_to_B" | "B_to_A" | "bidirectional" | null
- linking_info: Object describing the connection, or null if none.
- qa_pairs: List of QA pairs, or null if no valid multi-hop questions can be formed. Each pair contains:
    - question: Terse, keyword-focused multi-hop question
    - answer: Answer requiring synthesis of both chunks
"""

MULTI_HOP_USER_TEMPLATE = """Analyze the connection between these chunks.

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

NO_CONTEXT_FILTER_SYSTEM_PROMPT = (
    "Answer the following question as best you can. "
    "Do not say you don't know - make your best attempt."
)

EQUIVALENCE_JUDGE_SYSTEM_PROMPT = """You are evaluating whether two answers convey
the same information.
Respond with JSON only:
{
  "is_equivalent": true/false,
  "reasoning": "brief explanation"
}
Be strict - partial matches or vague answers should be marked false."""
