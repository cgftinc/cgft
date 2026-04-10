"""First-class corpus profile for threading context across pipeline stages.

``CorpusProfile`` consolidates the outputs of stages 2-3 (corpus summary,
entity extraction, capability detection) into a single mutable object.
The linker enriches it during linking; the generator reads the enriched
version for prompt conditioning.
"""

from __future__ import annotations

import functools
import logging
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cgft.qa_generation.corpus_capabilities import CorpusCapabilities

logger = logging.getLogger(__name__)


@dataclass
class EntityPattern:
    """An extracted entity with document-frequency quality signal.

    Attributes:
        name: The entity string (e.g. "Redis", "TTL").
        type: One of "entity", "code_pattern", "domain_term".
        document_frequency: Fraction of sample chunks containing this entity (0.0-1.0).
        idf_weight: log(N/df) — higher values indicate more discriminative entities.
    """

    name: str
    type: str  # "entity" | "code_pattern" | "domain_term"
    document_frequency: float = 0.0
    idf_weight: float = 0.0
    quality_score: float = 0.0


@dataclass
class CorpusMetadataCensus:
    """Corpus-relative distributions for calibrating chunk scoring.

    Computed once over the profile pool after entity extraction.
    All percentile fields are derived from ``statistics.quantiles(data, n=4)``.
    """

    # Prevalence: fraction of chunks with each metadata type
    header_prevalence: float  # any of h1/h2/h3/title/section_header
    doc_id_prevalence: float  # file_name/document_id/source
    date_prevalence: float  # date/timestamp fields
    sequential_prevalence: float  # prev_chunk_id/next_chunk_id or file+index

    # Content length distribution (characters)
    content_length_p25: int
    content_length_p50: int
    content_length_p75: int

    # Entity density distribution (recognized entities per 1000 chars)
    entity_density_p25: float
    entity_density_p50: float
    entity_density_p75: float

    # Lexical diversity distribution (unique tokens / total tokens)
    lexical_diversity_p25: float
    lexical_diversity_p50: float
    lexical_diversity_p75: float

    # Derived classification
    metadata_regime: str  # "structured" | "mixed" | "unstructured"

    # Chunk count (from source)
    chunk_count: int

    @property
    def multi_file_corpus(self) -> bool:
        return self.doc_id_prevalence > 0.05


@dataclass
class CorpusProfile:
    """Shared mutable corpus context threaded across pipeline stages.

    Created by stages 2-3 (profiling + entity extraction), enriched by the
    linker during linking, and read by the generator for prompt conditioning.
    """

    # --- Set by initial profiling stage (stages 2-3) ---
    corpus_summary: str = ""
    corpus_queries: list[str] = field(default_factory=list)
    corpus_description: str = ""
    entity_patterns: list[EntityPattern] = field(default_factory=list)
    capabilities: CorpusCapabilities = field(
        default_factory=lambda: CorpusCapabilities(
            has_document_ids=False,
            has_section_headers=False,
            has_sequential_links=False,
            has_dates=False,
        )
    )

    # --- Entity quality signals (computed from profile_sample) ---
    entity_document_frequency: dict[str, float] = field(default_factory=dict)
    ubiquitous_df_threshold: float = 0.80

    # --- Token-level TF-IDF signals (computed from profile_sample) ---
    token_document_frequency: dict[str, int] = field(default_factory=dict)
    token_df_sample_size: int = 0

    # --- Search capability detection ---
    search_modes: set[str] = field(default_factory=lambda: {"lexical"})
    best_search_mode: str = "lexical"

    # --- Enriched by linker during linking ---
    discovered_entities: list[str] = field(default_factory=list)
    effective_query_templates: list[str] = field(default_factory=list)
    connection_distribution: dict[str, int] = field(default_factory=dict)
    _query_effectiveness: dict[str, list[int]] = field(default_factory=dict)

    # --- Entity-chunk graph (computed during profiling) ---
    entity_chunk_index: dict[str, set[str]] = field(default_factory=dict)
    # entity name (lower) -> set of chunk hashes containing it (unfiltered)
    chunk_entity_index: dict[str, list[str]] = field(default_factory=dict)
    # chunk hash -> list of entity names found in it (unfiltered)
    entity_cooccurrence: dict[tuple[str, str], int] = field(default_factory=dict)
    # (entity_a, entity_b) canonicalized pair -> count of chunks containing both

    # --- Census and suitability (computed after entity extraction) ---
    census: CorpusMetadataCensus | None = None
    chunk_suitability_scores: dict[str, float] = field(default_factory=dict)
    # key = chunk hash, value = suitability score [0, 1]

    # --- Convenience ---

    @property
    def corpus_queries_bulleted(self) -> str:
        """Queries formatted for prompt templates."""
        return "\n".join(f"- {q}" for q in self.corpus_queries)

    @functools.cached_property
    def entity_quality_map(self) -> dict[str, float]:
        """Cached mapping of entity name (lower) -> quality_score."""
        return {p.name.lower(): p.quality_score for p in self.entity_patterns}

    def get_discriminative_entities(
        self,
        max_df: float | None = None,
        min_df: float = 0.005,
    ) -> list[EntityPattern]:
        """Return entities within a useful DF band (not too rare, not ubiquitous)."""
        threshold = max_df if max_df is not None else self.ubiquitous_df_threshold
        return [e for e in self.entity_patterns if min_df <= e.document_frequency < threshold]

    def get_entity_names(self, *, discriminative_only: bool = True) -> list[str]:
        """Return entity name strings, optionally filtered by DF."""
        if discriminative_only:
            return [e.name for e in self.get_discriminative_entities()]
        return [e.name for e in self.entity_patterns]

    def get_domain_terms(self, *, discriminative_only: bool = True) -> list[str]:
        """Return domain term strings, optionally filtered by DF."""
        if discriminative_only:
            return [e.name for e in self.get_discriminative_entities() if e.type == "domain_term"]
        return [e.name for e in self.entity_patterns if e.type == "domain_term"]

    def get_tfidf_keyphrases(
        self,
        chunk_content: str,
        *,
        top_n: int = 5,
        min_token_len: int = 4,
    ) -> list[str]:
        """Extract top-N discriminative keyphrases from *chunk_content*.

        Uses corpus-level token document frequency to compute TF-IDF
        scores per token, returning the highest-scoring tokens.
        """
        import re

        if self.token_df_sample_size == 0:
            return []

        tokens = re.findall(r"\w+", chunk_content.lower())
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        df_threshold = self.token_df_sample_size * 0.8

        scored: list[tuple[float, str]] = []
        for tok, count in tf.items():
            if len(tok) < min_token_len:
                continue
            df = self.token_document_frequency.get(tok)
            if df is None or df >= df_threshold:
                continue
            tfidf = count * math.log(self.token_df_sample_size / max(df, 1))
            scored.append((tfidf, tok))

        scored.sort(key=lambda x: -x[0])
        return [tok for _, tok in scored[:top_n]]

    # --- Enrichment methods (called by linker) ---

    def add_discovered_entities(self, entities: list[str]) -> None:
        """Record entities discovered during linking."""
        for entity in entities:
            if entity and entity not in self.discovered_entities:
                self.discovered_entities.append(entity)

    def record_query_effectiveness(self, query: str, num_results: int) -> None:
        """Track how many results each query template returns."""
        self._query_effectiveness.setdefault(query, []).append(num_results)
        if num_results > 0 and query not in self.effective_query_templates:
            self.effective_query_templates.append(query)

    def update_connection_stats(self, connection_type: str) -> None:
        """Increment the count for a connection type discovered during linking."""
        self.connection_distribution[connection_type] = (
            self.connection_distribution.get(connection_type, 0) + 1
        )


_ENTITY_STOPWORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "it", "if",
    "you", "your", "this", "that", "with", "from", "by", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "can", "could", "would", "should", "may", "not", "no", "or", "and", "but",
    "so", "as", "its", "our", "we", "re", "s", "t",
})


def _score_entity_quality(pattern: EntityPattern) -> float:
    """Score entity quality from DF + name heuristics. Returns [0, 1].

    Combines three signals:
    - DF discriminativeness (50%): flat top at 0.02-0.10, taper to 0 at 0.40.
    - Type bonus (30%): entity > code_pattern > domain_term.
    - Name quality (20%): uppercase required for high score; all-stopword penalised.
    """
    df = pattern.document_frequency

    # DF discriminativeness: sweet spot 0.02-0.10
    if df <= 0.0:
        df_score = 0.0
    elif df < 0.02:
        df_score = df / 0.02
    elif df <= 0.10:
        df_score = 1.0
    elif df <= 0.40:
        df_score = 1.0 - (df - 0.10) / 0.30
    else:
        df_score = 0.0

    # Type bonus
    if pattern.type == "entity":
        type_score = 1.0
    elif pattern.type == "code_pattern":
        type_score = 0.85
    else:  # domain_term
        type_score = 0.6

    # Name quality
    if pattern.type == "code_pattern":
        name_score = 0.8  # regex patterns bypass text heuristics
    else:
        words = pattern.name.lower().split()
        if all(w in _ENTITY_STOPWORDS for w in words):
            name_score = 0.0
        else:
            has_upper = any(c.isupper() for c in pattern.name)
            name_score = 1.0 if has_upper else 0.3
            if len(pattern.name) < 4:
                name_score *= 0.5

    return df_score * 0.5 + type_score * 0.3 + name_score * 0.2


def build_entity_chunk_graph(
    entity_patterns: list[EntityPattern],
    chunks: list[Any],
) -> tuple[dict[str, set[str]], dict[str, list[str]], dict[tuple[str, str], int]]:
    """Single O(E×N) scan: build entity↔chunk bipartite graph and compute DF/quality.

    Mutates ``entity_patterns`` in place, setting ``document_frequency``,
    ``idf_weight``, and ``quality_score`` on each pattern.

    Returns (entity_chunk_index, chunk_entity_index, entity_cooccurrence).
    All indexes are **unfiltered** — consumers apply their own quality thresholds.
    """
    n = len(chunks)
    if n == 0:
        return {}, {}, {}

    entity_chunk_idx: dict[str, set[str]] = {}
    chunk_entity_idx: dict[str, list[str]] = {}

    for pattern in entity_patterns:
        name_lower = pattern.name.lower()
        matched: set[str] = set()

        for chunk in chunks:
            content = (chunk.content if hasattr(chunk, "content") else str(chunk)).lower()
            chunk_hash = getattr(chunk, "hash", str(id(chunk)))

            if pattern.type == "code_pattern":
                try:
                    if re.search(pattern.name, content, re.IGNORECASE):
                        matched.add(chunk_hash)
                        chunk_entity_idx.setdefault(chunk_hash, []).append(pattern.name)
                except re.error:
                    continue
            elif name_lower in content:
                matched.add(chunk_hash)
                chunk_entity_idx.setdefault(chunk_hash, []).append(pattern.name)

        entity_chunk_idx[name_lower] = matched
        count = len(matched)
        pattern.document_frequency = count / n
        pattern.idf_weight = math.log(n / max(count, 1))

    # Compute quality scores (needs DF to be set first)
    for pattern in entity_patterns:
        pattern.quality_score = _score_entity_quality(pattern)

    # Co-occurrence from the full graph
    cooccurrence: dict[tuple[str, str], int] = {}
    entity_names = sorted(
        name for name, hashes in entity_chunk_idx.items() if hashes
    )
    for i, e1 in enumerate(entity_names):
        for e2 in entity_names[i + 1:]:
            overlap = len(entity_chunk_idx[e1] & entity_chunk_idx[e2])
            if overlap > 0:
                cooccurrence[(e1, e2)] = overlap

    return entity_chunk_idx, chunk_entity_idx, cooccurrence


def compute_entity_document_frequency(
    entity_patterns: list[EntityPattern],
    sample_chunks: list[Any],
) -> None:
    """Compute DF and quality score for each entity. Thin wrapper around graph build.

    Mutates ``entity_patterns`` in place, setting ``document_frequency``,
    ``idf_weight``, and ``quality_score`` on each pattern.
    """
    build_entity_chunk_graph(entity_patterns, sample_chunks)


def compute_token_document_frequency(
    sample_chunks: list[Any],
    *,
    min_token_len: int = 3,
) -> tuple[dict[str, int], int]:
    """Compute per-token document frequency across a chunk sample.

    Returns (token_df_dict, sample_size).
    """
    import re

    n = len(sample_chunks)
    if n == 0:
        return {}, 0

    df: dict[str, int] = {}
    for chunk in sample_chunks:
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        unique_tokens = set(re.findall(r"\w+", content.lower()))
        for tok in unique_tokens:
            if len(tok) >= min_token_len:
                df[tok] = df.get(tok, 0) + 1

    return df, n


_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "must",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "from",
        "by",
        "about",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "no",
        "so",
        "if",
        "than",
        "too",
        "very",
        "just",
        "also",
        "only",
        "own",
        "same",
        "that",
        "this",
        "it",
        "its",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "any",
        "up",
        "down",
        "new",
        "old",
        "true",
        "false",
        "null",
        "none",
        "yes",
        "get",
        "set",
        "var",
        "let",
        "const",
        "def",
        "return",
        "class",
        "function",
        "import",
        "from",
        "type",
        "string",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "object",
        "value",
        "key",
        "name",
        "data",
        "item",
        "index",
        "error",
        "result",
        "response",
        "request",
        "input",
        "output",
        "file",
        "path",
        "config",
        "options",
        "args",
        "params",
        "default",
        "test",
        "example",
        "code",
        "note",
        "using",
        "use",
        "used",
        "see",
        "ignoring",
        "views",
        "alpha",
    }
)


def _is_quality_entity(name: str) -> bool:
    """Check if an entity name meets minimum quality bar."""
    if len(name) < 3:
        return False
    if name.lower() in _STOPWORDS:
        return False
    return True


def build_entity_patterns_from_extraction(
    entity_names: list[str],
    code_patterns: dict[str, str],
    domain_terms: list[str],
) -> list[EntityPattern]:
    """Convert raw extraction output into ``EntityPattern`` objects.

    Applies quality heuristics: minimum length, stopword rejection,
    and case-insensitive deduplication.
    """
    seen_lower: set[str] = set()
    patterns: list[EntityPattern] = []

    def _add(name: str, entity_type: str) -> None:
        name = name.strip()
        if not name:
            return
        # Code patterns skip text-quality checks (they're regexes).
        if entity_type != "code_pattern" and not _is_quality_entity(name):
            return
        key = name.lower()
        if key in seen_lower:
            return
        seen_lower.add(key)
        patterns.append(EntityPattern(name=name, type=entity_type))

    for name in entity_names:
        _add(name, "entity")
    for _category, regex_pattern in code_patterns.items():
        _add(regex_pattern, "code_pattern")
    for term in domain_terms:
        _add(term, "domain_term")
    return patterns


def extract_heuristic_entities(
    sample_chunks: list[Any],
) -> list[EntityPattern]:
    """Extract entity candidates directly from corpus text using heuristics.

    Three methods are combined and deduplicated:
    1. TF-IDF top terms — discriminative single tokens.
    2. Capitalized phrase extraction — multi-word proper nouns.
    3. Frequent n-grams (bigrams + trigrams) — recurring domain phrases.

    Because candidates come from the actual text, they match the corpus by
    construction — unlike LLM-invented names that can't be found downstream.
    """
    if not sample_chunks:
        return []

    n = len(sample_chunks)

    # --- Method 1: TF-IDF top terms ---
    token_df, _ = compute_token_document_frequency(sample_chunks, min_token_len=4)
    idf_terms: list[tuple[float, str]] = []
    for token, df in token_df.items():
        df_frac = df / n
        if df_frac < 0.005 or df_frac > 0.30:
            continue
        if token in _STOPWORDS:
            continue
        idf = math.log(n / df)
        idf_terms.append((idf, token))
    idf_terms.sort(reverse=True)
    tfidf_patterns = [
        EntityPattern(name=token, type="domain_term") for _, token in idf_terms[:80]
    ]

    # --- Method 2: Capitalized phrase extraction ---
    cap_phrase_re = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
    phrase_df: dict[str, int] = {}
    for chunk in sample_chunks:
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        seen_in_chunk: set[str] = set()
        for match in cap_phrase_re.finditer(content):
            phrase = match.group()
            if phrase.lower() not in _STOPWORDS and phrase not in seen_in_chunk:
                phrase_df[phrase] = phrase_df.get(phrase, 0) + 1
                seen_in_chunk.add(phrase)
    cap_patterns = [
        EntityPattern(name=phrase, type="entity")
        for phrase, df in phrase_df.items()
        if df >= 2
    ]

    # --- Method 3: Frequent bigrams and trigrams ---
    ngram_df: dict[tuple[str, ...], int] = {}
    for chunk in sample_chunks:
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        tokens = re.findall(r"\w+", content.lower())
        seen_in_chunk: set[tuple[str, ...]] = set()  # type: ignore[assignment]
        for size in (2, 3):
            for i in range(len(tokens) - size + 1):
                ngram = tuple(tokens[i : i + size])
                if ngram not in seen_in_chunk:
                    ngram_df[ngram] = ngram_df.get(ngram, 0) + 1
                    seen_in_chunk.add(ngram)
    ngram_patterns: list[EntityPattern] = []
    for ngram, df in ngram_df.items():
        df_frac = df / n
        if df_frac < 0.01 or df_frac > 0.25:
            continue
        if df < 3:
            continue
        if all(tok in _STOPWORDS for tok in ngram):
            continue
        joined = " ".join(ngram)
        if len(joined) < 6:
            continue
        ngram_patterns.append(EntityPattern(name=joined, type="domain_term"))

    # --- Deduplication ---
    # Priority: "entity" > "domain_term".  First seen wins within a type tier.
    seen_lower: dict[str, str] = {}  # lowercased name -> winning type
    result_map: dict[str, EntityPattern] = {}

    for pattern in tfidf_patterns + ngram_patterns + cap_patterns:
        key = pattern.name.lower()
        existing_type = seen_lower.get(key)
        if existing_type is None:
            seen_lower[key] = pattern.type
            result_map[key] = pattern
        elif existing_type == "domain_term" and pattern.type == "entity":
            # Upgrade to entity type
            seen_lower[key] = "entity"
            result_map[key] = pattern

    return list(result_map.values())


def diverse_profile_sample(
    source: Any,
    corpus_size: int,
    min_chars: int,
    rng: random.Random,
) -> list[Any]:
    """Build a diverse profile sample with stratified + random fill.

    Pool size is proportional to corpus size: 5% of corpus, floor 500, cap 10K.
    Ensures coverage across documents by stratifying on document ID metadata,
    then fills the remaining budget randomly. Degrades gracefully to pure random
    sampling when no document IDs exist.

    Args:
        source: A ``ChunkSource`` instance.
        corpus_size: Total number of chunks in the corpus (from ``source.get_chunk_count()``).
        min_chars: Minimum character count per chunk.
        rng: Seeded random generator for reproducibility.

    Returns:
        List of sampled chunks (up to pool_size).
    """
    pool_size = min(corpus_size, max(500, min(10_000, int(corpus_size * 0.05))))
    oversample = min(pool_size * 3, pool_size + 1000)
    all_chunks = source.sample_chunks(oversample, min_chars=min_chars)
    if not all_chunks:
        return []

    if len(all_chunks) <= pool_size:
        return list(all_chunks)

    # Phase 1: Stratified — ensure coverage across documents
    by_doc: dict[str, list[Any]] = defaultdict(list)
    for chunk in all_chunks:
        doc_id = _get_doc_id(chunk)
        by_doc[doc_id].append(chunk)

    n_docs = len(by_doc)
    stratified_budget = pool_size // 2
    per_doc = max(1, stratified_budget // max(n_docs, 1))

    stratified: list[Any] = []
    for doc_chunks in by_doc.values():
        k = min(per_doc, len(doc_chunks))
        stratified.extend(rng.sample(doc_chunks, k))

    # Phase 2: Random fill from remaining chunks
    selected_ids = {id(c) for c in stratified}
    remaining = [c for c in all_chunks if id(c) not in selected_ids]
    fill_count = min(pool_size - len(stratified), len(remaining))
    if fill_count > 0:
        random_fill = rng.sample(remaining, fill_count)
        stratified.extend(random_fill)

    return stratified[:pool_size]


def detect_search_capabilities(source: Any) -> tuple[set[str], str]:
    """Detect available search modes from a ChunkSource.

    Returns:
        Tuple of (available_modes, best_mode).
    """
    modes: set[str] = {"lexical"}
    try:
        caps = source.get_search_capabilities()
        modes = set(caps.get("modes", {"lexical"}))
    except Exception:
        logger.debug("Could not detect search capabilities, defaulting to lexical")

    if "hybrid" in modes:
        best = "hybrid"
    elif "vector" in modes:
        best = "vector"
    else:
        best = "lexical"

    return modes, best


def _get_doc_id(chunk: Any) -> str:
    """Extract a document identifier from chunk metadata."""
    md: dict[str, Any] = {}
    if hasattr(chunk, "metadata_dict"):
        md = chunk.metadata_dict or {}
    elif hasattr(chunk, "metadata"):
        raw = chunk.metadata
        if isinstance(raw, dict):
            md = raw
        elif isinstance(raw, (tuple, list)):
            md = dict(raw)
    elif isinstance(chunk, dict):
        md = chunk.get("metadata", {})

    for key in ("document_id", "file_name", "file", "source", "doc_id"):
        val = md.get(key)
        if val is not None:
            return str(val)
    return "unknown"


def _get_metadata_dict(chunk: Any) -> dict[str, Any]:
    """Extract metadata dict from any chunk-like object."""
    if hasattr(chunk, "metadata_dict"):
        return chunk.metadata_dict or {}
    if hasattr(chunk, "metadata"):
        raw = chunk.metadata
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, (tuple, list)):
            return dict(raw)
    if isinstance(chunk, dict):
        return chunk.get("metadata", {})
    return {}


_MARKUP_PATTERN = re.compile(r"\[.*?\]\(.*?\)|https?://\S+|<[^>]+>|[|>»›‹«]")
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"was this .{0,20}(useful|helpful)",
        r"click\s+here",
        r"subscribe\s*(to|now)",
        r"table of contents",
        r"back to top",
        r"previous\s*[|/]\s*next",
        r"skip to (content|main|nav)",
        r"copyright|©|all rights reserved",
        r"rate this",
        r"share (this|on)",
        r"follow us",
        r"terms (of|and) (service|use)",
    ]
]


def _compute_answerability(content: str) -> float:
    """Score content answerability using language-agnostic heuristics.

    Composite of 4 sub-signals measuring whether the content contains
    substantive, answerable information vs. boilerplate/navigation/stubs.
    Returns float in [0, 1].
    """
    if not content or len(content) < 20:
        return 0.0

    # Sub-signal A: Sentence density (0.35 weight)
    # Latin enders: only count . ! ? when followed by whitespace or end-of-string.
    # Filters code dots (foo.bar), version numbers (1.2.3), URLs (foo.com).
    # CJK/Devanagari enders: always unambiguous, count all.
    latin_enders = len(re.findall(r"[.!?](?=\s|$)", content))
    cjk_enders = len(re.findall(r"[。！？।]", content))
    ender_count = latin_enders + cjk_enders
    enders_per_1k = ender_count / max(len(content) / 1000, 0.001)
    sentence_density = min(1.0, enders_per_1k / 6.0)

    # Sub-signal B: Markup density — inverted (0.30 weight)
    markup_chars = sum(len(m) for m in _MARKUP_PATTERN.findall(content))
    markup_fraction = markup_chars / len(content)
    markup_score = max(0.0, 1.0 - markup_fraction * 3.0)

    # Sub-signal C: Long-line ratio (0.25 weight)
    lines = [line for line in content.split("\n") if line.strip()]
    if lines:
        long_line_ratio = sum(1 for line in lines if len(line.strip()) > 60) / len(lines)
    else:
        long_line_ratio = 0.0

    # Sub-signal D: Negative pattern penalty (0.10 weight)
    neg_matches = sum(1 for p in _BOILERPLATE_PATTERNS if p.search(content))
    negative_score = max(0.0, 1.0 - neg_matches * 0.33)

    return (
        0.35 * sentence_density
        + 0.30 * markup_score
        + 0.25 * long_line_ratio
        + 0.10 * negative_score
    )


_HEADER_KEYS: frozenset[str] = frozenset({"h1", "h2", "h3", "title", "section_header", "header"})
_DOC_ID_KEYS: frozenset[str] = frozenset({"file_name", "document_id", "source", "file", "doc_id"})
_DATE_KEYS: frozenset[str] = frozenset(
    {
        "date", "timestamp", "created_at", "updated_at", "published_at",
        "modified_at", "date_start", "date_end",
    }
)
_SEQ_KEYS: frozenset[str] = frozenset({"prev_chunk_id", "next_chunk_id"})
_INDEX_KEYS: frozenset[str] = frozenset({"chunk_index", "index", "position"})


def compute_header_prevalence(pool: list[Any]) -> float:
    """Compute fraction of chunks with header metadata. Lightweight, one pass.

    Used before full entity extraction to derive metadata_regime for
    ``select_diverse``. Returns 0.0 for an empty pool.
    """
    if not pool:
        return 0.0
    count = sum(
        1 for chunk in pool if any(_get_metadata_dict(chunk).get(k) for k in _HEADER_KEYS)
    )
    return count / len(pool)


def compute_metadata_census(
    pool: list[Any],
    entity_names: list[str],
    chunk_count: int,
) -> CorpusMetadataCensus:
    """Compute corpus-relative distributions from the profile pool.

    Single pass over *pool* computing metadata presence, content length,
    entity density, and lexical diversity. Uses ``statistics.quantiles``
    for p25/p50/p75. Returns a degenerate census (zeroed percentiles,
    regime="unstructured") if pool has fewer than 5 chunks.

    Args:
        pool: Profile pool chunks.
        entity_names: Extracted entity name strings (used for density).
        chunk_count: Total corpus size (stored on the census).

    Returns:
        A populated ``CorpusMetadataCensus``.
    """
    if len(pool) < 5:
        return CorpusMetadataCensus(
            header_prevalence=0.0,
            doc_id_prevalence=0.0,
            date_prevalence=0.0,
            sequential_prevalence=0.0,
            content_length_p25=0,
            content_length_p50=0,
            content_length_p75=0,
            entity_density_p25=0.0,
            entity_density_p50=0.0,
            entity_density_p75=0.0,
            lexical_diversity_p25=0.0,
            lexical_diversity_p50=0.0,
            lexical_diversity_p75=0.0,
            metadata_regime="unstructured",
            chunk_count=chunk_count,
        )

    entity_names_lower = [name.lower() for name in entity_names]

    header_count = 0
    doc_id_count = 0
    date_count = 0
    sequential_count = 0
    content_lengths: list[int] = []
    entity_densities: list[float] = []
    lexical_diversities: list[float] = []

    for chunk in pool:
        md = _get_metadata_dict(chunk)
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        content_lower = content.lower()

        if any(md.get(k) for k in _HEADER_KEYS):
            header_count += 1
        if any(md.get(k) for k in _DOC_ID_KEYS):
            doc_id_count += 1
        if any(md.get(k) for k in _DATE_KEYS):
            date_count += 1

        has_prev_next = bool(md.get("prev_chunk_id") or md.get("next_chunk_id"))
        has_file_index = bool(
            any(md.get(k) for k in _DOC_ID_KEYS) and any(md.get(k) for k in _INDEX_KEYS)
        )
        if has_prev_next or has_file_index:
            sequential_count += 1

        length = len(content)
        content_lengths.append(length)

        entity_count = sum(1 for name in entity_names_lower if name in content_lower)
        entity_densities.append(entity_count / max(length / 1000, 0.001))

        tokens = re.findall(r"\w+", content_lower)
        total_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        lexical_diversities.append(unique_tokens / max(total_tokens, 1))

    n = len(pool)
    header_prevalence = header_count / n
    doc_id_prevalence = doc_id_count / n
    date_prevalence = date_count / n
    sequential_prevalence = sequential_count / n

    cl_q = statistics.quantiles(content_lengths, n=4)
    ed_q = statistics.quantiles(entity_densities, n=4)
    ld_q = statistics.quantiles(lexical_diversities, n=4)

    if header_prevalence > 0.70:
        regime = "structured"
    elif header_prevalence >= 0.20:
        regime = "mixed"
    elif doc_id_prevalence > 0.5 and date_prevalence > 0.5:
        regime = "mixed"  # no headers but rich metadata (e.g. email)
    else:
        regime = "unstructured"

    return CorpusMetadataCensus(
        header_prevalence=header_prevalence,
        doc_id_prevalence=doc_id_prevalence,
        date_prevalence=date_prevalence,
        sequential_prevalence=sequential_prevalence,
        content_length_p25=int(cl_q[0]),
        content_length_p50=int(cl_q[1]),
        content_length_p75=int(cl_q[2]),
        entity_density_p25=ed_q[0],
        entity_density_p50=ed_q[1],
        entity_density_p75=ed_q[2],
        lexical_diversity_p25=ld_q[0],
        lexical_diversity_p50=ld_q[1],
        lexical_diversity_p75=ld_q[2],
        metadata_regime=regime,
        chunk_count=chunk_count,
    )


def select_diverse(
    pool: list[Any],
    n: int,
    *,
    rng: random.Random,
    stratify_key: Callable[[Any], str] | None = None,
    min_jaccard_distance: float = 0.3,
) -> list[Any]:
    """Select *n* maximally diverse chunks from *pool* using greedy max-diversity.

    Precomputes token sets and seeds with the broadest-vocabulary chunk.
    Each subsequent selection maximises the minimum Jaccard distance to
    already-selected chunks. Candidates below *min_jaccard_distance* from
    any selected chunk are skipped; any remaining budget is filled with
    leftover chunks in arbitrary order.

    When *stratify_key* is provided the pool is grouped by key and
    ``ceil(n / num_groups)`` chunks are selected per group, then the
    result is trimmed or filled to exactly *n*.

    Args:
        pool: Source chunks to select from.
        n: Target number of chunks to return.
        rng: Seeded random generator (used for fill-phase shuffles).
        stratify_key: Optional callable that maps a chunk to a group label.
        min_jaccard_distance: Diversity floor; candidates closer than this
            to any already-selected chunk are skipped in the greedy pass.

    Returns:
        List of selected chunks (may be shorter than *n* if pool is small).
    """
    if not pool or n <= 0:
        return []
    if len(pool) <= n:
        return list(pool)

    def _token_set(chunk: Any) -> frozenset[str]:
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        return frozenset(re.findall(r"\w+", content.lower()))

    def _jaccard_distance(a: frozenset[str], b: frozenset[str]) -> float:
        union = len(a | b)
        if union == 0:
            return 0.0
        return 1.0 - len(a & b) / union

    def _greedy_diverse(candidates: list[Any], k: int) -> list[Any]:
        if not candidates or k <= 0:
            return []
        if len(candidates) <= k:
            return list(candidates)

        # Precompute token sets — not recomputed inside the loop
        token_sets = [_token_set(c) for c in candidates]

        # Seed with the chunk having the most unique tokens
        seed_idx = max(range(len(candidates)), key=lambda i: len(token_sets[i]))
        selected: list[Any] = [candidates[seed_idx]]
        selected_sets: list[frozenset[str]] = [token_sets[seed_idx]]
        remaining: set[int] = set(range(len(candidates)))
        remaining.discard(seed_idx)

        while len(selected) < k and remaining:
            best_idx: int | None = None
            best_min_dist = -1.0

            for ri in remaining:
                min_dist = min(_jaccard_distance(token_sets[ri], s) for s in selected_sets)
                if min_dist < min_jaccard_distance:
                    continue
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = ri

            if best_idx is None:
                break  # All remaining candidates below diversity threshold

            selected.append(candidates[best_idx])
            selected_sets.append(token_sets[best_idx])
            remaining.discard(best_idx)

        # Fill remaining budget from leftover chunks in arbitrary order
        if len(selected) < k and remaining:
            fill = [candidates[i] for i in remaining]
            rng.shuffle(fill)
            selected.extend(fill[: k - len(selected)])

        return selected

    if stratify_key is not None:
        by_key: dict[str, list[Any]] = defaultdict(list)
        for chunk in pool:
            by_key[stratify_key(chunk)].append(chunk)

        num_groups = len(by_key)
        per_group = math.ceil(n / num_groups)

        result: list[Any] = []
        used_ids: set[int] = set()
        for group_chunks in by_key.values():
            for c in _greedy_diverse(group_chunks, per_group):
                if id(c) not in used_ids:
                    result.append(c)
                    used_ids.add(id(c))

        if len(result) > n:
            return result[:n]
        if len(result) < n:
            remaining_chunks = [c for c in pool if id(c) not in used_ids]
            rng.shuffle(remaining_chunks)
            result.extend(remaining_chunks[: n - len(result)])
        return result

    return _greedy_diverse(pool, n)


def _percentile_normalize(value: float, p25: float, p50: float, p75: float) -> float:
    """Map *value* to [0, 1] using census percentile boundaries.

    Division-by-zero is guarded with ``max(..., 1e-9)`` so degenerate censuses
    (all-zero percentiles) return a safe value rather than crashing.
    """
    eps = 1e-9
    if value <= p25:
        return value / max(p25, eps) * 0.25
    if value <= p50:
        return 0.25 + (value - p25) / max(p50 - p25, eps) * 0.25
    if value <= p75:
        return 0.50 + (value - p50) / max(p75 - p50, eps) * 0.25
    return min(1.0, 0.75 + (value - p75) / max(p75, eps) * 0.25)


def compute_chunk_suitability(
    chunk: Any,
    census: CorpusMetadataCensus,
    profile: CorpusProfile,
) -> float:
    """Score a chunk's suitability for QA generation relative to corpus distributions.

    Combines five signals — metadata presence, entity density, lexical diversity,
    content length, and TF-IDF distinctiveness — using regime-dependent weight
    vectors. Returns a float in [0, 1].

    Args:
        chunk: The chunk to score.
        census: Precomputed corpus metadata census for normalization.
        profile: Corpus profile supplying entity names and TF-IDF data.

    Returns:
        Suitability score in [0, 1].
    """
    md = _get_metadata_dict(chunk)
    content = chunk.content if hasattr(chunk, "content") else str(chunk)
    content_lower = content.lower()

    # 1. Metadata richness: count of distinct metadata categories [0, 1]
    richness = 0
    if any(md.get(k) for k in _HEADER_KEYS):
        richness += 1
    if any(md.get(k) for k in _DOC_ID_KEYS):
        richness += 1
    if any(md.get(k) for k in _DATE_KEYS):
        richness += 1
    has_seq = bool(md.get("prev_chunk_id") or md.get("next_chunk_id"))
    has_file_idx = bool(
        any(md.get(k) for k in _DOC_ID_KEYS) and any(md.get(k) for k in _INDEX_KEYS)
    )
    if has_seq or has_file_idx:
        richness += 1
    metadata_score = richness / 4.0

    # 2. Entity density: quality-weighted entities per 1000 chars, percentile-normalised
    chunk_hash = getattr(chunk, "hash", str(id(chunk)))
    eq_map = profile.entity_quality_map
    if chunk_hash in profile.chunk_entity_index:
        # Fast path: read from pre-computed entity-chunk graph
        chunk_entities = profile.chunk_entity_index[chunk_hash]
        quality_weighted_count = sum(eq_map.get(e.lower(), 0.5) for e in chunk_entities)
    else:
        # Fallback: inline scan for chunks not in the graph (API backends, on-the-fly)
        quality_weighted_count = sum(
            score for name, score in eq_map.items() if name in content_lower
        )
    raw_entity_density = quality_weighted_count / max(len(content) / 1000, 0.001)
    entity_density = _percentile_normalize(
        raw_entity_density,
        census.entity_density_p25,
        census.entity_density_p50,
        census.entity_density_p75,
    )

    # 3. Lexical diversity: unique tokens / total tokens, percentile-normalised
    tokens = re.findall(r"\w+", content_lower)
    raw_diversity = len(set(tokens)) / max(len(tokens), 1)
    lexical_diversity = _percentile_normalize(
        raw_diversity,
        census.lexical_diversity_p25,
        census.lexical_diversity_p50,
        census.lexical_diversity_p75,
    )

    # 4. Content length: percentile-normalised (saturating curve handles outliers)
    raw_length = float(len(content))
    content_length_score = _percentile_normalize(
        raw_length,
        float(census.content_length_p25),
        float(census.content_length_p50),
        float(census.content_length_p75),
    )

    # 5. TF-IDF distinctiveness: mean IDF of top-5 keyphrases
    keyphrases = profile.get_tfidf_keyphrases(content, top_n=5)
    if keyphrases and profile.token_df_sample_size > 0:
        n_docs = profile.token_df_sample_size
        idfs = [
            math.log(n_docs / max(profile.token_document_frequency.get(kp, 1), 1))
            for kp in keyphrases
        ]
        mean_idf = sum(idfs) / len(idfs)
        max_idf = math.log(max(profile.token_df_sample_size, 2))
        tfidf_distinctiveness = min(1.0, mean_idf / max(max_idf, 0.001))
    else:
        tfidf_distinctiveness = 0.0

    # 6. Answerability: content vs. boilerplate/navigation/stubs
    answerability = _compute_answerability(content)

    # Regime-dependent weight vectors (6 signals)
    regime = census.metadata_regime
    if regime == "structured":
        w_meta, w_ent, w_lex, w_len, w_tfidf, w_ans = (
            0.20, 0.20, 0.10, 0.10, 0.15, 0.25
        )
    elif regime == "mixed" and census.header_prevalence < 0.20:
        # Email-like: "mixed" via date/doc_id, not headers.
        # Answerability is likely uniform for uniform-prose corpora.
        w_meta, w_ent, w_lex, w_len, w_tfidf, w_ans = (
            0.10, 0.25, 0.20, 0.10, 0.20, 0.15
        )
    elif regime == "mixed":
        w_meta, w_ent, w_lex, w_len, w_tfidf, w_ans = (
            0.10, 0.20, 0.15, 0.10, 0.15, 0.30
        )
    else:  # unstructured
        w_meta, w_ent, w_lex, w_len, w_tfidf, w_ans = (
            0.00, 0.25, 0.20, 0.10, 0.20, 0.25
        )

    return (
        w_meta * metadata_score
        + w_ent * entity_density
        + w_lex * lexical_diversity
        + w_len * content_length_score
        + w_tfidf * tfidf_distinctiveness
        + w_ans * answerability
    )
