"""First-class corpus profile for threading context across pipeline stages.

``CorpusProfile`` consolidates the outputs of stages 2-3 (corpus summary,
entity extraction, capability detection) into a single mutable object.
The linker enriches it during linking; the generator reads the enriched
version for prompt conditioning.
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
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

    # --- Convenience ---

    @property
    def corpus_queries_bulleted(self) -> str:
        """Queries formatted for prompt templates."""
        return "\n".join(f"- {q}" for q in self.corpus_queries)

    def get_discriminative_entities(
        self,
        max_df: float | None = None,
    ) -> list[EntityPattern]:
        """Return entities below the ubiquity threshold (discriminative)."""
        threshold = max_df if max_df is not None else self.ubiquitous_df_threshold
        return [e for e in self.entity_patterns if e.document_frequency < threshold]

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

        scored: list[tuple[float, str]] = []
        for tok, count in tf.items():
            if len(tok) < min_token_len:
                continue
            df = self.token_document_frequency.get(tok)
            if df is None:
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


def compute_entity_document_frequency(
    entity_patterns: list[EntityPattern],
    sample_chunks: list[Any],
) -> None:
    """Compute document frequency for each entity against a chunk sample.

    Mutates ``entity_patterns`` in place, setting ``document_frequency``
    and ``idf_weight`` on each pattern.
    """
    n = len(sample_chunks)
    if n == 0:
        return

    chunk_contents = [
        (chunk.content if hasattr(chunk, "content") else str(chunk)).lower()
        for chunk in sample_chunks
    ]

    for pattern in entity_patterns:
        name_lower = pattern.name.lower()
        count = sum(1 for content in chunk_contents if name_lower in content)
        pattern.document_frequency = count / n
        pattern.idf_weight = math.log(n / max(count, 1))


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


def build_entity_patterns_from_extraction(
    entity_names: list[str],
    code_patterns: dict[str, str],
    domain_terms: list[str],
) -> list[EntityPattern]:
    """Convert raw extraction output into ``EntityPattern`` objects."""
    patterns: list[EntityPattern] = []
    for name in entity_names:
        if name.strip():
            patterns.append(EntityPattern(name=name.strip(), type="entity"))
    for _category, regex_pattern in code_patterns.items():
        if regex_pattern.strip():
            patterns.append(EntityPattern(name=regex_pattern.strip(), type="code_pattern"))
    for term in domain_terms:
        if term.strip():
            patterns.append(EntityPattern(name=term.strip(), type="domain_term"))
    return patterns


def diverse_profile_sample(
    source: Any,
    sample_size: int,
    min_chars: int,
    rng: random.Random,
) -> list[Any]:
    """Build a diverse profile sample with stratified + random fill.

    Ensures coverage across documents by stratifying on document ID
    metadata, then fills the remaining budget randomly.  Degrades
    gracefully to pure random sampling when no document IDs exist.

    Args:
        source: A ``ChunkSource`` instance.
        sample_size: Target number of chunks in the sample.
        min_chars: Minimum character count per chunk.
        rng: Seeded random generator for reproducibility.

    Returns:
        List of sampled chunks (up to *sample_size*).
    """
    oversample = min(sample_size * 3, sample_size + 1000)
    all_chunks = source.sample_chunks(oversample, min_chars=min_chars)
    if not all_chunks:
        return []

    if len(all_chunks) <= sample_size:
        return list(all_chunks)

    # Phase 1: Stratified — ensure coverage across documents
    by_doc: dict[str, list[Any]] = defaultdict(list)
    for chunk in all_chunks:
        doc_id = _get_doc_id(chunk)
        by_doc[doc_id].append(chunk)

    n_docs = len(by_doc)
    stratified_budget = sample_size // 2
    per_doc = max(1, stratified_budget // max(n_docs, 1))

    stratified: list[Any] = []
    for doc_chunks in by_doc.values():
        k = min(per_doc, len(doc_chunks))
        stratified.extend(rng.sample(doc_chunks, k))

    # Phase 2: Random fill from remaining chunks
    selected_ids = {id(c) for c in stratified}
    remaining = [c for c in all_chunks if id(c) not in selected_ids]
    fill_count = min(sample_size - len(stratified), len(remaining))
    if fill_count > 0:
        random_fill = rng.sample(remaining, fill_count)
        stratified.extend(random_fill)

    return stratified[:sample_size]


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
