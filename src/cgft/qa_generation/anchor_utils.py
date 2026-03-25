"""Shared anchor/search helper utilities for Cgft QA generation."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .anchor_selector import AnchorBundle

if TYPE_CHECKING:
    from .cgft_models import EntityExtractionConfig


def best_search_mode(source: Any) -> str:
    """Return the best search mode for *source*.

    Checks ``get_search_capabilities()`` when available. Falls back to
    ``"lexical"`` for backward compatibility.
    """
    if not hasattr(source, "get_search_capabilities"):
        return "lexical"
    try:
        caps = source.get_search_capabilities()
        modes = caps.get("modes", set())
    except Exception:
        return "lexical"
    if "hybrid" in modes:
        return "hybrid"
    if "vector" in modes:
        return "vector"
    return "lexical"


def generate_search_queries(
    chunk: Any,
    n: int = 3,
    *,
    source: Any | None = None,
    entity_extraction: Any | None = None,
) -> list[str]:
    """Generate search queries appropriate for the source's best mode.

    Delegates to ``generate_bm25_queries_from_extraction`` /
    ``generate_bm25_queries`` for lexical/hybrid backends, and to
    ``generate_vector_queries`` for vector-only backends.

    This is the preferred entry point; callers that don't have a
    ``source`` reference can still call the lower-level functions
    directly.
    """
    mode = best_search_mode(source) if source else "lexical"

    if entity_extraction is not None:
        queries = generate_bm25_queries_from_extraction(chunk, entity_extraction, n)
        if queries:
            return queries

    if mode == "vector":
        return generate_vector_queries(chunk, n)
    return generate_bm25_queries(chunk, n)


def generate_bm25_queries(chunk: Any, n: int = 3) -> list[str]:
    """Generate deterministic BM25 query candidates from chunk metadata/content."""
    if n <= 0:
        return []

    queries: list[str] = []

    meta: dict[str, Any] = {}
    if hasattr(chunk, "metadata_dict"):
        meta = dict(getattr(chunk, "metadata_dict") or {})
    elif hasattr(chunk, "metadata"):
        raw_meta = getattr(chunk, "metadata")
        if isinstance(raw_meta, dict):
            meta = dict(raw_meta)
        else:
            try:
                meta = dict(raw_meta or {})
            except Exception:
                meta = {}

    for key in ("h1", "h2", "h3", "header", "section_header", "title"):
        value = meta.get(key)
        if value:
            query = str(value).strip()
            if query and query not in queries:
                queries.append(query)
        if len(queries) >= n:
            return queries[:n]

    content = chunk.content if hasattr(chunk, "content") else str(chunk)
    first_sentence = str(content).split(".")[0].strip()
    if first_sentence and len(first_sentence) > 10 and first_sentence not in queries:
        queries.append(first_sentence[:200])

    if len(queries) < n:
        excerpt = str(content)[:150].strip()
        if excerpt and excerpt not in queries:
            queries.append(excerpt)

    return queries[:n]


def generate_bm25_queries_from_extraction(
    chunk: Any,
    extraction_config: EntityExtractionConfig,
    n: int = 3,
) -> list[str]:
    """Generate BM25 queries from LLM-extracted entity patterns.

    Falls back gracefully: returns an empty list if nothing matches, so callers
    can fall back to ``generate_bm25_queries``.
    """
    if n <= 0:
        return []

    content = chunk.content if hasattr(chunk, "content") else str(chunk)
    content_lower = content.lower()

    extracted: list[tuple[str, str]] = []

    for entity in extraction_config.entity_names or []:
        if entity and entity.lower() in content_lower:
            extracted.append(("entity", entity))

    for pattern_name, pattern in (extraction_config.code_patterns or {}).items():
        try:
            matches = re.findall(pattern, content)
            for match in matches[:3]:
                token = match[0] if isinstance(match, tuple) else match
                token = str(token).strip()
                if token:
                    extracted.append((pattern_name, token))
        except re.error:
            continue

    for term in extraction_config.domain_terms or []:
        if term and term.lower() in content_lower:
            extracted.append(("domain", term))

    templates = extraction_config.query_templates or ["{entity}"]
    queries: list[str] = []
    for _entity_type, entity in extracted:
        for template in templates:
            try:
                query = template.format(entity=entity)
            except (KeyError, ValueError):
                query = entity
            if query and query not in queries:
                queries.append(query)
            if len(queries) >= n:
                return queries

    # Fallback: use extracted entities directly as queries
    for _entity_type, entity in extracted:
        if entity not in queries:
            queries.append(entity)
        if len(queries) >= n:
            break

    return queries[:n]


def generate_vector_queries(chunk: Any, n: int = 3) -> list[str]:
    """Generate natural-language queries suited for vector/embedding search.

    Vector search benefits from full-sentence queries rather than
    keyword-focused ones, because the embedding model captures semantic
    meaning.
    """
    if n <= 0:
        return []

    queries: list[str] = []

    meta: dict[str, Any] = {}
    if hasattr(chunk, "metadata_dict"):
        meta = dict(getattr(chunk, "metadata_dict") or {})
    elif hasattr(chunk, "metadata"):
        raw_meta = getattr(chunk, "metadata")
        if isinstance(raw_meta, dict):
            meta = dict(raw_meta)
        else:
            try:
                meta = dict(raw_meta or {})
            except Exception:
                meta = {}

    # Build a natural-language query from header hierarchy
    headers = []
    for key in ("h1", "h2", "h3", "header", "section_header", "title"):
        value = meta.get(key)
        if value:
            headers.append(str(value).strip())
    if headers:
        queries.append(" - ".join(headers))

    content = chunk.content if hasattr(chunk, "content") else str(chunk)

    # Use first two sentences as a natural-language query
    sentences = [s.strip() for s in str(content).split(".") if s.strip()]
    if sentences:
        first_query = ". ".join(sentences[:2]).strip()
        if first_query and first_query not in queries:
            queries.append(first_query[:300])

    # Use a mid-section excerpt for diversity
    if len(content) > 200:
        mid = len(content) // 2
        excerpt = content[mid - 100 : mid + 100].strip()
        if excerpt and excerpt not in queries:
            queries.append(excerpt)

    return queries[:n]


def _get_chunk_file(chunk: Any) -> str:
    """Extract the file metadata string from a chunk, returning empty string if absent."""
    if hasattr(chunk, "get_metadata"):
        return str(chunk.get_metadata("file", "") or "")
    meta: dict[str, Any] = {}
    if hasattr(chunk, "metadata_dict"):
        meta = dict(getattr(chunk, "metadata_dict") or {})
    elif hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        meta = chunk.metadata
    return str(meta.get("file", "") or "")


def select_anchor_bundle_with_enrichment(
    *,
    selector: Any,
    primary_chunk: Any,
    corpus_pool: list[Any],
    source: Any,
    bm25_enrichment_queries: int,
    bm25_enrichment_top_k: int,
    max_related_refs: int = 3,
    qa_type: str | None = None,
    include_search_payload: bool = False,
    prebuilt_queries: list[str] | None = None,
    filter_same_file: bool = True,
    search_mode: str | None = None,
) -> AnchorBundle | tuple[AnchorBundle, list[str], list[dict[str, Any]]]:
    """Select an anchor bundle and attach related chunks as structural hints."""
    bundle = selector.select(primary_chunk, corpus_pool, qa_type=qa_type)
    queries = (
        prebuilt_queries
        if prebuilt_queries is not None
        else generate_search_queries(primary_chunk, bm25_enrichment_queries, source=source)
    )
    bm25_related = source.search_related(
        primary_chunk,
        queries,
        top_k=bm25_enrichment_top_k,
        mode=search_mode,
    )
    if filter_same_file:
        primary_file = _get_chunk_file(primary_chunk)
        if primary_file:
            bm25_related = [
                row
                for row in bm25_related
                if row.get("chunk") is not None and _get_chunk_file(row["chunk"]) != primary_file
            ]
    bundle.structural_hints["bm25_related"] = [
        row["chunk"] for row in bm25_related[:max_related_refs] if row.get("chunk") is not None
    ]
    if include_search_payload:
        return bundle, queries, bm25_related
    return bundle


def normalize_ref_id(text: str) -> str:
    """Normalize a reference identifier for case/path-insensitive matching."""
    ref = str(text or "").strip().lower()
    if not ref:
        return ""
    return ref.replace("\\", "/")


def extract_anchor_ref_ids(
    bundle: AnchorBundle,
    *,
    ref_id_fn: Callable[[Any], str],
) -> list[str]:
    """Collect unique reference IDs from an anchor bundle using callsite ID logic."""
    refs = [ref_id_fn(bundle.primary_chunk)]
    refs.extend(ref_id_fn(c) for c in bundle.secondary_chunks)
    refs.extend(ref_id_fn(c) for c in bundle.structural_hints.get("bm25_related", []))

    out: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        norm = normalize_ref_id(ref)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(ref)
    return out
