"""Shared anchor/BM25 helper utilities for Cgft QA generation."""

from __future__ import annotations

from typing import Any, Callable

from .anchor_selector import AnchorBundle


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
) -> AnchorBundle | tuple[AnchorBundle, list[str], list[dict[str, Any]]]:
    """Select an anchor bundle and attach BM25-related chunks as structural hints."""
    bundle = selector.select(primary_chunk, corpus_pool, qa_type=qa_type)
    queries = generate_bm25_queries(primary_chunk, bm25_enrichment_queries)
    bm25_related = source.search_related(primary_chunk, queries, top_k=bm25_enrichment_top_k)
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
