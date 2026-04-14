"""Quality metrics for QA datasets.

Each metric implements the ``Metric`` protocol and can be computed
independently.  Metrics that only need the JSONL data work standalone;
metrics requiring an LLM or corpus backend are clearly separated.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MetricResult:
    """Output of a single metric computation.

    Attributes:
        name: Machine-readable metric identifier.
        value: Primary scalar value (rate, score, etc.).
        details: Arbitrary nested diagnostics for drill-down.
    """

    name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Metric(Protocol):
    """Computes a quality metric over a list of QA items."""

    name: str

    def compute(self, items: list[dict[str, Any]]) -> MetricResult:
        """Return a ``MetricResult`` for *items* (JSONL row dicts)."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _tfidf_vectors(docs: list[list[str]]) -> list[dict[str, float]]:
    """Build sparse TF-IDF vectors for a list of tokenized documents."""
    n_docs = len(docs)
    df: Counter[str] = Counter()
    for tokens in docs:
        df.update(set(tokens))

    idf = {term: math.log(n_docs / count) for term, count in df.items() if count < n_docs}

    vectors: list[dict[str, float]] = []
    for tokens in docs:
        tf = Counter(tokens)
        vec = {}
        for term, count in tf.items():
            if term in idf:
                vec[term] = count * idf[term]
        vectors.append(vec)
    return vectors


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return dot / (norm_a * norm_b)


def _tfidf_cosine(docs: list[list[str]]) -> float:
    """Average pairwise cosine similarity using TF-IDF vectors."""
    if len(docs) < 2:
        return 0.0

    vectors = _tfidf_vectors(docs)

    # Pairwise cosine on a sample to bound cost
    max_pairs = 500
    n = len(vectors)
    total_pairs = n * (n - 1) // 2
    step = max(1, total_pairs // max_pairs)

    cosine_sum = 0.0
    pair_count = 0
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pair_idx % step == 0:
                cosine_sum += _cosine_sim(vectors[i], vectors[j])
                pair_count += 1
            pair_idx += 1

    return cosine_sum / pair_count if pair_count else 0.0


def _find_similar_pairs(
    docs: list[list[str]],
    originals: list[str],
    threshold: float,
    *,
    max_pairs: int = 500,
) -> list[dict[str, Any]]:
    """Find document pairs with TF-IDF cosine similarity above *threshold*.

    Uses step-sampling for large datasets, so results are approximate.
    """
    if len(docs) < 2:
        return []

    vectors = _tfidf_vectors(docs)
    n = len(vectors)
    total_pairs = n * (n - 1) // 2
    step = max(1, total_pairs // max_pairs)

    pairs: list[dict[str, Any]] = []
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pair_idx % step == 0:
                sim = _cosine_sim(vectors[i], vectors[j])
                if sim >= threshold:
                    pairs.append(
                        {
                            "i": i,
                            "j": j,
                            "similarity": round(sim, 4),
                            "text_a": originals[i][:80],
                            "text_b": originals[j][:80],
                        }
                    )
            pair_idx += 1

    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    return pairs


def _gini_coefficient(values: list[int]) -> float:
    """Gini coefficient for a list of non-negative integer values."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
    return cumulative / (n * total)


# Naturalness helpers

_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "if",
        "then",
        "can",
        "could",
        "should",
        "would",
        "will",
        "not",
        "no",
        "my",
        "your",
        "i",
        "you",
        "we",
        "they",
        "me",
        "us",
        "after",
        "before",
        "between",
        "during",
        "without",
        "about",
    }
)

_INTENT_TOKENS = frozenset(
    {
        "setup",
        "configure",
        "install",
        "use",
        "create",
        "fix",
        "debug",
        "compare",
        "migrate",
        "enable",
        "disable",
        "deploy",
        "connect",
        "add",
        "remove",
        "update",
        "troubleshoot",
        "error",
        "issue",
        "how",
        "why",
        "what",
        "when",
        "where",
        "which",
        "implement",
        "integrate",
        "build",
        "run",
        "start",
        "stop",
        "test",
        "monitor",
        "set",
        "get",
        "change",
        "switch",
        "convert",
        "export",
        "import",
        "check",
        "verify",
        "confirm",
        "ensure",
        "resolve",
        "prevent",
    }
)


# ---------------------------------------------------------------------------
# Concrete metrics (JSONL-only, no external deps)
# ---------------------------------------------------------------------------


class DiversityMetric:
    """Measures question diversity via n-gram overlap, TF-IDF similarity, and
    distribution entropy across QA types and styles."""

    name = "diversity"

    def compute(self, items: list[dict[str, Any]]) -> MetricResult:
        if not items:
            return MetricResult(name=self.name, value=0.0, details={"error": "no items"})

        questions = [str(item.get("question", "")) for item in items]
        tokenized = [_tokenize(q) for q in questions]

        # 1. Pairwise bigram Jaccard (sampled if large)
        bigram_sets = [set(_ngrams(t, 2)) for t in tokenized]
        max_pairs = 500
        n = len(bigram_sets)
        total_pairs = n * (n - 1) // 2
        step = max(1, total_pairs // max_pairs)

        jaccard_sum = 0.0
        pair_count = 0
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if pair_idx % step == 0:
                    jaccard_sum += _jaccard(bigram_sets[i], bigram_sets[j])
                    pair_count += 1
                pair_idx += 1

        avg_bigram_jaccard = jaccard_sum / pair_count if pair_count else 0.0

        # 2. TF-IDF cosine similarity
        avg_tfidf_cosine = _tfidf_cosine(tokenized)

        # 3. QA type distribution entropy
        type_counts = Counter(str(item.get("qa_type", "unknown")) for item in items)
        type_entropy = _entropy(type_counts, len(items))

        # 4. Style distribution entropy
        style_counts = Counter(
            str(item.get("style_observed", "") or item.get("style_target", "unknown"))
            for item in items
        )
        style_entropy = _entropy(style_counts, len(items))

        # Composite diversity score: low overlap = high diversity
        # Invert similarities so higher = more diverse
        diversity_score = 1.0 - (0.5 * avg_bigram_jaccard + 0.5 * avg_tfidf_cosine)

        return MetricResult(
            name=self.name,
            value=round(diversity_score, 4),
            details={
                "avg_bigram_jaccard": round(avg_bigram_jaccard, 4),
                "avg_tfidf_cosine": round(avg_tfidf_cosine, 4),
                "qa_type_entropy": round(type_entropy, 4),
                "qa_type_distribution": dict(type_counts),
                "style_entropy": round(style_entropy, 4),
                "style_distribution": dict(style_counts),
                "num_items": len(items),
                "num_pairs_sampled": pair_count,
            },
        )


class LexicalShortcutMetric:
    """Detects QA pairs where the answer appears verbatim in a single
    reference chunk, indicating the question can be trivially answered
    by string matching without reasoning."""

    name = "lexical_shortcut"

    def __init__(self, *, min_answer_tokens: int = 3) -> None:
        self.min_answer_tokens = min_answer_tokens

    def compute(self, items: list[dict[str, Any]]) -> MetricResult:
        if not items:
            return MetricResult(name=self.name, value=0.0, details={"error": "no items"})

        shortcut_count = 0
        shortcut_items: list[dict[str, Any]] = []

        for item in items:
            answer = str(item.get("answer", "")).strip().lower()
            answer_tokens = _tokenize(answer)
            if len(answer_tokens) < self.min_answer_tokens:
                continue

            ref_chunks = item.get("reference_chunks", [])
            for chunk in ref_chunks:
                content = str(chunk.get("content", "")).lower()
                if answer in content:
                    shortcut_count += 1
                    shortcut_items.append(
                        {
                            "question": str(item.get("question", ""))[:100],
                            "answer_preview": answer[:80],
                            "chunk_id": chunk.get("id", ""),
                        }
                    )
                    break

        rate = shortcut_count / len(items)
        return MetricResult(
            name=self.name,
            value=round(rate, 4),
            details={
                "shortcut_count": shortcut_count,
                "total_items": len(items),
                "min_answer_tokens": self.min_answer_tokens,
                "examples": shortcut_items[:5],
            },
        )


class HopCountDistributionMetric:
    """Reports the distribution of hop counts and QA types in the dataset."""

    name = "hop_count_distribution"

    def compute(self, items: list[dict[str, Any]]) -> MetricResult:
        if not items:
            return MetricResult(name=self.name, value=0.0, details={"error": "no items"})

        hop_counts: Counter[str] = Counter()
        qa_types: Counter[str] = Counter()
        chunk_counts: list[int] = []

        for item in items:
            qa_type = str(item.get("qa_type", "unknown"))
            qa_types[qa_type] += 1

            ref_chunks = item.get("reference_chunks", [])
            n_chunks = len(ref_chunks)
            chunk_counts.append(n_chunks)

            hop_count = item.get("min_hop_count")
            if hop_count is not None:
                hop_counts[str(hop_count)] += 1
            else:
                hop_counts[str(n_chunks)] += 1

        avg_chunks = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0.0

        return MetricResult(
            name=self.name,
            value=round(avg_chunks, 2),
            details={
                "hop_count_distribution": dict(hop_counts),
                "qa_type_distribution": dict(qa_types),
                "avg_reference_chunks": round(avg_chunks, 2),
                "min_reference_chunks": min(chunk_counts) if chunk_counts else 0,
                "max_reference_chunks": max(chunk_counts) if chunk_counts else 0,
                "total_items": len(items),
            },
        )


class AnswerLengthMetric:
    """Reports answer length statistics — unusually short or long answers
    may indicate generation issues."""

    name = "answer_length"

    def compute(self, items: list[dict[str, Any]]) -> MetricResult:
        if not items:
            return MetricResult(name=self.name, value=0.0, details={"error": "no items"})

        lengths = [len(str(item.get("answer", ""))) for item in items]
        avg_len = sum(lengths) / len(lengths)

        buckets = {"<50": 0, "50-200": 0, "200-500": 0, "500-1000": 0, ">1000": 0}
        for length in lengths:
            if length < 50:
                buckets["<50"] += 1
            elif length < 200:
                buckets["50-200"] += 1
            elif length < 500:
                buckets["200-500"] += 1
            elif length < 1000:
                buckets["500-1000"] += 1
            else:
                buckets[">1000"] += 1

        return MetricResult(
            name=self.name,
            value=round(avg_len, 1),
            details={
                "avg_chars": round(avg_len, 1),
                "min_chars": min(lengths),
                "max_chars": max(lengths),
                "length_distribution": buckets,
                "total_items": len(items),
            },
        )


class NaturalnessMetric:
    """Detects degenerate keyword-style questions (entity stuffing, missing
    search intent, unrealistic length) while accepting well-formed keyword
    queries that represent realistic user searches."""

    name = "naturalness"

    def __init__(
        self,
        *,
        max_keyword_tokens: int = 8,
        min_function_ratio: float = 0.05,
        degeneracy_threshold: float = 0.5,
    ) -> None:
        self.max_keyword_tokens = max_keyword_tokens
        self.min_function_ratio = min_function_ratio
        self.degeneracy_threshold = degeneracy_threshold

    def compute(self, items: list[dict[str, Any]]) -> MetricResult:
        if not items:
            return MetricResult(name=self.name, value=0.0, details={"error": "no items"})

        keyword_count = 0
        degenerate_count = 0
        length_flagged = 0
        stuffing_flagged = 0
        intent_flagged = 0
        examples: list[dict[str, Any]] = []

        for item in items:
            question = str(item.get("question", ""))
            if not self._is_keyword_style(item, question):
                continue

            keyword_count += 1
            tokens = _tokenize(question)
            n_tokens = len(tokens)
            if n_tokens == 0:
                continue

            penalties = 0.0

            # Signal 1: excessive length for keyword style
            if n_tokens > self.max_keyword_tokens:
                penalties += 0.4
                length_flagged += 1

            # Signal 2: function word starvation (entity stuffing)
            func_count = sum(1 for t in tokens if t in _FUNCTION_WORDS)
            func_ratio = func_count / n_tokens
            if func_ratio < self.min_function_ratio and n_tokens > 5:
                penalties += 0.3
                stuffing_flagged += 1

            # Signal 3: missing search intent
            has_intent = any(t in _INTENT_TOKENS for t in tokens)
            if not has_intent:
                penalties += 0.3
                intent_flagged += 1

            if penalties >= self.degeneracy_threshold:
                degenerate_count += 1
                if len(examples) < 5:
                    examples.append(
                        {
                            "question": question[:120],
                            "tokens": n_tokens,
                            "function_ratio": round(func_ratio, 3),
                            "has_intent": has_intent,
                            "penalty_score": round(penalties, 2),
                        }
                    )

        rate = degenerate_count / len(items) if items else 0.0
        return MetricResult(
            name=self.name,
            value=round(rate, 4),
            details={
                "degenerate_count": degenerate_count,
                "keyword_count": keyword_count,
                "total_items": len(items),
                "length_flagged": length_flagged,
                "stuffing_flagged": stuffing_flagged,
                "intent_flagged": intent_flagged,
                "examples": examples,
            },
        )

    @staticmethod
    def _is_keyword_style(item: dict[str, Any], question: str) -> bool:
        """Detect keyword-style questions via metadata or heuristic."""
        style = item.get("style_observed") or item.get("style_target", "")
        if style == "keyword":
            return True
        if style in ("natural", "expert"):
            return False
        # Heuristic fallback: no question mark, no interrogative opener
        if "?" in question:
            return False
        lower = question.lower().lstrip()
        interrogatives = ("how", "what", "why", "where", "when", "which", "who", "can", "does")
        if any(lower.startswith(w) for w in interrogatives):
            return False
        tokens = _tokenize(question)
        if len(tokens) <= 10:
            return True
        return False


class AnswerDiversityMetric:
    """Measures pairwise answer similarity — high similarity indicates
    the dataset produces repetitive or generic answers."""

    name = "answer_diversity"

    def __init__(self, *, similarity_threshold: float = 0.7) -> None:
        self.similarity_threshold = similarity_threshold

    def compute(self, items: list[dict[str, Any]]) -> MetricResult:
        if not items:
            return MetricResult(name=self.name, value=0.0, details={"error": "no items"})

        answers = [str(item.get("answer", "")) for item in items]
        tokenized = [_tokenize(a) for a in answers]

        avg_cosine = _tfidf_cosine(tokenized)

        high_sim_pairs = _find_similar_pairs(tokenized, answers, self.similarity_threshold)

        return MetricResult(
            name=self.name,
            value=round(avg_cosine, 4),
            details={
                "avg_tfidf_cosine": round(avg_cosine, 4),
                "high_similarity_pairs": len(high_sim_pairs),
                "similarity_threshold": self.similarity_threshold,
                "examples": high_sim_pairs[:5],
                "total_items": len(items),
            },
        )


class ReferenceCoherenceMetric:
    """Measures topical relatedness of reference chunks within each multi-hop
    QA pair.  Low coherence indicates kitchen-sink questions where chunks
    cover unrelated topics."""

    name = "reference_coherence"

    def __init__(self, *, incoherence_threshold: float = 0.05) -> None:
        self.incoherence_threshold = incoherence_threshold

    def compute(self, items: list[dict[str, Any]]) -> MetricResult:
        if not items:
            return MetricResult(name=self.name, value=0.0, details={"error": "no items"})

        multi_hop = [i for i in items if len(i.get("reference_chunks", [])) >= 2]

        if not multi_hop:
            return MetricResult(
                name=self.name,
                value=1.0,
                details={"note": "no multi-hop items", "total_items": len(items)},
            )

        coherence_scores: list[float] = []
        incoherent_items: list[dict[str, Any]] = []

        for item in multi_hop:
            chunks = item["reference_chunks"]
            token_sets = [set(_tokenize(str(c.get("content", "")))) for c in chunks]

            # Average pairwise Jaccard over chunk token sets
            n = len(token_sets)
            jacc_sum = 0.0
            n_pairs = 0
            for ci in range(n):
                for cj in range(ci + 1, n):
                    jacc_sum += _jaccard(token_sets[ci], token_sets[cj])
                    n_pairs += 1

            coherence = jacc_sum / n_pairs if n_pairs else 0.0
            coherence_scores.append(coherence)

            if coherence < self.incoherence_threshold:
                incoherent_items.append(
                    {
                        "question": str(item.get("question", ""))[:100],
                        "coherence": round(coherence, 4),
                        "chunk_count": len(chunks),
                    }
                )

        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        incoherence_rate = len(incoherent_items) / len(multi_hop)

        return MetricResult(
            name=self.name,
            value=round(avg_coherence, 4),
            details={
                "avg_coherence": round(avg_coherence, 4),
                "incoherence_rate": round(incoherence_rate, 4),
                "incoherent_count": len(incoherent_items),
                "multi_hop_items": len(multi_hop),
                "incoherence_threshold": self.incoherence_threshold,
                "examples": incoherent_items[:5],
                "total_items": len(items),
            },
        )


class ChunkConcentrationMetric:
    """Measures how evenly reference chunks are distributed across the
    dataset.  High concentration (Gini near 1.0) indicates the linker
    over-relies on a few generic chunks."""

    name = "chunk_concentration"

    def compute(self, items: list[dict[str, Any]]) -> MetricResult:
        if not items:
            return MetricResult(name=self.name, value=0.0, details={"error": "no items"})

        chunk_usage: Counter[str] = Counter()
        for item in items:
            for chunk in item.get("reference_chunks", []):
                chunk_id = str(chunk.get("id", ""))
                if chunk_id:
                    chunk_usage[chunk_id] += 1

        if not chunk_usage:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={"error": "no chunk IDs found", "total_items": len(items)},
            )

        counts = sorted(chunk_usage.values())
        gini = _gini_coefficient(counts)
        top_k = chunk_usage.most_common(10)

        return MetricResult(
            name=self.name,
            value=round(gini, 4),
            details={
                "gini_coefficient": round(gini, 4),
                "unique_chunks": len(chunk_usage),
                "total_references": sum(counts),
                "top_10_chunks": [{"id": cid[:16], "count": cnt} for cid, cnt in top_k],
                "max_usage": counts[-1] if counts else 0,
                "avg_usage": round(sum(counts) / len(counts), 2) if counts else 0,
                "total_items": len(items),
            },
        )


class NearDuplicateMetric:
    """Detects near-duplicate question pairs via TF-IDF cosine similarity.
    Reports the fraction of items that appear in at least one duplicate pair."""

    name = "near_duplicate"

    def __init__(self, *, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold

    def compute(self, items: list[dict[str, Any]]) -> MetricResult:
        if not items:
            return MetricResult(name=self.name, value=0.0, details={"error": "no items"})

        questions = [str(item.get("question", "")) for item in items]
        tokenized = [_tokenize(q) for q in questions]

        duplicate_pairs = _find_similar_pairs(tokenized, questions, self.similarity_threshold)

        involved: set[int] = set()
        for pair in duplicate_pairs:
            involved.add(pair["i"])
            involved.add(pair["j"])

        dupe_rate = len(involved) / len(items) if items else 0.0

        return MetricResult(
            name=self.name,
            value=round(dupe_rate, 4),
            details={
                "duplicate_pair_count": len(duplicate_pairs),
                "items_involved": len(involved),
                "similarity_threshold": self.similarity_threshold,
                "examples": duplicate_pairs[:5],
                "total_items": len(items),
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _entropy(counts: Counter, total: int) -> float:
    """Shannon entropy in bits."""
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy
