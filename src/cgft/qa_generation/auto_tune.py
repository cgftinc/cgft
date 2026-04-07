"""Corpus-aware parameter auto-tuning and warnings for CgftPipeline."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CorpusStats:
    """Statistics extracted from the corpus for parameter tuning."""

    chunk_count: int
    avg_chunk_length: int
    median_chunk_length: int
    multi_file_corpus: bool  # chunks span multiple source files
    has_rich_metadata: bool  # headers, cross-references present

    @classmethod
    def from_source(cls, source: Any) -> CorpusStats:
        """Extract stats from a ChunkSource."""
        collection = getattr(source, "collection", None)
        if collection is None:
            return cls(
                chunk_count=0,
                avg_chunk_length=0,
                median_chunk_length=0,
                multi_file_corpus=False,
                has_rich_metadata=False,
            )

        chunks = list(collection)
        chunk_count = len(chunks)
        if chunk_count == 0:
            return cls(
                chunk_count=0,
                avg_chunk_length=0,
                median_chunk_length=0,
                multi_file_corpus=False,
                has_rich_metadata=False,
            )

        lengths = [len(c.content) for c in chunks]
        avg_chunk_length = int(statistics.mean(lengths))
        median_chunk_length = int(statistics.median(lengths))

        # Check if chunks span multiple source files.
        files: set[str] = set()
        rich_metadata_count = 0
        rich_keys = {"h1", "h2", "h3", "section_header", "title", "cross_references"}
        for chunk in chunks:
            meta = chunk.metadata_dict if hasattr(chunk, "metadata_dict") else {}
            file_val = meta.get("file", "")
            if file_val:
                files.add(str(file_val))
            if any(meta.get(k) for k in rich_keys):
                rich_metadata_count += 1

        multi_file_corpus = len(files) > 1
        has_rich_metadata = rich_metadata_count > chunk_count * 0.3

        return cls(
            chunk_count=chunk_count,
            avg_chunk_length=avg_chunk_length,
            median_chunk_length=median_chunk_length,
            multi_file_corpus=multi_file_corpus,
            has_rich_metadata=has_rich_metadata,
        )


def emit_corpus_warnings(
    corpus_stats: CorpusStats,
    cfg: Any,
) -> list[str]:
    """Return warnings when config looks infeasible for the corpus."""
    warnings: list[str] = []
    chunk_count = corpus_stats.chunk_count
    if chunk_count == 0:
        warnings.append("Corpus has 0 chunks — auto-tune cannot assess feasibility.")
        return warnings

    total_samples = cfg.targets.total_samples
    multi_hop_ratio = cfg.targets.primary_type_distribution.get("multi_hop", 0.7)
    multi_hop_target = int(total_samples * multi_hop_ratio)

    if total_samples > chunk_count * 2:
        warnings.append(
            f"Corpus has {chunk_count} chunks but {total_samples} samples "
            f"requested. Expect high rejection rates — consider reducing "
            f"total_samples to ~{chunk_count}."
        )

    if multi_hop_target > chunk_count * 0.5:
        warnings.append(
            f"Requesting {multi_hop_target} multi_hop pairs from "
            f"{chunk_count} chunks. Multi-hop requires cross-chunk "
            f"linking — this may produce many rejections."
        )

    hop_dist = cfg.targets.hop_distribution
    high_hop_pct = sum(v for k, v in hop_dist.items() if int(k) >= 3)
    if high_hop_pct > 0.3 and chunk_count < 200:
        warnings.append(
            f"3+ hop questions are {high_hop_pct:.0%} of multi_hop "
            f"target, but corpus only has {chunk_count} chunks. "
            f"Consider reducing hop targets."
        )

    if not corpus_stats.multi_file_corpus:
        warnings.append(
            "Corpus appears to be from a single source file. "
            "Multi-hop cross-document questions may have limited "
            "diversity."
        )

    return warnings


def auto_tune(
    corpus_stats: CorpusStats,
    cfg: Any,
) -> dict[str, Any]:
    """Derive sensible defaults from corpus characteristics.

    Returns dict of adjusted values. Does NOT mutate cfg — caller applies.

    Rules:
    - Small corpus (<100 chunks): reduce total_samples to
      min(user_target, chunk_count)
    - total_samples capped at 2 * chunk_count
    - Low metadata: reduce multi_hop ratio
    - Short chunks: reduce hop targets
    """
    adjustments: dict[str, Any] = {}
    chunk_count = corpus_stats.chunk_count
    if chunk_count == 0:
        return adjustments

    total_samples = cfg.targets.total_samples

    # Cap total_samples at 2 * chunk_count.
    if total_samples > chunk_count * 2:
        adjustments["total_samples"] = min(total_samples, chunk_count * 2)

    # Small corpus: cap at chunk_count.
    if chunk_count < 100 and total_samples > chunk_count:
        adjustments["total_samples"] = min(
            adjustments.get("total_samples", total_samples),
            chunk_count,
        )

    # Low metadata richness: reduce multi_hop ratio.
    if not corpus_stats.has_rich_metadata:
        current_dist = dict(cfg.targets.primary_type_distribution)
        mh = current_dist.get("multi_hop", 0.7)
        if mh > 0.4:
            adjusted_mh = max(0.3, mh - 0.2)
            adjusted_lookup = 1.0 - adjusted_mh
            adjustments["primary_type_distribution"] = {
                "lookup": round(adjusted_lookup, 2),
                "multi_hop": round(adjusted_mh, 2),
            }

    # Short median chunks: reduce high-hop targets.
    if corpus_stats.median_chunk_length < 600:
        hop_dist = {int(k): float(v) for k, v in cfg.targets.hop_distribution.items()}
        high_hop_weight = sum(v for k, v in hop_dist.items() if k >= 3)
        if high_hop_weight > 0.2:
            adjusted_hop: dict[int, float] = {}
            redistributed = 0.0
            for k, v in hop_dist.items():
                if k >= 3:
                    new_v = v * 0.5
                    redistributed += v - new_v
                    adjusted_hop[k] = new_v
                else:
                    adjusted_hop[k] = v
            # Redistribute to 2-hop.
            adjusted_hop[2] = adjusted_hop.get(2, 0.0) + redistributed
            adjustments["hop_distribution"] = adjusted_hop

    return adjustments


def should_early_stop(
    batch_history: list[Any],
    early_stop_rejection_rate: float = 0.85,
    early_stop_window: int = 3,
) -> bool:
    """Check if recent batches indicate the pipeline is spinning."""
    if len(batch_history) < early_stop_window:
        return False
    recent = batch_history[-early_stop_window:]
    avg_acceptance = sum(b.acceptance_rate for b in recent) / len(recent)
    return avg_acceptance < (1.0 - early_stop_rejection_rate)


def compute_batch_heuristics(
    total_samples: int,
) -> tuple[int, int]:
    """Return (batch_size, max_parallel_batches) from total_samples.

    Lookup table:
    total_samples  | batch_size | max_parallel
    <= 25          | 5          | 1
    26-50          | 10         | 1
    51-100         | 20         | 1
    101-200        | 40         | 1
    201-500        | 50         | 1
    501-1000       | 100        | 2
    > 1000         | 100        | min(4, total // 250)
    """
    if total_samples <= 25:
        return 5, 1
    if total_samples <= 50:
        return 10, 1
    if total_samples <= 100:
        return 20, 1
    if total_samples <= 200:
        return 40, 1
    if total_samples <= 500:
        return 50, 1
    if total_samples <= 1000:
        return 100, 2
    return 100, min(4, total_samples // 250)
