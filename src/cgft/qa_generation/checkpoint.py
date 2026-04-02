"""Checkpoint management for micro-batch pipeline processing."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cgft.qa_generation.generated_qa import FilterVerdict, GeneratedQA

logger = logging.getLogger(__name__)


@dataclass
class BatchCheckpoint:
    """Results from a single completed micro-batch."""

    batch_idx: int
    passed: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    regens_count: int
    raw_count: int


def _json_safe(value: Any) -> Any:
    """Recursively strip non-JSON-serializable values from a structure."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items() if _is_primitive_key(k)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # Drop non-serializable objects (e.g. AnchorBundle, Chunk).
    return None


def _is_primitive_key(key: Any) -> bool:
    return isinstance(key, (str, int))


def serialize_generated_qa(item: GeneratedQA) -> dict[str, Any]:
    """Serialize a GeneratedQA to a JSON-safe dict."""
    row = dict(item.qa)
    row["__generation_metadata"] = _json_safe(dict(item.generation_metadata))
    row["__regeneration_history"] = _json_safe(list(item.regeneration_history))
    row["__journey_events"] = _json_safe(list(item.journey_events))
    if item.filter_verdict is not None:
        row["__filter_verdict"] = {
            "status": item.filter_verdict.status,
            "reason": item.filter_verdict.reason,
            "reasoning": item.filter_verdict.reasoning,
            "metadata": _json_safe(dict(item.filter_verdict.metadata)),
        }
    return row


def deserialize_generated_qa(row: dict[str, Any]) -> GeneratedQA:
    """Reconstruct a GeneratedQA from a serialized dict."""
    row = dict(row)
    meta = row.pop("__generation_metadata", {})
    history = row.pop("__regeneration_history", [])
    journey_events = row.pop("__journey_events", [])
    verdict_raw = row.pop("__filter_verdict", None)
    verdict = None
    if verdict_raw is not None:
        verdict = FilterVerdict(
            status=verdict_raw["status"],
            reason=verdict_raw["reason"],
            reasoning=verdict_raw.get("reasoning", ""),
            metadata=verdict_raw.get("metadata", {}),
        )
    return GeneratedQA(
        qa=row,
        generation_metadata=meta,
        filter_verdict=verdict,
        regeneration_history=history,
        journey_events=journey_events,
    )


def compute_config_hash(
    *,
    random_seed: int,
    total_samples: int,
    corpus_id: str,
    primary_type_distribution: dict[str, float] | None = None,
    acceptance_policy: str = "default",
) -> str:
    """Deterministic hash of pipeline config for checkpoint invalidation."""
    payload = {
        "random_seed": random_seed,
        "total_samples": total_samples,
        "corpus_id": corpus_id,
        "primary_type_distribution": (
            {str(k): float(v) for k, v in sorted((primary_type_distribution or {}).items())}
        ),
        "acceptance_policy": str(acceptance_policy),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Manifest:
    """Checkpoint manifest tracking pipeline progress."""

    config_hash: str = ""
    completed_batch_count: int = 0
    total_passed: int = 0
    total_rejected: int = 0
    requeue_round: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "completed_batch_count": self.completed_batch_count,
            "total_passed": self.total_passed,
            "total_rejected": self.total_rejected,
            "requeue_round": self.requeue_round,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        return cls(
            config_hash=str(d.get("config_hash", "")),
            completed_batch_count=int(d.get("completed_batch_count", 0)),
            total_passed=int(d.get("total_passed", 0)),
            total_rejected=int(d.get("total_rejected", 0)),
            requeue_round=int(d.get("requeue_round", 0)),
        )


@dataclass
class ResumeState:
    """State recovered from checkpoints for resuming a pipeline run."""

    passed_items: list[GeneratedQA] = field(default_factory=list)
    completed_batch_count: int = 0
    requeue_round: int = 0


class CheckpointManager:
    """Manages checkpoint I/O for micro-batch pipeline processing."""

    def __init__(self, checkpoint_dir: Path, config_hash: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.config_hash = config_hash
        self._manifest: Manifest | None = None

    @property
    def manifest_path(self) -> Path:
        return self.checkpoint_dir / "manifest.json"

    @property
    def passed_path(self) -> Path:
        return self.checkpoint_dir / "checkpoint_passed.jsonl"

    def _ensure_dir(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> Manifest | None:
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            return Manifest.from_dict(json.load(fh))

    def save_manifest(self, manifest: Manifest) -> None:
        self._ensure_dir()
        with self.manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest.to_dict(), fh, ensure_ascii=False)
        self._manifest = manifest

    def resume_state(self) -> ResumeState:
        """Load checkpoint state for resuming. Returns empty state if none."""
        manifest = self.load_manifest()
        if manifest is None:
            return ResumeState()

        if manifest.config_hash != self.config_hash:
            logger.warning(
                "Checkpoint config hash mismatch (got %s, expected %s) "
                "— clearing stale checkpoints",
                manifest.config_hash,
                self.config_hash,
            )
            self.cleanup()
            return ResumeState()

        passed_items: list[GeneratedQA] = []
        if self.passed_path.exists():
            with self.passed_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        passed_items.append(deserialize_generated_qa(json.loads(line)))

        return ResumeState(
            passed_items=passed_items,
            completed_batch_count=manifest.completed_batch_count,
            requeue_round=manifest.requeue_round,
        )

    def save_batch(
        self,
        batch_idx: int,
        passed: list[GeneratedQA],
        rejected: list[GeneratedQA],
        regens_count: int = 0,
    ) -> None:
        """Checkpoint a completed batch. Appends passed to cumulative JSONL."""
        self._ensure_dir()

        # Append passed items to cumulative file.
        with self.passed_path.open("a", encoding="utf-8") as fh:
            for item in passed:
                fh.write(json.dumps(serialize_generated_qa(item), ensure_ascii=False) + "\n")

        # Write per-batch rejected for diagnostics.
        rejected_path = self.checkpoint_dir / f"batch_{batch_idx:04d}_rejected.jsonl"
        with rejected_path.open("w", encoding="utf-8") as fh:
            for item in rejected:
                fh.write(json.dumps(serialize_generated_qa(item), ensure_ascii=False) + "\n")

        # Update manifest.
        manifest = self.load_manifest() or Manifest(config_hash=self.config_hash)
        manifest.completed_batch_count = batch_idx + 1
        manifest.total_passed += len(passed)
        manifest.total_rejected += len(rejected)
        self.save_manifest(manifest)

    def cleanup(self) -> None:
        """Remove checkpoint directory."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            logger.info("Cleaned up checkpoints at %s", self.checkpoint_dir)
