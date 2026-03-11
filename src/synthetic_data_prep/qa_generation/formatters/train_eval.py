"""Train/eval formatter for CgftPipeline outputs."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from synthetic_data_prep.qa_generation.cgft_models import CgftContext, OutputConfig, SplitConfig
from synthetic_data_prep.qa_generation.generated_qa import GeneratedQA
from synthetic_data_prep.qa_generation.style_controls import classify_query_style


class TrainEvalFormatter:
    """Writes stratified train/eval JSONL artifacts."""

    def __init__(self, *, output_cfg: OutputConfig, split_cfg: SplitConfig) -> None:
        self.output_cfg = output_cfg
        self.split_cfg = split_cfg

    def format(self, items: list[GeneratedQA], context: CgftContext) -> dict[str, Any]:
        rows = [self._to_row(item) for item in items]
        train_rows, eval_rows = self._stratified_split(rows)

        output_dir = Path(self.output_cfg.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / self.output_cfg.train_jsonl
        eval_path = output_dir / self.output_cfg.eval_jsonl

        with train_path.open("w", encoding="utf-8") as fh:
            for row in train_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with eval_path.open("w", encoding="utf-8") as fh:
            for row in eval_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {
            "train_dataset": train_rows,
            "eval_dataset": eval_rows,
            "stats": {
                "total": len(rows),
                "train": len(train_rows),
                "eval": len(eval_rows),
                "train_ratio": self.split_cfg.train_ratio,
                "stratify_by": list(self.split_cfg.stratify_by),
            },
            "output_paths": {
                "train_jsonl": str(train_path),
                "eval_jsonl": str(eval_path),
            },
            "rejected_dataset": context.get("rejected_items", []),
        }

    def _to_row(self, item: GeneratedQA) -> dict[str, Any]:
        eval_scores = dict(item.qa.get("eval_scores", {}) or {})
        transform_meta = dict(item.generation_metadata.get("transformation", {}) or {})
        style_target = str(transform_meta.get("target_style", "")).strip() or str(
            item.qa.get("style_target", "")
        ).strip()
        style_observed = str(eval_scores.get("query_style_observed", "")).strip()
        if not style_observed:
            style_observed = classify_query_style(str(item.qa.get("question", "")))
        return {
            "task_id": item.generation_metadata.get("task_id", ""),
            "question": item.qa.get("question", ""),
            "answer": item.qa.get("answer", ""),
            "qa_type": item.qa.get("qa_type", item.generation_metadata.get("qa_type_target", "unknown")),
            "style_target": style_target,
            "style_observed": style_observed,
            "min_hop_count": item.qa.get("min_hop_count"),
            "reference_chunks": item.qa.get("reference_chunks", []),
            "generation_metadata": {
                "qa_type_target": item.generation_metadata.get("qa_type_target"),
                "target_hop_count": item.generation_metadata.get("target_hop_count"),
                "generation_mode": item.generation_metadata.get("generation_mode"),
                "refinement_count": item.generation_metadata.get("refinement_count", 0),
                "task_id": item.generation_metadata.get("task_id"),
                "transformation": transform_meta,
            },
        }

    def _stratified_split(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rng = random.Random(self.split_cfg.seed)
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            qa_type = str(row.get("qa_type", "unknown"))
            style = str(row.get("style_observed", "")).strip() or str(
                row.get("style_target", "unknown")
            ).strip()
            buckets[(qa_type, style)].append(row)

        train_rows: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        for bucket_rows in buckets.values():
            rng.shuffle(bucket_rows)
            split_idx = int(len(bucket_rows) * self.split_cfg.train_ratio)
            if split_idx == 0 and len(bucket_rows) > 1:
                split_idx = 1
            elif split_idx >= len(bucket_rows) and len(bucket_rows) > 1:
                split_idx = len(bucket_rows) - 1
            train_rows.extend(bucket_rows[:split_idx])
            eval_rows.extend(bucket_rows[split_idx:])

        rng.shuffle(train_rows)
        rng.shuffle(eval_rows)
        return train_rows, eval_rows
