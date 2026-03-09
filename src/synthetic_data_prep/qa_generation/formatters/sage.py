"""SageFormatter — JSONL output + human-readable report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from synthetic_data_prep.qa_generation.generated_qa import GeneratedQA
from synthetic_data_prep.qa_generation.sage_utils import (
    SagePipelineConfig,
    _init_rollout_metrics,
    execution_trace_to_dict,
    write_report,
)

logger = logging.getLogger(__name__)


class SageFormatter:
    """Converts passed SAGE items into JSONL output and a human-readable report.

    Applies a minimum search steps filter and collects pipeline statistics.
    """

    def __init__(self, cfg: SagePipelineConfig) -> None:
        self.cfg = cfg

    def format(
        self, items: list[GeneratedQA], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert passed items to result dicts, write JSONL + report, return stats."""
        cfg = self.cfg
        anchors = context.get("anchors", [])

        # Build result dicts from passed items
        all_results: list[dict[str, Any]] = []
        for item in items:
            gen_meta = item.generation_metadata
            chunk_idx = gen_meta.get("chunk_idx", -1)
            target_steps = gen_meta.get("target_steps", cfg.n_search_steps)
            anchor = anchors[chunk_idx] if 0 <= chunk_idx < len(anchors) else None
            chunk_text = gen_meta.get("chunk_text", "")

            verdict = item.filter_verdict
            trace = verdict.metadata.get("trace") if verdict else None
            overlap_ratio = (
                verdict.metadata.get("overlap_ratio") if verdict else None
            )
            anchor_ref_ids = (
                verdict.metadata.get("anchor_ref_ids", []) if verdict else []
            )

            actual_hop_count = trace.actual_hop_count if trace else 0
            search_steps = actual_hop_count

            # Determine status
            reason = verdict.reason if verdict else "unknown"
            if reason == "correct_and_hop_target_met_with_low_anchor_overlap":
                status_str = "pass_reanchored"
            elif reason == "correct_but_below_hop_target":
                status_str = "pass_relaxed"
            else:
                status_str = "pass"

            q_refines = gen_meta.get("question_refinements", 0)
            anchor_regens = gen_meta.get("anchor_regenerations", 0)
            round_idx = q_refines

            result: dict[str, Any] = {
                "question": item.qa["question"],
                "answer": item.qa["answer"],
                "search_steps": search_steps,
                "round": round_idx,
                "source_chunk": chunk_text[:500],
                "status": status_str,
                "status_reason": reason,
                "target_qa_type": (
                    anchor.target_qa_type if anchor else "unknown"
                ),
                "target_hop_count": target_steps,
                "intended_hop_count": target_steps,
                "actual_hop_count": actual_hop_count,
                "question_refinements": q_refines,
                "anchor_regenerations": anchor_regens,
                "golden_chunks": trace.golden_chunks if trace else [],
                "evidence_refs": trace.golden_chunks if trace else [],
            }
            if trace:
                result["execution_trace"] = execution_trace_to_dict(trace)
            if overlap_ratio is not None:
                result["anchor_overlap"] = overlap_ratio
            if anchor and anchor_ref_ids:
                result["anchor_reference_ids"] = anchor_ref_ids
                result["reference_chunks"] = [
                    str(anchor.primary_chunk)[:300],
                    *[str(c)[:300] for c in anchor.secondary_chunks],
                    *[
                        str(c)[:300]
                        for c in anchor.structural_hints.get("bm25_related", [])
                    ],
                ]

            all_results.append(result)

        # Apply min_search_steps filter
        filtered: list[dict[str, Any]] = []
        dropped_by_min_step = 0
        for r in all_results:
            if r["search_steps"] >= cfg.min_search_steps:
                filtered.append(r)
            else:
                dropped_by_min_step += 1

        # Collect stats
        max_rounds = cfg.refinement.max_question_refinements
        stats: dict[str, Any] = {
            "total": context.get("total_generated", len(items)),
            "passed": len(all_results),
            "strict_passed": sum(
                1 for r in all_results if r["status"] == "pass"
            ),
            "relaxed_passed": sum(
                1 for r in all_results if r["status"] == "pass_relaxed"
            ),
            "dropped_by_min_step": dropped_by_min_step,
            "rounds": [0] * (max_rounds + 1),
            "failure_reasons": context.get("failure_reasons", {}),
            "rollout_metrics": context.get(
                "rollout_metrics", _init_rollout_metrics()
            ),
            "rollout_metrics_by_type": context.get(
                "rollout_metrics_by_type", {}
            ),
            "anchor_quality": context.get("anchor_quality", {}),
        }
        for r in all_results:
            round_idx = min(r.get("round", 0), max_rounds)
            stats["rounds"][round_idx] += 1

        # Write JSONL output
        output_paths: dict[str, str] = {}
        output_path = Path(cfg.output)
        with output_path.open("w") as f:
            for r in filtered:
                f.write(json.dumps(r) + "\n")
        output_paths["jsonl"] = str(output_path)
        logger.info("Wrote %d results to %s", len(filtered), output_path)

        # Write human-readable report
        report_path = output_path.with_suffix(".txt")
        write_report(report_path, filtered, stats, cfg)
        output_paths["report"] = str(report_path)
        logger.info("Wrote report to %s", report_path)

        return {
            "results": filtered,
            "all_results": all_results,
            "stats": stats,
            "output_paths": output_paths,
        }
