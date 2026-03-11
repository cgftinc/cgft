"""Environment-backed rollout generator for CgftPipeline."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from synthetic_data_prep.qa_generation.anchor_selector import AnchorBundle
from synthetic_data_prep.qa_generation.cgft_models import CgftContext, GenerationTask, LLMEnvGenerationConfig
from synthetic_data_prep.qa_generation.generated_qa import GeneratedQA
from synthetic_data_prep.qa_generation.helpers import render_template
from synthetic_data_prep.qa_generation.models import QADataPoint, ReferenceChunk
from synthetic_data_prep.trainer.client import RolloutClient

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_TEMPLATE_FIELD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_ESCAPED_OPEN_BRACE = "__CGFT_ESCAPED_OPEN_BRACE__"
_ESCAPED_CLOSE_BRACE = "__CGFT_ESCAPED_CLOSE_BRACE__"


def _render_template_safe(template: str, variables: dict[str, Any]) -> str:
    protected = (
        template.replace("{{", _ESCAPED_OPEN_BRACE).replace("}}", _ESCAPED_CLOSE_BRACE)
    )
    required_fields = set(_TEMPLATE_FIELD_RE.findall(protected))
    missing_fields = sorted(field for field in required_fields if field not in variables)
    if not missing_fields:
        return render_template(template, variables)

    logger.warning(
        "Template missing variables (%s); injecting empty values.",
        ", ".join(missing_fields),
    )
    merged = dict(variables)
    for field in missing_fields:
        merged[field] = ""
    return render_template(template, merged)


def _parse_qa_response(raw_text: str) -> tuple[str, str]:
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    for candidate in (text, *_JSON_BLOCK_RE.findall(text)):
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if question and answer:
            return question, answer

    question_match = _QUESTION_RE.search(text)
    answer_match = _ANSWER_RE.search(text)
    if question_match and answer_match:
        return question_match.group(1).strip(), answer_match.group(1).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines[0], "\n".join(lines[1:])
    return "", ""


def _chunk_to_reference_chunk(chunk: Any) -> ReferenceChunk:
    metadata: dict[str, Any] = {}
    if hasattr(chunk, "metadata_dict"):
        metadata = dict(chunk.metadata_dict)
    elif hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        metadata = dict(chunk.metadata)
    elif isinstance(chunk, dict):
        metadata = dict(chunk.get("metadata", {}) or {})

    chunk_id = (
        getattr(chunk, "hash", None)
        or metadata.get("id")
        or metadata.get("file")
        or metadata.get("document_id")
        or str(chunk)[:80]
    )
    content = chunk.content if hasattr(chunk, "content") else str(chunk)
    return {"id": str(chunk_id), "metadata": metadata, "content": str(content)}


def _format_secondary_chunks(anchor: AnchorBundle) -> str:
    chunks = list(anchor.secondary_chunks)
    chunks.extend(anchor.structural_hints.get("bm25_related", []))
    if not chunks:
        return "(none)"
    return "\n\n".join(
        (
            chunk.chunk_str(max_chars=600)
            if hasattr(chunk, "chunk_str")
            else str(chunk)[:600]
        )
        for chunk in chunks
    )


class EnvRolloutGenerator:
    """RolloutClient-backed generator with generic environment bundles."""

    def __init__(
        self,
        *,
        rollout_client: RolloutClient,
        linker: Any,
        cfg: LLMEnvGenerationConfig,
    ) -> None:
        self.rollout_client = rollout_client
        self.linker = linker
        self.cfg = cfg

    def _run_rollout(self, raw_example: dict[str, Any], *, example_index: int) -> dict[str, Any]:
        env_bundle = self.cfg.env_bundle
        limits = self.cfg.rollout_limits
        cls_bytes, meta_bytes = env_bundle.as_bytes_bundle()
        if env_bundle.has_paths():
            return self.rollout_client.stream_rollout(
                raw_example=raw_example,
                env_cls_path=env_bundle.env_cls_path,
                env_metadata_path=env_bundle.env_metadata_path,
                llm_model=self.cfg.model,
                llm_base_url=self.cfg.base_url,
                llm_api_key=self.cfg.api_key,
                max_turns=limits.max_turns,
                max_tool_calls=limits.max_tool_calls,
                max_completion_tokens=limits.max_completion_tokens,
                capture_messages=True,
                include_event_meta=False,
                example_index=example_index,
            )

        if cls_bytes is None or meta_bytes is None:
            raise ValueError(
                "LLM env generation requires env bundle paths or env bundle files."
            )
        return self.rollout_client.stream_rollout(
            raw_example=raw_example,
            env_cls_bytes=cls_bytes,
            env_metadata_bytes=meta_bytes,
            llm_model=self.cfg.model,
            llm_base_url=self.cfg.base_url,
            llm_api_key=self.cfg.api_key,
            max_turns=limits.max_turns,
            max_tool_calls=limits.max_tool_calls,
            max_completion_tokens=limits.max_completion_tokens,
            capture_messages=True,
            include_event_meta=False,
            example_index=example_index,
        )

    def generate(self, tasks: list[GenerationTask], context: CgftContext) -> list[GeneratedQA]:
        seed_lookup: dict[str, Any] = context.get("seed_chunk_lookup", {})
        corpus_pool: list[Any] = context.get("corpus_pool", [])
        corpus_summary = str(context.get("corpus_summary", "") or "").strip()
        corpus_queries = str(context.get("corpus_queries", "") or "").strip()
        corpus_description = str(context.get("corpus_description", "") or "").strip()
        generated: list[GeneratedQA] = []

        for idx, task in enumerate(tasks):
            seed_chunk = seed_lookup.get(task.seed_chunk_id)
            if seed_chunk is None and corpus_pool:
                seed_chunk = context.rng.choice(corpus_pool)
            if seed_chunk is None:
                logger.warning("Skipping task %s: no seed chunk available.", task.task_id)
                continue

            anchor = self.linker.link(
                seed_chunk,
                target_qa_type=task.qa_type,
                target_hop_count=task.target_hop_count,
                corpus_pool=corpus_pool,
            )

            variables = {
                "qa_type": task.qa_type,
                "target_hop_count": task.target_hop_count,
                "corpus_summary": corpus_summary,
                "corpus_queries": corpus_queries,
                "corpus_description": corpus_description,
                "regeneration_attempt": task.regeneration_attempt,
                "regeneration_prompt": task.regeneration_prompt,
                "source_task_id": task.source_task_id or task.task_id,
                "primary_chunk": (
                    anchor.primary_chunk.chunk_str(max_chars=1200)
                    if hasattr(anchor.primary_chunk, "chunk_str")
                    else str(anchor.primary_chunk)[:1200]
                ),
                "secondary_chunks": _format_secondary_chunks(anchor),
            }
            prompt = _render_template_safe(self.cfg.prompt_template, variables)
            regeneration_prompt = str(task.regeneration_prompt or "").strip()
            if regeneration_prompt:
                prompt = (
                    f"{prompt}\n\n"
                    "Use this regeneration feedback from the failed prior attempt:\n"
                    f"{regeneration_prompt}"
                )

            result = self._run_rollout(
                {
                    "prompt": prompt,
                    "task_id": task.task_id,
                },
                example_index=idx,
            )
            final_text = (result.get("final_assistant_text") or result.get("trace") or "").strip()
            question, answer = _parse_qa_response(final_text)
            if not question or not answer:
                logger.warning("Skipping task %s: rollout response missing QA.", task.task_id)
                continue

            reference_chunks = [_chunk_to_reference_chunk(anchor.primary_chunk)]
            reference_chunks.extend(_chunk_to_reference_chunk(chunk) for chunk in anchor.secondary_chunks)

            qa_point: QADataPoint = {
                "question": question,
                "answer": answer,
                "reference_chunks": reference_chunks,
                "qa_type": task.qa_type,
                "min_hop_count": task.target_hop_count,
                "is_co_located": None,
                "filter_status": None,
                "filter_reasoning": None,
                "no_context_answer": None,
                "eval_scores": {},
            }
            generated.append(
                GeneratedQA(
                    qa=qa_point,
                    generation_metadata={
                        "qa_type_target": task.qa_type,
                        "target_hop_count": task.target_hop_count,
                        "anchor_bundle": anchor,
                        "generation_mode": "llm_env",
                        "refinement_count": 0,
                        "same_seed_refinement_count": 0,
                        "task_id": task.task_id,
                        "source_task_id": task.source_task_id or task.task_id,
                        "regeneration_attempt": task.regeneration_attempt,
                        "seed_chunk_id": task.seed_chunk_id,
                        "rollout_result": result,
                    },
                )
            )
        return generated
