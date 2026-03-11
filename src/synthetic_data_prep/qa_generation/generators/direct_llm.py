"""Direct LLM generator for CgftPipeline."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from synthetic_data_prep.qa_generation.anchor_selector import AnchorBundle
from synthetic_data_prep.qa_generation.cgft_models import CgftContext, GenerationTask, LLMDirectGenerationConfig
from synthetic_data_prep.qa_generation.generated_qa import GeneratedQA
from synthetic_data_prep.qa_generation.helpers import render_template
from synthetic_data_prep.qa_generation.models import QADataPoint, ReferenceChunk

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_TEMPLATE_FIELD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ESCAPED_OPEN_BRACE = "__CGFT_ESCAPED_OPEN_BRACE__"
_ESCAPED_CLOSE_BRACE = "__CGFT_ESCAPED_CLOSE_BRACE__"

_DEFAULT_TEMPLATE = (
    "Your task is to generate a {qa_type} question that will require "
    "{target_hop_count} search steps to answer by gathering information from multiple sources.\n\n"
    "You must first reason inside <think> and </think> about how to connect the information "
    "across the provided documents to create a challenging, multi-hop question.\n\n"
    "[[if regeneration_attempt]]attempt={regeneration_attempt}\n[[endif]]"
    "[[if source_task_id]]source_task_id={source_task_id}\n[[endif]]"
    "[[if previous_failure_type]]previous_failure_type={previous_failure_type}\n[[endif]]"
    "[[if previous_judge_reason_tag]]previous_judge_reason_tag={previous_judge_reason_tag}\n[[endif]]"
    "[[if overlap_triggered]]overlap_triggered={overlap_triggered}\n[[endif]]"
    "[[if expected_action]]expected_action={expected_action}\n[[endif]]"
    "Corpus summary:\n{corpus_summary}\n\n"
    "Example queries:\n{corpus_queries}\n\n"
    "Primary chunk:\n{primary_chunk}\n\n"
    "Secondary chunks:\n{secondary_chunks}\n\n"
    "[[if failed_question]]Failed question:\n{failed_question}\n\n[[endif]]"
    "[[if failed_answer]]Failed answer:\n{failed_answer}\n\n[[endif]]"
    "[[if regeneration_prompt]]Feedback:\n{regeneration_prompt}\n\n[[endif]]"
    "Requirements:\n"
    "- Generate a **complicated**, realistic, user-facing {qa_type} question connecting information across chunks.\n"
    "- The question should require around {target_hop_count} retrieval/search steps to answer and avoid single-lookup shortcuts.\n"
    "- Use only chunk evidence; do not use outside knowledge.\n"
    "- Keep the question standalone and understandable without seeing source chunks.\n"
    "- Ensure the answer is correct, specific, and uniquely determined by the question and provided evidence.\n"
    "- Prefer task-oriented documentation use-cases over internal implementation trivia.\n"
    "- Avoid forced cross-topic stitching that is unlikely for a single user intent.\n"
    "- Reduce lexical shortcut risk by paraphrasing direct chunk phrasing and adding compositional constraints.\n"
    "[[if previous_failure_type]]- If previous_failure_type=too_easy, keep answer facts stable and make the question harder.\n[[endif]]"
    "[[if previous_failure_type]]- If previous_failure_type=unsupported, revise answer using current chunk evidence only.\n[[endif]]"
    "- Output exactly one question and one answer.\n\n"
    'First output your reasoning in <think>...</think>, then provide:\n'
    '```json\n{{\"question\": \"...\", \"answer\": \"...\", \"answering_steps\": \"...\"}}\n```'
)


def _render_template_safe(template: str, variables: dict[str, Any]) -> str:
    try:
        return render_template(template, variables)
    except KeyError:
        protected = (
            template.replace("{{", _ESCAPED_OPEN_BRACE).replace("}}", _ESCAPED_CLOSE_BRACE)
        )
        required_fields = set(_TEMPLATE_FIELD_RE.findall(protected))
        missing_fields = sorted(field for field in required_fields if field not in variables)
        if not missing_fields:
            raise

        logger.warning(
            "Template missing variables (%s); injecting empty values.",
            ", ".join(missing_fields),
        )
        merged = dict(variables)
        for field in missing_fields:
            merged[field] = ""
        return render_template(template, merged)


def _create_chat_completion_with_fallback(client: OpenAI, **kwargs: Any) -> Any:
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        retry_kwargs = dict(kwargs)
        if "unsupported parameter" in msg and "max_tokens" in msg:
            retry_kwargs.pop("max_tokens", None)
            return client.chat.completions.create(**retry_kwargs)
        if "unsupported parameter" in msg and "max_completion_tokens" in msg:
            retry_kwargs.pop("max_completion_tokens", None)
            return client.chat.completions.create(**retry_kwargs)
        raise


def _parse_qa_response(raw_text: str) -> tuple[str, str]:
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    # Strip <think> blocks before parsing
    text = _THINK_BLOCK_RE.sub("", text).strip()

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
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        if hasattr(chunk, "chunk_str"):
            parts.append(f"[{idx}] {chunk.chunk_str()}")
        else:
            parts.append(f"[{idx}] {str(chunk)}")
    return "\n\n".join(parts)


class DirectLLMGenerator:
    """Direct-call generator using task+anchor intent."""

    def __init__(
        self,
        *,
        client: OpenAI,
        linker: Any,
        cfg: LLMDirectGenerationConfig,
    ) -> None:
        self.client = client
        self.linker = linker
        self.cfg = cfg

    def generate(self, tasks: list[GenerationTask], context: CgftContext) -> list[GeneratedQA]:
        seed_lookup: dict[str, Any] = context.get("seed_chunk_lookup", {})
        corpus_pool: list[Any] = context.get("corpus_pool", [])
        corpus_summary = str(context.get("corpus_summary", "") or "").strip()
        corpus_queries = str(context.get("corpus_queries", "") or "").strip()
        corpus_description = str(context.get("corpus_description", "") or "").strip()
        generated: list[GeneratedQA] = []

        for task in tasks:
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

            requested_qa_type = str(task.qa_type).strip() or "lookup"
            requested_hop_count = max(1, int(task.target_hop_count or 1))
            resolved_qa_type = str(getattr(anchor, "target_qa_type", "")).strip() or requested_qa_type
            try:
                resolved_hop_count = int(getattr(anchor, "target_hop_count", requested_hop_count))
            except (TypeError, ValueError):
                resolved_hop_count = requested_hop_count
            resolved_hop_count = max(1, resolved_hop_count)

            template = self.cfg.prompt_templates_by_qa_type.get(
                resolved_qa_type,
                self.cfg.prompt_templates_by_qa_type.get(requested_qa_type, _DEFAULT_TEMPLATE),
            )
            variables = {
                "qa_type": resolved_qa_type,
                "target_hop_count": resolved_hop_count,
                "corpus_summary": corpus_summary,
                "corpus_queries": corpus_queries,
                "corpus_description": corpus_description,
                "regeneration_attempt": task.regeneration_attempt,
                "regeneration_prompt": task.regeneration_prompt,
                "source_task_id": task.source_task_id,
                "previous_failure_type": task.previous_failure_type,
                "previous_judge_reason_tag": task.previous_judge_reason_tag,
                "overlap_triggered": "true" if task.overlap_triggered else "",
                "expected_action": task.expected_action,
                "failed_question": task.failed_question,
                "failed_answer": task.failed_answer,
                "primary_chunk": (
                    anchor.primary_chunk.chunk_str()
                    if hasattr(anchor.primary_chunk, "chunk_str")
                    else str(anchor.primary_chunk)
                ),
                "secondary_chunks": _format_secondary_chunks(anchor),
            }
            prompt = _render_template_safe(template, variables)
            system_prompt = _render_template_safe(self.cfg.system_prompt, variables)

            user_content = prompt
            regeneration_prompt = str(task.regeneration_prompt or "").strip()
            if regeneration_prompt:
                user_content = (
                    f"{user_content}\n\n"
                    "Use this regeneration feedback from the failed prior attempt:\n"
                    f"{regeneration_prompt}"
                )
            completion = _create_chat_completion_with_fallback(
                self.client,
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_completion_tokens=self.cfg.max_completion_tokens,
                timeout=self.cfg.timeout,
                temperature=1.0,
            )
            raw_text = completion.choices[0].message.content or ""
            question, answer = _parse_qa_response(raw_text)
            if not question or not answer:
                logger.warning("Skipping task %s: failed to parse QA response.", task.task_id)
                continue

            reference_chunks = [_chunk_to_reference_chunk(anchor.primary_chunk)]
            reference_chunks.extend(_chunk_to_reference_chunk(chunk) for chunk in anchor.secondary_chunks)
            bm25_chunks = anchor.structural_hints.get("bm25_related", [])
            reference_chunks.extend(_chunk_to_reference_chunk(chunk) for chunk in bm25_chunks)

            qa_point: QADataPoint = {
                "question": question,
                "answer": answer,
                "reference_chunks": reference_chunks,
                "qa_type": resolved_qa_type,
                "min_hop_count": resolved_hop_count,
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
                        "qa_type_target": resolved_qa_type,
                        "target_hop_count": resolved_hop_count,
                        "qa_type_requested": requested_qa_type,
                        "target_hop_count_requested": requested_hop_count,
                        "anchor_bundle": anchor,
                        "generation_mode": "llm_direct",
                        "refinement_count": 0,
                        "same_seed_refinement_count": 0,
                        "task_id": task.task_id,
                        "source_task_id": task.source_task_id or task.task_id,
                        "regeneration_attempt": task.regeneration_attempt,
                        "seed_chunk_id": task.seed_chunk_id,
                    },
                )
            )
        return generated
