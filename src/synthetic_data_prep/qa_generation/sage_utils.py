"""SAGE shared utilities — prompts, config, parsing, execution trace, rollout helpers.

Consolidates reusable functions from ``scripts/sage_pipeline.py`` that are
consumed by the SAGE protocol implementations (SageGenerator, SageFilter,
SageFeedbackRegenerator, SageFormatter).
"""

from __future__ import annotations

import json
import logging
import random
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

import synthetic_data_prep
from synthetic_data_prep.envs.cgft_search_env import CgftSearchEnv
from synthetic_data_prep.trainer.client import RolloutClient

from .anchor_utils import (
    extract_anchor_ref_ids as _shared_extract_anchor_ref_ids,
    generate_bm25_queries as _shared_generate_bm25_queries,
    select_anchor_bundle_with_enrichment as _shared_select_anchor_bundle_with_enrichment,
)
from .anchor_selector import AnchorBundle
from .corpus_capabilities import CorpusCapabilities  # noqa: F401 — re-exported
from .helpers import render_template
from .hybrid_prompts import CORPUS_SYSTEM_PROMPT, CORPUS_USER_TEMPLATE
from .response_parsers import parse_corpus_summary_response
from .style_controls import StyleControlConfig, normalize_style_distribution

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

question_generation_prompt = """\
Your task is to generate a complicated question that will require a search agent \
target_search_step search steps to answer by \
gathering information using a search engine.
You will first reason about the initial document and plan for gathering comprehensive \
information inside <think> and </think>.
You will then make a tool call and it will return the top searched results to collect information.
You must conduct reasoning inside <think> and </think> first every time you get new information.
You will call the search engine for n_search_step steps. After n_search_step searches, you must \
provide the question inside <question> and </question>, the answer inside <answer> and </answer>, \
and the answering step inside <answering steps> and </answering steps>. You can use your own \
knowledge to construct the search query, but the final answer and each of the answering step must \
be supported by the information you gathered from the search engine.
The question should be understandable standalone as the agent will use the question to search for \
information without access to the initial document.
If corpus context is provided in the user message, align the question style and topic scope with \
that context.
An example question: How much did the film in which Jake Gyllenhaal played his second lead role \
gross in its initial run at the box office?
Make sure the answer is correct and **unique** for the question generated.
Initial document: context\
"""

search_agent_prompt = """\
Answer the given question by using a search engine.
You will first reason about the question inside <think> and </think>, for instance, break down \
the question into multiple sub-questions that you will search for.
When you need retrieval, use the rollout tool-calling contract only.
Do not fabricate tool outputs (for example, do not write your own <information> blocks). \
Tool results are returned in user messages after each tool call.
After receiving the information, you must reason about it inside <think> and </think> before \
issuing a new query or providing the final answer.
Each of your reasoning step should be grounded in the retrieved information. Do not use your own \
knowledge, but you can use commonsense knowledge or arithmetic knowledge.
Do not use your own knowledge to write the query, the query should be based on the question and \
the retrieved documents.
Do not infer the entities in the question, but you can use the entities in the retrieved documents \
to write the query.
You can search as many times as your want. Try to break down the question for each search query \
and gather comprehensive information.
If you have gathered enough information to answer the question, you can provide the answer to the \
query inside <answer> and </answer>. The answer can be short or long-form depending on what the \
question requires.
When you are ready to finish, your final assistant message must be only \
<answer>...</answer> (no <think>, no tool calls, no extra text before/after).
Keep the final answer concise and directly responsive to the question.
Generate an answer based on the retrieved information, instead of your own knowledge.
CRITICAL RULES:
1. Every message MUST include either a search tool call OR a final <answer>...</answer>, \
never both in the same message and never only <think> tags.
2. You MUST search before answering. Do NOT provide <answer> until you have received and \
read search results. Never answer from your own knowledge.
3. Always close your tool calls: <tool_call>...</tool_call>.
This is an example answer: <answer>Beijing</answer>. Question: question\
"""

llmaaj_prompt = """\
Judge whether the following [response] to [question] is correct or not based on the precise and \
unambiguous [correct_answer_list] below. Each answer in the [correct_answer_list] is separated \
by a comma.
[question]: {{QUESTION}}
[response]: {{MODEL_ANSWER}}
Your judgment must be in the format and criteria specified below:
extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted \
answer as 'None' if there is no exact, final answer to extract from the response.
[correct_answer_list]: {{GOLD_ANSWER}}
reasoning: Explain why the extracted_final_answer is correct or incorrect based on \
[correct_answer_list], focusing only on if there are meaningful differences between answer in \
the [correct_answer_list] and the extracted_final_answer. Focus on recall, i.e. if the \
extracted_final_answer covers all the points in the answer in the [correct_answer_list]. It is \
ok if it provides more details. It is also ok if the extracted_final_answer misses minor point \
from the correct_answer, as long as it is evident that they are referring to the same thing. Do \
not comment on any background to the problem, do not attempt to solve the problem, do not argue \
for any answer different than [correct_answer_list], focus only on whether the answers match. \
Ignore capitalization.
correct: Answer 'yes' if extracted_final_answer matches any of the answers in \
[correct_answer_list] given above, or is within a small margin of error for numerical problems. \
Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if \
the extracted answer is incorrect.
confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if \
there is no confidence score available.\
"""

execution_feedback_incorrect_prompt = """\
You will be given an output from a question generator agent, which generates a complicated \
question, answer pair; as well as the output from a search agent, which attempts to solve the \
question generated in a fixed number of turns.
The answer from the search agent is not the same as the data generator agent. Your task is to \
examine their traces and output the correct question, answer pair based on their retrieved \
documents. You can update either the question, the answer or both.

# Current question and answer (the gold standard)
Question: {{CURRENT_QUESTION}}
Answer: {{CURRENT_ANSWER}}

IMPORTANT grounding rules:
- The current answer above is the AUTHORITATIVE gold standard. Only modify it if the retrieved \
documents from the search agent CLEARLY AND SPECIFICALLY contradict it with direct evidence.
- If the search agent performed 0 search steps (no tool calls), its answer is based on \
parametric knowledge and should NOT be trusted. In that case, keep the current answer and only \
refine the question to make it easier to search for.
- When in doubt between the data generator's answer and the search agent's answer, prefer the \
data generator's answer since it was produced with access to the source documents.
- Do not adopt vague paraphrases from the search agent. Preserve specific technical details \
(exact class names, method names, parameter values) from the current answer.

You will first reason about why is there a discrepancy between the search agent's answer and the \
data generator's answer. Output your reasoning trace inside <reason> and </reason>. You will then \
reason about how to update the question answer pair to make sure it is correct and requires the \
agent target_step search step to answer. A search step is defined as a call to the search tool. \
Output your reasoning trace inside <think> and </think>. For factual information, you should ONLY \
rely on the context provided for the data generator agent and the documents retrieved by both the \
data generator and search agent (inside <information> and </information>).
If you find it non-trivial to update just the question and answer, you can generate a new question \
answer pair ONLY based on the retrieved documents.
The updated question should require the search agent at least target_step search steps to answer. \
The answer can be short or long-form depending on the question. The question should \
be understandable standalone, as the search agent will solve the question without access to the \
documents (they will need to search for them).
When you are ready to provide the new question, answer pair, you can provide the question inside \
<question> and </question>, the answer inside <answer> and </answer>, and the search step inside \
<search steps> and </search steps>. For each search step, output the exact search question; the \
sub-answer to the search question; and the retrieved document from the search agent and data \
generator agent's output that supports the sub-answer. Make sure each step is absolutely needed to \
answer the question and there is no short cut. Tip: use retrieved document from different steps so \
avoid two sub-queries being solved by one search query.
# Data generator agent
Prompt: {{DATA_GENERATOR_PROMPT}}
Agent's output: {{DATA_GENERATOR_RESPONSE}}
# Search agent
Prompt: {{SEARCH_AGENT_PROMPT}}
Agent's output: {{SEARCH_AGENT_RESPONSE}}
# Your output\
"""

execution_feedback_too_easy_prompt = """\
You will be given an output from a question generator agent, which generates a complicated \
question, answer pair to be solved by a search agent for at least target_step **search** steps; \
as well as the output from a search agent, which attempts to solve the question generated. The \
search agent is able to solve the question in less than target_step search steps. Your task is \
to update the question so that it requires the search agent more steps to solve.

# Current question and answer (the gold standard)
Question: {{CURRENT_QUESTION}}
Answer: {{CURRENT_ANSWER}}

IMPORTANT: The current answer above is CORRECT. Do NOT change factual details in the answer. \
Focus only on making the question harder so it requires more search steps. Preserve all specific \
technical details (exact class names, method names, parameter values) in the answer.

You will first reason about why the search agent is able to solve the question in fewer steps. \
Output your reasoning trace inside <reason> and </reason>. You will then reason about how to \
update the question so that it will require more search steps. For factual information, you should \
ONLY rely on the context provided for the data generator agent and the documents retrieved by both \
the data generator and search agent (inside <information> and </information>), without relying on \
other information not in the retrieved context. Output your reasoning trace inside <think> and \
</think>. If you find it non-trivial to update the plan, you can generate a new question answer \
pair ONLY based on the retrieved documents.
The updated question should require the search agent at least target_step search steps to answer. \
Note that some of the answering steps do not involve search and thus do not count. The \
answer can be short or long-form depending on the question. The question should be \
understandable standalone, as the agent will solve the question without access to the documents \
(they will need to search for them).
When you are ready to provide the new question, answer pair, you can provide the question inside \
<question> and </question>, the answer inside <answer> and </answer>, and the search step inside \
<search steps> and </search steps>. For each search step, output the exact search question; the \
sub-answer to the search question; and the retrieved document from the search agent and data \
generator agent's output that supports the sub-answer. Make sure each step is absolutely needed to \
answer the question and there is no short cut. Tip: use retrieved document from different steps so \
avoid two sub-queries being solved by one search query.
# Data generator agent
Prompt: {{DATA_GENERATOR_PROMPT}}
Agent's output: {{DATA_GENERATOR_RESPONSE}}
# Search agent
Prompt: {{SEARCH_AGENT_PROMPT}}
Agent's output: {{SEARCH_AGENT_RESPONSE}}
# Your output\
"""

question_generation_with_refs_prompt = """\
Your task is to generate a complicated {target_qa_type} question that will require \
{target_hop_count} search steps to answer by gathering information from multiple sources.

You will reason about the documents provided and generate a multi-hop question that \
connects information across them. The question must require searching for and combining \
information from multiple sources to answer.

You must first reason inside <think> and </think> about how to connect the information \
across the provided documents to create a challenging question.

Primary document:
{primary_chunk}

{reference_section}

Corpus context (user-provided intent and example search behavior):
{corpus_context}

Requirements:
- The question should require {target_hop_count} search steps to answer
- The question type should be: {target_qa_type}
- The answer should be correct and specific; short or long-form answers are both allowed
- The question must be understandable standalone without access to these documents
- The answer must be supported by the provided documents
- Make sure the answer is correct and **unique** for the question generated
- The question should feel like a realistic search query for this corpus and user context
- Prefer task-oriented, user-facing information over internal implementation trivia
- Avoid forced cross-topic stitching that is unlikely for one user intent

Provide the question inside <question> and </question>, the answer inside \
<answer> and </answer>, and the answering steps inside <answering steps> and \
</answering steps>. Each answering step should describe what search query to use \
and what information it retrieves.\
"""

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SageModelConfig:
    """LLM endpoint configuration for a single agent role."""

    model: str
    api_key: str = ""
    base_url: str = "https://app.cgft.io/api/llm"


@dataclass
class CorpusAutoProfileConfig:
    """Auto-generated corpus profile controls for richer prompt grounding."""

    enabled: bool = True
    num_top_level_samples: int = 4
    num_random_samples: int = 4
    min_chunk_chars: int = 400
    include_chunk_excerpts_per_step: bool = False
    per_step_excerpt_count: int = 2
    per_step_excerpt_max_chars: int = 320


@dataclass
class CorpusContextConfig:
    """User-provided corpus intent signals for relevance-aware generation."""

    description: str = ""
    example_queries: list[str] = field(default_factory=list)
    auto_profile: CorpusAutoProfileConfig = field(default_factory=CorpusAutoProfileConfig)


@dataclass
class PromptConfig:
    """Overridable prompt templates.

    Any field left as empty string falls back to the module-level default.
    Values may be literal prompt text *or* ``file:path/to/prompt.txt``.
    """

    question_generation: str = ""
    question_generation_with_refs: str = ""
    search_agent: str = ""
    judge: str = ""
    feedback_incorrect: str = ""
    feedback_too_easy: str = ""

    def resolve(self) -> None:
        """Replace ``file:`` prefixes with file contents."""
        for fld in (
            "question_generation",
            "question_generation_with_refs",
            "search_agent",
            "judge",
            "feedback_incorrect",
            "feedback_too_easy",
        ):
            val = getattr(self, fld)
            if val:
                object.__setattr__(self, fld, _load_prompt(val))

    def get_question_generation(self) -> str:
        return self.question_generation or question_generation_prompt

    def get_question_generation_with_refs(self) -> str:
        return self.question_generation_with_refs or question_generation_with_refs_prompt

    def get_search_agent(self) -> str:
        return self.search_agent or search_agent_prompt

    def get_judge(self) -> str:
        return self.judge or llmaaj_prompt

    def get_feedback_incorrect(self) -> str:
        return self.feedback_incorrect or execution_feedback_incorrect_prompt

    def get_feedback_too_easy(self) -> str:
        return self.feedback_too_easy or execution_feedback_too_easy_prompt


@dataclass
class AnchorConfig:
    """Anchor-based chunk selection configuration."""

    enabled: bool = True
    bm25_enrichment_top_k: int = 5
    bm25_enrichment_queries: int = 3
    type_distribution: dict[str, float] | None = None
    target_hop_counts: dict[str, int] | None = None


@dataclass
class RefinementConfig:
    """Tier-2 style refinement controls."""

    max_question_refinements: int = 3
    max_anchor_regenerations: int = 2
    hop_tolerance: int = 0
    min_golden_chunk_overlap: float = 0.5


@dataclass
class SageQueryRewriteConfig:
    """Search-tool query rewriting controls for BM25 retrieval."""

    enabled: bool = False
    strategy: str = "intent"
    max_terms: int = 16
    max_chars: int = 140
    log_rewrites: bool = False
    model: SageModelConfig = field(default_factory=lambda: SageModelConfig(model="gpt-5-mini"))


@dataclass
class SagePipelineConfig:
    """Full SAGE pipeline configuration loaded from YAML."""

    # Platform
    api_key: str
    base_url: str = "https://app.cgft.io"

    # Corpus
    corpus_id: str = ""
    docs_path: str = ""
    corpus_name: str = "sage-corpus"
    corpus_context: CorpusContextConfig = field(default_factory=CorpusContextConfig)

    # Per-agent model configs
    question_generator: SageModelConfig = field(
        default_factory=lambda: SageModelConfig(model="gpt-5.2")
    )
    search_agent: SageModelConfig = field(default_factory=lambda: SageModelConfig(model="gpt-4.1"))
    judge: SageModelConfig = field(default_factory=lambda: SageModelConfig(model="gpt-4.1"))
    feedback: SageModelConfig = field(default_factory=lambda: SageModelConfig(model="gpt-4.1"))

    # Anchor selection
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    query_rewrite: SageQueryRewriteConfig = field(default_factory=SageQueryRewriteConfig)
    style_control: StyleControlConfig = field(default_factory=StyleControlConfig)

    # Prompt overrides
    prompts: PromptConfig = field(default_factory=PromptConfig)

    # Pipeline parameters
    num_samples: int = 5
    n_search_steps: int = 2
    n_rollouts: int = 4
    search_max_turns: int = 16
    search_max_tool_calls: int = 24
    search_no_answer_retries: int = 2
    strict_answer_contract: bool = True
    answer_contract_max_chars: int = 2000
    answer_repair_enabled: bool = True
    answer_repair_model: str = ""
    rollout_log_full_messages: bool = False
    rollout_log_event_meta: bool = True
    max_feedback_rounds: int = 3
    min_search_steps: int = 2
    min_chunk_chars: int = 400
    max_retries: int = 3
    retry_delay: float = 5.0

    # Output
    output: str = "sage_output.jsonl"

    def resolve_api_keys(self) -> None:
        """Fill in agent api_keys that were left empty with the platform api_key."""
        for agent_cfg in (
            self.question_generator,
            self.search_agent,
            self.judge,
            self.feedback,
            self.query_rewrite.model,
        ):
            if not agent_cfg.api_key:
                agent_cfg.api_key = self.api_key


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_prompt(value: str) -> str:
    """Load a prompt from a file path or return the string as-is."""
    if value.startswith("file:"):
        path = Path(value[5:].strip())
        return path.read_text()
    return value


def _parse_example_queries(raw: Any) -> list[str]:
    """Parse example queries from YAML (list or newline/comma-delimited string)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(q).strip() for q in raw if str(q).strip()]
    if isinstance(raw, str):
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) > 1:
            return [ln.lstrip("- ").strip() for ln in lines if ln.lstrip("- ").strip()]
        return [q.strip() for q in raw.split(",") if q.strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _parse_model_config(raw: dict[str, Any] | str) -> SageModelConfig:
    """Parse a model config from YAML — accepts a dict or a bare model string."""
    if isinstance(raw, str):
        return SageModelConfig(model=raw)
    return SageModelConfig(
        model=raw.get("model", "gpt-4.1"),
        api_key=raw.get("api_key", ""),
        base_url=raw.get("base_url", "https://app.cgft.io/api/llm"),
    )


def _parse_corpus_auto_profile(raw: Any) -> CorpusAutoProfileConfig:
    """Parse corpus auto-profile settings from YAML."""
    if not isinstance(raw, dict):
        return CorpusAutoProfileConfig()
    return CorpusAutoProfileConfig(
        enabled=bool(raw.get("enabled", True)),
        num_top_level_samples=max(0, int(raw.get("num_top_level_samples", 4))),
        num_random_samples=max(0, int(raw.get("num_random_samples", 4))),
        min_chunk_chars=max(0, int(raw.get("min_chunk_chars", 400))),
        include_chunk_excerpts_per_step=bool(raw.get("include_chunk_excerpts_per_step", False)),
        per_step_excerpt_count=max(0, int(raw.get("per_step_excerpt_count", 2))),
        per_step_excerpt_max_chars=max(40, int(raw.get("per_step_excerpt_max_chars", 320))),
    )


def _render_corpus_user_context(corpus_context: CorpusContextConfig) -> str:
    provided_queries = ", ".join(corpus_context.example_queries) if corpus_context.example_queries else ""
    return (
        f"Description: {corpus_context.description}\n"
        f"Example queries provided by user: {provided_queries}"
    ).strip()


def _chunk_text_for_prompt(chunk: Any) -> str:
    if hasattr(chunk, "chunk_str"):
        return str(chunk.chunk_str())
    if hasattr(chunk, "content"):
        return str(getattr(chunk, "content"))
    return str(chunk)


def _compact_text_block(text: str, *, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def generate_corpus_auto_profile(
    *,
    source: Any,
    client: Any,
    model_cfg: SageModelConfig,
    corpus_context: CorpusContextConfig,
) -> dict[str, Any]:
    """Generate hybrid-style corpus profile (summary + likely queries)."""
    auto_cfg = corpus_context.auto_profile
    if not auto_cfg.enabled:
        raise ValueError("corpus_auto_profile_disabled")

    top_level_chunks: list[Any] = []
    if auto_cfg.num_top_level_samples > 0 and hasattr(source, "get_top_level_chunks"):
        top_level_chunks = list(source.get_top_level_chunks() or [])

    sampled_top_level = (
        random.sample(top_level_chunks, min(auto_cfg.num_top_level_samples, len(top_level_chunks)))
        if top_level_chunks
        else []
    )
    sampled_random = (
        source.sample_chunks(auto_cfg.num_random_samples, min_chars=auto_cfg.min_chunk_chars)
        if auto_cfg.num_random_samples > 0
        else []
    )

    if not sampled_top_level and not sampled_random:
        raise ValueError("no_chunks_available_for_auto_profile")

    variables = {
        "user_context": _render_corpus_user_context(corpus_context),
        "top_level_content": "\n\n".join(_chunk_text_for_prompt(chunk) for chunk in sampled_top_level),
        "random_content": "\n\n".join(_chunk_text_for_prompt(chunk) for chunk in sampled_random),
    }
    user_prompt = render_template(CORPUS_USER_TEMPLATE, variables)
    response = _create_chat_completion_with_token_fallback(
        client,
        model=model_cfg.model,
        messages=[
            {"role": "system", "content": CORPUS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1.0 if "gpt" in model_cfg.model.lower() else 0.2,
        max_tokens=2000,
        max_completion_tokens=2000,
    )
    response_text = response.choices[0].message.content or ""
    summary, queries = parse_corpus_summary_response(response_text)
    cleaned_queries = [str(q).strip() for q in queries if str(q).strip()]
    if not summary.strip() or not cleaned_queries:
        raise ValueError(
            "auto_profile_parse_failed: expected non-empty summary and example_queries"
        )

    return {
        "summary": summary.strip(),
        "example_queries": cleaned_queries,
        "raw_response": response_text,
        "sampled_top_level_chunks": [_chunk_text_for_prompt(chunk) for chunk in sampled_top_level],
        "sampled_random_chunks": [_chunk_text_for_prompt(chunk) for chunk in sampled_random],
    }


def render_corpus_context(corpus_context: CorpusContextConfig) -> str:
    """Render corpus context into a compact prompt block."""
    parts: list[str] = []
    desc = corpus_context.description.strip()
    if desc:
        parts.append(f"Description: {desc}")
    if corpus_context.example_queries:
        parts.append("Example user queries:")
        parts.extend(f"- {q}" for q in corpus_context.example_queries)
    return "\n".join(parts).strip()


def render_merged_corpus_context(
    corpus_context: CorpusContextConfig,
    corpus_profile: dict[str, Any] | None = None,
) -> str:
    """Render manual corpus context merged with auto-derived profile data."""
    parts: list[str] = []
    desc = corpus_context.description.strip()
    if desc:
        parts.append(f"Description: {desc}")
    if corpus_context.example_queries:
        parts.append("Example user queries:")
        parts.extend(f"- {q}" for q in corpus_context.example_queries)

    profile = corpus_profile or {}
    summary = str(profile.get("summary", "")).strip()
    if summary:
        parts.append("Auto-derived corpus summary:")
        parts.append(summary)

    auto_queries = [
        str(q).strip()
        for q in profile.get("example_queries", [])
        if str(q).strip()
    ]
    if auto_queries:
        parts.append("Auto-derived likely user queries:")
        parts.extend(f"- {q}" for q in auto_queries)

    auto_cfg = corpus_context.auto_profile
    if auto_cfg.include_chunk_excerpts_per_step:
        candidates: list[str] = []
        for key in ("sampled_top_level_chunks", "sampled_random_chunks"):
            vals = profile.get(key, [])
            if isinstance(vals, list):
                candidates.extend(str(v) for v in vals if str(v).strip())
        if candidates and auto_cfg.per_step_excerpt_count > 0:
            parts.append("Sample corpus chunk excerpts:")
            for idx, text in enumerate(candidates[: auto_cfg.per_step_excerpt_count], start=1):
                excerpt = _compact_text_block(
                    text,
                    max_chars=auto_cfg.per_step_excerpt_max_chars,
                )
                if excerpt:
                    parts.append(f"- [{idx}] {excerpt}")

    return "\n".join(parts).strip()


def augment_prompt_with_corpus_context(prompt: str, corpus_context_text: str) -> str:
    """Append corpus-intent guidance to a system prompt when available."""
    ctx = corpus_context_text.strip()
    if not ctx:
        return prompt
    # Keep augmentation idempotent across repeated setup calls.
    marker = "Additional corpus context for relevance:"
    if marker in prompt:
        prompt = prompt.split(marker, 1)[0].rstrip()
    return (
        f"{prompt}\n\n"
        f"{marker}\n"
        f"{ctx}\n\n"
        "Use this context to match likely user intent, query phrasing, and domain focus."
    )


def load_sage_config(path: str) -> SagePipelineConfig:
    """Load a SagePipelineConfig from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = raw.get("models", {})
    pipeline = raw.get("pipeline", {})
    anchor_raw = raw.get("anchor", {})
    refinement_raw = raw.get("refinement", {})
    query_rewrite_raw = raw.get("query_rewrite", {})
    style_control_raw = raw.get("style_control", {}) or {}
    if not isinstance(style_control_raw, dict):
        style_control_raw = {}
    prompts_raw = raw.get("prompts", {})
    corpus_context_raw = raw.get("corpus_context", {})
    max_feedback_rounds = pipeline.get("max_feedback_rounds", 3)

    if isinstance(corpus_context_raw, str):
        corpus_context_cfg = CorpusContextConfig(description=corpus_context_raw.strip())
    elif isinstance(corpus_context_raw, dict):
        auto_profile_cfg = _parse_corpus_auto_profile(corpus_context_raw.get("auto_profile", {}))
        corpus_context_cfg = CorpusContextConfig(
            description=str(corpus_context_raw.get("description", "")).strip(),
            example_queries=_parse_example_queries(corpus_context_raw.get("example_queries", [])),
            auto_profile=auto_profile_cfg,
        )
    else:
        corpus_context_cfg = CorpusContextConfig()

    anchor_cfg = AnchorConfig(
        enabled=anchor_raw.get("enabled", True),
        bm25_enrichment_top_k=anchor_raw.get("bm25_enrichment_top_k", 5),
        bm25_enrichment_queries=anchor_raw.get("bm25_enrichment_queries", 3),
        type_distribution=anchor_raw.get("type_distribution"),
        target_hop_counts=anchor_raw.get("target_hop_counts"),
    )
    refinement_cfg = RefinementConfig(
        max_question_refinements=refinement_raw.get(
            "max_question_refinements", max_feedback_rounds
        ),
        max_anchor_regenerations=refinement_raw.get("max_anchor_regenerations", 2),
        hop_tolerance=refinement_raw.get("hop_tolerance", 0),
        min_golden_chunk_overlap=refinement_raw.get("min_golden_chunk_overlap", 0.5),
    )
    query_rewrite_cfg = SageQueryRewriteConfig(
        enabled=query_rewrite_raw.get("enabled", False),
        strategy=str(query_rewrite_raw.get("strategy", "intent")).lower(),
        max_terms=query_rewrite_raw.get("max_terms", 16),
        max_chars=query_rewrite_raw.get("max_chars", 140),
        log_rewrites=query_rewrite_raw.get("log_rewrites", False),
        model=_parse_model_config(models.get("query_rewriter", {"model": "gpt-5-mini"})),
    )
    style_control_cfg = StyleControlConfig(
        enabled=bool(style_control_raw.get("enabled", True)),
        distribution=normalize_style_distribution(style_control_raw.get("distribution")),
    )

    cfg = SagePipelineConfig(
        api_key=raw["api_key"],
        base_url=raw.get("base_url", "https://app.cgft.io"),
        corpus_id=raw.get("corpus_id", ""),
        docs_path=raw.get("docs_path", ""),
        corpus_name=raw.get("corpus_name", "sage-corpus"),
        corpus_context=corpus_context_cfg,
        question_generator=_parse_model_config(
            models.get("question_generator", {"model": "gpt-5.2"})
        ),
        search_agent=_parse_model_config(models.get("search_agent", {"model": "gpt-4.1"})),
        judge=_parse_model_config(models.get("judge", {"model": "gpt-4.1"})),
        feedback=_parse_model_config(models.get("feedback", {"model": "gpt-4.1"})),
        anchor=anchor_cfg,
        refinement=refinement_cfg,
        query_rewrite=query_rewrite_cfg,
        style_control=style_control_cfg,
        prompts=PromptConfig(
            question_generation=prompts_raw.get("question_generation", ""),
            question_generation_with_refs=prompts_raw.get("question_generation_with_refs", ""),
            search_agent=prompts_raw.get("search_agent", ""),
            judge=prompts_raw.get("judge", ""),
            feedback_incorrect=prompts_raw.get("feedback_incorrect", ""),
            feedback_too_easy=prompts_raw.get("feedback_too_easy", ""),
        ),
        num_samples=pipeline.get("num_samples", 5),
        n_search_steps=pipeline.get("n_search_steps", 2),
        n_rollouts=pipeline.get("n_rollouts", 4),
        search_max_turns=pipeline.get("search_max_turns", 16),
        search_max_tool_calls=pipeline.get("search_max_tool_calls", 24),
        search_no_answer_retries=pipeline.get("search_no_answer_retries", 2),
        strict_answer_contract=pipeline.get("strict_answer_contract", True),
        answer_contract_max_chars=pipeline.get("answer_contract_max_chars", 2000),
        answer_repair_enabled=pipeline.get("answer_repair_enabled", True),
        answer_repair_model=pipeline.get("answer_repair_model", ""),
        rollout_log_full_messages=pipeline.get("rollout_log_full_messages", False),
        rollout_log_event_meta=pipeline.get("rollout_log_event_meta", True),
        max_feedback_rounds=max_feedback_rounds,
        min_search_steps=pipeline.get("min_search_steps", 2),
        min_chunk_chars=pipeline.get("min_chunk_chars", 400),
        max_retries=pipeline.get("max_retries", 3),
        retry_delay=pipeline.get("retry_delay", 5.0),
        output=raw.get("output", "sage_output.jsonl"),
    )
    cfg.resolve_api_keys()
    cfg.prompts.resolve()
    return cfg


# ---------------------------------------------------------------------------
# Environment subclasses
# ---------------------------------------------------------------------------


class QuestionGenEnv(CgftSearchEnv):
    system_prompt = question_generation_prompt


class SearchAgentEnv(CgftSearchEnv):
    system_prompt = search_agent_prompt


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_xml_tag(text: str, tag: str) -> str | None:
    """Extract content between <tag>...</tag> (case-insensitive, dotall)."""
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def parse_generated_qa(text: str) -> dict[str, str | None]:
    """Parse generator output into question, answer, answering_steps."""
    return {
        "question": _extract_xml_tag(text, "question"),
        "answer": _extract_xml_tag(text, "answer"),
        "answering_steps": _extract_xml_tag(text, "answering steps"),
    }


def parse_search_agent_answer(text: str) -> str | None:
    """Extract <answer> from search agent output."""
    return _extract_xml_tag(text, "answer")


def parse_search_agent_answer_robust(text: str) -> str | None:
    """Extract answer from common formats used by search-agent models."""
    answer = parse_search_agent_answer(text)
    if answer:
        return answer

    tail = "\n".join(text.strip().splitlines()[-8:])
    m = re.search(r"(?im)^\s*(?:final\s+)?answer\s*[:\-]\s*(.+?)\s*$", tail)
    if m:
        candidate = m.group(1).strip()
        bad_markers = (
            "instruction",
            "dev instructions",
            "answering step",
            "<think>",
            "<search>",
            "tool:",
            "query",
        )
        if (
            candidate
            and len(candidate) <= 200
            and len(candidate.split()) <= 24
            and not any(marker in candidate.lower() for marker in bad_markers)
        ):
            return candidate
    return None


def _message_text_blocks(message: dict[str, Any]) -> str:
    """Extract text blocks from a streamed message payload."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = str(block.get("text", "")).strip()
                if txt:
                    texts.append(txt)
        return "\n".join(texts)
    return ""


def recover_search_agent_answer(messages: list[dict[str, Any]]) -> str | None:
    """Recover answer across assistant messages."""
    assistant_texts: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        text = _message_text_blocks(msg)
        if text:
            assistant_texts.append(text)

    for text in reversed(assistant_texts):
        answer = parse_search_agent_answer(text)
        if answer:
            return answer

    if assistant_texts:
        return parse_search_agent_answer_robust(assistant_texts[-1])
    return None


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    """Return text from the last assistant message, if present."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        text = _message_text_blocks(msg).strip()
        if text:
            return text
    return ""


def validate_strict_answer_contract(
    final_assistant_text: str,
    *,
    max_answer_chars: int = 2000,
) -> dict[str, Any]:
    """Validate strict final-answer format contract."""

    def _only_think_blocks(fragment: str) -> bool:
        s = (fragment or "").strip()
        if not s:
            return True
        while s:
            m = re.match(r"(?is)^<think>.*?</think>\s*", s)
            if not m:
                return False
            s = s[m.end() :].strip()
        return True

    text = (final_assistant_text or "").strip()
    if not text:
        return {"valid": False, "reason": "empty_final_message", "answer": None}

    matches = list(re.finditer(r"(?is)<answer>(.*?)</answer>", text))
    if not matches:
        return {"valid": False, "reason": "no_answer_tag", "answer": None}
    if len(matches) > 1:
        return {"valid": False, "reason": "multiple_answer_tags", "answer": None}

    m = matches[0]
    prefix = text[: m.start()].strip()
    suffix = text[m.end() :].strip()
    if prefix and not _only_think_blocks(prefix):
        return {"valid": False, "reason": "text_before_answer_tag", "answer": None}
    if suffix and not _only_think_blocks(suffix):
        return {"valid": False, "reason": "text_after_answer_tag", "answer": None}

    answer = m.group(1).strip()
    if not answer:
        return {"valid": False, "reason": "empty_answer_tag", "answer": None}
    if len(answer) > max_answer_chars:
        return {"valid": False, "reason": "answer_too_long", "answer": None}

    return {"valid": True, "reason": "ok", "answer": answer}


def _create_chat_completion_with_token_fallback(
    client: OpenAI,
    **kwargs: Any,
) -> Any:
    """Call chat.completions.create with max_tokens/max_completion_tokens fallback."""
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


def run_answer_repair(
    *,
    question: str,
    messages: list[dict[str, Any]],
    fallback_trace: str,
    client: OpenAI,
    model: str,
    max_answer_chars: int = 2000,
) -> dict[str, Any]:
    """Repair/normalize rollout output into strict <answer>...</answer> format."""
    snippets: list[str] = []
    for msg in messages[-10:]:
        role = str(msg.get("role", "?"))
        text = _message_text_blocks(msg).strip()
        if text:
            snippets.append(f"[{role}]\n{text}")
    if not snippets and fallback_trace:
        snippets.append(f"[assistant]\n{fallback_trace.strip()}")
    trace_excerpt = "\n\n".join(snippets).strip()
    if len(trace_excerpt) > 12000:
        trace_excerpt = trace_excerpt[-12000:]

    prompt = (
        "You are repairing an agent output to satisfy a strict output contract.\n"
        "Return EXACTLY one XML tag in this format and nothing else:\n"
        "<answer>...</answer>\n\n"
        "Rules:\n"
        "- Do not include <think>, <search>, explanations, markdown, or extra text.\n"
        "- Use only information from the trace.\n"
        "- If the trace does not support a confident final answer, output <answer></answer>.\n"
        f"- Keep answer at most {max_answer_chars} characters.\n\n"
        f"Question:\n{question}\n\n"
        f"Trace:\n{trace_excerpt}"
    )
    temperature = 1 if "gpt" in model.lower() else 0
    req_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 256,
        "max_completion_tokens": 256,
    }
    response = _create_chat_completion_with_token_fallback(client, **req_kwargs)
    repaired_text = (response.choices[0].message.content or "").strip()
    contract = validate_strict_answer_contract(
        repaired_text,
        max_answer_chars=max_answer_chars,
    )
    return {
        "text": repaired_text,
        "answer": contract.get("answer"),
        "contract_valid": bool(contract.get("valid")),
        "contract_reason": contract.get("reason"),
    }


def parse_judge_response(text: str) -> dict[str, Any]:
    """Parse LLM-as-a-Judge response into structured fields."""
    correct_match = re.search(r"correct:\s*(yes|no)", text, re.IGNORECASE)
    confidence_match = re.search(r"confidence:\s*(\d+)", text)
    extracted_match = re.search(r"extracted_final_answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    reasoning_match = re.search(
        r"reasoning:\s*(.+?)(?=\ncorrect:|\nconfidence:|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    return {
        "correct": (correct_match.group(1).lower() == "yes") if correct_match else False,
        "extracted_answer": extracted_match.group(1).strip() if extracted_match else None,
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else None,
        "confidence": int(confidence_match.group(1)) if confidence_match else 100,
    }


def _extract_search_queries_from_text(text: str) -> list[str]:
    """Extract search queries from text-mode tool-call traces."""
    queries: list[str] = []
    seen: set[str] = set()

    def add_query(q: str) -> None:
        q = str(q).strip().strip('"').strip("'")
        if not q:
            return
        if len(q) > 400:
            q = q[:400]
        if q not in seen:
            seen.add(q)
            queries.append(q)

    for m in re.finditer(r"(?is)<search>\s*(.*?)\s*</search>", text):
        payload = m.group(1).strip()
        qm = re.search(r'(?is)"query"\s*:\s*"([^"]+)"', payload)
        add_query(qm.group(1) if qm else payload)

    for m in re.finditer(r'(?is)<search\b[^>]*\bquery\s*=\s*"([^"]+)"', text):
        add_query(m.group(1))
    for m in re.finditer(r"(?is)<search\b[^>]*\bquery\s*=\s*'([^']+)'", text):
        add_query(m.group(1))

    for m in re.finditer(r'(?is)<search>\s*\{[^{}]*"query"\s*:\s*"([^"]+)"', text):
        add_query(m.group(1))

    for m in re.finditer(r"(?is)<tool_call>\s*(.*?)\s*</tool_call>", text):
        payload = m.group(1).strip()
        parsed_payload = None
        try:
            parsed_payload = json.loads(payload)
        except Exception:
            parsed_payload = None

        if isinstance(parsed_payload, dict):
            name = str(parsed_payload.get("name", "")).strip().lower()
            if name == "search":
                args = parsed_payload.get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if isinstance(args, dict):
                    query = args.get("query", "") or args.get("q", "")
                    add_query(str(query))
                    continue

        qm = re.search(
            r'(?is)"name"\s*:\s*"search".*?"query"\s*:\s*"([^"]+)"',
            payload,
        )
        if qm:
            add_query(qm.group(1))

    for m in re.finditer(r"(?im)tool:\s*search\b.*?query\s*[:=]\s*(.+)$", text):
        add_query(m.group(1))

    return queries


def _is_text_tool_result_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    text = _message_text_blocks(message)
    return bool(re.search(r"(?im)\btool:\s*search\b", text))


def count_search_steps_text_mode(messages: list[dict[str, Any]]) -> int:
    """Count search steps when rollouts are rendered as plain text messages."""
    query_steps = 0
    tool_steps = 0
    for msg in messages:
        text = _message_text_blocks(msg)
        if not text:
            continue
        if msg.get("role") == "assistant":
            query_steps += len(_extract_search_queries_from_text(text))
        elif _is_text_tool_result_message(msg):
            tool_steps += 1
    return max(query_steps, tool_steps)


def count_search_steps(messages: list[dict[str, Any]]) -> int:
    """Count search steps from structured tool events and text-mode traces."""
    structured_count = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if block.get("name") == "search":
                        structured_count += 1

    if structured_count > 0:
        return structured_count

    return count_search_steps_text_mode(messages)


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------


@dataclass
class RetrievalStep:
    step_number: int
    query: str
    retrieved_chunks: list[str] = field(default_factory=list)


@dataclass
class ExecutionTrace:
    steps: list[RetrievalStep]
    resolved: bool
    actual_hop_count: int
    golden_chunks: list[str]
    crosses_document_boundary: bool | None = None


def _normalize_ref_id(text: str) -> str:
    ref = str(text or "").strip().lower()
    if not ref:
        return ""
    ref = ref.replace("\\", "/")
    return ref


def _chunk_ref_id(chunk: Any) -> str:
    """Best-effort stable identifier for a chunk."""
    meta: dict[str, Any] = {}
    if hasattr(chunk, "metadata_dict"):
        meta = chunk.metadata_dict or {}
    elif hasattr(chunk, "metadata"):
        meta = chunk.metadata or {}
    elif isinstance(chunk, dict):
        meta = chunk.get("metadata", {}) or {}

    for key in ("document_id", "file_name", "file", "source", "doc_id", "filename"):
        val = meta.get(key)
        if val:
            return str(val)

    if hasattr(chunk, "hash"):
        h = getattr(chunk, "hash")
        if h:
            return str(h)

    content = chunk.content if hasattr(chunk, "content") else str(chunk)
    return str(content).strip()[:120]


def _extract_anchor_ref_ids(anchor: AnchorBundle) -> list[str]:
    return _shared_extract_anchor_ref_ids(anchor, ref_id_fn=_chunk_ref_id)


def _tool_result_to_text(content: Any) -> str:
    """Normalize a tool_result content payload into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_refs_from_search_result_text(text: str) -> list[str]:
    """Parse filenames/doc identifiers from CgftSearchEnv search tool output."""
    refs: list[str] = []
    seen: set[str] = set()
    invalid_refs = {"—", "-", "n/a", "na", "unknown", "none"}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.lower().startswith("metadata:"):
            for key in ("file_name", "filename", "document_id", "source", "doc_id"):
                km = re.search(
                    rf"""['"]{key}['"]\s*:\s*['"]([^'"]+)['"]""",
                    line,
                )
                if km:
                    candidate = km.group(1).strip()
                    if candidate and candidate.lower() not in invalid_refs:
                        cnorm = _normalize_ref_id(candidate)
                        if cnorm and cnorm not in seen:
                            seen.add(cnorm)
                            refs.append(candidate)
                    break

        m = re.match(r"^\d+\.\s+(.+?)\s+\(score:", line)
        if not m:
            m = re.match(r"^\d+\.\s+(.+?)\s+\(filtered\)", line)
        if not m:
            continue
        ref = m.group(1).strip()
        if ref.lower() in invalid_refs:
            continue
        norm = _normalize_ref_id(ref)
        if norm and norm not in seen:
            seen.add(norm)
            refs.append(ref)
    return refs


def build_execution_trace(
    messages: list[dict[str, Any]],
    *,
    resolved: bool,
) -> ExecutionTrace:
    """Build a lightweight execution trace from streamed rollout messages."""
    steps: list[RetrievalStep] = []
    for msg in messages:
        content = msg.get("content", "")

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use" and block.get("name") == "search":
                    tool_input = block.get("input", {}) or {}
                    query = str(tool_input.get("query", "") or tool_input.get("q", ""))
                    steps.append(
                        RetrievalStep(
                            step_number=len(steps) + 1,
                            query=query,
                        )
                    )
                elif btype == "tool_result" and steps:
                    content_text = _tool_result_to_text(block.get("content", ""))
                    refs = _extract_refs_from_search_result_text(content_text)
                    if refs:
                        seen = {_normalize_ref_id(r) for r in steps[-1].retrieved_chunks}
                        for ref in refs:
                            norm = _normalize_ref_id(ref)
                            if norm and norm not in seen:
                                seen.add(norm)
                                steps[-1].retrieved_chunks.append(ref)
            continue

        text = _message_text_blocks(msg)
        if not text:
            continue

        if msg.get("role") == "assistant":
            queries = _extract_search_queries_from_text(text)
            for query in queries:
                steps.append(
                    RetrievalStep(
                        step_number=len(steps) + 1,
                        query=query,
                    )
                )
        elif _is_text_tool_result_message(msg):
            if not steps:
                steps.append(
                    RetrievalStep(
                        step_number=1,
                        query="",
                    )
                )
            refs = _extract_refs_from_search_result_text(text.replace("Tool: search", ""))
            if refs:
                seen = {_normalize_ref_id(r) for r in steps[-1].retrieved_chunks}
                for ref in refs:
                    norm = _normalize_ref_id(ref)
                    if norm and norm not in seen:
                        seen.add(norm)
                        steps[-1].retrieved_chunks.append(ref)

    golden_chunks: list[str] = []
    seen_golden: set[str] = set()
    for step in steps:
        for ref in step.retrieved_chunks:
            norm = _normalize_ref_id(ref)
            if norm and norm not in seen_golden:
                seen_golden.add(norm)
                golden_chunks.append(ref)

    crosses_boundary: bool | None = None
    if golden_chunks:
        crosses_boundary = len({_normalize_ref_id(r) for r in golden_chunks}) > 1

    return ExecutionTrace(
        steps=steps,
        resolved=resolved,
        actual_hop_count=len(steps),
        golden_chunks=golden_chunks,
        crosses_document_boundary=crosses_boundary,
    )


def compute_reference_overlap(
    expected_refs: list[str],
    observed_refs: list[str],
) -> float | None:
    """Compute overlap ratio between expected and observed reference identifiers."""
    expected_norm = {_normalize_ref_id(r) for r in expected_refs if _normalize_ref_id(r)}
    observed_norm = {_normalize_ref_id(r) for r in observed_refs if _normalize_ref_id(r)}
    if not expected_norm:
        return None
    if not observed_norm:
        return None

    matched: set[str] = set()
    for exp in expected_norm:
        if exp in observed_norm:
            matched.add(exp)
            continue
        for obs in observed_norm:
            if exp in obs or obs in exp:
                matched.add(exp)
                break
    return len(matched) / len(expected_norm)


def execution_trace_to_dict(trace: ExecutionTrace) -> dict[str, Any]:
    return {
        "steps": [
            {
                "step_number": s.step_number,
                "query": s.query,
                "retrieved_chunks": s.retrieved_chunks,
            }
            for s in trace.steps
        ],
        "resolved": trace.resolved,
        "actual_hop_count": trace.actual_hop_count,
        "golden_chunks": trace.golden_chunks,
        "crosses_document_boundary": trace.crosses_document_boundary,
    }


# ---------------------------------------------------------------------------
# Rollout utilities
# ---------------------------------------------------------------------------


def _init_rollout_metrics() -> dict[str, Any]:
    """Initialize rollout consistency metrics."""
    return {
        "rollouts": 0,
        "judge_correct": 0,
        "judge_incorrect": 0,
        "contract_pass": 0,
        "contract_fail": 0,
        "strict_answers": 0,
        "repaired_answers": 0,
        "fallback_answers": 0,
        "repair_attempts": 0,
        "repair_success": 0,
        "no_answer_after_retries": 0,
        "retries_used": 0,
        "contract_fail_reasons": Counter(),
    }


def _update_rollout_metrics(
    metrics: dict[str, Any],
    rollout: dict[str, Any],
    judge_correct: bool | None = None,
    *,
    include_base: bool = True,
) -> None:
    """Update rollout consistency metrics from one rollout and optional judge outcome."""
    if include_base:
        metrics["rollouts"] += 1
        metrics["retries_used"] += max(0, int(rollout.get("answer_attempts", 1)) - 1)

        if rollout.get("contract_valid"):
            metrics["contract_pass"] += 1
        else:
            metrics["contract_fail"] += 1
            reason = str(rollout.get("contract_reason") or "unknown")
            metrics["contract_fail_reasons"][reason] += 1

        source = str(rollout.get("answer_source") or "none")
        if source == "strict":
            metrics["strict_answers"] += 1
        elif source == "repair":
            metrics["repaired_answers"] += 1
        elif source == "fallback":
            metrics["fallback_answers"] += 1

        if rollout.get("answer_repair_attempted"):
            metrics["repair_attempts"] += 1
        if rollout.get("answer_repair_success"):
            metrics["repair_success"] += 1

        if not (rollout.get("answer") or "").strip():
            metrics["no_answer_after_retries"] += 1

    if judge_correct is True:
        metrics["judge_correct"] += 1
    elif judge_correct is False:
        metrics["judge_incorrect"] += 1


def bundle_environment(
    env_class: type[CgftSearchEnv],
    constructor_args: dict[str, Any],
    pip_dependencies: list[str] | None = None,
) -> tuple[bytes, bytes]:
    """Bundle an env class and return (env_cls_bytes, env_metadata_bytes)."""
    from benchmax.bundle.bundler import bundle_env, write_bundle_files

    deps = pip_dependencies or ["aiohttp"]
    bundle = bundle_env(
        env_class,
        pip_dependencies=deps,
        local_modules=[synthetic_data_prep],
        constructor_args=constructor_args,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        pkl_path = tmp_path / "env-cls.pkl"
        meta_path = tmp_path / "env-metadata.json"
        write_bundle_files(bundle, pkl_path, meta_path)
        return pkl_path.read_bytes(), meta_path.read_bytes()


def _is_retryable(exc: Exception | None = None, result: dict[str, Any] | None = None) -> bool:
    """Check whether a rollout failure is transient and worth retrying."""
    if exc is not None:
        msg = str(exc)
        if "HTTP 424" in msg:
            return True
        if any(f"HTTP {c}" in msg for c in ("429", "500", "502", "503", "504")):
            return True
    if result is not None:
        event = result.get("event", "")
        error = str(result.get("error", ""))
        if event in ("worker_error", "error"):
            if "424" in error or "finish_reason" in error.lower():
                return True
    return False


def _stream_rollout_with_retry(
    rollout_client: RolloutClient,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Wrapper around stream_rollout that retries on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = rollout_client.stream_rollout(**kwargs)
            if _is_retryable(result=result) and attempt < max_retries:
                time.sleep(retry_delay)
                continue
            return result
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc=exc) and attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------


def run_question_generation(
    chunk_text: str,
    n_search_steps: int,
    corpus_context: str,
    rollout_client: RolloutClient,
    env_cls_bytes: bytes,
    env_metadata_bytes: bytes,
    model_cfg: SageModelConfig,
    rollout_log_full_messages: bool = False,
    rollout_log_event_meta: bool = True,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict[str, Any]:
    """Step 1: Generate a multi-hop question from a seed chunk (rollout-based)."""
    question_input = f"Initial document: {chunk_text}\nn_search_step: {n_search_steps}"
    if corpus_context.strip():
        question_input += (
            "\n\nCorpus context for relevance:\n"
            f"{corpus_context}\n"
            "Generate a question that matches likely user search intent for this corpus."
        )
    raw_example = {
        "question": question_input,
        "answer": "",
    }
    result = _stream_rollout_with_retry(
        rollout_client,
        max_retries=max_retries,
        retry_delay=retry_delay,
        raw_example=raw_example,
        env_cls_bytes=env_cls_bytes,
        env_metadata_bytes=env_metadata_bytes,
        max_turns=8,
        max_tool_calls=12,
        capture_messages=True,
        llm_api_key=model_cfg.api_key,
        llm_base_url=model_cfg.base_url,
        llm_model=model_cfg.model,
        max_completion_tokens=5128,
        full_messages=rollout_log_full_messages,
        include_event_meta=rollout_log_event_meta,
    )
    final_text = result.get("final_assistant_text", "")
    parsed = parse_generated_qa(final_text)
    return {
        **parsed,
        "generator_trace": final_text,
        "generator_messages": result.get("messages", []),
        "success": result.get("success", False),
    }


def _generate_bm25_queries(chunk: Any, n: int = 3) -> list[str]:
    """Generate BM25 search queries from chunk content without LLM calls."""
    return _shared_generate_bm25_queries(chunk, n=n)


def _format_reference_section(anchor: AnchorBundle) -> str:
    """Format reference chunks for the generation prompt."""
    parts = []
    for i, chunk in enumerate(anchor.secondary_chunks, 1):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        parts.append(f"Reference document [{i}]:\n{content}")

    bm25_related = anchor.structural_hints.get("bm25_related", [])
    offset = len(anchor.secondary_chunks)
    for i, chunk in enumerate(bm25_related, offset + 1):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        parts.append(f"Reference document [{i}] (related via search):\n{content}")

    if not parts:
        return ""
    return (
        "Reference documents (use these to create a multi-hop question that "
        "connects information across documents):\n\n" + "\n\n".join(parts)
    )


def _select_anchor_bundle_with_enrichment(
    *,
    selector: Any,  # AnchorSelector
    primary_chunk: Any,
    corpus_pool: list[Any],
    source: Any,  # CorporaChunkSource
    cfg: SagePipelineConfig,
    qa_type: str | None = None,
) -> AnchorBundle:
    """Select an anchor bundle and attach BM25-related chunks."""
    result = _shared_select_anchor_bundle_with_enrichment(
        selector=selector,
        primary_chunk=primary_chunk,
        corpus_pool=corpus_pool,
        source=source,
        bm25_enrichment_queries=cfg.anchor.bm25_enrichment_queries,
        bm25_enrichment_top_k=cfg.anchor.bm25_enrichment_top_k,
        max_related_refs=3,
        qa_type=qa_type,
        include_search_payload=False,
    )
    if isinstance(result, tuple):
        return result[0]
    return result


def run_question_generation_direct(
    anchor: AnchorBundle,
    client: OpenAI,
    model_cfg: SageModelConfig,
    prompt_template: str = "",
    corpus_context: str = "",
) -> dict[str, Any]:
    """Generate Q/A via direct LLM call using pre-selected reference chunks."""
    primary_content = (
        anchor.primary_chunk.content
        if hasattr(anchor.primary_chunk, "content")
        else str(anchor.primary_chunk)
    )
    reference_section = _format_reference_section(anchor)

    hints_parts = []
    for k, v in anchor.structural_hints.items():
        if k == "bm25_related":
            continue
        hints_parts.append(f"- {k}: {v}")
    hints_str = "\n".join(hints_parts) if hints_parts else "None"  # noqa: F841

    qa_type_display = anchor.target_qa_type.replace("_", " ")

    template = prompt_template or question_generation_with_refs_prompt
    prompt = template.format(
        target_qa_type=qa_type_display,
        target_hop_count=anchor.target_hop_count,
        primary_chunk=primary_content,
        reference_section=reference_section,
        corpus_context=corpus_context or "No corpus context provided.",
    )

    try:
        req_kwargs: dict[str, Any] = {
            "model": model_cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0,
            "max_tokens": 4096,
            "max_completion_tokens": 4096,
        }
        response = _create_chat_completion_with_token_fallback(client, **req_kwargs)
        final_text = response.choices[0].message.content or ""
    except Exception:
        final_text = ""

    parsed = parse_generated_qa(final_text)
    return {
        **parsed,
        "generator_trace": final_text,
        "generator_messages": [],
        "success": bool(parsed.get("question")),
    }


def run_search_agent(
    question: str,
    rollout_client: RolloutClient,
    env_cls_bytes: bytes,
    env_metadata_bytes: bytes,
    model_cfg: SageModelConfig,
    n_rollouts: int = 4,
    search_max_turns: int = 16,
    search_max_tool_calls: int = 24,
    search_no_answer_retries: int = 2,
    strict_answer_contract: bool = True,
    answer_contract_max_chars: int = 2000,
    answer_repair_enabled: bool = True,
    answer_repair_client: OpenAI | None = None,
    answer_repair_model: str = "",
    rollout_log_full_messages: bool = False,
    rollout_log_event_meta: bool = True,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> list[dict[str, Any]]:
    """Step 2: Run N independent search agent rollouts attempting to answer the question."""
    results = []
    for i in range(n_rollouts):
        base_question = f"Question: {question}"
        max_attempts = max(1, search_no_answer_retries + 1)
        rollout_record: dict[str, Any] | None = None
        for attempt in range(max_attempts):
            if attempt > 0:
                raw_example = {
                    "question": (
                        f"{base_question}\n\n"
                        "IMPORTANT: You MUST search before answering. Do NOT answer "
                        "from your own knowledge. Your first message must include a "
                        "search tool call like: <tool_call>\n"
                        '{"name":"search","arguments":{"query":"your query here"}}\n'
                        "</tool_call>\n"
                        "After retrieving results, provide your answer inside "
                        "<answer>...</answer>."
                    ),
                    "answer": "",
                }
            else:
                raw_example = {"question": base_question, "answer": ""}
            try:
                result = _stream_rollout_with_retry(
                    rollout_client,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    raw_example=raw_example,
                    env_cls_bytes=env_cls_bytes,
                    env_metadata_bytes=env_metadata_bytes,
                    max_turns=search_max_turns,
                    max_tool_calls=search_max_tool_calls,
                    capture_messages=True,
                    llm_api_key=model_cfg.api_key,
                    llm_base_url=model_cfg.base_url,
                    llm_model=model_cfg.model,
                    max_completion_tokens=5128,
                    full_messages=rollout_log_full_messages,
                    include_event_meta=rollout_log_event_meta,
                    example_index=i,
                )
                final_text = result.get("final_assistant_text", "")
                messages = result.get("messages", [])
                final_assistant_text = _last_assistant_text(messages) or final_text
                contract = validate_strict_answer_contract(
                    final_assistant_text,
                    max_answer_chars=answer_contract_max_chars,
                )

                answer = None
                answer_source = "none"
                repair_attempted = False
                repair_success = False
                repair_contract_reason = None

                if strict_answer_contract:
                    if contract["valid"]:
                        answer = contract["answer"]
                        answer_source = "strict"
                    elif (
                        answer_repair_enabled
                        and answer_repair_client is not None
                        and answer_repair_model
                    ):
                        repair_attempted = True
                        try:
                            repair = run_answer_repair(
                                question=question,
                                messages=messages,
                                fallback_trace=final_text,
                                client=answer_repair_client,
                                model=answer_repair_model,
                                max_answer_chars=answer_contract_max_chars,
                            )
                            repair_contract_reason = repair.get("contract_reason")
                            if (
                                repair.get("contract_valid")
                                and (repair.get("answer") or "").strip()
                            ):
                                answer = str(repair["answer"]).strip()
                                answer_source = "repair"
                                repair_success = True
                        except Exception as exc:
                            repair_contract_reason = f"repair_error:{exc.__class__.__name__}"
                    if not (answer or "").strip():
                        recovered = parse_search_agent_answer(final_assistant_text)
                        if not recovered:
                            recovered = recover_search_agent_answer(messages)
                        if (recovered or "").strip():
                            answer = str(recovered).strip()
                            answer_source = "fallback"
                else:
                    answer = parse_search_agent_answer_robust(final_text)
                    if not answer:
                        answer = recover_search_agent_answer(messages)
                    if answer:
                        answer_source = "fallback"

                contract_reason = str(contract.get("reason") or "unknown")
                if repair_success:
                    contract_reason = "ok_after_repair"
                elif repair_contract_reason and not contract["valid"]:
                    contract_reason = f"{contract_reason}|{repair_contract_reason}"

                rollout_record = {
                    "answer": answer,
                    "trace": final_text,
                    "messages": messages,
                    "search_step_count": count_search_steps(messages),
                    "text_mode_trace": bool(
                        count_search_steps_text_mode(messages)
                        and not any(isinstance(m.get("content", ""), list) for m in messages)
                    ),
                    "success": result.get("success", False),
                    "answer_attempts": attempt + 1,
                    "contract_valid": bool(contract["valid"] or repair_success),
                    "contract_reason": contract_reason,
                    "answer_source": answer_source,
                    "answer_repair_attempted": repair_attempted,
                    "answer_repair_success": repair_success,
                }
                if (answer or "").strip():
                    break
            except Exception:
                rollout_record = {
                    "answer": None,
                    "trace": "",
                    "messages": [],
                    "search_step_count": 0,
                    "success": False,
                    "answer_attempts": attempt + 1,
                    "contract_valid": False,
                    "contract_reason": "rollout_error",
                    "answer_source": "none",
                    "answer_repair_attempted": False,
                    "answer_repair_success": False,
                }
                break
        results.append(
            rollout_record
            or {
                "answer": None,
                "trace": "",
                "messages": [],
                "search_step_count": 0,
                "success": False,
                "answer_attempts": 0,
                "contract_valid": False,
                "contract_reason": "no_record",
                "answer_source": "none",
                "answer_repair_attempted": False,
                "answer_repair_success": False,
            }
        )
    return results


def run_llm_judge(
    question: str,
    gold_answer: str,
    agent_answer: str,
    client: OpenAI,
    model: str,
    prompt_template: str = "",
) -> dict[str, Any]:
    """Step 3: LLM-as-a-Judge — compare the search agent's answer to the gold answer."""
    template = prompt_template or llmaaj_prompt
    prompt = template.replace("{{QUESTION}}", question)
    prompt = prompt.replace("{{MODEL_ANSWER}}", agent_answer or "None")
    prompt = prompt.replace("{{GOLD_ANSWER}}", gold_answer)
    judge_temperature = 1 if "gpt" in model.lower() else 0
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=judge_temperature,
        max_completion_tokens=1024,
    )
    judge_text = response.choices[0].message.content or ""
    return parse_judge_response(judge_text)


def aggregate_best_of_n(
    rollout_results: list[dict[str, Any]],
    judge_results: list[dict[str, Any]],
    target_search_steps: int,
) -> dict[str, Any]:
    """Step 4: Aggregate best-of-N results and determine pass/fail."""
    best_idx = None
    best_steps = 0

    for i, (rollout, judge) in enumerate(zip(rollout_results, judge_results)):
        if judge["correct"]:
            if best_idx is None or rollout["search_step_count"] >= best_steps:
                best_idx = i
                best_steps = rollout["search_step_count"]

    if best_idx is not None:
        return {
            "best_of_n": 1,
            "best_answer": rollout_results[best_idx]["answer"],
            "best_trace": rollout_results[best_idx]["trace"],
            "best_messages": rollout_results[best_idx]["messages"],
            "best_n_search_steps": best_steps,
            "pass_check": best_steps >= target_search_steps,
        }

    fallback_idx = max(
        range(len(rollout_results)),
        key=lambda i: rollout_results[i]["search_step_count"],
    )
    return {
        "best_of_n": 0,
        "best_answer": rollout_results[fallback_idx]["answer"],
        "best_trace": rollout_results[fallback_idx]["trace"],
        "best_messages": rollout_results[fallback_idx]["messages"],
        "best_n_search_steps": rollout_results[fallback_idx]["search_step_count"],
        "pass_check": False,
    }


def run_feedback(
    qa_data: dict[str, Any],
    best_rollout: dict[str, Any],
    target_steps: int,
    feedback_type: str,
    client: OpenAI,
    model: str,
    prompt_incorrect: str = "",
    prompt_too_easy: str = "",
    search_agent_prompt_override: str = "",
    corpus_context: str = "",
) -> dict[str, Any]:
    """Step 5: Refine the question/answer based on feedback type."""
    if feedback_type == "incorrect":
        tmpl = prompt_incorrect or execution_feedback_incorrect_prompt
    else:
        tmpl = prompt_too_easy or execution_feedback_too_easy_prompt

    prompt = tmpl.replace("target_step", str(target_steps))
    prompt = prompt.replace("{{CURRENT_QUESTION}}", qa_data.get("question", ""))
    prompt = prompt.replace("{{CURRENT_ANSWER}}", qa_data.get("answer", ""))
    sa_prompt = search_agent_prompt_override or search_agent_prompt
    prompt = prompt.replace(
        "{{DATA_GENERATOR_PROMPT}}",
        qa_data.get("original_generator_prompt", qa_data.get("generator_trace", "")),
    )
    prompt = prompt.replace("{{DATA_GENERATOR_RESPONSE}}", qa_data.get("generator_trace", ""))
    prompt = prompt.replace("{{SEARCH_AGENT_PROMPT}}", sa_prompt)
    prompt = prompt.replace("{{SEARCH_AGENT_RESPONSE}}", best_rollout.get("best_trace", ""))
    search_steps = best_rollout.get("best_n_search_steps", 0)
    if search_steps == 0:
        prompt += (
            "\n\nWARNING: The search agent performed 0 search steps — its answer "
            "is entirely from parametric knowledge and should NOT be trusted. "
            "Keep the current gold answer unchanged and only refine the question "
            "to make it more searchable."
        )
    if corpus_context.strip():
        prompt += (
            "\n\n# Corpus context\n"
            f"{corpus_context}\n"
            "Keep the refined question aligned with this intended user context."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_completion_tokens=4096,
    )
    feedback_text = response.choices[0].message.content or ""
    parsed = parse_generated_qa(feedback_text)

    updated = dict(qa_data)
    if parsed["question"]:
        updated["question"] = parsed["question"]
    if parsed["answer"]:
        updated["answer"] = parsed["answer"]
    if parsed["answering_steps"]:
        updated["answering_steps"] = parsed["answering_steps"]
    updated["feedback_trace"] = feedback_text
    return updated


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------


def _wrap(text: str, width: int = 80, indent: str = "  ") -> str:
    """Wrap text with hanging indent, preserving explicit newlines."""
    import textwrap

    lines = text.split("\n")
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(
            textwrap.wrap(
                line,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
            )
            or [indent]
        )
    return "\n".join(wrapped)


def write_report(
    path: Path,
    results: list[dict[str, Any]],
    stats: dict[str, Any],
    cfg: SagePipelineConfig,
) -> None:
    """Write a human-readable plain-text report alongside the JSONL output."""
    lines: list[str] = []
    w = lines.append

    w("SAGE Pipeline Report")
    w("=" * 70)
    w("")

    w("Summary")
    w("-" * 70)
    w(f"  Total seed chunks:        {stats['total']}")
    w(f"  Passed (all rounds):      {stats['passed']}")
    w(f"    Strict pass:            {stats.get('strict_passed', 0)}")
    w(f"    Relaxed pass:           {stats.get('relaxed_passed', 0)}")
    for i, count in enumerate(stats["rounds"]):
        if count > 0:
            w(f"    Round {i}:                {count}")
    w(f"  Dropped by min-step:      {stats.get('dropped_by_min_step', 0)}")
    w(f"  After min-step filter:    {len(results)}")
    fail_reasons = stats.get("failure_reasons", {})
    if fail_reasons:
        w("  Failure reasons:")
        for key, val in fail_reasons.items():
            if val:
                w(f"    {key}:".ljust(30) + f"{val}")
    w("")

    w("Configuration")
    w("-" * 70)
    w(f"  Target search steps:      {cfg.n_search_steps}")
    w(f"  Best-of-N rollouts:       {cfg.n_rollouts}")
    w(f"  Search max turns:         {cfg.search_max_turns}")
    w(f"  Search max tool calls:    {cfg.search_max_tool_calls}")
    w(f"  Search no-answer retries: {cfg.search_no_answer_retries}")
    w(f"  Strict answer contract:   {cfg.strict_answer_contract}")
    w(f"  Answer contract max chars: {cfg.answer_contract_max_chars}")
    w(f"  Answer repair enabled:    {cfg.answer_repair_enabled}")
    if cfg.answer_repair_enabled:
        w(f"  Answer repair model:      {cfg.answer_repair_model or cfg.judge.model}")
    w(f"  Log full rollout msgs:    {cfg.rollout_log_full_messages}")
    w(f"  Log rollout event meta:   {cfg.rollout_log_event_meta}")
    w(f"  Max question refinements: {cfg.refinement.max_question_refinements}")
    w(f"  Max anchor regenerations: {cfg.refinement.max_anchor_regenerations}")
    w(f"  Hop tolerance:            {cfg.refinement.hop_tolerance}")
    w(f"  Min golden overlap:       {cfg.refinement.min_golden_chunk_overlap}")
    w(f"  Min search steps filter:  {cfg.min_search_steps}")
    w(f"  Anchor selection:         {'enabled' if cfg.anchor.enabled else 'disabled'}")
    w(f"  Query rewrite:            {'enabled' if cfg.query_rewrite.enabled else 'disabled'}")
    if cfg.query_rewrite.enabled:
        w(f"    Strategy:               {cfg.query_rewrite.strategy}")
        w(
            "    Limits:                 "
            f"max_terms={cfg.query_rewrite.max_terms}, "
            f"max_chars={cfg.query_rewrite.max_chars}"
        )
        w(f"    Log rewrites:           {cfg.query_rewrite.log_rewrites}")
    w(
        "  Corpus context:          "
        f"{'provided' if render_corpus_context(cfg.corpus_context) else 'not provided'}"
    )
    if cfg.corpus_context.description:
        w(f"    Description:            {cfg.corpus_context.description}")
    if cfg.corpus_context.example_queries:
        w(f"    Example queries:        {len(cfg.corpus_context.example_queries)}")
    w("")
    w("  Models:")
    w(f"    Question generator:     {cfg.question_generator.model}")
    w(f"    Search agent:           {cfg.search_agent.model}")
    w(f"    Judge:                  {cfg.judge.model}")
    w(f"    Feedback:               {cfg.feedback.model}")
    if cfg.query_rewrite.enabled and cfg.query_rewrite.strategy == "llm":
        w(f"    Query rewriter:         {cfg.query_rewrite.model.model}")
    w("")

    rollout_metrics = stats.get("rollout_metrics", {})
    if rollout_metrics:
        w("Rollout Metrics")
        w("-" * 70)
        total_rollouts = rollout_metrics.get("rollouts", 0)
        judge_correct = rollout_metrics.get("judge_correct", 0)
        judge_incorrect = rollout_metrics.get("judge_incorrect", 0)
        denom = max(1, judge_correct + judge_incorrect)
        w(f"  Total rollout candidates: {total_rollouts}")
        w(f"  Judge correct:            {judge_correct}")
        w(f"  Judge incorrect:          {judge_incorrect}")
        w(f"  Judge correct rate:       {(judge_correct / denom) * 100:.1f}%")
        w(f"  Contract pass:            {rollout_metrics.get('contract_pass', 0)}")
        w(f"  Contract fail:            {rollout_metrics.get('contract_fail', 0)}")
        w(f"  Strict answers:           {rollout_metrics.get('strict_answers', 0)}")
        w(f"  Repaired answers:         {rollout_metrics.get('repaired_answers', 0)}")
        w(f"  Fallback answers:         {rollout_metrics.get('fallback_answers', 0)}")
        w(f"  Repair attempts:          {rollout_metrics.get('repair_attempts', 0)}")
        w(f"  Repair success:           {rollout_metrics.get('repair_success', 0)}")
        w(f"  No answer after retries:  {rollout_metrics.get('no_answer_after_retries', 0)}")
        w(f"  Retries used:             {rollout_metrics.get('retries_used', 0)}")
        reasons = rollout_metrics.get("contract_fail_reasons", {})
        if reasons:
            w("  Contract fail reasons:")
            for reason, count in sorted(reasons.items(), key=lambda x: (-x[1], x[0])):
                if count:
                    w(f"    {reason}: {count}")
        w("")

    rollout_metrics_by_type = stats.get("rollout_metrics_by_type", {})
    if rollout_metrics_by_type:
        w("Rollout Metrics By Type")
        w("-" * 70)
        for qa_type, m in sorted(rollout_metrics_by_type.items()):
            judge_total = max(1, m.get("judge_correct", 0) + m.get("judge_incorrect", 0))
            w(
                f"  {qa_type}: rollouts={m.get('rollouts', 0)}, "
                f"judge_correct_rate="
                f"{(m.get('judge_correct', 0) / judge_total) * 100:.1f}%, "
                f"contract_fail={m.get('contract_fail', 0)}, "
                f"repair_success={m.get('repair_success', 0)}, "
                f"no_answer={m.get('no_answer_after_retries', 0)}"
            )
        w("")

    anchor_quality = stats.get("anchor_quality", {})
    if anchor_quality:
        w("Anchor Quality")
        w("-" * 70)
        for qa_type, qstats in sorted(anchor_quality.items()):
            w(
                f"  {qa_type}: attempts={qstats.get('attempts', 0)}, "
                f"passes={qstats.get('passes', 0)}, "
                f"regenerations={qstats.get('regenerations', 0)}, "
                f"discards={qstats.get('discards', 0)}"
            )
        w("")

    if not results:
        w("No questions passed the pipeline filters.")
    else:
        w(f"Generated Questions ({len(results)})")
        w("-" * 70)
        for i, r in enumerate(results, 1):
            status_map = {
                "pass": "PASS",
                "pass_reanchored": "PASS (reanchored)",
                "pass_relaxed": "PASS (relaxed)",
            }
            status_label = status_map.get(r.get("status", ""), r.get("status", "PASS"))
            w("")
            qa_type = r.get("target_qa_type", "")
            type_str = f"  |  type: {qa_type}" if qa_type and qa_type != "unknown" else ""
            w(
                f"  [{i}] {status_label}  |  round {r['round']}  |  "
                f"{r['search_steps']} search steps{type_str}"
            )
            w("")
            w("  Q: " + r["question"])
            w("  A: " + r["answer"])
            w("")
            chunk_preview = r.get("source_chunk", "")
            if chunk_preview:
                w("  Source chunk (excerpt):")
                w(_wrap(chunk_preview[:300], width=70, indent="    "))
            w("")
            w("  " + "." * 66)

    w("")
    path.write_text("\n".join(lines) + "\n")
