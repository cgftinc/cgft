"""Typed models and config loaders for the Cgft QA pipeline."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from cgft_utils.qa_generation.anchor_selector import (
    DEFAULT_TARGET_HOP_COUNTS,
)
from cgft_utils.qa_generation.style_controls import (
    QUERY_STYLE_EXPERT,
    QUERY_STYLE_KEYWORD,
    QUERY_STYLE_NATURAL,
    normalize_style_distribution,
)
from cgft_utils.qa_generation.retrieval_query import QueryRewriteConfig

CORPUS_SYSTEM_PROMPT = (
    "You are a technical analyst specializing in document corpus analysis.\n"
    "Your goal is to understand the overall themes, content type, and typical query that a user might search for.\n"
    "\n"
    "Based on the context and thoughts, form an understanding of the corpus, and then provide a short summary and "
    "example search queries. Weigh the user's input if provided.\n"
    "\n"
    "Guideline:\n"
    "- Try to generalize and identify the overall theme of the corpus.\n"
    "- Use your context of the theme, domain, etc. to guess the persona and purpose of searching this corpus\n"
    "- i.e.\n"
    "    - student finding their learning notes\n"
    "    - developer looking up API documentation\n"
    "    - journalist researching a topic\n"
    "    - business analyst gathering market research\n"
    "- Use that persona to guess the types of queries they might perform.\n"
    "\n"
    "For the summary:\n"
    "- First line - corpus themes (documentation, tutorials, reference, etc.)\n"
    "- Second line - content domain (technical, business, scientific, etc.)\n"
    "- Third line - user persona and purpose (likely developer looking up API documentation)\n"
    "- Do NOT cite specific chunk content in the summary.\n"
    "\n"
    "For the example queries:\n"
    "- Provide 5-10 realistic example queries a user might search for in this corpus.\n"
    "- Use the inferred user persona and purpose to guide the query style.\n"
    "- Queries can have incomplete information, as often users do not remember full context.\n"
    "\n"
    'Return JSON with:\n'
    '- thoughts: Your analysis and reasoning here\n'
    '- summary: our summary here (3 lines as described above)\n'
    '- example_queries: List of example queries in the form of ["query1", "query2", ...]'
)

CORPUS_USER_TEMPLATE = (
    "Analyze the following document corpus:\n"
    "\n"
    "<user_context>\n"
    "{user_context}\n"
    "</user_context>\n"
    "\n"
    "<top_level_chunks>\n"
    "{top_level_content}\n"
    "</top_level_chunks>\n"
    "\n"
    "<random_sampled_chunks>\n"
    "{random_content}\n"
    "</random_sampled_chunks>\n"
    "\n"
    "Return your analysis as JSON with keys: thoughts, summary, example_queries"
)

DEFAULT_QA_TYPE_DISTRIBUTION = {
    "lookup": 0.333,
    "co_located_multi_hop": 0.200,
    "cross_document_multi_hop": 0.333,
    "sequential_reasoning": 0.133,
    "synthesis": 0.0,
}

DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating whether retrieved text chunks contain ALL information \
needed to fully answer a question.

Rules:
- The answer must be FULLY supported by the retrieved chunks. Partial information is NOT enough.
- For multi-hop questions, ALL intermediate facts must be present in the chunks.
- Do not use your own knowledge — only consider what is explicitly in the chunks.

Respond with JSON only:
{"answerable": <bool>, "confidence": <float 0-1>, "reasoning": "<brief explanation>"}"""

DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE = """\
Question: {question}

Gold answer: {answer}

Retrieved chunks:
{chunks_text}

Can the question be fully answered using ONLY the retrieved chunks above?"""

DEFAULT_GROUNDING_JUDGE_SYSTEM_PROMPT = """\
You are an expert grounding judge evaluating whether a gold answer is fully supported by provided evidence chunks.

Rules:
- Use only the provided evidence chunks.
- The answer must be FULLY supported. Partial support is not enough.
- For multi-hop answers, all intermediate facts must be present in the evidence.
- Do not use external knowledge.
- Identify which specific chunk IDs provide the supporting evidence.
- CRITICAL: Reject meta-questions about the QA generation process itself. If the question asks \
about "chunks", "multi-hop questions", "generation", "retrieval", or why a question "cannot be \
created/generated", return answerable=false. These are invalid QA pairs that describe generation \
failures rather than real domain questions about the corpus content.

Respond with JSON only:
{"answerable": <bool>, "confidence": <float 0-1>, "reasoning": "<brief explanation>", "supporting_chunk_ids": ["<id1>", "<id2>", ...]}"""

DEFAULT_GROUNDING_JUDGE_USER_TEMPLATE = """\
Question: {question}

Gold answer: {answer}

Evidence chunks:
{chunks_text}

Can the gold answer be fully grounded using ONLY the evidence chunks above?"""

# Strict prompts preserved for opt-in use.
STRICT_RETRIEVAL_JUDGE_SYSTEM_PROMPT = """\
You are a retrieval-difficulty and grounding judge for BM25-style search.

Rules:
- Use only retrieved chunks as evidence.
- Return answerable=true only for FAIL cases in this filter.
- Use reason_tag values:
  - too_easy_lexical: answer is supported and naive lexical retrieval is clearly sufficient.
  - unsupported: chunks do not fully support the gold answer.
  - challenging_answerable_pass: supported, but retrieval is not naively easy.
- If shortcut evidence is weak or uncertain, choose challenging_answerable_pass.

Return JSON only with keys:
{"answerable": <bool>, "confidence": <float 0-1>, "reason_tag": "<tag>", "reasoning": "<short explanation>", "lexical_anchor_evidence": "<optional>", "support_gap": "<optional>"}"""

STRICT_RETRIEVAL_JUDGE_USER_TEMPLATE = """\
Question: {question}

Gold answer: {answer}

Retrieved chunks:
{chunks_text}

Evaluate in order:
1) Support check: can the full answer be grounded in the chunks?
2) If unsupported, set answerable=true and reason_tag=unsupported.
3) If supported, check lexical shortcut risk for naive BM25 retrieval.
4) If shortcut is clearly sufficient, set answerable=true and reason_tag=too_easy_lexical.
5) Otherwise set answerable=false and reason_tag=challenging_answerable_pass."""


@dataclass
class ModelConfig:
    """Model endpoint settings."""

    model: str = "gpt-5-mini"
    api_key: str = ""
    base_url: str = "https://app.cgft.io/api/llm"


@dataclass
class RolloutLimits:
    """Rollout execution limits."""

    max_turns: int = 16
    max_tool_calls: int = 24
    max_completion_tokens: int = 2048
    timeout: float = 120.0


@dataclass
class EnvBundleConfig:
    """Environment bundle selectors for rollout components."""

    env_cls_path: str = ""
    env_metadata_path: str = ""
    env_cls_file: str = ""
    env_metadata_file: str = ""

    def as_bytes_bundle(self) -> tuple[bytes | None, bytes | None]:
        cls_bytes: bytes | None = None
        meta_bytes: bytes | None = None
        if self.env_cls_file:
            cls_bytes = Path(self.env_cls_file).read_bytes()
        if self.env_metadata_file:
            meta_bytes = Path(self.env_metadata_file).read_bytes()
        return cls_bytes, meta_bytes

    def has_paths(self) -> bool:
        return bool(self.env_cls_path and self.env_metadata_path)

    def has_files(self) -> bool:
        return bool(self.env_cls_file and self.env_metadata_file)


@dataclass
class PlatformConfig:
    """Top-level credentials."""

    api_key: str
    base_url: str = "https://app.cgft.io"
    llm_api_key: str = ""
    llm_base_url: str = "https://app.cgft.io/api/llm"


@dataclass
class CorpusConfig:
    """Corpus source selection."""

    docs_path: str = ""
    corpus_id: str = ""
    corpus_name: str = "cgft-corpus"
    show_summary: bool = True
    min_chunk_chars: int = 400


@dataclass
class EntityExtractionConfig:
    """LLM-generated patterns for extracting linkable entities from chunk content."""

    entity_names: list[str] = field(default_factory=list)
    code_patterns: dict[str, str] = field(default_factory=dict)
    domain_terms: list[str] = field(default_factory=list)
    query_templates: list[str] = field(default_factory=list)
    confidence: str = "low"  # "low" | "medium" | "high"


@dataclass
class EntityExtractionLLMConfig:
    """LLM settings for the entity extraction call."""

    model: str = ""
    api_key: str = ""
    base_url: str = ""


@dataclass
class CorpusContextConfig:
    """Optional corpus-intent and auto-profile controls for prompt conditioning."""

    enabled: bool = True
    description: str = ""
    example_queries: list[str] = field(default_factory=list)
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    num_top_level_samples: int = 4
    num_random_samples: int = 4
    min_chunk_chars: int = 400
    system_prompt: str = CORPUS_SYSTEM_PROMPT
    user_template: str = CORPUS_USER_TEMPLATE
    generate_entity_patterns: bool = True
    entity_extraction: EntityExtractionConfig | None = None
    entity_extraction_llm: EntityExtractionLLMConfig = field(default_factory=EntityExtractionLLMConfig)


@dataclass
class TargetsConfig:
    """Target matrix settings for qa_type generation."""

    total_samples: int = 200
    qa_type_distribution: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_QA_TYPE_DISTRIBUTION)
    )

    def normalized_qa_type_distribution(self) -> dict[str, float]:
        total = sum(v for v in self.qa_type_distribution.values() if v >= 0)
        if total <= 0:
            dist = dict(DEFAULT_QA_TYPE_DISTRIBUTION)
            total = sum(dist.values())
            return {k: v / total for k, v in dist.items()}
        return {
            qa_type: max(0.0, float(weight)) / total
            for qa_type, weight in self.qa_type_distribution.items()
        }


@dataclass
class StructuralLinkerConfig:
    """Config for `StructuralChunkLinker`."""

    type_distribution: dict[str, float] | None = None
    target_hop_counts: dict[str, int] | None = None
    bm25_enrichment_queries: int = 3
    bm25_enrichment_top_k: int = 5
    max_related_refs: int = 3
    corpus_pool_size: int = 200


@dataclass
class LLMGuidedLinkerConfig:
    """Config for `LLMGuidedChunkLinker`."""

    model: str = "gpt-5-mini"
    api_key: str = ""
    base_url: str = ""
    system_prompt: str = ""
    user_template: str = ""
    top_k_bm25: int = 5
    top_related_chunks: int = 3
    context_preview_chars: int = 200


@dataclass
class LinkerConfig:
    """Chunk-linking mode selection."""

    type: str = "structural"
    structural: StructuralLinkerConfig = field(default_factory=StructuralLinkerConfig)
    llm_guided: LLMGuidedLinkerConfig = field(default_factory=LLMGuidedLinkerConfig)


@dataclass
class LLMDirectGenerationConfig:
    """Direct LLM generation controls."""

    model: str = "gpt-5.2"
    api_key: str = ""
    base_url: str = ""
    max_completion_tokens: int = 4096
    timeout: float = 120.0
    system_prompt: str = "You are an expert QA dataset author."
    prompt_templates_by_qa_type: dict[str, str] = field(default_factory=dict)
    max_concurrent: int = 8
    batch_enabled: bool = True
    show_batch_progress: bool = True


@dataclass
class LLMEnvGenerationConfig:
    """Rollout-backed generation controls."""

    model: str = "gpt-5-mini"
    api_key: str = ""
    base_url: str = ""
    env_bundle: EnvBundleConfig = field(default_factory=EnvBundleConfig)
    rollout_limits: RolloutLimits = field(default_factory=RolloutLimits)
    prompt_template: str = (
        "Generate one QA pair as JSON with keys question and answer. "
        "qa_type={qa_type}; target_hop_count={target_hop_count}.\n\n"
        "Corpus summary:\n{corpus_summary}\n\n"
        "Example queries:\n{corpus_queries}\n\n"
        "Primary:\n{primary_chunk}\n\nSecondary:\n{secondary_chunks}\n"
    )


@dataclass
class GenerationConfig:
    """Generator mode selection."""

    mode: str = "llm_direct"
    llm_direct: LLMDirectGenerationConfig = field(default_factory=LLMDirectGenerationConfig)
    llm_env: LLMEnvGenerationConfig = field(default_factory=LLMEnvGenerationConfig)


@dataclass
class TransformationConfig:
    """Post-generation question transformation controls."""

    # LLM settings
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    system_prompt: str = ""
    user_template: str = ""

    # Noise level for style-appropriate mutations
    noise_level: str = "light"  # "none", "light", "moderate"

    # Style distribution for rewrites
    style_distribution: dict[str, float] = field(
        default_factory=lambda: {
            QUERY_STYLE_KEYWORD: 0.33,
            QUERY_STYLE_NATURAL: 0.34,
            QUERY_STYLE_EXPERT: 0.33,
        }
    )

    # Validation settings
    validation_enabled: bool = True
    validation_model: str = ""

    preserve_original_in_metadata: bool = True


@dataclass
class DeterministicGuardsConfig:
    """Deterministic quality gates."""

    enabled: bool = True
    min_question_chars: int = 12
    min_answer_chars: int = 24
    min_reference_chunks: int = 1


@dataclass
class RetrievalLLMFilterConfig:
    """Retrieval + judge filter settings."""

    enabled: bool = True
    judge_model: str = "gpt-5-mini"
    judge_api_key: str = ""
    judge_base_url: str = ""
    judge_system_prompt: str = DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT
    judge_user_template: str = DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE
    top_k: int = 5
    overlap_threshold: float = 0.5
    too_easy_confidence_threshold: float = 0.75
    too_easy_overlap_threshold: float = 0.65
    stats_key: str = "retrieval_too_easy_filter_stats"
    max_concurrent: int = 8
    batch_enabled: bool = True
    show_batch_progress: bool = True


@dataclass
class GroundingLLMFilterConfig:
    """Grounding/answerability judge filter settings."""

    enabled: bool = True
    judge_model: str = "gpt-5-mini"
    judge_api_key: str = ""
    judge_base_url: str = ""
    judge_system_prompt: str = DEFAULT_GROUNDING_JUDGE_SYSTEM_PROMPT
    judge_user_template: str = DEFAULT_GROUNDING_JUDGE_USER_TEMPLATE
    top_k: int = 5
    stats_key: str = "grounding_filter_stats"
    max_concurrent: int = 8
    batch_enabled: bool = True
    show_batch_progress: bool = True


@dataclass
class LLMEnvFilterConfig:
    """Rollout-backed filter settings."""

    enabled: bool = True
    model: str = "gpt-5-mini"
    api_key: str = ""
    base_url: str = ""
    judge_model: str = "gpt-5-mini"
    judge_api_key: str = ""
    judge_base_url: str = ""
    env_bundle: EnvBundleConfig = field(default_factory=EnvBundleConfig)
    rollout_limits: RolloutLimits = field(default_factory=RolloutLimits)


@dataclass
class FilteringConfig:
    """Filter chain mode selection."""

    deterministic_guards: DeterministicGuardsConfig = field(default_factory=DeterministicGuardsConfig)
    filters: list[str] = field(
        default_factory=lambda: ["grounding_llm", "retrieval_too_easy_llm"]
    )
    query_rewrite: QueryRewriteConfig = field(default_factory=QueryRewriteConfig)
    retrieval_llm: RetrievalLLMFilterConfig = field(default_factory=RetrievalLLMFilterConfig)
    grounding_llm: GroundingLLMFilterConfig = field(default_factory=GroundingLLMFilterConfig)


@dataclass
class RefinementConfig:
    """Refinement controls for retrying failed filter candidates."""

    enabled: bool = True
    model: str = "gpt-5-mini"
    api_key: str = ""
    base_url: str = ""
    max_refinements_per_item: int = 2
    max_same_seed_attempts_before_reanchor: int = 3
    max_rounds: int = 4
    max_total_regenerations: int = 400
    prompt_template: str = (
        "Refine this QA pair while preserving answer correctness and anchor intent.\n"
        "qa_type={qa_type}\nfeedback={feedback}\n\n"
        "Current question: {question}\nCurrent answer: {answer}\n\n"
        "Return JSON with keys question and answer."
    )


@dataclass
class SplitConfig:
    """Train/eval split settings."""

    train_ratio: float = 0.8
    stratify_by: list[str] = field(default_factory=lambda: ["qa_type", "style"])
    seed: int = 42


@dataclass
class OutputConfig:
    """Output artifact settings."""

    dir: str = "outputs/cgft"
    train_jsonl: str = "train.jsonl"
    eval_jsonl: str = "eval.jsonl"


@dataclass
class CgftPipelineConfig:
    """Unified configuration for the Cgft pipeline."""

    platform: PlatformConfig
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    corpus_context: CorpusContextConfig = field(default_factory=CorpusContextConfig)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    linker: LinkerConfig = field(default_factory=LinkerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    transformation: TransformationConfig = field(default_factory=TransformationConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    random_seed: int = 42
    verbose: bool = True

    def resolve_api_keys(self) -> None:
        """Fill unset component API keys/base URLs with shared platform LLM settings."""
        shared_llm_key = self.platform.llm_api_key or self.platform.api_key
        shared_llm_base_url = (
            self.platform.llm_base_url.strip() or "https://app.cgft.io/api/llm"
        )

        if not self.generation.llm_direct.api_key:
            self.generation.llm_direct.api_key = shared_llm_key
        if not self.generation.llm_direct.base_url:
            self.generation.llm_direct.base_url = shared_llm_base_url
        if not self.generation.llm_env.api_key:
            self.generation.llm_env.api_key = shared_llm_key
        if not self.generation.llm_env.base_url:
            self.generation.llm_env.base_url = shared_llm_base_url

        if not self.transformation.api_key:
            self.transformation.api_key = shared_llm_key
        if not self.transformation.base_url:
            self.transformation.base_url = shared_llm_base_url
        if not self.transformation.model:
            self.transformation.model = self.generation.llm_direct.model
        self.transformation.style_distribution = normalize_style_distribution(
            self.transformation.style_distribution
        )

        if not self.corpus_context.api_key:
            self.corpus_context.api_key = shared_llm_key
        if not self.corpus_context.base_url:
            self.corpus_context.base_url = shared_llm_base_url
        if not self.corpus_context.model:
            self.corpus_context.model = self.generation.llm_direct.model

        if not self.corpus_context.entity_extraction_llm.api_key:
            self.corpus_context.entity_extraction_llm.api_key = self.corpus_context.api_key
        if not self.corpus_context.entity_extraction_llm.base_url:
            self.corpus_context.entity_extraction_llm.base_url = self.corpus_context.base_url
        if not self.corpus_context.entity_extraction_llm.model:
            self.corpus_context.entity_extraction_llm.model = self.corpus_context.model

        if not self.linker.llm_guided.api_key:
            self.linker.llm_guided.api_key = shared_llm_key
        if not self.linker.llm_guided.base_url:
            self.linker.llm_guided.base_url = shared_llm_base_url

        if not self.filtering.retrieval_llm.judge_api_key:
            self.filtering.retrieval_llm.judge_api_key = shared_llm_key
        if not self.filtering.retrieval_llm.judge_base_url:
            self.filtering.retrieval_llm.judge_base_url = shared_llm_base_url

        if not self.filtering.grounding_llm.judge_api_key:
            self.filtering.grounding_llm.judge_api_key = shared_llm_key
        if not self.filtering.grounding_llm.judge_base_url:
            self.filtering.grounding_llm.judge_base_url = shared_llm_base_url

        if not self.refinement.api_key:
            self.refinement.api_key = shared_llm_key
        if not self.refinement.base_url:
            self.refinement.base_url = shared_llm_base_url

        if self.refinement.max_total_regenerations <= 0:
            self.refinement.max_total_regenerations = max(1, self.targets.total_samples * 2)


@dataclass
class GenerationTask:
    """One QA generation intent."""

    task_id: str
    qa_type: str
    target_hop_count: int
    seed_chunk_id: str
    regeneration_prompt: str = ""
    regeneration_attempt: int = 0
    source_task_id: str = ""
    previous_failure_type: str = ""
    previous_judge_reason_tag: str = ""
    overlap_triggered: bool = False
    expected_action: str = ""
    failed_question: str = ""
    failed_answer: str = ""


@dataclass
class CgftContext:
    """Shared runtime context for Cgft components."""

    config: CgftPipelineConfig
    source: Any
    rng: random.Random = field(default_factory=random.Random)
    state: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def setdefault(self, key: str, value: Any) -> Any:
        return self.state.setdefault(key, value)

    def __getitem__(self, key: str) -> Any:
        return self.state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.state[key] = value


@dataclass
class CgftRunStats:
    """Typed run stats emitted by CgftPipeline."""

    raw_candidates_total: int = 0
    passed_total: int = 0
    rejected_total: int = 0
    regenerated_total: int = 0
    round_limit: int = 0


def _load_distribution(raw: Any, *, fallback: dict[str, float]) -> dict[str, float]:
    if not isinstance(raw, dict):
        return dict(fallback)
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if not out:
        return dict(fallback)
    return out


def _load_string_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        out = [str(item).strip() for item in raw if str(item).strip()]
        return out
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if "\n" in text:
            return [line.strip() for line in text.splitlines() if line.strip()]
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return []


def allocate_largest_remainder_generic(
    total: int,
    distribution: Mapping[str, float],
) -> dict[str, int]:
    """Allocate integer counts over arbitrary keys using largest remainder.

    The output is deterministic:
    - input keys are sorted before fractional tie-breaking
    - ties are resolved by key name
    """
    if total <= 0:
        return {str(key): 0 for key in sorted(distribution)}

    sanitized = {str(key): max(0.0, float(value)) for key, value in distribution.items()}
    if not sanitized:
        return {}

    norm_total = sum(sanitized.values())
    if norm_total <= 0:
        uniform = 1.0 / len(sanitized)
        normalized = {key: uniform for key in sorted(sanitized)}
    else:
        normalized = {key: value / norm_total for key, value in sanitized.items()}

    keys = sorted(normalized)
    raw = {key: normalized[key] * total for key in keys}
    base = {key: int(math.floor(raw[key])) for key in keys}
    remainder = total - sum(base.values())
    if remainder > 0:
        order = sorted(
            keys,
            key=lambda key: (raw[key] - base[key], normalized[key], key),
            reverse=True,
        )
        for key in order[:remainder]:
            base[key] += 1
    return base


def matrix_target_counts(targets: TargetsConfig) -> dict[str, int]:
    """Allocate deterministic target counts over qa_type."""
    return allocate_largest_remainder_generic(
        targets.total_samples,
        targets.normalized_qa_type_distribution(),
    )


def build_generation_tasks(
    cfg: CgftPipelineConfig,
    *,
    seed_chunk_ids: list[str],
) -> list[GenerationTask]:
    """Build tasks from matrix targets with deterministic seed assignment."""
    if not seed_chunk_ids:
        return []
    rng = random.Random(cfg.random_seed)
    qa_counts = matrix_target_counts(cfg.targets)
    hop_map = {
        **DEFAULT_TARGET_HOP_COUNTS,
        **(cfg.linker.structural.target_hop_counts or {}),
    }

    tasks: list[GenerationTask] = []
    idx = 0
    for qa_type in sorted(qa_counts):
        count = qa_counts[qa_type]
        for _ in range(max(0, count)):
            seed_chunk_id = seed_chunk_ids[idx % len(seed_chunk_ids)]
            idx += 1
            tasks.append(
                GenerationTask(
                    task_id=f"task_{len(tasks):05d}",
                    qa_type=qa_type,
                    target_hop_count=int(hop_map.get(qa_type, hop_map.get("multi_hop", 2))),
                    seed_chunk_id=seed_chunk_id,
                )
            )
    rng.shuffle(tasks)
    return tasks


def _parse_model_cfg(raw: Any, *, fallback: ModelConfig) -> ModelConfig:
    if isinstance(raw, str):
        return ModelConfig(model=raw, api_key=fallback.api_key, base_url=fallback.base_url)
    if not isinstance(raw, dict):
        return fallback
    return ModelConfig(
        model=str(raw.get("model", fallback.model)),
        api_key=str(raw.get("api_key", fallback.api_key)),
        base_url=str(raw.get("base_url", fallback.base_url)),
    )


def _collect_removed_config_keys(raw: dict[str, Any]) -> list[str]:
    """Return removed config keys present in user YAML."""
    removed_keys: list[str] = []

    targets_raw = raw.get("targets", {}) or {}
    filtering_raw = raw.get("filtering", {}) or {}
    guards_raw = filtering_raw.get("deterministic_guards", {}) or {}
    retrieval_raw = filtering_raw.get("retrieval_llm", {}) or {}
    refinement_raw = raw.get("refinement", {}) or {}

    if "style_distribution_by_qa_type" in targets_raw:
        removed_keys.append("targets.style_distribution_by_qa_type")
    if "enforce_style_mismatch_guard" in guards_raw:
        removed_keys.append("filtering.deterministic_guards.enforce_style_mismatch_guard")

    if "mode" in filtering_raw:
        removed_keys.append("filtering.mode")
    if "llm_env" in filtering_raw:
        removed_keys.append("filtering.llm_env")
    if "strategy" in refinement_raw:
        removed_keys.append("refinement.strategy")

    for key in (
        "route_unsupported_to_failure",
        "unknown_answerable_confidence_threshold",
        "require_multi_chunk_evidence_for_non_lookup_pass",
        "min_matched_reference_chunks_for_non_lookup_pass",
    ):
        if key in retrieval_raw:
            removed_keys.append(f"filtering.retrieval_llm.{key}")

    return removed_keys


def load_cgft_config(path: str | Path) -> CgftPipelineConfig:
    """Load `CgftPipelineConfig` from YAML."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    removed_keys = _collect_removed_config_keys(raw)
    if removed_keys:
        joined = ", ".join(sorted(removed_keys))
        raise ValueError(
            "Unsupported config keys for the clean-break Cgft filtering/refinement API: "
            f"{joined}. "
            "Use filtering.filters with supported stages {'grounding_llm', 'retrieval_too_easy_llm'} "
            "and generator-based refinement without refinement.strategy."
        )

    platform_raw = raw.get("platform", {}) or {}
    api_key = str(platform_raw.get("api_key", "")).strip()
    if not api_key:
        raise ValueError("Missing required config value: platform.api_key")
    platform = PlatformConfig(
        api_key=api_key,
        base_url=str(platform_raw.get("base_url", "https://app.cgft.io")).strip(),
        llm_api_key=str(platform_raw.get("llm_api_key", "")).strip(),
        llm_base_url=str(
            platform_raw.get("llm_base_url", "https://app.cgft.io/api/llm")
        ).strip(),
    )

    corpus_raw = raw.get("corpus", {}) or {}
    corpus = CorpusConfig(
        docs_path=str(corpus_raw.get("docs_path", "")).strip(),
        corpus_id=str(corpus_raw.get("corpus_id", "")).strip(),
        corpus_name=str(corpus_raw.get("corpus_name", "cgft-corpus")).strip(),
        show_summary=bool(corpus_raw.get("show_summary", True)),
        min_chunk_chars=max(0, int(corpus_raw.get("min_chunk_chars", 400))),
    )

    corpus_context_raw = raw.get("corpus_context", {}) or {}
    corpus_context = CorpusContextConfig(
        enabled=bool(corpus_context_raw.get("enabled", True)),
        description=str(corpus_context_raw.get("description", "")).strip(),
        example_queries=_load_string_list(corpus_context_raw.get("example_queries", [])),
        model=str(corpus_context_raw.get("model", "")).strip(),
        api_key=str(corpus_context_raw.get("api_key", "")).strip(),
        base_url=str(corpus_context_raw.get("base_url", "")).strip(),
        num_top_level_samples=max(0, int(corpus_context_raw.get("num_top_level_samples", 4))),
        num_random_samples=max(0, int(corpus_context_raw.get("num_random_samples", 4))),
        min_chunk_chars=max(0, int(corpus_context_raw.get("min_chunk_chars", 400))),
        system_prompt=str(
            corpus_context_raw.get("system_prompt", CORPUS_SYSTEM_PROMPT)
        ).strip()
        or CORPUS_SYSTEM_PROMPT,
        user_template=str(
            corpus_context_raw.get("user_template", CORPUS_USER_TEMPLATE)
        ).strip()
        or CORPUS_USER_TEMPLATE,
        generate_entity_patterns=bool(corpus_context_raw.get("generate_entity_patterns", True)),
        entity_extraction_llm=EntityExtractionLLMConfig(
            model=str(
                (corpus_context_raw.get("entity_extraction_llm") or {}).get("model", "")
            ).strip(),
            api_key=str(
                (corpus_context_raw.get("entity_extraction_llm") or {}).get("api_key", "")
            ).strip(),
            base_url=str(
                (corpus_context_raw.get("entity_extraction_llm") or {}).get("base_url", "")
            ).strip(),
        ),
    )

    targets_raw = raw.get("targets", {}) or {}
    targets = TargetsConfig(
        total_samples=max(1, int(targets_raw.get("total_samples", 200))),
        qa_type_distribution=_load_distribution(
            targets_raw.get("qa_type_distribution"),
            fallback=dict(DEFAULT_QA_TYPE_DISTRIBUTION),
        ),
    )

    linker_raw = raw.get("linker", {}) or {}
    structural_raw = linker_raw.get("structural", {}) or {}
    llm_linker_raw = linker_raw.get("llm_guided", {}) or {}
    linker = LinkerConfig(
        type=str(linker_raw.get("type", "structural")).strip().lower() or "structural",
        structural=StructuralLinkerConfig(
            type_distribution=structural_raw.get("type_distribution"),
            target_hop_counts=structural_raw.get("target_hop_counts"),
            bm25_enrichment_queries=max(1, int(structural_raw.get("bm25_enrichment_queries", 3))),
            bm25_enrichment_top_k=max(1, int(structural_raw.get("bm25_enrichment_top_k", 5))),
            max_related_refs=max(1, int(structural_raw.get("max_related_refs", 3))),
            corpus_pool_size=max(10, int(structural_raw.get("corpus_pool_size", 200))),
        ),
        llm_guided=LLMGuidedLinkerConfig(
            model=str(llm_linker_raw.get("model", "gpt-5-mini")),
            api_key=str(llm_linker_raw.get("api_key", "")),
            base_url=str(llm_linker_raw.get("base_url", "")).strip(),
            system_prompt=str(llm_linker_raw.get("system_prompt", "")),
            user_template=str(llm_linker_raw.get("user_template", "")),
            top_k_bm25=max(1, int(llm_linker_raw.get("top_k_bm25", 5))),
            top_related_chunks=max(1, int(llm_linker_raw.get("top_related_chunks", 3))),
            context_preview_chars=max(50, int(llm_linker_raw.get("context_preview_chars", 200))),
        ),
    )

    generation_raw = raw.get("generation", {}) or {}
    direct_raw = generation_raw.get("llm_direct", {}) or {}
    env_gen_raw = generation_raw.get("llm_env", {}) or {}
    env_gen_bundle_raw = env_gen_raw.get("env_bundle", {}) or {}
    env_gen_limits_raw = env_gen_raw.get("rollout_limits", {}) or {}
    generation = GenerationConfig(
        mode=str(generation_raw.get("mode", "llm_direct")).strip().lower() or "llm_direct",
        llm_direct=LLMDirectGenerationConfig(
            model=str(direct_raw.get("model", LLMDirectGenerationConfig().model)),
            api_key=str(direct_raw.get("api_key", "")),
            base_url=str(direct_raw.get("base_url", "")).strip(),
            max_completion_tokens=max(
                100,
                int(
                    direct_raw.get(
                        "max_completion_tokens",
                        LLMDirectGenerationConfig().max_completion_tokens,
                    )
                ),
            ),
            timeout=max(10.0, float(direct_raw.get("timeout", 120.0))),
            system_prompt=str(
                direct_raw.get("system_prompt", LLMDirectGenerationConfig().system_prompt)
            ),
            prompt_templates_by_qa_type=dict(direct_raw.get("prompt_templates_by_qa_type", {}) or {}),
            max_concurrent=max(1, int(direct_raw.get("max_concurrent", 8))),
            batch_enabled=bool(direct_raw.get("batch_enabled", True)),
            show_batch_progress=bool(direct_raw.get("show_batch_progress", True)),
        ),
        llm_env=LLMEnvGenerationConfig(
            model=str(env_gen_raw.get("model", "gpt-5-mini")),
            api_key=str(env_gen_raw.get("api_key", "")),
            base_url=str(env_gen_raw.get("base_url", "")).strip(),
            env_bundle=EnvBundleConfig(
                env_cls_path=str(env_gen_bundle_raw.get("env_cls_path", "")),
                env_metadata_path=str(env_gen_bundle_raw.get("env_metadata_path", "")),
                env_cls_file=str(env_gen_bundle_raw.get("env_cls_file", "")),
                env_metadata_file=str(env_gen_bundle_raw.get("env_metadata_file", "")),
            ),
            rollout_limits=RolloutLimits(
                max_turns=max(1, int(env_gen_limits_raw.get("max_turns", 16))),
                max_tool_calls=max(1, int(env_gen_limits_raw.get("max_tool_calls", 24))),
                max_completion_tokens=max(
                    100, int(env_gen_limits_raw.get("max_completion_tokens", 2048))
                ),
                timeout=max(10.0, float(env_gen_limits_raw.get("timeout", 120.0))),
            ),
            prompt_template=str(env_gen_raw.get("prompt_template", LLMEnvGenerationConfig().prompt_template)),
        ),
    )

    transformation_raw = raw.get("transformation", {}) or {}
    transformation = TransformationConfig(
        model=str(transformation_raw.get("model", "")).strip(),
        api_key=str(transformation_raw.get("api_key", "")).strip(),
        base_url=str(transformation_raw.get("base_url", "")).strip(),
        system_prompt=str(transformation_raw.get("system_prompt", "")).strip(),
        user_template=str(transformation_raw.get("user_template", "")).strip(),
        noise_level=str(transformation_raw.get("noise_level", "light")).strip() or "light",
        style_distribution=normalize_style_distribution(
            transformation_raw.get(
                "style_distribution",
                TransformationConfig().style_distribution,
            )
        ),
        validation_enabled=bool(transformation_raw.get("validation_enabled", True)),
        validation_model=str(transformation_raw.get("validation_model", "")).strip(),
        preserve_original_in_metadata=bool(
            transformation_raw.get("preserve_original_in_metadata", True)
        ),
    )

    filtering_raw = raw.get("filtering", {}) or {}
    guards_raw = filtering_raw.get("deterministic_guards", {}) or {}
    filters_raw = filtering_raw.get("filters", None)
    if filters_raw is None:
        filters_raw = list(FilteringConfig().filters)
    rewrite_raw = filtering_raw.get("query_rewrite", {}) or {}
    retrieval_raw = filtering_raw.get("retrieval_llm", {}) or {}
    grounding_raw = filtering_raw.get("grounding_llm", {}) or {}
    if isinstance(filters_raw, str):
        chain_filters = [
            token.strip().lower()
            for token in filters_raw.split(",")
            if token and token.strip()
        ]
    else:
        chain_filters = [
            str(token).strip().lower()
            for token in list(filters_raw)
            if str(token).strip()
        ]
    filtering = FilteringConfig(
        deterministic_guards=DeterministicGuardsConfig(
            enabled=bool(guards_raw.get("enabled", True)),
            min_question_chars=max(1, int(guards_raw.get("min_question_chars", 12))),
            min_answer_chars=max(1, int(guards_raw.get("min_answer_chars", 24))),
            min_reference_chunks=max(0, int(guards_raw.get("min_reference_chunks", 1))),
        ),
        filters=chain_filters,
        query_rewrite=QueryRewriteConfig(
            enabled=bool(rewrite_raw.get("enabled", True)),
            max_terms=max(1, int(rewrite_raw.get("max_terms", 16))),
            max_chars=max(20, int(rewrite_raw.get("max_chars", 140))),
            apply_to_all_rows=bool(rewrite_raw.get("apply_to_all_rows", True)),
        ),
        retrieval_llm=RetrievalLLMFilterConfig(
            enabled=bool(retrieval_raw.get("enabled", True)),
            judge_model=str(retrieval_raw.get("judge_model", "gpt-5-mini")),
            judge_api_key=str(retrieval_raw.get("judge_api_key", "")),
            judge_base_url=str(retrieval_raw.get("judge_base_url", "")).strip(),
            judge_system_prompt=(
                str(
                    retrieval_raw.get(
                        "judge_system_prompt",
                        DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT,
                    )
                ).strip()
                or DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT
            ),
            judge_user_template=(
                str(
                    retrieval_raw.get(
                        "judge_user_template",
                        DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE,
                    )
                ).strip()
                or DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE
            ),
            top_k=max(1, int(retrieval_raw.get("top_k", 5))),
            overlap_threshold=max(0.0, min(1.0, float(retrieval_raw.get("overlap_threshold", 0.5)))),
            too_easy_confidence_threshold=max(
                0.0,
                min(1.0, float(retrieval_raw.get("too_easy_confidence_threshold", 0.75))),
            ),
            too_easy_overlap_threshold=max(
                0.0,
                min(1.0, float(retrieval_raw.get("too_easy_overlap_threshold", 0.65))),
            ),
            stats_key=(
                str(retrieval_raw.get("stats_key", "retrieval_too_easy_filter_stats")).strip()
                or "retrieval_too_easy_filter_stats"
            ),
            max_concurrent=max(1, int(retrieval_raw.get("max_concurrent", 8))),
            batch_enabled=bool(retrieval_raw.get("batch_enabled", True)),
        ),
        grounding_llm=GroundingLLMFilterConfig(
            enabled=bool(grounding_raw.get("enabled", True)),
            judge_model=str(grounding_raw.get("judge_model", "gpt-5-mini")),
            judge_api_key=str(grounding_raw.get("judge_api_key", "")),
            judge_base_url=str(grounding_raw.get("judge_base_url", "")).strip(),
            judge_system_prompt=(
                str(
                    grounding_raw.get(
                        "judge_system_prompt",
                        DEFAULT_GROUNDING_JUDGE_SYSTEM_PROMPT,
                    )
                ).strip()
                or DEFAULT_GROUNDING_JUDGE_SYSTEM_PROMPT
            ),
            judge_user_template=(
                str(
                    grounding_raw.get(
                        "judge_user_template",
                        DEFAULT_GROUNDING_JUDGE_USER_TEMPLATE,
                    )
                ).strip()
                or DEFAULT_GROUNDING_JUDGE_USER_TEMPLATE
            ),
            top_k=max(1, int(grounding_raw.get("top_k", 5))),
            stats_key=(
                str(grounding_raw.get("stats_key", "grounding_filter_stats")).strip()
                or "grounding_filter_stats"
            ),
            max_concurrent=max(1, int(grounding_raw.get("max_concurrent", 8))),
            batch_enabled=bool(grounding_raw.get("batch_enabled", True)),
        ),
    )

    refinement_raw = raw.get("refinement", {}) or {}
    refinement = RefinementConfig(
        enabled=bool(refinement_raw.get("enabled", True)),
        model=str(refinement_raw.get("model", "gpt-5-mini")),
        api_key=str(refinement_raw.get("api_key", "")),
        base_url=str(refinement_raw.get("base_url", "")).strip(),
        max_refinements_per_item=max(
            0, int(refinement_raw.get("max_refinements_per_item", 2))
        ),
        max_same_seed_attempts_before_reanchor=max(
            0,
            int(
                refinement_raw.get(
                    "max_same_seed_attempts_before_reanchor",
                    RefinementConfig().max_same_seed_attempts_before_reanchor,
                )
            ),
        ),
        max_rounds=max(0, int(refinement_raw.get("max_rounds", 4))),
        max_total_regenerations=max(
            0, int(refinement_raw.get("max_total_regenerations", targets.total_samples * 2))
        ),
        prompt_template=str(refinement_raw.get("prompt_template", RefinementConfig().prompt_template)),
    )

    split_raw = raw.get("split", {}) or {}
    split = SplitConfig(
        train_ratio=max(0.1, min(0.95, float(split_raw.get("train_ratio", 0.8)))),
        stratify_by=list(split_raw.get("stratify_by", ["qa_type", "style"]) or ["qa_type", "style"]),
        seed=int(split_raw.get("seed", 42)),
    )

    output_raw = raw.get("output", {}) or {}
    output = OutputConfig(
        dir=str(output_raw.get("dir", "outputs/cgft")).strip(),
        train_jsonl=str(output_raw.get("train_jsonl", "train.jsonl")).strip(),
        eval_jsonl=str(output_raw.get("eval_jsonl", "eval.jsonl")).strip(),
    )

    cfg = CgftPipelineConfig(
        platform=platform,
        corpus=corpus,
        corpus_context=corpus_context,
        targets=targets,
        linker=linker,
        generation=generation,
        transformation=transformation,
        filtering=filtering,
        refinement=refinement,
        split=split,
        output=output,
        random_seed=int(raw.get("random_seed", 42)),
    )
    cfg.resolve_api_keys()
    return cfg
