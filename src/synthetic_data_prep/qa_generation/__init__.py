"""QA generation package exports (CgftPipeline only)."""

from .anchor_selector import AnchorBundle, AnchorSelector
from .anchor_utils import (
    extract_anchor_ref_ids,
    generate_bm25_queries,
    select_anchor_bundle_with_enrichment,
)
from .cgft_models import (
    CgftContext,
    CgftPipelineConfig,
    CgftRunStats,
    CorpusContextConfig,
    GenerationTask,
    load_cgft_config,
    matrix_target_counts,
)
from .cgft_pipeline import CgftPipeline, run_cgft_pipeline, run_cgft_pipeline_from_config
from .corpus_capabilities import CorpusCapabilities
from .filters import (
    DeterministicGuardsFilter,
    EnvRolloutFilter,
    GroundingLLMFilter,
    RetrievalDifficultyFilter,
    RetrievalLLMFilter,
)
from .formatters import SageFormatter, TrainEvalFormatter
from .generated_qa import FilterVerdict, GeneratedQA
from .generators import DirectLLMGenerator, EnvRolloutGenerator, SageGenerator
from .legacy_prompt_archive import (
    HYBRID_PROMPTS,
    SAGE_PROMPTS,
    get_hybrid_prompts,
    get_sage_prompts,
    hybrid_multi_hop_user_template_for_cgft,
    hybrid_single_hop_user_template_for_cgft,
    render_hybrid_multi_hop_system_prompt,
    render_hybrid_single_hop_system_prompt,
    render_hybrid_style_instruction,
)
from .linkers import AdaptiveChunkLinker, LLMGuidedChunkLinker, StructuralChunkLinker
from .orchestrator import Pipeline
from .protocols import (
    ChunkLinker,
    EvaluatorFilter,
    Formatter,
    LLMBasedFilter,
    LLMEnvBasedFilter,
    LLMEnvSupportedGenerator,
    LLMSupportedGenerator,
    QuestionGenerator,
    Refiner,
)
from .regenerators import FeedbackRefiner, GenerationRetryRegenerator

__all__ = [
    "AnchorBundle",
    "AnchorSelector",
    "CorpusCapabilities",
    "GeneratedQA",
    "FilterVerdict",
    "GenerationTask",
    "CgftContext",
    "CgftRunStats",
    "CorpusContextConfig",
    "CgftPipelineConfig",
    "load_cgft_config",
    "matrix_target_counts",
    "ChunkLinker",
    "QuestionGenerator",
    "LLMSupportedGenerator",
    "LLMEnvSupportedGenerator",
    "EvaluatorFilter",
    "LLMBasedFilter",
    "LLMEnvBasedFilter",
    "Refiner",
    "Formatter",
    "StructuralChunkLinker",
    "LLMGuidedChunkLinker",
    "AdaptiveChunkLinker",
    "HYBRID_PROMPTS",
    "SAGE_PROMPTS",
    "get_hybrid_prompts",
    "get_sage_prompts",
    "render_hybrid_style_instruction",
    "render_hybrid_single_hop_system_prompt",
    "render_hybrid_multi_hop_system_prompt",
    "hybrid_single_hop_user_template_for_cgft",
    "hybrid_multi_hop_user_template_for_cgft",
    "DirectLLMGenerator",
    "EnvRolloutGenerator",
    "DeterministicGuardsFilter",
    "RetrievalLLMFilter",
    "GroundingLLMFilter",
    "EnvRolloutFilter",
    "RetrievalDifficultyFilter",
    "FeedbackRefiner",
    "GenerationRetryRegenerator",
    "SageFormatter",
    "SageGenerator",
    "Pipeline",
    "TrainEvalFormatter",
    "CgftPipeline",
    "run_cgft_pipeline",
    "run_cgft_pipeline_from_config",
    "generate_bm25_queries",
    "select_anchor_bundle_with_enrichment",
    "extract_anchor_ref_ids",
]
