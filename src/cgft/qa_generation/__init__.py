"""QA generation package exports (CgftPipeline only)."""

from .anchor_selector import AnchorBundle, AnchorSelector
from .anchor_utils import (
    extract_anchor_ref_ids,
    generate_bm25_queries,
    select_anchor_bundle_with_enrichment,
)
from .batch_processor import BatchResponse, BatchResult, batch_process_sync
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
    RetrievalLLMFilter,
)
from .formatters import TrainEvalFormatter
from .generated_qa import FilterVerdict, GeneratedQA
from .generators import DirectLLMGenerator, EnvRolloutGenerator
from .linkers import AdaptiveChunkLinker, LLMGuidedChunkLinker, StructuralChunkLinker
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

__all__ = [
    "BatchResponse",
    "BatchResult",
    "batch_process_sync",
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
    "DirectLLMGenerator",
    "EnvRolloutGenerator",
    "DeterministicGuardsFilter",
    "RetrievalLLMFilter",
    "GroundingLLMFilter",
    "EnvRolloutFilter",
    "TrainEvalFormatter",
    "CgftPipeline",
    "run_cgft_pipeline",
    "run_cgft_pipeline_from_config",
    "generate_bm25_queries",
    "select_anchor_bundle_with_enrichment",
    "extract_anchor_ref_ids",
]
