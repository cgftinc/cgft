# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode
pip install -e .[dev]

# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/

# Run tests
pytest

# Run a single test
pytest tests/path/to/test_file.py::test_function_name

# Add a dependency
uv add <package>
```

Line length is 100 characters. Ruff rules in use: E, F, I, N, W, UP.

**Python version:** 3.12 required. Python 3.13 is explicitly unsupported due to a `pathlib.Path` pickle incompatibility (enforced in `src/synthetic_data_prep/utils.py`).

## Architecture

This library provides an end-to-end pipeline for fine-tuning LLMs on RAG workloads using reinforcement learning. The pipeline flows: **document chunking → corpus indexing → synthetic QA generation → RL environment → training job**.

### Module Overview

**`chunkers/`** — Markdown document chunking with a 3-stage pipeline:
1. Split by markdown headers (H1/H2/H3), preserving hierarchy in metadata
2. Fuse adjacent short sections to avoid over-fragmentation
3. Recursive character splitting for large sections with overlap

`Chunk` is a frozen dataclass with auto-computed SHA256 hash. `ChunkCollection` provides file-structure-aware neighbor lookup and context retrieval.

**`corpus/`** — Swappable backend abstraction via the `ChunkSource` Protocol:
- `CorporaChunkSource` — CGFT Corpus API (BM25 search)
- `TpufChunkSource` — Turbopuffer vector DB (embeddings + BM25)

Both are used to index chunks and expose a search API consumed by RL environments.

**`qa_generation/`** — Synthetic QA dataset generation via `generate_dataset()`. Produces two types:
- *Single-hop*: one chunk answers the question (1 LLM call)
- *Multi-hop*: multiple chunks required (2 LLM calls)

Output is a `QADataset` of `QADataPoint` objects, serializable to JSONL or JSON.

**`envs/`** — RL training environments extending `benchmax.envs.base_env.BaseEnv`:
- `SearchEnv` — base class with `chunk_overlap_reward_function()`: rewards ≥25% text overlap between model answer and reference chunks; penalizes ≥4 tool calls with 0 reward
- `CorporaSearchEnv` / `TpufSearchEnv` — concrete implementations that expose a `search_corpus` BM25 tool
- Extend `SearchEnv` to build custom environments with different tools while reusing the base reward logic

**`trainer/`** — Training job orchestration via `train()`:
1. Uploads JSONL dataset to blob storage
2. Zips environment class + local module dependencies and uploads
3. Launches training job on the CGFT platform, returns `experiment_id`

**`multi_model/`** — Utilities for calling multiple LLM providers and comparing pricing/responses.

### Key Data Types

- `Chunk` / `ChunkCollection` (`chunkers/models.py`)
- `QADataPoint` / `QADataset` / `ReferenceChunk` (`qa_generation/models.py`)
- `ChunkSource` Protocol (`corpus/source.py`) — implemented by both corpus backends
- `SearchEnv` (`envs/search_env.py`) — base RL environment; extend this for custom environments

### External Integrations

- **CGFT Platform** — training job management and blob storage
- **Corpora API** — document storage and BM25 search
- **Turbopuffer** — vector database
- **OpenAI API** — LLM for QA generation (primary)
- **benchmax** — RL training framework (`BaseEnv`, `ToolDefinition`, `StandardizedExample`)

