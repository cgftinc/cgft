# cgft

End-to-end pipeline for fine-tuning LLMs on RAG workloads using reinforcement learning.

**Flow:** document chunking → corpus indexing → synthetic QA generation → RL environment → training job

## Installation

```bash
pip install git+https://github.com/cgftinc/cgft.git
```

Or clone and install in development mode:

```bash
git clone https://github.com/cgftinc/cgft.git
cd cgft
pip install -e .[dev]
```

## Quick Start

```python
from cgft.chunkers import Chunk, ChunkCollection
from cgft.corpus import CorporaChunkSource
from cgft.qa_generation import CgftPipeline
from cgft.envs import CorporaSearchEnv
from cgft.trainer import train
```

## Usage Examples
Goolge Colab Notebooks: https://drive.google.com/drive/u/0/folders/1idySaAEmm2ruJLAkjMpJBFqM9cTY2Hv5

## CLI

```bash
cgft-pipeline --config path/to/config.yaml
```

## Modules

- **`chunkers/`** — Markdown document chunking (split → fuse → recursive character split)
- **`corpus/`** — Swappable corpus backends (`CorporaChunkSource`, `TpufChunkSource`)
- **`qa_generation/`** — Synthetic QA dataset generation (single-hop & multi-hop)
- **`envs/`** — RL training environments (extend `SearchEnv` for custom environments)
- **`trainer/`** — Training job orchestration via `train()`
- **`multi_model/`** — Multi-LLM provider utilities

## Documentation
https://cgft.io/docs/
