# cgft-utils

End-to-end pipeline for fine-tuning LLMs on RAG workloads using reinforcement learning.

**Flow:** document chunking → corpus indexing → synthetic QA generation → RL environment → training job

## Installation

```bash
pip install git+https://github.com/cgftinc/cgft-utils.git
```

Or clone and install in development mode:

```bash
git clone https://github.com/cgftinc/cgft-utils.git
cd cgft-utils
pip install -e .[dev]
```

## Quick Start

```python
from cgft_utils.chunkers import Chunk, ChunkCollection
from cgft_utils.corpus import CorporaChunkSource
from cgft_utils.qa_generation import CgftPipeline
from cgft_utils.envs import CorporaSearchEnv
from cgft_utils.trainer import train
```

See `notebooks/` for full usage examples (also available on Google Colab).

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

### overview

- [introduction](docs/overview/introduction.mdx)
- [what-is-rl](docs/overview/what-is-rl.mdx)

### bring-your-env

- [train](docs/bring-your-env/train.mdx)
- [your-own-env](docs/bring-your-env/your-own-env.mdx)

### experiment

- [managing-experiments](docs/experiment/managing-experiments.mdx)
- [sharing](docs/experiment/sharing.mdx)

### rag

- [env](docs/rag/env.mdx)
- [rl-for-rag](docs/rag/rl-for-rag.mdx)
- [synthetic_datagen](docs/rag/synthetic_datagen.mdx)
- [train](docs/rag/train.mdx)
