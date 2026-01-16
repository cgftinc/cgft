"""Storage utilities for persisting and loading chunks."""

from pathlib import Path
from typing import Any

import yaml

from synthetic_data_prep.models import Chunk


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    """Save chunks to a YAML file.

    Args:
        chunks: List of Chunk objects to save.
        path: File path to save to (will be created if doesn't exist).

    Example:
        >>> chunks = chunker.chunk_folder("./docs")
        >>> save_chunks(chunks, "chunks.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [chunk.to_dict() for chunk in chunks]

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=1000,  # Prevent line wrapping in content
        )


def load_chunks(path: str | Path) -> list[Chunk]:
    """Load chunks from a YAML file.

    Args:
        path: File path to load from.

    Returns:
        List of Chunk objects.

    Raises:
        FileNotFoundError: If the file doesn't exist.

    Example:
        >>> chunks = load_chunks("chunks.yaml")
        >>> inspector = ChunkInspector(chunks)
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = yaml.safe_load(f)

    return [Chunk(**item) for item in data]
