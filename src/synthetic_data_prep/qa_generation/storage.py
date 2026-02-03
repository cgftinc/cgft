"""Storage utilities for persisting and loading QA datasets."""

import json
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from synthetic_data_prep.qa_generation.models import (
    QADataPoint,
    QADataset,
    ReferenceChunk,
)


def _to_literal_scalars(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert multiline strings to LiteralScalarString for block style output."""
    result = []
    for item in data:
        new_item = dict(item)
        # Convert answer to literal scalar if multiline
        if isinstance(new_item.get("answer"), str) and "\n" in new_item["answer"]:
            new_item["answer"] = LiteralScalarString(new_item["answer"])
        # Convert reference chunk contents to literal scalars
        if "reference_chunks" in new_item:
            new_chunks = []
            for chunk in new_item["reference_chunks"]:
                new_chunk = dict(chunk)
                if isinstance(new_chunk.get("content"), str) and "\n" in new_chunk["content"]:
                    new_chunk["content"] = LiteralScalarString(new_chunk["content"])
                new_chunks.append(new_chunk)
            new_item["reference_chunks"] = new_chunks
        result.append(new_item)
    return result


def _parse_data_points(data: list[dict[str, Any]]) -> list[QADataPoint]:
    """Parse a list of dicts into QADataPoint objects."""
    data_points = []
    for item in data:
        reference_chunks = [
            ReferenceChunk(
                id=chunk["id"],
                metadata=chunk.get("metadata", {}),
                content=chunk["content"],
            )
            for chunk in item.get("reference_chunks", [])
        ]
        data_points.append(
            QADataPoint(
                question=item["question"],
                answer=item["answer"],
                reference_chunks=reference_chunks,
                qa_type=item.get("qa_type", "single_hop"),
            )
        )
    return data_points


# === YAML ===


def save_qa_dataset(dataset: QADataset, path: str | Path) -> None:
    """Save QA dataset to a YAML file (human-readable format).

    Args:
        dataset: QADataset to save
        path: File path to save to (will be created if doesn't exist)

    Example:
        >>> dataset = QADataset(data_points=[...])
        >>> save_qa_dataset(dataset, "qa_pairs.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _to_literal_scalars(dataset.to_list())

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 1000

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def load_qa_dataset(path: str | Path) -> QADataset:
    """Load QA dataset from a YAML file.

    Args:
        path: File path to load from

    Returns:
        QADataset created from loaded data

    Raises:
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> dataset = load_qa_dataset("qa_pairs.yaml")
        >>> print(dataset.summary())
    """
    path = Path(path)

    yaml = YAML()
    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = yaml.load(f)

    return QADataset(data_points=_parse_data_points(data))


# === JSONL ===


def save_qa_dataset_jsonl(dataset: QADataset, path: str | Path) -> None:
    """Save QA dataset to a JSONL file (one JSON object per line).

    This format is compatible with HuggingFace ``load_dataset("json", ...)``.

    Args:
        dataset: QADataset to save
        path: File path to save to (will be created if doesn't exist)

    Example:
        >>> dataset = QADataset(data_points=[...])
        >>> save_qa_dataset_jsonl(dataset, "qa_pairs.jsonl")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for dp in dataset.data_points:
            f.write(json.dumps(dp.to_dict(), ensure_ascii=False) + "\n")


def load_qa_dataset_jsonl(path: str | Path) -> QADataset:
    """Load QA dataset from a JSONL file.

    Args:
        path: File path to load from

    Returns:
        QADataset created from loaded data

    Raises:
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> dataset = load_qa_dataset_jsonl("qa_pairs.jsonl")
        >>> print(dataset.summary())
    """
    path = Path(path)

    data_points: list[QADataPoint] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data_points.extend(_parse_data_points([json.loads(line)]))

    return QADataset(data_points=data_points)


def qa_dataset_to_jsonl_bytes(dataset: QADataset) -> bytes:
    """Serialize QA dataset to JSONL bytes (for upload).

    Args:
        dataset: QADataset to serialize

    Returns:
        UTF-8 encoded JSONL bytes
    """
    lines = [json.dumps(dp.to_dict(), ensure_ascii=False) for dp in dataset.data_points]
    return ("\n".join(lines) + "\n").encode("utf-8")
