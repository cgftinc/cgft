"""QA Generation module for creating synthetic question-answer pairs."""

from .models import QADataPoint, QADataset, ReferenceChunk
from .pipeline import generate_dataset
from .storage import (
    load_qa_dataset,
    load_qa_dataset_jsonl,
    save_qa_dataset,
    save_qa_dataset_jsonl,
)

__all__ = [
    "QADataPoint",
    "QADataset",
    "ReferenceChunk",
    "generate_dataset",
    "load_qa_dataset",
    "load_qa_dataset_jsonl",
    "save_qa_dataset",
    "save_qa_dataset_jsonl",
]
