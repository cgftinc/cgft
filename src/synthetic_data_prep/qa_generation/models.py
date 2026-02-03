"""Data models for QA generation datasets."""

from typing import Any

from pydantic import BaseModel, Field


class ReferenceChunk(BaseModel):
    """A reference chunk that contains the answer to a question.

    Attributes:
        id: Unique identifier (hash) of the chunk
        metadata: Metadata about the chunk (file, headers, index, etc.)
        content: The text content of the chunk
    """

    id: str = Field(description="Unique identifier (hash) of the chunk")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the chunk (file, headers, index, etc.)"
    )
    content: str = Field(description="The text content of the chunk")

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {"id": self.id, "metadata": self.metadata, "content": self.content}


class QADataPoint(BaseModel):
    """A single question-answer data point with reference chunks.

    Attributes:
        question: The search query or question
        answer: The expected answer
        reference_chunks: List of chunks that contain the answer
        qa_type: Type of QA pair ("single_hop" or "multi_hop")
    """

    question: str = Field(description="The search query or question")
    answer: str = Field(description="The expected answer")
    reference_chunks: list[ReferenceChunk] = Field(
        description="List of chunks that contain the answer"
    )
    qa_type: str = Field(
        default="single_hop",
        description="Type of QA pair (single_hop or multi_hop)"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "question": self.question,
            "answer": self.answer,
            "reference_chunks": [chunk.to_dict() for chunk in self.reference_chunks],
            "qa_type": self.qa_type,
        }


class QADataset(BaseModel):
    """A collection of QA data points.

    Attributes:
        data_points: List of QA data points
    """

    data_points: list[QADataPoint] = Field(
        default_factory=list,
        description="List of QA data points"
    )

    def __iter__(self):
        """Iterate over all data points."""
        return iter(self.data_points)

    def __len__(self):
        """Return total number of data points."""
        return len(self.data_points)

    def __getitem__(self, idx):
        """Get data point by index."""
        return self.data_points[idx]

    def add(self, data_point: QADataPoint) -> None:
        """Add a data point to the dataset."""
        self.data_points.append(data_point)

    def extend(self, data_points: list[QADataPoint]) -> None:
        """Add multiple data points to the dataset."""
        self.data_points.extend(data_points)

    def merge(self, other: "QADataset") -> "QADataset":
        """Merge this dataset with another, returning a new dataset.

        Args:
            other: Another QADataset to merge with

        Returns:
            New QADataset containing all data points from both datasets
        """
        return QADataset(data_points=self.data_points + other.data_points)

    def filter_by_type(self, qa_type: str) -> "QADataset":
        """Filter data points by QA type.

        Args:
            qa_type: Type to filter by ("single_hop" or "multi_hop")

        Returns:
            New QADataset with filtered data points
        """
        filtered = [dp for dp in self.data_points if dp.qa_type == qa_type]
        return QADataset(data_points=filtered)

    @property
    def single_hop_count(self) -> int:
        """Number of single-hop data points."""
        return sum(1 for dp in self.data_points if dp.qa_type == "single_hop")

    @property
    def multi_hop_count(self) -> int:
        """Number of multi-hop data points."""
        return sum(1 for dp in self.data_points if dp.qa_type == "multi_hop")

    def to_list(self) -> list[dict]:
        """Convert to list of dictionaries."""
        return [dp.to_dict() for dp in self.data_points]

    def summary(self) -> str:
        """Return a summary string of the dataset."""
        return (
            f"QADataset: {len(self)} total data points\n"
            f"  Single-hop: {self.single_hop_count}\n"
            f"  Multi-hop: {self.multi_hop_count}"
        )
