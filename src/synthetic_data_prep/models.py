"""Data models for synthetic data preparation."""

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A chunk of text with associated metadata.

    Attributes:
        content: The text content of the chunk.
        metadata: Dictionary containing metadata about the chunk,
            such as header hierarchy (h1, h2, h3), source file, and index.
    """

    content: str = Field(description="The text content of the chunk")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the chunk (headers, file, index, etc.)"
    )

    def __len__(self) -> int:
        """Return the length of the chunk content."""
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dict with 'content' and 'metadata' keys.
        """
        return {"content": self.content, "metadata": self.metadata}
