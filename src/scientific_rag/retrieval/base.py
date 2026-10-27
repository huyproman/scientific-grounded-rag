"""
Abstract Base Class for standalone retrieval baselines.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from scientific_rag.data.models import Chunk, RetrievalResult


class BaseRetriever(ABC):
    """Common interface for all standalone scientific retrievers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def index(self, chunks: List[Chunk]) -> None:
        """Builds retrieval index from a list of chunks."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieves top_k ranked chunk candidates for a query string.
        Returns ordered list of RetrievalResult objects.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serializes index artifacts to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseRetriever":
        """Loads serialized index artifacts from disk."""
        pass
