"""
Data models for scientific documents, chunks, queries, and retrieval results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """Represents a canonical scientific document (paper abstract)."""
    doc_id: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def full_text(self) -> str:
        if self.title:
            return f"{self.title}\n{self.text}".strip()
        return self.text.strip()


@dataclass
class Chunk:
    """Represents a deterministic context-preserving passage chunk."""
    chunk_id: str          # Strict format: {doc_id}#c{chunk_index}
    doc_id: str            # Parent document ID
    title: str
    text: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """Represents a scientific claim or query."""
    query_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Represents a single retrieved chunk candidate with score and rank."""
    chunk_id: str
    doc_id: str
    score: float
    rank: int
    title: str = ""
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSummary:
    """Represents aggregate and per-query benchmark evaluation metrics."""
    retriever_name: str
    split: str
    num_queries: int
    metrics: Dict[str, float]
    per_query_results: List[Dict[str, Any]]
    runtime_seconds: float
    timestamp: str
    device: str
    config: Dict[str, Any] = field(default_factory=dict)
