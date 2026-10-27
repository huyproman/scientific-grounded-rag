"""
Reciprocal Rank Fusion (RRF) hybrid retrieval combining lexical (BM25) and dense retrievers.
Fuses on unique parent-document IDs while tracking contributing chunks for evidence selection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from scientific_rag.data.models import Chunk, RetrievalResult
from scientific_rag.retrieval.base import BaseRetriever
from scientific_rag.evaluation.aggregation import retrieve_unique_documents


@dataclass
class FusedDocument:
    """Represents a fused unique document candidate with contributing chunk provenance."""
    doc_id: str
    rrf_score: float
    rank: int
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_chunk: Optional[RetrievalResult] = None
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    dense_chunk: Optional[RetrievalResult] = None
    selected_evidence_chunk: Optional[RetrievalResult] = None
    contributing_chunks: List[RetrievalResult] = field(default_factory=list)


def compute_rrf_score(ranks: List[int], k: int = 60) -> float:
    """
    Computes standard Reciprocal Rank Fusion score:
    score(d) = sum(1 / (k + rank_i(d))) where rank_i is 1-based.
    """
    return sum(1.0 / (k + r) for r in ranks)


class HybridRRFRetriever(BaseRetriever):
    """
    Hybrid retriever executing Reciprocal Rank Fusion over lexical (BM25) and dense retrievers.
    
    Invariants:
    1. Reuses existing BM25 and Dense retrievers without modification.
    2. Fuses on unique parent-document IDs.
    3. Maintains complete chunk provenance (contributing BM25 and dense chunks).
    4. Deterministic tie-breaking by (rrf_score DESC, doc_id ASC).
    5. Preserves BaseRetriever interface (retrieve returns RetrievalResult list using representative evidence chunk).
    """

    def __init__(
        self,
        bm25_retriever: BaseRetriever,
        dense_retriever: BaseRetriever,
        rrf_k: int = 60,
        name: str = "HybridRRF",
    ):
        super().__init__(name=name)
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.chunks = getattr(bm25_retriever, "chunks", []) or getattr(dense_retriever, "chunks", [])

    def index(self, chunks: List[Chunk]) -> None:
        """Indexes chunks across both underlying retrievers."""
        self.bm25.index(chunks)
        self.dense.index(chunks)
        self.chunks = chunks

    def fuse_documents(
        self,
        query: str,
        target_doc_k: int = 20,
        candidate_depth: Optional[int] = None,
    ) -> List[FusedDocument]:
        """
        Executes both retrievers, aggregates to unique documents, and fuses using RRF.
        
        Returns ordered list of FusedDocument objects (length <= target_doc_k).
        """
        fetch_k = candidate_depth or max(target_doc_k * 5, 100)

        bm25_docs, bm25_chunks = retrieve_unique_documents(
            self.bm25, query, target_doc_k=fetch_k
        )
        dense_docs, dense_chunks = retrieve_unique_documents(
            self.dense, query, target_doc_k=fetch_k
        )

        doc_data: Dict[str, Dict[str, Any]] = {}

        for rank_idx, (doc_id, chunk) in enumerate(zip(bm25_docs, bm25_chunks), start=1):
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "bm25_rank": None,
                    "bm25_score": None,
                    "bm25_chunk": None,
                    "dense_rank": None,
                    "dense_score": None,
                    "dense_chunk": None,
                    "contributing": [],
                }
            doc_data[doc_id]["bm25_rank"] = rank_idx
            doc_data[doc_id]["bm25_score"] = chunk.score
            doc_data[doc_id]["bm25_chunk"] = chunk
            doc_data[doc_id]["contributing"].append(chunk)

        for rank_idx, (doc_id, chunk) in enumerate(zip(dense_docs, dense_chunks), start=1):
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "bm25_rank": None,
                    "bm25_score": None,
                    "bm25_chunk": None,
                    "dense_rank": None,
                    "dense_score": None,
                    "dense_chunk": None,
                    "contributing": [],
                }
            doc_data[doc_id]["dense_rank"] = rank_idx
            doc_data[doc_id]["dense_score"] = chunk.score
            doc_data[doc_id]["dense_chunk"] = chunk
            doc_data[doc_id]["contributing"].append(chunk)

        fused_items: List[FusedDocument] = []
        for doc_id, data in doc_data.items():
            ranks: List[int] = []
            if data["bm25_rank"] is not None:
                ranks.append(data["bm25_rank"])
            if data["dense_rank"] is not None:
                ranks.append(data["dense_rank"])

            score = compute_rrf_score(ranks, k=self.rrf_k)

            # Evidence selection policy:
            # Prefer dense chunk (semantic alignment), fall back to BM25 chunk
            selected_chunk = data["dense_chunk"] or data["bm25_chunk"]

            fused_items.append(
                FusedDocument(
                    doc_id=doc_id,
                    rrf_score=score,
                    rank=0,  # assigned after sorting
                    bm25_rank=data["bm25_rank"],
                    bm25_score=data["bm25_score"],
                    bm25_chunk=data["bm25_chunk"],
                    dense_rank=data["dense_rank"],
                    dense_score=data["dense_score"],
                    dense_chunk=data["dense_chunk"],
                    selected_evidence_chunk=selected_chunk,
                    contributing_chunks=data["contributing"],
                )
            )

        fused_items.sort(key=lambda d: (-d.rrf_score, d.doc_id))

        # Assign final 1-based ranks and truncate to target_doc_k
        final_docs: List[FusedDocument] = []
        for rank_idx, item in enumerate(fused_items[:target_doc_k], start=1):
            item.rank = rank_idx
            final_docs.append(item)

        return final_docs

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        BaseRetriever compatibility method.
        Returns top_k ranked chunk candidates (representative evidence chunks from fused documents).
        """
        fused = self.fuse_documents(query, target_doc_k=top_k)
        results: List[RetrievalResult] = []
        for item in fused:
            if item.selected_evidence_chunk:
                rep = item.selected_evidence_chunk
                results.append(
                    RetrievalResult(
                        chunk_id=rep.chunk_id,
                        doc_id=item.doc_id,
                        score=item.rrf_score,
                        rank=item.rank,
                        title=rep.title,
                        text=rep.text,
                        metadata={
                            **rep.metadata,
                            "rrf_score": item.rrf_score,
                            "bm25_rank": item.bm25_rank,
                            "dense_rank": item.dense_rank,
                        },
                    )
                )
        return results

    def save(self, path: Path) -> None:
        """Underlying retrievers are persisted independently."""
        pass

    @classmethod
    def load(cls, path: Path) -> "HybridRRFRetriever":
        raise NotImplementedError("Load individual BM25 and Dense indexes, then instantiate HybridRRFRetriever.")
