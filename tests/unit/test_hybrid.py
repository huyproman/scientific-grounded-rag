"""
Unit tests for Reciprocal Rank Fusion (RRF) hybrid retriever.
Covers hand-computed scores, ties, missing documents, duplicate chunks, and chunk provenance.
"""

from typing import List
import pytest

from scientific_rag.data.models import Chunk, RetrievalResult
from scientific_rag.retrieval.base import BaseRetriever
from scientific_rag.retrieval.hybrid import FusedDocument, HybridRRFRetriever, compute_rrf_score


class MockRetriever(BaseRetriever):
    """Mock retriever returning predetermined RetrievalResult lists."""

    def __init__(self, name: str, results: List[RetrievalResult]):
        super().__init__(name=name)
        self.results = results
        self.chunks = [
            Chunk(chunk_id=r.chunk_id, doc_id=r.doc_id, chunk_index=i, text=r.text, title=r.title)
            for i, r in enumerate(results)
        ]

    def index(self, chunks: List[Chunk]) -> None:
        pass

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        return self.results[:top_k]

    def save(self, path) -> None:
        pass

    @classmethod
    def load(cls, path):
        pass


def test_t01_compute_rrf_score_known_values():
    """T01: Verifies exact hand-computed RRF scores with default k=60."""
    # Rank 1 in both: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032786885
    score_1_1 = compute_rrf_score([1, 1], k=60)
    assert pytest.approx(score_1_1, 1e-6) == 2.0 / 61.0

    # Rank 1 in BM25, Rank 2 in Dense: 1/61 + 1/62 ≈ 0.01639344 + 0.01612903 = 0.03252247
    score_1_2 = compute_rrf_score([1, 2], k=60)
    assert pytest.approx(score_1_2, 1e-6) == (1.0 / 61.0 + 1.0 / 62.0)

    # Missing in one (rank 1 only): 1/61
    score_1_only = compute_rrf_score([1], k=60)
    assert pytest.approx(score_1_only, 1e-6) == 1.0 / 61.0


def test_t02_deterministic_fusion_and_ranking():
    """T02: Verifies that hybrid fusion ranks documents deterministically by RRF score."""
    bm25_res = [
        RetrievalResult(chunk_id="c_docA_1", doc_id="docA", score=10.0, rank=1, text="textA"),
        RetrievalResult(chunk_id="c_docB_1", doc_id="docB", score=8.0, rank=2, text="textB"),
        RetrievalResult(chunk_id="c_docC_1", doc_id="docC", score=6.0, rank=3, text="textC"),
    ]
    dense_res = [
        RetrievalResult(chunk_id="c_docB_2", doc_id="docB", score=0.95, rank=1, text="textB2"),
        RetrievalResult(chunk_id="c_docC_2", doc_id="docC", score=0.90, rank=2, text="textC2"),
        RetrievalResult(chunk_id="c_docA_2", doc_id="docA", score=0.85, rank=3, text="textA2"),
    ]

    r_bm25 = MockRetriever("bm25", bm25_res)
    r_dense = MockRetriever("dense", dense_res)
    hybrid = HybridRRFRetriever(r_bm25, r_dense, rrf_k=60)

    fused = hybrid.fuse_documents("query", target_doc_k=3)
    assert len(fused) == 3

    # docA: bm25 rank 1, dense rank 3 -> 1/61 + 1/63 ≈ 0.0163934 + 0.0158730 = 0.032266
    # docB: bm25 rank 2, dense rank 1 -> 1/62 + 1/61 ≈ 0.0161290 + 0.0163934 = 0.032522
    # docC: bm25 rank 3, dense rank 2 -> 1/63 + 1/62 ≈ 0.0158730 + 0.0161290 = 0.032002
    # Expected order: docB, docA, docC
    assert [d.doc_id for d in fused] == ["docB", "docA", "docC"]
    assert fused[0].rank == 1
    assert fused[1].rank == 2
    assert fused[2].rank == 3


def test_t03_chunk_metadata_and_provenance_preservation():
    """T03: Verifies contributing chunk provenance is retained for evidence generation."""
    bm25_res = [
        RetrievalResult(chunk_id="c1", doc_id="doc1", score=5.0, rank=1, text="bm25 text")
    ]
    dense_res = [
        RetrievalResult(chunk_id="c2", doc_id="doc1", score=0.8, rank=1, text="dense text")
    ]

    hybrid = HybridRRFRetriever(MockRetriever("bm25", bm25_res), MockRetriever("dense", dense_res))
    fused = hybrid.fuse_documents("query", target_doc_k=1)

    assert len(fused) == 1
    doc = fused[0]
    assert doc.doc_id == "doc1"
    assert doc.bm25_chunk.chunk_id == "c1"
    assert doc.dense_chunk.chunk_id == "c2"
    # Evidence policy prefers dense chunk when available
    assert doc.selected_evidence_chunk.chunk_id == "c2"
    assert len(doc.contributing_chunks) == 2


def test_t04_missing_documents_in_one_retriever():
    """T04: Verifies documents retrieved by only one retriever are properly scored and included."""
    bm25_res = [
        RetrievalResult(chunk_id="c_docA", doc_id="docA", score=10.0, rank=1, text="A"),
        RetrievalResult(chunk_id="c_docB", doc_id="docB", score=5.0, rank=2, text="B"),
    ]
    dense_res = [
        RetrievalResult(chunk_id="c_docC", doc_id="docC", score=0.9, rank=1, text="C"),
    ]

    hybrid = HybridRRFRetriever(MockRetriever("bm25", bm25_res), MockRetriever("dense", dense_res))
    fused = hybrid.fuse_documents("query", target_doc_k=3)

    assert len(fused) == 3
    doc_ids = [d.doc_id for d in fused]
    assert "docA" in doc_ids
    assert "docB" in doc_ids
    assert "docC" in doc_ids

    # docA has rank 1 in bm25, missing in dense -> score = 1/61
    # docC has rank 1 in dense, missing in bm25 -> score = 1/61
    # tie break: docA < docC lexicographically
    assert doc_ids[0] == "docA"
    assert doc_ids[1] == "docC"
    assert doc_ids[2] == "docB"


def test_t05_tie_breaking_is_deterministic():
    """T05: Verifies that identical scores are broken deterministically by doc_id ascending."""
    bm25_res = [
        RetrievalResult(chunk_id="c_z", doc_id="doc_z", score=10.0, rank=1, text="z"),
        RetrievalResult(chunk_id="c_a", doc_id="doc_a", score=10.0, rank=2, text="a"),
    ]
    dense_res = [
        RetrievalResult(chunk_id="c_a_d", doc_id="doc_a", score=0.9, rank=1, text="a"),
        RetrievalResult(chunk_id="c_z_d", doc_id="doc_z", score=0.9, rank=2, text="z"),
    ]

    hybrid = HybridRRFRetriever(MockRetriever("bm25", bm25_res), MockRetriever("dense", dense_res))
    fused = hybrid.fuse_documents("query", target_doc_k=2)

    # Both doc_a and doc_z have ranks [1, 2] -> identical RRF score
    assert pytest.approx(fused[0].rrf_score, 1e-7) == fused[1].rrf_score
    # doc_a must precede doc_z by deterministic alphabetical order
    assert fused[0].doc_id == "doc_a"
    assert fused[1].doc_id == "doc_z"


def test_t06_empty_retrievers_return_empty():
    """T06: Verifies graceful handling of empty result lists."""
    hybrid = HybridRRFRetriever(MockRetriever("bm25", []), MockRetriever("dense", []))
    fused = hybrid.fuse_documents("query", target_doc_k=5)
    assert fused == []

    chunk_res = hybrid.retrieve("query", top_k=5)
    assert chunk_res == []
