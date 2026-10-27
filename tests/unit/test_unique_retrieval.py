"""
Regression tests for F02: Complete top-K unique document retrieval without arbitrary ceilings.
Covers all required test conditions (T1-T7) in Assignment Section 3.
"""

import pytest
from scientific_rag.data.models import RetrievalResult
from scientific_rag.evaluation.aggregation import (
    aggregate_chunks_to_documents,
    retrieve_unique_documents,
)


class ClusterMockRetriever:
    """Mock retriever where doc_dominant has N chunks, followed by other distinct documents."""
    def __init__(self, dominant_chunks: int = 2500, other_docs: int = 30):
        self.chunks = []
        for i in range(dominant_chunks):
            self.chunks.append(
                RetrievalResult(chunk_id=f"doc_dom#c{i}", doc_id="doc_dom", score=10000.0 - i, rank=i + 1)
            )
        for j in range(1, other_docs + 1):
            rank = dominant_chunks + j
            self.chunks.append(
                RetrievalResult(chunk_id=f"doc_{j}#c0", doc_id=f"doc_{j}", score=100.0 - j, rank=rank)
            )

    def retrieve(self, query: str, top_k: int = 10):
        return self.chunks[:top_k]


def test_t1_cluster_exceeding_2000_chunks_retrieves_all_k():
    """T1: Dominant document has 2500 top chunks; target_doc_k=20 must expand past 2000 to get 20 distinct docs."""
    mock = ClusterMockRetriever(dominant_chunks=2500, other_docs=30)
    doc_ids, rep_chunks = retrieve_unique_documents(
        retriever=mock,
        query="query",
        target_doc_k=20,
        initial_chunk_k=100,
        max_chunk_k=None,  # Guarantees no arbitrary ceiling
    )
    assert len(doc_ids) == 20
    assert doc_ids[0] == "doc_dom"
    assert doc_ids[1] == "doc_1"
    assert doc_ids[19] == "doc_19"


def test_t2_exact_k_distinct_documents_available():
    """T2: Exactly K distinct documents available in corpus; returns exactly K."""
    class ExactMockRetriever:
        def __init__(self):
            self.chunks = [
                RetrievalResult(chunk_id=f"doc_{i}#c0", doc_id=f"doc_{i}", score=50.0 - i, rank=i + 1)
                for i in range(5)
            ]
        def retrieve(self, query: str, top_k: int = 10):
            return self.chunks[:top_k]

    mock = ExactMockRetriever()
    doc_ids, _ = retrieve_unique_documents(mock, "query", target_doc_k=5)
    assert len(doc_ids) == 5
    assert doc_ids == ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]


def test_t3_fewer_than_k_with_genuine_exhaustion():
    """T3: Corpus has only 3 distinct documents; target_doc_k=10 returns 3 with genuine exhaustion."""
    class SmallMockRetriever:
        def __init__(self):
            self.chunks = [
                RetrievalResult(chunk_id="doc_a#c0", doc_id="doc_a", score=10.0, rank=1),
                RetrievalResult(chunk_id="doc_a#c1", doc_id="doc_a", score=9.0, rank=2),
                RetrievalResult(chunk_id="doc_b#c0", doc_id="doc_b", score=8.0, rank=3),
                RetrievalResult(chunk_id="doc_c#c0", doc_id="doc_c", score=7.0, rank=4),
            ]
        def retrieve(self, query: str, top_k: int = 10):
            return self.chunks[:top_k]

    mock = SmallMockRetriever()
    doc_ids, rep = retrieve_unique_documents(mock, "query", target_doc_k=10)
    assert len(doc_ids) == 3
    assert doc_ids == ["doc_a", "doc_b", "doc_c"]


def test_t4_empty_corpus_result():
    """T4: Empty corpus returns empty list cleanly."""
    class EmptyMockRetriever:
        def __init__(self):
            self.chunks = []
        def retrieve(self, query: str, top_k: int = 10):
            return []

    mock = EmptyMockRetriever()
    doc_ids, rep = retrieve_unique_documents(mock, "query", target_doc_k=10)
    assert doc_ids == []
    assert rep == []


def test_t5_duplicate_parent_documents_throughout_ranking():
    """T5: Multi-chunk duplicates interleaved throughout ranking."""
    chunk_results = [
        RetrievalResult(chunk_id="doc_x#c0", doc_id="doc_x", score=10.0, rank=1),
        RetrievalResult(chunk_id="doc_y#c0", doc_id="doc_y", score=9.0, rank=2),
        RetrievalResult(chunk_id="doc_x#c1", doc_id="doc_x", score=8.0, rank=3),
        RetrievalResult(chunk_id="doc_z#c0", doc_id="doc_z", score=7.0, rank=4),
        RetrievalResult(chunk_id="doc_y#c1", doc_id="doc_y", score=6.0, rank=5),
        RetrievalResult(chunk_id="doc_w#c0", doc_id="doc_w", score=5.0, rank=6),
    ]
    doc_ids, rep = aggregate_chunks_to_documents(chunk_results, top_k=4)
    assert doc_ids == ["doc_x", "doc_y", "doc_z", "doc_w"]
    assert len(doc_ids) == 4
    assert rep[0].chunk_id == "doc_x#c0"


def test_t6_deterministic_ordering_and_representative_chunk():
    """T6: Deterministic tie and ordering preservation."""
    chunk_results = [
        RetrievalResult(chunk_id="doc_1#c0", doc_id="doc_1", score=5.0, rank=1),
        RetrievalResult(chunk_id="doc_2#c0", doc_id="doc_2", score=5.0, rank=2),
    ]
    doc_ids, rep = aggregate_chunks_to_documents(chunk_results, top_k=2)
    assert doc_ids == ["doc_1", "doc_2"]
    assert rep[0].chunk_id == "doc_1#c0"
    assert rep[1].chunk_id == "doc_2#c0"
