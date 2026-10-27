"""
Unit tests for chunk-to-document rank aggregation and deduplication logic.
"""

from scientific_rag.data.models import RetrievalResult
from scientific_rag.evaluation.aggregation import (
    aggregate_chunks_to_documents,
    retrieve_unique_documents,
)


def test_aggregation_deduplicates_same_doc():
    # doc_101 has 3 chunks, doc_102 has 1 chunk
    chunk_results = [
        RetrievalResult(chunk_id="doc_101#c0", doc_id="doc_101", score=10.0, rank=1),
        RetrievalResult(chunk_id="doc_101#c1", doc_id="doc_101", score=9.0, rank=2),
        RetrievalResult(chunk_id="doc_102#c0", doc_id="doc_102", score=8.0, rank=3),
        RetrievalResult(chunk_id="doc_101#c2", doc_id="doc_101", score=7.0, rank=4),
        RetrievalResult(chunk_id="doc_103#c0", doc_id="doc_103", score=6.0, rank=5),
    ]

    doc_ids, rep_chunks = aggregate_chunks_to_documents(chunk_results, top_k=3)

    # Must preserve earliest appearance order: doc_101, doc_102, doc_103
    assert doc_ids == ["doc_101", "doc_102", "doc_103"]
    assert len(rep_chunks) == 3
    # Best chunk for doc_101 should be doc_101#c0 with score 10.0
    assert rep_chunks[0].chunk_id == "doc_101#c0"
    assert rep_chunks[0].score == 10.0
    # Best chunk for doc_102 should be doc_102#c0 with score 8.0
    assert rep_chunks[1].chunk_id == "doc_102#c0"


def test_aggregation_respects_top_k():
    chunk_results = [
        RetrievalResult(chunk_id=f"doc_{i}#c0", doc_id=f"doc_{i}", score=float(100 - i), rank=i)
        for i in range(1, 10)
    ]
    doc_ids, rep_chunks = aggregate_chunks_to_documents(chunk_results, top_k=5)
    assert len(doc_ids) == 5
    assert doc_ids == ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]


def test_aggregation_handles_empty():
    doc_ids, rep_chunks = aggregate_chunks_to_documents([], top_k=10)
    assert doc_ids == []
    assert rep_chunks == []


def test_retrieve_unique_documents_dynamic_expansion():
    # Mock retriever where doc_1 has 50 chunks, and doc_2 has 1 chunk
    class MockRetriever:
        def retrieve(self, query: str, top_k: int = 10):
            res = [
                RetrievalResult(chunk_id=f"doc_1#c{i}", doc_id="doc_1", score=100.0 - i, rank=i + 1)
                for i in range(50)
            ]
            res.append(RetrievalResult(chunk_id="doc_2#c0", doc_id="doc_2", score=1.0, rank=51))
            return res[:top_k]

    mock = MockRetriever()
    # With initial_chunk_k=10, initial retrieval only has chunks of doc_1.
    # Dynamic expansion should expand depth until doc_2 is also included to reach target_doc_k=2.
    doc_ids, rep_chunks = retrieve_unique_documents(
        mock, "query", target_doc_k=2, initial_chunk_k=10, max_chunk_k=100
    )
    assert doc_ids == ["doc_1", "doc_2"]
    assert len(doc_ids) == 2

