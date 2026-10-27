"""
Integration tests for end-to-end scientific retrieval pipeline.
"""

from tests.fixtures.synthetic_data import create_synthetic_corpus, create_synthetic_queries_and_qrels
from scientific_rag.evaluation.aggregation import aggregate_chunks_to_documents
from scientific_rag.evaluation.benchmark import run_retrieval_benchmark
from scientific_rag.evaluation.metrics import compute_all_metrics
from scientific_rag.preprocessing.chunker import DeterministicSentenceChunker
from scientific_rag.retrieval.bm25 import BM25Retriever
from scientific_rag.retrieval.dense import DenseRetriever


def test_end_to_end_bm25_pipeline():
    docs = create_synthetic_corpus()
    chunker = DeterministicSentenceChunker(sentences_per_chunk=2, sentence_overlap=0)
    chunks = chunker.chunk_corpus(docs)

    retriever = BM25Retriever()
    retriever.index(chunks)

    queries, qrels = create_synthetic_queries_and_qrels()

    # Query q1: "How does Cas9 induce targeted double-strand breaks in DNA?"
    # Gold: doc_101
    results = retriever.retrieve(queries["q1"].text, top_k=5)
    doc_ids, _ = aggregate_chunks_to_documents(results, top_k=3)

    assert "doc_101" in doc_ids
    assert doc_ids[0] == "doc_101"

    metrics = compute_all_metrics(doc_ids, {"doc_101"})
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr@10"] == 1.0
    assert metrics["ndcg@10"] == 1.0


def test_end_to_end_dense_pipeline():
    docs = create_synthetic_corpus()
    chunker = DeterministicSentenceChunker(sentences_per_chunk=2, sentence_overlap=0)
    chunks = chunker.chunk_corpus(docs)

    retriever = DenseRetriever(device="cpu", batch_size=4)
    retriever.index(chunks)

    queries, qrels = create_synthetic_queries_and_qrels()

    results = retriever.retrieve(queries["q2"].text, top_k=5)
    doc_ids, _ = aggregate_chunks_to_documents(results, top_k=3)

    # Gold for q2 is doc_102 (Histone H3)
    assert "doc_102" in doc_ids
    metrics = compute_all_metrics(doc_ids, {"doc_102"})
    assert metrics["recall@5"] == 1.0


def test_benchmark_runner_synthetic(tmp_path):
    docs = create_synthetic_corpus()
    chunker = DeterministicSentenceChunker(sentences_per_chunk=2, sentence_overlap=0)
    chunks = chunker.chunk_corpus(docs)

    retriever = BM25Retriever()
    retriever.index(chunks)

    queries, qrels = create_synthetic_queries_and_qrels()
    summary = run_retrieval_benchmark(
        retriever=retriever,
        queries=queries,
        qrels=qrels,
        split_name="synthetic_test",
        output_dir=tmp_path,
        candidate_chunk_k=10,
        target_doc_k=5,
    )

    assert summary.num_queries == 4
    assert summary.metrics["recall@5"] > 0.0
    assert (tmp_path / "benchmark_bm25okapi_synthetic_test.json").exists()
