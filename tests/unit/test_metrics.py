"""
Unit tests for Information Retrieval evaluation metrics.
Covers all required metric edge cases in Assignment Section 9.5.
"""

import math
import pytest
from scientific_rag.evaluation.metrics import (
    compute_all_metrics,
    dcg_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k_single_relevant_first():
    ranked = ["doc_1", "doc_2", "doc_3"]
    gold = {"doc_1"}
    assert recall_at_k(ranked, gold, k=1) == 1.0
    assert recall_at_k(ranked, gold, k=5) == 1.0


def test_recall_at_k_below_k():
    ranked = ["doc_2", "doc_3", "doc_4", "doc_1"]
    gold = {"doc_1"}
    assert recall_at_k(ranked, gold, k=2) == 0.0
    assert recall_at_k(ranked, gold, k=3) == 0.0
    assert recall_at_k(ranked, gold, k=4) == 1.0


def test_recall_at_k_multiple_relevant():
    ranked = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    gold = {"doc_1", "doc_3", "doc_9"}  # 3 gold docs, 2 retrieved in top-5
    assert recall_at_k(ranked, gold, k=3) == 2.0 / 3.0
    assert recall_at_k(ranked, gold, k=5) == 2.0 / 3.0


def test_recall_at_k_empty_cases():
    assert recall_at_k([], {"doc_1"}, k=5) == 0.0
    assert recall_at_k(["doc_1"], set(), k=5) == 0.0
    assert recall_at_k(["doc_1"], {"doc_1"}, k=0) == 0.0


def test_mrr_at_k_first_rank():
    ranked = ["doc_1", "doc_2"]
    gold = {"doc_1"}
    assert mrr_at_k(ranked, gold, k=10) == 1.0


def test_mrr_at_k_second_rank():
    ranked = ["doc_2", "doc_1", "doc_3"]
    gold = {"doc_1"}
    assert mrr_at_k(ranked, gold, k=10) == 0.5


def test_mrr_at_k_outside_k():
    ranked = ["doc_2", "doc_3", "doc_1"]
    gold = {"doc_1"}
    assert mrr_at_k(ranked, gold, k=2) == 0.0


def test_mrr_at_k_empty():
    assert mrr_at_k([], {"doc_1"}, k=10) == 0.0
    assert mrr_at_k(["doc_1"], set(), k=10) == 0.0


def test_ndcg_at_k_perfect_ranking():
    # When all relevant documents are ranked at the top, nDCG = 1.0
    ranked = ["doc_1", "doc_2", "doc_3"]
    gold = {"doc_1", "doc_2"}
    assert ndcg_at_k(ranked, gold, k=3) == 1.0
    assert ndcg_at_k(ranked, gold, k=10) == 1.0


def test_ndcg_at_k_suboptimal_ranking():
    # doc_1 is relevant, ranked at position 2
    # Actual DCG = 1 / log2(2 + 1) = 1 / 1.58496 = 0.63093
    # Ideal DCG = 1 / log2(1 + 1) = 1.0
    # Expected nDCG = 0.63093 / 1.0 = 0.63093
    ranked = ["doc_noise", "doc_1"]
    gold = {"doc_1"}
    score = ndcg_at_k(ranked, gold, k=5)
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert pytest.approx(score, rel=1e-4) == expected


def test_ndcg_at_k_zero_hits():
    ranked = ["doc_noise_1", "doc_noise_2"]
    gold = {"doc_1"}
    assert ndcg_at_k(ranked, gold, k=5) == 0.0


def test_compute_all_metrics():
    ranked = ["doc_1", "doc_2", "doc_3"]
    gold = {"doc_1"}
    metrics = compute_all_metrics(ranked, gold)
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["recall@20"] == 1.0
    assert metrics["mrr@10"] == 1.0
    assert metrics["ndcg@10"] == 1.0
