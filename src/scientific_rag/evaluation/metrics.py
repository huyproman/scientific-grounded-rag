"""
Standard Information Retrieval (IR) evaluation metrics.
Implements Recall@K, MRR@10, and nDCG@10 with binary relevance semantics.
"""

import math
from typing import Dict, List, Set, Union


def recall_at_k(
    ranked_doc_ids: List[str],
    gold_doc_ids: Set[str],
    k: int,
) -> float:
    """
    Computes Recall@K:
    Proportion of relevant gold documents retrieved in top-K unique documents.
    """
    if not gold_doc_ids or k <= 0:
        return 0.0
    top_k_docs = set(ranked_doc_ids[:k])
    hits = len(top_k_docs.intersection(gold_doc_ids))
    return float(hits) / float(len(gold_doc_ids))


def mrr_at_k(
    ranked_doc_ids: List[str],
    gold_doc_ids: Set[str],
    k: int = 10,
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR@K):
    Reciprocal rank of the first relevant document in top-K.
    """
    if not gold_doc_ids or k <= 0:
        return 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / float(rank)
    return 0.0


def dcg_at_k(
    ranked_doc_ids: List[str],
    gold_relevance: Dict[str, int],
    k: int = 10,
) -> float:
    """Computes Discounted Cumulative Gain (DCG@K)."""
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = gold_relevance.get(doc_id, 0)
        if rel > 0:
            dcg += float(rel) / math.log2(rank + 1)
    return dcg


def ndcg_at_k(
    ranked_doc_ids: List[str],
    gold_relevance: Union[Dict[str, int], Set[str]],
    k: int = 10,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (nDCG@K).
    Supports binary (Set[str]) and graded (Dict[str, int]) relevance.
    """
    if not gold_relevance or k <= 0:
        return 0.0

    if isinstance(gold_relevance, set):
        gold_rel_dict = {doc_id: 1 for doc_id in gold_relevance}
    else:
        gold_rel_dict = dict(gold_relevance)

    actual_dcg = dcg_at_k(ranked_doc_ids, gold_rel_dict, k=k)
    if actual_dcg == 0.0:
        return 0.0

    ideal_scores = sorted(gold_rel_dict.values(), reverse=True)
    ideal_dcg = 0.0
    for rank, rel in enumerate(ideal_scores[:k], start=1):
        if rel > 0:
            ideal_dcg += float(rel) / math.log2(rank + 1)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_all_metrics(
    ranked_doc_ids: List[str],
    gold_doc_ids: Set[str],
) -> Dict[str, float]:
    """Computes Recall@5, Recall@10, Recall@20, MRR@10, and nDCG@10 for a single query."""
    return {
        "recall@5": recall_at_k(ranked_doc_ids, gold_doc_ids, k=5),
        "recall@10": recall_at_k(ranked_doc_ids, gold_doc_ids, k=10),
        "recall@20": recall_at_k(ranked_doc_ids, gold_doc_ids, k=20),
        "mrr@10": mrr_at_k(ranked_doc_ids, gold_doc_ids, k=10),
        "ndcg@10": ndcg_at_k(ranked_doc_ids, gold_doc_ids, k=10),
    }
