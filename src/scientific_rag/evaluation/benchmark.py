"""
Automated benchmark harness for evaluating standalone scientific retrieval baselines.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from scientific_rag.common.paths import BENCHMARK_OUTPUT_DIR
from scientific_rag.data.models import EvaluationSummary, Query
from scientific_rag.evaluation.aggregation import (
    aggregate_chunks_to_documents,
    retrieve_unique_documents,
)
from scientific_rag.evaluation.metrics import compute_all_metrics
from scientific_rag.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


def run_retrieval_benchmark(
    retriever: BaseRetriever,
    queries: Dict[str, Query],
    qrels: Dict[str, Dict[str, int]],
    split_name: str = "test",
    output_dir: Path = BENCHMARK_OUTPUT_DIR,
    candidate_chunk_k: int = 50,
    target_doc_k: int = 20,
) -> EvaluationSummary:
    """
    Executes standardized IR benchmark for a given retriever against queries and qrels.

    Discipline:
    - Queries are processed deterministically.
    - Chunk candidates are retrieved to sufficient depth (candidate_chunk_k) to yield target_doc_k unique docs.
    - Dedupes chunks to parent document IDs.
    - Computes Recall@5, Recall@10, Recall@20, MRR@10, nDCG@10.
    - Macro-averages metrics across all queries with positive relevance judgments.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    per_query_results: List[Dict[str, Any]] = []

    metric_sums = {
        "recall@5": 0.0,
        "recall@10": 0.0,
        "recall@20": 0.0,
        "mrr@10": 0.0,
        "ndcg@10": 0.0,
    }

    eval_query_ids = [qid for qid in sorted(queries.keys()) if qid in qrels]
    num_eval_queries = len(eval_query_ids)

    if num_eval_queries == 0:
        raise ValueError(f"No matching queries found between queries dict and qrels for split '{split_name}'.")

    for qid in eval_query_ids:
        query_obj = queries[qid]
        gold_docs = {doc_id for doc_id, score in qrels[qid].items() if score > 0}

        # Retrieve with dynamic depth expansion to guarantee target_doc_k unique documents
        ranked_doc_ids, rep_chunks = retrieve_unique_documents(
            retriever=retriever,
            query=query_obj.text,
            target_doc_k=target_doc_k,
            initial_chunk_k=candidate_chunk_k,
        )

        query_metrics = compute_all_metrics(ranked_doc_ids, gold_docs)

        for m_key, m_val in query_metrics.items():
            metric_sums[m_key] += m_val

        per_query_results.append({
            "query_id": qid,
            "query_text": query_obj.text,
            "gold_doc_ids": sorted(list(gold_docs)),
            "retrieved_doc_ids": ranked_doc_ids,
            "retrieved_top_chunks": [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "score": c.score, "rank": c.rank}
                for c in rep_chunks[:5]
            ],
            "metrics": query_metrics,
        })

    elapsed_time = time.perf_counter() - start_time
    aggregate_metrics = {
        m_key: round(m_sum / float(num_eval_queries), 5)
        for m_key, m_sum in metric_sums.items()
    }

    timestamp_str = datetime.now(timezone.utc).isoformat()
    summary = EvaluationSummary(
        retriever_name=retriever.name,
        split=split_name,
        num_queries=num_eval_queries,
        metrics=aggregate_metrics,
        per_query_results=per_query_results,
        runtime_seconds=round(elapsed_time, 2),
        timestamp=timestamp_str,
        device="cpu",
        config={
            "candidate_chunk_k": candidate_chunk_k,
            "target_doc_k": target_doc_k,
        },
    )

    safe_name = retriever.name.lower().replace(" ", "_").replace("-", "_")
    output_file = output_dir / f"benchmark_{safe_name}_{split_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "retriever_name": summary.retriever_name,
                "split": summary.split,
                "num_queries": summary.num_queries,
                "metrics": summary.metrics,
                "runtime_seconds": summary.runtime_seconds,
                "timestamp": summary.timestamp,
                "device": summary.device,
                "config": summary.config,
                "per_query_results": summary.per_query_results,
            },
            f,
            indent=2,
        )

    logger.info(
        "Benchmark '%s' on split '%s' finished in %.2fs: %s (Artifact: %s)",
        retriever.name,
        split_name,
        elapsed_time,
        aggregate_metrics,
        output_file,
    )
    return summary
