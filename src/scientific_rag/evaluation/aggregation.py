"""
Chunk-to-Document rank aggregation and deduplication logic.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from scientific_rag.data.models import RetrievalResult


def aggregate_chunks_to_documents(
    chunk_results: List[RetrievalResult],
    top_k: int = 10,
) -> Tuple[List[str], List[RetrievalResult]]:
    """
    Aggregates chunk-level retrieval candidates into a deduplicated parent document ranking.

    Aggregation Rule:
    - Traverses ranked chunk candidates in descending order of retrieval score.
    - Each parent document ID is recorded at its EARLIEST appearance (highest score / best rank).
    - Subsequent chunks from the same parent document are skipped to prevent duplicate document inflation.
    - Halts once `top_k` unique documents are accumulated, or all chunk candidates are exhausted.

    Returns:
    - deduplicated_doc_ids: List of unique document IDs in rank order (length <= top_k).
    - representative_chunks: Corresponding best-scoring RetrievalResult for each unique document.
    """
    seen_docs: Set[str] = set()
    deduplicated_doc_ids: List[str] = []
    representative_chunks: List[RetrievalResult] = []

    for chunk in chunk_results:
        doc_id = chunk.doc_id
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            deduplicated_doc_ids.append(doc_id)
            representative_chunks.append(chunk)

            if len(deduplicated_doc_ids) >= top_k:
                break

    return deduplicated_doc_ids, representative_chunks


def retrieve_unique_documents(
    retriever: Any,
    query: str,
    target_doc_k: int = 20,
    initial_chunk_k: int = 100,
    max_chunk_k: Optional[int] = None,
) -> Tuple[List[str], List[RetrievalResult]]:
    """
    Retrieves chunks with dynamic depth expansion to guarantee that exactly
    target_doc_k unique documents are obtained, or the underlying corpus is
    genuinely exhausted.

    Correctness Invariant:
    - No arbitrary candidate ceiling (such as 2000 chunks) may truncate retrieval.
    - If max_chunk_k is None, expands up to the full searchable corpus size.
    - Retrieval only terminates when target_doc_k unique documents are accumulated,
      or when the retriever cannot return more chunks (true corpus exhaustion).
    """
    # Determine total available chunks if retriever exposes them
    total_chunks = len(getattr(retriever, "chunks", []))
    effective_max_k = max_chunk_k or (total_chunks if total_chunks > 0 else 50000)

    current_chunk_k = max(initial_chunk_k, target_doc_k)
    while True:
        chunk_results = retriever.retrieve(query, top_k=current_chunk_k)
        doc_ids, rep_chunks = aggregate_chunks_to_documents(chunk_results, top_k=target_doc_k)

        # Condition A: target_doc_k reached
        if len(doc_ids) >= target_doc_k:
            return doc_ids, rep_chunks

        # Condition B: True corpus exhaustion (retriever returned fewer chunks than requested)
        if len(chunk_results) < current_chunk_k:
            return doc_ids, rep_chunks

        # Condition C: Reached maximum searchable corpus depth
        if current_chunk_k >= effective_max_k:
            return doc_ids, rep_chunks

        # Otherwise expand candidate depth exponentially
        current_chunk_k = min(current_chunk_k * 2, effective_max_k)


