"""
Complete 300-claim real LangGraph evaluation runner.
Executes the actual compiled LangGraph workflow:
  START -> retrieve (top-k=5) -> generate -> verify -> (conditional retry: expand top-k=15) -> verify -> safe finalization -> END
Records comprehensive execution traces, separates proposed vs final verdicts,
tracks token usage and latency, and saves incremental JSONL checkpoints.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from scientific_rag.common.paths import ARTIFACTS_DIR, RAW_DATA_DIR
from scientific_rag.data.loader import DatasetLoader
from scientific_rag.data.sentence_provenance import SciFactSentenceProvenance
from scientific_rag.evaluation.grounding import EvaluationCaseStatus, evaluate_grounded_analyses
from scientific_rag.generation.adapter import NineRouterAdapter, ProviderExecutionError
from scientific_rag.generation.schemas import ClaimVerdict, GroundedClaimAnalysis
from scientific_rag.retrieval.bm25 import BM25Retriever
from scientific_rag.retrieval.dense import DenseRetriever
from scientific_rag.retrieval.hybrid import HybridRRFRetriever
from scientific_rag.verification.gold_alignment import PostHocGoldEvaluator
from scientific_rag.verification.support import PretrainedNLIProxy
from scientific_rag.verification.verifier import MultiLayerEvidenceVerifier
from scientific_rag.workflows.graph import WorkflowState, create_verification_graph

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("langgraph_eval_300")


import hashlib

import subprocess

# Canonical configuration dictionary for deterministic SHA256 fingerprinting
CANONICAL_CONFIG = {
    "dataset": "SciFact",
    "split": "test",
    "claims_count": 300,
    "corpus_documents": 5183,
    "retrieval": {
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "dense_model": "BAAI/bge-small-en-v1.5",
        "rrf_k": 60,
        "initial_top_k": 5,
        "expanded_top_k": 15,
    },
    "generation": {
        "provider": "9router",
        "model": "ag/gemini-3.8-flash-high",
        "temperature": 0.0,
        "max_graph_retries": 1,
    },
    "verification": {
        "nli_model": "cross-encoder/nli-deberta-v3-xsmall",
        "nli_threshold": 0.60,
        "canonical_sentence_provenance": True,
    },
    "execution_mode": "REAL_9ROUTER_GEMINI_LANGGRAPH",
}

CONFIG_FINGERPRINT = hashlib.sha256(json.dumps(CANONICAL_CONFIG, sort_keys=True).encode("utf-8")).hexdigest()
AUTHORITATIVE_RUN_ID = f"p01_r2_final_accounted_graph_{CONFIG_FINGERPRINT[:8]}"

def get_git_revision() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_REVISION"

SOURCE_REVISION = get_git_revision()


def run_full_langgraph_benchmark_300(
    concurrency: int = 3,
    limit: Optional[int] = None,
    checkpoint_filename: str = "r2_langgraph_300_checkpoint.jsonl",
    summary_filename: str = "r2_langgraph_benchmark_300.json",
    run_id: str = AUTHORITATIVE_RUN_ID,
) -> Dict[str, Any]:
    """
    Executes the real LangGraph workflow for all 300 SciFact dev claims with full audit tracing.
    Enforces strict checkpoint compatibility: rejects or ignores legacy records lacking the valid run_id or config SHA256.
    """
    loader = DatasetLoader(RAW_DATA_DIR)
    queries = loader.load_queries(split="test")
    qrels = loader.load_qrels(split="test")
    rationales = loader.load_gold_rationales()

    provenance = SciFactSentenceProvenance()
    provenance.load_corpus()

    bm25 = BM25Retriever.load(ARTIFACTS_DIR / "indices" / "bm25_index.pkl")
    dense = DenseRetriever.load(ARTIFACTS_DIR / "indices" / "dense_index", device="cpu")
    hybrid = HybridRRFRetriever(bm25_retriever=bm25, dense_retriever=dense, rrf_k=60)

    nli_model = PretrainedNLIProxy(enable_neural=True)
    verifier = MultiLayerEvidenceVerifier(qrels=qrels, rationales=rationales, nli_proxy=nli_model, provenance=provenance)
    post_hoc_evaluator = PostHocGoldEvaluator(qrels=qrels, rationales=rationales)

    checkpoint_file = ARTIFACTS_DIR / "benchmarks" / checkpoint_filename
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    completed_map = {}
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    rec = json.loads(line_str)
                    qid = rec.get("claim_id")
                    rec_run_id = rec.get("run_id")
                    rec_config_sha = rec.get("config_fingerprint")
                    # Strict compatibility check:
                    # Only accept records matching both authoritative run_id AND config_fingerprint!
                    if qid and rec_run_id == run_id and rec_config_sha == CONFIG_FINGERPRINT:
                        completed_map[qid] = rec
                except Exception:
                    pass
        logger.info("Found %d compatible records for run '%s' in LangGraph checkpoint.", len(completed_map), run_id)

    query_list = list(queries.values()) if isinstance(queries, dict) else queries
    if limit:
        query_list = query_list[:limit]
        logger.info("Limiting LangGraph evaluation to %d queries.", len(query_list))

    remaining_queries = [q for q in query_list if q.query_id not in completed_map]
    logger.info("Remaining queries to evaluate via LangGraph: %d / %d", len(remaining_queries), len(query_list))

    def process_claim_langgraph(q):
        collected_requests = []
        def record_request_event(ev_dict):
            collected_requests.append(ev_dict)

        adapter = NineRouterAdapter(
            provenance=provenance,
            timeout_seconds=60,
            request_sink=record_request_event,
        )
        graph = create_verification_graph(
            retriever=hybrid,
            generator=adapter,
            verifier_fn=verifier.verify,
            max_retries=1,
        )

        start_time = time.time()
        initial_state: WorkflowState = {
            "claim_id": q.query_id,
            "claim_text": q.text,
            "evidence_items": [],
            "analysis": None,
            "verification": None,
            "retry_count": 0,
            "max_retries": 1,
            "final_verdict": None,
            "is_verified_conclusive": False,
            "workflow_outcome": "",
            "generation_attempts": [],
            "provider_requests": [],
        }

        try:
            time.sleep(0.5)  # Pace requests to avoid circuit-breaker trip
            final_state = graph.invoke(initial_state)
            workflow_latency = time.time() - start_time
            analysis: Optional[GroundedClaimAnalysis] = final_state.get("analysis")

            if analysis is not None:
                analysis.model_metadata.latency_seconds = workflow_latency
                post_hoc_evaluator.evaluate_analysis(analysis)

            provider_reqs = final_state.get("provider_requests", []) or collected_requests
            gen_attempts = [a.model_dump() for a in final_state.get("generation_attempts", [])]

            return {
                "run_id": run_id,
                "config_fingerprint": CONFIG_FINGERPRINT,
                "source_revision": SOURCE_REVISION,
                "claim_id": q.query_id,
                "status": EvaluationCaseStatus.COMPLETED.value,
                "execution_mode": "REAL_9ROUTER_GEMINI_LANGGRAPH",
                "provider": "9router",
                "requested_model": "ag/gemini-3.8-flash-high",
                "proposed_verdict": analysis.verdict.value if analysis else None,
                "final_workflow_verdict": final_state.get("final_verdict"),
                "is_verified_conclusive": final_state.get("is_verified_conclusive"),
                "workflow_outcome": final_state.get("workflow_outcome"),
                "retry_count": final_state.get("retry_count", 0),
                "workflow_latency_seconds": workflow_latency,
                "analysis_dict": analysis.model_dump() if analysis else None,
                "generation_attempts": gen_attempts,
                "provider_requests": provider_reqs,
            }

        except Exception as e:
            logger.warning("LangGraph execution failed for query %s: %s", q.query_id, e)
            return {
                "run_id": run_id,
                "config_fingerprint": CONFIG_FINGERPRINT,
                "source_revision": SOURCE_REVISION,
                "claim_id": q.query_id,
                "status": EvaluationCaseStatus.PROVIDER_FAILED.value,
                "execution_mode": "REAL_9ROUTER_GEMINI_LANGGRAPH",
                "provider": "9router",
                "requested_model": "ag/gemini-3.8-flash-high",
                "error": str(e),
                "workflow_latency_seconds": time.time() - start_time,
                "provider_requests": collected_requests,
            }

    start_total = time.time()
    with open(checkpoint_file, "a", encoding="utf-8") as ck_f:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_qid = {executor.submit(process_claim_langgraph, q): q.query_id for q in remaining_queries}
            done_count = 0
            for future in as_completed(future_to_qid):
                res = future.result()
                qid = res["claim_id"]
                completed_map[qid] = res
                ck_f.write(json.dumps(res) + "\n")
                ck_f.flush()
                done_count += 1
                if done_count % 15 == 0 or done_count == len(remaining_queries):
                    logger.info("LangGraph progress: %d / %d completed in %.1fs...", done_count, len(remaining_queries), time.time() - start_total)

    # Reconstruct records and compute summary (Step 8, Step 10)
    records = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_provider_requests_count = 0
    retries_triggered = 0

    for q in query_list:
        rec = completed_map.get(q.query_id)
        if rec and rec.get("status") == EvaluationCaseStatus.COMPLETED.value and "analysis_dict" in rec:
            an_obj = GroundedClaimAnalysis.model_validate(rec["analysis_dict"])
            records.append({"status": EvaluationCaseStatus.COMPLETED.value, "analysis": an_obj})
            reqs = rec.get("provider_requests", [])
            if reqs:
                total_prompt_tokens += sum((req.get("prompt_tokens") or 0) for req in reqs)
                total_completion_tokens += sum((req.get("completion_tokens") or 0) for req in reqs)
                total_tokens += sum((req.get("total_tokens") or 0) for req in reqs)
                total_provider_requests_count += len(reqs)
            elif an_obj.model_metadata:
                total_prompt_tokens += an_obj.model_metadata.prompt_tokens or 0
                total_completion_tokens += an_obj.model_metadata.completion_tokens or 0
                total_tokens += an_obj.model_metadata.total_tokens or 0
                total_provider_requests_count += an_obj.model_metadata.provider_requests_count or 1
            if rec.get("retry_count", 0) > 0:
                retries_triggered += 1
        elif rec and rec.get("status") == EvaluationCaseStatus.PROVIDER_FAILED.value:
            records.append({"status": EvaluationCaseStatus.PROVIDER_FAILED.value, "analysis": None})
            reqs = rec.get("provider_requests", [])
            total_provider_requests_count += len(reqs)
            total_prompt_tokens += sum((req.get("prompt_tokens") or 0) for req in reqs)
            total_completion_tokens += sum((req.get("completion_tokens") or 0) for req in reqs)
            total_tokens += sum((req.get("total_tokens") or 0) for req in reqs)
        else:
            records.append({"status": EvaluationCaseStatus.SCHEMA_INVALID.value, "analysis": None})

    summary = evaluate_grounded_analyses(records, evaluation_mode="REAL_9ROUTER_GEMINI_LANGGRAPH")

    out_file = ARTIFACTS_DIR / "benchmarks" / summary_filename
    result_data = {
        "run_id": run_id,
        "config_fingerprint": CONFIG_FINGERPRINT,
        "source_revision": SOURCE_REVISION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_mode": "REAL_9ROUTER_GEMINI_LANGGRAPH",
        "provider": "9router",
        "model": "ag/gemini-3.8-flash-high",
        "total_attempted": summary.total_attempted,
        "total_completed": summary.total_completed,
        "schema_valid_rate": summary.schema_valid_rate,
        "provider_failure_rate": summary.provider_failure_rate,
        "total_citations": summary.total_citations_generated,
        "valid_citations_count": summary.valid_citations_count,
        "citation_validity_rate": summary.citation_validity_rate,
        "unsupported_citation_rate": summary.unsupported_citation_rate,
        "gold_doc_aligned_count": summary.gold_doc_aligned_count,
        "gold_doc_overlap_rate": summary.gold_doc_overlap_rate,
        "gold_sentence_evaluable_count": summary.gold_sentence_evaluable_count,
        "gold_sentence_aligned_count": summary.gold_sentence_aligned_count,
        "gold_sentence_overlap_rate": summary.gold_sentence_overlap_rate,
        "gold_sentence_precision": summary.gold_sentence_precision,
        "nli_entailed_citations": summary.nli_entailed_count,
        "nli_contradicted_citations": summary.nli_contradicted_count,
        "abstention_rate": summary.abstention_rate,
        "retries_triggered_count": retries_triggered,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_provider_requests": total_provider_requests_count,
        "cost_status": "UNKNOWN",
        "cost_note": "Gateway 9Router does not expose billing rates; paid inference was authorized by the operator without a spending cap",
    }

    with open(out_file, "w") as f:
        json.dump(result_data, f, indent=2)

    logger.info("Saved LangGraph 300 benchmark summary to %s", out_file)
    return result_data


if __name__ == "__main__":
    run_full_langgraph_benchmark_300()
