"""
Independent verification and recomputation script for the final LangGraph benchmark.
Recomputes all key metrics from the raw JSONL checkpoint records without relying on summary files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def recompute_langgraph_benchmark(
    checkpoint_path: Path,
    expected_run_id: Optional[str] = None,
    expected_config_sha: Optional[str] = None,
):
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Filter for expected run_id and config_fingerprint if specified 
    if expected_run_id:
        records = [r for r in records if r.get("run_id") == expected_run_id]
    if expected_config_sha:
        records = [r for r in records if r.get("config_fingerprint") == expected_config_sha]

    total_attempted = len(records)
    completed = [r for r in records if r.get("status") == "COMPLETED"]
    failed = [r for r in records if r.get("status") == "PROVIDER_FAILED"]

    total_completed = len(completed)
    total_failed = len(failed)

    failure_distribution = {}
    for r in failed:
        err = str(r.get("error", "unknown"))
        if "503" in err or "circuit" in err.lower():
            k = "HTTP_503_CIRCUIT_BREAKER"
        elif "timeout" in err.lower() or "timed out" in err.lower():
            k = "GATEWAY_TIMEOUT"
        else:
            k = "OTHER_FAILURE"
        failure_distribution[k] = failure_distribution.get(k, 0) + 1

    proposed_counts = {}
    final_counts = {}
    retries_count = 0

    total_citations = 0
    valid_citations = 0
    gold_doc_matches = 0
    gold_sentence_evaluable = 0
    gold_sentence_matches = 0
    consistent_citations = 0
    unsupported_citations = 0

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_provider_requests = 0

    provider_latencies = []
    completed_workflow_latencies = []
    failed_workflow_latencies = []

    for r in records:
        w_lat = r.get("workflow_latency_seconds")
        if w_lat is not None:
            if r.get("status") == "COMPLETED":
                completed_workflow_latencies.append(w_lat)
            else:
                failed_workflow_latencies.append(w_lat)

        reqs = r.get("provider_requests", [])
        for req in reqs:
            p_lat = req.get("latency_seconds")
            if p_lat is not None:
                provider_latencies.append(p_lat)

    for r in completed:
        pv = r.get("proposed_verdict")
        fv = r.get("final_workflow_verdict")
        proposed_counts[pv] = proposed_counts.get(pv, 0) + 1
        final_counts[fv] = final_counts.get(fv, 0) + 1

        if r.get("retry_count", 0) > 0:
            retries_count += 1

        ad = r.get("analysis_dict", {})
        cits = ad.get("citations", [])
        total_citations += len(cits)

        vf = ad.get("verification", {})
        valid_citations += vf.get("valid_citations", 0)
        gold_doc_matches += vf.get("gold_aligned_citations", 0)

        for cd in vf.get("citation_details", []):
            if cd.get("gold_sentence_aligned") is not None:
                gold_sentence_evaluable += 1
                if cd.get("gold_sentence_aligned") is True:
                    gold_sentence_matches += 1

            # Verdict-aware consistency
            nli_v = cd.get("nli_verdict")
            if pv == "SUPPORTED" and nli_v == "ENTAILMENT":
                consistent_citations += 1
            elif pv == "CONTRADICTED" and nli_v == "CONTRADICTION":
                consistent_citations += 1
            else:
                unsupported_citations += 1

    # Read provider requests from ALL records (both completed and failed)
    for r in records:
        reqs = r.get("provider_requests", [])
        if reqs:
            total_provider_requests += len(reqs)
            p_tok = sum((req.get("prompt_tokens") or 0) for req in reqs)
            c_tok = sum((req.get("completion_tokens") or 0) for req in reqs)
            t_tok = sum((req.get("total_tokens") or 0) for req in reqs)
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_tokens += t_tok
        elif r.get("status") == "COMPLETED":
            mm = r.get("analysis_dict", {}).get("model_metadata", {})
            total_prompt_tokens += mm.get("prompt_tokens") or 0
            total_completion_tokens += mm.get("completion_tokens") or 0
            total_tokens += mm.get("total_tokens") or 0
            total_provider_requests += mm.get("provider_requests_count") or (2 if r.get("retry_count", 0) > 0 else 1)

    mean_prov_lat = (sum(provider_latencies) / len(provider_latencies)) if provider_latencies else 0.0
    mean_comp_wf_lat = (sum(completed_workflow_latencies) / len(completed_workflow_latencies)) if completed_workflow_latencies else 0.0
    mean_fail_wf_lat = (sum(failed_workflow_latencies) / len(failed_workflow_latencies)) if failed_workflow_latencies else 0.0

    return {
        "run_id": expected_run_id or (records[0].get("run_id") if records else "UNKNOWN"),
        "total_attempted": total_attempted,
        "total_completed": total_completed,
        "total_failed": total_failed,
        "failure_distribution": failure_distribution,
        "schema_valid_rate": total_completed / total_attempted if total_attempted else 0.0,
        "provider_failure_rate": total_failed / total_attempted if total_attempted else 0.0,
        "proposed_verdicts": proposed_counts,
        "final_workflow_verdicts": final_counts,
        "retries_count": retries_count,
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "citation_validity_rate": valid_citations / max(total_citations, 1),
        "semantically_consistent_citations": consistent_citations,
        "unsupported_citations_count": unsupported_citations,
        "unsupported_citation_rate": unsupported_citations / max(total_citations, 1),
        "gold_doc_matches": gold_doc_matches,
        "gold_doc_match_rate": gold_doc_matches / max(total_citations, 1),
        "gold_sentence_evaluable": gold_sentence_evaluable,
        "gold_sentence_matches": gold_sentence_matches,
        "gold_sentence_match_rate": gold_sentence_matches / max(gold_sentence_evaluable, 1),
        "gold_sentence_precision": gold_sentence_matches / max(gold_sentence_evaluable, 1),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_provider_requests": total_provider_requests,
        "mean_provider_request_latency_seconds": mean_prov_lat,
        "mean_completed_workflow_latency_seconds": mean_comp_wf_lat,
        "mean_failed_workflow_latency_seconds": mean_fail_wf_lat,
        "cost_status": "UNKNOWN",
    }


if __name__ == "__main__":
    ckpt = Path("artifacts/benchmarks/r2_langgraph_300_checkpoint.jsonl")
    res = recompute_langgraph_benchmark(ckpt)
    print(json.dumps(res, indent=2))
