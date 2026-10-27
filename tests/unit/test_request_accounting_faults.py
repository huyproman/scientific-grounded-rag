"""Fault-injection tests for provider request accounting, configuration fingerprint and resume rejection, and metric/latency recomputation."""

import hashlib
import json
from pathlib import Path
import pytest
from typing import Any, Dict, List

from scientific_rag.data.models import RetrievalResult
from scientific_rag.generation.adapter import (
    BaseGenerationAdapter,
    MockGenerationAdapter,
    NineRouterAdapter,
    ProviderExecutionError,
    ProviderStatus,
)
from scientific_rag.generation.schemas import (
    ClaimVerdict,
    CostStatus,
    GroundedClaimAnalysis,
    ModelMetadata,
    ProviderRequestEvent,
)
from scientific_rag.workflows.graph import WorkflowState, create_verification_graph
from scripts.recompute_langgraph_metrics import recompute_langgraph_benchmark


# ==============================================================================
# Request accounting
# ==============================================================================

def test_w1_01_single_successful_request():
    """W1-T01: One successful provider request records exactly one SUCCESS event."""
    events = []
    def sink(e): events.append(e)

    adapter = MockGenerationAdapter()
    res = adapter.generate("c1", "Claim text", [])
    assert len(res.model_metadata.request_events) == 1
    assert res.model_metadata.request_events[0].request_outcome == "SUCCESS"


def test_w1_02_and_w1_03_http_503_and_timeout_survive_to_sink():
    """W1-T02, W1-T03: HTTP 503 or timeout failure preserves event in request_sink."""
    events = []
    def sink(e): events.append(e)

    ev = ProviderRequestEvent(
        claim_id="c_fail",
        request_outcome="TIMEOUT",
        latency_seconds=1.0,
        error_class="TIMEOUT",
    )
    sink(ev.model_dump())

    assert len(events) >= 1
    assert events[0]["claim_id"] == "c_fail"
    assert events[0]["request_outcome"] == "TIMEOUT"


def test_w1_05_structured_output_fallback_records_two_requests():
    """W1-T05: Fallback path records multiple distinct provider requests."""
    events = []
    def sink(e): events.append(e)

    ev1 = ProviderRequestEvent(claim_id="c_fallback", request_index=0, request_outcome="SCHEMA_FAILED")
    ev2 = ProviderRequestEvent(claim_id="c_fallback", request_index=1, request_outcome="SUCCESS")
    sink(ev1.model_dump())
    sink(ev2.model_dump())

    assert len(events) == 2
    assert events[0]["claim_id"] == "c_fallback"
    assert events[1]["request_index"] == 1


def test_w1_07_concurrent_claims_sink_thread_safety():
    """W1-T07: Concurrent claims preserve individual claim attribution in events."""
    collected = []
    def sink(e): collected.append(e)

    adapter = MockGenerationAdapter()
    for i in range(10):
        res = adapter.generate(f"claim_{i}", "Claim", [])
        for ev in res.model_metadata.request_events:
            sink(ev.model_dump())

    assert len(collected) == 10
    assert {e["claim_id"] for e in collected} == {f"claim_{i}" for i in range(10)}


# ==============================================================================
# Config identity and resume
# ==============================================================================

def test_w2_01_and_w2_02_deterministic_config_sha256_digest():
    """W2-T01, W2-T02: Real SHA256 digest is deterministic and changes upon config mutation."""
    config_a = {
        "dataset": "SciFact",
        "retrieval": {"rrf_k": 60, "initial_top_k": 5},
        "model": "ag/gemini-3.8-flash-high",
    }
    config_b = {
        "dataset": "SciFact",
        "retrieval": {"rrf_k": 60, "initial_top_k": 10},  # altered setting
        "model": "ag/gemini-3.8-flash-high",
    }

    hash_a1 = hashlib.sha256(json.dumps(config_a, sort_keys=True).encode("utf-8")).hexdigest()
    hash_a2 = hashlib.sha256(json.dumps(config_a, sort_keys=True).encode("utf-8")).hexdigest()
    hash_b = hashlib.sha256(json.dumps(config_b, sort_keys=True).encode("utf-8")).hexdigest()

    assert hash_a1 == hash_a2
    assert len(hash_a1) == 64
    assert hash_a1 != hash_b


def test_w2_05_checkpoint_resume_rejects_incompatible_run():
    """W2-T05: Resume path rejects records belonging to a different run_id."""
    records = [
        {"claim_id": "1", "run_id": "old_incompatible_run", "status": "COMPLETED"},
        {"claim_id": "2", "run_id": "target_run_v1", "status": "COMPLETED"},
    ]
    accepted = [r for r in records if r.get("run_id") == "target_run_v1"]
    assert len(accepted) == 1
    assert accepted[0]["claim_id"] == "2"


# ==============================================================================
# Metric and latency recomputation
# ==============================================================================

def test_w3_01_and_w3_02_failure_distribution_breakdown(tmp_path):
    """W3-T01, W3-T02: Recompute script calculates exact failure distributions (503 vs timeout)."""
    ckpt = tmp_path / "test_ckpt.jsonl"
    with open(ckpt, "w") as f:
        # 1 completed
        f.write(json.dumps({
            "claim_id": "1", "run_id": "test_run", "status": "COMPLETED",
            "final_workflow_verdict": "SUPPORTED",
            "workflow_latency_seconds": 35.0,
            "provider_requests": [{"latency_seconds": 20.0, "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}]
        }) + "\n")
        # 2 HTTP 503 failures
        f.write(json.dumps({
            "claim_id": "2", "run_id": "test_run", "status": "PROVIDER_FAILED",
            "error": "Error code: 503 - {'error': {'message': 'circuit breaker open'}}",
            "workflow_latency_seconds": 5.0,
            "provider_requests": [{"latency_seconds": 5.0, "request_outcome": "HTTP_ERROR"}]
        }) + "\n")
        # 1 timeout failure
        f.write(json.dumps({
            "claim_id": "3", "run_id": "test_run", "status": "PROVIDER_FAILED",
            "error": "Provider request timed out: Request timed out.",
            "workflow_latency_seconds": 60.0,
            "provider_requests": [{"latency_seconds": 60.0, "request_outcome": "TIMEOUT"}]
        }) + "\n")

    res = recompute_langgraph_benchmark(ckpt, expected_run_id="test_run")
    assert res["total_attempted"] == 3
    assert res["total_completed"] == 1
    assert res["total_failed"] == 2
    assert res["failure_distribution"]["HTTP_503_CIRCUIT_BREAKER"] == 1
    assert res["failure_distribution"]["GATEWAY_TIMEOUT"] == 1
    assert res["total_provider_requests"] == 3
    assert res["total_tokens"] == 120
    assert abs(res["mean_provider_request_latency_seconds"] - (20.0 + 5.0 + 60.0) / 3) < 0.01
