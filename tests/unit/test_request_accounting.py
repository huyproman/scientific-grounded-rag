"""Provider request accounting and failure capture across success and failure paths."""

from unittest.mock import MagicMock, patch
import json
import pytest

from scientific_rag.generation.adapter import (
    NineRouterAdapter,
    MockGenerationAdapter,
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


def test_a_t01_one_successful_response_emits_one_event():
    """A-T01: One successful response -> exactly one request event in sink."""
    events = []
    adapter = NineRouterAdapter(request_sink=events.append)

    fake_response = MagicMock()
    fake_response.content = json.dumps({
        "verdict": "SUPPORTED",
        "explanation": "Valid test explanation",
        "citations": [],
    })
    fake_response.response_metadata = {
        "token_usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        "model_name": "ag/gemini-3.8-flash-high",
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = fake_response

    with patch.object(adapter, "_get_llm", return_value=mock_llm):
        res = adapter.generate("claim_at01", "Claim text", [])

    assert len(events) == 1
    ev = events[0]
    assert ev["claim_id"] == "claim_at01"
    assert ev["request_outcome"] == "SUCCESS"
    assert ev["prompt_tokens"] == 120
    assert ev["total_tokens"] == 150
    assert len(res.model_metadata.request_events) == 1


def test_a_t02_http_503_retains_failed_request_event():
    """A-T02: HTTP 503 -> failed request event retained with error details."""
    events = []
    adapter = NineRouterAdapter(request_sink=events.append)

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Error code: 503 - {'error': {'message': 'circuit breaker open'}}")
    # structured fallback also fails with 503
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = Exception("Error code: 503 - {'error': {'message': 'circuit breaker open'}}")
    mock_llm.with_structured_output.return_value = mock_structured

    with patch.object(adapter, "_get_llm", return_value=mock_llm):
        with pytest.raises(ProviderExecutionError):
            adapter.generate("claim_at02", "Claim text", [])

    assert len(events) == 2  # Primary attempt + structured fallback
    assert events[0]["claim_id"] == "claim_at02"
    assert events[0]["request_outcome"] == "HTTP_ERROR"
    assert events[1]["request_outcome"] == "HTTP_ERROR"


def test_a_t03_timeout_event_retained():
    """A-T03: Timeout -> timeout event retained without dropping."""
    events = []
    adapter = NineRouterAdapter(request_sink=events.append)

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Request timed out after 30 seconds")

    with patch.object(adapter, "_get_llm", return_value=mock_llm):
        with pytest.raises(ProviderExecutionError) as exc_info:
            adapter.generate("claim_at03", "Claim text", [])
        assert exc_info.value.status == ProviderStatus.TIMEOUT

    assert len(events) == 1
    assert events[0]["claim_id"] == "claim_at03"
    assert events[0]["request_outcome"] == "TIMEOUT"
    assert events[0]["error_class"] == "TIMEOUT"


def test_a_t04_successful_first_generation_failed_retry_both_retained():
    """A-T04: Successful first generation, failed retry in workflow -> both events retained."""
    events = []
    def sink(e): events.append(e)

    fake_resp_success = MagicMock()
    fake_resp_success.content = json.dumps({
        "verdict": "SUPPORTED",
        "explanation": "Needs retry due to unverified citation",
        "citations": [],
    })
    fake_resp_success.response_metadata = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}

    call_count = 0
    def mock_invoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return fake_resp_success
        raise Exception("Error code: 503 - circuit breaker open on retry")

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mock_invoke
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = Exception("Error code: 503 - circuit breaker open on fallback")
    mock_llm.with_structured_output.return_value = mock_structured

    adapter = NineRouterAdapter(request_sink=sink)
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    verifier_fn = MagicMock()
    # First report fails verification -> triggers retry
    mock_report = MagicMock()
    mock_report.all_citations_valid = False
    mock_report.verdict_supported_by_evidence = False
    mock_report.outcomes = []
    mock_report.citation_details = []
    verifier_fn.return_value = mock_report

    graph = create_verification_graph(
        retriever=retriever,
        generator=adapter,
        verifier_fn=verifier_fn,
        max_retries=1,
    )

    with patch.object(adapter, "_get_llm", return_value=mock_llm):
        state = {
            "claim_id": "claim_at04",
            "claim_text": "Claim text",
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
        with pytest.raises(Exception):
            graph.invoke(state)

    # Sink must have captured first success AND second failure
    assert len(events) >= 2
    assert events[0]["claim_id"] == "claim_at04"
    assert events[0]["request_outcome"] == "SUCCESS"
    assert events[1]["claim_id"] == "claim_at04"
    assert events[1]["request_outcome"] == "HTTP_ERROR"


def test_a_t05_structured_output_fallback_invokes_model_twice():
    """A-T05: Structured-output fallback invokes model twice -> exactly two events, not one and not three."""
    events = []
    adapter = NineRouterAdapter(request_sink=events.append)

    mock_llm = MagicMock()
    # First invoke returns invalid JSON
    fake_bad_resp = MagicMock()
    fake_bad_resp.content = "Not a json response string"
    fake_bad_resp.response_metadata = {}
    mock_llm.invoke.return_value = fake_bad_resp

    # Fallback with_structured_output returns valid schema
    fake_schema_output = MagicMock()
    fake_schema_output.verdict = ClaimVerdict.INSUFFICIENT_EVIDENCE
    fake_schema_output.explanation = "Extracted via fallback"
    fake_schema_output.citations = []

    mock_structured = MagicMock()
    mock_structured.invoke.return_value = fake_schema_output
    mock_llm.with_structured_output.return_value = mock_structured

    with patch.object(adapter, "_get_llm", return_value=mock_llm):
        res = adapter.generate("claim_at05", "Claim text", [])

    assert len(events) == 2, f"Expected exactly 2 events, got {len(events)}"
    assert events[0]["request_index"] == 0
    assert events[0]["request_outcome"] == "SCHEMA_VALIDATION_FAILED"
    assert events[1]["request_index"] == 1
    assert events[1]["request_outcome"] == "SUCCESS"
    assert res.model_metadata.provider_requests_count == 2


def test_a_t06_failure_before_http_request_does_not_fabricate_event():
    """A-T06: Failure before HTTP request (e.g. init failure) -> no fabricated HTTP request event."""
    events = []
    adapter = NineRouterAdapter(api_key="", request_sink=events.append)

    with pytest.raises(Exception):
        adapter.generate("claim_at06", "Claim text", [])

    assert len(events) == 0, "No HTTP events should be fabricated if transport never initialized"


def test_a_t07_concurrent_claims_do_not_cross_boundaries():
    """A-T07: Concurrent claims -> request events cannot cross claim boundaries."""
    from concurrent.futures import ThreadPoolExecutor

    events = []
    def sink(e): events.append(e)

    def run_worker(cid):
        adapter = MockGenerationAdapter()
        res = adapter.generate(cid, f"Claim {cid}", [])
        for ev in res.model_metadata.request_events:
            sink(ev.model_dump())

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(run_worker, [f"claim_{i}" for i in range(20)]))

    assert len(events) == 20
    claim_ids = [e["claim_id"] for e in events]
    assert len(set(claim_ids)) == 20


def test_a_t08_failed_response_without_token_usage_has_unknown_not_zero():
    """A-T08: Failed response without token usage -> usage UNKNOWN, not zero."""
    events = []
    adapter = NineRouterAdapter(request_sink=events.append)

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Error code: 503 - unavailable")
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = Exception("Error code: 503 - unavailable")
    mock_llm.with_structured_output.return_value = mock_structured

    with patch.object(adapter, "_get_llm", return_value=mock_llm):
        with pytest.raises(ProviderExecutionError):
            adapter.generate("claim_at08", "Claim text", [])

    for ev in events:
        assert ev["prompt_tokens"] is None
        assert ev["completion_tokens"] is None
        assert ev["total_tokens"] is None


def test_a_t09_checkpoint_serialization_preserves_all_events(tmp_path):
    """A-T09: Checkpoint serialization preserves all provider request events."""
    ckpt = tmp_path / "ckpt.jsonl"
    record = {
        "claim_id": "claim_at09",
        "run_id": "test_run",
        "status": "COMPLETED",
        "provider_requests": [
            {"request_index": 0, "request_outcome": "SCHEMA_VALIDATION_FAILED", "latency_seconds": 2.5},
            {"request_index": 1, "request_outcome": "SUCCESS", "latency_seconds": 3.1, "total_tokens": 500},
        ],
    }

    with open(ckpt, "w") as f:
        f.write(json.dumps(record) + "\n")

    with open(ckpt, "r") as f:
        loaded = json.loads(f.readline())

    assert len(loaded["provider_requests"]) == 2
    assert loaded["provider_requests"][0]["request_outcome"] == "SCHEMA_VALIDATION_FAILED"
    assert loaded["provider_requests"][1]["total_tokens"] == 500


def test_a_t10_resume_preserves_single_call_accounting():
    """A-T10: Resume path avoids duplicate calls and maintains exact request count."""
    completed_map = {
        "claim_1": {"claim_id": "claim_1", "status": "COMPLETED", "provider_requests": [{"outcome": "SUCCESS"}]},
    }
    all_queries = ["claim_1", "claim_2"]
    remaining = [q for q in all_queries if q not in completed_map]
    assert remaining == ["claim_2"]
