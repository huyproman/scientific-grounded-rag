"""
Unit tests for LangChain generation schemas, adapter, prompts, and LangGraph workflow.
"""

from pathlib import Path

import pytest

from scientific_rag.data.models import Chunk, RetrievalResult
from scientific_rag.generation.adapter import (
    GenerationOutputSchema,
    MockGenerationAdapter,
    NineRouterAdapter,
    NineRouterPreflightError,
)
from scientific_rag.generation.prompts import create_grounding_prompt, format_evidence_context
from scientific_rag.generation.schemas import (
    Citation,
    ClaimVerdict,
    GroundedClaimAnalysis,
    ModelMetadata,
    VerificationOutcome,
    VerificationReport,
)
from scientific_rag.retrieval.base import BaseRetriever
from scientific_rag.retrieval.hybrid import HybridRRFRetriever
from scientific_rag.workflows.graph import WorkflowState, create_verification_graph


class DummyRetriever(BaseRetriever):
    def __init__(self, name: str, chunks: list):
        super().__init__(name=name)
        self.chunks = chunks

    def index(self, chunks): pass
    def retrieve(self, query: str, top_k: int = 10):
        return [
            RetrievalResult(chunk_id=c.chunk_id, doc_id=c.doc_id, score=1.0, rank=i+1, text=c.text, title=c.title)
            for i, c in enumerate(self.chunks[:top_k])
        ]
    def save(self, path): pass
    @classmethod
    def load(cls, path): pass


def test_t01_structured_output_schema_parsing():
    """T01: Verifies Pydantic schemas parse valid and reject invalid JSON/enums."""
    valid_data = {
        "verdict": "SUPPORTED",
        "explanation": "Evidence clearly confirms the biological mechanism.",
        "citations": [
            {
                "parent_doc_id": "doc123",
                "chunk_id": "chunk_456",
                "evidence_text": "Study demonstrates positive correlation.",
                "relationship": "supports",
            }
        ],
    }
    obj = GenerationOutputSchema.model_validate(valid_data)
    assert obj.verdict == ClaimVerdict.SUPPORTED
    assert len(obj.citations) == 1
    assert obj.citations[0].parent_doc_id == "doc123"

    # Reject invalid verdict enum
    with pytest.raises(Exception):
        GenerationOutputSchema.model_validate({**valid_data, "verdict": "INVALID_VERDICT"})


def test_t02_mock_adapter_deterministic_generation():
    """T02: Verifies Mock adapter produces predictable, schema-valid analysis."""
    adapter = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)
    evidence = [
        RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Sample scientific text 1"),
        RetrievalResult(chunk_id="c2", doc_id="d2", score=0.9, rank=2, text="Sample scientific text 2"),
    ]
    analysis = adapter.generate(claim_id="claim_01", claim_text="Test claim", evidence=evidence)

    assert analysis.claim_id == "claim_01"
    assert analysis.verdict == ClaimVerdict.SUPPORTED
    assert len(analysis.citations) == 2
    assert analysis.citations[0].parent_doc_id == "d1"
    assert analysis.citations[0].chunk_id == "c1"
    assert analysis.model_metadata.is_zero_cost_verified is True


def test_t03_mock_adapter_insufficient_evidence():
    """T03: Verifies mock adapter properly abstains when evidence is empty."""
    adapter = MockGenerationAdapter()
    analysis = adapter.generate(claim_id="claim_02", claim_text="Test claim", evidence=[])

    assert analysis.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
    assert analysis.citations == []


def test_t04_prompt_formatting_and_grounding_template():
    """T04: Verifies prompt formatting includes all evidence items with IDs."""
    evidence = [
        RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Text one", title="Title One"),
        RetrievalResult(chunk_id="c2", doc_id="d2", score=0.8, rank=2, text="Text two", title="Title Two"),
    ]
    formatted = format_evidence_context(evidence)
    assert "Document ID: d1" in formatted
    assert "Chunk ID: c1" in formatted
    assert "Document ID: d2" in formatted
    assert "Chunk ID: c2" in formatted

    prompt = create_grounding_prompt()
    messages = prompt.format_messages(
        claim_id="c1",
        claim_text="my claim",
        evidence_context=formatted,
    )
    assert len(messages) == 2
    assert "INSUFFICIENT_EVIDENCE" in messages[0].content


def test_t05_9router_preflight_missing_key_rejection():
    """T05: Verifies 9Router adapter rejects execution when API key is missing."""
    adapter = NineRouterAdapter(base_url="http://127.0.0.1:59999", api_key="", model_name="ag/gemini-3.8-flash-high")
    with pytest.raises(NineRouterPreflightError, match="Missing LLM_API_KEY"):
        adapter.preflight_check()


def test_t06_langgraph_conditional_workflow_execution():
    """T06: Verifies LangGraph StateGraph executes retrieve -> generate -> verify cycle."""
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="Evidence text 1", title="Doc 1"),
        Chunk(chunk_id="c2", doc_id="d2", chunk_index=0, text="Evidence text 2", title="Doc 2"),
    ]
    r_bm25 = DummyRetriever("bm25", chunks)
    r_dense = DummyRetriever("dense", chunks)
    hybrid = HybridRRFRetriever(r_bm25, r_dense)
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)

    def mock_verifier(analysis, evidence):
        return VerificationReport(
            total_citations=len(analysis.citations),
            valid_citations=len(analysis.citations),
            all_citations_valid=True,
            verdict_supported_by_evidence=True,
            outcomes=[VerificationOutcome.VALID_CITATION],
        )

    graph = create_verification_graph(hybrid, generator, mock_verifier, max_retries=1)
    initial_state = {
        "claim_id": "test_graph_claim",
        "claim_text": "Scientific statement",
        "evidence_items": [],
        "analysis": None,
        "verification": None,
        "retry_count": 0,
        "max_retries": 1,
        "final_verdict": None,
    }

    result = graph.invoke(initial_state)
    assert result["analysis"] is not None
    assert result["verification"] is not None
    assert result["final_verdict"] == "SUPPORTED"
    assert result["retry_count"] == 0


def test_remote_inference_is_opt_in_and_makes_no_request(monkeypatch):
    """With PAID_INFERENCE_AUTHORIZED unset, no network call is attempted."""
    import urllib.request

    from scientific_rag.config import load_config
    from scientific_rag.generation.adapter import NineRouterAdapter, NineRouterPreflightError, ProviderExecutionError

    def fail(*a, **k):
        raise AssertionError("network request attempted")

    monkeypatch.delenv("PAID_INFERENCE_AUTHORIZED", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", fail)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", fail)

    assert load_config(Path('/nonexistent.env')).paid_inference_authorized is False
    adapter = NineRouterAdapter(base_url="http://127.0.0.1:59999", api_key="k")
    assert adapter.paid_inference_authorized is False
    with pytest.raises(NineRouterPreflightError):
        adapter.preflight_check()
    with pytest.raises(ProviderExecutionError):
        adapter.generate("c1", "text", [])

    monkeypatch.setenv("PAID_INFERENCE_AUTHORIZED", "true")
    assert NineRouterAdapter(api_key="k").paid_inference_authorized is True
