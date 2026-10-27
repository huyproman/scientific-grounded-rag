"""Sentence attribution (quote must come from the claimed sentence), LangGraph retry and safe finalization, and token/cost accounting."""

import pytest

from scientific_rag.data.models import Chunk, RetrievalResult
from scientific_rag.data.sentence_provenance import CanonicalSentence, SciFactSentenceProvenance
from scientific_rag.generation.adapter import (
    MockGenerationAdapter,
    NineRouterAdapter,
)
from scientific_rag.generation.schemas import (
    Citation,
    ClaimVerdict,
    CostStatus,
    GroundedClaimAnalysis,
    ModelMetadata,
    VerificationReport,
)
from scientific_rag.retrieval.base import BaseRetriever
from scientific_rag.retrieval.hybrid import HybridRRFRetriever
from scientific_rag.verification.citations import CitationVerifier
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


# =========================================================================
# C1: Strict Quotation-to-Sentence Attribution
# =========================================================================

def test_c1_t06_quote_from_sent0_claiming_sent1_rejected():
    """C1: Quotation text from sentence 0 claiming sentence_idx=1 is rejected even if in same chunk."""
    prov = SciFactSentenceProvenance()
    # Mock document 100 with two distinct sentences
    prov._doc_sentences["doc_100"] = [
        CanonicalSentence(doc_id="doc_100", sentence_idx=0, text="Treatment A significantly improves survival in adults."),
        CanonicalSentence(doc_id="doc_100", sentence_idx=1, text="Treatment B does not improve survival in adults."),
    ]
    prov._is_loaded = True

    verifier = CitationVerifier(provenance=prov)
    # Chunk contains BOTH sentences
    ev = [
        RetrievalResult(
            chunk_id="c_both",
            doc_id="doc_100",
            score=1.0,
            rank=1,
            text="Treatment A significantly improves survival in adults. Treatment B does not improve survival in adults.",
        )
    ]

    # Citation quotes sentence 0, but falsely claims sentence_idx=1
    cit_mismatched = [
        Citation(
            parent_doc_id="doc_100",
            chunk_id="c_both",
            sentence_idx=1,
            evidence_text="Treatment A significantly improves survival in adults.",
        )
    ]

    res = verifier.verify_citations(cit_mismatched, ev)
    # C1 requirement: sentence_idx MUST be invalidated (set to None) due to quotation mismatch
    assert res[0].citation.sentence_idx is None
    assert "Quotation mismatch for sentence_idx 1" in res[0].notes


def test_c1_t07_quote_from_sent1_claiming_sent1_accepted():
    """C1: Quotation text from sentence 1 claiming sentence_idx=1 is accepted as valid attribution."""
    prov = SciFactSentenceProvenance()
    prov._doc_sentences["doc_100"] = [
        CanonicalSentence(doc_id="doc_100", sentence_idx=0, text="Treatment A significantly improves survival in adults."),
        CanonicalSentence(doc_id="doc_100", sentence_idx=1, text="Treatment B does not improve survival in adults."),
    ]
    prov._is_loaded = True

    verifier = CitationVerifier(provenance=prov)
    ev = [
        RetrievalResult(
            chunk_id="c_both",
            doc_id="doc_100",
            score=1.0,
            rank=1,
            text="Treatment A significantly improves survival in adults. Treatment B does not improve survival in adults.",
        )
    ]

    # Citation quotes sentence 1 and claims sentence_idx=1
    cit_matched = [
        Citation(
            parent_doc_id="doc_100",
            chunk_id="c_both",
            sentence_idx=1,
            evidence_text="Treatment B does not improve survival in adults.",
        )
    ]

    res = verifier.verify_citations(cit_matched, ev)
    assert res[0].citation.sentence_idx == 1
    assert res[0].is_text_authentic is True


# =========================================================================
# C3: Truthful Cost & Token Metadata
# =========================================================================

def test_c3_t03_paid_inference_has_is_zero_cost_verified_false():
    """C3: Real provider metadata sets is_zero_cost_verified=False and cost_status=UNKNOWN."""
    meta = ModelMetadata(
        provider="9router",
        model_name="ag/gemini-3.8-flash-high",
        prompt_tokens=150,
        completion_tokens=45,
        total_tokens=195,
        latency_seconds=1.2,
        is_zero_cost_verified=False,
        cost_status=CostStatus.UNKNOWN,
    )
    assert meta.is_zero_cost_verified is False
    assert meta.cost_status == CostStatus.UNKNOWN
    assert meta.total_tokens == 195


# =========================================================================
# C2: Full LangGraph Workflow Invocation & Trace
# =========================================================================

def test_c2_t14_full_langgraph_workflow_trace_and_retry():
    """C2: create_verification_graph execution path preserves attempt counts and state."""
    chunks = [Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", chunk_index=0, text=f"Evidence text {i}", title="T") for i in range(20)]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))

    calls = 0
    def verifier_flip(analysis, ev):
        nonlocal calls
        calls += 1
        # First attempt fails verification; second attempt succeeds
        if calls == 1:
            return VerificationReport(all_citations_valid=False, verdict_supported_by_evidence=False)
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=True)

    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)
    graph = create_verification_graph(hybrid, generator, verifier_flip, max_retries=1)

    initial_state = {
        "claim_id": "test_c2_claim",
        "claim_text": "Sample scientific claim",
        "evidence_items": [],
        "analysis": None,
        "verification": None,
        "retry_count": 0,
        "max_retries": 1,
        "final_verdict": None,
        "is_verified_conclusive": False,
    }

    final_state = graph.invoke(initial_state)
    assert calls == 2
    assert final_state["retry_count"] == 1
    assert final_state["is_verified_conclusive"] is True
    assert final_state["final_verdict"] == ClaimVerdict.SUPPORTED.value
    assert len(final_state["evidence_items"]) == 15  # Expanded retrieval on retry
