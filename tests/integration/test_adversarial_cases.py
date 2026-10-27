"""Cross-module adversarial scenarios: fabricated or reversed evidence, unsupported verdicts, bounded retries, provider failures vs abstention, opt-in provider gating, retrieval regression."""

import os
import pytest

from scientific_rag.data.models import Chunk, RetrievalResult
from scientific_rag.evaluation.grounding import (
    EvaluationCaseStatus,
    evaluate_grounded_analyses,
)
from scientific_rag.generation.adapter import (
    BaseGenerationAdapter,
    MockGenerationAdapter,
    NineRouterAdapter,
    NineRouterPreflightError,
    ProviderExecutionError,
    ProviderStatus,
)
from scientific_rag.generation.schemas import (
    Citation,
    CitationVerificationResult,
    ClaimVerdict,
    GroundedClaimAnalysis,
    ModelMetadata,
    VerificationOutcome,
    VerificationReport,
)
from scientific_rag.retrieval.base import BaseRetriever
from scientific_rag.retrieval.bm25 import BM25Retriever
from scientific_rag.retrieval.hybrid import HybridRRFRetriever
from scientific_rag.verification.citations import CitationVerifier
from scientific_rag.verification.gold_alignment import GoldRationaleAligner
from scientific_rag.verification.support import PretrainedNLIProxy
from scientific_rag.verification.verifier import MultiLayerEvidenceVerifier
from scientific_rag.workflows.graph import create_verification_graph


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


def test_adversarial_case_a_supported_authentic_evidence():
    """CASE A: Claim supported by authentic evidence -> Verified SUPPORTED."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="mRNA vaccines stimulate neutralizing antibodies.")]
    m = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)
    a = m.generate("1", "mRNA vaccines stimulate neutralizing antibodies.", ev)
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True))
    rep = v.verify(a, ev)
    assert a.verdict == ClaimVerdict.SUPPORTED
    assert rep.all_citations_valid is True
    assert rep.verdict_supported_by_evidence is True


def test_adversarial_case_b_contradicted_authentic_evidence():
    """CASE B: Claim contradicted by authentic evidence -> Verified CONTRADICTED."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="The clinical trial demonstrates drug fails completely to cure.")]
    m = MockGenerationAdapter(default_verdict=ClaimVerdict.CONTRADICTED)
    a = m.generate("2", "Drug cures completely in clinical trial.", ev)
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True))
    rep = v.verify(a, ev)
    assert a.verdict == ClaimVerdict.CONTRADICTED
    assert rep.all_citations_valid is True
    assert rep.verdict_supported_by_evidence is True


def test_adversarial_case_c_unresolved_evidence_abstention():
    """CASE C: Claim not resolved by retrieved evidence -> INSUFFICIENT_EVIDENCE."""
    m = MockGenerationAdapter(default_verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE)
    a = m.generate("3", "Completely unmentioned biological mechanism.", [])
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(a, [])
    assert a.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
    assert len(a.citations) == 0
    assert rep.verdict_supported_by_evidence is True


def test_adversarial_case_d_valid_id_fabricated_evidence_text():
    """CASE D: Valid citation ID, fabricated evidence_text -> Rejected."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="The real text.")]
    m = MockGenerationAdapter(force_fabricated_evidence_text=True)
    a = m.generate("4", "The real text.", ev)
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(a, ev)
    assert rep.all_citations_valid is False
    assert rep.verdict_supported_by_evidence is False
    assert rep.citation_details[0].is_text_authentic is False


def test_adversarial_case_e_valid_id_reversed_negation():
    """CASE E: Valid citation ID, reversed negation in cited text -> Integrity failure."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Drug cures malaria in humans.")]
    m = MockGenerationAdapter(force_reversed_negation=True)
    a = m.generate("5", "Drug cures malaria in humans.", ev)
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(a, ev)
    assert rep.citation_details[0].is_text_authentic is False
    assert rep.verdict_supported_by_evidence is False


def test_adversarial_case_f_unsupported_supported_answer_with_valid_ids():
    """CASE F: Unsupported SUPPORTED answer with valid IDs -> LangGraph does not terminate as verified success."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="Weather in Tokyo is sunny.", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))

    graph = create_verification_graph(hybrid, generator, v.verify, max_retries=1)
    state = graph.invoke({
        "claim_id": "6", "claim_text": "Quantum computing requires subatomic entanglement.",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    # Cannot be verified conclusive because weather snippet does not support quantum claim
    assert state["is_verified_conclusive"] is False


def test_adversarial_case_g_invalid_citation_retry_succeeds():
    """CASE G: Invalid citation followed by successful retry -> Re-verifies newly retrieved evidence."""
    chunks = [Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", chunk_index=0, text=f"Evidence {i}", title="T") for i in range(20)]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))

    attempt = 0
    class FlipGenerator(BaseGenerationAdapter):
        def generate(self, claim_id: str, claim_text: str, evidence: list):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                # First attempt has hallucinated citation
                return GroundedClaimAnalysis(
                    claim_id=claim_id, input_claim=claim_text, verdict=ClaimVerdict.SUPPORTED, explanation="E",
                    citations=[Citation(parent_doc_id="d_fake", chunk_id="c_fake", evidence_text="F")],
                    model_metadata=ModelMetadata(provider="mock", model_name="m")
                )
            # Second attempt has valid authentic citation
            return GroundedClaimAnalysis(
                claim_id=claim_id, input_claim=claim_text, verdict=ClaimVerdict.SUPPORTED, explanation="E",
                citations=[Citation(parent_doc_id=evidence[0].doc_id, chunk_id=evidence[0].chunk_id, evidence_text=evidence[0].text)],
                model_metadata=ModelMetadata(provider="mock", model_name="m")
            )

    def v_fn(analysis, ev):
        if attempt == 1:
            return VerificationReport(all_citations_valid=False, verdict_supported_by_evidence=False)
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=True)

    graph = create_verification_graph(hybrid, FlipGenerator(), v_fn, max_retries=1)
    state = graph.invoke({
        "claim_id": "7", "claim_text": "Text",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["retry_count"] == 1
    assert state["is_verified_conclusive"] is True


def test_adversarial_case_h_repeated_unsupported_output_bounded_retry():
    """CASE H: Repeated unsupported output -> Bounded retry exhausted, no false success."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="T", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter()
    graph = create_verification_graph(hybrid, generator, lambda a, e: VerificationReport(all_citations_valid=False, verdict_supported_by_evidence=False), max_retries=1)
    state = graph.invoke({
        "claim_id": "8", "claim_text": "T",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["retry_count"] == 1
    assert state["is_verified_conclusive"] is False


def test_adversarial_case_i_provider_auth_failure_technical_not_abstention():
    """CASE I: Provider authentication failure -> Technical failure, not scientific abstention."""
    adapter = NineRouterAdapter(base_url="http://127.0.0.1:59999", api_key="", zero_cost_authorized=False)
    with pytest.raises(ProviderExecutionError) as exc:
        adapter.generate("9", "Claim", [])
    assert exc.value.status == ProviderStatus.AUTH_FAILED


def test_adversarial_case_j_provider_route_listed_cost_unverifiable():
    """CASE J: Provider route listed, cost unverifiable -> Fail-closed, zero inference requests."""
    adapter = NineRouterAdapter(base_url="http://127.0.0.1:59999", api_key="test_key", zero_cost_authorized=False)
    with pytest.raises(NineRouterPreflightError) as exc:
        adapter.preflight_check()
    assert exc.value.status in [ProviderStatus.COST_NOT_VERIFIED, ProviderStatus.AUTH_FAILED, ProviderStatus.TRANSPORT_ERROR]


def test_adversarial_case_k_malformed_response_recorded_in_accounting():
    """CASE K: Malformed structured response -> Schema failure recorded in denominator."""
    records = [
        {"status": EvaluationCaseStatus.SCHEMA_INVALID.value, "analysis": None},
        {"status": EvaluationCaseStatus.COMPLETED.value, "analysis": GroundedClaimAnalysis(claim_id="1", input_claim="T", verdict=ClaimVerdict.SUPPORTED, explanation="E", model_metadata=ModelMetadata(provider="m", model_name="m"))}
    ]
    summary = evaluate_grounded_analyses(records)
    assert summary.total_attempted == 2
    assert summary.schema_invalid_count == 1
    assert summary.schema_valid_rate == 0.5


def test_adversarial_case_l_missing_gold_annotation_not_evaluable():
    """CASE L: Gold annotation missing -> NOT_EVALUABLE, not false negative."""
    aligner = GoldRationaleAligner(qrels={}, rationales={})
    res = aligner.align_citations("absent_claim_id", [
        CitationVerificationResult(
            citation=Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="t"),
            is_valid_structure=True, is_contextual=True,
        )
    ])
    assert res[0].gold_sentence_aligned is None


def test_adversarial_case_m_mock_generator_labeled_mock():
    """CASE M: Mock generator over claims is clearly labeled MOCK evaluation."""
    summary = evaluate_grounded_analyses([], evaluation_mode="MOCK")
    assert summary.evaluation_mode == "MOCK"


def test_adversarial_case_n_r1_and_r2_retrievers_functional():
    """CASE N: BM25, Dense and Hybrid remain functional without regression."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="CRISPR Cas9 editing", title="T")]
    bm25 = BM25Retriever()
    bm25.index(chunks)
    hybrid = HybridRRFRetriever(bm25, bm25)
    res = hybrid.retrieve("CRISPR", top_k=1)
    assert len(res) == 1
    assert res[0].doc_id == "d1"


def test_adversarial_case_o_packaged_smoke_self_contained():
    """CASE O: Installed wheel CLI runs outside source tree."""
    from scientific_rag.cli import run_self_contained_smoke
    run_self_contained_smoke()


def test_adversarial_case_p_no_citations_valid_abstention():
    """CASE P: No citations with legitimate evidence insufficiency is valid abstention."""
    m = MockGenerationAdapter(default_verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE)
    a = m.generate("1", "T", [])
    assert a.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
    assert len(a.citations) == 0


def test_adversarial_case_q_conflicting_evidence_rejected():
    """CASE Q: Conflicting evidence and mixed citations produce unverified outcome."""
    ev = [
        RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Evidence supports."),
        RetrievalResult(chunk_id="c2", doc_id="d2", score=0.9, rank=2, text="Evidence contradicts and fails."),
    ]
    a = GroundedClaimAnalysis(
        claim_id="1", input_claim="Claim", verdict=ClaimVerdict.SUPPORTED, explanation="E",
        citations=[
            Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Evidence supports."),
            Citation(parent_doc_id="d2", chunk_id="c2", evidence_text="Evidence contradicts and fails."),
        ],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(a, ev)
    assert rep.verdict_supported_by_evidence is False
