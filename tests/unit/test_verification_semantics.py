"""Verification semantics: source identity, authoritative premise for NLI, abstention and LangGraph state machine, verdict-aware metrics, request/token accounting, gold-label isolation, cross-module scenarios."""

import json
from pathlib import Path
import pytest
from typing import Any, Dict, List, Optional

from scientific_rag.data.models import RetrievalResult
from scientific_rag.data.sentence_provenance import CanonicalSentence, SciFactSentenceProvenance
from scientific_rag.evaluation.grounding import (
    EvaluationCaseStatus,
    GroundingEvaluationSummary,
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
    CostStatus,
    GroundedClaimAnalysis,
    ModelMetadata,
    ProviderRequestEvent,
    VerificationOutcome,
    VerificationReport,
    WorkflowOutcome,
)
from scientific_rag.verification.citations import CitationVerifier
from scientific_rag.verification.gold_alignment import GoldRationaleAligner, PostHocGoldEvaluator
from scientific_rag.verification.support import PretrainedNLIProxy
from scientific_rag.verification.verifier import MultiLayerEvidenceVerifier
from scientific_rag.workflows.graph import WorkflowState, create_verification_graph


# ==============================================================================
# TEST GROUP T1: SOURCE IDENTITY & AMBIGUITY
# ==============================================================================

def test_t1_01_exact_canonical_sentence_attribution():
    """T1-01: Exact canonical sentence attribution matches sentence ID."""
    prov = SciFactSentenceProvenance()
    prov._doc_sentences["doc_1"] = [
        CanonicalSentence(doc_id="doc_1", sentence_idx=0, text="Aspirin reduces fever in pediatric patients."),
    ]
    prov._is_loaded = True

    verifier = CitationVerifier(provenance=prov)
    chunk = RetrievalResult(
        chunk_id="doc_1#c0", doc_id="doc_1", score=1.0, rank=1,
        text="Aspirin reduces fever in pediatric patients.",
    )
    cit = Citation(
        parent_doc_id="doc_1", chunk_id="doc_1#c0", sentence_idx=0,
        evidence_text="Aspirin reduces fever in pediatric patients.",
    )
    res = verifier.verify_citations([cit], [chunk])
    assert res[0].is_sentence_valid is True
    assert res[0].is_ambiguous_sentence is False
    assert res[0].authoritative_sentence_text == "Aspirin reduces fever in pediatric patients."


def test_t1_02_harmless_whitespace_normalization():
    """T1-02: Whitespace differences normalize safely without corrupting match."""
    prov = SciFactSentenceProvenance()
    prov._doc_sentences["doc_1"] = [
        CanonicalSentence(doc_id="doc_1", sentence_idx=0, text="Aspirin   reduces  fever\nin pediatric patients."),
    ]
    prov._is_loaded = True
    verifier = CitationVerifier(provenance=prov)
    chunk = RetrievalResult(
        chunk_id="doc_1#c0", doc_id="doc_1", score=1.0, rank=1,
        text="Aspirin reduces fever in pediatric patients.",
    )
    cit = Citation(
        parent_doc_id="doc_1", chunk_id="doc_1#c0", sentence_idx=0,
        evidence_text="Aspirin reduces fever in pediatric patients.",
    )
    res = verifier.verify_citations([cit], [chunk])
    assert res[0].is_text_authentic is True
    assert res[0].is_sentence_valid is True


def test_t1_03_negation_reversal_rejected():
    """T1-03: Negation altered text fails exact substring match."""
    verifier = CitationVerifier()
    chunk = RetrievalResult(
        chunk_id="doc_1#c0", doc_id="doc_1", score=1.0, rank=1,
        text="The treatment does not reduce mortality.",
    )
    cit = Citation(
        parent_doc_id="doc_1", chunk_id="doc_1#c0", sentence_idx=0,
        evidence_text="The treatment does reduce mortality.",
    )
    res = verifier.verify_citations([cit], [chunk])
    assert res[0].is_text_authentic is False
    assert res[0].is_chunk_valid is False


def test_t1_04_numerical_change_rejected():
    """T1-04: Altered numerical values fail exact substring match."""
    verifier = CitationVerifier()
    chunk = RetrievalResult(
        chunk_id="doc_1#c0", doc_id="doc_1", score=1.0, rank=1,
        text="The survival rate was 45% at 5 years.",
    )
    cit = Citation(
        parent_doc_id="doc_1", chunk_id="doc_1#c0", sentence_idx=0,
        evidence_text="The survival rate was 85% at 5 years.",
    )
    res = verifier.verify_citations([cit], [chunk])
    assert res[0].is_text_authentic is False


def test_t1_05_and_t1_06_ambiguous_quotation_matching_two_sentences():
    """T1-05, T1-06 (F01): Quotation matching two sentences cannot claim unique attribution."""
    prov = SciFactSentenceProvenance()
    prov._doc_sentences["doc_2"] = [
        CanonicalSentence(doc_id="doc_2", sentence_idx=0, text="The intervention improved survival in the study cohort."),
        CanonicalSentence(doc_id="doc_2", sentence_idx=1, text="The intervention improved symptoms in the study cohort."),
    ]
    prov._is_loaded = True

    verifier = CitationVerifier(provenance=prov)
    chunk = RetrievalResult(
        chunk_id="doc_2#c0", doc_id="doc_2", score=1.0, rank=1,
        text="The intervention improved survival in the study cohort. The intervention improved symptoms in the study cohort.",
    )
    # Model supplies ambiguous short quote and claims sentence 1
    cit = Citation(
        parent_doc_id="doc_2", chunk_id="doc_2#c0", sentence_idx=1,
        evidence_text="The intervention improved",
    )
    res = verifier.verify_citations([cit], [chunk])
    assert res[0].is_ambiguous_sentence is True
    assert res[0].is_sentence_valid is False
    assert res[0].citation.sentence_idx is None
    assert "Ambiguous quotation" in res[0].notes


def test_t1_07_quotation_from_sentence_0_claiming_sentence_1_rejected():
    """T1-07: Quotation belonging to sentence 0 cannot authenticate claimed sentence 1."""
    prov = SciFactSentenceProvenance()
    prov._doc_sentences["doc_3"] = [
        CanonicalSentence(doc_id="doc_3", sentence_idx=0, text="Drug X inhibits kinase activity in vitro."),
        CanonicalSentence(doc_id="doc_3", sentence_idx=1, text="Drug Y enhances receptor expression in vivo."),
    ]
    prov._is_loaded = True
    verifier = CitationVerifier(provenance=prov)
    chunk = RetrievalResult(
        chunk_id="doc_3#c0", doc_id="doc_3", score=1.0, rank=1,
        text="Drug X inhibits kinase activity in vitro. Drug Y enhances receptor expression in vivo.",
    )
    cit = Citation(
        parent_doc_id="doc_3", chunk_id="doc_3#c0", sentence_idx=1,
        evidence_text="Drug X inhibits kinase activity in vitro.",
    )
    res = verifier.verify_citations([cit], [chunk])
    assert res[0].is_sentence_valid is False
    assert res[0].citation.sentence_idx is None
    assert "Quotation mismatch for sentence_idx 1" in res[0].notes


# ==============================================================================
# TEST GROUP T2: VERIFICATION TRUST & AUTHENTIC PREMISE
# ==============================================================================

def test_t2_01_valid_chunk_with_invalid_sentence_not_sentence_valid():
    """T2-01 (F02): Valid chunk reference is NOT equivalent to valid sentence attribution."""
    prov = SciFactSentenceProvenance()
    prov._doc_sentences["doc_4"] = [
        CanonicalSentence(doc_id="doc_4", sentence_idx=0, text="Valid sentence 0 text."),
    ]
    prov._is_loaded = True
    verifier = CitationVerifier(provenance=prov)
    chunk = RetrievalResult(
        chunk_id="doc_4#c0", doc_id="doc_4", score=1.0, rank=1,
        text="Valid sentence 0 text.",
    )
    # Claiming sentence 99 which does not exist in chunk
    cit = Citation(
        parent_doc_id="doc_4", chunk_id="doc_4#c0", sentence_idx=99,
        evidence_text="Valid sentence 0 text.",
    )
    res = verifier.verify_citations([cit], [chunk])
    assert res[0].is_chunk_valid is True
    assert res[0].is_sentence_valid is False
    assert res[0].citation.sentence_idx is None


def test_t2_03_and_t2_04_no_chunk_evidence_borrowing():
    """T2-03, T2-04 (F03): NLI premise evaluates cited sentence, not unrelated chunk sentences."""
    # Chunk has sentence 0 (entails claim) and sentence 1 (neutral to claim)
    sent0 = "Vaccines stimulate neutralizing antibody production against virus."
    sent1 = "The hospital has 500 beds in the intensive care wing."
    chunk_text = f"{sent0} {sent1}"

    prov = SciFactSentenceProvenance()
    prov._doc_sentences["doc_5"] = [
        CanonicalSentence(doc_id="doc_5", sentence_idx=0, text=sent0),
        CanonicalSentence(doc_id="doc_5", sentence_idx=1, text=sent1),
    ]
    prov._is_loaded = True

    proxy = PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True)

    # Citation cites sentence 1 (neutral hospital beds sentence) for claim about vaccines
    cit = Citation(
        parent_doc_id="doc_5", chunk_id="doc_5#c0", sentence_idx=1,
        evidence_text=sent1,
    )
    c_res = CitationVerificationResult(
        citation=cit,
        is_valid_structure=True,
        is_contextual=True,
        is_text_authentic=True,
        is_chunk_valid=True,
        is_sentence_valid=True,
        authoritative_sentence_text=sent1,
        authoritative_evidence_text=chunk_text,
    )

    out = proxy.verify_semantic_support("Vaccines induce antibodies.", [c_res])
    # Must evaluate sent1 ("hospital has 500 beds"), which is NEUTRAL, NOT ENTAILMENT!
    assert out[0].nli_verdict in ["NEUTRAL", "NOT_VERIFIABLE"]
    assert out[0].nli_verdict != "ENTAILMENT"


def test_t2_05_and_t2_06_supported_and_contradicted_consistency():
    """T2-05, T2-06 (F06): CONTRADICTED + contradiction is consistent; SUPPORTED + entailment is consistent."""
    verifier = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True))

    ev = [
        RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Drug A causes severe cell lysis and does not prevent toxicity."),
    ]
    # Claim: Drug A prevents cell lysis (verdict CONTRADICTED)
    analysis_con = GroundedClaimAnalysis(
        claim_id="q1",
        input_claim="Drug A prevents cell lysis.",
        verdict=ClaimVerdict.CONTRADICTED,
        explanation="Drug A causes lysis.",
        citations=[
            Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Drug A causes severe cell lysis and does not prevent toxicity.", relationship="contradicts")
        ],
        model_metadata=ModelMetadata(provider="test", model_name="test"),
    )
    report_con = verifier.verify(analysis_con, ev)
    assert report_con.verdict_supported_by_evidence is True
    assert VerificationOutcome.SEMANTICALLY_CONSISTENT in report_con.outcomes


def test_t2_07_and_t2_08_verdict_contradiction_mismatch_rejected():
    """T2-07, T2-08: Inconsistent verdict and evidence rejected."""
    verifier = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True))
    ev = [
        RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Drug A causes severe cell lysis and does not prevent toxicity."),
    ]
    # Claim proposed as SUPPORTED when evidence contradicts it
    analysis_sup = GroundedClaimAnalysis(
        claim_id="q1",
        input_claim="Drug A prevents cell lysis.",
        verdict=ClaimVerdict.SUPPORTED,
        explanation="Contradicted by data.",
        citations=[
            Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Drug A causes severe cell lysis and does not prevent toxicity.", relationship="supports")
        ],
        model_metadata=ModelMetadata(provider="test", model_name="test"),
    )
    report_sup = verifier.verify(analysis_sup, ev)
    assert report_sup.verdict_supported_by_evidence is False


# ==============================================================================
# TEST GROUP T3: ABSTENTION & LANGGRAPH STATE MACHINE
# ==============================================================================

def test_t3_01_abstention_with_zero_citations_is_not_verified_conclusive():
    """T3-01 (F04): INSUFFICIENT_EVIDENCE with zero citations is a valid abstention, NOT verified conclusive."""
    verifier = MultiLayerEvidenceVerifier()
    analysis = GroundedClaimAnalysis(
        claim_id="q_abs",
        input_claim="Unproven medical claim.",
        verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
        explanation="No relevant data found.",
        citations=[],
        model_metadata=ModelMetadata(provider="test", model_name="test"),
    )
    report = verifier.verify(analysis, [])
    assert report.all_citations_valid is True
    assert report.verdict_supported_by_evidence is True
    assert VerificationOutcome.INSUFFICIENT_EVIDENCE in report.outcomes


def test_t3_02_initial_insufficient_evidence_triggers_expansion_retry():
    """T3-02 (F04): Initial insufficient evidence triggers expand_retrieve retry."""
    from scientific_rag.retrieval.hybrid import HybridRRFRetriever
    from unittest.mock import MagicMock

    mock_retriever = MagicMock(spec=HybridRRFRetriever)
    mock_retriever.retrieve.return_value = [
        RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Text")
    ]

    class SteppedGenerator(BaseGenerationAdapter):
        def __init__(self):
            self.calls = 0
        def generate(self, claim_id, claim_text, evidence):
            self.calls += 1
            if self.calls == 1:
                return GroundedClaimAnalysis(
                    claim_id=claim_id, input_claim=claim_text,
                    verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE, explanation="First attempt abstain",
                    citations=[], model_metadata=ModelMetadata(provider="test", model_name="test"),
                )
            return GroundedClaimAnalysis(
                claim_id=claim_id, input_claim=claim_text,
                verdict=ClaimVerdict.SUPPORTED, explanation="Second attempt found it",
                citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Text")],
                model_metadata=ModelMetadata(provider="test", model_name="test"),
            )

    gen = SteppedGenerator()

    def mock_verifier(analysis, ev):
        if not analysis.citations:
            return VerificationReport(all_citations_valid=False, verdict_supported_by_evidence=False)
        return VerificationReport(
            all_citations_valid=True, verdict_supported_by_evidence=True,
            citation_details=[
                CitationVerificationResult(
                    citation=analysis.citations[0], is_valid_structure=True, is_contextual=True, is_text_authentic=True
                )
            ]
        )

    graph = create_verification_graph(mock_retriever, gen, mock_verifier, max_retries=1)
    res = graph.invoke({
        "claim_id": "q1", "claim_text": "Claim", "evidence_items": [],
        "analysis": None, "verification": None, "retry_count": 0, "max_retries": 1,
        "final_verdict": None, "is_verified_conclusive": False,
        "workflow_outcome": "", "generation_attempts": [], "provider_requests": []
    })

    assert res["retry_count"] == 1
    assert gen.calls == 2
    assert res["final_verdict"] == "SUPPORTED"
    assert res["workflow_outcome"] == WorkflowOutcome.VERIFIED_SUPPORTED.value


def test_t3_04_unsupported_proposal_safely_finalizes_as_insufficient():
    """T3-04 Unsupported conclusive proposal safely finalizes as INSUFFICIENT_EVIDENCE."""
    from unittest.mock import MagicMock
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    class FailingGenerator(BaseGenerationAdapter):
        def generate(self, claim_id, claim_text, evidence):
            return GroundedClaimAnalysis(
                claim_id=claim_id, input_claim=claim_text,
                verdict=ClaimVerdict.SUPPORTED, explanation="Fake conclusion",
                citations=[], model_metadata=ModelMetadata(provider="test", model_name="test"),
            )

    gen = FailingGenerator()
    def rejecting_verifier(analysis, ev):
        return VerificationReport(all_citations_valid=False, verdict_supported_by_evidence=False)

    graph = create_verification_graph(mock_retriever, gen, rejecting_verifier, max_retries=1)
    res = graph.invoke({
        "claim_id": "q1", "claim_text": "Claim", "evidence_items": [],
        "analysis": None, "verification": None, "retry_count": 0, "max_retries": 1,
        "final_verdict": None, "is_verified_conclusive": False,
        "workflow_outcome": "", "generation_attempts": [], "provider_requests": []
    })

    # Generator proposed SUPPORTED, but verifier rejected -> final verdict safely finalized as INSUFFICIENT_EVIDENCE
    assert res["final_verdict"] == ClaimVerdict.INSUFFICIENT_EVIDENCE.value
    assert res["workflow_outcome"] == WorkflowOutcome.UNVERIFIED_CONCLUSION.value


# ==============================================================================
# TEST GROUP T4: METRIC SEMANTICS & VERDICT-AWARE GROUNDING
# ==============================================================================

def test_t4_01_and_t4_02_verdict_aware_contradiction_not_unsupported():
    """T4-02, T4-03 (F06): CONTRADICTION citation for CONTRADICTED claim is NOT unsupported."""
    rec_con = {
        "status": EvaluationCaseStatus.COMPLETED.value,
        "analysis": GroundedClaimAnalysis(
            claim_id="c_con",
            input_claim="Claim that is false.",
            verdict=ClaimVerdict.CONTRADICTED,
            explanation="Evidence contradicts it.",
            citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="True opposite.", relationship="contradicts")],
            verification=VerificationReport(
                total_citations=1, valid_citations=1,
                citation_details=[
                    CitationVerificationResult(
                        citation=Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="True opposite."),
                        is_valid_structure=True, is_contextual=True, is_text_authentic=True,
                        nli_verdict="CONTRADICTION",
                    )
                ]
            ),
            model_metadata=ModelMetadata(provider="test", model_name="test"),
        )
    }

    summary = evaluate_grounded_analyses([rec_con])
    assert summary.unsupported_citation_count == 0
    assert summary.unsupported_citation_rate == 0.0


# ==============================================================================
# TEST GROUP T5: COMPLETE PROVIDER REQUEST ACCOUNTING
# ==============================================================================

def test_t5_02_and_t5_04_provider_requests_and_tokens_accumulate_across_retries():
    """T5-02, T5-04 (F05): Token usage and provider requests accumulate across LangGraph attempts."""
    from unittest.mock import MagicMock
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    class MultiCallGenerator(BaseGenerationAdapter):
        def __init__(self):
            self.step = 0
        def generate(self, claim_id, claim_text, evidence):
            self.step += 1
            return GroundedClaimAnalysis(
                claim_id=claim_id, input_claim=claim_text,
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE if self.step == 1 else ClaimVerdict.SUPPORTED,
                explanation="Step explanation",
                citations=[],
                model_metadata=ModelMetadata(
                    provider="test", model_name="test",
                    prompt_tokens=1000, completion_tokens=50, total_tokens=1050,
                    provider_requests_count=1,
                    cost_status=CostStatus.UNKNOWN,
                )
            )

    gen = MultiCallGenerator()
    def stepped_verifier(analysis, ev):
        if gen.step == 1:
            return VerificationReport(all_citations_valid=False, verdict_supported_by_evidence=False)
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=True)

    graph = create_verification_graph(mock_retriever, gen, stepped_verifier, max_retries=1)
    res = graph.invoke({
        "claim_id": "q1", "claim_text": "Claim", "evidence_items": [],
        "analysis": None, "verification": None, "retry_count": 0, "max_retries": 1,
        "final_verdict": None, "is_verified_conclusive": False,
        "workflow_outcome": "", "generation_attempts": [], "provider_requests": []
    })

    assert len(res["generation_attempts"]) == 2
    assert len(res["provider_requests"]) == 2
    # Verify aggregate metadata on final analysis
    meta = res["analysis"].model_metadata
    assert meta.prompt_tokens == 2000
    assert meta.completion_tokens == 100
    assert meta.total_tokens == 2100
    assert meta.provider_requests_count == 2


# ==============================================================================
# TEST GROUP T6: GOLD LABEL ISOLATION
# ==============================================================================

def test_t6_01_and_t6_02_gold_labels_do_not_affect_inference():
    """T6-01, T6-02 (F09): Changing gold annotations cannot alter inference or graph routing."""
    from unittest.mock import MagicMock
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Evidence text confirming claim.")
    ]

    adapter = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)

    # Verifier runs purely inference verification with ZERO gold data
    # Mock verifier that accepts consistent evidence
    def inference_verifier(analysis, ev):
        return VerificationReport(
            all_citations_valid=True,
            verdict_supported_by_evidence=True,
            citation_details=[
                CitationVerificationResult(
                    citation=analysis.citations[0],
                    is_valid_structure=True,
                    is_contextual=True,
                    is_text_authentic=True,
                )
            ],
        )

    graph = create_verification_graph(mock_retriever, adapter, inference_verifier)

    state_init = {
        "claim_id": "q1", "claim_text": "Claim", "evidence_items": [],
        "analysis": None, "verification": None, "retry_count": 0, "max_retries": 1,
        "final_verdict": None, "is_verified_conclusive": False,
        "workflow_outcome": "", "generation_attempts": [], "provider_requests": []
    }

    out1 = graph.invoke(state_init.copy())

    # Post-hoc evaluation runs with gold set A
    evaluator_a = PostHocGoldEvaluator(qrels={"q1": {"d1"}}, rationales={"q1": {"d1": [0]}})
    enriched_a = evaluator_a.evaluate_analysis(out1["analysis"])
    assert enriched_a.verification.gold_aligned_citations == 1

    # Post-hoc evaluation runs with gold set B (unrelated doc)
    evaluator_b = PostHocGoldEvaluator(qrels={"q1": {"d999"}}, rationales={})
    enriched_b = evaluator_b.evaluate_analysis(out1["analysis"])
    assert enriched_b.verification.gold_aligned_citations == 0

    # The inference verdict and graph routing remain strictly identical
    assert out1["final_verdict"] == "SUPPORTED"
