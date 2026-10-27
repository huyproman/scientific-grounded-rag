"""
Unit tests for multi-layer evidence verification:
- Citation validity (valid vs hallucinated IDs).
- Gold rationale alignment.
- Bounded CPU NLI proxy.
- Aggregate VerificationReport.
"""

import pytest

from scientific_rag.data.models import RetrievalResult
from scientific_rag.generation.schemas import (
    Citation,
    ClaimVerdict,
    GroundedClaimAnalysis,
    ModelMetadata,
    VerificationOutcome,
)
from scientific_rag.verification.citations import CitationVerifier
from scientific_rag.verification.gold_alignment import GoldRationaleAligner
from scientific_rag.verification.support import BoundedNLIProxy
from scientific_rag.verification.verifier import MultiLayerEvidenceVerifier


def test_t01_citation_validity_valid_context():
    """T01: Verifies citations present in retrieved evidence context are marked valid."""
    evidence = [
        RetrievalResult(chunk_id="chunk_1", doc_id="doc_A", score=1.0, rank=1, text="Sample text A"),
        RetrievalResult(chunk_id="chunk_2", doc_id="doc_B", score=0.8, rank=2, text="Sample text B"),
    ]
    citations = [
        Citation(parent_doc_id="doc_A", chunk_id="chunk_1", evidence_text="Sample text A"),
    ]

    verifier = CitationVerifier()
    res = verifier.verify_citations(citations, evidence)

    assert len(res) == 1
    assert res[0].is_valid_structure is True
    assert res[0].is_contextual is True


def test_t02_citation_validity_hallucinated_id_rejected():
    """T02: Verifies hallucinated chunk or doc IDs are flagged as non-contextual."""
    evidence = [
        RetrievalResult(chunk_id="chunk_1", doc_id="doc_A", score=1.0, rank=1, text="Sample text A"),
    ]
    citations = [
        Citation(parent_doc_id="doc_A", chunk_id="fake_chunk_99", evidence_text="Fake"),
        Citation(parent_doc_id="fake_doc_99", chunk_id="chunk_1", evidence_text="Mismatched doc"),
    ]

    verifier = CitationVerifier()
    res = verifier.verify_citations(citations, evidence)

    assert len(res) == 2
    assert res[0].is_contextual is False
    assert "was not present in the retrieved evidence context" in res[0].notes

    assert res[1].is_contextual is False
    assert "belongs to document 'doc_A'" in res[1].notes


def test_t03_gold_rationale_alignment():
    """T03: Verifies alignment against gold qrels documents and sentence rationale sets."""
    qrels = {"claim_100": {"doc_A", "doc_B"}}
    rationales = {"claim_100": {"doc_A": [[0, 1]]}}

    aligner = GoldRationaleAligner(qrels=qrels, rationales=rationales)

    evidence = [
        RetrievalResult(chunk_id="c1", doc_id="doc_A", score=1.0, rank=1, text="Text"),
        RetrievalResult(chunk_id="c2", doc_id="doc_C", score=0.5, rank=2, text="Text C"),
    ]
    citations = [
        Citation(parent_doc_id="doc_A", chunk_id="c1", evidence_text="T", sentence_idx=1),
        Citation(parent_doc_id="doc_C", chunk_id="c2", evidence_text="T", sentence_idx=0),
    ]

    citation_verifier = CitationVerifier()
    init_res = citation_verifier.verify_citations(citations, evidence)
    aligned = aligner.align_citations("claim_100", init_res)

    assert aligned[0].gold_doc_aligned is True
    assert aligned[0].gold_sentence_aligned is True

    assert aligned[1].gold_doc_aligned is False
    assert aligned[1].gold_sentence_aligned is False


def test_t04_bounded_nli_proxy_entailment_and_contradiction():
    """T04: Verifies CPU NLI proxy correctly identifies entailment and contradiction."""
    nli = BoundedNLIProxy()

    # Entailment pair
    premise_ent = "The clinical trial confirms that vitamin D supplementation significantly reduces respiratory infections."
    hypo_ent = "Vitamin D reduces respiratory infections in patients."
    v_ent, c_ent = nli.evaluate_pair(premise_ent, hypo_ent)
    assert v_ent == "ENTAILMENT"
    assert c_ent > 0.5

    # Contradiction pair
    premise_con = "The study found that vitamin D fails to reduce respiratory infection rates."
    hypo_con = "Vitamin D reduces respiratory infections in patients."
    v_con, c_con = nli.evaluate_pair(premise_con, hypo_con)
    assert v_con == "CONTRADICTION"
    assert c_con > 0.5


def test_t05_multi_layer_evidence_verifier_complete_report():
    """T05: Verifies end-to-end MultiLayerEvidenceVerifier generates comprehensive report."""
    qrels = {"claim_01": {"doc_1"}}
    verifier = MultiLayerEvidenceVerifier(qrels=qrels)

    evidence = [
        RetrievalResult(chunk_id="c1", doc_id="doc_1", score=1.0, rank=1, text="Vitamin D prevents rickets."),
    ]
    analysis = GroundedClaimAnalysis(
        claim_id="claim_01",
        input_claim="Vitamin D prevents rickets in children.",
        verdict=ClaimVerdict.SUPPORTED,
        explanation="Evidence demonstrates vitamin D prevents rickets.",
        citations=[
            Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="Vitamin D prevents rickets.")
        ],
        model_metadata=ModelMetadata(provider="mock", model_name="mock"),
    )

    report = verifier.verify(analysis, evidence)
    assert report.total_citations == 1
    assert report.valid_citations == 1
    assert report.hallucinated_citations == 0
    assert report.gold_aligned_citations == 1
    assert report.all_citations_valid is True
    assert VerificationOutcome.VALID_CITATION in report.outcomes
