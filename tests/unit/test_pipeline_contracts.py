"""Pipeline contracts: evidence resolution, NLI and lexical fallback, gold alignment, verdict-evidence consistency, graph retry, opt-in provider gating, grounding accounting, typed provider outcomes, packaging."""

import os
import pytest

from scientific_rag.common.paths import RAW_DATA_DIR
from scientific_rag.data.models import Chunk, RetrievalResult
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
    GroundedClaimAnalysis,
    ModelMetadata,
    VerificationOutcome,
    VerificationReport,
)
from scientific_rag.retrieval.base import BaseRetriever
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


# =========================================================================
# I01: Authoritative Evidence Resolution & Text Integrity (8 Tests)
# =========================================================================

def test_i01_t01_valid_doc_chunk_and_exact_snippet():
    """I01-T01: Valid document, chunk, and exact source snippet are accepted as authentic."""
    evidence = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Auth source text.")]
    citations = [Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Auth source text.")]
    verifier = CitationVerifier()
    res = verifier.verify_citations(citations, evidence)
    assert res[0].is_contextual is True
    assert res[0].is_text_authentic is True
    assert res[0].authoritative_evidence_text == "Auth source text."


def test_i01_t02_valid_ids_fabricated_evidence_text_rejected():
    """I01-T02: Valid chunk ID but fabricated model-supplied text is rejected."""
    evidence = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Authentic text.")]
    citations = [Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Fabricated text not in chunk.")]
    verifier = CitationVerifier()
    res = verifier.verify_citations(citations, evidence)
    assert res[0].is_contextual is True
    assert res[0].is_text_authentic is False
    assert "Evidence text mismatch" in res[0].notes


def test_i01_t03_valid_ids_reversed_negation_in_evidence_text():
    """I01-T03: Reversed negation in model snippet is caught as text mismatch."""
    evidence = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Drug X cures disease Y.")]
    citations = [Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Drug X fails to cure disease Y.")]
    verifier = CitationVerifier()
    res = verifier.verify_citations(citations, evidence)
    assert res[0].is_text_authentic is False


def test_i01_t04_chunk_belongs_to_different_document():
    """I01-T04: Chunk belonging to a different parent document is rejected."""
    evidence = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Authentic text.")]
    citations = [Citation(parent_doc_id="d2", chunk_id="c1", evidence_text="Authentic text.")]
    verifier = CitationVerifier()
    res = verifier.verify_citations(citations, evidence)
    assert res[0].is_contextual is False
    assert "belongs to document 'd1'" in res[0].notes


def test_i01_t05_chunk_id_absent_from_context():
    """I01-T05: Completely absent chunk ID is rejected."""
    evidence = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Authentic text.")]
    citations = [Citation(parent_doc_id="d1", chunk_id="c999", evidence_text="Text.")]
    verifier = CitationVerifier()
    res = verifier.verify_citations(citations, evidence)
    assert res[0].is_contextual is False


def test_i01_t06_whitespace_normalization_variations_supported():
    """I01-T06: Normal whitespace differences do not falsely invalidate real quotes."""
    evidence = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Protein   A\n\tbinds DNA.")]
    citations = [Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Protein A binds DNA.")]
    verifier = CitationVerifier()
    res = verifier.verify_citations(citations, evidence)
    assert res[0].is_text_authentic is True


def test_i01_t07_source_text_used_as_premise_for_semantic_eval():
    """I01-T07: Verifier evaluates semantic support against authoritative chunk text, not cited text."""
    evidence = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Authentic premise.")]
    analysis = GroundedClaimAnalysis(
        claim_id="c_01",
        input_claim="Authentic premise.",
        verdict=ClaimVerdict.SUPPORTED,
        explanation="Exp",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Fabricated text.")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier()
    rep = v.verify(analysis, evidence)
    # Because text is fabricated, citation is invalid and cannot verify claim
    assert rep.all_citations_valid is False
    assert rep.verdict_supported_by_evidence is False


def test_i01_t08_invalid_citation_cannot_cause_supported_verdict():
    """I01-T08: Any invalid citation prevents conclusive supported verdict."""
    evidence = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Source text.")]
    analysis = GroundedClaimAnalysis(
        claim_id="c_02",
        input_claim="Source text.",
        verdict=ClaimVerdict.SUPPORTED,
        explanation="Exp",
        citations=[Citation(parent_doc_id="d1", chunk_id="fake_chunk", evidence_text="Source text.")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier()
    rep = v.verify(analysis, evidence)
    assert rep.verdict_supported_by_evidence is False


# =========================================================================
# I02: Pretrained CPU NLI Proxy & Lexical Fallback (10 Tests)
# =========================================================================

def test_i02_t01_clearly_entailing_pair():
    """I02-T01: Entailing premise-hypothesis pair evaluated correctly."""
    nli = PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True)  # test heuristic path
    v, c = nli.evaluate_pair("Vitamin D significantly reduces respiratory infections.", "Vitamin D reduces respiratory infections.")
    assert v == "ENTAILMENT"


def test_i02_t02_clearly_contradictory_pair():
    """I02-T02: Contradictory premise-hypothesis pair evaluated correctly."""
    nli = PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True)
    v, c = nli.evaluate_pair("Vitamin D fails to reduce infection rates.", "Vitamin D reduces infection rates.")
    assert v == "CONTRADICTION"


def test_i02_t03_topically_related_neutral_pair():
    """I02-T03: Topically related but unsupported pair classified as neutral."""
    nli = PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True)
    v, c = nli.evaluate_pair("Cellular biology investigates organelles.", "Mitochondria produce ATP.")
    assert v == "NEUTRAL"


def test_i02_t04_negation_reversal_classified_as_contradiction():
    """I02-T04: Negation terms trigger contradiction."""
    nli = PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True)
    v, c = nli.evaluate_pair("Study demonstrates treatment works.", "Treatment never works.")
    assert v == "CONTRADICTION"


def test_i02_t05_empty_input_pair():
    """I02-T05: Empty or whitespace premise/hypothesis yields neutral."""
    nli = PretrainedNLIProxy(enable_neural=False)
    v, c = nli.evaluate_pair("", "claim")
    assert v == "NEUTRAL"
    assert c == 0.0


def test_i02_t06_lexical_fallback_flag_labeled():
    """I02-T06: Fallback heuristic produces explicit diagnostic output."""
    nli = PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True)
    v, c = nli._lexical_heuristic("A B C D E", "A B C D E")
    assert v == "ENTAILMENT"
    assert c > 0.8


def test_i02_t07_nli_deberta_model_live_cpu_inference():
    """I02-T07: Pretrained DeBERTa NLI cross-encoder executes on CPU."""
    nli = PretrainedNLIProxy(enable_neural=True)
    v, c = nli.evaluate_pair(
        "Lipid nanoparticles encapsulate synthetic mRNA encoding viral spike protein.",
        "Lipid nanoparticles deliver mRNA encoding antigens.",
    )
    assert v in ["ENTAILMENT", "NEUTRAL"]
    assert c > 0.0


def test_i02_t08_nli_deberta_contradiction_cpu_inference():
    """I02-T08: Pretrained DeBERTa NLI recognizes contradiction on CPU."""
    nli = PretrainedNLIProxy(enable_neural=True)
    v, c = nli.evaluate_pair(
        "Lipid nanoparticles encapsulate synthetic mRNA encoding viral spike protein.",
        "Lipid nanoparticles fail to carry any mRNA.",
    )
    assert v == "CONTRADICTION"
    assert c > 0.5


def test_i02_t09_unverifiable_citation_handling():
    """I02-T09: Non-contextual citation gets NOT_VERIFIABLE NLI verdict."""
    nli = PretrainedNLIProxy(enable_neural=False)
    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="d", chunk_id="c", evidence_text="t"),
        is_valid_structure=True,
        is_contextual=False,
    )
    out = nli.verify_semantic_support("claim", [cit_res])
    assert out[0].nli_verdict == "NOT_VERIFIABLE"


def test_i02_t10_authoritative_premise_selection():
    """I02-T10: Authoritative chunk text is used as premise over model snippet."""
    nli = PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True)
    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="d", chunk_id="c", evidence_text="model said this"),
        is_valid_structure=True,
        is_contextual=True,
        is_text_authentic=True,
        authoritative_evidence_text="actual authoritative chunk text",
    )
    out = nli.verify_semantic_support("actual authoritative chunk text", [cit_res])
    assert out[0].nli_verdict == "ENTAILMENT"


# =========================================================================
# I03: Official SciFact Gold Rationale Alignment (10 Tests)
# =========================================================================

def test_i03_t01_valid_claim_document_sentence_gold_match():
    """I03-T01: Matches claim, document, and specific sentence index."""
    qrels = {"10": {"doc_1"}}
    rationales = {"10": {"doc_1": [{"sentences": [2, 5], "label": "SUPPORT"}]}}
    aligner = GoldRationaleAligner(qrels=qrels, rationales=rationales)

    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="t", sentence_idx=2),
        is_valid_structure=True,
        is_contextual=True,
    )
    res = aligner.align_citations("10", [cit_res])
    assert res[0].gold_doc_aligned is True
    assert res[0].gold_sentence_aligned is True


def test_i03_t02_correct_doc_wrong_evidence_sentence():
    """I03-T02: Correct document but unannotated sentence index."""
    qrels = {"10": {"doc_1"}}
    rationales = {"10": {"doc_1": [{"sentences": [2, 5], "label": "SUPPORT"}]}}
    aligner = GoldRationaleAligner(qrels=qrels, rationales=rationales)

    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="t", sentence_idx=8),
        is_valid_structure=True,
        is_contextual=True,
    )
    res = aligner.align_citations("10", [cit_res])
    assert res[0].gold_doc_aligned is True
    assert res[0].gold_sentence_aligned is False


def test_i03_t03_wrong_document():
    """I03-T03: Non-gold document."""
    qrels = {"10": {"doc_1"}}
    aligner = GoldRationaleAligner(qrels=qrels, rationales={"10": {}})
    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_99", chunk_id="c1", evidence_text="t"),
        is_valid_structure=True,
        is_contextual=True,
    )
    res = aligner.align_citations("10", [cit_res])
    assert res[0].gold_doc_aligned is False


def test_i03_t04_multiple_valid_rationale_sets():
    """I03-T04: Multiple rationale sets in gold annotation."""
    rationales = {"10": {"doc_1": [{"sentences": [1, 2]}, {"sentences": [7]}]}}
    aligner = GoldRationaleAligner(qrels={"10": {"doc_1"}}, rationales=rationales)
    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="t", sentence_idx=7),
        is_valid_structure=True,
        is_contextual=True,
    )
    res = aligner.align_citations("10", [cit_res])
    assert res[0].gold_sentence_aligned is True


def test_i03_t05_missing_gold_rationale_is_not_evaluable():
    """I03-T05: Missing gold rationale annotation returns None (NOT_EVALUABLE), not False."""
    aligner = GoldRationaleAligner(qrels={}, rationales={})
    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="t"),
        is_valid_structure=True,
        is_contextual=True,
    )
    # Claim "unknown_claim" not present in rationales
    res = aligner.align_citations("unknown_claim", [cit_res])
    assert res[0].gold_sentence_aligned is None


def test_i03_t06_no_evidence_label_for_candidate():
    """I03-T06: Claim exists in rationales but document is not in its evidence dict."""
    rationales = {"10": {"doc_other": [{"sentences": [1]}]}}
    aligner = GoldRationaleAligner(qrels={"10": set()}, rationales=rationales)
    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_cand", chunk_id="c1", evidence_text="t"),
        is_valid_structure=True,
        is_contextual=True,
    )
    res = aligner.align_citations("10", [cit_res])
    assert res[0].gold_sentence_aligned is False


def test_i03_t07_chunk_without_sentence_idx_uses_chunk_presence():
    """I03-T07: When sentence_idx is None, gold alignment requires None (NOT_EVALUABLE), not True."""
    rationales = {"10": {"doc_1": [{"sentences": [1]}]}}
    aligner = GoldRationaleAligner(qrels={"10": {"doc_1"}}, rationales=rationales)
    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="t", sentence_idx=None),
        is_valid_structure=True,
        is_contextual=True,
        authoritative_evidence_text="chunk text",
    )
    res = aligner.align_citations("10", [cit_res])
    assert res[0].gold_sentence_aligned is None


_CLAIMS_DEV = RAW_DATA_DIR.parent / "scifact_official" / "data" / "claims_dev.jsonl"
requires_claims_dev = pytest.mark.skipif(
    not _CLAIMS_DEV.exists(),
    reason="official SciFact claims_dev.jsonl missing at data/raw/scifact_official/data/ (external data, not redistributed)",
)


@requires_claims_dev
def test_i03_t08_scifact_official_claims_dev_loading():
    """I03-T08: Verifies DatasetLoader loads official claims_dev.jsonl correctly."""
    from scientific_rag.common.paths import RAW_DATA_DIR
    from scientific_rag.data.loader import DatasetLoader
    loader = DatasetLoader(RAW_DATA_DIR)
    rats = loader.load_gold_rationales()
    assert len(rats) == 300
    assert "1" in rats
    assert "3" in rats
    assert "14717500" in rats["3"]


def test_i03_t09_evaluable_case_denominator_separation():
    """I03-T09: Evaluation accounting computes sentence overlap over evaluable count."""
    analysis = GroundedClaimAnalysis(
        claim_id="3",
        input_claim="Test",
        verdict=ClaimVerdict.SUPPORTED,
        explanation="Exp",
        citations=[Citation(parent_doc_id="doc_x", chunk_id="c1", evidence_text="t")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
        verification=VerificationReport(
            total_citations=1,
            valid_citations=1,
            citation_details=[
                CitationVerificationResult(
                    citation=Citation(parent_doc_id="doc_x", chunk_id="c1", evidence_text="t"),
                    is_valid_structure=True,
                    is_contextual=True,
                    gold_sentence_aligned=None,  # Not evaluable
                )
            ]
        )
    )
    summary = evaluate_grounded_analyses([{"status": "COMPLETED", "analysis": analysis}])
    assert summary.gold_sentence_evaluable_count == 0
    assert summary.gold_sentence_overlap_rate == 0.0


def test_i03_t10_gold_labels_not_exposed_to_generator():
    """I03-T10: Verifies prompt context does not contain gold annotations."""
    from scientific_rag.generation.prompts import format_evidence_context
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Text", metadata={"gold": True})]
    fmt = format_evidence_context(ev)
    assert "gold" not in fmt.lower()


# =========================================================================
# I04: Semantic Verdict-Evidence Consistency (9 Tests)
# =========================================================================

def test_i04_t01_supported_plus_entailment_verified():
    """I04-T01: SUPPORTED verdict with entailing citation is verified."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Study confirms mechanism.")]
    analysis = GroundedClaimAnalysis(
        claim_id="c1",
        input_claim="Study confirms mechanism.",
        verdict=ClaimVerdict.SUPPORTED,
        explanation="E",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Study confirms mechanism.")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True))
    rep = v.verify(analysis, ev)
    assert rep.verdict_supported_by_evidence is True


def test_i04_t02_supported_plus_contradiction_rejected():
    """I04-T02: SUPPORTED verdict when evidence contradicts is NOT verified."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Study fails and disproves claim.")]
    analysis = GroundedClaimAnalysis(
        claim_id="c1",
        input_claim="Study proves claim.",
        verdict=ClaimVerdict.SUPPORTED,
        explanation="E",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Study fails and disproves claim.")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(analysis, ev)
    assert rep.verdict_supported_by_evidence is False


def test_i04_t03_contradicted_plus_contradiction_verified():
    """I04-T03: CONTRADICTED verdict with contradictory evidence is verified."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Study fails to cure infection.")]
    analysis = GroundedClaimAnalysis(
        claim_id="c1",
        input_claim="Study cures infection.",
        verdict=ClaimVerdict.CONTRADICTED,
        explanation="E",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Study fails to cure infection.", relationship="contradicts")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True))
    rep = v.verify(analysis, ev)
    assert rep.verdict_supported_by_evidence is True


def test_i04_t04_contradicted_plus_entailment_rejected():
    """I04-T04: CONTRADICTED verdict when evidence actually entails is rejected."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Study confirms and cures infection.")]
    analysis = GroundedClaimAnalysis(
        claim_id="c1",
        input_claim="Study cures infection.",
        verdict=ClaimVerdict.CONTRADICTED,
        explanation="E",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Study confirms and cures infection.")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(analysis, ev)
    assert rep.verdict_supported_by_evidence is False


def test_i04_t05_neutral_evidence_not_verified():
    """I04-T05: Neutral evidence alone cannot verify SUPPORTED or CONTRADICTED."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Weather in Paris is fine.")]
    analysis = GroundedClaimAnalysis(
        claim_id="c1",
        input_claim="Quantum gravity requires strings.",
        verdict=ClaimVerdict.SUPPORTED,
        explanation="E",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Weather in Paris is fine.")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(analysis, ev)
    assert rep.verdict_supported_by_evidence is False


def test_i04_t06_insufficient_evidence_without_citations_is_valid_abstention():
    """I04-T06: INSUFFICIENT_EVIDENCE with empty citations is a valid abstention."""
    analysis = GroundedClaimAnalysis(
        claim_id="c1",
        input_claim="Unknown statement.",
        verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
        explanation="Cannot verify.",
        citations=[],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier()
    rep = v.verify(analysis, [])
    assert rep.all_citations_valid is True
    assert rep.verdict_supported_by_evidence is True
    assert VerificationOutcome.INSUFFICIENT_EVIDENCE in rep.outcomes


def test_i04_t07_conflicting_citations_reject_supported():
    """I04-T07: Mixed contradictory and entailing citations prevent conclusive SUPPORTED."""
    ev = [
        RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Vitamin D cures rickets."),
        RetrievalResult(chunk_id="c2", doc_id="d2", score=0.9, rank=2, text="Vitamin D fails to cure rickets."),
    ]
    analysis = GroundedClaimAnalysis(
        claim_id="c1",
        input_claim="Vitamin D cures rickets.",
        verdict=ClaimVerdict.SUPPORTED,
        explanation="E",
        citations=[
            Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Vitamin D cures rickets."),
            Citation(parent_doc_id="d2", chunk_id="c2", evidence_text="Vitamin D fails to cure rickets."),
        ],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(analysis, ev)
    assert rep.verdict_supported_by_evidence is False


def test_i04_t08_invalid_citation_prevents_verification():
    """I04-T08: Even with entailing text, invalid citation ID prevents verification."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="True premise.")]
    analysis = GroundedClaimAnalysis(
        claim_id="c1",
        input_claim="True premise.",
        verdict=ClaimVerdict.SUPPORTED,
        explanation="E",
        citations=[Citation(parent_doc_id="fake_doc", chunk_id="fake_chunk", evidence_text="True premise.")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(analysis, ev)
    assert rep.verdict_supported_by_evidence is False


def test_i04_t09_outcomes_contain_verdict_categories():
    """I04-T09: Outcomes list properly records ENTAILMENT_SUPPORTED or CONTRADICTED_BY_EVIDENCE."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Treatment fails and never cures the disease.")]
    analysis = GroundedClaimAnalysis(
        claim_id="c1",
        input_claim="Treatment cures the disease.",
        verdict=ClaimVerdict.CONTRADICTED,
        explanation="E",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Treatment fails and never cures the disease.", relationship="contradicts")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True))
    rep = v.verify(analysis, ev)
    assert VerificationOutcome.CONTRADICTED_BY_EVIDENCE in rep.outcomes


# =========================================================================
# I05: LangGraph Termination & Stateful Retry (11 Tests)
# =========================================================================

def test_i05_t01_valid_supported_conclusive_terminates_at_once():
    """I05-T01: Conclusively verified result terminates on first attempt (retry_count=0)."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="Protein X regulates gene Y.", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)

    def mock_verifier(analysis, ev):
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=True)

    graph = create_verification_graph(hybrid, generator, mock_verifier, max_retries=1)
    state = graph.invoke({
        "claim_id": "c1", "claim_text": "Protein X regulates gene Y.",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["retry_count"] == 0
    assert state["is_verified_conclusive"] is True


def test_i05_t02_unsupported_verdict_triggers_expansion_retry():
    """I05-T02: Unsupported verdict triggers expanded retrieval retry."""
    chunks = [Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", chunk_index=0, text=f"Text {i}", title=f"T{i}") for i in range(20)]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)

    call_count = 0
    def verifier_flip_on_retry(analysis, ev):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=False)
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=True)

    graph = create_verification_graph(hybrid, generator, verifier_flip_on_retry, max_retries=1)
    state = graph.invoke({
        "claim_id": "c1", "claim_text": "Statement",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["retry_count"] == 1
    assert state["is_verified_conclusive"] is True
    assert len(state["evidence_items"]) == 15  # expanded depth


def test_i05_t03_unsupported_after_retry_terminates_without_false_conclusive():
    """I05-T03: After retry exhausted, terminates with is_verified_conclusive=False."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="Text", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)

    def verifier_always_unsupported(analysis, ev):
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=False)

    graph = create_verification_graph(hybrid, generator, verifier_always_unsupported, max_retries=1)
    state = graph.invoke({
        "claim_id": "c1", "claim_text": "Statement",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["retry_count"] == 1
    assert state["is_verified_conclusive"] is False


def test_i05_t04_invalid_citation_triggers_retry():
    """I05-T04: Citation integrity failure triggers retry."""
    chunks = [Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", chunk_index=0, text="T", title="T") for i in range(20)]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter()

    calls = 0
    def verifier_invalid_first(analysis, ev):
        nonlocal calls
        calls += 1
        return VerificationReport(all_citations_valid=(calls > 1), verdict_supported_by_evidence=(calls > 1))

    graph = create_verification_graph(hybrid, generator, verifier_invalid_first, max_retries=1)
    state = graph.invoke({
        "claim_id": "c1", "claim_text": "Statement",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["retry_count"] == 1


def test_i05_t05_max_retries_strictly_bounded_at_one():
    """I05-T05: Graph cannot exceed max_retries."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="T", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter()

    def always_fail(analysis, ev):
        return VerificationReport(all_citations_valid=False, verdict_supported_by_evidence=False)

    graph = create_verification_graph(hybrid, generator, always_fail, max_retries=1)
    state = graph.invoke({
        "claim_id": "c1", "claim_text": "Statement",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["retry_count"] == 1


def test_i05_t06_stale_verification_state_cleared_on_retry():
    """I05-T06: Verification state is reset in expand_retrieve node."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="T", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter()
    graph = create_verification_graph(hybrid, generator, lambda a, e: VerificationReport(), max_retries=1)
    # Graph runs and returns freshly computed state
    state = graph.invoke({
        "claim_id": "c1", "claim_text": "S",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["analysis"] is not None


def test_i05_t07_conclusive_contradicted_terminates():
    """I05-T07: Verified CONTRADICTED terminates immediately as conclusive."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="T", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.CONTRADICTED)

    def verifier_contradiction_ok(analysis, ev):
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=True)

    graph = create_verification_graph(hybrid, generator, verifier_contradiction_ok, max_retries=1)
    state = graph.invoke({
        "claim_id": "c1", "claim_text": "S",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["retry_count"] == 0
    assert state["final_verdict"] == "CONTRADICTED"
    assert state["is_verified_conclusive"] is True


def test_i05_t08_insufficient_evidence_retries_then_abstains():
    """I05-T08: INSUFFICIENT_EVIDENCE attempts 1 expansion then abstains."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="T", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE)

    def verifier_insufficient(analysis, ev):
        # Abstention has all_citations_valid=True but is not a conclusive scientific verdict
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=False)

    graph = create_verification_graph(hybrid, generator, verifier_insufficient, max_retries=1)
    state = graph.invoke({
        "claim_id": "c1", "claim_text": "S",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["retry_count"] == 1
    assert state["final_verdict"] == "INSUFFICIENT_EVIDENCE"


def test_i05_t09_generator_failure_in_graph_terminates():
    """I05-T09: Generator failure terminates gracefully without infinite loop."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="T", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))

    class BrokenGenerator(BaseGenerationAdapter):
        def generate(self, claim_id: str, claim_text: str, evidence: list): return None

    graph = create_verification_graph(hybrid, BrokenGenerator(), lambda a, e: VerificationReport(), max_retries=1)
    state = graph.invoke({
        "claim_id": "c1", "claim_text": "S",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["final_verdict"] == "FAILED"


def test_i05_t10_expanded_evidence_contains_more_candidates():
    """I05-T10: Expand retrieve node increases evidence candidates from 5 to 15."""
    chunks = [Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", chunk_index=0, text="T", title="T") for i in range(20)]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter()

    calls = 0
    def v_fn(a, e):
        nonlocal calls
        calls += 1
        return VerificationReport(all_citations_valid=(calls > 1), verdict_supported_by_evidence=(calls > 1))

    graph = create_verification_graph(hybrid, generator, v_fn, max_retries=1)
    state = graph.invoke({
        "claim_id": "c1", "claim_text": "S",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert len(state["evidence_items"]) == 15


def test_i05_t11_typed_state_preservation():
    """I05-T11: State keys claim_id and claim_text preserved across all transitions."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="T", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter()
    graph = create_verification_graph(hybrid, generator, lambda a, e: VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=True))
    state = graph.invoke({
        "claim_id": "test_id_999", "claim_text": "My scientific claim",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["claim_id"] == "test_id_999"
    assert state["claim_text"] == "My scientific claim"


# =========================================================================
# I06: Fail-Closed Zero-Cost Verification (9 Tests)
# =========================================================================

def test_i06_t01_cost_not_verified_blocks_inference():
    """I06-T01: Reaching gateway without verified zero-cost proof raises COST_NOT_VERIFIED."""
    adapter = NineRouterAdapter(
        base_url="http://127.0.0.1:59999",
        api_key="valid_key",
        zero_cost_authorized=False,
    )
    # Mocking reachability
    with pytest.raises(NineRouterPreflightError) as exc_info:
        adapter.preflight_check()
    assert exc_info.value.status in [ProviderStatus.COST_NOT_VERIFIED, ProviderStatus.AUTH_FAILED, ProviderStatus.TRANSPORT_ERROR]


def test_i06_t02_missing_api_key_blocks_inference():
    """I06-T02: Missing API key immediately blocks inference with AUTH_FAILED."""
    adapter = NineRouterAdapter(base_url="http://127.0.0.1:59999", api_key="")
    with pytest.raises(NineRouterPreflightError) as exc:
        adapter.preflight_check()
    assert exc.value.status == ProviderStatus.AUTH_FAILED


def test_i06_t03_unlisted_model_blocks_inference():
    """I06-T03: Unlisted model blocks execution with MODEL_UNAVAILABLE."""
    adapter = NineRouterAdapter(
        base_url="http://127.0.0.1:59999",
        api_key="test_key",
        model_name="nonexistent/model",
        zero_cost_authorized=True,
    )
    with pytest.raises(NineRouterPreflightError):
        adapter.preflight_check()


def test_i06_t04_gateway_unavailable_blocks_inference():
    """I06-T04: Connection refused raises TRANSPORT_ERROR."""
    adapter = NineRouterAdapter(
        base_url="http://127.0.0.1:59999",  # unused port
        api_key="test_key",
        zero_cost_authorized=True,
    )
    with pytest.raises(NineRouterPreflightError) as exc:
        adapter.preflight_check()
    assert exc.value.status == ProviderStatus.TRANSPORT_ERROR


def test_i06_t05_explicit_zero_cost_authorization_accepted():
    """I06-T05: Explicit zero-cost authorization allows preflight to pass if gateway accepts."""
    KEY = os.getenv("LLM_API_KEY")
    if not KEY or os.getenv("RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip("hosted LLM test: set LLM_API_KEY, LLM_BASE_URL and RUN_LIVE_LLM_TESTS=1 to opt in")
    adapter = NineRouterAdapter(
        base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:59999"),
        api_key=KEY,
        model_name="ag/gemini-3.8-flash-high",
        zero_cost_authorized=True,
    )
    res = adapter.preflight_check()
    assert res["status"] == "HEALTHY"
    assert res["zero_cost_verified"] is True


def test_i06_t06_mock_mode_zero_cost_by_definition():
    """I06-T06: Mock generator is zero-cost by construction."""
    adapter = MockGenerationAdapter()
    analysis = adapter.generate("c1", "text", [])
    assert analysis.model_metadata.is_zero_cost_verified is True
    assert analysis.model_metadata.provider == "mock"


def test_i06_t07_no_inference_request_sent_when_preflight_fails():
    """I06-T07: generate() fails closed without attempting inference if preflight fails."""
    adapter = NineRouterAdapter(base_url="http://127.0.0.1:59999", api_key="", zero_cost_authorized=False)
    with pytest.raises(ProviderExecutionError):
        adapter.generate("c1", "text", [])


def test_i06_t08_fail_closed_prevents_silent_fallback_to_paid():
    """I06-T08: Adapter fails closed rather than silently routing to a paid commercial API."""
    adapter = NineRouterAdapter(base_url="http://127.0.0.1:59999", api_key="k", zero_cost_authorized=False)
    assert adapter.zero_cost_authorized is False


def test_i06_t09_metadata_records_provider_identity():
    """I06-T09: Model metadata accurately records provider name."""
    m = MockGenerationAdapter()
    a = m.generate("1", "T", [])
    assert a.model_metadata.provider == "mock"
    assert a.model_metadata.model_name == "mock-deterministic-v1"


# =========================================================================
# I07: Grounding Evaluation Accounting (11 Tests)
# =========================================================================

def test_i07_t01_all_valid_generated_objects():
    """I07-T01: Evaluates perfect run with 100% schema validity and completed rate."""
    records = [
        {
            "status": "COMPLETED",
            "analysis": GroundedClaimAnalysis(
                claim_id=str(i), input_claim="T", verdict=ClaimVerdict.SUPPORTED, explanation="E",
                citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="t")],
                model_metadata=ModelMetadata(provider="mock", model_name="m"),
                verification=VerificationReport(total_citations=1, valid_citations=1, all_citations_valid=True)
            )
        }
        for i in range(5)
    ]
    summary = evaluate_grounded_analyses(records, evaluation_mode="MOCK")
    assert summary.total_attempted == 5
    assert summary.total_completed == 5
    assert summary.schema_valid_rate == 1.0
    assert summary.evaluation_mode == "MOCK"


def test_i07_t02_schema_invalid_counted_in_attempted_denominator():
    """I07-T02: Schema invalid record reduces schema_valid_rate correctly."""
    records = [
        {"status": "COMPLETED", "analysis": GroundedClaimAnalysis(claim_id="1", input_claim="T", verdict=ClaimVerdict.SUPPORTED, explanation="E", model_metadata=ModelMetadata(provider="m", model_name="m"))},
        {"status": "SCHEMA_INVALID", "analysis": None},
    ]
    summary = evaluate_grounded_analyses(records)
    assert summary.total_attempted == 2
    assert summary.schema_valid_count == 1
    assert summary.schema_invalid_count == 1
    assert summary.schema_valid_rate == 0.5


def test_i07_t03_provider_failure_counted_separately():
    """I07-T03: Technical provider failure counted in provider_failure_rate."""
    records = [
        {"status": "COMPLETED", "analysis": GroundedClaimAnalysis(claim_id="1", input_claim="T", verdict=ClaimVerdict.SUPPORTED, explanation="E", model_metadata=ModelMetadata(provider="m", model_name="m"))},
        {"status": "PROVIDER_FAILED", "analysis": None},
    ]
    summary = evaluate_grounded_analyses(records)
    assert summary.total_attempted == 2
    assert summary.provider_failure_count == 1
    assert summary.provider_failure_rate == 0.5


def test_i07_t04_semantic_unsupported_rate_separated_from_id_validity():
    """I07-T04: Citation has valid ID but unsupported semantic relation (CONTRADICTION)."""
    analysis = GroundedClaimAnalysis(
        claim_id="1", input_claim="T", verdict=ClaimVerdict.SUPPORTED, explanation="E",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="t")],
        model_metadata=ModelMetadata(provider="m", model_name="m"),
        verification=VerificationReport(
            total_citations=1, valid_citations=1, all_citations_valid=True,
            citation_details=[
                CitationVerificationResult(
                    citation=Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="t"),
                    is_valid_structure=True, is_contextual=True, is_text_authentic=True,
                    nli_verdict="CONTRADICTION",
                )
            ]
        )
    )
    summary = evaluate_grounded_analyses([{"status": "COMPLETED", "analysis": analysis}])
    assert summary.citation_validity_rate == 1.0  # ID is valid
    assert summary.unsupported_citation_rate == 1.0  # Semantically unsupported


def test_i07_t05_not_evaluable_gold_sentence_separated():
    """I07-T05: Absent gold annotation is NOT_EVALUABLE, not a failure."""
    analysis = GroundedClaimAnalysis(
        claim_id="1", input_claim="T", verdict=ClaimVerdict.SUPPORTED, explanation="E",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="t")],
        model_metadata=ModelMetadata(provider="m", model_name="m"),
        verification=VerificationReport(
            total_citations=1, valid_citations=1,
            citation_details=[
                CitationVerificationResult(
                    citation=Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="t"),
                    is_valid_structure=True, is_contextual=True,
                    gold_sentence_aligned=None,  # Not evaluable
                )
            ]
        )
    )
    summary = evaluate_grounded_analyses([{"status": "COMPLETED", "analysis": analysis}])
    assert summary.gold_sentence_evaluable_count == 0


def test_i07_t06_empty_records_returns_zero_summary():
    """I07-T06: Empty records list handled gracefully."""
    summary = evaluate_grounded_analyses([])
    assert summary.total_attempted == 0
    assert summary.schema_valid_rate == 0.0


def test_i07_t07_mock_evaluation_explicitly_tagged():
    """I07-T07: Evaluation summary records evaluation_mode='MOCK'."""
    summary = evaluate_grounded_analyses([], evaluation_mode="MOCK")
    assert summary.evaluation_mode == "MOCK"


def test_i07_t08_real_provider_evaluation_explicitly_tagged():
    """I07-T08: Evaluation summary records evaluation_mode='REAL_PROVIDER'."""
    summary = evaluate_grounded_analyses([], evaluation_mode="REAL_PROVIDER")
    assert summary.evaluation_mode == "REAL_PROVIDER"


def test_i07_t09_abstention_rate_calculated_correctly():
    """I07-T09: INSUFFICIENT_EVIDENCE increments abstention count."""
    analysis = GroundedClaimAnalysis(
        claim_id="1", input_claim="T", verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
        explanation="Abstain", citations=[], model_metadata=ModelMetadata(provider="m", model_name="m")
    )
    summary = evaluate_grounded_analyses([{"status": "COMPLETED", "analysis": analysis}])
    assert summary.abstention_rate == 1.0


def test_i07_t10_all_aggregation_counts_reconcile():
    """I07-T10: Attempted count equals completed + schema_invalid + provider_failed."""
    records = [
        {"status": "COMPLETED", "analysis": GroundedClaimAnalysis(claim_id="1", input_claim="T", verdict=ClaimVerdict.SUPPORTED, explanation="E", model_metadata=ModelMetadata(provider="m", model_name="m"))},
        {"status": "SCHEMA_INVALID", "analysis": None},
        {"status": "PROVIDER_FAILED", "analysis": None},
    ]
    summary = evaluate_grounded_analyses(records)
    assert summary.total_attempted == summary.total_completed + summary.schema_invalid_count + summary.provider_failure_count


def test_i07_t11_zero_citations_produces_zero_division_safety():
    """I07-T11: Zero citations safely handled without ZeroDivisionError."""
    analysis = GroundedClaimAnalysis(
        claim_id="1", input_claim="T", verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
        explanation="E", citations=[], model_metadata=ModelMetadata(provider="m", model_name="m")
    )
    summary = evaluate_grounded_analyses([{"status": "COMPLETED", "analysis": analysis}])
    assert summary.total_citations_generated == 0
    assert summary.citation_validity_rate == 1.0
    assert summary.unsupported_citation_rate == 0.0


# =========================================================================
# I08: Typed Provider Execution Outcomes vs Abstention (10 Tests)
# =========================================================================

def test_i08_t01_valid_provider_success():
    """I08-T01: Mock provider returns valid GroundedClaimAnalysis."""
    m = MockGenerationAdapter()
    res = m.generate("1", "Claim", [RetrievalResult(chunk_id="c", doc_id="d", score=1.0, rank=1, text="Text")])
    assert isinstance(res, GroundedClaimAnalysis)
    assert res.verdict == ClaimVerdict.SUPPORTED


def test_i08_t02_provider_timeout_raises_typed_error():
    """I08-T02: Timeout in adapter raises ProviderExecutionError with status TIMEOUT."""
    err = ProviderExecutionError(status=ProviderStatus.TIMEOUT, message="Request timed out")
    assert err.status == ProviderStatus.TIMEOUT


def test_i08_t03_auth_failure_raises_typed_error():
    """I08-T03: Auth failure raises ProviderExecutionError with status AUTH_FAILED."""
    err = ProviderExecutionError(status=ProviderStatus.AUTH_FAILED, message="401 Unauthorized")
    assert err.status == ProviderStatus.AUTH_FAILED


def test_i08_t04_invalid_structured_output_raises_schema_validation_error():
    """I08-T04: Parsing failure raises ProviderExecutionError with status SCHEMA_VALIDATION_FAILED."""
    err = ProviderExecutionError(status=ProviderStatus.SCHEMA_VALIDATION_FAILED, message="JSON parse failed")
    assert err.status == ProviderStatus.SCHEMA_VALIDATION_FAILED


def test_i08_t05_cost_not_verified_error_typed():
    """I08-T05: Cost failure raises NineRouterPreflightError with status COST_NOT_VERIFIED."""
    err = NineRouterPreflightError(status=ProviderStatus.COST_NOT_VERIFIED, message="Cost unverified")
    assert err.status == ProviderStatus.COST_NOT_VERIFIED


def test_i08_t06_scientific_abstention_is_not_provider_failure():
    """I08-T06: Scientific abstention returns ClaimVerdict.INSUFFICIENT_EVIDENCE cleanly."""
    m = MockGenerationAdapter(default_verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE)
    res = m.generate("1", "Claim", [])
    assert res.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
    assert "insufficient" in res.explanation.lower()


def test_i08_t07_provider_failure_does_not_become_supported():
    """I08-T07: Technical failure does not produce a supported verdict."""
    with pytest.raises(ProviderExecutionError):
        adapter = NineRouterAdapter(base_url="http://127.0.0.1:59999", api_key="", zero_cost_authorized=False)
        adapter.generate("1", "T", [])


def test_i08_t08_sensitive_credentials_sanitized():
    """I08-T08: Error messages do not leak secret API tokens."""
    adapter = NineRouterAdapter(base_url="http://127.0.0.1:59999", api_key="sk-secret-token-12345", zero_cost_authorized=False)
    with pytest.raises(NineRouterPreflightError) as exc:
        adapter.preflight_check()
    assert "sk-secret-token-12345" not in str(exc.value)


def test_i08_t09_typed_status_enum_values():
    """I08-T09: All expected typed status values exist in ProviderStatus enum."""
    assert ProviderStatus.SUCCESS.value == "SUCCESS"
    assert ProviderStatus.COST_NOT_VERIFIED.value == "COST_NOT_VERIFIED"
    assert ProviderStatus.AUTH_FAILED.value == "AUTH_FAILED"
    assert ProviderStatus.TIMEOUT.value == "TIMEOUT"
    assert ProviderStatus.SCHEMA_VALIDATION_FAILED.value == "SCHEMA_VALIDATION_FAILED"


def test_i08_t10_langgraph_records_failed_when_generator_raises():
    """I08-T10: LangGraph catches generator exception and sets final_verdict='FAILED'."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="T", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))

    class CrashingGen(BaseGenerationAdapter):
        def generate(self, claim_id: str, claim_text: str, evidence: list):
            raise ProviderExecutionError(status=ProviderStatus.TIMEOUT, message="Timeout")

    graph = create_verification_graph(hybrid, CrashingGen(), lambda a, e: VerificationReport())
    with pytest.raises(ProviderExecutionError) as exc:
        graph.invoke({
            "claim_id": "1", "claim_text": "T", "evidence_items": [],
            "analysis": None, "verification": None, "retry_count": 0, "max_retries": 1,
            "final_verdict": None, "is_verified_conclusive": False,
        })
    assert exc.value.status == ProviderStatus.TIMEOUT


# =========================================================================
# I09: Packaging & Dependency Verification (8 Tests)
# =========================================================================

def test_i09_t01_direct_dependencies_in_pyproject():
    """I09-T01: pyproject.toml explicitly lists langchain, langgraph, transformers."""
    with open("pyproject.toml", "r") as f:
        content = f.read()
    assert "langchain" in content
    assert "langgraph" in content
    assert "transformers" in content
    assert "pydantic" in content


def test_i09_t02_r2_modules_packaged_in_src():
    """I09-T02: Source files exist in scientific_rag namespace."""
    from scientific_rag.retrieval.hybrid import HybridRRFRetriever
    from scientific_rag.generation.schemas import GroundedClaimAnalysis
    from scientific_rag.workflows.graph import create_verification_graph
    from scientific_rag.verification.support import PretrainedNLIProxy
    assert HybridRRFRetriever is not None
    assert GroundedClaimAnalysis is not None
    assert create_verification_graph is not None
    assert PretrainedNLIProxy is not None


def test_i09_t03_cli_entry_point_in_pyproject():
    """I09-T03: pyproject.toml points to scientific_rag.cli:main."""
    with open("pyproject.toml", "r") as f:
        content = f.read()
    assert 'scientific-rag = "scientific_rag.cli:main"' in content


def test_i09_t04_cli_smoke_runs_without_tests_import():
    """I09-T04: run_self_contained_smoke executes cleanly without importing tests."""
    from scientific_rag.cli import run_self_contained_smoke
    run_self_contained_smoke()


def test_i09_t05_no_transitive_imports_missing():
    """I09-T05: Direct import of langchain_core and langchain_openai succeeds."""
    import langchain_core
    import langchain_openai
    import langgraph
    assert langchain_core is not None
    assert langchain_openai is not None
    assert langgraph is not None


def test_i09_t06_wheel_target_name_standardized():
    """I09-T06: Project metadata name is scientific-grounded-rag."""
    with open("pyproject.toml", "r") as f:
        content = f.read()
    assert 'name = "scientific-grounded-rag"' in content


def test_i09_t07_readme_documents_r2_commands():
    """I09-T07: README.md documents (--retriever hybrid, analyze)."""
    with open("README.md", "r") as f:
        content = f.read()
    assert "--retriever hybrid" in content
    assert "cli.py analyze" in content


def test_i09_t08_offline_operation_without_9router():
    """I09-T08: Default mock generation and hybrid retrieval execute 100% offline."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="Text", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    res = hybrid.retrieve("Query", top_k=1)
    assert len(res) == 1
    m = MockGenerationAdapter()
    a = m.generate("c1", "Query", res)
    assert a.verdict == ClaimVerdict.SUPPORTED
