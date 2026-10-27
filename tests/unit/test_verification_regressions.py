"""Regressions: exact-substring citation text, gold sentence alignment requires verified attribution, safe NLI fallback, gold loader path resolution, safe finalization."""

from pathlib import Path
import pytest

from scientific_rag.common.paths import RAW_DATA_DIR
from scientific_rag.data.models import Chunk, RetrievalResult
from scientific_rag.generation.adapter import (
    BaseGenerationAdapter,
    MockGenerationAdapter,
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
# Strict citation text authenticity
# =========================================================================

def test_exact_source_quotation_is_authentic():
    """Exact source quotation is verified as authentic."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="The clinical trial demonstrates that drug X improves survival.")]
    cit = [Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="The clinical trial demonstrates that drug X improves survival.")]
    v = CitationVerifier()
    res = v.verify_citations(cit, ev)
    assert res[0].is_contextual is True
    assert res[0].is_text_authentic is True


def test_harmless_whitespace_and_casing_authentic():
    """Normal whitespace differences preserve authenticity."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="The clinical trial   demonstrates\n\tthat drug X improves survival.")]
    cit = [Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="the clinical trial demonstrates that drug x improves survival")]
    v = CitationVerifier()
    res = v.verify_citations(cit, ev)
    assert res[0].is_text_authentic is True


def test_negation_reversal_rejected():
    """Removing negation ('did not improve' -> 'did improve') is rejected despite high token overlap."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="The intervention did not improve survival.")]
    cit = [Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="The intervention did improve survival.")]
    v = CitationVerifier()
    res = v.verify_citations(cit, ev)
    # Fuzzy 80% overlap is removed; exact normalized substring fails
    assert res[0].is_text_authentic is False
    assert "not an exact normalized substring" in res[0].notes


def test_changed_number_or_percentage_rejected():
    """Altered numerical fact (45% vs 85%) is rejected."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Overall response rate reached 45% in patients.")]
    cit = [Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Overall response rate reached 85% in patients.")]
    v = CitationVerifier()
    res = v.verify_citations(cit, ev)
    assert res[0].is_text_authentic is False


def test_changed_comparison_direction_rejected():
    """Altered comparison direction ('decreases' vs 'increases') is rejected."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Compound A significantly decreases tumor volume.")]
    cit = [Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Compound A significantly increases tumor volume.")]
    v = CitationVerifier()
    res = v.verify_citations(cit, ev)
    assert res[0].is_text_authentic is False


def test_high_lexical_overlap_modified_meaning_rejected():
    """Paraphrase modifying meaning is rejected as a literal quotation."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="We observed no adverse events during the thirty day evaluation period.")]
    cit = [Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="We observed severe adverse events during the thirty day evaluation period.")]
    v = CitationVerifier()
    res = v.verify_citations(cit, ev)
    assert res[0].is_text_authentic is False


def test_valid_chunk_id_fabricated_text_rejected():
    """Valid chunk ID but completely fabricated snippet rejected."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Actual study text.")]
    cit = [Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Invented study text not in chunk.")]
    v = CitationVerifier()
    res = v.verify_citations(cit, ev)
    assert res[0].is_text_authentic is False


def test_citation_from_another_document_invalid():
    """Chunk belonging to a different document rejected."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Actual study text.")]
    cit = [Citation(parent_doc_id="d2", chunk_id="c1", evidence_text="Actual study text.")]
    v = CitationVerifier()
    res = v.verify_citations(cit, ev)
    assert res[0].is_contextual is False


def test_downstream_nli_uses_authoritative_chunk_text():
    """Semantic NLI verifier evaluates authoritative chunk text, never fabricated text."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Drug X fails to cure patients.")]
    analysis = GroundedClaimAnalysis(
        claim_id="1", input_claim="Drug X cures patients.", verdict=ClaimVerdict.SUPPORTED, explanation="E",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Fabricated text claiming drug X cures.")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(analysis, ev)
    # The citation is invalid because of text mismatch; claim cannot be verified supported
    assert rep.all_citations_valid is False
    assert rep.verdict_supported_by_evidence is False


def test_invalid_quotation_cannot_produce_verified_supported():
    """Any quotation mismatch blocks verified conclusive output."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Authentic chunk content.")]
    analysis = GroundedClaimAnalysis(
        claim_id="1", input_claim="Authentic chunk content.", verdict=ClaimVerdict.SUPPORTED, explanation="E",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Modified chunk content.")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(analysis, ev)
    assert rep.verdict_supported_by_evidence is False


# =========================================================================
# Gold sentence alignment attribution
# =========================================================================

def test_verified_sentence_idx_matches_gold():
    """Correct gold document + verified sentence_idx produces True."""
    qrels = {"10": {"doc_1"}}
    rationales = {"10": {"doc_1": [{"sentences": [3, 4], "label": "SUPPORT"}]}}
    aligner = GoldRationaleAligner(qrels=qrels, rationales=rationales)

    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="t", sentence_idx=3),
        is_valid_structure=True, is_contextual=True, is_text_authentic=True,
    )
    res = aligner.align_citations("10", [cit_res])
    assert res[0].gold_doc_aligned is True
    assert res[0].gold_sentence_aligned is True


def test_verified_sentence_idx_wrong_sentence():
    """Correct gold document + wrong sentence_idx produces False."""
    qrels = {"10": {"doc_1"}}
    rationales = {"10": {"doc_1": [{"sentences": [3, 4], "label": "SUPPORT"}]}}
    aligner = GoldRationaleAligner(qrels=qrels, rationales=rationales)

    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="t", sentence_idx=9),
        is_valid_structure=True, is_contextual=True, is_text_authentic=True,
    )
    res = aligner.align_citations("10", [cit_res])
    assert res[0].gold_doc_aligned is True
    assert res[0].gold_sentence_aligned is False


def test_missing_sentence_idx_returns_none_not_true():
    """Correct gold document + missing sentence_idx returns None (NOT_EVALUABLE), NOT True."""
    qrels = {"10": {"doc_1"}}
    rationales = {"10": {"doc_1": [{"sentences": [3, 4], "label": "SUPPORT"}]}}
    aligner = GoldRationaleAligner(qrels=qrels, rationales=rationales)

    # In previous bug, presence of authoritative_evidence_text made this True!
    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="t", sentence_idx=None),
        is_valid_structure=True, is_contextual=True, is_text_authentic=True,
        authoritative_evidence_text="chunk containing multiple sentences",
    )
    res = aligner.align_citations("10", [cit_res])
    assert res[0].gold_doc_aligned is True
    # Missing sentence attribution must be None, never True
    assert res[0].gold_sentence_aligned is None


def test_non_gold_document_with_sentence_idx():
    """Document not in gold rationales returns False."""
    rationales = {"10": {"doc_other": [{"sentences": [1]}]}}
    aligner = GoldRationaleAligner(qrels={"10": set()}, rationales=rationales)

    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_wrong", chunk_id="c1", evidence_text="t", sentence_idx=1),
        is_valid_structure=True, is_contextual=True, is_text_authentic=True,
    )
    res = aligner.align_citations("10", [cit_res])
    assert res[0].gold_doc_aligned is False
    assert res[0].gold_sentence_aligned is False


def test_missing_claim_annotation_returns_none():
    """Claim absent from gold rationales returns None (NOT_EVALUABLE)."""
    aligner = GoldRationaleAligner(qrels={}, rationales={})
    cit_res = CitationVerificationResult(
        citation=Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="t", sentence_idx=1),
        is_valid_structure=True, is_contextual=True, is_text_authentic=True,
    )
    res = aligner.align_citations("absent_claim_999", [cit_res])
    assert res[0].gold_sentence_aligned is None


def test_evaluable_denominator_excludes_none():
    """Accounting denominator correctly excludes NOT_EVALUABLE sentence cases."""
    from scientific_rag.evaluation.grounding import evaluate_grounded_analyses
    analysis = GroundedClaimAnalysis(
        claim_id="1", input_claim="T", verdict=ClaimVerdict.SUPPORTED, explanation="E",
        citations=[Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="t")],
        model_metadata=ModelMetadata(provider="m", model_name="m"),
        verification=VerificationReport(
            total_citations=1, valid_citations=1,
            citation_details=[
                CitationVerificationResult(
                    citation=Citation(parent_doc_id="doc_1", chunk_id="c1", evidence_text="t"),
                    is_valid_structure=True, is_contextual=True, is_text_authentic=True,
                    gold_sentence_aligned=None,  # Not evaluable
                )
            ]
        )
    )
    summary = evaluate_grounded_analyses([{"status": "COMPLETED", "analysis": analysis}])
    assert summary.gold_sentence_evaluable_count == 0
    assert summary.gold_sentence_overlap_rate == 0.0


# =========================================================================
# Safe NLI failure and fallback policy
# =========================================================================

def test_disabled_neural_returns_not_verifiable():
    """When neural NLI is disabled, evaluate_pair returns NOT_VERIFIABLE, not ENTAILMENT."""
    nli = PretrainedNLIProxy(enable_neural=False)
    v, c = nli.evaluate_pair("Premise text.", "Premise text.")
    # Lexical heuristic cannot return ENTAILMENT to satisfy verification
    assert v == "NOT_VERIFIABLE"
    assert c == 0.0


def test_lexical_diagnostic_separated_with_explicit_name():
    """compute_lexical_diagnostic explicitly tags output as DIAGNOSTIC_."""
    nli = PretrainedNLIProxy(enable_neural=False)
    v, c = nli.compute_lexical_diagnostic("A B C D E", "A B C D E")
    assert v == "DIAGNOSTIC_ENTAILMENT"
    assert c > 0.5


def test_nli_unavailable_cannot_verify_verdict():
    """NOT_VERIFIABLE NLI verdict prevents conclusive verdict_supported_by_evidence."""
    ev = [RetrievalResult(chunk_id="c1", doc_id="d1", score=1.0, rank=1, text="Authentic chunk content.")]
    analysis = GroundedClaimAnalysis(
        claim_id="1", input_claim="Authentic chunk content.", verdict=ClaimVerdict.SUPPORTED, explanation="E",
        citations=[Citation(parent_doc_id="d1", chunk_id="c1", evidence_text="Authentic chunk content.")],
        model_metadata=ModelMetadata(provider="mock", model_name="m"),
    )
    # Neural NLI disabled -> returns NOT_VERIFIABLE
    v = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))
    rep = v.verify(analysis, ev)
    assert rep.all_citations_valid is True
    # Without genuine neural entailment, verdict is not supported
    assert rep.verdict_supported_by_evidence is False


def test_neural_inference_live_cpu_works():
    """When neural NLI is enabled on CPU, genuine entailment/contradiction is returned."""
    nli = PretrainedNLIProxy(enable_neural=True)
    v_ent, c_ent = nli.evaluate_pair("Lipid nanoparticles deliver mRNA.", "Lipid nanoparticles carry mRNA.")
    assert v_ent in ["ENTAILMENT", "NEUTRAL"]

    v_con, c_con = nli.evaluate_pair("Treatment completely fails to improve survival.", "Treatment improves survival.")
    assert v_con == "CONTRADICTION"
    assert c_con > 0.5


def test_unverified_nli_blocks_langgraph_conclusive():
    """LangGraph cannot finalize conclusive SUPPORTED when NLI is NOT_VERIFIABLE."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="Some text", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)
    verifier = MultiLayerEvidenceVerifier(nli_proxy=PretrainedNLIProxy(enable_neural=False))

    graph = create_verification_graph(hybrid, generator, verifier.verify, max_retries=1)
    state = graph.invoke({
        "claim_id": "1", "claim_text": "Claim",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    # Unverified evidence must fall back to INSUFFICIENT_EVIDENCE
    assert state["is_verified_conclusive"] is False
    assert state["final_verdict"] == ClaimVerdict.INSUFFICIENT_EVIDENCE.value


# =========================================================================
# Gold rationale loader determinism
# =========================================================================

_CLAIMS_DEV = RAW_DATA_DIR.parent / "scifact_official" / "data" / "claims_dev.jsonl"
requires_claims_dev = pytest.mark.skipif(
    not _CLAIMS_DEV.exists(),
    reason="official SciFact claims_dev.jsonl missing at data/raw/scifact_official/data/ (external data, not redistributed)",
)


@requires_claims_dev
def test_explicit_valid_path_loads():
    """Explicit valid path loads annotations cleanly."""
    from scientific_rag.common.paths import RAW_DATA_DIR
    from scientific_rag.data.loader import DatasetLoader
    loader = DatasetLoader(RAW_DATA_DIR)
    dev_path = RAW_DATA_DIR.parent / "scifact_official" / "data" / "claims_dev.jsonl"
    if not dev_path.exists():
        dev_path = RAW_DATA_DIR / "scifact_official" / "data" / "claims_dev.jsonl"
    rats = loader.load_gold_rationales(claims_path=dev_path)
    assert len(rats) == 300


def test_explicit_missing_path_raises_filenotfound():
    """Explicit missing path raises FileNotFoundError with actionable message."""
    from scientific_rag.common.paths import RAW_DATA_DIR
    from scientific_rag.data.loader import DatasetLoader
    loader = DatasetLoader(RAW_DATA_DIR)
    with pytest.raises(FileNotFoundError, match="Explicitly requested"):
        loader.load_gold_rationales(claims_path=Path("/tmp/nonexistent_claims.jsonl"))


def test_no_candidate_files_no_name_error():
    """When no candidate files exist, returns empty dict cleanly without NameError."""
    from scientific_rag.data.loader import DatasetLoader
    loader = DatasetLoader(Path("/tmp/empty_dir_no_data"))
    # In previous bug, referenced undefined `fallback` -> NameError
    rats = loader.load_gold_rationales(claims_path=None)
    assert rats == {}


@requires_claims_dev
def test_default_discovery_finds_claims_dev():
    """Default discovery without explicit path finds official claims_dev.jsonl."""
    from scientific_rag.common.paths import RAW_DATA_DIR
    from scientific_rag.data.loader import DatasetLoader
    loader = DatasetLoader(RAW_DATA_DIR)
    rats = loader.load_gold_rationales()
    assert len(rats) == 300
    assert "1" in rats


# =========================================================================
# LangGraph safe finalization contract
# =========================================================================

def test_conclusive_supported_finalizes_supported():
    """Genuinely verified SUPPORTED finalizes as SUPPORTED."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="Exact authentic text", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)

    def v_ok(analysis, ev):
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=True)

    graph = create_verification_graph(hybrid, generator, v_ok, max_retries=1)
    state = graph.invoke({
        "claim_id": "1", "claim_text": "Exact authentic text",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["is_verified_conclusive"] is True
    assert state["final_verdict"] == ClaimVerdict.SUPPORTED.value


def test_conclusive_contradicted_finalizes_contradicted():
    """Genuinely verified CONTRADICTED finalizes as CONTRADICTED."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="Fails text", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.CONTRADICTED)

    def v_con_ok(analysis, ev):
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=True)

    graph = create_verification_graph(hybrid, generator, v_con_ok, max_retries=1)
    state = graph.invoke({
        "claim_id": "1", "claim_text": "Succeeds text",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["is_verified_conclusive"] is True
    assert state["final_verdict"] == ClaimVerdict.CONTRADICTED.value


def test_unsupported_supported_finalizes_insufficient_evidence():
    """Generator proposes SUPPORTED, but verifier rejects -> final_verdict MUST be INSUFFICIENT_EVIDENCE."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="Irrelevant text", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)

    # In previous bug: final_verdict remained "SUPPORTED" even though is_verified_conclusive was False!
    def v_reject(analysis, ev):
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=False)

    graph = create_verification_graph(hybrid, generator, v_reject, max_retries=1)
    state = graph.invoke({
        "claim_id": "1", "claim_text": "Scientific claim",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["retry_count"] == 1
    assert state["is_verified_conclusive"] is False
    # Must never return unverified SUPPORTED to user
    assert state["final_verdict"] == ClaimVerdict.INSUFFICIENT_EVIDENCE.value


def test_unsupported_contradicted_finalizes_insufficient_evidence():
    """Generator proposes CONTRADICTED, but verifier rejects -> final_verdict is INSUFFICIENT_EVIDENCE."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="Irrelevant text", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.CONTRADICTED)

    def v_reject(analysis, ev):
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=False)

    graph = create_verification_graph(hybrid, generator, v_reject, max_retries=1)
    state = graph.invoke({
        "claim_id": "1", "claim_text": "Scientific claim",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["is_verified_conclusive"] is False
    assert state["final_verdict"] == ClaimVerdict.INSUFFICIENT_EVIDENCE.value


def test_retry_succeeds_finalizes_supported():
    """Verification succeeds on attempt 2 -> final_verdict is SUPPORTED."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="T", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))
    generator = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)

    calls = 0
    def v_flip(analysis, ev):
        nonlocal calls
        calls += 1
        if calls == 1:
            return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=False)
        return VerificationReport(all_citations_valid=True, verdict_supported_by_evidence=True)

    graph = create_verification_graph(hybrid, generator, v_flip, max_retries=1)
    state = graph.invoke({
        "claim_id": "1", "claim_text": "S",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["retry_count"] == 1
    assert state["is_verified_conclusive"] is True
    assert state["final_verdict"] == ClaimVerdict.SUPPORTED.value


def test_generator_failure_finalizes_failed():
    """Generator failure leads to FAILED verdict."""
    chunks = [Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, text="T", title="T")]
    hybrid = HybridRRFRetriever(DummyRetriever("b", chunks), DummyRetriever("d", chunks))

    class NullGen(BaseGenerationAdapter):
        def generate(self, claim_id: str, claim_text: str, evidence: list): return None

    graph = create_verification_graph(hybrid, NullGen(), lambda a, e: VerificationReport(), max_retries=1)
    state = graph.invoke({
        "claim_id": "1", "claim_text": "S",
        "evidence_items": [], "analysis": None, "verification": None,
        "retry_count": 0, "max_retries": 1, "final_verdict": None, "is_verified_conclusive": False,
    })
    assert state["is_verified_conclusive"] is False
    assert state["final_verdict"] == "FAILED"
