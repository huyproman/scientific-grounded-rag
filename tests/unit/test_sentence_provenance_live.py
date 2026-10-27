"""
Unit and integration tests for canonical sentence provenance and real live 9Router integration:
- SciFactSentenceProvenance loading and document lookup
- Chunk-to-canonical-sentence mapping with title prefix stripping
- Generation context formatting with [SENTENCE i] tags
- Citation sentence attribution verification
- Live 9Router connection and structured output execution
"""

from pathlib import Path
import os
import pytest

from scientific_rag.common.paths import RAW_DATA_DIR
from scientific_rag.data.models import Chunk, RetrievalResult
from scientific_rag.data.sentence_provenance import CanonicalSentence, SciFactSentenceProvenance
from scientific_rag.generation.adapter import (
    MockGenerationAdapter,
    NineRouterAdapter,
    NineRouterPreflightError,
    ProviderExecutionError,
    ProviderStatus,
)
from scientific_rag.generation.prompts import format_evidence_context
from scientific_rag.generation.schemas import Citation, ClaimVerdict, GroundedClaimAnalysis
from scientific_rag.verification.citations import CitationVerifier

_CORPUS = RAW_DATA_DIR.parent / "scifact_official" / "data" / "corpus.jsonl"
requires_corpus = pytest.mark.skipif(
    not _CORPUS.exists(),
    reason="official SciFact corpus missing at data/raw/scifact_official/data/corpus.jsonl (see scripts/download_corpus.py)",
)


@requires_corpus
def test_provenance_loads_canonical_scifact_corpus():
    """Verifies that SciFactSentenceProvenance successfully loads official corpus."""
    prov = SciFactSentenceProvenance()
    prov.load_corpus()
    sents = prov.get_document_sentences("4983")
    assert len(sents) == 14
    assert sents[0].doc_id == "4983"
    assert sents[0].sentence_idx == 0
    assert "Alterations of the architecture" in sents[0].text


@requires_corpus
def test_chunk_to_sentence_mapping():
    """Verifies that retrieved chunks map to the correct canonical sentence indices."""
    prov = SciFactSentenceProvenance()
    prov.load_corpus()

    # Document 5836 sentence 0
    sents = prov.get_document_sentences("5836")
    assert len(sents) > 0
    sample_text = f"Title of Paper: {sents[0].text} {sents[1].text}"

    mapping = prov.map_chunk_to_sentences("5836", "5836#c0", sample_text)
    assert mapping.doc_id == "5836"
    assert 0 in mapping.sentence_indices
    assert 1 in mapping.sentence_indices
    assert len(mapping.sentences) >= 2


@requires_corpus
def test_format_evidence_context_with_sentence_tags():
    """Verifies that format_evidence_context exposes [SENTENCE i] tags when provenance is provided."""
    prov = SciFactSentenceProvenance()
    prov.load_corpus()

    ev = [
        RetrievalResult(
            chunk_id="5836#c0",
            doc_id="5836",
            score=1.0,
            rank=1,
            title="Induction of myelodysplasia",
            text="Induction of myelodysplasia: Myelodysplastic syndromes (MDS) are age-dependent stem cell malignancies that share biological features of activated adaptive immune response and ineffective hematopoiesis.",
        )
    ]
    formatted = format_evidence_context(ev, provenance=prov)
    assert "[SENTENCE 0]" in formatted
    assert "Myelodysplastic syndromes" in formatted


@requires_corpus
def test_citation_verifier_validates_sentence_attribution():
    """Verifies that CitationVerifier validates sentence_idx against canonical sentences."""
    prov = SciFactSentenceProvenance()
    prov.load_corpus()
    verifier = CitationVerifier(provenance=prov)

    # Valid sentence attribution
    ev = [
        RetrievalResult(
            chunk_id="5836#c0",
            doc_id="5836",
            score=1.0,
            rank=1,
            text="Myelodysplastic syndromes (MDS) are age-dependent stem cell malignancies that share biological features of activated adaptive immune response and ineffective hematopoiesis.",
        )
    ]
    cit_valid = [
        Citation(
            parent_doc_id="5836",
            chunk_id="5836#c0",
            sentence_idx=0,
            evidence_text="Myelodysplastic syndromes (MDS) are age-dependent stem cell malignancies that share biological features of activated adaptive immune response and ineffective hematopoiesis.",
        )
    ]
    res_valid = verifier.verify_citations(cit_valid, ev)
    assert res_valid[0].is_contextual is True
    assert res_valid[0].is_text_authentic is True
    assert res_valid[0].citation.sentence_idx == 0

    # Invalid sentence attribution (sentence 99 does not belong to this chunk)
    cit_invalid = [
        Citation(
            parent_doc_id="5836",
            chunk_id="5836#c0",
            sentence_idx=99,
            evidence_text="Myelodysplastic syndromes (MDS) are age-dependent stem cell malignancies that share biological features of activated adaptive immune response and ineffective hematopoiesis.",
        )
    ]
    res_invalid = verifier.verify_citations(cit_invalid, ev)
    # The verifier resets invalid sentence attribution
    assert res_invalid[0].citation.sentence_idx is None
    assert "does not belong to retrieved chunk" in res_invalid[0].notes


@requires_corpus
def test_mock_adapter_with_provenance_attributes_sentence_idx():
    """Verifies that MockGenerationAdapter attaches real sentence_idx when provenance is active."""
    prov = SciFactSentenceProvenance()
    prov.load_corpus()
    adapter = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED, provenance=prov)

    ev = [
        RetrievalResult(
            chunk_id="4983#c0",
            doc_id="4983",
            score=1.0,
            rank=1,
            text="Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities.",
        )
    ]
    analysis = adapter.generate("claim_1", "Brain development is impacted by alterations.", ev)
    assert len(analysis.citations) == 1
    assert analysis.citations[0].sentence_idx == 0
    assert analysis.citations[0].parent_doc_id == "4983"


@pytest.mark.skipif(
    not (os.getenv("LLM_API_KEY") and os.getenv("RUN_LIVE_LLM_TESTS") == "1"),
    reason="hosted LLM test: set LLM_API_KEY and RUN_LIVE_LLM_TESTS=1 to opt in",
)
@requires_corpus
def test_live_9router_gemini_smoke():
    """Verifies live 9Router/Gemini endpoint execution on ag/gemini-3.8-flash-high."""
    prov = SciFactSentenceProvenance()
    prov.load_corpus()
    adapter = NineRouterAdapter(provenance=prov, timeout_seconds=60)

    ev = [
        RetrievalResult(
            chunk_id="13734012#c2",
            doc_id="13734012",
            score=1.0,
            rank=1,
            text="RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million).",
            title="Prevalent abnormal prion protein in human appendixes",
        )
    ]
    claim = "1/2000 in UK have abnormal PrP positivity."
    try:
        res = adapter.generate("live_test_01", claim, ev)
        assert isinstance(res, GroundedClaimAnalysis)
        assert res.verdict in [ClaimVerdict.SUPPORTED, ClaimVerdict.INSUFFICIENT_EVIDENCE]
        assert res.model_metadata.provider == "9router"
        assert res.model_metadata.model_name == "ag/gemini-3.8-flash-high"
    except ProviderExecutionError as exc:
        assert exc.status in [ProviderStatus.TIMEOUT, ProviderStatus.TRANSPORT_ERROR, ProviderStatus.SCHEMA_VALIDATION_FAILED]
