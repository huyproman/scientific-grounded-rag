"""
Unified pipeline coordinator executing citation validity, gold alignment, and semantic support.
Builds a VerificationReport including verdict-evidence consistency.
"""

from typing import Any, Dict, List, Optional, Set
from scientific_rag.data.models import RetrievalResult
from scientific_rag.generation.schemas import (
    ClaimVerdict,
    GroundedClaimAnalysis,
    VerificationOutcome,
    VerificationReport,
)
from scientific_rag.verification.citations import CitationVerifier
from scientific_rag.verification.gold_alignment import GoldRationaleAligner
from scientific_rag.verification.support import PretrainedNLIProxy


class MultiLayerEvidenceVerifier:
    """
    Coordinates three-layer scientific verification:
    Layer 1: Citation Validity & Authoritative Source Text Resolution.
    Layer 2: Gold Rationale & Document Alignment on SciFact.
    Layer 3: Pretrained CPU NLI Semantic Support & Contradiction Check.
    """

    def __init__(
        self,
        qrels: Optional[Dict[str, Set[str]]] = None,
        rationales: Optional[Dict[str, Dict[str, Any]]] = None,
        nli_proxy: Optional[PretrainedNLIProxy] = None,
        provenance: Optional[Any] = None,
    ):
        self.citation_verifier = CitationVerifier(provenance=provenance)
        self.gold_aligner = GoldRationaleAligner(qrels=qrels, rationales=rationales)
        self.nli_proxy = nli_proxy or PretrainedNLIProxy(enable_neural=True)

    def verify(
        self,
        analysis: GroundedClaimAnalysis,
        evidence_items: List[RetrievalResult],
    ) -> VerificationReport:
        """Runs the full three-layer verification suite on a generated claim analysis."""
        citations = analysis.citations

        if not citations:
            is_insufficient = analysis.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
            return VerificationReport(
                total_citations=0,
                valid_citations=0,
                valid_chunk_citations=0,
                valid_sentence_citations=0,
                ambiguous_sentence_citations=0,
                invalid_sentence_citations=0,
                semantically_consistent_citations=0,
                hallucinated_citations=0,
                gold_aligned_citations=0,
                nli_entailed_citations=0,
                # For backward compatibility with tests expecting True on clean zero-citation abstention:
                all_citations_valid=is_insufficient,
                all_citations_sentence_valid=is_insufficient,
                verdict_supported_by_evidence=is_insufficient,
                outcomes=[VerificationOutcome.INSUFFICIENT_EVIDENCE],
                citation_details=[],
            )

        details = self.citation_verifier.verify_citations(citations, evidence_items)

        if self.gold_aligner is not None:
            details = self.gold_aligner.align_citations(analysis.claim_id, details)

        details = self.nli_proxy.verify_semantic_support(analysis.input_claim, details)

        total = len(details)
        valid_chunk = sum(1 for d in details if d.is_chunk_valid)
        valid_sent = sum(1 for d in details if d.is_sentence_valid)
        ambiguous_sent = sum(1 for d in details if d.is_ambiguous_sentence)
        invalid_sent = sum(1 for d in details if d.sentence_evaluable and not d.is_sentence_valid and not d.is_ambiguous_sentence)
        hallucinated = total - valid_chunk

        gold_aligned = sum(1 for d in details if d.gold_doc_aligned)
        nli_entailed = sum(1 for d in details if d.nli_verdict == "ENTAILMENT")
        nli_contradicted = sum(1 for d in details if d.nli_verdict == "CONTRADICTION")

        for d in details:
            if analysis.verdict == ClaimVerdict.SUPPORTED:
                d.is_semantically_consistent = bool(d.nli_verdict == "ENTAILMENT" and d.is_text_authentic and d.is_contextual)
            elif analysis.verdict == ClaimVerdict.CONTRADICTED:
                d.is_semantically_consistent = bool(d.nli_verdict == "CONTRADICTION" and d.is_text_authentic and d.is_contextual)
            else:
                d.is_semantically_consistent = False

        consistent_citations = sum(1 for d in details if d.is_semantically_consistent is True)

        outcomes: List[VerificationOutcome] = []
        if hallucinated > 0:
            outcomes.append(VerificationOutcome.INVALID_CITATION)
        else:
            outcomes.append(VerificationOutcome.VALID_CITATION)

        if valid_sent > 0:
            outcomes.append(VerificationOutcome.VALID_SENTENCE_ATTRIBUTION)
        if ambiguous_sent > 0:
            outcomes.append(VerificationOutcome.AMBIGUOUS_SENTENCE_ATTRIBUTION)
        if invalid_sent > 0:
            outcomes.append(VerificationOutcome.INVALID_SENTENCE_ATTRIBUTION)

        if gold_aligned > 0:
            outcomes.append(VerificationOutcome.GOLD_ALIGNED)
        else:
            outcomes.append(VerificationOutcome.NOT_GOLD_ALIGNED)

        if nli_entailed > 0:
            outcomes.append(VerificationOutcome.ENTAILMENT_SUPPORTED)
        if nli_contradicted > 0:
            outcomes.append(VerificationOutcome.CONTRADICTED_BY_EVIDENCE)

        if consistent_citations > 0:
            outcomes.append(VerificationOutcome.SEMANTICALLY_CONSISTENT)
        elif total > 0:
            outcomes.append(VerificationOutcome.SEMANTICALLY_INCONSISTENT)

        all_valid = (hallucinated == 0 and total > 0)
        all_sent_valid = (valid_sent == total and total > 0)

        # Verdict-Evidence Consistency Policy
        verdict_supported = False
        if all_valid:
            if analysis.verdict == ClaimVerdict.SUPPORTED:
                # Must have at least one entailment and NO contradictory citations
                verdict_supported = (nli_entailed > 0 and nli_contradicted == 0)
            elif analysis.verdict == ClaimVerdict.CONTRADICTED:
                # Must have at least one contradiction and NO entailing citations
                verdict_supported = (nli_contradicted > 0 and nli_entailed == 0)
            elif analysis.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE:
                verdict_supported = False

        return VerificationReport(
            total_citations=total,
            valid_citations=valid_chunk,
            valid_chunk_citations=valid_chunk,
            valid_sentence_citations=valid_sent,
            ambiguous_sentence_citations=ambiguous_sent,
            invalid_sentence_citations=invalid_sent,
            semantically_consistent_citations=consistent_citations,
            hallucinated_citations=hallucinated,
            gold_aligned_citations=gold_aligned,
            nli_entailed_citations=nli_entailed,
            all_citations_valid=all_valid,
            all_citations_sentence_valid=all_sent_valid,
            verdict_supported_by_evidence=verdict_supported,
            outcomes=outcomes,
            citation_details=details,
        )
