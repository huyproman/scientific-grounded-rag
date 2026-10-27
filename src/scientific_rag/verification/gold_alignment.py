"""
Gold rationale and document alignment using official SciFact annotations.
Evaluates document-level relevance and sentence-level evidence rationale matches
with explicit NOT_EVALUABLE handling when annotations are absent.
"""

from typing import Any, Dict, List, Optional, Set
from scientific_rag.generation.schemas import Citation, CitationVerificationResult


class GoldRationaleAligner:
    """
    Aligns citations and claims against ground-truth SciFact annotations.
    
    Data structures supported:
    - qrels: Dict[claim_id, Set[doc_id]] for binary/graded document relevance.
    - rationales: Dict[claim_id, Dict[doc_id, List[Dict[str, Any]]]] from SciFact claims_dev.jsonl.
    """

    def __init__(
        self,
        qrels: Optional[Dict[str, Set[str]]] = None,
        rationales: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.qrels = qrels or {}
        self.rationales = rationales or {}

    def align_citations(
        self,
        claim_id: str,
        verification_results: List[CitationVerificationResult],
    ) -> List[CitationVerificationResult]:
        """Enriches CitationVerificationResults with gold document and sentence alignment."""
        gold_docs = self.qrels.get(claim_id, set())
        claim_rationales = self.rationales.get(claim_id)

        has_rationale_annotation = claim_rationales is not None

        for res in verification_results:
            doc_id = res.citation.parent_doc_id

            # Document-level gold overlap
            res.gold_doc_aligned = doc_id in gold_docs

            # Sentence-level gold rationale overlap
            if not has_rationale_annotation:
                # Missing annotation -> NOT_EVALUABLE (None), not false match
                res.gold_sentence_aligned = None
            elif doc_id not in claim_rationales:
                # Document has no evidence for this claim in gold annotations
                res.gold_sentence_aligned = False
            else:
                doc_rat_data = claim_rationales[doc_id]
                gold_sentence_indices: Set[int] = set()

                if isinstance(doc_rat_data, list):
                    for item in doc_rat_data:
                        if isinstance(item, dict) and "sentences" in item:
                            gold_sentence_indices.update(item["sentences"])
                        elif isinstance(item, list):
                            gold_sentence_indices.update(item)
                        elif isinstance(item, int):
                            gold_sentence_indices.add(item)

                # Correctness Invariant:
                # Positive sentence-level alignment REQUIRES an actual verified sentence attribution.
                # A chunk containing multiple sentences cannot be assumed to match gold sentences merely because
                # the document has gold rationales. Absence of validated sentence attribution must yield None (NOT_EVALUABLE).
                if res.citation.sentence_idx is not None:
                    res.gold_sentence_aligned = res.citation.sentence_idx in gold_sentence_indices
                else:
                    # No validated sentence attribution -> NOT_EVALUABLE
                    res.gold_sentence_aligned = None

        return verification_results


class PostHocGoldEvaluator:
    """
    Independent post-hoc evaluator for gold document relevance and sentence rationales.
    Structurally isolated from inference-time generation and graph routing.
    """

    def __init__(
        self,
        qrels: Optional[Dict[str, Set[str]]] = None,
        rationales: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.aligner = GoldRationaleAligner(qrels=qrels, rationales=rationales)

    def evaluate_analysis(self, analysis: Any) -> Any:
        """Enriches an already generated and verified analysis with gold alignment scores post-hoc."""
        if hasattr(analysis, "verification") and analysis.verification and analysis.verification.citation_details:
            self.aligner.align_citations(analysis.claim_id, analysis.verification.citation_details)
            aligned_docs = sum(1 for d in analysis.verification.citation_details if d.gold_doc_aligned)
            analysis.verification.gold_aligned_citations = aligned_docs
        return analysis
