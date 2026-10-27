"""
Evaluation metrics and benchmark runner for evidence-grounded generation and citation verification.
Implements accounting across ATTEMPTED, SCHEMA_VALID/INVALID, PROVIDER_FAILED, NOT_EVALUABLE.
Explicitly distinguishes mock pipeline evaluation from real LLM generation quality.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json

from scientific_rag.generation.schemas import ClaimVerdict, GroundedClaimAnalysis


class EvaluationCaseStatus(str, Enum):
    """Categorical accounting status for an attempted claim evaluation."""
    ATTEMPTED = "ATTEMPTED"
    COMPLETED = "COMPLETED"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass
class GroundingEvaluationSummary:
    """Detailed metrics summary for grounded claim generation evaluation."""
    total_attempted: int = 0
    total_completed: int = 0
    evaluation_mode: str = "MOCK"  # "MOCK" or "REAL_PROVIDER"

    schema_valid_count: int = 0
    schema_invalid_count: int = 0
    schema_valid_rate: float = 0.0

    provider_failure_count: int = 0
    provider_failure_rate: float = 0.0

    total_citations_generated: int = 0
    valid_citations_count: int = 0
    # Structural resolvability only: cited chunk exists in retrieved context, belongs to cited doc, quote is a substring.
    # NOT evidence support and NOT factual verification; the key name is kept for artifact compatibility.
    citation_validity_rate: float = 0.0

    unsupported_citation_count: int = 0
    unsupported_citation_rate: float = 0.0

    gold_doc_aligned_count: int = 0
    gold_doc_overlap_rate: float = 0.0

    gold_sentence_evaluable_count: int = 0
    gold_sentence_aligned_count: int = 0
    gold_sentence_overlap_rate: float = 0.0

    valid_chunk_citations_count: int = 0
    valid_sentence_citations_count: int = 0
    ambiguous_sentence_citations_count: int = 0
    invalid_sentence_citations_count: int = 0
    semantically_consistent_citations_count: int = 0
    gold_sentence_precision: float = 0.0

    nli_evaluable_count: int = 0
    nli_entailed_count: int = 0
    nli_contradicted_count: int = 0
    nli_entailed_rate: float = 0.0

    supported_verdict_count: int = 0
    contradicted_verdict_count: int = 0
    insufficient_evidence_count: int = 0
    abstention_rate: float = 0.0


def evaluate_grounded_analyses(
    records: List[Dict[str, Any]],
    evaluation_mode: str = "MOCK",
) -> GroundingEvaluationSummary:
    """
    Computes grounding evaluation metrics over a population of evaluation records.
    Explicitly accounts for:
    - total_attempted queries in denominator (including schema/parse failures and provider failures).
    - separates ID/context validity from semantic unsupportedness.
    - treats absent annotations as NOT_EVALUABLE, not false negative.
    """
    total_attempted = len(records)
    if total_attempted == 0:
        return GroundingEvaluationSummary(evaluation_mode=evaluation_mode)

    completed = 0
    schema_valid = 0
    schema_invalid = 0
    provider_failures = 0

    total_citations = 0
    valid_citations = 0
    unsupported_citations = 0

    gold_doc_aligned = 0
    gold_sent_evaluable = 0
    gold_sent_aligned = 0

    nli_evaluable = 0
    nli_entailed = 0
    nli_contradicted = 0

    supported_cnt = 0
    contradicted_cnt = 0
    insufficient_cnt = 0

    for rec in records:
        status = rec.get("status", EvaluationCaseStatus.COMPLETED.value)
        analysis: Optional[GroundedClaimAnalysis] = rec.get("analysis")

        if status == EvaluationCaseStatus.PROVIDER_FAILED.value:
            provider_failures += 1
            continue
        elif status == EvaluationCaseStatus.SCHEMA_INVALID.value:
            schema_invalid += 1
            continue

        if analysis is None:
            schema_invalid += 1
            continue

        completed += 1
        schema_valid += 1

        if analysis.verdict == ClaimVerdict.SUPPORTED:
            supported_cnt += 1
        elif analysis.verdict == ClaimVerdict.CONTRADICTED:
            contradicted_cnt += 1
        elif analysis.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE:
            insufficient_cnt += 1

        cit_count = len(analysis.citations)
        total_citations += cit_count

        if analysis.verification:
            v = analysis.verification
            valid_citations += v.valid_citations

            for d in v.citation_details:
                # Verdict-aware semantic support vs unsupportedness:
                # A citation is unsupported if:
                # 1. Structure or context or text is invalid/unauthentic, OR
                # 2. For SUPPORTED: NLI is CONTRADICTION, NEUTRAL, or NOT_VERIFIABLE, OR
                # 3. For CONTRADICTED: NLI is ENTAILMENT, NEUTRAL, or NOT_VERIFIABLE, OR
                # 4. For INSUFFICIENT_EVIDENCE: citations attached to non-grounded claims are considered unsupported.
                if not d.is_contextual or not d.is_text_authentic:
                    unsupported_citations += 1
                elif analysis.verdict == ClaimVerdict.SUPPORTED and d.nli_verdict in ["CONTRADICTION", "NEUTRAL", "NOT_VERIFIABLE", None]:
                    unsupported_citations += 1
                elif analysis.verdict == ClaimVerdict.CONTRADICTED and d.nli_verdict in ["ENTAILMENT", "NEUTRAL", "NOT_VERIFIABLE", None]:
                    unsupported_citations += 1
                elif analysis.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE:
                    unsupported_citations += 1

                if d.gold_doc_aligned:
                    gold_doc_aligned += 1

                if d.gold_sentence_aligned is not None:
                    gold_sent_evaluable += 1
                    if d.gold_sentence_aligned:
                        gold_sent_aligned += 1

                if d.nli_verdict not in [None, "NOT_VERIFIABLE"]:
                    nli_evaluable += 1
                    if d.nli_verdict == "ENTAILMENT":
                        nli_entailed += 1
                    elif d.nli_verdict == "CONTRADICTION":
                        nli_contradicted += 1

    cit_denom = max(total_citations, 1)

    return GroundingEvaluationSummary(
        total_attempted=total_attempted,
        total_completed=completed,
        evaluation_mode=evaluation_mode,
        schema_valid_count=schema_valid,
        schema_invalid_count=schema_invalid,
        schema_valid_rate=schema_valid / total_attempted,
        provider_failure_count=provider_failures,
        provider_failure_rate=provider_failures / total_attempted,
        total_citations_generated=total_citations,
        valid_citations_count=valid_citations,
        citation_validity_rate=valid_citations / cit_denom if total_citations > 0 else 1.0,
        unsupported_citation_count=unsupported_citations,
        unsupported_citation_rate=unsupported_citations / cit_denom if total_citations > 0 else 0.0,
        gold_doc_aligned_count=gold_doc_aligned,
        gold_doc_overlap_rate=gold_doc_aligned / cit_denom if total_citations > 0 else 0.0,
        gold_sentence_evaluable_count=gold_sent_evaluable,
        gold_sentence_aligned_count=gold_sent_aligned,
        gold_sentence_overlap_rate=gold_sent_aligned / max(gold_sent_evaluable, 1) if gold_sent_evaluable > 0 else 0.0,
        valid_chunk_citations_count=valid_citations,
        valid_sentence_citations_count=gold_sent_evaluable,
        gold_sentence_precision=gold_sent_aligned / max(gold_sent_evaluable, 1) if gold_sent_evaluable > 0 else 0.0,
        nli_evaluable_count=nli_evaluable,
        nli_entailed_count=nli_entailed,
        nli_contradicted_count=nli_contradicted,
        nli_entailed_rate=nli_entailed / max(nli_evaluable, 1) if nli_evaluable > 0 else 0.0,
        supported_verdict_count=supported_cnt,
        contradicted_verdict_count=contradicted_cnt,
        insufficient_evidence_count=insufficient_cnt,
        abstention_rate=insufficient_cnt / total_attempted,
    )
