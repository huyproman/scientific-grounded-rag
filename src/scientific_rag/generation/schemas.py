"""
Pydantic schemas and typed data models for structured evidence-grounded scientific claim analysis.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClaimVerdict(str, Enum):
    """Authoritative verdict for a scientific claim against retrieved evidence."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class Citation(BaseModel):
    """Represents a specific citation to an evidence chunk."""
    parent_doc_id: str = Field(description="Parent document ID (e.g., SciFact corpus doc ID)")
    chunk_id: str = Field(description="Specific chunk ID cited from the retrieval context")
    evidence_text: str = Field(description="Exact snippet or sentence from the cited chunk")
    relationship: str = Field(
        default="supports",
        description="Relationship of cited evidence to statement: supports, contradicts, or contextual",
    )
    sentence_idx: Optional[int] = Field(
        default=None,
        description="0-based sentence index within parent document if sentence-level attribution is resolved",
    )


class VerificationOutcome(str, Enum):
    """Categorical outcomes for automated verification."""
    VALID_CITATION = "VALID_CITATION"
    INVALID_CITATION = "INVALID_CITATION"
    GOLD_ALIGNED = "GOLD_ALIGNED"
    NOT_GOLD_ALIGNED = "NOT_GOLD_ALIGNED"
    ENTAILMENT_SUPPORTED = "ENTAILMENT_SUPPORTED"
    CONTRADICTED_BY_EVIDENCE = "CONTRADICTED_BY_EVIDENCE"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS_SENTENCE_ATTRIBUTION = "AMBIGUOUS_SENTENCE_ATTRIBUTION"
    INVALID_SENTENCE_ATTRIBUTION = "INVALID_SENTENCE_ATTRIBUTION"
    VALID_SENTENCE_ATTRIBUTION = "VALID_SENTENCE_ATTRIBUTION"
    SEMANTICALLY_CONSISTENT = "SEMANTICALLY_CONSISTENT"
    SEMANTICALLY_INCONSISTENT = "SEMANTICALLY_INCONSISTENT"


class WorkflowOutcome(str, Enum):
    """Categorical workflow outcome for LangGraph state machine."""
    VERIFIED_SUPPORTED = "VERIFIED_SUPPORTED"
    VERIFIED_CONTRADICTED = "VERIFIED_CONTRADICTED"
    SCIENTIFIC_ABSTENTION = "SCIENTIFIC_ABSTENTION"
    UNVERIFIED_CONCLUSION = "UNVERIFIED_CONCLUSION"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"


class CitationVerificationResult(BaseModel):
    """Verification details for a single citation with multidimensional validity."""
    citation: Citation
    is_valid_structure: bool = Field(description="True if parent_doc_id and chunk_id exist in context")
    is_contextual: bool = Field(description="True if chunk was actually supplied in the retrieved context")
    is_text_authentic: bool = Field(default=True, description="True if cited text matches authoritative retrieved chunk")
    is_chunk_valid: bool = Field(default=False, description="True if chunk exists and quotation is authentic")
    is_sentence_valid: bool = Field(default=False, description="True if sentence_idx is verified against canonical sentence uniquely")
    is_ambiguous_sentence: bool = Field(default=False, description="True if quotation matches multiple canonical sentences in chunk/document")
    sentence_evaluable: bool = Field(default=False, description="True if sentence attribution was attempted or evaluable")
    is_semantically_consistent: Optional[bool] = Field(default=None, description="True if NLI relation is consistent with the claimed verdict")
    authoritative_evidence_text: Optional[str] = Field(default=None, description="Canonical text from retrieved chunk")
    authoritative_sentence_text: Optional[str] = Field(default=None, description="Exact canonical sentence text if sentence attribution resolved")
    claimed_sentence_idx: Optional[int] = Field(default=None, description="Original claimed sentence index before verification")
    gold_doc_aligned: Optional[bool] = Field(
        default=None, description="True if parent_doc_id matches gold annotation"
    )
    gold_sentence_aligned: Optional[bool] = Field(
        default=None, description="True if cited sentence/chunk matches gold rationale span"
    )
    nli_verdict: Optional[str] = Field(
        default=None, description="ENTAILMENT, CONTRADICTION, or NEUTRAL from NLI proxy"
    )
    nli_confidence: Optional[float] = Field(default=None, description="NLI proxy confidence score")
    notes: str = Field(default="", description="Explanatory notes or error message")


class VerificationReport(BaseModel):
    """Aggregate verification assessment over all citations in an analysis."""
    total_citations: int = 0
    valid_citations: int = 0
    valid_chunk_citations: int = 0
    valid_sentence_citations: int = 0
    ambiguous_sentence_citations: int = 0
    invalid_sentence_citations: int = 0
    semantically_consistent_citations: int = 0
    hallucinated_citations: int = 0
    gold_aligned_citations: int = 0
    nli_entailed_citations: int = 0
    all_citations_valid: bool = False
    all_citations_sentence_valid: bool = False
    verdict_supported_by_evidence: bool = False
    outcomes: List[VerificationOutcome] = Field(default_factory=list)
    citation_details: List[CitationVerificationResult] = Field(default_factory=list)


class CostStatus(str, Enum):
    """Cost verification status for model inference execution."""
    REPORTED = "REPORTED"
    DERIVED_FROM_VERIFIED_RATE = "DERIVED_FROM_VERIFIED_RATE"
    NOT_REPORTED = "NOT_REPORTED"
    UNKNOWN = "UNKNOWN"


class ProviderRequestEvent(BaseModel):
    """Immutable audit record for an individual provider inference request."""
    request_id: str = Field(default="", description="Unique or correlation request identifier")
    workflow_run_id: str = Field(default="", description="Parent workflow run identifier")
    claim_id: str = Field(default="", description="Claim ID being processed")
    attempt_index: int = Field(default=0, description="0-based workflow generation attempt index")
    request_index: int = Field(default=0, description="0-based request index within the attempt")
    model_requested: str = Field(default="", description="Requested model identifier")
    model_responded: Optional[str] = Field(default=None, description="Model identifier returned in response")
    request_outcome: str = Field(default="SUCCESS", description="SUCCESS, TIMEOUT, SCHEMA_FAILED, AUTH_FAILED, TRANSPORT_ERROR")
    latency_seconds: float = Field(default=0.0, description="Provider round-trip HTTP latency")
    prompt_tokens: Optional[int] = Field(default=None)
    completion_tokens: Optional[int] = Field(default=None)
    total_tokens: Optional[int] = Field(default=None)
    cost_status: CostStatus = Field(default=CostStatus.UNKNOWN)
    cost_amount: Optional[float] = Field(default=None)
    cost_currency: Optional[str] = Field(default=None)
    error_class: Optional[str] = Field(default=None)


class ModelMetadata(BaseModel):
    """Metadata detailing the generator provider, model, tokens, and cost verification."""
    provider: str = Field(description="Provider name (e.g. 9router, mock)")
    model_name: str = Field(description="Exact model identifier (e.g. ag/gemini-3.8-flash-high)")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens consumed if reported")
    prompt_tokens: Optional[int] = Field(default=None)
    completion_tokens: Optional[int] = Field(default=None)
    latency_seconds: Optional[float] = Field(default=None)
    is_zero_cost_verified: bool = Field(
        default=False, description="Deprecated: True only for offline mock; False for paid live inference"
    )
    cost_status: CostStatus = Field(
        default=CostStatus.UNKNOWN, description="Truthful billing/cost status: REPORTED, NOT_REPORTED, or UNKNOWN"
    )
    cost_amount: Optional[float] = Field(default=None, description="Monetary cost if reported by provider")
    cost_currency: Optional[str] = Field(default=None, description="Billing currency e.g. USD")
    cost_source: Optional[str] = Field(default=None, description="Provenance of cost data")
    provider_requests_count: int = Field(default=1, description="Actual provider HTTP requests executed")
    request_events: List[ProviderRequestEvent] = Field(default_factory=list, description="Audit log of provider HTTP requests")


class GroundedClaimAnalysis(BaseModel):
    """
    Authoritative structured schema for scientific evidence-grounded claim analysis.
    Distinguishes retrieval relevance, attribution correctness, gold alignment, and semantic support.
    """
    claim_id: str = Field(description="Unique claim identifier (e.g. SciFact query ID)")
    input_claim: str = Field(description="Original scientific claim text")
    verdict: ClaimVerdict = Field(
        description="Verdict against retrieved evidence: SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE"
    )
    explanation: str = Field(
        description="Grounded scientific explanation synthesizing the retrieved evidence"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of explicit citations attributing assertions to specific evidence chunks",
    )
    verification: Optional[VerificationReport] = Field(
        default=None,
        description="Independent multi-layer verification report evaluated after generation",
    )
    model_metadata: ModelMetadata = Field(
        description="Metadata on the generator and inference cost verification"
    )
