"""
Pydantic API request and response schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field



class ComponentStatus(BaseModel):
    status: str = Field(description="HEALTHY, DEGRADED, or UNAVAILABLE")
    details: Optional[str] = None


class LiveHealthResponse(BaseModel):
    status: str = "ok"
    service: str = "scientific-grounded-rag"


class ReadyHealthResponse(BaseModel):
    status: str = Field(description="READY or NOT_READY")
    retrieval_backend: str
    components: Dict[str, ComponentStatus]



class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, description="Scientific search query or claim text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    dense_backend: Optional[str] = Field(default=None, description="Optional backend override ('qdrant' or 'file')")


class RetrievedChunkItem(BaseModel):
    chunk_id: str
    doc_id: str
    title: str = ""
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    request_id: str
    query: str
    retrieval_backend: str
    total_results: int
    results: List[RetrievedChunkItem]



class AnalyzeRequest(BaseModel):
    claim_text: str = Field(min_length=1, description="Scientific claim text to verify")
    claim_id: Optional[str] = Field(default=None, description="Optional claim identifier")
    dense_backend: Optional[str] = Field(default=None, description="Optional backend override ('qdrant' or 'file')")


class CitationItem(BaseModel):
    doc_id: str
    chunk_id: str
    sentence_idx: Optional[int] = None
    quote: str


class CitationOutcomeItem(BaseModel):
    citation_index: int
    is_chunk_valid: bool
    is_sentence_valid: bool
    is_ambiguous_sentence: bool
    authoritative_sentence_text: Optional[str] = None
    gold_doc_aligned: Optional[bool] = None
    gold_sentence_aligned: Optional[bool] = None
    nli_verdict: Optional[str] = None
    nli_confidence: Optional[float] = None
    notes: str = ""


class VerificationSummary(BaseModel):
    total_citations: int
    valid_citations: int
    valid_sentence_citations: int
    all_citations_valid: bool
    verdict_supported_by_evidence: bool
    citation_details: List[CitationOutcomeItem] = Field(default_factory=list)


class ModelUsageMetadata(BaseModel):
    provider: str
    model_name: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_status: str = "UNKNOWN"
    provider_requests_count: int = 1


class AnalyzeResponse(BaseModel):
    request_id: str
    claim_id: str
    claim_text: str
    final_verdict: str = Field(description="SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE")
    is_verified_conclusive: bool
    workflow_outcome: str
    explanation: str
    citations: List[CitationItem]
    verification: Optional[VerificationSummary] = None
    retry_count: int = 0
    latency_seconds: float
    model_metadata: Optional[ModelUsageMetadata] = None



class ErrorResponse(BaseModel):
    request_id: str
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
