"""
LangChain provider adapter supporting Mock deterministic generator and OpenAI-compatible 9Router gateway.
Remote calls are opt-in (PAID_INFERENCE_AUTHORIZED) and provider failures are typed.
"""

from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from pydantic import BaseModel, Field

from scientific_rag.data.models import RetrievalResult
from scientific_rag.generation.prompts import create_grounding_prompt, format_evidence_context
from scientific_rag.generation.schemas import (
    Citation,
    ClaimVerdict,
    CostStatus,
    GroundedClaimAnalysis,
    ModelMetadata,
    ProviderRequestEvent,
)

logger = logging.getLogger("scientific_rag.generation.adapter")


class ProviderStatus(str, Enum):
    """Explicit typed status for provider inference execution."""
    SUCCESS = "SUCCESS"
    COST_NOT_VERIFIED = "COST_NOT_VERIFIED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    AUTH_FAILED = "AUTH_FAILED"
    TIMEOUT = "TIMEOUT"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"


class GenerationOutputSchema(BaseModel):
    """Intermediary structured Pydantic schema requested from the LLM."""
    verdict: ClaimVerdict = Field(
        description="Verdict against retrieved evidence: SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE"
    )
    explanation: str = Field(
        description="Grounded scientific explanation synthesizing the retrieved evidence"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Explicit citations attributing assertions to specific evidence chunks",
    )


class ProviderExecutionError(Exception):
    """Base exception for provider technical execution failures."""
    def __init__(self, status: ProviderStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class NineRouterPreflightError(ProviderExecutionError):
    """Raised when 9Router endpoint reachability or the opt-in check fails."""
    pass


class BaseGenerationAdapter(ABC):
    """Abstract interface for scientific claim generation adapters."""

    @abstractmethod
    def generate(
        self,
        claim_id: str,
        claim_text: str,
        evidence: List[RetrievalResult],
    ) -> GroundedClaimAnalysis:
        """Generates structured GroundedClaimAnalysis for a scientific claim."""
        pass


class MockGenerationAdapter(BaseGenerationAdapter):
    """
    Deterministic mock generator for offline testing, regression, and CI.
    Produces predictable outputs matching or challenging evidence without external API calls.
    Explicitly tags model metadata with provider='mock'.
    """

    def __init__(
        self,
        default_verdict: ClaimVerdict = ClaimVerdict.SUPPORTED,
        force_hallucinated_citation: bool = False,
        force_fabricated_evidence_text: bool = False,
        force_empty_citations: bool = False,
        force_reversed_negation: bool = False,
        provenance: Optional[Any] = None,
    ):
        self.default_verdict = default_verdict
        self.force_hallucinated_citation = force_hallucinated_citation
        self.force_fabricated_evidence_text = force_fabricated_evidence_text
        self.force_empty_citations = force_empty_citations
        self.force_reversed_negation = force_reversed_negation
        self.provenance = provenance

    def generate(
        self,
        claim_id: str,
        claim_text: str,
        evidence: List[RetrievalResult],
    ) -> GroundedClaimAnalysis:
        start_time = time.time()

        if not evidence or self.default_verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE:
            lat = time.time() - start_time
            return GroundedClaimAnalysis(
                claim_id=claim_id,
                input_claim=claim_text,
                verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE,
                explanation="The retrieved evidence context contains insufficient information to verify or refute this claim.",
                citations=[],
                model_metadata=ModelMetadata(
                    provider="mock",
                    model_name="mock-deterministic-v1",
                    total_tokens=42,
                    prompt_tokens=30,
                    completion_tokens=12,
                    latency_seconds=lat,
                    is_zero_cost_verified=True,
                    cost_status=CostStatus.REPORTED,
                    cost_amount=0.0,
                    provider_requests_count=1,
                    request_events=[
                        ProviderRequestEvent(
                            claim_id=claim_id,
                            attempt_index=0,
                            request_index=0,
                            model_requested="mock-deterministic-v1",
                            model_responded="mock-deterministic-v1",
                            request_outcome="SUCCESS",
                            latency_seconds=lat,
                            prompt_tokens=30,
                            completion_tokens=12,
                            total_tokens=42,
                            cost_status=CostStatus.REPORTED,
                            cost_amount=0.0,
                        )
                    ],
                ),
            )

        citations: List[Citation] = []
        if not self.force_empty_citations:
            if self.force_hallucinated_citation:
                citations.append(
                    Citation(
                        parent_doc_id="hallucinated_doc_9999",
                        chunk_id="hallucinated_chunk_9999",
                        evidence_text="Fabricated evidence not present in context.",
                        relationship="supports",
                    )
                )
            else:
                for chunk in evidence[:2]:
                    if self.force_fabricated_evidence_text:
                        ev_text = "Completely fabricated text that never appeared in the chunk."
                    elif self.force_reversed_negation:
                        ev_text = "The study definitively fails and does not produce the claimed effect."
                    else:
                        ev_text = chunk.text

                    sent_idx = None
                    if getattr(self, "provenance", None) is not None:
                        mapping = self.provenance.map_chunk_to_sentences(chunk.doc_id, chunk.chunk_id, chunk.text)
                        if mapping.sentence_indices:
                            sent_idx = mapping.sentence_indices[0]
                            ev_text = mapping.sentences[0].text

                    citations.append(
                        Citation(
                            parent_doc_id=chunk.doc_id,
                            chunk_id=chunk.chunk_id,
                            sentence_idx=sent_idx,
                            evidence_text=ev_text,
                            relationship="supports" if self.default_verdict == ClaimVerdict.SUPPORTED else "contradicts",
                        )
                    )

        explanation = (
            f"Based on evidence from {len(citations)} source chunks, the claim is evaluated as "
            f"{self.default_verdict.value}."
        )

        latency = time.time() - start_time
        return GroundedClaimAnalysis(
            claim_id=claim_id,
            input_claim=claim_text,
            verdict=self.default_verdict,
            explanation=explanation,
            citations=citations,
            model_metadata=ModelMetadata(
                provider="mock",
                model_name="mock-deterministic-v1",
                total_tokens=150,
                prompt_tokens=100,
                completion_tokens=50,
                latency_seconds=latency,
                is_zero_cost_verified=True,
                cost_status=CostStatus.REPORTED,
                cost_amount=0.0,
                provider_requests_count=1,
                request_events=[
                    ProviderRequestEvent(
                        claim_id=claim_id,
                        attempt_index=0,
                        request_index=0,
                        model_requested="mock-deterministic-v1",
                        model_responded="mock-deterministic-v1",
                        request_outcome="SUCCESS",
                        latency_seconds=latency,
                        prompt_tokens=100,
                        completion_tokens=50,
                        total_tokens=150,
                        cost_status=CostStatus.REPORTED,
                        cost_amount=0.0,
                    )
                ],
            ),
        )


class NineRouterAdapter(BaseGenerationAdapter):
    """
    LangChain-powered adapter targeting local OpenAI-compatible 9Router gateway.
    Remote inference is blocked unless PAID_INFERENCE_AUTHORIZED is true; preflight validates key, endpoint and model route.
    Records every provider request event across all successes and failures.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: int = 30,
        paid_inference_authorized: Optional[bool] = None,
        zero_cost_authorized: Optional[bool] = None,
        provenance: Optional[Any] = None,
        request_sink: Optional[Any] = None,
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
        self.request_sink = request_sink

        # An explicit empty string is respected (negative-auth tests).
        resolved_key = api_key if api_key is not None else os.getenv("LLM_API_KEY") or ""

        self.api_key = resolved_key
        self.model_name = model_name or os.getenv("LLM_MODEL") or "ag/gemini-3.8-flash-high"
        self.timeout_seconds = timeout_seconds
        self.provenance = provenance

        # `zero_cost_authorized` is a legacy alias. Remote inference is opt-in:
        # without an explicit argument or PAID_INFERENCE_AUTHORIZED=true it stays blocked.
        if zero_cost_authorized is not None:
            paid_inference_authorized = zero_cost_authorized
        elif paid_inference_authorized is None:
            paid_inference_authorized = os.getenv("PAID_INFERENCE_AUTHORIZED", "false").lower() in ("1", "true", "yes")
        self.paid_inference_authorized = paid_inference_authorized
        self.zero_cost_authorized = paid_inference_authorized

        self._llm = None

    def preflight_check(self) -> Dict[str, Any]:
        """
        Preflight, in order: API key present, inference opted in (no network request is made
        before this), endpoint reachable via /v1/models, requested model listed.
        """
        if not self.api_key:
            raise NineRouterPreflightError(
                status=ProviderStatus.AUTH_FAILED,
                message="Missing LLM_API_KEY for 9Router gateway authentication. Set LLM_API_KEY environment variable.",
            )

        if not self.paid_inference_authorized:
            raise NineRouterPreflightError(
                status=ProviderStatus.COST_NOT_VERIFIED,
                message=(
                    "Remote inference is disabled by default because the endpoint may bill per request. "
                    "Set PAID_INFERENCE_AUTHORIZED=true to opt in."
                ),
            )

        models_url = f"{self.base_url}/v1/models"
        req = urllib.request.Request(models_url)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("User-Agent", "ScientificRAG-Preflight/1.0")

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status != 200:
                    raise NineRouterPreflightError(
                        status=ProviderStatus.TRANSPORT_ERROR,
                        message=f"Gateway returned HTTP {response.status}",
                    )
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in [401, 403]:
                raise NineRouterPreflightError(
                    status=ProviderStatus.AUTH_FAILED,
                    message=f"Authentication failed on 9Router gateway (HTTP {e.code}).",
                )
            raise NineRouterPreflightError(
                status=ProviderStatus.TRANSPORT_ERROR,
                message=f"HTTP error on gateway {models_url}: {e}",
            )
        except urllib.error.URLError as e:
            raise NineRouterPreflightError(
                status=ProviderStatus.TRANSPORT_ERROR,
                message=f"Failed to connect to 9Router endpoint at {models_url}: {e}",
            )

        available_models = [m.get("id") for m in body.get("data", [])]
        if self.model_name not in available_models:
            raise NineRouterPreflightError(
                status=ProviderStatus.MODEL_UNAVAILABLE,
                message=f"Requested model '{self.model_name}' not listed in 9Router available models: {available_models}",
            )

        return {
            "status": "HEALTHY",
            "endpoint": self.base_url,
            "model": self.model_name,
            "inference_authorized": True,
            "available_model_count": len(available_models),
        }

    def _get_llm(self):
        if self._llm is None:
            self.preflight_check()

            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                base_url=f"{self.base_url}/v1",
                api_key=self.api_key,
                model=self.model_name,
                temperature=0.0,
                timeout=self.timeout_seconds,
                max_retries=2,
            )
        return self._llm

    def generate(
        self,
        claim_id: str,
        claim_text: str,
        evidence: List[RetrievalResult],
    ) -> GroundedClaimAnalysis:
        """
        Executes generation with typed provider failure handling.
        Technical failure raises ProviderExecutionError rather than masquerading as scientific abstention.
        """
        start_time = time.time()
        try:
            llm = self._get_llm()
        except ProviderExecutionError:
            raise
        except Exception as e:
            raise ProviderExecutionError(
                status=ProviderStatus.TRANSPORT_ERROR,
                message=f"Failed to initialize model client: {e}",
            )

        prompt_template = create_grounding_prompt()
        evidence_str = format_evidence_context(evidence, provenance=self.provenance)

        messages = prompt_template.format_messages(
            claim_id=claim_id,
            claim_text=claim_text,
            evidence_context=evidence_str,
        )

        output = None
        response = None
        request_count = 1
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        request_events: List[ProviderRequestEvent] = []

        t0 = time.time()
        try:
            response = llm.invoke(messages)
            t_req1 = time.time() - t0
            raw_content = response.content if hasattr(response, "content") else str(response)

            # Robust JSON block extraction (handles ```json ... ``` or raw embedded JSON)
            json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = raw_content

            parsed_dict = json.loads(json_str)
            output = GenerationOutputSchema.model_validate(parsed_dict)

            if hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
                usage = response.response_metadata.get("token_usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")
            elif hasattr(response, "usage_metadata") and isinstance(response.usage_metadata, dict):
                usage = response.usage_metadata
                prompt_tokens = usage.get("input_tokens")
                completion_tokens = usage.get("output_tokens")
                total_tokens = usage.get("total_tokens")

            ev_success = ProviderRequestEvent(
                claim_id=claim_id,
                attempt_index=0,
                request_index=0,
                model_requested=self.model_name,
                model_responded=getattr(response, "response_metadata", {}).get("model_name"),
                request_outcome="SUCCESS",
                latency_seconds=t_req1,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_status=CostStatus.UNKNOWN,
            )
            request_events.append(ev_success)
            if self.request_sink is not None:
                try:
                    self.request_sink(ev_success.model_dump())
                except Exception:
                    pass

        except Exception as primary_exc:
            t_req1 = time.time() - t0
            err_p = str(primary_exc).lower()
            if "timeout" in err_p or "timed out" in err_p:
                ev1 = ProviderRequestEvent(
                    claim_id=claim_id,
                    attempt_index=0,
                    request_index=0,
                    model_requested=self.model_name,
                    request_outcome="TIMEOUT",
                    latency_seconds=t_req1,
                    error_class="TIMEOUT",
                )
                request_events.append(ev1)
                if self.request_sink is not None:
                    try:
                        self.request_sink(ev1.model_dump())
                    except Exception:
                        pass
                raise ProviderExecutionError(status=ProviderStatus.TIMEOUT, message=f"Provider request timed out: {primary_exc}")

            ev1_outcome = "HTTP_ERROR" if ("503" in err_p or "service unavailable" in err_p) else "SCHEMA_VALIDATION_FAILED"
            ev1 = ProviderRequestEvent(
                claim_id=claim_id,
                attempt_index=0,
                request_index=0,
                model_requested=self.model_name,
                request_outcome=ev1_outcome,
                latency_seconds=t_req1,
                error_class=primary_exc.__class__.__name__,
            )
            request_events.append(ev1)
            if self.request_sink is not None:
                try:
                    self.request_sink(ev1.model_dump())
                except Exception:
                    pass

            request_count = 2
            t1 = time.time()
            try:
                structured_llm = llm.with_structured_output(GenerationOutputSchema)
                output = structured_llm.invoke(messages)
                t_req2 = time.time() - t1
                ev2 = ProviderRequestEvent(
                    claim_id=claim_id,
                    attempt_index=0,
                    request_index=1,
                    model_requested=self.model_name,
                    request_outcome="SUCCESS",
                    latency_seconds=t_req2,
                    cost_status=CostStatus.UNKNOWN,
                )
                request_events.append(ev2)
                if self.request_sink is not None:
                    try:
                        self.request_sink(ev2.model_dump())
                    except Exception:
                        pass
            except Exception as e:
                t_req2 = time.time() - t1
                err_msg = str(e).lower()
                outcome = "TIMEOUT" if ("timeout" in err_msg or "timed out" in err_msg) else ("HTTP_ERROR" if "503" in err_msg else "SCHEMA_FAILED")
                ev2 = ProviderRequestEvent(
                    claim_id=claim_id,
                    attempt_index=0,
                    request_index=1,
                    model_requested=self.model_name,
                    request_outcome=outcome,
                    latency_seconds=t_req2,
                    error_class=e.__class__.__name__,
                )
                request_events.append(ev2)
                if self.request_sink is not None:
                    try:
                        self.request_sink(ev2.model_dump())
                    except Exception:
                        pass
                if "timeout" in err_msg or "timed out" in err_msg:
                    raise ProviderExecutionError(status=ProviderStatus.TIMEOUT, message=f"Provider request timed out: {e}")
                raise ProviderExecutionError(
                    status=ProviderStatus.SCHEMA_VALIDATION_FAILED,
                    message=f"Model output failed structured schema validation: {primary_exc} (fallback: {e})",
                )

        latency = time.time() - start_time
        return GroundedClaimAnalysis(
            claim_id=claim_id,
            input_claim=claim_text,
            verdict=output.verdict,
            explanation=output.explanation,
            citations=output.citations,
            model_metadata=ModelMetadata(
                provider="9router",
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                is_zero_cost_verified=False,  # C3: Paid inference; zero-cost is FALSE
                cost_status=CostStatus.UNKNOWN,  # C3: Gateway does not expose price per token
                cost_source="9router_internal_gateway",
                provider_requests_count=request_count,
                request_events=request_events,
            ),
        )
