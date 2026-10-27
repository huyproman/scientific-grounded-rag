"""
Conditional stateful verification workflow implemented with LangGraph.
Executes bounded retry: Retrieve -> Generate -> Verify -> (if unverified: Bounded Expand & Retry) -> Return/Abstain.
Invariants:
- Conclusive termination REQUIRES that evidence actually supports the generated verdict.
- Stale verification state is completely recomputed upon candidate expansion retry.
- Bounded retry strictly capped at max_retries (default 1).
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from scientific_rag.data.models import RetrievalResult
from scientific_rag.generation.adapter import BaseGenerationAdapter
from scientific_rag.generation.schemas import (
    ClaimVerdict,
    GroundedClaimAnalysis,
    VerificationReport,
    WorkflowOutcome,
)
from scientific_rag.retrieval.hybrid import HybridRRFRetriever


class WorkflowState(TypedDict):
    """Typed state passing through the LangGraph verification workflow."""
    claim_id: str
    claim_text: str
    evidence_items: List[RetrievalResult]
    analysis: Optional[GroundedClaimAnalysis]
    verification: Optional[VerificationReport]
    retry_count: int
    max_retries: int
    final_verdict: Optional[str]
    is_verified_conclusive: bool
    workflow_outcome: str
    generation_attempts: List[GroundedClaimAnalysis]
    provider_requests: List[Dict[str, Any]]


def create_verification_graph(
    retriever: HybridRRFRetriever,
    generator: BaseGenerationAdapter,
    verifier_fn: Any,
    max_retries: int = 1,
) -> Any:
    """
    Constructs a LangGraph StateGraph implementing conditional stateful retry:
    START -> retrieve -> generate -> verify -> (route: verified conclusive -> END, unverified -> retry_expansion -> generate -> verify -> END)
    """

    def retrieve_node(state: WorkflowState) -> Dict[str, Any]:
        """Initial hybrid retrieval."""
        top_k = 5
        evidence = retriever.retrieve(state["claim_text"], top_k=top_k)
        return {
            "evidence_items": evidence,
            "retry_count": state.get("retry_count", 0),
            "is_verified_conclusive": False,
            "generation_attempts": state.get("generation_attempts") or [],
            "provider_requests": state.get("provider_requests") or [],
        }

    def generate_node(state: WorkflowState) -> Dict[str, Any]:
        """Grounded generation from current evidence context with attempt preservation."""
        analysis = generator.generate(
            claim_id=state["claim_id"],
            claim_text=state["claim_text"],
            evidence=state["evidence_items"],
        )
        attempts = list(state.get("generation_attempts") or [])
        attempts.append(analysis)

        requests = list(state.get("provider_requests") or [])
        if hasattr(analysis, "model_metadata") and analysis.model_metadata.request_events:
            for ev in analysis.model_metadata.request_events:
                requests.append(ev.model_dump())
        elif hasattr(analysis, "model_metadata"):
            requests.append({
                "claim_id": state["claim_id"],
                "attempt_index": len(attempts) - 1,
                "model_requested": analysis.model_metadata.model_name,
                "prompt_tokens": analysis.model_metadata.prompt_tokens,
                "completion_tokens": analysis.model_metadata.completion_tokens,
                "total_tokens": analysis.model_metadata.total_tokens,
                "latency_seconds": analysis.model_metadata.latency_seconds,
                "cost_status": analysis.model_metadata.cost_status.value if hasattr(analysis.model_metadata.cost_status, "value") else str(analysis.model_metadata.cost_status),
                "provider_requests_count": analysis.model_metadata.provider_requests_count,
            })

        return {
            "analysis": analysis,
            "generation_attempts": attempts,
            "provider_requests": requests,
        }

    def verify_node(state: WorkflowState) -> Dict[str, Any]:
        """Runs multi-layer verification on generated analysis and refreshes state."""
        analysis = state["analysis"]
        if analysis is None:
            return {
                "verification": None,
                "final_verdict": "FAILED",
                "is_verified_conclusive": False,
                "workflow_outcome": WorkflowOutcome.TECHNICAL_FAILURE.value,
            }

        report = verifier_fn(analysis, state["evidence_items"])
        analysis.verification = report

        # Conclusive verification condition:
        # Must be SUPPORTED or CONTRADICTED, citations must be valid, and evidence must verify it.
        # INSUFFICIENT_EVIDENCE is NEVER verified conclusive.
        is_conclusive = bool(
            analysis.verdict in [ClaimVerdict.SUPPORTED, ClaimVerdict.CONTRADICTED]
            and report.all_citations_valid
            and report.verdict_supported_by_evidence
        )

        # Enforce Safe Finalization Contract:
        # The user-facing final_verdict can NEVER claim conclusive SUPPORTED or CONTRADICTED
        # when evidence verification rejected that conclusion.
        if is_conclusive:
            final_v = analysis.verdict.value
            outcome = (
                WorkflowOutcome.VERIFIED_SUPPORTED.value
                if analysis.verdict == ClaimVerdict.SUPPORTED
                else WorkflowOutcome.VERIFIED_CONTRADICTED.value
            )
        else:
            final_v = ClaimVerdict.INSUFFICIENT_EVIDENCE.value
            if analysis.verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE:
                outcome = WorkflowOutcome.SCIENTIFIC_ABSTENTION.value
            else:
                outcome = WorkflowOutcome.UNVERIFIED_CONCLUSION.value

        attempts = state.get("generation_attempts") or [analysis]
        total_p_tokens = sum((a.model_metadata.prompt_tokens or 0) for a in attempts)
        total_c_tokens = sum((a.model_metadata.completion_tokens or 0) for a in attempts)
        total_tokens = sum((a.model_metadata.total_tokens or 0) for a in attempts)
        total_reqs = sum((a.model_metadata.provider_requests_count or 1) for a in attempts)

        if analysis.model_metadata:
            if total_p_tokens > 0:
                analysis.model_metadata.prompt_tokens = total_p_tokens
            if total_c_tokens > 0:
                analysis.model_metadata.completion_tokens = total_c_tokens
            if total_tokens > 0:
                analysis.model_metadata.total_tokens = total_tokens
            analysis.model_metadata.provider_requests_count = max(total_reqs, 1)

        return {
            "verification": report,
            "analysis": analysis,
            "final_verdict": final_v,
            "is_verified_conclusive": is_conclusive,
            "workflow_outcome": outcome,
        }

    def expand_retrieve_node(state: WorkflowState) -> Dict[str, Any]:
        """Expanded retrieval for bounded retry upon insufficient or unverified initial evidence."""
        expanded_evidence = retriever.retrieve(state["claim_text"], top_k=15)
        return {
            "evidence_items": expanded_evidence,
            "retry_count": state.get("retry_count", 0) + 1,
            "verification": None,
            "analysis": None,
            "is_verified_conclusive": False,
        }

    def route_after_verification(state: WorkflowState) -> str:
        """
        Conditional routing logic:
        - If verdict is verified and conclusive: terminate (finish -> END).
        - If unverified/insufficient AND retry_count < max_retries: retry via expanded retrieval.
        - Otherwise terminate (abstain/return unverified result).
        """
        is_conclusive = state.get("is_verified_conclusive", False)
        retry_count = state.get("retry_count", 0)
        max_r = state.get("max_retries", 1)

        if is_conclusive:
            return "finish"

        if retry_count < max_r:
            return "retry_expansion"

        return "finish"

    builder = StateGraph(WorkflowState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("verify", verify_node)
    builder.add_node("expand_retrieve", expand_retrieve_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "verify")

    builder.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "finish": END,
            "retry_expansion": "expand_retrieve",
        },
    )

    builder.add_edge("expand_retrieve", "generate")

    return builder.compile()
