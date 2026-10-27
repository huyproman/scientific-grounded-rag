"""
FastAPI application for Scientific Grounded RAG serving.
Exposes /health/live, /health/ready, /v1/retrieve, and /v1/analyze.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from scientific_rag.api.dependencies import ServiceContainer, get_service_container
from scientific_rag.api.logging import current_request_id, setup_structured_logging
from scientific_rag.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CitationItem,
    CitationOutcomeItem,
    ComponentStatus,
    ErrorResponse,
    LiveHealthResponse,
    ModelUsageMetadata,
    ReadyHealthResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunkItem,
    VerificationSummary,
)
from scientific_rag.generation.adapter import ProviderExecutionError, ProviderStatus
from scientific_rag.workflows.graph import WorkflowState, create_verification_graph

logger = logging.getLogger("scientific_rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and graceful shutdown."""
    setup_structured_logging(log_level="INFO", log_format="json")
    logger.info("Starting Scientific Grounded RAG serving layer...")
    container = get_service_container()
    ready, components = container.check_readiness()
    logger.info("Startup readiness check: ready=%s, components=%s", ready, components)
    yield
    logger.info("Shutting down Scientific Grounded RAG serving layer...")


app = FastAPI(
    title="Scientific Retrieval & Evidence-Grounded RAG API",
    description="Local reference serving layer for scientific claim retrieval and evidence verification.",
    version="0.1.0",
    lifespan=lifespan,
)



@app.middleware("http")
async def correlation_and_logging_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = current_request_id.set(req_id)
    t0 = time.time()

    logger.info(
        "Request started: %s %s",
        request.method,
        request.url.path,
        extra={
            "request_id": req_id,
            "endpoint": request.url.path,
            "method": request.method,
        },
    )

    try:
        response: Response = await call_next(request)
        elapsed_ms = (time.time() - t0) * 1000.0

        response.headers["X-Request-ID"] = req_id

        logger.info(
            "Request finished: %s %s -> HTTP %d in %.2fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            extra={
                "request_id": req_id,
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "latency_ms": elapsed_ms,
            },
        )
        return response
    except Exception as exc:
        elapsed_ms = (time.time() - t0) * 1000.0
        logger.error(
            "Unhandled request error in %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
            extra={
                "request_id": req_id,
                "endpoint": request.url.path,
                "method": request.method,
                "latency_ms": elapsed_ms,
                "error_category": exc.__class__.__name__,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                request_id=req_id,
                error_code="INTERNAL_SERVER_ERROR",
                message="An unexpected internal error occurred.",
                details={"error": str(exc)},
            ).model_dump(),
            headers={"X-Request-ID": req_id},
        )
    finally:
        current_request_id.reset(token)



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = current_request_id.get() or "unknown"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            request_id=req_id,
            error_code="VALIDATION_ERROR",
            message="Request body failed validation schema.",
            details={"errors": exc.errors()},
        ).model_dump(),
        headers={"X-Request-ID": req_id},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    req_id = current_request_id.get() or "unknown"
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            request_id=req_id,
            error_code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
        ).model_dump(),
        headers={"X-Request-ID": req_id},
    )



@app.get("/health/live", response_model=LiveHealthResponse, tags=["Health"])
async def health_live():
    """Liveness probe: returns ok if process is active and responsive."""
    return LiveHealthResponse(status="ok", service="scientific-grounded-rag")


@app.get("/health/ready", response_model=ReadyHealthResponse, tags=["Health"])
async def health_ready(container: ServiceContainer = Depends(get_service_container)):
    """Readiness probe: validates local indices and configured retrieval backend."""
    ready, components = container.check_readiness()

    status_str = "READY" if ready else "NOT_READY"
    http_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE

    component_models = {
        name: ComponentStatus(status=info["status"], details=info.get("details"))
        for name, info in components.items()
    }

    resp = ReadyHealthResponse(
        status=status_str,
        retrieval_backend=container.config.dense_backend.value,
        components=component_models,
    )

    if not ready:
        return JSONResponse(status_code=http_code, content=resp.model_dump())
    return resp



@app.post("/v1/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
async def retrieve_endpoint(
    req: RetrieveRequest,
    container: ServiceContainer = Depends(get_service_container),
):
    """
    Retrieves ranked scientific passage chunks using Hybrid RRF over BM25 and configured dense backend.
    """
    req_id = current_request_id.get() or str(uuid.uuid4())
    backend = req.dense_backend or container.config.dense_backend.value

    try:
        retriever = container.get_retriever(backend_override=backend)
        raw_results = retriever.retrieve(query=req.query, top_k=req.top_k)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Required index artifact missing: {e}",
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {e}",
        )

    items = [
        RetrievedChunkItem(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            title=r.title,
            text=r.text,
            score=r.score,
            rank=r.rank,
            metadata=r.metadata,
        )
        for r in raw_results
    ]

    return RetrieveResponse(
        request_id=req_id,
        query=req.query,
        retrieval_backend=backend,
        total_results=len(items),
        results=items,
    )


@app.post("/v1/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_endpoint(
    req: AnalyzeRequest,
    container: ServiceContainer = Depends(get_service_container),
):
    """
    Executes compiled LangGraph workflow:
      hybrid retrieve -> grounded generate -> multi-layer verify -> conditional retry -> safe finalization.
    """
    req_id = current_request_id.get() or str(uuid.uuid4())
    claim_id = req.claim_id or f"claim_{uuid.uuid4().hex[:8]}"
    backend = req.dense_backend or container.config.dense_backend.value

    # Build LangGraph workflow using configured dependencies
    try:
        retriever = container.get_retriever(backend_override=backend)
        generator = container.get_generator()
        verifier = container.get_verifier()
        graph = create_verification_graph(
            retriever=retriever,
            generator=generator,
            verifier_fn=verifier.verify,
            max_retries=1,
        )
    except Exception as e:
        logger.error("Failed to assemble verification graph: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize pipeline components: {e}",
        )

    initial_state: WorkflowState = {
        "claim_id": claim_id,
        "claim_text": req.claim_text,
        "evidence_items": [],
        "analysis": None,
        "verification": None,
        "retry_count": 0,
        "max_retries": 1,
        "final_verdict": None,
        "is_verified_conclusive": False,
        "workflow_outcome": "",
        "generation_attempts": [],
        "provider_requests": [],
    }

    t0 = time.time()
    semaphore = container.get_semaphore()

    try:
        async with semaphore:
            # Run LangGraph synchronously in thread pool
            loop = asyncio.get_running_loop()
            final_state = await loop.run_in_executor(None, graph.invoke, initial_state)
    except ProviderExecutionError as pe:
        # Typed technical error handling (Section 6.5)
        logger.error("Provider execution error: status=%s, msg=%s", pe.status, pe.message)
        if pe.status == ProviderStatus.TIMEOUT:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Provider request timed out: {pe.message}",
            )
        elif pe.status in (ProviderStatus.TRANSPORT_ERROR, ProviderStatus.MODEL_UNAVAILABLE):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Provider service unavailable: {pe.message}",
            )
        elif pe.status == ProviderStatus.AUTH_FAILED:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Provider authentication failed: {pe.message}",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Provider failed with status {pe.status}: {pe.message}",
            )
    except Exception as e:
        err_msg = str(e).lower()
        if "timeout" in err_msg or "timed out" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Workflow execution timed out: {e}",
            )
        elif "503" in err_msg or "unavailable" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Upstream service unavailable: {e}",
            )
        logger.error("LangGraph execution failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failure: {e}",
        )

    latency = time.time() - t0
    analysis = final_state.get("analysis")

    # Format response citations
    citations_list: List[CitationItem] = []
    if analysis and analysis.citations:
        for c in analysis.citations:
            citations_list.append(
                CitationItem(
                    doc_id=getattr(c, "parent_doc_id", getattr(c, "doc_id", "")),
                    chunk_id=c.chunk_id,
                    sentence_idx=getattr(c, "sentence_idx", None),
                    quote=getattr(c, "evidence_text", getattr(c, "quote", "")),
                )
            )

    # Format verification summary
    verif_summary = None
    verification = final_state.get("verification")
    if verification:
        details = []
        for idx, d in enumerate(verification.citation_details, start=1):
            details.append(
                CitationOutcomeItem(
                    citation_index=getattr(d, "citation_index", idx),
                    is_chunk_valid=d.is_chunk_valid,
                    is_sentence_valid=d.is_sentence_valid,
                    is_ambiguous_sentence=d.is_ambiguous_sentence,
                    authoritative_sentence_text=d.authoritative_sentence_text,
                    gold_doc_aligned=d.gold_doc_aligned,
                    gold_sentence_aligned=d.gold_sentence_aligned,
                    nli_verdict=d.nli_verdict,
                    nli_confidence=d.nli_confidence,
                    notes=d.notes,
                )
            )
        verif_summary = VerificationSummary(
            total_citations=verification.total_citations,
            valid_citations=verification.valid_citations,
            valid_sentence_citations=verification.valid_sentence_citations,
            all_citations_valid=verification.all_citations_valid,
            verdict_supported_by_evidence=verification.verdict_supported_by_evidence,
            citation_details=details,
        )

    # Format usage metadata
    usage_meta = None
    if analysis and analysis.model_metadata:
        mm = analysis.model_metadata
        usage_meta = ModelUsageMetadata(
            provider=mm.provider,
            model_name=mm.model_name,
            prompt_tokens=mm.prompt_tokens,
            completion_tokens=mm.completion_tokens,
            total_tokens=mm.total_tokens,
            cost_status=mm.cost_status.value if hasattr(mm.cost_status, "value") else str(mm.cost_status),
            provider_requests_count=mm.provider_requests_count,
        )

    return AnalyzeResponse(
        request_id=req_id,
        claim_id=claim_id,
        claim_text=req.claim_text,
        final_verdict=str(final_state.get("final_verdict")),
        is_verified_conclusive=bool(final_state.get("is_verified_conclusive")),
        workflow_outcome=str(final_state.get("workflow_outcome")),
        explanation=analysis.explanation if analysis else "",
        citations=citations_list,
        verification=verif_summary,
        retry_count=int(final_state.get("retry_count", 0)),
        latency_seconds=latency,
        model_metadata=usage_meta,
    )
