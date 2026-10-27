"""Qdrant retriever and index compatibility manifests, hybrid RRF integration, FastAPI serving contracts, readiness probes, error handling and logging."""

from unittest.mock import MagicMock, patch
import json
import pytest
from fastapi.testclient import TestClient

from scientific_rag.api.app import app
from scientific_rag.api.dependencies import ServiceContainer, get_service_container
from scientific_rag.common.paths import INDEX_ARTIFACTS_DIR
from scientific_rag.common.compatibility import IncompatibleIndexError
from scientific_rag.config import DenseBackendType, SystemConfig, load_config
from scientific_rag.data.models import Chunk, RetrievalResult
from scientific_rag.generation.adapter import (
    MockGenerationAdapter,
    ProviderExecutionError,
    ProviderStatus,
)
from scientific_rag.generation.schemas import (
    ClaimVerdict,
    CostStatus,
    GroundedClaimAnalysis,
    ModelMetadata,
    VerificationReport,
)
from scientific_rag.retrieval.bm25 import BM25Retriever
from scientific_rag.retrieval.hybrid import HybridRRFRetriever
from scientific_rag.retrieval.qdrant import (
    MANIFEST_POINT_ID,
    QdrantRetriever,
    chunk_id_to_uuid,
)


# ==============================================================================
# Qdrant retriever and compatibility
# ==============================================================================

def test_qdrant_chunk_id_to_uuid_deterministic():
    """Deterministic mapping from chunk_id to valid Qdrant UUID."""
    uuid1 = chunk_id_to_uuid("12345#c0")
    uuid2 = chunk_id_to_uuid("12345#c0")
    uuid3 = chunk_id_to_uuid("12345#c1")

    assert uuid1 == uuid2, "Deterministic mapping must produce identical UUIDs"
    assert uuid1 != uuid3, "Distinct chunks must produce distinct UUIDs"
    assert len(uuid1) == 36, "Must be standard 36-char string UUID"


def test_qdrant_retriever_empty_chunk_rejection():
    """Empty chunk list is rejected fail-closed."""
    retriever = QdrantRetriever(collection_name="test_col")
    with pytest.raises(ValueError, match="Cannot index an empty chunk list"):
        retriever.index([])


def test_qdrant_retriever_empty_query_rejection():
    """Empty search query is rejected."""
    retriever = QdrantRetriever(collection_name="test_col")
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.retrieve("")
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.retrieve("   ")


def test_qdrant_retriever_incompatible_dimension_rejection():
    """Collection with conflicting vector dimension raises IncompatibleIndexError."""
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "dim_mismatch_col"
    mock_client.get_collections.return_value.collections = [mock_col]

    mock_info = MagicMock()
    mock_info.config.params.vectors.size = 768
    mock_client.get_collection.return_value = mock_info

    retriever = QdrantRetriever(
        collection_name="dim_mismatch_col",
        client=mock_client,
    )

    with pytest.raises(IncompatibleIndexError, match="vector dimension"):
        retriever.ensure_collection(dimension=384, recreate=False)


def test_qdrant_retriever_incompatible_model_rejection():
    """Collection indexed with conflicting model name raises IncompatibleIndexError."""
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "model_mismatch_col"
    mock_client.get_collections.return_value.collections = [mock_col]

    mock_info = MagicMock()
    mock_info.config.params.vectors.size = 384
    mock_client.get_collection.return_value = mock_info

    mock_manifest_point = MagicMock()
    mock_manifest_point.payload = {
        "is_collection_manifest": True,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "distance": "Cosine",
    }
    mock_client.retrieve.return_value = [mock_manifest_point]

    retriever = QdrantRetriever(
        collection_name="model_mismatch_col",
        model_name_or_path="BAAI/bge-small-en-v1.5",
        client=mock_client,
    )

    with pytest.raises(IncompatibleIndexError, match="incompatible with current model"):
        retriever.ensure_collection(dimension=384, recreate=False)


def test_qdrant_retriever_compatible_collection_accepted():
    """Fully compatible collection and manifest are accepted without error."""
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "compatible_col"
    mock_client.get_collections.return_value.collections = [mock_col]

    mock_info = MagicMock()
    mock_info.config.params.vectors.size = 384
    mock_client.get_collection.return_value = mock_info

    mock_manifest_point = MagicMock()
    mock_manifest_point.payload = {
        "is_collection_manifest": True,
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "distance": "Cosine",
        "dataset_name": "BEIR-SciFact",
        "corpus_fingerprint": "expected_corpus_fp_123",
        "preprocessing_fingerprint": "prep_fp_456",
        "preprocessing_config": {"sentences_per_chunk": 3, "sentence_overlap": 1},
    }
    mock_client.retrieve.return_value = [mock_manifest_point]

    retriever = QdrantRetriever(
        collection_name="compatible_col",
        model_name_or_path="BAAI/bge-small-en-v1.5",
        dataset_name="BEIR-SciFact",
        expected_corpus_fingerprint="expected_corpus_fp_123",
        expected_preprocessing_fingerprint="prep_fp_456",
        expected_preprocessing_config={"sentences_per_chunk": 3, "sentence_overlap": 1},
        client=mock_client,
    )

    # Should succeed cleanly
    retriever.ensure_collection(dimension=384, recreate=False)
    assert retriever._collection_validated is True


def test_qdrant_retriever_different_corpus_rejection():
    """Collection indexed for different corpus raises IncompatibleIndexError."""
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "corpus_mismatch_col"
    mock_client.get_collections.return_value.collections = [mock_col]

    mock_info = MagicMock()
    mock_info.config.params.vectors.size = 384
    mock_client.get_collection.return_value = mock_info

    mock_manifest_point = MagicMock()
    mock_manifest_point.payload = {
        "is_collection_manifest": True,
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "distance": "Cosine",
        "dataset_name": "BEIR-SciFact",
        "corpus_fingerprint": "corpus_hash_A",
    }
    mock_client.retrieve.return_value = [mock_manifest_point]

    retriever = QdrantRetriever(
        collection_name="corpus_mismatch_col",
        model_name_or_path="BAAI/bge-small-en-v1.5",
        dataset_name="BEIR-SciFact",
        expected_corpus_fingerprint="corpus_hash_B",
        client=mock_client,
    )

    with pytest.raises(IncompatibleIndexError, match="corpus fingerprint"):
        retriever.ensure_collection(dimension=384, recreate=False)


def test_qdrant_retriever_different_preprocessing_fingerprint_rejection():
    """Collection indexed with conflicting preprocessing fingerprint raises IncompatibleIndexError."""
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "prep_mismatch_col"
    mock_client.get_collections.return_value.collections = [mock_col]

    mock_info = MagicMock()
    mock_info.config.params.vectors.size = 384
    mock_client.get_collection.return_value = mock_info

    mock_manifest_point = MagicMock()
    mock_manifest_point.payload = {
        "is_collection_manifest": True,
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "distance": "Cosine",
        "dataset_name": "BEIR-SciFact",
        "preprocessing_fingerprint": "prep_hash_old",
    }
    mock_client.retrieve.return_value = [mock_manifest_point]

    retriever = QdrantRetriever(
        collection_name="prep_mismatch_col",
        model_name_or_path="BAAI/bge-small-en-v1.5",
        expected_preprocessing_fingerprint="prep_hash_new",
        client=mock_client,
    )

    with pytest.raises(IncompatibleIndexError, match="preprocessing fingerprint"):
        retriever.ensure_collection(dimension=384, recreate=False)


def test_qdrant_retriever_different_preprocessing_config_param_rejection():
    """Collection indexed with conflicting preprocessing params (e.g. sentences_per_chunk) raises IncompatibleIndexError."""
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "prep_cfg_mismatch_col"
    mock_client.get_collections.return_value.collections = [mock_col]

    mock_info = MagicMock()
    mock_info.config.params.vectors.size = 384
    mock_client.get_collection.return_value = mock_info

    mock_manifest_point = MagicMock()
    mock_manifest_point.payload = {
        "is_collection_manifest": True,
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "distance": "Cosine",
        "dataset_name": "BEIR-SciFact",
        "preprocessing_config": {"sentences_per_chunk": 3, "sentence_overlap": 1},
    }
    mock_client.retrieve.return_value = [mock_manifest_point]

    retriever = QdrantRetriever(
        collection_name="prep_cfg_mismatch_col",
        model_name_or_path="BAAI/bge-small-en-v1.5",
        expected_preprocessing_config={"sentences_per_chunk": 5, "sentence_overlap": 1},
        client=mock_client,
    )

    with pytest.raises(IncompatibleIndexError, match="preprocessing config 'sentences_per_chunk=3'"):
        retriever.ensure_collection(dimension=384, recreate=False)


def test_qdrant_retriever_idempotent_reindex():
    """Re-indexing identical chunks updates payloads idempotently without error."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    retriever = QdrantRetriever(collection_name="idempotent_col", client=mock_client)
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    import numpy as np
    mock_model.encode.return_value = np.zeros((2, 384))
    retriever._model = mock_model

    chunks = [
        Chunk(chunk_id="doc1#c0", doc_id="doc1", title="Title 1", text="Sentence 1.", chunk_index=0),
        Chunk(chunk_id="doc1#c1", doc_id="doc1", title="Title 1", text="Sentence 2.", chunk_index=1),
    ]

    retriever.index(chunks, recreate_collection=True)
    # Upsert should have been called for manifest + chunks + manifest update
    assert mock_client.upsert.call_count >= 2

    # Second index call (idempotent re-index with same collection mock)
    mock_col = MagicMock()
    mock_col.name = "idempotent_col"
    mock_client.get_collections.return_value.collections = [mock_col]
    mock_info = MagicMock()
    mock_info.config.params.vectors.size = 384
    mock_client.get_collection.return_value = mock_info
    from scientific_rag.retrieval.qdrant import compute_chunks_corpus_fingerprint
    corpus_fp = compute_chunks_corpus_fingerprint(chunks)
    mock_manifest = MagicMock()
    mock_manifest.payload = {
        "is_collection_manifest": True,
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "distance": "Cosine",
        "dataset_name": "BEIR-SciFact",
        "corpus_fingerprint": corpus_fp,
    }
    mock_client.retrieve.return_value = [mock_manifest]

    retriever.index(chunks, recreate_collection=False)
    assert len(retriever.chunks) == 2


def test_qdrant_retriever_missing_manifest_point_rejection():
    """Collection containing data points but missing manifest point raises IncompatibleIndexError."""
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "legacy_unmanifested_col"
    mock_client.get_collections.return_value.collections = [mock_col]

    mock_info = MagicMock()
    mock_info.config.params.vectors.size = 384
    mock_client.get_collection.return_value = mock_info
    mock_client.retrieve.return_value = []  # No manifest point found
    mock_client.count.return_value.count = 500  # Points exist

    retriever = QdrantRetriever(
        collection_name="legacy_unmanifested_col",
        client=mock_client,
    )

    with pytest.raises(IncompatibleIndexError, match="missing the authoritative collection manifest point"):
        retriever.ensure_collection(dimension=384, recreate=False)


def test_qdrant_retrieve_filters_manifest_point():
    """Retrieval query results strictly filter out the internal manifest point."""
    mock_client = MagicMock()
    mock_client.query_points.return_value.points = [
        MagicMock(id=MANIFEST_POINT_ID, score=0.99, payload={"is_collection_manifest": True}),
        MagicMock(id=chunk_id_to_uuid("c1#c0"), score=0.88, payload={"chunk_id": "c1#c0", "doc_id": "c1", "title": "Paper 1", "text": "Content 1"}),
    ]

    retriever = QdrantRetriever(collection_name="test_col", client=mock_client)
    mock_model = MagicMock()
    import numpy as np
    mock_model.encode.return_value = np.zeros((1, 384))
    retriever._model = mock_model

    results = retriever.retrieve("query", top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "c1#c0"
    assert results[0].doc_id == "c1"


def test_qdrant_save_and_load(tmp_path):
    """Save and load Qdrant connection metadata."""
    save_path = tmp_path / "qdrant_meta"
    retriever = QdrantRetriever(
        collection_name="custom_scifact_col",
        host="10.0.0.99",
        port=6339,
    )
    retriever.save(save_path)

    loaded = QdrantRetriever.load(save_path)
    assert loaded.collection_name == "custom_scifact_col"
    assert loaded.host == "10.0.0.99"
    assert loaded.port == 6339


def test_qdrant_hybrid_rrf_integration_provenance():
    """QdrantRetriever integrates with HybridRRFRetriever and preserves document provenance."""
    chunks = [
        Chunk(chunk_id="doc1#c0", doc_id="doc1", title="Gene Editing", text="CRISPR Cas9 endonuclease.", chunk_index=0),
        Chunk(chunk_id="doc2#c0", doc_id="doc2", title="Neural Nets", text="Deep learning architectures.", chunk_index=0),
    ]

    bm25 = BM25Retriever()
    bm25.index(chunks)

    mock_qdrant = MagicMock(spec=QdrantRetriever)
    mock_qdrant.chunks = chunks
    mock_qdrant.retrieve.return_value = [
        RetrievalResult(chunk_id="doc1#c0", doc_id="doc1", score=0.91, rank=1, title="Gene Editing", text="CRISPR Cas9 endonuclease."),
        RetrievalResult(chunk_id="doc2#c0", doc_id="doc2", score=0.45, rank=2, title="Neural Nets", text="Deep learning architectures."),
    ]

    hybrid = HybridRRFRetriever(bm25_retriever=bm25, dense_retriever=mock_qdrant, rrf_k=60)
    results = hybrid.retrieve("Cas9 endonuclease gene editing", top_k=2)

    assert len(results) == 2
    assert results[0].doc_id == "doc1"
    assert results[0].chunk_id == "doc1#c0"
    assert results[0].score > 0.0


# ==============================================================================
# Configuration
# ==============================================================================

def test_config_loader_defaults_and_overrides(monkeypatch):
    """SystemConfig loads portable defaults and respects environment overrides."""
    monkeypatch.setenv("QDRANT_PORT", "6337")
    monkeypatch.setenv("DENSE_BACKEND", "file")
    monkeypatch.setenv("LLM_MODEL", "custom/test-model")

    cfg = load_config()
    assert cfg.qdrant_port == 6337
    assert cfg.dense_backend == DenseBackendType.FILE
    assert cfg.llm_model == "custom/test-model"
    assert cfg.api_port == 8008  # Default preserved


# ==============================================================================
# FastAPI serving and readiness
# ==============================================================================

requires_indices = pytest.mark.skipif(
    not (INDEX_ARTIFACTS_DIR / "bm25_index.pkl").exists(),
    reason="prebuilt retrieval indices missing (run scripts/download_corpus.py and scripts/build_index.py)",
)


@pytest.fixture
def api_test_client():
    return TestClient(app)


def test_api_health_live(api_test_client):
    """GET /health/live returns HTTP 200 with service status."""
    res = api_test_client.get("/health/live")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "scientific-grounded-rag"
    assert "x-request-id" in res.headers


@requires_indices
def test_api_health_ready_healthy(api_test_client):
    """GET /health/ready returns HTTP 200 when required components exist and models are operational."""
    res = api_test_client.get("/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "READY"
    assert "components" in data
    assert data["components"]["bm25_index"]["status"] == "HEALTHY"
    assert data["components"]["embedding_model"]["status"] == "HEALTHY"
    assert data["components"]["nli_model"]["status"] == "HEALTHY"


def test_api_health_ready_unhealthy_on_missing_collection():
    """GET /health/ready returns HTTP 503 when local collection is missing or incompatible."""
    cfg = SystemConfig(qdrant_collection="nonexistent_mock_collection_xyz")
    container = ServiceContainer(config=cfg)

    app.dependency_overrides[get_service_container] = lambda: container
    client = TestClient(app)
    try:
        res = client.get("/health/ready")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "NOT_READY"
        assert data["components"]["qdrant"]["status"] == "UNAVAILABLE"
    finally:
        app.dependency_overrides.clear()


def test_api_health_ready_unhealthy_on_embedding_failure():
    """GET /health/ready returns HTTP 503 when embedding model fails operational probe."""
    container = ServiceContainer()
    mock_retriever = MagicMock()
    mock_retriever.ensure_collection.return_value = None
    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("GPU/CPU memory allocation failure")
    mock_retriever._get_model.return_value = mock_model
    container.dense_qdrant = mock_retriever

    app.dependency_overrides[get_service_container] = lambda: container
    client = TestClient(app)
    try:
        res = client.get("/health/ready")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "NOT_READY"
        assert data["components"]["embedding_model"]["status"] == "UNAVAILABLE"
    finally:
        app.dependency_overrides.clear()


def test_api_health_ready_unhealthy_on_nli_failure():
    """GET /health/ready returns HTTP 503 when NLI model initialization fails."""
    container = ServiceContainer()
    mock_verifier = MagicMock()
    mock_nli = MagicMock()
    mock_nli._initialize_model.return_value = False  # Load failed
    mock_verifier.nli_proxy = mock_nli
    container.verifier = mock_verifier

    app.dependency_overrides[get_service_container] = lambda: container
    client = TestClient(app)
    try:
        res = client.get("/health/ready")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "NOT_READY"
        assert data["components"]["nli_model"]["status"] == "UNAVAILABLE"
    finally:
        app.dependency_overrides.clear()


@requires_indices
def test_api_retrieve_valid_request(api_test_client):
    """POST /v1/retrieve returns ranked results with exact provenance."""
    payload = {
        "query": "0-dimensional biomaterials show inductive properties",
        "top_k": 3,
    }
    res = api_test_client.post("/v1/retrieve", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_results"] == 3
    assert len(data["results"]) == 3
    for r in data["results"]:
        assert "doc_id" in r
        assert "chunk_id" in r
        assert "score" in r
        assert "rank" in r
        assert r["rank"] >= 1


def test_api_retrieve_validation_errors(api_test_client):
    """POST /v1/retrieve validates empty query and top_k bounds."""
    # Empty query
    res1 = api_test_client.post("/v1/retrieve", json={"query": "", "top_k": 5})
    assert res1.status_code == 422
    assert res1.json()["error_code"] == "VALIDATION_ERROR"

    # top_k = 0
    res2 = api_test_client.post("/v1/retrieve", json={"query": "test", "top_k": 0})
    assert res2.status_code == 422

    # top_k > 100
    res3 = api_test_client.post("/v1/retrieve", json={"query": "test", "top_k": 105})
    assert res3.status_code == 422


@requires_indices
def test_api_analyze_mock_supported(api_test_client):
    """POST /v1/analyze returns valid SUPPORTED verdict contract."""
    mock_gen = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)

    container = get_service_container()
    orig_gen = container.generator
    container.generator = mock_gen

    try:
        payload = {
            "claim_text": "0-dimensional biomaterials show inductive properties.",
            "claim_id": "test_c_supported",
        }
        res = api_test_client.post("/v1/analyze", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["claim_id"] == "test_c_supported"
        assert data["final_verdict"] in ["SUPPORTED", "INSUFFICIENT_EVIDENCE"]
        assert "citations" in data
        assert "verification" in data
        assert res.headers.get("x-request-id") is not None
    finally:
        container.generator = orig_gen


@requires_indices
def test_api_analyze_abstention_distinct_from_error(api_test_client):
    """Scientific abstention returns HTTP 200 with INSUFFICIENT_EVIDENCE, not an error."""
    mock_gen = MockGenerationAdapter(default_verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE)

    container = get_service_container()
    orig_gen = container.generator
    container.generator = mock_gen

    try:
        payload = {
            "claim_text": "An unsubstantiated speculative claim with no scientific evidence.",
            "claim_id": "test_c_insufficient",
        }
        res = api_test_client.post("/v1/analyze", json=payload)
        assert res.status_code == 200, "Scientific abstention must be HTTP 200, not a technical error"
        data = res.json()
        assert data["final_verdict"] == "INSUFFICIENT_EVIDENCE"
        assert data["is_verified_conclusive"] is False
        assert data["workflow_outcome"] == "SCIENTIFIC_ABSTENTION"
    finally:
        container.generator = orig_gen


@requires_indices
def test_api_analyze_provider_timeout_returns_504(api_test_client):
    """Provider timeout raises HTTP 504 Gateway Timeout, never masquerading as abstention."""
    mock_gen = MagicMock(spec=MockGenerationAdapter)
    mock_gen.generate.side_effect = ProviderExecutionError(
        status=ProviderStatus.TIMEOUT,
        message="Request timed out after 60s",
    )

    container = get_service_container()
    orig_gen = container.generator
    container.generator = mock_gen

    try:
        payload = {"claim_text": "Valid scientific claim text."}
        res = api_test_client.post("/v1/analyze", json=payload)
        assert res.status_code == 504, f"Expected HTTP 504, got {res.status_code}"
        data = res.json()
        assert "timed out" in data["message"].lower()
    finally:
        container.generator = orig_gen


@requires_indices
def test_api_analyze_provider_503_returns_503(api_test_client):
    """Provider 503 circuit-breaker raises HTTP 503 Service Unavailable."""
    mock_gen = MagicMock(spec=MockGenerationAdapter)
    mock_gen.generate.side_effect = ProviderExecutionError(
        status=ProviderStatus.TRANSPORT_ERROR,
        message="Upstream 503 Service Unavailable",
    )

    container = get_service_container()
    orig_gen = container.generator
    container.generator = mock_gen

    try:
        payload = {"claim_text": "Valid scientific claim text."}
        res = api_test_client.post("/v1/analyze", json=payload)
        assert res.status_code == 503, f"Expected HTTP 503, got {res.status_code}"
        data = res.json()
        assert "service unavailable" in data["message"].lower()
    finally:
        container.generator = orig_gen


def test_request_correlation_id_passthrough(api_test_client):
    """Custom X-Request-ID header is propagated back in response."""
    custom_id = "test-custom-correlation-id-9988"
    res = api_test_client.get("/health/live", headers={"X-Request-ID": custom_id})
    assert res.status_code == 200
    assert res.headers.get("x-request-id") == custom_id
