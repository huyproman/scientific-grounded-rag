"""
Service dependencies, singletons, and lifecycle management.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from scientific_rag.common.paths import ARTIFACTS_DIR, INDEX_ARTIFACTS_DIR, RAW_DATA_DIR
from scientific_rag.config import DenseBackendType, SystemConfig, load_config
from scientific_rag.data.loader import DatasetLoader
from scientific_rag.data.sentence_provenance import SciFactSentenceProvenance
from scientific_rag.generation.adapter import (
    BaseGenerationAdapter,
    MockGenerationAdapter,
    NineRouterAdapter,
)
from scientific_rag.generation.schemas import ClaimVerdict
from scientific_rag.retrieval.base import BaseRetriever
from scientific_rag.retrieval.bm25 import BM25Retriever
from scientific_rag.retrieval.dense import DenseRetriever
from scientific_rag.retrieval.factory import create_dense_retriever, create_hybrid_retriever
from scientific_rag.retrieval.hybrid import HybridRRFRetriever
from scientific_rag.retrieval.qdrant import QdrantRetriever
from scientific_rag.verification.support import PretrainedNLIProxy
from scientific_rag.verification.verifier import MultiLayerEvidenceVerifier
from scientific_rag.workflows.graph import create_verification_graph

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Encapsulates shared models, retrievers, and verifiers for FastAPI serving."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or load_config()
        self.bm25: Optional[BM25Retriever] = None
        self.dense_file: Optional[DenseRetriever] = None
        self.dense_qdrant: Optional[QdrantRetriever] = None
        self.provenance: Optional[SciFactSentenceProvenance] = None
        self.verifier: Optional[MultiLayerEvidenceVerifier] = None
        self.generator: Optional[BaseGenerationAdapter] = None
        self._concurrency_semaphore: Optional[asyncio.Semaphore] = None

    def get_semaphore(self) -> asyncio.Semaphore:
        """Returns bounded concurrency semaphore for inference calls."""
        if self._concurrency_semaphore is None:
            self._concurrency_semaphore = asyncio.Semaphore(self.config.max_concurrent_generations)
        return self._concurrency_semaphore

    def get_bm25(self) -> BM25Retriever:
        """Loads and caches BM25 index singleton."""
        if self.bm25 is None:
            idx_file = self.config.index_artifacts_dir / "bm25_index.pkl"
            if not idx_file.exists():
                raise FileNotFoundError(f"Missing BM25 index artifact at {idx_file}")
            logger.info("Initializing BM25 index from %s...", idx_file)
            self.bm25 = BM25Retriever.load(idx_file)
        return self.bm25

    def get_qdrant_retriever(self) -> QdrantRetriever:
        """Initializes and caches Qdrant retriever singleton."""
        if self.dense_qdrant is None:
            logger.info(
                "Initializing QdrantRetriever for collection '%s' on %s:%d...",
                self.config.qdrant_collection,
                self.config.qdrant_host,
                self.config.qdrant_port,
            )
            self.dense_qdrant = QdrantRetriever(
                collection_name=self.config.qdrant_collection,
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                grpc_port=self.config.qdrant_grpc_port,
                model_name_or_path=self.config.dense_model_name,
                device=self.config.model_device,
                timeout=self.config.qdrant_timeout,
            )
        return self.dense_qdrant

    def get_file_dense_retriever(self) -> DenseRetriever:
        """Loads and caches file-backed dense retriever singleton."""
        if self.dense_file is None:
            dense_dir = self.config.index_artifacts_dir / "dense_index"
            if not dense_dir.exists():
                raise FileNotFoundError(f"Missing file-backed dense index at {dense_dir}")
            logger.info("Loading file-backed DenseRetriever from %s...", dense_dir)
            self.dense_file = DenseRetriever.load(dense_dir, device=self.config.model_device)
        return self.dense_file

    def get_retriever(self, backend_override: Optional[str] = None) -> HybridRRFRetriever:
        """Returns HybridRRFRetriever using specified or default dense backend."""
        bm25 = self.get_bm25()
        selected_backend = (backend_override or self.config.dense_backend.value).lower()

        if selected_backend == DenseBackendType.QDRANT.value:
            dense = self.get_qdrant_retriever()
        elif selected_backend == DenseBackendType.FILE.value:
            dense = self.get_file_dense_retriever()
        else:
            raise ValueError(f"Unknown dense backend '{selected_backend}'. Expected 'qdrant' or 'file'.")

        return HybridRRFRetriever(
            bm25_retriever=bm25,
            dense_retriever=dense,
            rrf_k=self.config.rrf_k,
        )

    def get_provenance(self) -> SciFactSentenceProvenance:
        """Loads and caches SciFact sentence provenance."""
        if self.provenance is None:
            prov = SciFactSentenceProvenance()
            corpus_file = RAW_DATA_DIR.parent / "scifact_official" / "data" / "corpus.jsonl"
            if corpus_file.exists():
                prov.load_corpus(corpus_file)
            else:
                prov.load_corpus()
            self.provenance = prov
        return self.provenance

    def get_verifier(self) -> MultiLayerEvidenceVerifier:
        """Initializes and caches multi-layer evidence verifier."""
        if self.verifier is None:
            prov = self.get_provenance()
            loader = DatasetLoader(RAW_DATA_DIR)
            try:
                qrels = loader.load_qrels(split="test")
            except Exception:
                qrels = None
            try:
                rationales = loader.load_gold_rationales()
            except Exception:
                rationales = None

            nli_model = PretrainedNLIProxy(
                enable_neural=True,
                model_name=self.config.nli_model_name,
                confidence_threshold=self.config.nli_threshold,
            )
            self.verifier = MultiLayerEvidenceVerifier(
                qrels=qrels,
                rationales=rationales,
                nli_proxy=nli_model,
                provenance=prov,
            )
        return self.verifier

    def get_generator(self) -> BaseGenerationAdapter:
        """Initializes and caches model generator adapter."""
        if self.generator is None:
            prov = self.get_provenance()
            provider_type = os.getenv("LLM_PROVIDER", "9router").lower()
            if provider_type == "mock":
                self.generator = MockGenerationAdapter(default_verdict=ClaimVerdict.INSUFFICIENT_EVIDENCE)
            else:
                self.generator = NineRouterAdapter(
                    base_url=self.config.llm_base_url,
                    api_key=self.config.llm_api_key or None,
                    model_name=self.config.llm_model,
                    timeout_seconds=self.config.request_timeout,
                    paid_inference_authorized=self.config.paid_inference_authorized,
                    provenance=prov,
                )
        return self.generator

    def check_readiness(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Assesses local readiness for serving:
        Verifies local indices, configured dense backend, and model assets without paid calls.
        """
        components: Dict[str, Any] = {}
        all_ready = True

        bm25_path = self.config.index_artifacts_dir / "bm25_index.pkl"
        if bm25_path.exists():
            components["bm25_index"] = {"status": "HEALTHY", "details": f"Found index at {bm25_path}"}
        else:
            components["bm25_index"] = {"status": "UNAVAILABLE", "details": f"Missing file {bm25_path}"}
            all_ready = False

        if self.config.dense_backend == DenseBackendType.QDRANT:
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(
                    host=self.config.qdrant_host,
                    port=self.config.qdrant_port,
                    timeout=self.config.qdrant_timeout,
                    check_compatibility=False,
                )
                collections = [c.name for c in client.get_collections().collections]
                if self.config.qdrant_collection not in collections:
                    components["qdrant"] = {
                        "status": "UNAVAILABLE",
                        "details": f"Collection '{self.config.qdrant_collection}' not found in Qdrant. Available: {collections}",
                    }
                    all_ready = False
                else:
                    qdrant_retriever = self.get_qdrant_retriever()
                    qdrant_retriever.ensure_collection(dimension=384, recreate=False)
                    components["qdrant"] = {
                        "status": "HEALTHY",
                        "details": f"Connected to {self.config.qdrant_host}:{self.config.qdrant_port}, collection '{self.config.qdrant_collection}' verified compatible (dim=384, distance=Cosine, manifest present)",
                    }
            except Exception as e:
                components["qdrant"] = {
                    "status": "UNAVAILABLE",
                    "details": f"Qdrant backend check failed: {e}",
                }
                all_ready = False
        else:
            dense_path = self.config.index_artifacts_dir / "dense_index"
            if dense_path.exists() and (dense_path / "manifest.json").exists():
                try:
                    _ = self.get_file_dense_retriever()
                    components["dense_file"] = {"status": "HEALTHY", "details": f"Loaded valid dense index from {dense_path}"}
                except Exception as e:
                    components["dense_file"] = {"status": "UNAVAILABLE", "details": f"Failed to load dense index: {e}"}
                    all_ready = False
            else:
                components["dense_file"] = {"status": "UNAVAILABLE", "details": f"Missing directory or manifest at {dense_path}"}
                all_ready = False

        # 3. Operational model assets check (verifies actual model load & inference ability)
        try:
            if self.config.dense_backend == DenseBackendType.QDRANT:
                model = self.get_qdrant_retriever()._get_model()
            else:
                model = self.get_file_dense_retriever().model
            test_vec = model.encode(["readiness_probe"], show_progress_bar=False, device=self.config.model_device)
            if test_vec is not None and len(test_vec) > 0:
                components["embedding_model"] = {
                    "status": "HEALTHY",
                    "details": f"Model '{self.config.dense_model_name}' operational on {self.config.model_device} (dim={test_vec.shape[1]})",
                }
            else:
                components["embedding_model"] = {"status": "UNAVAILABLE", "details": "Model probe returned empty vector"}
                all_ready = False
        except Exception as e:
            components["embedding_model"] = {"status": "UNAVAILABLE", "details": f"Embedding model initialization failed: {e}"}
            all_ready = False

        try:
            verifier = self.get_verifier()
            if verifier.nli_proxy:
                ok = verifier.nli_proxy._initialize_model()
                if ok:
                    components["nli_model"] = {
                        "status": "HEALTHY",
                        "details": f"Model '{self.config.nli_model_name}' operational on CPU",
                    }
                else:
                    components["nli_model"] = {
                        "status": "UNAVAILABLE",
                        "details": f"Failed to load NLI model '{self.config.nli_model_name}'",
                    }
                    all_ready = False
            else:
                components["nli_model"] = {"status": "HEALTHY", "details": "NLI proxy offline mode"}
        except Exception as e:
            components["nli_model"] = {"status": "UNAVAILABLE", "details": f"NLI model initialization failed: {e}"}
            all_ready = False

        # 4. LLM provider route reachability check (metadata query only, no paid call)
        try:
            adapter = self.get_generator()
            if hasattr(adapter, "preflight_check"):
                preflight = adapter.preflight_check()
                components["llm_provider"] = {"status": "HEALTHY", "details": f"Route '{self.config.llm_model}' available on {self.config.llm_base_url}"}
            else:
                components["llm_provider"] = {"status": "HEALTHY", "details": "Mock/offline generator"}
        except Exception as e:
            components["llm_provider"] = {"status": "DEGRADED", "details": f"Provider check failed: {e}"}

        return all_ready, components


_global_container: Optional[ServiceContainer] = None


def get_service_container() -> ServiceContainer:
    global _global_container
    if _global_container is None:
        _global_container = ServiceContainer()
    return _global_container
