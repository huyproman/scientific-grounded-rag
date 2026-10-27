"""
Retriever factory module supporting dynamic backend switching between Qdrant and file-backed indices..
"""

import logging
from pathlib import Path
from typing import Optional

from scientific_rag.common.paths import INDEX_ARTIFACTS_DIR
from scientific_rag.config import DenseBackendType, SystemConfig, load_config
from scientific_rag.retrieval.base import BaseRetriever
from scientific_rag.retrieval.bm25 import BM25Retriever
from scientific_rag.retrieval.dense import DenseRetriever
from scientific_rag.retrieval.hybrid import HybridRRFRetriever
from scientific_rag.retrieval.qdrant import QdrantRetriever

logger = logging.getLogger(__name__)


def create_dense_retriever(
    backend: Optional[str] = None,
    config: Optional[SystemConfig] = None,
) -> BaseRetriever:
    """
    Instantiates dense retriever according to configured backend ('qdrant' or 'file').
    """
    cfg = config or load_config()
    selected_backend = (backend or cfg.dense_backend.value).lower()

    if selected_backend == DenseBackendType.QDRANT.value:
        logger.info(
            "Creating QdrantRetriever for collection '%s' on %s:%d...",
            cfg.qdrant_collection,
            cfg.qdrant_host,
            cfg.qdrant_port,
        )
        return QdrantRetriever(
            collection_name=cfg.qdrant_collection,
            host=cfg.qdrant_host,
            port=cfg.qdrant_port,
            grpc_port=cfg.qdrant_grpc_port,
            model_name_or_path=cfg.dense_model_name,
            device=cfg.model_device,
            timeout=cfg.qdrant_timeout,
        )
    elif selected_backend == DenseBackendType.FILE.value:
        dense_dir = cfg.index_artifacts_dir / "dense_index"
        logger.info("Loading file-backed DenseRetriever from %s...", dense_dir)
        return DenseRetriever.load(dense_dir, device=cfg.model_device)
    else:
        raise ValueError(f"Unknown dense backend: '{selected_backend}'. Valid options: 'qdrant', 'file'.")


def create_hybrid_retriever(
    dense_backend: Optional[str] = None,
    config: Optional[SystemConfig] = None,
    bm25_path: Optional[Path] = None,
) -> HybridRRFRetriever:
    """
    Creates full HybridRRFRetriever fusing BM25 with configured dense backend.
    """
    cfg = config or load_config()
    bm25_file = bm25_path or (cfg.index_artifacts_dir / "bm25_index.pkl")
    logger.info("Loading BM25 index from %s...", bm25_file)
    bm25 = BM25Retriever.load(bm25_file)

    dense = create_dense_retriever(backend=dense_backend, config=cfg)
    return HybridRRFRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        rrf_k=cfg.rrf_k,
    )
