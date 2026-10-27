"""
Typed configuration loader for Scientific Grounded RAG.
Loads configuration from environment variables or .env file with validated defaults.
"""

from enum import Enum
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from scientific_rag.common.paths import (
    ARTIFACTS_DIR,
    INDEX_ARTIFACTS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)


class DenseBackendType(str, Enum):
    FILE = "file"
    QDRANT = "qdrant"


class SystemConfig(BaseModel):
    """Authoritative typed system configuration."""

    qdrant_host: str = Field(default="127.0.0.1")
    qdrant_port: int = Field(default=6335)
    qdrant_grpc_port: int = Field(default=6336)
    qdrant_collection: str = Field(default="scifact_chunks")
    qdrant_timeout: float = Field(default=10.0)

    dense_backend: DenseBackendType = Field(default=DenseBackendType.QDRANT)
    dense_model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    model_device: str = Field(default="cpu")
    rrf_k: int = Field(default=60)
    initial_top_k: int = Field(default=5)
    expanded_top_k: int = Field(default=15)

    llm_base_url: str = Field(default="http://127.0.0.1:8000")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="ag/gemini-3.8-flash-high")
    request_timeout: int = Field(default=60)
    max_concurrent_generations: int = Field(default=3)
    paid_inference_authorized: bool = Field(default=False)

    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8008)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    nli_model_name: str = Field(default="cross-encoder/nli-deberta-v3-xsmall")
    nli_threshold: float = Field(default=0.60)
    canonical_sentence_provenance: bool = Field(default=True)

    project_root: Path = Field(default=PROJECT_ROOT)
    raw_data_dir: Path = Field(default=RAW_DATA_DIR)
    index_artifacts_dir: Path = Field(default=INDEX_ARTIFACTS_DIR)
    artifacts_dir: Path = Field(default=ARTIFACTS_DIR)


def load_config(env_file: Optional[Path] = None) -> SystemConfig:
    """
    Loads configuration by reading environment variables with optional .env loading.
    Does not crash if .env does not exist.
    """
    # Attempt to load python-dotenv if installed
    try:
        from dotenv import load_dotenv
        target_env = env_file or (PROJECT_ROOT / ".env")
        if target_env.exists():
            load_dotenv(target_env, override=False)
    except ImportError:
        pass

    backend_str = os.getenv("DENSE_BACKEND", "qdrant").lower()
    backend = DenseBackendType.QDRANT if backend_str == "qdrant" else DenseBackendType.FILE

    return SystemConfig(
        qdrant_host=os.getenv("QDRANT_HOST", "127.0.0.1"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6335")),
        qdrant_grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6336")),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "scifact_chunks"),
        qdrant_timeout=float(os.getenv("QDRANT_TIMEOUT", "10.0")),
        dense_backend=backend,
        dense_model_name=os.getenv("DENSE_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
        model_device=os.getenv("MODEL_DEVICE", "cpu"),
        rrf_k=int(os.getenv("RRF_K", "60")),
        initial_top_k=int(os.getenv("INITIAL_TOP_K", "5")),
        expanded_top_k=int(os.getenv("EXPANDED_TOP_K", "15")),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "ag/gemini-3.8-flash-high"),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
        max_concurrent_generations=int(os.getenv("MAX_CONCURRENT_GENERATIONS", "3")),
        paid_inference_authorized=os.getenv("PAID_INFERENCE_AUTHORIZED", "false").lower() in ("1", "true", "yes"),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8008")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv("LOG_FORMAT", "json"),
        nli_model_name=os.getenv("NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-xsmall"),
        nli_threshold=float(os.getenv("NLI_THRESHOLD", "0.60")),
        canonical_sentence_provenance=os.getenv("CANONICAL_SENTENCE_PROVENANCE", "true").lower() in ("1", "true", "yes"),
    )
