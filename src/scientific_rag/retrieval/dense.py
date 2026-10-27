"""
Standalone Dense Bi-Encoder retriever using BAAI/bge-small-en-v1.5 on CPU.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from scientific_rag.common.compatibility import (
    DenseIndexConfig,
    IncompatibleIndexError,
    IndexManifest,
    PreprocessingConfig,
)
from scientific_rag.data.models import Chunk, RetrievalResult
from scientific_rag.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense semantic baseline retriever using BAAI/bge-small-en-v1.5."""

    DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        super().__init__(name="BGE-Small-Dense")
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.model: Optional[SentenceTransformer] = None
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def _load_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name_or_path, device=self.device)
        return self.model

    def index(self, chunks: List[Chunk]) -> None:
        """Encodes all chunks in batches on CPU and normalizes embeddings."""
        if not chunks:
            raise ValueError("Cannot index empty chunk list.")
        self.chunks = list(chunks)
        model = self._load_model()

        texts = [c.text for c in self.chunks]
        # Encode with L2 normalization for cosine similarity via dot product
        raw_embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )
        self.embeddings = np.ascontiguousarray(raw_embeddings, dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Encodes query and computes inner product similarity against normalized chunk embeddings."""
        if self.embeddings is None or not self.chunks:
            raise RuntimeError("Dense index not built. Call index() or load() first.")

        query_text = query.strip()
        if not query_text:
            return []

        model = self._load_model()
        query_embedding = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
        )[0]  # shape: (embedding_dim,)

        # Dot product with normalized vectors = cosine similarity in [-1, 1]
        scores = np.dot(self.embeddings, query_embedding)

        # Deterministic ranking: sort by (-score, chunk_id)
        indexed_scores = [(float(scores[i]), self.chunks[i]) for i in range(len(self.chunks))]
        indexed_scores.sort(key=lambda item: (-item[0], item[1].chunk_id))

        results: List[RetrievalResult] = []
        for rank, (score, chunk) in enumerate(indexed_scores[:top_k], start=1):
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=float(score),
                    rank=rank,
                    title=chunk.title,
                    text=chunk.text,
                    metadata=chunk.metadata.copy(),
                )
            )
        return results

    def save(
        self,
        dir_path: Path,
        dataset_name: str = "BEIR-SciFact",
        dataset_hash: str = "default_hash",
        preprocessing_config: Optional[PreprocessingConfig] = None,
    ) -> None:
        """Saves embedding array, chunk metadata, and index manifest to directory."""
        dir_path = Path(dir_path).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.embeddings is None:
            raise RuntimeError("No embeddings to save.")

        np.save(dir_path / "embeddings.npy", self.embeddings)

        prep_conf = preprocessing_config or PreprocessingConfig()
        dense_conf = DenseIndexConfig(
            model_name_or_path=self.model_name_or_path,
            embedding_dimension=int(self.embeddings.shape[1]),
        )

        manifest = IndexManifest(
            retriever_type="dense",
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            num_chunks=len(self.chunks),
            preprocessing_fingerprint=prep_conf.fingerprint(),
            retriever_fingerprint=dense_conf.fingerprint(),
            preprocessing_config=prep_conf.__dict__,
            retriever_config=dense_conf.__dict__,
        )

        with open(dir_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        meta = {
            "model_name_or_path": self.model_name_or_path,
            "device": self.device,
            "num_chunks": len(self.chunks),
            "embedding_dim": int(self.embeddings.shape[1]),
        }
        with open(dir_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with open(dir_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(
        cls,
        dir_path: Path,
        device: str = "cpu",
        expected_dataset_name: Optional[str] = None,
        expected_dataset_hash: Optional[str] = None,
        expected_preprocessing_config: Optional[PreprocessingConfig] = None,
        expected_dense_config: Optional[DenseIndexConfig] = None,
    ) -> "DenseRetriever":
        """Loads dense retriever from saved directory and validates compatibility."""
        dir_path = Path(dir_path).resolve()
        if not dir_path.exists():
            raise FileNotFoundError(f"Dense index directory not found at {dir_path}")

        manifest_file = dir_path / "manifest.json"
        if not manifest_file.exists():
            raise IncompatibleIndexError(
                f"Index directory at {dir_path} is missing manifest.json. "
                "It may be corrupted or legacy. Rebuild with 'python cli.py index --retriever dense'."
            )

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = IndexManifest.from_dict(json.load(f))

        exp_prep_fp = expected_preprocessing_config.fingerprint() if expected_preprocessing_config else None
        exp_dense_fp = expected_dense_config.fingerprint() if expected_dense_config else None

        manifest.validate_compatibility(
            expected_retriever_type="dense",
            expected_dataset_name=expected_dataset_name,
            expected_dataset_hash=expected_dataset_hash,
            expected_preprocessing_fingerprint=exp_prep_fp,
            expected_retriever_fingerprint=exp_dense_fp,
        )

        with open(dir_path / "config.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        with open(dir_path / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        embeddings = np.load(dir_path / "embeddings.npy")

        retriever = cls(
            model_name_or_path=meta.get("model_name_or_path", cls.DEFAULT_MODEL_NAME),
            device=device,
        )
        retriever.chunks = chunks
        retriever.embeddings = embeddings
        return retriever
