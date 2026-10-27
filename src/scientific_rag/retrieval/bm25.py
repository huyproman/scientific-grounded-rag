"""
Standalone Lexical BM25 retriever using rank-bm25 (BM25Okapi).
"""

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from scientific_rag.common.compatibility import (
    BM25IndexConfig,
    IncompatibleIndexError,
    IndexManifest,
    PreprocessingConfig,
)
from scientific_rag.data.models import Chunk, RetrievalResult
from scientific_rag.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """Lexical baseline retriever implementing BM25Okapi with deterministic tie-breaking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__(name="BM25Okapi")
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []
        self._tokenizer_regex = re.compile(r"\b\w+\b")

    def tokenize(self, text: str) -> List[str]:
        """Simple, deterministic lowercase word tokenization."""
        return self._tokenizer_regex.findall(text.lower())

    def index(self, chunks: List[Chunk]) -> None:
        """Tokenizes chunks and builds inverted BM25 index."""
        if not chunks:
            raise ValueError("Cannot index empty chunk list.")
        self.chunks = list(chunks)
        tokenized_corpus = [self.tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieves and ranks chunks using BM25 scoring."""
        if self.bm25 is None or not self.chunks:
            raise RuntimeError("BM25 index not built. Call index() or load() first.")

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        indexed_scores = [(scores[i], self.chunks[i]) for i in range(len(self.chunks))]
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
        path: Path,
        dataset_name: str = "BEIR-SciFact",
        dataset_hash: str = "default_hash",
        preprocessing_config: Optional[PreprocessingConfig] = None,
    ) -> None:
        """Persists BM25 model, chunk metadata, and index manifest to disk."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        prep_conf = preprocessing_config or PreprocessingConfig()
        bm25_conf = BM25IndexConfig(k1=self.k1, b=self.b)

        manifest = IndexManifest(
            retriever_type="bm25",
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            num_chunks=len(self.chunks),
            preprocessing_fingerprint=prep_conf.fingerprint(),
            retriever_fingerprint=bm25_conf.fingerprint(),
            preprocessing_config=prep_conf.__dict__,
            retriever_config=bm25_conf.__dict__,
        )

        data = {
            "name": self.name,
            "k1": self.k1,
            "b": self.b,
            "bm25": self.bm25,
            "chunks": self.chunks,
            "manifest": manifest.to_dict(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(
        cls,
        path: Path,
        expected_dataset_name: Optional[str] = None,
        expected_dataset_hash: Optional[str] = None,
        expected_preprocessing_config: Optional[PreprocessingConfig] = None,
        expected_bm25_config: Optional[BM25IndexConfig] = None,
    ) -> "BM25Retriever":
        """Loads serialized BM25 retriever from disk and validates compatibility."""
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"BM25 index artifact not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        manifest_data = data.get("manifest")
        if not manifest_data:
            raise IncompatibleIndexError(
                f"Index artifact at {path} is missing compatibility manifest. "
                "It may be corrupted or from a legacy version. Rebuild with 'python cli.py index --retriever bm25'."
            )

        manifest = IndexManifest.from_dict(manifest_data)

        exp_prep_fp = expected_preprocessing_config.fingerprint() if expected_preprocessing_config else None
        exp_bm25_fp = expected_bm25_config.fingerprint() if expected_bm25_config else None

        manifest.validate_compatibility(
            expected_retriever_type="bm25",
            expected_dataset_name=expected_dataset_name,
            expected_dataset_hash=expected_dataset_hash,
            expected_preprocessing_fingerprint=exp_prep_fp,
            expected_retriever_fingerprint=exp_bm25_fp,
        )

        retriever = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        retriever.bm25 = data["bm25"]
        retriever.chunks = data["chunks"]
        return retriever
