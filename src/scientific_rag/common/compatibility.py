"""
Index compatibility specification, hashing, and stale artifact validation.
Enforces that deserialized retrieval indices match expected corpus, chunking, and model configurations.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class IncompatibleIndexError(ValueError):
    """Raised when an index artifact does not match expected configuration or corpus."""
    pass


@dataclass
class PreprocessingConfig:
    sentences_per_chunk: int = 3
    sentence_overlap: int = 1
    include_title: bool = True
    preprocessing_version: str = "1.0.0"

    def fingerprint(self) -> str:
        data = {
            "sentences_per_chunk": self.sentences_per_chunk,
            "sentence_overlap": self.sentence_overlap,
            "include_title": self.include_title,
            "preprocessing_version": self.preprocessing_version,
        }
        encoded = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class BM25IndexConfig:
    k1: float = 1.5
    b: float = 0.75
    tokenizer: str = "regex_lowercase_words"
    algorithm: str = "BM25Okapi"

    def fingerprint(self) -> str:
        data = {
            "k1": self.k1,
            "b": self.b,
            "tokenizer": self.tokenizer,
            "algorithm": self.algorithm,
        }
        encoded = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class DenseIndexConfig:
    model_name_or_path: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    normalize_embeddings: bool = True
    hub_revision: Optional[str] = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

    def fingerprint(self) -> str:
        data = {
            "model_name_or_path": self.model_name_or_path,
            "embedding_dimension": self.embedding_dimension,
            "normalize_embeddings": self.normalize_embeddings,
            "hub_revision": self.hub_revision,
        }
        encoded = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class IndexManifest:
    retriever_type: str  # "bm25" or "dense"
    dataset_name: str
    dataset_hash: str    # e.g., corpus content or manifest hash
    num_chunks: int
    preprocessing_fingerprint: str
    retriever_fingerprint: str
    preprocessing_config: Dict[str, Any]
    retriever_config: Dict[str, Any]
    format_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexManifest":
        required = [
            "retriever_type", "dataset_name", "dataset_hash", "num_chunks",
            "preprocessing_fingerprint", "retriever_fingerprint",
            "preprocessing_config", "retriever_config"
        ]
        for r in required:
            if r not in data:
                raise IncompatibleIndexError(f"Missing required manifest field: '{r}'")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def validate_compatibility(
        self,
        expected_retriever_type: str,
        expected_dataset_name: Optional[str] = None,
        expected_dataset_hash: Optional[str] = None,
        expected_preprocessing_fingerprint: Optional[str] = None,
        expected_retriever_fingerprint: Optional[str] = None,
    ) -> None:
        """Validates that this manifest matches expected environment specifications."""
        if self.retriever_type != expected_retriever_type:
            raise IncompatibleIndexError(
                f"Retriever type mismatch: index is '{self.retriever_type}', expected '{expected_retriever_type}'"
            )

        if expected_dataset_name and self.dataset_name != expected_dataset_name:
            raise IncompatibleIndexError(
                f"Dataset name mismatch: index was built for '{self.dataset_name}', expected '{expected_dataset_name}'. "
                f"Run 'python cli.py index --retriever {self.retriever_type}' to rebuild."
            )

        if expected_dataset_hash and self.dataset_hash != expected_dataset_hash:
            raise IncompatibleIndexError(
                f"Dataset hash mismatch: index dataset hash '{self.dataset_hash[:8]}...' does not match "
                f"current corpus hash '{expected_dataset_hash[:8]}...'. "
                f"Corpus has changed. Run 'python cli.py index --retriever {self.retriever_type}' to rebuild."
            )

        if expected_preprocessing_fingerprint and self.preprocessing_fingerprint != expected_preprocessing_fingerprint:
            raise IncompatibleIndexError(
                f"Preprocessing configuration mismatch: index chunking fingerprint '{self.preprocessing_fingerprint}' "
                f"does not match expected '{expected_preprocessing_fingerprint}'. "
                f"Run 'python cli.py index --retriever {self.retriever_type}' to rebuild."
            )

        if expected_retriever_fingerprint and self.retriever_fingerprint != expected_retriever_fingerprint:
            raise IncompatibleIndexError(
                f"Retriever configuration mismatch: index retriever fingerprint '{self.retriever_fingerprint}' "
                f"does not match expected '{expected_retriever_fingerprint}'. "
                f"Run 'python cli.py index --retriever {self.retriever_type}' to rebuild."
            )


def resolve_dataset_fingerprint(data_dir: Path) -> Tuple[str, str]:
    """
    Returns (dataset_name, dataset_hash) from local data directory.
    Uses manifest.json if present, or hashes corpus.jsonl.
    """
    data_dir = Path(data_dir).resolve()
    manifest_file = data_dir.parent.parent / "manifest.json"
    if not manifest_file.exists():
        manifest_file = data_dir / "manifest.json"

    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        name = meta.get("dataset_name", "BEIR-SciFact")
        h = meta.get("locally_computed_sha256", meta.get("observed_md5", "unknown"))
        return name, h

    corpus_file = data_dir / "corpus.jsonl"
    if corpus_file.exists():
        hasher = hashlib.sha256()
        with open(corpus_file, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        return "BEIR-SciFact", hasher.hexdigest()

    return "BEIR-SciFact", "default_hash"
