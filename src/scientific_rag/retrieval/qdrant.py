"""
Qdrant-backed dense retriever for scientific claim passages.
Implements BaseRetriever interface, supporting batch idempotent indexing,
canonical chunk provenance preservation, and seamless integration with HybridRRFRetriever.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer

from scientific_rag.common.compatibility import IncompatibleIndexError
from scientific_rag.data.models import Chunk, RetrievalResult
from scientific_rag.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


MANIFEST_POINT_ID = "00000000-0000-0000-0000-000000000000"


def chunk_id_to_uuid(chunk_id: str) -> str:
    """
    Deterministically maps an arbitrary string chunk_id (e.g., '123#c0')
    to a valid Qdrant UUID point ID using UUIDv5.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def compute_chunks_corpus_fingerprint(chunks: List[Chunk]) -> str:
    """
    Computes a deterministic SHA256 fingerprint over a collection of Chunks.
    Sorts chunks by (doc_id, chunk_index, chunk_id) and hashes canonical attributes.
    """
    hasher = hashlib.sha256()
    sorted_chunks = sorted(chunks, key=lambda c: (str(c.doc_id), int(c.chunk_index), str(c.chunk_id)))
    for c in sorted_chunks:
        text_hash = hashlib.sha256(c.text.encode("utf-8")).hexdigest()
        entry = f"{c.chunk_id}|{c.doc_id}|{c.chunk_index}|{c.title}|{text_hash}\n"
        hasher.update(entry.encode("utf-8"))
    return hasher.hexdigest()


def compute_preprocessing_fingerprint(config: Dict[str, Any]) -> str:
    """Computes a deterministic SHA256 fingerprint for chunking/preprocessing configuration."""
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class QdrantRetriever(BaseRetriever):
    """
    Dense semantic retriever backed by Qdrant vector database.
    Adheres strictly to the BaseRetriever contract and CPU-first compute constraints.
    """

    def __init__(
        self,
        collection_name: str = "scifact_chunks",
        host: str = "127.0.0.1",
        port: int = 6335,
        grpc_port: int = 6336,
        model_name_or_path: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        timeout: float = 10.0,
        name: str = "QdrantDenseRetriever",
        client: Optional[Any] = None,
        dataset_name: Optional[str] = None,
        expected_corpus_fingerprint: Optional[str] = None,
        expected_preprocessing_fingerprint: Optional[str] = None,
        expected_preprocessing_config: Optional[Dict[str, Any]] = None,
        hub_revision: Optional[str] = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    ):
        super().__init__(name=name)
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.timeout = timeout
        self.dataset_name = dataset_name
        self.expected_corpus_fingerprint = expected_corpus_fingerprint
        self.expected_preprocessing_fingerprint = expected_preprocessing_fingerprint
        self.expected_preprocessing_config = expected_preprocessing_config
        self.hub_revision = hub_revision

        self._client = client
        self._model: Optional[SentenceTransformer] = None
        self.chunks: List[Chunk] = []
        self._collection_validated: bool = False

    def _get_client(self):
        """Initializes or returns active QdrantClient instance."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError(
                    "qdrant-client package is required for QdrantRetriever. "
                    "Install with 'pip install qdrant-client'."
                )
            self._client = QdrantClient(
                host=self.host,
                port=self.port,
                timeout=self.timeout,
                check_compatibility=False,
            )
        return self._client

    def _get_model(self) -> SentenceTransformer:
        """Lazy loads sentence-transformers bi-encoder model."""
        if self._model is None:
            logger.info("Loading dense embedding model '%s' on %s...", self.model_name_or_path, self.device)
            self._model = SentenceTransformer(self.model_name_or_path, device=self.device)
        return self._model

    def ensure_collection(
        self,
        dimension: int = 384,
        recreate: bool = False,
        corpus_fingerprint: Optional[str] = None,
        preprocessing_fingerprint: Optional[str] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
        num_chunks: Optional[int] = None,
    ) -> None:
        """
        Validates collection existence, vector dimension, distance function, and stored manifest identity
        including corpus fingerprint and preprocessing configuration.
        Creates collection with manifest if absent, or verifies compatibility if present.
        """
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = self._get_client()
        collections_res = client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections_res)

        if exists and recreate:
            logger.info("Recreating Qdrant collection '%s'...", self.collection_name)
            client.delete_collection(self.collection_name)
            exists = False

        target_dataset = dataset_name or self.dataset_name or "BEIR-SciFact"
        target_corpus_fp = corpus_fingerprint or self.expected_corpus_fingerprint
        target_prep_fp = preprocessing_fingerprint or self.expected_preprocessing_fingerprint
        target_prep_cfg = preprocessing_config or self.expected_preprocessing_config
        if target_prep_cfg is not None and target_prep_fp is None:
            target_prep_fp = compute_preprocessing_fingerprint(target_prep_cfg)

        if not exists:
            logger.info(
                "Creating Qdrant collection '%s' (dim=%d, distance=COSINE)...",
                self.collection_name,
                dimension,
            )
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            manifest_payload = {
                "is_collection_manifest": True,
                "manifest_version": "2.0",
                "collection_name": self.collection_name,
                "model_name": self.model_name_or_path,
                "model_revision": self.hub_revision,
                "dimension": dimension,
                "distance": "Cosine",
                "dataset_name": target_dataset,
            }
            if target_corpus_fp:
                manifest_payload["corpus_fingerprint"] = target_corpus_fp
            if target_prep_fp:
                manifest_payload["preprocessing_fingerprint"] = target_prep_fp
            if target_prep_cfg:
                manifest_payload["preprocessing_config"] = target_prep_cfg
            if num_chunks is not None:
                manifest_payload["num_chunks"] = num_chunks

            client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=MANIFEST_POINT_ID,
                        vector=[0.0] * dimension,
                        payload=manifest_payload,
                    )
                ],
                wait=True,
            )
            self._collection_validated = True
        else:
            info = client.get_collection(self.collection_name)
            vectors_cfg = info.config.params.vectors
            existing_size = vectors_cfg.size if hasattr(vectors_cfg, "size") else None
            if existing_size and existing_size != dimension:
                raise IncompatibleIndexError(
                    f"Qdrant collection '{self.collection_name}' has vector dimension {existing_size}, "
                    f"but model '{self.model_name_or_path}' produces dimension {dimension}."
                )

            try:
                manifest_points = client.retrieve(
                    collection_name=self.collection_name,
                    ids=[MANIFEST_POINT_ID],
                    with_payload=True,
                )
            except Exception:
                manifest_points = []

            if manifest_points:
                m_payload = manifest_points[0].payload or {}
                m_model = m_payload.get("model_name")
                m_dim = m_payload.get("dimension")
                m_dist = m_payload.get("distance")
                m_dataset = m_payload.get("dataset_name")
                m_corpus_fp = m_payload.get("corpus_fingerprint")
                m_prep_fp = m_payload.get("preprocessing_fingerprint")
                m_prep_cfg = m_payload.get("preprocessing_config")

                if m_model and m_model != self.model_name_or_path:
                    raise IncompatibleIndexError(
                        f"Qdrant collection '{self.collection_name}' was indexed with model '{m_model}', "
                        f"which is incompatible with current model '{self.model_name_or_path}'."
                    )
                if m_dim and m_dim != dimension:
                    raise IncompatibleIndexError(
                        f"Qdrant collection '{self.collection_name}' manifest indicates dimension {m_dim}, "
                        f"which does not match expected dimension {dimension}."
                    )
                if m_dist and m_dist != "Cosine":
                    raise IncompatibleIndexError(
                        f"Qdrant collection '{self.collection_name}' manifest indicates distance metric {m_dist}, "
                        "expected 'Cosine'."
                    )
                if target_dataset and m_dataset and m_dataset != target_dataset:
                    raise IncompatibleIndexError(
                        f"Qdrant collection '{self.collection_name}' was indexed for dataset '{m_dataset}', "
                        f"which does not match expected dataset '{target_dataset}'."
                    )
                if target_corpus_fp and m_corpus_fp and m_corpus_fp != target_corpus_fp:
                    raise IncompatibleIndexError(
                        f"Qdrant collection '{self.collection_name}' was indexed with corpus fingerprint '{m_corpus_fp}', "
                        f"which does not match expected corpus fingerprint '{target_corpus_fp}'."
                    )
                if target_prep_fp and m_prep_fp and m_prep_fp != target_prep_fp:
                    raise IncompatibleIndexError(
                        f"Qdrant collection '{self.collection_name}' was indexed with preprocessing fingerprint '{m_prep_fp}', "
                        f"which does not match expected preprocessing fingerprint '{target_prep_fp}'."
                    )
                if target_prep_cfg and m_prep_cfg:
                    for k, v in target_prep_cfg.items():
                        if k in m_prep_cfg and m_prep_cfg[k] != v:
                            raise IncompatibleIndexError(
                                f"Qdrant collection '{self.collection_name}' was indexed with preprocessing config '{k}={m_prep_cfg[k]}', "
                                f"which does not match expected '{k}={v}'."
                            )
            else:
                count_info = client.count(self.collection_name, exact=True)
                if count_info.count > 0:
                    raise IncompatibleIndexError(
                        f"Qdrant collection '{self.collection_name}' contains {count_info.count} points "
                        "but is missing the authoritative collection manifest point. Rebuild required."
                    )
                else:
                    manifest_payload = {
                        "is_collection_manifest": True,
                        "manifest_version": "2.0",
                        "collection_name": self.collection_name,
                        "model_name": self.model_name_or_path,
                        "model_revision": self.hub_revision,
                        "dimension": dimension,
                        "distance": "Cosine",
                        "dataset_name": target_dataset,
                    }
                    if target_corpus_fp:
                        manifest_payload["corpus_fingerprint"] = target_corpus_fp
                    if target_prep_fp:
                        manifest_payload["preprocessing_fingerprint"] = target_prep_fp
                    if target_prep_cfg:
                        manifest_payload["preprocessing_config"] = target_prep_cfg
                    if num_chunks is not None:
                        manifest_payload["num_chunks"] = num_chunks

                    client.upsert(
                        collection_name=self.collection_name,
                        points=[
                            PointStruct(
                                id=MANIFEST_POINT_ID,
                                vector=[0.0] * dimension,
                                payload=manifest_payload,
                            )
                        ],
                        wait=True,
                    )
            self._collection_validated = True

    def index(
        self,
        chunks: List[Chunk],
        batch_size: int = 64,
        recreate_collection: bool = False,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
    ) -> None:
        """
        Encodes and batch-upserts document chunks into Qdrant.
        Operation is strictly idempotent: re-indexing identical chunk_ids updates payload without duplicating points.
        Validates index compatibility across corpus, preprocessing, and model before indexing.
        """
        from qdrant_client.models import PointStruct

        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

        model = self._get_model()
        dimension = model.get_sentence_embedding_dimension()

        corpus_fp = compute_chunks_corpus_fingerprint(chunks)
        prep_cfg = preprocessing_config or self.expected_preprocessing_config
        prep_fp = compute_preprocessing_fingerprint(prep_cfg) if prep_cfg else None

        self.ensure_collection(
            dimension=dimension,
            recreate=recreate_collection,
            corpus_fingerprint=corpus_fp,
            preprocessing_fingerprint=prep_fp,
            preprocessing_config=prep_cfg,
            dataset_name=dataset_name or self.dataset_name,
            num_chunks=len(chunks),
        )

        client = self._get_client()
        self.chunks = chunks

        logger.info("Indexing %d chunks into Qdrant collection '%s'...", len(chunks), self.collection_name)

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            texts = [c.text for c in batch_chunks]

            embeddings = model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
                device=self.device,
            )

            points = []
            for chunk, vec in zip(batch_chunks, embeddings):
                point_id = chunk_id_to_uuid(chunk.chunk_id)
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vec.tolist(),
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        manifest_payload = {
            "is_collection_manifest": True,
            "manifest_version": "2.0",
            "collection_name": self.collection_name,
            "model_name": self.model_name_or_path,
            "model_revision": self.hub_revision,
            "dimension": dimension,
            "distance": "Cosine",
            "dataset_name": dataset_name or self.dataset_name or "BEIR-SciFact",
            "corpus_fingerprint": corpus_fp,
            "num_chunks": len(chunks),
        }
        if prep_fp:
            manifest_payload["preprocessing_fingerprint"] = prep_fp
        if prep_cfg:
            manifest_payload["preprocessing_config"] = prep_cfg

        client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=MANIFEST_POINT_ID,
                    vector=[0.0] * dimension,
                    payload=manifest_payload,
                )
            ],
            wait=True,
        )

        logger.info("Successfully indexed %d chunks into Qdrant collection '%s'.", len(chunks), self.collection_name)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Executes dense vector similarity search in Qdrant.
        Validates collection compatibility upon initialization.
        Returns ordered RetrievalResult objects with exact chunk provenance.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        client = self._get_client()
        model = self._get_model()

        if not self._collection_validated:
            dimension = model.get_sentence_embedding_dimension()
            self.ensure_collection(dimension=dimension, recreate=False)

        query_vec = model.encode(
            [query],
            normalize_embeddings=True,
            device=self.device,
        )[0]

        try:
            hits_res = client.query_points(
                collection_name=self.collection_name,
                query=query_vec.tolist(),
                limit=top_k,
                with_payload=True,
            )
            hits = hits_res.points
        except Exception:
            hits = client.search(
                collection_name=self.collection_name,
                query_vector=query_vec.tolist(),
                limit=top_k,
                with_payload=True,
            )

        # Sort with deterministic tie-breaking by (score DESC, chunk_id ASC)
        scored_hits = []
        for hit in hits:
            if str(hit.id) == MANIFEST_POINT_ID:
                continue
            payload = hit.payload or {}
            if payload.get("is_collection_manifest"):
                continue
            chunk_id = payload.get("chunk_id", str(hit.id))
            scored_hits.append((float(hit.score), chunk_id, payload))

        scored_hits.sort(key=lambda item: (-item[0], item[1]))

        results: List[RetrievalResult] = []
        for rank, (score, chunk_id, payload) in enumerate(scored_hits[:top_k], start=1):
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    doc_id=payload.get("doc_id", ""),
                    score=float(score),
                    rank=rank,
                    title=payload.get("title", ""),
                    text=payload.get("text", ""),
                    metadata=payload.get("metadata", {}).copy(),
                )
            )

        return results

    def save(self, path: Path) -> None:
        """Saves Qdrant connection and collection metadata to directory."""
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        manifest_data = {
            "retriever_type": "qdrant",
            "collection_name": self.collection_name,
            "host": self.host,
            "port": self.port,
            "grpc_port": self.grpc_port,
            "model_name_or_path": self.model_name_or_path,
            "device": self.device,
        }
        with open(path / "qdrant_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "QdrantRetriever":
        """Loads Qdrant retriever from metadata directory."""
        path = Path(path).resolve()
        manifest_file = path / "qdrant_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Missing qdrant_manifest.json at {path}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            collection_name=data.get("collection_name", "scifact_chunks"),
            host=data.get("host", "127.0.0.1"),
            port=data.get("port", 6335),
            grpc_port=data.get("grpc_port", 6336),
            model_name_or_path=data.get("model_name_or_path", "BAAI/bge-small-en-v1.5"),
            device=data.get("device", "cpu"),
        )
