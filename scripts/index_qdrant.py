"""
Qdrant indexing script for scientific corpus.
Supports batch upserting, precomputed embedding reuse, idempotency, and collection compatibility checks.
"""

from dataclasses import asdict
import argparse
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import time
from typing import List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from scientific_rag.common.compatibility import (
    IncompatibleIndexError,
    PreprocessingConfig,
    resolve_dataset_fingerprint,
)
from scientific_rag.common.paths import ARTIFACTS_DIR, INDEX_ARTIFACTS_DIR, RAW_DATA_DIR
from scientific_rag.config import load_config
from scientific_rag.data.loader import DatasetLoader
from scientific_rag.data.models import Chunk
from scientific_rag.preprocessing.chunker import DeterministicSentenceChunker
from scientific_rag.retrieval.qdrant import (
    MANIFEST_POINT_ID,
    QdrantRetriever,
    chunk_id_to_uuid,
    compute_chunks_corpus_fingerprint,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("index_qdrant")


def index_to_qdrant(
    collection_name: str,
    host: str = "127.0.0.1",
    port: int = 6335,
    recreate: bool = False,
    batch_size: int = 256,
    limit: Optional[int] = None,
    use_precomputed: bool = True,
) -> int:
    """
    Indexes document chunks into Qdrant vector database.
    Reuses precomputed embeddings from artifacts/indices/dense_index if available.
    """
    client = QdrantClient(host=host, port=port, timeout=30.0, check_compatibility=False)
    dimension = 384  # BGE-small-en-v1.5 dimension

    # 1. Source chunks and embeddings
    dense_idx_dir = INDEX_ARTIFACTS_DIR / "dense_index"
    chunks: List[Chunk] = []
    embeddings: Optional[np.ndarray] = None
    dataset_name: str = "BEIR-SciFact"
    corpus_fingerprint: Optional[str] = None
    preprocessing_fingerprint: Optional[str] = None
    preprocessing_config: Optional[dict] = None

    if use_precomputed and (dense_idx_dir / "chunks.pkl").exists() and (dense_idx_dir / "embeddings.npy").exists():
        manifest_file = dense_idx_dir / "manifest.json"
        if not manifest_file.exists():
            raise IncompatibleIndexError(f"Missing manifest.json in precomputed dense index at {dense_idx_dir}.")
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        retriever_cfg = manifest_data.get("retriever_config", {})
        if retriever_cfg.get("model_name_or_path") != "BAAI/bge-small-en-v1.5":
            raise IncompatibleIndexError(
                f"Precomputed index model '{retriever_cfg.get('model_name_or_path')}' is incompatible with expected 'BAAI/bge-small-en-v1.5'."
            )
        if retriever_cfg.get("embedding_dimension") != dimension:
            raise IncompatibleIndexError(
                f"Precomputed index dimension {retriever_cfg.get('embedding_dimension')} != {dimension}."
            )

        dataset_name = manifest_data.get("dataset_name", "BEIR-SciFact")
        corpus_fingerprint = manifest_data.get("dataset_hash")
        preprocessing_fingerprint = manifest_data.get("preprocessing_fingerprint")
        preprocessing_config = manifest_data.get("preprocessing_config")

        logger.info("Loading precomputed chunks and embeddings from %s...", dense_idx_dir)
        with open(dense_idx_dir / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        embeddings = np.load(dense_idx_dir / "embeddings.npy")
        logger.info("Loaded %d precomputed chunks and embeddings array %s.", len(chunks), embeddings.shape)
    else:
        logger.info("Precomputed index not found or disabled. Chunking and embedding raw corpus...")
        loader = DatasetLoader(RAW_DATA_DIR)
        corpus = loader.load_corpus()
        chunker = DeterministicSentenceChunker()
        chunks = chunker.chunk_corpus(corpus)

        dataset_name, corpus_fingerprint = resolve_dataset_fingerprint(RAW_DATA_DIR)
        prep_conf = PreprocessingConfig()
        preprocessing_fingerprint = prep_conf.fingerprint()
        preprocessing_config = asdict(prep_conf)

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        texts = [c.text for c in chunks]
        logger.info("Generating embeddings for %d chunks on CPU...", len(texts))
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    if limit and limit > 0:
        chunks = chunks[:limit]
        embeddings = embeddings[:limit]
        logger.info("Limiting indexing to %d chunks.", limit)

    total_chunks = len(chunks)

    # If corpus_fingerprint is still None, compute from chunks directly
    if corpus_fingerprint is None and chunks:
        corpus_fingerprint = compute_chunks_corpus_fingerprint(chunks)

    # 2. Collection management and compatibility via QdrantRetriever
    retriever_helper = QdrantRetriever(
        collection_name=collection_name,
        host=host,
        port=port,
        client=client,
        dataset_name=dataset_name,
        expected_corpus_fingerprint=corpus_fingerprint,
        expected_preprocessing_fingerprint=preprocessing_fingerprint,
        expected_preprocessing_config=preprocessing_config,
    )
    retriever_helper.ensure_collection(
        dimension=dimension,
        recreate=recreate,
        corpus_fingerprint=corpus_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        preprocessing_config=preprocessing_config,
        dataset_name=dataset_name,
        num_chunks=total_chunks,
    )
    logger.info("Collection '%s' validated and ready for indexing.", collection_name)
    logger.info("Upserting %d points into Qdrant collection '%s' in batches of %d...", total_chunks, collection_name, batch_size)

    start_time = time.time()
    for i in tqdm(range(0, total_chunks, batch_size), desc="Upserting to Qdrant"):
        batch_chunks = chunks[i : i + batch_size]
        batch_vecs = embeddings[i : i + batch_size]

        points = []
        for chunk, vec in zip(batch_chunks, batch_vecs):
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
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    manifest_payload = {
        "is_collection_manifest": True,
        "manifest_version": "2.0",
        "collection_name": collection_name,
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimension": dimension,
        "distance": "Cosine",
        "dataset_name": dataset_name,
        "corpus_fingerprint": corpus_fingerprint,
        "num_chunks": total_chunks,
    }
    if preprocessing_fingerprint:
        manifest_payload["preprocessing_fingerprint"] = preprocessing_fingerprint
    if preprocessing_config:
        manifest_payload["preprocessing_config"] = preprocessing_config

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=MANIFEST_POINT_ID,
                vector=[0.0] * dimension,
                payload=manifest_payload,
            )
        ],
        wait=True,
    )

    elapsed = time.time() - start_time
    count_info = client.count(collection_name=collection_name, exact=True)
    logger.info(
        "Successfully indexed %d chunks into '%s' in %.2fs (exact collection count: %d).",
        total_chunks,
        collection_name,
        elapsed,
        count_info.count,
    )
    return count_info.count


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Index scientific passage chunks into Qdrant vector database.")
    parser.add_argument("--collection", type=str, default=config.qdrant_collection, help="Qdrant collection name")
    parser.add_argument("--host", type=str, default=config.qdrant_host, help="Qdrant host")
    parser.add_argument("--port", type=int, default=config.qdrant_port, help="Qdrant port")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection if it already exists")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for upserting points")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of chunks to index")
    parser.add_argument("--no-precomputed", action="store_true", help="Do not use precomputed embeddings")

    args = parser.parse_args()
    index_to_qdrant(
        collection_name=args.collection,
        host=args.host,
        port=args.port,
        recreate=args.recreate,
        batch_size=args.batch_size,
        limit=args.limit,
        use_precomputed=not args.no_precomputed,
    )


if __name__ == "__main__":
    main()
