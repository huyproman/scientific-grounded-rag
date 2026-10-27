#!/usr/bin/env python3
"""Script to build and serialize retrieval indices with compatibility manifests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scientific_rag.common.compatibility import PreprocessingConfig, resolve_dataset_fingerprint
from scientific_rag.common.paths import INDEX_ARTIFACTS_DIR, RAW_DATA_DIR
from scientific_rag.data.loader import DatasetLoader
from scientific_rag.preprocessing.chunker import DeterministicSentenceChunker
from scientific_rag.retrieval.bm25 import BM25Retriever
from scientific_rag.retrieval.dense import DenseRetriever

if __name__ == "__main__":
    loader = DatasetLoader(RAW_DATA_DIR)
    docs = loader.load_corpus()
    prep_config = PreprocessingConfig(sentences_per_chunk=3, sentence_overlap=1, include_title=True)
    chunker = DeterministicSentenceChunker(
        sentences_per_chunk=prep_config.sentences_per_chunk,
        sentence_overlap=prep_config.sentence_overlap,
        include_title_in_text=prep_config.include_title,
    )
    chunks = chunker.chunk_corpus(docs)
    print(f"Loaded {len(docs)} documents, created {len(chunks)} chunks.")

    dataset_name, dataset_hash = resolve_dataset_fingerprint(RAW_DATA_DIR)
    INDEX_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building BM25 index with compatibility manifest...")
    bm25 = BM25Retriever()
    bm25.index(chunks)
    bm25.save(
        INDEX_ARTIFACTS_DIR / "bm25_index.pkl",
        dataset_name=dataset_name,
        dataset_hash=dataset_hash,
        preprocessing_config=prep_config,
    )
    print("BM25 index saved.")

    print("Building Dense BGE index on CPU with compatibility manifest...")
    dense = DenseRetriever(device="cpu", batch_size=64)
    dense.index(chunks)
    dense.save(
        INDEX_ARTIFACTS_DIR / "dense_index",
        dataset_name=dataset_name,
        dataset_hash=dataset_hash,
        preprocessing_config=prep_config,
    )
    print("Dense index saved.")
