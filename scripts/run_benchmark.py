#!/usr/bin/env python3
"""Script to execute the official SciFact IR benchmark across both baselines with compatibility validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scientific_rag.common.compatibility import PreprocessingConfig, resolve_dataset_fingerprint
from scientific_rag.common.paths import INDEX_ARTIFACTS_DIR, RAW_DATA_DIR
from scientific_rag.data.loader import DatasetLoader
from scientific_rag.evaluation.benchmark import run_retrieval_benchmark
from scientific_rag.retrieval.bm25 import BM25Retriever
from scientific_rag.retrieval.dense import DenseRetriever

if __name__ == "__main__":
    split = "test" if "--train" not in sys.argv else "train"
    dataset_name, dataset_hash = resolve_dataset_fingerprint(RAW_DATA_DIR)
    prep_config = PreprocessingConfig()

    loader = DatasetLoader(RAW_DATA_DIR)
    queries = loader.load_queries(split=split)
    qrels = loader.load_qrels(split=split)
    print(f"Loaded {len(queries)} queries and qrels for split '{split}'.")

    bm25 = BM25Retriever.load(
        INDEX_ARTIFACTS_DIR / "bm25_index.pkl",
        expected_dataset_name=dataset_name,
        expected_dataset_hash=dataset_hash,
        expected_preprocessing_config=prep_config,
    )
    bm25_res = run_retrieval_benchmark(bm25, queries, qrels, split_name=split)
    print("BM25 Results:", bm25_res.metrics)

    dense = DenseRetriever.load(
        INDEX_ARTIFACTS_DIR / "dense_index",
        device="cpu",
        expected_dataset_name=dataset_name,
        expected_dataset_hash=dataset_hash,
        expected_preprocessing_config=prep_config,
    )
    dense_res = run_retrieval_benchmark(dense, queries, qrels, split_name=split)
    print("Dense BGE Results:", dense_res.metrics)
