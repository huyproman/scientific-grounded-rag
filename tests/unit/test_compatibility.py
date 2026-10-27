"""
Regression tests for F01: Index compatibility & stale artifact detection.
Covers all 7 required test conditions (T1-T7) in Assignment Section 2.
"""

import tempfile
from pathlib import Path
import pytest

from scientific_rag.common.compatibility import (
    BM25IndexConfig,
    DenseIndexConfig,
    IncompatibleIndexError,
    IndexManifest,
    PreprocessingConfig,
)
from scientific_rag.data.models import Chunk
from scientific_rag.retrieval.bm25 import BM25Retriever
from scientific_rag.retrieval.dense import DenseRetriever


def create_sample_chunks():
    return [
        Chunk(
            chunk_id="doc1#c0",
            doc_id="doc1",
            title="Genetics",
            text="CRISPR Cas9 endonuclease enables targeted genetic modifications in mammalian cells.",
            chunk_index=0,
        )
    ]


def test_t1_same_dataset_config_loads_successfully():
    chunks = create_sample_chunks()
    bm25 = BM25Retriever()
    bm25.index(chunks)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "bm25.pkl"
        prep_conf = PreprocessingConfig(sentences_per_chunk=3)
        bm25.save(path, dataset_name="SciFact", dataset_hash="hash_123", preprocessing_config=prep_conf)

        loaded = BM25Retriever.load(
            path,
            expected_dataset_name="SciFact",
            expected_dataset_hash="hash_123",
            expected_preprocessing_config=prep_conf,
        )
        assert loaded.name == bm25.name
        assert len(loaded.chunks) == 1


def test_t2_changed_chunk_size_or_overlap_rejected():
    chunks = create_sample_chunks()
    bm25 = BM25Retriever()
    bm25.index(chunks)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "bm25.pkl"
        bm25.save(path, preprocessing_config=PreprocessingConfig(sentences_per_chunk=3))

        with pytest.raises(IncompatibleIndexError) as exc_info:
            BM25Retriever.load(
                path,
                expected_preprocessing_config=PreprocessingConfig(sentences_per_chunk=5),
            )
        assert "Preprocessing configuration mismatch" in str(exc_info.value)


def test_t3_changed_corpus_identity_rejected():
    chunks = create_sample_chunks()
    bm25 = BM25Retriever()
    bm25.index(chunks)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "bm25.pkl"
        bm25.save(path, dataset_hash="old_corpus_hash")

        with pytest.raises(IncompatibleIndexError) as exc_info:
            BM25Retriever.load(
                path,
                expected_dataset_hash="new_corpus_hash",
            )
        assert "Dataset hash mismatch" in str(exc_info.value)


def test_t4_changed_bm25_parameters_rejected():
    chunks = create_sample_chunks()
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(chunks)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "bm25.pkl"
        bm25.save(path)

        with pytest.raises(IncompatibleIndexError) as exc_info:
            BM25Retriever.load(
                path,
                expected_bm25_config=BM25IndexConfig(k1=2.0, b=0.75),
            )
        assert "Retriever configuration mismatch" in str(exc_info.value)


def test_t5_changed_dense_model_identity_rejected():
    chunks = create_sample_chunks()
    dense = DenseRetriever(device="cpu", batch_size=2)
    dense.index(chunks)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "dense"
        dense.save(path)

        with pytest.raises(IncompatibleIndexError) as exc_info:
            DenseRetriever.load(
                path,
                expected_dense_config=DenseIndexConfig(model_name_or_path="BAAI/bge-large-en-v1.5"),
            )
        assert "Retriever configuration mismatch" in str(exc_info.value)


def test_t6_missing_or_corrupted_index_metadata():
    chunks = create_sample_chunks()
    bm25 = BM25Retriever()
    bm25.index(chunks)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "bm25.pkl"
        # Save legacy pickle without manifest
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"name": "BM25", "k1": 1.5, "b": 0.75, "bm25": bm25.bm25, "chunks": chunks}, f)

        with pytest.raises(IncompatibleIndexError) as exc_info:
            BM25Retriever.load(path)
        assert "missing compatibility manifest" in str(exc_info.value)


def test_t7_dense_missing_manifest_rejected():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dense_dir = Path(tmp_dir) / "dense"
        dense_dir.mkdir()
        # Missing manifest.json
        with pytest.raises(IncompatibleIndexError) as exc_info:
            DenseRetriever.load(dense_dir)
        assert "missing manifest.json" in str(exc_info.value)
