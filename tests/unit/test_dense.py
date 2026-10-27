"""
Unit tests for Dense retriever (BGE-Small on CPU) indexing, retrieval, and serialization.
"""

import tempfile
from pathlib import Path
from scientific_rag.data.models import Chunk
from scientific_rag.retrieval.dense import DenseRetriever


def test_dense_retrieval_synthetic():
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            title="Genetics",
            text="CRISPR Cas9 endonuclease enables targeted genetic modification in eukaryotic cells.",
            chunk_index=0,
        ),
        Chunk(
            chunk_id="c2",
            doc_id="d2",
            title="Astrophysics",
            text="Black hole mergers create gravitational wave ripples through spacetime curvature.",
            chunk_index=0,
        ),
    ]

    retriever = DenseRetriever(device="cpu", batch_size=2)
    retriever.index(chunks)

    results = retriever.retrieve("genetic editing Cas9", top_k=1)
    assert len(results) == 1
    assert results[0].doc_id == "d1"
    assert results[0].rank == 1


def test_dense_save_and_load():
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            title="Test",
            text="Semantic search evaluates embeddings using cosine similarity.",
            chunk_index=0,
        )
    ]

    retriever = DenseRetriever(device="cpu", batch_size=2)
    retriever.index(chunks)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = Path(tmp_dir) / "dense_test"
        retriever.save(save_dir)

        loaded = DenseRetriever.load(save_dir, device="cpu")
        results = loaded.retrieve("semantic search", top_k=1)

        assert len(results) == 1
        assert results[0].doc_id == "d1"
