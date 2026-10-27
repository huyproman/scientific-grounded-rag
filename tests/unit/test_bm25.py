"""
Unit tests for BM25Okapi retriever indexing, ranking, and serialization.
"""

import tempfile
from pathlib import Path
from scientific_rag.data.models import Chunk
from scientific_rag.retrieval.bm25 import BM25Retriever


def test_bm25_retrieval_and_ranking():
    chunks = [
        Chunk(
            chunk_id="doc1#c0",
            doc_id="doc1",
            title="Genetics",
            text="CRISPR Cas9 endonuclease enables targeted genetic engineering in mammalian genomes.",
            chunk_index=0,
        ),
        Chunk(
            chunk_id="doc2#c0",
            doc_id="doc2",
            title="Astrophysics",
            text="Gravitational waves emitted from binary black hole coalescences warp spacetime.",
            chunk_index=0,
        ),
        Chunk(
            chunk_id="doc3#c0",
            doc_id="doc3",
            title="Neuroscience",
            text="Synaptic plasticity in hippocampal pyramidal neurons underlies spatial memory consolidation.",
            chunk_index=0,
        ),
    ]

    retriever = BM25Retriever()
    retriever.index(chunks)

    results = retriever.retrieve("CRISPR Cas9 genetic engineering", top_k=2)
    assert len(results) == 2
    # Top result must be doc1
    assert results[0].doc_id == "doc1"
    assert results[0].rank == 1
    assert results[0].score > 0.0


def test_bm25_save_and_load():
    chunks = [
        Chunk(
            chunk_id="doc1#c0",
            doc_id="doc1",
            title="Test",
            text="Information retrieval evaluation involves recall and precision metrics.",
            chunk_index=0,
        )
    ]

    retriever = BM25Retriever()
    retriever.index(chunks)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "bm25_test.pkl"
        retriever.save(save_path)

        loaded_retriever = BM25Retriever.load(save_path)
        results = loaded_retriever.retrieve("information retrieval", top_k=1)

        assert len(results) == 1
        assert results[0].chunk_id == "doc1#c0"
