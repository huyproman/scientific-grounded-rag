"""
Unit tests for deterministic sentence chunking and stable chunk IDs.
"""

from scientific_rag.data.models import Document
from scientific_rag.preprocessing.chunker import DeterministicSentenceChunker


def test_chunker_stable_identifiers():
    doc = Document(
        doc_id="paper_42",
        title="CRISPR Editing",
        text="Sentence one. Sentence two. Sentence three. Sentence four.",
        metadata={"author": "Alice"},
    )

    chunker = DeterministicSentenceChunker(sentences_per_chunk=2, sentence_overlap=1)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 3
    # Check stable ID format: {doc_id}#c{chunk_index}
    assert chunks[0].chunk_id == "paper_42#c0"
    assert chunks[1].chunk_id == "paper_42#c1"
    assert chunks[2].chunk_id == "paper_42#c2"

    assert chunks[0].doc_id == "paper_42"
    assert chunks[0].title == "CRISPR Editing"
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["author"] == "Alice"


def test_chunker_determinism():
    doc = Document(
        doc_id="paper_99",
        title="Epigenetics",
        text="First fact here. Second fact there. Third fact everywhere.",
    )
    chunker = DeterministicSentenceChunker(sentences_per_chunk=2, sentence_overlap=0)

    run_1 = chunker.chunk_document(doc)
    run_2 = chunker.chunk_document(doc)

    assert len(run_1) == len(run_2)
    for c1, c2 in zip(run_1, run_2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.chunk_index == c2.chunk_index


def test_chunker_empty_and_single_sentence():
    chunker = DeterministicSentenceChunker()

    # Empty document
    empty_doc = Document(doc_id="empty_1", title="", text="")
    assert chunker.chunk_document(empty_doc) == []

    # Title-only document
    title_doc = Document(doc_id="title_only", title="Important Paper Title", text="")
    chunks = chunker.chunk_document(title_doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "title_only#c0"
    assert chunks[0].text == "Important Paper Title"
