"""
Deterministic, context-preserving sentence chunker with stable chunk identification.
"""

import re
from typing import Any, Dict, List

from scientific_rag.data.models import Chunk, Document


class DeterministicSentenceChunker:
    """
    Chunks scientific documents into sentences with stable identifiers.
    Chunk ID specification: {doc_id}#c{chunk_index}
    """

    def __init__(
        self,
        sentences_per_chunk: int = 3,
        sentence_overlap: int = 1,
        include_title_in_text: bool = True,
    ):
        if sentences_per_chunk <= 0:
            raise ValueError("sentences_per_chunk must be positive integer.")
        if sentence_overlap < 0 or sentence_overlap >= sentences_per_chunk:
            raise ValueError("sentence_overlap must be >= 0 and < sentences_per_chunk.")

        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.include_title_in_text = include_title_in_text

        # Regex for splitting scientific text into sentences respecting abbreviations
        self._sentence_regex = re.compile(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+"
        )

    def split_sentences(self, text: str) -> List[str]:
        """Splits raw text into clean, non-empty sentences."""
        clean_text = text.strip()
        if not clean_text:
            return []
        sentences = [s.strip() for s in self._sentence_regex.split(clean_text) if s.strip()]
        return sentences

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Chunks a single document into stable Chunk objects."""
        raw_sentences = self.split_sentences(doc.text)

        # If document text is empty, but title exists, treat title as single chunk
        if not raw_sentences:
            if doc.title:
                return [
                    Chunk(
                        chunk_id=f"{doc.doc_id}#c0",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=doc.title,
                        chunk_index=0,
                        metadata=doc.metadata.copy(),
                    )
                ]
            return []

        chunks: List[Chunk] = []
        step = self.sentences_per_chunk - self.sentence_overlap
        chunk_idx = 0

        for start_idx in range(0, len(raw_sentences), step):
            end_idx = min(start_idx + self.sentences_per_chunk, len(raw_sentences))
            chunk_sentences = raw_sentences[start_idx:end_idx]
            if not chunk_sentences:
                continue

            chunk_body = " ".join(chunk_sentences)
            if self.include_title_in_text and doc.title:
                chunk_text = f"{doc.title}: {chunk_body}"
            else:
                chunk_text = chunk_body

            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#c{chunk_idx}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=chunk_text,
                    chunk_index=chunk_idx,
                    metadata={
                        **doc.metadata,
                        "sentence_start": start_idx,
                        "sentence_end": end_idx,
                    },
                )
            )
            chunk_idx += 1

            if end_idx >= len(raw_sentences):
                break

        return chunks

    def chunk_corpus(self, documents: List[Document]) -> List[Chunk]:
        """Processes an entire collection of documents deterministically."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
