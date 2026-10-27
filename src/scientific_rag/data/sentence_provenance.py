"""
Canonical SciFact source sentence provenance mapping.
Loads official SciFact corpus abstracts, tracks 0-based sentence indices,
and maps retrieved chunks to exact source sentence IDs.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("scientific_rag.data.sentence_provenance")


@dataclass
class CanonicalSentence:
    """Represents a canonical source sentence from the official SciFact corpus."""
    doc_id: str
    sentence_idx: int
    text: str


@dataclass
class ChunkSentenceMapping:
    """Maps a retrieved chunk to its constituent canonical source sentences."""
    chunk_id: str
    doc_id: str
    sentence_indices: List[int]
    sentences: List[CanonicalSentence]


class SciFactSentenceProvenance:
    """
    Manages canonical sentence provenance for the SciFact dataset.
    
    Invariants:
    1. Preserves official SciFact abstract sentences and their exact 0-based indices.
    2. Maps retrieved chunks ({doc_id}#c{chunk_index}) to canonical sentence IDs deterministically.
    3. Handles title prefixes and sentence normalization safely.
    4. Provides evidence formatting exposing [SENTENCE {idx}] tags for the LLM generator.
    """

    def __init__(self, corpus_path: Optional[Path] = None):
        self.corpus_path = corpus_path
        self._doc_sentences: Dict[str, List[CanonicalSentence]] = {}
        self._is_loaded = False

    def load_corpus(self, corpus_path: Optional[Path] = None) -> None:
        """Loads official SciFact corpus (data/raw/scifact_official/data/corpus.jsonl)."""
        target_path = corpus_path or self.corpus_path
        if not target_path or not target_path.exists():
            candidates = [
                Path("data/raw/scifact_official/data/corpus.jsonl"),
                Path("data/raw/scifact/scifact_official/data/corpus.jsonl"),
            ]
            target_path = next((p for p in candidates if p.exists()), None)

        if not target_path or not target_path.exists():
            logger.warning("Official SciFact corpus not found at %s. Canonical sentence mapping unavailable.", target_path)
            return

        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_id = str(data.get("doc_id", "")).strip()
                abstract_sentences = data.get("abstract", [])
                
                sentences: List[CanonicalSentence] = []
                for idx, sent_text in enumerate(abstract_sentences):
                    sentences.append(
                        CanonicalSentence(
                            doc_id=doc_id,
                            sentence_idx=idx,
                            text=sent_text.strip(),
                        )
                    )
                self._doc_sentences[doc_id] = sentences

        self._is_loaded = True
        logger.info("Loaded canonical sentence provenance for %d documents from %s.", len(self._doc_sentences), target_path)

    def get_document_sentences(self, doc_id: str) -> List[CanonicalSentence]:
        """Returns the canonical sentences for a document."""
        if not self._is_loaded:
            self.load_corpus()
        return self._doc_sentences.get(str(doc_id), [])

    def map_chunk_to_sentences(self, doc_id: str, chunk_id: str, chunk_text: str) -> ChunkSentenceMapping:
        """
        Maps a chunk to its canonical source sentences by normalized string containment.
        Handles title prefix and overlapping chunks.
        """
        doc_sentences = self.get_document_sentences(doc_id)
        if not doc_sentences:
            return ChunkSentenceMapping(
                chunk_id=chunk_id,
                doc_id=doc_id,
                sentence_indices=[],
                sentences=[],
            )

        # Normalize chunk text (strip title if chunk starts with "Title: ...")
        clean_chunk = re.sub(r"^.*?:\s*", "", chunk_text).lower()

        matched_sentences: List[CanonicalSentence] = []
        matched_indices: List[int] = []

        # Canonical sentence alignment:
        # Strict normalized substring matching only.
        # Fuzzy 75% token overlap is eliminated to prevent semantic inversions or negated sentence mismatches.
        clean_chunk_collapsed = re.sub(r"\s+", " ", clean_chunk).strip()

        for sent in doc_sentences:
            sent_norm = re.sub(r"\s+", " ", sent.text.lower()).strip()
            if not sent_norm:
                continue

            if sent_norm in clean_chunk_collapsed or clean_chunk_collapsed in sent_norm:
                matched_sentences.append(sent)
                matched_indices.append(sent.sentence_idx)
            else:
                # Allow minor trailing punctuation variation (e.g. trailing period)
                sent_stripped = sent_norm.rstrip(".").rstrip()
                if sent_stripped and sent_stripped in clean_chunk_collapsed:
                    matched_sentences.append(sent)
                    matched_indices.append(sent.sentence_idx)

        return ChunkSentenceMapping(
            chunk_id=chunk_id,
            doc_id=doc_id,
            sentence_indices=matched_indices,
            sentences=matched_sentences,
        )

    def find_matching_sentences(
        self,
        doc_id: str,
        quotation: str,
        chunk_sentences: Optional[List[CanonicalSentence]] = None,
    ) -> List[CanonicalSentence]:
        """
        Finds all canonical sentences in the document (or chunk) containing the given quotation.
        Used to detect ambiguous quotations that appear in multiple sentences.
        """
        if not quotation or not quotation.strip():
            return []

        clean_quote = re.sub(r"\s+", " ", quotation.lower()).strip().rstrip(".").rstrip()
        if not clean_quote:
            return []

        candidates = chunk_sentences if chunk_sentences is not None else self.get_document_sentences(doc_id)
        matches: List[CanonicalSentence] = []

        for sent in candidates:
            sent_norm = re.sub(r"\s+", " ", sent.text.lower()).strip().rstrip(".").rstrip()
            if not sent_norm:
                continue

            if clean_quote in sent_norm or sent_norm in clean_quote:
                matches.append(sent)

        return matches

    def format_evidence_with_sentences(self, chunk_results: list) -> str:
        """
        Formats retrieved chunk evidence exposing explicit [SENTENCE {idx}] tags.
        Allows the LLM generator to accurately attribute statements to canonical sentence IDs.
        """
        if not chunk_results:
            return "[No evidence snippets retrieved.]"

        formatted_blocks = []
        for rank_idx, item in enumerate(chunk_results, start=1):
            mapping = self.map_chunk_to_sentences(item.doc_id, item.chunk_id, item.text)

            block_lines = [
                f"[{rank_idx}] Document ID: {item.doc_id} | Chunk ID: {item.chunk_id}",
                f"Title: {item.title}",
                "Evidence Sentences:"
            ]

            if mapping.sentences:
                for sent in mapping.sentences:
                    block_lines.append(f"  - [SENTENCE {sent.sentence_idx}]: {sent.text}")
            else:
                block_lines.append(f"  - Snippet: {item.text.strip()}")

            formatted_blocks.append("\n".join(block_lines))

        return "\n---\n".join(formatted_blocks)
