"""
Citation structural, contextual, and sentence-level validity verification.
Verifies that generated citations match actual documents, chunks, and canonical sentences in the active context,
and enforces that evidence text resolves strictly against authoritative retrieved chunk contents.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from scientific_rag.data.models import RetrievalResult
from scientific_rag.data.sentence_provenance import SciFactSentenceProvenance
from scientific_rag.generation.schemas import (
    Citation,
    CitationVerificationResult,
    VerificationOutcome,
)


def normalize_text_for_comparison(text: str) -> str:
    """Normalizes whitespace and punctuation for robust evidence snippet comparison."""
    if not text:
        return ""
    clean = re.sub(r"\s+", " ", text).strip().lower()
    return clean


class CitationVerifier:
    """
    Verifies citation validity against retrieved evidence context.
    
    Invariants:
    1. parent_doc_id is non-empty and well-formed.
    2. chunk_id is non-empty and well-formed.
    3. chunk exists in the provided retrieval evidence context.
    4. chunk belongs to the claimed parent_doc_id.
    5. Exact quotation matching: quotation MUST be an exact normalized substring of the chunk.
    6. Sentence attribution validation : verifies claimed sentence_idx against canonical sentences.
    """

    def __init__(self, provenance: Optional[SciFactSentenceProvenance] = None):
        self.provenance = provenance

    def verify_citations(
        self,
        citations: List[Citation],
        evidence_items: List[RetrievalResult],
    ) -> List[CitationVerificationResult]:
        """Verifies each citation against the supplied evidence context with text and sentence integrity checks."""
        context_chunks: Dict[str, RetrievalResult] = {
            item.chunk_id: item for item in evidence_items
        }

        results: List[CitationVerificationResult] = []

        for citation in citations:
            doc_id = citation.parent_doc_id.strip() if citation.parent_doc_id else ""
            chunk_id = citation.chunk_id.strip() if citation.chunk_id else ""

            is_valid_structure = bool(doc_id and chunk_id)

            is_contextual = False
            is_text_authentic = False
            authoritative_snippet: Optional[str] = None
            notes = []

            # Multidimensional attribution validity fields
            claimed_sentence_idx = citation.sentence_idx
            is_sentence_valid = False
            is_ambiguous_sentence = False
            sentence_evaluable = (claimed_sentence_idx is not None)
            authoritative_sentence_text: Optional[str] = None

            if not is_valid_structure:
                notes.append("Empty or malformed parent_doc_id or chunk_id.")
            elif chunk_id not in context_chunks:
                notes.append(f"Chunk ID '{chunk_id}' was not present in the retrieved evidence context.")
            else:
                chunk = context_chunks[chunk_id]
                if chunk.doc_id != doc_id:
                    notes.append(
                        f"Chunk '{chunk_id}' belongs to document '{chunk.doc_id}', not cited '{doc_id}'."
                    )
                else:
                    is_contextual = True
                    authoritative_snippet = chunk.text

                    # Enforce strict evidence text integrity
                    if citation.evidence_text:
                        norm_cited = normalize_text_for_comparison(citation.evidence_text)
                        norm_source = normalize_text_for_comparison(chunk.text)

                        # Clean trailing ellipses or punctuation
                        norm_cited = norm_cited.rstrip(".").rstrip()

                        if norm_cited and norm_cited in norm_source:
                            is_text_authentic = True
                        else:
                            is_text_authentic = False
                            notes.append(
                                "Evidence text mismatch: quotation is not an exact normalized substring of the authoritative source chunk."
                            )
                    else:
                        is_text_authentic = False
                        notes.append("No quotation text supplied in citation.")

                    # Sentence Attribution Verification:
                    # Verifies that:
                    # 1. sentence_idx belongs to this chunk.
                    # 2. cited quotation uniquely identifies the claimed sentence without ambiguity.
                    # 3. quotation belongs to that exact sentence, not another sentence in the chunk.
                    if citation.sentence_idx is not None and self.provenance is not None:
                        mapping = self.provenance.map_chunk_to_sentences(doc_id, chunk_id, chunk.text)
                        if citation.sentence_idx not in mapping.sentence_indices:
                            notes.append(
                                f"Claimed sentence_idx {citation.sentence_idx} does not belong to retrieved chunk {chunk_id}."
                            )
                            citation.sentence_idx = None
                        elif citation.evidence_text:
                            # Check ambiguity: find all sentences in chunk containing quotation
                            matching_sents = self.provenance.find_matching_sentences(
                                doc_id, citation.evidence_text, chunk_sentences=mapping.sentences
                            )
                            if len(matching_sents) > 1:
                                is_ambiguous_sentence = True
                                notes.append(
                                    f"Ambiguous quotation matches multiple canonical sentences {[s.sentence_idx for s in matching_sents]} in chunk {chunk_id}; unique sentence attribution cannot be proven."
                                )
                                citation.sentence_idx = None
                            elif len(matching_sents) == 1:
                                target_sent = matching_sents[0]
                                if target_sent.sentence_idx == citation.sentence_idx:
                                    is_sentence_valid = True
                                    authoritative_sentence_text = target_sent.text
                                else:
                                    notes.append(
                                        f"Quotation mismatch for sentence_idx {citation.sentence_idx}: quotation belongs to canonical sentence {target_sent.sentence_idx}, not claimed sentence {citation.sentence_idx}."
                                    )
                                    citation.sentence_idx = None
                            else:
                                target_sent = next((s for s in mapping.sentences if s.sentence_idx == citation.sentence_idx), None)
                                if target_sent is not None:
                                    norm_cited_sent = normalize_text_for_comparison(citation.evidence_text).rstrip(".").rstrip()
                                    norm_target_sent = normalize_text_for_comparison(target_sent.text).rstrip(".").rstrip()
                                    if norm_cited_sent in norm_target_sent or norm_target_sent in norm_cited_sent:
                                        # Check doc-level ambiguity
                                        doc_matches = self.provenance.find_matching_sentences(doc_id, citation.evidence_text)
                                        if len(doc_matches) > 1:
                                            is_ambiguous_sentence = True
                                            notes.append(
                                                f"Ambiguous quotation matches multiple canonical sentences in document {doc_id}; unique attribution cannot be proven."
                                            )
                                            citation.sentence_idx = None
                                        else:
                                            is_sentence_valid = True
                                            authoritative_sentence_text = target_sent.text
                                    else:
                                        notes.append(
                                            f"Quotation mismatch for sentence_idx {citation.sentence_idx}: cited quote does not match text of claimed canonical sentence."
                                        )
                                        citation.sentence_idx = None
                                else:
                                    notes.append(f"Canonical sentence {citation.sentence_idx} not found in chunk {chunk_id}.")
                                    citation.sentence_idx = None

            is_chunk_valid = bool(is_valid_structure and is_contextual and is_text_authentic)

            results.append(
                CitationVerificationResult(
                    citation=citation,
                    is_valid_structure=is_valid_structure,
                    is_contextual=is_contextual,
                    is_text_authentic=is_text_authentic,
                    is_chunk_valid=is_chunk_valid,
                    is_sentence_valid=is_sentence_valid,
                    is_ambiguous_sentence=is_ambiguous_sentence,
                    sentence_evaluable=sentence_evaluable,
                    authoritative_evidence_text=authoritative_snippet,
                    authoritative_sentence_text=authoritative_sentence_text,
                    claimed_sentence_idx=claimed_sentence_idx,
                    notes="; ".join(notes) if notes else "Citation verified against authoritative retrieved context.",
                )
            )

        return results
