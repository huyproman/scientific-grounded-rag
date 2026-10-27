"""
Prompt templates and formatting for evidence-grounded scientific claim generation.
Exposes canonical sentence identifiers [SENTENCE i] for precise evidence attribution.
Enforces strict valid JSON output adhering to schema.
"""

from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate

from scientific_rag.data.models import RetrievalResult
from scientific_rag.data.sentence_provenance import SciFactSentenceProvenance

SYSTEM_PROMPT = """You are an expert scientific claim verifier operating under strict evidence-grounding constraints.
Your task is to analyze an input scientific claim against ONLY the supplied retrieved evidence sentences.

You MUST respond with a single valid JSON object strictly matching the following schema.
Do NOT output markdown commentary, explanations outside JSON, or text before/after the JSON.

Expected JSON Schema:
{{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
  "explanation": "Scientific synthesis of evidence",
  "citations": [
    {{
      "parent_doc_id": "document_id",
      "chunk_id": "chunk_id",
      "sentence_idx": 0,
      "evidence_text": "Exact sentence text quoted from the evidence context",
      "relationship": "supports" | "contradicts"
    }}
  ]
}}

Strict Grounding Rules:
1. Rely EXCLUSIVELY on the provided evidence sentences. Do NOT assume, extrapolate, or invent external scientific knowledge.
2. If the supplied evidence does not directly support or refute the claim, you MUST output verdict: "INSUFFICIENT_EVIDENCE" and citations: [].
3. If the evidence directly supports the claim, output verdict: "SUPPORTED".
4. If the evidence directly contradicts the claim, output verdict: "CONTRADICTED".
5. In each citation, sentence_idx MUST be the integer index from [SENTENCE i], and evidence_text MUST be the exact verbatim sentence text.
6. Do NOT fabricate document IDs, chunk IDs, sentence indices, or study details.
"""

USER_PROMPT_TEMPLATE = """Claim ID: {claim_id}
Scientific Claim: {claim_text}

Retrieved Evidence Context:
{evidence_context}

Output ONLY the raw JSON object adhering to the schema:
"""


def format_evidence_context(
    evidence_items: List[RetrievalResult],
    provenance: Optional[SciFactSentenceProvenance] = None,
) -> str:
    """
    Formats a list of retrieved chunks into an inspectable numbered prompt context
    exposing canonical sentence indices.
    """
    if provenance is not None:
        return provenance.format_evidence_with_sentences(evidence_items)

    if not evidence_items:
        return "[No evidence snippets retrieved.]"

    formatted_blocks = []
    for idx, item in enumerate(evidence_items, start=1):
        block = (
            f"[{idx}] Document ID: {item.doc_id} | Chunk ID: {item.chunk_id}\n"
            f"Title: {item.title}\n"
            f"Snippet: {item.text.strip()}\n"
        )
        formatted_blocks.append(block)

    return "\n---\n".join(formatted_blocks)


def create_grounding_prompt() -> ChatPromptTemplate:
    """Builds the canonical LangChain ChatPromptTemplate for grounded claim generation."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE),
    ])
