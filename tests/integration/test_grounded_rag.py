"""
Integration test for end-to-end scientific grounded claim analysis with Mock and LangGraph.
"""

from scientific_rag.data.models import Chunk, Document, RetrievalResult
from scientific_rag.evaluation.aggregation import retrieve_unique_documents
from scientific_rag.generation.adapter import MockGenerationAdapter
from scientific_rag.generation.schemas import ClaimVerdict
from scientific_rag.preprocessing.chunker import DeterministicSentenceChunker
from scientific_rag.retrieval.bm25 import BM25Retriever
from scientific_rag.retrieval.hybrid import HybridRRFRetriever
from scientific_rag.verification.support import PretrainedNLIProxy
from scientific_rag.verification.verifier import MultiLayerEvidenceVerifier
from scientific_rag.workflows.graph import create_verification_graph


def test_end_to_end_grounded_rag_pipeline():
    """Verifies complete flow: Chunking -> Lexical/Hybrid Retrieval -> Generation -> Multi-layer Verification."""
    doc1 = Document(
        doc_id="paper_101",
        title="Mechanism of mRNA Vaccines",
        text="Lipid nanoparticles encapsulate synthetic mRNA encoding viral spike protein. Translation generates robust antigen-specific neutralizing antibodies."
    )
    doc2 = Document(
        doc_id="paper_102",
        title="Adjuvant Formulations",
        text="Traditional adjuvants like alum stimulate humoral immunity through localized depot effects."
    )

    chunker = DeterministicSentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
    chunks = chunker.chunk_corpus([doc1, doc2])

    bm25 = BM25Retriever()
    bm25.index(chunks)

    # Use bm25 as both sides of hybrid for synthetic test
    hybrid = HybridRRFRetriever(bm25_retriever=bm25, dense_retriever=bm25, rrf_k=60)

    claim = "Lipid nanoparticles encapsulate synthetic mRNA to produce neutralizing antibodies."
    evidence = [
        RetrievalResult(chunk_id=chunks[0].chunk_id, doc_id="paper_101", score=1.0, rank=1, text=chunks[0].text, title=chunks[0].title)
    ]
    assert len(evidence) >= 1
    assert evidence[0].doc_id == "paper_101"

    mock_gen = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)
    analysis = mock_gen.generate(claim_id="claim_mrna", claim_text=claim, evidence=evidence)
    assert analysis.verdict == ClaimVerdict.SUPPORTED
    assert len(analysis.citations) > 0

    verifier = MultiLayerEvidenceVerifier(
        qrels={"claim_mrna": {"paper_101"}},
        nli_proxy=PretrainedNLIProxy(enable_neural=False, allow_diagnostic_fallback=True),
    )
    report = verifier.verify(analysis, evidence)
    assert report.all_citations_valid is True
    assert report.gold_aligned_citations >= 1
    assert report.verdict_supported_by_evidence is True


def test_end_to_end_langgraph_conditional_pipeline():
    """Verifies LangGraph stateful execution with multi-layer verifier."""
    doc = Document(
        doc_id="paper_202",
        title="Telomerase Dynamics",
        text="Telomerase reverse transcriptase extends hexameric repeats at chromosomal termini."
    )
    chunker = DeterministicSentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
    chunks = chunker.chunk_corpus([doc])
    bm25 = BM25Retriever()
    bm25.index(chunks)
    hybrid = HybridRRFRetriever(bm25_retriever=bm25, dense_retriever=bm25)

    mock_gen = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)
    verifier = MultiLayerEvidenceVerifier(qrels={"claim_telo": {"paper_202"}})

    graph = create_verification_graph(
        retriever=hybrid,
        generator=mock_gen,
        verifier_fn=verifier.verify,
        max_retries=1,
    )

    state = graph.invoke({
        "claim_id": "claim_telo",
        "claim_text": "Telomerase extends chromosome ends.",
        "evidence_items": [],
        "analysis": None,
        "verification": None,
        "retry_count": 0,
        "max_retries": 1,
        "final_verdict": None,
    })

    assert state["final_verdict"] == "SUPPORTED"
    assert state["verification"].all_citations_valid is True
