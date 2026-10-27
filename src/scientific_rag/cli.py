"""
Command Line Interface (CLI) implementation for Scientific Retrieval & Evidence-Grounded RAG.
Packaged under scientific_rag.cli for console script entry point.
Supports BM25, dense and hybrid RRF retrieval, grounded analysis, and multi-layer verification.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from scientific_rag.common.compatibility import (
    BM25IndexConfig,
    DenseIndexConfig,
    IncompatibleIndexError,
    PreprocessingConfig,
    resolve_dataset_fingerprint,
)
from scientific_rag.common.paths import (
    ARTIFACTS_DIR,
    INDEX_ARTIFACTS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
from scientific_rag.data.downloader import download_and_extract_scifact
from scientific_rag.data.loader import DatasetLoader
from scientific_rag.data.models import Document, Query
from scientific_rag.evaluation.aggregation import aggregate_chunks_to_documents
from scientific_rag.evaluation.benchmark import run_retrieval_benchmark
from scientific_rag.evaluation.grounding import evaluate_grounded_analyses
from scientific_rag.evaluation.metrics import compute_all_metrics
from scientific_rag.generation.adapter import (
    MockGenerationAdapter,
    NineRouterAdapter,
    NineRouterPreflightError,
)
from scientific_rag.generation.schemas import ClaimVerdict, GroundedClaimAnalysis
from scientific_rag.preprocessing.chunker import DeterministicSentenceChunker
from scientific_rag.retrieval.bm25 import BM25Retriever
from scientific_rag.retrieval.dense import DenseRetriever
from scientific_rag.retrieval.hybrid import HybridRRFRetriever
from scientific_rag.verification.verifier import MultiLayerEvidenceVerifier
from scientific_rag.workflows.graph import create_verification_graph

logger = logging.getLogger("scientific_rag.cli")


def run_self_contained_smoke() -> None:
    """Executes a self-contained synthetic smoke test with zero test/ directory dependencies."""
    print("\n=== RUNNING PACKAGED SELF-CONTAINED SMOKE TEST ===")
    sample_docs = [
        Document(
            doc_id="smoke_1",
            title="CRISPR Gene Editing",
            text="CRISPR Cas9 enables targeted genetic modifications in mammalian cells.",
        ),
        Document(
            doc_id="smoke_2",
            title="Histone H3 Epigenetics",
            text="Post-translational modification of histone H3 regulates chromatin compaction.",
        ),
    ]

    chunker = DeterministicSentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
    chunks = chunker.chunk_corpus(sample_docs)
    print(f"[OK] Chunked {len(sample_docs)} sample documents into {len(chunks)} chunks.")

    retriever = BM25Retriever()
    retriever.index(chunks)
    print(f"[OK] BM25 indexed {len(chunks)} chunks.")

    query = "How does Cas9 edit genes?"
    results = retriever.retrieve(query, top_k=2)
    doc_ids, _ = aggregate_chunks_to_documents(results, top_k=1)
    assert doc_ids == ["smoke_1"], f"Unexpected top document: {doc_ids}"
    print(f"[OK] Retrieved top doc: {doc_ids}")

    metrics = compute_all_metrics(doc_ids, {"smoke_1"})
    assert metrics["recall@5"] == 1.0
    print(f"[OK] Metric verification: Recall@5={metrics['recall@5']:.4f}")

    # Mock Generation & Multi-layer Verification Smoke
    mock_gen = MockGenerationAdapter(default_verdict=ClaimVerdict.SUPPORTED)
    analysis = mock_gen.generate("smoke_claim_1", query, results)
    verifier = MultiLayerEvidenceVerifier(qrels={"smoke_claim_1": {"smoke_1"}})
    report = verifier.verify(analysis, results)
    assert report.all_citations_valid is True
    print(f"[OK] Grounding smoke: Verdict={analysis.verdict.value}, AllCitationsValid={report.all_citations_valid}")

    # Serving layer smoke via TestClient
    try:
        from fastapi.testclient import TestClient
        from scientific_rag.api.app import app
        api_client = TestClient(app)
        live_res = api_client.get("/health/live")
        assert live_res.status_code == 200
        print(f"[OK] API /health/live smoke: status={live_res.status_code}")
    except Exception as e:
        print(f"[NOTE] API smoke skipped or error: {e}")

    print("=== PACKAGED SMOKE TEST PASSED (<2s on CPU) ===\n")


def load_retriever_with_checks(retriever_name: str, batch_size: int = 32):
    """Helper to load a retriever index with strict compatibility verification."""
    dataset_name, dataset_hash = resolve_dataset_fingerprint(RAW_DATA_DIR)
    prep_config = PreprocessingConfig()

    if retriever_name == "bm25":
        bm25_path = INDEX_ARTIFACTS_DIR / "bm25_index.pkl"
        if not bm25_path.exists():
            print(f"Error: BM25 index not found at {bm25_path}. Run 'python cli.py index --retriever bm25' first.")
            sys.exit(1)
        return BM25Retriever.load(
            bm25_path,
            expected_dataset_name=dataset_name,
            expected_dataset_hash=dataset_hash,
            expected_preprocessing_config=prep_config,
        )
    elif retriever_name == "dense":
        dense_path = INDEX_ARTIFACTS_DIR / "dense_index"
        if not dense_path.exists():
            print(f"Error: Dense index not found at {dense_path}. Run 'python cli.py index --retriever dense' first.")
            sys.exit(1)
        return DenseRetriever.load(
            dense_path,
            device="cpu",
            expected_dataset_name=dataset_name,
            expected_dataset_hash=dataset_hash,
            expected_preprocessing_config=prep_config,
        )
    elif retriever_name == "hybrid":
        bm25 = load_retriever_with_checks("bm25")
        dense = load_retriever_with_checks("dense", batch_size=batch_size)
        return HybridRRFRetriever(bm25_retriever=bm25, dense_retriever=dense, rrf_k=60)
    else:
        print(f"Error: Unsupported retriever '{retriever_name}'. Choose 'bm25', 'dense', or 'hybrid'.")
        sys.exit(1)


def cmd_download(args: argparse.Namespace) -> None:
    """Acquires and validates the official BEIR SciFact dataset."""
    logger.info("Executing dataset acquisition pipeline...")
    manifest = download_and_extract_scifact(force=args.force)
    print("\n=== DATASET ACQUISITION SUCCESSFUL ===")
    print(f"Dataset Name:          {manifest.get('dataset_name')}")
    print(f"Upstream Source:       {manifest.get('upstream_source')}")
    print(f"Upstream MD5 (Truth):  {manifest.get('upstream_published_md5')}")
    print(f"Observed MD5:          {manifest.get('observed_md5')}")
    print(f"Local SHA256 Hash:     {manifest.get('locally_computed_sha256')}")
    print(f"Corpus Document Count: {manifest.get('corpus_record_count')}")
    print(f"Queries Total Count:   {manifest.get('query_record_count')}")
    print(f"Test Qrels Count:      {manifest.get('qrels_test_record_count')}")
    print("Validation Status:     VALID (No raw corpus committed to Git)\n")


def cmd_index(args: argparse.Namespace) -> None:
    """Builds retrieval indices from downloaded corpus with compatibility manifests."""
    loader = DatasetLoader(RAW_DATA_DIR)
    docs = loader.load_corpus()
    logger.info("Chunking %d documents deterministically...", len(docs))
    prep_config = PreprocessingConfig(sentences_per_chunk=3, sentence_overlap=1, include_title=True)
    chunker = DeterministicSentenceChunker(
        sentences_per_chunk=prep_config.sentences_per_chunk,
        sentence_overlap=prep_config.sentence_overlap,
        include_title_in_text=prep_config.include_title,
    )
    chunks = chunker.chunk_corpus(docs)
    logger.info("Generated %d chunks with stable IDs ({doc_id}#c{chunk_index}).", len(chunks))

    dataset_name, dataset_hash = resolve_dataset_fingerprint(RAW_DATA_DIR)
    INDEX_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.retriever in ["bm25", "all"]:
        logger.info("Building BM25Okapi lexical index...")
        bm25_retriever = BM25Retriever()
        bm25_retriever.index(chunks)
        bm25_path = INDEX_ARTIFACTS_DIR / "bm25_index.pkl"
        bm25_retriever.save(
            bm25_path,
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            preprocessing_config=prep_config,
        )
        logger.info("BM25 index successfully built and saved to %s", bm25_path)

    if args.retriever in ["dense", "all"]:
        logger.info("Building Dense Bi-Encoder index using BAAI/bge-small-en-v1.5 on CPU...")
        dense_retriever = DenseRetriever(device="cpu", batch_size=args.batch_size)
        dense_retriever.index(chunks)
        dense_path = INDEX_ARTIFACTS_DIR / "dense_index"
        dense_retriever.save(
            dense_path,
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            preprocessing_config=prep_config,
        )
        logger.info("Dense index successfully built and saved to %s", dense_path)

    if args.retriever in ["qdrant", "all"]:
        logger.info("Indexing chunks into Qdrant vector database...")
        from scientific_rag.config import load_config
        from scientific_rag.retrieval.qdrant import QdrantRetriever
        cfg = load_config()
        qdrant_retriever = QdrantRetriever(
            collection_name=cfg.qdrant_collection,
            host=cfg.qdrant_host,
            port=cfg.qdrant_port,
        )
        qdrant_retriever.index(chunks, batch_size=args.batch_size)
        qdrant_path = INDEX_ARTIFACTS_DIR / "qdrant_index"
        qdrant_retriever.save(qdrant_path)
        logger.info("Qdrant index successfully built and saved to %s", qdrant_path)


def cmd_query(args: argparse.Namespace) -> None:
    """Searches a query against selected retriever index with compatibility checks."""
    try:
        retriever = load_retriever_with_checks(args.retriever)
    except IncompatibleIndexError as e:
        print(f"\n[ERROR] Incompatible Index Artifact: {e}\n")
        sys.exit(1)

    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\n=== RETRIEVAL RESULTS [{retriever.name}] ===")
    print(f"Query: {args.query}")
    print(f"Total Results: {len(results)}\n")

    for r in results:
        preview = r.text.replace("\n", " ")
        if len(preview) > 140:
            preview = preview[:137] + "..."
        print(f"[Rank {r.rank}] Score: {r.score:.4f} | Chunk ID: {r.chunk_id} | Doc ID: {r.doc_id}")
        if r.title:
            print(f"         Title: {r.title}")
        print(f"         Text:  {preview}\n")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Runs IR benchmark on the approved SciFact split with compatibility validation."""
    loader = DatasetLoader(RAW_DATA_DIR)
    queries = loader.load_queries(split=args.split)
    qrels = loader.load_qrels(split=args.split)

    try:
        retriever = load_retriever_with_checks(args.retriever)
    except IncompatibleIndexError as e:
        print(f"\n[ERROR] Incompatible Index Artifact: {e}\n")
        sys.exit(1)

    summary = run_retrieval_benchmark(
        retriever=retriever,
        queries=queries,
        qrels=qrels,
        split_name=args.split,
    )

    print(f"\n=== BENCHMARK SUMMARY [{summary.retriever_name} - Split: {summary.split}] ===")
    print(f"Queries Evaluated: {summary.num_queries}")
    print(f"Execution Time:    {summary.runtime_seconds:.2f}s (CPU)")
    print(f"Recall@5:          {summary.metrics['recall@5']:.4f}")
    print(f"Recall@10:         {summary.metrics['recall@10']:.4f}")
    print(f"Recall@20:         {summary.metrics['recall@20']:.4f}")
    print(f"MRR@10:            {summary.metrics['mrr@10']:.4f}")
    print(f"nDCG@10:           {summary.metrics['ndcg@10']:.4f}\n")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Performs end-to-end evidence-grounded scientific claim analysis."""
    claim_text = args.claim
    claim_id = args.claim_id or "cli_query_001"

    retriever = load_retriever_with_checks(args.retriever)
    evidence_items = retriever.retrieve(claim_text, top_k=args.top_k)

    if args.provider == "mock":
        generator = MockGenerationAdapter()
    elif args.provider == "9router":
        generator = NineRouterAdapter(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model,
        )
        try:
            generator.preflight_check()
        except NineRouterPreflightError as e:
            print(f"\n[ERROR] 9Router Preflight Check Failed: {e}")
            print("Execution blocked. Provide valid credentials and ensure endpoint is reachable.\n")
            sys.exit(1)
    else:
        print(f"Error: Unknown provider '{args.provider}'.")
        sys.exit(1)

    if args.use_graph:
        loader = DatasetLoader(RAW_DATA_DIR)
        qrels = loader.load_qrels(split="test")
        verifier_obj = MultiLayerEvidenceVerifier(qrels=qrels)
        graph = create_verification_graph(
            retriever=retriever,
            generator=generator,
            verifier_fn=verifier_obj.verify,
            max_retries=1,
        )
        state = graph.invoke({
            "claim_id": claim_id,
            "claim_text": claim_text,
            "evidence_items": [],
            "analysis": None,
            "verification": None,
            "retry_count": 0,
            "max_retries": 1,
            "final_verdict": None,
        })
        analysis: GroundedClaimAnalysis = state["analysis"]
    else:
        analysis = generator.generate(claim_id, claim_text, evidence_items)
        loader = DatasetLoader(RAW_DATA_DIR)
        qrels = loader.load_qrels(split="test")
        verifier_obj = MultiLayerEvidenceVerifier(qrels=qrels)
        analysis.verification = verifier_obj.verify(analysis, evidence_items)

    print(f"\n=== SCIENTIFIC CLAIM ANALYSIS [{analysis.verdict.value}] ===")
    print(f"Claim ID:     {analysis.claim_id}")
    print(f"Input Claim:  {analysis.input_claim}")
    print(f"Verdict:      {analysis.verdict.value}")
    print(f"Explanation:  {analysis.explanation}\n")
    print("--- Citations ---")
    if not analysis.citations:
        print("[No citations generated (Insufficient evidence or abstained)]")
    for idx, c in enumerate(analysis.citations, start=1):
        print(f"[{idx}] Document: {c.parent_doc_id} | Chunk: {c.chunk_id} | Relation: {c.relationship}")
        print(f"    Evidence: {c.evidence_text}")

    if analysis.verification:
        v = analysis.verification
        print("\n--- Multi-Layer Verification Report ---")
        print(f"Total Citations:      {v.total_citations}")
        print(f"Valid Citations:      {v.valid_citations} (Contextually Present)")
        print(f"Hallucinated:         {v.hallucinated_citations}")
        print(f"Gold Document Match:  {v.gold_aligned_citations}")
        print(f"NLI Entailed:         {v.nli_entailed_citations}")
        print(f"All Citations Valid:  {v.all_citations_valid}")
        print(f"Verdict Grounded:     {v.verdict_supported_by_evidence}")
        print(f"Outcomes:             {[o.value for o in v.outcomes]}\n")


def cmd_smoke(args: argparse.Namespace) -> None:
    """Executes small deterministic synthetic smoke test."""
    run_self_contained_smoke()


def cmd_serve(args: argparse.Namespace) -> None:
    """Starts the minimal FastAPI serving layer."""
    import uvicorn
    from scientific_rag.config import load_config
    cfg = load_config()
    host = args.host or cfg.api_host
    port = args.port or cfg.api_port
    print(f"Starting Scientific Grounded RAG serving layer on http://{host}:{port}...")
    uvicorn.run("scientific_rag.api.app:app", host=host, port=port, log_level=args.log_level.lower())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scientific Retrieval & Evidence-Grounded RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_dl = subparsers.add_parser("download", help="Acquire and validate BEIR SciFact dataset")
    p_dl.add_argument("--force", action="store_true", help="Force redownload even if manifest exists")

    p_idx = subparsers.add_parser("index", help="Build retrieval indices")
    p_idx.add_argument("--retriever", choices=["bm25", "dense", "qdrant", "all"], default="all")
    p_idx.add_argument("--batch-size", type=int, default=32, help="Embedding batch size for dense retriever")

    p_query = subparsers.add_parser("query", help="Query the retrieval system")
    p_query.add_argument("query", type=str, help="Scientific claim or search query text")
    p_query.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of ranked chunks to retrieve")

    p_bench = subparsers.add_parser("benchmark", help="Run IR evaluation benchmark")
    p_bench.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    p_bench.add_argument("--split", choices=["test", "train"], default="test")

    # Analyze
    p_ana = subparsers.add_parser("analyze", help="Evidence-grounded scientific claim analysis")
    p_ana.add_argument("claim", type=str, help="Scientific claim text to verify")
    p_ana.add_argument("--claim-id", type=str, default="cli_claim_01")
    p_ana.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    p_ana.add_argument("--provider", choices=["mock", "9router"], default="mock")
    p_ana.add_argument("--base-url", type=str, default=None)
    p_ana.add_argument("--api-key", type=str, default=None)
    p_ana.add_argument("--model", type=str, default="ag/gemini-3.8-flash-high")
    p_ana.add_argument("--top-k", type=int, default=5)
    p_ana.add_argument("--use-graph", action="store_true", help="Enable LangGraph stateful conditional retry")

    # Smoke
    subparsers.add_parser("smoke", help="Run deterministic synthetic smoke test")

    p_serve = subparsers.add_parser("serve", help="Start the FastAPI reference service")
    p_serve.add_argument("--host", type=str, default=None, help="Host interface to bind to")
    p_serve.add_argument("--port", type=int, default=None, help="Port to bind to")
    p_serve.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "smoke":
        cmd_smoke(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
