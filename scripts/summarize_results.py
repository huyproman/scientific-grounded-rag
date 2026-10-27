"""Print a markdown metrics table derived only from artifacts/benchmarks/*.json."""
import json
import sys
from pathlib import Path

B = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/benchmarks")
KEYS = ["recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10"]

print("| Retriever | split | n | " + " | ".join(KEYS) + " |\n|---|---|---|" + "---|" * len(KEYS))
for f in ("benchmark_bm25okapi_test.json", "benchmark_bge_small_dense_test.json", "benchmark_hybridrrf_test.json"):
    d = json.load(open(B / f))
    print(f"| {d['retriever_name']} | {d['split']} | {d['num_queries']} | "
          + " | ".join(f"{d['metrics'][k]:.4f}" for k in KEYS) + " |")

g = json.load(open(B / "r2_langgraph_benchmark_300.json"))
print("\n| Generation/verification metric (run %s) | value |\n|---|---|" % g["run_id"])
for k in ("total_attempted", "total_completed", "total_citations", "valid_citations_count",
          "citation_validity_rate", "unsupported_citation_rate", "nli_entailed_citations",
          "nli_contradicted_citations", "gold_doc_aligned_count", "gold_doc_overlap_rate",
          "gold_sentence_aligned_count", "gold_sentence_evaluable_count", "gold_sentence_overlap_rate"):
    print(f"| {k} | {g[k]} |")
