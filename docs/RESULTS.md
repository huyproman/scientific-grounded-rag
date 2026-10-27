# Results provenance

All numbers are read from gitignored local artifacts by `python scripts/summarize_results.py` (default dir `artifacts/benchmarks`). Nothing here is hand-edited. Tables below were generated 2026-10-27.

## Retrieval (document level)
- Dataset: BEIR SciFact; corpus 5,183 abstracts (22,443 chunks); split `test`, 300 queries (`num_queries` in each artifact; `data/manifest.json` records 339 test qrels rows over those queries).
- Config in artifacts: `candidate_chunk_k=50`, `target_doc_k=20`; chunker 3 sentences / 1 overlap / title prefixed (`configs/default.yaml`). Hybrid: RRF k=60 over doc ids.
- Artifacts: `artifacts/benchmarks/benchmark_{bm25okapi,bge_small_dense,hybridrrf}_test.json` (timestamps 2026-10-19/20, CPU).

| Retriever | split | n | recall@5 | recall@10 | recall@20 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|---|
| BM25Okapi | test | 300 | 0.7031 | 0.7617 | 0.8124 | 0.6131 | 0.6430 |
| BGE-Small-Dense | test | 300 | 0.7827 | 0.8439 | 0.8870 | 0.6864 | 0.7214 |
| HybridRRF | test | 300 | 0.7762 | 0.8378 | 0.8852 | 0.6713 | 0.7069 |

Dense >= Hybrid on all five metrics. Hybrid did not outperform dense. No explanation is offered because none was tested. No fusion weights or k were retuned in this remediation. The earlier README figures (Hybrid R@10 0.8240 / nDCG 0.8120, Dense 0.7830/0.7725, BM25 0.7710/0.7684) match no artifact and were removed.

## Generation / verification run
- Artifact: `artifacts/benchmarks/r2_langgraph_benchmark_300.json` (and `r2_langgraph_recomputed_metrics.json`, `r2_langgraph_300_checkpoint.jsonl`). Run `p01_r2_final_accounted_graph_631f0dc8`, source revision `b28f6412`, model `ag/gemini-3.8-flash-high` via a private gateway, 2026-10-21.
- 300 attempted, 294 completed, 6 `GATEWAY_TIMEOUT`.

| Metric | Value | Denominator |
|---|---|---|
| citation resolvability (`citation_validity_rate`) | 1.0 (216) | 216 citations |
| unsupported_citation_rate | 0.5509 (119) | 216 citations |
| NLI ENTAILMENT / CONTRADICTION / NEUTRAL | 45 / 70 / 101 | 216 citations (counted from checkpoint `citation_details`) |
| gold_doc_overlap_rate | 0.8056 (174) | 216 citations |
| gold_sentence_overlap_rate | 0.5694 (123) | 216 evaluable |
| final verdicts INSUFFICIENT / CONTRADICTED / SUPPORTED | 214 / 44 / 36 | 294 completed |
| tokens / requests | 3,012,454 / 519 | cost status UNKNOWN |

## What each citation metric means
| Metric | Checks | Does not check |
|---|---|---|
| resolvability (code key `citation_validity_rate`) | non-empty ids; chunk is in retrieved context; chunk belongs to cited doc; quote is a normalized substring of chunk (`verification/citations.py`) | whether the text supports the claim |
| gold overlap | cited doc / sentence appears in SciFact gold annotations (post-hoc, never given to generator) | semantic support |
| NLI proxy | `cross-encoder/nli-deberta-v3-xsmall` label for claim vs cited text, threshold 0.60 | factual truth; it is not manual verification |
| unsupported rate | verdict-aware combination of the above (`evaluation/grounding.py`) | factuality |

The JSON key `citation_validity_rate` is kept for compatibility with stored artifacts and tests; user-facing labels say "resolvability". Final verdicts have never been scored against SciFact gold labels.

## Abstention definitions
- README / table above: final `INSUFFICIENT_EVIDENCE` after safe finalization, 214/294 = 72.8% of completed claims.
- Artifact key `abstention_rate` = 0.5233 = 157/300: proposed (pre-verification) `INSUFFICIENT_EVIDENCE` verdicts over all attempted claims, including the 6 provider failures. The two differ because verification downgrades ungrounded verdicts and because the denominators differ.

## Split note
BEIR `test` = the 300 SciFact claims whose annotations come from the original dev set. Train qrels were not used for tuning anything here.
