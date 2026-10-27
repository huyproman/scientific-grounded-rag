# Scientific Retrieval & Evidence-Grounded RAG

Hybrid retrieval (BM25 + dense, fused with Reciprocal Rank Fusion) over BEIR SciFact abstracts, plus an LLM claim-verification workflow that must cite retrieved evidence and is checked by citation, gold-annotation and NLI-proxy verifiers. Includes a FastAPI reference service and an optional Qdrant dense backend.

**Problem.** LLMs state biomedical conclusions without showing the source passage. This project retrieves candidate passages, asks a generator for a verdict with quoted citations, verifies the citations mechanically, and downgrades any verdict whose evidence is not entailed to `INSUFFICIENT_EVIDENCE`.

**Pipeline.** 3-sentence title-prefixed chunks (5,183 abstracts, 22,443 chunks) -> BM25Okapi and `BAAI/bge-small-en-v1.5` dense retrieval -> document-level RRF (k=60) -> generation with `{doc_id}#c{chunk_index}` citations -> citation resolvability check, sentence provenance, NLI proxy (`cross-encoder/nli-deberta-v3-xsmall`) -> safe finalization.

**Status.** Reference/portfolio implementation, not a production service. Original project source is MIT-licensed (see [Licensing](#licensing)). Result details and third-party inventory are in `docs/`.

---

## 1. Results (derived from artifacts)

Numbers below are copied from `artifacts/benchmarks/*.json` by `scripts/summarize_results.py`; full provenance, split, and denominators in [`docs/RESULTS.md`](docs/RESULTS.md). The artifacts are gitignored, so they can only be re-checked against a local benchmark run.

BEIR SciFact `test` split, 300 claims, document-level metrics, candidate_chunk_k=50, target_doc_k=20:

| Retriever | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| BM25Okapi | 0.7031 | 0.7617 | 0.8124 | 0.6131 | 0.6430 |
| BGE-small dense | 0.7827 | **0.8439** | 0.8870 | 0.6864 | **0.7214** |
| Hybrid RRF (k=60) | 0.7762 | 0.8378 | 0.8852 | 0.6713 | 0.7069 |

**Hybrid did not outperform dense under the frozen evaluation.** Dense is equal or higher than hybrid on every reported metric; both beat BM25. No confidence intervals or significance tests were computed (n=300).

Live generation run (`p01_r2_final_accounted_graph_631f0dc8`, one model through a private OpenAI-compatible gateway, 294/300 claims completed, 6 gateway timeouts). These are *verification-pipeline diagnostics*, not label accuracy; final verdicts were never scored against SciFact SUPPORT/CONTRADICT labels.

| Measure (denominator = 216 citations) | Value | Meaning |
|---|---|---|
| Citation resolvability (structural) | 216/216 (100%) | cited chunk exists in the retrieved context, belongs to the cited doc, quote is a substring. Says nothing about whether the passage supports the claim. |
| NLI proxy: entailment / contradiction / neutral | 45 / 70 / 101 | small cross-encoder on claim vs cited text. A proxy, not manual verification of truth. |
| Unsupported-citation rate | 55.1% (119/216) | verdict-aware: citation not backed by NLI in the direction of the verdict |
| Gold-document overlap | 80.56% (174/216) | cited doc is a gold evidence doc (post-hoc scoring only) |
| Gold-sentence overlap | 56.94% (123/216) | cited sentence is a gold rationale sentence |

Final verdicts over 294 completed: 214 `INSUFFICIENT_EVIDENCE`, 44 `CONTRADICTED`, 36 `SUPPORTED`. The abstention rate (214/294 = 72.8% of completed claims) is a safety behaviour, but all 300 test claims have gold evidence, so abstention is not evidence of correctness. Tokens: 3,012,454 over 519 provider requests; cost unknown.

The stored artifact `abstention_rate` (0.5233 = 157/300) is a different quantity: it counts the generator's *proposed* verdicts before verification, over all 300 attempted claims. The 214/294 figure counts *final* verdicts after unsupported verdicts are downgraded to `INSUFFICIENT_EVIDENCE`, over completed claims.

Split note: BEIR SciFact `test` (300 claims) is the evaluation split; no hyperparameters (RRF k=60, NLI threshold 0.60) were tuned on it, but none were validated on a dev split either.

---

## 2. Architecture

```
                             [Client / HTTP Consumer]
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │   FastAPI Serving Layer (Port 8008)  │
                     │  - GET /health/live                 │
                     │  - GET /health/ready                │
                     │  - POST /v1/retrieve                │
                     │  - POST /v1/analyze                 │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │   Correlation & Structured Logging  │
                     │  - X-Request-ID Header Injection    │
                     │  - Single-line JSON Observability   │
                     │  - Bounded Semaphore Concurrency    │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │    Compiled LangGraph StateGraph    │
                     │  (Stateful Verification & Retry)    │
                     └──────────────────┬──────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
    ┌─────────────────────────┐                   ┌─────────────────────────┐
    │     Hybrid Retrieval    │                   │   Grounded Generator    │
    │  (Reciprocal Rank Fusion│                   │  (LangChain ChatOpenAI  │
    │        k=60)            │                   │   via OpenAI Gateway)   │
    └────────────┬────────────┘                   └────────────┬────────────┘
                 │                                             │
        ┌────────┴────────┐                                    ▼
        ▼                 ▼                       ┌─────────────────────────┐
 ┌─────────────┐   ┌─────────────┐                │   Multi-Layer Verifier  │
 │ BM25 Index  │   │ Dense Store │                │  1. Citation Structure  │
 │ (rank-bm25) │   │ (BAAI/bge)  │                │  2. Sentence Provenance │
 └─────────────┘   └──────┬──────┘                │  3. Neural NLI Support  │
                          │                       └────────────┬────────────┘
                ┌─────────┴─────────┐                          │
                ▼                   ▼                          ▼
         ┌─────────────┐     ┌─────────────┐      ┌─────────────────────────┐
         │ File-Backed │     │   Qdrant    │      │    Safe Finalization    │
         │ NumPy Index │     │ (Port 6335) │      │  (Downgrade ungrounded  │
         │  (Offline)  │     │ (Container) │      │   claims to Insufficient│
         └─────────────┘     └─────────────┘      └─────────────────────────┘
```

---

## 3. Usage

### 3.1 Offline testing (no data, no network, no credentials)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # or: pip install -r requirements.lock && pip install -e . --no-deps
HF_HUB_OFFLINE=1 pytest -q          # needs the two HF models cached; see below
scientific-rag smoke                # synthetic fixtures, mock generator, no LLM
```
Last verified 2026-10-27 on a clean `git archive` checkout (Python 3.12, fresh venv, no `data/` or `artifacts/`): `pytest -q` -> **249 passed, 16 skipped**. The 16 skips are explicit prerequisites: tests needing prebuilt indices (6), and tests needing the official SciFact corpus/claims files or the hosted-LLM opt-in (10). With the full local data and indices present: 263 passed, 2 skipped (hosted-LLM opt-in). Re-run the command to confirm on your machine; counts change as tests change.

The NLI and embedding models (`BAAI/bge-small-en-v1.5`, `cross-encoder/nli-deberta-v3-xsmall`) are downloaded from Hugging Face on first use; offline runs need them in the local cache. Tests never call a hosted LLM: gateway tests use the unroutable placeholder `http://127.0.0.1:59999` and hosted tests are skipped unless `RUN_LIVE_LLM_TESTS=1` and `LLM_API_KEY` are set.

### 3.2 Local runtime (retrieval + mock generation)
```bash
cp .env.example .env               # then set DENSE_BACKEND=file; the default (qdrant) needs the Docker container from 3.3
scientific-rag download             # BEIR SciFact zip from the BEIR host; MD5-verified; not committed
scientific-rag index --retriever all
scientific-rag query "0-dimensional biomaterials" --retriever hybrid --top-k 5
python cli.py analyze "0-dimensional biomaterials show inductive properties." --retriever hybrid --provider mock   # same as: scientific-rag analyze ...
```
Without the dataset or indices these commands exit with an explicit "index not found / run download" message rather than fabricating output.
The sentence-provenance and gold-rationale features additionally need the *official* SciFact release (`data/raw/scifact_official/data/{corpus,claims_dev}.jsonl`), which `scientific-rag download` does not fetch; obtain it from the SciFact authors and place it there. Without it, sentence attribution and gold alignment are unavailable (tests are skipped).

### 3.3 Full benchmark with data
```bash
scientific-rag index --retriever all
python scripts/run_benchmark.py     # BM25 + dense baselines -> artifacts/benchmarks/benchmark_*_test.json
scientific-rag benchmark --retriever hybrid --split test   # hybrid RRF
python scripts/summarize_results.py # markdown table from those files
```
Optional Qdrant dense backend: `docker compose up -d`, then `python scripts/index_qdrant.py --collection scifact_chunks --recreate`, set `DENSE_BACKEND=qdrant`. Docker and a running Qdrant container are required only for this backend; equality of Qdrant and file-backend rankings has not been verified. `scripts/verify_clean_room.sh` also needs Docker and network access.

### 3.4 API service
```bash
LLM_PROVIDER=mock scientific-rag serve --host 127.0.0.1 --port 8008
curl -s http://127.0.0.1:8008/health/live
curl -s http://127.0.0.1:8008/health/ready          # 503 until indices and models exist
curl -X POST http://127.0.0.1:8008/v1/retrieve -H "Content-Type: application/json" \
  -d '{"query": "0-dimensional biomaterials show inductive properties", "top_k": 3, "dense_backend": "file"}'
curl -X POST http://127.0.0.1:8008/v1/analyze -H "Content-Type: application/json" \
  -d '{"claim_text": "0-dimensional biomaterials show inductive properties.", "claim_id": "demo"}'
```
(`LLM_PROVIDER=mock` must be set in the server's environment, not on curl.)

### 3.5 Optional hosted generation
Live generation uses any OpenAI-compatible endpoint. Set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in `.env`. The default `LLM_BASE_URL` is a local placeholder (`http://127.0.0.1:8000`); no hosted endpoint is bundled, calls may be billed by your provider, and remote generation is blocked unless you opt in with `PAID_INFERENCE_AUTHORIZED=true` (default `false`, both in code and `.env.example`); no network request is made while it is unset. If the endpoint is unreachable the API returns typed 503/504 errors. The reported live run is not reproducible without its original gateway.

---

## 4. Limitations
1. CPU latency: dense embedding and NLI take ~1-3 s per claim (observed on author hardware).
2. The NLI proxy is a small general-purpose cross-encoder; it is not a scientific fact checker.
3. Citation resolvability is structural; see Results for support/NLI numbers.
4. No end-to-end label accuracy (SUPPORT/CONTRADICT) is reported; one model, one run, 6/300 timeouts.
5. Chunker uses a regex sentence splitter; only tested on SciFact. Evaluation is document-level.
6. Out of scope: distributed serving, auth, UI, GraphRAG.

## Licensing
Original project code, documentation and synthetic fixtures are licensed under the [MIT License](LICENSE), Copyright (c) 2026 Nguyen Quang Huy. This source-only grant does not cover upstream datasets, models, dependencies or excluded derived artifacts; their separate terms remain applicable.
- Dataset: BEIR SciFact is not redistributed here. Upstream terms recorded in `data/manifest.json`: annotations CC BY 4.0, abstracts ODC-By 1.0, while the BEIR catalog lists SciFact as CC BY-NC 2.0. Treat use of the data as non-commercial unless you verify otherwise.
- Models are downloaded from Hugging Face, not shipped. Third-party dependency and model inventory: [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).
