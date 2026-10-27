# Third-party inventory (not legal advice; verify upstream terms before redistribution)

Licenses of Python packages read from installed metadata in a fresh venv on 2026-10-27 (`pip install -e .[dev]`, resolved versions differ from `requirements.lock`).

| Component | Role | License (as reported) |
|---|---|---|
| rank-bm25 | BM25 | Apache-2.0 |
| sentence-transformers, transformers, torch | embeddings / NLI | Apache-2.0 (torch: BSD-style/Apache composite; see its metadata) |
| numpy | arrays | BSD-3-Clause and bundled permissive |
| pydantic, langchain, langchain-core, langchain-openai, langgraph, fastapi | runtime | MIT |
| qdrant-client | optional dense backend | Apache-2.0 |
| uvicorn, python-dotenv, httpx | runtime | BSD-3-Clause |
| tqdm | progress | MPL-2.0 AND MIT |
| pytest (dev) | tests | MIT |
| Qdrant server image (compose.yaml) | optional container | Apache-2.0 upstream; not verified offline |

Models (downloaded at runtime, not bundled): `BAAI/bge-small-en-v1.5` - MIT per local Hugging Face model card; `cross-encoder/nli-deberta-v3-xsmall` - Apache-2.0 per upstream card, NOT verified offline (card not in local cache).

Data (not bundled; `.gitignore` excludes `/data/`): BEIR SciFact. `data/manifest.json` records: annotations CC BY 4.0 (AI2), abstracts ODC-By 1.0 (S2ORC), BEIR code Apache-2.0, and the BEIR catalog's CC BY-NC 2.0 entry for SciFact. Not re-verified online. The conflicting catalog entry means non-commercial use is the safe assumption; attribution to the SciFact authors and BEIR is required in any derived publication.

Generated artifacts (`artifacts/`, indices, benchmark JSON with query/abstract ids and scores) are gitignored and contain data derived from the above; do not publish them without checking those terms. Pickle indices must not be loaded from untrusted sources.

Original project code, documentation and synthetic fixtures are licensed under MIT, Copyright (c) 2026 Nguyen Quang Huy; see `../LICENSE` and README "Licensing". This source-only grant excludes upstream datasets, models, dependencies and excluded derived artifacts. The upstream qualifications above remain unchanged.
