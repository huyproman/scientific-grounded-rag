#!/usr/bin/env bash
set -eo pipefail

echo "========================================================================"
echo "Clean-room verification"
echo "========================================================================"

CLEAN_ROOM_DIR="/tmp/clean_room_p01_r3"
REPO_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_PID=""
UVICORN_LOG="$CLEAN_ROOM_DIR/uvicorn.log"
CLEAN_COMPOSE_PROJECT="scirag-cr-${RANDOM}-$(( $(date +%s) % 100000 ))"

# Dynamically allocate safe, non-conflicting loopback ports
SERVER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
CLEAN_QDRANT_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
CLEAN_QDRANT_GRPC_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")

echo "Clean-room allocated ports:"
echo "  FastAPI Server : 127.0.0.1:$SERVER_PORT"
echo "  Qdrant HTTP    : 127.0.0.1:$CLEAN_QDRANT_PORT"
echo "  Qdrant gRPC    : 127.0.0.1:$CLEAN_QDRANT_GRPC_PORT"
echo "  Compose Project: $CLEAN_COMPOSE_PROJECT"

cleanup() {
    echo "--- Cleaning up clean-room resources ---"
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping Uvicorn server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "$CLEAN_COMPOSE_PROJECT" ] && [ -f "$CLEAN_ROOM_DIR/compose.clean_room.yaml" ]; then
        echo "Tearing down dedicated clean-room Qdrant Compose project ($CLEAN_COMPOSE_PROJECT)..."
        docker compose -p "$CLEAN_COMPOSE_PROJECT" -f "$CLEAN_ROOM_DIR/compose.clean_room.yaml" down -v --remove-orphans 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

rm -rf "$CLEAN_ROOM_DIR"

echo "[1/10] Cloning committed repository into isolated clean directory..."
git clone --no-hardlinks "$REPO_SRC" "$CLEAN_ROOM_DIR"
cd "$CLEAN_ROOM_DIR"
CURRENT_COMMIT=$(git rev-parse HEAD)
echo "Clean-room source revision: $CURRENT_COMMIT"

echo "[2/10] Creating fresh isolated Python 3.12 virtual environment (zero inherited packages)..."
python3 -m venv .venv
source .venv/bin/activate
export PATH="$CLEAN_ROOM_DIR/.venv/bin:$PATH"

echo "[3/10] Installing locked dependencies from requirements.lock..."
pip install --upgrade pip
pip install -r requirements.lock

echo "[4/10] Installing package in editable mode without resolving external dependencies..."
pip install --no-deps -e .

echo "[5/10] Running dependency integrity check (pip check)..."
pip check

echo "[6/10] Verifying CLI installation and running self-contained smoke test..."
which scientific-rag
scientific-rag --help | head -n 8
t0_smoke=$(date +%s)
scientific-rag smoke
t1_smoke=$(date +%s)
echo "Smoke test duration: $((t1_smoke - t0_smoke))s"

echo "[7/10] Provisioning dedicated clean-room Qdrant Compose instance with isolated storage..."
cat <<EOF > "$CLEAN_ROOM_DIR/compose.clean_room.yaml"
services:
  qdrant_clean_room:
    image: qdrant/qdrant:v1.16.3
    restart: "no"
    ports:
      - "127.0.0.1:${CLEAN_QDRANT_PORT}:6333"
      - "127.0.0.1:${CLEAN_QDRANT_GRPC_PORT}:6334"
    volumes:
      - clean_room_qdrant_data:/qdrant/storage

volumes:
  clean_room_qdrant_data:
EOF

docker compose -p "$CLEAN_COMPOSE_PROJECT" -f "$CLEAN_ROOM_DIR/compose.clean_room.yaml" up -d

echo "Polling http://127.0.0.1:$CLEAN_QDRANT_PORT/readyz for clean-room Qdrant startup..."
QDRANT_READY=false
for i in $(seq 1 30); do
    if curl -s -f --max-time 1 "http://127.0.0.1:$CLEAN_QDRANT_PORT/readyz" >/dev/null 2>&1; then
        echo "Clean-room Qdrant is operational (attempt $i)."
        QDRANT_READY=true
        break
    fi
    sleep 1
done

if [ "$QDRANT_READY" != "true" ]; then
    echo "ERROR: Clean-room Qdrant container failed to become ready within 30 seconds!"
    docker compose -p "$CLEAN_COMPOSE_PROJECT" -f "$CLEAN_ROOM_DIR/compose.clean_room.yaml" logs
    exit 1
fi

echo "[8/10] Indexing synthetic fixture and verifying persistence across container restart..."
# Create synthetic fixture and index into clean_room_scifact_col
python -c "
from scientific_rag.data.models import Chunk, Document
from scientific_rag.preprocessing.chunker import DeterministicSentenceChunker
from scientific_rag.retrieval.qdrant import QdrantRetriever
from scientific_rag.retrieval.bm25 import BM25Retriever
from pathlib import Path

docs = [
    Document(doc_id='synth_01', title='Cellular Senescence', text='Cellular senescence halts cell proliferation in response to DNA damage.'),
    Document(doc_id='synth_02', title='Telomere Biology', text='Telomerase reverse transcriptase maintains telomeric length during replication.'),
]
chunker = DeterministicSentenceChunker(sentences_per_chunk=1, sentence_overlap=0)
chunks = chunker.chunk_corpus(docs)

# Build isolated BM25 index
bm25 = BM25Retriever()
bm25.index(chunks)
idx_dir = Path('artifacts/indices')
idx_dir.mkdir(parents=True, exist_ok=True)
bm25.save(idx_dir / 'bm25_index.pkl')

# Build isolated Qdrant collection on dedicated port
qdrant = QdrantRetriever(
    collection_name='clean_room_scifact_col',
    host='127.0.0.1',
    port=$CLEAN_QDRANT_PORT,
    grpc_port=$CLEAN_QDRANT_GRPC_PORT,
)
qdrant.index(chunks, recreate_collection=True)
print('Synthetic fixture successfully indexed into clean_room_scifact_col')
"

echo "Executing container restart to verify volume persistence..."
docker compose -p "$CLEAN_COMPOSE_PROJECT" -f "$CLEAN_ROOM_DIR/compose.clean_room.yaml" stop
docker compose -p "$CLEAN_COMPOSE_PROJECT" -f "$CLEAN_ROOM_DIR/compose.clean_room.yaml" start

for i in $(seq 1 30); do
    if curl -s -f --max-time 1 "http://127.0.0.1:$CLEAN_QDRANT_PORT/readyz" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

python -c "
from qdrant_client import QdrantClient
from scientific_rag.retrieval.qdrant import MANIFEST_POINT_ID
client = QdrantClient(host='127.0.0.1', port=$CLEAN_QDRANT_PORT, check_compatibility=False)
count_info = client.count('clean_room_scifact_col', exact=True)
assert count_info.count >= 2, f'Expected at least 2 points after restart, got {count_info.count}'
pts = client.retrieve('clean_room_scifact_col', ids=[MANIFEST_POINT_ID], with_payload=True)
assert pts and pts[0].payload.get('is_collection_manifest'), 'Manifest point missing after restart!'
print(f'Persistence verified: {count_info.count} points and manifest point survived restart cycle.')
"

echo "[9/10] Starting managed Uvicorn HTTP server on 127.0.0.1:$SERVER_PORT..."
export QDRANT_HOST="127.0.0.1"
export QDRANT_PORT="$CLEAN_QDRANT_PORT"
export QDRANT_GRPC_PORT="$CLEAN_QDRANT_GRPC_PORT"
export QDRANT_COLLECTION="clean_room_scifact_col"
export API_PORT="$SERVER_PORT"
export DENSE_BACKEND="qdrant"
export LLM_PROVIDER="mock"

# Launch in background with redirected stdout/stderr and captured PID
uvicorn scientific_rag.api.app:app --host 127.0.0.1 --port "$SERVER_PORT" --log-level info > "$UVICORN_LOG" 2>&1 &
SERVER_PID=$!
echo "Uvicorn process spawned with PID $SERVER_PID. Output log: $UVICORN_LOG"

# Bounded readiness polling (up to 30s)
echo "Polling http://127.0.0.1:$SERVER_PORT/health/live for server startup..."
SERVER_READY=false
for i in $(seq 1 30); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Uvicorn process (PID $SERVER_PID) exited prematurely!"
        cat "$UVICORN_LOG"
        exit 1
    fi

    if curl -s --max-time 1 "http://127.0.0.1:$SERVER_PORT/health/live" | grep -q '"status":"ok"'; then
        echo "Uvicorn server is live and responding (attempt $i)."
        SERVER_READY=true
        break
    fi
    sleep 1
done

if [ "$SERVER_READY" != "true" ]; then
    echo "ERROR: Server failed to become live within 30 seconds!"
    cat "$UVICORN_LOG"
    exit 1
fi

echo "[10/10] Executing actual HTTP client calls against running ASGI service..."
python -c "
import urllib.request
import json
import sys

base = 'http://127.0.0.1:$SERVER_PORT'

# 1. /health/live
print('--- 1. Testing GET /health/live ---')
with urllib.request.urlopen(f'{base}/health/live', timeout=10) as resp:
    assert resp.status == 200, f'Expected 200, got {resp.status}'
    data = json.loads(resp.read().decode())
    print('[HTTP OK] /health/live:', data)
    assert data['status'] == 'ok'

# 2. /health/ready
print('--- 2. Testing GET /health/ready ---')
with urllib.request.urlopen(f'{base}/health/ready', timeout=10) as resp:
    assert resp.status == 200, f'Expected 200, got {resp.status}'
    data = json.loads(resp.read().decode())
    print('[HTTP OK] /health/ready:', data['status'])
    assert data['status'] == 'READY'

# 3. /v1/retrieve
print('--- 3. Testing POST /v1/retrieve ---')
req_data = json.dumps({'query': 'Telomerase reverse transcriptase maintains telomeric length', 'top_k': 2}).encode()
req = urllib.request.Request(
    f'{base}/v1/retrieve',
    data=req_data,
    headers={'Content-Type': 'application/json', 'X-Request-ID': 'clean-room-req-01'},
)
with urllib.request.urlopen(req, timeout=10) as resp:
    assert resp.status == 200, f'Expected 200, got {resp.status}'
    assert resp.headers.get('X-Request-ID') == 'clean-room-req-01', 'Correlation ID header mismatch'
    data = json.loads(resp.read().decode())
    print('[HTTP OK] /v1/retrieve:', data['total_results'], 'results, top doc:', data['results'][0]['doc_id'])
    assert len(data['results']) >= 1

# 4. /v1/analyze
print('--- 4. Testing POST /v1/analyze (Offline / Mock Generation Profile) ---')
req_data = json.dumps({'claim_text': 'Telomerase maintains telomeric length during cell replication.'}).encode()
req = urllib.request.Request(
    f'{base}/v1/analyze',
    data=req_data,
    headers={'Content-Type': 'application/json', 'X-Request-ID': 'clean-room-req-02'},
)
with urllib.request.urlopen(req, timeout=15) as resp:
    assert resp.status == 200, f'Expected 200, got {resp.status}'
    assert resp.headers.get('X-Request-ID') == 'clean-room-req-02'
    data = json.loads(resp.read().decode())
    print('[HTTP OK] /v1/analyze final verdict:', data['final_verdict'], 'outcome:', data['workflow_outcome'])
    assert data['final_verdict'] in ['SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT_EVIDENCE']
"

echo "========================================================================"
echo "GENUINE CLEAN-ROOM REPRODUCIBILITY VERIFIED (100% SUCCESS, EXIT CODE 0)"
echo "========================================================================"
