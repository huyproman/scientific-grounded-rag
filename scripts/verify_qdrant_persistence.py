"""
Verification script for Work Package 1 (Qdrant Real Persistence Test).
Tests collection creation, vector upsert, search, container restart, persistence, and cleanup.
"""

import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def run_persistence_test(host: str = "127.0.0.1", port: int = 6335, collection_name: str = "test_persistence_col"):
    client = QdrantClient(host=host, port=port, timeout=10.0)

    print(f"Step 1 & 2: Checking Qdrant readiness on {host}:{port}...")
    collections = client.get_collections().collections
    print(f"Connected successfully. Current collections: {[c.name for c in collections]}")

    if any(c.name == collection_name for c in collections):
        client.delete_collection(collection_name)

    print(f"Step 3: Creating collection '{collection_name}' with 4-dim vectors...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )

    test_vector = [0.1, 0.2, 0.3, 0.4]
    test_payload = {"doc_id": "doc_persist_1", "chunk_id": "c_persist_1", "content": "Persistent test content"}

    print("Step 4: Upserting synthetic point ID 1...")
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=1,
                vector=test_vector,
                payload=test_payload,
            )
        ],
        wait=True,
    )

    print("Step 5: Verifying retrieval before restart...")
    res1 = client.retrieve(collection_name=collection_name, ids=[1], with_payload=True, with_vectors=True)
    assert len(res1) == 1, "Failed to retrieve point before restart"
    assert res1[0].payload["chunk_id"] == "c_persist_1"
    print(f"Verified point before restart: {res1[0].id} -> payload: {res1[0].payload}")

    return True


def verify_after_restart(host: str = "127.0.0.1", port: int = 6335, collection_name: str = "test_persistence_col"):
    client = QdrantClient(host=host, port=port, timeout=10.0)

    print(f"Step 8: Confirming collection '{collection_name}' exists after restart...")
    collections = [c.name for c in client.get_collections().collections]
    assert collection_name in collections, f"Collection '{collection_name}' not found after restart!"

    print("Step 9: Confirming vector data is still available after restart...")
    res2 = client.retrieve(collection_name=collection_name, ids=[1], with_payload=True, with_vectors=True)
    assert len(res2) == 1, "Point not found after restart!"
    assert res2[0].payload["chunk_id"] == "c_persist_1"
    print(f"SUCCESS: Point survived container restart: {res2[0].id} -> payload: {res2[0].payload}")

    print("Step 10: Cleaning up temporary test collection...")
    client.delete_collection(collection_name)
    print("Cleanup complete.")
    return True


if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "before"
    if action == "before":
        run_persistence_test()
    elif action == "after":
        verify_after_restart()
