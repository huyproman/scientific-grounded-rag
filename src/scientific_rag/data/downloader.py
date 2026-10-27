"""
BEIR SciFact dataset acquisition, checksum validation, and safe extraction pipeline.
"""

import hashlib
import json
import logging
import os
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from scientific_rag.common.paths import DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)

CANONICAL_BEIR_SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
UPSTREAM_EXPECTED_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"


def compute_file_hashes(file_path: Path) -> Dict[str, str]:
    """Computes both MD5 and SHA256 hashes of a file in streaming chunks."""
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    return {
        "md5": md5_hash.hexdigest(),
        "sha256": sha256_hash.hexdigest(),
    }


def safe_extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Safely extracts a ZIP archive preventing path traversal (Zip Slip vulnerability)."""
    extract_to = extract_to.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target_path = (extract_to / member.filename).resolve()
            if not str(target_path).startswith(str(extract_to)):
                raise ValueError(f"Malicious path traversal detected in ZIP archive: {member.filename}")
        zf.extractall(extract_to)


def download_and_extract_scifact(
    dest_dir: Path = RAW_DATA_DIR,
    url: str = CANONICAL_BEIR_SCIFACT_URL,
    expected_md5: str = UPSTREAM_EXPECTED_MD5,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Acquires, validates, and unpacks the approved BEIR SciFact dataset.
    Returns acquisition manifest metadata.
    """
    dest_dir = dest_dir.resolve()
    zip_dest_dir = DATA_DIR / "downloads"
    zip_dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dest_dir / "scifact.zip"
    manifest_path = DATA_DIR / "manifest.json"

    if not force and manifest_path.exists() and (dest_dir / "corpus.jsonl").exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("validation_status") == "VALID":
            logger.info("Dataset already present and validated against manifest.")
            return manifest

    logger.info("Downloading BEIR SciFact dataset from %s ...", url)
    req = urllib.request.Request(url, headers={"User-Agent": "P01-Scientific-RAG-Client/0.1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, open(zip_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    logger.info("Verifying archive integrity against upstream MD5: %s", expected_md5)
    hashes = compute_file_hashes(zip_path)
    observed_md5 = hashes["md5"]
    observed_sha256 = hashes["sha256"]

    if observed_md5.lower() != expected_md5.lower():
        raise ValueError(
            f"MD5 Checksum Mismatch! Expected upstream: {expected_md5}, Observed: {observed_md5}"
        )

    logger.info("MD5 verified successfully! Extracting archive to %s ...", dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_extract_zip(zip_path, DATA_DIR / "raw")

    # Upstream extracts into DATA_DIR / "raw" / "scifact"
    extracted_folder = DATA_DIR / "raw" / "scifact"
    if not extracted_folder.exists():
        raise FileNotFoundError(f"Expected extracted directory not found at {extracted_folder}")

    corpus_file = extracted_folder / "corpus.jsonl"
    queries_file = extracted_folder / "queries.jsonl"
    qrels_train = extracted_folder / "qrels" / "train.tsv"
    qrels_test = extracted_folder / "qrels" / "test.tsv"

    for required_file in [corpus_file, queries_file, qrels_train, qrels_test]:
        if not required_file.exists():
            raise FileNotFoundError(f"Missing required logical dataset file: {required_file}")

    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus_count = sum(1 for line in f if line.strip())
    with open(queries_file, "r", encoding="utf-8") as f:
        queries_count = sum(1 for line in f if line.strip())
    with open(qrels_train, "r", encoding="utf-8") as f:
        qrels_train_count = sum(1 for line in f if line.strip()) - 1  # header
    with open(qrels_test, "r", encoding="utf-8") as f:
        qrels_test_count = sum(1 for line in f if line.strip()) - 1  # header

    manifest: Dict[str, Any] = {
        "dataset_name": "BEIR-SciFact",
        "upstream_source": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "download_url": url,
        "acquisition_timestamp": datetime.now(timezone.utc).isoformat(),
        "upstream_published_md5": expected_md5,
        "observed_md5": observed_md5,
        "locally_computed_sha256": observed_sha256,
        "archive_size_bytes": zip_path.stat().st_size,
        "validation_status": "VALID",
        "corpus_record_count": corpus_count,
        "query_record_count": queries_count,
        "qrels_train_record_count": qrels_train_count,
        "qrels_test_record_count": qrels_test_count,
        "expected_counts": {
            "corpus": 5183,
            "queries": 1109,
            "test_claims": 300,
            "train_claims": 809,
        },
        "licensing_provenance": {
            "upstream_annotations": "CC BY 4.0 (Allen Institute for AI)",
            "upstream_abstracts": "ODC-By 1.0 (Semantic Scholar Open Research Corpus - S2ORC)",
            "upstream_codebase": "Apache 2.0 (BEIR / SciFact)",
            "beir_catalog_notice": "CC BY-NC 2.0 (Table 1, Thakur et al., NeurIPS 2021)",
            "operational_posture": "Strictly non-commercial academic and educational portfolio research",
            "git_redistribution_policy": "NO raw corpus data committed to version control",
        },
        "file_inventory": [
            str(p.relative_to(extracted_folder))
            for p in extracted_folder.rglob("*")
            if p.is_file()
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Dataset acquisition and manifest successfully recorded at %s", manifest_path)
    return manifest
