#!/usr/bin/env python3
"""Script to download and validate the official BEIR SciFact corpus."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scientific_rag.data.downloader import download_and_extract_scifact

if __name__ == "__main__":
    force = "--force" in sys.argv
    manifest = download_and_extract_scifact(force=force)
    print(f"Acquisition complete. Verified MD5: {manifest['observed_md5']}")
