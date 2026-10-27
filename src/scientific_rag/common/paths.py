"""
Project path constants and resolution helpers.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIGS_DIR = PROJECT_ROOT / "configs"
TESTS_DIR = PROJECT_ROOT / "tests"

RAW_DATA_DIR = DATA_DIR / "raw" / "scifact"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BENCHMARK_OUTPUT_DIR = ARTIFACTS_DIR / "benchmarks"
INDEX_ARTIFACTS_DIR = ARTIFACTS_DIR / "indices"
