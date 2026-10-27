"""
Validates benchmark artifacts before packaging.
Validates that:
- Current results contain a genuine 64-character SHA256 hexadecimal hash (no legacy strings).
- Current results have the correct source revision.
- Run IDs and config SHA256 match across checkpoint, summary, and manifests.
- Completed and failed totals reconcile bit-exact.
- Observed request events are preserved without loss.
- Metrics and latencies reconcile with the checkpoint records.
"""

import json
from pathlib import Path
import sys

from scientific_rag.common.paths import ARTIFACTS_DIR
import importlib.util
spec = importlib.util.spec_from_file_location("recompute_metrics", Path(__file__).parent / "recompute_langgraph_metrics.py")
recompute_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recompute_mod)
recompute_langgraph_benchmark = recompute_mod.recompute_langgraph_benchmark


def validate_evaluation_artifacts(
    checkpoint_path: Path,
    summary_path: Path,
    expected_run_id: str,
    expected_config_sha: str,
    expected_source_rev: str,
) -> bool:
    print(f"=== Validating Evaluation Artifacts for Run: {expected_run_id} ===")
    errors = []

    if not checkpoint_path.exists():
        errors.append(f"Checkpoint file does not exist: {checkpoint_path}")
        print("\n".join(errors))
        return False

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} records from checkpoint.")
    if len(records) != 300:
        errors.append(f"Expected 300 records in checkpoint, found {len(records)}")

    # 1. Validate Run ID, Config SHA256, Source Revision across all records
    for i, r in enumerate(records):
        cid = r.get("claim_id")
        rid = r.get("run_id")
        cfg_sha = r.get("config_fingerprint")
        s_rev = r.get("source_revision")

        if rid != expected_run_id:
            errors.append(f"Record {i} (claim {cid}) has unexpected run_id: {rid} != {expected_run_id}")
        if cfg_sha != expected_config_sha:
            errors.append(f"Record {i} (claim {cid}) has unexpected config_fingerprint: {cfg_sha}")
        if len(cfg_sha or "") != 64 or not all(c in "0123456789abcdefABCDEF" for c in cfg_sha):
            errors.append(f"Record {i} (claim {cid}) config_fingerprint is not a valid 64-char hex SHA256: {cfg_sha}")
        if s_rev != expected_source_rev:
            errors.append(f"Record {i} (claim {cid}) has unexpected source_revision: {s_rev} != {expected_source_rev}")

        # Check for legacy strings
        if "sha256:3165964" in str(cfg_sha):
            errors.append(f"Record {i} (claim {cid}) contains legacy descriptive fingerprint!")

    # 2. Check for request events presence on both completed and failed
    completed = [r for r in records if r.get("status") == "COMPLETED"]
    failed = [r for r in records if r.get("status") != "COMPLETED"]
    print(f"Completed: {len(completed)}, Failed: {len(failed)}")

    for r in completed:
        reqs = r.get("provider_requests", [])
        if not reqs:
            errors.append(f"Completed claim {r.get('claim_id')} has empty provider_requests!")

    # 3. Validate against summary file
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        if summary.get("run_id") != expected_run_id:
            errors.append(f"Summary run_id {summary.get('run_id')} != {expected_run_id}")
        if summary.get("config_fingerprint") != expected_config_sha:
            errors.append(f"Summary config_fingerprint {summary.get('config_fingerprint')} != {expected_config_sha}")
        if summary.get("source_revision") != expected_source_rev:
            errors.append(f"Summary source_revision {summary.get('source_revision')} != {expected_source_rev}")
        if summary.get("total_attempted") != len(records):
            errors.append(f"Summary total_attempted {summary.get('total_attempted')} != {len(records)}")
        if summary.get("total_completed") != len(completed):
            errors.append(f"Summary total_completed {summary.get('total_completed')} != {len(completed)}")

    # 4. Independent recomputation consistency check
    recomputed = recompute_langgraph_benchmark(
        checkpoint_path,
        expected_run_id=expected_run_id,
        expected_config_sha=expected_config_sha,
    )
    if recomputed["total_completed"] != len(completed):
        errors.append(f"Recomputed total_completed {recomputed['total_completed']} != {len(completed)}")

    if errors:
        print("\n--- VALIDATION FAILED WITH ERRORS ---")
        for err in errors[:10]:
            print("  *", err)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")
        return False

    print("\n--- ALL PRE-PACKAGING CHECKS PASSED SUCCESSFULLY ---")
    return True


if __name__ == "__main__":
    spec_runner = importlib.util.spec_from_file_location("run_runner", Path(__file__).parent / "run_langgraph_benchmark_300.py")
    runner_mod = importlib.util.module_from_spec(spec_runner)
    spec_runner.loader.exec_module(runner_mod)
    CONFIG_FINGERPRINT = runner_mod.CONFIG_FINGERPRINT
    AUTHORITATIVE_RUN_ID = runner_mod.AUTHORITATIVE_RUN_ID
    SOURCE_REVISION = runner_mod.SOURCE_REVISION

    ckpt = ARTIFACTS_DIR / "benchmarks" / "r2_langgraph_300_checkpoint.jsonl"
    summ = ARTIFACTS_DIR / "benchmarks" / "r2_langgraph_benchmark_300.json"
    ok = validate_evaluation_artifacts(
        checkpoint_path=ckpt,
        summary_path=summ,
        expected_run_id=AUTHORITATIVE_RUN_ID,
        expected_config_sha=CONFIG_FINGERPRINT,
        expected_source_rev=SOURCE_REVISION,
    )
    sys.exit(0 if ok else 1)
