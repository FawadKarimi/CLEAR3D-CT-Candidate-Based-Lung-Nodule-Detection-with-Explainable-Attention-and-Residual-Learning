"""Exact candidate regression against an immutable prepared baseline."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
from . import candidates
from .geometry import to_world
from .pipeline import _verify_preparation_artifacts
from .provenance import (canonical_hash, read_json, read_jsonl, sha256,
                         write_json)


_CANDIDATE_KEYS = ("zyx", "log_response", "sigma_mm")


def _first_difference(expected, actual):
    for index, (left, right) in enumerate(zip(expected, actual)):
        if canonical_hash(left) != canonical_hash(right):
            return {"index": index, "expected": left, "actual": right}
    if len(expected) != len(actual):
        return {"index": min(len(expected), len(actual)),
                "expected_count": len(expected), "actual_count": len(actual)}
    return None


def _compare_scan(prepared, scan, candidate_config, spacing):
    scan_dir = prepared/scan["scan_id"]
    expected_rows = read_jsonl(scan_dir/"proposals.jsonl")
    expected = [{key: row[key] for key in _CANDIDATE_KEYS}
                for row in expected_rows]
    record = read_json(scan_dir/"record.json")
    affine = read_json(scan_dir/"geometry.json")["affine"]
    image = np.load(scan_dir/"image.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(scan_dir/"lung_mask.npy", mmap_mode="r", allow_pickle=False)
    started = time.perf_counter()
    selected, diagnostics = candidates.generate(
        image, mask, candidate_config, spacing)
    actual = diagnostics.pop("precap_candidates")
    elapsed = time.perf_counter() - started
    actual_rows = [
        {**row, "id": f"{scan['scan_id']}:candidate:{index:06d}",
         "scan_id": scan["scan_id"],
         "center_lps": to_world(row["zyx"], affine).tolist()}
        for index, row in enumerate(actual)]
    expected_selected = expected[:int(candidate_config["cap"])]
    expected_rows_hash = canonical_hash(expected_rows)
    actual_rows_hash = canonical_hash(actual_rows)
    expected_selected_hash = canonical_hash(expected_selected)
    selected_hash = canonical_hash(selected)
    expected_diagnostics_hash = canonical_hash(record["candidate_diagnostics"])
    diagnostics_hash = canonical_hash(diagnostics)
    checks = {
        "proposal_rows_exact": actual_rows_hash == expected_rows_hash,
        "capped_candidates_exact": selected_hash == expected_selected_hash,
        "diagnostics_exact": diagnostics_hash == expected_diagnostics_hash,
    }
    result = {
        "scan_id": scan["scan_id"],
        "series_uid": scan["series_uid"],
        "candidate_rows": len(actual),
        "baseline_proposal_sha256": expected_rows_hash,
        "current_proposal_sha256": actual_rows_hash,
        "baseline_capped_sha256": expected_selected_hash,
        "current_capped_sha256": selected_hash,
        "baseline_diagnostics_sha256": expected_diagnostics_hash,
        "current_diagnostics_sha256": diagnostics_hash,
        "seconds": elapsed,
        "checks": checks,
        "first_proposal_difference": _first_difference(
            expected_rows, actual_rows),
        "acceptance_pass": all(checks.values()),
    }
    del image, mask, selected, actual, actual_rows
    gc.collect()
    return result


def validate(prepared, report, expected_preparation_sha256,
             expected_scan_ids, expected_proposals):
    prepared, report = Path(prepared).resolve(), Path(report).resolve()
    if report.exists():
        raise FileExistsError(report)
    if prepared == report.parent or prepared in report.parents:
        raise ValueError("Regression report must be outside the baseline")
    preparation_path = prepared/"preparation.json"
    preparation_sha256 = sha256(preparation_path)
    if preparation_sha256 != expected_preparation_sha256:
        raise ValueError("Baseline preparation hash is not the accepted hash")
    preparation = read_json(preparation_path)
    source_root = Path(__file__).resolve().parent
    code_sha256 = {name: sha256(source_root/name) for name in (
        "candidate_regression.py", "candidates.py", "geometry.py",
        "pipeline.py", "provenance.py")}
    config = preparation["config"]
    scans = read_jsonl(prepared/"scans.jsonl")
    pool = read_jsonl(prepared/"proposal_pool.jsonl")
    scan_ids = [scan.get("scan_id") for scan in scans]
    if not scans or scan_ids != list(expected_scan_ids):
        raise ValueError("Baseline scan identities differ from the accepted baseline")
    if len(pool) != expected_proposals:
        raise ValueError("Baseline proposal count differs from the accepted baseline")
    _verify_preparation_artifacts(
        prepared, scans, pool, preparation, config)
    print(json.dumps({"event": "baseline_verified", "scans": len(scans),
                      "proposals": len(pool)}), flush=True)
    del pool
    gc.collect()
    if any(scan.get("status") != "ok" for scan in scans):
        raise ValueError("Candidate regression requires successful baseline scans")
    candidate_config = {**config["candidates"],
                        "threshold": preparation["minimum_pool_threshold"]}
    results = []
    for index, scan in enumerate(scans, 1):
        print(json.dumps({"event": "scan_start", "index": index,
                          "total": len(scans), "scan_id": scan["scan_id"]}),
              flush=True)
        result = _compare_scan(
            prepared, scan, candidate_config, config["preprocess"]["spacing_mm"])
        results.append(result)
        print(json.dumps({"event": "scan_complete",
                          "scan_id": scan["scan_id"],
                          "seconds": result["seconds"],
                          "acceptance_pass": result["acceptance_pass"]}),
              flush=True)
    checks = {
        "baseline_integrity_verified": True,
        "all_proposal_rows_exact": all(
            row["checks"]["proposal_rows_exact"] for row in results),
        "all_capped_candidates_exact": all(
            row["checks"]["capped_candidates_exact"] for row in results),
        "all_diagnostics_exact": all(
            row["checks"]["diagnostics_exact"] for row in results),
    }
    final_pool = read_jsonl(prepared/"proposal_pool.jsonl")
    _verify_preparation_artifacts(
        prepared, scans, final_pool, preparation, config)
    final_code_sha256 = {name: sha256(source_root/name)
                         for name in code_sha256}
    if (sha256(preparation_path) != preparation_sha256 or
            final_code_sha256 != code_sha256):
        raise ValueError("Baseline or candidate implementation changed during replay")
    print(json.dumps({"event": "baseline_reverified", "scans": len(scans),
                      "proposals": len(final_pool)}), flush=True)
    del final_pool
    gc.collect()
    output = {
        "schema_version": 1,
        "kind": "candidate_optimization_exact_regression",
        "baseline": str(prepared),
        "baseline_preparation_sha256": preparation_sha256,
        "candidate_implementation_sha256": code_sha256["candidates.py"],
        "validation_code_sha256": code_sha256,
        "scans": results,
        "checks": checks,
        "acceptance_pass": all(checks.values()),
    }
    write_json(report, output)
    print(f"REPORT={report}")
    print(f"REPORT_SHA256={sha256(report)}")
    print(f"CANDIDATE_REGRESSION_PASS={output['acceptance_pass']}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True,
                        help="Accepted preparation output to verify and replay")
    parser.add_argument("--report", required=True,
                        help="New JSON report path (must not already exist)")
    parser.add_argument("--expected-preparation-sha256", required=True,
                        help="Trusted SHA-256 of the accepted preparation.json")
    parser.add_argument("--expected-scan-id", action="append", required=True,
                        help="Accepted scan ID in exact preparation order; repeat per scan")
    parser.add_argument("--expected-proposals", type=int, required=True,
                        help="Accepted full proposal-pool row count")
    args = parser.parse_args(argv)
    result = validate(
        args.prepared, args.report, args.expected_preparation_sha256,
        args.expected_scan_id, args.expected_proposals)
    return 0 if result["acceptance_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
