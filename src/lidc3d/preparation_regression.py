"""Compare scientific payloads from two independently prepared smoke runs."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from .pipeline import (_preparation_implementation_hash,
                       _verify_preparation_artifacts)
from .provenance import canonical_hash, read_json, read_jsonl, sha256, write_json


_ARRAY_FILES = ("image.npy", "valid.npy", "lung_mask.npy", "base_mask.npy")
_SCIENTIFIC_FILES = (*_ARRAY_FILES, "geometry.json", "proposals.jsonl")
_SCAN_FILES = {"checkpoint.json", "record.json", *_SCIENTIFIC_FILES}
_CHECKPOINT_KEYS = {
    "schema_version", "kind", "scan_id", "scan_hash", "preparation_key",
    "files", "proposal_count", "status", "preparation_seconds"}
_PREPARATION_KEYS = {
    "schema_version", "preparation_key", "cohort_hash", "config",
    "minimum_pool_threshold", "status", "files", "artifacts"}
_PATH_FIELDS = tuple(f"{name}_path" for name in
                     ("image", "valid", "lung_mask", "base_mask"))


def _normalize_record(record, root=None, scan_id=None):
    result = copy.deepcopy(record)
    for field in _PATH_FIELDS:
        value = result.pop(field, None)
        if root is not None:
            name = field.removesuffix("_path") + ".npy"
            expected = Path(root).resolve()/scan_id/name
            if not isinstance(value, str) or value != str(expected):
                raise ValueError(f"Record path is not bound to its preparation: {field}")
            result[field] = f"{scan_id}/{name}"
        else:
            result[field] = Path(value).name if value is not None else None
    return result


def _checkpoint_semantics(checkpoint):
    if set(checkpoint) != _CHECKPOINT_KEYS:
        raise ValueError("Preparation checkpoint schema changed")
    expected_files = ({"record.json", "proposals.jsonl", "geometry.json",
                       *_ARRAY_FILES}
                      if checkpoint.get("status") == "ok" else
                      {"record.json", "proposals.jsonl"})
    files = checkpoint.get("files")
    if (not isinstance(files, dict) or set(files) != expected_files or
            any(not isinstance(value, dict) or set(value) != {"bytes", "sha256"}
                for value in files.values())):
        raise ValueError("Preparation checkpoint file schema changed")
    result = copy.deepcopy(checkpoint)
    result.pop("preparation_key")
    result.pop("preparation_seconds")
    result["files"]["record.json"] = "normalized_output_path_evidence"
    return result


def _plan_semantics(plan):
    result = copy.deepcopy(plan)
    result.pop("implementation_hash", None)
    result.pop("preparation_key", None)
    return result


def _preparation_semantics(preparation):
    if set(preparation) != _PREPARATION_KEYS:
        raise ValueError("Preparation marker schema changed")
    files = preparation.get("files")
    artifacts = preparation.get("artifacts")
    if (not isinstance(files, dict) or
            set(files) != {"scans.jsonl", "proposal_pool.jsonl"} or
            not isinstance(artifacts, dict) or
            any(not isinstance(value, dict) or set(value) != {"bytes", "sha256"}
                for value in artifacts.values())):
        raise ValueError("Preparation marker artifact schema changed")
    result = copy.deepcopy(preparation)
    result.pop("preparation_key")
    result["files"]["scans.jsonl"] = "normalized_output_path_evidence"
    for name in result["artifacts"]:
        if name.endswith(("/record.json", "/checkpoint.json")):
            result["artifacts"][name] = "normalized_output_path_evidence"
    return result


def _file_evidence(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def _valid_plan_key(plan):
    payload = copy.deepcopy(plan)
    declared = payload.pop("preparation_key", None)
    return declared is not None and declared == canonical_hash(payload)


def _root_inventory(root, scan_ids):
    required = {"preparation_plan.json", "preparation.json", "scans.jsonl",
                "proposal_pool.jsonl", *scan_ids}
    entries = {path.name: path for path in root.iterdir()}
    allowed = required | {".prepare.lock"}
    safe = set(entries) == required or set(entries) == allowed
    safe &= all(not path.is_symlink() for path in entries.values())
    safe &= all(entries[name].is_dir() for name in scan_ids if name in entries)
    safe &= all(entries[name].is_file() for name in required - set(scan_ids)
                if name in entries)
    scan_entries = {}
    for scan_id in scan_ids:
        scan_dir = entries.get(scan_id)
        children = {} if scan_dir is None or not scan_dir.is_dir() else {
            path.name: path for path in scan_dir.iterdir()}
        scan_safe = (set(children) == _SCAN_FILES and
                     all(path.is_file() and not path.is_symlink()
                         for path in children.values()))
        safe &= scan_safe
        scan_entries[scan_id] = {
            "entries": sorted(children), "safe_and_exact": bool(scan_safe)}
    lock = entries.get(".prepare.lock")
    safe &= lock is None or (lock.is_file() and lock.stat().st_size == 0)
    return {"entries": sorted(entries), "scan_entries": scan_entries,
            "safe_and_exact": bool(safe)}


def _independent_files(left, right):
    left_stat, right_stat = left.stat(), right.stat()
    return (left_stat.st_dev != right_stat.st_dev or
            left_stat.st_ino != right_stat.st_ino)


def compare(baseline, candidate, report, expected_baseline_sha256,
            expected_scan_ids, expected_proposals):
    baseline, candidate = Path(baseline).resolve(), Path(candidate).resolve()
    report = Path(report).resolve()
    if (baseline == candidate or baseline in candidate.parents or
            candidate in baseline.parents):
        raise ValueError("Preparation roots must be distinct and non-nested")
    if report.exists():
        raise FileExistsError(report)
    if any(root == report.parent or root in report.parents
           for root in (baseline, candidate)):
        raise ValueError("Regression report must be outside preparation roots")
    baseline_preparation_path = baseline/"preparation.json"
    candidate_preparation_path = candidate/"preparation.json"
    baseline_sha = sha256(baseline_preparation_path)
    if baseline_sha != expected_baseline_sha256:
        raise ValueError("Baseline preparation hash is not the accepted hash")

    source_root = Path(__file__).resolve().parent
    code_sha256 = {name: sha256(source_root/name) for name in (
        "preparation_regression.py", "pipeline.py", "ingest.py",
        "geometry.py", "candidates.py", "provenance.py")}
    baseline_preparation = read_json(baseline_preparation_path)
    candidate_preparation = read_json(candidate_preparation_path)
    candidate_sha = sha256(candidate_preparation_path)
    baseline_scans = read_jsonl(baseline/"scans.jsonl")
    candidate_scans = read_jsonl(candidate/"scans.jsonl")
    baseline_pool = read_jsonl(baseline/"proposal_pool.jsonl")
    candidate_pool = read_jsonl(candidate/"proposal_pool.jsonl")
    baseline_config = baseline_preparation["config"]
    candidate_config = candidate_preparation["config"]
    _verify_preparation_artifacts(
        baseline, baseline_scans, baseline_pool,
        baseline_preparation, baseline_config)
    _verify_preparation_artifacts(
        candidate, candidate_scans, candidate_pool,
        candidate_preparation, candidate_config)

    baseline_ids = [row.get("scan_id") for row in baseline_scans]
    candidate_ids = [row.get("scan_id") for row in candidate_scans]
    expected_scan_ids = list(expected_scan_ids)
    if not expected_scan_ids or expected_proposals <= 0:
        raise ValueError("Expected scans and proposal count must be positive")
    baseline_plan_path = baseline/"preparation_plan.json"
    candidate_plan_path = candidate/"preparation_plan.json"
    baseline_plan_sha = sha256(baseline_plan_path)
    candidate_plan_sha = sha256(candidate_plan_path)
    baseline_plan = read_json(baseline_plan_path)
    candidate_plan = read_json(candidate_plan_path)
    baseline_inventory = _root_inventory(baseline, expected_scan_ids)
    candidate_inventory = _root_inventory(candidate, expected_scan_ids)
    file_comparisons = {}
    checkpoint_comparisons = {}
    record_comparisons = {}
    timing = {}
    independent_files = []
    for scan_id in expected_scan_ids:
        file_comparisons[scan_id] = {}
        for name in _SCIENTIFIC_FILES:
            left = _file_evidence(baseline/scan_id/name)
            right = _file_evidence(candidate/scan_id/name)
            file_comparisons[scan_id][name] = {
                "baseline": left, "candidate": right,
                "exact": left == right}
            independent_files.append(_independent_files(
                baseline/scan_id/name, candidate/scan_id/name))
        left_record = _normalize_record(
            read_json(baseline/scan_id/"record.json"), baseline, scan_id)
        right_record = _normalize_record(
            read_json(candidate/scan_id/"record.json"), candidate, scan_id)
        record_comparisons[scan_id] = {
            "baseline_sha256": canonical_hash(left_record),
            "candidate_sha256": canonical_hash(right_record),
            "exact": canonical_hash(left_record) == canonical_hash(right_record)}
        baseline_checkpoint = read_json(baseline/scan_id/"checkpoint.json")
        candidate_checkpoint = read_json(candidate/scan_id/"checkpoint.json")
        left_checkpoint = _checkpoint_semantics(baseline_checkpoint)
        right_checkpoint = _checkpoint_semantics(candidate_checkpoint)
        timing[scan_id] = {
            "baseline_seconds": baseline_checkpoint.get("preparation_seconds"),
            "candidate_seconds": candidate_checkpoint.get("preparation_seconds")}
        checkpoint_comparisons[scan_id] = {
            "baseline_sha256": canonical_hash(left_checkpoint),
            "candidate_sha256": canonical_hash(right_checkpoint),
            "exact": canonical_hash(left_checkpoint) == canonical_hash(right_checkpoint)}

    normalized_baseline_scans = [
        _normalize_record(row, baseline, row["scan_id"])
        for row in baseline_scans]
    normalized_candidate_scans = [
        _normalize_record(row, candidate, row["scan_id"])
        for row in candidate_scans]
    aggregate_pool_evidence = {
        "baseline": _file_evidence(baseline/"proposal_pool.jsonl"),
        "candidate": _file_evidence(candidate/"proposal_pool.jsonl")}
    independent_files.append(_independent_files(
        baseline/"proposal_pool.jsonl", candidate/"proposal_pool.jsonl"))
    timings_valid = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0
        for row in timing.values() for value in row.values())
    checks = {
        "baseline_authenticated": True,
        "root_inventories_exact": (
            baseline_inventory["safe_and_exact"] and
            candidate_inventory["safe_and_exact"]),
        "independently_materialized_files": all(independent_files),
        "scan_ids_exact": (baseline_ids == expected_scan_ids == candidate_ids),
        "proposal_counts_exact": (
            len(baseline_pool) == expected_proposals == len(candidate_pool)),
        "all_scans_successful": all(
            row.get("status") == "ok"
            for row in (*baseline_scans, *candidate_scans)),
        "cohort_hash_exact": (
            baseline_preparation.get("cohort_hash") ==
            candidate_preparation.get("cohort_hash")),
        "config_exact": canonical_hash(baseline_config) == canonical_hash(candidate_config),
        "threshold_exact": (
            baseline_preparation.get("minimum_pool_threshold") ==
            candidate_preparation.get("minimum_pool_threshold")),
        "implementation_changed": (
            baseline_plan.get("implementation_hash") !=
            candidate_plan.get("implementation_hash")),
        "candidate_uses_current_implementation": (
            candidate_plan.get("implementation_hash") ==
            _preparation_implementation_hash()),
        "plan_keys_valid": (
            _valid_plan_key(baseline_plan) and _valid_plan_key(candidate_plan)),
        "preparation_key_changed": (
            baseline_plan.get("preparation_key") !=
            candidate_plan.get("preparation_key")),
        "plan_semantics_exact": (
            canonical_hash(_plan_semantics(baseline_plan)) ==
            canonical_hash(_plan_semantics(candidate_plan))),
        "preparation_semantics_exact": (
            canonical_hash(_preparation_semantics(baseline_preparation)) ==
            canonical_hash(_preparation_semantics(candidate_preparation))),
        "marker_keys_match_plans": (
            baseline_preparation.get("preparation_key") ==
            baseline_plan.get("preparation_key") and
            candidate_preparation.get("preparation_key") ==
            candidate_plan.get("preparation_key")),
        "artifact_inventories_exact": (
            set(baseline_preparation.get("artifacts") or {}) ==
            set(candidate_preparation.get("artifacts") or {})),
        "preparation_timings_valid": timings_valid,
        "proposal_pool_bytes_exact": (
            aggregate_pool_evidence["baseline"] ==
            aggregate_pool_evidence["candidate"]),
        "normalized_scan_records_exact": (
            canonical_hash(normalized_baseline_scans) ==
            canonical_hash(normalized_candidate_scans)),
        "per_scan_records_exact": all(
            row["exact"] for row in record_comparisons.values()),
        "checkpoint_semantics_exact": all(
            row["exact"] for row in checkpoint_comparisons.values()),
        "all_scientific_files_byte_exact": all(
            evidence["exact"] for scan in file_comparisons.values()
            for evidence in scan.values()),
    }

    _verify_preparation_artifacts(
        baseline, baseline_scans, baseline_pool,
        baseline_preparation, baseline_config)
    _verify_preparation_artifacts(
        candidate, candidate_scans, candidate_pool,
        candidate_preparation, candidate_config)
    final_baseline_inventory = _root_inventory(baseline, expected_scan_ids)
    final_candidate_inventory = _root_inventory(candidate, expected_scan_ids)
    final_code_sha256 = {name: sha256(source_root/name) for name in code_sha256}
    if (sha256(baseline_preparation_path) != baseline_sha or
            sha256(candidate_preparation_path) != candidate_sha or
            sha256(baseline_plan_path) != baseline_plan_sha or
            sha256(candidate_plan_path) != candidate_plan_sha or
            final_baseline_inventory != baseline_inventory or
            final_candidate_inventory != candidate_inventory or
            final_code_sha256 != code_sha256):
        raise ValueError("Baseline or comparison implementation changed during audit")
    checks["root_inventories_unchanged"] = True
    output = {
        "schema_version": 1,
        "kind": "preparation_end_to_end_exact_regression",
        "baseline": str(baseline),
        "candidate": str(candidate),
        "baseline_preparation_sha256": baseline_sha,
        "candidate_preparation_sha256": candidate_sha,
        "baseline_plan_sha256": baseline_plan_sha,
        "candidate_plan_sha256": candidate_plan_sha,
        "comparison_code_sha256": code_sha256,
        "baseline_root_inventory": baseline_inventory,
        "candidate_root_inventory": candidate_inventory,
        "final_baseline_root_inventory": final_baseline_inventory,
        "final_candidate_root_inventory": final_candidate_inventory,
        "expected_scan_ids": expected_scan_ids,
        "expected_proposals": expected_proposals,
        "file_comparisons": file_comparisons,
        "record_comparisons": record_comparisons,
        "checkpoint_comparisons": checkpoint_comparisons,
        "preparation_seconds": timing,
        "aggregate_pool_evidence": aggregate_pool_evidence,
        "checks": checks,
        "acceptance_pass": all(checks.values()),
    }
    write_json(report, output)
    print(f"REPORT={report}")
    print(f"REPORT_SHA256={sha256(report)}")
    print(f"PREPARATION_REGRESSION_PASS={output['acceptance_pass']}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-scan-id", action="append", required=True)
    parser.add_argument("--expected-proposals", type=int, required=True)
    args = parser.parse_args(argv)
    result = compare(
        args.baseline, args.candidate, args.report,
        args.expected_baseline_sha256, args.expected_scan_id,
        args.expected_proposals)
    return 0 if result["acceptance_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
