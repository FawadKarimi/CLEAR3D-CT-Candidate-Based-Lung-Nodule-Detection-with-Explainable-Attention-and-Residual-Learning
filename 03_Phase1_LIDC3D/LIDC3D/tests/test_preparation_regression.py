import json
from pathlib import Path
import pytest

from lidc3d import preparation_regression


def _json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _prepared(root, implementation):
    root.mkdir()
    scan_id = "scan_test"
    scan_dir = root/scan_id
    scan_dir.mkdir()
    config = {"candidates": {"threshold": None},
              "preprocess": {"spacing_mm": 1.0}}
    record = {"scan_id": scan_id, "series_uid": "1.2.3", "status": "ok",
              "failure_reason": None,
              "image_path": str((root/scan_id/"image.npy").resolve()),
              "valid_path": str((root/scan_id/"valid.npy").resolve()),
              "lung_mask_path": str((root/scan_id/"lung_mask.npy").resolve()),
              "base_mask_path": str((root/scan_id/"base_mask.npy").resolve())}
    plan = {
        "schema_version": 1, "kind": "lidc3d_preparation_plan",
        "implementation": "atomic_scan_checkpoints_v1",
        "implementation_hash": implementation,
        "cohort_hash": "cohort",
        "config_hash": "config", "minimum_pool_threshold": 0.005,
        "scan_ids": [scan_id]}
    plan["preparation_key"] = preparation_regression.canonical_hash(plan)
    preparation = {"schema_version": 1, "config": config,
                   "cohort_hash": "cohort", "minimum_pool_threshold": 0.005,
                   "preparation_key": plan["preparation_key"],
                   "status": "PROPOSALS_NOT_YET_FROZEN",
                   "files": {"scans.jsonl": "scan-evidence",
                             "proposal_pool.jsonl": "pool-evidence"},
                   "artifacts": {
                       f"{scan_id}/record.json": {"bytes": 1, "sha256": "record"},
                       f"{scan_id}/checkpoint.json": {
                           "bytes": 1, "sha256": "checkpoint"}}}
    _json(root/"preparation_plan.json", plan)
    _jsonl(root/"scans.jsonl", [record])
    proposal = {"scan_id": scan_id, "id": f"{scan_id}:candidate:000000",
                "zyx": [1.0, 2.0, 3.0], "center_lps": [3.0, 2.0, 1.0],
                "log_response": 0.5, "sigma_mm": 1.0}
    _jsonl(root/"proposal_pool.jsonl", [proposal])
    _jsonl(scan_dir/"proposals.jsonl", [proposal])
    _json(scan_dir/"record.json", record)
    _json(scan_dir/"geometry.json", {"affine": [[1, 0], [0, 1]]})
    for name in preparation_regression._ARRAY_FILES:
        (scan_dir/name).write_bytes(b"same-array")
    checkpoint_files = {
        name: {"bytes": (scan_dir/name).stat().st_size,
               "sha256": preparation_regression.sha256(scan_dir/name)}
        for name in {"record.json", "proposals.jsonl", "geometry.json",
                     *preparation_regression._ARRAY_FILES}}
    _json(scan_dir/"checkpoint.json", {
        "schema_version": 1, "kind": "lidc3d_scan_preparation",
        "scan_id": scan_id, "scan_hash": "scan-hash", "proposal_count": 1,
        "status": "ok", "preparation_key": plan["preparation_key"],
        "preparation_seconds": 1.0, "files": checkpoint_files})
    _json(root/"preparation.json", preparation)
    return preparation


def test_end_to_end_comparison_allows_only_expected_bookkeeping_differences(
        tmp_path, monkeypatch):
    baseline, candidate = tmp_path/"v2", tmp_path/"v3"
    _prepared(baseline, "old")
    _prepared(candidate,
              preparation_regression._preparation_implementation_hash())
    trusted = preparation_regression.sha256(baseline/"preparation.json")
    monkeypatch.setattr(preparation_regression,
                        "_verify_preparation_artifacts", lambda *args: None)
    report = tmp_path/"report.json"
    result = preparation_regression.compare(
        baseline, candidate, report, trusted, ["scan_test"], 1)
    assert result["acceptance_pass"]
    assert report.is_file()
    assert all(result["checks"].values())

    (candidate/"scan_test"/"image.npy").write_bytes(b"changed-array")
    changed = preparation_regression.compare(
        baseline, candidate, tmp_path/"changed.json",
        trusted, ["scan_test"], 1)
    assert not changed["acceptance_pass"]
    assert not changed["checks"]["all_scientific_files_byte_exact"]


def test_normalization_does_not_hide_scientific_record_changes():
    left = {"scan_id": "scan", "image_path": "/v2/scan/image.npy",
            "candidate_diagnostics": {"before_cap": 2}}
    right = {"scan_id": "scan", "image_path": "/v3/scan/image.npy",
             "candidate_diagnostics": {"before_cap": 3}}
    assert preparation_regression._normalize_record(left) != \
        preparation_regression._normalize_record(right)


def test_inventory_and_record_paths_are_exact(tmp_path):
    root = tmp_path/"prepared"
    _prepared(root, "implementation")
    inventory = preparation_regression._root_inventory(root, ["scan_test"])
    assert inventory["safe_and_exact"]

    extra = root/"scan_test"/"unexpected.tmp"
    extra.write_text("unexpected")
    assert not preparation_regression._root_inventory(
        root, ["scan_test"])["safe_and_exact"]
    extra.unlink()

    record = json.loads((root/"scan_test"/"record.json").read_text())
    record["image_path"] = str(root/"scan_test"/".."/"scan_test"/"image.npy")
    try:
        preparation_regression._normalize_record(record, root, "scan_test")
    except ValueError as error:
        assert "not bound" in str(error)
    else:
        raise AssertionError("noncanonical record path was accepted")


@pytest.mark.parametrize("target", ["preparation.json", "checkpoint.json"])
def test_unknown_schema_field_is_rejected(tmp_path, monkeypatch, target):
    baseline, candidate = tmp_path/"v2", tmp_path/"v3"
    _prepared(baseline, "old")
    _prepared(candidate,
              preparation_regression._preparation_implementation_hash())
    path = (candidate/target if target == "preparation.json" else
            candidate/"scan_test"/target)
    value = json.loads(path.read_text())
    value["unexpected"] = True
    _json(path, value)
    trusted = preparation_regression.sha256(baseline/"preparation.json")
    monkeypatch.setattr(preparation_regression,
                        "_verify_preparation_artifacts", lambda *args: None)
    with pytest.raises(ValueError, match="schema changed"):
        preparation_regression.compare(
            baseline, candidate, tmp_path/"report.json",
            trusted, ["scan_test"], 1)
