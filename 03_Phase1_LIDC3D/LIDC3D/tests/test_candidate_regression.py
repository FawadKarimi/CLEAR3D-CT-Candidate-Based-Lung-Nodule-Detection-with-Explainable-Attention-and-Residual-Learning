import json
from pathlib import Path
import numpy as np
import pytest

from lidc3d import candidate_regression


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_compare_scan_requires_exact_rows_capped_output_and_diagnostics(
        tmp_path, monkeypatch):
    scan = {"scan_id": "scan_test", "series_uid": "1.2.3"}
    scan_dir = tmp_path/scan["scan_id"]
    scan_dir.mkdir()
    expected = [
        {"zyx": [1.0, 2.0, 3.0], "log_response": 0.8,
         "sigma_mm": 1.5},
        {"zyx": [4.0, 5.0, 6.0], "log_response": 0.7,
         "sigma_mm": 2.0},
    ]
    _write_jsonl(scan_dir/"proposals.jsonl", [
        {**row, "id": f"scan_test:candidate:{index:06d}",
         "scan_id": "scan_test", "center_lps": row["zyx"]}
        for index, row in enumerate(expected)])
    _write_json(scan_dir/"record.json", {
        "candidate_diagnostics": {"before_cap": 2, "nms_suppressed": 1}})
    _write_json(scan_dir/"geometry.json", {"affine": np.eye(4).tolist()})
    np.save(scan_dir/"image.npy", np.zeros((2, 2, 2), np.float32))
    np.save(scan_dir/"lung_mask.npy", np.ones((2, 2, 2), bool))

    def exact_generate(image, mask, config, spacing):
        return expected[:1], {"before_cap": 2, "nms_suppressed": 1,
                              "precap_candidates": expected}

    monkeypatch.setattr(candidate_regression.candidates, "generate", exact_generate)
    result = candidate_regression._compare_scan(
        tmp_path, scan, {"cap": 1}, 1.0)
    assert result["acceptance_pass"]
    assert result["first_proposal_difference"] is None

    def changed_generate(image, mask, config, spacing):
        changed = [{**expected[0], "log_response": 0.75}, expected[1]]
        return changed[:1], {"before_cap": 2, "nms_suppressed": 0,
                             "precap_candidates": changed}

    monkeypatch.setattr(candidate_regression.candidates, "generate", changed_generate)
    result = candidate_regression._compare_scan(
        tmp_path, scan, {"cap": 1}, 1.0)
    assert not result["acceptance_pass"]
    assert result["first_proposal_difference"]["index"] == 0
    assert not result["checks"]["proposal_rows_exact"]
    assert not result["checks"]["capped_candidates_exact"]
    assert not result["checks"]["diagnostics_exact"]


def test_compare_scan_rejects_json_type_changes(tmp_path, monkeypatch):
    scan = {"scan_id": "scan_test", "series_uid": "1.2.3"}
    scan_dir = tmp_path/scan["scan_id"]
    scan_dir.mkdir()
    expected = {"zyx": [1.0, 2.0, 3.0], "log_response": 1,
                "sigma_mm": 1.0}
    _write_jsonl(scan_dir/"proposals.jsonl", [{
        **expected, "id": "scan_test:candidate:000000",
        "scan_id": "scan_test", "center_lps": expected["zyx"]}])
    _write_json(scan_dir/"record.json", {
        "candidate_diagnostics": {"cap_hit": True}})
    _write_json(scan_dir/"geometry.json", {"affine": np.eye(4).tolist()})
    np.save(scan_dir/"image.npy", np.zeros((1, 1, 1), np.float32))
    np.save(scan_dir/"lung_mask.npy", np.ones((1, 1, 1), bool))
    actual = [{**expected, "log_response": 1.0}]
    monkeypatch.setattr(candidate_regression.candidates, "generate",
                        lambda *args: (actual, {
                            "cap_hit": 1, "precap_candidates": actual}))
    result = candidate_regression._compare_scan(
        tmp_path, scan, {"cap": 1}, 1.0)
    assert not result["acceptance_pass"]
    assert not result["checks"]["proposal_rows_exact"]
    assert not result["checks"]["capped_candidates_exact"]
    assert not result["checks"]["diagnostics_exact"]


def test_main_returns_nonzero_on_failed_acceptance(monkeypatch, tmp_path):
    monkeypatch.setattr(candidate_regression, "validate",
                        lambda *args: {"acceptance_pass": False})
    assert candidate_regression.main([
        "--prepared", str(tmp_path/"baseline"),
        "--report", str(tmp_path/"report.json"),
        "--expected-preparation-sha256", "a" * 64,
        "--expected-scan-id", "scan_test",
        "--expected-proposals", "1"]) == 1


def test_validate_pins_baseline_hash_scan_order_and_proposal_count(
        tmp_path, monkeypatch):
    prepared = tmp_path/"prepared"
    prepared.mkdir()
    preparation = {
        "config": {"candidates": {"threshold": None},
                   "preprocess": {"spacing_mm": 1.0}},
        "minimum_pool_threshold": 0.005,
    }
    _write_json(prepared/"preparation.json", preparation)
    scans = [{"scan_id": "scan_test", "series_uid": "1.2.3",
              "status": "ok"}]
    _write_jsonl(prepared/"scans.jsonl", scans)
    _write_jsonl(prepared/"proposal_pool.jsonl", [{"scan_id": "scan_test"}])
    trusted_hash = candidate_regression.sha256(prepared/"preparation.json")
    monkeypatch.setattr(candidate_regression,
                        "_verify_preparation_artifacts", lambda *args: None)
    monkeypatch.setattr(candidate_regression, "_compare_scan",
                        lambda *args: {
                            "checks": {"proposal_rows_exact": True,
                                       "capped_candidates_exact": True,
                                       "diagnostics_exact": True},
                            "scan_id": "scan_test", "seconds": 0.0,
                            "acceptance_pass": True})
    report = tmp_path/"report.json"
    result = candidate_regression.validate(
        prepared, report, trusted_hash, ["scan_test"], 1)
    assert result["acceptance_pass"]
    assert report.is_file()

    with pytest.raises(ValueError, match="accepted hash"):
        candidate_regression.validate(
            prepared, tmp_path/"bad-hash.json", "0" * 64,
            ["scan_test"], 1)
    with pytest.raises(ValueError, match="identities"):
        candidate_regression.validate(
            prepared, tmp_path/"bad-scan.json", trusted_hash,
            ["scan_other"], 1)
    with pytest.raises(ValueError, match="proposal count"):
        candidate_regression.validate(
            prepared, tmp_path/"bad-count.json", trusted_hash,
            ["scan_test"], 2)
