import csv
import json
from pathlib import Path
import pytest
from lidc3d.luna16 import load_protocol
from lidc3d.provenance import sha256


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path):
    assignments = tmp_path / "assignments.csv"
    annotations = tmp_path / "annotations.csv"
    excluded = tmp_path / "excluded.csv"
    protocol = tmp_path / "protocol.json"
    af = ["seriesuid", "patient_id", "official_subset",
          "patient_grouped_subset", "assignment_changed"]
    rf = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]
    assignment_rows = [
        {"seriesuid": "s0", "patient_id": "p0", "official_subset": 0,
         "patient_grouped_subset": 0, "assignment_changed": "False"},
        {"seriesuid": "s1", "patient_id": "p1", "official_subset": 1,
         "patient_grouped_subset": 1, "assignment_changed": "False"},
        {"seriesuid": "s2", "patient_id": "p2", "official_subset": 2,
         "patient_grouped_subset": 2, "assignment_changed": "False"},
    ]
    write_csv(assignments, af, assignment_rows)
    write_csv(annotations, rf, [
        {"seriesuid": uid, "coordX": 1, "coordY": 2, "coordZ": 3,
         "diameter_mm": 6} for uid in ("s0", "s1", "s2")
    ])
    write_csv(excluded, rf, [
        {"seriesuid": "s0", "coordX": 4, "coordY": 5, "coordZ": 6,
         "diameter_mm": -1}
    ])
    folds = []
    for fold in range(10):
        test = fold
        validation = (fold + 1) % 10
        train = [s for s in range(10) if s not in (test, validation)]
        counts = {
            "train": sum(int(r["patient_grouped_subset"]) in train
                         for r in assignment_rows),
            "validation": sum(int(r["patient_grouped_subset"]) == validation
                              for r in assignment_rows),
            "test": sum(int(r["patient_grouped_subset"]) == test
                        for r in assignment_rows),
        }
        folds.append({"fold": fold, "roles": {
            "train": {"subsets": train, "series": counts["train"]},
            "validation": {"subsets": [validation],
                           "series": counts["validation"]},
            "test": {"subsets": [test], "series": counts["test"]},
        }})
    data = {
        "schema_version": 2,
        "acceptance_pass": True,
        "assignment_sha256": sha256(assignments),
        "reference_files": {
            "annotations": {"sha256": sha256(annotations)},
            "excluded_annotations": {"sha256": sha256(excluded)},
        },
        "series_count": 3,
        "patient_count": 3,
        "folds": folds,
    }
    protocol.write_text(json.dumps(data))
    return assignments, protocol, annotations, excluded


def test_protocol_adapter_binds_references_and_fold_roles(tmp_path):
    paths = fixture(tmp_path)
    result = load_protocol(*paths, fold=0)
    assert result["roles"] == {"s0": "test", "s1": "validation", "s2": "train"}
    assert result["included"]["s0"][0]["radius_mm"] == 3
    ignored = result["excluded"]["s0"][0]
    assert ignored["radius_mm"] == 5
    assert ignored["diameter_mm"] == -1
    assert ignored["strict_radius"] is True


def test_protocol_adapter_rejects_reference_hash_drift(tmp_path):
    paths = fixture(tmp_path)
    paths[2].write_text(paths[2].read_text() + "\n")
    with pytest.raises(ValueError, match="annotations SHA-256"):
        load_protocol(*paths, fold=0)


def test_protocol_adapter_rejects_patient_leakage(tmp_path):
    paths = fixture(tmp_path)
    rows = list(csv.DictReader(paths[0].open()))
    rows[1]["patient_id"] = "p0"
    write_csv(paths[0], list(rows[0]), rows)
    protocol = json.loads(paths[1].read_text())
    protocol["assignment_sha256"] = sha256(paths[0])
    protocol["patient_count"] = 2
    paths[1].write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Patient leakage"):
        load_protocol(*paths, fold=0)


def test_protocol_adapter_rejects_nonpositive_included_diameter(tmp_path):
    paths = fixture(tmp_path)
    rows = list(csv.DictReader(paths[2].open()))
    rows[0]["diameter_mm"] = 0
    write_csv(paths[2], list(rows[0]), rows)
    protocol = json.loads(paths[1].read_text())
    protocol["reference_files"]["annotations"]["sha256"] = sha256(paths[2])
    paths[1].write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Nonpositive included diameter"):
        load_protocol(*paths, fold=0)
