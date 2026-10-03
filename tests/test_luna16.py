import csv
import json
from pathlib import Path
import pytest
from lidc3d.luna16 import load_protocol, build_cohort
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



def inventory_fixture(tmp_path, hashed=True):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for index, (uid, patient) in enumerate((("s0", "p0"), ("s1", "p1"), ("s2", "p2"))):
        path = raw / f"{uid}.dcm"
        path.write_bytes(uid.encode())
        row = {
            "path": str(path), "patient_id": patient, "study_uid": f"study-{index}",
            "series_uid": uid, "sop_uid": f"sop-{index}", "rows": 2, "columns": 2,
            "position": [0, 0, index], "orientation": [1, 0, 0, 0, 1, 0],
            "pixel_spacing": [1, 1], "image_type": ["ORIGINAL", "PRIMARY", "AXIAL"],
            "transfer_syntax": "1.2.840.10008.1.2.1",
        }
        if hashed:
            row["sha256"] = sha256(path)
        rows.append(row)
    return {"source_root": str(raw), "slices": rows, "xml": [], "errors": [],
            "status": "HEADER_INVENTORY_ONLY_NOT_COHORT_FREEZE"}


def test_build_cohort_emits_patient_independent_manifests(tmp_path):
    paths = fixture(tmp_path)
    scans = build_cohort(
        inventory_fixture(tmp_path), *paths, fold=0, output=tmp_path / "out")
    assert {scan["split"] for scan in scans} == {"train", "val", "test"}
    assert len({scan["patient_id"] for scan in scans}) == 3
    assert all(scan["status"] == "manifest_only_not_prepared" for scan in scans)
    summary = json.loads((tmp_path / "out" / "cohort_summary.json").read_text())
    assert summary["acceptance_pass"] is True
    assert summary["role_counts"] == {"test": 1, "train": 1, "val": 1}


def test_build_cohort_requires_prehashed_inventory(tmp_path):
    paths = fixture(tmp_path)
    with pytest.raises(ValueError, match="--hash-files"):
        build_cohort(
            inventory_fixture(tmp_path, hashed=False), *paths,
            fold=0, output=tmp_path / "out")


def test_luna16_radius_boundary_is_strict():
    from lidc3d.references import candidate_label, inside_ignore
    from lidc3d.evaluation import match
    region = {
        "id": "r", "center_lps": [0, 0, 0], "radius_mm": 5,
        "strict_radius": True, "point_only": True,
    }
    boundary = [5, 0, 0]
    assert candidate_label(boundary, [region], []) == 0
    assert inside_ignore(boundary, [region]) is False
    result = match(
        [{"id": "p", "center_lps": boundary, "score": 1.0}],
        [region], [])
    assert result["tp"] == 0 and result["fp"] == 1 and result["fn"] == 1
