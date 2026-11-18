"""Strict LUNA16 reference and patient-grouped fold adapter."""
from __future__ import annotations
import csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .provenance import (read_json, sha256, canonical_hash, safe_output,
                         write_json, write_jsonl)


_ASSIGNMENT_FIELDS = [
    "seriesuid", "patient_id", "official_subset",
    "patient_grouped_subset", "assignment_changed",
]
_REFERENCE_FIELDS = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]


def _rows(path, fields):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(
                f"Unexpected CSV fields in {Path(path).name}: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _references(path, assignments, state):
    result = defaultdict(list)
    for index, row in enumerate(_rows(path, _REFERENCE_FIELDS), 1):
        uid = row["seriesuid"]
        if uid not in assignments:
            raise ValueError(f"{state} reference outside assignment: {uid}")
        try:
            center = [float(row[key]) for key in ("coordX", "coordY", "coordZ")]
            diameter = float(row["diameter_mm"])
        except ValueError as error:
            raise ValueError(f"Invalid numeric {state} reference row {index}") from error
        if not np.isfinite(center).all() or not np.isfinite(diameter):
            raise ValueError(f"Nonfinite {state} reference row {index}")
        if state == "included":
            if diameter <= 0:
                raise ValueError(f"Nonpositive included diameter row {index}")
            effective = diameter
        else:
            effective = 10.0 if diameter < 0 else diameter
            if effective <= 0:
                raise ValueError(f"Invalid excluded diameter row {index}")
        result[uid].append({
            "id": f"luna16:{state}:{uid}:{index:06d}",
            "center_lps": center,
            "diameter_mm": diameter,
            "radius_mm": effective / 2.0,
            "point_only": True,
            "mask_path": None,
            "strict_radius": True,
            "reference_state": state,
        })
    return dict(result)


def load_protocol(assignments_path, protocol_path, annotations_path,
                  excluded_annotations_path, fold):
    """Validate and adapt the frozen LUNA16 inputs without reading CT pixels."""
    protocol = read_json(protocol_path)
    if protocol.get("schema_version") != 2 or not protocol.get("acceptance_pass"):
        raise ValueError("Approved LUNA16 protocol v2 required")
    if not isinstance(fold, int) or not 0 <= fold <= 9:
        raise ValueError("Fold must be an integer from 0 through 9")

    declared = protocol.get("reference_files", {})
    required = {
        "assignments": (
            assignments_path, protocol.get("assignment_sha256")),
        "annotations": (
            annotations_path, declared.get("annotations", {}).get("sha256")),
        "excluded_annotations": (
            excluded_annotations_path,
            declared.get("excluded_annotations", {}).get("sha256")),
    }
    hashes = {}
    for name, (path, expected) in required.items():
        actual = sha256(path)
        hashes[name] = actual
        if not expected or actual != expected:
            raise ValueError(f"{name} SHA-256 disagrees with frozen protocol")

    rows = _rows(assignments_path, _ASSIGNMENT_FIELDS)
    assignments = {}
    for row in rows:
        uid = row["seriesuid"]
        if uid in assignments:
            raise ValueError(f"Duplicate assigned series: {uid}")
        try:
            official = int(row["official_subset"])
            grouped = int(row["patient_grouped_subset"])
        except ValueError as error:
            raise ValueError(f"Invalid subset for series: {uid}") from error
        if official not in range(10) or grouped not in range(10):
            raise ValueError(f"Subset outside 0..9 for series: {uid}")
        changed = row["assignment_changed"]
        if changed not in ("True", "False"):
            raise ValueError(f"Invalid assignment_changed for series: {uid}")
        if (official != grouped) != (changed == "True"):
            raise ValueError(f"Inconsistent assignment_changed for series: {uid}")
        assignments[uid] = {
            "seriesuid": uid,
            "patient_id": row["patient_id"],
            "official_subset": official,
            "patient_grouped_subset": grouped,
            "assignment_changed": changed == "True",
        }

    if len(assignments) != protocol.get("series_count"):
        raise ValueError("Assignment series count disagrees with protocol")
    patients = defaultdict(set)
    for row in assignments.values():
        patients[row["patient_id"]].add(row["patient_grouped_subset"])
    leaked = {p: sorted(v) for p, v in patients.items() if len(v) != 1}
    if leaked:
        raise ValueError(f"Patient leakage across grouped subsets: {leaked}")
    if len(patients) != protocol.get("patient_count"):
        raise ValueError("Assignment patient count disagrees with protocol")

    included = _references(annotations_path, assignments, "included")
    excluded = _references(excluded_annotations_path, assignments, "excluded")
    folds = {item["fold"]: item for item in protocol.get("folds", [])}
    if set(folds) != set(range(10)):
        raise ValueError("Protocol must define exactly folds 0 through 9")
    selected = folds[fold]
    subset_roles = {}
    for role in ("train", "validation", "test"):
        subsets = selected["roles"][role]["subsets"]
        for subset in subsets:
            if subset in subset_roles:
                raise ValueError("A subset appears in multiple fold roles")
            subset_roles[subset] = role
    if set(subset_roles) != set(range(10)):
        raise ValueError("Fold roles do not partition all ten subsets")

    roles = {
        uid: subset_roles[row["patient_grouped_subset"]]
        for uid, row in assignments.items()
    }
    counts = Counter(roles.values())
    for role in ("train", "validation", "test"):
        if counts[role] != selected["roles"][role]["series"]:
            raise ValueError(f"{role} series count disagrees with protocol")

    return {
        "protocol": protocol,
        "hashes": hashes,
        "assignments": assignments,
        "included": included,
        "excluded": excluded,
        "roles": roles,
        "fold": fold,
    }



def build_cohort(inventory, assignments_path, protocol_path, annotations_path,
                 excluded_annotations_path, fold, output):
    """Build fold-specific manifests from a previously hashed DICOM inventory."""
    spec = load_protocol(
        assignments_path, protocol_path, annotations_path,
        excluded_annotations_path, fold)
    output = Path(output).resolve()
    safe_output(output, inventory["source_root"])
    output.mkdir(parents=True, exist_ok=False)

    by_series = defaultdict(list)
    for row in inventory["slices"]:
        by_series[row["series_uid"]].append(row)
    missing = sorted(set(spec["assignments"]) - set(by_series))
    if missing:
        raise ValueError(f"Assigned LUNA16 series missing from inventory: {missing[:5]}")

    audit_path = output / "luna16_reference_adapter_audit.json"
    audit = {
        "schema_version": 1,
        "status": "PROTOCOL_BOUND_REFERENCES_NOT_YET_PREPARED",
        "fold": fold,
        "protocol_hashes": spec["hashes"],
        "inventory_hash": canonical_hash(inventory),
        "series": len(spec["assignments"]),
        "included_rows": sum(map(len, spec["included"].values())),
        "excluded_rows": sum(map(len, spec["excluded"].values())),
        "strict_radius_matching": True,
    }
    write_json(audit_path, audit)

    scans = []
    for uid, assignment in sorted(spec["assignments"].items()):
        rows = by_series[uid]
        patients = {row["patient_id"] for row in rows}
        studies = {row["study_uid"] for row in rows}
        if patients != {assignment["patient_id"]}:
            raise ValueError(f"Assignment/inventory patient mismatch: {uid}")
        if len(studies) != 1:
            raise ValueError(f"Series spans multiple studies: {uid}")
        if any("LOCALIZER" in row["image_type"] for row in rows):
            raise ValueError(f"Localizer in assigned LUNA16 series: {uid}")
        if any(not row.get("sha256") for row in rows):
            raise ValueError(
                "LUNA16 cohort construction requires an inventory created with --hash-files")
        paths = [row["path"] for row in rows]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate inventory paths: {uid}")
        scan_id = "scan_" + canonical_hash({
            "patient": assignment["patient_id"],
            "study": next(iter(studies)),
            "series": uid,
        })[:20]
        role = spec["roles"][uid]
        scans.append({
            "scan_id": scan_id,
            "patient_id": assignment["patient_id"],
            "study_uid": next(iter(studies)),
            "series_uid": uid,
            "source_root": inventory["source_root"],
            "dicom_paths": paths,
            "source_hashes": {row["path"]: row["sha256"] for row in rows},
            "positive": spec["included"].get(uid, []),
            "ignored": spec["excluded"].get(uid, []),
            "reference_audit_path": str(audit_path),
            "unresolved_associations": 0,
            "official_subset": assignment["official_subset"],
            "patient_grouped_subset": assignment["patient_grouped_subset"],
            "split": "val" if role == "validation" else role,
            "fold": fold,
            "status": "manifest_only_not_prepared",
        })

    patient_roles = defaultdict(set)
    for scan in scans:
        patient_roles[scan["patient_id"]].add(scan["split"])
    leakage = {p: sorted(v) for p, v in patient_roles.items() if len(v) != 1}
    if leakage:
        raise ValueError(f"Patient leakage in emitted manifests: {leakage}")

    development = [scan for scan in scans if scan["split"] != "test"]
    withheld = [scan for scan in scans if scan["split"] == "test"]
    write_jsonl(output / "cohort_development.jsonl", development)
    write_jsonl(output / "cohort_test_WITHHOLD.jsonl", withheld)
    write_json(output / "cohort_summary.json", {
        **audit,
        "status": "MANIFESTS_CREATED_NOT_PREPARED_NOT_FROZEN",
        "patients": len(patient_roles),
        "role_counts": dict(sorted(Counter(s["split"] for s in scans).items())),
        "development_manifest_sha256": sha256(output / "cohort_development.jsonl"),
        "withheld_manifest_sha256": sha256(output / "cohort_test_WITHHOLD.jsonl"),
        "acceptance_pass": len(scans) == spec["protocol"]["series_count"] and not leakage,
    })
    return scans
