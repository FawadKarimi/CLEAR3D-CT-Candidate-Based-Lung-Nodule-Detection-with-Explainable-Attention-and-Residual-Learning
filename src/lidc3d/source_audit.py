"""Diagnostic source audit. No reconstruction, slice selection, or release policy.

Only repeated-position files are opened; other evidence is the hashed inventory.
The pylidc package is never imported. SQLite is read directly in read-only mode.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import logging
import math
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import sysconfig
import warnings

import numpy as np
import pydicom
import yaml

from .provenance import sha256, write_json

SCHEMA_VERSION = 2
CONFIG_SCHEMA_VERSION = 1
TOLERANCES = {
    "repeated_position_atol_mm", "z_atol_mm", "orientation_atol",
    "in_plane_spacing_atol_mm", "position_drift_atol_mm", "spacing_atol_mm",
    "spacing_rtol", "hu_atol",
}
IDENTITY = ("patient_id", "study_uid", "series_uid")
REQUIRED = {
    "scans": {"id": "integer", "patient_id": "text", "study_instance_uid": "text",
              "series_instance_uid": "text"},
    "annotations": {"id": "integer", "scan_id": "integer"},
    "contours": {"id": "integer", "annotation_id": "integer", "inclusion": "integer",
                 "image_z_position": "real", "dicom_file_name": "text"},
    "zvals": {"id": "integer", "scan_id": "integer", "val": "real"},
}
METADATA = (
    "PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
    "SOPClassUID", "Modality", "ImagePositionPatient", "ImageOrientationPatient",
    "PixelSpacing", "SliceThickness", "SpacingBetweenSlices", "FrameOfReferenceUID",
    "InstanceNumber", "AcquisitionNumber", "AcquisitionDate", "AcquisitionTime",
    "AcquisitionDateTime", "ContentDate", "ContentTime", "SeriesNumber",
    "TemporalPositionIdentifier", "ReconstructionDiameter", "ConvolutionKernel",
    "KVP", "Exposure", "GantryDetectorTilt", "ImageType", "RescaleSlope",
    "RescaleIntercept", "RescaleType", "PixelPaddingValue", "PixelPaddingRangeLimit",
    "Rows", "Columns", "SamplesPerPixel", "PhotometricInterpretation",
    "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation", "NumberOfFrames",
    "ModalityLUTSequence", "RealWorldValueMappingSequence",
)


def _error(error):
    return {"type": type(error).__name__, "message": str(error)}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value, length):
    try:
        array = np.asarray(value, dtype=float)
        return array if array.shape == (length,) and np.isfinite(array).all() else None
    except (TypeError, ValueError):
        return None


def _json_value(value):
    """Retain invalid values explicitly without emitting nonstandard JSON NaN."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        return [_json_value(v) for v in value]
    return str(value)


def _emit(handle, row):
    handle.write(json.dumps(_json_value({"schema_version": SCHEMA_VERSION, **row}),
                            sort_keys=True, allow_nan=False) + "\n")


def _identity(row):
    return tuple(row[key] for key in IDENTITY)


def _inside(path, root):
    return path == root or root in path.parents


def _load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported audit configuration schema_version")
    if set(config) - {"schema_version", "tolerances", "protected_roots"}:
        raise ValueError("Unknown audit configuration fields; use configs/source_audit.yaml")
    tolerances = config.get("tolerances", {})
    if not isinstance(tolerances, dict) or set(tolerances) != TOLERANCES:
        raise ValueError("All diagnostic tolerances must be specified explicitly")
    if any(not _finite(v) or v < 0 for v in tolerances.values()):
        raise ValueError("Diagnostic tolerances must be finite and nonnegative")
    if tolerances["orientation_atol"] >= 0.1:
        raise ValueError("orientation_atol must be below 0.1 to diagnose a nondegenerate normal")
    roots = config.get("protected_roots", [])
    if not isinstance(roots, list) or any(not isinstance(p, str) or not Path(p).is_absolute() for p in roots):
        raise ValueError("protected_roots must contain absolute paths")
    return config


def _source_path(path, raw_root):
    """Validate without opening a DICOM; reject traversal even if it returns inside."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"Raw-source boundary violation: DICOM path must be absolute: {path}")
    if ".." in path.parts:
        raise ValueError(f"Raw-source boundary violation: parent traversal in DICOM path: {path}")
    resolved = path.resolve()
    if resolved == raw_root or not _inside(resolved, raw_root):
        raise ValueError(f"Raw-source boundary violation: DICOM path resolves outside source_root: {path}")
    return resolved


def _prepare_inventory(inventory):
    if not isinstance(inventory, dict) or not isinstance(inventory.get("slices"), list):
        raise ValueError("Expected LIDC3D header inventory with slices and source_root")
    root = Path(inventory["source_root"])
    if not root.is_absolute():
        raise ValueError("Inventory source_root must be absolute")
    root = root.resolve()
    rows = inventory["slices"]
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k] for k in (*IDENTITY, "sop_uid", "path")):
            raise ValueError("Inventory slice needs nonempty identity, SOP UID and path strings")
        _source_path(row["path"], root)
    # Stable record IDs do not depend on inventory ordering; retain repeated records.
    rows = sorted(rows, key=lambda r: (*_identity(r), r["path"], r["sop_uid"],
                                     json.dumps(r, sort_keys=True)))
    return root, [{**r, "record_id": i + 1} for i, r in enumerate(rows)]


def _protect_output(output, raw_root, rows, inventory, database, config_path, config, lexical_raw_root):
    resolved = output.resolve()
    roots = {raw_root, Path(lexical_raw_root).absolute(), database.parent.resolve(), config_path.parent.resolve(),
             Path(__file__).resolve().parent.parent, Path(__file__).resolve().parents[2] / ".git"}
    roots.update(Path(p).resolve() for p in config.get("protected_roots", []))
    roots.update(Path(p).resolve() for key, p in sysconfig.get_paths().items()
                 if key in {"purelib", "platlib", "stdlib", "platstdlib"})
    # Protect contained targets reached through directory symlinks as well as aliases.
    roots.update(Path(r["path"]).resolve().parent for r in rows)
    roots.update(Path(r["path"]).absolute().parent for r in rows)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    # Resolve ancestors too: a link beneath a protected alias may point back out.
    ancestry = [resolved, output.absolute(), *(p.resolve() for p in output.parents)]
    if any(_inside(candidate, p) for p in roots for candidate in ancestry):
        raise ValueError("Output is inside a protected source, configuration or package tree")
    if any(_inside(p, resolved) for p in (inventory, database, config_path)):
        raise ValueError("Output would contain an input")
    return sorted(map(str, roots))


def _no_sidecars(path):
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("SQLite sidecar present; audit requires a quiescent database without sidecars")


@contextmanager
def readonly_database(path):
    """No pylidc loader, writable connection, or SQLite sidecar creation."""
    _no_sidecars(path)
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA trusted_schema=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("BEGIN")
        yield conn
    finally:
        conn.close()


def _database(conn, provenance):
    schema = [dict(r) for r in conn.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name")]
    provenance["database_schema"] = schema
    columns = {}
    for table, required in REQUIRED.items():
        if not any(r["type"] == "table" and r["name"] == table and
                   "VIRTUAL TABLE" not in (r["sql"] or "").upper() for r in schema):
            raise ValueError(f"Unsupported schema: missing ordinary table {table}")
        info = [dict(r) for r in conn.execute(f'PRAGMA table_info("{table}")')]
        columns[table] = info
        provenance["database_columns"] = columns
        by_name = {r["name"]: r for r in info}
        for name, kind in required.items():
            declared = by_name.get(name, {}).get("type", "").upper()
            valid = {"integer": "INT" in declared or declared == "BOOLEAN",
                     "text": any(s in declared for s in ("CHAR", "TEXT", "CLOB")),
                     "real": any(s in declared for s in ("REAL", "FLOAT", "DOUBLE", "NUMERIC"))}[kind]
            if not valid:
                raise ValueError(f"Unsupported schema: {table}.{name} needs {kind} affinity")
        if by_name["id"]["pk"] != 1 or sum(bool(r["pk"]) for r in info) != 1:
            raise ValueError(f"Unsupported schema: {table}.id must be the sole primary key")
    if [r[0] for r in conn.execute("PRAGMA quick_check")] != ["ok"]:
        raise ValueError("SQLite quick_check failed")
    scans = [dict(r) for r in conn.execute(
        "SELECT id, patient_id, study_instance_uid AS study_uid, series_instance_uid AS series_uid FROM scans ORDER BY id")]
    annotations = [dict(r) for r in conn.execute("SELECT id, scan_id FROM annotations ORDER BY id")]
    zvals = [dict(r) for r in conn.execute("SELECT id, scan_id, val FROM zvals ORDER BY id")]
    # LEFT JOIN retains orphan contours for an explicit failed audit, never an inner-join loss.
    contours = [dict(r) for r in conn.execute(
        "SELECT c.id, c.annotation_id, c.inclusion, c.image_z_position, c.dicom_file_name, "
        "a.scan_id, s.patient_id, s.study_instance_uid AS study_uid, s.series_instance_uid AS series_uid "
        "FROM contours c LEFT JOIN annotations a ON a.id=c.annotation_id "
        "LEFT JOIN scans s ON s.id=a.scan_id ORDER BY c.id")]
    scan_ids = {s["id"] for s in scans}
    annotation_ids = {a["id"] for a in annotations}
    problems = []
    for s in scans:
        if any(not isinstance(s[k], str) or not s[k] for k in IDENTITY):
            problems.append({"code": "invalid_scan_identity", "scan_id": s["id"]})
    for a in annotations:
        if a["scan_id"] not in scan_ids:
            problems.append({"code": "orphan_annotation", "annotation_id": a["id"]})
    for z in zvals:
        if z["scan_id"] not in scan_ids or not _finite(z["val"]):
            problems.append({"code": "invalid_zval", "zval_id": z["id"]})
    for c in contours:
        if c["annotation_id"] not in annotation_ids or c["scan_id"] not in scan_ids:
            problems.append({"code": "orphan_contour", "contour_id": c["id"]})
    provenance["database_integrity_findings"] = problems
    if problems:
        raise ValueError("Database relational/value integrity failed; see provenance")
    return scans, annotations, zvals, contours


def _position_groups(rows, tolerance):
    valid = [(r, _vector(r.get("position"), 3)) for r in rows]
    valid = [(r, p) for r, p in valid if p is not None]
    valid.sort(key=lambda item: (item[1][2], item[0]["record_id"]))
    parent = list(range(len(valid)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (_, p) in enumerate(valid):
        for j in range(i + 1, len(valid)):
            q = valid[j][1]
            if q[2] - p[2] > tolerance:
                break
            if np.max(np.abs(q - p)) <= tolerance:
                parent[find(j)] = find(i)
    grouped = defaultdict(list)
    for i, (row, _) in enumerate(valid):
        grouped[find(i)].append(row)
    return sorted((sorted(g, key=lambda r: r["record_id"]) for g in grouped.values() if len(g) > 1),
                  key=lambda g: g[0]["record_id"])


def _geometry(rows, tol):
    findings, values = set(), []
    reference = None
    for row in rows:
        p, o, spacing = (_vector(row.get(k), n) for k, n in
                         (("position", 3), ("orientation", 6), ("pixel_spacing", 2)))
        normal = None
        if p is None:
            findings.add("invalid_position")
        if o is None or not np.allclose(
                [np.linalg.norm(o[:3]), np.linalg.norm(o[3:]), o[:3] @ o[3:]],
                [1, 1, 0], atol=tol["orientation_atol"], rtol=0):
            findings.add("invalid_orientation")
        else:
            normal = np.cross(o[:3], o[3:])
            normal /= np.linalg.norm(normal)
            if not np.allclose(np.abs(normal), [0, 0, 1], atol=tol["orientation_atol"], rtol=0):
                findings.add("non_axial_geometry_unsupported")
        if spacing is None or (spacing <= 0).any():
            findings.add("invalid_pixel_spacing")
        if any(not isinstance(row.get(k), int) or isinstance(row[k], bool) or row[k] <= 0 for k in ("rows", "columns")):
            findings.add("invalid_dimensions")
        if "LOCALIZER" in row.get("image_type", []):
            findings.add("localizer_unsupported")
        if reference is None:
            reference = (o, spacing, row.get("rows"), row.get("columns"), normal)
        else:
            ro, rs, height, width, _ = reference
            if o is not None and ro is not None and not np.allclose(o, ro, atol=tol["orientation_atol"], rtol=0):
                findings.add("variable_orientation")
            if spacing is not None and rs is not None and not np.allclose(spacing, rs, atol=tol["in_plane_spacing_atol_mm"], rtol=0):
                findings.add("variable_pixel_spacing")
            if (row.get("rows"), row.get("columns")) != (height, width):
                findings.add("variable_dimensions")
        values.append({"record_id": row["record_id"], "sop_uid": row["sop_uid"],
                       "path": row["path"], "position_lps_mm": row.get("position"),
                       "lps_z_mm": float(p[2]) if p is not None else None,
                       "image_normal_lps": normal.tolist() if normal is not None else None,
                       "normal_projection_mm": float(p @ normal) if p is not None and normal is not None else None})
    if reference and reference[4] is not None and all(v["lps_z_mm"] is not None for v in values):
        ps = np.array([r["position"] for r in rows], dtype=float)
        normal = reference[4]
        relative = ps - ps[0]
        drift = relative - np.outer(relative @ normal, normal)
        if np.linalg.norm(drift, axis=1).max() > tol["position_drift_atol_mm"]:
            findings.add("in_plane_drift_unsupported")
        projections = sorted(set(float(p @ normal) for p in ps))
        gaps = np.diff(projections)
        gaps = gaps[gaps > tol["repeated_position_atol_mm"]]
        if len(gaps) > 1 and not np.allclose(gaps, np.median(gaps), atol=tol["spacing_atol_mm"], rtol=tol["spacing_rtol"]):
            findings.add("irregular_spacing")
    return {"status": "unsupported_or_ambiguous" if findings else ("diagnostic_geometry_consistent" if rows else "no_ct_records"),
            "findings": sorted(findings), "positions": values,
            "evidence": "inventory_headers; frame count and encoding rechecked only for repeated groups"}


def _z_comparison(rows, db_z, atol):
    ct = [(r["record_id"], _vector(r.get("position"), 3)) for r in rows]
    ct = [(i, float(p[2])) for i, p in ct if p is not None]
    by_z = sorted((z, i) for i, z in ct)
    keys = [z for z, _ in by_z]
    matches = [{"zval_id": d["id"], "lps_z_mm": d["val"],
                "ct_record_ids": sorted(i for _, i in by_z[bisect_left(keys, d["val"] - atol):bisect_right(keys, d["val"] + atol)])}
               for d in sorted(db_z, key=lambda d: d["id"])]
    covered = {i for m in matches for i in m["ct_record_ids"]}
    sorted_db, sorted_ct = sorted(d["val"] for d in db_z), sorted(z for _, z in ct)
    multiset_equal = len(rows) == len(ct) == len(sorted_db) and all(abs(a - b) <= atol for a, b in zip(sorted_ct, sorted_db))
    return {"coordinate": "LPS_z_not_normal_projection", "ct_count": len(rows), "db_zval_count": len(db_z),
            "count_delta_ct_minus_db": len(rows) - len(db_z), "sorted_multiset_within_tolerance": multiset_equal,
            "db_zval_candidates": matches, "ct_without_db_zval": sorted(i for i, _ in ct if i not in covered),
            "db_zvals_without_ct": [m["zval_id"] for m in matches if not m["ct_record_ids"]],
            "ambiguous_db_zvals": [m["zval_id"] for m in matches if len(m["ct_record_ids"]) > 1]}


class _PydicomWarnings(logging.Handler):
    """Capture logger-only warnings as well as warnings.warn, without changing levels."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append({"level": record.levelname, "message": record.getMessage()})


@contextmanager
def _pydicom_warnings():
    logger, handler = logging.getLogger("pydicom"), _PydicomWarnings()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        handler.close()


def _inspect_file(row, tol, raw_root):
    # Recheck immediately before I/O and use the contained resolved path for all reads.
    path = _source_path(row["path"], raw_root)
    record = {"record_id": row["record_id"], "source_path": row["path"], "resolved_source_path": str(path),
              "inventory_sop_uid": row["sop_uid"], "sha256_before": None, "sha256_after": None,
              "source_opened": False,
              "decode_status": "failed", "padding_status": "unavailable", "hu_status": "unavailable",
              "metadata": {}, "findings": []}
    raw = valid = hu = None
    with warnings.catch_warnings(record=True) as caught, _pydicom_warnings() as log_warnings:
        warnings.simplefilter("always")
        try:
            record["sha256_before"] = sha256(path)
            ds = pydicom.dcmread(path)
            record["source_opened"] = True
            record["metadata"] = {name: _json_value(getattr(ds, name, None)) for name in METADATA}
            record["metadata"]["TransferSyntaxUID"] = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
            if row.get("sha256") and row["sha256"] != record["sha256_before"]:
                record["findings"].append("inventory_hash_mismatch")
            for key, tag in zip((*IDENTITY, "sop_uid"), ("PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")):
                if row[key] != str(getattr(ds, tag, "")):
                    record["findings"].append(f"inventory_{key}_mismatch")
            for key, tag, n, threshold in (("position", "ImagePositionPatient", 3, "repeated_position_atol_mm"),
                                           ("orientation", "ImageOrientationPatient", 6, "orientation_atol"),
                                           ("pixel_spacing", "PixelSpacing", 2, "in_plane_spacing_atol_mm")):
                actual, saved = _vector(getattr(ds, tag, None), n), _vector(row.get(key), n)
                if actual is None or saved is None or not np.allclose(actual, saved, atol=tol[threshold], rtol=0):
                    record["findings"].append(f"inventory_{key}_mismatch")
            if list(getattr(ds, "ImageType", [])) != row.get("image_type", []):
                record["findings"].append("inventory_image_type_mismatch")
            if "LOCALIZER" in getattr(ds, "ImageType", []):
                record["findings"].append("localizer_unsupported")
            if ds.Modality != "CT" or int(getattr(ds, "NumberOfFrames", 1)) != 1 or ds.SamplesPerPixel != 1 or ds.PhotometricInterpretation != "MONOCHROME2":
                raise ValueError("Unsupported pixel geometry/encoding; single-frame MONOCHROME2 CT required")
            decoded = ds.pixel_array.copy()
            if decoded.ndim != 2 or decoded.shape != (row["rows"], row["columns"]):
                raise ValueError("Pixel dimensions disagree with inventory")
            raw = decoded
            record["decode_status"] = "decoded"
            record["raw_dtype"] = str(raw.dtype)
            mask = np.ones(raw.shape, dtype=bool)
            if hasattr(ds, "PixelPaddingRangeLimit") and not hasattr(ds, "PixelPaddingValue"):
                raise ValueError("Padding range limit without padding value")
            if hasattr(ds, "PixelPaddingValue"):
                lo, hi = float(ds.PixelPaddingValue), float(getattr(ds, "PixelPaddingRangeLimit", ds.PixelPaddingValue))
                if not np.isfinite([lo, hi]).all():
                    raise ValueError("Nonfinite pixel padding")
                mask &= ~((raw >= min(lo, hi)) & (raw <= max(lo, hi)))
            valid = mask
            record["padding_status"] = "computed"
            record["valid_pixel_count"] = int(valid.sum())
            if hasattr(ds, "ModalityLUTSequence") or hasattr(ds, "RealWorldValueMappingSequence"):
                raise ValueError("LUT/real-world mapping unsupported for HU comparison")
            slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
            if not np.isfinite([slope, intercept]).all() or slope <= 0 or str(getattr(ds, "RescaleType", "HU")).upper() != "HU":
                raise ValueError("Missing/unsupported HU rescaling")
            converted = raw.astype(np.float64) * slope + intercept
            if not np.isfinite(converted).all():
                raise ValueError("Nonfinite rescaled pixels")
            hu = converted
            record["hu_status"] = "computed_linear_rescale"
        except Exception as error:
            record["error"] = _error(error)
        finally:
            try:
                record["sha256_after"] = sha256(path)
            except OSError as error:
                record["hash_error"] = _error(error)
        record["warnings"] = [str(w.message) for w in caught]
        record["pydicom_log_warnings"] = log_warnings
    record["bytes_unchanged"] = record["sha256_before"] is not None and record["sha256_before"] == record["sha256_after"]
    record["findings"].sort()
    return record, raw, valid, hu


def _content(group, tol, raw_root):
    inspected = [_inspect_file(r, tol, raw_root) for r in group]
    pairs = []
    for a, b in itertools.combinations(inspected, 2):
        ma, ra, va, ha = a
        mb, rb, vb, hb = b
        shape_ok = ra is not None and rb is not None and ra.shape == rb.shape
        padding_ok = shape_ok and va is not None and vb is not None
        shared = va & vb if padding_ok else None
        hu_ok = padding_ok and ha is not None and hb is not None and bool(shared.any())
        pairs.append({"record_ids": [ma["record_id"], mb["record_id"]],
                      "raw_pixels_equal": bool(np.array_equal(ra, rb)) if ra is not None and rb is not None else None,
                      "padding_validity_equal": bool(np.array_equal(va, vb)) if padding_ok else None,
                      "common_valid_pixel_count": int(shared.sum()) if padding_ok else None,
                      "hu_equal_on_common_valid_pixels": bool(np.allclose(ha[shared], hb[shared], atol=tol["hu_atol"], rtol=0)) if hu_ok else None,
                      "max_abs_hu_difference_on_common_valid_pixels": float(np.max(np.abs(ha[shared] - hb[shared]))) if hu_ok else None,
                      "hu_comparison_status": "compared" if hu_ok else "unavailable_or_no_common_valid_pixels"})
    members = [v[0] for v in inspected]
    blockers = {m["record_id"]: set(m["findings"]) for m in members}
    actual_rows = [{**r, **{key: m["metadata"].get(tag) for key, tag in (
        ("position", "ImagePositionPatient"), ("orientation", "ImageOrientationPatient"),
        ("pixel_spacing", "PixelSpacing"), ("rows", "Rows"), ("columns", "Columns"),
        ("image_type", "ImageType"))}} for r, m in zip(group, members)]
    for row in actual_rows:
        row["image_type"] = row["image_type"] or []
    actual_geometry = _geometry(actual_rows, tol)
    for m in members:
        reasons = blockers[m["record_id"]]
        m["inspected_group_geometry_findings"] = actual_geometry["findings"]
        reasons.update("inspected_geometry_" + reason for reason in actual_geometry["findings"])
        if not m["source_opened"] or "error" in m or m["decode_status"] != "decoded":
            reasons.add("source_inspection_incomplete")
        if "hash_error" in m or not m["bytes_unchanged"]:
            reasons.add("source_bytes_changed_or_hash_unavailable")
        if m["warnings"] or m["pydicom_log_warnings"]:
            reasons.add("unresolved_read_or_decode_warning")
    for pair in pairs:
        for field in ("raw_pixels_equal", "padding_validity_equal", "hu_equal_on_common_valid_pixels"):
            if pair[field] is not True:
                reason = field + ("_conflict" if pair[field] is False else "_unavailable")
                for record_id in pair["record_ids"]:
                    blockers[record_id].add(reason)
    span = np.ptp(np.array([r["position"] for r in group], dtype=float), axis=0)
    if span.max() > tol["repeated_position_atol_mm"]:
        for reasons in blockers.values():
            reasons.add("repeated_position_tolerance_chain_ambiguous")
    # A comparison involving an uncertain peer cannot establish a clean group.
    # In particular, do not implicitly prefer the warning-free or decodable member.
    if any(blockers.values()):
        for reasons in blockers.values():
            if not reasons:
                reasons.add("repeated_position_peer_evidence_unresolved")
    for member in members:
        member["inspection_blockers"] = sorted(blockers[member["record_id"]])
    return members, pairs


def _link_contours(contours, rows, scan_count, geometry, global_sops, raw_root, tol, inspections):
    filename, sop = defaultdict(list), defaultdict(list)
    for r in rows:
        path = Path(r["path"])
        names = {path.name, str(path)}
        try:
            names.add(path.relative_to(raw_root).as_posix())
        except ValueError:
            pass
        for name in names:
            filename[name].append(r["record_id"])
        sop[r["sop_uid"]].append(r["record_id"])
    zrows = sorted((float(p[2]), r["record_id"]) for r in rows if (p := _vector(r.get("position"), 3)) is not None)
    zs = [z for z, _ in zrows]
    by_id = {r["record_id"]: r for r in rows}
    for c in sorted(contours, key=lambda c: c["id"]):
        name, z = c["dicom_file_name"], c["image_z_position"]
        fn = sorted(filename.get(name, [])) if isinstance(name, str) else []
        uid = sorted(sop.get(name, [])) if isinstance(name, str) else []
        exact = sorted(set(fn + uid))
        position = sorted(i for _, i in zrows[bisect_left(zs, z - tol["z_atol_mm"]):bisect_right(zs, z + tol["z_atol_mm"])]) if _finite(z) else []
        contradictory = sorted(set(exact) - set(position)) if _finite(z) else []
        # Check the actual inspected z directly as well: tolerance errors must not accumulate.
        if _finite(z):
            for record_id in exact:
                actual = _vector(inspections.get(record_id, {}).get("metadata", {}).get("ImagePositionPatient"), 3)
                if actual is not None and abs(float(actual[2]) - z) > tol["z_atol_mm"]:
                    contradictory.append(record_id)
            contradictory = sorted(set(contradictory))
        ambiguous = len(exact) > 1 or scan_count != 1 or any(global_sops[by_id[i]["sop_uid"]] != 1 for i in exact)
        status = ("filename_position_contradiction" if contradictory else
                  "ambiguous_exact_link" if exact and ambiguous else
                  "exact_link_geometry_unresolved" if exact and (not _finite(z) or geometry["status"] != "diagnostic_geometry_consistent") else
                  "exact_link_inspection_unresolved" if any(inspections[i]["inspection_blockers"] for i in exact if i in inspections) else
                  "verified_inspected_source_link" if exact and exact[0] in inspections else
                  "exact_inventory_record_link_source_not_rechecked" if exact else
                  "z_only_inventory_link" if position else "missing_link")
        yield {"contour_id": c["id"], "annotation_id": c["annotation_id"], "scan_id": c["scan_id"],
               "identity": {k: c[k] for k in IDENTITY},
               "original": {k: c[k] for k in ("dicom_file_name", "image_z_position", "inclusion")},
               "exact_filename_record_ids": fn, "literal_sop_uid_record_ids": uid,
               "z_candidate_record_ids": position, "all_candidate_record_ids": sorted(set(exact + position)),
               "position_contradiction_record_ids": contradictory,
               "identity_ambiguous": ambiguous, "status": status,
               "exact_inventory_record_id": exact[0] if status in ("exact_inventory_record_link_source_not_rechecked", "verified_inspected_source_link") else None,
               "verified_inspected_record_id": exact[0] if status == "verified_inspected_source_link" else None,
               "exact_candidate_inspections": [{"record_id": i, "source_opened": inspections[i]["source_opened"],
                                                "inspection_blockers": inspections[i]["inspection_blockers"]}
                                               for i in exact if i in inspections],
               "reader_identity": "unresolved", "inclusion_valid": c["inclusion"] in (0, 1)}


def _reference_summary(group, links):
    ids = {r["record_id"] for r in group}
    exact_members, verified_members, z_members, details = set(), set(), set(), []
    for link in links:
        exact = ids & set(link["exact_filename_record_ids"] + link["literal_sop_uid_record_ids"])
        z = ids & set(link["z_candidate_record_ids"])
        verified = {link["verified_inspected_record_id"]} & ids
        if exact or z:
            details.append({"contour_id": link["contour_id"], "annotation_id": link["annotation_id"],
                            "inclusion": link["original"]["inclusion"], "status": link["status"],
                            "exact_candidate_record_ids": sorted(exact), "z_candidate_record_ids": sorted(z),
                            "verified_inspected_record_ids": sorted(verified)})
        exact_members.update(exact)
        verified_members.update(verified)
        z_members.update(z)

    def cardinality(items):
        return "neither" if not items else "one" if len(items) == 1 else "multiple"

    return {"exact_candidate_members": sorted(exact_members), "exact_candidate_cardinality": cardinality(exact_members),
            "verified_inspected_members": sorted(verified_members), "verified_inspected_member_cardinality": cardinality(verified_members),
            "z_candidate_members": sorted(z_members), "contours": details,
            "meaning": "references in this database only; neither does not mean annotation absence in XML"}


def _code_provenance():
    root = Path(__file__).resolve().parents[2]
    result = {"module_sha256": sha256(Path(__file__)), "revision": None, "dirty": None}
    try:
        result["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        result.update(dirty=bool(status), worktree_status=status.splitlines())
        result["code_files_sha256"] = {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src" / "lidc3d").glob("*.py"))}
    except (OSError, subprocess.CalledProcessError) as error:
        result["git_error"] = _error(error)
    return result


def _packages():
    packages = {}
    for name in ("numpy", "pydicom", "pylidc", "PyYAML", "torch", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def _run(inventory, rows, raw_root, database, config, handles, summary, provenance):
    tol = config["tolerances"]
    with readonly_database(database) as conn:
        provenance["sqlite"] = {"uri_mode": "ro", "query_only": conn.execute("PRAGMA query_only").fetchone()[0],
                                "version": sqlite3.sqlite_version, "transaction": "read snapshot", "sidecar_policy": "refuse"}
        scans, annotations, zvals, contours = _database(conn, provenance)
    summary["counts"].update(ct_records=len(rows), database_scans=len(scans), database_annotations=len(annotations),
                             database_contours=len(contours), database_zvals=len(zvals))
    ct_by, db_by, z_by, contour_by = (defaultdict(list) for _ in range(4))
    annotation_counts = Counter(a["scan_id"] for a in annotations)
    for r in rows:
        ct_by[_identity(r)].append(r)
    for s in scans:
        db_by[_identity(s)].append(s)
    for z in zvals:
        z_by[z["scan_id"]].append(z)
    for c in contours:
        contour_by[_identity(c)].append(c)
    sop_records = defaultdict(list)
    for r in rows:
        sop_records[r["sop_uid"]].append(r["record_id"])
    sops = {s: len(ids) for s, ids in sop_records.items()}
    summary["duplicate_sop_ids"] = [{"sop_uid": s, "record_ids": sop_records[s]}
                                    for s in sorted(s for s, n in sops.items() if n > 1)]
    summary["identity_differences"] = {}
    for name, size in (("patients", 1), ("patient_studies", 2), ("patient_study_series", 3)):
        ct_keys, db_keys = {k[:size] for k in ct_by}, {k[:size] for k in db_by}
        summary["identity_differences"][name] = {"ct_only": sorted(ct_keys - db_keys), "db_only": sorted(db_keys - ct_keys)}
    summary["duplicate_database_identities"] = [{"identity": dict(zip(IDENTITY, k)), "scan_ids": [s["id"] for s in db_by[k]]}
                                              for k in sorted(db_by) if len(db_by[k]) > 1]
    summary["inventory_errors"] = sorted(inventory.get("errors", []), key=lambda r: json.dumps(r, sort_keys=True))
    linkage_counts = Counter()
    inspection_blocker_counts = Counter()
    for key in sorted(set(ct_by) | set(db_by)):
        series_rows, db_scans = ct_by[key], db_by[key]
        geometry = _geometry(series_rows, tol)
        groups = _position_groups(series_rows, tol["repeated_position_atol_mm"])
        content = [_content(group, tol, raw_root) for group in groups]
        inspections = {m["record_id"]: m for members, _ in content for m in members}
        links = list(_link_contours(contour_by[key], series_rows, len(db_scans), geometry, sops, raw_root, tol, inspections))
        for link in links:
            _emit(handles["contour_linkage"], link)
            linkage_counts[link["status"]] += 1
            if not link["inclusion_valid"]:
                summary["operational_errors"].append({"code": "invalid_contour_inclusion", "contour_id": link["contour_id"]})
        group_ids = []
        for group, (members, pairs) in zip(groups, content):
            group_id = f"position-{group[0]['record_id']:09d}"
            group_ids.append(group_id)
            positions = np.array([r["position"] for r in group], dtype=float)
            _emit(handles["duplicate_positions"], {
                "group_id": group_id, "identity": dict(zip(IDENTITY, key)), "members": members, "pairwise": pairs,
                "position_span_lps_mm": np.ptp(positions, axis=0).tolist(),
                "tolerance_chain_ambiguous": bool(np.ptp(positions, axis=0).max() > tol["repeated_position_atol_mm"]),
                "annotation_references": _reference_summary(group, links), "resolution": "unresolved_no_slice_selected"})
            summary["counts"]["repeated_position_groups"] += 1
            summary["counts"]["inspected_dicom_records"] += len(members)
            for member in members:
                inspection_blocker_counts.update(member["inspection_blockers"])
                if "error" in member or "hash_error" in member or not member["bytes_unchanged"] or member["findings"]:
                    summary["operational_errors"].append({"code": "dicom_inspection_failed_or_inventory_changed",
                                                          "record_id": member["record_id"], "group_id": group_id})
        _emit(handles["series_audit"], {
            "identity": dict(zip(IDENTITY, key)), "ct_count": len(series_rows), "database_scan_ids": [s["id"] for s in db_scans],
            "identity_status": "ct_only" if not db_scans else "db_only" if not series_rows else "duplicate_database_identity" if len(db_scans) > 1 else "matched",
            "geometry": geometry, "z_comparisons": [{"scan_id": s["id"], **_z_comparison(series_rows, z_by[s["id"]], tol["z_atol_mm"])} for s in db_scans],
            "duplicate_sop_ids": sorted({r["sop_uid"] for r in series_rows if sops[r["sop_uid"]] > 1}),
            "repeated_position_group_ids": group_ids,
            "annotation_count": sum(annotation_counts[s["id"]] for s in db_scans),
            "contour_linkage_counts": dict(sorted(Counter(link["status"] for link in links).items()))})
    summary["counts"]["series_union"] = len(set(ct_by) | set(db_by))
    summary["contour_linkage_counts"] = dict(sorted(linkage_counts.items()))
    summary["inspection_blocker_counts"] = dict(sorted(inspection_blocker_counts.items()))


def audit_source(inventory_path, annotation_db, config_path, output, invocation=None):
    """Return summary; failures after safe output creation leave five failure artifacts.

    Preflight failures (unsafe/existing output, unreadable config/inventory) raise
    without creating outputs. CLI maps all operational failures to exit status 1.
    """
    inventory_path, database, config_path = (Path(p).resolve() for p in (inventory_path, annotation_db, config_path))
    output = Path(output).absolute()
    before = {"inventory": sha256(inventory_path), "database": sha256(database), "config": sha256(config_path)}
    config = _load_config(config_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    raw_root, rows = _prepare_inventory(inventory)
    protected = _protect_output(output, raw_root, rows, inventory_path, database, config_path, config, inventory["source_root"])
    provenance = {"schema_version": SCHEMA_VERSION, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "invocation": list(invocation) if invocation is not None else list(sys.argv), "cwd": str(Path.cwd()),
                  "configuration": config, "code": _code_provenance(), "python": sys.version,
                  "platform": platform.platform(), "packages": _packages(), "protected_roots": protected,
                  "inputs": {name: {"path": str(path), "sha256_before": before[name]} for name, path in
                             (("inventory", inventory_path), ("database", database), ("config", config_path))},
                  "dicom_hash_scope": "only inspected repeated-position members; entire CT archive NOT cryptographically verified",
                  "inventory_hash_scope": "inventory byte immutability only; not cryptographic verification of referenced DICOM files",
                  "source_boundary_policy": "absolute paths contained by resolved source_root; no parent traversal; rechecked before inspection",
                  "warning_policy": "all captured read/decode warnings block inspected-source verification; no harmless exceptions",
                  "installed_pylidc_source_inspected": False, "pylidc_imported_or_loader_executed": False}
    summary = {"schema_version": SCHEMA_VERSION, "audit_status": "failed", "source_approved": False,
               "training_released": False, "resolution": "diagnostic_only_requires_separate_review",
               "coverage": {"reader_identity": "unresolved_not_inferred_from_annotation_ids",
                            "ordinary_dicom_source_evidence": "inventory_only_source_not_rechecked",
                            "small_nodule_annotations": "unresolved_not_verified_against_XML",
                            "non_nodule_annotations": "unresolved_not_verified_against_XML",
                            "annotation_counts_are_distinct_nodules": False, "negative_labels_created": False,
                            "xml_reference_standard_replaced": False},
               "counts": {"repeated_position_groups": 0, "inspected_dicom_records": 0}, "operational_errors": []}
    output.mkdir(parents=True, exist_ok=False)
    handles = {}
    try:
        for name in ("series_audit", "duplicate_positions", "contour_linkage"):
            handles[name] = (output / f"{name}.jsonl").open("x", encoding="utf-8", newline="\n")
        _run(inventory, rows, raw_root, database, config, handles, summary, provenance)
    except Exception as error:
        summary["operational_errors"].append(_error(error))
    finally:
        for handle in handles.values():
            handle.close()
    for name, info in provenance["inputs"].items():
        try:
            info["sha256_after"] = sha256(info["path"])
            info["bytes_unchanged"] = info["sha256_after"] == before[name]
        except OSError as error:
            info.update(sha256_after=None, bytes_unchanged=False, verification_error=_error(error))
        if not info["bytes_unchanged"]:
            summary["operational_errors"].append({"code": "input_changed_or_unreadable", "input": name})
    try:
        _no_sidecars(database)
    except ValueError as error:
        summary["operational_errors"].append(_error(error))
    if not summary["operational_errors"]:
        summary["audit_status"] = "completed"
    provenance["audit_status"] = summary["audit_status"]
    provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["artifact_sha256"] = {f"{name}.jsonl": sha256(output / f"{name}.jsonl") for name in handles}
    write_json(output / "provenance.json", _json_value(provenance))
    # The summary is the final completion marker; a partial artifact set is not a run.
    write_json(output / "audit_summary.json", _json_value(summary))
    return summary


def audit_command(args):
    """At most four terminal lines, including operational failures."""
    try:
        result = audit_source(args.inventory, args.annotation_db, args.config, args.out)
    except Exception as error:
        print("Audit failed: " + str(error).replace("\n", " ").replace("\r", " "))
        print("Source approved: no; training released: no.")
        raise SystemExit(1) from None
    print(f"Audit {result['audit_status']}; source approved: no; training released: no.")
    print(f"Repeated-position groups: {result['counts']['repeated_position_groups']}; inspected DICOM records: {result['counts']['inspected_dicom_records']}.")
    print(f"Operational failures: {len(result['operational_errors'])}; details: {Path(args.out).resolve()}".replace("\n", " ").replace("\r", " "))
    if result["audit_status"] != "completed":
        raise SystemExit(1)
