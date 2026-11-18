"""Strict single-frame CT reader. Unsupported geometry fails explicitly."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pydicom
from .geometry import Volume
from .provenance import sha256


def inventory(root, hash_files=False):
    """Read headers without decoding pixels; no source files are modified."""
    rows, xml, errors = [], [], []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".xml":
            xml.append({"path": str(path.resolve()), "sha256": sha256(path)})
            continue
        if path.suffix.lower() not in ("", ".dcm", ".dicom", ".ima"):
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if getattr(ds, "Modality", None) != "CT":
                continue
            row = {"path": str(path.resolve()), "patient_id": str(ds.PatientID),
                "study_uid": str(ds.StudyInstanceUID), "series_uid": str(ds.SeriesInstanceUID),
                "sop_uid": str(ds.SOPInstanceUID), "rows": int(ds.Rows), "columns": int(ds.Columns),
                "position": list(map(float, ds.ImagePositionPatient)),
                "orientation": list(map(float, ds.ImageOrientationPatient)),
                "pixel_spacing": list(map(float, ds.PixelSpacing)),
                "image_type": list(getattr(ds, "ImageType", [])),
                "transfer_syntax": str(ds.file_meta.TransferSyntaxUID)}
            if hash_files:
                row["sha256"] = sha256(path)
            rows.append(row)
        except Exception as error:
            errors.append({"path": str(path.resolve()), "error": type(error).__name__, "reason": str(error)})
    return {"source_root": str(Path(root).resolve()), "slices": rows, "xml": xml, "errors": errors,
            "status": "HEADER_INVENTORY_ONLY_NOT_COHORT_FREEZE"}


def _regularize_slice_axis(hu, valid, projected, requested_spacing_mm):
    """Interpolate a nonuniform slice axis onto a bounded regular grid.

    The output includes both end slice centers and never uses a step larger
    than requested_spacing_mm. A linearly interpolated voxel is valid only
    when both source samples are valid.
    """
    if requested_spacing_mm <= 0 or not np.isfinite(requested_spacing_mm):
        raise ValueError("Coordinate-aware spacing must be finite and positive")
    extent = float(projected[-1] - projected[0])
    intervals = max(1, int(np.ceil(extent / requested_spacing_mm)))
    target = np.linspace(projected[0], projected[-1], intervals + 1)
    upper = np.searchsorted(projected, target, side="left")
    upper = np.clip(upper, 0, len(projected) - 1)
    lower = np.maximum(upper - 1, 0)
    exact = np.isclose(target, projected[upper], atol=1e-9, rtol=0)
    lower[exact] = upper[exact]
    denominator = projected[upper] - projected[lower]
    weight = np.zeros_like(target)
    distinct = upper != lower
    weight[distinct] = ((target[distinct] - projected[lower[distinct]]) /
                        denominator[distinct])
    output_hu = np.empty((len(target), *hu.shape[1:]), np.float32)
    output_valid = np.empty((len(target), *valid.shape[1:]), bool)
    for index, (lo, hi, alpha) in enumerate(zip(lower, upper, weight)):
        if lo == hi:
            output_hu[index] = hu[lo]
            output_valid[index] = valid[lo]
        else:
            output_valid[index] = valid[lo] & valid[hi]
            output_hu[index] = (1-alpha)*hu[lo] + alpha*hu[hi]
            output_hu[index][~output_valid[index]] = -1000
    return output_hu, output_valid, target, extent / intervals


def load_series(paths, config, *, coordinate_aware_spacing_mm=None,
                max_interpolation_gap_mm=None):
    datasets = [pydicom.dcmread(p) for p in paths]
    if len(datasets) < 2:
        raise ValueError("At least two CT slices required")
    first = datasets[0]
    identity = (str(first.PatientID), str(first.StudyInstanceUID), str(first.SeriesInstanceUID))
    orient = np.array(first.ImageOrientationPatient, float)
    atol = config["orientation_atol"]
    if orient.shape != (6,) or not np.isfinite(orient).all():
        raise ValueError("Invalid orientation")
    col_direction, row_direction = orient[:3], orient[3:]
    if not np.allclose([np.linalg.norm(col_direction), np.linalg.norm(row_direction),
                        col_direction @ row_direction], [1, 1, 0], atol=atol, rtol=0):
        raise ValueError("Nonorthonormal orientation")
    normal = np.cross(col_direction, row_direction)
    spacing = np.array(first.PixelSpacing, float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Invalid in-plane spacing")
    sops = set()
    for ds in datasets:
        if (str(ds.PatientID), str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID)) != identity:
            raise ValueError("Mixed patient/study/series")
        if ds.Modality != "CT" or int(getattr(ds, "NumberOfFrames", 1)) != 1:
            raise ValueError("Only single-frame CT supported")
        if "LOCALIZER" in getattr(ds, "ImageType", []):
            raise ValueError("Localizer")
        if str(ds.SOPInstanceUID) in sops:
            raise ValueError("Duplicate SOP UID")
        sops.add(str(ds.SOPInstanceUID))
        if (ds.Rows, ds.Columns) != (first.Rows, first.Columns):
            raise ValueError("Variable dimensions")
        if not np.allclose(ds.ImageOrientationPatient, orient, atol=atol, rtol=0):
            raise ValueError("Variable orientation")
        if not np.allclose(ds.PixelSpacing, spacing, atol=1e-6, rtol=0):
            raise ValueError("Variable in-plane spacing")
        if getattr(ds, "PhotometricInterpretation", None) != "MONOCHROME2":
            raise ValueError("Unsupported photometric interpretation")
    datasets.sort(key=lambda ds: float(np.array(ds.ImagePositionPatient) @ normal))
    positions = np.array([ds.ImagePositionPatient for ds in datasets], float)
    if not np.isfinite(positions).all():
        raise ValueError("Nonfinite positions")
    projected = positions @ normal
    diffs = np.diff(projected)
    dz = float(np.median(diffs))
    if dz <= 0 or (diffs <= config["position_atol_mm"]).any():
        raise ValueError("Duplicate/reversed slice positions")
    uniform = np.allclose(diffs, dz, rtol=config["spacing_rtol"],
                          atol=config["spacing_atol_mm"])
    if not uniform and coordinate_aware_spacing_mm is None:
        raise ValueError("Irregular slice spacing or missing slices")
    expected = positions[0] + (projected - projected[0])[:, None] * normal
    if np.max(np.linalg.norm(positions - expected, axis=1)) > config["position_atol_mm"]:
        raise ValueError("Gantry tilt/in-plane drift unsupported")
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack([normal*dz, row_direction*spacing[0], col_direction*spacing[1]])
    affine[:3, 3] = positions[0]
    hu, valid = [], []
    digest = hashlib.sha256()
    source_hashes = {}
    for ds in datasets:
        raw = ds.pixel_array
        digest.update(np.asarray(raw, dtype="<i4").tobytes())
        slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
        if not np.isfinite([slope, intercept]).all() or slope <= 0:
            raise ValueError("Invalid HU rescale")
        digest.update(np.asarray([slope, intercept], dtype="<f8").tobytes())
        good = np.ones(raw.shape, bool)
        if hasattr(ds, "PixelPaddingValue"):
            low = float(ds.PixelPaddingValue)
            high = float(getattr(ds, "PixelPaddingRangeLimit", low))
            good &= ~((raw >= min(low, high)) & (raw <= max(low, high)))
        converted = raw.astype(np.float32)*slope + intercept
        good &= np.isfinite(converted)
        converted[~good] = -1000
        hu.append(converted)
        valid.append(good)
        source_hashes[str(Path(ds.filename).resolve())] = sha256(ds.filename)
    hu, valid = np.stack(hu), np.stack(valid)
    source_positions = positions.copy()
    source_diffs = diffs.copy()
    if not uniform:
        if max_interpolation_gap_mm is None or max_interpolation_gap_mm <= 0:
            raise ValueError("Positive max interpolation gap required for irregular spacing")
        if float(diffs.max()) > max_interpolation_gap_mm:
            raise ValueError("Irregular slice gap exceeds approved interpolation limit")
        hu, valid, target, dz = _regularize_slice_axis(
            hu, valid, projected, coordinate_aware_spacing_mm)
        positions = positions[0] + (target - target[0])[:, None] * normal
        affine[:3, :3] = np.column_stack(
            [normal*dz, row_direction*spacing[0], col_direction*spacing[1]])
    metadata = {"patient_id": identity[0], "study_uid": identity[1], "series_uid": identity[2],
        "sop_uids": [str(ds.SOPInstanceUID) for ds in datasets],
        "slice_positions": positions.tolist(), "native_spacing_zyx": [dz, *spacing.tolist()],
        "pixel_fingerprint": digest.hexdigest(), "source_hashes": source_hashes,
        "coordinate_aware_resampling": not uniform,
        "source_slice_positions": source_positions.tolist(),
        "source_projected_positions_mm": projected.tolist(),
        "source_slice_gaps_mm": source_diffs.tolist()}
    return Volume(hu, valid, affine, metadata)
