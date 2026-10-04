"""File-oriented preparation with separate cohort, candidate and prediction stages."""
from __future__ import annotations
import copy
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import importlib.metadata
from itertools import chain
import json
import os
from pathlib import Path
import re
import shutil
import time
import numpy as np
from .ingest import load_series
from .geometry import resample_lps, normalize, to_world
from .references import xml_root, parse_xml, make_references
from .splits import patient_split
from .candidates import lung_mask, generate
from .evaluation import DetectionCurves
from .provenance import write_json, write_jsonl, read_jsonl, read_json, sha256, canonical_hash, safe_output


def _iter_jsonl(path):
    """Read an ordered JSONL artifact without retaining its rows."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class PreparedProposalPool:
    """Reiterable disk-backed pool; per-scan reads use verified checkpoints.

    Callers must verify the preparation before using per_scan for selection.
    No proposal dictionaries are cached across scans or iterations.
    """
    def __init__(self, output, scans):
        self.output = Path(output)
        self.scan_ids = tuple(scan.get("scan_id") for scan in scans)

    def __iter__(self):
        return _iter_jsonl(self.output/"proposal_pool.jsonl")

    def __len__(self):
        return sum(read_json(self.output/uid/"checkpoint.json")["proposal_count"]
                   for uid in self.scan_ids)

    def per_scan(self, scan_id):
        if scan_id not in self.scan_ids:
            raise ValueError("Scan outside prepared proposal pool")
        return _iter_jsonl(self.output/scan_id/"proposals.jsonl")


def _checkpoint_proposals(output, scans):
    return chain.from_iterable(
        _iter_jsonl(output/scan["scan_id"]/"proposals.jsonl") for scan in scans)


def _load_configured_series(paths, config):
    """Load a series under one explicit, frozen geometry policy."""
    geometry = config["geometry"]
    spacing = geometry.get("coordinate_aware_spacing_mm")
    max_gap = geometry.get("max_interpolation_gap_mm")
    if (spacing is None) != (max_gap is None):
        raise ValueError(
            "Coordinate-aware spacing and maximum interpolation gap must be configured together")
    return load_series(
        paths, geometry, coordinate_aware_spacing_mm=spacing,
        max_interpolation_gap_mm=max_gap)


def build_cohort(inventory, config, output):
    """Create draft references and proposed patient partitions, not a release.

    Integrity exclusions and alternative reconstructions are reason-coded.
    A reference-review artifact is mandatory before development freezing.
    """
    output = Path(output).resolve()
    safe_output(output, inventory["source_root"])
    output.mkdir(parents=True, exist_ok=False)
    by_series, xmls = defaultdict(list), defaultdict(list)
    exceptions = []
    for row in inventory["slices"]:
        by_series[row["series_uid"]].append(row)
    for item in inventory["xml"]:
        try:
            root = xml_root(item["path"])
            series = root.findtext(".//SeriesInstanceUid") or root.findtext(".//SeriesInstanceUID")
            if not series:
                raise ValueError("Missing XML series UID")
            xmls[series].append(item)
        except Exception as error:
            exceptions.append({"xml": item["path"], "stage": "source_xml", "reason": str(error)})
    examinations = defaultdict(list)
    for uid, slices in by_series.items():
        first = slices[0]
        # Multiple distinct XML versions must be resolved rather than concatenated.
        xml_options = {x["sha256"]: x for x in xmls.get(uid, [])}
        if len(xml_options) != 1:
            exceptions.append({"series_uid": uid, "stage": "source_linkage", "reason": "missing_or_multiple_distinct_xml"})
            continue
        if any("LOCALIZER" in r["image_type"] for r in slices):
            exceptions.append({"series_uid": uid, "stage": "source_geometry", "reason": "localizer"})
            continue
        orient = np.array(first["orientation"])
        normal = np.cross(orient[:3], orient[3:])
        positions = np.sort(np.array([r["position"] for r in slices]) @ normal)
        if len(positions) < 2:
            exceptions.append({"series_uid": uid, "stage": "source_geometry", "reason": "fewer_than_two_slices"})
            continue
        rank = (float(np.median(np.diff(positions))), -float(np.ptp(positions)), uid)
        examinations[(first["patient_id"], first["study_uid"])].append((rank, slices, next(iter(xml_options.values()))))
    scans = []
    for (patient, study), options in sorted(examinations.items()):
        chosen = False
        for rank, slices, xml in sorted(options, key=lambda row: row[0]):
            uid = slices[0]["series_uid"]
            if chosen:
                exceptions.append({"series_uid": uid, "stage": "reconstruction_selection", "reason": "lower_rank_alternative"})
                continue
            scan_id = "scan_" + canonical_hash({"patient": patient, "study": study, "series": uid})[:20]
            try:
                volume = _load_configured_series([r["path"] for r in slices], config)
                marks = parse_xml(xml["path"], volume)
                positive, ignored, audit = make_references(marks, volume.affine, output/scan_id, scan_id)
                write_json(output/f"{scan_id}_reference_audit.json", audit)
                record = {"scan_id": scan_id, "patient_id": patient, "study_uid": study, "series_uid": uid,
                    "source_root": inventory["source_root"],
                    "dicom_paths": [r["path"] for r in slices], "xml_path": xml["path"], "xml_sha256": xml["sha256"],
                    "pixel_fingerprint": volume.metadata["pixel_fingerprint"], "native_affine": volume.affine.tolist(),
                    "source_hashes": volume.metadata["source_hashes"], "positive": positive, "ignored": ignored,
                    "reference_audit_path": str((output/f"{scan_id}_reference_audit.json").resolve()),
                    "unresolved_associations": len(audit["ambiguities"])}
                scans.append(record)
                chosen = True
                del volume
            except Exception as error:
                exceptions.append({"series_uid": uid, "stage": "source_or_reference_integrity",
                    "reason": f"{type(error).__name__}: {error}"})
    if not scans:
        write_json(output/"exceptions.json", exceptions)
        raise ValueError("No reconstructable annotation-linked scans; inspect exceptions")
    assignments, split_record = patient_split(scans)
    for scan in scans:
        scan["split"] = assignments[scan["patient_id"]]
    write_jsonl(output/"cohort_development.jsonl", [s for s in scans if s["split"] != "test"])
    write_jsonl(output/"cohort_test_WITHHOLD.jsonl", [s for s in scans if s["split"] == "test"])
    write_json(output/"patient_split.json", split_record)
    write_json(output/"exceptions.json", exceptions)
    write_json(output/"cohort_summary.json", {"status": "DRAFT_NOT_FROZEN",
        "source_inventory_hash": canonical_hash(inventory), "config_hash": canonical_hash(config),
        "scans": len(scans), "patients": len(assignments),
        "unresolved_associations": sum(s["unresolved_associations"] for s in scans),
        "warning": "Counts describe only this build; integrity exclusions and reference review require audit"})


def _prepare_scan(scan, config, scan_dir, final_dir, threshold):
    """Prepare one scan. Durable checkpoint publication is owned by the caller."""
    record = copy.deepcopy(scan)
    record.update({"status": "ok", "failure_reason": None})
    _progress("scan_stage", scan_id=scan["scan_id"], stage="load_native")
    native = _load_configured_series(scan["dicom_paths"], config)
    for key in ("patient_id", "study_uid", "series_uid"):
        if native.metadata[key] != scan[key]:
            raise ValueError(f"Loaded DICOM {key} disagrees with cohort manifest")
    expected_hashes = {str(Path(path).resolve()): value
                       for path, value in scan["source_hashes"].items()}
    if native.metadata["source_hashes"] != expected_hashes:
        raise ValueError("Loaded DICOM source hashes disagree with cohort manifest")
    record["pixel_fingerprint"] = native.metadata["pixel_fingerprint"]
    record["native_affine"] = native.affine.tolist()
    record["source_hashes"] = native.metadata["source_hashes"]
    record["coordinate_aware_resampling"] = native.metadata["coordinate_aware_resampling"]
    _progress("scan_stage", scan_id=scan["scan_id"], stage="resample_canonical")
    volume = resample_lps(native, config["preprocess"]["spacing_mm"])
    del native
    image = normalize(volume.hu, config["preprocess"]["window_hu"])
    image[~volume.valid] = 0
    _progress("scan_stage", scan_id=scan["scan_id"], stage="lung_mask")
    mask, base, mask_status = lung_mask(volume.hu, volume.valid,
        spacing=config["preprocess"]["spacing_mm"], air_hu=config["preprocess"]["mask_air_hu"],
        min_component_mm3=config["preprocess"]["mask_min_component_mm3"],
        dilation_mm=config["preprocess"]["mask_dilation_mm"])
    c = {**config["candidates"], "threshold": threshold}
    _progress("scan_stage", scan_id=scan["scan_id"], stage="candidate_generation")
    _, diagnostics = generate(image, mask, c, config["preprocess"]["spacing_mm"])
    raw_proposals = diagnostics.pop("precap_candidates")
    proposals = [{**candidate, "id": f"{scan['scan_id']}:candidate:{i:06d}",
        "scan_id": scan["scan_id"],
        "center_lps": to_world(candidate["zyx"], volume.affine).tolist()}
        for i, candidate in enumerate(raw_proposals)]
    _progress("scan_stage", scan_id=scan["scan_id"], stage="save_arrays")
    for name, array in [("image", image), ("valid", volume.valid),
                        ("lung_mask", mask), ("base_mask", base)]:
        path = scan_dir/f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        record[f"{name}_path"] = str(final_dir/f"{name}.npy")
    record["affine"] = volume.affine.tolist()
    record["mask_diagnostics"] = mask_status
    record["candidate_diagnostics"] = diagnostics
    write_json(scan_dir/"geometry.json", {"affine": record["affine"],
        "shape": list(image.shape), "metadata": volume.metadata,
        "array_order": "zyx", "patient_coordinates": "LPS_mm"})
    return record, proposals


def _progress(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def _scan_paths(output, scan_id):
    if (not isinstance(scan_id, str) or
            re.fullmatch(r"scan_[A-Za-z0-9][A-Za-z0-9_-]{0,127}", scan_id) is None):
        raise ValueError("Unsafe scan_id")
    final = output/scan_id
    staging = output/f".{scan_id}.incomplete"
    if final.resolve().parent != output or staging.resolve().parent != output:
        raise ValueError("Unsafe scan output path")
    return final, staging


def _source_evidence(scan):
    paths = [str(Path(path).resolve()) for path in scan.get("dicom_paths", [])]
    hashes = {str(Path(path).resolve()): value
              for path, value in scan.get("source_hashes", {}).items()}
    if not paths or len(paths) != len(set(paths)) or set(paths) != set(hashes):
        raise ValueError("source_hashes must exactly cover unique dicom_paths")
    root = Path(scan["source_root"]).resolve()
    for path in map(Path, paths):
        if root != path and root not in path.parents:
            raise ValueError("DICOM path escapes source_root")
        if not path.is_file() or path.is_symlink() or sha256(path) != hashes[str(path)]:
            raise ValueError("Raw source changed since cohort construction")
    return hashes


def _collect_preparation_artifacts(output, scans):
    artifacts = {}
    for scan in scans:
        scan_dir, _ = _scan_paths(output, scan["scan_id"])
        checkpoint_path = scan_dir/"checkpoint.json"
        checkpoint = read_json(checkpoint_path)
        prefix = scan["scan_id"] + "/"
        for name, evidence in checkpoint["files"].items():
            artifacts[prefix + name] = evidence
        artifacts[prefix + "checkpoint.json"] = {
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path)}
    return artifacts


def _verify_preparation_artifacts(output, scans, pool, preparation, config):
    if (preparation.get("schema_version") != 1 or
            preparation.get("status") != "PROPOSALS_NOT_YET_FROZEN" or
            preparation.get("cohort_hash") is None or
            canonical_hash(preparation.get("config")) != canonical_hash(config)):
        raise ValueError("Invalid preparation commit marker")
    aggregate_files = preparation.get("files")
    if set(aggregate_files or {}) != {"scans.jsonl", "proposal_pool.jsonl"}:
        raise ValueError("Preparation aggregate inventory changed")
    for name, expected in aggregate_files.items():
        if sha256(output/name) != expected:
            raise ValueError(f"Preparation aggregate changed: {name}")
    declared = preparation.get("artifacts")
    if not isinstance(declared, dict):
        raise ValueError("Preparation artifact inventory is missing")
    expected_paths = set()
    aggregate = iter(pool)
    sentinel = object()
    for scan in scans:
        _source_evidence(scan)
        scan_dir, _ = _scan_paths(output, scan["scan_id"])
        checkpoint_path = scan_dir/"checkpoint.json"
        relative_checkpoint = f"{scan['scan_id']}/checkpoint.json"
        expected_paths.add(relative_checkpoint)
        evidence = declared.get(relative_checkpoint, {})
        if (not checkpoint_path.is_file() or checkpoint_path.is_symlink() or
                checkpoint_path.stat().st_size != evidence.get("bytes") or
                sha256(checkpoint_path) != evidence.get("sha256")):
            raise ValueError(f"Preparation artifact changed: {relative_checkpoint}")
        checkpoint = read_json(checkpoint_path)
        if (checkpoint.get("preparation_key") != preparation.get("preparation_key") or
                checkpoint.get("scan_id") != scan["scan_id"] or
                checkpoint.get("status") != scan.get("status")):
            raise ValueError("Preparation checkpoint identity changed")
        names = ({"record.json", "proposals.jsonl", "geometry.json", "image.npy",
                  "valid.npy", "lung_mask.npy", "base_mask.npy"}
                 if scan.get("status") == "ok" else {"record.json", "proposals.jsonl"})
        if set(checkpoint.get("files") or {}) != names:
            raise ValueError("Preparation checkpoint file inventory changed")
        for name, checkpoint_evidence in checkpoint["files"].items():
            relative = f"{scan['scan_id']}/{name}"
            expected_paths.add(relative)
            path = output/relative
            evidence = declared.get(relative)
            if (evidence != checkpoint_evidence or not path.is_file() or path.is_symlink() or
                    path.stat().st_size != evidence.get("bytes") or
                    sha256(path) != evidence.get("sha256")):
                raise ValueError(f"Preparation artifact changed: {relative}")
        if read_json(scan_dir/"record.json") != scan:
            raise ValueError("Preparation scan aggregate changed")
        count = 0
        for row in _iter_jsonl(scan_dir/"proposals.jsonl"):
            if (row.get("scan_id") != scan["scan_id"] or
                    row.get("id") != f"{scan['scan_id']}:candidate:{count:06d}" or
                    next(aggregate, sentinel) != row):
                raise ValueError("Preparation artifact aggregation changed")
            count += 1
        if count != checkpoint.get("proposal_count"):
            raise ValueError("Preparation checkpoint proposal count changed")
    if set(declared) != expected_paths or next(aggregate, sentinel) is not sentinel:
        raise ValueError("Preparation artifact aggregation changed")


def _load_scan_checkpoint(scan, preparation_key, scan_dir, record_dir=None):
    record_dir = scan_dir if record_dir is None else record_dir
    checkpoint = read_json(scan_dir/"checkpoint.json")
    if checkpoint.get("schema_version") != 1 or checkpoint.get("kind") != "lidc3d_scan_preparation":
        raise ValueError("Unsupported preparation checkpoint schema")
    if checkpoint.get("scan_id") != scan["scan_id"]:
        raise ValueError("Preparation checkpoint scan ID changed")
    if checkpoint.get("scan_hash") != canonical_hash(scan):
        raise ValueError("Preparation checkpoint cohort record changed")
    if checkpoint.get("preparation_key") != preparation_key:
        raise ValueError("Preparation checkpoint plan changed")
    status = checkpoint.get("status")
    if status not in ("ok", "failed"):
        raise ValueError("Invalid preparation checkpoint status")
    expected_names = ({"record.json", "proposals.jsonl", "geometry.json",
                       "image.npy", "valid.npy", "lung_mask.npy", "base_mask.npy"}
                      if status == "ok" else {"record.json", "proposals.jsonl"})
    expected_files = checkpoint.get("files")
    if not isinstance(expected_files, dict) or set(expected_files) != expected_names:
        raise ValueError("Preparation checkpoint file inventory changed")
    entries = list(scan_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("Preparation checkpoint contains unsafe entries")
    actual_files = {path.name for path in entries if path.name != "checkpoint.json"}
    if actual_files != expected_names:
        raise ValueError("Preparation checkpoint file inventory changed")
    for relative, evidence in expected_files.items():
        path = scan_dir/relative
        if (path.stat().st_size != evidence.get("bytes") or
                sha256(path) != evidence.get("sha256")):
            raise ValueError(f"Preparation checkpoint artifact changed: {relative}")
    record = read_json(scan_dir/"record.json")
    proposals = read_jsonl(scan_dir/"proposals.jsonl")
    if record.get("scan_id") != scan["scan_id"]:
        raise ValueError("Checkpoint record belongs to another scan")
    if any(row.get("scan_id") != scan["scan_id"] for row in proposals):
        raise ValueError("Checkpoint proposal belongs to another scan")
    if len(proposals) != checkpoint.get("proposal_count"):
        raise ValueError("Checkpoint proposal count changed")
    if record.get("status") != checkpoint.get("status"):
        raise ValueError("Checkpoint status changed")
    if status == "failed":
        if proposals or not record.get("failure_reason"):
            raise ValueError("Failed checkpoint semantics changed")
        return record, proposals
    if record.get("failure_reason") is not None:
        raise ValueError("Successful checkpoint contains a failure reason")
    geometry = read_json(scan_dir/"geometry.json")
    shape = tuple(geometry.get("shape", []))
    if len(shape) != 3 or any(not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("Checkpoint geometry shape is invalid")
    for name, dtype in (("image", np.dtype("float32")),
                        ("valid", np.dtype("bool")),
                        ("lung_mask", np.dtype("bool")),
                        ("base_mask", np.dtype("bool"))):
        array = np.load(scan_dir/f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"Checkpoint {name} array semantics changed")
        if Path(record.get(f"{name}_path", "")).resolve() != (record_dir/f"{name}.npy").resolve():
            raise ValueError(f"Checkpoint {name} record path changed")
    affine = np.asarray(geometry.get("affine"), dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all() or
            not np.array_equal(affine, np.asarray(record.get("affine"), dtype=float))):
        raise ValueError("Checkpoint affine semantics changed")
    expected_ids = [f"{scan['scan_id']}:candidate:{index:06d}"
                    for index in range(len(proposals))]
    if [row.get("id") for row in proposals] != expected_ids:
        raise ValueError("Checkpoint proposal identities changed")
    return record, proposals


def _write_scan_checkpoint(scan, preparation_key, scan_dir, record, proposals,
                           preparation_seconds):
    write_json(scan_dir/"record.json", record)
    write_jsonl(scan_dir/"proposals.jsonl", proposals)
    files = {str(path.relative_to(scan_dir)): {
                 "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(scan_dir.iterdir()) if path.is_file()}
    write_json(scan_dir/"checkpoint.json", {"schema_version": 1,
        "kind": "lidc3d_scan_preparation", "scan_id": scan["scan_id"],
        "scan_hash": canonical_hash(scan), "preparation_key": preparation_key,
        "files": files, "proposal_count": len(proposals),
        "status": record["status"],
        "preparation_seconds": preparation_seconds})


def _write_aggregate(output, name, writer, value):
    pending = output/f".{name}.incomplete"
    if pending.exists():
        pending.unlink()
    writer(pending, value)
    with pending.open("rb") as handle:
        os.fsync(handle.fileno())
    pending.replace(output/name)
    descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_scan(staging, scan_dir, output):
    for path in staging.iterdir():
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    descriptor = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    staging.replace(scan_dir)
    descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _preparation_lock(output):
    lock = (output/".prepare.lock").open("a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another preparation process is using this output") from error
        yield
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _preparation_implementation_hash():
    root = Path(__file__).resolve().parent
    files = {name: sha256(root/name) for name in
             ("pipeline.py", "ingest.py", "geometry.py", "candidates.py",
              "provenance.py")}
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "pydicom", "scikit-image")}
    return canonical_hash({"files": files, "packages": packages})


def prepare_candidates(cohort, config, output, allow_test=False, resume=False,
                       materialize_pool=True):
    """CT-only proposal pools with durable, verified per-scan checkpoints.

    The Python API retains its list return by default for existing callers.
    CLI and large-cohort callers use materialize_pool=False for bounded memory.
    """
    if any(s["split"] == "test" for s in cohort) and not allow_test:
        raise ValueError("Test preparation requires a separately reviewed release")
    if not cohort:
        raise ValueError("Preparation cohort is empty")
    scan_ids = [s.get("scan_id") for s in cohort]
    if len(set(scan_ids)) != len(scan_ids):
        raise ValueError("Duplicate scan_id in preparation cohort")
    output = Path(output).resolve()
    for root in {s["source_root"] for s in cohort}:
        safe_output(output, root)
    for scan_id in scan_ids:
        _scan_paths(output, scan_id)
    selected_threshold = config["candidates"]["threshold"]
    threshold = (min(config["candidates"]["threshold_search"])
                 if selected_threshold is None else selected_threshold)
    plan = {"schema_version": 1, "kind": "lidc3d_preparation_plan",
            "implementation": "atomic_scan_checkpoints_v1",
            "implementation_hash": _preparation_implementation_hash(),
            "cohort_hash": canonical_hash(cohort),
            "config_hash": canonical_hash(config),
            "minimum_pool_threshold": threshold,
            "scan_ids": scan_ids}
    preparation_key = canonical_hash(plan)
    plan["preparation_key"] = preparation_key
    plan_path = output/"preparation_plan.json"
    if resume:
        if not output.is_dir() or not plan_path.is_file():
            raise ValueError("Cannot resume legacy or unbound preparation output; use a new output directory")
        if read_json(plan_path) != plan:
            raise ValueError("Resume cohort or configuration differs from preparation plan")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write_aggregate(output, "preparation_plan.json", write_json, plan)
    with _preparation_lock(output):
        scans, pool = _prepare_candidates_locked(
            cohort, config, output, threshold, preparation_key, resume)
        return scans, list(pool) if materialize_pool else pool


def _prepare_candidates_locked(cohort, config, output, threshold,
                               preparation_key, resume):
    for index, scan in enumerate(cohort, 1):
        _progress("source_verify", index=index, total=len(cohort),
                  scan_id=scan["scan_id"])
        _source_evidence(scan)
    final_path = output/"preparation.json"
    completed = final_path.exists()
    if completed:
        final = read_json(final_path)
        if (final.get("schema_version") != 1 or
                final.get("status") != "PROPOSALS_NOT_YET_FROZEN" or
                final.get("preparation_key") != preparation_key or
                final.get("cohort_hash") != canonical_hash(cohort) or
                canonical_hash(final.get("config")) != canonical_hash(config) or
                final.get("minimum_pool_threshold") != threshold):
            raise ValueError("Final preparation belongs to another plan")
        expected_files = final.get("files")
        if set(expected_files or {}) != {"scans.jsonl", "proposal_pool.jsonl"}:
            raise ValueError("Final preparation file inventory changed")
        for name, expected in expected_files.items():
            if sha256(output/name) != expected:
                raise ValueError(f"Final preparation artifact changed: {name}")
    checkpoints = {}
    for scan in cohort:
        scan_dir, _ = _scan_paths(output, scan["scan_id"])
        if (scan_dir/"checkpoint.json").is_file():
            record, proposals = _load_scan_checkpoint(scan, preparation_key, scan_dir)
            checkpoints[scan["scan_id"]] = record, len(proposals)
            del proposals
        elif completed:
            raise ValueError(
                f"Completed preparation is missing scan checkpoint: {scan['scan_id']}")
    scans, proposal_count = [], 0
    _progress("run_start", scans=len(cohort), resume=resume,
              preparation_key=preparation_key)
    for index, scan in enumerate(cohort, 1):
        started = time.perf_counter()
        scan_dir, staging = _scan_paths(output, scan["scan_id"])
        _progress("scan_start", index=index, total=len(cohort), scan_id=scan["scan_id"])
        if scan["scan_id"] in checkpoints:
            record, count = checkpoints[scan["scan_id"]]
            scans.append(record)
            proposal_count += count
            _progress("scan_resumed", index=index, total=len(cohort),
                      scan_id=scan["scan_id"], status=record["status"],
                      proposals=count)
            continue
        if completed:
            raise ValueError(f"Completed preparation is missing scan checkpoint: {scan['scan_id']}")
        if scan_dir.exists():
            if not resume:
                raise ValueError(f"Published scan directory lacks checkpoint: {scan['scan_id']}")
            shutil.rmtree(scan_dir)
        if staging.exists():
            if not resume:
                raise FileExistsError(staging)
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            record, proposals = _prepare_scan(scan, config, staging, scan_dir, threshold)
        except Exception as error:
            shutil.rmtree(staging)
            staging.mkdir()
            record = copy.deepcopy(scan)
            record.update({"status": "failed",
                           "failure_reason": f"{type(error).__name__}: {error}"})
            proposals = []
        preparation_seconds = time.perf_counter()-started
        _progress("scan_stage", scan_id=scan["scan_id"], stage="checkpoint")
        _write_scan_checkpoint(scan, preparation_key, staging, record, proposals,
                               preparation_seconds)
        record, proposals = _load_scan_checkpoint(
            scan, preparation_key, staging, record_dir=scan_dir)
        _publish_scan(staging, scan_dir, output)
        count = len(proposals)
        proposal_count += count
        del proposals
        scans.append(record)
        _progress("scan_complete", index=index, total=len(cohort),
                  scan_id=scan["scan_id"], status=record["status"],
                  proposals=count, seconds=preparation_seconds)
    pool = PreparedProposalPool(output, scans)
    if completed:
        aggregate_scans = read_jsonl(output/"scans.jsonl")
        if aggregate_scans != scans:
            raise ValueError("Final aggregates disagree with per-scan checkpoints")
        _verify_preparation_artifacts(
            output, aggregate_scans, pool, final, config)
        _progress("run_already_complete", scans=len(scans), proposals=proposal_count)
        return scans, pool
    _progress("source_reverify", scans=len(cohort))
    for scan in cohort:
        _source_evidence(scan)
    _write_aggregate(output, "scans.jsonl", write_jsonl, scans)
    _write_aggregate(output, "proposal_pool.jsonl", write_jsonl,
                     _checkpoint_proposals(output, scans))
    preparation = {"schema_version": 1, "preparation_key": preparation_key,
        "cohort_hash": canonical_hash(cohort), "config": config,
        "minimum_pool_threshold": threshold, "status": "PROPOSALS_NOT_YET_FROZEN",
        "files": {name: sha256(output/name)
                  for name in ("scans.jsonl", "proposal_pool.jsonl")},
        "artifacts": _collect_preparation_artifacts(output, scans)}
    _write_aggregate(output, "preparation.json", write_json, preparation)
    _progress("run_complete", scans=len(scans), proposals=proposal_count,
              failed=sum(row["status"] != "ok" for row in scans))
    return scans, pool


def select_candidates(scans, pool, config):
    """Finite prespecified validation search, recall -> count -> config order."""
    val_scans = [s for s in scans if s["split"] == "val"]
    if not val_scans:
        raise ValueError("No validation cohort")
    from .training import cases_from
    trace, options = [], []
    for index, threshold in enumerate(config["candidates"]["threshold_search"]):
        chosen = filter_pool(val_scans, pool, threshold, config["candidates"]["cap"])
        predictions = [{**p, "score": p["log_response"]} for p in chosen if p["scan_id"] in {s["scan_id"] for s in val_scans}]
        summary = DetectionCurves(cases_from(val_scans, predictions)).summarize()
        if summary["cpm"] is None:
            raise ValueError("Candidate recall undefined on validation")
        recall = summary["candidate_ceiling"]
        trace.append({"threshold": threshold, "candidate_recall": recall, "candidates": len(predictions)})
        options.append((-recall, len(predictions), index, threshold))
    threshold = min(options)[-1]
    return threshold, trace


def filter_pool(scans, pool, threshold, cap):
    selected = []
    for scan in scans:
        source = pool.per_scan(scan["scan_id"]) if isinstance(pool, PreparedProposalPool) else pool
        rows = [p for p in source if p["scan_id"] == scan["scan_id"] and p["log_response"] >= threshold]
        rows.sort(key=lambda p: (-p["log_response"], p["id"]))
        selected.extend(rows[:cap])
    return selected


def freeze_bundle(scans_path, pool_path, config, review, output, test=False):
    scans_path, pool_path = Path(scans_path).resolve(), Path(pool_path).resolve()
    if (scans_path.parent != pool_path.parent or scans_path.name != "scans.jsonl" or
            pool_path.name != "proposal_pool.jsonl"):
        raise ValueError("Freeze requires canonical preparation aggregate paths")
    scans = read_jsonl(scans_path)
    pool = PreparedProposalPool(scans_path.parent, scans)
    preparation = read_json(scans_path.parent / "preparation.json")
    _verify_preparation_artifacts(
        scans_path.parent, scans, pool, preparation, config)
    if canonical_hash(preparation["config"]) != canonical_hash(config):
        raise ValueError("Freeze configuration differs from actual preparation")
    if any(s.get("status") != "ok" for s in scans):
        raise ValueError("Failed preparation records block release")
    if any(s.get("unresolved_associations", 0) for s in scans):
        raise ValueError("Unresolved reader associations block release; adjudicate and rebuild references")
    if review.get("scans_sha256") != sha256(scans_path) or review.get("status") != "approved":
        raise ValueError("Reference review must approve this exact scans artifact")
    for key in ("reviewer", "geometry_audit", "annotation_audit", "source_exclusions_audit", "prior_test_access"):
        if not review.get(key):
            raise ValueError(f"Missing reference review field: {key}")
    if test:
        if any(s["split"] != "test" for s in scans) or config["candidates"]["threshold"] is None:
            raise ValueError("Test bundle requires test-only scans and a previously frozen threshold")
        threshold, trace = config["candidates"]["threshold"], []
    else:
        if any(s["split"] == "test" for s in scans):
            raise ValueError("Development bundle must exclude test records")
        threshold, trace = select_candidates(scans, pool, config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    resolved = copy.deepcopy(config)
    resolved["candidates"]["threshold"] = threshold
    candidates = filter_pool(scans, pool, threshold, resolved["candidates"]["cap"])
    write_jsonl(output/"candidates.jsonl", candidates)
    write_jsonl(output/"scans.jsonl", scans)
    write_json(output/"candidate_selection.json", {"threshold": threshold, "validation_trace": trace})
    write_json(output/"resolved_config.json", resolved)
    write_json(output/"reference_review.json", review)
    files = {}
    paths = [output/"candidates.jsonl", output/"scans.jsonl", output/"resolved_config.json", output/"reference_review.json"]
    for scan in scans:
        paths.extend(Path(scan[k]) for k in ("image_path", "valid_path", "lung_mask_path", "base_mask_path") if k in scan)
        paths.append(Path(scan["reference_audit_path"]))
        for ref in scan["positive"]+scan.get("ignored", []):
            if ref.get("mask_path"):
                paths.append(Path(ref["mask_path"]))
    for path in paths:
        files[str(path.resolve())] = sha256(path)
    bundle = {"schema_version": 1, "project": "independent_lidc3d", "scans": str(output/"scans.jsonl"),
        "candidates": str(output/"candidates.jsonl"), "files": files,
        "reference_audit_approved": True, "development_config_frozen": True,
        "resolved_config": str(output/"resolved_config.json"), "split_scope": "test" if test else "development"}
    write_json(output/"bundle.json", bundle)
    return bundle
