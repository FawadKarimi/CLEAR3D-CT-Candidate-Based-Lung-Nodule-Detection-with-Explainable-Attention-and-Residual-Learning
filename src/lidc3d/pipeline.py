"""File-oriented preparation with separate cohort, candidate and prediction stages."""
from __future__ import annotations
import copy
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from .ingest import load_series
from .geometry import resample_lps, normalize, to_world
from .references import xml_root, parse_xml, make_references
from .splits import patient_split
from .candidates import lung_mask, generate
from .evaluation import DetectionCurves
from .provenance import write_json, write_jsonl, read_jsonl, read_json, sha256, canonical_hash, safe_output


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


def prepare_candidates(cohort, config, output, allow_test=False):
    """CT-only proposal pools; threshold search uses validation references later."""
    if any(s["split"] == "test" for s in cohort) and not allow_test:
        raise ValueError("Test preparation requires a separately reviewed release")
    output = Path(output).resolve()
    for root in {s["source_root"] for s in cohort}:
        safe_output(output, root)
    output.mkdir(parents=True, exist_ok=False)
    scans, pool = [], []
    selected_threshold = config["candidates"]["threshold"]
    threshold = min(config["candidates"]["threshold_search"]) if selected_threshold is None else selected_threshold
    for scan in cohort:
        started = time.perf_counter()
        record = copy.deepcopy(scan)
        record.update({"status": "ok", "failure_reason": None})
        scan_dir = output/scan["scan_id"]
        scan_dir.mkdir()
        try:
            for path, expected in scan["source_hashes"].items():
                if sha256(path) != expected:
                    raise ValueError("Raw source changed since cohort construction")
            native = _load_configured_series(scan["dicom_paths"], config)
            if native.metadata["series_uid"] != scan["series_uid"]:
                raise ValueError("Loaded DICOM series disagrees with cohort manifest")
            record["pixel_fingerprint"] = native.metadata["pixel_fingerprint"]
            record["native_affine"] = native.affine.tolist()
            record["source_hashes"] = native.metadata["source_hashes"]
            record["coordinate_aware_resampling"] = native.metadata["coordinate_aware_resampling"]
            volume = resample_lps(native, config["preprocess"]["spacing_mm"])
            del native
            image = normalize(volume.hu, config["preprocess"]["window_hu"])
            image[~volume.valid] = 0
            mask, base, mask_status = lung_mask(volume.hu, volume.valid,
                spacing=config["preprocess"]["spacing_mm"], air_hu=config["preprocess"]["mask_air_hu"],
                min_component_mm3=config["preprocess"]["mask_min_component_mm3"],
                dilation_mm=config["preprocess"]["mask_dilation_mm"])
            c = {**config["candidates"], "threshold": threshold}
            _, diagnostics = generate(image, mask, c, config["preprocess"]["spacing_mm"])
            proposals = diagnostics.pop("precap_candidates")
            for i, candidate in enumerate(proposals):
                pool.append({**candidate, "id": f"{scan['scan_id']}:candidate:{i:06d}",
                    "scan_id": scan["scan_id"], "center_lps": to_world(candidate["zyx"], volume.affine).tolist()})
            for name, array in [("image", image), ("valid", volume.valid), ("lung_mask", mask), ("base_mask", base)]:
                path = scan_dir/f"{name}.npy"
                np.save(path, array, allow_pickle=False)
                record[f"{name}_path"] = str(path)
            record["affine"] = volume.affine.tolist()
            record["mask_diagnostics"] = mask_status
            record["candidate_diagnostics"] = diagnostics
            write_json(scan_dir/"geometry.json", {"affine": record["affine"], "shape": list(image.shape),
                "metadata": volume.metadata, "array_order": "zyx", "patient_coordinates": "LPS_mm"})
        except Exception as error:
            record["status"] = "failed"
            record["failure_reason"] = f"{type(error).__name__}: {error}"
            # No case-specific rescue; failed eligible scans retain references.
            pool = [p for p in pool if p["scan_id"] != scan["scan_id"]]
        record["preparation_seconds"] = time.perf_counter()-started
        scans.append(record)
    write_jsonl(output/"scans.jsonl", scans)
    write_jsonl(output/"proposal_pool.jsonl", pool)
    write_json(output/"preparation.json", {"cohort_hash": canonical_hash(cohort), "config": config,
        "minimum_pool_threshold": threshold, "status": "PROPOSALS_NOT_YET_FROZEN"})
    return scans, pool


def select_candidates(scans, pool, config):
    """Finite prespecified validation search, recall -> count -> config order."""
    val_scans = [s for s in scans if s["split"] == "val"]
    if not val_scans:
        raise ValueError("No validation cohort")
    from .training import cases_from
    trace, options = [], []
    for index, threshold in enumerate(config["candidates"]["threshold_search"]):
        chosen = filter_pool(scans, pool, threshold, config["candidates"]["cap"])
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
        rows = [p for p in pool if p["scan_id"] == scan["scan_id"] and p["log_response"] >= threshold]
        rows.sort(key=lambda p: (-p["log_response"], p["id"]))
        selected.extend(rows[:cap])
    return selected


def freeze_bundle(scans_path, pool_path, config, review, output, test=False):
    scans, pool = read_jsonl(scans_path), read_jsonl(pool_path)
    preparation = read_json(Path(scans_path).parent / "preparation.json")
    if canonical_hash(preparation["config"]) != canonical_hash(config):
        raise ValueError("Freeze configuration differs from actual preparation")
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
