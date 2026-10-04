import copy
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from lidc3d import pipeline
from lidc3d.geometry import Volume
from lidc3d.provenance import read_jsonl, sha256


@pytest.fixture
def resume_case(tmp_path, monkeypatch):
    raw = tmp_path/"raw"
    raw.mkdir()
    audit = tmp_path/"reference-audit.json"
    audit.write_text("{}\n")
    cohort, identities = [], {}
    for index in range(2):
        uid = f"s{index}"
        source = (raw/f"{uid}.dcm").resolve()
        source.write_bytes(f"raw-{uid}".encode())
        identities[uid] = {"patient_id": f"p{index}",
            "study_uid": f"study{index}", "series_uid": uid}
        cohort.append({"scan_id": f"scan_{uid}", **identities[uid],
            "source_root": str(raw.resolve()), "dicom_paths": [str(source)],
            "source_hashes": {str(source): sha256(source)}, "positive": [],
            "ignored": [], "reference_audit_path": str(audit.resolve()),
            "unresolved_associations": 0,
            "split": "train" if index == 0 else "val",
            "status": "manifest_only_not_prepared"})
    config = {"geometry": {}, "preprocess": {"spacing_mm": 1.0,
        "window_hu": [-1000, 400], "mask_air_hu": -400,
        "mask_min_component_mm3": 1, "mask_dilation_mm": 0},
        "candidates": {"threshold": 0.1, "threshold_search": [0.1],
                       "sigmas_mm": [1.0], "nms_mm": 1.0, "cap": 10}}
    calls = Counter()

    def fake_load(paths, _config):
        source = Path(paths[0]).resolve()
        uid = source.stem
        calls[uid] += 1
        return Volume(np.full((2, 3, 4), -700, np.float32),
            np.ones((2, 3, 4), bool), np.eye(4), {
                **identities[uid], "pixel_fingerprint": f"fingerprint-{uid}",
                "source_hashes": {str(source): sha256(source)},
                "coordinate_aware_resampling": uid == "s0"})

    def fake_resample(volume, spacing):
        return Volume(volume.hu.copy(), volume.valid.copy(), volume.affine.copy(),
            {**volume.metadata, "spacing_mm": spacing,
             "native_affine": volume.affine.tolist()})

    def fake_mask(hu, valid, **_kwargs):
        mask = np.ones_like(valid)
        return mask, mask.copy(), {"components_retained": 1,
            "base_voxels": int(mask.sum()), "dilated_voxels": int(mask.sum()),
            "valid_voxels": int(valid.sum()), "retained_fraction": 1.0,
            "base_touches_z_boundary": True}

    def fake_generate(_image, _mask, _config, _spacing):
        proposal = {"zyx": [1.0, 1.0, 2.0], "log_response": 0.75,
                    "sigma_mm": 1.0}
        return [proposal], {"positive_peaks_in_mask": 1,
            "threshold_peaks": 1, "nms_suppressed": 0, "before_cap": 1,
            "cap_hit": False, "precap_candidates": [proposal]}

    monkeypatch.setattr(pipeline, "_load_configured_series", fake_load)
    monkeypatch.setattr(pipeline, "resample_lps", fake_resample)
    monkeypatch.setattr(pipeline, "lung_mask", fake_mask)
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    return {"cohort": cohort, "config": config,
            "output": tmp_path/"prepared", "calls": calls}


def interrupt_after_first(case, monkeypatch):
    real_save, count = pipeline.np.save, 0

    def interrupting_save(path, array, *args, **kwargs):
        nonlocal count
        real_save(path, array, *args, **kwargs)
        count += 1
        if count == 5:
            raise KeyboardInterrupt("synthetic interruption")

    with monkeypatch.context() as scoped:
        scoped.setattr(pipeline.np, "save", interrupting_save)
        with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
            pipeline.prepare_candidates(case["cohort"], case["config"],
                                        case["output"])
    assert (case["output"]/"preparation_plan.json").is_file()
    assert (case["output"]/"scan_s0"/"checkpoint.json").is_file()
    assert not (case["output"]/"scan_s1").exists()
    assert (case["output"] / ".scan_s1.incomplete").is_dir()
    assert not (case["output"]/"preparation.json").exists()


def snapshot(root):
    return [(str(path.relative_to(root)), path.stat().st_size, sha256(path))
            for path in sorted(Path(root).rglob("*")) if path.is_file()]


def test_resume_reuses_checkpoint_and_recovers_staging(resume_case, monkeypatch):
    case = resume_case
    interrupt_after_first(case, monkeypatch)
    image = case["output"]/"scan_s0"/"image.npy"
    before = sha256(image), image.stat().st_mtime_ns
    scans, proposals = pipeline.prepare_candidates(case["cohort"], case["config"],
                                                    case["output"], resume=True)
    assert case["calls"] == Counter({"s1": 2, "s0": 1})
    assert (sha256(image), image.stat().st_mtime_ns) == before
    assert [row["scan_id"] for row in scans] == ["scan_s0", "scan_s1"]
    assert all(row["status"] == "ok" for row in scans)
    assert [row["id"] for row in proposals] == [
        "scan_s0:candidate:000000", "scan_s1:candidate:000000"]
    assert read_jsonl(case["output"]/"scans.jsonl") == scans
    assert (case["output"]/"preparation.json").is_file()
    assert not (case["output"] / ".scan_s1.incomplete").exists()
    for row in scans:
        assert Path(row["image_path"]).parent == case["output"]/row["scan_id"]


def test_resume_rebuilds_markerless_scan_directory(resume_case, monkeypatch):
    case = resume_case
    interrupt_after_first(case, monkeypatch)
    staging = case["output"] / ".scan_s1.incomplete"
    staging.replace(case["output"] / "scan_s1")
    pipeline.prepare_candidates(case["cohort"], case["config"],
                                case["output"], resume=True)
    assert case["calls"] == Counter({"s1": 2, "s0": 1})
    assert (case["output"] / "scan_s1" / "checkpoint.json").is_file()


@pytest.mark.parametrize("drift", ["config", "cohort"])
def test_resume_rejects_plan_drift_before_mutation(resume_case, monkeypatch, drift):
    case = resume_case
    interrupt_after_first(case, monkeypatch)
    cohort, config = copy.deepcopy(case["cohort"]), copy.deepcopy(case["config"])
    if drift == "config":
        config["preprocess"]["window_hu"][1] = 350
    else:
        cohort[1]["patient_id"] = "changed"
    before = snapshot(case["output"])
    with pytest.raises(ValueError, match="cohort or configuration"):
        pipeline.prepare_candidates(cohort, config, case["output"], resume=True)
    assert snapshot(case["output"]) == before


def test_resume_rejects_source_or_committed_artifact_drift(resume_case, monkeypatch):
    case = resume_case
    interrupt_after_first(case, monkeypatch)
    source = Path(case["cohort"][0]["dicom_paths"][0])
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Raw source changed"):
        pipeline.prepare_candidates(case["cohort"], case["config"],
                                    case["output"], resume=True)
    source.write_bytes(b"raw-s0")
    image = case["output"]/"scan_s0"/"image.npy"
    with image.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        pipeline.prepare_candidates(case["cohort"], case["config"],
                                    case["output"], resume=True)


def test_unsafe_and_duplicate_scan_ids_are_rejected(resume_case):
    case = resume_case
    bad = copy.deepcopy(case["cohort"])
    bad[0]["scan_id"] = "../escape"
    with pytest.raises(ValueError, match="Unsafe scan_id"):
        pipeline.prepare_candidates(bad, case["config"], case["output"])
    duplicate = copy.deepcopy(case["cohort"])
    duplicate[1]["scan_id"] = duplicate[0]["scan_id"]
    with pytest.raises(ValueError, match="Duplicate scan_id"):
        pipeline.prepare_candidates(duplicate, case["config"],
                                    case["output"].with_name("duplicate"))


def test_resume_recovers_interrupted_aggregation_without_recompute(
        resume_case, monkeypatch):
    case = resume_case
    real_write = pipeline._write_aggregate
    calls = 0

    def interrupting_write(*args, **kwargs):
        nonlocal calls
        real_write(*args, **kwargs)
        if args[1] == "scans.jsonl":
            calls += 1
            raise KeyboardInterrupt("aggregate interruption")

    with monkeypatch.context() as scoped:
        scoped.setattr(pipeline, "_write_aggregate", interrupting_write)
        with pytest.raises(KeyboardInterrupt, match="aggregate interruption"):
            pipeline.prepare_candidates(case["cohort"], case["config"],
                                        case["output"])
    assert (case["output"]/"scans.jsonl").is_file()
    assert not (case["output"]/"preparation.json").exists()
    before = case["calls"].copy()
    pipeline.prepare_candidates(case["cohort"], case["config"],
                                case["output"], resume=True)
    assert case["calls"] == before
    assert (case["output"]/"preparation.json").is_file()


def test_completed_aggregate_drift_is_rejected(resume_case):
    case = resume_case
    pipeline.prepare_candidates(case["cohort"], case["config"], case["output"])
    with (case["output"]/"scans.jsonl").open("a") as handle:
        handle.write("{}\n")
    before = case["calls"].copy()
    with pytest.raises(ValueError, match="Final preparation artifact changed"):
        pipeline.prepare_candidates(case["cohort"], case["config"],
                                    case["output"], resume=True)
    assert case["calls"] == before


def test_concurrent_resume_is_rejected(resume_case, monkeypatch):
    case = resume_case
    interrupt_after_first(case, monkeypatch)
    with pipeline._preparation_lock(case["output"]):
        with pytest.raises(RuntimeError, match="Another preparation process"):
            pipeline.prepare_candidates(case["cohort"], case["config"],
                                        case["output"], resume=True)


def test_completed_run_revalidates_source_and_arrays(resume_case):
    case = resume_case
    pipeline.prepare_candidates(case["cohort"], case["config"], case["output"])
    source = Path(case["cohort"][0]["dicom_paths"][0])
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Raw source changed"):
        pipeline.prepare_candidates(case["cohort"], case["config"],
                                    case["output"], resume=True)
    source.write_bytes(b"raw-s0")
    with (case["output"]/"scan_s0"/"image.npy").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="artifact changed"):
        pipeline.prepare_candidates(case["cohort"], case["config"],
                                    case["output"], resume=True)


def test_source_hashes_must_exactly_cover_paths(resume_case):
    case = resume_case
    extra = Path(case["cohort"][0]["source_root"])/"extra.dcm"
    extra.write_bytes(b"extra")
    case["cohort"][0]["dicom_paths"].append(str(extra))
    with pytest.raises(ValueError, match="exactly cover"):
        pipeline.prepare_candidates(case["cohort"], case["config"], case["output"])


@pytest.mark.parametrize("identity", ["patient_id", "study_uid"])
def test_loaded_identity_mismatch_is_failed_and_blocks_freeze(
        resume_case, monkeypatch, identity):
    case = resume_case
    real_load = pipeline._load_configured_series

    def mismatched(*args, **kwargs):
        volume = real_load(*args, **kwargs)
        volume.metadata[identity] = "wrong"
        return volume

    monkeypatch.setattr(pipeline, "_load_configured_series", mismatched)
    scans, _ = pipeline.prepare_candidates(
        case["cohort"], case["config"], case["output"])
    assert all(row["status"] == "failed" for row in scans)
    assert all(identity in row["failure_reason"] for row in scans)


@pytest.mark.parametrize("name", ["scans.jsonl", "proposal_pool.jsonl"])
def test_freeze_rejects_preparation_aggregate_drift(resume_case, name):
    case = resume_case
    pipeline.prepare_candidates(case["cohort"], case["config"], case["output"])
    with (case["output"]/name).open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="Preparation aggregate changed"):
        pipeline.freeze_bundle(
            case["output"]/"scans.jsonl",
            case["output"]/"proposal_pool.jsonl",
            case["config"], {}, case["output"].with_name("frozen"))


def test_disk_pool_fresh_and_complete_resume_without_aggregate_loading(resume_case, monkeypatch):
    case = resume_case
    real_read = pipeline.read_jsonl

    def bounded_read(path):
        assert Path(path).name != "proposal_pool.jsonl"
        return real_read(path)

    monkeypatch.setattr(pipeline, "read_jsonl", bounded_read)
    scans, pool = pipeline.prepare_candidates(case["cohort"], case["config"],
        case["output"], materialize_pool=False)
    assert isinstance(pool, pipeline.PreparedProposalPool)
    expected = read_jsonl(case["output"]/"proposal_pool.jsonl")
    assert list(pool) == list(pool) == expected
    assert len(pool) == 2
    assert (case["output"]/"proposal_pool.jsonl").read_bytes() == b"".join(
        (case["output"]/scan["scan_id"]/"proposals.jsonl").read_bytes()
        for scan in scans)
    before = snapshot(case["output"])
    calls = case["calls"].copy()
    resumed_scans, resumed_pool = pipeline.prepare_candidates(
        case["cohort"], case["config"], case["output"], resume=True,
        materialize_pool=False)
    assert resumed_scans == scans and list(resumed_pool) == expected
    assert case["calls"] == calls
    assert snapshot(case["output"]) == before


def test_disk_pool_interrupted_resume(resume_case, monkeypatch):
    case = resume_case
    interrupt_after_first(case, monkeypatch)
    scans, pool = pipeline.prepare_candidates(case["cohort"], case["config"],
        case["output"], resume=True, materialize_pool=False)
    assert case["calls"]["s0"] == 1
    assert len(scans) == 2 and len(pool) == 2
    assert list(pool) == read_jsonl(case["output"]/"proposal_pool.jsonl")


@pytest.mark.parametrize("mutation", ["truncate", "extra", "reverse"])
def test_streaming_verifier_rejects_rehashed_aggregate_drift(resume_case, mutation):
    from lidc3d.provenance import read_json
    case = resume_case
    scans, pool = pipeline.prepare_candidates(case["cohort"], case["config"],
        case["output"], materialize_pool=False)
    path = case["output"]/"proposal_pool.jsonl"
    lines = path.read_text().splitlines(True)
    if mutation == "truncate":
        lines = lines[:-1]
    elif mutation == "extra":
        lines.append(lines[0])
    else:
        lines.reverse()
    path.write_text("".join(lines))
    preparation = read_json(case["output"]/"preparation.json")
    preparation["files"]["proposal_pool.jsonl"] = sha256(path)
    with pytest.raises(ValueError, match="aggregation changed"):
        pipeline._verify_preparation_artifacts(case["output"], scans, pool,
                                               preparation, case["config"])


def test_disk_selection_matches_list_threshold_ties_caps(tmp_path):
    from lidc3d.provenance import write_jsonl
    scans = [{"scan_id": "scan_s0"}, {"scan_id": "scan_s1"}]
    rows = []
    for scan in scans:
        directory = tmp_path/scan["scan_id"]
        directory.mkdir()
        proposals = [{"scan_id": scan["scan_id"], "id": f"{scan['scan_id']}:{i}",
                      "log_response": response}
                     for i, response in enumerate([.5, .2, .5, .8])]
        write_jsonl(directory/"proposals.jsonl", proposals)
        rows.extend(proposals)
    write_jsonl(tmp_path/"proposal_pool.jsonl", rows)
    pool = pipeline.PreparedProposalPool(tmp_path, scans)
    for threshold in [.2, .5, .9]:
        for cap in [0, 1, 3, 10]:
            assert pipeline.filter_pool(scans, pool, threshold, cap) == pipeline.filter_pool(
                scans, rows, threshold, cap)


def test_freeze_does_not_load_aggregate_pool(resume_case, monkeypatch):
    case = resume_case
    pipeline.prepare_candidates(case["cohort"], case["config"], case["output"])
    real_read = pipeline.read_jsonl
    def guarded(path):
        assert Path(path).name != "proposal_pool.jsonl"
        return real_read(path)
    monkeypatch.setattr(pipeline, "read_jsonl", guarded)
    # Hash verification must complete before the missing review is rejected.
    with pytest.raises(ValueError, match="Reference review"):
        pipeline.freeze_bundle(case["output"]/"scans.jsonl",
            case["output"]/"proposal_pool.jsonl", case["config"], {},
            case["output"].with_name("frozen"))


def test_disk_threshold_trace_and_frozen_candidates_match_lists(resume_case, monkeypatch):
    from lidc3d.provenance import read_json
    case = resume_case
    case["cohort"][1]["positive"] = [{"id": "nodule", "center_lps": [1., 1., 2.],
                                       "radius_mm": 2.}]
    case["config"]["candidates"]["threshold_search"] = [.1, .5, .9]
    scans, pool = pipeline.prepare_candidates(case["cohort"], case["config"],
        case["output"], materialize_pool=False)
    rows = list(pool)
    assert pipeline.select_candidates(scans, pool, case["config"]) == pipeline.select_candidates(
        scans, rows, case["config"])
    threshold, trace = pipeline.select_candidates(scans, rows, case["config"])
    review = {"status": "approved", "scans_sha256": sha256(case["output"]/"scans.jsonl"),
              **{key: "synthetic test review" for key in ("reviewer", "geometry_audit",
                 "annotation_audit", "source_exclusions_audit", "prior_test_access")}}
    real_read = pipeline.read_jsonl
    def guarded(path):
        assert Path(path).name != "proposal_pool.jsonl"
        return real_read(path)
    monkeypatch.setattr(pipeline, "read_jsonl", guarded)
    output = case["output"].with_name("frozen")
    pipeline.freeze_bundle(case["output"]/"scans.jsonl",
        case["output"]/"proposal_pool.jsonl", case["config"], review, output)
    assert read_jsonl(output/"candidates.jsonl") == pipeline.filter_pool(
        scans, rows, threshold, case["config"]["candidates"]["cap"])
    assert read_json(output/"candidate_selection.json") == {
        "threshold": threshold, "validation_trace": trace}
