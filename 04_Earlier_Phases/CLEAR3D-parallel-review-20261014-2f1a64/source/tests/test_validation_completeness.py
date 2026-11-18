"""Synthetic CPU regressions: no clinical data or GPU operations."""
import copy
from pathlib import Path

import pytest
import torch

from lidc3d import training
from lidc3d.provenance import read_json, read_jsonl, sha256, write_jsonl
from test_ingest_and_guards import make_bundle


def validation_case(tmp_path, zero=False):
    bundle, cfg, scans, candidates = make_bundle(tmp_path)
    extra = copy.deepcopy(scans[1])
    extra.update(scan_id="s2", patient_id="p2", pixel_fingerprint="synthetic-fingerprint-2")
    scans.append(extra)
    candidates.extend({**row, "id": row["id"]+"-s2", "scan_id": "s2"}
                      for row in list(candidates) if row["scan_id"] == "s1")
    if zero:
        empty = copy.deepcopy(extra)
        empty.update(scan_id="empty", patient_id="p-empty", pixel_fingerprint="synthetic-empty")
        scans.append(empty)
    # Rewrite only synthetic fixture files, then bind their actual hashes.
    for key, rows in [("scans", scans), ("candidates", candidates)]:
        path = Path(bundle[key])
        path.unlink()
        write_jsonl(path, rows)
        bundle["files"][str(path)] = sha256(path)
    return bundle, cfg, scans, candidates


@pytest.mark.parametrize("failure_pass", [1, 2, 3], ids=["first-epoch", "later-epoch", "final-best"])
def test_incomplete_validation_cannot_select(tmp_path, monkeypatch, failure_pass):
    bundle, cfg, scans, candidates = validation_case(tmp_path)
    output = tmp_path / "run"
    real_predict = training.predict
    real_curves = training.DetectionCurves
    calls = {"predict": 0, "curves": 0, "operating": 0, "failed_forward": 0}
    before = {}

    class TracedCurves(real_curves):
        def __init__(self, cases):
            calls["curves"] += 1
            super().__init__(cases)

        def operating_threshold(self, *args, **kwargs):
            calls["operating"] += 1
            return super().operating_threshold(*args, **kwargs)

    def fail_during_one_scan(model, *args, **kwargs):
        calls["predict"] += 1
        if calls["predict"] != failure_pass:
            return real_predict(model, *args, **kwargs)
        for name in ["best.pt", "last.pt"]:
            if (output / name).exists():
                before[name] = sha256(output / name)
        real_forward = model.forward
        forwarded = 0

        def fail_second_batch(x):
            nonlocal forwarded
            forwarded += 1
            if forwarded == 2:
                calls["failed_forward"] += 1
                raise RuntimeError("synthetic partial scan failure")
            return real_forward(x)

        # s1 has three candidates with batch size 2: fail after its first batch.
        # s2 still completes normally; rejecting only an all-failed pass is wrong.
        with monkeypatch.context() as scoped:
            scoped.setattr(model, "forward", fail_second_batch)
            predictions, statuses = real_predict(model, *args, **kwargs)
        assert [s["status"] for s in statuses] == ["failed", "ok"]
        assert {p["scan_id"] for p in predictions} == {"s2"}
        return predictions, statuses

    monkeypatch.setattr(training, "predict", fail_during_one_scan)
    monkeypatch.setattr(training, "DetectionCurves", TracedCurves)
    with pytest.raises(ValueError, match="Incomplete validation") as caught:
        training.train(bundle, cfg, output, 17, "cpu")
    assert calls == {"predict": failure_pass, "curves": failure_pass-1,
                     "operating": 0, "failed_forward": 1}
    prefix = "validation" if failure_pass == 3 else f"validation_epoch_{failure_pass-1:03d}"
    statuses = read_jsonl(output / f"{prefix}_scan_status.jsonl")
    assert [s["scan_id"] for s in statuses] == ["s1", "s2"]
    assert statuses[0]["candidates"] == 3 and statuses[0]["predictions"] == 0
    assert "synthetic partial scan failure" in statuses[0]["reason"]
    assert statuses[1]["status"] == "ok" and statuses[1]["predictions"] == 3
    assert {p["scan_id"] for p in read_jsonl(output / f"{prefix}_predictions.jsonl")} == {"s2"}
    evidence = read_json(output / f"{prefix}_failure.json")
    assert evidence["expected_scan_ids"] == ["s1", "s2"] and evidence["status"] == "INCOMPLETE"
    assert read_json(output / "failure.json")["status"] == "INCOMPLETE"
    assert not (output / "selection.json").exists()
    for name in ["best.pt", "last.pt"]:
        if name in before:
            assert sha256(output / name) == before[name]
        else:
            assert not (output / name).exists()
    if failure_pass <= 2:
        assert not (output / f"epoch_{failure_pass-1:03d}.json").exists()
        assert not (output / f"val_predictions_epoch_{failure_pass-1:03d}.jsonl").exists()
    # Check the actual train frame at the exception, not just files: neither
    # best-CPM nor stopping state may change on the failed validation pass.
    tb = caught.value.__traceback__
    while tb and tb.tb_frame.f_code.co_name != "train":
        tb = tb.tb_next
    assert tb is not None
    local = tb.tb_frame.f_locals
    if before:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        assert (local["checkpoint_best"], local["stopping_best"], local["stale"]) == (
            saved["best_cpm"], saved["stopping_best"], saved["stale"])
    else:
        assert (local["checkpoint_best"], local["stopping_best"], local["stale"]) == (
            -float("inf"), -float("inf"), 0)


def test_zero_candidate_scan_is_successful_and_retained(tmp_path):
    bundle, cfg, scans, candidates = validation_case(tmp_path, zero=True)
    output = tmp_path / "run"
    training.train(bundle, cfg, output, 17, "cpu")
    for prefix in ["validation_epoch_000", "validation_epoch_001", "validation"]:
        statuses = read_jsonl(output / f"{prefix}_scan_status.jsonl")
        assert [s["scan_id"] for s in statuses] == ["s1", "s2", "empty"]
        empty = statuses[-1]
        assert empty["status"] == "ok" and empty["reason"] is None
        assert empty["candidates"] == empty["predictions"] == 0
        assert not (output / f"{prefix}_failure.json").exists()
    assert read_json(output / "selection.json")["status"] == "TRAINING_COMPLETE"
    assert not (output / "failure.json").exists()
    cases = training.cases_from([s for s in scans if s["split"] == "val"], [])
    assert [c["scan_id"] for c in cases] == ["s1", "s2", "empty"]


@pytest.mark.parametrize("statuses", [
    [],
    [{"scan_id": "unexpected", "status": "ok", "candidates": 0, "predictions": 0}],
    [{"scan_id": "s1", "status": "ok", "candidates": 2, "predictions": 1}],
    [{"scan_id": "s1", "status": "ok"}],
    [{"scan_id": "s1", "status": "ok", "candidates": 0, "predictions": 0}]*2,
])
def test_missing_duplicate_or_incomplete_scan_status_is_rejected(tmp_path, statuses):
    with pytest.raises(ValueError, match="Incomplete validation"):
        training._record_validation(tmp_path, "validation", [{"scan_id": "s1"}], [], statuses)
    assert read_jsonl(tmp_path / "validation_scan_status.jsonl") == statuses
    assert read_json(tmp_path / "validation_failure.json")["expected_scan_ids"] == ["s1"]
