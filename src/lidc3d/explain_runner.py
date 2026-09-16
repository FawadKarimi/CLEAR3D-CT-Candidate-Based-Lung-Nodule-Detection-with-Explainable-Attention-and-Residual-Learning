"""Deterministic explanation case selection and per-case artifacts."""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from .provenance import read_json, read_jsonl, write_json, sha256, verify_bundle, canonical_hash
from .data import CandidateDataset
from .geometry import to_world, to_index
from .models import DetectorClassifier
from .references import mask_lookup
from .evaluation import match, clustered_case_mean
from . import xai


def select_cases(scans, predictions):
    selected = []
    for scan in scans:
        preds = sorted([p for p in predictions if p["scan_id"] == scan["scan_id"]],
                       key=lambda p: (-p["score"], p["id"]))
        refs = sorted(scan["positive"], key=lambda r: r["id"])
        matched_ids = set()
        if preds and refs:
            eligible = np.array([[np.linalg.norm(np.array(p["center_lps"])-r["center_lps"]) <= r["radius_mm"]
                                  for p in preds] for r in refs])
            # Maximize coverage, then minimize candidate score ranks. This
            # resolves competing lesions without using one candidate twice.
            penalty = (len(preds)+1)*(len(refs)+1)
            cost = np.full((len(refs), len(preds)+len(refs)), float(penalty))
            cost[:, :len(preds)] = np.where(eligible, np.arange(len(preds))[None], 2*penalty)
            rows, cols = linear_sum_assignment(cost)
            for r, p in zip(rows, cols):
                if p < len(preds) and eligible[r, p]:
                    selected.append({"candidate": preds[p], "reference": refs[r], "type": "covered_positive"})
                    matched_ids.add(preds[p]["id"])
        evaluator = match(preds, refs, scan.get("ignored", []))
        evaluated_positive = {p for _, p in evaluator["pairs"]}
        from .references import inside_ignore
        negatives = [p for p in preds if p["id"] not in evaluated_positive and p["id"] not in matched_ids and
            (any(np.linalg.norm(np.array(p["center_lps"])-r["center_lps"]) <= r["radius_mm"] for r in refs)
             or not inside_ignore(p["center_lps"], scan.get("ignored", [])))]
        if negatives:
            selected.append({"candidate": negatives[0], "reference": None, "type": "evaluator_negative"})
    return selected


def patch_reference_mask(reference, scan, start, size):
    target = np.zeros((size,)*3, bool)
    if not reference or not reference.get("mask_path"):
        return target
    voxels, native_affine = mask_lookup(reference["mask_path"])
    # Pull back every patch voxel, matching nearest-neighbor mask resampling.
    indices = np.indices((size,)*3).reshape(3, -1).T + np.asarray(start)
    native = np.floor(to_index(to_world(indices, np.array(scan["affine"])), native_affine)+0.5).astype(int)
    target.ravel()[:] = [tuple(index) in voxels for index in native]
    return target


def explain(bundle_path, checkpoint, predictions_path, output, device="cuda", randomization=False):
    bundle = read_json(bundle_path)
    scans = verify_bundle(bundle)
    predictions = read_jsonl(predictions_path)
    provenance_path = Path(predictions_path).parent / "provenance.json"
    provenance = read_json(provenance_path)
    if provenance.get("checkpoint_sha256") != sha256(checkpoint) or provenance.get("predictions_sha256") != sha256(predictions_path):
        raise ValueError("Explanation checkpoint/prediction provenance mismatch")
    if provenance.get("bundle_hash") != canonical_hash(bundle):
        raise ValueError("Explanation bundle differs from inference bundle")
    by_scan = {s["scan_id"]: s for s in scans}
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DetectorClassifier(state["config"]["model"]).to(device)
    model.load_state_dict(state["model"])
    cfg, size = state["config"]["xai"], state["config"]["model"]["patch_size"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected = select_cases(scans, predictions)
    write_json(output/"case_selection.json", selected)
    summary, clustered = [], {}
    for index, item in enumerate(selected):
        candidate, reference = item["candidate"], item["reference"]
        scan = by_scan[candidate["scan_id"]]
        folder = output/f"case_{index:06d}"
        folder.mkdir()
        data = CandidateDataset([scan], [candidate], size)[0]
        image = data["image"][None].to(device)
        valid = data["valid"][0].numpy()
        target = patch_reference_mask(reference, scan, data["start_zyx"], size)
        maps, records, failures = {}, {}, []
        try:
            cam = xai.gradcam(model, image)
            np.save(folder/"gradcam_native.npy", cam["native"].cpu().numpy(), allow_pickle=False)
            maps["gradcam"] = xai.normalize_map(cam["map"][0, 0].cpu().numpy(), valid)
        except Exception as error:
            failures.append({"method": "gradcam", "reason": str(error)})
        for label, baseline in [("ig", torch.zeros_like(image)), ("ig_blur", xai.blurred_reference(image, valid))]:
            try:
                ig = xai.integrated_gradients(model, image, baseline, cfg["ig_intervals"], cfg["ig_atol"], cfg["ig_rtol"])
                np.save(folder/f"{label}_signed.npy", ig["signed"].cpu().numpy(), allow_pickle=False)
                records[label] = {k:v for k,v in ig.items() if k not in ("signed", "positive")}
                if not ig["complete"]:
                    raise ValueError("IG completeness tolerance not met")
                maps[label] = xai.normalize_map(ig["positive"][0, 0].cpu().numpy(), valid)
            except Exception as error:
                failures.append({"method": label, "reason": str(error)})
        try:
            scales, attention = xai.attention_maps(model, image, valid)
            maps["attention"] = attention
            for scale, array in scales.items():
                np.save(folder/f"attention_scale_{scale}.npy", array, allow_pickle=False)
        except Exception as error:
            failures.append({"method": "attention", "reason": str(error)})
        if all(k in maps for k in ("gradcam", "ig", "attention")):
            components = [maps[k] for k in ("gradcam", "ig", "attention")]
            maps["fixed_mixture"] = xai.normalize_map(sum(w*m for w,m in zip(cfg["mixture_weights"], components)), valid)
            maps["equal_mixture"] = xai.normalize_map(np.mean(components, axis=0), valid)
        maps.update(xai.controls(valid.shape, valid, cfg["seed"]+index))
        for name, array in maps.items():
            np.save(folder/f"{name}.npy", array, allow_pickle=False)
            records.setdefault(name, {})["localization"] = {str(f): xai.localization(array, target, valid, f) for f in cfg["top_fractions"]}
            records[name]["blur_perturbation"] = xai.perturbation_curves(model, image, array, valid,
                block=cfg["perturbation_block"], steps=cfg["perturbation_steps"])
            records[name]["air_perturbation"] = xai.perturbation_curves(model, image, array, valid, torch.zeros_like(image),
                block=cfg["perturbation_block"], steps=cfg["perturbation_steps"])
        pairs = {f"{a}__{b}": xai.agreement(maps[a], maps[b], valid) for a,b in itertools.combinations(maps, 2)}
        stability_records = {name: xai.stability(model,image,valid,maps[name],name,cfg,cfg["seed"]+index)
                             for name in ("gradcam","ig","attention") if name in maps}
        random_records = []
        if randomization:
            for layers, randomized in xai.randomized_models(model, cfg["seed"]):
                for method in ("gradcam","ig","attention"):
                    if method not in maps:
                        continue
                    try:
                        changed = xai.method_map(randomized,image,valid,method,cfg)
                        random_records.append({"layers": layers, "method": method, **xai.agreement(maps[method], changed, valid)})
                    except Exception as error:
                        random_records.append({"layers": layers, "method": method, "failure": str(error)})
        result = {"candidate_id": candidate["id"], "patient_id": scan["patient_id"], "split": scan["split"],
            "type": item["type"], "records": records, "agreement": pairs, "failures": failures,
            "randomization": random_records, "stability": stability_records, "start_zyx": data["start_zyx"].tolist(),
            "status": "COMPUTED_REQUIRES_REVIEW"}
        write_json(folder/"results.json", result)
        summary.append({"candidate_id": candidate["id"], "patient_id": scan["patient_id"], "failures": failures})
        for method, result in records.items():
            for fraction, values in result.get("localization", {}).items():
                for metric in ("dice", "iou", "pointing"):
                    key=f"{method}:top{fraction}:{metric}"
                    clustered.setdefault(key,[]).append({"patient_id":scan["patient_id"],"value":values[metric]})
            for replacement in ("blur", "air"):
                for metric in ("insertion_auc", "deletion_auc"):
                    key=f"{method}:{replacement}:{metric}"
                    clustered.setdefault(key,[]).append({"patient_id":scan["patient_id"],"value":result[f"{replacement}_perturbation"][metric]})
    write_json(output/"clustered_summary.json", {key:clustered_case_mean(rows) for key,rows in clustered.items()})
    write_json(output/"summary.json", {"cases": summary, "checkpoint_sha256": sha256(checkpoint),
        "predictions_sha256": sha256(predictions_path), "bundle_hash": canonical_hash(bundle),
        "status": "XAI_COMPUTED_REQUIRES_REVIEW",
        "randomization_requested": randomization,
        "optional_analysis_not_run": ["validation-fitted mixture replay"],
        "limitation": "Intervals condition on available nondegenerate maps; failures must accompany summaries"})
