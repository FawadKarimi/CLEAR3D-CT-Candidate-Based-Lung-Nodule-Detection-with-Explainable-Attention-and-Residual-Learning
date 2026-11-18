"""Finish the missing report from frozen scores.

This file does not train, does not load a checkpoint, and does not choose a
threshold. Confirmation predictions already on disk are read once for the
metrics the protocol asked to report. The six reviewed patients are not opened.
"""
import csv
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

ROOT = "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025"
PREP = "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared"
ANN = "/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1/annotations.csv"
REVIEW_RUNS = "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/runs"
CF = ROOT + "/runs/phase-cf"
T2 = ROOT + "/runs/phase-track2"
OUT = ROOT + "/runs/phase-finish"
REVIEWED = {
    "LIDC-IDRI-0020", "LIDC-IDRI-0428", "LIDC-IDRI-0634",
    "LIDC-IDRI-0722", "LIDC-IDRI-0865", "LIDC-IDRI-0895",
}
KINDS = ("zuo", "wang", "clear", "ablation")
RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
BINS = (("under 6 mm", 0.0, 6.0), ("6 to 10 mm", 6.0, 10.0), ("over 10 mm", 10.0, 1.0e9))
CROP = 32


def average_precision(score, label):
    label = np.asarray(label).astype(np.int64)
    positives = int((label == 1).sum())
    if positives == 0:
        return None
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    y = label[order]
    tp = np.cumsum(y == 1)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / positives
    previous = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - previous) * precision))


def roc_auc(score, label):
    label = np.asarray(label).astype(np.int64)
    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(np.asarray(score, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    sum_pos = ranks[label == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibration(score, label, bins=10):
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    n = len(score)
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (score >= low) & (score <= high)
        else:
            mask = (score >= low) & (score < high)
        count = int(mask.sum())
        if count == 0:
            rows.append({"low": float(low), "high": float(high), "n": 0})
            continue
        confidence = float(score[mask].mean())
        observed = float(label[mask].mean())
        ece += (count / n) * abs(observed - confidence)
        rows.append({
            "low": float(low),
            "high": float(high),
            "n": count,
            "mean_score": confidence,
            "observed_positive_rate": observed,
        })
    brier = float(np.mean((score - label) ** 2))
    return {"bins": rows, "ece": ece, "brier": brier}


def bin_name(diameter):
    for name, low, high in BINS:
        if low <= diameter < high:
            return name
    return "unmatched"


def read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_annotations():
    found = defaultdict(list)
    with open(ANN, newline="") as handle:
        for row in csv.DictReader(handle):
            found[row["seriesuid"]].append((
                float(row["coordX"]), float(row["coordY"]), float(row["coordZ"]), float(row["diameter_mm"]),
            ))
    return found


def load_catalog():
    catalog = []
    for name in os.listdir(PREP):
        folder = os.path.join(PREP, name)
        if not os.path.isdir(folder):
            continue
        record = json.load(open(os.path.join(folder, "record.json")))
        record["folder"] = name
        catalog.append(record)
    return catalog


def match_diameter(row, annotations):
    best = None
    best_distance = None
    for x_pos, y_pos, z_pos, diameter in annotations.get(row["series"], []):
        distance = float(np.linalg.norm(np.array([row["x"], row["y"], row["z"]]) - np.array([x_pos, y_pos, z_pos])))
        if best_distance is None or distance < best_distance:
            best = (distance, diameter, np.array([x_pos, y_pos, z_pos]))
            best_distance = distance
    if best is None:
        return None
    distance, diameter, _ = best
    if distance <= max(diameter / 2.0, 1.0):
        return diameter
    return None


def center_status(image_shape, inverse, row):
    ijk = inverse @ np.array([row["x"], row["y"], row["z"], 1.0])
    zyx = np.array([float(ijk[0]), float(ijk[1]), float(ijk[2])])
    center = np.floor(zyx + 0.5).astype(int)
    shape = np.array(image_shape)
    inside = bool(np.all(center >= 0) and np.all(center < shape))
    if not inside:
        return "outside"
    half = CROP // 2
    start = center - half
    stop = center + half
    if np.any(start < 0) or np.any(stop > shape):
        return "padded"
    return "interior"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slice_signature(image, affine):
    digest = hashlib.sha256()
    digest.update(np.array(image.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(affine, dtype=np.float64).tobytes())
    mid = int(image.shape[0] // 2)
    for index in (0, mid, int(image.shape[0]) - 1):
        digest.update(np.ascontiguousarray(image[index]).tobytes())
    return digest.hexdigest()


def hash_large_checkpoints():
    found = []
    if not os.path.isdir(REVIEW_RUNS):
        return [{"error": "review run directory is absent", "path": REVIEW_RUNS}]
    for dirpath, _, names in os.walk(REVIEW_RUNS):
        for name in names:
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if name == "banks.npz" or (name.endswith(".pt") and size >= 100_000_000):
                found.append((path, size))
    found.sort()
    rows = []
    for path, size in found:
        rows.append({
            "path": path,
            "bytes": size,
            "sha256": sha256_file(path),
            "deserialized": False,
        })
        print("HASH", size, path, flush=True)
    return rows


def classification_report(annotations):
    summary = json.load(open(CF + "/summary.json"))
    report = {}
    for split, filename in (("development", "dev.jsonl"), ("confirmation", "confirm.jsonl")):
        rows = read_jsonl(CF + "/" + filename)
        block = {"n": len(rows), "patients": len({row["patient"] for row in rows})}
        positives = [row for row in rows if row["label"] == 1]
        block["prevalence"] = len(positives) / max(len(rows), 1)
        diameters = [match_diameter(row, annotations) for row in positives]
        block["positive_rows"] = len(positives)
        block["positive_rows_matched"] = sum(item is not None for item in diameters)
        series_ids = {row["series"] for row in rows}
        references = []
        for series in series_ids:
            for item in annotations.get(series, []):
                references.append(item[3])
        block["reference_nodules"] = len(references)
        block["models"] = {}
        for kind in KINDS:
            stored = summary["models"][kind]["dev"] if split == "development" else summary["confirm"][kind]
            pack = np.load(CF + "/" + kind + "-" + ("dev" if split == "development" else "confirm") + "-scores.npz")
            score = pack["score"]
            label = pack["label"]
            if len(score) != len(rows) or not np.array_equal(label, np.array([row["label"] for row in rows])):
                raise RuntimeError("score rows do not match " + filename + " for " + kind)
            auc = roc_auc(score, label)
            if abs(auc - stored["auc"]) > 1e-6:
                raise RuntimeError("roc recomputation does not match the frozen summary for " + kind)
            cut = float(stored["threshold"])
            pred = score >= cut
            model_row = {
                "threshold_unchanged": cut,
                "roc_auc": auc,
                "pr_auc": average_precision(score, label),
                "calibration": calibration(score, label),
                "subgroups": {},
            }
            label_arr = np.array([row["label"] for row in rows])
            positive_index = np.flatnonzero(label_arr == 1)
            for name, _, _ in BINS:
                model_row["subgroups"][name] = {"positive_rows": 0, "true_positives": 0, "false_negatives": 0}
            model_row["subgroups"]["unmatched"] = {"positive_rows": 0, "true_positives": 0, "false_negatives": 0}
            for offset, diameter in enumerate(diameters):
                name = "unmatched" if diameter is None else bin_name(diameter)
                slot = model_row["subgroups"][name]
                slot["positive_rows"] += 1
                index = int(positive_index[offset])
                if pred[index]:
                    slot["true_positives"] += 1
                else:
                    slot["false_negatives"] += 1
            for slot in model_row["subgroups"].values():
                slot["recall"] = slot["true_positives"] / slot["positive_rows"] if slot["positive_rows"] else None
            block["models"][kind] = model_row
            print("METRIC", split, kind, "pr", round(model_row["pr_auc"], 4), "ece", round(model_row["calibration"]["ece"], 4), flush=True)
        block["reference_bins"] = {name: 0 for name, _, _ in BINS}
        for diameter in references:
            block["reference_bins"][bin_name(diameter)] += 1
        report[split] = block
    return report


def boundary_and_duplicates(catalog, annotations):
    split = json.load(open(CF + "/split.json"))
    role_of = {}
    for role in ("fit", "dev", "confirm"):
        for patient in split[role]:
            role_of[patient] = role
    for patient in REVIEWED:
        role_of[patient] = "excluded"
    by_series = {}
    for record in catalog:
        by_series[record["series_uid"]] = record
    counts = defaultdict(lambda: defaultdict(int))
    signatures = defaultdict(list)
    series_by_role = defaultdict(set)
    candidate_keys = defaultdict(set)
    duplicate_rows = defaultdict(int)
    rows_by_role = defaultdict(int)
    for role, filename in (("fit", "fit.jsonl"), ("dev", "dev.jsonl"), ("confirm", "confirm.jsonl")):
        grouped = defaultdict(list)
        for row in read_jsonl(CF + "/" + filename):
            grouped[row["scan"]].append(row)
            rows_by_role[role] += 1
            key = (row["series"], round(row["x"], 3), round(row["y"], 3), round(row["z"], 3), row["label"])
            if key in candidate_keys[role]:
                duplicate_rows[role] += 1
            candidate_keys[role].add(key)
        for scan, rows in grouped.items():
            record = by_series[rows[0]["series"]]
            if record["patient_id"] in REVIEWED:
                raise RuntimeError("reviewed patient appeared in a scored manifest")
            image = np.load(os.path.join(PREP, scan, "image.npy"), mmap_mode="r")
            geometry = json.load(open(os.path.join(PREP, scan, "geometry.json")))
            affine = np.array(geometry["affine"], dtype=float)
            inverse = np.linalg.inv(affine)
            series_by_role[role].add(record["series_uid"])
            signatures[slice_signature(image, affine)].append({
                "patient": record["patient_id"],
                "series": record["series_uid"],
                "role": role,
                "folder": scan,
            })
            for row in rows:
                counts[role][center_status(image.shape, inverse, row)] += 1
        print("SCANNED", role, len(grouped), flush=True)
    excluded_series = set()
    for record in catalog:
        if record["patient_id"] not in REVIEWED:
            continue
        excluded_series.add(record["series_uid"])
        image = np.load(os.path.join(PREP, record["folder"], "image.npy"), mmap_mode="r")
        geometry = json.load(open(os.path.join(PREP, record["folder"], "geometry.json")))
        signatures[slice_signature(image, np.array(geometry["affine"], dtype=float))].append({
            "patient": record["patient_id"],
            "series": record["series_uid"],
            "role": "excluded",
            "folder": record["folder"],
        })
    collisions = []
    cross = []
    signature_groups = 0
    for members in signatures.values():
        folders = {item["folder"] for item in members}
        roles = {item["role"] for item in members}
        if len(folders) < 2:
            continue
        signature_groups += 1
        full = []
        for item in members:
            path = os.path.join(PREP, item["folder"], "image.npy")
            full.append({**item, "sha256": sha256_file(path)})
        digest = {item["sha256"] for item in full}
        if len(digest) < len(full):
            collisions.append(full)
        if len(roles) > 1:
            cross.append([{"patient": item["patient"], "role": item["role"], "series": item["series"], "sha256": item["sha256"]} for item in full])
    bank = np.load(CF + "/fit-bank.npz")
    y = bank["y"]
    patients = bank["patient"]
    reference_by_role = {}
    for role, series_ids in series_by_role.items():
        reference_by_role[role] = sum(len(annotations.get(series, [])) for series in series_ids)
    return {
        "crop_padding": {role: dict(counts[role]) for role in ("fit", "dev", "confirm")},
        "candidate_rows": dict(rows_by_role),
        "unique_coordinate_keys": {role: len(candidate_keys[role]) for role in candidate_keys},
        "repeated_coordinate_rows": dict(duplicate_rows),
        "reference_nodules": reference_by_role,
        "series_overlap": {
            "fit_dev": len(series_by_role["fit"] & series_by_role["dev"]),
            "fit_confirm": len(series_by_role["fit"] & series_by_role["confirm"]),
            "dev_confirm": len(series_by_role["dev"] & series_by_role["confirm"]),
            "excluded_with_scored": len(excluded_series & (series_by_role["fit"] | series_by_role["dev"] | series_by_role["confirm"])),
        },
        "slice_signature_groups_with_two_folders": signature_groups,
        "full_hash_collisions": collisions,
        "cross_role_slice_matches": cross,
        "fit_bank_rows": int(len(y)),
        "fit_bank_positive_rows": int((y == 1).sum()),
        "fit_bank_negative_rows": int((y == 0).sum()),
        "fit_bank_unique_patients": int(len(set(patients.tolist()))),
        "sampling_note": "The fit bank stores each drawn candidate once. Epochs then draw those rows again with replacement. Repeated draws are not extra nodules.",
    }


def any_mark_coverage(marks, nodules):
    if not nodules:
        return 0.0
    hit = 0
    for nodule in nodules:
        for mark in marks:
            if mark["scan"] == nodule["scan"] and float(np.linalg.norm(mark["world"] - nodule["world"])) <= nodule["radius"]:
                hit += 1
                break
    return hit / len(nodules)


def detection_report(catalog, annotations):
    split = json.load(open(CF + "/split.json"))
    summary = json.load(open(T2 + "/summary.json"))
    confirm = set(split["confirm"])
    nodules = []
    scans = 0
    for record in catalog:
        if record["patient_id"] not in confirm:
            continue
        scans += 1
        for x_pos, y_pos, z_pos, diameter in annotations.get(record["series_uid"], []):
            nodules.append({
                "patient": record["patient_id"],
                "scan": record["series_uid"],
                "world": np.array([x_pos, y_pos, z_pos], dtype=np.float64),
                "radius": diameter / 2.0,
                "diameter": diameter,
            })
    pack = np.load(T2 + "/marks.npz")
    marks = []
    world = pack["world"]
    for index in range(len(world)):
        marks.append({
            "patient": str(pack["patient"][index]),
            "scan": str(pack["scan"][index]),
            "world": world[index].astype(np.float64),
            "spot": float(pack["spot"][index]),
            "zuo": float(pack["zuo"][index]),
            "wang": float(pack["wang"][index]),
            "clear": float(pack["clear"][index]),
            "ablation": float(pack["ablation"][index]),
        })
    models = {}
    for key in ("spot", "zuo", "wang", "clear", "ablation"):
        curve = froc(marks, nodules, scans, key)
        stored = summary["models"][key]["cpm"]
        curve["stored_cpm"] = stored
        curve["cpm_matches_stored"] = abs(curve["cpm"] - stored) < 1e-3
        curve["any_mark_coverage"] = any_mark_coverage(marks, nodules)
        curve["stored_coverage"] = summary["models"][key]["coverage_at_retained_marks"]
        curve["coverage_matches_stored"] = abs(curve["any_mark_coverage"] - curve["stored_coverage"]) < 1e-3
        models[key] = curve
        print("FROC", key, round(curve["cpm"], 4), "match", curve["cpm_matches_stored"], flush=True)
    return {"scans": scans, "nodules": len(nodules), "marks": len(marks), "models": models, "seconds": summary.get("seconds")}


def froc(marks, nodules, n_scans, key):
    order = sorted(marks, key=lambda item: -item[key])
    used = set()
    first_fp = [None] * len(nodules)
    fp = 0
    for mark in order:
        best = None
        best_distance = None
        for index, nodule in enumerate(nodules):
            if index in used or nodule["scan"] != mark["scan"]:
                continue
            distance = float(np.linalg.norm(mark["world"] - nodule["world"]))
            if distance <= nodule["radius"] and (best_distance is None or distance < best_distance):
                best = index
                best_distance = distance
        if best is None:
            fp += 1
        else:
            used.add(best)
            first_fp[best] = fp
    n_nod = max(len(nodules), 1)
    sensitivity = []
    for rate in RATES:
        hits = sum(item is not None and item / n_scans <= rate + 1e-12 for item in first_fp)
        sensitivity.append(hits / n_nod)
    subgroups = {}
    for name, low, high in BINS:
        indexes = [i for i, nodule in enumerate(nodules) if low <= nodule["diameter"] < high]
        slot = {"nodules": len(indexes)}
        slot["missed_by_candidate_generation"] = sum(first_fp[i] is None for i in indexes)
        slot["sensitivity"] = []
        for rate in RATES:
            if not indexes:
                slot["sensitivity"].append(None)
                continue
            hits = sum(first_fp[i] is not None and first_fp[i] / n_scans <= rate + 1e-12 for i in indexes)
            slot["sensitivity"].append(hits / len(indexes))
        subgroups[name] = slot
    return {
        "sensitivity": sensitivity,
        "cpm": float(sum(sensitivity) / len(sensitivity)),
        "coverage": sum(item is not None for item in first_fp) / n_nod,
        "missed": sum(item is None for item in first_fp),
        "subgroups": subgroups,
        "froc_points": len(order),
    }


def write_markdown(payload):
    lines = []
    add = lines.append
    add("# Completion report, 2026-10-25")
    add("")
    add("These numbers are computed from the frozen score files. No threshold, epoch, or checkpoint was changed. Confirmation was not scored again.")
    add("")
    add("## Track 1, added metrics")
    add("")
    add("PR-AUC is average precision. Calibration uses ten equal-width bins of the predicted positive probability. ECE is the count-weighted absolute gap between each bin's mean score and its observed positive rate. Brier is the mean squared error of that probability. The threshold is the one already stored for each model.")
    add("")
    for split in ("development", "confirmation"):
        block = payload["classification"][split]
        add("### " + split)
        add("")
        add("Candidates %d, patients %d, prevalence %.6f, reference nodules %d, positive candidate rows %d, of which %d match a reference within its radius." % (
            block["n"], block["patients"], block["prevalence"], block["reference_nodules"], block["positive_rows"], block["positive_rows_matched"],
        ))
        add("")
        add("| model | PR-AUC | ROC-AUC | ECE | Brier | threshold |")
        add("| --- | ---: | ---: | ---: | ---: | ---: |")
        for kind in KINDS:
            row = block["models"][kind]
            add("| %s | %.4f | %.4f | %.4f | %.6f | %.4f |" % (
                kind, row["pr_auc"], row["roc_auc"], row["calibration"]["ece"], row["calibration"]["brier"], row["threshold_unchanged"],
            ))
        add("")
        add("Positive-candidate recall at that same threshold, by reference diameter. A row with no reference inside the radius is unmatched. This is not a new endpoint and it is not a subgroup superiority test.")
        add("")
        add("| model | under 6 mm | 6 to 10 mm | over 10 mm | unmatched |")
        add("| --- | ---: | ---: | ---: | ---: |")
        for kind in KINDS:
            groups = block["models"][kind]["subgroups"]
            cells = []
            for name in ("under 6 mm", "6 to 10 mm", "over 10 mm", "unmatched"):
                slot = groups[name]
                if slot["positive_rows"] == 0 or slot["recall"] is None:
                    cells.append("n=0")
                else:
                    cells.append("%.3f (%d/%d)" % (slot["recall"], slot["true_positives"], slot["positive_rows"]))
            add("| %s | %s | %s | %s | %s |" % (kind, cells[0], cells[1], cells[2], cells[3]))
        add("")
        add("Reference nodules on these scans: " + ", ".join(
            "%s %d" % (name, block["reference_bins"][name]) for name, _, _ in BINS
        ) + ".")
        add("")
    add("## Sampling and boundaries")
    add("")
    account = payload["accounting"]
    add("Fit bank rows %d, positive %d, negative %d, unique patients %d." % (
        account["fit_bank_rows"], account["fit_bank_positive_rows"], account["fit_bank_negative_rows"], account["fit_bank_unique_patients"],
    ))
    add(account["sampling_note"])
    add("")
    add("| role | candidate rows | unique coordinates | repeated coordinates | reference nodules | interior crops | padded crops | outside centers |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for role in ("fit", "dev", "confirm"):
        pad = account["crop_padding"].get(role, {})
        add("| %s | %d | %d | %d | %d | %d | %d | %d |" % (
            role,
            account["candidate_rows"].get(role, 0),
            account["unique_coordinate_keys"].get(role, 0),
            account["repeated_coordinate_rows"].get(role, 0),
            account["reference_nodules"].get(role, 0),
            pad.get("interior", 0),
            pad.get("padded", 0),
            pad.get("outside", 0),
        ))
    add("")
    overlap = account["series_overlap"]
    add("Series overlap is fit/dev %d, fit/confirm %d, dev/confirm %d, excluded-with-scored %d." % (
        overlap["fit_dev"], overlap["fit_confirm"], overlap["dev_confirm"], overlap["excluded_with_scored"],
    ))
    add("Slice-signature groups that contain two folders: %d. Cross-role slice matches: %d." % (
        account["slice_signature_groups_with_two_folders"], len(account["cross_role_slice_matches"]),
    ))
    add("A slice signature is the shape, the affine, and three axial slices. Full-file hashes were computed only when two folders shared a signature.")
    add("")
    add("## Track 2, FROC from the saved marks")
    add("")
    det = payload["detection"]
    add("Recomputed from marks.npz and the annotations. Scans %d, nodules %d, marks %d. Wall time stored with the run: %s seconds. Peak VRAM was not sampled during that run, so no memory figure is added here." % (
        det["scans"], det["nodules"], det["marks"], det["seconds"],
    ))
    add("")
    add("| model | CPM | matches stored | coverage | missed | sensitivities at 0.125, 0.25, 0.5, 1, 2, 4, 8 |")
    add("| --- | ---: | --- | ---: | ---: | --- |")
    for key in ("spot", "zuo", "wang", "clear", "ablation"):
        row = det["models"][key]
        sens = ", ".join("%.4f" % value for value in row["sensitivity"])
        add("| %s | %.4f | %s | %.4f | %d | %s |" % (
            key, row["cpm"], "yes" if row["cpm_matches_stored"] else "no", row["any_mark_coverage"], row["missed"], sens,
        ))
    add("")
    add("Sensitivity inside each diameter bin uses the same global false-positive count. Bins are not separate operating points.")
    add("")
    for key in ("spot", "zuo", "wang", "clear", "ablation"):
        add("### " + key)
        add("")
        add("| bin | nodules | missed before any mark | sensitivity at the seven rates |")
        add("| --- | ---: | ---: | --- |")
        for name, _, _ in BINS:
            slot = det["models"][key]["subgroups"][name]
            sens = ", ".join("n/a" if value is None else "%.3f" % value for value in slot["sensitivity"])
            add("| %s | %d | %d | %s |" % (name, slot["nodules"], slot["missed_by_candidate_generation"], sens))
        add("")
    add("## Hashes")
    add("")
    add("SHA-256 of the large October banks and of every checkpoint at least 100 MB under the October run directory. The files were not deserialized and were not copied.")
    add("")
    if not payload["hashes"]:
        add("No matching file was found.")
    for item in payload["hashes"]:
        if "sha256" not in item:
            add("- " + item.get("error", "missing") + ": " + item.get("path", ""))
            continue
        add("- `%s`" % item["sha256"])
        add("  %s (%d bytes)" % (item["path"], item["bytes"]))
    add("")
    add("## What this pass does not change")
    add("")
    add("- The published Wang stage-1 detector and the paper's anisotropic 32x48x48, 16x24x24, and 8x12x12 stage-2 inputs were not trained. Building that network after the confirmation read would be a new method, and the confirmation set is closed.")
    add("- No vessel or airway label exists on these volumes. None was invented. Hard-negative weights in the finished run remain the training-bank losses after epoch 2.")
    add("- The finished runs stored a state dict and a threshold. Their optimizer and RNG state were not kept, so those files are not resume sources. New training was not started, so a replacement checkpoint was not written over them.")
    add("- Learning rate stayed at the values fixed before the first development score. Focal loss was not added. Either change would select a new configuration after confirmation.")
    add("- Intensity augmentation beyond the training-only flips was not applied to the frozen models.")
    add("- CLEAR3D remains below both reproductions on the frozen confirmation endpoint. This report does not replace that result.")
    add("")
    text = "\n".join(lines) + "\n"
    open(OUT + "/COMPLETION.md", "w").write(text)
    open(ROOT + "/COMPLETION.md", "w").write(text)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("START", flush=True)
    annotations = load_annotations()
    catalog = load_catalog()
    payload = {
        "classification": classification_report(annotations),
        "accounting": boundary_and_duplicates(catalog, annotations),
        "detection": detection_report(catalog, annotations),
        "hashes": hash_large_checkpoints(),
    }
    json.dump(payload, open(OUT + "/report.json", "w"))
    write_markdown(payload)
    decisions = ROOT + "/DECISIONS.md"
    marker = "completion_report_20261025"
    text = open(decisions).read()
    if marker not in text:
        with open(decisions, "a") as handle:
            handle.write("\n%s: metrics, hashes, duplicate signatures, and boundary counts were computed from frozen files. Confirmation was not rescored. No threshold was changed.\n" % marker)
    print("WROTE", OUT + "/COMPLETION.md", flush=True)


if __name__ == "__main__":
    main()
