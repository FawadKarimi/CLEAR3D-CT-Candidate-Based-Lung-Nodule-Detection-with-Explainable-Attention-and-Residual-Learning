"""Track 2 scan-to-mark pass.

Stage 1 is a fixed 96-voxel U-Net heatmap. It is not Wang's detector.
Training uses fit patients only, for a fixed epoch count. Confirmation is
scored once after that, and is not used to pick an epoch or a threshold.
The six reviewed patients are never opened.
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
import torch.nn as nn

ROOT = "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025"
sys.path.insert(0, ROOT + "/isolated")
sys.path.insert(0, "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src")
import phase_cf
from lidc3d.geometry import crop

PREP = phase_cf.PREP
ANN = phase_cf.ANN
GPU_UUID = phase_cf.GPU_UUID
SEED = 17
BLOCK = 96
STRIDE = 64
NMS_MM = 8.0
KEEP = 200
TISSUE = 0.20
RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SMOKE = os.environ.get("SMOKE") == "1"
RUN = ROOT + ("/runs/phase-track2/smoke" if SMOKE else "/runs/phase-track2")
EPOCHS = 1 if SMOKE else 4
WALL_S = 3 * 3600


class SpotNet(nn.Module):
    """Small heatmap net. Channels stay small so the 11 GB GPU can hold a 96-cube."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.c3 = nn.Sequential(nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.c4 = nn.Sequential(nn.Conv3d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.u3 = nn.Conv3d(64, 32, 3, padding=1)
        self.u2 = nn.Conv3d(48, 16, 3, padding=1)
        self.u1 = nn.Conv3d(24, 8, 3, padding=1)
        self.out = nn.Conv3d(8, 1, 1)

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        z = torch.nn.functional.interpolate(h4, size=h3.shape[-3:], mode="trilinear", align_corners=False)
        z = torch.relu(self.u3(torch.cat([z, h3], 1)))
        z = torch.nn.functional.interpolate(z, size=h2.shape[-3:], mode="trilinear", align_corners=False)
        z = torch.relu(self.u2(torch.cat([z, h2], 1)))
        z = torch.nn.functional.interpolate(z, size=h1.shape[-3:], mode="trilinear", align_corners=False)
        z = torch.relu(self.u1(torch.cat([z, h1], 1)))
        return self.out(z)


def gpu_free_mib():
    import subprocess
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in text.splitlines():
        uuid, free = [part.strip() for part in line.split(",")]
        if uuid == GPU_UUID:
            return int(free)
    raise RuntimeError("target GPU not listed")


def load_catalog():
    by_patient = defaultdict(list)
    for name in os.listdir(PREP):
        folder = os.path.join(PREP, name)
        if not os.path.isdir(folder):
            continue
        record = json.load(open(os.path.join(folder, "record.json")))
        record["folder"] = name
        by_patient[record["patient_id"]].append(record)
    return by_patient


def load_annotations():
    found = defaultdict(list)
    with open(ANN, newline="") as handle:
        for row in csv.DictReader(handle):
            found[row["seriesuid"]].append(
                (
                    float(row["coordX"]),
                    float(row["coordY"]),
                    float(row["coordZ"]),
                    float(row["diameter_mm"]),
                )
            )
    return found


def world_of(affine, zyx):
    point = affine @ np.array([zyx[0], zyx[1], zyx[2], 1.0], dtype=float)
    return point[:3]


def ijk_of(inverse, world):
    point = inverse @ np.array([world[0], world[1], world[2], 1.0], dtype=float)
    return point[:3]


def origins(length):
    if length <= BLOCK:
        return [0]
    values = list(range(0, length - BLOCK + 1, STRIDE))
    last = length - BLOCK
    if values[-1] != last:
        values.append(last)
    return values


def gaussian(points, size=BLOCK, sigma=3.0):
    target = np.zeros((size, size, size), dtype=np.float32)
    if not points:
        return target
    grid_z, grid_y, grid_x = np.ogrid[:size, :size, :size]
    for z_pos, y_pos, x_pos in points:
        ball = np.exp(
            -((grid_z - z_pos) ** 2 + (grid_y - y_pos) ** 2 + (grid_x - x_pos) ** 2)
            / (2.0 * sigma * sigma)
        )
        target = np.maximum(target, ball.astype(np.float32))
    return target


def far_from(affine, zyx, nodules, millimeters):
    here = world_of(affine, zyx)
    for nodule in nodules:
        if np.linalg.norm(here - world_of(affine, nodule)) < millimeters:
            return False
    return True


def examples_for_scan(image, affine, nodules, rng):
    inverse = np.linalg.inv(affine)
    centers = [ijk_of(inverse, nodule[:3]) for nodule in nodules]
    pairs = []
    for center in centers:
        jitter = rng.uniform(-8.0, 8.0, size=3)
        patch, start = crop(np.asarray(image), center + jitter, BLOCK)
        relative = center - np.asarray(start, dtype=float)
        pairs.append((patch.astype(np.float32), gaussian([relative])))
        for _ in range(30):
            guess = rng.uniform(0, np.array(image.shape))
            if far_from(affine, guess, centers, 20.0):
                negative, _ = crop(np.asarray(image), guess, BLOCK)
                pairs.append((negative.astype(np.float32), gaussian([])))
                break
    if centers:
        return pairs
    guess = rng.uniform(0, np.array(image.shape))
    negative, _ = crop(np.asarray(image), guess, BLOCK)
    pairs.append((negative.astype(np.float32), gaussian([])))
    return pairs


def run_epoch(model, optimizer, catalog, annotations, patients, device, rng, started):
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0], device=device))
    model.train()
    total = 0.0
    steps = 0
    batch_x = []
    batch_y = []

    def flush():
        nonlocal total, steps, batch_x, batch_y
        if not batch_x:
            return
        optimizer.zero_grad(set_to_none=True)
        inputs = phase_cf.zscore(np.stack(batch_x))
        tensor = torch.from_numpy(np.ascontiguousarray(inputs[:, None])).to(device)
        target = torch.from_numpy(np.ascontiguousarray(np.stack(batch_y)[:, None])).to(device)
        loss = loss_fn(model(tensor), target)
        if not torch.isfinite(loss.detach()):
            raise RuntimeError("non-finite detection loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach())
        steps += 1
        batch_x = []
        batch_y = []

    for number, patient in enumerate(patients, 1):
        if time.time() - started > WALL_S:
            break
        for record in catalog[patient]:
            nodules = annotations.get(record["series_uid"], [])
            image = np.load(os.path.join(PREP, record["folder"], "image.npy"), mmap_mode="r")
            geometry = json.load(open(os.path.join(PREP, record["folder"], "geometry.json")))
            affine = np.array(geometry["affine"], float)
            for patch, target in examples_for_scan(image, affine, nodules, rng):
                batch_x.append(patch)
                batch_y.append(target)
                if len(batch_x) == 2:
                    flush()
        if number % 50 == 0:
            print("TRAIN", number, len(patients), flush=True)
    flush()
    if steps == 0:
        raise RuntimeError("training epoch had no batches")
    return total / steps


def interior(index, origin, length):
    if index < 16 and origin > 0:
        return False
    if index >= BLOCK - 16 and origin + BLOCK < length:
        return False
    return True


def block_at(image, origin):
    out = np.zeros((BLOCK, BLOCK, BLOCK), dtype=np.float32)
    src = np.asarray(
        image[origin[0] : origin[0] + BLOCK, origin[1] : origin[1] + BLOCK, origin[2] : origin[2] + BLOCK]
    )
    out[: src.shape[0], : src.shape[1], : src.shape[2]] = src
    return out


def peaks_in_block(prob, origin, shape, affine):
    core = prob[1:-1, 1:-1, 1:-1]
    mask = core >= 0.02
    mask &= core >= prob[:-2, 1:-1, 1:-1]
    mask &= core >= prob[2:, 1:-1, 1:-1]
    mask &= core >= prob[1:-1, :-2, 1:-1]
    mask &= core >= prob[1:-1, 2:, 1:-1]
    mask &= core >= prob[1:-1, 1:-1, :-2]
    mask &= core >= prob[1:-1, 1:-1, 2:]
    found = []
    for z_pos, y_pos, x_pos in zip(*np.where(mask)):
        z_pos += 1
        y_pos += 1
        x_pos += 1
        if not (
            interior(z_pos, origin[0], shape[0])
            and interior(y_pos, origin[1], shape[1])
            and interior(x_pos, origin[2], shape[2])
        ):
            continue
        zyx = np.array([origin[0] + z_pos, origin[1] + y_pos, origin[2] + x_pos], dtype=float)
        found.append((float(prob[z_pos, y_pos, x_pos]), world_of(affine, zyx), zyx))
    return found


def suppress(peaks):
    peaks = sorted(peaks, key=lambda item: -item[0])
    kept = []
    for score, world, zyx in peaks:
        if any(np.linalg.norm(world - item[1]) < NMS_MM for item in kept):
            continue
        kept.append((score, world, zyx))
        if len(kept) == KEEP:
            break
    return kept


def detect_scan(model, image, affine, device):
    model.eval()
    found = []
    shape = image.shape
    with torch.no_grad():
        for z0 in origins(shape[0]):
            for y0 in origins(shape[1]):
                for x0 in origins(shape[2]):
                    block = block_at(image, (z0, y0, x0))
                    if float(block[::8, ::8, ::8].max()) < TISSUE:
                        continue
                    scored = phase_cf.zscore(block[None])
                    tensor = torch.from_numpy(np.ascontiguousarray(scored[:, None])).to(device)
                    prob = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
                    found.extend(peaks_in_block(prob, (z0, y0, x0), shape, affine))
    return suppress(found)


def stage2_scores(models, image, inverse, peaks, device):
    if not peaks:
        return {name: np.zeros(0, dtype=np.float32) for name in models}
    patches = []
    for _, world, _ in peaks:
        row = {"x": float(world[0]), "y": float(world[1]), "z": float(world[2])}
        patch, _ = phase_cf.take_patch(image, inverse, row)
        patches.append(patch)
    scored = phase_cf.zscore(np.stack(patches))
    tensor = torch.from_numpy(np.ascontiguousarray(scored[:, None])).to(device)
    output = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            parts = []
            for start in range(0, tensor.shape[0], 32):
                prob = torch.softmax(model(tensor[start : start + 32]).float(), 1)[:, 1]
                parts.append(prob.detach().cpu().numpy())
            output[name] = np.concatenate(parts)
    return output


def load_stage2(device):
    models = {
        "zuo": phase_cf.ZuoNet(),
        "wang": phase_cf.WangNet(),
        "clear": phase_cf.ClearNet(True),
        "ablation": phase_cf.ClearNet(False),
    }
    for name, model in models.items():
        state = torch.load(phase_cf.RUN + "/" + name + ".pt", map_location=device)
        model.load_state_dict(state["state"])
        model.to(device)
        model.eval()
    return models


def infer_patients(spot, stage2, catalog, annotations, patients, device, started):
    marks = []
    nodules = []
    scans = 0
    scans_by_patient = defaultdict(int)
    for number, patient in enumerate(patients, 1):
        if time.time() - started > WALL_S:
            raise RuntimeError("wall cap reached before inference finished")
        for record in catalog[patient]:
            scans += 1
            scans_by_patient[patient] += 1
            image = np.load(os.path.join(PREP, record["folder"], "image.npy"), mmap_mode="r")
            geometry = json.load(open(os.path.join(PREP, record["folder"], "geometry.json")))
            affine = np.array(geometry["affine"], float)
            inverse = np.linalg.inv(affine)
            for x_pos, y_pos, z_pos, diameter in annotations.get(record["series_uid"], []):
                nodules.append(
                    {
                        "patient": patient,
                        "scan": record["series_uid"],
                        "world": np.array([x_pos, y_pos, z_pos], dtype=float),
                        "radius": diameter / 2.0,
                    }
                )
            peaks = detect_scan(spot, image, affine, device)
            scores = stage2_scores(stage2, image, inverse, peaks, device)
            for index, (spot_score, world, _) in enumerate(peaks):
                marks.append(
                    {
                        "patient": patient,
                        "scan": record["series_uid"],
                        "world": world,
                        "spot": spot_score,
                        "zuo": float(scores["zuo"][index]),
                        "wang": float(scores["wang"][index]),
                        "clear": float(scores["clear"][index]),
                        "ablation": float(scores["ablation"][index]),
                    }
                )
        if number % 10 == 0:
            print("INFER", number, len(patients), "marks", len(marks), flush=True)
    return marks, nodules, scans, scans_by_patient


def curve(marks, nodules, n_scans, key):
    order = sorted(marks, key=lambda item: -item[key])
    used = set()
    fps = []
    sens = []
    tp = 0
    fp = 0
    n_nod = max(len(nodules), 1)
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
            tp += 1
        fps.append(fp / n_scans)
        sens.append(tp / n_nod)
    values = []
    for rate in RATES:
        best = 0.0
        for sensitivity, rate_now in zip(sens, fps):
            if rate_now <= rate + 1e-12:
                best = max(best, sensitivity)
        values.append(best)
    coverage = 0.0
    if nodules:
        hit = 0
        for nodule in nodules:
            for mark in marks:
                if mark["scan"] == nodule["scan"] and np.linalg.norm(mark["world"] - nodule["world"]) <= nodule["radius"]:
                    hit += 1
                    break
        coverage = hit / len(nodules)
    return {
        "sensitivity": values,
        "cpm": float(sum(values) / len(values)),
        "coverage_at_retained_marks": coverage,
        "marks": len(marks),
        "nodules": len(nodules),
        "scans": n_scans,
    }


def paired(marks, nodules, scans_by_patient, draws=200):
    rng = np.random.default_rng(SEED)
    by_mark = defaultdict(list)
    by_nodule = defaultdict(list)
    for mark in marks:
        by_mark[mark["patient"]].append(mark)
    for nodule in nodules:
        by_nodule[nodule["patient"]].append(nodule)
    patients = np.array(sorted(scans_by_patient))
    if len(patients) == 0:
        return {}
    out = {name: [] for name in ("clear_minus_zuo", "clear_minus_wang", "clear_minus_spot")}
    for _ in range(draws):
        sample = rng.choice(patients, size=len(patients), replace=True)
        copies_m = []
        copies_n = []
        n_scans = 0
        seen = defaultdict(int)
        for patient in sample:
            seen[patient] += 1
            replica = str(seen[patient])
            n_scans += scans_by_patient[patient]
            for mark in by_mark[patient]:
                copies_m.append({**mark, "scan": mark["scan"] + "#" + replica})
            for nodule in by_nodule[patient]:
                copies_n.append({**nodule, "scan": nodule["scan"] + "#" + replica})
        if n_scans == 0 or not copies_n:
            continue
        scores = {name: curve(copies_m, copies_n, n_scans, name)["cpm"] for name in ("clear", "zuo", "wang", "spot")}
        out["clear_minus_zuo"].append(scores["clear"] - scores["zuo"])
        out["clear_minus_wang"].append(scores["clear"] - scores["wang"])
        out["clear_minus_spot"].append(scores["clear"] - scores["spot"])
    summary = {}
    for name, values in out.items():
        arr = np.array(values, dtype=float)
        summary[name] = {
            "mean": float(arr.mean()) if len(arr) else None,
            "low": float(np.quantile(arr, 0.025)) if len(arr) else None,
            "high": float(np.quantile(arr, 0.975)) if len(arr) else None,
        }
    return summary


def assert_roundtrip(catalog, annotations, patient):
    record = catalog[patient][0]
    geometry = json.load(open(os.path.join(PREP, record["folder"], "geometry.json")))
    affine = np.array(geometry["affine"], float)
    inverse = np.linalg.inv(affine)
    world = np.array(annotations[record["series_uid"]][0][:3], dtype=float)
    back = world_of(affine, ijk_of(inverse, world))
    error = float(np.max(np.abs(back - world)))
    if error > 1e-3:
        raise RuntimeError("world coordinate roundtrip failed")
    print("ROUNDTRIP_MM", round(error, 6), flush=True)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    free = gpu_free_mib()
    print("GPU_FREE_MIB", free, flush=True)
    if free < 10000:
        raise SystemExit("GPU free memory below 10 GiB; not launching")
    os.makedirs(RUN, exist_ok=True)
    started = time.time()
    split = json.load(open(ROOT + "/runs/phase-cf/split.json"))
    reviewed = set(split["excluded_reviewed"])
    if reviewed & set(split["confirm"]) or reviewed & set(split["fit"]):
        raise RuntimeError("reviewed patients leaked into fit or confirmation")
    catalog = load_catalog()
    annotations = load_annotations()
    fit = split["fit"][:30] if SMOKE else split["fit"]
    assert_roundtrip(catalog, annotations, fit[0])
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    model = SpotNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(1, EPOCHS + 1):
        if time.time() - started > WALL_S:
            break
        loss = run_epoch(model, optimizer, catalog, annotations, fit, device, rng, started)
        history.append({"epoch": epoch, "loss": loss})
        print("EPOCH", epoch, round(loss, 4), flush=True)
    torch.save({"state": model.state_dict(), "history": history}, RUN + "/spotnet.pt")
    stage2 = load_stage2(device)
    if SMOKE:
        dev = split["dev"][:2]
        marks, nodules, scans, _ = infer_patients(model, stage2, catalog, annotations, dev, device, started)
        summary = {
            "smoke": True,
            "history": history,
            "dev_patients": dev,
            "marks": len(marks),
            "nodules": len(nodules),
            "scans": scans,
            "spot": curve(marks, nodules, max(scans, 1), "spot"),
        }
        json.dump(summary, open(RUN + "/summary.json", "w"), indent=2)
        print("SMOKE_DONE", len(marks), len(nodules), round(time.time() - started, 1), flush=True)
        return
    marks, nodules, scans, scans_by_patient = infer_patients(
        model, stage2, catalog, annotations, split["confirm"], device, started
    )
    summary = {
        "smoke": False,
        "seconds": time.time() - started,
        "history": history,
        "confirmation_patients": len(split["confirm"]),
        "scans": scans,
        "nodules": len(nodules),
        "marks": len(marks),
        "models": {name: curve(marks, nodules, scans, name) for name in ("spot", "zuo", "wang", "clear", "ablation")},
        "paired": paired(marks, nodules, scans_by_patient),
        "departures": [
            "Stage 1 is a small 96-cube U-Net, not Wang's cross-layer detector.",
            "Blocks are skipped when a coarse subsample stays below 0.20 on the stored -1000 to 400 normalization.",
            "At most 200 peaks are kept per scan after 8 mm suppression.",
            "A mark matches a nodule when the world distance is within the nodule radius.",
            "Sensitivity at each false-positive rate is the best point that does not exceed that rate.",
            "Stage 2 weights are the frozen Track 1 checkpoints. Their classification threshold is not reused.",
            "Epoch count is 4. It was not chosen from a detection score.",
        ],
    }
    json.dump(summary, open(RUN + "/summary.json", "w"), indent=2)
    np.savez_compressed(
        RUN + "/marks.npz",
        patient=np.array([item["patient"] for item in marks]),
        scan=np.array([item["scan"] for item in marks]),
        world=np.array([item["world"] for item in marks], dtype=np.float32) if marks else np.zeros((0, 3), np.float32),
        spot=np.array([item["spot"] for item in marks], dtype=np.float32),
        zuo=np.array([item["zuo"] for item in marks], dtype=np.float32),
        wang=np.array([item["wang"] for item in marks], dtype=np.float32),
        clear=np.array([item["clear"] for item in marks], dtype=np.float32),
        ablation=np.array([item["ablation"] for item in marks], dtype=np.float32),
    )
    print("DONE", round(summary["seconds"], 1), flush=True)
    for name, row in summary["models"].items():
        print("CPM", name, round(row["cpm"], 4), [round(v, 4) for v in row["sensitivity"]], "coverage", round(row["coverage_at_retained_marks"], 4), flush=True)


if __name__ == "__main__":
    main()
