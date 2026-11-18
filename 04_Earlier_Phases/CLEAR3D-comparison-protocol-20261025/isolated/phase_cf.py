"""Shared false-positive-reduction comparison.

Development uses prepared train patients only. Confirmation is the prepared
validation patients except the six already scored. This file never treats the
redesign-001 named test as untouched.
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

sys.path.insert(0, "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/src")
from lidc3d.geometry import crop

PREP = "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared"
CAND = "/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1/candidates_V2.csv"
ANN = "/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1/annotations.csv"
ROOT = "/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025"
RUN = ROOT + "/runs/phase-cf"
REVIEWED = {
    "LIDC-IDRI-0020", "LIDC-IDRI-0428", "LIDC-IDRI-0634",
    "LIDC-IDRI-0722", "LIDC-IDRI-0865", "LIDC-IDRI-0895",
}
GPU_INDEX = "1"
GPU_UUID = "GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90"
EPOCHS = 1 if os.environ.get("SMOKE") == "1" else 8
NEG_CAP = {"zuo": 7, "wang": 3, "clear": 7, "ablation": 7}
SEED = 17


def positive_f1(tp, fp, fn):
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * precision * recall / (precision + recall)


def confusion(score, label, cut):
    pred = score >= cut
    truth = label == 1
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / max(len(label), 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": spec, "accuracy": acc,
        "f1": positive_f1(tp, fp, fn), "n": int(len(label)),
        "positives": int(truth.sum()), "negatives": int((~truth).sum()),
    }


def roc_auc(score, label):
    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    sum_pos = ranks[label == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def choose_cut(score, label):
    grid = np.quantile(score, np.linspace(0, 1, 201))
    best = None
    for cut in np.unique(grid):
        row = confusion(score, label, float(cut))
        rank = (row["f1"], float(cut))
        if best is None or rank > best[0]:
            best = (rank, float(cut), row)
    return best[1], best[2]


class ZuoNet(nn.Module):
    """Compact embedded multi-branch 3D net attributed to Zuo et al. 2020.

    Departure: three depth branches with global average pooling, not the
    paper's exact pool strides. Convolution bias starts at 0.2.
    """

    def __init__(self):
        super().__init__()
        self.s1 = nn.Conv3d(1, 32, 3, padding=1)
        self.s2 = nn.Conv3d(32, 64, 3, padding=1)
        self.s3 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.fc = nn.Linear(32 + 64 + 128, 2)
        for layer in (self.s1, self.s2, self.s3):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.constant_(layer.bias, 0.2)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x[:, :, 7:25, 7:25, 7:25]
        h1 = torch.relu(self.s1(x))
        h2 = torch.relu(self.s2(self.pool(h1)))
        h3 = torch.relu(self.s3(self.pool(h2)))
        parts = [h.mean(dim=(2, 3, 4)) for h in (h1, h2, h3)]
        return self.fc(torch.cat(parts, 1))


class WangNet(nn.Module):
    """Three-scale stage-2 classifier attributed to Wang et al. 2026.

    Departure: cubic 32, 16, and 8 voxel views from one center, not the
    paper's 32x48x48, 16x24x24, and 8x12x12 patches.
    """

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList()
        for _ in range(3):
            self.branches.append(nn.Sequential(
                nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            ))
        self.fuse = nn.Linear(96, 32)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        views = (x, x[:, :, 8:24, 8:24, 8:24], x[:, :, 12:20, 12:20, 12:20])
        parts = [branch(view).mean(dim=(2, 3, 4)) for branch, view in zip(self.branches, views)]
        return self.fc(torch.relu(self.fuse(torch.cat(parts, 1))))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.c2(torch.relu(self.c1(x))))


class ClearNet(nn.Module):
    """Fine branch plus context branch. fuse=False is the ablation."""

    def __init__(self, fuse):
        super().__init__()
        self.fuse_on = fuse
        self.fine = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(inplace=True), ResBlock(16))
        self.context = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            ResBlock(16),
        )
        self.mix = nn.Linear(32, 16) if fuse else None
        self.fc = nn.Linear(16 if fuse else 32, 2)

    def forward(self, x):
        fine = self.fine(x[:, :, 8:24, 8:24, 8:24]).mean(dim=(2, 3, 4))
        context = self.context(x).mean(dim=(2, 3, 4))
        if self.fuse_on:
            hidden = torch.relu(self.mix(torch.cat([fine, context], 1)))
            return self.fc(hidden)
        return self.fc(torch.cat([fine, context], 1))


def build_models():
    return {
        "zuo": ZuoNet(),
        "wang": WangNet(),
        "clear": ClearNet(True),
        "ablation": ClearNet(False),
    }


def load_records():
    series = {}
    patients = defaultdict(list)
    for name in os.listdir(PREP):
        folder = os.path.join(PREP, name)
        if not os.path.isdir(folder):
            continue
        record = json.load(open(os.path.join(folder, "record.json")))
        series[record["series_uid"]] = record
        patients[record["patient_id"]].append(record["series_uid"])
    return series, patients


def assign_roles(patients, series):
    train_patients = sorted(pid for pid, uids in patients.items() if series[uids[0]].get("split") == "train")
    val_patients = sorted(pid for pid, uids in patients.items() if series[uids[0]].get("split") == "val")
    confirmation = [pid for pid in val_patients if pid not in REVIEWED]
    rng = np.random.default_rng(SEED)
    order = np.array(train_patients)
    rng.shuffle(order)
    if os.environ.get("SMOKE") == "1":
        fit, dev = order[:40].tolist(), order[40:50].tolist()
    else:
        cut = int(round(0.8 * len(order)))
        fit, dev = order[:cut].tolist(), order[cut:].tolist()
    role = {}
    for pid in fit:
        role[pid] = "fit"
    for pid in dev:
        role[pid] = "dev"
    for pid in confirmation:
        role[pid] = "confirm"
    for pid in REVIEWED:
        role[pid] = "excluded"
    payload = {
        "seed": SEED,
        "fit": fit,
        "dev": dev,
        "confirm": confirmation,
        "excluded_reviewed": sorted(REVIEWED),
        "rule": "Prepared train patients, seed 17, 80/20 fit/dev. Confirmation is prepared val except the six reviewed patients. Reviewed six are in neither.",
    }
    os.makedirs(RUN, exist_ok=True)
    json.dump(payload, open(RUN + "/split.json", "w"), indent=2)
    print("SPLIT", {key: len(payload[key]) for key in ("fit", "dev", "confirm", "excluded_reviewed")}, flush=True)
    return role


def index_candidates(series, role_of):
    buckets = {name: open(RUN + "/" + name + ".jsonl", "w") for name in ("fit", "dev", "confirm")}
    counts = defaultdict(int)
    outside = 0
    with open(CAND, newline="") as handle:
        for row in csv.DictReader(handle):
            record = series.get(row["seriesuid"])
            if record is None:
                counts["absent_series"] += 1
                continue
            role = role_of.get(record["patient_id"])
            if role in (None, "excluded"):
                counts["excluded"] += 1
                continue
            label = int(row["class"])
            if label not in (0, 1):
                counts["other_class"] += 1
                continue
            item = {
                "series": row["seriesuid"],
                "scan": record["scan_id"],
                "patient": record["patient_id"],
                "x": float(row["coordX"]),
                "y": float(row["coordY"]),
                "z": float(row["coordZ"]),
                "label": label,
            }
            buckets[role].write(json.dumps(item) + "\n")
            counts[role + ":" + str(label)] += 1
    for handle in buckets.values():
        handle.close()
    json.dump(counts, open(RUN + "/index-counts.json", "w"), indent=2)
    print("INDEX", dict(counts), "outside_placeholder", outside, flush=True)
    return counts


def read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def patient_weights(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["patient"]] += 1
    weights = np.array([1.0 / counts[row["patient"]] for row in rows], dtype=np.float64)
    weights /= weights.sum()
    return weights


def zscore(batch):
    flat = batch.reshape(batch.shape[0], -1)
    mean = flat.mean(axis=1)[:, None, None, None]
    std = flat.std(axis=1)[:, None, None, None]
    return (batch - mean) / (std + 1e-6)


def take_patch(image, affine_inv, row):
    ijk = affine_inv @ np.array([row["x"], row["y"], row["z"], 1.0])
    zyx = [float(ijk[0]), float(ijk[1]), float(ijk[2])]
    center = np.floor(np.asarray(zyx) + 0.5).astype(int)
    inside = bool(np.all(center >= 0) and np.all(center < np.array(image.shape)))
    patch, _ = crop(np.asarray(image), zyx, 32)
    return patch.astype(np.float32), inside


def extract_fit(role_name, neg_per_pos):
    rows = read_jsonl(RUN + "/" + role_name + ".jsonl")
    pos = [row for row in rows if row["label"] == 1]
    neg = [row for row in rows if row["label"] == 0]
    rng = np.random.default_rng(SEED)
    need = min(len(neg), neg_per_pos * max(len(pos), 1))
    choice = rng.choice(len(neg), size=need, replace=False) if need else []
    chosen = pos + [neg[int(i)] for i in choice]
    by_scan = defaultdict(list)
    for index, row in enumerate(chosen):
        by_scan[row["scan"]].append(index)
    patches = np.zeros((len(chosen), 32, 32, 32), dtype=np.float32)
    labels = np.zeros(len(chosen), dtype=np.uint8)
    kept = np.zeros(len(chosen), dtype=bool)
    skipped = 0
    for number, (scan, indexes) in enumerate(by_scan.items(), 1):
        image = np.load(os.path.join(PREP, scan, "image.npy"), mmap_mode="r")
        geometry = json.load(open(os.path.join(PREP, scan, "geometry.json")))
        inverse = np.linalg.inv(np.array(geometry["affine"], float))
        for index in indexes:
            patch, inside = take_patch(image, inverse, chosen[index])
            if not inside or patch.shape != (32, 32, 32):
                skipped += 1
                continue
            patches[index] = patch
            labels[index] = chosen[index]["label"]
            kept[index] = True
        if number % 50 == 0:
            print("EXTRACT", role_name, number, len(by_scan), flush=True)
    bank = {
        "x": patches[kept],
        "y": labels[kept],
        "patient": np.array([chosen[i]["patient"] for i in range(len(chosen)) if kept[i]]),
    }
    rows = [{"patient": pid} for pid in bank["patient"]]
    bank["w"] = patient_weights(rows)
    np.savez_compressed(RUN + "/fit-bank.npz", x=bank["x"], y=bank["y"], patient=bank["patient"])
    print("BANK", role_name, "kept", int(kept.sum()), "skipped_outside", skipped, "pos", int((bank["y"] == 1).sum()), flush=True)
    return bank


def load_bank(name):
    data = np.load(RUN + "/" + name + "-bank.npz")
    rows = [{"patient": str(pid)} for pid in data["patient"]]
    return {"x": data["x"], "y": data["y"].astype(np.int64), "w": patient_weights(rows)}


def batch_from(bank, indexes, augment, rng):
    x = bank["x"][indexes].astype(np.float32)
    if augment:
        for axis in (1, 2, 3):
            if rng.random() < 0.5:
                x = np.flip(x, axis=axis)
        x = np.ascontiguousarray(x)
    x = zscore(x)
    tensor = torch.from_numpy(x[:, None])
    target = torch.from_numpy(bank["y"][indexes].astype(np.int64))
    return tensor, target


def negative_weights(model, bank, device, rng):
    neg = np.flatnonzero(bank["y"] == 0)
    losses = np.zeros(len(neg), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(neg), 32):
            indexes = neg[start:start + 32]
            xb, yb = batch_from(bank, indexes, False, rng)
            nll = nn.functional.cross_entropy(model(xb.to(device)), yb.to(device), reduction="none")
            losses[start:start + len(indexes)] = nll.detach().cpu().numpy()
    weights = bank["w"][neg] * (losses + 1e-6)
    return weights / weights.sum()


def run_epoch(model, bank, optimizer, kind, device, rng, train, neg_override=None):
    pos = np.flatnonzero(bank["y"] == 1)
    neg = np.flatnonzero(bank["y"] == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("training bank is missing a class")
    pos_w = bank["w"][pos]
    pos_w = pos_w / pos_w.sum()
    neg_w = bank["w"][neg] if neg_override is None else neg_override
    neg_w = neg_w / neg_w.sum()
    n_pos = 4 if kind != "wang" else 2
    n_neg = n_pos * NEG_CAP[kind]
    steps = max(1, int(np.ceil(len(pos) / n_pos)))
    if kind == "wang":
        weight = None
    else:
        counts = np.bincount(bank["y"], minlength=2).astype(np.float32)
        weight = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    total = 0.0
    model.train(train)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for _ in range(steps):
            ip = rng.choice(len(pos), size=n_pos, replace=True, p=pos_w)
            inn = rng.choice(len(neg), size=n_neg, replace=True, p=neg_w)
            indexes = np.concatenate([pos[ip], neg[inn]])
            xb, yb = batch_from(bank, indexes, train, rng)
            xb = xb.to(device)
            yb = yb.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if train:
                loss.backward()
                if not torch.isfinite(loss.detach()):
                    raise RuntimeError("non-finite loss")
                optimizer.step()
            total += float(loss.detach())
    return total / steps


def score_rows(model, rows, device):
    model.eval()
    by_scan = defaultdict(list)
    for index, row in enumerate(rows):
        by_scan[row["scan"]].append(index)
    scores = np.zeros(len(rows), dtype=np.float32)
    labels = np.zeros(len(rows), dtype=np.int64)
    patients = []
    skipped = 0
    with torch.no_grad():
        for scan, indexes in by_scan.items():
            image = np.load(os.path.join(PREP, scan, "image.npy"), mmap_mode="r")
            geometry = json.load(open(os.path.join(PREP, scan, "geometry.json")))
            inverse = np.linalg.inv(np.array(geometry["affine"], float))
            patches = []
            use = []
            for index in indexes:
                patch, inside = take_patch(image, inverse, rows[index])
                labels[index] = rows[index]["label"]
                if not inside:
                    skipped += 1
                    scores[index] = 0.0
                    continue
                patches.append(patch)
                use.append(index)
            if not patches:
                continue
            batch = zscore(np.stack(patches))
            tensor = torch.from_numpy(np.ascontiguousarray(batch[:, None])).to(device)
            for start in range(0, len(use), 32):
                piece = tensor[start:start + 32]
                prob = torch.softmax(model(piece).float(), 1)[:, 1].cpu().numpy()
                for offset, value in enumerate(prob):
                    scores[use[start + offset]] = value
    patients = [row["patient"] for row in rows]
    return scores, labels, patients, skipped


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


def main():
    assert abs(positive_f1(70, 6, 16) - 0.8641975308641974) < 1e-12
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
    free = gpu_free_mib()
    print("GPU_FREE_MIB", free, flush=True)
    if free < 10000:
        raise SystemExit("GPU free memory below 10 GiB; not launching")
    os.makedirs(RUN, exist_ok=True)
    started = time.time()
    series, patients = load_records()
    role = assign_roles(patients, series)
    index_candidates(series, role)
    if os.environ.get("SMOKE") == "1":
        kinds = ("zuo",)
    else:
        kinds = ("zuo", "wang", "clear", "ablation")
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    models = build_models()
    full = extract_fit("fit", 7)
    summary = {"smoke": os.environ.get("SMOKE") == "1", "models": {}, "departures": [
        "Shared cubic 32-voxel crops and per-crop z-score.",
        "Zuo view is the center 18 voxels. Wang views are cubic 32, 16, and 8, not 32x48x48, 16x24x24, and 8x12x12.",
        "Negatives are a seeded draw of 7 per positive from the training patients, not every LUNA candidate. Wang uses a seeded 3-per-positive subset of that draw.",
        "No brightness filter and no lung-mask exclusion. A candidate is ignored only when its center is outside the array.",
        "Epoch count is fixed at 8 before evaluation. Threshold is chosen once on the development candidates.",
        "Hard-negative weights after epoch 2 use training-bank losses only. There is no vessel mask.",
    ]}
    dev_rows = read_jsonl(RUN + "/dev.jsonl")
    for kind in kinds:
        if time.time() - started > 7 * 3600:
            summary["models"][kind] = {"status": "not_started_budget"}
            break
        model = models[kind].to(device)
        bank = full
        if kind == "wang":
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            pos_n = int((full["y"] == 1).sum())
            neg = np.flatnonzero(full["y"] == 0)
            keep_neg = np.random.default_rng(SEED).choice(neg, size=min(len(neg), 3 * pos_n), replace=False)
            keep = np.concatenate([np.flatnonzero(full["y"] == 1), keep_neg])
            rows = [{"patient": pid} for pid in full["patient"][keep]]
            bank = {"x": full["x"][keep], "y": full["y"][keep], "w": patient_weights(rows), "patient": full["patient"][keep]}
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        history = []
        hard = None
        for epoch in range(1, EPOCHS + 1):
            if time.time() - started > 7 * 3600:
                break
            rng = np.random.default_rng(SEED + epoch)
            loss = run_epoch(model, bank, optimizer, kind, device, rng, True, hard)
            history.append({"epoch": epoch, "loss": loss})
            print("EPOCH", kind, epoch, round(loss, 4), flush=True)
            if epoch == 2:
                hard = negative_weights(model, bank, device, rng)
        grad_ok = any(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())
        dev_score, dev_label, dev_patient, skipped = score_rows(model, dev_rows, device)
        cut, dev_metrics = choose_cut(dev_score, dev_label)
        dev_metrics["auc"] = roc_auc(dev_score, dev_label)
        dev_metrics["threshold"] = cut
        dev_metrics["skipped_outside"] = skipped
        torch.save({"state": model.state_dict(), "threshold": cut, "kind": kind}, RUN + "/" + kind + ".pt")
        np.savez_compressed(RUN + "/" + kind + "-dev-scores.npz", score=dev_score, label=dev_label, patient=np.array(dev_patient))
        summary["models"][kind] = {"history": history, "gradient_seen": bool(grad_ok), "dev": dev_metrics}
        print("DEV", kind, {k: dev_metrics[k] for k in ("f1", "recall", "precision", "accuracy", "tp", "fp", "fn", "tn")}, flush=True)
    if os.environ.get("SMOKE") != "1":
        confirm_rows = read_jsonl(RUN + "/confirm.jsonl")
        summary["confirm"] = {}
        frozen = {}
        for kind in summary["models"]:
            if "dev" not in summary["models"][kind]:
                continue
            state = torch.load(RUN + "/" + kind + ".pt", map_location=device)
            model = build_models()[kind].to(device)
            model.load_state_dict(state["state"])
            cut = float(state["threshold"])
            score, label, patient, skipped = score_rows(model, confirm_rows, device)
            metrics = confusion(score, label, cut)
            metrics["auc"] = roc_auc(score, label)
            metrics["threshold"] = cut
            metrics["skipped_outside"] = skipped
            np.savez_compressed(
                RUN + "/" + kind + "-confirm-scores.npz",
                score=score, label=label, patient=np.array(patient),
            )
            summary["confirm"][kind] = metrics
            frozen[kind] = (score, label, np.array(patient), cut)
            print("CONFIRM", kind, {k: metrics[k] for k in ("f1", "recall", "precision", "accuracy", "tp", "fp", "fn", "tn")}, flush=True)
        summary["paired"] = paired_differences(frozen)
    summary["seconds"] = time.time() - started
    json.dump(summary, open(RUN + "/summary.json", "w"), indent=2)
    print("DONE", round(summary["seconds"], 1), flush=True)


def paired_differences(frozen, draws=200):
    if "clear" not in frozen:
        return {}
    rng = np.random.default_rng(SEED)
    out = {}
    base_patients = frozen["clear"][2]
    for other in ("zuo", "wang", "ablation"):
        if other not in frozen:
            continue
        diffs = []
        patients = np.unique(base_patients)
        for _ in range(draws):
            sample = rng.choice(patients, size=len(patients), replace=True)

            def pooled(kind):
                score, label, owner, cut = frozen[kind]
                parts_s = []
                parts_y = []
                for pid in sample:
                    mask = owner == pid
                    parts_s.append(score[mask])
                    parts_y.append(label[mask])
                return confusion(np.concatenate(parts_s), np.concatenate(parts_y), cut)["f1"]

            diffs.append(pooled("clear") - pooled(other))
        arr = np.array(diffs)
        out["clear_minus_" + other] = {
            "mean": float(arr.mean()),
            "low": float(np.quantile(arr, 0.025)),
            "high": float(np.quantile(arr, 0.975)),
        }
    return out


if __name__ == "__main__":
    main()
