"""Development-only completion of the remaining model, sampling, and optimizer items.

Confirmation is closed. This file never opens confirm.jsonl, the confirmation
scores, or the six reviewed patients. Development scores from this pass are a
second look at development. They do not replace the frozen confirmation result.
"""
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
SEED = 17
SMOKE = os.environ.get("SMOKE") == "1"
RUN = ROOT + ("/runs/phase-finish-dev/smoke" if SMOKE else "/runs/phase-finish-dev")
EPOCHS = 1 if SMOKE else 8
WALL_S = 3 * 3600
WANG_SIZE = (32, 48, 48)
FORBIDDEN = ("confirm.jsonl", "confirm-scores")


def refuse_confirmation(path):
    text = str(path)
    if any(token in text for token in FORBIDDEN):
        raise SystemExit("confirmation is closed")


def crop_aniso(array, center_zyx, size, fill=0):
    size = np.asarray(size, dtype=int)
    if size.ndim == 0:
        size = np.array([int(size)] * 3)
    center = np.floor(np.asarray(center_zyx, dtype=float) + 0.5).astype(int)
    start = center - size // 2
    stop = start + size
    shape = np.asarray(array.shape)
    inside = bool(np.all(center >= 0) and np.all(center < shape))
    src_start = np.maximum(start, 0)
    src_stop = np.minimum(stop, shape)
    out = np.full(tuple(int(v) for v in size), fill, dtype=np.float32)
    if np.all(src_stop > src_start):
        src = tuple(slice(int(a), int(b)) for a, b in zip(src_start, src_stop))
        dst = tuple(slice(int(a), int(b)) for a, b in zip(src_start - start, src_stop - start))
        out[dst] = np.asarray(array[src], dtype=np.float32)
    covered = float(np.prod(np.maximum(src_stop - src_start, 0)) / np.prod(size))
    return out, start, inside, covered


def self_check():
    image = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    reference, start = crop(image, [1.6, 2.4, 3.5], 8)
    mine, mine_start, inside, covered = crop_aniso(image, [1.6, 2.4, 3.5], (8, 8, 8))
    if not np.array_equal(reference, mine) or not np.array_equal(start, mine_start):
        raise RuntimeError("anisotropic crop disagrees with the cubic crop")
    if inside and not (0.0 < covered <= 1.0):
        raise RuntimeError("valid fraction is outside (0, 1]")
    edge, _, edge_inside, edge_covered = crop_aniso(image, [0.2, 5.0, 5.0], (8, 8, 8))
    if edge_inside and not edge_covered < 1.0:
        raise RuntimeError("a border crop was treated as fully inside")
    if not np.isfinite(edge).all():
        raise RuntimeError("border crop is non-finite")
    counts = np.array([7.0, 1.0])
    if not np.allclose(counts * (counts.sum() / (2.0 * counts)), [4.0, 4.0]):
        raise RuntimeError("class-weight balance changed")


class WangAniso(nn.Module):
    """Wang stage-2 shape: 32x48x48, 16x24x24, and 8x12x12 from one center."""

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
        views = (
            x,
            x[:, :, 8:24, 12:36, 12:36],
            x[:, :, 12:20, 18:30, 18:30],
        )
        parts = [branch(view).mean(dim=(2, 3, 4)) for branch, view in zip(self.branches, views)]
        return self.fc(torch.relu(self.fuse(torch.cat(parts, 1))))


class ZuoStrideNet(nn.Module):
    """Zuo depth branches. The shallow pool uses stride 4 and the next uses stride 2."""

    def __init__(self):
        super().__init__()
        self.s1 = nn.Conv3d(1, 32, 3, padding=1)
        self.s2 = nn.Conv3d(32, 64, 3, padding=1)
        self.s3 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=4, stride=4)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 + 64 + 128, 2)
        for layer in (self.s1, self.s2, self.s3):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.constant_(layer.bias, 0.2)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x[:, :, 7:25, 7:25, 7:25]
        h1 = torch.relu(self.s1(x))
        h2 = torch.relu(self.s2(self.pool4(h1)))
        h3 = torch.relu(self.s3(self.pool2(h2)))
        parts = [h.mean(dim=(2, 3, 4)) for h in (h1, h2, h3)]
        return self.fc(torch.cat(parts, 1))


def read_split():
    payload = json.load(open(phase_cf.RUN + "/split.json"))
    for key in ("fit", "dev", "confirm"):
        if set(payload[key]) & phase_cf.REVIEWED:
            raise RuntimeError("reviewed patient is inside the frozen split")
    return payload


def read_rows(name, patients):
    path = phase_cf.RUN + "/" + name + ".jsonl"
    refuse_confirmation(path)
    allowed = set(patients)
    rows = []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row["patient"] in allowed:
                rows.append(row)
    return rows


def sample_rows(rows, neg_per_pos):
    pos = [row for row in rows if row["label"] == 1]
    neg = [row for row in rows if row["label"] == 0]
    rng = np.random.default_rng(SEED)
    need = min(len(neg), neg_per_pos * max(len(pos), 1))
    choice = rng.choice(len(neg), size=need, replace=False) if need else []
    return pos + [neg[int(i)] for i in choice]


def extract(rows, size):
    chosen = rows
    by_scan = defaultdict(list)
    for index, row in enumerate(chosen):
        by_scan[row["scan"]].append(index)
    patches = np.zeros((len(chosen), int(size[0]), int(size[1]), int(size[2])), dtype=np.float32)
    labels = np.zeros(len(chosen), dtype=np.uint8)
    valid = np.zeros(len(chosen), dtype=np.float32)
    kept = np.zeros(len(chosen), dtype=bool)
    skipped = 0
    for number, (scan, indexes) in enumerate(by_scan.items(), 1):
        image = np.load(os.path.join(PREP, scan, "image.npy"), mmap_mode="r")
        geometry = json.load(open(os.path.join(PREP, scan, "geometry.json")))
        inverse = np.linalg.inv(np.array(geometry["affine"], dtype=float))
        for index in indexes:
            row = chosen[index]
            ijk = inverse @ np.array([row["x"], row["y"], row["z"], 1.0])
            zyx = [float(ijk[0]), float(ijk[1]), float(ijk[2])]
            patch, _, inside, covered = crop_aniso(image, zyx, size)
            if not inside or covered <= 0:
                skipped += 1
                continue
            patches[index] = patch
            labels[index] = row["label"]
            valid[index] = covered
            kept[index] = True
        if number % 50 == 0:
            print("EXTRACT", number, len(by_scan), flush=True)
    patients = np.array([chosen[i]["patient"] for i in range(len(chosen)) if kept[i]])
    bank = {
        "x": patches[kept],
        "y": labels[kept].astype(np.int64),
        "patient": patients,
        "valid": valid[kept],
    }
    bank["w"] = combine_weights(bank)
    print("BANK", int(kept.sum()), "skipped", skipped, "pos", int((bank["y"] == 1).sum()), flush=True)
    return bank


def combine_weights(bank):
    rows = [{"patient": str(pid)} for pid in bank["patient"]]
    weights = phase_cf.patient_weights(rows) * np.asarray(bank["valid"], dtype=np.float64)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError("sample weights are unusable")
    return weights / weights.sum()


def load_cubic_bank():
    data = np.load(phase_cf.RUN + "/fit-bank.npz")
    bank = {
        "x": data["x"],
        "y": data["y"].astype(np.int64),
        "patient": data["patient"],
        "valid": np.ones(len(data["y"]), dtype=np.float32),
    }
    bank["w"] = combine_weights(bank)
    return bank


def shift_batch(batch, shift, axis):
    moved = np.roll(batch, shift, axis=axis)
    if shift > 0:
        index = [slice(None)] * moved.ndim
        index[axis] = slice(0, shift)
        moved[tuple(index)] = 0
    elif shift < 0:
        index = [slice(None)] * moved.ndim
        index[axis] = slice(shift, None)
        moved[tuple(index)] = 0
    return moved


def augment_batch(batch, rng):
    for axis in (1, 2, 3):
        if rng.random() < 0.5:
            batch = np.flip(batch, axis=axis)
    turns = int(rng.integers(0, 4))
    if turns:
        batch = np.rot90(batch, k=turns, axes=(2, 3))
    for axis in (1, 2, 3):
        shift = int(rng.integers(-2, 3))
        if shift:
            batch = shift_batch(batch, shift, axis)
    scale = float(rng.uniform(0.9, 1.1))
    offset = float(rng.uniform(-0.05, 0.05))
    return np.ascontiguousarray(batch * scale + offset)


def batch_from(bank, indexes, train, rng):
    batch = bank["x"][indexes].astype(np.float32)
    if train:
        batch = augment_batch(batch, rng)
    batch = phase_cf.zscore(np.ascontiguousarray(batch))
    tensor = torch.from_numpy(np.ascontiguousarray(batch[:, None]))
    target = torch.from_numpy(bank["y"][indexes].astype(np.int64))
    return tensor, target


def loss_of(logits, target, spec, device, counts):
    if spec["loss"] == "focal":
        log_p = torch.log_softmax(logits, dim=1)
        chosen = log_p.gather(1, target[:, None]).squeeze(1)
        probability = chosen.exp()
        return (-((1.0 - probability) ** 2.0) * chosen).mean()
    weight = None
    if spec["class_weight"]:
        weight = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), device=device)
    return nn.functional.cross_entropy(logits, target, weight=weight)


def run_epoch(model, bank, optimizer, spec, device, rng, train, neg_override=None):
    pos = np.flatnonzero(bank["y"] == 1)
    neg = np.flatnonzero(bank["y"] == 0)
    pos_w = bank["w"][pos]
    pos_w = pos_w / pos_w.sum()
    neg_w = bank["w"][neg] if neg_override is None else neg_override
    neg_w = neg_w / neg_w.sum()
    n_pos = spec["n_pos"]
    n_neg = n_pos * spec["neg_per_pos"]
    steps = max(1, int(np.ceil(len(pos) / n_pos)))
    counts = np.bincount(bank["y"], minlength=2).astype(np.float32)
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
            loss = loss_of(model(xb), yb, spec, device, counts)
            if train:
                if not torch.isfinite(loss.detach()):
                    raise RuntimeError("non-finite loss")
                loss.backward()
                if spec["clip"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total += float(loss.detach())
    return total / steps


def hard_negative(model, bank, spec, device):
    neg = np.flatnonzero(bank["y"] == 0)
    losses = np.zeros(len(neg), dtype=np.float64)
    counts = np.bincount(bank["y"], minlength=2).astype(np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(neg), 16):
            indexes = neg[start:start + 16]
            xb, yb = batch_from(bank, indexes, False, np.random.default_rng(0))
            logits = model(xb.to(device))
            target = yb.to(device)
            if spec["loss"] == "focal":
                log_p = torch.log_softmax(logits, dim=1)
                chosen = log_p.gather(1, target[:, None]).squeeze(1)
                probability = chosen.exp()
                values = (-((1.0 - probability) ** 2.0) * chosen)
            else:
                weight = None
                if spec["class_weight"]:
                    weight = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), device=device)
                values = nn.functional.cross_entropy(logits, target, weight=weight, reduction="none")
            losses[start:start + len(indexes)] = values.detach().cpu().numpy()
    weights = bank["w"][neg] * (losses + 1e-6)
    return weights / weights.sum()


def score_rows(model, rows, size, device):
    by_scan = defaultdict(list)
    for index, row in enumerate(rows):
        by_scan[row["scan"]].append(index)
    scores = np.zeros(len(rows), dtype=np.float32)
    labels = np.array([row["label"] for row in rows], dtype=np.int64)
    patients = [row["patient"] for row in rows]
    skipped = 0
    model.eval()
    with torch.no_grad():
        for number, (scan, indexes) in enumerate(by_scan.items(), 1):
            image = np.load(os.path.join(PREP, scan, "image.npy"), mmap_mode="r")
            geometry = json.load(open(os.path.join(PREP, scan, "geometry.json")))
            inverse = np.linalg.inv(np.array(geometry["affine"], dtype=float))
            patches = []
            use = []
            for index in indexes:
                row = rows[index]
                ijk = inverse @ np.array([row["x"], row["y"], row["z"], 1.0])
                patch, _, inside, _ = crop_aniso(image, [float(ijk[0]), float(ijk[1]), float(ijk[2])], size)
                if not inside or patch.shape != tuple(size):
                    skipped += 1
                    continue
                patches.append(patch)
                use.append(index)
            if not patches:
                continue
            batch = phase_cf.zscore(np.stack(patches).astype(np.float32))
            tensor = torch.from_numpy(np.ascontiguousarray(batch[:, None])).to(device)
            for start in range(0, len(use), 16):
                prob = torch.softmax(model(tensor[start:start + 16]).float(), 1)[:, 1].cpu().numpy()
                for offset, value in enumerate(prob):
                    scores[use[start + offset]] = float(value)
            if number % 20 == 0:
                print("SCORE", number, len(by_scan), flush=True)
    return scores, labels, patients, skipped


def save_checkpoint(model, optimizer, epoch, rng, kind, threshold, path):
    torch.save({
        "state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "epoch": epoch,
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "kind": kind,
        "development_threshold": threshold,
        "confirmation": "closed",
    }, path)


def gradient_report(model):
    missing = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            missing.append(name)
        elif float(parameter.grad.abs().sum()) == 0.0:
            missing.append(name)
    return missing


def specs():
    return [
        {
            "name": "wang_aniso",
            "family": "wang",
            "size": WANG_SIZE,
            "loss": "ce",
            "class_weight": False,
            "optimizer": "adam",
            "lr": 1e-4,
            "neg_per_pos": 3,
            "n_pos": 2,
            "clip": False,
            "note": "Stage-2 input is 32x48x48 with center 16x24x24 and 8x12x12. Adam at 0.0001, unweighted cross-entropy, three negatives per positive.",
        },
        {
            "name": "zuo_stride",
            "family": "zuo",
            "size": (32, 32, 32),
            "loss": "ce",
            "class_weight": True,
            "optimizer": "sgd",
            "lr": 0.01,
            "neg_per_pos": 7,
            "n_pos": 4,
            "clip": False,
            "note": "Shallow pool stride 4, next pool stride 2, center 18 cube, bias 0.2. Eight epochs, not the paper's fifteen. Negatives remain the seeded draw.",
        },
        {
            "name": "clear_aug",
            "family": "clear",
            "size": (32, 32, 32),
            "loss": "ce",
            "class_weight": True,
            "optimizer": "sgd",
            "lr": 0.01,
            "neg_per_pos": 7,
            "n_pos": 4,
            "clip": False,
            "note": "Existing CLEAR3D candidate with flips, 90-degree rotation, translation of at most 2 voxels, and intensity scale 0.9 to 1.1.",
        },
        {
            "name": "clear_focal",
            "family": "clear",
            "size": (32, 32, 32),
            "loss": "focal",
            "class_weight": False,
            "optimizer": "sgd",
            "lr": 0.01,
            "neg_per_pos": 7,
            "n_pos": 4,
            "clip": False,
            "note": "Focal loss gamma 2 without class weights. The 7-to-1 draw is unchanged, so the class weight is not stacked on the focal term.",
        },
        {
            "name": "clear_lr",
            "family": "clear",
            "size": (32, 32, 32),
            "loss": "ce",
            "class_weight": True,
            "optimizer": "sgd",
            "lr": 0.001,
            "neg_per_pos": 7,
            "n_pos": 4,
            "clip": False,
            "note": "Same augmented CLEAR3D objective at learning rate 0.001. The original run used 0.01.",
        },
    ]


def build(spec):
    if spec["family"] == "wang":
        return WangAniso()
    if spec["family"] == "zuo":
        return ZuoStrideNet()
    return phase_cf.ClearNet(True)


def make_optimizer(model, spec):
    if spec["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=spec["lr"])
    return torch.optim.SGD(model.parameters(), lr=spec["lr"], momentum=0.9)


def train_one(spec, bank, dev_rows, device, started):
    model = build(spec).to(device)
    optimizer = make_optimizer(model, spec)
    history = []
    hard = None
    rng = np.random.default_rng(SEED)
    for epoch in range(1, EPOCHS + 1):
        if time.time() - started > WALL_S:
            return {"status": "stopped_on_wall", "history": history}
        rng = np.random.default_rng(SEED + epoch)
        loss = run_epoch(model, bank, optimizer, spec, device, rng, True, hard)
        history.append({"epoch": epoch, "loss": loss})
        print("EPOCH", spec["name"], epoch, round(loss, 4), flush=True)
        if epoch == 2 and not SMOKE:
            hard = hard_negative(model, bank, spec, device)
    missing = gradient_report(model)
    score, label, patient, skipped = score_rows(model, dev_rows, spec["size"], device)
    cut, metrics = phase_cf.choose_cut(score, label)
    metrics["auc"] = phase_cf.roc_auc(score, label)
    metrics["pr_auc"] = __import__("finish_metrics").average_precision(score, label)
    metrics["threshold"] = cut
    metrics["skipped_outside"] = skipped
    path = RUN + "/" + spec["name"] + ".pt"
    save_checkpoint(model, optimizer, EPOCHS, rng, spec["name"], cut, path)
    np.savez_compressed(
        RUN + "/" + spec["name"] + "-dev-scores.npz",
        score=score, label=label, patient=np.array(patient),
    )
    changed = 0
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.grad is not None:
            changed += int((parameter.grad != 0).sum())
    auc_text = "none" if metrics["auc"] is None else round(float(metrics["auc"]), 4)
    print("DEV", spec["name"], round(metrics["f1"], 4), auc_text, flush=True)
    return {
        "status": "scored_on_development",
        "history": history,
        "dev": {key: (float(value) if isinstance(value, (float, np.floating)) else int(value) if isinstance(value, (int, np.integer)) else value) for key, value in metrics.items()},
        "gradients_missing": missing,
        "nonzero_gradient_values": changed,
        "parameter_count": total,
        "scheduler": None,
        "checkpoint": path,
        "note": spec["note"],
    }


def paired_against_stored(summary):
    if SMOKE or "dev" not in summary.get("clear_aug", {}):
        return {}
    stored = json.load(open(phase_cf.RUN + "/summary.json"))
    new = np.load(RUN + "/clear_aug-dev-scores.npz")
    out = {}
    rng = np.random.default_rng(SEED)
    patients = np.array([str(item) for item in new["patient"]])
    unique = np.unique(patients)
    grouped = {patient: np.flatnonzero(patients == patient) for patient in unique}
    for other in ("zuo", "wang", "clear"):
        old = np.load(phase_cf.RUN + "/" + other + "-dev-scores.npz")
        old_patients = np.array([str(item) for item in old["patient"]])
        if len(old["score"]) != len(new["score"]) or not np.array_equal(old_patients, patients):
            continue
        old_cut = float(stored["models"][other]["dev"]["threshold"])
        new_cut = float(summary["clear_aug"]["dev"]["threshold"])
        diffs = []
        for _ in range(200):
            draw = rng.choice(unique, size=len(unique), replace=True)
            parts_new = []
            parts_old_s = []
            parts_old_y = []
            for patient in draw:
                pick = grouped[patient]
                parts_new.append(new["score"][pick])
                parts_old_s.append(old["score"][pick])
                parts_old_y.append(old["label"][pick])
            oy = np.concatenate(parts_old_y).astype(np.int64)
            diffs.append(
                positive_f1_from_scores(np.concatenate(parts_new), oy, new_cut)
                - positive_f1_from_scores(np.concatenate(parts_old_s), oy, old_cut)
            )
        arr = np.array(diffs, dtype=float)
        out["clear_aug_minus_" + other] = {
            "mean": float(arr.mean()),
            "low": float(np.quantile(arr, 0.025)),
            "high": float(np.quantile(arr, 0.975)),
        }
    return out


def positive_f1_from_scores(score, label, cut):
    row = phase_cf.confusion(score, label, cut)
    return row["f1"]


def write_report(payload):
    lines = [
        "# Development review, 2026-10-26",
        "",
        "This pass trains on fit patients and scores development patients. Confirmation was not opened. A higher development score does not replace the frozen confirmation result.",
        "",
        "Vessel and airway labels are not in the prepared volumes. No vessel mask was created. Hard-negative weights after epoch 2 still come from the training-bank loss.",
        "",
        "Wang stage 1, the attention residual U-Net, was not trained. The paper's module is not specified here precisely enough to build it without inventing the block. The stage-2 crop size is specified, and that is the Wang model in this pass.",
        "",
        "Augmentation on every model in this pass is training-only: axis flips, rotation by a multiple of 90 degrees, translation of at most 2 voxels with the vacated edge set to zero, and an intensity scale from 0.9 to 1.1 plus a shift of at most 0.05. Sample weights include the fraction of each crop that lies inside the scan.",
        "",
        "| model | development F1 | recall | precision | ROC-AUC | PR-AUC | threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["models"].items():
        if "dev" not in row:
            lines.append("| %s | %s | | | | | |" % (name, row.get("status", "absent")))
            continue
        dev = row["dev"]
        lines.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
            name, dev["f1"], dev["recall"], dev["precision"], dev["auc"], dev["pr_auc"], dev["threshold"],
        ))
    lines.append("")
    lines.append("Stored development F1 from the frozen run, for reference only: Zuo-style 0.4187, Wang-style 0.2298, CLEAR3D 0.1706, ablation 0.1752.")
    lines.append("")
    if payload.get("paired_development"):
        lines.append("Paired development difference of augmented CLEAR3D minus the stored development score, 200 patient resamples, seed 17.")
        lines.append("")
        for name, row in payload["paired_development"].items():
            lines.append("- %s: mean %.4f, interval %.4f to %.4f" % (name, row["mean"], row["low"], row["high"]))
        lines.append("")
    lines.append("Checkpoints store the model, optimizer, epoch, NumPy RNG, and torch RNG. The scheduler entry is null because this recipe has no scheduler.")
    lines.append("")
    text = "\n".join(lines) + "\n"
    open(RUN + "/DEV_REVIEW.md", "w").write(text)
    if not SMOKE:
        open(ROOT + "/DEV_REVIEW.md", "w").write(text)


def main():
    self_check()
    os.makedirs(RUN, exist_ok=True)
    free = phase_cf.gpu_free_mib()
    print("GPU_FREE_MIB", free, flush=True)
    if free < 10000:
        raise SystemExit("GPU free memory below 10 GiB; not launching")
    started = time.time()
    split = read_split()
    fit_patients = split["fit"][:40] if SMOKE else split["fit"]
    dev_patients = split["dev"][:2] if SMOKE else split["dev"]
    dev_rows = read_rows("dev", dev_patients)
    if SMOKE:
        dev_rows = dev_rows[:400]
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    summary = {}
    cubic = None
    planned = specs()[:1] if SMOKE else specs()
    for spec in planned:
        if time.time() - started > WALL_S:
            summary[spec["name"]] = {"status": "not_started_budget"}
            continue
        if spec["family"] == "wang":
            bank = extract(sample_rows(read_rows("fit", fit_patients), spec["neg_per_pos"]), spec["size"])
        else:
            if cubic is None:
                cubic = load_cubic_bank()
            bank = cubic
        summary[spec["name"]] = train_one(spec, bank, dev_rows, device, started)
        if spec["family"] == "wang":
            del bank
    payload = {
        "smoke": SMOKE,
        "seconds": time.time() - started,
        "models": summary,
        "confirmation": "closed",
    }
    if not SMOKE:
        payload["paired_development"] = paired_against_stored(summary)
    json.dump(payload, open(RUN + "/summary.json", "w"), indent=2)
    write_report(payload)
    print("DONE", round(payload["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
