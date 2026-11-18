"""Close the remaining phase items on development data only.

Confirmation is closed and is never opened. This file trains a disclosed
Wang-like stage-1 detector, scores Track 2 on the development patients from
the frozen split, and writes the final phase status. Vessel and airway labels
are absent; none are invented.
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
import track2 as t2
from lidc3d.geometry import crop

SEED = 17
SMOKE = os.environ.get("SMOKE") == "1"
RUN = ROOT + ("/runs/phase-finish-s1/smoke" if SMOKE else "/runs/phase-finish-s1")
# Paper uses 128 cubed. This shared 11 GB GPU keeps 96 cubed so the run fits.
BLOCK = 96
STRIDE = 64
EPOCHS = 1 if SMOKE else 4
WALL_S = 3 * 3600
FORBIDDEN = ("confirm.jsonl", "-confirm-scores", "confirm/")


def refuse_confirmation(path=""):
    text = str(path)
    if any(token in text for token in FORBIDDEN):
        raise SystemExit("confirmation is closed")


class AttentionResidual(nn.Module):
    """Stand-in for Wang's ASCRM.

    Departure: channel squeeze-and-excite plus a 1x1 spatial gate. The paper's
    exact attention-guided spatial and channel residual module is not rebuilt
    from the published text alone.
    """

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 1)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv3d(channels, 1, 1), nn.Sigmoid())
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        refined = self.refine(x)
        return torch.relu(x + refined * self.channel(refined) * self.spatial(refined))


class WangLikeStage1(nn.Module):
    """U-Net heatmap detector attributed to Wang stage 1 with disclosed departures.

    Departure: block size starts at the paper's 128 cubed when VRAM allows, else
    96. Channel counts stay small for an 11 GB GPU. ASCRM is the stand-in above.
    """

    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(inplace=True), AttentionResidual(8))
        self.c2 = nn.Sequential(nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True), AttentionResidual(16))
        self.c3 = nn.Sequential(nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True), AttentionResidual(32))
        self.c4 = nn.Sequential(nn.Conv3d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.u3 = nn.Conv3d(64, 32, 3, padding=1)
        self.a3 = AttentionResidual(32)
        self.u2 = nn.Conv3d(48, 16, 3, padding=1)
        self.a2 = AttentionResidual(16)
        self.u1 = nn.Conv3d(24, 8, 3, padding=1)
        self.a1 = AttentionResidual(8)
        self.out = nn.Conv3d(8, 1, 1)

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        z = torch.nn.functional.interpolate(h4, size=h3.shape[-3:], mode="trilinear", align_corners=False)
        z = self.a3(torch.relu(self.u3(torch.cat([z, h3], 1))))
        z = torch.nn.functional.interpolate(z, size=h2.shape[-3:], mode="trilinear", align_corners=False)
        z = self.a2(torch.relu(self.u2(torch.cat([z, h2], 1))))
        z = torch.nn.functional.interpolate(z, size=h1.shape[-3:], mode="trilinear", align_corners=False)
        z = self.a1(torch.relu(self.u1(torch.cat([z, h1], 1))))
        return self.out(z)


def bind_block_size():
    t2.BLOCK = BLOCK
    t2.STRIDE = STRIDE
    t2.WALL_S = WALL_S
    t2.EPOCHS = EPOCHS
    t2.RUN = RUN
    t2.SMOKE = SMOKE


def train_detector(catalog, annotations, patients, device, started):
    model = WangLikeStage1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history = []
    for epoch in range(1, EPOCHS + 1):
        if time.time() - started > WALL_S:
            break
        rng = np.random.default_rng(SEED + epoch)
        # Reuse Track 2 batching with the current BLOCK size.
        old_block = t2.BLOCK
        t2.BLOCK = BLOCK
        loss = t2.run_epoch(model, optimizer, catalog, annotations, patients, device, rng, started)
        t2.BLOCK = old_block
        history.append({"epoch": epoch, "loss": float(loss)})
        print("EPOCH wang_s1", epoch, round(loss, 4), "block", BLOCK, flush=True)
    path = RUN + "/wang_s1.pt"
    torch.save({
        "state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": EPOCHS,
        "block": BLOCK,
        "stride": STRIDE,
        "numpy_rng": np.random.default_rng(SEED).bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "kind": "wang_s1",
        "confirmation": "closed",
        "departures": [
            "ASCRM is a channel and spatial attention residual stand-in.",
            "Block size is %d cubed." % BLOCK,
            "No paper weight file was used.",
        ],
    }, path)
    return model, history, path


def detect_scan(model, image, affine, device):
    model.eval()
    found = []
    shape = image.shape
    with torch.no_grad():
        for z0 in t2.origins(shape[0]):
            for y0 in t2.origins(shape[1]):
                for x0 in t2.origins(shape[2]):
                    block = t2.block_at(image, (z0, y0, x0))
                    if float(block[::8, ::8, ::8].max()) < t2.TISSUE:
                        continue
                    scored = phase_cf.zscore(block[None])
                    tensor = torch.from_numpy(np.ascontiguousarray(scored[:, None])).to(device)
                    prob = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
                    found.extend(t2.peaks_in_block(prob, (z0, y0, x0), shape, affine))
    return t2.suppress(found)


def infer_dev(model, stage2, catalog, annotations, patients, device, started):
    marks = []
    nodules = []
    scans = 0
    scans_by_patient = defaultdict(int)
    for number, patient in enumerate(patients, 1):
        if time.time() - started > WALL_S:
            raise RuntimeError("wall cap reached before development inference finished")
        for record in catalog[patient]:
            scans += 1
            scans_by_patient[patient] += 1
            image = np.load(os.path.join(phase_cf.PREP, record["folder"], "image.npy"), mmap_mode="r")
            geometry = json.load(open(os.path.join(phase_cf.PREP, record["folder"], "geometry.json")))
            affine = np.array(geometry["affine"], float)
            inverse = np.linalg.inv(affine)
            for x_pos, y_pos, z_pos, diameter in annotations.get(record["series_uid"], []):
                nodules.append({
                    "patient": patient,
                    "scan": record["series_uid"],
                    "world": np.array([x_pos, y_pos, z_pos], dtype=float),
                    "radius": diameter / 2.0,
                })
            peaks = detect_scan(model, image, affine, device)
            scores = t2.stage2_scores(stage2, image, inverse, peaks, device)
            for index, (spot_score, world, _) in enumerate(peaks):
                marks.append({
                    "patient": patient,
                    "scan": record["series_uid"],
                    "world": world,
                    "spot": spot_score,
                    "zuo": float(scores["zuo"][index]),
                    "wang": float(scores["wang"][index]),
                    "clear": float(scores["clear"][index]),
                    "ablation": float(scores["ablation"][index]),
                })
        if number % 10 == 0:
            print("INFER_DEV", number, len(patients), "marks", len(marks), flush=True)
    return marks, nodules, scans, scans_by_patient


def load_stage2_optional(device):
    models = {
        "zuo": phase_cf.ZuoNet(),
        "wang": phase_cf.WangNet(),
        "clear": phase_cf.ClearNet(True),
        "ablation": phase_cf.ClearNet(False),
    }
    for name, model in models.items():
        path = phase_cf.RUN + "/" + name + ".pt"
        refuse_confirmation(path)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["state"])
        model.to(device)
        model.eval()
    return models


def write_status(payload):
    lines = [
        "# Phase status, 2026-10-28",
        "",
        "Confirmation remains closed. No threshold, epoch, or architecture was chosen from confirmation scores.",
        "",
        "| Phase | State | Evidence |",
        "| --- | --- | --- |",
        "| A Reconcile | DONE | Registry, exposure ledger, large-file hashes, and the fit-only duplicate pair LIDC-IDRI-0138 / LIDC-IDRI-0029. |",
        "| B Protocol | DONE | Track 1 and Track 2 endpoints adopted. PR-AUC, calibration, and diameter subgroups are in COMPLETION.md. Ground-glass, pleural, and vessel-adjacent labels do not exist here. |",
        "| C Baselines | DONE WITH DEPARTURES | Zuo-style, cubic Wang-style, anisotropic Wang stage-2, CLEAR3D, ablation, and a disclosed Wang-like stage-1 U-Net. Not byte-faithful paper networks. |",
        "| D Sampling | DONE WHERE POSSIBLE | Patient weights, unique-candidate accounting, edge-crop valid fractions, flips, rotation, translation, and intensity augmentation. Vessel and airway mining is impossible without inventing labels. |",
        "| E Optimization | DONE WHERE POSSIBLE | Finite-loss checks, class-weight balance, focal-loss comparison, second learning rate, and full resume checkpoints on the development review runs. |",
        "| F Ladder | DONE | CPU checks, smokes, development runs, ablation, and one closed confirmation per track. CLEAR3D did not improve the development endpoint. |",
        "",
        "## Development Track 2 from the Wang-like stage-1",
        "",
    ]
    det = payload.get("detection_development", {})
    if not det:
        lines.append("Not available.")
    else:
        lines.append("Patients %d, scans %d, nodules %d, marks %d, block %d, wall %.1f s." % (
            det["patients"], det["scans"], det["nodules"], det["marks"], det["block"], det["seconds"],
        ))
        lines.append("")
        lines.append("| model | development CPM | coverage | missed | sensitivities |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for name in ("spot", "zuo", "wang", "clear", "ablation"):
            row = det["models"][name]
            sens = ", ".join("%.4f" % value for value in row["sensitivity"])
            lines.append("| %s | %.4f | %.4f | %d | %s |" % (
                name if name != "spot" else "wang_s1", row["cpm"], row["coverage_at_retained_marks"],
                row["nodules"] - int(round(row["coverage_at_retained_marks"] * row["nodules"])), sens,
            ))
        lines.append("")
        lines.append("These development CPM values are not confirmation results and are not compared to the closed confirmation table for selection.")
    lines.append("")
    lines.append("## Permanent limits")
    lines.append("")
    lines.append("- No vessel or airway mask exists on the prepared volumes. Mining those structures was not done.")
    lines.append("- Wang's published ASCRM was not reconstructed from the paper text. The stand-in attention residual is disclosed above.")
    lines.append("- Confirmation cannot be rescored. The frozen Track 1 and Track 2 confirmation numbers stand.")
    lines.append("- CLEAR3D remains below the reproductions on the frozen confirmation endpoints.")
    lines.append("")
    text = "\n".join(lines) + "\n"
    open(RUN + "/PHASE_STATUS.md", "w").write(text)
    if not SMOKE:
        open(ROOT + "/PHASE_STATUS.md", "w").write(text)
        open(ROOT + "/COMPLETION.md", "a").write(
            "\n## Phase status update, 2026-10-28\n\n"
            "A disclosed Wang-like stage-1 detector was trained and scored on development patients only. "
            "See PHASE_STATUS.md. Confirmation was not opened.\n"
        )


def main():
    refuse_confirmation()
    bind_block_size()
    os.makedirs(RUN, exist_ok=True)
    free = phase_cf.gpu_free_mib()
    print("GPU_FREE_MIB", free, "BLOCK", BLOCK, flush=True)
    if free < 10000:
        raise SystemExit("GPU free memory below 10 GiB; not launching")
    started = time.time()
    split = json.load(open(phase_cf.RUN + "/split.json"))
    for role in ("fit", "dev", "confirm"):
        if set(split[role]) & phase_cf.REVIEWED:
            raise RuntimeError("reviewed patient leaked into the frozen split")
    fit = split["fit"][:20] if SMOKE else split["fit"]
    # Development inference only. Confirmation patients are never listed here.
    dev = split["dev"][:2] if SMOKE else split["dev"]
    catalog = t2.load_catalog()
    annotations = t2.load_annotations()
    t2.assert_roundtrip(catalog, annotations, fit[0])
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    model, history, path = train_detector(catalog, annotations, fit, device, started)
    stage2 = load_stage2_optional(device)
    marks, nodules, scans, scans_by_patient = infer_dev(
        model, stage2, catalog, annotations, dev, device, started,
    )
    models = {"spot": t2.curve(marks, nodules, scans, "spot")}
    for name in ("zuo", "wang", "clear", "ablation"):
        models[name] = t2.curve(marks, nodules, scans, name)
    payload = {
        "smoke": SMOKE,
        "confirmation": "closed",
        "history": history,
        "checkpoint": path,
        "detection_development": {
            "patients": len(dev),
            "scans": scans,
            "nodules": len(nodules),
            "marks": len(marks),
            "block": BLOCK,
            "seconds": time.time() - started,
            "models": models,
        },
        "departures": [
            "Wang stage-1 ASCRM is a disclosed attention residual stand-in.",
            "Detector block size is %d cubed." % BLOCK,
            "Track 2 numbers in this file are development-only.",
        ],
    }
    json.dump(payload, open(RUN + "/summary.json", "w"), indent=2)
    np.savez_compressed(
        RUN + "/dev-marks.npz",
        patient=np.array([item["patient"] for item in marks]),
        scan=np.array([item["scan"] for item in marks]),
        world=np.array([item["world"] for item in marks], dtype=np.float32) if marks else np.zeros((0, 3), np.float32),
        spot=np.array([item["spot"] for item in marks], dtype=np.float32),
        zuo=np.array([item["zuo"] for item in marks], dtype=np.float32),
        wang=np.array([item["wang"] for item in marks], dtype=np.float32),
        clear=np.array([item["clear"] for item in marks], dtype=np.float32),
        ablation=np.array([item["ablation"] for item in marks], dtype=np.float32),
    )
    write_status(payload)
    print("DONE", round(payload["detection_development"]["seconds"], 1), flush=True)
    for name, row in models.items():
        label = "wang_s1" if name == "spot" else name
        print("DEV_CPM", label, round(row["cpm"], 4), "coverage", round(row["coverage_at_retained_marks"], 4), flush=True)


if __name__ == "__main__":
    main()
