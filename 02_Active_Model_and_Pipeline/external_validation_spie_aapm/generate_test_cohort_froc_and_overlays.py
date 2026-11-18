"""Script to generate publication-grade FROC curves and 3-plane anatomical overlays
for the SPIE-AAPM Lung CT Test Cohort Evaluation (Cap=4,000)."""
import csv, json, math, os, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pydicom
from scipy import ndimage as ndi

BASE = Path('/experiment-storage/chaidir/fawad-dsai')
ROOT = BASE / 'CLEAR3D_CT project/02_Active_Model_and_Pipeline'
EXT_DIR = ROOT / 'external_validation_spie_aapm'
RAW_TEST = EXT_DIR / 'raw_test'
OUT_OVERLAYS = EXT_DIR / 'test_cohort_overlays'
OUT_OVERLAYS.mkdir(exist_ok=True, parents=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE / 'CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/src'))

from lidc3d.geometry import resample_lps, to_index, to_world, normalize
from lidc3d.ingest import load_series
from lidc3d.candidates import lung_mask

def plot_froc_curves(summary_json_path: Path, output_png_path: Path):
    if not summary_json_path.exists():
        print(f"Summary JSON {summary_json_path} does not exist yet. Skipping FROC plot.")
        return
    with open(summary_json_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    cohorts = [
        ("Full Test Cohort (60 scans, 73 nodules)", data["cohort_results"]["full_test_cohort"], "#1f77b4", "-"),
        ("Unexposed Test Cohort (58 scans, 70 nodules)", data["cohort_results"]["unexposed_test_scans"], "#2ca02c", "--"),
        ("Technical Pilot (2 scans, 3 nodules)", data["cohort_results"]["technical_pilot_scans"], "#ff7f0e", ":")
    ]

    grid_rates = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    for label, cdata, color, ls in cohorts:
        froc = cdata["froc_cpm_fixed_5mm"]
        sens = froc["sensitivities"]
        cpm = froc["cpm"]
        ci = froc.get("bootstrap_95ci", {}).get("ci_95", [None, None])
        ci_str = f" [95% CI: {ci[0]:.3f} - {ci[1]:.3f}]" if ci[0] is not None else ""
        leg_entry = label + "\n" + f"CPM = {cpm:.4f}" + ci_str
        ax.plot(grid_rates, sens, marker='o', linewidth=2, linestyle=ls, color=color, label=leg_entry)

    ax.set_xscale('log', base=2)
    ax.set_xticks(grid_rates)
    ax.set_xticklabels([str(x) for x in grid_rates], fontsize=11)
    ax.set_xlabel("False Positives per Scan (Fixed 5.0 mm Matching)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Nodule Detection Sensitivity", fontsize=12, fontweight='bold')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("CLEAR3D-MSFv2 External Validation on SPIE-AAPM Lung CT Test Cohort\n(Automated Multiscale LoG Proposals, Cap=4,000, 4-View TTA Ensemble)", fontsize=13, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    fig.savefig(output_png_path, dpi=300)
    plt.close(fig)
    print(f"FROC curve plot saved to {output_png_path}", flush=True)

def generate_target_overlay(scan_id: str, nodule_idx: int, target_lps: np.ndarray,
                            matched: bool, dist_mm: float, cand_rank: int, clf_rank: int,
                            prob: float, out_path: Path):
    dcm_dir = RAW_TEST / scan_id
    if not dcm_dir.exists():
        print(f"Directory {dcm_dir} missing, skipping overlay.")
        return

    dcm_files = sorted(list(dcm_dir.glob("*.dcm")))
    if not dcm_files:
        return

    ingest_cfg = {
        'orientation_atol': 1e-4, 'position_atol_mm': 1e-2,
        'spacing_rtol': 1e-3, 'spacing_atol_mm': 1e-3
    }
    vol = load_series([str(p) for p in dcm_files], ingest_cfg)
    vol_r = resample_lps(vol, spacing=1.0)

    mask, _, _ = lung_mask(
        vol_r.hu, vol_r.valid, spacing=1.0, air_hu=-400,
        min_component_mm3=10000, dilation_mm=5
    )

    norm_hu = normalize(vol_r.hu, window=(-1000, 400))
    norm_hu[~vol_r.valid] = 0.0

    target_zyx = to_index(target_lps, vol_r.affine)
    tz, ty, tx = int(round(target_zyx[0])), int(round(target_zyx[1])), int(round(target_zyx[2]))

    D, H, W = norm_hu.shape
    tz = max(0, min(D - 1, tz))
    ty = max(0, min(H - 1, ty))
    tx = max(0, min(W - 1, tx))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=200)

    # 1. Axial (Z slice)
    crop_radius = 40
    ymin, ymax = max(0, ty - crop_radius), min(H, ty + crop_radius)
    xmin, xmax = max(0, tx - crop_radius), min(W, tx + crop_radius)
    zmin, zmax = max(0, tz - crop_radius), min(D, tz + crop_radius)

    axial_crop = norm_hu[tz, ymin:ymax, xmin:xmax]
    axial_mask = mask[tz, ymin:ymax, xmin:xmax]
    axes[0].imshow(axial_crop, cmap='gray', origin='lower')
    axes[0].contour(axial_mask, levels=[0.5], colors='cyan', linewidths=1.0, linestyles='--')
    cy_a = ty - ymin
    cx_a = tx - xmin
    marker_color = 'lime' if matched else 'red'
    circle = patches.Circle((cx_a, cy_a), 5.0, edgecolor=marker_color, facecolor='none', linewidth=2.0)
    axes[0].add_patch(circle)
    axes[0].plot(cx_a, cy_a, marker='+', color=marker_color, markersize=12, markeredgewidth=2)
    axes[0].set_title(f"Axial (Z={tz})", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # 2. Coronal (Y slice)
    coronal_crop = norm_hu[zmin:zmax, ty, xmin:xmax]
    coronal_mask = mask[zmin:zmax, ty, xmin:xmax]
    axes[1].imshow(coronal_crop, cmap='gray', origin='lower')
    axes[1].contour(coronal_mask, levels=[0.5], colors='cyan', linewidths=1.0, linestyles='--')
    cz_c = tz - zmin
    cx_c = tx - xmin
    circle = patches.Circle((cx_c, cz_c), 5.0, edgecolor=marker_color, facecolor='none', linewidth=2.0)
    axes[1].add_patch(circle)
    axes[1].plot(cx_c, cz_c, marker='+', color=marker_color, markersize=12, markeredgewidth=2)
    axes[1].set_title(f"Coronal (Y={ty})", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # 3. Sagittal (X slice)
    sagittal_crop = norm_hu[zmin:zmax, ymin:ymax, tx]
    sagittal_mask = mask[zmin:zmax, ymin:ymax, tx]
    axes[2].imshow(sagittal_crop, cmap='gray', origin='lower')
    axes[2].contour(sagittal_mask, levels=[0.5], colors='cyan', linewidths=1.0, linestyles='--')
    cz_s = tz - zmin
    cy_s = ty - ymin
    circle = patches.Circle((cy_s, cz_s), 5.0, edgecolor=marker_color, facecolor='none', linewidth=2.0)
    axes[2].add_patch(circle)
    axes[2].plot(cy_s, cz_s, marker='+', color=marker_color, markersize=12, markeredgewidth=2)
    axes[2].set_title(f"Sagittal (X={tx})", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    status_str = "DETECTED WITHIN 5mm" if matched else "MISSED BY GENERATOR"
    info_text = (
        f"Scan: {scan_id} | Target #{nodule_idx} | Status: {status_str}\n"
        f"Ground Truth LPS: [{target_lps[0]:.1f}, {target_lps[1]:.1f}, {target_lps[2]:.1f}] mm | "
        f"Nearest Dist: {dist_mm:.2f} mm\n"
        f"Proposal Rank: {cand_rank if cand_rank else 'N/A'} / 4,000 | "
        f"Classifier Rank: {clf_rank if clf_rank else 'N/A'} / 4,000 | "
        f"Calibrated Prob: {prob:.6f}"
    )
    fig.suptitle(info_text, fontsize=11, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Overlay saved to {out_path}", flush=True)

def generate_key_case_overlays():
    # Load ground truth
    gt_path = EXT_DIR / "test_ground_truth.json"
    with open(gt_path) as f:
        gt_data = json.load(f)

    # Load scored matches and ranks
    scores_path = EXT_DIR / "scores_cap4000.csv"
    if not scores_path.exists():
        print("Scores CSV not available yet.")
        return

    print("Parsing scores for key cases...", flush=True)
    cands_by_patient = defaultdict(list)
    with open(scores_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = r['patient_id']
            # Only keep rows for patients we want to visualize to save RAM
            cands_by_patient[pid].append(r)

    # Key cases to visualize
    key_cases = [
        "LUNGx-CT001", "LUNGx-CT002", "LUNGx-CT003", "LUNGx-CT004",
        "LUNGx-CT005", "LUNGx-CT008", "LUNGx-CT016", "LUNGx-CT017"
    ]

    for pid in key_cases:
        if pid not in cands_by_patient or pid not in gt_data:
            continue
        cands = cands_by_patient[pid]
        # Sort candidates by probability descending to determine classifier rank
        sorted_by_prob = sorted(cands, key=lambda x: float(x['probability_calibrated']), reverse=True)
        clf_ranks = {c['candidate_id']: idx + 1 for idx, c in enumerate(sorted_by_prob)}

        targets = gt_data[pid]
        dcm_dir = RAW_TEST / pid
        dcm_files = sorted(list(dcm_dir.glob("*.dcm")))
        if not dcm_files:
            continue

        for tg in targets:
            nidx = int(tg['nodule_index'])
            inst_num = int(tg['instance_number'])
            x1, y1 = int(tg['pixel_x_1based']), int(tg['pixel_y_1based'])

            target_dcm = None
            for p in dcm_files:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                if int(ds.InstanceNumber) == inst_num:
                    target_dcm = ds
                    break
            if not target_dcm:
                continue

            ipp = [float(x) for x in target_dcm.ImagePositionPatient]
            spacing = [float(x) for x in target_dcm.PixelSpacing]
            iop = [float(x) for x in target_dcm.ImageOrientationPatient]
            x0, y0 = x1 - 1, y1 - 1
            gt_X = ipp[0] + x0 * spacing[0] * iop[0] + y0 * spacing[1] * iop[3]
            gt_Y = ipp[1] + x0 * spacing[0] * iop[1] + y0 * spacing[1] * iop[4]
            gt_Z = ipp[2]
            target_lps = np.array([gt_X, gt_Y, gt_Z], dtype=np.float64)

            # Find matching candidate or closest candidate
            matched_cands = [c for c in cands if c['is_nodule_match'] == '1' and str(c['target_id']) == str(nidx)]
            if matched_cands:
                best_c = matched_cands[0]
                matched = True
                dist_mm = float(best_c['target_distance_mm'])
                cand_rank = int(best_c['rank'])
                clf_rank = clf_ranks[best_c['candidate_id']]
                prob = float(best_c['probability_calibrated'])
            else:
                matched = False
                # compute min distance across all candidates
                c_coords = np.array([[float(c['coordX']), float(c['coordY']), float(c['coordZ'])] for c in cands])
                dists = np.linalg.norm(c_coords - target_lps, axis=1)
                min_idx = int(np.argmin(dists))
                dist_mm = float(dists[min_idx])
                cand_rank = int(cands[min_idx]['rank'])
                clf_rank = clf_ranks[cands[min_idx]['candidate_id']]
                prob = float(cands[min_idx]['probability_calibrated'])

            out_png = OUT_OVERLAYS / f"{pid}_target{nidx}.png"
            generate_target_overlay(
                pid, nidx, target_lps, matched, dist_mm, cand_rank, clf_rank, prob, out_png
            )

def main():
    print("--- Test Cohort Overlays & FROC Visualization Generator ---")
    summary_path = EXT_DIR / "test_cohort_cap4000_evaluation_summary.json"
    froc_png_path = EXT_DIR / "froc_test_cohort_cap4000.png"
    if summary_path.exists():
        plot_froc_curves(summary_path, froc_png_path)
    generate_key_case_overlays()

if __name__ == "__main__":
    main()
