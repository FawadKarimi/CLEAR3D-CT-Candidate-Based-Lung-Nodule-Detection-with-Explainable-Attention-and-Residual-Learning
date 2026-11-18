import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

EXP3 = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline/runs/exp_fold0_focal_and_matched_inner_20261115")
EXP4 = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline/runs/exp_fold0_round2_focal_mining_20261115")

sys.path.insert(0, str(EXP3))
import run_fold0_focal_experiment as m3

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def write_status_atomic(stage: str, extra: Dict[str, Any] = None) -> None:
    payload = {"stage": stage, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid()}
    if extra:
        payload.update(extra)
    tmp = EXP4 / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, EXP4 / "status.json")

def log(msg: str, log_path: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")

def run_official_eval_fixed(pred_csv: Path, eval_out_dir: Path) -> Dict[str, Any]:
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    scans = sorted({r["seriesuid"] for r in rows})
    suid_p = eval_out_dir / "seriesuids.csv"
    with open(suid_p, "w", newline="") as f:
        w = csv.writer(f)
        for s in scans:
            w.writerow([s])
    cmd = [
        sys.executable, str(m3.EVAL_PY),
        str(m3.REFD / "annotations.csv"),
        str(m3.REFD / "annotations_excluded.csv"),
        str(suid_p), str(pred_csv), str(eval_out_dir),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    fps_grid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    froc_files = [p for p in eval_out_dir.glob("froc_*.txt") if "bootstrapping" not in p.name]
    sens_at_fps: Dict[str, float] = {}
    if froc_files:
        pts = []
        for line in froc_files[0].read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) == 3:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        pts.sort(key=lambda x: x[0])
        f_arr = np.array([p[0] for p in pts], dtype=np.float64)
        s_arr = np.array([p[1] for p in pts], dtype=np.float64)
        for g in fps_grid:
            sens_at_fps[str(g)] = float(np.interp(g, f_arr, s_arr))
    cpm = float(np.mean(list(sens_at_fps.values()))) if sens_at_fps else 0.0

    boot_summary: Dict[str, Any] = {}
    boot_files = list(eval_out_dir.glob("froc_*_bootstrapping.csv"))
    if boot_files:
        with open(boot_files[0], newline="") as f:
            b_rows = list(csv.DictReader(f))
        if b_rows:
            fp_rates = np.array([float(r["FPrate"]) for r in b_rows], dtype=np.float64)
            s_mean = np.array([float(r["Sensivity[Mean]"]) for r in b_rows], dtype=np.float64)
            s_low = np.array([float(r["Sensivity[Lower bound]"]) for r in b_rows], dtype=np.float64)
            s_high = np.array([float(r["Sensivity[Upper bound]"]) for r in b_rows], dtype=np.float64)
            boot_summary = {
                "cpm_mean": float(np.mean([np.interp(g, fp_rates, s_mean) for g in fps_grid])),
                "cpm_lower_bound": float(np.mean([np.interp(g, fp_rates, s_low) for g in fps_grid])),
                "cpm_upper_bound": float(np.mean([np.interp(g, fp_rates, s_high) for g in fps_grid])),
                "froc_lower_at_fps": {str(g): float(np.interp(g, fp_rates, s_low)) for g in fps_grid},
                "froc_upper_at_fps": {str(g): float(np.interp(g, fp_rates, s_high)) for g in fps_grid},
            }
    return {"scans": len(scans), "cpm": cpm, "froc": sens_at_fps, "official_bootstrap_bounds": boot_summary}

def scan_bootstrap_cls(rows: List[Dict[str, Any]], probs: np.ndarray, thresh: float, n_boot: int = 1000, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    scans = sorted({r["seriesuid"] for r in rows})
    scan_to_idx: Dict[str, List[int]] = {s: [] for s in scans}
    y_all = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    for i, r in enumerate(rows):
        scan_to_idx[r["seriesuid"]].append(i)
    scan_arrays = [(y_all[idx], probs[idx]) for s in scans for idx in [scan_to_idx[s]]]
    n_scans = len(scan_arrays)
    metrics_boot: Dict[str, List[float]] = {k: [] for k in ["sensitivity", "precision", "specificity", "accuracy", "positive_f1"]}
    for _ in range(n_boot):
        pick = rng.integers(0, n_scans, size=n_scans)
        yb = np.concatenate([scan_arrays[i][0] for i in pick])
        pb = np.concatenate([scan_arrays[i][1] for i in pick])
        pred = (pb >= thresh).astype(np.int64)
        tp = int(np.sum((pred == 1) & (yb == 1)))
        fp = int(np.sum((pred == 1) & (yb == 0)))
        tn = int(np.sum((pred == 0) & (yb == 0)))
        fn = int(np.sum((pred == 0) & (yb == 1)))
        metrics_boot["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        metrics_boot["precision"].append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        metrics_boot["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        metrics_boot["accuracy"].append((tp + tn) / len(yb))
        metrics_boot["positive_f1"].append((2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return {
        k: {"mean": float(np.mean(v)), "ci95_low": float(np.percentile(v, 2.5)), "ci95_high": float(np.percentile(v, 97.5))}
        for k, v in metrics_boot.items()
    }

def run_experiment() -> None:
    t_all = time.time()
    log_path = EXP4 / "experiment_run.log"
    ckpt_dir = EXP4 / "checkpoints"
    pred_dir = EXP4 / "predictions"
    eval_dir = EXP4 / "official_cad_eval"
    device = torch.device("cuda:0")

    try:
        write_status_atomic("step0_preflight")
        sel = m3.verify_governance_hold()
        provenance = {
            "run_fold0_round2_focal_experiment.py": sha256_file(EXP4 / "run_fold0_round2_focal_experiment.py"),
            "arm_c_focal_gamma2.pt": sha256_file(EXP3 / "checkpoints/arm_c_focal_gamma2.pt"),
            "inner_focal_gamma2.pt": sha256_file(EXP3 / "checkpoints/inner_focal_gamma2.pt"),
            "selection-001.json": sha256_file(m3.V2 / "selection-001.json"),
        }
        log(f"=== STEP 0: Pre-Flight Governance & Provenance === {json.dumps(provenance)}", log_path)

        # Load Round-1 Focal checkpoints as Round-2 Miners (strictly cohort-isolated)
        armc_ckpt = torch.load(EXP3 / "checkpoints/arm_c_focal_gamma2.pt", map_location="cpu", weights_only=True)
        infoc_ckpt = torch.load(EXP3 / "checkpoints/inner_focal_gamma2.pt", map_location="cpu", weights_only=True)
        armc_sd = armc_ckpt["model_state_dict"]
        infoc_sd = infoc_ckpt["model_state_dict"]

        write_status_atomic("stage1a_load_data_and_rand")
        folds_list, scans_by_fold, pos_cands, clean_negs, eval_cands_by_scan = m3.load_cohort_and_candidates()
        fold0_scans = scans_by_fold[0]
        fold1_scans = scans_by_fold[1]
        assert len({s["seriesuid"] for s in fold0_scans} & {s["seriesuid"] for s in fold1_scans}) == 0

        t_s1a = time.time()
        pos_p51_all, pos_v51_all, pos_folds_all = m3.extract_all_positive_crops(folds_list, pos_cands)
        rand_p51, rand_v51, rand_folds = m3.extract_random_neg_crops_per_fold(folds_list, clean_negs, m3.RAND_PER_FOLD, seed=777)
        s1a_sec = float(time.time() - t_s1a)
        log(f"Stage 1A extracted {len(pos_p51_all)} pos and {len(rand_p51)} rand neg crops in {s1a_sec:.1f}s", log_path)

        # Prepare Outer Fold 0 (Folds 1..9) & Inner (Folds 2..9) pos/rand arrays identically to EXP3
        f0_pos_mask = np.isin(pos_folds_all, list(range(1, 10)))
        f0_rand_mask = np.isin(rand_folds, list(range(1, 10)))
        f0_pos_p51, f0_pos_v51 = pos_p51_all[f0_pos_mask], pos_v51_all[f0_pos_mask]
        f0_rand_p51, f0_rand_v51 = rand_p51[f0_rand_mask], rand_v51[f0_rand_mask]

        in_pos_mask = np.isin(pos_folds_all, list(range(2, 10)))
        in_rand_mask = np.isin(rand_folds, list(range(2, 10)))
        in_pos_p51, in_pos_v51 = pos_p51_all[in_pos_mask], pos_v51_all[in_pos_mask]
        in_rand_p51, in_rand_v51 = rand_p51[in_rand_mask], rand_v51[in_rand_mask]

        # === STAGE 2A: Mine Round-2 hard negatives using Round-1 Focal models ===
        write_status_atomic("stage2a_mine_round2_focal")
        f0_train_scans = [s for f in range(1, 10) for s in scans_by_fold[f]]
        f0_hard_meta, in_hard_meta, mine_sec = m3.mine_both_scouts_single_pass(
            armc_sd, infoc_sd, f0_train_scans, clean_negs, device, top_k=12000
        )
        # Verify cohort isolation on mined hard negatives
        assert all(m[0] != 0 for m in f0_hard_meta), "Fold 0 leaked into f0_hard_meta!"
        assert all(m[0] not in (0, 1) for m in in_hard_meta), "Fold 0 or 1 leaked into in_hard_meta!"
        log(f"Stage 2A Round-2 mining complete in {mine_sec:.1f}s | F0 top12000={f0_hard_meta[-1][5]:.4f} | Inner top12000={in_hard_meta[-1][5]:.4f}", log_path)

        write_status_atomic("stage2b_extract_round2_hard_banks")
        t_hb = time.time()
        f0_hard_p51, f0_hard_v51, f0_hard_scores = m3.extract_hard_bank(f0_hard_meta, folds_list)
        in_hard_p51, in_hard_v51, in_hard_scores = m3.extract_hard_bank(in_hard_meta, folds_list)
        hb_sec = float(time.time() - t_hb)
        f0_sec_weights = 0.5 + 0.5 * f0_hard_scores
        in_sec_weights = 0.5 + 0.5 * in_hard_scores
        log(f"Stage 2B extracted both 12,000 Round-2 hard banks in {hb_sec:.1f}s", log_path)

        # === STAGE 3: Train Arm D (Round-2 Focal) & Inner D (Round-2 Focal) ===
        write_status_atomic("stage3a_train_arm_d_round2_focal")
        armd_sd, armd_loss, armd_traces, armd_sec = m3.train_main_model_split_rng(
            "cuda:0", f0_pos_p51, f0_pos_v51, f0_rand_p51, f0_rand_v51, f0_hard_p51, f0_hard_v51, f0_sec_weights,
            loss_mode="focal", seed_init=4200, seed_stream_base=5000,
        )
        torch.save({"model_state_dict": armd_sd, "arm": "arm_d_round2_focal_gamma2", "epochs": m3.MAIN_EPOCHS}, ckpt_dir / "arm_d_round2_focal_gamma2.pt")
        log(f"Stage 3A Arm D (Round-2 Focal) trained in {armd_sec:.1f}s | final_loss={armd_loss:.4f}", log_path)

        write_status_atomic("stage3b_train_inner_d_round2_focal")
        ind_sd, ind_loss, ind_traces, ind_sec = m3.train_main_model_split_rng(
            "cuda:0", in_pos_p51, in_pos_v51, in_rand_p51, in_rand_v51, in_hard_p51, in_hard_v51, in_sec_weights,
            loss_mode="focal", seed_init=4201, seed_stream_base=5100,
        )
        torch.save({"model_state_dict": ind_sd, "arm": "inner_d_round2_focal_gamma2", "epochs": m3.MAIN_EPOCHS}, ckpt_dir / "inner_d_round2_focal_gamma2.pt")
        log(f"Stage 3B Inner D (Round-2 Focal) trained in {ind_sec:.1f}s | final_loss={ind_loss:.4f}", log_path)

        del rand_p51, rand_v51, f0_hard_p51, f0_hard_v51, in_hard_p51, in_hard_v51, pos_p51_all, pos_v51_all
        gc.collect()

        # === STAGE 4: Score Outer Fold 0 (Arm D) & Inner Fold 1 (Inner D) ===
        write_status_atomic("stage4_score_outer_and_inner")
        f0_rows, f0_score_sec = m3.score_scans_4view_tta({"arm_d": armd_sd}, fold0_scans, eval_cands_by_scan, fold_label=0, device=device)
        armd_csv = pred_dir / "arm_d_fold0_predictions.csv"
        m3.write_pred_csv(f0_rows["arm_d"], armd_csv)

        f1_rows, f1_score_sec = m3.score_scans_4view_tta({"in_d": ind_sd}, fold1_scans, eval_cands_by_scan, fold_label=1, device=device)
        ind_csv = pred_dir / "inner_d_fold1_predictions.csv"
        m3.write_pred_csv(f1_rows["in_d"], ind_csv)
        log(f"Stage 4 scored Arm D Fold 0 ({f0_score_sec:.1f}s) & Inner D Fold 1 ({f1_score_sec:.1f}s)", log_path)

        # === STAGE 5: Fixed Matched-Recipe Nested Calibration + Official FROC + Scan Bootstrap ===
        write_status_atomic("stage5_eval_and_calibration")
        cal_in_d = m3.fit_nested_platt_and_thresholds(
            np.array([float(r["probability"]) for r in f1_rows["in_d"]], dtype=np.float64),
            np.array([int(r["label"]) for r in f1_rows["in_d"]], dtype=np.int64),
        )
        p_raw = np.array([float(r["probability"]) for r in f0_rows["arm_d"]], dtype=np.float64)
        y_out = np.array([int(r["label"]) for r in f0_rows["arm_d"]], dtype=np.int64)
        outer_d = m3.apply_nested_calibration_to_outer(p_raw, y_out, cal_in_d)

        cal_rows = []
        for r, pc in zip(f0_rows["arm_d"], outer_d["p_cal_outer"]):
            rc = dict(r)
            rc["probability"] = f"{float(pc):.8f}"
            cal_rows.append(rc)
        cal_csv = pred_dir / "arm_d_fold0_calibrated_predictions.csv"
        m3.write_pred_csv(cal_rows, cal_csv)

        froc_raw = run_official_eval_fixed(armd_csv, eval_dir / "arm_d_raw")
        froc_cal = run_official_eval_fixed(cal_csv, eval_dir / "arm_d_cal")
        boot_05 = scan_bootstrap_cls(f0_rows["arm_d"], p_raw, 0.5)
        boot_tau = scan_bootstrap_cls(f0_rows["arm_d"], p_raw, cal_in_d["tau_raw_0_star"])

        summary = {
            "governance": {"reference_release": sel.get("reference_release"), "withheld_subset0_excluded": True, "quarantined_subset1_excluded": True},
            "provenance_sha256": provenance,
            "compute_costs_seconds": {
                "stage1a_pos_rand_extraction": s1a_sec,
                "stage2a_round2_mining_both": mine_sec,
                "stage2b_round2_hard_extraction": hb_sec,
                "arm_d_round2_focal_22ep": armd_sec,
                "inner_d_round2_focal_22ep": ind_sec,
                "outer_fold0_tta_1arm": f0_score_sec,
                "inner_fold1_tta_1model": f1_score_sec,
                "total_wall_clock": float(time.time() - t_all),
            },
            "rng_stream_traces": {"arm_d": armd_traces, "inner_d": ind_traces},
            "calibration_fit_inner_d_fold1": cal_in_d,
            "results_arm_d_round2_focal": {
                "roc_auc": outer_d["roc_auc"],
                "pr_auc": outer_d["pr_auc"],
                "rank_order_preserved": outer_d["rank_order_preserved"],
                "metrics_matched_inner": outer_d["metrics"],
                "bootstrap_ci95_raw_0p5": boot_05,
                "bootstrap_ci95_matched_tau_star": boot_tau,
                "froc_raw": froc_raw,
                "froc_cal": froc_cal,
                "froc_cpm_abs_diff": float(abs(froc_raw["cpm"] - froc_cal["cpm"])),
            },
        }
        (EXP4 / "results_summary.json").write_text(json.dumps(summary, indent=2))
        write_status_atomic("completed", {"total_wall_clock_sec": summary["compute_costs_seconds"]["total_wall_clock"]})
        log(f"EXPERIMENT COMPLETE in {summary['compute_costs_seconds']['total_wall_clock']:.1f}s | Arm D CPM={froc_raw['cpm']:.4f}", log_path)
    except Exception as e:
        write_status_atomic("failed", {"error": str(e)})
        log(f"EXPERIMENT FAILED: {e}\n{traceback.format_exc()}", log_path)
        raise

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["verify-fixtures", "run-experiment"], default="verify-fixtures")
    args = parser.parse_args()
    m3.verify_governance_hold()
    if args.mode == "verify-fixtures":
        # Quick fixture verification: governance HOLD, bootstrapping CSV parser on EXP3 artifact, and checkpoint load
        sel = m3.verify_governance_hold()
        assert sel.get("reference_release") == "HOLD"
        assert (EXP3 / "checkpoints/arm_c_focal_gamma2.pt").exists()
        assert (EXP3 / "checkpoints/inner_focal_gamma2.pt").exists()
        boot_csv = EXP3 / "official_cad_eval/arm_a_raw/froc_arm_a_fold0_predictions_bootstrapping.csv"
        with open(boot_csv, newline="") as f:
            b_rows = list(csv.DictReader(f))
        assert len(b_rows) > 0 and "FPrate" in b_rows[0] and "Sensivity[Mean]" in b_rows[0]
        print("ALL ROUND-2 FOCAL FIXTURE ASSERTIONS PASSED.")
    else:
        run_experiment()

if __name__ == "__main__":
    main()
