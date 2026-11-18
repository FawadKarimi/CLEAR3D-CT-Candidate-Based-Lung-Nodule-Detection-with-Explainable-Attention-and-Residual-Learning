import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

EXP3 = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline/runs/exp_fold0_focal_and_matched_inner_20261115")
EXP2 = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline/runs/exp_fold0_matched_negsampling_20261115_v2")
EVAL_OUT = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline/runs/exp_fold0_focal_and_matched_inner_20261115_eval_cpu")

sys.path.insert(0, str(EXP3))
import run_fold0_focal_experiment as exp3_mod

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_csv(p: Path):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

def run_official_eval_fixed(pred_csv: Path, eval_dir: Path):
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(pred_csv)
    scans = sorted({r["seriesuid"] for r in rows})
    suid_p = eval_dir / "seriesuids.csv"
    with open(suid_p, "w", newline="") as f:
        w = csv.writer(f)
        for s in scans:
            w.writerow([s])
    cmd = [
        sys.executable,
        str(exp3_mod.EVAL_PY),
        str(exp3_mod.REFD / "annotations.csv"),
        str(exp3_mod.REFD / "annotations_excluded.csv"),
        str(suid_p),
        str(pred_csv),
        str(eval_dir),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    fps_grid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    froc_files = [p for p in eval_dir.glob("froc_*.txt") if "bootstrapping" not in p.name]
    sens_at_fps = {}
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

    boot_summary = {}
    boot_files = list(eval_dir.glob("froc_*_bootstrapping.csv"))
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

def scan_bootstrap_cls(rows, probs, thresh, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    scans = sorted({r["seriesuid"] for r in rows})
    scan_to_idx = {s: [] for s in scans}
    y_all = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    for i, r in enumerate(rows):
        scan_to_idx[r["seriesuid"]].append(i)
    scan_arrays = [(y_all[idx], probs[idx]) for s in scans for idx in [scan_to_idx[s]]]
    n_scans = len(scan_arrays)
    metrics_boot = {k: [] for k in ["sensitivity", "precision", "specificity", "accuracy", "positive_f1"]}
    for _ in range(n_boot):
        pick = rng.integers(0, n_scans, size=n_scans)
        yb = np.concatenate([scan_arrays[i][0] for i in pick])
        pb = np.concatenate([scan_arrays[i][1] for i in pick])
        pred = (pb >= thresh).astype(np.int64)
        tp = int(np.sum((pred == 1) & (yb == 1)))
        fp = int(np.sum((pred == 1) & (yb == 0)))
        tn = int(np.sum((pred == 0) & (yb == 0)))
        fn = int(np.sum((pred == 0) & (yb == 1)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = (tp + tn) / len(yb)
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        metrics_boot["sensitivity"].append(sens)
        metrics_boot["precision"].append(prec)
        metrics_boot["specificity"].append(spec)
        metrics_boot["accuracy"].append(acc)
        metrics_boot["positive_f1"].append(f1)
    return {
        k: {
            "mean": float(np.mean(v)),
            "ci95_low": float(np.percentile(v, 2.5)),
            "ci95_high": float(np.percentile(v, 97.5)),
        }
        for k, v in metrics_boot.items()
    }

def main():
    sel = exp3_mod.verify_governance_hold()
    arma_csv = EXP3 / "predictions/arm_a_fold0_predictions.csv"
    armc_csv = EXP3 / "predictions/arm_c_fold0_predictions.csv"
    ince_csv = EXP3 / "predictions/inner_ce_fold1_predictions.csv"
    infoc_csv = EXP3 / "predictions/inner_focal_fold1_predictions.csv"
    unif_csv = EXP2 / "predictions/inner_fold1_predictions.csv"

    arma_rows = load_csv(arma_csv)
    armc_rows = load_csv(armc_csv)
    ince_rows = load_csv(ince_csv)
    infoc_rows = load_csv(infoc_csv)
    unif_rows = load_csv(unif_csv)

    assert [r["candidate_id"] for r in arma_rows] == [r["candidate_id"] for r in armc_rows]
    assert [r["seriesuid"] for r in arma_rows] == [r["seriesuid"] for r in armc_rows]

    cal_in_ce = exp3_mod.fit_nested_platt_and_thresholds(
        np.array([float(r["probability"]) for r in ince_rows], dtype=np.float64),
        np.array([int(r["label"]) for r in ince_rows], dtype=np.int64),
    )
    cal_in_foc = exp3_mod.fit_nested_platt_and_thresholds(
        np.array([float(r["probability"]) for r in infoc_rows], dtype=np.float64),
        np.array([int(r["label"]) for r in infoc_rows], dtype=np.int64),
    )
    cal_in_unif = exp3_mod.fit_nested_platt_and_thresholds(
        np.array([float(r["probability"]) for r in unif_rows], dtype=np.float64),
        np.array([int(r["label"]) for r in unif_rows], dtype=np.int64),
    )

    def eval_arm(name, rows, raw_csv, cal_matched):
        p_raw = np.array([float(r["probability"]) for r in rows], dtype=np.float64)
        y_out = np.array([int(r["label"]) for r in rows], dtype=np.int64)
        outer_matched = exp3_mod.apply_nested_calibration_to_outer(p_raw, y_out, cal_matched)
        outer_unif = exp3_mod.apply_nested_calibration_to_outer(p_raw, y_out, cal_in_unif)

        cal_rows = []
        for r, pc in zip(rows, outer_matched["p_cal_outer"]):
            rc = dict(r)
            rc["probability"] = f"{float(pc):.8f}"
            cal_rows.append(rc)
        cal_csv = EVAL_OUT / f"predictions/{name}_fold0_calibrated_predictions.csv"
        exp3_mod.write_pred_csv(cal_rows, cal_csv)

        froc_raw = run_official_eval_fixed(raw_csv, EVAL_OUT / f"official_cad_eval/{name}_raw")
        froc_cal = run_official_eval_fixed(cal_csv, EVAL_OUT / f"official_cad_eval/{name}_cal")

        tau_raw_matched = cal_matched["tau_raw_0_star"]
        tau_raw_unif = cal_in_unif["tau_raw_0_star"]
        boot_05 = scan_bootstrap_cls(rows, p_raw, 0.5)
        boot_matched = scan_bootstrap_cls(rows, p_raw, tau_raw_matched)
        boot_unif = scan_bootstrap_cls(rows, p_raw, tau_raw_unif)

        return {
            "rows": len(rows),
            "scans": len({r["seriesuid"] for r in rows}),
            "positives": int(np.sum(y_out)),
            "roc_auc": outer_matched["roc_auc"],
            "pr_auc": outer_matched["pr_auc"],
            "rank_order_preserved": outer_matched["rank_order_preserved"],
            "metrics_matched_inner": outer_matched["metrics"],
            "metrics_uniform_inner_transfer_check": outer_unif["metrics"],
            "bootstrap_ci95_raw_0p5": boot_05,
            "bootstrap_ci95_matched_tau_star": boot_matched,
            "bootstrap_ci95_uniform_tau_star": boot_unif,
            "froc_raw": froc_raw,
            "froc_cal": froc_cal,
            "froc_cpm_abs_diff": float(abs(froc_raw["cpm"] - froc_cal["cpm"])),
        }

    res_arma = eval_arm("arm_a", arma_rows, arma_csv, cal_in_ce)
    res_armc = eval_arm("arm_c", armc_rows, armc_csv, cal_in_foc)

    hashes = {}
    for p in sorted((EXP3 / "checkpoints").glob("*.pt")) + sorted((EXP3 / "predictions").glob("*.csv")):
        hashes[str(p.relative_to(EXP3))] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}

    summary = {
        "governance": {"reference_release": sel.get("reference_release"), "withheld_subset0_excluded": True, "quarantined_subset1_excluded": True},
        "source_exp3_dir": str(EXP3),
        "artifact_hashes_exp3": hashes,
        "calibration_fits_inner_fold1": {"matched_ce": cal_in_ce, "matched_focal": cal_in_foc, "uniform_run2": cal_in_unif},
        "results": {"arm_a_ce_control": res_arma, "arm_c_focal_intervention": res_armc},
    }
    out_json = EVAL_OUT / "results_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print("SAVED:", out_json)
    for k, res in summary["results"].items():
        print(f"=== {k} ===")
        print(f"  ROC-AUC={res['roc_auc']:.4f} | PR-AUC={res['pr_auc']:.4f} | FROC CPM(raw)={res['froc_raw']['cpm']:.4f} | CPM(cal)={res['froc_cal']['cpm']:.4f}")
        print(f"  FROC bounds: {res['froc_raw']['official_bootstrap_bounds']}")
        for mname, mdict in res["metrics_matched_inner"].items():
            print(f"  [Matched] {mname}: {mdict}")
        for mname, mdict in res["metrics_uniform_inner_transfer_check"].items():
            print(f"  [UnifXfer] {mname}: {mdict}")

if __name__ == "__main__":
    main()
