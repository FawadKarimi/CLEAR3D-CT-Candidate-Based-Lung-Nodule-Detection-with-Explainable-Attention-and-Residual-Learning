"""
Orchestrator for Broader LUNA16 10-Fold Cross-Validation
Manages parallel execution across GPU 1 and GPU 3.
Pools predictions upon completion and runs official LUNA16 CAD evaluation.
"""
import concurrent.futures
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
CV_DIR = ROOT / "runs/crossval-10fold-clear3d-20261109"
CKPT_DIR = CV_DIR / "checkpoints"
PRED_DIR = CV_DIR / "predictions"
LOG_DIR = CV_DIR / "logs"
STATUS_FILE = CV_DIR / "status.json"
PYTHON_BIN = "/experiment-storage/chaidir/env/env-310-fawad/bin/python3"
REFD = BASE / "data/reference/luna16-zenodo-3723295/resolved-reference-v1"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def is_fold_done(fold_k):
    pred_path = PRED_DIR / f"fold_{fold_k}_predictions.csv"
    ckpt_path = CKPT_DIR / f"fold_{fold_k}.pt"
    if not (pred_path.exists() and ckpt_path.exists()):
        return False
    if pred_path.stat().st_size < 1000 or ckpt_path.stat().st_size < 1000:
        return False
    if STATUS_FILE.exists():
        data = json.loads(STATUS_FILE.read_text())
        fold_info = data.get("folds", {}).get(str(fold_k), {})
        if fold_info.get("status") == "completed":
            return True
    return False

def execute_fold(fold_k, device_id):
    if is_fold_done(fold_k):
        print(f"[Orchestrator] Fold {fold_k} is already completed. Skipping.", flush=True)
        return fold_k, True
    
    print(f"[Orchestrator] Launching Fold {fold_k} on GPU {device_id}...", flush=True)
    cmd = [
        PYTHON_BIN, "-u", str(ROOT / "run_outer_fold_cross_validation_20261109.py"),
        "--fold", str(fold_k),
        "--device", str(device_id),
        "--epochs", "12"
    ]
    stdout_log = LOG_DIR / f"fold_{fold_k}_stdout.log"
    with open(stdout_log, "a") as f_out:
        p = subprocess.Popen(cmd, stdout=f_out, stderr=subprocess.STDOUT)
        ret = p.wait()
        
    if ret == 0 and is_fold_done(fold_k):
        print(f"[Orchestrator] Fold {fold_k} completed successfully on GPU {device_id}!", flush=True)
        return fold_k, True
    else:
        print(f"[Orchestrator] Fold {fold_k} FAILED on GPU {device_id} (code {ret})!", flush=True)
        return fold_k, False

def pool_and_evaluate():
    print("[Orchestrator] All 10 folds verified! Concatenating pooled predictions...", flush=True)
    pooled_csv = CV_DIR / "pooled_oof_predictions.csv"
    
    total_rows = 0
    pos_count = 0
    scans_seen = set()
    
    fieldnames = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "candidate_id", "rank", "label"]
    with open(pooled_csv, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for fold_k in range(10):
            fold_csv = PRED_DIR / f"fold_{fold_k}_predictions.csv"
            assert fold_csv.exists(), f"Missing {fold_csv}"
            with open(fold_csv, "r", newline="") as f_in:
                reader = csv.DictReader(f_in)
                for r in reader:
                    writer.writerow(r)
                    total_rows += 1
                    scans_seen.add(r["seriesuid"])
                    if int(r["label"]) == 1:
                        pos_count += 1
                        
    pooled_sha = sha256_file(pooled_csv)
    print(f"[Orchestrator] Pooled predictions written: {pooled_csv}")
    print(f"  Total rows: {total_rows:,} (expected 682,787)")
    print(f"  Positive candidates: {pos_count:,} (expected 1,387)")
    print(f"  Unique scans: {len(scans_seen):,} (expected 797)")
    print(f"  SHA-256: {pooled_sha}")
    
    # Save manifest of all artifacts
    eval_out_dir = CV_DIR / "official_cad_eval"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run official evaluator
    evaluator_script = ROOT / "paper-ablation-verify-package-20261107-004/evaluator/noduleCADEvaluationLUNA16.py"
    ann_file = REFD / "annotations.csv"
    exc_file = REFD / "annotations_excluded.csv"
    seriesuids_csv = CV_DIR / "seriesuids_list.csv"
    with open(seriesuids_csv, "w") as f:
        f.write("seriesuid\n" + "\n".join(sorted(scans_seen)) + "\n")
        
    print(f"[Orchestrator] Running official LUNA16 evaluator on pooled predictions...", flush=True)
    cmd_eval = [
        PYTHON_BIN, str(evaluator_script),
        str(ann_file),
        str(exc_file),
        str(seriesuids_csv),
        str(pooled_csv),
        str(eval_out_dir)
    ]
    res = subprocess.run(cmd_eval, capture_output=True, text=True)
    print(f"Evaluator stdout:\n{res.stdout}\nEvaluator stderr:\n{res.stderr}")
    print("[Orchestrator] Completed official evaluation!")

if __name__ == "__main__":
    print("=== STARTING 10-FOLD BROADER CROSS-VALIDATION ORCHESTRATION ===", flush=True)
    
    # Check what folds are pending
    pending = [k for k in range(10) if not is_fold_done(k)]
    print(f"Pending folds: {pending}", flush=True)
    
    # Worker assignments: alternate GPU 1 and GPU 3
    # GPU 1 takes even folds, GPU 3 takes odd folds
    gpu1_folds = [k for k in pending if k % 2 == 0]
    gpu3_folds = [k for k in pending if k % 2 == 1]
    
    print(f"GPU 1 worker queue: {gpu1_folds}")
    print(f"GPU 3 worker queue: {gpu3_folds}")
    
    def worker(folds, dev_id):
        for f in folds:
            ok = execute_fold(f, dev_id)
            if not ok[1]:
                print(f"Worker for GPU {dev_id} stopped due to failure on fold {f}")
                return False
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fut1 = executor.submit(worker, gpu1_folds, 1)
        fut3 = executor.submit(worker, gpu3_folds, 3)
        res1 = fut1.result()
        res3 = fut3.result()
        
    all_done = all(is_fold_done(k) for k in range(10))
    if all_done:
        print("[Orchestrator] All 10 folds successfully finished! Proceeding to pooling and evaluation.")
        pool_and_evaluate()
    else:
        print("[Orchestrator] Some folds did not complete. Check status and logs.")
