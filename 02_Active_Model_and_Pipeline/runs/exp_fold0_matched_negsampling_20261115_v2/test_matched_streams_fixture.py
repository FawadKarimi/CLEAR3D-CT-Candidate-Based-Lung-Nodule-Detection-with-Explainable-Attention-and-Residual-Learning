#!/usr/bin/env python3
"""
Deterministic Fixture Verification Suite for Fold 0 Matched Negative-Sampling
and Nested Patient-Grouped Calibration Experiment.
Does NOT launch GPU training.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from run_fold0_matched_experiment import (
    CLEAR3DMSFv2,
    MAIN_EPOCHS,
    MAIN_STEPS,
    LABEL_SMOOTHING,
    make_training_rng_streams,
    compute_epoch_lr_and_sw,
    sample_aug_params_dev708,
    aug_gpu_from_params,
    validate_dev_cohort,
    validate_sec_weights,
    load_checkpoint_cpu,
    fit_nested_platt_and_thresholds,
    apply_nested_calibration_to_outer,
    SEL_PATH,
    PREV_RUN,
    EXP_DIR,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def simulate_schedule_and_streams(
    n_pos: int,
    n_rand: int,
    sec_bank_indices: np.ndarray,
    sec_bank_weights: np.ndarray,
    seed_init: int = 4200,
    seed_stream_base: int = 5000,
    epochs: int = MAIN_EPOCHS,
    steps_per_epoch: int = MAIN_STEPS,
):
    torch.manual_seed(seed_init)
    model = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2)

    streams = make_training_rng_streams(seed_stream_base)
    rng_pos = streams["rng_pos"]
    rng_rand = streams["rng_rand"]
    rng_sec = streams["rng_sec"]
    rng_aug = streams["rng_aug"]

    ip_all, ir_all, ih_all = [], [], []
    aug_all = []
    lr_all, sw_all = [], []

    for ep in range(epochs):
        lr, sw = compute_epoch_lr_and_sw(ep, epochs)
        lr_all.append(lr)
        sw_all.append(sw)
        for _ in range(steps_per_epoch):
            ip = rng_pos.choice(n_pos, size=10, replace=True)
            ir = rng_rand.choice(n_rand, size=12, replace=False)
            ih_local = rng_sec.choice(len(sec_bank_indices), size=10, replace=False, p=sec_bank_weights)
            ih = sec_bank_indices[ih_local]
            aug_p = sample_aug_params_dev708(rng_aug, B=32)

            ip_all.append(ip)
            ir_all.append(ir)
            ih_all.append(ih)
            aug_all.append(aug_p)

    return {
        "init_state": init_state,
        "param_count": sum(p.numel() for p in model.parameters()),
        "buffer_count": sum(b.numel() for b in model.buffers()),
        "tensor_count": len(init_state),
        "opt_defaults": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in opt.defaults.items()},
        "lr_schedule": np.array(lr_all, dtype=np.float64),
        "sw_schedule": np.stack(sw_all, axis=0),
        "ip": np.stack(ip_all, axis=0),
        "ir": np.stack(ir_all, axis=0),
        "ih": np.stack(ih_all, axis=0),
        "aug": aug_all,
    }


def test_cohort_and_weight_guards():
    scans = [{"patient_id": f"P{i}", "series_uid": f"S{i}", "scan_id": f"ID{i}", "fold": i % 10} for i in range(20)]
    validate_dev_cohort(scans, {"Q1"}, {"QID1"}, {"WS1"}, {"WP1"}, expected_scans=20, expected_patients=20)
    # Withheld leak test
    bad_withheld = list(scans)
    bad_withheld[0] = {"patient_id": "WP1", "series_uid": "S0", "scan_id": "ID0", "fold": 0}
    try:
        validate_dev_cohort(bad_withheld, {"Q1"}, {"QID1"}, {"WS1"}, {"WP1"}, expected_scans=20, expected_patients=20)
        raise RuntimeError("Should have caught withheld patient leak!")
    except AssertionError:
        pass
    # Quarantined leak test
    bad_quar = list(scans)
    bad_quar[0] = {"patient_id": "Q1", "series_uid": "S0", "scan_id": "ID0", "fold": 0}
    try:
        validate_dev_cohort(bad_quar, {"Q1"}, {"QID1"}, {"WS1"}, {"WP1"}, expected_scans=20, expected_patients=20)
        raise RuntimeError("Should have caught quarantined patient leak!")
    except AssertionError:
        pass
    # Secondary weight guard test
    try:
        validate_sec_weights(np.zeros(12), min_count=10)
        raise RuntimeError("Should have caught all-zero sec_weights!")
    except AssertionError:
        pass
    return True


def main():
    assert test_cohort_and_weight_guards()

    n_pos = 1139
    n_rand = 16000
    n_rem = 514882

    rng_bank = np.random.default_rng(9000)
    control_sec_idx = np.arange(12000, dtype=np.int64)
    fake_scout_scores = np.linspace(0.9939, 0.2330, 12000, dtype=np.float64)
    control_sec_weights = validate_sec_weights(0.5 + 0.5 * fake_scout_scores, min_count=10)

    interv_sec_idx = np.sort(rng_bank.choice(n_rem, size=12000, replace=False))
    interv_sec_weights = validate_sec_weights(np.full(12000, 1.0 / 12000.0, dtype=np.float64), min_count=10)

    ctrl = simulate_schedule_and_streams(
        n_pos=n_pos, n_rand=n_rand,
        sec_bank_indices=control_sec_idx, sec_bank_weights=control_sec_weights,
        seed_init=4200, seed_stream_base=5000,
    )
    intv = simulate_schedule_and_streams(
        n_pos=n_pos, n_rand=n_rand,
        sec_bank_indices=interv_sec_idx, sec_bank_weights=interv_sec_weights,
        seed_init=4200, seed_stream_base=5000,
    )

    init_max_diff = max(
        float((ctrl["init_state"][k] - intv["init_state"][k]).abs().max().item())
        for k in ctrl["init_state"]
    )
    ip_equal = bool(np.array_equal(ctrl["ip"], intv["ip"]))
    ir_equal = bool(np.array_equal(ctrl["ir"], intv["ir"]))
    ih_differs = bool(not np.array_equal(ctrl["ih"], intv["ih"]))
    lr_equal = bool(np.array_equal(ctrl["lr_schedule"], intv["lr_schedule"]))
    sw_equal = bool(np.array_equal(ctrl["sw_schedule"], intv["sw_schedule"]))

    aug_keys = ["shifts", "ks", "flip_z", "flip_x", "scale", "shift"]
    aug_equal_per_key = {
        k: bool(all(np.array_equal(a[k], b[k]) for a, b in zip(ctrl["aug"], intv["aug"])))
        for k in aug_keys
    }

    dummy_p = np.random.randn(32, 51, 51, 51).astype(np.float16)
    dummy_v = np.ones((32, 51, 51, 51), dtype=bool)
    p_c, v_c = aug_gpu_from_params(dummy_p, dummy_v, ctrl["aug"][0], torch.device("cpu"))
    p_i, v_i = aug_gpu_from_params(dummy_p, dummy_v, intv["aug"][0], torch.device("cpu"))
    aug_p_diff = float((p_c - p_i).abs().max().item())
    aug_v_diff = float((v_c - v_i).abs().max().item())

    saved_ckpt_path = PREV_RUN / "checkpoints/fold_0.pt"
    saved_ckpt = load_checkpoint_cpu(saved_ckpt_path)
    saved_sd = saved_ckpt["model_state_dict"]
    saved_arch_matches = (
        len(saved_sd) == ctrl["tensor_count"]
        and all(k in saved_sd and tuple(saved_sd[k].shape) == tuple(ctrl["init_state"][k].shape) for k in ctrl["init_state"])
    )

    df_f0 = pd.read_csv(PREV_RUN / "predictions/fold_0_predictions.csv")
    df_f1 = pd.read_csv(PREV_RUN / "predictions/fold_1_predictions.csv")
    cal_fit = fit_nested_platt_and_thresholds(
        p_raw_inner=df_f1["probability"].to_numpy(dtype=np.float64),
        y_inner=df_f1["label"].to_numpy(dtype=np.int64),
    )
    outer_eval = apply_nested_calibration_to_outer(
        p_raw_outer=df_f0["probability"].to_numpy(dtype=np.float64),
        y_outer=df_f0["label"].to_numpy(dtype=np.int64),
        cal_dict=cal_fit,
    )

    with open(SEL_PATH) as f:
        sel_json = json.load(f)

    report = {
        "governance": {
            "reference_release": sel_json.get("reference_release"),
            "selection_json_sha256": sha256_file(SEL_PATH),
            "cohort_and_weight_guards_verified": True,
            "training_launched": False,
        },
        "saved_control_checkpoint": {
            "path": str(saved_ckpt_path),
            "sha256": sha256_file(saved_ckpt_path),
            "size_bytes": saved_ckpt_path.stat().st_size,
            "fold": int(saved_ckpt.get("fold", -1)),
            "epochs": int(saved_ckpt.get("epochs", -1)),
            "tensor_count": len(saved_sd),
            "param_count": ctrl["param_count"],
            "buffer_count": ctrl["buffer_count"],
            "architecture_matches_fixture": bool(saved_arch_matches),
        },
        "fixture_parity_verification": {
            "steps_verified": int(ctrl["ip"].shape[0]),
            "positive_pool_size": n_pos,
            "primary_neg_bank_size": n_rand,
            "remaining_clean_neg_pool": n_rem,
            "label_smoothing": LABEL_SMOOTHING,
            "init_weights_max_abs_diff": init_max_diff,
            "optimizer_defaults": ctrl["opt_defaults"],
            "lr_schedule_exact_match": lr_equal,
            "sample_weight_schedule_exact_match": sw_equal,
            "positive_indices_ip_exact_match": ip_equal,
            "primary_negative_indices_ir_exact_match": ir_equal,
            "second_bank_indices_ih_changed_as_intended": ih_differs,
            "aug_params_exact_match_per_key": aug_equal_per_key,
            "aug_tensor_output_max_abs_diff": max(aug_p_diff, aug_v_diff),
        },
        "calibration_and_threshold_verification": {
            "calibration_fit_summary_on_fold1_fixture": cal_fit,
            "outer_fold0_rank_order_preserved": outer_eval["rank_order_preserved"],
            "outer_fold0_metrics_by_rule": outer_eval["metrics"],
        },
    }

    assert init_max_diff == 0.0
    assert ip_equal and ir_equal and ih_differs and lr_equal and sw_equal
    assert all(aug_equal_per_key.values()) and aug_p_diff == 0.0 and aug_v_diff == 0.0
    assert saved_arch_matches
    assert outer_eval["rank_order_preserved"]
    assert sel_json.get("reference_release") == "HOLD"

    out_path = EXP_DIR / "fixture_verification_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("ALL FIXTURE ASSERTIONS PASSED.")
    print(f"fixture_verification_report.json sha256={sha256_file(out_path)}")


if __name__ == "__main__":
    main()
