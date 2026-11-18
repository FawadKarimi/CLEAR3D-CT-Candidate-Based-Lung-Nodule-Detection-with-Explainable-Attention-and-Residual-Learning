#!/usr/bin/env python3
import hashlib
import json
import numpy as np
import torch

from run_fold0_focal_experiment import (
    CLEAR3DMSFv2,
    MAIN_EPOCHS,
    MAIN_STEPS,
    make_training_rng_streams,
    compute_epoch_lr_and_sw,
    compute_batch_loss,
    sample_aug_params_dev708,
    validate_dev_cohort,
    validate_sec_weights,
    SEL_PATH,
    EXP_DIR,
)

def main():
    sel = json.loads(SEL_PATH.read_text())
    assert sel["reference_release"] == "HOLD"

    scans = [{"patient_id": f"P{i}", "series_uid": f"S{i}", "scan_id": f"ID{i}", "fold": i % 10} for i in range(20)]
    validate_dev_cohort(scans, {"Q1"}, {"QID1"}, {"WS1"}, {"WP1"}, expected_scans=20, expected_patients=20)
    validate_sec_weights(np.ones(12), min_count=10)

    # Verify identical 4-stream RNG draws for Arm A (CE) and Arm C (Focal)
    def draw_streams():
        st = make_training_rng_streams(5000)
        sec_p = validate_sec_weights(0.5 + 0.5 * np.linspace(0.99, 0.44, 12000))
        ip, ir, ih, aug = [], [], [], []
        for _ in range(MAIN_EPOCHS * MAIN_STEPS):
            ip.append(st["rng_pos"].choice(1139, size=10, replace=True))
            ir.append(st["rng_rand"].choice(16000, size=12, replace=False))
            ih.append(st["rng_sec"].choice(12000, size=10, replace=False, p=sec_p))
            aug.append(sample_aug_params_dev708(st["rng_aug"], B=32))
        return np.stack(ip), np.stack(ir), np.stack(ih), aug

    a = draw_streams()
    c = draw_streams()
    assert np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]) and np.array_equal(a[2], c[2])
    for k in ("shifts", "ks", "flip_z", "flip_x", "scale", "shift"):
        assert all(np.array_equal(x[k], y[k]) for x, y in zip(a[3], c[3]))

    # Verify CE vs Focal loss computation and gradients
    logits = torch.randn(32, 2, requires_grad=True)
    targets = torch.tensor([1] * 10 + [0] * 22, dtype=torch.long)
    _, sw_np = compute_epoch_lr_and_sw(0, MAIN_EPOCHS)
    sw = torch.from_numpy(sw_np)
    l_ce = compute_batch_loss(logits, targets, sw, loss_mode="ce")
    l_foc = compute_batch_loss(logits, targets, sw, loss_mode="focal")
    assert torch.isfinite(l_ce) and torch.isfinite(l_foc) and l_foc.item() > 0.0
    l_foc.backward()
    assert logits.grad is not None and torch.all(torch.isfinite(logits.grad))

    print("ALL FOCAL EXPERIMENT FIXTURE ASSERTIONS PASSED.")

if __name__ == "__main__":
    main()
