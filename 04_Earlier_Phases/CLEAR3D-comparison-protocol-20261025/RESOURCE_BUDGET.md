# Resource budget

Status: DRAFT. This reconciliation used zero GPU hours.

Future device, only after adoption: RTX 2080 Ti UUID GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90, and only when that GPU has at least 10 GiB free. Do not kill other users.

Proposed ceiling, not permission: 4 training trials (Zuo, Wang stage 2, one CLEAR3D candidate, one ablation), 0 confirmation reads, 2 hours per trial, 8 GPU hours total, no automatic extension. A smoke test must be timed before treating 2 hours as enough for a full manifest. Failed trials count. Do not copy the banks.npz files. Custom October trainers are not resume sources because they did not store model, optimizer, scheduler, and RNG together.
