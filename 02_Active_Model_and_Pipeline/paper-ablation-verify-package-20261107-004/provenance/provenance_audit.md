# CLEAR3D-CT Development Ablation Provenance & Audit Report

## 1. Checkpoint-to-Prediction Cryptographic Binding
Every scoring run in this ablation package was executed under single-process isolation via `user_watchdog` and enforced strict cryptographic pre-flight checks:
- **Fold 0 Baseline**:
  - Model Checkpoint: `runs/clear3d-msf-ensemble-20261106-004/checkpoints/fold-0.pt`
  - Checkpoint SHA-256: `739d1d7883d66922f568be4cdc66f491bdabe8eb9e1c2c22ecb8c5bf7b8f5494`
  - Verification: Checked at line 178 of `score_clear3d_msf_ablation_fold0_20261107.py` via `hash_file(cp) == expected_sha`.
  - Output Predictions: `predictions/fold0_baseline_scores.csv` (SHA-256: `39c9bded3d5a942937b97dcc575ad6e90432e5e7b2e32525b5908e918768a3fd`).
- **Control B No-Mining**:
  - Model Checkpoint: `runs/clear3d-msf-ablation-nomining-dev-20261107-001/checkpoints/fold-0.pt`
  - Checkpoint SHA-256: `941aaa0f357cf339e937d4b728a503fc60364aab543e419dfa487370a260d7ba`
  - Verification: Checked at line 178 of `score_clear3d_nomining_ablation_20261107.py`.
  - Output Predictions: `predictions/control_b_nomining_scores.csv` (SHA-256: `bb90835d9905fe77a03e11382681cd26ad160ec5c426ca34a9a247e303b9ec24`).
- **Control A Fine-Only**:
  - Model Checkpoint: `runs/clear3d-msf-ablation-fineonly-dev-20261107-001/checkpoints/fold-0.pt`
  - Checkpoint SHA-256: `0907483e7e1aac661cb20063eb08cb8c543b11369b74bbcdb741d153a5f241af`
  - Verification: Checked at line 178 of `score_clear3d_fineonly_ablation_20261107.py`.
  - Output Predictions: `predictions/control_a_fineonly_scores.csv` (SHA-256: `0afb983ecd66a07de2fde28e428c5f7243643a904cf104657b9c719ee2e5f9ba`).

## 2. Preserved Uncertainty & Checkpoint Metadata Limitations
- Standard PyTorch `.pt` binary archives serialize Python dictionaries storing tensor parameters and scalar floats (`model`, `scale_oof`, `bias_oof`, `nat_cut_oof`).
- PyTorch does NOT cryptographically sign or record the executing script's SHA-256 digest or git commit hash inside the binary file format.
- Therefore, while the scoring worker was strictly hash-bound to the checkpoint file, the binding of the checkpoint file to its training script invocation is established through surrounding execution artifacts (`invocation.json`, `worker.log`, `summary.json`, and timestamp ordering), rather than internal binary introspection.

## 3. External Planned Comparator Status (MedicalNet)
- MedicalNet ResNet-10/18 weights are verified and present on disk, passing all 823 CPU unit tests.
- MedicalNet was **NEVER** trained or evaluated on GPU in the CLEAR3D v2 pipeline.
- It remains an external planned comparison only and must NOT be reported as a completed empirical comparator.

## 4. Sensitivity Vector Audit & Retraction of Hallucinated Vector B
- **Vector A** (`[0.760417, 0.859375, 0.916667, 0.937500, 0.963542, 0.973958, 0.979167]`, CPM = `0.912946`): Authentic, verified FROC curve from Run 004 5-fold ensemble on 122,689 development candidates (`runs/clear3d-msf-official-dev-eval-20261106-004/froc999999/froc_cap999999.txt`).
- **Vector B** (`[0.791667, 0.864583, 0.906250, 0.927083, 0.963542, 0.968750, 0.968750]`, CPM = `0.912946`): Generated as an unverified conversational hallucination during turn 2863. No script or log on the server contains Vector B. It is permanently retracted.
