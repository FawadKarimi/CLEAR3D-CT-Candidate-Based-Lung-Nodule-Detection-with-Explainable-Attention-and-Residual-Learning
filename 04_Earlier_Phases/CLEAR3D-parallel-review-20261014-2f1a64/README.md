# Consolidated CLEAR3D diagnosis and proposed diagnostic

Accepted failure input: `446915a67d7faff15b9ba4e00eb0d2ed1a0121125a850136e57d247e23f443fa`.
No research pixels, GPU execution, retries, ResNet10 launch or production changes occurred in this review. Reference HOLD and cap4000 NOT_ADOPTED remain unchanged. Three parallel agents audited the model, analyzed existing scalar evidence, and wrote generated tests. The coordinating agent reconciled their findings and prepared/reviewed the separate diagnostic entry.

## Outcome

No confirmed active model-path implementation defect justifies changing numerical behavior. The full-hybrid generated CPU fixture has input-sensitive logits and nonzero gradients through encoder, bridges, transformer, fusion and classifier. This uses random fixture weights, not the actual pretrained checkpoint or research pixels. It does not establish useful learned features on CT.

The accepted failed run made153 updates with0skips;98 boundaries clipped at1. All sampled parameter families changed. Predictions varied, but positive and negative training means stayed close. The weighted-balanced objective admits a constant p=.5 stationary baseline. The short configured schedule reduced CNN/head LR from1e-4 to1e-6 by epoch2. These facts are not proof of a broken scheduler or a cause of failed learning.

The780-second attempt exhausted its envelope before the final endpoint. Epoch1 training took313.855s versus184.877s in epoch0. Per-event durable logging/CPU synchronization and shared-GPU variation are plausible contributors; existing evidence cannot apportion their costs. No timing claim is extrapolated to full-cohort training.

## Ranked hypotheses, not causal findings

1. Weak class-discriminative features on this tiny repeated population, compatible with the balanced constant-predictor baseline. Fixed-weight research-input sensitivity is unmeasured; the proposed fixed-batch memorization test addresses a narrower question.
2. Short schedule and small late updates limit adaptation. This is configured behavior; no unsupported learning-rate search or production change is proposed. The separate diagnostic holds the reviewed epoch0 group rates constant.
3. Frozen-BN/pretraining-input domain mismatch or normalization/pooling attenuation. No defect confirmed; actual official preprocessing assumptions are not fully established by this failure package. Demonstration transforms cannot be elevated to mandatory pretraining facts.
4. Instrumentation synchronization/fsync plus shared-host/GPU variability contributed to runtime. Plausible, unisolated. Diagnostics remain enabled; no repeat is authorized to measure overhead now.
5. Global detach/frozen optimizer path is disfavored by source trace, actual update evidence and generated gradients. CPU fixtures cannot exclude data-dependent or CUDA-only issues.

See A-model/MODEL-PATH-AUDIT.md for exact source locations and B-evidence/REPORT.md plus results.json for reproducible, hash-bound aggregates. Validation findings remain already-disclosed descriptive evidence, never a setting-selection signal.

## Verification

All133 parent source files and all44 installed pins matched before/after accepted-environment CPU tests. Nine targeted tests passed, exits0, in5.932s and3.575s process wall time; cumulative child peak RSS1,152,456KiB (about1.10GiB). CUDA_VISIBLE_DEVICES was empty. Complete logs, executable identity and commands are in cpu-results/. No unchanged full suite or GPU qualification was repeated.

Preserved local supplementary results include a missing-scikit-image import failure before numerical fixture execution. No installation was performed. The accepted environment already supplied the pinned dependency. CPU fixtures do not qualify future CUDA performance or resource compliance. Generated Tiny model checks cover the new24-update/checkpoint seam; full hybrid checks cover forward/backward and observer equivalence separately.

## Proposed next action

Review D-plan/DIAGNOSTIC-PLAN.md, fixed16-binding.json, proposed-command.sh and DRAFT-disposition.json. A separate explicit adoption/execution authorization is still required. Default runner invocation is PLAN_ONLY and imports no Torch. No adopted record exists here.

The exact proposed experiment is ResNet18 only,16 mechanically selected training records (4positive/12negative),24 FP32 updates of the same batch, microbatch2/accumulation8, weighted denominator24, constant epoch0 group LRs and augmentation disabled. The new policy is explicit, not a repair/retry of the prior experiment. It has a separate300-second sampled worker envelope, no validation, no extra cases, no automatic retry and no hyperparameter search. Loss reduction is descriptive; technical completion is not learning/generalization acceptance.

## Artifacts and limitations

exact-additive.patch adds only the new diagnostic entry and two test files; no parent source changes. Patch replay is checked byte-for-byte. The binding pins all parent sources via the unchanged parent binding, configuration, environment, weights/provenance, accepted sidecar/draw/selection/preparation identities, exact16indices, resource limits and output path. Index/manifest identities are retained as provenance; the fixed-row worker does not need to reread the full index or original proposal streams.

Large research arrays, weights and checkpoint tensor bytes are not included or read. Original failure evidence's omitted-checkpoints.json preserves their inventory. This compact package cannot independently reproduce tensor-level comparisons without those original bytes and later authorization. The original ZIP and historical artifacts remain unchanged.

Deploy package files to their pinned isolated root with no overwrites unless existing bytes are identical; do not copy into the live checkout. Create only the runs parent after nonsymlink parent checks. The proposed shell rejects existing output/log/status paths and returns the controller status; acceptance retains actual owned-worker exit and cleanup. Fresh resource/identity gates still apply immediately before a separately authorized execution. Cooperative samples are not hard containment; overshoot, rapid allocation and controller-death risks remain. Never signal resident GPU processes.

Times in inherited evidence retain their original source; this review's host wall measurements are monotonic elapsed durations. The directory date is a naming label, not a substitute for clock evidence.
