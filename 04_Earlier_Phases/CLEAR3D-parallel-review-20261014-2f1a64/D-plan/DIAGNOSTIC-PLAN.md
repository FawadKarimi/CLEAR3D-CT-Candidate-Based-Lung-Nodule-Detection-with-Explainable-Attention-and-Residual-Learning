# Proposed fixed-batch, train-only diagnostic

**DRAFT — not executed or adopted.** Scope `clear3d-fixed16-train-only-overfit-v1`. This is a new, separately authorized experiment proposed after the accepted timeout evidence, not a retry/resume or extension of its780-second envelope. Reference release remains HOLD; cap4000 remains NOT_ADOPTED. ResNet10 is not part of this proposal.

## Question and fixed population

Can the existing ResNet18 hybrid change predictions and reduce loss on one deliberately fixed, small training batch while retaining verified finite gradients and actual optimizer updates? This probes memorization capability and instrumentation behavior; it cannot establish generalization, model quality, candidate-policy adequacy or a causal explanation for the earlier nearly constant probabilities.

Choose exactly the first four distinct indices from the positive section (positions0–239) and first twelve distinct indices from the negative section (positions240–959) of the **accepted epoch0 draw plan**, preserving occurrence order. Concatenate positive-first. `fixed_batch.py:40-45` defines this deterministic rule. Bind the resulting16 indices before execution, verify sidecar payloads/labels and unique candidate IDs, and refuse any mismatch. No proposal processing, new label evaluation, patient replacement or favorable-case search is permitted. All rows must belong to the accepted12 training scans, with patient separation from the unchanged validation cohort. Four positives are a diagnostic subset of the same15 unique training positives; this adds no positive coverage.

## Exact proposed policy

| Item | Prespecified value |
|---|---|
| Backbone | Official-byte-bound ResNet18 with unchanged reviewed bridges, deterministic operators, frozen BN statistics/trainable affine terms, pretrained weights and source |
| Inputs | Fixed16 patches,48³,4positive/12negative; crop once and cache bounded CPU tensors; record patch hashes |
| Order/augmentation | Positive-first fixed order each repetition; augmentation disabled, explicitly new diagnostic policy |
| Updates |24 repetitions of the same batch;2rows/microbatch;8microbatches/accumulation boundary;24 actual AdamW updates,0skips |
| Consumption |384 row observations,96positive/288negative;192microbatches; no new cases |
| Objective | Existing class-weighted summed CE with weights[1,3]; each boundary denominator12×1+4×3=24; existing normalization and clipping1 |
| Precision | FP32, disabled scaler, strict determinism; seed17 |
| Learning rates | Constant accepted epoch0 group rates: encoder/bridges/head/fusion1e-4; transformer5e-5; existing decay and parameter groups |
| Model mode | Training mode during updates, including existing dropout; BN statistics remain frozen. Before/after evaluation uses eval mode on these same training rows only |
| Endpoint | Fixed update24; no best-step selection, early success extension, validation, threshold search or accuracy winner |
| Checkpoints | Fresh INITIALIZED then TRAINED states, exact scope/configuration/source/input/adoption bindings; validation_complete=false; no resume interface |

The immutable parent configuration remains an input binding. The separate diagnostic policy explicitly overrides augmentation and the three-epoch LR schedule; it does not pretend those behaviors are unchanged. Numerical optimization implementation, normalization, BN, dropout probabilities, clipping, architecture and initial weights remain unchanged. `fixed_batch.py:13-23` binds the policy/limits; `train_fixed` uses existing `training.parameter_groups`, `optimization.train_epoch`, diagnostics and engineering checkpoint primitives.

## Proposed budget and monitoring

One new300-second total controller envelope, setup at most120seconds, no budget transfer or automatic expansion. Stop at the first failed guard or timeout; do not rerun, resume, lower settings or substitute rows. Existing selected GPU UUID/model/driver and shared-use permission remain required. Resident GPU processes must never be signalled.

Limits:8GiB sampled worker RSS,4GiB allocator,6GiB sampled owned-process VRAM, at least10GiB GPU free before launch; at least32GiB host RAM initially with8GiB remaining headroom;20GiB disk free;1-second sampled observations. The inherited limit dictionary contains a validation timer but no validation phase is authorized or performed. Only the launched single worker may be cleaned up; no multiprocessing/DataLoader workers. PID identity/pidfd and parent-death protections retain their existing limitations.

The cached16 FP32 image patches occupy16×48³×4=7,077,888bytes (~6.75MiB) before copies. Holding individual patches plus stacked batches roughly doubles image storage; valid patches and temporary arrays add overhead. This is an estimate, not total RSS. Model/optimizer/checkpoint copies and attention dominate other allocations.300seconds is a proposed cap, not a predicted completion time. The failed run does not justify linearly extrapolating reliable latency; shared load, diagnostic synchronization and durable I/O remain uncertain.

## Acceptance and interpretation

Technical success requires verified fresh bindings and adopted scope; worker/controller/shell success;24 finite nonzero-gradient boundaries, each denominator24, zero skips and matching AdamW counters; complete fixed consumption; finite before/after training-row losses/probabilities; coherent INITIALIZED→TRAINED checkpoint ancestry; useful owned-worker GPU samples; limits respected and owned worker reaped. Sampled parameter changes and class/family diagnostics must be retained and reconciled, not interpreted as proof of learning. Failures retain partial diagnostics, committed checkpoints and exit/cleanup evidence.

A final training-only weighted loss at most80% of initial loss is a **prespecified descriptive memorization indicator**, not a technical pass condition or scientific quality target. Failure to reach it within24updates is inconclusive; do not tune or extend the run automatically. Report class-specific losses/probabilities and input-dependent logits alongside aggregate change. Evaluate no validation data and use none of its previously disclosed results to select settings.

## Read/write boundaries and execution gates

Preparation of this proposal reads existing code, configurations and accepted metadata/evidence only. **No research pixels or CUDA were accessed for this workstream.** A future execution requires separate explicit adoption of the output-bound DRAFT and authorization for precisely the selected training scans' image.npy/valid.npy pairs. Do not reuse any earlier three-epoch disposition. Record actual user attribution separately from independently authenticated evidence and preserve all six unresolved holds, exposure/pretraining uncertainty and sampled-monitoring limitations.

Fresh preflight must verify new runner/binding, unchanged complete parent source inventory, accepted interpreter/44pins, official checkpoint/provenance, preparation, accepted selection/sidecar/draw hashes, fixed indices, safe nonsymlink output ancestry and absent destination. No existing run directory is reused. Use the packaged proposed shell invocation and its exact binding/output paths; it remains plan-only without the execution flag and a matching ADOPTED disposition hash. Output paths are fixed by the binding, not selected interactively at launch.

Authorized future reads are the pinned metadata/reference snapshot already embodied in accepted selection/labels, checkpoint/provenance and the selected training image/valid arrays. No separate reference files, original proposal streams, masks, raw CT, withheld data or validation arrays. Write only a new output directory containing invocation/adoption bindings, row and cached-patch receipts, diagnostic/boundary logs, before/after same-training-row summaries, monitoring/exits, checkpoint commits/tensors and inventories. Preserve the failed run, every prior checkpoint and accepted artifacts unchanged. Compact review evidence may omit large tensors only with exact retained path/size/hash and a statement that independent tensor comparison requires those original bytes.

Remaining uncertainties: pretrained-input distribution compatibility, research feature separability, adequacy of24updates at fixed LR, diagnostic overhead on CUDA and shared-device timing. Generated CPU tests establish only their generated fixture behavior. This proposal neither resolves those uncertainties in advance nor grants scientific/reference approval.
