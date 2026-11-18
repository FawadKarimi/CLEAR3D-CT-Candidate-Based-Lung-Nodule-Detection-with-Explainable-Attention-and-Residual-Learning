# Model-path audit

Scope: static inspection of the source/configuration snapshot in the accepted failure package SHA-256 `446915a67d7faff15b9ba4e00eb0d2ed1a0121125a850136e57d247e23f443fa`. No research arrays, weights, CUDA, training or source modifications were performed by this workstream. All source references below are relative to `accepted/source/src/lidc3d/`; configuration references are relative to `accepted/`.

## Conclusion

No new confirmed feature-detachment, disconnected-optimizer, normalization-denominator or constant-output implementation defect was found in the inspected active path. This is a static conclusion, not proof of numerical health on actual inputs. Several mechanisms can produce nearly constant probabilities without a broken graph. They need controlled training-only diagnosis rather than validation-driven settings changes.

## Exact active path

| Stage | Operation and shape for B×1×48×48×48 | Source |
|---|---|---|
| Input | Read-only memory-mapped crop; copy; deterministic per-seed/epoch/candidate augmentation; invalid voxels set to zero; FP32. Upstream HU clipping maps [-1000,400] to [0,1]. | `data.py:12-14,32-50`; `geometry.py:59-63`; configuration:143-151 |
| MedicalNet stem | 7-cube convolution, stride2, 64 channels →24³; frozen-statistics BN/ReLU; deterministic 3-cube max-pool stride2 →12³. | `medicalnet_vendor.py:126-143`; `medicalnet_adapter.py:21-46,95-98` |
| Encoder | layer1 64×12³; layer2 128×6³; layer3 256×6³ dilation2; layer4 512×6³ dilation4. Depth18 has 2 blocks/stage; depth10 has1. | `medicalnet_vendor.py:137-143,217-227` |
| Bridges | Learned bias-free 1×1×1 convolutions 64→32,128→64,512→128. Layer3 is used upstream of layer4, not dropped from the graph. Last-scale nonoverlap mean2 produces128×3³. Outputs32×12³,64×6³,128×3³. | `medicalnet_adapter.py:82-102` |
| Transformers | Each scale projects to128; tokens1728,216,27; fixed 3D sinusoidal position;2 pre-LN attention/FF blocks,4heads,FF256; residuals; final LN; output projection back to32/64/128. | `models.py:90-152`; configuration:125-142 |
| Fusion | Concatenate original and transformed feature at each scale;1×1 projection to128/GN8/ReLU. Start coarse, differentiable trilinear resize to6³ then12³; concatenate local128 and upsampled128;1×1/GN8/ReLU at each merge. | `models.py:163-181,194-214` |
| Pool and head | Spatial mean over12³,Linear128→128/ReLU/dropout0.3/Linear128→2. Softmax used for diagnostics/prediction; weighted CE consumes raw logits. | `models.py:41-43,183-184,215`; `optimization.py:190-198` |

The 1728-token scale has 1728²=2,985,984 query/key positions per head per attention layer, versus46,656 and729 at the other scales. This is a structural cost indicator, not measured allocation or throughput; kernel implementation and strict determinism affect actual cost.

## Gradient and normalization findings

1. **Active shortcut corrected.** Vendored shortcut A uses `.data` at `medicalnet_vendor.py:26-37`; the actual depth18 constructor replaces all three channel/stride transitions at `medicalnet_adapter.py:57-62`. Replacement slices the input and concatenates zeros without detaching (`48-51`). Merely finding `.data` in vendored source is not evidence it executed. Depth10 uses learned shortcut B.
2. **No active feature detach found.** Encoder, bridges, attention, fusion, spatial mean and head retain autograd operations. `diagnostics.py:10-16,43-50` detaches observational samples only; hooks do not return replacement outputs. Prediction `no_grad` is confined to validation (`training.py:61-75`).
3. **Frozen BN is statistics-only.** `MedicalEncoder.train` first applies ordinary train mode, then sets backbone BN modules to eval (`medicalnet_adapter.py:90-94`). BN weight/bias remain trainable. This preserves running statistics and batch counter while affine terms can change. It does not mean the encoder is frozen. The existing policy explicitly binds this distinction (`medicalnet_contract.py:9`).
4. **All trainable parameters enter optimizer once.** `training.py:24-43` verifies exhaustive/disjoint parameter groups. Encoder bridges are in the CNN family, transformers in transformer family, fusion/head in head family. GroupNorm/LayerNorm parameters and biases receive zero decay; BN weights receive configured1e-4 decay because BN is absent from `norm_ids`. This is a confirmed policy detail, not a demonstrated failure and not authorization to change decay.
5. **Loss normalization is mathematically coherent.** Summed class-weighted CE is backpropagated per microbatch; gradients are divided once by actual summed class weights at the accumulation boundary before global-norm clipping (`optimization.py:118-138,190-219`). There is no additional divide by microbatch/accumulation count. AdamW advances only after finite-gradient checks. FP32 uses disabled scaler, not AMP recovery.
6. **Learning-rate schedule is aggressive by design, not observed as zero.** For three epochs/warmup1/minLR1e-6, `training.py:46-51` gives encoder/head `[1e-4,5.05e-5,1e-6]` and transformer `[5e-5,2.55e-5,1e-6]`; applied before training each epoch (`scoped_epochs.py:111-124`). Whether that is adequate learning exposure is unresolved; no search or change is proposed from validation evidence.

## Interpreting feature variation and flat outputs

These are hypotheses, not causal conclusions:

- **Balanced effective objective permits an uninformative baseline.** The fixed draw plan is240 positive/720negative with weights3/1. Aggregate positive and negative weight mass are720 each. For a single constant positive probability, weighted CE is minimized at0.5, with normalized loss ln2≈0.693147. This explains what a constant predictor would look like; it does not establish why the network approaches it. Per-boundary random class compositions and dropout still cause updates around that aggregate baseline.
- **Normalization and pooling can attenuate distinctions.** GroupNorm and token LayerNorm remove particular scale/offset directions, then global spatial averaging discards spatial arrangement. They do not mathematically force identical logits: residual/local feature paths and learned channel mixing remain. Input-dependent logits and per-family gradient tests should challenge this hypothesis with generated patterns.
- **Pretrained input distribution may differ.** Current input is fixed48³, clipped linear HU scaling, invalid zeros and limited augmentation. The accepted package binds official provenance but does not include the original preprocessing implementation or checkpoint-training data distribution. Therefore this audit cannot assert that the specific original checkpoint was trained with the same scaling, spacing, crop size or invalid-value behavior, nor that a public demonstration's preprocessing defines pretraining. Frozen BN statistics can amplify distribution mismatch; that remains unmeasured here. Do not normalize differently merely to obtain a pass.
- **Sparse positive diversity and same-ID augmentation.** Only15 unique positive candidates supply240 positive draws each epoch. Candidate augmentation is keyed by seed/epoch/ID, so repeated copies of a given candidate in one epoch receive the same augmentation (`data.py:39-48`). The repeats add weight, not240 independent positive observations. This is deliberate sampling semantics, not hidden extra coverage.

## Instrumentation and budget

The code confirms potentially expensive observational work: each encoder/transformer/head hook samples tensors and converts them through `.double().cpu().tolist()` (`diagnostics.py:10-16,30-39`); every event calls durable JSON append/flush/fsync (`26-29`; `optimization.py:50-55`). Per-boundary parameter samples and gradient finiteness checks also synchronize device data. This establishes synchronization/I/O opportunities, **not the fraction of the780-second budget they consumed**. Current logs have no instrumented-vs-uninstrumented matched timing decomposition. It would be incorrect to attribute all timeout cost to diagnostics or to promise completion after removing them.

Activation summaries are bounded samples of flattened tensor elements, not full norms, variances, per-input separability tests or class-conditional representations (`diagnostics.py:18-20,35-38`). Encoder summaries contain three scales; transformer hooks cover returned feature tensors; the head summary includes both class logits. A high activation range can coexist with almost identical class probability. Sampled changed parameter values establish some parameter movement but do not establish useful feature learning; AdamW decay itself can change weights.

## Metadata caveat and recommendations

`medicalnet_adapter.py:111` still embeds the literal phrase `original-name mirror, official equivalence unresolved` in `model.config`, whereas the actual runner separately verifies pinned official provenance (`medicalnet_finetuning.py:18-31`). Treat the literal as stale adapter-era descriptive metadata, not a fresh adjudication of the reviewed official-byte comparison. Do not alter historical checkpoint identities to clean up wording; report both channels explicitly. The compact failure package does not itself contain the official archive/member bytes for a new independent comparison.

Recommended tests are generated input-dependent logits/full-path gradients, frozen-statistics versus trainable-affine behavior, and controlled weighted-CE normalization checks. No numerical code patch is justified solely by this static audit. Proposed research diagnostic must remain fixed train-only and separately authorized, with no validation tuning, no recovery/resume of the failed run, and no cap4000 adoption. Reference HOLD remains.
