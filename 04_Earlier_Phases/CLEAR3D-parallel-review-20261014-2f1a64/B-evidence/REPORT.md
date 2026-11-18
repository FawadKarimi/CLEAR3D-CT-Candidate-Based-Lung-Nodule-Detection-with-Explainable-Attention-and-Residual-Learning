# B. Existing-evidence analysis

Bound to failure ZIP SHA-256 `446915a67d7faff15b9ba4e00eb0d2ed1a0121125a850136e57d247e23f443fa`. This analysis rereads only packaged scalar logs, configuration and source; no research pixels, weights, checkpoint tensors, CUDA or new training. `analyze.py` uses Python standard library only. `results.json` records input hashes, reproducible aggregates and limitations.

## Confirmed observations

- ResNet18 made **153 actual updates, zero skips**, with 2,448 draws inside completed updates and two further forward rows. Epoch 0 and 1 completed 960 draws/60 updates each; epoch 2 completed 528 draws/33 updates. ResNet10 was not run. Final epoch 2 endpoint was not reached.
- The 7,505 diagnostic events comprise 6,125 activation events, 1,225 class-loss events, 153 optimizer boundaries and two complete-validation events. All recorded activation samples are finite; no sampled activation event is entirely zero. These are sampled/channel summaries, not proof of useful between-input feature separation.
- Gradient norms before clipping are finite and nonzero, 0.602256–2.864671; **98/153 boundaries clip at 1**. Clipping does not zero gradients: after-clipping norms remain nonzero. Encoder, transformer and head/bridge samples change at every completed boundary. This refutes no optimizer advancement and sampled global freeze, but does not establish that every layer has useful data gradients; AdamW decay can change parameters independently.
- Learning rates match the configured short scheduler: CNN/head `1e-4 → 5.05e-5 → 1e-6`; transformer `5e-5 → 2.55e-5 → 1e-6`. Epoch 2 thus has a 50.5-fold CNN/head and 25.5-fold transformer drop from epoch 1. This is an observed configured behavior, not a newly diagnosed scheduler defect. No setting is changed here.
- Mean per-boundary relative sampled parameter displacement falls from epoch 0 to partial epoch 2: encoder `3.68e-5 → 2.49e-7`, transformer `2.89e-5 → 4.04e-7`, head/bridge `4.36e-5 → 3.27e-7`. These are sample ratios, not full parameter norms or unique changed parameter counts.

## Nearly constant predictions: what the data actually show

| Epoch | Negative draws | Positive draws | Mean p(positive), negative | Mean p(positive), positive | Mean CE negative | Mean CE positive |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 720 | 240 | .475724 | .477937 | .649509 | .742859 |
| 1 | 720 | 240 | .481304 | .476664 | .659179 | .743735 |
| 2 partial | 400 | 130 | .488527 | .484806 | .672785 | .726311 |

Training probabilities are **not literally constant**: epoch-0 negative range .3487–.6027 and positive .3139–.5949; standard deviations are about .045. Variation combines distinct inputs, augmentation, dropout and changes over optimization time. It cannot demonstrate input dependence at fixed weights in eval mode. Class means remain close and offer no evidence of learned class separation in this sample.

The accepted loss weights are [1,3], with 720 negative and 240 positive draws per full epoch. Their weighted masses are both 720. The whole-epoch optimal *constant* prediction for that weighted objective is p=.5, with normalized CE log(2)=.693147. Recorded epoch-0 weighted loss .696184 is close to that reference. This explains why a class-uninformative output can sit near .5 under this objective; it does **not** identify why features fail to separate classes. Shuffled group-specific denominators mean the whole-epoch reference is not the stationary point of every minibatch.

Already-disclosed validation is descriptive only: epoch-0 class means .476049/.476070; epoch-1 .486666/.486714, with narrow ranges. No validation result is used here to select learning rate, schedule, model, cap, cases or stopping target.

## Time and monitoring

| Interval from recorded markers | Seconds | Interpretation |
|---|---:|---|
| First resource sample to epoch-0 training | 26.756 | Observed setup portion, not full controller preflight |
| Epoch-0 training marker to validation marker | 184.877 | Training plus post-training checks |
| Epoch-0 validation marker to epoch-1 training | 7.161 | Validation, checkpoint publication and intervening bookkeeping |
| Epoch-1 training marker to validation marker | 313.855 | Same 960 draws, approximately 70% longer than epoch 0 |
| Epoch-1 validation marker to epoch-2 training | 19.277 | Validation, publication and bookkeeping |
| Epoch-2 training marker to last resource sample | 227.269 | Incomplete training interval; excludes final cleanup |

Watchdog time is 780.441s, controller 780.959s, shell 784.871s. The literal watchdog reason is `EXTERNAL_TERMINATION`; the controller deadline predicate and near-780s observation support the accepted fixed-budget interpretation. It is not a numerical-loss failure. Owned worker exit -15, controller/shell exit 1, owned worker absent. Checkpoints retain coherent ancestry through epoch 1; no epoch-2 completion is claimed.

673 monitoring samples include 665 usable owned-GPU observations; peak sampled RSS 2,287,816,704 bytes, VRAM 1,098,907,648 bytes. Sample gaps average 1.160s (maximum 1.544s), so these are sampled peaks. Both resident GPU PIDs remain present with constant allocated-memory observations; allocated memory is not utilization, and it cannot establish whether they caused slowdown. No process was signalled for this analysis.

## Ranked hypotheses and discriminating evidence

1. **Class-uninformative or weakly input-dependent representation under this initialization/input pipeline.** Close class means support investigation; finite activations and updates do not resolve it. Generated paired-input, fixed-state/eval-mode logits and family gradient tests are the proper first discriminator. Research-fixed-batch overfit remains proposed only.
2. **The three-epoch schedule gives very little optimization at useful rates.** Final epoch is at 1e-6 with visibly smaller sampled changes. Confirmed mechanism, unproven cause of flat predictions. A separately reviewed fixed-rate train-only diagnostic can distinguish optimization capacity; do not silently change the accepted run.
3. **Pretrained-input/frozen-BN mismatch.** Requires architecture/input/provenance audit; current scalar summaries neither establish nor exclude mismatch. Do not automatically alter normalization or BN.
4. **Diagnostic synchronization and durability overhead contributes to deadline.** Source `diagnostics.py:15–16,35–37,56–60,65–71` performs device-to-CPU sampling; `optimization.py:50–55` opens/flushes/fsyncs each event. There are 7,505 diagnostic events, plus 153 separate boundary events. These are definite synchronization/I/O mechanisms, but their time cost is unmeasured. The epoch slowdown cannot be assigned to instrumentation alone without a matched measurement; shared-GPU load, storage latency and CPU scheduling remain alternatives.
5. **Clipping limits learning.** It acts on 98 boundaries, but finite nonzero gradients and parameter updates persist. This is lower-priority than graph/input sensitivity and fixed-batch learnability, and provides no justification to change the clipping policy.

## Source locations and reproducibility

- `source/src/lidc3d/scoped_epochs.py:114–150`: actual learning-rate assignment, train, complete validation and checkpoint publication; explains phase-label limitations.
- `source/src/lidc3d/training.py:24–54`: trainable parameter groups and epoch schedule.
- `source/src/lidc3d/optimization.py:76–173`: normalization, finite checks, clipping and actual optimizer update.
- `source/src/lidc3d/diagnostics.py:12–71`: bounded detached diagnostic samples; detach here is instrumentation, not evidence of a detached model path.

Run from the consolidated review root:

```text
python B-evidence/analyze.py --accepted accepted --output B-evidence/results.json
```

Observed local invocation exited 0; 7,505 events/153 boundaries reconciled, 2,450 forward rows, zero recorded skips, consecutive update counters and finite sampled activations verified. No numerical implementation patch is proposed by this evidence analysis. Reference HOLD and cap4000 NOT_ADOPTED remain.
