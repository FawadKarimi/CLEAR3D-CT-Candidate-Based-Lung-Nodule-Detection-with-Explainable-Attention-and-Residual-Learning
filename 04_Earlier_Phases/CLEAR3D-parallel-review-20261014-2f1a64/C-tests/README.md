# Focused generated CPU verification

The five standalone unittest probes import the exact accepted source. They make no production changes and read no checkpoint or research file. The production-shape fixture constructs the actual corrected ResNet18 backbone and MedicalEncoder/bridges/transformer/fusion/classifier with randomly initialized weights. `load_encoder` is mocked only to supply this generated fixture; this is explicitly not a pretrained-behavior or checkpoint-provenance test.

Coverage:

1. Balanced weighted constant predictor: 240 positives at weight 3 and 720 negatives at weight 1 yield a stationary constant probability of 0.5 and loss ln(2). This algebra does not establish that the trained model is constant or explain why it failed to separate inputs.
2. Production 48-cube path: different generated inputs produce different features/logits; weighted CE propagates finite, nonzero gradients to input, backbone, bridges, transformer, fusion, classifier. Frozen BN running state stays fixed and affine parameters remain trainable.
3. Transformer channel normalization plus spatial pooling does not structurally force a constant learned output for the controlled fixture.
4. Corrected shortcut preserves the precise expected subsampled input gradient.
5. Bounded diagnostic hooks and microbatch/capture observers preserve exact logits, gradients, AdamW result and RNG state in a small dropout-containing fixture. This does not measure CUDA synchronization or filesystem overhead, nor replace previously accepted full-path instrumentation tests.

Local supplementary run: Python 3.14.6, Torch 2.13.0+cpu, one CPU thread, strict deterministic algorithms, CUDA_VISIBLE_DEVICES empty. Four original tests PASS in 4.282 s; the added balanced-objective test PASS in 0.004 s. Logs and shell exits are retained. The only subsequent change in an already-passed test is detaching its scalar assertion to remove a warning. The accepted pinned Python/PyTorch environment must run the final five-test file before these are treated as accepted-environment regressions.

Exact proposed accepted-environment command (paths supplied by coordinator):

```sh
env -u PYTHONHOME -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$ACCEPTED_PYTHON" -B "$REVIEW/C-tests/test_model_path.py" --source "$ACCEPTED_SOURCE/src"
```

No tolerance tuning, production numerical change, research sample selection, CUDA execution or model-quality inference is made. ResNet10 is not launched; the targeted new production path probe uses ResNet18, which exhausted the previous run budget. Prior both-backbone qualification is reused.

Auxiliary edit evidence: two apply_patch attempts failed context validation (hunks were ordered backwards); neither changed a file. Reordered hunks applied successfully. No test or compute attempt failed and no failed compute was retried.

## Proposed fixed-batch runner tests

`test_fixed_batch.py` adds four generated tests: deterministic first-distinct selection and refusal; separate adoption and six-hold enforcement; subprocess PLAN_ONLY/no Torch import; a Tiny model exercising 24 actual boundaries, 24 denominator, 192 microbatches, epoch-zero constant learning rates, unchanged-validation boundary and committed INITIALIZED/TRAINED checkpoint ancestry.

Local supplementary run: 3 structural tests passed; the numerical test could not import accepted training.py because local Python lacks scikit-image. This dependency failure is retained in local-fixed-batch.log and local-fixed-batch.exit. No installation, compute retry or mocking of the missing dependency was performed. The accepted server environment must run this test with `--checkpoints enabled`; local `--checkpoints disabled` was explicitly chosen because Windows directory-fsync semantics differ, but execution stopped before model construction anyway.

Exact accepted-environment addition:
`$ACCEPTED_PYTHON -B $REVIEW/C-tests/test_fixed_batch.py --source $ACCEPTED_SOURCE/src --runner $REVIEW/D-plan/fixed_batch.py --checkpoints enabled`
