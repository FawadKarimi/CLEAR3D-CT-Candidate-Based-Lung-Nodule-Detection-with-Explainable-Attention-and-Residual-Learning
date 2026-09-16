# Independent LIDC3D research implementation

This is a new project implementing the prospective two-stage 3D thesis protocol. **It is separate from ConsenDxFormer and LungDxFormer.** It imports neither repository, their models, labels, predictions, checkpoints, splits nor processed images. The only proposed shared resource is the raw LIDC-IDRI CT/XML archive on the university GPU server, accessed read-only.

**Status: research software, synthetic verification only. `PENDING_VERIFIED_EXPERIMENT`.** No patient cohort has been processed here, no model has been trained on LIDC-IDRI, and no performance or clinical utility has been established.

## Start here

Read `docs/Experiment_Design.md`, `docs/Implementation_Decisions.md` and `docs/Artifact_Contract.md`. `validation/` contains the actual software check reports generated with this package. Passing those checks does not validate a clinical reference standard, annotation-parser coverage or detector performance.

The first server action is a read-only environment check. We will execute server steps individually and inspect each output before proceeding. No source data should be uploaded to this conversation.

The package requires Python 3.10 or newer and a compatible PyTorch installation. Use an independent environment and output directory. Do not upgrade the environment used by either existing project. The dependency ranges in `pyproject.toml` are installation requirements, not a claim that every allowed combination has been tested. `validation/environment.json` records the specific local verification environment; the server environment must be recorded separately.

## Modules

| Module | Purpose |
| --- | --- |
| `ingest.py` | Read-only header inventory; strict single-frame CT reconstruction, source hashes |
| `geometry.py` | Native z-y-x ↔ LPS affines, canonical 1 mm resampling, windowing and patches |
| `references.py` | XML contours/points, reader consolidation, majority masks, ignored regions |
| `splits.py` | Independent deterministic patient partition and fallback record |
| `candidates.py` | Classical image-derived lung mask and streaming multiscale LoG peaks |
| `pipeline.py` | Draft cohort, candidate pool, validation threshold selection and artifact freeze |
| `models.py` | Width-16 3D residual encoder, independent scale Transformers, fusion controls |
| `data.py`, `training.py` | Reproducible patch augmentation, accumulated weighted loss, validation CPM selection, checkpoint resume |
| `evaluation.py` | Custom matching/FROC/CPM, candidate metrics, clustered bootstrap, optional permutation/Holm |
| `xai.py`, `explain_runner.py` | Grad-CAM, IG, attention, map controls, perturbations, stability and randomization |
| `provenance.py`, `cli.py` | Hashes, exclusive artifact creation, configuration correspondence and command interfaces |

## Execution order

1. Inspect the server's Python/PyTorch/CUDA environment, raw CT/XML paths and free storage.
2. Create an isolated project/environment. Inspect CLI help after installation.
3. Run `environment` and `preflight`; archive the outputs. Preflight uses synthetic tensors.
4. Run `inventory`; review source availability and unsupported files.
5. Run `cohort`; this builds **draft** references and partitions. Review XML alignment, source exclusions, association ambiguity and cross-project test exposure before release.
6. Run `prepare` on the development cohort; inspect mask and proposal failures. Run `freeze` only with a review record approving that exact artifact.
7. Generate the ablation configs from the resolved reference config. Run the selected development experiments using `train` and new output directories.
8. Finalize the complete development decisions and test-release record. Test preparation and inference are separate actions.
9. Run `predict`, then `evaluate` and `bootstrap`. The predictor receives no annotation data; the runner joins saved predictions to references afterward for evaluation.
10. Run `explain` with the saved inference provenance. Include parameter randomization for the planned sanity-check analysis. This workload is substantial and needs its own measured preflight.

These names identify CLI subcommands, not a request to execute the entire sequence now. Each command's required arguments are available through its `--help` option.

## Important boundaries

- The custom evaluator is **not** the official LUNA16 evaluator. Duplicate-positive candidates count as detection false positives after one-to-one matching.
- The software's release files are workflow safeguards, not a security boundary. Protect withheld test files through actual directory permissions/role separation where available, and record prior access honestly.
- XML support is deliberately strict. Missing reader IDs, ambiguous associations and unsupported structures require audit; do not suppress errors or silently exclude large patient subsets to obtain a runnable cohort.
- A checkpoint contains serialized Python/PyTorch objects. Load only checkpoints produced by this project or otherwise trusted.
- No pretraining, malignancy diagnosis, end-to-end clinical validation or state-of-the-art claim is made.
- Learned proposal comparators, external published-method reproductions and mask-supervised alignment require the additional implementation/evidence gates identified in the design. The included LoG/CNN/hybrid experiments support a narrower internal comparison.

## Source changes

New outputs are created exclusively; a repeated output path fails instead of overwriting a completed run. `best.pt` and `last.pt` are the intentional atomic updates within an active run. Resume writes to a new output directory and resumes from a completed epoch, not from a partially executed epoch.

## Technical references

Implementation interfaces were checked against the [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html), [SciPy Gaussian Laplacian documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_laplace.html), and [PyTorch multi-head attention documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html). These are engineering references, not additions to the thesis bibliography or sources of experimental results.
