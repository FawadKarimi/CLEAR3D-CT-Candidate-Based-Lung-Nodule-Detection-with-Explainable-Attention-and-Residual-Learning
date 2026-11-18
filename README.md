# CLEAR3D-CT: Candidate-Based Lung Nodule Detection with Explainable Attention and Residual Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

This repository consolidates the active **CLEAR3D-CT** two-stage 3D lung nodule detection codebase, Phase-1 **LIDC3D** preparation & provenance pipeline, earlier benchmark comparisons, external SPIE-AAPM LUNGx validation scripts, official LUNA16 FROC evaluation tooling, and journal manuscript/audit packages.

## Repository Layout

- **`01_Paper_and_Manuscripts/`** – Journal submission and verification package (`journal_package_20261117/`): double-blind JIIM manuscript & Title Page (`.docx`), CMIG highlights, JOIV camera-ready `.docx`, cover letter, 300-DPI figures (`fig1`–`fig4`), editorial/literature audit reports (`.md`), and programmatic build & check scripts (`build_jiim_package.py`, `build_joiv_docx.py`, `verify_package.py`), alongside `legacy_drafts_and_templates/`.
- **`02_Active_Model_and_Pipeline/`** – Active **CLEAR3D-MSFv2** and baseline detector codebase (`clear3d_msf_model_20261106_002.py`, `clear3d_fineonly_model_20261107_001.py`, `loader.py`, 10-fold CV & holdout evaluation scripts), `external_validation_spie_aapm/` scripts/metrics, `paper-ablation-verify-package-20261107-004/` official LUNA16 FROC evaluator, and lightweight JSON/CSV metric summaries in `runs/`.
- **`03_Phase1_LIDC3D/LIDC3D/`** – Updated Phase-1 `lidc3d` package (`src/lidc3d/`, `scripts/`, `configs/`, `docs/`, `tests/`), synchronized alongside the original root-level Phase-1 snapshot preserved from `main`.
- **`04_Earlier_Phases/`** – Earlier Phase-2 comparison protocols (`CLEAR3D-comparison-protocol-20261025/` for ZuoNet / WangNet / ClearNet baselines), `CLEAR3D-parallel-review-20261014-2f1a64/`, and `CLEAR3D-broader-loader-root/`.

## Data, Checkpoints, and Excluded Artifacts

Large binary assets are kept outside Git via `.gitignore`:
- Raw CT DICOM/MHD volumes (`LIDC-IDRI`, `LUNA16`, `SPIE-AAPM LUNGx`) and prepared `.npy`/`.npz` 3D patch tensors
- Trained PyTorch checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- Multi-megabyte candidate pools, per-scan prediction dumps (`*scores*.csv`, `*pooled_oof_predictions*.csv`, `*.jsonl`), local server symlinks (`06_Shared_Data_Links/`), and raw archive bundles (`05_Historical_Archives_and_Reviews/`, `_archive_v2/`)

All scripts accept or configure dataset/checkpoint roots via CLI arguments or environment/path settings on the local workstation.

---

## Original Root Phase-1 LIDC3D Documentation (`main`)

# Independent "CLEAR3D-CT: Candidate-Based Lung Nodule Detection with Explainable Attention and Residual Learning in a Two-Stage 3D CT Framework" research implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

This is a new project implementing the prospective two-stage 3D thesis protocol.It imports neither repository, their models, labels, predictions, checkpoints, splits nor processed images. The only proposed shared resource is the raw LIDC-IDRI CT/XML archive on the university GPU server, accessed read-only.

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
