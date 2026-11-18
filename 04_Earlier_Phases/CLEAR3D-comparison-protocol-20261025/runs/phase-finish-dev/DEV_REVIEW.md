# Development review, 2026-10-26

This pass trains on fit patients and scores development patients. Confirmation was not opened. A higher development score does not replace the frozen confirmation result.

Vessel and airway labels are not in the prepared volumes. No vessel mask was created. Hard-negative weights after epoch 2 still come from the training-bank loss.

Wang stage 1, the attention residual U-Net, was not trained. The paper's module is not specified here precisely enough to build it without inventing the block. The stage-2 crop size is specified, and that is the Wang model in this pass.

Augmentation on every model in this pass is training-only: axis flips, rotation by a multiple of 90 degrees, translation of at most 2 voxels with the vacated edge set to zero, and an intensity scale from 0.9 to 1.1 plus a shift of at most 0.05. Sample weights include the fraction of each crop that lies inside the scan.

| model | development F1 | recall | precision | ROC-AUC | PR-AUC | threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wang_aniso | 0.1115 | 0.1849 | 0.0798 | 0.7492 | 0.0978 | 0.5198 |
| zuo_stride | 0.3732 | 0.6189 | 0.2671 | 0.9634 | 0.4231 | 0.7238 |
| clear_aug | 0.1456 | 0.2415 | 0.1042 | 0.7614 | 0.2008 | 0.6304 |
| clear_focal | 0.1274 | 0.2113 | 0.0912 | 0.6219 | 0.1735 | 0.3431 |
| clear_lr | 0.1342 | 0.2226 | 0.0961 | 0.6677 | 0.1695 | 0.7306 |

Stored development F1 from the frozen run, for reference only: Zuo-style 0.4187, Wang-style 0.2298, CLEAR3D 0.1706, ablation 0.1752.

Paired development difference of augmented CLEAR3D minus the stored development score, 200 patient resamples, seed 17.

- clear_aug_minus_zuo: mean -0.2737, interval -0.3319 to -0.2094
- clear_aug_minus_wang: mean -0.0846, interval -0.1224 to -0.0508
- clear_aug_minus_clear: mean -0.0235, interval -0.0466 to 0.0012

Checkpoints store the model, optimizer, epoch, NumPy RNG, and torch RNG. The scheduler entry is null because this recipe has no scheduler.

