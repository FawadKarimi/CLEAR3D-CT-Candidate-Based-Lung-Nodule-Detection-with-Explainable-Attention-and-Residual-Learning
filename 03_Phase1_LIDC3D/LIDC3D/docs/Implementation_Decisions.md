# Implementation decisions and evidence limits

## Project identity correction

The author explicitly confirmed that no separate 3D implementation existed. ConsenDxFormer and LungDxFormer are different projects. The previous request to reconcile this thesis with their source was therefore misplaced. This package starts the 3D implementation independently. The previous source report is superseded on the question of project identity.

The sentence in revised thesis §3.1 stating that the author reports an existing pipeline must be replaced at the next manuscript update with:

> The original draft specified a prospective two-stage 3D framework without a corresponding verified implementation. An independent implementation has subsequently been developed for software verification. Real-data integration, reference-standard auditing and confirmatory experiments remain pending.

This correction concerns implementation history. It provides no experimental evidence and does not authorize empirical Results or Discussion claims.

## New explicit choices

All values below are prospective engineering decisions. They have not been optimized using patient data.

| Decision | Choice in this package | Weakness addressed / traceability |
| --- | --- | --- |
| CT scope | Single-frame MONOCHROME2 CT only; no InstanceNumber fallback; unsupported multiframe/tilted or irregular geometry fails with reason | R05, R13, R14: guessing unsupported geometry would compromise labels |
| Geometry tolerance | Orientation absolute tolerance 1e-4; in-plane spacing 1e-6 mm; position drift 0.1 mm; spacing absolute 0.1 mm plus relative 1% | R13: make previously unresolved tolerances inspectable; development audit may require versioned revision |
| Output axes | Canonical LPS-aligned 1 mm grid enclosing native voxel centers; array order z-y-x | R13: make the x-axis reflection anatomically interpretable |
| Interpolation validity | HU linear; invalid-source support must be below 1e-6 before an output is valid; categorical masks nearest neighbor | R05/R13: padded/invalid values cannot become apparent anatomy |
| Contours | Integer pixel-center rasterization with inclusion/exclusion regions; SOP UID plus physical-z linkage; reader IDs required | R05: parser behavior and holes explicit; real XML schema coverage unverified |
| Lung mask | Axial air threshold −400 HU; remove air components touching image edges; 26-connected 3D components; retain up to two components ≥10,000 mm³; fill axial holes; 5 mm dilation | R10–R12: concrete, auditable initial algorithm; border nodules/abdominal gas/abnormal lungs remain risks |
| LoG threshold search | [0.005, 0.01, 0.02, 0.04, 0.08] on normalized image responses; validation recall then candidate count then listed order | R12: finite search specified; selected value remains pending |
| Filtering | SciPy reflect boundary, Gaussian truncation at 4 standard deviations; no prefilter masking | R12: define edge behavior; stream scale volumes to reduce memory |
| Crop | Round voxel coordinate half-up; even crop starts at center−size/2; candidate at voxel size/2; padding zero with validity mask | R13/R20: deterministic map-to-scan transform |
| Residual encoder | 3³ stride-2 stem, no max-pool; four stages of two basic blocks; widths 16,32,64,128; stage strides 1,2,2,2 | R14: actual topology specified rather than inferred from “ResNet-18” |
| Normalization | GroupNorm, gcd(8, channel count) groups; used consistently in CNN controls and hybrid | R14/R25: avoid micro-batch-dependent BatchNorm statistics |
| Transformer | Independent parameters across scales; pre-norm; GELU; final LayerNorm; 0.1 attention and feed-forward dropout | R14: make capacity and tensor behavior explicit |
| Positions | Fixed separable sinusoidal z/y/x coordinates on [−1,1], concatenated and truncated to 128 channels | R14: three-dimensional encoding is explicit; ablation switches it off |
| Hierarchical fusion | Project CNN+Transformer pairs to 128 channels; interpolate coarse→medium→fine with trilinear interpolation, align_corners=False; concatenate and learn 1³ projections | R15: distinguish learned feature operations from heuristic XAI weights |
| Capacity controls | CNN width 16; width 32 as the initial approximate MAC control; residual parameter and compute differences must be reported | R14/R27: compute matching is approximate, not a claim of equal capacity or latency |
| Weight decay | Exclude all biases and GroupNorm/LayerNorm parameters; other trainable parameters receive 1e-4 | R25: optimizer configuration fully enumerated |
| Accumulation | Sum weighted losses; divide accumulated gradients by actual summed class weights, including incomplete groups | R25: reproduce weighted cross-entropy normalization, not an average of unequal microbatch means |
| Model selection | Strict maximum validation CPM selects checkpoint, ties earlier; patience separately uses minimum improvement 0.001 | R24: distinguish checkpoint ranking from stopping tolerance |
| Matcher tie handling | Sort IDs; SciPy linear_sum_assignment with cardinality-dominating dummy cost; version-locked solver resolves remaining equal-cost ties | R08/R09: deterministic assignment; does not claim an extra unimplemented lexicographic optimization |
| Cluster bootstrap | Cache deterministic scan match counts, then apply patient multiplicities | R22: algebraically equal to replicated scan IDs; tested against explicit replication |
| XAI selection | Maximum coverage assignment, then minimum summed candidate score ranks; one candidate cannot represent two lesions | R20: resolve conflicts in “highest candidate per lesion” explicitly |
| Stability | Five seeded Gaussian-noise repeats and one left-right reflection per case/method | R19: fixed replication count rather than retrospective example selection |

For 48³ input the selected CNN grids are 12³, 6³ and 3³, with 1,728, 216 and 27 tokens respectively. The measured architecture signature is saved under `validation/architecture_signatures.json`. Parameter/MAC calculations concern software structure; they are not detector performance or GPU timing. The theoretical CNN-path receptive fields are 47, 103 and 215 input voxels; padding and attention mean these are not lesion-resolution guarantees.

## Known restrictions requiring real-data work

1. The XML adapter has controlled-fixture verification, not demonstrated coverage of the server archive. Audit XML variants, reader identifiers, contours, point marks and exclusions against native CT images before freezing a cohort.
2. Competing reader associations block release. This version emits the ambiguity register but does not provide an adjudication GUI or automatically resolve ambiguous clinical references. A reviewed adapter amendment/rebuild is required where ambiguity is present.
3. Classical masks and LoG scales may miss difficult nodules. Measure proposal recall before expensive classifier training; classifier changes cannot recover missing proposals.
4. Pure Transformer, learned proposal, published multi-view FPR and mask-supervised alignment arms are extension gates. Their outcomes cannot be implied from the included core runs. Alignment is explicitly rejected by the reference runner rather than silently ignored.
5. Optional temperature scaling and calibrated operating-point reporting are not enabled in this version. Primary scores are raw softmax rankings and are not described as calibrated clinical probabilities.
6. Validation-fitted XAI mixing is available as a function; a complete replay must save its validation case IDs, full search trace and selected weights and then apply exactly those weights to the saved test maps. No automatic test fitting is permitted.
7. Patient-clustered detector intervals are implemented for CPM and paired differences. Candidate metrics are exported as point estimates. A manuscript must not attach independent-candidate confidence intervals to them; add clustered computation if intervals for those secondary endpoints are needed.
8. GPU feasibility, acquisition robustness, real-data performance, speed and reproducibility across devices remain `[NEEDS VERIFIED DATA]`. A CPU synthetic test pass cannot settle them.

These limits are implementation/evidence gates, not hidden assumptions or experimental findings.
