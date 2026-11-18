# -*- coding: utf-8 -*-
"""Canonical verified manuscript content shared across JIIM (Double-Blind) and JOIV (Template-Native) builds."""

TITLE_CASE = (
    "CLEAR3D-CT: Multi-Scale Volumetric Candidate Classification with Hard-Negative Mining "
    "and Nested Platt Calibration for Pulmonary Nodule False-Positive Reduction"
)
RUNNING_TITLE = "CLEAR3D-CT Multi-Scale Nodule False-Positive Reduction"

AUTHORS = [
    {
        "name": "Mohammad Fawad Karimi",
        "aff_idx": "a",
        "dept": "Department of Informatics, Faculty of Mathematics and Natural Sciences",
        "inst": "Universitas Syiah Kuala, Banda Aceh 23111, Indonesia",
        "email": "mfkarimi@mhs.usk.ac.id",
        "corr": True,
    },
    {
        "name": "Kahlil Muchtar",
        "aff_idx": "b,*",
        "dept": "Department of Electrical and Computer Engineering",
        "inst": "Universitas Syiah Kuala, Banda Aceh 23111, Indonesia",
        "email": "kahlil@usk.ac.id",
        "corr": False,
    },
    {
        "name": "Hammam Riza",
        "aff_idx": "a",
        "dept": "Department of Informatics, Faculty of Mathematics and Natural Sciences",
        "inst": "Universitas Syiah Kuala, Banda Aceh 23111, Indonesia",
        "email": "hammam.riza@usk.ac.id",
        "corr": False,
    },
]

#Wait check correspondence email in JOIV_Template / orig_docx:
# In orig_docx (CLEAR3D_CT_IEEE), 1st author is mfkarimi@mhs.usk.ac.id, and JOIV uses single corresponding author email line:
CORR_EMAIL_TEXT = "Corresponding author: mfkarimi@mhs.usk.ac.id"

KEYWORDS_LIST = [
    "computed tomography",
    "pulmonary nodules",
    "false-positive reduction",
    "multi-scale 3D CNN",
    "hard-negative mining",
    "focal loss",
    "probability calibration",
    "LUNA16",
]

# Structured Abstract for JIIM (strictly <= 250 words)
JIIM_ABSTRACT_SECTIONS = [
    (
        "Purpose: ",
        "Thoracic CT candidate generators produce over 500 non-nodule proposals per nodule (~571:1 skew), causing operating-threshold instability in second-stage false-positive reduction (FPR). We investigate whether multi-scale 3D squeeze-and-excitation residual representation combined with leak-free out-of-fold (OOF) hard-negative mining, focal loss, and nested Platt calibration improves natural-prevalence generalization."
    ),
    (
        "Materials and Methods: ",
        "CLEAR3D-MSFv2—a randomly initialized three-branch volumetric network spanning 21×21×21 focal (1 mm), 33×33×33 fine (1 mm), and 25×25×25 context (~2 mm) patches fused via cross-scale token self-attention—was trained on LUNA16 candidates_V2 while quarantining Subset 1. Patient-disjoint 10-fold validation on dev708 (708 scans; 603,652 candidates) isolated fold-wise OOF Scout mining. Split-RNG Fold 0 ablations (70 scans) with matched Inner Fold 1 Platt calibration (75 scans) were evaluated frozen once on untouched HOLD Subset 0 (89 scans; 79,135 candidates) and externally on SPIE-AAPM LUNGx (58 scans; 70 targets)."
    ),
    (
        "Results: ",
        "On dev708 10-fold validation, prespecified mining improved all 10 folds, raising pooled CPM from 0.6821 to 0.8680 (PR-AUC 0.4439 to 0.7240). Cross-objective threshold transfer failed completely (0.0% F1). On untouched HOLD Subset 0, frozen single-model Arm E (8-view TTA) attained 0.9993 ROC-AUC, 0.8226 PR-AUC, 0.8750 CPM (95% CI 0.8036-0.9340), and 75.54% positive F1 (+8.11 pp over an archived five-model ensemble). On LUNGx, Stage-1 LoG coverage capped sensitivity at 47/70 (67.14%, CPM 0.4340), while Stage-2 recovered 47/47 covered targets."
    ),
    (
        "Conclusion: ",
        "Fold-isolated mining paired with matched nested calibration delivers strong single-model precision generalization, while external deployment hinges on Stage-1 proposal recall."
    ),
]

# Single-paragraph JOIV Abstract: strictly 230-250 words covering objectives, materials, method, results, implication
JOIV_ABSTRACT_TEXT = (
    "Second-stage false-positive reduction (FPR) in thoracic computed tomography (CT) computer-aided detection encounters extreme candidate imbalance (~571 non-nodules per nodule) and threshold instability across objectives. This study develops and audits CLEAR3D-CT (CLEAR3D-MSFv2), a randomly initialized three-branch 3D squeeze-and-excitation residual network fused via cross-scale token self-attention over 21×21×21 focal (1 mm), 33×33×33 fine (1 mm), and 25×25×25 context (~2 mm) patches, engineered to preserve precision under natural distributions without artificial class balancing or post-hoc tuning. Under a patient-disjoint LUNA16 candidates_V2 protocol quarantining duplicate-volume Subset 1 and withholding HOLD Subset 0 (89 scans; 79,135 candidates), we evaluated a prespecified 10-fold cross-validation on 708 development scans (603,652 candidates) using fold-isolated out-of-fold (OOF) hard-negative mining, executed split-RNG Fold 0 ablations paired with Inner Fold 1 nested Platt calibration and eight-view test-time augmentation, and examined external transfer on 58 SPIE-AAPM LUNGx scans (232,000 Laplacian-of-Gaussian proposals). On dev708 10-fold validation, prespecified mining improved all 10 folds, increasing pooled Competition Performance Metric (CPM) from 0.6821 to 0.8680 (PR-AUC 0.4439 to 0.7240). On untouched HOLD Subset 0, frozen single-model Arm E achieved 0.9993 ROC-AUC, 0.8226 PR-AUC, 0.8750 CPM, and 75.54% positive F1 (+8.11 percentage points above an archived five-model ensemble), whereas LUNGx sensitivity remained capped at 47/70 (67.14%, CPM 0.4340) by first-stage proposal coverage. These results establish OOF mining and matched inner calibration as reliable requirements for 3D CT FPR deployment and highlight learned Stage-1 detection as the primary priority for future multi-center validation."
)

# Prose sections (structured modularly so JIIM maps to IMRAD + Related Work + Discussion + Declarations,
# and JOIV maps to I. Introduction → II. The Materials and Method → III. Results and Discussion → IV. Conclusion)

SEC_INTRO = [
    (
        "Lung cancer remains the leading cause of cancer-related mortality worldwide, accounting for approximately 1.8 million deaths annually across recent global cancer surveillance estimates [1], [2]. Large randomized screening trials such as the National Lung Screening Trial demonstrated that low-dose thoracic computed tomography (CT) reduces lung cancer mortality by 20% relative to chest radiography [3], prompting broadened annual screening recommendations [4]. However, widespread clinical implementation of low-dose CT screening imposes substantial radiological workload and generates frequent false-positive findings stemming from pulmonary vessels, pleural thickenings, fibrotic scars, and chest-wall structures [5]. Multi-reader consensus databases such as LIDC-IDRI [6] and standardized evaluation frameworks such as LUNA16 [7] and the SPIE-AAPM-NCI LUNGx challenge [8] have enabled reproducible computer-aided detection (CAD) benchmarking across these challenging anatomical mimics."
    ),
    (
        "Conventional pulmonary nodule CAD pipelines decouple two complementary stages: a sensitive Stage-1 candidate generator that proposes suspicious spatial coordinates, followed by a Stage-2 false-positive reduction (FPR) classifier that assigns calibrated malignancy or nodule probabilities to each proposed locus [7], [9]. Because Stage-1 generators prioritize high sensitivity, LUNA16 candidates_V2.csv contains 754,975 candidates across 888 scans—an average exceeding 840 proposals per scan and roughly 571 negative candidates per positive coordinate in typical folds. Under such severe class skew, overall classification accuracy easily exceeds 99% even when false-positive precision collapses, rendering accuracy and even area under the receiver operating characteristic curve (ROC-AUC) insufficiently discriminative compared with free-response operating characteristics (FROC), precision-recall area under the curve (PR-AUC), and positive-class F1 [12], [32]."
    ),
    (
        "Recent deep learning literature has achieved strong FROC Competition Performance Metric (CPM) scores on LUNA16 using 3D multi-scale convolutional networks [9], [14], [16]–[18], transformer-augmented backbones [25], [28], [31], [37], attention-guided extractors [24], [26], [38]–[40], and 2D/2.5D projection detectors [41], [42]. Nevertheless, direct clinical translation is frequently obscured by four methodological pitfalls identified across reporting guidelines such as CLAIM 2024, CONSORT-AI, and MINIMAR [33]–[35]: (1) reporting candidate F1 or accuracy solely on artificially balanced (1:1 or 1:3) subsets rather than natural ~571:1 candidate populations; (2) leaking information through shared normalization statistics, duplicate volumes, or global mining across training and evaluation folds; (3) transferring fixed operating thresholds or Platt calibration parameters blindly across mismatched training losses or mining distributions; and (4) conflating architectural proposals (such as pretrained backbones or dense voxel transformers) with what was actually implemented and ablated."
    ),
    (
        "To resolve these gaps rigorously, this study presents CLEAR3D-CT (implemented in our primary architecture as CLEAR3D-MSFv2), a randomly initialized dual-resolution, three-branch 3D squeeze-and-excitation residual network with Gaussian center-focus encoding and cross-scale token self-attention. Rather than claiming unexecuted pretraining or selective curated subsets, we establish an auditable end-to-end record spanning five chronological phases: (i) reconciliation of archived baseline defects and exclusion of LUNA16 Subset 1 duplicate-patient scans; (ii) a controlled prespecified 10-fold patient-disjoint cross-validation on 708 development scans (dev708; 603,652 candidates) isolating out-of-fold (OOF) hard-negative mining; (iii) split-RNG-controlled Fold 0 ablations testing cross-entropy versus focal objectives, Round-1 versus Round-2 versus mixed mining, and 4-view versus 8-view reflection test-time augmentation (TTA) with matched Inner Fold 1 Platt calibration; (iv) a single frozen one-shot evaluation on untouched HOLD Subset 0 (89 scans; 79,135 candidates); and (v) exploratory external testing on 58 unexposed SPIE-AAPM LUNGx scans (232,000 Laplacian-of-Gaussian proposals) exposing the Stage-1 coverage bottleneck."
    ),
]

SEC_RELWORK_SUBS = [
    (
        "Multi-Scale 3D Architectures and Class Imbalance in LUNA16 FPR",
        [
            "Early volumetric FPR systems showed that fusing parallel spatial fields of view enables simultaneous suppression of large vessels and small perifissural artifacts [9], [15]. Zuo et al. embedded multi-resolution contextual branches into 3D CNN classification [14], Yuan et al. combined parallel 20^3, 30^3, and 40^3 pathways to obtain 0.881 CPM on LUNA16 Track 2 [18], and Mittapalli and Thanikaiselvan reached 0.948 CPM via compound multi-scale feature fusions with iterative class balancing [16]. Concurrently, squeeze-and-excitation (SE) channel recalibration and 3D Res2Net blocks improved boundary sensitivity [17], while anchor-free 3D sphere representations [20], single-stage and multi-scale attention frameworks [21], [22], asymmetric residual factorization (3D ARCNN, CPM 0.912) [29], global channel-spatial attention (CPM 0.929) [38], and modified 3D region proposal networks (CPM 0.901) [30] advanced both Track 1 and Track 2 performance.",
            "Hybrid CNN-transformer designs have recently broadened receptive-field modeling. Mkindu et al. integrated 3D multi-scale vision transformers for nodule detection [25], Ma et al. achieved 0.9073 CPM in the Journal of Imaging Informatics in Medicine using a Transformer-in-CNN (TiCNet) detector [28], Zamanidoost et al. introduced OMS-CNN (CPM 0.892) [36] and dual-attention OMS-CNN with 3D Swin modules (CPM 0.9112) [37], while Ramezani et al. evaluated deformable detection transformers with focal loss (Lung-DETR, CPM 0.7967) [31]. Other recent 2024–2026 pipelines explored 2D multi-view YOLOv8x with background negatives (10-fold CPM 0.863) [41], whole-volume 2.5D fusion (CPM 0.7795) [42], AttentNet (CPM 0.874) [39], and dual-stage cross-layer attention (6-fold CPM 0.878) [40]. However, as summarized systematically in Table VII, high F1 scores (>94–97%) in Track 2 studies are routinely evaluated on artificially balanced 1:1 or 1:3 test sets rather than natural ~571:1 full candidate lists [12], [32], [39], [40].",
        ],
    ),
    (
        "Hard-Negative Mining, Focal Optimization, and Calibration Shift",
        [
            "Under dense candidate imbalance, focal loss [10] down-weights well-classified easy background examples so gradient capacity concentrates on ambiguous vascular bifurcations and scarred parenchyma. Simultaneously, hard-negative mining—exemplified by iterative clinical lesion harvesting [19]—feeds high-scoring false alarms back into balanced minibatches. Despite their widespread adoption, both focal weighting and non-uniform negative sampling fundamentally warp network logit priors relative to natural screening prevalence [11], [23], [27], [32]. Without matched calibration parameters and operating cutoffs fitted on unbiased OOF validation sets sharing the target loss family and sampling geometry, probability calibration degrades sharply and cross-recipe threshold transfer can fail silently [11], [32], [33]. CLEAR3D-CT directly quantifies and resolves this failure mode through matched nested Inner Fold 1 Platt calibration."
        ],
    ),
]

SEC_METHODS_SUBS = [
    (
        "Datasets, Governance Boundaries, and Cohort Isolation",
        [
            "Three thoracic CT cohorts govern this study, as itemized in Table I. Primary development and internal holdout validation utilize the public LUNA16 challenge corpus [6], [7] using official candidates_V2.csv coordinates (excluding annotations_excluded.csv findings per LUNA16 protocol during reference matching while retaining full candidate lists at inference to prevent optimistic filtering bias). Because LUNA16 Subset 1 exhibits patient/series audit anomalies relative to our legacy confirmation records (confirm_83 administrative status unsettled; 91 quarantined patients/series across Subset 1, 6 reviewed scans, and duplicate volumes), we strictly excluded all Subset 1 data from all post-audit controlled studies.",
            "Our prespecified 10-fold patient-disjoint cross-validation corpus (dev708) comprises 708 scans across 707 unique patients from Subsets 2–9 (603,652 total candidates; 1,249 positive candidates matching 946 reference nodules; single multi-scan patient assigned deterministically to Fold 2) with strictly zero overlap against Subset 1 and HOLD Subset 0. In parallel, our controlled single-model ablation framework partitions Subsets 2–9 into 8 training folds (638 scans), Inner Fold 1 calibration validation (75 scans; 64,386 candidates; 118 positives), and Outer Fold 0 development evaluation (70 scans; 62,942 candidates; 110 positives; 571:1 negative-to-positive ratio). Subset 0 (89 scans; 89 unique patients; 79,135 candidates; 138 positives; 78,997 negatives) remained frozen as the one-shot HOLD Subset 0 test set with zero development patient overlap. Finally, external cross-cohort evaluation uses the SPIE-AAPM-NCI LUNGx challenge [8], analyzing 58 unexposed test scans (70 diagnostic target centers) after reserving 2 calibration pilot scans (60 scans / 73 targets total).",
        ],
    ),
    (
        "Volumetric Preprocessing, Center-Symmetric Dual-Resolution Inputs, and Architecture",
        [
            "Forensic coordinate-frame verification confirmed that all prepared CT volumes in our workspace are resampled to exact 1.0 × 1.0 × 1.0 mm isotropic spacing in right-anterior-superior/LPS Cartesian coordinates (overturning an unverified 0.75 mm description inherited from external literature drafts). CT intensities are clipped to [-1000, +400] Hounsfield Units (HU) and linearly rescaled to [0, 1]. Around every candidate world coordinate mapped via inverse affine transformation, we extract a 51×51×51 voxel cube paired with its exact scanner boundary validity mask.",
            "Unlike asymmetric 48×48×48 crops that shift sub-voxel centers under discrete reflection, CLEAR3D-MSFv2 enforces exact odd-dimension center symmetry at voxel index 16 of the 33×33×33 fine field and index 12 of the 25×25×25 coarse field. From the 51×51×51 patch, GPU tensor slicing constructs: (1) a 33×33×33 fine grid (1.0 mm spacing; 33 mm field of view); (2) a central 21×21×21 focal sub-cube extracted from indices [6:27]; and (3) a 25×25×25 context grid smoothed by stride-2 separable binomial averaging ([0.25, 0.5, 0.25], ~2.0 mm spacing, ~50 mm field of view). Intensity normalization computes shared valid-voxel mean and standard deviation solely within the 33×33×33 fine region and encodes three distinct input channels per branch: shifted HU intensity (2*x - 1; padded -1 outside bounds), clamped local z-score in [-5, +5], and an analytical 3D Gaussian center-focus prior (sigma = 4.5 mm fine, 10.0 mm context) that focuses convolutional receptors cleanly on the candidate coordinate without hard masking peri-nodular vessels.",
            "As illustrated in Fig. 1, CLEAR3D-MSFv2(in_channels=3, dim=160, dropout=0.25) is strictly randomly initialized (zero pretrained Med3D/MedicalNet weights [13]). Each of the three parallel branches (Focal 21^3, Fine 33^3, Context 25^3) begins with an unpadded 2×2×2 Conv3d stem with GroupNorm and SiLU activation followed by three 3D Squeeze-and-Excitation Residual blocks (SEResBlock3D, reduction ratio 4). Global average pooling (augmented with inner 8 mm / 4^3 sub-cube pooling and max pooling) projects each scale via LayerNorm-SiLU linear heads into three 160-dimensional scale tokens. Rather than using voxel-level Vision Transformers, three learnable scale positional embeddings are added before a 4-head nn.MultiheadAttention residual block and two-layer SiLU feed-forward network over the three scale tokens ([B, 3, 160]), flattened into a 480-d vector and mapped via Linear(480→160)-LayerNorm-SiLU-Dropout(0.25)-Linear(160→2) to two class logits (log-odds z1 - z0). Full implementation files and training protocols ensure transparent distinction between the earlier unexecuted MedicalNet-Transformer concept and this audited CNN-token attention architecture.",
        ],
    ),
    (
        "Out-of-Fold Hard-Negative Mining, Controlled Training Arms, and Split-RNG Parity",
        [
            "To prevent subtle memorization or calibration bias across folds, our prespecified training pipeline enforces strict fold-isolated mining. Within each fold k (both dev708 10-fold CV and Outer Fold 0 ablations), a 12-epoch Scout CLEAR3DMSFv2 model (125 steps/epoch; batch 12 positives + 20 random negatives; AdamW lr=6e-4, weight decay 1e-4, cosine schedule) is trained strictly on training folds and used to rank all background candidates inside those same training folds. Minibatches during the main training phase balance 10 positives + 12 hard negatives + 10 random negatives (batch size 32; 125 steps/epoch = 2,750 gradient updates over 22 epochs baseline-improved, or 3,250 updates over 26 epochs in Arm E) drawn from 16,000 random negatives and 12,000 Scout-mined hard negatives.",
            "In Outer Fold 0 ablations, we enforce byte-for-byte split random number generator (split-RNG) parity using separate deterministic generator streams for image patches (ip_sha256), random negatives (ir_sha256), and hard negatives (ih_sha256), isolating algorithmic differences across four primary arms: Arm A (Round-1 Scout mined negatives, label-smoothed cross-entropy, 22 epochs); Arm C (Round-1 mined negatives, focal loss γ=2.0, 22 epochs); Arm D (Round-2 iteratively re-mined hard negatives, focal γ=2.0, 22 epochs); and Arm E (frozen composite configuration: 6,000 Round-1 + 6,000 Round-2 hard negatives weighted 0.5 + 0.5*max(s1, s2), softened focal loss γ=1.5 with label smoothing 0.005, 26 cosine epochs). All positive patches undergo random ±1 voxel cyclic jitter, random axial 90-degree rotations, orthogonal reflections, and mild affine contrast perturbation [0.95, 1.05] during training.",
        ],
    ),
    (
        "Nested Inner-Fold Platt Calibration, Test-Time Augmentation, and Statistical Protocol",
        [
            "At inference, CLEAR3D-CT evaluates both 4-view TTA (identity, x-flip, yz-flip, xyz-flip) and full 8-view reflection TTA across all z, y, and x axes, averaging pre-sigmoid log-odds (z1 - z0) prior to calibration. For binary operating thresholds, raw logits are calibrated strictly on an independent nested validation cohort (Inner Fold 1, n=64,386 candidates, never touching Outer Fold 0 or HOLD Subset 0) via regularized logistic regression minimizing negative log-likelihood: p_cal = sigmoid(T_0 * log_odds + b_0). Two operating rules are carried frozen to held-out evaluation: calibrated p_cal >= 0.50 (cal@0.5) and the inner-fold F1-maximizing cutoff tau* (selected automatically on Inner Fold 1).",
            "For LUNA16 FROC scoring, we apply the official LUNA16 CAD evaluator (noduleCADEvaluationLUNA16.py) with exact included-nodule radius matching R = d / 2.0 (clarifying that 5.0 mm radius fallback applies exclusively to excluded findings where annotated diameter < 0), scoring all candidates across [0.125, 0.25, 0.5, 1, 2, 4, 8] FPs/scan with 1,000-2,000 patient-clustered bootstrap resamples for 95% confidence intervals. For LUNGx external validation, we apply a 3D multi-sigma Laplacian-of-Gaussian (LoG) Stage-1 detector capped at 4,000 proposals/scan (5 mm fixed matching radius on verified zero-error DICOM LPS transforms) evaluated under Rule A (closest covering proposal) and Rule B (highest-scoring covering proposal). Notice that all binary natural-prevalence metrics (ROC-AUC, PR-AUC, Precision, Recall, Specificity, Accuracy, Pos-F1) are computed over unfiltered candidates_V2 rows without dropping excluded findings from inference denominators.",
        ],
    ),
]

SEC_RESULTS_SUBS = [
    (
        "Reconciliation of Historical Baselines and Comparator Integrity",
        [
            "Table II establishes a clean forensic baseline prior to interpreting our primary 10-fold and frozen holdout experiments. In early Phase 2 testing (83 confirmation scans), the scratch two-branch ClearNet CNN (275,232 parameters without SE blocks or multi-scale token attention) attained only 0.8143 ROC-AUC and 0.1697 Pos-F1 on Track 1 (versus 0.9751 / 0.4878 for an adapted ZuoNet), and 0.1606 Track 2 CPM largely because Stage-1 SpotNet proposals covered only 64/121 nodules (52.9%). Likewise, naive development adjustments under heavily downsampled 7:1 negative caps (clear_aug 0.1456 Pos-F1, clear_focal 0.1274) degraded performance relative to baseline clear (0.1706 Pos-F1).",
            "On 142 audited development scans (122,689 unfiltered candidates; 192 nodules), correcting an earlier evaluation bug where MedicalNet ResNet-18 had been scored on mismatched raw32 crops (erroneously yielding 0.00284 AP, 0.5861 ROC-AUC, and 0.0030 CPM) to proper 48^3 valid-mask z-score inputs yielded MedicalNet 0.0975 CPM (95% CI [0.0631, 0.1404]), 0.7765 ROC-AUC, and 0.1648 AP (PR-AUC 0.1678). On those same 142 scans, CLEAR3D-HighRes scored 0.8006 CPM (AP 0.4994) and the archived Run 004 ensemble scored 0.9129 CPM (AP 0.7974; paired delta over MedicalNet +0.8155, CI [0.7614, 0.8713]). Crucially, removing OOF hard-negative mining while matching 2,750 updates (Control B, training on 16,728 total examples rather than 28,536) dropped 142-scan CPM from 0.9040 to 0.8750 and unfiltered AP from 0.7811 to 0.6055, whereas neither static 60% hard-prior weighting (0.8929 vs. 0.9040, p = 0.699) nor gradual curriculum scheduling (0.9010 vs. 0.9040, p = 0.937) outperformed standard Scout mining. Furthermore, fit-OOF exposure auditing flagged the legacy Run 004 ensemble for global fit-pool mining dependence, motivating our clean dev708 and nested Inner Fold 1 designs.",
        ],
    ),
    (
        "Prespecified 10-Fold Patient-Disjoint Cross-Validation on dev708",
        [
            "Applying the clean fold-isolated OOF hard-negative mining recipe (12-epoch Scout → 16k random + 12k OOF hard negatives → 22-epoch Main + 4-view TTA) across all 10 patient-disjoint folds of dev708 (708 scans; 707 patients; 603,652 total candidates; 1,249 positives) yielded dramatic, consistent gains without post-hoc tuning (Table III and Fig. 4). Pooled OOF CPM rose from 0.6821 (95% CI [0.6346, 0.7243]) under the baseline OOF recipe to 0.8680 (95% CI [0.8388, 0.8967]), an improvement of +0.1859 CPM. Notably, low-FP sensitivity jumped at clinically demanding operating points: from 21.14% to 65.96% (+44.82 pp) at 0.125 FPs/scan and from 39.01% to 78.33% (+39.32 pp) at 0.25 FPs/scan, while sensitivity at 8.0 FPs/scan climbed from 94.61% to 97.25%.",
            "Equally important for cross-cohort robustness, fold-level variance contracted sharply: fold mean CPM increased from 0.7224 ± 0.1246 (range 0.4946 in Fold 3 to 0.9201 in Fold 5) to 0.8676 ± 0.0359 (range 0.8116 to 0.9355). Across paired folds, mean improvement was +0.1452 CPM (95% CI [+0.0845, +0.2122]), with 10 of 10 folds improving (minimum gain +0.0154 in Fold 5, maximum gain +0.3641 in Fold 3). Across natural unfiltered candidates, pooled ROC-AUC increased from 0.9958 to 0.9979, PR-AUC surged from 0.4439 to 0.7240 (+0.2801), and positive F1 at an uncalibrated cutoff of 0.5 rose from 21.69% (1,192 TP / 8,551 FP) to 36.67% (1,200 TP / 4,095 FP), eliminating over half of raw false alarms while detecting more positives.",
        ],
    ),
    (
        "Split-RNG Single-Model Ablations and Cross-Objective Calibration Shift on Outer Fold 0",
        [
            "Table IV details our split-RNG-controlled single-model experiments on Outer Fold 0 (70 scans; 62,942 candidates; 110 positives) paired with matched Inner Fold 1 Platt fits (75 scans; 64,386 candidates; 118 positives). Replacing label-smoothed cross-entropy (Arm A: 0.8937 CPM, PR-AUC 0.7392, matched tau* Pos-F1 66.09%) with Round-1 focal loss (γ=2.0, Arm C 4-view) raised FROC CPM to 0.9063 and tau* Pos-F1 to 67.57% (66.96% precision, 68.18% recall; 75 TP / 37 FP). Conversely, switching to exclusive Round-2 iterative mining (Arm D 4-view) increased calibrated precision to 71.43% and Pos-F1 to 69.77% (75 TP / 30 FP) at a slightly lower CPM of 0.8905, revealing an interpretable precision-ranking trade-off between first-round boundary mining and second-round ultra-hard mining.",
            "Critically, naively transferring calibration or thresholds from the uniform-negative baseline control (raw tau*_unif = 0.9912, Inner Fold 1 T_0 = 0.6849, b_0 = 5.0693) onto any focal or Round-2-mined model caused total operational collapse: sensitivity fell to 0/110 (0.0% Pos-F1) across Arms C, D, and E (Fig. 3b). Because focal optimization concentrates logit mass away from extreme saturation near 1.0, matched inner-fold fitting adjusted tau*_raw cleanly down into [0.8965, 0.9171] and restored Pos-F1 to 67.57–70.35%. Comparing 4-view versus 8-view reflection TTA showed consistent PR-AUC gains (+0.0129 on Arm C to 0.7460; +0.0084 on Arm D to 0.7387; +0.0165 on Arm E to 0.7779) with marginal FROC rank stability. Combining mixed 6k Round-1 + 6k Round-2 mining, softened focal loss (γ=1.5, ls=0.005), 26 epochs, and 8-view TTA (Arm E 8v) produced the highest single-model Outer Fold 0 discrimination (ROC-AUC 0.9990, PR-AUC 0.7779, cal@0.5 Pos-F1 70.35% at 78.65% precision, CPM 0.9016 [0.8070, 0.9725]). In accordance with full disclosure, the exploratory Arm E 4-view runner script cloned an Arm A/C key template such that both JSON entries log the identical Arm E 4-view checkpoint (0.9016 CPM), and matched-pool control Arm B trained on an expanded 514,882-candidate hard pool scored lower (0.8651 CPM, PR-AUC 0.6218).",
        ],
    ),
    (
        "Frozen One-Shot Generalization on Untouched HOLD Subset 0",
        [
            "Applying single-model Arm E (8-view TTA) and its frozen Inner Fold 1 Platt parameters (T_0 = 0.6194, b_0 = 2.7570, tau*_raw = 0.9171, tau*_cal = 0.3609) once to untouched HOLD Subset 0 (89 scans; 89 patients; 79,135 candidates; 138 positives; zero development patient overlap) confirmed strong generalization without overfit leakage (Table V, Fig. 2, Fig. 3). On HOLD Subset 0, frozen Arm E 8v achieved 0.9993 ROC-AUC, 0.8226 PR-AUC, and 0.8750 CPM (95% CI [0.8036, 0.9340]), reaching 73.21%, 75.89%, 81.25%, 91.07%, 94.64%, 96.43%, and 100.00% FROC sensitivity across 0.125 to 8.0 FPs/scan.",
            "At the frozen inner-fold operating cutoff tau*, Arm E 8v produced 105 TP, 35 FP, 78,962 TN, and 33 FN (0.39 FPs/scan)—corresponding to 76.09% sensitivity, 75.00% precision, 99.96% specificity, 99.91% accuracy, 75.54% positive F1, and 87.75% macro F1. At calibrated probability 0.50 (cal@0.5), precision rose to 84.40% with only 17 FPs across 89 scans (0.19 FPs/scan; 74.49% Pos-F1). Compared head-to-head against the archived five-model Run 004 ensemble evaluated on the exact same 89 scans (CPM 0.9043; 118 TP, 94 FP, 55.66% precision, 85.51% recall, 67.43% Pos-F1 at nat_cut = 0.99995), frozen single-model Arm E 8v used one-fifth of the ensemble checkpoints while improving positive F1 by +8.11 percentage points, raising precision by +19.34 points, and cutting false alarms by 62.8% (94 → 35 FPs).",
        ],
    ),
    (
        "Exploratory External Generalization on Unexposed SPIE-AAPM LUNGx and Stage-1 Bottleneck",
        [
            "Evaluating the archived CLEAR3D-MSFv2 ensemble on 58 unexposed external SPIE-AAPM LUNGx test scans (70 diagnostic targets; 232,000 multi-scale 3D LoG candidates capped at 4,000/scan under verified 0.000000 mm DICOM coordinate equivalence) isolates Stage-1 versus Stage-2 domain transfer (Table VI). Across the 58 scans, handcrafted LoG candidate generation captured only 47 of 70 ground-truth nodules (67.14%, 95% CI [55.38%, 78.38%]), imposing a strict 67.14% sensitivity ceiling independent of Stage-2 classifier weights.",
            "Because missing proposals cannot be rescued by Stage-2 classification, end-to-end FROC CPM dropped to 0.4279 (95% CI [0.3469, 0.5193]) under closest-proposal matching (Rule A) and 0.4340 (95% CI [0.3531, 0.5216]) under highest-scoring proposal matching (Rule B; full 60-scan cohort incl. 2 calibration pilot scans: Rule A 0.4188, Rule B 0.4247, 48/73 = 65.75% coverage). Crucially, conditional on Stage-1 coverage, CLEAR3D-MSFv2 detected all 47/47 covered nodules (100% covered recall at threshold 0.5 under Rule B, 17.22 FPs/scan) and 39/47 covered nodules (83.0% covered recall) at the strict natural threshold (2.66 FPs/scan). This matches Phase 2 Track 2 evidence (64/121 = 52.9% SpotNet coverage), demonstrating quantitatively that cross-domain CAD deployment bottlenecks lie in Stage-1 proposal sensitivity rather than Stage-2 false-positive discrimination.",
        ],
    ),
]

SEC_DISCUSSION_ONLY = [
    (
        "Methodological Lessons on Natural-Prevalence Benchmarking and Calibration Hygiene",
        [
            "Placing CLEAR3D-CT alongside recent 2017–2026 LUNA16 studies (Table VII) clarifies an important literature divergence. Studies training on LUNA16 10-fold Track 2 proposals or whole-scan detectors report competitive CPMs spanning 0.7795 (whole-volume 2.5D [42]) and 0.7967 (Lung-DETR [31]) through 0.863 (multi-view YOLOv8x [41]), 0.874–0.881 (AttentNet [39], dual-stage Wang et al. [40], Yuan et al. [18]), 0.891–0.9073 (LUNA16 combined [7], OMS-CNN [36], Lin et al. [30], TiCNet [28]), up to 0.9112–0.948 (DA OMS-CNN [37], 3D ARCNN [29], Li et al. [38], Mittapalli [16]). Our clean dev708 10-fold CPM of 0.8680 (excluding easier/quarantined Subset 1 and HOLD Subset 0) and frozen single-model HOLD Subset 0 CPM of 0.8750 (100% sensitivity at 8 FPs/scan; Fold 0 Arm C 0.9063; archived 5-model ensemble 0.9043/0.9129) sit competitively inside this benchmark band while offering stricter patient isolation and single-model inference efficiency.",
            "Equally vital is reconciling apparent positive-F1 gaps. Papers reporting 94.2% to 97.5% F1 [39], [40] evaluate balanced 1:3 or 1:1 test sets where random precision is 25–50%. Indeed, when Run 004 and HighRes are evaluated on balanced 1:3 subsets of dev-142 in Table II, Run 004 achieves 97.95% positive F1 (98.96% accuracy) despite attaining 25.02% uncalibrated F1 and 0.7974 AP on all 122,689 unfiltered candidates. On untouched HOLD Subset 0 containing 78,997 negatives against 138 positives (572:1 skew), frozen Arm E 8v reaches 75.54% positive F1 (75.00% precision) and 0.8226 PR-AUC natively without artificial negative pruning. Reporting calibrated natural-prevalence F1 alongside FROC CPM obeys modern CLAIM 2024 [33], Saito-Rehmsmeier [12], and Mosquera et al. [32] recommendations."
        ],
    ),
    (
        "Limitations, Threats to Validity, and Future Work",
        [
            "Several limitations constrain interpretation and guide next steps. First, although prespecified 10-fold dev708 validated the OOF hard-negative mining recipe at scale (0.8680 CPM), the final Arm E composite recipe (6k Round-1 + 6k Round-2 mixed mining, softened focal γ=1.5, ls=0.005, 26 epochs, 8-view TTA) was validated via Outer Fold 0 and frozen one-shot HOLD Subset 0 testing rather than across all 10 dev708 folds due to compute budgeting, and Arm E combined four design changes simultaneously rather than isolating each factor one-at-a-time (OFAT).",
            "Second, while HOLD Subset 0 guaranteed zero patient overlap with our 708 development scans and zero hyperparameter tuning after freezing Arm E, exploratory pipeline iterations earlier in project history traversed LUNA16 Subsets 0–9 overall; hence HOLD Subset 0 represents an untouched internal cohort under our audited post-freeze protocol rather than an independent multi-institution external trial [34], [35]. Third, raw candidates_V2 labeling treats annotations_excluded findings as negative during natural classification scoring (which slightly depresses apparent natural precision; e.g., excluded filtering raised Run 004 dev-142 AP from 0.588–0.797 upward). Fourth, Subset 1 (confirm_83) administrative provenance remains unsettled and strictly quarantined; LUNGx external evaluation proved limited by handcrafted Stage-1 LoG recall (67.14%); and formal institutional IRB exemption archival confirmation remains a pre-publication administrative item.",
            "Future work will therefore execute four concrete steps: (1) full 10-fold cross-validation of the frozen Arm E mixed-mining + softened-focal recipe paired with single-factor OFAT ablations isolating γ=1.5, label smoothing 0.005, and 26-epoch scheduling; (2) replacing handcrafted LoG with a frozen learned 3D Stage-1 detector paired cleanly with CLEAR3D-MSFv2 to close the 32.86% LUNGx proposal miss gap; (3) prospective validation on independent multi-center cohorts such as LNDb [34], [35] with complete lesion masks; and (4) clean open-source release of deterministic training, calibration, and FROC evaluation scripts upon acceptance."
        ],
    ),
]

SEC_CONCLUSION = [
    (
        "This study presented an audited, reproducible evaluation of CLEAR3D-CT (CLEAR3D-MSFv2) for 3D CT pulmonary nodule false-positive reduction under natural ~571:1 candidate prevalence and strict patient-disjoint governance. Across 708 clean development scans (603,652 candidates), fold-isolated out-of-fold hard-negative mining improved 10 of 10 folds, raising pooled 10-fold CPM by +0.1859 (0.6821 to 0.8680) and PR-AUC by +0.2801 (0.4439 to 0.7240). Controlled Fold 0 ablations showed that focal training improves operating trade-offs only when paired with matched nested inner-fold Platt calibration—preventing the 0.0% F1 collapse seen under cross-objective threshold transfer—while frozen one-shot testing of single-model Arm E (8-view TTA) on untouched HOLD Subset 0 attained 0.9993 ROC-AUC, 0.8226 PR-AUC, 0.8750 CPM, and 75.54% positive F1 (+8.11 points above an archived five-model ensemble). Finally, external SPIE-AAPM LUNGx testing demonstrated 100% Stage-2 recovery on covered nodules alongside a 67.14% LoG proposal ceiling, establishing first-stage proposal sensitivity as the central priority for future multi-center CAD translation."
    ),
]

# --- TABLES (7 Publication Tables) ---
TABLES = [
    {
        "num_roman": "I",
        "num_arab": "1",
        "title": "Audited Datasets, Governance Boundaries, and Cohort Isolation",
        "headers": ["Cohort / Split", "Scans (Pts)", "Candidates (Pos)", "Protocol Role & Leak-Isolation Boundary"],
        "widths_dxa_jiim": [1900, 1100, 1500, 4860],
        "widths_dxa_joiv": [1100, 760, 950, 2090],
        "rows": [
            ["Phase 2 Archive (Track 1/2)", "83 (83)", "66,135 (161)", "Legacy Scratch ZuoNet/WangNet/ClearNet benchmark (121 nodules; Subset 1)."],
            ["Dev-142 Audit Slice", "142 (142)", "122,689 (265)", "Unfiltered full-candidate comparator (Run 004, HighRes, Corrected MedicalNet; 192 nodules)."],
            ["Quarantined Subset 1", "89–91 pts", "Excluded", "Strictly excluded from dev708, Fold 0, & HOLD (duplicate-volume / confirm_83 unsettled)."],
            ["Prespecified dev708 (10-Fold)", "708 (707)", "603,652 (1,249)", "Patient-disjoint Subsets 2–9 (946 nodules); fold-isolated Scout mining; zero Subset 0/1 overlap."],
            ["Outer Fold 0 / Inner Fold 1", "70 / 75", "62,942 / 64,386", "Split-RNG single-model ablation on Fold 0 (110 pos); nested Platt fit on Inner Fold 1 (118 pos)."],
            ["Frozen HOLD Subset 0", "89 (89)", "79,135 (138)", "Untouched one-shot test (78,997 neg; 0 dev overlap); zero tuning after freezing Arm E."],
            ["External SPIE-AAPM LUNGx", "58 (+2 pilot)", "232,000 LoG", "Unexposed diagnostic CT cohort (70 test / 73 full targets; 4,000 LoG caps/scan)."],
        ],
    },
    {
        "num_roman": "II",
        "num_arab": "2",
        "title": "Historical Baselines, Uncontrolled Comparators, and Forensic Methodological Reconciliations",
        "headers": ["Model / Audit Case", "Cohort (Scans)", "Metric Profile", "Audited Methodological Root Cause"],
        "widths_dxa_jiim": [2100, 1300, 2300, 3660],
        "widths_dxa_joiv": [1180, 740, 1280, 1700],
        "rows": [
            ["ClearNet vs. ZuoNet (Track 1)", "Phase 2 (83)", "F1+: 0.1697 vs 0.4878; ROC 0.814 vs 0.975", "Scratch 2-branch CNN lacked SE, focus priors & scale attn."],
            ["Track 2 SpotNet Rescore", "Phase 2 (83)", "CPM: 0.1606 vs 0.3967; Cov: 64/121 (52.9%)", "Missing 57/121 proposals (<6 mm: 35/40) capped ceiling."],
            ["DEV_REVIEW (NEG_CAP=7)", "Dev-142 (142)", "F1+: clear 0.1706 > aug 0.1456 > focal 0.127", "7:1 neg cap, 8 epochs & missing focus/mining prevented gain."],
            ["MedicalNet (Mismatched raw32)", "Dev-142 (142)", "AP: 0.00284, ROC: 0.5861, CPM: 0.0030", "BUGGY raw32 crop artifact; formally retracted vs 0.0424->0.904."],
            ["MedicalNet (Corrected 48^3)", "Dev-142 (142)", "CPM: 0.0975 [0.063–0.140], AP: 0.1648", "Pretrained Med3D ResNet-18 [13] valid-mask z-score on natural 122,689 rows."],
            ["CLEAR3D-HighRes (Fold 0)", "Dev-142 (142)", "CPM: 0.8006 [0.736–0.852], AP: 0.4994", "High-res baseline; 1:3 balanced Pos-F1 0.9355 (unfilt AP 0.4994)."],
            ["Run 004 Ens vs Controls A/B", "Dev-142 (142)", "Ens 0.9129; F0 0.9040; FineOnly 0.8653; NoMine 0.8750", "Control B removed mining (16,728 bank → 0.8750; AP 0.6055)."],
            ["Static HardPrior / Curriculum", "Dev-142 (142)", "CPM: 0.8929 (p=0.699) / 0.9010 (p=0.937)", "No statistically significant gain over Fold 0 Scout (0.9040)."],
        ],
    },
    {
        "num_roman": "III",
        "num_arab": "3",
        "title": "Prespecified 10-Fold Patient-Disjoint Cross-Validation on dev708 (708 Scans, 603,652 Candidates)",
        "headers": ["Fold", "Scans (Cands / Pos)", "Baseline CPM", "Improved CPM", "Delta CPM", "Imp. PR-AUC", "Imp. F1+@0.5"],
        "widths_dxa_jiim": [750, 1850, 1250, 1250, 1150, 1450, 1660],
        "widths_dxa_joiv": [420, 960, 720, 720, 660, 710, 710],
        "rows": [
            ["Fold 0", "70 (62,942 / 110)", "0.8397", "0.8968", "+0.0571", "0.7504", "37.45%"],
            ["Fold 1", "75 (64,386 / 118)", "0.8179", "0.9017", "+0.0838", "0.6477", "31.36%"],
            ["Fold 2", "76 (63,886 / 127)", "0.6295", "0.8512", "+0.2217", "0.7753", "36.52%"],
            ["Fold 3", "69 (57,075 / 126)", "0.4946", "0.8587", "+0.3641", "0.7417", "33.56%"],
            ["Fold 4", "68 (57,121 / 101)", "0.6452", "0.8807", "+0.2355", "0.7510", "33.45%"],
            ["Fold 5", "71 (60,006 / 144)", "0.9201", "0.9355", "+0.0154", "0.7785", "40.52%"],
            ["Fold 6", "72 (61,818 / 134)", "0.7965", "0.8437", "+0.0472", "0.7179", "37.93%"],
            ["Fold 7", "71 (57,492 / 139)", "0.7253", "0.8116", "+0.0863", "0.7703", "39.24%"],
            ["Fold 8", "67 (57,103 / 125)", "0.6487", "0.8481", "+0.1994", "0.7268", "38.74%"],
            ["Fold 9", "69 (61,823 / 125)", "0.7070", "0.8484", "+0.1414", "0.6936", "38.67%"],
            ["Fold Mean±SD", "10 Folds (10/10 imp.)", "0.7224±0.1246", "0.8676±0.0359", "+0.1452", "—", "—"],
            ["Pooled OOF", "708 (603,652 / 1,249)", "0.6821", "0.8680", "+0.1859", "0.7240 (vs 0.444)", "36.67% (vs 21.7%)"],
        ],
    },
    {
        "num_roman": "IV",
        "num_arab": "4",
        "title": "Controlled Split-RNG Single-Model Ablations on Outer Fold 0 (70 Scans, 62,942 Candidates)",
        "headers": ["Arm / Configuration", "TTA", "CPM [95% CI]", "ROC-AUC", "PR-AUC", "Prec / Rec @tau*", "Pos-F1 (@tau* | cal@0.5)"],
        "widths_dxa_jiim": [2200, 550, 1650, 850, 850, 1500, 1760],
        "widths_dxa_joiv": [1100, 340, 860, 500, 500, 780, 820],
        "rows": [
            ["Uniform Control (R1 Rand)", "4v", "0.8968", "0.9982", "0.7504", "81.48% / 60.00%", "69.11% | 0.0% xfer"],
            ["Arm A (R1 Mined + CE)", "4v", "0.8937 [0.806–0.966]", "0.9987", "0.7392", "63.33% / 69.09%", "66.09% | 67.69%"],
            ["Arm B (514k Matched Pool CE)", "4v", "0.8651", "0.9986", "0.6218", "50.93% / 74.55%", "60.52% | 0.00%"],
            ["Arm C (R1 + Focal γ=2.0)", "4v", "0.9063 [0.810–0.969]", "0.9987", "0.7331", "66.96% / 68.18%", "67.57% | 62.30%"],
            ["Arm C (R1 + Focal γ=2.0)", "8v", "0.9048 [0.807–0.970]", "0.9987", "0.7460", "68.87% / 66.36%", "67.59% | 63.78%"],
            ["Arm D (R2 + Focal γ=2.0)", "4v", "0.8905 [0.807–0.964]", "0.9988", "0.7303", "71.43% / 68.18%", "69.77% | 60.92%"],
            ["Arm D (R2 + Focal γ=2.0)", "8v", "0.8857 [0.793–0.964]", "0.9986", "0.7387", "73.40% / 62.73%", "67.65% | 63.95%"],
            ["Arm E (Mix R1+R2, g=1.5, 26e)", "4v", "0.9016 [0.803–0.967]", "0.9989", "0.7613", "59.85% / 74.55%", "66.40% | 69.31%"],
            ["Arm E (Mix R1+R2, g=1.5, 26e)", "8v", "0.9016 [0.807–0.973]", "0.9990", "0.7779", "65.04% / 72.73%", "68.67% | 70.35%"],
        ],
    },
    {
        "num_roman": "V",
        "num_arab": "5",
        "title": "Frozen One-Shot Generalization on Untouched HOLD Subset 0 (89 Scans, 79,135 Candidates)",
        "headers": ["Model & Operating Point", "TP / FP / FN", "Prec (%)", "Rec (%)", "Spec / Acc (%)", "Pos-F1 (%)", "CPM / AUC / PR-AUC"],
        "widths_dxa_jiim": [2100, 1250, 850, 850, 1250, 900, 2160],
        "widths_dxa_joiv": [1120, 720, 480, 480, 680, 500, 920],
        "rows": [
            ["Frozen Arm E 8v (raw @0.50)", "133 / 370 / 5", "26.44", "96.38", "99.53 / 99.53", "41.50", "0.8750 / 0.9993 / 0.8226"],
            ["Frozen Arm E 8v (cal @0.50)", "92 / 17 / 46", "84.40", "66.67", "99.98 / 99.92", "74.49", "0.8750 / 0.9993 / 0.8226"],
            ["Frozen Arm E 8v (tau*=0.9171)", "105 / 35 / 33", "75.00", "76.09", "99.96 / 99.91", "75.54", "0.8750 [0.804–0.934]"],
            ["Archived 5-Model Ens (nat_cut)", "118 / 94 / 20", "55.66", "85.51", "99.88 / 99.86", "67.43", "0.9043 [0.845–0.953]"],
            ["dev708 Pooled FROC Grid (8 pts):", "0.125: 0.660", "0.25: 0.783", "0.5: 0.854", "1.0: 0.910 | 2.0: 0.937", "4.0: 0.960", "8.0: 0.973 (CPM 0.8680)"],
            ["HOLD Arm E 8v FROC Grid:", "0.125: 0.732", "0.25: 0.759", "0.5: 0.813", "1.0: 0.911 | 2.0: 0.946", "4.0: 0.964", "8.0: 1.000 (CPM 0.8750)"],
        ],
    },
    {
        "num_roman": "VI",
        "num_arab": "6",
        "title": "Exploratory External Generalization on Unexposed SPIE-AAPM LUNGx (LoG Cap 4,000/Scan)",
        "headers": ["Cohort / Matching Rule", "Scans (Targets)", "LoG Coverage", "nat_cut Sens (FPs/scan)", "@0.5 Sens (FPs/scan)", "CPM [95% CI]"],
        "widths_dxa_jiim": [2000, 1100, 1500, 1650, 1550, 1560],
        "widths_dxa_joiv": [1040, 640, 820, 840, 780, 780],
        "rows": [
            ["Unexposed Test — Rule A (Closest)", "58 (70)", "47/70 (67.14%)", "55.71% (39/70, 2.66)", "64.29% (45/70, 17.22)", "0.4279 [0.347–0.519]"],
            ["Unexposed Test — Rule B (Highest)", "58 (70)", "47/70 (67.14%)", "55.71% (39/70, 2.66)", "67.14% (47/70, 17.22)", "0.4340 [0.353–0.522]"],
            ["Full Cohort (incl. 2 Pilot) — Rule A", "60 (73)", "48/73 (65.75%)", "54.79% (40/73, 2.72)", "63.01% (46/73, 18.00)", "0.4188 [0.340–0.509]"],
            ["Full Cohort (incl. 2 Pilot) — Rule B", "60 (73)", "48/73 (65.75%)", "54.79% (40/73, 2.72)", "65.75% (48/73, 18.00)", "0.4247 [0.342–0.513]"],
        ],
    },
    {
        "num_roman": "VII",
        "num_arab": "7",
        "title": "Methodological Context Comparison Against Peer-Reviewed LUNA16 Literature (2017–2026)",
        "headers": ["Study & Year", "Track & Modality", "Protocol & Balance", "Reported CPM", "Candidate F1 / Context"],
        "widths_dxa_jiim": [1950, 1750, 2100, 1200, 2360],
        "widths_dxa_joiv": [1060, 920, 1080, 680, 1160],
        "rows": [
            ["Setio et al. (2017) [7]", "Track 2 Combined", "10-fold CV (888 scans)", "0.891", "Official challenge ensemble"],
            ["Dou et al. (2017) [9]", "Track 2 3D DCNN", "10-fold CV (candidates_V2)", "0.839", "96.8% balanced Acc"],
            ["Mittapalli & Thanikaiselvan (2021) [16]", "Track 2 Multi-scale", "10-fold CV, iterative bal.", "0.948", "Balanced iterative scheme"],
            ["Yuan et al. (2021) [18]", "Track 2 Multi-path 3D", "10-fold CV (candidates_V2)", "0.881", "Sens 0.962 @ 8 FPs/scan"],
            ["Hao et al. (2023) [26]", "Track 2 Central Attn", "10-fold CV (candidates_V2)", "NR", "AUC 95.67%, Sens 92.64%"],
            ["Ma et al. (TiCNet, 2024) [28]", "Track 1 3D CNN-Trans.", "10-fold CV (JIIM)", "0.9073", "Whole-scan detector"],
            ["Liu et al. (3D ARCNN, 2024) [29]", "Track 2 3D Residual", "10-fold CV (candidates_V2)", "0.912", "Sens 95.8% @ 8 FPs/scan"],
            ["Lin et al. (2024) [30]", "Track 1 3D RPN", "10-fold CV (888 scans)", "0.901", "Whole-scan detector"],
            ["Ramezani et al. (Lung-DETR, 2024) [31]", "Track 1 Def. DETR", "LUNA16 benchmark", "0.7967", "End-to-end 3D transformer"],
            ["Zamanidoost et al. (OMS-CNN, 2025) [36]", "Track 1 Multi-scale", "10-fold CV", "0.892", "Whole-scan multi-scale CNN"],
            ["Zamanidoost et al. (DA OMS, 2025) [37]", "Track 1 Dual-Attn+Swin", "10-fold CV", "0.9112", "Dual attention + 3D Swin"],
            ["Li et al. (2025) [38]", "Track 1+2 MS+Attn", "10-fold CV", "0.929", "Sens 0.977 @ 2 FPs/scan"],
            ["Almahasneh et al. (AttentNet, 2025) [39]", "Track 1+2 3D Attn", "10-fold CV (1:3 bal. F1)", "0.874", "94.2% Pos-F1 on 1:3 sample"],
            ["Wang et al. (2026) [40]", "Track 1+2 Cross-Attn", "6-fold CV (1:1 bal. F1)", "0.878", "97.5% Pos-F1 on 1:1 sample"],
            ["Silva & Pedrini (2026) [41]", "Track 1 2D YOLOv8x", "10-fold CV (2D proj.)", "0.863", "2D slice detection"],
            ["Chou et al. (2026) [42]", "Track 1 2.5D Volume", "10-fold CV", "0.7795", "Whole-volume 2.5D baseline"],
            ["CLEAR3D-CT (Ours: dev708 10-Fold)", "Track 2 3-Branch 3D", "Zero Sub0/Sub1 leak (4v)", "0.8680", "PR-AUC 0.7240; 10/10 imp."],
            ["CLEAR3D-CT (Ours: Frozen HOLD 8v)", "Track 2 3-Branch 3D", "89 untouched scans (~572:1)", "0.8750", "Nat Pos-F1 75.54%; PR 0.823"],
            ["CLEAR3D-CT (Ours: F0 Arm C / Ens)", "Track 2 3-Branch 3D", "F0 single (70) / Arch Ens", "0.9063 / 0.9129", "1:3 Dev Pos-F1 97.95% (Ens)"],
        ],
    },
]

FIGURES = [
    {
        "num": "1",
        "path": "journal_package_20261117/figures/fig1_clear3d_architecture.png",
        "caption": "Fig. 1. Architecture of CLEAR3D-CT (CLEAR3D-MSFv2, dim = 160, dropout = 0.25), illustrating center-symmetric 21^3 focal, 33^3 fine (1 mm), and 25^3 context (~2 mm) three-channel volumetric stems, 3D squeeze-and-excitation residual stages, and cross-scale 4-head multi-head self-attention over three scale tokens followed by nested Inner Fold 1 Platt calibration and 8-view reflection TTA."
    },
    {
        "num": "2",
        "path": "journal_package_20261117/figures/fig2_froc_comparison.png",
        "caption": "Fig. 2. Official LUNA16 scan-level Free-Response Receiver Operating Characteristic (FROC) sensitivity curves evaluated on unfiltered candidates across dev708 10-fold Baseline OOF (CPM 0.6821) versus Improved OOF (CPM 0.8680), Outer Fold 0 Arm C (0.9063) and Arm E 8v (0.9016), archived five-model ensemble (0.9043), and untouched frozen HOLD Subset 0 Arm E 8v (0.8750, 100.0% sensitivity at 8 FPs/scan)."
    },
    {
        "num": "3",
        "path": "journal_package_20261117/figures/fig3_pr_and_calibration.png",
        "caption": "Fig. 3. Natural-prevalence discrimination (~572:1 negative skew) and calibration transfer behavior. (a) Candidate-level precision-recall curves for dev708 Baseline vs. Improved, Fold 0 Arm E 8v (PR-AUC 0.7779), and untouched HOLD Subset 0 Arm E 8v (PR-AUC 0.8226). (b) Precision, recall, and positive F1 comparing complete cross-objective collapse (0.0% F1+) under naive threshold transfer against matched Inner Fold 1 calibration on Fold 0 (70.35% cal@0.5) and HOLD Subset 0 (75.54% frozen tau*, +8.11 pp over archived 5-model ensemble)."
    },
    {
        "num": "4",
        "path": "journal_package_20261117/figures/fig4_dev708_10fold_paired.png",
        "caption": "Fig. 4. Per-fold official LUNA16 CPM across all 10 patient-disjoint folds of dev708 (708 scans; 603,652 candidates) showing consistent gains (+0.015 to +0.364 CPM; 10/10 folds improved) and substantial variance contraction (SD 0.1246 → 0.0359) from prespecified fold-isolated out-of-fold hard-negative mining."
    },
]

# Synchronized 42 English Verified References matching 04_VERIFIED_LITERATURE_MATRIX.md
REFS = [
    "[1] F. Bray, M. Laversanne, H. Sung, J. Ferlay, R. L. Siegel, I. Soerjomataram, and A. Jemal, “Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries,” CA Cancer J. Clin., vol. 74, no. 3, pp. 229–263, 2024, doi: 10.3322/caac.21834.",
    "[2] R. L. Siegel, T. B. Kratzer, A. N. Giaquinto, H. Sung, and A. Jemal, “Cancer statistics, 2025,” CA Cancer J. Clin., vol. 75, no. 1, pp. 10–45, 2025, doi: 10.3322/caac.21871.",
    "[3] D. R. Aberle et al. (NLST Research Team), “Reduced lung-cancer mortality with low-dose computed tomographic screening,” N. Engl. J. Med., vol. 365, no. 5, pp. 395–409, 2011, doi: 10.1056/NEJMoa1102873.",
    "[4] US Preventive Services Task Force, “Screening for lung cancer: US Preventive Services Task Force recommendation statement,” JAMA, vol. 325, no. 10, pp. 962–970, 2021, doi: 10.1001/jama.2021.1117.",
    "[5] M. M. Hammer et al., “Patient and radiologist factors associated with false-positive lung cancer screening examinations,” Acad. Radiol., vol. 29, suppl. 2, pp. S18–S22, 2020, doi: 10.1016/j.acra.2020.07.040.",
    "[6] S. G. Armato III et al., “The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans,” Med. Phys., vol. 38, no. 2, pp. 915–931, 2011, doi: 10.1118/1.3528204.",
    "[7] A. A. A. Setio et al., “Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge,” Med. Image Anal., vol. 42, pp. 1–13, 2017, doi: 10.1016/j.media.2017.06.015.",
    "[8] S. G. Armato III et al., “SPIE-AAPM-NCI Lung Nodule Classification Challenge Dataset (LUNGx),” The Cancer Imaging Archive (TCIA), 2015, doi: 10.7937/K9/TCIA.2015.UZLSU3FL.",
    "[9] Q. Dou, H. Chen, L. Yu, J. Qin, and P.-A. Heng, “Multilevel contextual 3-D CNNs for false positive reduction in pulmonary nodule detection,” IEEE Trans. Biomed. Eng., vol. 64, no. 7, pp. 1558–1567, 2017, doi: 10.1109/TBME.2016.2613502.",
    "[10] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for dense object detection,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Venice, Italy, 2017, pp. 2980–2988, doi: 10.1109/ICCV.2017.324.",
    "[11] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. Mach. Learn. Res. (ICML), vol. 70, pp. 1321–1330, 2017, arXiv:1706.04599.",
    "[12] T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” PLoS ONE, vol. 10, no. 3, p. e0118432, 2015, doi: 10.1371/journal.pone.0118432.",
    "[13] S. Chen, K. Ma, and Y. Zheng, “Med3D: Transfer learning for 3D medical image analysis,” arXiv preprint arXiv:1904.00625, 2019, doi: 10.48550/arXiv.1904.00625.",
    "[14] W. Zuo, F. Zhou, and Y. He, “An embedded multi-branch 3D convolution neural network for false positive reduction in lung nodule detection,” J. Digit. Imaging, vol. 33, no. 4, pp. 846–857, 2020, doi: 10.1007/s10278-020-00326-0.",
    "[15] Y. Gu et al., “A survey of computer-aided diagnosis of lung nodules from CT scans using deep learning,” Comput. Biol. Med., vol. 137, art. no. 104806, 2021, doi: 10.1016/j.compbiomed.2021.104806.",
    "[16] P. S. Mittapalli and V. Thanikaiselvan, “Multiscale CNN with compound fusions for false positive reduction in lung nodule detection,” Artif. Intell. Med., vol. 113, art. no. 102017, 2021, doi: 10.1016/j.artmed.2021.102017.",
    "[17] H. Peng, H. Sun, and Y. Guo, “3D multi-scale deep convolutional neural networks for pulmonary nodule detection,” PLoS ONE, vol. 16, no. 1, p. e0244406, 2021, doi: 10.1371/journal.pone.0244406.",
    "[18] H. Yuan, Z. Fan, Y. Wu, and J. Cheng, “An efficient multi-path 3D convolutional neural network for false-positive reduction of pulmonary nodule detection,” Int. J. Comput. Assist. Radiol. Surg., vol. 16, no. 12, pp. 2269–2277, 2021, doi: 10.1007/s11548-021-02478-y.",
    "[19] J. Cai, A. P. Harrison, Y. Zheng, K. Yan, Y. Huo, J. Xiao, L. Yang, and L. Lu, “Lesion-Harvester: Iteratively mining unlabeled lesions and hard-negative examples at scale,” IEEE Trans. Med. Imaging, vol. 40, no. 1, pp. 59–70, 2021, doi: 10.1109/TMI.2020.3022034.",
    "[20] X. Luo, T. Song, G. Wang, J. Chen, Y. Chen, K. Li, D. N. Metaxas, and S. Zhang, “SCPM-Net: An anchor-free 3D lung nodule detection network using sphere representation and center points matching,” Med. Image Anal., vol. 75, art. no. 102287, 2022, doi: 10.1016/j.media.2021.102287.",
    "[21] Y.-S. Huang, P.-R. Chou, H.-M. Chen, Y.-C. Chang, and R.-F. Chang, “One-stage pulmonary nodule detection using 3-D DCNN with feature fusion and attention mechanism in CT image,” Comput. Methods Programs Biomed., vol. 220, art. no. 106786, 2022, doi: 10.1016/j.cmpb.2022.106786.",
    "[22] S. Zheng, S. Kong, Z. Huang, L. Pan, T. Zeng, B. Zheng, M. Yang, and Z. Liu, “A lower false positive pulmonary nodule detection approach for early lung cancer screening,” Diagnostics, vol. 12, no. 11, art. no. 2660, 2022, doi: 10.3390/diagnostics12112660.",
    "[23] S. Rajaraman, P. Ganesan, and S. Antani, “Deep learning model calibration for improving performance in class-imbalanced medical image classification tasks,” PLoS ONE, vol. 17, no. 1, p. e0262838, 2022, doi: 10.1371/journal.pone.0262838.",
    "[24] Z. Zhang, Y. Tie, D. Zhang, F. Liu, and L. Qi, “Quantum-Involution inspire false positive reduction in pulmonary nodule detection,” Biomed. Signal Process. Control, vol. 84, art. no. 104850, 2023, doi: 10.1016/j.bspc.2023.104850.",
    "[25] H. Mkindu, L. Wu, and Y. Zhao, “3D multi-scale vision transformer for lung nodule detection in chest CT images,” Signal Image Video Process., vol. 17, no. 5, pp. 2473–2480, 2023, doi: 10.1007/s11760-022-02464-0.",
    "[26] K. Hao, B. Li, X. Zhou, et al., “Lung nodule false positive reduction using a central attention convolutional neural network on imbalanced data,” in Proc. SPIE Med. Imaging: Comput.-Aided Diagn., vol. 12465, 2023, doi: 10.1117/12.2654216.",
    "[27] T. Dawood, C. Chen, R. Razavi, A. P. King, and E. Puyol-Anton, “Addressing uncertainty in deep learning for medical image analysis: A review of uncertainty estimation and calibration,” Med. Image Anal., vol. 88, art. no. 102861, 2023, doi: 10.1016/j.media.2023.102861.",
    "[28] L. Ma, G. Li, X. Feng, Q. Fan, and L. Liu, “TiCNet: Transformer in convolutional neural network for pulmonary nodule detection on CT images,” J. Imaging Inform. Med., vol. 37, no. 1, pp. 196–208, 2024, doi: 10.1007/s10278-023-00904-y.",
    "[29] B. Liu, H. Song, Q. Li, Y. Lin, X. Weng, Z. Su, and J. Yang, “3D ARCNN: An asymmetric residual CNN for false positive reduction in pulmonary nodule,” IEEE Trans. Nanobioscience, vol. 23, no. 1, pp. 18–25, 2024, doi: 10.1109/TNB.2023.3278706.",
    "[30] C.-Y. Lin, S.-M. Guo, J.-J. J. Lien, W.-T. Lin, Y.-S. Liu, C.-H. Lai, and C.-Y. Hsu, “Modified 3D region proposal network for automated pulmonary nodule detection,” Cancer Imaging, vol. 24, no. 1, art. no. 40, 2024, doi: 10.1186/s40644-024-00683-x.",
    "[31] H. Ramezani, D. Aleman, and D. Letourneau, “Lung-DETR: Deformable detection transformer for sparse lung nodule anomaly detection,” arXiv preprint arXiv:2409.05200, 2024, doi: 10.48550/arXiv.2409.05200.",
    "[32] C. Mosquera, L. Ferrer, D. Milone, D. Luna, and E. Ferrante, “Class imbalance on medical image classification: Towards better evaluation practices for discrimination and calibration performance,” Eur. Radiol., vol. 34, no. 12, pp. 7895–7903, 2024, doi: 10.1007/s00330-024-10834-z.",
    "[33] A. S. Tejani et al., “Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update,” Radiol. Artif. Intell., vol. 6, no. 4, p. e240300, 2024, doi: 10.1148/ryai.240300.",
    "[34] X. Liu, S. Cruz Rivera, D. Moher, M. J. Calvert, and A. K. Denniston, “Reporting guidelines for clinical trial reports for interventions involving artificial intelligence: the CONSORT-AI extension,” Nat. Med., vol. 26, no. 9, pp. 1364–1374, 2020, doi: 10.1038/s41591-020-1034-x.",
    "[35] T. Hernandez-Boussard, S. Bozkurt, J. P. A. Ioannidis, and N. H. Shah, “Transparent reporting of AI in medical research,” Nat. Med., vol. 26, no. 10, pp. 1364–1366, 2020, doi: 10.1038/s41591-020-1041-1.",
    "[36] Y. Zamanidoost, T. Ould-Bachir, and S. Martel, “OMS-CNN: Optimized multi-scale CNN for lung nodule detection based on Faster R-CNN,” IEEE J. Biomed. Health Inform., 2025, doi: 10.1109/JBHI.2024.3507360.",
    "[37] Y. Zamanidoost, T. Ould-Bachir, and S. Martel, “DA OMS-CNN: Dual attention and 3D Swin transformer-based lung nodule detection,” Informatics, vol. 12, no. 3, art. no. 65, 2025, doi: 10.3390/informatics12030065.",
    "[38] Y. Li, Y. Zhao, L. Wang, X. Chen, and Z. Liu, “Lung nodule detection using a multi-scale convolutional neural network and global channel spatial attention mechanisms,” Sci. Rep., vol. 15, art. no. 12313, 2025, doi: 10.1038/s41598-025-97187-w.",
    "[39] M. Almahasneh, A. Pai, M. J. Al-Rawi, and Y. Shen, “AttentNet: 3D attention-based lung nodule detection and classification,” SN Comput. Sci., vol. 6, art. no. 292, 2025, doi: 10.1007/s42979-025-03799-4.",
    "[40] L. Wang, Q. Wang, H. Gao, et al., “A dual-stage 3D CNN with cross-layer attention for pulmonary nodule detection,” Digital Health, vol. 12, pp. 1–16, 2026, doi: 10.1177/20552076261419238.",
    "[41] L. Silva and H. Pedrini, “Multi-view YOLOv8x with background images for pulmonary nodule detection,” Sensors, vol. 26, no. 19, art. no. 6246, 2026, doi: 10.3390/s26196246.",
    "[42] L.-F. Chou, S.-C. Tai, C.-H. Wu, et al., “Automated pulmonary nodule detection across whole thoracic CT volumes via multi-planar 2.5D fusion,” Diagnostics, vol. 16, no. 18, art. no. 3038, 2026, doi: 10.3390/diagnostics16183038.",
]

ACKNOWLEDGMENT_BLIND = (
    "[Redacted for Double-Blind Peer Review. Funding scholarship, host university informatics department, "
    "computing GPU resources, and thesis co-supervisors are disclosed on the separate Title Page.]"
)

ACKNOWLEDGMENT_FULL = (
    "The first author gratefully acknowledges support from the Kemitraan Negara Berkembang (KNB) "
    "Scholarship Program, Ministry of Education, Culture, Research, and Technology of the Republic of Indonesia, "
    "and computational infrastructure hosted at the Faculty of Mathematics and Natural Sciences and Department "
    "of Informatics, Universitas Syiah Kuala, Banda Aceh, Indonesia. The authors thank the curators of the public "
    "LIDC-IDRI, LUNA16, and SPIE-AAPM-NCI LUNGx datasets for open scientific benchmark access."
)

DECLARATIONS_BLIND = [
    ("Funding: ", "Supported by a governmental graduate scholarship grant (details blinded on the Title Page)."),
    ("Competing Interests: ", "The authors declare that they have no competing financial or non-financial interests."),
    ("Ethical Approval and Consent to Participate: ", "This computational study exclusively analyzed de-identified public thoracic CT benchmarks (LIDC-IDRI, LUNA16, SPIE-AAPM LUNGx); institutional ethics exemption documentation is archived separately for the Title Page."),
    ("Author Contributions (CRediT): ", "[Redacted for double-blind review; detailed in Title Page.]"),
    ("Data and Code Availability: ", "All datasets (LUNA16 candidates_V2, LIDC-IDRI, SPIE-AAPM LUNGx) are publicly accessible via official challenge repositories and TCIA. Deterministic training, split-governance manifests, nested Platt fitting, and FROC evaluation scripts will be released publicly upon acceptance."),
    ("Use of Generative AI: ", "Generative AI coding and editorial assistants supported script verification, literature audit organization, and English language polishing under full human author verification of every statistical artifact, citation, and clinical claim."),
]

if __name__ == "__main__":
    import re
    jiim_abs_wc = len((" ".join(lbl + txt for lbl, txt in JIIM_ABSTRACT_SECTIONS)).split())
    joiv_abs_wc = len(JOIV_ABSTRACT_TEXT.split())
    print(f"JIIM Abstract Words: {jiim_abs_wc} (limit <=250)")
    print(f"JOIV Abstract Words: {joiv_abs_wc} (required 230-250)")
    assert jiim_abs_wc <= 250, f"JIIM abstract too long: {jiim_abs_wc}"
    assert 230 <= joiv_abs_wc <= 250, f"JOIV abstract out of 230-250 bounds: {joiv_abs_wc}"
    all_prose = " ".join(SEC_INTRO) + " "
    for _, ps in SEC_RELWORK_SUBS + SEC_METHODS_SUBS + SEC_RESULTS_SUBS + SEC_DISCUSSION_ONLY:
        all_prose += " ".join(ps) + " "
    all_prose += " ".join(SEC_CONCLUSION) + " "
    for tb in TABLES:
        for row in tb["rows"]:
            all_prose += " ".join(row) + " "
    cited = set()
    for m in re.finditer(r"\[(\d+)(?:[\u2013–-](\d+))?\]", all_prose):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        for k in range(a, b + 1):
            cited.add(k)
    missing = set(range(1, 43)) - cited
    extra = cited - set(range(1, 43))
    print(f"Total REFS: {len(REFS)}, Cited count: {len(cited)}, Missing: {sorted(missing)}, Extra: {sorted(extra)}")
    assert not missing and not extra, f"Citation mismatch: missing={missing}, extra={extra}"
    print("MANUSCRIPT_DATA_SELF_CHECK_OK")
