# Baseline specifications

Author-reported numbers were read from the papers on 25 October 2026. No reproduction has been trained.

Wang et al., Digital Health, 1 January 2026. DOI 10.1177/20552076261419238. The paper labels its comparison row 2025.

LUNA16 stage-2 classifier, Table 3 average: recall 0.949, precision 0.975, F1 0.959, AUC 0.980.
LIDC-IDRI stage-2, Table 5, six-model average after removing scans used to train each model: recall 0.932, precision 0.942, F1 0.937, AUC 0.968.
Full-system LUNA CPM 0.878. Table 8 sensitivities at 0.125 through 8 FP/scan: 0.676, 0.793, 0.891, 0.939, 0.948, 0.949, 0.949. These detection figures are not the classifier tables.
Stage 2 uses patches 32x48x48, 16x24x24, and 8x12x12, cross-entropy, Adam at 0.0001, and a 1:3 positive-to-negative training ratio. The printed cross-validation is six subsets, 0 through 5.

Zuo et al. 2020, DOI 10.1007/s10278-020-00326-0. Abstract: accuracy 0.9783, sensitivity 0.8771, precision 0.9426, specificity 0.9925. Table 6 CPM 0.830, with sensitivities 0.630, 0.753, 0.819, 0.869, 0.903, 0.915, 0.920. The classification subset discussed with Table 3 is 1320 positives and 542061 negatives among 543381 crops. The CPM submission used all 551065 candidates. Those are different experiments.
Zuo samples 18 cubed raw CT from candidate coordinates, shifts positive candidate centers, uses no lung segmentation, trains from scratch for 15 epochs at batch 128 and learning rate 0.01, and uses Keras class_weight auto.

None of zuo18-001, luna-expand-001, multibranch-001, med48-001, or redesign-001 is that reproduction. redesign-001 positives are brightness-filtered reference centers, and its negatives are a 7-per-positive sample.
