#!/usr/bin/env python3
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline')
FIG_DIR = Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/01_Paper_and_Manuscripts/journal_package_20261117/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 9.5,
    'axes.edgecolor': '#334155',
    'axes.linewidth': 0.9,
})

# ==========================================
# FIGURE 1: CLEAR3D-MSFv2 Architecture Diagram
# ==========================================
def make_fig1():
    fig, ax = plt.subplots(figsize=(10.2, 4.4), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis('off')

    def box(x, y, w, h, fc, ec, title, subtitle='', title_size=8.5, sub_size=7.3, lw=1.2):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35,rounding_size=0.9', facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(p)
        if subtitle:
            ax.text(x + w/2, y + h*0.64, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color='#0f172a')
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=sub_size, color='#1e293b')
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color='#0f172a')

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='#334155', lw=1.25))

    # Title banner note
    ax.text(50, 42.3, 'CLEAR3D-CT (CLEAR3D-MSFv2, dim=160, dropout=0.25) — Randomly Initialized Volumetric Multi-Scale Network', ha='center', va='center', fontsize=9.8, fontweight='bold', color='#0f172a')

    # Input Volume
    box(1.5, 11.5, 16.5, 22.0, '#e2e8f0', '#475569',
        '3D CT Patch (51\u00b3)\n1 mm Isotropic',
        'HU [-1000, 400]\n+ Valid Mask\nShared Fine Ref:\n(mean, std from 33\u00b3)\n3 Channels:\n1) Intensity\n2) Local z-score\n3) Gaussian Focus', title_size=8.2, sub_size=7.0)

    # Three Branches
    box(22.5, 29.0, 22.5, 9.2, '#dbeafe', '#1d4ed8',
        'Focal Branch (21\u00b3 @ 1 mm)',
        'Center Crop -> Conv3D + GN + SiLU\n3x SEResBlock3D -> GAP -> Linear(160)\n+ Learned Scale Token Emb [0]', sub_size=6.9)
    box(22.5, 17.4, 22.5, 9.2, '#ccfbf1', '#0f766e',
        'Fine Branch (33\u00b3 @ 1 mm)',
        'Full Fine Field -> Conv3D + GN + SiLU\n3x SEResBlock3D -> GAP -> Linear(160)\n+ Learned Scale Token Emb [1]', sub_size=6.9)
    box(22.5, 5.8, 22.5, 9.2, '#fef3c7', '#b45309',
        'Context Branch (25\u00b3 @ ~2 mm)',
        '2x Downsampled -> Conv3D + GN + SiLU\n3x SEResBlock3D -> GAP -> Linear(160)\n+ Learned Scale Token Emb [2]', sub_size=6.9)

    for yc in (33.6, 22.0, 10.4):
        arrow(18.5, 22.5, 22.1, yc)

    # Stack + Cross-Scale Attention
    box(50.0, 12.0, 21.5, 20.5, '#ede9fe', '#6d28d9',
        'Cross-Scale Token\nSelf-Attention',
        'Stack 3 Scale Tokens [B, 3, 160]\n4-Head nn.MultiheadAttention\nResidual + LayerNorm + FFN (2x)\nFlatten -> [B, 480]\n(3 Tokens, NOT Voxel ViT)', title_size=8.3, sub_size=7.0)

    for yc in (33.6, 22.0, 10.4):
        arrow(45.5, yc, 49.5, 22.2)

    # Classifier & Inference Calibration
    box(76.5, 23.0, 21.5, 13.0, '#fee2e2', '#b91c1c',
        'Classification Head',
        'Linear(480->160) + LN + SiLU\nDropout(0.25) -> Linear(2 logits)\nLogit Diff: (z_1 - z_0)', title_size=8.2, sub_size=7.0)
    box(76.5, 6.0, 21.5, 13.0, '#dcfce7', '#15803d',
        'Frozen Calibration & TTA',
        '8-View TTA (z, y, x flips)\nInner-Fold Platt (T_0, b_0)\nFrozen F1 Cutoff tau*\nOutputs Ranked Probabilities', title_size=8.2, sub_size=7.0)

    arrow(72.0, 24.5, 76.0, 29.0)
    arrow(87.25, 22.5, 87.25, 19.5)

    # Bottom training recipe callout
    ax.text(50, 1.8, 'Arm E Training: Mixed Round-1 (6k) + Round-2 (6k) Hard Negatives (0.5 + 0.5*max) + 16k Random Negs | Softened Focal (\u03b3=1.5, ls=0.005), 26 ep', ha='center', va='center', fontsize=7.6, style='italic', color='#334155', bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8fafc', edgecolor='#cbd5e1'))
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig1_clear3d_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# FIGURE 2: Official LUNA16 FROC Curves
# ==========================================
def load_froc(path):
    fps, sens = [], []
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            try:
                fp, se = float(parts[0]), float(parts[1])
                fps.append(fp); sens.append(se)
            except ValueError:
                pass
    fps, sens = np.array(fps), np.array(sens)
    ord_i = np.argsort(fps, kind='mergesort')
    return fps[ord_i], sens[ord_i]

def make_fig2():
    curves = [
        ('Fold 0 Arm C: R1 Focal (70 scans, CPM=0.9063)', ROOT / 'runs/exp_fold0_focal_and_matched_inner_20261115_eval_cpu/official_cad_eval/arm_c_raw/froc_arm_c_fold0_predictions.txt', '#d97706', '-.', 1.9, '^'),
        ('Fold 0 Arm E: Mixed+Focal 8v (70 scans, CPM=0.9016)', ROOT / 'runs/exp_fold0_arm_e_8view_tta_20261116/official_cad_eval/arm_e_8view/froc_arm_e_fold0_8view_raw.txt', '#7c3aed', '--', 2.0, 's'),
        ('Archived 5-Model Ensemble (89 scans, CPM=0.9043)', ROOT / 'runs/clear3d-msf-official-test-fold0-001/froc_eval/cad/froc_clear3d_msf_test0.txt', '#1d4ed8', ':', 2.1, 'D'),
        ('HOLD Subset 0 Frozen Arm E-8v (89 scans, CPM=0.8750)', ROOT / 'runs/exp_hold_test_arm_e_8view_20261116/official_cad_eval/arm_e_hold_8view/froc_arm_e_hold_8view_raw.txt', '#dc2626', '-', 2.5, 'o'),
        ('dev708 10-Fold Improved OOF (708 scans, CPM=0.8680)', ROOT / 'runs/cv_improved_10fold_dev708_20261114/official_cad_eval/improved_pooled_dev708/froc_pooled_oof_predictions.txt', '#0f766e', '-', 2.3, 'v'),
        ('dev708 10-Fold Baseline OOF (708 scans, CPM=0.6821)', ROOT / 'runs/cv_improved_10fold_dev708_20261114/official_cad_eval/baseline_pooled_dev708/froc_baseline_dev708_pooled_oof_predictions.txt', '#64748b', '--', 1.8, 'x'),
    ]
    grid = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=300)
    for label, path, color, ls, lw, mk in curves:
        fps, sens = load_froc(path)
        mask = fps >= 0.04
        ax.plot(fps[mask], sens[mask], linestyle=ls, color=color, linewidth=lw, label=label, zorder=3)
        pts = np.interp(grid, fps, sens)
        ax.scatter(grid, pts, color=color, marker=mk, s=26, zorder=4, edgecolors='white', linewidths=0.4 if mk != 'x' else 0.8)
    for g in grid:
        ax.axvline(g, color='#cbd5e1', linestyle=':', linewidth=0.8, zorder=1)
    ax.set_xscale('log')
    ax.set_xlim(0.11, 8.8)
    ax.set_ylim(0.15, 1.03)
    ax.set_xticks(grid)
    ax.set_xticklabels(['0.125', '0.25', '0.5', '1', '2', '4', '8'])
    ax.set_xlabel('Average False Positives per Scan (log scale)', fontweight='bold')
    ax.set_ylabel('Nodule Detection Sensitivity', fontweight='bold')
    ax.set_title('Official LUNA16 Scan-Level FROC Curves Across Controlled Cohorts', fontweight='bold', pad=8)
    ax.grid(True, axis='y', linestyle='--', alpha=0.45)
    ax.legend(loc='lower right', fontsize=7.6, framealpha=0.96, edgecolor='#cbd5e1')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig2_froc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# FIGURE 3: PR Curves + Operating Point F1
# ==========================================
def load_y_score(path):
    ys, ps = [], []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            ys.append(int(r['label']))
            ps.append(float(r['probability']))
    return np.array(ys, dtype=np.int64), np.array(ps, dtype=np.float64)

def make_fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.5), dpi=300)
    pr_sets = [
        ('HOLD Subset 0 Arm E-8v (79,135 cands)', ROOT / 'runs/exp_hold_test_arm_e_8view_20261116/predictions/arm_e_hold_8view_raw.csv', '#dc2626', '-', 2.3),
        ('Fold 0 Arm E-8v (62,942 cands)', ROOT / 'runs/exp_fold0_arm_e_8view_tta_20261116/predictions/arm_e_fold0_8view_raw.csv', '#7c3aed', '-.', 1.9),
        ('dev708 10-Fold Improved (603,652 cands)', ROOT / 'runs/cv_improved_10fold_dev708_20261114/pooled_oof_predictions.csv', '#0f766e', '--', 2.0),
        ('dev708 10-Fold Baseline (603,652 cands)', ROOT / 'runs/cv_improved_10fold_dev708_20261114/baseline_dev708_pooled_oof_predictions.csv', '#64748b', ':', 1.9),
    ]
    for name, path, col, ls, lw in pr_sets:
        y, s = load_y_score(path)
        ap = average_precision_score(y, s)
        p, r, _ = precision_recall_curve(y, s)
        # Subsample step points safely while preserving visual precision
        idx = np.linspace(0, len(r)-1, min(len(r), 4000), dtype=int)
        ax1.plot(r[idx], p[idx], color=col, linestyle=ls, lw=lw, label=f'{name} | AP={ap:.4f}')

    # Highlight frozen HOLD tau* vs raw 0.5 vs Archived 5-model tau_0
    ax1.scatter([0.7609], [0.7500], color='#dc2626', s=65, marker='*', zorder=6, edgecolors='black', linewidths=0.6, label='HOLD Frozen tau* (Rec 76.1%, Prec 75.0%)')
    ax1.scatter([0.6667], [0.8440], color='#b91c1c', s=42, marker='o', zorder=6, edgecolors='black', linewidths=0.6, label='HOLD cal@0.5 (Rec 66.7%, Prec 84.4%)')
    ax1.scatter([0.8551], [0.5566], color='#1d4ed8', s=42, marker='D', zorder=6, edgecolors='black', linewidths=0.6, label='Arch. 5-Ens tau_0 (Rec 85.5%, Prec 55.7%)')
    ax1.set_xlim(0.0, 1.02)
    ax1.set_ylim(0.0, 1.03)
    ax1.set_xlabel('Candidate Recall (Sensitivity)', fontweight='bold')
    ax1.set_ylabel('Candidate Precision', fontweight='bold')
    ax1.set_title('(a) Natural-Prevalence Precision-Recall Curves', fontweight='bold', fontsize=9.5)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='lower left', fontsize=6.7, framealpha=0.95, edgecolor='#cbd5e1')

    # Panel B: Operating Point Comparison & Cross-Recipe Collapse
    cats = ['Cross-Xfer\n(Arm C)', 'HOLD\nraw@0.5', 'Arch. Ens.\n(tau_0)', 'Fold 0 E-8v\ncal@0.5', 'HOLD E-8v\ncal@0.5', 'HOLD E-8v\nFrozen tau*']
    prec = [0.0, 26.44, 55.66, 78.65, 84.40, 75.00]
    rec  = [0.0, 96.38, 85.51, 63.64, 66.67, 76.09]
    f1p  = [0.0, 41.50, 67.43, 70.35, 74.49, 75.54]
    x = np.arange(len(cats))
    w = 0.26
    b1 = ax2.bar(x - w, prec, w, color='#1d4ed8', edgecolor='#0f172a', linewidth=0.5, label='Precision (%)')
    b2 = ax2.bar(x,     rec,  w, color='#0f766e', edgecolor='#0f172a', linewidth=0.5, label='Recall (%)')
    b3 = ax2.bar(x + w, f1p,  w, color='#dc2626', edgecolor='#0f172a', linewidth=0.5, label='Positive F1 (%)')
    for bar in b3:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5, f'{h:.1f}' if h > 0 else '0.0', ha='center', va='bottom', fontsize=6.8, fontweight='bold', color='#991b1b')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, fontsize=7.4)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel('Candidate Metric (%)', fontweight='bold')
    ax2.set_title('(b) Calibration Transfer vs. Cross-Objective Failure', fontweight='bold', fontsize=9.5)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax2.legend(loc='upper left', fontsize=7.4, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig3_pr_and_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# FIGURE 4: dev708 Per-Fold 10-Fold Paired CPM
# ==========================================
def make_fig4():
    d = json.loads((ROOT / 'runs/cv_improved_10fold_dev708_20261114/results_summary.json').read_text())
    folds = [str(i) for i in range(10)]
    base_cpm = [d['per_fold'][f]['baseline_dev708']['cpm'] for f in folds]
    imp_cpm  = [d['per_fold'][f]['improved_dev708']['cpm'] for f in folds]
    deltas   = [d['per_fold'][f]['delta_cpm'] for f in folds]
    bm = d['fold_distribution']['baseline_mean_cpm']
    im = d['fold_distribution']['improved_mean_cpm']

    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=300)
    x = np.arange(10)
    w = 0.36
    ax.bar(x - w/2, base_cpm, w, color='#94a3b8', edgecolor='#334155', linewidth=0.7, label=f'Baseline OOF Recipe (Fold Mean {bm:.4f}, Pooled 0.6821)')
    bars = ax.bar(x + w/2, imp_cpm, w, color='#0f766e', edgecolor='#0f172a', linewidth=0.7, label=f'Prespecified Improved Recipe (Fold Mean {im:.4f}, Pooled 0.8680)')
    ax.axhline(bm, color='#475569', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.axhline(im, color='#0f766e', linestyle='-', linewidth=1.2, alpha=0.8)
    for i, dl in enumerate(deltas):
        top = max(base_cpm[i], imp_cpm[i])
        ax.text(x[i], top + 0.015, f'+{dl:.3f}', ha='center', va='bottom', fontsize=7.2, fontweight='bold', color='#0f766e')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds], fontsize=8.3)
    ax.set_ylim(0.0, 1.22)
    ax.set_ylabel('Official LUNA16 CPM', fontweight='bold')
    ax.set_title('Patient-Disjoint 10-Fold Cross-Validation on dev708 (708 Scans, 603,652 Candidates; 10/10 Folds Improved)', fontweight='bold', fontsize=9.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=7.6, framealpha=0.97, edgecolor='#cccccc')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_dev708_10fold_paired.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print('ALL_4_FIGURES_GENERATED_OK')