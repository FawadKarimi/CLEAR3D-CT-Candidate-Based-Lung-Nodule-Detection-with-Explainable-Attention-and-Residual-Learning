import json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXT_DIR = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline/external_validation_spie_aapm")

with open(EXT_DIR / "test_cohort_cap4000_corrected_evaluation_summary.json") as f:
    d = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

grid_rates = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

def plot_cohort_curves(ax, rule_data, title):
    cohorts = [
        ("Full Test Cohort (60 scans, 73 nodules)", rule_data["full_test_cohort"], "#1f77b4", "-"),
        ("Unexposed Test (58 scans, 70 nodules)", rule_data["unexposed_test_scans"], "#2ca02c", "--"),
        ("Technical Pilot (2 scans, 3 nodules)", rule_data["technical_pilot_scans"], "#ff7f0e", ":")
    ]
    for label, cdata, color, ls in cohorts:
        froc = cdata["froc_cpm_fixed_5mm"]
        sens = froc["sensitivities"]
        cpm = froc["cpm"]
        ci = froc.get("bootstrap_95ci", {}).get("ci_95", [None, None])
        ci_str = f" [95% CI: {ci[0]:.4f} - {ci[1]:.4f}]" if ci[0] is not None else ""
        ax.plot(grid_rates, sens, marker='o', linewidth=2.2, linestyle=ls, color=color,
                label=f"{label}\nCPM = {cpm:.4f}{ci_str}")
        
    ax.set_xscale('log', base=2)
    ax.set_xticks(grid_rates)
    ax.set_xticklabels([str(x) for x in grid_rates], fontsize=10)
    ax.set_xlabel("False Positives per Scan (Fixed 5.0 mm Matching)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Nodule Detection Sensitivity", fontsize=11, fontweight='bold')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

plot_cohort_curves(ax1, d["rules_compared"]["rule_A_closest_proposal"], 
                   "Rule A: Closest Proposal Matching (Original Rule)\n[Unexposed CPM: 0.4279 (95% CI: 0.3469 - 0.5193)]")

plot_cohort_curves(ax2, d["rules_compared"]["rule_B_highest_scoring_proposal"], 
                   "Rule B: Highest-Scoring Proposal Matching\n[Unexposed CPM: 0.4340 (95% CI: 0.3531 - 0.5216)]")

fig.suptitle("CLEAR3D-MSFv2 SPIE-AAPM Independent Test Evaluation (Cap=4,000)\nIndependent Rule Comparison & Corrected Patient-Clustered Bootstrap CIs", 
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_png = EXT_DIR / "froc_test_cohort_cap4000_corrected.png"
fig.savefig(out_png, dpi=300)
plt.close(fig)
print(f"Corrected FROC comparison plot saved to {out_png}", flush=True)
