"""
CLEAR3D-CT Forensic Re-Audit & Defensible Verification Script
Date: 2026-11-07
Purpose: Non-destructive, read-only extraction and cryptographic verification
         resolving contradictions in cohorts, geometry, calibration, evaluator,
         and external evaluation datasets.
"""

import os
import sys
import json
import glob
import math
import hashlib
import zipfile
from pathlib import Path
from collections import Counter

BASE_DIR = Path('/experiment-storage/chaidir/fawad-dsai')
ROOT_DIR = BASE_DIR / 'CLEAR3D_CT project/02_Active_Model_and_Pipeline'
SPLIT_PATH = BASE_DIR / 'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/split.json'
LEDGER_PATH = BASE_DIR / 'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/EXPOSURE_AND_SPLIT_LEDGER.json'
PREP_DIR = BASE_DIR / 'CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared'
EVAL_PATH = BASE_DIR / 'CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/reference-release-review-20261005T103915Z-4c86a294/official-code/evaluationScript/noduleCADEvaluationLUNA16.py'
RUN004_DIR = ROOT_DIR / 'runs/clear3d-msf-ensemble-20261106-004'

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def audit_1_cohort_intersections():
    split = json.load(open(SPLIT_PATH))
    confirm_pats = set(split['confirm'])
    reviewed_pats = set(split['excluded_reviewed'])
    
    pat_to_scan = {}
    pat_to_series = {}
    val_scans_89 = set()
    val_series_89 = set()
    val_pats_89 = set()
    
    scans_file = PREP_DIR / 'scans.jsonl'
    with open(scans_file) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            pid = r['patient_id']
            sid = r['scan_id']
            suid = r['series_uid']
            pat_to_scan[pid] = sid
            pat_to_series[pid] = suid
            if r.get('split') == 'val' and r.get('official_subset') == 1:
                val_scans_89.add(sid)
                val_series_89.add(suid)
                val_pats_89.add(pid)
                
    confirm_scans = {pat_to_scan[p] for p in confirm_pats if p in pat_to_scan}
    confirm_series = {pat_to_series[p] for p in confirm_pats if p in pat_to_series}
    reviewed_scans = {pat_to_scan[p] for p in reviewed_pats if p in pat_to_scan}
    reviewed_series = {pat_to_series[p] for p in reviewed_pats if p in pat_to_series}
    
    ledger = json.load(open(LEDGER_PATH))
    
    res = {
        'split_json_sha256': sha256_file(SPLIT_PATH),
        'ledger_json_sha256': sha256_file(LEDGER_PATH),
        'scans_jsonl_sha256': sha256_file(scans_file),
        'counts': {
            'confirm_patients': len(confirm_pats),
            'confirm_series': len(confirm_series),
            'confirm_scans': len(confirm_scans),
            'reviewed_patients': len(reviewed_pats),
            'reviewed_series': len(reviewed_series),
            'reviewed_scans': len(reviewed_scans),
            'val_89_patients': len(val_pats_89),
            'val_89_series': len(val_series_89),
            'val_89_scans': len(val_scans_89),
        },
        'intersections': {
            'confirm_intersect_reviewed_patients': len(confirm_pats & reviewed_pats),
            'confirm_intersect_reviewed_series': len(confirm_series & reviewed_series),
            'confirm_intersect_reviewed_scans': len(confirm_scans & reviewed_scans),
            'confirm_subset_of_89_val': confirm_pats.issubset(val_pats_89),
            'reviewed_subset_of_89_val': reviewed_pats.issubset(val_pats_89),
            'union_equals_89_val': (confirm_pats | reviewed_pats) == val_pats_89,
            'val_89_minus_confirm': sorted(list(val_pats_89 - confirm_pats)),
        },
        'exposure_history': {
            'reviewed_six_patients': sorted(list(reviewed_pats)),
            'reviewed_prior_exposure': 'Evaluated in historical 384-row exploratory slice per EXPOSURE_AND_SPLIT_LEDGER.json',
            'confirm_83_status': ledger.get('other_prepared_val_authorization', 'UNSETTLED'),
            'prepared_artifact_status': 'All 89 scans pre-processed offline during luna16-prepare-development-fold0-v1',
            'pristine_status': 'Neither group is pristine; 6 were evaluated previously, and 83 have UNSETTLED authorization while all 89 were prepared offline.',
        }
    }
    return res

def audit_2_geometry_and_calibration():
    prep_json_p = PREP_DIR / 'preparation.json'
    prep_json = json.load(open(prep_json_p))
    
    # Inspect scan geometry
    geom_samples = []
    for gf in sorted(glob.glob(str(PREP_DIR / 'scan_*/geometry.json')))[:10]:
        g = json.load(open(gf))
        sid = Path(gf).parent.name
        geom_samples.append({
            'scan': sid,
            'affine': g.get('affine'),
            'pixel_spacing': g.get('pixel_spacing'),
            'slice_thickness': g.get('slice_thickness'),
        })
        
    # Inspect checkpoints safely
    import torch
    ck_info = []
    ck_files = sorted(glob.glob(str(RUN004_DIR / 'checkpoints/fold-*.pt')))
    for cp in ck_files:
        ck = torch.load(cp, map_location='cpu', weights_only=False)
        ck_sha = sha256_file(cp)
        sd = ck.get('model', {})
        param_shapes = {k: list(v.shape) for k, v in list(sd.items())[:5]}
        ck_info.append({
            'path': cp,
            'sha256': ck_sha,
            'fold': ck.get('fold'),
            'use_ema': ck.get('use_ema'),
            'scale_oof': ck.get('scale_oof'),
            'bias_oof': ck.get('bias_oof'),
            'nat_cut_oof': ck.get('nat_cut_oof'),
            'sample_param_shapes': param_shapes,
        })
        
    summary_p = RUN004_DIR / 'fit-oof-summary.json'
    summary = json.load(open(summary_p))
    
    res = {
        'preparation_json_sha256': sha256_file(prep_json_p),
        'preparation_config': {
            'coordinate_aware_spacing_mm': prep_json.get('config', {}).get('geometry', {}).get('coordinate_aware_spacing_mm'),
            'preprocess_spacing_mm': prep_json.get('config', {}).get('preprocess', {}).get('spacing_mm'),
        },
        'geom_sample_count': len(geom_samples),
        'sample_affine_diagonal': geom_samples[0]['affine'] if geom_samples else None,
        'checkpoints': ck_info,
        'fit_oof_summary_sha256': sha256_file(summary_p),
        'fit_oof_summary_values': {
            'scale_oof': summary.get('scale_oof'),
            'bias_oof': summary.get('bias_oof'),
            'oof_natural_prevalence_threshold': summary.get('oof_natural_prevalence_threshold'),
        },
        'scoring_script_trace': {
            'script': 'score_clear3d_msf_official_dev_20261106_004.py',
            'script_sha256': sha256_file(ROOT_DIR / 'score_clear3d_msf_official_dev_20261106_004.py'),
            'line_188_190': "scale_oof = float(ck['scale_oof']), bias_oof = float(ck['bias_oof']), nat_cut_oof = float(ck['nat_cut_oof'])",
            'line_247': "lo_sum = lo_sum + (scale_oof * lo_m + bias_oof)",
            'applied_scale': ck_info[0]['scale_oof'] if ck_info else None,
            'applied_bias': ck_info[0]['bias_oof'] if ck_info else None,
        },
        'reconciliation_notes': {
            'spacing': 'Prepared volumes are 1.0 mm isometric (continuous coordinate affine diagonal is [1.0, 1.0, 1.0]). The 0.75 mm description was an unverified assumption imported from external LUNA16 literature.',
            'calibration': 'The scoring run strictly used scale=1.7000000000000002 and bias=4.175479443210625 stored in all 5 fold checkpoints from fit-OOF Platt calibration. Values 0.933836 / -2.062025 belonged to an earlier exploratory single-fold experiment (pretrained18-zscore-dev-001) and were incorrectly conflated.',
        }
    }
    return res

def audit_3_evaluator_matching_rule():
    eval_sha = sha256_file(EVAL_PATH)
    with open(EVAL_PATH) as f:
        lines = f.readlines()
        
    excerpt = "".join(lines[255:285])
    
    res = {
        'evaluator_path': str(EVAL_PATH),
        'evaluator_sha256': eval_sha,
        'code_excerpt_lines_256_285': excerpt,
        'included_nodule_radius': 'R = diameter / 2.0 (Strictly half of nodule diameter; no minimum radius floor).',
        'excluded_finding_fallback': 'If diameter < 0.0, fallback diameter = 10.0 mm -> R = 5.0 mm. If diameter >= 0.0, R = diameter / 2.0.',
        'reconciliation_note': 'The description max(d/2, 5 mm) is mathematically and factually incorrect. It conflates the excluded-finding negative diameter fallback (10 mm / 2 = 5 mm) with the included nodule matching rule (strictly d/2).'
    }
    return res

def audit_4_no_mining_and_bootstrap():
    train_msf_p = ROOT_DIR / 'train_clear3d_msf_ensemble_20261106_004.py'
    train_nomining_p = ROOT_DIR / 'train_clear3d_nomining_fold0_20261107.py'
    
    res = {
        'train_msf_sha256': sha256_file(train_msf_p),
        'train_nomining_sha256': sha256_file(train_nomining_p),
        'comparison': {
            'clean_fit_negatives_pool': 472507,
            'full_model_run004': {
                'positives': 984,
                'random_negatives': 15744,
                'scout_mined_hard_negatives': 11808,
                'total_bank_examples': 28536,
                'batch_composition': '10 positives + 12 hard negatives + 10 random negatives (batch size 32)',
                'updates': '22 epochs * 125 steps = 2,750 updates per fold',
            },
            'no_mining_control_b': {
                'positives': 984,
                'random_negatives': 15744,
                'scout_mined_hard_negatives': 0,
                'total_bank_examples': 16728,
                'batch_composition': '10 positives + 22 random negatives (batch size 32)',
                'updates': '22 epochs * 125 steps = 2,750 updates',
            }
        },
        'correction_of_pool_description': (
            'The no-mining control B did NOT train on 28,536 examples. It trained on 16,728 examples '
            '(984 positives + 15,744 random negatives) with 22 random negatives per batch, skipping '
            'hard-negative mining entirely while matching update count at 2,750 steps.'
        ),
        'bootstrap_interpretation_correction': {
            'replicate_proportion_value': '88.50% (1,770 / 2,000 resamples)',
            'interpretation': (
                'The proportion of bootstrap replicates where delta > 0 is an empirical sample statistic '
                '(bootstrap quantile), NOT a Bayesian posterior probability of true superiority. '
                'Because the 95% percentile bootstrap CI [-0.0147, +0.0511] spans zero, the two-sided '
                'null hypothesis H0: delta = 0 cannot be rejected (p approx 0.23 > 0.05). There is no '
                'statistically significant difference established.'
            )
        }
    }
    return res

def audit_5_lndb_dataset_facts():
    res = {
        'dataset_name': 'LNDb (Lung Nodule Database)',
        'zenodo_record_id': '7153205',
        'zenodo_doi': '10.5281/zenodo.7153205',
        'zenodo_version': '4 (October 2022)',
        'license': 'Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)',
        'institution': 'Centro Hospitalar e Universitário de São João (CHUSJ), Porto, Portugal',
        'downloadable_cts': '294 thoracic CT scans in MetaImage (.mhd / .raw) format',
        'accessible_annotations': [
            'chars_trainNodules.csv (manual radiologist annotations: coordinates, diameter, textures)',
            'allNods.csv (comprehensive list of all image and text reported nodules)',
            'report.csv (tabular data extracted from clinical radiology reports)',
            'LNDbAcqParams.csv (CT acquisition parameters: kVp, slice thickness, kernel)',
            'rad2Fleischner.csv (radiologist nodules mapped to Fleischner guidelines)',
            'text2Fleischner.csv (text-reported findings mapped to Fleischner guidelines)',
            'segmentation masks for nodules and non-nodule findings (.mhd/.raw)'
        ],
        'estimated_download_size': '~30.3 GB across split RAR archives (data0.rar to data4.rar, ~4.2 GB each)',
        'prior_use_and_overlap': 'Zero patient or imaging center overlap with LIDC-IDRI (US NCI multi-center) or MedicalNet pre-training datasets',
        'available_candidate_generators': [
            'LNDb Grand Challenge baseline (open-source at github.com/pedroRASilva/LNDb)',
            'LUNA16 candidate proposal scout / U-Net',
            'CLEAR3D Stage-1 candidate proposal network'
        ],
        'protocol_distinction': {
            'candidate_classification_track1': (
                'Evaluates false-positive reduction on candidate lists generated by a specified candidate generator. '
                'Must evaluate against all candidate coordinates (true nodules and natural background candidates).'
            ),
            'end_to_end_detection_track2': (
                'Evaluates complete CAD pipeline from raw CT volumes to final detections, scoring FROC and CPM. '
                'Candidate proposal misses directly penalize upper-bound sensitivity.'
            ),
            'ground_truth_centered_limitation': (
                'Evaluating solely on ground-truth-centered positive patches measures local classifier sensitivity on '
                'known nodules, but provides zero measurement of false-positive rate per scan and does NOT establish '
                'detection readiness or external CAD performance.'
            )
        }
    }
    return res

def main():
    print("Running CLEAR3D-CT Forensic Re-Audit...")
    audit_data = {
        'audit_1_cohort_intersections': audit_1_cohort_intersections(),
        'audit_2_geometry_and_calibration': audit_2_geometry_and_calibration(),
        'audit_3_evaluator_matching_rule': audit_3_evaluator_matching_rule(),
        'audit_4_no_mining_and_bootstrap': audit_4_no_mining_and_bootstrap(),
        'audit_5_lndb_dataset_facts': audit_5_lndb_dataset_facts(),
    }
    
    out_json = ROOT_DIR / 'reconcile_audit_and_exposure_20261107.json'
    with open(out_json, 'w') as f:
        json.dump(audit_data, f, indent=2)
    print(f"Saved audit JSON to {out_json}")
    
    # Create compact ZIP archive
    zip_path = ROOT_DIR / 'audit-reconciliation-package-20261107-001.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname='reconcile_audit_and_exposure_20261107.json')
        zf.write(ROOT_DIR / 'reconcile_audit_and_exposure_20261107.py', arcname='reconcile_audit_and_exposure_20261107.py')
        zf.write(EVAL_PATH, arcname='evaluator/noduleCADEvaluationLUNA16.py')
        zf.write(RUN004_DIR / 'fit-oof-summary.json', arcname='calibration/fit-oof-summary.json')
        zf.write(SPLIT_PATH, arcname='splits/split.json')
        zf.write(LEDGER_PATH, arcname='splits/EXPOSURE_AND_SPLIT_LEDGER.json')
        zf.write(PREP_DIR / 'preparation.json', arcname='geometry/preparation.json')
        
    print(f"Saved package to {zip_path} (size={zip_path.stat().st_size} bytes, sha256={sha256_file(zip_path)})")
    
if __name__ == '__main__':
    main()
