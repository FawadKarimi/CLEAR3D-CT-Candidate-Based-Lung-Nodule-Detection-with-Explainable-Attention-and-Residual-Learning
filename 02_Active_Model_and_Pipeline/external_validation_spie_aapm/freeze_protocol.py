import hashlib, json
from pathlib import Path

BASE = Path('/experiment-storage/chaidir/fawad-dsai')
ROOT = BASE / 'CLEAR3D_CT project/02_Active_Model_and_Pipeline'
EXT_DIR = ROOT / 'external_validation_spie_aapm'

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

ck_hashes = {}
for i in range(5):
    p = ROOT / f'runs/clear3d-msf-ensemble-20261106-004/checkpoints/fold-{i}.pt'
    ck_hashes[f'fold-{i}'] = {'path': str(p.relative_to(ROOT)), 'sha256': sha256_file(p)}

model_file = ROOT / 'clear3d_msf_model_20261106_002.py'
model_sha = sha256_file(model_file)

protocol = {
    'protocol_name': 'CLEAR3D-MSFv2 SPIE-AAPM Independent Test Evaluation (Cap=4000)',
    'timestamp': '2026-11-08T18:55:00Z',
    'status': 'FROZEN_BEFORE_EVALUATION',
    'selection_basis': {
        'candidate_cap': 4000,
        'selection_cohort': 'Permitted 142 LUNA16 Development Scans (runs/generatedneg-fullscan-dev-cap4000-20261102-001)',
        'dev_coverage_measurement': '92.19% nodule sensitivity (177/192 nodules) at cap=4000',
        'external_tuning': False
    },
    'frozen_system': {
        'architecture': 'CLEAR3DMSFv2',
        'model_code': {'path': 'clear3d_msf_model_20261106_002.py', 'sha256': model_sha},
        'checkpoints': ck_hashes,
        'inference': {
            'tta': '4-view (Identity, Sagittal_flip, CoronalAxial_flip, XYZ_flip)',
            'aggregation': 'pre_sigmoid_logit_average_across_5_folds',
            'precision': 'fp16_autocast_cuda1',
            'batch_size': 64
        },
        'calibration': {
            'method': 'Platt_scaling_and_natural_prevalence_thresholding',
            'source': '100% fit-only 5-fold patient-disjoint OOF (zero dev access)',
            'scale_oof': 1.7000000000000002,
            'bias_oof': 4.175479443210625,
            'nat_cut_oof': 0.9999467134475708,
            'cut_half': 0.5
        }
    },
    'candidate_generator': {
        'algorithm': 'Multiscale Laplacian of Gaussian (LoG) 3D Blob Detector with physical NMS',
        'code': 'lidc3d/src/lidc3d/candidates.py',
        'sigmas_mm': [1.0, 1.5, 2.0, 3.0],
        'threshold': 0.005,
        'nms_mm': 3.0,
        'cap': 4000
    },
    'matching_protocol': {
        'rule': 'Fixed Euclidean distance < 5.0 mm',
        'tie_breaking': 'Closest candidate assigned as True Positive; duplicate hits (< 5mm) ignored',
        'evaluation_type': 'NON_OFFICIAL_FIXED_5MM_MATCHING',
        'distinction_from_official_luna16': 'LUNA16 official uses max(radius, 5.0 mm); SPIE-AAPM LUNGx ground truth provides only center coordinates without lesion diameters, necessitating a fixed 5.0 mm radius.'
    },
    'data_exposure_accounting': {
        'total_scans': 60,
        'total_nodules': 73,
        'technical_pilot_scans': ['LUNGx-CT001', 'LUNGx-CT002'],
        'technical_pilot_nodules': 3,
        'unexposed_scans': [f'LUNGx-CT{i:03d}' for i in range(3, 61)],
        'unexposed_scan_count': 58,
        'unexposed_nodule_count': 70
    },
    'governance': {
        'reference_release': 'HOLD',
        'confirmation_cohort': '83 patients strictly restricted / untouched',
        'internal_test_fold0': 'Untouched / zero retraining or tuning',
        'manuscript_writing': 'DEFERRED'
    }
}

out_p = EXT_DIR / 'frozen_evaluation_protocol_cap4000.json'
with open(out_p, 'w') as f:
    json.dump(protocol, f, indent=2)

h = sha256_file(out_p)
print(f"Protocol saved to {out_p} (SHA256: {h})")
