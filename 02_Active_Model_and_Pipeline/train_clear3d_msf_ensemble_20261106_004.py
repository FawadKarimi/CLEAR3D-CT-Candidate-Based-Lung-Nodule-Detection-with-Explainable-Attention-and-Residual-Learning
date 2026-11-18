"""CLEAR3D-MSF v2 (Run 004): Deterministic AMP Tensor-Core + GPU-Downsampled 5-Fold Fit-Only Ensemble
with Full Fit-Pool (472k) Hard-Negative Mining & Thermal Duty Pacing for Shared RTX 2080 Ti (GPU 1).
Strictly adheres to all scientific guards:
- Fit-only training, mining, OOF model selection, and OOF Platt/prior/threshold calibration (0 dev touch).
- Excludes LIDC-IDRI-0138 and LIDC-IDRI-0029, and filters annotations_excluded.csv (+5mm margin) and near-included (+5mm) border negatives.
- Monitored single-process GPU worker via user_watchdog; bit-deterministic AMP CUDA ops; reference_release HOLD.
"""
from __future__ import annotations
import argparse, copy, csv, ctypes, hashlib, json, math, os, random, signal, sys, time
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('/experiment-storage/chaidir/fawad-dsai')
ROOT = BASE / 'CLEAR3D_CT project/02_Active_Model_and_Pipeline'
WORK = BASE / 'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/source'
sys.path[:0] = [str(ROOT), str(WORK / 'src'), str(WORK / 'scripts')]

import numpy as np
import user_watchdog as w
import loader

G = 1024 ** 3
LIMITS = {
    'setup_seconds': 600,
    'run_seconds': 18000,
    'validation_seconds': 3600,
    'total_seconds': 19800,
    'rss_bytes': 8 * G,
    'allocator_bytes': 4 * G,
    'gpu_bytes': 6 * G,
    'sample_seconds': 2.0,
    'min_ram_bytes': 32 * G,
    'reserve_ram_bytes': 8 * G,
    'min_gpu_bytes': 10 * G,
    'min_disk_bytes': 20 * G,
}

SELECT = ROOT / 'selection-001.json'
SELECTION_SHA = '8ab4cda286f59a69f942527a80fb6a97ed4271f9e4c902aa9ed06a9c50334001'
BAL_PATH = ROOT / 'balanced-candidate-classification-20261106-001.json'
BAL_SHA = '3e485ec412da5543999673b8cab25ca547af64efc35c93d0a62cf1fdbe012401'
SPLIT_PATH = BASE / 'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/split.json'
FIT_JSONL = BASE / 'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/fit.jsonl'
FIT_JSONL_SHA = '3207691fc556a871258b181e6366e2e8743cc09edbac601de73c2fcc321c3d76'
REFD = BASE / 'data/reference/luna16-zenodo-3723295/resolved-reference-v1'
EXC_SHA = '73a5c139257294fe45a68620ed1b2cf383ba8b22af1501d2f498abbd6b802c25'
ANN_SHA = 'db9adb75b381f3e95b1f601d9b63f2e42bd6536ab5a06dc539c5ec8a4005cec7'
BIND = BASE / 'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/D-long/current-fit-binding.json'
PREP = BASE / 'CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared'
MODEL_FILE = ROOT / 'clear3d_msf_model_20261106_002.py'
MODEL_SHA = 'e83899960b095eab9f92fe104f13e3380c9c5bc48e6ebc82b173003a520914e7'
OUT = ROOT / 'runs/clear3d-msf-ensemble-20261106-004'

N_FOLDS = 5
RANDOM_NEG_COUNT = 15744   # 16x positives IID uniform across clean fit negatives
HARD_NEG_COUNT = 11808     # 12x positives (~top 2.5% of 472,507 clean fit negatives)
SCOUT_EPOCHS = 12
SCOUT_STEPS = 115
FOLD_EPOCHS = 22
FOLD_STEPS = 125
BATCH_SIZE = 32
STEP_COOL_SLEEP = 0.035    # Keeps shared RTX 2080 Ti (GPU 1) below 89C thermal throttle


def require(ok: bool, msg: str):
    if not ok:
        raise ValueError(msg)


def verify_inputs():
    require(loader.hash_file(SELECT) == SELECTION_SHA, 'selection-001 changed')
    require(loader.hash_file(BAL_PATH) == BAL_SHA, 'balanced selection changed')
    require(loader.hash_file(FIT_JSONL) == FIT_JSONL_SHA, 'fit.jsonl changed')
    require(loader.hash_file(REFD / 'annotations_excluded.csv') == EXC_SHA, 'excluded annotations changed')
    require(loader.hash_file(REFD / 'annotations.csv') == ANN_SHA, 'included annotations changed')
    require(loader.hash_file(MODEL_FILE) == MODEL_SHA, 'model file changed')
    sel = json.loads(SELECT.read_text())
    require(sel['reference_release'] == 'HOLD', 'reference_release must be HOLD')
    split = loader.get_split(SPLIT_PATH)
    for role, expected in [('fit', 3936), ('dev', 1060)]:
        rows = sel[role]['selected']
        require(len(rows) == expected, f'{role} count mismatch')
        require(len({r['id'] for r in rows}) == len(rows), f'{role} duplicate id')
        require(all(r['patient'] in split[role] and r['patient'] not in loader.EXCLUDED for r in rows), f'{role} split violation')
    require(not (split['fit'] & split['dev']), 'fit/dev split overlap')
    return sel, split


def downsample_55_to_27(img55: np.ndarray, val55: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separable [0.25, 0.5, 0.25] stride-2 valid-masked downsample: 55^3 -> 27^3 (center 27 -> 13)."""
    w = img55 * val55.astype(np.float32)
    m = val55.astype(np.float32)
    for ax in (0, 1, 2):
        s0 = [slice(None)] * 3; s0[ax] = slice(0, -2, 2)
        s1 = [slice(None)] * 3; s1[ax] = slice(1, -1, 2)
        s2 = [slice(None)] * 3; s2[ax] = slice(2, None, 2)
        w = 0.25 * w[tuple(s0)] + 0.5 * w[tuple(s1)] + 0.25 * w[tuple(s2)]
        m = 0.25 * m[tuple(s0)] + 0.5 * m[tuple(s1)] + 0.25 * m[tuple(s2)]
    cval = m >= 0.5
    cimg = np.where(cval, w / np.maximum(m, 1e-6), 0.0)
    return np.clip(cimg, 0.0, 1.0), cval


def gpu_extract_from_p51(p51, v51):
    """Exact zero-diff GPU equivalent of extract_eval_pair on (B, 51, 51, 51) float32 tensors."""
    import torch
    f_val = v51[:, 9:42, 9:42, 9:42]
    f_img = torch.where(f_val > 0.5, p51[:, 9:42, 9:42, 9:42], torch.zeros_like(f_val))
    w_t = p51 * v51
    m_t = v51
    for ax in (1, 2, 3):
        s0 = [slice(None)] * 4; s0[ax] = slice(0, -2, 2)
        s1 = [slice(None)] * 4; s1[ax] = slice(1, -1, 2)
        s2 = [slice(None)] * 4; s2[ax] = slice(2, None, 2)
        w_t = 0.25 * w_t[tuple(s0)] + 0.5 * w_t[tuple(s1)] + 0.25 * w_t[tuple(s2)]
        m_t = 0.25 * m_t[tuple(s0)] + 0.5 * m_t[tuple(s1)] + 0.25 * m_t[tuple(s2)]
    cval = (m_t >= 0.5).float()
    cimg = torch.where(m_t >= 0.5, w_t / torch.clamp_min(m_t, 1e-6), torch.zeros_like(w_t)).clamp(0.0, 1.0)
    return f_img, f_val, cimg, cval


def slice_p51_batch(img_vol: np.ndarray, val_vol: np.ndarray, p_img: np.ndarray, p_val: np.ndarray, zyxs: np.ndarray, pad: int = 28):
    from lidc3d.geometry import crop
    n = len(zyxs)
    p51_b = np.empty((n, 51, 51, 51), dtype=np.float32)
    v51_b = np.empty((n, 51, 51, 51), dtype=np.float32)
    centers = np.floor(zyxs + 0.5).astype(int) + pad
    D, H, W = p_img.shape
    for i in range(n):
        c = centers[i]
        s0, s1, s2 = int(c[0] - 25), int(c[1] - 25), int(c[2] - 25)
        e0, e1, e2 = s0 + 51, s1 + 51, s2 + 51
        if s0 >= 0 and s1 >= 0 and s2 >= 0 and e0 <= D and e1 <= H and e2 <= W:
            p51_b[i] = p_img[s0:e0, s1:e1, s2:e2]
            v51_b[i] = p_val[s0:e0, s1:e1, s2:e2]
        else:
            p55, _ = crop(img_vol, zyxs[i], 55, 0)
            v55, _ = crop(val_vol, zyxs[i], 55, False)
            p51_b[i] = p55[2:53, 2:53, 2:53]
            v51_b[i] = v55[2:53, 2:53, 2:53]
    return p51_b, v51_b


def pack_u16(img: np.ndarray, val: np.ndarray) -> np.ndarray:
    q = np.clip(np.rint(img * 32767.0), 0.0, 32767.0).astype(np.uint16)
    return q | (val.astype(np.uint16) << np.uint16(15))


def unpack_u16(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    val = (u >> np.uint16(15)).astype(np.float32)
    img = (u & np.uint16(0x7FFF)).astype(np.float32) * (1.0 / 32767.0)
    return img, val


def extract_pair_u16(img_vol: np.ndarray, val_vol: np.ndarray, zyx) -> tuple[np.ndarray, np.ndarray]:
    from lidc3d.geometry import crop
    p55, _ = crop(img_vol, zyx, 55, 0)
    v55, _ = crop(val_vol, zyx, 55, False)
    f37 = pack_u16(p55[9:46, 9:46, 9:46], v55[9:46, 9:46, 9:46])
    cimg27, cval27 = downsample_55_to_27(p55, v55)
    c27 = pack_u16(cimg27, cval27)
    return f37, c27


def extract_eval_pair(img_vol: np.ndarray, val_vol: np.ndarray, zyx) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from lidc3d.geometry import crop
    p55, _ = crop(img_vol, zyx, 55, 0)
    v55, _ = crop(val_vol, zyx, 55, False)
    f_img = np.where(v55[11:44, 11:44, 11:44], p55[11:44, 11:44, 11:44], 0.0).astype(np.float32)
    f_val = v55[11:44, 11:44, 11:44].astype(np.float32)
    cimg27, cval27 = downsample_55_to_27(p55, v55)
    c_img = cimg27[1:26, 1:26, 1:26].astype(np.float32)
    c_val = cval27[1:26, 1:26, 1:26].astype(np.float32)
    return f_img, f_val, c_img, c_val


def load_scan_arrays(scan_id: str, expected_patient: str, expected_series: str, use_mmap: bool = False):
    d = (PREP / scan_id)
    require(not d.is_symlink() and d.resolve().parent == PREP.resolve(), 'invalid scan path')
    rec = json.loads((d / 'record.json').read_text())
    require(rec['patient_id'] == expected_patient and rec['series_uid'] == expected_series, 'scan identity mismatch')
    geo = json.loads((d / 'geometry.json').read_text())
    aff = np.asarray(geo['affine'], dtype=np.float64)
    inv = np.linalg.inv(aff)
    mm = 'r' if use_mmap else None
    img = np.load(d / 'image.npy', mmap_mode=mm, allow_pickle=False)
    val = np.load(d / 'valid.npy', mmap_mode=mm, allow_pickle=False)
    require(img.ndim == 3 and img.shape == val.shape, 'volume shape mismatch')
    return img, val, inv


def roc_auc_np(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    vals = scores[order]
    i = 0
    n = len(vals)
    while i < n:
        j = i + 1
        while j < n and vals[j] == vals[i]:
            j += 1
        if j > i + 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def fit_platt_balanced(oof_log_odds: np.ndarray, oof_labels: np.ndarray) -> tuple[float, float]:
    zp = oof_log_odds[oof_labels == 1].astype(np.float64)
    zn = oof_log_odds[oof_labels == 0].astype(np.float64)

    best_bce_obj = float('inf')
    best_a = 1.0
    best_b_bce = 0.0

    for a in np.linspace(0.50, 2.20, 35):
        azp = a * zp
        azn = a * zn
        lo, hi = -10.0, 10.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            sp = 1.0 / (1.0 + np.exp(np.clip(azp + mid, -40.0, 40.0)))
            sn = 1.0 / (1.0 + np.exp(np.clip(-(azn + mid), -40.0, 40.0)))
            grad = 0.5 * (-sp.mean() + sn.mean())
            if grad < 0.0:
                lo = mid
            else:
                hi = mid
        b_cur = 0.5 * (lo + hi)
        bce = 0.5 * (np.logaddexp(0.0, -(azp + b_cur)).mean() + np.logaddexp(0.0, azn + b_cur).mean())
        obj = bce + 0.002 * ((a - 1.0) ** 2)
        if obj < best_bce_obj:
            best_bce_obj = obj
            best_a = float(a)
            best_b_bce = float(b_cur)

    azp = best_a * zp
    azn = best_a * zn
    tau = 0.25
    best_ref_obj = float('inf')
    best_b = best_b_bce
    for b in np.linspace(best_b_bce - 1.5, best_b_bce + 1.5, 1201):
        up = azp + b
        un = azn + b
        bce = 0.5 * (np.logaddexp(0.0, -up).mean() + np.logaddexp(0.0, un).mean())
        s01 = 0.5 * (
            (1.0 / (1.0 + np.exp(np.clip(up / tau, -40.0, 40.0)))).mean()
            + (1.0 / (1.0 + np.exp(np.clip(-un / tau, -40.0, 40.0)))).mean()
        )
        obj = 0.35 * bce + 0.65 * s01
        if obj < best_ref_obj:
            best_ref_obj = obj
            best_b = float(b)

    return best_a, best_b


def build_clean_fit_pool(split_fit: set[str]):
    exc_refs = defaultdict(list)
    with (REFD / 'annotations_excluded.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            d = float(r['diameter_mm'])
            exc_refs[r['seriesuid']].append((
                [float(r[k]) for k in ('coordX', 'coordY', 'coordZ')],
                (d / 2.0 if d > 0.0 else 3.0) + 5.0,
            ))
    inc_refs = defaultdict(list)
    with (REFD / 'annotations.csv').open(newline='') as f:
        for idx, r in enumerate(csv.DictReader(f)):
            d = float(r['diameter_mm'])
            inc_refs[r['seriesuid']].append((
                idx,
                [float(r[k]) for k in ('coordX', 'coordY', 'coordZ')],
                d,
            ))

    raw_rows = loader.candidate_rows(FIT_JSONL, 'fit', split_fit)
    filtered, dropped_exc = loader.filter_excluded(raw_rows, exc_refs)

    pos_rows = []
    neg_rows = []
    dropped_near_inc = 0
    for r in filtered:
        if r['label'] == 1:
            pos_rows.append(r)
        else:
            regs = inc_refs.get(r['series'], ())
            if regs:
                c = (r['x'], r['y'], r['z'])
                if any(math.dist(c, ac) <= (ad / 2.0 + 5.0) for _, ac, ad in regs):
                    dropped_near_inc += 1
                    continue
            neg_rows.append(r)

    hit_counts = Counter()
    matched_info = []
    for r in pos_rows:
        c = (r['x'], r['y'], r['z'])
        best = min((math.dist(c, ac), aidx, ad) for aidx, ac, ad in inc_refs[r['series']])
        hit_counts[best[1]] += 1
        matched_info.append((best[1], best[2]))
    pos_diams = np.array([d for _, d in matched_info], dtype=np.float64)
    pos_weights = np.array([
        (1.25 if d < 6.0 else 1.0) / math.sqrt(hit_counts[aidx])
        for aidx, d in matched_info
    ], dtype=np.float64)

    pat_pos = Counter()
    all_fit_pats = sorted(split_fit)
    for r in pos_rows:
        pat_pos[r['patient']] += 1
    rng = random.Random(17)
    by_bucket = defaultdict(list)
    for p in all_fit_pats:
        by_bucket[min(pat_pos[p], 4)].append(p)
    pat_fold = {}
    cursor = 0
    for b in sorted(by_bucket):
        plist = by_bucket[b]
        rng.shuffle(plist)
        for p in plist:
            pat_fold[p] = cursor % N_FOLDS
            cursor += 1

    rng_neg = random.Random(20261106)
    picked_rand = sorted(rng_neg.sample(range(len(neg_rows)), RANDOM_NEG_COUNT))
    rand_neg_set = set(picked_rand)
    rand_neg_rows = [neg_rows[i] for i in picked_rand]

    meta = {
        'raw_fit_rows': len(raw_rows),
        'dropped_excluded_5mm': dropped_exc,
        'dropped_near_included_5mm': dropped_near_inc,
        'fit_positives': len(pos_rows),
        'fit_positives_lt6mm': int((pos_diams < 6.0).sum()),
        'fit_positives_ge6mm': int((pos_diams >= 6.0).sum()),
        'fit_clean_negatives': len(neg_rows),
        'random_negatives': len(rand_neg_rows),
        'unique_fit_nodules': len(hit_counts),
    }
    return pos_rows, pos_diams, pos_weights, neg_rows, rand_neg_rows, rand_neg_set, pat_fold, meta


def sample_augmented_batch(f37_bank: np.ndarray, c27_bank: np.ndarray, indices: np.ndarray, rng: np.random.Generator):
    b = len(indices)
    fi = np.empty((b, 33, 33, 33), dtype=np.float32)
    fv = np.empty((b, 33, 33, 33), dtype=np.float32)
    ci = np.empty((b, 25, 25, 25), dtype=np.float32)
    cv = np.empty((b, 25, 25, 25), dtype=np.float32)

    do_shift = rng.random(b) < 0.50
    dzs = np.where(do_shift, rng.integers(-1, 2, size=b), 0)
    dys = np.where(do_shift, rng.integers(-1, 2, size=b), 0)
    dxs = np.where(do_shift, rng.integers(-1, 2, size=b), 0)
    ks = rng.integers(0, 4, size=b)
    flip_z = rng.random(b) < 0.5
    flip_x = rng.random(b) < 0.5
    do_int = rng.random(b) < 0.5
    scales = rng.uniform(0.95, 1.05, size=b).astype(np.float32)
    shifts = rng.uniform(-0.02, 0.02, size=b).astype(np.float32)

    for j, idx in enumerate(indices):
        dz, dy, dx = int(dzs[j]), int(dys[j]), int(dxs[j])
        f_im, f_va = unpack_u16(f37_bank[idx, 2 + dz:35 + dz, 2 + dy:35 + dy, 2 + dx:35 + dx])
        c_im, c_va = unpack_u16(c27_bank[idx, 1:26, 1:26, 1:26])

        k = int(ks[j])
        if k:
            f_im = np.rot90(f_im, k, (1, 2)); f_va = np.rot90(f_va, k, (1, 2))
            c_im = np.rot90(c_im, k, (1, 2)); c_va = np.rot90(c_va, k, (1, 2))
        if flip_z[j]:
            f_im = f_im[::-1]; f_va = f_va[::-1]
            c_im = c_im[::-1]; c_va = c_va[::-1]
        if flip_x[j]:
            f_im = f_im[:, :, ::-1]; f_va = f_va[:, :, ::-1]
            c_im = c_im[:, :, ::-1]; c_va = c_va[:, :, ::-1]
        if do_int[j]:
            f_im = np.where(f_va > 0.5, np.clip(f_im * scales[j] + shifts[j], 0.0, 1.0), 0.0)
            c_im = np.where(c_va > 0.5, np.clip(c_im * scales[j] + shifts[j], 0.0, 1.0), 0.0)

        fi[j] = f_im; fv[j] = f_va
        ci[j] = c_im; cv[j] = c_va
    return fi, fv, ci, cv


def eval_bank_tta8(model, f37_bank: np.ndarray, c27_bank: np.ndarray, indices: np.ndarray, device, f_fine, f_ctx) -> np.ndarray:
    import torch
    from clear3d_msf_model_20261106_002 import make_inputs, log_odds_tta8
    model.eval()
    out = np.empty(len(indices), dtype=np.float32)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        for s in range(0, len(indices), 64):
            sub = indices[s:s + 64]
            f_im, f_va = unpack_u16(f37_bank[sub, 2:35, 2:35, 2:35])
            c_im, c_va = unpack_u16(c27_bank[sub, 1:26, 1:26, 1:26])
            vf, vc = make_inputs(
                torch.from_numpy(np.ascontiguousarray(f_im)).to(device),
                torch.from_numpy(np.ascontiguousarray(f_va)).to(device),
                torch.from_numpy(np.ascontiguousarray(c_im)).to(device),
                torch.from_numpy(np.ascontiguousarray(c_va)).to(device),
                f_fine, f_ctx,
            )
            out[s:s + len(sub)] = log_odds_tta8(model, vf, vc).cpu().numpy()
    return out


def work(smoke: bool = False):
    import torch
    import torch.nn.functional as F
    from lidc3d.provenance import seed_all
    from clear3d_msf_model_20261106_002 import CLEAR3DMSFv2, build_focus_grids, make_inputs, log_odds_tta8

    t_start = time.time()
    sel, split = verify_inputs()
    b_bind = json.loads(BIND.read_text())
    w.bindings(b_bind['parent_binding'], b_bind['parent_binding_sha256'])

    seed_all(17)
    require(torch.cuda.device_count() == 1, 'GPU visibility must be 1')
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(
        LIMITS['allocator_bytes'] / torch.cuda.get_device_properties(0).total_memory
    )
    f_fine, f_ctx = build_focus_grids(device)

    if not smoke:
        (OUT / 'checkpoints').mkdir()
        w.save(OUT / 'phase.json', {'name': 'training'})

    pos_rows, pos_diams, pos_weights, neg_rows, rand_neg_rows, rand_neg_set, pat_fold, pool_meta = build_clean_fit_pool(split['fit'])
    print('POOL_META', pool_meta, f'elapsed={time.time()-t_start:.1f}s', flush=True)

    if smoke:
        pos_rows = pos_rows[::40]; pos_diams = pos_diams[::40]; pos_weights = pos_weights[::40]
        neg_rows = neg_rows[::3500]
        rand_neg_rows = neg_rows[::3]
        rand_neg_set = set(range(0, len(neg_rows), 3))

    n_pos = len(pos_rows)
    n_rand = len(rand_neg_rows)
    n_hard = min(40, len(neg_rows) - n_rand) if smoke else HARD_NEG_COUNT
    total_bank = n_pos + n_rand + n_hard

    f37_bank = np.empty((total_bank, 37, 37, 37), dtype=np.uint16)
    c27_bank = np.empty((total_bank, 27, 27, 27), dtype=np.uint16)

    # Extract positives + random negatives grouped by scan (sparse mmap I/O)
    stage1_rows = pos_rows + rand_neg_rows
    by_scan_s1 = defaultdict(list)
    for idx, r in enumerate(stage1_rows):
        by_scan_s1[r['scan']].append((idx, r))

    for sid, items in sorted(by_scan_s1.items()):
        r0 = items[0][1]
        img, val, inv = load_scan_arrays(sid, r0['patient'], r0['series'], use_mmap=(len(items) < 64))
        for idx, r in items:
            zyx = (inv @ np.array([r['x'], r['y'], r['z'], 1.0]))[:3]
            f37_bank[idx], c27_bank[idx] = extract_pair_u16(img, val, zyx)
        del img, val
    print(f'EXTRACTED_STAGE1 n={len(stage1_rows)} elapsed={time.time()-t_start:.1f}s', flush=True)

    # Train Stage-1 Scout on positives + random negatives with deterministic AMP
    seed_all(17)
    scout = CLEAR3DMSFv2().to(device)
    scout_opt = torch.optim.AdamW(scout.parameters(), lr=8e-4, weight_decay=1e-2)
    scout_scaler = torch.amp.GradScaler('cuda')
    s_epochs = 2 if smoke else SCOUT_EPOCHS
    s_steps = 6 if smoke else SCOUT_STEPS
    p_prob = pos_weights / pos_weights.sum()
    rng_s = np.random.default_rng(1700)
    total_updates = 0
    skipped_updates = 0

    for ep in range(s_epochs):
        lr = 1e-5 + 0.5 * (8e-4 - 1e-5) * (1.0 + math.cos(math.pi * ep / max(s_epochs - 1, 1)))
        for g in scout_opt.param_groups:
            g['lr'] = lr
        scout.train()
        ep_loss = 0.0
        for _ in range(s_steps):
            ip = rng_s.choice(n_pos, size=12, replace=True, p=p_prob)
            in_r = n_pos + rng_s.choice(n_rand, size=20, replace=(n_rand < 20))
            b_idx = np.concatenate([ip, in_r])
            fi, fv, ci, cv = sample_augmented_batch(f37_bank, c27_bank, b_idx, rng_s)
            vf, vc = make_inputs(
                torch.from_numpy(fi).to(device), torch.from_numpy(fv).to(device),
                torch.from_numpy(ci).to(device), torch.from_numpy(cv).to(device),
                f_fine, f_ctx,
            )
            targets = torch.tensor([1] * 12 + [0] * 20, device=device, dtype=torch.long)
            sample_w = torch.tensor([16.0 / 12.0] * 12 + [16.0 / 20.0] * 20, device=device, dtype=torch.float32)
            scout_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = scout(vf, vc)
                losses = F.cross_entropy(logits.float(), targets, reduction='none', label_smoothing=0.015)
                loss = (losses * sample_w).mean()
            require(bool(torch.isfinite(loss)), 'nonfinite scout loss')
            prev_scale = scout_scaler.get_scale()
            scout_scaler.scale(loss).backward()
            scout_scaler.unscale_(scout_opt)
            torch.nn.utils.clip_grad_norm_(scout.parameters(), 2.0)
            scout_scaler.step(scout_opt)
            scout_scaler.update()
            if scout_scaler.get_scale() < prev_scale:
                skipped_updates += 1
            else:
                total_updates += 1
            ep_loss += float(loss.detach())
            if not smoke:
                time.sleep(STEP_COOL_SLEEP)
        print(f'SCOUT_EPOCH {ep} loss={ep_loss/s_steps:.4f} elapsed={time.time()-t_start:.1f}s', flush=True)

    # Fast padded-slice + GPU-downsampled AMP mining across ALL clean fit negatives (~11 min total)
    scout.eval()
    mine_candidates_by_scan = defaultdict(list)
    for neg_i, r in enumerate(neg_rows):
        if neg_i not in rand_neg_set:
            mine_candidates_by_scan[r['scan']].append((neg_i, r))

    mined_scores = []  # (score, neg_i)
    for scan_num, (sid, items) in enumerate(sorted(mine_candidates_by_scan.items()), 1):
        r0 = items[0][1]
        from lidc3d.geometry import crop
        img, val, inv = load_scan_arrays(sid, r0['patient'], r0['series'], use_mmap=True)
        xyz1 = np.array([[r['x'], r['y'], r['z'], 1.0] for _, r in items], dtype=np.float64)
        zyxs = (xyz1 @ inv.T)[:, :3]
        z_order = np.argsort(zyxs[:, 0], kind='mergesort')
        items_ord = [items[int(k)] for k in z_order]
        zyxs_ord = zyxs[z_order]
        for start in range(0, len(items_ord), 64):
            chunk = items_ord[start:start + 64]
            czyx = zyxs_ord[start:start + 64]
            m_c = len(chunk)
            p51_b = np.empty((m_c, 51, 51, 51), dtype=np.float32)
            v51_b = np.empty((m_c, 51, 51, 51), dtype=np.float32)
            for k_i, zc in enumerate(czyx):
                p51_b[k_i], _ = crop(img, zc, 51, 0)
                v51_b[k_i], _ = crop(val, zc, 51, False)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(
                    torch.from_numpy(p51_b).to(device),
                    torch.from_numpy(v51_b).to(device),
                )
                vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                l0 = scout(vf, vc).float()
                probs = torch.sigmoid(l0[:, 1] - l0[:, 0]).cpu().numpy()
            for (neg_i, _), sc in zip(chunk, probs):
                mined_scores.append((float(sc), neg_i))
        del img, val
        if scan_num % 100 == 0 or scan_num == len(mine_candidates_by_scan):
            print(f'MINING_PROGRESS {scan_num}/{len(mine_candidates_by_scan)} elapsed={time.time()-t_start:.1f}s', flush=True)

    mined_scores.sort(key=lambda t: (-t[0], neg_rows[t[1]]['id']))
    top_hard = mined_scores[:n_hard]
    hard_neg_rows = [neg_rows[neg_i] for _, neg_i in top_hard]
    hard_neg_scout_scores = np.array([sc for sc, _ in top_hard], dtype=np.float64)
    print(f'MINING_DONE scored={len(mined_scores)} top1={hard_neg_scout_scores[0]:.4f} top_end={hard_neg_scout_scores[-1]:.4f} elapsed={time.time()-t_start:.1f}s', flush=True)
    del scout, scout_opt, scout_scaler, mined_scores

    # Extract top_hard into bank slots [n_pos + n_rand : total_bank] (sparse mmap I/O)
    by_scan_hard = defaultdict(list)
    for offset, r in enumerate(hard_neg_rows):
        by_scan_hard[r['scan']].append((n_pos + n_rand + offset, r))
    for sid, items in sorted(by_scan_hard.items()):
        r0 = items[0][1]
        img, val, inv = load_scan_arrays(sid, r0['patient'], r0['series'], use_mmap=(len(items) < 64))
        for bank_idx, r in items:
            zyx = (inv @ np.array([r['x'], r['y'], r['z'], 1.0]))[:3]
            f37_bank[bank_idx], c27_bank[bank_idx] = extract_pair_u16(img, val, zyx)
        del img, val
    print(f'EXTRACTED_HARD n={n_hard} elapsed={time.time()-t_start:.1f}s', flush=True)

    # Train 5-Fold Patient-Disjoint Ensemble on fit with deterministic AMP
    pos_folds = np.array([pat_fold[r['patient']] for r in pos_rows], dtype=np.int64)
    rand_folds = np.array([pat_fold[r['patient']] for r in rand_neg_rows], dtype=np.int64)
    hard_folds = np.array([pat_fold[r['patient']] for r in hard_neg_rows], dtype=np.int64)

    oof_ema_pos = np.zeros(n_pos, dtype=np.float32)
    oof_ema_rand = np.zeros(n_rand, dtype=np.float32)
    oof_raw_pos = np.zeros(n_pos, dtype=np.float32)
    oof_raw_rand = np.zeros(n_rand, dtype=np.float32)

    f_epochs = 2 if smoke else FOLD_EPOCHS
    f_steps = 6 if smoke else FOLD_STEPS
    fold_models_ema = []
    fold_models_raw = []
    fold_summaries = []

    for fold in range(2 if smoke else N_FOLDS):
        seed_all(100 + fold)
        model = CLEAR3DMSFv2().to(device)
        ema_model = copy.deepcopy(model).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2)
        scaler = torch.amp.GradScaler('cuda')
        rng_f = np.random.default_rng(2000 + fold)

        tr_pos = np.flatnonzero(pos_folds != fold)
        va_pos = np.flatnonzero(pos_folds == fold)
        tr_rand = np.flatnonzero(rand_folds != fold)
        va_rand = np.flatnonzero(rand_folds == fold)
        tr_hard = np.flatnonzero(hard_folds != fold)
        require(len(tr_pos) > 0 and len(tr_rand) > 0 and len(tr_hard) > 0, f'empty fold {fold} train partition')

        tr_pw = pos_weights[tr_pos] / pos_weights[tr_pos].sum()
        tr_hw = (0.5 + 0.5 * hard_neg_scout_scores[tr_hard])
        tr_hw = tr_hw / tr_hw.sum()

        for ep in range(f_epochs):
            if ep == 0:
                lr = 4e-4
            else:
                prog = (ep - 1) / max(f_epochs - 2, 1)
                lr = 1e-5 + 0.5 * (8e-4 - 1e-5) * (1.0 + math.cos(math.pi * prog))
            for g in opt.param_groups:
                g['lr'] = lr

            cut_ep = int(0.70 * f_epochs)
            hard_share = 0.25 if ep < cut_ep else (
                0.08 + 0.17 * 0.5 * (1.0 + math.cos(math.pi * (ep - cut_ep) / max(f_epochs - 1 - cut_ep, 1)))
            )
            w_pos_each = 16.0 / 10.0
            w_rand_each = (16.0 * (1.0 - hard_share)) / 12.0
            w_hard_each = (16.0 * hard_share) / 10.0
            sample_w = torch.tensor(
                [w_pos_each] * 10 + [w_rand_each] * 12 + [w_hard_each] * 10,
                device=device, dtype=torch.float32,
            )
            targets = torch.tensor([1] * 10 + [0] * 22, device=device, dtype=torch.long)

            model.train()
            ep_loss = 0.0
            for _ in range(f_steps):
                ip = tr_pos[rng_f.choice(len(tr_pos), size=10, replace=True, p=tr_pw)]
                ir = n_pos + tr_rand[rng_f.choice(len(tr_rand), size=12, replace=(len(tr_rand) < 12))]
                ih = n_pos + n_rand + tr_hard[rng_f.choice(len(tr_hard), size=10, replace=(len(tr_hard) < 10), p=tr_hw)]
                b_idx = np.concatenate([ip, ir, ih])

                fi, fv, ci, cv = sample_augmented_batch(f37_bank, c27_bank, b_idx, rng_f)
                vf, vc = make_inputs(
                    torch.from_numpy(fi).to(device), torch.from_numpy(fv).to(device),
                    torch.from_numpy(ci).to(device), torch.from_numpy(cv).to(device),
                    f_fine, f_ctx,
                )
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(vf, vc)
                    losses = F.cross_entropy(logits.float(), targets, reduction='none', label_smoothing=0.015)
                    loss = (losses * sample_w).mean()
                require(bool(torch.isfinite(loss)), 'nonfinite training loss')
                prev_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                step_skipped = scaler.get_scale() < prev_scale
                if step_skipped:
                    skipped_updates += 1
                else:
                    total_updates += 1
                    with torch.no_grad():
                        decay = 0.0 if ep < max(1, f_epochs // 4) else 0.995
                        for p_ema, p_cur in zip(ema_model.parameters(), model.parameters()):
                            p_ema.mul_(decay).add_(p_cur, alpha=1.0 - decay)
                        for b_ema, b_cur in zip(ema_model.buffers(), model.buffers()):
                            b_ema.copy_(b_cur)
                ep_loss += float(loss.detach())
                if not smoke:
                    time.sleep(STEP_COOL_SLEEP)

            if (ep + 1) % 5 == 0 or ep == f_epochs - 1:
                print(f'FOLD {fold} EP {ep+1}/{f_epochs} loss={ep_loss/f_steps:.4f} elapsed={time.time()-t_start:.1f}s', flush=True)

        # Held-out fit OOF scoring (8-view TTA)
        if len(va_pos) > 0:
            oof_ema_pos[va_pos] = eval_bank_tta8(ema_model, f37_bank, c27_bank, va_pos, device, f_fine, f_ctx)
            oof_raw_pos[va_pos] = eval_bank_tta8(model, f37_bank, c27_bank, va_pos, device, f_fine, f_ctx)
        if len(va_rand) > 0:
            oof_ema_rand[va_rand] = eval_bank_tta8(ema_model, f37_bank, c27_bank, n_pos + va_rand, device, f_fine, f_ctx)
            oof_raw_rand[va_rand] = eval_bank_tta8(model, f37_bank, c27_bank, n_pos + va_rand, device, f_fine, f_ctx)

        f_scores = np.concatenate([oof_ema_pos[va_pos], oof_ema_rand[va_rand]])
        f_labels = np.concatenate([np.ones(len(va_pos), dtype=np.int64), np.zeros(len(va_rand), dtype=np.int64)])
        f_auc_ema = roc_auc_np(f_scores, f_labels)
        f_auc_raw = roc_auc_np(np.concatenate([oof_raw_pos[va_pos], oof_raw_rand[va_rand]]), f_labels)
        print(f'FOLD {fold} DONE fit_oof_auc_ema={f_auc_ema:.4f} raw={f_auc_raw:.4f} elapsed={time.time()-t_start:.1f}s', flush=True)

        fold_models_ema.append(ema_model)
        fold_models_raw.append(model)
        fold_summaries.append({
            'fold': fold,
            'val_pos': int(len(va_pos)),
            'val_rand_neg': int(len(va_rand)),
            'fit_oof_auc_ema': f_auc_ema,
            'fit_oof_auc_raw': f_auc_raw,
        })

    if smoke:
        scale_s, bias_s = fit_platt_balanced(
            np.concatenate([oof_ema_pos, oof_ema_rand]),
            np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_rand, dtype=np.int64)]),
        )
        print(f'SMOKE_PASS updates={total_updates} skips={skipped_updates} scale={scale_s:.3f} bias={bias_s:.3f}', flush=True)
        return

    # Aggregate Fit-OOF across all 565 fit patients (ZERO dev access!)
    oof_y = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_rand, dtype=np.int64)])
    oof_lo_ema = np.concatenate([oof_ema_pos, oof_ema_rand])
    oof_lo_raw = np.concatenate([oof_raw_pos, oof_raw_rand])
    auc_ema = roc_auc_np(oof_lo_ema, oof_y)
    auc_raw = roc_auc_np(oof_lo_raw, oof_y)
    use_ema = auc_ema >= auc_raw
    chosen_models = fold_models_ema if use_ema else fold_models_raw
    chosen_oof_lo = oof_lo_ema if use_ema else oof_lo_raw

    scale_oof, bias_oof = fit_platt_balanced(chosen_oof_lo, oof_y)
    oof_prob = 1.0 / (1.0 + np.exp(-np.clip(scale_oof * chosen_oof_lo + bias_oof, -40.0, 40.0)))
    oof_pos_prob = oof_prob[:n_pos]
    oof_neg_prob = oof_prob[n_pos:]
    oof_pos_recall = float((oof_pos_prob >= 0.5).mean())
    oof_lt6_recall = float((oof_pos_prob[pos_diams < 6.0] >= 0.5).mean())
    oof_ge6_recall = float((oof_pos_prob[pos_diams >= 6.0] >= 0.5).mean())
    oof_neg_spec = float((oof_neg_prob < 0.5).mean())
    oof_bal_acc = 0.5 * (oof_pos_recall + oof_neg_spec)

    nat_scale = float(pool_meta['fit_clean_negatives']) / float(n_rand)
    best_nat_f1, best_nat_cut = -1.0, 0.5
    for q in np.linspace(0.80, 0.9998, 500):
        cut = float(np.quantile(oof_prob, q))
        tp = float((oof_pos_prob >= cut).sum())
        fp = float((oof_neg_prob >= cut).sum()) * nat_scale
        fn = float(n_pos) - tp
        f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if tp > 0 else 0.0
        if f1 > best_nat_f1:
            best_nat_f1, best_nat_cut = f1, cut

    del f37_bank, c27_bank

    # Fit-only OOF diagnostic on balanced-candidate-classification-20261106-001 roles['fit'] (1,968 rows, 0 dev touch)
    bal_fit_rows = json.loads(BAL_PATH.read_text())['roles']['fit']
    by_scan_bf = defaultdict(list)
    for i, r in enumerate(bal_fit_rows):
        by_scan_bf[r['scan']].append((i, r))
    bf_probs = np.empty(len(bal_fit_rows), dtype=np.float64)
    bf_labels = np.array([int(r['label']) for r in bal_fit_rows], dtype=np.int64)
    for m in chosen_models:
        m.eval()
    for sid, items in sorted(by_scan_bf.items()):
        r0 = items[0][1]
        held_fold = pat_fold[r0['patient']]
        m_oof = chosen_models[held_fold]
        img, val, inv = load_scan_arrays(sid, r0['patient'], r0['series'], use_mmap=True)
        m_cnt = len(items)
        fi = np.empty((m_cnt, 33, 33, 33), dtype=np.float32)
        fv = np.empty((m_cnt, 33, 33, 33), dtype=np.float32)
        ci = np.empty((m_cnt, 25, 25, 25), dtype=np.float32)
        cv = np.empty((m_cnt, 25, 25, 25), dtype=np.float32)
        for k_i, (_, r) in enumerate(items):
            require(pat_fold[r['patient']] == held_fold, 'scan patient fold mismatch')
            zyx = (inv @ np.array([r['x'], r['y'], r['z'], 1.0]))[:3]
            fi[k_i], fv[k_i], ci[k_i], cv[k_i] = extract_eval_pair(img, val, zyx)
        del img, val
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            vf, vc = make_inputs(
                torch.from_numpy(fi).to(device), torch.from_numpy(fv).to(device),
                torch.from_numpy(ci).to(device), torch.from_numpy(cv).to(device),
                f_fine, f_ctx,
            )
            lo = scale_oof * log_odds_tta8(m_oof, vf, vc) + bias_oof
            ps = torch.sigmoid(lo).cpu().numpy()
        for (orig_i, _), sc in zip(items, ps):
            bf_probs[orig_i] = float(sc)
    bf_acc = float(((bf_probs >= 0.5) == (bf_labels == 1)).mean())
    bf_auc = roc_auc_np(bf_probs, bf_labels)

    ckpt_shas = []
    for k, m in enumerate(chosen_models):
        cp = OUT / 'checkpoints' / f'fold-{k}.pt'
        tmp = OUT / 'checkpoints' / f'fold-{k}.tmp'
        torch.save({
            'fold': k,
            'model': m.state_dict(),
            'use_ema': use_ema,
            'scale_oof': scale_oof,
            'bias_oof': bias_oof,
            'nat_cut_oof': best_nat_cut,
        }, tmp)
        os.replace(tmp, cp)
        ckpt_shas.append(loader.hash_file(cp))

    fit_oof_report = {
        'scope': '100% fit-only 5-fold patient-disjoint OOF evaluation & Platt/prior calibration; zero dev access',
        'pool_meta': pool_meta,
        'oof_auc_ema': auc_ema,
        'oof_auc_raw': auc_raw,
        'selected_variant': 'ema' if use_ema else 'raw',
        'scale_oof': scale_oof,
        'bias_oof': bias_oof,
        'oof_balanced_pos_recall_at_0_5': oof_pos_recall,
        'oof_balanced_lt6mm_recall_at_0_5': oof_lt6_recall,
        'oof_balanced_ge6mm_recall_at_0_5': oof_ge6_recall,
        'oof_balanced_neg_recall_at_0_5': oof_neg_spec,
        'oof_expected_1to1_accuracy_at_0_5': oof_bal_acc,
        'oof_balanced_fit1968_single_fold_acc_at_0_5': bf_acc,
        'oof_balanced_fit1968_single_fold_auc': bf_auc,
        'oof_natural_prevalence_threshold': best_nat_cut,
        'oof_natural_prevalence_est_f1': best_nat_f1,
        'folds': fold_summaries,
        'checkpoint_sha256': ckpt_shas,
    }
    w.save(OUT / 'fit-oof-summary.json', fit_oof_report)
    print('FIT_OOF_SUMMARY', json.dumps(fit_oof_report), flush=True)

    # Validation phase: score selection-001 dev.selected (1,060 candidates) with 5-fold x 8-view TTA
    w.save(OUT / 'phase.json', {'name': 'validation'})
    dev_rows = sel['dev']['selected']
    by_scan_dev = defaultdict(list)
    for i, r in enumerate(dev_rows):
        by_scan_dev[r['scan']].append((i, r))

    dev_scores = [0.0] * len(dev_rows)
    dev_scores_prob_avg = [0.0] * len(dev_rows)
    dev_scores_uncal = [0.0] * len(dev_rows)
    dev_labels = [int(r['label']) for r in dev_rows]

    for sid, items in sorted(by_scan_dev.items()):
        r0 = items[0][1]
        img, val, inv = load_scan_arrays(sid, r0['patient'], r0['series'], use_mmap=True)
        m_cnt = len(items)
        fi = np.empty((m_cnt, 33, 33, 33), dtype=np.float32)
        fv = np.empty((m_cnt, 33, 33, 33), dtype=np.float32)
        ci = np.empty((m_cnt, 25, 25, 25), dtype=np.float32)
        cv = np.empty((m_cnt, 25, 25, 25), dtype=np.float32)
        for k_i, (_, r) in enumerate(items):
            zyx = (inv @ np.array([r['x'], r['y'], r['z'], 1.0]))[:3]
            fi[k_i], fv[k_i], ci[k_i], cv[k_i] = extract_eval_pair(img, val, zyx)
        del img, val

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            vf, vc = make_inputs(
                torch.from_numpy(fi).to(device), torch.from_numpy(fv).to(device),
                torch.from_numpy(ci).to(device), torch.from_numpy(cv).to(device),
                f_fine, f_ctx,
            )
            lo_raw_sum = torch.zeros(m_cnt, device=device, dtype=torch.float32)
            lo_cal_sum = torch.zeros(m_cnt, device=device, dtype=torch.float32)
            prob_sum = torch.zeros(m_cnt, device=device, dtype=torch.float32)
            for m_fold in chosen_models:
                lo_raw = log_odds_tta8(m_fold, vf, vc)
                lo_cal = scale_oof * lo_raw + bias_oof
                lo_raw_sum = lo_raw_sum + lo_raw
                lo_cal_sum = lo_cal_sum + lo_cal
                prob_sum = prob_sum + torch.sigmoid(lo_cal)
            nf = float(len(chosen_models))
            p_main = torch.sigmoid(lo_cal_sum / nf).cpu().tolist()
            p_lin = (prob_sum / nf).cpu().tolist()
            p_unc = torch.sigmoid(lo_raw_sum / nf).cpu().tolist()

        for (orig_i, _), s_m, s_l, s_u in zip(items, p_main, p_lin, p_unc):
            require(math.isfinite(s_m) and math.isfinite(s_l) and math.isfinite(s_u), 'nonfinite dev score')
            dev_scores[orig_i] = float(s_m)
            dev_scores_prob_avg[orig_i] = float(s_l)
            dev_scores_uncal[orig_i] = float(s_u)

    w.save(OUT / 'predictions.json', {
        'score': dev_scores,
        'label': dev_labels,
        'ids': [r['id'] for r in dev_rows],
        'scope': 'sampled development only; 5-fold fit-only CLEAR3D-MSFv2 ensemble + 8-view TTA + fit-OOF Platt calibration; reference HOLD',
    })
    w.save(OUT / 'predictions-detailed.json', {
        'score': dev_scores,
        'score_prob_avg': dev_scores_prob_avg,
        'score_uncalibrated': dev_scores_uncal,
        'scale_oof': scale_oof,
        'bias_oof': bias_oof,
        'nat_cut_oof': best_nat_cut,
        'label': dev_labels,
        'ids': [r['id'] for r in dev_rows],
        'scope': 'sampled development only; detailed audit variants; reference HOLD',
    })
    torch.cuda.synchronize()
    peak_gpu = int(torch.cuda.max_memory_allocated())
    require(peak_gpu <= LIMITS['allocator_bytes'], 'GPU allocator cap exceeded')
    res = {
        'status': 'PASS',
        'updates': total_updates,
        'skips': skipped_updates,
        'dev_predictions': len(dev_scores),
        'gpu_peak_bytes': peak_gpu,
        'fit_oof_auc': max(auc_ema, auc_raw),
        'fit_oof_expected_1to1_acc': oof_bal_acc,
        'fit_oof_balanced1968_single_fold_acc': bf_acc,
        'fit_oof_balanced1968_single_fold_auc': bf_auc,
        'selection_sha256': SELECTION_SHA,
        'model_sha256': MODEL_SHA,
        'checkpoint_sha256': ckpt_shas,
        'reference_release': 'HOLD',
        'scope': 'exploratory sampled development; fit-only 5-fold AMP ensemble + 472k hard-negative mining; no test claim',
    }
    w.save(OUT / 'worker-result.json', res)
    w.save(OUT / 'phase.json', {'name': 'complete'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--gate-fd', type=int)
    p.add_argument('--parent-pid', type=int)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()

    if a.smoke:
        work(smoke=True)
        return 0

    if a.worker:
        require(ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) == 0 and os.getppid() == a.parent_pid, 'parent gate')
        require(os.read(a.gate_fd, 1) == b'1', 'gate')
        os.close(a.gate_fd)
        try:
            work(smoke=False)
        except BaseException as e:
            w.save(OUT / 'worker-failure.json', {'error': repr(e)})
            raise
        return 0

    if not a.execute:
        print('PLAN ONLY', LIMITS)
        return 0

    verify_inputs()
    b_bind = json.loads(BIND.read_text())
    w.bindings(b_bind['parent_binding'], b_bind['parent_binding_sha256'])
    require(not OUT.exists() and OUT.parent.is_dir(), 'output exists or parent absent')
    require(w.available_ram() >= LIMITS['min_ram_bytes'] and w.disk_free(OUT.parent) >= LIMITS['min_disk_bytes'], 'insufficient host RAM/disk')
    gpu = w.reviewed.gpu_snapshot()
    require(gpu['free_bytes'] >= LIMITS['min_gpu_bytes'], f"GPU free memory below 10 GiB: {gpu['free_bytes']}")

    OUT.mkdir()
    w.save(OUT / 'phase.json', {'name': 'setup'})
    w.save(OUT / 'gpu-before.json', gpu)
    w.save(OUT / 'invocation.json', {
        'selection_sha256': SELECTION_SHA,
        'balanced_sha256': BAL_SHA,
        'fit_jsonl_sha256': FIT_JSONL_SHA,
        'script_sha256': loader.hash_file(__file__),
        'model_sha256': loader.hash_file(MODEL_FILE),
        'loader_sha256': loader.hash_file(loader.__file__),
        'limits': LIMITS,
        'reference_release': 'HOLD',
        'cap4000': 'NOT_ADOPTED',
        'scope': 'exploratory fit-only 5-fold CLEAR3D-MSFv2 AMP multi-scale ensemble with full 472k fit candidates_V2 hard-negative mining; zero dev tuning',
    })

    def validate(path, device, limits):
        r = json.loads((path / 'worker-result.json').read_text())
        require(r['status'] == 'PASS' and r['dev_predictions'] == 1060, 'worker evidence invalid')
        return r

    stop = [False]
    handlers = {s: signal.signal(s, lambda *args: stop.__setitem__(0, True)) for s in (signal.SIGINT, signal.SIGTERM)}
    child = None
    try:
        with (OUT / 'worker.log').open('xb', buffering=0) as log:
            child = w.OwnedWorker([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'], w.child_env('cuda'), log)
            w.save(OUT / 'owned-worker.json', child.identity)
            result = w.supervise(child, OUT, 'cuda', LIMITS, observe=w.reviewed.gpu_snapshot, stop=lambda: stop[0], validate_evidence=validate, completion_reason='EXPLORATORY_COMPLETE')
        return 0 if result['status'] == 'PASS' else 1
    finally:
        if child is not None and child.proc.poll() is None:
            child.cleanup()
        for s, h in handlers.items():
            signal.signal(s, h)


if __name__ == '__main__':
    raise SystemExit(main())
