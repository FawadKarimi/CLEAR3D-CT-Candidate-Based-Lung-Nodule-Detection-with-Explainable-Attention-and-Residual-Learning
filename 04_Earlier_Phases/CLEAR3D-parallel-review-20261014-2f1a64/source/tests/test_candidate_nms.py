import numpy as np
from scipy import ndimage as ndi

from lidc3d import candidates


def reference_nms(proposals, sigmas, spacing, nms_mm):
    selected = []
    suppressed = 0
    for score, scale, z, y, x in proposals:
        center = np.array([z, y, x], float)
        if any(np.linalg.norm(center - row["zyx"]) * spacing < nms_mm
               for row in selected):
            suppressed += 1
            continue
        selected.append({"zyx": center.tolist(), "log_response": score,
                         "sigma_mm": sigmas[scale]})
    return selected, suppressed


def random_proposals(seed, count=400):
    rng = np.random.default_rng(seed)
    rows = [(float(rng.integers(0, 20)), int(rng.integers(0, 4)),
             *map(int, rng.integers(0, 31, size=3)))
            for _ in range(count)]
    return sorted(rows, key=lambda row: (-row[0], row[1], row[2], row[3], row[4]))


def test_indexed_nms_exactly_matches_reference_randomized():
    sigmas = [1, 1.5, 2, 3]
    configurations = [(1.0, 3.0), (0.7, 3.1), (2.0, 0.5),
                      (1.0, 0.0), (1.0, -1.0),
                      (1.0, float("nan")), (1.0, float("inf"))]
    for seed in range(5):
        proposals = random_proposals(seed)
        for spacing, nms_mm in configurations:
            expected = reference_nms(proposals, sigmas, spacing, nms_mm)
            actual = candidates._greedy_nms(
                proposals, sigmas, spacing, nms_mm)
            assert actual == expected


def test_indexed_nms_preserves_strict_boundary_and_priority():
    proposals = [
        (10.0, 0, 0, 0, 0),
        (9.0, 1, 0, 0, 2),   # Strictly inside: suppressed.
        (8.0, 2, 0, 0, 3),   # Exactly at 3 mm: retained.
        (7.0, 3, 0, 0, 3),   # Same center, lower priority: suppressed.
    ]
    selected, suppressed = candidates._greedy_nms(
        proposals, [1, 1.5, 2, 3], 1.0, 3.0)
    assert [row["zyx"] for row in selected] == [[0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 3.0]]
    assert suppressed == 2


def test_indexed_nms_preserves_greedy_chain_and_float_boundaries():
    chain = [(3.0, 0, 0, 0, 0), (2.0, 0, 0, 0, 2),
             (1.0, 0, 0, 0, 4)]
    selected, suppressed = candidates._greedy_nms(chain, [1], 1.0, 3.0)
    assert [row["zyx"] for row in selected] == [[0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 4.0]]
    assert suppressed == 1

    pair = [(2.0, 0, 0, 0, 0), (1.0, 0, 0, 0, 1)]
    inside = np.nextafter(1.0, 0.0)
    outside = np.nextafter(1.0, 2.0)
    assert candidates._greedy_nms(pair, [1], inside, 1.0)[1] == 1
    assert candidates._greedy_nms(pair, [1], 1.0, 1.0)[1] == 0
    assert candidates._greedy_nms(pair, [1], outside, 1.0)[1] == 0


def test_indexed_nms_falls_back_outside_exact_grid_coordinate_domain():
    proposals = [
        (2.0, 0, 0, 0, 9366824361445148),
        (1.0, 0, 0, 0, 18733648722890298),
    ]
    args = (proposals, [1], 1.8932217563645826,
            1.7733475669133744e16)
    assert candidates._greedy_nms(*args) == reference_nms(*args)


def test_indexed_nms_matches_reference_across_cell_boundaries():
    proposals = [
        (4.0, 0, 0, 0, 3),
        (3.0, 0, 0, 0, 4),  # Adjacent bin, suppressed by first center.
        (2.0, 0, 4, 4, 4),  # Diagonal distance is outside the radius.
        (1.0, 0, 0, 0, 8),  # Retained after an earlier suppression.
    ]
    assert candidates._greedy_nms(proposals, [1], 1.0, 3.0) == \
        reference_nms(proposals, [1], 1.0, 3.0)

    corner_pair = [(2.0, 0, 3, 3, 3), (1.0, 0, 4, 4, 4)]
    assert candidates._greedy_nms(corner_pair, [1], 1.0, 3.0) == \
        reference_nms(corner_pair, [1], 1.0, 3.0)

    negative_pair = [(2.0, 0, -5, -5, -5), (1.0, 0, -4, -4, -4)]
    assert candidates._greedy_nms(negative_pair, [1], 1.0, 3.0) == \
        reference_nms(negative_pair, [1], 1.0, 3.0)


def test_indexed_nms_falls_back_when_finite_grid_ratio_overflows():
    proposals = [(2.0, 0, 0, 0, 0), (1.0, 0, 0, 0, 1)]
    args = (proposals, [1], np.nextafter(0.0, 1.0), 1.0)
    assert candidates._greedy_nms(*args) == reference_nms(*args)


def test_indexed_nms_accepts_one_shot_proposal_iterators():
    proposals = [(2.0, 0, 0, 0, 0), (1.0, 0, 0, 0, 1)]
    expected = reference_nms(proposals, [1], 1.0, 3.0)
    assert candidates._greedy_nms(iter(proposals), [1], 1.0, 3.0) == expected


def reference_generate(image, mask, config, spacing=1.0):
    """Frozen literal fb079ed candidate-generation implementation."""
    threshold = config["threshold"]
    sigmas = config["sigmas_mm"]

    def response(sigma):
        return -sigma**2 / spacing**2 * ndi.gaussian_laplace(
            image.astype(np.float32), sigma / spacing,
            mode=config.get("gaussian_mode", "reflect"),
            truncate=config.get("gaussian_truncate", 4))

    prev = None
    current = response(sigmas[0])
    proposals = []
    peak_count = 0
    for scale, sigma in enumerate(sigmas):
        nxt = response(sigmas[scale + 1]) if scale + 1 < len(sigmas) else None
        keep = mask & np.isfinite(current) & (current > 0)
        keep &= current == ndi.maximum_filter(
            current, size=3, mode="constant", cval=-np.inf)
        if prev is not None:
            keep &= current > ndi.maximum_filter(
                prev, size=3, mode="constant", cval=-np.inf)
        if nxt is not None:
            keep &= current >= ndi.maximum_filter(
                nxt, size=3, mode="constant", cval=-np.inf)
        peak_count += int(keep.sum())
        keep &= current >= threshold
        for z, y, x in np.argwhere(keep):
            proposals.append((float(current[z, y, x]), scale,
                              int(z), int(y), int(x)))
        prev, current = current, nxt
    proposals.sort(key=lambda p: (-p[0], p[1], p[2], p[3], p[4]))
    selected, suppressed = reference_nms(
        proposals, sigmas, spacing, config["nms_mm"])
    cap = int(config["cap"])
    return selected[:cap], {"positive_peaks_in_mask": peak_count,
        "threshold_peaks": len(proposals), "nms_suppressed": suppressed,
        "before_cap": len(selected), "cap_hit": len(selected) >= cap,
        "precap_candidates": selected}


def test_full_generate_is_exactly_equal_to_frozen_reference():
    for seed, spacing in enumerate((0.7, 1.0, 1.25)):
        rng = np.random.default_rng(seed)
        image = rng.normal(0.2, 0.5, size=(9, 10, 11)).astype(np.float32)
        mask = rng.random(image.shape) > 0.2
        original_image, original_mask = image.copy(), mask.copy()
        config = {"threshold": 0.005, "sigmas_mm": [0.8, 1.3, 2.0],
                  "nms_mm": 2.4, "cap": 7,
                  "gaussian_mode": "reflect", "gaussian_truncate": 4}
        assert candidates.generate(image, mask, config, spacing) == \
            reference_generate(image, mask, config, spacing)
        assert np.array_equal(image, original_image)
        assert np.array_equal(mask, original_mask)


def test_indexed_nms_distance_work_is_linear_for_sparse_centers(monkeypatch):
    calls = 0
    real_norm = candidates.np.linalg.norm

    def counted_norm(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_norm(*args, **kwargs)

    monkeypatch.setattr(candidates.np.linalg, "norm", counted_norm)
    proposals = [(float(5000 - index), 0, 0, 0, index * 6)
                 for index in range(5000)]
    selected, suppressed = candidates._greedy_nms(
        proposals, [1], 1.0, 3.0)
    assert len(selected) == 5000
    assert suppressed == 0
    assert calls <= len(proposals)
