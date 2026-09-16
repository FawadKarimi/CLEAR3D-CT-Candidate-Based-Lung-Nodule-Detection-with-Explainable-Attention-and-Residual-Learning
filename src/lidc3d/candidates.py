"""Annotation-free lung masking and multiscale LoG proposals."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def lung_mask(hu, valid, spacing=1.0, air_hu=-400, min_component_mm3=10000, dilation_mm=5):
    """Prospective classical mask, not a validated clinical segmentation.

    Remove air connected to the 2D image border on each axial slice; keep the
    two largest qualifying 3D components; fill axial holes (vessels/nodules),
    then dilate in physical space. No annotation-informed repair is allowed.
    """
    air = (hu < air_hu) & valid
    internal = np.zeros_like(air)
    for z, plane in enumerate(air):
        labels, _ = ndi.label(plane, structure=np.ones((3, 3)))
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        internal[z] = plane & ~np.isin(labels, border)
    labels, count = ndi.label(internal, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())
    eligible = [i for i in range(1, count + 1) if sizes[i] * spacing**3 >= min_component_mm3]
    chosen = sorted(eligible, key=lambda i: (-sizes[i], i))[:2]
    if not chosen:
        raise ValueError("lung_mask_empty_or_too_small")
    mask = np.isin(labels, chosen)
    for z in range(mask.shape[0]):
        mask[z] = ndi.binary_fill_holes(mask[z])
    base = mask & valid
    if dilation_mm:
        mask = ndi.distance_transform_edt(~base, sampling=spacing) <= dilation_mm
    mask &= valid
    diagnostics = {"components_retained": len(chosen), "base_voxels": int(base.sum()),
                   "dilated_voxels": int(mask.sum()), "valid_voxels": int(valid.sum()),
                   "retained_fraction": float(mask.sum()/max(valid.sum(), 1)),
                   "base_touches_z_boundary": bool(base[0].any() or base[-1].any())}
    return mask, base, diagnostics


def generate(image, mask, config, spacing=1.0):
    """Stream scales to avoid materializing a four-scale full-volume array."""
    threshold = config["threshold"]
    if threshold is None:
        raise ValueError("LoG threshold unresolved: run validation selection first")
    sigmas = config["sigmas_mm"]
    if not sigmas or sorted(set(sigmas)) != list(sigmas) or min(sigmas) <= 0:
        raise ValueError("Sigmas must be unique, increasing, and positive")
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
        keep &= current == ndi.maximum_filter(current, size=3, mode="constant", cval=-np.inf)
        # Earlier scales win equal adjacent-scale responses.
        if prev is not None:
            keep &= current > ndi.maximum_filter(prev, size=3, mode="constant", cval=-np.inf)
        if nxt is not None:
            keep &= current >= ndi.maximum_filter(nxt, size=3, mode="constant", cval=-np.inf)
        peak_count += int(keep.sum())
        keep &= current >= threshold
        for z, y, x in np.argwhere(keep):
            proposals.append((float(current[z, y, x]), scale, int(z), int(y), int(x)))
        prev, current = current, nxt
    proposals.sort(key=lambda p: (-p[0], p[1], p[2], p[3], p[4]))
    selected = []
    suppressed = 0
    # Do not stop at the cap: retain suppression diagnostics and pre-cap centers.
    for score, scale, z, y, x in proposals:
        center = np.array([z, y, x], float)
        if any(np.linalg.norm(center - row["zyx"]) * spacing < config["nms_mm"]
               for row in selected):
            suppressed += 1
            continue
        selected.append({"zyx": center.tolist(), "log_response": score,
                         "sigma_mm": sigmas[scale]})
    cap = int(config["cap"])
    if cap < 1:
        raise ValueError("Candidate cap must be positive")
    return selected[:cap], {"positive_peaks_in_mask": peak_count,
        "threshold_peaks": len(proposals), "nms_suppressed": suppressed,
        "before_cap": len(selected), "cap_hit": len(selected) >= cap,
        "precap_candidates": selected}
