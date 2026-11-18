"""All arrays use z,y,x; homogeneous affines map those indices to DICOM LPS mm."""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
from scipy import ndimage as ndi


@dataclass
class Volume:
    hu: np.ndarray
    valid: np.ndarray
    affine: np.ndarray
    metadata: dict


def to_world(zyx, affine):
    xyz = np.asarray(zyx, dtype=float)
    return xyz @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]


def to_index(lps, affine):
    return to_world(lps, np.linalg.inv(affine))


def resample_lps(volume, spacing=1.0):
    """Canonical LPS axes permit anatomical left/right augmentation.

    The grid encloses all native voxel centers. An output sample is valid only
    when its complete linear interpolation stencil has valid source support.
    """
    if spacing <= 0:
        raise ValueError("Spacing must be positive")
    corners = list(itertools.product(*[(0, n - 1) for n in volume.hu.shape]))
    world = to_world(corners, volume.affine)
    low, high = world.min(0), world.max(0)
    shape = (np.ceil((high - low) / spacing).astype(int) + 1)[::-1]
    affine = np.eye(4)
    affine[:3, :3] = np.array([[0, 0, spacing], [0, spacing, 0], [spacing, 0, 0]])
    affine[:3, 3] = low
    transform = np.linalg.inv(volume.affine) @ affine
    args = dict(matrix=transform[:3, :3], offset=transform[:3, 3],
                output_shape=tuple(shape), order=1, mode="constant", prefilter=False)
    hu = ndi.affine_transform(volume.hu.astype(np.float32), cval=-1000, **args)
    support = ndi.affine_transform(volume.valid.astype(np.float32), cval=0, **args)
    valid = support >= 1 - 1e-6
    hu[~valid] = -1000
    return Volume(hu, valid, affine, {**volume.metadata, "spacing_mm": spacing,
                                    "native_affine": volume.affine.tolist()})


def resample_mask(mask, native_affine, target):
    transform = np.linalg.inv(native_affine) @ target.affine
    return ndi.affine_transform(mask.astype(np.uint8), transform[:3, :3],
        transform[:3, 3], output_shape=target.hu.shape, order=0,
        mode="constant", cval=0, prefilter=False).astype(bool)


def normalize(hu, window=(-1000, 400)):
    low, high = window
    if high <= low:
        raise ValueError("Invalid HU window")
    return ((np.clip(hu, low, high) - low) / (high - low)).astype(np.float32)


def crop(array, center_zyx, size, fill=0):
    """Round half up; even crop covers [center-size//2, center+size//2).

    Returns the patch and integer start index, including negative starts.
    A candidate lies at index size//2, not between the central two voxels.
    """
    if size <= 0:
        raise ValueError("Patch size must be positive")
    center = np.floor(np.asarray(center_zyx) + 0.5).astype(int)
    start = center - size // 2
    stop = start + size
    src_start = np.maximum(start, 0)
    src_stop = np.minimum(stop, array.shape)
    out = np.full((size,) * 3, fill, dtype=array.dtype)
    if np.all(src_stop > src_start):
        src = tuple(slice(a, b) for a, b in zip(src_start, src_stop))
        dst = tuple(slice(a, b) for a, b in zip(src_start-start, src_stop-start))
        out[dst] = array[src]
    return out, start
