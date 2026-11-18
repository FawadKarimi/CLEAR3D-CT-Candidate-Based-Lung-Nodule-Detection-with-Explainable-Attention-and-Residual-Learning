"""Eligibility rules for the proposed benchmark.

The brightness component filter used by redesign-001 is intentionally absent.
A nodule is not removed because it is faint, small, ground-glass, pleural,
or vessel-adjacent. This module only checks identifiers and geometry bounds
supplied by the caller. It does not open images.
"""


def eligible_reference(nodule, image_shape):
    z, y, x = nodule["zyx"]
    depth, height, width = image_shape
    inside = 0 <= z < depth and 0 <= y < height and 0 <= x < width
    return bool(inside and nodule.get("prespecified_include", True))


def eligible_candidate(candidate, allowed_series):
    return candidate["series_uid"] in set(allowed_series) and candidate["class_label"] in (0, 1)
