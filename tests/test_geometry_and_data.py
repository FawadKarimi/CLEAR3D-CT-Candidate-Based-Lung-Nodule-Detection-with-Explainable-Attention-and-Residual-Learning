import numpy as np
import pytest
from lidc3d.geometry import Volume, to_world, to_index, resample_lps, crop, normalize
from lidc3d.candidates import generate, lung_mask
from lidc3d.references import candidate_label, consolidate, make_references
from lidc3d.splits import patient_split


def test_oblique_roundtrip_and_resampling():
    affine = np.array([[0, 0, 0.7, 20], [0, 0.8, 0, -10], [2, 0, 0, 5], [0,0,0,1]], float)
    points = np.array([[0,0,0], [3,4,5], [1.2,3.4,5.6]])
    assert np.allclose(to_index(to_world(points, affine), affine), points)
    volume = Volume(np.full((4,5,6), -500., np.float32), np.ones((4,5,6), bool), affine, {})
    result = resample_lps(volume)
    assert np.allclose(result.hu[result.valid], -500)
    assert np.allclose(to_world([0,0,0], result.affine), [20,-10,5])


def test_even_crop_and_padding():
    array = np.arange(27).reshape(3,3,3)
    patch, start = crop(array, [0,0,0], 4, -1)
    assert tuple(start) == (-2,-2,-2)
    assert patch[2,2,2] == array[0,0,0]
    assert patch[0,0,0] == -1
    assert normalize(np.array([-2000, -1000, -300, 400, 900])).tolist() == [0,0,0.5,1,1]


def test_log_bright_blob_and_physical_suppression():
    coords = np.indices((25,25,25))
    image = np.exp(-sum((coords[i]-12)**2 for i in range(3))/(2*2**2)).astype(np.float32)
    cfg = {"threshold": .01, "sigmas_mm": [1,1.5,2,3], "nms_mm": 3, "cap": 400}
    candidates, diag = generate(image, np.ones(image.shape, bool), cfg)
    assert candidates and candidates[0]["zyx"] == [12,12,12]
    assert candidates[0]["log_response"] > 0
    for i,a in enumerate(candidates):
        assert all(np.linalg.norm(np.array(a["zyx"])-b["zyx"]) >= 3 for b in candidates[i+1:])
    with pytest.raises(ValueError, match="unresolved"):
        generate(image, np.ones(image.shape,bool), {**cfg, "threshold": None})


def test_lung_mask_empty_failure_and_internal_air():
    hu = np.zeros((8,24,24), np.float32)
    hu[:, 6:18, 6:18] = -800
    valid = np.ones_like(hu, bool)
    mask, base, info = lung_mask(hu, valid, min_component_mm3=10, dilation_mm=1)
    assert base[:,10,10].all() and mask.sum() > base.sum()
    with pytest.raises(ValueError, match="empty"):
        lung_mask(np.zeros_like(hu), valid, min_component_mm3=10)


def test_duplicate_positive_training_labels_and_ignore_precedence():
    reference = [{"center_lps": [0,0,0], "radius_mm": 2}]
    ignored = [{"center_lps": [0,0,0], "radius_mm": 5}]
    assert candidate_label([0,0,0], reference, ignored) == 1
    assert candidate_label([1,0,0], reference, ignored) == 1
    assert candidate_label([4,0,0], reference, ignored) == -1
    assert candidate_label([8,0,0], reference, ignored) == 0


def mark(identifier, reader, center, coords=((1,1,1),), diameter=4):
    return {"id": identifier, "reader": reader, "kind": "nodule", "center_lps": center,
            "diameter_mm": diameter, "coords": np.array(coords)}


def test_consolidation_no_same_reader_and_competing_associations():
    marks = [mark("a","r1",[0,0,0]), mark("b","r2",[0,0,0]), mark("c","r2",[1,0,0])]
    clusters, trace, ambiguity = consolidate(marks)
    assert ambiguity
    assert all(len({m["reader"] for m in c}) == len(c) for c in clusters)


def test_strict_majority_mask_and_three_reader_threshold(tmp_path):
    marks = [mark("a","1",[0,0,0]), mark("b","2",[0,0,0]), mark("c","3",[0,0,0],((2,2,2),))]
    positive, ignored, audit = make_references(marks, np.eye(4), tmp_path/"masks", "scan")
    assert len(positive) == 1 and len(ignored) == 0
    assert np.load(positive[0]["mask_path"])["coords_zyx"].tolist() == [[1,1,1]]


def test_patient_split_repeated_exams_deterministic():
    scans = [{"patient_id": f"p{i}", "positive": []} for i in range(30)]
    scans += [{"patient_id": "p1", "positive": []}]
    first, record = patient_split(scans)
    assert first == patient_split(list(reversed(scans)))[0]
    assert len(first) == 30 and set(first.values()) == {"train","val","test"}
