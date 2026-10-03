from types import SimpleNamespace
import numpy as np
import pytest
from lidc3d import ingest


CFG = {"orientation_atol": 1e-6, "position_atol_mm": 1e-4,
       "spacing_rtol": 1e-5, "spacing_atol_mm": 1e-4}


def dataset(z, value, index):
    return SimpleNamespace(
        PatientID="p", StudyInstanceUID="study", SeriesInstanceUID="series",
        SOPInstanceUID=f"sop-{index}", Modality="CT", NumberOfFrames=1,
        ImageType=[], Rows=2, Columns=2,
        ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
        ImagePositionPatient=[0, 0, z], PixelSpacing=[1, 1],
        PhotometricInterpretation="MONOCHROME2",
        RescaleSlope=1, RescaleIntercept=0,
        pixel_array=np.full((2, 2), value, np.int16),
        filename=f"slice-{index}.dcm")


@pytest.fixture
def fake_reader(monkeypatch):
    mapping = {}
    monkeypatch.setattr(ingest.pydicom, "dcmread", lambda path: mapping[path])
    monkeypatch.setattr(ingest, "sha256", lambda path: f"hash:{path}")
    return mapping


def test_strict_loader_still_rejects_irregular_spacing(fake_reader):
    fake_reader.update({
        "a": dataset(0, 0, 0),
        "b": dataset(.5, 50, 1),
        "c": dataset(1.75, 175, 2),
    })
    with pytest.raises(ValueError, match="Irregular slice spacing"):
        ingest.load_series(["a", "b", "c"], CFG)


def test_coordinate_aware_loader_interpolates_physical_axis(fake_reader):
    fake_reader.update({
        "a": dataset(0, 0, 0),
        "b": dataset(.5, 50, 1),
        "c": dataset(1.75, 175, 2),
    })
    volume = ingest.load_series(
        ["c", "a", "b"], CFG, coordinate_aware_spacing_mm=.5,
        max_interpolation_gap_mm=1.25)
    z = np.asarray(volume.metadata["slice_positions"])[:, 2]
    assert np.allclose(volume.hu[:, 0, 0], 100*z)
    assert np.all(volume.valid)
    assert volume.metadata["coordinate_aware_resampling"] is True
    assert np.allclose(
        np.diff(z), volume.metadata["native_spacing_zyx"][0])
    assert np.isclose(z[[0, -1]], [0, 1.75]).all()


def test_coordinate_aware_loader_rejects_unapproved_gap(fake_reader):
    fake_reader.update({
        "a": dataset(0, 0, 0),
        "b": dataset(.5, 50, 1),
        "c": dataset(3, 300, 2),
    })
    with pytest.raises(ValueError, match="exceeds approved"):
        ingest.load_series(
            ["a", "b", "c"], CFG, coordinate_aware_spacing_mm=1,
            max_interpolation_gap_mm=1.25)


def test_coordinate_aware_validity_requires_both_support_slices(fake_reader):
    first = dataset(0, 0, 0)
    middle = dataset(.5, 50, 1)
    last = dataset(1.75, 175, 2)
    middle.PixelPaddingValue = 50
    fake_reader.update({"a": first, "b": middle, "c": last})
    volume = ingest.load_series(
        ["a", "b", "c"], CFG, coordinate_aware_spacing_mm=.5,
        max_interpolation_gap_mm=1.25)
    assert volume.valid[0].all()
    assert not volume.valid[1].any()
    assert not volume.valid[2].any()
    assert volume.valid[-1].all()
    assert np.all(volume.hu[~volume.valid] == -1000)


def test_regular_series_is_unchanged_with_opt_in_arguments(fake_reader):
    fake_reader.update({
        "a": dataset(0, 0, 0),
        "b": dataset(1, 100, 1),
        "c": dataset(2, 200, 2),
    })
    strict = ingest.load_series(["a", "b", "c"], CFG)
    opted = ingest.load_series(
        ["a", "b", "c"], CFG, coordinate_aware_spacing_mm=.5,
        max_interpolation_gap_mm=1.25)
    assert np.array_equal(strict.hu, opted.hu)
    assert np.array_equal(strict.valid, opted.valid)
    assert np.array_equal(strict.affine, opted.affine)
    assert opted.metadata["coordinate_aware_resampling"] is False
