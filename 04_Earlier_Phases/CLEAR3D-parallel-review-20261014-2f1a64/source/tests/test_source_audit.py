"""Synthetic CT and temporary SQLite only; no pylidc installation or patient data."""
import errno
from contextlib import closing
from collections import Counter
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian
import pytest
import yaml

from lidc3d import source_audit as audit
from lidc3d.provenance import sha256


def _dicom(path, index, position, patient="SYNTHETIC", series="1.2.3.4", pixels=None, **tags):
    path.parent.mkdir(parents=True, exist_ok=True)
    uid = f"1.2.826.0.1.3680043.10.999.{index}"
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = uid
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    values = dict(SOPClassUID=CTImageStorage, SOPInstanceUID=uid, PatientID=patient,
                  StudyInstanceUID="1.2.3", SeriesInstanceUID=series, Modality="CT",
                  ImagePositionPatient=position, ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
                  PixelSpacing=[.7, .7], Rows=2, Columns=2, SamplesPerPixel=1,
                  PhotometricInterpretation="MONOCHROME2", BitsAllocated=16, BitsStored=16,
                  HighBit=15, PixelRepresentation=1, RescaleSlope=1, RescaleIntercept=-1000,
                  InstanceNumber=index, ImageType=["ORIGINAL", "PRIMARY", "AXIAL"])
    values.update(tags)
    for name, value in values.items():
        setattr(ds, name, value)
    ds.PixelData = np.asarray(pixels if pixels is not None else [[0, 100], [200, 300]], dtype="<i2").tobytes()
    ds.save_as(path, enforce_file_format=True)
    return {"path": str(path.resolve()), "patient_id": patient, "study_uid": "1.2.3",
            "series_uid": series, "sop_uid": str(ds.SOPInstanceUID), "rows": 2, "columns": 2,
            "position": list(position), "orientation": list(map(float, ds.ImageOrientationPatient)),
            "pixel_spacing": [.7, .7], "image_type": list(ds.ImageType), "transfer_syntax": str(meta.TransferSyntaxUID)}


class Fixture:
    def __init__(self, tmp_path, positions=(0, 1)):
        self.root = tmp_path
        self.raw = tmp_path / "raw"
        self.db = tmp_path / "package" / "pylidc.sqlite"
        self.db.parent.mkdir()
        self.inv = tmp_path / "inventory.json"
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        self.config = config_dir / "audit.yaml"
        self.config.write_text((Path(__file__).parents[1] / "configs/source_audit.yaml").read_text(), encoding="utf-8")
        self.rows = [_dicom(self.raw / f"slice-{i}.dcm", i + 1, [0, 0, z]) for i, z in enumerate(positions)]
        self.save_inventory()
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.executescript("""
                CREATE TABLE scans (id INTEGER PRIMARY KEY, patient_id VARCHAR, study_instance_uid VARCHAR, series_instance_uid VARCHAR, sorted_dicom_file_names TEXT);
                CREATE TABLE annotations (id INTEGER PRIMARY KEY, scan_id INTEGER REFERENCES scans(id));
                CREATE TABLE contours (id INTEGER PRIMARY KEY, annotation_id INTEGER REFERENCES annotations(id), inclusion BOOLEAN,
                                       image_z_position FLOAT, dicom_file_name VARCHAR, coords VARCHAR);
                CREATE TABLE zvals (id INTEGER PRIMARY KEY, scan_id INTEGER REFERENCES scans(id), val FLOAT);
                INSERT INTO scans VALUES (1, 'SYNTHETIC', '1.2.3', '1.2.3.4', NULL);
                INSERT INTO annotations VALUES (1, 1);
            """)
            conn.executemany("INSERT INTO zvals VALUES (?, 1, ?)", enumerate(sorted(set(positions)), start=1))

    def save_inventory(self):
        self.inv.write_text(json.dumps({"source_root": str(self.raw), "slices": self.rows, "errors": []}), encoding="utf-8")

    def sql(self, sql, parameters=()):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute(sql, parameters)

    def contour(self, index, name, z, inclusion=1):
        self.sql("INSERT INTO contours VALUES (?, 1, ?, ?, ?, '0,0')", (index, inclusion, z, name))

    def run(self, name="audit", provider=None):
        output = self.root / name
        summary = audit.audit_source(self.inv, self.db, self.config, output, invocation=["synthetic-audit"], pylidc_source_root=provider)
        data = {key: [json.loads(line) for line in (output / f"{key}.jsonl").read_text().splitlines()]
                for key in ("series_audit", "duplicate_positions", "contour_linkage")}
        data["provenance"] = json.loads((output / "provenance.json").read_text())
        assert summary["source_approved"] is False and summary["training_released"] is False
        assert summary["schema_version"] == data["provenance"]["schema_version"] == 4
        assert data["provenance"]["configuration"]["schema_version"] == 1
        assert all(r["schema_version"] == 4 for key in ("series_audit", "duplicate_positions", "contour_linkage") for r in data[key])
        return summary, data


def test_matching_missing_and_equal_counts_mismatched_positions(tmp_path):
    f = Fixture(tmp_path)
    f.rows.append(_dicom(f.raw / "other.dcm", 9, [0, 0, 0], patient="OTHER", series="1.2.3.9"))
    f.save_inventory()
    f.sql("INSERT INTO scans VALUES (2, 'DB_ONLY', '1.2.8', '1.2.8.1', NULL)")
    f.sql("UPDATE zvals SET val=9 WHERE id=2")
    summary, data = f.run()
    assert summary["audit_status"] == "completed"
    assert not summary["source_approved"] and not summary["training_released"]
    assert len(data["series_audit"]) == 3
    series = next(r for r in data["series_audit"] if r["identity"]["patient_id"] == "SYNTHETIC")
    comparison = series["z_comparisons"][0]
    assert comparison["count_delta_ct_minus_db"] == 0
    assert not comparison["sorted_multiset_within_tolerance"]
    assert comparison["db_zvals_without_ct"] == [2]
    assert summary["identity_differences"]["patients"] == {"ct_only": [("OTHER",)], "db_only": [("DB_ONLY",)]}
    assert data["provenance"]["sqlite"]["query_only"] == 1
    assert data["provenance"]["database_columns"]["contours"]


@pytest.mark.parametrize("conflict", [False, True])
def test_repeated_content_and_separate_reference_cardinalities(tmp_path, conflict):
    f = Fixture(tmp_path, (0, 0, 1))
    if conflict:
        ds = pydicom.dcmread(f.rows[1]["path"])
        ds.PixelData = np.full((2, 2), 500, dtype="<i2").tobytes()
        ds.save_as(ds.filename, enforce_file_format=True)
    f.contour(1, "slice-0.dcm", 0, inclusion=0)
    f.contour(2, "unrelated-name", 0)
    summary, data = f.run()
    group = data["duplicate_positions"][0]
    pair = group["pairwise"][0]
    assert summary["audit_status"] == "completed"
    assert pair["raw_pixels_equal"] is not conflict
    assert pair["padding_validity_equal"] is True
    assert pair["hu_equal_on_common_valid_pixels"] is not conflict
    assert len({m["metadata"]["SOPInstanceUID"] for m in group["members"]}) == 2
    assert all(m["sha256_before"] == m["sha256_after"] for m in group["members"])
    references = group["annotation_references"]
    assert references["verified_inspected_member_cardinality"] == ("neither" if conflict else "one")
    assert data["contour_linkage"][0]["status"] == ("exact_link_inspection_unresolved" if conflict else "verified_inspected_source_link")
    assert len(references["z_candidate_members"]) == 2
    assert references["contours"][0]["inclusion"] == 0
    assert data["contour_linkage"][1]["status"] == "z_only_inventory_link"
    assert data["contour_linkage"][1]["verified_inspected_record_id"] is None


@pytest.mark.parametrize("tag,value,padding_equal,hu_equal", [
    ("RescaleSlope", 2, True, False), ("RescaleIntercept", -500, True, False),
    ("PixelPaddingValue", 0, False, True),
])
def test_equal_raw_different_rescale_or_padding(tmp_path, tag, value, padding_equal, hu_equal):
    f = Fixture(tmp_path, (0, 0))
    ds = pydicom.dcmread(f.rows[1]["path"])
    setattr(ds, tag, value)
    if tag == "PixelPaddingValue":
        ds[tag].VR = "SS"
    ds.save_as(ds.filename, enforce_file_format=True)
    summary, data = f.run()
    pair = data["duplicate_positions"][0]["pairwise"][0]
    assert summary["audit_status"] == "completed"
    assert pair["raw_pixels_equal"] is True
    assert pair["padding_validity_equal"] is padding_equal
    assert pair["hu_equal_on_common_valid_pixels"] is hu_equal


def test_filename_collisions_missing_literal_sop_and_z_only(tmp_path):
    f = Fixture(tmp_path, (0, 0, 1))
    # Same basename inside a series is ambiguous, even if z could choose one.
    f.rows.append(_dicom(f.raw / "nested" / "slice-0.dcm", 4, [0, 0, 0]))
    f.rows.append(_dicom(f.raw / "elsewhere" / "slice-2.dcm", 5, [0, 0, 8], series="1.2.3.8"))
    f.save_inventory()
    f.contour(1, "slice-0.dcm", 0)
    f.contour(2, "absent", 9)
    f.contour(3, "absent", 1)
    f.contour(4, f.rows[2]["sop_uid"], 1)
    f.contour(5, "slice-2.dcm", 0)
    f.contour(6, f.rows[2]["sop_uid"] + ".dcm", 1)
    _, data = f.run()
    links = data["contour_linkage"]
    assert [c["status"] for c in links] == ["ambiguous_exact_link", "missing_link", "z_only_inventory_link",
                                           "exact_inventory_record_link_source_not_rechecked", "filename_position_contradiction", "z_only_inventory_link"]
    assert len(links[0]["exact_filename_record_ids"]) == 2
    assert len(links[2]["z_candidate_record_ids"]) == 1
    assert len(links[4]["exact_filename_record_ids"]) == 1  # Other series cannot collide.
    assert links[5]["literal_sop_uid_record_ids"] == []  # No suffix stripping to invent UID evidence.


@pytest.mark.parametrize("references,expected", [(0, "neither"), (1, "one"), (2, "multiple")])
def test_repeated_group_reference_cardinality(tmp_path, references, expected):
    f = Fixture(tmp_path, (0, 0))
    for i in range(references):
        f.contour(i + 1, f"slice-{i}.dcm", 0)
    _, data = f.run()
    ref = data["duplicate_positions"][0]["annotation_references"]
    assert ref["verified_inspected_member_cardinality"] == expected
    assert ref["exact_candidate_cardinality"] == expected


def test_duplicate_database_identity_and_sops_do_not_verify(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    f.sql("INSERT INTO scans VALUES (2, 'SYNTHETIC', '1.2.3', '1.2.3.4', NULL)")
    f.rows[1]["sop_uid"] = f.rows[0]["sop_uid"]
    f.save_inventory()
    f.contour(1, "slice-0.dcm", 0)
    summary, data = f.run()
    assert summary["duplicate_database_identities"][0]["scan_ids"] == [1, 2]
    assert len(summary["duplicate_sop_ids"]) == 1
    assert data["contour_linkage"][0]["verified_inspected_record_id"] is None


@pytest.mark.parametrize("sql", ["ALTER TABLE contours RENAME TO unsupported_contours",
                                  "UPDATE annotations SET scan_id=90", "UPDATE zvals SET val=NULL"])
def test_unsupported_schema_and_relational_integrity_fail(tmp_path, sql):
    f = Fixture(tmp_path)
    f.sql(sql)
    summary, data = f.run()
    assert summary["audit_status"] == "failed"
    assert summary["operational_errors"]
    assert data["provenance"]["database_schema"]
    assert len(list((tmp_path / "audit").iterdir())) == 5
    assert all(v["bytes_unchanged"] for v in data["provenance"]["inputs"].values())


@pytest.mark.parametrize("orientation,code", [([1, 0, 0, 0, 0, 1], "non_axial_geometry_unsupported"),
                                              ([0, 0, 0, 0, 0, 0], "invalid_orientation")])
def test_unsupported_geometry_and_lps_z_not_projection(tmp_path, orientation, code):
    f = Fixture(tmp_path)
    for r in f.rows:
        r["orientation"] = orientation
    f.save_inventory()
    f.contour(1, "slice-0.dcm", 0)
    summary, data = f.run()
    geometry = data["series_audit"][0]["geometry"]
    assert summary["audit_status"] == "completed"
    assert code in geometry["findings"]
    assert data["contour_linkage"][0]["status"] == "exact_link_geometry_unresolved"
    if code == "non_axial_geometry_unsupported":
        assert geometry["positions"][1]["normal_projection_mm"] == 0
        assert geometry["positions"][1]["lps_z_mm"] == 1


def test_full_position_not_just_z_and_tolerance_chain(tmp_path):
    f = Fixture(tmp_path, (0, 0, 0))
    f.rows[1]["position"][0] = 1
    f.rows[2]["position"][0] = 2
    f.save_inventory()
    _, data = f.run()
    assert data["duplicate_positions"] == []
    assert "in_plane_drift_unsupported" in data["series_audit"][0]["geometry"]["findings"]
    for i, r in enumerate(f.rows):
        r["position"][0] = i * .00000075
    f.save_inventory()
    _, data = f.run("chain")
    assert data["duplicate_positions"][0]["tolerance_chain_ambiguous"]


def test_decoding_failure_is_not_successful_comparison(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    ds = pydicom.dcmread(f.rows[1]["path"])
    ds.PixelData = b"\0\0"
    ds.save_as(ds.filename, enforce_file_format=True)
    summary, data = f.run()
    assert summary["audit_status"] == "failed"
    member = data["duplicate_positions"][0]["members"][1]
    assert member["decode_status"] == "failed" and member["error"]
    pair = data["duplicate_positions"][0]["pairwise"][0]
    assert pair["raw_pixels_equal"] is None and pair["hu_equal_on_common_valid_pixels"] is None


def test_only_repeated_members_opened_and_hashed(tmp_path, monkeypatch):
    f = Fixture(tmp_path, (0, 0, 1, 2))
    opened, hashed = [], []
    read = audit.pydicom.dcmread
    digest = audit.sha256

    def tracked_read(path, *args, **kwargs):
        opened.append(str(path))
        return read(path, *args, **kwargs)

    def tracked_hash(path):
        if Path(path).suffix == ".dcm":
            hashed.append(str(path))
        return digest(path)

    monkeypatch.setattr(audit.pydicom, "dcmread", tracked_read)
    monkeypatch.setattr(audit, "sha256", tracked_hash)
    summary, data = f.run()
    expected = {r["path"] for r in f.rows[:2]}
    assert set(opened) == expected and set(hashed) == expected
    assert summary["counts"]["inspected_dicom_records"] == 2
    assert "NOT cryptographically verified" in data["provenance"]["dicom_hash_scope"]
    assert "pylidc" not in sys.modules


def test_readonly_inputs_and_sqlite_write_refused(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    inputs = [f.inv, f.db, f.config, *(Path(r["path"]) for r in f.rows)]
    hashes = {p: sha256(p) for p in inputs}
    modes = {p: p.stat().st_mode for p in inputs}
    try:
        for path in inputs:
            path.chmod(0o444)
        with audit.readonly_database(f.db) as conn:
            assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("UPDATE scans SET patient_id='changed'")
        summary, data = f.run()
        assert summary["audit_status"] == "completed"
        assert hashes == {p: sha256(p) for p in inputs}
        assert all(v["bytes_unchanged"] for v in data["provenance"]["inputs"].values())
        assert list(f.db.parent.iterdir()) == [f.db]
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


def test_overwrite_and_protected_paths(tmp_path):
    f = Fixture(tmp_path)
    f.run()
    before = {p.name: sha256(p) for p in (tmp_path / "audit").iterdir()}
    with pytest.raises(FileExistsError):
        f.run()
    assert before == {p.name: sha256(p) for p in (tmp_path / "audit").iterdir()}
    for output in (f.raw / "output", f.db.parent / "output", f.config.parent / "output", Path(audit.__file__).parent / "audit-output"):
        with pytest.raises(ValueError, match="protected"):
            audit.audit_source(f.inv, f.db, f.config, output)
        assert not output.exists()


def _directory_link(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # A junction exercises resolve() protection on Windows without symlink privilege.
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        if result.returncode:
            pytest.fail("Neither directory symlinks nor junctions available: " + result.stderr)


def test_resolved_symlink_protection_in_both_directions(tmp_path):
    f = Fixture(tmp_path)
    alias = tmp_path / "raw-alias"
    _directory_link(alias, f.raw)
    try:
        with pytest.raises(ValueError, match="protected"):
            audit.audit_source(f.inv, f.db, f.config, alias / "output")
    finally:
        alias.rmdir() if os.name == "nt" and not alias.is_symlink() else alias.unlink()
    # An input alias into an external tree is rejected before any DICOM read.
    external = tmp_path / "external"
    external.mkdir()
    link = f.raw / "linked"
    _directory_link(link, external)
    try:
        r = _dicom(external / "input.dcm", 10, [0, 0, 2])
        r["path"] = str(link / "input.dcm")
        f.rows.append(r)
        f.save_inventory()
        with pytest.raises(ValueError, match="Raw-source boundary violation"):
            audit.audit_source(f.inv, f.db, f.config, external / "output")
    finally:
        link.rmdir() if os.name == "nt" and not link.is_symlink() else link.unlink()


def test_deterministic_findings_despite_inventory_order(tmp_path):
    f = Fixture(tmp_path, (0, 0, 1, 1))
    f.contour(2, "slice-2.dcm", 1)
    f.contour(1, "slice-0.dcm", 0)
    first, _ = f.run("first")
    f.rows.reverse()
    f.save_inventory()
    second, _ = f.run("second")
    assert first == second
    for name in ("series_audit", "duplicate_positions", "contour_linkage"):
        assert (tmp_path / "first" / f"{name}.jsonl").read_bytes() == (tmp_path / "second" / f"{name}.jsonl").read_bytes()


def test_input_tamper_detected_after_audit(tmp_path, monkeypatch):
    f = Fixture(tmp_path)
    original = audit._run

    def mutate(*args):
        original(*args)
        f.inv.write_text(f.inv.read_text() + " ")

    monkeypatch.setattr(audit, "_run", mutate)
    summary, data = f.run()
    assert summary["audit_status"] == "failed"
    assert not data["provenance"]["inputs"]["inventory"]["bytes_unchanged"]


def test_sidecar_refusal(tmp_path):
    f = Fixture(tmp_path)
    Path(str(f.db) + "-wal").write_bytes(b"synthetic-sidecar")
    summary, _ = f.run()
    assert summary["audit_status"] == "failed"
    assert "sidecar" in str(summary["operational_errors"])


def test_cli_line_budget_and_failure_exit(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-m", "lidc3d.cli", "audit-source", "--inventory", str(f.inv),
               "--annotation-db", str(f.db), "--config", str(f.config), "--out", str(tmp_path / "cli")]
    success = subprocess.run(command, env=env, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    assert len(success.stdout.splitlines()) <= 8 and not success.stderr
    failure = subprocess.run(command, env=env, capture_output=True, text=True)
    assert failure.returncode == 1
    assert len(failure.stdout.splitlines()) <= 8 and not failure.stderr
    assert "Source approved: no" in failure.stdout


def test_invalid_config_refused_without_output(tmp_path):
    f = Fixture(tmp_path)
    cfg = yaml.safe_load(f.config.read_text())
    cfg["tolerances"]["z_atol_mm"] = float("nan")
    f.config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="finite"):
        f.run()
    assert not (tmp_path / "audit").exists()


def test_bad_padding_or_hu_never_claims_success_for_that_comparison(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    ds = pydicom.dcmread(f.rows[1]["path"])
    ds.add_new(0x00280121, "SS", 100)  # Range limit without the required padding value.
    ds.save_as(ds.filename, enforce_file_format=True)
    summary, data = f.run()
    pair = data["duplicate_positions"][0]["pairwise"][0]
    assert summary["audit_status"] == "failed"
    assert pair["raw_pixels_equal"] is True
    assert pair["padding_validity_equal"] is None
    assert pair["hu_equal_on_common_valid_pixels"] is None
    del ds.PixelPaddingRangeLimit
    del ds.RescaleSlope
    ds.save_as(ds.filename, enforce_file_format=True)
    summary, data = f.run("missing-rescale")
    pair = data["duplicate_positions"][0]["pairwise"][0]
    assert summary["audit_status"] == "failed"
    assert pair["padding_validity_equal"] is True
    assert pair["hu_equal_on_common_valid_pixels"] is None


def test_empty_valid_intersection_and_padding_ranges(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    for row in f.rows:
        ds = pydicom.dcmread(row["path"])
        ds.add_new(0x00280120, "SS", 300)
        ds.add_new(0x00280121, "SS", 0)
        ds.save_as(ds.filename, enforce_file_format=True)
    summary, data = f.run()
    pair = data["duplicate_positions"][0]["pairwise"][0]
    assert summary["audit_status"] == "completed"
    assert pair["common_valid_pixel_count"] == 0
    assert pair["padding_validity_equal"] is True
    assert pair["hu_equal_on_common_valid_pixels"] is None


def test_inspected_inventory_mismatch_invalidates_link_and_cli_fails(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    f.contour(1, "slice-0.dcm", 0)
    ds = pydicom.dcmread(f.rows[0]["path"])
    ds.ImagePositionPatient = [0, 0, 99]
    ds.save_as(ds.filename, enforce_file_format=True)
    summary, data = f.run()
    assert summary["audit_status"] == "failed"
    assert data["contour_linkage"][0]["status"] == "filename_position_contradiction"
    assert data["contour_linkage"][0]["verified_inspected_record_id"] is None
    assert data["duplicate_positions"][0]["annotation_references"]["verified_inspected_member_cardinality"] == "neither"
    command = [sys.executable, "-B", "-m", "lidc3d.cli", "audit-source", "--inventory", str(f.inv),
               "--annotation-db", str(f.db), "--config", str(f.config), "--out", str(tmp_path / "failed-cli")]
    result = subprocess.run(command, env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}, capture_output=True, text=True)
    assert result.returncode == 1 and "Audit failed" in result.stdout
    assert not result.stderr and len(result.stdout.splitlines()) <= 8


def test_resolved_alias_cannot_escape_protected_tree(tmp_path):
    f = Fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    inner, alias = f.raw / "escape", tmp_path / "alias"
    _directory_link(inner, outside)
    _directory_link(alias, f.raw)
    try:
        with pytest.raises(ValueError, match="protected"):
            audit.audit_source(f.inv, f.db, f.config, alias / "escape" / "out")
        assert not (outside / "out").exists()
    finally:
        alias.rmdir() if os.name == "nt" and not alias.is_symlink() else alias.unlink()
        inner.rmdir() if os.name == "nt" and not inner.is_symlink() else inner.unlink()


def test_tolerances_apply_to_z_multisets_and_positions(tmp_path):
    f = Fixture(tmp_path)
    f.sql("UPDATE zvals SET val=val+0.00005")
    _, data = f.run()
    assert data["series_audit"][0]["z_comparisons"][0]["sorted_multiset_within_tolerance"]
    config = yaml.safe_load(f.config.read_text())
    config["tolerances"]["z_atol_mm"] = 0
    f.config.write_text(yaml.safe_dump(config))
    _, data = f.run("exact-z")
    assert not data["series_audit"][0]["z_comparisons"][0]["sorted_multiset_within_tolerance"]


@pytest.mark.parametrize("change,flag", [
    ({"orientation": [1, 0, 0, 0, -1, 0]}, "variable_orientation"),
    ({"pixel_spacing": [.8, .7]}, "variable_pixel_spacing"),
    ({"position": [0, 0, float("nan")]}, "invalid_position"),
    ({"image_type": ["LOCALIZER"]}, "localizer_unsupported"),
    ({"columns": 3}, "variable_dimensions"),
])
def test_invalid_or_variable_inventory_geometry(tmp_path, change, flag):
    f = Fixture(tmp_path)
    f.rows[1].update(change)
    f.save_inventory()
    summary, data = f.run()
    assert summary["audit_status"] == "completed"
    assert flag in data["series_audit"][0]["geometry"]["findings"]


def test_unreadable_duplicate_and_dimension_error_are_unavailable(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    Path(f.rows[1]["path"]).write_bytes(b"not a DICOM")
    summary, data = f.run()
    assert summary["audit_status"] == "failed"
    assert data["duplicate_positions"][0]["pairwise"][0]["raw_pixels_equal"] is None
    f.rows[1] = _dicom(Path(f.rows[1]["path"]), 2, [0, 0, 0])
    f.rows[1]["rows"] = 10
    f.save_inventory()
    summary, data = f.run("dimension-error")
    assert summary["audit_status"] == "failed"
    assert data["duplicate_positions"][0]["pairwise"][0]["raw_pixels_equal"] is None


def test_invalid_inclusion_preserved_and_missing_geometry_visible(tmp_path):
    f = Fixture(tmp_path)
    f.contour(1, "slice-0.dcm", 0, inclusion=2)
    del f.rows[1]["orientation"]
    f.save_inventory()
    summary, data = f.run()
    assert summary["audit_status"] == "failed"
    assert data["contour_linkage"][0]["original"]["inclusion"] == 2
    assert not data["contour_linkage"][0]["inclusion_valid"]
    assert "invalid_orientation" in data["series_audit"][0]["geometry"]["findings"]


def test_ordinary_exact_inventory_link_never_reopens_or_verifies_source(tmp_path, monkeypatch):
    f = Fixture(tmp_path)
    f.contour(1, "slice-0.dcm", 0)
    digest = audit.sha256

    def forbid_dicom_read(*args, **kwargs):
        pytest.fail("Ordinary inventory slice must not be opened")

    def reject_dicom_hash(path):
        assert Path(path).suffix != ".dcm", "Ordinary DICOM must not be hashed"
        return digest(path)

    monkeypatch.setattr(audit.pydicom, "dcmread", forbid_dicom_read)
    monkeypatch.setattr(audit, "sha256", reject_dicom_hash)
    summary, data = f.run()
    link = data["contour_linkage"][0]
    assert link["status"] == "exact_inventory_record_link_source_not_rechecked"
    assert "verified" not in link["status"]
    assert link["exact_inventory_record_id"] == 1
    assert link["verified_inspected_record_id"] is None
    assert link["exact_candidate_inspections"] == []
    assert summary["counts"]["inspected_dicom_records"] == 0
    assert summary["contour_linkage_counts"] == {link["status"]: 1}
    assert "inventory byte immutability only" in data["provenance"]["inventory_hash_scope"]


def test_complete_inspected_evidence_links_both_members_without_selecting(tmp_path):
    f = Fixture(tmp_path, (0, 0, 1))
    for i in range(3):
        f.contour(i + 1, f"slice-{i}.dcm", 0 if i < 2 else 1)
    before = {r["path"]: sha256(r["path"]) for r in f.rows}
    summary, data = f.run()
    links = data["contour_linkage"]
    assert [r["status"] for r in links] == ["verified_inspected_source_link", "verified_inspected_source_link",
                                           "exact_inventory_record_link_source_not_rechecked"]
    group = data["duplicate_positions"][0]
    assert len(group["members"]) == 2
    assert group["resolution"] == "unresolved_no_slice_selected"
    assert group["annotation_references"]["verified_inspected_members"] == [1, 2]
    assert group["annotation_references"]["verified_inspected_member_cardinality"] == "multiple"
    for member in group["members"]:
        assert member["source_opened"] and member["bytes_unchanged"]
        assert member["inspection_blockers"] == []
        assert not member["warnings"] and not member["pydicom_log_warnings"]
    assert data["series_audit"][0]["ct_count"] == 3
    assert summary["counts"]["ct_records"] == 3
    assert before == {r["path"]: sha256(r["path"]) for r in f.rows}


@pytest.mark.parametrize("conflict", ["raw", "padding", "hu", "unavailable"])
def test_pixel_conflicts_block_every_member_without_a_preference(tmp_path, conflict):
    f = Fixture(tmp_path, (0, 0, 0))
    for i in range(3):
        f.contour(i + 1, f"slice-{i}.dcm", 0)
    ds = pydicom.dcmread(f.rows[2]["path"])
    if conflict == "raw":
        ds.PixelData = np.full((2, 2), 500, dtype="<i2").tobytes()
    elif conflict == "padding":
        ds.add_new(0x00280120, "SS", 0)
    elif conflict == "hu":
        ds.RescaleIntercept = -999
    else:
        ds.PixelData = b"\0\0"
    ds.save_as(ds.filename, enforce_file_format=True)
    summary, data = f.run()
    links = data["contour_linkage"]
    assert all(r["status"] == "exact_link_inspection_unresolved" for r in links)
    assert all(r["verified_inspected_record_id"] is None for r in links)
    group = data["duplicate_positions"][0]
    assert len(group["members"]) == 3 and len(group["pairwise"]) == 3
    assert group["resolution"] == "unresolved_no_slice_selected"
    assert group["annotation_references"]["exact_candidate_cardinality"] == "multiple"
    assert group["annotation_references"]["verified_inspected_member_cardinality"] == "neither"
    assert all(m["inspection_blockers"] for m in group["members"])
    assert summary["inspection_blocker_counts"]


def test_real_pydicom_pixel_warning_is_preserved_and_blocks_verification(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    for i in range(2):
        f.contour(i + 1, f"slice-{i}.dcm", 0)
    ds = pydicom.dcmread(f.rows[1]["path"])
    ds.PixelData += b"\0\0"  # Valid samples plus excess padding: pydicom warns while decoding.
    ds.save_as(ds.filename, enforce_file_format=True)
    summary, data = f.run()
    group = data["duplicate_positions"][0]
    assert summary["audit_status"] == "completed"  # Warnings are diagnostic, never approval.
    assert group["members"][1]["warnings"]
    assert group["members"][1]["decode_status"] == "decoded"
    assert "unresolved_read_or_decode_warning" in group["members"][1]["inspection_blockers"]
    assert group["pairwise"][0]["raw_pixels_equal"] is True
    assert all(r["status"] == "exact_link_inspection_unresolved" for r in data["contour_linkage"])
    assert all(r["verified_inspected_record_id"] is None for r in data["contour_linkage"])


def test_logger_only_pydicom_warning_blocks_verification_and_handler_is_removed(tmp_path, monkeypatch):
    f = Fixture(tmp_path, (0, 0))
    f.contour(1, "slice-0.dcm", 0)
    original = audit.pydicom.dcmread
    logger = logging.getLogger("pydicom")
    handlers_before = logger.handlers[:]

    def warned_read(path, *args, **kwargs):
        logger.warning("Synthetic identity warning from pydicom logger")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(audit.pydicom, "dcmread", warned_read)
    summary, data = f.run()
    member = data["duplicate_positions"][0]["members"][0]
    assert summary["audit_status"] == "completed"
    assert not member["warnings"]
    assert member["pydicom_log_warnings"] == [{"level": "WARNING", "message": "Synthetic identity warning from pydicom logger"}]
    assert data["contour_linkage"][0]["status"] == "exact_link_inspection_unresolved"
    assert logger.handlers == handlers_before


@pytest.mark.parametrize("path_kind", ["outside", "traversal_outside", "traversal_inside", "relative"])
def test_raw_boundary_rejected_before_any_dicom_read_or_hash(tmp_path, monkeypatch, path_kind):
    f = Fixture(tmp_path, (0, 0))
    outside = _dicom(tmp_path / "outside.dcm", 3, [0, 0, 0])
    paths = {"outside": outside["path"], "traversal_outside": str(f.raw / ".." / "outside.dcm"),
             "traversal_inside": str(f.raw / "nested" / ".." / "slice-1.dcm"), "relative": "raw/slice-1.dcm"}
    f.rows[1]["path"] = paths[path_kind]
    f.save_inventory()
    digest = audit.sha256

    def forbidden_read(*args, **kwargs):
        pytest.fail("Boundary violations must fail before reading any DICOM")

    def guarded_hash(path):
        assert Path(path).suffix != ".dcm", "Boundary violation must not hash a DICOM"
        return digest(path)

    monkeypatch.setattr(audit.pydicom, "dcmread", forbidden_read)
    monkeypatch.setattr(audit, "sha256", guarded_hash)
    with pytest.raises(ValueError, match="Raw-source boundary violation"):
        f.run()
    assert not (tmp_path / "audit").exists()


def _file_symlink(link, target):
    try:
        link.symlink_to(target)
    except OSError as error:
        unavailable = {errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP}
        if error.errno in unavailable or getattr(error, "winerror", None) == 1314:
            pytest.skip(f"OS cannot create file symlinks with current privileges: {error}")
        raise


def test_file_symlink_outside_raw_root_never_opened(tmp_path, monkeypatch):
    f = Fixture(tmp_path, (0, 0))
    outside = _dicom(tmp_path / "outside.dcm", 3, [0, 0, 0])
    link = f.raw / "outside-link.dcm"
    _file_symlink(link, Path(outside["path"]))
    f.rows[1]["path"] = str(link)
    f.save_inventory()
    digest = audit.sha256

    def forbid_read(*args, **kwargs):
        pytest.fail("Escaping symlink must not open any DICOM")

    def guarded_hash(path):
        assert Path(path).suffix != ".dcm"
        return digest(path)

    monkeypatch.setattr(audit.pydicom, "dcmread", forbid_read)
    monkeypatch.setattr(audit, "sha256", guarded_hash)
    with pytest.raises(ValueError, match="resolves outside source_root"):
        f.run()


def test_contained_file_symlink_is_valid(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    link = f.raw / "contained-link.dcm"
    _file_symlink(link, Path(f.rows[0]["path"]))
    f.rows[0]["path"] = str(link)
    f.save_inventory()
    f.contour(1, link.name, 0)
    summary, data = f.run()
    assert summary["audit_status"] == "completed"
    assert data["contour_linkage"][0]["status"] == "verified_inspected_source_link"


def test_inspection_rechecks_boundary_after_inventory_preparation(tmp_path, monkeypatch):
    f = Fixture(tmp_path, (0, 0))
    outside = _dicom(tmp_path / "external" / "outside.dcm", 3, [0, 0, 0])
    prepare = audit._prepare_inventory

    def changed_inventory(inventory):
        root, rows = prepare(inventory)
        rows[0]["path"] = outside["path"]
        return root, rows

    def forbid_read(*args, **kwargs):
        pytest.fail("Changed path must be rejected immediately before inspection")

    monkeypatch.setattr(audit, "_prepare_inventory", changed_inventory)
    monkeypatch.setattr(audit.pydicom, "dcmread", forbid_read)
    summary, _ = f.run()
    assert summary["audit_status"] == "failed"
    assert "Raw-source boundary violation" in str(summary["operational_errors"])


@pytest.mark.parametrize("tag,value", [("PatientID", "OTHER"), ("StudyInstanceUID", "1.2.9"),
                                      ("SeriesInstanceUID", "1.2.9"), ("SOPInstanceUID", "1.2.9")])
def test_inspected_identity_must_match_inventory_and_database_context(tmp_path, tag, value):
    f = Fixture(tmp_path, (0, 0))
    f.contour(1, "slice-0.dcm", 0)
    ds = pydicom.dcmread(f.rows[0]["path"])
    setattr(ds, tag, value)
    ds.save_as(ds.filename, enforce_file_format=True)
    summary, data = f.run()
    assert summary["audit_status"] == "failed"
    assert data["contour_linkage"][0]["status"] == "exact_link_inspection_unresolved"
    assert data["contour_linkage"][0]["verified_inspected_record_id"] is None


def test_changed_dicom_hash_blocks_inspected_source_link(tmp_path, monkeypatch):
    f = Fixture(tmp_path, (0, 0))
    f.contour(1, "slice-0.dcm", 0)
    read = audit.pydicom.dcmread

    def mutated_after_read(path, *args, **kwargs):
        ds = read(path, *args, **kwargs)
        if Path(path) == Path(f.rows[0]["path"]):
            with Path(path).open("ab") as handle:  # Deliberate synthetic tampering by the test only.
                handle.write(b"\0\0")
        return ds

    monkeypatch.setattr(audit.pydicom, "dcmread", mutated_after_read)
    summary, data = f.run()
    member = data["duplicate_positions"][0]["members"][0]
    assert summary["audit_status"] == "failed"
    assert not member["bytes_unchanged"]
    assert "source_bytes_changed_or_hash_unavailable" in member["inspection_blockers"]
    assert data["contour_linkage"][0]["status"] == "exact_link_inspection_unresolved"


def test_actual_group_geometry_checked_in_addition_to_inventory(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    f.contour(1, "slice-0.dcm", 0)
    for i, row in enumerate(f.rows):
        ds = pydicom.dcmread(row["path"])
        # Both are individually within inventory tolerance, but they disagree with each other.
        ds.ImageOrientationPatient = [1, 0.00009 * (-1 if i else 1), 0, 0, 1, 0]
        ds.save_as(ds.filename, enforce_file_format=True)
    _, data = f.run()
    assert "variable_orientation" in data["duplicate_positions"][0]["members"][0]["inspected_group_geometry_findings"]
    assert data["contour_linkage"][0]["status"] == "exact_link_inspection_unresolved"


def test_contour_z_compared_directly_to_inspected_position(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    f.contour(1, "slice-0.dcm", -0.0001)
    ds = pydicom.dcmread(f.rows[0]["path"])
    ds.ImagePositionPatient = [0, 0, 0.0000005]
    ds.save_as(ds.filename, enforce_file_format=True)
    _, data = f.run()
    link = data["contour_linkage"][0]
    assert link["status"] == "filename_position_contradiction"
    assert link["verified_inspected_record_id"] is None


def _provider(tmp_path):
    root = tmp_path / "provider-source"
    root.mkdir()
    for name in (*audit.PROVIDER_FILES, "__init__.py"):
        # Deliberately invalid Python: fingerprinting must not parse or execute it.
        (root / name).write_text("not executable Python !\nraise AssertionError('do not import')\n", encoding="utf-8")
    return root


def _legacy(f, names="legacy-a,legacy-b"):
    f.sql("UPDATE scans SET sorted_dicom_file_names=? WHERE id=1", (names,))


def test_crosswalk_numeric_order_provider_hashes_and_no_import(tmp_path, monkeypatch):
    import builtins
    f = Fixture(tmp_path, (-10, 0, 10))
    root = _provider(tmp_path)
    _legacy(f, "003.dcm,001.dcm,002.dcm")
    f.sql("DELETE FROM zvals")
    for row in ((10, 1, 10), (90, 1, -10), (3, 1, 0)):
        f.sql("INSERT INTO zvals VALUES (?, ?, ?)", row)
    for i, (name, z) in enumerate((("003.dcm", -10), ("001.dcm", 0), ("002.dcm", 10)), 1):
        f.contour(i, name, z)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"pylidc", "sqlalchemy"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    summary, data = f.run(provider=root)
    assert summary["audit_status"] == "completed"
    assert summary["crosswalk_status_counts"][audit.CROSSWALK_SUCCESS] == 3
    assert set(summary["crosswalk_status_counts"]) == set(audit.CROSSWALK_STATUSES)
    assert set(summary["crosswalk_scan_finding_counts"]) == set(audit.SCAN_FINDINGS)
    scan = data["series_audit"][0]["database_crosswalks"][0]
    assert scan["original_sorted_dicom_file_names"] == "003.dcm,001.dcm,002.dcm"
    assert [z["id"] for z in scan["ordered_finite_zvals"]] == [90, 3, 10]
    assert scan["filename_count"] == scan["zval_count"] == 3
    assert scan["index_mapping_valid"] and not scan["findings"]
    for i, link in enumerate(data["contour_linkage"]):
        assert link["status"] == audit.CROSSWALK_SUCCESS
        assert link["crosswalk"]["matched_legacy_index"] == i
        assert link["crosswalk"]["contour_minus_paired_z_mm"] == 0
        assert link["crosswalk"]["cardinalities"]["inventory_records"] == 1
        assert link["crosswalk_inventory_record_id"] is not None
        assert link["verified_inspected_record_id"] is link["exact_inventory_record_id"] is None
    provider = data["provenance"]["provider_source"]
    assert provider["status"] == "valid" and provider["bytes_unchanged"]
    assert provider["resolved_root"] == str(root.resolve())
    for name, info in provider["files"].items():
        assert info["sha256_before"] == info["sha256_after"] == sha256(root / name)
        assert info["bytes_unchanged"]
    assert not data["provenance"]["pylidc_imported_or_loader_executed"]
    assert summary["counts"]["inspected_dicom_records"] == 0


@pytest.mark.parametrize("names,contour,z,status,finding", [
    ("legacy-a", "legacy-a", 0, "crosswalk_index_count_mismatch", "filename_zval_count_mismatch"),
    ("legacy-a,legacy-b,extra", "extra", 1, "crosswalk_index_count_mismatch", "filename_zval_count_mismatch"),
    ("legacy-a,", "legacy-a", 0, "crosswalk_invalid", "empty_legacy_filename"),
    ("legacy-a,legacy-a", "legacy-a", 0, "crosswalk_filename_nonunique", "nonunique_legacy_filename"),
    ("legacy-a,legacy-b", "absent", 0, "crosswalk_filename_missing", None),
    ("legacy-a,legacy-b", None, 0, "crosswalk_filename_missing", None),
    ("legacy-a,legacy-b", "", 0, "crosswalk_filename_missing", None),
    ("legacy-a,legacy-b", "legacy-a", 1, "crosswalk_contour_z_contradiction", None),
    ("legacy-a,legacy-b", "legacy-a", None, "crosswalk_invalid_contour_z", None),
    (None, "legacy-a", 0, "crosswalk_unavailable", "legacy_list_not_text"),
    (" legacy-a,LEGACY-B", "legacy-a", 0, "crosswalk_filename_missing", None),
    (" legacy-a,LEGACY-B", "legacy-b", 1, "crosswalk_filename_missing", None),
    (" legacy-a,LEGACY-B", " legacy-a", 0, audit.CROSSWALK_SUCCESS, None),
])
def test_crosswalk_statuses_and_exact_tokens(tmp_path, names, contour, z, status, finding):
    f = Fixture(tmp_path)
    _legacy(f, names)
    f.contour(1, contour, z)
    summary, data = f.run(provider=_provider(tmp_path))
    crosswalk = data["contour_linkage"][0]["crosswalk"]
    assert crosswalk["status"] == status
    assert summary["crosswalk_status_counts"][status] == 1
    if finding:
        assert summary["crosswalk_scan_finding_counts"][finding] == 1
    if status == "crosswalk_filename_nonunique":
        assert crosswalk["matched_legacy_indices"] == [0, 1]
        assert crosswalk["matched_legacy_index"] is crosswalk["paired_zval"] is None
        assert len(crosswalk["indexed_candidate_evidence"]) == 2
        assert len(crosswalk["inventory_record_ids"]) == 2
    if status == "crosswalk_contour_z_contradiction":
        assert crosswalk["contour_minus_paired_z_mm"] == 1
        assert crosswalk["inventory_record_ids"] != data["contour_linkage"][0]["z_candidate_record_ids"]


@pytest.mark.parametrize("value", [None, float("inf"), float("-inf"), "invalid"])
def test_crosswalk_invalid_zvals_retained(tmp_path, value):
    f = Fixture(tmp_path)
    _legacy(f)
    f.sql("UPDATE zvals SET val=? WHERE id=2", (value,))
    f.contour(1, "legacy-a", 0)
    summary, data = f.run(provider=_provider(tmp_path))
    assert summary["audit_status"] == "failed"
    assert summary["counts"]["database_zvals"] == 2
    assert summary["crosswalk_scan_finding_counts"]["nonfinite_zval"] == 1
    assert data["contour_linkage"][0]["status"] == "crosswalk_invalid"
    assert data["series_audit"][0]["database_crosswalks"][0]["invalid_zvals"][0]["id"] == 2
    assert data["series_audit"][0]["z_comparisons"][0]["invalid_zvals"][0]["id"] == 2


def test_crosswalk_equal_zvals_only_affected_indices_ambiguous(tmp_path):
    f = Fixture(tmp_path)
    _legacy(f, "a,b,c")
    f.sql("INSERT INTO zvals VALUES (3,1,0)")
    for i, (name, z) in enumerate((("a", 0), ("b", 0), ("c", 1)), 1):
        f.contour(i, name, z)
    summary, data = f.run(provider=_provider(tmp_path))
    assert [link["status"] for link in data["contour_linkage"]] == [
        "crosswalk_paired_z_ambiguous", "crosswalk_paired_z_ambiguous", audit.CROSSWALK_SUCCESS]
    assert summary["crosswalk_scan_finding_counts"]["equal_zvals"] == 1
    scan = data["series_audit"][0]["database_crosswalks"][0]
    assert scan["ambiguous_z_indices"] == [0, 1]
    assert scan["index_mapping_valid"] and not scan["all_indices_positionally_unambiguous"]


@pytest.mark.parametrize("positions,expected", [((0, 1), audit.CROSSWALK_SUCCESS),
                                               ((1, 2), "crosswalk_z_inventory_missing"),
                                               ((0, 0, 1), "crosswalk_z_inventory_ambiguous")])
def test_crosswalk_inventory_cardinality_never_selects_or_verifies(tmp_path, positions, expected):
    f = Fixture(tmp_path, positions)
    f.sql("DELETE FROM zvals")
    f.sql("INSERT INTO zvals VALUES (1,1,0)")
    f.sql("INSERT INTO zvals VALUES (2,1,1)")
    _legacy(f)
    f.contour(1, "legacy-a", 0)
    _, data = f.run(provider=_provider(tmp_path))
    link = data["contour_linkage"][0]
    assert link["status"] == expected
    assert link["verified_inspected_record_id"] is None
    if positions == (0, 0, 1):
        group = data["duplicate_positions"][0]
        assert group["resolution"] == "unresolved_no_slice_selected"
        assert len(group["members"]) == len(link["crosswalk"]["inventory_record_ids"]) == 2
        assert group["annotation_references"]["crosswalk_candidate_cardinality"] == "multiple"
        assert group["annotation_references"]["verified_inspected_members"] == []
        assert "preferred" not in json.dumps(data)


@pytest.mark.parametrize("literal_kind", ["filename", "sop"])
def test_literal_precedence_and_crosswalk_disagreement_kept(tmp_path, literal_kind):
    f = Fixture(tmp_path)
    name = "slice-0.dcm" if literal_kind == "filename" else f.rows[0]["sop_uid"]
    _legacy(f, "other," + name)
    f.contour(1, name, 0)
    _, data = f.run(provider=_provider(tmp_path))
    link = data["contour_linkage"][0]
    assert link["status"] == audit.LITERAL_CROSSWALK_CONFLICT
    assert link["literal_linkage_status"] == "exact_inventory_record_link_source_not_rechecked"
    assert link["crosswalk"]["status"] == "crosswalk_contour_z_contradiction"
    assert link["literal_crosswalk_candidate_disagreement"]
    assert link["crosswalk_inventory_record_id"] is None
    assert link["exact_inventory_record_id"] is link["verified_inspected_record_id"] is None


def test_crosswalk_gated_without_provider_and_generic_fallback(tmp_path):
    f = Fixture(tmp_path)
    _legacy(f)
    f.contour(1, "legacy-a", 0)
    f.contour(2, "absent", 8)
    summary, data = f.run()
    assert summary["provider_source_status"] == "not_supplied"
    assert [c["status"] for c in data["contour_linkage"]] == ["z_only_inventory_link", "missing_link"]
    assert all(c["crosswalk"]["status"] == "crosswalk_unavailable" for c in data["contour_linkage"])


@pytest.mark.parametrize("failure", ["missing", "unreadable", "changed", "removed"])
def test_provider_failures_invalidate_artifact_set(tmp_path, monkeypatch, failure):
    f = Fixture(tmp_path)
    _legacy(f)
    f.contour(1, "legacy-a", 0)
    root = _provider(tmp_path)
    target = root / "Scan.py"
    if failure == "missing":
        target.unlink()
    elif failure == "unreadable":
        digest = audit.sha256

        def unreadable(path):
            if Path(path) == target:
                raise PermissionError("synthetic denied read")
            return digest(path)

        monkeypatch.setattr(audit, "sha256", unreadable)
    else:
        run = audit._run

        def mutate(*args):
            run(*args)
            if failure == "changed":
                target.write_text("changed synthetic provider source", encoding="utf-8")
            else:
                target.unlink()

        monkeypatch.setattr(audit, "_run", mutate)
    summary, data = f.run(provider=root)
    assert summary["audit_status"] == "failed"
    assert summary["provider_source_status"] == "invalid"
    assert not data["provenance"]["provider_source"]["files"]["Scan.py"]["bytes_unchanged"]
    assert "provider_source_provenance_invalid" in str(summary["operational_errors"])


def test_provider_escaped_file_is_never_hashed(tmp_path, monkeypatch):
    f = Fixture(tmp_path)
    root = _provider(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("external bytes", encoding="utf-8")
    (root / "Scan.py").unlink()
    _file_symlink(root / "Scan.py", outside)
    digest = audit.sha256

    def guarded(path):
        assert Path(path).resolve() != outside.resolve()
        return digest(path)

    monkeypatch.setattr(audit, "sha256", guarded)
    summary, data = f.run(provider=root)
    assert summary["provider_source_status"] == "invalid"
    assert "escapes" in str(data["provenance"]["provider_source"]["files"]["Scan.py"])


@pytest.mark.parametrize("alias", [False, True])
def test_provider_output_directory_protected(tmp_path, alias):
    f = Fixture(tmp_path)
    root = _provider(tmp_path)
    output = root / "audit"
    if alias:
        link = tmp_path / "provider-alias"
        _directory_link(link, root)
        output = link / "audit"
    with pytest.raises(ValueError, match="protected"):
        audit.audit_source(f.inv, f.db, f.config, output, pylidc_source_root=root)
    assert not output.exists()


def test_crosswalk_deterministic_inventory_and_db_insert_order(tmp_path):
    f = Fixture(tmp_path)
    root = _provider(tmp_path)
    _legacy(f)
    f.contour(1, "legacy-a", 0)
    f.contour(2, "legacy-b", 1)
    _, first = f.run("first", root)
    f.rows.reverse()
    f.save_inventory()
    f.sql("DELETE FROM zvals")
    f.sql("INSERT INTO zvals VALUES (2,1,1)")
    f.sql("INSERT INTO zvals VALUES (1,1,0)")
    f.sql("DELETE FROM contours")
    f.contour(2, "legacy-b", 1)
    f.contour(1, "legacy-a", 0)
    _, second = f.run("second", root)
    for artifact in ("series_audit", "contour_linkage", "duplicate_positions"):
        assert first[artifact] == second[artifact]
        assert (f.root / "first" / f"{artifact}.jsonl").read_bytes() == (f.root / "second" / f"{artifact}.jsonl").read_bytes()


def test_malformed_and_orphan_rows_are_preserved(tmp_path):
    f = Fixture(tmp_path)
    _legacy(f)
    f.sql("INSERT INTO scans VALUES (2,NULL,'study','series','a')")
    f.sql("INSERT INTO annotations VALUES (2,2)")
    f.sql("INSERT INTO annotations VALUES (3,99)")
    f.sql("INSERT INTO zvals VALUES (3,99,0)")
    f.sql("INSERT INTO contours VALUES (1,2,1,0,'a','0,0')")
    f.sql("INSERT INTO contours VALUES (2,3,1,NULL,'orphan','0,0')")
    summary, data = f.run(provider=_provider(tmp_path))
    assert summary["audit_status"] == "failed"
    assert len(data["contour_linkage"]) == summary["counts"]["database_contours"] == 2
    assert {s for row in data["series_audit"] for s in row["database_scan_ids"]} == {1, 2}
    assert any(row["identity_status"] == "orphan_contour_context" for row in data["series_audit"])
    assert all("row" in finding for finding in data["provenance"]["database_integrity_findings"])


def test_crosswalk_documentation_migration():
    guide = (Path(__file__).parents[1] / "docs/Source_Audit.md").read_text(encoding="utf-8")
    for phrase in ("schema_version: 4", "configuration remains schema version **1**", "No in-place migration",
                   "v2 `z_only_inventory_link`", "--pylidc-source-root", "lower-InstanceNumber", "NOT EXECUTED"):
        assert phrase in guide


def test_required_legacy_list_column(tmp_path):
    f = Fixture(tmp_path)
    f.sql("ALTER TABLE scans DROP COLUMN sorted_dicom_file_names")
    summary, _ = f.run(provider=_provider(tmp_path))
    assert summary["audit_status"] == "failed"
    assert "scans.sorted_dicom_file_names needs text affinity" in str(summary["operational_errors"])


def test_duplicate_names_elsewhere_invalidate_whole_scan_mapping(tmp_path):
    f = Fixture(tmp_path, (0, 1, 2))
    _legacy(f, "a,b,b")
    f.contour(1, "a", 0)
    _, data = f.run(provider=_provider(tmp_path))
    assert data["contour_linkage"][0]["status"] == "crosswalk_invalid"


def test_crosswalk_uses_joined_scan_not_other_scan_names(tmp_path):
    f = Fixture(tmp_path)
    _legacy(f, "a,b")
    f.sql("INSERT INTO scans VALUES (2,'SYNTHETIC','1.2.3','1.2.3.4','b,a')")
    f.sql("INSERT INTO annotations VALUES (2,2)")
    f.sql("INSERT INTO zvals VALUES (3,2,0)")
    f.sql("INSERT INTO zvals VALUES (4,2,1)")
    f.contour(1, "a", 0)
    f.sql("INSERT INTO contours VALUES (2,2,1,1,'a','0,0')")
    _, data = f.run(provider=_provider(tmp_path))
    links = data["contour_linkage"]
    assert [c["crosswalk"]["matched_legacy_index"] for c in links] == [0, 1]
    assert [c["crosswalk"]["paired_zval"]["id"] for c in links] == [1, 4]
    assert all(c["status"] == "crosswalk_identity_ambiguous" for c in links)
    assert links[0]["crosswalk"]["inventory_record_ids"] != links[1]["crosswalk"]["inventory_record_ids"]


def test_crosswalk_inventory_scope_db_only_and_ct_only_retained(tmp_path):
    f = Fixture(tmp_path)
    _legacy(f)
    f.rows.append(_dicom(f.raw / "other" / "slice.dcm", 3, [0, 0, 0], patient="CT_ONLY", series="1.2.3.9"))
    f.save_inventory()
    f.sql("INSERT INTO scans VALUES (2,'DB_ONLY','study','series','legacy-a')")
    f.sql("INSERT INTO annotations VALUES (2,2)")
    f.sql("INSERT INTO zvals VALUES (3,2,0)")
    f.contour(1, "legacy-a", 0)
    f.sql("INSERT INTO contours VALUES (2,2,1,0,'legacy-a','0,0')")
    summary, data = f.run(provider=_provider(tmp_path))
    links = {c["contour_id"]: c for c in data["contour_linkage"]}
    assert links[1]["status"] == audit.CROSSWALK_SUCCESS
    assert len(links[1]["crosswalk"]["inventory_record_ids"]) == 1
    assert links[2]["status"] == "crosswalk_z_inventory_missing"
    assert {s["identity_status"] for s in data["series_audit"]} == {"ct_only", "db_only", "matched"}
    assert summary["counts"]["series_union"] == 3


@pytest.mark.parametrize("case,expected", [
    ("sop_collision", "crosswalk_identity_ambiguous"),
    ("geometry", "crosswalk_geometry_unresolved"),
    ("inspected", "crosswalk_inspected_group_ambiguous"),
])
def test_crosswalk_context_cannot_promote_inspected_evidence(tmp_path, case, expected):
    f = Fixture(tmp_path, (0, 0.0000005) if case == "inspected" else (0, 1))
    _legacy(f)
    if case == "sop_collision":
        f.rows[1]["sop_uid"] = f.rows[0]["sop_uid"]
    elif case == "geometry":
        f.rows[0]["orientation"] = [0, 0, 0, 0, 0, 0]
    else:
        config = yaml.safe_load(f.config.read_text())
        config["tolerances"]["z_atol_mm"] = 0
        f.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    f.save_inventory()
    f.contour(1, "legacy-a", 0)
    _, data = f.run(provider=_provider(tmp_path))
    link = data["contour_linkage"][0]
    assert link["status"] == expected
    assert link["verified_inspected_record_id"] is None
    if case == "inspected":
        assert len(data["duplicate_positions"][0]["members"]) == 2
        assert len(link["crosswalk"]["inventory_record_ids"]) == 1
        assert len(link["crosswalk"]["intersecting_repeated_position_record_ids"]) == 2


@pytest.mark.parametrize("z,status", [(0.000099, audit.CROSSWALK_SUCCESS),
                                      (0.000101, "crosswalk_contour_z_contradiction")])
def test_crosswalk_direct_z_tolerance_and_signed_delta(tmp_path, z, status):
    f = Fixture(tmp_path)
    _legacy(f)
    f.contour(1, "legacy-a", -z)
    _, data = f.run(provider=_provider(tmp_path))
    link = data["contour_linkage"][0]
    assert link["status"] == status
    assert link["crosswalk"]["contour_minus_paired_z_mm"] == -z


def test_cli_provider_option_enables_crosswalk(tmp_path):
    f = Fixture(tmp_path)
    _legacy(f)
    f.contour(1, "legacy-a", 0)
    root = _provider(tmp_path)
    output = tmp_path / "cli-audit"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-B", "-m", "lidc3d.cli", "audit-source",
                           "--inventory", str(f.inv), "--annotation-db", str(f.db),
                           "--config", str(f.config), "--out", str(output),
                           "--pylidc-source-root", str(root)], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert len(proc.stdout.splitlines()) <= 4
    summary = json.loads((output / "audit_summary.json").read_text())
    assert summary["crosswalk_status_counts"][audit.CROSSWALK_SUCCESS] == 1


@pytest.mark.parametrize("positions,volume_status,uniform", [
    ((0, 1, 2), "uniform_spacing_geometry_diagnostic_only_not_released", True),
    ((0, 1, 3), "requires_explicit_coordinate_aware_resampling_not_released", False),
])
@pytest.mark.parametrize("kind", ["legacy", "literal_filename", "literal_sop"])
def test_point_linkage_separate_from_volume_geometry(tmp_path, positions, volume_status, uniform, kind):
    f = Fixture(tmp_path, positions)
    name = {"legacy": "legacy-a", "literal_filename": "slice-0.dcm", "literal_sop": f.rows[0]["sop_uid"]}[kind]
    _legacy(f, name + ",legacy-b,legacy-c")
    f.contour(1, name, 0)
    summary, data = f.run(provider=_provider(tmp_path))
    link, series = data["contour_linkage"][0], data["series_audit"][0]
    assert summary["audit_status"] == "completed"
    assert link["crosswalk"]["status"] == audit.CROSSWALK_SUCCESS
    if kind == "legacy":
        assert link["status"] == audit.CROSSWALK_SUCCESS
        assert link["crosswalk_inventory_record_id"] == 1
        assert link["exact_inventory_record_id"] is None
    else:
        assert link["status"] == "exact_inventory_record_link_source_not_rechecked"
        assert link["exact_inventory_record_id"] == 1
        assert link["crosswalk_inventory_record_id"] is None
    assert link["verified_inspected_record_id"] is None
    assessment = link["geometry_assessment"]
    assert assessment == series["geometry_assessment"]
    assert assessment["series"] == {k: series["geometry"][k] for k in ("status", "findings")}
    assert assessment["linkage"]["status"] == "diagnostic_point_linkage_compatible"
    assert assessment["linkage"]["blocking_reasons"] == []
    assert assessment["series"]["findings"] == ([] if uniform else ["irregular_spacing"])
    assert assessment["volume_reconstruction"] == {
        "status": volume_status, "uniform_spacing_geometry_eligible": uniform,
        "ordinary_uniform_spacing_preparation_eligible": False, "approved": False}
    assert [p["lps_z_mm"] for p in series["geometry"]["positions"]] == list(positions)
    assert not summary["coverage"]["xml_reference_standard_replaced"]
    assert not summary["coverage"]["negative_labels_created"]
    assert "unresolved" in summary["coverage"]["reader_identity"]


@pytest.mark.parametrize("finding", [
    "invalid_position", "invalid_orientation", "non_axial_geometry_unsupported",
    "invalid_pixel_spacing", "invalid_dimensions", "localizer_unsupported",
    "variable_orientation", "variable_pixel_spacing", "variable_dimensions",
    "in_plane_drift_unsupported", "future_unknown_geometry_finding",
])
@pytest.mark.parametrize("irregular", [False, True])
def test_every_unsafe_and_unknown_geometry_finding_blocks_point_linkage(tmp_path, monkeypatch, finding, irregular):
    f = Fixture(tmp_path, (0, 1, 3) if irregular else (0, 1, 2))
    _legacy(f, "legacy-a,slice-1.dcm,legacy-c")
    f.contour(1, "legacy-a", 0)
    f.contour(2, "slice-1.dcm", 1)
    original = audit._geometry

    def diagnosed(rows, tol):
        geometry = original(rows, tol)
        geometry["findings"] = sorted(set(geometry["findings"]) | {finding})
        geometry["status"] = "unsupported_or_ambiguous"
        return geometry

    monkeypatch.setattr(audit, "_geometry", diagnosed)
    _, data = f.run(provider=_provider(tmp_path))
    assert [c["status"] for c in data["contour_linkage"]] == [
        "crosswalk_geometry_unresolved", "exact_link_geometry_unresolved"]
    for link in data["contour_linkage"]:
        assert link["crosswalk"]["status"] == "crosswalk_geometry_unresolved"
        assessment = link["geometry_assessment"]
        assert assessment["linkage"]["status"] == "geometry_unresolved"
        assert finding in assessment["linkage"]["blocking_reasons"]
        assert assessment["linkage"]["nonblocking_findings"] == (["irregular_spacing"] if irregular else [])
        assert assessment["volume_reconstruction"]["status"] == "geometry_unresolved_not_released"
        assert not assessment["volume_reconstruction"]["uniform_spacing_geometry_eligible"]
        assert link["crosswalk_inventory_record_id"] is link["exact_inventory_record_id"] is link["verified_inspected_record_id"] is None


@pytest.mark.parametrize("status,findings", [
    ("future_geometry_status", []), ("future_geometry_status", ["irregular_spacing"]),
    ("unsupported_or_ambiguous", []), ("diagnostic_geometry_consistent", ["irregular_spacing"]),
    ("no_ct_records", []),
])
def test_unknown_or_inconsistent_geometry_status_fails_closed(status, findings):
    assessment = audit._geometry_assessment({"status": status, "findings": findings})
    assert assessment["linkage"]["status"] == "geometry_unresolved"
    assert "series_geometry_status_unresolved" in assessment["linkage"]["blocking_reasons"]
    assert not assessment["volume_reconstruction"]["uniform_spacing_geometry_eligible"]
    assert not assessment["volume_reconstruction"]["ordinary_uniform_spacing_preparation_eligible"]


@pytest.mark.parametrize("case,expected", [
    ("contradiction", "crosswalk_contour_z_contradiction"),
    ("multiple", "crosswalk_z_inventory_ambiguous"),
    ("repeated", "crosswalk_inspected_group_ambiguous"),
    ("provider_absent", "crosswalk_unavailable"),
    ("provider_missing", "crosswalk_unavailable"),
    ("provider_changed", audit.CROSSWALK_SUCCESS),
])
def test_irregular_spacing_does_not_bypass_existing_evidence_gates(tmp_path, monkeypatch, case, expected):
    positions = (0, 0, 1, 3) if case == "multiple" else (0, 0.0000005, 1, 3) if case == "repeated" else (0, 1, 3)
    f = Fixture(tmp_path, positions)
    _legacy(f, ",".join(f"legacy-{i}" for i in range(len(set(positions)))))
    f.contour(1, "legacy-0", 1 if case == "contradiction" else 0)
    provider = None if case == "provider_absent" else _provider(tmp_path)
    if case == "repeated":
        config = yaml.safe_load(f.config.read_text())
        config["tolerances"]["z_atol_mm"] = 0
        f.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    if case == "provider_missing":
        (provider / "Scan.py").unlink()
    if case == "provider_changed":
        original = audit._run

        def mutate(*args):
            original(*args)
            (provider / "Scan.py").write_text("synthetic changed source", encoding="utf-8")

        monkeypatch.setattr(audit, "_run", mutate)
    summary, data = f.run(provider=provider)
    link = data["contour_linkage"][0]
    assert link["geometry_assessment"]["series"]["findings"] == ["irregular_spacing"]
    assert link["crosswalk"]["status"] == expected
    assert link["verified_inspected_record_id"] is None
    assert not link["geometry_assessment"]["volume_reconstruction"]["ordinary_uniform_spacing_preparation_eligible"]
    if case in ("provider_missing", "provider_changed"):
        assert summary["audit_status"] == "failed"
        assert summary["provider_source_status"] == "invalid"
        assert "provisional" in link["crosswalk"]["evidence_scope"]
    elif case == "provider_absent":
        assert link["status"] == "z_only_inventory_link"
    else:
        assert link["status"] == expected
        assert link["crosswalk_inventory_record_id"] is None


@pytest.mark.parametrize("with_provider", [False, True])
def test_literal_crosswalk_conflict_clears_inspected_verification(tmp_path, with_provider):
    f = Fixture(tmp_path, (0, 0, 1, 2))
    _legacy(f, "other,slice-0.dcm,third")
    f.contour(1, "slice-0.dcm", 0)
    summary, data = f.run(provider=_provider(tmp_path) if with_provider else None)
    link = data["contour_linkage"][0]
    assert summary["audit_status"] == "completed"
    assert link["literal_linkage_status"] == "verified_inspected_source_link"
    assert link["status"] == audit.LITERAL_CROSSWALK_CONFLICT
    assert link["literal_crosswalk_candidate_disagreement"]
    assert link["exact_filename_record_ids"] == [1]
    assert link["crosswalk"]["inventory_record_ids"] == [3]
    assert link["all_candidate_record_ids"] == [1, 2, 3]
    assert link["exact_inventory_record_id"] is link["verified_inspected_record_id"] is link["crosswalk_inventory_record_id"] is None
    group = data["duplicate_positions"][0]
    assert group["annotation_references"]["verified_inspected_members"] == []
    assert len(group["members"]) == 2 and group["resolution"] == "unresolved_no_slice_selected"
    assert summary["contour_linkage_counts"] == {audit.LITERAL_CROSSWALK_CONFLICT: 1}


def test_agreeing_crosswalk_keeps_limited_literal_inspected_verification(tmp_path):
    f = Fixture(tmp_path, (0, 0.0000005, 1))
    _legacy(f, "slice-0.dcm,other,third")
    f.contour(1, "slice-0.dcm", 0)
    config = yaml.safe_load(f.config.read_text())
    config["tolerances"]["z_atol_mm"] = 0
    f.config.write_text(yaml.safe_dump(config), encoding="utf-8")
    _, data = f.run(provider=_provider(tmp_path))
    link = data["contour_linkage"][0]
    assert link["literal_linkage_status"] == link["status"] == "verified_inspected_source_link"
    assert not link["literal_crosswalk_candidate_disagreement"]
    assert link["exact_filename_record_ids"] == link["crosswalk"]["inventory_record_ids"] == [1]
    assert link["verified_inspected_record_id"] == link["exact_inventory_record_id"] == 1
    assert link["crosswalk_inventory_record_id"] is None
    assert data["duplicate_positions"][0]["resolution"] == "unresolved_no_slice_selected"


def test_schema_v4_counts_and_deterministic_irregular_artifacts(tmp_path):
    f = Fixture(tmp_path, (0, 1, 3))
    provider = _provider(tmp_path)
    _legacy(f, "legacy-a,slice-0.dcm,legacy-c")
    contours = [(1, "legacy-a", 0), (2, "slice-0.dcm", 0), (3, "missing", 1),
                (4, "legacy-c", 1), (5, "legacy-c", 3)]
    for c in contours:
        f.contour(*c)
    summary, first = f.run("first", provider)
    links = first["contour_linkage"]
    assert summary["contour_linkage_counts"] == dict(Counter(c["status"] for c in links))
    assert {k: v for k, v in summary["crosswalk_status_counts"].items() if v} == dict(Counter(c["crosswalk"]["status"] for c in links))
    assert sum(summary["contour_linkage_counts"].values()) == sum(summary["crosswalk_status_counts"].values()) == len(links) == 5
    assert summary["contour_linkage_geometry_counts"] == dict(Counter(c["geometry_assessment"]["linkage"]["status"] for c in links))
    assert summary["series_volume_reconstruction_counts"] == dict(Counter(s["geometry_assessment"]["volume_reconstruction"]["status"] for s in first["series_audit"]))
    assert first["series_audit"][0]["contour_linkage_counts"] == summary["contour_linkage_counts"]
    assert summary["contour_linkage_counts"][audit.LITERAL_CROSSWALK_CONFLICT] == 1
    f.rows.reverse()
    f.save_inventory()
    f.sql("DELETE FROM zvals")
    for row in ((3, 1, 3), (2, 1, 1), (1, 1, 0)):
        f.sql("INSERT INTO zvals VALUES (?,?,?)", row)
    f.sql("DELETE FROM contours")
    for c in reversed(contours):
        f.contour(*c)
    again, second = f.run("second", provider)
    assert summary == again
    for artifact in ("series_audit", "contour_linkage", "duplicate_positions"):
        assert first[artifact] == second[artifact]
        assert (f.root / "first" / f"{artifact}.jsonl").read_bytes() == (f.root / "second" / f"{artifact}.jsonl").read_bytes()


def test_schema_v4_artifacts_contain_no_nonfinite_json_number_tokens(tmp_path):
    f = Fixture(tmp_path)
    _legacy(f)
    f.rows[1]["position"][2] = float("nan")
    f.save_inventory()
    f.sql("UPDATE zvals SET val=? WHERE id=2", (float("inf"),))
    f.contour(1, "legacy-a", float("-inf"))
    summary, data = f.run(provider=_provider(tmp_path))
    assert summary["audit_status"] == "failed"

    def reject_constant(token):
        pytest.fail("Nonstandard JSON number token: " + token)

    for name in ("series_audit.jsonl", "contour_linkage.jsonl", "duplicate_positions.jsonl", "provenance.json", "audit_summary.json"):
        text = (f.root / "audit" / name).read_text(encoding="utf-8")
        records = text.splitlines() if name.endswith("jsonl") else [text]
        for record in records:
            parsed = json.loads(record, parse_constant=reject_constant)
            assert parsed["schema_version"] == 4
            json.dumps(parsed, allow_nan=False)
    assert data["contour_linkage"][0]["original"]["image_z_position"] == {"nonfinite": "-inf"}
    assert data["series_audit"][0]["geometry"]["positions"][1]["position_lps_mm"][2] == {"nonfinite": "nan"}
