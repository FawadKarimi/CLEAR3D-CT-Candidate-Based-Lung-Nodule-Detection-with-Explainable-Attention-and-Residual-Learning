"""Synthetic CT and temporary SQLite only; no pylidc installation or patient data."""
import json
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
        with sqlite3.connect(self.db) as conn:
            conn.executescript("""
                CREATE TABLE scans (id INTEGER PRIMARY KEY, patient_id VARCHAR, study_instance_uid VARCHAR, series_instance_uid VARCHAR);
                CREATE TABLE annotations (id INTEGER PRIMARY KEY, scan_id INTEGER REFERENCES scans(id));
                CREATE TABLE contours (id INTEGER PRIMARY KEY, annotation_id INTEGER REFERENCES annotations(id), inclusion BOOLEAN,
                                       image_z_position FLOAT, dicom_file_name VARCHAR, coords VARCHAR);
                CREATE TABLE zvals (id INTEGER PRIMARY KEY, scan_id INTEGER REFERENCES scans(id), val FLOAT);
                INSERT INTO scans VALUES (1, 'SYNTHETIC', '1.2.3', '1.2.3.4');
                INSERT INTO annotations VALUES (1, 1);
            """)
            conn.executemany("INSERT INTO zvals VALUES (?, 1, ?)", enumerate(sorted(set(positions)), start=1))

    def save_inventory(self):
        self.inv.write_text(json.dumps({"source_root": str(self.raw), "slices": self.rows, "errors": []}), encoding="utf-8")

    def sql(self, sql, parameters=()):
        with sqlite3.connect(self.db) as conn:
            conn.execute(sql, parameters)

    def contour(self, index, name, z, inclusion=1):
        self.sql("INSERT INTO contours VALUES (?, 1, ?, ?, ?, '0,0')", (index, inclusion, z, name))

    def run(self, name="audit"):
        output = self.root / name
        summary = audit.audit_source(self.inv, self.db, self.config, output, invocation=["synthetic-audit"])
        data = {key: [json.loads(line) for line in (output / f"{key}.jsonl").read_text().splitlines()]
                for key in ("series_audit", "duplicate_positions", "contour_linkage")}
        data["provenance"] = json.loads((output / "provenance.json").read_text())
        return summary, data


def test_matching_missing_and_equal_counts_mismatched_positions(tmp_path):
    f = Fixture(tmp_path)
    f.rows.append(_dicom(f.raw / "other.dcm", 9, [0, 0, 0], patient="OTHER", series="1.2.3.9"))
    f.save_inventory()
    f.sql("INSERT INTO scans VALUES (2, 'DB_ONLY', '1.2.8', '1.2.8.1')")
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
    assert references["verified_member_cardinality"] == "one"
    assert len(references["z_candidate_members"]) == 2
    assert references["contours"][0]["inclusion"] == 0
    assert data["contour_linkage"][1]["status"] == "z_only_unverified"
    assert data["contour_linkage"][1]["verified_record_id"] is None


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
    assert [c["status"] for c in links] == ["ambiguous_exact_link", "missing_link", "z_only_unverified",
                                           "verified_exact_inventory_link", "filename_position_contradiction", "z_only_unverified"]
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
    assert ref["verified_member_cardinality"] == expected
    assert ref["exact_candidate_cardinality"] == expected


def test_duplicate_database_identity_and_sops_do_not_verify(tmp_path):
    f = Fixture(tmp_path, (0, 0))
    f.sql("INSERT INTO scans VALUES (2, 'SYNTHETIC', '1.2.3', '1.2.3.4')")
    f.rows[1]["sop_uid"] = f.rows[0]["sop_uid"]
    f.save_inventory()
    f.contour(1, "slice-0.dcm", 0)
    summary, data = f.run()
    assert summary["duplicate_database_identities"][0]["scan_ids"] == [1, 2]
    assert len(summary["duplicate_sop_ids"]) == 1
    assert data["contour_linkage"][0]["verified_record_id"] is None


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
    # An input file reached via a link to an external tree protects the target tree too.
    external = tmp_path / "external"
    external.mkdir()
    link = f.raw / "linked"
    _directory_link(link, external)
    try:
        r = _dicom(external / "input.dcm", 10, [0, 0, 2])
        r["path"] = str(link / "input.dcm")
        f.rows.append(r)
        f.save_inventory()
        with pytest.raises(ValueError, match="protected"):
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
    assert data["contour_linkage"][0]["status"] == "exact_link_source_unverified"
    assert data["contour_linkage"][0]["verified_record_id"] is None
    assert data["duplicate_positions"][0]["annotation_references"]["verified_member_cardinality"] == "neither"
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
