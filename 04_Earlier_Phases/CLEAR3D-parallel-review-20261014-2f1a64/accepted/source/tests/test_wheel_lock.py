import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/make_wheel_lock.py"
spec = importlib.util.spec_from_file_location("wheel_lock", SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def make_wheel(tmp_path, filename="example_pkg-1.0-py3-none-any.whl", records=None):
    p = tmp_path / filename
    if records is None:
        records = [("example_pkg-1.0.dist-info/METADATA", "Name: Example.Pkg\nVersion: 1.0\n")]
    with zipfile.ZipFile(p, "w") as z:
        for name, body in records:
            z.writestr(name, body)
    return p


def test_vendored_metadata_ignored_exact_archive_digest(tmp_path):
    p = make_wheel(tmp_path, records=[
        ("example_pkg-1.0.dist-info/METADATA", "Name: Example.Pkg\nVersion: 1.0\n"),
        ("example_pkg/_vendor/a-7.dist-info/METADATA", "Name: a\nVersion: 7\n"),
        ("example_pkg/_vendor/b-9.dist-info/METADATA", "not valid identity metadata"),
        ("payload.bin", b"payload"),
    ])
    assert helper.lock_lines(tmp_path) == [
        "example-pkg==1.0 --hash=sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    ]


@pytest.mark.parametrize("records", [
    [],
    [("vendor/example_pkg-1.0.dist-info/METADATA", "Name: Example.Pkg\nVersion: 1.0\n")],
    [("example_pkg-1.0.dist-info/METADATA", "Name: Example.Pkg\nVersion: 1.0\n"),
     ("other-2.dist-info/METADATA", "Name: other\nVersion: 2\n")],
])
def test_missing_or_ambiguous_top_level(tmp_path, records):
    make_wheel(tmp_path, records=records)
    with pytest.raises(ValueError, match="one top-level"):
        helper.lock_lines(tmp_path)


def test_duplicate_selected_zip_entry_rejected(tmp_path):
    record = ("example_pkg-1.0.dist-info/METADATA", "Name: Example.Pkg\nVersion: 1.0\n")
    with pytest.warns(UserWarning, match="Duplicate name"):
        make_wheel(tmp_path, records=[record, record])
    with pytest.raises(ValueError, match="one top-level"):
        helper.lock_lines(tmp_path)


@pytest.mark.parametrize("body", [
    "Name: another\nVersion: 1.0\n", "Name: Example.Pkg\nVersion: 2.0\n",
])
def test_filename_metadata_identity_mismatch(tmp_path, body):
    make_wheel(tmp_path, records=[("example_pkg-1.0.dist-info/METADATA", body)])
    with pytest.raises(ValueError, match="filename and metadata identity mismatch"):
        helper.lock_lines(tmp_path)


def test_duplicate_normalized_package_names(tmp_path):
    make_wheel(tmp_path)
    make_wheel(tmp_path, "example_pkg-2.0-py3-none-any.whl", [
        ("example_pkg-2.0.dist-info/METADATA", "Name: example-pkg\nVersion: 2.0\n")])
    with pytest.raises(ValueError, match="Duplicate package wheel: example-pkg"):
        helper.lock_lines(tmp_path)


@pytest.mark.parametrize("body", [
    "Version: 1.0\n", "Name: Example.Pkg\n",
    "Name: Example.Pkg\nName: Example.Pkg\nVersion: 1.0\n",
    "Name: Example.Pkg\nVersion: 1.0\nVersion: 1.0\n",
])
def test_missing_or_duplicate_identity_headers(tmp_path, body):
    make_wheel(tmp_path, records=[("example_pkg-1.0.dist-info/METADATA", body)])
    with pytest.raises(ValueError, match="one metadata Name and Version"):
        helper.lock_lines(tmp_path)


def test_directory_identity_mismatch(tmp_path):
    make_wheel(tmp_path, records=[("other-1.0.dist-info/METADATA", "Name: Example.Pkg\nVersion: 1.0\n")])
    with pytest.raises(ValueError, match="directory identity mismatch"):
        helper.lock_lines(tmp_path)


def test_invalid_filename_rejected(tmp_path):
    make_wheel(tmp_path, "invalid.whl")
    with pytest.raises(ValueError):
        helper.lock_lines(tmp_path)


def test_build_tag_and_equivalent_pep440_version(tmp_path):
    make_wheel(tmp_path, "example_pkg-1.0-2-py3-none-any.whl", [
        ("example_pkg-1.0.dist-info/METADATA", "Name: EXAMPLE-pkg\nVersion: 1.0.0\n")])
    assert helper.lock_lines(tmp_path)[0].startswith("example-pkg==1.0.0 --hash=sha256:")


def test_cli_does_not_overwrite_output(tmp_path):
    wheelhouse = tmp_path / "wheels"; wheelhouse.mkdir(); make_wheel(wheelhouse)
    output = tmp_path / "lock"; output.write_text("preserve\n")
    p = subprocess.run([sys.executable, "-B", str(SCRIPT), str(wheelhouse), str(output)], capture_output=True)
    assert p.returncode != 0
    assert output.read_text() == "preserve\n"


def test_validation_failure_leaves_no_lock(tmp_path):
    wheelhouse = tmp_path / "wheels"; wheelhouse.mkdir(); make_wheel(wheelhouse, records=[])
    output = tmp_path / "lock"
    p = subprocess.run([sys.executable, "-B", str(SCRIPT), str(wheelhouse), str(output)], capture_output=True)
    assert p.returncode != 0
    assert not output.exists()
