import json
from pathlib import Path
import pytest
from lidc3d.hash_inventory import enrich_inventory
from lidc3d.provenance import sha256


def make_inventory(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    slices = []
    for index in range(3):
        path = raw / f"{index}.dcm"
        path.write_bytes(f"slice-{index}".encode())
        slices.append({"path": str(path), "series_uid": "series"})
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({
        "source_root": str(raw), "slices": slices, "xml": [], "errors": [],
    }))
    return inventory, raw


def test_hash_inventory_and_resume_from_complete_journal(tmp_path):
    inventory, _ = make_inventory(tmp_path)
    journal = tmp_path / "hashes.jsonl"
    first = enrich_inventory(inventory, tmp_path / "hashed1.json", journal, 0)
    assert first["computed_this_run"] == 3
    assert first["reused_from_journal"] == 0
    second = enrich_inventory(inventory, tmp_path / "hashed2.json", journal, 0)
    assert second["computed_this_run"] == 0
    assert second["reused_from_journal"] == 3
    data = json.loads((tmp_path / "hashed2.json").read_text())
    assert all(row["sha256"] == sha256(row["path"]) for row in data["slices"])


def test_hash_inventory_recomputes_changed_file(tmp_path):
    inventory, raw = make_inventory(tmp_path)
    journal = tmp_path / "hashes.jsonl"
    enrich_inventory(inventory, tmp_path / "hashed1.json", journal, 0)
    (raw / "1.dcm").write_bytes(b"changed-longer")
    result = enrich_inventory(inventory, tmp_path / "hashed2.json", journal, 0)
    assert result["computed_this_run"] == 1
    assert result["reused_from_journal"] == 2


def test_hash_inventory_rejects_paths_outside_source_root(tmp_path):
    inventory, _ = make_inventory(tmp_path)
    outside = tmp_path / "outside.dcm"
    outside.write_bytes(b"x")
    data = json.loads(inventory.read_text())
    data["slices"][0]["path"] = str(outside)
    inventory.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="outside source_root"):
        enrich_inventory(
            inventory, tmp_path / "hashed.json",
            tmp_path / "hashes.jsonl", 0)


def test_hash_inventory_does_not_overwrite_output(tmp_path):
    inventory, _ = make_inventory(tmp_path)
    output = tmp_path / "hashed.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        enrich_inventory(
            inventory, output, tmp_path / "hashes.jsonl", 0)
    assert output.read_text() == "keep"
