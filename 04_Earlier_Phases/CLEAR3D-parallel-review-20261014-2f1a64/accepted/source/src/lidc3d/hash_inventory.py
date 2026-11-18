"""Resumable SHA-256 enrichment for an existing DICOM header inventory."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .provenance import sha256


def _load_journal(path):
    records = {}
    path = Path(path)
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid hash journal line {number}") from error
            key = row.get("path")
            if not key:
                raise ValueError(f"Invalid hash journal path at line {number}")
            # A later record supersedes an earlier one when a source file was
            # intentionally replaced between resumptions.
            records[key] = row
    return records


def enrich_inventory(inventory_path, output_path, journal_path, progress_every=1000):
    inventory_path = Path(inventory_path).resolve()
    output_path = Path(output_path).resolve()
    journal_path = Path(journal_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    inventory = json.loads(inventory_path.read_text())
    root = Path(inventory["source_root"]).resolve()
    rows = inventory.get("slices", [])
    if not rows:
        raise ValueError("Inventory contains no CT slices")
    paths = [str(Path(row["path"]).resolve()) for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Inventory contains duplicate DICOM paths")
    if any(not Path(path).is_relative_to(root) for path in paths):
        raise ValueError("Inventory contains a path outside source_root")

    cached = _load_journal(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    reused = computed = 0
    with journal_path.open("a", encoding="utf-8", buffering=1) as journal:
        for index, (row, resolved) in enumerate(zip(rows, paths), 1):
            path = Path(resolved)
            before = path.stat()
            prior = cached.get(resolved)
            if (prior and prior.get("bytes") == before.st_size
                    and prior.get("mtime_ns") == before.st_mtime_ns
                    and isinstance(prior.get("sha256"), str)
                    and len(prior["sha256"]) == 64):
                digest = prior["sha256"]
                reused += 1
            else:
                digest = sha256(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (
                        after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Source changed while hashing: {resolved}")
                record = {
                    "path": resolved, "bytes": after.st_size,
                    "mtime_ns": after.st_mtime_ns, "sha256": digest,
                }
                journal.write(json.dumps(record, sort_keys=True) + "\n")
                cached[resolved] = record
                computed += 1
            row["path"] = resolved
            row["sha256"] = digest
            if progress_every and (index % progress_every == 0 or index == len(rows)):
                print(json.dumps({
                    "processed": index, "total": len(rows),
                    "computed_this_run": computed, "reused_from_journal": reused,
                }), flush=True)
        journal.flush()
        os.fsync(journal.fileno())

    for resolved in paths:
        current = Path(resolved).stat()
        record = cached[resolved]
        if (current.st_size, current.st_mtime_ns) != (
                record["bytes"], record["mtime_ns"]):
            raise ValueError(f"Source changed before inventory finalization: {resolved}")

    inventory["status"] = "HASHED_HEADER_INVENTORY_NOT_COHORT_FREEZE"
    inventory["hash_enrichment"] = {
        "algorithm": "sha256", "source_inventory": str(inventory_path),
        "journal": str(journal_path), "slice_count": len(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        output_path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(inventory, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    result = {
        "output": str(output_path), "journal": str(journal_path),
        "slices": len(rows), "computed_this_run": computed,
        "reused_from_journal": reused, "output_sha256": sha256(output_path),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Resume-safe SHA-256 enrichment of a LIDC3D inventory")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    enrich_inventory(
        args.inventory, args.out, args.journal, args.progress_every)


if __name__ == "__main__":
    main()
