"""Bound epoch checkpoint sets, immutable generations and fail-closed publication.

Hashes detect accidental damage, not hostile pickle or forged manifests. Only
locally generated, trusted checkpoint sets may be loaded. One writer per run.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import shutil
import uuid
import torch
from .provenance import canonical_hash, read_json, sha256, write_json

SCHEMA = "lidc3d-checkpoint-set-v1"


def code_identity():
    """Bind every installed package Python source, independently of output paths."""
    root = Path(__file__).parent
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*.py"))}
    return {"scope": "lidc3d/**/*.py", "files": files, "sha256": canonical_hash(files)}


def _sync_dir(path):
    # Directory fsync is required by this publication protocol (POSIX).
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_atomic(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_dir(path.parent)


def _copy_atomic(source, target):
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    with open(source, "rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, target)
    _sync_dir(target.parent)


def _boundary(stage):
    """Named publication boundary for synthetic crash injection; no runtime hook."""


def _validate_pair(last, best, binding, record=None):
    for state in (last, best):
        if state.get("checkpoint_binding") != binding:
            raise ValueError("Checkpoint binding mismatch")
        for key in ("config_hash", "bundle_hash", "seed"):
            if state.get(key) != binding[key]:
                raise ValueError("Checkpoint identity mismatch: " + key)
        if canonical_hash(state["config"]) != binding["config_hash"]:
            raise ValueError("Checkpoint configuration mismatch")
        if type(state.get("epoch")) is not int or state["epoch"] < 0:
            raise ValueError("Invalid checkpoint epoch")
        if (not math.isfinite(state["best_cpm"]) or not math.isfinite(state["stopping_best"])
                or type(state.get("stale")) is not int or state["stale"] < 0):
            raise ValueError("Invalid checkpoint selection state")
        if state["scheduler"]["epoch"] != state["epoch"] or state["scheduler"]["rule"] != "epoch_lr_v1":
            raise ValueError("Inconsistent checkpoint scheduler")
    if best["epoch"] > last["epoch"] or best["best_cpm"] != last["best_cpm"]:
        raise ValueError("Inconsistent best checkpoint")
    if record is not None:
        expected = {"epoch": last["epoch"], "best_epoch": best["epoch"],
                    "best_cpm": last["best_cpm"], "stopping_best": last["stopping_best"],
                    "stale": last["stale"]}
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("Checkpoint commit metadata mismatch")


def load_set(last_path, expected):
    """Validate commit, BOTH files and identities before returning CPU states.

    Normal resume uses root/last.pt. If root publication was interrupted it is
    rejected. Recovery may explicitly name an immutable generation's last.pt;
    the caller chooses that exact sealed generation, never a heuristic search.
    """
    last_path = Path(last_path)
    if last_path.name != "last.pt":
        raise ValueError("Resume requires a committed last.pt")
    root = last_path.parent
    if (root / "checkpoint_commit.json").exists():
        pointer = read_json(root / "checkpoint_commit.json")
        if pointer.get("status") != "COMMITTED":
            raise ValueError("Incomplete checkpoint publication; explicit committed generation required")
        relative = Path(pointer["generation"])
        if len(relative.parts) != 2 or relative.parts[0] != "checkpoints" or relative.name.startswith("."):
            raise ValueError("Invalid checkpoint generation path")
        generation = root / relative
        if generation.resolve().parent != (root / "checkpoints").resolve():
            raise ValueError("Checkpoint generation escapes run")
        if sha256(generation / "commit.json") != pointer["commit_sha256"]:
            raise ValueError("Checkpoint commit hash mismatch")
    else:
        generation = root
    record = read_json(generation / "commit.json")
    if record.get("schema") != SCHEMA or record.get("status") != "COMMITTED":
        raise ValueError("Unsupported or incomplete checkpoint commit")
    binding = record["binding"]
    if not isinstance(binding.get("run_id"), str) or not binding["run_id"]:
        raise ValueError("Missing checkpoint run identity")
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError("Resume identity mismatch")
    if set(record["files"]) != {"last.pt", "best.pt"}:
        raise ValueError("Incomplete checkpoint inventory")
    for name, digest in record["files"].items():
        if sha256(generation / name) != digest or sha256(root / name) != digest:
            raise ValueError("Checkpoint file hash mismatch: " + name)
    # Hash checks precede pickle deserialization. Trust is still required.
    last = torch.load(generation / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(generation / "best.pt", map_location="cpu", weights_only=False)
    _validate_pair(last, best, binding, record)
    if last["epoch"] == best["epoch"] and record["files"]["last.pt"] != record["files"]["best.pt"]:
        raise ValueError("Same-epoch checkpoints differ")
    return {"last": last, "best": best, "binding": binding, "record": record,
            "generation": generation, "commit_sha256": sha256(generation / "commit.json")}


def publish_set(output, state, new_best, binding, imported=None):
    """Never overwrite immutable generations. Root aliases are committed together.

    PUBLISHING is persisted BEFORE mutation; COMMITTED is published only after
    durable last/best files, the sealed record and both verified root aliases.
    Interrupted root publication fails closed; previous generations survive.
    """
    output = Path(output)
    previous = None
    if (output / "checkpoint_commit.json").exists():
        previous = load_set(output / "last.pt", binding)
    elif any((output / name).exists() for name in ("last.pt", "best.pt", "checkpoints")):
        raise ValueError("Uncommitted checkpoint files already exist")
    if imported is not None and previous is not None:
        raise ValueError("Import requires a fresh checkpoint destination")
    parent = imported or previous
    best_state = state if new_best else parent["best"] if parent else None
    if best_state is None:
        raise ValueError("First checkpoint must include a best model")
    _validate_pair(state, best_state, binding)
    if previous and state["epoch"] != previous["last"]["epoch"] + 1:
        raise ValueError("Checkpoint epoch is not the next committed epoch")
    attempt = uuid.uuid4().hex
    generation = output / "checkpoints" / (f"epoch-{state['epoch']:04d}-" + attempt)
    parent_ref = ({"commit_sha256": parent["commit_sha256"],
                   "generation": str(parent["generation"])} if parent else None)
    try:
        _json_atomic(output / "checkpoint_commit.json", {"schema": SCHEMA,
            "status": "PUBLISHING", "attempt": attempt, "generation": str(generation), "previous_committed": parent_ref})
        _boundary("intent")
        generation.mkdir(parents=True, exist_ok=False)
        _sync_dir(output)
        _sync_dir(generation.parent)
        if imported:
            _copy_atomic(imported["generation"] / "last.pt", generation / "last.pt")
        else:
            with (generation / "last.pt").open("xb") as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
        _sync_dir(generation)
        _boundary("last")
        best_source = generation / "last.pt" if new_best else parent["generation"] / "best.pt"
        _copy_atomic(best_source, generation / "best.pt")
        _boundary("best")
        record = {"schema": SCHEMA, "status": "COMMITTED", "binding": binding,
            "epoch": state["epoch"], "best_epoch": best_state["epoch"],
            "best_cpm": state["best_cpm"], "stopping_best": state["stopping_best"],
            "stale": state["stale"], "parent": parent_ref,
            "files": {name: sha256(generation / name) for name in ("last.pt", "best.pt")}}
        _json_atomic(generation / "commit.json", record)
        _boundary("sealed")
        _copy_atomic(generation / "last.pt", output / "last.pt")
        _boundary("root_last")
        _copy_atomic(generation / "best.pt", output / "best.pt")
        _boundary("root_best")
        for name, digest in record["files"].items():
            if sha256(output / name) != digest:
                raise ValueError("Checkpoint alias copy mismatch")
        _json_atomic(output / "checkpoint_commit.json", {"schema": SCHEMA,
            "status": "COMMITTED", "generation": generation.relative_to(output).as_posix(),
            "commit_sha256": sha256(generation / "commit.json")})
        _boundary("committed")
    except BaseException as error:
        # Never remove a partial generation or mutate an earlier committed set.
        write_json(output / ("checkpoint_failure-" + attempt + ".json"), {
            "status": "INCOMPLETE", "type": type(error).__name__, "reason": str(error),
            "attempt": attempt, "generation": str(generation), "previous_committed": parent_ref})
        raise
