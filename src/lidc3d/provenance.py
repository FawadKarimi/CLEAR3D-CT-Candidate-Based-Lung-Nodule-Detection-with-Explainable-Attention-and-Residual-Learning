"""Small immutable artifacts and explicit experiment state; never infer evidence."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    """Exclusive creation prevents an old result being silently overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def safe_output(output, source):
    """A shared dataset is read-only input; derived output must live elsewhere."""
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the shared raw dataset directory")


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def environment():
    import importlib.metadata
    result = {"python": sys.version, "platform": platform.platform(),
              "packages": {}}
    for name in ("torch", "numpy", "scipy", "pydicom", "scikit-image", "scikit-learn", "PyYAML"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda"] = torch.version.cuda
        result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        result["cuda_available"] = None
    return result


def source_signature(root):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and
                   p.suffix in (".py", ".yaml", ".toml"))
    return {str(p.relative_to(root)): sha256(p) for p in files}


def verify_bundle(bundle):
    """Verify all declared file hashes and independently enforce patient grouping."""
    for path, expected in bundle["files"].items():
        if sha256(path) != expected:
            raise ValueError(f"Artifact changed: {path}")
    scans = read_jsonl(bundle["scans"])
    if len({s["scan_id"] for s in scans}) != len(scans):
        raise ValueError("Duplicate scan_id")
    assignments, fingerprints = {}, {}
    for scan in scans:
        patient, split = scan["patient_id"], scan["split"]
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        if assignments.setdefault(patient, split) != split:
            raise ValueError("Patient crosses partitions")
        fingerprint = scan.get("pixel_fingerprint")
        if not fingerprint:
            raise ValueError("Missing source pixel fingerprint")
        prior = fingerprints.setdefault(fingerprint, (patient, split, scan["scan_id"]))
        if prior != (patient, split, scan["scan_id"]):
            raise ValueError("Repeated pixel volume needs duplicate resolution")
    return scans
