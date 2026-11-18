"""Server-only regression of streaming preparation against the accepted pilot.

Run from the repository root with PYTHONPATH=src. Writes a new output tree;
never resumes or rewrites the accepted baseline. No training or freezing.
"""
import json
from pathlib import Path
import subprocess
import sys
from lidc3d.provenance import sha256, read_jsonl, write_json

BASE = Path("artifacts/luna16-prepare-pilot20-fold0-v1")
OUT = Path("artifacts/luna16-prepare-pilot20-streaming-v2")
MANIFEST_HASH = "efb24cb6777d8528843aebb7edfaadd37381104c55b024e73e7c778cfc7bd303"
BASE_HASH = "b5303cc52e50a85ee685b729788fbb9d932c696a7aba52fd9bc8b7881448df63"


def snapshot(root):
    return {str(p.relative_to(root)): {"sha256": sha256(p),
            "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
            "mode": p.stat().st_mode}
            for p in sorted(root.rglob("*")) if p.is_file()}


def run(command, name):
    print(f"START: {name}", flush=True)
    with (OUT/f"{name}.log").open("x") as log:
        result = subprocess.run(["/usr/bin/time", "-v", *command],
                                stdout=log, stderr=subprocess.STDOUT)
    (OUT/f"{name}.status").write_text(f"{result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"{name} failed: {result.returncode}; inspect {OUT}/{name}.log")
    print(f"PASS: {name}", flush=True)


def main():
    manifest = BASE/"pilot-cohort.jsonl"
    if sha256(manifest) != MANIFEST_HASH:
        raise ValueError("Accepted pilot manifest hash changed")
    if sha256(BASE/"prepared"/"preparation.json") != BASE_HASH:
        raise ValueError("Accepted pilot preparation marker changed")
    cohort = read_jsonl(manifest)
    if len(cohort) != 20:
        raise ValueError("Expected 20 pilot scans")
    OUT.mkdir(parents=True, exist_ok=False)
    prepared = OUT/"prepared"
    command = [sys.executable, "-u", "-m", "lidc3d.cli", "prepare",
               "--cohort", str(manifest), "--config", "configs/reference.yaml",
               "--out", str(prepared)]
    run(command, "prepare")
    comparison = [sys.executable, "-u", "-m", "lidc3d.preparation_regression",
        "--baseline", str(BASE/"prepared"), "--candidate", str(prepared),
        "--report", str(OUT/"exact-equivalence.json"),
        "--expected-baseline-sha256", BASE_HASH, "--expected-proposals", "833188"]
    for scan in cohort:
        comparison.extend(["--expected-scan-id", scan["scan_id"]])
    run(comparison, "comparison")
    before = snapshot(prepared)
    run([*command, "--resume"], "resume")
    after = snapshot(prepared)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    events = [json.loads(line) for line in (OUT/"resume.log").read_text().splitlines()
              if line.startswith('{"event":')]
    resumed = [e["scan_id"] for e in events if e["event"] == "scan_resumed"]
    checks = {"exact_equivalence": json.loads((OUT/"exact-equivalence.json").read_text())["acceptance_pass"],
              "prepared_tree_unchanged": not changed,
              "all_scans_resumed": resumed == [s["scan_id"] for s in cohort],
              "no_recomputation": not any(e["event"] == "scan_stage" for e in events),
              "already_complete": any(e["event"] == "run_already_complete" for e in events)}
    report = {"checks": checks, "acceptance_pass": all(checks.values()),
              "changed_paths": changed, "pilot_manifest_sha256": MANIFEST_HASH,
              "baseline_preparation_sha256": BASE_HASH,
              "candidate_preparation_sha256": sha256(prepared/"preparation.json"),
              "comparison_report_sha256": sha256(OUT/"exact-equivalence.json")}
    write_json(OUT/"streaming-validation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print(f"REPORT: {OUT}/streaming-validation.json", flush=True)
    print(f"REPORT_SHA256: {sha256(OUT/'streaming-validation.json')}", flush=True)
    if not report["acceptance_pass"]:
        raise RuntimeError("Streaming pilot acceptance failed")


if __name__ == "__main__":
    main()
