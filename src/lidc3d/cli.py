"""Command-line entry points; start with inventory, never with test inference."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import yaml
from .provenance import (read_json, read_jsonl, write_json, write_jsonl, environment,
                         sha256, canonical_hash, verify_bundle, safe_output)


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def preflight(config, output, device):
    import torch
    from .models import DetectorClassifier, signature
    torch.set_num_threads(2)
    device = torch.device(device)
    model = DetectorClassifier(config["model"]).to(device)
    result = {"environment": environment(), "architecture": signature(model, config["model"]["patch_size"]),
              "evidence": "SYNTHETIC_SOFTWARE_PREFLIGHT_NOT_MEDICAL_EXPERIMENT"}
    n = config["model"]["patch_size"]
    x = torch.zeros(config["training"]["microbatch"], 1, n, n, n, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    logits = model(x)
    torch.nn.functional.cross_entropy(logits, torch.arange(len(x), device=device)%2).backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite gradients")
    if device.type == "cuda":
        torch.cuda.synchronize()
    result["single_synthetic_forward_backward_seconds"] = time.perf_counter()-started
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    result["status"] = "passed"
    write_json(output, result)


def predict_command(args):
    import torch
    from .models import DetectorClassifier
    from .training import predict, cases_from
    bundle = read_json(args.bundle)
    scans = verify_bundle(bundle)
    if any(s["split"] != args.split for s in scans):
        scans = [s for s in scans if s["split"] == args.split]
    if not scans:
        raise ValueError("Requested partition empty")
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if args.split == "test":
        if not args.release:
            raise ValueError("Test inference requires a reviewed release JSON")
        release = read_json(args.release)
        if release.get("status") != "approved" or release.get("test_bundle_hash") != canonical_hash(bundle):
            raise ValueError("Test release does not approve this bundle")
        if sha256(args.checkpoint) not in release.get("checkpoint_sha256", []):
            raise ValueError("Checkpoint absent from test release")
        if not release.get("development_decisions_frozen") or not release.get("prior_test_access_disclosure"):
            raise ValueError("Incomplete test release")
    model = DetectorClassifier(state["config"]["model"]).to(args.device)
    model.load_state_dict(state["model"])
    candidates = read_jsonl(bundle["candidates"])
    selected_ids = {s["scan_id"] for s in scans}
    candidates = [p for p in candidates if p["scan_id"] in selected_ids]
    frozen_config = read_json(bundle["resolved_config"])
    for key in ("geometry", "preprocess", "candidates"):
        if frozen_config[key] != state["config"][key]:
            raise ValueError(f"Checkpoint and inference {key} differ")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    predictions, status = predict(model, scans, candidates, state["config"]["model"]["patch_size"], torch.device(args.device))
    write_jsonl(out/"predictions.jsonl", predictions)
    write_jsonl(out/"scan_status.jsonl", status)
    # This merge is explicitly after inference; it belongs to evaluation.
    write_json(out/"evaluation_cases.json", cases_from(scans, predictions))
    write_json(out/"provenance.json", {"checkpoint_sha256": sha256(args.checkpoint),
        "bundle_hash": canonical_hash(bundle), "split": args.split, "predictions_sha256": sha256(out/"predictions.jsonl"),
        "seed": state["seed"],
        "environment": environment(), "status": "PREDICTIONS_SAVED_NOT_YET_REVIEWED"})


def main():
    parser = argparse.ArgumentParser(description="Independent 3D LIDC-IDRI research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("environment")
    p.add_argument("--out")
    p = sub.add_parser("inventory")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hash-files", action="store_true")
    p = sub.add_parser("audit-source", help="Read-only diagnostic audit; never approves source use")
    p.add_argument("--pylidc-source-root", help="Read/hash provider source only; required to enable legacy filename/z evidence")
    p.add_argument("--inventory", required=True)
    p.add_argument("--annotation-db", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("cohort")
    p.add_argument("--inventory", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("luna16-cohort")
    p.add_argument("--inventory", required=True)
    p.add_argument("--assignments", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--excluded-annotations", required=True)
    p.add_argument("--fold", type=int, choices=range(10), required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cohort", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--release", help="Required before preparing withheld test CTs")
    p = sub.add_parser("freeze")
    p.add_argument("--scans", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--test", action="store_true")
    p = sub.add_parser("preflight")
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--bundle", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume")
    p = sub.add_parser("predict")
    p.add_argument("--bundle", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--release")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--cases", required=True)
    p.add_argument("--selection", required=True, help="Validation-derived selection.json")
    p.add_argument("--out", required=True)
    p = sub.add_parser("bootstrap")
    p.add_argument("--arms", required=True, help="JSON: arm -> [{seed, cases}], paired by seed")
    p.add_argument("--primary", required=True, help="Primary arm to subtract other arms from")
    p.add_argument("--repetitions", type=int, default=2000)
    p.add_argument("--permutation", action="store_true", help="Run only for the two prespecified primary contrasts")
    p.add_argument("--out", required=True)
    p = sub.add_parser("explain")
    p.add_argument("--bundle", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.add_argument("--randomization", action="store_true")
    args = parser.parse_args()
    if args.command == "environment":
        result = environment()
        if args.out:
            write_json(args.out, result)
        else:
            print(json.dumps(result, indent=2))
    elif args.command == "inventory":
        from .ingest import inventory
        safe_output(args.out, args.raw)
        write_json(args.out, inventory(args.raw, args.hash_files))
    elif args.command == "audit-source":
        from .source_audit import audit_command
        audit_command(args)
    elif args.command == "cohort":
        from .pipeline import build_cohort
        build_cohort(read_json(args.inventory), load_config(args.config), args.out)
    elif args.command == "luna16-cohort":
        from .luna16 import build_cohort
        build_cohort(
            read_json(args.inventory), args.assignments, args.protocol,
            args.annotations, args.excluded_annotations, args.fold, args.out)
    elif args.command == "prepare":
        from .pipeline import prepare_candidates
        cohort = read_jsonl(args.cohort)
        allow = False
        if any(s["split"] == "test" for s in cohort):
            release = read_json(args.release) if args.release else {}
            allow = release.get("status") == "approved" and release.get("cohort_sha256") == sha256(args.cohort)
        prepare_candidates(cohort, load_config(args.config), args.out, allow)
    elif args.command == "freeze":
        from .pipeline import freeze_bundle
        freeze_bundle(args.scans, args.pool, load_config(args.config), read_json(args.review), args.out, args.test)
    elif args.command == "preflight":
        preflight(load_config(args.config), args.out, args.device)
    elif args.command == "train":
        from .training import train
        train(read_json(args.bundle), load_config(args.config), args.out, args.seed, args.device, args.resume)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "evaluate":
        from .evaluation import DetectionCurves, candidate_metrics, match
        cases, selection = read_json(args.cases), read_json(args.selection)
        threshold = selection["operating_threshold"]
        point = [match([p for p in c["predictions"] if threshold is not None and p["score"] >= threshold],
                       c["positive"], c.get("ignored", [])) for c in cases]
        total_refs = sum(len(c["positive"]) for c in cases)
        write_json(args.out, {"detection": DetectionCurves(cases).summarize(),
            "candidate": candidate_metrics(cases, threshold), "operating_threshold": threshold,
            "operating_sensitivity": sum(v["tp"] for v in point)/total_refs if total_refs else None,
            "operating_fp_per_scan": sum(v["fp"] for v in point)/len(cases),
            "status": "COMPUTED_REQUIRES_ARTIFACT_REVIEW"})
    elif args.command == "bootstrap":
        from .evaluation import DetectionCurves, paired_bootstrap, paired_permutation, holm_adjust
        specification = read_json(args.arms)
        if args.primary not in specification:
            raise ValueError("Primary arm absent")
        arms, seeds = {}, None
        for name in [args.primary] + [n for n in specification if n != args.primary]:
            rows = sorted(specification[name], key=lambda r: r["seed"])
            current = [r["seed"] for r in rows]
            if len(set(current)) != len(current) or (seeds is not None and seeds != current):
                raise ValueError("Training seeds must be explicitly paired")
            seeds = current
            curves = []
            for row in rows:
                provenance = read_json(Path(row["cases"]).parent / "provenance.json")
                if provenance.get("seed") != row["seed"]:
                    raise ValueError("Seed disagrees with inference provenance")
                curves.append(DetectionCurves(read_json(row["cases"])))
            arms[name] = curves
        result = paired_bootstrap(arms, args.repetitions)
        result["training_seeds"] = seeds
        if args.permutation:
            if len(arms) != 3:
                raise ValueError("Prespecified Holm family contains exactly two contrasts")
            values = {name: paired_permutation(arms[args.primary], curves)
                      for name, curves in arms.items() if name != args.primary}
            adjusted = holm_adjust([v["p_value"] for v in values.values()])
            for value, pvalue in zip(values.values(), adjusted):
                value["holm_adjusted_p"] = pvalue
            result["permutation"] = values
        write_json(args.out, result)
    elif args.command == "explain":
        from .explain_runner import explain
        explain(args.bundle, args.checkpoint, args.predictions, args.out, args.device, args.randomization)


if __name__ == "__main__":
    main()
