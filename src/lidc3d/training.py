"""Validation-only training and inference over fixed CT-only proposals."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import CandidateDataset
from .models import DetectorClassifier, signature
from .evaluation import DetectionCurves
from .provenance import (seed_all, environment, write_json, write_jsonl, read_jsonl,
                         canonical_hash, sha256, verify_bundle, source_signature)
from .references import candidate_label


def parameter_groups(model, cfg):
    buckets = {}
    norm_ids = {id(p) for module in model.modules() if isinstance(module, (nn.GroupNorm, nn.LayerNorm))
                for p in module.parameters(recurse=False)}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        family = "cnn" if name.startswith("encoder.") else "transformer" if name.startswith("transformers.") else "head"
        decay = 0.0 if name.endswith("bias") or id(p) in norm_ids else cfg["weight_decay"]
        key = (family, decay)
        buckets.setdefault(key, {"params": [], "names": [], "lr": cfg[f"lr_{family}"],
                                  "weight_decay": decay, "family": family})
        buckets[key]["params"].append(p)
        buckets[key]["names"].append(name)
    groups = list(buckets.values())
    ids = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Parameter groups must cover each trainable parameter exactly once")
    return groups


def epoch_lr(base, epoch, cfg):
    warmup, maximum = cfg["warmup_epochs"], cfg["max_epochs"]
    if epoch < warmup:
        return base * (epoch+1)/warmup
    progress = (epoch-warmup+1)/max(maximum-warmup, 1)
    return cfg["min_lr"] + 0.5*(base-cfg["min_lr"])*(1+math.cos(math.pi*progress))


def predict(model, scans, candidates, size, device, batch_size=2):
    """No reference/label access: inputs are scan arrays and CT-derived centers.

    Each scan is atomic: any prediction failure discards that scan's partial
    predictions and emits a failure status. Zero-candidate scans remain present.
    """
    result, status = [], []
    model.eval()
    for scan in scans:
        started = time.perf_counter()
        source = [r for r in candidates if r["scan_id"] == scan["scan_id"]]
        safe_scan = {k: scan[k] for k in ("scan_id", "image_path", "valid_path") if k in scan}
        safe_rows = [{k: r[k] for k in ("id", "scan_id", "zyx", "center_lps")} for r in source]
        rows = []
        reason = None
        try:
            if scan.get("status", "ok") != "ok":
                raise ValueError(scan.get("failure_reason", "preprocessing_failed"))
            loader = DataLoader(CandidateDataset([safe_scan], safe_rows, size), batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch in loader:
                    logits = model(batch["image"].to(device)).float().cpu()
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite logits")
                    scores = logits.softmax(1)[:, 1]
                    for i, index in enumerate(batch["index"].tolist()):
                        rows.append({**safe_rows[index], "logits": logits[i].tolist(), "score": float(scores[i])})
        except Exception as error:
            rows = []
            reason = f"{type(error).__name__}: {error}"
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result.extend(rows)
        status.append({"scan_id": scan["scan_id"], "status": "failed" if reason else "ok",
                       "reason": reason, "candidates": len(source), "predictions": len(rows),
                       "seconds": time.perf_counter()-started})
    return result, status


def cases_from(scans, predictions):
    scans_by_id = {s["scan_id"]: s for s in scans}
    if any(p["scan_id"] not in scans_by_id for p in predictions):
        raise ValueError("Predictions outside frozen cohort")
    by_scan = {s["scan_id"]: [] for s in scans}
    for p in predictions:
        by_scan[p["scan_id"]].append(p)
    return [{"scan_id": s["scan_id"], "patient_id": s["patient_id"], "positive": s["positive"],
             "ignored": s.get("ignored", []), "predictions": by_scan[s["scan_id"]]} for s in scans]


def atomic_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(bundle, config, output, seed, device="cpu", resume=None):
    """New output per run. Epoch-boundary resume, including RNG/optimizer state."""
    if not bundle.get("reference_audit_approved") or not bundle.get("development_config_frozen"):
        raise ValueError("Reference audit and candidate configuration freeze required")
    if config["training"].get("alignment_weight", 0) != 0:
        raise ValueError("Mask-supervised alignment is not part of the reference runner; separate implementation gate")
    scans = verify_bundle(bundle)
    if any(s["split"] == "test" for s in scans):
        raise ValueError("Training requires a development-only bundle; withhold the test manifest")
    from .provenance import read_json
    frozen = read_json(bundle["resolved_config"])
    for key in ("geometry", "preprocess", "candidates"):
        if frozen[key] != config[key]:
            raise ValueError(f"Training configuration differs from prepared {key}")
    candidates = read_jsonl(bundle["candidates"])
    train_scans = [s for s in scans if s["split"] == "train"]
    val_scans = [s for s in scans if s["split"] == "val"]
    train_ids = {s["scan_id"]: s for s in train_scans}
    val_ids = {s["scan_id"] for s in val_scans}
    # Labels are constructed only in the training/evaluation layer.
    train_rows = []
    for row in candidates:
        if row["scan_id"] in train_ids:
            scan = train_ids[row["scan_id"]]
            label = candidate_label(row["center_lps"], scan["positive"], scan.get("ignored", []))
            if label != -1:
                train_rows.append({**row, "label": label})
    val_rows = [r for r in candidates if r["scan_id"] in val_ids]
    if not train_rows or {r["label"] for r in train_rows} != {0, 1}:
        raise ValueError("Training candidates require both classes")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = config["training"]
    device = torch.device(device)
    seed_all(seed)
    model = DetectorClassifier(config["model"]).to(device)
    groups = parameter_groups(model, cfg)
    bases = [g["lr"] for g in groups]
    optimizer = torch.optim.AdamW(groups)
    amp = bool(cfg["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed)
    dataset = CandidateDataset(train_scans, train_rows, config["model"]["patch_size"], True, seed)
    loader = DataLoader(dataset, batch_size=cfg["microbatch"], shuffle=True, generator=generator, num_workers=0)
    checkpoint_best, stopping_best, stale, start_epoch = -float("inf"), -float("inf"), 0, 0
    config_hash, bundle_hash = canonical_hash(config), canonical_hash(bundle)
    if resume:
        # Only load locally generated/trusted checkpoints; torch pickle is executable.
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["config_hash"] != config_hash or state["bundle_hash"] != bundle_hash or state["seed"] != seed:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        generator.set_state(state["loader_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        start_epoch = state["epoch"]+1
        checkpoint_best, stopping_best, stale = state["best_cpm"], state["stopping_best"], state["stale"]
        # Keep the previous best, rather than mistaking last for best.
        previous_best = Path(resume).parent / "best.pt"
        if not previous_best.exists():
            raise ValueError("Resume requires the previous run's best.pt alongside last.pt")
        import shutil
        shutil.copy2(previous_best, output / "best.pt")
    write_json(output/"config.json", config)
    write_json(output/"environment.json", environment())
    write_json(output/"architecture.json", signature(model, config["model"]["patch_size"]))
    write_json(output/"sources.json", source_signature(Path(__file__).parents[2]))
    write_json(output/"identity.json", {"seed": seed, "config_hash": config_hash, "bundle_hash": bundle_hash,
        "amp": amp, "resume": str(resume) if resume else None,
        "parameter_groups": [{k:v for k,v in g.items() if k != "params"} for g in groups]})
    weights = torch.tensor(cfg["class_weights"], device=device, dtype=torch.float32)
    try:
        for epoch in range(start_epoch, cfg["max_epochs"]):
            dataset.epoch = epoch
            model.train()
            for group, base in zip(optimizer.param_groups, bases):
                group["lr"] = epoch_lr(base, epoch, cfg)
            optimizer.zero_grad(set_to_none=True)
            numerator, denominator, group_denominator = 0., 0., 0.
            started = time.perf_counter()
            for step, batch in enumerate(loader):
                x, y = batch["image"].to(device), batch["label"].to(device)
                # Sum weighted losses. Divide accumulated gradients by actual sum
                # of class weights, including an incomplete final microbatch group.
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y, weight=weights, reduction="sum")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss).backward()
                batch_denominator = float(weights[y].sum())
                numerator += float(loss.detach())
                denominator += batch_denominator
                group_denominator += batch_denominator
                if (step+1) % cfg["accumulation"] == 0 or step+1 == len(loader):
                    scaler.unscale_(optimizer)
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.div_(group_denominator)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"], error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    group_denominator = 0.
            predictions, statuses = predict(model, val_scans, val_rows, config["model"]["patch_size"], device, cfg["microbatch"])
            curves = DetectionCurves(cases_from(val_scans, predictions))
            metrics = curves.summarize()
            cpm = metrics["cpm"]
            if cpm is None:
                raise ValueError("Validation CPM undefined")
            # Checkpoint takes every strict maximum; patience uses min_delta.
            new_best = cpm > checkpoint_best
            if new_best:
                checkpoint_best = cpm
            if cpm > stopping_best + cfg["min_delta"]:
                stopping_best, stale = cpm, 0
            else:
                stale += 1
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best_cpm": checkpoint_best, "stopping_best": stopping_best, "stale": stale,
                "config": config, "config_hash": config_hash, "bundle_hash": bundle_hash, "seed": seed,
                "scheduler": {"epoch": epoch, "base_lrs": bases, "rule": "epoch_lr_v1"},
                "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
            atomic_checkpoint(output/"last.pt", state)
            if new_best:
                atomic_checkpoint(output/"best.pt", state)
                write_jsonl(output/f"val_predictions_epoch_{epoch:03d}.jsonl", predictions)
            write_json(output/f"epoch_{epoch:03d}.json", {"epoch": epoch, "weighted_loss": numerator/denominator,
                "validation": metrics, "validation_scan_status": statuses,
                "learning_rates": [g["lr"] for g in optimizer.param_groups], "seconds": time.perf_counter()-started,
                "new_best": new_best, "stale": stale})
            if stale >= cfg["patience"]:
                break
        best = torch.load(output/"best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        predictions, statuses = predict(model, val_scans, val_rows, config["model"]["patch_size"], device, cfg["microbatch"])
        curves = DetectionCurves(cases_from(val_scans, predictions))
        write_jsonl(output/"validation_predictions.jsonl", predictions)
        write_jsonl(output/"validation_scan_status.jsonl", statuses)
        write_json(output/"selection.json", {"selected_epoch": best["epoch"], "validation_cpm": best["best_cpm"],
            "operating_threshold": curves.operating_threshold(), "checkpoint_sha256": sha256(output/"best.pt"),
            "config_hash": config_hash, "bundle_hash": bundle_hash, "seed": seed, "status": "TRAINING_COMPLETE"})
    except BaseException as error:
        write_json(output/"failure.json", {"type": type(error).__name__, "reason": str(error),
            "status": "INCOMPLETE", "resume_scope": "last completed epoch only"})
        raise
    return model
