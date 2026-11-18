"""Validation-only training and inference over fixed CT-only proposals."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import CandidateDataset
from .models import DetectorClassifier, signature
from .evaluation import DetectionCurves
from .provenance import (seed_all, environment, write_json, write_jsonl, read_jsonl,
                         canonical_hash, sha256, verify_bundle)
from .references import candidate_label
from .checkpoints import code_identity, load_set, publish_set
from .optimization import (POLICY, new_update_state, validate_update_state,
                           append_diagnostic, train_epoch)


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
        source = candidates.for_scan(scan["scan_id"]) if hasattr(candidates,"for_scan") else [r for r in candidates if r["scan_id"] == scan["scan_id"]]
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


def _record_validation(output, prefix, scans, predictions, statuses):
    """Persist the complete status cohort before allowing validation selection.

    A normally completed zero-candidate scan is valid. Failed scans remain in
    the evidence; successful scans cannot make an incomplete pass eligible.
    """
    status_path = output / f"{prefix}_scan_status.jsonl"
    write_jsonl(status_path, statuses)
    expected = [scan["scan_id"] for scan in scans]
    reported = [status.get("scan_id") for status in statuses]
    complete = (len(reported) == len(expected) and
                set(reported) == set(expected) and
                all(status.get("status") == "ok" and
                    type(status.get("candidates")) is int and
                    status["candidates"] >= 0 and
                    status.get("predictions") == status["candidates"]
                    for status in statuses))
    if not complete:
        # predict() has already discarded any failed scan's partial predictions.
        write_jsonl(output / f"{prefix}_predictions.jsonl", predictions)
        failure_path = output / f"{prefix}_failure.json"
        write_json(failure_path, {"status": "INCOMPLETE", "stage": prefix,
            "expected_scan_ids": expected, "scan_status_file": status_path.name,
            "predictions_file": f"{prefix}_predictions.jsonl"})
        raise ValueError(f"Incomplete validation; see {failure_path.name}")


def cases_from(scans, predictions):
    scans_by_id = {s["scan_id"]: s for s in scans}
    if any(p["scan_id"] not in scans_by_id for p in predictions):
        raise ValueError("Predictions outside frozen cohort")
    by_scan = {s["scan_id"]: [] for s in scans}
    for p in predictions:
        by_scan[p["scan_id"]].append(p)
    return [{"scan_id": s["scan_id"], "patient_id": s["patient_id"], "positive": s["positive"],
             "ignored": s.get("ignored", []), "predictions": by_scan[s["scan_id"]]} for s in scans]


def train(bundle, config, output, seed, device="cpu", resume=None, observer=None):
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
    from .indexed import CandidateIndex, TrackedLoader
    observe = observer or (lambda *args, **kwargs: None)
    train_scans = [s for s in scans if s["split"] == "train"]
    val_scans = [s for s in scans if s["split"] == "val"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    index = None
    try:
        observe("indexing")
        index = CandidateIndex(bundle["candidates"], scans, output)
        train_rows, val_rows = index.train, index.val
        if not train_rows or not all(index.receipt["training_labels"][str(k)] for k in (0,1)):
            raise ValueError("Training candidates require both classes")
        observe("model_setup")
        cfg = config["training"]
        transfer_identity = None
        if 'medicalnet' in config:
            from . import medicalnet_finetuning
            transfer_identity = medicalnet_finetuning.validate(config, seed)
        if 'training_sampling' in config:
            from .medicalnet_finetuning import TrainingSample
            train_rows = TrainingSample(train_rows, config['training_sampling'], seed)
        device = torch.device(device)
        seed_all(seed)
        model = (medicalnet_finetuning.build(config, seed) if transfer_identity is not None
                 else DetectorClassifier(config["model"])).to(device)
        groups = parameter_groups(model, cfg)
        bases = [g["lr"] for g in groups]
        optimizer = torch.optim.AdamW(groups)
        amp = bool(cfg["amp"] and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        update_state = new_update_state()
        generator = torch.Generator().manual_seed(seed)
        dataset = CandidateDataset(train_scans, train_rows, config["model"]["patch_size"], True, seed)
        loader = DataLoader(dataset, batch_size=cfg["microbatch"], shuffle=True, generator=generator, num_workers=0)
        checkpoint_best, stopping_best, stale, start_epoch = -float("inf"), -float("inf"), 0, 0
        config_hash, bundle_hash = canonical_hash(config), canonical_hash(bundle)
        sources = code_identity()
        expected = {"config_hash": config_hash, "bundle_hash": bundle_hash, "seed": seed,
                    "code_sha256": sources["sha256"], "device_type": device.type, "amp": amp,
                    "optimization_policy_sha256": canonical_hash(POLICY),
                    "candidate_index_sha256": canonical_hash(index.receipt)}
        if transfer_identity is not None:
            expected['medicalnet_identity'] = transfer_identity
        if 'training_sampling' in config:
            expected['training_sampling_sha256'] = canonical_hash(config['training_sampling'])
        binding = {**expected, "run_id": uuid.uuid4().hex}
        restored = None
        if resume:
            restored = load_set(resume, expected)
            binding = restored["binding"]
            state = restored["last"]
            if state['scheduler'] != {'epoch': state['epoch'], 'base_lrs': bases, 'rule': 'epoch_lr_v1'}:
                raise ValueError('Resume scheduler does not match configured base rates')
            update_state = validate_update_state(state["update_state"])
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_epoch = state["epoch"] + 1
            checkpoint_best, stopping_best, stale = state["best_cpm"], state["stopping_best"], state["stale"]
            # Import the exact bytes and ancestry into the fresh execution directory.
            publish_set(output, state, False, binding, imported=restored)
        write_json(output/"config.json", config)
        write_json(output/"environment.json", environment())
        write_json(output/"architecture.json", signature(model, config["model"]["patch_size"]))
        write_json(output/"sources.json", sources)
        write_json(output/"identity.json", {"seed": seed, "config_hash": config_hash, "bundle_hash": bundle_hash,
            "amp": amp, "checkpoint_binding": binding, "execution_id": uuid.uuid4().hex,
            "resume": str(resume) if resume else None,
            "parameter_groups": [{k:v for k,v in g.items() if k != "params"} for g in groups]})
        # Restore AFTER metadata/probes. No setup operation may consume resumed RNG.
        if restored is not None:
            generator.set_state(state["loader_rng"].cpu())
            torch.set_rng_state(state["torch_rng"].cpu())
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            if device.type == "cuda" and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
            del restored, state
        weights = torch.tensor(cfg["class_weights"], device=device, dtype=torch.float32)
        for epoch in range(start_epoch, cfg["max_epochs"] if stale < cfg["patience"] else start_epoch):
            observe("training", epoch=epoch)
            dataset.epoch = epoch
            if 'training_sampling' in config:
                train_rows.set_epoch(epoch)
                write_json(output/f'sampling_epoch_{epoch:03d}.json', train_rows.receipt)
            model.train()
            for group, base in zip(optimizer.param_groups, bases):
                group["lr"] = epoch_lr(base, epoch, cfg)
            started = time.perf_counter()
            updates_before = update_state["optimizer_updates"]
            weighted_loss = train_epoch(model, TrackedLoader(loader, output/f"consumption_epoch_{epoch:03d}.json", len(dataset)), optimizer, scaler, weights, device, amp,
                cfg["accumulation"], cfg["clip_grad_norm"], update_state,
                lambda event: append_diagnostic(output/"optimization_boundaries.jsonl", event), epoch)
            observe("validation", epoch=epoch, final=False)
            predictions, statuses = predict(model, val_scans, val_rows, config["model"]["patch_size"], device, cfg["microbatch"])
            _record_validation(output, f"validation_epoch_{epoch:03d}", val_scans, predictions, statuses)
            observe("metrics", epoch=epoch)
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
                "update_state": copy.deepcopy(update_state),
                "epoch": epoch, "best_cpm": checkpoint_best, "stopping_best": stopping_best, "stale": stale,
                "config": config, "config_hash": config_hash, "bundle_hash": bundle_hash, "seed": seed,
                "checkpoint_binding": binding,
                "scheduler": {"epoch": epoch, "base_lrs": bases, "rule": "epoch_lr_v1"},
                "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
            observe("checkpoint", epoch=epoch)
            publish_set(output, state, new_best, binding)
            if new_best:
                write_jsonl(output/f"val_predictions_epoch_{epoch:03d}.jsonl", predictions)
            write_json(output/f"epoch_{epoch:03d}.json", {"epoch": epoch, "weighted_loss": weighted_loss,
                "optimizer_updates_this_epoch": update_state["optimizer_updates"]-updates_before,
                "update_state": copy.deepcopy(update_state),
                "validation": metrics, "validation_scan_status": statuses,
                "learning_rates": [g["lr"] for g in optimizer.param_groups], "seconds": time.perf_counter()-started,
                "new_best": new_best, "stale": stale})
            if stale >= cfg["patience"]:
                break
        if update_state["optimizer_updates"] == 0:
            raise FloatingPointError("No optimizer updates; completion forbidden")
        final_set = load_set(output/"last.pt", binding)
        best = final_set["best"]
        del final_set
        model.load_state_dict(best["model"])
        best_epoch, best_cpm = best["epoch"], best["best_cpm"]
        del best  # optimizer tensors stay on CPU and are released before prediction.
        observe("validation", epoch=best_epoch, final=True)
        predictions, statuses = predict(model, val_scans, val_rows, config["model"]["patch_size"], device, cfg["microbatch"])
        _record_validation(output, "validation", val_scans, predictions, statuses)
        observe("final_metrics")
        curves = DetectionCurves(cases_from(val_scans, predictions))
        write_jsonl(output/"validation_predictions.jsonl", predictions)
        write_json(output/"selection.json", {"selected_epoch": best_epoch, "validation_cpm": best_cpm,
            "operating_threshold": curves.operating_threshold(), "checkpoint_sha256": sha256(output/"best.pt"),
            "config_hash": config_hash, "bundle_hash": bundle_hash, "seed": seed, "status": "TRAINING_COMPLETE"})
        observe("complete")
    except BaseException as error:
        write_json(output/"failure.json", {"type": type(error).__name__, "reason": str(error),
            "status": "INCOMPLETE", "resume_scope": "explicit coherent committed checkpoint set only"})
        raise
    finally:
        if index is not None: index.close()
    return model
