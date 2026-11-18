"""Weighted accumulation and bounded AMP skip dispatch, with durable diagnostics.

A nonfinite AMP gradient is not proof of loss-scale overflow. GradScaler may
back off at an effective-batch boundary; unresolved numerical failures stop at
explicit bounds. No group is replayed and no gradient is replaced or repaired.
"""
from __future__ import annotations
import copy
import json
import math
import os
from pathlib import Path
import torch
from torch import nn

POLICY = {"version": "bounded-amp-v1", "max_consecutive_skips": 8,
          "max_total_skips": 16, "require_update_each_epoch": True}
COUNTERS = ("boundaries", "optimizer_updates", "skipped_updates", "consecutive_skips",
            "scaler_update_calls", "scale_changes")


def new_update_state():
    return {"policy": dict(POLICY), **{k: 0 for k in COUNTERS}}


def validate_update_state(state):
    if not isinstance(state, dict) or state.get("policy") != POLICY:
        raise ValueError("Checkpoint optimization policy mismatch")
    if any(type(state.get(k)) is not int or state[k] < 0 for k in COUNTERS):
        raise ValueError("Invalid checkpoint update counters")
    if (state["boundaries"] != state["optimizer_updates"] + state["skipped_updates"]
        or state["scaler_update_calls"] != state["boundaries"]
        or state["scale_changes"] > state["scaler_update_calls"]
        or state["consecutive_skips"] > min(state["skipped_updates"], POLICY["max_consecutive_skips"])
        or state["skipped_updates"] > POLICY["max_total_skips"]):
        raise ValueError("Inconsistent checkpoint update counters")
    return copy.deepcopy(state)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def append_diagnostic(path, event):
    # Flush before propagating failures. This file is within a new run directory.
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(event), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def gradient_status(named_parameters):
    bad, missing, count = {}, [], 0
    for name, p in named_parameters:
        if p.grad is None:
            missing.append(name)
            continue
        if p.grad.is_sparse:
            raise ValueError("Sparse gradients are outside this reviewed policy")
        count += 1
        if not bool(torch.isfinite(p.grad).all()):
            bad[name] = {"nan": int(torch.isnan(p.grad).sum()),
                         "positive_inf": int(torch.isposinf(p.grad).sum()),
                         "negative_inf": int(torch.isneginf(p.grad).sum()),
                         "dtype": str(p.grad.dtype), "shape": list(p.grad.shape)}
    return {"gradient_tensors": count, "missing": missing, "nonfinite": bad}


def finish_accumulation(model, optimizer, scaler, denominator, max_norm, amp,
                        state, emit, context, diagnostics=None):
    """Unscale once, diagnose, then either clip finite grads or let scaler skip.

    Optimizer post-hooks count actual step calls; return values and scale changes
    are not used to guess whether an optimizer update occurred. AdamW's reviewed
    non-fused path is required (custom/fused AMP step hooks are not equivalent).
    """
    event = {"kind": "accumulation_boundary", **context, "amp": bool(amp),
             "denominator": denominator, "clip_limit": max_norm, "scale_before": float(scaler.get_scale()),
             "status": "FAIL", "optimizer_updates": 0, "scaler_update_calls": 0,
             "cause": "NOT_ESTABLISHED"}
    named = list(model.named_parameters())
    calls = [0]
    handle = None
    try:
        if bool(scaler.is_enabled()) != bool(amp):
            raise ValueError("AMP/scaler enablement mismatch")
        if getattr(optimizer, "_step_supports_amp_scaling", False):
            raise ValueError("Custom AMP optimizer path is not reviewed")
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("Invalid accumulated class-weight denominator")
        if not math.isfinite(max_norm) or max_norm <= 0:
            raise ValueError("Invalid clipping bound")
        if not math.isfinite(event["scale_before"]) or event["scale_before"] <= 0:
            raise FloatingPointError("Invalid GradScaler scale")
        event["scaled_gradients"] = gradient_status(named)
        scaler.unscale_(optimizer)
        unscaled = event["unscaled_gradients"] = gradient_status(named)
        if not unscaled["gradient_tensors"]:
            raise FloatingPointError("No gradients at update boundary")
        overflow = bool(unscaled["nonfinite"])
        if overflow:
            event["classification"] = "NONFINITE_INDIVIDUAL_GRADIENTS"
            if not amp:
                raise FloatingPointError("Nonfinite non-AMP gradients")
            if (state["consecutive_skips"] >= POLICY["max_consecutive_skips"]
                    or state["skipped_updates"] >= POLICY["max_total_skips"]):
                event["classification"] = "AMP_SKIP_BOUND_EXHAUSTED"
                raise FloatingPointError("Bounded AMP overflow recovery exhausted")
            # Leave gradients untouched: GradScaler's unscale_ recorded inf/NaN.
            # Its documented step/update dispatch must skip and back off.
        else:
            for _, p in named:
                if p.grad is not None:
                    p.grad.div_(denominator)
            normalized = event["normalized_gradients"] = gradient_status(named)
            if normalized["nonfinite"]:
                event["classification"] = "NONFINITE_AFTER_DENOMINATOR_NORMALIZATION"
                raise FloatingPointError("Normalization created nonfinite gradients")
            grads = [p.grad for _, p in named if p.grad is not None]
            event["classification"] = "FINITE_GRADIENTS_NORM_CHECK"
            try:
                norm = nn.utils.get_total_norm(grads, norm_type=2.0, error_if_nonfinite=True)
            except RuntimeError as error:
                if "non-finite" in str(error):
                    event["classification"] = "NONFINITE_NORM_WITH_FINITE_GRADIENTS"
                raise
            event["norm_before_clip"] = float(norm)
            # The original guard remains enabled. No precision change or fallback.
            nn.utils.clip_grad_norm_([p for _, p in named], max_norm, error_if_nonfinite=True)
            if diagnostics is not None: event["norm_after_clip"]=float(nn.utils.get_total_norm(grads,norm_type=2.0,error_if_nonfinite=True))
            if gradient_status(named)["nonfinite"]:
                raise FloatingPointError("Clipping produced nonfinite gradients")
        def stepped(opt, args, kwargs):
            calls[0] += 1
        handle = optimizer.register_step_post_hook(stepped)
        if diagnostics is not None: diagnostics.capture(model)
        scaler.step(optimizer)
        event["optimizer_updates"] = calls[0]
        if calls[0] != (0 if overflow else 1):
            raise RuntimeError("GradScaler dispatch disagrees with verified gradients")
        scaler.update()
        event["scaler_update_calls"] = 1
        after = event["scale_after"] = float(scaler.get_scale())
        if not math.isfinite(after) or after <= 0:
            raise FloatingPointError("GradScaler produced an invalid scale")
        if overflow and not after < event["scale_before"]:
            raise RuntimeError("Skipped AMP step did not reduce scale")
        state["boundaries"] += 1
        state["optimizer_updates"] += calls[0]
        state["skipped_updates"] += int(overflow)
        state["consecutive_skips"] = state["consecutive_skips"] + 1 if overflow else 0
        state["scaler_update_calls"] += 1
        state["scale_changes"] += int(after != event["scale_before"])
        event["status"] = "SKIPPED_AMP_NONFINITE" if overflow else "UPDATED"
        if not overflow:
            event["classification"] = "FINITE_UPDATE"
        optimizer.zero_grad(set_to_none=True)
        return not overflow
    except BaseException as error:
        event["failure"] = {"type": type(error).__name__, "reason": str(error)}
        raise
    finally:
        if handle is not None:
            handle.remove()
        event["counters"] = copy.deepcopy(state)
        emit(event)
        if diagnostics is not None: diagnostics.boundary(model, optimizer, event)


def _train_epoch(model, loader, optimizer, scaler, weights, device, amp,
                accumulation, max_norm, state, emit, epoch, diagnostics=None):
    """Same weighted summed CE and actual weight denominator as the runner."""
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("Invalid accumulation count")
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("Class weights must be finite and positive")
    before_updates = state["optimizer_updates"]
    optimizer.zero_grad(set_to_none=True)
    numerator, denominator, group_denominator = 0., 0., 0.
    microbatches = []
    for step, batch in enumerate(loader):
        event = {"kind": "microbatch_failure", "epoch": epoch, "microbatch": step}
        try:
            x, y = batch["image"].to(device), batch["label"].to(device)
            if diagnostics is not None: diagnostics.context={'epoch':epoch,'microbatch':step}
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y, weight=weights, reduction="sum")
            if not bool(torch.isfinite(loss)):
                event["classification"] = "NONFINITE_UNSCALED_LOSS"
                raise FloatingPointError("Nonfinite training loss")
            if diagnostics is not None: diagnostics.microbatch(logits,y,weights,epoch,step)
            batch_denominator = float(weights[y].sum())
            scaled_loss = scaler.scale(loss)
            micro = {"step": step, "samples": int(y.numel()), "weighted_loss_sum": float(loss.detach()),
                     "weight_sum": batch_denominator, "scale": float(scaler.get_scale()),
                     "scaled_loss_finite": bool(torch.isfinite(scaled_loss))}
            if microbatches and micro["scale"] != microbatches[0]["scale"]:
                raise RuntimeError("Loss scale changed within accumulation group")
            microbatches.append(micro)
            scaled_loss.backward()
            numerator += micro["weighted_loss_sum"]
            denominator += batch_denominator
            group_denominator += batch_denominator
        except BaseException as error:
            event.update(failure={"type":type(error).__name__,"reason":str(error)},
                         microbatches=microbatches, counters=copy.deepcopy(state))
            emit(event)
            raise
        if (step+1) % accumulation == 0 or step+1 == len(loader):
            finish_accumulation(model, optimizer, scaler, group_denominator, max_norm, amp, state, emit,
                {"epoch": epoch, "last_microbatch": step, "microbatches": microbatches,
                 "incomplete_group": len(microbatches) < accumulation}, **({"diagnostics":diagnostics} if diagnostics is not None else {}))
            group_denominator = 0.
            microbatches = []
    if state["optimizer_updates"] == before_updates:
        emit({"kind":"epoch_failure", "epoch":epoch, "classification":"ZERO_OPTIMIZER_UPDATES",
              "counters":copy.deepcopy(state)})
        raise FloatingPointError("Epoch completed with zero optimizer updates")
    validate_update_state(state)
    return numerator / denominator


def train_epoch(model,loader,optimizer,scaler,weights,device,amp,accumulation,max_norm,state,emit,epoch,diagnostics=None):
    if diagnostics is not None:diagnostics.attach(model)
    try:return _train_epoch(model,loader,optimizer,scaler,weights,device,amp,accumulation,max_norm,state,emit,epoch,diagnostics)
    finally:
        if diagnostics is not None:diagnostics.close()
