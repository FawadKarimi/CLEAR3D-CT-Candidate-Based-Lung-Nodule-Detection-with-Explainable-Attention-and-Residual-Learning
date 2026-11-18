"""Post-hoc methods and controls. Spatial agreement does not establish faithfulness."""
from __future__ import annotations
import copy
import math
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F


def margin(output):
    logits = output["logits"] if isinstance(output, dict) else output
    return logits[:, 1]-logits[:, 0]


def gradcam(model, image):
    """Coarsest CNN tensor before fusion; gradient traverses the full head."""
    if image.shape[0] != 1:
        raise ValueError("XAI runner operates one candidate at a time")
    model.eval()
    output = model(image, explain=True)
    feature = output["features"][-1]
    gradient = torch.autograd.grad(margin(output).sum(), feature)[0]
    native = F.relu((gradient.mean((-3, -2, -1), keepdim=True)*feature).sum(1, keepdim=True))
    upsampled = F.interpolate(native, size=image.shape[-3:], mode="trilinear", align_corners=False)
    return {"native": native.detach(), "map": upsampled.detach(), "target": "logit_margin",
            "layer": "encoder.stages.3", "score": float(margin(output).item())}


def integrated_gradients(model, image, baseline=None, intervals=(64, 128, 256), atol=1e-4, rtol=0.01):
    model.eval()
    baseline = torch.zeros_like(image) if baseline is None else baseline
    difference = image-baseline
    with torch.no_grad():
        target_delta = margin(model(image))-margin(model(baseline))
    residuals = []
    for n in intervals:
        if n < 1:
            raise ValueError("Integration intervals must be positive")
        total = torch.zeros_like(image)
        for step in range(n+1):
            point = (baseline + (step/n)*difference).detach().requires_grad_(True)
            gradient = torch.autograd.grad(margin(model(point)).sum(), point)[0]
            total += gradient.detach()*(0.5 if step in (0, n) else 1.0)
        attribution = difference*(total/n)
        residual = attribution.flatten(1).sum(1)-target_delta
        passed = torch.isfinite(attribution).all() and torch.all(residual.abs() <= atol+rtol*target_delta.abs())
        residuals.append({"intervals": n, "residual": residual.detach().cpu().tolist()})
        if passed:
            break
    return {"signed": attribution.detach(), "positive": attribution.clamp_min(0).detach(),
            "target_delta": target_delta.detach().cpu().tolist(), "residuals": residuals,
            "complete": bool(passed), "intervals": n}


def normalize_map(array, valid):
    array = np.asarray(array, float)
    valid = np.asarray(valid, bool)
    if array.shape != valid.shape or not valid.any() or not np.isfinite(array).all():
        raise ValueError("Invalid map shape/values/support")
    array = np.where(valid, np.maximum(array, 0), 0)
    values = array[valid]
    if values.max()-values.min() <= 1e-12 or values.sum() <= 0:
        raise ValueError("Degenerate constant/zero map")
    return array/values.sum()


def attention_maps(model, image, valid):
    model.eval()
    with torch.no_grad():
        result = model(image, explain=True)
    required = {str(i) for i in model.config["scales"]}
    if set(result["attention"]) != required or not required:
        raise ValueError("Missing required attention scales")
    maps = {}
    for key, native in result["attention"].items():
        up = F.interpolate(native, size=image.shape[-3:], mode="trilinear", align_corners=False)
        maps[key] = normalize_map(up[0, 0].cpu().numpy(), valid)
    return maps, normalize_map(np.mean(list(maps.values()), axis=0), valid)


def top_mask(array, valid, fraction=0.1):
    if not 0 < fraction <= 1:
        raise ValueError("Fraction outside (0,1]")
    flat = np.flatnonzero(valid.ravel())
    count = max(1, int(np.ceil(len(flat)*fraction)))
    order = np.lexsort((flat, -np.asarray(array).ravel()[flat]))
    result = np.zeros(valid.size, bool)
    result[flat[order[:count]]] = True
    return result.reshape(valid.shape)


def localization(array, target, valid, fraction=0.1):
    target = np.asarray(target, bool) & valid
    if not target.any():
        return {"dice": None, "iou": None, "pointing": None, "reason": "empty_reference_mask"}
    normalized = normalize_map(array, valid)
    predicted = top_mask(normalized, valid, fraction)
    intersection = int((predicted & target).sum())
    maximum = np.argmax(np.where(valid, normalized, -np.inf))
    return {"dice": 2*intersection/int(predicted.sum()+target.sum()),
            "iou": intersection/int((predicted|target).sum()),
            "pointing": bool(target.ravel()[maximum])}


def agreement(a, b, valid, fraction=0.1):
    a, b = normalize_map(a, valid), normalize_map(b, valid)
    first, second = top_mask(a, valid, fraction), top_mask(b, valid, fraction)
    correlation = float(spearmanr(a[valid], b[valid]).statistic)
    return {"spearman": correlation if np.isfinite(correlation) else None,
            "top_dice": float(2*(first&second).sum()/(first.sum()+second.sum()))}


def controls(shape, valid, seed=2026):
    coords = np.indices(shape)
    center = np.asarray(shape)//2
    gaussian = np.exp(-sum((coords[d]-center[d])**2 for d in range(3))/(2*5**2))
    random = np.random.default_rng(seed).random(shape)
    return {"center_gaussian": normalize_map(gaussian, valid), "random": normalize_map(random, valid)}


def blurred_reference(image, valid=None, sigma=3):
    array = image.detach().cpu().numpy()
    blurred = ndi.gaussian_filter(array, sigma=(0, 0, sigma, sigma, sigma), mode="reflect", truncate=4)
    if valid is not None:
        blurred *= np.asarray(valid)[None, None]
    return torch.as_tensor(blurred, device=image.device, dtype=image.dtype)


def perturbation_curves(model, image, array, valid, baseline=None, block=4, steps=21):
    """Fixed block ranking/target for the complete insertion/deletion replay."""
    model.eval()
    if image.shape[0:2] != (1, 1):
        raise ValueError("One one-channel candidate required")
    array = normalize_map(array, valid)
    baseline = blurred_reference(image, valid) if baseline is None else baseline
    blocks = []
    shape = array.shape
    for z in range(0, shape[0], block):
        for y in range(0, shape[1], block):
            for x in range(0, shape[2], block):
                sl = (slice(z, z+block), slice(y, y+block), slice(x, x+block))
                if valid[sl].any():
                    blocks.append((float(array[sl][valid[sl]].mean()), (z, y, x), sl))
    blocks.sort(key=lambda row: (-row[0], row[1]))
    fractions, insertion, deletion, replaced = np.linspace(0, 1, steps), [], [], []
    with torch.no_grad():
        for fraction in fractions:
            n = int(np.floor(fraction*len(blocks)+0.5))
            mask = np.zeros(shape, bool)
            for _, _, sl in blocks[:n]:
                mask[sl] = valid[sl]
            tensor_mask = torch.as_tensor(mask, device=image.device)[None, None]
            inserted = torch.where(tensor_mask, image, baseline)
            deleted = torch.where(tensor_mask, baseline, image)
            insertion.append(float(margin(model(inserted)).item()))
            deletion.append(float(margin(model(deleted)).item()))
            replaced.append(n)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {"fractions": fractions.tolist(), "blocks_replaced": replaced,
            "ranking": [list(b[1]) for b in blocks], "insertion": insertion, "deletion": deletion,
            "insertion_auc": float(integrate(insertion, fractions)),
            "deletion_auc": float(integrate(deletion, fractions)), "target": "uncalibrated_logit_margin"}


def randomized_models(model, seed=2026):
    """Cumulative head-to-input randomization; original model is untouched."""
    clone = copy.deepcopy(model)
    roots = ["head"]
    if model.hybrid:
        roots += ["merge" if hasattr(model, "merge") else "concat", "local_fusion", "gates", "transformers"]
    roots += [f"encoder.stages.{i}" for i in (3, 2, 1, 0)] + ["encoder.stem"]
    changed = []
    # fork_rng prevents a diagnostic from changing subsequent experiment RNGs.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        for root in roots:
            module = clone.get_submodule(root)
            for child in module.modules():
                if hasattr(child, "reset_parameters"):
                    child.reset_parameters()
                elif isinstance(child, torch.nn.MultiheadAttention):
                    child._reset_parameters()
            changed.append(root)
            yield list(changed), clone


def fit_mixture(validation_cases, step=0.05):
    """Validation-only positive-mask localization fit; not a training objective.

    Cases: {split, patient_id, maps:[CAM,IG,attention], mask, valid}. All maps
    must be nondegenerate; an invalid case must be logged before calling here.
    """
    if not validation_cases or any(c["split"] != "val" for c in validation_cases):
        raise ValueError("Mixture fitting requires validation cases only")
    denominator = round(1/step)
    if not np.isclose(denominator*step, 1):
        raise ValueError("Step must divide the simplex unit")
    trace = []
    for a in range(denominator+1):
        for b in range(denominator-a+1):
            weights = np.array([a, b, denominator-a-b])/denominator
            per_patient = {}
            for case in validation_cases:
                maps = [normalize_map(m, case["valid"]) for m in case["maps"]]
                value = localization(sum(w*m for w,m in zip(weights, maps)), case["mask"], case["valid"])["dice"]
                if value is not None:
                    per_patient.setdefault(case["patient_id"], []).append(value)
            objective = float(np.mean([np.mean(v) for v in per_patient.values()])) if per_patient else None
            trace.append({"weights": weights.tolist(), "objective": objective})
    valid = [r for r in trace if r["objective"] is not None]
    if not valid:
        raise ValueError("No valid validation masks")
    selected = min(valid, key=lambda r: (-r["objective"], float(np.sum((np.array(r["weights"])-1/3)**2)), r["weights"]))
    return {"selected": selected, "trace": trace, "objective": "mean_patient_positive_case_Dice_top10pct"}


def method_map(model, image, valid, method, config):
    if method == "gradcam":
        return normalize_map(gradcam(model,image)["map"][0,0].cpu().numpy(),valid)
    if method == "attention":
        return attention_maps(model,image,valid)[1]
    if method == "ig":
        result=integrated_gradients(model,image,intervals=config["ig_intervals"],atol=config["ig_atol"],rtol=config["ig_rtol"])
        if not result["complete"]:
            raise ValueError("IG completeness tolerance not met")
        return normalize_map(result["positive"][0,0].cpu().numpy(),valid)
    raise ValueError("Unknown attribution method")


def stability(model, image, valid, original_map, method, config, seed):
    """Noise replicates plus one reflected-input map pulled back to original axes."""
    rng=np.random.default_rng(seed)
    with torch.no_grad():
        original_score=float(margin(model(image)).item())
    results=[]
    for i in range(config["stability_repetitions"]+1):
        reflect=i==config["stability_repetitions"]
        if reflect:
            changed=torch.flip(image,[-1]);changed_valid=valid[:,:,::-1].copy()
        else:
            noise=torch.as_tensor(rng.normal(0,.01,image.shape),device=image.device,dtype=image.dtype)
            changed=(image+noise*torch.as_tensor(valid,device=image.device)[None,None]).clamp(0,1)
            changed_valid=valid
        with torch.no_grad():
            changed_score=float(margin(model(changed)).item())
        record={"transform":"left_right" if reflect else "noise","replicate":i,
                "original_margin":original_score,"changed_margin":changed_score,
                "prediction_sign_changed":bool((original_score>=0)!=(changed_score>=0))}
        try:
            changed_map=method_map(model,changed,changed_valid,method,config)
            if reflect:
                changed_map=changed_map[:,:,::-1]
            record.update(agreement(original_map,changed_map,valid))
        except Exception as error:
            record["failure"]=str(error)
        results.append(record)
    return results
