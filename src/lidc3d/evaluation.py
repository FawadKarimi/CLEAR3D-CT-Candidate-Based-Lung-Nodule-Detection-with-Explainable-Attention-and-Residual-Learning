"""Custom detection evaluator, deliberately not advertised as official LUNA16."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from .references import inside_ignore, candidate_label
from .provenance import canonical_hash

BUDGETS = np.array([0.125, 0.25, 0.5, 1., 2., 4., 8.])


def match(predictions, reference, ignored):
    """Maximum cardinality, then minimum total normalized distance.

    Sorted identifiers plus SciPy's deterministic solver resolve remaining
    equal-cost ties. The SciPy version is part of the environment lock.
    """
    predictions = sorted(predictions, key=lambda r: r["id"])
    reference = sorted(reference, key=lambda r: r["id"])
    if any(np.asarray(p["center_lps"]).shape != (3,) or not np.isfinite(p["center_lps"]).all() for p in predictions):
        raise ValueError("Invalid prediction coordinates")
    if any(np.asarray(r["center_lps"]).shape != (3,) or not np.isfinite(r["center_lps"]).all() or
           not np.isfinite(r["radius_mm"]) or r["radius_mm"] <= 0 for r in reference):
        raise ValueError("Invalid reference geometry")
    if len({p["id"] for p in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    if len({r["id"] for r in reference}) != len(reference):
        raise ValueError("Duplicate reference IDs")
    eligible = np.zeros((len(reference), len(predictions)), bool)
    pairs = []
    if reference and predictions:
        centers = np.array([p["center_lps"] for p in predictions], float)
        radii = np.array([r["radius_mm"] for r in reference], float)
        if not np.isfinite(centers).all() or not np.isfinite(radii).all() or (radii <= 0).any():
            raise ValueError("Invalid geometry")
        distances = np.linalg.norm(np.array([r["center_lps"] for r in reference])[:, None]-centers[None], axis=-1)
        ratios = distances/radii[:, None]
        eligible = ratios <= 1
        # One dummy per reference. A single additional real match outweighs
        # the largest possible change in summed eligible normalized distances.
        penalty = len(reference) + 1
        cost = np.full((len(reference), len(predictions)+len(reference)), float(penalty))
        cost[:, :len(predictions)] = np.where(eligible, ratios, penalty*2)
        rows, cols = linear_sum_assignment(cost)
        pairs = [(reference[i]["id"], predictions[j]["id"]) for i, j in zip(rows, cols)
                 if j < len(predictions) and eligible[i, j]]
    matched = {p for _, p in pairs}
    fp, ignored_count = 0, 0
    for j, prediction in enumerate(predictions):
        if prediction["id"] in matched:
            continue
        if eligible[:, j].any() or not inside_ignore(prediction["center_lps"], ignored):
            fp += 1
        else:
            ignored_count += 1
    return {"tp": len(pairs), "fp": fp, "fn": len(reference)-len(pairs),
            "ignored": ignored_count, "pairs": pairs}


class DetectionCurves:
    """Cache scan-local match counts at every distinct score threshold.

    Patient bootstrap weights duplicate each scan's entire independent matching
    problem. This is algebraically equivalent to explicit replicated IDs and
    avoids repeating the assignment solver for every bootstrap replicate.
    """
    def __init__(self, cases):
        if not cases:
            raise ValueError("No eligible evaluation scans")
        if len({c["scan_id"] for c in cases}) != len(cases):
            raise ValueError("Duplicate scan IDs")
        self.cases = cases
        self.cohort_hash = canonical_hash(sorted([{k:c.get(k, []) for k in ("scan_id", "patient_id", "positive", "ignored")}
                                                  for c in cases], key=lambda c:c["scan_id"]))
        self.patients = sorted({c["patient_id"] for c in cases})
        scores = [float(p["score"]) for c in cases for p in c["predictions"]]
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite prediction scores")
        self.thresholds = np.array([np.inf] + sorted(set(scores), reverse=True))
        self.tp = np.zeros((len(self.patients), len(self.thresholds)), np.int64)
        self.fp = np.zeros_like(self.tp)
        self.nrefs = np.zeros(len(self.patients), np.int64)
        self.nscans = np.zeros(len(self.patients), np.int64)
        for case in cases:
            patient = self.patients.index(case["patient_id"])
            self.nrefs[patient] += len(case["positive"])
            self.nscans[patient] += 1
            local = np.array(sorted({p["score"] for p in case["predictions"]}, reverse=True))
            counts_tp, counts_fp = [0], [0]
            for threshold in local:
                kept = [p for p in case["predictions"] if p["score"] >= threshold]
                value = match(kept, case["positive"], case.get("ignored", []))
                counts_tp.append(value["tp"])
                counts_fp.append(value["fp"])
            lookup = np.searchsorted(-local, -self.thresholds, side="right")
            self.tp[patient] += np.asarray(counts_tp)[lookup]
            self.fp[patient] += np.asarray(counts_fp)[lookup]

    def summarize(self, weights=None, budgets=BUDGETS):
        weights = np.ones(len(self.patients), dtype=np.int64) if weights is None else np.asarray(weights)
        if weights.shape != (len(self.patients),) or (weights < 0).any():
            raise ValueError("Invalid patient multiplicities")
        nrefs, nscans = int(weights@self.nrefs), int(weights@self.nscans)
        if nrefs == 0 or nscans == 0:
            return {"cpm": None, "reason": "no_positive_references_or_no_scans", "nrefs": nrefs, "nscans": nscans}
        tp, fp = weights@self.tp, weights@self.fp
        sensitivity, burden = tp/nrefs, fp/nscans
        values = [float(sensitivity[burden <= b].max()) for b in budgets]
        return {"cpm": float(np.mean(values)), "sensitivity_at_budgets": values,
            "budgets": list(map(float, budgets)), "nrefs": nrefs, "nscans": nscans,
            "candidate_ceiling": float(sensitivity[-1]), "max_fp_per_scan": float(burden.max()),
            "tp": tp.tolist(), "fp": fp.tolist(), "sensitivity": sensitivity.tolist(),
            "fp_per_scan": burden.tolist(),
            "thresholds": [None if np.isinf(t) else float(t) for t in self.thresholds]}

    def operating_threshold(self, budget=1):
        result = self.summarize()
        if result["cpm"] is None:
            raise ValueError("Validation reference set has no positives")
        sens, fp = np.array(result["sensitivity"]), np.array(result["fp_per_scan"])
        feasible = np.flatnonzero(fp <= budget)
        best = feasible[np.argmax(sens[feasible])]
        # Thresholds are descending: first maximizing sensitivity is highest.
        return None if best == 0 else float(self.thresholds[best])


def paired_bootstrap(arms, repetitions=2000, seed=2026):
    """arms: name -> one DetectionCurves object per declared training seed."""
    names = list(arms)
    template = arms[names[0]][0]
    count = len(arms[names[0]])
    for curves in arms.values():
        if len(curves) != count:
            raise ValueError("Unpaired seed count")
        for curve in curves:
            if curve.cohort_hash != template.cohort_hash or curve.patients != template.patients or not np.array_equal(curve.nrefs, template.nrefs) or not np.array_equal(curve.nscans, template.nscans):
                raise ValueError("Models do not share the same patient/reference cohort")
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in names}
    n = len(template.patients)
    draws = []
    for _ in range(repetitions):
        weights = np.bincount(rng.integers(0, n, size=n), minlength=n)
        draws.append(weights.tolist())
        for name, curves in arms.items():
            values = [c.summarize(weights)["cpm"] for c in curves]
            samples[name].append(None if any(v is None for v in values) else float(np.mean(values)))
    def interval(values):
        valid = [v for v in values if v is not None]
        invalid = len(values)-len(valid)
        return {"ci95": np.percentile(valid, [2.5, 97.5]).tolist() if valid else None,
                "valid_replicates": len(valid), "invalid_replicates": invalid,
                "unstable": invalid/len(values) > 0.05}
    contrasts = {}
    for other in names[1:]:
        diff = [None if a is None or b is None else a-b for a, b in zip(samples[names[0]], samples[other])]
        contrasts[f"{names[0]}_minus_{other}"] = {**interval(diff), "draw_values": diff}
    summaries = {}
    for name, curves in arms.items():
        values = [c.summarize()["cpm"] for c in curves]
        valid = all(v is not None for v in values)
        summaries[name] = {**interval(samples[name]), "seed_cpm": values,
            "mean_cpm": float(np.mean(values)) if valid else None,
            "seed_sd": float(np.std(values, ddof=1)) if valid and len(values)>1 else None,
            "seed_range": [min(values), max(values)] if valid else None}
    return {"arms": summaries, "paired_contrasts": contrasts, "seed": seed,
            "repetitions": repetitions, "patient_order": template.patients, "draw_counts": draws}


def candidate_metrics(cases, threshold):
    from sklearn.metrics import roc_auc_score
    labels, scores = [], []
    for case in cases:
        for p in case["predictions"]:
            label = candidate_label(p["center_lps"], case["positive"], case.get("ignored", []))
            if label != -1:
                labels.append(label)
                scores.append(p["score"])
    y, s = np.array(labels), np.array(scores)
    pred = s >= (np.inf if threshold is None else threshold)
    tp, tn = int(((y == 1)&pred).sum()), int(((y == 0)&~pred).sum())
    fp, fn = int(((y == 0)&pred).sum()), int(((y == 1)&~pred).sum())
    def ratio(a, b):
        return a/b if b else None
    return {"n": len(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": ratio(tp, tp+fn), "specificity": ratio(tn, tn+fp),
        "precision": ratio(tp, tp+fp), "f1": ratio(2*tp, 2*tp+fp+fn),
        "auroc": float(roc_auc_score(y, s)) if len(set(labels)) == 2 else None,
        "brier": float(np.mean((s-y)**2)) if len(y) else None,
        "scope": "nonignored generated candidates; not patient cancer diagnosis"}


def paired_permutation(first, second, repetitions=10000, seed=2027):
    """Patient-block score swapping synchronized over matched training seeds."""
    if len(first) != len(second) or not first:
        raise ValueError("Paired nonempty seeds required")
    template = first[0]
    if any(c.cohort_hash != template.cohort_hash for c in first+second):
        raise ValueError("Reference cohorts differ")
    aligned = []
    for a,b in zip(first, second):
        thresholds = np.array([np.inf]+sorted(set(a.thresholds[1:])|set(b.thresholds[1:]), reverse=True))
        ai = np.searchsorted(-a.thresholds[1:], -thresholds, side="right")
        bi = np.searchsorted(-b.thresholds[1:], -thresholds, side="right")
        aligned.append((a.tp[:,ai], a.fp[:,ai], b.tp[:,bi], b.fp[:,bi]))
    total_refs, total_scans = template.nrefs.sum(), template.nscans.sum()
    if not total_refs:
        return {"p_value": None, "reason": "no_positive_references"}
    def cpm(tp,fp):
        sens, burden = tp/total_refs, fp/total_scans
        return np.mean([sens[burden<=budget].max() for budget in BUDGETS])
    def difference(swap):
        values=[]
        for at,af,bt,bf in aligned:
            left_tp=(~swap)@at+swap@bt;left_fp=(~swap)@af+swap@bf
            right_tp=swap@at+(~swap)@bt;right_fp=swap@af+(~swap)@bf
            values.append(cpm(left_tp,left_fp)-cpm(right_tp,right_fp))
        return float(np.mean(values))
    observed=difference(np.zeros(len(template.patients),bool))
    rng=np.random.default_rng(seed)
    draws=[difference(rng.integers(0,2,len(template.patients)).astype(bool)) for _ in range(repetitions)]
    extreme=sum(abs(v)>=abs(observed)-1e-12 for v in draws)
    return {"observed_difference": observed, "p_value": (1+extreme)/(1+repetitions),
            "repetitions": repetitions, "seed": seed, "permuted_differences": draws}


def holm_adjust(p_values):
    """Return adjusted values in input order; no correction across undefined p."""
    if any(p is None or not 0<=p<=1 for p in p_values):
        raise ValueError("Entire prespecified family must have defined p-values")
    order=np.argsort(p_values,kind="stable")
    adjusted=np.zeros(len(p_values));maximum=0.
    for rank,index in enumerate(order):
        maximum=max(maximum,(len(p_values)-rank)*p_values[index])
        adjusted[index]=min(1.,maximum)
    return adjusted.tolist()


def clustered_case_mean(rows, repetitions=2000, seed=2026):
    """Pooled case mean with patient-clustered percentile CI; None is missing."""
    patients=sorted({r["patient_id"] for r in rows})
    if not patients:
        return {"mean":None,"ci95":None,"reason":"no_cases"}
    sums=np.array([sum(r["value"] for r in rows if r["patient_id"]==p and r["value"] is not None) for p in patients])
    counts=np.array([sum(r["value"] is not None for r in rows if r["patient_id"]==p) for p in patients])
    rng=np.random.default_rng(seed);draws=[];invalid=0
    for _ in range(repetitions):
        selected=rng.integers(0,len(patients),len(patients))
        denominator=counts[selected].sum()
        if denominator:
            draws.append(float(sums[selected].sum()/denominator))
        else:
            invalid+=1
    return {"mean":float(sums.sum()/counts.sum()) if counts.sum() else None,
            "ci95":np.percentile(draws,[2.5,97.5]).tolist() if draws else None,
            "cases":int(counts.sum()),"patients":len(patients),"missing_cases":len(rows)-int(counts.sum()),
            "invalid_replicates":invalid,"unstable":invalid/repetitions>.05}
