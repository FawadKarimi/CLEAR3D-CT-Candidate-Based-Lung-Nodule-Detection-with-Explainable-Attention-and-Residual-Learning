"""Deterministic patient allocation independent of candidate counts."""
import numpy as np


def patient_split(scans, seed=42):
    groups = {}
    for scan in scans:
        groups.setdefault(scan["patient_id"], []).extend(r["diameter_mm"] for r in scan["positive"])
    def label(values):
        return "negative" if not values else "small" if min(values) < 6 else "medium" if min(values) < 10 else "large"
    strata = {p: label(d) for p, d in groups.items()}
    def counts():
        return {s: sum(v == s for v in strata.values()) for s in set(strata.values())}
    # At least seven permits >=1 patient in each 15% holdout after rounding.
    fallback = "size_stratified"
    if min(counts().values(), default=0) < 7:
        strata = {p: ("positive" if d else "negative") for p, d in groups.items()}
        fallback = "positive_status_stratified"
    if min(counts().values(), default=0) < 7:
        strata = {p: "all" for p in groups}
        fallback = "unstratified"
    rng, result = np.random.default_rng(seed), {}
    for stratum in sorted(set(strata.values())):
        ids = sorted(p for p in groups if strata[p] == stratum)
        ids = list(rng.permutation(ids))
        n_val = int(np.floor(len(ids)*0.15+0.5))
        n_test = int(np.floor(len(ids)*0.15+0.5))
        n_train = len(ids)-n_val-n_test
        if min(n_train, n_val, n_test) < 1:
            raise ValueError("Too few patients for three nonempty partitions")
        for i, patient in enumerate(ids):
            result[str(patient)] = "train" if i < n_train else "val" if i < n_train+n_val else "test"
    return result, {"seed": seed, "allocation": fallback,
        "rounding": "round each 15% holdout half-up; training gets remainder per stratum",
        "strata": strata, "assignments": result}
