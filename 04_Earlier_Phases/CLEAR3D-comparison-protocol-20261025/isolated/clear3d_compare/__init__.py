"""Protocol helpers for a future CLEAR3D comparison.

This package does not train a model, read CT volumes, or score a test set.
"""

FP_PER_SCAN = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def positive_f1(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    if min(tp, fp, fn) < 0:
        raise ValueError("counts must be non-negative")
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * precision * recall / (precision + recall)


def cpm(sensitivities):
    values = tuple(float(v) for v in sensitivities)
    if len(values) != len(FP_PER_SCAN):
        raise ValueError("CPM requires sensitivity at all seven FP/scan points")
    if any(v < 0.0 or v > 1.0 for v in values):
        raise ValueError("sensitivity must lie in [0, 1]")
    return sum(values) / len(values)


def partitions_are_separated(train, validation, test):
    groups = (set(train), set(validation), set(test))
    return groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])
