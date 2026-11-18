import numpy as np
import pytest
from lidc3d.evaluation import match, DetectionCurves, paired_bootstrap


def ref(id, x=0, radius=1):
    return {"id": id, "center_lps": [x,0,0], "radius_mm": radius}


def pred(id, x=0, score=.5):
    return {"id": id, "center_lps": [x,0,0], "score": score}


def case(id="s", patient="p", positives=None, predictions=None):
    return {"scan_id": id, "patient_id": patient, "positive": positives or [], "predictions": predictions or [], "ignored": []}


def test_max_cardinality_beats_greedy_nearest():
    # Shared candidate at 0 can match either; candidate -1 only the first.
    result = match([pred("shared",0),pred("only_a",-1)], [ref("a",-.5,1),ref("b",.5,1)], [])
    assert result["tp"] == 2 and result["fp"] == 0


def test_boundary_duplicate_and_ignore_precedence():
    result = match([pred("a",0),pred("b",1),pred("ignored",4),pred("fp",10)],
                   [ref("r",0,1)], [ref("ignore",0,5)])
    assert (result["tp"],result["fp"],result["ignored"]) == (1,2,1)


def test_empty_scans_remain_in_denominator_and_ties_joint():
    cases = [case("a","p1",[ref("r")],[pred("tp",0,.5),pred("fp",10,.5)]), case("b","p2")]
    curve = DetectionCurves(cases)
    summary = curve.summarize()
    assert summary["nscans"] == 2
    assert summary["fp_per_scan"] == [0, .5]
    assert summary["sensitivity_at_budgets"][:3] == [0,0,1]
    assert len(summary["thresholds"]) == 2
    assert curve.operating_threshold(.125) is None


def test_uncovered_reference_limits_ceiling():
    curve = DetectionCurves([case(positives=[ref("a"),ref("b",20)],predictions=[pred("a")])])
    assert curve.summarize()["candidate_ceiling"] == .5
    assert max(curve.summarize()["sensitivity"]) <= .5


def test_bootstrap_multiplicity_equals_explicit_replicate_ids():
    original = [case("s1","p1",[ref("r")],[pred("a")]), case("s2","p2",[],[pred("fp",10)])]
    weighted = DetectionCurves(original).summarize([2,1])
    explicit = [case("s1a","p1a",[ref("r")],[pred("a")]), case("s1b","p1b",[ref("r")],[pred("a")]), original[1]]
    replicated = DetectionCurves(explicit).summarize()
    assert weighted["cpm"] == replicated["cpm"]
    assert weighted["fp_per_scan"] == replicated["fp_per_scan"]


def test_paired_bootstrap_identical_models_zero_difference():
    cases = [case("s1","p1",[ref("r")],[pred("a")]),case("s2","p2",[ref("r")],[pred("a")])]
    curves = DetectionCurves(cases)
    result = paired_bootstrap({"hybrid":[curves]*3,"cnn":[curves]*3}, 20)
    assert result["paired_contrasts"]["hybrid_minus_cnn"]["ci95"] == [0.,0.]


def test_undefined_all_negative_cohort():
    result = DetectionCurves([case()]).summarize()
    assert result["cpm"] is None


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        DetectionCurves([case(predictions=[pred("bad",score=float('nan'))])])
