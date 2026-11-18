import unittest

from clear3d_compare import cpm, partitions_are_separated, positive_f1
from clear3d_compare.eligibility import eligible_candidate, eligible_reference
from clear3d_compare.gate import launch_allowed


class MetricTests(unittest.TestCase):
    def test_redesign_reported_f1_is_the_count_formula(self):
        self.assertAlmostEqual(positive_f1(70, 6, 16), 0.8641975308641974)

    def test_zero_true_positives(self):
        self.assertEqual(positive_f1(0, 3, 4), 0.0)

    def test_cpm_requires_seven_points(self):
        with self.assertRaises(ValueError):
            cpm([0.9, 0.9])
        self.assertAlmostEqual(cpm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), 0.4)

    def test_partition_overlap_is_rejected(self):
        self.assertTrue(partitions_are_separated(["a"], ["b"], ["c"]))
        self.assertFalse(partitions_are_separated(["a"], ["a"], ["c"]))


class EligibilityTests(unittest.TestCase):
    def test_faint_nodule_stays_eligible(self):
        nodule = {"zyx": (10, 20, 30), "prespecified_include": True, "brightness_pass": False}
        self.assertTrue(eligible_reference(nodule, (40, 40, 40)))

    def test_outside_volume_is_ineligible(self):
        nodule = {"zyx": (50, 0, 0), "prespecified_include": True}
        self.assertFalse(eligible_reference(nodule, (40, 40, 40)))

    def test_candidate_must_belong_to_the_declared_series(self):
        self.assertFalse(eligible_candidate({"series_uid": "other", "class_label": 0}, ["allowed"]))


class GateTests(unittest.TestCase):
    def test_draft_population_cannot_launch(self):
        draft = {"reference_release": "HOLD", "cap4000": "NOT_ADOPTED", "evaluation_population": "DRAFT", "gpu_budget": "DRAFT"}
        self.assertFalse(launch_allowed(draft))


if __name__ == "__main__":
    unittest.main()
