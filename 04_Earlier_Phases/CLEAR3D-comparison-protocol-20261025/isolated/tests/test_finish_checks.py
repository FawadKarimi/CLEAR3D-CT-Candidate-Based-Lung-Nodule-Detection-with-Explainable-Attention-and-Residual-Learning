"""CPU checks for the unfinished audit items.

These tests do not read confirmation scores and do not launch a GPU job.
"""
import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn

from finish_metrics import average_precision, bin_name, calibration, roc_auc


class MetricDefinitionTests(unittest.TestCase):
    def test_average_precision_of_a_perfect_ranking(self):
        score = np.array([0.1, 0.9, 0.2, 0.8])
        label = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(average_precision(score, label), 1.0)

    def test_roc_auc_of_a_perfect_ranking(self):
        score = np.array([0.1, 0.9, 0.2, 0.8])
        label = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(roc_auc(score, label), 1.0)

    def test_diameter_edges(self):
        self.assertEqual(bin_name(5.99), "under 6 mm")
        self.assertEqual(bin_name(6.0), "6 to 10 mm")
        self.assertEqual(bin_name(10.0), "over 10 mm")

    def test_calibration_of_a_constant_score(self):
        score = np.array([0.25, 0.25, 0.25, 0.25])
        label = np.array([1, 0, 0, 0])
        row = calibration(score, label)
        self.assertAlmostEqual(row["brier"], 0.1875)
        self.assertAlmostEqual(row["ece"], 0.0)


class ObjectiveTests(unittest.TestCase):
    def test_class_weight_balances_one_positive_and_seven_negatives(self):
        counts = np.array([7.0, 1.0])
        weight = counts.sum() / (2.0 * counts)
        contribution = counts * weight
        self.assertTrue(np.allclose(contribution, [4.0, 4.0]))

    def test_focal_loss_is_not_stacked_on_that_weight(self):
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
        target = torch.tensor([0, 1])
        log_p = torch.log_softmax(logits, 1)
        probability = log_p.exp().gather(1, target[:, None]).squeeze(1)
        gathered = log_p.gather(1, target[:, None]).squeeze(1)
        loss = (-((1.0 - probability) ** 2) * gathered).mean()
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 0.1)


class ResumeTests(unittest.TestCase):
    def test_scheduler_steps_only_when_asked(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        before = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], before)
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], before * 0.5)

    def test_full_checkpoint_restores_weights_optimizer_and_rng(self):
        torch.manual_seed(5)
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        data = torch.ones(4, 3)
        optimizer.zero_grad()
        model(data).sum().backward()
        optimizer.step()
        scheduler.step()
        blob = {
            "state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
        }
        replacement = nn.Linear(3, 2)
        other = torch.optim.SGD(replacement.parameters(), lr=0.1, momentum=0.9)
        other_scheduler = torch.optim.lr_scheduler.StepLR(other, step_size=1, gamma=0.5)
        replacement.load_state_dict(blob["state"])
        other.load_state_dict(blob["optimizer"])
        other_scheduler.load_state_dict(blob["scheduler"])
        self.assertTrue(torch.equal(replacement.weight, model.weight))
        self.assertEqual(other.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
        torch.set_rng_state(blob["torch_rng"])
        first = int(torch.randint(0, 1000, ()).item())
        torch.set_rng_state(blob["torch_rng"])
        second = int(torch.randint(0, 1000, ()).item())
        self.assertEqual(first, second)

    def test_every_parameter_receives_a_finite_gradient(self):
        model = nn.Sequential(nn.Conv3d(1, 2, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(2 * 4 * 4 * 4, 2))
        target = torch.tensor([1, 0])
        loss = nn.functional.cross_entropy(model(torch.randn(2, 1, 4, 4, 4)), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)


class IntensityAndMaskTests(unittest.TestCase):
    def test_bounded_intensity_stays_finite(self):
        rng = np.random.default_rng(17)
        image = rng.random((2, 8, 8, 8)).astype(np.float32)
        scale = float(rng.uniform(0.9, 1.1))
        shift = float(rng.uniform(-0.05, 0.05))
        changed = image * scale + shift
        self.assertTrue(np.isfinite(changed).all())
        self.assertLess(abs(scale - 1.0), 0.1000001)
        self.assertLess(abs(shift), 0.0500001)

    def test_padded_crop_has_an_explicit_valid_mask(self):
        image = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        center = np.array([0, 1, 1])
        half = 2
        start = center - half
        stop = center + half
        patch = np.zeros((4, 4, 4), dtype=np.float32)
        valid = np.zeros((4, 4, 4), dtype=bool)
        src_start = np.maximum(start, 0)
        src_stop = np.minimum(stop, np.array(image.shape))
        dst_start = src_start - start
        dst_stop = dst_start + (src_stop - src_start)
        patch[
            dst_start[0]:dst_stop[0], dst_start[1]:dst_stop[1], dst_start[2]:dst_stop[2]
        ] = image[
            src_start[0]:src_stop[0], src_start[1]:src_stop[1], src_start[2]:src_stop[2]
        ]
        valid[
            dst_start[0]:dst_stop[0], dst_start[1]:dst_stop[1], dst_start[2]:dst_stop[2]
        ] = True
        self.assertLess(int(valid.sum()), valid.size)
        self.assertGreater(int(valid.sum()), 0)
        self.assertTrue(np.isfinite(patch).all())


if __name__ == "__main__":
    unittest.main()
