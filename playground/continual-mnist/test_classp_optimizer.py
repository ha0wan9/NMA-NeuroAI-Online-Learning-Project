#!/usr/bin/env python3
"""Algorithm-gate tests for the paper-equation CLASSP implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from classp_optimizer import CLASSP


class ClasspAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def tearDown(self) -> None:
        torch.set_default_dtype(torch.float32)

    @staticmethod
    def _step(optimizer: CLASSP, parameter: torch.nn.Parameter, gradient) -> None:
        parameter.grad = torch.as_tensor(gradient, dtype=parameter.dtype)
        optimizer.step()

    def test_p1_hand_calculation_and_persistent_accumulator(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
        optimizer = CLASSP([parameter], lr=0.2, p=1, eps=0.5)

        self._step(optimizer, parameter, [2.0, -1.0])
        expected_first = torch.tensor([
            3.0 - 0.2 * 2.0 / (0.5 + 2.0),
            -2.0 - 0.2 * -1.0 / (0.5 + 1.0),
        ])
        self.assertTrue(torch.allclose(parameter, expected_first))

        self._step(optimizer, parameter, [1.0, 3.0])
        expected_second = expected_first - torch.tensor([
            0.2 * 1.0 / (0.5 + 3.0),
            0.2 * 3.0 / (0.5 + 4.0),
        ])
        self.assertTrue(torch.allclose(parameter, expected_second))
        self.assertTrue(
            torch.equal(optimizer.state[parameter]["grad_sum"], torch.tensor([3.0, 4.0]))
        )

    def test_p2_hand_calculation(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = CLASSP([parameter], lr=0.1, p=2, eps=0.25)
        self._step(optimizer, parameter, [3.0, 4.0])
        expected = torch.tensor([1.0, 2.0]) - 0.1 * torch.tensor([3.0, 4.0]) / torch.sqrt(
            torch.tensor([9.25, 16.25])
        )
        self.assertTrue(torch.allclose(parameter, expected))
        self.assertTrue(
            torch.equal(optimizer.state[parameter]["grad_sum"], torch.tensor([9.0, 16.0]))
        )

    def test_threshold_is_elementwise_and_preserves_rejected_state(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        optimizer = CLASSP([parameter], lr=0.1, p=2, threshold=1.0, eps=0.5)

        self._step(optimizer, parameter, [2.0, 0.5, -3.0])
        state_after_first = optimizer.state[parameter]["grad_sum"].clone()
        value_after_first = parameter.detach().clone()
        self.assertTrue(torch.equal(state_after_first, torch.tensor([4.0, 0.0, 9.0])))

        self._step(optimizer, parameter, [0.5, 2.0, 0.25])
        self.assertEqual(parameter[0].item(), value_after_first[0].item())
        self.assertEqual(parameter[2].item(), value_after_first[2].item())
        self.assertTrue(
            torch.equal(optimizer.state[parameter]["grad_sum"], torch.tensor([4.0, 4.0, 9.0]))
        )

    def test_strict_threshold_rejects_equal_squared_gradient(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = CLASSP([parameter], lr=0.1, threshold=4.0, eps=1.0)
        self._step(optimizer, parameter, [2.0])
        self.assertEqual(parameter.item(), 1.0)
        self.assertEqual(optimizer.state[parameter]["grad_sum"].item(), 0.0)

    def test_apply_decay_false_accumulates_and_uses_unscaled_update(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = CLASSP([parameter], lr=0.25, p=2, eps=1.0, apply_decay=False)

        self._step(optimizer, parameter, [2.0])
        self.assertEqual(parameter.item(), 1.5)
        self.assertEqual(optimizer.state[parameter]["grad_sum"].item(), 4.0)

        parameter.grad = torch.tensor([1.0])
        optimizer.step(apply_decay=True)
        expected = 1.5 - 0.25 / (6.0 ** 0.5)
        self.assertAlmostEqual(parameter.item(), expected)
        self.assertEqual(optimizer.state[parameter]["grad_sum"].item(), 5.0)

    def test_p2_threshold_zero_matches_pytorch_adagrad(self) -> None:
        initial = torch.tensor([1.0, -2.0, 0.5])
        classp_parameter = torch.nn.Parameter(initial.clone())
        adagrad_parameter = torch.nn.Parameter(initial.clone())
        epsilon_inside_root = 1e-4

        classp = CLASSP(
            [classp_parameter], lr=0.07, p=2, threshold=0.0,
            eps=epsilon_inside_root,
        )
        adagrad = torch.optim.Adagrad(
            [adagrad_parameter], lr=0.07, eps=0.0,
            initial_accumulator_value=epsilon_inside_root,
        )

        for gradient in ([0.4, -0.2, 0.0], [-0.1, 0.3, 0.6], [0.2, 0.1, -0.4]):
            classp_parameter.grad = torch.tensor(gradient)
            adagrad_parameter.grad = torch.tensor(gradient)
            classp.step()
            adagrad.step()
            self.assertTrue(torch.allclose(classp_parameter, adagrad_parameter, atol=1e-12))

    def test_nonfinite_gradient_fails_fast(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = CLASSP([parameter])
        parameter.grad = torch.tensor([float("nan")])
        with self.assertRaises(FloatingPointError):
            optimizer.step()


if __name__ == "__main__":
    unittest.main(verbosity=2)
