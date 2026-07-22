#!/usr/bin/env python3
"""Synthetic unit tests for the matched-LR study runner.

These tests never load or train full MNIST.  A separate explicit CUDA smoke
gate exercises the real dataset and all four conditions.
"""

from __future__ import annotations

import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_matched_lr_validation as runner
import validate_matched_lr_results as validator


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(991)
        self.train_tasks = []
        self.test_tasks = []
        for task in range(2):
            train_x = torch.randn(8, runner.INPUT_SIZE, generator=generator)
            train_y = ((train_x[:, :10].argmax(dim=1) + task) % 10).long()
            test_x = torch.randn(8, runner.INPUT_SIZE, generator=generator)
            test_y = ((test_x[:, :10].argmax(dim=1) + task) % 10).long()
            self.train_tasks.append(TensorDataset(train_x, train_y))
            self.test_tasks.append(TensorDataset(test_x, test_y))
        self.permutations = runner.make_permutations(2)
        self.common_patch = mock.patch.dict(
            runner.COMMON_CONFIG,
            {
                "hidden_size": 8,
                "batch_size": 4,
                "evaluation_batch_size": 4,
                "pc_inference_steps": 3,
            },
        )
        self.mode_patch = mock.patch.dict(
            runner.MODE_CONFIG["smoke"],
            {"train_samples_per_task": 8, "test_samples_per_task": 8},
        )
        self.common_patch.start()
        self.mode_patch.start()

    def tearDown(self) -> None:
        self.mode_patch.stop()
        self.common_patch.stop()

    def run_synthetic(self, condition: str, diagnostics: bool = True):
        return runner.run_condition(
            condition_id=condition,
            seed=42,
            mode="smoke",
            device=torch.device("cpu"),
            train_tasks=self.train_tasks,
            test_tasks=self.test_tasks,
            permutations=self.permutations,
            diagnostics=diagnostics,
            dataset_hashes={"synthetic": "0" * 64},
        )

    def test_stream_order_is_deterministic_and_seed_specific(self) -> None:
        first = runner.sample_order(100, seed=42, task_id=1, limit=20)
        second = runner.sample_order(100, seed=42, task_id=1, limit=20)
        different = runner.sample_order(100, seed=43, task_id=1, limit=20)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))
        self.assertEqual(len(torch.unique(first)), 20)

    def test_fixed_permutations_do_not_depend_on_experimental_seed(self) -> None:
        first = runner.make_permutations(4)
        second = runner.make_permutations(4)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertTrue(torch.equal(first[0], torch.arange(runner.INPUT_SIZE)))

    def test_actual_bp_pc_initialization_is_matched(self) -> None:
        bp_model, pc_model, audit = runner.make_matched_models(16, seed=123)
        self.assertTrue(audit["all_linear_tensors_matched"])
        self.assertEqual(
            runner.linear_parameter_checksums(bp_model),
            runner.linear_parameter_checksums(pc_model),
        )

    def test_canonical_metrics_from_fixed_matrix(self) -> None:
        matrix = [
            [0.10, 0.20, 0.30],
            [0.80, 0.20, 0.30],
            [0.70, 0.75, 0.30],
            [0.60, 0.70, 0.90],
        ]
        metrics = runner.compute_continual_metrics(matrix)
        np.testing.assert_allclose(metrics["adaptation_by_task"], [0.70, 0.55, 0.60])
        np.testing.assert_allclose(metrics["bwt_by_task_0_to_n_minus_2"], [-0.20, -0.05])
        np.testing.assert_allclose(
            metrics["forgetting_by_task_0_to_n_minus_2"], [0.20, 0.05]
        )
        self.assertAlmostEqual(metrics["final_average_accuracy"], 2.2 / 3)
        self.assertAlmostEqual(metrics["mean_bwt"], -0.125)
        self.assertAlmostEqual(metrics["mean_forgetting"], 0.125)

    def test_matrix_online_counts_and_persistent_adam(self) -> None:
        result = self.run_synthetic("bp-adam")
        self.assertEqual(np.asarray(result["accuracy_matrix"]).shape, (3, 2))
        self.assertEqual(result["counts"]["actual_samples"], 16)
        self.assertEqual(result["counts"]["actual_optimizer_updates"], 4)
        self.assertEqual(
            [len(values) for values in result["online_predict_before_update"]["batch_accuracy_by_task"]],
            [2, 2],
        )
        self.assertEqual(
            [state["step_min"] for state in result["optimizer_audit"]["state_after_each_task"]],
            [2, 4],
        )
        runner.validate_result(runner.seal_result(result))

    def test_pc_classp_uses_one_persistent_optimizer(self) -> None:
        result = self.run_synthetic("pc-classp")
        audit = result["optimizer_audit"]
        self.assertEqual(audit["instances_created"], 1)
        self.assertTrue(audit["persistent_across_task_stream"])
        self.assertTrue(audit["pc_trainer_optimizer_identity_match"])
        self.assertEqual(
            [state["step_min"] for state in audit["state_after_each_task"]],
            [2, 4],
        )
        self.assertGreater(
            audit["state_after_each_task"][1]["classp_accumulator_sum"],
            audit["state_after_each_task"][0]["classp_accumulator_sum"],
        )

    def test_diagnostics_are_scientifically_neutral(self) -> None:
        enabled = self.run_synthetic("bp-classp", diagnostics=True)
        disabled = self.run_synthetic("bp-classp", diagnostics=False)
        for key in (
            "accuracy_matrix",
            "metrics",
            "online_predict_before_update",
            "final_linear_parameter_checksum",
            "counts",
        ):
            self.assertEqual(enabled[key], disabled[key])
        self.assertTrue(enabled["diagnostics"]["enabled"])
        self.assertFalse(disabled["diagnostics"]["enabled"])

    def test_neutrality_artifact_comparator(self) -> None:
        enabled_payload = self.run_synthetic("bp-classp", diagnostics=True)
        disabled_payload = self.run_synthetic("bp-classp", diagnostics=False)
        for payload in (enabled_payload, disabled_payload):
            payload["source"] = {"identity": "synthetic"}
            payload["environment"] = {"device": "cpu"}
        enabled = runner.seal_result(enabled_payload)
        disabled = runner.seal_result(disabled_payload)
        with tempfile.TemporaryDirectory() as temporary:
            enabled_path = Path(temporary) / "enabled.json"
            disabled_path = Path(temporary) / "disabled.json"
            runner.atomic_write_json_exclusive(enabled_path, enabled)
            runner.atomic_write_json_exclusive(disabled_path, disabled)
            report = validator.compare_neutrality(enabled_path, disabled_path)
        self.assertEqual(report["status"], "passed")

    def test_interaction_analysis_excludes_bridge_seed(self) -> None:
        cells = {}
        for seed_index, seed in enumerate(runner.PRIMARY_SEEDS):
            base = 0.1 + 0.01 * seed_index
            values = {
                "bp-adam": base,
                "bp-classp": base + 0.01,
                "pc-adam": base + 0.02,
                "pc-classp": base + 0.05,
            }
            for condition, value in values.items():
                cells[(condition, seed)] = {
                    "metrics": {metric: value for metric in validator.INTERACTION_METRICS}
                }
        analysis = validator.interaction_analysis(cells)
        interaction = analysis["metrics"]["final_average_accuracy"]["delta_delta"]
        self.assertEqual(len(interaction["values"]), 5)
        self.assertEqual(interaction["positive_sign_count"], 5)
        self.assertTrue(analysis["configuration_specific_differential_benefit_gate"]["passed"])

    def test_nonfinite_values_are_rejected(self) -> None:
        result = runner.seal_result(self.run_synthetic("bp-adam"))
        result["accuracy_matrix"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            runner.validate_result(result)

    def test_atomic_write_is_exclusive_and_hash_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            runner.atomic_write_json_exclusive(path, {"value": 1})
            first_hash = runner.sha256_file(path)
            with self.assertRaises(FileExistsError):
                runner.atomic_write_json_exclusive(path, {"value": 2})
            self.assertEqual(first_hash, runner.sha256_file(path))
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_output_paths_are_seed_condition_and_mode_specific(self) -> None:
        root = Path("/tmp/example")
        paths = {
            runner.output_paths(root, mode, condition, seed)["result"]
            for mode in ("smoke", "full")
            for condition in runner.CONDITIONS
            for seed in (7, 42)
        }
        self.assertEqual(len(paths), 16)

    def test_failure_artifact_is_preserved_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                runner, "load_permuted_mnist", side_effect=RuntimeError("synthetic failure")
            ):
                with contextlib.redirect_stderr(io.StringIO()):
                    return_code = runner.main(
                        [
                            "--mode", "smoke",
                            "--condition", "bp-adam",
                            "--seed", "42",
                            "--device", "cpu",
                            "--output-dir", temporary,
                        ]
                    )
            self.assertEqual(return_code, 1)
            paths = runner.output_paths(Path(temporary), "smoke", "bp-adam", 42)
            self.assertTrue(paths["failure"].exists())
            self.assertFalse(paths["result"].exists())
            failure = json.loads(paths["failure"].read_text())
            self.assertEqual(failure["status"], "failed-cell")
            self.assertIn("synthetic failure", failure["error"])

    def test_full_mode_requires_confirmation_cuda_and_frozen_seed(self) -> None:
        base = argparse_namespace(
            mode="full", seed=7, device="cuda", confirm_protocol=None, diagnostics="on"
        )
        with self.assertRaises(ValueError):
            runner.validate_cli_protocol(base)
        base.confirm_protocol = runner.PROTOCOL_ID
        base.device = "cpu"
        with self.assertRaises(ValueError):
            runner.validate_cli_protocol(base)
        base.device = "cuda"
        base.diagnostics = "off"
        with self.assertRaises(ValueError):
            runner.validate_cli_protocol(base)


def argparse_namespace(**values):
    return type("Args", (), values)()


if __name__ == "__main__":
    unittest.main(verbosity=2)
