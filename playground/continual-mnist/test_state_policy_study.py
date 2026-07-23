#!/usr/bin/env python3
"""Synthetic runner, diagnostics, validator, and analysis tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_matched_lr_validation as frozen
import run_state_policy_protocol as protocol
import run_state_policy_study as runner
import validate_state_policy_results as validator


def synthetic_source() -> dict:
    return {
        "repository": {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "tracked_dirty": False,
            "tracked_status": [],
        },
        "predictive_coding_submodule": {
            "commit": "c" * 40,
            "tree": "d" * 40,
            "tracked_dirty": False,
            "tracked_status": [],
        },
        "files": {"synthetic.py": "e" * 64},
        "dependency_lock_sha256": "f" * 64,
        "parent_protocol": {
            "protocol_id": frozen.PROTOCOL_ID,
            "runner_sha256": "0" * 64,
        },
    }


def host_description(source: dict) -> dict:
    dependencies = {
        "python": "3.11.0",
        "numpy": "2.0.0",
        "torch": "2.10.0",
        "torchvision": "0.25.0",
        "cuda_runtime": "12.6",
        "cudnn": 90000,
    }
    return {
        "source": source,
        "source_identity": runner.source_identity(source),
        "dependency_identity": dependencies,
        "dataset_file_sha256": {"MNIST/raw/train": "1" * 64},
        "hostname": "local-study-host",
        "gpu_identity": {
            "name": "NVIDIA GeForce RTX 4090",
            "uuid": "GPU-4090",
            "compute_capability": [8, 9],
        },
        "gpu_driver": "600.1",
    }


def local_manifest(source: dict) -> dict:
    return protocol.build_manifest(
        host_description(source),
        baseline_memory_mib=100,
    )


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
        common = {
            "hidden_size": 8,
            "batch_size": 4,
            "evaluation_batch_size": 4,
            "pc_inference_steps": 3,
        }
        self.patches = [
            mock.patch.dict(runner.COMMON_CONFIG, common),
            mock.patch.dict(frozen.COMMON_CONFIG, common),
            mock.patch.dict(
                runner.MODE_CONFIG["smoke"],
                {"train_samples_per_task": 8, "test_samples_per_task": 8},
            ),
            mock.patch.dict(
                frozen.MODE_CONFIG["smoke"],
                {"train_samples_per_task": 8, "test_samples_per_task": 8},
            ),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()

    def run_synthetic(self, condition: str, *, diagnostics: bool = True):
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

    def test_all_eight_conditions_and_read_only_diagnostics(self) -> None:
        for condition in runner.CONDITIONS:
            with self.subTest(condition=condition):
                result = self.run_synthetic(condition)
                audit = result["optimizer_audit"]
                expected_resets = 1 if condition.endswith("task-reset") else 0
                self.assertEqual(audit["reset_count"], expected_resets)
                self.assertEqual(audit["instances_created"], 1)
                self.assertTrue(audit["optimizer_object_retained"])
                self.assertTrue(
                    all(
                        boundary["network_parameters_preserved"]
                        for boundary in audit["boundary_summaries"]
                    )
                )
                diagnostics = result["diagnostics"]
                self.assertEqual(
                    len(diagnostics["task_parameter_displacement"]), 2
                )
                self.assertEqual(
                    len(diagnostics["first_batch_after_boundary"]), 2
                )
                final_state = diagnostics[
                    "optimizer_state_after_each_task"
                ][-1]
                self.assertGreater(final_state["state_bytes"], 0)
                if "-adam-" in condition:
                    self.assertIsNotNone(final_state["adam_first_moment_l2"])
                    self.assertIsNotNone(final_state["adam_second_moment_l2"])
                else:
                    self.assertIsNotNone(
                        final_state["classp_accumulator_sum"]
                    )
                    self.assertIsNotNone(
                        final_state["classp_accumulator_l2"]
                    )
                    self.assertIsNotNone(
                        final_state["classp_accumulator_nonzero_fraction"]
                    )
                if condition.startswith("pc-"):
                    self.assertTrue(audit["pc_trainer_optimizer_identity_match"])
                    self.assertTrue(audit["pc_latent_reset_every_batch"])
                runner.validate_result(runner.seal_result(result))

    def test_persistent_conditions_match_frozen_scientific_path(self) -> None:
        mapping = {
            "bp-adam-persistent": "bp-adam",
            "bp-classp-persistent": "bp-classp",
            "pc-adam-persistent": "pc-adam",
            "pc-classp-persistent": "pc-classp",
        }
        for new_condition, old_condition in mapping.items():
            with self.subTest(condition=new_condition):
                new = self.run_synthetic(new_condition)
                old = frozen.run_condition(
                    condition_id=old_condition,
                    seed=42,
                    mode="smoke",
                    device=torch.device("cpu"),
                    train_tasks=self.train_tasks,
                    test_tasks=self.test_tasks,
                    permutations=self.permutations,
                    diagnostics=True,
                    dataset_hashes={"synthetic": "0" * 64},
                )
                for field in (
                    "data",
                    "initialization",
                    "counts",
                    "online_predict_before_update",
                    "accuracy_matrix",
                    "metrics",
                    "final_linear_parameter_checksum",
                ):
                    self.assertEqual(new[field], old[field], field)
                self.assertEqual(
                    new["diagnostics"]["gradient_l2_by_task"],
                    old["diagnostics"]["gradient_l2_by_task"],
                )

    def test_diagnostics_are_neutral_for_every_condition(self) -> None:
        scientific_fields = (
            "data",
            "initialization",
            "optimizer_audit",
            "counts",
            "online_predict_before_update",
            "accuracy_matrix",
            "metrics",
            "final_linear_parameter_checksum",
        )
        for condition in runner.CONDITIONS:
            with self.subTest(condition=condition):
                enabled = self.run_synthetic(condition, diagnostics=True)
                disabled = self.run_synthetic(condition, diagnostics=False)
                for field in scientific_fields:
                    if field == "optimizer_audit":
                        enabled_audit = dict(enabled[field])
                        disabled_audit = dict(disabled[field])
                        for boundary in enabled_audit["boundary_summaries"]:
                            boundary.pop("optimizer_diagnostics_before")
                            boundary.pop("optimizer_diagnostics_after")
                        for boundary in disabled_audit["boundary_summaries"]:
                            boundary.pop("optimizer_diagnostics_before")
                            boundary.pop("optimizer_diagnostics_after")
                        self.assertEqual(enabled_audit, disabled_audit, field)
                    else:
                        self.assertEqual(enabled[field], disabled[field], field)

    def test_full_task_reset_records_exactly_nineteen_boundaries(self) -> None:
        generator = torch.Generator().manual_seed(112)
        train = []
        test = []
        for task in range(20):
            values = torch.randn(2, runner.INPUT_SIZE, generator=generator)
            labels = ((values[:, :10].argmax(dim=1) + task) % 10).long()
            train.append(TensorDataset(values, labels))
            test.append(TensorDataset(values.clone(), labels.clone()))
        with (
            mock.patch.dict(
                runner.COMMON_CONFIG,
                {
                    "hidden_size": 4,
                    "batch_size": 2,
                    "evaluation_batch_size": 2,
                    "pc_inference_steps": 3,
                },
            ),
            mock.patch.dict(
                runner.MODE_CONFIG["full"],
                {"train_samples_per_task": 2, "test_samples_per_task": 2},
            ),
        ):
            permutations = runner.make_permutations(20)
            for condition in (
                "bp-adam-task-reset",
                "bp-classp-task-reset",
                "pc-adam-task-reset",
                "pc-classp-task-reset",
            ):
                with self.subTest(condition=condition):
                    result = runner.run_condition(
                        condition_id=condition,
                        seed=7,
                        mode="full",
                        device=torch.device("cpu"),
                        train_tasks=train,
                        test_tasks=test,
                        permutations=permutations,
                        diagnostics=True,
                    )
                    audit = result["optimizer_audit"]
                    self.assertEqual(audit["reset_count"], 19)
                    self.assertEqual(
                        audit["reset_task_indices"], list(range(1, 20))
                    )
                    self.assertEqual(
                        [
                            state["step_min"]
                            for state in audit["state_after_each_task"]
                        ],
                        [1] * 20,
                    )

    def test_condition_paths_are_unique(self) -> None:
        paths = {
            runner.output_paths(
                Path("/tmp/example"), mode, condition, seed
            )["result"]
            for mode in ("smoke", "full")
            for condition in runner.CONDITIONS
            for seed in runner.ALL_FULL_SEEDS
        }
        self.assertEqual(len(paths), 96)


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = synthetic_source()
        self.manifest = local_manifest(self.source)

    def stub_cell(
        self,
        *,
        condition: str,
        seed: int,
        value: float = 0.5,
    ) -> dict:
        order_index = self.manifest["condition_order_by_seed"][
            str(seed)
        ].index(condition)
        host = self.manifest["host"]
        return {
            "condition_id": condition,
            "source_identity": self.manifest["source_identity"],
            "source": self.source,
            "environment": {
                "hostname": host["hostname"],
                "gpu_identity": host["gpu_identity"],
                "gpu_driver": host["gpu_driver"],
                "dependencies": self.manifest["dependency_identity"],
            },
            "config": {"seed": seed, "requested_device": "cuda"},
            "run_manifest": {
                "manifest_hash": self.manifest["manifest_hash"],
                "local_host": host,
            },
            "data": {
                "dataset_file_sha256": self.manifest[
                    "dataset_file_sha256"
                ],
                "combined_permutation_checksum": "p",
                "combined_train_order_checksum": f"stream-{seed}",
            },
            "initialization": {
                "paired_initialization_checksum": f"init-{seed}",
            },
            "counts": {
                "samples_by_task": [1, 1],
                "optimizer_updates_by_task": [1, 1],
            },
            "metrics": {
                metric: value for metric in validator.ANALYSIS_METRICS
            },
            "timing": {
                "started_at_utc": (
                    f"2026-07-23T00:00:{order_index:02d}+00:00"
                ),
                "elapsed_seconds": 1.0,
            },
            "memory": {"peak_cuda_bytes": 10},
        }

    def test_single_gpu_and_condition_order_are_enforced(self) -> None:
        seed = 7
        cells = {
            (condition, seed): self.stub_cell(
                condition=condition, seed=seed
            )
            for condition in runner.CONDITIONS
        }
        validator.validate_cross_cell_invariants(
            cells, mode="full", manifest=self.manifest
        )
        changed = cells[("bp-adam-persistent", seed)]
        changed["environment"]["gpu_identity"] = {
            "name": "other",
            "uuid": "other",
            "compute_capability": [0, 0],
        }
        with self.assertRaises(ValueError):
            validator.validate_cross_cell_invariants(
                cells, mode="full", manifest=self.manifest
            )

        cells = {
            (condition, seed): self.stub_cell(
                condition=condition, seed=seed
            )
            for condition in runner.CONDITIONS
        }
        expected = self.manifest["condition_order_by_seed"][str(seed)]
        first = cells[(expected[0], seed)]["timing"]["started_at_utc"]
        second = cells[(expected[1], seed)]["timing"]["started_at_utc"]
        cells[(expected[0], seed)]["timing"]["started_at_utc"] = second
        cells[(expected[1], seed)]["timing"]["started_at_utc"] = first
        with self.assertRaises(ValueError):
            validator.validate_cross_cell_invariants(
                cells, mode="full", manifest=self.manifest
            )

    def test_unpaired_interrupted_result_does_not_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = runner.output_paths(
                root, "full", "bp-adam-persistent", 7
            )["result"]
            runner.atomic_write_json_exclusive(path, {"interrupted": True})
            cells = validator.discover_results(
                root, mode="full", manifest=self.manifest
            )
            self.assertEqual(cells, {})
            self.assertTrue(path.exists())

    def analysis_cells(self, rescue_pc: float, omega: float) -> dict:
        cells = {}
        for seed in runner.ALL_FULL_SEEDS:
            base = seed / 1_000_000
            for metric in validator.ANALYSIS_METRICS:
                values = {
                    "bp-adam-persistent": base,
                    "bp-adam-task-reset": base,
                    "bp-classp-persistent": base,
                    "bp-classp-task-reset": base + (rescue_pc - omega),
                    "pc-adam-persistent": base,
                    "pc-adam-task-reset": base,
                    "pc-classp-persistent": base,
                    "pc-classp-task-reset": base + rescue_pc,
                }
                for condition, value in values.items():
                    cells.setdefault((condition, seed), {"metrics": {}})
                    cells[(condition, seed)]["metrics"][metric] = value
                    cells[(condition, seed)].update(
                        {
                            "environment": {
                                "hostname": "local",
                                "gpu_identity": {
                                    "name": "GPU",
                                    "uuid": "one",
                                },
                                "gpu_driver": "driver",
                                "dependencies": {"cuda_runtime": "12.6"},
                            },
                            "timing": {"elapsed_seconds": 1.0},
                            "memory": {"peak_cuda_bytes": 10},
                        }
                    )
        return cells

    def test_delta_rescue_omega_gates_bridge_and_forgetting_direction(self) -> None:
        analysis = validator.confirmatory_analysis(
            self.analysis_cells(rescue_pc=0.2, omega=0.1)
        )
        rows = analysis["metrics"]["final_average_accuracy"]["per_seed"]
        self.assertEqual(
            [row["seed"] for row in rows], list(runner.PRIMARY_SEEDS)
        )
        self.assertTrue(
            all(abs(row["rescue_pc"] - 0.2) < 1e-12 for row in rows)
        )
        self.assertTrue(
            all(abs(row["omega"] - 0.1) < 1e-12 for row in rows)
        )
        self.assertEqual(
            analysis["metrics"]["final_average_accuracy"][
                "bridge_seed_reported_separately"
            ]["seed"],
            42,
        )
        gates = analysis["hierarchical_final_accuracy_gates"]
        self.assertTrue(
            gates["gate_1_classp_specific_reset_rescue_under_pc"][
                "passed"
            ]
        )
        self.assertTrue(gates["gate_2_pc_specific_rescue"]["evaluated"])
        self.assertIn(
            "lower is better",
            analysis["metrics"]["mean_forgetting"]["direction_note"],
        )

    def test_second_gate_is_hierarchically_masked(self) -> None:
        analysis = validator.confirmatory_analysis(
            self.analysis_cells(rescue_pc=-0.2, omega=0.1)
        )
        gate = analysis["hierarchical_final_accuracy_gates"][
            "gate_2_pc_specific_rescue"
        ]
        self.assertFalse(gate["evaluated"])
        self.assertIsNone(gate["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
