#!/usr/bin/env python3
"""Dependency-free local protocol and synthetic lifecycle tests."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_state_policy_protocol as protocol
import run_state_policy_study as runner
import validate_state_policy_results as validator
from test_state_policy_study import host_description, synthetic_source


class ManifestAndIdleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = synthetic_source()
        self.description = host_description(self.source)

    def test_manifest_self_hash_and_tamper_detection(self) -> None:
        manifest = protocol.build_manifest(
            self.description, baseline_memory_mib=100
        )
        self.assertEqual(
            manifest["manifest_hash"], runner.manifest_hash(manifest)
        )
        self.assertEqual(manifest["kind"], "single-gpu-local")
        self.assertFalse(manifest["execution"]["distributed_execution"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            protocol.write_json_exclusive(path, manifest)
            self.assertEqual(
                runner.load_run_manifest(path)["manifest_hash"],
                manifest["manifest_hash"],
            )
            tampered = dict(manifest)
            tampered["host"] = {**manifest["host"], "hostname": "other"}
            path.unlink()
            protocol.write_json_exclusive(path, tampered)
            with self.assertRaises(ValueError):
                runner.load_run_manifest(path)

    def test_williams_rows_cover_all_six_seeds(self) -> None:
        orders = protocol.williams_condition_orders()
        self.assertEqual(list(orders), list(protocol.ORDERED_SEEDS))
        self.assertEqual(
            orders[runner.BRIDGE_SEED],
            [
                protocol.CANONICAL_CONDITIONS[index - 1]
                for index in protocol.WILLIAMS_BASE_ROW
            ],
        )
        self.assertTrue(
            all(
                sorted(order)
                == sorted(protocol.CANONICAL_CONDITIONS)
                for order in orders.values()
            )
        )

    def test_manifest_rejects_source_data_dependency_and_gpu_mismatch(self) -> None:
        manifest = protocol.build_manifest(
            self.description, baseline_memory_mib=100
        )
        for key, replacement in (
            ("source_identity", "wrong"),
            ("dependency_identity", {"torch": "wrong"}),
            ("dataset_file_sha256", {"raw": "wrong"}),
            ("gpu_identity", {"name": "wrong"}),
        ):
            changed = dict(self.description)
            changed[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_manifest_description(manifest, changed)

    @staticmethod
    def sample(**changes):
        value = {
            "utilization_percent": 0,
            "temperature_c": 30,
            "memory_used_mib": 100,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_uuid": "GPU-4090",
            "foreign_processes": [],
            "available_ram_bytes": protocol.MIN_AVAILABLE_RAM_BYTES,
            "disk_free_bytes": protocol.MIN_DISK_BYTES,
        }
        value.update(changes)
        return value

    def test_idle_gate_requires_two_samples_ten_seconds_apart(self) -> None:
        samples = iter(
            [
                self.sample(utilization_percent=20),
                self.sample(),
                self.sample(),
            ]
        )
        sleeps = []
        accepted = protocol.wait_for_idle(
            lambda: next(samples),
            baseline_memory_mib=100,
            timeout_seconds=60,
            interval_seconds=10,
            sleep_fn=sleeps.append,
        )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(sleeps, [10, 10])

    def test_idle_gate_rejects_each_resource_violation(self) -> None:
        failures = (
            self.sample(utilization_percent=6),
            self.sample(temperature_c=41),
            self.sample(memory_used_mib=613),
            self.sample(foreign_processes=["0 123 G game"]),
            self.sample(
                available_ram_bytes=protocol.MIN_AVAILABLE_RAM_BYTES - 1
            ),
            self.sample(disk_free_bytes=protocol.MIN_DISK_BYTES - 1),
        )
        self.assertTrue(
            all(
                not protocol.idle_sample_passes(sample, 100)
                for sample in failures
            )
        )

    def test_attempt_paths_and_exclusive_records_do_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cell"
            first = protocol.next_attempt_directory(root)
            first.mkdir(parents=True)
            (first / "partial.tmp").write_text("preserve", encoding="utf-8")
            second = protocol.next_attempt_directory(root)
            self.assertEqual(second.name, "attempt-002")
            self.assertEqual(
                (first / "partial.tmp").read_text(encoding="utf-8"),
                "preserve",
            )
            record = Path(temporary) / "record.json"
            protocol.same_or_refuse(record, {"value": 1})
            protocol.same_or_refuse(record, {"value": 1})
            with self.assertRaises(FileExistsError):
                protocol.same_or_refuse(record, {"value": 2})

    def test_cli_exposes_only_local_protocol_commands(self) -> None:
        for command in (
            "preflight",
            "smoke",
            "bridge",
            "heldout",
            "resume",
            "status",
            "validate",
        ):
            argv = [command]
            if command == "resume":
                argv.extend(
                    [
                        "--stage",
                        "smoke",
                        "--reviewed-failure-sha256",
                        "x",
                    ]
                )
            if command == "validate":
                argv.extend(["--stage", "smoke"])
            args = protocol.parse_args(argv)
            self.assertEqual(args.command, command)


class SyntheticLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = synthetic_source()
        self.description = host_description(self.source)
        self.manifest = protocol.build_manifest(
            self.description, baseline_memory_mib=100
        )
        self.counter = 0

    def invoke(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            stderr
        ):
            return protocol.main(argv), stdout.getvalue(), stderr.getvalue()

    def fake_preflight(
        self, *, state_root: Path, data_root: Path
    ) -> tuple[dict, list[dict]]:
        del data_root
        layout = protocol.state_layout(state_root)
        protocol.ensure_layout(layout)
        protocol.write_json_exclusive(layout["manifest"], self.manifest)
        protocol.append_jsonl(
            layout["operational_log"],
            {"event": "synthetic-preflight"},
        )
        return self.manifest, [{"synthetic": True}, {"synthetic": True}]

    def fake_result(
        self,
        *,
        mode: str,
        condition: str,
        seed: int,
    ) -> dict:
        self.counter += 1
        host = self.manifest["host"]
        optimizer = runner.CONDITIONS[condition]["optimizer"]
        state = {
            "task_index": 0,
            "parameters_with_state": 6,
            "state_tensor_count": 18 if optimizer == "adam" else 6,
            "state_tensor_elements": 100,
            "state_bytes": 400,
            "step_min": 1,
            "step_max": 1,
            "adam_first_moment_l2": 0.1 if optimizer == "adam" else None,
            "adam_second_moment_l2": 0.2 if optimizer == "adam" else None,
            "classp_accumulator_sum": (
                0.3 if optimizer == "classp" else None
            ),
            "classp_accumulator_l2": (
                0.4 if optimizer == "classp" else None
            ),
            "classp_accumulator_nonzero_fraction": (
                0.5 if optimizer == "classp" else None
            ),
        }
        metric_value = (
            list(runner.CONDITIONS).index(condition) / 100
            + seed / 10_000_000
        )
        result = {
            "schema_version": 1,
            "protocol_id": runner.PROTOCOL_ID,
            "status": "validated-cell",
            "mode": mode,
            "condition_id": condition,
            "config": {
                "seed": seed,
                "requested_device": "cuda",
                "diagnostics": True,
            },
            "source": self.source,
            "source_identity": self.manifest["source_identity"],
            "environment": {
                "hostname": host["hostname"],
                "gpu_identity": host["gpu_identity"],
                "gpu_driver": host["gpu_driver"],
                "dependencies": self.manifest["dependency_identity"],
            },
            "run_manifest": {
                "manifest_hash": self.manifest["manifest_hash"],
                "local_host": host,
            },
            "data": {
                "dataset_file_sha256": self.manifest[
                    "dataset_file_sha256"
                ],
                "combined_permutation_checksum": "permutations",
                "combined_train_order_checksum": f"stream-{seed}",
            },
            "initialization": {
                "paired_initialization_checksum": f"init-{seed}",
            },
            "optimizer_audit": {"boundary_summaries": []},
            "counts": {
                "samples_by_task": [1],
                "optimizer_updates_by_task": [1],
            },
            "online_predict_before_update": {"batch_accuracy_by_task": [[0.5]]},
            "accuracy_matrix": [[0.0], [metric_value]],
            "metrics": {
                metric: metric_value
                for metric in validator.ANALYSIS_METRICS
            },
            "diagnostics": {
                "enabled": True,
                "gradient_l2_by_task": [[0.1]],
                "task_parameter_displacement": [
                    {
                        "task_index": 0,
                        "l2": 0.2,
                        "relative_l2": 0.02,
                        "parameter_l2_before": 10.0,
                        "parameter_l2_after": 10.1,
                    }
                ],
                "optimizer_state_after_each_task": [state],
                "first_batch_after_boundary": [
                    {
                        "task_index": 0,
                        "boundary_action": "initial-task-no-boundary",
                        "gradient_l2": 0.1,
                        "parameter_update": {
                            "l2": 0.01,
                            "relative_l2": 0.001,
                            "parameter_l2_before": 10.0,
                            "parameter_l2_after": 10.0,
                        },
                        "optimizer_state_after_update": state,
                    }
                ],
            },
            "final_linear_parameter_checksum": f"final-{condition}-{seed}",
            "timing": {
                "started_at_utc": f"{self.counter:06d}",
                "completed_at_utc": f"{self.counter:06d}-done",
                "elapsed_seconds": 1.0,
            },
            "memory": {"peak_cuda_bytes": 1024},
        }
        return runner.seal_result(result)

    def fake_run_cell(self, **kwargs) -> dict:
        layout = kwargs["layout"]
        stage = kwargs["stage"]
        condition = kwargs["condition"]
        seed = kwargs["seed"]
        pass_index = kwargs["pass_index"]
        mode = "smoke" if stage == "smoke" else "full"
        cell_root = protocol.cell_root_for(
            layout,
            stage=stage,
            condition=condition,
            seed=seed,
            pass_index=pass_index,
        )
        cell_root.mkdir(parents=True, exist_ok=True)
        existing = protocol.find_valid_pair(
            cell_root,
            manifest=self.manifest,
            mode=mode,
            condition=condition,
            seed=seed,
        )
        if existing is not None:
            return {
                "status": "skipped-valid",
                "condition_id": condition,
                "seed": seed,
            }
        attempt = protocol.next_attempt_directory(cell_root)
        attempt.mkdir()
        result = self.fake_result(
            mode=mode, condition=condition, seed=seed
        )
        paths = runner.output_paths(attempt, mode, condition, seed)
        runner.atomic_write_json_exclusive(paths["result"], result)
        receipt = {
            "schema_version": 1,
            "protocol_id": runner.PROTOCOL_ID,
            "artifact": paths["result"].name,
            "artifact_sha256": runner.sha256_file(paths["result"]),
            "artifact_bytes": paths["result"].stat().st_size,
            "canonical_payload_sha256": result["integrity"][
                "canonical_payload_sha256"
            ],
            "manifest_hash": self.manifest["manifest_hash"],
            "source_identity": self.manifest["source_identity"],
        }
        runner.atomic_write_json_exclusive(paths["receipt"], receipt)
        protocol.append_jsonl(
            layout["operational_log"],
            {
                "event": "synthetic-cell",
                "stage": stage,
                "condition_id": condition,
                "seed": seed,
            },
        )
        return {
            "status": "validated-cell",
            "condition_id": condition,
            "seed": seed,
            "result": str(paths["result"]),
        }

    def test_preflight_to_complete_validation_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "state"
            common = ["--state-root", str(root)]
            with (
                mock.patch.object(
                    protocol,
                    "perform_preflight",
                    side_effect=self.fake_preflight,
                ),
                mock.patch.object(
                    protocol, "run_cell", side_effect=self.fake_run_cell
                ),
                mock.patch.object(runner, "validate_result"),
            ):
                code, _, error = self.invoke(["preflight", *common])
                self.assertEqual(code, 0, error)
                code, _, error = self.invoke(["smoke", *common])
                self.assertEqual(code, 0, error)
                layout = protocol.state_layout(root)
                self.assertTrue(
                    (
                        layout["reports"]
                        / "smoke-optimizer-state-report.md"
                    ).exists()
                )

                code, _, error = self.invoke(["bridge", *common])
                self.assertEqual(code, 0, error)
                bridge_path = (
                    layout["reports"] / "bridge-intuition-report.md"
                )
                bridge_hash = runner.sha256_file(bridge_path)
                code, status_stdout, error = self.invoke(
                    ["status", *common]
                )
                self.assertEqual(code, 0, error)
                for forbidden in (
                    "final_average_accuracy",
                    "mean_bwt",
                    "delta_",
                    "omega",
                ):
                    self.assertNotIn(forbidden, status_stdout.lower())

                code, stdout, _ = self.invoke(["heldout", *common])
                self.assertEqual(code, 1)
                self.assertNotIn("delta_", stdout)
                self.assertEqual(
                    protocol.stage_progress(
                        layout=layout,
                        manifest=self.manifest,
                        stage="heldout",
                    )["completed_cells"],
                    0,
                )

                code, stdout, error = self.invoke(
                    [
                        "heldout",
                        *common,
                        "--confirm-protocol",
                        runner.PROTOCOL_ID,
                        "--bridge-review-sha256",
                        bridge_hash,
                    ]
                )
                self.assertEqual(code, 0, error)
                self.assertNotIn("delta_", stdout)
                self.assertTrue(
                    (layout["reports"] / "complete-validation.json").exists()
                )
                self.assertTrue(
                    (layout["reports"] / "complete-synthesis.md").exists()
                )
                self.assertTrue(
                    (layout["reports"] / "artifact-manifest.sha256").exists()
                )

                code, stdout, error = self.invoke(
                    ["validate", *common, "--stage", "complete"]
                )
                self.assertEqual(code, 0, error)
                self.assertNotIn("delta_", stdout)
                self.assertEqual(
                    protocol.stage_progress(
                        layout=layout,
                        manifest=self.manifest,
                        stage="heldout",
                    )["completed_cells"],
                    40,
                )

    def test_failure_stops_queue_and_creates_reviewable_halt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "state"
            layout = protocol.state_layout(root)
            protocol.ensure_layout(layout)
            protocol.write_json_exclusive(
                layout["manifest"], self.manifest
            )
            attempt = (
                layout["smoke_pass_1"]
                / "bp-adam-persistent"
                / "attempt-001"
            )
            failure = protocol.CellFailure("synthetic invariant", attempt)
            with mock.patch.object(
                protocol, "run_cell", side_effect=failure
            ) as run_cell:
                with self.assertRaises(protocol.CellFailure):
                    protocol.run_stage(
                        layout=layout,
                        manifest=self.manifest,
                        stage="smoke",
                        data_root=Path("/synthetic"),
                        resume=False,
                    )
            self.assertEqual(run_cell.call_count, 1)
            halts = list(layout["halts"].glob("halt-*.json"))
            self.assertEqual(len(halts), 1)
            status = protocol.command_status(layout)
            self.assertIn(
                "resume_review_sha256", status["stages"]["smoke"]
            )

    def test_resume_requires_exact_structural_review_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "state"
            layout = protocol.state_layout(root)
            protocol.ensure_layout(layout)
            protocol.write_json_exclusive(
                layout["manifest"], self.manifest
            )
            token = protocol.resume_review_sha256(
                layout=layout,
                manifest=self.manifest,
                stage="smoke",
            )
            with mock.patch.object(
                protocol, "run_cell", side_effect=RuntimeError("stop")
            ):
                code, _, error = self.invoke(
                    [
                        "resume",
                        "--state-root",
                        str(root),
                        "--stage",
                        "smoke",
                        "--reviewed-failure-sha256",
                        "wrong",
                    ]
                )
                self.assertEqual(code, 1)
                self.assertIn(token, error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
