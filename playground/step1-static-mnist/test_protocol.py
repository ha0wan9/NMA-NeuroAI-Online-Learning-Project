from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).with_name("experiment.py")
SPEC = importlib.util.spec_from_file_location("step1_static_mnist_experiment", MODULE_PATH)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


class ProtocolTests(unittest.TestCase):
    def test_print_config_is_static_and_claim_limited(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = experiment.main(["--print-config"])
        config = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(config["protocol_id"], "step1-static-mnist-v0.1")
        self.assertFalse(config["constraints"]["continual_learning"])
        self.assertFalse(config["constraints"]["result_claims_authorized"])

    def test_stratified_realized_order_is_deterministic_and_balanced(self):
        targets = torch.arange(10).repeat_interleave(20)
        first = experiment.stratified_indices(targets, 100, seed=7)
        second = experiment.stratified_indices(targets, 100, seed=7)
        self.assertEqual(first, second)
        selected_targets = targets[first]
        self.assertEqual(torch.bincount(selected_targets, minlength=10).tolist(), [10] * 10)

    def test_pc_and_bp_start_with_matching_trainable_parameters(self):
        config = experiment.CONFIGS["smoke"]
        pc_model = experiment.build_model(config, "pc", seed=3)
        bp_model = experiment.build_model(config, "bp", seed=3)
        self.assertEqual(
            experiment.trainable_parameter_hash(pc_model),
            experiment.trainable_parameter_hash(bp_model),
        )
        self.assertEqual(
            [tuple(parameter.shape) for parameter in pc_model.parameters()],
            [tuple(parameter.shape) for parameter in bp_model.parameters()],
        )

    def test_atomic_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            experiment.atomic_write_new({"status": "completed"}, path)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                experiment.atomic_write_new({"status": "different"}, path)
            self.assertEqual(json.loads(path.read_text()), {"status": "completed"})

    def test_record_validation_enforces_matched_stream_and_initialization(self):
        identity = {"realized_train_order_sha256": "same"}
        record = {
            "schema_version": 1,
            "protocol_id": experiment.PROTOCOL_ID,
            "interpretation": experiment.INTERPRETATION,
            "status": "completed",
            "config": {},
            "seed": 0,
            "methods_requested": ["pc", "bp"],
            "runs": [
                {"status": "completed", "stream_identity": identity,
                 "initial_parameters_sha256": "same"},
                {"status": "completed", "stream_identity": identity,
                 "initial_parameters_sha256": "same"},
            ],
            "provenance": {},
        }
        experiment.validate_record(record)
        record["runs"][1]["stream_identity"] = {"realized_train_order_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "realized stream"):
            experiment.validate_record(record)

    def test_stream_identity_separates_train_and_test(self):
        train = list(range(100))
        test = list(range(100, 200))
        self.assertNotEqual(experiment.sequence_hash(train), experiment.sequence_hash(test))


if __name__ == "__main__":
    unittest.main()
