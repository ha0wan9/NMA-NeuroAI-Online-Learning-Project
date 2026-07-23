"""Dependency-free checks for the frozen Step 1 protocol."""

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("experiment.py")
SPEC = importlib.util.spec_from_file_location("step1_experiment", MODULE_PATH)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


class ProtocolTests(unittest.TestCase):
    def test_frozen_defaults(self):
        protocol = experiment.protocol_from_args(experiment.parse_args([]))
        self.assertEqual(protocol.seeds, (7, 42, 123))
        self.assertEqual(protocol.methods, ("bp", "pc"))
        self.assertEqual(protocol.batch_size, 500)
        self.assertEqual(protocol.hidden_layers, 2)
        self.assertEqual(protocol.pc_steps, 20)
        self.assertEqual(protocol.split_seed, 20260717)

    def test_smoke_overrides(self):
        args = experiment.parse_args([
            "--seeds", "42", "--epochs", "1", "--batch-size", "64",
            "--pc-steps", "4", "--max-train-samples", "256",
        ])
        protocol = experiment.protocol_from_args(args)
        self.assertEqual(protocol.seeds, (42,))
        self.assertEqual(protocol.epochs, 1)
        self.assertEqual(protocol.max_train_samples, 256)

    def test_invalid_values_are_rejected(self):
        args = experiment.parse_args(["--batch-size", "0"])
        with self.assertRaisesRegex(ValueError, "batch_size"):
            experiment.protocol_from_args(args)

    def test_hidden_layer_override(self):
        protocol = experiment.protocol_from_args(
            experiment.parse_args(["--hidden-size", "512", "--hidden-layers", "4"])
        )
        self.assertEqual(protocol.hidden_size, 512)
        self.assertEqual(protocol.hidden_layers, 4)

    def test_diagnostic_batch_size_defaults_off_and_parses(self):
        default = experiment.protocol_from_args(experiment.parse_args([]))
        self.assertEqual(default.diagnostic_batch_size, 0)
        enabled = experiment.protocol_from_args(
            experiment.parse_args(["--diagnostic-batch-size", "128"])
        )
        self.assertEqual(enabled.diagnostic_batch_size, 128)

    def test_diagnostic_batch_size_rejects_too_small(self):
        args = experiment.parse_args(["--diagnostic-batch-size", "1"])
        with self.assertRaisesRegex(ValueError, "diagnostic_batch_size"):
            experiment.protocol_from_args(args)
        negative = experiment.parse_args(["--diagnostic-batch-size", "-4"])
        with self.assertRaisesRegex(ValueError, "diagnostic_batch_size"):
            experiment.protocol_from_args(negative)

    def test_gradient_signal_to_noise_matches_hand_computation(self):
        # A perfectly consistent coordinate (zero variance) and a noisy,
        # zero-mean coordinate: SNR should be very large then ~zero.
        stats = experiment.gradient_signal_to_noise([[2.0, 1.0], [2.0, -1.0]])
        self.assertEqual(stats["coordinates"], 2)
        # Coordinate 0: |mean|=2, std=0 -> 2/eps (huge); coordinate 1: mean=0 -> 0.
        self.assertGreater(stats["mean_snr"], 1e6)
        # A single coordinate with known mean/std gives an exact ratio.
        one = experiment.gradient_signal_to_noise([[3.0], [1.0]])
        self.assertAlmostEqual(one["mean_snr"], 2.0 / 1.0, places=6)

    def test_gradient_signal_to_noise_rejects_degenerate_input(self):
        with self.assertRaisesRegex(ValueError, "two per-example"):
            experiment.gradient_signal_to_noise([[1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "equal length"):
            experiment.gradient_signal_to_noise([[1.0, 2.0], [3.0]])

    def test_sweep_levels_reject_duplicates_and_nonpositive_values(self):
        sweep_path = Path(__file__).with_name("relaxation_sweep.py")
        sweep_spec = importlib.util.spec_from_file_location("relaxation_sweep", sweep_path)
        sweep = importlib.util.module_from_spec(sweep_spec)
        assert sweep_spec.loader is not None
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            sweep_spec.loader.exec_module(sweep)
        finally:
            sys.path.pop(0)
        sweep.validate_levels((1, 5, 10, 20))
        with self.assertRaisesRegex(ValueError, "unique"):
            sweep.validate_levels((1, 1))
        with self.assertRaisesRegex(ValueError, "positive"):
            sweep.validate_levels((0, 1))

    def test_sweep_rejects_nonfinite_values(self):
        sweep_path = Path(__file__).with_name("relaxation_sweep.py")
        sweep_spec = importlib.util.spec_from_file_location("relaxation_sweep_validation", sweep_path)
        sweep = importlib.util.module_from_spec(sweep_spec)
        assert sweep_spec.loader is not None
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            sweep_spec.loader.exec_module(sweep)
        finally:
            sys.path.pop(0)
        sweep.reject_nonfinite({"valid": [0.0, 1.0, None]})
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            sweep.reject_nonfinite({"invalid": float("nan")})

    def test_architecture_sweep_definitions(self):
        architecture_path = Path(__file__).with_name("architecture_sweep.py")
        architecture_spec = importlib.util.spec_from_file_location("architecture_sweep", architecture_path)
        architecture = importlib.util.module_from_spec(architecture_spec)
        assert architecture_spec.loader is not None
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            sys.modules[architecture_spec.name] = architecture
            architecture_spec.loader.exec_module(architecture)
        finally:
            sys.path.pop(0)
        architecture.validate_architectures(architecture.ARCHITECTURES)
        self.assertEqual(
            [item.architecture_id for item in architecture.ARCHITECTURES],
            ["baseline-256x2", "wide-512x2", "deep-256x4"],
        )
        self.assertEqual(architecture.PC_STEPS, 5)
        self.assertEqual(
            [architecture.expected_parameter_count(item) for item in architecture.ARCHITECTURES],
            [269322, 669706, 400906],
        )


if __name__ == "__main__":
    unittest.main()
