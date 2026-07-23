"""Dependency-free checks for continual-learning metrics and defaults."""

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))
SPEC = importlib.util.spec_from_file_location("continual_experiment", MODULE_DIR / "continual_experiment.py")
continual = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = continual
SPEC.loader.exec_module(continual)


class ContinualProtocolTests(unittest.TestCase):
    def test_frozen_defaults(self):
        protocol = continual.protocol_from_args(continual.parse_args([]))
        self.assertEqual(protocol.scenarios, ("split", "permuted"))
        self.assertEqual(protocol.seeds, (7, 42, 123))
        self.assertEqual(protocol.tasks, 5)
        self.assertEqual(protocol.epochs_per_task, 1)
        self.assertEqual(protocol.batch_size, 500)

    def test_metric_definitions(self):
        matrix = [
            [0.1, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.6, 0.75, 0.1],
            [0.5, 0.65, 0.7],
        ]
        metrics = continual.continual_metrics(matrix, prequential=0.25)
        self.assertAlmostEqual(metrics["final_average_accuracy"], 0.6166666667)
        self.assertAlmostEqual(metrics["backward_transfer"], -0.2)
        self.assertAlmostEqual(metrics["average_forgetting"], 0.2)
        self.assertAlmostEqual(metrics["mean_adaptation_gain"], 0.65)
        self.assertEqual(metrics["prequential_accuracy"], 0.25)

    def test_invalid_task_count_is_rejected(self):
        args = continual.parse_args(["--tasks", "4"])
        with self.assertRaisesRegex(ValueError, "five tasks"):
            continual.protocol_from_args(args)

    def test_distribution_preserves_paired_values(self):
        class FakeNumpy:
            @staticmethod
            def mean(values):
                return sum(values) / len(values)

            @staticmethod
            def std(values, ddof):
                self_mean = sum(values) / len(values)
                return (sum((value - self_mean) ** 2 for value in values) / (len(values) - ddof)) ** 0.5

        summary = continual.distribution([1.0, 2.0, 3.0], FakeNumpy)
        self.assertEqual(summary["values"], [1.0, 2.0, 3.0])
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["sample_std"], 1.0)


if __name__ == "__main__":
    unittest.main()
