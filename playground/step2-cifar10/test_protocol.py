"""Dependency-free checks for the CIFAR-10 paper and SplitCIFAR protocols."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = load("step2_common", "common.py")
paper = load("step2_paper", "paper_experiment.py")
split = load("step2_split", "split_experiment.py")
representations = load("step2_representations", "representation_viz.py")


class PaperProtocolTests(unittest.TestCase):
    def test_frozen_defaults_match_archived_figure(self):
        protocol = paper.protocol_from_args(paper.parse_args([]))
        self.assertEqual(protocol.seeds, common.PAPER_SEEDS)
        self.assertEqual(protocol.epochs, 80)
        self.assertEqual(protocol.batch_size, 200)
        self.assertEqual(protocol.pc_steps, 16)
        self.assertEqual(protocol.pc_state_lr, 0.5)
        self.assertEqual(protocol.train_per_class, 5000)
        self.assertEqual(protocol.test_per_class, 1000)

    def test_archived_targets_and_learning_rates(self):
        self.assertAlmostEqual(
            sum(common.PAPER_BEST_ACCURACY["pc"].values()) / 3,
            0.7159864012481404,
        )
        self.assertAlmostEqual(
            sum(common.PAPER_BEST_ACCURACY["bp"].values()) / 3,
            0.6911224561686419,
        )
        protocol = paper.protocol_from_args(paper.parse_args([]))
        self.assertEqual(paper.parameter_lr(protocol, "pc", 698841058), 1e-4)
        self.assertEqual(paper.parameter_lr(protocol, "bp", 2283198659), 5e-5)

    def test_nonpaper_seed_requires_explicit_rates(self):
        with self.assertRaisesRegex(ValueError, "non-paper seeds"):
            paper.protocol_from_args(paper.parse_args(["--seeds", "7"]))
        protocol = paper.protocol_from_args(
            paper.parse_args(["--seeds", "7", "--bp-lr", "0.001", "--pc-lr", "0.001"])
        )
        self.assertEqual(protocol.seeds, (7,))


class SplitProtocolTests(unittest.TestCase):
    def test_frozen_defaults(self):
        protocol = split.protocol_from_args(split.parse_args([]))
        self.assertEqual(protocol.seeds, (7, 42, 123))
        self.assertEqual(protocol.tasks, 5)
        self.assertEqual(protocol.epochs_per_task, 1)
        self.assertEqual(protocol.batch_size, 200)
        self.assertEqual(protocol.bp_lr, 7.5e-5)
        self.assertEqual(protocol.pc_lr, 2.5e-5)

    def test_invalid_task_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "five tasks"):
            split.protocol_from_args(split.parse_args(["--tasks", "4"]))

    def test_representation_diagnostics_are_opt_in(self):
        default = split.protocol_from_args(split.parse_args([]))
        enabled = split.protocol_from_args(split.parse_args(["--representation-diagnostics"]))
        self.assertFalse(default.representation_diagnostics)
        self.assertTrue(enabled.representation_diagnostics)
        self.assertEqual(enabled.anchor_samples_per_class, 20)

    def test_continual_metrics(self):
        matrix = [
            [0.1, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.6, 0.75, 0.1],
            [0.5, 0.65, 0.7],
        ]
        metrics = common.continual_metrics(matrix, prequential=0.25)
        self.assertAlmostEqual(metrics["final_average_accuracy"], 0.6166666667)
        self.assertAlmostEqual(metrics["backward_transfer"], -0.2)
        self.assertAlmostEqual(metrics["average_forgetting"], 0.2)
        self.assertAlmostEqual(metrics["mean_adaptation_gain"], 0.65)

    def test_representation_geometry_metrics(self):
        features = np.asarray([
            [1.0, 0.0, 0.0], [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0], [0.1, 0.9, 0.0],
        ])
        labels = np.asarray([0, 0, 1, 1])
        rdm = representations.correlation_rdm(features, np)
        summary = representations.rdm_summary(rdm, labels, np)
        self.assertTrue(np.allclose(rdm, rdm.T))
        self.assertTrue(np.allclose(np.diag(rdm), 0.0))
        self.assertGreater(summary["class_separation"], 0.0)
        self.assertAlmostEqual(representations.rdm_drift(rdm, rdm, np), 0.0)

    def test_mds_alignment_preserves_pairwise_distances(self):
        rdm = np.asarray([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        coordinates = representations.classical_mds(rdm, np)
        reflected = coordinates * np.asarray([-1.0, 1.0])
        aligned = representations.align_coordinates(coordinates, reflected, np)
        original_distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        aligned_distances = np.linalg.norm(aligned[:, None] - aligned[None, :], axis=2)
        self.assertTrue(np.allclose(original_distances, aligned_distances))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is optional locally")
    def test_3d_representation_trajectory_uses_time_and_aligned_mds(self):
        centroids = {
            layer: np.asarray([[0.0, 0.0], [1.0, 1.0]])
            for layer in representations.LAYER_NAMES
        }
        history = [
            {"step": 0, "centroids": centroids},
            {"step": 1, "centroids": {
                layer: values + 0.25 for layer, values in centroids.items()
            }},
        ]
        figure = representations.plot_representation_trajectory_3d(history, np)
        self.assertEqual(len(figure.axes), len(representations.LAYER_NAMES))
        for axis in figure.axes:
            self.assertEqual(axis.get_xlabel(), "task checkpoint")
            self.assertEqual(axis.get_ylabel(), "aligned MDS 1")
            self.assertEqual(axis.get_zlabel(), "aligned MDS 2")

    @unittest.skipUnless(importlib.util.find_spec("plotly"), "plotly is optional locally")
    def test_interactive_3d_representation_has_four_scenes(self):
        centroids = {
            layer: np.asarray([[0.0, 0.0], [1.0, 1.0]])
            for layer in representations.LAYER_NAMES
        }
        figure = representations.plotly_representation_trajectory_3d(
            [{"step": 0, "centroids": centroids}], np
        )
        self.assertEqual(len(figure.data), 2 * len(representations.LAYER_NAMES))
        self.assertEqual(figure.layout.scene.xaxis.title.text, "task checkpoint")

    @unittest.skipUnless(importlib.util.find_spec("plotly"), "plotly is optional locally")
    def test_interactive_rdm_timeline_has_slider_and_frames(self):
        rdm_a = np.asarray([[0.0, 0.2], [0.2, 0.0]])
        rdm_b = np.asarray([[0.0, 0.7], [0.7, 0.0]])
        history = [
            {
                "step": 0,
                "label": "T0:E0",
                "rdms": {layer: rdm_a for layer in representations.LAYER_NAMES},
            },
            {
                "step": 1,
                "label": "T0:E1",
                "rdms": {layer: rdm_b for layer in representations.LAYER_NAMES},
            },
        ]
        figure = representations.plotly_rdm_timeline(history, np.asarray([0, 1]), np)
        self.assertEqual(len(figure.data), len(representations.LAYER_NAMES))
        self.assertEqual(len(figure.frames), 2)
        self.assertEqual(len(figure.layout.sliders[0].steps), 2)
        self.assertEqual(figure.layout.sliders[0].active, 0)
        self.assertTrue(np.allclose(np.asarray(figure.data[0].z), rdm_a))
        self.assertEqual(figure.layout.sliders[0].steps[1].label, "T0:E1")
        self.assertEqual(figure.layout.coloraxis.cmin, 0.0)
        self.assertEqual(figure.layout.coloraxis.cmax, 2.0)
        self.assertEqual(
            [button.label for button in figure.layout.updatemenus[0].buttons],
            ["Play", "Pause"],
        )
        self.assertEqual(np.asarray(figure.data[0].customdata).shape, (2, 2, 2))
        self.assertIn("dissimilarity", figure.data[0].hovertemplate)


if __name__ == "__main__":
    unittest.main()
