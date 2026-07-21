#!/usr/bin/env python3
"""Backfill TensorBoard event files from immutable CIFAR experiment JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from representation_viz import (
    log_comparison_dashboard,
    plot_accuracy_matrix,
    plot_representation_trajectory_3d,
    write_plotly_trajectory,
)


def writer_for(root: Path, label: str, method: str, seed: int, config: dict[str, Any]) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("tensorboard is required for export") from exc
    writer = SummaryWriter(log_dir=str(root / label / f"{method}-seed-{seed}"))
    writer.add_text("run/config", json.dumps(config, sort_keys=True, indent=2), 0)
    return writer


def export_static(payload: dict[str, Any], root: Path, label: str) -> None:
    for run in payload["runs"]:
        writer = writer_for(root, label, run["method"], run["seed"], payload["protocol"])
        for row in run["history"]:
            step = row["epoch"]
            writer.add_scalar("test/accuracy", row["test"]["accuracy"], step)
            writer.add_scalar("test/loss_per_example", row["test"]["loss_per_example"], step)
            writer.add_scalar("train/samples_seen", row["samples_seen"], step)
        writer.add_scalar("summary/best_test_accuracy", run["best_test_accuracy"], 0)
        dynamics = run.get("pc_first_batch_dynamics")
        if dynamics:
            for name in ("loss", "energy", "overall"):
                for step, value in enumerate(dynamics[name]):
                    writer.add_scalar(f"pc_inference/{name}", value, step)
        writer.close()


def export_split(payload: dict[str, Any], root: Path, label: str) -> None:
    import numpy as np

    for seed_result in payload["seed_results"]:
        for run in seed_result["runs"]:
            writer = writer_for(root, label, run["method"], run["seed"], payload["protocol"])
            for boundary, row in enumerate(run["test_accuracy_matrix"]):
                for task_id, value in enumerate(row):
                    writer.add_scalar(f"test/task_{task_id}_accuracy", value, boundary)
            for boundary, row in enumerate(run["validation_accuracy_matrix"]):
                for task_id, value in enumerate(row):
                    writer.add_scalar(f"validation/task_{task_id}_accuracy", value, boundary)
            for task_id, value in enumerate(run["metrics"]["prequential_accuracy_by_task"], 1):
                writer.add_scalar("stream/prequential_accuracy_task", value, task_id)
            for name in (
                "final_average_accuracy", "prequential_accuracy", "backward_transfer",
                "average_forgetting", "mean_adaptation_gain",
            ):
                writer.add_scalar(f"summary/{name}", run["metrics"][name], 5)
            for dynamics in run.get("pc_first_batch_dynamics_by_task") or []:
                for name in ("loss", "energy", "overall"):
                    for step, value in enumerate(dynamics[name]):
                        writer.add_scalar(
                            f"pc_inference/task_{dynamics['task_id']}/{name}", value, step
                        )
            trajectory_history = []
            for boundary in run.get("representation_diagnostics") or []:
                step = boundary["boundary"]
                for layer, summary in boundary["layer_summaries"].items():
                    for metric, value in summary.items():
                        writer.add_scalar(f"representations/{layer}/{metric}", value, step)
                for task_id, layers in boundary["rdm_drift_by_task"].items():
                    for layer, value in layers.items():
                        writer.add_scalar(
                            f"representations/{layer}/task_{task_id}_rdm_drift", value, step
                        )
                if boundary.get("mds_class_centroids"):
                    trajectory_history.append({
                        "step": step,
                        "centroids": boundary["mds_class_centroids"],
                    })
                    writer.add_figure(
                        "representations/trajectory_3d",
                        plot_representation_trajectory_3d(trajectory_history, np),
                        step, close=True,
                    )
            writer.add_figure(
                "continual/test_accuracy_matrix",
                plot_accuracy_matrix(
                    run["test_accuracy_matrix"],
                    f"{run['method'].upper()} seed {run['seed']}: test accuracy",
                ),
                5, close=True,
            )
            writer.add_figure(
                "continual/validation_accuracy_matrix",
                plot_accuracy_matrix(
                    run["validation_accuracy_matrix"],
                    f"{run['method'].upper()} seed {run['seed']}: validation accuracy",
                ),
                5, close=True,
            )
            if trajectory_history:
                interactive_path = write_plotly_trajectory(
                    trajectory_history,
                    np,
                    Path(writer.log_dir) / "artifacts" / "representation_trajectory_3d.html",
                )
                writer.add_text(
                    "representations/trajectory_3d_interactive",
                    f"Interactive Plotly artifact: `{interactive_path}`",
                    trajectory_history[-1]["step"],
                )
            writer.close()
    log_comparison_dashboard(payload, root, label)


def export_artifact(artifact: Path, root: Path) -> None:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    label = artifact.stem
    protocol_id = payload.get("protocol_id", "")
    if protocol_id.startswith("song-2024-figure4i"):
        export_static(payload, root, label)
    elif protocol_id.startswith("split-cifar10"):
        export_split(payload, root, label)
    else:
        raise ValueError(f"unsupported protocol_id: {protocol_id!r}")
    print(root / label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument(
        "--tensorboard-dir", type=Path,
        default=Path("playground/step2-cifar10/runs/tensorboard"),
    )
    args = parser.parse_args()
    for artifact in args.artifacts:
        export_artifact(artifact, args.tensorboard_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
