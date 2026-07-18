#!/usr/bin/env python3
"""Deterministically summarize a PC relaxation sweep artifact."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable


def distribution(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def summarize_runs(
    runs: list[dict[str, Any]], extractors: dict[str, Callable[[dict[str, Any]], float]],
) -> dict[str, Any]:
    return {
        name: distribution([float(extractor(run)) for run in runs])
        for name, extractor in extractors.items()
    }


def paired_delta(
    treatment: list[dict[str, Any]], control: list[dict[str, Any]],
    extractors: dict[str, Callable[[dict[str, Any]], float]],
) -> dict[str, Any]:
    control_by_seed = {run["seed"]: run for run in control}
    return {
        name: distribution([
            float(extractor(run) - extractor(control_by_seed[run["seed"]]))
            for run in treatment
        ])
        for name, extractor in extractors.items()
    }


STATIC_EXTRACTORS: dict[str, Callable[[dict[str, Any]], float]] = {
    "final_test_accuracy": lambda run: run["test"]["accuracy"],
    "final_train_accuracy": lambda run: run["train_final"]["accuracy"],
    "epoch1_validation_accuracy": lambda run: run["history"][1]["validation"]["accuracy"],
    "epoch1_validation_gain": lambda run: (
        run["history"][1]["validation"]["accuracy"]
        - run["history"][0]["validation"]["accuracy"]
    ),
    "post_update_curve_mean": lambda run: statistics.fmean(
        point["validation"]["accuracy"] for point in run["history"][1:]
    ),
    "update_seconds_per_1000_samples": lambda run: (
        run["training_seconds"] / run["timed_update_samples"] * 1000
    ),
    "peak_accelerator_memory_bytes": lambda run: run["peak_accelerator_memory_bytes"],
}


CONTINUAL_EXTRACTORS: dict[str, Callable[[dict[str, Any]], float]] = {
    "final_average_accuracy": lambda run: run["metrics"]["final_average_accuracy"],
    "prequential_accuracy": lambda run: run["metrics"]["prequential_accuracy"],
    "mean_adaptation_gain": lambda run: run["metrics"]["mean_adaptation_gain"],
    "average_forgetting": lambda run: run["metrics"]["average_forgetting"],
    "backward_transfer": lambda run: run["metrics"]["backward_transfer"],
    "update_seconds_per_1000_samples": lambda run: (
        run["stream_processing_seconds"] / run["timed_stream_samples"] * 1000
    ),
    "peak_accelerator_memory_bytes": lambda run: run["peak_accelerator_memory_bytes"],
}


def scenario_runs(result: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    return [
        item["runs"][0]
        for item in result["scenario_results"]
        if item["scenario"] == scenario
    ]


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    static_control = raw["static"]["bp_control"]["runs"]
    summary: dict[str, Any] = {
        "study_id": raw["study_id"],
        "protocol_id": raw["protocol_id"],
        "source_created_at": raw["created_at"],
        "seeds": raw["seeds"],
        "relaxation_steps": raw["relaxation_steps"],
        "environment": raw["environment"],
        "static": {
            "bp": summarize_runs(static_control, STATIC_EXTRACTORS),
            "pc": {},
        },
        "continual": {},
    }
    for condition in raw["static"]["pc_conditions"]:
        runs = condition["result"]["runs"]
        summary["static"]["pc"][str(condition["relaxation_steps"])] = {
            "metrics": summarize_runs(runs, STATIC_EXTRACTORS),
            "pc_minus_bp": paired_delta(runs, static_control, STATIC_EXTRACTORS),
        }

    continual_control = raw["continual"]["bp_control"]
    for scenario in ("split", "permuted"):
        control_runs = scenario_runs(continual_control, scenario)
        scenario_summary = {
            "bp": summarize_runs(control_runs, CONTINUAL_EXTRACTORS),
            "pc": {},
        }
        for condition in raw["continual"]["pc_conditions"]:
            runs = scenario_runs(condition["result"], scenario)
            scenario_summary["pc"][str(condition["relaxation_steps"])] = {
                "metrics": summarize_runs(runs, CONTINUAL_EXTRACTORS),
                "pc_minus_bp": paired_delta(runs, control_runs, CONTINUAL_EXTRACTORS),
            }
        summary["continual"][scenario] = scenario_summary
    return summary


def fmt(metric: dict[str, Any], percentage: bool = True) -> str:
    scale = 100 if percentage else 1
    mean = metric["mean"] * scale
    sd = metric["sample_std"]
    return f"{mean:.3f}" if sd is None else f"{mean:.3f} ± {sd * scale:.3f}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Relaxation sweep deterministic tables",
        "",
        f"Protocol: `{summary['protocol_id']}`; seeds: `{summary['seeds']}`.",
        "Values are mean ± sample SD. Accuracy-like metrics are percentage points.",
        "Runtime is synchronized timed training-loop/stream-processing seconds per",
        "1,000 timed samples; it includes loading/transfers and, in continual runs,",
        "predict-before-update work. One",
        "diagnostic batch is excluded symmetrically from each static run and each",
        "continual task.",
        "",
        "## Static MNIST",
        "",
        "| Method | Final test | Epoch-1 validation | Epoch-1 gain | Curve mean | s/1k updates |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    bp = summary["static"]["bp"]
    lines.append(
        f"| BP | {fmt(bp['final_test_accuracy'])} | {fmt(bp['epoch1_validation_accuracy'])} | "
        f"{fmt(bp['epoch1_validation_gain'])} | {fmt(bp['post_update_curve_mean'])} | "
        f"{fmt(bp['update_seconds_per_1000_samples'], False)} |"
    )
    for steps, condition in summary["static"]["pc"].items():
        metrics = condition["metrics"]
        lines.append(
            f"| PC T={steps} | {fmt(metrics['final_test_accuracy'])} | "
            f"{fmt(metrics['epoch1_validation_accuracy'])} | "
            f"{fmt(metrics['epoch1_validation_gain'])} | "
            f"{fmt(metrics['post_update_curve_mean'])} | "
            f"{fmt(metrics['update_seconds_per_1000_samples'], False)} |"
        )
    for scenario, title in (("split", "SplitMNIST"), ("permuted", "Permuted MNIST")):
        lines.extend([
            "", f"## {title}", "",
            "| Method | Final average | Prequential | Adaptation gain | Forgetting | BWT | s/1k updates |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        block = summary["continual"][scenario]
        methods = [("BP", block["bp"])] + [
            (f"PC T={steps}", condition["metrics"])
            for steps, condition in block["pc"].items()
        ]
        for label, metrics in methods:
            lines.append(
                f"| {label} | {fmt(metrics['final_average_accuracy'])} | "
                f"{fmt(metrics['prequential_accuracy'])} | "
                f"{fmt(metrics['mean_adaptation_gain'])} | "
                f"{fmt(metrics['average_forgetting'])} | "
                f"{fmt(metrics['backward_transfer'])} | "
                f"{fmt(metrics['update_seconds_per_1000_samples'], False)} |"
            )
    return "\n".join(lines) + "\n"


def write_csv(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for setting, metrics in [("bp", summary["static"]["bp"])] + [
        (f"pc-t{steps}", condition["metrics"])
        for steps, condition in summary["static"]["pc"].items()
    ]:
        for name, value in metrics.items():
            rows.append(("static", setting, name, value["mean"], value["sample_std"]))
    for scenario, block in summary["continual"].items():
        for setting, metrics in [("bp", block["bp"])] + [
            (f"pc-t{steps}", condition["metrics"])
            for steps, condition in block["pc"].items()
        ]:
            for name, value in metrics.items():
                rows.append((scenario, setting, name, value["mean"], value["sample_std"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("scenario", "setting", "metric", "mean", "sample_std"))
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "tables.md").write_text(render_markdown(summary), encoding="utf-8")
    write_csv(summary, args.output_dir / "summary.csv")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
