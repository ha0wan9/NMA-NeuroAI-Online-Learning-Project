#!/usr/bin/env python3
"""Deterministically summarize a matched BP–PC architecture sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from relaxation_analysis import (
    CONTINUAL_EXTRACTORS,
    STATIC_EXTRACTORS,
    fmt,
    paired_delta,
    summarize_runs,
)


def method_runs(runs: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [run for run in runs if run["method"] == method]


def continual_method_runs(result: dict[str, Any], scenario: str, method: str) -> list[dict[str, Any]]:
    return [
        run
        for item in result["scenario_results"]
        if item["scenario"] == scenario
        for run in item["runs"]
        if run["method"] == method
    ]


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    baseline_condition = next(
        condition for condition in raw["conditions"]
        if condition["architecture"]["architecture_id"] == "baseline-256x2"
    )
    baseline_static = {
        method: method_runs(baseline_condition["static"]["runs"], method)
        for method in ("bp", "pc")
    }
    summary: dict[str, Any] = {
        "study_id": raw["study_id"],
        "protocol_id": raw["protocol_id"],
        "source_created_at": raw["created_at"],
        "seeds": raw["seeds"],
        "pc_steps": raw["pc_steps"],
        "environment": raw["environment"],
        "conditions": {},
    }
    for condition in raw["conditions"]:
        architecture = condition["architecture"]
        identifier = architecture["architecture_id"]
        static_bp = method_runs(condition["static"]["runs"], "bp")
        static_pc = method_runs(condition["static"]["runs"], "pc")
        block: dict[str, Any] = {
            "architecture": architecture,
            "learned_parameter_count": static_bp[0]["learned_parameter_count"],
            "static": {
                "bp": summarize_runs(static_bp, STATIC_EXTRACTORS),
                "pc": summarize_runs(static_pc, STATIC_EXTRACTORS),
                "pc_minus_bp": paired_delta(static_pc, static_bp, STATIC_EXTRACTORS),
                "bp_minus_baseline": paired_delta(static_bp, baseline_static["bp"], STATIC_EXTRACTORS),
                "pc_minus_baseline": paired_delta(static_pc, baseline_static["pc"], STATIC_EXTRACTORS),
            },
            "continual": {},
        }
        for scenario in ("split", "permuted"):
            bp_runs = continual_method_runs(condition["continual"], scenario, "bp")
            pc_runs = continual_method_runs(condition["continual"], scenario, "pc")
            baseline_bp = continual_method_runs(baseline_condition["continual"], scenario, "bp")
            baseline_pc = continual_method_runs(baseline_condition["continual"], scenario, "pc")
            block["continual"][scenario] = {
                "bp": summarize_runs(bp_runs, CONTINUAL_EXTRACTORS),
                "pc": summarize_runs(pc_runs, CONTINUAL_EXTRACTORS),
                "pc_minus_bp": paired_delta(pc_runs, bp_runs, CONTINUAL_EXTRACTORS),
                "bp_minus_baseline": paired_delta(bp_runs, baseline_bp, CONTINUAL_EXTRACTORS),
                "pc_minus_baseline": paired_delta(pc_runs, baseline_pc, CONTINUAL_EXTRACTORS),
            }
        summary["conditions"][identifier] = block
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Architecture sweep deterministic tables", "",
        f"Protocol: `{summary['protocol_id']}`; PC `T={summary['pc_steps']}`; seeds `{summary['seeds']}`.",
        "Values are mean ± sample SD. Accuracy-like metrics are percentage points.", "",
        "## Static MNIST", "",
        "| Architecture | Parameters | BP final | PC final | PC−BP | BP epoch-1 gain | PC epoch-1 gain | PC/BP time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for identifier, condition in summary["conditions"].items():
        bp, pc, delta = condition["static"]["bp"], condition["static"]["pc"], condition["static"]["pc_minus_bp"]
        ratio = pc["update_seconds_per_1000_samples"]["mean"] / bp["update_seconds_per_1000_samples"]["mean"]
        lines.append(
            f"| {identifier} | {condition['learned_parameter_count']:,} | "
            f"{fmt(bp['final_test_accuracy'])} | {fmt(pc['final_test_accuracy'])} | "
            f"{fmt(delta['final_test_accuracy'])} | {fmt(bp['epoch1_validation_gain'])} | "
            f"{fmt(pc['epoch1_validation_gain'])} | {ratio:.3f}× |"
        )
    for scenario, title in (("split", "SplitMNIST"), ("permuted", "Permuted MNIST")):
        lines.extend([
            "", f"## {title}", "",
            "| Architecture | Method | Final | Prequential | Adaptation | Forgetting | s/1k timed samples |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for identifier, condition in summary["conditions"].items():
            block = condition["continual"][scenario]
            for method in ("bp", "pc"):
                metrics = block[method]
                lines.append(
                    f"| {identifier} | {method.upper()} | {fmt(metrics['final_average_accuracy'])} | "
                    f"{fmt(metrics['prequential_accuracy'])} | {fmt(metrics['mean_adaptation_gain'])} | "
                    f"{fmt(metrics['average_forgetting'])} | "
                    f"{fmt(metrics['update_seconds_per_1000_samples'], False)} |"
                )
    lines.extend([
        "", "## Architecture-minus-baseline effects", "",
        "Paired differences compare each larger architecture with `baseline-256x2`",
        "for the same method and seed.", "",
        "| Scenario | Architecture | Method | Final Δ (pp) | Plasticity Δ (pp) | Time ratio | Memory Δ (MiB) |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    baseline = summary["conditions"]["baseline-256x2"]
    for identifier, condition in summary["conditions"].items():
        if identifier == "baseline-256x2":
            continue
        for scenario in ("static", "split", "permuted"):
            block = condition["static"] if scenario == "static" else condition["continual"][scenario]
            baseline_block = baseline["static"] if scenario == "static" else baseline["continual"][scenario]
            final_metric = "final_test_accuracy" if scenario == "static" else "final_average_accuracy"
            plasticity_metric = "epoch1_validation_gain" if scenario == "static" else "mean_adaptation_gain"
            for method in ("bp", "pc"):
                delta = block[f"{method}_minus_baseline"]
                time_ratio = (
                    block[method]["update_seconds_per_1000_samples"]["mean"]
                    / baseline_block[method]["update_seconds_per_1000_samples"]["mean"]
                )
                memory_delta_mib = delta["peak_accelerator_memory_bytes"]["mean"] / (1024 ** 2)
                lines.append(
                    f"| {scenario} | {identifier} | {method.upper()} | "
                    f"{fmt(delta[final_metric])} | {fmt(delta[plasticity_metric])} | "
                    f"{time_ratio:.3f}× | {memory_delta_mib:.3f} |"
                )
    return "\n".join(lines) + "\n"


def write_csv(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for identifier, condition in summary["conditions"].items():
        for scenario in ("static", "split", "permuted"):
            block = condition["static"] if scenario == "static" else condition["continual"][scenario]
            for method in ("bp", "pc"):
                for metric, value in block[method].items():
                    rows.append((identifier, scenario, method, metric, value["mean"], value["sample_std"]))
                for metric, value in block[f"{method}_minus_baseline"].items():
                    rows.append((
                        identifier, scenario, f"{method}_minus_baseline", metric,
                        value["mean"], value["sample_std"],
                    ))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("architecture", "scenario", "method", "metric", "mean", "sample_std"))
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "tables.md").write_text(render_markdown(summary), encoding="utf-8")
    write_csv(summary, args.output_dir / "summary.csv")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
