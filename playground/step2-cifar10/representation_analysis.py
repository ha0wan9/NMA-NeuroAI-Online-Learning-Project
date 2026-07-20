#!/usr/bin/env python3
"""Deterministic analysis of full SplitCIFAR representation diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


LAYERS = ("conv1", "conv2", "fc1", "logits")
BEHAVIOR_FIELDS = (
    "test_accuracy_matrix",
    "validation_accuracy_matrix",
    "metrics",
    "initial_parameter_checksum",
    "stream_checksums",
    "samples_seen",
)


def distribution(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0 + 1.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def run_map(payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (seed_result["seed"], run["method"]): run
        for seed_result in payload["seed_results"]
        for run in seed_result["runs"]
    }


def forgetting(run: dict[str, Any], task_id: int) -> float:
    matrix = run["test_accuracy_matrix"]
    return matrix[task_id + 1][task_id] - matrix[-1][task_id]


def analyze(representations: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    representation_runs, control_runs = run_map(representations), run_map(control)
    keys = sorted(representation_runs)
    if keys != sorted(control_runs):
        raise RuntimeError("representation and control run identities differ")
    mismatches = {
        f"{seed}:{method}": [
            field for field in BEHAVIOR_FIELDS
            if representation_runs[(seed, method)].get(field) != control_runs[(seed, method)].get(field)
        ]
        for seed, method in keys
    }
    behavioral_match = all(not fields for fields in mismatches.values())

    final_drift: dict[str, Any] = {}
    drift_by_boundary: dict[str, Any] = {}
    label_alignment: dict[str, Any] = {}
    for layer in LAYERS:
        per_method: dict[str, Any] = {}
        paired_by_seed = []
        for method in ("bp", "pc"):
            seed_values = []
            for seed in (7, 42, 123):
                final = representation_runs[(seed, method)]["representation_diagnostics"][-1]
                seed_values.append(statistics.mean(
                    final["rdm_drift_by_task"][str(task_id)][layer]
                    for task_id in range(4)
                ))
            per_method[method] = distribution(seed_values)
        for position, seed in enumerate((7, 42, 123)):
            paired_by_seed.append(per_method["pc"]["values"][position] - per_method["bp"]["values"][position])
        per_method["pc_minus_bp"] = distribution(paired_by_seed)

        paired_task_drift, paired_task_forgetting = [], []
        raw_correlations = {}
        for method in ("bp", "pc"):
            drift_values, forgetting_values = [], []
            for seed in (7, 42, 123):
                run = representation_runs[(seed, method)]
                final = run["representation_diagnostics"][-1]["rdm_drift_by_task"]
                for task_id in range(4):
                    drift_values.append(final[str(task_id)][layer])
                    forgetting_values.append(forgetting(run, task_id))
            raw_correlations[method] = {
                "n": len(drift_values),
                "pearson": pearson(drift_values, forgetting_values),
                "spearman": pearson(rankdata(drift_values), rankdata(forgetting_values)),
            }
        for seed in (7, 42, 123):
            bp, pc = representation_runs[(seed, "bp")], representation_runs[(seed, "pc")]
            bp_final = bp["representation_diagnostics"][-1]["rdm_drift_by_task"]
            pc_final = pc["representation_diagnostics"][-1]["rdm_drift_by_task"]
            for task_id in range(4):
                paired_task_drift.append(pc_final[str(task_id)][layer] - bp_final[str(task_id)][layer])
                paired_task_forgetting.append(forgetting(pc, task_id) - forgetting(bp, task_id))
        final_drift[layer] = {
            **per_method,
            "raw_drift_forgetting_correlation": raw_correlations,
            "paired_task_delta": {
                "n": len(paired_task_drift),
                "drift": distribution(paired_task_drift),
                "forgetting": distribution(paired_task_forgetting),
                "pearson": pearson(paired_task_drift, paired_task_forgetting),
                "spearman": pearson(rankdata(paired_task_drift), rankdata(paired_task_forgetting)),
                "both_lower_count": sum(
                    drift < 0.0 and forget < 0.0
                    for drift, forget in zip(paired_task_drift, paired_task_forgetting, strict=True)
                ),
            },
        }

        layer_boundaries = {}
        layer_alignment = {}
        for method in ("bp", "pc"):
            boundary_values = {}
            alignment_values = {}
            for boundary in range(6):
                alignment_at_boundary = [
                    representation_runs[(seed, method)]["representation_diagnostics"][boundary]
                    ["layer_summaries"][layer]["label_alignment"]
                    for seed in (7, 42, 123)
                ]
                alignment_values[str(boundary)] = distribution(alignment_at_boundary)
                if boundary >= 2:
                    old_task_values = []
                    for seed in (7, 42, 123):
                        diagnostics = representation_runs[(seed, method)]["representation_diagnostics"][boundary]
                        old_task_values.append(statistics.mean(
                            diagnostics["rdm_drift_by_task"][str(task_id)][layer]
                            for task_id in range(boundary - 1)
                        ))
                    boundary_values[str(boundary)] = distribution(old_task_values)
            layer_boundaries[method] = boundary_values
            layer_alignment[method] = alignment_values
        drift_by_boundary[layer] = layer_boundaries
        label_alignment[layer] = layer_alignment

    behavioral_metrics = {}
    for method in ("bp", "pc"):
        behavioral_metrics[method] = {
            metric: distribution([
                representation_runs[(seed, method)]["metrics"][metric]
                for seed in (7, 42, 123)
            ])
            for metric in (
                "final_average_accuracy",
                "prequential_accuracy",
                "average_forgetting",
                "mean_adaptation_gain",
            )
        }

    return {
        "analysis_id": "split-cifar10-representation-analysis-v1",
        "representation_protocol_id": representations["protocol_id"],
        "control_protocol_id": control["protocol_id"],
        "behavioral_fields": list(BEHAVIOR_FIELDS),
        "behavioral_match": behavioral_match,
        "behavioral_mismatches": mismatches,
        "seeds": [7, 42, 123],
        "old_tasks_at_final_boundary": [0, 1, 2, 3],
        "behavioral_metrics": behavioral_metrics,
        "final_old_task_rdm_drift": final_drift,
        "old_task_rdm_drift_by_boundary": drift_by_boundary,
        "label_alignment_by_boundary": label_alignment,
        "interpretation_guardrail": (
            "RDM drift correlations are descriptive and task-confounded; "
            "plasticity metrics must accompany stability claims."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(
        json.loads(args.representations.read_text(encoding="utf-8")),
        json.loads(args.control.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
