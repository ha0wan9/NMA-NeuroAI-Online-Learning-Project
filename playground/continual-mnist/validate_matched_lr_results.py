#!/usr/bin/env python3
"""Validate staged matched-LR artifacts and summarize complete full studies."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_matched_lr_validation as runner


SCIENTIFIC_NEUTRALITY_FIELDS = (
    "source",
    "environment",
    "data",
    "initialization",
    "optimizer_audit",
    "counts",
    "online_predict_before_update",
    "accuracy_matrix",
    "metrics",
    "final_linear_parameter_checksum",
)
INTERACTION_METRICS = (
    "final_average_accuracy",
    "online_predict_before_update_accuracy",
    "mean_adaptation",
    "mean_bwt",
    "mean_forgetting",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_receipt(result_path: Path, result: dict[str, Any]) -> None:
    receipt_path = result_path.with_name(f"{result_path.stem}.receipt.json")
    if not receipt_path.exists():
        raise ValueError(f"missing receipt for {result_path.name}")
    receipt = load_json(receipt_path)
    if receipt.get("artifact") != result_path.name:
        raise ValueError(f"receipt target mismatch for {result_path.name}")
    if receipt.get("artifact_sha256") != runner.sha256_file(result_path):
        raise ValueError(f"artifact SHA-256 mismatch for {result_path.name}")
    if receipt.get("artifact_bytes") != result_path.stat().st_size:
        raise ValueError(f"artifact size mismatch for {result_path.name}")
    if receipt.get("protocol_id") != result.get("protocol_id"):
        raise ValueError(f"receipt protocol mismatch for {result_path.name}")


def discover_results(output_dir: Path, mode: str) -> dict[tuple[str, int], dict[str, Any]]:
    pattern = f"{runner.PROTOCOL_ID}__{mode}__*__seed-*.json"
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(output_dir.glob(pattern)):
        if path.name.endswith((".receipt.json", ".failure.json")):
            continue
        result = load_json(path)
        runner.validate_result(result)
        validate_receipt(path, result)
        if result.get("mode") != mode:
            raise ValueError(f"mode mismatch in {path.name}")
        key = (result["condition_id"], int(result["config"]["seed"]))
        if key in cells:
            raise ValueError(f"duplicate cell {key}")
        result["_validated_path"] = str(path)
        cells[key] = result
    return cells


def source_identity(result: dict[str, Any]) -> str:
    source = result["source"]
    value = {
        "repository_commit": source["repository"]["commit"],
        "repository_tracked_dirty": source["repository"]["tracked_dirty"],
        "pc_commit": source["predictive_coding_submodule"]["commit"],
        "pc_tracked_dirty": source["predictive_coding_submodule"]["tracked_dirty"],
        "files": source["files"],
    }
    return runner.sha256_bytes(runner.canonical_json_bytes(value))


def validate_cross_cell_invariants(
    cells: dict[tuple[str, int], dict[str, Any]], mode: str
) -> None:
    if not cells:
        raise ValueError("no result cells found")
    identities = {source_identity(result) for result in cells.values()}
    if len(identities) != 1:
        raise ValueError("source identity differs across cells")

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (_, seed), result in cells.items():
        by_seed[seed].append(result)
        if result["config"]["requested_device"] != "cuda":
            raise ValueError(f"cell {result['condition_id']}, seed {seed} did not use CUDA")
        if result["environment"]["gpu"] is None:
            raise ValueError(f"cell {result['condition_id']}, seed {seed} lacks GPU provenance")

    for seed, results in by_seed.items():
        permutation_ids = {
            result["data"]["combined_permutation_checksum"] for result in results
        }
        stream_ids = {
            result["data"]["combined_train_order_checksum"] for result in results
        }
        initialization_ids = {
            result["initialization"]["paired_initialization_checksum"]
            for result in results
        }
        if len(permutation_ids) != 1:
            raise ValueError(f"permutation mismatch for seed {seed}")
        if len(stream_ids) != 1:
            raise ValueError(f"training stream mismatch for seed {seed}")
        if len(initialization_ids) != 1:
            raise ValueError(f"initialization mismatch for seed {seed}")
        if mode == "full" and any(
            result["source"]["repository"]["tracked_dirty"]
            or result["source"]["predictive_coding_submodule"]["tracked_dirty"]
            for result in results
        ):
            raise ValueError(f"full result for seed {seed} used a dirty source revision")


def expected_cells(mode: str) -> set[tuple[str, int]]:
    seeds = (
        (runner.LEGACY_BRIDGE_SEED,)
        if mode == "smoke"
        else runner.ALL_FULL_SEEDS
    )
    return {(condition, seed) for condition in runner.CONDITIONS for seed in seeds}


def summarize_values(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "range": [min(values), max(values)],
        "positive_sign_count": sum(value > 0 for value in values),
        "zero_sign_count": sum(value == 0 for value in values),
        "negative_sign_count": sum(value < 0 for value in values),
    }


def interaction_analysis(
    cells: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "protocol_id": runner.PROTOCOL_ID,
        "primary_held_out_seeds": list(runner.PRIMARY_SEEDS),
        "legacy_bridge_seed_excluded": runner.LEGACY_BRIDGE_SEED,
        "metrics": {},
    }
    for metric in INTERACTION_METRICS:
        delta_pc: list[float] = []
        delta_bp: list[float] = []
        interaction: list[float] = []
        per_seed: list[dict[str, float | int]] = []
        for seed in runner.PRIMARY_SEEDS:
            pc = (
                cells[("pc-classp", seed)]["metrics"][metric]
                - cells[("pc-adam", seed)]["metrics"][metric]
            )
            bp = (
                cells[("bp-classp", seed)]["metrics"][metric]
                - cells[("bp-adam", seed)]["metrics"][metric]
            )
            difference = pc - bp
            delta_pc.append(pc)
            delta_bp.append(bp)
            interaction.append(difference)
            per_seed.append(
                {"seed": seed, "delta_pc": pc, "delta_bp": bp, "delta_delta": difference}
            )
        analysis["metrics"][metric] = {
            "per_seed": per_seed,
            "delta_pc": summarize_values(delta_pc),
            "delta_bp": summarize_values(delta_bp),
            "delta_delta": summarize_values(interaction),
            "direction_note": (
                "lower is better; differences are reported on the raw metric scale"
                if metric == "mean_forgetting"
                else "higher is better"
            ),
        }
    accuracy_signs = analysis["metrics"]["final_average_accuracy"]["delta_delta"][
        "positive_sign_count"
    ]
    analysis["configuration_specific_differential_benefit_gate"] = {
        "criterion": "positive final-accuracy interaction in at least 4/5 held-out seeds",
        "positive_seeds": accuracy_signs,
        "passed": accuracy_signs >= 4,
        "claim_ceiling": (
            "A pass supports only a configuration-specific differential benefit; "
            "it does not establish mechanistic synergy or biological superiority."
        ),
    }
    return analysis


def compare_neutrality(first_path: Path, second_path: Path) -> dict[str, Any]:
    first = load_json(first_path)
    second = load_json(second_path)
    runner.validate_result(first)
    runner.validate_result(second)
    for key in ("protocol_id", "mode", "condition_id"):
        if first[key] != second[key]:
            raise ValueError(f"neutrality comparison differs on {key}")
    if first["config"]["seed"] != second["config"]["seed"]:
        raise ValueError("neutrality comparison uses different seeds")
    if first["diagnostics"]["enabled"] == second["diagnostics"]["enabled"]:
        raise ValueError("neutrality comparison needs diagnostics on and off")
    mismatches = [
        key for key in SCIENTIFIC_NEUTRALITY_FIELDS if first[key] != second[key]
    ]
    if mismatches:
        raise ValueError(f"diagnostics changed scientific fields: {mismatches}")
    return {
        "status": "passed",
        "condition_id": first["condition_id"],
        "seed": first["config"]["seed"],
        "scientific_fields_compared": list(SCIENTIFIC_NEUTRALITY_FIELDS),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"))
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--compare-neutrality",
        nargs=2,
        type=Path,
        metavar=("DIAGNOSTICS_ON_JSON", "DIAGNOSTICS_OFF_JSON"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.compare_neutrality:
            report = compare_neutrality(*args.compare_neutrality)
        else:
            if args.output_dir is None or args.mode is None:
                raise ValueError("--output-dir and --mode are required for directory validation")
            failure_files = sorted(
                args.output_dir.glob(
                    f"{runner.PROTOCOL_ID}__{args.mode}__*.failure.json"
                )
            )
            if failure_files:
                raise ValueError(
                    "failure artifacts present: "
                    + ", ".join(path.name for path in failure_files)
                )
            cells = discover_results(args.output_dir, args.mode)
            validate_cross_cell_invariants(cells, args.mode)
            if args.require_complete:
                expected = expected_cells(args.mode)
                actual = set(cells)
                if actual != expected:
                    raise ValueError(
                        f"cell set mismatch; missing={sorted(expected - actual)}, "
                        f"unexpected={sorted(actual - expected)}"
                    )
            report = {
                "status": "passed",
                "mode": args.mode,
                "validated_cells": len(cells),
                "source_identity": source_identity(next(iter(cells.values()))),
            }
            if args.mode == "full" and expected_cells("full").issubset(cells):
                report["interaction_analysis"] = interaction_analysis(cells)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"VALIDATION FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
