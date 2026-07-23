#!/usr/bin/env python3
"""Validate local state-policy artifacts and gate held-out analysis."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_state_policy_study as runner


ANALYSIS_METRICS = (
    "final_average_accuracy",
    "online_predict_before_update_accuracy",
    "mean_adaptation",
    "mean_bwt",
    "mean_forgetting",
)
EXACT_REPEAT_FIELDS = (
    "source_identity",
    "data",
    "initialization",
    "optimizer_audit",
    "counts",
    "online_predict_before_update",
    "accuracy_matrix",
    "metrics",
    "diagnostics",
    "final_linear_parameter_checksum",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {path}") from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_receipt(
    result_path: Path,
    result: dict[str, Any],
    manifest: dict[str, Any],
) -> Path:
    receipt_path = result_path.with_name(f"{result_path.stem}.receipt.json")
    if not receipt_path.exists():
        raise ValueError(f"missing receipt for {result_path}")
    receipt = load_json(receipt_path)
    required = {
        "protocol_id": runner.PROTOCOL_ID,
        "artifact": result_path.name,
        "artifact_sha256": runner.sha256_file(result_path),
        "artifact_bytes": result_path.stat().st_size,
        "canonical_payload_sha256": result["integrity"][
            "canonical_payload_sha256"
        ],
        "manifest_hash": manifest["manifest_hash"],
        "source_identity": result["source_identity"],
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"receipt {key} mismatch for {result_path}")
    return receipt_path


def validate_attempt_pair(
    result_path: Path,
    *,
    manifest: dict[str, Any],
    mode: str,
    condition: str,
    seed: int,
) -> dict[str, Any]:
    result = load_json(result_path)
    runner.validate_result(result, require_provenance=True)
    validate_receipt(result_path, result, manifest)
    if (
        result["mode"] != mode
        or result["condition_id"] != condition
        or int(result["config"]["seed"]) != seed
    ):
        raise ValueError("attempt result does not match the requested cell")
    if result["run_manifest"]["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("attempt result uses a different manifest")
    validate_cell_against_manifest(result, manifest, mode=mode)
    return result


def discover_results(
    output_dir: Path,
    *,
    mode: str,
    manifest: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    pattern = f"{runner.PROTOCOL_ID}__{mode}__*__seed-*.json"
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(output_dir.rglob(pattern)):
        if path.name.endswith((".receipt.json", ".failure.json")):
            continue
        receipt_path = path.with_name(f"{path.stem}.receipt.json")
        if not receipt_path.exists():
            continue
        result = load_json(path)
        runner.validate_result(result, require_provenance=True)
        receipt_path = validate_receipt(path, result, manifest)
        if result.get("mode") != mode:
            raise ValueError(f"mode mismatch in {path}")
        if result["run_manifest"]["manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError(f"manifest mismatch in {path}")
        key = (result["condition_id"], int(result["config"]["seed"]))
        if key in cells:
            raise ValueError(
                f"duplicate valid cell {key}: "
                f"{cells[key]['_validated_path']} and {path}"
            )
        result["_validated_path"] = str(path)
        result["_validated_receipt_path"] = str(receipt_path)
        result["_artifact_sha256"] = runner.sha256_file(path)
        cells[key] = result
    return cells


def expected_cells(
    mode: str, seeds: Sequence[int] | None = None
) -> set[tuple[str, int]]:
    if seeds is None:
        seeds = (
            (runner.BRIDGE_SEED,)
            if mode == "smoke"
            else runner.ALL_FULL_SEEDS
        )
    return {
        (condition, seed)
        for seed in seeds
        for condition in runner.CONDITIONS
    }


def gpu_identity(result: dict[str, Any]) -> dict[str, Any]:
    return runner.host_identity(result["environment"])


def validate_cell_against_manifest(
    result: dict[str, Any],
    manifest: dict[str, Any],
    *,
    mode: str,
) -> None:
    seed = int(result["config"]["seed"])
    if result["source_identity"] != manifest.get("source_identity"):
        raise ValueError(f"source mismatch for {result['condition_id']}, seed {seed}")
    if result["environment"]["dependencies"] != manifest.get(
        "dependency_identity"
    ):
        raise ValueError(
            f"dependency mismatch for {result['condition_id']}, seed {seed}"
        )
    if result["data"]["dataset_file_sha256"] != manifest.get(
        "dataset_file_sha256"
    ):
        raise ValueError(f"dataset mismatch for {result['condition_id']}, seed {seed}")
    if gpu_identity(result) != manifest.get("host"):
        raise ValueError(f"local host/GPU mismatch for seed {seed}")
    if result["run_manifest"].get("local_host") != manifest.get("host"):
        raise ValueError(f"recorded local host mismatch for seed {seed}")
    if result["condition_id"] not in manifest["condition_order_by_seed"].get(
        str(seed), ()
    ):
        raise ValueError(f"condition not in frozen order for seed {seed}")
    if result["config"]["requested_device"] != "cuda":
        raise ValueError("validated study cells must use CUDA")
    if result["environment"]["gpu_identity"] is None:
        raise ValueError("validated study cell lacks GPU identity")
    if mode == "full":
        if result["source"]["repository"]["tracked_dirty"]:
            raise ValueError(f"full cell used dirty repository source for seed {seed}")
        if result["source"]["predictive_coding_submodule"]["tracked_dirty"]:
            raise ValueError(f"full cell used dirty PC source for seed {seed}")


def validate_cross_cell_invariants(
    cells: dict[tuple[str, int], dict[str, Any]],
    *,
    mode: str,
    manifest: dict[str, Any],
) -> None:
    if not cells:
        raise ValueError("no result cells found")
    for result in cells.values():
        validate_cell_against_manifest(result, manifest, mode=mode)

    identities = {result["source_identity"] for result in cells.values()}
    if identities != {manifest["source_identity"]}:
        raise ValueError("source identity differs across cells")
    host_identities = {
        runner.sha256_bytes(runner.canonical_json_bytes(gpu_identity(result)))
        for result in cells.values()
    }
    if len(host_identities) != 1:
        raise ValueError("more than one host/GPU identity appears in the study")

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (_, seed), result in cells.items():
        by_seed[seed].append(result)
    for seed, results in by_seed.items():
        for label, values in (
            (
                "permutation",
                {r["data"]["combined_permutation_checksum"] for r in results},
            ),
            (
                "training stream",
                {r["data"]["combined_train_order_checksum"] for r in results},
            ),
            (
                "initialization",
                {
                    r["initialization"]["paired_initialization_checksum"]
                    for r in results
                },
            ),
            (
                "sample count",
                {
                    runner.sha256_bytes(
                        runner.canonical_json_bytes(r["counts"]["samples_by_task"])
                    )
                    for r in results
                },
            ),
            (
                "update count",
                {
                    runner.sha256_bytes(
                        runner.canonical_json_bytes(
                            r["counts"]["optimizer_updates_by_task"]
                        )
                    )
                    for r in results
                },
            ),
        ):
            if len(values) != 1:
                raise ValueError(f"{label} mismatch within seed {seed}")
        if len(results) == len(runner.CONDITIONS):
            actual_order = [
                result["condition_id"]
                for result in sorted(
                    results,
                    key=lambda value: value["timing"]["started_at_utc"],
                )
            ]
            expected_order = manifest["condition_order_by_seed"][str(seed)]
            if actual_order != expected_order:
                raise ValueError(f"condition execution order mismatch for seed {seed}")


def validate_directory(
    output_dir: Path,
    *,
    mode: str,
    manifest: dict[str, Any],
    seeds: Sequence[int],
    require_complete: bool,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    cells = discover_results(output_dir, mode=mode, manifest=manifest)
    allowed = expected_cells(mode, seeds)
    unexpected = set(cells) - allowed
    if unexpected:
        raise ValueError(f"unexpected cells: {sorted(unexpected)}")
    validate_cross_cell_invariants(cells, mode=mode, manifest=manifest)
    if require_complete and set(cells) != allowed:
        raise ValueError(
            f"cell set mismatch; missing={sorted(allowed - set(cells))}, "
            f"unexpected={sorted(set(cells) - allowed)}"
        )
    failure_files = sorted(
        str(path)
        for path in output_dir.rglob(
            f"{runner.PROTOCOL_ID}__{mode}__*.failure.json"
        )
    )
    result_artifacts = {
        path
        for path in output_dir.rglob(
            f"{runner.PROTOCOL_ID}__{mode}__*__seed-*.json"
        )
        if not path.name.endswith((".receipt.json", ".failure.json"))
    }
    receipt_artifacts = set(
        output_dir.rglob(
            f"{runner.PROTOCOL_ID}__{mode}__*__seed-*.receipt.json"
        )
    )
    unpaired_artifacts = sorted(
        [
            str(path)
            for path in result_artifacts
            if path.with_name(f"{path.stem}.receipt.json") not in receipt_artifacts
        ]
        + [
            str(path)
            for path in receipt_artifacts
            if path.with_name(path.name.removesuffix(".receipt.json") + ".json")
            not in result_artifacts
        ]
    )
    report = {
        "status": "passed",
        "mode": mode,
        "seeds": list(seeds),
        "complete": set(cells) == allowed,
        "validated_cells": len(cells),
        "manifest_hash": manifest["manifest_hash"],
        "source_identity": manifest["source_identity"],
        "preserved_failure_artifacts": failure_files,
        "preserved_unpaired_attempt_artifacts": unpaired_artifacts,
        "cells": [
            {
                "condition_id": condition,
                "seed": seed,
                "gpu_identity": result["environment"]["gpu_identity"],
                "result_path": result["_validated_path"],
                "receipt_path": result["_validated_receipt_path"],
                "artifact_sha256": result["_artifact_sha256"],
                "elapsed_seconds": result["timing"]["elapsed_seconds"],
            }
            for (condition, seed), result in sorted(cells.items())
        ],
    }
    return cells, report


def assert_matching_fields(
    first: dict[tuple[str, int], dict[str, Any]],
    second: dict[tuple[str, int], dict[str, Any]],
    fields: Iterable[str],
    comparison: str,
) -> None:
    if set(first) != set(second):
        raise ValueError(f"{comparison} cell sets differ")
    for key in sorted(first):
        mismatches = [
            field for field in fields if first[key][field] != second[key][field]
        ]
        if mismatches:
            raise ValueError(f"{comparison} mismatch for {key}: {mismatches}")


def smoke_repeat_analysis(
    *,
    pass_1: Path,
    pass_2: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    first, first_report = validate_directory(
        pass_1,
        mode="smoke",
        manifest=manifest,
        seeds=(runner.BRIDGE_SEED,),
        require_complete=True,
    )
    second, second_report = validate_directory(
        pass_2,
        mode="smoke",
        manifest=manifest,
        seeds=(runner.BRIDGE_SEED,),
        require_complete=True,
    )
    assert_matching_fields(
        first,
        second,
        EXACT_REPEAT_FIELDS,
        "single-GPU deterministic smoke repetition",
    )
    report = {
        "status": "passed",
        "protocol_id": runner.PROTOCOL_ID,
        "manifest_hash": manifest["manifest_hash"],
        "validated_smoke_cells": 16,
        "complete_passes": 2,
        "exact_scientific_repetition": True,
        "pass_1": first_report,
        "pass_2": second_report,
    }
    return report, first


def render_smoke_markdown(
    report: dict[str, Any],
    cells: dict[tuple[str, int], dict[str, Any]],
) -> str:
    lines = [
        "# matched-lr-state-policy-v1 smoke and optimizer-state report",
        "",
        f"- Status: `{report['status']}`",
        f"- Validated cells: `{report['validated_smoke_cells']}`",
        f"- Complete passes: `{report['complete_passes']}`",
        f"- Exact scientific repetition: `{report['exact_scientific_repetition']}`",
        f"- Manifest SHA-256: `{report['manifest_hash']}`",
        "",
        "The values below are commissioning diagnostics from open bridge seed 42, "
        "not held-out evidence.",
        "",
        "| Condition | Final steps | State bytes | Adam m L2 | Adam v L2 | "
        "CLASSP sum | CLASSP L2 | CLASSP nonzero | Task displacement L2 sum | "
        "First updates L2 mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in runner.CONDITIONS:
        result = cells[(condition, runner.BRIDGE_SEED)]
        states = result["diagnostics"]["optimizer_state_after_each_task"]
        final = states[-1]
        displacements = result["diagnostics"]["task_parameter_displacement"]
        first_updates = result["diagnostics"]["first_batch_after_boundary"]
        lines.append(
            f"| `{condition}` | {final['step_min']}..{final['step_max']} | "
            f"{final['state_bytes']} | "
            f"{format_optional(final['adam_first_moment_l2'])} | "
            f"{format_optional(final['adam_second_moment_l2'])} | "
            f"{format_optional(final['classp_accumulator_sum'])} | "
            f"{format_optional(final['classp_accumulator_l2'])} | "
            f"{format_optional(final['classp_accumulator_nonzero_fraction'])} | "
            f"{sum(row['l2'] for row in displacements):.6g} | "
            f"{statistics.fmean(row['parameter_update']['l2'] for row in first_updates):.6g} |"
        )
    lines.extend(
        [
            "",
            "Validation establishes exact repeated outputs and recorded optimizer "
            "invariants on this synthetic-size smoke configuration. It does not "
            "establish a scientific effect.",
            "",
        ]
    )
    return "\n".join(lines)


def format_optional(value: float | int | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def bridge_analysis(
    cells: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, Any]:
    expected = expected_cells("full", (runner.BRIDGE_SEED,))
    if set(cells) != expected:
        raise ValueError("bridge analysis requires all eight seed-42 cells")
    records = []
    for condition in runner.CONDITIONS:
        result = cells[(condition, runner.BRIDGE_SEED)]
        optimizer_states = result["diagnostics"]["optimizer_state_after_each_task"]
        boundaries = result["optimizer_audit"]["boundary_summaries"]
        records.append(
            {
                "condition_id": condition,
                "metrics": {
                    metric: result["metrics"][metric]
                    for metric in ANALYSIS_METRICS
                },
                "runtime_seconds": result["timing"]["elapsed_seconds"],
                "peak_cuda_bytes": result["memory"]["peak_cuda_bytes"],
                "task_parameter_displacement": result["diagnostics"][
                    "task_parameter_displacement"
                ],
                "optimizer_state_after_each_task": optimizer_states,
                "first_batch_after_boundary": result["diagnostics"][
                    "first_batch_after_boundary"
                ],
                "boundary_state": [
                    {
                        "before_task_index": boundary["before_task_index"],
                        "action": boundary["action"],
                        "state_before": boundary[
                            "optimizer_diagnostics_before"
                        ],
                        "state_after": boundary["optimizer_diagnostics_after"],
                        "parameters_preserved": boundary[
                            "network_parameters_preserved"
                        ],
                    }
                    for boundary in boundaries
                ],
            }
        )
    bridge_seconds = sum(record["runtime_seconds"] for record in records)
    return {
        "status": "passed",
        "protocol_id": runner.PROTOCOL_ID,
        "bridge_seed": runner.BRIDGE_SEED,
        "excluded_from_confirmation": True,
        "records": records,
        "bridge_total_seconds": bridge_seconds,
        "heldout_runtime_estimate_seconds": bridge_seconds * len(
            runner.PRIMARY_SEEDS
        ),
        "review_policy": (
            "Observation and interpretation require human review. Bridge values "
            "may expose a flaw but may not tune v1."
        ),
    }


def render_bridge_markdown(
    *,
    validation: dict[str, Any],
    analysis: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    lines = [
        "# matched-lr-state-policy-v1 bridge intuition report",
        "",
        f"- Status: `{validation['status']}`",
        f"- Validated bridge cells: `{validation['validated_cells']}` of `8`",
        f"- Seed: `{runner.BRIDGE_SEED}` (open bridge; excluded from confirmation)",
        f"- Manifest SHA-256: `{manifest['manifest_hash']}`",
        f"- Bridge runtime: `{analysis['bridge_total_seconds']:.3f}` seconds",
        "- Held-out runtime estimate from bridge: "
        f"`{analysis['heldout_runtime_estimate_seconds']:.3f}` seconds",
        "",
        "## Observations",
        "",
        "| Condition | Online | Final | Adaptation | BWT | Forgetting ↓ | "
        "Parameter movement ΣL2 | First update mean L2 | Final state bytes | "
        "Runtime (s) | Peak CUDA bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in analysis["records"]:
        metrics = record["metrics"]
        movement = sum(
            row["l2"] for row in record["task_parameter_displacement"]
        )
        first_update = statistics.fmean(
            row["parameter_update"]["l2"]
            for row in record["first_batch_after_boundary"]
        )
        final_state = record["optimizer_state_after_each_task"][-1]
        lines.append(
            f"| `{record['condition_id']}` | "
            f"{metrics['online_predict_before_update_accuracy']:.6g} | "
            f"{metrics['final_average_accuracy']:.6g} | "
            f"{metrics['mean_adaptation']:.6g} | "
            f"{metrics['mean_bwt']:.6g} | "
            f"{metrics['mean_forgetting']:.6g} | {movement:.6g} | "
            f"{first_update:.6g} | {final_state['state_bytes']} | "
            f"{record['runtime_seconds']:.6g} | {record['peak_cuda_bytes']} |"
        )
    lines.extend(
        [
            "",
            "Optimizer trajectories, per-boundary reset transients, Adam moments, "
            "CLASSP accumulators, first-batch gradients/updates, and parameter "
            "preservation checks are retained in the adjacent JSON report.",
            "",
            "## Interpretations",
            "",
            "- Human review pending. Record interpretations here only after "
            "checking the observations against the implementation and protocol.",
            "",
            "## Limitations",
            "",
            "- This is one openly inspected seed and is excluded from confirmation.",
            "- Read-only diagnostics add synchronization, allocation, and timing "
            "overhead; neutrality tests cover scientific outputs and final parameters, "
            "not runtime or peak memory.",
            "- A bridge anomaly can trigger a new protocol version; it cannot justify "
            "silent tuning of v1.",
            "",
            "## Counterclaim",
            "",
            "- Apparent reset effects could reflect optimizer transient scale rather "
            "than a method-specific continual-learning mechanism.",
            "",
            "## Discriminating check",
            "",
            "- Compare boundary-first gradients, update magnitudes, state trajectories, "
            "and task displacements across the matched Adam/CLASSP and BP/PC cells "
            "before authorizing held-out execution.",
            "",
            "Validation establishes artifact integrity, provenance, frozen "
            "configuration, and recorded invariants. It does not establish that a "
            "scientific interpretation is true.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "values": numbers,
        "mean": statistics.fmean(numbers),
        "sample_standard_deviation": (
            statistics.stdev(numbers) if len(numbers) > 1 else None
        ),
        "range": [min(numbers), max(numbers)],
        "positive_sign_count": sum(value > 0 for value in numbers),
        "zero_sign_count": sum(value == 0 for value in numbers),
        "negative_sign_count": sum(value < 0 for value in numbers),
    }


def seed_contrasts(
    cells: dict[tuple[str, int], dict[str, Any]],
    *,
    seed: int,
    metric: str,
) -> dict[str, float | int]:
    values = {
        condition: float(cells[(condition, seed)]["metrics"][metric])
        for condition in runner.CONDITIONS
    }
    deltas = {
        "delta_bp_adam": values["bp-adam-task-reset"]
        - values["bp-adam-persistent"],
        "delta_bp_classp": values["bp-classp-task-reset"]
        - values["bp-classp-persistent"],
        "delta_pc_adam": values["pc-adam-task-reset"]
        - values["pc-adam-persistent"],
        "delta_pc_classp": values["pc-classp-task-reset"]
        - values["pc-classp-persistent"],
    }
    rescue_pc = deltas["delta_pc_classp"] - deltas["delta_pc_adam"]
    rescue_bp = deltas["delta_bp_classp"] - deltas["delta_bp_adam"]
    return {
        "seed": seed,
        **deltas,
        "rescue_pc": rescue_pc,
        "rescue_bp": rescue_bp,
        "omega": rescue_pc - rescue_bp,
    }


def resource_analysis(
    cells: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, Any]:
    records = [
        {
            "condition_id": condition,
            "seed": seed,
            "elapsed_seconds": result["timing"]["elapsed_seconds"],
            "peak_cuda_bytes": result["memory"]["peak_cuda_bytes"],
        }
        for (condition, seed), result in sorted(cells.items())
    ]
    return {
        "host": gpu_identity(next(iter(cells.values()))),
        "cells": records,
        "runtime_seconds": summarize_values(
            [record["elapsed_seconds"] for record in records]
        ),
        "peak_cuda_bytes": summarize_values(
            [record["peak_cuda_bytes"] for record in records]
        ),
        "pooling_policy": "one frozen RTX 4090 identity",
    }


def confirmatory_analysis(
    cells: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, Any]:
    if set(cells) != expected_cells("full", runner.ALL_FULL_SEEDS):
        raise ValueError("confirmatory analysis requires all 48 validated cells")
    analysis: dict[str, Any] = {
        "protocol_id": runner.PROTOCOL_ID,
        "primary_held_out_seeds": list(runner.PRIMARY_SEEDS),
        "bridge_seed_excluded": runner.BRIDGE_SEED,
        "metrics": {},
    }
    for metric in ANALYSIS_METRICS:
        per_seed = [
            seed_contrasts(cells, seed=seed, metric=metric)
            for seed in runner.PRIMARY_SEEDS
        ]
        bridge = seed_contrasts(cells, seed=runner.BRIDGE_SEED, metric=metric)
        analysis["metrics"][metric] = {
            "per_seed": per_seed,
            "bridge_seed_reported_separately": bridge,
            "delta_bp_adam": summarize_values(
                [row["delta_bp_adam"] for row in per_seed]
            ),
            "delta_bp_classp": summarize_values(
                [row["delta_bp_classp"] for row in per_seed]
            ),
            "delta_pc_adam": summarize_values(
                [row["delta_pc_adam"] for row in per_seed]
            ),
            "delta_pc_classp": summarize_values(
                [row["delta_pc_classp"] for row in per_seed]
            ),
            "rescue_pc": summarize_values([row["rescue_pc"] for row in per_seed]),
            "rescue_bp": summarize_values([row["rescue_bp"] for row in per_seed]),
            "omega": summarize_values([row["omega"] for row in per_seed]),
            "direction_note": (
                "lower is better; all differences remain on the raw forgetting scale"
                if metric == "mean_forgetting"
                else "higher is better"
            ),
        }

    final_accuracy = analysis["metrics"]["final_average_accuracy"]
    gate_1_positive = final_accuracy["rescue_pc"]["positive_sign_count"]
    gate_1_passed = gate_1_positive >= 4
    gate_2_positive = final_accuracy["omega"]["positive_sign_count"]
    analysis["hierarchical_final_accuracy_gates"] = {
        "gate_1_classp_specific_reset_rescue_under_pc": {
            "criterion": "positive R_PC in at least 4/5 held-out seeds",
            "positive_seeds": gate_1_positive,
            "passed": gate_1_passed,
        },
        "gate_2_pc_specific_rescue": {
            "criterion": (
                "only if gate 1 passes: positive Omega in at least 4/5 "
                "held-out seeds"
            ),
            "evaluated": gate_1_passed,
            "positive_seeds": gate_2_positive,
            "passed": gate_2_positive >= 4 if gate_1_passed else None,
        },
        "claim_ceiling": (
            "No statistical-significance, mechanistic-synergy, "
            "biological-superiority, or general-validity claim follows from "
            "these gates."
        ),
    }
    analysis["resources"] = resource_analysis(cells)
    return analysis


def render_complete_markdown(
    *,
    validation: dict[str, Any],
    analysis: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    lines = [
        "# matched-lr-state-policy-v1 complete validation and synthesis",
        "",
        f"- Status: `{validation['status']}`",
        f"- Validated cells: `{validation['validated_cells']}` of `48`",
        f"- Manifest SHA-256: `{manifest['manifest_hash']}`",
        f"- Source identity: `{manifest['source_identity']}`",
        f"- Bridge seed excluded from confirmation: `{runner.BRIDGE_SEED}`",
        "",
        "Validation establishes artifact integrity, pairing, provenance, frozen "
        "configuration, and recorded state-policy invariants. It does not by "
        "itself establish that the scientific interpretation is true.",
        "",
        "## Hierarchical final-accuracy gates",
        "",
    ]
    gates = analysis["hierarchical_final_accuracy_gates"]
    gate_1 = gates["gate_1_classp_specific_reset_rescue_under_pc"]
    gate_2 = gates["gate_2_pc_specific_rescue"]
    lines.extend(
        [
            f"- Gate 1: `{gate_1['passed']}` "
            f"({gate_1['positive_seeds']}/5 positive held-out seeds).",
            f"- Gate 2 evaluated: `{gate_2['evaluated']}`; result: "
            f"`{gate_2['passed']}` ({gate_2['positive_seeds']}/5 positive "
            "held-out seeds before hierarchical masking).",
            f"- Claim ceiling: {gates['claim_ceiling']}",
            "",
        ]
    )
    for metric in ANALYSIS_METRICS:
        block = analysis["metrics"][metric]
        lines.extend(
            [
                f"## {metric}",
                "",
                block["direction_note"] + ".",
                "",
                "| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | "
                "R BP | R PC | Ω |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in block["per_seed"]:
            lines.append(
                "| {seed} | {delta_bp_adam:.12g} | {delta_bp_classp:.12g} | "
                "{delta_pc_adam:.12g} | {delta_pc_classp:.12g} | "
                "{rescue_bp:.12g} | {rescue_pc:.12g} | {omega:.12g} |".format(
                    **row
                )
            )
        bridge = block["bridge_seed_reported_separately"]
        lines.extend(
            [
                "",
                "Bridge seed 42 (reported, excluded from summaries):",
                "",
                "| Seed | δ BP Adam | δ BP CLASSP | δ PC Adam | δ PC CLASSP | "
                "R BP | R PC | Ω |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
                "| {seed} | {delta_bp_adam:.12g} | {delta_bp_classp:.12g} | "
                "{delta_pc_adam:.12g} | {delta_pc_classp:.12g} | "
                "{rescue_bp:.12g} | {rescue_pc:.12g} | {omega:.12g} |".format(
                    **bridge
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Runtime and peak memory",
            "",
            "All cells used the one frozen local RTX 4090 identity.",
            "",
            "| Seed | Condition | Runtime (s) | Peak CUDA bytes |",
            "|---:|---|---:|---:|",
        ]
    )
    for record in analysis["resources"]["cells"]:
        lines.append(
            f"| {record['seed']} | `{record['condition_id']}` | "
            f"{record['elapsed_seconds']:.6f} | {record['peak_cuda_bytes']} |"
        )
    lines.append("")
    if validation["preserved_failure_artifacts"]:
        lines.extend(
            [
                "## Preserved failure history",
                "",
                *[
                    f"- `{path}`"
                    for path in validation["preserved_failure_artifacts"]
                ],
                "",
            ]
        )
    if validation["preserved_unpaired_attempt_artifacts"]:
        lines.extend(
            [
                "## Preserved interrupted attempts",
                "",
                "These unpaired artifacts did not count as scientific cells:",
                "",
                *[
                    f"- `{path}`"
                    for path in validation[
                        "preserved_unpaired_attempt_artifacts"
                    ]
                ],
                "",
            ]
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "bridge", "complete"))
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-pass-1", type=Path)
    parser.add_argument("--smoke-pass-2", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-markdown", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = runner.load_run_manifest(args.run_manifest)
        if args.stage == "smoke":
            if args.smoke_pass_1 is None or args.smoke_pass_2 is None:
                raise ValueError("smoke validation requires both pass directories")
            report, cells = smoke_repeat_analysis(
                pass_1=args.smoke_pass_1,
                pass_2=args.smoke_pass_2,
                manifest=manifest,
            )
            markdown = render_smoke_markdown(report, cells)
        else:
            if args.output_dir is None:
                raise ValueError("--output-dir is required")
            seeds = (
                (runner.BRIDGE_SEED,)
                if args.stage == "bridge"
                else runner.ALL_FULL_SEEDS
            )
            cells, report = validate_directory(
                args.output_dir,
                mode="full",
                manifest=manifest,
                seeds=seeds,
                require_complete=True,
            )
            if args.stage == "bridge":
                report["bridge_analysis"] = bridge_analysis(cells)
                markdown = render_bridge_markdown(
                    validation=report,
                    analysis=report["bridge_analysis"],
                    manifest=manifest,
                )
            else:
                report["confirmatory_analysis"] = confirmatory_analysis(cells)
                markdown = render_complete_markdown(
                    validation=report,
                    analysis=report["confirmatory_analysis"],
                    manifest=manifest,
                )
        if args.report_json:
            runner.atomic_write_json_exclusive(args.report_json, report)
        if args.report_markdown:
            atomic_write_text_exclusive(args.report_markdown, markdown)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "stage": args.stage,
                    "validated_cells": report.get(
                        "validated_cells",
                        report.get("validated_smoke_cells"),
                    ),
                    "complete": report.get("complete", True),
                    "manifest_hash": report["manifest_hash"],
                    "report_json": (
                        None
                        if args.report_json is None
                        else str(args.report_json)
                    ),
                    "report_markdown": (
                        None
                        if args.report_markdown is None
                        else str(args.report_markdown)
                    ),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(
            f"VALIDATION FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
