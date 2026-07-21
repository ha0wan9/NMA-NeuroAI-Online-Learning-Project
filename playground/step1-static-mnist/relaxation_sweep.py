#!/usr/bin/env python3
"""Run the matched PC relaxation-step ablation in static and continual MNIST."""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import continual_experiment
import experiment


RELAXATION_STEPS = (1, 5, 10, 20)
SEEDS = (7, 42, 123)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relaxation-steps", type=int, nargs="+", default=list(RELAXATION_STEPS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def validate_levels(levels: tuple[int, ...]) -> None:
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("relaxation steps must be positive")
    if len(set(levels)) != len(levels):
        raise ValueError("relaxation steps must be unique")


def static_protocol(
    methods: tuple[str, ...], seeds: tuple[int, ...], steps: int,
    device: str, data_root: str, smoke: bool,
) -> experiment.Protocol:
    return experiment.Protocol(
        methods=methods, seeds=seeds, pc_steps=steps, device=device, data_root=data_root,
        epochs=1 if smoke else 10,
        batch_size=64 if smoke else 500,
        max_train_samples=256 if smoke else None,
        max_valid_samples=256 if smoke else None,
        max_test_samples=256 if smoke else None,
    )


def continual_protocol(
    methods: tuple[str, ...], seeds: tuple[int, ...], steps: int,
    device: str, data_root: str, smoke: bool,
) -> continual_experiment.ContinualProtocol:
    return continual_experiment.ContinualProtocol(
        methods=methods, seeds=seeds, pc_steps=steps, device=device, data_root=data_root,
        batch_size=64 if smoke else 500,
        max_train_samples_per_task=256 if smoke else None,
        max_eval_samples_per_task=256 if smoke else None,
    )


def checksum_map_static(result: dict[str, Any]) -> dict[int, str]:
    return {run["seed"]: run["initial_parameter_checksum"] for run in result["runs"]}


def checksum_map_continual(result: dict[str, Any]) -> dict[tuple[str, int], str]:
    return {
        (scenario["scenario"], scenario["seed"]): scenario["runs"][0]["initial_parameter_checksum"]
        for scenario in result["scenario_results"]
    }


def task_spec_map(result: dict[str, Any]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    return {
        (scenario["scenario"], scenario["seed"]): scenario["task_specs"]
        for scenario in result["scenario_results"]
    }


def reject_nonfinite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite result at {path}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_nonfinite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_nonfinite(child, f"{path}[{index}]")


def verify_matched_initialization(
    static_bp: dict[str, Any], static_pc: list[dict[str, Any]],
    continual_bp: dict[str, Any], continual_pc: list[dict[str, Any]],
) -> None:
    static_control = checksum_map_static(static_bp)
    continual_control = checksum_map_continual(continual_bp)
    task_control = task_spec_map(continual_bp)
    for condition in static_pc:
        if checksum_map_static(condition["result"]) != static_control:
            raise RuntimeError("static sweep initialization does not match BP control")
    for condition in continual_pc:
        if checksum_map_continual(condition["result"]) != continual_control:
            raise RuntimeError("continual sweep initialization does not match BP control")
        if task_spec_map(condition["result"]) != task_control:
            raise RuntimeError("continual sweep task streams do not match BP control")


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    levels, seeds = tuple(args.relaxation_steps), tuple(args.seeds)
    validate_levels(levels)
    static_runtime = experiment.load_runtime()
    continual_runtime = experiment.load_runtime()

    static_bp_protocol = static_protocol(("bp",), seeds, levels[-1], args.device, args.data_root, args.smoke)
    continual_bp_protocol = continual_protocol(("bp",), seeds, levels[-1], args.device, args.data_root, args.smoke)
    static_bp = experiment.run(static_bp_protocol, static_runtime)
    continual_bp = continual_experiment.run(continual_bp_protocol, continual_runtime)

    static_pc, continual_pc = [], []
    for steps in levels:
        static_config = static_protocol(("pc",), seeds, steps, args.device, args.data_root, args.smoke)
        continual_config = continual_protocol(("pc",), seeds, steps, args.device, args.data_root, args.smoke)
        static_pc.append({
            "relaxation_steps": steps,
            "protocol": asdict(static_config),
            "result": experiment.run(static_config, static_runtime),
        })
        continual_pc.append({
            "relaxation_steps": steps,
            "protocol": asdict(continual_config),
            "result": continual_experiment.run(continual_config, continual_runtime),
        })

    verify_matched_initialization(static_bp, static_pc, continual_bp, continual_pc)
    result = {
        "study_id": "pc-relaxation-plasticity",
        "protocol_id": "pc-relaxation-sweep-v1",
        "claim_status": "verified run output; interpretation requires paired multi-seed analysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "relaxation_steps": list(levels),
        "seeds": list(seeds),
        "smoke": args.smoke,
        "environment": {
            "python": platform.python_version(),
            "device": static_bp["environment"]["device"],
            "execution_host": static_bp["environment"]["execution_host"],
            "runtime_image": static_bp["environment"]["runtime_image"],
            "runtime_image_id": static_bp["environment"]["runtime_image_id"],
        },
        "static": {"bp_control": static_bp, "pc_conditions": static_pc},
        "continual": {"bp_control": continual_bp, "pc_conditions": continual_pc},
    }
    reject_nonfinite(result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing run artifact: {args.output}")
    result = run_sweep(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
