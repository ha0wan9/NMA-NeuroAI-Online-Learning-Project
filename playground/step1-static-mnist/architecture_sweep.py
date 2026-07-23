#!/usr/bin/env python3
"""Run matched BP–PC architecture-scale experiments on static/continual MNIST."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import continual_experiment
import experiment
from relaxation_sweep import reject_nonfinite, task_spec_map


@dataclass(frozen=True)
class Architecture:
    architecture_id: str
    hidden_size: int
    hidden_layers: int

    def validate(self) -> None:
        if not self.architecture_id:
            raise ValueError("architecture_id is required")
        if self.hidden_size <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden_size and hidden_layers must be positive")


ARCHITECTURES = (
    Architecture("baseline-256x2", 256, 2),
    Architecture("wide-512x2", 512, 2),
    Architecture("deep-256x4", 256, 4),
)
SEEDS = (7, 42, 123)
PC_STEPS = 5


def expected_parameter_count(architecture: Architecture) -> int:
    width, layers = architecture.hidden_size, architecture.hidden_layers
    return (784 + 1) * width + (layers - 1) * (width + 1) * width + (width + 1) * 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--pc-steps", type=int, default=PC_STEPS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def validate_architectures(architectures: tuple[Architecture, ...]) -> None:
    if not architectures:
        raise ValueError("at least one architecture is required")
    for architecture in architectures:
        architecture.validate()
    identifiers = [architecture.architecture_id for architecture in architectures]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("architecture IDs must be unique")


def static_protocol(
    architecture: Architecture, seeds: tuple[int, ...], pc_steps: int,
    device: str, data_root: str, smoke: bool,
) -> experiment.Protocol:
    return experiment.Protocol(
        seeds=seeds, methods=("bp", "pc"), pc_steps=pc_steps,
        hidden_size=architecture.hidden_size, hidden_layers=architecture.hidden_layers,
        device=device, data_root=data_root, epochs=1 if smoke else 10,
        batch_size=64 if smoke else 500,
        max_train_samples=256 if smoke else None,
        max_valid_samples=256 if smoke else None,
        max_test_samples=256 if smoke else None,
    )


def continual_protocol(
    architecture: Architecture, seeds: tuple[int, ...], pc_steps: int,
    device: str, data_root: str, smoke: bool,
) -> continual_experiment.ContinualProtocol:
    return continual_experiment.ContinualProtocol(
        seeds=seeds, methods=("bp", "pc"), pc_steps=pc_steps,
        hidden_size=architecture.hidden_size, hidden_layers=architecture.hidden_layers,
        device=device, data_root=data_root, batch_size=64 if smoke else 500,
        max_train_samples_per_task=256 if smoke else None,
        max_eval_samples_per_task=256 if smoke else None,
    )


def validate_condition(condition: dict[str, Any]) -> None:
    architecture = Architecture(**condition["architecture"])
    expected = expected_parameter_count(architecture)
    static_by_seed: dict[int, dict[str, Any]] = {}
    for run in condition["static"]["runs"]:
        static_by_seed.setdefault(run["seed"], {})[run["method"]] = run
        if run["learned_parameter_count"] != expected:
            raise RuntimeError(f"unexpected parameter count for {architecture.architecture_id}")
        if run["method"] == "pc" and run["pc_first_batch_dynamics"]["overall_decreased"] is not True:
            raise RuntimeError(f"static PC objective did not decrease for {architecture.architecture_id}")
    for seed, paired in static_by_seed.items():
        if paired["bp"]["learned_parameter_count"] != paired["pc"]["learned_parameter_count"]:
            raise RuntimeError(f"static BP/PC parameter mismatch at seed {seed}")
    for scenario in condition["continual"]["scenario_results"]:
        by_method = {run["method"]: run for run in scenario["runs"]}
        if by_method["bp"]["learned_parameter_count"] != by_method["pc"]["learned_parameter_count"]:
            raise RuntimeError("paired continual methods have different learned parameter counts")
        if by_method["bp"]["learned_parameter_count"] != expected:
            raise RuntimeError(f"unexpected continual parameter count for {architecture.architecture_id}")
        diagnostics = by_method["pc"]["pc_first_batch_dynamics_by_task"]
        if not all(item["overall_decreased"] is True for item in diagnostics):
            raise RuntimeError(f"continual PC objective did not decrease for {architecture.architecture_id}")


def validate_cross_architecture(conditions: list[dict[str, Any]]) -> None:
    static_split = conditions[0]["static"]["dataset_index_checksums"]
    continual_tasks = task_spec_map(conditions[0]["continual"])
    for condition in conditions:
        validate_condition(condition)
        if condition["static"]["dataset_index_checksums"] != static_split:
            raise RuntimeError("static data split differs across architectures")
        if task_spec_map(condition["continual"]) != continual_tasks:
            raise RuntimeError("continual task streams differ across architectures")


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    architectures, seeds = ARCHITECTURES, tuple(args.seeds)
    validate_architectures(architectures)
    if args.pc_steps != PC_STEPS:
        raise ValueError(f"protocol v2 freezes pc_steps to {PC_STEPS}")
    static_runtime = experiment.load_runtime()
    continual_runtime = experiment.load_runtime()
    conditions = []
    for architecture in architectures:
        static_config = static_protocol(
            architecture, seeds, args.pc_steps, args.device, args.data_root, args.smoke
        )
        continual_config = continual_protocol(
            architecture, seeds, args.pc_steps, args.device, args.data_root, args.smoke
        )
        conditions.append({
            "architecture": asdict(architecture),
            "static_protocol": asdict(static_config),
            "continual_protocol": asdict(continual_config),
            "static": experiment.run(static_config, static_runtime),
            "continual": continual_experiment.run(continual_config, continual_runtime),
        })
    validate_cross_architecture(conditions)
    result = {
        "study_id": "pc-network-scale",
        "protocol_id": "pc-network-scale-v2",
        "claim_status": "verified run output; interpretation requires paired multi-seed analysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "pc_steps": args.pc_steps,
        "smoke": args.smoke,
        "architectures": [asdict(architecture) for architecture in architectures],
        "environment": {
            "python": platform.python_version(),
            "device": conditions[0]["static"]["environment"]["device"],
            "execution_host": conditions[0]["static"]["environment"]["execution_host"],
            "runtime_image": conditions[0]["static"]["environment"]["runtime_image"],
            "runtime_image_id": conditions[0]["static"]["environment"]["runtime_image_id"],
        },
        "conditions": conditions,
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
