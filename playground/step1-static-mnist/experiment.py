#!/usr/bin/env python3
"""Run the Step 1 static-MNIST PC/BP feasibility comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SUBMODULE = ROOT / "playground" / "predictive-coding"
if str(SUBMODULE) not in sys.path:
    sys.path.insert(0, str(SUBMODULE))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    import predictive_coding as pc
except ImportError as exc:  # pragma: no cover - exercised by setup failures
    raise SystemExit(
        "Predictive-coding submodule unavailable; run "
        "`git submodule update --init --recursive` from the repository root."
    ) from exc


PROTOCOL_ID = "step1-static-mnist-v0.1"
INTERPRETATION = "static-feasibility-only"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    train_size: int
    test_size: int
    epochs: int
    hidden_size: int
    batch_size: int
    inference_steps: int
    parameter_lr: float
    latent_lr: float


CONFIGS = {
    "smoke": ExperimentConfig(
        name="smoke",
        train_size=1_000,
        test_size=500,
        epochs=1,
        hidden_size=64,
        batch_size=100,
        inference_steps=5,
        parameter_lr=0.001,
        latent_lr=0.01,
    )
}


def seed_everything(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }


def stratified_indices(targets: torch.Tensor, count: int, seed: int) -> list[int]:
    """Select a deterministic, class-balanced subset and realized order."""
    generator = torch.Generator().manual_seed(seed)
    labels = sorted(int(label) for label in torch.unique(targets))
    per_class, remainder = divmod(count, len(labels))
    selected: list[int] = []
    for offset, label in enumerate(labels):
        candidates = torch.where(targets == label)[0]
        take = per_class + (offset < remainder)
        if take > len(candidates):
            raise ValueError(f"requested {take} examples for class {label}, found {len(candidates)}")
        order = torch.randperm(len(candidates), generator=generator)
        selected.extend(candidates[order[:take]].tolist())
    realized = torch.randperm(len(selected), generator=generator).tolist()
    return [selected[index] for index in realized]


def sequence_hash(values: list[int]) -> str:
    payload = ",".join(map(str, values)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_data(config: ExperimentConfig, seed: int) -> tuple[DataLoader, DataLoader, dict[str, str]]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])
    train = datasets.MNIST(ROOT / "data", train=True, download=True, transform=transform)
    test = datasets.MNIST(ROOT / "data", train=False, download=True, transform=transform)
    train_indices = stratified_indices(train.targets, config.train_size, seed)
    test_indices = stratified_indices(test.targets, config.test_size, seed + 10_000)
    train_loader = DataLoader(
        Subset(train, train_indices), batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(Subset(test, test_indices), batch_size=500, shuffle=False, num_workers=0)
    identity = {
        "dataset": "MNIST",
        "train_indices_sha256": sequence_hash(train_indices),
        "test_indices_sha256": sequence_hash(test_indices),
        "realized_train_order_sha256": sequence_hash(train_indices * config.epochs),
    }
    return train_loader, test_loader, identity


def build_model(config: ExperimentConfig, method: str, seed: int) -> nn.Sequential:
    seed_everything(seed)
    layers: list[nn.Module] = [nn.Linear(784, config.hidden_size)]
    if method == "pc":
        layers.append(pc.PCLayer())
    layers.extend([nn.ReLU(), nn.Linear(config.hidden_size, config.hidden_size)])
    if method == "pc":
        layers.append(pc.PCLayer())
    layers.extend([nn.ReLU(), nn.Linear(config.hidden_size, 10)])
    return nn.Sequential(*layers).train()


def trainable_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for inputs, targets in loader:
        predictions = model(inputs).argmax(dim=1)
        correct += int((predictions == targets).sum())
        total += len(targets)
    model.train()
    return correct / total


def run_method(
    method: str,
    config: ExperimentConfig,
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    stream_identity: dict[str, str],
) -> dict[str, Any]:
    model = build_model(config, method, seed)
    initial_hash = trainable_parameter_hash(model)
    updates = 0
    started = time.perf_counter()
    if method == "pc":
        trainer = pc.PCTrainer(
            model,
            T=config.inference_steps,
            optimizer_x_fn=torch.optim.SGD,
            optimizer_x_kwargs={"lr": config.latent_lr},
            optimizer_p_fn=torch.optim.Adam,
            optimizer_p_kwargs={"lr": config.parameter_lr},
            update_p_at="last",
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.parameter_lr)

    for _epoch in range(config.epochs):
        for inputs, labels in train_loader:
            targets = F.one_hot(labels, num_classes=10).float()
            if method == "pc":
                trainer.train_on_batch(
                    inputs=inputs,
                    loss_fn=lambda output, target: 0.5 * (output - target).pow(2).sum(),
                    loss_fn_kwargs={"target": targets},
                )
            else:
                optimizer.zero_grad()
                loss = 0.5 * (model(inputs) - targets).pow(2).sum()
                loss.backward()
                optimizer.step()
            updates += 1

    return {
        "status": "completed",
        "method": method,
        "seed": seed,
        "initial_parameters_sha256": initial_hash,
        "stream_identity": stream_identity,
        "metrics": {"test_accuracy": evaluate(model, test_loader)},
        "resources": {
            "duration_seconds": time.perf_counter() - started,
            "parameter_updates": updates,
            "inference_steps_per_update": config.inference_steps if method == "pc" else 0,
            "peak_memory_bytes": None,
        },
        "warnings": [],
        "errors": [],
    }


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip()


def provenance() -> dict[str, Any]:
    return {
        "repository_commit": git_output("rev-parse", "HEAD"),
        "repository_dirty": bool(git_output("status", "--porcelain")),
        "predictive_coding_commit": git_output(
            "-C", str(SUBMODULE), "rev-parse", "HEAD"
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "platform": platform.platform(),
    }


def atomic_write_new(record: dict[str, Any], destination: Path) -> None:
    """Atomically create destination and refuse to replace prior evidence."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite run artifact: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "protocol_id",
        "interpretation",
        "status",
        "config",
        "seed",
        "methods_requested",
        "runs",
        "provenance",
    }
    if missing := required - record.keys():
        raise ValueError(f"record missing fields: {sorted(missing)}")
    if record["protocol_id"] != PROTOCOL_ID or record["interpretation"] != INTERPRETATION:
        raise ValueError("record does not belong to the Step 1 static protocol")
    if record["status"] not in {"completed", "failed"}:
        raise ValueError(f"invalid record status: {record['status']!r}")
    if not record["runs"]:
        raise ValueError("record contains no method runs")
    identities = [run.get("stream_identity") for run in record["runs"]]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("method runs do not share one realized stream")
    completed = [run for run in record["runs"] if run.get("status") == "completed"]
    if len(completed) > 1:
        hashes = {run.get("initial_parameters_sha256") for run in completed}
        if len(hashes) != 1:
            raise ValueError("method runs do not share initial trainable parameters")


def run_seed(config: ExperimentConfig, seed: int, methods: list[str]) -> dict[str, Any]:
    seed_everything(seed)
    train_loader, test_loader, stream_identity = load_data(config, seed)
    runs: list[dict[str, Any]] = []
    failed = False
    for method in methods:
        try:
            runs.append(run_method(method, config, seed, train_loader, test_loader, stream_identity))
        except Exception as error:  # preserve the failure before returning nonzero
            failed = True
            runs.append(
                {
                    "status": "failed",
                    "method": method,
                    "seed": seed,
                    "stream_identity": stream_identity,
                    "warnings": [],
                    "errors": [repr(error)],
                }
            )
            break
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "interpretation": INTERPRETATION,
        "status": "failed" if failed else "completed",
        "config": asdict(config),
        "seed": seed,
        "methods_requested": methods,
        "runs": runs,
        "provenance": provenance(),
    }


def printable_config() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "interpretation": INTERPRETATION,
        "configs": {name: asdict(config) for name, config in CONFIGS.items()},
        "constraints": {
            "device": "cpu",
            "continual_learning": False,
            "result_claims_authorized": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--methods", choices=["pc", "bp"], nargs="+", default=["pc", "bp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    args = parser.parse_args(argv)
    if args.print_config:
        print(json.dumps(printable_config(), indent=2, sort_keys=True))
        return 0
    if args.config is None:
        parser.error("--config is required unless --print-config is used")

    any_failed = False
    for seed in args.seeds:
        destination = args.output_dir / f"{args.config}-seed-{seed}.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite run artifact: {destination}")
        record = run_seed(CONFIGS[args.config], seed, args.methods)
        validate_record(record)
        atomic_write_new(record, destination)
        print(destination)
        any_failed = any_failed or record["status"] != "completed"
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
