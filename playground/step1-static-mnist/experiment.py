#!/usr/bin/env python3
"""Matched static-MNIST comparison of backpropagation and predictive coding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Protocol:
    seeds: tuple[int, ...] = (7, 42, 123)
    methods: tuple[str, ...] = ("bp", "pc")
    epochs: int = 10
    batch_size: int = 500
    hidden_size: int = 256
    hidden_layers: int = 2
    parameter_lr: float = 0.001
    pc_steps: int = 20
    pc_state_lr: float = 0.01
    validation_fraction: float = 0.2
    split_seed: int = 20260717
    max_train_samples: int | None = None
    max_valid_samples: int | None = None
    max_test_samples: int | None = None
    data_root: str = "data"
    device: str = "auto"

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not self.methods or set(self.methods) - {"bp", "pc"}:
            raise ValueError("methods must contain only 'bp' and/or 'pc'")
        for name in ("epochs", "batch_size", "hidden_size", "hidden_layers", "pc_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.parameter_lr <= 0 or self.pc_state_lr <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        for name in ("max_train_samples", "max_valid_samples", "max_test_samples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--methods", choices=("bp", "pc"), nargs="+", default=["bp", "pc"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--parameter-lr", type=float, default=0.001)
    parser.add_argument("--pc-steps", type=int, default=20)
    parser.add_argument("--pc-state-lr", type=float, default=0.01)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260717)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def protocol_from_args(args: argparse.Namespace) -> Protocol:
    protocol = Protocol(
        seeds=tuple(args.seeds), methods=tuple(args.methods), epochs=args.epochs,
        batch_size=args.batch_size, hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        parameter_lr=args.parameter_lr, pc_steps=args.pc_steps,
        pc_state_lr=args.pc_state_lr, validation_fraction=args.validation_fraction,
        split_seed=args.split_seed, max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples, max_test_samples=args.max_test_samples,
        data_root=args.data_root, device=args.device,
    )
    protocol.validate()
    return protocol


def load_runtime() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        import torch.optim as optim
        import torchvision
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch runtime unavailable. Install torch, torchvision, numpy, pandas, "
            "matplotlib, seaborn, and tqdm before running the experiment."
        ) from exc

    pc_root = Path(__file__).resolve().parents[1] / "predictive-coding"
    if not (pc_root / "predictive_coding" / "pc_trainer.py").exists():
        raise RuntimeError("predictive-coding submodule is missing; initialize git submodules")
    sys.path.insert(0, str(pc_root))
    try:
        import predictive_coding as pc
    except ImportError as exc:
        raise RuntimeError(
            "Predictive-coding dependencies are unavailable. See this experiment's README."
        ) from exc
    return locals()


def seed_everything(seed: int, torch: Any, np: Any) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def limited_subset(dataset: Any, limit: int | None, torch: Any) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, range(limit))


def resolved_dataset_indices(dataset: Any) -> list[int]:
    if not hasattr(dataset, "indices"):
        return list(range(len(dataset)))
    parent = resolved_dataset_indices(dataset.dataset)
    return [parent[int(index)] for index in dataset.indices]


def dataset_index_checksum(dataset: Any) -> str:
    payload = json.dumps(resolved_dataset_indices(dataset), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def make_datasets(protocol: Protocol, runtime: dict[str, Any]) -> tuple[Any, Any, Any]:
    torch, datasets, transforms = runtime["torch"], runtime["datasets"], runtime["transforms"]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])
    full_train = datasets.MNIST(protocol.data_root, train=True, download=True, transform=transform)
    test = datasets.MNIST(protocol.data_root, train=False, download=True, transform=transform)
    valid_size = round(len(full_train) * protocol.validation_fraction)
    train_size = len(full_train) - valid_size
    generator = torch.Generator().manual_seed(protocol.split_seed)
    train, valid = torch.utils.data.random_split(full_train, [train_size, valid_size], generator=generator)
    return (
        limited_subset(train, protocol.max_train_samples, torch),
        limited_subset(valid, protocol.max_valid_samples, torch),
        limited_subset(test, protocol.max_test_samples, torch),
    )


def make_loader(dataset: Any, protocol: Protocol, seed: int, shuffle: bool, torch: Any) -> Any:
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=protocol.batch_size, shuffle=shuffle, generator=generator,
        num_workers=0, drop_last=False,
    )


def build_models(protocol: Protocol, runtime: dict[str, Any]) -> dict[str, Any]:
    nn, pc = runtime["nn"], runtime["pc"]
    bp_modules = []
    input_size = 784
    for _ in range(protocol.hidden_layers):
        bp_modules.extend((nn.Linear(input_size, protocol.hidden_size), nn.ReLU()))
        input_size = protocol.hidden_size
    bp_modules.append(nn.Linear(input_size, 10))
    bp = nn.Sequential(*bp_modules)

    pc_modules = []
    input_size = 784
    for _ in range(protocol.hidden_layers):
        pc_modules.extend((
            nn.Linear(input_size, protocol.hidden_size), pc.PCLayer(), nn.ReLU(),
        ))
        input_size = protocol.hidden_size
    pc_modules.append(nn.Linear(input_size, 10))
    pc_model = nn.Sequential(*pc_modules)
    bp_linears = [module for module in bp if isinstance(module, nn.Linear)]
    pc_linears = [module for module in pc_model if isinstance(module, nn.Linear)]
    for source, target in zip(bp_linears, pc_linears, strict=True):
        target.load_state_dict(source.state_dict())
    return {"bp": bp, "pc": pc_model}


def parameter_checksum(model: Any, nn: Any) -> str:
    digest = hashlib.sha256()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            for parameter in module.parameters():
                digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def learned_parameter_count(model: Any, nn: Any) -> int:
    return sum(
        parameter.numel()
        for module in model.modules()
        if isinstance(module, nn.Linear)
        for parameter in module.parameters()
    )


def latent_state_elements(model: Any, pc: Any) -> int:
    return sum(
        module.get_x().numel()
        for module in model.modules()
        if isinstance(module, pc.PCLayer) and module.get_x() is not None
    )


def squared_error(logits: Any, target: Any) -> Any:
    return 0.5 * (logits - target).pow(2).sum()


def evaluate(model: Any, loader: Any, device: Any, runtime: dict[str, Any]) -> dict[str, float]:
    torch, functional = runtime["torch"], runtime["functional"]
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            targets = functional.one_hot(labels, num_classes=10).float()
            loss_sum += squared_error(logits, targets).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.numel()
    return {"loss_per_example": loss_sum / count, "accuracy": correct / count}


def synchronize(device: Any, torch: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warmup_methods(protocol: Protocol, device: Any, runtime: dict[str, Any]) -> None:
    """Run one discarded update per selected method before timing."""
    torch, functional, optim, pc = (
        runtime["torch"], runtime["functional"], runtime["optim"], runtime["pc"]
    )
    inputs = torch.zeros(min(protocol.batch_size, 8), 784, device=device)
    labels = torch.arange(inputs.shape[0], device=device) % 10
    targets = functional.one_hot(labels, num_classes=10).float()
    for method in protocol.methods:
        model = build_models(protocol, runtime)[method].to(device)
        model.train()
        if method == "bp":
            optimizer = optim.Adam(model.parameters(), lr=protocol.parameter_lr)
            optimizer.zero_grad()
            squared_error(model(inputs), targets).backward()
            optimizer.step()
        else:
            trainer = pc.PCTrainer(
                model, T=protocol.pc_steps, optimizer_x_fn=optim.SGD,
                optimizer_x_kwargs={"lr": protocol.pc_state_lr}, update_p_at="last",
                optimizer_p_fn=optim.Adam,
                optimizer_p_kwargs={"lr": protocol.parameter_lr},
            )
            trainer.train_on_batch(
                inputs=inputs, loss_fn=squared_error,
                loss_fn_kwargs={"target": targets},
            )
        synchronize(device, torch)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train_method(
    method: str, model: Any, datasets: tuple[Any, Any, Any], protocol: Protocol,
    seed: int, device: Any, runtime: dict[str, Any],
) -> dict[str, Any]:
    torch, nn, functional, optim, pc = (
        runtime["torch"], runtime["nn"], runtime["functional"],
        runtime["optim"], runtime["pc"]
    )
    train_set, valid_set, test_set = datasets
    train_loader = make_loader(train_set, protocol, seed, True, torch)
    train_eval_loader = make_loader(train_set, protocol, seed, False, torch)
    valid_loader = make_loader(valid_set, protocol, seed, False, torch)
    test_loader = make_loader(test_set, protocol, seed, False, torch)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = None
    trainer = None
    if method == "bp":
        optimizer = optim.Adam(model.parameters(), lr=protocol.parameter_lr)
    else:
        trainer = pc.PCTrainer(
            model, T=protocol.pc_steps, optimizer_x_fn=optim.SGD,
            optimizer_x_kwargs={"lr": protocol.pc_state_lr}, update_p_at="last",
            optimizer_p_fn=optim.Adam, optimizer_p_kwargs={"lr": protocol.parameter_lr},
        )

    history = [{"epoch": 0, "samples_seen": 0, "validation": evaluate(model, valid_loader, device, runtime)}]
    samples_seen, timed_update_samples, dynamics = 0, 0, None
    update_seconds, diagnostic_update_seconds = 0.0, 0.0
    for epoch in range(1, protocol.epochs + 1):
        model.train()
        epoch_update_started = None
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            targets = functional.one_hot(labels, num_classes=10).float()
            is_diagnostic_batch = dynamics is None if method == "pc" else samples_seen == 0
            if is_diagnostic_batch:
                synchronize(device, torch)
                diagnostic_started = time.perf_counter()
            elif epoch_update_started is None:
                synchronize(device, torch)
                epoch_update_started = time.perf_counter()
            if method == "bp":
                optimizer.zero_grad()
                loss = squared_error(model(inputs), targets)
                loss.backward()
                optimizer.step()
            else:
                capture = dynamics is None
                results = trainer.train_on_batch(
                    inputs=inputs, loss_fn=squared_error,
                    loss_fn_kwargs={"target": targets},
                    is_return_results_every_t=capture,
                )
                if capture:
                    dynamics = {
                        "loss": results["loss"], "energy": results["energy"],
                        "overall": results["overall"],
                        "overall_decreased": (
                            results["overall"][-1] <= results["overall"][0]
                            if len(results["overall"]) > 1 else None
                        ),
                    }
            if is_diagnostic_batch:
                synchronize(device, torch)
                diagnostic_update_seconds += time.perf_counter() - diagnostic_started
            else:
                timed_update_samples += labels.numel()
            samples_seen += labels.numel()
        if epoch_update_started is not None:
            synchronize(device, torch)
            update_seconds += time.perf_counter() - epoch_update_started
        history.append({
            "epoch": epoch, "samples_seen": samples_seen,
            "validation": evaluate(model, valid_loader, device, runtime),
        })
    return {
        "method": method, "seed": seed, "history": history,
        "train_final": evaluate(model, train_eval_loader, device, runtime),
        "test": evaluate(model, test_loader, device, runtime),
        "training_seconds": update_seconds,
        "timed_update_samples": timed_update_samples,
        "diagnostic_update_seconds": diagnostic_update_seconds,
        "learned_parameter_count": learned_parameter_count(model, nn),
        "latent_state_elements": latent_state_elements(model, pc),
        "pc_first_batch_dynamics": dynamics,
        "peak_accelerator_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def run(protocol: Protocol, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, np, nn = runtime["torch"], runtime["np"], runtime["nn"]
    if protocol.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(protocol.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU execution was required, but PyTorch cannot access ROCm/CUDA")
    datasets = make_datasets(protocol, runtime)
    warmup_methods(protocol, device, runtime)
    runs = []
    for seed in protocol.seeds:
        seed_everything(seed, torch, np)
        models = build_models(protocol, runtime)
        checksums = {name: parameter_checksum(model, nn) for name, model in models.items()}
        if checksums["bp"] != checksums["pc"]:
            raise RuntimeError("paired models do not share identical initial linear parameters")
        for method in protocol.methods:
            result = train_method(method, models[method], datasets, protocol, seed, device, runtime)
            result["initial_parameter_checksum"] = checksums[method]
            runs.append(result)
    return {
        "protocol_id": "step1-static-mnist-v1",
        "claim_status": "verified run output; interpretation requires paired-seed review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": asdict(protocol),
        "dataset_sizes": dict(zip(("train", "validation", "test"), map(len, datasets), strict=True)),
        "dataset_index_checksums": {
            name: dataset_index_checksum(dataset)
            for name, dataset in zip(("train", "validation", "test"), datasets, strict=True)
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torchvision": runtime["torchvision"].__version__,
            "numpy": np.__version__, "device": str(device),
            "execution_host": os.getenv("EXPERIMENT_EXECUTION_HOST"),
            "runtime_image": os.getenv("EXPERIMENT_RUNTIME_IMAGE"),
            "runtime_image_id": os.getenv("EXPERIMENT_RUNTIME_IMAGE_ID"),
        },
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = protocol_from_args(args)
    if args.print_config:
        print(json.dumps(asdict(protocol), indent=2))
        return 0
    if args.output is None:
        raise SystemExit("--output is required for experiment runs")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing run artifact: {args.output}")
    results = run(protocol, load_runtime())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
