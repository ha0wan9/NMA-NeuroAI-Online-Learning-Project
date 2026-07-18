#!/usr/bin/env python3
"""Matched BP–PC continual-learning runs on SplitMNIST and Permuted MNIST."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment import (
    build_models,
    latent_state_elements,
    learned_parameter_count,
    load_runtime,
    parameter_checksum,
    seed_everything,
    squared_error,
)


SPLIT_CLASS_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))


@dataclass(frozen=True)
class ContinualProtocol:
    scenarios: tuple[str, ...] = ("split", "permuted")
    seeds: tuple[int, ...] = (7, 42, 123)
    methods: tuple[str, ...] = ("bp", "pc")
    tasks: int = 5
    epochs_per_task: int = 1
    batch_size: int = 500
    hidden_size: int = 256
    hidden_layers: int = 2
    parameter_lr: float = 0.001
    pc_steps: int = 20
    pc_state_lr: float = 0.01
    validation_fraction: float = 0.2
    split_seed: int = 20260717
    permutation_seed: int = 0
    max_train_samples_per_task: int | None = None
    max_eval_samples_per_task: int | None = None
    data_root: str = "data"
    device: str = "auto"

    def validate(self) -> None:
        if not self.scenarios or set(self.scenarios) - {"split", "permuted"}:
            raise ValueError("scenarios must contain only 'split' and/or 'permuted'")
        if not self.methods or set(self.methods) - {"bp", "pc"}:
            raise ValueError("methods must contain only 'bp' and/or 'pc'")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.tasks != 5:
            raise ValueError("protocol v1 freezes both benchmarks to five tasks")
        for name in ("epochs_per_task", "batch_size", "hidden_size", "hidden_layers", "pc_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.parameter_lr <= 0 or self.pc_state_lr <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        for name in ("max_train_samples_per_task", "max_eval_samples_per_task"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")


class TaskDataset:
    """A deterministic ordered MNIST view with an optional pixel permutation."""

    def __init__(self, dataset: Any, indices: list[int], permutation: Any = None):
        self.dataset = dataset
        self.indices = indices
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Any, Any]:
        image, label = self.dataset[self.indices[position]]
        if self.permutation is not None:
            image = image[self.permutation]
        return image, label


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", choices=("split", "permuted"), nargs="+", default=["split", "permuted"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--methods", choices=("bp", "pc"), nargs="+", default=["bp", "pc"])
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--epochs-per-task", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--parameter-lr", type=float, default=0.001)
    parser.add_argument("--pc-steps", type=int, default=20)
    parser.add_argument("--pc-state-lr", type=float, default=0.01)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260717)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--max-train-samples-per-task", type=int)
    parser.add_argument("--max-eval-samples-per-task", type=int)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def protocol_from_args(args: argparse.Namespace) -> ContinualProtocol:
    protocol = ContinualProtocol(
        scenarios=tuple(args.scenarios), seeds=tuple(args.seeds), methods=tuple(args.methods),
        tasks=args.tasks, epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size, hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        parameter_lr=args.parameter_lr, pc_steps=args.pc_steps,
        pc_state_lr=args.pc_state_lr, validation_fraction=args.validation_fraction,
        split_seed=args.split_seed, permutation_seed=args.permutation_seed,
        max_train_samples_per_task=args.max_train_samples_per_task,
        max_eval_samples_per_task=args.max_eval_samples_per_task,
        data_root=args.data_root, device=args.device,
    )
    protocol.validate()
    return protocol


def select_device(protocol: ContinualProtocol, torch: Any) -> Any:
    device = torch.device(
        "cuda" if protocol.device == "auto" and torch.cuda.is_available()
        else "cpu" if protocol.device == "auto"
        else protocol.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU execution was required, but PyTorch cannot access ROCm/CUDA")
    return device


def load_mnist(protocol: ContinualProtocol, runtime: dict[str, Any]) -> tuple[Any, Any, list[int], list[int]]:
    torch, datasets, transforms = runtime["torch"], runtime["datasets"], runtime["transforms"]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])
    train_dataset = datasets.MNIST(protocol.data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(protocol.data_root, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(protocol.split_seed)
    shuffled = torch.randperm(len(train_dataset), generator=generator).tolist()
    valid_size = round(len(shuffled) * protocol.validation_fraction)
    return train_dataset, test_dataset, shuffled[valid_size:], shuffled[:valid_size]


def limited(indices: list[int], maximum: int | None) -> list[int]:
    return indices if maximum is None else indices[:maximum]


def task_order(indices: list[int], seed: int, task_id: int, torch: Any) -> list[int]:
    generator = torch.Generator().manual_seed(seed * 10_000 + task_id)
    order = torch.randperm(len(indices), generator=generator).tolist()
    return [indices[position] for position in order]


def task_checksum(indices: list[int], permutation: Any = None) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(indices, separators=(",", ":")).encode())
    if permutation is not None:
        digest.update(permutation.cpu().numpy().tobytes())
    return digest.hexdigest()


def build_tasks(
    scenario: str, seed: int, protocol: ContinualProtocol, runtime: dict[str, Any],
    train_dataset: Any, test_dataset: Any, train_indices: list[int], valid_indices: list[int],
) -> list[dict[str, Any]]:
    torch = runtime["torch"]
    train_targets, test_targets = train_dataset.targets, test_dataset.targets
    permutations = [torch.arange(784)]
    permutation_generator = torch.Generator().manual_seed(protocol.permutation_seed)
    permutations.extend(torch.randperm(784, generator=permutation_generator) for _ in range(protocol.tasks - 1))
    tasks = []
    for task_id in range(protocol.tasks):
        if scenario == "split":
            classes = SPLIT_CLASS_PAIRS[task_id]
            train_pool = [index for index in train_indices if int(train_targets[index]) in classes]
            valid_pool = [index for index in valid_indices if int(train_targets[index]) in classes]
            test_pool = [index for index in range(len(test_dataset)) if int(test_targets[index]) in classes]
            permutation = None
            descriptor = {"classes": list(classes)}
        else:
            train_pool, valid_pool = list(train_indices), list(valid_indices)
            test_pool = list(range(len(test_dataset)))
            permutation = permutations[task_id]
            descriptor = {"permutation_checksum": hashlib.sha256(permutation.numpy().tobytes()).hexdigest()}
        ordered_train = task_order(train_pool, seed, task_id, torch)
        ordered_train = limited(ordered_train, protocol.max_train_samples_per_task)
        valid_pool = limited(valid_pool, protocol.max_eval_samples_per_task)
        test_pool = limited(test_pool, protocol.max_eval_samples_per_task)
        tasks.append({
            "task_id": task_id,
            "train": TaskDataset(train_dataset, ordered_train, permutation),
            "validation": TaskDataset(train_dataset, valid_pool, permutation),
            "test": TaskDataset(test_dataset, test_pool, permutation),
            "stream_checksum": task_checksum(ordered_train, permutation),
            "descriptor": descriptor,
        })
    return tasks


def make_loader(dataset: Any, batch_size: int, torch: Any) -> Any:
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def synchronize(device: Any, torch: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def accuracy(model: Any, dataset: Any, protocol: ContinualProtocol, device: Any, runtime: dict[str, Any]) -> float:
    torch = runtime["torch"]
    model.eval()
    correct, count = 0, 0
    with torch.no_grad():
        for inputs, labels in make_loader(dataset, protocol.batch_size, torch):
            inputs, labels = inputs.to(device), labels.to(device)
            correct += (model(inputs).argmax(dim=1) == labels).sum().item()
            count += labels.numel()
    return correct / count


def accuracy_row(model: Any, tasks: list[dict[str, Any]], split: str, protocol: ContinualProtocol, device: Any, runtime: dict[str, Any]) -> list[float]:
    return [accuracy(model, task[split], protocol, device, runtime) for task in tasks]


def continual_metrics(matrix: list[list[float]], prequential: float) -> dict[str, Any]:
    task_count = len(matrix[0])
    final = matrix[-1]
    post_task = [matrix[task_id + 1][task_id] for task_id in range(task_count)]
    pre_task = [matrix[task_id][task_id] for task_id in range(task_count)]
    bwt_values = [final[task_id] - post_task[task_id] for task_id in range(task_count - 1)]
    forgetting_values = [
        max(row[task_id] for row in matrix[task_id + 1:]) - final[task_id]
        for task_id in range(task_count - 1)
    ]
    adaptation = [post - pre for post, pre in zip(post_task, pre_task, strict=True)]
    return {
        "final_average_accuracy": sum(final) / task_count,
        "prequential_accuracy": prequential,
        "backward_transfer": sum(bwt_values) / len(bwt_values),
        "average_forgetting": sum(forgetting_values) / len(forgetting_values),
        "mean_adaptation_gain": sum(adaptation) / len(adaptation),
        "adaptation_gain_by_task": adaptation,
        "final_accuracy_by_task": final,
        "post_training_accuracy_by_task": post_task,
    }


def warmup_methods(protocol: ContinualProtocol, device: Any, runtime: dict[str, Any]) -> None:
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
            loss = squared_error(model(inputs), targets)
            loss.backward()
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


def distribution(values: list[float], np: Any) -> dict[str, Any]:
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }


def aggregate_results(results: list[dict[str, Any]], np: Any) -> list[dict[str, Any]]:
    metric_names = (
        "final_average_accuracy", "prequential_accuracy", "backward_transfer",
        "average_forgetting", "mean_adaptation_gain",
    )
    resource_names = (
        "stream_processing_seconds", "evaluation_seconds",
        "peak_accelerator_memory_bytes", "final_latent_state_elements",
        "peak_latent_state_elements",
    )
    aggregates = []
    for scenario in sorted({result["scenario"] for result in results}):
        selected = [result for result in results if result["scenario"] == scenario]
        methods: dict[str, Any] = {}
        for method in sorted({run["method"] for result in selected for run in result["runs"]}):
            method_runs = [
                run for result in selected for run in result["runs"]
                if run["method"] == method
            ]
            methods[method] = {
                name: distribution([run["metrics"][name] for run in method_runs], np)
                for name in metric_names
            }
            methods[method].update({
                name: distribution([run[name] for run in method_runs], np)
                for name in resource_names
            })
        paired = [result["pc_minus_bp"] for result in selected if result["pc_minus_bp"]]
        aggregates.append({
            "scenario": scenario,
            "seeds": [result["seed"] for result in selected],
            "methods": methods,
            "pc_minus_bp": {
                name: distribution([difference[name] for difference in paired], np)
                for name in paired[0]
            } if paired else None,
        })
    return aggregates


def train_method(
    method: str, model: Any, tasks: list[dict[str, Any]], protocol: ContinualProtocol,
    seed: int, device: Any, runtime: dict[str, Any],
) -> dict[str, Any]:
    torch, nn, functional, optim, pc = (
        runtime["torch"], runtime["nn"], runtime["functional"],
        runtime["optim"], runtime["pc"],
    )
    model.to(device)
    warmup_inputs, _ = next(iter(make_loader(tasks[0]["train"], protocol.batch_size, torch)))
    model.eval()
    with torch.no_grad():
        model(warmup_inputs.to(device))
    synchronize(device, torch)
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

    synchronize(device, torch)
    evaluation_started = time.perf_counter()
    test_matrix = [accuracy_row(model, tasks, "test", protocol, device, runtime)]
    validation_matrix = [accuracy_row(model, tasks, "validation", protocol, device, runtime)]
    synchronize(device, torch)
    evaluation_seconds = time.perf_counter() - evaluation_started
    prequential_correct, prequential_count, samples_seen = 0, 0, 0
    timed_stream_samples, diagnostic_update_seconds = 0, 0.0
    prequential_by_task, pc_dynamics = [], []
    peak_latent_state = 0
    stream_seconds = 0.0
    for task_id, task in enumerate(tasks):
        task_correct, task_count = 0, 0
        captured = False
        task_stream_started = None
        for _epoch in range(protocol.epochs_per_task):
            for inputs, labels in make_loader(task["train"], protocol.batch_size, torch):
                inputs, labels = inputs.to(device), labels.to(device)
                model.eval()
                with torch.no_grad():
                    predictions = model(inputs).argmax(dim=1)
                task_correct += (predictions == labels).sum().item()
                task_count += labels.numel()
                targets = functional.one_hot(labels, num_classes=10).float()
                model.train()
                is_diagnostic_batch = not captured if method == "pc" else task_count == labels.numel()
                if is_diagnostic_batch:
                    synchronize(device, torch)
                    diagnostic_started = time.perf_counter()
                elif task_stream_started is None:
                    synchronize(device, torch)
                    task_stream_started = time.perf_counter()
                if method == "bp":
                    optimizer.zero_grad()
                    loss = squared_error(model(inputs), targets)
                    loss.backward()
                    optimizer.step()
                else:
                    results = trainer.train_on_batch(
                        inputs=inputs, loss_fn=squared_error,
                        loss_fn_kwargs={"target": targets},
                        is_return_results_every_t=not captured,
                    )
                    if not captured:
                        pc_dynamics.append({
                            "task_id": task_id, "loss": results["loss"],
                            "energy": results["energy"], "overall": results["overall"],
                            "overall_decreased": (
                                results["overall"][-1] <= results["overall"][0]
                                if len(results["overall"]) > 1 else None
                            ),
                        })
                        captured = True
                    peak_latent_state = max(peak_latent_state, latent_state_elements(model, pc))
                if is_diagnostic_batch:
                    synchronize(device, torch)
                    diagnostic_update_seconds += time.perf_counter() - diagnostic_started
                else:
                    timed_stream_samples += labels.numel()
                samples_seen += labels.numel()
        if task_stream_started is not None:
            synchronize(device, torch)
            stream_seconds += time.perf_counter() - task_stream_started
        prequential_correct += task_correct
        prequential_count += task_count
        prequential_by_task.append(task_correct / task_count)
        evaluation_started = time.perf_counter()
        test_matrix.append(accuracy_row(model, tasks, "test", protocol, device, runtime))
        validation_matrix.append(accuracy_row(model, tasks, "validation", protocol, device, runtime))
        synchronize(device, torch)
        boundary_elapsed = time.perf_counter() - evaluation_started
        evaluation_seconds += boundary_elapsed
    metrics = continual_metrics(test_matrix, prequential_correct / prequential_count)
    metrics["prequential_accuracy_by_task"] = prequential_by_task
    return {
        "method": method, "seed": seed, "test_accuracy_matrix": test_matrix,
        "validation_accuracy_matrix": validation_matrix, "metrics": metrics,
        "samples_seen": samples_seen, "stream_processing_seconds": stream_seconds,
        "timed_stream_samples": timed_stream_samples,
        "diagnostic_update_seconds": diagnostic_update_seconds,
        "evaluation_seconds": evaluation_seconds,
        "learned_parameter_count": learned_parameter_count(model, nn),
        "final_latent_state_elements": latent_state_elements(model, pc),
        "peak_latent_state_elements": peak_latent_state,
        "pc_first_batch_dynamics_by_task": pc_dynamics or None,
        "peak_accelerator_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def paired_difference(bp: dict[str, Any], pc: dict[str, Any]) -> dict[str, float]:
    metric_names = (
        "final_average_accuracy", "prequential_accuracy", "backward_transfer",
        "average_forgetting", "mean_adaptation_gain",
    )
    differences = {name: pc["metrics"][name] - bp["metrics"][name] for name in metric_names}
    differences.update({
        "stream_processing_seconds": pc["stream_processing_seconds"] - bp["stream_processing_seconds"],
        "stream_processing_ratio": pc["stream_processing_seconds"] / bp["stream_processing_seconds"],
        "peak_accelerator_memory_bytes": pc["peak_accelerator_memory_bytes"] - bp["peak_accelerator_memory_bytes"],
        "final_latent_state_elements": pc["final_latent_state_elements"] - bp["final_latent_state_elements"],
        "peak_latent_state_elements": pc["peak_latent_state_elements"] - bp["peak_latent_state_elements"],
    })
    return differences


def run(protocol: ContinualProtocol, runtime: dict[str, Any]) -> dict[str, Any]:
    torch, np, nn = runtime["torch"], runtime["np"], runtime["nn"]
    device = select_device(protocol, torch)
    train_dataset, test_dataset, train_indices, valid_indices = load_mnist(protocol, runtime)
    warmup_methods(protocol, device, runtime)
    scenario_results = []
    for scenario in protocol.scenarios:
        for seed in protocol.seeds:
            seed_everything(seed, torch, np)
            tasks = build_tasks(
                scenario, seed, protocol, runtime, train_dataset, test_dataset,
                train_indices, valid_indices,
            )
            models = build_models(protocol, runtime)
            checksums = {name: parameter_checksum(model, nn) for name, model in models.items()}
            if checksums["bp"] != checksums["pc"]:
                raise RuntimeError("paired models do not share identical initial parameters")
            runs = []
            for method in protocol.methods:
                result = train_method(method, models[method], tasks, protocol, seed, device, runtime)
                result["initial_parameter_checksum"] = checksums[method]
                runs.append(result)
            by_method = {result["method"]: result for result in runs}
            scenario_results.append({
                "scenario": scenario, "seed": seed,
                "task_specs": [{
                    "task_id": task["task_id"], "descriptor": task["descriptor"],
                    "train_samples": len(task["train"]),
                    "validation_samples": len(task["validation"]),
                    "test_samples": len(task["test"]),
                    "stream_checksum": task["stream_checksum"],
                } for task in tasks],
                "runs": runs,
                "pc_minus_bp": paired_difference(by_method["bp"], by_method["pc"])
                if set(by_method) == {"bp", "pc"} else None,
            })
    return {
        "protocol_id": "continual-mnist-pc-bp-v1",
        "claim_status": "verified run output; no superiority claim without multi-seed review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": asdict(protocol),
        "regime_notes": {
            "split": "class-incremental, single 10-way head, no task ID to model",
            "permuted": "domain-incremental, fixed permutations, no task ID to model",
            "replay": "none; Permuted MNIST revisits base images under a new domain by benchmark design",
            "evaluation": "predict-before-update on train stream; all task test sets after each boundary",
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torchvision": runtime["torchvision"].__version__,
            "numpy": np.__version__, "device": str(device),
            "execution_host": os.getenv("EXPERIMENT_EXECUTION_HOST"),
            "runtime_image": os.getenv("EXPERIMENT_RUNTIME_IMAGE"),
            "runtime_image_id": os.getenv("EXPERIMENT_RUNTIME_IMAGE_ID"),
        },
        "scenario_results": scenario_results,
        "aggregate_results": aggregate_results(scenario_results, np),
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
