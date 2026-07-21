#!/usr/bin/env python3
"""Matched replay-free BP-PC experiment on five-task SplitCIFAR-10."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    CLASS_PAIRS,
    EXPECTED_PARAMETER_COUNT,
    SPLIT_FIXED_LR,
    build_model,
    continual_metrics,
    copy_learned_parameters,
    distribution,
    environment_record,
    index_checksum,
    latent_state_elements,
    learned_parameter_count,
    log_pc_dynamics,
    load_runtime,
    make_summary_writer,
    make_trainer,
    parameter_checksum,
    seed_everything,
    select_device,
    squared_error,
    synchronize,
)
from representation_viz import (
    RepresentationTracker,
    balanced_anchor_indices,
    log_comparison_dashboard,
    plot_accuracy_matrix,
)


@dataclass(frozen=True)
class SplitProtocol:
    seeds: tuple[int, ...] = (7, 42, 123)
    methods: tuple[str, ...] = ("bp", "pc")
    tasks: int = 5
    epochs_per_task: int = 1
    batch_size: int = 200
    pc_steps: int = 16
    pc_state_lr: float = 0.5
    bp_lr: float = SPLIT_FIXED_LR["bp"]
    pc_lr: float = SPLIT_FIXED_LR["pc"]
    validation_fraction: float = 0.1
    split_seed: int = 20260717
    max_train_samples_per_task: int | None = None
    max_eval_samples_per_task: int | None = None
    representation_diagnostics: bool = False
    anchor_samples_per_class: int = 20
    data_root: str = "data"
    device: str = "auto"

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not self.methods or set(self.methods) - {"bp", "pc"}:
            raise ValueError("methods must contain only bp and/or pc")
        if self.tasks != 5:
            raise ValueError("SplitCIFAR-10 v1 requires five tasks")
        for name in ("epochs_per_task", "batch_size", "pc_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pc_state_lr <= 0 or self.bp_lr <= 0 or self.pc_lr <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        for name in ("max_train_samples_per_task", "max_eval_samples_per_task"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.anchor_samples_per_class <= 0 or self.anchor_samples_per_class > 1000:
            raise ValueError("anchor_samples_per_class must be in [1, 1000]")


class OrderedSubset:
    def __init__(self, dataset: Any, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Any, Any]:
        return self.dataset[self.indices[position]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--methods", choices=("bp", "pc"), nargs="+", default=["bp", "pc"])
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--epochs-per-task", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--pc-steps", type=int, default=16)
    parser.add_argument("--pc-state-lr", type=float, default=0.5)
    parser.add_argument("--bp-lr", type=float, default=SPLIT_FIXED_LR["bp"])
    parser.add_argument("--pc-lr", type=float, default=SPLIT_FIXED_LR["pc"])
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=20260717)
    parser.add_argument("--max-train-samples-per-task", type=int)
    parser.add_argument("--max-eval-samples-per-task", type=int)
    parser.add_argument("--representation-diagnostics", action="store_true")
    parser.add_argument("--anchor-samples-per-class", type=int, default=20)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def protocol_from_args(args: argparse.Namespace) -> SplitProtocol:
    protocol = SplitProtocol(
        seeds=tuple(args.seeds), methods=tuple(args.methods), tasks=args.tasks,
        epochs_per_task=args.epochs_per_task, batch_size=args.batch_size,
        pc_steps=args.pc_steps, pc_state_lr=args.pc_state_lr,
        bp_lr=args.bp_lr, pc_lr=args.pc_lr,
        validation_fraction=args.validation_fraction, split_seed=args.split_seed,
        max_train_samples_per_task=args.max_train_samples_per_task,
        max_eval_samples_per_task=args.max_eval_samples_per_task,
        representation_diagnostics=args.representation_diagnostics,
        anchor_samples_per_class=args.anchor_samples_per_class,
        data_root=args.data_root, device=args.device,
    )
    protocol.validate()
    return protocol


def load_cifar(protocol: SplitProtocol, runtime: dict[str, Any]) -> tuple[Any, Any, list[int], list[int]]:
    torch, datasets, transforms = runtime["torch"], runtime["datasets"], runtime["transforms"]
    transform = transforms.ToTensor()
    train = datasets.CIFAR10(protocol.data_root, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(protocol.data_root, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(protocol.split_seed)
    shuffled = torch.randperm(len(train), generator=generator).tolist()
    valid_size = round(len(shuffled) * protocol.validation_fraction)
    return train, test, shuffled[valid_size:], shuffled[:valid_size]


def ordered(indices: list[int], seed: int, task_id: int, torch: Any) -> list[int]:
    generator = torch.Generator().manual_seed(seed * 10_000 + task_id)
    positions = torch.randperm(len(indices), generator=generator).tolist()
    return [indices[position] for position in positions]


def limit(indices: list[int], maximum: int | None) -> list[int]:
    return indices if maximum is None else indices[:maximum]


def build_tasks(seed: int, protocol: SplitProtocol, runtime: dict[str, Any],
                train: Any, test: Any, train_indices: list[int],
                valid_indices: list[int]) -> list[dict[str, Any]]:
    tasks = []
    train_targets, test_targets = train.targets, test.targets
    for task_id, classes in enumerate(CLASS_PAIRS):
        train_pool = [i for i in train_indices if int(train_targets[i]) in classes]
        valid_pool = [i for i in valid_indices if int(train_targets[i]) in classes]
        test_pool = [i for i in range(len(test)) if int(test_targets[i]) in classes]
        train_order = limit(ordered(train_pool, seed, task_id, runtime["torch"]), protocol.max_train_samples_per_task)
        valid_pool = limit(valid_pool, protocol.max_eval_samples_per_task)
        test_pool = limit(test_pool, protocol.max_eval_samples_per_task)
        tasks.append({
            "task_id": task_id,
            "classes": list(classes),
            "train": OrderedSubset(train, train_order),
            "validation": OrderedSubset(train, valid_pool),
            "test": OrderedSubset(test, test_pool),
            "stream_checksum": index_checksum(train_order),
            "validation_checksum": index_checksum(valid_pool),
            "test_checksum": index_checksum(test_pool),
        })
    return tasks


def make_loader(dataset: Any, protocol: SplitProtocol, runtime: dict[str, Any],
                generator: Any | None = None) -> Any:
    return runtime["torch"].utils.data.DataLoader(
        dataset, batch_size=protocol.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False, generator=generator,
    )


def accuracy(model: Any, dataset: Any, protocol: SplitProtocol, device: Any,
             runtime: dict[str, Any]) -> float:
    return evaluate_accuracy(model, make_loader(dataset, protocol, runtime), device, runtime)


def evaluate_accuracy(model: Any, loader: Any, device: Any, runtime: dict[str, Any]) -> float:
    torch = runtime["torch"]
    model.eval()
    correct, count = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            correct += (model(inputs).argmax(1) == labels).sum().item()
            count += labels.numel()
    if count == 0:
        raise RuntimeError("empty evaluation task")
    return correct / count


def accuracy_row(model: Any, tasks: list[dict[str, Any]], split: str,
                 protocol: SplitProtocol, device: Any,
                 runtime: dict[str, Any]) -> list[float]:
    return [accuracy(model, task[split], protocol, device, runtime) for task in tasks]


def train_method(method: str, model: Any, tasks: list[dict[str, Any]],
                 protocol: SplitProtocol, seed: int, device: Any,
                 runtime: dict[str, Any], tensorboard_dir: Path | None = None,
                 run_label: str = "split", anchor_set: Any | None = None) -> dict[str, Any]:
    torch, functional = runtime["torch"], runtime["functional"]
    model.to(device)
    parameter_lr = protocol.bp_lr if method == "bp" else protocol.pc_lr
    trainer = make_trainer(model, parameter_lr, protocol.pc_steps, protocol.pc_state_lr, runtime)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    evaluation_started = time.perf_counter()
    test_matrix = [accuracy_row(model, tasks, "test", protocol, device, runtime)]
    validation_matrix = [accuracy_row(model, tasks, "validation", protocol, device, runtime)]
    synchronize(device, torch)
    evaluation_seconds = time.perf_counter() - evaluation_started
    prequential_correct, prequential_count, samples_seen = 0, 0, 0
    stream_seconds, peak_latent = 0.0, 0
    prequential_by_task, dynamics = [], []
    writer = make_summary_writer(
        tensorboard_dir, run_label, method, seed,
        {**asdict(protocol), "method": method, "seed": seed,
         "parameter_lr": parameter_lr},
    )
    representation_tracker = (
        RepresentationTracker(
            model,
            make_loader(
                anchor_set, protocol, runtime,
                torch.Generator().manual_seed(protocol.split_seed + 31_416),
            ),
            device, runtime, writer,
        )
        if anchor_set is not None else None
    )
    representation_diagnostics = (
        [representation_tracker.record(0)] if representation_tracker is not None else None
    )
    for task_id, task in enumerate(tasks):
        task_correct, task_count = 0, 0
        captured = False
        synchronize(device, torch)
        stream_started = time.perf_counter()
        for _ in range(protocol.epochs_per_task):
            for inputs, labels in make_loader(task["train"], protocol, runtime):
                inputs, labels = inputs.to(device), labels.to(device)
                model.eval()
                with torch.no_grad():
                    predictions = model(inputs).argmax(1)
                task_correct += (predictions == labels).sum().item()
                task_count += labels.numel()
                target = functional.one_hot(labels, num_classes=10).float()
                model.train()
                capture = method == "pc" and not captured
                result = trainer.train_on_batch(
                    inputs=inputs, loss_fn=squared_error,
                    loss_fn_kwargs={"target": target},
                    is_return_results_every_t=capture,
                )
                if capture:
                    record = {
                        "task_id": task_id, "loss": result["loss"],
                        "energy": result["energy"], "overall": result["overall"],
                        "overall_decreased": result["overall"][-1] <= result["overall"][0],
                    }
                    dynamics.append(record)
                    captured = True
                peak_latent = max(peak_latent, latent_state_elements(model, runtime))
                samples_seen += labels.numel()
        synchronize(device, torch)
        stream_seconds += time.perf_counter() - stream_started
        prequential_correct += task_correct
        prequential_count += task_count
        prequential_by_task.append(task_correct / task_count)
        evaluation_started = time.perf_counter()
        test_matrix.append(accuracy_row(model, tasks, "test", protocol, device, runtime))
        validation_matrix.append(accuracy_row(model, tasks, "validation", protocol, device, runtime))
        synchronize(device, torch)
        evaluation_seconds += time.perf_counter() - evaluation_started
        if writer is not None:
            boundary = task_id + 1
            writer.add_scalar("stream/prequential_accuracy_task", prequential_by_task[-1], boundary)
            writer.add_scalar("train/samples_seen", samples_seen, boundary)
            writer.add_scalar("runtime/stream_seconds_cumulative", stream_seconds, boundary)
            for evaluated_task, value in enumerate(test_matrix[-1]):
                writer.add_scalar(f"test/task_{evaluated_task}_accuracy", value, boundary)
            for evaluated_task, value in enumerate(validation_matrix[-1]):
                writer.add_scalar(f"validation/task_{evaluated_task}_accuracy", value, boundary)
            writer.flush()
        if representation_tracker is not None:
            representation_diagnostics.append(
                representation_tracker.record(task_id + 1, test_matrix)
            )
    metrics = continual_metrics(test_matrix, prequential_correct / prequential_count)
    metrics["prequential_accuracy_by_task"] = prequential_by_task
    if writer is not None:
        for name in (
            "final_average_accuracy", "prequential_accuracy", "backward_transfer",
            "average_forgetting", "mean_adaptation_gain",
        ):
            writer.add_scalar(f"summary/{name}", metrics[name], protocol.tasks)
        for record in dynamics:
            log_pc_dynamics(writer, record, f"pc_inference/task_{record['task_id']}")
        writer.add_figure(
            "continual/test_accuracy_matrix",
            plot_accuracy_matrix(test_matrix, f"{method.upper()} seed {seed}: test accuracy"),
            protocol.tasks, close=True,
        )
        writer.add_figure(
            "continual/validation_accuracy_matrix",
            plot_accuracy_matrix(validation_matrix, f"{method.upper()} seed {seed}: validation accuracy"),
            protocol.tasks, close=True,
        )
        writer.close()
    if not all(math.isfinite(value) for value in (
        metrics["final_average_accuracy"], metrics["prequential_accuracy"],
        metrics["average_forgetting"], metrics["mean_adaptation_gain"],
    )):
        raise RuntimeError("non-finite continual metric")
    return {
        "method": method, "seed": seed, "parameter_lr": parameter_lr,
        "test_accuracy_matrix": test_matrix,
        "validation_accuracy_matrix": validation_matrix,
        "metrics": metrics, "samples_seen": samples_seen,
        "stream_processing_seconds": stream_seconds,
        "evaluation_seconds": evaluation_seconds,
        "learned_parameter_count": learned_parameter_count(model, runtime),
        "final_latent_state_elements": latent_state_elements(model, runtime),
        "peak_latent_state_elements": peak_latent,
        "pc_first_batch_dynamics_by_task": dynamics or None,
        "representation_diagnostics": representation_diagnostics,
        "peak_accelerator_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def paired_difference(bp: dict[str, Any], pc: dict[str, Any]) -> dict[str, float]:
    names = (
        "final_average_accuracy", "prequential_accuracy", "backward_transfer",
        "average_forgetting", "mean_adaptation_gain",
    )
    differences = {name: pc["metrics"][name] - bp["metrics"][name] for name in names}
    differences["stream_processing_ratio"] = pc["stream_processing_seconds"] / bp["stream_processing_seconds"]
    return differences


def aggregate(results: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    np = runtime["np"]
    names = (
        "final_average_accuracy", "prequential_accuracy", "backward_transfer",
        "average_forgetting", "mean_adaptation_gain",
    )
    methods = {}
    for method in ("bp", "pc"):
        selected = [run for result in results for run in result["runs"] if run["method"] == method]
        if selected:
            methods[method] = {
                name: distribution([run["metrics"][name] for run in selected], np)
                for name in names
            }
    paired = [result["pc_minus_bp"] for result in results if result["pc_minus_bp"]]
    return {
        "methods": methods,
        "pc_minus_bp": {
            name: distribution([item[name] for item in paired], np)
            for name in paired[0]
        } if paired else None,
    }


def run(protocol: SplitProtocol, runtime: dict[str, Any],
        tensorboard_dir: Path | None = None, run_label: str = "split") -> dict[str, Any]:
    torch = runtime["torch"]
    device = select_device(protocol.device, torch)
    train, test, train_indices, valid_indices = load_cifar(protocol, runtime)
    anchor_indices = (
        balanced_anchor_indices(
            test, protocol.anchor_samples_per_class, protocol.split_seed + 31_415, torch
        )
        if protocol.representation_diagnostics else []
    )
    anchor_set = OrderedSubset(test, anchor_indices) if anchor_indices else None
    results = []
    for seed in protocol.seeds:
        seed_everything(seed, runtime)
        tasks = build_tasks(seed, protocol, runtime, train, test, train_indices, valid_indices)
        bp = build_model(False, runtime)
        pc = build_model(True, runtime)
        copy_learned_parameters(bp, pc, runtime)
        models = {"bp": bp, "pc": pc}
        checksums = {name: parameter_checksum(model, runtime) for name, model in models.items()}
        if checksums["bp"] != checksums["pc"]:
            raise RuntimeError("paired initial parameters do not match")
        if any(learned_parameter_count(model, runtime) != EXPECTED_PARAMETER_COUNT for model in models.values()):
            raise RuntimeError("paper architecture parameter count changed")
        runs = []
        for method in protocol.methods:
            result = train_method(
                method, models[method], tasks, protocol, seed, device, runtime,
                tensorboard_dir, run_label,
                anchor_set,
            )
            result["initial_parameter_checksum"] = checksums[method]
            runs.append(result)
        by_method = {item["method"]: item for item in runs}
        results.append({
            "seed": seed,
            "task_specs": [{
                "task_id": task["task_id"], "classes": task["classes"],
                "train_samples": len(task["train"]),
                "validation_samples": len(task["validation"]),
                "test_samples": len(task["test"]),
                "stream_checksum": task["stream_checksum"],
                "validation_checksum": task["validation_checksum"],
                "test_checksum": task["test_checksum"],
            } for task in tasks],
            "runs": runs,
            "pc_minus_bp": paired_difference(by_method["bp"], by_method["pc"])
            if set(by_method) == {"bp", "pc"} else None,
        })
    payload = {
        "protocol_id": "split-cifar10-paper-pc-extension-v1",
        "claim_status": "project extension; not part of Song et al. Figure 4i",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": asdict(protocol),
        "regime_notes": {
            "regime": "class-incremental, five disjoint class pairs, shared ten-way head",
            "stream": "one pass per task, predict before update, no replay or future access",
            "task_information": "boundaries used by evaluator but not passed to either model",
            "learning_rates": "fixed before launch from paper source-data cross-seed means; SplitCIFAR test data was not used for tuning",
            "updates": "paper learning-rule semantics retained: T=16 and parameter updates at every inference step for both RBP and PC",
        },
        "base_split_checksums": {
            "train": index_checksum(train_indices), "validation": index_checksum(valid_indices),
            "representation_anchors": index_checksum(anchor_indices) if anchor_indices else None,
        },
        "environment": environment_record(runtime, device),
        "seed_results": results,
        "aggregate_results": aggregate(results, runtime),
    }
    if tensorboard_dir is not None:
        log_comparison_dashboard(payload, tensorboard_dir, run_label)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = protocol_from_args(args)
    if args.print_config:
        print(json.dumps(asdict(protocol), indent=2))
        return 0
    if args.output is None:
        raise SystemExit("--output is required")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {args.output}")
    result = run(
        protocol, load_runtime(), args.tensorboard_dir,
        args.output.stem if args.output is not None else "split",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
