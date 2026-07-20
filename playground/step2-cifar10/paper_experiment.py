#!/usr/bin/env python3
"""Exact-as-practical reproduction of Song et al. (2024), Figure 4i."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    EXPECTED_PARAMETER_COUNT,
    PAPER_BEST_ACCURACY,
    PAPER_BEST_LR,
    PAPER_SEEDS,
    build_model,
    environment_record,
    evaluate,
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


@dataclass(frozen=True)
class PaperProtocol:
    seeds: tuple[int, ...] = PAPER_SEEDS
    methods: tuple[str, ...] = ("bp", "pc")
    epochs: int = 80
    batch_size: int = 200
    pc_steps: int = 16
    pc_state_lr: float = 0.5
    train_per_class: int = 5000
    test_per_class: int = 1000
    bp_lr: float | None = None
    pc_lr: float | None = None
    data_root: str = "data"
    device: str = "auto"

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not self.methods or set(self.methods) - {"bp", "pc"}:
            raise ValueError("methods must contain only bp and/or pc")
        for name in ("epochs", "batch_size", "pc_steps", "train_per_class", "test_per_class"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pc_state_lr <= 0:
            raise ValueError("pc_state_lr must be positive")
        for method, override in (("bp", self.bp_lr), ("pc", self.pc_lr)):
            if override is not None and override <= 0:
                raise ValueError(f"{method}_lr must be positive")
            unknown = [seed for seed in self.seeds if seed not in PAPER_BEST_LR[method]]
            if method in self.methods and override is None and unknown:
                raise ValueError(f"{method}_lr is required for non-paper seeds {unknown}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PAPER_SEEDS))
    parser.add_argument("--methods", choices=("bp", "pc"), nargs="+", default=["bp", "pc"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--pc-steps", type=int, default=16)
    parser.add_argument("--pc-state-lr", type=float, default=0.5)
    parser.add_argument("--train-per-class", type=int, default=5000)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--bp-lr", type=float)
    parser.add_argument("--pc-lr", type=float)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def protocol_from_args(args: argparse.Namespace) -> PaperProtocol:
    protocol = PaperProtocol(
        seeds=tuple(args.seeds), methods=tuple(args.methods), epochs=args.epochs,
        batch_size=args.batch_size, pc_steps=args.pc_steps,
        pc_state_lr=args.pc_state_lr, train_per_class=args.train_per_class,
        test_per_class=args.test_per_class, bp_lr=args.bp_lr, pc_lr=args.pc_lr,
        data_root=args.data_root, device=args.device,
    )
    protocol.validate()
    return protocol


def parameter_lr(protocol: PaperProtocol, method: str, seed: int) -> float:
    override = protocol.bp_lr if method == "bp" else protocol.pc_lr
    return override if override is not None else PAPER_BEST_LR[method][seed]


def paper_subset(dataset: Any, per_class: int, torch: Any) -> tuple[Any, list[int]]:
    """Replicate the archived helper, including its equal-size off-by-one."""
    targets = torch.as_tensor(dataset.targets)
    selected: list[int] = []
    for label in range(10):
        candidates = (targets == label).nonzero(as_tuple=False).squeeze(1)
        shuffled = candidates[torch.randperm(candidates.numel())]
        count = per_class if per_class < candidates.numel() else candidates.numel() - 1
        selected.extend(int(value) for value in shuffled[:count].tolist())
    return torch.utils.data.Subset(dataset, selected), selected


def make_loader(dataset: Any, protocol: PaperProtocol, runtime: dict[str, Any]) -> Any:
    return runtime["torch"].utils.data.DataLoader(
        dataset, batch_size=protocol.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )


def make_datasets(protocol: PaperProtocol, runtime: dict[str, Any]) -> tuple[Any, Any, list[int], list[int]]:
    datasets, transforms, torch = runtime["datasets"], runtime["transforms"], runtime["torch"]
    transform = transforms.ToTensor()
    train_full = datasets.CIFAR10(protocol.data_root, train=True, download=True, transform=transform)
    test_full = datasets.CIFAR10(protocol.data_root, train=False, download=True, transform=transform)
    train, train_indices = paper_subset(train_full, protocol.train_per_class, torch)
    test, test_indices = paper_subset(test_full, protocol.test_per_class, torch)
    return train, test, train_indices, test_indices


def train_one(method: str, seed: int, protocol: PaperProtocol, device: Any,
              runtime: dict[str, Any], tensorboard_dir: Path | None = None,
              run_label: str = "paper") -> dict[str, Any]:
    torch, functional = runtime["torch"], runtime["functional"]
    seed_everything(seed, runtime)
    train_set, test_set, train_indices, test_indices = make_datasets(protocol, runtime)
    model = build_model(method == "pc", runtime).to(device)
    if learned_parameter_count(model, runtime) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("paper architecture parameter count changed")
    initial_checksum = parameter_checksum(model, runtime)
    trainer = make_trainer(
        model, parameter_lr(protocol, method, seed), protocol.pc_steps,
        protocol.pc_state_lr, runtime,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, Any]] = []
    samples_seen, update_seconds, dynamics = 0, 0.0, None
    writer = make_summary_writer(
        tensorboard_dir, run_label, method, seed,
        {**asdict(protocol), "method": method, "seed": seed,
         "parameter_lr": parameter_lr(protocol, method, seed)},
    )
    print(f"static start method={method} seed={seed} epochs={protocol.epochs}", flush=True)
    for epoch in range(1, protocol.epochs + 1):
        model.train()
        loader = make_loader(train_set, protocol, runtime)
        synchronize(device, torch)
        started = time.perf_counter()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            target = functional.one_hot(labels, num_classes=10).float()
            capture = method == "pc" and dynamics is None
            result = trainer.train_on_batch(
                inputs=inputs, loss_fn=squared_error,
                loss_fn_kwargs={"target": target},
                is_return_results_every_t=capture,
            )
            if capture:
                dynamics = {
                    "loss": result["loss"], "energy": result["energy"],
                    "overall": result["overall"],
                    "overall_decreased": result["overall"][-1] <= result["overall"][0],
                }
            samples_seen += labels.numel()
        synchronize(device, torch)
        update_seconds += time.perf_counter() - started
        test_metrics = evaluate(model, make_loader(test_set, protocol, runtime), device, runtime)
        history.append({"epoch": epoch, "samples_seen": samples_seen, "test": test_metrics})
        if writer is not None:
            writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
            writer.add_scalar("test/loss_per_example", test_metrics["loss_per_example"], epoch)
            writer.add_scalar("train/samples_seen", samples_seen, epoch)
            writer.add_scalar("runtime/update_seconds_cumulative", update_seconds, epoch)
            writer.flush()
        print(
            f"static progress method={method} seed={seed} epoch={epoch}/{protocol.epochs} "
            f"accuracy={test_metrics['accuracy']:.6f} update_seconds={update_seconds:.1f}",
            flush=True,
        )
        if not math.isfinite(test_metrics["accuracy"]):
            raise RuntimeError("non-finite paper metric")
    best = max(history, key=lambda row: row["test"]["accuracy"])
    log_pc_dynamics(writer, dynamics)
    if writer is not None:
        writer.add_scalar("summary/best_test_accuracy", best["test"]["accuracy"], 0)
        writer.close()
    source_target = PAPER_BEST_ACCURACY.get(method, {}).get(seed)
    return {
        "method": method, "seed": seed,
        "parameter_lr": parameter_lr(protocol, method, seed),
        "history": history,
        "best_test_accuracy": best["test"]["accuracy"],
        "best_epoch": best["epoch"],
        "final_test_accuracy": history[-1]["test"]["accuracy"],
        "source_target_accuracy": source_target,
        "absolute_target_difference": (
            best["test"]["accuracy"] - source_target if source_target is not None else None
        ),
        "within_registered_one_pp_tolerance": (
            abs(best["test"]["accuracy"] - source_target) <= 0.01
            if source_target is not None else None
        ),
        "initial_parameter_checksum": initial_checksum,
        "train_index_checksum": index_checksum(train_indices),
        "test_index_checksum": index_checksum(test_indices),
        "dataset_sizes": {"train": len(train_set), "test": len(test_set)},
        "samples_seen": samples_seen,
        "update_seconds": update_seconds,
        "learned_parameter_count": learned_parameter_count(model, runtime),
        "latent_state_elements": latent_state_elements(model, runtime),
        "pc_first_batch_dynamics": dynamics,
        "peak_accelerator_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def run(protocol: PaperProtocol, runtime: dict[str, Any],
        tensorboard_dir: Path | None = None, run_label: str = "paper") -> dict[str, Any]:
    torch, np = runtime["torch"], runtime["np"]
    device = select_device(protocol.device, torch)
    runs = [
        train_one(method, seed, protocol, device, runtime, tensorboard_dir, run_label)
        for seed in protocol.seeds
        for method in protocol.methods
    ]
    aggregates = {}
    for method in protocol.methods:
        selected = [run for run in runs if run["method"] == method]
        values = [run["best_test_accuracy"] for run in selected]
        aggregates[method] = {
            "best_test_accuracy_mean": float(np.mean(values)),
            "best_test_accuracy_sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "all_source_targets_within_one_pp": all(
                run["within_registered_one_pp_tolerance"] is True for run in selected
            ),
        }
    if set(aggregates) == {"bp", "pc"}:
        aggregates["pc_minus_bp_best_accuracy_mean"] = (
            aggregates["pc"]["best_test_accuracy_mean"]
            - aggregates["bp"]["best_test_accuracy_mean"]
        )
    return {
        "protocol_id": "song-2024-figure4i-cifar10-reproduction-v1",
        "claim_status": "paper-protocol reproduction; test-informed model selection is preserved and disclosed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": asdict(protocol),
        "paper_protocol_notes": {
            "source": "Song et al. Nature Neuroscience 2024, official repository revision recorded below",
            "selection": "per-seed learning rates and minimum test error were selected from archived source data",
            "updates": "both RBP and PC use PCTrainer T=16 with parameter updates at every inference step",
            "data": "archived subset helper omits one random example per class when the requested class size equals the dataset class size",
            "test_caveat": "test batches are shuffled, drop_last=True, and evaluated every epoch exactly as in the archived protocol",
        },
        "environment": environment_record(runtime, device),
        "runs": runs,
        "aggregates": aggregates,
    }


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
        args.output.stem if args.output is not None else "paper",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
