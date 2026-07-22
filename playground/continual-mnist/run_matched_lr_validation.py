#!/usr/bin/env python3
"""Run one auditable cell of the matched-LR paper-equation PC×CLASSP study.

The CLI deliberately runs exactly one condition and one seed.  Full runs are
CUDA-only, require an exact protocol confirmation, and refuse uncommitted or
untracked study sources.  Smoke runs use the same training/evaluation path on
two tasks truncated to 1,000 train and 1,000 test examples per task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# Required by deterministic CUDA matrix multiplication.  It must be set before
# the first CUDA operation in this process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PC_LIBRARY_ROOT = SCRIPT_DIR.parent / "predictive-coding"
PROTOCOL_PATH = (
    REPO_ROOT
    / ".research/studies/pc-classp-matched-lr-paper-v1/protocol.md"
)
sys.path.insert(0, str(PC_LIBRARY_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import predictive_coding as pc
from classp_optimizer import CLASSP


PROTOCOL_ID = "matched-lr-paper-v1"
SCHEMA_VERSION = 1
INPUT_SIZE = 28 * 28
OUTPUT_SIZE = 10
PERMUTATION_SEED = 0
PRIMARY_SEEDS = (7, 123, 2026, 31415, 271828)
LEGACY_BRIDGE_SEED = 42
ALL_FULL_SEEDS = PRIMARY_SEEDS + (LEGACY_BRIDGE_SEED,)
CONDITIONS = {
    "bp-adam": {"method": "bp", "optimizer": "adam"},
    "bp-classp": {"method": "bp", "optimizer": "classp"},
    "pc-adam": {"method": "pc", "optimizer": "adam"},
    "pc-classp": {"method": "pc", "optimizer": "classp"},
}
COMMON_CONFIG = {
    "epochs_per_task": 1,
    "parameter_learning_rate": 3e-4,
    "batch_size": 500,
    "evaluation_batch_size": 1000,
    "hidden_size": 256,
    "pc_inference_steps": 20,
    "pc_latent_learning_rate": 0.01,
    "classp_p": 2.0,
    "classp_threshold": 1e-5,
    "classp_epsilon": 1e-5,
    "classp_apply_decay": True,
}
MODE_CONFIG = {
    "smoke": {
        "n_tasks": 2,
        "train_samples_per_task": 1000,
        "test_samples_per_task": 1000,
    },
    "full": {
        "n_tasks": 20,
        "train_samples_per_task": None,
        "test_samples_per_task": None,
    },
}


class FixedOrderSampler(Sampler[int]):
    """Yield a precomputed order without consulting global RNG state."""

    def __init__(self, indices: Sequence[int] | torch.Tensor):
        self.indices = [int(index) for index in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class PermutedDataset(Dataset):
    def __init__(self, dataset: Dataset, permutation: torch.Tensor):
        self.dataset = dataset
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image[self.permutation], label


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_checksum(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def combined_checksum(items: Iterable[str]) -> str:
    return sha256_bytes("\n".join(items).encode("utf-8"))


def run_command(command: Sequence[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(command), cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def git_metadata(path: Path) -> dict[str, Any]:
    try:
        commit = run_command(["git", "rev-parse", "HEAD"], cwd=path)
        tree = run_command(["git", "rev-parse", "HEAD^{tree}"], cwd=path)
        status = run_command(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=path
        )
        return {
            "commit": commit,
            "tree": tree,
            "tracked_dirty": bool(status),
            "tracked_status": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "commit": "unknown",
            "tree": "unknown",
            "tracked_dirty": True,
            "tracked_status": [f"metadata-error: {type(error).__name__}"],
        }


def source_provenance() -> dict[str, Any]:
    files = [Path(__file__).resolve(), SCRIPT_DIR / "classp_optimizer.py"]
    if PROTOCOL_PATH.exists():
        files.append(PROTOCOL_PATH)
    return {
        "repository": git_metadata(REPO_ROOT),
        "predictive_coding_submodule": git_metadata(PC_LIBRARY_ROOT),
        "files": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in files
        },
        "classp_paper": {
            "citation": "Ludwig (2024), arXiv:2405.09637",
            "local_pdf_sha256": "0e528058503ba9424969661bccf13df9150f709bd0a073c5cc4bdc98b80e8e56",
        },
        "official_classp_code": {
            "repository": "https://github.com/oswaldoludwig/CLASSP",
            "commit": "ea3fe3c67279e27db8edea73d5f3ad522bb15dc1",
            "classp_py_blob": "2bdf432b3f8bfdc4fc249cc6afadb9f150313032",
            "normative": False,
        },
    }


def environment_provenance(device: torch.device) -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        try:
            nvidia_smi = run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ]
            )
        except (OSError, subprocess.CalledProcessError):
            nvidia_smi = "unavailable"
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "nvidia_smi": nvidia_smi,
        }
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "requested_device": str(device),
        "gpu": gpu,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def assert_full_source_ready(source: dict[str, Any]) -> None:
    required_tracked = [
        Path(__file__).resolve(),
        SCRIPT_DIR / "classp_optimizer.py",
        PROTOCOL_PATH,
    ]
    untracked: list[str] = []
    for path in required_tracked:
        relative = path.relative_to(REPO_ROOT)
        completed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            untracked.append(str(relative))
    if untracked:
        raise RuntimeError(
            "full mode requires committed study sources; untracked: "
            + ", ".join(untracked)
        )
    if source["repository"]["tracked_dirty"]:
        raise RuntimeError("full mode requires a clean tracked repository revision")
    if source["predictive_coding_submodule"]["tracked_dirty"]:
        raise RuntimeError("full mode requires a clean predictive-coding submodule")


def assert_safe_staging_directory(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(relative)],
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "run outputs inside the repository must use an ignored staging "
            "directory (recommended: playground/continual-mnist/run-staging/)"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_permutations(n_tasks: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(PERMUTATION_SEED)
    return [torch.arange(INPUT_SIZE, dtype=torch.int64)] + [
        torch.randperm(INPUT_SIZE, generator=generator)
        for _ in range(n_tasks - 1)
    ]


def load_permuted_mnist(n_tasks: int, data_root: Path):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(torch.flatten)]
    )
    try:
        train_base = datasets.MNIST(
            str(data_root), train=True, download=False, transform=transform
        )
        test_base = datasets.MNIST(
            str(data_root), train=False, download=False, transform=transform
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"MNIST is not available at {data_root}; prepare it before running"
        ) from error
    permutations = make_permutations(n_tasks)
    return (
        [PermutedDataset(train_base, value) for value in permutations],
        [PermutedDataset(test_base, value) for value in permutations],
        permutations,
        dataset_file_hashes(train_base),
    )


def dataset_file_hashes(dataset: datasets.MNIST) -> dict[str, str]:
    raw_folder = Path(dataset.raw_folder)
    return {
        str(path.relative_to(raw_folder.parent)): sha256_file(path)
        for path in sorted(raw_folder.glob("*"))
        if path.is_file()
    }


def sample_order(
    dataset_size: int,
    seed: int,
    task_id: int,
    limit: int | None,
) -> torch.Tensor:
    order_seed = ((seed + 1) * 1_000_003 + (task_id + 1) * 97_003) % (2**63 - 1)
    generator = torch.Generator().manual_seed(order_seed)
    order = torch.randperm(dataset_size, generator=generator)
    if limit is not None:
        if not 0 < limit <= dataset_size:
            raise ValueError(f"invalid sample limit {limit} for size {dataset_size}")
        order = order[:limit]
    return order


def make_bp_model(hidden_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, OUTPUT_SIZE),
    )


def make_pc_model(hidden_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, hidden_size),
        pc.PCLayer(),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        pc.PCLayer(),
        nn.ReLU(),
        nn.Linear(hidden_size, OUTPUT_SIZE),
    )


def linear_layers(model: nn.Module) -> list[nn.Linear]:
    return [module for module in model.modules() if isinstance(module, nn.Linear)]


def linear_parameter_checksums(model: nn.Module) -> list[dict[str, str]]:
    return [
        {
            "weight": tensor_checksum(layer.weight),
            "bias": tensor_checksum(layer.bias),
        }
        for layer in linear_layers(model)
    ]


def linear_model_checksum(model: nn.Module) -> str:
    checksums = linear_parameter_checksums(model)
    return sha256_bytes(canonical_json_bytes(checksums))


def make_matched_models(hidden_size: int, seed: int) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    # BP initialization is the canonical tensor draw.  PC is independently
    # constructed and then receives exact copies, avoiding assumptions about
    # whether PCLayer construction consumes random numbers.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        bp_model = make_bp_model(hidden_size)
        torch.manual_seed(seed ^ 0x5A17)
        pc_model = make_pc_model(hidden_size)

    bp_layers = linear_layers(bp_model)
    pc_layers = linear_layers(pc_model)
    if len(bp_layers) != len(pc_layers):
        raise RuntimeError("BP and PC have different numbers of Linear layers")
    with torch.no_grad():
        for bp_layer, pc_layer in zip(bp_layers, pc_layers):
            pc_layer.weight.copy_(bp_layer.weight)
            pc_layer.bias.copy_(bp_layer.bias)

    bp_checksums = linear_parameter_checksums(bp_model)
    pc_checksums = linear_parameter_checksums(pc_model)
    matched = bp_checksums == pc_checksums
    if not matched:
        raise RuntimeError("actual BP and PC Linear tensors do not match")
    audit = {
        "initialization_seed": seed,
        "linear_layers": len(bp_layers),
        "all_linear_tensors_matched": matched,
        "bp_linear_checksums": bp_checksums,
        "pc_linear_checksums": pc_checksums,
        "paired_initialization_checksum": sha256_bytes(
            canonical_json_bytes(bp_checksums)
        ),
    }
    return bp_model, pc_model, audit


def half_squared_error(output: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (output - _target).square().sum()


@torch.no_grad()
def predict_batch(model: nn.Module, data: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    output = model(data)
    if was_training:
        model.train()
    if not torch.isfinite(output).all():
        raise FloatingPointError("model produced non-finite predictions")
    return output


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    limit: int | None,
) -> float:
    was_training = model.training
    model.eval()
    sample_count = len(dataset) if limit is None else limit
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=FixedOrderSampler(range(sample_count)),
        num_workers=0,
    )
    correct = 0
    total = 0
    for data, label in loader:
        output = model(data.to(device))
        if not torch.isfinite(output).all():
            raise FloatingPointError("evaluation produced non-finite predictions")
        label = label.to(device)
        correct += int((output.argmax(dim=-1) == label).sum().item())
        total += int(label.numel())
    if was_training:
        model.train()
    return correct / total


def gradient_l2(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("training produced a non-finite gradient")
        total += float(parameter.grad.detach().double().square().sum().item())
    return math.sqrt(total)


def optimizer_state_summary(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    entries: list[str] = []
    steps: list[int] = []
    accumulator_sum = 0.0
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            pieces = [f"{group_index}:{parameter_index}"]
            for key in sorted(state):
                value = state[key]
                if torch.is_tensor(value):
                    if not torch.isfinite(value).all():
                        raise FloatingPointError("optimizer state is non-finite")
                    pieces.append(f"{key}={tensor_checksum(value)}")
                    if key == "step":
                        steps.append(int(value.item()))
                    if key == "grad_sum":
                        accumulator_sum += float(value.double().sum().item())
                else:
                    pieces.append(f"{key}={value}")
                    if key == "step":
                        steps.append(int(value))
            entries.append("|".join(pieces))
    return {
        "state_checksum": combined_checksum(entries),
        "parameters_with_state": len(optimizer.state),
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "classp_accumulator_sum": accumulator_sum
        if isinstance(optimizer, CLASSP)
        else None,
    }


def compute_continual_metrics(
    accuracy_matrix: Sequence[Sequence[float]],
) -> dict[str, Any]:
    matrix = np.asarray(accuracy_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] + 1:
        raise ValueError("accuracy matrix must have shape (N+1) x N")
    n_tasks = matrix.shape[1]
    if n_tasks < 2:
        raise ValueError("continual metrics require at least two tasks")

    pre_task = np.asarray([matrix[task, task] for task in range(n_tasks)])
    post_task = np.asarray([matrix[task + 1, task] for task in range(n_tasks)])
    adaptation = post_task - pre_task
    final = matrix[-1, :]
    bwt = np.asarray(
        [final[task] - post_task[task] for task in range(n_tasks - 1)]
    )
    forgetting = np.asarray(
        [
            np.max(matrix[task + 1 :, task]) - final[task]
            for task in range(n_tasks - 1)
        ]
    )
    return {
        "final_accuracy_by_task": final.tolist(),
        "final_average_accuracy": float(final.mean()),
        "pre_task_diagonal_accuracy": pre_task.tolist(),
        "post_task_diagonal_accuracy": post_task.tolist(),
        "adaptation_by_task": adaptation.tolist(),
        "mean_adaptation": float(adaptation.mean()),
        "bwt_by_task_0_to_n_minus_2": bwt.tolist(),
        "mean_bwt": float(bwt.mean()),
        "forgetting_by_task_0_to_n_minus_2": forgetting.tolist(),
        "mean_forgetting": float(forgetting.mean()),
    }


def finite_numbers(value: Any, path: str = "root") -> list[str]:
    failures: list[str] = []
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return failures
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            failures.append(path)
        return failures
    if isinstance(value, dict):
        for key, child in value.items():
            failures.extend(finite_numbers(child, f"{path}.{key}"))
        return failures
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            failures.extend(finite_numbers(child, f"{path}[{index}]"))
    return failures


def run_condition(
    *,
    condition_id: str,
    seed: int,
    mode: str,
    device: torch.device,
    train_tasks: Sequence[Dataset],
    test_tasks: Sequence[Dataset],
    permutations: Sequence[torch.Tensor],
    diagnostics: bool,
    dataset_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if condition_id not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition_id}")
    if mode not in MODE_CONFIG:
        raise ValueError(f"unknown mode: {mode}")
    mode_config = dict(MODE_CONFIG[mode])
    n_tasks = mode_config["n_tasks"]
    if len(train_tasks) != n_tasks or len(test_tasks) != n_tasks:
        raise ValueError("task count does not match mode")
    if len(permutations) != n_tasks:
        raise ValueError("permutation count does not match mode")

    seed_everything(seed)
    condition = CONDITIONS[condition_id]
    bp_model, pc_model, initialization = make_matched_models(
        COMMON_CONFIG["hidden_size"], seed
    )
    if condition["method"] == "bp":
        model = bp_model.to(device)
        del pc_model
    else:
        model = pc_model.to(device)
        del bp_model
    model.train()

    if condition["optimizer"] == "classp":
        optimizer: torch.optim.Optimizer = CLASSP(
            model.parameters(),
            lr=COMMON_CONFIG["parameter_learning_rate"],
            p=COMMON_CONFIG["classp_p"],
            threshold=COMMON_CONFIG["classp_threshold"],
            eps=COMMON_CONFIG["classp_epsilon"],
            apply_decay=COMMON_CONFIG["classp_apply_decay"],
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=COMMON_CONFIG["parameter_learning_rate"]
        )

    trainer = None
    trainer_optimizer_identity_match = None
    if condition["method"] == "pc":
        trainer = pc.PCTrainer(
            model,
            T=COMMON_CONFIG["pc_inference_steps"],
            optimizer_x_fn=optim.SGD,
            optimizer_x_kwargs={"lr": COMMON_CONFIG["pc_latent_learning_rate"]},
            update_x_at="all",
            update_p_at="last",
            manual_optimizer_p_fn=lambda: optimizer,
        )
        trainer_optimizer_identity_match = trainer.get_optimizer_p() is optimizer
        if not trainer_optimizer_identity_match:
            raise RuntimeError("PCTrainer did not retain the supplied optimizer")

    train_orders = [
        sample_order(
            len(train_tasks[task]),
            seed,
            task,
            mode_config["train_samples_per_task"],
        )
        for task in range(n_tasks)
    ]
    permutation_checksums = [tensor_checksum(value) for value in permutations]
    train_order_checksums = [tensor_checksum(value) for value in train_orders]
    task_stream_checksums = [
        combined_checksum([permutation_checksums[index], train_order_checksums[index]])
        for index in range(n_tasks)
    ]

    start_time = time.perf_counter()
    started_at = utc_now()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    accuracy_matrix = np.zeros((n_tasks + 1, n_tasks), dtype=np.float64)
    for task in range(n_tasks):
        accuracy_matrix[0, task] = evaluate(
            model,
            test_tasks[task],
            device,
            COMMON_CONFIG["evaluation_batch_size"],
            mode_config["test_samples_per_task"],
        )

    online_batch_accuracy: list[list[float]] = []
    online_correct_by_task: list[int] = []
    online_samples_by_task: list[int] = []
    diagnostic_gradient_l2: list[list[float]] = []
    optimizer_states: list[dict[str, Any]] = []
    samples_processed_by_task: list[int] = []
    updates_by_task: list[int] = []

    for task_id in range(n_tasks):
        loader = DataLoader(
            train_tasks[task_id],
            batch_size=COMMON_CONFIG["batch_size"],
            sampler=FixedOrderSampler(train_orders[task_id]),
            num_workers=0,
        )
        task_batch_accuracy: list[float] = []
        task_gradient_l2: list[float] = []
        task_correct = 0
        task_samples = 0
        task_updates = 0

        for data, label in loader:
            data = data.to(device)
            label = label.to(device)
            prediction = predict_batch(model, data)
            batch_correct = int((prediction.argmax(dim=-1) == label).sum().item())
            task_batch_accuracy.append(batch_correct / int(label.numel()))
            task_correct += batch_correct
            task_samples += int(label.numel())

            target = F.one_hot(label, num_classes=OUTPUT_SIZE).float()
            model.train()
            if condition["method"] == "pc":
                assert trainer is not None
                progress = trainer.train_on_batch(
                    inputs=data,
                    loss_fn=half_squared_error,
                    loss_fn_kwargs={"_target": target},
                    is_reset_optimizer_x_at_batch_start=True,
                    is_reset_optimizer_p_at_batch_start=False,
                )
                if finite_numbers(progress):
                    raise FloatingPointError("PC training progress is non-finite")
                current_gradient_l2 = gradient_l2(model)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = half_squared_error(model(data), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("BP loss is non-finite")
                loss.backward()
                current_gradient_l2 = gradient_l2(model)
                optimizer.step()

            if diagnostics:
                task_gradient_l2.append(current_gradient_l2)
            task_updates += 1

        online_batch_accuracy.append(task_batch_accuracy)
        online_correct_by_task.append(task_correct)
        online_samples_by_task.append(task_samples)
        diagnostic_gradient_l2.append(task_gradient_l2)
        samples_processed_by_task.append(task_samples)
        updates_by_task.append(task_updates)
        optimizer_states.append(optimizer_state_summary(optimizer))

        for evaluation_task in range(n_tasks):
            accuracy_matrix[task_id + 1, evaluation_task] = evaluate(
                model,
                test_tasks[evaluation_task],
                device,
                COMMON_CONFIG["evaluation_batch_size"],
                mode_config["test_samples_per_task"],
            )

    metrics = compute_continual_metrics(accuracy_matrix.tolist())
    total_online_correct = sum(online_correct_by_task)
    total_online_samples = sum(online_samples_by_task)
    metrics["online_predict_before_update_accuracy"] = (
        total_online_correct / total_online_samples
    )
    metrics["online_predict_before_update_accuracy_by_task"] = [
        correct / count
        for correct, count in zip(online_correct_by_task, online_samples_by_task)
    ]

    expected_samples = sum(len(order) for order in train_orders)
    expected_updates = sum(
        math.ceil(len(order) / COMMON_CONFIG["batch_size"])
        for order in train_orders
    )
    actual_samples = sum(samples_processed_by_task)
    actual_updates = sum(updates_by_task)
    if actual_samples != expected_samples or actual_updates != expected_updates:
        raise RuntimeError("sample/update count invariant failed")
    for task_index, summary in enumerate(optimizer_states):
        expected_cumulative = sum(updates_by_task[: task_index + 1])
        if summary["step_min"] != expected_cumulative or summary["step_max"] != expected_cumulative:
            raise RuntimeError(
                f"optimizer persistence invariant failed after task {task_index}"
            )

    elapsed = time.perf_counter() - start_time
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    final_model_checksum = linear_model_checksum(model)
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "validated-cell",
        "mode": mode,
        "condition_id": condition_id,
        "cohort": (
            "smoke"
            if mode == "smoke"
            else "legacy-bridge"
            if seed == LEGACY_BRIDGE_SEED
            else "primary-held-out"
        ),
        "config": {
            **COMMON_CONFIG,
            **mode_config,
            **condition,
            "condition_id": condition_id,
            "seed": seed,
            "permutation_seed": PERMUTATION_SEED,
            "loss": "one-hot half-squared error, summed over batch and classes",
            "pc_update_p_at": "last" if condition["method"] == "pc" else None,
            "diagnostics": diagnostics,
            "requested_device": device.type,
        },
        "data": {
            "dataset": "torchvision MNIST default train/test split",
            "dataset_file_sha256": dataset_hashes or {},
            "permutation_checksums": permutation_checksums,
            "combined_permutation_checksum": combined_checksum(permutation_checksums),
            "train_order_checksums": train_order_checksums,
            "combined_train_order_checksum": combined_checksum(train_order_checksums),
            "task_stream_checksums": task_stream_checksums,
            "train_samples_by_task": [len(value) for value in train_orders],
            "test_samples_by_task": [
                len(dataset)
                if mode_config["test_samples_per_task"] is None
                else mode_config["test_samples_per_task"]
                for dataset in test_tasks
            ],
        },
        "initialization": initialization,
        "optimizer_audit": {
            "instances_created": 1,
            "persistent_across_task_stream": True,
            "pc_trainer_optimizer_identity_match": trainer_optimizer_identity_match,
            "state_after_each_task": optimizer_states,
        },
        "counts": {
            "expected_samples": expected_samples,
            "actual_samples": actual_samples,
            "samples_by_task": samples_processed_by_task,
            "expected_optimizer_updates": expected_updates,
            "actual_optimizer_updates": actual_updates,
            "optimizer_updates_by_task": updates_by_task,
        },
        "online_predict_before_update": {
            "batch_accuracy_by_task": online_batch_accuracy,
            "correct_by_task": online_correct_by_task,
            "samples_by_task": online_samples_by_task,
        },
        "accuracy_matrix": accuracy_matrix.tolist(),
        "metrics": metrics,
        "diagnostics": {
            "enabled": diagnostics,
            "gradient_l2_by_task": diagnostic_gradient_l2 if diagnostics else None,
        },
        "final_linear_parameter_checksum": final_model_checksum,
        "timing": {
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "elapsed_seconds": elapsed,
        },
        "memory": {"peak_cuda_bytes": peak_memory},
    }
    nonfinite = finite_numbers(
        {
            "accuracy_matrix": result["accuracy_matrix"],
            "metrics": result["metrics"],
            "online": result["online_predict_before_update"],
            "diagnostics": result["diagnostics"],
        }
    )
    if nonfinite:
        raise FloatingPointError(f"non-finite result fields: {nonfinite}")
    return result


def seal_result(result: dict[str, Any]) -> dict[str, Any]:
    if "integrity" in result:
        raise ValueError("result is already sealed")
    sealed = dict(result)
    sealed["integrity"] = {
        "canonical_payload_sha256": sha256_bytes(canonical_json_bytes(result)),
        "definition": "SHA-256 of canonical JSON with the integrity object omitted",
    }
    return sealed


def validate_result(result: dict[str, Any]) -> None:
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected schema version")
    if result.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected protocol")
    condition_id = result.get("condition_id")
    if condition_id not in CONDITIONS:
        raise ValueError("unexpected condition")
    config = result["config"]
    n_tasks = config["n_tasks"]
    matrix = np.asarray(result["accuracy_matrix"], dtype=np.float64)
    if matrix.shape != (n_tasks + 1, n_tasks):
        raise ValueError("incorrect accuracy matrix shape")
    recomputed = compute_continual_metrics(result["accuracy_matrix"])
    for key, value in recomputed.items():
        if not np.allclose(value, result["metrics"][key], atol=1e-12):
            raise ValueError(f"metric mismatch: {key}")
    if result["counts"]["expected_samples"] != result["counts"]["actual_samples"]:
        raise ValueError("sample count mismatch")
    if (
        result["counts"]["expected_optimizer_updates"]
        != result["counts"]["actual_optimizer_updates"]
    ):
        raise ValueError("optimizer update count mismatch")
    if not result["initialization"]["all_linear_tensors_matched"]:
        raise ValueError("initialization mismatch")
    if result["optimizer_audit"]["instances_created"] != 1:
        raise ValueError("optimizer instance count mismatch")
    if not result["optimizer_audit"]["persistent_across_task_stream"]:
        raise ValueError("optimizer was not persistent")
    nonfinite = finite_numbers(
        {
            "accuracy_matrix": result["accuracy_matrix"],
            "metrics": result["metrics"],
            "online": result["online_predict_before_update"],
        }
    )
    if nonfinite:
        raise ValueError(f"non-finite scientific values: {nonfinite}")
    integrity = result.get("integrity")
    if integrity is not None:
        payload = dict(result)
        payload.pop("integrity")
        expected = sha256_bytes(canonical_json_bytes(payload))
        if integrity.get("canonical_payload_sha256") != expected:
            raise ValueError("canonical payload hash mismatch")


def output_paths(output_dir: Path, mode: str, condition: str, seed: int) -> dict[str, Path]:
    stem = f"{PROTOCOL_ID}__{mode}__{condition}__seed-{seed}"
    return {
        "result": output_dir / f"{stem}.json",
        "receipt": output_dir / f"{stem}.receipt.json",
        "failure": output_dir / f"{stem}.failure.json",
    }


def atomic_write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
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


def ensure_unique_paths(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing artifacts: " + ", ".join(existing))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=sorted(MODE_CONFIG))
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confirm-protocol")
    parser.add_argument("--diagnostics", choices=("on", "off"), default="on")
    return parser.parse_args(argv)


def validate_cli_protocol(args: argparse.Namespace) -> None:
    if args.mode == "smoke" and args.seed != LEGACY_BRIDGE_SEED:
        raise ValueError(f"smoke mode is frozen to seed {LEGACY_BRIDGE_SEED}")
    if args.mode == "full":
        if args.confirm_protocol != PROTOCOL_ID:
            raise ValueError(
                f"full mode requires --confirm-protocol {PROTOCOL_ID}"
            )
        if args.device != "cuda":
            raise ValueError("full mode requires --device cuda")
        if args.seed not in ALL_FULL_SEEDS:
            raise ValueError(f"full mode seed must be one of {ALL_FULL_SEEDS}")
        if args.diagnostics != "on":
            raise ValueError("full mode requires --diagnostics on")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_dir, args.mode, args.condition, args.seed)
    ensure_unique_paths(paths)
    started_at = utc_now()
    source: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    try:
        assert_safe_staging_directory(args.output_dir)
        validate_cli_protocol(args)
        device = torch.device(args.device)
        source = source_provenance()
        if args.mode == "full":
            assert_full_source_ready(source)
        seed_everything(args.seed)
        environment = environment_provenance(device)
        train_tasks, test_tasks, permutations, dataset_hashes = load_permuted_mnist(
            MODE_CONFIG[args.mode]["n_tasks"], REPO_ROOT / "data"
        )
        result = run_condition(
            condition_id=args.condition,
            seed=args.seed,
            mode=args.mode,
            device=device,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            permutations=permutations,
            diagnostics=args.diagnostics == "on",
            dataset_hashes=dataset_hashes,
        )
        result["source"] = source
        result["environment"] = environment
        result = seal_result(result)
        validate_result(result)
        atomic_write_json_exclusive(paths["result"], result)
        result_sha256 = sha256_file(paths["result"])
        receipt = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "artifact": paths["result"].name,
            "artifact_sha256": result_sha256,
            "artifact_bytes": paths["result"].stat().st_size,
            "created_at_utc": utc_now(),
        }
        atomic_write_json_exclusive(paths["receipt"], receipt)
        print(
            json.dumps(
                {
                    "status": "validated-cell",
                    "condition": args.condition,
                    "seed": args.seed,
                    "mode": args.mode,
                    "device": args.device,
                    "final_average_accuracy": result["metrics"]["final_average_accuracy"],
                    "result": str(paths["result"]),
                    "sha256": result_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "failed-cell",
            "mode": args.mode,
            "condition_id": args.condition,
            "seed": args.seed,
            "device": args.device,
            "diagnostics": args.diagnostics,
            "started_at_utc": started_at,
            "failed_at_utc": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "source": source,
            "environment": environment,
        }
        try:
            atomic_write_json_exclusive(paths["failure"], failure)
        except BaseException as write_error:
            print(
                f"failed to preserve failure artifact {paths['failure']}: {write_error}",
                file=sys.stderr,
            )
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
