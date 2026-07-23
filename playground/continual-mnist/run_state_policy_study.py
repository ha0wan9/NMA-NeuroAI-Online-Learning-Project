#!/usr/bin/env python3
"""Run one cell of the matched-LR parameter-optimizer state-policy study.

The completed ``matched-lr-paper-v1`` runner is imported as a read-only source
of the frozen stream, model, objective, metric, and hashing implementations.
This file owns the new state-policy manipulation and artifact schema.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / ".research/studies/pc-classp-matched-lr-state-policy-v1/protocol.md"
)
sys.path.insert(0, str(SCRIPT_DIR))

import run_matched_lr_validation as frozen
from classp_optimizer import CLASSP


PROTOCOL_ID = "matched-lr-state-policy-v1"
SCHEMA_VERSION = 1
PRIMARY_SEEDS = (7, 123, 2026, 31415, 271828)
BRIDGE_SEED = 42
ALL_FULL_SEEDS = PRIMARY_SEEDS + (BRIDGE_SEED,)
INPUT_SIZE = frozen.INPUT_SIZE
OUTPUT_SIZE = frozen.OUTPUT_SIZE
PERMUTATION_SEED = frozen.PERMUTATION_SEED
COMMON_CONFIG = dict(frozen.COMMON_CONFIG)
MODE_CONFIG = {key: dict(value) for key, value in frozen.MODE_CONFIG.items()}
CONDITIONS = {
    "bp-adam-persistent": {
        "method": "bp",
        "optimizer": "adam",
        "state_policy": "persistent",
    },
    "bp-adam-task-reset": {
        "method": "bp",
        "optimizer": "adam",
        "state_policy": "task-reset",
    },
    "bp-classp-persistent": {
        "method": "bp",
        "optimizer": "classp",
        "state_policy": "persistent",
    },
    "bp-classp-task-reset": {
        "method": "bp",
        "optimizer": "classp",
        "state_policy": "task-reset",
    },
    "pc-adam-persistent": {
        "method": "pc",
        "optimizer": "adam",
        "state_policy": "persistent",
    },
    "pc-adam-task-reset": {
        "method": "pc",
        "optimizer": "adam",
        "state_policy": "task-reset",
    },
    "pc-classp-persistent": {
        "method": "pc",
        "optimizer": "classp",
        "state_policy": "persistent",
    },
    "pc-classp-task-reset": {
        "method": "pc",
        "optimizer": "classp",
        "state_policy": "task-reset",
    },
}
SOURCE_FILES = (
    Path(__file__).resolve(),
    SCRIPT_DIR / "validate_state_policy_results.py",
    SCRIPT_DIR / "run_state_policy_protocol.py",
    SCRIPT_DIR / "run_matched_lr_validation.py",
    SCRIPT_DIR / "classp_optimizer.py",
    PROTOCOL_PATH,
    REPO_ROOT / "uv.lock",
)

# Re-export frozen helpers used by tests and the validator.
FixedOrderSampler = frozen.FixedOrderSampler
PermutedDataset = frozen.PermutedDataset
atomic_write_json_exclusive = frozen.atomic_write_json_exclusive
canonical_json_bytes = frozen.canonical_json_bytes
combined_checksum = frozen.combined_checksum
compute_continual_metrics = frozen.compute_continual_metrics
finite_numbers = frozen.finite_numbers
linear_model_checksum = frozen.linear_model_checksum
linear_parameter_checksums = frozen.linear_parameter_checksums
make_matched_models = frozen.make_matched_models
make_permutations = frozen.make_permutations
sample_order = frozen.sample_order
seal_result = frozen.seal_result
sha256_bytes = frozen.sha256_bytes
sha256_file = frozen.sha256_file
tensor_checksum = frozen.tensor_checksum
utc_now = frozen.utc_now


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Hash a manifest with its self-referential hash field omitted."""
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return sha256_bytes(canonical_json_bytes(payload))


def load_run_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("unexpected run-manifest schema version")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("run manifest has the wrong protocol")
    if manifest.get("manifest_hash") != manifest_hash(manifest):
        raise ValueError("run-manifest hash mismatch")
    if tuple(manifest.get("conditions", ())) != tuple(CONDITIONS):
        raise ValueError("run manifest has the wrong condition set or order")
    if manifest.get("kind") != "single-gpu-local":
        raise ValueError("run manifest is not a single-GPU local manifest")
    if manifest.get("status") != "frozen":
        raise ValueError("run manifest is not frozen")
    seeds = manifest.get("seeds", {})
    if seeds.get("bridge") != BRIDGE_SEED:
        raise ValueError("run manifest has the wrong bridge seed")
    if tuple(seeds.get("held_out", ())) != PRIMARY_SEEDS:
        raise ValueError("run manifest has the wrong held-out seeds")
    orders = manifest.get("condition_order_by_seed", {})
    if set(orders) != {str(seed) for seed in (BRIDGE_SEED, *PRIMARY_SEEDS)}:
        raise ValueError("run manifest has the wrong condition-order seed set")
    if any(sorted(order) != sorted(CONDITIONS) for order in orders.values()):
        raise ValueError("run manifest contains an invalid condition order")
    return manifest


def git_metadata(path: Path) -> dict[str, Any]:
    return frozen.git_metadata(path)


def source_provenance() -> dict[str, Any]:
    files = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in SOURCE_FILES
        if path.exists()
    }
    return {
        "repository": git_metadata(REPO_ROOT),
        "predictive_coding_submodule": git_metadata(frozen.PC_LIBRARY_ROOT),
        "files": files,
        "dependency_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
        "parent_protocol": {
            "protocol_id": frozen.PROTOCOL_ID,
            "runner_sha256": sha256_file(SCRIPT_DIR / "run_matched_lr_validation.py"),
        },
    }


def source_identity(source: dict[str, Any]) -> str:
    value = {
        "repository_commit": source["repository"]["commit"],
        "repository_tree": source["repository"]["tree"],
        "repository_tracked_dirty": source["repository"]["tracked_dirty"],
        "pc_commit": source["predictive_coding_submodule"]["commit"],
        "pc_tree": source["predictive_coding_submodule"]["tree"],
        "pc_tracked_dirty": source["predictive_coding_submodule"]["tracked_dirty"],
        "files": source["files"],
        "dependency_lock_sha256": source["dependency_lock_sha256"],
    }
    return sha256_bytes(canonical_json_bytes(value))


def assert_source_ready(source: dict[str, Any]) -> None:
    untracked: list[str] = []
    for path in SOURCE_FILES:
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
            "study execution requires committed sources; untracked: "
            + ", ".join(untracked)
        )
    if source["repository"]["tracked_dirty"]:
        raise RuntimeError("study execution requires a clean tracked repository")
    if source["predictive_coding_submodule"]["tracked_dirty"]:
        raise RuntimeError("study execution requires a clean predictive-coding submodule")


def dependency_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def environment_provenance(device: torch.device) -> dict[str, Any]:
    environment = frozen.environment_provenance(device)
    environment["dependencies"] = dependency_identity()
    if device.type == "cuda":
        try:
            line = frozen.run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ]
            )
            name, uuid, driver, memory_mib = [
                part.strip() for part in line.split(",", maxsplit=3)
            ]
        except (OSError, subprocess.CalledProcessError, ValueError):
            properties = torch.cuda.get_device_properties(device)
            name = properties.name
            uuid = "unavailable"
            driver = "unavailable"
            memory_mib = str(properties.total_memory // (1024 * 1024))
        properties = torch.cuda.get_device_properties(device)
        environment["gpu_identity"] = {
            "name": name,
            "uuid": uuid,
            "compute_capability": [properties.major, properties.minor],
        }
        environment["gpu_driver"] = driver
        environment["gpu_total_memory_mib"] = int(memory_mib)
    else:
        environment["gpu_identity"] = None
    return environment


def host_identity(environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "hostname": environment["hostname"],
        "gpu_identity": environment["gpu_identity"],
        "gpu_driver": environment.get("gpu_driver"),
        "cuda_runtime": environment["dependencies"]["cuda_runtime"],
    }


def validate_local_host(
    manifest: dict[str, Any], environment: dict[str, Any]
) -> dict[str, Any]:
    actual = host_identity(environment)
    expected = manifest.get("host")
    if actual != expected:
        raise ValueError("current host/GPU does not match the frozen local manifest")
    return actual


def validate_manifest_for_cell(
    manifest: dict[str, Any],
    *,
    mode: str,
    condition: str,
    seed: int,
    source: dict[str, Any],
    environment: dict[str, Any],
    dataset_hashes: dict[str, str],
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if manifest.get("source_identity") != source_identity(source):
        raise ValueError("run-manifest source identity does not match checkout")
    if manifest.get("dependency_identity") != environment["dependencies"]:
        raise ValueError("run-manifest dependency identity does not match host")
    if manifest.get("dataset_file_sha256") != dataset_hashes:
        raise ValueError("run-manifest MNIST hashes do not match local data")
    if mode == "full" and seed not in ALL_FULL_SEEDS:
        raise ValueError(f"full mode seed must be one of {ALL_FULL_SEEDS}")
    if mode == "smoke" and seed != BRIDGE_SEED:
        raise ValueError(f"smoke mode is frozen to seed {BRIDGE_SEED}")
    order = manifest["condition_order_by_seed"].get(str(seed), [])
    if sorted(order) != sorted(CONDITIONS) or condition not in order:
        raise ValueError(f"manifest condition order is invalid for seed {seed}")
    return validate_local_host(manifest, environment)


def make_optimizer(
    model: torch.nn.Module, optimizer_name: str
) -> torch.optim.Optimizer:
    if optimizer_name == "classp":
        return CLASSP(
            model.parameters(),
            lr=COMMON_CONFIG["parameter_learning_rate"],
            p=COMMON_CONFIG["classp_p"],
            threshold=COMMON_CONFIG["classp_threshold"],
            eps=COMMON_CONFIG["classp_epsilon"],
            apply_decay=COMMON_CONFIG["classp_apply_decay"],
        )
    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(), lr=COMMON_CONFIG["parameter_learning_rate"]
        )
    raise ValueError(f"unknown optimizer: {optimizer_name}")


def parameter_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    """Clone trainable network parameters, excluding transient PC latents."""
    with torch.no_grad():
        return [
            parameter.detach().clone()
            for layer in frozen.linear_layers(model)
            for parameter in layer.parameters()
        ]


def parameter_displacement(
    before: Sequence[torch.Tensor], model: torch.nn.Module
) -> dict[str, float]:
    """Return absolute and relative L2 movement from ``before``."""
    current_parameters = [
        parameter
        for layer in frozen.linear_layers(model)
        for parameter in layer.parameters()
    ]
    if len(before) != len(current_parameters):
        raise ValueError("parameter snapshot length mismatch")
    delta_squared = 0.0
    before_squared = 0.0
    after_squared = 0.0
    with torch.no_grad():
        for saved, current in zip(before, current_parameters):
            saved_double = saved.detach().double()
            current_double = current.detach().double()
            delta_squared += float(
                (current_double - saved_double).square().sum().item()
            )
            before_squared += float(saved_double.square().sum().item())
            after_squared += float(current_double.square().sum().item())
    displacement_l2 = math.sqrt(delta_squared)
    parameter_l2_before = math.sqrt(before_squared)
    return {
        "l2": displacement_l2,
        "relative_l2": (
            displacement_l2 / parameter_l2_before
            if parameter_l2_before
            else 0.0
        ),
        "parameter_l2_before": parameter_l2_before,
        "parameter_l2_after": math.sqrt(after_squared),
    }


def optimizer_state_diagnostics(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Aggregate optimizer state without mutating it."""
    steps: list[int] = []
    tensor_count = 0
    tensor_elements = 0
    tensor_bytes = 0
    adam_first_moment_squared = 0.0
    adam_second_moment_squared = 0.0
    classp_sum = 0.0
    classp_squared = 0.0
    classp_nonzero = 0
    classp_elements = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            for key, value in optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value):
                    if not torch.isfinite(value).all():
                        raise FloatingPointError("optimizer state is non-finite")
                    tensor_count += 1
                    tensor_elements += value.numel()
                    tensor_bytes += value.numel() * value.element_size()
                    if key == "step":
                        steps.append(int(value.item()))
                    elif key == "exp_avg":
                        adam_first_moment_squared += float(
                            value.detach().double().square().sum().item()
                        )
                    elif key == "exp_avg_sq":
                        adam_second_moment_squared += float(
                            value.detach().double().square().sum().item()
                        )
                    elif key == "grad_sum":
                        double = value.detach().double()
                        classp_sum += float(double.sum().item())
                        classp_squared += float(double.square().sum().item())
                        classp_nonzero += int(torch.count_nonzero(value).item())
                        classp_elements += value.numel()
                elif key == "step":
                    steps.append(int(value))
    return {
        "parameters_with_state": len(optimizer.state),
        "state_tensor_count": tensor_count,
        "state_tensor_elements": tensor_elements,
        "state_bytes": tensor_bytes,
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "adam_first_moment_l2": (
            math.sqrt(adam_first_moment_squared)
            if isinstance(optimizer, optim.Adam)
            else None
        ),
        "adam_second_moment_l2": (
            math.sqrt(adam_second_moment_squared)
            if isinstance(optimizer, optim.Adam)
            else None
        ),
        "classp_accumulator_sum": classp_sum
        if isinstance(optimizer, CLASSP)
        else None,
        "classp_accumulator_l2": math.sqrt(classp_squared)
        if isinstance(optimizer, CLASSP)
        else None,
        "classp_accumulator_nonzero_fraction": (
            classp_nonzero / classp_elements if classp_elements else 0.0
        )
        if isinstance(optimizer, CLASSP)
        else None,
    }


def apply_boundary_policy(
    *,
    task_id: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_reference: torch.optim.Optimizer,
    state_policy: str,
    trainer: Any,
    diagnostics: bool,
) -> dict[str, Any]:
    """Apply the policy before ``task_id`` without changing model parameters."""
    if task_id <= 0:
        raise ValueError("boundary policy applies only before tasks after task 0")
    parameter_before = linear_model_checksum(model)
    state_before = frozen.optimizer_state_summary(optimizer)
    diagnostic_state_before = (
        optimizer_state_diagnostics(optimizer) if diagnostics else None
    )
    if state_policy == "task-reset":
        optimizer.state.clear()
        action = "cleared-parameter-optimizer-state"
    elif state_policy == "persistent":
        action = "retained-parameter-optimizer-state"
    else:
        raise ValueError(f"unknown state policy: {state_policy}")
    state_after = frozen.optimizer_state_summary(optimizer)
    diagnostic_state_after = (
        optimizer_state_diagnostics(optimizer) if diagnostics else None
    )
    parameter_after = linear_model_checksum(model)
    if parameter_before != parameter_after:
        raise RuntimeError("network parameters changed at a task boundary")
    object_retained = optimizer is optimizer_reference
    if not object_retained:
        raise RuntimeError("parameter optimizer object changed at a task boundary")
    trainer_match = None if trainer is None else trainer.get_optimizer_p() is optimizer
    if trainer_match is False:
        raise RuntimeError("PC trainer lost the parameter optimizer object")
    if state_policy == "task-reset" and state_after["parameters_with_state"] != 0:
        raise RuntimeError("task-reset did not clear parameter optimizer state")
    if state_policy == "persistent" and state_before != state_after:
        raise RuntimeError("persistent boundary changed parameter optimizer state")
    return {
        "before_task_index": task_id,
        "action": action,
        "network_parameter_checksum_before": parameter_before,
        "network_parameter_checksum_after": parameter_after,
        "network_parameters_preserved": True,
        "optimizer_object_retained": object_retained,
        "pc_trainer_optimizer_identity_match": trainer_match,
        "optimizer_state_before": state_before,
        "optimizer_state_after": state_after,
        "optimizer_diagnostics_before": diagnostic_state_before,
        "optimizer_diagnostics_after": diagnostic_state_after,
    }


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
    run_manifest_hash: str = "synthetic",
    local_host: dict[str, Any] | None = None,
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

    frozen.seed_everything(seed)
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

    optimizer = make_optimizer(model, condition["optimizer"])
    optimizer_reference = optimizer
    trainer = None
    trainer_optimizer_identity_match = None
    if condition["method"] == "pc":
        trainer = frozen.pc.PCTrainer(
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
        accuracy_matrix[0, task] = frozen.evaluate(
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
    boundary_summaries: list[dict[str, Any]] = []
    reset_task_indices: list[int] = []
    samples_processed_by_task: list[int] = []
    updates_by_task: list[int] = []
    task_parameter_displacement: list[dict[str, Any]] = []
    diagnostic_optimizer_states: list[dict[str, Any]] = []
    first_batch_after_boundary: list[dict[str, Any]] = []

    for task_id in range(n_tasks):
        boundary_action = "initial-task-no-boundary"
        if task_id > 0:
            boundary = apply_boundary_policy(
                task_id=task_id,
                model=model,
                optimizer=optimizer,
                optimizer_reference=optimizer_reference,
                state_policy=condition["state_policy"],
                trainer=trainer,
                diagnostics=diagnostics,
            )
            boundary_summaries.append(boundary)
            boundary_action = boundary["action"]
            if condition["state_policy"] == "task-reset":
                reset_task_indices.append(task_id)

        task_start_parameters = parameter_snapshot(model) if diagnostics else None
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
            prediction = frozen.predict_batch(model, data)
            batch_correct = int((prediction.argmax(dim=-1) == label).sum().item())
            task_batch_accuracy.append(batch_correct / int(label.numel()))
            task_correct += batch_correct
            task_samples += int(label.numel())

            target = F.one_hot(label, num_classes=OUTPUT_SIZE).float()
            model.train()
            first_update_parameters = (
                parameter_snapshot(model)
                if diagnostics and task_updates == 0
                else None
            )
            if condition["method"] == "pc":
                assert trainer is not None
                progress = trainer.train_on_batch(
                    inputs=data,
                    loss_fn=frozen.half_squared_error,
                    loss_fn_kwargs={"_target": target},
                    is_reset_optimizer_x_at_batch_start=True,
                    is_reset_optimizer_p_at_batch_start=False,
                )
                if finite_numbers(progress):
                    raise FloatingPointError("PC training progress is non-finite")
                current_gradient_l2 = frozen.gradient_l2(model)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = frozen.half_squared_error(model(data), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("BP loss is non-finite")
                loss.backward()
                current_gradient_l2 = frozen.gradient_l2(model)
                optimizer.step()

            if diagnostics:
                task_gradient_l2.append(current_gradient_l2)
                if task_updates == 0:
                    assert first_update_parameters is not None
                    first_batch_after_boundary.append(
                        {
                            "task_index": task_id,
                            "boundary_action": boundary_action,
                            "gradient_l2": current_gradient_l2,
                            "parameter_update": parameter_displacement(
                                first_update_parameters, model
                            ),
                            "optimizer_state_after_update": (
                                optimizer_state_diagnostics(optimizer)
                            ),
                        }
                    )
            task_updates += 1

        online_batch_accuracy.append(task_batch_accuracy)
        online_correct_by_task.append(task_correct)
        online_samples_by_task.append(task_samples)
        diagnostic_gradient_l2.append(task_gradient_l2)
        samples_processed_by_task.append(task_samples)
        updates_by_task.append(task_updates)
        optimizer_states.append(frozen.optimizer_state_summary(optimizer))
        if diagnostics:
            assert task_start_parameters is not None
            task_parameter_displacement.append(
                {
                    "task_index": task_id,
                    **parameter_displacement(task_start_parameters, model),
                }
            )
            diagnostic_optimizer_states.append(
                {
                    "task_index": task_id,
                    **optimizer_state_diagnostics(optimizer),
                }
            )

        for evaluation_task in range(n_tasks):
            accuracy_matrix[task_id + 1, evaluation_task] = frozen.evaluate(
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
        expected_step = (
            sum(updates_by_task[: task_index + 1])
            if condition["state_policy"] == "persistent"
            else updates_by_task[task_index]
        )
        if summary["step_min"] != expected_step or summary["step_max"] != expected_step:
            raise RuntimeError(
                f"optimizer state-policy step invariant failed after task {task_index}"
            )

    elapsed = time.perf_counter() - start_time
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "validated-cell",
        "mode": mode,
        "condition_id": condition_id,
        "cohort": (
            "smoke"
            if mode == "smoke"
            else "bridge"
            if seed == BRIDGE_SEED
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
            "pc_latent_reset_every_batch": condition["method"] == "pc",
            "diagnostics": diagnostics,
            "requested_device": device.type,
        },
        "run_manifest": {
            "manifest_hash": run_manifest_hash,
            "local_host": local_host or {"hostname": "synthetic"},
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
            "optimizer_object_retained": optimizer is optimizer_reference,
            "state_policy": condition["state_policy"],
            "optimizer_state_persistent": condition["state_policy"] == "persistent",
            "reset_count": len(reset_task_indices),
            "reset_task_indices": reset_task_indices,
            "pc_trainer_optimizer_identity_match": trainer_optimizer_identity_match,
            "pc_latent_reset_every_batch": condition["method"] == "pc",
            "boundary_summaries": boundary_summaries,
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
            "task_parameter_displacement": (
                task_parameter_displacement if diagnostics else None
            ),
            "optimizer_state_after_each_task": (
                diagnostic_optimizer_states if diagnostics else None
            ),
            "first_batch_after_boundary": (
                first_batch_after_boundary if diagnostics else None
            ),
        },
        "final_linear_parameter_checksum": linear_model_checksum(model),
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


def validate_result(
    result: dict[str, Any], *, require_provenance: bool = False
) -> None:
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected schema version")
    if result.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected protocol")
    condition_id = result.get("condition_id")
    if condition_id not in CONDITIONS:
        raise ValueError("unexpected condition")
    condition = CONDITIONS[condition_id]
    config = result["config"]
    if any(config.get(key) != value for key, value in condition.items()):
        raise ValueError("condition configuration mismatch")
    for key, value in COMMON_CONFIG.items():
        if config.get(key) != value:
            raise ValueError(f"frozen configuration mismatch: {key}")
    mode = result.get("mode")
    if mode not in MODE_CONFIG:
        raise ValueError("unexpected execution mode")
    for key, value in MODE_CONFIG[mode].items():
        if config.get(key) != value:
            raise ValueError(f"frozen mode configuration mismatch: {key}")
    seed = config.get("seed")
    if mode == "smoke" and seed != BRIDGE_SEED:
        raise ValueError("smoke seed mismatch")
    if mode == "full" and seed not in ALL_FULL_SEEDS:
        raise ValueError("full seed mismatch")
    if mode == "full" and not config.get("diagnostics"):
        raise ValueError("full cell did not enable diagnostics")
    n_tasks = config["n_tasks"]
    matrix = np.asarray(result["accuracy_matrix"], dtype=np.float64)
    if matrix.shape != (n_tasks + 1, n_tasks):
        raise ValueError("incorrect accuracy matrix shape")
    recomputed = compute_continual_metrics(result["accuracy_matrix"])
    for key, value in recomputed.items():
        if not np.allclose(value, result["metrics"][key], atol=1e-12):
            raise ValueError(f"metric mismatch: {key}")
    counts = result["counts"]
    if counts["expected_samples"] != counts["actual_samples"]:
        raise ValueError("sample count mismatch")
    if counts["expected_optimizer_updates"] != counts["actual_optimizer_updates"]:
        raise ValueError("optimizer update count mismatch")
    if not result["initialization"]["all_linear_tensors_matched"]:
        raise ValueError("initialization mismatch")

    audit = result["optimizer_audit"]
    if audit["instances_created"] != 1 or not audit["optimizer_object_retained"]:
        raise ValueError("optimizer object persistence mismatch")
    if audit["state_policy"] != condition["state_policy"]:
        raise ValueError("optimizer state-policy audit mismatch")
    expected_resets = list(range(1, n_tasks)) if condition["state_policy"] == "task-reset" else []
    if audit["reset_task_indices"] != expected_resets:
        raise ValueError("optimizer reset task indices mismatch")
    if audit["reset_count"] != len(expected_resets):
        raise ValueError("optimizer reset count mismatch")
    boundaries = audit["boundary_summaries"]
    if len(boundaries) != n_tasks - 1:
        raise ValueError("optimizer boundary summary count mismatch")
    for boundary in boundaries:
        if (
            not boundary["network_parameters_preserved"]
            or boundary["network_parameter_checksum_before"]
            != boundary["network_parameter_checksum_after"]
        ):
            raise ValueError("network parameters changed at a boundary")
        if not boundary["optimizer_object_retained"]:
            raise ValueError("optimizer object changed at a boundary")
        if condition["method"] == "pc" and not boundary[
            "pc_trainer_optimizer_identity_match"
        ]:
            raise ValueError("PC trainer optimizer identity mismatch")
        if condition["state_policy"] == "task-reset":
            if boundary["optimizer_state_after"]["parameters_with_state"] != 0:
                raise ValueError("task reset left optimizer state")
        elif boundary["optimizer_state_before"] != boundary["optimizer_state_after"]:
            raise ValueError("persistent boundary changed optimizer state")
    for task_index, summary in enumerate(audit["state_after_each_task"]):
        expected_step = (
            sum(counts["optimizer_updates_by_task"][: task_index + 1])
            if condition["state_policy"] == "persistent"
            else counts["optimizer_updates_by_task"][task_index]
        )
        if summary["step_min"] != expected_step or summary["step_max"] != expected_step:
            raise ValueError("optimizer step audit mismatch")
    if condition["method"] == "pc":
        if not audit["pc_trainer_optimizer_identity_match"]:
            raise ValueError("PC trainer did not retain parameter optimizer")
        if not audit["pc_latent_reset_every_batch"]:
            raise ValueError("PC latent reset audit mismatch")
    diagnostics = result["diagnostics"]
    if diagnostics["enabled"]:
        for key in (
            "gradient_l2_by_task",
            "task_parameter_displacement",
            "optimizer_state_after_each_task",
            "first_batch_after_boundary",
        ):
            if len(diagnostics[key]) != n_tasks:
                raise ValueError(f"diagnostic task count mismatch: {key}")
        for task_index, state in enumerate(
            diagnostics["optimizer_state_after_each_task"]
        ):
            expected_step = (
                sum(counts["optimizer_updates_by_task"][: task_index + 1])
                if condition["state_policy"] == "persistent"
                else counts["optimizer_updates_by_task"][task_index]
            )
            if (
                state["step_min"] != expected_step
                or state["step_max"] != expected_step
            ):
                raise ValueError("diagnostic optimizer step range mismatch")
            if condition["optimizer"] == "adam":
                if (
                    state["adam_first_moment_l2"] is None
                    or state["adam_second_moment_l2"] is None
                    or state["classp_accumulator_sum"] is not None
                ):
                    raise ValueError("Adam moment diagnostics mismatch")
            else:
                if (
                    state["classp_accumulator_sum"] is None
                    or state["classp_accumulator_l2"] is None
                    or state["classp_accumulator_nonzero_fraction"] is None
                    or state["adam_first_moment_l2"] is not None
                ):
                    raise ValueError("CLASSP accumulator diagnostics mismatch")
        for boundary in audit["boundary_summaries"]:
            if (
                boundary["optimizer_diagnostics_before"] is None
                or boundary["optimizer_diagnostics_after"] is None
            ):
                raise ValueError("boundary diagnostics missing")

    nonfinite = finite_numbers(
        {
            "accuracy_matrix": result["accuracy_matrix"],
            "metrics": result["metrics"],
            "online": result["online_predict_before_update"],
            "diagnostics": result["diagnostics"],
        }
    )
    if nonfinite:
        raise ValueError(f"non-finite scientific values: {nonfinite}")
    integrity = result.get("integrity")
    if integrity is not None:
        payload = dict(result)
        payload.pop("integrity")
        if integrity.get("canonical_payload_sha256") != sha256_bytes(
            canonical_json_bytes(payload)
        ):
            raise ValueError("canonical payload hash mismatch")
    if require_provenance:
        for key in ("source", "source_identity", "environment", "run_manifest"):
            if key not in result:
                raise ValueError(f"result lacks required provenance: {key}")
        if result["source_identity"] != source_identity(result["source"]):
            raise ValueError("result source identity mismatch")


def output_paths(
    output_dir: Path, mode: str, condition: str, seed: int
) -> dict[str, Path]:
    stem = f"{PROTOCOL_ID}__{mode}__{condition}__seed-{seed}"
    return {
        "result": output_dir / f"{stem}.json",
        "receipt": output_dir / f"{stem}.receipt.json",
        "failure": output_dir / f"{stem}.failure.json",
    }


def ensure_unique_paths(paths: dict[str, Path]) -> None:
    frozen.ensure_unique_paths(paths)


def describe_host(device_name: str, data_root: Path) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    source = source_provenance()
    assert_source_ready(source)
    environment = environment_provenance(device)
    _, _, _, hashes = frozen.load_permuted_mnist(2, data_root)
    return {
        "protocol_id": PROTOCOL_ID,
        "source": source,
        "source_identity": source_identity(source),
        "dependency_identity": environment["dependencies"],
        "dataset_file_sha256": hashes,
        "hostname": environment["hostname"],
        "gpu_identity": environment["gpu_identity"],
        "gpu_driver": environment.get("gpu_driver"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe-host", action="store_true")
    parser.add_argument("--mode", choices=sorted(MODE_CONFIG))
    parser.add_argument("--condition", choices=sorted(CONDITIONS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--confirm-protocol")
    parser.add_argument("--diagnostics", choices=("on", "off"))
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    return parser.parse_args(argv)


def validate_cli_protocol(args: argparse.Namespace) -> None:
    required = {
        "--mode": args.mode,
        "--condition": args.condition,
        "--seed": args.seed,
        "--device": args.device,
        "--diagnostics": args.diagnostics,
        "--run-manifest": args.run_manifest,
        "--output-dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("missing required cell arguments: " + ", ".join(missing))
    if args.mode == "smoke" and args.seed != BRIDGE_SEED:
        raise ValueError(f"smoke mode is frozen to seed {BRIDGE_SEED}")
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
    if args.describe_host:
        try:
            print(
                json.dumps(
                    describe_host(args.device or "cuda", args.data_root),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            return 0
        except (OSError, RuntimeError, ValueError) as error:
            print(f"HOST DESCRIPTION FAILED: {type(error).__name__}: {error}", file=sys.stderr)
            return 1

    try:
        validate_cli_protocol(args)
    except (RuntimeError, ValueError) as error:
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    assert args.output_dir is not None
    assert args.mode is not None
    assert args.condition is not None
    assert args.seed is not None
    assert args.device is not None
    assert args.run_manifest is not None
    assert args.diagnostics is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_dir, args.mode, args.condition, args.seed)
    started_at = utc_now()
    source: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    try:
        ensure_unique_paths(paths)
        frozen.assert_safe_staging_directory(args.output_dir)
        manifest = load_run_manifest(args.run_manifest)
        device = torch.device(args.device)
        source = source_provenance()
        assert_source_ready(source)
        frozen.seed_everything(args.seed)
        environment = environment_provenance(device)
        train_tasks, test_tasks, permutations, dataset_hashes = (
            frozen.load_permuted_mnist(
                MODE_CONFIG[args.mode]["n_tasks"], args.data_root
            )
        )
        local_host = validate_manifest_for_cell(
            manifest,
            mode=args.mode,
            condition=args.condition,
            seed=args.seed,
            source=source,
            environment=environment,
            dataset_hashes=dataset_hashes,
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
            run_manifest_hash=manifest["manifest_hash"],
            local_host=local_host,
        )
        result["source"] = source
        result["source_identity"] = source_identity(source)
        result["environment"] = environment
        result = seal_result(result)
        validate_result(result, require_provenance=True)
        atomic_write_json_exclusive(paths["result"], result)
        result_sha256 = sha256_file(paths["result"])
        receipt = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "artifact": paths["result"].name,
            "artifact_sha256": result_sha256,
            "artifact_bytes": paths["result"].stat().st_size,
            "canonical_payload_sha256": result["integrity"][
                "canonical_payload_sha256"
            ],
            "manifest_hash": manifest["manifest_hash"],
            "source_identity": result["source_identity"],
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
                    "host": local_host,
                    "result": str(paths["result"]),
                    "receipt": str(paths["receipt"]),
                    "sha256": result_sha256,
                    "manifest_hash": manifest["manifest_hash"],
                    "actual_samples": result["counts"]["actual_samples"],
                    "actual_optimizer_updates": result["counts"][
                        "actual_optimizer_updates"
                    ],
                    "elapsed_seconds": result["timing"]["elapsed_seconds"],
                },
                sort_keys=True,
                allow_nan=False,
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
            "run_manifest_path": str(args.run_manifest),
            "run_manifest_hash": None if manifest is None else manifest.get("manifest_hash"),
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
