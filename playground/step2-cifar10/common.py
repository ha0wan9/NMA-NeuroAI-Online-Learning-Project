"""Shared model, runtime, accounting, and metric helpers for CIFAR-10 PC runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any


CLASS_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
PAPER_SEEDS = (1482555873, 698841058, 2283198659)
PAPER_BEST_LR = {
    "pc": {1482555873: 5e-5, 698841058: 1e-4, 2283198659: 2.5e-5},
    "bp": {1482555873: 7.5e-5, 698841058: 1e-4, 2283198659: 5e-5},
}
PAPER_BEST_ACCURACY = {
    "pc": {
        1482555873: 0.7128571493893254,
        698841058: 0.7194898031195816,
        2283198659: 0.7156122512355143,
    },
    "bp": {
        1482555873: 0.6921428627505595,
        698841058: 0.6930612331750442,
        2283198659: 0.688163272580322,
    },
}
SPLIT_FIXED_LR = {"pc": 2.5e-5, "bp": 7.5e-5}
EXPECTED_PARAMETER_COUNT = 4_275_402


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
            "PyTorch runtime unavailable; use the pinned CyberEngine ROCm image."
        ) from exc

    pc_root = Path(__file__).resolve().parents[1] / "predictive-coding"
    if not (pc_root / "predictive_coding" / "pc_trainer.py").exists():
        raise RuntimeError("predictive-coding submodule is missing")
    sys.path.insert(0, str(pc_root))
    try:
        import predictive_coding as pc
    except ImportError as exc:
        raise RuntimeError("predictive-coding dependencies are unavailable") from exc
    return locals()


def seed_everything(seed: int, runtime: dict[str, Any]) -> None:
    torch, np = runtime["torch"], runtime["np"]
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str, torch: Any) -> Any:
    device = torch.device(
        "cuda" if requested == "auto" and torch.cuda.is_available()
        else "cpu" if requested == "auto"
        else requested
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU execution was required, but ROCm/CUDA is unavailable")
    return device


def build_model(predictive_coding: bool, runtime: dict[str, Any]) -> Any:
    nn, pc = runtime["nn"], runtime["pc"]
    modules: list[Any] = []
    for layer in (
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
    ):
        modules.append(layer)
        if predictive_coding:
            modules.append(pc.PCLayer())
        modules.append(nn.ReLU())
    modules.extend((nn.Flatten(), nn.Linear(8192, 512)))
    if predictive_coding:
        modules.append(pc.PCLayer())
    modules.extend((nn.ReLU(), nn.Linear(512, 10)))
    return nn.Sequential(*modules)


def copy_learned_parameters(source: Any, target: Any, runtime: dict[str, Any]) -> None:
    nn = runtime["nn"]
    source_layers = [m for m in source.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    target_layers = [m for m in target.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for source_layer, target_layer in zip(source_layers, target_layers, strict=True):
        target_layer.load_state_dict(source_layer.state_dict())


def learned_parameters(model: Any, runtime: dict[str, Any]) -> list[Any]:
    nn = runtime["nn"]
    return [
        parameter
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
        for parameter in module.parameters()
    ]


def learned_parameter_count(model: Any, runtime: dict[str, Any]) -> int:
    return sum(parameter.numel() for parameter in learned_parameters(model, runtime))


def parameter_checksum(model: Any, runtime: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for parameter in learned_parameters(model, runtime):
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def latent_state_elements(model: Any, runtime: dict[str, Any]) -> int:
    pc = runtime["pc"]
    return sum(
        module.get_x().numel()
        for module in model.modules()
        if isinstance(module, pc.PCLayer) and module.get_x() is not None
    )


def squared_error(outputs: Any, target: Any) -> Any:
    return 0.5 * (outputs - target).pow(2).sum()


def make_trainer(model: Any, parameter_lr: float, pc_steps: int, state_lr: float,
                 runtime: dict[str, Any]) -> Any:
    optim, pc = runtime["optim"], runtime["pc"]
    return pc.PCTrainer(
        model,
        T=pc_steps,
        optimizer_x_fn=optim.SGD,
        optimizer_x_kwargs={"lr": state_lr},
        x_lr_discount=0.5,
        x_lr_amplifier=1.0,
        update_x_at="all",
        update_p_at="all",
        optimizer_p_fn=optim.Adam,
        optimizer_p_kwargs={"lr": parameter_lr, "weight_decay": 0.01},
        plot_progress_at=[],
    )


def evaluate(model: Any, loader: Any, device: Any, runtime: dict[str, Any]) -> dict[str, float]:
    torch, functional = runtime["torch"], runtime["functional"]
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            target = functional.one_hot(labels, num_classes=10).float()
            loss_sum += squared_error(outputs, target).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            count += labels.numel()
    if count == 0:
        raise RuntimeError("evaluation loader produced no samples")
    return {"loss_per_example": loss_sum / count, "accuracy": correct / count, "samples": count}


def synchronize(device: Any, torch: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def index_checksum(indices: list[int]) -> str:
    return hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()


def environment_record(runtime: dict[str, Any], device: Any) -> dict[str, Any]:
    torch, np, torchvision = runtime["torch"], runtime["np"], runtime["torchvision"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "execution_host": os.getenv("EXPERIMENT_EXECUTION_HOST"),
        "runtime_image": os.getenv("EXPERIMENT_RUNTIME_IMAGE"),
        "runtime_image_id": os.getenv("EXPERIMENT_RUNTIME_IMAGE_ID"),
        "paper_source_revision": "625d52678b3d4cab2dbc403b1d6ee0cfc004aea0",
        "pc_submodule_revision": "5bf803c3636a39928488941f05539c70ab4df0b1",
    }


def distribution(values: list[float], np: Any) -> dict[str, Any]:
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }


def make_summary_writer(tensorboard_dir: Path | None, run_label: str,
                        method: str, seed: int, config: dict[str, Any]) -> Any:
    """Create an optional per-method writer without making TensorBoard mandatory."""
    if tensorboard_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard tracking requested but tensorboard is unavailable; "
            "use remote-run.sh, which provisions the persistent dependency."
        ) from exc
    log_dir = tensorboard_dir / run_label / f"{method}-seed-{seed}"
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=10)
    writer.add_text("run/config", json.dumps(config, sort_keys=True, indent=2), 0)
    return writer


def log_pc_dynamics(writer: Any, dynamics: dict[str, Any] | None,
                    prefix: str = "pc_inference") -> None:
    if writer is None or dynamics is None:
        return
    for name in ("loss", "energy", "overall"):
        for step, value in enumerate(dynamics[name]):
            writer.add_scalar(f"{prefix}/{name}", value, step)


def continual_metrics(matrix: list[list[float]], prequential: float) -> dict[str, Any]:
    task_count = len(matrix[0])
    final = matrix[-1]
    post_task = [matrix[task_id + 1][task_id] for task_id in range(task_count)]
    pre_task = [matrix[task_id][task_id] for task_id in range(task_count)]
    bwt = [final[i] - post_task[i] for i in range(task_count - 1)]
    forgetting = [
        max(row[i] for row in matrix[i + 1:]) - final[i]
        for i in range(task_count - 1)
    ]
    adaptation = [post - pre for post, pre in zip(post_task, pre_task, strict=True)]
    return {
        "final_average_accuracy": sum(final) / task_count,
        "prequential_accuracy": prequential,
        "backward_transfer": sum(bwt) / len(bwt),
        "average_forgetting": sum(forgetting) / len(forgetting),
        "mean_adaptation_gain": sum(adaptation) / len(adaptation),
        "adaptation_gain_by_task": adaptation,
        "final_accuracy_by_task": final,
        "post_training_accuracy_by_task": post_task,
    }
