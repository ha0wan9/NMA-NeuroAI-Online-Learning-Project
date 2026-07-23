#!/usr/bin/env python3
"""Run ``matched-lr-state-policy-v1`` sequentially on one local RTX 4090."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_STATE_ROOT = (
    SCRIPT_DIR / "run-staging/matched-lr-state-policy-v1"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
MIN_DISK_BYTES = 20 * 1024**3
MIN_AVAILABLE_RAM_BYTES = 16 * 1024**3
MAX_GPU_UTILIZATION_PERCENT = 5
MAX_GPU_TEMPERATURE_C = 40
MAX_BASELINE_MEMORY_DELTA_MIB = 512
IDLE_SAMPLE_INTERVAL_SECONDS = 10
IDLE_TIMEOUT_SECONDS = 10 * 60
WILLIAMS_BASE_ROW = (1, 2, 8, 3, 7, 4, 6, 5)

sys.path.insert(0, str(SCRIPT_DIR))

import run_state_policy_study as runner
import validate_state_policy_results as validator


CANONICAL_CONDITIONS = tuple(runner.CONDITIONS)
ORDERED_SEEDS = (runner.BRIDGE_SEED, *runner.PRIMARY_SEEDS)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    runner.atomic_write_json_exclusive(path, payload)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def same_or_refuse(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if load_json(path) != payload:
            raise FileExistsError(f"refusing to overwrite differing {path}")
        return
    write_json_exclusive(path, payload)


def same_text_or_refuse(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite differing {path}")
        return
    validator.atomic_write_text_exclusive(path, text)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, sort_keys=True
    ) + "\n"
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        os.write(descriptor, line.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ProtocolLock:
    def __init__(self, state_root: Path) -> None:
        self.path = state_root / ".protocol.lock"
        self.descriptor: int | None = None

    def __enter__(self) -> "ProtocolLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError as error:
            raise RuntimeError(
                f"another protocol command holds {self.path}"
            ) from error
        os.write(
            self.descriptor,
            f"pid={os.getpid()} host={platform.node()}\n".encode(),
        )
        os.fsync(self.descriptor)
        return self

    def __exit__(self, *_: Any) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
        self.path.unlink(missing_ok=True)


def state_layout(state_root: Path) -> dict[str, Path]:
    return {
        "root": state_root,
        "manifest": state_root / "manifest.json",
        "operational_log": state_root / "operational-log.jsonl",
        "smoke": state_root / "smoke",
        "smoke_pass_1": state_root / "smoke/pass-1",
        "smoke_pass_2": state_root / "smoke/pass-2",
        "full": state_root / "full",
        "bridge": state_root / "full/seed-42",
        "heldout": state_root / "full/heldout",
        "reports": state_root / "reports",
        "halts": state_root / "reports/halts",
    }


def ensure_layout(layout: dict[str, Path]) -> None:
    for key in (
        "root",
        "smoke_pass_1",
        "smoke_pass_2",
        "bridge",
        "heldout",
        "reports",
        "halts",
    ):
        layout[key].mkdir(parents=True, exist_ok=True)


def parse_mem_available(text: str) -> int:
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable is absent from /proc/meminfo")


def parse_gpu_sample(
    query_line: str,
    pmon_text: str,
    *,
    available_ram_bytes: int,
    disk_free_bytes: int,
    allowed_pids: set[int] | None = None,
) -> dict[str, Any]:
    parts = [part.strip() for part in query_line.split(",", maxsplit=4)]
    if len(parts) != 5:
        raise ValueError("unexpected nvidia-smi query output")
    utilization, temperature, memory_used, gpu_name, gpu_uuid = parts
    processes = []
    allowed_pids = allowed_pids or set()
    for line in pmon_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if (
            len(fields) >= 4
            and fields[1] != "-"
            and int(fields[1]) not in allowed_pids
        ):
            processes.append(stripped)
    return {
        "sampled_at_utc": runner.utc_now(),
        "utilization_percent": int(utilization),
        "temperature_c": int(temperature),
        "memory_used_mib": int(memory_used),
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "foreign_processes": processes,
        "available_ram_bytes": available_ram_bytes,
        "disk_free_bytes": disk_free_bytes,
    }


def sample_local_host(state_root: Path) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,temperature.gpu,memory.used,name,uuid",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pmon = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1", "-i", "0"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    memory_text = Path("/proc/meminfo").read_text(encoding="utf-8")
    return parse_gpu_sample(
        query,
        pmon,
        available_ram_bytes=parse_mem_available(memory_text),
        disk_free_bytes=shutil.disk_usage(state_root).free,
        allowed_pids={os.getpid()},
    )


def idle_sample_passes(
    sample: dict[str, Any], baseline_memory_mib: int
) -> bool:
    return (
        sample["utilization_percent"] <= MAX_GPU_UTILIZATION_PERCENT
        and sample["temperature_c"] <= MAX_GPU_TEMPERATURE_C
        and abs(sample["memory_used_mib"] - baseline_memory_mib)
        <= MAX_BASELINE_MEMORY_DELTA_MIB
        and not sample["foreign_processes"]
        and sample["available_ram_bytes"] >= MIN_AVAILABLE_RAM_BYTES
        and sample["disk_free_bytes"] >= MIN_DISK_BYTES
    )


def wait_for_idle(
    sample_fn: Callable[[], dict[str, Any]],
    *,
    baseline_memory_mib: int,
    timeout_seconds: float = IDLE_TIMEOUT_SECONDS,
    interval_seconds: float = IDLE_SAMPLE_INTERVAL_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    accepted: list[dict[str, Any]] = []
    while time.monotonic() <= deadline:
        sample = sample_fn()
        if idle_sample_passes(sample, baseline_memory_mib):
            accepted.append(sample)
            if len(accepted) == 2:
                return accepted
        else:
            accepted = []
        if time.monotonic() + interval_seconds > deadline:
            break
        sleep_fn(interval_seconds)
    raise TimeoutError("local RTX 4090 did not satisfy the idle gate in 10 minutes")


def verify_locked_dependencies() -> dict[str, Any]:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to verify the locked environment")
    lock_command = [uv, "lock", "--check"]
    completed = subprocess.run(
        lock_command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "locked dependency verification failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    sync_command = [uv, "sync", "--locked", "--extra", "cuda", "--check"]
    synchronized = subprocess.run(
        sync_command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if synchronized.returncode != 0:
        detail = synchronized.stderr.strip() or synchronized.stdout.strip()
        raise RuntimeError(
            "the CUDA environment is not synchronized with uv.lock: " + detail
        )
    return {
        "lock_command": lock_command,
        "environment_command": sync_command,
        "status": "passed",
        "lockfile_sha256": runner.sha256_file(REPO_ROOT / "uv.lock"),
    }


def williams_condition_orders() -> dict[int, list[str]]:
    index_to_condition = {
        index: condition
        for index, condition in enumerate(CANONICAL_CONDITIONS, 1)
    }
    orders: dict[int, list[str]] = {}
    for offset, seed in enumerate(ORDERED_SEEDS):
        row = [
            ((index - 1 + offset) % len(CANONICAL_CONDITIONS)) + 1
            for index in WILLIAMS_BASE_ROW
        ]
        orders[seed] = [index_to_condition[index] for index in row]
    return orders


def build_manifest(
    description: dict[str, Any],
    *,
    baseline_memory_mib: int,
    dependency_lock_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gpu = description["gpu_identity"]
    if gpu is None or "RTX 4090" not in gpu["name"]:
        raise ValueError("v1 preflight requires the local RTX 4090")
    orders = williams_condition_orders()
    host = {
        "hostname": description["hostname"],
        "gpu_identity": gpu,
        "gpu_driver": description["gpu_driver"],
        "cuda_runtime": description["dependency_identity"]["cuda_runtime"],
    }
    manifest = {
        "schema_version": 1,
        "protocol_id": runner.PROTOCOL_ID,
        "kind": "single-gpu-local",
        "status": "frozen",
        "created_at_utc": runner.utc_now(),
        "conditions": list(CANONICAL_CONDITIONS),
        "seeds": {
            "bridge": runner.BRIDGE_SEED,
            "held_out": list(runner.PRIMARY_SEEDS),
        },
        "condition_order_by_seed": {
            str(seed): orders[seed] for seed in ORDERED_SEEDS
        },
        "source": description["source"],
        "source_identity": description["source_identity"],
        "dependency_identity": description["dependency_identity"],
        "dependency_lock_check": dependency_lock_check
        or {
            "status": "synthetic",
            "lockfile_sha256": description["source"][
                "dependency_lock_sha256"
            ],
        },
        "dataset_file_sha256": description["dataset_file_sha256"],
        "host": host,
        "idle_policy": {
            "baseline_memory_mib": baseline_memory_mib,
            "samples_required": 2,
            "sample_interval_seconds": IDLE_SAMPLE_INTERVAL_SECONDS,
            "timeout_seconds": IDLE_TIMEOUT_SECONDS,
            "max_utilization_percent": MAX_GPU_UTILIZATION_PERCENT,
            "max_temperature_c": MAX_GPU_TEMPERATURE_C,
            "max_baseline_memory_delta_mib": MAX_BASELINE_MEMORY_DELTA_MIB,
            "minimum_available_ram_bytes": MIN_AVAILABLE_RAM_BYTES,
            "minimum_disk_free_bytes": MIN_DISK_BYTES,
            "foreign_graphics_or_compute_processes_permitted": False,
        },
        "execution": {
            "device": "cuda",
            "gpu_count": 1,
            "sequential": True,
            "smoke_passes": 2,
            "distributed_execution": False,
            "artifact_transfer": False,
        },
        "analysis_policy": {
            "bridge_seed_open_and_excluded": runner.BRIDGE_SEED,
            "held_out_seeds": list(runner.PRIMARY_SEEDS),
            "held_out_partial_metrics_visible": False,
            "confirmatory_analysis_requires_cells": 48,
            "bridge_report_review_required": True,
        },
    }
    manifest["manifest_hash"] = runner.manifest_hash(manifest)
    return manifest


def validate_manifest_description(
    manifest: dict[str, Any], description: dict[str, Any]
) -> None:
    if manifest["source_identity"] != description["source_identity"]:
        raise ValueError("manifest source does not match the current checkout")
    if manifest["dependency_identity"] != description["dependency_identity"]:
        raise ValueError("manifest dependencies do not match the current environment")
    if manifest["dataset_file_sha256"] != description["dataset_file_sha256"]:
        raise ValueError("manifest MNIST hashes do not match local data")
    expected_host = {
        "hostname": description["hostname"],
        "gpu_identity": description["gpu_identity"],
        "gpu_driver": description["gpu_driver"],
        "cuda_runtime": description["dependency_identity"]["cuda_runtime"],
    }
    if manifest["host"] != expected_host:
        raise ValueError("manifest host/GPU does not match the current RTX 4090")


def perform_preflight(
    *,
    state_root: Path,
    data_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    layout = state_layout(state_root)
    ensure_layout(layout)
    lock_check = verify_locked_dependencies()
    description = runner.describe_host("cuda", data_root)
    gpu = description["gpu_identity"]
    if gpu is None or "RTX 4090" not in gpu["name"]:
        raise RuntimeError("preflight requires the stable local RTX 4090")
    initial = sample_local_host(state_root)
    existing = (
        runner.load_run_manifest(layout["manifest"])
        if layout["manifest"].exists()
        else None
    )
    if existing is not None:
        validate_manifest_description(existing, description)
        if (
            existing["dependency_lock_check"]["lockfile_sha256"]
            != lock_check["lockfile_sha256"]
        ):
            raise ValueError("manifest lockfile identity no longer matches")
        baseline_memory_mib = existing["idle_policy"][
            "baseline_memory_mib"
        ]
    else:
        baseline_memory_mib = initial["memory_used_mib"]
    idle_samples = wait_for_idle(
        lambda: sample_local_host(state_root),
        baseline_memory_mib=baseline_memory_mib,
    )
    if existing is not None:
        manifest = existing
    else:
        manifest = build_manifest(
            description,
            baseline_memory_mib=baseline_memory_mib,
            dependency_lock_check=lock_check,
        )
        write_json_exclusive(layout["manifest"], manifest)
    append_jsonl(
        layout["operational_log"],
        {
            "event": "preflight-passed",
            "at_utc": runner.utc_now(),
            "manifest_hash": manifest["manifest_hash"],
            "idle_samples": idle_samples,
        },
    )
    return manifest, idle_samples


def require_manifest(layout: dict[str, Path]) -> dict[str, Any]:
    if not layout["manifest"].exists():
        raise RuntimeError("run preflight before this command")
    return runner.load_run_manifest(layout["manifest"])


def next_attempt_directory(cell_root: Path) -> Path:
    existing = sorted(
        path
        for path in cell_root.glob("attempt-*")
        if path.is_dir() and path.name.removeprefix("attempt-").isdigit()
    )
    next_index = (
        max(int(path.name.removeprefix("attempt-")) for path in existing) + 1
        if existing
        else 1
    )
    return cell_root / f"attempt-{next_index:03d}"


def result_path_in(
    attempt: Path, *, mode: str, condition: str, seed: int
) -> Path:
    return runner.output_paths(attempt, mode, condition, seed)["result"]


def valid_pair_in(
    attempt: Path,
    *,
    manifest: dict[str, Any],
    mode: str,
    condition: str,
    seed: int,
) -> dict[str, Any] | None:
    result_path = result_path_in(
        attempt, mode=mode, condition=condition, seed=seed
    )
    receipt_path = result_path.with_name(f"{result_path.stem}.receipt.json")
    if not result_path.exists() or not receipt_path.exists():
        return None
    return validator.validate_attempt_pair(
        result_path,
        manifest=manifest,
        mode=mode,
        condition=condition,
        seed=seed,
    )


def find_valid_pair(
    cell_root: Path,
    *,
    manifest: dict[str, Any],
    mode: str,
    condition: str,
    seed: int,
) -> tuple[Path, dict[str, Any]] | None:
    found = []
    for attempt in sorted(cell_root.glob("attempt-*")):
        if not attempt.is_dir():
            continue
        result = valid_pair_in(
            attempt,
            manifest=manifest,
            mode=mode,
            condition=condition,
            seed=seed,
        )
        if result is not None:
            found.append((attempt, result))
    if len(found) > 1:
        raise ValueError(f"duplicate sealed attempts in {cell_root}")
    return found[0] if found else None


def cell_root_for(
    layout: dict[str, Path],
    *,
    stage: str,
    condition: str,
    seed: int,
    pass_index: int | None = None,
) -> Path:
    if stage == "smoke":
        if pass_index not in (1, 2):
            raise ValueError("smoke cells require pass 1 or 2")
        return layout[f"smoke_pass_{pass_index}"] / condition
    if stage == "bridge":
        return layout["bridge"] / condition
    if stage == "heldout":
        return layout["heldout"] / f"seed-{seed}" / condition
    raise ValueError(f"unknown stage: {stage}")


def runner_command(
    *,
    manifest_path: Path,
    output_dir: Path,
    mode: str,
    condition: str,
    seed: int,
    data_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_state_policy_study.py"),
        "--mode",
        mode,
        "--condition",
        condition,
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--diagnostics",
        "on",
        "--run-manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--data-root",
        str(data_root),
    ]
    if mode == "full":
        command.extend(["--confirm-protocol", runner.PROTOCOL_ID])
    return command


class CellFailure(RuntimeError):
    def __init__(self, message: str, attempt: Path) -> None:
        super().__init__(message)
        self.attempt = attempt


def run_cell(
    *,
    layout: dict[str, Path],
    manifest: dict[str, Any],
    stage: str,
    condition: str,
    seed: int,
    data_root: Path,
    pass_index: int | None,
    resume: bool,
) -> dict[str, Any]:
    mode = "smoke" if stage == "smoke" else "full"
    cell_root = cell_root_for(
        layout,
        stage=stage,
        condition=condition,
        seed=seed,
        pass_index=pass_index,
    )
    cell_root.mkdir(parents=True, exist_ok=True)
    existing = find_valid_pair(
        cell_root,
        manifest=manifest,
        mode=mode,
        condition=condition,
        seed=seed,
    )
    if existing is not None:
        attempt, result = existing
        return {
            "status": "skipped-valid",
            "stage": stage,
            "condition_id": condition,
            "seed": seed,
            "attempt": str(attempt),
            "artifact_sha256": result["_artifact_sha256"]
            if "_artifact_sha256" in result
            else runner.sha256_file(
                result_path_in(
                    attempt, mode=mode, condition=condition, seed=seed
                )
            ),
            "elapsed_seconds": result["timing"]["elapsed_seconds"],
        }
    prior_attempts = sorted(cell_root.glob("attempt-*"))
    if prior_attempts and not resume:
        raise RuntimeError(
            f"incomplete attempt exists for {condition}, seed {seed}; use resume"
        )
    attempt = next_attempt_directory(cell_root)
    attempt.mkdir(parents=False, exist_ok=False)
    idle_samples = wait_for_idle(
        lambda: sample_local_host(layout["root"]),
        baseline_memory_mib=manifest["idle_policy"]["baseline_memory_mib"],
    )
    same_or_refuse(attempt / "idle-gate.json", {"samples": idle_samples})
    append_jsonl(
        layout["operational_log"],
        {
            "event": "cell-started",
            "at_utc": runner.utc_now(),
            "stage": stage,
            "pass_index": pass_index,
            "condition_id": condition,
            "seed": seed,
            "attempt": str(attempt),
        },
    )
    command = runner_command(
        manifest_path=layout["manifest"],
        output_dir=attempt,
        mode=mode,
        condition=condition,
        seed=seed,
        data_root=data_root,
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    same_text_or_refuse(attempt / "runner.stdout.log", completed.stdout)
    same_text_or_refuse(attempt / "runner.stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise CellFailure(
            f"runner failed for {condition}, seed {seed} "
            f"with exit code {completed.returncode}",
            attempt,
        )
    try:
        result_path = result_path_in(
            attempt, mode=mode, condition=condition, seed=seed
        )
        result = validator.validate_attempt_pair(
            result_path,
            manifest=manifest,
            mode=mode,
            condition=condition,
            seed=seed,
        )
    except BaseException as error:
        raise CellFailure(
            f"result validation failed for {condition}, seed {seed}: {error}",
            attempt,
        ) from error
    outcome = {
        "status": "validated-cell",
        "stage": stage,
        "condition_id": condition,
        "seed": seed,
        "attempt": str(attempt),
        "result": str(result_path),
        "artifact_sha256": runner.sha256_file(result_path),
        "elapsed_seconds": result["timing"]["elapsed_seconds"],
    }
    append_jsonl(
        layout["operational_log"],
        {
            "event": "cell-validated",
            "at_utc": runner.utc_now(),
            **outcome,
        },
    )
    return outcome


def create_halt(
    *,
    layout: dict[str, Path],
    stage: str,
    condition: str,
    seed: int,
    attempt: Path | None,
    error: BaseException,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "protocol_id": runner.PROTOCOL_ID,
        "status": "human-review-required",
        "stage": stage,
        "condition_id": condition,
        "seed": seed,
        "attempt": None if attempt is None else str(attempt),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "created_at_utc": runner.utc_now(),
    }
    halt_hash = runner.sha256_bytes(runner.canonical_json_bytes(payload))
    payload["halt_hash"] = halt_hash
    path = layout["halts"] / f"halt-{halt_hash}.json"
    same_or_refuse(path, payload)
    append_jsonl(
        layout["operational_log"],
        {
            "event": "stage-halted",
            "at_utc": runner.utc_now(),
            "stage": stage,
            "halt_hash": halt_hash,
            "halt_path": str(path),
        },
    )
    return payload


def stage_queue(
    stage: str,
) -> list[tuple[int | None, int, str]]:
    orders = williams_condition_orders()
    if stage == "smoke":
        return [
            (pass_index, runner.BRIDGE_SEED, condition)
            for pass_index in (1, 2)
            for condition in orders[runner.BRIDGE_SEED]
        ]
    if stage == "bridge":
        return [
            (None, runner.BRIDGE_SEED, condition)
            for condition in orders[runner.BRIDGE_SEED]
        ]
    if stage == "heldout":
        return [
            (None, seed, condition)
            for seed in runner.PRIMARY_SEEDS
            for condition in orders[seed]
        ]
    raise ValueError(f"unknown stage: {stage}")


def stage_progress(
    *,
    layout: dict[str, Path],
    manifest: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    complete = 0
    incomplete_attempts: list[str] = []
    runtimes = []
    artifacts = []
    for pass_index, seed, condition in stage_queue(stage):
        mode = "smoke" if stage == "smoke" else "full"
        cell_root = cell_root_for(
            layout,
            stage=stage,
            condition=condition,
            seed=seed,
            pass_index=pass_index,
        )
        valid = find_valid_pair(
            cell_root,
            manifest=manifest,
            mode=mode,
            condition=condition,
            seed=seed,
        )
        if valid is not None:
            attempt, result = valid
            complete += 1
            runtimes.append(result["timing"]["elapsed_seconds"])
            result_path = result_path_in(
                attempt, mode=mode, condition=condition, seed=seed
            )
            artifacts.append(
                {
                    "condition_id": condition,
                    "seed": seed,
                    "pass_index": pass_index,
                    "path": str(result_path),
                    "sha256": runner.sha256_file(result_path),
                    "elapsed_seconds": result["timing"]["elapsed_seconds"],
                }
            )
        elif cell_root.exists():
            incomplete_attempts.extend(
                str(path)
                for path in sorted(cell_root.glob("attempt-*"))
                if path.is_dir()
            )
    failures = []
    for path in sorted(layout["halts"].glob("halt-*.json")):
        payload = load_json(path)
        if payload["stage"] != stage:
            continue
        failures.append(
            {
                "path": str(path),
                "sha256": runner.sha256_file(path),
                "halt_hash": payload["halt_hash"],
            }
        )
    return {
        "stage": stage,
        "completed_cells": complete,
        "expected_cells": len(stage_queue(stage)),
        "complete": complete == len(stage_queue(stage)),
        "incomplete_attempts": incomplete_attempts,
        "failures": failures,
        "runtime_seconds_completed": sum(runtimes),
        "artifacts": artifacts,
    }


def resume_review_sha256(
    *,
    layout: dict[str, Path],
    manifest: dict[str, Any],
    stage: str,
) -> str:
    progress = stage_progress(layout=layout, manifest=manifest, stage=stage)
    review_payload = {
        "protocol_id": runner.PROTOCOL_ID,
        "manifest_hash": manifest["manifest_hash"],
        "stage": stage,
        "completed_artifacts": [
            {
                "path": item["path"],
                "sha256": item["sha256"],
            }
            for item in progress["artifacts"]
        ],
        "incomplete_attempts": [
            {
                "path": path,
                "files": [
                    {
                        "path": str(file),
                        "sha256": runner.sha256_file(file),
                    }
                    for file in sorted(Path(path).rglob("*"))
                    if file.is_file()
                ],
            }
            for path in progress["incomplete_attempts"]
        ],
        "failures": progress["failures"],
    }
    return runner.sha256_bytes(runner.canonical_json_bytes(review_payload))


def authorize_resume(
    *,
    layout: dict[str, Path],
    manifest: dict[str, Any],
    stage: str,
    supplied_sha256: str | None,
) -> str:
    expected = resume_review_sha256(
        layout=layout, manifest=manifest, stage=stage
    )
    if supplied_sha256 != expected:
        raise RuntimeError(
            "resume requires explicit review; pass "
            f"--reviewed-failure-sha256 {expected}"
        )
    receipt = {
        "schema_version": 1,
        "protocol_id": runner.PROTOCOL_ID,
        "stage": stage,
        "reviewed_state_sha256": expected,
        "reviewed_at_utc": runner.utc_now(),
    }
    path = layout["reports"] / f"resume-review-{stage}-{expected}.json"
    if not path.exists():
        write_json_exclusive(path, receipt)
    append_jsonl(
        layout["operational_log"],
        {
            "event": "resume-authorized",
            "at_utc": runner.utc_now(),
            "stage": stage,
            "reviewed_state_sha256": expected,
            "receipt": str(path),
        },
    )
    return expected


def run_stage(
    *,
    layout: dict[str, Path],
    manifest: dict[str, Any],
    stage: str,
    data_root: Path,
    resume: bool,
    reviewed_failure_sha256: str | None = None,
) -> list[dict[str, Any]]:
    progress = stage_progress(layout=layout, manifest=manifest, stage=stage)
    if progress["complete"]:
        return []
    if not resume and (
        progress["completed_cells"] > 0 or progress["incomplete_attempts"]
    ):
        raise RuntimeError(
            f"{stage} already has preserved progress; use resume"
        )
    if resume:
        authorize_resume(
            layout=layout,
            manifest=manifest,
            stage=stage,
            supplied_sha256=reviewed_failure_sha256,
        )
    outcomes = []
    for pass_index, seed, condition in stage_queue(stage):
        try:
            outcome = run_cell(
                layout=layout,
                manifest=manifest,
                stage=stage,
                condition=condition,
                seed=seed,
                data_root=data_root,
                pass_index=pass_index,
                resume=resume,
            )
            outcomes.append(outcome)
        except BaseException as error:
            attempt = error.attempt if isinstance(error, CellFailure) else None
            create_halt(
                layout=layout,
                stage=stage,
                condition=condition,
                seed=seed,
                attempt=attempt,
                error=error,
            )
            raise
    return outcomes


def write_smoke_reports(
    layout: dict[str, Path], manifest: dict[str, Any]
) -> dict[str, Any]:
    report, cells = validator.smoke_repeat_analysis(
        pass_1=layout["smoke_pass_1"],
        pass_2=layout["smoke_pass_2"],
        manifest=manifest,
    )
    json_path = layout["reports"] / "smoke-validation.json"
    markdown_path = (
        layout["reports"] / "smoke-optimizer-state-report.md"
    )
    same_or_refuse(json_path, report)
    same_text_or_refuse(
        markdown_path,
        validator.render_smoke_markdown(report, cells),
    )
    return {
        "status": "passed",
        "validated_cells": 16,
        "report_json": str(json_path),
        "report_markdown": str(markdown_path),
        "report_markdown_sha256": runner.sha256_file(markdown_path),
    }


def write_bridge_reports(
    layout: dict[str, Path], manifest: dict[str, Any]
) -> dict[str, Any]:
    cells, validation = validator.validate_directory(
        layout["bridge"],
        mode="full",
        manifest=manifest,
        seeds=(runner.BRIDGE_SEED,),
        require_complete=True,
    )
    analysis = validator.bridge_analysis(cells)
    report = {**validation, "bridge_analysis": analysis}
    json_path = layout["reports"] / "bridge-intuition-report.json"
    markdown_path = layout["reports"] / "bridge-intuition-report.md"
    same_or_refuse(json_path, report)
    same_text_or_refuse(
        markdown_path,
        validator.render_bridge_markdown(
            validation=validation,
            analysis=analysis,
            manifest=manifest,
        ),
    )
    return {
        "status": "passed",
        "validated_cells": 8,
        "report_json": str(json_path),
        "report_markdown": str(markdown_path),
        "report_markdown_sha256": runner.sha256_file(markdown_path),
    }


def require_bridge_review(
    *,
    layout: dict[str, Path],
    supplied_sha256: str | None,
    confirmation: str | None,
) -> dict[str, Any]:
    if confirmation != runner.PROTOCOL_ID:
        raise ValueError(
            f"heldout requires --confirm-protocol {runner.PROTOCOL_ID}"
        )
    report_path = layout["reports"] / "bridge-intuition-report.md"
    if not report_path.exists():
        raise RuntimeError("bridge report is missing; validate bridge first")
    actual_sha256 = runner.sha256_file(report_path)
    if supplied_sha256 != actual_sha256:
        raise ValueError(
            "heldout requires the exact reviewed bridge report SHA-256 "
            f"{actual_sha256}"
        )
    review = {
        "schema_version": 1,
        "protocol_id": runner.PROTOCOL_ID,
        "bridge_report": str(report_path),
        "bridge_report_sha256": actual_sha256,
        "protocol_confirmation": confirmation,
    }
    same_or_refuse(layout["reports"] / "bridge-review.json", review)
    return review


def intended_run_order() -> dict[str, Any]:
    return {
        "protocol_id": runner.PROTOCOL_ID,
        "smoke": [
            {
                "pass_index": pass_index,
                "seed": seed,
                "condition_id": condition,
            }
            for pass_index, seed, condition in stage_queue("smoke")
        ],
        "bridge": [
            {"seed": seed, "condition_id": condition}
            for _, seed, condition in stage_queue("bridge")
        ],
        "heldout": [
            {"seed": seed, "condition_id": condition}
            for _, seed, condition in stage_queue("heldout")
        ],
    }


def write_artifact_manifest(state_root: Path, output_path: Path) -> None:
    lines = []
    for path in sorted(state_root.rglob("*")):
        if (
            not path.is_file()
            or path == output_path
            or path.name == ".protocol.lock"
        ):
            continue
        relative = path.relative_to(state_root)
        lines.append(f"{runner.sha256_file(path)}  {relative}")
    same_text_or_refuse(output_path, "\n".join(lines) + "\n")


def write_complete_reports(
    layout: dict[str, Path], manifest: dict[str, Any]
) -> dict[str, Any]:
    cells, validation = validator.validate_directory(
        layout["full"],
        mode="full",
        manifest=manifest,
        seeds=runner.ALL_FULL_SEEDS,
        require_complete=True,
    )
    analysis = validator.confirmatory_analysis(cells)
    report = {**validation, "confirmatory_analysis": analysis}
    json_path = layout["reports"] / "complete-validation.json"
    markdown_path = layout["reports"] / "complete-synthesis.md"
    same_or_refuse(json_path, report)
    same_text_or_refuse(
        markdown_path,
        validator.render_complete_markdown(
            validation=validation,
            analysis=analysis,
            manifest=manifest,
        ),
    )
    intended_path = layout["reports"] / "intended-run-order.json"
    same_or_refuse(intended_path, intended_run_order())
    actual = {
        "protocol_id": runner.PROTOCOL_ID,
        "cells": [
            {
                "condition_id": condition,
                "seed": seed,
                "started_at_utc": result["timing"]["started_at_utc"],
                "completed_at_utc": result["timing"]["completed_at_utc"],
                "result_path": result["_validated_path"],
                "artifact_sha256": result["_artifact_sha256"],
            }
            for (condition, seed), result in sorted(
                cells.items(),
                key=lambda item: item[1]["timing"]["started_at_utc"],
            )
        ],
    }
    actual_path = layout["reports"] / "actual-run-order.json"
    same_or_refuse(actual_path, actual)
    hashes_path = layout["reports"] / "artifact-manifest.sha256"
    write_artifact_manifest(layout["root"], hashes_path)
    return {
        "status": "passed",
        "validated_cells": 48,
        "report_json": str(json_path),
        "report_markdown": str(markdown_path),
        "intended_run_order": str(intended_path),
        "actual_run_order": str(actual_path),
        "operational_log": str(layout["operational_log"]),
        "artifact_manifest": str(hashes_path),
    }


def report_hashes(layout: dict[str, Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path), "sha256": runner.sha256_file(path)}
        for path in sorted(layout["reports"].glob("*"))
        if path.is_file()
    ]


def command_status(layout: dict[str, Path]) -> dict[str, Any]:
    if not layout["manifest"].exists():
        return {
            "status": "preflight-required",
            "state_root": str(layout["root"]),
            "manifest": None,
        }
    manifest = require_manifest(layout)
    stages = {
        stage: stage_progress(
            layout=layout, manifest=manifest, stage=stage
        )
        for stage in ("smoke", "bridge", "heldout")
    }
    for stage, progress in stages.items():
        if not progress["complete"]:
            progress["resume_review_sha256"] = resume_review_sha256(
                layout=layout,
                manifest=manifest,
                stage=stage,
            )
    return {
        "status": "structural-status",
        "state_root": str(layout["root"]),
        "manifest": str(layout["manifest"]),
        "manifest_hash": manifest["manifest_hash"],
        "stages": stages,
        "reports": report_hashes(layout),
        "heldout_effects_exposed": False,
    }


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--state-root", type=Path, default=DEFAULT_STATE_ROOT
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("preflight", "smoke", "bridge", "status"):
        child = subparsers.add_parser(name)
        add_common_arguments(child)

    heldout = subparsers.add_parser("heldout")
    add_common_arguments(heldout)
    heldout.add_argument("--confirm-protocol")
    heldout.add_argument("--bridge-review-sha256")

    resume = subparsers.add_parser("resume")
    add_common_arguments(resume)
    resume.add_argument(
        "--stage", required=True, choices=("smoke", "bridge", "heldout")
    )
    resume.add_argument("--reviewed-failure-sha256", required=True)
    resume.add_argument("--confirm-protocol")
    resume.add_argument("--bridge-review-sha256")

    validate = subparsers.add_parser("validate")
    add_common_arguments(validate)
    validate.add_argument(
        "--stage", required=True, choices=("smoke", "bridge", "complete")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    layout = state_layout(args.state_root)
    try:
        if args.command == "status":
            result = command_status(layout)
        else:
            with ProtocolLock(args.state_root):
                ensure_layout(layout)
                if args.command == "preflight":
                    manifest, idle_samples = perform_preflight(
                        state_root=args.state_root,
                        data_root=args.data_root,
                    )
                    result = {
                        "status": "preflight-passed",
                        "manifest": str(layout["manifest"]),
                        "manifest_hash": manifest["manifest_hash"],
                        "host": manifest["host"],
                        "idle_samples": len(idle_samples),
                    }
                else:
                    manifest = require_manifest(layout)
                    if args.command == "smoke":
                        run_stage(
                            layout=layout,
                            manifest=manifest,
                            stage="smoke",
                            data_root=args.data_root,
                            resume=False,
                        )
                        result = write_smoke_reports(layout, manifest)
                    elif args.command == "bridge":
                        write_smoke_reports(layout, manifest)
                        run_stage(
                            layout=layout,
                            manifest=manifest,
                            stage="bridge",
                            data_root=args.data_root,
                            resume=False,
                        )
                        result = write_bridge_reports(layout, manifest)
                    elif args.command == "heldout":
                        write_bridge_reports(layout, manifest)
                        require_bridge_review(
                            layout=layout,
                            supplied_sha256=args.bridge_review_sha256,
                            confirmation=args.confirm_protocol,
                        )
                        run_stage(
                            layout=layout,
                            manifest=manifest,
                            stage="heldout",
                            data_root=args.data_root,
                            resume=False,
                        )
                        result = write_complete_reports(layout, manifest)
                    elif args.command == "resume":
                        if args.stage == "bridge":
                            write_smoke_reports(layout, manifest)
                        if args.stage == "heldout":
                            write_bridge_reports(layout, manifest)
                            require_bridge_review(
                                layout=layout,
                                supplied_sha256=args.bridge_review_sha256,
                                confirmation=args.confirm_protocol,
                            )
                        run_stage(
                            layout=layout,
                            manifest=manifest,
                            stage=args.stage,
                            data_root=args.data_root,
                            resume=True,
                            reviewed_failure_sha256=(
                                args.reviewed_failure_sha256
                            ),
                        )
                        if args.stage == "smoke":
                            result = write_smoke_reports(layout, manifest)
                        elif args.stage == "bridge":
                            result = write_bridge_reports(layout, manifest)
                        else:
                            result = write_complete_reports(layout, manifest)
                    elif args.command == "validate":
                        if args.stage == "smoke":
                            result = write_smoke_reports(layout, manifest)
                        elif args.stage == "bridge":
                            result = write_bridge_reports(layout, manifest)
                        else:
                            result = write_complete_reports(layout, manifest)
                    else:
                        raise AssertionError(args.command)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as error:
        print(
            f"PROTOCOL FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
