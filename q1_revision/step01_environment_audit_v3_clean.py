#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 01 — ENVIRONMENT, CUDA, WSL2, PACKAGE AND REPRODUCIBILITY AUDIT

Project:
Artificial Intelligence-Based Multi-Horizon Directional Forecasting
of Indian Equities: A Leakage-Controlled and Dependence-Aware Evaluation

Purpose
-------
1. Verify that the immutable Step-00 scientific configuration is intact.
2. Verify that the final held-out test remains LOCKED_NOT_OPENED.
3. Audit Python, Conda, Linux/WSL, CPU, RAM, disk, package versions,
   NVIDIA driver, CUDA and PyTorch.
4. Detect unexpected concurrent CUDA compute processes before the GPU test.
5. Run a small deterministic FP32 CUDA forward/backward smoke test.
6. Persist an auditable environment snapshot with SHA-256 hashes.

This script DOES NOT:
- download market data,
- open final-test arrays,
- train any forecasting model,
- tune any model/threshold,
- inspect old v2 results.

Expected location
-----------------
<repository>/q1_revision/step01_environment_audit.py

Run
---
python step01_environment_audit.py

Re-verify an already frozen Step-01 audit
----------------------------------------
python step01_environment_audit.py --verify-only
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent

if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError(
        "step01_environment_audit.py must be located inside "
        "<repository>/q1_revision/.\n"
        f"Current script directory: {SCRIPT_DIR}"
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import step00_config as cfg
except Exception as exc:
    raise RuntimeError(
        "Could not import q1_revision/step00_config.py. "
        "Step 00 must exist and remain unchanged before Step 01."
    ) from exc


STEP_ID = "01"
STEP_NAME = "environment_cuda_wsl2_reproducibility_audit"
SCHEMA_VERSION = "1.1"

ENV_AUDIT_JSON = cfg.ENVIRONMENT_DIR / "01_environment_audit.json"
PACKAGE_CSV = cfg.ENVIRONMENT_DIR / "01_package_versions.csv"
SOURCE_HASH_CSV = cfg.ENVIRONMENT_DIR / "01_source_file_hashes.csv"
PIP_FREEZE_TXT = cfg.ENVIRONMENT_DIR / "01_pip_freeze.txt"
NVIDIA_SMI_TXT = cfg.ENVIRONMENT_DIR / "01_nvidia_smi.txt"
NVIDIA_PROCESS_TXT = cfg.ENVIRONMENT_DIR / "01_nvidia_compute_processes.txt"
SYSTEM_COMMANDS_TXT = cfg.ENVIRONMENT_DIR / "01_system_commands.txt"
CONDA_EXPLICIT_TXT = cfg.ENVIRONMENT_DIR / "01_conda_list_explicit.txt"
CONDA_EXPORT_TXT = cfg.ENVIRONMENT_DIR / "01_conda_env_export_no_builds.yml"

STEP01_MANIFEST = cfg.MANIFEST_DIR / "01_environment_audit_manifest.json"
STEP01_MANIFEST_SHA = cfg.MANIFEST_DIR / "01_environment_audit_manifest.sha256"
STEP01_FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step01_environment_audit_frozen.py"
STEP01_SUMMARY = cfg.REPORT_DIR / "01_environment_audit_summary.txt"

FAILED_AUDIT_DIR = cfg.AUDIT_DIR / "step01_failed_attempts"

REQUIRED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "yfinance",
    "pyarrow",
    "joblib",
    "matplotlib",
)

OPTIONAL_PACKAGES = (
    "torchvision",
    "torchaudio",
    "statsmodels",
    "xgboost",
    "lightgbm",
    "optuna",
    "einops",
    "tqdm",
    "psutil",
    "ta",
    "packaging",
    "pandas-ta",
    "tensorflow",
)

MINIMUM_PYTHON = (3, 10)
GPU_SMOKE_MATRIX_SIZE = 512
GPU_SMOKE_SEED = int(cfg.GLOBAL_RANDOM_SEED)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(temporary, path)


def run_command(
    command: list[str],
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.perf_counter() - started
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_seconds": elapsed,
            "ok": completed.returncode == 0,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
            "elapsed_seconds": elapsed,
            "ok": False,
        }


def get_distribution_version(distribution_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024.0 ** 3)


def bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0 ** 2)


def verify_step00_contract() -> Dict[str, Any]:
    cfg.validate_configuration()

    payload = cfg.build_configuration_payload()
    fingerprint = cfg.canonical_json_sha256(payload)

    manifest_path, manifest_sha256, freeze_status = cfg.verify_existing_freeze(
        configuration_payload=payload,
        configuration_fingerprint=fingerprint,
    )

    lock_path = cfg.get_final_test_lock_path()
    with lock_path.open("r", encoding="utf-8") as handle:
        lock_payload = json.load(handle)

    if lock_payload.get("state") != "LOCKED_NOT_OPENED":
        raise RuntimeError(
            "Final-test lock is not LOCKED_NOT_OPENED. "
            "Step 01 refuses to continue."
        )

    if lock_payload.get("final_test_opened") is not False:
        raise RuntimeError(
            "Final-test lock indicates final_test_opened != False."
        )

    if lock_payload.get("final_test_scored") is not False:
        raise RuntimeError(
            "Final-test lock indicates final_test_scored != False."
        )

    return {
        "configuration_fingerprint_sha256": fingerprint,
        "step00_manifest_path": str(manifest_path),
        "step00_manifest_sha256": manifest_sha256,
        "step00_freeze_status": freeze_status,
        "final_test_lock_path": str(lock_path),
        "final_test_lock_sha256": sha256_file(lock_path),
        "final_test_state": lock_payload.get("state"),
        "final_test_opened": lock_payload.get("final_test_opened"),
        "final_test_scored": lock_payload.get("final_test_scored"),
    }


def read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def get_memory_info() -> Dict[str, Any]:
    meminfo = read_text_if_exists(Path("/proc/meminfo"))
    parsed_kib: Dict[str, int] = {}

    for line in meminfo.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            parsed_kib[key] = int(parts[0])
        except ValueError:
            pass

    total_kib = parsed_kib.get("MemTotal")
    available_kib = parsed_kib.get("MemAvailable")
    swap_total_kib = parsed_kib.get("SwapTotal")

    return {
        "mem_total_gib": (
            round(total_kib / 1024 / 1024, 3)
            if total_kib is not None else None
        ),
        "mem_available_gib": (
            round(available_kib / 1024 / 1024, 3)
            if available_kib is not None else None
        ),
        "swap_total_gib": (
            round(swap_total_kib / 1024 / 1024, 3)
            if swap_total_kib is not None else None
        ),
    }


def get_cpu_model() -> Optional[str]:
    cpuinfo = read_text_if_exists(Path("/proc/cpuinfo"))
    for line in cpuinfo.splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def detect_wsl() -> Dict[str, Any]:
    proc_version = read_text_if_exists(Path("/proc/version")).strip()
    kernel_release = platform.release()
    combined = f"{proc_version} {kernel_release}".lower()

    is_wsl = (
        "microsoft" in combined
        or bool(os.environ.get("WSL_DISTRO_NAME"))
        or bool(os.environ.get("WSL_INTEROP"))
    )

    is_wsl2 = (
        is_wsl
        and (
            "wsl2" in combined
            or "microsoft-standard-wsl2" in combined
            or bool(os.environ.get("WSL_INTEROP"))
        )
    )

    return {
        "is_wsl": bool(is_wsl),
        "is_wsl2": bool(is_wsl2),
        "wsl_distro_name": os.environ.get("WSL_DISTRO_NAME"),
        "wsl_interop_present": bool(os.environ.get("WSL_INTEROP")),
        "proc_version": proc_version,
        "kernel_release": kernel_release,
    }


def collect_system_info() -> Dict[str, Any]:
    disk_usage = shutil.disk_usage(cfg.REPOSITORY_ROOT_DIR)

    info = {
        "timestamp_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "python_version_info": list(sys.version_info[:5]),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": get_cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "repository_disk_total_gib": round(bytes_to_gib(disk_usage.total), 3),
        "repository_disk_used_gib": round(bytes_to_gib(disk_usage.used), 3),
        "repository_disk_free_gib": round(bytes_to_gib(disk_usage.free), 3),
    }

    info.update(get_memory_info())
    info["wsl"] = detect_wsl()

    return info


def collect_package_versions() -> Tuple[list[Dict[str, Any]], list[str]]:
    """
    Record required/optional package availability and versions.

    Import names and distribution names are kept separately because, for
    example, scikit-learn imports as `sklearn` and pandas-ta as `pandas_ta`.
    """

    registry = [
        ("numpy", "numpy", True, "numerical computing"),
        ("pandas", "pandas", True, "tabular/time-series processing"),
        ("scipy", "scipy", True, "scientific/statistical computing"),
        ("sklearn", "scikit-learn", True, "classical ML and metrics"),
        ("torch", "torch", True, "deep-learning framework"),
        ("yfinance", "yfinance", True, "Yahoo Finance market-data acquisition"),
        ("pyarrow", "pyarrow", True, "Parquet storage"),
        ("joblib", "joblib", True, "model/scaler serialization"),
        ("matplotlib", "matplotlib", True, "publication graphics"),

        ("torchvision", "torchvision", False, "PyTorch ecosystem audit"),
        ("torchaudio", "torchaudio", False, "PyTorch ecosystem audit"),
        ("statsmodels", "statsmodels", False, "statistical diagnostics"),
        ("xgboost", "xgboost", False, "tree-based baseline"),
        ("lightgbm", "lightgbm", False, "LightGBM baseline"),
        ("optuna", "optuna", False, "optimization tooling if later required"),
        ("einops", "einops", False, "tensor transformations for modern models"),
        ("tqdm", "tqdm", False, "progress reporting"),
        ("psutil", "psutil", False, "system/process diagnostics"),
        ("ta", "ta", False, "technical indicators if retained"),
        ("pandas_ta", "pandas-ta", False, "alternative technical indicators"),
        ("packaging", "packaging", False, "version handling"),
        (
            "tensorflow",
            "tensorflow",
            False,
            "environment audit only; revised neural pipeline uses PyTorch",
        ),
    ]

    import importlib.util

    rows: list[Dict[str, Any]] = []
    missing_required: list[str] = []

    for import_name, distribution_name, required, purpose in registry:
        try:
            module_available = importlib.util.find_spec(import_name) is not None
        except Exception:
            module_available = False

        version = get_distribution_version(distribution_name)

        if required and not module_available:
            missing_required.append(distribution_name)

        rows.append(
            {
                "import_name": import_name,
                "distribution": distribution_name,
                "required": required,
                "purpose": purpose,
                "module_available": bool(module_available),
                "installed_distribution": version is not None,
                "version": version or "",
            }
        )

    return rows, missing_required


def collect_source_hashes() -> list[Dict[str, Any]]:
    """
    Hash active Step 00, frozen Step-00 snapshot, and current Step 01.

    Current Step 00 must exactly match the script snapshot frozen by Step 00.
    """

    current_step00 = Path(cfg.__file__).resolve()
    frozen_step00 = cfg.get_frozen_script_path().resolve()
    current_step01 = SCRIPT_FILE.resolve()

    rows: list[Dict[str, Any]] = []

    for role, path in (
        ("current_step00", current_step00),
        ("frozen_step00_snapshot", frozen_step00),
        ("current_step01", current_step01),
    ):
        if not path.exists():
            raise RuntimeError(f"Required source file is missing: {path}")

        rows.append(
            {
                "role": role,
                "file_name": path.name,
                "absolute_path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )

    current_sha = next(
        row["sha256"] for row in rows if row["role"] == "current_step00"
    )
    frozen_sha = next(
        row["sha256"] for row in rows if row["role"] == "frozen_step00_snapshot"
    )

    if current_sha != frozen_sha:
        raise RuntimeError(
            "Current step00_config.py differs from the frozen Step-00 script "
            "snapshot. Refusing to continue because source-code drift was detected."
        )

    return rows


def locate_nvidia_smi() -> Optional[str]:
    candidates = [
        shutil.which("nvidia-smi"),
        "/usr/lib/wsl/lib/nvidia-smi",
        "/usr/bin/nvidia-smi",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def nvidia_smi_audit() -> Dict[str, Any]:
    executable = locate_nvidia_smi()

    result: Dict[str, Any] = {
        "nvidia_smi_path": executable,
        "available": executable is not None,
        "gpu_query": None,
        "compute_process_query": None,
        "unexpected_compute_processes": [],
    }

    if executable is None:
        return result

    gpu_query = run_command(
        [
            executable,
            "--query-gpu=index,name,uuid,driver_version,"
            "memory.total,memory.used,memory.free,"
            "utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_seconds=30,
    )

    process_query = run_command(
        [
            executable,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_seconds=30,
    )

    result["gpu_query"] = gpu_query
    result["compute_process_query"] = process_query

    if process_query.get("ok"):
        unexpected: list[Dict[str, Any]] = []

        for raw_line in process_query.get("stdout", "").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue

            pid_text = parts[0]
            process_name = parts[1]
            used_memory = parts[2] if len(parts) >= 3 else None

            try:
                pid = int(pid_text)
            except ValueError:
                pid = None

            unexpected.append(
                {
                    "pid": pid,
                    "process_name": process_name,
                    "used_memory_mib": used_memory,
                    "raw": line,
                }
            )

        result["unexpected_compute_processes"] = unexpected

    return result


def configure_torch_determinism(torch: Any) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    torch.use_deterministic_algorithms(True, warn_only=False)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch.backends, "cuda"):
        matmul_backend = getattr(torch.backends.cuda, "matmul", None)
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = False

        for function_name, value in (
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_math_sdp", True),
        ):
            function = getattr(torch.backends.cuda, function_name, None)
            if callable(function):
                function(value)

    settings["deterministic_algorithms_enabled"] = bool(
        torch.are_deterministic_algorithms_enabled()
    )
    settings["cudnn_deterministic"] = bool(
        getattr(torch.backends.cudnn, "deterministic", False)
    )
    settings["cudnn_benchmark"] = bool(
        getattr(torch.backends.cudnn, "benchmark", False)
    )

    return settings


def deterministic_cuda_trial(torch: Any, device: Any) -> Dict[str, Any]:
    torch.manual_seed(GPU_SMOKE_SEED)
    torch.cuda.manual_seed_all(GPU_SMOKE_SEED)

    n = GPU_SMOKE_MATRIX_SIZE

    x_cpu = torch.arange(
        n * n,
        dtype=torch.float32,
    ).reshape(n, n)
    x_cpu = (x_cpu.remainder(257.0) - 128.0) / 128.0

    w_cpu = torch.arange(
        n * n,
        dtype=torch.float32,
    ).reshape(n, n)
    w_cpu = (w_cpu.remainder(251.0) - 125.0) / 125.0

    x = x_cpu.to(device=device)
    w = w_cpu.to(device=device).requires_grad_(True)

    start = time.perf_counter()

    output = torch.matmul(x, w)
    output = torch.tanh(output)
    loss = (output.square()).mean()

    loss.backward()
    torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    grad = w.grad.detach()

    result = {
        "loss": float(loss.detach().cpu().item()),
        "output_checksum": float(output.detach().double().sum().cpu().item()),
        "gradient_checksum": float(grad.double().sum().cpu().item()),
        "gradient_abs_max": float(grad.abs().max().cpu().item()),
        "elapsed_seconds": elapsed,
    }

    del x, w, output, loss, grad
    return result


def pytorch_cuda_audit() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "import_ok": False,
        "cuda_available": False,
        "smoke_test_pass": False,
        "exact_repeatability_pass": False,
        "error": None,
    }

    try:
        import torch
    except Exception as exc:
        result["error"] = f"torch import failed: {repr(exc)}"
        return result

    result["import_ok"] = True
    result["torch_version"] = str(torch.__version__)
    result["torch_compiled_cuda_version"] = str(torch.version.cuda)
    result["cudnn_version"] = (
        int(torch.backends.cudnn.version())
        if torch.backends.cudnn.version() is not None
        else None
    )
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_device_count"] = int(torch.cuda.device_count())

    try:
        result["torch_cuda_arch_list"] = list(torch.cuda.get_arch_list())
    except Exception as exc:
        result["torch_cuda_arch_list"] = []
        result["torch_cuda_arch_list_error"] = repr(exc)

    if not result["cuda_available"] or result["cuda_device_count"] < 1:
        result["error"] = "PyTorch reports no usable CUDA device."
        return result

    try:
        deterministic_settings = configure_torch_determinism(torch)
        result["deterministic_settings"] = deterministic_settings

        devices = []

        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "major": int(props.major),
                    "minor": int(props.minor),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gib": round(bytes_to_gib(props.total_memory), 3),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )

        result["devices"] = devices

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        free_before, total_before = torch.cuda.mem_get_info(device)
        result["memory_before_smoke_gib"] = {
            "free": round(bytes_to_gib(free_before), 3),
            "total": round(bytes_to_gib(total_before), 3),
        }

        trial_1 = deterministic_cuda_trial(torch, device)
        torch.cuda.synchronize(device)

        allocated_after_trial_1 = torch.cuda.max_memory_allocated(device)
        reserved_after_trial_1 = torch.cuda.max_memory_reserved(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        trial_2 = deterministic_cuda_trial(torch, device)
        torch.cuda.synchronize(device)

        allocated_after_trial_2 = torch.cuda.max_memory_allocated(device)
        reserved_after_trial_2 = torch.cuda.max_memory_reserved(device)

        exact_repeatability = (
            trial_1["loss"] == trial_2["loss"]
            and trial_1["output_checksum"] == trial_2["output_checksum"]
            and trial_1["gradient_checksum"] == trial_2["gradient_checksum"]
            and trial_1["gradient_abs_max"] == trial_2["gradient_abs_max"]
        )

        free_after, total_after = torch.cuda.mem_get_info(device)

        result["trial_1"] = trial_1
        result["trial_2"] = trial_2
        result["exact_repeatability_pass"] = bool(exact_repeatability)
        result["peak_memory_allocated_mib"] = round(
            bytes_to_mib(max(allocated_after_trial_1, allocated_after_trial_2)),
            3,
        )
        result["peak_memory_reserved_mib"] = round(
            bytes_to_mib(max(reserved_after_trial_1, reserved_after_trial_2)),
            3,
        )
        result["memory_after_smoke_gib"] = {
            "free": round(bytes_to_gib(free_after), 3),
            "total": round(bytes_to_gib(total_after), 3),
        }

        result["smoke_test_pass"] = bool(exact_repeatability)
        torch.cuda.empty_cache()

    except Exception as exc:
        result["error"] = repr(exc)
        result["smoke_test_pass"] = False

    return result


def capture_reproducibility_snapshots(
    nvidia_info: Dict[str, Any],
) -> Dict[str, Any]:

    snapshots: Dict[str, Any] = {}

    pip_result = run_command(
        [sys.executable, "-m", "pip", "freeze"],
        timeout_seconds=120,
    )
    pip_text = pip_result.get("stdout", "")
    if pip_result.get("stderr"):
        pip_text += "\n# STDERR\n" + pip_result["stderr"]
    atomic_write_text(PIP_FREEZE_TXT, pip_text)
    snapshots["pip_freeze"] = pip_result

    gpu_query = nvidia_info.get("gpu_query")
    if isinstance(gpu_query, dict):
        text = gpu_query.get("stdout", "")
        if gpu_query.get("stderr"):
            text += "\n# STDERR\n" + gpu_query["stderr"]
    else:
        text = "nvidia-smi unavailable.\n"
    atomic_write_text(NVIDIA_SMI_TXT, text)

    process_query = nvidia_info.get("compute_process_query")
    if isinstance(process_query, dict):
        text = process_query.get("stdout", "")
        if process_query.get("stderr"):
            text += "\n# STDERR\n" + process_query["stderr"]
        if not text.strip():
            text = "No compute processes reported by nvidia-smi before CUDA smoke test.\n"
    else:
        text = "nvidia-smi compute-process query unavailable.\n"
    atomic_write_text(NVIDIA_PROCESS_TXT, text)

    command_specs = [
        ("uname -a", ["uname", "-a"]),
        ("lscpu", ["lscpu"]),
        ("free -h", ["free", "-h"]),
        ("df -h repository", ["df", "-h", str(cfg.REPOSITORY_ROOT_DIR)]),
    ]

    chunks = []
    for title, command in command_specs:
        command_result = run_command(command, timeout_seconds=30)
        snapshots[title] = command_result

        chunks.append("=" * 100)
        chunks.append(title)
        chunks.append("=" * 100)
        chunks.append("$ " + " ".join(command))
        chunks.append(command_result.get("stdout", "").rstrip())
        if command_result.get("stderr"):
            chunks.append("STDERR:")
            chunks.append(command_result["stderr"].rstrip())
        chunks.append("")

    atomic_write_text(SYSTEM_COMMANDS_TXT, "\n".join(chunks) + "\n")

    conda_path = shutil.which("conda") or os.environ.get("CONDA_EXE")

    if conda_path:
        conda_explicit = run_command(
            [conda_path, "list", "--explicit"],
            timeout_seconds=120,
        )
        text = conda_explicit.get("stdout", "")
        if conda_explicit.get("stderr"):
            text += "\n# STDERR\n" + conda_explicit["stderr"]
        atomic_write_text(CONDA_EXPLICIT_TXT, text)
        snapshots["conda_list_explicit"] = conda_explicit

        conda_export = run_command(
            [conda_path, "env", "export", "--no-builds"],
            timeout_seconds=120,
        )
        text = conda_export.get("stdout", "")
        if conda_export.get("stderr"):
            text += "\n# STDERR\n" + conda_export["stderr"]
        atomic_write_text(CONDA_EXPORT_TXT, text)
        snapshots["conda_env_export_no_builds"] = conda_export
    else:
        atomic_write_text(
            CONDA_EXPLICIT_TXT,
            "conda executable was not found via PATH.\n",
        )
        atomic_write_text(
            CONDA_EXPORT_TXT,
            "conda executable was not found via PATH.\n",
        )
        snapshots["conda"] = {
            "available": False,
            "note": "conda executable was not found via PATH",
        }

    return snapshots


def evaluate_critical_checks(
    step00_info: Dict[str, Any],
    system_info: Dict[str, Any],
    missing_required_packages: list[str],
    nvidia_info: Dict[str, Any],
    torch_info: Dict[str, Any],
) -> Dict[str, Any]:

    checks: list[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    add(
        "step00_freeze_verified",
        step00_info.get("step00_freeze_status") == "VERIFIED_EXISTING",
        str(step00_info.get("step00_freeze_status")),
    )

    add(
        "final_test_locked",
        (
            step00_info.get("final_test_state") == "LOCKED_NOT_OPENED"
            and step00_info.get("final_test_opened") is False
            and step00_info.get("final_test_scored") is False
        ),
        str(step00_info.get("final_test_state")),
    )

    python_ok = sys.version_info[:2] >= MINIMUM_PYTHON
    add(
        "python_version_supported",
        python_ok,
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    add(
        "required_packages_present",
        len(missing_required_packages) == 0,
        (
            "all required packages present"
            if not missing_required_packages
            else "missing: " + ", ".join(missing_required_packages)
        ),
    )

    add(
        "nvidia_smi_available",
        bool(nvidia_info.get("available")),
        str(nvidia_info.get("nvidia_smi_path")),
    )

    process_query = nvidia_info.get("compute_process_query")
    process_guard_observable = (
        isinstance(process_query, dict)
        and bool(process_query.get("ok"))
    )

    unexpected_processes = nvidia_info.get(
        "unexpected_compute_processes",
        [],
    )

    if process_guard_observable:
        add(
            "no_concurrent_cuda_compute_processes",
            len(unexpected_processes) == 0,
            (
                "none detected"
                if len(unexpected_processes) == 0
                else f"{len(unexpected_processes)} external compute process(es) detected"
            ),
        )
    else:
        checks.append(
            {
                "name": "no_concurrent_cuda_compute_processes",
                "passed": True,
                "detail": (
                    "nvidia-smi process enumeration unavailable; "
                    "manual process guard remains required before heavy training"
                ),
                "warning": True,
            }
        )

    add(
        "torch_import",
        bool(torch_info.get("import_ok")),
        str(torch_info.get("torch_version")),
    )

    add(
        "torch_cuda_available",
        bool(torch_info.get("cuda_available")),
        f"device_count={torch_info.get('cuda_device_count')}",
    )

    add(
        "deterministic_fp32_cuda_smoke",
        bool(torch_info.get("smoke_test_pass")),
        (
            "exact repeatability PASS"
            if torch_info.get("exact_repeatability_pass")
            else str(torch_info.get("error"))
        ),
    )

    wsl = system_info.get("wsl", {})
    checks.append(
        {
            "name": "wsl2_detected",
            "passed": bool(wsl.get("is_wsl2")),
            "detail": (
                f"distro={wsl.get('wsl_distro_name')}, "
                f"kernel={wsl.get('kernel_release')}"
            ),
            "warning": not bool(wsl.get("is_wsl2")),
        }
    )

    critical_failures = [
        item
        for item in checks
        if not item.get("passed", False)
        and not item.get("warning", False)
    ]

    warnings = [
        item
        for item in checks
        if item.get("warning", False)
    ]

    return {
        "checks": checks,
        "critical_failure_count": len(critical_failures),
        "warning_count": len(warnings),
        "critical_failures": critical_failures,
        "warnings": warnings,
        "overall_pass": len(critical_failures) == 0,
    }


def write_failed_attempt(
    payload: Dict[str, Any],
) -> Path:
    FAILED_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = FAILED_AUDIT_DIR / f"01_environment_audit_FAILED_{timestamp}.json"
    atomic_write_json(path, payload)
    return path


def artifact_entry(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "relative_to_experiment": str(path.relative_to(cfg.EXPERIMENT_ROOT_DIR)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def freeze_successful_audit(
    step00_info: Dict[str, Any],
    audit_payload: Dict[str, Any],
    package_rows: list[Dict[str, Any]],
) -> Tuple[str, str]:

    if STEP01_MANIFEST.exists():
        raise RuntimeError(
            "Step-01 manifest already exists. Refusing to overwrite it. "
            "Run with --verify-only instead."
        )

    if STEP01_FROZEN_SCRIPT.exists():
        raise RuntimeError(
            "Frozen Step-01 script already exists unexpectedly. "
            "Refusing to overwrite."
        )

    cfg.ENVIRONMENT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    atomic_write_json(ENV_AUDIT_JSON, audit_payload)

    atomic_write_csv(
        PACKAGE_CSV,
        package_rows,
        fieldnames=[
            "import_name",
            "distribution",
            "required",
            "purpose",
            "module_available",
            "installed_distribution",
            "version",
        ],
    )

    source_rows = collect_source_hashes()
    atomic_write_csv(
        SOURCE_HASH_CSV,
        source_rows,
        fieldnames=[
            "role",
            "file_name",
            "absolute_path",
            "size_bytes",
            "sha256",
        ],
    )

    shutil.copy2(SCRIPT_FILE, STEP01_FROZEN_SCRIPT)

    artifact_paths = [
        ENV_AUDIT_JSON,
        PACKAGE_CSV,
        SOURCE_HASH_CSV,
        PIP_FREEZE_TXT,
        NVIDIA_SMI_TXT,
        NVIDIA_PROCESS_TXT,
        SYSTEM_COMMANDS_TXT,
        CONDA_EXPLICIT_TXT,
        CONDA_EXPORT_TXT,
        STEP01_FROZEN_SCRIPT,
    ]

    artifacts = {
        path.name: artifact_entry(path)
        for path in artifact_paths
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP_ID,
        "step_name": STEP_NAME,
        "status": "PASS",
        "frozen_at_utc": utc_now_iso(),
        "experiment_tag": cfg.EXPERIMENT_TAG,
        "experiment_root": str(cfg.EXPERIMENT_ROOT_DIR),
        "configuration_fingerprint_sha256": step00_info[
            "configuration_fingerprint_sha256"
        ],
        "step00_manifest_sha256": step00_info["step00_manifest_sha256"],
        "final_test_state": step00_info["final_test_state"],
        "final_test_opened": False,
        "final_test_scored": False,
        "artifacts": artifacts,
    }

    atomic_write_json(STEP01_MANIFEST, manifest)
    manifest_sha256 = sha256_file(STEP01_MANIFEST)

    atomic_write_text(
        STEP01_MANIFEST_SHA,
        f"{manifest_sha256}  {STEP01_MANIFEST.name}\n",
    )

    summary_lines = [
        "STEP 01 — ENVIRONMENT / CUDA / WSL2 REPRODUCIBILITY AUDIT",
        "=" * 65,
        "Status: PASS",
        f"Experiment tag: {cfg.EXPERIMENT_TAG}",
        f"Configuration fingerprint: {step00_info['configuration_fingerprint_sha256']}",
        f"Final-test state: {step00_info['final_test_state']}",
        f"Python: {audit_payload['system']['python_version']}",
        f"Conda environment: {audit_payload['system'].get('conda_default_env')}",
        f"WSL2 detected: {audit_payload['system']['wsl'].get('is_wsl2')}",
        f"PyTorch: {audit_payload['pytorch_cuda'].get('torch_version')}",
        f"PyTorch CUDA build: {audit_payload['pytorch_cuda'].get('torch_compiled_cuda_version')}",
        f"CUDA available: {audit_payload['pytorch_cuda'].get('cuda_available')}",
        f"CUDA device count: {audit_payload['pytorch_cuda'].get('cuda_device_count')}",
    ]

    devices = audit_payload["pytorch_cuda"].get("devices", [])
    if devices:
        summary_lines.append(
            "GPU 0: "
            f"{devices[0].get('name')} | "
            f"{devices[0].get('total_memory_gib')} GiB | "
            f"CC {devices[0].get('compute_capability')}"
        )

    summary_lines.extend(
        [
            "Deterministic FP32 CUDA smoke: "
            f"{audit_payload['pytorch_cuda'].get('smoke_test_pass')}",
            "Exact repeatability: "
            f"{audit_payload['pytorch_cuda'].get('exact_repeatability_pass')}",
            "Peak allocated during smoke: "
            f"{audit_payload['pytorch_cuda'].get('peak_memory_allocated_mib')} MiB",
            "Peak reserved during smoke: "
            f"{audit_payload['pytorch_cuda'].get('peak_memory_reserved_mib')} MiB",
            f"Step-01 manifest SHA-256: {manifest_sha256}",
        ]
    )

    atomic_write_text(
        STEP01_SUMMARY,
        "\n".join(summary_lines) + "\n",
    )

    return manifest_sha256, sha256_file(STEP01_FROZEN_SCRIPT)


def verify_existing_step01() -> int:
    print("=" * 108)
    print("STEP 01 — VERIFY EXISTING ENVIRONMENT AUDIT FREEZE")
    print("=" * 108)

    step00_info = verify_step00_contract()

    if not STEP01_MANIFEST.exists():
        print("[FAIL] Step-01 manifest is missing.")
        return 20

    if not STEP01_MANIFEST_SHA.exists():
        print("[FAIL] Step-01 manifest SHA sidecar is missing.")
        return 21

    with STEP01_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("status") != "PASS":
        print("[FAIL] Stored Step-01 status is not PASS.")
        return 22

    if (
        manifest.get("configuration_fingerprint_sha256")
        != step00_info["configuration_fingerprint_sha256"]
    ):
        print("[FAIL] Step-01 / Step-00 configuration fingerprint mismatch.")
        return 23

    if manifest.get("final_test_state") != "LOCKED_NOT_OPENED":
        print("[FAIL] Stored Step-01 final-test state is unsafe.")
        return 24

    expected_manifest_sha = (
        STEP01_MANIFEST_SHA.read_text(encoding="utf-8").strip().split()[0]
    )
    actual_manifest_sha = sha256_file(STEP01_MANIFEST)

    if expected_manifest_sha != actual_manifest_sha:
        print("[FAIL] Step-01 manifest SHA mismatch.")
        print(f"Expected: {expected_manifest_sha}")
        print(f"Actual  : {actual_manifest_sha}")
        return 25

    artifact_failures = 0

    for name, entry in manifest.get("artifacts", {}).items():
        path = Path(entry["path"])
        expected = entry["sha256"]

        if not path.exists():
            print(f"[FAIL] Missing artifact: {name}")
            artifact_failures += 1
            continue

        actual = sha256_file(path)

        if actual != expected:
            print(f"[FAIL] Hash mismatch: {name}")
            artifact_failures += 1
        else:
            print(f"[PASS] {name}")

    if artifact_failures:
        print("=" * 108)
        print("STEP 01 VERIFICATION STATUS: FAIL")
        print("=" * 108)
        return 26

    print("-" * 108)
    print(f"Configuration fingerprint : {step00_info['configuration_fingerprint_sha256']}")
    print(f"Step-01 manifest SHA-256  : {actual_manifest_sha}")
    print("Final-test state           : LOCKED_NOT_OPENED")
    print("=" * 108)
    print("STEP 01 VERIFICATION STATUS: PASS")
    print("=" * 108)
    return 0


def run_new_audit() -> int:
    print("=" * 108)
    print("STEP 01 — ENVIRONMENT, CUDA, WSL2, PACKAGE AND REPRODUCIBILITY AUDIT")
    print("=" * 108)

    if STEP01_MANIFEST.exists():
        print("[ABORT] A frozen Step-01 manifest already exists.")
        print("Run:")
        print("  python step01_environment_audit.py --verify-only")
        return 10

    print("[1/8] Verifying immutable Step-00 configuration and final-test lock...")
    try:
        step00_info = verify_step00_contract()
    except Exception as exc:
        print(f"[FAIL] Step-00 verification failed: {exc}")
        return 11

    print(
        "      Configuration fingerprint: "
        f"{step00_info['configuration_fingerprint_sha256']}"
    )
    print(f"      Final-test state: {step00_info['final_test_state']}")

    print("[2/8] Auditing Python, Conda, Linux/WSL, CPU, RAM and disk...")
    system_info = collect_system_info()

    print("[3/8] Auditing required and optional Python packages...")
    package_rows, missing_required_packages = collect_package_versions()

    print("[4/8] Querying NVIDIA driver/GPU and checking concurrent CUDA processes...")
    nvidia_info = nvidia_smi_audit()

    unexpected_processes = nvidia_info.get(
        "unexpected_compute_processes",
        [],
    )

    if unexpected_processes:
        print("      WARNING: external CUDA compute process(es) detected:")
        for item in unexpected_processes:
            print(
                "        "
                f"PID={item.get('pid')} | "
                f"{item.get('process_name')} | "
                f"used_memory={item.get('used_memory_mib')}"
            )
    else:
        process_query = nvidia_info.get("compute_process_query")
        if isinstance(process_query, dict) and process_query.get("ok"):
            print("      No external CUDA compute processes detected.")
        else:
            print(
                "      WARNING: nvidia-smi could not reliably enumerate "
                "compute processes on this host."
            )

    print("[5/8] Running deterministic FP32 PyTorch/CUDA smoke test...")
    torch_info = pytorch_cuda_audit()

    print("[6/8] Capturing pip/Conda/system reproducibility snapshots...")
    snapshot_info = capture_reproducibility_snapshots(
        nvidia_info=nvidia_info,
    )

    print("[7/8] Evaluating environment contract...")
    check_result = evaluate_critical_checks(
        step00_info=step00_info,
        system_info=system_info,
        missing_required_packages=missing_required_packages,
        nvidia_info=nvidia_info,
        torch_info=torch_info,
    )

    source_rows = collect_source_hashes()

    audit_payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "step": STEP_ID,
        "step_name": STEP_NAME,
        "audit_generated_utc": utc_now_iso(),
        "experiment_tag": cfg.EXPERIMENT_TAG,
        "experiment_root": str(cfg.EXPERIMENT_ROOT_DIR),
        "step00": step00_info,
        "system": system_info,
        "packages": package_rows,
        "missing_required_packages": missing_required_packages,
        "source_code_hashes": source_rows,
        "nvidia_smi": nvidia_info,
        "pytorch_cuda": torch_info,
        "snapshot_command_status": snapshot_info,
        "contract_evaluation": check_result,
        "scientific_data_access": {
            "raw_market_data_opened": False,
            "development_arrays_opened": False,
            "validation_arrays_opened": False,
            "final_test_arrays_opened": False,
            "final_test_predictions_generated": False,
        },
    }

    for check in check_result["checks"]:
        prefix = "PASS" if check.get("passed") else "FAIL"
        if check.get("warning"):
            prefix = "WARN"
        print(
            f"      [{prefix}] "
            f"{check['name']}: {check['detail']}"
        )

    if not check_result["overall_pass"]:
        failed_path = write_failed_attempt(audit_payload)

        print()
        print("=" * 108)
        print("STEP 01 STATUS: FAIL — ENVIRONMENT NOT FROZEN")
        print("=" * 108)
        print(f"Failure diagnostic: {failed_path}")
        print(
            "Correct the reported critical issue(s), then rerun this same "
            "Step-01 script. No successful Step-01 manifest was frozen."
        )
        return 30

    print("[8/8] Freezing successful Step-01 environment audit...")

    manifest_sha256, frozen_script_sha256 = freeze_successful_audit(
        step00_info=step00_info,
        audit_payload=audit_payload,
        package_rows=package_rows,
    )

    devices = torch_info.get("devices", [])
    gpu0 = devices[0] if devices else {}

    print()
    print("-" * 108)
    print(f"Python executable             : {system_info['python_executable']}")
    print(
        "Python version                : "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(f"Conda environment             : {system_info.get('conda_default_env')}")
    print(
        "WSL / WSL2                    : "
        f"{system_info['wsl'].get('is_wsl')} / "
        f"{system_info['wsl'].get('is_wsl2')}"
    )
    print(f"CPU                           : {system_info.get('cpu_model')}")
    print(f"Logical CPU count             : {system_info.get('logical_cpu_count')}")
    print(f"RAM total                     : {system_info.get('mem_total_gib')} GiB")
    print(f"Repository disk free          : {system_info.get('repository_disk_free_gib')} GiB")
    print("-" * 108)
    print(f"PyTorch                       : {torch_info.get('torch_version')}")
    print(f"PyTorch CUDA build            : {torch_info.get('torch_compiled_cuda_version')}")
    print(f"cuDNN                         : {torch_info.get('cudnn_version')}")
    print(f"CUDA available                : {torch_info.get('cuda_available')}")
    print(f"CUDA device count             : {torch_info.get('cuda_device_count')}")
    print(f"GPU 0                         : {gpu0.get('name')}")
    print(f"GPU 0 compute capability      : {gpu0.get('compute_capability')}")
    print(f"GPU 0 total memory            : {gpu0.get('total_memory_gib')} GiB")
    print(
        "Smoke peak allocated          : "
        f"{torch_info.get('peak_memory_allocated_mib')} MiB"
    )
    print(
        "Smoke peak reserved           : "
        f"{torch_info.get('peak_memory_reserved_mib')} MiB"
    )
    print(
        "Deterministic FP32 smoke      : "
        f"{torch_info.get('smoke_test_pass')}"
    )
    print(
        "Exact repeatability           : "
        f"{torch_info.get('exact_repeatability_pass')}"
    )
    print("-" * 108)
    print(
        "Configuration SHA-256         : "
        f"{step00_info['configuration_fingerprint_sha256']}"
    )
    print(f"Step-01 frozen script SHA-256 : {frozen_script_sha256}")
    print(f"Step-01 manifest SHA-256      : {manifest_sha256}")
    print(f"Environment audit JSON        : {ENV_AUDIT_JSON}")
    print(f"Package inventory             : {PACKAGE_CSV}")
    print(f"Source-code hashes            : {SOURCE_HASH_CSV}")
    print(f"pip freeze                    : {PIP_FREEZE_TXT}")
    print(f"Final-test state              : {step00_info['final_test_state']}")
    print("=" * 108)
    print("PASS: Step 01 environment audit completed and frozen successfully.")
    print("PASS: Step-00 configuration remains unchanged.")
    print("PASS: Deterministic FP32 CUDA execution is operational.")
    print("PASS: No scientific market/test data were opened by Step 01.")
    print("PASS: Final held-out test remains LOCKED_NOT_OPENED.")
    print("=" * 108)
    print("Do not run the next step yet. Paste the complete terminal output for audit.")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and freeze the clean experiment software/hardware "
            "environment without opening market or final-test data."
        )
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the already frozen Step-01 audit without recreating it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verify_only:
        return verify_existing_step01()

    return run_new_audit()


if __name__ == "__main__":
    sys.exit(main())
