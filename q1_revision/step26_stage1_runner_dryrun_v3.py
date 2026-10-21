#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 26 — CLEAN V3 STAGE-1 RUNNER IMPLEMENTATION + SYNTHETIC DRY-RUN AUDIT

Purpose
-------
Audit the ACTUAL reusable Stage-1 runner path after Steps 00-25 are frozen and
before launching any of the 80 development fits.

This step does not open numerical development arrays and does not inspect
validation performance.  In one isolated subprocess per candidate it exercises:

  * the frozen Step-24 official-source model construction/adaptation path;
  * the candidate-specific frozen Step-24 physical microbatch;
  * the Step-25 logical-batch engine with exact sample-weighted accumulation;
  * one synthetic AdamW update at the candidate-specific frozen learning rate;
  * gradient clipping exactly once per logical batch;
  * sequential no-grad validation with exact sample-weighted BCE;
  * sigmoid probabilities and the five prespecified Stage-1 metrics;
  * checkpoint save -> deliberate parameter perturbation -> reload -> prediction
    equivalence;
  * train-mode forward compatibility for every physical tail microbatch size
    that can actually occur in the five frozen development folds;
  * repository cleanliness after execution.

The dry-run logical batch is intentionally incomplete whenever possible and
uses an ACTUAL tail size implied by the frozen Step-25 candidate/fold batch
plan.  This tests the future Stage-1 implementation without fitting any model
to real development observations.

The script is intentionally reusable: Step 27 should import the runner helpers
defined here rather than reimplementing the training engine.

Run:
    python step26_stage1_runner_dryrun_v3.py

Verify after an audited PASS:
    python step26_stage1_runner_dryrun_v3.py --verify-only
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

import step00_config as cfg


# =============================================================================
# 1. FROZEN IDENTIFIERS / CONSTANTS
# =============================================================================

STEP_NUMBER = 26
STEP_NAME = "STEP 26 — STAGE-1 RUNNER IMPLEMENTATION + SYNTHETIC DRY-RUN AUDIT"
SCHEMA_VERSION = "1.0"

EXPECTED_CONFIGURATION_FINGERPRINT = (
    "9675cb42e5544d9a3cda235208b33cc4c27f7583531535505b42056aa86b04dd"
)
EXPECTED_STEP24_FINGERPRINT = (
    "3040ed0cce6b860f9cfefa9464b509bfdba4431a7bcab370f1ab71e8983865eb"
)
EXPECTED_STEP24_MANIFEST_SHA256 = (
    "405a294dd9804527d4b116b4a9a9ca0fec8af0d63b15a384abd5cfb8a5517b04"
)
EXPECTED_STEP25_FINGERPRINT = (
    "058e69bd9fba9f7af1d2d51a9f5674e35c13380783b9f9d20d3161a22b7ee67a"
)
EXPECTED_STEP25_MANIFEST_SHA256 = (
    "1a903e32e2f22cfe73dcf6bade48a28d098169f967aff279eb25426bd96218ef"
)

EXPECTED_CANDIDATES = 16
EXPECTED_FOLDS = 5
EXPECTED_FITS = 80

SEED = 42
LOOKBACK = 60
FEATURES = 63
PRED_LEN = 5
OUTPUTS = 3
LOGICAL_BATCH = 256

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
THRESHOLD = 0.50

CHECKPOINT_RELOAD_LOGIT_TOL = 1e-6
WORKER_TIMEOUT_SECONDS = 1800
RESULT_PREFIX = "__STEP26_RESULT__"

MODEL_ORDER = (
    "DLINEAR", "NLINEAR", "PATCHTST", "TIMESNET",
    "ITRANSFORMER", "FREDFORMER", "FRNET", "MLF",
)

SCRIPT = Path(__file__).resolve()
Q1_DIR = SCRIPT.parent

STEP24_SCRIPT = Q1_DIR / "step24_development_training_preflight_v3.py"
STEP25_SCRIPT = Q1_DIR / "step25_stage1_training_engine_contract_v3.py"

STEP24_FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step24_development_training_preflight_frozen.py"
STEP25_FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step25_stage1_training_engine_contract_frozen.py"

STEP24_FREEZE = cfg.MANIFEST_DIR / "24_training_runtime_preflight_freeze.json"
STEP24_MANIFEST = cfg.MANIFEST_DIR / "24_training_runtime_preflight_manifest.json"
STEP24_MANIFEST_SHA = cfg.MANIFEST_DIR / "24_training_runtime_preflight_manifest.sha256"

STEP25_FREEZE = cfg.MANIFEST_DIR / "25_stage1_training_engine_contract_freeze.json"
STEP25_MANIFEST = cfg.MANIFEST_DIR / "25_stage1_training_engine_contract_manifest.json"
STEP25_MANIFEST_SHA = cfg.MANIFEST_DIR / "25_stage1_training_engine_contract_manifest.sha256"
STEP25_BATCH_PLAN = cfg.AUDIT_DIR / "25_stage1_fold_batch_plan.csv"

STEP10_TEST_SEAL = cfg.MANIFEST_DIR / "10_final_test_seal.json"
FORBIDDEN_TEST = "10_final_test_locked_panel.parquet"

RESULTS_CSV = cfg.AUDIT_DIR / "26_stage1_runner_synthetic_dryrun.csv"
TAIL_CSV = cfg.AUDIT_DIR / "26_stage1_tail_microbatch_compatibility.csv"
ARTIFACT_SCHEMA_JSON = cfg.AUDIT_DIR / "26_stage1_fit_artifact_schema.json"
SOURCE_HASH_CSV = cfg.AUDIT_DIR / "26_stage1_runner_source_hashes.csv"
WORKER_LOG_DIR = cfg.AUDIT_DIR / "26_stage1_runner_worker_logs"

FREEZE_JSON = cfg.MANIFEST_DIR / "26_stage1_runner_dryrun_freeze.json"
MANIFEST_JSON = cfg.MANIFEST_DIR / "26_stage1_runner_dryrun_manifest.json"
MANIFEST_SHA = cfg.MANIFEST_DIR / "26_stage1_runner_dryrun_manifest.sha256"
FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step26_stage1_runner_dryrun_frozen.py"
SUMMARY_TXT = cfg.REPORT_DIR / "26_stage1_runner_dryrun_summary.txt"
FAILED_DIR = cfg.AUDIT_DIR / "step26_failed_attempts"


# =============================================================================
# 2. GENERIC HELPERS
# =============================================================================

def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON missing:\n{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)


def semantic_hash(payload: Any) -> str:
    return cfg.canonical_json_sha256(payload)


def verify_sidecar(manifest: Path, sidecar: Path, label: str) -> str:
    if not manifest.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"{label} manifest/sidecar missing.")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    observed = sha256_file(manifest)
    if expected != observed:
        raise RuntimeError(
            f"{label} manifest SHA mismatch.\n"
            f"Expected: {expected}\nObserved: {observed}"
        )
    return observed


def recompute_freeze_fp(payload: dict, field: str) -> str:
    work = dict(payload)
    work.pop(field, None)
    work.pop("generated_at_utc", None)
    return semantic_hash(work)


def canonical_outputs() -> tuple[Path, ...]:
    return (
        RESULTS_CSV,
        TAIL_CSV,
        ARTIFACT_SCHEMA_JSON,
        SOURCE_HASH_CSV,
        FREEZE_JSON,
        MANIFEST_JSON,
        MANIFEST_SHA,
        FROZEN_SCRIPT,
        SUMMARY_TXT,
    )


def assert_fresh_target() -> None:
    existing = [p for p in canonical_outputs() if p.exists()]
    if existing:
        raise RuntimeError(
            "Step-26 canonical output(s) already exist; refusing overwrite:\n"
            + "\n".join(str(p) for p in existing)
            + "\nUse --verify-only after a successful frozen run."
        )


def archive_partial() -> Path | None:
    if MANIFEST_JSON.exists():
        return None
    existing = [p for p in canonical_outputs() if p.exists()]
    if not existing and not WORKER_LOG_DIR.exists():
        return None

    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    dst = FAILED_DIR / (
        "partial_artifacts_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    dst.mkdir()

    for p in existing:
        if p.exists():
            try:
                shutil.move(str(p), str(dst / p.name))
            except Exception:
                pass

    if WORKER_LOG_DIR.exists():
        try:
            shutil.move(str(WORKER_LOG_DIR), str(dst / WORKER_LOG_DIR.name))
        except Exception:
            pass

    return dst


# =============================================================================
# 3. UPSTREAM VERIFICATION
# =============================================================================

def import_step24():
    import step24_development_training_preflight_v3 as s24
    return s24


def import_step25():
    import step25_stage1_training_engine_contract_v3 as s25
    return s25


def verify_current_script_matches_frozen(current: Path, frozen: Path, label: str) -> str:
    if not current.is_file() or not frozen.is_file():
        raise FileNotFoundError(f"{label} current/frozen script missing.")
    current_sha = sha256_file(current)
    frozen_sha = sha256_file(frozen)
    if current_sha != frozen_sha:
        raise RuntimeError(
            f"{label} working script differs from its frozen copy.\n"
            f"Working: {current_sha}\nFrozen : {frozen_sha}"
        )
    return current_sha


def verify_upstream() -> dict:
    s25 = import_step25()

    upstream25 = s25.verify_upstream()

    if upstream25["step00"]["fingerprint"] != EXPECTED_CONFIGURATION_FINGERPRINT:
        raise RuntimeError("Step-00 configuration fingerprint drift.")

    step24_manifest_sha = verify_sidecar(
        STEP24_MANIFEST, STEP24_MANIFEST_SHA, "Step 24"
    )
    if step24_manifest_sha != EXPECTED_STEP24_MANIFEST_SHA256:
        raise RuntimeError("Unexpected Step-24 manifest SHA.")

    step25_manifest_sha = verify_sidecar(
        STEP25_MANIFEST, STEP25_MANIFEST_SHA, "Step 25"
    )
    if step25_manifest_sha != EXPECTED_STEP25_MANIFEST_SHA256:
        raise RuntimeError(
            "Unexpected Step-25 manifest SHA.\n"
            f"Expected: {EXPECTED_STEP25_MANIFEST_SHA256}\n"
            f"Observed: {step25_manifest_sha}"
        )

    step24 = load_json(STEP24_FREEZE)
    if (
        step24["training_runtime_preflight_fingerprint_sha256"]
        != EXPECTED_STEP24_FINGERPRINT
    ):
        raise RuntimeError("Unexpected Step-24 preflight fingerprint.")
    if (
        recompute_freeze_fp(
            step24, "training_runtime_preflight_fingerprint_sha256"
        )
        != EXPECTED_STEP24_FINGERPRINT
    ):
        raise RuntimeError("Step-24 preflight fingerprint no longer recomputes.")

    step25 = load_json(STEP25_FREEZE)
    if (
        step25["stage1_training_engine_fingerprint_sha256"]
        != EXPECTED_STEP25_FINGERPRINT
    ):
        raise RuntimeError("Unexpected Step-25 training-engine fingerprint.")
    if (
        recompute_freeze_fp(
            step25, "stage1_training_engine_fingerprint_sha256"
        )
        != EXPECTED_STEP25_FINGERPRINT
    ):
        raise RuntimeError("Step-25 training-engine fingerprint no longer recomputes.")

    if step25["status"] != "STAGE1_TRAINING_ENGINE_CONTRACT_FROZEN":
        raise RuntimeError("Step-25 freeze status drift.")

    scope = step25["stage1_scope"]
    if (
        int(scope["candidate_configurations"]) != EXPECTED_CANDIDATES
        or int(scope["chronological_folds"]) != EXPECTED_FOLDS
        or int(scope["total_fits"]) != EXPECTED_FITS
        or int(scope["seed"]) != SEED
    ):
        raise RuntimeError("Step-25 Stage-1 scope drift.")

    engine = step25["training_engine"]
    expected_engine = {
        "logical_batch_size": LOGICAL_BATCH,
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "loss": "BCEWithLogitsLoss(reduction='mean')",
        "class_weighting": "none",
        "gradient_clip_norm": GRAD_CLIP,
        "dtype": "float32",
        "amp_enabled": False,
        "tf32_enabled": False,
        "scheduler": "none",
        "zero_grad_policy": "once before each logical batch",
        "accumulation_loss_weight": "micro_n / logical_n",
        "gradient_clip_policy": "once after complete logical-batch accumulation",
        "optimizer_step_policy": "one AdamW step per logical batch",
    }
    for key, expected in expected_engine.items():
        if engine.get(key) != expected:
            raise RuntimeError(
                f"Step-25 training-engine drift: {key}: "
                f"{engine.get(key)!r} != {expected!r}"
            )

    if step25["data_ordering"]["epoch_permutation"] != (
        "numpy.random.SeedSequence([42, fold, epoch])"
    ):
        raise RuntimeError("Step-25 shuffle-seed contract drift.")
    if step25["data_ordering"]["candidate_id_participates_in_permutation_seed"]:
        raise RuntimeError("Candidate ID unexpectedly participates in shuffle seed.")

    stop = step25["stopping_checkpointing"]
    if (
        stop["metric"] != "validation_mean_directional_bce"
        or stop["mode"] != "min"
        or not bool(stop["strict_improvement"])
        or float(stop["min_delta"]) != 0.0
        or int(stop["max_epochs"]) != 100
        or int(stop["minimum_epoch_before_stopping"]) != 10
        or int(stop["patience"]) != 12
        or not bool(stop["restore_best_checkpoint_before_metrics"])
    ):
        raise RuntimeError("Step-25 checkpoint/early-stopping contract drift.")

    if float(step25["stage1_prediction"]["threshold"]) != THRESHOLD:
        raise RuntimeError("Step-25 Stage-1 threshold drift.")
    if step25["stage1_prediction"]["threshold_optimized_during_stage1"]:
        raise RuntimeError("Step-25 unexpectedly permits threshold tuning.")

    protection = step25["final_test_protection"]
    if (
        protection["step10_seal_status"] != "SEALED_NOT_OPENED"
        or protection["sealed_test_parquet_opened_or_hashed_by_step25"]
        or protection["global_final_test_state"] != "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError("Step-25 final-test protection drift.")

    seal = load_json(STEP10_TEST_SEAL)
    if (
        seal.get("status") != "SEALED_NOT_OPENED"
        or seal.get("global_final_test_lock_state") != "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError("Step-10 final-test seal is not pristine.")

    if not STEP25_BATCH_PLAN.is_file():
        raise FileNotFoundError(STEP25_BATCH_PLAN)

    step25_manifest = load_json(STEP25_MANIFEST)
    plan_record = step25_manifest["artifacts"].get(STEP25_BATCH_PLAN.name)
    if not plan_record:
        raise RuntimeError("Step-25 batch-plan artifact is not registered.")
    if sha256_file(STEP25_BATCH_PLAN) != plan_record["sha256"]:
        raise RuntimeError("Step-25 batch-plan artifact hash drift.")

    step24_script_sha = verify_current_script_matches_frozen(
        STEP24_SCRIPT, STEP24_FROZEN_SCRIPT, "Step 24"
    )
    step25_script_sha = verify_current_script_matches_frozen(
        STEP25_SCRIPT, STEP25_FROZEN_SCRIPT, "Step 25"
    )

    schedule = step24["candidate_microbatch_schedule"]
    if len(schedule) != EXPECTED_CANDIDATES:
        raise RuntimeError("Step-24 schedule does not contain 16 candidates.")

    return {
        "s25": s25,
        "upstream25": upstream25,
        "step24": step24,
        "step25": step25,
        "step24_manifest_sha256": step24_manifest_sha,
        "step25_manifest_sha256": step25_manifest_sha,
        "step24_script_sha256": step24_script_sha,
        "step25_script_sha256": step25_script_sha,
        "schedule": schedule,
        "test_seal_status": seal["status"],
        "global_final_test_state": protection["global_final_test_state"],
        "global_lock_path": Path(upstream25["step00"]["lock_path"]),
        "global_lock_sha256": upstream25["step00"]["lock_sha256"],
    }


# =============================================================================
# 4. FROZEN DATA-ORDER / DETERMINISM HELPERS — REUSED BY STEP 27
# =============================================================================

def set_training_determinism(seed: int = SEED) -> None:
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

    if hasattr(torch.backends, "cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def epoch_permutation(n: int, fold: int, epoch: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    if fold < 1 or fold > EXPECTED_FOLDS:
        raise ValueError("fold must be in 1..5")
    if epoch < 1:
        raise ValueError("epoch must be 1-indexed")

    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, int(fold), int(epoch)])
    )
    return rng.permutation(n)


# =============================================================================
# 5. CANDIDATE MODEL CONSTRUCTION — EXACT STEP-24 SOURCE PATH
# =============================================================================

def build_candidate_model(
    model_id: str,
    config_id: str,
    physical_microbatch: int,
    device,
):
    import torch
    import torch.nn as nn

    s24 = import_step24()

    (
        _s23,
        _s0,
        upstream,
        _m23,
        _m23_sha,
        _f23,
        _fp23,
        f22,
        _fp22,
    ) = s24.verify_all()

    repository_map = {
        r["repo_id"]: r
        for r in upstream["freeze17"]["repositories"]
    }
    model_map = s24.step20_model_map(upstream["freeze20"])

    matches = [
        r for r in f22["candidate_source_configurations"]
        if r["model_id"] == model_id and r["config_id"] == config_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Candidate source-config lookup failure: {model_id}/{config_id}"
        )

    record = matches[0]
    repository = repository_map[record["repo_id"]]
    s24.assert_single_repository_clean(repository)

    source_config = copy.deepcopy(record["source_config"])
    runtime_bindings = {}

    if source_config.get("device") == "__RUNTIME_DEVICE__":
        source_config["device"] = device
        runtime_bindings["device"] = str(device)

    if model_id == "MLF" and "bs" in source_config:
        runtime_bindings["bs_original_frozen_value"] = source_config["bs"]
        source_config["bs"] = int(physical_microbatch)
        runtime_bindings["bs_effective_runtime_value"] = int(physical_microbatch)

    target, import_root, module_name, preferred_class, module_file = s24.import_target(
        model_id=model_id,
        model_record=model_map[model_id],
        repository_root=Path(repository["local_path"]),
    )

    if record["configuration_mode"] == "direct_constructor_kwargs":
        backbone = target(**source_config)
    elif record["configuration_mode"] == "config_namespace":
        backbone = target(SimpleNamespace(**source_config))
    else:
        raise RuntimeError(
            f"Unknown configuration mode: {record['configuration_mode']}"
        )

    class DirectionalAdapter(nn.Module):
        def __init__(self, mid, bb):
            super().__init__()
            self.mid = mid
            self.backbone = bb
            self.readout = (
                None
                if mid == "TIMESNET"
                else nn.Linear(PRED_LEN * FEATURES, OUTPUTS)
            )

        def forward(self, x):
            if self.mid == "TIMESNET":
                mark = torch.ones(
                    (x.shape[0], x.shape[1]),
                    dtype=x.dtype,
                    device=x.device,
                )
                logits = self.backbone(x, mark, None, None)
                if tuple(logits.shape) != (x.shape[0], OUTPUTS):
                    raise RuntimeError(
                        f"TimesNet output contract drift: {tuple(logits.shape)}"
                    )
                return logits, None

            if self.mid == "PATCHTST":
                forecast = self.backbone(x.transpose(1, 2))
                if tuple(forecast.shape) != (
                    x.shape[0], FEATURES, PRED_LEN
                ):
                    raise RuntimeError(
                        f"PatchTST output contract drift: {tuple(forecast.shape)}"
                    )
                forecast = forecast.transpose(1, 2).contiguous()

            elif self.mid == "ITRANSFORMER":
                forecast = self.backbone(x, None, None, None)
                if isinstance(forecast, (tuple, list)):
                    forecast = forecast[0]

            elif self.mid == "MLF":
                forecast = self.backbone(x, None, None, None)
                if not isinstance(forecast, (tuple, list)) or not forecast:
                    raise RuntimeError("MLF output contract drift.")
                forecast = forecast[0]

            else:
                forecast = self.backbone(x)

            if tuple(forecast.shape) != (
                x.shape[0], PRED_LEN, FEATURES
            ):
                raise RuntimeError(
                    f"{self.mid} forecast contract drift: {tuple(forecast.shape)}"
                )

            logits = self.readout(
                forecast.reshape(forecast.shape[0], -1)
            )
            if tuple(logits.shape) != (x.shape[0], OUTPUTS):
                raise RuntimeError(
                    f"Directional logits contract drift: {tuple(logits.shape)}"
                )
            return logits, forecast

    model = DirectionalAdapter(model_id, backbone).to(device)

    adapter_params = (
        0
        if model_id == "TIMESNET"
        else sum(p.numel() for p in model.readout.parameters())
    )
    if model_id != "TIMESNET" and adapter_params != s24.COMMON_HEAD_PARAMS:
        raise RuntimeError(
            f"Directional adapter parameter drift: {adapter_params}"
        )

    for name, parameter in model.named_parameters():
        if parameter.device != device:
            raise RuntimeError(f"Device drift: {name}")
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            raise RuntimeError(f"Non-FP32 model parameter: {name}")

    learning_rate = float(record["training_config"]["learning_rate"])

    metadata = {
        "model_id": model_id,
        "config_id": config_id,
        "repo_id": record["repo_id"],
        "repository_root": str(repository["local_path"]),
        "configuration_mode": record["configuration_mode"],
        "learning_rate": learning_rate,
        "runtime_bindings": runtime_bindings,
        "import_root": str(import_root),
        "module_name": module_name,
        "preferred_class": preferred_class,
        "module_file": str(module_file),
        "adapter_parameter_count": int(adapter_params),
        "trainable_parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }

    return model, metadata, repository


# =============================================================================
# 6. REUSABLE STAGE-1 TRAINING / VALIDATION ENGINE
# =============================================================================

def _to_cpu_float32(array_like) -> np.ndarray:
    return np.asarray(array_like, dtype=np.float32)


def train_logical_batch(
    model,
    optimizer,
    criterion,
    x_source,
    y_source,
    logical_indices: np.ndarray,
    physical_microbatch: int,
    device,
) -> dict:
    """
    Execute EXACTLY one logical-batch update.

    The logical objective is the mean BCE over the actual logical batch N:
        sum_i mean_BCE_i * (n_i / N)

    This handles complete and incomplete logical batches without padding,
    duplication, dropping, or nominal-accumulation misweighting.
    """
    import torch

    logical_indices = np.asarray(logical_indices, dtype=np.int64)
    logical_n = int(len(logical_indices))
    if logical_n <= 0 or logical_n > LOGICAL_BATCH:
        raise ValueError(f"Invalid logical batch size: {logical_n}")

    optimizer.zero_grad(set_to_none=True)
    model.train()

    weighted_loss = 0.0
    microbatch_sizes = []

    for start in range(0, logical_n, int(physical_microbatch)):
        micro_idx = logical_indices[
            start:min(start + int(physical_microbatch), logical_n)
        ]
        micro_n = int(len(micro_idx))
        microbatch_sizes.append(micro_n)

        x_np = _to_cpu_float32(x_source[micro_idx])
        y_np = _to_cpu_float32(y_source[micro_idx])

        xb = torch.from_numpy(x_np).to(
            device=device, dtype=torch.float32, non_blocking=False
        )
        yb = torch.from_numpy(y_np).to(
            device=device, dtype=torch.float32, non_blocking=False
        )

        logits, forecast = model(xb)
        if tuple(logits.shape) != (micro_n, OUTPUTS):
            raise RuntimeError(
                f"Training logits shape drift: {tuple(logits.shape)}"
            )
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError("Non-finite training logits.")

        loss = criterion(logits, yb)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Non-finite training BCE.")

        weight = micro_n / logical_n
        scaled_loss = loss * weight
        scaled_loss.backward()

        weighted_loss += float(loss.detach().cpu().item()) * weight

        del xb, yb, logits, forecast, loss, scaled_loss

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=GRAD_CLIP
    )
    grad_norm_value = float(
        grad_norm.detach().cpu().item()
        if hasattr(grad_norm, "detach")
        else grad_norm
    )
    if not math.isfinite(grad_norm_value):
        raise FloatingPointError("Non-finite pre-clip gradient norm.")

    nonzero_gradient_tensors = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError(
                    f"Non-finite gradient after accumulation: {name}"
                )
            if bool(torch.count_nonzero(parameter.grad).item() > 0):
                nonzero_gradient_tensors += 1

    if nonzero_gradient_tensors == 0:
        raise RuntimeError("All accumulated gradients are zero.")

    optimizer.step()

    return {
        "logical_batch_n": logical_n,
        "microbatch_sizes": microbatch_sizes,
        "microbatch_count": len(microbatch_sizes),
        "microbatch_weight_sum": float(
            sum(n / logical_n for n in microbatch_sizes)
        ),
        "weighted_mean_bce": float(weighted_loss),
        "preclip_gradient_norm": grad_norm_value,
        "nonzero_gradient_tensors": nonzero_gradient_tensors,
        "optimizer_steps": 1,
    }


def evaluate_directional(
    model,
    criterion,
    x_source,
    y_source,
    physical_microbatch: int,
    device,
) -> dict:
    """Sequential, no-shuffle, no-grad validation in original row order."""
    import torch

    n = int(len(x_source))
    if n != int(len(y_source)) or n <= 0:
        raise ValueError("Validation X/y length mismatch or empty validation set.")

    model.eval()

    total_bce = 0.0
    logits_parts = []
    probability_parts = []

    with torch.no_grad():
        for start in range(0, n, int(physical_microbatch)):
            end = min(start + int(physical_microbatch), n)

            x_np = _to_cpu_float32(x_source[start:end])
            y_np = _to_cpu_float32(y_source[start:end])

            xb = torch.from_numpy(x_np).to(
                device=device, dtype=torch.float32, non_blocking=False
            )
            yb = torch.from_numpy(y_np).to(
                device=device, dtype=torch.float32, non_blocking=False
            )

            logits, _forecast = model(xb)
            if tuple(logits.shape) != (end - start, OUTPUTS):
                raise RuntimeError(
                    f"Validation logits shape drift: {tuple(logits.shape)}"
                )
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError("Non-finite validation logits.")

            loss = criterion(logits, yb)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("Non-finite validation BCE.")

            weight = (end - start) / n
            total_bce += float(loss.detach().cpu().item()) * weight

            prob = torch.sigmoid(logits)

            logits_parts.append(
                logits.detach().cpu().to(torch.float32).numpy()
            )
            probability_parts.append(
                prob.detach().cpu().to(torch.float32).numpy()
            )

            del xb, yb, logits, prob, loss

    logits_np = np.concatenate(logits_parts, axis=0)
    prob_np = np.concatenate(probability_parts, axis=0)

    if logits_np.shape != (n, OUTPUTS) or prob_np.shape != (n, OUTPUTS):
        raise RuntimeError("Validation output concatenation shape drift.")
    if not np.isfinite(logits_np).all() or not np.isfinite(prob_np).all():
        raise FloatingPointError("Non-finite validation output array.")

    return {
        "validation_mean_directional_bce": float(total_bce),
        "logits": logits_np,
        "probabilities": prob_np,
        "row_index": np.arange(n, dtype=np.int64),
        "validation_shuffle": False,
        "gradient_tracking": False,
    }


def compute_stage1_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        matthews_corrcoef,
        roc_auc_score,
    )

    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)

    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != OUTPUTS:
        raise ValueError("Metric input shape must be [N,3].")
    if not np.isfinite(p).all():
        raise FloatingPointError("Non-finite metric probabilities.")

    pred = (p >= float(threshold)).astype(np.int64)

    horizon_labels = ("1d", "3d", "5d")
    per_horizon = {}

    for j, label in enumerate(horizon_labels):
        yt = y[:, j]
        pt = p[:, j]
        yp = pred[:, j]

        if np.unique(yt).size != 2:
            raise RuntimeError(
                f"{label}: metric dry-run requires both binary classes."
            )

        per_horizon[label] = {
            "balanced_accuracy": float(
                balanced_accuracy_score(yt, yp)
            ),
            "roc_auc": float(
                roc_auc_score(yt, pt)
            ),
            "mcc": float(
                matthews_corrcoef(yt, yp)
            ),
            "pr_auc": float(
                average_precision_score(yt, pt)
            ),
            "brier_score": float(
                brier_score_loss(yt, pt)
            ),
        }

    macro = {}
    for metric in (
        "balanced_accuracy",
        "roc_auc",
        "mcc",
        "pr_auc",
        "brier_score",
    ):
        macro[metric] = float(
            np.mean(
                [per_horizon[h][metric] for h in horizon_labels]
            )
        )

    values = [
        value
        for row in per_horizon.values()
        for value in row.values()
    ] + list(macro.values())

    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite Stage-1 metric.")

    return {
        "threshold": float(threshold),
        "threshold_operator": "probability >= 0.50",
        "per_horizon": per_horizon,
        "macro_horizon_unweighted": macro,
    }


def checkpoint_payload(
    model,
    *,
    epoch: int,
    validation_mean_directional_bce: float,
    model_id: str,
    config_id: str,
    fold: int,
) -> dict:
    """
    CPU-only state_dict payload.  Optimizer state is deliberately NOT required
    for the best-checkpoint artifact because Stage-1 restores the best model
    for evaluation; incomplete fits are rerun rather than resumed mid-epoch.
    """
    state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
    }

    return {
        "schema_version": 1,
        "model_id": model_id,
        "config_id": config_id,
        "fold": int(fold),
        "epoch": int(epoch),
        "validation_mean_directional_bce": float(
            validation_mean_directional_bce
        ),
        "model_state_dict": state,
    }


def save_checkpoint_atomic(path: Path, payload: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint_into_model(path: Path, model) -> dict:
    import torch

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )
    return payload


# =============================================================================
# 7. ARTIFACT CONTRACT — CONSUMED BY THE REAL STAGE-1 RUNNER
# =============================================================================

def fit_artifact_schema() -> dict:
    return {
        "fit_directory_template":
            "stage1_fits/{model_id}/{config_id}/fold_{fold:02d}",
        "atomic_completion_rule":
            "fit_manifest.json is written last; only a hash-valid completed fit may be skipped",
        "incomplete_fit_rule":
            "delete/rerun the incomplete fit directory from scratch; no best-seed or partial-training promotion",
        "files": {
            "best_checkpoint.pt": {
                "required": True,
                "contents": [
                    "CPU model_state_dict",
                    "model_id",
                    "config_id",
                    "fold",
                    "best epoch",
                    "validation_mean_directional_bce",
                ],
            },
            "epoch_history.csv": {
                "required": True,
                "columns": [
                    "epoch",
                    "train_mean_directional_bce",
                    "validation_mean_directional_bce",
                    "learning_rate",
                    "mean_preclip_gradient_norm",
                    "epoch_seconds",
                    "strict_checkpoint_improvement",
                    "consecutive_nonimproving_epochs",
                    "stop_after_epoch",
                ],
            },
            "validation_predictions.npz": {
                "required": True,
                "arrays": {
                    "logits": "float32 [N,3]",
                    "probabilities": "float32 [N,3]",
                    "row_index": "int64 [N], original Step-14 validation row order",
                },
                "labels_duplicated": False,
                "label_source":
                    "immutable Step-14 validation y_direction artifact referenced in fit_manifest.json",
            },
            "fit_metrics.json": {
                "required": True,
                "contents": [
                    "per-horizon balanced_accuracy",
                    "per-horizon roc_auc",
                    "per-horizon mcc",
                    "per-horizon pr_auc",
                    "per-horizon brier_score",
                    "unweighted macro-horizon versions of the same five metrics",
                    "threshold=0.50",
                    "best epoch",
                    "best validation_mean_directional_bce",
                    "epochs run",
                    "early-stopping flag",
                    "trainable parameter count",
                    "fit duration",
                ],
            },
            "fit_manifest.json": {
                "required": True,
                "written_last": True,
                "contents": [
                    "model/config/fold/seed",
                    "source/config fingerprints",
                    "physical microbatch",
                    "logical batch",
                    "learning rate",
                    "input artifact paths+hashes",
                    "output artifact paths+hashes",
                    "repository commit/tree identifiers",
                    "final-test-not-accessed declaration",
                ],
            },
        },
        "selection_boundary": {
            "fit_runner_selects_configuration": False,
            "fit_runner_tunes_threshold": False,
            "fit_runner_selects_seed": False,
            "selection_is_a_later_development-only_step": True,
        },
        "prediction_semantics": {
            "probability": "sigmoid(logit)",
            "threshold": 0.50,
            "threshold_operator": "probability >= 0.50",
            "horizon_order": ["1d", "3d", "5d"],
            "validation_row_order": "original Step-14 validation row order",
        },
    }


# =============================================================================
# 8. ACTUAL FINAL-TAIL MICRO-BATCH SIZES FROM STEP 25
# =============================================================================

def candidate_tail_sizes(batch_plan: pd.DataFrame) -> dict[tuple[str, str], list[int]]:
    required = {
        "model_id",
        "config_id",
        "physical_microbatch_size",
        "final_logical_partition_json",
    }
    if not required.issubset(batch_plan.columns):
        raise RuntimeError("Step-25 batch-plan schema drift.")

    out = {}

    for key, group in batch_plan.groupby(["model_id", "config_id"], sort=True):
        physical_values = group["physical_microbatch_size"].astype(int).unique()
        if len(physical_values) != 1:
            raise RuntimeError(f"{key}: multiple physical microbatches in batch plan.")
        physical = int(physical_values[0])

        sizes = set()
        for value in group["final_logical_partition_json"]:
            parts = [int(x) for x in json.loads(value)]
            if not parts or any(n <= 0 or n > physical for n in parts):
                raise RuntimeError(f"{key}: invalid frozen final partition {parts}")
            sizes.update(parts)

        out[key] = sorted(sizes)

    if len(out) != EXPECTED_CANDIDATES:
        raise RuntimeError("Tail-size map does not contain 16 candidates.")

    return out


# =============================================================================
# 9. SYNTHETIC WORKER
# =============================================================================

def _synthetic_xy(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(
        size=(n, LOOKBACK, FEATURES)
    ).astype(np.float32)

    row = np.arange(n, dtype=np.int64)[:, None]
    col = np.arange(OUTPUTS, dtype=np.int64)[None, :]
    y = ((row + col) % 2).astype(np.float32)

    return x, y


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise RuntimeError("Array shape mismatch in comparison.")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def worker_execute(
    model_id: str,
    config_id: str,
    physical_microbatch: int,
    tail_sizes: list[int],
) -> dict:
    import torch
    import torch.nn as nn

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.dont_write_bytecode = True

    started = time.perf_counter()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    set_training_determinism(SEED)

    device = torch.device("cuda:0")
    device_index = 0

    model, metadata, repository = build_candidate_model(
        model_id=model_id,
        config_id=config_id,
        physical_microbatch=physical_microbatch,
        device=device,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(metadata["learning_rate"]),
        betas=BETAS,
        eps=ADAMW_EPS,
        weight_decay=WEIGHT_DECAY,
    )

    # Use one actual frozen tail size, if one exists, to make the synthetic
    # logical update deliberately incomplete. If every frozen final partition
    # is a full physical microbatch, use two full microbatches.
    partial_tails = [
        n for n in sorted(set(tail_sizes))
        if n < physical_microbatch
    ]

    if partial_tails:
        chosen_tail = int(partial_tails[0])
        dryrun_train_n = int(physical_microbatch + chosen_tail)
    else:
        chosen_tail = int(physical_microbatch)
        dryrun_train_n = int(2 * physical_microbatch)

    if dryrun_train_n > LOGICAL_BATCH:
        dryrun_train_n = LOGICAL_BATCH

    # Validation is sequential and deliberately crosses a microbatch boundary.
    dryrun_val_n = min(
        LOGICAL_BATCH,
        int(physical_microbatch + max(1, min(chosen_tail, physical_microbatch))),
    )

    x_train, y_train = _synthetic_xy(dryrun_train_n)
    x_val, y_val = _synthetic_xy(dryrun_val_n)

    permutation = epoch_permutation(
        dryrun_train_n, fold=1, epoch=1
    )

    update = train_logical_batch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        x_source=x_train,
        y_source=y_train,
        logical_indices=permutation,
        physical_microbatch=physical_microbatch,
        device=device,
    )

    if update["optimizer_steps"] != 1:
        raise RuntimeError("Dry-run did not execute exactly one optimizer step.")

    optimizer_state_tensors = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                optimizer_state_tensors += 1
                if value.is_floating_point() and not bool(
                    torch.isfinite(value).all().item()
                ):
                    raise FloatingPointError("Non-finite AdamW optimizer state.")

    if optimizer_state_tensors == 0:
        raise RuntimeError("AdamW optimizer state was not allocated.")

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not bool(
            torch.isfinite(parameter).all().item()
        ):
            raise FloatingPointError(
                f"Non-finite parameter after synthetic update: {name}"
            )

    evaluation_before = evaluate_directional(
        model=model,
        criterion=criterion,
        x_source=x_val,
        y_source=y_val,
        physical_microbatch=physical_microbatch,
        device=device,
    )

    metrics = compute_stage1_metrics(
        y_true=y_val,
        probabilities=evaluation_before["probabilities"],
        threshold=THRESHOLD,
    )

    # Checkpoint round-trip.
    checkpoint_dir = cfg.AUDIT_DIR / "26_stage1_runner_checkpoint_tmp"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_id}__{config_id}.pt"

    payload = checkpoint_payload(
        model,
        epoch=1,
        validation_mean_directional_bce=evaluation_before[
            "validation_mean_directional_bce"
        ],
        model_id=model_id,
        config_id=config_id,
        fold=1,
    )
    save_checkpoint_atomic(checkpoint_path, payload)
    checkpoint_sha = sha256_file(checkpoint_path)

    # Deliberately perturb one parameter before restore.
    with torch.no_grad():
        first_parameter = next(
            p for p in model.parameters()
            if p.requires_grad and p.numel() > 0
        )
        first_parameter.view(-1)[0].add_(0.125)

    loaded = load_checkpoint_into_model(checkpoint_path, model)

    if (
        int(loaded["epoch"]) != 1
        or loaded["model_id"] != model_id
        or loaded["config_id"] != config_id
    ):
        raise RuntimeError("Checkpoint metadata round-trip drift.")

    evaluation_after = evaluate_directional(
        model=model,
        criterion=criterion,
        x_source=x_val,
        y_source=y_val,
        physical_microbatch=physical_microbatch,
        device=device,
    )

    reload_logit_diff = _max_abs(
        evaluation_before["logits"],
        evaluation_after["logits"],
    )
    reload_probability_diff = _max_abs(
        evaluation_before["probabilities"],
        evaluation_after["probabilities"],
    )
    reload_bce_diff = abs(
        evaluation_before["validation_mean_directional_bce"]
        - evaluation_after["validation_mean_directional_bce"]
    )

    if (
        reload_logit_diff > CHECKPOINT_RELOAD_LOGIT_TOL
        or reload_probability_diff > CHECKPOINT_RELOAD_LOGIT_TOL
        or reload_bce_diff > CHECKPOINT_RELOAD_LOGIT_TOL
    ):
        raise RuntimeError(
            "Checkpoint restore prediction equivalence failed: "
            f"logit={reload_logit_diff}, prob={reload_probability_diff}, "
            f"bce={reload_bce_diff}"
        )

    try:
        checkpoint_path.unlink()
        if checkpoint_dir.exists() and not any(checkpoint_dir.iterdir()):
            checkpoint_dir.rmdir()
    except Exception:
        pass

    # Probe every physical tail size that can really occur in the five folds.
    tail_records = []
    model.train()
    with torch.no_grad():
        for tail_n in sorted(set(int(n) for n in tail_sizes)):
            x_tail, _ = _synthetic_xy(tail_n)
            xb = torch.from_numpy(x_tail).to(
                device=device, dtype=torch.float32
            )
            logits, _forecast = model(xb)
            if tuple(logits.shape) != (tail_n, OUTPUTS):
                raise RuntimeError(
                    f"Tail-size {tail_n} logits shape drift: {tuple(logits.shape)}"
                )
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError(
                    f"Non-finite logits for actual tail microbatch N={tail_n}"
                )
            tail_records.append(
                {
                    "physical_tail_n": int(tail_n),
                    "train_mode_forward_pass": True,
                    "finite_logits": True,
                    "logits_shape": list(logits.shape),
                }
            )
            del xb, logits, _forecast

    torch.cuda.synchronize(device_index)

    peak_allocated = int(torch.cuda.max_memory_allocated(device_index))
    peak_reserved = int(torch.cuda.max_memory_reserved(device_index))

    s24 = import_step24()
    s24.assert_single_repository_clean(repository)

    return {
        "status": "PASS",
        "model_id": model_id,
        "config_id": config_id,
        "physical_microbatch_size": int(physical_microbatch),
        "actual_frozen_tail_sizes": sorted(set(int(n) for n in tail_sizes)),
        "dryrun_logical_train_n": int(dryrun_train_n),
        "dryrun_validation_n": int(dryrun_val_n),
        "training_microbatch_sizes": update["microbatch_sizes"],
        "training_microbatch_weight_sum": update["microbatch_weight_sum"],
        "synthetic_train_mean_directional_bce": update["weighted_mean_bce"],
        "preclip_gradient_norm": update["preclip_gradient_norm"],
        "nonzero_gradient_tensors": update["nonzero_gradient_tensors"],
        "optimizer_steps": update["optimizer_steps"],
        "optimizer_state_tensor_count": int(optimizer_state_tensors),
        "validation_mean_directional_bce": evaluation_before[
            "validation_mean_directional_bce"
        ],
        "validation_row_order_preserved": bool(
            np.array_equal(
                evaluation_before["row_index"],
                np.arange(dryrun_val_n, dtype=np.int64),
            )
        ),
        "stage1_metrics": metrics,
        "checkpoint_sha256_before_delete": checkpoint_sha,
        "checkpoint_reload_logit_max_abs_diff": reload_logit_diff,
        "checkpoint_reload_probability_max_abs_diff": reload_probability_diff,
        "checkpoint_reload_bce_abs_diff": reload_bce_diff,
        "checkpoint_reload_pass": True,
        "tail_probe_records": tail_records,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "peak_cuda_reserved_role": "diagnostic_only",
        "learning_rate": metadata["learning_rate"],
        "weight_decay": WEIGHT_DECAY,
        "adamw_betas": list(BETAS),
        "adamw_eps": ADAMW_EPS,
        "gradient_clip_norm": GRAD_CLIP,
        "threshold": THRESHOLD,
        "threshold_operator": "probability >= 0.50",
        "dtype": "float32",
        "amp_enabled": False,
        "tf32_enabled": False,
        "trainable_parameter_count": metadata["trainable_parameter_count"],
        "runtime_bindings": metadata["runtime_bindings"],
        "repository_clean_after_worker": True,
        "development_numerical_arrays_opened": False,
        "validation_performance_inspected": False,
        "final_test_data_opened": False,
        "model_fitted_on_development_data": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def worker_main(
    model_id: str,
    config_id: str,
    physical_microbatch: int,
    tail_sizes_json: str,
) -> int:
    try:
        tail_sizes = [int(x) for x in json.loads(tail_sizes_json)]
        result = worker_execute(
            model_id=model_id,
            config_id=config_id,
            physical_microbatch=int(physical_microbatch),
            tail_sizes=tail_sizes,
        )
        print(
            RESULT_PREFIX
            + json.dumps(result, sort_keys=True, ensure_ascii=False)
        )
        return 0

    except Exception as exc:
        failure = {
            "status": "FAIL",
            "model_id": model_id,
            "config_id": config_id,
            "physical_microbatch_size": int(physical_microbatch),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "development_numerical_arrays_opened": False,
            "validation_performance_inspected": False,
            "final_test_data_opened": False,
            "model_fitted_on_development_data": False,
        }
        print(
            RESULT_PREFIX
            + json.dumps(failure, sort_keys=True, ensure_ascii=False)
        )
        return 1


# =============================================================================
# 10. MASTER WORKER ORCHESTRATION
# =============================================================================

def parse_worker_result(stdout: str) -> dict:
    matches = [
        line[len(RESULT_PREFIX):]
        for line in stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one worker result marker; found {len(matches)}."
        )
    return json.loads(matches[0])


def write_worker_log(
    model_id: str,
    config_id: str,
    completed: subprocess.CompletedProcess,
) -> Path:
    WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = WORKER_LOG_DIR / f"{model_id}__{config_id}__dryrun.log"
    atomic_text(
        path,
        (
            f"return_code={completed.returncode}\n\n"
            "===== STDOUT =====\n"
            f"{completed.stdout}\n\n"
            "===== STDERR =====\n"
            f"{completed.stderr}\n"
        ),
    )
    return path


def run_worker(
    model_id: str,
    config_id: str,
    physical_microbatch: int,
    tail_sizes: list[int],
) -> tuple[dict, Path]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--worker",
            "--model-id",
            model_id,
            "--config-id",
            config_id,
            "--physical-microbatch",
            str(int(physical_microbatch)),
            "--tail-sizes-json",
            json.dumps([int(n) for n in tail_sizes]),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
        timeout=WORKER_TIMEOUT_SECONDS,
    )

    log_path = write_worker_log(
        model_id, config_id, completed
    )

    try:
        result = parse_worker_result(completed.stdout)
    except Exception as exc:
        result = {
            "status": "FAIL",
            "model_id": model_id,
            "config_id": config_id,
            "physical_microbatch_size": int(physical_microbatch),
            "exception_type": "WorkerProtocolError",
            "exception_message": str(exc),
            "development_numerical_arrays_opened": False,
            "validation_performance_inspected": False,
            "final_test_data_opened": False,
            "model_fitted_on_development_data": False,
        }

    result["worker_return_code"] = int(completed.returncode)
    result["worker_log"] = str(log_path)

    return result, log_path


# =============================================================================
# 11. FREEZE / VERIFY
# =============================================================================

def source_hash_frame(upstream: dict) -> pd.DataFrame:
    paths = [
        ("step26_working_script", SCRIPT),
        ("step24_working_script", STEP24_SCRIPT),
        ("step24_frozen_script", STEP24_FROZEN_SCRIPT),
        ("step25_working_script", STEP25_SCRIPT),
        ("step25_frozen_script", STEP25_FROZEN_SCRIPT),
        ("step24_freeze", STEP24_FREEZE),
        ("step24_manifest", STEP24_MANIFEST),
        ("step25_freeze", STEP25_FREEZE),
        ("step25_manifest", STEP25_MANIFEST),
        ("step25_batch_plan", STEP25_BATCH_PLAN),
        ("step10_final_test_seal_metadata", STEP10_TEST_SEAL),
        ("global_final_test_lock_metadata", upstream["global_lock_path"]),
    ]

    rows = []
    for role, path in paths:
        path = Path(path)
        if path.name == FORBIDDEN_TEST:
            raise RuntimeError("Final-test parquet entered Step-26 provenance.")
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def freeze_step26(
    upstream: dict,
    results: list[dict],
    tail_rows: list[dict],
) -> tuple[str, str]:
    artifact_schema = fit_artifact_schema()
    atomic_json(ARTIFACT_SCHEMA_JSON, artifact_schema)

    results_frame = pd.DataFrame(
        [
            {
                "model_id": r["model_id"],
                "config_id": r["config_id"],
                "status": r["status"],
                "physical_microbatch_size": r["physical_microbatch_size"],
                "actual_frozen_tail_sizes_json": json.dumps(
                    r["actual_frozen_tail_sizes"]
                ),
                "dryrun_logical_train_n": r["dryrun_logical_train_n"],
                "dryrun_validation_n": r["dryrun_validation_n"],
                "training_microbatch_sizes_json": json.dumps(
                    r["training_microbatch_sizes"]
                ),
                "training_microbatch_weight_sum": r[
                    "training_microbatch_weight_sum"
                ],
                "optimizer_steps": r["optimizer_steps"],
                "optimizer_state_tensor_count": r[
                    "optimizer_state_tensor_count"
                ],
                "checkpoint_reload_logit_max_abs_diff": r[
                    "checkpoint_reload_logit_max_abs_diff"
                ],
                "checkpoint_reload_probability_max_abs_diff": r[
                    "checkpoint_reload_probability_max_abs_diff"
                ],
                "checkpoint_reload_bce_abs_diff": r[
                    "checkpoint_reload_bce_abs_diff"
                ],
                "checkpoint_reload_pass": r["checkpoint_reload_pass"],
                "validation_row_order_preserved": r[
                    "validation_row_order_preserved"
                ],
                "peak_cuda_allocated_mib": r[
                    "peak_cuda_allocated_bytes"
                ] / (1024 ** 2),
                "peak_cuda_reserved_mib": r[
                    "peak_cuda_reserved_bytes"
                ] / (1024 ** 2),
                "learning_rate": r["learning_rate"],
                "trainable_parameter_count": r[
                    "trainable_parameter_count"
                ],
                "elapsed_seconds": r["elapsed_seconds"],
                "worker_return_code": r["worker_return_code"],
                "worker_log": r["worker_log"],
            }
            for r in results
        ]
    )
    write_csv(RESULTS_CSV, results_frame)
    write_csv(TAIL_CSV, pd.DataFrame(tail_rows))

    source_hashes = source_hash_frame(upstream)
    write_csv(SOURCE_HASH_CSV, source_hashes)

    semantic = {
        "schema_version": SCHEMA_VERSION,
        "status": "STAGE1_RUNNER_IMPLEMENTATION_DRYRUN_FROZEN",
        "configuration_fingerprint_sha256": EXPECTED_CONFIGURATION_FINGERPRINT,
        "upstream": {
            "step24_training_runtime_preflight_fingerprint_sha256":
                EXPECTED_STEP24_FINGERPRINT,
            "step24_manifest_sha256":
                EXPECTED_STEP24_MANIFEST_SHA256,
            "step25_training_engine_fingerprint_sha256":
                EXPECTED_STEP25_FINGERPRINT,
            "step25_manifest_sha256":
                EXPECTED_STEP25_MANIFEST_SHA256,
        },
        "runner_contract": {
            "candidate_count": EXPECTED_CANDIDATES,
            "fold_count": EXPECTED_FOLDS,
            "planned_stage1_fits": EXPECTED_FITS,
            "seed": SEED,
            "logical_batch_size": LOGICAL_BATCH,
            "physical_microbatch_source": "frozen Step-24 schedule",
            "physical_microbatch_is_candidate_execution_contract": True,
            "training_order":
                "SeedSequence([42, fold, epoch]); candidate-independent",
            "optimizer": "AdamW",
            "adamw_betas": list(BETAS),
            "adamw_eps": ADAMW_EPS,
            "weight_decay": WEIGHT_DECAY,
            "loss": "BCEWithLogitsLoss(reduction='mean')",
            "partial_batch_weighting": "micro_n / logical_n",
            "gradient_clip_norm": GRAD_CLIP,
            "gradient_clip_timing":
                "once after complete logical-batch accumulation",
            "optimizer_steps_per_logical_batch": 1,
            "dtype": "float32",
            "amp_enabled": False,
            "tf32_enabled": False,
            "validation_shuffle": False,
            "validation_gradient_tracking": False,
            "validation_bce":
                "sample-weighted mean across all validation samples and 3 outputs",
            "probability": "sigmoid(logit)",
            "threshold": THRESHOLD,
            "threshold_operator": "probability >= 0.50",
            "horizon_order": ["1d", "3d", "5d"],
            "stage1_metrics": [
                "balanced_accuracy",
                "roc_auc",
                "mcc",
                "pr_auc",
                "brier_score",
            ],
            "macro_horizon_aggregation": "unweighted arithmetic mean",
            "checkpoint_restore_before_metrics": True,
            "scheduler": "none",
            "fit_worker_isolation": "one subprocess per model/config/fold",
        },
        "dryrun": {
            "synthetic_only": True,
            "candidate_records": len(results),
            "all_candidates_passed": all(r["status"] == "PASS" for r in results),
            "one_optimizer_step_per_candidate": True,
            "all_adamw_state_allocated": all(
                int(r["optimizer_state_tensor_count"]) > 0
                for r in results
            ),
            "all_checkpoint_roundtrips_passed": all(
                bool(r["checkpoint_reload_pass"])
                for r in results
            ),
            "all_validation_row_order_preserved": all(
                bool(r["validation_row_order_preserved"])
                for r in results
            ),
            "all_actual_tail_microbatch_shapes_passed": all(
                all(
                    bool(t["train_mode_forward_pass"])
                    and bool(t["finite_logits"])
                    for t in r["tail_probe_records"]
                )
                for r in results
            ),
        },
        "artifact_schema": artifact_schema,
        "selection_boundary": {
            "configuration_selection_inside_runner": False,
            "threshold_optimization_inside_runner": False,
            "seed_selection_inside_runner": False,
        },
        "pipeline_state": {
            "development_numerical_arrays_opened": False,
            "validation_performance_inspected": False,
            "stage1_real_fit_started": False,
            "synthetic_optimizer_steps_run": EXPECTED_CANDIDATES,
            "development_optimizer_steps_run": 0,
        },
        "final_test_protection": {
            "step10_seal_status": upstream["test_seal_status"],
            "final_test_parquet_opened_or_hashed_by_step26": False,
            "final_test_metric_computed": False,
            "global_final_test_state": upstream["global_final_test_state"],
        },
    }

    fingerprint = semantic_hash(semantic)
    freeze = {
        **semantic,
        "generated_at_utc": now(),
        "stage1_runner_dryrun_fingerprint_sha256": fingerprint,
    }
    atomic_json(FREEZE_JSON, freeze)
    shutil.copy2(SCRIPT, FROZEN_SCRIPT)

    summary_lines = [
        STEP_NAME,
        "=" * 110,
        "Status: PASS",
        f"Configuration fingerprint: {EXPECTED_CONFIGURATION_FINGERPRINT}",
        f"Step-24 preflight fingerprint: {EXPECTED_STEP24_FINGERPRINT}",
        f"Step-25 training-engine fingerprint: {EXPECTED_STEP25_FINGERPRINT}",
        f"Candidates dry-run: {len(results)}",
        "Development numerical arrays opened: False",
        "Validation performance inspected: False",
        "Final test accessed: False",
        "",
        "Candidate dry-run schedule:",
    ]

    for r in results:
        summary_lines.append(
            f"{r['model_id']:<14} {r['config_id']:<20} "
            f"mb={r['physical_microbatch_size']:<3} "
            f"trainN={r['dryrun_logical_train_n']:<3} "
            f"tails={r['actual_frozen_tail_sizes']} "
            f"ckpt_diff={r['checkpoint_reload_logit_max_abs_diff']:.3e}"
        )

    atomic_text(SUMMARY_TXT, "\n".join(summary_lines) + "\n")

    output_files = [
        RESULTS_CSV,
        TAIL_CSV,
        ARTIFACT_SCHEMA_JSON,
        SOURCE_HASH_CSV,
        FREEZE_JSON,
        FROZEN_SCRIPT,
        SUMMARY_TXT,
        *sorted(WORKER_LOG_DIR.glob("*.log")),
    ]

    artifacts = {}
    for path in output_files:
        key = (
            f"worker_log::{path.name}"
            if path.parent.resolve() == WORKER_LOG_DIR.resolve()
            else path.name
        )
        artifacts[key] = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "step": {
            "step_number": STEP_NUMBER,
            "step_name": STEP_NAME,
            "status": "PASS",
        },
        "frozen_at_utc": now(),
        "configuration_fingerprint_sha256":
            EXPECTED_CONFIGURATION_FINGERPRINT,
        "step24_manifest_sha256":
            EXPECTED_STEP24_MANIFEST_SHA256,
        "step25_manifest_sha256":
            EXPECTED_STEP25_MANIFEST_SHA256,
        "stage1_runner_dryrun_fingerprint_sha256":
            fingerprint,
        "counts": {
            "candidate_dryruns": len(results),
            "tail_shape_records": len(tail_rows),
            "planned_stage1_fits": EXPECTED_FITS,
        },
        "pipeline_state": freeze["pipeline_state"],
        "final_test_protection": freeze["final_test_protection"],
        "artifacts": artifacts,
    }

    atomic_json(MANIFEST_JSON, manifest)
    manifest_sha = sha256_file(MANIFEST_JSON)
    atomic_text(
        MANIFEST_SHA,
        f"{manifest_sha}  {MANIFEST_JSON.name}\n",
    )

    return fingerprint, manifest_sha


def verify_existing() -> int:
    print("=" * 140)
    print("STEP 26 — VERIFY EXISTING STAGE-1 RUNNER DRY-RUN FREEZE")
    print("=" * 140)

    upstream = verify_upstream()

    manifest_sha = verify_sidecar(
        MANIFEST_JSON, MANIFEST_SHA, "Step 26"
    )
    manifest = load_json(MANIFEST_JSON)
    freeze = load_json(FREEZE_JSON)

    if manifest.get("step", {}).get("status") != "PASS":
        print("[FAIL] Step-26 manifest status is not PASS.")
        return 20

    if (
        manifest.get("configuration_fingerprint_sha256")
        != EXPECTED_CONFIGURATION_FINGERPRINT
    ):
        print("[FAIL] Step-26 configuration fingerprint mismatch.")
        return 21

    if manifest.get("step24_manifest_sha256") != EXPECTED_STEP24_MANIFEST_SHA256:
        print("[FAIL] Step-24 manifest chain mismatch.")
        return 22
    if manifest.get("step25_manifest_sha256") != EXPECTED_STEP25_MANIFEST_SHA256:
        print("[FAIL] Step-25 manifest chain mismatch.")
        return 23

    stored = freeze["stage1_runner_dryrun_fingerprint_sha256"]
    recomputed = recompute_freeze_fp(
        freeze, "stage1_runner_dryrun_fingerprint_sha256"
    )
    if stored != recomputed:
        print("[FAIL] Step-26 semantic fingerprint no longer recomputes.")
        return 24
    if manifest.get("stage1_runner_dryrun_fingerprint_sha256") != stored:
        print("[FAIL] Step-26 manifest/freeze fingerprint mismatch.")
        return 25

    for record in manifest.get("artifacts", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            print(f"[FAIL] Step-26 artifact drift: {path}")
            return 26

    results = pd.read_csv(RESULTS_CSV)
    tails = pd.read_csv(TAIL_CSV)

    if len(results) != EXPECTED_CANDIDATES:
        print("[FAIL] Stored dry-run result count is not 16.")
        return 27
    if not (results["status"] == "PASS").all():
        print("[FAIL] Stored dry-run table contains a failure.")
        return 28
    if not results["checkpoint_reload_pass"].astype(bool).all():
        print("[FAIL] Stored checkpoint round-trip contains a failure.")
        return 29

    if (
        sha256_file(upstream["global_lock_path"])
        != upstream["global_lock_sha256"]
    ):
        print("[FAIL] Global final-test lock changed.")
        return 30

    print(f"[PASS] Configuration fingerprint : {EXPECTED_CONFIGURATION_FINGERPRINT}")
    print(f"[PASS] Step-24 manifest          : {EXPECTED_STEP24_MANIFEST_SHA256}")
    print(f"[PASS] Step-25 manifest          : {EXPECTED_STEP25_MANIFEST_SHA256}")
    print(f"[PASS] Candidate dry-runs        : {len(results)}")
    print(f"[PASS] Tail-shape probes         : {len(tails)}")
    print(f"[PASS] Planned Stage-1 fits      : {EXPECTED_FITS}")
    print(f"[PASS] Step-26 runner freeze     : {stored}")
    print(f"[PASS] Step-26 manifest          : {manifest_sha}")
    print(f"[PASS] Step-10 test seal         : {upstream['test_seal_status']}")
    print(f"[PASS] Global final-test state   : {upstream['global_final_test_state']}")
    print("=" * 140)
    print("STEP 26 VERIFICATION STATUS: PASS")
    print("=" * 140)
    return 0


# =============================================================================
# 12. MASTER
# =============================================================================

def run_master() -> int:
    stage = "verify"

    try:
        print("=" * 140)
        print(STEP_NAME)
        print("=" * 140)

        print("[1/8] Verifying frozen Steps 00-25 and current/frozen runner dependencies...")
        upstream = verify_upstream()
        print(f"        Configuration fingerprint : {EXPECTED_CONFIGURATION_FINGERPRINT}")
        print(f"        Step-24 manifest           : {EXPECTED_STEP24_MANIFEST_SHA256}")
        print(f"        Step-24 preflight freeze   : {EXPECTED_STEP24_FINGERPRINT}")
        print(f"        Step-25 manifest           : {EXPECTED_STEP25_MANIFEST_SHA256}")
        print(f"        Step-25 engine freeze      : {EXPECTED_STEP25_FINGERPRINT}")
        print(f"        Step-10 test seal          : {upstream['test_seal_status']}")
        print(f"        Global final-test state    : {upstream['global_final_test_state']}")

        stage = "collision_guard"
        print("[2/8] Checking immutable Step-26 output target...")
        assert_fresh_target()
        cfg.create_directories()
        WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)

        stage = "batch_plan"
        print("[3/8] Reading only frozen Step-25 batch-plan metadata...")
        batch_plan = pd.read_csv(STEP25_BATCH_PLAN)
        if len(batch_plan) != EXPECTED_FITS:
            raise RuntimeError(
                f"Expected {EXPECTED_FITS} Step-25 candidate/fold plan rows; "
                f"found {len(batch_plan)}."
            )
        tails_by_candidate = candidate_tail_sizes(batch_plan)
        print(f"        Candidate/fold plan rows   : {len(batch_plan)}")
        print("        Development numerical arrays opened: False")

        stage = "schedule"
        print("[4/8] Reconciling Step-24 physical microbatches with Step-25 plans...")
        schedule_map = {
            (r["model_id"], r["config_id"]): r
            for r in upstream["schedule"]
        }
        if len(schedule_map) != EXPECTED_CANDIDATES:
            raise RuntimeError("Step-24 schedule key count is not 16.")

        for key, group in batch_plan.groupby(["model_id", "config_id"], sort=True):
            physical = int(group["physical_microbatch_size"].iloc[0])
            if int(schedule_map[key]["physical_microbatch_size"]) != physical:
                raise RuntimeError(f"{key}: Step-24/Step-25 microbatch drift.")

        print("        Reconciled candidates      : 16")

        stage = "workers"
        print("[5/8] Running one isolated synthetic Stage-1 dry-run per candidate...")
        print("-" * 140)

        rank = {m: i for i, m in enumerate(MODEL_ORDER)}
        ordered_keys = sorted(
            schedule_map,
            key=lambda k: (rank[k[0]], k[1]),
        )

        results = []
        tail_rows = []

        for model_id, config_id in ordered_keys:
            schedule = schedule_map[(model_id, config_id)]
            physical = int(schedule["physical_microbatch_size"])
            tails = tails_by_candidate[(model_id, config_id)]

            print(
                f"DRYRUN {model_id:<14} {config_id:<20} "
                f"mb={physical:<3} actual_tail_sizes={tails}"
            )

            result, log_path = run_worker(
                model_id=model_id,
                config_id=config_id,
                physical_microbatch=physical,
                tail_sizes=tails,
            )
            results.append(result)

            if result.get("status") != "PASS":
                raise RuntimeError(
                    f"Synthetic Stage-1 dry-run failed for "
                    f"{model_id}/{config_id}: "
                    f"{result.get('exception_type')}: "
                    f"{result.get('exception_message')}\n"
                    f"Worker log: {log_path}"
                )

            for t in result["tail_probe_records"]:
                tail_rows.append(
                    {
                        "model_id": model_id,
                        "config_id": config_id,
                        "frozen_physical_microbatch_size": physical,
                        "physical_tail_n": int(t["physical_tail_n"]),
                        "train_mode_forward_pass": bool(
                            t["train_mode_forward_pass"]
                        ),
                        "finite_logits": bool(t["finite_logits"]),
                        "logits_shape_json": json.dumps(t["logits_shape"]),
                    }
                )

            print(
                f"       PASS trainN={result['dryrun_logical_train_n']:<3} "
                f"micro={result['training_microbatch_sizes']} "
                f"AdamW_state={result['optimizer_state_tensor_count']:<4} "
                f"ckpt_diff={result['checkpoint_reload_logit_max_abs_diff']:.3e} "
                f"peak_alloc={result['peak_cuda_allocated_bytes']/(1024**2):.1f} MiB"
            )

        print("-" * 140)

        stage = "artifact_schema"
        print("[6/8] Freezing per-fit artifact and selection-boundary schema...")
        schema = fit_artifact_schema()
        print("        best_checkpoint.pt         : required")
        print("        epoch_history.csv          : required")
        print("        validation_predictions.npz : required")
        print("        fit_metrics.json           : required")
        print("        fit_manifest.json          : required, written last")
        print("        Config selection in runner : False")
        print("        Threshold tuning in runner : False")
        print("        Seed selection in runner   : False")

        stage = "integrity"
        print("[7/8] Rechecking final-test protection and repository-clean evidence...")
        if any(not bool(r["repository_clean_after_worker"]) for r in results):
            raise RuntimeError("At least one repository was not clean after dry-run.")
        if any(bool(r["development_numerical_arrays_opened"]) for r in results):
            raise RuntimeError("A worker reports development numerical-array access.")
        if any(bool(r["final_test_data_opened"]) for r in results):
            raise RuntimeError("A worker reports final-test access.")

        if (
            sha256_file(upstream["global_lock_path"])
            != upstream["global_lock_sha256"]
        ):
            raise RuntimeError("Global final-test lock changed during Step 26.")

        print("        Repositories clean         : True")
        print("        Development numerical data : NOT OPENED")
        print("        Validation performance     : NOT INSPECTED")
        print("        Final-test data            : NOT OPENED")

        stage = "freeze"
        print("[8/8] Freezing the audited Stage-1 runner implementation...")
        fingerprint, manifest_sha = freeze_step26(
            upstream=upstream,
            results=results,
            tail_rows=tail_rows,
        )

        print()
        print("-" * 140)
        print("STEP-26 FROZEN RUNNER CONTRACT")
        print("-" * 140)
        print(f"Candidates dry-run             : {len(results)}/{EXPECTED_CANDIDATES}")
        print(f"Actual tail-shape probes       : {len(tail_rows)}")
        print(f"Planned real Stage-1 fits      : {EXPECTED_FITS}")
        print(f"Logical/effective batch        : {LOGICAL_BATCH}")
        print("Physical microbatch            : frozen Step-24 per candidate")
        print("Partial-batch weighting        : micro_n / logical_n")
        print("Optimizer                      : AdamW")
        print(f"AdamW betas / eps              : {BETAS} / {ADAMW_EPS}")
        print(f"Weight decay                   : {WEIGHT_DECAY}")
        print(f"Gradient clip                  : {GRAD_CLIP}")
        print("FP32 / AMP / TF32              : float32 / False / False")
        print("Validation                     : sequential, eval, no-grad")
        print("Checkpoint round-trip          : PASS for all 16")
        print("Prediction probability         : sigmoid(logit)")
        print("Stage-1 threshold              : probability >= 0.50")
        print("Metrics                        : BA, ROC-AUC, MCC, PR-AUC, Brier")
        print("Metric horizon aggregation     : unweighted macro over 1d/3d/5d")
        print("Configuration selection        : NOT performed by fit runner")
        print("Threshold optimization         : NOT performed by fit runner")
        print("Best-seed selection            : NOT performed by fit runner")
        print("-" * 140)
        print("PROTECTION STATE")
        print("-" * 140)
        print("Real Stage-1 fitting started   : False")
        print("Development numerical arrays   : NOT OPENED")
        print("Validation performance         : NOT INSPECTED")
        print(f"Step-10 final-test seal        : {upstream['test_seal_status']}")
        print(f"Global final-test state        : {upstream['global_final_test_state']}")
        print("-" * 140)
        print(f"Step-26 runner fingerprint     : {fingerprint}")
        print(f"Step-26 manifest SHA-256       : {manifest_sha}")
        print(f"Freeze                         : {FREEZE_JSON}")
        print(f"Manifest                       : {MANIFEST_JSON}")
        print("=" * 140)
        print(
            "PASS: The reusable Stage-1 runner path completed synthetic "
            "training/validation/checkpoint/metric dry-runs for all 16 candidates."
        )
        print(
            "PASS: Every physical tail microbatch size that can occur in the "
            "five frozen folds was accepted in train mode."
        )
        print(
            "PASS: No real development fit, validation-performance inspection, "
            "configuration selection, or final-test access occurred."
        )
        print("=" * 140)
        print(
            "Do not launch the 80 real Stage-1 fits yet. Paste the complete "
            "Step-26 terminal output for audit, then run --verify-only if instructed."
        )

        return 0

    except subprocess.TimeoutExpired as exc:
        archived = archive_partial()
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        failure = FAILED_DIR / (
            "26_stage1_runner_dryrun_FAILED_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
        atomic_json(
            failure,
            {
                "status": "FAIL",
                "stage": stage,
                "error_type": "TimeoutExpired",
                "error_message": str(exc),
                "failed_at_utc": now(),
                "archived_partial_artifacts":
                    None if archived is None else str(archived),
                "stage1_real_fit_started": False,
                "development_numerical_arrays_opened": False,
                "validation_performance_inspected": False,
                "final_test_data_opened": False,
            },
        )
        print()
        print("=" * 140)
        print("STEP 26 FAILED — RUNNER NOT FROZEN")
        print("=" * 140)
        print(f"Stage      : {stage}")
        print("Error type : TimeoutExpired")
        print(f"Error      : {exc}")
        print(f"Diagnostic : {failure}")
        if archived is not None:
            print(f"Archived partial Step-26 outputs: {archived}")
        print("=" * 140)
        print("Do not proceed to real Stage-1 fitting.")
        return 30

    except Exception as exc:
        archived = archive_partial()
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        failure = FAILED_DIR / (
            "26_stage1_runner_dryrun_FAILED_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
        atomic_json(
            failure,
            {
                "status": "FAIL",
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "failed_at_utc": now(),
                "archived_partial_artifacts":
                    None if archived is None else str(archived),
                "stage1_real_fit_started": False,
                "development_numerical_arrays_opened": False,
                "validation_performance_inspected": False,
                "final_test_data_opened": False,
            },
        )
        print()
        print("=" * 140)
        print("STEP 26 FAILED — RUNNER NOT FROZEN")
        print("=" * 140)
        print(f"Stage      : {stage}")
        print(f"Error type : {type(exc).__name__}")
        print(f"Error      : {exc}")
        print(f"Diagnostic : {failure}")
        if archived is not None:
            print(f"Archived partial Step-26 outputs: {archived}")
        print("=" * 140)
        print("Do not proceed to real Stage-1 fitting.")
        return 30


# =============================================================================
# 13. CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--config-id")
    parser.add_argument("--physical-microbatch", type=int)
    parser.add_argument("--tail-sizes-json")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.worker:
        if (
            not args.model_id
            or not args.config_id
            or args.physical_microbatch is None
            or args.tail_sizes_json is None
        ):
            raise SystemExit(
                "--worker requires --model-id, --config-id, "
                "--physical-microbatch and --tail-sizes-json"
            )
        return worker_main(
            model_id=args.model_id,
            config_id=args.config_id,
            physical_microbatch=args.physical_microbatch,
            tail_sizes_json=args.tail_sizes_json,
        )

    if args.verify_only:
        return verify_existing()

    return run_master()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStep 26 interrupted by user.")
        raise SystemExit(130)
