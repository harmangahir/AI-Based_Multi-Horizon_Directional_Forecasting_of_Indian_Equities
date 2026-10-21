#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 25 — CLEAN V3 STAGE-1 TRAINING-ENGINE CONTRACT + SYNTHETIC EQUIVALENCE PREFLIGHT

Purpose
-------
Freeze the exact Stage-1 optimization engine BEFORE any of the 80 real
candidate/fold fits are launched.

This step does NOT repeat the Step-24 GPU-capacity search.  Step 24 already
froze one physical microbatch + accumulation schedule for each of the 16
candidate configurations while preserving logical/effective batch size 256.

Step 25 instead proves and freezes the optimization semantics that the later
Stage-1 runner must use:

  * 16 candidate configurations x 5 chronological folds = 80 fits;
  * root screening seed = 42;
  * logical/effective batch = 256;
  * physical microbatch comes only from the frozen Step-24 schedule;
  * FP32, AMP disabled, TF32 disabled;
  * AdamW, candidate-specific frozen LR, weight_decay=1e-4;
  * BCEWithLogitsLoss(reduction="mean"), no class weighting;
  * one zero_grad + one optimizer.step per logical batch;
  * exact sample-weighted gradient accumulation:
        micro_mean_BCE * (micro_n / logical_n)
    so incomplete final logical batches retain the exact mean objective;
  * gradient clipping once, after all microbatches of the logical batch;
  * no padding, duplication or dropping of a final partial logical batch;
  * deterministic epoch permutations derived only from seed/fold/epoch, so all
    candidates within the same fold/epoch see the same sample ordering;
  * max_epochs=100, min_epoch_before_stopping=10, patience=12;
  * strict checkpoint improvement on minimum validation BCE;
  * ties do not overwrite the earlier best checkpoint;
  * model restored to the best-validation-BCE checkpoint before Stage-1 metrics;
  * fixed Stage-1 classification threshold = 0.50;
  * validation is sequential/no-shuffle and uses exact sample-weighted BCE;
  * no scheduler and no best-seed selection.

Synthetic tests
---------------
The step tests full-batch versus microbatch/accumulation equivalence using a
small deterministic directional network (not a forecasting candidate) on:

  1. a complete logical batch N=256; and
  2. an intentionally incomplete logical batch N=173.

Every unique physical microbatch frozen by Step 24 is tested.  The tests compare the aggregate BCE objective, pre-clip gradients, post-clip
optimizer update, and validation BCE aggregation for a stateless deterministic
reference network. CPU float64 provides a high-precision algebraic check; CUDA
float32 checks the numerical implementation of the accumulation engine. This
does NOT claim that a candidate containing batch-coupled modules (for example,
BatchNorm) is mathematically identical under different physical microbatches;
the Step-24 physical microbatch is therefore part of each candidate's frozen
execution contract.

No third-party forecasting architecture is fitted here.  No development-array
numerical values are opened.  Only the already-frozen Step-24 split-summary
counts are read.  No final-test data are opened or hashed.

Run
---
    python step25_stage1_training_engine_contract_v3.py

Verify after an audited PASS
----------------------------
    python step25_stage1_training_engine_contract_v3.py --verify-only
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
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import step00_config as cfg


# =============================================================================
# 1. FROZEN CONSTANTS / PATHS
# =============================================================================

STEP_NUMBER = 25
STEP_NAME = "STEP 25 — STAGE-1 TRAINING-ENGINE CONTRACT + SYNTHETIC EQUIVALENCE PREFLIGHT"
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

EXPECTED_CANDIDATES = 16
EXPECTED_FOLDS = 5
EXPECTED_STAGE1_FITS = 80

SEED = 42
LOGICAL_BATCH_SIZE = 256
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
MAX_EPOCHS = 100
MIN_EPOCH_BEFORE_STOPPING = 10
EARLY_STOPPING_PATIENCE = 12
SCREENING_THRESHOLD = 0.50

EXPECTED_OPTIMIZER = "AdamW"
EXPECTED_LOSS = "BCEWithLogitsLoss(reduction='mean')"
EXPECTED_CLASS_WEIGHTING = "none"
EXPECTED_CHECKPOINT_METRIC = "validation_mean_directional_bce"
EXPECTED_CHECKPOINT_MODE = "min"
EXPECTED_CHECKPOINT_TIE_BREAK = (
    "earlier epoch if validation_mean_directional_bce is exactly tied"
)

FULL_TEST_N = 256
PARTIAL_TEST_N = 173

CPU_LOSS_TOL = 1e-12
CPU_GRAD_TOL = 1e-11
CPU_PARAM_TOL = 1e-12

CUDA_LOSS_TOL = 5e-6
CUDA_GRAD_TOL = 5e-5
CUDA_PARAM_TOL = 5e-6

SCRIPT = Path(__file__).resolve()
Q1_DIR = SCRIPT.parent

STEP16_FREEZE = cfg.MANIFEST_DIR / "16_stage1_model_registry_freeze.json"
STEP16_MANIFEST = cfg.MANIFEST_DIR / "16_stage1_model_registry_manifest.json"
STEP16_MANIFEST_SHA = cfg.MANIFEST_DIR / "16_stage1_model_registry_manifest.sha256"

STEP24_FREEZE = cfg.MANIFEST_DIR / "24_training_runtime_preflight_freeze.json"
STEP24_MANIFEST = cfg.MANIFEST_DIR / "24_training_runtime_preflight_manifest.json"
STEP24_MANIFEST_SHA = cfg.MANIFEST_DIR / "24_training_runtime_preflight_manifest.sha256"

STEP10_TEST_SEAL = cfg.MANIFEST_DIR / "10_final_test_seal.json"
FORBIDDEN_TEST_FILE = "10_final_test_locked_panel.parquet"

CONTRACT_CSV = cfg.AUDIT_DIR / "25_stage1_training_engine_contract.csv"
BATCH_PLAN_CSV = cfg.AUDIT_DIR / "25_stage1_fold_batch_plan.csv"
EQUIVALENCE_CSV = cfg.AUDIT_DIR / "25_gradient_accumulation_equivalence.csv"
SHUFFLE_CSV = cfg.AUDIT_DIR / "25_deterministic_shuffle_audit.csv"
EARLY_STOP_CSV = cfg.AUDIT_DIR / "25_early_stopping_state_machine_audit.csv"
SOURCE_HASH_CSV = cfg.AUDIT_DIR / "25_training_engine_source_hashes.csv"

FREEZE_JSON = cfg.MANIFEST_DIR / "25_stage1_training_engine_contract_freeze.json"
MANIFEST_JSON = cfg.MANIFEST_DIR / "25_stage1_training_engine_contract_manifest.json"
MANIFEST_SHA = cfg.MANIFEST_DIR / "25_stage1_training_engine_contract_manifest.sha256"
FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step25_stage1_training_engine_contract_frozen.py"
SUMMARY_TXT = cfg.REPORT_DIR / "25_stage1_training_engine_contract_summary.txt"
FAILED_DIR = cfg.AUDIT_DIR / "step25_failed_attempts"


# =============================================================================
# 2. GENERIC HELPERS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(payload: Any) -> str:
    return cfg.canonical_json_sha256(payload)


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


def verify_sidecar(manifest_path: Path, sidecar_path: Path, label: str) -> str:
    if not manifest_path.exists() or not sidecar_path.exists():
        raise FileNotFoundError(f"{label} manifest/sidecar missing.")
    expected = sidecar_path.read_text(encoding="utf-8").strip().split()[0]
    observed = sha256_file(manifest_path)
    if expected != observed:
        raise RuntimeError(
            f"{label} manifest SHA mismatch.\nExpected: {expected}\nObserved: {observed}"
        )
    return observed


def recompute_freeze_fingerprint(payload: dict, field: str, remove_generated_at: bool = False) -> str:
    body = dict(payload)
    body.pop(field, None)
    if remove_generated_at:
        body.pop("generated_at_utc", None)
    return canonical_hash(body)


def artifact_by_name(manifest: dict, filename: str) -> dict:
    matches = []
    for key in ("artifacts", "canonical_artifacts"):
        container = manifest.get(key, {})
        records = container.values() if isinstance(container, dict) else container if isinstance(container, list) else []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            p = rec.get("path") or rec.get("file_path")
            if p and Path(p).name == filename:
                matches.append(rec)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one manifest artifact named {filename}; found {len(matches)}.")
    return matches[0]


def check_artifact_record(record: dict) -> Path:
    path = Path(record.get("path") or record.get("file_path"))
    if path.name == FORBIDDEN_TEST_FILE:
        raise RuntimeError("Final-test parquet entered Step-25 artifact verification.")
    if not path.is_file():
        raise FileNotFoundError(path)
    if "sha256" in record and sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Artifact hash mismatch: {path}")
    return path


def canonical_outputs() -> tuple[Path, ...]:
    return (
        CONTRACT_CSV,
        BATCH_PLAN_CSV,
        EQUIVALENCE_CSV,
        SHUFFLE_CSV,
        EARLY_STOP_CSV,
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
            "Step-25 canonical output(s) already exist; refusing overwrite:\n"
            + "\n".join(str(p) for p in existing)
            + "\nUse --verify-only after a successful frozen run."
        )


def archive_partial() -> Path | None:
    if MANIFEST_JSON.exists():
        return None
    existing = [p for p in canonical_outputs() if p.exists()]
    if not existing:
        return None
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    dst = FAILED_DIR / ("partial_artifacts_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    dst.mkdir()
    for path in existing:
        try:
            shutil.move(str(path), str(dst / path.name))
        except Exception:
            pass
    return dst


# =============================================================================
# 3. VERIFY STEP 00 / STEP 16 / STEP 24 / FINAL-TEST PROTECTION
# =============================================================================

def verify_step00() -> dict:
    cfg.validate_configuration()
    payload = cfg.build_configuration_payload()
    fingerprint = cfg.canonical_json_sha256(payload)

    if fingerprint != EXPECTED_CONFIGURATION_FINGERPRINT:
        raise RuntimeError(
            "Active configuration fingerprint drift.\n"
            f"Expected: {EXPECTED_CONFIGURATION_FINGERPRINT}\n"
            f"Observed: {fingerprint}"
        )

    _, manifest_sha, status = cfg.verify_existing_freeze(
        configuration_payload=payload,
        configuration_fingerprint=fingerprint,
    )
    if status != "VERIFIED_EXISTING":
        raise RuntimeError("Step-00 freeze is not VERIFIED_EXISTING.")

    lock_path = cfg.get_final_test_lock_path()
    lock = load_json(lock_path)
    if (
        lock.get("state") != "LOCKED_NOT_OPENED"
        or lock.get("final_test_opened") is not False
        or lock.get("final_test_scored") is not False
    ):
        raise RuntimeError("Global final-test lock is not pristine.")

    return {
        "fingerprint": fingerprint,
        "step00_manifest_sha256": manifest_sha,
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "lock_state": lock["state"],
    }


def verify_upstream() -> dict:
    step00 = verify_step00()

    step16_manifest_sha = verify_sidecar(
        STEP16_MANIFEST, STEP16_MANIFEST_SHA, "Step 16"
    )
    step24_manifest_sha = verify_sidecar(
        STEP24_MANIFEST, STEP24_MANIFEST_SHA, "Step 24"
    )

    if step24_manifest_sha != EXPECTED_STEP24_MANIFEST_SHA256:
        raise RuntimeError(
            "Unexpected Step-24 manifest SHA.\n"
            f"Expected: {EXPECTED_STEP24_MANIFEST_SHA256}\n"
            f"Observed: {step24_manifest_sha}"
        )

    step16_manifest = load_json(STEP16_MANIFEST)
    step24_manifest = load_json(STEP24_MANIFEST)
    step16 = load_json(STEP16_FREEZE)
    step24 = load_json(STEP24_FREEZE)

    if step24_manifest.get("step", {}).get("status") != "PASS":
        raise RuntimeError("Step-24 manifest status is not PASS.")
    if step24_manifest.get("configuration_fingerprint_sha256") != step00["fingerprint"]:
        raise RuntimeError("Step-24 configuration fingerprint mismatch.")

    stored24 = step24["training_runtime_preflight_fingerprint_sha256"]
    recomputed24 = recompute_freeze_fingerprint(
        step24,
        "training_runtime_preflight_fingerprint_sha256",
        remove_generated_at=True,
    )
    if stored24 != recomputed24 or stored24 != EXPECTED_STEP24_FINGERPRINT:
        raise RuntimeError(
            "Step-24 freeze fingerprint mismatch.\n"
            f"Expected: {EXPECTED_STEP24_FINGERPRINT}\n"
            f"Stored: {stored24}\nRecomputed: {recomputed24}"
        )

    if step24.get("status") != "STAGE1_DEVELOPMENT_TRAINING_RUNTIME_PREFLIGHT_FROZEN":
        raise RuntimeError("Unexpected Step-24 freeze status.")

    if (
        step24["final_test_protection"]["step10_seal_status"] != "SEALED_NOT_OPENED"
        or step24["final_test_protection"]["sealed_test_parquet_opened_or_hashed_by_step24"] is not False
        or step24["final_test_protection"]["global_final_test_state"] != "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError("Step-24 final-test protection drift.")

    seal = load_json(STEP10_TEST_SEAL)
    if (
        seal.get("status") != "SEALED_NOT_OPENED"
        or seal.get("global_final_test_lock_state") != "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError("Step-10 final-test seal is not pristine.")

    # Verify Step-16 embedded fingerprint and manifest relationship.
    stored16 = step16["registry_freeze_fingerprint_sha256"]
    recomputed16 = recompute_freeze_fingerprint(
        step16,
        "registry_freeze_fingerprint_sha256",
        remove_generated_at=False,
    )
    if stored16 != recomputed16:
        raise RuntimeError("Step-16 registry freeze fingerprint no longer recomputes.")
    if step16_manifest.get("registry_freeze_fingerprint_sha256") != stored16:
        raise RuntimeError("Step-16 manifest/freeze fingerprint mismatch.")

    # Frozen clean-V3 Stage-1 protocol.
    training_protocol = step16["training_protocol"]
    optimization = training_protocol["optimization"]
    data_contract = training_protocol["data_contract"]
    screening = training_protocol["screening"]
    reproducibility = training_protocol["reproducibility"]
    stage1 = step16["selection_protocol"]["stage1_configuration_selection"]

    expected_optimization = {
        "batch_size": LOGICAL_BATCH_SIZE,
        "optimizer": EXPECTED_OPTIMIZER,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "minimum_epoch_before_stopping": MIN_EPOCH_BEFORE_STOPPING,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "weight_decay": WEIGHT_DECAY,
        "direction_loss": EXPECTED_LOSS,
        "direction_loss_horizon_aggregation":
            "unweighted arithmetic mean across 1d, 3d and 5d BCE",
        "checkpoint_metric": EXPECTED_CHECKPOINT_METRIC,
        "checkpoint_mode": EXPECTED_CHECKPOINT_MODE,
        "checkpoint_tie_break": EXPECTED_CHECKPOINT_TIE_BREAK,
        "mixed_precision_amp": False,
        "numeric_dtype": "float32",
    }

    for key, expected in expected_optimization.items():
        observed = optimization.get(key)
        if observed != expected:
            raise RuntimeError(
                f"Step-16 optimization drift: {key}; "
                f"expected {expected!r}, observed {observed!r}"
            )

    expected_data_contract = {
        "class_weighting": EXPECTED_CLASS_WEIGHTING,
        "continuous_predictors": 63,
        "direction_outputs": 3,
        "feature_selection": False,
        "folds": [1, 2, 3, 4, 5],
        "lookback_sessions": 60,
        "uses_auxiliary_regression": False,
        "uses_regime_id": False,
        "uses_stock_id": False,
    }

    for key, expected in expected_data_contract.items():
        observed = data_contract.get(key)
        if observed != expected:
            raise RuntimeError(
                f"Step-16 data-contract drift: {key}; "
                f"expected {expected!r}, observed {observed!r}"
            )

    expected_screening = {
        "fits_per_configuration": EXPECTED_FOLDS,
        "probability_threshold": SCREENING_THRESHOLD,
        "seed": SEED,
        "threshold_tuned_during_stage1": False,
        "validation_fold_predictions_retained": True,
        "validation_metadata_retained": True,
    }

    for key, expected in expected_screening.items():
        observed = screening.get(key)
        if observed != expected:
            raise RuntimeError(
                f"Step-16 screening drift: {key}; "
                f"expected {expected!r}, observed {observed!r}"
            )

    expected_reproducibility = {
        "all_random_generators_seeded_from_run_seed": True,
        "best_seed_selection": False,
        "deterministic_algorithms_required_where_supported": True,
        "screening_seed_is_not_a_promoted_best_seed": True,
        "training_shuffle": True,
        "validation_shuffle": False,
    }

    for key, expected in expected_reproducibility.items():
        observed = reproducibility.get(key)
        if observed != expected:
            raise RuntimeError(
                f"Step-16 reproducibility drift: {key}; "
                f"expected {expected!r}, observed {observed!r}"
            )

    if int(stage1["screening_seed"]) != SEED:
        raise RuntimeError("Stage-1 screening seed drift.")
    if list(stage1["folds"]) != [1, 2, 3, 4, 5]:
        raise RuntimeError("Stage-1 fold contract drift.")
    if float(stage1["screening_threshold"]) != SCREENING_THRESHOLD:
        raise RuntimeError("Stage-1 threshold drift.")
    if bool(stage1["best_seed_selection"]):
        raise RuntimeError("Best-seed selection unexpectedly enabled.")
    if bool(stage1["test_used"]):
        raise RuntimeError("Step-16 Stage-1 selection unexpectedly uses test data.")
    if bool(stage1["no_fold_weighting_by_sample_count"]) is not True:
        raise RuntimeError("Step-16 fold-weighting contract drift.")
    if bool(stage1["no_horizon_weighting"]) is not True:
        raise RuntimeError("Step-16 horizon-weighting contract drift.")

    # Step-24 schedule must contain all 16 candidates at effective batch 256.
    schedule = step24["candidate_microbatch_schedule"]
    if len(schedule) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"Expected 16 Step-24 schedules; found {len(schedule)}.")

    seen = set()
    for row in schedule:
        key = (row["model_id"], row["config_id"])
        if key in seen:
            raise RuntimeError(f"Duplicate Step-24 schedule: {key}")
        seen.add(key)
        physical = int(row["physical_microbatch_size"])
        accum = int(row["gradient_accumulation_steps"])
        logical = int(row["logical_batch_size"])
        effective = int(row["effective_batch_size"])
        if logical != LOGICAL_BATCH_SIZE or effective != LOGICAL_BATCH_SIZE:
            raise RuntimeError(f"Effective-batch drift in Step-24 schedule: {key}")
        if physical * accum != LOGICAL_BATCH_SIZE:
            raise RuntimeError(f"Microbatch/accumulation product mismatch: {key}")

    return {
        "step00": step00,
        "step16": step16,
        "step16_manifest": step16_manifest,
        "step16_manifest_sha256": step16_manifest_sha,
        "step16_fingerprint": stored16,
        "step24": step24,
        "step24_manifest": step24_manifest,
        "step24_manifest_sha256": step24_manifest_sha,
        "step24_fingerprint": stored24,
        "training_protocol": training_protocol,
        "optimization": optimization,
        "data_contract": data_contract,
        "screening": screening,
        "reproducibility": reproducibility,
        "stage1": stage1,
        "schedule": schedule,
        "test_seal_status": seal["status"],
    }


# =============================================================================
# 4. DEVELOPMENT COUNT / BATCH-PLAN AUDIT (NO NUMERICAL ARRAY LOAD)
# =============================================================================

def load_step24_split_summary(step24_manifest: dict) -> pd.DataFrame:
    record = artifact_by_name(step24_manifest, "24_development_split_summary.csv")
    path = check_artifact_record(record)
    frame = pd.read_csv(path)

    required = {
        "fold", "partition", "n_sequences", "lookback",
        "feature_count", "direction_outputs",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("Step-24 split-summary columns are incomplete.")

    if len(frame) != EXPECTED_FOLDS * 2:
        raise RuntimeError("Expected 10 Step-24 fold/partition records.")

    return frame


def microbatch_partition_sizes(logical_n: int, physical_mb: int) -> list[int]:
    if logical_n <= 0 or physical_mb <= 0:
        raise ValueError("Batch sizes must be positive.")
    return [
        min(physical_mb, logical_n - start)
        for start in range(0, logical_n, physical_mb)
    ]


def build_fold_batch_plan(schedule: list[dict], split: pd.DataFrame) -> pd.DataFrame:
    train_counts = {}
    val_counts = {}

    for fold in range(1, EXPECTED_FOLDS + 1):
        tr = split[(split["fold"] == fold) & (split["partition"] == "train")]
        va = split[(split["fold"] == fold) & (split["partition"] == "validation")]
        if len(tr) != 1 or len(va) != 1:
            raise RuntimeError(f"Fold {fold}: expected one train and one validation count.")
        train_counts[fold] = int(tr.iloc[0]["n_sequences"])
        val_counts[fold] = int(va.iloc[0]["n_sequences"])

    rows = []

    for sched in sorted(schedule, key=lambda r: (r["model_id"], r["config_id"])):
        physical = int(sched["physical_microbatch_size"])

        for fold in range(1, EXPECTED_FOLDS + 1):
            train_n = train_counts[fold]
            val_n = val_counts[fold]
            logical_batches = math.ceil(train_n / LOGICAL_BATCH_SIZE)
            final_n = train_n % LOGICAL_BATCH_SIZE or LOGICAL_BATCH_SIZE

            final_parts = microbatch_partition_sizes(final_n, physical)
            weight_sum = sum(n / final_n for n in final_parts)

            if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-15):
                raise RuntimeError(
                    f"Final logical-batch microbatch weights do not sum to one: "
                    f"{sched['model_id']}/{sched['config_id']}/fold{fold}"
                )

            if sum(final_parts) != final_n:
                raise RuntimeError("Final microbatch partition does not preserve sample count.")

            rows.append(
                {
                    "model_id": sched["model_id"],
                    "config_id": sched["config_id"],
                    "fold": fold,
                    "train_sequences": train_n,
                    "validation_sequences": val_n,
                    "logical_batch_size": LOGICAL_BATCH_SIZE,
                    "physical_microbatch_size": physical,
                    "full_logical_accumulation_steps": int(
                        sched["gradient_accumulation_steps"]
                    ),
                    "train_logical_batches_per_epoch": logical_batches,
                    "final_logical_batch_size": final_n,
                    "final_logical_batch_is_partial": final_n < LOGICAL_BATCH_SIZE,
                    "final_logical_microbatch_count": len(final_parts),
                    "final_logical_partition_json": json.dumps(final_parts),
                    "final_loss_weight_sum": weight_sum,
                    "drop_last": False,
                    "pad_final_batch": False,
                    "duplicate_final_samples": False,
                }
            )

    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_STAGE1_FITS:
        raise RuntimeError(
            f"Expected {EXPECTED_STAGE1_FITS} candidate/fold batch plans; found {len(frame)}."
        )

    return frame


# =============================================================================
# 5. DETERMINISTIC SHUFFLING CONTRACT
# =============================================================================

def permutation_for_epoch(n: int, fold: int, epoch: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    if fold not in range(1, EXPECTED_FOLDS + 1):
        raise ValueError("fold must be 1..5.")
    if epoch < 1:
        raise ValueError("epoch is 1-indexed.")

    # Candidate ID is deliberately NOT included: every candidate in the same
    # fold/epoch receives the same sample ordering.
    seed_sequence = np.random.SeedSequence([SEED, fold, epoch])
    rng = np.random.default_rng(seed_sequence)
    return rng.permutation(n)


def audit_shuffle(split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    digests = set()

    for fold in range(1, EXPECTED_FOLDS + 1):
        tr = split[(split["fold"] == fold) & (split["partition"] == "train")]
        n = int(tr.iloc[0]["n_sequences"])

        for epoch in (1, 2, MAX_EPOCHS):
            p1 = permutation_for_epoch(n, fold, epoch)
            p2 = permutation_for_epoch(n, fold, epoch)

            if not np.array_equal(p1, p2):
                raise RuntimeError("Deterministic permutation reproducibility failed.")
            if len(p1) != n or len(np.unique(p1)) != n:
                raise RuntimeError("Permutation is not a bijection over training indices.")

            digest = hashlib.sha256(p1.astype(np.int64, copy=False).tobytes()).hexdigest()
            digests.add(digest)

            rows.append(
                {
                    "root_seed": SEED,
                    "fold": fold,
                    "epoch": epoch,
                    "n_sequences": n,
                    "candidate_id_in_seed": False,
                    "seed_derivation": "numpy.random.SeedSequence([42, fold, epoch])",
                    "permutation_sha256": digest,
                    "repeated_generation_identical": True,
                    "is_bijection": True,
                }
            )

    if len(digests) != len(rows):
        raise RuntimeError("Unexpected duplicate deterministic permutation digest.")

    return pd.DataFrame(rows)


# =============================================================================
# 6. SYNTHETIC GRADIENT / OPTIMIZER / VALIDATION EQUIVALENCE
# =============================================================================

def configure_torch_runtime():
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

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


def tiny_model(dtype, device):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(7, 11, bias=True),
        nn.Tanh(),
        nn.Linear(11, 3, bias=True),
    )
    return model.to(device=device, dtype=dtype)


def synthetic_xy(n: int, dtype, device):
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + n)

    x_cpu = torch.randn((n, 7), generator=gen, dtype=torch.float64)
    rows = torch.arange(n, dtype=torch.int64).unsqueeze(1)
    cols = torch.arange(3, dtype=torch.int64).unsqueeze(0)
    y_cpu = ((rows + cols) % 2).to(torch.float64)

    return x_cpu.to(device=device, dtype=dtype), y_cpu.to(device=device, dtype=dtype)


def clone_gradients(model) -> dict[str, Any]:
    out = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Missing synthetic gradient: {name}")
        out[name] = parameter.grad.detach().clone()
    return out


def max_tensor_dict_difference(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    if set(a) != set(b):
        raise RuntimeError("Tensor dictionary key mismatch.")
    max_diff = 0.0
    for key in keys:
        diff = float((a[key] - b[key]).abs().max().detach().cpu().item())
        max_diff = max(max_diff, diff)
    return max_diff


def state_dict_clone(model) -> dict[str, Any]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def validation_bce_chunked(model, x, y, microbatch: int) -> float:
    import torch
    import torch.nn as nn

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    n = int(x.shape[0])
    total = 0.0

    model.eval()
    with torch.no_grad():
        for start in range(0, n, microbatch):
            end = min(start + microbatch, n)
            logits = model(x[start:end])
            loss = criterion(logits, y[start:end])
            total += float(loss.detach().cpu().item()) * ((end - start) / n)

    return total


def run_equivalence_case(device_name: str, dtype_name: str, n: int, microbatch: int) -> dict:
    import torch
    import torch.nn as nn

    device = torch.device(device_name)
    dtype = torch.float64 if dtype_name == "float64" else torch.float32

    configure_torch_runtime()

    base = tiny_model(dtype, device)
    initial = copy.deepcopy(base.state_dict())

    direct = tiny_model(dtype, device)
    direct.load_state_dict(initial)
    accum = tiny_model(dtype, device)
    accum.load_state_dict(initial)

    x, y = synthetic_xy(n, dtype, device)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    optimizer_direct = torch.optim.AdamW(
        direct.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_accum = torch.optim.AdamW(
        accum.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    # Direct logical-batch update.
    direct.train()
    optimizer_direct.zero_grad(set_to_none=True)
    direct_logits = direct(x)
    direct_loss = criterion(direct_logits, y)
    direct_loss.backward()
    direct_grads = clone_gradients(direct)
    direct_preclip_norm = torch.nn.utils.clip_grad_norm_(
        direct.parameters(), GRADIENT_CLIP_NORM
    )
    optimizer_direct.step()
    direct_after = state_dict_clone(direct)

    # Exact sample-weighted accumulated logical-batch update.
    accum.train()
    optimizer_accum.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    partition = []

    for start in range(0, n, microbatch):
        end = min(start + microbatch, n)
        micro_n = end - start
        partition.append(micro_n)

        logits = accum(x[start:end])
        micro_loss = criterion(logits, y[start:end])
        weight = micro_n / n
        (micro_loss * weight).backward()
        accumulated_loss += float(micro_loss.detach().cpu().item()) * weight

    accum_grads = clone_gradients(accum)
    accum_preclip_norm = torch.nn.utils.clip_grad_norm_(
        accum.parameters(), GRADIENT_CLIP_NORM
    )
    optimizer_accum.step()
    accum_after = state_dict_clone(accum)

    # Validation aggregation check on the same fixed model state.
    validator = tiny_model(dtype, device)
    validator.load_state_dict(initial)
    validator.eval()
    with torch.no_grad():
        validation_full = float(
            criterion(validator(x), y).detach().cpu().item()
        )
    validation_chunked = validation_bce_chunked(
        validator, x, y, microbatch
    )

    loss_diff = abs(float(direct_loss.detach().cpu().item()) - accumulated_loss)
    grad_diff = max_tensor_dict_difference(direct_grads, accum_grads)
    param_diff = max_tensor_dict_difference(direct_after, accum_after)
    val_diff = abs(validation_full - validation_chunked)
    norm_diff = abs(
        float(direct_preclip_norm.detach().cpu().item())
        - float(accum_preclip_norm.detach().cpu().item())
    )

    if dtype_name == "float64":
        loss_tol = CPU_LOSS_TOL
        grad_tol = CPU_GRAD_TOL
        param_tol = CPU_PARAM_TOL
    else:
        loss_tol = CUDA_LOSS_TOL
        grad_tol = CUDA_GRAD_TOL
        param_tol = CUDA_PARAM_TOL

    passed = bool(
        loss_diff <= loss_tol
        and val_diff <= loss_tol
        and grad_diff <= grad_tol
        and norm_diff <= grad_tol
        and param_diff <= param_tol
        and sum(partition) == n
        and math.isclose(
            sum(size / n for size in partition),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    )

    if not passed:
        raise RuntimeError(
            "Gradient-accumulation equivalence failed.\n"
            f"device={device_name}, dtype={dtype_name}, N={n}, mb={microbatch}\n"
            f"loss_diff={loss_diff}, grad_diff={grad_diff}, "
            f"norm_diff={norm_diff}, param_diff={param_diff}, val_diff={val_diff}"
        )

    return {
        "device": device_name,
        "dtype": dtype_name,
        "logical_batch_n": n,
        "physical_microbatch_size": microbatch,
        "microbatch_count": len(partition),
        "partition_json": json.dumps(partition),
        "partition_weight_sum": sum(size / n for size in partition),
        "direct_bce": float(direct_loss.detach().cpu().item()),
        "accumulated_weighted_bce": accumulated_loss,
        "loss_abs_diff": loss_diff,
        "preclip_gradient_max_abs_diff": grad_diff,
        "preclip_norm_abs_diff": norm_diff,
        "post_adamw_parameter_max_abs_diff": param_diff,
        "validation_full_bce": validation_full,
        "validation_chunked_bce": validation_chunked,
        "validation_abs_diff": val_diff,
        "loss_tolerance": loss_tol,
        "gradient_tolerance": grad_tol,
        "parameter_tolerance": param_tol,
        "gradient_clip_once_after_full_logical_batch": True,
        "optimizer_steps_per_logical_batch": 1,
        "passed": True,
    }


def run_equivalence_suite(schedule: list[dict]) -> pd.DataFrame:
    import torch

    unique_microbatches = sorted(
        {int(row["physical_microbatch_size"]) for row in schedule},
        reverse=True,
    )

    rows = []

    # High-precision algebraic reference.
    for n in (FULL_TEST_N, PARTIAL_TEST_N):
        for mb in unique_microbatches:
            rows.append(
                run_equivalence_case(
                    device_name="cpu",
                    dtype_name="float64",
                    n=n,
                    microbatch=mb,
                )
            )

    # Actual frozen Stage-1 numerical runtime.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for Step-25 FP32 equivalence checks.")

    for n in (FULL_TEST_N, PARTIAL_TEST_N):
        for mb in unique_microbatches:
            rows.append(
                run_equivalence_case(
                    device_name="cuda:0",
                    dtype_name="float32",
                    n=n,
                    microbatch=mb,
                )
            )

    frame = pd.DataFrame(rows)

    if not frame["passed"].all():
        raise RuntimeError("At least one accumulation-equivalence record failed.")

    return frame


# =============================================================================
# 7. EARLY-STOPPING / CHECKPOINT STATE-MACHINE AUDIT
# =============================================================================

def simulate_early_stopping(losses: list[float]) -> pd.DataFrame:
    best = math.inf
    best_epoch = None
    stale = 0
    rows = []

    for epoch, val_bce in enumerate(losses, start=1):
        improved = val_bce < best

        if improved:
            best = val_bce
            best_epoch = epoch
            stale = 0
            checkpoint_replaced = True
        else:
            stale += 1
            checkpoint_replaced = False

        stop = bool(
            epoch >= MIN_EPOCH_BEFORE_STOPPING
            and stale >= EARLY_STOPPING_PATIENCE
        )

        rows.append(
            {
                "epoch": epoch,
                "validation_bce": val_bce,
                "strict_improvement": improved,
                "checkpoint_replaced": checkpoint_replaced,
                "best_epoch_after_epoch": best_epoch,
                "best_validation_bce_after_epoch": best,
                "consecutive_nonimproving_epochs": stale,
                "early_stop_allowed_by_min_epoch": (
                    epoch >= MIN_EPOCH_BEFORE_STOPPING
                ),
                "stop_after_epoch": stop,
            }
        )

        if stop:
            break

    return pd.DataFrame(rows)


def audit_early_stopping() -> pd.DataFrame:
    records = []

    # Case A: best at epoch 1, then no improvement. With patience=12,
    # stopping occurs after epoch 13; the min-epoch gate is also satisfied.
    case_a = [0.50] + [0.51] * 20
    a = simulate_early_stopping(case_a)
    stop_rows = a[a["stop_after_epoch"]]
    if len(stop_rows) != 1 or int(stop_rows.iloc[0]["epoch"]) != 13:
        raise RuntimeError("Early-stopping patience semantics failed case A.")
    if int(stop_rows.iloc[0]["best_epoch_after_epoch"]) != 1:
        raise RuntimeError("Best-checkpoint epoch failed case A.")
    a.insert(0, "case", "patience_after_epoch1_best")
    records.append(a)

    # Case B: strict improvement at epoch 8 resets patience.
    case_b = [0.50] + [0.51] * 6 + [0.49] + [0.50] * 20
    b = simulate_early_stopping(case_b)
    stop_rows = b[b["stop_after_epoch"]]
    if len(stop_rows) != 1 or int(stop_rows.iloc[0]["epoch"]) != 20:
        raise RuntimeError("Early-stopping patience-reset semantics failed case B.")
    if int(stop_rows.iloc[0]["best_epoch_after_epoch"]) != 8:
        raise RuntimeError("Best-checkpoint epoch failed case B.")
    b.insert(0, "case", "patience_reset_by_epoch8_improvement")
    records.append(b)

    # Case C: exact ties do not replace the earlier checkpoint.
    case_c = [0.50, 0.49, 0.49, 0.49, 0.48]
    c = simulate_early_stopping(case_c)
    tie_rows = c[c["epoch"].isin([3, 4])]
    if tie_rows["checkpoint_replaced"].any():
        raise RuntimeError("Validation-BCE ties unexpectedly overwrite checkpoint.")
    if int(c.iloc[-1]["best_epoch_after_epoch"]) != 5:
        raise RuntimeError("Strict checkpoint improvement semantics failed case C.")
    c.insert(0, "case", "ties_keep_earlier_checkpoint")
    records.append(c)

    out = pd.concat(records, ignore_index=True)

    return out


# =============================================================================
# 8. CONTRACT OUTPUT
# =============================================================================

def build_contract_frame(upstream: dict, batch_plan: pd.DataFrame) -> pd.DataFrame:
    schedule_by_key = {
        (r["model_id"], r["config_id"]): r
        for r in upstream["schedule"]
    }

    rows = []
    for key in sorted(schedule_by_key):
        r = schedule_by_key[key]
        rows.append(
            {
                "model_id": r["model_id"],
                "config_id": r["config_id"],
                "stage1_folds": "1,2,3,4,5",
                "stage1_fit_count_for_candidate": EXPECTED_FOLDS,
                "screening_seed": SEED,
                "logical_batch_size": LOGICAL_BATCH_SIZE,
                "physical_microbatch_size": int(r["physical_microbatch_size"]),
                "full_logical_accumulation_steps": int(r["gradient_accumulation_steps"]),
                "effective_batch_size": int(r["effective_batch_size"]),
                "optimizer": EXPECTED_OPTIMIZER,
                "weight_decay": WEIGHT_DECAY,
                "loss": "BCEWithLogitsLoss(reduction='mean')",
                "class_weighting": EXPECTED_CLASS_WEIGHTING,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "max_epochs": MAX_EPOCHS,
                "minimum_epoch_before_stopping": MIN_EPOCH_BEFORE_STOPPING,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "checkpoint_metric": EXPECTED_CHECKPOINT_METRIC,
                "checkpoint_mode": EXPECTED_CHECKPOINT_MODE,
                "checkpoint_tie_break": EXPECTED_CHECKPOINT_TIE_BREAK,
                "checkpoint_improvement_operator": "strict_less_than",
                "screening_threshold": SCREENING_THRESHOLD,
                "scheduler": "none",
                "amp_enabled": False,
                "tf32_enabled": False,
                "dtype": "float32",
                "drop_last": False,
                "shuffle_training": True,
                "shuffle_validation": False,
                "shuffle_seed_contract": "SeedSequence([42, fold, epoch]); candidate-independent",
                "partial_logical_batch_policy": (
                    "preserve all samples; partition into <=physical_microbatch_size; "
                    "weight each microbatch mean BCE by micro_n/logical_n"
                ),
                "zero_grad_policy": "once before each logical batch",
                "gradient_clip_policy": "once after complete logical-batch accumulation",
                "optimizer_step_policy": "exactly once per logical batch",
                "validation_bce_policy": "exact sample-weighted mean across validation microbatches",
                "restore_policy": "restore strict minimum-validation-BCE checkpoint before metrics",
                "best_seed_selection": False,
                "validation_used_for": "checkpointing and prespecified early stopping only",
                "final_test_access": "prohibited",
            }
        )

    frame = pd.DataFrame(rows)

    if len(frame) != EXPECTED_CANDIDATES:
        raise RuntimeError("Training-engine contract does not contain 16 candidates.")

    if int(batch_plan.groupby(["model_id", "config_id"]).size().sum()) != EXPECTED_STAGE1_FITS:
        raise RuntimeError("Candidate/fold plan is not 80 fits.")

    return frame


# =============================================================================
# 9. PROVENANCE / FREEZE
# =============================================================================

def source_hash_frame(upstream: dict, split_summary_path: Path) -> pd.DataFrame:
    paths = [
        ("current_step00", Path(cfg.__file__).resolve()),
        ("current_step25", SCRIPT.resolve()),
        ("step16_freeze", STEP16_FREEZE.resolve()),
        ("step16_manifest", STEP16_MANIFEST.resolve()),
        ("step16_manifest_sha", STEP16_MANIFEST_SHA.resolve()),
        ("step24_freeze", STEP24_FREEZE.resolve()),
        ("step24_manifest", STEP24_MANIFEST.resolve()),
        ("step24_manifest_sha", STEP24_MANIFEST_SHA.resolve()),
        ("step24_split_summary", split_summary_path.resolve()),
        ("step10_final_test_seal_metadata", STEP10_TEST_SEAL.resolve()),
        ("global_final_test_lock_metadata", Path(upstream["step00"]["lock_path"]).resolve()),
    ]

    rows = []
    for role, path in paths:
        if path.name == FORBIDDEN_TEST_FILE:
            raise RuntimeError("Final-test parquet entered Step-25 provenance.")
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def step25_semantic_payload(
    upstream: dict,
    contract: pd.DataFrame,
    batch_plan: pd.DataFrame,
    equivalence: pd.DataFrame,
    shuffle: pd.DataFrame,
    early_stop: pd.DataFrame,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "STAGE1_TRAINING_ENGINE_CONTRACT_FROZEN",
        "configuration_fingerprint_sha256": upstream["step00"]["fingerprint"],
        "upstream": {
            "step16_registry_fingerprint_sha256": upstream["step16_fingerprint"],
            "step16_manifest_sha256": upstream["step16_manifest_sha256"],
            "step24_training_runtime_preflight_fingerprint_sha256": upstream["step24_fingerprint"],
            "step24_manifest_sha256": upstream["step24_manifest_sha256"],
        },
        "stage1_scope": {
            "candidate_configurations": EXPECTED_CANDIDATES,
            "chronological_folds": EXPECTED_FOLDS,
            "total_fits": EXPECTED_STAGE1_FITS,
            "seed": SEED,
        },
        "training_engine": {
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "physical_microbatch_source": "frozen Step-24 candidate schedule",
            "physical_microbatch_is_part_of_candidate_execution_contract": True,
            "optimizer": EXPECTED_OPTIMIZER,
            "weight_decay": WEIGHT_DECAY,
            "loss": "BCEWithLogitsLoss(reduction='mean')",
            "class_weighting": EXPECTED_CLASS_WEIGHTING,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "dtype": "float32",
            "amp_enabled": False,
            "tf32_enabled": False,
            "scheduler": "none",
            "zero_grad_policy": "once before each logical batch",
            "accumulation_loss_weight": "micro_n / logical_n",
            "gradient_clip_policy": "once after complete logical-batch accumulation",
            "optimizer_step_policy": "one AdamW step per logical batch",
            "partial_final_batch": {
                "drop": False,
                "pad": False,
                "duplicate": False,
                "use_actual_logical_batch_n": True,
            },
        },
        "data_ordering": {
            "training_shuffle": True,
            "validation_shuffle": False,
            "epoch_permutation": "numpy.random.SeedSequence([42, fold, epoch])",
            "candidate_id_participates_in_permutation_seed": False,
        },
        "stopping_checkpointing": {
            "max_epochs": MAX_EPOCHS,
            "minimum_epoch_before_stopping": MIN_EPOCH_BEFORE_STOPPING,
            "patience": EARLY_STOPPING_PATIENCE,
            "metric": EXPECTED_CHECKPOINT_METRIC,
            "mode": EXPECTED_CHECKPOINT_MODE,
            "strict_improvement": True,
            "min_delta": 0.0,
            "tie_policy": "retain earlier checkpoint",
            "stop_rule": (
                "after completing an epoch, stop iff epoch>=10 and "
                "consecutive_nonimproving_epochs>=12"
            ),
            "restore_best_checkpoint_before_metrics": True,
        },
        "stage1_prediction": {
            "threshold": SCREENING_THRESHOLD,
            "threshold_optimized_during_stage1": False,
        },
        "synthetic_accumulation_arithmetic_equivalence": {
            "complete_logical_batch_n": FULL_TEST_N,
            "partial_logical_batch_n": PARTIAL_TEST_N,
            "unique_physical_microbatches_tested": sorted(
                contract["physical_microbatch_size"].astype(int).unique().tolist(),
                reverse=True,
            ),
            "cpu_float64_reference_records": int(
                ((equivalence["device"] == "cpu") & (equivalence["dtype"] == "float64")).sum()
            ),
            "cuda_float32_records": int(
                ((equivalence["device"] == "cuda:0") & (equivalence["dtype"] == "float32")).sum()
            ),
            "all_passed": bool(equivalence["passed"].all()),
            "scope": (
                "loss-weighting/gradient-accumulation engine arithmetic on a "
                "stateless deterministic reference network; not an assertion "
                "of architecture-wide physical-batch equivalence"
            ),
        },
        "audits": {
            "candidate_contract_rows": int(len(contract)),
            "candidate_fold_batch_plan_rows": int(len(batch_plan)),
            "shuffle_audit_rows": int(len(shuffle)),
            "early_stopping_audit_rows": int(len(early_stop)),
        },
        "pipeline_state": {
            "third_party_forecasting_model_imported": False,
            "third_party_forecasting_model_fitted": False,
            "development_array_numerical_values_opened": False,
            "validation_performance_inspected": False,
            "stage1_fitting_started": False,
            "threshold_selection_performed": False,
            "best_seed_selection_performed": False,
        },
        "final_test_protection": {
            "step10_seal_status": upstream["test_seal_status"],
            "sealed_test_parquet_opened_or_hashed_by_step25": False,
            "final_test_performance_computed": False,
            "global_final_test_state": upstream["step00"]["lock_state"],
        },
    }


def freeze_outputs(upstream, split_path, contract, batch_plan, equivalence, shuffle, early_stop):
    semantic = step25_semantic_payload(
        upstream, contract, batch_plan, equivalence, shuffle, early_stop
    )
    fingerprint = canonical_hash(semantic)

    freeze = {
        **semantic,
        "generated_at_utc": utc_now(),
        "stage1_training_engine_fingerprint_sha256": fingerprint,
    }
    atomic_json(FREEZE_JSON, freeze)
    shutil.copy2(SCRIPT, FROZEN_SCRIPT)

    source_hashes = source_hash_frame(upstream, split_path)
    write_csv(SOURCE_HASH_CSV, source_hashes)

    summary_lines = [
        STEP_NAME,
        "=" * 112,
        "Status: PASS",
        f"Configuration fingerprint: {upstream['step00']['fingerprint']}",
        f"Step-24 preflight fingerprint: {upstream['step24_fingerprint']}",
        f"Stage-1 candidate configurations: {EXPECTED_CANDIDATES}",
        f"Chronological folds: {EXPECTED_FOLDS}",
        f"Total Stage-1 fits: {EXPECTED_STAGE1_FITS}",
        f"Root screening seed: {SEED}",
        f"Logical/effective batch: {LOGICAL_BATCH_SIZE}",
        "Physical microbatch: frozen Step-24 candidate schedule",
        "Optimizer: AdamW",
        f"Weight decay: {WEIGHT_DECAY}",
        "Loss: BCEWithLogitsLoss(mean), no class weighting",
        f"Gradient clip: {GRADIENT_CLIP_NORM} once per logical batch",
        f"Epoch cap/min/patience: {MAX_EPOCHS}/{MIN_EPOCH_BEFORE_STOPPING}/{EARLY_STOPPING_PATIENCE}",
        "Checkpoint: strict minimum validation_mean_directional_bce; exact tie retains earlier epoch",
        f"Stage-1 threshold: {SCREENING_THRESHOLD:.2f}",
        "Final partial logical batch: retained; exact sample-weighted accumulation",
        "Validation performance inspected by Step 25: False",
        "Stage-1 fitting started: False",
        f"Step-10 final-test seal: {upstream['test_seal_status']}",
        f"Global final-test state: {upstream['step00']['lock_state']}",
        "",
        "Step-24 physical microbatch schedule:",
    ]

    for r in contract.itertuples(index=False):
        summary_lines.append(
            f"{r.model_id:<14} {r.config_id:<20} "
            f"mb={r.physical_microbatch_size:<3} "
            f"accum={r.full_logical_accumulation_steps:<3} "
            f"effective={r.effective_batch_size}"
        )

    atomic_text(SUMMARY_TXT, "\n".join(summary_lines) + "\n")

    output_files = [
        CONTRACT_CSV,
        BATCH_PLAN_CSV,
        EQUIVALENCE_CSV,
        SHUFFLE_CSV,
        EARLY_STOP_CSV,
        SOURCE_HASH_CSV,
        FREEZE_JSON,
        FROZEN_SCRIPT,
        SUMMARY_TXT,
    ]

    artifacts = {
        p.name: {
            "path": str(p),
            "size_bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
        }
        for p in output_files
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "step": {
            "step_number": STEP_NUMBER,
            "step_name": STEP_NAME,
            "status": "PASS",
        },
        "frozen_at_utc": utc_now(),
        "configuration_fingerprint_sha256": upstream["step00"]["fingerprint"],
        "step16_manifest_sha256": upstream["step16_manifest_sha256"],
        "step24_manifest_sha256": upstream["step24_manifest_sha256"],
        "step24_training_runtime_preflight_fingerprint_sha256": upstream["step24_fingerprint"],
        "stage1_training_engine_fingerprint_sha256": fingerprint,
        "counts": {
            "candidate_contract_rows": len(contract),
            "candidate_fold_batch_plan_rows": len(batch_plan),
            "synthetic_equivalence_rows": len(equivalence),
            "shuffle_audit_rows": len(shuffle),
            "early_stopping_audit_rows": len(early_stop),
            "stage1_fits_planned": EXPECTED_STAGE1_FITS,
        },
        "pipeline_state": freeze["pipeline_state"],
        "final_test_protection": freeze["final_test_protection"],
        "artifacts": artifacts,
    }

    atomic_json(MANIFEST_JSON, manifest)
    manifest_sha = sha256_file(MANIFEST_JSON)
    atomic_text(MANIFEST_SHA, f"{manifest_sha}  {MANIFEST_JSON.name}\n")

    return fingerprint, manifest_sha


# =============================================================================
# 10. VERIFY EXISTING STEP 25
# =============================================================================

def verify_existing() -> int:
    print("=" * 140)
    print("STEP 25 — VERIFY EXISTING STAGE-1 TRAINING-ENGINE CONTRACT")
    print("=" * 140)

    upstream = verify_upstream()
    manifest_sha = verify_sidecar(MANIFEST_JSON, MANIFEST_SHA, "Step 25")
    manifest = load_json(MANIFEST_JSON)
    freeze = load_json(FREEZE_JSON)

    if manifest.get("step", {}).get("status") != "PASS":
        print("[FAIL] Stored Step-25 status is not PASS.")
        return 20

    if manifest.get("configuration_fingerprint_sha256") != upstream["step00"]["fingerprint"]:
        print("[FAIL] Step-25 configuration fingerprint mismatch.")
        return 21

    if manifest.get("step24_manifest_sha256") != upstream["step24_manifest_sha256"]:
        print("[FAIL] Step-24 manifest chain mismatch.")
        return 22

    if (
        manifest.get("step24_training_runtime_preflight_fingerprint_sha256")
        != upstream["step24_fingerprint"]
    ):
        print("[FAIL] Step-24 preflight fingerprint chain mismatch.")
        return 23

    for record in manifest.get("artifacts", {}).values():
        path = Path(record["path"])
        if not path.exists() or sha256_file(path) != record["sha256"]:
            print(f"[FAIL] Step-25 artifact drift: {path}")
            return 24

    stored = freeze["stage1_training_engine_fingerprint_sha256"]
    recomputed = recompute_freeze_fingerprint(
        freeze,
        "stage1_training_engine_fingerprint_sha256",
        remove_generated_at=True,
    )
    if stored != recomputed:
        print("[FAIL] Step-25 semantic fingerprint no longer recomputes.")
        return 25

    if manifest.get("stage1_training_engine_fingerprint_sha256") != stored:
        print("[FAIL] Step-25 manifest/freeze fingerprint mismatch.")
        return 26

    contract = pd.read_csv(CONTRACT_CSV)
    plan = pd.read_csv(BATCH_PLAN_CSV)
    equivalence = pd.read_csv(EQUIVALENCE_CSV)

    if len(contract) != EXPECTED_CANDIDATES:
        print("[FAIL] Stored candidate contract count is not 16.")
        return 27
    if len(plan) != EXPECTED_STAGE1_FITS:
        print("[FAIL] Stored candidate/fold plan count is not 80.")
        return 28
    if not equivalence["passed"].astype(bool).all():
        print("[FAIL] Stored equivalence matrix contains a failure.")
        return 29

    if sha256_file(Path(upstream["step00"]["lock_path"])) != upstream["step00"]["lock_sha256"]:
        print("[FAIL] Global final-test lock changed.")
        return 30

    print(f"[PASS] Configuration fingerprint : {upstream['step00']['fingerprint']}")
    print(f"[PASS] Step-24 manifest          : {upstream['step24_manifest_sha256']}")
    print(f"[PASS] Step-24 preflight freeze  : {upstream['step24_fingerprint']}")
    print(f"[PASS] Candidate contracts       : {len(contract)}")
    print(f"[PASS] Candidate/fold plans      : {len(plan)}")
    print(f"[PASS] Planned Stage-1 fits      : {EXPECTED_STAGE1_FITS}")
    print(f"[PASS] Equivalence records       : {len(equivalence)}")
    print(f"[PASS] Step-25 engine freeze     : {stored}")
    print(f"[PASS] Step-25 manifest          : {manifest_sha}")
    print(f"[PASS] Step-10 test seal         : {upstream['test_seal_status']}")
    print(f"[PASS] Global final-test state   : {upstream['step00']['lock_state']}")
    print("=" * 140)
    print("STEP 25 VERIFICATION STATUS: PASS")
    print("=" * 140)
    return 0


# =============================================================================
# 11. MASTER
# =============================================================================

def run_master() -> int:
    stage = "verify"

    try:
        print("=" * 140)
        print(STEP_NAME)
        print("=" * 140)

        print("[1/9] Verifying frozen Step 00, Step 16 and Step 24 contracts...")
        upstream = verify_upstream()
        print(f"        Configuration fingerprint : {upstream['step00']['fingerprint']}")
        print(f"        Step-16 registry freeze    : {upstream['step16_fingerprint']}")
        print(f"        Step-24 manifest           : {upstream['step24_manifest_sha256']}")
        print(f"        Step-24 preflight freeze   : {upstream['step24_fingerprint']}")
        print(f"        Step-10 final-test seal    : {upstream['test_seal_status']}")
        print(f"        Global final-test state    : {upstream['step00']['lock_state']}")

        stage = "collision_guard"
        print("[2/9] Checking immutable Step-25 output target...")
        assert_fresh_target()
        cfg.create_directories()

        stage = "split_summary"
        print("[3/9] Reading only the frozen Step-24 fold/split sample-count summary...")
        split_record = artifact_by_name(
            upstream["step24_manifest"],
            "24_development_split_summary.csv",
        )
        split_path = check_artifact_record(split_record)
        split = load_step24_split_summary(upstream["step24_manifest"])

        for fold in range(1, EXPECTED_FOLDS + 1):
            tr = int(
                split[(split["fold"] == fold) & (split["partition"] == "train")]
                .iloc[0]["n_sequences"]
            )
            va = int(
                split[(split["fold"] == fold) & (split["partition"] == "validation")]
                .iloc[0]["n_sequences"]
            )
            print(f"        Fold {fold}: train={tr:,} validation={va:,}")

        print("        Development numerical arrays opened: False")
        print("        Final-test data opened           : False")

        stage = "batch_plan"
        print("[4/9] Building all 16 x 5 = 80 frozen logical/microbatch plans...")
        batch_plan = build_fold_batch_plan(upstream["schedule"], split)
        write_csv(BATCH_PLAN_CSV, batch_plan)

        partial_counts = (
            batch_plan[["fold", "final_logical_batch_size"]]
            .drop_duplicates()
            .sort_values("fold")
        )
        for r in partial_counts.itertuples(index=False):
            print(
                f"        Fold {r.fold}: final logical train batch N="
                f"{r.final_logical_batch_size}"
            )
        print(f"        Candidate/fold plans: {len(batch_plan)}")

        stage = "shuffle"
        print("[5/9] Auditing deterministic, candidate-independent epoch shuffling...")
        shuffle = audit_shuffle(split)
        write_csv(SHUFFLE_CSV, shuffle)
        print(f"        Permutation audit records : {len(shuffle)}")
        print("        Candidate ID in shuffle seed: False")

        stage = "equivalence"
        print("[6/9] Testing sample-weighted accumulation-engine arithmetic...")
        unique_mbs = sorted(
            {int(r["physical_microbatch_size"]) for r in upstream["schedule"]},
            reverse=True,
        )
        print(f"        Unique frozen microbatches: {unique_mbs}")
        print(f"        Complete synthetic N       : {FULL_TEST_N}")
        print(f"        Partial synthetic N        : {PARTIAL_TEST_N}")

        equivalence = run_equivalence_suite(upstream["schedule"])
        write_csv(EQUIVALENCE_CSV, equivalence)

        cpu_count = int(
            ((equivalence["device"] == "cpu") & (equivalence["dtype"] == "float64")).sum()
        )
        cuda_count = int(
            ((equivalence["device"] == "cuda:0") & (equivalence["dtype"] == "float32")).sum()
        )
        print(f"        CPU float64 equivalence    : {cpu_count}/{cpu_count} PASS")
        print(f"        CUDA FP32 equivalence      : {cuda_count}/{cuda_count} PASS")
        print(
            "        Max CUDA loss diff         : "
            f"{equivalence[equivalence['device']=='cuda:0']['loss_abs_diff'].max():.3e}"
        )
        print(
            "        Max CUDA gradient diff     : "
            f"{equivalence[equivalence['device']=='cuda:0']['preclip_gradient_max_abs_diff'].max():.3e}"
        )
        print(
            "        Max CUDA parameter diff    : "
            f"{equivalence[equivalence['device']=='cuda:0']['post_adamw_parameter_max_abs_diff'].max():.3e}"
        )

        stage = "early_stopping"
        print("[7/9] Auditing early-stopping and checkpoint state-machine semantics...")
        early_stop = audit_early_stopping()
        write_csv(EARLY_STOP_CSV, early_stop)
        print("        Strict improvement          : validation_bce < best_validation_bce")
        print("        Tie policy                  : retain earlier checkpoint")
        print("        Minimum stopping epoch      : 10")
        print("        Patience                    : 12")
        print("        Case-A expected stop epoch  : 13")
        print("        Case-B expected stop epoch  : 20")

        stage = "contract"
        print("[8/9] Writing the 16-candidate Stage-1 engine contract...")
        contract = build_contract_frame(upstream, batch_plan)
        write_csv(CONTRACT_CSV, contract)
        print(f"        Candidate contracts         : {len(contract)}")
        print(f"        Planned Stage-1 fits         : {EXPECTED_STAGE1_FITS}")
        print("        Scheduler                    : none")
        print("        Drop final logical batch     : False")
        print("        Stage-1 threshold            : 0.50")

        stage = "freeze"
        print("[9/9] Freezing the Stage-1 training engine...")
        fingerprint, manifest_sha = freeze_outputs(
            upstream=upstream,
            split_path=split_path,
            contract=contract,
            batch_plan=batch_plan,
            equivalence=equivalence,
            shuffle=shuffle,
            early_stop=early_stop,
        )

        print()
        print("-" * 140)
        print("STEP-25 FROZEN TRAINING-ENGINE CONTRACT")
        print("-" * 140)
        print(f"Candidate configurations       : {EXPECTED_CANDIDATES}")
        print(f"Chronological folds            : {EXPECTED_FOLDS}")
        print(f"Planned Stage-1 fits           : {EXPECTED_STAGE1_FITS}")
        print(f"Root seed                      : {SEED}")
        print(f"Logical/effective batch        : {LOGICAL_BATCH_SIZE}")
        print("Physical microbatches          : frozen Step-24 schedule")
        print("Partial-batch loss             : mean_BCE_i * (micro_n / logical_n)")
        print("Gradient clipping              : once after full logical-batch accumulation")
        print("Optimizer steps                : exactly one per logical batch")
        print("Validation shuffle             : False")
        print("Validation BCE aggregation     : exact sample-weighted mean")
        print(
            f"Epoch cap/min/patience          : "
            f"{MAX_EPOCHS}/{MIN_EPOCH_BEFORE_STOPPING}/{EARLY_STOPPING_PATIENCE}"
        )
        print("Checkpoint improvement         : strict lower validation_mean_directional_bce")
        print("Equal validation BCE           : retain earlier checkpoint")
        print("Best checkpoint restored       : before Stage-1 metrics")
        print(f"Screening threshold            : {SCREENING_THRESHOLD:.2f}")
        print("Scheduler                      : none")
        print("Best-seed selection            : False")
        print("-" * 140)
        print("PROTECTION STATE")
        print("-" * 140)
        print("Third-party forecasting fit    : False")
        print("Development numerical data     : NOT OPENED")
        print("Validation performance         : NOT INSPECTED")
        print("Stage-1 fitting started        : False")
        print("Threshold optimization         : False")
        print(f"Step-10 final-test seal        : {upstream['test_seal_status']}")
        print(f"Global final-test state        : {upstream['step00']['lock_state']}")
        print("-" * 140)
        print(f"Step-25 engine fingerprint     : {fingerprint}")
        print(f"Step-25 manifest SHA-256       : {manifest_sha}")
        print(f"Freeze                         : {FREEZE_JSON}")
        print(f"Manifest                       : {MANIFEST_JSON}")
        print("=" * 140)
        print(
            "PASS: Full and incomplete logical-batch accumulation arithmetic was "
            "verified on a deterministic reference network in CPU float64 and CUDA FP32."
        )
        print(
            "PASS: The 80-fit Stage-1 optimization, shuffling, checkpoint and "
            "early-stopping contract is frozen without inspecting validation performance."
        )
        print(
            "PASS: The final test remains sealed and locked."
        )
        print("=" * 140)
        print(
            "Do not launch the 80 Stage-1 fits yet. Paste the complete Step-25 "
            "terminal output for audit, then run --verify-only if instructed."
        )

        return 0

    except Exception as exc:
        archived = archive_partial()
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        failure_path = FAILED_DIR / (
            "25_stage1_training_engine_FAILED_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
        atomic_json(
            failure_path,
            {
                "status": "FAIL",
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "failed_at_utc": utc_now(),
                "archived_partial_artifacts": None if archived is None else str(archived),
                "stage1_fitting_started": False,
                "validation_performance_inspected": False,
                "final_test_data_opened": False,
            },
        )

        print()
        print("=" * 140)
        print("STEP 25 FAILED — TRAINING ENGINE NOT FROZEN")
        print("=" * 140)
        print(f"Stage      : {stage}")
        print(f"Error type : {type(exc).__name__}")
        print(f"Error      : {exc}")
        print(f"Diagnostic : {failure_path}")
        if archived is not None:
            print(f"Archived partial Step-25 outputs: {archived}")
        print("=" * 140)
        print("Do not proceed to Stage-1 model fitting.")
        return 30


# =============================================================================
# 12. CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing frozen Step-25 contract without rerunning synthetic tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verify_only:
        return verify_existing()
    return run_master()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStep 25 interrupted by user.")
        raise SystemExit(130)
