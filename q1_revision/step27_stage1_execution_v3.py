#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 27 — CLEAN V3 RESUMABLE STAGE-1 DEVELOPMENT EXECUTION

Purpose
-------
Execute the 80 real development-only Stage-1 fits after Steps 00-26 have been
verified and frozen.

This script DOES NOT select A/B winners.  It only produces and freezes the
80 fit-level development results.  Configuration selection remains a separate
downstream development-only step.

Frozen scientific/execution contract
------------------------------------
  * 16 candidate configurations x 5 chronological folds = 80 fits
  * Stage-1 seed = 42 only
  * joint 1d/3d/5d directional outputs
  * logical/effective batch = 256
  * candidate-specific physical microbatch = frozen Step-24 schedule
  * candidate-independent epoch permutation =
        np.random.default_rng(np.random.SeedSequence([42, fold, epoch]))
  * FP32, AMP=False, TF32=False
  * AdamW, candidate-specific frozen LR, weight_decay=1e-4
  * BCEWithLogitsLoss(reduction="mean"), no class weighting
  * exact sample-weighted accumulation:
        micro_mean_BCE * (micro_n / actual_logical_n)
  * gradient clipping global L2 norm 1.0 exactly once per logical batch
  * max_epochs=100, minimum stopping epoch=10, patience=12
  * strict checkpoint improvement on validation_mean_directional_bce
  * exact ties retain the earlier checkpoint
  * validation sequential/no-shuffle/no-grad
  * best checkpoint restored before Stage-1 metrics/predictions
  * prediction probability = sigmoid(logit)
  * threshold = probability >= 0.50
  * metrics = BA, ROC-AUC, MCC, PR-AUC, Brier per horizon and unweighted macro
  * no threshold tuning, no best-seed selection, no configuration selection
  * final-test access is prohibited

Critical implementation rule
----------------------------
The actual reusable training/model/checkpoint/metric helpers are imported from
the frozen Step-26 runner.  Step 27 does not reimplement that engine.

Resumability
------------
Each model/config/fold runs in a fresh subprocess. A fit is considered complete
only when fit_manifest.json exists AND every canonical artifact registered in
that manifest passes hash verification.  fit_manifest.json is written last by
the worker.  Invalid/incomplete fit directories are archived and rerun from
scratch.

Recommended staged execution
----------------------------
1) Freeze/verify the Step-27 source and development input register only:
       python step27_stage1_execution_v3.py --prepare-only

2) After auditing PREPARE PASS, run exactly one real fit:
       python step27_stage1_execution_v3.py --max-new-fits 1

3) After auditing that fit, resume the same immutable script:
       python step27_stage1_execution_v3.py

4) After all 80 fits are frozen:
       python step27_stage1_execution_v3.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import step00_config as cfg
import step26_stage1_runner_dryrun_v3 as s26


# =============================================================================
# 1. FROZEN IDENTIFIERS / CONSTANTS
# =============================================================================

STEP_NUMBER = 27
STEP_NAME = "STEP 27 — RESUMABLE STAGE-1 DEVELOPMENT EXECUTION"
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
EXPECTED_STEP26_FINGERPRINT = (
    "f9198f5799650beedeb42400edc79a9f572d66f9cfe44ade4aae65e059248d22"
)
EXPECTED_STEP26_MANIFEST_SHA256 = (
    "dc2b3240ca6bd05cf65ce24da2bd877ff2daf7fdf0da0513dce916819340e3fe"
)

MODEL_ORDER = (
    "DLINEAR",
    "NLINEAR",
    "PATCHTST",
    "TIMESNET",
    "ITRANSFORMER",
    "FREDFORMER",
    "FRNET",
    "MLF",
)
FOLDS = (1, 2, 3, 4, 5)

EXPECTED_MODELS = 8
EXPECTED_CANDIDATES = 16
EXPECTED_FOLDS = 5
EXPECTED_FITS = 80

SEED = 42
LOOKBACK = 60
FEATURES = 63
OUTPUTS = 3
HORIZONS = (1, 3, 5)
LOGICAL_BATCH = 256

WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
MAX_EPOCHS = 100
MINIMUM_EPOCH_BEFORE_STOPPING = 10
EARLY_STOPPING_PATIENCE = 12
SCREENING_THRESHOLD = 0.50

RESULT_PREFIX = "__STEP27_RESULT__"

SCRIPT = Path(__file__).resolve()
Q1_DIR = SCRIPT.parent

STEP26_SCRIPT = Q1_DIR / "step26_stage1_runner_dryrun_v3.py"
STEP26_FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step26_stage1_runner_dryrun_frozen.py"
STEP26_FREEZE = cfg.MANIFEST_DIR / "26_stage1_runner_dryrun_freeze.json"
STEP26_MANIFEST = cfg.MANIFEST_DIR / "26_stage1_runner_dryrun_manifest.json"
STEP26_MANIFEST_SHA = cfg.MANIFEST_DIR / "26_stage1_runner_dryrun_manifest.sha256"

STEP25_BATCH_PLAN = cfg.AUDIT_DIR / "25_stage1_fold_batch_plan.csv"
STEP10_TEST_SEAL = cfg.MANIFEST_DIR / "10_final_test_seal.json"
FORBIDDEN_TEST_FILENAME = "10_final_test_locked_panel.parquet"

RUN_ROOT = cfg.EXPERIMENT_ROOT_DIR / "stage1_fits"

SOURCE_FREEZE_JSON = cfg.MANIFEST_DIR / "27_stage1_execution_source_freeze.json"
SOURCE_FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step27_stage1_execution_frozen.py"

INPUT_REGISTER_JSON = cfg.MANIFEST_DIR / "27_stage1_development_input_register.json"

FIT_REGISTER_CSV = cfg.AUDIT_DIR / "27_stage1_fit_register.csv"
COMPLETION_HASH_CSV = cfg.AUDIT_DIR / "27_stage1_completion_hash_register.csv"
OUTPUT_HASH_CSV = cfg.AUDIT_DIR / "27_stage1_execution_output_hashes.csv"

FREEZE_JSON = cfg.MANIFEST_DIR / "27_stage1_execution_freeze.json"
MANIFEST_JSON = cfg.MANIFEST_DIR / "27_stage1_execution_manifest.json"
MANIFEST_SHA = cfg.MANIFEST_DIR / "27_stage1_execution_manifest.sha256"
SUMMARY_TXT = cfg.REPORT_DIR / "27_stage1_execution_summary.txt"

FAILED_DIR = cfg.AUDIT_DIR / "step27_failed_attempts"
INCOMPLETE_ARCHIVE_DIR = cfg.AUDIT_DIR / "step27_incomplete_fit_archives"


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


def load_json(path: Path) -> dict:
    if not path.is_file():
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


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)


def canonical_hash(payload: Any) -> str:
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


def recompute_freeze_fingerprint(payload: dict, field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    body.pop("generated_at_utc", None)
    return canonical_hash(body)


def ensure_not_forbidden(path: Path) -> None:
    if path.name == FORBIDDEN_TEST_FILENAME:
        raise RuntimeError("Final-test parquet entered Step-27 path handling.")


def fit_dir(model_id: str, config_id: str, fold: int) -> Path:
    return (
        RUN_ROOT
        / model_id
        / config_id
        / f"fold_{int(fold):02d}"
    )


def fit_manifest_path(model_id: str, config_id: str, fold: int) -> Path:
    return fit_dir(model_id, config_id, fold) / "fit_manifest.json"


def worker_log_path(model_id: str, config_id: str, fold: int) -> Path:
    return fit_dir(model_id, config_id, fold) / "orchestration_worker.log"


# =============================================================================
# 3. VERIFY FROZEN STEP 26 + FULL UPSTREAM CHAIN
# =============================================================================

def verify_upstream() -> dict:
    upstream = s26.verify_upstream()

    if upstream["upstream25"]["step00"]["fingerprint"] != EXPECTED_CONFIGURATION_FINGERPRINT:
        raise RuntimeError("Step-00 configuration fingerprint drift.")

    if upstream["step24_manifest_sha256"] != EXPECTED_STEP24_MANIFEST_SHA256:
        raise RuntimeError("Step-24 manifest SHA drift.")
    if upstream["step25_manifest_sha256"] != EXPECTED_STEP25_MANIFEST_SHA256:
        raise RuntimeError("Step-25 manifest SHA drift.")

    if upstream["step24"]["training_runtime_preflight_fingerprint_sha256"] != EXPECTED_STEP24_FINGERPRINT:
        raise RuntimeError("Step-24 preflight fingerprint drift.")
    if upstream["step25"]["stage1_training_engine_fingerprint_sha256"] != EXPECTED_STEP25_FINGERPRINT:
        raise RuntimeError("Step-25 engine fingerprint drift.")

    step26_manifest_sha = verify_sidecar(
        STEP26_MANIFEST,
        STEP26_MANIFEST_SHA,
        "Step 26",
    )
    if step26_manifest_sha != EXPECTED_STEP26_MANIFEST_SHA256:
        raise RuntimeError(
            "Unexpected Step-26 manifest SHA.\n"
            f"Expected: {EXPECTED_STEP26_MANIFEST_SHA256}\n"
            f"Observed: {step26_manifest_sha}"
        )

    step26_manifest = load_json(STEP26_MANIFEST)
    step26_freeze = load_json(STEP26_FREEZE)

    if step26_manifest.get("step", {}).get("status") != "PASS":
        raise RuntimeError("Step-26 manifest status is not PASS.")

    stored26 = step26_freeze["stage1_runner_dryrun_fingerprint_sha256"]
    recomputed26 = recompute_freeze_fingerprint(
        step26_freeze,
        "stage1_runner_dryrun_fingerprint_sha256",
    )

    if stored26 != EXPECTED_STEP26_FINGERPRINT or recomputed26 != stored26:
        raise RuntimeError("Step-26 runner fingerprint drift.")

    if (
        step26_manifest.get("stage1_runner_dryrun_fingerprint_sha256")
        != EXPECTED_STEP26_FINGERPRINT
    ):
        raise RuntimeError("Step-26 manifest/freeze fingerprint mismatch.")

    if not STEP26_SCRIPT.is_file() or not STEP26_FROZEN_SCRIPT.is_file():
        raise FileNotFoundError("Current/frozen Step-26 source missing.")

    step26_working_sha = sha256_file(STEP26_SCRIPT)
    step26_frozen_sha = sha256_file(STEP26_FROZEN_SCRIPT)

    if step26_working_sha != step26_frozen_sha:
        raise RuntimeError(
            "Current Step-26 source differs from frozen source.\n"
            f"Working: {step26_working_sha}\n"
            f"Frozen : {step26_frozen_sha}"
        )

    if (
        step26_freeze["final_test_protection"]["step10_seal_status"]
        != "SEALED_NOT_OPENED"
        or step26_freeze["final_test_protection"][
            "final_test_parquet_opened_or_hashed_by_step26"
        ]
        or step26_freeze["final_test_protection"]["global_final_test_state"]
        != "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError("Step-26 final-test protection drift.")

    seal = load_json(STEP10_TEST_SEAL)
    if (
        seal.get("status") != "SEALED_NOT_OPENED"
        or seal.get("global_final_test_lock_state") != "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError("Step-10 final-test seal is not pristine.")

    return {
        **upstream,
        "step26_freeze": step26_freeze,
        "step26_manifest": step26_manifest,
        "step26_manifest_sha256": step26_manifest_sha,
        "step26_source_sha256": step26_working_sha,
        "test_seal_status": seal["status"],
        "global_final_test_state": "LOCKED_NOT_OPENED",
    }


# =============================================================================
# 4. IMMUTABLE STEP-27 SOURCE FREEZE
# =============================================================================

def build_source_semantic(upstream: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "STAGE1_EXECUTION_SOURCE_FROZEN",
        "configuration_fingerprint_sha256": EXPECTED_CONFIGURATION_FINGERPRINT,
        "step27_execution_source_sha256": sha256_file(SCRIPT),
        "upstream": {
            "step24_fingerprint_sha256": EXPECTED_STEP24_FINGERPRINT,
            "step24_manifest_sha256": EXPECTED_STEP24_MANIFEST_SHA256,
            "step25_fingerprint_sha256": EXPECTED_STEP25_FINGERPRINT,
            "step25_manifest_sha256": EXPECTED_STEP25_MANIFEST_SHA256,
            "step26_fingerprint_sha256": EXPECTED_STEP26_FINGERPRINT,
            "step26_manifest_sha256": EXPECTED_STEP26_MANIFEST_SHA256,
            "step26_source_sha256": upstream["step26_source_sha256"],
        },
        "execution_scope": {
            "models": EXPECTED_MODELS,
            "candidate_configurations": EXPECTED_CANDIDATES,
            "folds": list(FOLDS),
            "total_fits": EXPECTED_FITS,
            "seed": SEED,
            "configuration_selection_inside_step27": False,
            "threshold_optimization_inside_step27": False,
            "best_seed_selection_inside_step27": False,
        },
        "final_test_policy": {
            "allowed": False,
            "test_parquet_opened_or_hashed": False,
        },
    }


def ensure_source_freeze(upstream: dict) -> tuple[str, str]:
    semantic = build_source_semantic(upstream)
    expected_fp = canonical_hash(semantic)
    current_script_sha = semantic["step27_execution_source_sha256"]

    if not SOURCE_FREEZE_JSON.exists():
        if SOURCE_FROZEN_SCRIPT.exists():
            raise RuntimeError(
                "Frozen Step-27 script exists without source-freeze JSON."
            )

        freeze = {
            **semantic,
            "generated_at_utc": utc_now(),
            "stage1_execution_source_fingerprint_sha256": expected_fp,
        }
        atomic_json(SOURCE_FREEZE_JSON, freeze)
        shutil.copy2(SCRIPT, SOURCE_FROZEN_SCRIPT)

    freeze = load_json(SOURCE_FREEZE_JSON)
    stored_fp = freeze["stage1_execution_source_fingerprint_sha256"]
    recomputed = recompute_freeze_fingerprint(
        freeze,
        "stage1_execution_source_fingerprint_sha256",
    )

    if stored_fp != expected_fp or recomputed != stored_fp:
        raise RuntimeError("Step-27 source-freeze fingerprint drift.")

    if freeze["step27_execution_source_sha256"] != current_script_sha:
        raise RuntimeError("Current Step-27 script differs from source freeze.")

    if not SOURCE_FROZEN_SCRIPT.is_file():
        raise FileNotFoundError(SOURCE_FROZEN_SCRIPT)

    frozen_sha = sha256_file(SOURCE_FROZEN_SCRIPT)
    if frozen_sha != current_script_sha:
        raise RuntimeError(
            "Current Step-27 script differs from frozen Step-27 copy."
        )

    return stored_fp, current_script_sha


# =============================================================================
# 5. DEVELOPMENT INPUT REGISTER — NO FINAL-TEST ACCESS
# =============================================================================

def assert_development_filename(
    path: Path,
    fold: int,
    split: str,
    member: str,
) -> None:
    expected = f"14_fold_{fold}_{split}_{member}.npy"
    if path.name != expected:
        raise RuntimeError(
            f"Unexpected development filename.\n"
            f"Expected: {expected}\nObserved: {path.name}"
        )

    lower_name = path.name.lower()
    if any(token in lower_name for token in ("final", "test", "heldout", "holdout")):
        raise RuntimeError(f"Final-like token in development filename: {path}")


def resolve_development_array(
    fold: int,
    split: str,
    member: str,
) -> Path:
    filename = f"14_fold_{fold}_{split}_{member}.npy"

    matches = [
        p
        for p in cfg.EXPERIMENT_ROOT_DIR.rglob(filename)
        if p.is_file()
        and p.name != FORBIDDEN_TEST_FILENAME
        and "third_party_sources" not in {part.lower() for part in p.parts}
        and "audit" not in {part.lower() for part in p.parts}
        and "manifests" not in {part.lower() for part in p.parts}
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve development array {filename}. "
            f"Found {len(matches)}:\n"
            + "\n".join(str(p) for p in matches)
        )

    path = matches[0]
    ensure_not_forbidden(path)
    assert_development_filename(path, fold, split, member)
    return path


def inspect_npy_header(path: Path) -> dict:
    ensure_not_forbidden(path)
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    result = {
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "ndim": int(arr.ndim),
    }
    del arr
    return result


def build_input_semantic(upstream: dict, source_fp: str) -> dict:
    batch_plan = pd.read_csv(STEP25_BATCH_PLAN)

    if len(batch_plan) != EXPECTED_FITS:
        raise RuntimeError(
            f"Expected {EXPECTED_FITS} Step-25 batch-plan rows; "
            f"found {len(batch_plan)}."
        )

    records = []

    for fold in FOLDS:
        expected_train = (
            batch_plan[batch_plan["fold"] == fold]["train_sequences"]
            .astype(int)
            .unique()
        )
        expected_val = (
            batch_plan[batch_plan["fold"] == fold]["validation_sequences"]
            .astype(int)
            .unique()
        )

        if len(expected_train) != 1 or len(expected_val) != 1:
            raise RuntimeError(f"Fold {fold}: Step-25 count contract is ambiguous.")

        expected_train_n = int(expected_train[0])
        expected_val_n = int(expected_val[0])

        for split, expected_n in (
            ("train", expected_train_n),
            ("validation", expected_val_n),
        ):
            x_path = resolve_development_array(
                fold, split, "X_full"
            )
            y_path = resolve_development_array(
                fold, split, "y_direction"
            )

            x_header = inspect_npy_header(x_path)
            y_header = inspect_npy_header(y_path)

            if x_header["ndim"] != 3 or x_header["shape"][1:] != [LOOKBACK, FEATURES]:
                raise RuntimeError(
                    f"Fold {fold} {split} X shape drift: {x_header['shape']}"
                )
            if y_header["ndim"] != 2 or y_header["shape"][1:] != [OUTPUTS]:
                raise RuntimeError(
                    f"Fold {fold} {split} y shape drift: {y_header['shape']}"
                )
            if x_header["shape"][0] != expected_n:
                raise RuntimeError(
                    f"Fold {fold} {split} X row-count drift: "
                    f"{x_header['shape'][0]} != {expected_n}"
                )
            if y_header["shape"][0] != expected_n:
                raise RuntimeError(
                    f"Fold {fold} {split} y row-count drift: "
                    f"{y_header['shape'][0]} != {expected_n}"
                )

            for member, path, header in (
                ("X_full", x_path, x_header),
                ("y_direction", y_path, y_header),
            ):
                records.append(
                    {
                        "fold": int(fold),
                        "split": split,
                        "member": member,
                        "path": str(path.resolve()),
                        "size_bytes": int(path.stat().st_size),
                        "sha256": sha256_file(path),
                        "shape": header["shape"],
                        "dtype": header["dtype"],
                    }
                )

    if len(records) != EXPECTED_FOLDS * 2 * 2:
        raise RuntimeError("Expected 20 development input records.")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "STAGE1_DEVELOPMENT_INPUT_REGISTER_FROZEN",
        "configuration_fingerprint_sha256": EXPECTED_CONFIGURATION_FINGERPRINT,
        "stage1_execution_source_fingerprint_sha256": source_fp,
        "step25_batch_plan": {
            "path": str(STEP25_BATCH_PLAN.resolve()),
            "sha256": sha256_file(STEP25_BATCH_PLAN),
            "rows": int(len(batch_plan)),
        },
        "records": sorted(
            records,
            key=lambda r: (r["fold"], r["split"], r["member"]),
        ),
        "protection_state": {
            "development_npy_headers_read": True,
            "development_artifact_bytes_hashed": True,
            "development_numerical_values_used_for_training": False,
            "development_validation_performance_computed": False,
            "final_test_parquet_opened_or_hashed": False,
            "global_final_test_state": upstream["global_final_test_state"],
        },
    }


def ensure_input_register(
    upstream: dict,
    source_fp: str,
) -> tuple[str, dict]:
    semantic = build_input_semantic(upstream, source_fp)
    expected_fp = canonical_hash(semantic)

    if not INPUT_REGISTER_JSON.exists():
        payload = {
            **semantic,
            "generated_at_utc": utc_now(),
            "stage1_development_input_fingerprint_sha256": expected_fp,
        }
        atomic_json(INPUT_REGISTER_JSON, payload)

    payload = load_json(INPUT_REGISTER_JSON)
    stored_fp = payload["stage1_development_input_fingerprint_sha256"]
    recomputed = recompute_freeze_fingerprint(
        payload,
        "stage1_development_input_fingerprint_sha256",
    )

    if stored_fp != expected_fp or recomputed != stored_fp:
        raise RuntimeError(
            "Development input register no longer matches current frozen files."
        )

    return stored_fp, payload


def input_record_map(register: dict) -> dict[tuple[int, str, str], dict]:
    out = {}
    for record in register["records"]:
        key = (
            int(record["fold"]),
            str(record["split"]),
            str(record["member"]),
        )
        if key in out:
            raise RuntimeError(f"Duplicate input-register key: {key}")
        out[key] = record

    if len(out) != 20:
        raise RuntimeError("Input register key count is not 20.")
    return out


# =============================================================================
# 6. FROZEN CANDIDATE/FOLD EXECUTION MAP
# =============================================================================

def execution_maps(upstream: dict) -> tuple[dict, pd.DataFrame]:
    schedule = upstream["schedule"]
    if len(schedule) != EXPECTED_CANDIDATES:
        raise RuntimeError("Step-24 schedule does not contain 16 candidates.")

    schedule_map = {
        (r["model_id"], r["config_id"]): r
        for r in schedule
    }
    if len(schedule_map) != EXPECTED_CANDIDATES:
        raise RuntimeError("Step-24 schedule keys are not unique.")

    models = {k[0] for k in schedule_map}
    if models != set(MODEL_ORDER):
        raise RuntimeError(
            f"Unexpected model set in Step-24 schedule: {sorted(models)}"
        )

    for model_id in MODEL_ORDER:
        config_ids = sorted(
            config_id
            for mid, config_id in schedule_map
            if mid == model_id
        )
        if len(config_ids) != 2:
            raise RuntimeError(
                f"{model_id}: expected two candidate configurations."
            )

    batch_plan = pd.read_csv(STEP25_BATCH_PLAN)
    if len(batch_plan) != EXPECTED_FITS:
        raise RuntimeError("Step-25 batch plan is not 80 rows.")

    for key, group in batch_plan.groupby(
        ["model_id", "config_id"],
        sort=True,
    ):
        if key not in schedule_map:
            raise RuntimeError(f"Step-25 plan contains unknown candidate: {key}")

        physical_values = (
            group["physical_microbatch_size"].astype(int).unique()
        )
        if len(physical_values) != 1:
            raise RuntimeError(f"{key}: multiple physical microbatch sizes.")

        if int(physical_values[0]) != int(
            schedule_map[key]["physical_microbatch_size"]
        ):
            raise RuntimeError(f"{key}: Step-24/Step-25 physical batch drift.")

        if sorted(group["fold"].astype(int).tolist()) != list(FOLDS):
            raise RuntimeError(f"{key}: incomplete Step-25 fold plan.")

    return schedule_map, batch_plan


# =============================================================================
# 7. FIT MANIFEST VERIFICATION / RESUME
# =============================================================================

def finite_metric_payload(metrics: dict) -> bool:
    try:
        per_h = metrics["per_horizon"]
        macro = metrics["macro_horizon_unweighted"]

        for h in ("1d", "3d", "5d"):
            for metric in (
                "balanced_accuracy",
                "roc_auc",
                "mcc",
                "pr_auc",
                "brier_score",
            ):
                if not math.isfinite(float(per_h[h][metric])):
                    return False

        for metric in (
            "balanced_accuracy",
            "roc_auc",
            "mcc",
            "pr_auc",
            "brier_score",
        ):
            if not math.isfinite(float(macro[metric])):
                return False

        return True
    except Exception:
        return False


def verify_completed_fit(
    model_id: str,
    config_id: str,
    fold: int,
    *,
    source_fp: str,
    input_fp: str,
    schedule_map: dict,
    input_map: dict,
) -> dict | None:
    path = fit_manifest_path(model_id, config_id, fold)
    if not path.is_file():
        return None

    try:
        payload = load_json(path)

        if payload.get("status") != "STAGE1_FIT_COMPLETE":
            return None
        if payload.get("configuration_fingerprint_sha256") != EXPECTED_CONFIGURATION_FINGERPRINT:
            return None
        if payload.get("stage1_execution_source_fingerprint_sha256") != source_fp:
            return None
        if payload.get("stage1_development_input_fingerprint_sha256") != input_fp:
            return None
        if payload.get("step26_runner_fingerprint_sha256") != EXPECTED_STEP26_FINGERPRINT:
            return None

        if payload.get("model_id") != model_id:
            return None
        if payload.get("config_id") != config_id:
            return None
        if int(payload.get("fold")) != int(fold):
            return None
        if int(payload.get("seed")) != SEED:
            return None
        if int(payload.get("logical_batch_size")) != LOGICAL_BATCH:
            return None

        physical_expected = int(
            schedule_map[(model_id, config_id)]["physical_microbatch_size"]
        )
        if int(payload.get("physical_microbatch_size")) != physical_expected:
            return None

        if float(payload.get("threshold")) != SCREENING_THRESHOLD:
            return None

        inputs = payload.get("inputs", {})
        expected_input_keys = {
            "train_x": (fold, "train", "X_full"),
            "train_y": (fold, "train", "y_direction"),
            "validation_x": (fold, "validation", "X_full"),
            "validation_y": (fold, "validation", "y_direction"),
        }

        for manifest_key, register_key in expected_input_keys.items():
            rec = inputs.get(manifest_key)
            source = input_map[register_key]
            if not rec:
                return None
            if Path(rec["path"]).resolve() != Path(source["path"]).resolve():
                return None
            if rec["sha256"] != source["sha256"]:
                return None
            if int(rec["size_bytes"]) != int(source["size_bytes"]):
                return None

        artifacts = payload.get("artifacts", {})
        for key in (
            "best_checkpoint",
            "epoch_history",
            "validation_predictions",
            "fit_metrics",
        ):
            rec = artifacts.get(key)
            if not rec:
                return None
            artifact_path = Path(rec["path"])
            if not artifact_path.is_file():
                return None
            if int(rec["size_bytes"]) != int(artifact_path.stat().st_size):
                return None
            if rec["sha256"] != sha256_file(artifact_path):
                return None

        metrics_path = Path(artifacts["fit_metrics"]["path"])
        metrics = load_json(metrics_path)

        if float(metrics.get("threshold")) != SCREENING_THRESHOLD:
            return None
        if not finite_metric_payload(metrics):
            return None
        if not math.isfinite(
            float(metrics["best_validation_mean_directional_bce"])
        ):
            return None

        protection = payload.get("protection_state", {})
        if (
            not bool(protection.get("development_training_used"))
            or not bool(protection.get("development_validation_used"))
            or bool(protection.get("threshold_optimization_performed"))
            or bool(protection.get("best_seed_selection_performed"))
            or bool(protection.get("configuration_selection_performed"))
            or bool(protection.get("final_test_data_opened"))
            or bool(protection.get("final_test_metric_computed"))
        ):
            return None

        return payload

    except Exception:
        return None


def archive_incomplete_fit(
    model_id: str,
    config_id: str,
    fold: int,
) -> Path | None:
    directory = fit_dir(model_id, config_id, fold)
    if not directory.exists():
        return None

    INCOMPLETE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = (
        INCOMPLETE_ARCHIVE_DIR
        / f"{model_id}__{config_id}__fold_{fold:02d}__{stamp}"
    )

    counter = 1
    while destination.exists():
        destination = (
            INCOMPLETE_ARCHIVE_DIR
            / f"{model_id}__{config_id}__fold_{fold:02d}__{stamp}__{counter}"
        )
        counter += 1

    shutil.move(str(directory), str(destination))
    return destination


# =============================================================================
# 8. ACTUAL REAL DEVELOPMENT FIT WORKER
# =============================================================================

def worker_fit(
    model_id: str,
    config_id: str,
    fold: int,
) -> dict:
    import torch
    import torch.nn as nn

    wall_start = time.perf_counter()

    upstream = verify_upstream()
    source_fp, source_sha = ensure_source_freeze(upstream)
    input_fp, input_register = ensure_input_register(
        upstream,
        source_fp,
    )
    input_map = input_record_map(input_register)
    schedule_map, batch_plan = execution_maps(upstream)

    if (model_id, config_id) not in schedule_map:
        raise RuntimeError(
            f"Unknown frozen candidate: {model_id}/{config_id}"
        )
    if fold not in FOLDS:
        raise RuntimeError(f"Unexpected fold: {fold}")

    schedule = schedule_map[(model_id, config_id)]
    physical_microbatch = int(schedule["physical_microbatch_size"])

    fold_plan = batch_plan[
        (batch_plan["model_id"] == model_id)
        & (batch_plan["config_id"] == config_id)
        & (batch_plan["fold"] == fold)
    ]
    if len(fold_plan) != 1:
        raise RuntimeError("Candidate/fold Step-25 plan lookup failed.")

    expected_train_n = int(fold_plan.iloc[0]["train_sequences"])
    expected_val_n = int(fold_plan.iloc[0]["validation_sequences"])

    train_x_rec = input_map[(fold, "train", "X_full")]
    train_y_rec = input_map[(fold, "train", "y_direction")]
    val_x_rec = input_map[(fold, "validation", "X_full")]
    val_y_rec = input_map[(fold, "validation", "y_direction")]

    x_train = np.load(
        train_x_rec["path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    y_train = np.load(
        train_y_rec["path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    x_val = np.load(
        val_x_rec["path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    y_val = np.load(
        val_y_rec["path"],
        mmap_mode="r",
        allow_pickle=False,
    )

    if len(x_train) != expected_train_n or len(y_train) != expected_train_n:
        raise RuntimeError("Training row count drift.")
    if len(x_val) != expected_val_n or len(y_val) != expected_val_n:
        raise RuntimeError("Validation row count drift.")

    train_unique = np.unique(np.asarray(y_train))
    val_unique = np.unique(np.asarray(y_val))

    if not set(train_unique.tolist()).issubset({0, 1}):
        raise RuntimeError(
            f"Non-binary training labels: {train_unique.tolist()}"
        )
    if not set(val_unique.tolist()).issubset({0, 1}):
        raise RuntimeError(
            f"Non-binary validation labels: {val_unique.tolist()}"
        )

    for j, horizon in enumerate(HORIZONS):
        val_classes = np.unique(
            np.asarray(y_val[:, j], dtype=np.int64)
        )
        if not np.array_equal(
            val_classes,
            np.asarray([0, 1], dtype=np.int64),
        ):
            raise RuntimeError(
                f"Fold {fold} horizon {horizon}d validation labels "
                f"do not contain both classes: {val_classes.tolist()}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    s26.set_training_determinism(SEED)

    device = torch.device("cuda:0")
    device_index = 0

    model, build_metadata, repository = s26.build_candidate_model(
        model_id=model_id,
        config_id=config_id,
        physical_microbatch=physical_microbatch,
        device=device,
    )

    learning_rate = float(build_metadata["learning_rate"])

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=s26.BETAS,
        eps=s26.ADAMW_EPS,
        weight_decay=WEIGHT_DECAY,
    )

    trainable_parameters = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    total_parameters = int(
        sum(p.numel() for p in model.parameters())
    )

    directory = fit_dir(model_id, config_id, fold)
    directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = directory / "best_checkpoint.pt"
    history_path = directory / "epoch_history.csv"
    predictions_path = directory / "validation_predictions.npz"
    metrics_path = directory / "fit_metrics.json"
    manifest_path = directory / "fit_manifest.json"

    # Worker starts from a clean canonical fit directory. The master archives
    # any prior incomplete/invalid directory before launching this worker.
    for path in (
        checkpoint_path,
        history_path,
        predictions_path,
        metrics_path,
        manifest_path,
    ):
        if path.exists():
            path.unlink()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    best_val_bce = math.inf
    best_epoch = None
    stale_epochs = 0
    stopped_early = False

    epoch_rows = []
    total_training_seconds = 0.0
    total_validation_seconds = 0.0

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()

        permutation = s26.epoch_permutation(
            len(x_train),
            fold=fold,
            epoch=epoch,
        )

        sample_weighted_train_bce_sum = 0.0
        train_samples_seen = 0
        grad_norms = []
        logical_updates = 0

        for logical_start in range(
            0,
            len(permutation),
            LOGICAL_BATCH,
        ):
            logical_indices = permutation[
                logical_start:
                logical_start + LOGICAL_BATCH
            ]

            update = s26.train_logical_batch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                x_source=x_train,
                y_source=y_train,
                logical_indices=logical_indices,
                physical_microbatch=physical_microbatch,
                device=device,
            )

            logical_n = int(update["logical_batch_n"])

            sample_weighted_train_bce_sum += (
                float(update["weighted_mean_bce"]) * logical_n
            )
            train_samples_seen += logical_n
            grad_norms.append(
                float(update["preclip_gradient_norm"])
            )
            logical_updates += 1

        torch.cuda.synchronize(device_index)
        train_seconds = time.perf_counter() - train_started
        total_training_seconds += train_seconds

        if train_samples_seen != len(x_train):
            raise RuntimeError(
                f"Epoch {epoch}: training coverage mismatch "
                f"{train_samples_seen} != {len(x_train)}"
            )
        if logical_updates != math.ceil(len(x_train) / LOGICAL_BATCH):
            raise RuntimeError("Logical-update count drift.")

        train_bce = (
            sample_weighted_train_bce_sum / train_samples_seen
        )

        validation_started = time.perf_counter()
        val_result = s26.evaluate_directional(
            model=model,
            criterion=criterion,
            x_source=x_val,
            y_source=y_val,
            physical_microbatch=physical_microbatch,
            device=device,
        )
        torch.cuda.synchronize(device_index)
        validation_seconds = (
            time.perf_counter() - validation_started
        )
        total_validation_seconds += validation_seconds

        val_bce = float(
            val_result["validation_mean_directional_bce"]
        )

        if not math.isfinite(train_bce):
            raise FloatingPointError("Non-finite epoch training BCE.")
        if not math.isfinite(val_bce):
            raise FloatingPointError("Non-finite validation BCE.")

        improved = val_bce < best_val_bce

        if improved:
            best_val_bce = val_bce
            best_epoch = int(epoch)
            stale_epochs = 0

            checkpoint = s26.checkpoint_payload(
                model,
                epoch=epoch,
                validation_mean_directional_bce=val_bce,
                model_id=model_id,
                config_id=config_id,
                fold=fold,
            )
            s26.save_checkpoint_atomic(
                checkpoint_path,
                checkpoint,
            )
        else:
            stale_epochs += 1

        stop_after_epoch = bool(
            epoch >= MINIMUM_EPOCH_BEFORE_STOPPING
            and stale_epochs >= EARLY_STOPPING_PATIENCE
        )

        epoch_seconds = time.perf_counter() - epoch_started

        epoch_rows.append(
            {
                "epoch": int(epoch),
                "train_mean_directional_bce": float(train_bce),
                "validation_mean_directional_bce": float(val_bce),
                "learning_rate": learning_rate,
                "mean_preclip_gradient_norm": float(
                    np.mean(grad_norms)
                ),
                "epoch_seconds": float(epoch_seconds),
                "training_seconds": float(train_seconds),
                "validation_seconds": float(validation_seconds),
                "logical_optimizer_updates": int(logical_updates),
                "strict_checkpoint_improvement": bool(improved),
                "consecutive_nonimproving_epochs": int(stale_epochs),
                "stop_after_epoch": bool(stop_after_epoch),
            }
        )

        print(
            f"EPOCH {model_id}/{config_id}/fold{fold} "
            f"{epoch:03d} "
            f"trainBCE={train_bce:.6f} "
            f"valBCE={val_bce:.6f} "
            f"best={best_val_bce:.6f}@{best_epoch} "
            f"stale={stale_epochs:02d} "
            f"sec={epoch_seconds:.1f}",
            flush=True,
        )

        del val_result

        if stop_after_epoch:
            stopped_early = True
            break

    if best_epoch is None or not checkpoint_path.is_file():
        raise RuntimeError("No best validation checkpoint was produced.")

    # Restore the strict minimum-validation-BCE checkpoint BEFORE Stage-1
    # predictions/metrics, exactly as frozen in Steps 25-26.
    checkpoint_loaded = s26.load_checkpoint_into_model(
        checkpoint_path,
        model,
    )

    if int(checkpoint_loaded["epoch"]) != int(best_epoch):
        raise RuntimeError("Restored checkpoint epoch mismatch.")
    if not math.isclose(
        float(checkpoint_loaded["validation_mean_directional_bce"]),
        float(best_val_bce),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("Restored checkpoint BCE mismatch.")

    best_eval = s26.evaluate_directional(
        model=model,
        criterion=criterion,
        x_source=x_val,
        y_source=y_val,
        physical_microbatch=physical_microbatch,
        device=device,
    )

    restored_bce = float(
        best_eval["validation_mean_directional_bce"]
    )
    if not math.isclose(
        restored_bce,
        best_val_bce,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError(
            "Best-checkpoint validation BCE did not reproduce after restore.\n"
            f"Stored: {best_val_bce}\nRestored: {restored_bce}"
        )

    y_val_int = np.asarray(
        y_val,
        dtype=np.int64,
    )

    stage1_metrics = s26.compute_stage1_metrics(
        y_true=y_val_int,
        probabilities=best_eval["probabilities"],
        threshold=SCREENING_THRESHOLD,
    )

    # Required canonical prediction artifact. Labels are NOT duplicated.
    pred_tmp = predictions_path.with_name(
        predictions_path.name + ".tmp.npz"
    )
    np.savez_compressed(
        pred_tmp,
        logits=best_eval["logits"].astype(
            np.float32, copy=False
        ),
        probabilities=best_eval["probabilities"].astype(
            np.float32, copy=False
        ),
        row_index=best_eval["row_index"].astype(
            np.int64, copy=False
        ),
    )
    os.replace(pred_tmp, predictions_path)

    history_frame = pd.DataFrame(epoch_rows)
    s26.write_csv(history_path, history_frame)

    peak_allocated_bytes = int(
        torch.cuda.max_memory_allocated(device_index)
    )
    peak_reserved_bytes = int(
        torch.cuda.max_memory_reserved(device_index)
    )

    wall_seconds = float(time.perf_counter() - wall_start)

    metrics_payload = {
        "schema_version": 1,
        "model_id": model_id,
        "config_id": config_id,
        "fold": int(fold),
        "seed": SEED,
        "threshold": SCREENING_THRESHOLD,
        "threshold_operator": "probability >= 0.50",
        "best_epoch": int(best_epoch),
        "best_validation_mean_directional_bce": float(
            best_val_bce
        ),
        "epochs_run": int(len(epoch_rows)),
        "stopped_early": bool(stopped_early),
        "trainable_parameter_count": trainable_parameters,
        "total_parameter_count": total_parameters,
        "training_seconds": float(total_training_seconds),
        "validation_seconds_total": float(
            total_validation_seconds
        ),
        "wall_seconds": wall_seconds,
        "peak_cuda_allocated_bytes": peak_allocated_bytes,
        "peak_cuda_reserved_bytes": peak_reserved_bytes,
        "peak_cuda_reserved_role": "diagnostic_only",
        "per_horizon": stage1_metrics["per_horizon"],
        "macro_horizon_unweighted":
            stage1_metrics["macro_horizon_unweighted"],
    }
    atomic_json(metrics_path, metrics_payload)

    # Source repository must remain pristine after the real fit.
    s26.import_step24().assert_single_repository_clean(
        repository
    )

    artifact_records = {}
    for key, path in (
        ("best_checkpoint", checkpoint_path),
        ("epoch_history", history_path),
        ("validation_predictions", predictions_path),
        ("fit_metrics", metrics_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_records[key] = {
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }

    input_records = {
        "train_x": train_x_rec,
        "train_y": train_y_rec,
        "validation_x": val_x_rec,
        "validation_y": val_y_rec,
    }

    # fit_manifest.json is the canonical completion record and MUST be written
    # last by the worker.
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "STAGE1_FIT_COMPLETE",
        "completed_at_utc": utc_now(),
        "configuration_fingerprint_sha256":
            EXPECTED_CONFIGURATION_FINGERPRINT,
        "stage1_execution_source_fingerprint_sha256":
            source_fp,
        "stage1_execution_source_sha256":
            source_sha,
        "stage1_development_input_fingerprint_sha256":
            input_fp,
        "step24_preflight_fingerprint_sha256":
            EXPECTED_STEP24_FINGERPRINT,
        "step25_engine_fingerprint_sha256":
            EXPECTED_STEP25_FINGERPRINT,
        "step26_runner_fingerprint_sha256":
            EXPECTED_STEP26_FINGERPRINT,
        "model_id": model_id,
        "config_id": config_id,
        "fold": int(fold),
        "seed": SEED,
        "logical_batch_size": LOGICAL_BATCH,
        "physical_microbatch_size": physical_microbatch,
        "full_logical_accumulation_steps": int(
            schedule["gradient_accumulation_steps"]
        ),
        "partial_logical_batch_rule":
            "exact sample-weighted micro_n/logical_n; no drop/pad/duplicate",
        "learning_rate": learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "optimizer": "AdamW",
        "adamw_betas": list(s26.BETAS),
        "adamw_eps": s26.ADAMW_EPS,
        "loss": "BCEWithLogitsLoss(reduction='mean')",
        "dtype": "float32",
        "amp_enabled": False,
        "tf32_enabled": False,
        "scheduler": "none",
        "train_shuffle":
            "SeedSequence([42, fold, epoch]); candidate-independent",
        "validation_shuffle": False,
        "checkpoint_metric":
            "validation_mean_directional_bce",
        "checkpoint_mode": "min",
        "checkpoint_strict_improvement": True,
        "checkpoint_tie_policy":
            "retain earlier epoch on exact tie",
        "max_epochs": MAX_EPOCHS,
        "minimum_epoch_before_stopping":
            MINIMUM_EPOCH_BEFORE_STOPPING,
        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,
        "threshold": SCREENING_THRESHOLD,
        "threshold_operator": "probability >= 0.50",
        "horizon_order": list(HORIZONS),
        "epochs_run": int(len(epoch_rows)),
        "best_epoch": int(best_epoch),
        "best_validation_mean_directional_bce":
            float(best_val_bce),
        "stopped_early": bool(stopped_early),
        "trainable_parameter_count": trainable_parameters,
        "total_parameter_count": total_parameters,
        "inputs": {
            key: {
                "path": value["path"],
                "size_bytes": int(value["size_bytes"]),
                "sha256": value["sha256"],
                "shape": value["shape"],
                "dtype": value["dtype"],
            }
            for key, value in input_records.items()
        },
        "artifacts": artifact_records,
        "model_build_metadata": build_metadata,
        "source_repository_record": repository,
        "protection_state": {
            "development_training_used": True,
            "development_validation_used": True,
            "threshold_optimization_performed": False,
            "best_seed_selection_performed": False,
            "configuration_selection_performed": False,
            "final_test_data_opened": False,
            "final_test_metric_computed": False,
        },
    }

    atomic_json(manifest_path, manifest)

    torch.cuda.synchronize(device_index)
    torch.cuda.empty_cache()

    return {
        "status": "COMPLETE",
        "model_id": model_id,
        "config_id": config_id,
        "fold": int(fold),
        "epochs_run": int(len(epoch_rows)),
        "best_epoch": int(best_epoch),
        "best_validation_mean_directional_bce":
            float(best_val_bce),
        "macro_balanced_accuracy":
            float(
                stage1_metrics["macro_horizon_unweighted"][
                    "balanced_accuracy"
                ]
            ),
        "macro_roc_auc":
            float(
                stage1_metrics["macro_horizon_unweighted"][
                    "roc_auc"
                ]
            ),
        "macro_mcc":
            float(
                stage1_metrics["macro_horizon_unweighted"][
                    "mcc"
                ]
            ),
        "macro_pr_auc":
            float(
                stage1_metrics["macro_horizon_unweighted"][
                    "pr_auc"
                ]
            ),
        "macro_brier_score":
            float(
                stage1_metrics["macro_horizon_unweighted"][
                    "brier_score"
                ]
            ),
        "physical_microbatch_size": physical_microbatch,
        "trainable_parameter_count": trainable_parameters,
        "peak_cuda_allocated_mib":
            peak_allocated_bytes / (1024 ** 2),
        "wall_seconds": wall_seconds,
        "fit_manifest_path": str(manifest_path.resolve()),
        "fit_manifest_sha256": sha256_file(manifest_path),
    }


def worker_main(
    model_id: str,
    config_id: str,
    fold: int,
) -> int:
    try:
        result = worker_fit(
            model_id=model_id,
            config_id=config_id,
            fold=int(fold),
        )
        print(
            RESULT_PREFIX
            + json.dumps(
                result,
                sort_keys=True,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    except Exception as exc:
        failure = {
            "status": "FAIL",
            "model_id": model_id,
            "config_id": config_id,
            "fold": int(fold),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "final_test_data_opened": False,
            "configuration_selection_performed": False,
        }
        print(
            RESULT_PREFIX
            + json.dumps(
                failure,
                sort_keys=True,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1


# =============================================================================
# 9. STREAMING WORKER ORCHESTRATION
# =============================================================================

def run_worker_streaming(
    model_id: str,
    config_id: str,
    fold: int,
) -> dict:
    directory = fit_dir(model_id, config_id, fold)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = worker_log_path(
        model_id,
        config_id,
        fold,
    )

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    command = [
        sys.executable,
        str(SCRIPT),
        "--worker",
        "--model-id",
        model_id,
        "--config-id",
        config_id,
        "--fold",
        str(int(fold)),
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    lines = []
    result_payloads = []

    assert process.stdout is not None

    for raw in process.stdout:
        line = raw.rstrip("\n")
        lines.append(line)

        if line.startswith(RESULT_PREFIX):
            try:
                result_payloads.append(
                    json.loads(
                        line[len(RESULT_PREFIX):]
                    )
                )
            except Exception:
                pass
        else:
            print(f"        {line}", flush=True)

    return_code = process.wait()

    atomic_text(
        log_path,
        "\n".join(lines) + "\n",
    )

    if len(result_payloads) != 1:
        result = {
            "status": "FAIL",
            "model_id": model_id,
            "config_id": config_id,
            "fold": int(fold),
            "exception_type": "WorkerProtocolError",
            "exception_message":
                f"Expected exactly one worker result marker; "
                f"found {len(result_payloads)}.",
        }
    else:
        result = result_payloads[0]

    result["worker_return_code"] = int(return_code)
    result["worker_log"] = str(log_path.resolve())
    return result


# =============================================================================
# 10. FINAL 80-FIT REGISTER + STEP-27 FREEZE
# =============================================================================

def flatten_fit(
    manifest: dict,
    worker_log: Path | None,
) -> dict:
    metrics = load_json(
        Path(manifest["artifacts"]["fit_metrics"]["path"])
    )

    row = {
        "model_id": manifest["model_id"],
        "config_id": manifest["config_id"],
        "fold": int(manifest["fold"]),
        "seed": int(manifest["seed"]),
        "physical_microbatch_size": int(
            manifest["physical_microbatch_size"]
        ),
        "learning_rate": float(manifest["learning_rate"]),
        "train_n": int(
            manifest["inputs"]["train_x"]["shape"][0]
        ),
        "validation_n": int(
            manifest["inputs"]["validation_x"]["shape"][0]
        ),
        "epochs_run": int(manifest["epochs_run"]),
        "best_epoch": int(manifest["best_epoch"]),
        "best_validation_mean_directional_bce": float(
            manifest["best_validation_mean_directional_bce"]
        ),
        "stopped_early": bool(manifest["stopped_early"]),
        "trainable_parameter_count": int(
            manifest["trainable_parameter_count"]
        ),
        "total_parameter_count": int(
            manifest["total_parameter_count"]
        ),
        "training_seconds": float(
            metrics["training_seconds"]
        ),
        "wall_seconds": float(metrics["wall_seconds"]),
        "peak_cuda_allocated_mib": float(
            metrics["peak_cuda_allocated_bytes"]
        ) / (1024 ** 2),
        "macro_balanced_accuracy": float(
            metrics["macro_horizon_unweighted"][
                "balanced_accuracy"
            ]
        ),
        "macro_roc_auc": float(
            metrics["macro_horizon_unweighted"][
                "roc_auc"
            ]
        ),
        "macro_mcc": float(
            metrics["macro_horizon_unweighted"]["mcc"]
        ),
        "macro_pr_auc": float(
            metrics["macro_horizon_unweighted"][
                "pr_auc"
            ]
        ),
        "macro_brier_score": float(
            metrics["macro_horizon_unweighted"][
                "brier_score"
            ]
        ),
        "fit_manifest_path": str(
            fit_manifest_path(
                manifest["model_id"],
                manifest["config_id"],
                int(manifest["fold"]),
            ).resolve()
        ),
        "fit_manifest_sha256": sha256_file(
            fit_manifest_path(
                manifest["model_id"],
                manifest["config_id"],
                int(manifest["fold"]),
            )
        ),
        "worker_log": (
            None
            if worker_log is None
            else str(worker_log.resolve())
        ),
    }

    for horizon_label in ("1d", "3d", "5d"):
        for metric in (
            "balanced_accuracy",
            "roc_auc",
            "mcc",
            "pr_auc",
            "brier_score",
        ):
            row[f"{horizon_label}_{metric}"] = float(
                metrics["per_horizon"][horizon_label][metric]
            )

    return row


def freeze_complete_stage(
    upstream: dict,
    source_fp: str,
    source_sha: str,
    input_fp: str,
    input_register: dict,
    schedule_map: dict,
    all_manifests: list[dict],
) -> tuple[str, str]:
    if len(all_manifests) != EXPECTED_FITS:
        raise RuntimeError("Cannot freeze Step 27 without 80 fit manifests.")

    rows = []

    for manifest in all_manifests:
        log_path = worker_log_path(
            manifest["model_id"],
            manifest["config_id"],
            int(manifest["fold"]),
        )
        rows.append(
            flatten_fit(
                manifest,
                log_path if log_path.is_file() else None,
            )
        )

    fit_df = pd.DataFrame(rows).sort_values(
        ["model_id", "config_id", "fold"]
    ).reset_index(drop=True)

    if len(fit_df) != EXPECTED_FITS:
        raise RuntimeError("Fit register does not contain 80 rows.")
    if fit_df[["model_id", "config_id", "fold"]].duplicated().any():
        raise RuntimeError("Duplicate candidate/fold in fit register.")

    if set(fit_df["model_id"]) != set(MODEL_ORDER):
        raise RuntimeError("Fit-register model set drift.")

    for (model_id, config_id), group in fit_df.groupby(
        ["model_id", "config_id"],
        sort=True,
    ):
        if sorted(group["fold"].astype(int).tolist()) != list(FOLDS):
            raise RuntimeError(
                f"{model_id}/{config_id}: incomplete fold set."
            )
        params = group["trainable_parameter_count"].astype(int).unique()
        if len(params) != 1:
            raise RuntimeError(
                f"{model_id}/{config_id}: parameter count drift across folds."
            )

    metric_columns = [
        c
        for c in fit_df.columns
        if (
            c.endswith("balanced_accuracy")
            or c.endswith("roc_auc")
            or c.endswith("mcc")
            or c.endswith("pr_auc")
            or c.endswith("brier_score")
            or c
            in (
                "best_validation_mean_directional_bce",
                "training_seconds",
                "wall_seconds",
                "peak_cuda_allocated_mib",
            )
        )
    ]
    for col in metric_columns:
        if not np.isfinite(
            fit_df[col].astype(float)
        ).all():
            raise RuntimeError(
                f"Non-finite fit-register values in {col}."
            )

    atomic_csv(FIT_REGISTER_CSV, fit_df)

    completion_rows = []
    for manifest in all_manifests:
        manifest_path = fit_manifest_path(
            manifest["model_id"],
            manifest["config_id"],
            int(manifest["fold"]),
        )

        completion_rows.append(
            {
                "model_id": manifest["model_id"],
                "config_id": manifest["config_id"],
                "fold": int(manifest["fold"]),
                "fit_manifest_path": str(
                    manifest_path.resolve()
                ),
                "fit_manifest_sha256":
                    sha256_file(manifest_path),
                "best_checkpoint_sha256":
                    manifest["artifacts"][
                        "best_checkpoint"
                    ]["sha256"],
                "epoch_history_sha256":
                    manifest["artifacts"][
                        "epoch_history"
                    ]["sha256"],
                "validation_predictions_sha256":
                    manifest["artifacts"][
                        "validation_predictions"
                    ]["sha256"],
                "fit_metrics_sha256":
                    manifest["artifacts"][
                        "fit_metrics"
                    ]["sha256"],
            }
        )

    completion_df = pd.DataFrame(
        completion_rows
    ).sort_values(
        ["model_id", "config_id", "fold"]
    ).reset_index(drop=True)
    atomic_csv(COMPLETION_HASH_CSV, completion_df)

    semantic = {
        "schema_version": SCHEMA_VERSION,
        "status": "STAGE1_80_FIT_EXECUTION_FROZEN",
        "configuration_fingerprint_sha256":
            EXPECTED_CONFIGURATION_FINGERPRINT,
        "upstream": {
            "step24_fingerprint_sha256":
                EXPECTED_STEP24_FINGERPRINT,
            "step24_manifest_sha256":
                EXPECTED_STEP24_MANIFEST_SHA256,
            "step25_fingerprint_sha256":
                EXPECTED_STEP25_FINGERPRINT,
            "step25_manifest_sha256":
                EXPECTED_STEP25_MANIFEST_SHA256,
            "step26_fingerprint_sha256":
                EXPECTED_STEP26_FINGERPRINT,
            "step26_manifest_sha256":
                EXPECTED_STEP26_MANIFEST_SHA256,
            "step27_execution_source_fingerprint_sha256":
                source_fp,
            "step27_execution_source_sha256":
                source_sha,
            "step27_development_input_fingerprint_sha256":
                input_fp,
        },
        "execution_scope": {
            "models": EXPECTED_MODELS,
            "candidate_configurations": EXPECTED_CANDIDATES,
            "folds": list(FOLDS),
            "completed_fits": EXPECTED_FITS,
            "seed": SEED,
        },
        "protocol": {
            "logical_batch_size": LOGICAL_BATCH,
            "physical_microbatch_source":
                "frozen Step-24 candidate schedule",
            "candidate_independent_shuffle":
                "SeedSequence([42, fold, epoch])",
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "loss":
                "BCEWithLogitsLoss(reduction='mean')",
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "max_epochs": MAX_EPOCHS,
            "minimum_epoch_before_stopping":
                MINIMUM_EPOCH_BEFORE_STOPPING,
            "patience": EARLY_STOPPING_PATIENCE,
            "checkpoint_metric":
                "validation_mean_directional_bce",
            "checkpoint_mode": "min",
            "threshold": SCREENING_THRESHOLD,
            "configuration_selection_performed": False,
            "threshold_optimization_performed": False,
            "best_seed_selection_performed": False,
        },
        "fit_register": {
            "path": str(FIT_REGISTER_CSV.resolve()),
            "rows": int(len(fit_df)),
            "sha256": sha256_file(FIT_REGISTER_CSV),
        },
        "completion_hash_register": {
            "path": str(COMPLETION_HASH_CSV.resolve()),
            "rows": int(len(completion_df)),
            "sha256": sha256_file(COMPLETION_HASH_CSV),
        },
        "development_input_register": {
            "path": str(INPUT_REGISTER_JSON.resolve()),
            "sha256": sha256_file(INPUT_REGISTER_JSON),
            "fingerprint_sha256": input_fp,
        },
        "protection_state": {
            "development_training_used": True,
            "development_validation_used": True,
            "configuration_selection_performed": False,
            "threshold_optimization_performed": False,
            "best_seed_selection_performed": False,
            "final_test_data_opened": False,
            "final_test_metric_computed": False,
            "global_final_test_state":
                upstream["global_final_test_state"],
        },
    }

    fingerprint = canonical_hash(semantic)
    freeze = {
        **semantic,
        "generated_at_utc": utc_now(),
        "stage1_execution_fingerprint_sha256":
            fingerprint,
    }
    atomic_json(FREEZE_JSON, freeze)

    summary_lines = [
        STEP_NAME,
        "=" * 112,
        "Status: PASS",
        f"Completed development fits: {EXPECTED_FITS}/{EXPECTED_FITS}",
        f"Models: {EXPECTED_MODELS}",
        f"Candidates: {EXPECTED_CANDIDATES}",
        f"Folds: {list(FOLDS)}",
        f"Seed: {SEED}",
        "Configuration selection performed: False",
        "Threshold optimization performed: False",
        "Best-seed selection performed: False",
        f"Step-10 final-test seal: {upstream['test_seal_status']}",
        f"Global final-test state: {upstream['global_final_test_state']}",
        "",
        "Step 27 freezes fit-level development results only.",
        "A/B configuration selection must occur in a later development-only step.",
    ]
    atomic_text(
        SUMMARY_TXT,
        "\n".join(summary_lines) + "\n",
    )

    pre_manifest_outputs = [
        FIT_REGISTER_CSV,
        COMPLETION_HASH_CSV,
        FREEZE_JSON,
        SUMMARY_TXT,
        SOURCE_FREEZE_JSON,
        SOURCE_FROZEN_SCRIPT,
        INPUT_REGISTER_JSON,
    ]

    output_records = {}
    for path in pre_manifest_outputs:
        output_records[path.name] = {
            "path": str(path.resolve()),
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
        "frozen_at_utc": utc_now(),
        "configuration_fingerprint_sha256":
            EXPECTED_CONFIGURATION_FINGERPRINT,
        "stage1_execution_fingerprint_sha256":
            fingerprint,
        "step27_execution_source_fingerprint_sha256":
            source_fp,
        "step27_development_input_fingerprint_sha256":
            input_fp,
        "completed_fits": EXPECTED_FITS,
        "configuration_selection_performed": False,
        "final_test_accessed": False,
        "artifacts": output_records,
    }
    atomic_json(MANIFEST_JSON, manifest)

    manifest_sha = sha256_file(MANIFEST_JSON)
    atomic_text(
        MANIFEST_SHA,
        f"{manifest_sha}  {MANIFEST_JSON.name}\n",
    )

    output_hash_rows = []
    for path in (
        *pre_manifest_outputs,
        MANIFEST_JSON,
        MANIFEST_SHA,
    ):
        output_hash_rows.append(
            {
                "path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    atomic_csv(
        OUTPUT_HASH_CSV,
        pd.DataFrame(output_hash_rows),
    )

    return fingerprint, manifest_sha


# =============================================================================
# 11. FINAL VERIFICATION
# =============================================================================

def verify_existing_complete() -> int:
    print("=" * 140)
    print("STEP 27 — VERIFY EXISTING 80-FIT STAGE-1 EXECUTION FREEZE")
    print("=" * 140)

    upstream = verify_upstream()
    source_fp, source_sha = ensure_source_freeze(upstream)
    input_fp, input_register = ensure_input_register(
        upstream,
        source_fp,
    )
    input_map = input_record_map(input_register)
    schedule_map, _ = execution_maps(upstream)

    if not FREEZE_JSON.is_file():
        print("[FAIL] Step-27 global freeze does not exist.")
        return 20

    manifest_sha = verify_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 27",
    )

    manifest = load_json(MANIFEST_JSON)
    freeze = load_json(FREEZE_JSON)

    if manifest.get("step", {}).get("status") != "PASS":
        print("[FAIL] Step-27 manifest status is not PASS.")
        return 21

    stored_fp = freeze["stage1_execution_fingerprint_sha256"]
    recomputed_fp = recompute_freeze_fingerprint(
        freeze,
        "stage1_execution_fingerprint_sha256",
    )

    if stored_fp != recomputed_fp:
        print("[FAIL] Step-27 execution fingerprint no longer recomputes.")
        return 22

    if manifest.get("stage1_execution_fingerprint_sha256") != stored_fp:
        print("[FAIL] Step-27 manifest/freeze fingerprint mismatch.")
        return 23

    if int(manifest.get("completed_fits", -1)) != EXPECTED_FITS:
        print("[FAIL] Step-27 manifest does not report 80 completed fits.")
        return 24

    completed = []

    for model_id in MODEL_ORDER:
        config_ids = sorted(
            config_id
            for mid, config_id in schedule_map
            if mid == model_id
        )
        for config_id in config_ids:
            for fold in FOLDS:
                fit = verify_completed_fit(
                    model_id,
                    config_id,
                    fold,
                    source_fp=source_fp,
                    input_fp=input_fp,
                    schedule_map=schedule_map,
                    input_map=input_map,
                )
                if fit is None:
                    print(
                        f"[FAIL] Fit verification failed: "
                        f"{model_id}/{config_id}/fold{fold}"
                    )
                    return 25
                completed.append(fit)

    if len(completed) != EXPECTED_FITS:
        print("[FAIL] Verified fit count is not 80.")
        return 26

    for record in manifest.get("artifacts", {}).values():
        path = Path(record["path"])
        if not path.is_file():
            print(f"[FAIL] Step-27 artifact missing: {path}")
            return 27
        if sha256_file(path) != record["sha256"]:
            print(f"[FAIL] Step-27 artifact drift: {path}")
            return 28

    if (
        sha256_file(upstream["global_lock_path"])
        != upstream["global_lock_sha256"]
    ):
        print("[FAIL] Global final-test lock changed.")
        return 29

    print(f"[PASS] Configuration fingerprint : {EXPECTED_CONFIGURATION_FINGERPRINT}")
    print(f"[PASS] Step-26 runner freeze     : {EXPECTED_STEP26_FINGERPRINT}")
    print(f"[PASS] Step-27 source freeze     : {source_fp}")
    print(f"[PASS] Development input freeze  : {input_fp}")
    print(f"[PASS] Verified Stage-1 fits     : {len(completed)}")
    print("[PASS] Configuration selection   : NOT PERFORMED")
    print(f"[PASS] Step-27 execution freeze  : {stored_fp}")
    print(f"[PASS] Step-27 manifest          : {manifest_sha}")
    print(f"[PASS] Step-10 test seal         : {upstream['test_seal_status']}")
    print(f"[PASS] Global final-test state   : {upstream['global_final_test_state']}")
    print("=" * 140)
    print("STEP 27 VERIFICATION STATUS: PASS")
    print("=" * 140)
    return 0


# =============================================================================
# 12. MASTER
# =============================================================================

def ordered_candidate_keys(schedule_map: dict) -> list[tuple[str, str]]:
    rank = {model_id: i for i, model_id in enumerate(MODEL_ORDER)}
    return sorted(
        schedule_map,
        key=lambda key: (
            rank[key[0]],
            key[1],
        ),
    )


def master_main(args) -> int:
    print("=" * 140)
    print(STEP_NAME)
    print("=" * 140)

    upstream = verify_upstream()
    print(f"[1/6] Upstream Steps 00-26 verified.")
    print(f"      Configuration fingerprint : {EXPECTED_CONFIGURATION_FINGERPRINT}")
    print(f"      Step-26 runner freeze      : {EXPECTED_STEP26_FINGERPRINT}")
    print(f"      Step-10 final-test seal    : {upstream['test_seal_status']}")
    print(f"      Global final-test state    : {upstream['global_final_test_state']}")

    print("[2/6] Freezing/verifying immutable Step-27 execution source...")
    source_fp, source_sha = ensure_source_freeze(upstream)
    print(f"      Step-27 source SHA-256     : {source_sha}")
    print(f"      Step-27 source fingerprint : {source_fp}")

    print("[3/6] Building/verifying development-only input register...")
    input_fp, input_register = ensure_input_register(
        upstream,
        source_fp,
    )
    input_map = input_record_map(input_register)

    print(f"      Development input records  : {len(input_register['records'])}")
    print(f"      Development input fingerprint: {input_fp}")
    print("      Development NPY headers     : READ")
    print("      Development bytes           : HASHED")
    print("      Validation performance      : NOT COMPUTED")
    print("      Final-test parquet          : NOT OPENED OR HASHED")

    print("[4/6] Reconciling frozen candidate/fold execution map...")
    schedule_map, batch_plan = execution_maps(upstream)
    print(f"      Candidate configurations    : {len(schedule_map)}")
    print(f"      Candidate/fold plans        : {len(batch_plan)}")
    print(f"      Planned Stage-1 fits        : {EXPECTED_FITS}")

    if args.prepare_only:
        print()
        print("=" * 140)
        print("STEP 27 PREPARE STATUS: PASS")
        print("=" * 140)
        print("Step-27 execution source         : FROZEN")
        print("Development input register       : FROZEN")
        print("Real Stage-1 fits started        : False")
        print("Configuration selection          : NOT PERFORMED")
        print("Threshold optimization           : NOT PERFORMED")
        print("Best-seed selection              : NOT PERFORMED")
        print(f"Step-10 final-test seal          : {upstream['test_seal_status']}")
        print(f"Global final-test state          : {upstream['global_final_test_state']}")
        print("=" * 140)
        print(
            "NEXT: do not run all 80 fits yet. Audit this output, then run "
            "--max-new-fits 1 when instructed."
        )
        return 0

    if FREEZE_JSON.exists():
        raise RuntimeError(
            f"Step 27 is already globally frozen:\n{FREEZE_JSON}\n"
            "Use --verify-only; do not overwrite."
        )

    print("[5/6] Executing/resuming requested frozen Stage-1 fits...")
    print("-" * 140)

    ordered_keys = ordered_candidate_keys(schedule_map)

    if args.model_id is not None:
        if args.model_id not in MODEL_ORDER:
            raise ValueError(
                f"--model-id must be one of {list(MODEL_ORDER)}"
            )
        ordered_keys = [
            key for key in ordered_keys
            if key[0] == args.model_id
        ]

    if args.config_id is not None:
        ordered_keys = [
            key for key in ordered_keys
            if key[1] == args.config_id
        ]
        if not ordered_keys:
            raise ValueError(
                "--config-id does not identify a frozen candidate "
                "within the current model filter."
            )

    folds = list(FOLDS)
    if args.fold is not None:
        if args.fold not in FOLDS:
            raise ValueError("--fold must be 1..5")
        folds = [int(args.fold)]

    new_fits = 0
    skipped_fits = 0

    for model_id, config_id in ordered_keys:
        physical = int(
            schedule_map[(model_id, config_id)][
                "physical_microbatch_size"
            ]
        )

        for fold in folds:
            prior = verify_completed_fit(
                model_id,
                config_id,
                fold,
                source_fp=source_fp,
                input_fp=input_fp,
                schedule_map=schedule_map,
                input_map=input_map,
            )

            if prior is not None:
                skipped_fits += 1
                metrics = load_json(
                    Path(
                        prior["artifacts"]["fit_metrics"]["path"]
                    )
                )
                print(
                    f"SKIP {model_id:<14} {config_id:<20} "
                    f"fold={fold} COMPLETE+HASHED "
                    f"best={prior['best_epoch']} "
                    f"BA={metrics['macro_horizon_unweighted']['balanced_accuracy']:.6f}"
                )
                continue

            if (
                args.max_new_fits is not None
                and new_fits >= args.max_new_fits
            ):
                print("-" * 140)
                print("STEP 27 PAUSED — RESUMABLE, NOT GLOBALLY FROZEN")
                print(f"New fits completed this invocation: {new_fits}")
                print(f"Hash-valid fits skipped         : {skipped_fits}")
                print("Configuration selection          : NOT PERFORMED")
                print("Final-test data opened           : False")
                print("=" * 140)
                return 2

            archived = archive_incomplete_fit(
                model_id,
                config_id,
                fold,
            )
            if archived is not None:
                print(
                    f"ARCHIVE incomplete prior fit -> {archived}"
                )

            print(
                f"RUN  {model_id:<14} {config_id:<20} "
                f"fold={fold} mb={physical}"
            )

            result = run_worker_streaming(
                model_id,
                config_id,
                fold,
            )

            if (
                result.get("status") != "COMPLETE"
                or int(result.get("worker_return_code", -1)) != 0
            ):
                FAILED_DIR.mkdir(parents=True, exist_ok=True)
                failure_path = FAILED_DIR / (
                    f"27_stage1_fit_FAILED__{model_id}__{config_id}"
                    f"__fold_{fold:02d}__"
                    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    + ".json"
                )
                atomic_json(
                    failure_path,
                    {
                        "status": "FAIL",
                        "failed_at_utc": utc_now(),
                        "failed_fit": result,
                        "completed_new_fits_before_failure":
                            new_fits,
                        "configuration_selection_performed":
                            False,
                        "final_test_data_opened": False,
                    },
                )

                print()
                print("=" * 140)
                print("STEP 27 FIT FAILED — GLOBAL STAGE NOT FROZEN")
                print("=" * 140)
                print(
                    f"Fit       : {model_id}/{config_id}/fold{fold}"
                )
                print(
                    f"Error     : {result.get('exception_type')}: "
                    f"{result.get('exception_message')}"
                )
                print(
                    f"Worker log: {result.get('worker_log')}"
                )
                print(f"Diagnostic: {failure_path}")
                print("Completed valid fits remain resumable: True")
                print("Configuration selection          : NOT PERFORMED")
                print("Final-test data opened           : False")
                print("=" * 140)
                return 1

            verified = verify_completed_fit(
                model_id,
                config_id,
                fold,
                source_fp=source_fp,
                input_fp=input_fp,
                schedule_map=schedule_map,
                input_map=input_map,
            )
            if verified is None:
                raise RuntimeError(
                    "Worker reported COMPLETE but fit_manifest verification failed."
                )

            new_fits += 1

            print(
                f"PASS {model_id:<14} {config_id:<20} "
                f"fold={fold} "
                f"epochs={result['epochs_run']:<3} "
                f"best={result['best_epoch']:<3} "
                f"valBCE={result['best_validation_mean_directional_bce']:.6f} "
                f"BA={result['macro_balanced_accuracy']:.6f} "
                f"AUC={result['macro_roc_auc']:.6f} "
                f"peak={result['peak_cuda_allocated_mib']:.1f}MiB"
            )

    print("-" * 140)

    # Count ALL 80 fits, including valid fits outside any operational filter.
    all_manifests = []
    missing = []

    for model_id, config_id in ordered_candidate_keys(schedule_map):
        for fold in FOLDS:
            fit = verify_completed_fit(
                model_id,
                config_id,
                fold,
                source_fp=source_fp,
                input_fp=input_fp,
                schedule_map=schedule_map,
                input_map=input_map,
            )
            if fit is None:
                missing.append(
                    f"{model_id}/{config_id}/fold{fold}"
                )
            else:
                all_manifests.append(fit)

    if missing:
        print()
        print("=" * 140)
        print("STEP 27 INCOMPLETE — RESUMABLE, NOT GLOBALLY FROZEN")
        print("=" * 140)
        print(f"Verified completed fits          : {len(all_manifests)}/{EXPECTED_FITS}")
        print(f"Remaining fits                   : {len(missing)}")
        print(f"New fits completed this run      : {new_fits}")
        print("Configuration selection          : NOT PERFORMED")
        print("Threshold optimization           : NOT PERFORMED")
        print("Best-seed selection              : NOT PERFORMED")
        print("Final-test data opened           : False")
        print("=" * 140)
        return 2

    print("[6/6] Freezing the complete 80-fit development execution corpus...")

    if (
        sha256_file(upstream["global_lock_path"])
        != upstream["global_lock_sha256"]
    ):
        raise RuntimeError(
            "Global final-test lock changed during Stage-1 execution."
        )

    fingerprint, manifest_sha = freeze_complete_stage(
        upstream=upstream,
        source_fp=source_fp,
        source_sha=source_sha,
        input_fp=input_fp,
        input_register=input_register,
        schedule_map=schedule_map,
        all_manifests=all_manifests,
    )

    print()
    print("=" * 140)
    print("STEP 27 — 80-FIT STAGE-1 EXECUTION COMPLETE")
    print("=" * 140)
    print(f"Verified completed fits          : {EXPECTED_FITS}/{EXPECTED_FITS}")
    print(f"Models                           : {EXPECTED_MODELS}")
    print(f"Candidate configurations         : {EXPECTED_CANDIDATES}")
    print(f"Chronological folds              : {list(FOLDS)}")
    print(f"Screening seed                   : {SEED}")
    print("Configuration selection          : NOT PERFORMED")
    print("Threshold optimization           : NOT PERFORMED")
    print("Best-seed selection              : NOT PERFORMED")
    print(f"Step-27 execution fingerprint    : {fingerprint}")
    print(f"Step-27 manifest SHA-256         : {manifest_sha}")
    print(f"Step-10 final-test seal          : {upstream['test_seal_status']}")
    print(f"Global final-test state          : {upstream['global_final_test_state']}")
    print("=" * 140)
    print(
        "PASS: All 80 Stage-1 development fits are complete and hash-verified."
    )
    print(
        "PASS: Step 27 froze fit-level results only; A/B configuration "
        "selection was not performed."
    )
    print(
        "PASS: No threshold optimization, best-seed selection, or final-test "
        "access occurred."
    )
    print("=" * 140)
    print(
        "NEXT: audit this output, then run --verify-only when instructed. "
        "Do not perform A/B configuration selection yet."
    )
    return 0


# =============================================================================
# 13. CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=STEP_NAME)

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Freeze/verify the Step-27 source and development input register "
            "without starting any real fit."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an already-complete 80-fit Step-27 freeze.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
    )
    parser.add_argument("--model-id")
    parser.add_argument("--config-id")
    parser.add_argument("--fold", type=int)
    parser.add_argument(
        "--max-new-fits",
        type=int,
        default=None,
        help=(
            "Operational/resume control only. Execute at most this many new "
            "already-frozen fits in this invocation."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_new_fits is not None and args.max_new_fits < 1:
        raise SystemExit("--max-new-fits must be >= 1")

    if args.worker:
        if (
            args.model_id is None
            or args.config_id is None
            or args.fold is None
        ):
            raise SystemExit(
                "--worker requires --model-id, --config-id and --fold"
            )
        return worker_main(
            args.model_id,
            args.config_id,
            args.fold,
        )

    if args.prepare_only and args.verify_only:
        raise SystemExit(
            "--prepare-only and --verify-only are mutually exclusive"
        )

    if args.verify_only:
        return verify_existing_complete()

    return master_main(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nStep 27 interrupted by user. Hash-valid completed fits remain resumable."
        )
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print()
        print("=" * 140)
        print("STEP 27 FAILED")
        print("=" * 140)
        print(f"{type(exc).__name__}: {exc}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        print("=" * 140)
        print("Do not modify frozen upstream Steps 00-26.")
        print("Do not access the final test.")
        raise SystemExit(1)
