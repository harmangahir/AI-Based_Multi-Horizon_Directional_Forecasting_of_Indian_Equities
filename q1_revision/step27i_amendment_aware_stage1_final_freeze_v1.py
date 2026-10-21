#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 27I — AMENDMENT-AWARE FINAL STAGE-1 80-FIT FREEZE

Purpose
-------
Create the final Stage-1 development-execution freeze after all 80 candidate/fold
fits exist, while explicitly preserving the frozen Step-27F TIMESNET_B protocol
amendment (physical microbatch 2 -> 64).

Why Step 27I exists
-------------------
The immutable original Step-27 global freeze describes physical microbatch
provenance as the unamended Step-24 candidate schedule. That is no longer fully
true for TIMESNET_B because Step-27F was frozen BEFORE any completed TIMESNET_B
fit after source audit confirmed that TimesNet period selection averages FFT
amplitudes across the current physical batch. Therefore the physical microbatch
change affects forward semantics and cannot be described as a pure execution
optimization.

Step 27I does NOT train models, select configurations, tune thresholds, choose
seeds, or open the final test. It:
  1. verifies the Step-27F/G/H amendment/provenance chain;
  2. verifies the immutable original Step-27 source and development input register;
  3. builds the amendment-aware execution map;
  4. hash-verifies all 80 completed fit manifests and their canonical artifacts;
  5. creates an amendment-aware 80-row fit register and completion-hash register;
  6. freezes a new Stage-1 corpus manifest with explicit amendment semantics;
  7. leaves the original ordinary Step-27 global freeze untouched/uncreated.

Recommended usage
-----------------
First:
    python step27i_amendment_aware_stage1_final_freeze_v1.py --prepare-only

After auditing PASS:
    python step27i_amendment_aware_stage1_final_freeze_v1.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import step00_config as cfg
import step27_stage1_execution_v3 as s27
import step27g_timesnetb_mb64_compatibility_runner_v1 as s27g
import step27h_amendment_aware_stage1_continuation_v1 as s27h


# =============================================================================
# 1. IDENTITY / FROZEN PROVENANCE
# =============================================================================

STEP_NUMBER = "27I"
STEP_NAME = "STEP 27I — AMENDMENT-AWARE FINAL STAGE-1 80-FIT FREEZE"
SCHEMA_VERSION = "1.0"
SCRIPT = Path(__file__).resolve()

EXPECTED_CONFIGURATION_FINGERPRINT = (
    "9675cb42e5544d9a3cda235208b33cc4c27f7583531535505b42056aa86b04dd"
)

EXPECTED_STEP27_SOURCE_SHA256 = (
    "83ac36dcc5e178051ac90865970f4062626a05f48eda4c0548b3a8b1799ec074"
)
EXPECTED_STEP27_SOURCE_FINGERPRINT = (
    "3978415d953a1561ebe345180529c97bb8e4a39a89219668008a25b3ad72b3d9"
)

EXPECTED_STEP27F_AMENDMENT_SHA256 = (
    "04c30b59b8bef7327b44e1764240ead258edbba64f278e31ca29249141dfdce8"
)

EXPECTED_STEP27G_SOURCE_SHA256 = (
    "566ee24f269d1ec049cb815258409527e47ab0a438adf4bd78f7731f6a4a318b"
)
EXPECTED_STEP27G_SOURCE_FINGERPRINT = (
    "b5efacae2f49f9110e82dd9a8a2ae08e7053621f5342b5cf8f42b78f15a9de66"
)

EXPECTED_STEP27H_SOURCE_SHA256 = (
    "7347838428c75aba0235ee848c659d80c7d8b979d9c19b311429a357e157085c"
)
EXPECTED_STEP27H_SOURCE_FINGERPRINT = (
    "e5ce7b8b3d5cb74d2f494413f1490cfaa9809eae3ecd7552955aa7618998cbb0"
)

TIMESNET_MODEL_ID = "TIMESNET"
TIMESNET_A_CONFIG_ID = "TIMESNET_A"
TIMESNET_B_CONFIG_ID = "TIMESNET_B"
TIMESNET_B_OLD_MICROBATCH = 2
TIMESNET_B_NEW_MICROBATCH = 64
LOGICAL_BATCH = 256
TIMESNET_B_NEW_ACCUMULATION = 4

EXPECTED_FITS = 80
EXPECTED_MODELS = 8
EXPECTED_CANDIDATES = 16


# =============================================================================
# 2. STEP-27I OUTPUTS — DISTINCT FROM ORDINARY STEP-27 GLOBAL FREEZE
# =============================================================================

SOURCE_FREEZE_JSON = (
    cfg.MANIFEST_DIR
    / "27i_stage1_amendment_aware_final_freeze_source_freeze.json"
)
SOURCE_FROZEN_SCRIPT = (
    cfg.MANIFEST_DIR
    / "step27i_stage1_amendment_aware_final_freeze_frozen.py"
)

FIT_REGISTER_CSV = (
    cfg.AUDIT_DIR
    / "27i_stage1_amendment_aware_fit_register.csv"
)
COMPLETION_HASH_CSV = (
    cfg.AUDIT_DIR
    / "27i_stage1_amendment_aware_completion_hash_register.csv"
)
OUTPUT_HASH_CSV = (
    cfg.AUDIT_DIR
    / "27i_stage1_amendment_aware_output_hashes.csv"
)

FREEZE_JSON = (
    cfg.MANIFEST_DIR
    / "27i_stage1_amendment_aware_execution_freeze.json"
)
MANIFEST_JSON = (
    cfg.MANIFEST_DIR
    / "27i_stage1_amendment_aware_execution_manifest.json"
)
MANIFEST_SHA = (
    cfg.MANIFEST_DIR
    / "27i_stage1_amendment_aware_execution_manifest.sha256"
)
SUMMARY_TXT = (
    cfg.REPORT_DIR
    / "27i_stage1_amendment_aware_execution_summary.txt"
)


# =============================================================================
# 3. BASIC HELPERS
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(payload: Any) -> str:
    return cfg.canonical_json_sha256(payload)


def ensure_parent_dirs() -> None:
    for path in (
        SOURCE_FREEZE_JSON,
        SOURCE_FROZEN_SCRIPT,
        FIT_REGISTER_CSV,
        COMPLETION_HASH_CSV,
        OUTPUT_HASH_CSV,
        FREEZE_JSON,
        MANIFEST_JSON,
        MANIFEST_SHA,
        SUMMARY_TXT,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 4. FREEZE STEP-27I SOURCE
# =============================================================================

def build_source_semantic() -> dict:
    return {
        "schema_version": 1,
        "status": "STEP27I_AMENDMENT_AWARE_FINAL_FREEZE_SOURCE_FROZEN",
        "configuration_fingerprint_sha256":
            EXPECTED_CONFIGURATION_FINGERPRINT,
        "step27i_source_sha256": sha256_file(SCRIPT),
        "original_step27": {
            "source_sha256": EXPECTED_STEP27_SOURCE_SHA256,
            "source_fingerprint_sha256":
                EXPECTED_STEP27_SOURCE_FINGERPRINT,
        },
        "step27f_protocol_amendment": {
            "manifest_sha256": EXPECTED_STEP27F_AMENDMENT_SHA256,
            "model_id": TIMESNET_MODEL_ID,
            "config_id": TIMESNET_B_CONFIG_ID,
            "physical_microbatch_old": TIMESNET_B_OLD_MICROBATCH,
            "physical_microbatch_new": TIMESNET_B_NEW_MICROBATCH,
            "logical_batch_unchanged": LOGICAL_BATCH,
            "forward_semantics_affected": True,
        },
        "step27g": {
            "source_sha256": EXPECTED_STEP27G_SOURCE_SHA256,
            "source_fingerprint_sha256":
                EXPECTED_STEP27G_SOURCE_FINGERPRINT,
        },
        "step27h": {
            "source_sha256": EXPECTED_STEP27H_SOURCE_SHA256,
            "source_fingerprint_sha256":
                EXPECTED_STEP27H_SOURCE_FINGERPRINT,
        },
        "scope": {
            "expected_models": EXPECTED_MODELS,
            "expected_candidate_configurations": EXPECTED_CANDIDATES,
            "expected_fits": EXPECTED_FITS,
            "training_performed_by_step27i": False,
            "configuration_selection_performed": False,
            "threshold_optimization_performed": False,
            "best_seed_selection_performed": False,
            "final_test_allowed": False,
        },
    }


def ensure_source_freeze() -> tuple[str, str]:
    ensure_parent_dirs()

    semantic = build_source_semantic()
    source_sha = semantic["step27i_source_sha256"]
    fingerprint = canonical_hash(semantic)

    if not SOURCE_FREEZE_JSON.exists():
        if SOURCE_FROZEN_SCRIPT.exists():
            raise RuntimeError(
                "Step-27I frozen source exists without source-freeze JSON."
            )

        payload = {
            **semantic,
            "step27i_source_fingerprint_sha256": fingerprint,
        }
        s27.atomic_json(SOURCE_FREEZE_JSON, payload)
        shutil.copy2(SCRIPT, SOURCE_FROZEN_SCRIPT)

    payload = load_json(SOURCE_FREEZE_JSON)

    if payload.get("step27i_source_sha256") != source_sha:
        raise RuntimeError(
            "Current Step-27I source differs from its frozen source."
        )

    if (
        payload.get("step27i_source_fingerprint_sha256")
        != fingerprint
    ):
        raise RuntimeError("Step-27I source fingerprint drift.")

    recompute = dict(payload)
    stored_fp = recompute.pop(
        "step27i_source_fingerprint_sha256", None
    )
    if canonical_hash(recompute) != stored_fp:
        raise RuntimeError(
            "Step-27I source-freeze semantic no longer recomputes."
        )

    if not SOURCE_FROZEN_SCRIPT.is_file():
        raise FileNotFoundError(SOURCE_FROZEN_SCRIPT)

    if sha256_file(SOURCE_FROZEN_SCRIPT) != source_sha:
        raise RuntimeError("Frozen Step-27I source copy drift.")

    return fingerprint, source_sha


# =============================================================================
# 5. VERIFY AMENDMENT / CONTINUATION PROVENANCE
# =============================================================================

def verify_amendment_chain() -> dict:
    amendment = s27g.verify_step27f_amendment()

    if sha256_file(s27g.SCRIPT) != EXPECTED_STEP27G_SOURCE_SHA256:
        raise RuntimeError("Current Step-27G source SHA drift.")

    g_fp, g_sha = s27g.ensure_source_freeze()
    if g_sha != EXPECTED_STEP27G_SOURCE_SHA256:
        raise RuntimeError("Frozen Step-27G source SHA drift.")
    if g_fp != EXPECTED_STEP27G_SOURCE_FINGERPRINT:
        raise RuntimeError("Frozen Step-27G source fingerprint drift.")

    s27h.verify_step27g_provenance()
    h_fp, h_sha = s27h.ensure_source_freeze()

    if h_sha != EXPECTED_STEP27H_SOURCE_SHA256:
        raise RuntimeError("Frozen Step-27H source SHA drift.")
    if h_fp != EXPECTED_STEP27H_SOURCE_FINGERPRINT:
        raise RuntimeError("Frozen Step-27H source fingerprint drift.")

    return amendment


# =============================================================================
# 6. VERIFY ORIGINAL STEP-27 SOURCE, INPUTS, EXECUTION MAP, AND ALL 80 FITS
# =============================================================================

def verify_complete_corpus() -> dict:
    upstream = s27.verify_upstream()

    if (
        upstream.get("global_final_test_state")
        != "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError("Global final-test state is not LOCKED_NOT_OPENED.")

    source_fp, source_sha = s27.ensure_source_freeze(upstream)

    if source_sha != EXPECTED_STEP27_SOURCE_SHA256:
        raise RuntimeError("Original Step-27 source SHA drift.")
    if source_fp != EXPECTED_STEP27_SOURCE_FINGERPRINT:
        raise RuntimeError("Original Step-27 source fingerprint drift.")

    input_fp, input_register = s27.ensure_input_register(
        upstream,
        source_fp,
    )
    input_map = s27.input_record_map(input_register)

    schedule_map, batch_plan = s27g.amended_execution_maps(upstream)

    # Explicit protocol assertions.
    ta = schedule_map[(TIMESNET_MODEL_ID, TIMESNET_A_CONFIG_ID)]
    tb = schedule_map[(TIMESNET_MODEL_ID, TIMESNET_B_CONFIG_ID)]

    if int(ta["physical_microbatch_size"]) != 64:
        raise RuntimeError("TIMESNET_A physical microbatch drift.")
    if int(tb["physical_microbatch_size"]) != TIMESNET_B_NEW_MICROBATCH:
        raise RuntimeError("Amended TIMESNET_B physical microbatch drift.")
    if int(tb["gradient_accumulation_steps"]) != TIMESNET_B_NEW_ACCUMULATION:
        raise RuntimeError("Amended TIMESNET_B accumulation-step drift.")
    if (
        tb.get("step27f_protocol_amendment_sha256")
        != EXPECTED_STEP27F_AMENDMENT_SHA256
    ):
        raise RuntimeError("TIMESNET_B schedule amendment provenance drift.")

    if len(batch_plan) != EXPECTED_FITS:
        raise RuntimeError(
            f"Amended candidate/fold plan has {len(batch_plan)} rows, "
            f"expected {EXPECTED_FITS}."
        )

    all_manifests = []
    missing = []

    for model_id, config_id in s27.ordered_candidate_keys(schedule_map):
        for fold in s27.FOLDS:
            fit = s27g.amended_verify_completed_fit(
                model_id,
                config_id,
                int(fold),
                source_fp=source_fp,
                input_fp=input_fp,
                schedule_map=schedule_map,
                input_map=input_map,
            )
            if fit is None:
                missing.append(
                    f"{model_id}/{config_id}/fold{int(fold)}"
                )
            else:
                all_manifests.append(fit)

    if missing:
        raise RuntimeError(
            "Stage-1 corpus is not complete under amendment-aware "
            "verification. Missing/invalid fits:\n  - "
            + "\n  - ".join(missing)
        )

    if len(all_manifests) != EXPECTED_FITS:
        raise RuntimeError(
            f"Verified {len(all_manifests)} fits, expected {EXPECTED_FITS}."
        )

    # Re-verify the five amended TIMESNET_B manifests explicitly.
    tb_fits = [
        m for m in all_manifests
        if (
            m["model_id"] == TIMESNET_MODEL_ID
            and m["config_id"] == TIMESNET_B_CONFIG_ID
        )
    ]
    if len(tb_fits) != 5:
        raise RuntimeError("Expected five verified TIMESNET_B manifests.")

    for m in tb_fits:
        amendment = m.get("protocol_amendment", {})
        if amendment.get("step") != "27F":
            raise RuntimeError(
                "TIMESNET_B fit missing Step-27F amendment provenance."
            )
        if (
            amendment.get("manifest_sha256")
            != EXPECTED_STEP27F_AMENDMENT_SHA256
        ):
            raise RuntimeError(
                "TIMESNET_B fit amendment manifest SHA drift."
            )
        if int(m.get("physical_microbatch_size", -1)) != 64:
            raise RuntimeError(
                "TIMESNET_B fit physical microbatch is not 64."
            )
        if int(m.get("logical_batch_size", -1)) != LOGICAL_BATCH:
            raise RuntimeError("TIMESNET_B logical batch drift.")

    # The global final-test lock must not change during verification.
    if (
        sha256_file(Path(upstream["global_lock_path"]))
        != upstream["global_lock_sha256"]
    ):
        raise RuntimeError(
            "Global final-test lock changed during Step-27I verification."
        )

    return {
        "upstream": upstream,
        "source_fp": source_fp,
        "source_sha": source_sha,
        "input_fp": input_fp,
        "input_register": input_register,
        "input_map": input_map,
        "schedule_map": schedule_map,
        "batch_plan": batch_plan,
        "all_manifests": all_manifests,
    }


# =============================================================================
# 7. BUILD REGISTERS WITH ORIGINAL STEP-27 FLATTENING LOGIC
# =============================================================================

def build_fit_register(all_manifests: list[dict]) -> pd.DataFrame:
    rows = []

    for manifest in all_manifests:
        log_path = s27.worker_log_path(
            manifest["model_id"],
            manifest["config_id"],
            int(manifest["fold"]),
        )
        rows.append(
            s27.flatten_fit(
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

    if set(fit_df["model_id"]) != set(s27.MODEL_ORDER):
        raise RuntimeError("Fit-register model set drift.")

    for (model_id, config_id), group in fit_df.groupby(
        ["model_id", "config_id"],
        sort=True,
    ):
        if sorted(group["fold"].astype(int).tolist()) != list(s27.FOLDS):
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
        if not np.isfinite(fit_df[col].astype(float)).all():
            raise RuntimeError(
                f"Non-finite fit-register values in {col}."
            )

    # Explicitly confirm that the register captures amended TimesNet-B.
    tb = fit_df[
        (fit_df["model_id"] == TIMESNET_MODEL_ID)
        & (fit_df["config_id"] == TIMESNET_B_CONFIG_ID)
    ]

    if len(tb) != 5:
        raise RuntimeError("Fit register has incomplete TIMESNET_B rows.")

    if "physical_microbatch_size" in tb.columns:
        observed = set(
            tb["physical_microbatch_size"].astype(int).tolist()
        )
        if observed != {TIMESNET_B_NEW_MICROBATCH}:
            raise RuntimeError(
                f"TIMESNET_B fit-register microbatch drift: {observed}"
            )

    return fit_df


def build_completion_register(
    all_manifests: list[dict],
) -> pd.DataFrame:
    rows = []

    for manifest in all_manifests:
        manifest_path = s27.fit_manifest_path(
            manifest["model_id"],
            manifest["config_id"],
            int(manifest["fold"]),
        )

        rows.append(
            {
                "model_id": manifest["model_id"],
                "config_id": manifest["config_id"],
                "fold": int(manifest["fold"]),
                "fit_manifest_path": str(manifest_path.resolve()),
                "fit_manifest_sha256": sha256_file(manifest_path),
                "best_checkpoint_sha256":
                    manifest["artifacts"]["best_checkpoint"]["sha256"],
                "epoch_history_sha256":
                    manifest["artifacts"]["epoch_history"]["sha256"],
                "validation_predictions_sha256":
                    manifest["artifacts"]["validation_predictions"]["sha256"],
                "fit_metrics_sha256":
                    manifest["artifacts"]["fit_metrics"]["sha256"],
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["model_id", "config_id", "fold"]
    ).reset_index(drop=True)

    if len(out) != EXPECTED_FITS:
        raise RuntimeError(
            "Completion hash register does not contain 80 rows."
        )

    return out


# =============================================================================
# 8. AMENDMENT-AWARE FINAL FREEZE
# =============================================================================

def build_freeze_semantic(
    state: dict,
    fit_df: pd.DataFrame,
    completion_df: pd.DataFrame,
) -> dict:
    upstream = state["upstream"]

    return {
        "schema_version": SCHEMA_VERSION,
        "step": STEP_NUMBER,
        "status": "STAGE1_80_FIT_EXECUTION_FROZEN_AMENDMENT_AWARE",
        "configuration_fingerprint_sha256":
            EXPECTED_CONFIGURATION_FINGERPRINT,
        "upstream": {
            "step24_fingerprint_sha256": s27.EXPECTED_STEP24_FINGERPRINT,
            "step24_manifest_sha256": s27.EXPECTED_STEP24_MANIFEST_SHA256,
            "step25_fingerprint_sha256": s27.EXPECTED_STEP25_FINGERPRINT,
            "step25_manifest_sha256": s27.EXPECTED_STEP25_MANIFEST_SHA256,
            "step26_fingerprint_sha256": s27.EXPECTED_STEP26_FINGERPRINT,
            "step26_manifest_sha256": s27.EXPECTED_STEP26_MANIFEST_SHA256,
            "step27_execution_source_fingerprint_sha256":
                state["source_fp"],
            "step27_execution_source_sha256":
                state["source_sha"],
            "step27_development_input_fingerprint_sha256":
                state["input_fp"],
            "step27f_protocol_amendment_sha256":
                EXPECTED_STEP27F_AMENDMENT_SHA256,
            "step27g_source_sha256":
                EXPECTED_STEP27G_SOURCE_SHA256,
            "step27g_source_fingerprint_sha256":
                EXPECTED_STEP27G_SOURCE_FINGERPRINT,
            "step27h_source_sha256":
                EXPECTED_STEP27H_SOURCE_SHA256,
            "step27h_source_fingerprint_sha256":
                EXPECTED_STEP27H_SOURCE_FINGERPRINT,
        },
        "execution_scope": {
            "models": EXPECTED_MODELS,
            "candidate_configurations": EXPECTED_CANDIDATES,
            "folds": list(s27.FOLDS),
            "completed_fits": EXPECTED_FITS,
            "seed": s27.SEED,
        },
        "protocol": {
            "logical_batch_size": s27.LOGICAL_BATCH,
            "physical_microbatch_source": (
                "Frozen Step-24 candidate schedule for all configurations "
                "except TIMESNET_B, which is governed by the frozen "
                "Step-27F pre-performance protocol amendment."
            ),
            "timesnet_b_protocol_amendment": {
                "classification": (
                    "Pre-performance protocol amendment; NOT a pure "
                    "execution-only optimization because official TimesNet "
                    "FFT_for_Period averages frequency amplitudes across the "
                    "current physical batch before period selection."
                ),
                "model_id": TIMESNET_MODEL_ID,
                "config_id": TIMESNET_B_CONFIG_ID,
                "physical_microbatch_old": TIMESNET_B_OLD_MICROBATCH,
                "physical_microbatch_new": TIMESNET_B_NEW_MICROBATCH,
                "logical_batch_unchanged": LOGICAL_BATCH,
                "full_logical_accumulation_steps_new":
                    TIMESNET_B_NEW_ACCUMULATION,
                "timesnet_a_physical_microbatch": 64,
                "a_b_timesnet_physical_batch_semantics_matched": True,
                "amendment_manifest_sha256":
                    EXPECTED_STEP27F_AMENDMENT_SHA256,
                "all_other_training_protocol_fields_unchanged": True,
            },
            "gradient_accumulation_interpretation": (
                "Preserves logical-loss/update arithmetic; does NOT imply "
                "forward-equivalence to a physical batch of 256."
            ),
            "candidate_independent_shuffle":
                "SeedSequence([42, fold, epoch])",
            "optimizer": "AdamW",
            "weight_decay": s27.WEIGHT_DECAY,
            "loss": "BCEWithLogitsLoss(reduction='mean')",
            "gradient_clip_norm": s27.GRADIENT_CLIP_NORM,
            "max_epochs": s27.MAX_EPOCHS,
            "minimum_epoch_before_stopping":
                s27.MINIMUM_EPOCH_BEFORE_STOPPING,
            "patience": s27.EARLY_STOPPING_PATIENCE,
            "checkpoint_metric":
                "validation_mean_directional_bce",
            "checkpoint_mode": "min",
            "threshold": s27.SCREENING_THRESHOLD,
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
            "path": str(s27.INPUT_REGISTER_JSON.resolve()),
            "sha256": sha256_file(s27.INPUT_REGISTER_JSON),
            "fingerprint_sha256": state["input_fp"],
        },
        "ordinary_step27_global_freeze": {
            "used": False,
            "reason": (
                "Ordinary Step-27 freeze is not amendment-aware for "
                "TIMESNET_B physical-batch semantics."
            ),
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


def ordinary_step27_global_freeze_guard() -> None:
    # Source/input freeze artifacts are expected. The final ordinary global
    # freeze/manifest must not have been created, because Step-27H intentionally
    # blocked it once all 80 fits existed.
    unexpected = [
        p
        for p in (
            s27.FREEZE_JSON,
            s27.MANIFEST_JSON,
            s27.MANIFEST_SHA,
        )
        if p.exists()
    ]
    if unexpected:
        raise RuntimeError(
            "Unexpected ordinary Step-27 global-freeze artifacts exist. "
            "Do not overwrite or mix them with Step-27I; audit first:\n  - "
            + "\n  - ".join(str(p) for p in unexpected)
        )


def freeze_complete_stage(state: dict) -> tuple[str, str]:
    ensure_parent_dirs()
    ordinary_step27_global_freeze_guard()

    all_manifests = state["all_manifests"]
    if len(all_manifests) != EXPECTED_FITS:
        raise RuntimeError(
            "Cannot freeze Step-27I without 80 verified fit manifests."
        )

    # Guard against a stale/changed final-test lock immediately before write.
    upstream = state["upstream"]
    if (
        sha256_file(Path(upstream["global_lock_path"]))
        != upstream["global_lock_sha256"]
    ):
        raise RuntimeError(
            "Global final-test lock changed before Step-27I freeze."
        )

    fit_df = build_fit_register(all_manifests)
    completion_df = build_completion_register(all_manifests)

    s27.atomic_csv(FIT_REGISTER_CSV, fit_df)
    s27.atomic_csv(COMPLETION_HASH_CSV, completion_df)

    semantic = build_freeze_semantic(
        state,
        fit_df,
        completion_df,
    )
    fingerprint = canonical_hash(semantic)

    freeze_payload = {
        **semantic,
        "generated_at_utc": s27.utc_now(),
        "stage1_execution_amendment_aware_fingerprint_sha256":
            fingerprint,
    }
    s27.atomic_json(FREEZE_JSON, freeze_payload)

    summary_lines = [
        STEP_NAME,
        "=" * 112,
        "Status: PASS",
        f"Completed development fits: {EXPECTED_FITS}/{EXPECTED_FITS}",
        f"Models: {EXPECTED_MODELS}",
        f"Candidates: {EXPECTED_CANDIDATES}",
        f"Folds: {list(s27.FOLDS)}",
        f"Seed: {s27.SEED}",
        "",
        "TIMESNET_B protocol amendment:",
        f"  physical microbatch: {TIMESNET_B_OLD_MICROBATCH} -> "
        f"{TIMESNET_B_NEW_MICROBATCH}",
        f"  logical batch: {LOGICAL_BATCH} (UNCHANGED)",
        "  classification: pre-performance semantic protocol amendment",
        "  reason: TimesNet FFT period selection is physical-batch-dependent",
        f"  Step-27F SHA-256: {EXPECTED_STEP27F_AMENDMENT_SHA256}",
        "",
        "Configuration selection performed: False",
        "Threshold optimization performed: False",
        "Best-seed selection performed: False",
        f"Step-10 final-test seal: {upstream['test_seal_status']}",
        f"Global final-test state: {upstream['global_final_test_state']}",
        "",
        "Step 27I freezes fit-level development results only.",
        "A/B configuration selection must occur in a later development-only step.",
        "The ordinary Step-27 global freeze was intentionally not used.",
    ]
    s27.atomic_text(
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
        s27.INPUT_REGISTER_JSON,
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
        "frozen_at_utc": s27.utc_now(),
        "configuration_fingerprint_sha256":
            EXPECTED_CONFIGURATION_FINGERPRINT,
        "stage1_execution_amendment_aware_fingerprint_sha256":
            fingerprint,
        "step27_execution_source_fingerprint_sha256":
            state["source_fp"],
        "step27_development_input_fingerprint_sha256":
            state["input_fp"],
        "step27f_protocol_amendment_sha256":
            EXPECTED_STEP27F_AMENDMENT_SHA256,
        "step27g_source_fingerprint_sha256":
            EXPECTED_STEP27G_SOURCE_FINGERPRINT,
        "step27h_source_fingerprint_sha256":
            EXPECTED_STEP27H_SOURCE_FINGERPRINT,
        "completed_fits": EXPECTED_FITS,
        "configuration_selection_performed": False,
        "threshold_optimization_performed": False,
        "best_seed_selection_performed": False,
        "final_test_accessed": False,
        "artifacts": output_records,
    }
    s27.atomic_json(MANIFEST_JSON, manifest)

    manifest_sha = sha256_file(MANIFEST_JSON)
    s27.atomic_text(
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

    s27.atomic_csv(
        OUTPUT_HASH_CSV,
        pd.DataFrame(output_hash_rows),
    )

    # Final lock guard after all output writes.
    if (
        sha256_file(Path(upstream["global_lock_path"]))
        != upstream["global_lock_sha256"]
    ):
        raise RuntimeError(
            "Global final-test lock changed during Step-27I freeze."
        )

    return fingerprint, manifest_sha


# =============================================================================
# 9. VERIFY AN EXISTING STEP-27I FREEZE
# =============================================================================

def verify_existing_final_freeze() -> tuple[str, str]:
    required = [
        FIT_REGISTER_CSV,
        COMPLETION_HASH_CSV,
        OUTPUT_HASH_CSV,
        FREEZE_JSON,
        MANIFEST_JSON,
        MANIFEST_SHA,
        SUMMARY_TXT,
        SOURCE_FREEZE_JSON,
        SOURCE_FROZEN_SCRIPT,
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(
            "Step-27I final freeze is incomplete. Missing:\n  - "
            + "\n  - ".join(str(p) for p in missing)
        )

    freeze = load_json(FREEZE_JSON)
    if (
        freeze.get("status")
        != "STAGE1_80_FIT_EXECUTION_FROZEN_AMENDMENT_AWARE"
    ):
        raise RuntimeError("Step-27I freeze status drift.")

    stored_fp = freeze.get(
        "stage1_execution_amendment_aware_fingerprint_sha256"
    )
    semantic = dict(freeze)
    semantic.pop("generated_at_utc", None)
    semantic.pop(
        "stage1_execution_amendment_aware_fingerprint_sha256",
        None,
    )
    if canonical_hash(semantic) != stored_fp:
        raise RuntimeError("Step-27I freeze fingerprint no longer recomputes.")

    if int(freeze["execution_scope"]["completed_fits"]) != EXPECTED_FITS:
        raise RuntimeError("Step-27I completed-fit count drift.")

    if freeze["protection_state"]["final_test_data_opened"]:
        raise RuntimeError("Step-27I freeze reports final-test access.")

    amendment = freeze["protocol"]["timesnet_b_protocol_amendment"]
    if int(amendment["physical_microbatch_old"]) != 2:
        raise RuntimeError("Step-27I amendment old microbatch drift.")
    if int(amendment["physical_microbatch_new"]) != 64:
        raise RuntimeError("Step-27I amendment new microbatch drift.")
    if int(amendment["logical_batch_unchanged"]) != 256:
        raise RuntimeError("Step-27I logical batch drift.")

    fit_df = pd.read_csv(FIT_REGISTER_CSV)
    if len(fit_df) != EXPECTED_FITS:
        raise RuntimeError("Existing Step-27I fit register is not 80 rows.")

    completion_df = pd.read_csv(COMPLETION_HASH_CSV)
    if len(completion_df) != EXPECTED_FITS:
        raise RuntimeError(
            "Existing Step-27I completion register is not 80 rows."
        )

    manifest = load_json(MANIFEST_JSON)
    if int(manifest.get("completed_fits", -1)) != EXPECTED_FITS:
        raise RuntimeError("Existing Step-27I manifest fit-count drift.")
    if bool(manifest.get("final_test_accessed")):
        raise RuntimeError("Existing Step-27I manifest reports final-test access.")

    manifest_sha = sha256_file(MANIFEST_JSON)
    expected_sha_line = f"{manifest_sha}  {MANIFEST_JSON.name}"
    observed_sha_line = MANIFEST_SHA.read_text(
        encoding="utf-8"
    ).strip()
    if observed_sha_line != expected_sha_line:
        raise RuntimeError("Step-27I manifest SHA sidecar drift.")

    hashes = pd.read_csv(OUTPUT_HASH_CSV)
    for _, row in hashes.iterrows():
        path = Path(row["path"])
        if not path.is_file():
            raise RuntimeError(
                f"Step-27I hashed output missing: {path}"
            )
        if sha256_file(path) != str(row["sha256"]):
            raise RuntimeError(
                f"Step-27I output hash drift: {path}"
            )

    return str(stored_fp), manifest_sha


# =============================================================================
# 10. PREPARE / FINAL ENTRYPOINTS
# =============================================================================

def prepare_only() -> int:
    print("=" * 140)
    print(STEP_NAME)
    print("=" * 140)

    amendment = verify_amendment_chain()
    source_fp_i, source_sha_i = ensure_source_freeze()
    ordinary_step27_global_freeze_guard()
    state = verify_complete_corpus()

    print("[PASS] Step-27F protocol amendment verified.")
    print(
        f"[PASS] Step-27F amendment SHA-256     : "
        f"{EXPECTED_STEP27F_AMENDMENT_SHA256}"
    )
    print(
        f"[PASS] Step-27G source fingerprint     : "
        f"{EXPECTED_STEP27G_SOURCE_FINGERPRINT}"
    )
    print(
        f"[PASS] Step-27H source fingerprint     : "
        f"{EXPECTED_STEP27H_SOURCE_FINGERPRINT}"
    )
    print(
        f"[PASS] Step-27I source SHA-256         : "
        f"{source_sha_i}"
    )
    print(
        f"[PASS] Step-27I source fingerprint     : "
        f"{source_fp_i}"
    )
    print(
        f"[PASS] Original Step-27 source          : "
        f"{state['source_fp']}"
    )
    print(
        f"[PASS] Development input fingerprint    : "
        f"{state['input_fp']}"
    )
    print(
        f"[PASS] Amendment-aware verified fits    : "
        f"{len(state['all_manifests'])}/{EXPECTED_FITS}"
    )
    print(
        f"[PASS] TIMESNET_B physical microbatch   : "
        f"{TIMESNET_B_OLD_MICROBATCH} -> {TIMESNET_B_NEW_MICROBATCH}"
    )
    print(
        f"[PASS] TIMESNET_B logical batch         : "
        f"{LOGICAL_BATCH} (UNCHANGED)"
    )
    print("[PASS] New training performed            : False")
    print("[PASS] Configuration selection           : NOT PERFORMED")
    print("[PASS] Threshold optimization            : NOT PERFORMED")
    print("[PASS] Best-seed selection               : NOT PERFORMED")
    print("[PASS] Final-test data opened             : False")
    print(
        f"[PASS] Global final-test state           : "
        f"{state['upstream']['global_final_test_state']}"
    )

    print()
    print("=" * 140)
    print("STEP 27I PREPARE STATUS: PASS")
    print("=" * 140)
    print(f"Verified Stage-1 fits          : {EXPECTED_FITS}/{EXPECTED_FITS}")
    print("Ordinary Step-27 global freeze : NOT USED")
    print("Step-27I final freeze written  : False")
    print("Final-test state               : LOCKED_NOT_OPENED")
    print("=" * 140)
    print(
        "NEXT: audit this output. Then run the same script without "
        "--prepare-only to create the amendment-aware final Stage-1 freeze."
    )

    _ = amendment
    return 0


def final_freeze() -> int:
    print("=" * 140)
    print(STEP_NAME)
    print("=" * 140)

    verify_amendment_chain()
    source_fp_i, source_sha_i = ensure_source_freeze()

    # Idempotent verify if a complete Step-27I freeze already exists.
    if MANIFEST_JSON.exists():
        fingerprint, manifest_sha = verify_existing_final_freeze()
        print("[PASS] Existing Step-27I freeze verified; no files rewritten.")
        print(
            f"       Step-27I source SHA-256      : {source_sha_i}"
        )
        print(
            f"       Step-27I source fingerprint  : {source_fp_i}"
        )
        print(
            f"       Stage-1 freeze fingerprint   : {fingerprint}"
        )
        print(
            f"       Manifest SHA-256             : {manifest_sha}"
        )
        print("       Completed fits                : 80/80")
        print("       Configuration selection       : NOT PERFORMED")
        print("       Final-test data opened         : False")
        return 0

    ordinary_step27_global_freeze_guard()
    state = verify_complete_corpus()

    print("[1/4] Amendment/provenance chain verified.")
    print("[2/4] All 80 fit manifests and canonical artifacts hash-verified.")
    print(
        "[3/4] Final-test lock unchanged; configuration selection "
        "has not been performed."
    )
    print("[4/4] Writing amendment-aware Stage-1 freeze...")

    fingerprint, manifest_sha = freeze_complete_stage(state)

    # Verify what was just written.
    verified_fp, verified_manifest_sha = verify_existing_final_freeze()
    if verified_fp != fingerprint:
        raise RuntimeError(
            "Post-write Step-27I fingerprint verification mismatch."
        )
    if verified_manifest_sha != manifest_sha:
        raise RuntimeError(
            "Post-write Step-27I manifest SHA verification mismatch."
        )

    print()
    print("=" * 140)
    print("STEP 27I — AMENDMENT-AWARE STAGE-1 FREEZE COMPLETE")
    print("=" * 140)
    print(f"Verified completed fits          : {EXPECTED_FITS}/{EXPECTED_FITS}")
    print(f"Models                           : {EXPECTED_MODELS}")
    print(f"Candidate configurations         : {EXPECTED_CANDIDATES}")
    print(f"Chronological folds              : {list(s27.FOLDS)}")
    print(f"Screening seed                   : {s27.SEED}")
    print(
        f"TIMESNET_B physical microbatch   : "
        f"{TIMESNET_B_OLD_MICROBATCH} -> {TIMESNET_B_NEW_MICROBATCH}"
    )
    print(f"TIMESNET_B logical batch         : {LOGICAL_BATCH} (UNCHANGED)")
    print(
        "TimesNet batch-dependent FFT     : EXPLICITLY RECORDED IN FREEZE"
    )
    print("Configuration selection          : NOT PERFORMED")
    print("Threshold optimization           : NOT PERFORMED")
    print("Best-seed selection              : NOT PERFORMED")
    print(f"Stage-1 freeze fingerprint       : {fingerprint}")
    print(f"Step-27I manifest SHA-256        : {manifest_sha}")
    print(
        f"Step-10 final-test seal          : "
        f"{state['upstream']['test_seal_status']}"
    )
    print(
        f"Global final-test state          : "
        f"{state['upstream']['global_final_test_state']}"
    )
    print("=" * 140)
    print(
        "PASS: All 80 Stage-1 development fits are complete, "
        "hash-verified, and frozen under amendment-aware provenance."
    )
    print(
        "PASS: The next step may perform DEVELOPMENT-ONLY A/B "
        "configuration selection according to the pre-frozen selection rule."
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Verify all 80 fits and freeze Step-27I source, but do not "
            "write the final amendment-aware Stage-1 freeze."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare_only:
        return prepare_only()
    return final_freeze()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nStep 27I interrupted by user. No training is performed by "
            "this step; rerun safely after checking output artifacts."
        )
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print()
        print("=" * 140)
        print("STEP 27I FAILED")
        print("=" * 140)
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        print("=" * 140)
        raise SystemExit(1)
