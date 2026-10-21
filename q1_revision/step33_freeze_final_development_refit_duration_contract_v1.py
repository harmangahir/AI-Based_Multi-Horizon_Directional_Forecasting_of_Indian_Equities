#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 33 — FINAL-DEVELOPMENT REFIT DURATION CONTRACT FREEZE
===========================================================

Purpose
-------
Resolve a pre-final-test protocol underspecification:

The frozen protocol specifies:
  * selected architecture/configuration;
  * five fixed seeds 42..46;
  * Step-14 final-development arrays;
  * frozen development-OOF thresholds;
  * no post-OOF hyperparameter changes;
  * no final-test tuning;

but does not explicitly specify the training-duration rule for the
final-development refit, where no validation partition remains available.

Frozen completion rule
----------------------
For each selected architecture and fixed seed:

    final_refit_epochs(model, seed)
        = median of the five rolling-origin best_epoch values
          for that same selected configuration and same fixed seed.

Since there are exactly five folds, the median is always an observed
integer epoch and requires no rounding.

This rule:
  * is architecture-agnostic;
  * is seed-preserving;
  * does not use BA/AUC/MCC/PR-AUC/test results;
  * does not select/promote any seed;
  * does not use final-test data;
  * does not alter model/configuration/LR/optimizer/loss/batch contract;
  * does not alter already-frozen OOF thresholds.

Final-development execution contract
------------------------------------
  * source arrays:
      Step-14 final_development arrays
  * n:
      42,360 common multitask sequences
  * seeds:
      42,43,44,45,46
  * models:
      8 selected configurations
  * total final refits:
      40
  * validation:
      NONE
  * early stopping:
      NONE during final refit
  * checkpoint:
      final scheduled epoch
  * logical batch:
      256
  * physical microbatch:
      same selected-config schedule already frozen
  * final-development shuffle:
      SeedSequence([seed, 0, epoch])
      where 0 is the frozen final-development partition token
  * thresholds:
      not used for fitting
  * final test:
      LOCKED_NOT_OPENED
"""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent

ROOT = (
    HERE.parent
    / "stock_q1_revision_project"
    / "experiments"
    / "q1_revision_v3_clean_2016_2026"
).resolve()

AUDIT_DIR = ROOT / "audits"
MANIFEST_DIR = ROOT / "manifests"

STEP30_MANIFEST = (
    MANIFEST_DIR
    / "30_stage2_global_final_freeze_manifest.json"
)

STEP32_MANIFEST = (
    MANIFEST_DIR
    / "32_development_threshold_final_freeze_manifest.json"
)

STEP14_MANIFEST = (
    MANIFEST_DIR
    / "14_model_array_manifest.json"
)

OUT_SCHEDULE = (
    AUDIT_DIR
    / "33_final_development_refit_epoch_schedule.csv"
)

OUT_SOURCE_EPOCH_AUDIT = (
    AUDIT_DIR
    / "33_refit_duration_source_best_epochs.csv"
)

OUT_MANIFEST = (
    MANIFEST_DIR
    / "33_final_development_refit_duration_contract.json"
)

FROZEN_SOURCE = (
    MANIFEST_DIR
    / "step33_freeze_final_development_refit_duration_contract_v1_frozen.py"
)


# =============================================================================
# AUTHORITATIVE UPSTREAM HASHES
# =============================================================================

EXPECTED_STEP30_MANIFEST_SHA256 = (
    "8216ac6947a2c2c2741758f76d9c38f085f59ac4158f9fcdc58effb18c1ab420"
)

EXPECTED_STEP30_GLOBAL_FINGERPRINT = (
    "328c77bac04ad61dabb9fe50f9b481389a94b95709cefe3c00fc680e7c807bed"
)

EXPECTED_STEP32_MANIFEST_SHA256 = (
    "474b77ac670bbb7f61870372cdb98a6d7ac3ecaa4db8864ceb096fe9b26c39f0"
)

EXPECTED_STEP32_GLOBAL_FINGERPRINT = (
    "bfc00e361e883fb2066e4254c67f73ac3235d2abafde32e3ad989b59e1f8e92d"
)


# =============================================================================
# FROZEN REFIT CONTRACT
# =============================================================================

SELECTED = {
    "DLINEAR": "DLINEAR_B",
    "NLINEAR": "NLINEAR_B",
    "PATCHTST": "PATCHTST_B",
    "TIMESNET": "TIMESNET_B",
    "ITRANSFORMER": "ITRANSFORMER_A",
    "FREDFORMER": "FREDFORMER_B",
    "FRNET": "FRNET_A",
    "MLF": "MLF_A",
}

PHYSICAL_MICROBATCH = {
    "DLINEAR": 128,
    "NLINEAR": 128,
    "PATCHTST": 64,
    "TIMESNET": 64,
    "ITRANSFORMER": 128,
    "FREDFORMER": 16,
    "FRNET": 64,
    "MLF": 32,
}

SEEDS = [42, 43, 44, 45, 46]
FOLDS = [1, 2, 3, 4, 5]

EXPECTED_FINAL_DEVELOPMENT_N = 42360
LOGICAL_BATCH_SIZE = 256
FINAL_DEVELOPMENT_PARTITION_TOKEN = 0

EXPECTED_SCHEDULE_ROWS = 8 * 5
EXPECTED_SOURCE_ROWS = 8 * 5 * 5

TIMESNET_B_AMENDMENT_SHA256 = (
    "04c30b59b8bef7327b44e1764240ead258edbba64f278e31ca29249141dfdce8"
)


# =============================================================================
# HELPERS
# =============================================================================

def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if not path.is_file():
        fail(f"Missing file: {path}")

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def canonical_json_sha256(obj) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"Missing JSON: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        fail(f"Expected JSON object: {path}")

    return obj


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        fail(
            f"{label}: expected {expected!r}, "
            f"got {actual!r}"
        )


def require_false(value, label: str) -> None:
    if value is not False:
        fail(
            f"{label}: expected False, got {value!r}"
        )


def freeze_source() -> tuple[str, str]:
    src = Path(__file__).resolve()
    source_sha = sha256_file(src)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    if FROZEN_SOURCE.exists():
        frozen_sha = sha256_file(FROZEN_SOURCE)

        if frozen_sha != source_sha:
            fail(
                "Step-33 frozen source already exists but differs "
                "from executable source.\n"
                f"Frozen : {frozen_sha}\n"
                f"Current: {source_sha}"
            )
    else:
        shutil.copy2(src, FROZEN_SOURCE)

    frozen_sha = sha256_file(FROZEN_SOURCE)

    require_equal(
        frozen_sha,
        source_sha,
        "Step33 frozen source SHA256",
    )

    return source_sha, frozen_sha


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 140)
    print(
        "STEP 33 — FINAL-DEVELOPMENT REFIT "
        "DURATION CONTRACT FREEZE"
    )
    print("=" * 140)

    print(f"Experiment root      : {ROOT}")
    print(
        "Duration rule        : median best_epoch across "
        "5 rolling folds for same model + same fixed seed"
    )
    print("Performance metrics  : NOT USED")
    print("Best-seed selection  : NONE")
    print("Final test            : LOCKED_NOT_OPENED")
    print("-" * 140)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_MANIFEST.exists():
        print("Step-33 final contract already exists:")
        print(OUT_MANIFEST)
        print("No artifacts modified.")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # 1. Verify Step-30 robustness freeze.
    # -------------------------------------------------------------------------

    step30_sha = sha256_file(STEP30_MANIFEST)

    require_equal(
        step30_sha,
        EXPECTED_STEP30_MANIFEST_SHA256,
        "Step30 manifest SHA256",
    )

    step30 = load_json(STEP30_MANIFEST)

    require_equal(
        step30.get("status"),
        "STAGE2_GLOBAL_ROBUSTNESS_FINAL_FREEZE_COMPLETE",
        "Step30 status",
    )

    require_equal(
        step30.get("global_stage2_freeze_fingerprint"),
        EXPECTED_STEP30_GLOBAL_FINGERPRINT,
        "Step30 global fingerprint",
    )

    p30 = step30.get("protection_state", {})

    require_false(
        p30.get("final_test_data_opened"),
        "Step30 final_test_data_opened",
    )

    logical_inventory_meta = (
        step30["outputs"]["logical_fit_inventory"]
    )

    logical_inventory_path = Path(
        logical_inventory_meta["path"]
    ).expanduser().resolve()

    require_equal(
        sha256_file(logical_inventory_path),
        logical_inventory_meta["sha256"],
        "Step30 logical inventory SHA256",
    )

    inventory = pd.read_csv(logical_inventory_path)

    require_equal(
        len(inventory),
        200,
        "Step30 logical inventory rows",
    )

    print("[1] STEP-30 ROBUSTNESS FREEZE")
    print(f"Logical fits : {len(inventory)}/200")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 2. Verify Step-32 threshold freeze.
    # -------------------------------------------------------------------------

    step32_sha = sha256_file(STEP32_MANIFEST)

    require_equal(
        step32_sha,
        EXPECTED_STEP32_MANIFEST_SHA256,
        "Step32 manifest SHA256",
    )

    step32 = load_json(STEP32_MANIFEST)

    require_equal(
        step32.get("status"),
        "STEP32_DEVELOPMENT_THRESHOLD_FINAL_FREEZE_COMPLETE",
        "Step32 status",
    )

    require_equal(
        step32.get("global_step32_fingerprint"),
        EXPECTED_STEP32_GLOBAL_FINGERPRINT,
        "Step32 global fingerprint",
    )

    p32 = step32.get("protection_state", {})

    require_false(
        p32.get("architecture_selection_performed"),
        "Step32 architecture selection",
    )

    require_false(
        p32.get("best_seed_selection_performed"),
        "Step32 best-seed selection",
    )

    require_false(
        p32.get("best_seed_promotion_performed"),
        "Step32 best-seed promotion",
    )

    require_false(
        p32.get("final_test_data_opened"),
        "Step32 final-test access",
    )

    require_equal(
        p32.get("threshold_optimization_performed"),
        True,
        "Step32 threshold optimization state",
    )

    require_equal(
        p32.get("threshold_optimization_scope"),
        "DEVELOPMENT_ONLY_POOLED_OOF",
        "Step32 threshold scope",
    )

    print("[2] STEP-32 THRESHOLD FREEZE")
    print("Frozen thresholds : 24/24")
    print("Final test        : LOCKED_NOT_OPENED")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 3. Verify Step-14 final-development arrays.
    # -------------------------------------------------------------------------

    step14 = load_json(STEP14_MANIFEST)

    state14 = step14.get("pipeline_state", {})
    test14 = step14.get("test_protection", {})

    require_equal(
        state14.get("final_development_refit_arrays_created"),
        True,
        "Step14 final-development arrays",
    )

    require_equal(
        test14.get("final_test_arrays_created"),
        False,
        "Step14 final-test arrays created",
    )

    require_equal(
        test14.get("global_final_test_state"),
        "LOCKED_NOT_OPENED",
        "Step14 final-test state",
    )

    final_dir = (
        ROOT
        / "feature_data"
        / "step14_model_arrays"
        / "final_development"
    )

    final_x = (
        final_dir
        / "14_final_development_X_full.npy"
    )

    final_y = (
        final_dir
        / "14_final_development_y_direction.npy"
    )

    x = np.load(
        final_x,
        mmap_mode="r",
        allow_pickle=False,
    )

    y = np.load(
        final_y,
        mmap_mode="r",
        allow_pickle=False,
    )

    require_equal(
        x.shape,
        (EXPECTED_FINAL_DEVELOPMENT_N, 60, 63),
        "Final-development X shape",
    )

    require_equal(
        y.shape,
        (EXPECTED_FINAL_DEVELOPMENT_N, 3),
        "Final-development y shape",
    )

    if not set(np.unique(y).tolist()).issubset({0, 1}):
        fail("Final-development directional labels are not binary.")

    final_x_sha = sha256_file(final_x)
    final_y_sha = sha256_file(final_y)

    print("[3] FINAL-DEVELOPMENT ARRAYS")
    print(
        f"Sequences        : {EXPECTED_FINAL_DEVELOPMENT_N:,}"
    )
    print("X_full           : (42360, 60, 63)")
    print("y_direction      : (42360, 3)")
    print("Final test arrays: NOT CREATED")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 4. Freeze Step-33 source before deriving schedule.
    # -------------------------------------------------------------------------

    source_sha, frozen_source_sha = freeze_source()

    print("[4] STEP-33 SOURCE FREEZE")
    print(f"Executable SHA256 : {source_sha}")
    print(f"Frozen SHA256     : {frozen_source_sha}")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 5. Read ONLY training-duration fields from the 200 frozen manifests.
    # -------------------------------------------------------------------------

    required_cols = {
        "model_id",
        "config_id",
        "seed",
        "fold",
        "fit_manifest_path",
        "fit_manifest_sha256",
    }

    missing = required_cols - set(inventory.columns)

    if missing:
        fail(
            "Step30 logical inventory missing columns: "
            f"{sorted(missing)}"
        )

    source_rows = []

    for model_id, config_id in SELECTED.items():

        for seed in SEEDS:

            subset = inventory.loc[
                (inventory["model_id"].astype(str) == model_id)
                & (inventory["config_id"].astype(str) == config_id)
                & (inventory["seed"].astype(int) == seed)
            ].copy()

            subset = subset.sort_values("fold")

            require_equal(
                len(subset),
                5,
                f"{model_id}/seed{seed}: fold count",
            )

            require_equal(
                subset["fold"].astype(int).tolist(),
                FOLDS,
                f"{model_id}/seed{seed}: folds",
            )

            for r in subset.itertuples(index=False):

                manifest_path = Path(
                    str(r.fit_manifest_path)
                ).expanduser().resolve()

                require_equal(
                    sha256_file(manifest_path),
                    str(r.fit_manifest_sha256),
                    (
                        f"{model_id}/seed{seed}/fold{int(r.fold)} "
                        "manifest SHA256"
                    ),
                )

                m = load_json(manifest_path)

                best_epoch = int(
                    m.get("best_epoch", -1)
                )

                epochs_run = int(
                    m.get("epochs_run", -1)
                )

                if best_epoch < 1:
                    fail(
                        f"{model_id}/seed{seed}/fold{r.fold}: "
                        f"invalid best_epoch={best_epoch}"
                    )

                if epochs_run < best_epoch:
                    fail(
                        f"{model_id}/seed{seed}/fold{r.fold}: "
                        "epochs_run < best_epoch"
                    )

                source_rows.append(
                    {
                        "model_id": model_id,
                        "config_id": config_id,
                        "seed": seed,
                        "fold": int(r.fold),
                        "best_epoch": best_epoch,
                        "epochs_run": epochs_run,
                        "fit_manifest_path":
                            str(manifest_path),
                        "fit_manifest_sha256":
                            str(r.fit_manifest_sha256),
                        "performance_metric_read":
                            False,
                    }
                )

    source_df = pd.DataFrame(source_rows).sort_values(
        [
            "model_id",
            "config_id",
            "seed",
            "fold",
        ]
    ).reset_index(drop=True)

    require_equal(
        len(source_df),
        EXPECTED_SOURCE_ROWS,
        "Source best-epoch rows",
    )

    source_df.to_csv(
        OUT_SOURCE_EPOCH_AUDIT,
        index=False,
    )

    source_sha_csv = sha256_file(
        OUT_SOURCE_EPOCH_AUDIT
    )

    print("[5] TRAINING-DURATION SOURCE AUDIT")
    print(
        f"Frozen fit manifests read : {len(source_df)}/200"
    )
    print("Fields used                : best_epoch only")
    print("Performance metrics used   : NONE")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 6. Derive fixed final-refit epoch schedule.
    # -------------------------------------------------------------------------

    schedule_rows = []

    for model_id, config_id in SELECTED.items():

        for seed in SEEDS:

            g = source_df.loc[
                (source_df["model_id"] == model_id)
                & (source_df["config_id"] == config_id)
                & (source_df["seed"].astype(int) == seed)
            ].sort_values("fold")

            require_equal(
                len(g),
                5,
                f"{model_id}/seed{seed}: epoch source count",
            )

            best_epochs = (
                g["best_epoch"]
                .astype(int)
                .tolist()
            )

            median_epoch = int(
                statistics.median(best_epochs)
            )

            if median_epoch not in best_epochs:
                fail(
                    "With five integer folds the median must be "
                    "an observed best epoch."
                )

            schedule_rows.append(
                {
                    "model_id": model_id,
                    "config_id": config_id,
                    "seed": seed,
                    "fold_best_epochs":
                        ",".join(
                            str(x)
                            for x in best_epochs
                        ),
                    "final_refit_epochs":
                        median_epoch,
                    "duration_rule":
                        "MEDIAN_OF_5_SAME_SEED_ROLLING_FOLD_BEST_EPOCHS",
                    "validation_during_final_refit":
                        False,
                    "early_stopping_during_final_refit":
                        False,
                    "checkpoint_policy":
                        "FINAL_SCHEDULED_EPOCH",
                    "final_development_n":
                        EXPECTED_FINAL_DEVELOPMENT_N,
                    "logical_batch_size":
                        LOGICAL_BATCH_SIZE,
                    "physical_microbatch_size":
                        PHYSICAL_MICROBATCH[model_id],
                    "shuffle_seed_contract":
                        "SeedSequence([seed,0,epoch])",
                    "final_development_partition_token":
                        FINAL_DEVELOPMENT_PARTITION_TOKEN,
                    "best_seed_selection":
                        False,
                    "final_test_used":
                        False,
                }
            )

    schedule_df = pd.DataFrame(
        schedule_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "seed",
        ]
    ).reset_index(drop=True)

    require_equal(
        len(schedule_df),
        EXPECTED_SCHEDULE_ROWS,
        "Final-refit schedule rows",
    )

    schedule_df.to_csv(
        OUT_SCHEDULE,
        index=False,
    )

    schedule_sha = sha256_file(
        OUT_SCHEDULE
    )

    schedule_fp = canonical_json_sha256(
        schedule_df.to_dict(
            orient="records"
        )
    )

    print("[6] FROZEN FINAL-REFIT EPOCH SCHEDULE")
    print()

    for r in schedule_df.itertuples(index=False):
        print(
            f"{r.model_id:14s} "
            f"seed={int(r.seed)} "
            f"fold-best=[{r.fold_best_epochs}] "
            f"-> refit_epochs={int(r.final_refit_epochs)}"
        )

    print("-" * 140)
    print(f"Schedule rows        : {len(schedule_df)}/40")
    print(f"Schedule SHA256      : {schedule_sha}")
    print(f"Schedule fingerprint : {schedule_fp}")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 7. Freeze final-development execution contract.
    # -------------------------------------------------------------------------

    logical_batches = (
        EXPECTED_FINAL_DEVELOPMENT_N
        // LOGICAL_BATCH_SIZE
    )

    logical_remainder = (
        EXPECTED_FINAL_DEVELOPMENT_N
        % LOGICAL_BATCH_SIZE
    )

    freeze_basis = {
        "step30_manifest_sha256":
            step30_sha,
        "step30_global_fingerprint":
            EXPECTED_STEP30_GLOBAL_FINGERPRINT,
        "step32_manifest_sha256":
            step32_sha,
        "step32_global_fingerprint":
            EXPECTED_STEP32_GLOBAL_FINGERPRINT,
        "step33_source_sha256":
            source_sha,
        "step33_frozen_source_sha256":
            frozen_source_sha,
        "duration_rule":
            "MEDIAN_OF_5_SAME_SEED_ROLLING_FOLD_BEST_EPOCHS",
        "selected_configs":
            SELECTED,
        "seeds":
            SEEDS,
        "final_development_n":
            EXPECTED_FINAL_DEVELOPMENT_N,
        "logical_batch_size":
            LOGICAL_BATCH_SIZE,
        "physical_microbatch_by_model":
            PHYSICAL_MICROBATCH,
        "shuffle_seed_contract":
            "SeedSequence([seed,0,epoch])",
        "validation_during_refit":
            False,
        "early_stopping_during_refit":
            False,
        "checkpoint_policy":
            "FINAL_SCHEDULED_EPOCH",
        "schedule_sha256":
            schedule_sha,
        "schedule_fingerprint":
            schedule_fp,
        "final_test_data_opened":
            False,
    }

    global_step33_fingerprint = (
        canonical_json_sha256(
            freeze_basis
        )
    )

    manifest = {
        "status":
            "STEP33_FINAL_DEVELOPMENT_REFIT_DURATION_CONTRACT_FROZEN",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "experiment_root":
            str(ROOT),

        "protocol_gap_resolution": {
            "gap":
                "Final-development refit training duration "
                "was not explicitly specified in Step-16.",

            "classification":
                "PRE_FINAL_TEST_PROTOCOL_COMPLETION_AMENDMENT",

            "rule":
                "For each selected architecture and fixed seed, "
                "use the median of the five rolling-origin "
                "best_epoch values for the same architecture, "
                "configuration and seed.",

            "performance_metrics_used_to_define_rule":
                False,

            "final_test_used":
                False,
        },

        "upstream": {
            "step30_manifest_sha256":
                step30_sha,
            "step30_global_fingerprint":
                EXPECTED_STEP30_GLOBAL_FINGERPRINT,
            "step32_manifest_sha256":
                step32_sha,
            "step32_global_fingerprint":
                EXPECTED_STEP32_GLOBAL_FINGERPRINT,
        },

        "final_development_data": {
            "source":
                "Step-14 final-development common-multitask arrays",
            "n":
                EXPECTED_FINAL_DEVELOPMENT_N,
            "x_full_path":
                str(final_x.resolve()),
            "x_full_sha256":
                final_x_sha,
            "y_direction_path":
                str(final_y.resolve()),
            "y_direction_sha256":
                final_y_sha,
            "lookback":
                60,
            "predictors":
                63,
            "horizons":
                [1, 3, 5],
        },

        "refit_contract": {
            "selected_configurations":
                SELECTED,
            "seeds":
                SEEDS,
            "total_seed_models":
                40,
            "duration_rule":
                "MEDIAN_OF_5_SAME_SEED_ROLLING_FOLD_BEST_EPOCHS",
            "validation_during_refit":
                False,
            "early_stopping_during_refit":
                False,
            "checkpoint_policy":
                "FINAL_SCHEDULED_EPOCH",
            "logical_batch_size":
                LOGICAL_BATCH_SIZE,
            "physical_microbatch_by_model":
                PHYSICAL_MICROBATCH,
            "full_logical_batches_per_epoch":
                logical_batches,
            "partial_logical_batch_size":
                logical_remainder,
            "shuffle_seed_contract":
                "numpy.random.SeedSequence([seed,0,epoch])",
            "final_development_partition_token":
                FINAL_DEVELOPMENT_PARTITION_TOKEN,
            "same_optimizer_lr_weight_decay_loss_and_gradient_contract":
                True,
            "timesnet_b_protocol_amendment_sha256":
                TIMESNET_B_AMENDMENT_SHA256,
            "thresholds_used_during_training":
                False,
            "probability_ensemble_after_refit":
                "ARITHMETIC_MEAN_ACROSS_FIXED_SEEDS_42_46",
        },

        "protection_state": {
            "architecture_selection_performed":
                False,
            "best_seed_selection_performed":
                False,
            "best_seed_promotion_performed":
                False,
            "post_threshold_tuning_performed":
                False,
            "final_test_data_opened":
                False,
            "final_test_metric_computed":
                False,
            "final_test_state":
                "LOCKED_NOT_OPENED",
        },

        "outputs": {
            "source_best_epoch_audit": {
                "path":
                    str(
                        OUT_SOURCE_EPOCH_AUDIT.resolve()
                    ),
                "rows":
                    200,
                "sha256":
                    source_sha_csv,
            },
            "final_refit_epoch_schedule": {
                "path":
                    str(
                        OUT_SCHEDULE.resolve()
                    ),
                "rows":
                    40,
                "sha256":
                    schedule_sha,
                "content_fingerprint":
                    schedule_fp,
            },
            "frozen_source": {
                "path":
                    str(
                        FROZEN_SOURCE.resolve()
                    ),
                "sha256":
                    frozen_source_sha,
            },
        },

        "global_step33_fingerprint":
            global_step33_fingerprint,

        "next_protocol_gate": {
            "allowed_next_step":
                "FINAL_DEVELOPMENT_REFIT_EXECUTION",
            "total_refits":
                40,
            "final_test_allowed_now":
                False,
            "note":
                "Execute only the frozen 40-model final-development "
                "refit schedule. Freeze all refit artifacts before "
                "creating or opening final-test arrays.",
        },
    }

    with OUT_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    manifest_sha = sha256_file(
        OUT_MANIFEST
    )

    print("[7] STEP-33 FINAL CONTRACT FREEZE")
    print(
        f"Global Step-33 fingerprint : "
        f"{global_step33_fingerprint}"
    )
    print(
        f"Freeze manifest SHA256     : "
        f"{manifest_sha}"
    )
    print()

    print("=" * 140)
    print(
        "STEP 33 COMPLETE — FINAL-DEVELOPMENT "
        "REFIT DURATION CONTRACT FROZEN"
    )
    print("=" * 140)

    print("Final-development sequences : 42,360")
    print("Selected architectures      : 8/8")
    print("Fixed seeds                 : 5/5")
    print("Planned final refits        : 40")
    print(
        "Duration rule              : median same-seed "
        "best_epoch across 5 folds"
    )
    print("Validation during refit     : NONE")
    print("Early stopping during refit : NONE")
    print("Checkpoint                  : FINAL SCHEDULED EPOCH")
    print("Best-seed selection         : NOT PERFORMED")
    print("Final-test data opened      : False")
    print("Final-test state            : LOCKED_NOT_OPENED")
    print("-" * 140)
    print(
        "NEXT GATE: final-development refit execution."
    )
    print(
        "DO NOT create/open final-test arrays yet."
    )
    print("=" * 140)


if __name__ == "__main__":
    main()