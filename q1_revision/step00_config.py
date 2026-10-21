# =============================================================================
# STEP 00 — Q1 REVISION CLEAN EXPERIMENT CONFIGURATION
# =============================================================================
#
# Project:
# Artificial Intelligence-Based Multi-Horizon Directional Forecasting of
# Indian Equities: A Leakage-Controlled and Dependence-Aware Evaluation
#
# Purpose
# -------
# This module defines and freezes the high-level scientific, anti-leakage,
# execution, reproducibility, and output-protection contract for the clean
# Q1 revision experiment.
#
# IMPORTANT
# ---------
# 1. This is a NEW clean experiment. The old q1_revision_v2_sota_2016_2026
#    experiment is historical/audit material only and must not be resumed.
# 2. The final held-out test must not influence feature selection, model
#    selection, hyperparameter/configuration selection, threshold selection,
#    checkpoint selection, seed selection, or inferential-rule selection.
# 3. Development decisions use expanding-window rolling-origin validation.
# 4. Historical lookback may cross a development/validation/test boundary
#    when the observations precede the prediction reference date. This is
#    causal and is not leakage.
# 5. Future targets may never cross their evaluation boundary.
# 6. Stage-1 configuration screening uses threshold 0.50 only. OOF threshold
#    tuning occurs later, after configuration selection, on development OOF
#    probabilities only.
# 7. No best-seed promotion/cherry-picking is permitted.
# 8. The final test is opened once, only after a separate pretest registry
#    freezes every development decision and inferential rule.
# 9. GPU feasibility is not sufficient: memory AND throughput must be audited
#    for heavy configurations before Stage-1 training.
# 10. Scientific settings must never be changed in response to weak/null
#     validation or test performance.
#
# Expected location:
#     <repository>/q1_revision/step00_config.py
#
# Run:
#     python q1_revision/step00_config.py
#
# Re-run behavior:
#     - First run: creates and freezes the clean Step-00 configuration.
#     - Later runs: verify the existing freeze; they do NOT overwrite it.
#
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import sys

import numpy as np
import pandas as pd


# =============================================================================
# 1. PROJECT IDENTITY
# =============================================================================

PROJECT_NAME = "leakage_controlled_dependence_aware_indian_equity_forecasting"

WORKING_MANUSCRIPT_TITLE = (
    "Artificial Intelligence-Based Multi-Horizon Directional Forecasting "
    "of Indian Equities: A Leakage-Controlled and Dependence-Aware Evaluation"
)

METHODOLOGY_POSITIONING = (
    "Leakage-Controlled and Dependence-Aware Multi-Horizon Directional "
    "Forecasting of Indian Equities"
)

EXPERIMENT_TAG = "q1_revision_v3_clean_2016_2026"
EXPERIMENT_VERSION = "3.0"
CONFIG_SCHEMA_VERSION = "2.0"

# Historical reference only. Never use its outputs as clean-run evidence.
LEGACY_EXPERIMENT_TAG = "q1_revision_v2_sota_2016_2026"

TARGET_JOURNAL_CONTEXT = (
    "International Journal of Computational Intelligence Systems (Springer)"
)

SCIENTIFIC_POSITIONING = (
    "Methodological rigor, leakage control, auditability, modern comparison, "
    "dependence-aware inference, and defensible evaluation; not architectural "
    "dominance. Null or modest predictive results are acceptable."
)


# =============================================================================
# 2. REPOSITORY LOCATION
# =============================================================================
# The script is expected at:
#     <repository>/q1_revision/step00_config.py
# Repository paths are derived from __file__, not Path.cwd().
# =============================================================================

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent
REPOSITORY_ROOT_DIR = SCRIPT_DIR.parent

repository_markers = [REPOSITORY_ROOT_DIR / "README.md"]
missing_repository_markers = [p for p in repository_markers if not p.exists()]

if missing_repository_markers:
    print(
        "\nWARNING:\n"
        "The expected README.md was not found at the inferred repository root.\n"
        "This does not automatically invalidate the experiment, but confirm that\n"
        "step00_config.py is stored under <repository>/q1_revision/.\n\n"
        f"Inferred repository root: {REPOSITORY_ROOT_DIR}\n"
    )


# =============================================================================
# 3. ISOLATED CLEAN EXPERIMENT DIRECTORY TREE
# =============================================================================

PROJECT_ROOT_DIR = REPOSITORY_ROOT_DIR / "stock_q1_revision_project"
EXPERIMENTS_ROOT_DIR = PROJECT_ROOT_DIR / "experiments"
EXPERIMENT_ROOT_DIR = EXPERIMENTS_ROOT_DIR / EXPERIMENT_TAG

# Historical experiment path is recorded only for audit/display. Nothing in
# this Step-00 script reads or writes that experiment.
LEGACY_EXPERIMENT_ROOT_DIR = EXPERIMENTS_ROOT_DIR / LEGACY_EXPERIMENT_TAG

# In v3, raw downloads are experiment-local to eliminate accidental reuse of
# the abandoned v2 raw directory. Later steps may verify external cache files,
# but only by explicit hash-based logic.
RAW_DATA_DIR = EXPERIMENT_ROOT_DIR / "raw_data"
REFERENCE_DATA_DIR = EXPERIMENT_ROOT_DIR / "reference_data"
AUDIT_DIR = EXPERIMENT_ROOT_DIR / "audits"
PROCESSED_DATA_DIR = EXPERIMENT_ROOT_DIR / "processed_data"
FEATURE_DATA_DIR = EXPERIMENT_ROOT_DIR / "feature_data"
TARGET_DATA_DIR = EXPERIMENT_ROOT_DIR / "target_data"
SPLIT_DATA_DIR = EXPERIMENT_ROOT_DIR / "split_data"
SEQUENCE_DATA_DIR = EXPERIMENT_ROOT_DIR / "sequence_data"
SCALER_DIR = EXPERIMENT_ROOT_DIR / "scalers"
REGIME_DIR = EXPERIMENT_ROOT_DIR / "regimes"
ARRAY_DATA_DIR = EXPERIMENT_ROOT_DIR / "arrays"

MODEL_SOURCE_DIR = EXPERIMENT_ROOT_DIR / "model_sources"
MODEL_REGISTRY_DIR = EXPERIMENT_ROOT_DIR / "model_registry"
PREFLIGHT_DIR = EXPERIMENT_ROOT_DIR / "preflight"
STAGE1_SCREENING_DIR = EXPERIMENT_ROOT_DIR / "stage1_modern_screening"
ROBUSTNESS_DIR = EXPERIMENT_ROOT_DIR / "robustness"
OOF_DIR = EXPERIMENT_ROOT_DIR / "oof"

MODEL_OUTPUT_DIR = EXPERIMENT_ROOT_DIR / "model_outputs"
CHECKPOINT_DIR = MODEL_OUTPUT_DIR / "checkpoints"
PREDICTION_OUTPUT_DIR = EXPERIMENT_ROOT_DIR / "prediction_outputs"
DEVELOPMENT_PREDICTION_DIR = PREDICTION_OUTPUT_DIR / "development"
FINAL_TEST_PREDICTION_DIR = PREDICTION_OUTPUT_DIR / "final_test"
INFERENCE_OUTPUT_DIR = EXPERIMENT_ROOT_DIR / "inference_outputs"
COMPLEXITY_OUTPUT_DIR = EXPERIMENT_ROOT_DIR / "complexity_outputs"
UTILITY_OUTPUT_DIR = EXPERIMENT_ROOT_DIR / "utility_outputs"

CONFIG_DIR = EXPERIMENT_ROOT_DIR / "config"
ENVIRONMENT_DIR = EXPERIMENT_ROOT_DIR / "environment"
MANIFEST_DIR = EXPERIMENT_ROOT_DIR / "manifests"
LOG_DIR = EXPERIMENT_ROOT_DIR / "logs"
REPORT_DIR = EXPERIMENT_ROOT_DIR / "reports"

MANUSCRIPT_OUTPUT_DIR = EXPERIMENT_ROOT_DIR / "manuscript_outputs"
MANUSCRIPT_FIGURE_DIR = MANUSCRIPT_OUTPUT_DIR / "figures"
MANUSCRIPT_TABLE_DIR = MANUSCRIPT_OUTPUT_DIR / "tables"
MANUSCRIPT_DIAGNOSTIC_DIR = MANUSCRIPT_OUTPUT_DIR / "diagnostics"

REQUIRED_DIRECTORIES = (
    PROJECT_ROOT_DIR,
    EXPERIMENTS_ROOT_DIR,
    EXPERIMENT_ROOT_DIR,
    RAW_DATA_DIR,
    REFERENCE_DATA_DIR,
    AUDIT_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_DATA_DIR,
    TARGET_DATA_DIR,
    SPLIT_DATA_DIR,
    SEQUENCE_DATA_DIR,
    SCALER_DIR,
    REGIME_DIR,
    ARRAY_DATA_DIR,
    MODEL_SOURCE_DIR,
    MODEL_REGISTRY_DIR,
    PREFLIGHT_DIR,
    STAGE1_SCREENING_DIR,
    ROBUSTNESS_DIR,
    OOF_DIR,
    MODEL_OUTPUT_DIR,
    CHECKPOINT_DIR,
    PREDICTION_OUTPUT_DIR,
    DEVELOPMENT_PREDICTION_DIR,
    FINAL_TEST_PREDICTION_DIR,
    INFERENCE_OUTPUT_DIR,
    COMPLEXITY_OUTPUT_DIR,
    UTILITY_OUTPUT_DIR,
    CONFIG_DIR,
    ENVIRONMENT_DIR,
    MANIFEST_DIR,
    LOG_DIR,
    REPORT_DIR,
    MANUSCRIPT_OUTPUT_DIR,
    MANUSCRIPT_FIGURE_DIR,
    MANUSCRIPT_TABLE_DIR,
    MANUSCRIPT_DIAGNOSTIC_DIR,
)


# =============================================================================
# 4. RANDOMNESS AND REPRODUCIBILITY
# =============================================================================

GLOBAL_RANDOM_SEED = 42
STAGE1_SCREENING_SEED = 42
NEURAL_SEEDS = (42, 43, 44, 45, 46)

BOOTSTRAP_MASTER_SEED_CLASSIFICATION = 20260712
BOOTSTRAP_MASTER_SEED_UTILITY = 20260713

# Stage-1 deterministic epoch permutation rule:
# seed = base + fold * multiplier + epoch
STAGE1_EPOCH_SHUFFLE_SEED_BASE = 42
STAGE1_EPOCH_SHUFFLE_FOLD_MULTIPLIER = 100000

os.environ["PYTHONHASHSEED"] = str(GLOBAL_RANDOM_SEED)
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)


# =============================================================================
# 5. MARKET UNIVERSE
# =============================================================================

STOCK_TICKERS = (
    "MARUTI.NS",
    "AXISBANK.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "SBIN.NS",
    "ULTRACEMCO.NS",
    "LT.NS",
    "ASIANPAINT.NS",
    "TITAN.NS",
    "RELIANCE.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "BAJFINANCE.NS",
    "INFY.NS",
    "TCS.NS",
    "SUNPHARMA.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "BHARTIARTL.NS",
)

MARKET_INDEX_TICKER = "^NSEI"
MARKET_INDEX_NAME = "NIFTY 50"


# =============================================================================
# 6. RAW DATA WINDOW AND SOURCE-DATA POLICY
# =============================================================================

RAW_DATA_START_DATE = pd.Timestamp("2016-01-01")
RAW_DATA_END_DATE = pd.Timestamp("2026-06-30")  # scientific inclusive end
YFINANCE_DOWNLOAD_END_EXCLUSIVE = pd.Timestamp("2026-07-01")

DATA_PROVIDER = "Yahoo Finance via yfinance"
YFINANCE_AUTO_ADJUST = False
YFINANCE_ACTIONS = True

MINIMUM_BENCHMARK_COVERAGE_FRACTION = 0.98

NO_PRICE_IMPUTATION = True
NO_VOLUME_IMPUTATION = True
NO_FEATURE_IMPUTATION = True
NO_TARGET_IMPUTATION = True

# Generic provider/calendar anomaly rule. No old anomaly dates are hard-coded.
DATE_WIDE_ANOMALY_RULE_ENABLED = True
DATE_WIDE_ANOMALY_MIN_STOCK_FRACTION = 0.90
DATE_WIDE_ANOMALY_DEFINITION = (
    "remove a market date across the complete stock panel only when at least "
    "90% of stocks simultaneously exhibit zero volume and identical OHLC"
)

CORPORATE_ACTION_POLICY = (
    "download auto_adjust=False and actions=True; audit raw OHLC, adjusted "
    "close, dividends, splits, and date-specific adjustment factors before "
    "constructing canonical adjusted OHLC"
)

MARKET_CALENDAR_POLICY = "nifty50_reference_trading_calendar"


# =============================================================================
# 7. FORECASTING DESIGN
# =============================================================================

FORECAST_HORIZONS = (1, 3, 5)
MAX_FORECAST_HORIZON = max(FORECAST_HORIZONS)
LOOKBACK_WINDOW_LENGTH = 60

PRIMARY_TARGET_TYPE = "future_adjusted_return_direction"
POSITIVE_CLASS_DEFINITION = "future_adjusted_return_strictly_greater_than_zero"
ZERO_RETURN_CLASS = "non_up"
MULTITASK_REQUIRES_ALL_HORIZONS = True


# =============================================================================
# 8. PREDICTOR CONTRACT
# =============================================================================

STOCK_FEATURE_COUNT = 47
MARKET_CONTEXT_FEATURE_COUNT = 16
TOTAL_CONTINUOUS_FEATURE_COUNT = 63

INCLUDE_NEWS_FEATURES = False
INCLUDE_SENTIMENT_FEATURES = False
INCLUDE_FUNDAMENTAL_FEATURES = False
INCLUDE_FUTURE_INFORMATION = False
INCLUDE_NIFTY_VOLUME_FEATURES = False

FEATURE_CONSTRUCTION_POLICY = "causal_current_and_past_observations_only"


# =============================================================================
# 9. VOLATILITY-REGIME DESIGN
# =============================================================================

VOLATILITY_WINDOW = 20
REGIME_LOWER_QUANTILE = 0.33
REGIME_UPPER_QUANTILE = 0.67
REGIME_NAMES = ("Low", "Normal", "High")
REGIME_THRESHOLD_POLICY = "stock_specific_training_only_quantiles"


# =============================================================================
# 10. DEVELOPMENT / FINAL TEST DESIGN
# =============================================================================
# Development: 2016-2024
# Final held-out test: 2025-01-01 through 2026-06-30
# =============================================================================

DEVELOPMENT_START_DATE = pd.Timestamp("2016-01-01")
DEVELOPMENT_END_DATE = pd.Timestamp("2024-12-31")

FINAL_TEST_START_DATE = pd.Timestamp("2025-01-01")
FINAL_TEST_END_DATE = pd.Timestamp("2026-06-30")


# =============================================================================
# 11. ROLLING-ORIGIN DEVELOPMENT FOLDS
# =============================================================================

ROLLING_ORIGIN_FOLDS = (
    {
        "fold": 1,
        "train_start": pd.Timestamp("2016-01-01"),
        "train_end": pd.Timestamp("2019-12-31"),
        "validation_start": pd.Timestamp("2020-01-01"),
        "validation_end": pd.Timestamp("2020-12-31"),
    },
    {
        "fold": 2,
        "train_start": pd.Timestamp("2016-01-01"),
        "train_end": pd.Timestamp("2020-12-31"),
        "validation_start": pd.Timestamp("2021-01-01"),
        "validation_end": pd.Timestamp("2021-12-31"),
    },
    {
        "fold": 3,
        "train_start": pd.Timestamp("2016-01-01"),
        "train_end": pd.Timestamp("2021-12-31"),
        "validation_start": pd.Timestamp("2022-01-01"),
        "validation_end": pd.Timestamp("2022-12-31"),
    },
    {
        "fold": 4,
        "train_start": pd.Timestamp("2016-01-01"),
        "train_end": pd.Timestamp("2022-12-31"),
        "validation_start": pd.Timestamp("2023-01-01"),
        "validation_end": pd.Timestamp("2023-12-31"),
    },
    {
        "fold": 5,
        "train_start": pd.Timestamp("2016-01-01"),
        "train_end": pd.Timestamp("2023-12-31"),
        "validation_start": pd.Timestamp("2024-01-01"),
        "validation_end": pd.Timestamp("2024-12-31"),
    },
)

N_DEVELOPMENT_FOLDS = len(ROLLING_ORIGIN_FOLDS)


# =============================================================================
# 12. MODEL-SELECTION AND THRESHOLD POLICY
# =============================================================================

PRIMARY_MODEL_SELECTION_METRIC = "balanced_accuracy"
SECONDARY_MODEL_SELECTION_METRIC = "roc_auc"
MODEL_SELECTION_POLICY = "rolling_origin_development_validation_only"

STAGE1_FIXED_THRESHOLD = 0.50
STAGE1_ALLOW_THRESHOLD_OPTIMIZATION = False

THRESHOLD_SELECTION_POLICY = "development_out_of_fold_predictions_only"
OOF_THRESHOLD_MIN = 0.30
OOF_THRESHOLD_MAX = 0.70
OOF_THRESHOLD_STEP = 0.01
OOF_THRESHOLD_GRID = tuple(
    round(OOF_THRESHOLD_MIN + i * OOF_THRESHOLD_STEP, 2)
    for i in range(
        int(round((OOF_THRESHOLD_MAX - OOF_THRESHOLD_MIN) / OOF_THRESHOLD_STEP)) + 1
    )
)
OOF_THRESHOLD_SELECTION_METRIC = "balanced_accuracy"
OOF_THRESHOLD_TIEBREAK = (
    "closest_to_0.50",
    "lower_threshold",
)

FINAL_REFIT_POLICY = (
    "refit_frozen_configuration_on_complete_2016_2024_development_period"
)
FINAL_TEST_POLICY = "single_use_after_registry_freeze"


# =============================================================================
# 13. LEAKAGE-CONTROL POLICY
# =============================================================================

TRAIN_ONLY_PREPROCESSING = True
PURGE_FUTURE_TARGET_BOUNDARIES = True
FINAL_TEST_MAY_NOT_INFLUENCE_SELECTION = True
ALLOW_BEST_SEED_SELECTION = False
ALLOW_POST_TEST_MODEL_PROMOTION = False
ALLOW_POST_TEST_THRESHOLD_CHANGE = False
ALLOW_POST_TEST_FEATURE_SELECTION = False
ALLOW_POST_TEST_HYPERPARAMETER_CHANGE = False
ALLOW_POST_TEST_INFERENCE_RULE_CHANGE = False


# =============================================================================
# 14. IMPORTANT SEQUENCE POLICY
# =============================================================================
# Causal lookback is allowed across chronological boundaries. For example,
# an early-2025 final-test endpoint may use late-2024 observations as history.
# Its future target must nevertheless remain inside the final-test period.
# =============================================================================

USE_CAUSAL_CROSS_BOUNDARY_LOOKBACK = True
REQUIRE_LOOKBACK_WITHIN_SAME_SPLIT = False
REQUIRE_TARGET_WITHIN_SAME_EVALUATION_SPLIT = True
REQUIRE_SAME_TICKER_CONTINUITY = True
REQUIRE_TRADING_SESSION_CONTINUITY = True


# =============================================================================
# 15. DATA-PROCESSING POLICY
# =============================================================================

SCALER_POLICY = "fit_on_training_partition_only"
SCALER_TYPE = "MinMaxScaler"
CLIP_VALIDATION_AFTER_SCALING = False
CLIP_FINAL_TEST_AFTER_SCALING = False

MISSING_VALUE_POLICY = (
    "causal_feature_warmup_and_complete_case_after_feature_construction"
)


# =============================================================================
# 16. MODERN BENCHMARK FAMILY REGISTRY — FAMILY/CONFIG IDs ONLY
# =============================================================================
# Exact architecture hyperparameters and source/adaptation details will be
# frozen later in the dedicated audited model registry. Step 00 freezes the
# candidate universe so it cannot be expanded after observing validation data.
# =============================================================================

MODERN_MODEL_FAMILIES = (
    "DLinear",
    "NLinear",
    "PatchTST",
    "TimesNet",
    "iTransformer",
    "Fredformer",
    "FRNet",
    "MLF",
)

MODERN_STAGE1_CONFIGS = (
    "DLINEAR_A",
    "DLINEAR_B",
    "NLINEAR_A",
    "NLINEAR_B",
    "PATCHTST_A",
    "PATCHTST_B",
    "TIMESNET_A",
    "TIMESNET_B",
    "ITRANSFORMER_A",
    "ITRANSFORMER_B",
    "FREDFORMER_A",
    "FREDFORMER_B",
    "FRNET_A",
    "FRNET_B",
    "MLF_A",
    "MLF_B",
)

MODEL_FAMILY_TO_CONFIGS = {
    "DLinear": ("DLINEAR_A", "DLINEAR_B"),
    "NLinear": ("NLINEAR_A", "NLINEAR_B"),
    "PatchTST": ("PATCHTST_A", "PATCHTST_B"),
    "TimesNet": ("TIMESNET_A", "TIMESNET_B"),
    "iTransformer": ("ITRANSFORMER_A", "ITRANSFORMER_B"),
    "Fredformer": ("FREDFORMER_A", "FREDFORMER_B"),
    "FRNet": ("FRNET_A", "FRNET_B"),
    "MLF": ("MLF_A", "MLF_B"),
}

STAGE1_EXPECTED_FIT_COUNT = len(MODERN_STAGE1_CONFIGS) * N_DEVELOPMENT_FOLDS


# =============================================================================
# 17. FROZEN STAGE-1 TRAINING PROTOCOL
# =============================================================================

STAGE1_LOGICAL_BATCH_SIZE = 256
STAGE1_PRECISION = "float32"
STAGE1_AMP_ENABLED = False
STAGE1_OPTIMIZER = "AdamW"
STAGE1_WEIGHT_DECAY = 1e-4
STAGE1_SCHEDULER = None
STAGE1_DIRECTION_LOSS = "BCEWithLogitsLoss_mean"
STAGE1_CLASS_WEIGHTING = None
STAGE1_GRADIENT_CLIP_NORM = 1.0
STAGE1_MAX_EPOCHS = 100
STAGE1_MIN_EPOCH_BEFORE_STOP = 10
STAGE1_EARLY_STOPPING_PATIENCE = 12
STAGE1_CHECKPOINT_METRIC = "minimum_validation_directional_bce"

STAGE1_CONFIG_SELECTION_PRIMARY = "mean_5fold_macro_horizon_balanced_accuracy"
STAGE1_CONFIG_SELECTION_TIEBREAKS = (
    "higher_mean_macro_roc_auc",
    "higher_mean_macro_mcc",
    "higher_mean_macro_pr_auc",
    "lower_mean_macro_brier",
    "lower_trainable_parameter_count",
    "lexical_config_id",
)

# Execution-only microbatching may be introduced after preflight. It must not
# change the logical batch or the scientific objective.
ALLOW_EXECUTION_ONLY_MICROBATCHING = True
REQUIRE_EXACT_SAMPLE_WEIGHTED_ACCUMULATION = True
REQUIRE_ONE_OPTIMIZER_STEP_PER_LOGICAL_BATCH = True

# Old v2 findings (e.g. FRNET_B/MLF_B 64x4) are NOT frozen here. The clean run
# must re-audit memory and throughput before choosing execution-only settings.
REUSE_LEGACY_MICROBATCH_SETTINGS_AUTOMATICALLY = False


# =============================================================================
# 18. GPU / THROUGHPUT PREFLIGHT CONTRACT
# =============================================================================

ALLOW_CONCURRENT_CUDA_HEAVY_JOBS = False
REQUIRE_GPU_PROCESS_GUARD = True
REQUIRE_HEAVY_CONFIG_MEMORY_PREFLIGHT = True
REQUIRE_HEAVY_CONFIG_THROUGHPUT_PREFLIGHT = True
REQUIRE_REAL_DATA_SMOKE_TEST_FOR_HEAVY_FAMILIES = True
REQUIRE_PEAK_MEMORY_ALLOCATED = True
REQUIRE_PEAK_MEMORY_RESERVED = True
REQUIRE_SECONDS_PER_LOGICAL_UPDATE = True
REQUIRE_EXAMPLES_PER_SECOND = True
REQUIRE_FRESH_SUBPROCESS_PER_FIT = True
REQUIRE_EPOCH_LEVEL_PROGRESS = True
REQUIRE_EPOCH_HEARTBEAT_JSON = True
REQUIRE_HASH_VERIFIED_RESUMABILITY = True


# =============================================================================
# 19. ROBUSTNESS-SEED / OOF DEVELOPMENT POLICY
# =============================================================================

ROBUSTNESS_SELECTED_CONFIGS_ONLY = True
ROBUSTNESS_REPORT_ALL_SEEDS = True
ENSEMBLE_SEEDS_POST_HOC = False


# =============================================================================
# 20. PRIMARY INFERENCE POLICY
# =============================================================================

PRIMARY_SIGNIFICANCE_LEVEL = 0.05
MULTIPLE_TESTING_METHOD = "benjamini_hochberg"

DEPENDENCE_AWARE_INFERENCE_METHODS = (
    "market_session_cluster_bootstrap",
    "circular_moving_block_bootstrap",
)

PRIMARY_BOOTSTRAP_REPLICATES = 2000
DEFAULT_MOVING_BLOCK_LENGTH = 20
MOVING_BLOCK_SENSITIVITY_LENGTHS = (10, 20, 40)

ROBUSTNESS_CRITERION = (
    "both_dependence_aware_methods_exclude_the_relevant_null_"
    "and_fdr_adjusted_q_below_0.05_and_effect_direction_agrees"
)


# =============================================================================
# 21. METRIC POLICY
# =============================================================================

PRIMARY_CLASSIFICATION_METRICS = (
    "balanced_accuracy",
    "roc_auc",
    "mcc",
    "pr_auc",
)

SECONDARY_CLASSIFICATION_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
)

CALIBRATION_METRICS = (
    "brier_score",
    "expected_calibration_error",
)

COMPUTATIONAL_METRICS = (
    "trainable_parameters",
    "training_time_seconds",
    "inference_time_seconds",
    "peak_gpu_memory_allocated_mb",
    "peak_gpu_memory_reserved_mb",
    "seconds_per_logical_update",
    "examples_per_second",
    "flops_if_available",
)


# =============================================================================
# 22. ECONOMIC / UTILITY DIAGNOSTIC POLICY
# =============================================================================

UTILITY_IS_SECONDARY_DIAGNOSTIC = True
UTILITY_IS_DEPLOYABLE_BACKTEST = False
UTILITY_COST_GRID_BPS = (0, 5, 10, 20, 50)


# =============================================================================
# 23. OUTPUT-PROTECTION / FINAL-TEST LOCK POLICY
# =============================================================================

WRITE_FINAL_TEST_PREDICTIONS_ONCE = True
WRITE_FINAL_REGISTRY_BEFORE_TEST = True
PRESERVE_SEED_LEVEL_PREDICTIONS = True

FINAL_TEST_INITIAL_STATE = "LOCKED_NOT_OPENED"
FINAL_TEST_UNLOCK_REQUIRES_SEPARATE_MANIFEST = True
FINAL_TEST_LOCK_FILENAME = "00_final_test_lock.json"
CONFIG_MANIFEST_FILENAME = "00_experiment_configuration.json"
CONFIG_SHA_FILENAME = "00_experiment_configuration.sha256"
FROZEN_SCRIPT_FILENAME = "step00_config_frozen.py"


# =============================================================================
# 24. HELPER FUNCTIONS
# =============================================================================

def timestamp_to_string(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def serialize_fold(fold: dict[str, Any]) -> dict[str, Any]:
    return {
        "fold": int(fold["fold"]),
        "train_start": timestamp_to_string(fold["train_start"]),
        "train_end": timestamp_to_string(fold["train_end"]),
        "validation_start": timestamp_to_string(fold["validation_start"]),
        "validation_end": timestamp_to_string(fold["validation_end"]),
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256_bytes(canonical_json.encode("utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def stage1_epoch_shuffle_seed(fold: int, epoch: int) -> int:
    return (
        STAGE1_EPOCH_SHUFFLE_SEED_BASE
        + int(fold) * STAGE1_EPOCH_SHUFFLE_FOLD_MULTIPLIER
        + int(epoch)
    )


# =============================================================================
# 25. CONFIGURATION VALIDATION
# =============================================================================

def validate_configuration() -> None:
    # Identity/isolation
    if EXPERIMENT_TAG == LEGACY_EXPERIMENT_TAG:
        raise AssertionError("Clean experiment tag must differ from legacy tag.")

    if EXPERIMENT_ROOT_DIR == LEGACY_EXPERIMENT_ROOT_DIR:
        raise AssertionError("Clean and legacy experiment roots must differ.")

    # Universe
    if len(STOCK_TICKERS) != 20:
        raise AssertionError(f"Expected 20 stocks; found {len(STOCK_TICKERS)}.")

    if len(set(STOCK_TICKERS)) != len(STOCK_TICKERS):
        raise AssertionError("Duplicate stock tickers detected.")

    if MARKET_INDEX_TICKER in STOCK_TICKERS:
        raise AssertionError("Market index must not appear in STOCK_TICKERS.")

    # Source window
    if RAW_DATA_START_DATE >= RAW_DATA_END_DATE:
        raise AssertionError("RAW_DATA_START_DATE must precede RAW_DATA_END_DATE.")

    if YFINANCE_DOWNLOAD_END_EXCLUSIVE != RAW_DATA_END_DATE + pd.Timedelta(days=1):
        raise AssertionError(
            "YFINANCE_DOWNLOAD_END_EXCLUSIVE must be one calendar day after "
            "the intended inclusive raw-data end date."
        )

    if not (0.0 < MINIMUM_BENCHMARK_COVERAGE_FRACTION <= 1.0):
        raise AssertionError("Coverage fraction must lie in (0, 1].")

    if not (0.0 < DATE_WIDE_ANOMALY_MIN_STOCK_FRACTION <= 1.0):
        raise AssertionError("Anomaly fraction must lie in (0, 1].")

    # Forecasting
    if tuple(sorted(FORECAST_HORIZONS)) != FORECAST_HORIZONS:
        raise AssertionError("FORECAST_HORIZONS must be strictly increasing.")

    if len(set(FORECAST_HORIZONS)) != len(FORECAST_HORIZONS):
        raise AssertionError("FORECAST_HORIZONS contains duplicates.")

    if any(
        not isinstance(h, int) or isinstance(h, bool) or h <= 0
        for h in FORECAST_HORIZONS
    ):
        raise AssertionError("Every forecast horizon must be a positive integer.")

    if LOOKBACK_WINDOW_LENGTH <= MAX_FORECAST_HORIZON:
        raise AssertionError("Lookback must exceed maximum forecast horizon.")

    # Features
    if STOCK_FEATURE_COUNT + MARKET_CONTEXT_FEATURE_COUNT != TOTAL_CONTINUOUS_FEATURE_COUNT:
        raise AssertionError("Feature-count contract is inconsistent.")

    if TOTAL_CONTINUOUS_FEATURE_COUNT != 63:
        raise AssertionError("Expected exactly 63 continuous predictors.")

    if INCLUDE_FUTURE_INFORMATION:
        raise AssertionError("Future information may not enter predictors.")

    # Dates / final test
    if DEVELOPMENT_START_DATE < RAW_DATA_START_DATE:
        raise AssertionError("Development begins before raw-data period.")

    if DEVELOPMENT_END_DATE >= FINAL_TEST_START_DATE:
        raise AssertionError("Development and final-test periods overlap.")

    if FINAL_TEST_END_DATE > RAW_DATA_END_DATE:
        raise AssertionError("Final test extends beyond raw-data period.")

    if FINAL_TEST_START_DATE <= DEVELOPMENT_END_DATE:
        raise AssertionError("Final test must begin after development.")

    # Folds
    expected_fold_numbers = tuple(range(1, N_DEVELOPMENT_FOLDS + 1))
    observed_fold_numbers = tuple(f["fold"] for f in ROLLING_ORIGIN_FOLDS)

    if observed_fold_numbers != expected_fold_numbers:
        raise AssertionError("Rolling-origin fold numbers are not sequential.")

    previous_validation_end = None

    for fold in ROLLING_ORIGIN_FOLDS:
        train_start = fold["train_start"]
        train_end = fold["train_end"]
        validation_start = fold["validation_start"]
        validation_end = fold["validation_end"]

        if not (
            DEVELOPMENT_START_DATE
            <= train_start
            <= train_end
            < validation_start
            <= validation_end
            <= DEVELOPMENT_END_DATE
        ):
            raise AssertionError(f"Invalid temporal ordering in fold {fold['fold']}.")

        if train_start != DEVELOPMENT_START_DATE:
            raise AssertionError(
                "Every expanding fold must use the same development start date."
            )

        if previous_validation_end is not None and validation_start <= previous_validation_end:
            raise AssertionError("Rolling-origin validation periods overlap.")

        previous_validation_end = validation_end

    if ROLLING_ORIGIN_FOLDS[-1]["validation_end"] != DEVELOPMENT_END_DATE:
        raise AssertionError("Final development fold must end at DEVELOPMENT_END_DATE.")

    # Seeds
    if len(NEURAL_SEEDS) < 5:
        raise AssertionError("At least five prespecified neural seeds are required.")

    if len(set(NEURAL_SEEDS)) != len(NEURAL_SEEDS):
        raise AssertionError("NEURAL_SEEDS contains duplicates.")

    if GLOBAL_RANDOM_SEED not in NEURAL_SEEDS:
        raise AssertionError("GLOBAL_RANDOM_SEED must appear in NEURAL_SEEDS.")

    if STAGE1_SCREENING_SEED != GLOBAL_RANDOM_SEED:
        raise AssertionError("Stage-1 screening seed is expected to equal global seed 42.")

    # Leakage controls
    required_true_controls = {
        "TRAIN_ONLY_PREPROCESSING": TRAIN_ONLY_PREPROCESSING,
        "PURGE_FUTURE_TARGET_BOUNDARIES": PURGE_FUTURE_TARGET_BOUNDARIES,
        "FINAL_TEST_MAY_NOT_INFLUENCE_SELECTION": FINAL_TEST_MAY_NOT_INFLUENCE_SELECTION,
        "USE_CAUSAL_CROSS_BOUNDARY_LOOKBACK": USE_CAUSAL_CROSS_BOUNDARY_LOOKBACK,
        "REQUIRE_TARGET_WITHIN_SAME_EVALUATION_SPLIT": REQUIRE_TARGET_WITHIN_SAME_EVALUATION_SPLIT,
        "REQUIRE_SAME_TICKER_CONTINUITY": REQUIRE_SAME_TICKER_CONTINUITY,
        "REQUIRE_TRADING_SESSION_CONTINUITY": REQUIRE_TRADING_SESSION_CONTINUITY,
        "WRITE_FINAL_TEST_PREDICTIONS_ONCE": WRITE_FINAL_TEST_PREDICTIONS_ONCE,
        "WRITE_FINAL_REGISTRY_BEFORE_TEST": WRITE_FINAL_REGISTRY_BEFORE_TEST,
        "PRESERVE_SEED_LEVEL_PREDICTIONS": PRESERVE_SEED_LEVEL_PREDICTIONS,
        "FINAL_TEST_UNLOCK_REQUIRES_SEPARATE_MANIFEST": FINAL_TEST_UNLOCK_REQUIRES_SEPARATE_MANIFEST,
    }

    for name, value in required_true_controls.items():
        if not value:
            raise AssertionError(f"Required control disabled: {name}")

    prohibited_true_controls = {
        "ALLOW_BEST_SEED_SELECTION": ALLOW_BEST_SEED_SELECTION,
        "ALLOW_POST_TEST_MODEL_PROMOTION": ALLOW_POST_TEST_MODEL_PROMOTION,
        "ALLOW_POST_TEST_THRESHOLD_CHANGE": ALLOW_POST_TEST_THRESHOLD_CHANGE,
        "ALLOW_POST_TEST_FEATURE_SELECTION": ALLOW_POST_TEST_FEATURE_SELECTION,
        "ALLOW_POST_TEST_HYPERPARAMETER_CHANGE": ALLOW_POST_TEST_HYPERPARAMETER_CHANGE,
        "ALLOW_POST_TEST_INFERENCE_RULE_CHANGE": ALLOW_POST_TEST_INFERENCE_RULE_CHANGE,
        "REQUIRE_LOOKBACK_WITHIN_SAME_SPLIT": REQUIRE_LOOKBACK_WITHIN_SAME_SPLIT,
        "ENSEMBLE_SEEDS_POST_HOC": ENSEMBLE_SEEDS_POST_HOC,
        "REUSE_LEGACY_MICROBATCH_SETTINGS_AUTOMATICALLY": REUSE_LEGACY_MICROBATCH_SETTINGS_AUTOMATICALLY,
        "ALLOW_CONCURRENT_CUDA_HEAVY_JOBS": ALLOW_CONCURRENT_CUDA_HEAVY_JOBS,
    }

    for name, value in prohibited_true_controls.items():
        if value:
            raise AssertionError(f"Prohibited behavior enabled: {name}")

    # Model registry
    flattened_configs = tuple(
        cfg
        for family in MODERN_MODEL_FAMILIES
        for cfg in MODEL_FAMILY_TO_CONFIGS[family]
    )

    if flattened_configs != MODERN_STAGE1_CONFIGS:
        raise AssertionError("Model family/config registry is inconsistent.")

    if len(MODERN_MODEL_FAMILIES) != 8:
        raise AssertionError("Expected eight modern benchmark families.")

    if len(MODERN_STAGE1_CONFIGS) != 16:
        raise AssertionError("Expected sixteen Stage-1 configurations.")

    if STAGE1_EXPECTED_FIT_COUNT != 80:
        raise AssertionError("Expected 16 configs × 5 folds = 80 Stage-1 fits.")

    # Stage-1 contract
    if STAGE1_FIXED_THRESHOLD != 0.50:
        raise AssertionError("Stage-1 threshold must remain fixed at 0.50.")

    if STAGE1_ALLOW_THRESHOLD_OPTIMIZATION:
        raise AssertionError("Stage-1 threshold optimization is prohibited.")

    if STAGE1_LOGICAL_BATCH_SIZE != 256:
        raise AssertionError("Stage-1 logical batch must remain 256.")

    if STAGE1_AMP_ENABLED:
        raise AssertionError("Stage-1 AMP must remain disabled under the frozen protocol.")

    if STAGE1_PRECISION != "float32":
        raise AssertionError("Stage-1 precision must remain float32.")

    if STAGE1_GRADIENT_CLIP_NORM != 1.0:
        raise AssertionError("Stage-1 global gradient clip norm must remain 1.0.")

    if STAGE1_MAX_EPOCHS != 100:
        raise AssertionError("Stage-1 maximum epochs must remain 100.")

    if STAGE1_EARLY_STOPPING_PATIENCE != 12:
        raise AssertionError("Stage-1 early-stopping patience must remain 12.")

    # OOF threshold contract
    if OOF_THRESHOLD_GRID[0] != 0.30 or OOF_THRESHOLD_GRID[-1] != 0.70:
        raise AssertionError("OOF threshold grid must span 0.30 through 0.70.")

    if 0.50 not in OOF_THRESHOLD_GRID:
        raise AssertionError("OOF threshold grid must contain 0.50.")

    if len(OOF_THRESHOLD_GRID) != 41:
        raise AssertionError("Expected 41 OOF threshold values at step 0.01.")

    # Inference
    if not (0.0 < PRIMARY_SIGNIFICANCE_LEVEL < 1.0):
        raise AssertionError("Significance level must lie in (0, 1).")

    if DEPENDENCE_AWARE_INFERENCE_METHODS != (
        "market_session_cluster_bootstrap",
        "circular_moving_block_bootstrap",
    ):
        raise AssertionError("Dependence-aware inference methods changed unexpectedly.")

    if DEFAULT_MOVING_BLOCK_LENGTH not in MOVING_BLOCK_SENSITIVITY_LENGTHS:
        raise AssertionError("Default block length must appear in sensitivity set.")

    if PRIMARY_BOOTSTRAP_REPLICATES < 1000:
        raise AssertionError("Primary bootstrap replicate count is unexpectedly small.")

    # GPU operational controls
    required_preflight_controls = {
        "REQUIRE_GPU_PROCESS_GUARD": REQUIRE_GPU_PROCESS_GUARD,
        "REQUIRE_HEAVY_CONFIG_MEMORY_PREFLIGHT": REQUIRE_HEAVY_CONFIG_MEMORY_PREFLIGHT,
        "REQUIRE_HEAVY_CONFIG_THROUGHPUT_PREFLIGHT": REQUIRE_HEAVY_CONFIG_THROUGHPUT_PREFLIGHT,
        "REQUIRE_PEAK_MEMORY_ALLOCATED": REQUIRE_PEAK_MEMORY_ALLOCATED,
        "REQUIRE_PEAK_MEMORY_RESERVED": REQUIRE_PEAK_MEMORY_RESERVED,
        "REQUIRE_SECONDS_PER_LOGICAL_UPDATE": REQUIRE_SECONDS_PER_LOGICAL_UPDATE,
        "REQUIRE_EXAMPLES_PER_SECOND": REQUIRE_EXAMPLES_PER_SECOND,
        "REQUIRE_FRESH_SUBPROCESS_PER_FIT": REQUIRE_FRESH_SUBPROCESS_PER_FIT,
        "REQUIRE_EPOCH_LEVEL_PROGRESS": REQUIRE_EPOCH_LEVEL_PROGRESS,
        "REQUIRE_EPOCH_HEARTBEAT_JSON": REQUIRE_EPOCH_HEARTBEAT_JSON,
        "REQUIRE_HASH_VERIFIED_RESUMABILITY": REQUIRE_HASH_VERIFIED_RESUMABILITY,
    }

    for name, value in required_preflight_controls.items():
        if not value:
            raise AssertionError(f"Required execution safeguard disabled: {name}")


# =============================================================================
# 26. CANONICAL SCIENTIFIC CONFIGURATION PAYLOAD
# =============================================================================

def build_configuration_payload() -> dict[str, Any]:
    """Build timestamp-free canonical payload for deterministic hashing."""

    return {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "experiment": {
            "project_name": PROJECT_NAME,
            "working_manuscript_title": WORKING_MANUSCRIPT_TITLE,
            "methodology_positioning": METHODOLOGY_POSITIONING,
            "scientific_positioning": SCIENTIFIC_POSITIONING,
            "target_journal_context": TARGET_JOURNAL_CONTEXT,
            "experiment_tag": EXPERIMENT_TAG,
            "experiment_version": EXPERIMENT_VERSION,
            "legacy_experiment_tag_audit_only": LEGACY_EXPERIMENT_TAG,
        },
        "universe": {
            "stock_count": len(STOCK_TICKERS),
            "stock_tickers": list(STOCK_TICKERS),
            "market_index_ticker": MARKET_INDEX_TICKER,
            "market_index_name": MARKET_INDEX_NAME,
        },
        "source_window": {
            "raw_data_start_date": timestamp_to_string(RAW_DATA_START_DATE),
            "raw_data_end_date_inclusive": timestamp_to_string(RAW_DATA_END_DATE),
            "yfinance_download_end_exclusive": timestamp_to_string(
                YFINANCE_DOWNLOAD_END_EXCLUSIVE
            ),
            "provider": DATA_PROVIDER,
            "auto_adjust": YFINANCE_AUTO_ADJUST,
            "actions": YFINANCE_ACTIONS,
            "minimum_benchmark_coverage_fraction": MINIMUM_BENCHMARK_COVERAGE_FRACTION,
            "no_price_imputation": NO_PRICE_IMPUTATION,
            "no_volume_imputation": NO_VOLUME_IMPUTATION,
            "no_feature_imputation": NO_FEATURE_IMPUTATION,
            "no_target_imputation": NO_TARGET_IMPUTATION,
            "date_wide_anomaly_rule_enabled": DATE_WIDE_ANOMALY_RULE_ENABLED,
            "date_wide_anomaly_min_stock_fraction": DATE_WIDE_ANOMALY_MIN_STOCK_FRACTION,
            "date_wide_anomaly_definition": DATE_WIDE_ANOMALY_DEFINITION,
            "corporate_action_policy": CORPORATE_ACTION_POLICY,
            "market_calendar_policy": MARKET_CALENDAR_POLICY,
        },
        "forecasting": {
            "forecast_horizons": list(FORECAST_HORIZONS),
            "max_forecast_horizon": MAX_FORECAST_HORIZON,
            "lookback_window_length": LOOKBACK_WINDOW_LENGTH,
            "primary_target_type": PRIMARY_TARGET_TYPE,
            "positive_class_definition": POSITIVE_CLASS_DEFINITION,
            "zero_return_class": ZERO_RETURN_CLASS,
            "multitask_requires_all_horizons": MULTITASK_REQUIRES_ALL_HORIZONS,
        },
        "predictors": {
            "stock_feature_count": STOCK_FEATURE_COUNT,
            "market_context_feature_count": MARKET_CONTEXT_FEATURE_COUNT,
            "total_continuous_feature_count": TOTAL_CONTINUOUS_FEATURE_COUNT,
            "include_news": INCLUDE_NEWS_FEATURES,
            "include_sentiment": INCLUDE_SENTIMENT_FEATURES,
            "include_fundamentals": INCLUDE_FUNDAMENTAL_FEATURES,
            "include_future_information": INCLUDE_FUTURE_INFORMATION,
            "include_nifty_volume_features": INCLUDE_NIFTY_VOLUME_FEATURES,
            "construction_policy": FEATURE_CONSTRUCTION_POLICY,
        },
        "regime_design": {
            "volatility_window": VOLATILITY_WINDOW,
            "lower_quantile": REGIME_LOWER_QUANTILE,
            "upper_quantile": REGIME_UPPER_QUANTILE,
            "regime_names": list(REGIME_NAMES),
            "threshold_policy": REGIME_THRESHOLD_POLICY,
        },
        "development_and_test": {
            "development_start_date": timestamp_to_string(DEVELOPMENT_START_DATE),
            "development_end_date": timestamp_to_string(DEVELOPMENT_END_DATE),
            "final_test_start_date": timestamp_to_string(FINAL_TEST_START_DATE),
            "final_test_end_date": timestamp_to_string(FINAL_TEST_END_DATE),
            "rolling_origin_folds": [serialize_fold(f) for f in ROLLING_ORIGIN_FOLDS],
        },
        "selection_policy": {
            "primary_metric": PRIMARY_MODEL_SELECTION_METRIC,
            "secondary_metric": SECONDARY_MODEL_SELECTION_METRIC,
            "model_selection_policy": MODEL_SELECTION_POLICY,
            "stage1_fixed_threshold": STAGE1_FIXED_THRESHOLD,
            "stage1_allow_threshold_optimization": STAGE1_ALLOW_THRESHOLD_OPTIMIZATION,
            "threshold_selection_policy": THRESHOLD_SELECTION_POLICY,
            "oof_threshold_grid": list(OOF_THRESHOLD_GRID),
            "oof_threshold_selection_metric": OOF_THRESHOLD_SELECTION_METRIC,
            "oof_threshold_tiebreak": list(OOF_THRESHOLD_TIEBREAK),
            "final_refit_policy": FINAL_REFIT_POLICY,
            "final_test_policy": FINAL_TEST_POLICY,
        },
        "leakage_control": {
            "train_only_preprocessing": TRAIN_ONLY_PREPROCESSING,
            "purge_future_target_boundaries": PURGE_FUTURE_TARGET_BOUNDARIES,
            "final_test_may_not_influence_selection": FINAL_TEST_MAY_NOT_INFLUENCE_SELECTION,
            "allow_best_seed_selection": ALLOW_BEST_SEED_SELECTION,
            "allow_post_test_model_promotion": ALLOW_POST_TEST_MODEL_PROMOTION,
            "allow_post_test_threshold_change": ALLOW_POST_TEST_THRESHOLD_CHANGE,
            "allow_post_test_feature_selection": ALLOW_POST_TEST_FEATURE_SELECTION,
            "allow_post_test_hyperparameter_change": ALLOW_POST_TEST_HYPERPARAMETER_CHANGE,
            "allow_post_test_inference_rule_change": ALLOW_POST_TEST_INFERENCE_RULE_CHANGE,
        },
        "sequence_policy": {
            "use_causal_cross_boundary_lookback": USE_CAUSAL_CROSS_BOUNDARY_LOOKBACK,
            "require_lookback_within_same_split": REQUIRE_LOOKBACK_WITHIN_SAME_SPLIT,
            "require_target_within_same_evaluation_split": REQUIRE_TARGET_WITHIN_SAME_EVALUATION_SPLIT,
            "require_same_ticker_continuity": REQUIRE_SAME_TICKER_CONTINUITY,
            "require_trading_session_continuity": REQUIRE_TRADING_SESSION_CONTINUITY,
        },
        "data_processing": {
            "scaler_policy": SCALER_POLICY,
            "scaler_type": SCALER_TYPE,
            "clip_validation_after_scaling": CLIP_VALIDATION_AFTER_SCALING,
            "clip_final_test_after_scaling": CLIP_FINAL_TEST_AFTER_SCALING,
            "missing_value_policy": MISSING_VALUE_POLICY,
        },
        "modern_benchmark_registry": {
            "families": list(MODERN_MODEL_FAMILIES),
            "configs": list(MODERN_STAGE1_CONFIGS),
            "family_to_configs": {
                family: list(configs)
                for family, configs in MODEL_FAMILY_TO_CONFIGS.items()
            },
            "expected_stage1_fit_count": STAGE1_EXPECTED_FIT_COUNT,
            "architecture_hyperparameters_frozen_here": False,
            "architecture_registry_freeze_stage": "later_dedicated_model_registry",
        },
        "stage1_training": {
            "seed": STAGE1_SCREENING_SEED,
            "logical_batch_size": STAGE1_LOGICAL_BATCH_SIZE,
            "precision": STAGE1_PRECISION,
            "amp_enabled": STAGE1_AMP_ENABLED,
            "optimizer": STAGE1_OPTIMIZER,
            "weight_decay": STAGE1_WEIGHT_DECAY,
            "scheduler": STAGE1_SCHEDULER,
            "direction_loss": STAGE1_DIRECTION_LOSS,
            "class_weighting": STAGE1_CLASS_WEIGHTING,
            "gradient_clip_norm": STAGE1_GRADIENT_CLIP_NORM,
            "max_epochs": STAGE1_MAX_EPOCHS,
            "min_epoch_before_stop": STAGE1_MIN_EPOCH_BEFORE_STOP,
            "early_stopping_patience": STAGE1_EARLY_STOPPING_PATIENCE,
            "checkpoint_metric": STAGE1_CHECKPOINT_METRIC,
            "configuration_selection_primary": STAGE1_CONFIG_SELECTION_PRIMARY,
            "configuration_selection_tiebreaks": list(STAGE1_CONFIG_SELECTION_TIEBREAKS),
            "epoch_shuffle_seed_formula": (
                "base + fold * 100000 + epoch"
            ),
            "allow_execution_only_microbatching": ALLOW_EXECUTION_ONLY_MICROBATCHING,
            "require_exact_sample_weighted_accumulation": REQUIRE_EXACT_SAMPLE_WEIGHTED_ACCUMULATION,
            "require_one_optimizer_step_per_logical_batch": REQUIRE_ONE_OPTIMIZER_STEP_PER_LOGICAL_BATCH,
            "reuse_legacy_microbatch_settings_automatically": REUSE_LEGACY_MICROBATCH_SETTINGS_AUTOMATICALLY,
        },
        "gpu_preflight": {
            "allow_concurrent_cuda_heavy_jobs": ALLOW_CONCURRENT_CUDA_HEAVY_JOBS,
            "require_gpu_process_guard": REQUIRE_GPU_PROCESS_GUARD,
            "require_heavy_config_memory_preflight": REQUIRE_HEAVY_CONFIG_MEMORY_PREFLIGHT,
            "require_heavy_config_throughput_preflight": REQUIRE_HEAVY_CONFIG_THROUGHPUT_PREFLIGHT,
            "require_real_data_smoke_test_for_heavy_families": REQUIRE_REAL_DATA_SMOKE_TEST_FOR_HEAVY_FAMILIES,
            "require_peak_memory_allocated": REQUIRE_PEAK_MEMORY_ALLOCATED,
            "require_peak_memory_reserved": REQUIRE_PEAK_MEMORY_RESERVED,
            "require_seconds_per_logical_update": REQUIRE_SECONDS_PER_LOGICAL_UPDATE,
            "require_examples_per_second": REQUIRE_EXAMPLES_PER_SECOND,
            "require_fresh_subprocess_per_fit": REQUIRE_FRESH_SUBPROCESS_PER_FIT,
            "require_epoch_level_progress": REQUIRE_EPOCH_LEVEL_PROGRESS,
            "require_epoch_heartbeat_json": REQUIRE_EPOCH_HEARTBEAT_JSON,
            "require_hash_verified_resumability": REQUIRE_HASH_VERIFIED_RESUMABILITY,
        },
        "randomness": {
            "global_random_seed": GLOBAL_RANDOM_SEED,
            "stage1_screening_seed": STAGE1_SCREENING_SEED,
            "neural_seeds": list(NEURAL_SEEDS),
            "bootstrap_master_seed_classification": BOOTSTRAP_MASTER_SEED_CLASSIFICATION,
            "bootstrap_master_seed_utility": BOOTSTRAP_MASTER_SEED_UTILITY,
        },
        "inference": {
            "alpha": PRIMARY_SIGNIFICANCE_LEVEL,
            "multiple_testing_method": MULTIPLE_TESTING_METHOD,
            "dependence_aware_methods": list(DEPENDENCE_AWARE_INFERENCE_METHODS),
            "primary_bootstrap_replicates": PRIMARY_BOOTSTRAP_REPLICATES,
            "default_moving_block_length": DEFAULT_MOVING_BLOCK_LENGTH,
            "moving_block_sensitivity_lengths": list(MOVING_BLOCK_SENSITIVITY_LENGTHS),
            "robustness_criterion": ROBUSTNESS_CRITERION,
        },
        "metrics": {
            "primary_classification_metrics": list(PRIMARY_CLASSIFICATION_METRICS),
            "secondary_classification_metrics": list(SECONDARY_CLASSIFICATION_METRICS),
            "calibration_metrics": list(CALIBRATION_METRICS),
            "computational_metrics": list(COMPUTATIONAL_METRICS),
        },
        "utility": {
            "secondary_diagnostic": UTILITY_IS_SECONDARY_DIAGNOSTIC,
            "deployable_backtest": UTILITY_IS_DEPLOYABLE_BACKTEST,
            "cost_grid_bps": list(UTILITY_COST_GRID_BPS),
        },
        "output_protection": {
            "write_final_test_predictions_once": WRITE_FINAL_TEST_PREDICTIONS_ONCE,
            "write_final_registry_before_test": WRITE_FINAL_REGISTRY_BEFORE_TEST,
            "preserve_seed_level_predictions": PRESERVE_SEED_LEVEL_PREDICTIONS,
            "ensemble_seeds_post_hoc": ENSEMBLE_SEEDS_POST_HOC,
            "final_test_initial_state": FINAL_TEST_INITIAL_STATE,
            "final_test_unlock_requires_separate_manifest": FINAL_TEST_UNLOCK_REQUIRES_SEPARATE_MANIFEST,
        },
    }


# =============================================================================
# 27. IMMUTABLE INITIALIZATION / VERIFICATION
# =============================================================================

def get_manifest_path() -> Path:
    return MANIFEST_DIR / CONFIG_MANIFEST_FILENAME


def get_manifest_sha_path() -> Path:
    return MANIFEST_DIR / CONFIG_SHA_FILENAME


def get_final_test_lock_path() -> Path:
    return MANIFEST_DIR / FINAL_TEST_LOCK_FILENAME


def get_frozen_script_path() -> Path:
    return MANIFEST_DIR / FROZEN_SCRIPT_FILENAME


def directory_has_any_content(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def create_directories() -> None:
    for directory_path in REQUIRED_DIRECTORIES:
        directory_path.mkdir(parents=True, exist_ok=True)


def build_final_test_lock(configuration_fingerprint: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_tag": EXPERIMENT_TAG,
        "configuration_fingerprint_sha256": configuration_fingerprint,
        "state": FINAL_TEST_INITIAL_STATE,
        "final_test_opened": False,
        "final_test_scored": False,
        "final_test_start_date": timestamp_to_string(FINAL_TEST_START_DATE),
        "final_test_end_date": timestamp_to_string(FINAL_TEST_END_DATE),
        "unlock_policy": (
            "do_not_modify_this_initial_lock; create a separate pretest-registry "
            "and explicit unlock/opening manifest after all development rules are frozen"
        ),
        "prohibited_before_unlock": [
            "final-test-based feature selection",
            "final-test-based configuration selection",
            "final-test-based hyperparameter tuning",
            "final-test-based threshold selection",
            "final-test-based checkpoint selection",
            "final-test-based seed selection",
            "final-test-based post-hoc model promotion",
            "final-test-based inferential-rule modification",
        ],
    }


def initialize_new_freeze(
    configuration_payload: dict[str, Any],
    configuration_fingerprint: str,
) -> tuple[Path, str, str]:
    manifest_path = get_manifest_path()

    # A non-empty experiment root without the canonical Step-00 manifest is
    # ambiguous and may represent a partial/stale run. Never overwrite it.
    if directory_has_any_content(EXPERIMENT_ROOT_DIR) and not manifest_path.exists():
        raise RuntimeError(
            "The clean experiment root already exists and is non-empty, but the "
            "Step-00 configuration manifest is absent. Refusing to overwrite or "
            "adopt this directory. Use a new experiment tag or manually inspect "
            f"the directory: {EXPERIMENT_ROOT_DIR}"
        )

    create_directories()

    frozen_script_path = get_frozen_script_path()
    if frozen_script_path.exists():
        raise RuntimeError(
            f"Frozen Step-00 script already exists unexpectedly: {frozen_script_path}"
        )

    shutil.copy2(SCRIPT_FILE, frozen_script_path)
    frozen_script_sha256 = sha256_file(frozen_script_path)

    lock_path = get_final_test_lock_path()
    if lock_path.exists():
        raise RuntimeError(f"Final-test lock already exists unexpectedly: {lock_path}")

    lock_payload = build_final_test_lock(configuration_fingerprint)
    atomic_write_json(lock_path, lock_payload)
    final_test_lock_sha256 = sha256_file(lock_path)

    runtime_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "repository_root": str(REPOSITORY_ROOT_DIR),
        "script_file": str(SCRIPT_FILE),
        "experiment_root": str(EXPERIMENT_ROOT_DIR),
        "legacy_experiment_root_audit_only": str(LEGACY_EXPERIMENT_ROOT_DIR),
        "configuration_fingerprint_sha256": configuration_fingerprint,
        "frozen_step00_script_sha256": frozen_script_sha256,
        "final_test_lock_sha256": final_test_lock_sha256,
    }

    manifest = {
        "configuration": configuration_payload,
        "runtime_metadata": runtime_metadata,
        "freeze_status": "FROZEN_NEW",
    }

    atomic_write_json(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)

    atomic_write_text(
        get_manifest_sha_path(),
        f"{manifest_sha256}  {manifest_path.name}\n",
    )

    return manifest_path, manifest_sha256, "FROZEN_NEW"


def verify_existing_freeze(
    configuration_payload: dict[str, Any],
    configuration_fingerprint: str,
) -> tuple[Path, str, str]:
    manifest_path = get_manifest_path()
    manifest_sha_path = get_manifest_sha_path()
    frozen_script_path = get_frozen_script_path()
    lock_path = get_final_test_lock_path()

    if not manifest_path.exists():
        raise RuntimeError("Cannot verify: Step-00 configuration manifest is missing.")

    if not manifest_sha_path.exists():
        raise RuntimeError("Cannot verify: Step-00 manifest SHA sidecar is missing.")

    if not frozen_script_path.exists():
        raise RuntimeError("Cannot verify: frozen Step-00 script snapshot is missing.")

    if not lock_path.exists():
        raise RuntimeError("Cannot verify: initial final-test lock is missing.")

    with manifest_path.open("r", encoding="utf-8") as handle:
        stored_manifest = json.load(handle)

    stored_config = stored_manifest.get("configuration")
    stored_runtime = stored_manifest.get("runtime_metadata", {})

    if stored_config != configuration_payload:
        stored_fp = stored_runtime.get("configuration_fingerprint_sha256")
        raise RuntimeError(
            "Current Step-00 scientific configuration differs from the already "
            "frozen configuration. Refusing to overwrite the experiment.\n"
            f"Stored fingerprint : {stored_fp}\n"
            f"Current fingerprint: {configuration_fingerprint}\n"
            "Use a new experiment tag for any scientific configuration change."
        )

    stored_fp = stored_runtime.get("configuration_fingerprint_sha256")
    if stored_fp != configuration_fingerprint:
        raise RuntimeError("Stored configuration fingerprint is inconsistent.")

    current_manifest_sha256 = sha256_file(manifest_path)
    sidecar_text = manifest_sha_path.read_text(encoding="utf-8").strip()
    expected_manifest_sha256 = sidecar_text.split()[0] if sidecar_text else ""

    if current_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "Step-00 manifest SHA-256 mismatch. The frozen manifest appears modified."
        )

    stored_frozen_script_sha256 = stored_runtime.get("frozen_step00_script_sha256")
    actual_frozen_script_sha256 = sha256_file(frozen_script_path)

    if actual_frozen_script_sha256 != stored_frozen_script_sha256:
        raise RuntimeError("Frozen Step-00 script snapshot SHA-256 mismatch.")

    stored_lock_sha256 = stored_runtime.get("final_test_lock_sha256")
    actual_lock_sha256 = sha256_file(lock_path)

    if actual_lock_sha256 != stored_lock_sha256:
        raise RuntimeError("Initial final-test lock SHA-256 mismatch.")

    with lock_path.open("r", encoding="utf-8") as handle:
        lock_payload = json.load(handle)

    if lock_payload.get("state") != FINAL_TEST_INITIAL_STATE:
        raise RuntimeError("Initial final-test lock state was modified.")

    if lock_payload.get("final_test_opened") is not False:
        raise RuntimeError("Initial final-test lock says final_test_opened != False.")

    if lock_payload.get("final_test_scored") is not False:
        raise RuntimeError("Initial final-test lock says final_test_scored != False.")

    if lock_payload.get("configuration_fingerprint_sha256") != configuration_fingerprint:
        raise RuntimeError("Final-test lock fingerprint does not match configuration.")

    # The current working script is expected to remain identical to the frozen
    # Step-00 snapshot. A scientific configuration change must use a new tag.
    current_script_sha256 = sha256_file(SCRIPT_FILE)
    if current_script_sha256 != stored_frozen_script_sha256:
        raise RuntimeError(
            "Current q1_revision/step00_config.py differs from the frozen Step-00 "
            "script snapshot. Restore the frozen file or start a new experiment tag."
        )

    return manifest_path, current_manifest_sha256, "VERIFIED_EXISTING"


def write_human_readable_summary(
    manifest_path: Path,
    configuration_fingerprint: str,
    manifest_sha256: str,
    freeze_status: str,
) -> Path:
    summary_path = REPORT_DIR / "00_step00_summary.txt"

    summary = (
        "STEP 00 — Q1 REVISION CLEAN EXPERIMENT CONFIGURATION\n"
        "====================================================\n"
        f"Status: {freeze_status}\n"
        f"Project name: {PROJECT_NAME}\n"
        f"Experiment tag: {EXPERIMENT_TAG}\n"
        f"Repository root: {REPOSITORY_ROOT_DIR}\n"
        f"Experiment root: {EXPERIMENT_ROOT_DIR}\n"
        f"Legacy experiment root (audit only): {LEGACY_EXPERIMENT_ROOT_DIR}\n"
        f"Stocks: {len(STOCK_TICKERS)}\n"
        f"Benchmark: {MARKET_INDEX_TICKER}\n"
        f"Raw window: {RAW_DATA_START_DATE.date()} -> {RAW_DATA_END_DATE.date()}\n"
        f"Development: {DEVELOPMENT_START_DATE.date()} -> {DEVELOPMENT_END_DATE.date()}\n"
        f"Final test: {FINAL_TEST_START_DATE.date()} -> {FINAL_TEST_END_DATE.date()}\n"
        f"Forecast horizons: {FORECAST_HORIZONS}\n"
        f"Lookback sessions: {LOOKBACK_WINDOW_LENGTH}\n"
        f"Continuous predictors: {TOTAL_CONTINUOUS_FEATURE_COUNT}\n"
        f"Development folds: {N_DEVELOPMENT_FOLDS}\n"
        f"Modern model families: {len(MODERN_MODEL_FAMILIES)}\n"
        f"Stage-1 configs: {len(MODERN_STAGE1_CONFIGS)}\n"
        f"Expected Stage-1 fits: {STAGE1_EXPECTED_FIT_COUNT}\n"
        f"Stage-1 threshold: {STAGE1_FIXED_THRESHOLD:.2f}\n"
        f"OOF threshold grid: {OOF_THRESHOLD_MIN:.2f} -> {OOF_THRESHOLD_MAX:.2f} "
        f"step {OOF_THRESHOLD_STEP:.2f}\n"
        f"Neural robustness seeds: {NEURAL_SEEDS}\n"
        f"Dependence-aware methods: {DEPENDENCE_AWARE_INFERENCE_METHODS}\n"
        f"Moving-block sensitivity: {MOVING_BLOCK_SENSITIVITY_LENGTHS}\n"
        f"Final test state: {FINAL_TEST_INITIAL_STATE}\n"
        f"Configuration manifest: {manifest_path}\n"
        f"Configuration fingerprint SHA-256: {configuration_fingerprint}\n"
        f"Manifest SHA-256: {manifest_sha256}\n"
    )

    # Summary is convenience output, not part of the scientific freeze. It may
    # be regenerated on verification without altering frozen manifests.
    atomic_write_text(summary_path, summary)
    return summary_path


# =============================================================================
# 28. HUMAN-READABLE TERMINAL SUMMARY
# =============================================================================

def print_configuration_summary(
    manifest_path: Path,
    fingerprint: str,
    manifest_sha256: str,
    freeze_status: str,
    summary_path: Path,
) -> None:
    print("=" * 108)
    print("STEP 00 — Q1 REVISION CLEAN EXPERIMENT CONFIGURATION")
    print("=" * 108)
    print(f"Project name                  : {PROJECT_NAME}")
    print(f"Experiment tag                : {EXPERIMENT_TAG}")
    print(f"Experiment version            : {EXPERIMENT_VERSION}")
    print(f"Freeze status                 : {freeze_status}")
    print(f"Repository root               : {REPOSITORY_ROOT_DIR}")
    print(f"Revision project root         : {PROJECT_ROOT_DIR}")
    print(f"Clean experiment root         : {EXPERIMENT_ROOT_DIR}")
    print(f"Legacy root (audit only)      : {LEGACY_EXPERIMENT_ROOT_DIR}")
    print("-" * 108)
    print(f"Stock universe                : {len(STOCK_TICKERS)} stocks")
    print(f"Market benchmark              : {MARKET_INDEX_TICKER}")
    print(
        "Raw source window             : "
        f"{RAW_DATA_START_DATE.date()} -> {RAW_DATA_END_DATE.date()} (inclusive)"
    )
    print(
        "yfinance end argument         : "
        f"{YFINANCE_DOWNLOAD_END_EXCLUSIVE.date()} (exclusive)"
    )
    print(
        "Development period            : "
        f"{DEVELOPMENT_START_DATE.date()} -> {DEVELOPMENT_END_DATE.date()}"
    )
    print(
        "Final held-out test            : "
        f"{FINAL_TEST_START_DATE.date()} -> {FINAL_TEST_END_DATE.date()}"
    )
    print(f"Rolling-origin folds          : {N_DEVELOPMENT_FOLDS}")
    print(f"Forecast horizons             : {FORECAST_HORIZONS}")
    print(f"Sequence lookback             : {LOOKBACK_WINDOW_LENGTH} sessions")
    print(
        "Continuous predictors          : "
        f"{TOTAL_CONTINUOUS_FEATURE_COUNT} "
        f"({STOCK_FEATURE_COUNT} stock + {MARKET_CONTEXT_FEATURE_COUNT} NIFTY)"
    )
    print(f"Neural robustness seeds       : {NEURAL_SEEDS}")
    print("-" * 108)
    print(f"Causal cross-boundary lookback: {USE_CAUSAL_CROSS_BOUNDARY_LOOKBACK}")
    print(f"Same-split lookback required  : {REQUIRE_LOOKBACK_WITHIN_SAME_SPLIT}")
    print(f"Target split confinement      : {REQUIRE_TARGET_WITHIN_SAME_EVALUATION_SPLIT}")
    print(f"Train-only preprocessing      : {TRAIN_ONLY_PREPROCESSING}")
    print(f"Best-seed selection           : {ALLOW_BEST_SEED_SELECTION}")
    print(f"Post-test model promotion     : {ALLOW_POST_TEST_MODEL_PROMOTION}")
    print("-" * 108)
    print("Rolling-origin development folds:")
    for fold in ROLLING_ORIGIN_FOLDS:
        print(
            f"  Fold {fold['fold']}: "
            f"Train {fold['train_start'].date()} -> {fold['train_end'].date()} | "
            f"Validate {fold['validation_start'].date()} -> {fold['validation_end'].date()}"
        )
    print("-" * 108)
    print(f"Modern model families         : {len(MODERN_MODEL_FAMILIES)}")
    print(f"Stage-1 candidate configs     : {len(MODERN_STAGE1_CONFIGS)}")
    print(f"Expected Stage-1 fits         : {STAGE1_EXPECTED_FIT_COUNT}")
    print(f"Stage-1 logical batch         : {STAGE1_LOGICAL_BATCH_SIZE}")
    print(f"Stage-1 precision / AMP       : {STAGE1_PRECISION} / {STAGE1_AMP_ENABLED}")
    print(f"Stage-1 fixed threshold       : {STAGE1_FIXED_THRESHOLD:.2f}")
    print(f"Stage-1 threshold tuning      : {STAGE1_ALLOW_THRESHOLD_OPTIMIZATION}")
    print(
        "OOF threshold grid            : "
        f"{OOF_THRESHOLD_MIN:.2f} -> {OOF_THRESHOLD_MAX:.2f} "
        f"step {OOF_THRESHOLD_STEP:.2f} ({len(OOF_THRESHOLD_GRID)} values)"
    )
    print("-" * 108)
    print(f"Concurrent CUDA-heavy jobs    : {ALLOW_CONCURRENT_CUDA_HEAVY_JOBS}")
    print(f"Memory preflight required     : {REQUIRE_HEAVY_CONFIG_MEMORY_PREFLIGHT}")
    print(f"Throughput preflight required : {REQUIRE_HEAVY_CONFIG_THROUGHPUT_PREFLIGHT}")
    print(f"Peak allocated required       : {REQUIRE_PEAK_MEMORY_ALLOCATED}")
    print(f"Peak reserved required        : {REQUIRE_PEAK_MEMORY_RESERVED}")
    print(f"Epoch heartbeat required      : {REQUIRE_EPOCH_HEARTBEAT_JSON}")
    print("-" * 108)
    print(f"Dependence-aware inference    : {DEPENDENCE_AWARE_INFERENCE_METHODS}")
    print(f"Bootstrap replicates          : {PRIMARY_BOOTSTRAP_REPLICATES}")
    print(f"Moving-block sensitivity      : {MOVING_BLOCK_SENSITIVITY_LENGTHS}")
    print(f"FDR method                    : {MULTIPLE_TESTING_METHOD}")
    print("-" * 108)
    print(f"Final-test state              : {FINAL_TEST_INITIAL_STATE}")
    print(f"Configuration manifest        : {manifest_path}")
    print(f"Configuration SHA-256         : {fingerprint}")
    print(f"Manifest SHA-256              : {manifest_sha256}")
    print(f"Human-readable summary        : {summary_path}")
    print("=" * 108)
    print("PASS: Step 00 configuration validation completed successfully.")
    if freeze_status == "FROZEN_NEW":
        print("PASS: A new immutable clean Step-00 freeze was created.")
    else:
        print("PASS: Existing immutable Step-00 freeze verified; nothing was overwritten.")
    print("PASS: The abandoned v2 experiment remains audit-only and was not resumed.")
    print("PASS: The final 2025-2026 test period remains LOCKED_NOT_OPENED.")
    print("=" * 108)
    print("Do not run the next step yet. Paste the complete terminal output for audit.")


# =============================================================================
# 29. MAIN
# =============================================================================

def main() -> None:
    validate_configuration()

    configuration_payload = build_configuration_payload()
    configuration_fingerprint = canonical_json_sha256(configuration_payload)
    manifest_path = get_manifest_path()

    if manifest_path.exists():
        (
            manifest_path,
            manifest_sha256,
            freeze_status,
        ) = verify_existing_freeze(
            configuration_payload=configuration_payload,
            configuration_fingerprint=configuration_fingerprint,
        )
    else:
        (
            manifest_path,
            manifest_sha256,
            freeze_status,
        ) = initialize_new_freeze(
            configuration_payload=configuration_payload,
            configuration_fingerprint=configuration_fingerprint,
        )

    summary_path = write_human_readable_summary(
        manifest_path=manifest_path,
        configuration_fingerprint=configuration_fingerprint,
        manifest_sha256=manifest_sha256,
        freeze_status=freeze_status,
    )

    print_configuration_summary(
        manifest_path=manifest_path,
        fingerprint=configuration_fingerprint,
        manifest_sha256=manifest_sha256,
        freeze_status=freeze_status,
        summary_path=summary_path,
    )


if __name__ == "__main__":
    main()
