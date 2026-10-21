#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 05 — MATERIALIZE THE FINAL CLEAN, LEAKAGE-CONTROLLED MARKET PANEL

This is the ONLY Step-05 materialization script for the clean V3 experiment.
It replaces the earlier 05A/05B/05C sequence.

Scientific role
---------------
1. Verify the complete Step-00 -> Step-04 immutable chain.
2. Load the frozen Step-04 final analysis calendar.
3. Align all 20 stocks and ^NSEI to that calendar.
4. Preserve Yahoo raw fields unchanged for audit.
5. Materialize the Step-04 canonical retrospective adjusted-OHLC fields
   strictly as AUDIT/REFERENCE fields:
       factor = Adj Close / Close
       Canonical_Adj_Open  = Open  * factor
       Canonical_Adj_High  = High  * factor
       Canonical_Adj_Low   = Low   * factor
       Canonical_Adj_Close = Adj Close
6. Materialize STRICTLY CAUSAL modelling price fields from adjacent returns,
   not from the retrospective adjusted-price LEVEL:
       gross_return[t] = AdjClose[t] / AdjClose[t-1]
       Model_Close is a forward synthetic total-return index with base 100.
       Model_Open/High/Low preserve same-session raw OHLC/Close geometry.
7. Retain Yahoo provider Volume exactly as supplied:
       Model_Volume = Volume
   No manual stock-split volume adjustment and no volume imputation.
8. Materialize a price-only NIFTY modelling panel on the same frozen calendar.
9. Reconcile retained corporate-action dates and extreme-return diagnostics.
10. Freeze all outputs, hashes, source-code hashes, and a Step-05 manifest.

Why both canonical adjusted fields and Model_* fields?
-----------------------------------------------------
The Step-04 policy froze AdjClose/Close adjusted OHLC construction for
traceability. Those retrospective adjusted LEVELS are therefore preserved,
but they are explicitly marked AUDIT_ONLY.

For downstream modelling, the script uses adjacent adjusted-close RATIOS.
A common future adjustment multiplier applied to both adjacent historical
levels cancels in the ratio. The synthetic Model_* level is then accumulated
forward from those realized adjacent returns. This avoids feeding the
retrospectively back-adjusted LEVEL directly to the forecasting pipeline.

Important boundary
------------------
This stage DOES NOT:
- construct technical indicators,
- construct the frozen 47 stock + 16 NIFTY predictor set,
- construct 1d/3d/5d targets,
- create train/validation/test samples,
- fit scalers,
- estimate volatility regimes,
- build sequences,
- train or select models,
- tune thresholds,
- score the held-out final test.

Expected location
-----------------
<repository>/q1_revision/step05_materialize_clean_model_panel.py

Run
---
python step05_materialize_clean_model_panel.py

Verify frozen Step 05
---------------------
python step05_materialize_clean_model_panel.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. SCRIPT / CONFIG LOCATION
# =============================================================================

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent

if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError(
        "step05_materialize_clean_model_panel.py must be located inside "
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
        "The clean Step-00 configuration must exist first."
    ) from exc


# =============================================================================
# 1. STEP CONSTANTS
# =============================================================================

STEP_NUMBER = 5
STEP_ID = "05"
STEP_NAME = "STEP 05 — FINAL CLEAN LEAKAGE-CONTROLLED MARKET PANEL"
SCHEMA_VERSION = "1.0"

EXPECTED_STOCK_COUNT = len(cfg.STOCK_TICKERS)
SIGNAL_TIMING_POLICY = (
    "session-t market features are available only after session-t close"
)

# Diagnostic threshold inherited from the Step-03/04 audit workflow.
# It is NOT a deletion threshold.
EXTREME_ABS_LOG_RETURN_THRESHOLD = 0.20

OHLC_TOLERANCE = 1e-10
RETURN_TOLERANCE = 1e-12

REQUIRED_PROVIDER_FIELDS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)

REQUIRED_STOCK_ACTION_FIELDS = (
    "Dividends",
    "Stock Splits",
)

AUDIT_ONLY_STOCK_FIELDS = (
    "Provider_Adjustment_Factor_Audit_Only",
    "Canonical_Adj_Open_Audit_Only",
    "Canonical_Adj_High_Audit_Only",
    "Canonical_Adj_Low_Audit_Only",
    "Canonical_Adj_Close_Audit_Only",
    "Provider_Adj_Close_Audit_Only",
)

MODELLING_STOCK_FIELDS = (
    "Model_Open",
    "Model_High",
    "Model_Low",
    "Model_Close",
    "Model_Volume",
)

RETURN_AUDIT_FIELDS = (
    "Model_Gross_Return_1session",
    "Model_Return_1session",
    "Model_Log_Return_1session",
)


# =============================================================================
# 2. FROZEN INPUT PATHS
# =============================================================================

RAW_SNAPSHOT_ROOT = cfg.RAW_DATA_DIR / "provider_snapshot"
RAW_PARQUET_DIR = RAW_SNAPSHOT_ROOT / "parquet"

STEP01_MANIFEST = cfg.MANIFEST_DIR / "01_environment_audit_manifest.json"
STEP01_MANIFEST_SHA = cfg.MANIFEST_DIR / "01_environment_audit_manifest.sha256"
STEP01_AUDIT_JSON = cfg.ENVIRONMENT_DIR / "01_environment_audit.json"

STEP02_MANIFEST = cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.json"
STEP02_MANIFEST_SHA = cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.sha256"

STEP03_MANIFEST = cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.json"
STEP03_MANIFEST_SHA = cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.sha256"

STEP04_MANIFEST = cfg.MANIFEST_DIR / "04_session_policy_manifest.json"
STEP04_MANIFEST_SHA = cfg.MANIFEST_DIR / "04_session_policy_manifest.sha256"
STEP04_POLICY = cfg.MANIFEST_DIR / "04_cleaning_policy.json"
STEP04_CALENDAR = cfg.REFERENCE_DATA_DIR / "04_final_analysis_calendar.csv"
STEP04_ACTION_REGISTER = cfg.AUDIT_DIR / "04_corporate_action_register.csv"
STEP04_LARGE_RETURN_AUDIT = (
    cfg.AUDIT_DIR / "04_large_return_corporate_action_audit.csv"
)


# =============================================================================
# 3. STEP-05 OUTPUT PATHS
# =============================================================================

OUTPUT_ROOT = cfg.PROCESSED_DATA_DIR / "step05_clean_model_panel"
STOCK_OUTPUT_DIR = OUTPUT_ROOT / "stocks"
BENCHMARK_OUTPUT_DIR = OUTPUT_ROOT / "benchmark"

COMBINED_STOCK_PARQUET = OUTPUT_ROOT / "05_stock_panel.parquet"
COMBINED_STOCK_CSV = OUTPUT_ROOT / "05_stock_panel.csv"

NIFTY_PARQUET = BENCHMARK_OUTPUT_DIR / "05_nifty50_panel.parquet"
NIFTY_CSV = BENCHMARK_OUTPUT_DIR / "05_nifty50_panel.csv"

STOCK_AUDIT_CSV = cfg.AUDIT_DIR / "05_stock_materialization_audit.csv"
ACTION_RECONCILIATION_CSV = cfg.AUDIT_DIR / "05_action_reconciliation.csv"
RETURN_RECONCILIATION_CSV = cfg.AUDIT_DIR / "05_return_reconciliation.csv"
OUTPUT_HASH_REGISTER_CSV = cfg.AUDIT_DIR / "05_output_hashes.csv"
SOURCE_HASH_CSV = cfg.AUDIT_DIR / "05_source_file_hashes.csv"
PANEL_SUMMARY_JSON = cfg.AUDIT_DIR / "05_panel_materialization_summary.json"

STEP05_MANIFEST_JSON = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.json"
STEP05_MANIFEST_SHA = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.sha256"
FROZEN_STEP05_SCRIPT = (
    cfg.MANIFEST_DIR / "step05_materialize_clean_model_panel_frozen.py"
)
STEP05_SUMMARY_TXT = cfg.REPORT_DIR / "05_clean_model_panel_summary.txt"

FAILED_ATTEMPT_DIR = cfg.AUDIT_DIR / "step05_failed_attempts"


# =============================================================================
# 4. OUTPUT SCHEMAS
# =============================================================================

STOCK_AUDIT_COLUMNS = [
    "ticker",
    "rows",
    "first_date",
    "last_date",
    "provider_required_null_total",
    "model_ohlcv_null_total",
    "audit_adjusted_ohlc_null_total",
    "negative_provider_volume_rows",
    "zero_provider_volume_rows",
    "negative_model_volume_rows",
    "zero_model_volume_rows",
    "provider_volume_exact_match",
    "adjustment_factor_min",
    "adjustment_factor_max",
    "model_close_min",
    "model_close_max",
    "model_return_min",
    "model_return_max",
    "model_log_return_min",
    "model_log_return_max",
    "split_event_rows",
    "dividend_event_rows",
    "capital_gain_event_rows",
    "model_ohlc_geometry_valid",
    "canonical_adjusted_ohlc_geometry_valid",
    "provider_adj_close_level_used_as_model_level",
    "provider_volume_transformed",
]

ACTION_RECONCILIATION_COLUMNS = [
    "ticker",
    "Date",
    "action_types_step04",
    "action_types_step05",
    "step04_final_session",
    "step05_present",
    "match",
]

RETURN_RECONCILIATION_COLUMNS = [
    "ticker",
    "Date",
    "model_log_return",
    "absolute_model_log_return",
    "step04_flag_present",
    "step04_adjusted_log_return",
    "difference_vs_step04",
    "reconciliation_status",
]

OUTPUT_HASH_COLUMNS = [
    "scope",
    "ticker",
    "file_type",
    "file_path",
    "relative_path",
    "size_bytes",
    "sha256",
]

SOURCE_HASH_COLUMNS = [
    "role",
    "file_name",
    "absolute_path",
    "size_bytes",
    "sha256",
]


# =============================================================================
# 5. GENERIC HELPERS
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
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
            default=json_default,
        )
        handle.write("\n")
    os.replace(temporary, path)


def write_dataframe_csv(
    path: Path,
    dataframe: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    output = dataframe.copy()

    if columns is not None:
        for column in columns:
            if column not in output.columns:
                output[column] = pd.NA
        output = output[columns]

    output.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
    )
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def get_distribution_version(distribution_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def safe_ticker_filename(ticker: str) -> str:
    return (
        ticker
        .replace("^", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def relative_to_experiment(path: Path) -> str:
    return str(path.resolve().relative_to(cfg.EXPERIMENT_ROOT_DIR.resolve()))


def numeric_series(
    frame: pd.DataFrame,
    column: str,
    *,
    required: bool = True,
    default: float = 0.0,
) -> pd.Series:
    if column not in frame.columns:
        if required:
            raise RuntimeError(f"Missing required column: {column}")
        return pd.Series(default, index=frame.index, dtype="float64")

    return pd.to_numeric(frame[column], errors="raise").astype("float64")


def action_types_from_values(
    dividend: float,
    split: float,
    capital_gain: float,
) -> str:
    action_types = []
    if dividend != 0.0:
        action_types.append("dividend")
    if split != 0.0:
        action_types.append("stock_split")
    if capital_gain != 0.0:
        action_types.append("capital_gain")
    return "|".join(action_types)


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
    )

    if normalized.isna().any():
        bad = series.loc[normalized.isna()].unique().tolist()
        raise RuntimeError(f"Could not normalize boolean values: {bad}")

    return normalized.astype(bool)


# =============================================================================
# 6. STEP-00 / FINAL-TEST VERIFICATION
# =============================================================================

def verify_step00_contract() -> Dict[str, Any]:
    cfg.validate_configuration()

    payload = cfg.build_configuration_payload()
    fingerprint = cfg.canonical_json_sha256(payload)

    manifest_path, manifest_sha, freeze_status = cfg.verify_existing_freeze(
        configuration_payload=payload,
        configuration_fingerprint=fingerprint,
    )

    if freeze_status != "VERIFIED_EXISTING":
        raise RuntimeError("Step-00 freeze is not VERIFIED_EXISTING.")

    lock_path = cfg.get_final_test_lock_path()

    with lock_path.open("r", encoding="utf-8") as handle:
        lock_payload = json.load(handle)

    if lock_payload.get("state") != "LOCKED_NOT_OPENED":
        raise RuntimeError("Final-test state is not LOCKED_NOT_OPENED.")

    if lock_payload.get("final_test_opened") is not False:
        raise RuntimeError("final_test_opened != False.")

    if lock_payload.get("final_test_scored") is not False:
        raise RuntimeError("final_test_scored != False.")

    return {
        "configuration_fingerprint_sha256": fingerprint,
        "step00_manifest_path": str(manifest_path),
        "step00_manifest_sha256": manifest_sha,
        "final_test_lock_path": str(lock_path),
        "final_test_lock_sha256": sha256_file(lock_path),
        "final_test_state": lock_payload.get("state"),
    }


# =============================================================================
# 7. UPSTREAM FREEZE VERIFICATION
# =============================================================================

def verify_sha_sidecar(
    manifest_path: Path,
    sha_path: Path,
    label: str,
) -> str:
    if not manifest_path.exists():
        raise RuntimeError(f"{label} manifest missing: {manifest_path}")

    if not sha_path.exists():
        raise RuntimeError(f"{label} manifest SHA sidecar missing: {sha_path}")

    expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(manifest_path)

    if expected != actual:
        raise RuntimeError(
            f"{label} manifest SHA mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    return actual


def verify_manifest_artifacts(
    manifest: Dict[str, Any],
    *,
    artifact_key: str = "artifacts",
    skip_names: tuple[str, ...] = (),
) -> None:
    failures = []

    for name, record in manifest.get(artifact_key, {}).items():
        if name in skip_names:
            continue

        if not isinstance(record, dict) or "path" not in record:
            continue

        path = Path(record["path"])

        if not path.exists():
            failures.append(f"missing: {name} -> {path}")
            continue

        expected = record.get("sha256")
        if expected and sha256_file(path) != expected:
            failures.append(f"hash mismatch: {name}")

    if failures:
        raise RuntimeError(
            "Upstream artifact verification failed:\n" + "\n".join(failures)
        )


def verify_step01(
    expected_configuration_fingerprint: str,
) -> Dict[str, Any]:
    manifest_sha = verify_sha_sidecar(
        STEP01_MANIFEST,
        STEP01_MANIFEST_SHA,
        "Step 01",
    )

    with STEP01_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("status") != "PASS":
        raise RuntimeError("Stored Step-01 status is not PASS.")

    if (
        manifest.get("configuration_fingerprint_sha256")
        != expected_configuration_fingerprint
    ):
        raise RuntimeError("Step-01 / Step-00 fingerprint mismatch.")

    verify_manifest_artifacts(manifest)

    if not STEP01_AUDIT_JSON.exists():
        raise RuntimeError("Step-01 environment audit JSON is missing.")

    with STEP01_AUDIT_JSON.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)

    stored_python = audit.get("system", {}).get("python_executable")

    if stored_python:
        if Path(stored_python).resolve() != Path(sys.executable).resolve():
            raise RuntimeError(
                "Current Python executable differs from Step 01.\n"
                f"Step 01: {stored_python}\n"
                f"Current : {sys.executable}"
            )

    package_drift = []

    for row in audit.get("packages", []):
        if not row.get("required", False):
            continue

        distribution = row.get("distribution")
        stored_version = row.get("version", "")
        current_version = get_distribution_version(distribution)

        if current_version != stored_version:
            package_drift.append(
                (distribution, stored_version, current_version)
            )

    if package_drift:
        detail = "\n".join(
            f"{name}: Step01={stored!r}, current={current!r}"
            for name, stored, current in package_drift
        )
        raise RuntimeError(
            "Required-package drift detected after Step 01:\n" + detail
        )

    return {
        "manifest_sha256": manifest_sha,
        "python_executable": stored_python,
        "conda_environment": audit.get("system", {}).get("conda_default_env"),
    }


def verify_step02(
    expected_configuration_fingerprint: str,
    expected_step01_sha: str,
) -> Dict[str, Any]:
    manifest_sha = verify_sha_sidecar(
        STEP02_MANIFEST,
        STEP02_MANIFEST_SHA,
        "Step 02",
    )

    with STEP02_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("step", {}).get("status") != "PASS":
        raise RuntimeError("Stored Step-02 status is not PASS.")

    if (
        manifest.get("step00", {}).get("configuration_fingerprint_sha256")
        != expected_configuration_fingerprint
    ):
        raise RuntimeError("Step-02 / Step-00 fingerprint mismatch.")

    if (
        manifest.get("step01", {}).get("step01_manifest_sha256")
        != expected_step01_sha
    ):
        raise RuntimeError("Step-02 / Step-01 manifest mismatch.")

    # Verify canonical supporting artifacts.
    verify_manifest_artifacts(
        manifest,
        artifact_key="canonical_artifacts",
        skip_names=("raw_snapshot_root",),
    )

    raw_records = manifest.get("raw_file_records", [])
    failures = []

    for record in raw_records:
        path = Path(record["file_path"])

        if not path.exists():
            failures.append(f"missing raw file: {path}")
            continue

        if sha256_file(path) != record["sha256"]:
            failures.append(f"raw hash mismatch: {path}")

    if failures:
        raise RuntimeError(
            "Step-02 raw-file verification failed:\n" + "\n".join(failures)
        )

    return {
        "manifest_sha256": manifest_sha,
        "verified_raw_file_count": len(raw_records),
        "manifest": manifest,
    }


def verify_step03(
    expected_configuration_fingerprint: str,
    expected_step01_sha: str,
    expected_step02_sha: str,
) -> Dict[str, Any]:
    manifest_sha = verify_sha_sidecar(
        STEP03_MANIFEST,
        STEP03_MANIFEST_SHA,
        "Step 03",
    )

    with STEP03_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("step", {}).get("status") != "PASS":
        raise RuntimeError("Stored Step-03 status is not PASS.")

    expected = {
        "configuration_fingerprint_sha256": expected_configuration_fingerprint,
        "step01_manifest_sha256": expected_step01_sha,
        "step02_manifest_sha256": expected_step02_sha,
    }

    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Step-03 chain mismatch: {key}")

    verify_manifest_artifacts(manifest)

    return {
        "manifest_sha256": manifest_sha,
        "manifest": manifest,
    }


def verify_step04(
    expected_configuration_fingerprint: str,
    expected_step01_sha: str,
    expected_step02_sha: str,
    expected_step03_sha: str,
) -> Dict[str, Any]:
    manifest_sha = verify_sha_sidecar(
        STEP04_MANIFEST,
        STEP04_MANIFEST_SHA,
        "Step 04",
    )

    with STEP04_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("step", {}).get("status") != "PASS":
        raise RuntimeError("Stored Step-04 status is not PASS.")

    expected = {
        "configuration_fingerprint_sha256": expected_configuration_fingerprint,
        "step01_manifest_sha256": expected_step01_sha,
        "step02_manifest_sha256": expected_step02_sha,
        "step03_manifest_sha256": expected_step03_sha,
    }

    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Step-04 chain mismatch: {key}")

    verify_manifest_artifacts(manifest)

    if not STEP04_POLICY.exists():
        raise RuntimeError(f"Missing Step-04 policy: {STEP04_POLICY}")

    if not STEP04_CALENDAR.exists():
        raise RuntimeError(f"Missing Step-04 calendar: {STEP04_CALENDAR}")

    with STEP04_POLICY.open("r", encoding="utf-8") as handle:
        policy = json.load(handle)

    volume_policy = (
        policy
        .get("corporate_action_policy", {})
        .get("volume_policy_for_step05", "")
        .lower()
    )

    adjusted_price_policy = (
        policy
        .get("corporate_action_policy", {})
        .get("canonical_adjusted_price_policy_for_step05", "")
        .lower()
    )

    # Step 04 is already hash-verified above.  This semantic check must
    # therefore confirm the meaning of the frozen policy without depending on
    # one exact English wording.  The current Step-04 policy states:
    # "retain provider Volume as supplied; do not impute; do not apply a
    # manual split adjustment".
    retains_provider_volume = (
        "retain provider volume" in volume_policy
    )

    prohibits_volume_imputation = any(
        phrase in volume_policy
        for phrase in (
            "do not impute",
            "no volume imputation",
            "without volume imputation",
        )
    )

    prohibits_manual_split_adjustment = any(
        phrase in volume_policy
        for phrase in (
            "do not apply a manual split adjustment",
            "no manual split adjustment",
            "without manual split adjustment",
            "manual split adjustment: none",
        )
    )

    if not (
        retains_provider_volume
        and prohibits_volume_imputation
        and prohibits_manual_split_adjustment
    ):
        raise RuntimeError(
            "Step-04 volume policy is not compatible with this Step 05.\n"
            f"Observed policy: {volume_policy}"
        )

    if (
        "adj close / close" not in adjusted_price_policy
        and "adjclose/close" not in adjusted_price_policy
    ):
        raise RuntimeError(
            "Step-04 canonical adjusted-price policy is not compatible "
            "with this Step 05.\n"
            f"Observed policy: {adjusted_price_policy}"
        )

    return {
        "manifest_sha256": manifest_sha,
        "manifest": manifest,
        "policy": policy,
    }


# =============================================================================
# 8. STEP-05 IMMUTABILITY / STAGING
# =============================================================================

def canonical_step05_outputs() -> tuple[Path, ...]:
    return (
        OUTPUT_ROOT,
        STOCK_AUDIT_CSV,
        ACTION_RECONCILIATION_CSV,
        RETURN_RECONCILIATION_CSV,
        OUTPUT_HASH_REGISTER_CSV,
        SOURCE_HASH_CSV,
        PANEL_SUMMARY_JSON,
        STEP05_MANIFEST_JSON,
        STEP05_MANIFEST_SHA,
        FROZEN_STEP05_SCRIPT,
        STEP05_SUMMARY_TXT,
    )


def assert_new_step05_target() -> None:
    collisions = [
        path
        for path in canonical_step05_outputs()
        if path.exists()
    ]

    if collisions:
        detail = "\n".join(str(path) for path in collisions)
        raise RuntimeError(
            "Step-05 canonical output(s) already exist. Refusing to overwrite.\n"
            f"{detail}\n\n"
            "If Step 05 already passed, run --verify-only."
        )


def create_staging_root() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (
        cfg.PROCESSED_DATA_DIR
        / f"_step05_staging_{timestamp}_{os.getpid()}"
    )

    if root.exists():
        raise RuntimeError(f"Unexpected existing staging path: {root}")

    (root / "stocks").mkdir(parents=True, exist_ok=False)
    (root / "benchmark").mkdir(parents=True, exist_ok=False)

    return root


def archive_partial_outputs() -> Optional[Path]:
    """
    If a failed attempt promoted canonical Step-05 artifacts before the
    manifest was frozen, move them into the failed-attempt archive so a clean
    rerun is possible without silently deleting evidence.
    """

    if STEP05_MANIFEST_JSON.exists():
        return None

    existing = [
        path
        for path in canonical_step05_outputs()
        if path.exists()
    ]

    if not existing:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = FAILED_ATTEMPT_DIR / f"partial_artifacts_{timestamp}"
    archive.mkdir(parents=True, exist_ok=False)

    for path in existing:
        destination = archive / path.name

        try:
            shutil.move(str(path), str(destination))
        except Exception:
            # Best effort only. The failure diagnostic will record the archive.
            pass

    return archive


# =============================================================================
# 9. LOADERS
# =============================================================================

def load_final_calendar(
    step04_policy: Dict[str, Any],
) -> pd.DataFrame:
    calendar = pd.read_csv(
        STEP04_CALENDAR,
        parse_dates=["Date"],
    )

    required = {
        "session_number",
        "Date",
        "calendar_year",
        "calendar_month",
        "day_of_week",
    }

    missing = required - set(calendar.columns)

    if missing:
        raise RuntimeError(
            "Step-04 final calendar is missing columns: "
            + ", ".join(sorted(missing))
        )

    calendar["Date"] = (
        pd.to_datetime(calendar["Date"], errors="raise")
        .dt.normalize()
    )

    calendar = (
        calendar
        .sort_values("session_number")
        .reset_index(drop=True)
    )

    if calendar["Date"].duplicated().any():
        raise RuntimeError("Duplicate dates in Step-04 final calendar.")

    if not calendar["Date"].is_monotonic_increasing:
        raise RuntimeError("Step-04 final calendar is not chronological.")

    expected_sessions = (
        step04_policy
        .get("final_analysis_calendar", {})
        .get("session_count")
    )

    if expected_sessions is not None:
        if int(expected_sessions) != len(calendar):
            raise RuntimeError(
                "Step-04 policy/calendar session-count mismatch.\n"
                f"Policy  : {expected_sessions}\n"
                f"Calendar: {len(calendar)}"
            )

    return calendar[
        [
            "session_number",
            "Date",
            "calendar_year",
            "calendar_month",
            "day_of_week",
        ]
    ].copy()


def load_raw_frame(ticker: str) -> pd.DataFrame:
    path = RAW_PARQUET_DIR / f"{safe_ticker_filename(ticker)}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing raw Parquet snapshot: {path}")

    frame = pd.read_parquet(path, engine="pyarrow")

    if "Date" not in frame.columns:
        raise RuntimeError(f"{ticker}: Date column missing.")

    frame["Date"] = (
        pd.to_datetime(frame["Date"], errors="raise")
        .dt.normalize()
    )

    if frame["Date"].duplicated().any():
        raise RuntimeError(f"{ticker}: duplicate raw dates.")

    return (
        frame
        .sort_values("Date")
        .reset_index(drop=True)
    )


def align_to_calendar(
    ticker: str,
    raw: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    stock: bool,
) -> pd.DataFrame:
    base = calendar[
        [
            "session_number",
            "Date",
            "calendar_year",
            "calendar_month",
            "day_of_week",
        ]
    ].copy()

    panel = base.merge(
        raw,
        on="Date",
        how="left",
        validate="one_to_one",
    )

    required = list(REQUIRED_PROVIDER_FIELDS)

    if stock:
        required.extend(REQUIRED_STOCK_ACTION_FIELDS)

    missing_columns = [
        column
        for column in required
        if column not in panel.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"{ticker}: missing required provider columns: "
            + ", ".join(missing_columns)
        )

    null_counts = panel[required].isna().sum()
    bad = null_counts.loc[null_counts > 0]

    if not bad.empty:
        raise RuntimeError(
            f"{ticker}: retained final calendar contains null provider data:\n"
            f"{bad.to_dict()}"
        )

    return panel


# =============================================================================
# 10. STOCK MATERIALIZATION
# =============================================================================

def materialize_stock_panel(
    ticker: str,
    aligned: pd.DataFrame,
) -> pd.DataFrame:
    result = aligned.copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Provider fields as numeric values
    # -------------------------------------------------------------------------
    open_ = numeric_series(result, "Open")
    high = numeric_series(result, "High")
    low = numeric_series(result, "Low")
    close = numeric_series(result, "Close")
    adj_close = numeric_series(result, "Adj Close")
    volume = numeric_series(result, "Volume")
    dividends = numeric_series(result, "Dividends")
    splits = numeric_series(result, "Stock Splits")

    capital_gains = numeric_series(
        result,
        "Capital Gains",
        required=False,
        default=0.0,
    )

    prices = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": adj_close,
        }
    )

    if (~np.isfinite(prices.to_numpy(dtype=float))).any():
        raise RuntimeError(f"{ticker}: non-finite provider prices.")

    if (prices <= 0).any().any():
        raise RuntimeError(f"{ticker}: non-positive provider price.")

    if (~np.isfinite(volume.to_numpy(dtype=float))).any():
        raise RuntimeError(f"{ticker}: non-finite provider Volume.")

    if (volume < 0).any():
        raise RuntimeError(f"{ticker}: negative provider Volume.")

    # -------------------------------------------------------------------------
    # A. Step-04 canonical adjusted OHLC — AUDIT/REFERENCE ONLY
    # -------------------------------------------------------------------------
    adjustment_factor = adj_close / close

    if (
        (~np.isfinite(adjustment_factor.to_numpy(dtype=float))).any()
        or (adjustment_factor <= 0).any()
    ):
        raise RuntimeError(f"{ticker}: invalid Adj Close / Close factor.")

    result["Provider_Adjustment_Factor_Audit_Only"] = adjustment_factor
    result["Canonical_Adj_Open_Audit_Only"] = open_ * adjustment_factor
    result["Canonical_Adj_High_Audit_Only"] = high * adjustment_factor
    result["Canonical_Adj_Low_Audit_Only"] = low * adjustment_factor
    result["Canonical_Adj_Close_Audit_Only"] = adj_close
    result["Provider_Adj_Close_Audit_Only"] = adj_close

    # -------------------------------------------------------------------------
    # B. Strictly causal modelling return increments
    #
    # We never use the provider adjusted-close LEVEL as Model_Close.
    # Adjacent ratios are used; common future back-adjustment multipliers
    # cancel between adjacent historical observations.
    # -------------------------------------------------------------------------
    gross_return = adj_close / adj_close.shift(1)

    invalid_return_pair = (
        gross_return.notna()
        & (
            (~np.isfinite(gross_return.to_numpy(dtype=float)))
            | (gross_return <= 0)
        )
    )

    if invalid_return_pair.any():
        bad_dates = result.loc[invalid_return_pair, "Date"].tolist()
        raise RuntimeError(
            f"{ticker}: invalid adjacent adjusted-close gross return "
            f"on dates {bad_dates}"
        )

    result["Model_Gross_Return_1session"] = gross_return
    result["Model_Return_1session"] = gross_return - 1.0
    result["Model_Log_Return_1session"] = np.log(gross_return)

    # Boundary convention for the synthetic index only.
    index_gross = gross_return.copy()
    index_gross.iloc[0] = 1.0

    if index_gross.isna().any():
        raise RuntimeError(
            f"{ticker}: unexpected interior null in adjacent-return series."
        )

    result["Model_Close"] = 100.0 * index_gross.cumprod()

    # -------------------------------------------------------------------------
    # C. Same-session OHLC geometry
    #
    # O/C, H/C and L/C are invariant to a common split scaling and use only
    # information known by the close of session t.
    # -------------------------------------------------------------------------
    result["Model_Open"] = result["Model_Close"] * (open_ / close)
    result["Model_High"] = result["Model_Close"] * (high / close)
    result["Model_Low"] = result["Model_Close"] * (low / close)

    # Step-04 policy: provider Volume is retained exactly as supplied.
    result["Model_Volume"] = volume

    # -------------------------------------------------------------------------
    # D. Explicit action metadata
    # -------------------------------------------------------------------------
    result["Dividend"] = dividends
    result["Stock_Split"] = splits
    result["Capital_Gain"] = capital_gains

    result["Corporate_Action_Today"] = (
        (dividends != 0.0)
        | (splits != 0.0)
        | (capital_gains != 0.0)
    )

    result["Action_Types"] = [
        action_types_from_values(
            float(dividends.iloc[i]),
            float(splits.iloc[i]),
            float(capital_gains.iloc[i]),
        )
        for i in range(len(result))
    ]

    result.insert(1, "Ticker", ticker)
    result["Instrument_Type"] = "stock"

    return result


# =============================================================================
# 11. STOCK VALIDATION
# =============================================================================

def validate_stock_panel(
    ticker: str,
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
) -> Dict[str, Any]:
    if len(panel) != len(calendar):
        raise RuntimeError(
            f"{ticker}: panel/calendar row-count mismatch."
        )

    if not panel["Date"].equals(calendar["Date"]):
        raise RuntimeError(
            f"{ticker}: panel dates differ from frozen calendar."
        )

    if panel["Date"].duplicated().any():
        raise RuntimeError(f"{ticker}: duplicate retained dates.")

    provider_required = list(REQUIRED_PROVIDER_FIELDS) + list(
        REQUIRED_STOCK_ACTION_FIELDS
    )

    provider_null_total = int(
        panel[provider_required].isna().sum().sum()
    )

    model_null_total = int(
        panel[list(MODELLING_STOCK_FIELDS)].isna().sum().sum()
    )

    audit_adjusted_null_total = int(
        panel[list(AUDIT_ONLY_STOCK_FIELDS)].isna().sum().sum()
    )

    if provider_null_total != 0:
        raise RuntimeError(
            f"{ticker}: provider required-field null total = "
            f"{provider_null_total}"
        )

    if model_null_total != 0:
        raise RuntimeError(
            f"{ticker}: Model OHLCV null total = {model_null_total}"
        )

    if audit_adjusted_null_total != 0:
        raise RuntimeError(
            f"{ticker}: canonical adjusted-audit null total = "
            f"{audit_adjusted_null_total}"
        )

    # Return fields are causally undefined on the first retained session only.
    for column in RETURN_AUDIT_FIELDS:
        null_mask = panel[column].isna()

        expected_null_count = 1

        if int(null_mask.sum()) != expected_null_count:
            raise RuntimeError(
                f"{ticker}: {column} expected exactly one boundary null; "
                f"found {int(null_mask.sum())}."
            )

        if not bool(null_mask.iloc[0]):
            raise RuntimeError(
                f"{ticker}: {column} boundary null is not on first session."
            )

    if (panel["Model_Volume"] < 0).any():
        raise RuntimeError(f"{ticker}: negative Model_Volume.")

    if not np.array_equal(
        panel["Model_Volume"].to_numpy(),
        pd.to_numeric(panel["Volume"], errors="raise").to_numpy(),
        equal_nan=True,
    ):
        raise RuntimeError(
            f"{ticker}: Model_Volume differs from provider Volume."
        )

    model_high_invalid = (
        panel["Model_High"]
        <
        panel[
            [
                "Model_Open",
                "Model_Low",
                "Model_Close",
            ]
        ].max(axis=1)
        - OHLC_TOLERANCE
    )

    model_low_invalid = (
        panel["Model_Low"]
        >
        panel[
            [
                "Model_Open",
                "Model_High",
                "Model_Close",
            ]
        ].min(axis=1)
        + OHLC_TOLERANCE
    )

    if model_high_invalid.any() or model_low_invalid.any():
        raise RuntimeError(f"{ticker}: Model OHLC geometry violation.")

    adj_high_invalid = (
        panel["Canonical_Adj_High_Audit_Only"]
        <
        panel[
            [
                "Canonical_Adj_Open_Audit_Only",
                "Canonical_Adj_Low_Audit_Only",
                "Canonical_Adj_Close_Audit_Only",
            ]
        ].max(axis=1)
        - OHLC_TOLERANCE
    )

    adj_low_invalid = (
        panel["Canonical_Adj_Low_Audit_Only"]
        >
        panel[
            [
                "Canonical_Adj_Open_Audit_Only",
                "Canonical_Adj_High_Audit_Only",
                "Canonical_Adj_Close_Audit_Only",
            ]
        ].min(axis=1)
        + OHLC_TOLERANCE
    )

    if adj_high_invalid.any() or adj_low_invalid.any():
        raise RuntimeError(
            f"{ticker}: canonical adjusted OHLC geometry violation."
        )

    returns = panel["Model_Return_1session"].dropna()
    log_returns = panel["Model_Log_Return_1session"].dropna()

    dividend_events = int(
        (pd.to_numeric(panel["Dividend"], errors="raise") != 0).sum()
    )

    split_events = int(
        (pd.to_numeric(panel["Stock_Split"], errors="raise") != 0).sum()
    )

    capital_events = int(
        (pd.to_numeric(panel["Capital_Gain"], errors="raise") != 0).sum()
    )

    return {
        "ticker": ticker,
        "rows": int(len(panel)),
        "first_date": panel["Date"].min().strftime("%Y-%m-%d"),
        "last_date": panel["Date"].max().strftime("%Y-%m-%d"),
        "provider_required_null_total": provider_null_total,
        "model_ohlcv_null_total": model_null_total,
        "audit_adjusted_ohlc_null_total": audit_adjusted_null_total,
        "negative_provider_volume_rows": int((panel["Volume"] < 0).sum()),
        "zero_provider_volume_rows": int((panel["Volume"] == 0).sum()),
        "negative_model_volume_rows": int((panel["Model_Volume"] < 0).sum()),
        "zero_model_volume_rows": int((panel["Model_Volume"] == 0).sum()),
        "provider_volume_exact_match": True,
        "adjustment_factor_min": float(
            panel["Provider_Adjustment_Factor_Audit_Only"].min()
        ),
        "adjustment_factor_max": float(
            panel["Provider_Adjustment_Factor_Audit_Only"].max()
        ),
        "model_close_min": float(panel["Model_Close"].min()),
        "model_close_max": float(panel["Model_Close"].max()),
        "model_return_min": float(returns.min()),
        "model_return_max": float(returns.max()),
        "model_log_return_min": float(log_returns.min()),
        "model_log_return_max": float(log_returns.max()),
        "split_event_rows": split_events,
        "dividend_event_rows": dividend_events,
        "capital_gain_event_rows": capital_events,
        "model_ohlc_geometry_valid": True,
        "canonical_adjusted_ohlc_geometry_valid": True,
        "provider_adj_close_level_used_as_model_level": False,
        "provider_volume_transformed": False,
    }


# =============================================================================
# 12. NIFTY MATERIALIZATION
# =============================================================================

def materialize_nifty_panel(
    aligned: pd.DataFrame,
) -> pd.DataFrame:
    result = aligned.copy().reset_index(drop=True)

    open_ = numeric_series(result, "Open")
    high = numeric_series(result, "High")
    low = numeric_series(result, "Low")
    close = numeric_series(result, "Close")
    adj_close = numeric_series(result, "Adj Close")
    volume = numeric_series(result, "Volume")

    prices = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": adj_close,
        }
    )

    if (~np.isfinite(prices.to_numpy(dtype=float))).any():
        raise RuntimeError("^NSEI: non-finite retained price.")

    if (prices <= 0).any().any():
        raise RuntimeError("^NSEI: non-positive retained price.")

    # Step-04 canonical factor retained for audit/reference consistency.
    factor = adj_close / close

    if (
        (~np.isfinite(factor.to_numpy(dtype=float))).any()
        or (factor <= 0).any()
    ):
        raise RuntimeError("^NSEI: invalid Adj Close / Close factor.")

    result["NIFTY_Provider_Adjustment_Factor_Audit_Only"] = factor
    result["NIFTY_Canonical_Adj_Open_Audit_Only"] = open_ * factor
    result["NIFTY_Canonical_Adj_High_Audit_Only"] = high * factor
    result["NIFTY_Canonical_Adj_Low_Audit_Only"] = low * factor
    result["NIFTY_Canonical_Adj_Close_Audit_Only"] = adj_close

    # Price-only NIFTY model representation. Close ratios are sufficient for
    # the index; no equity corporate-action transformation is required.
    gross = close / close.shift(1)

    invalid = (
        gross.notna()
        & (
            (~np.isfinite(gross.to_numpy(dtype=float)))
            | (gross <= 0)
        )
    )

    if invalid.any():
        raise RuntimeError("^NSEI: invalid adjacent Close return.")

    result["NIFTY_Gross_Return_1session"] = gross
    result["NIFTY_Return_1session"] = gross - 1.0
    result["NIFTY_Log_Return_1session"] = np.log(gross)

    index_gross = gross.copy()
    index_gross.iloc[0] = 1.0

    if index_gross.isna().any():
        raise RuntimeError("^NSEI: interior null in retained Close returns.")

    result["NIFTY_Model_Close"] = 100.0 * index_gross.cumprod()
    result["NIFTY_Model_Open"] = (
        result["NIFTY_Model_Close"] * (open_ / close)
    )
    result["NIFTY_Model_High"] = (
        result["NIFTY_Model_Close"] * (high / close)
    )
    result["NIFTY_Model_Low"] = (
        result["NIFTY_Model_Close"] * (low / close)
    )

    # NIFTY volume is preserved for audit only. It was not used to define
    # valid index sessions in Step 03/04.
    result["NIFTY_Provider_Volume_Audit_Only"] = volume
    result["NIFTY_Provider_Adj_Close_Audit_Only"] = adj_close

    result.insert(1, "Ticker", cfg.MARKET_INDEX_TICKER)
    result["Instrument_Type"] = "benchmark_index"

    return result


def validate_nifty_panel(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
) -> Dict[str, Any]:
    if len(panel) != len(calendar):
        raise RuntimeError("^NSEI: panel/calendar row-count mismatch.")

    if not panel["Date"].equals(calendar["Date"]):
        raise RuntimeError("^NSEI: panel dates differ from frozen calendar.")

    model_columns = [
        "NIFTY_Model_Open",
        "NIFTY_Model_High",
        "NIFTY_Model_Low",
        "NIFTY_Model_Close",
    ]

    null_total = int(panel[model_columns].isna().sum().sum())

    if null_total != 0:
        raise RuntimeError(
            f"^NSEI: price-model null total = {null_total}"
        )

    for column in (
        "NIFTY_Gross_Return_1session",
        "NIFTY_Return_1session",
        "NIFTY_Log_Return_1session",
    ):
        null_mask = panel[column].isna()

        if int(null_mask.sum()) != 1 or not bool(null_mask.iloc[0]):
            raise RuntimeError(
                f"^NSEI: {column} must have exactly one first-session null."
            )

    high_invalid = (
        panel["NIFTY_Model_High"]
        <
        panel[
            [
                "NIFTY_Model_Open",
                "NIFTY_Model_Low",
                "NIFTY_Model_Close",
            ]
        ].max(axis=1)
        - OHLC_TOLERANCE
    )

    low_invalid = (
        panel["NIFTY_Model_Low"]
        >
        panel[
            [
                "NIFTY_Model_Open",
                "NIFTY_Model_High",
                "NIFTY_Model_Close",
            ]
        ].min(axis=1)
        + OHLC_TOLERANCE
    )

    if high_invalid.any() or low_invalid.any():
        raise RuntimeError("^NSEI: model OHLC geometry violation.")

    return {
        "rows": int(len(panel)),
        "first_date": panel["Date"].min().strftime("%Y-%m-%d"),
        "last_date": panel["Date"].max().strftime("%Y-%m-%d"),
        "price_model_null_total": null_total,
        "provider_volume_zero_rows": int(
            (panel["NIFTY_Provider_Volume_Audit_Only"] == 0).sum()
        ),
        "model_ohlc_geometry_valid": True,
    }


# =============================================================================
# 13. RECONCILIATION AUDITS
# =============================================================================

def reconcile_corporate_actions(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    if not STEP04_ACTION_REGISTER.exists():
        raise RuntimeError(
            f"Missing Step-04 action register: {STEP04_ACTION_REGISTER}"
        )

    step04 = pd.read_csv(
        STEP04_ACTION_REGISTER,
        parse_dates=["Date"],
    )

    step04["Date"] = (
        pd.to_datetime(step04["Date"], errors="raise")
        .dt.normalize()
    )

    if "is_final_analysis_session" not in step04.columns:
        raise RuntimeError(
            "Step-04 action register lacks is_final_analysis_session."
        )

    step04_final_mask = normalize_bool_series(
        step04["is_final_analysis_session"]
    )

    step04_final = (
        step04.loc[step04_final_mask]
        .copy()
        .sort_values(["Date", "ticker"])
        .reset_index(drop=True)
    )

    step05_actions = (
        combined.loc[
            combined["Corporate_Action_Today"].astype(bool),
            [
                "Ticker",
                "Date",
                "Action_Types",
            ],
        ]
        .copy()
        .rename(
            columns={
                "Ticker": "ticker",
                "Action_Types": "action_types_step05",
            }
        )
    )

    step04_subset = step04_final[
        [
            "ticker",
            "Date",
            "action_types",
        ]
    ].rename(
        columns={
            "action_types": "action_types_step04",
        }
    )

    merged = step04_subset.merge(
        step05_actions,
        on=["ticker", "Date"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    merged["step04_final_session"] = (
        merged["_merge"] != "right_only"
    )
    merged["step05_present"] = (
        merged["_merge"] != "left_only"
    )

    merged["match"] = (
        (merged["_merge"] == "both")
        & (
            merged["action_types_step04"].fillna("")
            == merged["action_types_step05"].fillna("")
        )
    )

    if not merged["match"].all():
        bad = merged.loc[~merged["match"]].copy()
        raise RuntimeError(
            "Step-04 / Step-05 corporate-action reconciliation failed:\n"
            + bad.to_string(index=False)
        )

    return (
        merged[
            ACTION_RECONCILIATION_COLUMNS
        ]
        .sort_values(["Date", "ticker"])
        .reset_index(drop=True)
    )


def reconcile_extreme_returns(
    combined: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    current = (
        combined.loc[
            combined["Model_Log_Return_1session"].abs()
            > EXTREME_ABS_LOG_RETURN_THRESHOLD,
            [
                "Ticker",
                "Date",
                "Model_Log_Return_1session",
            ],
        ]
        .copy()
        .rename(
            columns={
                "Ticker": "ticker",
                "Model_Log_Return_1session": "model_log_return",
            }
        )
    )

    current["absolute_model_log_return"] = (
        current["model_log_return"].abs()
    )

    if STEP04_LARGE_RETURN_AUDIT.exists():
        step04 = pd.read_csv(
            STEP04_LARGE_RETURN_AUDIT,
            parse_dates=["Date"],
        )

        if "is_final_analysis_session" in step04.columns:
            final_mask = normalize_bool_series(
                step04["is_final_analysis_session"]
            )
            step04 = step04.loc[final_mask].copy()

        step04 = step04[
            [
                "ticker",
                "Date",
                "adjusted_log_return",
            ]
        ].rename(
            columns={
                "adjusted_log_return": "step04_adjusted_log_return",
            }
        )
    else:
        step04 = pd.DataFrame(
            columns=[
                "ticker",
                "Date",
                "step04_adjusted_log_return",
            ]
        )

    merged = current.merge(
        step04,
        on=["ticker", "Date"],
        how="outer",
        indicator=True,
    )

    merged["step04_flag_present"] = (
        merged["_merge"] != "left_only"
    )

    merged["difference_vs_step04"] = (
        merged["model_log_return"]
        - merged["step04_adjusted_log_return"]
    )

    statuses = []

    for _, row in merged.iterrows():
        merge_state = row["_merge"]

        if merge_state == "both":
            difference = row["difference_vs_step04"]

            if pd.notna(difference) and abs(float(difference)) <= 1e-10:
                status = "MATCH"
            else:
                status = "VALUE_DIFFERS_DUE_TO_RETAINED_CALENDAR_ADJACENCY"
        elif merge_state == "left_only":
            status = "NEW_FLAG_AFTER_FINAL_CALENDAR_COMPRESSION"
        else:
            status = "STEP04_FLAG_NOT_EXTREME_ON_FINAL_CALENDAR"

        statuses.append(status)

    merged["reconciliation_status"] = statuses

    output = merged[
        RETURN_RECONCILIATION_COLUMNS
    ].sort_values(
        ["Date", "ticker"],
        na_position="last",
    ).reset_index(drop=True)

    summary = {
        "step05_extreme_flag_count": int(len(current)),
        "step04_final_session_flag_count": int(len(step04)),
        "exact_match_count": int(
            (output["reconciliation_status"] == "MATCH").sum()
        ),
        "new_flag_count": int(
            (
                output["reconciliation_status"]
                == "NEW_FLAG_AFTER_FINAL_CALENDAR_COMPRESSION"
            ).sum()
        ),
        "missing_old_flag_count": int(
            (
                output["reconciliation_status"]
                == "STEP04_FLAG_NOT_EXTREME_ON_FINAL_CALENDAR"
            ).sum()
        ),
        "value_difference_count": int(
            (
                output["reconciliation_status"]
                == "VALUE_DIFFERS_DUE_TO_RETAINED_CALENDAR_ADJACENCY"
            ).sum()
        ),
    }

    # This is diagnostic only. No row is deleted solely because of return size.
    return output, summary


# =============================================================================
# 14. WRITE / ROUNDTRIP / HASH HELPERS
# =============================================================================

def write_dataframe_pair(
    dataframe: pd.DataFrame,
    *,
    parquet_path: Path,
    csv_path: Path,
) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_parquet(
        parquet_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    csv_frame = dataframe.copy()
    csv_frame["Date"] = (
        pd.to_datetime(csv_frame["Date"])
        .dt.strftime("%Y-%m-%d")
    )

    csv_frame.to_csv(
        csv_path,
        index=False,
        float_format="%.17g",
    )


def verify_parquet_roundtrip(
    original: pd.DataFrame,
    parquet_path: Path,
) -> None:
    reloaded = pd.read_parquet(
        parquet_path,
        engine="pyarrow",
    )

    if len(reloaded) != len(original):
        raise RuntimeError(
            f"Parquet roundtrip row mismatch: {parquet_path}"
        )

    if list(reloaded.columns) != list(original.columns):
        raise RuntimeError(
            f"Parquet roundtrip column mismatch: {parquet_path}"
        )

    if "Date" in original.columns:
        original_dates = pd.to_datetime(
            original["Date"],
            errors="raise",
        ).dt.normalize()

        reloaded_dates = pd.to_datetime(
            reloaded["Date"],
            errors="raise",
        ).dt.normalize()

        if not np.array_equal(
            original_dates.to_numpy(),
            reloaded_dates.to_numpy(),
        ):
            raise RuntimeError(
                f"Parquet roundtrip date mismatch: {parquet_path}"
            )


def staged_output_record(
    *,
    scope: str,
    ticker: str,
    file_type: str,
    staging_path: Path,
    final_path: Path,
) -> Dict[str, Any]:
    return {
        "scope": scope,
        "ticker": ticker,
        "file_type": file_type,
        "file_path": str(final_path),
        "relative_path": relative_to_experiment(final_path),
        "size_bytes": int(staging_path.stat().st_size),
        "sha256": sha256_file(staging_path),
    }


# =============================================================================
# 15. SOURCE-CODE HASH AUDIT
# =============================================================================

def find_frozen_script(
    manifest_path: Path,
    target_name: str,
    *,
    artifact_key: str = "artifacts",
) -> Optional[Path]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    for record in manifest.get(artifact_key, {}).values():
        if not isinstance(record, dict):
            continue

        path_text = record.get("path")
        if not path_text:
            continue

        path = Path(path_text)

        if path.name == target_name:
            return path

    return None


def collect_source_hashes() -> pd.DataFrame:
    current_step00 = Path(cfg.__file__).resolve()
    frozen_step00 = cfg.get_frozen_script_path().resolve()

    frozen_step01 = find_frozen_script(
        STEP01_MANIFEST,
        "step01_environment_audit_frozen.py",
    )

    with STEP02_MANIFEST.open("r", encoding="utf-8") as handle:
        step02_manifest = json.load(handle)

    step02_record = (
        step02_manifest
        .get("canonical_artifacts", {})
        .get("frozen_step02_script")
    )

    frozen_step02 = (
        Path(step02_record["path"])
        if step02_record
        else None
    )

    frozen_step03 = find_frozen_script(
        STEP03_MANIFEST,
        "step03_market_integrity_audit_frozen.py",
    )

    frozen_step04 = find_frozen_script(
        STEP04_MANIFEST,
        "step04_session_and_corporate_action_policy_frozen.py",
    )

    sources: list[tuple[str, Path]] = [
        ("current_step00", current_step00),
        ("frozen_step00", frozen_step00),
        ("current_step05", SCRIPT_FILE.resolve()),
    ]

    for role, path in (
        ("frozen_step01", frozen_step01),
        ("frozen_step02", frozen_step02),
        ("frozen_step03", frozen_step03),
        ("frozen_step04", frozen_step04),
    ):
        if path is not None:
            sources.append((role, path))

    rows = []

    for role, path in sources:
        if not path.exists():
            raise RuntimeError(f"Required source file missing: {path}")

        rows.append(
            {
                "role": role,
                "file_name": path.name,
                "absolute_path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )

    current_step00_sha = next(
        row["sha256"]
        for row in rows
        if row["role"] == "current_step00"
    )

    frozen_step00_sha = next(
        row["sha256"]
        for row in rows
        if row["role"] == "frozen_step00"
    )

    if current_step00_sha != frozen_step00_sha:
        raise RuntimeError(
            "Active step00_config.py differs from the frozen Step-00 source."
        )

    return pd.DataFrame(rows, columns=SOURCE_HASH_COLUMNS)


# =============================================================================
# 16. PANEL-LEVEL VALIDATION
# =============================================================================

def validate_combined_stock_panel(
    combined: pd.DataFrame,
    calendar: pd.DataFrame,
) -> Dict[str, Any]:
    expected_rows = len(calendar) * EXPECTED_STOCK_COUNT

    if len(combined) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} stock rows; "
            f"found {len(combined):,}."
        )

    if combined[["Ticker", "Date"]].duplicated().any():
        raise RuntimeError("Duplicate ticker-date keys in combined panel.")

    counts = (
        combined
        .groupby("Date")["Ticker"]
        .nunique()
    )

    if not (counts == EXPECTED_STOCK_COUNT).all():
        raise RuntimeError(
            "Incomplete stock cross-section on one or more retained sessions."
        )

    calendar_dates = pd.DatetimeIndex(calendar["Date"])
    observed_dates = pd.DatetimeIndex(
        combined["Date"].drop_duplicates().sort_values()
    )

    if not np.array_equal(
        calendar_dates.to_numpy(),
        observed_dates.to_numpy(),
    ):
        raise RuntimeError(
            "Combined stock panel date set differs from frozen calendar."
        )

    model_null_total = int(
        combined[list(MODELLING_STOCK_FIELDS)]
        .isna()
        .sum()
        .sum()
    )

    if model_null_total != 0:
        raise RuntimeError(
            f"Combined Model OHLCV null total = {model_null_total}"
        )

    negative_model_volume = int(
        (combined["Model_Volume"] < 0).sum()
    )

    if negative_model_volume != 0:
        raise RuntimeError(
            f"Negative Model_Volume rows = {negative_model_volume}"
        )

    return {
        "expected_rows": int(expected_rows),
        "observed_rows": int(len(combined)),
        "session_count": int(len(calendar)),
        "stock_count": int(EXPECTED_STOCK_COUNT),
        "min_stocks_per_session": int(counts.min()),
        "max_stocks_per_session": int(counts.max()),
        "model_ohlcv_null_total": model_null_total,
        "negative_model_volume_rows": negative_model_volume,
        "zero_model_volume_rows": int(
            (combined["Model_Volume"] == 0).sum()
        ),
        "duplicate_ticker_date_rows": 0,
    }


# =============================================================================
# 17. SUMMARY / MANIFEST
# =============================================================================

def build_panel_summary(
    *,
    step00: Dict[str, Any],
    step01: Dict[str, Any],
    step02: Dict[str, Any],
    step03: Dict[str, Any],
    step04: Dict[str, Any],
    calendar: pd.DataFrame,
    stock_audit: pd.DataFrame,
    combined_validation: Dict[str, Any],
    nifty_validation: Dict[str, Any],
    action_reconciliation: pd.DataFrame,
    return_reconciliation_summary: Dict[str, Any],
    source_hashes: pd.DataFrame,
    lock_sha_after: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "step": {
            "step_number": STEP_NUMBER,
            "step_id": STEP_ID,
            "step_name": STEP_NAME,
            "status": "PASS",
        },
        "generated_at_utc": utc_now_iso(),
        "experiment_tag": cfg.EXPERIMENT_TAG,
        "configuration_fingerprint_sha256": (
            step00["configuration_fingerprint_sha256"]
        ),
        "upstream_chain": {
            "step00_manifest_sha256": step00["step00_manifest_sha256"],
            "step01_manifest_sha256": step01["manifest_sha256"],
            "step02_manifest_sha256": step02["manifest_sha256"],
            "step03_manifest_sha256": step03["manifest_sha256"],
            "step04_manifest_sha256": step04["manifest_sha256"],
        },
        "signal_timing_policy": SIGNAL_TIMING_POLICY,
        "calendar": {
            "sessions": int(len(calendar)),
            "first_session": calendar["Date"].min().strftime("%Y-%m-%d"),
            "last_session": calendar["Date"].max().strftime("%Y-%m-%d"),
        },
        "stock_panel": combined_validation,
        "nifty_panel": nifty_validation,
        "representation_policy": {
            "step04_canonical_adjusted_ohlc_materialized": True,
            "canonical_adjusted_levels_role": "audit_reference_only",
            "provider_adj_close_level_used_as_model_level": False,
            "model_return_formula": (
                "AdjClose_t / AdjClose_previous_retained_session - 1"
            ),
            "model_log_return_formula": (
                "log(AdjClose_t / AdjClose_previous_retained_session)"
            ),
            "model_close": (
                "forward synthetic total-return index accumulated from "
                "adjacent adjusted-close ratios with base 100"
            ),
            "model_open_high_low": (
                "Model_Close multiplied by same-session raw O/C, H/C, L/C ratios"
            ),
            "model_volume": "Yahoo provider Volume exactly as supplied",
            "manual_split_volume_adjustment": False,
            "volume_imputation": False,
            "future_split_factor_explicitly_used_in_model_level": False,
            "future_dividend_factor_explicitly_used_in_model_level": False,
        },
        "corporate_action_reconciliation": {
            "retained_action_rows": int(len(action_reconciliation)),
            "all_rows_match_step04": bool(
                action_reconciliation["match"].all()
                if not action_reconciliation.empty
                else True
            ),
        },
        "return_reconciliation": return_reconciliation_summary,
        "stock_audit_records": stock_audit.to_dict(orient="records"),
        "source_code_hashes": source_hashes.to_dict(orient="records"),
        "scientific_boundary": {
            "technical_indicators_constructed": False,
            "frozen_47_stock_predictors_constructed": False,
            "frozen_16_nifty_predictors_constructed": False,
            "forecast_targets_constructed": False,
            "scalers_fit": False,
            "regimes_estimated": False,
            "sequences_constructed": False,
            "model_selection_performed": False,
            "threshold_selection_performed": False,
            "final_test_scoring_performed": False,
        },
        "final_test_lock_integrity": {
            "state": step00["final_test_state"],
            "lock_sha256_before": step00["final_test_lock_sha256"],
            "lock_sha256_after": lock_sha_after,
            "lock_unchanged": bool(
                lock_sha_after == step00["final_test_lock_sha256"]
            ),
        },
    }


def artifact_record(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": relative_to_experiment(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def step05_manifest_artifacts() -> tuple[Path, ...]:
    return (
        COMBINED_STOCK_PARQUET,
        COMBINED_STOCK_CSV,
        NIFTY_PARQUET,
        NIFTY_CSV,
        STOCK_AUDIT_CSV,
        ACTION_RECONCILIATION_CSV,
        RETURN_RECONCILIATION_CSV,
        OUTPUT_HASH_REGISTER_CSV,
        SOURCE_HASH_CSV,
        PANEL_SUMMARY_JSON,
        FROZEN_STEP05_SCRIPT,
        STEP05_SUMMARY_TXT,
    )


def write_step05_manifest(
    *,
    step00: Dict[str, Any],
    step01: Dict[str, Any],
    step02: Dict[str, Any],
    step03: Dict[str, Any],
    step04: Dict[str, Any],
    individual_stock_files: list[Dict[str, Any]],
) -> str:
    artifacts = {
        path.name: artifact_record(path)
        for path in step05_manifest_artifacts()
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "step": {
            "step_number": STEP_NUMBER,
            "step_id": STEP_ID,
            "step_name": STEP_NAME,
            "status": "PASS",
        },
        "frozen_at_utc": utc_now_iso(),
        "experiment_tag": cfg.EXPERIMENT_TAG,
        "configuration_fingerprint_sha256": (
            step00["configuration_fingerprint_sha256"]
        ),
        "step00_manifest_sha256": step00["step00_manifest_sha256"],
        "step01_manifest_sha256": step01["manifest_sha256"],
        "step02_manifest_sha256": step02["manifest_sha256"],
        "step03_manifest_sha256": step03["manifest_sha256"],
        "step04_manifest_sha256": step04["manifest_sha256"],
        "final_test_state": step00["final_test_state"],
        "individual_stock_output_records": individual_stock_files,
        "artifacts": artifacts,
    }

    atomic_write_json(STEP05_MANIFEST_JSON, manifest)
    manifest_sha = sha256_file(STEP05_MANIFEST_JSON)

    atomic_write_text(
        STEP05_MANIFEST_SHA,
        f"{manifest_sha}  {STEP05_MANIFEST_JSON.name}\n",
    )

    return manifest_sha


# =============================================================================
# 18. VERIFY EXISTING STEP 05
# =============================================================================

def verify_existing_step05() -> int:
    print("=" * 124)
    print("STEP 05 — VERIFY EXISTING CLEAN MODEL-PANEL FREEZE")
    print("=" * 124)

    step00 = verify_step00_contract()
    step01 = verify_step01(
        step00["configuration_fingerprint_sha256"]
    )
    step02 = verify_step02(
        step00["configuration_fingerprint_sha256"],
        step01["manifest_sha256"],
    )
    step03 = verify_step03(
        step00["configuration_fingerprint_sha256"],
        step01["manifest_sha256"],
        step02["manifest_sha256"],
    )
    step04 = verify_step04(
        step00["configuration_fingerprint_sha256"],
        step01["manifest_sha256"],
        step02["manifest_sha256"],
        step03["manifest_sha256"],
    )

    if not STEP05_MANIFEST_JSON.exists():
        print("[FAIL] Step-05 manifest is missing.")
        return 20

    if not STEP05_MANIFEST_SHA.exists():
        print("[FAIL] Step-05 manifest SHA sidecar is missing.")
        return 21

    actual_manifest_sha = verify_sha_sidecar(
        STEP05_MANIFEST_JSON,
        STEP05_MANIFEST_SHA,
        "Step 05",
    )

    with STEP05_MANIFEST_JSON.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("step", {}).get("status") != "PASS":
        print("[FAIL] Stored Step-05 status is not PASS.")
        return 22

    expected_chain = {
        "configuration_fingerprint_sha256": (
            step00["configuration_fingerprint_sha256"]
        ),
        "step01_manifest_sha256": step01["manifest_sha256"],
        "step02_manifest_sha256": step02["manifest_sha256"],
        "step03_manifest_sha256": step03["manifest_sha256"],
        "step04_manifest_sha256": step04["manifest_sha256"],
    }

    for field, expected in expected_chain.items():
        if manifest.get(field) != expected:
            print(f"[FAIL] Step-05 chain mismatch: {field}")
            return 23

    failures = []

    for name, record in manifest.get("artifacts", {}).items():
        path = Path(record["path"])

        if not path.exists():
            failures.append(f"missing: {name}")
            continue

        if sha256_file(path) != record["sha256"]:
            failures.append(f"hash mismatch: {name}")

    for record in manifest.get("individual_stock_output_records", []):
        path = Path(record["file_path"])

        if not path.exists():
            failures.append(f"missing stock output: {path}")
            continue

        if sha256_file(path) != record["sha256"]:
            failures.append(f"stock output hash mismatch: {path}")

    if failures:
        for item in failures:
            print(f"[FAIL] {item}")

        print("=" * 124)
        print("STEP 05 VERIFICATION STATUS: FAIL")
        print("=" * 124)
        return 24

    print(
        "[PASS] Configuration fingerprint : "
        f"{step00['configuration_fingerprint_sha256']}"
    )
    print(
        "[PASS] Step-01 manifest          : "
        f"{step01['manifest_sha256']}"
    )
    print(
        "[PASS] Step-02 manifest          : "
        f"{step02['manifest_sha256']}"
    )
    print(
        "[PASS] Step-03 manifest          : "
        f"{step03['manifest_sha256']}"
    )
    print(
        "[PASS] Step-04 manifest          : "
        f"{step04['manifest_sha256']}"
    )
    print(
        "[PASS] Step-05 manifest          : "
        f"{actual_manifest_sha}"
    )
    print(
        "[PASS] Final-test state          : "
        f"{step00['final_test_state']}"
    )
    print("=" * 124)
    print("STEP 05 VERIFICATION STATUS: PASS")
    print("=" * 124)

    return 0


# =============================================================================
# 19. FAILURE DIAGNOSTIC
# =============================================================================

def write_failure_diagnostic(
    *,
    stage: str,
    error: Exception,
    staging_root: Optional[Path],
    archived_partial: Optional[Path],
) -> Path:
    FAILED_ATTEMPT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = (
        FAILED_ATTEMPT_DIR
        / f"05_clean_model_panel_FAILED_{timestamp}.json"
    )

    payload = {
        "step": STEP_NAME,
        "status": "FAIL",
        "failed_at_utc": utc_now_iso(),
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "python_executable": sys.executable,
        "staging_root": (
            str(staging_root)
            if staging_root is not None
            else None
        ),
        "archived_partial_artifacts": (
            str(archived_partial)
            if archived_partial is not None
            else None
        ),
        "canonical_manifest_exists": STEP05_MANIFEST_JSON.exists(),
    }

    atomic_write_json(path, payload)
    return path


# =============================================================================
# 20. MAIN MATERIALIZATION
# =============================================================================

def run_new_materialization() -> int:
    print("=" * 124)
    print(STEP_NAME)
    print("=" * 124)

    stage = "prerequisite_verification"
    staging_root: Optional[Path] = None

    try:
        print("[1/11] Verifying Step 00 and final-test lock...")
        step00 = verify_step00_contract()

        print(
            "       Configuration fingerprint: "
            f"{step00['configuration_fingerprint_sha256']}"
        )
        print(
            "       Final-test state: "
            f"{step00['final_test_state']}"
        )

        print("[2/11] Verifying frozen Step-01 environment...")
        step01 = verify_step01(
            step00["configuration_fingerprint_sha256"]
        )
        print(
            "       Step-01 manifest: "
            f"{step01['manifest_sha256']}"
        )
        print(
            "       Python executable: "
            f"{sys.executable}"
        )

        print("[3/11] Verifying frozen Step-02 raw acquisition...")
        step02 = verify_step02(
            step00["configuration_fingerprint_sha256"],
            step01["manifest_sha256"],
        )
        print(
            "       Step-02 manifest: "
            f"{step02['manifest_sha256']}"
        )
        print(
            "       Raw files verified: "
            f"{step02['verified_raw_file_count']}"
        )

        print("[4/11] Verifying Step-03 and Step-04 integrity/policy freezes...")
        step03 = verify_step03(
            step00["configuration_fingerprint_sha256"],
            step01["manifest_sha256"],
            step02["manifest_sha256"],
        )
        step04 = verify_step04(
            step00["configuration_fingerprint_sha256"],
            step01["manifest_sha256"],
            step02["manifest_sha256"],
            step03["manifest_sha256"],
        )

        print(
            "       Step-03 manifest: "
            f"{step03['manifest_sha256']}"
        )
        print(
            "       Step-04 manifest: "
            f"{step04['manifest_sha256']}"
        )

        stage = "output_collision_guard"

        print("[5/11] Checking immutable Step-05 output target...")
        assert_new_step05_target()
        cfg.create_directories()

        stage = "load_final_calendar"

        print("[6/11] Loading frozen Step-04 final analysis calendar...")
        calendar = load_final_calendar(
            step04["policy"]
        )

        print(
            "       Sessions: "
            f"{len(calendar):,}"
        )
        print(
            "       Date range: "
            f"{calendar['Date'].min().date()} -> "
            f"{calendar['Date'].max().date()}"
        )

        stage = "stock_materialization"

        print("[7/11] Materializing 20 stock panels in staging area...")
        staging_root = create_staging_root()
        staging_stock_dir = staging_root / "stocks"
        staging_benchmark_dir = staging_root / "benchmark"

        stock_panels: Dict[str, pd.DataFrame] = {}
        stock_audit_rows = []
        staged_hash_records: list[Dict[str, Any]] = []

        for number, ticker in enumerate(
            cfg.STOCK_TICKERS,
            start=1,
        ):
            print(
                f"       [{number:02d}/{EXPECTED_STOCK_COUNT:02d}] "
                f"{ticker}"
            )

            raw = load_raw_frame(ticker)

            aligned = align_to_calendar(
                ticker=ticker,
                raw=raw,
                calendar=calendar,
                stock=True,
            )

            panel = materialize_stock_panel(
                ticker=ticker,
                aligned=aligned,
            )

            audit = validate_stock_panel(
                ticker=ticker,
                panel=panel,
                calendar=calendar,
            )

            stock_panels[ticker] = panel
            stock_audit_rows.append(audit)

            safe = safe_ticker_filename(ticker)

            staged_parquet = (
                staging_stock_dir
                / f"{safe}_panel.parquet"
            )

            staged_csv = (
                staging_stock_dir
                / f"{safe}_panel.csv"
            )

            final_parquet = (
                STOCK_OUTPUT_DIR
                / f"{safe}_panel.parquet"
            )

            final_csv = (
                STOCK_OUTPUT_DIR
                / f"{safe}_panel.csv"
            )

            write_dataframe_pair(
                panel,
                parquet_path=staged_parquet,
                csv_path=staged_csv,
            )

            verify_parquet_roundtrip(
                panel,
                staged_parquet,
            )

            staged_hash_records.extend(
                [
                    staged_output_record(
                        scope="individual_stock",
                        ticker=ticker,
                        file_type="parquet",
                        staging_path=staged_parquet,
                        final_path=final_parquet,
                    ),
                    staged_output_record(
                        scope="individual_stock",
                        ticker=ticker,
                        file_type="csv",
                        staging_path=staged_csv,
                        final_path=final_csv,
                    ),
                ]
            )

        stock_audit = pd.DataFrame(
            stock_audit_rows,
            columns=STOCK_AUDIT_COLUMNS,
        )

        stage = "combined_and_benchmark_materialization"

        print("[8/11] Building combined stock panel and NIFTY panel...")

        combined = pd.concat(
            [
                stock_panels[ticker]
                for ticker in cfg.STOCK_TICKERS
            ],
            ignore_index=True,
        )

        combined = (
            combined
            .sort_values(
                [
                    "Date",
                    "Ticker",
                ]
            )
            .reset_index(drop=True)
        )

        combined_validation = validate_combined_stock_panel(
            combined,
            calendar,
        )

        staged_combined_parquet = (
            staging_root / "05_stock_panel.parquet"
        )
        staged_combined_csv = (
            staging_root / "05_stock_panel.csv"
        )

        write_dataframe_pair(
            combined,
            parquet_path=staged_combined_parquet,
            csv_path=staged_combined_csv,
        )

        verify_parquet_roundtrip(
            combined,
            staged_combined_parquet,
        )

        staged_hash_records.extend(
            [
                staged_output_record(
                    scope="combined_stock_panel",
                    ticker="ALL_STOCKS",
                    file_type="parquet",
                    staging_path=staged_combined_parquet,
                    final_path=COMBINED_STOCK_PARQUET,
                ),
                staged_output_record(
                    scope="combined_stock_panel",
                    ticker="ALL_STOCKS",
                    file_type="csv",
                    staging_path=staged_combined_csv,
                    final_path=COMBINED_STOCK_CSV,
                ),
            ]
        )

        raw_nifty = load_raw_frame(
            cfg.MARKET_INDEX_TICKER
        )

        aligned_nifty = align_to_calendar(
            ticker=cfg.MARKET_INDEX_TICKER,
            raw=raw_nifty,
            calendar=calendar,
            stock=False,
        )

        nifty = materialize_nifty_panel(
            aligned_nifty
        )

        nifty_validation = validate_nifty_panel(
            nifty,
            calendar,
        )

        staged_nifty_parquet = (
            staging_benchmark_dir
            / "05_nifty50_panel.parquet"
        )

        staged_nifty_csv = (
            staging_benchmark_dir
            / "05_nifty50_panel.csv"
        )

        write_dataframe_pair(
            nifty,
            parquet_path=staged_nifty_parquet,
            csv_path=staged_nifty_csv,
        )

        verify_parquet_roundtrip(
            nifty,
            staged_nifty_parquet,
        )

        staged_hash_records.extend(
            [
                staged_output_record(
                    scope="benchmark",
                    ticker=cfg.MARKET_INDEX_TICKER,
                    file_type="parquet",
                    staging_path=staged_nifty_parquet,
                    final_path=NIFTY_PARQUET,
                ),
                staged_output_record(
                    scope="benchmark",
                    ticker=cfg.MARKET_INDEX_TICKER,
                    file_type="csv",
                    staging_path=staged_nifty_csv,
                    final_path=NIFTY_CSV,
                ),
            ]
        )

        stage = "reconciliation"

        print("[9/11] Reconciling corporate actions and extreme returns...")
        action_reconciliation = reconcile_corporate_actions(
            combined
        )

        (
            return_reconciliation,
            return_reconciliation_summary,
        ) = reconcile_extreme_returns(
            combined
        )

        print(
            "       Retained corporate-action rows: "
            f"{len(action_reconciliation):,}"
        )
        print(
            "       Step-05 extreme |log-return| flags: "
            f"{return_reconciliation_summary['step05_extreme_flag_count']}"
        )
        print(
            "       Exact Step-04 return-flag matches: "
            f"{return_reconciliation_summary['exact_match_count']}"
        )

        stage = "promote_and_freeze"

        print("[10/11] Promoting staged panel and freezing Step-05 artifacts...")

        source_hashes = collect_source_hashes()

        # Promote data root atomically.
        if OUTPUT_ROOT.exists():
            raise RuntimeError(
                "Canonical Step-05 output root appeared unexpectedly."
            )

        os.replace(
            staging_root,
            OUTPUT_ROOT,
        )
        staging_root = None

        # Re-hash canonical files after promotion.
        for record in staged_hash_records:
            canonical_path = Path(record["file_path"])

            if not canonical_path.exists():
                raise RuntimeError(
                    f"Promoted output missing: {canonical_path}"
                )

            if sha256_file(canonical_path) != record["sha256"]:
                raise RuntimeError(
                    f"Output hash changed during promotion: "
                    f"{canonical_path}"
                )

        write_dataframe_csv(
            STOCK_AUDIT_CSV,
            stock_audit,
            STOCK_AUDIT_COLUMNS,
        )

        write_dataframe_csv(
            ACTION_RECONCILIATION_CSV,
            action_reconciliation,
            ACTION_RECONCILIATION_COLUMNS,
        )

        write_dataframe_csv(
            RETURN_RECONCILIATION_CSV,
            return_reconciliation,
            RETURN_RECONCILIATION_COLUMNS,
        )

        output_hash_frame = pd.DataFrame(
            staged_hash_records,
            columns=OUTPUT_HASH_COLUMNS,
        )

        write_dataframe_csv(
            OUTPUT_HASH_REGISTER_CSV,
            output_hash_frame,
            OUTPUT_HASH_COLUMNS,
        )

        write_dataframe_csv(
            SOURCE_HASH_CSV,
            source_hashes,
            SOURCE_HASH_COLUMNS,
        )

        lock_sha_after = sha256_file(
            cfg.get_final_test_lock_path()
        )

        if (
            lock_sha_after
            != step00["final_test_lock_sha256"]
        ):
            raise RuntimeError(
                "Final-test lock changed during Step 05."
            )

        panel_summary = build_panel_summary(
            step00=step00,
            step01=step01,
            step02=step02,
            step03=step03,
            step04=step04,
            calendar=calendar,
            stock_audit=stock_audit,
            combined_validation=combined_validation,
            nifty_validation=nifty_validation,
            action_reconciliation=action_reconciliation,
            return_reconciliation_summary=(
                return_reconciliation_summary
            ),
            source_hashes=source_hashes,
            lock_sha_after=lock_sha_after,
        )

        atomic_write_json(
            PANEL_SUMMARY_JSON,
            panel_summary,
        )

        shutil.copy2(
            SCRIPT_FILE,
            FROZEN_STEP05_SCRIPT,
        )

        summary_lines = [
            STEP_NAME,
            "=" * 92,
            "Status: PASS",
            (
                "Configuration fingerprint: "
                + step00[
                    "configuration_fingerprint_sha256"
                ]
            ),
            (
                "Analysis sessions: "
                f"{len(calendar)}"
            ),
            (
                "Stock count: "
                f"{EXPECTED_STOCK_COUNT}"
            ),
            (
                "Combined stock rows: "
                f"{len(combined)}"
            ),
            (
                "Expected stock rows: "
                f"{len(calendar) * EXPECTED_STOCK_COUNT}"
            ),
            (
                "NIFTY rows: "
                f"{len(nifty)}"
            ),
            (
                "Model OHLCV null total: "
                f"{combined_validation['model_ohlcv_null_total']}"
            ),
            (
                "Zero provider/model volume rows retained: "
                f"{combined_validation['zero_model_volume_rows']}"
            ),
            (
                "Provider volume transformed: False"
            ),
            (
                "Provider Adj Close level used as Model_Close: False"
            ),
            (
                "Step-04 canonical adjusted OHLC: materialized audit-only"
            ),
            (
                "Final-test state: "
                f"{step00['final_test_state']}"
            ),
        ]

        atomic_write_text(
            STEP05_SUMMARY_TXT,
            "\n".join(summary_lines) + "\n",
        )

        individual_stock_records = [
            record
            for record in staged_hash_records
            if record["scope"] == "individual_stock"
        ]

        manifest_sha = write_step05_manifest(
            step00=step00,
            step01=step01,
            step02=step02,
            step03=step03,
            step04=step04,
            individual_stock_files=individual_stock_records,
        )

        stage = "console_report"

        print("[11/11] Final Step-05 report...")
        print()
        print("-" * 124)
        print("FINAL MATERIALIZED PANEL")
        print("-" * 124)
        print(
            "Analysis sessions              : "
            f"{len(calendar):,}"
        )
        print(
            "Stocks                         : "
            f"{EXPECTED_STOCK_COUNT}"
        )
        print(
            "Combined stock rows            : "
            f"{len(combined):,}"
        )
        print(
            "Expected stock rows            : "
            f"{len(calendar) * EXPECTED_STOCK_COUNT:,}"
        )
        print(
            "NIFTY rows                     : "
            f"{len(nifty):,}"
        )
        print(
            "Min stocks/session             : "
            f"{combined_validation['min_stocks_per_session']}"
        )
        print(
            "Max stocks/session             : "
            f"{combined_validation['max_stocks_per_session']}"
        )
        print(
            "Model OHLCV null total         : "
            f"{combined_validation['model_ohlcv_null_total']}"
        )
        print(
            "Negative Model_Volume rows     : "
            f"{combined_validation['negative_model_volume_rows']}"
        )
        print(
            "Zero Model_Volume rows         : "
            f"{combined_validation['zero_model_volume_rows']}"
        )

        print("-" * 124)
        print("REPRESENTATION POLICY")
        print("-" * 124)
        print(
            "Canonical AdjClose/Close OHLC  : "
            "MATERIALIZED — AUDIT/REFERENCE ONLY"
        )
        print(
            "Provider Adj Close LEVEL       : "
            "NOT USED AS MODEL LEVEL"
        )
        print(
            "Model return                   : "
            "adjacent Adj Close ratio"
        )
        print(
            "Model price level              : "
            "forward synthetic return index (base 100)"
        )
        print(
            "Model intraday geometry        : "
            "same-session raw O/C, H/C, L/C ratios"
        )
        print(
            "Model Volume                   : "
            "provider Volume unchanged"
        )
        print(
            "Manual split-volume adjustment : "
            "NONE"
        )
        print(
            "Signal timing                  : "
            f"{SIGNAL_TIMING_POLICY}"
        )

        print("-" * 124)
        print("STOCK AUDIT")
        print("-" * 124)
        print(
            f"{'Ticker':<17}"
            f"{'Rows':>7}"
            f"{'Splits':>8}"
            f"{'Div':>7}"
            f"{'RetMin':>13}"
            f"{'RetMax':>13}"
            f"{'ZeroVol':>9}"
        )
        print("-" * 80)

        for row in stock_audit.itertuples(index=False):
            print(
                f"{row.ticker:<17}"
                f"{row.rows:>7}"
                f"{row.split_event_rows:>8}"
                f"{row.dividend_event_rows:>7}"
                f"{row.model_return_min:>13.6f}"
                f"{row.model_return_max:>13.6f}"
                f"{row.zero_model_volume_rows:>9}"
            )

        print("-" * 124)
        print(
            "Retained action rows reconciled : "
            f"{len(action_reconciliation):,}"
        )
        print(
            "Extreme return flags (Step 05)  : "
            f"{return_reconciliation_summary['step05_extreme_flag_count']}"
        )
        print(
            "Exact Step-04 flag matches       : "
            f"{return_reconciliation_summary['exact_match_count']}"
        )
        print(
            "New flags after calendar removal : "
            f"{return_reconciliation_summary['new_flag_count']}"
        )
        print(
            "Old flags no longer extreme      : "
            f"{return_reconciliation_summary['missing_old_flag_count']}"
        )
        print(
            "Value-different matched dates     : "
            f"{return_reconciliation_summary['value_difference_count']}"
        )

        print("-" * 124)
        print("OUTPUT FILES")
        print("-" * 124)
        print(
            f"Combined stock Parquet : {COMBINED_STOCK_PARQUET}"
        )
        print(
            f"Combined stock CSV     : {COMBINED_STOCK_CSV}"
        )
        print(
            f"NIFTY Parquet          : {NIFTY_PARQUET}"
        )
        print(
            f"NIFTY CSV              : {NIFTY_CSV}"
        )
        print(
            f"Stock audit            : {STOCK_AUDIT_CSV}"
        )
        print(
            f"Action reconciliation  : {ACTION_RECONCILIATION_CSV}"
        )
        print(
            f"Return reconciliation  : {RETURN_RECONCILIATION_CSV}"
        )
        print(
            f"Output hashes          : {OUTPUT_HASH_REGISTER_CSV}"
        )
        print(
            f"Panel summary          : {PANEL_SUMMARY_JSON}"
        )
        print(
            f"Step-05 manifest       : {STEP05_MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256       : {manifest_sha}"
        )
        print(
            f"Final-test state       : {step00['final_test_state']}"
        )

        print("=" * 124)
        print(
            "PASS: Step-00 through Step-04 chain integrity verified."
        )
        print(
            "PASS: Final clean stock panel has the exact frozen calendar "
            "and complete 20-stock cross-section."
        )
        print(
            "PASS: Step-04 canonical adjusted OHLC was materialized for "
            "audit/reference without using its retrospective level as Model_Close."
        )
        print(
            "PASS: Modelling price levels use adjacent adjusted-return "
            "increments accumulated forward from base 100."
        )
        print(
            "PASS: Provider Volume is unchanged; no manual split adjustment "
            "or volume imputation was applied."
        )
        print(
            "PASS: No technical indicators, targets, scalers, regimes, "
            "sequences, models, or thresholds were constructed."
        )
        print(
            "PASS: Final held-out test remains LOCKED_NOT_OPENED."
        )
        print("=" * 124)
        print(
            "Do not run Step 06 yet. Paste the complete terminal output for audit."
        )

        return 0

    except Exception as exc:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(
                staging_root,
                ignore_errors=True,
            )

        archived_partial = archive_partial_outputs()

        failure_path = write_failure_diagnostic(
            stage=stage,
            error=exc,
            staging_root=staging_root,
            archived_partial=archived_partial,
        )

        print()
        print("=" * 124)
        print("STEP 05 FAILED — PANEL NOT FROZEN")
        print("=" * 124)
        print(f"Stage      : {stage}")
        print(f"Error type : {type(exc).__name__}")
        print(f"Error      : {exc}")
        print(f"Diagnostic : {failure_path}")

        if archived_partial is not None:
            print(
                "Archived partial artifacts: "
                f"{archived_partial}"
            )

        print("=" * 124)
        print("Do not proceed to Step 06.")

        return 30


# =============================================================================
# 21. CLI / ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and freeze the single clean V3 Step-05 market panel."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an already frozen Step-05 panel without recomputing it."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verify_only:
        return verify_existing_step05()

    return run_new_materialization()


if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print("\nStep 05 interrupted by user.")
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 124)
        print("STEP 05 UNHANDLED FAILURE")
        print("=" * 124)
        print(f"{type(exc).__name__}: {exc}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        print("=" * 124)
        sys.exit(1)
