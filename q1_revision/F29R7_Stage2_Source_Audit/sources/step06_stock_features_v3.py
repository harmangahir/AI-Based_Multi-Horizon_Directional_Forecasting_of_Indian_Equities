#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 06 — CAUSAL 47-FEATURE STOCK TECHNICAL PANEL (CLEAN V3)

Purpose
-------
Construct the frozen 47 stock-specific continuous predictors from the single
verified Step-05 clean model panel.

Canonical modelling source
--------------------------
processed_data/step05_clean_model_panel/05_stock_panel.parquet

The script DOES NOT use the retrospective audit-only adjusted-price levels as
its modelling source. It uses only the Step-05 Model_* representation.

Continuity policy
-----------------
Step 04 removed a valid NIFTY reference session when it satisfied the frozen
date-wide anomaly rule. Technical indicators MUST NOT bridge that missing valid
market session as though the previous and next retained dates were consecutive
sessions. Therefore all recursive/rolling/cumulative technical calculations
restart after each valid-reference session excluded by the frozen Step-04 rule.

For the present frozen experiment, this policy yields two continuity segments
per stock around the excluded valid session 2025-03-18. The date itself is NOT
hard-coded; it is read from the verified Step-04 cleaning policy.

Scientific boundary
-------------------
This stage:
- constructs exactly 47 stock features;
- uses only same-session or historical information;
- performs no scaling;
- creates no targets;
- estimates no regimes;
- creates no train/validation/test samples;
- builds no model sequences;
- trains/selects no models;
- tunes no thresholds;
- does not score the held-out final test.

Signal timing
-------------
Features for session t are available only after session t closes.

Run
---
python step06_stock_features_v3.py

Verify frozen Step 06
---------------------
python step06_stock_features_v3.py --verify-only
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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. SCRIPT / CONFIG LOCATION
# =============================================================================

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent

if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError(
        "step06_stock_features_v3.py must be located inside "
        "<repository>/q1_revision/.\n"
        f"Current script directory: {SCRIPT_DIR}"
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import step00_config as cfg
except Exception as exc:
    raise RuntimeError(
        "Could not import q1_revision/step00_config.py."
    ) from exc


# =============================================================================
# 1. STEP CONSTANTS
# =============================================================================

STEP_NUMBER = 6
STEP_ID = "06"
STEP_NAME = "STEP 06 — CAUSAL 47-FEATURE STOCK TECHNICAL PANEL"
SCHEMA_VERSION = "1.0"

EXPECTED_STOCK_COUNT = len(cfg.STOCK_TICKERS)
EXPECTED_FEATURE_COUNT = 47
EXPECTED_FIRST_COMPLETE_POSITION = 33

SIGNAL_TIMING_POLICY = (
    "features_at_session_t_are_available_after_session_t_close"
)

FEATURE_COLUMNS_REQUIRED_FROM_STEP05 = (
    "session_number",
    "Date",
    "Ticker",
    "Model_Open",
    "Model_High",
    "Model_Low",
    "Model_Close",
    "Model_Volume",
    "Model_Return_1session",
    "Model_Log_Return_1session",
)


# =============================================================================
# 2. FROZEN INPUT PATHS
# =============================================================================

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
STEP04_CLASSIFICATION = cfg.AUDIT_DIR / "04_session_classification.csv"

STEP05_MANIFEST = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.json"
STEP05_MANIFEST_SHA = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.sha256"
STEP05_PANEL = (
    cfg.PROCESSED_DATA_DIR
    / "step05_clean_model_panel"
    / "05_stock_panel.parquet"
)


# =============================================================================
# 3. STEP-06 OUTPUT PATHS
# =============================================================================

STEP06_OUTPUT_DIR = cfg.FEATURE_DATA_DIR / "step06_stock_features"

FEATURE_PANEL_PARQUET = STEP06_OUTPUT_DIR / "06_stock_features_47.parquet"
FEATURE_PANEL_CSV = STEP06_OUTPUT_DIR / "06_stock_features_47.csv"

COMPLETE_FEATURE_PANEL_PARQUET = (
    STEP06_OUTPUT_DIR / "06_stock_features_47_complete.parquet"
)
COMPLETE_FEATURE_PANEL_CSV = (
    STEP06_OUTPUT_DIR / "06_stock_features_47_complete.csv"
)

FEATURE_DICTIONARY_CSV = cfg.AUDIT_DIR / "06_stock_feature_dictionary.csv"
SEGMENT_AUDIT_CSV = cfg.AUDIT_DIR / "06_stock_feature_segment_audit.csv"
WARMUP_AUDIT_CSV = cfg.AUDIT_DIR / "06_stock_feature_warmup_audit.csv"
CAUSALITY_AUDIT_CSV = cfg.AUDIT_DIR / "06_stock_feature_prefix_causality_audit.csv"
RETURN_CONSISTENCY_CSV = cfg.AUDIT_DIR / "06_stock_feature_return_consistency.csv"
STOCK_SUMMARY_CSV = cfg.AUDIT_DIR / "06_stock_feature_stock_summary.csv"
HASH_REGISTER_CSV = cfg.AUDIT_DIR / "06_stock_feature_hashes.csv"
SOURCE_HASH_CSV = cfg.AUDIT_DIR / "06_source_file_hashes.csv"
STEP06_SUMMARY_JSON = cfg.AUDIT_DIR / "06_stock_feature_summary.json"

MANIFEST_JSON = cfg.MANIFEST_DIR / "06_stock_feature_manifest.json"
MANIFEST_SHA = cfg.MANIFEST_DIR / "06_stock_feature_manifest.sha256"
FROZEN_STEP06_SCRIPT = cfg.MANIFEST_DIR / "step06_stock_features_frozen.py"
SUMMARY_TXT = cfg.REPORT_DIR / "06_stock_feature_summary.txt"

FAILED_ATTEMPT_DIR = cfg.AUDIT_DIR / "step06_failed_attempts"


# =============================================================================
# 4. LOCKED 47-FEATURE SPECIFICATION
# =============================================================================

STOCK_FEATURE_GROUPS = {

    "OHLCV_and_return": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "log_return_1d",
        "high_low_range",
        "open_close_change",
        "volume_change",
        "return_lag_1",
        "return_lag_2",
        "return_lag_3",
        "return_lag_5",
    ],

    "rolling_trend_and_volatility": [
        "rolling_mean_5",
        "rolling_mean_10",
        "rolling_mean_20",
        "rolling_std_5",
        "rolling_std_10",
        "rolling_std_20",
        "rolling_min_20",
        "rolling_max_20",
    ],

    "moving_average_and_momentum": [
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "stoch_k_14",
        "stoch_d_3",
    ],

    "band_range_and_trend_strength": [
        "bollinger_upper_20",
        "bollinger_lower_20",
        "bollinger_width_20",
        "atr_14",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "aroon_up_25",
        "aroon_down_25",
        "aroon_osc_25",
    ],

    "volume_and_relative_position": [
        "obv",
        "ad_line",
        "vwap_20",
        "volume_sma_20",
        "close_to_rolling_mean_20",
        "close_to_rolling_max_20",
        "close_to_rolling_min_20",
    ],
}

FEATURE_COLUMNS = [
    feature
    for group_features in STOCK_FEATURE_GROUPS.values()
    for feature in group_features
]

if len(FEATURE_COLUMNS) != EXPECTED_FEATURE_COUNT:
    raise AssertionError(
        f"Expected {EXPECTED_FEATURE_COUNT} features; "
        f"found {len(FEATURE_COLUMNS)}."
    )

if len(set(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
    raise AssertionError("Duplicate feature names detected.")


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def get_distribution_version(distribution_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def relative_to_experiment(path: Path) -> str:
    return str(
        path.resolve().relative_to(cfg.EXPERIMENT_ROOT_DIR.resolve())
    )


def artifact_record(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": relative_to_experiment(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


# =============================================================================
# 6. STEP-00 / FINAL-TEST LOCK VERIFICATION
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
# 7. UPSTREAM MANIFEST VERIFICATION
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
            "Upstream artifact verification failed:\n"
            + "\n".join(failures)
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
            "Step-02 raw-file verification failed:\n"
            + "\n".join(failures)
        )

    return {
        "manifest_sha256": manifest_sha,
        "verified_raw_file_count": len(raw_records),
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

    if not STEP04_CLASSIFICATION.exists():
        raise RuntimeError(
            f"Missing Step-04 session classification: {STEP04_CLASSIFICATION}"
        )

    with STEP04_POLICY.open("r", encoding="utf-8") as handle:
        policy = json.load(handle)

    return {
        "manifest_sha256": manifest_sha,
        "policy": policy,
    }


def verify_step05(
    expected_configuration_fingerprint: str,
    expected_step01_sha: str,
    expected_step02_sha: str,
    expected_step03_sha: str,
    expected_step04_sha: str,
) -> Dict[str, Any]:
    manifest_sha = verify_sha_sidecar(
        STEP05_MANIFEST,
        STEP05_MANIFEST_SHA,
        "Step 05",
    )

    with STEP05_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("step", {}).get("status") != "PASS":
        raise RuntimeError("Stored Step-05 status is not PASS.")

    expected = {
        "configuration_fingerprint_sha256": expected_configuration_fingerprint,
        "step01_manifest_sha256": expected_step01_sha,
        "step02_manifest_sha256": expected_step02_sha,
        "step03_manifest_sha256": expected_step03_sha,
        "step04_manifest_sha256": expected_step04_sha,
    }

    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Step-05 chain mismatch: {key}")

    verify_manifest_artifacts(manifest)

    failures = []

    for record in manifest.get("individual_stock_output_records", []):
        path = Path(record["file_path"])

        if not path.exists():
            failures.append(f"missing individual stock output: {path}")
            continue

        if sha256_file(path) != record["sha256"]:
            failures.append(f"individual stock hash mismatch: {path}")

    if failures:
        raise RuntimeError(
            "Step-05 individual-stock verification failed:\n"
            + "\n".join(failures)
        )

    if not STEP05_PANEL.exists():
        raise RuntimeError(
            f"Canonical Step-05 panel missing: {STEP05_PANEL}"
        )

    matching = [
        record
        for record in manifest.get("artifacts", {}).values()
        if isinstance(record, dict)
        and Path(record.get("path", "")).name == STEP05_PANEL.name
    ]

    if len(matching) != 1:
        raise RuntimeError(
            "Could not identify exactly one canonical combined-stock "
            "Parquet artifact in the Step-05 manifest."
        )

    record = matching[0]

    if Path(record["path"]).resolve() != STEP05_PANEL.resolve():
        raise RuntimeError(
            "Step-05 combined-stock Parquet path differs from the canonical "
            "clean-V3 path."
        )

    actual_panel_sha = sha256_file(STEP05_PANEL)

    if actual_panel_sha != record["sha256"]:
        raise RuntimeError("Step-05 combined-stock panel hash mismatch.")

    return {
        "manifest_sha256": manifest_sha,
        "panel_sha256": actual_panel_sha,
        "manifest": manifest,
    }


# =============================================================================
# 8. CONTINUITY-BREAK POLICY
# =============================================================================

def load_continuity_break_dates(
    step04_policy: Dict[str, Any],
) -> list[pd.Timestamp]:
    anomaly_policy = step04_policy.get(
        "date_wide_anomaly_policy",
        {},
    )

    dates = anomaly_policy.get(
        "valid_nifty_sessions_excluded_by_rule",
        [],
    )

    declared_count = anomaly_policy.get(
        "valid_nifty_session_exclusion_count"
    )

    if declared_count is not None and int(declared_count) != len(dates):
        raise RuntimeError(
            "Step-04 continuity-break count does not match the "
            "frozen excluded-session list."
        )

    break_dates = sorted(
        {
            pd.Timestamp(value).normalize()
            for value in dates
        }
    )

    calendar = pd.read_csv(
        STEP04_CALENDAR,
        parse_dates=["Date"],
    )

    calendar_dates = set(
        pd.to_datetime(
            calendar["Date"],
            errors="raise",
        )
        .dt.normalize()
        .tolist()
    )

    classification = pd.read_csv(
        STEP04_CLASSIFICATION,
        parse_dates=["Date"],
    )

    classification["Date"] = (
        pd.to_datetime(
            classification["Date"],
            errors="raise",
        )
        .dt.normalize()
    )

    for break_date in break_dates:
        if break_date in calendar_dates:
            raise RuntimeError(
                f"Continuity-break date still appears in the final calendar: "
                f"{break_date.date()}"
            )

        matching = classification.loc[
            classification["Date"] == break_date
        ]

        if len(matching) != 1:
            raise RuntimeError(
                "Could not uniquely reconcile continuity break with "
                f"Step-04 classification: {break_date.date()}"
            )

        row = matching.iloc[0]

        if not bool(row["candidate_nifty_session"]):
            raise RuntimeError(
                f"Continuity break {break_date.date()} was not a valid "
                "candidate NIFTY session."
            )

        if not bool(row["date_wide_anomaly_candidate"]):
            raise RuntimeError(
                f"Continuity break {break_date.date()} did not satisfy "
                "the frozen date-wide anomaly rule."
            )

    return break_dates


def assign_continuity_segments(
    dates: pd.Series,
    break_dates: list[pd.Timestamp],
) -> pd.Series:
    date_values = (
        pd.to_datetime(dates, errors="raise")
        .dt.normalize()
        .to_numpy(dtype="datetime64[ns]")
    )

    if not break_dates:
        return pd.Series(
            np.zeros(len(dates), dtype=np.int64),
            index=dates.index,
            dtype="int64",
        )

    break_values = np.asarray(
        [
            np.datetime64(date, "ns")
            for date in break_dates
        ],
        dtype="datetime64[ns]",
    )

    segment_ids = np.searchsorted(
        break_values,
        date_values,
        side="left",
    )

    return pd.Series(
        segment_ids,
        index=dates.index,
        dtype="int64",
    )


# =============================================================================
# 9. TECHNICAL-INDICATOR IMPLEMENTATION
# =============================================================================

def compute_wilder_average(
    series: pd.Series,
    window: int,
) -> pd.Series:

    if window <= 0:

        raise ValueError(
            "Wilder window must be positive."
        )

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    output = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    consecutive_valid = 0

    seed_end = None

    for position, value in enumerate(
        values
    ):

        if np.isfinite(
            value
        ):

            consecutive_valid += 1

        else:

            consecutive_valid = 0

        if consecutive_valid == window:

            seed_end = position

            break

    if seed_end is None:

        return pd.Series(
            output,
            index=series.index,
            dtype=float,
        )

    seed_start = (
        seed_end
        - window
        + 1
    )

    output[
        seed_end
    ] = np.mean(
        values[
            seed_start:
            seed_end + 1
        ]
    )

    for position in range(
        seed_end + 1,
        len(
            values
        ),
    ):

        current = (
            values[
                position
            ]
        )

        previous = (
            output[
                position - 1
            ]
        )

        if (
            np.isfinite(
                current
            )
            and
            np.isfinite(
                previous
            )
        ):

            output[
                position
            ] = (
                (
                    window - 1
                )
                * previous
                + current
            ) / window

        elif np.isfinite(
            current
        ):

            restart_start = (
                position
                - window
                + 1
            )

            if (
                restart_start
                >= 0
                and
                np.isfinite(
                    values[
                        restart_start:
                        position + 1
                    ]
                ).all()
            ):

                output[
                    position
                ] = np.mean(
                    values[
                        restart_start:
                        position + 1
                    ]
                )

    return pd.Series(
        output,
        index=series.index,
        dtype=float,
    )


def compute_wilder_rsi(
    close: pd.Series,
    window: int = 14,
) -> pd.Series:

    change = close.diff()

    gain = (
        change.clip(
            lower=0
        )
    )

    loss = -(
        change.clip(
            upper=0
        )
    )

    avg_gain = (
        compute_wilder_average(
            gain,
            window,
        )
    )

    avg_loss = (
        compute_wilder_average(
            loss,
            window,
        )
    )

    result = pd.Series(
        np.nan,
        index=close.index,
        dtype=float,
    )

    valid = (
        avg_gain.notna()
        &
        avg_loss.notna()
    )

    both_zero = (
        valid
        &
        avg_gain.eq(
            0
        )
        &
        avg_loss.eq(
            0
        )
    )

    no_loss = (
        valid
        &
        avg_gain.gt(
            0
        )
        &
        avg_loss.eq(
            0
        )
    )

    no_gain = (
        valid
        &
        avg_gain.eq(
            0
        )
        &
        avg_loss.gt(
            0
        )
    )

    regular = (
        valid
        &
        avg_gain.gt(
            0
        )
        &
        avg_loss.gt(
            0
        )
    )

    result.loc[
        both_zero
    ] = 50.0

    result.loc[
        no_loss
    ] = 100.0

    result.loc[
        no_gain
    ] = 0.0

    relative_strength = (
        avg_gain.loc[
            regular
        ]
        /
        avg_loss.loc[
            regular
        ]
    )

    result.loc[
        regular
    ] = (
        100.0
        -
        (
            100.0
            /
            (
                1.0
                +
                relative_strength
            )
        )
    )

    return result


def aroon_up(
    values: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    periods_since_high = int(
        np.argmax(
            values[
                ::-1
            ]
        )
    )

    return (
        100.0
        *
        (
            len(
                values
            )
            -
            periods_since_high
        )
        /
        len(
            values
        )
    )


def aroon_down(
    values: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    periods_since_low = int(
        np.argmin(
            values[
                ::-1
            ]
        )
    )

    return (
        100.0
        *
        (
            len(
                values
            )
            -
            periods_since_low
        )
        /
        len(
            values
        )
    )


def build_segment_features(
    segment: pd.DataFrame,
) -> pd.DataFrame:

    table = (
        segment
        .copy()
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    if table[
        "Ticker"
    ].nunique() != 1:

        raise AssertionError(
            "Segment contains more than one ticker."
        )

    if table[
        "continuity_segment_id"
    ].nunique() != 1:

        raise AssertionError(
            "Segment contains more than one continuity ID."
        )

    # -------------------------------------------------------------------------
    # Frozen Step-05 Model_* series become the canonical OHLCV inputs.
    # -------------------------------------------------------------------------

    table[
        "open"
    ] = pd.to_numeric(
        table[
            "Model_Open"
        ],
        errors="raise",
    )

    table[
        "high"
    ] = pd.to_numeric(
        table[
            "Model_High"
        ],
        errors="raise",
    )

    table[
        "low"
    ] = pd.to_numeric(
        table[
            "Model_Low"
        ],
        errors="raise",
    )

    table[
        "close"
    ] = pd.to_numeric(
        table[
            "Model_Close"
        ],
        errors="raise",
    )

    table[
        "volume"
    ] = pd.to_numeric(
        table[
            "Model_Volume"
        ],
        errors="raise",
    )

    required_base = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if (
        table[
            required_base
        ]
        .isna()
        .any()
        .any()
    ):

        raise AssertionError(
            "Missing modelling OHLCV."
        )

    if (
        table[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        <= 0
    ).any().any():

        raise AssertionError(
            "Non-positive modelling price."
        )

    if (
        table[
            "volume"
        ]
        < 0
    ).any():

        raise AssertionError(
            "Negative modelling volume."
        )

    # =========================================================================
    # GROUP 1 — OHLCV AND RETURNS
    # =========================================================================

    table[
        "return_1d"
    ] = (
        table[
            "close"
        ]
        .pct_change(
            fill_method=None
        )
    )

    table[
        "log_return_1d"
    ] = np.log(
        table[
            "close"
        ]
        /
        table[
            "close"
        ].shift(
            1
        )
    )

    table[
        "high_low_range"
    ] = (
        (
            table[
                "high"
            ]
            -
            table[
                "low"
            ]
        )
        /
        table[
            "close"
        ]
    )

    table[
        "open_close_change"
    ] = (
        (
            table[
                "close"
            ]
            -
            table[
                "open"
            ]
        )
        /
        table[
            "open"
        ]
    )

    table[
        "volume_change"
    ] = (
        np.log1p(
            table[
                "volume"
            ]
        )
        .diff()
    )

    for lag in (
        1,
        2,
        3,
        5,
    ):

        table[
            f"return_lag_{lag}"
        ] = (
            table[
                "return_1d"
            ]
            .shift(
                lag
            )
        )

    # =========================================================================
    # GROUP 2 — ROLLING TREND AND VOLATILITY
    # =========================================================================

    for window in (
        5,
        10,
        20,
    ):

        table[
            f"rolling_mean_{window}"
        ] = (
            table[
                "close"
            ]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

        table[
            f"rolling_std_{window}"
        ] = (
            table[
                "log_return_1d"
            ]
            .rolling(
                window=window,
                min_periods=window,
            )
            .std(
                ddof=1
            )
        )

    table[
        "rolling_min_20"
    ] = (
        table[
            "close"
        ]
        .rolling(
            window=20,
            min_periods=20,
        )
        .min()
    )

    table[
        "rolling_max_20"
    ] = (
        table[
            "close"
        ]
        .rolling(
            window=20,
            min_periods=20,
        )
        .max()
    )

    # =========================================================================
    # GROUP 3 — EMA / MACD / RSI / STOCHASTIC
    # =========================================================================

    table[
        "ema_12"
    ] = (
        table[
            "close"
        ]
        .ewm(
            span=12,
            adjust=False,
            min_periods=12,
        )
        .mean()
    )

    table[
        "ema_26"
    ] = (
        table[
            "close"
        ]
        .ewm(
            span=26,
            adjust=False,
            min_periods=26,
        )
        .mean()
    )

    table[
        "macd"
    ] = (
        table[
            "ema_12"
        ]
        -
        table[
            "ema_26"
        ]
    )

    table[
        "macd_signal"
    ] = (
        table[
            "macd"
        ]
        .ewm(
            span=9,
            adjust=False,
            min_periods=9,
        )
        .mean()
    )

    table[
        "macd_hist"
    ] = (
        table[
            "macd"
        ]
        -
        table[
            "macd_signal"
        ]
    )

    table[
        "rsi_14"
    ] = (
        compute_wilder_rsi(
            table[
                "close"
            ],
            window=14,
        )
    )

    lowest_low_14 = (
        table[
            "low"
        ]
        .rolling(
            window=14,
            min_periods=14,
        )
        .min()
    )

    highest_high_14 = (
        table[
            "high"
        ]
        .rolling(
            window=14,
            min_periods=14,
        )
        .max()
    )

    stochastic_range = (
        highest_high_14
        -
        lowest_low_14
    )

    valid_stochastic = (
        lowest_low_14.notna()
        &
        highest_high_14.notna()
    )

    table[
        "stoch_k_14"
    ] = np.nan

    nonzero_range = (
        valid_stochastic
        &
        stochastic_range.ne(
            0
        )
    )

    zero_range = (
        valid_stochastic
        &
        stochastic_range.eq(
            0
        )
    )

    table.loc[
        nonzero_range,
        "stoch_k_14",
    ] = (
        100.0
        *
        (
            table.loc[
                nonzero_range,
                "close",
            ]
            -
            lowest_low_14.loc[
                nonzero_range
            ]
        )
        /
        stochastic_range.loc[
            nonzero_range
        ]
    )

    table.loc[
        zero_range,
        "stoch_k_14",
    ] = 50.0

    table[
        "stoch_d_3"
    ] = (
        table[
            "stoch_k_14"
        ]
        .rolling(
            window=3,
            min_periods=3,
        )
        .mean()
    )

    # =========================================================================
    # GROUP 4 — BOLLINGER / ATR / ADX / AROON
    # =========================================================================

    bollinger_middle = (
        table[
            "close"
        ]
        .rolling(
            window=20,
            min_periods=20,
        )
        .mean()
    )

    bollinger_std = (
        table[
            "close"
        ]
        .rolling(
            window=20,
            min_periods=20,
        )
        .std(
            ddof=1
        )
    )

    table[
        "bollinger_upper_20"
    ] = (
        bollinger_middle
        +
        2.0
        * bollinger_std
    )

    table[
        "bollinger_lower_20"
    ] = (
        bollinger_middle
        -
        2.0
        * bollinger_std
    )

    table[
        "bollinger_width_20"
    ] = (
        (
            table[
                "bollinger_upper_20"
            ]
            -
            table[
                "bollinger_lower_20"
            ]
        )
        /
        bollinger_middle
    )

    previous_close = (
        table[
            "close"
        ]
        .shift(
            1
        )
    )

    true_range = pd.concat(
        [
            (
                table[
                    "high"
                ]
                -
                table[
                    "low"
                ]
            ),
            (
                table[
                    "high"
                ]
                -
                previous_close
            ).abs(),
            (
                table[
                    "low"
                ]
                -
                previous_close
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )

    table[
        "atr_14"
    ] = (
        compute_wilder_average(
            true_range,
            window=14,
        )
    )

    upward_move = (
        table[
            "high"
        ]
        .diff()
    )

    downward_move = -(
        table[
            "low"
        ]
        .diff()
    )

    positive_dm = pd.Series(
        np.where(
            (
                upward_move
                >
                downward_move
            )
            &
            (
                upward_move
                >
                0
            ),
            upward_move,
            0.0,
        ),
        index=table.index,
        dtype=float,
    )

    negative_dm = pd.Series(
        np.where(
            (
                downward_move
                >
                upward_move
            )
            &
            (
                downward_move
                >
                0
            ),
            downward_move,
            0.0,
        ),
        index=table.index,
        dtype=float,
    )

    smoothed_positive_dm = (
        compute_wilder_average(
            positive_dm,
            window=14,
        )
    )

    smoothed_negative_dm = (
        compute_wilder_average(
            negative_dm,
            window=14,
        )
    )

    table[
        "plus_di_14"
    ] = pd.Series(
        np.nan,
        index=table.index,
        dtype=float,
    )

    table[
        "minus_di_14"
    ] = pd.Series(
        np.nan,
        index=table.index,
        dtype=float,
    )

    valid_positive_atr = (
        table[
            "atr_14"
        ].notna()
        &
        table[
            "atr_14"
        ].gt(
            0
        )
        &
        smoothed_positive_dm.notna()
        &
        smoothed_negative_dm.notna()
    )

    valid_zero_atr = (
        table[
            "atr_14"
        ].notna()
        &
        table[
            "atr_14"
        ].eq(
            0
        )
        &
        smoothed_positive_dm.notna()
        &
        smoothed_negative_dm.notna()
    )

    table.loc[
        valid_positive_atr,
        "plus_di_14",
    ] = (
        100.0
        *
        smoothed_positive_dm.loc[
            valid_positive_atr
        ]
        /
        table.loc[
            valid_positive_atr,
            "atr_14",
        ]
    )

    table.loc[
        valid_positive_atr,
        "minus_di_14",
    ] = (
        100.0
        *
        smoothed_negative_dm.loc[
            valid_positive_atr
        ]
        /
        table.loc[
            valid_positive_atr,
            "atr_14",
        ]
    )

    table.loc[
        valid_zero_atr,
        [
            "plus_di_14",
            "minus_di_14",
        ],
    ] = 0.0

    di_denominator = (
        table[
            "plus_di_14"
        ]
        +
        table[
            "minus_di_14"
        ]
    )

    directional_index = pd.Series(
        np.nan,
        index=table.index,
        dtype=float,
    )

    valid_di = (
        table[
            "plus_di_14"
        ].notna()
        &
        table[
            "minus_di_14"
        ].notna()
    )

    nonzero_di = (
        valid_di
        &
        di_denominator.ne(
            0
        )
    )

    zero_di = (
        valid_di
        &
        di_denominator.eq(
            0
        )
    )

    directional_index.loc[
        nonzero_di
    ] = (
        100.0
        *
        (
            table.loc[
                nonzero_di,
                "plus_di_14",
            ]
            -
            table.loc[
                nonzero_di,
                "minus_di_14",
            ]
        ).abs()
        /
        di_denominator.loc[
            nonzero_di
        ]
    )

    directional_index.loc[
        zero_di
    ] = 0.0

    table[
        "adx_14"
    ] = (
        compute_wilder_average(
            directional_index,
            window=14,
        )
    )

    table[
        "aroon_up_25"
    ] = (
        table[
            "high"
        ]
        .rolling(
            window=25,
            min_periods=25,
        )
        .apply(
            aroon_up,
            raw=True,
        )
    )

    table[
        "aroon_down_25"
    ] = (
        table[
            "low"
        ]
        .rolling(
            window=25,
            min_periods=25,
        )
        .apply(
            aroon_down,
            raw=True,
        )
    )

    table[
        "aroon_osc_25"
    ] = (
        table[
            "aroon_up_25"
        ]
        -
        table[
            "aroon_down_25"
        ]
    )

    # =========================================================================
    # GROUP 5 — VOLUME / RELATIVE POSITION
    # =========================================================================

    price_direction = (
        np.sign(
            table[
                "close"
            ]
            .diff()
        )
        .fillna(
            0.0
        )
    )

    table[
        "obv"
    ] = (
        price_direction
        *
        table[
            "volume"
        ]
    ).cumsum()

    high_low_denominator = (
        table[
            "high"
        ]
        -
        table[
            "low"
        ]
    )

    close_location_value = pd.Series(
        0.0,
        index=table.index,
        dtype=float,
    )

    nonzero_hl = (
        high_low_denominator.ne(
            0
        )
    )

    close_location_value.loc[
        nonzero_hl
    ] = (
        (
            (
                table.loc[
                    nonzero_hl,
                    "close",
                ]
                -
                table.loc[
                    nonzero_hl,
                    "low",
                ]
            )
            -
            (
                table.loc[
                    nonzero_hl,
                    "high",
                ]
                -
                table.loc[
                    nonzero_hl,
                    "close",
                ]
            )
        )
        /
        high_low_denominator.loc[
            nonzero_hl
        ]
    )

    table[
        "ad_line"
    ] = (
        close_location_value
        *
        table[
            "volume"
        ]
    ).cumsum()

    typical_price = (
        (
            table[
                "high"
            ]
            +
            table[
                "low"
            ]
            +
            table[
                "close"
            ]
        )
        /
        3.0
    )

    rolling_volume_sum = (
        table[
            "volume"
        ]
        .rolling(
            window=20,
            min_periods=20,
        )
        .sum()
    )

    table[
        "vwap_20"
    ] = (
        (
            typical_price
            *
            table[
                "volume"
            ]
        )
        .rolling(
            window=20,
            min_periods=20,
        )
        .sum()
        /
        rolling_volume_sum
    )

    table[
        "volume_sma_20"
    ] = (
        table[
            "volume"
        ]
        .rolling(
            window=20,
            min_periods=20,
        )
        .mean()
    )

    table[
        "close_to_rolling_mean_20"
    ] = (
        table[
            "close"
        ]
        /
        table[
            "rolling_mean_20"
        ]
        -
        1.0
    )

    table[
        "close_to_rolling_max_20"
    ] = (
        table[
            "close"
        ]
        /
        table[
            "rolling_max_20"
        ]
        -
        1.0
    )

    table[
        "close_to_rolling_min_20"
    ] = (
        table[
            "close"
        ]
        /
        table[
            "rolling_min_20"
        ]
        -
        1.0
    )

    # -------------------------------------------------------------------------
    # Replace infinities only. Warm-up NaNs remain intentionally.
    # -------------------------------------------------------------------------

    table[
        FEATURE_COLUMNS
    ] = (
        table[
            FEATURE_COLUMNS
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    output_columns = [
        "session_number",
        "Date",
        "Ticker",
        "continuity_segment_id",
    ] + FEATURE_COLUMNS

    return table[
        output_columns
    ].copy()


def build_feature_dictionary() -> pd.DataFrame:

    descriptions = {

        "open": (
            "Step-05 forward synthetic return-index opening level"
        ),

        "high": (
            "Step-05 forward synthetic return-index high level"
        ),

        "low": (
            "Step-05 forward synthetic return-index low level"
        ),

        "close": (
            "Step-05 forward synthetic return-index closing level"
        ),

        "volume": (
            "Yahoo provider Volume retained unchanged by Step 05"
        ),

        "return_1d": (
            "Within-continuity-segment simple close-to-close return"
        ),

        "log_return_1d": (
            "Within-continuity-segment logarithmic close-to-close return"
        ),

        "high_low_range": (
            "(high - low) / close"
        ),

        "open_close_change": (
            "(close - open) / open"
        ),

        "volume_change": (
            "First difference of log(1 + provider Volume)"
        ),

        "return_lag_1": "1-session lag of return_1d",
        "return_lag_2": "2-session lag of return_1d",
        "return_lag_3": "3-session lag of return_1d",
        "return_lag_5": "5-session lag of return_1d",

        "rolling_mean_5": "5-session rolling mean of close",
        "rolling_mean_10": "10-session rolling mean of close",
        "rolling_mean_20": "20-session rolling mean of close",

        "rolling_std_5": (
            "5-session sample standard deviation of log return"
        ),

        "rolling_std_10": (
            "10-session sample standard deviation of log return"
        ),

        "rolling_std_20": (
            "20-session sample standard deviation of log return"
        ),

        "rolling_min_20": "20-session rolling minimum close",
        "rolling_max_20": "20-session rolling maximum close",

        "ema_12": "12-session exponential moving average",
        "ema_26": "26-session exponential moving average",
        "macd": "EMA(12) minus EMA(26)",
        "macd_signal": "9-session EMA of MACD",
        "macd_hist": "MACD minus MACD signal",

        "rsi_14": "14-session Wilder RSI",

        "stoch_k_14": (
            "14-session stochastic %K"
        ),

        "stoch_d_3": (
            "3-session moving average of stochastic %K"
        ),

        "bollinger_upper_20": (
            "20-session mean plus 2 sample standard deviations"
        ),

        "bollinger_lower_20": (
            "20-session mean minus 2 sample standard deviations"
        ),

        "bollinger_width_20": (
            "Bollinger-band width divided by 20-session mean"
        ),

        "atr_14": "14-session Wilder average true range",
        "adx_14": "14-session Wilder average directional index",
        "plus_di_14": "14-session positive directional indicator",
        "minus_di_14": "14-session negative directional indicator",

        "aroon_up_25": "25-session Aroon Up",
        "aroon_down_25": "25-session Aroon Down",
        "aroon_osc_25": "Aroon Up minus Aroon Down",

        "obv": (
            "Segment-reset on-balance volume using provider Volume"
        ),

        "ad_line": (
            "Segment-reset accumulation/distribution line"
        ),

        "vwap_20": (
            "20-session rolling volume-weighted typical price"
        ),

        "volume_sma_20": (
            "20-session moving average of provider Volume"
        ),

        "close_to_rolling_mean_20": (
            "close / rolling_mean_20 - 1"
        ),

        "close_to_rolling_max_20": (
            "close / rolling_max_20 - 1"
        ),

        "close_to_rolling_min_20": (
            "close / rolling_min_20 - 1"
        ),
    }

    rows = []

    feature_number = 0

    for group_name, features in (
        STOCK_FEATURE_GROUPS.items()
    ):

        for feature in features:

            feature_number += 1

            rows.append(
                {
                    "feature_number": (
                        feature_number
                    ),
                    "feature": (
                        feature
                    ),
                    "feature_group": (
                        group_name
                    ),
                    "description": (
                        descriptions[
                            feature
                        ]
                    ),
                    "causal": True,
                    "target_derived": False,
                    "scaling_applied_here": False,
                }
            )

    return pd.DataFrame(
        rows
    )


def run_prefix_causality_audit(
    source: pd.DataFrame,
    full_features: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for (
        ticker,
        segment_id,
    ), segment_source in (
        source.groupby(
            [
                "Ticker",
                "continuity_segment_id",
            ],
            sort=True,
        )
    ):

        segment_source = (
            segment_source
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        full_segment = (
            full_features.loc[
                (
                    full_features[
                        "Ticker"
                    ]
                    == ticker
                )
                &
                (
                    full_features[
                        "continuity_segment_id"
                    ]
                    == segment_id
                )
            ]
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        for fraction in (
            0.50,
            0.90,
        ):

            cutoff = int(
                np.floor(
                    len(
                        segment_source
                    )
                    * fraction
                )
            ) - 1

            minimum_cutoff = min(
                50,
                len(
                    segment_source
                )
                - 1,
            )

            cutoff = min(
                max(
                    minimum_cutoff,
                    cutoff,
                ),
                len(
                    segment_source
                )
                - 1,
            )

            prefix = (
                segment_source
                .iloc[
                    :cutoff + 1
                ]
                .copy()
            )

            prefix_features = (
                build_segment_features(
                    prefix
                )
            )

            cutoff_date = (
                prefix_features[
                    "Date"
                ].iloc[
                    -1
                ]
            )

            prefix_values = (
                prefix_features[
                    FEATURE_COLUMNS
                ]
                .iloc[
                    -1
                ]
                .to_numpy(
                    dtype=float
                )
            )

            matching = (
                full_segment.loc[
                    full_segment[
                        "Date"
                    ]
                    == cutoff_date
                ]
            )

            if len(
                matching
            ) != 1:

                raise AssertionError(
                    "Prefix causality audit could not find "
                    "exactly one matching full-panel row."
                )

            full_values = (
                matching[
                    FEATURE_COLUMNS
                ]
                .iloc[
                    0
                ]
                .to_numpy(
                    dtype=float
                )
            )

            match = bool(
                np.allclose(
                    prefix_values,
                    full_values,
                    rtol=1e-11,
                    atol=1e-11,
                    equal_nan=True,
                )
            )

            finite = (
                np.isfinite(
                    prefix_values
                )
                &
                np.isfinite(
                    full_values
                )
            )

            maximum_difference = (
                float(
                    np.max(
                        np.abs(
                            prefix_values[
                                finite
                            ]
                            -
                            full_values[
                                finite
                            ]
                        )
                    )
                )
                if finite.any()
                else 0.0
            )

            records.append(
                {
                    "ticker": ticker,
                    "continuity_segment_id": int(
                        segment_id
                    ),
                    "cutoff_fraction": (
                        fraction
                    ),
                    "cutoff_date": (
                        cutoff_date
                    ),
                    "all_features_match": (
                        match
                    ),
                    "maximum_absolute_difference": (
                        maximum_difference
                    ),
                }
            )

    audit = pd.DataFrame(
        records
    )

    if not audit[
        "all_features_match"
    ].all():

        failed = (
            audit.loc[
                ~audit[
                    "all_features_match"
                ]
            ]
        )

        raise AssertionError(
            "Prefix causality audit failed:\n"
            f"{failed}"
        )

    return audit


# =============================================================================
# 10. STEP-05 / STEP-06 RETURN-CONSISTENCY AUDIT
# =============================================================================

def audit_return_consistency(
    source: pd.DataFrame,
    feature_panel: pd.DataFrame,
) -> pd.DataFrame:
    left = feature_panel[
        [
            "Ticker",
            "Date",
            "continuity_segment_id",
            "return_1d",
            "log_return_1d",
        ]
    ].copy()

    right = source[
        [
            "Ticker",
            "Date",
            "Model_Return_1session",
            "Model_Log_Return_1session",
        ]
    ].copy()

    merged = left.merge(
        right,
        on=["Ticker", "Date"],
        how="left",
        validate="one_to_one",
    )

    merged["is_segment_start"] = (
        merged
        .groupby(
            ["Ticker", "continuity_segment_id"],
            sort=False,
        )
        .cumcount()
        .eq(0)
    )

    rows = []

    for ticker, table in merged.groupby("Ticker", sort=True):
        compare = table.loc[
            ~table["is_segment_start"]
        ].copy()

        if compare.empty:
            raise RuntimeError(
                f"{ticker}: no non-boundary rows for return reconciliation."
            )

        if (
            compare[
                [
                    "return_1d",
                    "log_return_1d",
                    "Model_Return_1session",
                    "Model_Log_Return_1session",
                ]
            ]
            .isna()
            .any()
            .any()
        ):
            raise RuntimeError(
                f"{ticker}: unexpected null in non-boundary return audit."
            )

        return_diff = (
            compare["return_1d"]
            - compare["Model_Return_1session"]
        ).abs()

        log_diff = (
            compare["log_return_1d"]
            - compare["Model_Log_Return_1session"]
        ).abs()

        max_return_diff = float(return_diff.max())
        max_log_diff = float(log_diff.max())

        all_match = bool(
            max_return_diff <= 1e-10
            and max_log_diff <= 1e-10
        )

        if not all_match:
            raise RuntimeError(
                f"{ticker}: Step-06 returns differ from Step-05 returns "
                "away from continuity boundaries."
            )

        boundaries = table.loc[
            table["is_segment_start"]
        ].copy()

        if not boundaries["return_1d"].isna().all():
            raise RuntimeError(
                f"{ticker}: Step-06 return is not reset at every "
                "continuity-segment start."
            )

        if not boundaries["log_return_1d"].isna().all():
            raise RuntimeError(
                f"{ticker}: Step-06 log return is not reset at every "
                "continuity-segment start."
            )

        source_nonnull_reset_rows = int(
            boundaries["Model_Return_1session"]
            .notna()
            .sum()
        )

        rows.append(
            {
                "ticker": ticker,
                "compared_nonboundary_rows": int(len(compare)),
                "segment_start_rows": int(len(boundaries)),
                "segment_starts_with_nonnull_step05_return_reset_to_nan": (
                    source_nonnull_reset_rows
                ),
                "max_abs_simple_return_difference": max_return_diff,
                "max_abs_log_return_difference": max_log_diff,
                "all_nonboundary_returns_match": all_match,
                "all_segment_start_returns_reset": True,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 11. OUTPUT / HASH / STAGING HELPERS
# =============================================================================

def canonical_step06_outputs() -> tuple[Path, ...]:
    return (
        STEP06_OUTPUT_DIR,
        FEATURE_DICTIONARY_CSV,
        SEGMENT_AUDIT_CSV,
        WARMUP_AUDIT_CSV,
        CAUSALITY_AUDIT_CSV,
        RETURN_CONSISTENCY_CSV,
        STOCK_SUMMARY_CSV,
        HASH_REGISTER_CSV,
        SOURCE_HASH_CSV,
        STEP06_SUMMARY_JSON,
        MANIFEST_JSON,
        MANIFEST_SHA,
        FROZEN_STEP06_SCRIPT,
        SUMMARY_TXT,
    )


def assert_new_step06_target() -> None:
    collisions = [
        path
        for path in canonical_step06_outputs()
        if path.exists()
    ]

    if collisions:
        detail = "\n".join(str(path) for path in collisions)

        raise RuntimeError(
            "Step-06 canonical output(s) already exist. "
            "Refusing to overwrite.\n"
            f"{detail}\n\n"
            "If Step 06 already passed, run --verify-only."
        )


def create_staging_root() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (
        cfg.FEATURE_DATA_DIR
        / f"_step06_staging_{timestamp}_{os.getpid()}"
    )

    if root.exists():
        raise RuntimeError(f"Unexpected staging path: {root}")

    root.mkdir(parents=True, exist_ok=False)
    return root


def archive_partial_outputs() -> Optional[Path]:
    if MANIFEST_JSON.exists():
        return None

    existing = [
        path
        for path in canonical_step06_outputs()
        if path.exists()
    ]

    if not existing:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = FAILED_ATTEMPT_DIR / f"partial_artifacts_{timestamp}"
    archive.mkdir(parents=True, exist_ok=False)

    for path in existing:
        try:
            shutil.move(str(path), str(archive / path.name))
        except Exception:
            pass

    return archive


def write_pair_staged(
    dataframe: pd.DataFrame,
    parquet_path: Path,
    csv_path: Path,
) -> None:
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
    path: Path,
) -> None:
    reloaded = pd.read_parquet(
        path,
        engine="pyarrow",
    )

    if len(reloaded) != len(original):
        raise RuntimeError(
            f"Parquet roundtrip row-count mismatch: {path}"
        )

    if list(reloaded.columns) != list(original.columns):
        raise RuntimeError(
            f"Parquet roundtrip column-order mismatch: {path}"
        )

    expected_dates = (
        pd.to_datetime(
            original["Date"],
            errors="raise",
        )
        .dt.normalize()
    )

    observed_dates = (
        pd.to_datetime(
            reloaded["Date"],
            errors="raise",
        )
        .dt.normalize()
    )

    if not np.array_equal(
        expected_dates.to_numpy(),
        observed_dates.to_numpy(),
    ):
        raise RuntimeError(
            f"Parquet roundtrip date mismatch: {path}"
        )


def staged_hash_record(
    *,
    scope: str,
    file_format: str,
    staging_path: Path,
    final_path: Path,
) -> Dict[str, Any]:
    return {
        "scope": scope,
        "file_format": file_format,
        "file_path": str(final_path),
        "relative_path": relative_to_experiment(final_path),
        "size_bytes": int(staging_path.stat().st_size),
        "sha256": sha256_file(staging_path),
    }


# =============================================================================
# 12. SOURCE-CODE HASH AUDIT
# =============================================================================

def find_artifact_by_filename(
    manifest_path: Path,
    filename: str,
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

        if path.name == filename:
            return path

    return None


def collect_source_hashes() -> pd.DataFrame:
    current_step00 = Path(cfg.__file__).resolve()
    frozen_step00 = cfg.get_frozen_script_path().resolve()

    frozen_step01 = find_artifact_by_filename(
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

    frozen_step03 = find_artifact_by_filename(
        STEP03_MANIFEST,
        "step03_market_integrity_audit_frozen.py",
    )

    frozen_step04 = find_artifact_by_filename(
        STEP04_MANIFEST,
        "step04_session_and_corporate_action_policy_frozen.py",
    )

    frozen_step05 = find_artifact_by_filename(
        STEP05_MANIFEST,
        "step05_materialize_clean_model_panel_frozen.py",
    )

    sources: list[tuple[str, Path]] = [
        ("current_step00", current_step00),
        ("frozen_step00", frozen_step00),
        ("current_step06", SCRIPT_FILE.resolve()),
    ]

    for role, path in (
        ("frozen_step01", frozen_step01),
        ("frozen_step02", frozen_step02),
        ("frozen_step03", frozen_step03),
        ("frozen_step04", frozen_step04),
        ("frozen_step05", frozen_step05),
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

    return pd.DataFrame(
        rows,
        columns=[
            "role",
            "file_name",
            "absolute_path",
            "size_bytes",
            "sha256",
        ],
    )


# =============================================================================
# 13. STEP-06 MANIFEST / VERIFY
# =============================================================================

def step06_artifact_paths() -> tuple[Path, ...]:
    return (
        FEATURE_PANEL_PARQUET,
        FEATURE_PANEL_CSV,
        COMPLETE_FEATURE_PANEL_PARQUET,
        COMPLETE_FEATURE_PANEL_CSV,
        FEATURE_DICTIONARY_CSV,
        SEGMENT_AUDIT_CSV,
        WARMUP_AUDIT_CSV,
        CAUSALITY_AUDIT_CSV,
        RETURN_CONSISTENCY_CSV,
        STOCK_SUMMARY_CSV,
        HASH_REGISTER_CSV,
        SOURCE_HASH_CSV,
        STEP06_SUMMARY_JSON,
        FROZEN_STEP06_SCRIPT,
        SUMMARY_TXT,
    )


def write_step06_manifest(
    *,
    step00: Dict[str, Any],
    step01: Dict[str, Any],
    step02: Dict[str, Any],
    step03: Dict[str, Any],
    step04: Dict[str, Any],
    step05: Dict[str, Any],
) -> str:
    artifacts = {
        path.name: artifact_record(path)
        for path in step06_artifact_paths()
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
        "step05_manifest_sha256": step05["manifest_sha256"],
        "step05_panel_sha256": step05["panel_sha256"],
        "final_test_state": step00["final_test_state"],
        "artifacts": artifacts,
    }

    atomic_write_json(
        MANIFEST_JSON,
        manifest,
    )

    manifest_sha = sha256_file(MANIFEST_JSON)

    atomic_write_text(
        MANIFEST_SHA,
        f"{manifest_sha}  {MANIFEST_JSON.name}\n",
    )

    return manifest_sha


def verify_existing_step06() -> int:
    print("=" * 126)
    print("STEP 06 — VERIFY EXISTING STOCK-FEATURE FREEZE")
    print("=" * 126)

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
    step05 = verify_step05(
        step00["configuration_fingerprint_sha256"],
        step01["manifest_sha256"],
        step02["manifest_sha256"],
        step03["manifest_sha256"],
        step04["manifest_sha256"],
    )

    if not MANIFEST_JSON.exists():
        print("[FAIL] Step-06 manifest missing.")
        return 20

    if not MANIFEST_SHA.exists():
        print("[FAIL] Step-06 manifest SHA sidecar missing.")
        return 21

    actual_manifest_sha = verify_sha_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 06",
    )

    with MANIFEST_JSON.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("step", {}).get("status") != "PASS":
        print("[FAIL] Stored Step-06 status is not PASS.")
        return 22

    expected_chain = {
        "configuration_fingerprint_sha256": (
            step00["configuration_fingerprint_sha256"]
        ),
        "step01_manifest_sha256": step01["manifest_sha256"],
        "step02_manifest_sha256": step02["manifest_sha256"],
        "step03_manifest_sha256": step03["manifest_sha256"],
        "step04_manifest_sha256": step04["manifest_sha256"],
        "step05_manifest_sha256": step05["manifest_sha256"],
        "step05_panel_sha256": step05["panel_sha256"],
    }

    for field, expected in expected_chain.items():
        if manifest.get(field) != expected:
            print(f"[FAIL] Step-06 chain mismatch: {field}")
            return 23

    failures = []

    for name, record in manifest.get("artifacts", {}).items():
        path = Path(record["path"])

        if not path.exists():
            failures.append(f"missing: {name}")
            continue

        if sha256_file(path) != record["sha256"]:
            failures.append(f"hash mismatch: {name}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")

        print("=" * 126)
        print("STEP 06 VERIFICATION STATUS: FAIL")
        print("=" * 126)
        return 24

    print(
        "[PASS] Configuration fingerprint : "
        f"{step00['configuration_fingerprint_sha256']}"
    )
    print(f"[PASS] Step-05 manifest          : {step05['manifest_sha256']}")
    print(f"[PASS] Step-06 manifest          : {actual_manifest_sha}")
    print(f"[PASS] Final-test state          : {step00['final_test_state']}")
    print("=" * 126)
    print("STEP 06 VERIFICATION STATUS: PASS")
    print("=" * 126)

    return 0


# =============================================================================
# 14. FAILURE DIAGNOSTIC
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
        / f"06_stock_features_FAILED_{timestamp}.json"
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
        "canonical_manifest_exists": MANIFEST_JSON.exists(),
    }

    atomic_write_json(path, payload)
    return path


# =============================================================================
# 15. MAIN FEATURE MATERIALIZATION
# =============================================================================

def run_new_step06() -> int:
    print("=" * 126)
    print(STEP_NAME)
    print("=" * 126)

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

        print("[2/11] Verifying Step 01 through Step 05...")
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
        step05 = verify_step05(
            step00["configuration_fingerprint_sha256"],
            step01["manifest_sha256"],
            step02["manifest_sha256"],
            step03["manifest_sha256"],
            step04["manifest_sha256"],
        )

        print(
            "       Step-05 manifest: "
            f"{step05['manifest_sha256']}"
        )
        print(
            "       Step-05 panel SHA-256: "
            f"{step05['panel_sha256']}"
        )

        stage = "output_collision_guard"

        print("[3/11] Checking immutable Step-06 output target...")
        assert_new_step06_target()
        cfg.create_directories()

        stage = "load_policy_and_source"

        print("[4/11] Loading frozen continuity policy and Step-05 panel...")
        break_dates = load_continuity_break_dates(
            step04["policy"]
        )

        source = pd.read_parquet(
            STEP05_PANEL,
            engine="pyarrow",
        )

        source["Date"] = (
            pd.to_datetime(
                source["Date"],
                errors="raise",
            )
            .dt.normalize()
        )

        missing_columns = [
            column
            for column in FEATURE_COLUMNS_REQUIRED_FROM_STEP05
            if column not in source.columns
        ]

        if missing_columns:
            raise RuntimeError(
                "Step-05 panel is missing required modelling columns: "
                + ", ".join(missing_columns)
            )

        if source[["Ticker", "Date"]].duplicated().any():
            raise RuntimeError(
                "Duplicate ticker-date rows in Step-05 panel."
            )

        if source["Ticker"].nunique() != EXPECTED_STOCK_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_STOCK_COUNT} stocks; "
                f"found {source['Ticker'].nunique()}."
            )

        final_calendar = pd.read_csv(
            STEP04_CALENDAR,
            parse_dates=["Date"],
        )

        final_calendar["Date"] = (
            pd.to_datetime(
                final_calendar["Date"],
                errors="raise",
            )
            .dt.normalize()
        )

        expected_source_rows = (
            len(final_calendar)
            * EXPECTED_STOCK_COUNT
        )

        if len(source) != expected_source_rows:
            raise RuntimeError(
                "Unexpected Step-05 source row count.\n"
                f"Expected: {expected_source_rows:,}\n"
                f"Observed: {len(source):,}"
            )

        source_dates = pd.DatetimeIndex(
            source["Date"]
            .drop_duplicates()
            .sort_values()
        )

        calendar_dates = pd.DatetimeIndex(
            final_calendar["Date"]
        )

        if not np.array_equal(
            source_dates.to_numpy(),
            calendar_dates.to_numpy(),
        ):
            raise RuntimeError(
                "Step-05 panel dates differ from the frozen Step-04 calendar."
            )

        source["continuity_segment_id"] = (
            assign_continuity_segments(
                source["Date"],
                break_dates,
            )
        )

        segment_count = int(
            source["continuity_segment_id"].nunique()
        )

        expected_segment_count = (
            len(break_dates) + 1
        )

        if segment_count != expected_segment_count:
            raise RuntimeError(
                f"Expected {expected_segment_count} continuity segments; "
                f"found {segment_count}."
            )

        print(
            "       Input rows: "
            f"{len(source):,}"
        )
        print(
            "       Sessions: "
            f"{len(final_calendar):,}"
        )
        print(
            "       Continuity breaks: "
            + (
                ", ".join(
                    date.strftime("%Y-%m-%d")
                    for date in break_dates
                )
                if break_dates
                else "NONE"
            )
        )
        print(
            "       Segments/stock: "
            f"{expected_segment_count}"
        )

        stage = "feature_construction"

        print("[5/11] Constructing exactly 47 stock features by ticker × segment...")

        feature_tables = []
        segment_audit_rows = []
        warmup_audit_rows = []

        for ticker in sorted(cfg.STOCK_TICKERS):
            ticker_source = (
                source.loc[
                    source["Ticker"] == ticker
                ]
                .copy()
            )

            ticker_segment_ids = sorted(
                ticker_source["continuity_segment_id"].unique()
            )

            if len(ticker_segment_ids) != expected_segment_count:
                raise RuntimeError(
                    f"{ticker}: expected {expected_segment_count} "
                    "continuity segments."
                )

            for segment_id in ticker_segment_ids:
                segment = (
                    ticker_source.loc[
                        ticker_source["continuity_segment_id"]
                        == segment_id
                    ]
                    .sort_values("Date")
                    .reset_index(drop=True)
                )

                features = build_segment_features(
                    segment
                )

                feature_tables.append(
                    features
                )

                for feature in FEATURE_COLUMNS:
                    series = features[feature]
                    valid_positions = np.flatnonzero(
                        series.notna().to_numpy()
                    )

                    if len(valid_positions) == 0:
                        raise RuntimeError(
                            f"{ticker}, segment {segment_id}, "
                            f"{feature}: no valid values."
                        )

                    first_valid = int(valid_positions[0])

                    internal_missing = int(
                        series.iloc[
                            first_valid:
                        ]
                        .isna()
                        .sum()
                    )

                    if internal_missing != 0:
                        raise RuntimeError(
                            f"{ticker}, segment {segment_id}, "
                            f"{feature}: internal missing values after warm-up."
                        )

                    warmup_audit_rows.append(
                        {
                            "ticker": ticker,
                            "continuity_segment_id": int(segment_id),
                            "feature": feature,
                            "first_valid_position": first_valid,
                            "warmup_missing_rows": first_valid,
                            "internal_missing_values": internal_missing,
                        }
                    )

                complete_mask = (
                    features[FEATURE_COLUMNS]
                    .notna()
                    .all(axis=1)
                )

                complete_positions = np.flatnonzero(
                    complete_mask.to_numpy()
                )

                if len(complete_positions) == 0:
                    raise RuntimeError(
                        f"{ticker}, segment {segment_id}: "
                        "no complete feature observations."
                    )

                first_complete_position = int(
                    complete_positions[0]
                )

                if (
                    first_complete_position
                    != EXPECTED_FIRST_COMPLETE_POSITION
                ):
                    raise RuntimeError(
                        f"{ticker}, segment {segment_id}: "
                        f"expected first complete position "
                        f"{EXPECTED_FIRST_COMPLETE_POSITION}, "
                        f"found {first_complete_position}."
                    )

                segment_audit_rows.append(
                    {
                        "ticker": ticker,
                        "continuity_segment_id": int(segment_id),
                        "rows": int(len(features)),
                        "start_date": features["Date"].min(),
                        "end_date": features["Date"].max(),
                        "first_complete_position": first_complete_position,
                        "first_complete_date": features.loc[
                            first_complete_position,
                            "Date",
                        ],
                        "complete_feature_rows": int(
                            complete_mask.sum()
                        ),
                        "warmup_rows_removed_if_complete_case": (
                            first_complete_position
                        ),
                    }
                )

        feature_panel = (
            pd.concat(
                feature_tables,
                ignore_index=True,
            )
            .sort_values(
                ["Date", "Ticker"]
            )
            .reset_index(drop=True)
        )

        if len(feature_panel) != len(source):
            raise RuntimeError(
                "Feature-panel row count changed unexpectedly."
            )

        if len(FEATURE_COLUMNS) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "The feature list is not exactly 47."
            )

        if np.isinf(
            feature_panel[FEATURE_COLUMNS]
            .to_numpy(dtype=float)
        ).any():
            raise RuntimeError(
                "Infinite feature values detected."
            )

        complete_mask = (
            feature_panel[FEATURE_COLUMNS]
            .notna()
            .all(axis=1)
        )

        complete_panel = (
            feature_panel.loc[
                complete_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        segment_audit = pd.DataFrame(
            segment_audit_rows
        )

        warmup_audit = pd.DataFrame(
            warmup_audit_rows
        )

        expected_complete_rows_per_stock = (
            len(final_calendar)
            -
            (
                EXPECTED_FIRST_COMPLETE_POSITION
                * expected_segment_count
            )
        )

        expected_complete_rows_total = (
            expected_complete_rows_per_stock
            * EXPECTED_STOCK_COUNT
        )

        if len(complete_panel) != expected_complete_rows_total:
            raise RuntimeError(
                "Unexpected complete-feature row count.\n"
                f"Expected: {expected_complete_rows_total:,}\n"
                f"Observed: {len(complete_panel):,}"
            )

        stock_summary = (
            feature_panel
            .groupby("Ticker")
            .agg(
                total_rows=("Date", "size"),
                segments=("continuity_segment_id", "nunique"),
                first_date=("Date", "min"),
                last_date=("Date", "max"),
            )
            .reset_index()
        )

        complete_counts = (
            complete_panel
            .groupby("Ticker")
            .size()
            .rename("complete_feature_rows")
            .reset_index()
        )

        stock_summary = (
            stock_summary
            .merge(
                complete_counts,
                on="Ticker",
                how="left",
                validate="one_to_one",
            )
        )

        if not stock_summary["total_rows"].eq(
            len(final_calendar)
        ).all():
            raise RuntimeError(
                "Full feature rows are not balanced across stocks."
            )

        if not stock_summary["segments"].eq(
            expected_segment_count
        ).all():
            raise RuntimeError(
                "Continuity-segment count differs across stocks."
            )

        if not stock_summary["complete_feature_rows"].eq(
            expected_complete_rows_per_stock
        ).all():
            raise RuntimeError(
                "Complete-feature rows are not balanced across stocks."
            )

        print(
            "       Full feature rows: "
            f"{len(feature_panel):,}"
        )
        print(
            "       Complete feature rows: "
            f"{len(complete_panel):,}"
        )
        print(
            "       Complete rows/stock: "
            f"{expected_complete_rows_per_stock:,}"
        )
        print(
            "       First complete position: "
            f"{EXPECTED_FIRST_COMPLETE_POSITION}"
        )

        stage = "causality_and_consistency_audits"

        print("[6/11] Running prefix-causality and Step-05 return-consistency audits...")

        causality_audit = run_prefix_causality_audit(
            source=source,
            full_features=feature_panel,
        )

        return_consistency = audit_return_consistency(
            source=source,
            feature_panel=feature_panel,
        )

        if not causality_audit["all_features_match"].all():
            raise RuntimeError(
                "Prefix-causality audit contains failures."
            )

        if not return_consistency[
            "all_nonboundary_returns_match"
        ].all():
            raise RuntimeError(
                "Step-05 / Step-06 return consistency failed."
            )

        feature_dictionary = build_feature_dictionary()

        if len(feature_dictionary) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "Feature dictionary does not contain exactly 47 rows."
            )

        print(
            "       Prefix-causality tests: "
            f"{len(causality_audit)}"
        )
        print(
            "       Maximum causality diff: "
            f"{causality_audit['maximum_absolute_difference'].max():.3e}"
        )
        print(
            "       Return-consistency stocks passed: "
            f"{int(return_consistency['all_nonboundary_returns_match'].sum())}"
            f"/{len(return_consistency)}"
        )

        stage = "staging_outputs"

        print("[7/11] Writing feature panels to staging area...")
        staging_root = create_staging_root()

        staged_feature_parquet = (
            staging_root / FEATURE_PANEL_PARQUET.name
        )
        staged_feature_csv = (
            staging_root / FEATURE_PANEL_CSV.name
        )

        staged_complete_parquet = (
            staging_root / COMPLETE_FEATURE_PANEL_PARQUET.name
        )
        staged_complete_csv = (
            staging_root / COMPLETE_FEATURE_PANEL_CSV.name
        )

        write_pair_staged(
            feature_panel,
            staged_feature_parquet,
            staged_feature_csv,
        )

        verify_parquet_roundtrip(
            feature_panel,
            staged_feature_parquet,
        )

        write_pair_staged(
            complete_panel,
            staged_complete_parquet,
            staged_complete_csv,
        )

        verify_parquet_roundtrip(
            complete_panel,
            staged_complete_parquet,
        )

        staged_hashes = [
            staged_hash_record(
                scope="full_feature_panel",
                file_format="parquet",
                staging_path=staged_feature_parquet,
                final_path=FEATURE_PANEL_PARQUET,
            ),
            staged_hash_record(
                scope="full_feature_panel",
                file_format="csv",
                staging_path=staged_feature_csv,
                final_path=FEATURE_PANEL_CSV,
            ),
            staged_hash_record(
                scope="complete_feature_panel",
                file_format="parquet",
                staging_path=staged_complete_parquet,
                final_path=COMPLETE_FEATURE_PANEL_PARQUET,
            ),
            staged_hash_record(
                scope="complete_feature_panel",
                file_format="csv",
                staging_path=staged_complete_csv,
                final_path=COMPLETE_FEATURE_PANEL_CSV,
            ),
        ]

        stage = "promote_and_freeze"

        print("[8/11] Promoting feature panels and writing frozen audits...")

        if STEP06_OUTPUT_DIR.exists():
            raise RuntimeError(
                "Canonical Step-06 feature directory appeared unexpectedly."
            )

        os.replace(
            staging_root,
            STEP06_OUTPUT_DIR,
        )
        staging_root = None

        for record in staged_hashes:
            path = Path(record["file_path"])

            if not path.exists():
                raise RuntimeError(
                    f"Promoted Step-06 output missing: {path}"
                )

            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(
                    f"Step-06 output hash changed during promotion: {path}"
                )

        source_hashes = collect_source_hashes()

        write_dataframe_csv(
            FEATURE_DICTIONARY_CSV,
            feature_dictionary,
        )

        write_dataframe_csv(
            SEGMENT_AUDIT_CSV,
            segment_audit,
        )

        write_dataframe_csv(
            WARMUP_AUDIT_CSV,
            warmup_audit,
        )

        write_dataframe_csv(
            CAUSALITY_AUDIT_CSV,
            causality_audit,
        )

        write_dataframe_csv(
            RETURN_CONSISTENCY_CSV,
            return_consistency,
        )

        write_dataframe_csv(
            STOCK_SUMMARY_CSV,
            stock_summary,
        )

        supporting_paths = [
            ("feature_dictionary", FEATURE_DICTIONARY_CSV),
            ("segment_audit", SEGMENT_AUDIT_CSV),
            ("warmup_audit", WARMUP_AUDIT_CSV),
            ("causality_audit", CAUSALITY_AUDIT_CSV),
            ("return_consistency", RETURN_CONSISTENCY_CSV),
            ("stock_summary", STOCK_SUMMARY_CSV),
        ]

        output_hash_records = list(staged_hashes)

        for scope, path in supporting_paths:
            output_hash_records.append(
                {
                    "scope": scope,
                    "file_format": path.suffix.lstrip("."),
                    "file_path": str(path),
                    "relative_path": relative_to_experiment(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )

        output_hash_frame = pd.DataFrame(
            output_hash_records
        )

        write_dataframe_csv(
            HASH_REGISTER_CSV,
            output_hash_frame,
        )

        write_dataframe_csv(
            SOURCE_HASH_CSV,
            source_hashes,
        )

        stage = "freeze_manifest"

        print("[9/11] Freezing Step-06 summary, source snapshot, and manifest...")

        lock_sha_after = sha256_file(
            cfg.get_final_test_lock_path()
        )

        if (
            lock_sha_after
            != step00["final_test_lock_sha256"]
        ):
            raise RuntimeError(
                "Final-test lock changed during Step 06."
            )

        summary_payload = {
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
                "step00_manifest_sha256": (
                    step00["step00_manifest_sha256"]
                ),
                "step01_manifest_sha256": step01["manifest_sha256"],
                "step02_manifest_sha256": step02["manifest_sha256"],
                "step03_manifest_sha256": step03["manifest_sha256"],
                "step04_manifest_sha256": step04["manifest_sha256"],
                "step05_manifest_sha256": step05["manifest_sha256"],
                "step05_panel_sha256": step05["panel_sha256"],
            },
            "feature_design": {
                "feature_count": EXPECTED_FEATURE_COUNT,
                "feature_groups": {
                    group: len(features)
                    for group, features in STOCK_FEATURE_GROUPS.items()
                },
                "feature_order": FEATURE_COLUMNS,
                "signal_timing_policy": SIGNAL_TIMING_POLICY,
                "continuity_break_dates": [
                    date.strftime("%Y-%m-%d")
                    for date in break_dates
                ],
                "continuity_segment_count": expected_segment_count,
                "calculations_restart_after_each_removed_valid_session": True,
                "scaling_applied": False,
                "targets_constructed": False,
            },
            "row_counts": {
                "input_rows": int(len(source)),
                "full_feature_rows": int(len(feature_panel)),
                "complete_feature_rows": int(len(complete_panel)),
                "complete_rows_per_stock": int(
                    expected_complete_rows_per_stock
                ),
                "warmup_rows_per_segment": (
                    EXPECTED_FIRST_COMPLETE_POSITION
                ),
            },
            "causality_audit": {
                "records": int(len(causality_audit)),
                "all_passed": bool(
                    causality_audit["all_features_match"].all()
                ),
                "maximum_absolute_difference": float(
                    causality_audit[
                        "maximum_absolute_difference"
                    ].max()
                ),
            },
            "return_consistency": {
                "stocks_checked": int(len(return_consistency)),
                "all_nonboundary_returns_match": bool(
                    return_consistency[
                        "all_nonboundary_returns_match"
                    ].all()
                ),
                "all_segment_start_returns_reset": bool(
                    return_consistency[
                        "all_segment_start_returns_reset"
                    ].all()
                ),
                "post_break_source_returns_intentionally_reset": int(
                    return_consistency[
                        "segment_starts_with_nonnull_step05_return_reset_to_nan"
                    ].sum()
                ),
            },
            "scientific_boundary": {
                "nifty_features_constructed": False,
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
                "lock_sha256_before": (
                    step00["final_test_lock_sha256"]
                ),
                "lock_sha256_after": lock_sha_after,
                "lock_unchanged": True,
            },
        }

        atomic_write_json(
            STEP06_SUMMARY_JSON,
            summary_payload,
        )

        shutil.copy2(
            SCRIPT_FILE,
            FROZEN_STEP06_SCRIPT,
        )

        summary_lines = [
            STEP_NAME,
            "=" * 94,
            "Status: PASS",
            (
                "Configuration fingerprint: "
                + step00["configuration_fingerprint_sha256"]
            ),
            (
                "Step-05 manifest SHA-256: "
                + step05["manifest_sha256"]
            ),
            (
                "Step-05 panel SHA-256: "
                + step05["panel_sha256"]
            ),
            (
                "Input rows: "
                f"{len(source)}"
            ),
            (
                "Feature count: "
                f"{EXPECTED_FEATURE_COUNT}"
            ),
            (
                "Continuity segments/stock: "
                f"{expected_segment_count}"
            ),
            (
                "Continuity breaks: "
                + (
                    ",".join(
                        date.strftime("%Y-%m-%d")
                        for date in break_dates
                    )
                    if break_dates
                    else "NONE"
                )
            ),
            (
                "Full feature rows: "
                f"{len(feature_panel)}"
            ),
            (
                "Complete feature rows: "
                f"{len(complete_panel)}"
            ),
            (
                "Complete rows/stock: "
                f"{expected_complete_rows_per_stock}"
            ),
            (
                "First complete position: "
                f"{EXPECTED_FIRST_COMPLETE_POSITION}"
            ),
            (
                "Prefix-causality tests: "
                f"{len(causality_audit)}"
            ),
            (
                "Final-test state: "
                f"{step00['final_test_state']}"
            ),
        ]

        atomic_write_text(
            SUMMARY_TXT,
            "\n".join(summary_lines) + "\n",
        )

        manifest_sha = write_step06_manifest(
            step00=step00,
            step01=step01,
            step02=step02,
            step03=step03,
            step04=step04,
            step05=step05,
        )

        stage = "final_report"

        print("[10/11] Validating frozen Step-06 artifacts...")

        if not MANIFEST_JSON.exists() or not MANIFEST_SHA.exists():
            raise RuntimeError(
                "Step-06 manifest freeze failed."
            )

        print("[11/11] Final Step-06 report...")
        print()
        print("-" * 126)
        print("CAUSAL STOCK FEATURE PANEL")
        print("-" * 126)
        print(
            "Canonical source             : "
            "Step-05 unified clean model panel"
        )
        print(
            "Input rows                   : "
            f"{len(source):,}"
        )
        print(
            "Stock count                  : "
            f"{source['Ticker'].nunique()}"
        )
        print(
            "Feature count                : "
            f"{len(FEATURE_COLUMNS)}"
        )
        print(
            "Continuity segments/stock    : "
            f"{expected_segment_count}"
        )
        print(
            "Continuity break dates       : "
            + (
                ", ".join(
                    date.strftime("%Y-%m-%d")
                    for date in break_dates
                )
                if break_dates
                else "NONE"
            )
        )
        print(
            "Full feature-panel rows      : "
            f"{len(feature_panel):,}"
        )
        print(
            "Complete feature rows        : "
            f"{len(complete_panel):,}"
        )
        print(
            "Complete rows/stock          : "
            f"{expected_complete_rows_per_stock:,}"
        )
        print(
            "First complete position      : "
            f"{EXPECTED_FIRST_COMPLETE_POSITION}"
        )
        print(
            "Prefix-causality tests       : "
            f"{len(causality_audit)}"
        )
        print(
            "Prefix-causality all pass    : "
            f"{causality_audit['all_features_match'].all()}"
        )
        print(
            "Maximum causality difference : "
            f"{causality_audit['maximum_absolute_difference'].max():.3e}"
        )
        print(
            "Return consistency all pass  : "
            f"{return_consistency['all_nonboundary_returns_match'].all()}"
        )

        print("-" * 126)
        print("FEATURE GROUPS")
        print("-" * 126)

        for group_name, features in STOCK_FEATURE_GROUPS.items():
            print(
                f"{group_name:<42}: "
                f"{len(features):>2}"
            )

        print("-" * 126)
        print("CONTINUITY SEGMENT AUDIT")
        print("-" * 126)

        segment_display = (
            segment_audit
            .groupby("continuity_segment_id")
            .agg(
                stocks=("ticker", "nunique"),
                rows_per_stock=("rows", "first"),
                start_date=("start_date", "first"),
                end_date=("end_date", "first"),
                first_complete_position=(
                    "first_complete_position",
                    "first",
                ),
                complete_rows_per_stock=(
                    "complete_feature_rows",
                    "first",
                ),
            )
            .reset_index()
        )

        for row in segment_display.itertuples(index=False):
            print(
                f"Segment {row.continuity_segment_id}: "
                f"{pd.Timestamp(row.start_date).date()} -> "
                f"{pd.Timestamp(row.end_date).date()} | "
                f"rows/stock={row.rows_per_stock:,} | "
                f"first-complete={row.first_complete_position} | "
                f"complete/stock={row.complete_rows_per_stock:,}"
            )

        print("-" * 126)
        print("OUTPUT FILES")
        print("-" * 126)
        print(f"Feature panel        : {FEATURE_PANEL_PARQUET}")
        print(f"Complete panel       : {COMPLETE_FEATURE_PANEL_PARQUET}")
        print(f"Feature dictionary   : {FEATURE_DICTIONARY_CSV}")
        print(f"Segment audit        : {SEGMENT_AUDIT_CSV}")
        print(f"Warm-up audit        : {WARMUP_AUDIT_CSV}")
        print(f"Causality audit      : {CAUSALITY_AUDIT_CSV}")
        print(f"Return consistency   : {RETURN_CONSISTENCY_CSV}")
        print(f"Manifest             : {MANIFEST_JSON}")
        print(f"Manifest SHA-256     : {manifest_sha}")
        print(f"Final-test state     : {step00['final_test_state']}")
        print("=" * 126)
        print("PASS: Exactly 47 stock-specific predictors constructed.")
        print(
            "PASS: Feature calculations restart after each removed valid "
            "market session."
        )
        print("PASS: Prefix recomputation confirms causal feature construction.")
        print(
            "PASS: Step-06 non-boundary returns reconcile with frozen "
            "Step-05 returns."
        )
        print("PASS: No targets, scaling, regimes, sequences, or models were created.")
        print("PASS: Final held-out test remains LOCKED_NOT_OPENED.")
        print("=" * 126)
        print(
            "Do not run Step 07 yet. Paste the complete terminal output for audit."
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
        print("=" * 126)
        print("STEP 06 FAILED — FEATURES NOT FROZEN")
        print("=" * 126)
        print(f"Stage      : {stage}")
        print(f"Error type : {type(exc).__name__}")
        print(f"Error      : {exc}")
        print(f"Diagnostic : {failure_path}")

        if archived_partial is not None:
            print(
                "Archived partial artifacts: "
                f"{archived_partial}"
            )

        print("=" * 126)
        print("Do not proceed to Step 07.")

        return 30


# =============================================================================
# 16. CLI / ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct and freeze the clean-V3 causal 47-feature stock panel."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an already frozen Step-06 feature panel "
            "without recomputing it."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verify_only:
        return verify_existing_step06()

    return run_new_step06()


if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print("\nStep 06 interrupted by user.")
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 126)
        print("STEP 06 UNHANDLED FAILURE")
        print("=" * 126)
        print(f"{type(exc).__name__}: {exc}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        print("=" * 126)
        sys.exit(1)
