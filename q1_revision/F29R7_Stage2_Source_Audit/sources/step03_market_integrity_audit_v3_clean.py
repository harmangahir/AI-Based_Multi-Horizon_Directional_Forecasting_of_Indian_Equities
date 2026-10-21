#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 03 — MARKET COVERAGE, CALENDAR AND INTEGRITY AUDIT (CLEAN V3)

Purpose
-------
Audit the immutable Yahoo Finance snapshots frozen by Step 02 before any
cleaning, feature construction, target construction, scaling, regime
estimation, or modelling.

This stage identifies:
- Step-00 / Step-01 / Step-02 chain integrity,
- raw-file hash integrity,
- per-column missingness,
- price-valid NIFTY sessions,
- explicitly invalid NIFTY price rows,
- stock coverage relative to the candidate NIFTY price calendar,
- stock dates missing relative to candidate NIFTY sessions,
- stock-only dates,
- invalid OHLCV observations,
- zero-volume observations,
- flat-OHLC + zero-volume observations,
- date-wide anomaly candidates under the Step-00 frozen 90% rule,
- cross-sectional date consensus,
- unusually large adjusted-price log returns,
- corporate actions and date-specific adjustment-factor changes.

IMPORTANT
---------
This stage is AUDIT ONLY.

It DOES NOT:
- delete any raw observation,
- impute any value,
- repair prices or volumes,
- finalize the modelling calendar,
- construct adjusted OHLC,
- construct technical indicators,
- construct targets,
- fit scalers,
- estimate regimes,
- train/select/evaluate models,
- score the final held-out test.

The candidate NIFTY price calendar produced here is diagnostic. Step 04 must
make the explicit cleaning/calendar decision under the already frozen Step-00
data-integrity policy.

Expected location
-----------------
<repository>/q1_revision/step03_market_integrity_audit.py

Run
---
python step03_market_integrity_audit.py

Verify an already frozen Step 03
--------------------------------
python step03_market_integrity_audit.py --verify-only
"""

from __future__ import annotations

import argparse
import csv
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
        "step03_market_integrity_audit.py must be located inside "
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

STEP_NUMBER = 3
STEP_ID = "03"
STEP_NAME = "STEP 03 — MARKET COVERAGE, CALENDAR AND INTEGRITY AUDIT"
SCHEMA_VERSION = "1.0"

PRICE_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
)

OHLC_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
)

STOCK_CORE_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)

ACTION_COLUMNS = (
    "Dividends",
    "Stock Splits",
    "Capital Gains",
)

# Diagnostic only; not a deletion threshold.
LARGE_ABS_ADJUSTED_LOG_RETURN_THRESHOLD = 0.20

# Numerical tolerance only, not a scientific anomaly threshold.
ADJUSTMENT_FACTOR_LOG_CHANGE_EPSILON = 1e-10
FLAT_OHLC_ABS_TOLERANCE = 1e-10

# These values are frozen in Step 00.
PRELIMINARY_MIN_COVERAGE = float(
    cfg.MINIMUM_BENCHMARK_COVERAGE_FRACTION
)

DATE_WIDE_ANOMALY_RULE_ENABLED = bool(
    cfg.DATE_WIDE_ANOMALY_RULE_ENABLED
)

DATE_WIDE_ANOMALY_MIN_STOCK_FRACTION = float(
    cfg.DATE_WIDE_ANOMALY_MIN_STOCK_FRACTION
)

DATE_WIDE_ANOMALY_THRESHOLD_COUNT = int(
    math.ceil(
        DATE_WIDE_ANOMALY_MIN_STOCK_FRACTION
        * len(cfg.STOCK_TICKERS)
    )
)


# =============================================================================
# 2. FROZEN INPUT LOCATIONS
# =============================================================================

RAW_SNAPSHOT_ROOT = (
    cfg.RAW_DATA_DIR
    / "provider_snapshot"
)

RAW_CSV_DIR = (
    RAW_SNAPSHOT_ROOT
    / "csv"
)

RAW_PARQUET_DIR = (
    RAW_SNAPSHOT_ROOT
    / "parquet"
)

STEP01_MANIFEST = (
    cfg.MANIFEST_DIR
    / "01_environment_audit_manifest.json"
)

STEP01_MANIFEST_SHA = (
    cfg.MANIFEST_DIR
    / "01_environment_audit_manifest.sha256"
)

STEP01_AUDIT_JSON = (
    cfg.ENVIRONMENT_DIR
    / "01_environment_audit.json"
)

STEP02_HASH_REGISTER = (
    cfg.AUDIT_DIR
    / "02_raw_file_hashes.csv"
)

STEP02_ACQUISITION_MANIFEST = (
    cfg.MANIFEST_DIR
    / "02_market_data_acquisition_manifest.json"
)

STEP02_ACQUISITION_MANIFEST_SHA = (
    cfg.MANIFEST_DIR
    / "02_market_data_acquisition_manifest.sha256"
)


# =============================================================================
# 3. STEP-03 OUTPUT LOCATIONS
# =============================================================================

COLUMN_NULL_SUMMARY_CSV = (
    cfg.AUDIT_DIR
    / "03_column_null_summary.csv"
)

NIFTY_SESSION_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "03_nifty_session_audit.csv"
)

NIFTY_INVALID_PRICE_SESSIONS_CSV = (
    cfg.AUDIT_DIR
    / "03_nifty_invalid_price_sessions.csv"
)

CANDIDATE_REFERENCE_CALENDAR_CSV = (
    cfg.REFERENCE_DATA_DIR
    / "03_candidate_nifty_price_calendar.csv"
)

STOCK_COVERAGE_SUMMARY_CSV = (
    cfg.AUDIT_DIR
    / "03_stock_coverage_summary.csv"
)

MISSING_DATES_CSV = (
    cfg.AUDIT_DIR
    / "03_missing_dates_vs_nifty.csv"
)

STOCK_ONLY_DATES_CSV = (
    cfg.AUDIT_DIR
    / "03_stock_only_dates.csv"
)

INVALID_OHLCV_CSV = (
    cfg.AUDIT_DIR
    / "03_invalid_ohlcv_rows.csv"
)

ZERO_VOLUME_CSV = (
    cfg.AUDIT_DIR
    / "03_zero_volume_rows.csv"
)

CROSS_SECTIONAL_DATE_CONSENSUS_CSV = (
    cfg.AUDIT_DIR
    / "03_cross_sectional_date_consensus.csv"
)

SYNCHRONIZED_ANOMALIES_CSV = (
    cfg.AUDIT_DIR
    / "03_synchronized_date_anomalies.csv"
)

LARGE_RETURN_FLAGS_CSV = (
    cfg.AUDIT_DIR
    / "03_large_adjusted_return_flags.csv"
)

CORPORATE_ACTION_ADJUSTMENT_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "03_corporate_action_adjustment_audit.csv"
)

SOURCE_HASH_CSV = (
    cfg.AUDIT_DIR
    / "03_source_file_hashes.csv"
)

INTEGRITY_SUMMARY_JSON = (
    cfg.AUDIT_DIR
    / "03_market_integrity_summary.json"
)

STEP03_MANIFEST_JSON = (
    cfg.MANIFEST_DIR
    / "03_market_integrity_audit_manifest.json"
)

STEP03_MANIFEST_SHA = (
    cfg.MANIFEST_DIR
    / "03_market_integrity_audit_manifest.sha256"
)

FROZEN_STEP03_SCRIPT = (
    cfg.MANIFEST_DIR
    / "step03_market_integrity_audit_frozen.py"
)

STEP03_SUMMARY_TXT = (
    cfg.REPORT_DIR
    / "03_market_integrity_audit_summary.txt"
)

FAILED_ATTEMPT_DIR = (
    cfg.AUDIT_DIR
    / "step03_failed_attempts"
)


# =============================================================================
# 4. OUTPUT SCHEMAS
# =============================================================================

COLUMN_NULL_FIELDS = [
    "ticker",
    "instrument_type",
    "column",
    "row_count",
    "null_count",
    "null_percent",
]

NIFTY_AUDIT_FIELDS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "price_null",
    "nonpositive_price",
    "high_below_ohlc",
    "low_above_ohlc",
    "price_valid",
    "volume_null",
    "volume_zero",
    "volume_negative",
    "candidate_reference_session",
]

NIFTY_INVALID_FIELDS = NIFTY_AUDIT_FIELDS

CANDIDATE_CALENDAR_FIELDS = [
    "session_number",
    "Date",
    "calendar_year",
    "calendar_month",
    "day_of_week",
]

COVERAGE_FIELDS = [
    "ticker",
    "raw_rows",
    "raw_unique_dates",
    "raw_first_date",
    "raw_last_date",
    "reference_session_count",
    "reference_sessions_present",
    "reference_sessions_missing",
    "stock_only_dates",
    "date_coverage_fraction",
    "date_coverage_percent",
    "structurally_valid_reference_rows",
    "structural_valid_coverage_fraction",
    "structural_valid_coverage_percent",
    "structurally_invalid_reference_rows",
    "zero_volume_reference_rows",
    "zero_volume_flat_ohlc_reference_rows",
    "duplicate_date_rows",
    "preliminary_date_coverage_eligible",
    "preliminary_structural_coverage_eligible",
]

MISSING_DATE_FIELDS = [
    "ticker",
    "Date",
]

STOCK_ONLY_FIELDS = [
    "ticker",
    "Date",
    "nifty_raw_row_present",
    "nifty_price_valid",
    "Volume",
    "volume_zero",
    "flat_ohlc",
    "zero_volume_flat_ohlc",
    "structurally_invalid_ohlcv",
]

INVALID_OHLCV_FIELDS = [
    "ticker",
    "Date",
    "is_candidate_nifty_session",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "price_null",
    "nonpositive_price",
    "high_below_ohlc",
    "low_above_ohlc",
    "volume_null",
    "volume_negative",
]

ZERO_VOLUME_FIELDS = [
    "ticker",
    "Date",
    "is_candidate_nifty_session",
    "is_stock_only_date",
    "Volume",
    "flat_ohlc",
    "zero_volume_flat_ohlc",
]

DATE_CONSENSUS_FIELDS = [
    "Date",
    "nifty_raw_row_present",
    "nifty_price_valid",
    "candidate_nifty_session",
    "stocks_present",
    "stocks_nonzero_volume",
    "stocks_zero_volume",
    "stocks_flat_ohlc",
    "stocks_zero_volume_flat_ohlc",
    "stocks_structurally_invalid",
    "present_fraction",
    "zero_volume_fraction_of_panel",
    "zero_volume_flat_ohlc_fraction_of_panel",
    "structurally_invalid_fraction_of_panel",
    "date_wide_anomaly_threshold_count",
    "date_wide_anomaly_candidate",
    "widespread_stock_only_date_alert",
    "nifty_invalid_but_widespread_stock_presence",
    "present_tickers",
    "zero_volume_tickers",
    "zero_volume_flat_ohlc_tickers",
    "invalid_tickers",
]

SYNCHRONIZED_FIELDS = DATE_CONSENSUS_FIELDS

LARGE_RETURN_FIELDS = [
    "ticker",
    "previous_date",
    "Date",
    "previous_adjusted_close",
    "adjusted_close",
    "adjusted_log_return",
    "absolute_adjusted_log_return",
    "raw_close_log_return",
    "threshold",
    "current_is_candidate_nifty_session",
    "previous_is_candidate_nifty_session",
    "dividend_current",
    "split_current",
    "capital_gain_current",
    "dividend_previous",
    "split_previous",
    "capital_gain_previous",
    "corporate_action_current_or_previous",
]

CORPORATE_ACTION_FIELDS = [
    "ticker",
    "Date",
    "Close",
    "Adj Close",
    "adjustment_factor",
    "previous_adjustment_factor",
    "adjustment_factor_log_change",
    "Dividends",
    "Stock Splits",
    "Capital Gains",
    "corporate_action_event",
    "adjustment_factor_changed",
    "current_is_candidate_nifty_session",
]

SOURCE_HASH_FIELDS = [
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
    return datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    ).isoformat()


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            text
        )

    os.replace(
        temporary,
        path,
    )


def atomic_write_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write(
            "\n"
        )

    os.replace(
        temporary,
        path,
    )


def write_dataframe_csv(
    path: Path,
    dataframe: pd.DataFrame,
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    output = dataframe.copy()

    for field in fieldnames:
        if field not in output.columns:
            output[
                field
            ] = pd.NA

    output = output[
        fieldnames
    ]

    output.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d",
    )

    os.replace(
        temporary,
        path,
    )


def rows_to_dataframe(
    rows: list[Dict[str, Any]],
    fieldnames: list[str],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=fieldnames
        )

    frame = pd.DataFrame(
        rows
    )

    for field in fieldnames:
        if field not in frame.columns:
            frame[
                field
            ] = pd.NA

    return frame[
        fieldnames
    ]


def get_distribution_version(
    distribution_name: str,
) -> Optional[str]:
    try:
        return importlib_metadata.version(
            distribution_name
        )
    except importlib_metadata.PackageNotFoundError:
        return None


def safe_ticker_filename(
    ticker: str,
) -> str:
    return (
        ticker
        .replace("^", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def relative_to_experiment(
    path: Path,
) -> str:
    return str(
        path.resolve()
        .relative_to(
            cfg.EXPERIMENT_ROOT_DIR.resolve()
        )
    )


def numeric_series(
    frame: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            default,
            index=frame.index,
            dtype="float64",
        )

    return pd.to_numeric(
        frame[
            column
        ],
        errors="coerce",
    )


def boolean_to_int(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .fillna(False)
        .astype(bool)
        .astype(np.int64)
    )


# =============================================================================
# 6. STEP-00 / FINAL-TEST LOCK VERIFICATION
# =============================================================================

def verify_step00_contract() -> Dict[str, Any]:
    cfg.validate_configuration()

    configuration_payload = (
        cfg.build_configuration_payload()
    )

    configuration_fingerprint = (
        cfg.canonical_json_sha256(
            configuration_payload
        )
    )

    (
        manifest_path,
        manifest_sha256,
        freeze_status,
    ) = cfg.verify_existing_freeze(
        configuration_payload=(
            configuration_payload
        ),
        configuration_fingerprint=(
            configuration_fingerprint
        ),
    )

    if freeze_status != "VERIFIED_EXISTING":
        raise RuntimeError(
            "Step-00 freeze is not VERIFIED_EXISTING."
        )

    lock_path = (
        cfg.get_final_test_lock_path()
    )

    with lock_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        lock_payload = json.load(
            handle
        )

    if (
        lock_payload.get("state")
        != "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError(
            "Final-test state is not LOCKED_NOT_OPENED."
        )

    if (
        lock_payload.get(
            "final_test_opened"
        )
        is not False
    ):
        raise RuntimeError(
            "Final-test lock indicates final_test_opened != False."
        )

    if (
        lock_payload.get(
            "final_test_scored"
        )
        is not False
    ):
        raise RuntimeError(
            "Final-test lock indicates final_test_scored != False."
        )

    return {
        "configuration_fingerprint_sha256": (
            configuration_fingerprint
        ),
        "step00_manifest_path": (
            str(
                manifest_path
            )
        ),
        "step00_manifest_sha256": (
            manifest_sha256
        ),
        "step00_freeze_status": (
            freeze_status
        ),
        "final_test_lock_path": (
            str(
                lock_path
            )
        ),
        "final_test_lock_sha256": (
            sha256_file(
                lock_path
            )
        ),
        "final_test_state": (
            lock_payload.get(
                "state"
            )
        ),
        "final_test_opened": (
            lock_payload.get(
                "final_test_opened"
            )
        ),
        "final_test_scored": (
            lock_payload.get(
                "final_test_scored"
            )
        ),
    }


# =============================================================================
# 7. STEP-01 ENVIRONMENT VERIFICATION
# =============================================================================

def verify_step01_freeze(
    expected_configuration_fingerprint: str,
) -> Dict[str, Any]:
    if not STEP01_MANIFEST.exists():
        raise RuntimeError(
            "Step-01 manifest is missing."
        )

    if not STEP01_MANIFEST_SHA.exists():
        raise RuntimeError(
            "Step-01 manifest SHA sidecar is missing."
        )

    if not STEP01_AUDIT_JSON.exists():
        raise RuntimeError(
            "Step-01 environment audit JSON is missing."
        )

    with STEP01_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    if (
        manifest.get(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Stored Step-01 status is not PASS."
        )

    if (
        manifest.get(
            "configuration_fingerprint_sha256"
        )
        != expected_configuration_fingerprint
    ):
        raise RuntimeError(
            "Step-01 configuration fingerprint mismatch."
        )

    expected_manifest_sha = (
        STEP01_MANIFEST_SHA
        .read_text(
            encoding="utf-8"
        )
        .strip()
        .split()[0]
    )

    actual_manifest_sha = (
        sha256_file(
            STEP01_MANIFEST
        )
    )

    if (
        expected_manifest_sha
        != actual_manifest_sha
    ):
        raise RuntimeError(
            "Step-01 manifest SHA mismatch."
        )

    for name, record in (
        manifest
        .get(
            "artifacts",
            {},
        )
        .items()
    ):
        path = Path(
            record[
                "path"
            ]
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing Step-01 artifact: {name}"
            )

        if (
            sha256_file(
                path
            )
            != record[
                "sha256"
            ]
        ):
            raise RuntimeError(
                f"Step-01 artifact hash mismatch: {name}"
            )

    with STEP01_AUDIT_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        audit = json.load(
            handle
        )

    stored_python = (
        audit
        .get(
            "system",
            {},
        )
        .get(
            "python_executable"
        )
    )

    if (
        stored_python
        and Path(
            stored_python
        ).resolve()
        != Path(
            sys.executable
        ).resolve()
    ):
        raise RuntimeError(
            "Current Python executable differs from Step 01.\n"
            f"Step 01: {stored_python}\n"
            f"Current : {sys.executable}"
        )

    package_drift = []

    for row in audit.get(
        "packages",
        [],
    ):
        if not row.get(
            "required",
            False,
        ):
            continue

        distribution = (
            row.get(
                "distribution"
            )
        )

        stored_version = (
            row.get(
                "version",
                "",
            )
        )

        current_version = (
            get_distribution_version(
                distribution
            )
        )

        if (
            current_version
            != stored_version
        ):
            package_drift.append(
                (
                    distribution,
                    stored_version,
                    current_version,
                )
            )

    if package_drift:
        lines = [
            (
                f"{name}: "
                f"Step01={stored!r}, "
                f"current={current!r}"
            )
            for (
                name,
                stored,
                current,
            )
            in package_drift
        ]

        raise RuntimeError(
            "Required-package drift after Step 01:\n"
            + "\n".join(
                lines
            )
        )

    return {
        "step01_manifest_path": (
            str(
                STEP01_MANIFEST
            )
        ),
        "step01_manifest_sha256": (
            actual_manifest_sha
        ),
        "python_executable": (
            stored_python
        ),
        "conda_environment": (
            audit
            .get(
                "system",
                {},
            )
            .get(
                "conda_default_env"
            )
        ),
    }


# =============================================================================
# 8. STEP-02 ACQUISITION FREEZE VERIFICATION
# =============================================================================

def verify_step02_freeze(
    expected_configuration_fingerprint: str,
    expected_step01_manifest_sha256: str,
) -> Dict[str, Any]:
    if not STEP02_ACQUISITION_MANIFEST.exists():
        raise RuntimeError(
            "Step-02 acquisition manifest is missing."
        )

    if not STEP02_ACQUISITION_MANIFEST_SHA.exists():
        raise RuntimeError(
            "Step-02 manifest SHA sidecar is missing."
        )

    with STEP02_ACQUISITION_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    expected_manifest_sha = (
        STEP02_ACQUISITION_MANIFEST_SHA
        .read_text(
            encoding="utf-8"
        )
        .strip()
        .split()[0]
    )

    actual_manifest_sha = (
        sha256_file(
            STEP02_ACQUISITION_MANIFEST
        )
    )

    if (
        expected_manifest_sha
        != actual_manifest_sha
    ):
        raise RuntimeError(
            "Step-02 manifest SHA mismatch."
        )

    if (
        manifest
        .get(
            "step",
            {},
        )
        .get(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Stored Step-02 status is not PASS."
        )

    if (
        manifest
        .get(
            "step00",
            {},
        )
        .get(
            "configuration_fingerprint_sha256"
        )
        != expected_configuration_fingerprint
    ):
        raise RuntimeError(
            "Step-02 / Step-00 fingerprint mismatch."
        )

    if (
        manifest
        .get(
            "step01",
            {},
        )
        .get(
            "step01_manifest_sha256"
        )
        != expected_step01_manifest_sha256
    ):
        raise RuntimeError(
            "Step-02 / Step-01 manifest mismatch."
        )

    if (
        manifest
        .get(
            "final_test_lock_integrity",
            {},
        )
        .get(
            "lock_unchanged"
        )
        is not True
    ):
        raise RuntimeError(
            "Step-02 did not preserve final-test lock integrity."
        )

    failures = []

    for name, record in (
        manifest
        .get(
            "canonical_artifacts",
            {},
        )
        .items()
    ):
        if name == "raw_snapshot_root":
            continue

        path = Path(
            record[
                "path"
            ]
        )

        if not path.exists():
            failures.append(
                f"missing artifact: {name}"
            )
            continue

        if (
            sha256_file(
                path
            )
            != record[
                "sha256"
            ]
        ):
            failures.append(
                f"hash mismatch: {name}"
            )

    raw_file_records = (
        manifest.get(
            "raw_file_records",
            [],
        )
    )

    for record in raw_file_records:
        path = Path(
            record[
                "file_path"
            ]
        )

        if not path.exists():
            failures.append(
                f"missing raw file: {path}"
            )
            continue

        if (
            sha256_file(
                path
            )
            != record[
                "sha256"
            ]
        ):
            failures.append(
                f"raw hash mismatch: {path}"
            )

    if failures:
        raise RuntimeError(
            "Step-02 artifact verification failed:\n"
            + "\n".join(
                failures
            )
        )

    return {
        "step02_manifest_path": (
            str(
                STEP02_ACQUISITION_MANIFEST
            )
        ),
        "step02_manifest_sha256": (
            actual_manifest_sha
        ),
        "verified_raw_file_count": int(
            len(
                raw_file_records
            )
        ),
        "raw_snapshot_root": (
            str(
                RAW_SNAPSHOT_ROOT
            )
        ),
    }


# =============================================================================
# 9. OUTPUT IMMUTABILITY GUARD
# =============================================================================

def canonical_step03_outputs() -> tuple[Path, ...]:
    return (
        COLUMN_NULL_SUMMARY_CSV,
        NIFTY_SESSION_AUDIT_CSV,
        NIFTY_INVALID_PRICE_SESSIONS_CSV,
        CANDIDATE_REFERENCE_CALENDAR_CSV,
        STOCK_COVERAGE_SUMMARY_CSV,
        MISSING_DATES_CSV,
        STOCK_ONLY_DATES_CSV,
        INVALID_OHLCV_CSV,
        ZERO_VOLUME_CSV,
        CROSS_SECTIONAL_DATE_CONSENSUS_CSV,
        SYNCHRONIZED_ANOMALIES_CSV,
        LARGE_RETURN_FLAGS_CSV,
        CORPORATE_ACTION_ADJUSTMENT_AUDIT_CSV,
        SOURCE_HASH_CSV,
        INTEGRITY_SUMMARY_JSON,
        STEP03_MANIFEST_JSON,
        STEP03_MANIFEST_SHA,
        FROZEN_STEP03_SCRIPT,
        STEP03_SUMMARY_TXT,
    )


def assert_new_step03_target() -> None:
    collisions = [
        path
        for path
        in canonical_step03_outputs()
        if path.exists()
    ]

    if collisions:
        collision_text = "\n".join(
            str(
                path
            )
            for path
            in collisions
        )

        raise RuntimeError(
            "Step-03 canonical output(s) already exist. "
            "Refusing to overwrite the frozen audit.\n"
            f"{collision_text}\n\n"
            "If Step 03 completed successfully, run --verify-only."
        )


# =============================================================================
# 10. LOAD RAW SNAPSHOTS
# =============================================================================

def load_raw_frame(
    ticker: str,
) -> pd.DataFrame:
    safe_name = (
        safe_ticker_filename(
            ticker
        )
    )

    path = (
        RAW_PARQUET_DIR
        / f"{safe_name}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw Parquet file for {ticker}: {path}"
        )

    frame = pd.read_parquet(
        path,
        engine="pyarrow",
    )

    if "Date" not in frame.columns:
        raise AssertionError(
            f"{ticker}: Date column missing."
        )

    frame[
        "Date"
    ] = pd.to_datetime(
        frame[
            "Date"
        ],
        errors="raise",
    ).dt.normalize()

    if (
        frame[
            "Date"
        ]
        .duplicated(
            keep=False
        )
        .any()
    ):
        raise AssertionError(
            f"{ticker}: duplicate dates detected."
        )

    frame = (
        frame
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    return frame


# =============================================================================
# 11. COLUMN-MISSINGNESS AUDIT
# =============================================================================

def build_column_null_summary(
    frames: Dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    rows = []

    for (
        ticker,
        frame,
    ) in frames.items():

        instrument_type = (
            "benchmark_index"
            if ticker
            == cfg.MARKET_INDEX_TICKER
            else "stock"
        )

        for column in frame.columns:
            if column == "Date":
                continue

            null_count = int(
                frame[
                    column
                ]
                .isna()
                .sum()
            )

            rows.append(
                {
                    "ticker": ticker,
                    "instrument_type": (
                        instrument_type
                    ),
                    "column": column,
                    "row_count": int(
                        len(
                            frame
                        )
                    ),
                    "null_count": (
                        null_count
                    ),
                    "null_percent": (
                        100.0
                        * null_count
                        / len(
                            frame
                        )
                        if len(
                            frame
                        )
                        else np.nan
                    ),
                }
            )

    return rows_to_dataframe(
        rows,
        COLUMN_NULL_FIELDS,
    )


# =============================================================================
# 12. PRICE / OHLC / VOLUME DIAGNOSTICS
# =============================================================================

def calculate_price_validity(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    missing = [
        column
        for column
        in PRICE_COLUMNS
        if column
        not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing price columns: "
            + ", ".join(
                missing
            )
        )

    result = pd.DataFrame(
        {
            "Date": (
                frame[
                    "Date"
                ]
            )
        }
    )

    price_frame = (
        frame[
            list(
                PRICE_COLUMNS
            )
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    result[
        "price_null"
    ] = (
        price_frame
        .isna()
        .any(
            axis=1
        )
    )

    result[
        "nonpositive_price"
    ] = (
        (
            price_frame
            <= 0
        )
        .any(
            axis=1
        )
    )

    result[
        "high_below_ohlc"
    ] = (
        price_frame[
            "High"
        ]
        <
        price_frame[
            [
                "Open",
                "Low",
                "Close",
            ]
        ]
        .max(
            axis=1
        )
    )

    result[
        "low_above_ohlc"
    ] = (
        price_frame[
            "Low"
        ]
        >
        price_frame[
            [
                "Open",
                "High",
                "Close",
            ]
        ]
        .min(
            axis=1
        )
    )

    result[
        "price_valid"
    ] = ~(
        result[
            [
                "price_null",
                "nonpositive_price",
                "high_below_ohlc",
                "low_above_ohlc",
            ]
        ]
        .any(
            axis=1
        )
    )

    return result


def calculate_flat_ohlc(
    frame: pd.DataFrame,
) -> pd.Series:
    missing = [
        column
        for column
        in OHLC_COLUMNS
        if column
        not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing OHLC columns: "
            + ", ".join(
                missing
            )
        )

    values = (
        frame[
            list(
                OHLC_COLUMNS
            )
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    valid = ~(
        values
        .isna()
        .any(
            axis=1
        )
    )

    row_range = (
        values
        .max(
            axis=1
        )
        -
        values
        .min(
            axis=1
        )
    )

    return (
        valid
        &
        (
            row_range.abs()
            <= FLAT_OHLC_ABS_TOLERANCE
        )
    )


def stock_row_diagnostics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    price_diag = (
        calculate_price_validity(
            frame
        )
    )

    volume = (
        numeric_series(
            frame,
            "Volume",
        )
    )

    result = (
        price_diag.copy()
    )

    result[
        "volume_null"
    ] = (
        volume.isna()
    )

    result[
        "volume_negative"
    ] = (
        volume
        < 0
    )

    result[
        "volume_zero"
    ] = (
        volume
        == 0
    )

    result[
        "flat_ohlc"
    ] = (
        calculate_flat_ohlc(
            frame
        )
        .to_numpy()
    )

    result[
        "zero_volume_flat_ohlc"
    ] = (
        result[
            "volume_zero"
        ]
        &
        result[
            "flat_ohlc"
        ]
    )

    # Zero volume is intentionally audited separately rather than treated as
    # structurally invalid. The Step-00 date-wide anomaly policy decides
    # whether synchronized zero-volume + flat-OHLC dates are removable.
    result[
        "structurally_invalid_ohlcv"
    ] = (
        result[
            "price_null"
        ]
        |
        result[
            "nonpositive_price"
        ]
        |
        result[
            "high_below_ohlc"
        ]
        |
        result[
            "low_above_ohlc"
        ]
        |
        result[
            "volume_null"
        ]
        |
        result[
            "volume_negative"
        ]
    )

    return result


# =============================================================================
# 13. NIFTY SESSION AUDIT
# =============================================================================

def audit_nifty_sessions(
    benchmark: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DatetimeIndex,
    pd.DataFrame,
]:
    diagnostics = (
        calculate_price_validity(
            benchmark
        )
    )

    selected_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    result = (
        benchmark[
            selected_columns
        ]
        .copy()
        .merge(
            diagnostics,
            on="Date",
            how="left",
            validate="one_to_one",
        )
    )

    volume = (
        numeric_series(
            result,
            "Volume",
        )
    )

    result[
        "volume_null"
    ] = (
        volume.isna()
    )

    result[
        "volume_zero"
    ] = (
        volume
        == 0
    )

    result[
        "volume_negative"
    ] = (
        volume
        < 0
    )

    # Index volume is provider-dependent and is therefore NOT used to define
    # a candidate NIFTY trading session. Price validity is the criterion.
    result[
        "candidate_reference_session"
    ] = (
        result[
            "price_valid"
        ]
        .astype(
            bool
        )
    )

    valid_dates = pd.DatetimeIndex(
        result.loc[
            result[
                "candidate_reference_session"
            ],
            "Date",
        ]
        .sort_values()
        .unique()
    )

    invalid_rows = (
        result.loc[
            ~result[
                "candidate_reference_session"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    calendar = pd.DataFrame(
        {
            "session_number": np.arange(
                1,
                len(
                    valid_dates
                ) + 1,
                dtype=np.int64,
            ),
            "Date": (
                valid_dates
            ),
        }
    )

    calendar[
        "calendar_year"
    ] = (
        calendar[
            "Date"
        ]
        .dt.year
    )

    calendar[
        "calendar_month"
    ] = (
        calendar[
            "Date"
        ]
        .dt.month
    )

    calendar[
        "day_of_week"
    ] = (
        calendar[
            "Date"
        ]
        .dt.day_name()
    )

    return (
        result,
        valid_dates,
        calendar,
    )


# =============================================================================
# 14. STOCK COVERAGE / DATE AUDIT
# =============================================================================

def audit_stock_coverage(
    ticker: str,
    frame: pd.DataFrame,
    reference_dates: pd.DatetimeIndex,
    benchmark_audit: pd.DataFrame,
) -> Tuple[
    Dict[str, Any],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
]:
    stock_dates = pd.DatetimeIndex(
        frame[
            "Date"
        ]
        .unique()
    )

    reference_set = set(
        reference_dates
    )

    stock_set = set(
        stock_dates
    )

    missing_dates = sorted(
        reference_set
        - stock_set
    )

    stock_only_dates = sorted(
        stock_set
        - reference_set
    )

    common_dates = (
        reference_set
        & stock_set
    )

    diagnostics = (
        stock_row_diagnostics(
            frame
        )
    )

    diag = (
        frame[
            [
                "Date",
                "Volume",
            ]
        ]
        .merge(
            diagnostics,
            on="Date",
            how="left",
            validate="one_to_one",
        )
    )

    on_reference = (
        diag[
            "Date"
        ]
        .isin(
            reference_dates
        )
    )

    reference_rows = (
        diag.loc[
            on_reference
        ]
        .copy()
    )

    invalid_reference_rows = int(
        reference_rows[
            "structurally_invalid_ohlcv"
        ]
        .sum()
    )

    structurally_valid_reference_rows = int(
        (
            ~reference_rows[
                "structurally_invalid_ohlcv"
            ]
        )
        .sum()
    )

    zero_volume_reference_rows = int(
        reference_rows[
            "volume_zero"
        ]
        .sum()
    )

    zero_volume_flat_reference_rows = int(
        reference_rows[
            "zero_volume_flat_ohlc"
        ]
        .sum()
    )

    duplicate_date_rows = int(
        frame[
            "Date"
        ]
        .duplicated(
            keep=False
        )
        .sum()
    )

    reference_count = int(
        len(
            reference_dates
        )
    )

    date_coverage = (
        len(
            common_dates
        )
        / reference_count
        if reference_count
        else np.nan
    )

    structural_valid_coverage = (
        structurally_valid_reference_rows
        / reference_count
        if reference_count
        else np.nan
    )

    summary = {
        "ticker": ticker,
        "raw_rows": int(
            len(
                frame
            )
        ),
        "raw_unique_dates": int(
            frame[
                "Date"
            ]
            .nunique()
        ),
        "raw_first_date": (
            frame[
                "Date"
            ]
            .min()
            .strftime(
                "%Y-%m-%d"
            )
        ),
        "raw_last_date": (
            frame[
                "Date"
            ]
            .max()
            .strftime(
                "%Y-%m-%d"
            )
        ),
        "reference_session_count": (
            reference_count
        ),
        "reference_sessions_present": int(
            len(
                common_dates
            )
        ),
        "reference_sessions_missing": int(
            len(
                missing_dates
            )
        ),
        "stock_only_dates": int(
            len(
                stock_only_dates
            )
        ),
        "date_coverage_fraction": float(
            date_coverage
        ),
        "date_coverage_percent": float(
            100.0
            * date_coverage
        ),
        "structurally_valid_reference_rows": (
            structurally_valid_reference_rows
        ),
        "structural_valid_coverage_fraction": float(
            structural_valid_coverage
        ),
        "structural_valid_coverage_percent": float(
            100.0
            * structural_valid_coverage
        ),
        "structurally_invalid_reference_rows": (
            invalid_reference_rows
        ),
        "zero_volume_reference_rows": (
            zero_volume_reference_rows
        ),
        "zero_volume_flat_ohlc_reference_rows": (
            zero_volume_flat_reference_rows
        ),
        "duplicate_date_rows": (
            duplicate_date_rows
        ),
        "preliminary_date_coverage_eligible": bool(
            date_coverage
            >= PRELIMINARY_MIN_COVERAGE
        ),
        "preliminary_structural_coverage_eligible": bool(
            structural_valid_coverage
            >= PRELIMINARY_MIN_COVERAGE
        ),
    }

    missing_rows = [
        {
            "ticker": ticker,
            "Date": date,
        }
        for date
        in missing_dates
    ]

    benchmark_lookup = (
        benchmark_audit
        .set_index(
            "Date"
        )
    )

    stock_lookup = (
        diag
        .set_index(
            "Date"
        )
    )

    stock_only_rows = []

    for date in stock_only_dates:
        stock_row = (
            stock_lookup
            .loc[
                date
            ]
        )

        if isinstance(
            stock_row,
            pd.DataFrame,
        ):
            raise RuntimeError(
                f"{ticker}: duplicate stock-only date {date}"
            )

        nifty_raw_present = (
            date
            in benchmark_lookup.index
        )

        nifty_price_valid = (
            bool(
                benchmark_lookup
                .loc[
                    date,
                    "price_valid",
                ]
            )
            if nifty_raw_present
            else False
        )

        stock_only_rows.append(
            {
                "ticker": ticker,
                "Date": date,
                "nifty_raw_row_present": bool(
                    nifty_raw_present
                ),
                "nifty_price_valid": bool(
                    nifty_price_valid
                ),
                "Volume": (
                    stock_row[
                        "Volume"
                    ]
                ),
                "volume_zero": bool(
                    stock_row[
                        "volume_zero"
                    ]
                ),
                "flat_ohlc": bool(
                    stock_row[
                        "flat_ohlc"
                    ]
                ),
                "zero_volume_flat_ohlc": bool(
                    stock_row[
                        "zero_volume_flat_ohlc"
                    ]
                ),
                "structurally_invalid_ohlcv": bool(
                    stock_row[
                        "structurally_invalid_ohlcv"
                    ]
                ),
            }
        )

    return (
        summary,
        missing_rows,
        stock_only_rows,
    )


# =============================================================================
# 15. INVALID / ZERO-VOLUME ROW AUDIT
# =============================================================================

def audit_stock_rows(
    ticker: str,
    frame: pd.DataFrame,
    reference_dates: pd.DatetimeIndex,
) -> Tuple[
    list[Dict[str, Any]],
    list[Dict[str, Any]],
]:
    diagnostics = (
        stock_row_diagnostics(
            frame
        )
    )

    merged = (
        frame[
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]
        ]
        .merge(
            diagnostics,
            on="Date",
            how="left",
            validate="one_to_one",
        )
    )

    reference_set = set(
        reference_dates
    )

    invalid_rows = []
    zero_volume_rows = []

    for _, row in merged.iterrows():
        date = pd.Timestamp(
            row[
                "Date"
            ]
        )

        is_reference = (
            date
            in reference_set
        )

        if bool(
            row[
                "structurally_invalid_ohlcv"
            ]
        ):
            invalid_rows.append(
                {
                    "ticker": ticker,
                    "Date": date,
                    "is_candidate_nifty_session": bool(
                        is_reference
                    ),
                    "Open": row[
                        "Open"
                    ],
                    "High": row[
                        "High"
                    ],
                    "Low": row[
                        "Low"
                    ],
                    "Close": row[
                        "Close"
                    ],
                    "Adj Close": row[
                        "Adj Close"
                    ],
                    "Volume": row[
                        "Volume"
                    ],
                    "price_null": bool(
                        row[
                            "price_null"
                        ]
                    ),
                    "nonpositive_price": bool(
                        row[
                            "nonpositive_price"
                        ]
                    ),
                    "high_below_ohlc": bool(
                        row[
                            "high_below_ohlc"
                        ]
                    ),
                    "low_above_ohlc": bool(
                        row[
                            "low_above_ohlc"
                        ]
                    ),
                    "volume_null": bool(
                        row[
                            "volume_null"
                        ]
                    ),
                    "volume_negative": bool(
                        row[
                            "volume_negative"
                        ]
                    ),
                }
            )

        if bool(
            row[
                "volume_zero"
            ]
        ):
            zero_volume_rows.append(
                {
                    "ticker": ticker,
                    "Date": date,
                    "is_candidate_nifty_session": bool(
                        is_reference
                    ),
                    "is_stock_only_date": bool(
                        not is_reference
                    ),
                    "Volume": row[
                        "Volume"
                    ],
                    "flat_ohlc": bool(
                        row[
                            "flat_ohlc"
                        ]
                    ),
                    "zero_volume_flat_ohlc": bool(
                        row[
                            "zero_volume_flat_ohlc"
                        ]
                    ),
                }
            )

    return (
        invalid_rows,
        zero_volume_rows,
    )


# =============================================================================
# 16. LARGE ADJUSTED-RETURN AUDIT
# =============================================================================

def action_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return (
        numeric_series(
            frame,
            column,
            default=0.0,
        )
        .fillna(
            0.0
        )
    )


def audit_large_adjusted_returns(
    ticker: str,
    frame: pd.DataFrame,
    reference_dates: pd.DatetimeIndex,
) -> list[Dict[str, Any]]:
    work = (
        frame
        .copy()
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    adjusted_close = (
        numeric_series(
            work,
            "Adj Close",
        )
    )

    close = (
        numeric_series(
            work,
            "Close",
        )
    )

    work[
        "_adj"
    ] = (
        adjusted_close
    )

    work[
        "_close"
    ] = (
        close
    )

    work[
        "_prev_date"
    ] = (
        work[
            "Date"
        ]
        .shift(
            1
        )
    )

    work[
        "_prev_adj"
    ] = (
        work[
            "_adj"
        ]
        .shift(
            1
        )
    )

    work[
        "_prev_close"
    ] = (
        work[
            "_close"
        ]
        .shift(
            1
        )
    )

    valid_adj_pair = (
        (
            work[
                "_adj"
            ]
            > 0
        )
        &
        (
            work[
                "_prev_adj"
            ]
            > 0
        )
    )

    valid_close_pair = (
        (
            work[
                "_close"
            ]
            > 0
        )
        &
        (
            work[
                "_prev_close"
            ]
            > 0
        )
    )

    work[
        "_adj_log_return"
    ] = (
        np.where(
            valid_adj_pair,
            np.log(
                work[
                    "_adj"
                ]
                /
                work[
                    "_prev_adj"
                ]
            ),
            np.nan,
        )
    )

    work[
        "_raw_close_log_return"
    ] = (
        np.where(
            valid_close_pair,
            np.log(
                work[
                    "_close"
                ]
                /
                work[
                    "_prev_close"
                ]
            ),
            np.nan,
        )
    )

    for column in ACTION_COLUMNS:
        work[
            f"_{column}_current"
        ] = (
            action_series(
                work,
                column,
            )
        )

        work[
            f"_{column}_previous"
        ] = (
            work[
                f"_{column}_current"
            ]
            .shift(
                1
            )
            .fillna(
                0.0
            )
        )

    flags = (
        work.loc[
            pd.Series(
                work[
                    "_adj_log_return"
                ],
                index=work.index,
            )
            .abs()
            >
            LARGE_ABS_ADJUSTED_LOG_RETURN_THRESHOLD
        ]
        .copy()
    )

    reference_set = set(
        reference_dates
    )

    rows = []

    for _, row in flags.iterrows():
        current_date = pd.Timestamp(
            row[
                "Date"
            ]
        )

        previous_date = (
            pd.Timestamp(
                row[
                    "_prev_date"
                ]
            )
            if pd.notna(
                row[
                    "_prev_date"
                ]
            )
            else pd.NaT
        )

        dividend_current = float(
            row[
                "_Dividends_current"
            ]
        )

        split_current = float(
            row[
                "_Stock Splits_current"
            ]
        )

        capital_current = float(
            row[
                "_Capital Gains_current"
            ]
        )

        dividend_previous = float(
            row[
                "_Dividends_previous"
            ]
        )

        split_previous = float(
            row[
                "_Stock Splits_previous"
            ]
        )

        capital_previous = float(
            row[
                "_Capital Gains_previous"
            ]
        )

        action_nearby = any(
            value != 0.0
            for value in (
                dividend_current,
                split_current,
                capital_current,
                dividend_previous,
                split_previous,
                capital_previous,
            )
        )

        rows.append(
            {
                "ticker": ticker,
                "previous_date": (
                    previous_date
                ),
                "Date": (
                    current_date
                ),
                "previous_adjusted_close": float(
                    row[
                        "_prev_adj"
                    ]
                ),
                "adjusted_close": float(
                    row[
                        "_adj"
                    ]
                ),
                "adjusted_log_return": float(
                    row[
                        "_adj_log_return"
                    ]
                ),
                "absolute_adjusted_log_return": float(
                    abs(
                        row[
                            "_adj_log_return"
                        ]
                    )
                ),
                "raw_close_log_return": (
                    float(
                        row[
                            "_raw_close_log_return"
                        ]
                    )
                    if pd.notna(
                        row[
                            "_raw_close_log_return"
                        ]
                    )
                    else np.nan
                ),
                "threshold": (
                    LARGE_ABS_ADJUSTED_LOG_RETURN_THRESHOLD
                ),
                "current_is_candidate_nifty_session": bool(
                    current_date
                    in reference_set
                ),
                "previous_is_candidate_nifty_session": bool(
                    previous_date
                    in reference_set
                    if pd.notna(
                        previous_date
                    )
                    else False
                ),
                "dividend_current": (
                    dividend_current
                ),
                "split_current": (
                    split_current
                ),
                "capital_gain_current": (
                    capital_current
                ),
                "dividend_previous": (
                    dividend_previous
                ),
                "split_previous": (
                    split_previous
                ),
                "capital_gain_previous": (
                    capital_previous
                ),
                "corporate_action_current_or_previous": bool(
                    action_nearby
                ),
            }
        )

    return rows


# =============================================================================
# 17. CORPORATE-ACTION / ADJUSTMENT-FACTOR AUDIT
# =============================================================================

def audit_corporate_actions_and_adjustment_factors(
    ticker: str,
    frame: pd.DataFrame,
    reference_dates: pd.DatetimeIndex,
) -> list[Dict[str, Any]]:
    work = (
        frame
        .copy()
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    close = (
        numeric_series(
            work,
            "Close",
        )
    )

    adjusted = (
        numeric_series(
            work,
            "Adj Close",
        )
    )

    factor = np.where(
        (
            close > 0
        )
        &
        adjusted.notna(),
        adjusted
        / close,
        np.nan,
    )

    work[
        "_factor"
    ] = (
        factor
    )

    work[
        "_prev_factor"
    ] = (
        pd.Series(
            factor,
            index=work.index,
        )
        .shift(
            1
        )
    )

    valid_pair = (
        (
            work[
                "_factor"
            ]
            > 0
        )
        &
        (
            work[
                "_prev_factor"
            ]
            > 0
        )
    )

    work[
        "_factor_log_change"
    ] = (
        np.where(
            valid_pair,
            np.log(
                work[
                    "_factor"
                ]
                /
                work[
                    "_prev_factor"
                ]
            ),
            np.nan,
        )
    )

    for column in ACTION_COLUMNS:
        work[
            f"_{column}"
        ] = (
            action_series(
                work,
                column,
            )
        )

    work[
        "_corporate_action_event"
    ] = (
        (
            work[
                "_Dividends"
            ]
            != 0
        )
        |
        (
            work[
                "_Stock Splits"
            ]
            != 0
        )
        |
        (
            work[
                "_Capital Gains"
            ]
            != 0
        )
    )

    work[
        "_factor_changed"
    ] = (
        pd.Series(
            work[
                "_factor_log_change"
            ],
            index=work.index,
        )
        .abs()
        >
        ADJUSTMENT_FACTOR_LOG_CHANGE_EPSILON
    )

    flagged = (
        work.loc[
            work[
                "_corporate_action_event"
            ]
            |
            work[
                "_factor_changed"
            ]
        ]
        .copy()
    )

    reference_set = set(
        reference_dates
    )

    rows = []

    for _, row in flagged.iterrows():
        date = pd.Timestamp(
            row[
                "Date"
            ]
        )

        rows.append(
            {
                "ticker": ticker,
                "Date": date,
                "Close": (
                    row[
                        "Close"
                    ]
                ),
                "Adj Close": (
                    row[
                        "Adj Close"
                    ]
                ),
                "adjustment_factor": (
                    row[
                        "_factor"
                    ]
                ),
                "previous_adjustment_factor": (
                    row[
                        "_prev_factor"
                    ]
                ),
                "adjustment_factor_log_change": (
                    row[
                        "_factor_log_change"
                    ]
                ),
                "Dividends": (
                    row[
                        "_Dividends"
                    ]
                ),
                "Stock Splits": (
                    row[
                        "_Stock Splits"
                    ]
                ),
                "Capital Gains": (
                    row[
                        "_Capital Gains"
                    ]
                ),
                "corporate_action_event": bool(
                    row[
                        "_corporate_action_event"
                    ]
                ),
                "adjustment_factor_changed": bool(
                    row[
                        "_factor_changed"
                    ]
                ),
                "current_is_candidate_nifty_session": bool(
                    date
                    in reference_set
                ),
            }
        )

    return rows


# =============================================================================
# 18. CROSS-SECTIONAL DATE CONSENSUS
# =============================================================================

def build_cross_sectional_date_consensus(
    stock_frames: Dict[
        str,
        pd.DataFrame,
    ],
    benchmark_audit: pd.DataFrame,
    reference_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    panel_parts = []

    for (
        ticker,
        frame,
    ) in stock_frames.items():
        diagnostics = (
            stock_row_diagnostics(
                frame
            )
        )

        part = pd.DataFrame(
            {
                "Date": (
                    diagnostics[
                        "Date"
                    ]
                ),
                "ticker": (
                    ticker
                ),
                "present": (
                    1
                ),
                "volume_zero": (
                    boolean_to_int(
                        diagnostics[
                            "volume_zero"
                        ]
                    )
                ),
                "flat_ohlc": (
                    boolean_to_int(
                        diagnostics[
                            "flat_ohlc"
                        ]
                    )
                ),
                "zero_volume_flat_ohlc": (
                    boolean_to_int(
                        diagnostics[
                            "zero_volume_flat_ohlc"
                        ]
                    )
                ),
                "structurally_invalid": (
                    boolean_to_int(
                        diagnostics[
                            "structurally_invalid_ohlcv"
                        ]
                    )
                ),
            }
        )

        panel_parts.append(
            part
        )

    panel = pd.concat(
        panel_parts,
        ignore_index=True,
    )

    grouped = (
        panel
        .groupby(
            "Date",
            sort=True,
        )
    )

    count_frame = (
        grouped[
            [
                "present",
                "volume_zero",
                "flat_ohlc",
                "zero_volume_flat_ohlc",
                "structurally_invalid",
            ]
        ]
        .sum()
        .rename(
            columns={
                "present": (
                    "stocks_present"
                ),
                "volume_zero": (
                    "stocks_zero_volume"
                ),
                "flat_ohlc": (
                    "stocks_flat_ohlc"
                ),
                "zero_volume_flat_ohlc": (
                    "stocks_zero_volume_flat_ohlc"
                ),
                "structurally_invalid": (
                    "stocks_structurally_invalid"
                ),
            }
        )
        .reset_index()
    )

    # Build ticker-list lookups once. Avoid repeated full-panel filtering for
    # every date; this keeps the audit fast and deterministic.
    present_ticker_lookup = (
        panel
        .groupby(
            "Date",
            sort=True,
        )[
            "ticker"
        ]
        .agg(
            "|".join
        )
        .to_dict()
    )

    zero_ticker_lookup = (
        panel.loc[
            panel[
                "volume_zero"
            ]
            == 1
        ]
        .groupby(
            "Date",
            sort=True,
        )[
            "ticker"
        ]
        .agg(
            "|".join
        )
        .to_dict()
    )

    zero_flat_ticker_lookup = (
        panel.loc[
            panel[
                "zero_volume_flat_ohlc"
            ]
            == 1
        ]
        .groupby(
            "Date",
            sort=True,
        )[
            "ticker"
        ]
        .agg(
            "|".join
        )
        .to_dict()
    )

    invalid_ticker_lookup = (
        panel.loc[
            panel[
                "structurally_invalid"
            ]
            == 1
        ]
        .groupby(
            "Date",
            sort=True,
        )[
            "ticker"
        ]
        .agg(
            "|".join
        )
        .to_dict()
    )

    benchmark_lookup = (
        benchmark_audit
        .set_index(
            "Date"
        )
    )

    reference_set = set(
        reference_dates
    )

    rows = []

    total_stocks = (
        len(
            cfg.STOCK_TICKERS
        )
    )

    all_dates = sorted(
        set(
            count_frame[
                "Date"
            ]
        )
        |
        set(
            benchmark_audit[
                "Date"
            ]
        )
    )

    count_lookup = (
        count_frame
        .set_index(
            "Date"
        )
    )

    for date in all_dates:
        if date in count_lookup.index:
            counts = (
                count_lookup.loc[
                    date
                ]
            )

            stocks_present = int(
                counts[
                    "stocks_present"
                ]
            )

            stocks_zero_volume = int(
                counts[
                    "stocks_zero_volume"
                ]
            )

            stocks_flat = int(
                counts[
                    "stocks_flat_ohlc"
                ]
            )

            stocks_zero_flat = int(
                counts[
                    "stocks_zero_volume_flat_ohlc"
                ]
            )

            stocks_invalid = int(
                counts[
                    "stocks_structurally_invalid"
                ]
            )
        else:
            stocks_present = 0
            stocks_zero_volume = 0
            stocks_flat = 0
            stocks_zero_flat = 0
            stocks_invalid = 0

        nifty_raw_present = (
            date
            in benchmark_lookup.index
        )

        nifty_price_valid = (
            bool(
                benchmark_lookup
                .loc[
                    date,
                    "price_valid",
                ]
            )
            if nifty_raw_present
            else False
        )

        candidate_nifty = (
            date
            in reference_set
        )

        date_wide_candidate = bool(
            DATE_WIDE_ANOMALY_RULE_ENABLED
            and
            (
                stocks_zero_flat
                >= DATE_WIDE_ANOMALY_THRESHOLD_COUNT
            )
        )

        widespread_stock_only = bool(
            (
                not candidate_nifty
            )
            and
            (
                stocks_present
                >= DATE_WIDE_ANOMALY_THRESHOLD_COUNT
            )
        )

        nifty_invalid_widespread = bool(
            nifty_raw_present
            and
            (
                not nifty_price_valid
            )
            and
            (
                stocks_present
                >= DATE_WIDE_ANOMALY_THRESHOLD_COUNT
            )
        )

        rows.append(
            {
                "Date": date,
                "nifty_raw_row_present": bool(
                    nifty_raw_present
                ),
                "nifty_price_valid": bool(
                    nifty_price_valid
                ),
                "candidate_nifty_session": bool(
                    candidate_nifty
                ),
                "stocks_present": (
                    stocks_present
                ),
                "stocks_nonzero_volume": int(
                    stocks_present
                    - stocks_zero_volume
                ),
                "stocks_zero_volume": (
                    stocks_zero_volume
                ),
                "stocks_flat_ohlc": (
                    stocks_flat
                ),
                "stocks_zero_volume_flat_ohlc": (
                    stocks_zero_flat
                ),
                "stocks_structurally_invalid": (
                    stocks_invalid
                ),
                "present_fraction": float(
                    stocks_present
                    / total_stocks
                ),
                "zero_volume_fraction_of_panel": float(
                    stocks_zero_volume
                    / total_stocks
                ),
                "zero_volume_flat_ohlc_fraction_of_panel": float(
                    stocks_zero_flat
                    / total_stocks
                ),
                "structurally_invalid_fraction_of_panel": float(
                    stocks_invalid
                    / total_stocks
                ),
                "date_wide_anomaly_threshold_count": (
                    DATE_WIDE_ANOMALY_THRESHOLD_COUNT
                ),
                "date_wide_anomaly_candidate": (
                    date_wide_candidate
                ),
                "widespread_stock_only_date_alert": (
                    widespread_stock_only
                ),
                "nifty_invalid_but_widespread_stock_presence": (
                    nifty_invalid_widespread
                ),
                "present_tickers": (
                    present_ticker_lookup.get(
                        date,
                        "",
                    )
                ),
                "zero_volume_tickers": (
                    zero_ticker_lookup.get(
                        date,
                        "",
                    )
                ),
                "zero_volume_flat_ohlc_tickers": (
                    zero_flat_ticker_lookup.get(
                        date,
                        "",
                    )
                ),
                "invalid_tickers": (
                    invalid_ticker_lookup.get(
                        date,
                        "",
                    )
                ),
            }
        )

    result = rows_to_dataframe(
        rows,
        DATE_CONSENSUS_FIELDS,
    )

    result[
        "Date"
    ] = pd.to_datetime(
        result[
            "Date"
        ]
    )

    return (
        result
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# 19. SOURCE HASH AUDIT
# =============================================================================

def collect_source_hashes(
    step01_manifest: Dict[str, Any],
    step02_manifest: Dict[str, Any],
) -> pd.DataFrame:
    current_step00 = (
        Path(
            cfg.__file__
        ).resolve()
    )

    frozen_step00 = (
        cfg.get_frozen_script_path()
        .resolve()
    )

    frozen_step01: Optional[Path] = None

    for name, record in (
        step01_manifest
        .get(
            "artifacts",
            {},
        )
        .items()
    ):
        candidate = Path(
            record[
                "path"
            ]
        )

        if (
            "step01_environment_audit_frozen.py"
            in name
            or candidate.name
            == "step01_environment_audit_frozen.py"
        ):
            frozen_step01 = (
                candidate
            )
            break

    frozen_step02_record = (
        step02_manifest
        .get(
            "canonical_artifacts",
            {},
        )
        .get(
            "frozen_step02_script"
        )
    )

    frozen_step02 = (
        Path(
            frozen_step02_record[
                "path"
            ]
        )
        if frozen_step02_record
        else None
    )

    source_paths = [
        (
            "current_step00",
            current_step00,
        ),
        (
            "frozen_step00",
            frozen_step00,
        ),
        (
            "current_step03",
            SCRIPT_FILE.resolve(),
        ),
    ]

    if frozen_step01 is not None:
        source_paths.append(
            (
                "frozen_step01",
                frozen_step01,
            )
        )

    if frozen_step02 is not None:
        source_paths.append(
            (
                "frozen_step02",
                frozen_step02,
            )
        )

    rows = []

    for (
        role,
        path,
    ) in source_paths:
        if not path.exists():
            raise RuntimeError(
                f"Required source file is missing: {path}"
            )

        rows.append(
            {
                "role": role,
                "file_name": (
                    path.name
                ),
                "absolute_path": (
                    str(
                        path
                    )
                ),
                "size_bytes": int(
                    path.stat()
                    .st_size
                ),
                "sha256": (
                    sha256_file(
                        path
                    )
                ),
            }
        )

    current_step00_sha = next(
        row[
            "sha256"
        ]
        for row
        in rows
        if row[
            "role"
        ]
        == "current_step00"
    )

    frozen_step00_sha = next(
        row[
            "sha256"
        ]
        for row
        in rows
        if row[
            "role"
        ]
        == "frozen_step00"
    )

    if (
        current_step00_sha
        != frozen_step00_sha
    ):
        raise RuntimeError(
            "Active step00_config.py differs from its frozen source snapshot."
        )

    return rows_to_dataframe(
        rows,
        SOURCE_HASH_FIELDS,
    )


# =============================================================================
# 20. SUMMARY / MANIFEST HELPERS
# =============================================================================

def artifact_record(
    path: Path,
) -> Dict[str, Any]:
    return {
        "path": (
            str(
                path
            )
        ),
        "relative_path": (
            relative_to_experiment(
                path
            )
        ),
        "size_bytes": int(
            path.stat()
            .st_size
        ),
        "sha256": (
            sha256_file(
                path
            )
        ),
    }


def count_action_events(
    corporate_action_audit: pd.DataFrame,
) -> int:
    if corporate_action_audit.empty:
        return 0

    return int(
        corporate_action_audit[
            "corporate_action_event"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        .sum()
    )


def build_integrity_summary(
    *,
    step00_info: Dict[str, Any],
    step01_info: Dict[str, Any],
    step02_info: Dict[str, Any],
    frames: Dict[str, pd.DataFrame],
    nifty_audit: pd.DataFrame,
    reference_dates: pd.DatetimeIndex,
    coverage: pd.DataFrame,
    missing_dates: pd.DataFrame,
    stock_only_dates: pd.DataFrame,
    invalid_rows: pd.DataFrame,
    zero_volume_rows: pd.DataFrame,
    date_consensus: pd.DataFrame,
    synchronized_alerts: pd.DataFrame,
    large_returns: pd.DataFrame,
    corporate_action_audit: pd.DataFrame,
    source_hashes: pd.DataFrame,
    final_test_lock_sha_after: str,
) -> Dict[str, Any]:
    benchmark = (
        frames[
            cfg.MARKET_INDEX_TICKER
        ]
    )

    nifty_invalid = int(
        (
            ~nifty_audit[
                "price_valid"
            ]
        )
        .sum()
    )

    nifty_volume_nulls = int(
        nifty_audit[
            "volume_null"
        ]
        .sum()
    )

    nifty_volume_zero = int(
        nifty_audit[
            "volume_zero"
        ]
        .sum()
    )

    date_wide_candidates = (
        date_consensus.loc[
            date_consensus[
                "date_wide_anomaly_candidate"
            ]
        ]
    )

    widespread_stock_only = (
        date_consensus.loc[
            date_consensus[
                "widespread_stock_only_date_alert"
            ]
        ]
    )

    nifty_invalid_widespread = (
        date_consensus.loc[
            date_consensus[
                "nifty_invalid_but_widespread_stock_presence"
            ]
        ]
    )

    return {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "step": {
            "step_number": (
                STEP_NUMBER
            ),
            "step_id": (
                STEP_ID
            ),
            "step_name": (
                STEP_NAME
            ),
            "status": "PASS",
        },
        "generated_at_utc": (
            utc_now_iso()
        ),
        "experiment_tag": (
            cfg.EXPERIMENT_TAG
        ),
        "configuration_fingerprint_sha256": (
            step00_info[
                "configuration_fingerprint_sha256"
            ]
        ),
        "upstream_integrity": {
            "step00_manifest_sha256": (
                step00_info[
                    "step00_manifest_sha256"
                ]
            ),
            "step01_manifest_sha256": (
                step01_info[
                    "step01_manifest_sha256"
                ]
            ),
            "step02_manifest_sha256": (
                step02_info[
                    "step02_manifest_sha256"
                ]
            ),
            "verified_step02_raw_files": (
                step02_info[
                    "verified_raw_file_count"
                ]
            ),
        },
        "frozen_data_integrity_policy": {
            "minimum_benchmark_coverage_fraction": (
                PRELIMINARY_MIN_COVERAGE
            ),
            "date_wide_anomaly_rule_enabled": (
                DATE_WIDE_ANOMALY_RULE_ENABLED
            ),
            "date_wide_anomaly_min_stock_fraction": (
                DATE_WIDE_ANOMALY_MIN_STOCK_FRACTION
            ),
            "date_wide_anomaly_threshold_stock_count": (
                DATE_WIDE_ANOMALY_THRESHOLD_COUNT
            ),
            "date_wide_anomaly_definition": (
                cfg.DATE_WIDE_ANOMALY_DEFINITION
            ),
            "corporate_action_policy": (
                cfg.CORPORATE_ACTION_POLICY
            ),
            "market_calendar_policy": (
                cfg.MARKET_CALENDAR_POLICY
            ),
            "large_adjusted_log_return_threshold_diagnostic_only": (
                LARGE_ABS_ADJUSTED_LOG_RETURN_THRESHOLD
            ),
        },
        "benchmark_audit": {
            "raw_rows": int(
                len(
                    benchmark
                )
            ),
            "raw_first_date": (
                benchmark[
                    "Date"
                ]
                .min()
                .strftime(
                    "%Y-%m-%d"
                )
            ),
            "raw_last_date": (
                benchmark[
                    "Date"
                ]
                .max()
                .strftime(
                    "%Y-%m-%d"
                )
            ),
            "candidate_price_valid_sessions": int(
                len(
                    reference_dates
                )
            ),
            "invalid_price_rows": (
                nifty_invalid
            ),
            "volume_null_rows": (
                nifty_volume_nulls
            ),
            "zero_volume_rows": (
                nifty_volume_zero
            ),
        },
        "stock_audit": {
            "stock_count": int(
                len(
                    cfg.STOCK_TICKERS
                )
            ),
            "total_missing_reference_records": int(
                len(
                    missing_dates
                )
            ),
            "total_stock_only_records": int(
                len(
                    stock_only_dates
                )
            ),
            "total_structurally_invalid_rows": int(
                len(
                    invalid_rows
                )
            ),
            "total_zero_volume_rows": int(
                len(
                    zero_volume_rows
                )
            ),
            "total_zero_volume_flat_ohlc_rows": int(
                (
                    zero_volume_rows[
                        "zero_volume_flat_ohlc"
                    ]
                    .fillna(
                        False
                    )
                    .astype(
                        bool
                    )
                    .sum()
                )
                if not zero_volume_rows.empty
                else 0
            ),
            "large_adjusted_return_flags": int(
                len(
                    large_returns
                )
            ),
            "corporate_action_events": (
                count_action_events(
                    corporate_action_audit
                )
            ),
            "stocks_meeting_preliminary_date_coverage": int(
                coverage[
                    "preliminary_date_coverage_eligible"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
                .sum()
            ),
            "stocks_meeting_preliminary_structural_coverage": int(
                coverage[
                    "preliminary_structural_coverage_eligible"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
                .sum()
            ),
        },
        "cross_sectional_audit": {
            "union_dates_audited": int(
                len(
                    date_consensus
                )
            ),
            "date_wide_anomaly_candidates": int(
                len(
                    date_wide_candidates
                )
            ),
            "widespread_stock_only_date_alerts": int(
                len(
                    widespread_stock_only
                )
            ),
            "nifty_invalid_with_widespread_stock_presence": int(
                len(
                    nifty_invalid_widespread
                )
            ),
            "synchronized_alert_rows": int(
                len(
                    synchronized_alerts
                )
            ),
        },
        "source_code_hashes": (
            source_hashes
            .to_dict(
                orient="records"
            )
        ),
        "scientific_boundary": {
            "audit_only": True,
            "cleaning_applied": False,
            "rows_deleted": 0,
            "values_imputed": 0,
            "prices_repaired": False,
            "modelling_calendar_finalized": False,
            "features_constructed": False,
            "targets_constructed": False,
            "scalers_fit": False,
            "regimes_estimated": False,
            "model_selection_performed": False,
            "final_test_scoring_performed": False,
            "raw_quality_audit_includes_full_predeclared_source_window": True,
            "candidate_nifty_calendar_is_not_final": True,
        },
        "final_test_lock_integrity": {
            "state": (
                step00_info[
                    "final_test_state"
                ]
            ),
            "lock_sha256_before": (
                step00_info[
                    "final_test_lock_sha256"
                ]
            ),
            "lock_sha256_after": (
                final_test_lock_sha_after
            ),
            "lock_unchanged": bool(
                final_test_lock_sha_after
                == step00_info[
                    "final_test_lock_sha256"
                ]
            ),
        },
    }


# =============================================================================
# 21. STEP-03 FREEZE / VERIFY HELPERS
# =============================================================================

def step03_artifact_paths() -> tuple[Path, ...]:
    return (
        COLUMN_NULL_SUMMARY_CSV,
        NIFTY_SESSION_AUDIT_CSV,
        NIFTY_INVALID_PRICE_SESSIONS_CSV,
        CANDIDATE_REFERENCE_CALENDAR_CSV,
        STOCK_COVERAGE_SUMMARY_CSV,
        MISSING_DATES_CSV,
        STOCK_ONLY_DATES_CSV,
        INVALID_OHLCV_CSV,
        ZERO_VOLUME_CSV,
        CROSS_SECTIONAL_DATE_CONSENSUS_CSV,
        SYNCHRONIZED_ANOMALIES_CSV,
        LARGE_RETURN_FLAGS_CSV,
        CORPORATE_ACTION_ADJUSTMENT_AUDIT_CSV,
        SOURCE_HASH_CSV,
        INTEGRITY_SUMMARY_JSON,
        FROZEN_STEP03_SCRIPT,
        STEP03_SUMMARY_TXT,
    )


def write_step03_manifest(
    *,
    step00_info: Dict[str, Any],
    step01_info: Dict[str, Any],
    step02_info: Dict[str, Any],
) -> str:
    artifacts = {
        path.name: (
            artifact_record(
                path
            )
        )
        for path
        in step03_artifact_paths()
    }

    manifest = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "step": {
            "step_number": (
                STEP_NUMBER
            ),
            "step_id": (
                STEP_ID
            ),
            "step_name": (
                STEP_NAME
            ),
            "status": "PASS",
        },
        "frozen_at_utc": (
            utc_now_iso()
        ),
        "experiment_tag": (
            cfg.EXPERIMENT_TAG
        ),
        "configuration_fingerprint_sha256": (
            step00_info[
                "configuration_fingerprint_sha256"
            ]
        ),
        "step00_manifest_sha256": (
            step00_info[
                "step00_manifest_sha256"
            ]
        ),
        "step01_manifest_sha256": (
            step01_info[
                "step01_manifest_sha256"
            ]
        ),
        "step02_manifest_sha256": (
            step02_info[
                "step02_manifest_sha256"
            ]
        ),
        "final_test_state": (
            step00_info[
                "final_test_state"
            ]
        ),
        "artifacts": (
            artifacts
        ),
    }

    atomic_write_json(
        STEP03_MANIFEST_JSON,
        manifest,
    )

    manifest_sha = (
        sha256_file(
            STEP03_MANIFEST_JSON
        )
    )

    atomic_write_text(
        STEP03_MANIFEST_SHA,
        (
            f"{manifest_sha}  "
            f"{STEP03_MANIFEST_JSON.name}\n"
        ),
    )

    return manifest_sha


def verify_existing_step03() -> int:
    print(
        "=" * 125
    )
    print(
        "STEP 03 — VERIFY EXISTING MARKET-INTEGRITY AUDIT FREEZE"
    )
    print(
        "=" * 125
    )

    step00_info = (
        verify_step00_contract()
    )

    step01_info = (
        verify_step01_freeze(
            expected_configuration_fingerprint=(
                step00_info[
                    "configuration_fingerprint_sha256"
                ]
            )
        )
    )

    step02_info = (
        verify_step02_freeze(
            expected_configuration_fingerprint=(
                step00_info[
                    "configuration_fingerprint_sha256"
                ]
            ),
            expected_step01_manifest_sha256=(
                step01_info[
                    "step01_manifest_sha256"
                ]
            ),
        )
    )

    if not STEP03_MANIFEST_JSON.exists():
        print(
            "[FAIL] Step-03 manifest is missing."
        )
        return 20

    if not STEP03_MANIFEST_SHA.exists():
        print(
            "[FAIL] Step-03 manifest SHA sidecar is missing."
        )
        return 21

    with STEP03_MANIFEST_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    expected_manifest_sha = (
        STEP03_MANIFEST_SHA
        .read_text(
            encoding="utf-8"
        )
        .strip()
        .split()[0]
    )

    actual_manifest_sha = (
        sha256_file(
            STEP03_MANIFEST_JSON
        )
    )

    if (
        expected_manifest_sha
        != actual_manifest_sha
    ):
        print(
            "[FAIL] Step-03 manifest SHA mismatch."
        )
        return 22

    if (
        manifest
        .get(
            "step",
            {},
        )
        .get(
            "status"
        )
        != "PASS"
    ):
        print(
            "[FAIL] Stored Step-03 status is not PASS."
        )
        return 23

    if (
        manifest.get(
            "configuration_fingerprint_sha256"
        )
        != step00_info[
            "configuration_fingerprint_sha256"
        ]
    ):
        print(
            "[FAIL] Step-03 / Step-00 fingerprint mismatch."
        )
        return 24

    if (
        manifest.get(
            "step01_manifest_sha256"
        )
        != step01_info[
            "step01_manifest_sha256"
        ]
    ):
        print(
            "[FAIL] Step-03 / Step-01 manifest mismatch."
        )
        return 25

    if (
        manifest.get(
            "step02_manifest_sha256"
        )
        != step02_info[
            "step02_manifest_sha256"
        ]
    ):
        print(
            "[FAIL] Step-03 / Step-02 manifest mismatch."
        )
        return 26

    failures = []

    for name, record in (
        manifest
        .get(
            "artifacts",
            {},
        )
        .items()
    ):
        path = Path(
            record[
                "path"
            ]
        )

        if not path.exists():
            failures.append(
                f"missing: {name}"
            )
            continue

        if (
            sha256_file(
                path
            )
            != record[
                "sha256"
            ]
        ):
            failures.append(
                f"hash mismatch: {name}"
            )

    if failures:
        for failure in failures:
            print(
                f"[FAIL] {failure}"
            )

        print(
            "=" * 125
        )
        print(
            "STEP 03 VERIFICATION STATUS: FAIL"
        )
        print(
            "=" * 125
        )
        return 27

    print(
        f"[PASS] Step-00 fingerprint : "
        f"{step00_info['configuration_fingerprint_sha256']}"
    )
    print(
        f"[PASS] Step-01 manifest    : "
        f"{step01_info['step01_manifest_sha256']}"
    )
    print(
        f"[PASS] Step-02 manifest    : "
        f"{step02_info['step02_manifest_sha256']}"
    )
    print(
        f"[PASS] Step-03 manifest    : "
        f"{actual_manifest_sha}"
    )
    print(
        f"[PASS] Final-test state     : "
        f"{step00_info['final_test_state']}"
    )
    print(
        "=" * 125
    )
    print(
        "STEP 03 VERIFICATION STATUS: PASS"
    )
    print(
        "=" * 125
    )

    return 0


# =============================================================================
# 22. FAILURE DIAGNOSTIC
# =============================================================================

def write_failure_diagnostic(
    *,
    stage: str,
    error: Exception,
) -> Path:
    FAILED_ATTEMPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    path = (
        FAILED_ATTEMPT_DIR
        / (
            "03_market_integrity_audit_FAILED_"
            f"{timestamp}.json"
        )
    )

    payload = {
        "step": (
            STEP_NAME
        ),
        "status": "FAIL",
        "failed_at_utc": (
            utc_now_iso()
        ),
        "stage": (
            stage
        ),
        "error_type": (
            type(
                error
            ).__name__
        ),
        "error_message": (
            str(
                error
            )
        ),
        "python_executable": (
            sys.executable
        ),
        "canonical_step03_manifest_exists": bool(
            STEP03_MANIFEST_JSON.exists()
        ),
    }

    atomic_write_json(
        path,
        payload,
    )

    return path


# =============================================================================
# 23. MAIN AUDIT
# =============================================================================

def run_new_audit() -> int:
    print(
        "=" * 125
    )
    print(
        STEP_NAME
    )
    print(
        "=" * 125
    )

    stage = (
        "prerequisite_verification"
    )

    try:
        print(
            "[1/10] Verifying Step 00 and final-test lock..."
        )

        step00_info = (
            verify_step00_contract()
        )

        print(
            "       Configuration fingerprint: "
            f"{step00_info['configuration_fingerprint_sha256']}"
        )
        print(
            "       Final-test state: "
            f"{step00_info['final_test_state']}"
        )

        print(
            "[2/10] Verifying frozen Step-01 environment..."
        )

        step01_info = (
            verify_step01_freeze(
                expected_configuration_fingerprint=(
                    step00_info[
                        "configuration_fingerprint_sha256"
                    ]
                )
            )
        )

        print(
            "       Step-01 manifest: "
            f"{step01_info['step01_manifest_sha256']}"
        )
        print(
            "       Python: "
            f"{sys.executable}"
        )

        print(
            "[3/10] Verifying frozen Step-02 acquisition and raw hashes..."
        )

        step02_info = (
            verify_step02_freeze(
                expected_configuration_fingerprint=(
                    step00_info[
                        "configuration_fingerprint_sha256"
                    ]
                ),
                expected_step01_manifest_sha256=(
                    step01_info[
                        "step01_manifest_sha256"
                    ]
                ),
            )
        )

        print(
            "       Step-02 manifest: "
            f"{step02_info['step02_manifest_sha256']}"
        )
        print(
            "       Raw files verified: "
            f"{step02_info['verified_raw_file_count']}"
        )

        stage = (
            "output_collision_guard"
        )

        print(
            "[4/10] Checking immutable Step-03 output target..."
        )

        assert_new_step03_target()

        cfg.create_directories()

        stage = (
            "load_frozen_raw_snapshots"
        )

        print(
            "[5/10] Loading canonical raw Parquet snapshots..."
        )

        frames: Dict[
            str,
            pd.DataFrame,
        ] = {}

        for ticker in (
            list(
                cfg.STOCK_TICKERS
            )
            + [
                cfg.MARKET_INDEX_TICKER
            ]
        ):
            frame = (
                load_raw_frame(
                    ticker
                )
            )

            frames[
                ticker
            ] = (
                frame
            )

        benchmark = (
            frames[
                cfg.MARKET_INDEX_TICKER
            ]
        )

        stock_frames = {
            ticker: (
                frames[
                    ticker
                ]
            )
            for ticker
            in cfg.STOCK_TICKERS
        }

        print(
            "       Loaded instruments: "
            f"{len(frames)}"
        )
        print(
            "       Raw NIFTY rows: "
            f"{len(benchmark):,}"
        )

        stage = (
            "benchmark_calendar_audit"
        )

        print(
            "[6/10] Auditing NIFTY price validity and candidate reference calendar..."
        )

        (
            nifty_audit,
            reference_dates,
            candidate_calendar,
        ) = (
            audit_nifty_sessions(
                benchmark
            )
        )

        if len(
            reference_dates
        ) == 0:
            raise RuntimeError(
                "No price-valid NIFTY sessions detected."
            )

        nifty_invalid = (
            nifty_audit.loc[
                ~nifty_audit[
                    "price_valid"
                ]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        print(
            "       Candidate price-valid NIFTY sessions: "
            f"{len(reference_dates):,}"
        )
        print(
            "       Invalid NIFTY price rows: "
            f"{len(nifty_invalid):,}"
        )
        print(
            "       NIFTY volume null rows: "
            f"{int(nifty_audit['volume_null'].sum()):,}"
        )
        print(
            "       NIFTY zero-volume rows: "
            f"{int(nifty_audit['volume_zero'].sum()):,}"
        )

        stage = (
            "stock_integrity_audit"
        )

        print(
            "[7/10] Auditing stock coverage, OHLCV, returns and corporate actions..."
        )

        column_null_summary = (
            build_column_null_summary(
                frames
            )
        )

        coverage_rows = []
        missing_rows = []
        stock_only_rows = []
        invalid_rows = []
        zero_volume_rows = []
        large_return_rows = []
        corporate_action_rows = []

        for ticker in (
            cfg.STOCK_TICKERS
        ):
            frame = (
                stock_frames[
                    ticker
                ]
            )

            (
                coverage_summary,
                ticker_missing,
                ticker_stock_only,
            ) = (
                audit_stock_coverage(
                    ticker=ticker,
                    frame=frame,
                    reference_dates=(
                        reference_dates
                    ),
                    benchmark_audit=(
                        nifty_audit
                    ),
                )
            )

            coverage_rows.append(
                coverage_summary
            )

            missing_rows.extend(
                ticker_missing
            )

            stock_only_rows.extend(
                ticker_stock_only
            )

            (
                ticker_invalid,
                ticker_zero_volume,
            ) = (
                audit_stock_rows(
                    ticker=ticker,
                    frame=frame,
                    reference_dates=(
                        reference_dates
                    ),
                )
            )

            invalid_rows.extend(
                ticker_invalid
            )

            zero_volume_rows.extend(
                ticker_zero_volume
            )

            large_return_rows.extend(
                audit_large_adjusted_returns(
                    ticker=ticker,
                    frame=frame,
                    reference_dates=(
                        reference_dates
                    ),
                )
            )

            corporate_action_rows.extend(
                audit_corporate_actions_and_adjustment_factors(
                    ticker=ticker,
                    frame=frame,
                    reference_dates=(
                        reference_dates
                    ),
                )
            )

        coverage = rows_to_dataframe(
            coverage_rows,
            COVERAGE_FIELDS,
        )

        missing_dates = rows_to_dataframe(
            missing_rows,
            MISSING_DATE_FIELDS,
        )

        stock_only_dates = rows_to_dataframe(
            stock_only_rows,
            STOCK_ONLY_FIELDS,
        )

        invalid_ohlcv = rows_to_dataframe(
            invalid_rows,
            INVALID_OHLCV_FIELDS,
        )

        zero_volume = rows_to_dataframe(
            zero_volume_rows,
            ZERO_VOLUME_FIELDS,
        )

        large_returns = rows_to_dataframe(
            large_return_rows,
            LARGE_RETURN_FIELDS,
        )

        corporate_action_audit = rows_to_dataframe(
            corporate_action_rows,
            CORPORATE_ACTION_FIELDS,
        )

        print(
            "       Stocks meeting >= "
            f"{PRELIMINARY_MIN_COVERAGE:.0%} date coverage: "
            f"{int(coverage['preliminary_date_coverage_eligible'].sum())}"
            f"/{len(coverage)}"
        )
        print(
            "       Missing stock/reference records: "
            f"{len(missing_dates):,}"
        )
        print(
            "       Stock-only date records: "
            f"{len(stock_only_dates):,}"
        )
        print(
            "       Structurally invalid stock rows: "
            f"{len(invalid_ohlcv):,}"
        )
        print(
            "       Zero-volume stock rows: "
            f"{len(zero_volume):,}"
        )
        print(
            "       Large adjusted-return flags: "
            f"{len(large_returns):,}"
        )
        print(
            "       Corporate-action events: "
            f"{count_action_events(corporate_action_audit):,}"
        )

        stage = (
            "cross_sectional_consensus"
        )

        print(
            "[8/10] Auditing cross-sectional date consensus and frozen anomaly rule..."
        )

        date_consensus = (
            build_cross_sectional_date_consensus(
                stock_frames=(
                    stock_frames
                ),
                benchmark_audit=(
                    nifty_audit
                ),
                reference_dates=(
                    reference_dates
                ),
            )
        )

        synchronized_alerts = (
            date_consensus.loc[
                date_consensus[
                    "date_wide_anomaly_candidate"
                ]
                |
                date_consensus[
                    "widespread_stock_only_date_alert"
                ]
                |
                date_consensus[
                    "nifty_invalid_but_widespread_stock_presence"
                ]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        date_wide_count = int(
            date_consensus[
                "date_wide_anomaly_candidate"
            ]
            .sum()
        )

        widespread_stock_only_count = int(
            date_consensus[
                "widespread_stock_only_date_alert"
            ]
            .sum()
        )

        print(
            "       Frozen date-wide anomaly rule: "
            f">= {DATE_WIDE_ANOMALY_MIN_STOCK_FRACTION:.0%} "
            "of stocks with zero volume AND flat OHLC"
        )
        print(
            "       Threshold stock count: "
            f"{DATE_WIDE_ANOMALY_THRESHOLD_COUNT}/{len(cfg.STOCK_TICKERS)}"
        )
        print(
            "       Date-wide anomaly candidates: "
            f"{date_wide_count}"
        )
        print(
            "       Widespread stock-only alerts: "
            f"{widespread_stock_only_count}"
        )

        stage = (
            "freeze_outputs"
        )

        print(
            "[9/10] Writing immutable audit outputs and integrity manifest..."
        )

        with STEP01_MANIFEST.open(
            "r",
            encoding="utf-8",
        ) as handle:
            step01_manifest_payload = (
                json.load(
                    handle
                )
            )

        with STEP02_ACQUISITION_MANIFEST.open(
            "r",
            encoding="utf-8",
        ) as handle:
            step02_manifest_payload = (
                json.load(
                    handle
                )
            )

        source_hashes = (
            collect_source_hashes(
                step01_manifest=(
                    step01_manifest_payload
                ),
                step02_manifest=(
                    step02_manifest_payload
                ),
            )
        )

        write_dataframe_csv(
            COLUMN_NULL_SUMMARY_CSV,
            column_null_summary,
            COLUMN_NULL_FIELDS,
        )

        write_dataframe_csv(
            NIFTY_SESSION_AUDIT_CSV,
            nifty_audit,
            NIFTY_AUDIT_FIELDS,
        )

        write_dataframe_csv(
            NIFTY_INVALID_PRICE_SESSIONS_CSV,
            nifty_invalid,
            NIFTY_INVALID_FIELDS,
        )

        write_dataframe_csv(
            CANDIDATE_REFERENCE_CALENDAR_CSV,
            candidate_calendar,
            CANDIDATE_CALENDAR_FIELDS,
        )

        write_dataframe_csv(
            STOCK_COVERAGE_SUMMARY_CSV,
            coverage,
            COVERAGE_FIELDS,
        )

        write_dataframe_csv(
            MISSING_DATES_CSV,
            missing_dates,
            MISSING_DATE_FIELDS,
        )

        write_dataframe_csv(
            STOCK_ONLY_DATES_CSV,
            stock_only_dates,
            STOCK_ONLY_FIELDS,
        )

        write_dataframe_csv(
            INVALID_OHLCV_CSV,
            invalid_ohlcv,
            INVALID_OHLCV_FIELDS,
        )

        write_dataframe_csv(
            ZERO_VOLUME_CSV,
            zero_volume,
            ZERO_VOLUME_FIELDS,
        )

        write_dataframe_csv(
            CROSS_SECTIONAL_DATE_CONSENSUS_CSV,
            date_consensus,
            DATE_CONSENSUS_FIELDS,
        )

        write_dataframe_csv(
            SYNCHRONIZED_ANOMALIES_CSV,
            synchronized_alerts,
            SYNCHRONIZED_FIELDS,
        )

        write_dataframe_csv(
            LARGE_RETURN_FLAGS_CSV,
            large_returns,
            LARGE_RETURN_FIELDS,
        )

        write_dataframe_csv(
            CORPORATE_ACTION_ADJUSTMENT_AUDIT_CSV,
            corporate_action_audit,
            CORPORATE_ACTION_FIELDS,
        )

        write_dataframe_csv(
            SOURCE_HASH_CSV,
            source_hashes,
            SOURCE_HASH_FIELDS,
        )

        lock_sha_after = (
            sha256_file(
                cfg.get_final_test_lock_path()
            )
        )

        if (
            lock_sha_after
            != step00_info[
                "final_test_lock_sha256"
            ]
        ):
            raise RuntimeError(
                "Final-test lock changed during Step 03."
            )

        integrity_summary = (
            build_integrity_summary(
                step00_info=(
                    step00_info
                ),
                step01_info=(
                    step01_info
                ),
                step02_info=(
                    step02_info
                ),
                frames=(
                    frames
                ),
                nifty_audit=(
                    nifty_audit
                ),
                reference_dates=(
                    reference_dates
                ),
                coverage=(
                    coverage
                ),
                missing_dates=(
                    missing_dates
                ),
                stock_only_dates=(
                    stock_only_dates
                ),
                invalid_rows=(
                    invalid_ohlcv
                ),
                zero_volume_rows=(
                    zero_volume
                ),
                date_consensus=(
                    date_consensus
                ),
                synchronized_alerts=(
                    synchronized_alerts
                ),
                large_returns=(
                    large_returns
                ),
                corporate_action_audit=(
                    corporate_action_audit
                ),
                source_hashes=(
                    source_hashes
                ),
                final_test_lock_sha_after=(
                    lock_sha_after
                ),
            )
        )

        atomic_write_json(
            INTEGRITY_SUMMARY_JSON,
            integrity_summary,
        )

        shutil.copy2(
            SCRIPT_FILE,
            FROZEN_STEP03_SCRIPT,
        )

        summary_lines = [
            STEP_NAME,
            "=" * 92,
            "Status: PASS",
            (
                "Configuration fingerprint: "
                + step00_info[
                    "configuration_fingerprint_sha256"
                ]
            ),
            (
                "Step-01 manifest SHA-256: "
                + step01_info[
                    "step01_manifest_sha256"
                ]
            ),
            (
                "Step-02 manifest SHA-256: "
                + step02_info[
                    "step02_manifest_sha256"
                ]
            ),
            (
                "Raw NIFTY rows: "
                f"{len(benchmark)}"
            ),
            (
                "Candidate price-valid NIFTY sessions: "
                f"{len(reference_dates)}"
            ),
            (
                "Invalid NIFTY price rows: "
                f"{len(nifty_invalid)}"
            ),
            (
                "NIFTY volume null rows: "
                f"{int(nifty_audit['volume_null'].sum())}"
            ),
            (
                "NIFTY zero-volume rows: "
                f"{int(nifty_audit['volume_zero'].sum())}"
            ),
            (
                "Total missing stock/reference records: "
                f"{len(missing_dates)}"
            ),
            (
                "Total stock-only records: "
                f"{len(stock_only_dates)}"
            ),
            (
                "Structurally invalid stock rows: "
                f"{len(invalid_ohlcv)}"
            ),
            (
                "Zero-volume stock rows: "
                f"{len(zero_volume)}"
            ),
            (
                "Large adjusted-return flags: "
                f"{len(large_returns)}"
            ),
            (
                "Corporate-action events: "
                f"{count_action_events(corporate_action_audit)}"
            ),
            (
                "Date-wide anomaly candidates: "
                f"{date_wide_count}"
            ),
            (
                "Widespread stock-only alerts: "
                f"{widespread_stock_only_count}"
            ),
            (
                "Final-test state: "
                f"{step00_info['final_test_state']}"
            ),
        ]

        atomic_write_text(
            STEP03_SUMMARY_TXT,
            "\n".join(
                summary_lines
            )
            + "\n",
        )

        manifest_sha = (
            write_step03_manifest(
                step00_info=(
                    step00_info
                ),
                step01_info=(
                    step01_info
                ),
                step02_info=(
                    step02_info
                ),
            )
        )

        stage = (
            "console_report"
        )

        print(
            "[10/10] Final integrity report..."
        )
        print()
        print(
            "-" * 125
        )
        print(
            "NIFTY-50 PRICE-CALENDAR AUDIT"
        )
        print(
            "-" * 125
        )
        print(
            "Raw NIFTY rows                     : "
            f"{len(benchmark):,}"
        )
        print(
            "Raw NIFTY date range               : "
            f"{benchmark['Date'].min().date()} -> "
            f"{benchmark['Date'].max().date()}"
        )
        print(
            "Candidate price-valid sessions     : "
            f"{len(reference_dates):,}"
        )
        print(
            "Invalid NIFTY price rows           : "
            f"{len(nifty_invalid):,}"
        )
        print(
            "NIFTY Volume null rows             : "
            f"{int(nifty_audit['volume_null'].sum()):,}"
        )
        print(
            "NIFTY Volume zero rows             : "
            f"{int(nifty_audit['volume_zero'].sum()):,}"
        )
        print()

        print(
            "NIFTY per-column null counts:"
        )

        for column in (
            list(
                PRICE_COLUMNS
            )
            + [
                "Volume"
            ]
        ):
            print(
                f"  {column:<12}: "
                f"{int(benchmark[column].isna().sum()):,}"
            )

        print(
            "-" * 125
        )
        print(
            "STOCK COVERAGE AGAINST CANDIDATE NIFTY PRICE CALENDAR"
        )
        print(
            "-" * 125
        )

        header = (
            f"{'Ticker':<17}"
            f"{'Raw':>7}"
            f"{'Nifty':>8}"
            f"{'Present':>10}"
            f"{'Missing':>9}"
            f"{'Extra':>8}"
            f"{'DateCov':>10}"
            f"{'ValidCov':>10}"
            f"{'Invalid':>9}"
            f"{'ZeroVol':>9}"
            f"{'ZeroFlat':>10}"
        )

        print(
            header
        )
        print(
            "-" * 125
        )

        for row in (
            coverage
            .itertuples(
                index=False
            )
        ):
            print(
                f"{row.ticker:<17}"
                f"{row.raw_rows:>7}"
                f"{row.reference_session_count:>8}"
                f"{row.reference_sessions_present:>10}"
                f"{row.reference_sessions_missing:>9}"
                f"{row.stock_only_dates:>8}"
                f"{row.date_coverage_percent:>9.2f}%"
                f"{row.structural_valid_coverage_percent:>9.2f}%"
                f"{row.structurally_invalid_reference_rows:>9}"
                f"{row.zero_volume_reference_rows:>9}"
                f"{row.zero_volume_flat_ohlc_reference_rows:>10}"
            )

        print(
            "-" * 125
        )
        print(
            "Total missing stock/reference records : "
            f"{len(missing_dates):,}"
        )
        print(
            "Total stock-only date records         : "
            f"{len(stock_only_dates):,}"
        )
        print(
            "Total structurally invalid stock rows : "
            f"{len(invalid_ohlcv):,}"
        )
        print(
            "Total zero-volume stock rows          : "
            f"{len(zero_volume):,}"
        )
        print(
            "Large adjusted-return flags           : "
            f"{len(large_returns):,}"
        )
        print(
            "Corporate-action events               : "
            f"{count_action_events(corporate_action_audit):,}"
        )
        print(
            "Stocks >= "
            f"{PRELIMINARY_MIN_COVERAGE:.0%} date coverage          : "
            f"{int(coverage['preliminary_date_coverage_eligible'].sum())}"
            f"/{len(coverage)}"
        )
        print(
            "-" * 125
        )
        print(
            "FROZEN DATE-WIDE ANOMALY RULE"
        )
        print(
            "-" * 125
        )
        print(
            "Definition                           : "
            f"{cfg.DATE_WIDE_ANOMALY_DEFINITION}"
        )
        print(
            "Minimum stock fraction               : "
            f"{DATE_WIDE_ANOMALY_MIN_STOCK_FRACTION:.0%}"
        )
        print(
            "Threshold count                      : "
            f"{DATE_WIDE_ANOMALY_THRESHOLD_COUNT}"
            f"/{len(cfg.STOCK_TICKERS)}"
        )
        print(
            "Detected date-wide anomaly candidates: "
            f"{date_wide_count}"
        )
        print(
            "Widespread stock-only date alerts    : "
            f"{widespread_stock_only_count}"
        )

        if not synchronized_alerts.empty:
            print()
            print(
                "Alert dates:"
            )

            for row in (
                synchronized_alerts
                .itertuples(
                    index=False
                )
            ):
                print(
                    f"  {pd.Timestamp(row.Date).date()} | "
                    f"NIFTY-valid={row.candidate_nifty_session} | "
                    f"present={row.stocks_present:>2} | "
                    f"zero-volume={row.stocks_zero_volume:>2} | "
                    f"zero+flat={row.stocks_zero_volume_flat_ohlc:>2} | "
                    f"invalid={row.stocks_structurally_invalid:>2} | "
                    f"rule-candidate={row.date_wide_anomaly_candidate}"
                )

        print(
            "-" * 125
        )
        print(
            "OUTPUT FILES"
        )
        print(
            "-" * 125
        )
        print(
            f"Column null summary          : {COLUMN_NULL_SUMMARY_CSV}"
        )
        print(
            f"NIFTY session audit          : {NIFTY_SESSION_AUDIT_CSV}"
        )
        print(
            f"NIFTY invalid sessions       : {NIFTY_INVALID_PRICE_SESSIONS_CSV}"
        )
        print(
            f"Candidate NIFTY calendar     : {CANDIDATE_REFERENCE_CALENDAR_CSV}"
        )
        print(
            f"Stock coverage summary       : {STOCK_COVERAGE_SUMMARY_CSV}"
        )
        print(
            f"Missing-date register        : {MISSING_DATES_CSV}"
        )
        print(
            f"Stock-only date register     : {STOCK_ONLY_DATES_CSV}"
        )
        print(
            f"Invalid OHLCV register       : {INVALID_OHLCV_CSV}"
        )
        print(
            f"Zero-volume register         : {ZERO_VOLUME_CSV}"
        )
        print(
            f"Cross-sectional consensus    : {CROSS_SECTIONAL_DATE_CONSENSUS_CSV}"
        )
        print(
            f"Synchronized alerts          : {SYNCHRONIZED_ANOMALIES_CSV}"
        )
        print(
            f"Large-return flags           : {LARGE_RETURN_FLAGS_CSV}"
        )
        print(
            f"Corporate-action audit       : {CORPORATE_ACTION_ADJUSTMENT_AUDIT_CSV}"
        )
        print(
            f"Integrity summary            : {INTEGRITY_SUMMARY_JSON}"
        )
        print(
            f"Step-03 manifest             : {STEP03_MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256             : {manifest_sha}"
        )
        print(
            f"Final-test state             : {step00_info['final_test_state']}"
        )
        print(
            "=" * 125
        )
        print(
            "PASS: Step-00, Step-01 and Step-02 chain integrity verified."
        )
        print(
            "PASS: All frozen Step-02 raw-file hashes verified."
        )
        print(
            "PASS: Market coverage, NIFTY calendar, OHLCV, synchronized-date, "
            "return and corporate-action audits completed."
        )
        print(
            "PASS: No cleaning, deletion, repair, imputation or modelling was performed."
        )
        print(
            "PASS: Candidate NIFTY calendar is diagnostic only and is not yet final."
        )
        print(
            "PASS: Final held-out test remains LOCKED_NOT_OPENED."
        )
        print(
            "=" * 125
        )
        print(
            "Do not run Step 04 yet. Paste the complete terminal output for audit."
        )

        return 0

    except Exception as exc:
        failure_path = (
            write_failure_diagnostic(
                stage=(
                    stage
                ),
                error=(
                    exc
                ),
            )
        )

        print()
        print(
            "=" * 125
        )
        print(
            "STEP 03 FAILED — AUDIT NOT FROZEN"
        )
        print(
            "=" * 125
        )
        print(
            f"Stage      : {stage}"
        )
        print(
            f"Error type : {type(exc).__name__}"
        )
        print(
            f"Error      : {exc}"
        )
        print(
            f"Diagnostic : {failure_path}"
        )
        print(
            "=" * 125
        )
        print(
            "Do not proceed to Step 04."
        )

        return 30


# =============================================================================
# 24. CLI / ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen Step-02 market data without applying cleaning."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an already frozen Step-03 audit without recomputing it."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verify_only:
        return verify_existing_step03()

    return run_new_audit()


if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "\nStep 03 interrupted by user."
        )
        sys.exit(
            130
        )

    except Exception as exc:
        print()
        print(
            "=" * 125
        )
        print(
            "STEP 03 UNHANDLED FAILURE"
        )
        print(
            "=" * 125
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        print(
            "Full traceback:"
        )
        traceback.print_exc()
        print(
            "=" * 125
        )
        sys.exit(
            1
        )
