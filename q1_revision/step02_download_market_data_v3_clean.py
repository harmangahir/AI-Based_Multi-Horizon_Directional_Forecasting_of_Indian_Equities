#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 02 — FROZEN HISTORICAL MARKET-DATA ACQUISITION (CLEAN V3)

Project
-------
Artificial Intelligence-Based Multi-Horizon Directional Forecasting of
Indian Equities: A Leakage-Controlled and Dependence-Aware Evaluation

Purpose
-------
Download and cryptographically freeze the raw Yahoo Finance provider snapshot
for the clean Q1 revision experiment.

Scientific source window
------------------------
2016-01-01 through 2026-06-30 inclusive.

Universe
--------
20 prespecified NSE equities plus ^NSEI (NIFTY 50 reference index).

Scientific boundary
-------------------
This stage is acquisition-only. It DOES NOT:
- remove zero-volume rows,
- remove synchronized/anomalous sessions,
- repair OHLCV values,
- impute missing data,
- construct adjusted OHLC fields,
- adjust volume,
- construct technical indicators,
- construct targets,
- fit scalers,
- estimate regimes,
- define modelling samples,
- train/select/evaluate models,
- score the final held-out test.

The raw acquisition necessarily covers the calendar period later designated as
the final held-out test. That is not model evaluation. The Step-00 final-test
selection/scoring lock must remain unchanged throughout this stage.

Prerequisites
-------------
1. Step 00 is frozen and verifiable.
2. Step 01 has completed with PASS and its environment artifacts are intact.
3. This script is run from the same Python environment frozen by Step 01.

Expected location
-----------------
<repository>/q1_revision/step02_download_market_data.py

Run
---
python step02_download_market_data.py

Verify an already frozen Step 02
--------------------------------
python step02_download_market_data.py --verify-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# 0. SCRIPT / CONFIG LOCATION
# =============================================================================

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent

if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError(
        "step02_download_market_data.py must be located inside "
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

STEP_NUMBER = 2
STEP_ID = "02"
STEP_NAME = "STEP 02 — FROZEN HISTORICAL MARKET-DATA ACQUISITION"
SCHEMA_VERSION = "1.0"

DATA_SOURCE = "Yahoo Finance"
DATA_LIBRARY = "yfinance"

# yfinance end is exclusive.
YAHOO_API_END_DATE = cfg.RAW_DATA_END_DATE + pd.Timedelta(days=1)

MAX_DOWNLOAD_ATTEMPTS = 4
REQUEST_GAP_SECONDS = 1.25
RETRY_DELAYS_SECONDS = (5, 10, 20)

EXPECTED_CORE_COLUMNS = (
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
)

KNOWN_PROVIDER_FIELDS = {
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "Capital Gains",
}


# =============================================================================
# 2. OUTPUT LOCATIONS
# =============================================================================

# cfg.RAW_DATA_DIR is already experiment-specific in clean V3.
# Do NOT append cfg.EXPERIMENT_TAG again.
RAW_SNAPSHOT_ROOT = cfg.RAW_DATA_DIR / "provider_snapshot"
RAW_CSV_DIR = RAW_SNAPSHOT_ROOT / "csv"
RAW_PARQUET_DIR = RAW_SNAPSHOT_ROOT / "parquet"

DOWNLOAD_LOG_CSV = cfg.AUDIT_DIR / "02_download_log.csv"
RAW_FILE_HASH_CSV = cfg.AUDIT_DIR / "02_raw_file_hashes.csv"
SOURCE_HASH_CSV = cfg.AUDIT_DIR / "02_source_file_hashes.csv"

REFERENCE_CALENDAR_CSV = (
    cfg.REFERENCE_DATA_DIR / "02_nifty50_reference_trading_calendar.csv"
)

ACQUISITION_MANIFEST_JSON = (
    cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.json"
)
ACQUISITION_MANIFEST_SHA = (
    cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.sha256"
)
FROZEN_STEP02_SCRIPT = (
    cfg.MANIFEST_DIR / "step02_download_market_data_frozen.py"
)

STEP02_SUMMARY_TXT = (
    cfg.REPORT_DIR / "02_market_data_acquisition_summary.txt"
)

FAILED_ATTEMPT_DIR = (
    cfg.AUDIT_DIR / "step02_failed_attempts"
)

STEP01_MANIFEST = (
    cfg.MANIFEST_DIR / "01_environment_audit_manifest.json"
)
STEP01_MANIFEST_SHA = (
    cfg.MANIFEST_DIR / "01_environment_audit_manifest.sha256"
)
STEP01_AUDIT_JSON = (
    cfg.ENVIRONMENT_DIR / "01_environment_audit.json"
)


# =============================================================================
# 3. INSTRUMENT REGISTRY
# =============================================================================

INSTRUMENTS = tuple(
    [
        {
            "ticker": ticker,
            "instrument_type": "stock",
        }
        for ticker in cfg.STOCK_TICKERS
    ]
    + [
        {
            "ticker": cfg.MARKET_INDEX_TICKER,
            "instrument_type": "benchmark_index",
        }
    ]
)


# =============================================================================
# 4. GENERIC HELPERS
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def timestamp_string(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(text)

    os.replace(temporary, path)


def atomic_write_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

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
        handle.write("\n")

    os.replace(temporary, path)


def atomic_write_csv(
    path: Path,
    rows: Iterable[Dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    os.replace(temporary, path)


def get_distribution_version(
    distribution_name: str,
) -> Optional[str]:
    try:
        return importlib_metadata.version(distribution_name)
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
        path.resolve().relative_to(
            cfg.EXPERIMENT_ROOT_DIR.resolve()
        )
    )


# =============================================================================
# 5. STEP-00 / FINAL-TEST LOCK VERIFICATION
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
        step00_manifest_path,
        step00_manifest_sha256,
        freeze_status,
    ) = cfg.verify_existing_freeze(
        configuration_payload=configuration_payload,
        configuration_fingerprint=configuration_fingerprint,
    )

    if freeze_status != "VERIFIED_EXISTING":
        raise RuntimeError(
            "Step-00 freeze was not VERIFIED_EXISTING."
        )

    lock_path = cfg.get_final_test_lock_path()

    with lock_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        lock_payload = json.load(handle)

    if lock_payload.get("state") != "LOCKED_NOT_OPENED":
        raise RuntimeError(
            "Final-test state is not LOCKED_NOT_OPENED."
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
        "configuration_fingerprint_sha256": (
            configuration_fingerprint
        ),
        "step00_manifest_path": (
            str(step00_manifest_path)
        ),
        "step00_manifest_sha256": (
            step00_manifest_sha256
        ),
        "step00_freeze_status": (
            freeze_status
        ),
        "final_test_lock_path": (
            str(lock_path)
        ),
        "final_test_lock_sha256": (
            sha256_file(lock_path)
        ),
        "final_test_state": (
            lock_payload.get("state")
        ),
        "final_test_opened": (
            lock_payload.get("final_test_opened")
        ),
        "final_test_scored": (
            lock_payload.get("final_test_scored")
        ),
    }


# =============================================================================
# 6. STEP-01 ENVIRONMENT FREEZE VERIFICATION
# =============================================================================

def verify_step01_freeze(
    expected_configuration_fingerprint: str,
) -> Dict[str, Any]:
    if not STEP01_MANIFEST.exists():
        raise RuntimeError(
            "Step 01 has not been frozen successfully. Missing:\n"
            f"{STEP01_MANIFEST}\n"
            "Run and PASS Step 01 before Step 02."
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
        manifest = json.load(handle)

    if manifest.get("status") != "PASS":
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
            "Step-01 and Step-00 configuration fingerprints differ."
        )

    if (
        manifest.get("final_test_state")
        != "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError(
            "Step-01 did not record a locked final test."
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

    if expected_manifest_sha != actual_manifest_sha:
        raise RuntimeError(
            "Step-01 manifest SHA-256 mismatch."
        )

    # Verify every artifact Step 01 froze.
    artifact_failures = []

    for name, record in manifest.get(
        "artifacts",
        {},
    ).items():

        path = Path(
            record["path"]
        )

        if not path.exists():
            artifact_failures.append(
                f"missing artifact: {name}"
            )
            continue

        actual_sha = (
            sha256_file(path)
        )

        if (
            actual_sha
            != record["sha256"]
        ):
            artifact_failures.append(
                f"hash mismatch: {name}"
            )

    if artifact_failures:
        raise RuntimeError(
            "Step-01 artifact verification failed:\n"
            + "\n".join(
                artifact_failures
            )
        )

    with STEP01_AUDIT_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        audit_payload = json.load(handle)

    stored_python_executable = (
        audit_payload
        .get("system", {})
        .get("python_executable")
    )

    if (
        stored_python_executable
        and Path(stored_python_executable).resolve()
        != Path(sys.executable).resolve()
    ):
        raise RuntimeError(
            "Current Python executable differs from Step 01.\n"
            f"Step 01: {stored_python_executable}\n"
            f"Current : {sys.executable}"
        )

    package_drift = []

    for row in audit_payload.get(
        "packages",
        [],
    ):
        if not row.get("required", False):
            continue

        distribution_name = (
            row.get("distribution")
        )

        stored_version = (
            row.get("version", "")
        )

        current_version = (
            get_distribution_version(
                distribution_name
            )
        )

        if current_version != stored_version:
            package_drift.append(
                (
                    distribution_name,
                    stored_version,
                    current_version,
                )
            )

    if package_drift:
        messages = [
            (
                f"{name}: "
                f"Step01={stored!r}, "
                f"current={current!r}"
            )
            for name, stored, current
            in package_drift
        ]

        raise RuntimeError(
            "Required-package drift detected after Step 01:\n"
            + "\n".join(messages)
        )

    return {
        "step01_manifest_path": (
            str(STEP01_MANIFEST)
        ),
        "step01_manifest_sha256": (
            actual_manifest_sha
        ),
        "environment_audit_path": (
            str(STEP01_AUDIT_JSON)
        ),
        "python_executable": (
            stored_python_executable
        ),
        "conda_environment": (
            audit_payload
            .get("system", {})
            .get("conda_default_env")
        ),
        "pytorch_version": (
            audit_payload
            .get("pytorch_cuda", {})
            .get("torch_version")
        ),
        "pytorch_cuda_build": (
            audit_payload
            .get("pytorch_cuda", {})
            .get("torch_compiled_cuda_version")
        ),
    }


# =============================================================================
# 7. OUTPUT COLLISION / IMMUTABILITY GUARD
# =============================================================================

def canonical_step02_outputs() -> tuple[Path, ...]:
    return (
        RAW_SNAPSHOT_ROOT,
        DOWNLOAD_LOG_CSV,
        RAW_FILE_HASH_CSV,
        SOURCE_HASH_CSV,
        REFERENCE_CALENDAR_CSV,
        ACQUISITION_MANIFEST_JSON,
        ACQUISITION_MANIFEST_SHA,
        FROZEN_STEP02_SCRIPT,
        STEP02_SUMMARY_TXT,
    )


def assert_new_step02_target() -> None:
    collisions = [
        path
        for path in canonical_step02_outputs()
        if path.exists()
    ]

    if collisions:
        collision_text = "\n".join(
            str(path)
            for path in collisions
        )

        raise RuntimeError(
            "Step-02 canonical output(s) already exist. "
            "Refusing to overwrite an acquisition snapshot.\n"
            f"{collision_text}\n\n"
            "If Step 02 completed successfully, use --verify-only. "
            "If this is an interrupted attempt, inspect the artifacts "
            "before any cleanup."
        )


# =============================================================================
# 8. YFINANCE COLUMN / DATE NORMALIZATION
# =============================================================================

def flatten_yfinance_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert yfinance single-instrument MultiIndex columns to a single field
    level without changing provider values.
    """

    frame = dataframe.copy()

    if isinstance(
        frame.columns,
        pd.MultiIndex,
    ):
        columns = frame.columns

        hit_counts = []

        for level_number in range(
            columns.nlevels
        ):
            values = [
                str(value)
                for value
                in columns.get_level_values(
                    level_number
                )
            ]

            hit_count = sum(
                value in KNOWN_PROVIDER_FIELDS
                for value in values
            )

            hit_counts.append(
                (
                    hit_count,
                    level_number,
                )
            )

        best_hit_count, best_level = max(
            hit_counts
        )

        if best_hit_count == 0:
            raise RuntimeError(
                "Could not identify the provider-field "
                "level in yfinance MultiIndex columns."
            )

        frame.columns = [
            str(value)
            for value
            in columns.get_level_values(
                best_level
            )
        ]

    else:
        frame.columns = [
            str(column)
            for column in frame.columns
        ]

    if frame.columns.duplicated().any():
        duplicates = (
            pd.Index(
                frame.columns[
                    frame.columns.duplicated(
                        keep=False
                    )
                ]
            )
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Duplicate columns remain after yfinance "
            f"column normalization: {duplicates}"
        )

    return frame


def normalize_date_index(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert provider daily timestamps to timezone-naive local calendar dates.
    Only observations outside the predeclared scientific source window are
    excluded.
    """

    frame = dataframe.copy()

    frame.index = pd.to_datetime(
        frame.index,
        errors="raise",
    )

    if frame.index.tz is not None:
        frame.index = (
            frame.index
            .tz_localize(None)
        )

    frame.index = (
        frame.index
        .normalize()
    )

    frame.index.name = "Date"

    mask = (
        (
            frame.index
            >= cfg.RAW_DATA_START_DATE
        )
        &
        (
            frame.index
            <= cfg.RAW_DATA_END_DATE
        )
    )

    return frame.loc[
        mask
    ].copy()


def order_provider_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    preferred = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "Capital Gains",
    ]

    ordered = [
        column
        for column in preferred
        if column in dataframe.columns
    ]

    remaining = [
        column
        for column in dataframe.columns
        if column not in ordered
    ]

    return dataframe[
        ordered + remaining
    ].copy()


# =============================================================================
# 9. DOWNLOAD ONE INSTRUMENT
# =============================================================================

def download_one_instrument(
    ticker: str,
) -> Tuple[pd.DataFrame, int, str]:
    final_exception: Optional[Exception] = None

    for attempt in range(
        1,
        MAX_DOWNLOAD_ATTEMPTS + 1,
    ):
        try:
            print(
                f"  Attempt "
                f"{attempt}/{MAX_DOWNLOAD_ATTEMPTS}: "
                f"{ticker}"
            )

            frame = yf.download(
                tickers=ticker,
                start=timestamp_string(
                    cfg.RAW_DATA_START_DATE
                ),
                end=timestamp_string(
                    YAHOO_API_END_DATE
                ),
                interval="1d",
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
                repair=False,
                keepna=True,
                timeout=30,
            )

            if frame is None:
                raise RuntimeError(
                    "yfinance returned None."
                )

            if len(frame) == 0:
                raise RuntimeError(
                    "yfinance returned zero rows."
                )

            frame = flatten_yfinance_columns(
                dataframe=frame
            )

            frame = normalize_date_index(
                dataframe=frame
            )

            frame = order_provider_columns(
                dataframe=frame
            )

            missing_core = [
                column
                for column
                in EXPECTED_CORE_COLUMNS
                if column
                not in frame.columns
            ]

            if missing_core:
                raise RuntimeError(
                    "Missing required provider columns: "
                    + ", ".join(
                        missing_core
                    )
                )

            if len(frame) == 0:
                raise RuntimeError(
                    "No rows remain after enforcing "
                    "the requested source window."
                )

            if (
                frame.index
                .duplicated(
                    keep=False
                )
                .any()
            ):
                raise RuntimeError(
                    "Duplicate dates returned by provider."
                )

            if not (
                frame.index
                .is_monotonic_increasing
            ):
                raise RuntimeError(
                    "Provider dates are not monotonically increasing."
                )

            return (
                frame,
                attempt,
                "success",
            )

        except Exception as exc:
            final_exception = exc

            print(
                "    Download failure: "
                f"{type(exc).__name__}: {exc}"
            )

            if (
                attempt
                < MAX_DOWNLOAD_ATTEMPTS
            ):
                delay = (
                    RETRY_DELAYS_SECONDS[
                        min(
                            attempt - 1,
                            len(
                                RETRY_DELAYS_SECONDS
                            ) - 1,
                        )
                    ]
                )

                print(
                    f"    Retrying after "
                    f"{delay} seconds..."
                )

                time.sleep(
                    delay
                )

    raise RuntimeError(
        f"Failed to download {ticker} after "
        f"{MAX_DOWNLOAD_ATTEMPTS} attempts. "
        f"Last error: {final_exception}"
    )


# =============================================================================
# 10. RAW PROVIDER SNAPSHOT SUMMARY
# =============================================================================

def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def summarize_raw_frame(
    ticker: str,
    instrument_type: str,
    dataframe: pd.DataFrame,
    attempts_used: int,
    downloaded_at_utc: str,
) -> Dict[str, Any]:
    frame = dataframe

    core = frame[
        list(
            EXPECTED_CORE_COLUMNS
        )
    ]

    duplicate_date_rows = int(
        frame.index
        .duplicated(
            keep=False
        )
        .sum()
    )

    core_null_total = int(
        core
        .isna()
        .sum()
        .sum()
    )

    core_null_rows = int(
        core
        .isna()
        .any(axis=1)
        .sum()
    )

    volume = numeric_series(
        frame,
        "Volume",
    )

    zero_volume_rows = int(
        (
            volume == 0
        ).sum()
    )

    negative_volume_rows = int(
        (
            volume < 0
        ).sum()
    )

    dividend_column_present = (
        "Dividends"
        in frame.columns
    )

    split_column_present = (
        "Stock Splits"
        in frame.columns
    )

    dividend_event_rows = (
        int(
            (
                numeric_series(
                    frame,
                    "Dividends",
                )
                .fillna(0)
                != 0
            ).sum()
        )
        if dividend_column_present
        else 0
    )

    split_event_rows = (
        int(
            (
                numeric_series(
                    frame,
                    "Stock Splits",
                )
                .fillna(0)
                != 0
            ).sum()
        )
        if split_column_present
        else 0
    )

    return {
        "ticker": ticker,
        "instrument_type": instrument_type,
        "status": "SUCCESS",
        "attempts_used": int(
            attempts_used
        ),
        "downloaded_at_utc": (
            downloaded_at_utc
        ),
        "rows": int(
            len(frame)
        ),
        "first_date": (
            frame.index
            .min()
            .strftime(
                "%Y-%m-%d"
            )
        ),
        "last_date": (
            frame.index
            .max()
            .strftime(
                "%Y-%m-%d"
            )
        ),
        "unique_dates": int(
            frame.index
            .nunique()
        ),
        "duplicate_date_rows": (
            duplicate_date_rows
        ),
        "date_monotonic_increasing": bool(
            frame.index
            .is_monotonic_increasing
        ),
        "core_null_total": (
            core_null_total
        ),
        "core_null_rows": (
            core_null_rows
        ),
        "zero_volume_rows": (
            zero_volume_rows
        ),
        "negative_volume_rows": (
            negative_volume_rows
        ),
        "dividend_column_present": bool(
            dividend_column_present
        ),
        "split_column_present": bool(
            split_column_present
        ),
        "dividend_event_rows": (
            dividend_event_rows
        ),
        "split_event_rows": (
            split_event_rows
        ),
        "column_count": int(
            len(
                frame.columns
            )
        ),
        "columns": "|".join(
            frame.columns
        ),
    }


# =============================================================================
# 11. STAGING SNAPSHOT WRITES
# =============================================================================

def write_staging_snapshot(
    ticker: str,
    dataframe: pd.DataFrame,
    staging_csv_dir: Path,
    staging_parquet_dir: Path,
) -> Tuple[Path, Path]:
    safe_name = safe_ticker_filename(
        ticker
    )

    csv_path = (
        staging_csv_dir
        / f"{safe_name}.csv"
    )

    parquet_path = (
        staging_parquet_dir
        / f"{safe_name}.parquet"
    )

    csv_frame = (
        dataframe
        .reset_index()
    )

    csv_frame[
        "Date"
    ] = (
        pd.to_datetime(
            csv_frame["Date"]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    # CSV is an audit-friendly representation.
    # Parquet below is the canonical machine-readable raw snapshot.
    csv_frame.to_csv(
        csv_path,
        index=False,
        float_format="%.17g",
    )

    parquet_frame = (
        dataframe
        .reset_index()
    )

    parquet_frame.to_parquet(
        parquet_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    return (
        csv_path,
        parquet_path,
    )


def verify_snapshot_roundtrip(
    original_frame: pd.DataFrame,
    parquet_path: Path,
) -> None:
    """
    Confirm the canonical Parquet snapshot preserves rows, columns and dates.
    This is a storage-integrity check, not data cleaning.
    """

    reloaded = pd.read_parquet(
        parquet_path,
        engine="pyarrow",
    )

    expected_rows = int(
        len(
            original_frame
        )
    )

    if len(reloaded) != expected_rows:
        raise RuntimeError(
            "Parquet roundtrip row-count mismatch."
        )

    expected_columns = [
        "Date",
        *list(
            original_frame.columns
        ),
    ]

    if list(
        reloaded.columns
    ) != expected_columns:
        raise RuntimeError(
            "Parquet roundtrip column-order mismatch."
        )

    expected_dates = (
        pd.DatetimeIndex(
            original_frame.index
        )
        .tz_localize(None)
        .normalize()
    )

    observed_dates = (
        pd.to_datetime(
            reloaded["Date"],
            errors="raise",
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if not np.array_equal(
        expected_dates.to_numpy(),
        observed_dates.to_numpy(),
    ):
        raise RuntimeError(
            "Parquet roundtrip date mismatch."
        )


# =============================================================================
# 12. REFERENCE CALENDAR
# =============================================================================

def build_reference_calendar(
    benchmark_frame: pd.DataFrame,
) -> pd.DataFrame:
    calendar = pd.DataFrame(
        {
            "session_number": np.arange(
                1,
                len(
                    benchmark_frame
                ) + 1,
                dtype=np.int64,
            ),
            "Date": (
                benchmark_frame.index
            ),
        }
    )

    calendar[
        "Date"
    ] = pd.to_datetime(
        calendar["Date"]
    )

    calendar[
        "calendar_year"
    ] = (
        calendar["Date"]
        .dt.year
    )

    calendar[
        "calendar_month"
    ] = (
        calendar["Date"]
        .dt.month
    )

    calendar[
        "day_of_week"
    ] = (
        calendar["Date"]
        .dt.day_name()
    )

    return calendar


def write_reference_calendar_staging(
    calendar: pd.DataFrame,
    path: Path,
) -> None:
    output = (
        calendar.copy()
    )

    output[
        "Date"
    ] = (
        pd.to_datetime(
            output["Date"]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output.to_csv(
        path,
        index=False,
    )


# =============================================================================
# 13. HASH / SOURCE REGISTERS
# =============================================================================

def build_raw_file_hash_records(
    download_rows: list[Dict[str, Any]],
    staging_csv_dir: Path,
    staging_parquet_dir: Path,
) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []

    type_by_ticker = {
        row["ticker"]: (
            row["instrument_type"]
        )
        for row
        in download_rows
    }

    for instrument in INSTRUMENTS:
        ticker = (
            instrument["ticker"]
        )

        safe_name = (
            safe_ticker_filename(
                ticker
            )
        )

        for (
            file_format,
            staging_path,
            final_path,
        ) in (
            (
                "csv",
                staging_csv_dir
                / f"{safe_name}.csv",
                RAW_CSV_DIR
                / f"{safe_name}.csv",
            ),
            (
                "parquet",
                staging_parquet_dir
                / f"{safe_name}.parquet",
                RAW_PARQUET_DIR
                / f"{safe_name}.parquet",
            ),
        ):
            records.append(
                {
                    "ticker": ticker,
                    "instrument_type": (
                        type_by_ticker[
                            ticker
                        ]
                    ),
                    "file_format": (
                        file_format
                    ),
                    "canonical_for_pipeline": (
                        file_format
                        == "parquet"
                    ),
                    "file_path": (
                        str(
                            final_path
                        )
                    ),
                    "relative_path": (
                        relative_to_experiment(
                            final_path
                        )
                    ),
                    "size_bytes": int(
                        staging_path
                        .stat()
                        .st_size
                    ),
                    "sha256": (
                        sha256_file(
                            staging_path
                        )
                    ),
                }
            )

    return records


def collect_source_hashes() -> list[Dict[str, Any]]:
    current_step00 = (
        Path(
            cfg.__file__
        ).resolve()
    )

    frozen_step00 = (
        cfg.get_frozen_script_path()
        .resolve()
    )

    current_step02 = (
        SCRIPT_FILE.resolve()
    )

    with STEP01_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        step01_manifest = json.load(
            handle
        )

    frozen_step01_path: Optional[Path] = None

    for name, record in (
        step01_manifest
        .get(
            "artifacts",
            {},
        )
        .items()
    ):
        if (
            "step01_environment_audit_frozen.py"
            in name
            or str(
                record.get(
                    "path",
                    ""
                )
            ).endswith(
                "step01_environment_audit_frozen.py"
            )
        ):
            frozen_step01_path = Path(
                record["path"]
            )
            break

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
            "current_step02",
            current_step02,
        ),
    ]

    if frozen_step01_path is not None:
        source_paths.append(
            (
                "frozen_step01",
                frozen_step01_path,
            )
        )

    rows = []

    for role, path in source_paths:
        if not path.exists():
            raise RuntimeError(
                f"Source file missing: {path}"
            )

        rows.append(
            {
                "role": role,
                "file_name": (
                    path.name
                ),
                "absolute_path": (
                    str(path)
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

    step00_current_sha = next(
        row["sha256"]
        for row in rows
        if row["role"]
        == "current_step00"
    )

    step00_frozen_sha = next(
        row["sha256"]
        for row in rows
        if row["role"]
        == "frozen_step00"
    )

    if (
        step00_current_sha
        != step00_frozen_sha
    ):
        raise RuntimeError(
            "Active step00_config.py differs from "
            "the frozen Step-00 source snapshot."
        )

    return rows


# =============================================================================
# 14. ACQUISITION-LEVEL VALIDATION
# =============================================================================

def validate_acquisition(
    download_rows: list[Dict[str, Any]],
    downloaded_frames: Dict[str, pd.DataFrame],
) -> None:
    expected_instrument_count = (
        len(
            cfg.STOCK_TICKERS
        )
        + 1
    )

    if (
        len(
            download_rows
        )
        != expected_instrument_count
    ):
        raise AssertionError(
            "Unexpected number of downloaded instruments."
        )

    if (
        set(
            downloaded_frames
            .keys()
        )
        != set(
            instrument["ticker"]
            for instrument
            in INSTRUMENTS
        )
    ):
        raise AssertionError(
            "Downloaded instrument set differs from Step 00."
        )

    if not all(
        row["status"]
        == "SUCCESS"
        for row
        in download_rows
    ):
        raise AssertionError(
            "One or more instruments did not download successfully."
        )

    if any(
        row["duplicate_date_rows"]
        != 0
        for row
        in download_rows
    ):
        raise AssertionError(
            "Provider snapshot contains duplicate dates."
        )

    if not all(
        row[
            "date_monotonic_increasing"
        ]
        for row
        in download_rows
    ):
        raise AssertionError(
            "One or more provider series are not date-monotonic."
        )

    benchmark_frame = (
        downloaded_frames[
            cfg.MARKET_INDEX_TICKER
        ]
    )

    if benchmark_frame.empty:
        raise AssertionError(
            "NIFTY-50 reference series is empty."
        )

    if (
        benchmark_frame.index
        .min()
        < cfg.RAW_DATA_START_DATE
    ):
        raise AssertionError(
            "Benchmark has observations before requested start."
        )

    if (
        benchmark_frame.index
        .max()
        > cfg.RAW_DATA_END_DATE
    ):
        raise AssertionError(
            "Benchmark has observations after requested end."
        )

    observed_tickers = [
        row["ticker"]
        for row
        in download_rows
    ]

    if (
        len(
            observed_tickers
        )
        != len(
            set(
                observed_tickers
            )
        )
    ):
        raise AssertionError(
            "Duplicate instrument records detected."
        )


# =============================================================================
# 15. STAGING / PROMOTION HELPERS
# =============================================================================

def create_staging_root() -> Path:
    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    staging_root = (
        cfg.RAW_DATA_DIR
        / f"_step02_staging_{timestamp}_{os.getpid()}"
    )

    if staging_root.exists():
        raise RuntimeError(
            f"Unexpected existing staging path: {staging_root}"
        )

    (
        staging_root
        / "csv"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        staging_root
        / "parquet"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )

    return staging_root


def promote_raw_snapshot(
    staging_root: Path,
) -> None:
    if RAW_SNAPSHOT_ROOT.exists():
        raise RuntimeError(
            "Canonical raw snapshot root appeared during acquisition. "
            "Refusing to overwrite it."
        )

    os.replace(
        staging_root,
        RAW_SNAPSHOT_ROOT,
    )


# =============================================================================
# 16. MANIFEST / SUMMARY BUILDERS
# =============================================================================

def artifact_record(
    path: Path,
) -> Dict[str, Any]:
    return {
        "path": str(
            path
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


def build_calendar_summary(
    calendar: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "reference_ticker": (
            cfg.MARKET_INDEX_TICKER
        ),
        "session_count": int(
            len(
                calendar
            )
        ),
        "first_session": (
            calendar["Date"]
            .min()
            .strftime(
                "%Y-%m-%d"
            )
        ),
        "last_session": (
            calendar["Date"]
            .max()
            .strftime(
                "%Y-%m-%d"
            )
        ),
    }


def build_manifest(
    *,
    step00_info: Dict[str, Any],
    step01_info: Dict[str, Any],
    acquisition_started_at_utc: str,
    acquisition_finished_at_utc: str,
    download_rows: list[Dict[str, Any]],
    raw_file_hash_rows: list[Dict[str, Any]],
    calendar_summary: Dict[str, Any],
    source_hash_rows: list[Dict[str, Any]],
    lock_sha_after: str,
) -> Dict[str, Any]:

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
        "experiment_tag": (
            cfg.EXPERIMENT_TAG
        ),
        "experiment_root": (
            str(
                cfg.EXPERIMENT_ROOT_DIR
            )
        ),
        "acquisition_started_at_utc": (
            acquisition_started_at_utc
        ),
        "acquisition_finished_at_utc": (
            acquisition_finished_at_utc
        ),
        "step00": (
            step00_info
        ),
        "step01": (
            step01_info
        ),
        "data_source": {
            "provider": (
                DATA_SOURCE
            ),
            "library": (
                DATA_LIBRARY
            ),
            "yfinance_version": (
                yf.__version__
            ),
            "scientific_start_date": (
                timestamp_string(
                    cfg.RAW_DATA_START_DATE
                )
            ),
            "scientific_end_date_inclusive": (
                timestamp_string(
                    cfg.RAW_DATA_END_DATE
                )
            ),
            "yahoo_api_end_date_exclusive": (
                timestamp_string(
                    YAHOO_API_END_DATE
                )
            ),
            "interval": "1d",
            "auto_adjust": False,
            "actions": True,
            "repair": False,
            "keepna": True,
            "threads": False,
            "sequential_download": True,
        },
        "universe": {
            "stock_count": int(
                len(
                    cfg.STOCK_TICKERS
                )
            ),
            "instrument_count": int(
                len(
                    INSTRUMENTS
                )
            ),
            "stock_tickers": list(
                cfg.STOCK_TICKERS
            ),
            "benchmark_ticker": (
                cfg.MARKET_INDEX_TICKER
            ),
        },
        "download_records": (
            download_rows
        ),
        "raw_file_records": (
            raw_file_hash_rows
        ),
        "reference_calendar": (
            calendar_summary
        ),
        "source_code_hashes": (
            source_hash_rows
        ),
        "storage_contract": {
            "canonical_machine_readable_format": (
                "parquet"
            ),
            "audit_human_readable_format": (
                "csv"
            ),
            "parquet_engine": (
                "pyarrow"
            ),
            "parquet_compression": (
                "snappy"
            ),
            "csv_float_format": (
                "%.17g"
            ),
        },
        "scientific_boundary": {
            "cleaning_applied": False,
            "imputation_applied": False,
            "technical_features_constructed": False,
            "targets_constructed": False,
            "scalers_fit": False,
            "regimes_estimated": False,
            "model_selection_performed": False,
            "threshold_selection_performed": False,
            "final_test_scoring_performed": False,
            "raw_acquisition_includes_final_test_calendar_period": True,
            "note": (
                "Step 02 acquires and freezes the complete predeclared "
                "provider source window, including dates later belonging "
                "to the held-out test calendar. This does not unlock, "
                "score, or use the final test for modelling decisions."
            ),
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
                lock_sha_after
            ),
            "lock_unchanged": (
                lock_sha_after
                == step00_info[
                    "final_test_lock_sha256"
                ]
            ),
        },
        "canonical_artifacts": {
            "raw_snapshot_root": (
                str(
                    RAW_SNAPSHOT_ROOT
                )
            ),
            "download_log": (
                artifact_record(
                    DOWNLOAD_LOG_CSV
                )
            ),
            "raw_file_hash_register": (
                artifact_record(
                    RAW_FILE_HASH_CSV
                )
            ),
            "source_hash_register": (
                artifact_record(
                    SOURCE_HASH_CSV
                )
            ),
            "reference_calendar": (
                artifact_record(
                    REFERENCE_CALENDAR_CSV
                )
            ),
            "frozen_step02_script": (
                artifact_record(
                    FROZEN_STEP02_SCRIPT
                )
            ),
        },
    }


def build_summary_text(
    *,
    step00_info: Dict[str, Any],
    step01_info: Dict[str, Any],
    download_rows: list[Dict[str, Any]],
    calendar_summary: Dict[str, Any],
    manifest_sha256: str,
) -> str:
    lines = []

    lines.append(
        STEP_NAME
    )
    lines.append(
        "=" * 92
    )
    lines.append(
        "Status: PASS"
    )
    lines.append(
        f"Experiment tag: {cfg.EXPERIMENT_TAG}"
    )
    lines.append(
        "Configuration fingerprint: "
        + step00_info[
            "configuration_fingerprint_sha256"
        ]
    )
    lines.append(
        "Step-01 manifest SHA-256: "
        + step01_info[
            "step01_manifest_sha256"
        ]
    )
    lines.append(
        f"Python executable: {sys.executable}"
    )
    lines.append(
        f"yfinance: {yf.__version__}"
    )
    lines.append(
        "Scientific source window: "
        f"{cfg.RAW_DATA_START_DATE.date()} -> "
        f"{cfg.RAW_DATA_END_DATE.date()} inclusive"
    )
    lines.append(
        "Yahoo API end: "
        f"{YAHOO_API_END_DATE.date()} exclusive"
    )
    lines.append(
        f"Instrument count: {len(download_rows)}"
    )
    lines.append(
        f"Stock count: {len(cfg.STOCK_TICKERS)}"
    )
    lines.append(
        f"Benchmark: {cfg.MARKET_INDEX_TICKER}"
    )
    lines.append(
        "Reference sessions: "
        f"{calendar_summary['session_count']}"
    )
    lines.append(
        "Reference calendar: "
        f"{calendar_summary['first_session']} -> "
        f"{calendar_summary['last_session']}"
    )
    lines.append(
        f"Raw snapshot: {RAW_SNAPSHOT_ROOT}"
    )
    lines.append(
        f"Final-test state: {step00_info['final_test_state']}"
    )
    lines.append(
        f"Step-02 manifest SHA-256: {manifest_sha256}"
    )

    return (
        "\n".join(
            lines
        )
        + "\n"
    )


# =============================================================================
# 17. STEP-02 VERIFICATION
# =============================================================================

def verify_existing_step02() -> int:
    print(
        "=" * 118
    )
    print(
        "STEP 02 — VERIFY EXISTING MARKET-DATA ACQUISITION FREEZE"
    )
    print(
        "=" * 118
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

    if not (
        ACQUISITION_MANIFEST_JSON
        .exists()
    ):
        print(
            "[FAIL] Step-02 manifest is missing."
        )
        return 20

    if not (
        ACQUISITION_MANIFEST_SHA
        .exists()
    ):
        print(
            "[FAIL] Step-02 manifest SHA sidecar is missing."
        )
        return 21

    with ACQUISITION_MANIFEST_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    expected_manifest_sha = (
        ACQUISITION_MANIFEST_SHA
        .read_text(
            encoding="utf-8"
        )
        .strip()
        .split()[0]
    )

    actual_manifest_sha = (
        sha256_file(
            ACQUISITION_MANIFEST_JSON
        )
    )

    if (
        expected_manifest_sha
        != actual_manifest_sha
    ):
        print(
            "[FAIL] Step-02 manifest SHA mismatch."
        )
        return 22

    if (
        manifest
        .get("step", {})
        .get("status")
        != "PASS"
    ):
        print(
            "[FAIL] Stored Step-02 status is not PASS."
        )
        return 23

    if (
        manifest
        .get("step00", {})
        .get(
            "configuration_fingerprint_sha256"
        )
        != step00_info[
            "configuration_fingerprint_sha256"
        ]
    ):
        print(
            "[FAIL] Step-02 / Step-00 fingerprint mismatch."
        )
        return 24

    if (
        manifest
        .get("step01", {})
        .get(
            "step01_manifest_sha256"
        )
        != step01_info[
            "step01_manifest_sha256"
        ]
    ):
        print(
            "[FAIL] Step-02 / Step-01 manifest mismatch."
        )
        return 25

    failures = []

    # Verify canonical supporting artifacts.
    for name, record in (
        manifest
        .get(
            "canonical_artifacts",
            {}
        )
        .items()
    ):
        if name == "raw_snapshot_root":
            continue

        path = Path(
            record["path"]
        )

        if not path.exists():
            failures.append(
                f"missing: {name}"
            )
            continue

        actual_sha = (
            sha256_file(
                path
            )
        )

        if (
            actual_sha
            != record["sha256"]
        ):
            failures.append(
                f"hash mismatch: {name}"
            )

    # Verify every raw snapshot file.
    for record in manifest.get(
        "raw_file_records",
        [],
    ):
        path = Path(
            record["file_path"]
        )

        if not path.exists():
            failures.append(
                f"missing raw file: {path}"
            )
            continue

        actual_sha = (
            sha256_file(
                path
            )
        )

        if actual_sha != record["sha256"]:
            failures.append(
                f"raw hash mismatch: {path}"
            )

    if failures:
        for item in failures:
            print(
                f"[FAIL] {item}"
            )

        print(
            "=" * 118
        )
        print(
            "STEP 02 VERIFICATION STATUS: FAIL"
        )
        print(
            "=" * 118
        )
        return 26

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
        f"{actual_manifest_sha}"
    )
    print(
        f"[PASS] Raw files verified  : "
        f"{len(manifest.get('raw_file_records', []))}"
    )
    print(
        f"[PASS] Final-test state     : "
        f"{step00_info['final_test_state']}"
    )
    print(
        "=" * 118
    )
    print(
        "STEP 02 VERIFICATION STATUS: PASS"
    )
    print(
        "=" * 118
    )

    return 0


# =============================================================================
# 18. FAILED ATTEMPT DIAGNOSTIC
# =============================================================================

def write_failure_diagnostic(
    *,
    stage: str,
    error: Exception,
    acquisition_started_at_utc: Optional[str],
    partial_download_rows: list[Dict[str, Any]],
    staging_root: Optional[Path],
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
            "02_market_data_acquisition_FAILED_"
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
        "yfinance_version": (
            yf.__version__
        ),
        "acquisition_started_at_utc": (
            acquisition_started_at_utc
        ),
        "partial_download_records": (
            partial_download_rows
        ),
        "staging_root": (
            str(
                staging_root
            )
            if staging_root
            is not None
            else None
        ),
        "canonical_raw_snapshot_created": (
            RAW_SNAPSHOT_ROOT
            .exists()
        ),
        "canonical_manifest_created": (
            ACQUISITION_MANIFEST_JSON
            .exists()
        ),
    }

    atomic_write_json(
        path,
        payload,
    )

    return path


# =============================================================================
# 19. NEW ACQUISITION
# =============================================================================

def run_new_acquisition() -> int:
    print(
        "=" * 118
    )
    print(
        STEP_NAME
    )
    print(
        "=" * 118
    )

    stage = (
        "prerequisite_verification"
    )

    acquisition_started_at_utc: Optional[str] = None
    staging_root: Optional[Path] = None
    partial_download_rows: list[Dict[str, Any]] = []

    try:
        print(
            "[1/9] Verifying Step 00 and final-test lock..."
        )

        step00_info = (
            verify_step00_contract()
        )

        print(
            "      Configuration fingerprint: "
            f"{step00_info['configuration_fingerprint_sha256']}"
        )
        print(
            "      Final-test state: "
            f"{step00_info['final_test_state']}"
        )

        print(
            "[2/9] Verifying frozen Step-01 environment..."
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
            "      Step-01 manifest SHA-256: "
            f"{step01_info['step01_manifest_sha256']}"
        )
        print(
            "      Python executable: "
            f"{sys.executable}"
        )
        print(
            "      Conda environment: "
            f"{step01_info.get('conda_environment')}"
        )

        stage = (
            "output_collision_guard"
        )

        print(
            "[3/9] Checking immutable Step-02 output target..."
        )

        assert_new_step02_target()

        cfg.create_directories()

        staging_root = (
            create_staging_root()
        )

        staging_csv_dir = (
            staging_root
            / "csv"
        )

        staging_parquet_dir = (
            staging_root
            / "parquet"
        )

        print(
            "      Staging root: "
            f"{staging_root}"
        )

        stage = (
            "market_data_acquisition"
        )

        acquisition_started_at_utc = (
            utc_now_iso()
        )

        random.seed(
            cfg.GLOBAL_RANDOM_SEED
        )

        np.random.seed(
            cfg.GLOBAL_RANDOM_SEED
        )

        print(
            "[4/9] Downloading 20 equities + NIFTY-50 sequentially..."
        )
        print(
            f"      Provider: {DATA_SOURCE}"
        )
        print(
            f"      yfinance: {yf.__version__}"
        )
        print(
            "      Scientific window: "
            f"{cfg.RAW_DATA_START_DATE.date()} -> "
            f"{cfg.RAW_DATA_END_DATE.date()} inclusive"
        )
        print(
            "      Yahoo request end: "
            f"{YAHOO_API_END_DATE.date()} exclusive"
        )
        print(
            "      auto_adjust=False | actions=True | "
            "repair=False | keepna=True | threads=False"
        )
        print(
            "      Cleaning: NONE"
        )

        download_rows: list[
            Dict[str, Any]
        ] = []

        downloaded_frames: Dict[
            str,
            pd.DataFrame,
        ] = {}

        for (
            instrument_number,
            instrument,
        ) in enumerate(
            INSTRUMENTS,
            start=1,
        ):
            ticker = (
                instrument[
                    "ticker"
                ]
            )

            instrument_type = (
                instrument[
                    "instrument_type"
                ]
            )

            print()
            print(
                f"      "
                f"[{instrument_number:02d}/{len(INSTRUMENTS):02d}] "
                f"{ticker}"
            )

            (
                frame,
                attempts_used,
                _,
            ) = (
                download_one_instrument(
                    ticker=ticker
                )
            )

            downloaded_at_utc = (
                utc_now_iso()
            )

            summary = (
                summarize_raw_frame(
                    ticker=ticker,
                    instrument_type=(
                        instrument_type
                    ),
                    dataframe=frame,
                    attempts_used=(
                        attempts_used
                    ),
                    downloaded_at_utc=(
                        downloaded_at_utc
                    ),
                )
            )

            (
                csv_path,
                parquet_path,
            ) = (
                write_staging_snapshot(
                    ticker=ticker,
                    dataframe=frame,
                    staging_csv_dir=(
                        staging_csv_dir
                    ),
                    staging_parquet_dir=(
                        staging_parquet_dir
                    ),
                )
            )

            verify_snapshot_roundtrip(
                original_frame=frame,
                parquet_path=(
                    parquet_path
                ),
            )

            summary[
                "csv_sha256"
            ] = (
                sha256_file(
                    csv_path
                )
            )

            summary[
                "parquet_sha256"
            ] = (
                sha256_file(
                    parquet_path
                )
            )

            download_rows.append(
                summary
            )

            partial_download_rows = (
                list(
                    download_rows
                )
            )

            downloaded_frames[
                ticker
            ] = frame

            print(
                "        Success: "
                f"rows={len(frame):,} | "
                f"{frame.index.min().date()} -> "
                f"{frame.index.max().date()} | "
                f"nulls={summary['core_null_total']} | "
                f"zero_volume={summary['zero_volume_rows']}"
            )

            if (
                instrument_number
                < len(
                    INSTRUMENTS
                )
            ):
                time.sleep(
                    REQUEST_GAP_SECONDS
                )

        stage = (
            "acquisition_validation"
        )

        print()
        print(
            "[5/9] Validating acquisition-level integrity..."
        )

        validate_acquisition(
            download_rows=(
                download_rows
            ),
            downloaded_frames=(
                downloaded_frames
            ),
        )

        benchmark_frame = (
            downloaded_frames[
                cfg.MARKET_INDEX_TICKER
            ]
        )

        calendar = (
            build_reference_calendar(
                benchmark_frame=(
                    benchmark_frame
                )
            )
        )

        calendar_summary = (
            build_calendar_summary(
                calendar
            )
        )

        print(
            "      NIFTY reference sessions: "
            f"{calendar_summary['session_count']:,}"
        )
        print(
            "      Reference range: "
            f"{calendar_summary['first_session']} -> "
            f"{calendar_summary['last_session']}"
        )

        stage = (
            "staging_audit_build"
        )

        print(
            "[6/9] Building hashes, calendar and source-code audit..."
        )

        raw_file_hash_rows = (
            build_raw_file_hash_records(
                download_rows=(
                    download_rows
                ),
                staging_csv_dir=(
                    staging_csv_dir
                ),
                staging_parquet_dir=(
                    staging_parquet_dir
                ),
            )
        )

        source_hash_rows = (
            collect_source_hashes()
        )

        # Build staging audit files under the experiment audit/reference roots
        # using temporary names; canonical names appear only after raw snapshot
        # promotion succeeds.
        temp_download_log = (
            DOWNLOAD_LOG_CSV
            .with_name(
                DOWNLOAD_LOG_CSV.name
                + ".tmp_step02"
            )
        )

        temp_raw_hash_log = (
            RAW_FILE_HASH_CSV
            .with_name(
                RAW_FILE_HASH_CSV.name
                + ".tmp_step02"
            )
        )

        temp_source_hash_log = (
            SOURCE_HASH_CSV
            .with_name(
                SOURCE_HASH_CSV.name
                + ".tmp_step02"
            )
        )

        temp_calendar = (
            REFERENCE_CALENDAR_CSV
            .with_name(
                REFERENCE_CALENDAR_CSV.name
                + ".tmp_step02"
            )
        )

        for path in (
            temp_download_log,
            temp_raw_hash_log,
            temp_source_hash_log,
            temp_calendar,
        ):
            if path.exists():
                path.unlink()

        atomic_write_csv(
            temp_download_log,
            download_rows,
            fieldnames=list(
                download_rows[0]
                .keys()
            ),
        )

        atomic_write_csv(
            temp_raw_hash_log,
            raw_file_hash_rows,
            fieldnames=list(
                raw_file_hash_rows[0]
                .keys()
            ),
        )

        atomic_write_csv(
            temp_source_hash_log,
            source_hash_rows,
            fieldnames=list(
                source_hash_rows[0]
                .keys()
            ),
        )

        write_reference_calendar_staging(
            calendar=(
                calendar
            ),
            path=(
                temp_calendar
            ),
        )

        stage = (
            "raw_snapshot_promotion"
        )

        print(
            "[7/9] Promoting staged provider snapshot to immutable canonical location..."
        )

        promote_raw_snapshot(
            staging_root=(
                staging_root
            )
        )

        staging_root = None

        # Promote audit/reference files.
        os.replace(
            temp_download_log,
            DOWNLOAD_LOG_CSV,
        )

        os.replace(
            temp_raw_hash_log,
            RAW_FILE_HASH_CSV,
        )

        os.replace(
            temp_source_hash_log,
            SOURCE_HASH_CSV,
        )

        os.replace(
            temp_calendar,
            REFERENCE_CALENDAR_CSV,
        )

        # Freeze the exact source used.
        shutil.copy2(
            SCRIPT_FILE,
            FROZEN_STEP02_SCRIPT,
        )

        # Check raw file hashes again at canonical paths.
        for record in raw_file_hash_rows:
            canonical_path = Path(
                record[
                    "file_path"
                ]
            )

            actual_sha = (
                sha256_file(
                    canonical_path
                )
            )

            if actual_sha != record["sha256"]:
                raise RuntimeError(
                    "Raw snapshot hash changed during promotion:\n"
                    f"{canonical_path}"
                )

        stage = (
            "manifest_freeze"
        )

        print(
            "[8/9] Verifying lock unchanged and freezing acquisition manifest..."
        )

        lock_path = (
            cfg.get_final_test_lock_path()
        )

        lock_sha_after = (
            sha256_file(
                lock_path
            )
        )

        if (
            lock_sha_after
            != step00_info[
                "final_test_lock_sha256"
            ]
        ):
            raise RuntimeError(
                "Final-test lock file changed during Step 02."
            )

        acquisition_finished_at_utc = (
            utc_now_iso()
        )

        manifest = (
            build_manifest(
                step00_info=(
                    step00_info
                ),
                step01_info=(
                    step01_info
                ),
                acquisition_started_at_utc=(
                    acquisition_started_at_utc
                ),
                acquisition_finished_at_utc=(
                    acquisition_finished_at_utc
                ),
                download_rows=(
                    download_rows
                ),
                raw_file_hash_rows=(
                    raw_file_hash_rows
                ),
                calendar_summary=(
                    calendar_summary
                ),
                source_hash_rows=(
                    source_hash_rows
                ),
                lock_sha_after=(
                    lock_sha_after
                ),
            )
        )

        atomic_write_json(
            ACQUISITION_MANIFEST_JSON,
            manifest,
        )

        manifest_sha256 = (
            sha256_file(
                ACQUISITION_MANIFEST_JSON
            )
        )

        atomic_write_text(
            ACQUISITION_MANIFEST_SHA,
            (
                f"{manifest_sha256}  "
                f"{ACQUISITION_MANIFEST_JSON.name}\n"
            ),
        )

        summary_text = (
            build_summary_text(
                step00_info=(
                    step00_info
                ),
                step01_info=(
                    step01_info
                ),
                download_rows=(
                    download_rows
                ),
                calendar_summary=(
                    calendar_summary
                ),
                manifest_sha256=(
                    manifest_sha256
                ),
            )
        )

        atomic_write_text(
            STEP02_SUMMARY_TXT,
            summary_text,
        )

        stage = (
            "final_console_report"
        )

        print(
            "[9/9] Final acquisition report..."
        )
        print()
        print(
            "-" * 118
        )

        header = (
            f"{'Ticker':<17}"
            f"{'Type':<18}"
            f"{'Rows':>7}"
            f"{'First':>13}"
            f"{'Last':>13}"
            f"{'Nulls':>8}"
            f"{'ZeroVol':>9}"
            f"{'Div':>7}"
            f"{'Splits':>8}"
        )

        print(
            header
        )

        print(
            "-" * 118
        )

        for row in download_rows:
            print(
                f"{row['ticker']:<17}"
                f"{row['instrument_type']:<18}"
                f"{row['rows']:>7}"
                f"{row['first_date']:>13}"
                f"{row['last_date']:>13}"
                f"{row['core_null_total']:>8}"
                f"{row['zero_volume_rows']:>9}"
                f"{row['dividend_event_rows']:>7}"
                f"{row['split_event_rows']:>8}"
            )

        print(
            "-" * 118
        )
        print(
            "NIFTY reference sessions   : "
            f"{calendar_summary['session_count']:,}"
        )
        print(
            "Reference first session    : "
            f"{calendar_summary['first_session']}"
        )
        print(
            "Reference last session     : "
            f"{calendar_summary['last_session']}"
        )
        print(
            "Raw snapshot root          : "
            f"{RAW_SNAPSHOT_ROOT}"
        )
        print(
            "Canonical machine format   : Parquet"
        )
        print(
            "Download log               : "
            f"{DOWNLOAD_LOG_CSV}"
        )
        print(
            "Raw file hashes            : "
            f"{RAW_FILE_HASH_CSV}"
        )
        print(
            "Reference calendar         : "
            f"{REFERENCE_CALENDAR_CSV}"
        )
        print(
            "Acquisition manifest       : "
            f"{ACQUISITION_MANIFEST_JSON}"
        )
        print(
            "Manifest SHA-256           : "
            f"{manifest_sha256}"
        )
        print(
            "Final-test state           : "
            f"{step00_info['final_test_state']}"
        )
        print(
            "=" * 118
        )
        print(
            "PASS: All 20 equities and the NIFTY-50 benchmark were acquired."
        )
        print(
            "PASS: CSV audit copies and canonical Parquet snapshots were frozen."
        )
        print(
            "PASS: Step-00 and Step-01 integrity checks passed."
        )
        print(
            "PASS: No cleaning, feature construction, target construction, "
            "or model evaluation occurred."
        )
        print(
            "PASS: Final held-out test remains LOCKED_NOT_OPENED."
        )
        print(
            "=" * 118
        )
        print(
            "Do not run Step 03 yet. Paste the complete terminal output for audit."
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
                acquisition_started_at_utc=(
                    acquisition_started_at_utc
                ),
                partial_download_rows=(
                    partial_download_rows
                ),
                staging_root=(
                    staging_root
                ),
            )
        )

        # Staging is non-canonical. Remove it after recording the failure so a
        # clean rerun can start from an empty acquisition target.
        if (
            staging_root
            is not None
            and staging_root.exists()
        ):
            shutil.rmtree(
                staging_root,
                ignore_errors=True,
            )

        print()
        print(
            "=" * 118
        )
        print(
            "STEP 02 FAILED — ACQUISITION NOT FROZEN"
        )
        print(
            "=" * 118
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
            "=" * 118
        )
        print(
            "Do not proceed to Step 03. Correct the issue and rerun Step 02."
        )

        return 30


# =============================================================================
# 20. CLI / ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire and freeze the clean V3 Yahoo Finance "
            "provider snapshot."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify the already frozen Step-02 acquisition "
            "without downloading anything."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verify_only:
        return verify_existing_step02()

    return run_new_acquisition()


if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "\nStep 02 interrupted by user."
        )
        sys.exit(
            130
        )

    except Exception as exc:
        print()
        print(
            "=" * 118
        )
        print(
            "STEP 02 UNHANDLED FAILURE"
        )
        print(
            "=" * 118
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
            "=" * 118
        )
        sys.exit(
            1
        )
