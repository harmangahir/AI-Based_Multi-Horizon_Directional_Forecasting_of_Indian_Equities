#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 04 — SESSION CLASSIFICATION, CORPORATE-ACTION AUDIT,
          AND FROZEN CLEANING POLICY (CLEAN V3)

Purpose
-------
Convert the immutable Step-03 diagnostics into an explicit, auditable,
pre-modelling cleaning/calendar policy.

This stage:
1. Verifies the complete Step-00 -> Step-01 -> Step-02 -> Step-03 chain.
2. Classifies all source dates using the Step-03 cross-sectional consensus.
3. Applies ONLY the predeclared Step-00 date-wide anomaly rule:
      at least 90% of stocks simultaneously exhibit
      zero volume AND identical/flat OHLC.
4. Excludes stock dates without valid NIFTY context from the common panel.
5. Freezes the final common analysis calendar.
6. Builds a corporate-action event register.
7. Audits split dates and raw-volume behavior without transforming volume.
8. Audits large adjusted-return flags against nearby corporate actions.
9. Freezes the deterministic cleaning policy and all hashes.

Scientific boundary
-------------------
This stage DOES NOT:
- overwrite or alter any Step-02 raw snapshot,
- materialize the cleaned stock panel (Step 05),
- impute prices, volume, features, or targets,
- use validation/test predictive performance,
- tune models or thresholds,
- score the final held-out test.

The raw-quality rules applied here were frozen before model evaluation. Applying
those deterministic rules to the predeclared source window is a data-integrity
operation, not final-test model selection.

Corporate-action policy
-----------------------
Step 00 requires auditing raw OHLC, Adj Close, actions, and date-specific
adjustment factors before canonical adjusted OHLC is constructed.

Step 04 therefore freezes the following downstream policy:
- canonical adjusted OHLC in Step 05 will use:
      adjustment_factor = Adj Close / Close
      adjusted Open/High/Low/Close = raw field * adjustment_factor
- Adj Close remains the canonical adjusted close.
- provider Volume is retained as supplied; no manual split adjustment and no
  volume imputation are applied. Split dates are audited explicitly instead of
  inventing a second volume transformation.

Expected location
-----------------
<repository>/q1_revision/step04_session_and_corporate_action_policy.py

Run
---
python step04_session_and_corporate_action_policy.py

Verify an already frozen Step 04
--------------------------------
python step04_session_and_corporate_action_policy.py --verify-only
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
        "step04_session_and_corporate_action_policy.py must be located inside "
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

STEP_NUMBER = 4
STEP_ID = "04"
STEP_NAME = (
    "STEP 04 — SESSION CLASSIFICATION, CORPORATE-ACTION AUDIT, "
    "AND FROZEN CLEANING POLICY"
)
SCHEMA_VERSION = "1.0"

DATE_WIDE_RULE_ENABLED = bool(
    cfg.DATE_WIDE_ANOMALY_RULE_ENABLED
)

DATE_WIDE_MIN_STOCK_FRACTION = float(
    cfg.DATE_WIDE_ANOMALY_MIN_STOCK_FRACTION
)

DATE_WIDE_THRESHOLD_COUNT = int(
    math.ceil(
        DATE_WIDE_MIN_STOCK_FRACTION
        * len(cfg.STOCK_TICKERS)
    )
)

CORPORATE_ACTION_PROXIMITY_CALENDAR_DAYS = 7
SPLIT_VOLUME_WINDOW_SESSIONS = 20

ACTION_COLUMNS = (
    "Dividends",
    "Stock Splits",
    "Capital Gains",
)


# =============================================================================
# 2. FROZEN INPUT PATHS
# =============================================================================

RAW_SNAPSHOT_ROOT = (
    cfg.RAW_DATA_DIR
    / "provider_snapshot"
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

STEP02_MANIFEST = (
    cfg.MANIFEST_DIR
    / "02_market_data_acquisition_manifest.json"
)

STEP02_MANIFEST_SHA = (
    cfg.MANIFEST_DIR
    / "02_market_data_acquisition_manifest.sha256"
)

STEP03_MANIFEST = (
    cfg.MANIFEST_DIR
    / "03_market_integrity_audit_manifest.json"
)

STEP03_MANIFEST_SHA = (
    cfg.MANIFEST_DIR
    / "03_market_integrity_audit_manifest.sha256"
)

STEP03_SUMMARY = (
    cfg.AUDIT_DIR
    / "03_market_integrity_summary.json"
)

STEP03_NIFTY_AUDIT = (
    cfg.AUDIT_DIR
    / "03_nifty_session_audit.csv"
)

STEP03_REFERENCE_CALENDAR = (
    cfg.REFERENCE_DATA_DIR
    / "03_candidate_nifty_price_calendar.csv"
)

STEP03_CROSS_SECTIONAL_CONSENSUS = (
    cfg.AUDIT_DIR
    / "03_cross_sectional_date_consensus.csv"
)

STEP03_LARGE_RETURN_FLAGS = (
    cfg.AUDIT_DIR
    / "03_large_adjusted_return_flags.csv"
)

STEP03_CORPORATE_ACTION_AUDIT = (
    cfg.AUDIT_DIR
    / "03_corporate_action_adjustment_audit.csv"
)


# =============================================================================
# 3. STEP-04 OUTPUT PATHS
# =============================================================================

DATE_CLASSIFICATION_CSV = (
    cfg.AUDIT_DIR
    / "04_session_classification.csv"
)

CORPORATE_ACTION_REGISTER_CSV = (
    cfg.AUDIT_DIR
    / "04_corporate_action_register.csv"
)

SPLIT_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "04_stock_split_volume_audit.csv"
)

LARGE_RETURN_ACTION_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "04_large_return_corporate_action_audit.csv"
)

EXCLUDED_DATE_REGISTER_CSV = (
    cfg.AUDIT_DIR
    / "04_excluded_date_register.csv"
)

FINAL_REFERENCE_CALENDAR_CSV = (
    cfg.REFERENCE_DATA_DIR
    / "04_final_analysis_calendar.csv"
)

SOURCE_HASH_CSV = (
    cfg.AUDIT_DIR
    / "04_source_file_hashes.csv"
)

CLEANING_POLICY_JSON = (
    cfg.MANIFEST_DIR
    / "04_cleaning_policy.json"
)

STEP04_MANIFEST_JSON = (
    cfg.MANIFEST_DIR
    / "04_session_policy_manifest.json"
)

STEP04_MANIFEST_SHA = (
    cfg.MANIFEST_DIR
    / "04_session_policy_manifest.sha256"
)

FROZEN_STEP04_SCRIPT = (
    cfg.MANIFEST_DIR
    / "step04_session_and_corporate_action_policy_frozen.py"
)

STEP04_SUMMARY_TXT = (
    cfg.REPORT_DIR
    / "04_session_and_cleaning_policy_summary.txt"
)

FAILED_ATTEMPT_DIR = (
    cfg.AUDIT_DIR
    / "step04_failed_attempts"
)


# =============================================================================
# 4. OUTPUT SCHEMAS
# =============================================================================

DATE_CLASSIFICATION_FIELDS = [
    "Date",
    "day_of_week",
    "nifty_raw_row_present",
    "nifty_price_valid",
    "candidate_nifty_session",
    "stocks_present",
    "stocks_nonzero_volume",
    "stocks_zero_volume",
    "stocks_zero_volume_flat_ohlc",
    "stocks_structurally_invalid",
    "date_wide_anomaly_threshold_count",
    "date_wide_anomaly_candidate",
    "broad_stock_presence",
    "classification",
    "modelling_policy",
    "primary_exclusion_code",
    "rationale",
]

EXCLUSION_FIELDS = [
    "Date",
    "classification",
    "modelling_policy",
    "primary_exclusion_code",
    "candidate_nifty_session",
    "nifty_raw_row_present",
    "nifty_price_valid",
    "stocks_present",
    "stocks_nonzero_volume",
    "stocks_zero_volume",
    "stocks_zero_volume_flat_ohlc",
    "date_wide_anomaly_candidate",
    "reason",
]

FINAL_CALENDAR_FIELDS = [
    "session_number",
    "Date",
    "calendar_year",
    "calendar_month",
    "day_of_week",
]

CORPORATE_ACTION_FIELDS = [
    "ticker",
    "Date",
    "Dividends",
    "Stock Splits",
    "Capital Gains",
    "action_types",
    "action_event_count_on_row",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "adjustment_factor",
    "is_candidate_nifty_session",
    "is_final_analysis_session",
]

SPLIT_AUDIT_FIELDS = [
    "ticker",
    "Date",
    "split_ratio",
    "previous_stock_date",
    "next_stock_date",
    "previous_close",
    "split_date_close",
    "next_close",
    "previous_adj_close",
    "split_date_adj_close",
    "next_adj_close",
    "previous_volume",
    "split_date_volume",
    "next_volume",
    "pre_split_positive_volume_median",
    "post_split_positive_volume_median",
    "post_to_pre_volume_median_ratio",
    "pre_window_observations",
    "post_window_observations",
    "is_candidate_nifty_session",
    "is_final_analysis_session",
    "manual_volume_adjustment_applied",
    "volume_policy",
]

LARGE_RETURN_AUDIT_FIELDS = [
    "ticker",
    "Date",
    "previous_date",
    "adjusted_log_return",
    "absolute_adjusted_log_return",
    "raw_close_log_return",
    "corporate_action_current_or_previous_from_step03",
    "nearest_action_date",
    "nearest_action_types",
    "calendar_days_to_nearest_action",
    "action_within_7_calendar_days",
    "is_final_analysis_session",
    "automatic_deletion_recommended",
    "audit_interpretation",
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


def bool_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return (
        frame[
            column
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
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
        lock_payload.get(
            "state"
        )
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
        raise RuntimeError(
            f"{label} manifest missing: {manifest_path}"
        )

    if not sha_path.exists():
        raise RuntimeError(
            f"{label} manifest SHA sidecar missing: {sha_path}"
        )

    expected = (
        sha_path
        .read_text(
            encoding="utf-8"
        )
        .strip()
        .split()[0]
    )

    actual = (
        sha256_file(
            manifest_path
        )
    )

    if expected != actual:
        raise RuntimeError(
            f"{label} manifest SHA mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )

    return actual


def verify_step01_freeze(
    expected_configuration_fingerprint: str,
) -> Dict[str, Any]:
    manifest_sha = (
        verify_sha_sidecar(
            STEP01_MANIFEST,
            STEP01_MANIFEST_SHA,
            "Step 01",
        )
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
            "Step-01 / Step-00 configuration fingerprint mismatch."
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

    if not STEP01_AUDIT_JSON.exists():
        raise RuntimeError(
            "Step-01 environment audit JSON is missing."
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
            "Current Python executable differs from the Step-01 frozen environment.\n"
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

        if current_version != stored_version:
            package_drift.append(
                (
                    distribution,
                    stored_version,
                    current_version,
                )
            )

    if package_drift:
        detail = "\n".join(
            (
                f"{name}: Step01={stored!r}, current={current!r}"
            )
            for (
                name,
                stored,
                current,
            )
            in package_drift
        )

        raise RuntimeError(
            "Required-package drift detected after Step 01:\n"
            + detail
        )

    return {
        "step01_manifest_sha256": (
            manifest_sha
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


def verify_step02_freeze(
    expected_configuration_fingerprint: str,
    expected_step01_manifest_sha256: str,
) -> Dict[str, Any]:
    manifest_sha = (
        verify_sha_sidecar(
            STEP02_MANIFEST,
            STEP02_MANIFEST_SHA,
            "Step 02",
        )
    )

    with STEP02_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
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
            "Step-02 / Step-00 configuration mismatch."
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
            raise RuntimeError(
                f"Missing Step-02 artifact: {name}"
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
                f"Step-02 artifact hash mismatch: {name}"
            )

    for record in manifest.get(
        "raw_file_records",
        [],
    ):
        path = Path(
            record[
                "file_path"
            ]
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing Step-02 raw file: {path}"
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
                f"Step-02 raw hash mismatch: {path}"
            )

    return {
        "step02_manifest_sha256": (
            manifest_sha
        ),
        "verified_raw_file_count": int(
            len(
                manifest.get(
                    "raw_file_records",
                    [],
                )
            )
        ),
    }


def verify_step03_freeze(
    expected_configuration_fingerprint: str,
    expected_step01_manifest_sha256: str,
    expected_step02_manifest_sha256: str,
) -> Dict[str, Any]:
    manifest_sha = (
        verify_sha_sidecar(
            STEP03_MANIFEST,
            STEP03_MANIFEST_SHA,
            "Step 03",
        )
    )

    with STEP03_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
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
            "Stored Step-03 status is not PASS."
        )

    if (
        manifest.get(
            "configuration_fingerprint_sha256"
        )
        != expected_configuration_fingerprint
    ):
        raise RuntimeError(
            "Step-03 / Step-00 configuration mismatch."
        )

    if (
        manifest.get(
            "step01_manifest_sha256"
        )
        != expected_step01_manifest_sha256
    ):
        raise RuntimeError(
            "Step-03 / Step-01 manifest mismatch."
        )

    if (
        manifest.get(
            "step02_manifest_sha256"
        )
        != expected_step02_manifest_sha256
    ):
        raise RuntimeError(
            "Step-03 / Step-02 manifest mismatch."
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
                f"Missing Step-03 artifact: {name}"
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
                f"Step-03 artifact hash mismatch: {name}"
            )

    required_step03_files = (
        STEP03_SUMMARY,
        STEP03_NIFTY_AUDIT,
        STEP03_REFERENCE_CALENDAR,
        STEP03_CROSS_SECTIONAL_CONSENSUS,
        STEP03_LARGE_RETURN_FLAGS,
        STEP03_CORPORATE_ACTION_AUDIT,
    )

    for path in required_step03_files:
        if not path.exists():
            raise RuntimeError(
                f"Required Step-03 file missing: {path}"
            )

    return {
        "step03_manifest_sha256": (
            manifest_sha
        ),
    }


# =============================================================================
# 8. STEP-04 OUTPUT IMMUTABILITY GUARD
# =============================================================================

def canonical_step04_outputs() -> tuple[Path, ...]:
    return (
        DATE_CLASSIFICATION_CSV,
        CORPORATE_ACTION_REGISTER_CSV,
        SPLIT_AUDIT_CSV,
        LARGE_RETURN_ACTION_AUDIT_CSV,
        EXCLUDED_DATE_REGISTER_CSV,
        FINAL_REFERENCE_CALENDAR_CSV,
        SOURCE_HASH_CSV,
        CLEANING_POLICY_JSON,
        STEP04_MANIFEST_JSON,
        STEP04_MANIFEST_SHA,
        FROZEN_STEP04_SCRIPT,
        STEP04_SUMMARY_TXT,
    )


def assert_new_step04_target() -> None:
    collisions = [
        path
        for path
        in canonical_step04_outputs()
        if path.exists()
    ]

    if collisions:
        detail = "\n".join(
            str(
                path
            )
            for path
            in collisions
        )

        raise RuntimeError(
            "Step-04 canonical output(s) already exist. "
            "Refusing to overwrite a frozen policy.\n"
            f"{detail}\n\n"
            "If Step 04 already completed successfully, run --verify-only."
        )


# =============================================================================
# 9. LOAD RAW / STEP-03 DATA
# =============================================================================

def load_raw_frame(
    ticker: str,
) -> pd.DataFrame:
    path = (
        RAW_PARQUET_DIR
        / f"{safe_ticker_filename(ticker)}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw snapshot: {path}"
        )

    frame = pd.read_parquet(
        path,
        engine="pyarrow",
    )

    if "Date" not in frame.columns:
        raise RuntimeError(
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
        raise RuntimeError(
            f"{ticker}: duplicate dates detected."
        )

    return (
        frame
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )


def load_all_frames() -> Dict[
    str,
    pd.DataFrame,
]:
    return {
        ticker: (
            load_raw_frame(
                ticker
            )
        )
        for ticker
        in (
            list(
                cfg.STOCK_TICKERS
            )
            + [
                cfg.MARKET_INDEX_TICKER
            ]
        )
    }


def load_step03_inputs() -> Dict[str, Any]:
    candidate_calendar = pd.read_csv(
        STEP03_REFERENCE_CALENDAR,
        parse_dates=[
            "Date"
        ],
    )

    consensus = pd.read_csv(
        STEP03_CROSS_SECTIONAL_CONSENSUS,
        parse_dates=[
            "Date"
        ],
    )

    nifty_audit = pd.read_csv(
        STEP03_NIFTY_AUDIT,
        parse_dates=[
            "Date"
        ],
    )

    large_returns = pd.read_csv(
        STEP03_LARGE_RETURN_FLAGS,
        parse_dates=[
            "Date",
            "previous_date",
        ],
    )

    corporate_action_audit = pd.read_csv(
        STEP03_CORPORATE_ACTION_AUDIT,
        parse_dates=[
            "Date"
        ],
    )

    with STEP03_SUMMARY.open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(
            handle
        )

    for frame in (
        candidate_calendar,
        consensus,
        nifty_audit,
        large_returns,
        corporate_action_audit,
    ):
        if "Date" in frame.columns:
            frame[
                "Date"
            ] = pd.to_datetime(
                frame[
                    "Date"
                ],
                errors="raise",
            ).dt.normalize()

    if "previous_date" in large_returns.columns:
        large_returns[
            "previous_date"
        ] = pd.to_datetime(
            large_returns[
                "previous_date"
            ],
            errors="coerce",
        ).dt.normalize()

    return {
        "candidate_calendar": (
            candidate_calendar
        ),
        "consensus": (
            consensus
        ),
        "nifty_audit": (
            nifty_audit
        ),
        "large_returns": (
            large_returns
        ),
        "corporate_action_audit": (
            corporate_action_audit
        ),
        "summary": (
            summary
        ),
    }


# =============================================================================
# 10. SESSION CLASSIFICATION
# =============================================================================

def classify_sessions(
    consensus: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "Date",
        "nifty_raw_row_present",
        "nifty_price_valid",
        "candidate_nifty_session",
        "stocks_present",
        "stocks_nonzero_volume",
        "stocks_zero_volume",
        "stocks_zero_volume_flat_ohlc",
        "stocks_structurally_invalid",
        "date_wide_anomaly_threshold_count",
        "date_wide_anomaly_candidate",
    }

    missing = (
        required_columns
        - set(
            consensus.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Step-03 consensus is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    rows = []

    for _, source in (
        consensus
        .sort_values(
            "Date"
        )
        .iterrows()
    ):
        date = pd.Timestamp(
            source[
                "Date"
            ]
        )

        candidate_nifty = bool(
            source[
                "candidate_nifty_session"
            ]
        )

        rule_candidate = bool(
            source[
                "date_wide_anomaly_candidate"
            ]
        )

        stocks_present = int(
            source[
                "stocks_present"
            ]
        )

        broad_stock_presence = bool(
            stocks_present
            >= DATE_WIDE_THRESHOLD_COUNT
        )

        if (
            candidate_nifty
            and rule_candidate
        ):
            classification = (
                "SYSTEMIC_ZERO_VOLUME_FLAT_OHLC_ON_VALID_NIFTY_SESSION"
            )

            modelling_policy = (
                "EXCLUDE_FROM_ANALYSIS_CALENDAR"
            )

            exclusion_code = (
                "EXCLUDED_DATE_WIDE_ANOMALY"
            )

            rationale = (
                "The date is a price-valid NIFTY reference session, but it "
                "satisfies the predeclared Step-00 panel anomaly rule: at "
                f"least {DATE_WIDE_MIN_STOCK_FRACTION:.0%} of stocks "
                "simultaneously have zero volume and flat OHLC."
            )

        elif candidate_nifty:
            classification = (
                "VALID_REFERENCE_SESSION"
            )

            modelling_policy = (
                "RETAIN"
            )

            exclusion_code = (
                ""
            )

            rationale = (
                "Price-valid NIFTY reference session that does not satisfy "
                "the predeclared date-wide anomaly rule."
            )

        elif (
            rule_candidate
            and broad_stock_presence
        ):
            classification = (
                "PROVIDER_GHOST_OR_NONTRADING_DATE"
            )

            modelling_policy = (
                "EXCLUDE_FROM_COMMON_MODEL_PANEL"
            )

            exclusion_code = (
                "EXCLUDED_NO_VALID_BENCHMARK_AND_DATE_WIDE_ANOMALY"
            )

            rationale = (
                "Broad stock rows exist but there is no valid NIFTY price "
                "context, and the same date independently satisfies the "
                "predeclared date-wide zero-volume + flat-OHLC anomaly rule."
            )

        elif broad_stock_presence:
            classification = (
                "STOCK_SESSION_WITHOUT_VALID_NIFTY_CONTEXT"
            )

            modelling_policy = (
                "EXCLUDE_FROM_COMMON_MODEL_PANEL"
            )

            exclusion_code = (
                "EXCLUDED_NO_VALID_BENCHMARK_CONTEXT"
            )

            rationale = (
                "Broad stock observations are present, but the frozen study "
                "requires contemporaneous valid NIFTY context and this date "
                "is not in the price-valid NIFTY candidate calendar."
            )

        else:
            classification = (
                "NONREFERENCE_OTHER"
            )

            modelling_policy = (
                "EXCLUDE_FROM_COMMON_MODEL_PANEL"
            )

            exclusion_code = (
                "EXCLUDED_NONREFERENCE_OTHER"
            )

            rationale = (
                "Date is outside the price-valid NIFTY reference calendar "
                "and does not meet the broad-panel-presence criterion."
            )

        rows.append(
            {
                "Date": date,
                "day_of_week": (
                    date.day_name()
                ),
                "nifty_raw_row_present": bool(
                    source[
                        "nifty_raw_row_present"
                    ]
                ),
                "nifty_price_valid": bool(
                    source[
                        "nifty_price_valid"
                    ]
                ),
                "candidate_nifty_session": (
                    candidate_nifty
                ),
                "stocks_present": (
                    stocks_present
                ),
                "stocks_nonzero_volume": int(
                    source[
                        "stocks_nonzero_volume"
                    ]
                ),
                "stocks_zero_volume": int(
                    source[
                        "stocks_zero_volume"
                    ]
                ),
                "stocks_zero_volume_flat_ohlc": int(
                    source[
                        "stocks_zero_volume_flat_ohlc"
                    ]
                ),
                "stocks_structurally_invalid": int(
                    source[
                        "stocks_structurally_invalid"
                    ]
                ),
                "date_wide_anomaly_threshold_count": int(
                    source[
                        "date_wide_anomaly_threshold_count"
                    ]
                ),
                "date_wide_anomaly_candidate": (
                    rule_candidate
                ),
                "broad_stock_presence": (
                    broad_stock_presence
                ),
                "classification": (
                    classification
                ),
                "modelling_policy": (
                    modelling_policy
                ),
                "primary_exclusion_code": (
                    exclusion_code
                ),
                "rationale": (
                    rationale
                ),
            }
        )

    result = rows_to_dataframe(
        rows,
        DATE_CLASSIFICATION_FIELDS,
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
# 11. EXCLUSION REGISTER / FINAL CALENDAR
# =============================================================================

def build_exclusion_register(
    classification: pd.DataFrame,
) -> pd.DataFrame:
    excluded = (
        classification.loc[
            (
                classification[
                    "modelling_policy"
                ]
                != "RETAIN"
            )
            &
            (
                classification[
                    "stocks_present"
                ]
                > 0
            )
        ]
        .copy()
    )

    if excluded.empty:
        return pd.DataFrame(
            columns=EXCLUSION_FIELDS
        )

    excluded[
        "reason"
    ] = (
        excluded[
            "rationale"
        ]
    )

    return (
        excluded[
            EXCLUSION_FIELDS
        ]
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )


def build_final_analysis_calendar(
    candidate_calendar: pd.DataFrame,
    classification: pd.DataFrame,
) -> pd.DataFrame:
    candidate_dates = pd.DatetimeIndex(
        pd.to_datetime(
            candidate_calendar[
                "Date"
            ],
            errors="raise",
        )
        .dt.normalize()
        .sort_values()
        .unique()
    )

    valid_anomaly_dates = pd.DatetimeIndex(
        classification.loc[
            (
                classification[
                    "candidate_nifty_session"
                ]
                .astype(
                    bool
                )
            )
            &
            (
                classification[
                    "date_wide_anomaly_candidate"
                ]
                .astype(
                    bool
                )
            ),
            "Date",
        ]
        .sort_values()
        .unique()
    )

    final_dates = (
        candidate_dates
        .difference(
            valid_anomaly_dates,
            sort=True,
        )
    )

    final = pd.DataFrame(
        {
            "session_number": np.arange(
                1,
                len(
                    final_dates
                ) + 1,
                dtype=np.int64,
            ),
            "Date": (
                final_dates
            ),
        }
    )

    final[
        "calendar_year"
    ] = (
        final[
            "Date"
        ]
        .dt.year
    )

    final[
        "calendar_month"
    ] = (
        final[
            "Date"
        ]
        .dt.month
    )

    final[
        "day_of_week"
    ] = (
        final[
            "Date"
        ]
        .dt.day_name()
    )

    return final


def validate_final_calendar(
    candidate_calendar: pd.DataFrame,
    final_calendar: pd.DataFrame,
    classification: pd.DataFrame,
    stock_frames: Dict[
        str,
        pd.DataFrame,
    ],
) -> Dict[str, Any]:
    candidate_dates = set(
        pd.to_datetime(
            candidate_calendar[
                "Date"
            ]
        )
        .dt.normalize()
        .tolist()
    )

    final_dates = set(
        pd.to_datetime(
            final_calendar[
                "Date"
            ]
        )
        .dt.normalize()
        .tolist()
    )

    anomaly_dates = set(
        pd.to_datetime(
            classification.loc[
                (
                    classification[
                        "candidate_nifty_session"
                    ]
                    .astype(
                        bool
                    )
                )
                &
                (
                    classification[
                        "date_wide_anomaly_candidate"
                    ]
                    .astype(
                        bool
                    )
                ),
                "Date",
            ]
        )
        .dt.normalize()
        .tolist()
    )

    expected_final = (
        candidate_dates
        - anomaly_dates
    )

    if final_dates != expected_final:
        raise RuntimeError(
            "Final analysis calendar does not equal candidate NIFTY calendar "
            "minus predeclared valid-session date-wide anomalies."
        )

    if (
        final_calendar[
            "Date"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate dates in final analysis calendar."
        )

    if not (
        final_calendar[
            "Date"
        ]
        .is_monotonic_increasing
    ):
        raise RuntimeError(
            "Final analysis calendar is not chronological."
        )

    missing_by_stock = {}

    for (
        ticker,
        frame,
    ) in stock_frames.items():
        stock_dates = set(
            pd.to_datetime(
                frame[
                    "Date"
                ]
            )
            .dt.normalize()
            .tolist()
        )

        missing = sorted(
            final_dates
            - stock_dates
        )

        if missing:
            missing_by_stock[
                ticker
            ] = [
                date.strftime(
                    "%Y-%m-%d"
                )
                for date
                in missing
            ]

    if missing_by_stock:
        raise RuntimeError(
            "One or more stocks are missing final-calendar dates:\n"
            + json.dumps(
                missing_by_stock,
                indent=2,
            )
        )

    return {
        "candidate_session_count": int(
            len(
                candidate_dates
            )
        ),
        "valid_reference_anomaly_count": int(
            len(
                anomaly_dates
            )
        ),
        "final_session_count": int(
            len(
                final_dates
            )
        ),
        "all_20_stocks_present_on_final_calendar": True,
        "candidate_minus_anomaly_equals_final": True,
    }


# =============================================================================
# 12. CORPORATE-ACTION REGISTER
# =============================================================================

def numeric_action_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            0.0,
            index=frame.index,
            dtype="float64",
        )

    return (
        pd.to_numeric(
            frame[
                column
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
    )


def build_corporate_action_register(
    stock_frames: Dict[
        str,
        pd.DataFrame,
    ],
    candidate_dates: set[pd.Timestamp],
    final_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    rows = []

    for ticker in (
        cfg.STOCK_TICKERS
    ):
        frame = (
            stock_frames[
                ticker
            ]
            .copy()
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        dividends = (
            numeric_action_series(
                frame,
                "Dividends",
            )
        )

        splits = (
            numeric_action_series(
                frame,
                "Stock Splits",
            )
        )

        capital_gains = (
            numeric_action_series(
                frame,
                "Capital Gains",
            )
        )

        close = pd.to_numeric(
            frame[
                "Close"
            ],
            errors="coerce",
        )

        adjusted = pd.to_numeric(
            frame[
                "Adj Close"
            ],
            errors="coerce",
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

        for index in range(
            len(
                frame
            )
        ):
            dividend = float(
                dividends.iloc[
                    index
                ]
            )

            split = float(
                splits.iloc[
                    index
                ]
            )

            capital = float(
                capital_gains.iloc[
                    index
                ]
            )

            action_types = []

            if dividend != 0.0:
                action_types.append(
                    "dividend"
                )

            if split != 0.0:
                action_types.append(
                    "stock_split"
                )

            if capital != 0.0:
                action_types.append(
                    "capital_gain"
                )

            if not action_types:
                continue

            date = pd.Timestamp(
                frame.loc[
                    index,
                    "Date",
                ]
            )

            rows.append(
                {
                    "ticker": ticker,
                    "Date": date,
                    "Dividends": (
                        dividend
                    ),
                    "Stock Splits": (
                        split
                    ),
                    "Capital Gains": (
                        capital
                    ),
                    "action_types": (
                        "|".join(
                            action_types
                        )
                    ),
                    "action_event_count_on_row": int(
                        len(
                            action_types
                        )
                    ),
                    "Open": (
                        frame.loc[
                            index,
                            "Open",
                        ]
                    ),
                    "High": (
                        frame.loc[
                            index,
                            "High",
                        ]
                    ),
                    "Low": (
                        frame.loc[
                            index,
                            "Low",
                        ]
                    ),
                    "Close": (
                        frame.loc[
                            index,
                            "Close",
                        ]
                    ),
                    "Adj Close": (
                        frame.loc[
                            index,
                            "Adj Close",
                        ]
                    ),
                    "Volume": (
                        frame.loc[
                            index,
                            "Volume",
                        ]
                    ),
                    "adjustment_factor": (
                        factor[
                            index
                        ]
                    ),
                    "is_candidate_nifty_session": bool(
                        date
                        in candidate_dates
                    ),
                    "is_final_analysis_session": bool(
                        date
                        in final_dates
                    ),
                }
            )

    result = rows_to_dataframe(
        rows,
        CORPORATE_ACTION_FIELDS,
    )

    if not result.empty:
        result[
            "Date"
        ] = pd.to_datetime(
            result[
                "Date"
            ]
        )

        result = (
            result
            .sort_values(
                [
                    "Date",
                    "ticker",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    return result


# =============================================================================
# 13. SPLIT / RAW-VOLUME AUDIT
# =============================================================================

def positive_median(
    series: pd.Series,
) -> float:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    positive = (
        numeric.loc[
            numeric
            > 0
        ]
    )

    if positive.empty:
        return np.nan

    return float(
        positive.median()
    )


def build_split_volume_audit(
    stock_frames: Dict[
        str,
        pd.DataFrame,
    ],
    candidate_dates: set[pd.Timestamp],
    final_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    rows = []

    for ticker in (
        cfg.STOCK_TICKERS
    ):
        frame = (
            stock_frames[
                ticker
            ]
            .copy()
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        splits = (
            numeric_action_series(
                frame,
                "Stock Splits",
            )
        )

        split_indices = np.flatnonzero(
            splits.to_numpy()
            != 0.0
        )

        for index in split_indices:
            date = pd.Timestamp(
                frame.loc[
                    index,
                    "Date",
                ]
            )

            previous_index = (
                index - 1
                if index > 0
                else None
            )

            next_index = (
                index + 1
                if index + 1
                < len(
                    frame
                )
                else None
            )

            pre_start = max(
                0,
                index
                - SPLIT_VOLUME_WINDOW_SESSIONS,
            )

            post_end = min(
                len(
                    frame
                ),
                index
                + 1
                + SPLIT_VOLUME_WINDOW_SESSIONS,
            )

            pre_window = (
                frame.iloc[
                    pre_start:index
                ]
            )

            post_window = (
                frame.iloc[
                    index + 1:post_end
                ]
            )

            pre_median = (
                positive_median(
                    pre_window[
                        "Volume"
                    ]
                )
            )

            post_median = (
                positive_median(
                    post_window[
                        "Volume"
                    ]
                )
            )

            ratio = (
                float(
                    post_median
                    / pre_median
                )
                if (
                    np.isfinite(
                        pre_median
                    )
                    and
                    np.isfinite(
                        post_median
                    )
                    and
                    pre_median
                    > 0
                )
                else np.nan
            )

            def value_at(
                row_index: Optional[int],
                column: str,
            ) -> Any:
                if row_index is None:
                    return np.nan

                return frame.loc[
                    row_index,
                    column,
                ]

            def date_at(
                row_index: Optional[int],
            ) -> Any:
                if row_index is None:
                    return pd.NaT

                return pd.Timestamp(
                    frame.loc[
                        row_index,
                        "Date",
                    ]
                )

            rows.append(
                {
                    "ticker": ticker,
                    "Date": date,
                    "split_ratio": float(
                        splits.iloc[
                            index
                        ]
                    ),
                    "previous_stock_date": (
                        date_at(
                            previous_index
                        )
                    ),
                    "next_stock_date": (
                        date_at(
                            next_index
                        )
                    ),
                    "previous_close": (
                        value_at(
                            previous_index,
                            "Close",
                        )
                    ),
                    "split_date_close": (
                        value_at(
                            index,
                            "Close",
                        )
                    ),
                    "next_close": (
                        value_at(
                            next_index,
                            "Close",
                        )
                    ),
                    "previous_adj_close": (
                        value_at(
                            previous_index,
                            "Adj Close",
                        )
                    ),
                    "split_date_adj_close": (
                        value_at(
                            index,
                            "Adj Close",
                        )
                    ),
                    "next_adj_close": (
                        value_at(
                            next_index,
                            "Adj Close",
                        )
                    ),
                    "previous_volume": (
                        value_at(
                            previous_index,
                            "Volume",
                        )
                    ),
                    "split_date_volume": (
                        value_at(
                            index,
                            "Volume",
                        )
                    ),
                    "next_volume": (
                        value_at(
                            next_index,
                            "Volume",
                        )
                    ),
                    "pre_split_positive_volume_median": (
                        pre_median
                    ),
                    "post_split_positive_volume_median": (
                        post_median
                    ),
                    "post_to_pre_volume_median_ratio": (
                        ratio
                    ),
                    "pre_window_observations": int(
                        len(
                            pre_window
                        )
                    ),
                    "post_window_observations": int(
                        len(
                            post_window
                        )
                    ),
                    "is_candidate_nifty_session": bool(
                        date
                        in candidate_dates
                    ),
                    "is_final_analysis_session": bool(
                        date
                        in final_dates
                    ),
                    "manual_volume_adjustment_applied": False,
                    "volume_policy": (
                        "retain_provider_volume_as_supplied_no_manual_split_adjustment"
                    ),
                }
            )

    result = rows_to_dataframe(
        rows,
        SPLIT_AUDIT_FIELDS,
    )

    if not result.empty:
        for column in (
            "Date",
            "previous_stock_date",
            "next_stock_date",
        ):
            result[
                column
            ] = pd.to_datetime(
                result[
                    column
                ],
                errors="coerce",
            )

        result = (
            result
            .sort_values(
                [
                    "Date",
                    "ticker",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    return result


# =============================================================================
# 14. LARGE RETURN / CORPORATE-ACTION AUDIT
# =============================================================================

def audit_large_returns_against_actions(
    large_returns: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    final_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    rows = []

    if large_returns.empty:
        return pd.DataFrame(
            columns=LARGE_RETURN_AUDIT_FIELDS
        )

    actions = (
        corporate_actions.copy()
    )

    if not actions.empty:
        actions[
            "Date"
        ] = pd.to_datetime(
            actions[
                "Date"
            ]
        )

    for _, flag in (
        large_returns
        .iterrows()
    ):
        ticker = str(
            flag[
                "ticker"
            ]
        )

        date = pd.Timestamp(
            flag[
                "Date"
            ]
        )

        same_stock_actions = (
            actions.loc[
                actions[
                    "ticker"
                ]
                == ticker
            ]
            .copy()
            if not actions.empty
            else pd.DataFrame(
                columns=actions.columns
            )
        )

        nearest_date = pd.NaT
        nearest_types = ""
        nearest_days = np.nan

        if not same_stock_actions.empty:
            distances = (
                same_stock_actions[
                    "Date"
                ]
                .sub(
                    date
                )
                .dt.days
                .abs()
            )

            nearest_index = (
                distances.idxmin()
            )

            nearest_date = pd.Timestamp(
                same_stock_actions.loc[
                    nearest_index,
                    "Date",
                ]
            )

            nearest_types = str(
                same_stock_actions.loc[
                    nearest_index,
                    "action_types",
                ]
            )

            nearest_days = int(
                distances.loc[
                    nearest_index
                ]
            )

        within_window = bool(
            pd.notna(
                nearest_days
            )
            and
            int(
                nearest_days
            )
            <= CORPORATE_ACTION_PROXIMITY_CALENDAR_DAYS
        )

        rows.append(
            {
                "ticker": ticker,
                "Date": date,
                "previous_date": (
                    flag.get(
                        "previous_date",
                        pd.NaT,
                    )
                ),
                "adjusted_log_return": float(
                    flag[
                        "adjusted_log_return"
                    ]
                ),
                "absolute_adjusted_log_return": float(
                    flag[
                        "absolute_adjusted_log_return"
                    ]
                ),
                "raw_close_log_return": (
                    flag.get(
                        "raw_close_log_return",
                        np.nan,
                    )
                ),
                "corporate_action_current_or_previous_from_step03": bool(
                    flag.get(
                        "corporate_action_current_or_previous",
                        False,
                    )
                ),
                "nearest_action_date": (
                    nearest_date
                ),
                "nearest_action_types": (
                    nearest_types
                ),
                "calendar_days_to_nearest_action": (
                    nearest_days
                ),
                "action_within_7_calendar_days": (
                    within_window
                ),
                "is_final_analysis_session": bool(
                    date
                    in final_dates
                ),
                "automatic_deletion_recommended": False,
                "audit_interpretation": (
                    "Retain unless independent evidence establishes a provider "
                    "or data error. Extreme adjusted returns are not deleted "
                    "solely because they are extreme."
                ),
            }
        )

    result = rows_to_dataframe(
        rows,
        LARGE_RETURN_AUDIT_FIELDS,
    )

    if not result.empty:
        for column in (
            "Date",
            "previous_date",
            "nearest_action_date",
        ):
            result[
                column
            ] = pd.to_datetime(
                result[
                    column
                ],
                errors="coerce",
            )

        result = (
            result
            .sort_values(
                [
                    "Date",
                    "ticker",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    return result


# =============================================================================
# 15. SOURCE HASH AUDIT
# =============================================================================

def find_artifact_path_by_name(
    manifest: Dict[str, Any],
    target_name: str,
) -> Optional[Path]:
    for record in (
        manifest
        .get(
            "artifacts",
            {},
        )
        .values()
    ):
        path = Path(
            record[
                "path"
            ]
        )

        if path.name == target_name:
            return path

    return None


def collect_source_hashes() -> pd.DataFrame:
    with STEP01_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        step01_manifest = json.load(
            handle
        )

    with STEP02_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        step02_manifest = json.load(
            handle
        )

    with STEP03_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        step03_manifest = json.load(
            handle
        )

    current_step00 = (
        Path(
            cfg.__file__
        ).resolve()
    )

    frozen_step00 = (
        cfg.get_frozen_script_path()
        .resolve()
    )

    frozen_step01 = (
        find_artifact_path_by_name(
            step01_manifest,
            "step01_environment_audit_frozen.py",
        )
    )

    step02_frozen_record = (
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
            step02_frozen_record[
                "path"
            ]
        )
        if step02_frozen_record
        else None
    )

    frozen_step03 = (
        find_artifact_path_by_name(
            step03_manifest,
            "step03_market_integrity_audit_frozen.py",
        )
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
            "current_step04",
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

    if frozen_step03 is not None:
        source_paths.append(
            (
                "frozen_step03",
                frozen_step03,
            )
        )

    rows = []

    for (
        role,
        path,
    ) in source_paths:
        if not path.exists():
            raise RuntimeError(
                f"Required source file missing: {path}"
            )

        rows.append(
            {
                "role": (
                    role
                ),
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

    if current_step00_sha != frozen_step00_sha:
        raise RuntimeError(
            "Active step00_config.py differs from the frozen Step-00 source."
        )

    return rows_to_dataframe(
        rows,
        SOURCE_HASH_FIELDS,
    )


# =============================================================================
# 16. CLEANING POLICY
# =============================================================================

def build_cleaning_policy(
    *,
    step00_info: Dict[str, Any],
    step01_info: Dict[str, Any],
    step02_info: Dict[str, Any],
    step03_info: Dict[str, Any],
    classification: pd.DataFrame,
    exclusion_register: pd.DataFrame,
    candidate_calendar: pd.DataFrame,
    final_calendar: pd.DataFrame,
    calendar_validation: Dict[str, Any],
    corporate_actions: pd.DataFrame,
    split_audit: pd.DataFrame,
    large_return_audit: pd.DataFrame,
    source_hashes: pd.DataFrame,
    lock_sha_after: str,
) -> Dict[str, Any]:
    candidate_valid_anomaly_dates = (
        classification.loc[
            (
                bool_series(
                    classification,
                    "candidate_nifty_session",
                )
            )
            &
            (
                bool_series(
                    classification,
                    "date_wide_anomaly_candidate",
                )
            ),
            "Date",
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .tolist()
    )

    no_benchmark_dates = (
        exclusion_register.loc[
            exclusion_register[
                "primary_exclusion_code"
            ]
            ==
            "EXCLUDED_NO_VALID_BENCHMARK_CONTEXT",
            "Date",
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .tolist()
    )

    ghost_dates = (
        exclusion_register.loc[
            exclusion_register[
                "primary_exclusion_code"
            ]
            ==
            "EXCLUDED_NO_VALID_BENCHMARK_AND_DATE_WIDE_ANOMALY",
            "Date",
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .tolist()
    )

    dividend_rows = int(
        (
            pd.to_numeric(
                corporate_actions[
                    "Dividends"
                ],
                errors="coerce",
            )
            .fillna(
                0.0
            )
            != 0
        )
        .sum()
    ) if not corporate_actions.empty else 0

    split_rows = int(
        (
            pd.to_numeric(
                corporate_actions[
                    "Stock Splits"
                ],
                errors="coerce",
            )
            .fillna(
                0.0
            )
            != 0
        )
        .sum()
    ) if not corporate_actions.empty else 0

    capital_gain_rows = int(
        (
            pd.to_numeric(
                corporate_actions[
                    "Capital Gains"
                ],
                errors="coerce",
            )
            .fillna(
                0.0
            )
            != 0
        )
        .sum()
    ) if not corporate_actions.empty else 0

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
            "step03_manifest_sha256": (
                step03_info[
                    "step03_manifest_sha256"
                ]
            ),
        },
        "reference_calendar_policy": {
            "frozen_step00_policy": (
                cfg.MARKET_CALENDAR_POLICY
            ),
            "base_calendar": (
                "Step-03 price-valid Yahoo ^NSEI candidate calendar"
            ),
            "nifty_volume_required_for_session_validity": False,
            "candidate_session_count": int(
                len(
                    candidate_calendar
                )
            ),
            "stock_only_dates_without_valid_nifty_context_used": False,
        },
        "date_wide_anomaly_policy": {
            "enabled": (
                DATE_WIDE_RULE_ENABLED
            ),
            "frozen_definition": (
                cfg.DATE_WIDE_ANOMALY_DEFINITION
            ),
            "minimum_stock_fraction": (
                DATE_WIDE_MIN_STOCK_FRACTION
            ),
            "threshold_stock_count": (
                DATE_WIDE_THRESHOLD_COUNT
            ),
            "valid_nifty_sessions_excluded_by_rule": (
                candidate_valid_anomaly_dates
            ),
            "valid_nifty_session_exclusion_count": int(
                len(
                    candidate_valid_anomaly_dates
                )
            ),
            "rule_not_redefined_after_data_inspection": True,
        },
        "nonreference_date_policy": {
            "stock_sessions_without_valid_nifty_context": (
                no_benchmark_dates
            ),
            "provider_ghost_or_nontrading_dates": (
                ghost_dates
            ),
            "policy": (
                "Exclude from the common modelling panel because the frozen "
                "study requires a valid contemporaneous NIFTY context."
            ),
        },
        "final_analysis_calendar": {
            "session_count": int(
                len(
                    final_calendar
                )
            ),
            "first_session": (
                final_calendar[
                    "Date"
                ]
                .min()
                .strftime(
                    "%Y-%m-%d"
                )
            ),
            "last_session": (
                final_calendar[
                    "Date"
                ]
                .max()
                .strftime(
                    "%Y-%m-%d"
                )
            ),
            "all_20_stocks_present": (
                calendar_validation[
                    "all_20_stocks_present_on_final_calendar"
                ]
            ),
            "candidate_minus_valid_anomalies_equals_final": (
                calendar_validation[
                    "candidate_minus_anomaly_equals_final"
                ]
            ),
        },
        "corporate_action_policy": {
            "frozen_step00_policy": (
                cfg.CORPORATE_ACTION_POLICY
            ),
            "corporate_action_date_rows": int(
                len(
                    corporate_actions
                )
            ),
            "dividend_date_rows": (
                dividend_rows
            ),
            "split_date_rows": (
                split_rows
            ),
            "capital_gain_date_rows": (
                capital_gain_rows
            ),
            "automatic_event_date_deletion": False,
            "canonical_adjusted_price_policy_for_step05": (
                "factor = Adj Close / Close; adjusted Open/High/Low/Close "
                "= raw field * factor; Adj Close is canonical adjusted close"
            ),
            "volume_policy_for_step05": (
                "retain provider Volume as supplied; do not impute; do not "
                "apply a manual split adjustment"
            ),
            "reason_for_no_manual_volume_adjustment": (
                "Step 00 froze an audit requirement, not a manual volume "
                "transformation. Step 04 records split behavior explicitly "
                "and avoids an unsupported second adjustment."
            ),
        },
        "extreme_return_policy": {
            "flag_count": int(
                len(
                    large_return_audit
                )
            ),
            "automatic_deletion": False,
            "rule": (
                "Retain extreme adjusted returns unless independent evidence "
                "establishes a provider or data error."
            ),
        },
        "scientific_boundary": {
            "raw_data_overwritten": False,
            "cleaned_panel_materialized": False,
            "cleaning_materialization_stage": (
                "Step 05"
            ),
            "price_imputation": False,
            "volume_imputation": False,
            "feature_imputation": False,
            "target_imputation": False,
            "model_performance_used_to_make_cleaning_decisions": False,
            "final_test_model_scoring_performed": False,
        },
        "source_code_hashes": (
            source_hashes
            .to_dict(
                orient="records"
            )
        ),
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
            "lock_unchanged": bool(
                lock_sha_after
                == step00_info[
                    "final_test_lock_sha256"
                ]
            ),
        },
    }


# =============================================================================
# 17. STEP-04 MANIFEST
# =============================================================================

def step04_artifact_paths() -> tuple[Path, ...]:
    return (
        DATE_CLASSIFICATION_CSV,
        CORPORATE_ACTION_REGISTER_CSV,
        SPLIT_AUDIT_CSV,
        LARGE_RETURN_ACTION_AUDIT_CSV,
        EXCLUDED_DATE_REGISTER_CSV,
        FINAL_REFERENCE_CALENDAR_CSV,
        SOURCE_HASH_CSV,
        CLEANING_POLICY_JSON,
        FROZEN_STEP04_SCRIPT,
        STEP04_SUMMARY_TXT,
    )


def write_step04_manifest(
    *,
    step00_info: Dict[str, Any],
    step01_info: Dict[str, Any],
    step02_info: Dict[str, Any],
    step03_info: Dict[str, Any],
) -> str:
    artifacts = {
        path.name: (
            artifact_record(
                path
            )
        )
        for path
        in step04_artifact_paths()
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
        "step03_manifest_sha256": (
            step03_info[
                "step03_manifest_sha256"
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
        STEP04_MANIFEST_JSON,
        manifest,
    )

    manifest_sha = (
        sha256_file(
            STEP04_MANIFEST_JSON
        )
    )

    atomic_write_text(
        STEP04_MANIFEST_SHA,
        (
            f"{manifest_sha}  "
            f"{STEP04_MANIFEST_JSON.name}\n"
        ),
    )

    return manifest_sha


# =============================================================================
# 18. VERIFY EXISTING STEP 04
# =============================================================================

def verify_existing_step04() -> int:
    print(
        "=" * 132
    )
    print(
        "STEP 04 — VERIFY EXISTING SESSION / CLEANING POLICY FREEZE"
    )
    print(
        "=" * 132
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

    step03_info = (
        verify_step03_freeze(
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
            expected_step02_manifest_sha256=(
                step02_info[
                    "step02_manifest_sha256"
                ]
            ),
        )
    )

    if not STEP04_MANIFEST_JSON.exists():
        print(
            "[FAIL] Step-04 manifest is missing."
        )
        return 20

    if not STEP04_MANIFEST_SHA.exists():
        print(
            "[FAIL] Step-04 manifest SHA sidecar is missing."
        )
        return 21

    actual_manifest_sha = (
        verify_sha_sidecar(
            STEP04_MANIFEST_JSON,
            STEP04_MANIFEST_SHA,
            "Step 04",
        )
    )

    with STEP04_MANIFEST_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
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
        print(
            "[FAIL] Stored Step-04 status is not PASS."
        )
        return 22

    expected_chain = {
        "configuration_fingerprint_sha256": (
            step00_info[
                "configuration_fingerprint_sha256"
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
        "step03_manifest_sha256": (
            step03_info[
                "step03_manifest_sha256"
            ]
        ),
    }

    for (
        field,
        expected,
    ) in expected_chain.items():
        if (
            manifest.get(
                field
            )
            != expected
        ):
            print(
                f"[FAIL] Step-04 chain mismatch: {field}"
            )
            return 23

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
            "=" * 132
        )
        print(
            "STEP 04 VERIFICATION STATUS: FAIL"
        )
        print(
            "=" * 132
        )
        return 24

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
        f"{step03_info['step03_manifest_sha256']}"
    )
    print(
        f"[PASS] Step-04 manifest    : "
        f"{actual_manifest_sha}"
    )
    print(
        f"[PASS] Final-test state     : "
        f"{step00_info['final_test_state']}"
    )
    print(
        "=" * 132
    )
    print(
        "STEP 04 VERIFICATION STATUS: PASS"
    )
    print(
        "=" * 132
    )

    return 0


# =============================================================================
# 19. FAILURE DIAGNOSTIC
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
            "04_session_policy_FAILED_"
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
        "canonical_step04_manifest_exists": bool(
            STEP04_MANIFEST_JSON.exists()
        ),
    }

    atomic_write_json(
        path,
        payload,
    )

    return path


# =============================================================================
# 20. MAIN POLICY FREEZE
# =============================================================================

def run_new_policy() -> int:
    print(
        "=" * 132
    )
    print(
        STEP_NAME
    )
    print(
        "=" * 132
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
            "[2/10] Verifying Step 01 environment..."
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
            "[3/10] Verifying Step 02 acquisition..."
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
            "       Frozen raw files: "
            f"{step02_info['verified_raw_file_count']}"
        )

        print(
            "[4/10] Verifying Step 03 integrity audit..."
        )

        step03_info = (
            verify_step03_freeze(
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
                expected_step02_manifest_sha256=(
                    step02_info[
                        "step02_manifest_sha256"
                    ]
                ),
            )
        )

        print(
            "       Step-03 manifest: "
            f"{step03_info['step03_manifest_sha256']}"
        )

        stage = (
            "output_collision_guard"
        )

        print(
            "[5/10] Checking immutable Step-04 output target..."
        )

        assert_new_step04_target()

        cfg.create_directories()

        stage = (
            "load_inputs"
        )

        print(
            "[6/10] Loading Step-03 diagnostics and frozen raw stock snapshots..."
        )

        step03_inputs = (
            load_step03_inputs()
        )

        frames = (
            load_all_frames()
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

        candidate_calendar = (
            step03_inputs[
                "candidate_calendar"
            ]
        )

        consensus = (
            step03_inputs[
                "consensus"
            ]
        )

        large_returns = (
            step03_inputs[
                "large_returns"
            ]
        )

        step03_corporate_action_audit = (
            step03_inputs[
                "corporate_action_audit"
            ]
        )

        print(
            "       Candidate NIFTY sessions: "
            f"{len(candidate_calendar):,}"
        )
        print(
            "       Cross-sectional dates: "
            f"{len(consensus):,}"
        )
        print(
            "       Step-03 large-return flags: "
            f"{len(large_returns):,}"
        )

        stage = (
            "session_policy"
        )

        print(
            "[7/10] Classifying source dates and freezing the final analysis calendar..."
        )

        classification = (
            classify_sessions(
                consensus
            )
        )

        exclusion_register = (
            build_exclusion_register(
                classification
            )
        )

        final_calendar = (
            build_final_analysis_calendar(
                candidate_calendar=(
                    candidate_calendar
                ),
                classification=(
                    classification
                ),
            )
        )

        calendar_validation = (
            validate_final_calendar(
                candidate_calendar=(
                    candidate_calendar
                ),
                final_calendar=(
                    final_calendar
                ),
                classification=(
                    classification
                ),
                stock_frames=(
                    stock_frames
                ),
            )
        )

        candidate_dates = set(
            pd.to_datetime(
                candidate_calendar[
                    "Date"
                ]
            )
            .dt.normalize()
            .tolist()
        )

        final_dates = set(
            pd.to_datetime(
                final_calendar[
                    "Date"
                ]
            )
            .dt.normalize()
            .tolist()
        )

        valid_reference_anomalies = (
            classification.loc[
                (
                    bool_series(
                        classification,
                        "candidate_nifty_session",
                    )
                )
                &
                (
                    bool_series(
                        classification,
                        "date_wide_anomaly_candidate",
                    )
                )
            ]
            .copy()
        )

        print(
            "       Frozen date-wide rule: "
            f">= {DATE_WIDE_MIN_STOCK_FRACTION:.0%} "
            "zero-volume + flat-OHLC stocks"
        )
        print(
            "       Threshold count: "
            f"{DATE_WIDE_THRESHOLD_COUNT}/{len(cfg.STOCK_TICKERS)}"
        )
        print(
            "       Valid NIFTY sessions excluded by rule: "
            f"{len(valid_reference_anomalies)}"
        )
        print(
            "       Final retained sessions: "
            f"{len(final_calendar):,}"
        )

        stage = (
            "corporate_action_policy"
        )

        print(
            "[8/10] Auditing corporate actions, stock splits, volume and large returns..."
        )

        corporate_actions = (
            build_corporate_action_register(
                stock_frames=(
                    stock_frames
                ),
                candidate_dates=(
                    candidate_dates
                ),
                final_dates=(
                    final_dates
                ),
            )
        )

        split_audit = (
            build_split_volume_audit(
                stock_frames=(
                    stock_frames
                ),
                candidate_dates=(
                    candidate_dates
                ),
                final_dates=(
                    final_dates
                ),
            )
        )

        large_return_audit = (
            audit_large_returns_against_actions(
                large_returns=(
                    large_returns
                ),
                corporate_actions=(
                    corporate_actions
                ),
                final_dates=(
                    final_dates
                ),
            )
        )

        # Step 03 reported event-date rows. Verify the action register reproduces
        # that count exactly.
        step03_action_event_count = int(
            step03_corporate_action_audit.loc[
                bool_series(
                    step03_corporate_action_audit,
                    "corporate_action_event",
                )
            ]
            .shape[
                0
            ]
        )

        if (
            len(
                corporate_actions
            )
            != step03_action_event_count
        ):
            raise RuntimeError(
                "Corporate-action event-row count does not reproduce Step 03.\n"
                f"Step 03: {step03_action_event_count}\n"
                f"Step 04: {len(corporate_actions)}"
            )

        print(
            "       Corporate-action date rows: "
            f"{len(corporate_actions):,}"
        )
        print(
            "       Stock-split date rows: "
            f"{len(split_audit):,}"
        )
        print(
            "       Large adjusted-return flags retained for audit: "
            f"{len(large_return_audit):,}"
        )
        print(
            "       Manual volume split adjustment: NONE"
        )

        stage = (
            "freeze_outputs"
        )

        print(
            "[9/10] Writing policy artifacts, hashes and immutable manifest..."
        )

        source_hashes = (
            collect_source_hashes()
        )

        write_dataframe_csv(
            DATE_CLASSIFICATION_CSV,
            classification,
            DATE_CLASSIFICATION_FIELDS,
        )

        write_dataframe_csv(
            CORPORATE_ACTION_REGISTER_CSV,
            corporate_actions,
            CORPORATE_ACTION_FIELDS,
        )

        write_dataframe_csv(
            SPLIT_AUDIT_CSV,
            split_audit,
            SPLIT_AUDIT_FIELDS,
        )

        write_dataframe_csv(
            LARGE_RETURN_ACTION_AUDIT_CSV,
            large_return_audit,
            LARGE_RETURN_AUDIT_FIELDS,
        )

        write_dataframe_csv(
            EXCLUDED_DATE_REGISTER_CSV,
            exclusion_register,
            EXCLUSION_FIELDS,
        )

        write_dataframe_csv(
            FINAL_REFERENCE_CALENDAR_CSV,
            final_calendar,
            FINAL_CALENDAR_FIELDS,
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
                "Final-test lock changed during Step 04."
            )

        cleaning_policy = (
            build_cleaning_policy(
                step00_info=(
                    step00_info
                ),
                step01_info=(
                    step01_info
                ),
                step02_info=(
                    step02_info
                ),
                step03_info=(
                    step03_info
                ),
                classification=(
                    classification
                ),
                exclusion_register=(
                    exclusion_register
                ),
                candidate_calendar=(
                    candidate_calendar
                ),
                final_calendar=(
                    final_calendar
                ),
                calendar_validation=(
                    calendar_validation
                ),
                corporate_actions=(
                    corporate_actions
                ),
                split_audit=(
                    split_audit
                ),
                large_return_audit=(
                    large_return_audit
                ),
                source_hashes=(
                    source_hashes
                ),
                lock_sha_after=(
                    lock_sha_after
                ),
            )
        )

        atomic_write_json(
            CLEANING_POLICY_JSON,
            cleaning_policy,
        )

        shutil.copy2(
            SCRIPT_FILE,
            FROZEN_STEP04_SCRIPT,
        )

        summary_lines = [
            STEP_NAME,
            "=" * 96,
            "Status: PASS",
            (
                "Configuration fingerprint: "
                + step00_info[
                    "configuration_fingerprint_sha256"
                ]
            ),
            (
                "Candidate NIFTY sessions: "
                f"{len(candidate_calendar)}"
            ),
            (
                "Valid-reference anomaly exclusions: "
                f"{len(valid_reference_anomalies)}"
            ),
            (
                "Final analysis sessions: "
                f"{len(final_calendar)}"
            ),
            (
                "Final first session: "
                f"{final_calendar['Date'].min().date()}"
            ),
            (
                "Final last session: "
                f"{final_calendar['Date'].max().date()}"
            ),
            (
                "Exclusion-register rows: "
                f"{len(exclusion_register)}"
            ),
            (
                "Corporate-action date rows: "
                f"{len(corporate_actions)}"
            ),
            (
                "Stock-split date rows: "
                f"{len(split_audit)}"
            ),
            (
                "Large-return audit rows: "
                f"{len(large_return_audit)}"
            ),
            (
                "Volume policy: retain provider Volume; "
                "no manual split adjustment"
            ),
            (
                "Final-test state: "
                f"{step00_info['final_test_state']}"
            ),
        ]

        atomic_write_text(
            STEP04_SUMMARY_TXT,
            "\n".join(
                summary_lines
            )
            + "\n",
        )

        manifest_sha = (
            write_step04_manifest(
                step00_info=(
                    step00_info
                ),
                step01_info=(
                    step01_info
                ),
                step02_info=(
                    step02_info
                ),
                step03_info=(
                    step03_info
                ),
            )
        )

        stage = (
            "console_report"
        )

        print(
            "[10/10] Final Step-04 report..."
        )
        print()
        print(
            "-" * 132
        )
        print(
            "NONSTANDARD / EXCLUDED SOURCE DATES"
        )
        print(
            "-" * 132
        )

        interesting = (
            classification.loc[
                classification[
                    "modelling_policy"
                ]
                != "RETAIN"
            ]
            .copy()
        )

        for row in (
            interesting
            .itertuples(
                index=False
            )
        ):
            if (
                row.stocks_present
                == 0
            ):
                continue

            print(
                f"{pd.Timestamp(row.Date).date()} | "
                f"{row.day_of_week:<9} | "
                f"NIFTY-valid={str(row.candidate_nifty_session):<5} | "
                f"stocks={row.stocks_present:>2} | "
                f"nonzero-vol={row.stocks_nonzero_volume:>2} | "
                f"zero-vol={row.stocks_zero_volume:>2} | "
                f"zero+flat={row.stocks_zero_volume_flat_ohlc:>2} | "
                f"{row.primary_exclusion_code}"
            )

        print(
            "-" * 132
        )
        print(
            "FINAL ANALYSIS CALENDAR"
        )
        print(
            "-" * 132
        )
        print(
            "Step-03 candidate sessions        : "
            f"{len(candidate_calendar):,}"
        )
        print(
            "Valid NIFTY anomalies removed     : "
            f"{len(valid_reference_anomalies):,}"
        )
        print(
            "Final retained sessions           : "
            f"{len(final_calendar):,}"
        )
        print(
            "Final date range                  : "
            f"{final_calendar['Date'].min().date()} -> "
            f"{final_calendar['Date'].max().date()}"
        )
        print(
            "All 20 stocks present             : "
            f"{calendar_validation['all_20_stocks_present_on_final_calendar']}"
        )

        print(
            "-" * 132
        )
        print(
            "CORPORATE-ACTION / VOLUME POLICY"
        )
        print(
            "-" * 132
        )

        dividend_rows = int(
            (
                pd.to_numeric(
                    corporate_actions[
                        "Dividends"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0.0
                )
                != 0
            )
            .sum()
        ) if not corporate_actions.empty else 0

        split_rows = int(
            (
                pd.to_numeric(
                    corporate_actions[
                        "Stock Splits"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0.0
                )
                != 0
            )
            .sum()
        ) if not corporate_actions.empty else 0

        print(
            "Corporate-action date rows        : "
            f"{len(corporate_actions):,}"
        )
        print(
            "Dividend date rows                : "
            f"{dividend_rows:,}"
        )
        print(
            "Stock-split date rows             : "
            f"{split_rows:,}"
        )
        print(
            "Canonical adjusted-OHLC policy    : "
            "AdjClose/Close factor in Step 05"
        )
        print(
            "Volume policy                     : "
            "retain provider Volume; no manual split adjustment"
        )
        print(
            "Extreme-return automatic deletion : False"
        )

        if not large_return_audit.empty:
            print()
            print(
                "Large adjusted-return flags:"
            )

            for row in (
                large_return_audit
                .itertuples(
                    index=False
                )
            ):
                nearest = (
                    "none"
                    if pd.isna(
                        row.nearest_action_date
                    )
                    else str(
                        pd.Timestamp(
                            row.nearest_action_date
                        ).date()
                    )
                )

                print(
                    f"  {row.ticker:<17} "
                    f"{pd.Timestamp(row.Date).date()} | "
                    f"log-return={row.adjusted_log_return:+.6f} | "
                    f"nearest action={nearest} | "
                    f"within-7d={row.action_within_7_calendar_days}"
                )

        print(
            "-" * 132
        )
        print(
            "OUTPUT FILES"
        )
        print(
            "-" * 132
        )
        print(
            f"Session classification     : {DATE_CLASSIFICATION_CSV}"
        )
        print(
            f"Corporate-action register  : {CORPORATE_ACTION_REGISTER_CSV}"
        )
        print(
            f"Split-volume audit         : {SPLIT_AUDIT_CSV}"
        )
        print(
            f"Extreme-return audit       : {LARGE_RETURN_ACTION_AUDIT_CSV}"
        )
        print(
            f"Excluded-date register     : {EXCLUDED_DATE_REGISTER_CSV}"
        )
        print(
            f"Final analysis calendar    : {FINAL_REFERENCE_CALENDAR_CSV}"
        )
        print(
            f"Cleaning-policy JSON       : {CLEANING_POLICY_JSON}"
        )
        print(
            f"Step-04 manifest           : {STEP04_MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256           : {manifest_sha}"
        )
        print(
            f"Final-test state           : {step00_info['final_test_state']}"
        )
        print(
            "=" * 132
        )
        print(
            "PASS: Step-00 through Step-03 chain integrity verified."
        )
        print(
            "PASS: Final common analysis calendar frozen from predeclared rules."
        )
        print(
            "PASS: Corporate actions, splits and extreme returns audited without deleting legitimate events."
        )
        print(
            "PASS: Provider Volume retained without manual split adjustment or imputation."
        )
        print(
            "PASS: Raw Step-02 snapshots remain unchanged."
        )
        print(
            "PASS: Final held-out test remains LOCKED_NOT_OPENED."
        )
        print(
            "=" * 132
        )
        print(
            "Do not run Step 05 yet. Paste the complete terminal output for audit."
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
            "=" * 132
        )
        print(
            "STEP 04 FAILED — POLICY NOT FROZEN"
        )
        print(
            "=" * 132
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
            "=" * 132
        )
        print(
            "Do not proceed to Step 05."
        )

        return 30


# =============================================================================
# 21. CLI / ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the clean-V3 session/calendar and corporate-action policy."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an already frozen Step-04 policy without recomputing it."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verify_only:
        return verify_existing_step04()

    return run_new_policy()


if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "\nStep 04 interrupted by user."
        )
        sys.exit(
            130
        )

    except Exception as exc:
        print()
        print(
            "=" * 132
        )
        print(
            "STEP 04 UNHANDLED FAILURE"
        )
        print(
            "=" * 132
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
            "=" * 132
        )
        sys.exit(
            1
        )
