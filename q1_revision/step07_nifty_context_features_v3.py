#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 07 — CAUSAL 16-FEATURE NIFTY-50 MARKET-CONTEXT PANEL (CLEAN V3)

Purpose
-------
Construct and freeze the locked 16 price-only NIFTY-50 context predictors.

Critical temporal policy
------------------------
The NIFTY context is NOT forced to restart at a date removed from the stock
panel when the NIFTY observation on that date is itself valid.

The correct causal order is:

1. Read the immutable Yahoo ^NSEI snapshot frozen by Step 02.
2. Restrict it to the complete price-valid NIFTY calendar frozen by Step 03.
3. Construct all 16 NIFTY features on that uninterrupted valid-index calendar.
4. Retain every valid NIFTY observation that was excluded only from the stock
   analysis calendar by the Step-04 stock-panel anomaly rule.
5. Only AFTER NIFTY feature construction, align the completed NIFTY context
   rows to the Step-04 final stock-analysis calendar.

This matters because a valid NIFTY observation on an excluded stock date is
already known before the next retained stock session and is therefore legitimate
historical market context for that next session.

For the current frozen experiment, Step 04 identifies one such valid NIFTY
session, 2025-03-18. The date is NOT hard-coded by this script; it is derived
from the frozen Step-04 cleaning policy and cross-checked against the Step-03
and Step-04 calendars.

NIFTY volume policy
-------------------
NIFTY Volume and all NIFTY volume-derived predictors are prohibited.

Scientific boundary
-------------------
This stage:
- constructs exactly 16 price-only NIFTY predictors;
- uses only current/past NIFTY prices;
- performs no scaling;
- creates no targets;
- estimates no regimes;
- creates no sequences;
- trains/selects no model;
- tunes no threshold;
- does not score the final held-out test.

Signal timing
-------------
Session-t NIFTY context is available only after session-t close.

Run
---
python step07_nifty_context_features_v3.py

Verify frozen Step 07
---------------------
python step07_nifty_context_features_v3.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
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
        "step07_nifty_context_features_v3.py must be located inside "
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

STEP_NUMBER = 7
STEP_ID = "07"
STEP_NAME = "STEP 07 — CAUSAL 16-FEATURE NIFTY-50 MARKET-CONTEXT PANEL"
SCHEMA_VERSION = "1.0"

EXPECTED_FEATURE_COUNT = 16
EXPECTED_FIRST_COMPLETE_POSITION = 20
EXPECTED_FULL_SOURCE_MISSING_VALUES = 104

SIGNAL_TIMING_POLICY = (
    "session-t NIFTY features are available only after session-t close"
)

MARKET_INDEX_NAME = getattr(
    cfg,
    "MARKET_INDEX_NAME",
    "NIFTY 50",
)


# =============================================================================
# 2. FROZEN INPUT PATHS
# =============================================================================

RAW_NIFTY_PARQUET = (
    cfg.RAW_DATA_DIR
    / "provider_snapshot"
    / "parquet"
    / "NSEI.parquet"
)

STEP01_MANIFEST = cfg.MANIFEST_DIR / "01_environment_audit_manifest.json"
STEP01_MANIFEST_SHA = cfg.MANIFEST_DIR / "01_environment_audit_manifest.sha256"
STEP01_AUDIT_JSON = cfg.ENVIRONMENT_DIR / "01_environment_audit.json"

STEP02_MANIFEST = cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.json"
STEP02_MANIFEST_SHA = cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.sha256"

STEP03_MANIFEST = cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.json"
STEP03_MANIFEST_SHA = cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.sha256"
STEP03_CANDIDATE_CALENDAR = (
    cfg.REFERENCE_DATA_DIR
    / "03_candidate_nifty_price_calendar.csv"
)

STEP04_MANIFEST = cfg.MANIFEST_DIR / "04_session_policy_manifest.json"
STEP04_MANIFEST_SHA = cfg.MANIFEST_DIR / "04_session_policy_manifest.sha256"
STEP04_POLICY = cfg.MANIFEST_DIR / "04_cleaning_policy.json"
STEP04_FINAL_CALENDAR = (
    cfg.REFERENCE_DATA_DIR
    / "04_final_analysis_calendar.csv"
)
STEP04_CLASSIFICATION = (
    cfg.AUDIT_DIR
    / "04_session_classification.csv"
)

STEP05_MANIFEST = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.json"
STEP05_MANIFEST_SHA = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.sha256"

STEP06_MANIFEST = cfg.MANIFEST_DIR / "06_stock_feature_manifest.json"
STEP06_MANIFEST_SHA = cfg.MANIFEST_DIR / "06_stock_feature_manifest.sha256"


# =============================================================================
# 3. STEP-07 OUTPUT PATHS
# =============================================================================

STEP07_OUTPUT_DIR = (
    cfg.FEATURE_DATA_DIR
    / "step07_nifty_context"
)

FULL_SOURCE_FEATURE_PARQUET = (
    STEP07_OUTPUT_DIR
    / "07_nifty_context_16_full_calendar.parquet"
)

FULL_SOURCE_FEATURE_CSV = (
    STEP07_OUTPUT_DIR
    / "07_nifty_context_16_full_calendar.csv"
)

ALIGNED_FEATURE_PARQUET = (
    STEP07_OUTPUT_DIR
    / "07_nifty_context_16_aligned.parquet"
)

ALIGNED_FEATURE_CSV = (
    STEP07_OUTPUT_DIR
    / "07_nifty_context_16_aligned.csv"
)

COMPLETE_ALIGNED_PARQUET = (
    STEP07_OUTPUT_DIR
    / "07_nifty_context_16_aligned_complete.parquet"
)

COMPLETE_ALIGNED_CSV = (
    STEP07_OUTPUT_DIR
    / "07_nifty_context_16_aligned_complete.csv"
)

FEATURE_DICTIONARY_CSV = (
    cfg.AUDIT_DIR
    / "07_nifty_feature_dictionary.csv"
)

WARMUP_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "07_nifty_feature_warmup_audit.csv"
)

CAUSALITY_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "07_nifty_prefix_causality_audit.csv"
)

BRIDGE_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "07_nifty_bridge_audit.csv"
)

ALIGNMENT_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "07_nifty_alignment_audit.csv"
)

STEP05_OHLC_RECONCILIATION_CSV = (
    cfg.AUDIT_DIR
    / "07_nifty_step05_ohlc_reconciliation.csv"
)

HASH_REGISTER_CSV = (
    cfg.AUDIT_DIR
    / "07_nifty_feature_hashes.csv"
)

SOURCE_HASH_CSV = (
    cfg.AUDIT_DIR
    / "07_source_file_hashes.csv"
)

STEP07_SUMMARY_JSON = (
    cfg.AUDIT_DIR
    / "07_nifty_feature_summary.json"
)

MANIFEST_JSON = (
    cfg.MANIFEST_DIR
    / "07_nifty_context_feature_manifest.json"
)

MANIFEST_SHA = (
    cfg.MANIFEST_DIR
    / "07_nifty_context_feature_manifest.sha256"
)

FROZEN_STEP07_SCRIPT = (
    cfg.MANIFEST_DIR
    / "step07_nifty_context_features_frozen.py"
)

SUMMARY_TXT = (
    cfg.REPORT_DIR
    / "07_nifty_context_feature_summary.txt"
)

FAILED_ATTEMPT_DIR = (
    cfg.AUDIT_DIR
    / "step07_failed_attempts"
)


# =============================================================================
# 4. LOCKED 16-FEATURE SPECIFICATION
# =============================================================================

NIFTY_FEATURE_GROUPS = {

    "price_and_intraday_range": [
        "nifty_open",
        "nifty_high",
        "nifty_low",
        "nifty_close",
        "nifty_high_low_range",
    ],

    "returns": [
        "nifty_return_1d",
        "nifty_log_return_1d",
    ],

    "rolling_trend": [
        "nifty_rolling_mean_5",
        "nifty_rolling_mean_10",
        "nifty_rolling_mean_20",
    ],

    "rolling_volatility": [
        "nifty_rolling_std_5",
        "nifty_rolling_std_10",
        "nifty_rolling_std_20",
    ],

    "momentum": [
        "nifty_momentum_5",
        "nifty_momentum_10",
        "nifty_momentum_20",
    ],
}

NIFTY_FEATURE_COLUMNS = [
    feature
    for group_features in NIFTY_FEATURE_GROUPS.values()
    for feature in group_features
]

if len(NIFTY_FEATURE_COLUMNS) != EXPECTED_FEATURE_COUNT:
    raise AssertionError(
        f"Expected {EXPECTED_FEATURE_COUNT} NIFTY features; "
        f"found {len(NIFTY_FEATURE_COLUMNS)}."
    )

if len(set(NIFTY_FEATURE_COLUMNS)) != len(NIFTY_FEATURE_COLUMNS):
    raise AssertionError("Duplicate NIFTY feature names.")

if any("volume" in feature.lower() for feature in NIFTY_FEATURE_COLUMNS):
    raise AssertionError(
        "NIFTY volume-derived predictors are prohibited."
    )


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    dataframe.to_csv(
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
        path.resolve()
        .relative_to(
            cfg.EXPERIMENT_ROOT_DIR.resolve()
        )
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
        raise RuntimeError(
            "Step-00 freeze is not VERIFIED_EXISTING."
        )

    lock_path = cfg.get_final_test_lock_path()

    with lock_path.open("r", encoding="utf-8") as handle:
        lock_payload = json.load(handle)

    if lock_payload.get("state") != "LOCKED_NOT_OPENED":
        raise RuntimeError(
            "Final-test state is not LOCKED_NOT_OPENED."
        )

    if lock_payload.get("final_test_opened") is not False:
        raise RuntimeError(
            "final_test_opened != False."
        )

    if lock_payload.get("final_test_scored") is not False:
        raise RuntimeError(
            "final_test_scored != False."
        )

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
        raise RuntimeError(
            f"{label} manifest missing: {manifest_path}"
        )

    if not sha_path.exists():
        raise RuntimeError(
            f"{label} manifest SHA sidecar missing: {sha_path}"
        )

    expected = (
        sha_path
        .read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )

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

        if not isinstance(record, dict):
            continue

        path_text = record.get("path")

        if not path_text:
            continue

        path = Path(path_text)

        if not path.exists():
            failures.append(
                f"missing: {name} -> {path}"
            )
            continue

        expected = record.get("sha256")

        if expected and sha256_file(path) != expected:
            failures.append(
                f"hash mismatch: {name}"
            )

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
        raise RuntimeError(
            "Stored Step-01 status is not PASS."
        )

    if (
        manifest.get("configuration_fingerprint_sha256")
        != expected_configuration_fingerprint
    ):
        raise RuntimeError(
            "Step-01 / Step-00 fingerprint mismatch."
        )

    verify_manifest_artifacts(manifest)

    if not STEP01_AUDIT_JSON.exists():
        raise RuntimeError(
            "Step-01 environment audit JSON is missing."
        )

    with STEP01_AUDIT_JSON.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)

    stored_python = (
        audit
        .get("system", {})
        .get("python_executable")
    )

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
                (
                    distribution,
                    stored_version,
                    current_version,
                )
            )

    if package_drift:
        detail = "\n".join(
            f"{name}: Step01={stored!r}, current={current!r}"
            for name, stored, current in package_drift
        )

        raise RuntimeError(
            "Required-package drift detected after Step 01:\n"
            + detail
        )

    return {
        "manifest_sha256": manifest_sha,
        "python_executable": stored_python,
        "conda_environment": (
            audit
            .get("system", {})
            .get("conda_default_env")
        ),
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
        raise RuntimeError(
            "Stored Step-02 status is not PASS."
        )

    if (
        manifest
        .get("step00", {})
        .get("configuration_fingerprint_sha256")
        != expected_configuration_fingerprint
    ):
        raise RuntimeError(
            "Step-02 / Step-00 fingerprint mismatch."
        )

    if (
        manifest
        .get("step01", {})
        .get("step01_manifest_sha256")
        != expected_step01_sha
    ):
        raise RuntimeError(
            "Step-02 / Step-01 manifest mismatch."
        )

    verify_manifest_artifacts(
        manifest,
        artifact_key="canonical_artifacts",
        skip_names=("raw_snapshot_root",),
    )

    raw_records = manifest.get(
        "raw_file_records",
        [],
    )

    failures = []

    for record in raw_records:
        path = Path(record["file_path"])

        if not path.exists():
            failures.append(
                f"missing raw file: {path}"
            )
            continue

        if sha256_file(path) != record["sha256"]:
            failures.append(
                f"raw hash mismatch: {path}"
            )

    if failures:
        raise RuntimeError(
            "Step-02 raw-file verification failed:\n"
            + "\n".join(failures)
        )

    exact_nifty_records = [
        record
        for record in raw_records
        if Path(record["file_path"]).resolve()
        == RAW_NIFTY_PARQUET.resolve()
    ]

    if len(exact_nifty_records) != 1:
        raise RuntimeError(
            "Could not identify exactly one frozen raw NIFTY Parquet "
            "record in Step 02."
        )

    raw_nifty_sha = sha256_file(
        RAW_NIFTY_PARQUET
    )

    if (
        raw_nifty_sha
        != exact_nifty_records[0]["sha256"]
    ):
        raise RuntimeError(
            "Raw NIFTY Parquet hash mismatch."
        )

    return {
        "manifest_sha256": manifest_sha,
        "verified_raw_file_count": int(len(raw_records)),
        "raw_nifty_sha256": raw_nifty_sha,
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
        raise RuntimeError(
            "Stored Step-03 status is not PASS."
        )

    expected = {
        "configuration_fingerprint_sha256": (
            expected_configuration_fingerprint
        ),
        "step01_manifest_sha256": (
            expected_step01_sha
        ),
        "step02_manifest_sha256": (
            expected_step02_sha
        ),
    }

    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Step-03 chain mismatch: {key}"
            )

    verify_manifest_artifacts(manifest)

    if not STEP03_CANDIDATE_CALENDAR.exists():
        raise RuntimeError(
            f"Missing Step-03 candidate calendar: "
            f"{STEP03_CANDIDATE_CALENDAR}"
        )

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
        raise RuntimeError(
            "Stored Step-04 status is not PASS."
        )

    expected = {
        "configuration_fingerprint_sha256": (
            expected_configuration_fingerprint
        ),
        "step01_manifest_sha256": (
            expected_step01_sha
        ),
        "step02_manifest_sha256": (
            expected_step02_sha
        ),
        "step03_manifest_sha256": (
            expected_step03_sha
        ),
    }

    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Step-04 chain mismatch: {key}"
            )

    verify_manifest_artifacts(manifest)

    for path in (
        STEP04_POLICY,
        STEP04_FINAL_CALENDAR,
        STEP04_CLASSIFICATION,
    ):
        if not path.exists():
            raise RuntimeError(
                f"Required Step-04 file missing: {path}"
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
        raise RuntimeError(
            "Stored Step-05 status is not PASS."
        )

    expected = {
        "configuration_fingerprint_sha256": (
            expected_configuration_fingerprint
        ),
        "step01_manifest_sha256": (
            expected_step01_sha
        ),
        "step02_manifest_sha256": (
            expected_step02_sha
        ),
        "step03_manifest_sha256": (
            expected_step03_sha
        ),
        "step04_manifest_sha256": (
            expected_step04_sha
        ),
    }

    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Step-05 chain mismatch: {key}"
            )

    verify_manifest_artifacts(manifest)

    failures = []

    for record in manifest.get(
        "individual_stock_output_records",
        [],
    ):
        path = Path(record["file_path"])

        if not path.exists():
            failures.append(
                f"missing individual stock output: {path}"
            )
            continue

        if sha256_file(path) != record["sha256"]:
            failures.append(
                f"individual stock hash mismatch: {path}"
            )

    if failures:
        raise RuntimeError(
            "Step-05 individual-stock verification failed:\n"
            + "\n".join(failures)
        )

    return {
        "manifest_sha256": manifest_sha,
    }


def verify_step06(
    expected_configuration_fingerprint: str,
    expected_step01_sha: str,
    expected_step02_sha: str,
    expected_step03_sha: str,
    expected_step04_sha: str,
    expected_step05_sha: str,
) -> Dict[str, Any]:
    manifest_sha = verify_sha_sidecar(
        STEP06_MANIFEST,
        STEP06_MANIFEST_SHA,
        "Step 06",
    )

    with STEP06_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("step", {}).get("status") != "PASS":
        raise RuntimeError(
            "Stored Step-06 status is not PASS."
        )

    expected = {
        "configuration_fingerprint_sha256": (
            expected_configuration_fingerprint
        ),
        "step01_manifest_sha256": (
            expected_step01_sha
        ),
        "step02_manifest_sha256": (
            expected_step02_sha
        ),
        "step03_manifest_sha256": (
            expected_step03_sha
        ),
        "step04_manifest_sha256": (
            expected_step04_sha
        ),
        "step05_manifest_sha256": (
            expected_step05_sha
        ),
    }

    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Step-06 chain mismatch: {key}"
            )

    verify_manifest_artifacts(manifest)

    return {
        "manifest_sha256": manifest_sha,
    }


# =============================================================================
# 8. IMMUTABILITY / STAGING
# =============================================================================

def canonical_step07_outputs() -> tuple[Path, ...]:
    return (
        STEP07_OUTPUT_DIR,
        FEATURE_DICTIONARY_CSV,
        WARMUP_AUDIT_CSV,
        CAUSALITY_AUDIT_CSV,
        BRIDGE_AUDIT_CSV,
        ALIGNMENT_AUDIT_CSV,
        STEP05_OHLC_RECONCILIATION_CSV,
        HASH_REGISTER_CSV,
        SOURCE_HASH_CSV,
        STEP07_SUMMARY_JSON,
        MANIFEST_JSON,
        MANIFEST_SHA,
        FROZEN_STEP07_SCRIPT,
        SUMMARY_TXT,
    )


def assert_new_step07_target() -> None:
    collisions = [
        path
        for path in canonical_step07_outputs()
        if path.exists()
    ]

    if collisions:
        detail = "\n".join(
            str(path)
            for path in collisions
        )

        raise RuntimeError(
            "Step-07 canonical output(s) already exist. "
            "Refusing to overwrite.\n"
            f"{detail}\n\n"
            "If Step 07 already passed, run --verify-only."
        )


def create_staging_root() -> Path:
    timestamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )

    root = (
        cfg.FEATURE_DATA_DIR
        / f"_step07_staging_{timestamp}_{os.getpid()}"
    )

    if root.exists():
        raise RuntimeError(
            f"Unexpected staging path: {root}"
        )

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    return root


def archive_partial_outputs() -> Optional[Path]:
    if MANIFEST_JSON.exists():
        return None

    existing = [
        path
        for path in canonical_step07_outputs()
        if path.exists()
    ]

    if not existing:
        return None

    timestamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )

    archive = (
        FAILED_ATTEMPT_DIR
        / f"partial_artifacts_{timestamp}"
    )

    archive.mkdir(
        parents=True,
        exist_ok=False,
    )

    for path in existing:
        try:
            shutil.move(
                str(path),
                str(archive / path.name),
            )
        except Exception:
            pass

    return archive


# =============================================================================
# 9. CALENDAR / SOURCE LOADERS
# =============================================================================

def load_bridge_dates(
    step04_policy: Dict[str, Any],
) -> list[pd.Timestamp]:
    anomaly_policy = (
        step04_policy
        .get(
            "date_wide_anomaly_policy",
            {},
        )
    )

    values = anomaly_policy.get(
        "valid_nifty_sessions_excluded_by_rule",
        [],
    )

    declared_count = anomaly_policy.get(
        "valid_nifty_session_exclusion_count"
    )

    if (
        declared_count is not None
        and int(declared_count) != len(values)
    ):
        raise RuntimeError(
            "Step-04 excluded-valid-NIFTY count does not match "
            "the frozen date list."
        )

    dates = sorted(
        {
            pd.Timestamp(value).normalize()
            for value in values
        }
    )

    return dates


def load_candidate_nifty_source(
    bridge_dates: list[pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(
        RAW_NIFTY_PARQUET,
        engine="pyarrow",
    )

    raw["Date"] = (
        pd.to_datetime(
            raw["Date"],
            errors="raise",
        )
        .dt.normalize()
    )

    if raw["Date"].duplicated().any():
        raise RuntimeError(
            "Duplicate dates in raw NIFTY snapshot."
        )

    calendar = pd.read_csv(
        STEP03_CANDIDATE_CALENDAR,
        parse_dates=["Date"],
    )

    calendar["Date"] = (
        pd.to_datetime(
            calendar["Date"],
            errors="raise",
        )
        .dt.normalize()
    )

    required_calendar_columns = {
        "session_number",
        "Date",
    }

    missing = (
        required_calendar_columns
        - set(calendar.columns)
    )

    if missing:
        raise RuntimeError(
            "Step-03 candidate calendar missing columns: "
            + ", ".join(sorted(missing))
        )

    calendar = (
        calendar[
            [
                "session_number",
                "Date",
            ]
        ]
        .sort_values("session_number")
        .reset_index(drop=True)
    )

    if calendar["Date"].duplicated().any():
        raise RuntimeError(
            "Duplicate dates in Step-03 candidate NIFTY calendar."
        )

    if not calendar["Date"].is_monotonic_increasing:
        raise RuntimeError(
            "Step-03 candidate NIFTY calendar is not chronological."
        )

    source = calendar.merge(
        raw,
        on="Date",
        how="left",
        validate="one_to_one",
    )

    required_prices = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing_price_columns = [
        column
        for column in required_prices
        if column not in source.columns
    ]

    if missing_price_columns:
        raise RuntimeError(
            "NIFTY source missing OHLC columns: "
            + ", ".join(missing_price_columns)
        )

    if source[required_prices].isna().any().any():
        raise RuntimeError(
            "Price-valid NIFTY source contains missing OHLC values."
        )

    for column in required_prices:
        source[column] = pd.to_numeric(
            source[column],
            errors="raise",
        ).astype("float64")

    numeric_prices = (
        source[required_prices]
        .to_numpy(dtype=float)
    )

    if (~np.isfinite(numeric_prices)).any():
        raise RuntimeError(
            "Non-finite NIFTY OHLC value detected."
        )

    if (numeric_prices <= 0).any():
        raise RuntimeError(
            "Non-positive NIFTY OHLC value detected."
        )

    tolerance = 1e-10

    invalid_geometry = (
        (
            source["High"] + tolerance
            <
            source[
                [
                    "Open",
                    "Low",
                    "Close",
                ]
            ].max(axis=1)
        )
        |
        (
            source["Low"] - tolerance
            >
            source[
                [
                    "Open",
                    "High",
                    "Close",
                ]
            ].min(axis=1)
        )
    )

    if invalid_geometry.any():
        raise RuntimeError(
            "NIFTY OHLC geometry violations detected: "
            f"{int(invalid_geometry.sum())}"
        )

    source_date_set = set(
        source["Date"].tolist()
    )

    for bridge_date in bridge_dates:
        if bridge_date not in source_date_set:
            raise RuntimeError(
                "A Step-04 valid-NIFTY exclusion is missing from the "
                "Step-03 candidate NIFTY calendar: "
                f"{bridge_date.date()}"
            )

    return source, calendar


def load_final_stock_calendar() -> pd.DataFrame:
    calendar = pd.read_csv(
        STEP04_FINAL_CALENDAR,
        parse_dates=["Date"],
    )

    calendar["Date"] = (
        pd.to_datetime(
            calendar["Date"],
            errors="raise",
        )
        .dt.normalize()
    )

    required = {
        "session_number",
        "Date",
    }

    missing = (
        required
        - set(calendar.columns)
    )

    if missing:
        raise RuntimeError(
            "Step-04 final calendar missing columns: "
            + ", ".join(sorted(missing))
        )

    calendar = (
        calendar[
            [
                "session_number",
                "Date",
            ]
        ]
        .sort_values("session_number")
        .reset_index(drop=True)
    )

    if calendar["Date"].duplicated().any():
        raise RuntimeError(
            "Duplicate dates in Step-04 final calendar."
        )

    if not calendar["Date"].is_monotonic_increasing:
        raise RuntimeError(
            "Step-04 final calendar is not chronological."
        )

    return calendar


def validate_calendar_relation(
    source: pd.DataFrame,
    stock_calendar: pd.DataFrame,
    bridge_dates: list[pd.Timestamp],
) -> Dict[str, Any]:
    source_dates = pd.DatetimeIndex(
        source["Date"]
    )

    stock_dates = pd.DatetimeIndex(
        stock_calendar["Date"]
    )

    stock_missing_from_source = (
        stock_dates.difference(
            source_dates
        )
    )

    source_only_dates = (
        source_dates.difference(
            stock_dates
        )
    )

    expected_source_only = pd.DatetimeIndex(
        bridge_dates
    )

    if len(stock_missing_from_source) != 0:
        raise RuntimeError(
            "Final stock-calendar dates are missing from the "
            "price-valid NIFTY source:\n"
            f"{list(stock_missing_from_source)}"
        )

    if not source_only_dates.equals(
        expected_source_only
    ):
        raise RuntimeError(
            "The difference between the Step-03 price-valid NIFTY calendar "
            "and Step-04 final stock calendar is not exactly the frozen "
            "valid-NIFTY exclusion set.\n"
            f"Expected: {list(expected_source_only)}\n"
            f"Observed: {list(source_only_dates)}"
        )

    return {
        "source_rows": int(len(source)),
        "aligned_calendar_rows": int(len(stock_calendar)),
        "source_only_date_count": int(len(source_only_dates)),
        "source_only_dates": [
            date.strftime("%Y-%m-%d")
            for date in source_only_dates
        ],
        "stock_dates_missing_from_source": 0,
    }


# =============================================================================
# 10. FEATURE CONSTRUCTION
# =============================================================================

def construct_nifty_features(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct all 16 NIFTY context variables using only current/past
    NIFTY observations.
    """

    table = (
        source
        .copy()
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------------------
    # Price levels / intraday range
    # -------------------------------------------------------------------------

    table[
        "nifty_open"
    ] = table[
        "Open"
    ]

    table[
        "nifty_high"
    ] = table[
        "High"
    ]

    table[
        "nifty_low"
    ] = table[
        "Low"
    ]

    table[
        "nifty_close"
    ] = table[
        "Close"
    ]

    table[
        "nifty_high_low_range"
    ] = (
        (
            table[
                "High"
            ]
            -
            table[
                "Low"
            ]
        )
        /
        table[
            "Close"
        ]
    )

    # -------------------------------------------------------------------------
    # Returns
    # -------------------------------------------------------------------------

    table[
        "nifty_return_1d"
    ] = (
        table[
            "Close"
        ]
        .pct_change(
            fill_method=None
        )
    )

    table[
        "nifty_log_return_1d"
    ] = np.log(
        table[
            "Close"
        ]
        /
        table[
            "Close"
        ].shift(
            1
        )
    )

    # -------------------------------------------------------------------------
    # Rolling means, log-return volatility, momentum
    # -------------------------------------------------------------------------

    for window in (
        5,
        10,
        20,
    ):

        table[
            f"nifty_rolling_mean_{window}"
        ] = (
            table[
                "Close"
            ]
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

        table[
            f"nifty_rolling_std_{window}"
        ] = (
            table[
                "nifty_log_return_1d"
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
            f"nifty_momentum_{window}"
        ] = (
            table[
                "Close"
            ]
            /
            table[
                "Close"
            ].shift(
                window
            )
            -
            1.0
        )

    table[
        NIFTY_FEATURE_COLUMNS
    ] = (
        table[
            NIFTY_FEATURE_COLUMNS
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    output = table[
        [
            "Date",
        ]
        +
        NIFTY_FEATURE_COLUMNS
    ].copy()

    output.insert(
        1,
        "index_ticker",
        cfg.MARKET_INDEX_TICKER,
    )

    output.insert(
        2,
        "index_name",
        cfg.MARKET_INDEX_NAME,
    )

    return output

def build_warmup_audit(
    table: pd.DataFrame,
    panel_name: str,
) -> pd.DataFrame:

    records = []

    for feature in (
        NIFTY_FEATURE_COLUMNS
    ):

        series = table[
            feature
        ]

        valid_positions = (
            np.flatnonzero(
                series.notna()
                .to_numpy()
            )
        )

        if len(
            valid_positions
        ) == 0:

            raise AssertionError(
                f"{feature}: no valid observations."
            )

        first_valid = int(
            valid_positions[
                0
            ]
        )

        internal_missing = int(
            series.iloc[
                first_valid:
            ]
            .isna()
            .sum()
        )

        if internal_missing != 0:

            raise AssertionError(
                f"{panel_name} / {feature}: "
                f"{internal_missing} internal missing values."
            )

        records.append(
            {
                "panel_name": (
                    panel_name
                ),
                "feature": (
                    feature
                ),
                "first_valid_position": (
                    first_valid
                ),
                "first_valid_date": (
                    table.loc[
                        first_valid,
                        "Date",
                    ]
                ),
                "warmup_missing_rows": (
                    first_valid
                ),
                "internal_post_warmup_missing": (
                    internal_missing
                ),
            }
        )

    return pd.DataFrame(
        records
    )

def run_prefix_causality_audit(
    source: pd.DataFrame,
    full_features: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for fraction in (
        0.50,
        0.75,
        0.90,
    ):

        cutoff_position = int(
            np.floor(
                len(
                    source
                )
                *
                fraction
            )
        ) - 1

        cutoff_position = max(
            100,
            min(
                cutoff_position,
                len(
                    source
                )
                - 1,
            ),
        )

        prefix_source = (
            source.iloc[
                :cutoff_position + 1
            ]
            .copy()
        )

        prefix_features = (
            construct_nifty_features(
                prefix_source
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
                NIFTY_FEATURE_COLUMNS
            ]
            .iloc[
                -1
            ]
            .to_numpy(
                dtype=float
            )
        )

        matching = (
            full_features.loc[
                full_features[
                    "Date"
                ]
                == cutoff_date
            ]
        )

        if len(
            matching
        ) != 1:

            raise AssertionError(
                "Prefix audit could not identify "
                "one full-panel cutoff row."
            )

        full_values = (
            matching[
                NIFTY_FEATURE_COLUMNS
            ]
            .iloc[
                0
            ]
            .to_numpy(
                dtype=float
            )
        )

        matches = bool(
            np.allclose(
                prefix_values,
                full_values,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        )

        finite_mask = (
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
                            finite_mask
                        ]
                        -
                        full_values[
                            finite_mask
                        ]
                    )
                )
            )
            if finite_mask.any()
            else 0.0
        )

        records.append(
            {
                "cutoff_fraction": (
                    fraction
                ),
                "cutoff_position": (
                    cutoff_position
                ),
                "cutoff_date": (
                    cutoff_date
                ),
                "all_features_match": (
                    matches
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

        raise AssertionError(
            "NIFTY prefix-causality audit failed:\n"
            f"{audit.loc[~audit['all_features_match']]}"
        )

    return audit

def build_feature_dictionary() -> pd.DataFrame:

    definitions = {

        "nifty_open": (
            "Current NIFTY opening index level"
        ),

        "nifty_high": (
            "Current NIFTY high index level"
        ),

        "nifty_low": (
            "Current NIFTY low index level"
        ),

        "nifty_close": (
            "Current NIFTY closing index level"
        ),

        "nifty_high_low_range": (
            "(high - low) / close"
        ),

        "nifty_return_1d": (
            "close_t / close_(t-1) - 1"
        ),

        "nifty_log_return_1d": (
            "log(close_t / close_(t-1))"
        ),

        "nifty_rolling_mean_5": (
            "5-session rolling mean of NIFTY close"
        ),

        "nifty_rolling_mean_10": (
            "10-session rolling mean of NIFTY close"
        ),

        "nifty_rolling_mean_20": (
            "20-session rolling mean of NIFTY close"
        ),

        "nifty_rolling_std_5": (
            "5-session sample standard deviation "
            "of NIFTY log return"
        ),

        "nifty_rolling_std_10": (
            "10-session sample standard deviation "
            "of NIFTY log return"
        ),

        "nifty_rolling_std_20": (
            "20-session sample standard deviation "
            "of NIFTY log return"
        ),

        "nifty_momentum_5": (
            "close_t / close_(t-5) - 1"
        ),

        "nifty_momentum_10": (
            "close_t / close_(t-10) - 1"
        ),

        "nifty_momentum_20": (
            "close_t / close_(t-20) - 1"
        ),
    }

    rows = []

    feature_number = 0

    for group_name, features in (
        NIFTY_FEATURE_GROUPS.items()
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
                    "definition": (
                        definitions[
                            feature
                        ]
                    ),
                    "uses_volume": False,
                    "causal": True,
                    "constructed_before_final_stock_calendar_alignment": True,
                    "scaling_applied_here": False,
                    "target_derived": False,
                }
            )

    result = pd.DataFrame(
        rows
    )

    if len(
        result
    ) != EXPECTED_FEATURE_COUNT:

        raise AssertionError(
            "NIFTY feature dictionary is not 16 rows."
        )

    return result


# =============================================================================
# 11. FEATURE-PANEL VALIDATION
# =============================================================================

def validate_feature_panel(
    table: pd.DataFrame,
    *,
    expected_rows: int,
    panel_name: str,
    expect_warmup_pattern: bool,
) -> Dict[str, Any]:
    expected_columns = [
        "Date",
        "index_ticker",
        "index_name",
    ] + NIFTY_FEATURE_COLUMNS

    if list(table.columns) != expected_columns:
        raise RuntimeError(
            f"{panel_name}: unexpected columns/order."
        )

    if len(table) != expected_rows:
        raise RuntimeError(
            f"{panel_name}: expected {expected_rows:,} rows; "
            f"found {len(table):,}."
        )

    if table["Date"].duplicated().any():
        raise RuntimeError(
            f"{panel_name}: duplicate dates."
        )

    if not table["Date"].is_monotonic_increasing:
        raise RuntimeError(
            f"{panel_name}: dates are not chronological."
        )

    if not table["index_ticker"].eq(
        cfg.MARKET_INDEX_TICKER
    ).all():
        raise RuntimeError(
            f"{panel_name}: incorrect index ticker."
        )

    if not table["index_name"].eq(
        MARKET_INDEX_NAME
    ).all():
        raise RuntimeError(
            f"{panel_name}: incorrect index name."
        )

    if any(
        "volume" in column.lower()
        for column in table.columns
    ):
        raise RuntimeError(
            f"{panel_name}: prohibited NIFTY-volume field found."
        )

    numeric = (
        table[NIFTY_FEATURE_COLUMNS]
        .to_numpy(dtype=float)
    )

    if np.isinf(numeric).any():
        raise RuntimeError(
            f"{panel_name}: infinite feature values."
        )

    missing_total = int(
        table[NIFTY_FEATURE_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    complete_mask = (
        table[NIFTY_FEATURE_COLUMNS]
        .notna()
        .all(axis=1)
    )

    complete_positions = np.flatnonzero(
        complete_mask.to_numpy()
    )

    if len(complete_positions) == 0:
        raise RuntimeError(
            f"{panel_name}: no complete rows."
        )

    first_complete_position = int(
        complete_positions[0]
    )

    if expect_warmup_pattern:
        if (
            first_complete_position
            != EXPECTED_FIRST_COMPLETE_POSITION
        ):
            raise RuntimeError(
                f"{panel_name}: expected first complete position "
                f"{EXPECTED_FIRST_COMPLETE_POSITION}; "
                f"found {first_complete_position}."
            )

        if (
            missing_total
            != EXPECTED_FULL_SOURCE_MISSING_VALUES
        ):
            raise RuntimeError(
                f"{panel_name}: expected "
                f"{EXPECTED_FULL_SOURCE_MISSING_VALUES} feature warm-up "
                f"missing values; found {missing_total}."
            )

    return {
        "panel_name": panel_name,
        "rows": int(len(table)),
        "feature_count": int(len(NIFTY_FEATURE_COLUMNS)),
        "feature_missing_values": missing_total,
        "first_complete_position": (
            first_complete_position
        ),
        "first_complete_date": (
            table.loc[
                first_complete_position,
                "Date",
            ]
        ),
        "complete_rows": int(
            complete_mask.sum()
        ),
    }


# =============================================================================
# 12. DYNAMIC BRIDGE AUDIT
# =============================================================================

def build_bridge_blocks(
    source: pd.DataFrame,
    bridge_dates: list[pd.Timestamp],
) -> list[Dict[str, Any]]:
    if not bridge_dates:
        return []

    positions = {
        pd.Timestamp(date): int(position)
        for position, date in enumerate(
            source["Date"]
        )
    }

    bridge_positions = sorted(
        positions[date]
        for date in bridge_dates
    )

    blocks: list[list[int]] = []
    current = [bridge_positions[0]]

    for position in bridge_positions[1:]:
        if position == current[-1] + 1:
            current.append(position)
        else:
            blocks.append(current)
            current = [position]

    blocks.append(current)

    output = []

    for block_number, block in enumerate(
        blocks,
        start=1,
    ):
        first_position = block[0]
        last_position = block[-1]

        if first_position <= 0:
            raise RuntimeError(
                "A valid-NIFTY exclusion occurs at the first source row; "
                "bridge audit cannot establish pre-history."
            )

        if last_position >= len(source) - 1:
            raise RuntimeError(
                "A valid-NIFTY exclusion occurs at the last source row; "
                "bridge audit cannot establish post-history."
            )

        output.append(
            {
                "bridge_block_number": block_number,
                "first_bridge_position": first_position,
                "last_bridge_position": last_position,
                "first_bridge_date": pd.Timestamp(
                    source.loc[
                        first_position,
                        "Date",
                    ]
                ),
                "last_bridge_date": pd.Timestamp(
                    source.loc[
                        last_position,
                        "Date",
                    ]
                ),
                "bridge_session_count": int(
                    len(block)
                ),
                "pre_bridge_position": int(
                    first_position - 1
                ),
                "pre_bridge_date": pd.Timestamp(
                    source.loc[
                        first_position - 1,
                        "Date",
                    ]
                ),
                "post_bridge_position": int(
                    last_position + 1
                ),
                "post_bridge_date": pd.Timestamp(
                    source.loc[
                        last_position + 1,
                        "Date",
                    ]
                ),
            }
        )

    return output


def build_bridge_audit(
    source: pd.DataFrame,
    full_features: pd.DataFrame,
    aligned_features: pd.DataFrame,
    bridge_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    blocks = build_bridge_blocks(
        source,
        bridge_dates,
    )

    if not blocks:
        return pd.DataFrame(
            columns=[
                "bridge_block_number",
                "first_bridge_date",
                "last_bridge_date",
                "bridge_session_count",
                "pre_bridge_date",
                "post_bridge_date",
                "last_bridge_close",
                "post_bridge_close",
                "expected_post_bridge_return_1d",
                "observed_post_bridge_return_1d",
                "expected_post_bridge_log_return_1d",
                "observed_post_bridge_log_return_1d",
                "post_bridge_uses_last_valid_bridge_close",
                "full_vs_aligned_features_exact",
            ]
        )

    rows = []

    for block in blocks:
        last_bridge_position = (
            block[
                "last_bridge_position"
            ]
        )

        post_bridge_position = (
            block[
                "post_bridge_position"
            ]
        )

        last_bridge_date = (
            block[
                "last_bridge_date"
            ]
        )

        post_bridge_date = (
            block[
                "post_bridge_date"
            ]
        )

        last_bridge_close = float(
            source.loc[
                last_bridge_position,
                "Close",
            ]
        )

        post_bridge_close = float(
            source.loc[
                post_bridge_position,
                "Close",
            ]
        )

        expected_return = (
            post_bridge_close
            / last_bridge_close
            - 1.0
        )

        expected_log_return = float(
            np.log(
                post_bridge_close
                / last_bridge_close
            )
        )

        full_row = full_features.loc[
            full_features["Date"]
            == post_bridge_date
        ]

        aligned_row = aligned_features.loc[
            aligned_features["Date"]
            == post_bridge_date
        ]

        if len(full_row) != 1:
            raise RuntimeError(
                "Could not uniquely identify the post-bridge row in the "
                "full NIFTY feature panel."
            )

        if len(aligned_row) != 1:
            raise RuntimeError(
                "Could not uniquely identify the post-bridge row in the "
                "aligned NIFTY feature panel."
            )

        observed_return = float(
            full_row[
                "nifty_return_1d"
            ].iloc[0]
        )

        observed_log_return = float(
            full_row[
                "nifty_log_return_1d"
            ].iloc[0]
        )

        return_match = bool(
            np.isclose(
                expected_return,
                observed_return,
                rtol=1e-14,
                atol=1e-14,
            )
        )

        log_return_match = bool(
            np.isclose(
                expected_log_return,
                observed_log_return,
                rtol=1e-14,
                atol=1e-14,
            )
        )

        if not return_match or not log_return_match:
            raise RuntimeError(
                "Post-bridge NIFTY return does not use the immediately "
                "preceding valid NIFTY observation."
            )

        full_values = (
            full_row[
                NIFTY_FEATURE_COLUMNS
            ]
            .iloc[0]
            .to_numpy(dtype=float)
        )

        aligned_values = (
            aligned_row[
                NIFTY_FEATURE_COLUMNS
            ]
            .iloc[0]
            .to_numpy(dtype=float)
        )

        exact_feature_match = bool(
            np.allclose(
                full_values,
                aligned_values,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        )

        if not exact_feature_match:
            raise RuntimeError(
                "Post-feature alignment changed post-bridge "
                "NIFTY feature values."
            )

        rows.append(
            {
                "bridge_block_number": (
                    block[
                        "bridge_block_number"
                    ]
                ),
                "first_bridge_date": (
                    block[
                        "first_bridge_date"
                    ]
                ),
                "last_bridge_date": (
                    last_bridge_date
                ),
                "bridge_session_count": (
                    block[
                        "bridge_session_count"
                    ]
                ),
                "pre_bridge_date": (
                    block[
                        "pre_bridge_date"
                    ]
                ),
                "post_bridge_date": (
                    post_bridge_date
                ),
                "last_bridge_close": (
                    last_bridge_close
                ),
                "post_bridge_close": (
                    post_bridge_close
                ),
                "expected_post_bridge_return_1d": (
                    expected_return
                ),
                "observed_post_bridge_return_1d": (
                    observed_return
                ),
                "expected_post_bridge_log_return_1d": (
                    expected_log_return
                ),
                "observed_post_bridge_log_return_1d": (
                    observed_log_return
                ),
                "post_bridge_uses_last_valid_bridge_close": True,
                "full_vs_aligned_features_exact": (
                    exact_feature_match
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 13. STEP-05 NIFTY PROVIDER-OHLC RECONCILIATION
# =============================================================================

def find_step05_nifty_parquet() -> Path:
    with STEP05_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    candidates = []

    for record in manifest.get("artifacts", {}).values():
        if not isinstance(record, dict):
            continue

        path_text = record.get("path")

        if not path_text:
            continue

        path = Path(path_text)

        if path.name == "05_nifty50_panel.parquet":
            candidates.append(path)

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not identify exactly one frozen Step-05 NIFTY "
            "Parquet panel."
        )

    return candidates[0]


def reconcile_step05_provider_ohlc(
    source: pd.DataFrame,
    final_calendar: pd.DataFrame,
) -> pd.DataFrame:
    step05_path = find_step05_nifty_parquet()

    step05 = pd.read_parquet(
        step05_path,
        engine="pyarrow",
    )

    step05["Date"] = (
        pd.to_datetime(
            step05["Date"],
            errors="raise",
        )
        .dt.normalize()
    )

    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        column
        for column in required
        if column not in step05.columns
    ]

    if missing:
        raise RuntimeError(
            "Frozen Step-05 NIFTY panel lacks provider OHLC columns: "
            + ", ".join(missing)
        )

    final_dates = set(
        final_calendar["Date"].tolist()
    )

    source_aligned = (
        source.loc[
            source["Date"].isin(
                final_dates
            ),
            required,
        ]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    step05_aligned = (
        step05[required]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if not source_aligned["Date"].equals(
        step05_aligned["Date"]
    ):
        raise RuntimeError(
            "Step-05 NIFTY dates do not match the final-calendar "
            "subset of the Step-03 NIFTY source."
        )

    rows = []

    for column in (
        "Open",
        "High",
        "Low",
        "Close",
    ):
        left = pd.to_numeric(
            source_aligned[column],
            errors="raise",
        ).to_numpy(dtype=float)

        right = pd.to_numeric(
            step05_aligned[column],
            errors="raise",
        ).to_numpy(dtype=float)

        difference = np.abs(
            left - right
        )

        max_difference = float(
            difference.max()
            if len(difference)
            else 0.0
        )

        exact_match = bool(
            np.array_equal(
                left,
                right,
                equal_nan=True,
            )
        )

        if not exact_match:
            raise RuntimeError(
                f"Step-03 raw-source / Step-05 NIFTY provider "
                f"{column} values differ."
            )

        rows.append(
            {
                "field": column,
                "rows_compared": int(
                    len(left)
                ),
                "maximum_absolute_difference": (
                    max_difference
                ),
                "exact_match": (
                    exact_match
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 14. STAGING / HASH HELPERS
# =============================================================================

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
        pd.to_datetime(
            csv_frame["Date"],
            errors="raise",
        )
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

    original_dates = (
        pd.to_datetime(
            original["Date"],
            errors="raise",
        )
        .dt.normalize()
    )

    reloaded_dates = (
        pd.to_datetime(
            reloaded["Date"],
            errors="raise",
        )
        .dt.normalize()
    )

    if not np.array_equal(
        original_dates.to_numpy(),
        reloaded_dates.to_numpy(),
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
        "relative_path": relative_to_experiment(
            final_path
        ),
        "size_bytes": int(
            staging_path.stat().st_size
        ),
        "sha256": sha256_file(
            staging_path
        ),
    }


# =============================================================================
# 15. SOURCE-CODE HASH AUDIT
# =============================================================================

def find_artifact_by_filename(
    manifest_path: Path,
    filename: str,
    *,
    artifact_keys: tuple[str, ...] = (
        "artifacts",
        "canonical_artifacts",
    ),
) -> Optional[Path]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    for artifact_key in artifact_keys:
        for record in manifest.get(
            artifact_key,
            {},
        ).values():
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
    current_step00 = Path(
        cfg.__file__
    ).resolve()

    frozen_step00 = (
        cfg.get_frozen_script_path()
        .resolve()
    )

    frozen_step01 = find_artifact_by_filename(
        STEP01_MANIFEST,
        "step01_environment_audit_frozen.py",
    )

    frozen_step02 = find_artifact_by_filename(
        STEP02_MANIFEST,
        "step02_download_market_data_frozen.py",
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

    frozen_step06 = find_artifact_by_filename(
        STEP06_MANIFEST,
        "step06_stock_features_frozen.py",
    )

    sources: list[tuple[str, Path]] = [
        (
            "current_step00",
            current_step00,
        ),
        (
            "frozen_step00",
            frozen_step00,
        ),
        (
            "current_step07",
            SCRIPT_FILE.resolve(),
        ),
    ]

    for role, path in (
        (
            "frozen_step01",
            frozen_step01,
        ),
        (
            "frozen_step02",
            frozen_step02,
        ),
        (
            "frozen_step03",
            frozen_step03,
        ),
        (
            "frozen_step04",
            frozen_step04,
        ),
        (
            "frozen_step05",
            frozen_step05,
        ),
        (
            "frozen_step06",
            frozen_step06,
        ),
    ):
        if path is not None:
            sources.append(
                (
                    role,
                    path,
                )
            )

    rows = []

    for role, path in sources:
        if not path.exists():
            raise RuntimeError(
                f"Required source file missing: {path}"
            )

        rows.append(
            {
                "role": role,
                "file_name": path.name,
                "absolute_path": str(path),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(
                    path
                ),
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
# 16. MANIFEST / VERIFY
# =============================================================================

def step07_artifact_paths() -> tuple[Path, ...]:
    return (
        FULL_SOURCE_FEATURE_PARQUET,
        FULL_SOURCE_FEATURE_CSV,
        ALIGNED_FEATURE_PARQUET,
        ALIGNED_FEATURE_CSV,
        COMPLETE_ALIGNED_PARQUET,
        COMPLETE_ALIGNED_CSV,
        FEATURE_DICTIONARY_CSV,
        WARMUP_AUDIT_CSV,
        CAUSALITY_AUDIT_CSV,
        BRIDGE_AUDIT_CSV,
        ALIGNMENT_AUDIT_CSV,
        STEP05_OHLC_RECONCILIATION_CSV,
        HASH_REGISTER_CSV,
        SOURCE_HASH_CSV,
        STEP07_SUMMARY_JSON,
        FROZEN_STEP07_SCRIPT,
        SUMMARY_TXT,
    )


def write_manifest(
    *,
    step00: Dict[str, Any],
    step01: Dict[str, Any],
    step02: Dict[str, Any],
    step03: Dict[str, Any],
    step04: Dict[str, Any],
    step05: Dict[str, Any],
    step06: Dict[str, Any],
) -> str:
    artifacts = {
        path.name: artifact_record(
            path
        )
        for path in step07_artifact_paths()
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
            step00[
                "configuration_fingerprint_sha256"
            ]
        ),
        "step00_manifest_sha256": (
            step00[
                "step00_manifest_sha256"
            ]
        ),
        "step01_manifest_sha256": (
            step01[
                "manifest_sha256"
            ]
        ),
        "step02_manifest_sha256": (
            step02[
                "manifest_sha256"
            ]
        ),
        "step03_manifest_sha256": (
            step03[
                "manifest_sha256"
            ]
        ),
        "step04_manifest_sha256": (
            step04[
                "manifest_sha256"
            ]
        ),
        "step05_manifest_sha256": (
            step05[
                "manifest_sha256"
            ]
        ),
        "step06_manifest_sha256": (
            step06[
                "manifest_sha256"
            ]
        ),
        "raw_nifty_sha256": (
            step02[
                "raw_nifty_sha256"
            ]
        ),
        "final_test_state": (
            step00[
                "final_test_state"
            ]
        ),
        "artifacts": artifacts,
    }

    atomic_write_json(
        MANIFEST_JSON,
        manifest,
    )

    manifest_sha = sha256_file(
        MANIFEST_JSON
    )

    atomic_write_text(
        MANIFEST_SHA,
        f"{manifest_sha}  {MANIFEST_JSON.name}\n",
    )

    return manifest_sha


def verify_existing_step07() -> int:
    print("=" * 126)
    print("STEP 07 — VERIFY EXISTING NIFTY-CONTEXT FEATURE FREEZE")
    print("=" * 126)

    step00 = verify_step00_contract()

    step01 = verify_step01(
        step00[
            "configuration_fingerprint_sha256"
        ]
    )

    step02 = verify_step02(
        step00[
            "configuration_fingerprint_sha256"
        ],
        step01[
            "manifest_sha256"
        ],
    )

    step03 = verify_step03(
        step00[
            "configuration_fingerprint_sha256"
        ],
        step01[
            "manifest_sha256"
        ],
        step02[
            "manifest_sha256"
        ],
    )

    step04 = verify_step04(
        step00[
            "configuration_fingerprint_sha256"
        ],
        step01[
            "manifest_sha256"
        ],
        step02[
            "manifest_sha256"
        ],
        step03[
            "manifest_sha256"
        ],
    )

    step05 = verify_step05(
        step00[
            "configuration_fingerprint_sha256"
        ],
        step01[
            "manifest_sha256"
        ],
        step02[
            "manifest_sha256"
        ],
        step03[
            "manifest_sha256"
        ],
        step04[
            "manifest_sha256"
        ],
    )

    step06 = verify_step06(
        step00[
            "configuration_fingerprint_sha256"
        ],
        step01[
            "manifest_sha256"
        ],
        step02[
            "manifest_sha256"
        ],
        step03[
            "manifest_sha256"
        ],
        step04[
            "manifest_sha256"
        ],
        step05[
            "manifest_sha256"
        ],
    )

    if not MANIFEST_JSON.exists():
        print("[FAIL] Step-07 manifest missing.")
        return 20

    if not MANIFEST_SHA.exists():
        print("[FAIL] Step-07 manifest SHA sidecar missing.")
        return 21

    actual_manifest_sha = verify_sha_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 07",
    )

    with MANIFEST_JSON.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("step", {}).get("status") != "PASS":
        print("[FAIL] Stored Step-07 status is not PASS.")
        return 22

    expected_chain = {
        "configuration_fingerprint_sha256": (
            step00[
                "configuration_fingerprint_sha256"
            ]
        ),
        "step01_manifest_sha256": (
            step01[
                "manifest_sha256"
            ]
        ),
        "step02_manifest_sha256": (
            step02[
                "manifest_sha256"
            ]
        ),
        "step03_manifest_sha256": (
            step03[
                "manifest_sha256"
            ]
        ),
        "step04_manifest_sha256": (
            step04[
                "manifest_sha256"
            ]
        ),
        "step05_manifest_sha256": (
            step05[
                "manifest_sha256"
            ]
        ),
        "step06_manifest_sha256": (
            step06[
                "manifest_sha256"
            ]
        ),
        "raw_nifty_sha256": (
            step02[
                "raw_nifty_sha256"
            ]
        ),
    }

    for field, expected in expected_chain.items():
        if manifest.get(field) != expected:
            print(
                f"[FAIL] Step-07 chain mismatch: {field}"
            )
            return 23

    failures = []

    for name, record in manifest.get(
        "artifacts",
        {},
    ).items():
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

        if sha256_file(path) != record["sha256"]:
            failures.append(
                f"hash mismatch: {name}"
            )

    if failures:
        for failure in failures:
            print(
                f"[FAIL] {failure}"
            )

        print("=" * 126)
        print("STEP 07 VERIFICATION STATUS: FAIL")
        print("=" * 126)
        return 24

    print(
        "[PASS] Configuration fingerprint : "
        f"{step00['configuration_fingerprint_sha256']}"
    )
    print(
        "[PASS] Step-06 manifest          : "
        f"{step06['manifest_sha256']}"
    )
    print(
        "[PASS] Raw NIFTY SHA-256         : "
        f"{step02['raw_nifty_sha256']}"
    )
    print(
        "[PASS] Step-07 manifest          : "
        f"{actual_manifest_sha}"
    )
    print(
        "[PASS] Final-test state          : "
        f"{step00['final_test_state']}"
    )
    print("=" * 126)
    print("STEP 07 VERIFICATION STATUS: PASS")
    print("=" * 126)

    return 0


# =============================================================================
# 17. FAILURE DIAGNOSTIC
# =============================================================================

def write_failure_diagnostic(
    *,
    stage: str,
    error: Exception,
    staging_root: Optional[Path],
    archived_partial: Optional[Path],
) -> Path:
    FAILED_ATTEMPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )

    path = (
        FAILED_ATTEMPT_DIR
        / f"07_nifty_context_FAILED_{timestamp}.json"
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
        "canonical_manifest_exists": (
            MANIFEST_JSON.exists()
        ),
    }

    atomic_write_json(
        path,
        payload,
    )

    return path


# =============================================================================
# 18. MAIN
# =============================================================================

def run_new_step07() -> int:
    print("=" * 126)
    print(STEP_NAME)
    print("=" * 126)

    stage = "prerequisite_verification"
    staging_root: Optional[Path] = None

    try:
        print("[1/12] Verifying Step 00 and final-test lock...")

        step00 = verify_step00_contract()

        print(
            "       Configuration fingerprint: "
            f"{step00['configuration_fingerprint_sha256']}"
        )
        print(
            "       Final-test state: "
            f"{step00['final_test_state']}"
        )

        print("[2/12] Verifying Step 01 through Step 06...")

        step01 = verify_step01(
            step00[
                "configuration_fingerprint_sha256"
            ]
        )

        step02 = verify_step02(
            step00[
                "configuration_fingerprint_sha256"
            ],
            step01[
                "manifest_sha256"
            ],
        )

        step03 = verify_step03(
            step00[
                "configuration_fingerprint_sha256"
            ],
            step01[
                "manifest_sha256"
            ],
            step02[
                "manifest_sha256"
            ],
        )

        step04 = verify_step04(
            step00[
                "configuration_fingerprint_sha256"
            ],
            step01[
                "manifest_sha256"
            ],
            step02[
                "manifest_sha256"
            ],
            step03[
                "manifest_sha256"
            ],
        )

        step05 = verify_step05(
            step00[
                "configuration_fingerprint_sha256"
            ],
            step01[
                "manifest_sha256"
            ],
            step02[
                "manifest_sha256"
            ],
            step03[
                "manifest_sha256"
            ],
            step04[
                "manifest_sha256"
            ],
        )

        step06 = verify_step06(
            step00[
                "configuration_fingerprint_sha256"
            ],
            step01[
                "manifest_sha256"
            ],
            step02[
                "manifest_sha256"
            ],
            step03[
                "manifest_sha256"
            ],
            step04[
                "manifest_sha256"
            ],
            step05[
                "manifest_sha256"
            ],
        )

        print(
            "       Step-06 manifest: "
            f"{step06['manifest_sha256']}"
        )
        print(
            "       Raw NIFTY SHA-256: "
            f"{step02['raw_nifty_sha256']}"
        )

        stage = "output_collision_guard"

        print("[3/12] Checking immutable Step-07 output target...")

        assert_new_step07_target()
        cfg.create_directories()

        stage = "load_source_and_calendars"

        print("[4/12] Loading full price-valid NIFTY source and frozen calendars...")

        bridge_dates = load_bridge_dates(
            step04[
                "policy"
            ]
        )

        source, candidate_calendar = (
            load_candidate_nifty_source(
                bridge_dates
            )
        )

        stock_calendar = (
            load_final_stock_calendar()
        )

        calendar_relation = (
            validate_calendar_relation(
                source=source,
                stock_calendar=stock_calendar,
                bridge_dates=bridge_dates,
            )
        )

        print(
            "       Full price-valid NIFTY sessions: "
            f"{len(source):,}"
        )
        print(
            "       Final stock-analysis sessions : "
            f"{len(stock_calendar):,}"
        )
        print(
            "       Valid NIFTY bridge sessions   : "
            + (
                ", ".join(
                    date.strftime("%Y-%m-%d")
                    for date in bridge_dates
                )
                if bridge_dates
                else "NONE"
            )
        )
        print(
            "       NIFTY volume predictors       : PROHIBITED"
        )

        stage = "feature_construction"

        print("[5/12] Constructing exactly 16 NIFTY features on full valid-index history...")

        source_features = (
            construct_nifty_features(
                source
            )
        )

        source_summary = (
            validate_feature_panel(
                source_features,
                expected_rows=len(source),
                panel_name="full_nifty_calendar",
                expect_warmup_pattern=True,
            )
        )

        expected_complete_source_rows = (
            len(source)
            - EXPECTED_FIRST_COMPLETE_POSITION
        )

        if (
            source_summary[
                "complete_rows"
            ]
            != expected_complete_source_rows
        ):
            raise RuntimeError(
                "Unexpected complete-row count on full NIFTY source."
            )

        print(
            "       Feature count           : "
            f"{len(NIFTY_FEATURE_COLUMNS)}"
        )
        print(
            "       Warm-up missing values  : "
            f"{source_summary['feature_missing_values']}"
        )
        print(
            "       First complete position : "
            f"{source_summary['first_complete_position']}"
        )
        print(
            "       Complete source rows    : "
            f"{source_summary['complete_rows']:,}"
        )

        stage = "post_feature_alignment"

        print("[6/12] Aligning completed NIFTY context to final stock calendar...")

        aligned = (
            stock_calendar[
                [
                    "Date",
                ]
            ]
            .merge(
                source_features,
                on="Date",
                how="left",
                validate="one_to_one",
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if aligned["index_ticker"].isna().any():
            missing_dates = (
                aligned.loc[
                    aligned[
                        "index_ticker"
                    ].isna(),
                    "Date",
                ]
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )

            raise RuntimeError(
                "Missing NIFTY feature rows after final-calendar alignment:\n"
                + "\n".join(missing_dates)
            )

        aligned_summary = (
            validate_feature_panel(
                aligned,
                expected_rows=len(stock_calendar),
                panel_name="stock_aligned_nifty",
                expect_warmup_pattern=True,
            )
        )

        expected_complete_aligned_rows = (
            len(stock_calendar)
            - EXPECTED_FIRST_COMPLETE_POSITION
        )

        if (
            aligned_summary[
                "complete_rows"
            ]
            != expected_complete_aligned_rows
        ):
            raise RuntimeError(
                "Unexpected complete-row count after final-calendar alignment."
            )

        aligned_complete_mask = (
            aligned[
                NIFTY_FEATURE_COLUMNS
            ]
            .notna()
            .all(axis=1)
        )

        aligned_complete = (
            aligned.loc[
                aligned_complete_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        print(
            "       Aligned rows           : "
            f"{len(aligned):,}"
        )
        print(
            "       Complete aligned rows  : "
            f"{len(aligned_complete):,}"
        )

        stage = "causality_bridge_reconciliation"

        print("[7/12] Running causality, bridge and Step-05 OHLC reconciliations...")

        source_warmup = (
            build_warmup_audit(
                source_features,
                "full_nifty_calendar",
            )
        )

        aligned_warmup = (
            build_warmup_audit(
                aligned,
                "stock_aligned_nifty",
            )
        )

        warmup_audit = pd.concat(
            [
                source_warmup,
                aligned_warmup,
            ],
            ignore_index=True,
        )

        causality_audit = (
            run_prefix_causality_audit(
                source=source,
                full_features=source_features,
            )
        )

        bridge_audit = (
            build_bridge_audit(
                source=source,
                full_features=source_features,
                aligned_features=aligned,
                bridge_dates=bridge_dates,
            )
        )

        step05_ohlc_reconciliation = (
            reconcile_step05_provider_ohlc(
                source=source,
                final_calendar=stock_calendar,
            )
        )

        if not causality_audit[
            "all_features_match"
        ].all():
            raise RuntimeError(
                "NIFTY prefix-causality audit contains failures."
            )

        if (
            not step05_ohlc_reconciliation.empty
            and
            not step05_ohlc_reconciliation[
                "exact_match"
            ].all()
        ):
            raise RuntimeError(
                "Step-05 NIFTY OHLC reconciliation failed."
            )

        print(
            "       Prefix-causality tests : "
            f"{len(causality_audit)}"
        )
        print(
            "       Maximum causality diff : "
            f"{causality_audit['maximum_absolute_difference'].max():.3e}"
        )
        print(
            "       Bridge blocks audited  : "
            f"{len(bridge_audit)}"
        )
        print(
            "       Step-05 OHLC fields matched: "
            f"{int(step05_ohlc_reconciliation['exact_match'].sum())}"
            f"/{len(step05_ohlc_reconciliation)}"
        )

        stage = "feature_dictionary_alignment_audit"

        print("[8/12] Building feature dictionary and alignment audit...")

        feature_dictionary = (
            build_feature_dictionary()
        )

        if len(feature_dictionary) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                "NIFTY feature dictionary is not exactly 16 rows."
            )

        source_dates = pd.DatetimeIndex(
            source_features["Date"]
        )

        aligned_dates = pd.DatetimeIndex(
            aligned["Date"]
        )

        source_only = (
            source_dates.difference(
                aligned_dates
            )
        )

        alignment_audit = pd.DataFrame(
            [
                {
                    "source_rows": int(
                        len(source_features)
                    ),
                    "aligned_rows": int(
                        len(aligned)
                    ),
                    "source_only_date_count": int(
                        len(source_only)
                    ),
                    "source_only_dates": (
                        "|".join(
                            date.strftime("%Y-%m-%d")
                            for date in source_only
                        )
                    ),
                    "stock_dates_missing_from_source": (
                        calendar_relation[
                            "stock_dates_missing_from_source"
                        ]
                    ),
                    "aligned_calendar_exact_match": bool(
                        aligned_dates.equals(
                            pd.DatetimeIndex(
                                stock_calendar["Date"]
                            )
                        )
                    ),
                    "feature_construction_before_alignment": True,
                    "bridge_dates_retained_during_feature_construction": True,
                }
            ]
        )

        if not bool(
            alignment_audit[
                "aligned_calendar_exact_match"
            ].iloc[0]
        ):
            raise RuntimeError(
                "Aligned NIFTY dates do not exactly match "
                "the Step-04 final stock calendar."
            )

        stage = "staging_outputs"

        print("[9/12] Writing feature panels to staging area...")

        staging_root = create_staging_root()

        staged_full_parquet = (
            staging_root
            / FULL_SOURCE_FEATURE_PARQUET.name
        )
        staged_full_csv = (
            staging_root
            / FULL_SOURCE_FEATURE_CSV.name
        )
        staged_aligned_parquet = (
            staging_root
            / ALIGNED_FEATURE_PARQUET.name
        )
        staged_aligned_csv = (
            staging_root
            / ALIGNED_FEATURE_CSV.name
        )
        staged_complete_parquet = (
            staging_root
            / COMPLETE_ALIGNED_PARQUET.name
        )
        staged_complete_csv = (
            staging_root
            / COMPLETE_ALIGNED_CSV.name
        )

        write_pair_staged(
            source_features,
            staged_full_parquet,
            staged_full_csv,
        )

        verify_parquet_roundtrip(
            source_features,
            staged_full_parquet,
        )

        write_pair_staged(
            aligned,
            staged_aligned_parquet,
            staged_aligned_csv,
        )

        verify_parquet_roundtrip(
            aligned,
            staged_aligned_parquet,
        )

        write_pair_staged(
            aligned_complete,
            staged_complete_parquet,
            staged_complete_csv,
        )

        verify_parquet_roundtrip(
            aligned_complete,
            staged_complete_parquet,
        )

        staged_hashes = [
            staged_hash_record(
                scope="full_nifty_feature_calendar",
                file_format="parquet",
                staging_path=staged_full_parquet,
                final_path=FULL_SOURCE_FEATURE_PARQUET,
            ),
            staged_hash_record(
                scope="full_nifty_feature_calendar",
                file_format="csv",
                staging_path=staged_full_csv,
                final_path=FULL_SOURCE_FEATURE_CSV,
            ),
            staged_hash_record(
                scope="stock_aligned_nifty_features",
                file_format="parquet",
                staging_path=staged_aligned_parquet,
                final_path=ALIGNED_FEATURE_PARQUET,
            ),
            staged_hash_record(
                scope="stock_aligned_nifty_features",
                file_format="csv",
                staging_path=staged_aligned_csv,
                final_path=ALIGNED_FEATURE_CSV,
            ),
            staged_hash_record(
                scope="complete_aligned_nifty_features",
                file_format="parquet",
                staging_path=staged_complete_parquet,
                final_path=COMPLETE_ALIGNED_PARQUET,
            ),
            staged_hash_record(
                scope="complete_aligned_nifty_features",
                file_format="csv",
                staging_path=staged_complete_csv,
                final_path=COMPLETE_ALIGNED_CSV,
            ),
        ]

        stage = "promote_and_freeze"

        print("[10/12] Promoting Step-07 feature panels and writing audits...")

        if STEP07_OUTPUT_DIR.exists():
            raise RuntimeError(
                "Canonical Step-07 output directory appeared unexpectedly."
            )

        os.replace(
            staging_root,
            STEP07_OUTPUT_DIR,
        )

        staging_root = None

        for record in staged_hashes:
            path = Path(
                record[
                    "file_path"
                ]
            )

            if not path.exists():
                raise RuntimeError(
                    f"Promoted Step-07 output missing: {path}"
                )

            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(
                    f"Step-07 output hash changed during promotion: {path}"
                )

        source_hashes = collect_source_hashes()

        write_dataframe_csv(
            FEATURE_DICTIONARY_CSV,
            feature_dictionary,
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
            BRIDGE_AUDIT_CSV,
            bridge_audit,
        )

        write_dataframe_csv(
            ALIGNMENT_AUDIT_CSV,
            alignment_audit,
        )

        write_dataframe_csv(
            STEP05_OHLC_RECONCILIATION_CSV,
            step05_ohlc_reconciliation,
        )

        supporting_paths = [
            (
                "feature_dictionary",
                FEATURE_DICTIONARY_CSV,
            ),
            (
                "warmup_audit",
                WARMUP_AUDIT_CSV,
            ),
            (
                "causality_audit",
                CAUSALITY_AUDIT_CSV,
            ),
            (
                "bridge_audit",
                BRIDGE_AUDIT_CSV,
            ),
            (
                "alignment_audit",
                ALIGNMENT_AUDIT_CSV,
            ),
            (
                "step05_ohlc_reconciliation",
                STEP05_OHLC_RECONCILIATION_CSV,
            ),
        ]

        output_hash_records = list(
            staged_hashes
        )

        for scope, path in supporting_paths:
            output_hash_records.append(
                {
                    "scope": scope,
                    "file_format": (
                        path.suffix
                        .lstrip(".")
                    ),
                    "file_path": str(path),
                    "relative_path": (
                        relative_to_experiment(
                            path
                        )
                    ),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                    "sha256": (
                        sha256_file(
                            path
                        )
                    ),
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

        stage = "summary_manifest_freeze"

        print("[11/12] Freezing Step-07 summary, source snapshot and manifest...")

        lock_sha_after = sha256_file(
            cfg.get_final_test_lock_path()
        )

        if (
            lock_sha_after
            != step00[
                "final_test_lock_sha256"
            ]
        ):
            raise RuntimeError(
                "Final-test lock changed during Step 07."
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
                step00[
                    "configuration_fingerprint_sha256"
                ]
            ),
            "upstream_chain": {
                "step00_manifest_sha256": (
                    step00[
                        "step00_manifest_sha256"
                    ]
                ),
                "step01_manifest_sha256": (
                    step01[
                        "manifest_sha256"
                    ]
                ),
                "step02_manifest_sha256": (
                    step02[
                        "manifest_sha256"
                    ]
                ),
                "step03_manifest_sha256": (
                    step03[
                        "manifest_sha256"
                    ]
                ),
                "step04_manifest_sha256": (
                    step04[
                        "manifest_sha256"
                    ]
                ),
                "step05_manifest_sha256": (
                    step05[
                        "manifest_sha256"
                    ]
                ),
                "step06_manifest_sha256": (
                    step06[
                        "manifest_sha256"
                    ]
                ),
                "raw_nifty_sha256": (
                    step02[
                        "raw_nifty_sha256"
                    ]
                ),
            },
            "feature_design": {
                "feature_count": EXPECTED_FEATURE_COUNT,
                "feature_groups": {
                    group: len(features)
                    for group, features in NIFTY_FEATURE_GROUPS.items()
                },
                "feature_order": NIFTY_FEATURE_COLUMNS,
                "volume_used": False,
                "volume_derived_features_used": False,
                "scaling_applied": False,
                "targets_constructed": False,
                "signal_timing_policy": (
                    SIGNAL_TIMING_POLICY
                ),
            },
            "temporal_policy": {
                "feature_source_calendar": (
                    "Step-03 complete price-valid NIFTY calendar"
                ),
                "feature_source_rows": int(
                    len(source_features)
                ),
                "final_alignment_calendar": (
                    "Step-04 final stock analysis calendar"
                ),
                "aligned_rows": int(
                    len(aligned)
                ),
                "valid_nifty_sessions_excluded_from_stock_calendar": [
                    date.strftime("%Y-%m-%d")
                    for date in bridge_dates
                ],
                "feature_construction_before_stock_alignment": True,
                "valid_nifty_exclusion_dates_used_as_historical_context": True,
                "nifty_context_not_artificially_restarted_at_stock_only_anomaly": True,
            },
            "warmup": {
                "full_source_missing_values": (
                    source_summary[
                        "feature_missing_values"
                    ]
                ),
                "first_complete_position": (
                    source_summary[
                        "first_complete_position"
                    ]
                ),
                "complete_source_rows": (
                    source_summary[
                        "complete_rows"
                    ]
                ),
                "complete_aligned_rows": (
                    aligned_summary[
                        "complete_rows"
                    ]
                ),
            },
            "causality_audit": {
                "tests": int(
                    len(causality_audit)
                ),
                "all_passed": bool(
                    causality_audit[
                        "all_features_match"
                    ].all()
                ),
                "maximum_absolute_difference": float(
                    causality_audit[
                        "maximum_absolute_difference"
                    ].max()
                ),
            },
            "bridge_audit": {
                "bridge_blocks": int(
                    len(bridge_audit)
                ),
                "all_post_bridge_returns_use_last_valid_nifty_observation": bool(
                    bridge_audit[
                        "post_bridge_uses_last_valid_bridge_close"
                    ].all()
                    if not bridge_audit.empty
                    else True
                ),
                "all_full_vs_aligned_post_bridge_features_exact": bool(
                    bridge_audit[
                        "full_vs_aligned_features_exact"
                    ].all()
                    if not bridge_audit.empty
                    else True
                ),
            },
            "step05_provider_ohlc_reconciliation": {
                "fields_checked": int(
                    len(
                        step05_ohlc_reconciliation
                    )
                ),
                "all_exact_match": bool(
                    step05_ohlc_reconciliation[
                        "exact_match"
                    ].all()
                ),
            },
            "scientific_boundary": {
                "stock_features_modified": False,
                "63_feature_merge_performed": False,
                "forecast_targets_constructed": False,
                "scalers_fit": False,
                "regimes_estimated": False,
                "sequences_constructed": False,
                "model_selection_performed": False,
                "threshold_selection_performed": False,
                "final_test_scoring_performed": False,
            },
            "final_test_lock_integrity": {
                "state": (
                    step00[
                        "final_test_state"
                    ]
                ),
                "lock_sha256_before": (
                    step00[
                        "final_test_lock_sha256"
                    ]
                ),
                "lock_sha256_after": (
                    lock_sha_after
                ),
                "lock_unchanged": True,
            },
        }

        atomic_write_json(
            STEP07_SUMMARY_JSON,
            summary_payload,
        )

        shutil.copy2(
            SCRIPT_FILE,
            FROZEN_STEP07_SCRIPT,
        )

        summary_lines = [
            STEP_NAME,
            "=" * 94,
            "Status: PASS",
            (
                "Configuration fingerprint: "
                + step00[
                    "configuration_fingerprint_sha256"
                ]
            ),
            (
                "Raw NIFTY SHA-256: "
                + step02[
                    "raw_nifty_sha256"
                ]
            ),
            (
                "Full price-valid NIFTY sessions: "
                f"{len(source_features)}"
            ),
            (
                "Final stock-aligned sessions: "
                f"{len(aligned)}"
            ),
            (
                "NIFTY feature count: "
                f"{EXPECTED_FEATURE_COUNT}"
            ),
            (
                "Warm-up missing values: "
                f"{source_summary['feature_missing_values']}"
            ),
            (
                "First complete position: "
                f"{source_summary['first_complete_position']}"
            ),
            (
                "Complete source rows: "
                f"{source_summary['complete_rows']}"
            ),
            (
                "Complete aligned rows: "
                f"{aligned_summary['complete_rows']}"
            ),
            (
                "Valid-NIFTY bridge dates: "
                + (
                    ",".join(
                        date.strftime("%Y-%m-%d")
                        for date in bridge_dates
                    )
                    if bridge_dates
                    else "NONE"
                )
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

        manifest_sha = write_manifest(
            step00=step00,
            step01=step01,
            step02=step02,
            step03=step03,
            step04=step04,
            step05=step05,
            step06=step06,
        )

        stage = "final_report"

        print("[12/12] Final Step-07 report...")
        print()
        print("-" * 126)
        print("CAUSAL NIFTY-50 CONTEXT PANEL")
        print("-" * 126)
        print(
            "Full price-valid NIFTY rows    : "
            f"{len(source_features):,}"
        )
        print(
            "Stock-aligned NIFTY rows       : "
            f"{len(aligned):,}"
        )
        print(
            "NIFTY context features         : "
            f"{len(NIFTY_FEATURE_COLUMNS)}"
        )
        print(
            "Total warm-up missing values   : "
            f"{source_summary['feature_missing_values']}"
        )
        print(
            "First complete position        : "
            f"{source_summary['first_complete_position']}"
        )
        print(
            "First complete date            : "
            f"{pd.Timestamp(source_summary['first_complete_date']).date()}"
        )
        print(
            "Complete source rows           : "
            f"{source_summary['complete_rows']:,}"
        )
        print(
            "Complete aligned rows          : "
            f"{aligned_summary['complete_rows']:,}"
        )
        print(
            "Prefix-causality tests         : "
            f"{len(causality_audit)}"
        )
        print(
            "Prefix-causality all pass      : "
            f"{causality_audit['all_features_match'].all()}"
        )
        print(
            "Maximum causality difference   : "
            f"{causality_audit['maximum_absolute_difference'].max():.3e}"
        )
        print(
            "NIFTY volume feature count     : 0"
        )
        print(
            "Scaling applied                : False"
        )
        print(
            "Targets constructed            : False"
        )

        print("-" * 126)
        print("VALID-NIFTY BRIDGE AUDIT")
        print("-" * 126)

        if bridge_audit.empty:
            print(
                "No price-valid NIFTY session was excluded from the "
                "final stock calendar."
            )
        else:
            for row in bridge_audit.itertuples(index=False):
                print(
                    f"Block {row.bridge_block_number}: "
                    f"{pd.Timestamp(row.first_bridge_date).date()} -> "
                    f"{pd.Timestamp(row.last_bridge_date).date()} | "
                    f"bridge sessions={row.bridge_session_count} | "
                    f"next retained={pd.Timestamp(row.post_bridge_date).date()} | "
                    f"return check="
                    f"{row.post_bridge_uses_last_valid_bridge_close} | "
                    f"feature match="
                    f"{row.full_vs_aligned_features_exact}"
                )

        print("-" * 126)
        print("FEATURE GROUPS")
        print("-" * 126)

        for group_name, features in NIFTY_FEATURE_GROUPS.items():
            print(
                f"{group_name:<32}: "
                f"{len(features):>2}"
            )

        print("-" * 126)
        print("OUTPUT FILES")
        print("-" * 126)
        print(
            f"Full NIFTY features : {FULL_SOURCE_FEATURE_PARQUET}"
        )
        print(
            f"Aligned NIFTY       : {ALIGNED_FEATURE_PARQUET}"
        )
        print(
            f"Complete aligned    : {COMPLETE_ALIGNED_PARQUET}"
        )
        print(
            f"Feature dictionary  : {FEATURE_DICTIONARY_CSV}"
        )
        print(
            f"Bridge audit        : {BRIDGE_AUDIT_CSV}"
        )
        print(
            f"Causality audit     : {CAUSALITY_AUDIT_CSV}"
        )
        print(
            f"Step-05 OHLC audit  : {STEP05_OHLC_RECONCILIATION_CSV}"
        )
        print(
            f"Manifest            : {MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256    : {manifest_sha}"
        )
        print(
            f"Final-test state    : {step00['final_test_state']}"
        )
        print("=" * 126)
        print(
            "PASS: Exactly 16 price-only NIFTY context features constructed."
        )
        print(
            "PASS: NIFTY features were constructed on the complete "
            "price-valid NIFTY calendar before stock-calendar alignment."
        )
        print(
            "PASS: Valid NIFTY observations excluded only from the stock "
            "calendar remain available as causal historical market context."
        )
        print(
            "PASS: NIFTY volume and all volume-derived context variables "
            "are excluded."
        )
        print(
            "PASS: Prefix recomputation confirms NIFTY feature causality."
        )
        print(
            "PASS: No targets, scaling, regimes, sequences, or models were created."
        )
        print(
            "PASS: Final held-out test remains LOCKED_NOT_OPENED."
        )
        print("=" * 126)
        print(
            "Do not run Step 08 yet. Paste the complete terminal output for audit."
        )

        return 0

    except Exception as exc:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(
                staging_root,
                ignore_errors=True,
            )

        archived_partial = (
            archive_partial_outputs()
        )

        failure_path = (
            write_failure_diagnostic(
                stage=stage,
                error=exc,
                staging_root=staging_root,
                archived_partial=archived_partial,
            )
        )

        print()
        print("=" * 126)
        print("STEP 07 FAILED — NIFTY FEATURES NOT FROZEN")
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
        print("Do not proceed to Step 08.")

        return 30


# =============================================================================
# 19. CLI / ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct and freeze the clean-V3 causal 16-feature "
            "NIFTY context panel."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an already frozen Step-07 context panel "
            "without recomputing it."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verify_only:
        return verify_existing_step07()

    return run_new_step07()


if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print(
            "\nStep 07 interrupted by user."
        )
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 126)
        print("STEP 07 UNHANDLED FAILURE")
        print("=" * 126)
        print(f"{type(exc).__name__}: {exc}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        print("=" * 126)
        sys.exit(1)
