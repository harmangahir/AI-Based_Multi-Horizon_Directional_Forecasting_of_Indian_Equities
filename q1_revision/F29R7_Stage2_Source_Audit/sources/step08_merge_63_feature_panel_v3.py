#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 08 — MERGE 47 STOCK + 16 NIFTY FEATURES INTO LOCKED 63-FEATURE PANEL
(CLEAN V3)

Purpose
-------
Merge the frozen Step-06 stock-feature panel and the frozen Step-07
stock-aligned NIFTY context panel into one immutable 63-feature panel.

Inputs
------
Step 06:
    47 causal stock-specific predictors.

Step 07:
    16 causal price-only NIFTY-50 context predictors.

Merge policy
------------
Exact Date many-to-one merge:
    many stock rows on date t -> exactly one NIFTY context row on date t

No forward fill.
No backward fill.
No interpolation.
No imputation.
No target construction.
No scaling.
No regime construction.
No feature selection.

Critical complete-case policy
-----------------------------
The complete 63-feature panel is formed ONLY AFTER the 47 + 16 merge.

The resulting complete ticker-date keys are then required to match the frozen
Step-06 complete-stock-feature keys exactly. This is not assumed in advance:
it is verified computationally. For the current experiment, Step-07 NIFTY
warm-up ends earlier than the stock-feature warm-up and NIFTY context does not
restart at the stock-only anomaly date, so no additional complete-case rows
should be lost at Step 08.

Bridge-context policy
---------------------
Step 07 constructs NIFTY features on the complete price-valid NIFTY calendar
before aligning to the final stock calendar. Step 08 must preserve those
already-constructed context values exactly. Any post-bridge dates recorded by
the frozen Step-07 bridge audit are checked dynamically; no bridge date is
hard-coded here.

Scientific boundary
-------------------
This stage:
- fixes the ordered 63-feature interface;
- performs no scaling;
- constructs no target;
- estimates no regime;
- creates no sequence;
- trains/selects no model;
- tunes no threshold;
- does not score the held-out final test.

Run
---
python step08_merge_63_feature_panel_v3.py

Verify frozen Step 08
---------------------
python step08_merge_63_feature_panel_v3.py --verify-only
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
        "step08_merge_63_feature_panel_v3.py must be located inside "
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

STEP_NUMBER = 8
STEP_ID = "08"
STEP_NAME = "STEP 08 — LOCKED 63-FEATURE STOCK + NIFTY PANEL"
SCHEMA_VERSION = "1.0"

EXPECTED_STOCK_FEATURES = 47
EXPECTED_NIFTY_FEATURES = 16
EXPECTED_TOTAL_FEATURES = 63
EXPECTED_STOCK_COUNT = len(cfg.STOCK_TICKERS)

SIGNAL_TIMING_POLICY = (
    "all session-t predictors are available only after session-t close"
)

PROHIBITED_EXACT_COLUMNS = {
    "target",
    "label",
    "y",
    "y_1d",
    "y_3d",
    "y_5d",
    "future_return",
    "future_close",
    "future_price",
    "target_1d",
    "target_3d",
    "target_5d",
}


# =============================================================================
# 2. FROZEN UPSTREAM PATHS
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

STEP05_MANIFEST = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.json"
STEP05_MANIFEST_SHA = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.sha256"

STEP06_MANIFEST = cfg.MANIFEST_DIR / "06_stock_feature_manifest.json"
STEP06_MANIFEST_SHA = cfg.MANIFEST_DIR / "06_stock_feature_manifest.sha256"

STEP07_MANIFEST = cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.json"
STEP07_MANIFEST_SHA = cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.sha256"

STEP06_PANEL = (
    cfg.FEATURE_DATA_DIR
    / "step06_stock_features"
    / "06_stock_features_47.parquet"
)

STEP06_COMPLETE_PANEL = (
    cfg.FEATURE_DATA_DIR
    / "step06_stock_features"
    / "06_stock_features_47_complete.parquet"
)

STEP06_DICTIONARY = (
    cfg.AUDIT_DIR
    / "06_stock_feature_dictionary.csv"
)

STEP06_SUMMARY_JSON = (
    cfg.AUDIT_DIR
    / "06_stock_feature_summary.json"
)

STEP07_PANEL = (
    cfg.FEATURE_DATA_DIR
    / "step07_nifty_context"
    / "07_nifty_context_16_aligned.parquet"
)

STEP07_DICTIONARY = (
    cfg.AUDIT_DIR
    / "07_nifty_feature_dictionary.csv"
)

STEP07_SUMMARY_JSON = (
    cfg.AUDIT_DIR
    / "07_nifty_feature_summary.json"
)

STEP07_BRIDGE_AUDIT = (
    cfg.AUDIT_DIR
    / "07_nifty_bridge_audit.csv"
)


# =============================================================================
# 3. STEP-08 OUTPUT PATHS
# =============================================================================

STEP08_OUTPUT_DIR = (
    cfg.FEATURE_DATA_DIR
    / "step08_merged_63_features"
)

FULL_PANEL_PARQUET = (
    STEP08_OUTPUT_DIR
    / "08_merged_63_feature_panel.parquet"
)

FULL_PANEL_CSV = (
    STEP08_OUTPUT_DIR
    / "08_merged_63_feature_panel.csv"
)

COMPLETE_PANEL_PARQUET = (
    STEP08_OUTPUT_DIR
    / "08_merged_63_feature_panel_complete.parquet"
)

COMPLETE_PANEL_CSV = (
    STEP08_OUTPUT_DIR
    / "08_merged_63_feature_panel_complete.csv"
)

COMBINED_DICTIONARY_CSV = (
    cfg.AUDIT_DIR
    / "08_combined_63_feature_dictionary.csv"
)

NULL_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "08_63_feature_null_audit.csv"
)

MERGE_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "08_63_feature_merge_audit.csv"
)

SOURCE_RECONCILIATION_CSV = (
    cfg.AUDIT_DIR
    / "08_source_feature_reconciliation.csv"
)

COMPLETE_KEY_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "08_complete_key_reconciliation.csv"
)

STOCK_SUMMARY_CSV = (
    cfg.AUDIT_DIR
    / "08_63_feature_stock_summary.csv"
)

SEGMENT_SUMMARY_CSV = (
    cfg.AUDIT_DIR
    / "08_63_feature_segment_summary.csv"
)

BRIDGE_AUDIT_CSV = (
    cfg.AUDIT_DIR
    / "08_nifty_bridge_merge_audit.csv"
)

HASH_REGISTER_CSV = (
    cfg.AUDIT_DIR
    / "08_63_feature_hashes.csv"
)

SOURCE_HASH_CSV = (
    cfg.AUDIT_DIR
    / "08_source_file_hashes.csv"
)

STEP08_SUMMARY_JSON = (
    cfg.AUDIT_DIR
    / "08_63_feature_summary.json"
)

MANIFEST_JSON = (
    cfg.MANIFEST_DIR
    / "08_63_feature_panel_manifest.json"
)

MANIFEST_SHA = (
    cfg.MANIFEST_DIR
    / "08_63_feature_panel_manifest.sha256"
)

FROZEN_STEP08_SCRIPT = (
    cfg.MANIFEST_DIR
    / "step08_merge_63_feature_panel_frozen.py"
)

SUMMARY_TXT = (
    cfg.REPORT_DIR
    / "08_63_feature_panel_summary.txt"
)

FAILED_ATTEMPT_DIR = (
    cfg.AUDIT_DIR
    / "step08_failed_attempts"
)


# =============================================================================
# 4. GENERIC HELPERS
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def canonical_feature_interface_sha256(
    features: list[str],
) -> str:
    payload = json.dumps(
        features,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_text(payload)


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
# 5. STEP-00 / FINAL-TEST LOCK VERIFICATION
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
# 6. UPSTREAM MANIFEST VERIFICATION
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

    return {
        "manifest_sha256": manifest_sha,
        "verified_raw_file_count": int(len(raw_records)),
    }


def verify_simple_chained_step(
    *,
    manifest_path: Path,
    sha_path: Path,
    label: str,
    expected_configuration_fingerprint: str,
    expected_upstream_fields: Dict[str, str],
    artifact_key: str = "artifacts",
) -> Dict[str, Any]:
    manifest_sha = verify_sha_sidecar(
        manifest_path,
        sha_path,
        label,
    )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("step", {}).get("status") != "PASS":
        raise RuntimeError(
            f"Stored {label} status is not PASS."
        )

    if (
        manifest.get("configuration_fingerprint_sha256")
        != expected_configuration_fingerprint
    ):
        raise RuntimeError(
            f"{label} / Step-00 fingerprint mismatch."
        )

    for field, expected in expected_upstream_fields.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"{label} chain mismatch: {field}"
            )

    verify_manifest_artifacts(
        manifest,
        artifact_key=artifact_key,
    )

    return {
        "manifest_sha256": manifest_sha,
        "manifest": manifest,
    }


def verify_step03(
    expected_configuration_fingerprint: str,
    expected_step01_sha: str,
    expected_step02_sha: str,
) -> Dict[str, Any]:
    return verify_simple_chained_step(
        manifest_path=STEP03_MANIFEST,
        sha_path=STEP03_MANIFEST_SHA,
        label="Step 03",
        expected_configuration_fingerprint=(
            expected_configuration_fingerprint
        ),
        expected_upstream_fields={
            "step01_manifest_sha256": expected_step01_sha,
            "step02_manifest_sha256": expected_step02_sha,
        },
    )


def verify_step04(
    expected_configuration_fingerprint: str,
    expected_step01_sha: str,
    expected_step02_sha: str,
    expected_step03_sha: str,
) -> Dict[str, Any]:
    return verify_simple_chained_step(
        manifest_path=STEP04_MANIFEST,
        sha_path=STEP04_MANIFEST_SHA,
        label="Step 04",
        expected_configuration_fingerprint=(
            expected_configuration_fingerprint
        ),
        expected_upstream_fields={
            "step01_manifest_sha256": expected_step01_sha,
            "step02_manifest_sha256": expected_step02_sha,
            "step03_manifest_sha256": expected_step03_sha,
        },
    )


def verify_step05(
    expected_configuration_fingerprint: str,
    expected_step01_sha: str,
    expected_step02_sha: str,
    expected_step03_sha: str,
    expected_step04_sha: str,
) -> Dict[str, Any]:
    result = verify_simple_chained_step(
        manifest_path=STEP05_MANIFEST,
        sha_path=STEP05_MANIFEST_SHA,
        label="Step 05",
        expected_configuration_fingerprint=(
            expected_configuration_fingerprint
        ),
        expected_upstream_fields={
            "step01_manifest_sha256": expected_step01_sha,
            "step02_manifest_sha256": expected_step02_sha,
            "step03_manifest_sha256": expected_step03_sha,
            "step04_manifest_sha256": expected_step04_sha,
        },
    )

    failures = []

    for record in result["manifest"].get(
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

    return result


def verify_step06(
    expected_configuration_fingerprint: str,
    expected_step01_sha: str,
    expected_step02_sha: str,
    expected_step03_sha: str,
    expected_step04_sha: str,
    expected_step05_sha: str,
) -> Dict[str, Any]:
    result = verify_simple_chained_step(
        manifest_path=STEP06_MANIFEST,
        sha_path=STEP06_MANIFEST_SHA,
        label="Step 06",
        expected_configuration_fingerprint=(
            expected_configuration_fingerprint
        ),
        expected_upstream_fields={
            "step01_manifest_sha256": expected_step01_sha,
            "step02_manifest_sha256": expected_step02_sha,
            "step03_manifest_sha256": expected_step03_sha,
            "step04_manifest_sha256": expected_step04_sha,
            "step05_manifest_sha256": expected_step05_sha,
        },
    )

    for path in (
        STEP06_PANEL,
        STEP06_COMPLETE_PANEL,
        STEP06_DICTIONARY,
        STEP06_SUMMARY_JSON,
    ):
        if not path.exists():
            raise RuntimeError(
                f"Required Step-06 artifact missing: {path}"
            )

    return result


def verify_step07(
    expected_configuration_fingerprint: str,
    expected_step01_sha: str,
    expected_step02_sha: str,
    expected_step03_sha: str,
    expected_step04_sha: str,
    expected_step05_sha: str,
    expected_step06_sha: str,
) -> Dict[str, Any]:
    result = verify_simple_chained_step(
        manifest_path=STEP07_MANIFEST,
        sha_path=STEP07_MANIFEST_SHA,
        label="Step 07",
        expected_configuration_fingerprint=(
            expected_configuration_fingerprint
        ),
        expected_upstream_fields={
            "step01_manifest_sha256": expected_step01_sha,
            "step02_manifest_sha256": expected_step02_sha,
            "step03_manifest_sha256": expected_step03_sha,
            "step04_manifest_sha256": expected_step04_sha,
            "step05_manifest_sha256": expected_step05_sha,
            "step06_manifest_sha256": expected_step06_sha,
        },
    )

    for path in (
        STEP07_PANEL,
        STEP07_DICTIONARY,
        STEP07_SUMMARY_JSON,
        STEP07_BRIDGE_AUDIT,
    ):
        if not path.exists():
            raise RuntimeError(
                f"Required Step-07 artifact missing: {path}"
            )

    return result


# =============================================================================
# 7. IMMUTABILITY / STAGING
# =============================================================================

def canonical_step08_outputs() -> tuple[Path, ...]:
    return (
        STEP08_OUTPUT_DIR,
        COMBINED_DICTIONARY_CSV,
        NULL_AUDIT_CSV,
        MERGE_AUDIT_CSV,
        SOURCE_RECONCILIATION_CSV,
        COMPLETE_KEY_AUDIT_CSV,
        STOCK_SUMMARY_CSV,
        SEGMENT_SUMMARY_CSV,
        BRIDGE_AUDIT_CSV,
        HASH_REGISTER_CSV,
        SOURCE_HASH_CSV,
        STEP08_SUMMARY_JSON,
        MANIFEST_JSON,
        MANIFEST_SHA,
        FROZEN_STEP08_SCRIPT,
        SUMMARY_TXT,
    )


def assert_new_step08_target() -> None:
    collisions = [
        path
        for path in canonical_step08_outputs()
        if path.exists()
    ]

    if collisions:
        detail = "\n".join(
            str(path)
            for path in collisions
        )

        raise RuntimeError(
            "Step-08 canonical output(s) already exist. "
            "Refusing to overwrite.\n"
            f"{detail}\n\n"
            "If Step 08 already passed, run --verify-only."
        )


def create_staging_root() -> Path:
    timestamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )

    root = (
        cfg.FEATURE_DATA_DIR
        / f"_step08_staging_{timestamp}_{os.getpid()}"
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
        for path in canonical_step08_outputs()
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
# 8. LOCK THE 47 + 16 FEATURE INTERFACE
# =============================================================================

def load_feature_interfaces() -> Tuple[
    list[str],
    list[str],
    list[str],
    str,
    pd.DataFrame,
    pd.DataFrame,
]:
    with STEP06_SUMMARY_JSON.open("r", encoding="utf-8") as handle:
        step06_summary = json.load(handle)

    with STEP07_SUMMARY_JSON.open("r", encoding="utf-8") as handle:
        step07_summary = json.load(handle)

    stock_features = list(
        step06_summary[
            "feature_design"
        ][
            "feature_order"
        ]
    )

    nifty_features = list(
        step07_summary[
            "feature_design"
        ][
            "feature_order"
        ]
    )

    if len(stock_features) != EXPECTED_STOCK_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_STOCK_FEATURES} frozen stock features; "
            f"found {len(stock_features)}."
        )

    if len(nifty_features) != EXPECTED_NIFTY_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_NIFTY_FEATURES} frozen NIFTY features; "
            f"found {len(nifty_features)}."
        )

    if len(set(stock_features)) != len(stock_features):
        raise RuntimeError(
            "Duplicate stock feature names."
        )

    if len(set(nifty_features)) != len(nifty_features):
        raise RuntimeError(
            "Duplicate NIFTY feature names."
        )

    overlap = sorted(
        set(stock_features)
        &
        set(nifty_features)
    )

    if overlap:
        raise RuntimeError(
            "Stock and NIFTY feature names overlap:\n"
            f"{overlap}"
        )

    all_features = (
        stock_features
        +
        nifty_features
    )

    if len(all_features) != EXPECTED_TOTAL_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_FEATURES} total features; "
            f"found {len(all_features)}."
        )

    stock_dictionary = pd.read_csv(
        STEP06_DICTIONARY
    )

    nifty_dictionary = pd.read_csv(
        STEP07_DICTIONARY
    )

    if "feature" not in stock_dictionary.columns:
        raise RuntimeError(
            "Step-06 feature dictionary lacks feature column."
        )

    if "feature" not in nifty_dictionary.columns:
        raise RuntimeError(
            "Step-07 feature dictionary lacks feature column."
        )

    if (
        stock_dictionary["feature"].tolist()
        != stock_features
    ):
        raise RuntimeError(
            "Step-06 summary feature order differs from "
            "the frozen Step-06 dictionary."
        )

    if (
        nifty_dictionary["feature"].tolist()
        != nifty_features
    ):
        raise RuntimeError(
            "Step-07 summary feature order differs from "
            "the frozen Step-07 dictionary."
        )

    interface_hash = (
        canonical_feature_interface_sha256(
            all_features
        )
    )

    return (
        stock_features,
        nifty_features,
        all_features,
        interface_hash,
        stock_dictionary,
        nifty_dictionary,
    )


# =============================================================================
# 9. LOAD AND VALIDATE SOURCE PANELS
# =============================================================================

def load_source_panels(
    stock_features: list[str],
    nifty_features: list[str],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    stock = pd.read_parquet(
        STEP06_PANEL,
        engine="pyarrow",
    )

    stock_complete = pd.read_parquet(
        STEP06_COMPLETE_PANEL,
        engine="pyarrow",
    )

    nifty = pd.read_parquet(
        STEP07_PANEL,
        engine="pyarrow",
    )

    for table in (
        stock,
        stock_complete,
        nifty,
    ):
        table["Date"] = (
            pd.to_datetime(
                table["Date"],
                errors="raise",
            )
            .dt.normalize()
        )

    required_stock_columns = {
        "session_number",
        "Date",
        "Ticker",
        "continuity_segment_id",
        *stock_features,
    }

    missing_stock = (
        required_stock_columns
        -
        set(stock.columns)
    )

    if missing_stock:
        raise RuntimeError(
            "Step-06 full panel missing columns:\n"
            f"{sorted(missing_stock)}"
        )

    missing_complete_stock = (
        required_stock_columns
        -
        set(stock_complete.columns)
    )

    if missing_complete_stock:
        raise RuntimeError(
            "Step-06 complete panel missing columns:\n"
            f"{sorted(missing_complete_stock)}"
        )

    required_nifty_columns = {
        "Date",
        "index_ticker",
        "index_name",
        *nifty_features,
    }

    missing_nifty = (
        required_nifty_columns
        -
        set(nifty.columns)
    )

    if missing_nifty:
        raise RuntimeError(
            "Step-07 aligned panel missing columns:\n"
            f"{sorted(missing_nifty)}"
        )

    if stock[["Ticker", "Date"]].duplicated().any():
        raise RuntimeError(
            "Duplicate Step-06 full-panel ticker-date keys."
        )

    if stock_complete[["Ticker", "Date"]].duplicated().any():
        raise RuntimeError(
            "Duplicate Step-06 complete-panel ticker-date keys."
        )

    if nifty["Date"].duplicated().any():
        raise RuntimeError(
            "Duplicate Step-07 aligned NIFTY dates."
        )

    if stock["Ticker"].nunique() != EXPECTED_STOCK_COUNT:
        raise RuntimeError(
            f"Step-06 panel contains {stock['Ticker'].nunique()} stocks; "
            f"expected {EXPECTED_STOCK_COUNT}."
        )

    expected_tickers = set(
        cfg.STOCK_TICKERS
    )

    observed_tickers = set(
        stock["Ticker"].unique()
    )

    if observed_tickers != expected_tickers:
        raise RuntimeError(
            "Step-06 stock universe differs from Step-00."
        )

    stock_dates = pd.DatetimeIndex(
        stock["Date"]
        .drop_duplicates()
        .sort_values()
    )

    nifty_dates = pd.DatetimeIndex(
        nifty["Date"]
        .drop_duplicates()
        .sort_values()
    )

    if not stock_dates.equals(
        nifty_dates
    ):
        raise RuntimeError(
            "Step-06 full-panel dates and Step-07 aligned NIFTY dates differ."
        )

    expected_full_rows = (
        len(nifty_dates)
        * EXPECTED_STOCK_COUNT
    )

    if len(stock) != expected_full_rows:
        raise RuntimeError(
            "Unexpected Step-06 full-panel row count.\n"
            f"Expected from calendar × stocks: {expected_full_rows:,}\n"
            f"Observed: {len(stock):,}"
        )

    daily_counts = (
        stock
        .groupby("Date")["Ticker"]
        .nunique()
    )

    if not (
        daily_counts
        == EXPECTED_STOCK_COUNT
    ).all():
        raise RuntimeError(
            "Step-06 full panel lacks a complete stock cross-section."
        )

    complete_counts_by_date = (
        stock_complete
        .groupby("Date")["Ticker"]
        .nunique()
    )

    if not (
        complete_counts_by_date
        == EXPECTED_STOCK_COUNT
    ).all():
        raise RuntimeError(
            "Step-06 complete panel is not balanced by date."
        )

    complete_counts_by_ticker = (
        stock_complete
        .groupby("Ticker")
        .size()
    )

    if complete_counts_by_ticker.nunique() != 1:
        raise RuntimeError(
            "Step-06 complete panel is not balanced by ticker."
        )

    return (
        stock,
        stock_complete,
        nifty,
    )


# =============================================================================
# 10. EXACT-DATE MANY-TO-ONE MERGE
# =============================================================================

def merge_panels(
    stock: pd.DataFrame,
    nifty: pd.DataFrame,
    stock_features: list[str],
    nifty_features: list[str],
) -> pd.DataFrame:
    stock_columns = [
        "session_number",
        "Date",
        "Ticker",
        "continuity_segment_id",
    ] + stock_features

    nifty_columns = [
        "Date",
        "index_ticker",
        "index_name",
    ] + nifty_features

    merged = (
        stock[
            stock_columns
        ]
        .merge(
            nifty[
                nifty_columns
            ],
            on="Date",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )

    merged = (
        merged
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(drop=True)
    )

    if len(merged) != len(stock):
        raise RuntimeError(
            "Merge changed the stock-panel row count."
        )

    if merged["index_ticker"].isna().any():
        raise RuntimeError(
            "One or more stock rows lack aligned NIFTY context."
        )

    if merged[["Ticker", "Date"]].duplicated().any():
        raise RuntimeError(
            "Duplicate ticker-date keys after 63-feature merge."
        )

    return merged


# =============================================================================
# 11. SOURCE FEATURE RECONCILIATION
# =============================================================================

def reconcile_merged_sources(
    merged: pd.DataFrame,
    stock: pd.DataFrame,
    nifty: pd.DataFrame,
    stock_features: list[str],
    nifty_features: list[str],
) -> pd.DataFrame:
    rows = []

    # Stock features must be bit-for-bit equal after the merge.
    merged_stock = (
        merged[
            [
                "Ticker",
                "Date",
            ]
            + stock_features
        ]
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(drop=True)
    )

    original_stock = (
        stock[
            [
                "Ticker",
                "Date",
            ]
            + stock_features
        ]
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(drop=True)
    )

    if not merged_stock[["Ticker", "Date"]].equals(
        original_stock[["Ticker", "Date"]]
    ):
        raise RuntimeError(
            "Merged stock keys differ from Step-06 stock keys."
        )

    stock_left = (
        merged_stock[
            stock_features
        ]
        .to_numpy(dtype=float)
    )

    stock_right = (
        original_stock[
            stock_features
        ]
        .to_numpy(dtype=float)
    )

    stock_exact = bool(
        np.array_equal(
            stock_left,
            stock_right,
            equal_nan=True,
        )
    )

    if not stock_exact:
        raise RuntimeError(
            "One or more Step-06 stock feature values changed during merge."
        )

    finite_stock = (
        np.isfinite(stock_left)
        &
        np.isfinite(stock_right)
    )

    stock_max_diff = float(
        np.max(
            np.abs(
                stock_left[finite_stock]
                -
                stock_right[finite_stock]
            )
        )
        if finite_stock.any()
        else 0.0
    )

    rows.append(
        {
            "source": "step06_stock_features",
            "rows_or_dates_checked": int(
                len(original_stock)
            ),
            "feature_count": int(
                len(stock_features)
            ),
            "exact_match": stock_exact,
            "maximum_absolute_difference": (
                stock_max_diff
            ),
        }
    )

    # NIFTY values are copied to every stock row on a date. First verify
    # within-date equality, then compare one representative merged row/date
    # with the Step-07 aligned source.
    for feature in nifty_features:
        unique_counts = (
            merged
            .groupby("Date")[feature]
            .nunique(dropna=False)
        )

        if (
            unique_counts
            > 1
        ).any():
            raise RuntimeError(
                f"NIFTY feature {feature} differs across stocks "
                "on the same merged date."
            )

    merged_nifty = (
        merged[
            [
                "Date",
            ]
            + nifty_features
        ]
        .drop_duplicates(
            subset=["Date"]
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    original_nifty = (
        nifty[
            [
                "Date",
            ]
            + nifty_features
        ]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if not merged_nifty["Date"].equals(
        original_nifty["Date"]
    ):
        raise RuntimeError(
            "Merged NIFTY date sequence differs from Step 07."
        )

    nifty_left = (
        merged_nifty[
            nifty_features
        ]
        .to_numpy(dtype=float)
    )

    nifty_right = (
        original_nifty[
            nifty_features
        ]
        .to_numpy(dtype=float)
    )

    nifty_exact = bool(
        np.array_equal(
            nifty_left,
            nifty_right,
            equal_nan=True,
        )
    )

    if not nifty_exact:
        raise RuntimeError(
            "One or more Step-07 NIFTY feature values changed during merge."
        )

    finite_nifty = (
        np.isfinite(nifty_left)
        &
        np.isfinite(nifty_right)
    )

    nifty_max_diff = float(
        np.max(
            np.abs(
                nifty_left[finite_nifty]
                -
                nifty_right[finite_nifty]
            )
        )
        if finite_nifty.any()
        else 0.0
    )

    rows.append(
        {
            "source": "step07_nifty_features",
            "rows_or_dates_checked": int(
                len(original_nifty)
            ),
            "feature_count": int(
                len(nifty_features)
            ),
            "exact_match": nifty_exact,
            "maximum_absolute_difference": (
                nifty_max_diff
            ),
        }
    )

    return pd.DataFrame(rows)


# =============================================================================
# 12. COMPLETE-CASE INTERSECTION AND KEY RECONCILIATION
# =============================================================================

def build_complete_panel_and_key_audit(
    merged: pd.DataFrame,
    step06_complete: pd.DataFrame,
    all_features: list[str],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    complete_mask = (
        merged[
            all_features
        ]
        .notna()
        .all(axis=1)
    )

    complete = (
        merged.loc[
            complete_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    if complete.empty:
        raise RuntimeError(
            "Merged 63-feature panel has no complete rows."
        )

    step08_keys = (
        complete[
            [
                "Ticker",
                "Date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(drop=True)
    )

    step06_keys = (
        step06_complete[
            [
                "Ticker",
                "Date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(drop=True)
    )

    key_merge = (
        step06_keys
        .merge(
            step08_keys,
            on=[
                "Ticker",
                "Date",
            ],
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
    )

    step06_only = int(
        (
            key_merge["_merge"]
            == "left_only"
        ).sum()
    )

    step08_only = int(
        (
            key_merge["_merge"]
            == "right_only"
        ).sum()
    )

    exact_key_match = bool(
        step06_only == 0
        and
        step08_only == 0
        and
        len(step06_keys) == len(step08_keys)
    )

    if not exact_key_match:
        raise RuntimeError(
            "The common 63-feature complete-case keys differ from the "
            "frozen Step-06 complete-stock-feature keys.\n"
            f"Step-06-only keys: {step06_only}\n"
            f"Step-08-only keys: {step08_only}"
        )

    daily_counts = (
        complete
        .groupby("Date")["Ticker"]
        .nunique()
    )

    if not (
        daily_counts
        == EXPECTED_STOCK_COUNT
    ).all():
        raise RuntimeError(
            "Complete 63-feature panel is not balanced by date."
        )

    ticker_counts = (
        complete
        .groupby("Ticker")
        .size()
    )

    if ticker_counts.nunique() != 1:
        raise RuntimeError(
            "Complete 63-feature panel is not balanced by ticker."
        )

    audit = pd.DataFrame(
        [
            {
                "step06_complete_rows": int(
                    len(step06_keys)
                ),
                "step08_complete_rows": int(
                    len(step08_keys)
                ),
                "step06_complete_dates": int(
                    step06_keys[
                        "Date"
                    ].nunique()
                ),
                "step08_complete_dates": int(
                    step08_keys[
                        "Date"
                    ].nunique()
                ),
                "step06_only_keys": step06_only,
                "step08_only_keys": step08_only,
                "exact_ticker_date_key_match": (
                    exact_key_match
                ),
                "nifty_context_removed_additional_rows": (
                    step08_only > 0
                    or
                    step06_only > 0
                ),
                "complete_rows_per_stock": int(
                    ticker_counts.iloc[0]
                ),
                "minimum_stocks_per_complete_date": int(
                    daily_counts.min()
                ),
                "maximum_stocks_per_complete_date": int(
                    daily_counts.max()
                ),
            }
        ]
    )

    return (
        complete,
        audit,
    )


# =============================================================================
# 13. MERGE / NULL / BRIDGE AUDITS
# =============================================================================

def build_merge_audit(
    merged: pd.DataFrame,
    complete: pd.DataFrame,
    all_features: list[str],
) -> pd.DataFrame:
    numeric = (
        merged[
            all_features
        ]
        .to_numpy(dtype=float)
    )

    infinite_count = int(
        np.isinf(
            numeric
        ).sum()
    )

    if infinite_count != 0:
        raise RuntimeError(
            f"Infinite feature values detected: {infinite_count}"
        )

    daily_full_counts = (
        merged
        .groupby("Date")["Ticker"]
        .nunique()
    )

    daily_complete_counts = (
        complete
        .groupby("Date")["Ticker"]
        .nunique()
    )

    return pd.DataFrame(
        [
            {
                "full_rows": int(
                    len(merged)
                ),
                "full_dates": int(
                    merged[
                        "Date"
                    ].nunique()
                ),
                "stock_count": int(
                    merged[
                        "Ticker"
                    ].nunique()
                ),
                "feature_count": int(
                    len(all_features)
                ),
                "complete_rows": int(
                    len(complete)
                ),
                "complete_dates": int(
                    complete[
                        "Date"
                    ].nunique()
                ),
                "complete_rows_per_stock": int(
                    complete
                    .groupby("Ticker")
                    .size()
                    .iloc[0]
                ),
                "minimum_stocks_per_full_date": int(
                    daily_full_counts.min()
                ),
                "maximum_stocks_per_full_date": int(
                    daily_full_counts.max()
                ),
                "minimum_stocks_per_complete_date": int(
                    daily_complete_counts.min()
                ),
                "maximum_stocks_per_complete_date": int(
                    daily_complete_counts.max()
                ),
                "infinite_feature_values": (
                    infinite_count
                ),
                "nifty_many_to_one_merge": True,
                "forward_fill_used": False,
                "backward_fill_used": False,
                "interpolation_used": False,
                "imputation_used": False,
                "scaling_applied": False,
                "targets_constructed": False,
                "regimes_constructed": False,
                "feature_selection_applied": False,
            }
        ]
    )


def build_null_audit(
    merged: pd.DataFrame,
    stock_features: list[str],
    nifty_features: list[str],
) -> pd.DataFrame:
    rows = []

    for modality, features in (
        (
            "stock",
            stock_features,
        ),
        (
            "nifty",
            nifty_features,
        ),
    ):
        for feature in features:
            series = merged[
                feature
            ]

            null_mask = (
                series.isna()
            )

            valid_mask = (
                ~null_mask
            )

            first_valid_date = (
                merged.loc[
                    valid_mask,
                    "Date",
                ].min()
                if valid_mask.any()
                else pd.NaT
            )

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            ).to_numpy(dtype=float)

            rows.append(
                {
                    "modality": modality,
                    "feature": feature,
                    "null_count": int(
                        null_mask.sum()
                    ),
                    "null_percent": float(
                        100.0
                        * null_mask.sum()
                        / len(merged)
                    ),
                    "first_valid_date": (
                        first_valid_date
                    ),
                    "infinite_count": int(
                        np.isinf(
                            numeric
                        ).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_bridge_merge_audit(
    merged: pd.DataFrame,
    nifty: pd.DataFrame,
    nifty_features: list[str],
) -> pd.DataFrame:
    bridge_source = pd.read_csv(
        STEP07_BRIDGE_AUDIT,
        parse_dates=[
            "first_bridge_date",
            "last_bridge_date",
            "pre_bridge_date",
            "post_bridge_date",
        ],
    )

    if bridge_source.empty:
        return pd.DataFrame(
            columns=[
                "bridge_block_number",
                "post_bridge_date",
                "merged_stock_rows",
                "expected_stock_rows",
                "all_nifty_features_exact_match",
                "maximum_absolute_difference",
                "nifty_return_1d",
            ]
        )

    rows = []

    for bridge in bridge_source.itertuples(
        index=False
    ):
        post_date = pd.Timestamp(
            bridge.post_bridge_date
        ).normalize()

        original = nifty.loc[
            nifty["Date"]
            == post_date
        ]

        merged_rows = merged.loc[
            merged["Date"]
            == post_date
        ]

        if len(original) != 1:
            raise RuntimeError(
                "Expected one Step-07 NIFTY row on dynamic "
                f"post-bridge date {post_date.date()}."
            )

        if len(merged_rows) != EXPECTED_STOCK_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_STOCK_COUNT} merged stock rows on "
                f"{post_date.date()}; found {len(merged_rows)}."
            )

        original_values = (
            original[
                nifty_features
            ]
            .iloc[0]
            .to_numpy(dtype=float)
        )

        all_match = True
        maximum_difference = 0.0

        for _, merged_row in merged_rows.iterrows():
            merged_values = (
                merged_row[
                    nifty_features
                ]
                .to_numpy(dtype=float)
            )

            match = bool(
                np.allclose(
                    original_values,
                    merged_values,
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            )

            all_match = (
                all_match
                and
                match
            )

            finite = (
                np.isfinite(
                    original_values
                )
                &
                np.isfinite(
                    merged_values
                )
            )

            if finite.any():
                maximum_difference = max(
                    maximum_difference,
                    float(
                        np.max(
                            np.abs(
                                original_values[
                                    finite
                                ]
                                -
                                merged_values[
                                    finite
                                ]
                            )
                        )
                    ),
                )

        if not all_match:
            raise RuntimeError(
                "Step-07 NIFTY context changed during Step-08 merge "
                f"on post-bridge date {post_date.date()}."
            )

        rows.append(
            {
                "bridge_block_number": int(
                    bridge.bridge_block_number
                ),
                "post_bridge_date": (
                    post_date
                ),
                "merged_stock_rows": int(
                    len(merged_rows)
                ),
                "expected_stock_rows": int(
                    EXPECTED_STOCK_COUNT
                ),
                "all_nifty_features_exact_match": (
                    all_match
                ),
                "maximum_absolute_difference": (
                    maximum_difference
                ),
                "nifty_return_1d": float(
                    original[
                        "nifty_return_1d"
                    ].iloc[0]
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 14. COMBINED FEATURE DICTIONARY
# =============================================================================

def build_combined_dictionary(
    stock_dictionary: pd.DataFrame,
    nifty_dictionary: pd.DataFrame,
    stock_features: list[str],
    nifty_features: list[str],
) -> pd.DataFrame:
    if (
        stock_dictionary[
            "feature"
        ].tolist()
        != stock_features
    ):
        raise RuntimeError(
            "Step-06 dictionary order differs from locked stock interface."
        )

    if (
        nifty_dictionary[
            "feature"
        ].tolist()
        != nifty_features
    ):
        raise RuntimeError(
            "Step-07 dictionary order differs from locked NIFTY interface."
        )

    stock_description_column = (
        "description"
        if "description" in stock_dictionary.columns
        else "definition"
        if "definition" in stock_dictionary.columns
        else None
    )

    nifty_description_column = (
        "definition"
        if "definition" in nifty_dictionary.columns
        else "description"
        if "description" in nifty_dictionary.columns
        else None
    )

    if stock_description_column is None:
        raise RuntimeError(
            "Step-06 dictionary has no description/definition column."
        )

    if nifty_description_column is None:
        raise RuntimeError(
            "Step-07 dictionary has no description/definition column."
        )

    rows = []
    global_number = 0

    for row in stock_dictionary.itertuples(
        index=False
    ):
        global_number += 1

        rows.append(
            {
                "global_feature_number": (
                    global_number
                ),
                "modality": "stock",
                "feature": (
                    row.feature
                ),
                "feature_group": (
                    row.feature_group
                ),
                "definition": (
                    getattr(
                        row,
                        stock_description_column,
                    )
                ),
                "causal": True,
                "target_derived": False,
                "scaling_applied_here": False,
            }
        )

    for row in nifty_dictionary.itertuples(
        index=False
    ):
        global_number += 1

        rows.append(
            {
                "global_feature_number": (
                    global_number
                ),
                "modality": "nifty",
                "feature": (
                    row.feature
                ),
                "feature_group": (
                    row.feature_group
                ),
                "definition": (
                    getattr(
                        row,
                        nifty_description_column,
                    )
                ),
                "causal": True,
                "target_derived": False,
                "scaling_applied_here": False,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if len(result) != EXPECTED_TOTAL_FEATURES:
        raise RuntimeError(
            "Combined feature dictionary is not exactly 63 rows."
        )

    if result["feature"].duplicated().any():
        raise RuntimeError(
            "Duplicate feature names in combined dictionary."
        )

    return result


# =============================================================================
# 15. STOCK / SEGMENT SUMMARIES
# =============================================================================

def build_stock_summary(
    merged: pd.DataFrame,
    complete: pd.DataFrame,
) -> pd.DataFrame:
    full = (
        merged
        .groupby("Ticker")
        .agg(
            full_rows=(
                "Date",
                "size",
            ),
            first_date=(
                "Date",
                "min",
            ),
            last_date=(
                "Date",
                "max",
            ),
            continuity_segments=(
                "continuity_segment_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    complete_counts = (
        complete
        .groupby("Ticker")
        .size()
        .rename(
            "complete_63_feature_rows"
        )
        .reset_index()
    )

    return (
        full
        .merge(
            complete_counts,
            on="Ticker",
            how="left",
            validate="one_to_one",
        )
    )


def build_segment_summary(
    merged: pd.DataFrame,
    complete: pd.DataFrame,
) -> pd.DataFrame:
    full = (
        merged
        .groupby(
            "continuity_segment_id"
        )
        .agg(
            full_rows=(
                "Date",
                "size",
            ),
            full_dates=(
                "Date",
                "nunique",
            ),
            stocks=(
                "Ticker",
                "nunique",
            ),
            first_date=(
                "Date",
                "min",
            ),
            last_date=(
                "Date",
                "max",
            ),
        )
        .reset_index()
    )

    complete_summary = (
        complete
        .groupby(
            "continuity_segment_id"
        )
        .agg(
            complete_rows=(
                "Date",
                "size",
            ),
            complete_dates=(
                "Date",
                "nunique",
            ),
        )
        .reset_index()
    )

    return (
        full
        .merge(
            complete_summary,
            on="continuity_segment_id",
            how="left",
            validate="one_to_one",
        )
    )


# =============================================================================
# 16. STAGING / HASH HELPERS
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
# 17. SOURCE-CODE HASH AUDIT
# =============================================================================

def find_frozen_script(
    manifest_path: Path,
    step_prefix: str,
) -> Path:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    candidates = []

    for artifact_key in (
        "artifacts",
        "canonical_artifacts",
    ):
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

            if (
                path.suffix == ".py"
                and
                "frozen" in path.name.lower()
                and
                path.name.startswith(step_prefix)
            ):
                candidates.append(path)

    unique = []

    for path in candidates:
        if path not in unique:
            unique.append(path)

    if len(unique) != 1:
        raise RuntimeError(
            f"Could not identify exactly one frozen script for "
            f"{step_prefix}; found {len(unique)}."
        )

    return unique[0]


def collect_source_hashes() -> pd.DataFrame:
    current_step00 = Path(
        cfg.__file__
    ).resolve()

    frozen_step00 = (
        cfg.get_frozen_script_path()
        .resolve()
    )

    frozen_step01 = find_frozen_script(
        STEP01_MANIFEST,
        "step01_",
    )

    frozen_step02 = find_frozen_script(
        STEP02_MANIFEST,
        "step02_",
    )

    frozen_step03 = find_frozen_script(
        STEP03_MANIFEST,
        "step03_",
    )

    frozen_step04 = find_frozen_script(
        STEP04_MANIFEST,
        "step04_",
    )

    frozen_step05 = find_frozen_script(
        STEP05_MANIFEST,
        "step05_",
    )

    frozen_step06 = find_frozen_script(
        STEP06_MANIFEST,
        "step06_",
    )

    frozen_step07 = find_frozen_script(
        STEP07_MANIFEST,
        "step07_",
    )

    sources = [
        (
            "current_step00",
            current_step00,
        ),
        (
            "frozen_step00",
            frozen_step00,
        ),
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
        (
            "frozen_step07",
            frozen_step07,
        ),
        (
            "current_step08",
            SCRIPT_FILE.resolve(),
        ),
    ]

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
# 18. MANIFEST / VERIFY
# =============================================================================

def step08_artifact_paths() -> tuple[Path, ...]:
    return (
        FULL_PANEL_PARQUET,
        FULL_PANEL_CSV,
        COMPLETE_PANEL_PARQUET,
        COMPLETE_PANEL_CSV,
        COMBINED_DICTIONARY_CSV,
        NULL_AUDIT_CSV,
        MERGE_AUDIT_CSV,
        SOURCE_RECONCILIATION_CSV,
        COMPLETE_KEY_AUDIT_CSV,
        STOCK_SUMMARY_CSV,
        SEGMENT_SUMMARY_CSV,
        BRIDGE_AUDIT_CSV,
        HASH_REGISTER_CSV,
        SOURCE_HASH_CSV,
        STEP08_SUMMARY_JSON,
        FROZEN_STEP08_SCRIPT,
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
    step07: Dict[str, Any],
    interface_hash: str,
    all_features: list[str],
) -> str:
    artifacts = {
        path.name: artifact_record(
            path
        )
        for path in step08_artifact_paths()
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
        "step07_manifest_sha256": (
            step07[
                "manifest_sha256"
            ]
        ),
        "feature_interface_sha256": (
            interface_hash
        ),
        "ordered_63_features": (
            all_features
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


def verify_existing_step08() -> int:
    print("=" * 126)
    print("STEP 08 — VERIFY EXISTING 63-FEATURE PANEL FREEZE")
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

    step07 = verify_step07(
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
        step06[
            "manifest_sha256"
        ],
    )

    if not MANIFEST_JSON.exists():
        print("[FAIL] Step-08 manifest missing.")
        return 20

    if not MANIFEST_SHA.exists():
        print("[FAIL] Step-08 manifest SHA sidecar missing.")
        return 21

    actual_manifest_sha = verify_sha_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 08",
    )

    with MANIFEST_JSON.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("step", {}).get("status") != "PASS":
        print("[FAIL] Stored Step-08 status is not PASS.")
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
        "step07_manifest_sha256": (
            step07[
                "manifest_sha256"
            ]
        ),
    }

    for field, expected in expected_chain.items():
        if manifest.get(field) != expected:
            print(
                f"[FAIL] Step-08 chain mismatch: {field}"
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
        print("STEP 08 VERIFICATION STATUS: FAIL")
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
        "[PASS] Step-07 manifest          : "
        f"{step07['manifest_sha256']}"
    )
    print(
        "[PASS] 63-feature interface      : "
        f"{manifest['feature_interface_sha256']}"
    )
    print(
        "[PASS] Step-08 manifest          : "
        f"{actual_manifest_sha}"
    )
    print(
        "[PASS] Final-test state          : "
        f"{step00['final_test_state']}"
    )
    print("=" * 126)
    print("STEP 08 VERIFICATION STATUS: PASS")
    print("=" * 126)

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
        / f"08_merge_63_features_FAILED_{timestamp}.json"
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
# 20. MAIN
# =============================================================================

def run_new_step08() -> int:
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

        print("[2/12] Verifying Step 01 through Step 07...")

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

        step07 = verify_step07(
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
            step06[
                "manifest_sha256"
            ],
        )

        print(
            "       Step-06 manifest: "
            f"{step06['manifest_sha256']}"
        )
        print(
            "       Step-07 manifest: "
            f"{step07['manifest_sha256']}"
        )

        stage = "output_collision_guard"

        print("[3/12] Checking immutable Step-08 output target...")

        assert_new_step08_target()
        cfg.create_directories()

        stage = "feature_interface"

        print("[4/12] Locking the frozen 47 + 16 feature interface...")

        (
            stock_features,
            nifty_features,
            all_features,
            interface_hash,
            stock_dictionary,
            nifty_dictionary,
        ) = load_feature_interfaces()

        print(
            "       Stock features     : "
            f"{len(stock_features)}"
        )
        print(
            "       NIFTY features     : "
            f"{len(nifty_features)}"
        )
        print(
            "       Total features     : "
            f"{len(all_features)}"
        )
        print(
            "       Interface SHA-256  : "
            f"{interface_hash}"
        )

        stage = "load_sources"

        print("[5/12] Loading and validating frozen Step-06 / Step-07 panels...")

        (
            stock,
            step06_complete,
            nifty,
        ) = load_source_panels(
            stock_features=stock_features,
            nifty_features=nifty_features,
        )

        print(
            "       Step-06 full rows      : "
            f"{len(stock):,}"
        )
        print(
            "       Step-06 complete rows  : "
            f"{len(step06_complete):,}"
        )
        print(
            "       Step-07 aligned rows   : "
            f"{len(nifty):,}"
        )
        print(
            "       Shared sessions        : "
            f"{nifty['Date'].nunique():,}"
        )
        print(
            "       Stocks/session         : "
            f"{EXPECTED_STOCK_COUNT}"
        )

        stage = "merge"

        print("[6/12] Performing exact-date many-to-one 63-feature merge...")

        merged = merge_panels(
            stock=stock,
            nifty=nifty,
            stock_features=stock_features,
            nifty_features=nifty_features,
        )

        actual_feature_columns = [
            column
            for column in merged.columns
            if column in all_features
        ]

        if actual_feature_columns != all_features:
            raise RuntimeError(
                "Merged feature ordering differs from the locked "
                "63-feature interface."
            )

        prohibited = [
            column
            for column in merged.columns
            if column.lower()
            in PROHIBITED_EXACT_COLUMNS
        ]

        if prohibited:
            raise RuntimeError(
                "Target/future-like columns unexpectedly exist:\n"
                f"{prohibited}"
            )

        print(
            "       Merged rows         : "
            f"{len(merged):,}"
        )
        print(
            "       Merged dates        : "
            f"{merged['Date'].nunique():,}"
        )

        stage = "source_reconciliation"

        print("[7/12] Reconciling merged values exactly to Step-06 and Step-07 sources...")

        source_reconciliation = (
            reconcile_merged_sources(
                merged=merged,
                stock=stock,
                nifty=nifty,
                stock_features=stock_features,
                nifty_features=nifty_features,
            )
        )

        print(
            "       Step-06 stock values exact : "
            f"{bool(source_reconciliation.iloc[0]['exact_match'])}"
        )
        print(
            "       Step-07 NIFTY values exact : "
            f"{bool(source_reconciliation.iloc[1]['exact_match'])}"
        )

        stage = "complete_case_audit"

        print("[8/12] Forming common complete-case intersection and auditing keys...")

        (
            complete,
            complete_key_audit,
        ) = build_complete_panel_and_key_audit(
            merged=merged,
            step06_complete=step06_complete,
            all_features=all_features,
        )

        merge_audit = build_merge_audit(
            merged=merged,
            complete=complete,
            all_features=all_features,
        )

        null_audit = build_null_audit(
            merged=merged,
            stock_features=stock_features,
            nifty_features=nifty_features,
        )

        bridge_audit = build_bridge_merge_audit(
            merged=merged,
            nifty=nifty,
            nifty_features=nifty_features,
        )

        combined_dictionary = (
            build_combined_dictionary(
                stock_dictionary=stock_dictionary,
                nifty_dictionary=nifty_dictionary,
                stock_features=stock_features,
                nifty_features=nifty_features,
            )
        )

        stock_summary = build_stock_summary(
            merged=merged,
            complete=complete,
        )

        segment_summary = build_segment_summary(
            merged=merged,
            complete=complete,
        )

        print(
            "       Complete rows          : "
            f"{len(complete):,}"
        )
        print(
            "       Complete dates         : "
            f"{complete['Date'].nunique():,}"
        )
        print(
            "       Complete rows/stock    : "
            f"{complete.groupby('Ticker').size().iloc[0]:,}"
        )
        print(
            "       Step-06 key match      : "
            f"{bool(complete_key_audit.iloc[0]['exact_ticker_date_key_match'])}"
        )
        print(
            "       Additional NIFTY loss  : "
            f"{bool(complete_key_audit.iloc[0]['nifty_context_removed_additional_rows'])}"
        )
        print(
            "       Bridge blocks checked  : "
            f"{len(bridge_audit)}"
        )

        stage = "staging_outputs"

        print("[9/12] Writing merged panels to staging area...")

        staging_root = create_staging_root()

        staged_full_parquet = (
            staging_root
            / FULL_PANEL_PARQUET.name
        )
        staged_full_csv = (
            staging_root
            / FULL_PANEL_CSV.name
        )
        staged_complete_parquet = (
            staging_root
            / COMPLETE_PANEL_PARQUET.name
        )
        staged_complete_csv = (
            staging_root
            / COMPLETE_PANEL_CSV.name
        )

        write_pair_staged(
            merged,
            staged_full_parquet,
            staged_full_csv,
        )

        verify_parquet_roundtrip(
            merged,
            staged_full_parquet,
        )

        write_pair_staged(
            complete,
            staged_complete_parquet,
            staged_complete_csv,
        )

        verify_parquet_roundtrip(
            complete,
            staged_complete_parquet,
        )

        staged_hashes = [
            staged_hash_record(
                scope="full_63_feature_panel",
                file_format="parquet",
                staging_path=staged_full_parquet,
                final_path=FULL_PANEL_PARQUET,
            ),
            staged_hash_record(
                scope="full_63_feature_panel",
                file_format="csv",
                staging_path=staged_full_csv,
                final_path=FULL_PANEL_CSV,
            ),
            staged_hash_record(
                scope="complete_63_feature_panel",
                file_format="parquet",
                staging_path=staged_complete_parquet,
                final_path=COMPLETE_PANEL_PARQUET,
            ),
            staged_hash_record(
                scope="complete_63_feature_panel",
                file_format="csv",
                staging_path=staged_complete_csv,
                final_path=COMPLETE_PANEL_CSV,
            ),
        ]

        stage = "promote_and_audits"

        print("[10/12] Promoting Step-08 panels and writing frozen audits...")

        if STEP08_OUTPUT_DIR.exists():
            raise RuntimeError(
                "Canonical Step-08 output directory appeared unexpectedly."
            )

        os.replace(
            staging_root,
            STEP08_OUTPUT_DIR,
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
                    f"Promoted Step-08 output missing: {path}"
                )

            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(
                    f"Step-08 output hash changed during promotion: {path}"
                )

        source_hashes = collect_source_hashes()

        write_dataframe_csv(
            COMBINED_DICTIONARY_CSV,
            combined_dictionary,
        )

        write_dataframe_csv(
            NULL_AUDIT_CSV,
            null_audit,
        )

        write_dataframe_csv(
            MERGE_AUDIT_CSV,
            merge_audit,
        )

        write_dataframe_csv(
            SOURCE_RECONCILIATION_CSV,
            source_reconciliation,
        )

        write_dataframe_csv(
            COMPLETE_KEY_AUDIT_CSV,
            complete_key_audit,
        )

        write_dataframe_csv(
            STOCK_SUMMARY_CSV,
            stock_summary,
        )

        write_dataframe_csv(
            SEGMENT_SUMMARY_CSV,
            segment_summary,
        )

        write_dataframe_csv(
            BRIDGE_AUDIT_CSV,
            bridge_audit,
        )

        supporting_paths = [
            (
                "combined_feature_dictionary",
                COMBINED_DICTIONARY_CSV,
            ),
            (
                "null_audit",
                NULL_AUDIT_CSV,
            ),
            (
                "merge_audit",
                MERGE_AUDIT_CSV,
            ),
            (
                "source_reconciliation",
                SOURCE_RECONCILIATION_CSV,
            ),
            (
                "complete_key_reconciliation",
                COMPLETE_KEY_AUDIT_CSV,
            ),
            (
                "stock_summary",
                STOCK_SUMMARY_CSV,
            ),
            (
                "segment_summary",
                SEGMENT_SUMMARY_CSV,
            ),
            (
                "bridge_audit",
                BRIDGE_AUDIT_CSV,
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

        print("[11/12] Freezing Step-08 summary, source snapshot and manifest...")

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
                "Final-test lock changed during Step 08."
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
                "step07_manifest_sha256": (
                    step07[
                        "manifest_sha256"
                    ]
                ),
            },
            "feature_interface": {
                "stock_feature_count": (
                    len(stock_features)
                ),
                "nifty_feature_count": (
                    len(nifty_features)
                ),
                "total_feature_count": (
                    len(all_features)
                ),
                "stock_features": (
                    stock_features
                ),
                "nifty_features": (
                    nifty_features
                ),
                "ordered_63_features": (
                    all_features
                ),
                "interface_sha256": (
                    interface_hash
                ),
                "signal_timing_policy": (
                    SIGNAL_TIMING_POLICY
                ),
            },
            "merge_policy": {
                "merge_key": "Date",
                "cardinality": (
                    "many_stock_rows_to_one_nifty_row"
                ),
                "forward_fill": False,
                "backward_fill": False,
                "interpolation": False,
                "imputation": False,
                "complete_case_after_47_plus_16_merge": True,
            },
            "panel": {
                "full_rows": int(
                    len(merged)
                ),
                "full_dates": int(
                    merged[
                        "Date"
                    ].nunique()
                ),
                "stocks": int(
                    merged[
                        "Ticker"
                    ].nunique()
                ),
                "complete_rows": int(
                    len(complete)
                ),
                "complete_dates": int(
                    complete[
                        "Date"
                    ].nunique()
                ),
                "complete_rows_per_stock": int(
                    complete
                    .groupby("Ticker")
                    .size()
                    .iloc[0]
                ),
                "complete_keys_equal_step06_complete_keys": bool(
                    complete_key_audit.iloc[0][
                        "exact_ticker_date_key_match"
                    ]
                ),
                "nifty_context_removed_additional_rows": bool(
                    complete_key_audit.iloc[0][
                        "nifty_context_removed_additional_rows"
                    ]
                ),
            },
            "source_reconciliation": {
                "step06_stock_features_exact": bool(
                    source_reconciliation.iloc[0][
                        "exact_match"
                    ]
                ),
                "step07_nifty_features_exact": bool(
                    source_reconciliation.iloc[1][
                        "exact_match"
                    ]
                ),
            },
            "bridge_reconciliation": {
                "blocks_checked": int(
                    len(bridge_audit)
                ),
                "all_exact": bool(
                    bridge_audit[
                        "all_nifty_features_exact_match"
                    ].all()
                    if not bridge_audit.empty
                    else True
                ),
            },
            "scientific_boundary": {
                "forecast_targets_constructed": False,
                "scalers_fit": False,
                "regimes_estimated": False,
                "feature_selection_applied": False,
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
            STEP08_SUMMARY_JSON,
            summary_payload,
        )

        shutil.copy2(
            SCRIPT_FILE,
            FROZEN_STEP08_SCRIPT,
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
                "Step-06 manifest SHA-256: "
                + step06[
                    "manifest_sha256"
                ]
            ),
            (
                "Step-07 manifest SHA-256: "
                + step07[
                    "manifest_sha256"
                ]
            ),
            (
                "63-feature interface SHA-256: "
                + interface_hash
            ),
            (
                "Stock features: "
                f"{len(stock_features)}"
            ),
            (
                "NIFTY features: "
                f"{len(nifty_features)}"
            ),
            (
                "Total features: "
                f"{len(all_features)}"
            ),
            (
                "Full rows: "
                f"{len(merged)}"
            ),
            (
                "Complete rows: "
                f"{len(complete)}"
            ),
            (
                "Complete dates: "
                f"{complete['Date'].nunique()}"
            ),
            (
                "Complete rows/stock: "
                f"{complete.groupby('Ticker').size().iloc[0]}"
            ),
            (
                "Complete keys equal Step-06 complete keys: "
                f"{bool(complete_key_audit.iloc[0]['exact_ticker_date_key_match'])}"
            ),
            (
                "NIFTY removed additional complete rows: "
                f"{bool(complete_key_audit.iloc[0]['nifty_context_removed_additional_rows'])}"
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
            step07=step07,
            interface_hash=interface_hash,
            all_features=all_features,
        )

        stage = "final_report"

        print("[12/12] Final Step-08 report...")
        print()
        print("-" * 126)
        print("LOCKED 63-FEATURE PANEL")
        print("-" * 126)
        print(
            "Stock feature count           : "
            f"{len(stock_features)}"
        )
        print(
            "NIFTY feature count           : "
            f"{len(nifty_features)}"
        )
        print(
            "Total feature count           : "
            f"{len(all_features)}"
        )
        print(
            "63-feature interface SHA-256  : "
            f"{interface_hash}"
        )

        print("-" * 126)
        print("FULL PANEL")
        print("-" * 126)
        print(
            "Rows                          : "
            f"{len(merged):,}"
        )
        print(
            "Dates                         : "
            f"{merged['Date'].nunique():,}"
        )
        print(
            "Stocks                        : "
            f"{merged['Ticker'].nunique()}"
        )

        full_counts = (
            merged
            .groupby("Date")["Ticker"]
            .nunique()
        )

        print(
            "Minimum stocks/date           : "
            f"{int(full_counts.min())}"
        )
        print(
            "Maximum stocks/date           : "
            f"{int(full_counts.max())}"
        )

        print("-" * 126)
        print("COMPLETE 63-FEATURE PANEL")
        print("-" * 126)
        print(
            "Complete rows                 : "
            f"{len(complete):,}"
        )
        print(
            "Complete dates                : "
            f"{complete['Date'].nunique():,}"
        )
        print(
            "Complete rows/stock           : "
            f"{complete.groupby('Ticker').size().iloc[0]:,}"
        )
        print(
            "Exact Step-06 complete keys   : "
            f"{bool(complete_key_audit.iloc[0]['exact_ticker_date_key_match'])}"
        )
        print(
            "Additional NIFTY row loss     : "
            f"{bool(complete_key_audit.iloc[0]['nifty_context_removed_additional_rows'])}"
        )

        complete_counts = (
            complete
            .groupby("Date")["Ticker"]
            .nunique()
        )

        print(
            "Minimum stocks/date           : "
            f"{int(complete_counts.min())}"
        )
        print(
            "Maximum stocks/date           : "
            f"{int(complete_counts.max())}"
        )

        print("-" * 126)
        print("CONTINUITY SEGMENTS")
        print("-" * 126)

        for row in segment_summary.itertuples(
            index=False
        ):
            print(
                f"Segment {row.continuity_segment_id}: "
                f"{pd.Timestamp(row.first_date).date()} -> "
                f"{pd.Timestamp(row.last_date).date()} | "
                f"full dates={row.full_dates:,} | "
                f"complete dates={row.complete_dates:,} | "
                f"complete rows={row.complete_rows:,}"
            )

        print("-" * 126)
        print("NIFTY BRIDGE MERGE AUDIT")
        print("-" * 126)

        if bridge_audit.empty:
            print(
                "No Step-07 bridge blocks to audit."
            )
        else:
            for row in bridge_audit.itertuples(
                index=False
            ):
                print(
                    f"Block {row.bridge_block_number}: "
                    f"post-date={pd.Timestamp(row.post_bridge_date).date()} | "
                    f"merged rows={row.merged_stock_rows} | "
                    f"NIFTY exact={row.all_nifty_features_exact_match} | "
                    f"max diff={row.maximum_absolute_difference:.3e}"
                )

        print("-" * 126)
        print("PIPELINE STATE")
        print("-" * 126)
        print("Scaling applied              : False")
        print("Targets constructed          : False")
        print("Regimes constructed          : False")
        print("Feature selection applied    : False")
        print("Imputation performed         : False")

        print("-" * 126)
        print("OUTPUT FILES")
        print("-" * 126)
        print(
            f"Full 63-feature panel : {FULL_PANEL_PARQUET}"
        )
        print(
            f"Complete panel        : {COMPLETE_PANEL_PARQUET}"
        )
        print(
            f"Feature dictionary    : {COMBINED_DICTIONARY_CSV}"
        )
        print(
            f"Null audit            : {NULL_AUDIT_CSV}"
        )
        print(
            f"Merge audit           : {MERGE_AUDIT_CSV}"
        )
        print(
            f"Complete-key audit    : {COMPLETE_KEY_AUDIT_CSV}"
        )
        print(
            f"Manifest              : {MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256      : {manifest_sha}"
        )
        print(
            f"Final-test state      : {step00['final_test_state']}"
        )
        print("=" * 126)
        print(
            "PASS: 47 stock + 16 NIFTY predictors merged into exactly 63 features."
        )
        print(
            "PASS: Exact-date many-to-one context alignment verified."
        )
        print(
            "PASS: Every Step-06 stock feature and Step-07 NIFTY feature "
            "is reproduced exactly after merge."
        )
        print(
            "PASS: Common complete-case ticker-date keys exactly match the "
            "frozen Step-06 complete-feature keys."
        )
        print(
            "PASS: Step-07 bridge context is preserved exactly on every "
            "audited post-bridge date."
        )
        print(
            "PASS: No scaling, targets, regimes, imputation, sequences, "
            "or feature selection were created."
        )
        print(
            "PASS: Final held-out test remains LOCKED_NOT_OPENED."
        )
        print("=" * 126)
        print(
            "Do not run Step 09 yet. Paste the complete terminal output for audit."
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
        print("STEP 08 FAILED — 63-FEATURE PANEL NOT FROZEN")
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
        print("Do not proceed to Step 09.")

        return 30


# =============================================================================
# 21. CLI / ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge and freeze the clean-V3 ordered 63-feature panel."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an already frozen Step-08 panel without recomputing it."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verify_only:
        return verify_existing_step08()

    return run_new_step08()


if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print(
            "\nStep 08 interrupted by user."
        )
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 126)
        print("STEP 08 UNHANDLED FAILURE")
        print("=" * 126)
        print(f"{type(exc).__name__}: {exc}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        print("=" * 126)
        sys.exit(1)
