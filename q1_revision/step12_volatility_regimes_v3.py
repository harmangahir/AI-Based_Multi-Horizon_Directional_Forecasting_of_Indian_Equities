#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 12 — CLEAN V3 TRAIN-ONLY STOCK-SPECIFIC VOLATILITY REGIMES

This is the definitive Step-12 regime stage for the clean V3 pipeline.

Frozen regime policy
--------------------
Source:
    rolling_std_20 from the UNSCALED Step-10 development panel.

Thresholds:
    stock-specific q33 / q67
    pandas linear quantile interpolation

Encoding:
    0 = Low
    1 = Normal
    2 = High

Rule:
    Low    : volatility <= q33
    Normal : q33 < volatility <= q67
    High   : volatility > q67

Fold policy:
    For each expanding-origin fold, thresholds are estimated from that fold's
    TRAINING REFERENCE rows only, separately for every stock. The same frozen
    thresholds are applied unchanged to the fold's training and validation
    reference rows.

Final-development policy:
    One additional stock-specific threshold state is estimated from ALL
    feature-complete DEVELOPMENT REFERENCE rows. It is frozen for eventual
    one-time use with the final-development model. The sealed final-test
    parquet is NOT opened, hashed, transformed, labelled, or inspected here.

Integration with Step 11:
    Regimes are computed from unscaled rolling_std_20, then attached to the
    corresponding Step-11 scaled panels after exact ticker-date key
    reconciliation. The 63 scaled predictors remain byte-for-value unchanged.

Scientific boundary
-------------------
This stage DOES NOT:
- refit or alter MinMax scaling;
- use validation observations to fit regime thresholds;
- access final-test observations;
- perform feature selection;
- construct sequences;
- fit models;
- tune probability thresholds;
- compute final-test class/regime distributions or performance.

Run:
    python step12_volatility_regimes_v3.py

Verify:
    python step12_volatility_regimes_v3.py --verify-only
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
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# 0. SCRIPT / CONFIG
# =============================================================================

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent

if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError(
        "step12_volatility_regimes_v3.py must be inside <repository>/q1_revision/."
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg


# =============================================================================
# 1. FROZEN DESIGN
# =============================================================================

STEP_NUMBER = 12
STEP_ID = "12"
STEP_NAME = "STEP 12 — TRAIN-ONLY STOCK-SPECIFIC VOLATILITY REGIMES"
SCHEMA_VERSION = "1.0"

FOLDS = (1, 2, 3, 4, 5)

EXPECTED_FEATURE_COUNT = 63
EXPECTED_STOCK_COUNT = len(cfg.STOCK_TICKERS)

REGIME_SOURCE_COLUMN = "rolling_std_20"
REGIME_WINDOW_SESSIONS = 20

LOW_QUANTILE = 0.33
HIGH_QUANTILE = 0.67
QUANTILE_INTERPOLATION = "linear"

REGIME_LABELS = {
    0: "Low",
    1: "Normal",
    2: "High",
}


# =============================================================================
# 2. UPSTREAM INPUTS
# =============================================================================

STEP01_ENV = cfg.ENVIRONMENT_DIR / "01_environment_audit.json"

UPSTREAM_MANIFESTS = {
    1: (
        cfg.MANIFEST_DIR / "01_environment_audit_manifest.json",
        cfg.MANIFEST_DIR / "01_environment_audit_manifest.sha256",
    ),
    2: (
        cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.json",
        cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.sha256",
    ),
    3: (
        cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.json",
        cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.sha256",
    ),
    4: (
        cfg.MANIFEST_DIR / "04_session_policy_manifest.json",
        cfg.MANIFEST_DIR / "04_session_policy_manifest.sha256",
    ),
    5: (
        cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.json",
        cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.sha256",
    ),
    6: (
        cfg.MANIFEST_DIR / "06_stock_feature_manifest.json",
        cfg.MANIFEST_DIR / "06_stock_feature_manifest.sha256",
    ),
    7: (
        cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.json",
        cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.sha256",
    ),
    8: (
        cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.json",
        cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.sha256",
    ),
    9: (
        cfg.MANIFEST_DIR / "09_target_manifest.json",
        cfg.MANIFEST_DIR / "09_target_manifest.sha256",
    ),
    10: (
        cfg.MANIFEST_DIR / "10_chronological_split_manifest.json",
        cfg.MANIFEST_DIR / "10_chronological_split_manifest.sha256",
    ),
    11: (
        cfg.MANIFEST_DIR / "11_preprocessing_manifest.json",
        cfg.MANIFEST_DIR / "11_preprocessing_manifest.sha256",
    ),
}

STEP10_DEVELOPMENT = (
    cfg.FEATURE_DATA_DIR
    / "step10_chronological_splits"
    / "development"
    / "10_development_target_panel.parquet"
)

STEP10_MEMBERSHIP = (
    cfg.FEATURE_DATA_DIR
    / "step10_chronological_splits"
    / "10_split_membership.parquet"
)

STEP10_TEST_SEAL = cfg.MANIFEST_DIR / "10_final_test_seal.json"

# Deliberately never opened, hashed, stat'ed, or transformed in this stage.
FORBIDDEN_TEST_FILENAME = "10_final_test_locked_panel.parquet"

STEP11_ROOT = cfg.FEATURE_DATA_DIR / "step11_unified_preprocessing"

STEP11_FINAL_DEVELOPMENT = (
    STEP11_ROOT
    / "final_development"
    / "11_final_development_scaled.parquet"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

OUT_DIR = cfg.FEATURE_DATA_DIR / "step12_volatility_regimes"
FINAL_DEVELOPMENT_DIR = OUT_DIR / "final_development"

THRESHOLD_AUDIT_CSV = (
    cfg.AUDIT_DIR / "12_stock_specific_regime_thresholds.csv"
)

REGIME_DISTRIBUTION_CSV = (
    cfg.AUDIT_DIR / "12_regime_distribution.csv"
)

STOCK_REGIME_DISTRIBUTION_CSV = (
    cfg.AUDIT_DIR / "12_stock_regime_distribution.csv"
)

INTEGRITY_AUDIT_CSV = (
    cfg.AUDIT_DIR / "12_regime_integrity_audit.csv"
)

SOURCE_RECONCILIATION_CSV = (
    cfg.AUDIT_DIR / "12_regime_source_reconciliation.csv"
)

SOURCE_HASH_CSV = (
    cfg.AUDIT_DIR / "12_source_file_hashes.csv"
)

HASH_REGISTER_CSV = (
    cfg.AUDIT_DIR / "12_regime_hashes.csv"
)

SUMMARY_JSON = (
    cfg.AUDIT_DIR / "12_regime_summary.json"
)

MANIFEST_JSON = (
    cfg.MANIFEST_DIR / "12_volatility_regime_manifest.json"
)

MANIFEST_SHA = (
    cfg.MANIFEST_DIR / "12_volatility_regime_manifest.sha256"
)

FROZEN_SCRIPT = (
    cfg.MANIFEST_DIR / "step12_volatility_regimes_frozen.py"
)

SUMMARY_TXT = (
    cfg.REPORT_DIR / "12_volatility_regime_summary.txt"
)

FAILED_DIR = (
    cfg.AUDIT_DIR / "step12_failed_attempts"
)


# =============================================================================
# 4. GENERIC HELPERS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if value is pd.NA:
        return None

    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(text, encoding="utf-8")

    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    frame.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
    )

    os.replace(temporary, path)


def relative_to_experiment(path: Path) -> str:
    return str(
        path.resolve().relative_to(
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


def package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


# =============================================================================
# 5. STEP-00 / GLOBAL FINAL-TEST LOCK
# =============================================================================

def verify_step00() -> Dict[str, Any]:
    cfg.validate_configuration()

    payload = cfg.build_configuration_payload()

    fingerprint = cfg.canonical_json_sha256(payload)

    manifest_path, manifest_sha, status = cfg.verify_existing_freeze(
        configuration_payload=payload,
        configuration_fingerprint=fingerprint,
    )

    if status != "VERIFIED_EXISTING":
        raise RuntimeError(
            "Step-00 freeze is not VERIFIED_EXISTING."
        )

    lock_path = cfg.get_final_test_lock_path()

    lock = json.loads(
        lock_path.read_text(encoding="utf-8")
    )

    if lock.get("state") != "LOCKED_NOT_OPENED":
        raise RuntimeError(
            "Global final-test state is not LOCKED_NOT_OPENED."
        )

    if lock.get("final_test_opened") is not False:
        raise RuntimeError(
            "global final_test_opened != False."
        )

    if lock.get("final_test_scored") is not False:
        raise RuntimeError(
            "global final_test_scored != False."
        )

    return {
        "fingerprint": fingerprint,
        "step00_manifest_path": str(manifest_path),
        "step00_manifest_sha256": manifest_sha,
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "state": lock["state"],
    }


# =============================================================================
# 6. UPSTREAM VERIFICATION
# =============================================================================

def verify_sidecar(
    manifest_path: Path,
    sidecar_path: Path,
    label: str,
) -> str:
    if not manifest_path.exists():
        raise RuntimeError(
            f"{label} manifest missing: {manifest_path}"
        )

    if not sidecar_path.exists():
        raise RuntimeError(
            f"{label} SHA sidecar missing: {sidecar_path}"
        )

    expected = (
        sidecar_path.read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )

    observed = sha256_file(manifest_path)

    if observed != expected:
        raise RuntimeError(
            f"{label} manifest SHA mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}"
        )

    return observed


def find_artifact(
    manifest: Dict[str, Any],
    filename: str,
) -> Dict[str, Any]:
    matches = []

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

            if (
                path_text
                and
                Path(path_text).name
                ==
                filename
            ):
                matches.append(record)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest artifact named "
            f"{filename}; found {len(matches)}."
        )

    return matches[0]


def verify_step01_environment() -> Dict[str, Any]:
    if not STEP01_ENV.exists():
        raise RuntimeError(
            f"Step-01 environment audit missing: {STEP01_ENV}"
        )

    audit = json.loads(
        STEP01_ENV.read_text(encoding="utf-8")
    )

    expected_python = (
        audit
        .get("system", {})
        .get("python_executable")
    )

    if (
        expected_python
        and
        Path(expected_python).resolve()
        !=
        Path(sys.executable).resolve()
    ):
        raise RuntimeError(
            "Current Python differs from the frozen Step-01 environment.\n"
            f"Frozen : {expected_python}\n"
            f"Current: {sys.executable}"
        )

    drift = []

    for row in audit.get("packages", []):
        if not row.get("required", False):
            continue

        distribution = row.get("distribution")
        stored_version = row.get("version")
        current_version = package_version(distribution)

        if current_version != stored_version:
            drift.append(
                (
                    distribution,
                    stored_version,
                    current_version,
                )
            )

    if drift:
        detail = "\n".join(
            f"{name}: frozen={stored!r}, current={current!r}"
            for name, stored, current in drift
        )

        raise RuntimeError(
            "Required-package drift detected:\n"
            + detail
        )

    return {
        "python_executable": expected_python,
    }


def verify_step11_artifacts(
    step11_manifest: Dict[str, Any],
) -> None:
    failures = []

    for name, record in step11_manifest.get(
        "artifacts",
        {},
    ).items():

        if not isinstance(record, dict):
            continue

        path_text = record.get("path")

        if not path_text:
            continue

        path = Path(path_text)

        if path.name == FORBIDDEN_TEST_FILENAME:
            raise RuntimeError(
                "Programming error: final-test parquet appeared in "
                "the Step-11 artifact set."
            )

        if not path.exists():
            failures.append(
                f"missing: {name} -> {path}"
            )
            continue

        expected = record.get("sha256")

        if (
            expected
            and
            sha256_file(path)
            !=
            expected
        ):
            failures.append(
                f"hash mismatch: {name}"
            )

    if failures:
        raise RuntimeError(
            "Step-11 artifact verification failed:\n"
            + "\n".join(failures)
        )


def verify_upstream(
    step00: Dict[str, Any],
) -> Dict[str, Any]:
    manifest_sha: Dict[int, str] = {}
    manifests: Dict[int, Dict[str, Any]] = {}

    for step in range(1, 12):
        manifest_path, sidecar_path = (
            UPSTREAM_MANIFESTS[step]
        )

        manifest_sha[step] = verify_sidecar(
            manifest_path,
            sidecar_path,
            f"Step {step:02d}",
        )

        manifests[step] = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

    # Step 01.
    if manifests[1].get("status") != "PASS":
        raise RuntimeError(
            "Step-01 status is not PASS."
        )

    if (
        manifests[1].get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00["fingerprint"]
    ):
        raise RuntimeError(
            "Step-01 fingerprint mismatch."
        )

    # Step 02.
    if (
        manifests[2]
        .get("step", {})
        .get("status")
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Step-02 status is not PASS."
        )

    if (
        manifests[2]
        .get("step00", {})
        .get("configuration_fingerprint_sha256")
        !=
        step00["fingerprint"]
    ):
        raise RuntimeError(
            "Step-02 fingerprint mismatch."
        )

    if (
        manifests[2]
        .get("step01", {})
        .get("step01_manifest_sha256")
        !=
        manifest_sha[1]
    ):
        raise RuntimeError(
            "Step-02 -> Step-01 chain mismatch."
        )

    # Steps 03 -> 11.
    for step in range(3, 12):
        manifest = manifests[step]

        if (
            manifest
            .get("step", {})
            .get("status")
            !=
            "PASS"
        ):
            raise RuntimeError(
                f"Step-{step:02d} status is not PASS."
            )

        if (
            manifest.get(
                "configuration_fingerprint_sha256"
            )
            !=
            step00["fingerprint"]
        ):
            raise RuntimeError(
                f"Step-{step:02d} fingerprint mismatch."
            )

        for previous in range(1, step):
            field = (
                f"step{previous:02d}_manifest_sha256"
            )

            if (
                field in manifest
                and
                manifest[field]
                !=
                manifest_sha[previous]
            ):
                raise RuntimeError(
                    f"Step-{step:02d} chain mismatch: {field}"
                )

    verify_step01_environment()

    # Feature/split interface must remain unchanged.
    feature_columns = manifests[11].get(
        "ordered_63_features"
    )

    feature_interface_sha = manifests[11].get(
        "feature_interface_sha256"
    )

    split_design_sha = manifests[11].get(
        "split_design_sha256"
    )

    if (
        not isinstance(feature_columns, list)
        or
        len(feature_columns)
        !=
        EXPECTED_FEATURE_COUNT
    ):
        raise RuntimeError(
            "Step-11 manifest does not freeze exactly 63 predictors."
        )

    if REGIME_SOURCE_COLUMN not in feature_columns:
        raise RuntimeError(
            f"Frozen 63-feature interface does not contain "
            f"{REGIME_SOURCE_COLUMN}."
        )

    if not feature_interface_sha:
        raise RuntimeError(
            "Step-11 manifest lacks feature_interface_sha256."
        )

    if not split_design_sha:
        raise RuntimeError(
            "Step-11 manifest lacks split_design_sha256."
        )

    for step in (8, 9, 10):
        if (
            manifests[step].get(
                "ordered_63_features"
            )
            !=
            feature_columns
        ):
            raise RuntimeError(
                f"Step-{step:02d}/11 predictor order mismatch."
            )

        if (
            manifests[step].get(
                "feature_interface_sha256"
            )
            !=
            feature_interface_sha
        ):
            raise RuntimeError(
                f"Step-{step:02d}/11 feature-interface SHA mismatch."
            )

    if (
        manifests[10].get(
            "split_design_sha256"
        )
        !=
        split_design_sha
    ):
        raise RuntimeError(
            "Step-10/11 split-design SHA mismatch."
        )

    # Step-10 development + membership identity.
    step10_manifest = manifests[10]

    development_record = find_artifact(
        step10_manifest,
        STEP10_DEVELOPMENT.name,
    )

    membership_record = find_artifact(
        step10_manifest,
        STEP10_MEMBERSHIP.name,
    )

    if (
        Path(development_record["path"]).resolve()
        !=
        STEP10_DEVELOPMENT.resolve()
    ):
        raise RuntimeError(
            "Step-10 development-panel path is not canonical."
        )

    if (
        Path(membership_record["path"]).resolve()
        !=
        STEP10_MEMBERSHIP.resolve()
    ):
        raise RuntimeError(
            "Step-10 membership path is not canonical."
        )

    development_sha = sha256_file(
        STEP10_DEVELOPMENT
    )

    membership_sha = sha256_file(
        STEP10_MEMBERSHIP
    )

    if development_sha != development_record["sha256"]:
        raise RuntimeError(
            "Step-10 development-panel hash mismatch."
        )

    if membership_sha != membership_record["sha256"]:
        raise RuntimeError(
            "Step-10 membership hash mismatch."
        )

    # Step-11 source-integrity cross-check.
    step11_source = manifests[11].get(
        "source_integrity",
        {},
    )

    if (
        step11_source.get(
            "step10_development_panel_sha256"
        )
        !=
        development_sha
    ):
        raise RuntimeError(
            "Step-11 source record for development panel differs from Step 10."
        )

    if (
        step11_source.get(
            "step10_membership_sha256"
        )
        !=
        membership_sha
    ):
        raise RuntimeError(
            "Step-11 source record for membership ledger differs from Step 10."
        )

    # Step-11 artifact set is safe to re-hash.
    verify_step11_artifacts(
        manifests[11]
    )

    # Test-seal metadata only; never access the parquet itself.
    if not STEP10_TEST_SEAL.exists():
        raise RuntimeError(
            "Step-10 final-test seal is missing."
        )

    test_seal = json.loads(
        STEP10_TEST_SEAL.read_text(
            encoding="utf-8"
        )
    )

    if test_seal.get("status") != "SEALED_NOT_OPENED":
        raise RuntimeError(
            "Step-10 test seal is not SEALED_NOT_OPENED."
        )

    if (
        test_seal.get(
            "global_final_test_lock_state"
        )
        !=
        "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError(
            "Step-10 seal does not preserve LOCKED_NOT_OPENED."
        )

    # Exact Step-11 canonical files.
    for fold in FOLDS:
        for partition in (
            "train",
            "validation",
        ):
            path = (
                STEP11_ROOT
                /
                f"fold_{fold}"
                /
                f"11_fold_{fold}_{partition}_scaled.parquet"
            )

            record = find_artifact(
                manifests[11],
                path.name,
            )

            if Path(record["path"]).resolve() != path.resolve():
                raise RuntimeError(
                    f"Step-11 canonical path mismatch: {path.name}"
                )

            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(
                    f"Step-11 scaled-panel hash mismatch: {path.name}"
                )

    final_record = find_artifact(
        manifests[11],
        STEP11_FINAL_DEVELOPMENT.name,
    )

    if (
        Path(final_record["path"]).resolve()
        !=
        STEP11_FINAL_DEVELOPMENT.resolve()
    ):
        raise RuntimeError(
            "Step-11 final-development scaled-panel path is not canonical."
        )

    final_development_scaled_sha = sha256_file(
        STEP11_FINAL_DEVELOPMENT
    )

    if (
        final_development_scaled_sha
        !=
        final_record["sha256"]
    ):
        raise RuntimeError(
            "Step-11 final-development scaled-panel hash mismatch."
        )

    return {
        "manifest_sha": manifest_sha,
        "manifests": manifests,
        "feature_columns": list(feature_columns),
        "feature_interface_sha256": feature_interface_sha,
        "split_design_sha256": split_design_sha,
        "development_sha256": development_sha,
        "membership_sha256": membership_sha,
        "step11_final_development_scaled_sha256": (
            final_development_scaled_sha
        ),
        "test_seal_status": test_seal["status"],
        "sealed_test_sha256_from_metadata_only": (
            test_seal.get("sha256")
        ),
    }


# =============================================================================
# 7. OUTPUT IMMUTABILITY / STAGING
# =============================================================================

def canonical_outputs() -> tuple[Path, ...]:
    return (
        OUT_DIR,
        THRESHOLD_AUDIT_CSV,
        REGIME_DISTRIBUTION_CSV,
        STOCK_REGIME_DISTRIBUTION_CSV,
        INTEGRITY_AUDIT_CSV,
        SOURCE_RECONCILIATION_CSV,
        SOURCE_HASH_CSV,
        HASH_REGISTER_CSV,
        SUMMARY_JSON,
        MANIFEST_JSON,
        MANIFEST_SHA,
        FROZEN_SCRIPT,
        SUMMARY_TXT,
    )


def assert_fresh_target() -> None:
    collisions = [
        path
        for path in canonical_outputs()
        if path.exists()
    ]

    if collisions:
        raise RuntimeError(
            "Step-12 canonical output(s) already exist. "
            "Refusing to overwrite.\n"
            + "\n".join(
                str(path)
                for path in collisions
            )
            + "\nUse --verify-only if Step 12 already passed."
        )


def create_staging_root() -> Path:
    stamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )

    root = (
        cfg.FEATURE_DATA_DIR
        /
        f"_step12_staging_{stamp}_{os.getpid()}"
    )

    root.mkdir(
        parents=True,
        exist_ok=False,
    )

    for fold in FOLDS:
        (
            root
            /
            f"fold_{fold}"
        ).mkdir(
            exist_ok=False
        )

    (
        root
        /
        "final_development"
    ).mkdir(
        exist_ok=False
    )

    return root


def archive_partial() -> Optional[Path]:
    if MANIFEST_JSON.exists():
        return None

    existing = [
        path
        for path in canonical_outputs()
        if path.exists()
    ]

    if not existing:
        return None

    FAILED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )

    destination = (
        FAILED_DIR
        /
        f"partial_artifacts_{stamp}"
    )

    destination.mkdir(
        exist_ok=False
    )

    for path in existing:
        try:
            shutil.move(
                str(path),
                str(
                    destination
                    /
                    path.name
                ),
            )
        except Exception:
            pass

    return destination


# =============================================================================
# 8. LOAD RAW DEVELOPMENT / MEMBERSHIP
# =============================================================================

def load_development_sources(
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    development = pd.read_parquet(
        STEP10_DEVELOPMENT,
        engine="pyarrow",
    )

    membership = pd.read_parquet(
        STEP10_MEMBERSHIP,
        engine="pyarrow",
    )

    for frame in (
        development,
        membership,
    ):
        frame["Date"] = (
            pd.to_datetime(
                frame["Date"],
                errors="raise",
            )
            .dt.normalize()
        )

    for label, frame in (
        ("development", development),
        ("membership", membership),
    ):
        if frame[
            [
                "Ticker",
                "Date",
            ]
        ].duplicated().any():
            raise RuntimeError(
                f"Duplicate ticker-date keys in {label}."
            )

    if REGIME_SOURCE_COLUMN not in development.columns:
        raise RuntimeError(
            f"Missing regime source column: {REGIME_SOURCE_COLUMN}"
        )

    development[
        REGIME_SOURCE_COLUMN
    ] = pd.to_numeric(
        development[
            REGIME_SOURCE_COLUMN
        ],
        errors="raise",
    )

    source = development[
        REGIME_SOURCE_COLUMN
    ]

    if source.isna().any():
        raise RuntimeError(
            "Regime source contains missing values."
        )

    if not np.isfinite(
        source.to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            "Regime source contains non-finite values."
        )

    if (
        source
        <
        0.0
    ).any():
        raise RuntimeError(
            "Negative rolling volatility detected."
        )

    if set(
        development["Ticker"].unique()
    ) != set(
        cfg.STOCK_TICKERS
    ):
        raise RuntimeError(
            "Development stock universe differs from Step 00."
        )

    if development[
        "Ticker"
    ].nunique() != EXPECTED_STOCK_COUNT:
        raise RuntimeError(
            "Unexpected development stock count."
        )

    # Confirm that the regime source belongs to the frozen predictor interface.
    if REGIME_SOURCE_COLUMN not in feature_columns:
        raise RuntimeError(
            "Regime source is not part of the frozen 63-feature interface."
        )

    membership_development = (
        membership.loc[
            membership[
                "outer_partition"
            ]
            ==
            "development"
        ]
        .copy()
    )

    development_keys = (
        development[
            [
                "Ticker",
                "Date",
            ]
        ]
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    membership_keys = (
        membership_development[
            [
                "Ticker",
                "Date",
            ]
        ]
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if not development_keys.equals(
        membership_keys
    ):
        raise RuntimeError(
            "Step-10 development keys differ from development membership keys."
        )

    daily_counts = (
        development
        .groupby("Date")[
            "Ticker"
        ]
        .nunique()
    )

    if not (
        daily_counts
        ==
        EXPECTED_STOCK_COUNT
    ).all():
        raise RuntimeError(
            "Development panel is not a complete stock cross-section."
        )

    return (
        development
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        ),
        membership
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        ),
    )


# =============================================================================
# 9. EXTRACT RAW PARTITIONS
# =============================================================================

def extract_raw_partition(
    development: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    reference_column: str,
    label: str,
) -> pd.DataFrame:
    if reference_column not in membership.columns:
        raise RuntimeError(
            f"{label}: membership column missing: {reference_column}"
        )

    keys = (
        membership.loc[
            membership[
                reference_column
            ],
            [
                "Ticker",
                "Date",
            ],
        ]
        .copy()
    )

    result = (
        development
        .merge(
            keys,
            on=[
                "Ticker",
                "Date",
            ],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(result) != len(keys):
        raise RuntimeError(
            f"{label}: raw partition extraction lost rows."
        )

    if result.empty:
        raise RuntimeError(
            f"{label}: raw partition is empty."
        )

    counts = (
        result
        .groupby("Date")[
            "Ticker"
        ]
        .nunique()
    )

    if not (
        counts
        ==
        EXPECTED_STOCK_COUNT
    ).all():
        raise RuntimeError(
            f"{label}: incomplete stock cross-section."
        )

    return result


def extract_final_development_raw(
    development: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    temporary = membership.copy()

    temporary[
        "_final_development_reference"
    ] = (
        temporary[
            "outer_partition"
        ]
        ==
        "development"
    )

    return extract_raw_partition(
        development=development,
        membership=temporary,
        reference_column=(
            "_final_development_reference"
        ),
        label=(
            "final_development"
        ),
    )


# =============================================================================
# 10. LOAD STEP-11 SCALED PANELS
# =============================================================================

def step11_scaled_path(
    fold: int,
    partition: str,
) -> Path:
    return (
        STEP11_ROOT
        /
        f"fold_{fold}"
        /
        f"11_fold_{fold}_{partition}_scaled.parquet"
    )


def load_scaled_panel(
    path: Path,
    feature_columns: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    if path.name == FORBIDDEN_TEST_FILENAME:
        raise RuntimeError(
            "Programming error: final-test parquet requested in Step 12."
        )

    frame = pd.read_parquet(
        path,
        engine="pyarrow",
    )

    frame["Date"] = (
        pd.to_datetime(
            frame["Date"],
            errors="raise",
        )
        .dt.normalize()
    )

    if frame[
        [
            "Ticker",
            "Date",
        ]
    ].duplicated().any():
        raise RuntimeError(
            f"{label}: duplicate ticker-date keys."
        )

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in frame.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"{label}: missing scaled predictors:\n"
            f"{missing_features}"
        )

    values = (
        frame[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{label}: non-finite scaled predictor values."
        )

    return (
        frame
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# 11. QUANTILES / REGIME ASSIGNMENT
# =============================================================================

def quantile_linear(
    series: pd.Series,
    q: float,
) -> float:
    return float(
        series.quantile(
            q=q,
            interpolation=(
                QUANTILE_INTERPOLATION
            ),
        )
    )


def assign_regime(
    volatility: pd.Series,
    q33: float,
    q67: float,
) -> pd.Series:
    if not (
        np.isfinite(q33)
        and
        np.isfinite(q67)
        and
        q33
        <
        q67
    ):
        raise RuntimeError(
            "Regime thresholds must be finite and strictly ordered."
        )

    values = volatility.to_numpy(
        dtype=np.float64
    )

    regime = np.select(
        condlist=[
            values <= q33,
            (
                (values > q33)
                &
                (values <= q67)
            ),
            values > q67,
        ],
        choicelist=[
            0,
            1,
            2,
        ],
        default=-1,
    )

    if (
        regime
        <
        0
    ).any():
        raise RuntimeError(
            "Unassigned regime value detected."
        )

    return pd.Series(
        regime,
        index=volatility.index,
        dtype="int8",
    )


def fit_threshold_state(
    fit_frame: pd.DataFrame,
    *,
    scope: str,
    fold: Optional[int],
) -> pd.DataFrame:
    rows = []

    tickers = sorted(
        fit_frame[
            "Ticker"
        ].unique()
    )

    if set(tickers) != set(
        cfg.STOCK_TICKERS
    ):
        raise RuntimeError(
            f"{scope}: threshold-fit stock universe mismatch."
        )

    for ticker in tickers:
        stock = (
            fit_frame.loc[
                fit_frame[
                    "Ticker"
                ]
                ==
                ticker
            ]
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        volatility = stock[
            REGIME_SOURCE_COLUMN
        ]

        q33 = quantile_linear(
            volatility,
            LOW_QUANTILE,
        )

        q67 = quantile_linear(
            volatility,
            HIGH_QUANTILE,
        )

        if not (
            np.isfinite(q33)
            and
            np.isfinite(q67)
        ):
            raise RuntimeError(
                f"{scope}, {ticker}: non-finite threshold."
            )

        if not q33 < q67:
            raise RuntimeError(
                f"{scope}, {ticker}: q33 >= q67."
            )

        rows.append(
            {
                "scope": scope,
                "fold": (
                    np.nan
                    if fold is None
                    else int(fold)
                ),
                "ticker": ticker,
                "regime_source_column": (
                    REGIME_SOURCE_COLUMN
                ),
                "window_sessions": (
                    REGIME_WINDOW_SESSIONS
                ),
                "threshold_fit_rows": int(
                    len(stock)
                ),
                "threshold_fit_dates": int(
                    stock[
                        "Date"
                    ]
                    .nunique()
                ),
                "first_fit_date": (
                    stock[
                        "Date"
                    ]
                    .min()
                ),
                "last_fit_date": (
                    stock[
                        "Date"
                    ]
                    .max()
                ),
                "low_quantile": (
                    LOW_QUANTILE
                ),
                "high_quantile": (
                    HIGH_QUANTILE
                ),
                "quantile_interpolation": (
                    QUANTILE_INTERPOLATION
                ),
                "low_volatility_threshold_q33": (
                    q33
                ),
                "high_volatility_threshold_q67": (
                    q67
                ),
                "training_volatility_minimum": float(
                    volatility.min()
                ),
                "training_volatility_maximum": float(
                    volatility.max()
                ),
                "training_volatility_mean": float(
                    volatility.mean()
                ),
                "training_volatility_std_ddof1": float(
                    volatility.std(
                        ddof=1
                    )
                ),
                "source_is_unscaled": True,
                "thresholds_fit_from_predictor_values_only": True,
                "validation_used_for_threshold_fit": False,
                "final_test_used_for_threshold_fit": False,
            }
        )

    thresholds = pd.DataFrame(
        rows
    )

    if len(thresholds) != EXPECTED_STOCK_COUNT:
        raise RuntimeError(
            f"{scope}: expected {EXPECTED_STOCK_COUNT} threshold rows; "
            f"found {len(thresholds)}."
        )

    return thresholds


def apply_threshold_state(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    fit_scope: str,
) -> pd.DataFrame:
    threshold_map = (
        thresholds
        .set_index(
            "ticker"
        )[
            [
                "low_volatility_threshold_q33",
                "high_volatility_threshold_q67",
            ]
        ]
    )

    if set(
        threshold_map.index
    ) != set(
        frame[
            "Ticker"
        ].unique()
    ):
        raise RuntimeError(
            f"{fit_scope}: threshold/application stock universe mismatch."
        )

    parts = []

    for ticker, stock in frame.groupby(
        "Ticker",
        sort=True,
    ):
        stock = (
            stock
            .copy()
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        q33 = float(
            threshold_map.loc[
                ticker,
                "low_volatility_threshold_q33",
            ]
        )

        q67 = float(
            threshold_map.loc[
                ticker,
                "high_volatility_threshold_q67",
            ]
        )

        stock[
            "regime_id"
        ] = assign_regime(
            stock[
                REGIME_SOURCE_COLUMN
            ],
            q33,
            q67,
        )

        stock[
            "regime_name"
        ] = (
            stock[
                "regime_id"
            ]
            .map(
                REGIME_LABELS
            )
        )

        if stock[
            "regime_name"
        ].isna().any():
            raise RuntimeError(
                f"{fit_scope}, {ticker}: missing regime name."
            )

        parts.append(
            stock
        )

    return (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# 12. ATTACH REGIMES TO SCALED PANELS
# =============================================================================

def attach_regimes_to_scaled(
    scaled: pd.DataFrame,
    raw_labeled: pd.DataFrame,
    feature_columns: list[str],
    *,
    fit_scope: str,
    label: str,
) -> tuple[
    pd.DataFrame,
    Dict[str, Any],
]:
    scaled_keys = (
        scaled[
            [
                "Ticker",
                "Date",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    raw_keys = (
        raw_labeled[
            [
                "Ticker",
                "Date",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    if not scaled_keys.equals(
        raw_keys
    ):
        raise RuntimeError(
            f"{label}: Step-10 raw keys differ from Step-11 scaled keys."
        )

    original_predictors = (
        scaled[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
        .copy()
    )

    output = scaled.copy()

    output[
        "regime_id"
    ] = (
        raw_labeled[
            "regime_id"
        ]
        .astype(
            "int8"
        )
        .to_numpy()
    )

    output[
        "regime_name"
    ] = (
        raw_labeled[
            "regime_name"
        ]
        .astype(
            "object"
        )
        .to_numpy()
    )

    output[
        "regime_threshold_fit_scope"
    ] = fit_scope

    after_predictors = (
        output[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    predictors_exact = bool(
        np.array_equal(
            original_predictors,
            after_predictors,
        )
    )

    if not predictors_exact:
        raise RuntimeError(
            f"{label}: attaching regimes altered scaled predictors."
        )

    if not output[
        "regime_id"
    ].isin(
        [
            0,
            1,
            2,
        ]
    ).all():
        raise RuntimeError(
            f"{label}: invalid regime IDs."
        )

    return (
        output,
        {
            "scope": label,
            "rows": int(
                len(output)
            ),
            "dates": int(
                output[
                    "Date"
                ]
                .nunique()
            ),
            "stocks": int(
                output[
                    "Ticker"
                ]
                .nunique()
            ),
            "key_match_step10_to_step11": True,
            "scaled_predictors_preserved_exactly": (
                predictors_exact
            ),
            "regime_ids_valid": True,
            "threshold_fit_scope": fit_scope,
        },
    )


# =============================================================================
# 13. DISTRIBUTION AUDITS
# =============================================================================

def distribution_rows(
    frame: pd.DataFrame,
    *,
    scope: str,
    fold: Optional[int],
    partition: str,
) -> tuple[
    list[dict],
    list[dict],
]:
    aggregate = []
    stock_level = []

    counts = (
        frame[
            "regime_id"
        ]
        .value_counts()
        .reindex(
            [
                0,
                1,
                2,
            ],
            fill_value=0,
        )
    )

    total = int(
        len(frame)
    )

    for regime_id in (
        0,
        1,
        2,
    ):
        count = int(
            counts.loc[
                regime_id
            ]
        )

        aggregate.append(
            {
                "scope": scope,
                "fold": (
                    np.nan
                    if fold is None
                    else int(fold)
                ),
                "partition": partition,
                "regime_id": regime_id,
                "regime_name": (
                    REGIME_LABELS[
                        regime_id
                    ]
                ),
                "rows": count,
                "ratio": (
                    count
                    /
                    total
                ),
            }
        )

    for ticker, stock in frame.groupby(
        "Ticker",
        sort=True,
    ):
        counts = (
            stock[
                "regime_id"
            ]
            .value_counts()
            .reindex(
                [
                    0,
                    1,
                    2,
                ],
                fill_value=0,
            )
        )

        stock_total = int(
            len(stock)
        )

        for regime_id in (
            0,
            1,
            2,
        ):
            count = int(
                counts.loc[
                    regime_id
                ]
            )

            stock_level.append(
                {
                    "scope": scope,
                    "fold": (
                        np.nan
                        if fold is None
                        else int(fold)
                    ),
                    "partition": partition,
                    "ticker": ticker,
                    "regime_id": regime_id,
                    "regime_name": (
                        REGIME_LABELS[
                            regime_id
                        ]
                    ),
                    "rows": count,
                    "ratio": (
                        count
                        /
                        stock_total
                    ),
                }
            )

    return (
        aggregate,
        stock_level,
    )


# =============================================================================
# 14. PARQUET / SOURCE HASH HELPERS
# =============================================================================

def write_parquet(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    frame.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )


def verify_parquet_roundtrip(
    original: pd.DataFrame,
    path: Path,
    feature_columns: list[str],
) -> None:
    observed = pd.read_parquet(
        path,
        engine="pyarrow",
    )

    if len(observed) != len(original):
        raise RuntimeError(
            f"Parquet row-count mismatch: {path}"
        )

    if list(observed.columns) != list(
        original.columns
    ):
        raise RuntimeError(
            f"Parquet column-order mismatch: {path}"
        )

    for frame in (
        original,
        observed,
    ):
        frame["Date"] = (
            pd.to_datetime(
                frame["Date"],
                errors="raise",
            )
            .dt.normalize()
        )

    if not original[
        [
            "Ticker",
            "Date",
        ]
    ].equals(
        observed[
            [
                "Ticker",
                "Date",
            ]
        ]
    ):
        raise RuntimeError(
            f"Parquet key mismatch: {path}"
        )

    if not np.array_equal(
        original[
            feature_columns
        ].to_numpy(
            dtype=np.float64
        ),
        observed[
            feature_columns
        ].to_numpy(
            dtype=np.float64
        ),
    ):
        raise RuntimeError(
            f"Parquet scaled-feature mismatch: {path}"
        )

    if not np.array_equal(
        original[
            "regime_id"
        ].to_numpy(),
        observed[
            "regime_id"
        ].to_numpy(),
    ):
        raise RuntimeError(
            f"Parquet regime-ID mismatch: {path}"
        )


def source_hashes() -> pd.DataFrame:
    sources = [
        (
            "current_step00",
            Path(
                cfg.__file__
            ).resolve(),
        ),
        (
            "current_step12",
            SCRIPT_FILE.resolve(),
        ),
        (
            "step10_development",
            STEP10_DEVELOPMENT.resolve(),
        ),
        (
            "step10_membership",
            STEP10_MEMBERSHIP.resolve(),
        ),
        (
            "step10_test_seal_metadata",
            STEP10_TEST_SEAL.resolve(),
        ),
        (
            "step11_final_development_scaled",
            STEP11_FINAL_DEVELOPMENT.resolve(),
        ),
    ]

    for fold in FOLDS:
        for partition in (
            "train",
            "validation",
        ):
            sources.append(
                (
                    f"step11_fold_{fold}_{partition}_scaled",
                    step11_scaled_path(
                        fold,
                        partition,
                    ).resolve(),
                )
            )

    for step, (
        manifest_path,
        sidecar_path,
    ) in UPSTREAM_MANIFESTS.items():
        sources.extend(
            [
                (
                    f"step{step:02d}_manifest",
                    manifest_path.resolve(),
                ),
                (
                    f"step{step:02d}_manifest_sha",
                    sidecar_path.resolve(),
                ),
            ]
        )

    rows = []

    for role, path in sources:
        if path.name == FORBIDDEN_TEST_FILENAME:
            raise RuntimeError(
                "Programming error: sealed test parquet entered "
                "Step-12 source hash list."
            )

        if not path.exists():
            raise RuntimeError(
                f"Required source missing: {path}"
            )

        rows.append(
            {
                "role": role,
                "file_name": path.name,
                "absolute_path": str(path),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(path),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 15. VERIFY EXISTING STEP 12
# =============================================================================

def verify_existing() -> int:
    print("=" * 134)
    print("STEP 12 — VERIFY EXISTING VOLATILITY-REGIME FREEZE")
    print("=" * 134)

    step00 = verify_step00()
    upstream = verify_upstream(step00)

    manifest_sha = verify_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 12",
    )

    manifest = json.loads(
        MANIFEST_JSON.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest
        .get("step", {})
        .get("status")
        !=
        "PASS"
    ):
        print(
            "[FAIL] Stored Step-12 status is not PASS."
        )
        return 20

    if (
        manifest.get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00["fingerprint"]
    ):
        print(
            "[FAIL] Configuration fingerprint mismatch."
        )
        return 21

    for step in range(1, 12):
        field = (
            f"step{step:02d}_manifest_sha256"
        )

        if (
            manifest.get(field)
            !=
            upstream["manifest_sha"][step]
        ):
            print(
                f"[FAIL] Upstream chain mismatch: {field}"
            )
            return 22

    if (
        manifest.get(
            "feature_interface_sha256"
        )
        !=
        upstream["feature_interface_sha256"]
    ):
        print(
            "[FAIL] Feature-interface SHA mismatch."
        )
        return 23

    if (
        manifest.get(
            "split_design_sha256"
        )
        !=
        upstream["split_design_sha256"]
    ):
        print(
            "[FAIL] Split-design SHA mismatch."
        )
        return 24

    failures = []

    for name, record in manifest.get(
        "artifacts",
        {},
    ).items():
        path = Path(
            record["path"]
        )

        if not path.exists():
            failures.append(
                f"missing: {name}"
            )
            continue

        if (
            sha256_file(path)
            !=
            record["sha256"]
        ):
            failures.append(
                f"hash mismatch: {name}"
            )

    if failures:
        for failure in failures:
            print(
                f"[FAIL] {failure}"
            )

        print("=" * 134)
        print("STEP 12 VERIFICATION STATUS: FAIL")
        print("=" * 134)
        return 25

    if (
        sha256_file(
            Path(
                step00["lock_path"]
            )
        )
        !=
        step00["lock_sha256"]
    ):
        print(
            "[FAIL] Global final-test lock changed."
        )
        return 26

    print(
        "[PASS] Configuration fingerprint : "
        f"{step00['fingerprint']}"
    )
    print(
        "[PASS] Step-11 manifest          : "
        f"{upstream['manifest_sha'][11]}"
    )
    print(
        "[PASS] 63-feature interface      : "
        f"{upstream['feature_interface_sha256']}"
    )
    print(
        "[PASS] Split-design SHA-256      : "
        f"{upstream['split_design_sha256']}"
    )
    print(
        "[PASS] Step-12 manifest          : "
        f"{manifest_sha}"
    )
    print(
        "[PASS] Step-10 test seal         : "
        f"{upstream['test_seal_status']}"
    )
    print(
        "[PASS] Global final-test state   : "
        f"{step00['state']}"
    )
    print("=" * 134)
    print("STEP 12 VERIFICATION STATUS: PASS")
    print("=" * 134)

    return 0


# =============================================================================
# 16. MAIN RUN
# =============================================================================

def run() -> int:
    print("=" * 134)
    print(STEP_NAME)
    print("=" * 134)

    stage = "prerequisite_verification"
    staging: Optional[Path] = None

    try:
        print(
            "[1/10] Verifying Step 00 and immutable Step 01 -> Step 11 chain..."
        )

        step00 = verify_step00()
        upstream = verify_upstream(step00)

        print(
            "       Configuration fingerprint: "
            f"{step00['fingerprint']}"
        )
        print(
            "       Step-11 manifest          : "
            f"{upstream['manifest_sha'][11]}"
        )
        print(
            "       Development SHA-256       : "
            f"{upstream['development_sha256']}"
        )
        print(
            "       Membership SHA-256        : "
            f"{upstream['membership_sha256']}"
        )
        print(
            "       63-feature interface      : "
            f"{upstream['feature_interface_sha256']}"
        )
        print(
            "       Split-design SHA-256      : "
            f"{upstream['split_design_sha256']}"
        )
        print(
            "       Step-10 test seal         : "
            f"{upstream['test_seal_status']}"
        )
        print(
            "       Global final-test state   : "
            f"{step00['state']}"
        )
        print(
            "       Final-test parquet access : PROHIBITED"
        )

        stage = "collision_guard"

        print(
            "[2/10] Checking immutable Step-12 output target..."
        )

        assert_fresh_target()
        cfg.create_directories()

        stage = "source_load"

        print(
            "[3/10] Loading unscaled development source and Step-10 membership..."
        )

        feature_columns = upstream[
            "feature_columns"
        ]

        development, membership = (
            load_development_sources(
                feature_columns
            )
        )

        print(
            "       Development rows       : "
            f"{len(development):,}"
        )
        print(
            "       Development dates      : "
            f"{development['Date'].nunique():,}"
        )
        print(
            "       Regime source          : "
            f"{REGIME_SOURCE_COLUMN} (UNSCALED)"
        )
        print(
            "       q33 / q67              : "
            f"{LOW_QUANTILE} / {HIGH_QUANTILE}"
        )
        print(
            "       Quantile interpolation : "
            f"{QUANTILE_INTERPOLATION}"
        )

        stage = "staging"

        staging = create_staging_root()

        threshold_frames = []
        regime_distribution_rows = []
        stock_distribution_rows = []
        integrity_rows = []
        source_reconciliation_rows = []
        staged_records = []

        stage = "fold_regimes"

        print(
            "[4/10] Estimating fold-stock thresholds and labelling scaled fold panels..."
        )

        for fold in FOLDS:
            train_raw = extract_raw_partition(
                development=development,
                membership=membership,
                reference_column=(
                    f"fold_{fold}_train_reference"
                ),
                label=(
                    f"fold_{fold}_train_raw"
                ),
            )

            validation_raw = extract_raw_partition(
                development=development,
                membership=membership,
                reference_column=(
                    f"fold_{fold}_validation_reference"
                ),
                label=(
                    f"fold_{fold}_validation_raw"
                ),
            )

            if (
                train_raw["Date"].max()
                >=
                validation_raw["Date"].min()
            ):
                raise RuntimeError(
                    f"Fold {fold}: train/validation chronology violated."
                )

            thresholds = fit_threshold_state(
                fit_frame=train_raw,
                scope="fold_training",
                fold=fold,
            )

            train_labeled_raw = apply_threshold_state(
                train_raw,
                thresholds,
                fit_scope=(
                    f"fold_{fold}_training"
                ),
            )

            validation_labeled_raw = (
                apply_threshold_state(
                    validation_raw,
                    thresholds,
                    fit_scope=(
                        f"fold_{fold}_training"
                    ),
                )
            )

            train_scaled_source = load_scaled_panel(
                step11_scaled_path(
                    fold,
                    "train",
                ),
                feature_columns,
                label=(
                    f"fold_{fold}_train_scaled"
                ),
            )

            validation_scaled_source = (
                load_scaled_panel(
                    step11_scaled_path(
                        fold,
                        "validation",
                    ),
                    feature_columns,
                    label=(
                        f"fold_{fold}_validation_scaled"
                    ),
                )
            )

            (
                train_output,
                train_reconciliation,
            ) = attach_regimes_to_scaled(
                scaled=train_scaled_source,
                raw_labeled=(
                    train_labeled_raw
                ),
                feature_columns=(
                    feature_columns
                ),
                fit_scope=(
                    f"fold_{fold}_training"
                ),
                label=(
                    f"fold_{fold}_train"
                ),
            )

            (
                validation_output,
                validation_reconciliation,
            ) = attach_regimes_to_scaled(
                scaled=validation_scaled_source,
                raw_labeled=(
                    validation_labeled_raw
                ),
                feature_columns=(
                    feature_columns
                ),
                fit_scope=(
                    f"fold_{fold}_training"
                ),
                label=(
                    f"fold_{fold}_validation"
                ),
            )

            source_reconciliation_rows.extend(
                [
                    train_reconciliation,
                    validation_reconciliation,
                ]
            )

            # All three training regimes must exist for every stock.
            training_regime_complete = True

            for ticker, stock in train_labeled_raw.groupby(
                "Ticker",
                sort=True,
            ):
                observed = set(
                    stock[
                        "regime_id"
                    ].unique()
                    .tolist()
                )

                if observed != {
                    0,
                    1,
                    2,
                }:
                    training_regime_complete = False
                    raise RuntimeError(
                        f"Fold {fold}, {ticker}: training does not "
                        f"contain all three regimes: {sorted(observed)}"
                    )

            threshold_ordered = bool(
                (
                    thresholds[
                        "low_volatility_threshold_q33"
                    ]
                    <
                    thresholds[
                        "high_volatility_threshold_q67"
                    ]
                )
                .all()
            )

            if not threshold_ordered:
                raise RuntimeError(
                    f"Fold {fold}: threshold ordering failed."
                )

            integrity_rows.append(
                {
                    "scope": "fold",
                    "fold": fold,
                    "threshold_stock_count": int(
                        thresholds[
                            "ticker"
                        ].nunique()
                    ),
                    "threshold_rows": int(
                        len(thresholds)
                    ),
                    "thresholds_strictly_ordered": (
                        threshold_ordered
                    ),
                    "train_all_three_regimes_per_stock": (
                        training_regime_complete
                    ),
                    "training_threshold_fit_scope_valid": True,
                    "validation_thresholds_reused_unchanged": True,
                    "scaled_train_predictors_preserved_exactly": (
                        train_reconciliation[
                            "scaled_predictors_preserved_exactly"
                        ]
                    ),
                    "scaled_validation_predictors_preserved_exactly": (
                        validation_reconciliation[
                            "scaled_predictors_preserved_exactly"
                        ]
                    ),
                    "validation_used_for_threshold_fit": False,
                    "final_test_accessed": False,
                }
            )

            aggregate, stock = distribution_rows(
                train_labeled_raw,
                scope="fold",
                fold=fold,
                partition="train_reference",
            )

            regime_distribution_rows.extend(
                aggregate
            )

            stock_distribution_rows.extend(
                stock
            )

            aggregate, stock = distribution_rows(
                validation_labeled_raw,
                scope="fold",
                fold=fold,
                partition="validation_reference",
            )

            regime_distribution_rows.extend(
                aggregate
            )

            stock_distribution_rows.extend(
                stock
            )

            threshold_frames.append(
                thresholds
            )

            fold_dir = (
                staging
                /
                f"fold_{fold}"
            )

            threshold_path = (
                fold_dir
                /
                f"12_fold_{fold}_regime_thresholds.csv"
            )

            train_path = (
                fold_dir
                /
                f"12_fold_{fold}_train_scaled_regimes.parquet"
            )

            validation_path = (
                fold_dir
                /
                f"12_fold_{fold}_validation_scaled_regimes.parquet"
            )

            write_csv(
                threshold_path,
                thresholds,
            )

            write_parquet(
                train_output,
                train_path,
            )

            verify_parquet_roundtrip(
                train_output,
                train_path,
                feature_columns,
            )

            write_parquet(
                validation_output,
                validation_path,
            )

            verify_parquet_roundtrip(
                validation_output,
                validation_path,
                feature_columns,
            )

            for scope, path in (
                (
                    "fold_threshold_state",
                    threshold_path,
                ),
                (
                    "fold_train_scaled_regimes",
                    train_path,
                ),
                (
                    "fold_validation_scaled_regimes",
                    validation_path,
                ),
            ):
                staged_records.append(
                    {
                        "scope": scope,
                        "fold": fold,
                        "file_name": path.name,
                        "sha256": sha256_file(path),
                        "size_bytes": int(
                            path.stat().st_size
                        ),
                    }
                )

            train_counts = (
                train_labeled_raw[
                    "regime_id"
                ]
                .value_counts(
                    normalize=True
                )
                .reindex(
                    [
                        0,
                        1,
                        2,
                    ],
                    fill_value=0.0,
                )
            )

            validation_counts = (
                validation_labeled_raw[
                    "regime_id"
                ]
                .value_counts(
                    normalize=True
                )
                .reindex(
                    [
                        0,
                        1,
                        2,
                    ],
                    fill_value=0.0,
                )
            )

            print(
                f"       Fold {fold}: "
                f"thresholds=20 | "
                f"train rows={len(train_raw):>6,} | "
                f"val rows={len(validation_raw):>5,} | "
                f"train L/N/H="
                f"{train_counts[0]:.3f}/"
                f"{train_counts[1]:.3f}/"
                f"{train_counts[2]:.3f} | "
                f"val L/N/H="
                f"{validation_counts[0]:.3f}/"
                f"{validation_counts[1]:.3f}/"
                f"{validation_counts[2]:.3f}"
            )

        stage = "final_development_regimes"

        print(
            "[5/10] Estimating final-development stock thresholds without opening final test..."
        )

        final_raw = extract_final_development_raw(
            development=development,
            membership=membership,
        )

        final_thresholds = fit_threshold_state(
            fit_frame=final_raw,
            scope="final_development",
            fold=None,
        )

        final_labeled_raw = apply_threshold_state(
            final_raw,
            final_thresholds,
            fit_scope="final_development",
        )

        final_scaled_source = load_scaled_panel(
            STEP11_FINAL_DEVELOPMENT,
            feature_columns,
            label="final_development_scaled",
        )

        (
            final_output,
            final_reconciliation,
        ) = attach_regimes_to_scaled(
            scaled=final_scaled_source,
            raw_labeled=final_labeled_raw,
            feature_columns=feature_columns,
            fit_scope="final_development",
            label="final_development",
        )

        source_reconciliation_rows.append(
            final_reconciliation
        )

        final_training_regime_complete = True

        for ticker, stock in final_labeled_raw.groupby(
            "Ticker",
            sort=True,
        ):
            observed = set(
                stock[
                    "regime_id"
                ].unique()
                .tolist()
            )

            if observed != {
                0,
                1,
                2,
            }:
                final_training_regime_complete = False
                raise RuntimeError(
                    f"Final development, {ticker}: not all three "
                    f"regimes are represented."
                )

        final_threshold_ordered = bool(
            (
                final_thresholds[
                    "low_volatility_threshold_q33"
                ]
                <
                final_thresholds[
                    "high_volatility_threshold_q67"
                ]
            )
            .all()
        )

        if not final_threshold_ordered:
            raise RuntimeError(
                "Final-development threshold ordering failed."
            )

        integrity_rows.append(
            {
                "scope": "final_development",
                "fold": np.nan,
                "threshold_stock_count": int(
                    final_thresholds[
                        "ticker"
                    ].nunique()
                ),
                "threshold_rows": int(
                    len(final_thresholds)
                ),
                "thresholds_strictly_ordered": (
                    final_threshold_ordered
                ),
                "train_all_three_regimes_per_stock": (
                    final_training_regime_complete
                ),
                "training_threshold_fit_scope_valid": True,
                "validation_thresholds_reused_unchanged": (
                    np.nan
                ),
                "scaled_train_predictors_preserved_exactly": (
                    final_reconciliation[
                        "scaled_predictors_preserved_exactly"
                    ]
                ),
                "scaled_validation_predictors_preserved_exactly": (
                    np.nan
                ),
                "validation_used_for_threshold_fit": False,
                "final_test_accessed": False,
            }
        )

        aggregate, stock = distribution_rows(
            final_labeled_raw,
            scope="final_development",
            fold=None,
            partition="development_reference",
        )

        regime_distribution_rows.extend(
            aggregate
        )

        stock_distribution_rows.extend(
            stock
        )

        threshold_frames.append(
            final_thresholds
        )

        final_threshold_path = (
            staging
            /
            "final_development"
            /
            "12_final_development_regime_thresholds.csv"
        )

        final_panel_path = (
            staging
            /
            "final_development"
            /
            "12_final_development_scaled_regimes.parquet"
        )

        write_csv(
            final_threshold_path,
            final_thresholds,
        )

        write_parquet(
            final_output,
            final_panel_path,
        )

        verify_parquet_roundtrip(
            final_output,
            final_panel_path,
            feature_columns,
        )

        for scope, path in (
            (
                "final_development_threshold_state",
                final_threshold_path,
            ),
            (
                "final_development_scaled_regimes",
                final_panel_path,
            ),
        ):
            staged_records.append(
                {
                    "scope": scope,
                    "fold": None,
                    "file_name": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                }
            )

        final_counts = (
            final_labeled_raw[
                "regime_id"
            ]
            .value_counts(
                normalize=True
            )
            .reindex(
                [
                    0,
                    1,
                    2,
                ],
                fill_value=0.0,
            )
        )

        print(
            "       Final-development thresholds : 20"
        )
        print(
            "       Final-development rows       : "
            f"{len(final_raw):,}"
        )
        print(
            "       Final-development L/N/H      : "
            f"{final_counts[0]:.3f}/"
            f"{final_counts[1]:.3f}/"
            f"{final_counts[2]:.3f}"
        )
        print(
            "       Final-test parquet opened    : False"
        )

        stage = "combined_audits"

        print(
            "[6/10] Building combined threshold, distribution and integrity audits..."
        )

        thresholds_all = pd.concat(
            threshold_frames,
            ignore_index=True,
        )

        expected_threshold_rows = (
            (
                len(FOLDS)
                +
                1
            )
            *
            EXPECTED_STOCK_COUNT
        )

        if len(thresholds_all) != expected_threshold_rows:
            raise RuntimeError(
                "Combined threshold row count mismatch.\n"
                f"Expected: {expected_threshold_rows}\n"
                f"Observed: {len(thresholds_all)}"
            )

        if not (
            thresholds_all[
                "low_volatility_threshold_q33"
            ]
            <
            thresholds_all[
                "high_volatility_threshold_q67"
            ]
        ).all():
            raise RuntimeError(
                "One or more combined threshold pairs are not strictly ordered."
            )

        regime_distribution = pd.DataFrame(
            regime_distribution_rows
        )

        stock_distribution = pd.DataFrame(
            stock_distribution_rows
        )

        integrity_audit = pd.DataFrame(
            integrity_rows
        )

        reconciliation_audit = pd.DataFrame(
            source_reconciliation_rows
        )

        if not reconciliation_audit[
            "scaled_predictors_preserved_exactly"
        ].all():
            raise RuntimeError(
                "Scaled predictors were altered during regime attachment."
            )

        print(
            "       Combined threshold rows     : "
            f"{len(thresholds_all)}"
        )
        print(
            "       Expected                    : "
            f"{expected_threshold_rows}"
        )
        print(
            "       All q33 < q67              : "
            f"{bool((thresholds_all['low_volatility_threshold_q33'] < thresholds_all['high_volatility_threshold_q67']).all())}"
        )
        print(
            "       Scaled predictors unchanged : "
            f"{bool(reconciliation_audit['scaled_predictors_preserved_exactly'].all())}"
        )

        stage = "promotion"

        print(
            "[7/10] Promoting Step-12 panels and writing audits..."
        )

        if OUT_DIR.exists():
            raise RuntimeError(
                "Canonical Step-12 output directory appeared unexpectedly."
            )

        os.replace(
            staging,
            OUT_DIR,
        )

        staging = None

        promoted_records = []

        for record in staged_records:
            fold = record["fold"]

            if fold is None:
                canonical_path = (
                    FINAL_DEVELOPMENT_DIR
                    /
                    record["file_name"]
                )
            else:
                canonical_path = (
                    OUT_DIR
                    /
                    f"fold_{fold}"
                    /
                    record["file_name"]
                )

            if not canonical_path.exists():
                raise RuntimeError(
                    f"Promoted artifact missing: {canonical_path}"
                )

            if (
                sha256_file(canonical_path)
                !=
                record["sha256"]
            ):
                raise RuntimeError(
                    f"Promoted artifact hash mismatch: {canonical_path}"
                )

            promoted_records.append(
                {
                    "scope": record["scope"],
                    "fold": fold,
                    "file_path": str(
                        canonical_path
                    ),
                    "relative_path": (
                        relative_to_experiment(
                            canonical_path
                        )
                    ),
                    "size_bytes": int(
                        canonical_path.stat().st_size
                    ),
                    "sha256": record["sha256"],
                }
            )

        write_csv(
            THRESHOLD_AUDIT_CSV,
            thresholds_all,
        )

        write_csv(
            REGIME_DISTRIBUTION_CSV,
            regime_distribution,
        )

        write_csv(
            STOCK_REGIME_DISTRIBUTION_CSV,
            stock_distribution,
        )

        write_csv(
            INTEGRITY_AUDIT_CSV,
            integrity_audit,
        )

        write_csv(
            SOURCE_RECONCILIATION_CSV,
            reconciliation_audit,
        )

        write_csv(
            SOURCE_HASH_CSV,
            source_hashes(),
        )

        for path in (
            THRESHOLD_AUDIT_CSV,
            REGIME_DISTRIBUTION_CSV,
            STOCK_REGIME_DISTRIBUTION_CSV,
            INTEGRITY_AUDIT_CSV,
            SOURCE_RECONCILIATION_CSV,
            SOURCE_HASH_CSV,
        ):
            promoted_records.append(
                {
                    "scope": path.stem,
                    "fold": None,
                    "file_path": str(path),
                    "relative_path": (
                        relative_to_experiment(
                            path
                        )
                    ),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                    "sha256": sha256_file(path),
                }
            )

        stage = "freeze"

        print(
            "[8/10] Freezing Step-12 summary, source snapshot and manifest..."
        )

        if (
            sha256_file(
                Path(
                    step00["lock_path"]
                )
            )
            !=
            step00["lock_sha256"]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step 12."
            )

        summary = {
            "schema_version": SCHEMA_VERSION,
            "step": {
                "step_number": STEP_NUMBER,
                "step_id": STEP_ID,
                "step_name": STEP_NAME,
                "status": "PASS",
            },
            "generated_at_utc": utc_now(),
            "configuration_fingerprint_sha256": (
                step00["fingerprint"]
            ),
            "step11_manifest_sha256": (
                upstream["manifest_sha"][11]
            ),
            "feature_interface_sha256": (
                upstream["feature_interface_sha256"]
            ),
            "split_design_sha256": (
                upstream["split_design_sha256"]
            ),
            "regime_policy": {
                "source_column": REGIME_SOURCE_COLUMN,
                "source_is_unscaled": True,
                "window_sessions": (
                    REGIME_WINDOW_SESSIONS
                ),
                "low_quantile": LOW_QUANTILE,
                "high_quantile": HIGH_QUANTILE,
                "quantile_interpolation": (
                    QUANTILE_INTERPOLATION
                ),
                "mapping": {
                    "0": "Low",
                    "1": "Normal",
                    "2": "High",
                },
                "rule": (
                    "Low <= q33; Normal > q33 and <= q67; High > q67"
                ),
                "fold_threshold_fit_scope": (
                    "stock-specific fold-training reference rows only"
                ),
                "final_development_threshold_fit_scope": (
                    "stock-specific development reference rows only"
                ),
                "validation_used_for_threshold_fit": False,
                "final_test_used_for_threshold_fit": False,
            },
            "threshold_records": {
                "fold_stock_records": int(
                    len(FOLDS)
                    *
                    EXPECTED_STOCK_COUNT
                ),
                "final_development_stock_records": int(
                    EXPECTED_STOCK_COUNT
                ),
                "total": int(
                    len(thresholds_all)
                ),
                "all_strictly_ordered": True,
            },
            "reconciliation": {
                "records": int(
                    len(reconciliation_audit)
                ),
                "all_key_matches": bool(
                    reconciliation_audit[
                        "key_match_step10_to_step11"
                    ].all()
                ),
                "all_scaled_predictors_preserved_exactly": bool(
                    reconciliation_audit[
                        "scaled_predictors_preserved_exactly"
                    ].all()
                ),
            },
            "pipeline_state": {
                "fold_scaling_reused_from_step11": True,
                "fold_regime_labelling_complete": True,
                "final_development_regime_thresholds_created": True,
                "final_development_regime_labelling_complete": True,
                "final_test_regime_labelling_complete": False,
                "feature_selection_applied": False,
                "sequence_construction_performed": False,
                "models_fitted": False,
                "threshold_selection_performed": False,
            },
            "test_protection": {
                "step10_seal_status": (
                    upstream["test_seal_status"]
                ),
                "sealed_test_sha256_from_metadata_only": (
                    upstream[
                        "sealed_test_sha256_from_metadata_only"
                    ]
                ),
                "sealed_test_parquet_opened_or_hashed_by_step12": False,
                "global_final_test_state": (
                    step00["state"]
                ),
                "global_lock_unchanged": True,
            },
        }

        atomic_json(
            SUMMARY_JSON,
            summary,
        )

        shutil.copy2(
            SCRIPT_FILE,
            FROZEN_SCRIPT,
        )

        summary_lines = [
            STEP_NAME,
            "=" * 104,
            "Status: PASS",
            (
                "Configuration fingerprint: "
                f"{step00['fingerprint']}"
            ),
            (
                "Step-11 manifest SHA-256: "
                f"{upstream['manifest_sha'][11]}"
            ),
            (
                "63-feature interface SHA-256: "
                f"{upstream['feature_interface_sha256']}"
            ),
            (
                "Split-design SHA-256: "
                f"{upstream['split_design_sha256']}"
            ),
            (
                "Regime source: "
                f"{REGIME_SOURCE_COLUMN} (unscaled)"
            ),
            (
                "Fold-stock threshold records: "
                f"{len(FOLDS) * EXPECTED_STOCK_COUNT}"
            ),
            (
                "Final-development threshold records: "
                f"{EXPECTED_STOCK_COUNT}"
            ),
            (
                "Total threshold records: "
                f"{len(thresholds_all)}"
            ),
            (
                "Final-test parquet accessed: False"
            ),
            (
                "Global final-test state: "
                f"{step00['state']}"
            ),
        ]

        atomic_text(
            SUMMARY_TXT,
            "\n".join(summary_lines)
            +
            "\n",
        )

        for path in (
            SUMMARY_JSON,
            FROZEN_SCRIPT,
            SUMMARY_TXT,
        ):
            promoted_records.append(
                {
                    "scope": path.stem,
                    "fold": None,
                    "file_path": str(path),
                    "relative_path": (
                        relative_to_experiment(
                            path
                        )
                    ),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                    "sha256": sha256_file(path),
                }
            )

        hash_register = pd.DataFrame(
            promoted_records
        )

        write_csv(
            HASH_REGISTER_CSV,
            hash_register,
        )

        artifact_paths = [
            Path(record["file_path"])
            for record in promoted_records
        ] + [
            HASH_REGISTER_CSV,
        ]

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "step": {
                "step_number": STEP_NUMBER,
                "step_id": STEP_ID,
                "step_name": STEP_NAME,
                "status": "PASS",
            },
            "frozen_at_utc": utc_now(),
            "experiment_tag": cfg.EXPERIMENT_TAG,
            "configuration_fingerprint_sha256": (
                step00["fingerprint"]
            ),
            "step00_manifest_sha256": (
                step00["step00_manifest_sha256"]
            ),
            **{
                f"step{step:02d}_manifest_sha256": (
                    upstream[
                        "manifest_sha"
                    ][step]
                )
                for step in range(
                    1,
                    12,
                )
            },
            "feature_interface_sha256": (
                upstream["feature_interface_sha256"]
            ),
            "ordered_63_features": (
                feature_columns
            ),
            "split_design_sha256": (
                upstream["split_design_sha256"]
            ),
            "source_integrity": {
                "step10_development_panel_sha256": (
                    upstream["development_sha256"]
                ),
                "step10_membership_sha256": (
                    upstream["membership_sha256"]
                ),
                "step11_final_development_scaled_sha256": (
                    upstream[
                        "step11_final_development_scaled_sha256"
                    ]
                ),
            },
            "regime_policy": {
                "source_column": REGIME_SOURCE_COLUMN,
                "source_is_unscaled": True,
                "window_sessions": (
                    REGIME_WINDOW_SESSIONS
                ),
                "low_quantile": LOW_QUANTILE,
                "high_quantile": HIGH_QUANTILE,
                "quantile_interpolation": (
                    QUANTILE_INTERPOLATION
                ),
                "threshold_unit": "stock",
                "fold_threshold_fit_scope": (
                    "fold-training reference observations only"
                ),
                "final_development_threshold_fit_scope": (
                    "development reference observations only"
                ),
                "mapping": {
                    "0": "Low",
                    "1": "Normal",
                    "2": "High",
                },
                "rule": (
                    "Low <= q33; Normal > q33 and <= q67; High > q67"
                ),
                "validation_used_for_threshold_fit": False,
                "final_test_used_for_threshold_fit": False,
            },
            "threshold_state": {
                "fold_stock_threshold_records": int(
                    len(FOLDS)
                    *
                    EXPECTED_STOCK_COUNT
                ),
                "final_development_threshold_records": int(
                    EXPECTED_STOCK_COUNT
                ),
                "total_threshold_records": int(
                    len(thresholds_all)
                ),
                "all_q33_strictly_less_than_q67": True,
            },
            "integration_with_step11": {
                "scaled_predictors_reused": True,
                "scaled_predictors_preserved_exactly": bool(
                    reconciliation_audit[
                        "scaled_predictors_preserved_exactly"
                    ].all()
                ),
                "regime_source_taken_from_step10_unscaled_panel": True,
            },
            "test_protection": {
                "step10_seal_status": (
                    upstream["test_seal_status"]
                ),
                "global_final_test_state": (
                    step00["state"]
                ),
                "sealed_test_parquet_opened_or_hashed_by_step12": False,
                "test_regimes_computed": False,
                "test_performance_computed": False,
            },
            "pipeline_state": {
                "fold_scaling_complete": True,
                "fold_regime_labelling_complete": True,
                "final_development_regime_thresholds_created": True,
                "final_development_regime_labelling_complete": True,
                "feature_selection_applied": False,
                "sequence_construction_performed": False,
                "models_fitted": False,
            },
            "artifacts": {
                path.name: artifact_record(path)
                for path in artifact_paths
            },
        }

        atomic_json(
            MANIFEST_JSON,
            manifest,
        )

        manifest_sha = sha256_file(
            MANIFEST_JSON
        )

        atomic_text(
            MANIFEST_SHA,
            f"{manifest_sha}  {MANIFEST_JSON.name}\n",
        )

        stage = "validation"

        print(
            "[9/10] Validating frozen Step-12 artifact set..."
        )

        verify_sidecar(
            MANIFEST_JSON,
            MANIFEST_SHA,
            "Step 12",
        )

        for record in manifest[
            "artifacts"
        ].values():
            path = Path(
                record["path"]
            )

            if (
                not path.exists()
                or
                sha256_file(path)
                !=
                record["sha256"]
            ):
                raise RuntimeError(
                    f"Frozen Step-12 artifact mismatch: {path}"
                )

        if (
            sha256_file(
                Path(
                    step00["lock_path"]
                )
            )
            !=
            step00["lock_sha256"]
        ):
            raise RuntimeError(
                "Global final-test lock changed during final validation."
            )

        stage = "report"

        print(
            "[10/10] Final Step-12 report..."
        )

        print()
        print("-" * 134)
        print("REGIME POLICY")
        print("-" * 134)
        print(
            f"Source                       : {REGIME_SOURCE_COLUMN}"
        )
        print(
            "Source values                : UNSCALED"
        )
        print(
            f"Window                       : {REGIME_WINDOW_SESSIONS} sessions"
        )
        print(
            f"q33 / q67                    : {LOW_QUANTILE} / {HIGH_QUANTILE}"
        )
        print(
            f"Quantile interpolation       : {QUANTILE_INTERPOLATION}"
        )
        print(
            "Regime encoding              : 0=Low, 1=Normal, 2=High"
        )
        print(
            "Fold threshold fit           : stock-specific TRAIN reference only"
        )
        print(
            "Final-development threshold  : stock-specific DEVELOPMENT reference only"
        )

        print("-" * 134)
        print("FOLD-LEVEL REGIME DISTRIBUTION")
        print("-" * 134)

        for fold in FOLDS:
            for partition in (
                "train_reference",
                "validation_reference",
            ):
                subset = (
                    regime_distribution.loc[
                        (
                            regime_distribution[
                                "scope"
                            ]
                            ==
                            "fold"
                        )
                        &
                        (
                            regime_distribution[
                                "fold"
                            ]
                            ==
                            fold
                        )
                        &
                        (
                            regime_distribution[
                                "partition"
                            ]
                            ==
                            partition
                        )
                    ]
                    .sort_values(
                        "regime_id"
                    )
                )

                ratios = {
                    int(row.regime_id): float(
                        row.ratio
                    )
                    for row in subset.itertuples(
                        index=False
                    )
                }

                rows = int(
                    subset[
                        "rows"
                    ].sum()
                )

                print(
                    f"Fold {fold} {partition:<20} | "
                    f"rows={rows:>6,} | "
                    f"Low={ratios[0]:.4f} | "
                    f"Normal={ratios[1]:.4f} | "
                    f"High={ratios[2]:.4f}"
                )

        print("-" * 134)
        print("FINAL-DEVELOPMENT REGIME STATE")
        print("-" * 134)
        print(
            "Threshold records            : "
            f"{EXPECTED_STOCK_COUNT}"
        )
        print(
            "Development rows             : "
            f"{len(final_raw):,}"
        )
        print(
            "Regime ratios L/N/H          : "
            f"{final_counts[0]:.4f} / "
            f"{final_counts[1]:.4f} / "
            f"{final_counts[2]:.4f}"
        )
        print(
            "Scaled predictors preserved  : "
            f"{final_reconciliation['scaled_predictors_preserved_exactly']}"
        )
        print(
            "Final-test regime labels      : NOT CREATED"
        )

        print("-" * 134)
        print("THRESHOLD / INTEGRITY AUDIT")
        print("-" * 134)
        print(
            "Fold-stock threshold records : "
            f"{len(FOLDS) * EXPECTED_STOCK_COUNT}"
        )
        print(
            "Final-dev threshold records   : "
            f"{EXPECTED_STOCK_COUNT}"
        )
        print(
            "Total threshold records       : "
            f"{len(thresholds_all)}"
        )
        print(
            "All q33 < q67                : "
            f"{bool((thresholds_all['low_volatility_threshold_q33'] < thresholds_all['high_volatility_threshold_q67']).all())}"
        )
        print(
            "All scaled predictors exact   : "
            f"{bool(reconciliation_audit['scaled_predictors_preserved_exactly'].all())}"
        )

        print("-" * 134)
        print("PIPELINE STATE")
        print("-" * 134)
        print(
            "Fold scaling                 : REUSED FROM STEP 11"
        )
        print(
            "Fold regime labelling        : COMPLETE"
        )
        print(
            "Final-development regimes    : FROZEN"
        )
        print(
            "Feature selection applied    : False"
        )
        print(
            "Sequences constructed        : False"
        )
        print(
            "Models fitted                : False"
        )
        print(
            "Final-test parquet accessed  : False"
        )
        print(
            "Global final-test state      : "
            f"{step00['state']}"
        )

        print("-" * 134)
        print("OUTPUT FILES")
        print("-" * 134)
        print(
            f"Regime root          : {OUT_DIR}"
        )
        print(
            f"Threshold audit      : {THRESHOLD_AUDIT_CSV}"
        )
        print(
            f"Regime distribution : {REGIME_DISTRIBUTION_CSV}"
        )
        print(
            f"Stock distribution  : {STOCK_REGIME_DISTRIBUTION_CSV}"
        )
        print(
            f"Integrity audit      : {INTEGRITY_AUDIT_CSV}"
        )
        print(
            f"Source reconciliation: {SOURCE_RECONCILIATION_CSV}"
        )
        print(
            f"Manifest             : {MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256     : {manifest_sha}"
        )

        print("=" * 134)
        print(
            "PASS: 100 fold-stock q33/q67 threshold pairs were estimated "
            "from fold-training reference volatility only."
        )
        print(
            "PASS: 20 final-development stock threshold pairs were estimated "
            "from development reference volatility only."
        )
        print(
            "PASS: Validation regimes reuse unchanged fold-training thresholds."
        )
        print(
            "PASS: Regime labels were attached to Step-11 scaled panels without "
            "altering any of the 63 predictors."
        )
        print(
            "PASS: The sealed final-test parquet was not opened, hashed, "
            "labelled, transformed, or inspected."
        )
        print(
            "PASS: No feature selection, sequences, probability thresholds, "
            "or models were created."
        )
        print("=" * 134)
        print(
            "Do not run the old Step-12 script. "
            "Do not run Step 13 yet. Paste the complete terminal output for audit."
        )

        return 0

    except Exception as exc:
        if (
            staging is not None
            and
            staging.exists()
        ):
            shutil.rmtree(
                staging,
                ignore_errors=True,
            )

        archived = archive_partial()

        FAILED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        stamp = (
            datetime.now(timezone.utc)
            .strftime("%Y%m%dT%H%M%SZ")
        )

        failure_path = (
            FAILED_DIR
            /
            f"12_regimes_FAILED_{stamp}.json"
        )

        atomic_json(
            failure_path,
            {
                "step": STEP_NAME,
                "status": "FAIL",
                "failed_at_utc": utc_now(),
                "stage": stage,
                "error_type": (
                    type(exc).__name__
                ),
                "error_message": str(exc),
                "python_executable": (
                    sys.executable
                ),
                "archived_partial_artifacts": (
                    None
                    if archived is None
                    else str(archived)
                ),
                "canonical_manifest_exists": (
                    MANIFEST_JSON.exists()
                ),
            },
        )

        print()
        print("=" * 134)
        print("STEP 12 FAILED — VOLATILITY REGIMES NOT FROZEN")
        print("=" * 134)
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

        if archived is not None:
            print(
                f"Archived partial artifacts: {archived}"
            )

        print("=" * 134)
        print(
            "Do not proceed to Step 13."
        )

        return 30


# =============================================================================
# 17. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean-V3 train-only stock-specific volatility-regime stage."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an existing frozen Step-12 stage without recomputation."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verify_only:
        return verify_existing()

    return run()


if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "\nStep 12 interrupted by user."
        )
        sys.exit(
            130
        )

    except Exception as exc:
        print()
        print("=" * 134)
        print("STEP 12 UNHANDLED FAILURE")
        print("=" * 134)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        traceback.print_exc()
        print("=" * 134)
        sys.exit(
            1
        )
