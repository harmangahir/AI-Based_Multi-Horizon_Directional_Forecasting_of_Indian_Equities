#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 13 — CLEAN V3 CAUSAL 60-SESSION SEQUENCE INDEX CONSTRUCTION

Purpose
-------
Construct immutable, auditable sequence indices for:

1. five expanding-origin training partitions;
2. five expanding-origin validation partitions; and
3. the final-development training state.

Frozen causal sequence
----------------------
For reference endpoint t:

    X_t = [x_(t-59), ..., x_t]      shape = 60 x 63

Requirements:
- all 60 feature rows belong to the same ticker;
- all 60 feature rows belong to the same continuity segment;
- the window ends exactly at reference endpoint t;
- no feature row occurs after t;
- exact master-market session distance from first to last row is 59;
- target eligibility remains the frozen Step-10 partition eligibility.

Validation lookback policy
--------------------------
Validation endpoints MAY use causally available historical training rows.
The first 59 validation endpoints per stock therefore normally use some
training-period feature history.

The validation history source is the concatenation of that fold's Step-12
train + validation panels, both already transformed with the SAME fold-specific
training-fitted Step-11 scaler and labelled with the SAME fold-training-fitted
Step-12 regime thresholds.

Training history never uses validation rows.

Final-development state
-----------------------
A final-development sequence index is also frozen from the Step-12
final-development scaled+regime panel. It is development-only and does not
touch the sealed final-test parquet.

This stage DOES NOT:
- materialize 60x63 tensors;
- open/hash/inspect the sealed final-test parquet;
- perform feature selection;
- fit any model;
- tune any probability threshold;
- alter scaling or regime labels.

Run
---
python step13_sequence_indices_v3.py

Verify
------
python step13_sequence_indices_v3.py --verify-only
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

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent
if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError(
        "step13_sequence_indices_v3.py must be inside <repository>/q1_revision/."
    )
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg


# =============================================================================
# 1. FROZEN DESIGN
# =============================================================================

STEP_NUMBER = 13
STEP_ID = "13"
STEP_NAME = "STEP 13 — CAUSAL 60-SESSION SEQUENCE INDICES"
SCHEMA_VERSION = "1.0"

LOOKBACK = 60
HORIZONS = (1, 3, 5)
FOLDS = (1, 2, 3, 4, 5)

EXPECTED_FEATURE_COUNT = 63
EXPECTED_STOCK_COUNT = len(cfg.STOCK_TICKERS)

ALLOW_VALIDATION_LOOKBACK_INTO_TRAIN = True
REQUIRE_SAME_TICKER = True
REQUIRE_SAME_CONTINUITY_SEGMENT = True
REQUIRE_TARGET_INSIDE_ENDPOINT_PARTITION = True


# =============================================================================
# 2. UPSTREAM PATHS
# =============================================================================

STEP01_ENV = cfg.ENVIRONMENT_DIR / "01_environment_audit.json"

UPSTREAM = {
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
    12: (
        cfg.MANIFEST_DIR / "12_volatility_regime_manifest.json",
        cfg.MANIFEST_DIR / "12_volatility_regime_manifest.sha256",
    ),
}

STEP10_MEMBERSHIP = (
    cfg.FEATURE_DATA_DIR
    / "step10_chronological_splits"
    / "10_split_membership.parquet"
)

STEP10_TEST_SEAL = cfg.MANIFEST_DIR / "10_final_test_seal.json"
FORBIDDEN_TEST_FILENAME = "10_final_test_locked_panel.parquet"

STEP12_ROOT = cfg.FEATURE_DATA_DIR / "step12_volatility_regimes"

STEP12_FINAL_DEVELOPMENT = (
    STEP12_ROOT
    / "final_development"
    / "12_final_development_scaled_regimes.parquet"
)


# =============================================================================
# 3. OUTPUTS
# =============================================================================

OUT_DIR = cfg.FEATURE_DATA_DIR / "step13_sequence_indices"
FINAL_DEVELOPMENT_DIR = OUT_DIR / "final_development"

COUNT_SUMMARY_CSV = cfg.AUDIT_DIR / "13_sequence_count_summary.csv"
CROSS_BOUNDARY_AUDIT_CSV = cfg.AUDIT_DIR / "13_cross_boundary_lookback_audit.csv"
INTEGRITY_AUDIT_CSV = cfg.AUDIT_DIR / "13_sequence_integrity_audit.csv"
MEMBERSHIP_RECONCILIATION_CSV = (
    cfg.AUDIT_DIR / "13_membership_reconciliation_audit.csv"
)
SOURCE_HASH_CSV = cfg.AUDIT_DIR / "13_source_file_hashes.csv"
HASH_REGISTER_CSV = cfg.AUDIT_DIR / "13_sequence_index_hashes.csv"
SUMMARY_JSON = cfg.AUDIT_DIR / "13_sequence_index_summary.json"

MANIFEST_JSON = cfg.MANIFEST_DIR / "13_sequence_index_manifest.json"
MANIFEST_SHA = cfg.MANIFEST_DIR / "13_sequence_index_manifest.sha256"
FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step13_sequence_indices_frozen.py"
SUMMARY_TXT = cfg.REPORT_DIR / "13_sequence_index_summary.txt"
FAILED_DIR = cfg.AUDIT_DIR / "step13_failed_attempts"


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
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=json_default,
        )
        handle.write("\n")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(
        tmp,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
    )
    os.replace(tmp, path)


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
# 5. STEP-00 / ENVIRONMENT / CHAIN VERIFICATION
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
        raise RuntimeError("Step-00 freeze is not VERIFIED_EXISTING.")

    configured_lookback = getattr(cfg, "LOOKBACK", LOOKBACK)
    if int(configured_lookback) != LOOKBACK:
        raise RuntimeError(
            f"Configured lookback is {configured_lookback}, expected {LOOKBACK}."
        )

    # If these policy flags exist in the frozen config, they must agree.
    if hasattr(cfg, "USE_CAUSAL_CROSS_BOUNDARY_LOOKBACK"):
        if not bool(cfg.USE_CAUSAL_CROSS_BOUNDARY_LOOKBACK):
            raise RuntimeError(
                "Frozen configuration disables causal cross-boundary lookback."
            )

    if hasattr(cfg, "REQUIRE_LOOKBACK_WITHIN_SAME_SPLIT"):
        if bool(cfg.REQUIRE_LOOKBACK_WITHIN_SAME_SPLIT):
            raise RuntimeError(
                "Frozen configuration still requires lookback inside one split."
            )

    lock_path = cfg.get_final_test_lock_path()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    if lock.get("state") != "LOCKED_NOT_OPENED":
        raise RuntimeError("Global final-test state is not LOCKED_NOT_OPENED.")
    if lock.get("final_test_opened") is not False:
        raise RuntimeError("global final_test_opened != False.")
    if lock.get("final_test_scored") is not False:
        raise RuntimeError("global final_test_scored != False.")

    return {
        "fingerprint": fingerprint,
        "step00_manifest_path": str(manifest_path),
        "step00_manifest_sha256": manifest_sha,
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "state": lock["state"],
    }


def verify_sidecar(
    manifest_path: Path,
    sidecar_path: Path,
    label: str,
) -> str:
    if not manifest_path.exists() or not sidecar_path.exists():
        raise RuntimeError(f"{label} manifest/sidecar missing.")

    expected = (
        sidecar_path.read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )
    observed = sha256_file(manifest_path)

    if observed != expected:
        raise RuntimeError(
            f"{label} manifest SHA mismatch.\n"
            f"Expected: {expected}\nObserved: {observed}"
        )

    return observed


def verify_environment() -> None:
    if not STEP01_ENV.exists():
        raise RuntimeError(
            f"Step-01 environment audit missing: {STEP01_ENV}"
        )

    audit = json.loads(
        STEP01_ENV.read_text(encoding="utf-8")
    )

    expected_python = (
        audit.get("system", {}).get("python_executable")
    )

    if expected_python:
        if (
            Path(expected_python).resolve()
            !=
            Path(sys.executable).resolve()
        ):
            raise RuntimeError(
                "Current Python differs from frozen Step-01 environment.\n"
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
                (distribution, stored_version, current_version)
            )

    if drift:
        raise RuntimeError(
            "Required-package drift detected:\n"
            + "\n".join(
                f"{name}: frozen={old!r}, current={new!r}"
                for name, old, new in drift
            )
        )


def find_artifact(
    manifest: Dict[str, Any],
    filename: str,
) -> Dict[str, Any]:
    matches = []

    for key in ("artifacts", "canonical_artifacts"):
        for record in manifest.get(key, {}).values():
            if not isinstance(record, dict):
                continue
            path_text = record.get("path")
            if path_text and Path(path_text).name == filename:
                matches.append(record)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one manifest artifact named {filename}; "
            f"found {len(matches)}."
        )

    return matches[0]


def verify_step12_artifacts(
    step12_manifest: Dict[str, Any],
) -> None:
    failures = []

    for name, record in step12_manifest.get(
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
                "Step-12 artifact set."
            )

        if not path.exists():
            failures.append(f"missing: {name} -> {path}")
            continue

        expected = record.get("sha256")
        if expected and sha256_file(path) != expected:
            failures.append(f"hash mismatch: {name}")

    if failures:
        raise RuntimeError(
            "Step-12 artifact verification failed:\n"
            + "\n".join(failures)
        )


def step12_fold_path(
    fold: int,
    partition: str,
) -> Path:
    return (
        STEP12_ROOT
        / f"fold_{fold}"
        / f"12_fold_{fold}_{partition}_scaled_regimes.parquet"
    )


def verify_upstream(
    step00: Dict[str, Any],
) -> Dict[str, Any]:
    shas: Dict[int, str] = {}
    manifests: Dict[int, Dict[str, Any]] = {}

    for step in range(1, 13):
        manifest_path, sidecar_path = UPSTREAM[step]

        shas[step] = verify_sidecar(
            manifest_path,
            sidecar_path,
            f"Step {step:02d}",
        )

        manifests[step] = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

    if manifests[1].get("status") != "PASS":
        raise RuntimeError("Step-01 status is not PASS.")

    if (
        manifests[1].get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00["fingerprint"]
    ):
        raise RuntimeError("Step-01 fingerprint mismatch.")

    if (
        manifests[2]
        .get("step", {})
        .get("status")
        !=
        "PASS"
    ):
        raise RuntimeError("Step-02 status is not PASS.")

    if (
        manifests[2]
        .get("step00", {})
        .get("configuration_fingerprint_sha256")
        !=
        step00["fingerprint"]
    ):
        raise RuntimeError("Step-02 fingerprint mismatch.")

    if (
        manifests[2]
        .get("step01", {})
        .get("step01_manifest_sha256")
        !=
        shas[1]
    ):
        raise RuntimeError("Step-02 -> Step-01 chain mismatch.")

    for step in range(3, 13):
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
            field = f"step{previous:02d}_manifest_sha256"

            if (
                field in manifest
                and
                manifest[field] != shas[previous]
            ):
                raise RuntimeError(
                    f"Step-{step:02d} chain mismatch: {field}"
                )

    verify_environment()

    step12 = manifests[12]

    feature_columns = step12.get("ordered_63_features")
    feature_interface_sha = step12.get(
        "feature_interface_sha256"
    )
    split_design_sha = step12.get(
        "split_design_sha256"
    )

    if (
        not isinstance(feature_columns, list)
        or
        len(feature_columns) != EXPECTED_FEATURE_COUNT
    ):
        raise RuntimeError(
            "Step-12 manifest does not freeze exactly 63 predictors."
        )

    if not feature_interface_sha:
        raise RuntimeError(
            "Step-12 manifest lacks feature_interface_sha256."
        )

    if not split_design_sha:
        raise RuntimeError(
            "Step-12 manifest lacks split_design_sha256."
        )

    for step in (8, 9, 10, 11):
        if (
            manifests[step].get("ordered_63_features")
            !=
            feature_columns
        ):
            raise RuntimeError(
                f"Step-{step:02d}/12 predictor order mismatch."
            )

        if (
            manifests[step].get(
                "feature_interface_sha256"
            )
            !=
            feature_interface_sha
        ):
            raise RuntimeError(
                f"Step-{step:02d}/12 feature-interface SHA mismatch."
            )

    if (
        manifests[10].get("split_design_sha256")
        !=
        split_design_sha
        or
        manifests[11].get("split_design_sha256")
        !=
        split_design_sha
    ):
        raise RuntimeError(
            "Split-design SHA drift across Steps 10-12."
        )

    # Verify Step-12's own immutable artifact set.
    verify_step12_artifacts(step12)

    # Verify canonical fold panels and final-development panel.
    fold_panel_hashes: Dict[str, str] = {}

    for fold in FOLDS:
        for partition in ("train", "validation"):
            path = step12_fold_path(
                fold,
                partition,
            )

            record = find_artifact(
                step12,
                path.name,
            )

            if Path(record["path"]).resolve() != path.resolve():
                raise RuntimeError(
                    f"Step-12 canonical path mismatch: {path.name}"
                )

            observed = sha256_file(path)

            if observed != record["sha256"]:
                raise RuntimeError(
                    f"Step-12 panel hash mismatch: {path.name}"
                )

            fold_panel_hashes[
                f"fold_{fold}_{partition}"
            ] = observed

    final_record = find_artifact(
        step12,
        STEP12_FINAL_DEVELOPMENT.name,
    )

    if (
        Path(final_record["path"]).resolve()
        !=
        STEP12_FINAL_DEVELOPMENT.resolve()
    ):
        raise RuntimeError(
            "Step-12 final-development path is not canonical."
        )

    final_development_sha = sha256_file(
        STEP12_FINAL_DEVELOPMENT
    )

    if final_development_sha != final_record["sha256"]:
        raise RuntimeError(
            "Step-12 final-development panel hash mismatch."
        )

    # Step-10 membership remains the authoritative partition ledger.
    step10 = manifests[10]

    membership_record = find_artifact(
        step10,
        STEP10_MEMBERSHIP.name,
    )

    if (
        Path(membership_record["path"]).resolve()
        !=
        STEP10_MEMBERSHIP.resolve()
    ):
        raise RuntimeError(
            "Step-10 membership path is not canonical."
        )

    membership_sha = sha256_file(
        STEP10_MEMBERSHIP
    )

    if membership_sha != membership_record["sha256"]:
        raise RuntimeError(
            "Step-10 membership SHA mismatch."
        )

    # Step-10 seal metadata only. Never touch final-test parquet.
    if not STEP10_TEST_SEAL.exists():
        raise RuntimeError(
            "Step-10 final-test seal missing."
        )

    seal = json.loads(
        STEP10_TEST_SEAL.read_text(
            encoding="utf-8"
        )
    )

    if seal.get("status") != "SEALED_NOT_OPENED":
        raise RuntimeError(
            "Step-10 test seal is not SEALED_NOT_OPENED."
        )

    if (
        seal.get("global_final_test_lock_state")
        !=
        "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError(
            "Step-10 seal does not preserve LOCKED_NOT_OPENED."
        )

    split_design = manifests[10].get(
        "split_design",
        {},
    )

    folds = split_design.get(
        "folds",
        [],
    )

    if len(folds) != 5:
        raise RuntimeError(
            "Step-10 split design does not contain five folds."
        )

    return {
        "manifest_sha": shas,
        "manifests": manifests,
        "feature_columns": list(feature_columns),
        "feature_interface_sha256": feature_interface_sha,
        "split_design_sha256": split_design_sha,
        "split_design": split_design,
        "membership_sha256": membership_sha,
        "fold_panel_hashes": fold_panel_hashes,
        "final_development_panel_sha256": (
            final_development_sha
        ),
        "test_seal_status": seal["status"],
        "sealed_test_sha256_from_metadata_only": (
            seal.get("sha256")
        ),
    }


# =============================================================================
# 6. IMMUTABILITY / STAGING
# =============================================================================

def canonical_outputs() -> tuple[Path, ...]:
    return (
        OUT_DIR,
        COUNT_SUMMARY_CSV,
        CROSS_BOUNDARY_AUDIT_CSV,
        INTEGRITY_AUDIT_CSV,
        MEMBERSHIP_RECONCILIATION_CSV,
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
            "Step-13 canonical output(s) already exist. "
            "Refusing to overwrite.\n"
            + "\n".join(
                str(path)
                for path in collisions
            )
            + "\nUse --verify-only if Step 13 already passed."
        )


def create_staging_root() -> Path:
    stamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )

    root = (
        cfg.FEATURE_DATA_DIR
        /
        f"_step13_staging_{stamp}_{os.getpid()}"
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
# 7. LOAD MEMBERSHIP / STEP-12 PANELS
# =============================================================================

def load_membership() -> pd.DataFrame:
    frame = pd.read_parquet(
        STEP10_MEMBERSHIP,
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
            "Duplicate ticker-date keys in Step-10 membership."
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


def load_step12_panel(
    path: Path,
    feature_columns: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    if path.name == FORBIDDEN_TEST_FILENAME:
        raise RuntimeError(
            "Programming error: sealed final-test parquet "
            "requested by Step 13."
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

    for horizon in HORIZONS:
        target_date_column = (
            f"target_date_{horizon}d"
        )

        if target_date_column in frame.columns:
            frame[target_date_column] = (
                pd.to_datetime(
                    frame[
                        target_date_column
                    ],
                    errors="coerce",
                )
                .dt.normalize()
            )

    required = {
        "Ticker",
        "Date",
        "market_session_id",
        "continuity_segment_id",
        "regime_id",
        "regime_name",
        "regime_threshold_fit_scope",
        "eligible_1d",
        "eligible_3d",
        "eligible_5d",
        "eligible_common_multitask",
        "target_date_1d",
        "target_date_3d",
        "target_date_5d",
        *feature_columns,
    }

    missing = (
        required
        -
        set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"{label}: missing required columns:\n"
            f"{sorted(missing)}"
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

    if not frame[
        "regime_id"
    ].isin(
        [
            0,
            1,
            2,
        ]
    ).all():
        raise RuntimeError(
            f"{label}: invalid regime ID."
        )

    predictor_values = (
        frame[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        predictor_values
    ).all():
        raise RuntimeError(
            f"{label}: non-finite predictor values."
        )

    if set(
        frame["Ticker"].unique()
    ) != set(
        cfg.STOCK_TICKERS
    ):
        raise RuntimeError(
            f"{label}: stock universe mismatch."
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
# 8. MEMBERSHIP RECONCILIATION
# =============================================================================

def reconcile_partition_with_membership(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    reference_column: str,
    eligibility_prefix: str,
    label: str,
) -> Dict[str, Any]:
    needed = [
        "Ticker",
        "Date",
        reference_column,
        f"{eligibility_prefix}_eligible_1d",
        f"{eligibility_prefix}_eligible_3d",
        f"{eligibility_prefix}_eligible_5d",
        f"{eligibility_prefix}_common_multitask",
    ]

    missing = [
        column
        for column in needed
        if column not in membership.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label}: missing membership columns:\n{missing}"
        )

    expected = (
        membership.loc[
            membership[
                reference_column
            ],
            needed,
        ]
        .copy()
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

    observed_keys = (
        panel[
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

    expected_keys = (
        expected[
            [
                "Ticker",
                "Date",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    key_match = bool(
        observed_keys.equals(
            expected_keys
        )
    )

    if not key_match:
        raise RuntimeError(
            f"{label}: Step-12 panel keys differ from Step-10 membership."
        )

    panel_ordered = (
        panel
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

    eligibility_match = True

    mapping = {
        "eligible_1d": (
            f"{eligibility_prefix}_eligible_1d"
        ),
        "eligible_3d": (
            f"{eligibility_prefix}_eligible_3d"
        ),
        "eligible_5d": (
            f"{eligibility_prefix}_eligible_5d"
        ),
        "eligible_common_multitask": (
            f"{eligibility_prefix}_common_multitask"
        ),
    }

    for observed_column, expected_column in mapping.items():
        observed = (
            panel_ordered[
                observed_column
            ]
            .astype(bool)
            .to_numpy()
        )

        expected_values = (
            expected[
                expected_column
            ]
            .astype(bool)
            .to_numpy()
        )

        if not np.array_equal(
            observed,
            expected_values,
        ):
            eligibility_match = False
            raise RuntimeError(
                f"{label}: eligibility mismatch for {observed_column}."
            )

    return {
        "scope": label,
        "rows": int(
            len(panel_ordered)
        ),
        "dates": int(
            panel_ordered[
                "Date"
            ]
            .nunique()
        ),
        "stocks": int(
            panel_ordered[
                "Ticker"
            ]
            .nunique()
        ),
        "keys_exact_step10_to_step12": (
            key_match
        ),
        "eligibility_flags_exact_step10_to_step12": (
            eligibility_match
        ),
    }


# =============================================================================
# 9. HISTORY LOOKUP
# =============================================================================

def build_history_lookup(
    history_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    label: str,
) -> Dict[
    tuple[str, int],
    Dict[str, Any],
]:
    if history_frame[
        [
            "Ticker",
            "Date",
        ]
    ].duplicated().any():
        raise RuntimeError(
            f"{label}: duplicate history keys."
        )

    if not np.isfinite(
        history_frame[
            feature_columns
        ].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(
            f"{label}: non-finite history predictors."
        )

    lookup: Dict[
        tuple[str, int],
        Dict[str, Any],
    ] = {}

    for (
        ticker,
        segment_id,
    ), group in history_frame.groupby(
        [
            "Ticker",
            "continuity_segment_id",
        ],
        sort=True,
    ):
        table = (
            group
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        if table[
            "Date"
        ].duplicated().any():
            raise RuntimeError(
                f"{label}: duplicate dates for "
                f"{ticker}, segment={segment_id}."
            )

        positions = {
            pd.Timestamp(date): int(position)
            for position, date in enumerate(
                table[
                    "Date"
                ]
            )
        }

        lookup[
            (
                ticker,
                int(segment_id),
            )
        ] = {
            "table": table,
            "positions": positions,
        }

    return lookup


# =============================================================================
# 10. BUILD SEQUENCE INDEX
# =============================================================================

def build_sequence_index(
    endpoint_panel: pd.DataFrame,
    history_lookup: Dict[
        tuple[str, int],
        Dict[str, Any],
    ],
    *,
    scope: str,
    fold: Optional[int],
    partition: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    stock_id_map: Dict[str, int],
) -> pd.DataFrame:
    records = []

    for row in endpoint_panel.itertuples(
        index=False
    ):
        ticker = row.Ticker
        segment_id = int(
            row.continuity_segment_id
        )
        reference_date = pd.Timestamp(
            row.Date
        )
        reference_session_id = int(
            row.market_session_id
        )

        key = (
            ticker,
            segment_id,
        )

        if key not in history_lookup:
            raise RuntimeError(
                f"{scope}: missing history lookup for {key}."
            )

        history = history_lookup[key]
        table = history["table"]
        positions = history["positions"]

        if reference_date not in positions:
            raise RuntimeError(
                f"{scope}: endpoint not found in history: "
                f"{ticker} {reference_date.date()}."
            )

        end_position = int(
            positions[
                reference_date
            ]
        )

        history_available = bool(
            end_position
            >=
            (
                LOOKBACK
                -
                1
            )
        )

        sequence_start_date = pd.NaT
        sequence_end_date = reference_date
        sequence_start_market_session_id = pd.NA
        sequence_end_market_session_id = reference_session_id
        sequence_length = 0
        start_position = pd.NA
        historical_rows_before_partition = 0
        lookback_crosses_partition_start = False

        if history_available:
            start_position_int = (
                end_position
                -
                LOOKBACK
                +
                1
            )

            window = (
                table.iloc[
                    start_position_int:
                    end_position
                    +
                    1
                ]
                .copy()
            )

            if len(window) != LOOKBACK:
                raise RuntimeError(
                    f"{scope}: sequence length is not {LOOKBACK}."
                )

            if (
                REQUIRE_SAME_TICKER
                and
                window["Ticker"].nunique()
                !=
                1
            ):
                raise RuntimeError(
                    f"{scope}: sequence crosses ticker boundary."
                )

            if (
                REQUIRE_SAME_CONTINUITY_SEGMENT
                and
                window[
                    "continuity_segment_id"
                ].nunique()
                !=
                1
            ):
                raise RuntimeError(
                    f"{scope}: sequence crosses continuity boundary."
                )

            if (
                int(
                    window[
                        "continuity_segment_id"
                    ].iloc[
                        0
                    ]
                )
                !=
                segment_id
            ):
                raise RuntimeError(
                    f"{scope}: sequence continuity ID "
                    "differs from endpoint."
                )

            if (
                pd.Timestamp(
                    window[
                        "Date"
                    ].iloc[
                        -1
                    ]
                )
                !=
                reference_date
            ):
                raise RuntimeError(
                    f"{scope}: sequence does not end at endpoint."
                )

            if (
                window[
                    "Date"
                ]
                >
                reference_date
            ).any():
                raise RuntimeError(
                    f"{scope}: future feature row detected."
                )

            sequence_start_date = pd.Timestamp(
                window[
                    "Date"
                ].iloc[
                    0
                ]
            )

            sequence_start_market_session_id = int(
                window[
                    "market_session_id"
                ].iloc[
                    0
                ]
            )

            sequence_length = int(
                len(window)
            )

            start_position = int(
                start_position_int
            )

            session_difference = (
                reference_session_id
                -
                sequence_start_market_session_id
            )

            if session_difference != LOOKBACK - 1:
                raise RuntimeError(
                    f"{scope}: non-consecutive master-session window "
                    f"for {ticker} {reference_date.date()}; "
                    f"difference={session_difference}."
                )

            historical_rows_before_partition = int(
                (
                    window[
                        "Date"
                    ]
                    <
                    period_start
                ).sum()
            )

            lookback_crosses_partition_start = bool(
                historical_rows_before_partition
                >
                0
            )

            if partition in (
                "train",
                "final_development",
            ):
                if lookback_crosses_partition_start:
                    raise RuntimeError(
                        f"{scope}: training/development sequence "
                        "unexpectedly uses pre-partition history."
                    )

            if partition == "validation":
                if not ALLOW_VALIDATION_LOOKBACK_INTO_TRAIN:
                    if lookback_crosses_partition_start:
                        raise RuntimeError(
                            f"{scope}: validation cross-boundary "
                            "lookback is disabled."
                        )

                if (
                    historical_rows_before_partition
                    >
                    LOOKBACK
                    -
                    1
                ):
                    raise RuntimeError(
                        f"{scope}: too many pre-validation history rows."
                    )

        record = {
            "scope": scope,
            "fold": (
                pd.NA
                if fold is None
                else int(fold)
            ),
            "partition": partition,
            "ticker": ticker,
            "stock_id": int(
                stock_id_map[
                    ticker
                ]
            ),
            "continuity_segment_id": (
                segment_id
            ),
            "reference_date": (
                reference_date
            ),
            "reference_market_session_id": (
                reference_session_id
            ),
            "sequence_start_date": (
                sequence_start_date
            ),
            "sequence_end_date": (
                sequence_end_date
            ),
            "sequence_start_market_session_id": (
                sequence_start_market_session_id
            ),
            "sequence_end_market_session_id": (
                sequence_end_market_session_id
            ),
            "history_start_position_in_segment": (
                start_position
            ),
            "history_end_position_in_segment": (
                end_position
            ),
            "sequence_length": (
                sequence_length
            ),
            "sequence_history_available": (
                history_available
            ),
            "lookback_crosses_partition_start": (
                lookback_crosses_partition_start
            ),
            "historical_rows_before_partition": (
                historical_rows_before_partition
            ),
            "regime_id": int(
                row.regime_id
            ),
            "regime_name": row.regime_name,
            "regime_threshold_fit_scope": (
                row.regime_threshold_fit_scope
            ),
            "target_date_1d": (
                row.target_date_1d
            ),
            "target_date_3d": (
                row.target_date_3d
            ),
            "target_date_5d": (
                row.target_date_5d
            ),
            "target_eligible_1d": bool(
                row.eligible_1d
            ),
            "target_eligible_3d": bool(
                row.eligible_3d
            ),
            "target_eligible_5d": bool(
                row.eligible_5d
            ),
            "target_eligible_common_multitask": bool(
                row.eligible_common_multitask
            ),
        }

        for horizon in HORIZONS:
            record[
                f"sequence_eligible_{horizon}d"
            ] = bool(
                history_available
                and
                record[
                    f"target_eligible_{horizon}d"
                ]
            )

        record[
            "sequence_eligible_common_multitask"
        ] = bool(
            history_available
            and
            record[
                "target_eligible_common_multitask"
            ]
        )

        records.append(record)

    index = pd.DataFrame(records)

    if index.empty:
        raise RuntimeError(
            f"{scope}: sequence index is empty."
        )

    # Exact endpoint balance.
    if (
        index[
            "reference_date"
        ].nunique()
        *
        EXPECTED_STOCK_COUNT
        !=
        len(index)
    ):
        raise RuntimeError(
            f"{scope}: endpoint panel is not balanced."
        )

    # Target confinement remains the Step-10 frozen partition rule.
    for horizon in HORIZONS:
        eligible_column = (
            f"sequence_eligible_{horizon}d"
        )
        target_date_column = (
            f"target_date_{horizon}d"
        )

        eligible = index[
            eligible_column
        ]

        eligible_targets = pd.to_datetime(
            index.loc[
                eligible,
                target_date_column,
            ],
            errors="raise",
        )

        if (
            REQUIRE_TARGET_INSIDE_ENDPOINT_PARTITION
            and
            not eligible_targets.between(
                period_start,
                period_end,
                inclusive="both",
            ).all()
        ):
            raise RuntimeError(
                f"{scope}: {horizon}d target escaped "
                "the endpoint partition."
            )

        eligible_reference = pd.to_datetime(
            index.loc[
                eligible,
                "reference_date",
            ],
            errors="raise",
        )

        if (
            eligible_targets
            <=
            eligible_reference
        ).any():
            raise RuntimeError(
                f"{scope}: {horizon}d contains non-future target."
            )

    if not index[
        "sequence_eligible_common_multitask"
    ].equals(
        index[
            "sequence_eligible_5d"
        ]
    ):
        mismatch = int(
            (
                index[
                    "sequence_eligible_common_multitask"
                ]
                !=
                index[
                    "sequence_eligible_5d"
                ]
            ).sum()
        )

        raise RuntimeError(
            f"{scope}: common and 5d sequence eligibility "
            f"differ on {mismatch} rows."
        )

    return (
        index
        .sort_values(
            [
                "reference_date",
                "ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# 11. COUNT / INTEGRITY / CROSS-BOUNDARY AUDITS
# =============================================================================

def count_summary(
    index: pd.DataFrame,
    *,
    scope: str,
    fold: Optional[int],
    partition: str,
) -> Dict[str, Any]:
    reference_rows = int(
        len(index)
    )
    reference_dates = int(
        index[
            "reference_date"
        ].nunique()
    )
    history_rows = int(
        index[
            "sequence_history_available"
        ].sum()
    )

    sequence_counts = {
        horizon: int(
            index[
                f"sequence_eligible_{horizon}d"
            ].sum()
        )
        for horizon in HORIZONS
    }

    common = int(
        index[
            "sequence_eligible_common_multitask"
        ].sum()
    )

    # Definition-level accounting: no eligible sequence may lack history or
    # a frozen target-eligibility flag.
    for horizon in HORIZONS:
        expected = int(
            (
                index[
                    "sequence_history_available"
                ]
                &
                index[
                    f"target_eligible_{horizon}d"
                ]
            ).sum()
        )

        if sequence_counts[horizon] != expected:
            raise RuntimeError(
                f"{scope}: {horizon}d sequence-count accounting failed."
            )

    expected_common = int(
        (
            index[
                "sequence_history_available"
            ]
            &
            index[
                "target_eligible_common_multitask"
            ]
        ).sum()
    )

    if common != expected_common:
        raise RuntimeError(
            f"{scope}: common sequence-count accounting failed."
        )

    cross = int(
        (
            index[
                "sequence_history_available"
            ]
            &
            index[
                "lookback_crosses_partition_start"
            ]
        ).sum()
    )

    if partition in (
        "train",
        "final_development",
    ):
        expected_cross = 0

    elif partition == "validation":
        # First 59 validation endpoint dates per stock use some preceding
        # training history, assuming validation itself contains >=59 dates.
        expected_cross = int(
            min(
                LOOKBACK - 1,
                reference_dates,
            )
            *
            EXPECTED_STOCK_COUNT
        )

        if cross != expected_cross:
            raise RuntimeError(
                f"{scope}: validation cross-boundary count mismatch. "
                f"Expected {expected_cross:,}, observed {cross:,}."
            )

    else:
        raise RuntimeError(
            f"{scope}: unknown partition {partition}."
        )

    if cross != expected_cross:
        raise RuntimeError(
            f"{scope}: cross-boundary count mismatch."
        )

    return {
        "scope": scope,
        "fold": (
            np.nan
            if fold is None
            else int(fold)
        ),
        "partition": partition,
        "reference_dates": reference_dates,
        "reference_rows": reference_rows,
        "history_available_rows": history_rows,
        "history_unavailable_rows": int(
            reference_rows
            -
            history_rows
        ),
        "sequence_eligible_1d": (
            sequence_counts[1]
        ),
        "sequence_eligible_3d": (
            sequence_counts[3]
        ),
        "sequence_eligible_5d": (
            sequence_counts[5]
        ),
        "sequence_eligible_common_multitask": (
            common
        ),
        "cross_partition_lookback_endpoints": (
            cross
        ),
        "expected_cross_partition_lookback_endpoints": (
            expected_cross
        ),
    }


def integrity_audit(
    index: pd.DataFrame,
    *,
    scope: str,
    fold: Optional[int],
    partition: str,
) -> Dict[str, Any]:
    available = index[
        "sequence_history_available"
    ]

    rows = index.loc[
        available
    ]

    all_length_60 = bool(
        rows[
            "sequence_length"
        ].eq(
            LOOKBACK
        ).all()
    )

    all_end_at_reference = bool(
        (
            pd.to_datetime(
                rows[
                    "sequence_end_date"
                ]
            )
            ==
            pd.to_datetime(
                rows[
                    "reference_date"
                ]
            )
        ).all()
    )

    all_session_span_59 = bool(
        (
            pd.to_numeric(
                rows[
                    "sequence_end_market_session_id"
                ]
            )
            -
            pd.to_numeric(
                rows[
                    "sequence_start_market_session_id"
                ]
            )
            ==
            LOOKBACK
            -
            1
        ).all()
    )

    all_regimes_valid = bool(
        index[
            "regime_id"
        ].isin(
            [
                0,
                1,
                2,
            ]
        ).all()
    )

    unavailable = index.loc[
        ~available
    ]

    unavailable_lengths_zero = bool(
        unavailable[
            "sequence_length"
        ].eq(
            0
        ).all()
    )

    passed = bool(
        all(
            [
                all_length_60,
                all_end_at_reference,
                all_session_span_59,
                all_regimes_valid,
                unavailable_lengths_zero,
            ]
        )
    )

    if not passed:
        raise RuntimeError(
            f"{scope}: sequence integrity audit failed."
        )

    return {
        "scope": scope,
        "fold": (
            np.nan
            if fold is None
            else int(fold)
        ),
        "partition": partition,
        "available_sequences": int(
            available.sum()
        ),
        "all_sequence_lengths_60": (
            all_length_60
        ),
        "all_sequences_end_at_reference": (
            all_end_at_reference
        ),
        "all_master_session_spans_59": (
            all_session_span_59
        ),
        "all_endpoint_regime_ids_valid": (
            all_regimes_valid
        ),
        "history_unavailable_rows_have_length_zero": (
            unavailable_lengths_zero
        ),
        "future_feature_rows_detected": False,
        "passed": passed,
    }


def cross_boundary_audit(
    index: pd.DataFrame,
    *,
    scope: str,
    fold: Optional[int],
    partition: str,
) -> Dict[str, Any]:
    cross = (
        index[
            "sequence_history_available"
        ]
        &
        index[
            "lookback_crosses_partition_start"
        ]
    )

    cross_rows = index.loc[
        cross
    ]

    if cross_rows.empty:
        minimum_history = 0
        maximum_history = 0
        first_endpoint = pd.NaT
        last_endpoint = pd.NaT
        common_eligible_cross = 0
    else:
        minimum_history = int(
            cross_rows[
                "historical_rows_before_partition"
            ].min()
        )
        maximum_history = int(
            cross_rows[
                "historical_rows_before_partition"
            ].max()
        )
        first_endpoint = (
            cross_rows[
                "reference_date"
            ].min()
        )
        last_endpoint = (
            cross_rows[
                "reference_date"
            ].max()
        )
        common_eligible_cross = int(
            cross_rows[
                "sequence_eligible_common_multitask"
            ].sum()
        )

    return {
        "scope": scope,
        "fold": (
            np.nan
            if fold is None
            else int(fold)
        ),
        "partition": partition,
        "cross_boundary_endpoints": int(
            cross.sum()
        ),
        "cross_boundary_common_multitask_sequences": (
            common_eligible_cross
        ),
        "minimum_historical_rows_before_partition": (
            minimum_history
        ),
        "maximum_historical_rows_before_partition": (
            maximum_history
        ),
        "first_cross_boundary_endpoint": (
            first_endpoint
        ),
        "last_cross_boundary_endpoint": (
            last_endpoint
        ),
        "causal_cross_boundary_lookback_allowed": (
            ALLOW_VALIDATION_LOOKBACK_INTO_TRAIN
        ),
        "target_endpoint_still_partition_confined": (
            REQUIRE_TARGET_INSIDE_ENDPOINT_PARTITION
        ),
    }


# =============================================================================
# 12. PARQUET / CSV OUTPUT
# =============================================================================

def write_index_pair(
    frame: pd.DataFrame,
    parquet_path: Path,
    csv_path: Path,
) -> None:
    frame.to_parquet(
        parquet_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    csv_frame = frame.copy()

    for column in (
        "reference_date",
        "sequence_start_date",
        "sequence_end_date",
        "target_date_1d",
        "target_date_3d",
        "target_date_5d",
    ):
        if column in csv_frame.columns:
            csv_frame[column] = (
                pd.to_datetime(
                    csv_frame[
                        column
                    ],
                    errors="coerce",
                )
                .dt.strftime(
                    "%Y-%m-%d"
                )
            )

    csv_frame.to_csv(
        csv_path,
        index=False,
    )


def verify_index_roundtrip(
    original: pd.DataFrame,
    parquet_path: Path,
) -> None:
    observed = pd.read_parquet(
        parquet_path,
        engine="pyarrow",
    )

    if len(observed) != len(original):
        raise RuntimeError(
            f"Parquet row-count mismatch: {parquet_path}"
        )

    if list(observed.columns) != list(
        original.columns
    ):
        raise RuntimeError(
            f"Parquet column-order mismatch: {parquet_path}"
        )

    left = original[
        [
            "ticker",
            "reference_date",
        ]
    ].copy()

    right = observed[
        [
            "ticker",
            "reference_date",
        ]
    ].copy()

    left[
        "reference_date"
    ] = (
        pd.to_datetime(
            left[
                "reference_date"
            ],
            errors="raise",
        )
        .dt.normalize()
    )

    right[
        "reference_date"
    ] = (
        pd.to_datetime(
            right[
                "reference_date"
            ],
            errors="raise",
        )
        .dt.normalize()
    )

    if not left.equals(
        right
    ):
        raise RuntimeError(
            f"Parquet key mismatch: {parquet_path}"
        )

    for column in (
        "sequence_history_available",
        "sequence_eligible_1d",
        "sequence_eligible_3d",
        "sequence_eligible_5d",
        "sequence_eligible_common_multitask",
    ):
        if not np.array_equal(
            original[
                column
            ].to_numpy(),
            observed[
                column
            ].to_numpy(),
        ):
            raise RuntimeError(
                f"Parquet eligibility mismatch: {parquet_path}, {column}"
            )


# =============================================================================
# 13. SOURCE HASH REGISTER
# =============================================================================

def source_hashes() -> pd.DataFrame:
    sources = [
        (
            "current_step00",
            Path(
                cfg.__file__
            ).resolve(),
        ),
        (
            "current_step13",
            SCRIPT_FILE.resolve(),
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
            "step12_final_development",
            STEP12_FINAL_DEVELOPMENT.resolve(),
        ),
    ]

    for fold in FOLDS:
        for partition in (
            "train",
            "validation",
        ):
            sources.append(
                (
                    f"step12_fold_{fold}_{partition}",
                    step12_fold_path(
                        fold,
                        partition,
                    ).resolve(),
                )
            )

    for step, (
        manifest_path,
        sidecar_path,
    ) in UPSTREAM.items():
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
                "Step-13 source hash register."
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
# 14. VERIFY EXISTING STEP 13
# =============================================================================

def verify_existing() -> int:
    print("=" * 136)
    print("STEP 13 — VERIFY EXISTING CAUSAL SEQUENCE-INDEX FREEZE")
    print("=" * 136)

    step00 = verify_step00()
    upstream = verify_upstream(step00)

    manifest_sha = verify_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 13",
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
            "[FAIL] Stored Step-13 status is not PASS."
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

    for step in range(1, 13):
        field = f"step{step:02d}_manifest_sha256"

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

        print("=" * 136)
        print("STEP 13 VERIFICATION STATUS: FAIL")
        print("=" * 136)
        return 25

    if (
        sha256_file(
            Path(
                step00[
                    "lock_path"
                ]
            )
        )
        !=
        step00[
            "lock_sha256"
        ]
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
        "[PASS] Step-12 manifest          : "
        f"{upstream['manifest_sha'][12]}"
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
        "[PASS] Step-13 manifest          : "
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
    print("=" * 136)
    print("STEP 13 VERIFICATION STATUS: PASS")
    print("=" * 136)

    return 0


# =============================================================================
# 15. MAIN RUN
# =============================================================================

def run() -> int:
    print("=" * 136)
    print(STEP_NAME)
    print("=" * 136)

    stage = "prerequisite_verification"
    staging: Optional[Path] = None

    try:
        print(
            "[1/10] Verifying Step 00 and immutable Step 01 -> Step 12 chain..."
        )

        step00 = verify_step00()
        upstream = verify_upstream(step00)

        print(
            "       Configuration fingerprint: "
            f"{step00['fingerprint']}"
        )
        print(
            "       Step-12 manifest          : "
            f"{upstream['manifest_sha'][12]}"
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
            "[2/10] Checking immutable Step-13 output target..."
        )

        assert_fresh_target()
        cfg.create_directories()

        stage = "membership_load"

        print(
            "[3/10] Loading authoritative Step-10 membership ledger..."
        )

        membership = load_membership()

        feature_columns = upstream[
            "feature_columns"
        ]

        stock_id_map = {
            ticker: int(stock_id)
            for stock_id, ticker in enumerate(
                cfg.STOCK_TICKERS
            )
        }

        if len(stock_id_map) != EXPECTED_STOCK_COUNT:
            raise RuntimeError(
                "Stock-ID map does not match Step-00 stock count."
            )

        print(
            f"       Lookback               : {LOOKBACK}"
        )
        print(
            f"       Continuous predictors  : {len(feature_columns)}"
        )
        print(
            "       Sequence               : [t-59, ..., t]"
        )
        print(
            "       Validation train-history lookback : ALLOWED"
        )

        stage = "staging"

        staging = create_staging_root()

        count_rows = []
        integrity_rows = []
        cross_rows = []
        reconciliation_rows = []
        staged_records = []

        stage = "fold_sequences"

        print(
            "[4/10] Constructing five fold train/validation sequence indices..."
        )

        split_folds = {
            int(record["fold"]): record
            for record in upstream[
                "split_design"
            ][
                "folds"
            ]
        }

        for fold in FOLDS:
            if fold not in split_folds:
                raise RuntimeError(
                    f"Fold {fold} missing from frozen split design."
                )

            definition = split_folds[
                fold
            ]

            train_start = pd.Timestamp(
                definition[
                    "train_start"
                ]
            )

            train_end = pd.Timestamp(
                definition[
                    "train_end"
                ]
            )

            validation_start = pd.Timestamp(
                definition[
                    "validation_start"
                ]
            )

            validation_end = pd.Timestamp(
                definition[
                    "validation_end"
                ]
            )

            train_panel = load_step12_panel(
                step12_fold_path(
                    fold,
                    "train",
                ),
                feature_columns,
                label=f"fold_{fold}_train",
            )

            validation_panel = load_step12_panel(
                step12_fold_path(
                    fold,
                    "validation",
                ),
                feature_columns,
                label=f"fold_{fold}_validation",
            )

            if (
                train_panel[
                    "Date"
                ].max()
                >=
                validation_panel[
                    "Date"
                ].min()
            ):
                raise RuntimeError(
                    f"Fold {fold}: train/validation chronology violated."
                )

            reconciliation_rows.append(
                reconcile_partition_with_membership(
                    train_panel,
                    membership,
                    reference_column=(
                        f"fold_{fold}_train_reference"
                    ),
                    eligibility_prefix=(
                        f"fold_{fold}_train"
                    ),
                    label=(
                        f"fold_{fold}_train"
                    ),
                )
            )

            reconciliation_rows.append(
                reconcile_partition_with_membership(
                    validation_panel,
                    membership,
                    reference_column=(
                        f"fold_{fold}_validation_reference"
                    ),
                    eligibility_prefix=(
                        f"fold_{fold}_validation"
                    ),
                    label=(
                        f"fold_{fold}_validation"
                    ),
                )
            )

            train_history = build_history_lookup(
                train_panel,
                feature_columns,
                label=f"fold_{fold}_train_history",
            )

            validation_history_frame = (
                pd.concat(
                    [
                        train_panel,
                        validation_panel,
                    ],
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

            validation_history = build_history_lookup(
                validation_history_frame,
                feature_columns,
                label=f"fold_{fold}_train_plus_validation_history",
            )

            train_index = build_sequence_index(
                train_panel,
                train_history,
                scope=f"fold_{fold}_train",
                fold=fold,
                partition="train",
                period_start=train_start,
                period_end=train_end,
                stock_id_map=stock_id_map,
            )

            validation_index = build_sequence_index(
                validation_panel,
                validation_history,
                scope=f"fold_{fold}_validation",
                fold=fold,
                partition="validation",
                period_start=validation_start,
                period_end=validation_end,
                stock_id_map=stock_id_map,
            )

            fold_dir = (
                staging
                /
                f"fold_{fold}"
            )

            for partition, index in (
                (
                    "train",
                    train_index,
                ),
                (
                    "validation",
                    validation_index,
                ),
            ):
                scope = f"fold_{fold}_{partition}"

                count_rows.append(
                    count_summary(
                        index,
                        scope=scope,
                        fold=fold,
                        partition=partition,
                    )
                )

                integrity_rows.append(
                    integrity_audit(
                        index,
                        scope=scope,
                        fold=fold,
                        partition=partition,
                    )
                )

                cross_rows.append(
                    cross_boundary_audit(
                        index,
                        scope=scope,
                        fold=fold,
                        partition=partition,
                    )
                )

                parquet_path = (
                    fold_dir
                    /
                    f"13_fold_{fold}_{partition}_sequence_index.parquet"
                )

                csv_path = (
                    fold_dir
                    /
                    f"13_fold_{fold}_{partition}_sequence_index.csv"
                )

                write_index_pair(
                    index,
                    parquet_path,
                    csv_path,
                )

                verify_index_roundtrip(
                    index,
                    parquet_path,
                )

                for kind, path in (
                    (
                        "sequence_index_parquet",
                        parquet_path,
                    ),
                    (
                        "sequence_index_csv",
                        csv_path,
                    ),
                ):
                    staged_records.append(
                        {
                            "scope": scope,
                            "artifact_kind": kind,
                            "fold": fold,
                            "partition": partition,
                            "file_name": path.name,
                            "sha256": sha256_file(path),
                            "size_bytes": int(
                                path.stat().st_size
                            ),
                        }
                    )

            tr = count_rows[-2]
            va = count_rows[-1]

            print(
                f"       Fold {fold}: "
                f"train common={tr['sequence_eligible_common_multitask']:>6,} | "
                f"val common={va['sequence_eligible_common_multitask']:>5,} | "
                f"val cross-boundary={va['cross_partition_lookback_endpoints']:>5,}"
            )

        stage = "final_development_sequences"

        print(
            "[5/10] Constructing final-development sequence index..."
        )

        final_panel = load_step12_panel(
            STEP12_FINAL_DEVELOPMENT,
            feature_columns,
            label="final_development",
        )

        final_membership = membership.copy()
        final_membership[
            "_final_development_reference"
        ] = (
            final_membership[
                "outer_partition"
            ]
            ==
            "development"
        )

        reconciliation_rows.append(
            reconcile_partition_with_membership(
                final_panel,
                final_membership,
                reference_column=(
                    "_final_development_reference"
                ),
                eligibility_prefix=(
                    "outer_development"
                ),
                label=(
                    "final_development"
                ),
            )
        )

        final_history = build_history_lookup(
            final_panel,
            feature_columns,
            label="final_development_history",
        )

        development_design = upstream[
            "split_design"
        ][
            "development"
        ]

        final_index = build_sequence_index(
            final_panel,
            final_history,
            scope="final_development",
            fold=None,
            partition="final_development",
            period_start=pd.Timestamp(
                development_design[
                    "start"
                ]
            ),
            period_end=pd.Timestamp(
                development_design[
                    "end"
                ]
            ),
            stock_id_map=stock_id_map,
        )

        final_count = count_summary(
            final_index,
            scope="final_development",
            fold=None,
            partition="final_development",
        )

        final_integrity = integrity_audit(
            final_index,
            scope="final_development",
            fold=None,
            partition="final_development",
        )

        final_cross = cross_boundary_audit(
            final_index,
            scope="final_development",
            fold=None,
            partition="final_development",
        )

        count_rows.append(
            final_count
        )
        integrity_rows.append(
            final_integrity
        )
        cross_rows.append(
            final_cross
        )

        final_dir = (
            staging
            /
            "final_development"
        )

        final_parquet = (
            final_dir
            /
            "13_final_development_sequence_index.parquet"
        )

        final_csv = (
            final_dir
            /
            "13_final_development_sequence_index.csv"
        )

        write_index_pair(
            final_index,
            final_parquet,
            final_csv,
        )

        verify_index_roundtrip(
            final_index,
            final_parquet,
        )

        for kind, path in (
            (
                "sequence_index_parquet",
                final_parquet,
            ),
            (
                "sequence_index_csv",
                final_csv,
            ),
        ):
            staged_records.append(
                {
                    "scope": "final_development",
                    "artifact_kind": kind,
                    "fold": None,
                    "partition": "final_development",
                    "file_name": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                }
            )

        print(
            "       Final-development refs   : "
            f"{len(final_index):,}"
        )
        print(
            "       Final-development history: "
            f"{int(final_index['sequence_history_available'].sum()):,}"
        )
        print(
            "       Final-development common : "
            f"{final_count['sequence_eligible_common_multitask']:,}"
        )

        stage = "audits"

        print(
            "[6/10] Building count, cross-boundary, integrity and reconciliation audits..."
        )

        count_audit = pd.DataFrame(
            count_rows
        )

        integrity = pd.DataFrame(
            integrity_rows
        )

        cross = pd.DataFrame(
            cross_rows
        )

        reconciliation = pd.DataFrame(
            reconciliation_rows
        )

        if not integrity[
            "passed"
        ].all():
            raise RuntimeError(
                "One or more sequence integrity audits failed."
            )

        if not reconciliation[
            "keys_exact_step10_to_step12"
        ].all():
            raise RuntimeError(
                "Step-10/12 key reconciliation failed."
            )

        if not reconciliation[
            "eligibility_flags_exact_step10_to_step12"
        ].all():
            raise RuntimeError(
                "Step-10/12 eligibility reconciliation failed."
            )

        if len(integrity) != 11:
            raise RuntimeError(
                f"Expected 11 integrity records; found {len(integrity)}."
            )

        if len(reconciliation) != 11:
            raise RuntimeError(
                f"Expected 11 reconciliation records; found {len(reconciliation)}."
            )

        print(
            "       Partition indices          : 11"
        )
        print(
            "       All integrity audits pass : "
            f"{bool(integrity['passed'].all())}"
        )
        print(
            "       Step-10/12 keys exact      : "
            f"{bool(reconciliation['keys_exact_step10_to_step12'].all())}"
        )
        print(
            "       Step-10/12 flags exact     : "
            f"{bool(reconciliation['eligibility_flags_exact_step10_to_step12'].all())}"
        )

        stage = "promotion"

        print(
            "[7/10] Promoting sequence-index artifacts and writing frozen audits..."
        )

        if OUT_DIR.exists():
            raise RuntimeError(
                "Canonical Step-13 output directory appeared unexpectedly."
            )

        os.replace(
            staging,
            OUT_DIR,
        )
        staging = None

        promoted_records = []

        for record in staged_records:
            if record[
                "partition"
            ] == "final_development":
                path = (
                    FINAL_DEVELOPMENT_DIR
                    /
                    record[
                        "file_name"
                    ]
                )
            else:
                path = (
                    OUT_DIR
                    /
                    f"fold_{record['fold']}"
                    /
                    record[
                        "file_name"
                    ]
                )

            if not path.exists():
                raise RuntimeError(
                    f"Promoted Step-13 artifact missing: {path}"
                )

            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(
                    f"Promoted Step-13 artifact hash mismatch: {path}"
                )

            promoted_records.append(
                {
                    "scope": record["scope"],
                    "artifact_kind": (
                        record[
                            "artifact_kind"
                        ]
                    ),
                    "fold": record["fold"],
                    "partition": (
                        record[
                            "partition"
                        ]
                    ),
                    "file_path": str(path),
                    "relative_path": (
                        relative_to_experiment(path)
                    ),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                    "sha256": record["sha256"],
                }
            )

        write_csv(
            COUNT_SUMMARY_CSV,
            count_audit,
        )

        write_csv(
            CROSS_BOUNDARY_AUDIT_CSV,
            cross,
        )

        write_csv(
            INTEGRITY_AUDIT_CSV,
            integrity,
        )

        write_csv(
            MEMBERSHIP_RECONCILIATION_CSV,
            reconciliation,
        )

        write_csv(
            SOURCE_HASH_CSV,
            source_hashes(),
        )

        for path in (
            COUNT_SUMMARY_CSV,
            CROSS_BOUNDARY_AUDIT_CSV,
            INTEGRITY_AUDIT_CSV,
            MEMBERSHIP_RECONCILIATION_CSV,
            SOURCE_HASH_CSV,
        ):
            promoted_records.append(
                {
                    "scope": path.stem,
                    "artifact_kind": "audit",
                    "fold": None,
                    "partition": None,
                    "file_path": str(path),
                    "relative_path": (
                        relative_to_experiment(path)
                    ),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                    "sha256": sha256_file(path),
                }
            )

        stage = "freeze"

        print(
            "[8/10] Freezing Step-13 summary, source snapshot and manifest..."
        )

        if (
            sha256_file(
                Path(
                    step00[
                        "lock_path"
                    ]
                )
            )
            !=
            step00[
                "lock_sha256"
            ]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step 13."
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
                step00[
                    "fingerprint"
                ]
            ),
            "step12_manifest_sha256": (
                upstream[
                    "manifest_sha"
                ][
                    12
                ]
            ),
            "feature_interface_sha256": (
                upstream[
                    "feature_interface_sha256"
                ]
            ),
            "split_design_sha256": (
                upstream[
                    "split_design_sha256"
                ]
            ),
            "sequence_policy": {
                "lookback_sessions": LOOKBACK,
                "sequence_formula": (
                    "[x_(t-59), ..., x_t]"
                ),
                "feature_count": (
                    EXPECTED_FEATURE_COUNT
                ),
                "validation_may_use_prior_training_history": True,
                "training_may_use_validation_history": False,
                "same_ticker_required": True,
                "same_continuity_segment_required": True,
                "target_endpoint_confined_to_endpoint_partition": True,
                "feature_rows_after_reference_allowed": False,
            },
            "partitions": (
                count_audit.to_dict(
                    orient="records"
                )
            ),
            "integrity": {
                "audit_records": int(
                    len(integrity)
                ),
                "all_pass": bool(
                    integrity[
                        "passed"
                    ].all()
                ),
                "future_feature_rows_detected": int(
                    integrity[
                        "future_feature_rows_detected"
                    ].sum()
                ),
            },
            "pipeline_state": {
                "sequence_indices_constructed": True,
                "final_development_sequence_index_constructed": True,
                "sequence_arrays_materialized": False,
                "feature_selection_applied": False,
                "models_fitted": False,
                "probability_threshold_selection_performed": False,
                "final_test_sequences_constructed": False,
            },
            "test_protection": {
                "step10_seal_status": (
                    upstream[
                        "test_seal_status"
                    ]
                ),
                "sealed_test_sha256_from_metadata_only": (
                    upstream[
                        "sealed_test_sha256_from_metadata_only"
                    ]
                ),
                "sealed_test_parquet_opened_or_hashed_by_step13": False,
                "global_final_test_state": (
                    step00[
                        "state"
                    ]
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

        atomic_text(
            SUMMARY_TXT,
            "\n".join(
                [
                    STEP_NAME,
                    "=" * 104,
                    "Status: PASS",
                    (
                        "Step-12 manifest SHA-256: "
                        f"{upstream['manifest_sha'][12]}"
                    ),
                    (
                        "63-feature interface SHA-256: "
                        f"{upstream['feature_interface_sha256']}"
                    ),
                    (
                        "Split-design SHA-256: "
                        f"{upstream['split_design_sha256']}"
                    ),
                    f"Lookback sessions: {LOOKBACK}",
                    (
                        "Validation causal training-history lookback: True"
                    ),
                    (
                        "Final-development sequence index: True"
                    ),
                    (
                        "Sequence arrays materialized: False"
                    ),
                    (
                        "Final-test parquet accessed: False"
                    ),
                    (
                        "Global final-test state: "
                        f"{step00['state']}"
                    ),
                ]
            )
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
                    "artifact_kind": (
                        "freeze_metadata"
                    ),
                    "fold": None,
                    "partition": None,
                    "file_path": str(path),
                    "relative_path": (
                        relative_to_experiment(path)
                    ),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                    "sha256": sha256_file(path),
                }
            )

        write_csv(
            HASH_REGISTER_CSV,
            pd.DataFrame(
                promoted_records
            ),
        )

        artifact_paths = [
            Path(
                record[
                    "file_path"
                ]
            )
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
            "experiment_tag": (
                cfg.EXPERIMENT_TAG
            ),
            "configuration_fingerprint_sha256": (
                step00[
                    "fingerprint"
                ]
            ),
            "step00_manifest_sha256": (
                step00[
                    "step00_manifest_sha256"
                ]
            ),
            **{
                f"step{step:02d}_manifest_sha256": (
                    upstream[
                        "manifest_sha"
                    ][
                        step
                    ]
                )
                for step in range(
                    1,
                    13,
                )
            },
            "feature_interface_sha256": (
                upstream[
                    "feature_interface_sha256"
                ]
            ),
            "ordered_63_features": (
                feature_columns
            ),
            "split_design_sha256": (
                upstream[
                    "split_design_sha256"
                ]
            ),
            "stock_id_map": stock_id_map,
            "source_integrity": {
                "step10_membership_sha256": (
                    upstream[
                        "membership_sha256"
                    ]
                ),
                "step12_fold_panel_hashes": (
                    upstream[
                        "fold_panel_hashes"
                    ]
                ),
                "step12_final_development_panel_sha256": (
                    upstream[
                        "final_development_panel_sha256"
                    ]
                ),
            },
            "sequence_policy": {
                "lookback_sessions": LOOKBACK,
                "sequence_formula": (
                    "[x_(t-59), ..., x_t]"
                ),
                "feature_count": EXPECTED_FEATURE_COUNT,
                "causal_cross_split_validation_lookback_allowed": True,
                "lookback_required_inside_same_split": False,
                "same_ticker_required": True,
                "same_continuity_segment_required": True,
                "target_endpoint_required_inside_evaluation_partition": True,
                "feature_rows_after_reference_allowed": False,
            },
            "test_protection": {
                "step10_seal_status": (
                    upstream[
                        "test_seal_status"
                    ]
                ),
                "global_final_test_state": (
                    step00[
                        "state"
                    ]
                ),
                "sealed_test_parquet_opened_or_hashed_by_step13": False,
                "final_test_sequences_constructed": False,
                "test_performance_computed": False,
            },
            "pipeline_state": {
                "sequence_indices_constructed": True,
                "final_development_sequence_index_constructed": True,
                "sequence_arrays_materialized": False,
                "feature_selection_applied": False,
                "models_fitted": False,
                "probability_threshold_selection_performed": False,
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
            "[9/10] Validating frozen Step-13 artifact set..."
        )

        verify_sidecar(
            MANIFEST_JSON,
            MANIFEST_SHA,
            "Step 13",
        )

        for record in manifest[
            "artifacts"
        ].values():
            path = Path(
                record[
                    "path"
                ]
            )

            if (
                not path.exists()
                or
                sha256_file(path)
                !=
                record[
                    "sha256"
                ]
            ):
                raise RuntimeError(
                    f"Frozen Step-13 artifact mismatch: {path}"
                )

        if (
            sha256_file(
                Path(
                    step00[
                        "lock_path"
                    ]
                )
            )
            !=
            step00[
                "lock_sha256"
            ]
        ):
            raise RuntimeError(
                "Global final-test lock changed during final validation."
            )

        stage = "report"

        print(
            "[10/10] Final Step-13 report..."
        )

        print()
        print("-" * 136)
        print("SEQUENCE POLICY")
        print("-" * 136)
        print(
            f"Lookback                    : {LOOKBACK} sessions"
        )
        print(
            "Input interval               : t-59 through t"
        )
        print(
            "Validation training history  : ALLOWED"
        )
        print(
            "Training validation history  : PROHIBITED"
        )
        print(
            "Same ticker                  : REQUIRED"
        )
        print(
            "Same continuity segment      : REQUIRED"
        )
        print(
            "Target inside partition      : REQUIRED"
        )
        print(
            "Sequence arrays              : NOT MATERIALIZED"
        )

        print("-" * 136)
        print("ROLLING-ORIGIN SEQUENCE COUNTS")
        print("-" * 136)

        fold_counts = count_audit.loc[
            count_audit[
                "fold"
            ].notna()
        ].copy()

        for row in (
            fold_counts
            .sort_values(
                [
                    "fold",
                    "partition",
                ]
            )
            .itertuples(
                index=False
            )
        ):
            print(
                f"Fold {int(row.fold)} {row.partition:<10} | "
                f"refs={row.reference_rows:>6,} | "
                f"history={row.history_available_rows:>6,} | "
                f"1d={row.sequence_eligible_1d:>6,} | "
                f"3d={row.sequence_eligible_3d:>6,} | "
                f"5d={row.sequence_eligible_5d:>6,} | "
                f"common={row.sequence_eligible_common_multitask:>6,} | "
                f"cross={row.cross_partition_lookback_endpoints:>5,}"
            )

        print("-" * 136)
        print("CAUSAL VALIDATION CROSS-BOUNDARY LOOKBACK")
        print("-" * 136)

        for row in (
            cross.loc[
                cross[
                    "partition"
                ]
                ==
                "validation"
            ]
            .sort_values(
                "fold"
            )
            .itertuples(
                index=False
            )
        ):
            print(
                f"Fold {int(row.fold)}: "
                f"endpoints using training history="
                f"{row.cross_boundary_endpoints:,} | "
                f"historical rows/window="
                f"{row.minimum_historical_rows_before_partition}"
                f".."
                f"{row.maximum_historical_rows_before_partition} | "
                f"common eligible="
                f"{row.cross_boundary_common_multitask_sequences:,}"
            )

        print("-" * 136)
        print("FINAL-DEVELOPMENT SEQUENCE STATE")
        print("-" * 136)
        print(
            f"Reference rows               : {final_count['reference_rows']:,}"
        )
        print(
            f"History-available rows       : {final_count['history_available_rows']:,}"
        )
        print(
            f"1d sequences                 : {final_count['sequence_eligible_1d']:,}"
        )
        print(
            f"3d sequences                 : {final_count['sequence_eligible_3d']:,}"
        )
        print(
            f"5d/common sequences          : {final_count['sequence_eligible_5d']:,}"
        )

        print("-" * 136)
        print("INTEGRITY")
        print("-" * 136)
        print(
            f"Integrity records            : {len(integrity)}"
        )
        print(
            "All integrity audits pass   : "
            f"{bool(integrity['passed'].all())}"
        )
        print(
            "Future feature rows detected: "
            f"{int(integrity['future_feature_rows_detected'].sum())}"
        )
        print(
            "Membership reconciliations  : "
            f"{len(reconciliation)}"
        )
        print(
            "All membership keys exact   : "
            f"{bool(reconciliation['keys_exact_step10_to_step12'].all())}"
        )

        print("-" * 136)
        print("PIPELINE STATE")
        print("-" * 136)
        print(
            "Sequence indices            : COMPLETE"
        )
        print(
            "Final-development index     : COMPLETE"
        )
        print(
            "Sequence arrays             : NOT MATERIALIZED"
        )
        print(
            "Feature selection           : False"
        )
        print(
            "Models fitted               : False"
        )
        print(
            "Final-test parquet accessed : False"
        )
        print(
            "Global final-test state     : "
            f"{step00['state']}"
        )

        print("-" * 136)
        print("OUTPUT FILES")
        print("-" * 136)
        print(
            f"Sequence-index root  : {OUT_DIR}"
        )
        print(
            f"Count summary        : {COUNT_SUMMARY_CSV}"
        )
        print(
            f"Cross-boundary audit : {CROSS_BOUNDARY_AUDIT_CSV}"
        )
        print(
            f"Integrity audit      : {INTEGRITY_AUDIT_CSV}"
        )
        print(
            f"Membership audit     : {MEMBERSHIP_RECONCILIATION_CSV}"
        )
        print(
            f"Manifest             : {MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256     : {manifest_sha}"
        )

        print("=" * 136)
        print(
            "PASS: Causal 60-session fold train/validation sequence indices "
            "were constructed."
        )
        print(
            "PASS: Validation windows may use preceding training-period rows "
            "transformed with the same fold-training scaler."
        )
        print(
            "PASS: Every available sequence stays within one ticker and one "
            "continuity segment and ends at its reference endpoint."
        )
        print(
            "PASS: Target eligibility remains exactly the frozen Step-10 "
            "partition eligibility."
        )
        print(
            "PASS: Final-development sequence indices were frozen without "
            "accessing the final test."
        )
        print(
            "PASS: No 60x63 arrays, feature selection, probability thresholds, "
            "or models were created."
        )
        print("=" * 136)
        print(
            "Do not run the old Step-13 script. "
            "Do not run Step 14 yet. Paste the complete terminal output for audit."
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
            f"13_sequence_indices_FAILED_{stamp}.json"
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
        print("=" * 136)
        print("STEP 13 FAILED — SEQUENCE INDICES NOT FROZEN")
        print("=" * 136)
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

        print("=" * 136)
        print(
            "Do not proceed to Step 14."
        )

        return 30


# =============================================================================
# 16. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean-V3 causal 60-session sequence-index construction."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an existing frozen Step-13 stage without recomputation."
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
            "\nStep 13 interrupted by user."
        )
        sys.exit(
            130
        )

    except Exception as exc:
        print()
        print("=" * 136)
        print("STEP 13 UNHANDLED FAILURE")
        print("=" * 136)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        traceback.print_exc()
        print("=" * 136)
        sys.exit(
            1
        )
