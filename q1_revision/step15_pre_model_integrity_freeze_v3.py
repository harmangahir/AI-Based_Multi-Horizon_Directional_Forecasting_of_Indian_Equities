#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 15 — CLEAN V3 COMPREHENSIVE PRE-MODELLING INTEGRITY FREEZE

Purpose
-------
Independently audit the complete DEVELOPMENT modelling interface after
Step 14 and before ANY forecasting model is fitted.

This stage verifies:
- Step 00 -> Step 14 manifest continuity and environment consistency;
- every persisted Step-14 artifact against its frozen SHA-256;
- all five train/validation model-array pairs PLUS final-development arrays;
- exact array shapes, dtypes and row ordering;
- exact X_full -> X_stock / X_market modality slicing;
- metadata -> direction labels, auxiliary targets, stock IDs and regime IDs;
- exact reproduction of auxiliary-target scaling from persisted scalers;
- exact first-row and endpoint alignment to canonical Step-12 feature sources;
- deterministic full-window reconstruction samples from Step-12 sources;
- causal validation lookback across the train/validation boundary;
- chronological train/validation separation;
- balanced 20-stock representation;
- absence of target columns from the 63-predictor interface;
- Step-13 horizon-specific sequence eligibility versus Step-14 common-multitask
  materialization;
- final-development refit-array integrity;
- final-test protection state.

No model fitting.
No feature selection.
No probability-threshold selection.
No final-test parquet access.

Run
---
    python step15_pre_model_integrity_freeze_v3.py

Verify
------
    python step15_pre_model_integrity_freeze_v3.py --verify-only
"""

from __future__ import annotations

import argparse
import gc
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

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# 0. SCRIPT / CONFIG
# =============================================================================

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent

if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError(
        "step15_pre_model_integrity_freeze_v3.py must be inside "
        "<repository>/q1_revision/."
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg


# =============================================================================
# 1. FROZEN DESIGN
# =============================================================================

STEP_NUMBER = 15
STEP_ID = "15"
STEP_NAME = "STEP 15 — COMPREHENSIVE PRE-MODELLING INTEGRITY FREEZE"
SCHEMA_VERSION = "1.0"

FOLDS = (1, 2, 3, 4, 5)
PARTITIONS = ("train", "validation")
HORIZONS = (1, 3, 5)

LOOKBACK = 60
STOCK_FEATURE_COUNT = 47
MARKET_FEATURE_COUNT = 16
TOTAL_FEATURE_COUNT = 63
EXPECTED_STOCK_COUNT = len(cfg.STOCK_TICKERS)

EXPECTED_VALIDATION_CROSS_BOUNDARY = (
    (LOOKBACK - 1) * EXPECTED_STOCK_COUNT
)

EXPECTED_COMMON_COUNTS = {
    1: {"train": 17620, "validation": 4900},
    2: {"train": 22620, "validation": 4860},
    3: {"train": 27580, "validation": 4860},
    4: {"train": 32540, "validation": 4800},
    5: {"train": 37440, "validation": 4820},
}

EXPECTED_FINAL_DEVELOPMENT_COMMON = 42360

DIRECTION_COLUMNS = [
    "direction_1d",
    "direction_3d",
    "direction_5d",
]

AUXILIARY_TARGET_COLUMNS = [
    "future_close_1d",
    "future_close_3d",
    "future_close_5d",
    "future_return_1d",
    "future_return_3d",
    "future_return_5d",
    "future_log_return_1d",
    "future_log_return_3d",
    "future_log_return_5d",
    "realized_volatility_5d",
]

TARGET_METADATA_COLUMNS = [
    "target_date_1d",
    "target_date_3d",
    "target_date_5d",
]

MODEL_ARRAY_NAMES = (
    "X_full",
    "X_stock",
    "X_market",
    "y_direction",
    "y_aux_raw",
    "y_aux_scaled",
    "stock_id",
    "regime_id",
)

FULL_SEQUENCE_SAMPLE_SIZE = 64
ARRAY_CHUNK_SIZE = 512

EXPECTED_STEP14_MODEL_ARTIFACTS = 105
EXPECTED_STEP14_TOTAL_MANIFEST_ARTIFACTS = 116

FORBIDDEN_TEST_FILENAME = "10_final_test_locked_panel.parquet"
FORBIDDEN_TEST_TOKENS = (
    "final_test",
    "heldout",
    "held_out",
)


# =============================================================================
# 2. UPSTREAM PATHS
# =============================================================================

STEP01_ENVIRONMENT_AUDIT = (
    cfg.ENVIRONMENT_DIR / "01_environment_audit.json"
)

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
    13: (
        cfg.MANIFEST_DIR / "13_sequence_index_manifest.json",
        cfg.MANIFEST_DIR / "13_sequence_index_manifest.sha256",
    ),
    14: (
        cfg.MANIFEST_DIR / "14_model_array_manifest.json",
        cfg.MANIFEST_DIR / "14_model_array_manifest.sha256",
    ),
}

STEP10_TEST_SEAL = cfg.MANIFEST_DIR / "10_final_test_seal.json"

STEP12_ROOT = cfg.FEATURE_DATA_DIR / "step12_volatility_regimes"
STEP12_FINAL_DEVELOPMENT = (
    STEP12_ROOT
    / "final_development"
    / "12_final_development_scaled_regimes.parquet"
)

STEP13_ROOT = cfg.FEATURE_DATA_DIR / "step13_sequence_indices"
STEP13_FINAL_DEVELOPMENT_INDEX = (
    STEP13_ROOT
    / "final_development"
    / "13_final_development_sequence_index.parquet"
)

STEP14_ROOT = cfg.FEATURE_DATA_DIR / "step14_model_arrays"
STEP14_FINAL_DEVELOPMENT_DIR = STEP14_ROOT / "final_development"


# =============================================================================
# 3. OUTPUT PATHS
# =============================================================================

INTEGRITY_SUMMARY_CSV = (
    cfg.AUDIT_DIR / "15_pre_model_integrity_summary.csv"
)

HASH_VERIFICATION_CSV = (
    cfg.AUDIT_DIR / "15_artifact_hash_verification.csv"
)

SOURCE_ALIGNMENT_CSV = (
    cfg.AUDIT_DIR / "15_sequence_source_alignment_audit.csv"
)

LABEL_DISTRIBUTION_CSV = (
    cfg.AUDIT_DIR / "15_development_sequence_label_distribution.csv"
)

AUXILIARY_AUDIT_CSV = (
    cfg.AUDIT_DIR / "15_auxiliary_scaler_reproduction_audit.csv"
)

MODEL_INPUT_CONTRACT_CSV = (
    cfg.AUDIT_DIR / "15_model_input_contract.csv"
)

SOURCE_HASH_CSV = (
    cfg.AUDIT_DIR / "15_source_file_hashes.csv"
)

FREEZE_JSON = (
    cfg.MANIFEST_DIR / "15_pre_model_freeze.json"
)

MANIFEST_JSON = (
    cfg.MANIFEST_DIR / "15_pre_model_integrity_manifest.json"
)

MANIFEST_SHA = (
    cfg.MANIFEST_DIR / "15_pre_model_integrity_manifest.sha256"
)

HASH_REGISTER_CSV = (
    cfg.AUDIT_DIR / "15_integrity_output_hashes.csv"
)

FROZEN_SCRIPT = (
    cfg.MANIFEST_DIR / "step15_pre_model_integrity_freeze_frozen.py"
)

SUMMARY_TXT = (
    cfg.REPORT_DIR / "15_pre_model_integrity_summary.txt"
)

FAILED_DIR = (
    cfg.AUDIT_DIR / "step15_failed_attempts"
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


def atomic_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:
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


def atomic_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        text,
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def write_csv(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    frame.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
    )

    os.replace(
        temporary,
        path,
    )


def relative_to_experiment(
    path: Path,
) -> str:
    return str(
        path.resolve().relative_to(
            cfg.EXPERIMENT_ROOT_DIR.resolve()
        )
    )


def artifact_record(
    path: Path,
) -> Dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": relative_to_experiment(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def package_version(
    name: str,
) -> Optional[str]:
    try:
        return importlib_metadata.version(name)

    except importlib_metadata.PackageNotFoundError:
        return None


# =============================================================================
# 5. STEP 00 / ENVIRONMENT
# =============================================================================

def verify_step00() -> Dict[str, Any]:
    cfg.validate_configuration()

    payload = cfg.build_configuration_payload()

    fingerprint = cfg.canonical_json_sha256(
        payload
    )

    manifest_path, manifest_sha, status = cfg.verify_existing_freeze(
        configuration_payload=payload,
        configuration_fingerprint=fingerprint,
    )

    if status != "VERIFIED_EXISTING":
        raise RuntimeError(
            "Step-00 freeze is not VERIFIED_EXISTING."
        )

    configured_lookback = int(
        getattr(
            cfg,
            "LOOKBACK",
            LOOKBACK,
        )
    )

    if configured_lookback != LOOKBACK:
        raise RuntimeError(
            f"Configured lookback is {configured_lookback}; expected {LOOKBACK}."
        )

    lock_path = cfg.get_final_test_lock_path()

    lock = json.loads(
        lock_path.read_text(
            encoding="utf-8"
        )
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
        "step00_manifest_sha256": manifest_sha,
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "state": lock["state"],
    }


def verify_environment() -> Dict[str, Any]:
    if not STEP01_ENVIRONMENT_AUDIT.exists():
        raise RuntimeError(
            f"Step-01 environment audit missing: {STEP01_ENVIRONMENT_AUDIT}"
        )

    audit = json.loads(
        STEP01_ENVIRONMENT_AUDIT.read_text(
            encoding="utf-8"
        )
    )

    expected_python = (
        audit
        .get("system", {})
        .get("python_executable")
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

    for row in audit.get(
        "packages",
        [],
    ):
        if not row.get(
            "required",
            False,
        ):
            continue

        distribution = row.get(
            "distribution"
        )

        frozen_version = row.get(
            "version"
        )

        current_version = package_version(
            distribution
        )

        if current_version != frozen_version:
            drift.append(
                (
                    distribution,
                    frozen_version,
                    current_version,
                )
            )

    if drift:
        raise RuntimeError(
            "Required-package drift detected:\n"
            +
            "\n".join(
                f"{name}: frozen={frozen!r}, current={current!r}"
                for name, frozen, current in drift
            )
        )

    return {
        "python_executable": expected_python,
        "numpy_version": np.__version__,
        "joblib_version": package_version("joblib"),
    }


# =============================================================================
# 6. UPSTREAM MANIFEST VERIFICATION
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
        sidecar_path.read_text(
            encoding="utf-8"
        )
        .strip()
        .split()[0]
    )

    observed = sha256_file(
        manifest_path
    )

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

    for key in (
        "artifacts",
        "canonical_artifacts",
    ):
        for record in manifest.get(
            key,
            {},
        ).values():

            if not isinstance(
                record,
                dict,
            ):
                continue

            path_text = record.get(
                "path"
            )

            if (
                path_text
                and
                Path(path_text).name
                ==
                filename
            ):
                matches.append(
                    record
                )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest artifact named "
            f"{filename}; found {len(matches)}."
        )

    return matches[0]


def verify_upstream(
    step00: Dict[str, Any],
) -> Dict[str, Any]:
    shas: Dict[int, str] = {}
    manifests: Dict[int, Dict[str, Any]] = {}

    for step in range(
        1,
        15,
    ):
        manifest_path, sidecar_path = (
            UPSTREAM[
                step
            ]
        )

        shas[
            step
        ] = verify_sidecar(
            manifest_path,
            sidecar_path,
            f"Step {step:02d}",
        )

        manifests[
            step
        ] = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

    # Step 01.
    if manifests[
        1
    ].get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Step-01 status is not PASS."
        )

    if (
        manifests[
            1
        ].get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00[
            "fingerprint"
        ]
    ):
        raise RuntimeError(
            "Step-01 fingerprint mismatch."
        )

    # Step 02.
    if (
        manifests[
            2
        ]
        .get(
            "step",
            {},
        )
        .get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Step-02 status is not PASS."
        )

    if (
        manifests[
            2
        ]
        .get(
            "step00",
            {},
        )
        .get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00[
            "fingerprint"
        ]
    ):
        raise RuntimeError(
            "Step-02 fingerprint mismatch."
        )

    if (
        manifests[
            2
        ]
        .get(
            "step01",
            {},
        )
        .get(
            "step01_manifest_sha256"
        )
        !=
        shas[
            1
        ]
    ):
        raise RuntimeError(
            "Step-02 -> Step-01 chain mismatch."
        )

    # Steps 03 -> 14.
    for step in range(
        3,
        15,
    ):
        manifest = manifests[
            step
        ]

        if (
            manifest
            .get(
                "step",
                {},
            )
            .get(
                "status"
            )
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
            step00[
                "fingerprint"
            ]
        ):
            raise RuntimeError(
                f"Step-{step:02d} fingerprint mismatch."
            )

        for previous in range(
            1,
            step,
        ):
            field = (
                f"step{previous:02d}_manifest_sha256"
            )

            if (
                field in manifest
                and
                manifest[
                    field
                ]
                !=
                shas[
                    previous
                ]
            ):
                raise RuntimeError(
                    f"Step-{step:02d} chain mismatch: {field}"
                )

    environment = verify_environment()

    step08 = manifests[
        8
    ]

    step13 = manifests[
        13
    ]

    step14 = manifests[
        14
    ]

    all_features = step14.get(
        "ordered_63_features"
    )

    stock_features = step14.get(
        "stock_features"
    )

    market_features = step14.get(
        "market_features"
    )

    interface_sha = step14.get(
        "feature_interface_sha256"
    )

    split_design_sha = step14.get(
        "split_design_sha256"
    )

    if (
        not isinstance(
            all_features,
            list,
        )
        or
        len(
            all_features
        )
        !=
        TOTAL_FEATURE_COUNT
    ):
        raise RuntimeError(
            "Step-14 does not freeze exactly 63 predictors."
        )

    if (
        not isinstance(
            stock_features,
            list,
        )
        or
        len(
            stock_features
        )
        !=
        STOCK_FEATURE_COUNT
    ):
        raise RuntimeError(
            "Step-14 does not freeze exactly 47 stock predictors."
        )

    if (
        not isinstance(
            market_features,
            list,
        )
        or
        len(
            market_features
        )
        !=
        MARKET_FEATURE_COUNT
    ):
        raise RuntimeError(
            "Step-14 does not freeze exactly 16 market predictors."
        )

    if all_features != stock_features + market_features:
        raise RuntimeError(
            "Step-14 feature ordering is not 47 stock + 16 market."
        )

    if len(
        set(
            all_features
        )
    ) != TOTAL_FEATURE_COUNT:
        raise RuntimeError(
            "Duplicate predictor names detected."
        )

    if (
        step08.get(
            "ordered_63_features"
        )
        !=
        all_features
    ):
        raise RuntimeError(
            "Step-08/14 predictor-order mismatch."
        )

    if (
        step08.get(
            "feature_interface_sha256"
        )
        !=
        interface_sha
    ):
        raise RuntimeError(
            "Step-08/14 feature-interface SHA mismatch."
        )

    for step in (
        9,
        10,
        11,
        12,
        13,
    ):
        if (
            manifests[
                step
            ].get(
                "ordered_63_features"
            )
            !=
            all_features
        ):
            raise RuntimeError(
                f"Step-{step:02d}/14 predictor-order mismatch."
            )

        if (
            manifests[
                step
            ].get(
                "feature_interface_sha256"
            )
            !=
            interface_sha
        ):
            raise RuntimeError(
                f"Step-{step:02d}/14 feature-interface SHA mismatch."
            )

    for step in (
        10,
        11,
        12,
        13,
        14,
    ):
        if (
            manifests[
                step
            ].get(
                "split_design_sha256"
            )
            !=
            split_design_sha
        ):
            raise RuntimeError(
                f"Split-design SHA drift at Step {step:02d}."
            )

    # Target interface.
    if step14.get(
        "direction_target_order"
    ) != DIRECTION_COLUMNS:
        raise RuntimeError(
            "Step-14 directional-target order differs from frozen contract."
        )

    if step14.get(
        "auxiliary_target_order"
    ) != AUXILIARY_TARGET_COLUMNS:
        raise RuntimeError(
            "Step-14 auxiliary-target order differs from frozen contract."
        )

    prohibited = set(
        DIRECTION_COLUMNS
        +
        AUXILIARY_TARGET_COLUMNS
        +
        TARGET_METADATA_COLUMNS
    )

    overlap = sorted(
        prohibited
        &
        set(
            all_features
        )
    )

    if overlap:
        raise RuntimeError(
            "Target leakage detected in predictor interface:\n"
            f"{overlap}"
        )

    # Modality-order safety.
    if any(
        str(
            name
        ).startswith(
            "nifty_"
        )
        for name in stock_features
    ):
        raise RuntimeError(
            "A NIFTY predictor appears inside the first 47 stock features."
        )

    if not all(
        str(
            name
        ).startswith(
            "nifty_"
        )
        for name in market_features
    ):
        raise RuntimeError(
            "The final 16 predictors are not all NIFTY context features."
        )

    stock_id_map = step14.get(
        "stock_id_map"
    )

    expected_stock_id_map = {
        ticker: int(
            index
        )
        for index, ticker in enumerate(
            cfg.STOCK_TICKERS
        )
    }

    if stock_id_map != expected_stock_id_map:
        raise RuntimeError(
            "Step-14 stock-ID map differs from Step-00 stock ordering."
        )

    if step13.get(
        "stock_id_map"
    ) != stock_id_map:
        raise RuntimeError(
            "Step-13/14 stock-ID map mismatch."
        )

    # Pipeline state.
    state14 = step14.get(
        "pipeline_state",
        {},
    )

    required_true = (
        "rolling_fold_sequence_arrays_materialized",
        "rolling_fold_auxiliary_scalers_fitted",
        "final_development_refit_arrays_created",
        "final_development_auxiliary_scaler_fitted",
    )

    for key in required_true:
        if state14.get(
            key
        ) is not True:
            raise RuntimeError(
                f"Step-14 pipeline state is not complete: {key}"
            )

    required_false = (
        "feature_selection_applied",
        "models_fitted",
        "probability_threshold_selection_performed",
    )

    for key in required_false:
        if state14.get(
            key
        ) is not False:
            raise RuntimeError(
                f"Step-14 pipeline state violation: {key}"
            )

    protection14 = step14.get(
        "test_protection",
        {},
    )

    if protection14.get(
        "step10_seal_status"
    ) != "SEALED_NOT_OPENED":
        raise RuntimeError(
            "Step-14 does not preserve the Step-10 test seal."
        )

    if protection14.get(
        "global_final_test_state"
    ) != "LOCKED_NOT_OPENED":
        raise RuntimeError(
            "Step-14 does not preserve the global final-test lock."
        )

    for key in (
        "sealed_test_parquet_opened_or_hashed_by_step14",
        "final_test_arrays_created",
        "test_targets_scaled",
        "test_performance_computed",
    ):
        if protection14.get(
            key
        ) is not False:
            raise RuntimeError(
                f"Step-14 test-protection violation: {key}"
            )

    if not STEP10_TEST_SEAL.exists():
        raise RuntimeError(
            "Step-10 final-test seal metadata is missing."
        )

    seal = json.loads(
        STEP10_TEST_SEAL.read_text(
            encoding="utf-8"
        )
    )

    if seal.get(
        "status"
    ) != "SEALED_NOT_OPENED":
        raise RuntimeError(
            "Step-10 test seal is not SEALED_NOT_OPENED."
        )

    if (
        seal.get(
            "global_final_test_lock_state"
        )
        !=
        "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError(
            "Step-10 seal does not preserve LOCKED_NOT_OPENED."
        )

    return {
        "manifest_sha": shas,
        "manifests": manifests,
        "environment": environment,
        "all_features": list(
            all_features
        ),
        "stock_features": list(
            stock_features
        ),
        "market_features": list(
            market_features
        ),
        "feature_interface_sha256": interface_sha,
        "split_design_sha256": split_design_sha,
        "stock_id_map": stock_id_map,
        "test_seal_status": seal[
            "status"
        ],
        "sealed_test_sha256_from_metadata_only": (
            seal.get(
                "sha256"
            )
        ),
    }


# =============================================================================
# 7. CANONICAL INPUT PATHS
# =============================================================================

def step12_fold_path(
    fold: int,
    partition: str,
) -> Path:
    return (
        STEP12_ROOT
        /
        f"fold_{fold}"
        /
        f"12_fold_{fold}_{partition}_scaled_regimes.parquet"
    )


def step13_fold_index_path(
    fold: int,
    partition: str,
) -> Path:
    return (
        STEP13_ROOT
        /
        f"fold_{fold}"
        /
        f"13_fold_{fold}_{partition}_sequence_index.parquet"
    )


def step14_fold_array_path(
    fold: int,
    partition: str,
    array_name: str,
) -> Path:
    return (
        STEP14_ROOT
        /
        f"fold_{fold}"
        /
        f"14_fold_{fold}_{partition}_{array_name}.npy"
    )


def step14_fold_metadata_path(
    fold: int,
    partition: str,
) -> Path:
    return (
        STEP14_ROOT
        /
        f"fold_{fold}"
        /
        f"14_fold_{fold}_{partition}_metadata.parquet"
    )


def step14_fold_aux_scaler_path(
    fold: int,
) -> Path:
    return (
        STEP14_ROOT
        /
        f"fold_{fold}"
        /
        f"14_fold_{fold}_auxiliary_target_minmax_scaler.joblib"
    )


def step14_final_array_path(
    array_name: str,
) -> Path:
    return (
        STEP14_FINAL_DEVELOPMENT_DIR
        /
        f"14_final_development_{array_name}.npy"
    )


def step14_final_metadata_path() -> Path:
    return (
        STEP14_FINAL_DEVELOPMENT_DIR
        /
        "14_final_development_metadata.parquet"
    )


def step14_final_aux_scaler_path() -> Path:
    return (
        STEP14_FINAL_DEVELOPMENT_DIR
        /
        "14_final_development_auxiliary_target_minmax_scaler.joblib"
    )


# =============================================================================
# 8. OUTPUT IMMUTABILITY
# =============================================================================

def canonical_outputs() -> tuple[Path, ...]:
    return (
        INTEGRITY_SUMMARY_CSV,
        HASH_VERIFICATION_CSV,
        SOURCE_ALIGNMENT_CSV,
        LABEL_DISTRIBUTION_CSV,
        AUXILIARY_AUDIT_CSV,
        MODEL_INPUT_CONTRACT_CSV,
        SOURCE_HASH_CSV,
        FREEZE_JSON,
        MANIFEST_JSON,
        MANIFEST_SHA,
        HASH_REGISTER_CSV,
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
            "Step-15 canonical output(s) already exist. "
            "Refusing to overwrite.\n"
            +
            "\n".join(
                str(
                    path
                )
                for path in collisions
            )
            +
            "\nUse --verify-only if Step 15 already passed."
        )


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
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
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
                str(
                    path
                ),
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
# 9. STEP-14 ARTIFACT HASH AUDIT
# =============================================================================

def expected_step14_model_filenames() -> list[str]:
    names = []

    for fold in FOLDS:
        names.append(
            step14_fold_aux_scaler_path(
                fold
            ).name
        )

        for partition in PARTITIONS:
            for array_name in MODEL_ARRAY_NAMES:
                names.append(
                    step14_fold_array_path(
                        fold,
                        partition,
                        array_name,
                    ).name
                )

            names.append(
                step14_fold_metadata_path(
                    fold,
                    partition,
                ).name
            )

    names.append(
        step14_final_aux_scaler_path().name
    )

    for array_name in MODEL_ARRAY_NAMES:
        names.append(
            step14_final_array_path(
                array_name
            ).name
        )

    names.append(
        step14_final_metadata_path().name
    )

    if len(
        names
    ) != EXPECTED_STEP14_MODEL_ARTIFACTS:
        raise RuntimeError(
            "Programming error in expected Step-14 model-artifact inventory."
        )

    if len(
        set(
            names
        )
    ) != len(
        names
    ):
        raise RuntimeError(
            "Duplicate expected Step-14 model filenames."
        )

    return names


def has_forbidden_test_token(
    path: Path,
) -> bool:
    # The Step-10 seal JSON is metadata only and is explicitly permitted.
    # Step 15 may read/hash this seal document to verify that the held-out
    # test parquet remains sealed; it must not treat the seal's filename as
    # evidence of test-data access.
    try:
        if path.resolve() == STEP10_TEST_SEAL.resolve():
            return False
    except FileNotFoundError:
        # Missing files are handled by the caller's existence checks.
        pass

    lower = str(
        path
    ).lower()

    if path.name == FORBIDDEN_TEST_FILENAME:
        return True

    return any(
        token in lower
        for token in FORBIDDEN_TEST_TOKENS
    )


def verify_step14_artifacts(
    step14_manifest: Dict[str, Any],
) -> pd.DataFrame:
    artifacts = step14_manifest.get(
        "artifacts",
        {},
    )

    if not isinstance(
        artifacts,
        dict,
    ):
        raise RuntimeError(
            "Step-14 manifest artifacts are not a dictionary."
        )

    if len(
        artifacts
    ) != EXPECTED_STEP14_TOTAL_MANIFEST_ARTIFACTS:
        raise RuntimeError(
            "Unexpected Step-14 manifest artifact count.\n"
            f"Expected: {EXPECTED_STEP14_TOTAL_MANIFEST_ARTIFACTS}\n"
            f"Observed: {len(artifacts)}"
        )

    expected_model_names = set(
        expected_step14_model_filenames()
    )

    missing_model_names = (
        expected_model_names
        -
        set(
            artifacts.keys()
        )
    )

    if missing_model_names:
        raise RuntimeError(
            "Step-14 manifest is missing model artifacts:\n"
            f"{sorted(missing_model_names)}"
        )

    rows = []

    for name, record in sorted(
        artifacts.items()
    ):
        path = Path(
            record[
                "path"
            ]
        )

        if has_forbidden_test_token(
            path
        ):
            raise RuntimeError(
                f"Unexpected final-test artifact in Step-14 manifest: {path}"
            )

        if not path.exists():
            raise RuntimeError(
                f"Missing Step-14 artifact: {path}"
            )

        expected = record.get(
            "sha256"
        )

        observed = sha256_file(
            path
        )

        passed = bool(
            observed
            ==
            expected
        )

        if not passed:
            raise RuntimeError(
                f"Step-14 artifact SHA mismatch: {path}"
            )

        rows.append(
            {
                "artifact_name": name,
                "is_model_artifact": (
                    name in expected_model_names
                ),
                "path": str(
                    path
                ),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "size_bytes": int(
                    path.stat().st_size
                ),
                "passed": passed,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if int(
        result[
            "is_model_artifact"
        ].sum()
    ) != EXPECTED_STEP14_MODEL_ARTIFACTS:
        raise RuntimeError(
            "Step-14 model-artifact count mismatch."
        )

    if not result[
        "passed"
    ].all():
        raise RuntimeError(
            "One or more Step-14 artifact hashes failed."
        )

    return result


# =============================================================================
# 10. LOADERS
# =============================================================================

def normalize_dates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    for column in (
        "Date",
        "reference_date",
        "sequence_start_date",
        "sequence_end_date",
        "target_date_1d",
        "target_date_3d",
        "target_date_5d",
    ):
        if column in result.columns:
            result[
                column
            ] = (
                pd.to_datetime(
                    result[
                        column
                    ],
                    errors="coerce",
                )
                .dt.normalize()
            )

    return result


def load_step12_panel(
    path: Path,
    all_features: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    record = find_artifact(
        CURRENT_UPSTREAM[
            "manifests"
        ][
            12
        ],
        path.name,
    )

    if (
        Path(
            record[
                "path"
            ]
        ).resolve()
        !=
        path.resolve()
    ):
        raise RuntimeError(
            f"{label}: Step-12 canonical path mismatch."
        )

    if sha256_file(
        path
    ) != record[
        "sha256"
    ]:
        raise RuntimeError(
            f"{label}: Step-12 panel SHA mismatch."
        )

    frame = pd.read_parquet(
        path,
        engine="pyarrow",
    )

    frame = normalize_dates(
        frame
    )

    required = {
        "Ticker",
        "Date",
        "market_session_id",
        "continuity_segment_id",
        "regime_id",
        *all_features,
    }

    missing = (
        required
        -
        set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{label}: missing Step-12 columns:\n"
            f"{sorted(missing)}"
        )

    if frame[
        [
            "Ticker",
            "Date",
        ]
    ].duplicated().any():
        raise RuntimeError(
            f"{label}: duplicate Step-12 ticker-date keys."
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


def load_step13_index(
    path: Path,
    *,
    label: str,
) -> pd.DataFrame:
    record = find_artifact(
        CURRENT_UPSTREAM[
            "manifests"
        ][
            13
        ],
        path.name,
    )

    if (
        Path(
            record[
                "path"
            ]
        ).resolve()
        !=
        path.resolve()
    ):
        raise RuntimeError(
            f"{label}: Step-13 canonical path mismatch."
        )

    if sha256_file(
        path
    ) != record[
        "sha256"
    ]:
        raise RuntimeError(
            f"{label}: Step-13 index SHA mismatch."
        )

    frame = pd.read_parquet(
        path,
        engine="pyarrow",
    )

    return normalize_dates(
        frame
    )


def load_metadata(
    path: Path,
    *,
    expected_hash: str,
) -> pd.DataFrame:
    if sha256_file(
        path
    ) != expected_hash:
        raise RuntimeError(
            f"Metadata SHA mismatch: {path}"
        )

    frame = pd.read_parquet(
        path,
        engine="pyarrow",
    )

    return normalize_dates(
        frame
    )


# =============================================================================
# 11. EXACT ARRAY HELPERS
# =============================================================================

def chunked_exact_equal(
    first,
    second,
    chunk_size: int = ARRAY_CHUNK_SIZE,
) -> bool:
    if first.shape != second.shape:
        return False

    n = int(
        first.shape[
            0
        ]
    )

    for start in range(
        0,
        n,
        chunk_size,
    ):
        stop = min(
            start
            +
            chunk_size,
            n,
        )

        if not np.array_equal(
            np.asarray(
                first[
                    start:
                    stop
                ]
            ),
            np.asarray(
                second[
                    start:
                    stop
                ]
            ),
            equal_nan=True,
        ):
            return False

    return True


def chunked_all_finite(
    array,
    chunk_size: int = ARRAY_CHUNK_SIZE,
) -> bool:
    n = int(
        array.shape[
            0
        ]
    )

    for start in range(
        0,
        n,
        chunk_size,
    ):
        stop = min(
            start
            +
            chunk_size,
            n,
        )

        if not np.isfinite(
            np.asarray(
                array[
                    start:
                    stop
                ]
            )
        ).all():
            return False

    return True


# =============================================================================
# 12. STEP-12 FEATURE HISTORY
# =============================================================================

def build_source_history(
    train_panel: pd.DataFrame,
    validation_panel: Optional[pd.DataFrame],
    *,
    partition: str,
    all_features: list[str],
) -> tuple[
    pd.DataFrame,
    Dict[
        tuple[str, int],
        Dict[str, Any],
    ],
]:
    if partition in (
        "train",
        "final_development",
    ):
        source = (
            train_panel
            .copy()
        )

    elif partition == "validation":
        if validation_panel is None:
            raise RuntimeError(
                "Validation source history requires validation panel."
            )

        source = (
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

    else:
        raise RuntimeError(
            f"Unknown partition: {partition}"
        )

    if source[
        [
            "Ticker",
            "Date",
        ]
    ].duplicated().any():
        raise RuntimeError(
            f"{partition}: duplicate source ticker-date keys."
        )

    history = {}

    for (
        ticker,
        segment_id,
    ), group in source.groupby(
        [
            "Ticker",
            "continuity_segment_id",
        ],
        sort=True,
    ):
        ordered = (
            group
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        positions = {
            pd.Timestamp(
                date
            ): int(
                position
            )
            for position, date in enumerate(
                ordered[
                    "Date"
                ]
            )
        }

        history[
            (
                ticker,
                int(
                    segment_id
                ),
            )
        ] = {
            "frame": ordered,
            "positions": positions,
            "matrix": (
                ordered[
                    all_features
                ]
                .to_numpy(
                    dtype=np.float32
                )
            ),
            "session_ids": (
                ordered[
                    "market_session_id"
                ]
                .to_numpy(
                    dtype=np.int64
                )
            ),
        }

    return (
        source,
        history,
    )


def source_matrix_for_dates(
    source: pd.DataFrame,
    metadata: pd.DataFrame,
    date_column: str,
    all_features: list[str],
) -> np.ndarray:
    lookup = (
        source
        .set_index(
            [
                "Ticker",
                "Date",
            ]
        )[
            all_features
        ]
    )

    requested_index = pd.MultiIndex.from_arrays(
        [
            metadata[
                "ticker"
            ].to_numpy(),
            metadata[
                date_column
            ].to_numpy(),
        ],
        names=[
            "Ticker",
            "Date",
        ],
    )

    expected = (
        lookup
        .reindex(
            requested_index
        )
    )

    if expected.isna().any().any():
        raise RuntimeError(
            f"Unable to reconstruct source rows for {date_column}."
        )

    return (
        expected
        .to_numpy(
            dtype=np.float32
        )
    )


def audit_full_sequence_samples(
    X_full,
    metadata: pd.DataFrame,
    history: Dict[
        tuple[str, int],
        Dict[str, Any],
    ],
) -> tuple[
    int,
    float,
]:
    n = int(
        len(
            metadata
        )
    )

    base_indices = np.linspace(
        0,
        n
        -
        1,
        num=min(
            FULL_SEQUENCE_SAMPLE_SIZE,
            n,
        ),
        dtype=int,
    )

    sample_indices = set(
        int(
            value
        )
        for value in base_indices
    )

    crossing = (
        metadata.index[
            metadata[
                "lookback_crosses_partition_start"
            ]
        ]
        .tolist()
    )

    if crossing:
        sample_indices.add(
            int(
                crossing[
                    0
                ]
            )
        )

        sample_indices.add(
            int(
                crossing[
                    -1
                ]
            )
        )

        sample_indices.add(
            int(
                crossing[
                    len(
                        crossing
                    )
                    //
                    2
                ]
            )
        )

    maximum_difference = 0.0

    for row_index in sorted(
        sample_indices
    ):
        row = metadata.iloc[
            row_index
        ]

        key = (
            row[
                "ticker"
            ],
            int(
                row[
                    "continuity_segment_id"
                ]
            ),
        )

        if key not in history:
            raise RuntimeError(
                f"Missing source history for {key}."
            )

        source = history[
            key
        ]

        reference_date = pd.Timestamp(
            row[
                "reference_date"
            ]
        )

        if reference_date not in source[
            "positions"
        ]:
            raise RuntimeError(
                "Reference date absent from source history."
            )

        endpoint_position = int(
            source[
                "positions"
            ][
                reference_date
            ]
        )

        start_position = (
            endpoint_position
            -
            LOOKBACK
            +
            1
        )

        if start_position < 0:
            raise RuntimeError(
                "Negative sampled sequence start."
            )

        expected = (
            source[
                "matrix"
            ][
                start_position:
                endpoint_position
                +
                1
            ]
        )

        observed = np.asarray(
            X_full[
                row_index
            ]
        )

        if expected.shape != (
            LOOKBACK,
            TOTAL_FEATURE_COUNT,
        ):
            raise RuntimeError(
                "Sample expected sequence shape mismatch."
            )

        difference = np.abs(
            expected
            -
            observed
        )

        maximum_difference = max(
            maximum_difference,
            float(
                difference.max()
            ),
        )

        if not np.array_equal(
            expected,
            observed,
            equal_nan=True,
        ):
            raise RuntimeError(
                "\nFULL-SEQUENCE SAMPLE ALIGNMENT FAILED\n"
                f"Array row : {row_index}\n"
                f"Ticker    : {row['ticker']}\n"
                f"Reference : {reference_date}\n"
                f"Max diff  : {difference.max():.12e}"
            )

    return (
        len(
            sample_indices
        ),
        maximum_difference,
    )


# =============================================================================
# 13. EXPECTED MODEL ARTIFACT HASH LOOKUP
# =============================================================================

def step14_hash_for(
    step14_manifest: Dict[str, Any],
    path: Path,
) -> str:
    record = find_artifact(
        step14_manifest,
        path.name,
    )

    if (
        Path(
            record[
                "path"
            ]
        ).resolve()
        !=
        path.resolve()
    ):
        raise RuntimeError(
            f"Step-14 canonical path mismatch: {path.name}"
        )

    return str(
        record[
            "sha256"
        ]
    )


# =============================================================================
# 14. AUDIT ONE MATERIALIZED PARTITION
# =============================================================================

def audit_partition(
    *,
    scope: str,
    fold: Optional[int],
    partition: str,
    expected_n: int,
    metadata_path: Path,
    array_paths: Dict[str, Path],
    aux_scaler_path: Path,
    train_panel: pd.DataFrame,
    validation_panel: Optional[pd.DataFrame],
    all_features: list[str],
    stock_id_map: Dict[str, int],
    step14_manifest: Dict[str, Any],
) -> tuple[
    Dict[str, Any],
    Dict[str, Any],
    list[dict],
]:
    metadata = load_metadata(
        metadata_path,
        expected_hash=step14_hash_for(
            step14_manifest,
            metadata_path,
        ),
    )

    if len(
        metadata
    ) != expected_n:
        raise RuntimeError(
            f"{scope}: metadata count mismatch. "
            f"Expected {expected_n:,}, observed {len(metadata):,}."
        )

    if not np.array_equal(
        metadata[
            "array_row_index"
        ].to_numpy(
            dtype=np.int64
        ),
        np.arange(
            expected_n,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            f"{scope}: array_row_index is not contiguous."
        )

    arrays = {}

    for array_name, path in array_paths.items():
        expected_hash = step14_hash_for(
            step14_manifest,
            path,
        )

        if sha256_file(
            path
        ) != expected_hash:
            raise RuntimeError(
                f"{scope}: array SHA mismatch: {path.name}"
            )

        arrays[
            array_name
        ] = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

    scaler_hash = step14_hash_for(
        step14_manifest,
        aux_scaler_path,
    )

    if sha256_file(
        aux_scaler_path
    ) != scaler_hash:
        raise RuntimeError(
            f"{scope}: auxiliary scaler SHA mismatch."
        )

    aux_scaler = joblib.load(
        aux_scaler_path
    )

    if int(
        aux_scaler.n_features_in_
    ) != len(
        AUXILIARY_TARGET_COLUMNS
    ):
        raise RuntimeError(
            f"{scope}: auxiliary scaler does not contain 10 targets."
        )

    if tuple(
        aux_scaler.feature_range
    ) != (
        0,
        1,
    ):
        raise RuntimeError(
            f"{scope}: auxiliary scaler feature range changed."
        )

    if bool(
        aux_scaler.clip
    ):
        raise RuntimeError(
            f"{scope}: auxiliary scaler unexpectedly clips."
        )

    expected_shapes = {
        "X_full": (
            expected_n,
            LOOKBACK,
            TOTAL_FEATURE_COUNT,
        ),
        "X_stock": (
            expected_n,
            LOOKBACK,
            STOCK_FEATURE_COUNT,
        ),
        "X_market": (
            expected_n,
            LOOKBACK,
            MARKET_FEATURE_COUNT,
        ),
        "y_direction": (
            expected_n,
            len(
                DIRECTION_COLUMNS
            ),
        ),
        "y_aux_raw": (
            expected_n,
            len(
                AUXILIARY_TARGET_COLUMNS
            ),
        ),
        "y_aux_scaled": (
            expected_n,
            len(
                AUXILIARY_TARGET_COLUMNS
            ),
        ),
        "stock_id": (
            expected_n,
        ),
        "regime_id": (
            expected_n,
        ),
    }

    expected_dtypes = {
        "X_full": np.dtype(
            np.float32
        ),
        "X_stock": np.dtype(
            np.float32
        ),
        "X_market": np.dtype(
            np.float32
        ),
        "y_direction": np.dtype(
            np.int64
        ),
        "y_aux_raw": np.dtype(
            np.float32
        ),
        "y_aux_scaled": np.dtype(
            np.float32
        ),
        "stock_id": np.dtype(
            np.int64
        ),
        "regime_id": np.dtype(
            np.int64
        ),
    }

    for name in MODEL_ARRAY_NAMES:
        if arrays[
            name
        ].shape != expected_shapes[
            name
        ]:
            raise RuntimeError(
                f"{scope}: {name} shape mismatch. "
                f"Expected {expected_shapes[name]}, "
                f"observed {arrays[name].shape}."
            )

        if arrays[
            name
        ].dtype != expected_dtypes[
            name
        ]:
            raise RuntimeError(
                f"{scope}: {name} dtype mismatch."
            )

    X_full = arrays[
        "X_full"
    ]

    X_stock = arrays[
        "X_stock"
    ]

    X_market = arrays[
        "X_market"
    ]

    y_direction = arrays[
        "y_direction"
    ]

    y_aux_raw = arrays[
        "y_aux_raw"
    ]

    y_aux_scaled = arrays[
        "y_aux_scaled"
    ]

    stock_id = arrays[
        "stock_id"
    ]

    regime_id = arrays[
        "regime_id"
    ]

    stock_slice_exact = chunked_exact_equal(
        X_stock,
        X_full[
            :,
            :,
            :STOCK_FEATURE_COUNT
        ],
    )

    market_slice_exact = chunked_exact_equal(
        X_market,
        X_full[
            :,
            :,
            STOCK_FEATURE_COUNT:
        ],
    )

    if not stock_slice_exact:
        raise RuntimeError(
            f"{scope}: X_stock slice mismatch."
        )

    if not market_slice_exact:
        raise RuntimeError(
            f"{scope}: X_market slice mismatch."
        )

    for name in (
        "X_full",
        "X_stock",
        "X_market",
        "y_aux_raw",
        "y_aux_scaled",
    ):
        if not chunked_all_finite(
            arrays[
                name
            ]
        ):
            raise RuntimeError(
                f"{scope}: non-finite values in {name}."
            )

    expected_direction = (
        metadata[
            DIRECTION_COLUMNS
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        np.asarray(
            y_direction
        ),
        expected_direction,
    ):
        raise RuntimeError(
            f"{scope}: direction labels differ from metadata."
        )

    if not np.isin(
        y_direction,
        [
            0,
            1,
        ],
    ).all():
        raise RuntimeError(
            f"{scope}: directional labels are not binary."
        )

    expected_aux_raw = (
        metadata[
            AUXILIARY_TARGET_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    if not np.array_equal(
        np.asarray(
            y_aux_raw
        ),
        expected_aux_raw,
        equal_nan=True,
    ):
        raise RuntimeError(
            f"{scope}: raw auxiliary array differs from metadata."
        )

    expected_stock_id = (
        metadata[
            "ticker"
        ]
        .map(
            stock_id_map
        )
    )

    if expected_stock_id.isna().any():
        raise RuntimeError(
            f"{scope}: ticker absent from frozen stock-ID map."
        )

    expected_stock_id_array = (
        expected_stock_id
        .to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        np.asarray(
            stock_id
        ),
        expected_stock_id_array,
    ):
        raise RuntimeError(
            f"{scope}: stock IDs differ from metadata."
        )

    expected_regime_id = (
        metadata[
            "regime_id"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        np.asarray(
            regime_id
        ),
        expected_regime_id,
    ):
        raise RuntimeError(
            f"{scope}: regime IDs differ from metadata."
        )

    if not np.isin(
        regime_id,
        [
            0,
            1,
            2,
        ],
    ).all():
        raise RuntimeError(
            f"{scope}: invalid regime ID."
        )

    aux_raw_float64 = (
        metadata[
            AUXILIARY_TARGET_COLUMNS
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    expected_aux_scaled = (
        aux_scaler
        .transform(
            aux_raw_float64
        )
        .astype(
            np.float32
        )
    )

    auxiliary_scaler_exact = bool(
        np.array_equal(
            np.asarray(
                y_aux_scaled
            ),
            expected_aux_scaled,
            equal_nan=True,
        )
    )

    if not auxiliary_scaler_exact:
        difference = float(
            np.max(
                np.abs(
                    np.asarray(
                        y_aux_scaled
                    )
                    -
                    expected_aux_scaled
                )
            )
        )

        raise RuntimeError(
            "\nAUXILIARY SCALER REPRODUCTION FAILED\n"
            f"Scope    : {scope}\n"
            f"Max diff : {difference:.12e}"
        )

    stock_counts = (
        metadata[
            "ticker"
        ]
        .value_counts()
    )

    if len(
        stock_counts
    ) != EXPECTED_STOCK_COUNT:
        raise RuntimeError(
            f"{scope}: metadata does not contain 20 stocks."
        )

    if stock_counts.nunique() != 1:
        raise RuntimeError(
            f"{scope}: sequence counts are not balanced by stock."
        )

    expected_per_stock = (
        expected_n
        //
        EXPECTED_STOCK_COUNT
    )

    if int(
        stock_counts.iloc[
            0
        ]
    ) != expected_per_stock:
        raise RuntimeError(
            f"{scope}: unexpected sequences per stock."
        )

    cross_boundary_count = int(
        metadata[
            "lookback_crosses_partition_start"
        ].sum()
    )

    if partition == "validation":
        expected_cross = (
            EXPECTED_VALIDATION_CROSS_BOUNDARY
        )

    else:
        expected_cross = 0

    if cross_boundary_count != expected_cross:
        raise RuntimeError(
            f"{scope}: cross-boundary count mismatch. "
            f"Expected {expected_cross:,}, "
            f"observed {cross_boundary_count:,}."
        )

    for horizon in HORIZONS:
        if (
            metadata[
                f"target_date_{horizon}d"
            ]
            <=
            metadata[
                "reference_date"
            ]
        ).any():
            raise RuntimeError(
                f"{scope}: invalid {horizon}d target ordering."
            )

    source, history = build_source_history(
        train_panel,
        validation_panel,
        partition=partition,
        all_features=all_features,
    )

    expected_start_matrix = source_matrix_for_dates(
        source,
        metadata,
        "sequence_start_date",
        all_features,
    )

    expected_endpoint_matrix = source_matrix_for_dates(
        source,
        metadata,
        "reference_date",
        all_features,
    )

    start_exact = chunked_exact_equal(
        X_full[
            :,
            0,
            :
        ],
        expected_start_matrix,
    )

    endpoint_exact = chunked_exact_equal(
        X_full[
            :,
            -1,
            :
        ],
        expected_endpoint_matrix,
    )

    if not start_exact:
        raise RuntimeError(
            f"{scope}: sequence-start source alignment failed."
        )

    if not endpoint_exact:
        raise RuntimeError(
            f"{scope}: sequence-endpoint source alignment failed."
        )

    sample_count, sample_max_difference = (
        audit_full_sequence_samples(
            X_full,
            metadata,
            history,
        )
    )

    label_rows = []

    for target_index, horizon in enumerate(
        HORIZONS
    ):
        labels = np.asarray(
            y_direction[
                :,
                target_index
            ]
        )

        up = int(
            (
                labels
                ==
                1
            ).sum()
        )

        non_up = int(
            (
                labels
                ==
                0
            ).sum()
        )

        label_rows.append(
            {
                "scope": scope,
                "fold": (
                    np.nan
                    if fold is None
                    else int(
                        fold
                    )
                ),
                "partition": partition,
                "horizon_sessions": int(
                    horizon
                ),
                "rows": int(
                    expected_n
                ),
                "up_rows": up,
                "non_up_rows": non_up,
                "up_ratio": float(
                    up
                    /
                    expected_n
                ),
            }
        )

    scaled_aux_array = np.asarray(
        y_aux_scaled
    )

    outside_aux_count = int(
        (
            scaled_aux_array
            <
            0.0
        ).sum()
        +
        (
            scaled_aux_array
            >
            1.0
        ).sum()
    )

    if partition in (
        "train",
        "final_development",
    ):
        if outside_aux_count != 0:
            raise RuntimeError(
                f"{scope}: scaler-fit targets unexpectedly fall outside [0,1]."
            )

    summary = {
        "scope": scope,
        "fold": (
            np.nan
            if fold is None
            else int(
                fold
            )
        ),
        "partition": partition,
        "sequences": int(
            expected_n
        ),
        "sequences_per_stock": int(
            expected_per_stock
        ),
        "first_reference_date": (
            metadata[
                "reference_date"
            ].min()
        ),
        "last_reference_date": (
            metadata[
                "reference_date"
            ].max()
        ),
        "cross_boundary_endpoints": (
            cross_boundary_count
        ),
        "X_full_shape": str(
            X_full.shape
        ),
        "X_stock_shape": str(
            X_stock.shape
        ),
        "X_market_shape": str(
            X_market.shape
        ),
        "stock_slice_exact": (
            stock_slice_exact
        ),
        "market_slice_exact": (
            market_slice_exact
        ),
        "metadata_direction_exact": True,
        "metadata_aux_raw_exact": True,
        "metadata_stock_id_exact": True,
        "metadata_regime_id_exact": True,
        "auxiliary_scaler_reproduction_exact": (
            auxiliary_scaler_exact
        ),
        "sequence_start_alignment_exact": (
            start_exact
        ),
        "sequence_endpoint_alignment_exact": (
            endpoint_exact
        ),
        "full_sequence_samples_checked": int(
            sample_count
        ),
        "sampled_max_absolute_difference": float(
            sample_max_difference
        ),
        "auxiliary_values_outside_0_1": (
            outside_aux_count
        ),
        "passed": True,
    }

    auxiliary_summary = {
        "scope": scope,
        "fold": (
            np.nan
            if fold is None
            else int(
                fold
            )
        ),
        "partition": partition,
        "rows": int(
            expected_n
        ),
        "auxiliary_values": int(
            expected_n
            *
            len(
                AUXILIARY_TARGET_COLUMNS
            )
        ),
        "outside_0_1": int(
            outside_aux_count
        ),
        "transform_reproduction_exact": (
            auxiliary_scaler_exact
        ),
        "clip": bool(
            aux_scaler.clip
        ),
    }

    arrays.clear()

    del source
    del history
    del expected_start_matrix
    del expected_endpoint_matrix
    del expected_aux_scaled
    del aux_scaler

    gc.collect()

    return (
        summary,
        auxiliary_summary,
        label_rows,
    )


# =============================================================================
# 15. STEP-13 HORIZON-SPECIFIC MODEL-INPUT CONTRACT
# =============================================================================

def build_model_input_contract() -> pd.DataFrame:
    rows = []

    for fold in FOLDS:
        for partition in PARTITIONS:
            index = load_step13_index(
                step13_fold_index_path(
                    fold,
                    partition,
                ),
                label=(
                    f"fold_{fold}_{partition}_step13"
                ),
            )

            common_count = int(
                index[
                    "sequence_eligible_common_multitask"
                ].sum()
            )

            expected_common = (
                EXPECTED_COMMON_COUNTS[
                    fold
                ][
                    partition
                ]
            )

            if common_count != expected_common:
                raise RuntimeError(
                    f"Fold {fold} {partition}: Step-13 common count drift."
                )

            for horizon in HORIZONS:
                count = int(
                    index[
                        f"sequence_eligible_{horizon}d"
                    ].sum()
                )

                rows.append(
                    {
                        "scope": "rolling_fold",
                        "fold": fold,
                        "partition": partition,
                        "dataset_contract": (
                            f"horizon_specific_{horizon}d"
                        ),
                        "horizon_sessions": horizon,
                        "eligible_sequences": count,
                        "materialized_in_step14": (
                            horizon
                            ==
                            5
                        ),
                        "usage": (
                            "available_for_later_horizon_specific_modelling"
                        ),
                    }
                )

            rows.append(
                {
                    "scope": "rolling_fold",
                    "fold": fold,
                    "partition": partition,
                    "dataset_contract": (
                        "common_multitask_neural"
                    ),
                    "horizon_sessions": np.nan,
                    "eligible_sequences": common_count,
                    "materialized_in_step14": True,
                    "usage": (
                        "frozen_step14_common_multitask_arrays"
                    ),
                }
            )

    final_index = load_step13_index(
        STEP13_FINAL_DEVELOPMENT_INDEX,
        label="final_development_step13",
    )

    final_common = int(
        final_index[
            "sequence_eligible_common_multitask"
        ].sum()
    )

    if final_common != EXPECTED_FINAL_DEVELOPMENT_COMMON:
        raise RuntimeError(
            "Final-development Step-13 common count drift."
        )

    for horizon in HORIZONS:
        count = int(
            final_index[
                f"sequence_eligible_{horizon}d"
            ].sum()
        )

        rows.append(
            {
                "scope": "final_development",
                "fold": np.nan,
                "partition": "final_development",
                "dataset_contract": (
                    f"horizon_specific_{horizon}d"
                ),
                "horizon_sessions": horizon,
                "eligible_sequences": count,
                "materialized_in_step14": (
                    horizon
                    ==
                    5
                ),
                "usage": (
                    "available_for_later_horizon_specific_refit"
                ),
            }
        )

    rows.append(
        {
            "scope": "final_development",
            "fold": np.nan,
            "partition": "final_development",
            "dataset_contract": (
                "common_multitask_neural"
            ),
            "horizon_sessions": np.nan,
            "eligible_sequences": final_common,
            "materialized_in_step14": True,
            "usage": (
                "frozen_step14_final_development_common_multitask_arrays"
            ),
        }
    )

    result = pd.DataFrame(
        rows
    )

    # Frozen structural property:
    # 1d >= 3d >= 5d == common within every fold/partition.
    grouping = result.groupby(
        [
            "scope",
            "fold",
            "partition",
        ],
        dropna=False,
        sort=False,
    )

    for key, group in grouping:
        lookup = {
            row.dataset_contract: int(
                row.eligible_sequences
            )
            for row in group.itertuples(
                index=False
            )
        }

        required = {
            "horizon_specific_1d",
            "horizon_specific_3d",
            "horizon_specific_5d",
            "common_multitask_neural",
        }

        if set(
            lookup
        ) != required:
            raise RuntimeError(
                f"Model-input contract incomplete for {key}."
            )

        if not (
            lookup[
                "horizon_specific_1d"
            ]
            >=
            lookup[
                "horizon_specific_3d"
            ]
            >=
            lookup[
                "horizon_specific_5d"
            ]
            ==
            lookup[
                "common_multitask_neural"
            ]
        ):
            raise RuntimeError(
                f"Horizon/common count relation failed for {key}."
            )

    return result


# =============================================================================
# 16. SOURCE HASH REGISTER
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
            "current_step15",
            SCRIPT_FILE.resolve(),
        ),
        (
            "step10_test_seal_metadata",
            STEP10_TEST_SEAL.resolve(),
        ),
        (
            "step12_final_development",
            STEP12_FINAL_DEVELOPMENT.resolve(),
        ),
        (
            "step13_final_development_index",
            STEP13_FINAL_DEVELOPMENT_INDEX.resolve(),
        ),
        (
            "step14_final_metadata",
            step14_final_metadata_path().resolve(),
        ),
        (
            "step14_final_auxiliary_scaler",
            step14_final_aux_scaler_path().resolve(),
        ),
    ]

    for array_name in MODEL_ARRAY_NAMES:
        sources.append(
            (
                f"step14_final_{array_name}",
                step14_final_array_path(
                    array_name
                ).resolve(),
            )
        )

    for fold in FOLDS:
        for partition in PARTITIONS:
            sources.extend(
                [
                    (
                        f"step12_fold_{fold}_{partition}",
                        step12_fold_path(
                            fold,
                            partition,
                        ).resolve(),
                    ),
                    (
                        f"step13_fold_{fold}_{partition}",
                        step13_fold_index_path(
                            fold,
                            partition,
                        ).resolve(),
                    ),
                    (
                        f"step14_fold_{fold}_{partition}_metadata",
                        step14_fold_metadata_path(
                            fold,
                            partition,
                        ).resolve(),
                    ),
                ]
            )

            for array_name in MODEL_ARRAY_NAMES:
                sources.append(
                    (
                        f"step14_fold_{fold}_{partition}_{array_name}",
                        step14_fold_array_path(
                            fold,
                            partition,
                            array_name,
                        ).resolve(),
                    )
                )

        sources.append(
            (
                f"step14_fold_{fold}_auxiliary_scaler",
                step14_fold_aux_scaler_path(
                    fold
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
        if has_forbidden_test_token(
            path
        ):
            raise RuntimeError(
                f"Final-test artifact entered Step-15 source list: {path}"
            )

        if not path.exists():
            raise RuntimeError(
                f"Required Step-15 source missing: {path}"
            )

        rows.append(
            {
                "role": role,
                "file_name": path.name,
                "absolute_path": str(
                    path
                ),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(
                    path
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 17. VERIFY EXISTING STEP 15
# =============================================================================

def verify_existing() -> int:
    print("=" * 136)
    print("STEP 15 — VERIFY EXISTING PRE-MODELLING INTEGRITY FREEZE")
    print("=" * 136)

    step00 = verify_step00()

    upstream = verify_upstream(
        step00
    )

    manifest_sha = verify_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 15",
    )

    manifest = json.loads(
        MANIFEST_JSON.read_text(
            encoding="utf-8"
        )
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
        !=
        "PASS"
    ):
        print(
            "[FAIL] Stored Step-15 status is not PASS."
        )
        return 20

    if (
        manifest.get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00[
            "fingerprint"
        ]
    ):
        print(
            "[FAIL] Configuration fingerprint mismatch."
        )
        return 21

    for step in range(
        1,
        15,
    ):
        field = (
            f"step{step:02d}_manifest_sha256"
        )

        if (
            manifest.get(
                field
            )
            !=
            upstream[
                "manifest_sha"
            ][
                step
            ]
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
        upstream[
            "feature_interface_sha256"
        ]
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
        upstream[
            "split_design_sha256"
        ]
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

        if sha256_file(
            path
        ) != record[
            "sha256"
        ]:
            failures.append(
                f"hash mismatch: {name}"
            )

    if failures:
        for failure in failures:
            print(
                f"[FAIL] {failure}"
            )

        print("=" * 136)
        print("STEP 15 VERIFICATION STATUS: FAIL")
        print("=" * 136)
        return 25

    freeze = json.loads(
        FREEZE_JSON.read_text(
            encoding="utf-8"
        )
    )

    if freeze.get(
        "status"
    ) != "DEVELOPMENT_MODEL_INTERFACE_FROZEN_BEFORE_MODELLING":
        print(
            "[FAIL] Stored pre-model freeze status is invalid."
        )
        return 26

    if freeze.get(
        "modelling_started"
    ) is not False:
        print(
            "[FAIL] Stored freeze says modelling has started."
        )
        return 27

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
        return 28

    print(
        "[PASS] Configuration fingerprint : "
        f"{step00['fingerprint']}"
    )
    print(
        "[PASS] Step-14 manifest          : "
        f"{upstream['manifest_sha'][14]}"
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
        "[PASS] Step-15 manifest          : "
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
    print("STEP 15 VERIFICATION STATUS: PASS")
    print("=" * 136)

    return 0


# =============================================================================
# 18. MAIN RUN
# =============================================================================

CURRENT_UPSTREAM: Dict[str, Any] = {}


def run() -> int:
    global CURRENT_UPSTREAM

    print("=" * 136)
    print(STEP_NAME)
    print("=" * 136)

    stage = "verify"

    try:
        print(
            "[1/10] Verifying Step 00 and immutable Step 01 -> Step 14 chain..."
        )

        step00 = verify_step00()

        upstream = verify_upstream(
            step00
        )

        # Used by safe loaders to verify Step-12/13 canonical source hashes.
        CURRENT_UPSTREAM = upstream

        print(
            "       Configuration fingerprint: "
            f"{step00['fingerprint']}"
        )
        print(
            "       Step-14 manifest          : "
            f"{upstream['manifest_sha'][14]}"
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
            "[2/10] Checking immutable Step-15 output target..."
        )

        assert_fresh_target()

        cfg.create_directories()

        stage = "step14_hash_audit"

        print(
            "[3/10] Verifying every frozen Step-14 artifact SHA-256..."
        )

        step14_manifest = upstream[
            "manifests"
        ][
            14
        ]

        hash_verification = verify_step14_artifacts(
            step14_manifest
        )

        print(
            "       Step-14 manifest artifacts : "
            f"{len(hash_verification)}"
        )
        print(
            "       Model-interface artifacts  : "
            f"{int(hash_verification['is_model_artifact'].sum())}"
        )
        print(
            "       All hashes match           : "
            f"{bool(hash_verification['passed'].all())}"
        )

        stage = "rolling_fold_audit"

        print(
            "[4/10] Auditing five rolling-origin train/validation array pairs..."
        )

        all_features = upstream[
            "all_features"
        ]

        stock_id_map = upstream[
            "stock_id_map"
        ]

        integrity_rows = []
        auxiliary_rows = []
        label_rows = []

        for fold in FOLDS:
            train_panel = load_step12_panel(
                step12_fold_path(
                    fold,
                    "train",
                ),
                all_features,
                label=(
                    f"fold_{fold}_train_step12"
                ),
            )

            validation_panel = load_step12_panel(
                step12_fold_path(
                    fold,
                    "validation",
                ),
                all_features,
                label=(
                    f"fold_{fold}_validation_step12"
                ),
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
                    f"Fold {fold}: train/validation source chronology violated."
                )

            train_metadata_path = step14_fold_metadata_path(
                fold,
                "train",
            )

            validation_metadata_path = step14_fold_metadata_path(
                fold,
                "validation",
            )

            # Metadata chronology is checked after load inside audit; perform
            # an independent key-range check here too.
            train_metadata_preview = load_metadata(
                train_metadata_path,
                expected_hash=step14_hash_for(
                    step14_manifest,
                    train_metadata_path,
                ),
            )

            validation_metadata_preview = load_metadata(
                validation_metadata_path,
                expected_hash=step14_hash_for(
                    step14_manifest,
                    validation_metadata_path,
                ),
            )

            if (
                train_metadata_preview[
                    "reference_date"
                ].max()
                >=
                validation_metadata_preview[
                    "reference_date"
                ].min()
            ):
                raise RuntimeError(
                    f"Fold {fold}: train/validation endpoint chronology violated."
                )

            del train_metadata_preview
            del validation_metadata_preview

            scaler_path = step14_fold_aux_scaler_path(
                fold
            )

            for partition in PARTITIONS:
                expected_n = (
                    EXPECTED_COMMON_COUNTS[
                        fold
                    ][
                        partition
                    ]
                )

                metadata_path = step14_fold_metadata_path(
                    fold,
                    partition,
                )

                array_paths = {
                    name: step14_fold_array_path(
                        fold,
                        partition,
                        name,
                    )
                    for name in MODEL_ARRAY_NAMES
                }

                summary, auxiliary_summary, partition_labels = (
                    audit_partition(
                        scope=(
                            f"fold_{fold}_{partition}"
                        ),
                        fold=fold,
                        partition=partition,
                        expected_n=expected_n,
                        metadata_path=metadata_path,
                        array_paths=array_paths,
                        aux_scaler_path=scaler_path,
                        train_panel=train_panel,
                        validation_panel=(
                            validation_panel
                            if partition
                            ==
                            "validation"
                            else None
                        ),
                        all_features=all_features,
                        stock_id_map=stock_id_map,
                        step14_manifest=step14_manifest,
                    )
                )

                integrity_rows.append(
                    summary
                )

                auxiliary_rows.append(
                    auxiliary_summary
                )

                label_rows.extend(
                    partition_labels
                )

            train_row = integrity_rows[
                -2
            ]

            validation_row = integrity_rows[
                -1
            ]

            print(
                f"       Fold {fold}: "
                f"train n={train_row['sequences']:>6,} | "
                f"val n={validation_row['sequences']:>5,} | "
                f"val cross={validation_row['cross_boundary_endpoints']:>5,} | "
                f"sample max diff="
                f"{max(train_row['sampled_max_absolute_difference'], validation_row['sampled_max_absolute_difference']):.3e}"
            )

            del train_panel
            del validation_panel

            gc.collect()

        stage = "final_development_audit"

        print(
            "[5/10] Auditing final-development refit arrays and scaler..."
        )

        final_panel = load_step12_panel(
            STEP12_FINAL_DEVELOPMENT,
            all_features,
            label="final_development_step12",
        )

        final_summary, final_auxiliary, final_labels = (
            audit_partition(
                scope="final_development",
                fold=None,
                partition="final_development",
                expected_n=EXPECTED_FINAL_DEVELOPMENT_COMMON,
                metadata_path=step14_final_metadata_path(),
                array_paths={
                    name: step14_final_array_path(
                        name
                    )
                    for name in MODEL_ARRAY_NAMES
                },
                aux_scaler_path=step14_final_aux_scaler_path(),
                train_panel=final_panel,
                validation_panel=None,
                all_features=all_features,
                stock_id_map=stock_id_map,
                step14_manifest=step14_manifest,
            )
        )

        integrity_rows.append(
            final_summary
        )

        auxiliary_rows.append(
            final_auxiliary
        )

        label_rows.extend(
            final_labels
        )

        print(
            "       Final-development n        : "
            f"{final_summary['sequences']:,}"
        )
        print(
            "       Per stock                  : "
            f"{final_summary['sequences_per_stock']:,}"
        )
        print(
            "       Full-window sample max diff: "
            f"{final_summary['sampled_max_absolute_difference']:.3e}"
        )

        del final_panel

        gc.collect()

        stage = "model_input_contract"

        print(
            "[6/10] Freezing neural-common versus horizon-specific sample contracts..."
        )

        model_input_contract = (
            build_model_input_contract()
        )

        rolling = model_input_contract.loc[
            model_input_contract[
                "scope"
            ]
            ==
            "rolling_fold"
        ]

        for fold in FOLDS:
            subset = rolling.loc[
                rolling[
                    "fold"
                ]
                ==
                fold
            ]

            for partition in PARTITIONS:
                part = subset.loc[
                    subset[
                        "partition"
                    ]
                    ==
                    partition
                ]

                lookup = {
                    row.dataset_contract: int(
                        row.eligible_sequences
                    )
                    for row in part.itertuples(
                        index=False
                    )
                }

                print(
                    f"       Fold {fold} {partition:<10}: "
                    f"1d={lookup['horizon_specific_1d']:>6,} | "
                    f"3d={lookup['horizon_specific_3d']:>6,} | "
                    f"5d/common={lookup['common_multitask_neural']:>6,}"
                )

        stage = "audit_tables"

        print(
            "[7/10] Building integrity, source-alignment, labels and auxiliary audits..."
        )

        integrity_summary = pd.DataFrame(
            integrity_rows
        )

        auxiliary_audit = pd.DataFrame(
            auxiliary_rows
        )

        label_distribution = pd.DataFrame(
            label_rows
        )

        if len(
            integrity_summary
        ) != 11:
            raise RuntimeError(
                "Expected 11 materialized-partition integrity records."
            )

        if not integrity_summary[
            "passed"
        ].all():
            raise RuntimeError(
                "One or more materialized-partition audits failed."
            )

        for column in (
            "stock_slice_exact",
            "market_slice_exact",
            "metadata_direction_exact",
            "metadata_aux_raw_exact",
            "metadata_stock_id_exact",
            "metadata_regime_id_exact",
            "auxiliary_scaler_reproduction_exact",
            "sequence_start_alignment_exact",
            "sequence_endpoint_alignment_exact",
        ):
            if not integrity_summary[
                column
            ].all():
                raise RuntimeError(
                    f"Integrity failure: {column}"
                )

        if not auxiliary_audit[
            "transform_reproduction_exact"
        ].all():
            raise RuntimeError(
                "Auxiliary scaler reproduction failed."
            )

        source_alignment = (
            integrity_summary[
                [
                    "scope",
                    "fold",
                    "partition",
                    "sequence_start_alignment_exact",
                    "sequence_endpoint_alignment_exact",
                    "full_sequence_samples_checked",
                    "sampled_max_absolute_difference",
                    "cross_boundary_endpoints",
                ]
            ]
            .copy()
        )

        print(
            "       Integrity records           : "
            f"{len(integrity_summary)}"
        )
        print(
            "       All integrity audits pass   : "
            f"{bool(integrity_summary['passed'].all())}"
        )
        print(
            "       Auxiliary reproductions     : "
            f"{len(auxiliary_audit)}"
        )
        print(
            "       Sampled full-window max diff: "
            f"{integrity_summary['sampled_max_absolute_difference'].max():.3e}"
        )

        stage = "write_and_freeze"

        print(
            "[8/10] Writing audits and freezing the pre-modelling development interface..."
        )

        write_csv(
            INTEGRITY_SUMMARY_CSV,
            integrity_summary,
        )

        write_csv(
            HASH_VERIFICATION_CSV,
            hash_verification,
        )

        write_csv(
            SOURCE_ALIGNMENT_CSV,
            source_alignment,
        )

        write_csv(
            LABEL_DISTRIBUTION_CSV,
            label_distribution,
        )

        write_csv(
            AUXILIARY_AUDIT_CSV,
            auxiliary_audit,
        )

        write_csv(
            MODEL_INPUT_CONTRACT_CSV,
            model_input_contract,
        )

        write_csv(
            SOURCE_HASH_CSV,
            source_hashes(),
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
                "Global final-test lock changed during Step 15."
            )

        freeze = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "DEVELOPMENT_MODEL_INTERFACE_FROZEN_BEFORE_MODELLING"
            ),
            "generated_at_utc": utc_now(),
            "configuration_fingerprint_sha256": (
                step00[
                    "fingerprint"
                ]
            ),
            "feature_interface": {
                "stock_features": (
                    STOCK_FEATURE_COUNT
                ),
                "market_features": (
                    MARKET_FEATURE_COUNT
                ),
                "total_features": (
                    TOTAL_FEATURE_COUNT
                ),
                "ordered_features": (
                    all_features
                ),
                "feature_interface_sha256": (
                    upstream[
                        "feature_interface_sha256"
                    ]
                ),
            },
            "sequence_interface": {
                "lookback_sessions": LOOKBACK,
                "X_full_shape": "[n,60,63]",
                "X_stock_shape": "[n,60,47]",
                "X_market_shape": "[n,60,16]",
                "direction_target_order": (
                    DIRECTION_COLUMNS
                ),
                "auxiliary_target_order": (
                    AUXILIARY_TARGET_COLUMNS
                ),
                "causal_cross_validation_boundary_history": True,
                "target_endpoint_partition_confinement": True,
            },
            "common_multitask_counts": {
                "rolling_folds": (
                    EXPECTED_COMMON_COUNTS
                ),
                "final_development": (
                    EXPECTED_FINAL_DEVELOPMENT_COMMON
                ),
            },
            "model_input_contract": {
                "neural_common_multitask": (
                    "use frozen Step-14 common-multitask arrays"
                ),
                "horizon_specific_1d_3d_5d": (
                    "retain Step-13 horizon-specific eligibility; "
                    "do not silently restrict 1d/3d models to the 5d/common intersection"
                ),
                "horizon_specific_features_selected": False,
                "horizon_specific_arrays_materialized_here": False,
            },
            "stock_id_map": (
                stock_id_map
            ),
            "preprocessing": {
                "feature_scaling": (
                    "fold-specific training-only MinMaxScaler"
                ),
                "feature_scaler_clip": False,
                "auxiliary_scaling": (
                    "fold-specific training-common-endpoint MinMaxScaler"
                ),
                "auxiliary_scaler_clip": False,
                "directional_labels_scaled": False,
                "regime_source": (
                    "unscaled rolling_std_20"
                ),
                "regime_thresholds": (
                    "stock-specific fold-training q33/q67"
                ),
            },
            "integrity": {
                "step14_manifest_artifacts_verified": int(
                    len(
                        hash_verification
                    )
                ),
                "step14_model_artifacts_verified": int(
                    hash_verification[
                        "is_model_artifact"
                    ].sum()
                ),
                "all_hashes_match": bool(
                    hash_verification[
                        "passed"
                    ].all()
                ),
                "all_materialized_partition_audits_pass": bool(
                    integrity_summary[
                        "passed"
                    ].all()
                ),
                "all_stock_slices_exact": bool(
                    integrity_summary[
                        "stock_slice_exact"
                    ].all()
                ),
                "all_market_slices_exact": bool(
                    integrity_summary[
                        "market_slice_exact"
                    ].all()
                ),
                "all_sequence_start_rows_exact": bool(
                    integrity_summary[
                        "sequence_start_alignment_exact"
                    ].all()
                ),
                "all_sequence_endpoint_rows_exact": bool(
                    integrity_summary[
                        "sequence_endpoint_alignment_exact"
                    ].all()
                ),
                "all_auxiliary_transforms_reproduce": bool(
                    auxiliary_audit[
                        "transform_reproduction_exact"
                    ].all()
                ),
                "sampled_full_window_max_difference": float(
                    integrity_summary[
                        "sampled_max_absolute_difference"
                    ].max()
                ),
            },
            "final_test_protection": {
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
                "final_test_data_file_opened_or_hashed_by_step15": False,
                "final_test_arrays_created": False,
                "test_used_for_model_selection": False,
                "test_used_for_feature_selection": False,
                "test_used_for_threshold_selection": False,
                "test_used_for_seed_selection": False,
                "test_performance_computed": False,
                "global_final_test_state": (
                    step00[
                        "state"
                    ]
                ),
            },
            "prohibited_after_freeze_until_model_registry_is_frozen": [
                "changing predictor definitions",
                "changing predictor order",
                "changing target definitions",
                "changing lookback length",
                "changing fold boundaries",
                "changing target-purging rules",
                "changing causal cross-boundary validation lookback policy",
                "changing feature scaling using validation performance",
                "changing regime thresholds using validation performance",
                "changing auxiliary scaling using validation performance",
                "discarding valid horizon-specific 1d/3d samples solely to match the 5d/common set",
                "accessing final-test labels or final-test model performance",
            ],
            "modelling_started": False,
            "model_registry_frozen": False,
        }

        atomic_json(
            FREEZE_JSON,
            freeze,
        )

        freeze_sha = sha256_file(
            FREEZE_JSON
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
                    "=" * 108,
                    "Status: PASS",
                    (
                        "Configuration fingerprint: "
                        f"{step00['fingerprint']}"
                    ),
                    (
                        "Step-14 manifest SHA-256: "
                        f"{upstream['manifest_sha'][14]}"
                    ),
                    (
                        "Feature-interface SHA-256: "
                        f"{upstream['feature_interface_sha256']}"
                    ),
                    (
                        "Split-design SHA-256: "
                        f"{upstream['split_design_sha256']}"
                    ),
                    (
                        "Step-14 manifest artifacts verified: "
                        f"{len(hash_verification)}"
                    ),
                    (
                        "Step-14 model artifacts verified: "
                        f"{int(hash_verification['is_model_artifact'].sum())}"
                    ),
                    (
                        "Materialized partitions audited: "
                        f"{len(integrity_summary)}"
                    ),
                    (
                        "Sampled full-window max difference: "
                        f"{integrity_summary['sampled_max_absolute_difference'].max():.3e}"
                    ),
                    (
                        "Final-development common sequences: "
                        f"{EXPECTED_FINAL_DEVELOPMENT_COMMON}"
                    ),
                    "Model registry frozen: False",
                    "Models fitted: False",
                    "Final-test parquet accessed: False",
                    (
                        "Global final-test state: "
                        f"{step00['state']}"
                    ),
                ]
            )
            +
            "\n",
        )

        output_paths_pre_manifest = (
            INTEGRITY_SUMMARY_CSV,
            HASH_VERIFICATION_CSV,
            SOURCE_ALIGNMENT_CSV,
            LABEL_DISTRIBUTION_CSV,
            AUXILIARY_AUDIT_CSV,
            MODEL_INPUT_CONTRACT_CSV,
            SOURCE_HASH_CSV,
            FREEZE_JSON,
            FROZEN_SCRIPT,
            SUMMARY_TXT,
        )

        # Build the output-hash register BEFORE the manifest, exactly as in
        # earlier clean-V3 stages.  The register hashes only non-manifest
        # Step-15 outputs, avoiding any manifest/register circularity.
        write_csv(
            HASH_REGISTER_CSV,
            pd.DataFrame(
                [
                    {
                        "file_name": path.name,
                        "file_path": str(
                            path
                        ),
                        "relative_path": (
                            relative_to_experiment(
                                path
                            )
                        ),
                        "size_bytes": int(
                            path.stat().st_size
                        ),
                        "sha256": sha256_file(
                            path
                        ),
                    }
                    for path in output_paths_pre_manifest
                ]
            ),
        )

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
                    15,
                )
            },
            "feature_interface_sha256": (
                upstream[
                    "feature_interface_sha256"
                ]
            ),
            "ordered_63_features": (
                all_features
            ),
            "stock_features": (
                upstream[
                    "stock_features"
                ]
            ),
            "market_features": (
                upstream[
                    "market_features"
                ]
            ),
            "split_design_sha256": (
                upstream[
                    "split_design_sha256"
                ]
            ),
            "direction_target_order": (
                DIRECTION_COLUMNS
            ),
            "auxiliary_target_order": (
                AUXILIARY_TARGET_COLUMNS
            ),
            "stock_id_map": (
                stock_id_map
            ),
            "pre_model_freeze_sha256": (
                freeze_sha
            ),
            "integrity": {
                "step14_total_artifacts_verified": int(
                    len(
                        hash_verification
                    )
                ),
                "step14_model_artifacts_verified": int(
                    hash_verification[
                        "is_model_artifact"
                    ].sum()
                ),
                "materialized_partitions_audited": int(
                    len(
                        integrity_summary
                    )
                ),
                "all_passed": True,
            },
            "model_input_contract": {
                "common_multitask_arrays_frozen": True,
                "horizon_specific_eligibility_frozen_from_step13": True,
                "horizon_specific_arrays_materialized_in_step15": False,
                "horizon_specific_feature_selection_performed": False,
            },
            "pipeline_state": {
                "development_arrays_frozen": True,
                "final_development_arrays_frozen": True,
                "pre_model_integrity_complete": True,
                "model_registry_frozen": False,
                "feature_selection_applied": False,
                "models_fitted": False,
                "probability_threshold_selection_performed": False,
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
                "sealed_test_parquet_opened_or_hashed_by_step15": False,
                "final_test_arrays_created": False,
                "test_performance_computed": False,
            },
            "artifacts": {
                path.name: artifact_record(
                    path
                )
                for path in (
                    *output_paths_pre_manifest,
                    HASH_REGISTER_CSV,
                )
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

        stage = "validate"

        print(
            "[9/10] Validating frozen Step-15 outputs..."
        )

        verify_sidecar(
            MANIFEST_JSON,
            MANIFEST_SHA,
            "Step 15",
        )

        stored_manifest = json.loads(
            MANIFEST_JSON.read_text(
                encoding="utf-8"
            )
        )

        for record in stored_manifest[
            "artifacts"
        ].values():
            path = Path(
                record[
                    "path"
                ]
            )

            if not path.exists():
                raise RuntimeError(
                    f"Frozen Step-15 artifact missing: {path}"
                )

            if sha256_file(
                path
            ) != record[
                "sha256"
            ]:
                raise RuntimeError(
                    f"Frozen Step-15 artifact mismatch: {path}"
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
                "Global final-test lock changed during Step-15 validation."
            )

        stage = "report"

        print(
            "[10/10] Final Step-15 report..."
        )

        print()
        print("-" * 136)
        print("ARTIFACT INTEGRITY")
        print("-" * 136)
        print(
            "Step-14 manifest artifacts verified : "
            f"{len(hash_verification)}"
        )
        print(
            "Step-14 model artifacts verified    : "
            f"{int(hash_verification['is_model_artifact'].sum())}"
        )
        print(
            "All SHA-256 hashes match            : "
            f"{bool(hash_verification['passed'].all())}"
        )

        print("-" * 136)
        print("MATERIALIZED DATASET INTEGRITY")
        print("-" * 136)

        for row in (
            integrity_summary
            .sort_values(
                [
                    "fold",
                    "partition",
                ],
                na_position="last",
            )
            .itertuples(
                index=False
            )
        ):
            if pd.isna(
                row.fold
            ):
                label = "Final development"
            else:
                label = (
                    f"Fold {int(row.fold)} {row.partition}"
                )

            print(
                f"{label:<25} | "
                f"n={row.sequences:>6,} | "
                f"per-stock={row.sequences_per_stock:>4,} | "
                f"cross={row.cross_boundary_endpoints:>5,} | "
                f"full-window checks={row.full_sequence_samples_checked:>2,} | "
                f"max diff={row.sampled_max_absolute_difference:.3e}"
            )

        print("-" * 136)
        print("SOURCE / ARRAY CONSISTENCY")
        print("-" * 136)
        print(
            "Stock modality slices exact        : "
            f"{bool(integrity_summary['stock_slice_exact'].all())}"
        )
        print(
            "Market modality slices exact       : "
            f"{bool(integrity_summary['market_slice_exact'].all())}"
        )
        print(
            "Sequence starts match Step-12      : "
            f"{bool(integrity_summary['sequence_start_alignment_exact'].all())}"
        )
        print(
            "Sequence endpoints match Step-12   : "
            f"{bool(integrity_summary['sequence_endpoint_alignment_exact'].all())}"
        )
        print(
            "Auxiliary transforms reproduce     : "
            f"{bool(auxiliary_audit['transform_reproduction_exact'].all())}"
        )
        print(
            "Sampled full-window max difference : "
            f"{integrity_summary['sampled_max_absolute_difference'].max():.3e}"
        )

        print("-" * 136)
        print("MODEL-INPUT CONTRACT")
        print("-" * 136)
        print(
            "Neural common-multitask arrays     : FROZEN FROM STEP 14"
        )
        print(
            "1d/3d horizon-specific eligibility : RETAINED FROM STEP 13"
        )
        print(
            "5d horizon-specific = common       : VERIFIED"
        )
        print(
            "Horizon-specific feature selection : NOT PERFORMED"
        )
        print(
            "Horizon-specific arrays here        : NOT MATERIALIZED"
        )

        print("-" * 136)
        print("FINAL TEST PROTECTION")
        print("-" * 136)
        print(
            "Step-10 test seal                  : "
            f"{upstream['test_seal_status']}"
        )
        print(
            "Final-test data opened in Step 15  : False"
        )
        print(
            "Final-test performance             : NOT COMPUTED"
        )
        print(
            "Global final-test state            : "
            f"{step00['state']}"
        )

        print("-" * 136)
        print("FREEZE STATE")
        print("-" * 136)
        print(
            "63-feature interface               : FROZEN"
        )
        print(
            "1d / 3d / 5d targets              : FROZEN"
        )
        print(
            "Five rolling-origin folds          : FROZEN"
        )
        print(
            "60-session causal lookback         : FROZEN"
        )
        print(
            "Train-only feature scaling         : FROZEN"
        )
        print(
            "Train-only volatility regimes      : FROZEN"
        )
        print(
            "Auxiliary target scaling           : FROZEN"
        )
        print(
            "Rolling development arrays         : FROZEN"
        )
        print(
            "Final-development refit arrays     : FROZEN"
        )
        print(
            "Model registry                     : NOT YET FROZEN"
        )
        print(
            "Models fitted                      : False"
        )

        print("-" * 136)
        print("OUTPUT FILES")
        print("-" * 136)
        print(
            f"Integrity summary      : {INTEGRITY_SUMMARY_CSV}"
        )
        print(
            f"Hash verification      : {HASH_VERIFICATION_CSV}"
        )
        print(
            f"Source alignment       : {SOURCE_ALIGNMENT_CSV}"
        )
        print(
            f"Label distribution     : {LABEL_DISTRIBUTION_CSV}"
        )
        print(
            f"Auxiliary audit        : {AUXILIARY_AUDIT_CSV}"
        )
        print(
            f"Model-input contract   : {MODEL_INPUT_CONTRACT_CSV}"
        )
        print(
            f"Pre-model freeze       : {FREEZE_JSON}"
        )
        print(
            f"Freeze SHA-256         : {freeze_sha}"
        )
        print(
            f"Manifest               : {MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256       : {manifest_sha}"
        )

        print("=" * 136)
        print(
            "PASS: Every frozen Step-14 artifact matches its manifest SHA-256."
        )
        print(
            "PASS: All ten rolling-fold datasets and the final-development "
            "dataset reconcile with metadata and canonical Step-12 sources."
        )
        print(
            "PASS: Causal 60-session validation history is preserved without "
            "future feature leakage."
        )
        print(
            "PASS: Direction, auxiliary-target, stock-ID and regime-ID "
            "interfaces are internally consistent."
        )
        print(
            "PASS: Step-13 horizon-specific eligibility remains distinct from "
            "the Step-14 common-multitask materialization."
        )
        print(
            "PASS: The complete development modelling interface is frozen "
            "before model fitting."
        )
        print(
            "PASS: The sealed final-test parquet was not opened or hashed."
        )
        print("=" * 136)
        print(
            "Do not run the old Step-15 script. "
            "Do not run Step 16 yet. Paste the complete terminal output for audit."
        )

        return 0

    except Exception as exc:
        archived = archive_partial()

        FAILED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        stamp = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y%m%dT%H%M%SZ"
            )
        )

        failure_path = (
            FAILED_DIR
            /
            f"15_pre_model_integrity_FAILED_{stamp}.json"
        )

        atomic_json(
            failure_path,
            {
                "step": STEP_NAME,
                "status": "FAIL",
                "failed_at_utc": utc_now(),
                "stage": stage,
                "error_type": type(
                    exc
                ).__name__,
                "error_message": str(
                    exc
                ),
                "python_executable": (
                    sys.executable
                ),
                "archived_partial_artifacts": (
                    None
                    if archived is None
                    else str(
                        archived
                    )
                ),
                "canonical_manifest_exists": (
                    MANIFEST_JSON.exists()
                ),
            },
        )

        print()
        print("=" * 136)
        print("STEP 15 FAILED — PRE-MODELLING INTERFACE NOT FROZEN")
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
            "Do not proceed to Step 16."
        )

        return 30


# =============================================================================
# 19. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean-V3 comprehensive pre-modelling integrity freeze."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an existing frozen Step-15 stage without recomputation."
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
            "\nStep 15 interrupted by user."
        )
        sys.exit(
            130
        )

    except Exception as exc:
        print()
        print("=" * 136)
        print("STEP 15 UNHANDLED FAILURE")
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
