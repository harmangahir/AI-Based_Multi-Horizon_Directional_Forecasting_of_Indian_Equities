#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 35 — FINAL-TEST EVALUATION CONTRACT FREEZE

Purpose
-------
Freeze, BEFORE any final-test access:

1. final-test materialization contract;
2. causal history / sequence contract;
3. frozen preprocessing and volatility-regime application;
4. 40-checkpoint inference plan;
5. architecture-specific physical inference microbatches;
6. five-seed probability ensemble;
7. development-OOF frozen thresholds;
8. final-test metric definitions;
9. dependence-aware statistical inference;
10. descriptive subgroup/calibration analyses;
11. irreversible post-test prohibitions.

CRITICAL:
The sealed final-test parquet is NOT opened, read, hashed, stat'ed,
materialized, transformed, or inspected by this script.

The only information about the final test used here comes from the
already-frozen Step-10 seal metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent

if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError(
        "Place this script inside the repository q1_revision directory."
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg


REPOSITORY_ROOT = SCRIPT_DIR.parent

EXPERIMENT_ROOT = (
    REPOSITORY_ROOT
    / "stock_q1_revision_project"
    / "experiments"
    / "q1_revision_v3_clean_2016_2026"
)

MANIFEST_DIR = EXPERIMENT_ROOT / "manifests"
AUDIT_DIR = EXPERIMENT_ROOT / "audits"
FEATURE_DIR = EXPERIMENT_ROOT / "feature_data"

STEP10_SEAL = MANIFEST_DIR / "10_final_test_seal.json"

STEP11_MANIFEST = MANIFEST_DIR / "11_preprocessing_manifest.json"
STEP12_MANIFEST = MANIFEST_DIR / "12_volatility_regime_manifest.json"
STEP13_MANIFEST = MANIFEST_DIR / "13_sequence_index_manifest.json"
STEP14_MANIFEST = MANIFEST_DIR / "14_model_array_manifest.json"
STEP16_MANIFEST = MANIFEST_DIR / "16_stage1_model_registry_manifest.json"

STEP32_THRESHOLD_CSV = (
    AUDIT_DIR
    / "32_selected_development_thresholds.csv"
)

STEP34_MANIFEST = (
    MANIFEST_DIR
    / "34_final_development_refit_final_freeze_manifest.json"
)

STEP34_REGISTER = (
    AUDIT_DIR
    / "34_final_development_refit_register.csv"
)

STEP12_FINAL_PANEL = (
    FEATURE_DIR
    / "step12_volatility_regimes"
    / "final_development"
    / "12_final_development_scaled_regimes.parquet"
)

OUT_PLAN = (
    AUDIT_DIR
    / "35_final_test_inference_plan.csv"
)

OUT_THRESHOLDS = (
    AUDIT_DIR
    / "35_final_test_threshold_contract.csv"
)

OUT_MANIFEST = (
    MANIFEST_DIR
    / "35_final_test_evaluation_contract_freeze_manifest.json"
)

OUT_MANIFEST_SHA = (
    MANIFEST_DIR
    / "35_final_test_evaluation_contract_freeze_manifest.sha256"
)

FROZEN_SCRIPT = (
    MANIFEST_DIR
    / "step35_freeze_final_test_contract_v1_frozen.py"
)


# =============================================================================
# AUTHORITATIVE HASHES
# =============================================================================

EXPECTED_STEP34_MANIFEST_SHA = (
    "2224aa145e4bdbce2563586cd1a28742facd29638985f29fb4c6a34fa9bcaf48"
)

EXPECTED_STEP34_GLOBAL_FP = (
    "1ede9c52aaadf66ea4e82f777c4c03fcccece419bff0867c2a434e3fb13828c0"
)

EXPECTED_STEP32_MANIFEST_SHA = (
    "474b77ac670bbb7f61870372cdb98a6d7ac3ecaa4db8864ceb096fe9b26c39f0"
)

EXPECTED_STEP32_THRESHOLD_SHA = (
    "004cfce95ebd4a3e27955f7766e3aac91611bb76193fbea364c4ea0dd7f03751"
)

EXPECTED_STEP14_MANIFEST_SHA = (
    "b19b034bd31bb5d426e78c58f808d9e4c9cba16281e666984aae2c8bf6a150a0"
)

EXPECTED_STEP12_FINAL_PANEL_SHA = (
    "c166253e781031095c9c90fc864973e2b12e5fe2b8950d81e7fda0f9c2eba9bd"
)

EXPECTED_SELECTED_CONFIGS = {
    "DLINEAR": "DLINEAR_B",
    "NLINEAR": "NLINEAR_B",
    "PATCHTST": "PATCHTST_B",
    "TIMESNET": "TIMESNET_B",
    "ITRANSFORMER": "ITRANSFORMER_A",
    "FREDFORMER": "FREDFORMER_B",
    "FRNET": "FRNET_A",
    "MLF": "MLF_A",
}

EXPECTED_PHYSICAL_MICROBATCH = {
    "DLINEAR": 128,
    "NLINEAR": 128,
    "PATCHTST": 64,
    "TIMESNET": 64,
    "ITRANSFORMER": 128,
    "FREDFORMER": 16,
    "FRNET": 64,
    "MLF": 32,
}

EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_HORIZONS = (1, 3, 5)

EXPECTED_ARCHITECTURES = 8
EXPECTED_FINAL_REFITS = 40
EXPECTED_THRESHOLD_ROWS = 24

EXPECTED_TEST_REFERENCE_ROWS = 6680
EXPECTED_TEST_REFERENCE_DATES = 334
EXPECTED_TEST_COMMON_ROWS = 6480
EXPECTED_TEST_COMMON_DATES = 324
EXPECTED_STOCK_COUNT = 20

LOOKBACK = 60
FEATURE_COUNT = 63
OUTPUT_COUNT = 3


# =============================================================================
# UTILITIES
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """
    Hard guard: Step 35 is prohibited from hashing the sealed test parquet.
    """
    if path.name == "10_final_test_locked_panel.parquet":
        raise RuntimeError(
            "PROGRAMMING ERROR: Step 35 attempted to hash the sealed final-test parquet."
        )

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def canonical_json_hash(obj: Any) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return sha256_bytes(payload)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing JSON: {path}")

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def artifact_record(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "relative_path": str(
            path.resolve().relative_to(
                EXPERIMENT_ROOT.resolve()
            )
        ),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(
            path
        ),
    }


def require_equal(
    observed: Any,
    expected: Any,
    label: str,
) -> None:
    if observed != expected:
        raise RuntimeError(
            f"{label} mismatch:\n"
            f"  expected={expected}\n"
            f"  observed={observed}"
        )


def require_true(
    value: Any,
    label: str,
) -> None:
    if value is not True:
        raise RuntimeError(
            f"{label} is not True: {value}"
        )


def require_false(
    value: Any,
    label: str,
) -> None:
    if value is not False:
        raise RuntimeError(
            f"{label} is not False: {value}"
        )


def walk_artifact_records(
    obj: Any,
) -> Iterable[Dict[str, Any]]:

    if isinstance(
        obj,
        dict,
    ):
        if (
            isinstance(
                obj.get("path"),
                str,
            )
            and
            isinstance(
                obj.get("sha256"),
                str,
            )
        ):
            yield obj

        for value in obj.values():
            yield from walk_artifact_records(
                value
            )

    elif isinstance(
        obj,
        list,
    ):
        for value in obj:
            yield from walk_artifact_records(
                value
            )


def artifact_path(
    rec: Dict[str, Any],
) -> Path:

    if rec.get(
        "path"
    ):
        return Path(
            rec["path"]
        )

    if rec.get(
        "relative_path"
    ):
        return (
            EXPERIMENT_ROOT
            / rec[
                "relative_path"
            ]
        )

    raise RuntimeError(
        "Artifact record has no path."
    )


def find_artifact_by_basename(
    manifest: Dict[str, Any],
    basename: str,
) -> Dict[str, Any]:

    candidates = []

    for rec in walk_artifact_records(
        manifest
    ):
        p = artifact_path(
            rec
        )

        if p.name == basename:
            candidates.append(
                rec
            )

    unique = {}

    for rec in candidates:
        p = str(
            artifact_path(
                rec
            )
        )
        unique[
            p
        ] = rec

    candidates = list(
        unique.values()
    )

    if len(
        candidates
    ) != 1:
        raise RuntimeError(
            f"Expected exactly one artifact named {basename}; "
            f"found {len(candidates)}."
        )

    return candidates[
        0
    ]


def verify_artifact_record(
    rec: Dict[str, Any],
    label: str,
) -> Path:

    path = artifact_path(
        rec
    )

    if path.name == "10_final_test_locked_panel.parquet":
        raise RuntimeError(
            "PROGRAMMING ERROR: sealed final-test parquet entered "
            "Step-35 artifact verification."
        )

    if not path.exists():
        raise RuntimeError(
            f"{label} missing: {path}"
        )

    observed = sha256_file(
        path
    )

    expected = str(
        rec[
            "sha256"
        ]
    )

    if observed != expected:
        raise RuntimeError(
            f"{label} SHA mismatch:\n"
            f"  expected={expected}\n"
            f"  observed={observed}\n"
            f"  path={path}"
        )

    return path


def find_json_by_sha(
    prefix: str,
    expected_sha: str,
) -> Path:

    matches = []

    for path in sorted(
        MANIFEST_DIR.glob(
            f"{prefix}*.json"
        )
    ):
        if path.name == "10_final_test_seal.json":
            continue

        if sha256_file(
            path
        ) == expected_sha:
            matches.append(
                path
            )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            f"Could not uniquely locate {prefix} manifest by SHA. "
            f"matches={matches}"
        )

    return matches[
        0
    ]


def verify_manifest_sidecar_if_present(
    manifest_path: Path,
) -> None:

    sidecar = manifest_path.with_suffix(
        ".sha256"
    )

    if not sidecar.exists():
        return

    expected = (
        sidecar
        .read_text(
            encoding="utf-8"
        )
        .strip()
        .split()[0]
    )

    observed = sha256_file(
        manifest_path
    )

    if expected != observed:
        raise RuntimeError(
            f"Manifest sidecar mismatch: {manifest_path.name}"
        )


def select_final_development_predictor_scaler(
    step11: Dict[str, Any],
) -> tuple[Dict[str, Any], Path]:

    candidates = []

    for rec in walk_artifact_records(
        step11
    ):
        p = artifact_path(
            rec
        )

        text = str(
            p
        ).lower()

        if (
            p.suffix.lower()
            == ".joblib"
            and
            "scaler"
            in text
            and
            "final"
            in text
        ):
            candidates.append(
                rec
            )

    # Prefer explicit final-development naming.
    explicit = [
        r
        for r in candidates
        if "final_development"
        in str(
            artifact_path(
                r
            )
        ).lower()
    ]

    if explicit:
        candidates = explicit

    unique = {}

    for rec in candidates:
        unique[
            str(
                artifact_path(
                    rec
                )
            )
        ] = rec

    candidates = list(
        unique.values()
    )

    if len(
        candidates
    ) != 1:

        print(
            "\nStep-11 scaler candidates:"
        )

        for rec in candidates:
            print(
                " ",
                artifact_path(
                    rec
                )
            )

        raise RuntimeError(
            "Could not uniquely identify the Step-11 "
            "final-development predictor scaler."
        )

    rec = candidates[
        0
    ]

    path = verify_artifact_record(
        rec,
        "Step-11 final-development predictor scaler",
    )

    return (
        rec,
        path,
    )


def read_table(
    path: Path,
) -> pd.DataFrame:

    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path
        )

    if path.suffix.lower() in (
        ".parquet",
        ".pq",
    ):
        return pd.read_parquet(
            path,
            engine="pyarrow",
        )

    raise RuntimeError(
        f"Unsupported threshold artifact type: {path}"
    )


def select_final_development_regime_thresholds(
    step12: Dict[str, Any],
    expected_tickers: set[str],
) -> tuple[
    Dict[str, Any],
    Path,
    pd.DataFrame,
]:

    required = {
        "ticker",
        "low_volatility_threshold_q33",
        "high_volatility_threshold_q67",
    }

    valid = []

    for rec in walk_artifact_records(
        step12
    ):
        p = artifact_path(
            rec
        )

        text = str(
            p
        ).lower()

        # Compatibility fix:
        # Step-12 contains fold-specific, final-development, and audit
        # threshold tables with the same 20-stock schema.  Step 35 must
        # deterministically bind ONLY to the canonical final-development
        # threshold artifact frozen before final-test access.
        if p.name != "12_final_development_regime_thresholds.csv":
            continue

        if (
            "threshold"
            not in text
            or
            p.suffix.lower()
            not in (
                ".csv",
                ".parquet",
                ".pq",
            )
        ):
            continue

        p = verify_artifact_record(
            rec,
            f"Step-12 threshold candidate {p.name}",
        )

        try:
            df = read_table(
                p
            )
        except Exception:
            continue

        if not required.issubset(
            df.columns
        ):
            continue

        subset = df.copy()

        if "scope" in subset.columns:
            mask = (
                subset[
                    "scope"
                ]
                .astype(str)
                .str.contains(
                    "final_development",
                    case=False,
                    regex=False,
                )
            )

            if mask.any():
                subset = subset.loc[
                    mask
                ].copy()

        if (
            len(
                subset
            )
            != EXPECTED_STOCK_COUNT
            and
            "fold"
            in subset.columns
        ):
            no_fold = subset[
                "fold"
            ].isna()

            if int(
                no_fold.sum()
            ) == EXPECTED_STOCK_COUNT:
                subset = subset.loc[
                    no_fold
                ].copy()

        if len(
            subset
        ) != EXPECTED_STOCK_COUNT:
            continue

        if set(
            subset[
                "ticker"
            ].astype(str)
        ) != expected_tickers:
            continue

        q33 = subset[
            "low_volatility_threshold_q33"
        ].astype(float)

        q67 = subset[
            "high_volatility_threshold_q67"
        ].astype(float)

        if not np.isfinite(
            q33
        ).all():
            continue

        if not np.isfinite(
            q67
        ).all():
            continue

        if not (
            q33
            <
            q67
        ).all():
            continue

        valid.append(
            (
                rec,
                p,
                subset,
            )
        )

    if len(
        valid
    ) != 1:

        print(
            "\nValid Step-12 final-development threshold candidates:"
        )

        for _, p, subset in valid:
            print(
                f"  {p} rows={len(subset)}"
            )

        raise RuntimeError(
            "Could not uniquely identify final-development "
            "volatility-regime thresholds."
        )

    return valid[
        0
    ]


# =============================================================================
# FRESHNESS GUARD
# =============================================================================

for output in (
    OUT_PLAN,
    OUT_THRESHOLDS,
    OUT_MANIFEST,
    OUT_MANIFEST_SHA,
    FROZEN_SCRIPT,
):
    if output.exists():
        raise RuntimeError(
            f"Immutable Step-35 output already exists:\n{output}"
        )


# =============================================================================
# START
# =============================================================================

print(
    "=" * 140
)
print(
    "STEP 35 — FINAL-TEST EVALUATION CONTRACT FREEZE"
)
print(
    "=" * 140
)

print(
    "Final-test parquet access : PROHIBITED"
)
print(
    "Final-test hashing        : PROHIBITED"
)
print(
    "Final-test class balance  : NOT INSPECTED"
)
print(
    "Final-test metrics        : NOT COMPUTED"
)
print()


# =============================================================================
# 1. LOAD AND VERIFY UPSTREAM FREEZES
# =============================================================================

print(
    "[1] VERIFYING UPSTREAM FREEZES"
)
print(
    "-" * 140
)

m34 = read_json(
    STEP34_MANIFEST
)

require_equal(
    sha256_file(
        STEP34_MANIFEST
    ),
    EXPECTED_STEP34_MANIFEST_SHA,
    "Step-34 manifest SHA",
)

require_equal(
    m34.get(
        "status"
    ),
    "STEP34_FINAL_DEVELOPMENT_REFITS_FINAL_FREEZE_COMPLETE",
    "Step-34 status",
)

require_equal(
    m34.get(
        "global_step34_fingerprint"
    ),
    EXPECTED_STEP34_GLOBAL_FP,
    "Step-34 global fingerprint",
)

protection34 = m34.get(
    "protection_state",
    {},
)

require_false(
    protection34.get(
        "architecture_selection_performed"
    ),
    "Step-34 architecture selection",
)

require_false(
    protection34.get(
        "best_seed_selection_performed"
    ),
    "Step-34 best-seed selection",
)

require_false(
    protection34.get(
        "best_seed_promotion_performed"
    ),
    "Step-34 best-seed promotion",
)

require_false(
    protection34.get(
        "final_test_data_opened"
    ),
    "Step-34 final-test data opened",
)

require_false(
    protection34.get(
        "final_test_metric_computed"
    ),
    "Step-34 final-test metric computed",
)

require_equal(
    protection34.get(
        "final_test_state"
    ),
    "LOCKED_NOT_OPENED",
    "Step-34 final-test state",
)

m14 = read_json(
    STEP14_MANIFEST
)

require_equal(
    sha256_file(
        STEP14_MANIFEST
    ),
    EXPECTED_STEP14_MANIFEST_SHA,
    "Step-14 manifest SHA",
)

m11 = read_json(
    STEP11_MANIFEST
)

m12 = read_json(
    STEP12_MANIFEST
)

m13 = read_json(
    STEP13_MANIFEST
)

m16 = read_json(
    STEP16_MANIFEST
)

for mp in (
    STEP11_MANIFEST,
    STEP12_MANIFEST,
    STEP13_MANIFEST,
    STEP14_MANIFEST,
    STEP16_MANIFEST,
):
    verify_manifest_sidecar_if_present(
        mp
    )

step32_manifest_path = find_json_by_sha(
    "32",
    EXPECTED_STEP32_MANIFEST_SHA,
)

m32 = read_json(
    step32_manifest_path
)

print(
    f"Step-14 manifest SHA : {EXPECTED_STEP14_MANIFEST_SHA}"
)
print(
    f"Step-32 manifest SHA : {EXPECTED_STEP32_MANIFEST_SHA}"
)
print(
    f"Step-34 manifest SHA : {EXPECTED_STEP34_MANIFEST_SHA}"
)
print(
    f"Step-34 global FP    : {EXPECTED_STEP34_GLOBAL_FP}"
)
print(
    "PASS"
)


# =============================================================================
# 2. VERIFY PRISTINE GLOBAL LOCK + STEP-10 SEAL METADATA
# =============================================================================

print()
print(
    "[2] VERIFYING FINAL-TEST SEAL METADATA WITHOUT ACCESSING TEST DATA"
)
print(
    "-" * 140
)

lock_path = cfg.get_final_test_lock_path()

lock = read_json(
    lock_path
)

require_equal(
    lock.get(
        "state"
    ),
    "LOCKED_NOT_OPENED",
    "Global final-test state",
)

require_false(
    lock.get(
        "final_test_opened"
    ),
    "Global final_test_opened",
)

require_false(
    lock.get(
        "final_test_scored"
    ),
    "Global final_test_scored",
)

seal = read_json(
    STEP10_SEAL
)

require_equal(
    seal.get(
        "status"
    ),
    "SEALED_NOT_OPENED",
    "Step-10 test seal status",
)

require_equal(
    seal.get(
        "global_final_test_lock_state"
    ),
    "LOCKED_NOT_OPENED",
    "Step-10 global final-test lock state",
)

require_equal(
    int(
        seal[
            "reference_rows"
        ]
    ),
    EXPECTED_TEST_REFERENCE_ROWS,
    "Test reference rows from seal metadata",
)

require_equal(
    int(
        seal[
            "unique_reference_dates"
        ]
    ),
    EXPECTED_TEST_REFERENCE_DATES,
    "Test reference dates from seal metadata",
)

require_equal(
    int(
        seal[
            "common_multitask_structural_rows"
        ]
    ),
    EXPECTED_TEST_COMMON_ROWS,
    "Test common-multitask structural rows",
)

require_equal(
    int(
        seal[
            "stock_count"
        ]
    ),
    EXPECTED_STOCK_COUNT,
    "Test stock count",
)

require_equal(
    seal[
        "first_reference_date"
    ],
    "2025-01-01",
    "Test first reference date",
)

require_equal(
    seal[
        "last_reference_date"
    ],
    "2026-06-30",
    "Test last reference date",
)

# IMPORTANT:
# seal["file_path"] is stored only as metadata.
# Do not Path.exists(), stat(), hash(), or read it here.
sealed_test_path_string = str(
    seal[
        "file_path"
    ]
)

sealed_test_expected_sha = str(
    seal[
        "sha256"
    ]
)

print(
    f"Outer test rows         : {EXPECTED_TEST_REFERENCE_ROWS:,}"
)
print(
    f"Outer test dates        : {EXPECTED_TEST_REFERENCE_DATES:,}"
)
print(
    f"Common evaluation rows  : {EXPECTED_TEST_COMMON_ROWS:,}"
)
print(
    f"Common evaluation dates : {EXPECTED_TEST_COMMON_DATES:,}"
)
print(
    f"Stocks                  : {EXPECTED_STOCK_COUNT}"
)
print(
    "Test parquet read       : False"
)
print(
    "Test parquet hashed     : False"
)
print(
    "PASS"
)


# =============================================================================
# 3. VERIFY FINAL-DEVELOPMENT PREPROCESSING / REGIME STATE
# =============================================================================

print()
print(
    "[3] VERIFYING DEVELOPMENT-FROZEN TRANSFORMATION STATE"
)
print(
    "-" * 140
)

features = list(
    m14[
        "ordered_63_features"
    ]
)

require_equal(
    len(
        features
    ),
    FEATURE_COUNT,
    "Frozen feature count",
)

stock_id_map = {
    str(
        k
    ): int(
        v
    )
    for k, v in m14[
        "stock_id_map"
    ].items()
}

require_equal(
    len(
        stock_id_map
    ),
    EXPECTED_STOCK_COUNT,
    "Frozen stock ID map size",
)

expected_tickers = set(
    stock_id_map
)

scaler_rec, scaler_path = (
    select_final_development_predictor_scaler(
        m11
    )
)

predictor_scaler = joblib.load(
    scaler_path
)

require_equal(
    int(
        predictor_scaler.n_features_in_
    ),
    FEATURE_COUNT,
    "Final-development predictor scaler feature count",
)

require_equal(
    bool(
        predictor_scaler.clip
    ),
    False,
    "Final-development predictor scaler clip flag",
)

if hasattr(
    predictor_scaler,
    "feature_names_in_",
):
    require_equal(
        list(
            predictor_scaler.feature_names_in_
        ),
        features,
        "Final-development predictor scaler feature order",
    )

if not STEP12_FINAL_PANEL.exists():
    raise RuntimeError(
        f"Missing final-development Step-12 history panel: {STEP12_FINAL_PANEL}"
    )

require_equal(
    sha256_file(
        STEP12_FINAL_PANEL
    ),
    EXPECTED_STEP12_FINAL_PANEL_SHA,
    "Step-12 final-development history panel SHA",
)

(
    threshold_state_rec,
    threshold_state_path,
    final_regime_thresholds,
) = select_final_development_regime_thresholds(
    m12,
    expected_tickers,
)

print(
    f"Predictor scaler       : {scaler_path.name}"
)
print(
    f"Predictor scaler SHA   : {scaler_rec['sha256']}"
)
print(
    f"Predictor scaler clip  : {predictor_scaler.clip}"
)
print(
    f"Regime threshold rows  : {len(final_regime_thresholds)}"
)
print(
    f"Regime threshold file  : {threshold_state_path.name}"
)
print(
    f"Development history SHA: {EXPECTED_STEP12_FINAL_PANEL_SHA}"
)
print(
    "PASS"
)


# =============================================================================
# 4. VERIFY STEP-32 FROZEN THRESHOLDS
# =============================================================================

print()
print(
    "[4] VERIFYING DEVELOPMENT-OOF PROBABILITY THRESHOLDS"
)
print(
    "-" * 140
)

if not STEP32_THRESHOLD_CSV.exists():
    raise RuntimeError(
        f"Missing Step-32 threshold file: {STEP32_THRESHOLD_CSV}"
    )

require_equal(
    sha256_file(
        STEP32_THRESHOLD_CSV
    ),
    EXPECTED_STEP32_THRESHOLD_SHA,
    "Step-32 selected-threshold CSV SHA",
)

threshold_df = pd.read_csv(
    STEP32_THRESHOLD_CSV
)

require_equal(
    len(
        threshold_df
    ),
    EXPECTED_THRESHOLD_ROWS,
    "Frozen threshold row count",
)

require_equal(
    set(
        threshold_df[
            "model_id"
        ].astype(str)
    ),
    set(
        EXPECTED_SELECTED_CONFIGS
    ),
    "Threshold architecture set",
)

for model_id, config_id in EXPECTED_SELECTED_CONFIGS.items():

    subset = threshold_df.loc[
        threshold_df[
            "model_id"
        ]
        ==
        model_id
    ].copy()

    require_equal(
        len(
            subset
        ),
        3,
        f"{model_id} threshold count",
    )

    require_equal(
        set(
            subset[
                "horizon_days"
            ].astype(int)
        ),
        set(
            EXPECTED_HORIZONS
        ),
        f"{model_id} threshold horizons",
    )

    require_equal(
        set(
            subset[
                "config_id"
            ].astype(str)
        ),
        {
            config_id
        },
        f"{model_id} threshold config",
    )

    require_equal(
        set(
            subset[
                "threshold_operator"
            ].astype(str)
        ),
        {
            ">="
        },
        f"{model_id} threshold operator",
    )

    if subset[
        "final_test_used"
    ].astype(bool).any():
        raise RuntimeError(
            f"{model_id}: Step-32 thresholds unexpectedly used final test."
        )

threshold_df = threshold_df.sort_values(
    [
        "model_id",
        "horizon_days",
    ]
).reset_index(
    drop=True
)

print(
    f"Threshold rows          : {len(threshold_df)}/24"
)
print(
    "Threshold source        : pooled five-seed DEVELOPMENT OOF"
)
print(
    "Threshold operator      : probability >= frozen threshold"
)
print(
    "Final-test optimization : PROHIBITED"
)
print(
    "PASS"
)


# =============================================================================
# 5. VERIFY ALL 40 FINAL CHECKPOINTS AND BUILD INFERENCE PLAN
# =============================================================================

print()
print(
    "[5] VERIFYING 40 FROZEN FINAL-DEVELOPMENT CHECKPOINTS"
)
print(
    "-" * 140
)

register_rec = find_artifact_by_basename(
    m34,
    STEP34_REGISTER.name,
)

verify_artifact_record(
    register_rec,
    "Step-34 final refit register",
)

refit_df = pd.read_csv(
    STEP34_REGISTER
)

require_equal(
    len(
        refit_df
    ),
    EXPECTED_FINAL_REFITS,
    "Step-34 refit count",
)

plan_rows = []

for row in refit_df.itertuples(
    index=False
):

    model_id = str(
        row.model_id
    )

    config_id = str(
        row.config_id
    )

    seed = int(
        row.seed
    )

    require_equal(
        config_id,
        EXPECTED_SELECTED_CONFIGS[
            model_id
        ],
        f"{model_id}/seed{seed} selected config",
    )

    require_equal(
        int(
            row.physical_microbatch_size
        ),
        EXPECTED_PHYSICAL_MICROBATCH[
            model_id
        ],
        f"{model_id}/seed{seed} physical microbatch",
    )

    fit_manifest_path = Path(
        row.fit_manifest_path
    )

    if not fit_manifest_path.exists():
        raise RuntimeError(
            f"Missing fit manifest: {fit_manifest_path}"
        )

    require_equal(
        sha256_file(
            fit_manifest_path
        ),
        str(
            row.fit_manifest_sha256
        ),
        f"{model_id}/seed{seed} fit manifest SHA",
    )

    fit_manifest = read_json(
        fit_manifest_path
    )

    require_equal(
        str(
            fit_manifest[
                "model_id"
            ]
        ),
        model_id,
        f"{model_id}/seed{seed} fit model_id",
    )

    require_equal(
        str(
            fit_manifest[
                "config_id"
            ]
        ),
        config_id,
        f"{model_id}/seed{seed} fit config_id",
    )

    require_equal(
        int(
            fit_manifest[
                "seed"
            ]
        ),
        seed,
        f"{model_id}/seed{seed} fit seed",
    )

    require_equal(
        fit_manifest[
            "checkpoint_policy"
        ],
        "FINAL_SCHEDULED_EPOCH",
        f"{model_id}/seed{seed} checkpoint policy",
    )

    require_false(
        fit_manifest[
            "validation_used"
        ],
        f"{model_id}/seed{seed} validation use",
    )

    require_false(
        fit_manifest[
            "early_stopping_used"
        ],
        f"{model_id}/seed{seed} early stopping",
    )

    require_equal(
        int(
            fit_manifest[
                "physical_microbatch_size"
            ]
        ),
        EXPECTED_PHYSICAL_MICROBATCH[
            model_id
        ],
        f"{model_id}/seed{seed} manifest microbatch",
    )

    checkpoint_rec = find_artifact_by_basename(
        fit_manifest,
        "final_checkpoint.pt",
    )

    checkpoint_path = verify_artifact_record(
        checkpoint_rec,
        f"{model_id}/seed{seed} final checkpoint",
    )

    plan_rows.append(
        {
            "model_id":
                model_id,

            "config_id":
                config_id,

            "seed":
                seed,

            "scheduled_epochs":
                int(
                    row.scheduled_epochs
                ),

            "physical_inference_microbatch":
                EXPECTED_PHYSICAL_MICROBATCH[
                    model_id
                ],

            "inference_row_order":
                "reference_date_then_ticker",

            "inference_shuffle":
                False,

            "gradient_tracking":
                False,

            "probability_transform":
                "sigmoid(logit)",

            "seed_probability_retained":
                True,

            "ensemble_rule":
                "arithmetic_mean_of_seed_probabilities_42_43_44_45_46",

            "checkpoint_path":
                str(
                    checkpoint_path.resolve()
                ),

            "checkpoint_sha256":
                str(
                    checkpoint_rec[
                        "sha256"
                    ]
                ),

            "fit_manifest_path":
                str(
                    fit_manifest_path.resolve()
                ),

            "fit_manifest_sha256":
                str(
                    row.fit_manifest_sha256
                ),
        }
    )

plan_df = pd.DataFrame(
    plan_rows
).sort_values(
    [
        "model_id",
        "seed",
    ]
).reset_index(
    drop=True
)

require_equal(
    len(
        plan_df
    ),
    EXPECTED_FINAL_REFITS,
    "Final inference-plan row count",
)

for model_id in EXPECTED_SELECTED_CONFIGS:

    subset = plan_df.loc[
        plan_df[
            "model_id"
        ]
        ==
        model_id
    ]

    require_equal(
        set(
            subset[
                "seed"
            ].astype(int)
        ),
        set(
            EXPECTED_SEEDS
        ),
        f"{model_id} final inference seeds",
    )

print(
    f"Verified checkpoints     : {len(plan_df)}/40"
)
print(
    "Best-seed promotion      : PROHIBITED"
)
print(
    "Seed ensemble            : arithmetic mean of five probabilities"
)
print(
    "PASS"
)


# =============================================================================
# 6. CROSS-CHECK THRESHOLDS AGAINST REFIT CONFIGS
# =============================================================================

print()
print(
    "[6] CROSS-CHECKING FINAL MODELS AGAINST THRESHOLD CONTRACT"
)
print(
    "-" * 140
)

for model_id, config_id in EXPECTED_SELECTED_CONFIGS.items():

    threshold_configs = set(
        threshold_df.loc[
            threshold_df[
                "model_id"
            ]
            ==
            model_id,
            "config_id",
        ].astype(str)
    )

    plan_configs = set(
        plan_df.loc[
            plan_df[
                "model_id"
            ]
            ==
            model_id,
            "config_id",
        ].astype(str)
    )

    require_equal(
        threshold_configs,
        {
            config_id
        },
        f"{model_id} Step-32 config",
    )

    require_equal(
        plan_configs,
        {
            config_id
        },
        f"{model_id} Step-34 config",
    )

print(
    "All 8 architecture/config mappings agree."
)
print(
    "PASS"
)


# =============================================================================
# 7. FREEZE THE COMPLETE FINAL-TEST CONTRACT
# =============================================================================

print()
print(
    "[7] FREEZING FINAL-TEST MATERIALIZATION / INFERENCE / STATISTICAL CONTRACT"
)
print(
    "-" * 140
)

source_sha = sha256_file(
    SCRIPT_FILE
)

threshold_contract_rows = (
    threshold_df[
        [
            "model_id",
            "config_id",
            "horizon_days",
            "threshold_operator",
            "selected_threshold",
            "selected_balanced_accuracy",
            "fixed_050_balanced_accuracy",
            "balanced_accuracy_delta_vs_050",
            "final_test_used",
        ]
    ]
    .copy()
    .sort_values(
        [
            "model_id",
            "horizon_days",
        ]
    )
    .to_dict(
        orient="records"
    )
)

contract_core = {

    "schema_version":
        "1.0",

    "step":
        {
            "step_number":
                35,

            "step_name":
                "FINAL_TEST_EVALUATION_CONTRACT_FREEZE",
        },

    "source_script_sha256":
        source_sha,

    "upstream_freezes":
        {
            "step14_manifest_sha256":
                EXPECTED_STEP14_MANIFEST_SHA,

            "step32_manifest_sha256":
                EXPECTED_STEP32_MANIFEST_SHA,

            "step32_threshold_csv_sha256":
                EXPECTED_STEP32_THRESHOLD_SHA,

            "step34_manifest_sha256":
                EXPECTED_STEP34_MANIFEST_SHA,

            "step34_global_fingerprint":
                EXPECTED_STEP34_GLOBAL_FP,
        },

    "final_test_seal_metadata_only":
        {
            "sealed_path":
                sealed_test_path_string,

            "expected_sha256":
                sealed_test_expected_sha,

            "first_reference_date":
                seal[
                    "first_reference_date"
                ],

            "last_reference_date":
                seal[
                    "last_reference_date"
                ],

            "outer_reference_rows":
                EXPECTED_TEST_REFERENCE_ROWS,

            "outer_reference_dates":
                EXPECTED_TEST_REFERENCE_DATES,

            "stock_count":
                EXPECTED_STOCK_COUNT,

            "expected_common_multitask_rows":
                EXPECTED_TEST_COMMON_ROWS,

            "expected_common_multitask_dates":
                EXPECTED_TEST_COMMON_DATES,

            "parquet_opened_by_step35":
                False,

            "parquet_hashed_by_step35":
                False,

            "class_distribution_inspected_by_step35":
                False,
        },

    "predictor_preprocessing_contract":
        {
            "ordered_feature_count":
                FEATURE_COUNT,

            "ordered_features":
                features,

            "scaler_source":
                "Step-11 final-development predictor scaler",

            "scaler_path":
                str(
                    scaler_path.resolve()
                ),

            "scaler_sha256":
                str(
                    scaler_rec[
                        "sha256"
                    ]
                ),

            "fit_scope":
                "final-development predictor observations only",

            "fit_on_final_test":
                False,

            "refit_on_final_test":
                False,

            "transform_only":
                True,

            "feature_range":
                [
                    0.0,
                    1.0,
                ],

            "clip":
                False,

            "out_of_range_values":
                "retain transformed values outside [0,1]; do not clip",

            "target_columns_used_for_predictor_scaling":
                False,
        },

    "volatility_regime_contract":
        {
            "threshold_source":
                "Step-12 final-development per-stock q33/q67 thresholds",

            "threshold_artifact":
                str(
                    threshold_state_path.resolve()
                ),

            "threshold_artifact_sha256":
                str(
                    threshold_state_rec[
                        "sha256"
                    ]
                ),

            "threshold_records":
                EXPECTED_STOCK_COUNT,

            "threshold_fit_scope":
                "final-development reference observations only",

            "threshold_unit":
                "stock",

            "regime_source":
                "unscaled frozen 20-session volatility predictor",

            "test_used_for_threshold_fit":
                False,

            "threshold_refit_on_test":
                False,

            "regime_ids":
                {
                    "0":
                        "low",

                    "1":
                        "medium",

                    "2":
                        "high",
                },

            "test_regime_threshold_fit_scope_label":
                "final_development",
        },

    "test_sequence_materialization_contract":
        {
            "lookback_sessions":
                LOOKBACK,

            "sequence_formula":
                "[x_(t-59), ..., x_t]",

            "endpoint_source":
                "Step-10 final-test rows with eligible_common_multitask=True",

            "expected_endpoint_rows":
                EXPECTED_TEST_COMMON_ROWS,

            "expected_endpoint_dates":
                EXPECTED_TEST_COMMON_DATES,

            "expected_stocks_per_date":
                EXPECTED_STOCK_COUNT,

            "history_sources":
                [
                    "Step-12 frozen final-development scaled/regime panel",
                    "Step-36 transformed/regime-labelled final-test rows",
                ],

            "final_development_history_panel":
                str(
                    STEP12_FINAL_PANEL.resolve()
                ),

            "final_development_history_sha256":
                EXPECTED_STEP12_FINAL_PANEL_SHA,

            "history_rule":
                (
                    "For each final-test endpoint, use only same-ticker, "
                    "same-continuity-segment observations dated <= reference_date."
                ),

            "cross_boundary_lookback_into_final_development":
                True,

            "maximum_pre_test_history_rows":
                LOOKBACK - 1,

            "same_ticker_required":
                True,

            "same_continuity_segment_required":
                True,

            "consecutive_market_session_ids_required":
                True,

            "future_feature_rows_permitted":
                False,

            "target_dates_must_remain_inside_final_test_partition":
                True,

            "sort_order":
                [
                    "reference_date",
                    "ticker",
                ],

            "array_row_index":
                "0..n-1 after frozen sort order",

            "continuous_array":
                {
                    "name":
                        "X_full",

                    "shape":
                        [
                            EXPECTED_TEST_COMMON_ROWS,
                            LOOKBACK,
                            FEATURE_COUNT,
                        ],

                    "dtype":
                        "float32",
                },

            "direction_array":
                {
                    "name":
                        "y_direction",

                    "shape":
                        [
                            EXPECTED_TEST_COMMON_ROWS,
                            OUTPUT_COUNT,
                        ],

                    "order":
                        [
                            "1d",
                            "3d",
                            "5d",
                        ],

                    "dtype":
                        "int64",
                },

            "stock_id_dtype":
                "int64",

            "regime_id_dtype":
                "int64",

            "auxiliary_targets_required_for_final_inference":
                False,

            "final_test_auxiliary_scaler_fit":
                False,

            "final_test_auxiliary_transform":
                False,
        },

    "inference_contract":
        {
            "architectures":
                list(
                    EXPECTED_SELECTED_CONFIGS.keys()
                ),

            "selected_configs":
                EXPECTED_SELECTED_CONFIGS,

            "seeds":
                list(
                    EXPECTED_SEEDS
                ),

            "seed_models_per_architecture":
                5,

            "total_checkpoint_inferences":
                EXPECTED_FINAL_REFITS,

            "model_construction":
                "reuse frozen Step-26 build_candidate_model contract",

            "checkpoint_source":
                "Step-34 final scheduled-epoch checkpoints",

            "checkpoint_loading":
                "strict=True model_state_dict",

            "evaluation_mode":
                True,

            "gradient_tracking":
                False,

            "shuffle":
                False,

            "row_order":
                "reference_date_then_ticker",

            "numeric_dtype":
                "float32",

            "AMP":
                False,

            "TF32":
                False,

            "physical_inference_microbatch_by_architecture":
                EXPECTED_PHYSICAL_MICROBATCH,

            "incomplete_last_microbatch":
                "retain exact remainder; no drop, pad, duplication, or reordering",

            "probability_from_seed_model":
                "sigmoid(logit)",

            "seed_probability_ensemble":
                "arithmetic mean of probabilities from seeds 42,43,44,45,46",

            "mean_logits_before_sigmoid":
                False,

            "best_seed_selection":
                False,

            "architecture_selection_on_test":
                False,

            "PYTHONDONTWRITEBYTECODE":
                1,

            "timesnet_special_semantics":
                {
                    "physical_microbatch":
                        64,

                    "reason":
                        (
                            "Official TimesNet FFT_for_Period is batch-dependent; "
                            "final-test inference must preserve frozen sequential "
                            "physical batch size 64 and the exact frozen row order."
                        ),

                    "batch_size_equivalence_claim":
                        False,
                },

            "mlf_special_semantics":
                {
                    "physical_microbatch":
                        32,

                    "reason":
                        (
                            "MLF frozen source configuration binds runtime bs to "
                            "the physical microbatch; inference reconstruction must "
                            "use the same frozen value."
                        ),
                },
        },

    "probability_threshold_contract":
        {
            "source":
                "Step-32 pooled development OOF five-seed mean probabilities",

            "separate_by_architecture_and_horizon":
                True,

            "operator":
                ">=",

            "final_test_threshold_optimization":
                False,

            "final_test_threshold_change":
                False,

            "thresholds":
                threshold_contract_rows,
        },

    "final_test_metric_contract":
        {
            "evaluation_unit":
                "architecture x horizon using five-seed ensemble probability",

            "point_metric_rows":
                24,

            "positive_class":
                1,

            "primary_metric":
                "balanced_accuracy",

            "reported_metrics":
                [
                    "balanced_accuracy",
                    "roc_auc",
                    "mcc",
                    "pr_auc_average_precision",
                    "brier_score",
                    "sensitivity",
                    "specificity",
                    "predicted_positive_rate",
                    "observed_positive_rate",
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                    "n",
                ],

            "balanced_accuracy_definition":
                "0.5 * (sensitivity + specificity)",

            "roc_auc_input":
                "continuous five-seed mean probability",

            "pr_auc_definition":
                "average_precision_score on continuous probability",

            "brier_definition":
                "mean((probability - binary_target)^2)",

            "thresholded_metrics_use":
                "architecture-and-horizon Step-32 frozen threshold",

            "seed_specific_test_metrics":
                "descriptive only; never used for seed promotion",

            "macro_horizon_summary":
                "descriptive arithmetic mean across 1d,3d,5d; not a new selection criterion",
        },

    "dependence_aware_inference_contract":
        {
            "bootstrap_type":
                "circular moving block bootstrap",

            "resampling_axis":
                "ordered unique reference dates",

            "cross_section_policy":
                (
                    "When a reference date is resampled, retain all 20 stock "
                    "observations for that date together."
                ),

            "cross_sectional_dependence_preserved":
                True,

            "temporal_dependence_preserved_by_blocks":
                True,

            "primary_block_length_sessions":
                20,

            "sensitivity_block_lengths_sessions":
                [
                    10,
                    20,
                    40,
                ],

            "replications_per_block_length":
                2000,

            "bootstrap_rng_seed":
                20261013,

            "confidence_level":
                0.95,

            "confidence_interval":
                "percentile 2.5th and 97.5th percentiles",

            "primary_null_hypothesis":
                "balanced_accuracy <= 0.50",

            "primary_alternative":
                "balanced_accuracy > 0.50",

            "primary_bootstrap_test":
                (
                    "centered block-bootstrap one-sided test: "
                    "for observed delta = BA_obs - 0.50, use the bootstrap "
                    "sampling-error distribution (BA_boot - BA_obs); "
                    "p = (1 + count((BA_boot - BA_obs) >= delta)) / (B + 1)"
                ),

            "primary_multiple_testing_family":
                (
                    "all 8 architectures x 3 horizons = 24 balanced-accuracy "
                    "above-chance tests at block length 20"
                ),

            "multiple_testing_method":
                "Benjamini-Hochberg FDR",

            "FDR_alpha":
                0.05,

            "primary_significance_basis":
                "BH-adjusted q-value at block length 20",

            "block_length_10_and_40_role":
                "prespecified sensitivity analysis only",

            "architecture_pairwise_analysis":
                {
                    "pairs_per_horizon":
                        28,

                    "horizons":
                        [
                            1,
                            3,
                            5,
                        ],

                    "metric":
                        "balanced_accuracy_difference",

                    "method":
                        "same-date circular moving-block bootstrap",

                    "confidence_interval":
                        "95% percentile",

                    "inferential_role":
                        "exploratory/descriptive architecture comparison",

                    "used_for_post_test_model_promotion":
                        False,
                },
        },

    "prespecified_descriptive_test_analyses":
        {
            "volatility_regime_analysis":
                {
                    "groups":
                        [
                            0,
                            1,
                            2,
                        ],

                    "metrics":
                        [
                            "n",
                            "observed_positive_rate",
                            "balanced_accuracy",
                            "roc_auc",
                            "mcc",
                            "brier_score",
                        ],

                    "threshold":
                        "same frozen architecture-horizon threshold",

                    "role":
                        "descriptive heterogeneity analysis only",

                    "retuning":
                        False,
                },

            "stock_level_analysis":
                {
                    "groups":
                        "20 prespecified NSE stocks",

                    "metrics":
                        [
                            "n",
                            "observed_positive_rate",
                            "balanced_accuracy",
                            "roc_auc",
                            "brier_score",
                        ],

                    "role":
                        "descriptive heterogeneity analysis only",

                    "retuning":
                        False,
                },

            "calibration_analysis":
                {
                    "probabilities":
                        "five-seed ensemble probability",

                    "bins":
                        10,

                    "binning":
                        "fixed equal-width [0.0,0.1),...,[0.9,1.0]",

                    "report":
                        [
                            "bin_count",
                            "mean_predicted_probability",
                            "observed_positive_fraction",
                        ],

                    "recalibration":
                        False,
                },
        },

    "one_time_final_test_access_contract":
        {
            "allowed_only_after_step35_freeze":
                True,

            "execution_step":
                "STEP36_ONE_TIME_FINAL_TEST_MATERIALIZATION_INFERENCE_SCORING",

            "initial_lock_requirements":
                {
                    "state":
                        "LOCKED_NOT_OPENED",

                    "final_test_opened":
                        False,

                    "final_test_scored":
                        False,
                },

            "access_start_record_required_before_raw_read":
                True,

            "global_lock_open_flag_is_irreversible":
                True,

            "raw_file_access":
                (
                    "Read the sealed parquet bytes once into memory; compute "
                    "SHA-256 from the same byte buffer; decode the parquet from "
                    "that verified byte buffer."
                ),

            "expected_raw_file_sha256":
                sealed_test_expected_sha,

            "second_developmental_peek_after_open":
                False,

            "technical_resume_policy":
                (
                    "A technical rerun is permitted only to complete the exact "
                    "same frozen Step-36 code and contract after an execution "
                    "failure. No preprocessing, model, seed, batch, threshold, "
                    "metric, statistical, feature, or analysis choice may change."
                ),

            "after_success":
                {
                    "final_test_opened":
                        True,

                    "final_test_scored":
                        True,

                    "test_predictions_frozen":
                        True,

                    "test_metrics_frozen":
                        True,
                },
        },

    "irreversible_post_test_prohibitions":
        {
            "model_reselection":
                False,

            "architecture_promotion":
                False,

            "best_seed_selection":
                False,

            "seed_promotion":
                False,

            "hyperparameter_change":
                False,

            "epoch_change":
                False,

            "feature_change":
                False,

            "scaler_refit":
                False,

            "regime_threshold_refit":
                False,

            "threshold_reoptimization":
                False,

            "probability_calibration_fit_on_test":
                False,

            "retraining_after_test_metrics":
                False,

            "claim_suppression_due_to_null_results":
                False,

            "all_8_architectures_must_be_reported":
                True,
        },

    "claim_policy":
        {
            "architecture_superiority_not_presumed":
                True,

            "primary_directional_evidence":
                (
                    "final-test balanced accuracy with dependence-aware "
                    "block-bootstrap uncertainty and BH-FDR control"
                ),

            "null_or_modest_results_are_reportable":
                True,

            "test_results_may_not_trigger_model_changes":
                True,
        },

    "next_gate":
        {
            "allowed_next_step":
                "STEP36_ONE_TIME_FINAL_TEST_MATERIALIZATION_INFERENCE_SCORING",

            "final_test_allowed_before_step35_freeze":
                False,

            "final_test_allowed_after_successful_step35_freeze":
                True,
        },
}

contract_fp = canonical_json_hash(
    contract_core
)

contract_manifest = {
    **contract_core,

    "status":
        "STEP35_FINAL_TEST_EVALUATION_CONTRACT_FROZEN",

    "frozen_at_utc":
        utc_now(),

    "final_test_evaluation_contract_fingerprint":
        contract_fp,

    "protection_state":
        {
            "final_test_parquet_opened_by_step35":
                False,

            "final_test_parquet_hashed_by_step35":
                False,

            "final_test_class_distribution_inspected":
                False,

            "final_test_metric_computed":
                False,

            "final_test_state":
                "LOCKED_NOT_OPENED",

            "architecture_selection_performed":
                False,

            "best_seed_selection_performed":
                False,

            "threshold_reoptimization_performed":
                False,
        },
}

print(
    f"Contract fingerprint : {contract_fp}"
)
print(
    "PASS"
)


# =============================================================================
# 8. WRITE IMMUTABLE PLAN / THRESHOLD SNAPSHOT / SOURCE / MANIFEST
# =============================================================================

print()
print(
    "[8] WRITING IMMUTABLE STEP-35 ARTIFACTS"
)
print(
    "-" * 140
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

plan_df.to_csv(
    OUT_PLAN,
    index=False,
)

threshold_df.to_csv(
    OUT_THRESHOLDS,
    index=False,
)

shutil.copy2(
    SCRIPT_FILE,
    FROZEN_SCRIPT,
)

contract_manifest[
    "artifacts"
] = {
    OUT_PLAN.name:
        artifact_record(
            OUT_PLAN
        ),

    OUT_THRESHOLDS.name:
        artifact_record(
            OUT_THRESHOLDS
        ),

    FROZEN_SCRIPT.name:
        artifact_record(
            FROZEN_SCRIPT
        ),
}

atomic_json(
    OUT_MANIFEST,
    contract_manifest,
)

manifest_sha = sha256_file(
    OUT_MANIFEST
)

atomic_text(
    OUT_MANIFEST_SHA,
    f"{manifest_sha}  {OUT_MANIFEST.name}\n",
)

print(
    f"Inference plan       : {OUT_PLAN}"
)
print(
    f"Threshold snapshot   : {OUT_THRESHOLDS}"
)
print(
    f"Frozen source        : {FROZEN_SCRIPT}"
)
print(
    f"Contract manifest    : {OUT_MANIFEST}"
)
print(
    f"Contract manifest SHA: {manifest_sha}"
)
print(
    "PASS"
)


# =============================================================================
# 9. FINAL IMMUTABILITY / LOCK CHECK
# =============================================================================

print()
print(
    "[9] FINAL PROTECTION CHECK"
)
print(
    "-" * 140
)

require_equal(
    sha256_file(
        FROZEN_SCRIPT
    ),
    source_sha,
    "Frozen Step-35 source SHA",
)

require_equal(
    sha256_file(
        OUT_MANIFEST
    ),
    manifest_sha,
    "Step-35 manifest SHA",
)

lock_after = read_json(
    lock_path
)

require_equal(
    lock_after.get(
        "state"
    ),
    "LOCKED_NOT_OPENED",
    "Global final-test state after Step 35",
)

require_false(
    lock_after.get(
        "final_test_opened"
    ),
    "Global final_test_opened after Step 35",
)

require_false(
    lock_after.get(
        "final_test_scored"
    ),
    "Global final_test_scored after Step 35",
)

print(
    "Global final-test lock : LOCKED_NOT_OPENED"
)
print(
    "Final-test opened      : False"
)
print(
    "Final-test scored      : False"
)
print(
    "PASS"
)


# =============================================================================
# FINAL REPORT
# =============================================================================

print()
print(
    "=" * 140
)
print(
    "STEP 35 COMPLETE — FINAL-TEST EVALUATION CONTRACT FROZEN"
)
print(
    "=" * 140
)

print(
    f"Architectures                     : {EXPECTED_ARCHITECTURES}"
)
print(
    f"Fixed seed models                 : {EXPECTED_FINAL_REFITS}"
)
print(
    f"Frozen thresholds                 : {EXPECTED_THRESHOLD_ROWS}"
)
print(
    f"Expected final-test endpoints     : {EXPECTED_TEST_COMMON_ROWS:,}"
)
print(
    f"Expected final-test dates         : {EXPECTED_TEST_COMMON_DATES}"
)
print(
    f"Expected stocks/date              : {EXPECTED_STOCK_COUNT}"
)
print(
    "Probability ensemble             : MEAN OF FIVE SEED PROBABILITIES"
)
print(
    "Threshold tuning on final test   : PROHIBITED"
)
print(
    "Best-seed promotion              : PROHIBITED"
)
print(
    "Post-test model promotion        : PROHIBITED"
)
print(
    "Primary inferential metric       : BALANCED ACCURACY"
)
print(
    "Primary dependence correction    : CIRCULAR MOVING BLOCK BOOTSTRAP, L=20"
)
print(
    "Bootstrap replications           : 2,000"
)
print(
    "Sensitivity block lengths        : 10 / 20 / 40"
)
print(
    "Multiple testing                 : BH-FDR, 24 PRIMARY BA TESTS"
)
print(
    "Final-test parquet opened        : False"
)
print(
    "Final-test parquet hashed        : False"
)
print(
    "Final-test class balance seen    : False"
)
print(
    "Final-test metrics computed      : False"
)
print(
    "Final-test state                 : LOCKED_NOT_OPENED"
)
print(
    f"Contract fingerprint             : {contract_fp}"
)
print(
    f"Freeze manifest SHA256           : {manifest_sha}"
)
print(
    "=" * 140
)
print(
    "NEXT GATE: STEP36_ONE_TIME_FINAL_TEST_MATERIALIZATION_INFERENCE_SCORING"
)
print(
    "DO NOT OPEN THE FINAL TEST MANUALLY."
)
print(
    "=" * 140
)
