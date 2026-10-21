#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 11 — UNIFIED TRAIN-ONLY MINMAX PREPROCESSING + VALIDATION EXTRAPOLATION
(CLEAN V3)

This replaces the former Step 11 and Step 11B.

Policy
------
- 63 frozen predictors.
- MinMaxScaler(feature_range=(0,1), clip=False).
- One scaler per expanding-origin fold, fitted on TRAINING REFERENCE predictors only.
- Validation is transformed with the training-fitted scaler; no clipping/refitting.
- The former Step-11B validation extrapolation diagnostic is integrated here.
- One final-development scaler is fitted on DEVELOPMENT REFERENCE predictors only.
- The sealed final-test parquet is NEVER opened, hashed, transformed, or inspected.
- No regimes, feature selection, sequences, thresholds, or models are created.

Run:
    python step11_unified_preprocessing_v3.py

Verify:
    python step11_unified_preprocessing_v3.py --verify-only
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

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent
if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError(
        "step11_unified_preprocessing_v3.py must be inside <repository>/q1_revision/."
    )
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import step00_config as cfg


# =============================================================================
# 1. CONSTANTS
# =============================================================================

STEP_NUMBER = 11
STEP_ID = "11"
STEP_NAME = "STEP 11 — UNIFIED TRAIN-ONLY MINMAX PREPROCESSING AND VALIDATION EXTRAPOLATION"
SCHEMA_VERSION = "1.0"

FOLDS = (1, 2, 3, 4, 5)
EXPECTED_FEATURE_COUNT = 63
EXPECTED_STOCK_COUNT = len(cfg.STOCK_TICKERS)
FEATURE_RANGE = (0.0, 1.0)
SCALER_CLIP = False
TRAIN_SCALE_TOL = 1e-10
INVERSE_NORM_TOL = 1e-12

STEP01_ENV = cfg.ENVIRONMENT_DIR / "01_environment_audit.json"

MANIFESTS = {
    1: (cfg.MANIFEST_DIR / "01_environment_audit_manifest.json",
        cfg.MANIFEST_DIR / "01_environment_audit_manifest.sha256"),
    2: (cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.json",
        cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.sha256"),
    3: (cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.json",
        cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.sha256"),
    4: (cfg.MANIFEST_DIR / "04_session_policy_manifest.json",
        cfg.MANIFEST_DIR / "04_session_policy_manifest.sha256"),
    5: (cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.json",
        cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.sha256"),
    6: (cfg.MANIFEST_DIR / "06_stock_feature_manifest.json",
        cfg.MANIFEST_DIR / "06_stock_feature_manifest.sha256"),
    7: (cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.json",
        cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.sha256"),
    8: (cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.json",
        cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.sha256"),
    9: (cfg.MANIFEST_DIR / "09_target_manifest.json",
        cfg.MANIFEST_DIR / "09_target_manifest.sha256"),
    10: (cfg.MANIFEST_DIR / "10_chronological_split_manifest.json",
         cfg.MANIFEST_DIR / "10_chronological_split_manifest.sha256"),
}

STEP10_MEMBERSHIP = (
    cfg.FEATURE_DATA_DIR / "step10_chronological_splits" / "10_split_membership.parquet"
)
STEP10_DEVELOPMENT = (
    cfg.FEATURE_DATA_DIR / "step10_chronological_splits" / "development"
    / "10_development_target_panel.parquet"
)
STEP10_TEST_SEAL = cfg.MANIFEST_DIR / "10_final_test_seal.json"
FORBIDDEN_TEST_NAME = "10_final_test_locked_panel.parquet"

OUT_DIR = cfg.FEATURE_DATA_DIR / "step11_unified_preprocessing"
FINAL_DEV_DIR = OUT_DIR / "final_development"

FOLD_SUMMARY_CSV = cfg.AUDIT_DIR / "11_fold_preprocessing_summary.csv"
SCALER_AUDIT_CSV = cfg.AUDIT_DIR / "11_scaler_parameter_audit.csv"
VAL_RANGE_CSV = cfg.AUDIT_DIR / "11_validation_scaled_range_audit.csv"
VAL_RANKED_CSV = cfg.AUDIT_DIR / "11_validation_extrapolation_ranked.csv"
VAL_FOLD_CSV = cfg.AUDIT_DIR / "11_validation_extrapolation_fold_summary.csv"
INVERSE_CSV = cfg.AUDIT_DIR / "11_inverse_transform_audit.csv"
FINAL_DEV_AUDIT_CSV = cfg.AUDIT_DIR / "11_final_development_scaler_audit.csv"
SOURCE_HASH_CSV = cfg.AUDIT_DIR / "11_source_file_hashes.csv"
HASH_REGISTER_CSV = cfg.AUDIT_DIR / "11_preprocessing_hashes.csv"
SUMMARY_JSON = cfg.AUDIT_DIR / "11_preprocessing_summary.json"

STEP11_MANIFEST = cfg.MANIFEST_DIR / "11_preprocessing_manifest.json"
STEP11_SHA = cfg.MANIFEST_DIR / "11_preprocessing_manifest.sha256"
FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step11_unified_preprocessing_frozen.py"
SUMMARY_TXT = cfg.REPORT_DIR / "11_preprocessing_summary.txt"
FAILED_DIR = cfg.AUDIT_DIR / "step11_failed_attempts"


# =============================================================================
# 2. HELPERS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jdefault(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if x is pd.NA:
        return None
    raise TypeError(type(x).__name__)


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(
            payload, f, indent=2, sort_keys=True,
            ensure_ascii=False, default=jdefault
        )
        f.write("\n")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(
        tmp, index=False, date_format="%Y-%m-%d", float_format="%.17g"
    )
    os.replace(tmp, path)


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(cfg.EXPERIMENT_ROOT_DIR.resolve()))


def artifact(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": rel(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def pkgver(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


# =============================================================================
# 3. VERIFY STEP 00 / STEP 01-10
# =============================================================================

def verify_step00() -> Dict[str, Any]:
    cfg.validate_configuration()
    payload = cfg.build_configuration_payload()
    fp = cfg.canonical_json_sha256(payload)
    manifest_path, manifest_sha, status = cfg.verify_existing_freeze(
        configuration_payload=payload,
        configuration_fingerprint=fp,
    )
    if status != "VERIFIED_EXISTING":
        raise RuntimeError("Step-00 freeze is not VERIFIED_EXISTING.")

    lock_path = cfg.get_final_test_lock_path()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("state") != "LOCKED_NOT_OPENED":
        raise RuntimeError("Global final-test state is not LOCKED_NOT_OPENED.")
    if lock.get("final_test_opened") is not False:
        raise RuntimeError("global final_test_opened != False.")
    if lock.get("final_test_scored") is not False:
        raise RuntimeError("global final_test_scored != False.")

    return {
        "fingerprint": fp,
        "step00_manifest_sha": manifest_sha,
        "lock_path": str(lock_path),
        "lock_sha": sha256_file(lock_path),
        "state": lock["state"],
    }


def verify_sidecar(manifest: Path, sidecar: Path, label: str) -> str:
    if not manifest.exists() or not sidecar.exists():
        raise RuntimeError(f"{label} manifest/sidecar missing.")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    observed = sha256_file(manifest)
    if expected != observed:
        raise RuntimeError(
            f"{label} manifest SHA mismatch. expected={expected}, observed={observed}"
        )
    return observed


def find_artifact(manifest: Dict[str, Any], filename: str) -> Dict[str, Any]:
    matches = []
    for key in ("artifacts", "canonical_artifacts"):
        for rec in manifest.get(key, {}).values():
            if not isinstance(rec, dict) or not rec.get("path"):
                continue
            if Path(rec["path"]).name == filename:
                matches.append(rec)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one manifest artifact named {filename}; found {len(matches)}."
        )
    return matches[0]


def verify_environment() -> Dict[str, Any]:
    if not STEP01_ENV.exists():
        raise RuntimeError(f"Step-01 environment audit missing: {STEP01_ENV}")
    audit = json.loads(STEP01_ENV.read_text(encoding="utf-8"))
    expected_python = audit.get("system", {}).get("python_executable")
    if expected_python:
        if Path(expected_python).resolve() != Path(sys.executable).resolve():
            raise RuntimeError(
                "Current Python differs from frozen Step-01 environment.\n"
                f"Frozen : {expected_python}\nCurrent: {sys.executable}"
            )

    drift = []
    for row in audit.get("packages", []):
        if not row.get("required", False):
            continue
        current = pkgver(row.get("distribution"))
        if current != row.get("version"):
            drift.append(
                (row.get("distribution"), row.get("version"), current)
            )
    if drift:
        raise RuntimeError(
            "Required-package drift:\n"
            + "\n".join(
                f"{name}: frozen={old!r}, current={new!r}"
                for name, old, new in drift
            )
        )

    return {
        "python_executable": expected_python,
        "sklearn_version": sklearn.__version__,
        "joblib_version": pkgver("joblib"),
    }


def verify_upstream(step00: Dict[str, Any]) -> Dict[str, Any]:
    shas: Dict[int, str] = {}
    manifests: Dict[int, Dict[str, Any]] = {}

    for step in range(1, 11):
        mpath, spath = MANIFESTS[step]
        shas[step] = verify_sidecar(mpath, spath, f"Step {step:02d}")
        manifests[step] = json.loads(mpath.read_text(encoding="utf-8"))

    if manifests[1].get("status") != "PASS":
        raise RuntimeError("Step-01 status is not PASS.")
    if manifests[1].get("configuration_fingerprint_sha256") != step00["fingerprint"]:
        raise RuntimeError("Step-01 fingerprint mismatch.")

    if manifests[2].get("step", {}).get("status") != "PASS":
        raise RuntimeError("Step-02 status is not PASS.")
    if (
        manifests[2].get("step00", {}).get("configuration_fingerprint_sha256")
        != step00["fingerprint"]
    ):
        raise RuntimeError("Step-02 fingerprint mismatch.")
    if (
        manifests[2].get("step01", {}).get("step01_manifest_sha256")
        != shas[1]
    ):
        raise RuntimeError("Step-02 -> Step-01 chain mismatch.")

    for step in range(3, 11):
        m = manifests[step]
        if m.get("step", {}).get("status") != "PASS":
            raise RuntimeError(f"Step-{step:02d} status is not PASS.")
        if m.get("configuration_fingerprint_sha256") != step00["fingerprint"]:
            raise RuntimeError(f"Step-{step:02d} fingerprint mismatch.")
        for prev in range(1, step):
            key = f"step{prev:02d}_manifest_sha256"
            if key in m and m[key] != shas[prev]:
                raise RuntimeError(f"Step-{step:02d} chain mismatch: {key}")

    env = verify_environment()

    step10 = manifests[10]
    feature_columns = step10.get("ordered_63_features")
    interface_sha = step10.get("feature_interface_sha256")
    split_sha = step10.get("split_design_sha256")

    if not isinstance(feature_columns, list) or len(feature_columns) != 63:
        raise RuntimeError("Step-10 manifest does not freeze exactly 63 predictors.")
    if not interface_sha or not split_sha:
        raise RuntimeError("Step-10 manifest lacks interface/split-design SHA.")

    if manifests[8].get("ordered_63_features") != feature_columns:
        raise RuntimeError("Step-08/10 feature order mismatch.")
    if manifests[9].get("ordered_63_features") != feature_columns:
        raise RuntimeError("Step-09/10 feature order mismatch.")
    if manifests[8].get("feature_interface_sha256") != interface_sha:
        raise RuntimeError("Step-08/10 interface SHA mismatch.")
    if manifests[9].get("feature_interface_sha256") != interface_sha:
        raise RuntimeError("Step-09/10 interface SHA mismatch.")

    memrec = find_artifact(step10, STEP10_MEMBERSHIP.name)
    devrec = find_artifact(step10, STEP10_DEVELOPMENT.name)
    if Path(memrec["path"]).resolve() != STEP10_MEMBERSHIP.resolve():
        raise RuntimeError("Step-10 membership path is not canonical.")
    if Path(devrec["path"]).resolve() != STEP10_DEVELOPMENT.resolve():
        raise RuntimeError("Step-10 development path is not canonical.")

    memsha = sha256_file(STEP10_MEMBERSHIP)
    devsha = sha256_file(STEP10_DEVELOPMENT)
    if memsha != memrec["sha256"]:
        raise RuntimeError("Step-10 membership SHA mismatch.")
    if devsha != devrec["sha256"]:
        raise RuntimeError("Step-10 development SHA mismatch.")

    # IMPORTANT: only read seal metadata. Never open/hash/stat the final-test parquet.
    if not STEP10_TEST_SEAL.exists():
        raise RuntimeError("Step-10 final-test seal missing.")
    seal = json.loads(STEP10_TEST_SEAL.read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED_NOT_OPENED":
        raise RuntimeError("Step-10 final-test seal is not SEALED_NOT_OPENED.")
    if seal.get("global_final_test_lock_state") != "LOCKED_NOT_OPENED":
        raise RuntimeError("Step-10 seal does not preserve LOCKED_NOT_OPENED.")

    return {
        "shas": shas,
        "manifests": manifests,
        "feature_columns": list(feature_columns),
        "feature_interface_sha": interface_sha,
        "split_design_sha": split_sha,
        "membership_sha": memsha,
        "development_sha": devsha,
        "test_seal_status": seal["status"],
        "sealed_test_sha_metadata_only": seal.get("sha256"),
        "environment": env,
    }


# =============================================================================
# 4. IMMUTABILITY / LOAD SOURCES
# =============================================================================

def outputs() -> tuple[Path, ...]:
    return (
        OUT_DIR, FOLD_SUMMARY_CSV, SCALER_AUDIT_CSV, VAL_RANGE_CSV,
        VAL_RANKED_CSV, VAL_FOLD_CSV, INVERSE_CSV, FINAL_DEV_AUDIT_CSV,
        SOURCE_HASH_CSV, HASH_REGISTER_CSV, SUMMARY_JSON, STEP11_MANIFEST,
        STEP11_SHA, FROZEN_SCRIPT, SUMMARY_TXT,
    )


def assert_fresh() -> None:
    collisions = [p for p in outputs() if p.exists()]
    if collisions:
        raise RuntimeError(
            "Step-11 outputs already exist; refusing to overwrite.\n"
            + "\n".join(str(p) for p in collisions)
            + "\nUse --verify-only if Step 11 already passed."
        )


def create_stage() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = cfg.FEATURE_DATA_DIR / f"_step11_staging_{stamp}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    for fold in FOLDS:
        (root / f"fold_{fold}").mkdir()
    (root / "final_development").mkdir()
    return root


def archive_partial() -> Optional[Path]:
    if STEP11_MANIFEST.exists():
        return None
    existing = [p for p in outputs() if p.exists()]
    if not existing:
        return None
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = FAILED_DIR / f"partial_artifacts_{stamp}"
    dest.mkdir(exist_ok=False)
    for p in existing:
        try:
            shutil.move(str(p), str(dest / p.name))
        except Exception:
            pass
    return dest


def load_sources(feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = pd.read_parquet(STEP10_DEVELOPMENT, engine="pyarrow")
    membership = pd.read_parquet(STEP10_MEMBERSHIP, engine="pyarrow")

    for df in (development, membership):
        df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.normalize()

    for name, df in (("development", development), ("membership", membership)):
        if df[["Ticker", "Date"]].duplicated().any():
            raise RuntimeError(f"Duplicate ticker-date keys in {name}.")

    missing = [f for f in feature_columns if f not in development.columns]
    if missing:
        raise RuntimeError(f"Development panel missing predictors: {missing}")
    if development[feature_columns].isna().any().any():
        raise RuntimeError("Development panel contains predictor nulls.")
    if np.isinf(development[feature_columns].to_numpy(dtype=float)).any():
        raise RuntimeError("Development panel contains infinite predictors.")

    if set(development["Ticker"].unique()) != set(cfg.STOCK_TICKERS):
        raise RuntimeError("Development stock universe differs from Step 00.")

    mem_dev = membership.loc[membership["outer_partition"] == "development"]
    left = development[["Ticker", "Date"]].sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)
    right = mem_dev[["Ticker", "Date"]].sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("Development and membership development keys differ.")

    if not (
        development.groupby("Date")["Ticker"].nunique() == EXPECTED_STOCK_COUNT
    ).all():
        raise RuntimeError("Development cross-section is incomplete.")

    return (
        development.sort_values(["Date", "Ticker"]).reset_index(drop=True),
        membership.sort_values(["Date", "Ticker"]).reset_index(drop=True),
    )


# =============================================================================
# 5. EXTRACT PARTITIONS
# =============================================================================

def extract_partition(
    development: pd.DataFrame,
    membership: pd.DataFrame,
    reference_col: str,
    eligibility_prefix: str,
    label: str,
    fold: Optional[int],
) -> pd.DataFrame:
    needed = [
        "Ticker", "Date", reference_col,
        f"{eligibility_prefix}_eligible_1d",
        f"{eligibility_prefix}_eligible_3d",
        f"{eligibility_prefix}_eligible_5d",
        f"{eligibility_prefix}_common_multitask",
    ]
    missing = [c for c in needed if c not in membership.columns]
    if missing:
        raise RuntimeError(f"{label}: missing membership columns: {missing}")

    flags = membership.loc[membership[reference_col], needed].copy()
    flags = flags.rename(
        columns={
            f"{eligibility_prefix}_eligible_1d": "eligible_1d",
            f"{eligibility_prefix}_eligible_3d": "eligible_3d",
            f"{eligibility_prefix}_eligible_5d": "eligible_5d",
            f"{eligibility_prefix}_common_multitask": "eligible_common_multitask",
        }
    ).drop(columns=[reference_col])

    result = development.merge(
        flags, on=["Ticker", "Date"], how="inner", validate="one_to_one"
    ).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if len(result) != len(flags) or result.empty:
        raise RuntimeError(f"{label}: partition extraction mismatch/empty.")
    if not (
        result.groupby("Date")["Ticker"].nunique() == EXPECTED_STOCK_COUNT
    ).all():
        raise RuntimeError(f"{label}: incomplete stock cross-section.")

    result["preprocessing_fold"] = pd.NA if fold is None else int(fold)
    result["preprocessing_partition"] = label
    return result


def fold_partition(
    development: pd.DataFrame,
    membership: pd.DataFrame,
    fold: int,
    partition: str,
) -> pd.DataFrame:
    return extract_partition(
        development,
        membership,
        f"fold_{fold}_{partition}_reference",
        f"fold_{fold}_{partition}",
        partition,
        fold,
    )


def final_development_partition(
    development: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    m = membership.copy()
    m["_final_development_reference"] = m["outer_partition"].eq("development")
    return extract_partition(
        development,
        m,
        "_final_development_reference",
        "outer_development",
        "final_development",
        None,
    )


# =============================================================================
# 6. SCALING / AUDITS
# =============================================================================

def fit_scaler(df: pd.DataFrame, features: list[str], label: str) -> MinMaxScaler:
    scaler = MinMaxScaler(
        feature_range=FEATURE_RANGE,
        copy=True,
        clip=SCALER_CLIP,
    )
    scaler.fit(df[features].astype(np.float64))

    if scaler.n_features_in_ != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(f"{label}: scaler feature-count mismatch.")
    if hasattr(scaler, "feature_names_in_"):
        if list(scaler.feature_names_in_) != features:
            raise RuntimeError(f"{label}: scaler feature order mismatch.")

    constant = np.flatnonzero(np.asarray(scaler.data_range_) == 0.0)
    if len(constant):
        raise RuntimeError(
            f"{label}: constant predictors: {[features[i] for i in constant]}"
        )
    return scaler


def transform(
    df: pd.DataFrame,
    features: list[str],
    scaler: MinMaxScaler,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    original = df[features].to_numpy(dtype=np.float64)
    scaled = scaler.transform(df[features].astype(np.float64))
    if not np.isfinite(scaled).all():
        raise RuntimeError("Scaled values contain non-finite values.")
    output = df.copy()
    output.loc[:, features] = scaled
    return output, original, scaled


def validate_train_scale(
    scaled: np.ndarray,
    features: list[str],
    label: str,
) -> None:
    mn = scaled.min(axis=0)
    mx = scaled.max(axis=0)
    bad_min = np.flatnonzero(
        ~np.isclose(mn, 0.0, rtol=0.0, atol=TRAIN_SCALE_TOL)
    )
    bad_max = np.flatnonzero(
        ~np.isclose(mx, 1.0, rtol=0.0, atol=TRAIN_SCALE_TOL)
    )
    if len(bad_min):
        raise RuntimeError(
            f"{label}: scaled minima not zero: {[features[i] for i in bad_min]}"
        )
    if len(bad_max):
        raise RuntimeError(
            f"{label}: scaled maxima not one: {[features[i] for i in bad_max]}"
        )


def inverse_audit(
    scaler: MinMaxScaler,
    original: np.ndarray,
    scaled: np.ndarray,
    features: list[str],
    scope: str,
    fold: Optional[int],
    partition: str,
) -> Dict[str, Any]:
    reconstructed = scaler.inverse_transform(scaled)
    diff = np.abs(reconstructed - original)
    scale = np.maximum.reduce(
        [
            np.ones(original.shape[1]),
            np.max(np.abs(original), axis=0),
            np.asarray(scaler.data_range_, dtype=np.float64),
        ]
    )
    norm = diff / scale.reshape(1, -1)
    max_abs = float(diff.max())
    max_norm = float(norm.max())
    flat = int(np.argmax(norm))
    row, feat = np.unravel_index(flat, norm.shape)

    if max_norm > INVERSE_NORM_TOL:
        raise RuntimeError(
            f"{scope}: inverse-transform audit failed; "
            f"max_norm={max_norm:.3e}, feature={features[feat]}, row={row}"
        )

    return {
        "scope": scope,
        "fold": np.nan if fold is None else int(fold),
        "partition": partition,
        "rows": int(original.shape[0]),
        "features": int(original.shape[1]),
        "allclose": True,
        "maximum_absolute_difference": max_abs,
        "maximum_normalized_difference": max_norm,
        "normalized_tolerance": INVERSE_NORM_TOL,
        "worst_feature_index": int(feat),
        "worst_feature_name": features[feat],
        "worst_row_index": int(row),
    }


def scaler_audit_rows(
    scaler: MinMaxScaler,
    features: list[str],
    scope: str,
    fold: Optional[int],
    fit_df: pd.DataFrame,
) -> list[dict]:
    rows = []
    for i, feature in enumerate(features):
        rows.append(
            {
                "scope": scope,
                "fold": np.nan if fold is None else int(fold),
                "feature_number": i + 1,
                "feature": feature,
                "fit_rows": int(len(fit_df)),
                "fit_dates": int(fit_df["Date"].nunique()),
                "first_fit_date": fit_df["Date"].min(),
                "last_fit_date": fit_df["Date"].max(),
                "train_data_min": float(scaler.data_min_[i]),
                "train_data_max": float(scaler.data_max_[i]),
                "train_data_range": float(scaler.data_range_[i]),
                "scaler_scale": float(scaler.scale_[i]),
                "scaler_min": float(scaler.min_[i]),
                "clip": False,
                "predictors_only_fit": True,
                "validation_used_in_fit": False,
                "final_test_used_in_fit": False,
            }
        )
    return rows


def validation_audit_rows(
    fold: int,
    scaled: np.ndarray,
    features: list[str],
) -> list[dict]:
    rows = []
    for i, feature in enumerate(features):
        v = scaled[:, i]
        mn, mx = float(v.min()), float(v.max())
        below = int((v < 0.0).sum())
        above = int((v > 1.0).sum())
        lower = max(0.0, -mn)
        upper = max(0.0, mx - 1.0)
        max_ex = max(lower, upper)
        dominant = "above_1" if upper > lower else ("below_0" if lower > 0 else "none")
        rows.append(
            {
                "fold": fold,
                "feature_number": i + 1,
                "feature": feature,
                "validation_scaled_min": mn,
                "validation_scaled_max": mx,
                "below_zero_count": below,
                "above_one_count": above,
                "outside_0_1_count": below + above,
                "lower_extrapolation": lower,
                "upper_extrapolation": upper,
                "maximum_extrapolation": max_ex,
                "has_extrapolation": bool(below + above),
                "dominant_side": dominant,
                "clipping_applied": False,
                "validation_refit_performed": False,
            }
        )
    return rows


def scaler_metadata(
    scaler: MinMaxScaler,
    features: list[str],
    fit_df: pd.DataFrame,
    scope: str,
    fold: Optional[int],
    step00: Dict[str, Any],
    upstream: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "fold": fold,
        "generated_at_utc": utc_now(),
        "configuration_fingerprint_sha256": step00["fingerprint"],
        "feature_interface_sha256": upstream["feature_interface_sha"],
        "feature_order": features,
        "scaler": {
            "class": "sklearn.preprocessing.MinMaxScaler",
            "feature_range": [0.0, 1.0],
            "clip": False,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": pkgver("joblib"),
        },
        "fit_scope": {
            "rows": int(len(fit_df)),
            "dates": int(fit_df["Date"].nunique()),
            "first_date": fit_df["Date"].min().strftime("%Y-%m-%d"),
            "last_date": fit_df["Date"].max().strftime("%Y-%m-%d"),
            "predictor_columns_only": True,
            "target_columns_used_in_fit": False,
            "validation_predictors_used_in_fit": False,
            "final_test_accessed": False,
        },
        "parameters": {
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
            "data_range": scaler.data_range_.tolist(),
            "scale": scaler.scale_.tolist(),
            "min": scaler.min_.tolist(),
        },
    }


# =============================================================================
# 7. FILE WRITING / SOURCE HASHES
# =============================================================================

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(
        path, index=False, engine="pyarrow", compression="snappy"
    )


def verify_roundtrip(df: pd.DataFrame, path: Path) -> None:
    obs = pd.read_parquet(path, engine="pyarrow")
    if len(obs) != len(df) or list(obs.columns) != list(df.columns):
        raise RuntimeError(f"Parquet roundtrip mismatch: {path}")
    left = df[["Ticker", "Date"]].copy()
    right = obs[["Ticker", "Date"]].copy()
    left["Date"] = pd.to_datetime(left["Date"]).dt.normalize()
    right["Date"] = pd.to_datetime(right["Date"]).dt.normalize()
    if not left.equals(right):
        raise RuntimeError(f"Parquet key roundtrip mismatch: {path}")


def source_hashes() -> pd.DataFrame:
    # No final-test parquet is included.
    sources = [
        ("current_step00", Path(cfg.__file__).resolve()),
        ("current_step11", SCRIPT_FILE.resolve()),
        ("step10_membership", STEP10_MEMBERSHIP.resolve()),
        ("step10_development", STEP10_DEVELOPMENT.resolve()),
        ("step10_test_seal_metadata", STEP10_TEST_SEAL.resolve()),
    ]
    for step, (mpath, spath) in MANIFESTS.items():
        sources.extend(
            [
                (f"step{step:02d}_manifest", mpath.resolve()),
                (f"step{step:02d}_manifest_sha", spath.resolve()),
            ]
        )

    rows = []
    for role, path in sources:
        if path.name == FORBIDDEN_TEST_NAME:
            raise RuntimeError("Programming error: forbidden test parquet in source list.")
        if not path.exists():
            raise RuntimeError(f"Source missing: {path}")
        rows.append(
            {
                "role": role,
                "file_name": path.name,
                "absolute_path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 8. VERIFY EXISTING STEP 11
# =============================================================================

def verify_existing() -> int:
    print("=" * 132)
    print("STEP 11 — VERIFY EXISTING UNIFIED PREPROCESSING FREEZE")
    print("=" * 132)

    step00 = verify_step00()
    upstream = verify_upstream(step00)

    manifest_sha = verify_sidecar(STEP11_MANIFEST, STEP11_SHA, "Step 11")
    manifest = json.loads(STEP11_MANIFEST.read_text(encoding="utf-8"))

    if manifest.get("step", {}).get("status") != "PASS":
        print("[FAIL] Stored Step-11 status is not PASS.")
        return 20
    if manifest.get("configuration_fingerprint_sha256") != step00["fingerprint"]:
        print("[FAIL] Configuration fingerprint mismatch.")
        return 21
    for step in range(1, 11):
        key = f"step{step:02d}_manifest_sha256"
        if manifest.get(key) != upstream["shas"][step]:
            print(f"[FAIL] Upstream chain mismatch: {key}")
            return 22
    if manifest.get("feature_interface_sha256") != upstream["feature_interface_sha"]:
        print("[FAIL] Feature-interface SHA mismatch.")
        return 23
    if manifest.get("split_design_sha256") != upstream["split_design_sha"]:
        print("[FAIL] Split-design SHA mismatch.")
        return 24

    failures = []
    for name, rec in manifest.get("artifacts", {}).items():
        path = Path(rec["path"])
        if not path.exists():
            failures.append(f"missing: {name}")
        elif sha256_file(path) != rec["sha256"]:
            failures.append(f"hash mismatch: {name}")
    if failures:
        for x in failures:
            print(f"[FAIL] {x}")
        print("=" * 132)
        print("STEP 11 VERIFICATION STATUS: FAIL")
        print("=" * 132)
        return 25

    if sha256_file(Path(step00["lock_path"])) != step00["lock_sha"]:
        print("[FAIL] Global final-test lock changed.")
        return 26

    print(f"[PASS] Configuration fingerprint : {step00['fingerprint']}")
    print(f"[PASS] Step-10 manifest          : {upstream['shas'][10]}")
    print(f"[PASS] 63-feature interface      : {upstream['feature_interface_sha']}")
    print(f"[PASS] Split-design SHA-256      : {upstream['split_design_sha']}")
    print(f"[PASS] Step-11 manifest          : {manifest_sha}")
    print(f"[PASS] Step-10 test seal         : {upstream['test_seal_status']}")
    print(f"[PASS] Global final-test state   : {step00['state']}")
    print("=" * 132)
    print("STEP 11 VERIFICATION STATUS: PASS")
    print("=" * 132)
    return 0


# =============================================================================
# 9. MAIN
# =============================================================================

def run() -> int:
    print("=" * 132)
    print(STEP_NAME)
    print("=" * 132)

    stage = "verify"
    staging: Optional[Path] = None

    try:
        print("[1/10] Verifying Step 00 and immutable Step 01 -> Step 10 chain...")
        step00 = verify_step00()
        upstream = verify_upstream(step00)

        print(f"       Configuration fingerprint: {step00['fingerprint']}")
        print(f"       Step-10 manifest          : {upstream['shas'][10]}")
        print(f"       Development SHA-256       : {upstream['development_sha']}")
        print(f"       Membership SHA-256        : {upstream['membership_sha']}")
        print(f"       63-feature interface      : {upstream['feature_interface_sha']}")
        print(f"       Split-design SHA-256      : {upstream['split_design_sha']}")
        print(f"       Step-10 test seal         : {upstream['test_seal_status']}")
        print(f"       Global final-test state   : {step00['state']}")
        print("       Final-test parquet access : PROHIBITED")

        stage = "collision"
        print("[2/10] Checking immutable Step-11 output target...")
        assert_fresh()
        cfg.create_directories()

        stage = "load"
        print("[3/10] Loading development panel and structural membership ledger...")
        features = upstream["feature_columns"]
        development, membership = load_sources(features)
        print(
            f"       Development rows={len(development):,} | "
            f"dates={development['Date'].nunique():,} | "
            f"rows/stock={development.groupby('Ticker').size().iloc[0]:,}"
        )
        print(f"       Locked predictors={len(features)} | clip={SCALER_CLIP}")

        stage = "stage"
        staging = create_stage()

        fold_summary_rows = []
        scaler_rows = []
        val_rows = []
        inverse_rows = []
        staged_records = []

        stage = "folds"
        print("[4/10] Fitting five train-only fold scalers and transforming validation...")

        for fold in FOLDS:
            train = fold_partition(development, membership, fold, "train")
            val = fold_partition(development, membership, fold, "validation")

            if train["Date"].max() >= val["Date"].min():
                raise RuntimeError(f"Fold {fold}: chronology violation.")

            scaler = fit_scaler(train, features, f"fold_{fold}")
            train_scaled, train_orig, train_arr = transform(train, features, scaler)
            val_scaled, val_orig, val_arr = transform(val, features, scaler)

            validate_train_scale(train_arr, features, f"fold_{fold}")
            inverse_rows.append(
                inverse_audit(
                    scaler, train_orig, train_arr, features,
                    f"fold_{fold}_train", fold, "train"
                )
            )
            inverse_rows.append(
                inverse_audit(
                    scaler, val_orig, val_arr, features,
                    f"fold_{fold}_validation", fold, "validation"
                )
            )
            scaler_rows.extend(
                scaler_audit_rows(scaler, features, "fold_training", fold, train)
            )
            val_rows.extend(validation_audit_rows(fold, val_arr, features))

            fdir = staging / f"fold_{fold}"
            scaler_path = fdir / f"11_fold_{fold}_minmax_scaler.joblib"
            meta_path = fdir / f"11_fold_{fold}_scaler_metadata.json"
            train_path = fdir / f"11_fold_{fold}_train_scaled.parquet"
            val_path = fdir / f"11_fold_{fold}_validation_scaled.parquet"

            joblib.dump(scaler, scaler_path, compress=3)
            atomic_json(
                meta_path,
                scaler_metadata(
                    scaler, features, train, "fold_training", fold, step00, upstream
                ),
            )
            write_parquet(train_scaled, train_path)
            verify_roundtrip(train_scaled, train_path)
            write_parquet(val_scaled, val_path)
            verify_roundtrip(val_scaled, val_path)

            below = int((val_arr < 0.0).sum())
            above = int((val_arr > 1.0).sum())
            train_dates = int(train["Date"].nunique())
            val_dates = int(val["Date"].nunique())

            fold_summary_rows.append(
                {
                    "fold": fold,
                    "train_first_date": train["Date"].min(),
                    "train_last_date": train["Date"].max(),
                    "train_dates": train_dates,
                    "train_rows_used_to_fit_scaler": int(len(train)),
                    "train_eligible_1d_rows": int(train["eligible_1d"].sum()),
                    "train_eligible_3d_rows": int(train["eligible_3d"].sum()),
                    "train_eligible_5d_rows": int(train["eligible_5d"].sum()),
                    "train_common_multitask_rows": int(
                        train["eligible_common_multitask"].sum()
                    ),
                    "validation_first_date": val["Date"].min(),
                    "validation_last_date": val["Date"].max(),
                    "validation_dates": val_dates,
                    "validation_rows_transformed": int(len(val)),
                    "validation_eligible_1d_rows": int(val["eligible_1d"].sum()),
                    "validation_eligible_3d_rows": int(val["eligible_3d"].sum()),
                    "validation_eligible_5d_rows": int(val["eligible_5d"].sum()),
                    "validation_common_multitask_rows": int(
                        val["eligible_common_multitask"].sum()
                    ),
                    "validation_values_below_zero": below,
                    "validation_values_above_one": above,
                    "validation_values_outside_0_1": below + above,
                    "scaler_clip": False,
                    "validation_used_to_fit_scaler": False,
                    "targets_used_to_fit_scaler": False,
                    "final_test_accessed": False,
                }
            )

            for scope, path in [
                ("fold_scaler", scaler_path),
                ("fold_scaler_metadata", meta_path),
                ("fold_train_scaled", train_path),
                ("fold_validation_scaled", val_path),
            ]:
                staged_records.append(
                    {
                        "scope": scope,
                        "fold": fold,
                        "file_name": path.name,
                        "sha256": sha256_file(path),
                        "size_bytes": int(path.stat().st_size),
                    }
                )

            print(
                f"       Fold {fold}: fit rows={len(train):>6,} | "
                f"train dates={train_dates:>4,} | val rows={len(val):>5,} | "
                f"val dates={val_dates:>3,} | outside[0,1]={below+above:>6,}"
            )

        stage = "diagnostic"
        print("[5/10] Integrating former Step-11B extrapolation diagnostic...")
        fold_summary = pd.DataFrame(fold_summary_rows)
        scaler_audit = pd.DataFrame(scaler_rows)
        val_audit = pd.DataFrame(val_rows)

        expected_audit_rows = len(FOLDS) * EXPECTED_FEATURE_COUNT
        if len(val_audit) != expected_audit_rows:
            raise RuntimeError(
                f"Expected {expected_audit_rows} fold-feature diagnostics; "
                f"found {len(val_audit)}."
            )
        if val_audit["clipping_applied"].any():
            raise RuntimeError("Clipping unexpectedly occurred.")
        if val_audit["validation_refit_performed"].any():
            raise RuntimeError("Validation refit unexpectedly occurred.")

        ranked = val_audit.sort_values(
            ["maximum_extrapolation", "outside_0_1_count", "fold", "feature_number"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)

        fold_diag_rows = []
        for fold in FOLDS:
            a = val_audit.loc[val_audit["fold"] == fold]
            s = fold_summary.loc[fold_summary["fold"] == fold].iloc[0]
            total = int(s["validation_rows_transformed"]) * EXPECTED_FEATURE_COUNT
            outside = int(a["outside_0_1_count"].sum())
            fold_diag_rows.append(
                {
                    "fold": fold,
                    "validation_rows": int(s["validation_rows_transformed"]),
                    "total_validation_feature_values": total,
                    "outside_0_1_values": outside,
                    "outside_percent": 100.0 * outside / total,
                    "features_with_extrapolation": int(a["has_extrapolation"].sum()),
                    "global_scaled_min": float(a["validation_scaled_min"].min()),
                    "global_scaled_max": float(a["validation_scaled_max"].max()),
                    "maximum_extrapolation": float(a["maximum_extrapolation"].max()),
                    "clipping_applied": False,
                    "validation_refit_performed": False,
                }
            )
        fold_diag = pd.DataFrame(fold_diag_rows)

        for row in fold_diag.itertuples(index=False):
            print(
                f"       Fold {row.fold}: outside={row.outside_0_1_values:>6,} "
                f"({row.outside_percent:6.3f}%) | "
                f"features={row.features_with_extrapolation:>2}/63 | "
                f"min={row.global_scaled_min:+.6f} | "
                f"max={row.global_scaled_max:+.6f} | "
                f"max-extrap={row.maximum_extrapolation:.6f}"
            )

        stage = "final_dev"
        print("[6/10] Fitting final-development scaler; final test remains untouched...")
        final_dev = final_development_partition(development, membership)
        final_scaler = fit_scaler(final_dev, features, "final_development")
        final_scaled, final_orig, final_arr = transform(
            final_dev, features, final_scaler
        )
        validate_train_scale(final_arr, features, "final_development")
        inverse_rows.append(
            inverse_audit(
                final_scaler, final_orig, final_arr, features,
                "final_development", None, "final_development"
            )
        )
        scaler_rows.extend(
            scaler_audit_rows(
                final_scaler, features, "final_development", None, final_dev
            )
        )
        scaler_audit = pd.DataFrame(scaler_rows)

        fdir = staging / "final_development"
        final_scaler_path = fdir / "11_final_development_minmax_scaler.joblib"
        final_meta_path = fdir / "11_final_development_scaler_metadata.json"
        final_scaled_path = fdir / "11_final_development_scaled.parquet"

        joblib.dump(final_scaler, final_scaler_path, compress=3)
        atomic_json(
            final_meta_path,
            scaler_metadata(
                final_scaler, features, final_dev,
                "final_development", None, step00, upstream
            ),
        )
        write_parquet(final_scaled, final_scaled_path)
        verify_roundtrip(final_scaled, final_scaled_path)

        for scope, path in [
            ("final_development_scaler", final_scaler_path),
            ("final_development_scaler_metadata", final_meta_path),
            ("final_development_scaled", final_scaled_path),
        ]:
            staged_records.append(
                {
                    "scope": scope,
                    "fold": None,
                    "file_name": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                }
            )

        final_dev_audit = pd.DataFrame(
            [
                {
                    "fit_rows": int(len(final_dev)),
                    "fit_dates": int(final_dev["Date"].nunique()),
                    "rows_per_stock": int(final_dev.groupby("Ticker").size().iloc[0]),
                    "first_date": final_dev["Date"].min(),
                    "last_date": final_dev["Date"].max(),
                    "eligible_1d_rows": int(final_dev["eligible_1d"].sum()),
                    "eligible_3d_rows": int(final_dev["eligible_3d"].sum()),
                    "eligible_5d_rows": int(final_dev["eligible_5d"].sum()),
                    "common_multitask_rows": int(
                        final_dev["eligible_common_multitask"].sum()
                    ),
                    "scaler_feature_count": int(final_scaler.n_features_in_),
                    "clip": False,
                    "predictors_only_fit": True,
                    "final_test_accessed": False,
                    "final_test_transformed": False,
                }
            ]
        )

        print(
            f"       Final-development fit rows={len(final_dev):,} | "
            f"dates={final_dev['Date'].nunique():,} | "
            f"common={int(final_dev['eligible_common_multitask'].sum()):,}"
        )

        stage = "promote"
        print("[7/10] Promoting preprocessing artifacts and writing audits...")
        if OUT_DIR.exists():
            raise RuntimeError("Canonical Step-11 output directory appeared.")
        os.replace(staging, OUT_DIR)
        staging = None

        promoted = []
        for rec in staged_records:
            if rec["fold"] is None:
                path = FINAL_DEV_DIR / rec["file_name"]
            else:
                path = OUT_DIR / f"fold_{rec['fold']}" / rec["file_name"]
            if not path.exists() or sha256_file(path) != rec["sha256"]:
                raise RuntimeError(f"Promoted artifact mismatch: {path}")
            promoted.append(
                {
                    "scope": rec["scope"],
                    "fold": rec["fold"],
                    "file_path": str(path),
                    "relative_path": rel(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": rec["sha256"],
                }
            )

        inverse_df = pd.DataFrame(inverse_rows)
        if not inverse_df["allclose"].all():
            raise RuntimeError("One or more inverse-transform audits failed.")

        write_csv(FOLD_SUMMARY_CSV, fold_summary)
        write_csv(SCALER_AUDIT_CSV, scaler_audit)
        write_csv(VAL_RANGE_CSV, val_audit)
        write_csv(VAL_RANKED_CSV, ranked)
        write_csv(VAL_FOLD_CSV, fold_diag)
        write_csv(INVERSE_CSV, inverse_df)
        write_csv(FINAL_DEV_AUDIT_CSV, final_dev_audit)
        write_csv(SOURCE_HASH_CSV, source_hashes())

        for path in [
            FOLD_SUMMARY_CSV, SCALER_AUDIT_CSV, VAL_RANGE_CSV, VAL_RANKED_CSV,
            VAL_FOLD_CSV, INVERSE_CSV, FINAL_DEV_AUDIT_CSV, SOURCE_HASH_CSV,
        ]:
            promoted.append(
                {
                    "scope": path.stem,
                    "fold": None,
                    "file_path": str(path),
                    "relative_path": rel(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )

        stage = "freeze"
        print("[8/10] Freezing Step-11 summary, source snapshot and manifest...")
        if sha256_file(Path(step00["lock_path"])) != step00["lock_sha"]:
            raise RuntimeError("Global final-test lock changed during Step 11.")

        summary = {
            "schema_version": SCHEMA_VERSION,
            "step": {
                "step_number": STEP_NUMBER,
                "step_id": STEP_ID,
                "step_name": STEP_NAME,
                "status": "PASS",
            },
            "generated_at_utc": utc_now(),
            "configuration_fingerprint_sha256": step00["fingerprint"],
            "step10_manifest_sha256": upstream["shas"][10],
            "feature_interface_sha256": upstream["feature_interface_sha"],
            "split_design_sha256": upstream["split_design_sha"],
            "policy": {
                "scaler": "MinMaxScaler",
                "feature_range": [0.0, 1.0],
                "clip": False,
                "fold_fit_scope": "all fold-training reference predictors",
                "final_development_fit_scope": "all development reference predictors",
                "labels_used_in_fit": False,
                "validation_used_in_fit": False,
                "final_test_accessed": False,
            },
            "final_development": final_dev_audit.iloc[0].to_dict(),
            "inverse_transform": {
                "checks": int(len(inverse_df)),
                "all_pass": bool(inverse_df["allclose"].all()),
                "maximum_absolute_difference": float(
                    inverse_df["maximum_absolute_difference"].max()
                ),
                "maximum_normalized_difference": float(
                    inverse_df["maximum_normalized_difference"].max()
                ),
                "normalized_tolerance": INVERSE_NORM_TOL,
            },
            "pipeline_state": {
                "fold_feature_scaling_complete": True,
                "validation_extrapolation_diagnostic_complete": True,
                "final_development_scaler_created": True,
                "final_test_transformed": False,
                "regimes_constructed": False,
                "feature_selection_applied": False,
                "sequences_constructed": False,
                "models_fitted": False,
            },
            "test_protection": {
                "step10_seal_status": upstream["test_seal_status"],
                "sealed_test_sha256_from_metadata_only": (
                    upstream["sealed_test_sha_metadata_only"]
                ),
                "sealed_test_parquet_opened_or_hashed_by_step11": False,
                "global_final_test_state": step00["state"],
            },
        }
        atomic_json(SUMMARY_JSON, summary)

        shutil.copy2(SCRIPT_FILE, FROZEN_SCRIPT)
        atomic_text(
            SUMMARY_TXT,
            "\n".join(
                [
                    STEP_NAME,
                    "=" * 100,
                    "Status: PASS",
                    f"Step-10 manifest: {upstream['shas'][10]}",
                    f"63-feature interface: {upstream['feature_interface_sha']}",
                    f"Split-design SHA-256: {upstream['split_design_sha']}",
                    "Fold scalers: 5",
                    f"Final-development fit rows: {len(final_dev)}",
                    "Final-development scaler created: True",
                    "Final-test parquet accessed: False",
                    f"Global final-test state: {step00['state']}",
                    "Regimes constructed: False",
                ]
            )
            + "\n",
        )

        for path in [SUMMARY_JSON, FROZEN_SCRIPT, SUMMARY_TXT]:
            promoted.append(
                {
                    "scope": path.stem,
                    "fold": None,
                    "file_path": str(path),
                    "relative_path": rel(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )

        write_csv(HASH_REGISTER_CSV, pd.DataFrame(promoted))

        artifact_paths = [Path(x["file_path"]) for x in promoted] + [HASH_REGISTER_CSV]

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
            "configuration_fingerprint_sha256": step00["fingerprint"],
            "step00_manifest_sha256": step00["step00_manifest_sha"],
            **{
                f"step{step:02d}_manifest_sha256": upstream["shas"][step]
                for step in range(1, 11)
            },
            "feature_interface_sha256": upstream["feature_interface_sha"],
            "ordered_63_features": features,
            "split_design_sha256": upstream["split_design_sha"],
            "source_integrity": {
                "step10_membership_sha256": upstream["membership_sha"],
                "step10_development_panel_sha256": upstream["development_sha"],
            },
            "preprocessing_policy": {
                "scaler": "MinMaxScaler",
                "feature_range": [0.0, 1.0],
                "clip": False,
                "fold_scalers": 5,
                "horizon_dependent": False,
                "labels_used_in_fit": False,
                "validation_used_in_fit": False,
                "final_test_accessed": False,
            },
            "validation_extrapolation_diagnostic": {
                "integrated_former_step11b": True,
                "audit_rows": int(len(val_audit)),
                "expected_audit_rows": expected_audit_rows,
                "clipping_applied": False,
                "validation_refit_performed": False,
            },
            "final_development_state": {
                "scaler_created": True,
                "scaled_development_panel_created": True,
                "fit_rows": int(len(final_dev)),
                "fit_dates": int(final_dev["Date"].nunique()),
                "common_multitask_rows": int(
                    final_dev["eligible_common_multitask"].sum()
                ),
                "final_test_transformed": False,
            },
            "test_protection": {
                "step10_seal_status": upstream["test_seal_status"],
                "global_final_test_state": step00["state"],
                "sealed_test_parquet_opened_or_hashed_by_step11": False,
                "test_class_distribution_inspected": False,
                "test_performance_computed": False,
            },
            "pipeline_state": {
                "fold_feature_scaling_complete": True,
                "validation_extrapolation_diagnostic_complete": True,
                "final_development_scaler_created": True,
                "regimes_constructed": False,
                "feature_selection_applied": False,
                "sequence_construction_performed": False,
                "models_fitted": False,
            },
            "artifacts": {p.name: artifact(p) for p in artifact_paths},
        }

        atomic_json(STEP11_MANIFEST, manifest)
        manifest_sha = sha256_file(STEP11_MANIFEST)
        atomic_text(STEP11_SHA, f"{manifest_sha}  {STEP11_MANIFEST.name}\n")

        stage = "validate"
        print("[9/10] Validating frozen Step-11 artifact set...")
        verify_sidecar(STEP11_MANIFEST, STEP11_SHA, "Step 11")
        for rec in manifest["artifacts"].values():
            path = Path(rec["path"])
            if not path.exists() or sha256_file(path) != rec["sha256"]:
                raise RuntimeError(f"Frozen Step-11 artifact mismatch: {path}")
        if sha256_file(Path(step00["lock_path"])) != step00["lock_sha"]:
            raise RuntimeError("Global final-test lock changed.")

        stage = "report"
        print("[10/10] Final Step-11 report...")
        print()
        print("-" * 132)
        print("PREPROCESSING POLICY")
        print("-" * 132)
        print("Scaler                       : MinMaxScaler")
        print("Feature range                : [0, 1]")
        print("clip                         : False")
        print(f"Locked predictors            : {len(features)}")
        print("Labels used to fit scaler    : False")
        print("Final-test parquet opened    : False")

        print("-" * 132)
        print("ROLLING-ORIGIN PREPROCESSING")
        print("-" * 132)
        for row in fold_summary.itertuples(index=False):
            print(
                f"Fold {row.fold}: train dates={row.train_dates:>4,} | "
                f"fit rows={row.train_rows_used_to_fit_scaler:>6,} | "
                f"train common={row.train_common_multitask_rows:>6,} | "
                f"val dates={row.validation_dates:>3,} | "
                f"val rows={row.validation_rows_transformed:>5,} | "
                f"val common={row.validation_common_multitask_rows:>5,} | "
                f"outside[0,1]={row.validation_values_outside_0_1:>6,}"
            )

        print("-" * 132)
        print("INTEGRATED VALIDATION EXTRAPOLATION DIAGNOSTIC")
        print("-" * 132)
        for row in fold_diag.itertuples(index=False):
            print(
                f"Fold {row.fold}: outside={row.outside_0_1_values:>6,} "
                f"({row.outside_percent:6.3f}%) | "
                f"features={row.features_with_extrapolation:>2}/63 | "
                f"min={row.global_scaled_min:+.6f} | "
                f"max={row.global_scaled_max:+.6f} | "
                f"max-extrap={row.maximum_extrapolation:.6f}"
            )

        print("-" * 132)
        print("FINAL-DEVELOPMENT STATE")
        print("-" * 132)
        print(f"Fit rows                     : {len(final_dev):,}")
        print(f"Fit dates                    : {final_dev['Date'].nunique():,}")
        print(
            "Common multi-task rows       : "
            f"{int(final_dev['eligible_common_multitask'].sum()):,}"
        )
        print("Final-development scaler     : CREATED")
        print("Final-test transformed       : False")

        print("-" * 132)
        print("INVERSE-TRANSFORM AUDIT")
        print("-" * 132)
        print(f"Checks performed             : {len(inverse_df)}")
        print(f"All checks pass              : {bool(inverse_df['allclose'].all())}")
        print(
            "Maximum normalized error     : "
            f"{inverse_df['maximum_normalized_difference'].max():.3e}"
        )
        print(f"Allowed normalized tolerance : {INVERSE_NORM_TOL:.1e}")

        print("-" * 132)
        print("PIPELINE STATE")
        print("-" * 132)
        print("Fold scaling                 : COMPLETE")
        print("Validation diagnostic        : INTEGRATED")
        print("Final-development scaler     : FROZEN")
        print("Regimes constructed          : False")
        print("Feature selection applied    : False")
        print("Sequences constructed        : False")
        print("Models fitted                : False")
        print("Final-test parquet accessed  : False")
        print(f"Global final-test state      : {step00['state']}")

        print("-" * 132)
        print("OUTPUT FILES")
        print("-" * 132)
        print(f"Preprocessing root    : {OUT_DIR}")
        print(f"Fold summary          : {FOLD_SUMMARY_CSV}")
        print(f"Scaler audit          : {SCALER_AUDIT_CSV}")
        print(f"Validation range      : {VAL_RANGE_CSV}")
        print(f"Ranked extrapolation  : {VAL_RANKED_CSV}")
        print(f"Extrapolation summary : {VAL_FOLD_CSV}")
        print(f"Inverse audit         : {INVERSE_CSV}")
        print(f"Final-dev audit       : {FINAL_DEV_AUDIT_CSV}")
        print(f"Manifest              : {STEP11_MANIFEST}")
        print(f"Manifest SHA-256      : {manifest_sha}")
        print("=" * 132)
        print("PASS: Five fold scalers were fitted from training-reference predictors only.")
        print("PASS: Validation was transformed without clipping or refitting.")
        print("PASS: Former Step-11B extrapolation diagnostic is integrated.")
        print("PASS: Final-development scaler was fitted from development predictors only.")
        print("PASS: Sealed final-test parquet was not opened, hashed, transformed, or inspected.")
        print("PASS: No regimes, feature selection, sequences, thresholds, or models were created.")
        print("=" * 132)
        print(
            "Do not run the old Step 11B script. "
            "Do not run Step 12 yet. Paste the complete terminal output for audit."
        )
        return 0

    except Exception as exc:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        archived = archive_partial()
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        failure = FAILED_DIR / f"11_preprocessing_FAILED_{stamp}.json"
        atomic_json(
            failure,
            {
                "step": STEP_NAME,
                "status": "FAIL",
                "failed_at_utc": utc_now(),
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "python_executable": sys.executable,
                "archived_partial_artifacts": (
                    None if archived is None else str(archived)
                ),
            },
        )
        print()
        print("=" * 132)
        print("STEP 11 FAILED — PREPROCESSING NOT FROZEN")
        print("=" * 132)
        print(f"Stage      : {stage}")
        print(f"Error type : {type(exc).__name__}")
        print(f"Error      : {exc}")
        print(f"Diagnostic : {failure}")
        if archived is not None:
            print(f"Archived partial artifacts: {archived}")
        print("=" * 132)
        print("Do not proceed to Step 12.")
        return 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return verify_existing() if args.verify_only else run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStep 11 interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print()
        print("=" * 132)
        print("STEP 11 UNHANDLED FAILURE")
        print("=" * 132)
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("=" * 132)
        sys.exit(1)
