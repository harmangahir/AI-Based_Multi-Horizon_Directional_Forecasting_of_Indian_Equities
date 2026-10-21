#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 10 — CLEAN V3 CHRONOLOGICAL SPLITS, EXPANDING-ORIGIN FOLDS,
AND HORIZON-BOUNDARY PURGING

Frozen outer design
-------------------
Development : 2016-01-01 -> 2024-12-31
Final test  : 2025-01-01 -> 2026-06-30

Five expanding-origin folds
---------------------------
1: train through 2019, validate 2020
2: train through 2020, validate 2021
3: train through 2021, validate 2022
4: train through 2022, validate 2023
5: train through 2023, validate 2024

For horizon h, a reference is eligible for a partition only when the frozen
Step-09 exact endpoint exists and target_date_h is inside the same partition.

No scaling, regime fitting, feature selection, sequence construction, model
fitting, threshold tuning, or final-test class/performance inspection occurs.

Run:
    python step10_chronological_splits_v3.py

Verify:
    python step10_chronological_splits_v3.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
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
        "step10_chronological_splits_v3.py must be inside <repository>/q1_revision/."
    )
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg


# =============================================================================
# FROZEN DESIGN
# =============================================================================

STEP_NUMBER = 10
STEP_ID = "10"
STEP_NAME = (
    "STEP 10 — CHRONOLOGICAL SPLITS, EXPANDING-ORIGIN FOLDS, "
    "AND HORIZON-BOUNDARY PURGING"
)
SCHEMA_VERSION = "1.0"

HORIZONS = (1, 3, 5)
EXPECTED_STOCK_COUNT = len(cfg.STOCK_TICKERS)
EXPECTED_FEATURE_COUNT = 63

DEVELOPMENT_START = pd.Timestamp("2016-01-01")
DEVELOPMENT_END = pd.Timestamp("2024-12-31")
FINAL_TEST_START = pd.Timestamp("2025-01-01")
FINAL_TEST_END = pd.Timestamp("2026-06-30")

FOLDS = [
    (1, "2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    (2, "2016-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    (3, "2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    (4, "2016-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    (5, "2016-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
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


# =============================================================================
# INPUTS
# =============================================================================

STEP01_MANIFEST = cfg.MANIFEST_DIR / "01_environment_audit_manifest.json"
STEP01_SHA = cfg.MANIFEST_DIR / "01_environment_audit_manifest.sha256"
STEP01_ENV = cfg.ENVIRONMENT_DIR / "01_environment_audit.json"

STEP02_MANIFEST = cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.json"
STEP02_SHA = cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.sha256"

STEP03_MANIFEST = cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.json"
STEP03_SHA = cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.sha256"

STEP04_MANIFEST = cfg.MANIFEST_DIR / "04_session_policy_manifest.json"
STEP04_SHA = cfg.MANIFEST_DIR / "04_session_policy_manifest.sha256"

STEP05_MANIFEST = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.json"
STEP05_SHA = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.sha256"

STEP06_MANIFEST = cfg.MANIFEST_DIR / "06_stock_feature_manifest.json"
STEP06_SHA = cfg.MANIFEST_DIR / "06_stock_feature_manifest.sha256"

STEP07_MANIFEST = cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.json"
STEP07_SHA = cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.sha256"

STEP08_MANIFEST = cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.json"
STEP08_SHA = cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.sha256"

STEP09_MANIFEST = cfg.MANIFEST_DIR / "09_target_manifest.json"
STEP09_SHA = cfg.MANIFEST_DIR / "09_target_manifest.sha256"

STEP09_TARGET_PANEL = (
    cfg.FEATURE_DATA_DIR
    / "step09_targets"
    / "09_multi_horizon_target_panel.parquet"
)

UPSTREAM = {
    1: (STEP01_MANIFEST, STEP01_SHA),
    2: (STEP02_MANIFEST, STEP02_SHA),
    3: (STEP03_MANIFEST, STEP03_SHA),
    4: (STEP04_MANIFEST, STEP04_SHA),
    5: (STEP05_MANIFEST, STEP05_SHA),
    6: (STEP06_MANIFEST, STEP06_SHA),
    7: (STEP07_MANIFEST, STEP07_SHA),
    8: (STEP08_MANIFEST, STEP08_SHA),
    9: (STEP09_MANIFEST, STEP09_SHA),
}


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = cfg.FEATURE_DATA_DIR / "step10_chronological_splits"
DEV_DIR = OUT_DIR / "development"
TEST_DIR = OUT_DIR / "final_test_locked"

MEMBERSHIP_PARQUET = OUT_DIR / "10_split_membership.parquet"
MEMBERSHIP_CSV = OUT_DIR / "10_split_membership.csv"
DEV_PANEL_PARQUET = DEV_DIR / "10_development_target_panel.parquet"
DEV_PANEL_CSV = DEV_DIR / "10_development_target_panel.csv"
TEST_PANEL_PARQUET = TEST_DIR / "10_final_test_locked_panel.parquet"

FOLD_DEFINITION_CSV = cfg.AUDIT_DIR / "10_fold_definitions.csv"
OUTER_AUDIT_CSV = cfg.AUDIT_DIR / "10_outer_split_purge_audit.csv"
FOLD_AUDIT_CSV = cfg.AUDIT_DIR / "10_expanding_origin_purge_audit.csv"
DEV_CLASS_CSV = cfg.AUDIT_DIR / "10_development_only_class_distribution.csv"
BOUNDARY_CSV = cfg.AUDIT_DIR / "10_boundary_purge_examples.csv"
STRUCTURE_CSV = cfg.AUDIT_DIR / "10_split_structure_audit.csv"
SOURCE_HASH_CSV = cfg.AUDIT_DIR / "10_source_file_hashes.csv"
HASH_REGISTER_CSV = cfg.AUDIT_DIR / "10_split_hashes.csv"
SUMMARY_JSON = cfg.AUDIT_DIR / "10_chronological_split_summary.json"

TEST_SEAL_JSON = cfg.MANIFEST_DIR / "10_final_test_seal.json"
MANIFEST_JSON = cfg.MANIFEST_DIR / "10_chronological_split_manifest.json"
MANIFEST_SHA = cfg.MANIFEST_DIR / "10_chronological_split_manifest.sha256"
FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step10_chronological_splits_frozen.py"
SUMMARY_TXT = cfg.REPORT_DIR / "10_chronological_split_summary.txt"
FAILED_DIR = cfg.AUDIT_DIR / "step10_failed_attempts"


# =============================================================================
# HELPERS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def canonical_json_sha256(payload: Any) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=json_default,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(
            payload,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=json_default,
        )
        f.write("\n")
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
    return str(path.resolve().relative_to(cfg.EXPERIMENT_ROOT_DIR.resolve()))


def artifact(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": relative_to_experiment(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


# =============================================================================
# STEP-00 / UPSTREAM VERIFICATION
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

    lock_path = cfg.get_final_test_lock_path()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    if lock.get("state") != "LOCKED_NOT_OPENED":
        raise RuntimeError("Final-test state is not LOCKED_NOT_OPENED.")
    if lock.get("final_test_opened") is not False:
        raise RuntimeError("final_test_opened != False.")
    if lock.get("final_test_scored") is not False:
        raise RuntimeError("final_test_scored != False.")

    return {
        "fingerprint": fingerprint,
        "manifest_sha256": manifest_sha,
        "manifest_path": str(manifest_path),
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "state": lock.get("state"),
    }


def verify_sidecar(manifest: Path, sidecar: Path, label: str) -> str:
    if not manifest.exists() or not sidecar.exists():
        raise RuntimeError(f"{label} manifest/sidecar missing.")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(manifest)
    if expected != actual:
        raise RuntimeError(
            f"{label} manifest SHA mismatch. expected={expected}, actual={actual}"
        )
    return actual


def verify_artifacts(
    manifest: Dict[str, Any],
    key: str = "artifacts",
    skip: tuple[str, ...] = (),
) -> None:
    failures = []
    for name, rec in manifest.get(key, {}).items():
        if name in skip or not isinstance(rec, dict):
            continue
        ptext = rec.get("path")
        if not ptext:
            continue
        path = Path(ptext)
        if not path.exists():
            failures.append(f"missing {name}: {path}")
            continue
        expected = rec.get("sha256")
        if expected and sha256_file(path) != expected:
            failures.append(f"hash mismatch: {name}")
    if failures:
        raise RuntimeError("\n".join(failures))


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


def verify_upstream(step00: Dict[str, Any]) -> Dict[str, Any]:
    shas: Dict[int, str] = {}
    manifests: Dict[int, Dict[str, Any]] = {}

    for step in range(1, 10):
        manifest_path, sidecar = UPSTREAM[step]
        shas[step] = verify_sidecar(
            manifest_path,
            sidecar,
            f"Step {step:02d}",
        )
        manifests[step] = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifests[1].get("status") != "PASS":
        raise RuntimeError("Step-01 status is not PASS.")
    if (
        manifests[1].get("configuration_fingerprint_sha256")
        != step00["fingerprint"]
    ):
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

    for step in range(3, 10):
        if manifests[step].get("step", {}).get("status") != "PASS":
            raise RuntimeError(f"Step-{step:02d} status is not PASS.")
        if (
            manifests[step].get("configuration_fingerprint_sha256")
            != step00["fingerprint"]
        ):
            raise RuntimeError(f"Step-{step:02d} fingerprint mismatch.")
        for previous in range(1, step):
            field = f"step{previous:02d}_manifest_sha256"
            if field in manifests[step] and manifests[step][field] != shas[previous]:
                raise RuntimeError(
                    f"Step-{step:02d} chain mismatch for {field}."
                )

    if not STEP01_ENV.exists():
        raise RuntimeError("Step-01 environment audit missing.")
    env = json.loads(STEP01_ENV.read_text(encoding="utf-8"))
    stored_python = env.get("system", {}).get("python_executable")
    if (
        stored_python
        and Path(stored_python).resolve() != Path(sys.executable).resolve()
    ):
        raise RuntimeError(
            "Current Python differs from frozen Step-01 environment.\n"
            f"Frozen : {stored_python}\nCurrent: {sys.executable}"
        )

    verify_artifacts(manifests[1])
    verify_artifacts(
        manifests[2],
        key="canonical_artifacts",
        skip=("raw_snapshot_root",),
    )
    for rec in manifests[2].get("raw_file_records", []):
        path = Path(rec["file_path"])
        if not path.exists() or sha256_file(path) != rec["sha256"]:
            raise RuntimeError(f"Step-02 raw-file mismatch: {path}")

    for step in range(3, 10):
        verify_artifacts(manifests[step])

    for rec in manifests[5].get("individual_stock_output_records", []):
        path = Path(rec["file_path"])
        if not path.exists() or sha256_file(path) != rec["sha256"]:
            raise RuntimeError(f"Step-05 individual-stock mismatch: {path}")

    if not STEP09_TARGET_PANEL.exists():
        raise RuntimeError(f"Missing Step-09 target panel: {STEP09_TARGET_PANEL}")

    source_rec = find_artifact(manifests[9], STEP09_TARGET_PANEL.name)
    if Path(source_rec["path"]).resolve() != STEP09_TARGET_PANEL.resolve():
        raise RuntimeError("Step-09 target-panel path is not canonical.")
    source_sha = sha256_file(STEP09_TARGET_PANEL)
    if source_sha != source_rec["sha256"]:
        raise RuntimeError("Step-09 target-panel SHA mismatch.")

    features8 = manifests[8].get("ordered_63_features")
    features9 = manifests[9].get("ordered_63_features")
    interface8 = manifests[8].get("feature_interface_sha256")
    interface9 = manifests[9].get("feature_interface_sha256")

    if not isinstance(features8, list) or len(features8) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError("Step-08 manifest does not freeze 63 predictors.")
    if features9 != features8:
        raise RuntimeError("Step-09 predictor order differs from Step 08.")
    if not interface8 or interface9 != interface8:
        raise RuntimeError("Step-09 feature-interface SHA differs from Step 08.")

    stored_horizons = tuple(
        int(x)
        for x in manifests[9].get("target_design", {}).get(
            "horizons_market_sessions",
            [],
        )
    )
    if stored_horizons != HORIZONS:
        raise RuntimeError(
            f"Step-09 horizons {stored_horizons} differ from Step-10 {HORIZONS}."
        )

    return {
        "sha": shas,
        "manifest": manifests,
        "feature_columns": list(features8),
        "feature_interface_sha256": interface8,
        "step09_target_sha256": source_sha,
        "python_executable": stored_python,
    }


# =============================================================================
# SPLIT DESIGN / IMMUTABILITY
# =============================================================================

def split_design_payload() -> Dict[str, Any]:
    folds = []
    for fold, ts, te, vs, ve in FOLDS:
        folds.append(
            {
                "fold": fold,
                "train_start": ts,
                "train_end": te,
                "validation_start": vs,
                "validation_end": ve,
            }
        )
    return {
        "development": {
            "start": DEVELOPMENT_START.strftime("%Y-%m-%d"),
            "end": DEVELOPMENT_END.strftime("%Y-%m-%d"),
        },
        "final_test": {
            "start": FINAL_TEST_START.strftime("%Y-%m-%d"),
            "end": FINAL_TEST_END.strftime("%Y-%m-%d"),
        },
        "folds": folds,
        "horizons": list(HORIZONS),
        "endpoint_rule": (
            "reference_and_exact_target_endpoint_must_be_inside_same_partition"
        ),
    }


def validate_split_design() -> str:
    if not (
        DEVELOPMENT_START
        <= DEVELOPMENT_END
        < FINAL_TEST_START
        <= FINAL_TEST_END
    ):
        raise RuntimeError("Outer chronology is invalid.")
    if len(FOLDS) != 5:
        raise RuntimeError("Exactly five folds are required.")

    expected_validation_years = [2020, 2021, 2022, 2023, 2024]
    observed_years = []
    previous_train_end = None

    for expected_fold, (fold, ts, te, vs, ve) in enumerate(FOLDS, start=1):
        if fold != expected_fold:
            raise RuntimeError("Fold numbering is not 1..5.")
        ts, te, vs, ve = map(pd.Timestamp, (ts, te, vs, ve))
        if ts != DEVELOPMENT_START:
            raise RuntimeError(f"Fold {fold}: train start drift.")
        if not (ts <= te < vs <= ve <= DEVELOPMENT_END):
            raise RuntimeError(f"Fold {fold}: invalid boundaries.")
        if vs != te + pd.Timedelta(days=1):
            raise RuntimeError(f"Fold {fold}: train/validation periods not contiguous.")
        if (vs.month, vs.day) != (1, 1) or (ve.month, ve.day) != (12, 31):
            raise RuntimeError(f"Fold {fold}: validation is not a calendar year.")
        if previous_train_end is not None and te <= previous_train_end:
            raise RuntimeError(f"Fold {fold}: training window did not expand.")
        previous_train_end = te
        observed_years.append(vs.year)

    if observed_years != expected_validation_years:
        raise RuntimeError("Validation years differ from 2020..2024.")

    return canonical_json_sha256(split_design_payload())


def canonical_outputs() -> tuple[Path, ...]:
    return (
        OUT_DIR,
        FOLD_DEFINITION_CSV,
        OUTER_AUDIT_CSV,
        FOLD_AUDIT_CSV,
        DEV_CLASS_CSV,
        BOUNDARY_CSV,
        STRUCTURE_CSV,
        SOURCE_HASH_CSV,
        HASH_REGISTER_CSV,
        SUMMARY_JSON,
        TEST_SEAL_JSON,
        MANIFEST_JSON,
        MANIFEST_SHA,
        FROZEN_SCRIPT,
        SUMMARY_TXT,
    )


def assert_fresh_target() -> None:
    collisions = [path for path in canonical_outputs() if path.exists()]
    if collisions:
        raise RuntimeError(
            "Step-10 output(s) already exist; refusing to overwrite.\n"
            + "\n".join(str(x) for x in collisions)
            + "\nUse --verify-only if Step 10 already passed."
        )


def create_stage() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = cfg.FEATURE_DATA_DIR / f"_step10_staging_{stamp}_{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "development").mkdir()
    (path / "final_test_locked").mkdir()
    return path


def archive_partial() -> Optional[Path]:
    if MANIFEST_JSON.exists():
        return None
    existing = [path for path in canonical_outputs() if path.exists()]
    if not existing:
        return None
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = FAILED_DIR / f"partial_artifacts_{stamp}"
    dest.mkdir(exist_ok=False)
    for path in existing:
        try:
            shutil.move(str(path), str(dest / path.name))
        except Exception:
            pass
    return dest


# =============================================================================
# LOAD STEP-09 PANEL
# =============================================================================

def load_target_panel(feature_columns: list[str]) -> pd.DataFrame:
    panel = pd.read_parquet(STEP09_TARGET_PANEL, engine="pyarrow")
    panel["Date"] = pd.to_datetime(panel["Date"], errors="raise").dt.normalize()

    for h in HORIZONS:
        col = f"target_date_{h}d"
        panel[col] = pd.to_datetime(panel[col], errors="coerce").dt.normalize()

    required = {
        "Ticker",
        "Date",
        "market_session_id",
        "continuity_segment_id",
        *feature_columns,
        *AUXILIARY_TARGET_COLUMNS,
    }
    for h in HORIZONS:
        required |= {
            f"target_market_session_id_{h}d",
            f"target_continuity_segment_id_{h}d",
            f"target_date_{h}d",
            f"direction_{h}d",
        }

    missing = required - set(panel.columns)
    if missing:
        raise RuntimeError(f"Step-09 panel missing columns: {sorted(missing)}")

    if panel[["Ticker", "Date"]].duplicated().any():
        raise RuntimeError("Duplicate ticker-date keys in Step-09 panel.")
    if set(panel["Ticker"].unique()) != set(cfg.STOCK_TICKERS):
        raise RuntimeError("Step-09 stock universe differs from Step 00.")
    if panel[feature_columns].isna().any().any():
        raise RuntimeError("Predictor nulls detected in Step-09 reference panel.")
    if np.isinf(panel[feature_columns].to_numpy(dtype=float)).any():
        raise RuntimeError("Infinite predictor detected in Step-09 panel.")

    panel["market_session_id"] = pd.to_numeric(
        panel["market_session_id"],
        errors="raise",
    ).astype("int64")

    if not (
        panel.groupby("Date")["Ticker"].nunique()
        == EXPECTED_STOCK_COUNT
    ).all():
        raise RuntimeError("Reference panel has incomplete stock cross-sections.")
    if panel.groupby("Ticker").size().nunique() != 1:
        raise RuntimeError("Reference panel is unbalanced across stocks.")
    if not (
        panel.groupby("Date")["market_session_id"].nunique()
        == 1
    ).all():
        raise RuntimeError("A date maps to multiple market-session IDs.")

    for h in HORIZONS:
        ids = panel[f"target_market_session_id_{h}d"].notna()
        dates = panel[f"target_date_{h}d"].notna()
        labels = panel[f"direction_{h}d"].notna()

        if not np.array_equal(ids.to_numpy(), dates.to_numpy()):
            raise RuntimeError(f"{h}d target ID/date availability mismatch.")
        if not np.array_equal(ids.to_numpy(), labels.to_numpy()):
            raise RuntimeError(f"{h}d metadata/label availability mismatch.")

        if ids.any():
            diff = (
                panel.loc[ids, f"target_market_session_id_{h}d"].astype("int64")
                - panel.loc[ids, "market_session_id"].astype("int64")
            )
            if not (diff == h).all():
                raise RuntimeError(f"{h}d target-session distance mismatch.")

            target_seg = panel.loc[
                ids,
                f"target_continuity_segment_id_{h}d",
            ].astype("int64")
            ref_seg = panel.loc[ids, "continuity_segment_id"].astype("int64")
            if not np.array_equal(target_seg.to_numpy(), ref_seg.to_numpy()):
                raise RuntimeError(f"{h}d target crosses continuity segment.")

    dev = panel["Date"].between(
        DEVELOPMENT_START,
        DEVELOPMENT_END,
        inclusive="both",
    )
    test = panel["Date"].between(
        FINAL_TEST_START,
        FINAL_TEST_END,
        inclusive="both",
    )
    if (dev & test).any() or (~(dev | test)).any():
        raise RuntimeError("Outer-period assignment is not exhaustive/disjoint.")

    return panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)


# =============================================================================
# SPLIT / PURGE LOGIC
# =============================================================================

def balanced_subset(
    panel: pd.DataFrame,
    mask: pd.Series,
    label: str,
    allow_empty: bool = False,
) -> Dict[str, int]:
    subset = panel.loc[mask]
    if subset.empty:
        if allow_empty:
            return {"rows": 0, "dates": 0, "rows_per_stock": 0}
        raise RuntimeError(f"{label}: empty subset.")

    by_stock = subset.groupby("Ticker").size()
    if len(by_stock) != EXPECTED_STOCK_COUNT or by_stock.nunique() != 1:
        raise RuntimeError(f"{label}: unbalanced stock counts.")
    by_date = subset.groupby("Date")["Ticker"].nunique()
    if not (by_date == EXPECTED_STOCK_COUNT).all():
        raise RuntimeError(f"{label}: incomplete stock cross-section.")

    return {
        "rows": int(len(subset)),
        "dates": int(subset["Date"].nunique()),
        "rows_per_stock": int(by_stock.iloc[0]),
    }


def evaluate_partition(
    panel: pd.DataFrame,
    reference_mask: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: int,
    label: str,
) -> tuple[pd.Series, Dict[str, Any]]:
    target_date_col = f"target_date_{horizon}d"
    target_id_col = f"target_market_session_id_{horizon}d"

    available = panel[target_date_col].notna() & panel[target_id_col].notna()
    inside = panel[target_date_col].between(start, end, inclusive="both")

    eligible = reference_mask & available & inside
    purged = reference_mask & available & ~inside
    unavailable = reference_mask & ~available

    if (
        int(eligible.sum())
        + int(purged.sum())
        + int(unavailable.sum())
        != int(reference_mask.sum())
    ):
        raise RuntimeError(f"{label} {horizon}d accounting failure.")

    if purged.any():
        if not (panel.loc[purged, target_date_col] > end).all():
            raise RuntimeError(
                f"{label} {horizon}d contains a non-end boundary purge."
            )

    if eligible.any():
        if not panel.loc[eligible, "Date"].between(
            start,
            end,
            inclusive="both",
        ).all():
            raise RuntimeError(f"{label} {horizon}d reference escaped partition.")
        if not panel.loc[eligible, target_date_col].between(
            start,
            end,
            inclusive="both",
        ).all():
            raise RuntimeError(f"{label} {horizon}d endpoint escaped partition.")
        diff = (
            panel.loc[eligible, target_id_col].astype("int64")
            - panel.loc[eligible, "market_session_id"].astype("int64")
        )
        if not (diff == horizon).all():
            raise RuntimeError(f"{label} {horizon}d endpoint distance drift.")

    ref_b = balanced_subset(panel, reference_mask, f"{label} reference")
    elig_b = balanced_subset(panel, eligible, f"{label} eligible")
    purge_b = balanced_subset(
        panel,
        purged,
        f"{label} purged",
        allow_empty=True,
    )
    miss_b = balanced_subset(
        panel,
        unavailable,
        f"{label} unavailable",
        allow_empty=True,
    )

    return eligible, {
        "reference_rows": int(reference_mask.sum()),
        "reference_dates": ref_b["dates"],
        "target_available_before_boundary_purge": int(
            (reference_mask & available).sum()
        ),
        "boundary_purged_rows": int(purged.sum()),
        "boundary_purged_dates": purge_b["dates"],
        "target_unavailable_rows": int(unavailable.sum()),
        "target_unavailable_dates": miss_b["dates"],
        "eligible_rows": int(eligible.sum()),
        "eligible_dates": elig_b["dates"],
        "eligible_rows_per_stock": elig_b["rows_per_stock"],
    }


def common_mask(
    panel: pd.DataFrame,
    horizon_masks: Dict[int, pd.Series],
    label: str,
) -> pd.Series:
    mask = pd.Series(True, index=panel.index, dtype=bool)
    for h in HORIZONS:
        mask &= horizon_masks[h]

    mask &= panel[AUXILIARY_TARGET_COLUMNS].notna().all(axis=1)

    if not mask.equals(horizon_masks[5]):
        mismatch = int((mask != horizon_masks[5]).sum())
        raise RuntimeError(
            f"{label}: common mask differs from 5d eligibility on {mismatch} rows."
        )
    balanced_subset(panel, mask, f"{label} common")
    return mask


def build_outer(panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for partition, start, end in [
        ("development", DEVELOPMENT_START, DEVELOPMENT_END),
        ("final_test", FINAL_TEST_START, FINAL_TEST_END),
    ]:
        ref = membership["outer_partition"].eq(partition)
        masks = {}
        for h in HORIZONS:
            eligible, stats = evaluate_partition(
                panel,
                ref,
                start,
                end,
                h,
                f"outer {partition}",
            )
            masks[h] = eligible
            membership[f"outer_{partition}_eligible_{h}d"] = eligible
            rows.append(
                {
                    "scope": "outer",
                    "partition": partition,
                    "forecast_horizon_sessions": h,
                    "period_start": start,
                    "period_end": end,
                    **stats,
                }
            )
        membership[f"outer_{partition}_common_multitask"] = common_mask(
            panel,
            masks,
            f"outer {partition}",
        )
    return pd.DataFrame(rows)


def build_folds(panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for fold, ts, te, vs, ve in FOLDS:
        ts, te, vs, ve = map(pd.Timestamp, (ts, te, vs, ve))
        train_ref = panel["Date"].between(ts, te, inclusive="both")
        val_ref = panel["Date"].between(vs, ve, inclusive="both")

        if (train_ref & val_ref).any():
            raise RuntimeError(f"Fold {fold}: train/validation overlap.")
        if (
            membership.loc[train_ref | val_ref, "outer_partition"]
            != "development"
        ).any():
            raise RuntimeError(f"Fold {fold}: final-test row entered a dev fold.")

        balanced_subset(panel, train_ref, f"fold {fold} train reference")
        balanced_subset(panel, val_ref, f"fold {fold} validation reference")

        membership[f"fold_{fold}_train_reference"] = train_ref
        membership[f"fold_{fold}_validation_reference"] = val_ref

        train_masks: Dict[int, pd.Series] = {}
        val_masks: Dict[int, pd.Series] = {}

        for partition, ref, start, end, destination in [
            ("train", train_ref, ts, te, train_masks),
            ("validation", val_ref, vs, ve, val_masks),
        ]:
            for h in HORIZONS:
                eligible, stats = evaluate_partition(
                    panel,
                    ref,
                    start,
                    end,
                    h,
                    f"fold {fold} {partition}",
                )
                destination[h] = eligible
                membership[f"fold_{fold}_{partition}_eligible_{h}d"] = eligible
                rows.append(
                    {
                        "fold": fold,
                        "partition": partition,
                        "forecast_horizon_sessions": h,
                        "period_start": start,
                        "period_end": end,
                        **stats,
                    }
                )

        train_common = common_mask(panel, train_masks, f"fold {fold} train")
        val_common = common_mask(panel, val_masks, f"fold {fold} validation")

        membership[f"fold_{fold}_train_common_multitask"] = train_common
        membership[f"fold_{fold}_validation_common_multitask"] = val_common

        if (train_common & val_common).any():
            raise RuntimeError(f"Fold {fold}: common train/validation overlap.")
        if panel.loc[train_common, "Date"].max() >= panel.loc[val_common, "Date"].min():
            raise RuntimeError(f"Fold {fold}: common chronology violation.")

    return pd.DataFrame(rows)


# =============================================================================
# DEVELOPMENT-ONLY AUDITS
# =============================================================================

def development_class_distribution(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(
        scope: str,
        fold: Optional[int],
        partition: str,
        rule: str,
        h: int,
        mask: pd.Series,
    ) -> None:
        if (
            membership.loc[mask, "outer_partition"]
            != "development"
        ).any():
            raise RuntimeError(
                "Final-test row reached development-only class summary."
            )
        labels = panel.loc[mask, f"direction_{h}d"].astype("int64")
        rows.append(
            {
                "scope": scope,
                "fold": np.nan if fold is None else fold,
                "partition": partition,
                "sampling_rule": rule,
                "forecast_horizon_sessions": h,
                "rows": int(len(labels)),
                "up_rows": int((labels == 1).sum()),
                "non_up_rows": int((labels == 0).sum()),
                "up_ratio": float(labels.mean()),
            }
        )

    for h in HORIZONS:
        add(
            "outer_development",
            None,
            "development",
            "horizon_specific",
            h,
            membership[f"outer_development_eligible_{h}d"],
        )
        add(
            "outer_development",
            None,
            "development",
            "common_multitask",
            h,
            membership["outer_development_common_multitask"],
        )

    for fold, *_ in FOLDS:
        for partition in ("train", "validation"):
            for h in HORIZONS:
                add(
                    "expanding_origin",
                    fold,
                    partition,
                    "horizon_specific",
                    h,
                    membership[f"fold_{fold}_{partition}_eligible_{h}d"],
                )
                add(
                    "expanding_origin",
                    fold,
                    partition,
                    "common_multitask",
                    h,
                    membership[f"fold_{fold}_{partition}_common_multitask"],
                )

    return pd.DataFrame(rows)


def boundary_examples(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(
        scope: str,
        fold: Optional[int],
        partition: str,
        reference: pd.Series,
        eligible_col: str,
        h: int,
        limit: int,
    ) -> None:
        available = panel[f"target_date_{h}d"].notna()
        purged = reference & available & ~membership[eligible_col]
        sample = panel.loc[
            purged,
            [
                "Ticker",
                "Date",
                "market_session_id",
                f"target_market_session_id_{h}d",
                f"target_date_{h}d",
            ],
        ].sort_values(["Date", "Ticker"]).head(limit)

        for rec in sample.itertuples(index=False):
            rows.append(
                {
                    "scope": scope,
                    "fold": np.nan if fold is None else fold,
                    "partition": partition,
                    "forecast_horizon_sessions": h,
                    "ticker": rec.Ticker,
                    "reference_date": rec.Date,
                    "reference_market_session_id": int(rec.market_session_id),
                    "target_market_session_id": int(
                        getattr(rec, f"target_market_session_id_{h}d")
                    ),
                    "target_date": getattr(rec, f"target_date_{h}d"),
                    "reason": "exact_target_endpoint_crosses_partition_end",
                }
            )

    dev_ref = membership["outer_partition"].eq("development")
    for h in HORIZONS:
        add(
            "outer",
            None,
            "development",
            dev_ref,
            f"outer_development_eligible_{h}d",
            h,
            5,
        )

    for fold, *_ in FOLDS:
        for partition in ("train", "validation"):
            ref = membership[f"fold_{fold}_{partition}_reference"]
            for h in HORIZONS:
                add(
                    "expanding_origin",
                    fold,
                    partition,
                    ref,
                    f"fold_{fold}_{partition}_eligible_{h}d",
                    h,
                    3,
                )

    return pd.DataFrame(rows)


def fold_definition_table() -> pd.DataFrame:
    rows = [
        {
            "scope": "outer",
            "fold": np.nan,
            "partition": "development",
            "start_date": DEVELOPMENT_START,
            "end_date": DEVELOPMENT_END,
            "role": "model development and selection only",
        },
        {
            "scope": "outer",
            "fold": np.nan,
            "partition": "final_test",
            "start_date": FINAL_TEST_START,
            "end_date": FINAL_TEST_END,
            "role": "sealed one-time final evaluation only",
        },
    ]
    for fold, ts, te, vs, ve in FOLDS:
        rows.extend(
            [
                {
                    "scope": "expanding_origin",
                    "fold": fold,
                    "partition": "train",
                    "start_date": pd.Timestamp(ts),
                    "end_date": pd.Timestamp(te),
                    "role": "fold training",
                },
                {
                    "scope": "expanding_origin",
                    "fold": fold,
                    "partition": "validation",
                    "start_date": pd.Timestamp(vs),
                    "end_date": pd.Timestamp(ve),
                    "role": "fold model-selection evaluation",
                },
            ]
        )
    return pd.DataFrame(rows)


def structure_audit(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for partition in ("development", "final_test"):
        ref = membership["outer_partition"].eq(partition)
        bal = balanced_subset(panel, ref, f"outer {partition}")
        common = membership[f"outer_{partition}_common_multitask"]
        rows.append(
            {
                "scope": "outer",
                "fold": np.nan,
                "partition": partition,
                "reference_rows": bal["rows"],
                "reference_dates": bal["dates"],
                "reference_rows_per_stock": bal["rows_per_stock"],
                "common_multitask_rows": int(common.sum()),
                "common_multitask_dates": int(
                    panel.loc[common, "Date"].nunique()
                ),
            }
        )

    for fold, *_ in FOLDS:
        for partition in ("train", "validation"):
            ref = membership[f"fold_{fold}_{partition}_reference"]
            bal = balanced_subset(
                panel,
                ref,
                f"fold {fold} {partition}",
            )
            common = membership[f"fold_{fold}_{partition}_common_multitask"]
            rows.append(
                {
                    "scope": "expanding_origin",
                    "fold": fold,
                    "partition": partition,
                    "reference_rows": bal["rows"],
                    "reference_dates": bal["dates"],
                    "reference_rows_per_stock": bal["rows_per_stock"],
                    "common_multitask_rows": int(common.sum()),
                    "common_multitask_dates": int(
                        panel.loc[common, "Date"].nunique()
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# WRITING / SOURCE AUDIT
# =============================================================================

def write_pair(
    frame: pd.DataFrame,
    parquet_path: Path,
    csv_path: Path,
) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        parquet_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    csv_frame = frame.copy()
    for col in [
        x
        for x in csv_frame.columns
        if x == "Date" or x.startswith("target_date_")
    ]:
        csv_frame[col] = pd.to_datetime(
            csv_frame[col],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

    csv_frame.to_csv(
        csv_path,
        index=False,
        float_format="%.17g",
    )


def verify_roundtrip(original: pd.DataFrame, path: Path) -> None:
    observed = pd.read_parquet(path, engine="pyarrow")
    if len(observed) != len(original):
        raise RuntimeError(f"Roundtrip row mismatch: {path}")
    if list(observed.columns) != list(original.columns):
        raise RuntimeError(f"Roundtrip column mismatch: {path}")

    left = original[["Ticker", "Date"]].copy()
    right = observed[["Ticker", "Date"]].copy()
    left["Date"] = pd.to_datetime(left["Date"]).dt.normalize()
    right["Date"] = pd.to_datetime(right["Date"]).dt.normalize()
    if not left.equals(right):
        raise RuntimeError(f"Roundtrip key mismatch: {path}")


def source_hashes() -> pd.DataFrame:
    sources = [
        ("current_step00", Path(cfg.__file__).resolve()),
        ("current_step10", SCRIPT_FILE.resolve()),
        ("step09_target_panel", STEP09_TARGET_PANEL.resolve()),
    ]
    for step, (manifest, sidecar) in UPSTREAM.items():
        sources.extend(
            [
                (f"step{step:02d}_manifest", manifest.resolve()),
                (f"step{step:02d}_manifest_sha", sidecar.resolve()),
            ]
        )

    rows = []
    for role, path in sources:
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
# VERIFY EXISTING
# =============================================================================

def verify_existing() -> int:
    print("=" * 132)
    print("STEP 10 — VERIFY EXISTING CHRONOLOGICAL-SPLIT FREEZE")
    print("=" * 132)

    step00 = verify_step00()
    upstream = verify_upstream(step00)
    split_sha = validate_split_design()

    manifest_sha = verify_sidecar(MANIFEST_JSON, MANIFEST_SHA, "Step 10")
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))

    if manifest.get("step", {}).get("status") != "PASS":
        print("[FAIL] Stored Step-10 status is not PASS.")
        return 20

    if (
        manifest.get("configuration_fingerprint_sha256")
        != step00["fingerprint"]
    ):
        print("[FAIL] Configuration fingerprint mismatch.")
        return 21

    for step in range(1, 10):
        field = f"step{step:02d}_manifest_sha256"
        if manifest.get(field) != upstream["sha"][step]:
            print(f"[FAIL] Chain mismatch: {field}")
            return 22

    if (
        manifest.get("feature_interface_sha256")
        != upstream["feature_interface_sha256"]
    ):
        print("[FAIL] 63-feature interface mismatch.")
        return 23

    if manifest.get("split_design_sha256") != split_sha:
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
        for failure in failures:
            print(f"[FAIL] {failure}")
        print("=" * 132)
        print("STEP 10 VERIFICATION STATUS: FAIL")
        print("=" * 132)
        return 25

    seal = json.loads(TEST_SEAL_JSON.read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED_NOT_OPENED":
        print("[FAIL] Final-test seal status changed.")
        return 26
    if seal.get("sha256") != sha256_file(TEST_PANEL_PARQUET):
        print("[FAIL] Sealed final-test file SHA mismatch.")
        return 27
    if sha256_file(Path(step00["lock_path"])) != step00["lock_sha256"]:
        print("[FAIL] Global final-test lock changed.")
        return 28

    print(f"[PASS] Configuration fingerprint : {step00['fingerprint']}")
    print(f"[PASS] Step-09 manifest          : {upstream['sha'][9]}")
    print(
        f"[PASS] 63-feature interface      : "
        f"{upstream['feature_interface_sha256']}"
    )
    print(f"[PASS] Split-design SHA-256      : {split_sha}")
    print(f"[PASS] Step-10 manifest          : {manifest_sha}")
    print("[PASS] Final-test seal           : SEALED_NOT_OPENED")
    print(f"[PASS] Global final-test state   : {step00['state']}")
    print("=" * 132)
    print("STEP 10 VERIFICATION STATUS: PASS")
    print("=" * 132)
    return 0


# =============================================================================
# MAIN
# =============================================================================

def run() -> int:
    print("=" * 132)
    print(STEP_NAME)
    print("=" * 132)

    stage = "prerequisite_verification"
    staging: Optional[Path] = None

    try:
        print("[1/11] Verifying Step 00 and immutable Step 01 -> Step 09 chain...")
        step00 = verify_step00()
        upstream = verify_upstream(step00)
        split_sha = validate_split_design()

        print(f"       Configuration fingerprint: {step00['fingerprint']}")
        print(f"       Step-09 manifest          : {upstream['sha'][9]}")
        print(f"       Step-09 target SHA-256    : {upstream['step09_target_sha256']}")
        print(
            f"       63-feature interface      : "
            f"{upstream['feature_interface_sha256']}"
        )
        print(f"       Split-design SHA-256      : {split_sha}")
        print(f"       Final-test state          : {step00['state']}")

        stage = "collision_guard"
        print("[2/11] Checking immutable Step-10 output target...")
        assert_fresh_target()
        cfg.create_directories()

        stage = "source_load"
        print("[3/11] Loading and validating frozen Step-09 target panel...")
        panel = load_target_panel(upstream["feature_columns"])

        membership = panel[
            ["Ticker", "Date", "market_session_id", "continuity_segment_id"]
        ].copy()
        membership["outer_partition"] = np.where(
            panel["Date"].between(
                DEVELOPMENT_START,
                DEVELOPMENT_END,
                inclusive="both",
            ),
            "development",
            "final_test",
        )

        dev_ref = membership["outer_partition"].eq("development")
        test_ref = membership["outer_partition"].eq("final_test")

        dev_bal = balanced_subset(panel, dev_ref, "outer development")
        test_bal = balanced_subset(panel, test_ref, "outer final test")

        print(
            f"       Reference rows       : {len(panel):,} | "
            f"rows/stock={panel.groupby('Ticker').size().iloc[0]:,}"
        )
        print(
            f"       Development refs     : {dev_bal['rows']:,} "
            f"({dev_bal['dates']:,} dates)"
        )
        print(
            f"       Final-test refs      : {test_bal['rows']:,} "
            f"({test_bal['dates']:,} dates)"
        )

        stage = "purging"
        print("[4/11] Applying outer endpoint confinement...")
        outer_audit = build_outer(panel, membership)

        print("[5/11] Freezing five expanding-origin folds...")
        fold_audit = build_folds(panel, membership)
        struct_audit = structure_audit(panel, membership)

        for fold, *_ in FOLDS:
            train_common = int(
                membership[f"fold_{fold}_train_common_multitask"].sum()
            )
            val_common = int(
                membership[f"fold_{fold}_validation_common_multitask"].sum()
            )
            train_dates = int(
                panel.loc[
                    membership[f"fold_{fold}_train_reference"],
                    "Date",
                ].nunique()
            )
            val_dates = int(
                panel.loc[
                    membership[f"fold_{fold}_validation_reference"],
                    "Date",
                ].nunique()
            )
            print(
                f"       Fold {fold}: train dates={train_dates:>4,} | "
                f"train common={train_common:>6,} | "
                f"validation dates={val_dates:>3,} | "
                f"validation common={val_common:>5,}"
            )

        stage = "development_audits"
        print("[6/11] Computing development-only class/boundary audits...")
        dev_class = development_class_distribution(panel, membership)
        boundary = boundary_examples(panel, membership)
        fold_defs = fold_definition_table()

        outer_common_class = dev_class.loc[
            (dev_class["scope"] == "outer_development")
            & (dev_class["sampling_rule"] == "common_multitask")
        ].sort_values("forecast_horizon_sessions")

        for row in outer_common_class.itertuples(index=False):
            print(
                f"       Development common {row.forecast_horizon_sessions}d: "
                f"rows={row.rows:,} | up_ratio={row.up_ratio:.4f}"
            )
        print("       Final-test class ratios: NOT COMPUTED")

        stage = "staging"
        print("[7/11] Writing membership, development and sealed final-test panels...")

        dev_panel = panel.loc[dev_ref].copy().reset_index(drop=True)
        test_panel = panel.loc[test_ref].copy().reset_index(drop=True)

        staging = create_stage()

        membership_pq = staging / MEMBERSHIP_PARQUET.name
        membership_csv = staging / MEMBERSHIP_CSV.name
        dev_pq = staging / "development" / DEV_PANEL_PARQUET.name
        dev_csv = staging / "development" / DEV_PANEL_CSV.name
        test_pq = staging / "final_test_locked" / TEST_PANEL_PARQUET.name

        write_pair(membership, membership_pq, membership_csv)
        verify_roundtrip(membership, membership_pq)

        write_pair(dev_panel, dev_pq, dev_csv)
        verify_roundtrip(dev_panel, dev_pq)

        test_panel.to_parquet(
            test_pq,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )
        verify_roundtrip(test_panel, test_pq)
        staged_test_sha = sha256_file(test_pq)

        stage = "promotion"
        print("[8/11] Promoting split artifacts and writing audits...")
        if OUT_DIR.exists():
            raise RuntimeError("Canonical Step-10 output directory appeared.")
        os.replace(staging, OUT_DIR)
        staging = None

        if sha256_file(TEST_PANEL_PARQUET) != staged_test_sha:
            raise RuntimeError("Sealed final-test file changed during promotion.")

        write_csv(FOLD_DEFINITION_CSV, fold_defs)
        write_csv(OUTER_AUDIT_CSV, outer_audit)
        write_csv(FOLD_AUDIT_CSV, fold_audit)
        write_csv(DEV_CLASS_CSV, dev_class)
        write_csv(BOUNDARY_CSV, boundary)
        write_csv(STRUCTURE_CSV, struct_audit)
        write_csv(SOURCE_HASH_CSV, source_hashes())

        dev_common = int(
            membership["outer_development_common_multitask"].sum()
        )
        test_common = int(
            membership["outer_final_test_common_multitask"].sum()
        )

        seal = {
            "schema_version": SCHEMA_VERSION,
            "status": "SEALED_NOT_OPENED",
            "generated_at_utc": utc_now(),
            "configuration_fingerprint_sha256": step00["fingerprint"],
            "step09_manifest_sha256": upstream["sha"][9],
            "step09_target_panel_sha256": upstream["step09_target_sha256"],
            "global_final_test_lock_state": step00["state"],
            "global_final_test_lock_sha256": step00["lock_sha256"],
            "file_path": str(TEST_PANEL_PARQUET),
            "relative_path": relative_to_experiment(TEST_PANEL_PARQUET),
            "sha256": sha256_file(TEST_PANEL_PARQUET),
            "reference_rows": int(len(test_panel)),
            "unique_reference_dates": int(test_panel["Date"].nunique()),
            "stock_count": int(test_panel["Ticker"].nunique()),
            "first_reference_date": test_panel["Date"].min().strftime("%Y-%m-%d"),
            "last_reference_date": test_panel["Date"].max().strftime("%Y-%m-%d"),
            "common_multitask_structural_rows": test_common,
            "test_class_distribution_reported": False,
            "test_model_performance_computed": False,
            "permitted_use_before_final_model_freeze": "none",
            "note": (
                "This Step-10 seal does not replace or mutate the global "
                "Step-00 final-test lock."
            ),
        }
        atomic_json(TEST_SEAL_JSON, seal)

        stage = "freeze"
        print("[9/11] Freezing Step-10 summary, source snapshot and manifest...")

        lock_after = sha256_file(Path(step00["lock_path"]))
        if lock_after != step00["lock_sha256"]:
            raise RuntimeError("Global final-test lock changed during Step 10.")

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
            "feature_interface_sha256": upstream["feature_interface_sha256"],
            "split_design_sha256": split_sha,
            "step09_target_panel_sha256": upstream["step09_target_sha256"],
            "development_reference_rows": int(len(dev_panel)),
            "development_reference_dates": int(dev_panel["Date"].nunique()),
            "development_common_multitask_rows": dev_common,
            "final_test_reference_rows": int(len(test_panel)),
            "final_test_reference_dates": int(test_panel["Date"].nunique()),
            "final_test_common_multitask_structural_rows": test_common,
            "folds": split_design_payload()["folds"],
            "final_test_class_distribution_computed": False,
            "final_test_performance_computed": False,
            "scaling_applied": False,
            "regimes_constructed": False,
            "feature_selection_applied": False,
            "sequences_constructed": False,
            "models_fitted": False,
            "global_final_test_state": step00["state"],
            "global_final_test_lock_unchanged": True,
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
                    f"Step-09 manifest: {upstream['sha'][9]}",
                    (
                        "63-feature interface: "
                        f"{upstream['feature_interface_sha256']}"
                    ),
                    f"Split-design SHA-256: {split_sha}",
                    f"Development reference rows: {len(dev_panel)}",
                    f"Development common rows: {dev_common}",
                    f"Final-test reference rows: {len(test_panel)}",
                    f"Final-test structural common rows: {test_common}",
                    "Final-test class ratios reported: False",
                    f"Global final-test state: {step00['state']}",
                ]
            )
            + "\n",
        )

        artifact_paths = [
            MEMBERSHIP_PARQUET,
            MEMBERSHIP_CSV,
            DEV_PANEL_PARQUET,
            DEV_PANEL_CSV,
            TEST_PANEL_PARQUET,
            FOLD_DEFINITION_CSV,
            OUTER_AUDIT_CSV,
            FOLD_AUDIT_CSV,
            DEV_CLASS_CSV,
            BOUNDARY_CSV,
            STRUCTURE_CSV,
            SOURCE_HASH_CSV,
            SUMMARY_JSON,
            TEST_SEAL_JSON,
            FROZEN_SCRIPT,
            SUMMARY_TXT,
        ]

        hash_rows = [
            {
                "file_path": str(path),
                "relative_path": relative_to_experiment(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ]
        write_csv(HASH_REGISTER_CSV, pd.DataFrame(hash_rows))
        artifact_paths.append(HASH_REGISTER_CSV)

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
            "step00_manifest_sha256": step00["manifest_sha256"],
            **{
                f"step{step:02d}_manifest_sha256": upstream["sha"][step]
                for step in range(1, 10)
            },
            "step09_target_panel_sha256": upstream["step09_target_sha256"],
            "feature_interface_sha256": upstream["feature_interface_sha256"],
            "ordered_63_features": upstream["feature_columns"],
            "split_design_sha256": split_sha,
            "split_design": split_design_payload(),
            "outer_split": {
                "development_reference_rows": int(len(dev_panel)),
                "development_common_multitask_rows": dev_common,
                "final_test_reference_rows": int(len(test_panel)),
                "final_test_common_multitask_structural_rows": test_common,
            },
            "test_protection": {
                "global_final_test_state": step00["state"],
                "global_final_test_lock_unchanged": True,
                "step10_test_artifact_status": "SEALED_NOT_OPENED",
                "test_label_distribution_reported": False,
                "test_performance_computed": False,
                "test_used_for_model_selection": False,
                "test_used_for_feature_selection": False,
                "test_used_for_threshold_selection": False,
                "test_used_for_seed_selection": False,
            },
            "pipeline_state": {
                "chronological_split_assigned": True,
                "horizon_boundary_purging_applied": True,
                "scaling_applied": False,
                "regimes_constructed": False,
                "feature_selection_applied": False,
                "sequence_construction_performed": False,
                "models_fitted": False,
            },
            "artifacts": {
                path.name: artifact(path)
                for path in artifact_paths
            },
        }

        atomic_json(MANIFEST_JSON, manifest)
        manifest_sha = sha256_file(MANIFEST_JSON)
        atomic_text(
            MANIFEST_SHA,
            f"{manifest_sha}  {MANIFEST_JSON.name}\n",
        )

        stage = "validation"
        print("[10/11] Validating frozen Step-10 artifact set...")
        verify_sidecar(MANIFEST_JSON, MANIFEST_SHA, "Step 10")
        for rec in manifest["artifacts"].values():
            path = Path(rec["path"])
            if not path.exists() or sha256_file(path) != rec["sha256"]:
                raise RuntimeError(f"Frozen artifact mismatch: {path}")

        stage = "report"
        print("[11/11] Final Step-10 report...")
        print()
        print("-" * 132)
        print("OUTER PARTITION")
        print("-" * 132)
        print(f"Development reference rows    : {len(dev_panel):,}")
        print(f"Development reference dates   : {dev_panel['Date'].nunique():,}")
        print(f"Development common rows       : {dev_common:,}")
        print(f"Final-test reference rows     : {len(test_panel):,}")
        print(f"Final-test reference dates    : {test_panel['Date'].nunique():,}")
        print(f"Final-test structural common  : {test_common:,}")

        print("-" * 132)
        print("OUTER HORIZON PURGING")
        print("-" * 132)
        for row in outer_audit.itertuples(index=False):
            print(
                f"{row.partition:<11} {row.forecast_horizon_sessions}d | "
                f"refs={row.reference_rows:>6,} | "
                f"available={row.target_available_before_boundary_purge:>6,} | "
                f"purged={row.boundary_purged_rows:>4,} | "
                f"unavailable={row.target_unavailable_rows:>4,} | "
                f"eligible={row.eligible_rows:>6,}"
            )

        print("-" * 132)
        print("EXPANDING-ORIGIN FOLDS")
        print("-" * 132)
        for fold, *_ in FOLDS:
            tr = int(
                membership[f"fold_{fold}_train_common_multitask"].sum()
            )
            va = int(
                membership[f"fold_{fold}_validation_common_multitask"].sum()
            )
            tr_dates = int(
                panel.loc[
                    membership[f"fold_{fold}_train_reference"],
                    "Date",
                ].nunique()
            )
            va_dates = int(
                panel.loc[
                    membership[f"fold_{fold}_validation_reference"],
                    "Date",
                ].nunique()
            )
            print(
                f"Fold {fold}: train dates={tr_dates:>4,} | "
                f"train common={tr:>6,} | validation dates={va_dates:>3,} | "
                f"validation common={va:>5,}"
            )

        print("-" * 132)
        print("DEVELOPMENT-ONLY COMMON-MULTITASK CLASS DISTRIBUTION")
        print("-" * 132)
        for row in outer_common_class.itertuples(index=False):
            print(
                f"{row.forecast_horizon_sessions}d: rows={row.rows:,} | "
                f"up={row.up_rows:,} | non-up={row.non_up_rows:,} | "
                f"up-ratio={row.up_ratio:.4f}"
            )

        print("-" * 132)
        print("FINAL-TEST PROTECTION")
        print("-" * 132)
        print(f"Sealed test file             : {TEST_PANEL_PARQUET}")
        print(f"Sealed test SHA-256          : {sha256_file(TEST_PANEL_PARQUET)}")
        print("Step-10 seal status          : SEALED_NOT_OPENED")
        print("Test class distribution      : NOT COMPUTED")
        print("Test model performance       : NOT COMPUTED")
        print(f"Global final-test state      : {step00['state']}")

        print("-" * 132)
        print("PIPELINE STATE")
        print("-" * 132)
        print("Chronological split          : FROZEN")
        print("Horizon-boundary purge       : APPLIED")
        print("Scaling applied              : False")
        print("Regimes constructed          : False")
        print("Feature selection applied    : False")
        print("Sequences constructed        : False")
        print("Models fitted                : False")

        print("-" * 132)
        print("OUTPUT FILES")
        print("-" * 132)
        print(f"Membership ledger    : {MEMBERSHIP_PARQUET}")
        print(f"Development panel    : {DEV_PANEL_PARQUET}")
        print(f"Final-test sealed    : {TEST_PANEL_PARQUET}")
        print(f"Outer purge audit    : {OUTER_AUDIT_CSV}")
        print(f"Fold purge audit     : {FOLD_AUDIT_CSV}")
        print(f"Test seal            : {TEST_SEAL_JSON}")
        print(f"Manifest             : {MANIFEST_JSON}")
        print(f"Manifest SHA-256     : {manifest_sha}")
        print("=" * 132)
        print(
            "PASS: Development and final held-out periods are chronologically separated."
        )
        print("PASS: Five expanding-origin development folds are frozen.")
        print("PASS: Exact target endpoints crossing partition ends are purged.")
        print(
            "PASS: Development-only class balance is reported; final-test class balance is not inspected."
        )
        print(
            "PASS: Final-test artifact is sealed while the global Step-00 lock remains LOCKED_NOT_OPENED."
        )
        print(
            "PASS: No scaling, regime fitting, feature selection, sequences, thresholds, or models were created."
        )
        print("=" * 132)
        print("Do not run Step 11 yet. Paste the complete terminal output for audit.")
        return 0

    except Exception as exc:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

        archived = archive_partial()
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        failure = FAILED_DIR / f"10_splits_FAILED_{stamp}.json"
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
        print("STEP 10 FAILED — SPLITS NOT FROZEN")
        print("=" * 132)
        print(f"Stage      : {stage}")
        print(f"Error type : {type(exc).__name__}")
        print(f"Error      : {exc}")
        print(f"Diagnostic : {failure}")
        if archived is not None:
            print(f"Archived partial artifacts: {archived}")
        print("=" * 132)
        print("Do not proceed to Step 11.")
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
        print("\nStep 10 interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print()
        print("=" * 132)
        print("STEP 10 UNHANDLED FAILURE")
        print("=" * 132)
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        print("=" * 132)
        sys.exit(1)
