#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 09 — CLEAN V3 MULTI-HORIZON TARGET CONSTRUCTION

Constructs leakage-auditable 1-, 3-, and 5-market-session directional targets
and the frozen auxiliary regression targets from the verified Step-08 complete
63-feature panel.

Horizon definition
------------------
Horizons are defined on the complete price-valid NIFTY calendar frozen by
Step 03. For origin master-session ID s and horizon h, the requested endpoint
is s+h. A target is available only when that exact same-stock endpoint exists
and remains in the same frozen stock continuity segment.

This prevents a removed stock-panel anomaly session from being silently skipped
when defining 1-, 3-, or 5-session horizons.

Canonical target price
----------------------
Frozen Step-05 Model_Close.

Direction target
----------------
direction_h = 1 when future_return_h > 0
              0 when future_return_h <= 0
              NA when the exact endpoint is unavailable.

Held-out discipline
-------------------
This stage does not report directional class ratios and does not score or tune
anything on the held-out test. The final-test lock must remain
LOCKED_NOT_OPENED.

Run:
    python step09_construct_targets_v3.py

Verify:
    python step09_construct_targets_v3.py --verify-only
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
        "step09_construct_targets_v3.py must be inside <repository>/q1_revision/."
    )
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg


# =============================================================================
# CONSTANTS
# =============================================================================

STEP_NUMBER = 9
STEP_ID = "09"
STEP_NAME = "STEP 09 — MULTI-HORIZON DIRECTIONAL AND AUXILIARY TARGETS"
SCHEMA_VERSION = "1.0"

FORECAST_HORIZONS = (1, 3, 5)
MAX_HORIZON = 5
EXPECTED_FEATURE_COUNT = 63
EXPECTED_STOCK_COUNT = len(cfg.STOCK_TICKERS)

PRIMARY_TARGET_COLUMNS = [
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
STEP03_MASTER_CALENDAR = (
    cfg.REFERENCE_DATA_DIR / "03_candidate_nifty_price_calendar.csv"
)

STEP04_MANIFEST = cfg.MANIFEST_DIR / "04_session_policy_manifest.json"
STEP04_SHA = cfg.MANIFEST_DIR / "04_session_policy_manifest.sha256"
STEP04_POLICY = cfg.MANIFEST_DIR / "04_cleaning_policy.json"

STEP05_MANIFEST = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.json"
STEP05_SHA = cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.sha256"
STEP05_PANEL = (
    cfg.PROCESSED_DATA_DIR
    / "step05_clean_model_panel"
    / "05_stock_panel.parquet"
)

STEP06_MANIFEST = cfg.MANIFEST_DIR / "06_stock_feature_manifest.json"
STEP06_SHA = cfg.MANIFEST_DIR / "06_stock_feature_manifest.sha256"

STEP07_MANIFEST = cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.json"
STEP07_SHA = cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.sha256"

STEP08_MANIFEST = cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.json"
STEP08_SHA = cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.sha256"
STEP08_COMPLETE_PANEL = (
    cfg.FEATURE_DATA_DIR
    / "step08_merged_63_features"
    / "08_merged_63_feature_panel_complete.parquet"
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
}


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = cfg.FEATURE_DATA_DIR / "step09_targets"

TARGET_PANEL_PARQUET = OUT_DIR / "09_multi_horizon_target_panel.parquet"
TARGET_PANEL_CSV = OUT_DIR / "09_multi_horizon_target_panel.csv"

COMMON_PANEL_PARQUET = OUT_DIR / "09_common_multitask_eligible_panel.parquet"
COMMON_PANEL_CSV = OUT_DIR / "09_common_multitask_eligible_panel.csv"

TARGET_DICTIONARY_CSV = cfg.AUDIT_DIR / "09_target_dictionary.csv"
CONTINUITY_AUDIT_CSV = cfg.AUDIT_DIR / "09_target_continuity_audit.csv"
AVAILABILITY_AUDIT_CSV = (
    cfg.AUDIT_DIR / "09_target_availability_by_stock_segment.csv"
)
REFERENCE_AUDIT_CSV = cfg.AUDIT_DIR / "09_reference_price_alignment_audit.csv"
BOUNDARY_AUDIT_CSV = (
    cfg.AUDIT_DIR / "09_continuity_boundary_target_audit.csv"
)
AUXILIARY_AUDIT_CSV = cfg.AUDIT_DIR / "09_auxiliary_target_audit.csv"
COMMON_AUDIT_CSV = cfg.AUDIT_DIR / "09_common_multitask_key_audit.csv"
HASH_REGISTER_CSV = cfg.AUDIT_DIR / "09_target_hashes.csv"
SOURCE_HASH_CSV = cfg.AUDIT_DIR / "09_source_file_hashes.csv"
SUMMARY_JSON = cfg.AUDIT_DIR / "09_target_summary.json"

MANIFEST_JSON = cfg.MANIFEST_DIR / "09_target_manifest.json"
MANIFEST_SHA = cfg.MANIFEST_DIR / "09_target_manifest.sha256"
FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step09_construct_targets_frozen.py"
SUMMARY_TXT = cfg.REPORT_DIR / "09_target_summary.txt"

FAILED_DIR = cfg.AUDIT_DIR / "step09_failed_attempts"


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


def artifact(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
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


def verify_artifact_records(
    manifest: Dict[str, Any],
    key: str = "artifacts",
    skip: tuple[str, ...] = (),
) -> None:
    failures = []
    for name, record in manifest.get(key, {}).items():
        if name in skip or not isinstance(record, dict):
            continue
        ptext = record.get("path")
        if not ptext:
            continue
        path = Path(ptext)
        if not path.exists():
            failures.append(f"missing {name}: {path}")
            continue
        expected = record.get("sha256")
        if expected and sha256_file(path) != expected:
            failures.append(f"hash mismatch: {name}")
    if failures:
        raise RuntimeError("\n".join(failures))


# =============================================================================
# STEP-00 + UPSTREAM CHAIN VERIFICATION
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


def load_and_verify_upstream(step00: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    actual_sha: Dict[int, str] = {}
    manifests: Dict[int, Dict[str, Any]] = {}

    for step in range(1, 9):
        manifest_path, sidecar = UPSTREAM[step]
        actual_sha[step] = verify_sidecar(
            manifest_path,
            sidecar,
            f"Step {step:02d}",
        )
        manifests[step] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

    # Status + fingerprint.
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

    for step in range(3, 9):
        if manifests[step].get("step", {}).get("status") != "PASS":
            raise RuntimeError(f"Step-{step:02d} status is not PASS.")
        if (
            manifests[step].get("configuration_fingerprint_sha256")
            != step00["fingerprint"]
        ):
            raise RuntimeError(f"Step-{step:02d} fingerprint mismatch.")

    # Chain.
    if (
        manifests[2].get("step01", {}).get("step01_manifest_sha256")
        != actual_sha[1]
    ):
        raise RuntimeError("Step-02 -> Step-01 chain mismatch.")

    for step in range(3, 9):
        for previous in range(1, step):
            field = f"step{previous:02d}_manifest_sha256"
            # Steps 03-08 all carry the manifest chain used by this clean V3.
            if field in manifests[step]:
                if manifests[step][field] != actual_sha[previous]:
                    raise RuntimeError(
                        f"Step-{step:02d} chain mismatch for {field}."
                    )

    # Environment identity.
    if not STEP01_ENV.exists():
        raise RuntimeError("Step-01 environment audit missing.")
    env = json.loads(STEP01_ENV.read_text(encoding="utf-8"))
    stored_python = env.get("system", {}).get("python_executable")
    if stored_python and Path(stored_python).resolve() != Path(sys.executable).resolve():
        raise RuntimeError(
            "Current Python differs from frozen Step-01 environment.\n"
            f"Frozen : {stored_python}\nCurrent: {sys.executable}"
        )

    # Reverify upstream artifacts.
    verify_artifact_records(manifests[1])

    verify_artifact_records(
        manifests[2],
        key="canonical_artifacts",
        skip=("raw_snapshot_root",),
    )
    for record in manifests[2].get("raw_file_records", []):
        path = Path(record["file_path"])
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Step-02 raw-file mismatch: {path}")

    for step in range(3, 9):
        verify_artifact_records(manifests[step])

    for record in manifests[5].get("individual_stock_output_records", []):
        path = Path(record["file_path"])
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Step-05 individual-stock mismatch: {path}")

    # Required source files.
    if not STEP05_PANEL.exists():
        raise RuntimeError(f"Missing Step-05 panel: {STEP05_PANEL}")
    if not STEP08_COMPLETE_PANEL.exists():
        raise RuntimeError(f"Missing Step-08 complete panel: {STEP08_COMPLETE_PANEL}")

    ordered_features = manifests[8].get("ordered_63_features")
    interface_sha = manifests[8].get("feature_interface_sha256")
    if not isinstance(ordered_features, list) or len(ordered_features) != 63:
        raise RuntimeError("Step-08 manifest does not freeze exactly 63 predictors.")
    if not interface_sha:
        raise RuntimeError("Step-08 feature-interface SHA missing.")

    return {
        "sha": actual_sha,
        "manifest": manifests,
        "feature_columns": ordered_features,
        "feature_interface_sha256": interface_sha,
    }


# =============================================================================
# IMMUTABILITY
# =============================================================================

def canonical_outputs() -> tuple[Path, ...]:
    return (
        OUT_DIR,
        TARGET_DICTIONARY_CSV,
        CONTINUITY_AUDIT_CSV,
        AVAILABILITY_AUDIT_CSV,
        REFERENCE_AUDIT_CSV,
        BOUNDARY_AUDIT_CSV,
        AUXILIARY_AUDIT_CSV,
        COMMON_AUDIT_CSV,
        HASH_REGISTER_CSV,
        SOURCE_HASH_CSV,
        SUMMARY_JSON,
        MANIFEST_JSON,
        MANIFEST_SHA,
        FROZEN_SCRIPT,
        SUMMARY_TXT,
    )


def assert_fresh_target() -> None:
    collisions = [p for p in canonical_outputs() if p.exists()]
    if collisions:
        raise RuntimeError(
            "Step-09 canonical outputs already exist. Refusing overwrite.\n"
            + "\n".join(str(p) for p in collisions)
            + "\nRun --verify-only if Step 09 already passed."
        )


def create_stage() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = cfg.FEATURE_DATA_DIR / f"_step09_staging_{stamp}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def archive_partial() -> Optional[Path]:
    if MANIFEST_JSON.exists():
        return None
    existing = [p for p in canonical_outputs() if p.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = FAILED_DIR / f"partial_artifacts_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    for p in existing:
        try:
            shutil.move(str(p), str(archive / p.name))
        except Exception:
            pass
    return archive


# =============================================================================
# CALENDAR / CONTINUITY / SOURCES
# =============================================================================

def load_break_dates(policy: Dict[str, Any]) -> list[pd.Timestamp]:
    p = policy.get("date_wide_anomaly_policy", {})
    values = p.get("valid_nifty_sessions_excluded_by_rule", [])
    declared = p.get("valid_nifty_session_exclusion_count")
    if declared is not None and int(declared) != len(values):
        raise RuntimeError("Step-04 break-date count mismatch.")
    return sorted({pd.Timestamp(v).normalize() for v in values})


def segment_ids(dates: pd.Series, breaks: list[pd.Timestamp]) -> pd.Series:
    values = (
        pd.to_datetime(dates, errors="raise")
        .dt.normalize()
        .to_numpy(dtype="datetime64[ns]")
    )
    if not breaks:
        return pd.Series(np.zeros(len(values), dtype=np.int64), index=dates.index)
    b = np.asarray(
        [np.datetime64(x, "ns") for x in breaks],
        dtype="datetime64[ns]",
    )
    return pd.Series(
        np.searchsorted(b, values, side="left"),
        index=dates.index,
        dtype="int64",
    )


def load_master_calendar(breaks: list[pd.Timestamp]) -> pd.DataFrame:
    cal = pd.read_csv(STEP03_MASTER_CALENDAR, parse_dates=["Date"])
    required = {"session_number", "Date"}
    if not required.issubset(cal.columns):
        raise RuntimeError("Step-03 master calendar schema mismatch.")

    cal["Date"] = pd.to_datetime(cal["Date"], errors="raise").dt.normalize()
    cal = (
        cal[["session_number", "Date"]]
        .sort_values("session_number")
        .reset_index(drop=True)
    )

    if cal["Date"].duplicated().any() or not cal["Date"].is_monotonic_increasing:
        raise RuntimeError("Master calendar is not unique/chronological.")

    ids = pd.to_numeric(cal["session_number"], errors="raise").astype("int64")
    if len(ids) == 0:
        raise RuntimeError("Master calendar is empty.")
    expected = np.arange(ids.iloc[0], ids.iloc[0] + len(ids), dtype=np.int64)
    if not np.array_equal(ids.to_numpy(), expected):
        raise RuntimeError("Master session IDs are not contiguous.")

    master_dates = set(cal["Date"].tolist())
    for d in breaks:
        if d not in master_dates:
            raise RuntimeError(
                f"Frozen continuity-break date absent from master calendar: {d.date()}"
            )

    return cal.rename(columns={"session_number": "market_session_id"})


def load_references(
    feature_columns: list[str],
    master: pd.DataFrame,
) -> pd.DataFrame:
    ref = pd.read_parquet(STEP08_COMPLETE_PANEL, engine="pyarrow")
    ref["Date"] = pd.to_datetime(ref["Date"], errors="raise").dt.normalize()

    required = {
        "Date",
        "Ticker",
        "continuity_segment_id",
        "close",
        *feature_columns,
    }
    missing = required - set(ref.columns)
    if missing:
        raise RuntimeError(f"Step-08 reference columns missing: {sorted(missing)}")

    if ref[["Ticker", "Date"]].duplicated().any():
        raise RuntimeError("Duplicate Step-08 reference ticker-date keys.")
    if set(ref["Ticker"].unique()) != set(cfg.STOCK_TICKERS):
        raise RuntimeError("Step-08 stock universe differs from Step 00.")
    if ref[feature_columns].isna().any().any():
        raise RuntimeError("Step-08 complete panel contains predictor nulls.")
    if np.isinf(ref[feature_columns].to_numpy(dtype=float)).any():
        raise RuntimeError("Step-08 complete panel contains predictor infinities.")

    ref = ref.merge(master, on="Date", how="left", validate="many_to_one")
    if ref["market_session_id"].isna().any():
        raise RuntimeError("Reference date missing from master calendar.")
    ref["market_session_id"] = ref["market_session_id"].astype("int64")
    ref = ref.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    counts = ref.groupby("Ticker").size()
    if counts.nunique() != 1:
        raise RuntimeError("Reference panel is not balanced across stocks.")

    return ref


def load_price_lookup(
    master: pd.DataFrame,
    breaks: list[pd.Timestamp],
) -> pd.DataFrame:
    src = pd.read_parquet(STEP05_PANEL, engine="pyarrow")
    src["Date"] = pd.to_datetime(src["Date"], errors="raise").dt.normalize()

    required = {"Date", "Ticker", "Model_Close"}
    missing = required - set(src.columns)
    if missing:
        raise RuntimeError(f"Step-05 target-price columns missing: {sorted(missing)}")
    if src[["Ticker", "Date"]].duplicated().any():
        raise RuntimeError("Duplicate Step-05 ticker-date keys.")
    if set(src["Ticker"].unique()) != set(cfg.STOCK_TICKERS):
        raise RuntimeError("Step-05 stock universe differs from Step 00.")

    src["Model_Close"] = pd.to_numeric(
        src["Model_Close"], errors="raise"
    ).astype("float64")
    if not np.isfinite(src["Model_Close"].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite Step-05 Model_Close.")
    if (src["Model_Close"] <= 0).any():
        raise RuntimeError("Non-positive Step-05 Model_Close.")

    src = src.merge(master, on="Date", how="left", validate="many_to_one")
    if src["market_session_id"].isna().any():
        raise RuntimeError("Step-05 date missing from master calendar.")
    src["market_session_id"] = src["market_session_id"].astype("int64")
    src["target_continuity_segment_id"] = segment_ids(src["Date"], breaks)

    cross = src.groupby("Date")["Ticker"].nunique()
    if not (cross == EXPECTED_STOCK_COUNT).all():
        raise RuntimeError("Step-05 price source is not a complete stock panel.")

    lookup = src[
        [
            "Ticker",
            "market_session_id",
            "Date",
            "target_continuity_segment_id",
            "Model_Close",
        ]
    ].copy()

    if lookup[["Ticker", "market_session_id"]].duplicated().any():
        raise RuntimeError("Duplicate Step-05 ticker/master-session keys.")

    return lookup


# =============================================================================
# REFERENCE RECONCILIATION
# =============================================================================

def reference_audit(ref: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    right = lookup.rename(
        columns={
            "Model_Close": "step05_model_close",
            "target_continuity_segment_id": "step05_segment_id",
        }
    )
    cmp = ref[
        ["Ticker", "market_session_id", "continuity_segment_id", "close"]
    ].merge(
        right[
            ["Ticker", "market_session_id", "step05_model_close", "step05_segment_id"]
        ],
        on=["Ticker", "market_session_id"],
        how="left",
        validate="one_to_one",
    )
    if cmp["step05_model_close"].isna().any():
        raise RuntimeError("Reference close cannot be matched to Step-05 price source.")

    a = pd.to_numeric(cmp["close"], errors="raise").to_numpy(dtype=float)
    b = pd.to_numeric(cmp["step05_model_close"], errors="raise").to_numpy(dtype=float)
    price_exact = np.array_equal(a, b, equal_nan=True)

    segment_exact = np.array_equal(
        cmp["continuity_segment_id"].to_numpy(dtype=np.int64),
        cmp["step05_segment_id"].to_numpy(dtype=np.int64),
    )

    if not price_exact:
        raise RuntimeError("Step-08 close differs from Step-05 Model_Close.")
    if not segment_exact:
        raise RuntimeError("Step-08 and Step-05-derived continuity IDs differ.")

    return pd.DataFrame(
        [
            {
                "reference_rows": len(cmp),
                "all_reference_prices_match": True,
                "all_continuity_ids_match": True,
                "maximum_absolute_close_difference": float(
                    np.max(np.abs(a - b)) if len(a) else 0.0
                ),
            }
        ]
    )


# =============================================================================
# TARGETS
# =============================================================================

def lookup_endpoint(
    ref: pd.DataFrame,
    lookup: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    base = ref[
        ["Ticker", "market_session_id", "continuity_segment_id"]
    ].copy()
    base["requested_id"] = base["market_session_id"] + int(horizon)

    endpoint = lookup.rename(
        columns={
            "market_session_id": "candidate_id",
            "Date": "candidate_date",
            "Model_Close": "candidate_close",
        }
    )

    matched = base.merge(
        endpoint[
            [
                "Ticker",
                "candidate_id",
                "candidate_date",
                "target_continuity_segment_id",
                "candidate_close",
            ]
        ],
        left_on=["Ticker", "requested_id"],
        right_on=["Ticker", "candidate_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    same = (
        matched["candidate_id"].notna()
        & matched["target_continuity_segment_id"].notna()
        & (
            matched["target_continuity_segment_id"]
            == matched["continuity_segment_id"]
        )
    )

    out = pd.DataFrame(index=matched.index)
    out["target_available"] = same
    out["target_market_session_id"] = matched["candidate_id"].where(same).astype("Int64")
    out["target_date"] = matched["candidate_date"].where(same)
    out["target_segment_id"] = (
        matched["target_continuity_segment_id"].where(same).astype("Int64")
    )
    out["future_close"] = matched["candidate_close"].where(same)
    return out


def construct_targets(
    ref: pd.DataFrame,
    lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    panel = ref.copy()
    endpoints = {
        h: lookup_endpoint(ref, lookup, h)
        for h in range(1, MAX_HORIZON + 1)
    }

    current_close = pd.to_numeric(panel["close"], errors="raise").astype("float64")

    for h in FORECAST_HORIZONS:
        suffix = f"{h}d"
        ep = endpoints[h]

        panel[f"target_market_session_id_{suffix}"] = ep["target_market_session_id"]
        panel[f"target_continuity_segment_id_{suffix}"] = ep["target_segment_id"]
        panel[f"target_date_{suffix}"] = ep["target_date"]
        panel[f"future_close_{suffix}"] = ep["future_close"]

        future_close = panel[f"future_close_{suffix}"]
        future_return = future_close / current_close - 1.0

        panel[f"future_return_{suffix}"] = future_return
        panel[f"future_log_return_{suffix}"] = np.log(future_close / current_close)

        direction = pd.Series(pd.NA, index=panel.index, dtype="Int64")
        valid = future_return.notna()
        direction.loc[valid] = (future_return.loc[valid] > 0.0).astype("int64")
        panel[f"direction_{suffix}"] = direction

    future_closes: Dict[int, pd.Series] = {0: current_close}
    for h in range(1, MAX_HORIZON + 1):
        future_closes[h] = endpoints[h]["future_close"]

    one_step_logs = [
        np.log(future_closes[h] / future_closes[h - 1])
        for h in range(1, MAX_HORIZON + 1)
    ]
    matrix = np.column_stack(
        [
            s.to_numpy(dtype=float, na_value=np.nan)
            for s in one_step_logs
        ]
    )
    all_five = np.isfinite(matrix).all(axis=1)
    rv = np.full(len(panel), np.nan, dtype=float)
    rv[all_five] = np.std(matrix[all_five], axis=1, ddof=1)
    panel["realized_volatility_5d"] = rv

    return panel, endpoints


# =============================================================================
# AUDITS
# =============================================================================

def continuity_audit(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in FORECAST_HORIZONS:
        suffix = f"{h}d"
        valid = panel[f"direction_{suffix}"].notna()
        v = panel.loc[valid]
        if v.empty:
            raise RuntimeError(f"No valid {h}d targets.")

        diff = (
            v[f"target_market_session_id_{suffix}"].astype("int64")
            - v["market_session_id"].astype("int64")
        )
        cross = int(
            (
                v[f"target_continuity_segment_id_{suffix}"].astype("int64")
                != v["continuity_segment_id"].astype("int64")
            ).sum()
        )
        bad_date = int(
            (
                pd.to_datetime(v[f"target_date_{suffix}"])
                <= pd.to_datetime(v["Date"])
            ).sum()
        )
        labels = v[f"direction_{suffix}"].astype("int64")
        bad_label = int((~labels.isin([0, 1])).sum())

        if not (diff == h).all():
            raise RuntimeError(f"{h}d session-distance audit failed.")
        if cross or bad_date or bad_label:
            raise RuntimeError(f"{h}d continuity/date/label audit failed.")

        rows.append(
            {
                "forecast_horizon_sessions": h,
                "reference_rows": len(panel),
                "valid_target_rows": int(valid.sum()),
                "missing_target_rows": int((~valid).sum()),
                "session_difference_min": int(diff.min()),
                "session_difference_max": int(diff.max()),
                "cross_segment_targets": cross,
                "invalid_target_date_order": bad_date,
                "invalid_direction_labels": bad_label,
                "class_ratio_reported": False,
            }
        )
    return pd.DataFrame(rows)


def availability_audit(
    panel: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (ticker, seg), group in panel.groupby(
        ["Ticker", "continuity_segment_id"], sort=True
    ):
        group = group.sort_values("Date").reset_index(drop=True)

        available_ids = set(
            lookup.loc[
                (lookup["Ticker"] == ticker)
                & (lookup["target_continuity_segment_id"] == int(seg)),
                "market_session_id",
            ]
            .astype("int64")
            .tolist()
        )

        for h in FORECAST_HORIZONS:
            expected = np.asarray(
                [
                    int(x) + h in available_ids
                    for x in group["market_session_id"].astype("int64")
                ],
                dtype=bool,
            )
            observed = group[f"direction_{h}d"].notna().to_numpy(dtype=bool)

            if not np.array_equal(expected, observed):
                raise RuntimeError(
                    f"{ticker}, segment {seg}, {h}d availability mismatch."
                )

            rows.append(
                {
                    "ticker": ticker,
                    "continuity_segment_id": int(seg),
                    "segment_reference_rows": len(group),
                    "segment_start_date": group["Date"].min(),
                    "segment_end_date": group["Date"].max(),
                    "forecast_horizon_sessions": h,
                    "valid_target_rows": int(observed.sum()),
                    "missing_target_rows": int((~observed).sum()),
                    "structurally_expected_valid_rows": int(expected.sum()),
                    "structurally_expected_missing_rows": int((~expected).sum()),
                    "availability_exact_match": True,
                }
            )
    return pd.DataFrame(rows)


def auxiliary_audit(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for h in FORECAST_HORIZONS:
        direction_valid = panel[f"direction_{h}d"].notna().to_numpy(dtype=bool)
        for name in (
            f"future_close_{h}d",
            f"future_return_{h}d",
            f"future_log_return_{h}d",
        ):
            available = panel[name].notna().to_numpy(dtype=bool)
            if not np.array_equal(direction_valid, available):
                raise RuntimeError(f"Availability mismatch for {name}.")
            values = pd.to_numeric(
                panel.loc[available, name],
                errors="raise",
            ).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite values in {name}.")
            if name.startswith("future_close_") and (values <= 0).any():
                raise RuntimeError(f"Non-positive values in {name}.")

            rows.append(
                {
                    "target": name,
                    "available_rows": int(available.sum()),
                    "missing_rows": int((~available).sum()),
                    "availability_matches_direction": True,
                    "finite_when_available": True,
                }
            )

    rv_valid = panel["realized_volatility_5d"].notna().to_numpy(dtype=bool)
    d5_valid = panel["direction_5d"].notna().to_numpy(dtype=bool)
    if not np.array_equal(rv_valid, d5_valid):
        raise RuntimeError("5d realized-volatility availability mismatch.")

    rv = panel.loc[rv_valid, "realized_volatility_5d"].to_numpy(dtype=float)
    if not np.isfinite(rv).all() or (rv < 0).any():
        raise RuntimeError("Invalid realized_volatility_5d values.")

    rows.append(
        {
            "target": "realized_volatility_5d",
            "available_rows": int(rv_valid.sum()),
            "missing_rows": int((~rv_valid).sum()),
            "availability_matches_direction": True,
            "finite_when_available": True,
        }
    )
    return pd.DataFrame(rows)


def boundary_audit(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ticker, seg), group in panel.groupby(
        ["Ticker", "continuity_segment_id"], sort=True
    ):
        tail = group.sort_values("Date").tail(MAX_HORIZON)
        for row in tail.itertuples(index=False):
            rec = {
                "ticker": ticker,
                "continuity_segment_id": int(seg),
                "reference_date": row.Date,
                "reference_market_session_id": int(row.market_session_id),
            }
            for h in FORECAST_HORIZONS:
                rec[f"target_date_{h}d"] = getattr(row, f"target_date_{h}d")
                rec[f"direction_available_{h}d"] = bool(
                    pd.notna(getattr(row, f"direction_{h}d"))
                )
            rows.append(rec)
    return pd.DataFrame(rows)


def target_dictionary() -> pd.DataFrame:
    rows = []
    for h in FORECAST_HORIZONS:
        suffix = f"{h}d"
        rows.extend(
            [
                {
                    "column_name": f"target_market_session_id_{suffix}",
                    "target_role": "Target metadata",
                    "definition": (
                        f"Exact Step-03 master market-session ID {h} sessions "
                        "ahead when the same-stock endpoint exists in the same "
                        "continuity segment."
                    ),
                    "used_as_model_input": False,
                    "included_in_auxiliary_regression_vector": False,
                },
                {
                    "column_name": f"target_continuity_segment_id_{suffix}",
                    "target_role": "Target metadata",
                    "definition": "Continuity-segment ID of the valid endpoint.",
                    "used_as_model_input": False,
                    "included_in_auxiliary_regression_vector": False,
                },
                {
                    "column_name": f"target_date_{suffix}",
                    "target_role": "Target metadata",
                    "definition": f"Actual valid target date {h} market sessions ahead.",
                    "used_as_model_input": False,
                    "included_in_auxiliary_regression_vector": False,
                },
                {
                    "column_name": f"future_close_{suffix}",
                    "target_role": "Auxiliary regression target",
                    "definition": f"Frozen Step-05 Model_Close {h} master sessions ahead.",
                    "used_as_model_input": False,
                    "included_in_auxiliary_regression_vector": True,
                },
                {
                    "column_name": f"future_return_{suffix}",
                    "target_role": "Auxiliary regression target",
                    "definition": f"future_close_{suffix} / close_t - 1.",
                    "used_as_model_input": False,
                    "included_in_auxiliary_regression_vector": True,
                },
                {
                    "column_name": f"future_log_return_{suffix}",
                    "target_role": "Auxiliary regression target",
                    "definition": f"log(future_close_{suffix} / close_t).",
                    "used_as_model_input": False,
                    "included_in_auxiliary_regression_vector": True,
                },
                {
                    "column_name": f"direction_{suffix}",
                    "target_role": "Primary classification target",
                    "definition": (
                        f"1 if future_return_{suffix} > 0; 0 if <= 0; "
                        "NA when exact endpoint is unavailable."
                    ),
                    "used_as_model_input": False,
                    "included_in_auxiliary_regression_vector": False,
                },
            ]
        )

    rows.append(
        {
            "column_name": "realized_volatility_5d",
            "target_role": "Auxiliary regression target",
            "definition": (
                "Sample standard deviation (ddof=1) of the five consecutive "
                "future one-session Step-05 Model_Close log returns."
            ),
            "used_as_model_input": False,
            "included_in_auxiliary_regression_vector": True,
        }
    )
    return pd.DataFrame(rows)


def common_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = (
        panel[PRIMARY_TARGET_COLUMNS].notna().all(axis=1)
        & panel[AUXILIARY_TARGET_COLUMNS].notna().all(axis=1)
    )
    d5 = panel["direction_5d"].notna().to_numpy(dtype=bool)
    if not np.array_equal(mask.to_numpy(dtype=bool), d5):
        raise RuntimeError("Common multi-task mask differs from 5d availability.")

    common = panel.loc[mask].copy().reset_index(drop=True)
    if common.empty:
        raise RuntimeError("Common multi-task panel is empty.")

    per_stock = common.groupby("Ticker").size()
    if per_stock.nunique() != 1:
        raise RuntimeError("Common multi-task rows are not balanced by stock.")
    per_date = common.groupby("Date")["Ticker"].nunique()
    if not (per_date == EXPECTED_STOCK_COUNT).all():
        raise RuntimeError("Common multi-task rows are not balanced by date.")

    audit = pd.DataFrame(
        [
            {
                "reference_rows": len(panel),
                "common_multitask_rows": len(common),
                "common_dates": common["Date"].nunique(),
                "common_rows_per_stock": int(per_stock.iloc[0]),
                "minimum_stocks_per_common_date": int(per_date.min()),
                "maximum_stocks_per_common_date": int(per_date.max()),
                "common_mask_equals_5d_availability": True,
            }
        ]
    )
    return common, audit


# =============================================================================
# WRITE / HASH
# =============================================================================

def write_pair(frame: pd.DataFrame, parquet: Path, csv: Path) -> None:
    frame.to_parquet(
        parquet,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    c = frame.copy()
    for col in [
        x
        for x in c.columns
        if x == "Date" or x.startswith("target_date_")
    ]:
        c[col] = pd.to_datetime(c[col], errors="coerce").dt.strftime("%Y-%m-%d")

    c.to_csv(csv, index=False, float_format="%.17g")

    reread = pd.read_parquet(parquet, engine="pyarrow")
    if len(reread) != len(frame) or list(reread.columns) != list(frame.columns):
        raise RuntimeError(f"Parquet roundtrip mismatch: {parquet}")


def source_hashes() -> pd.DataFrame:
    rows = []

    sources = [
        ("current_step00", Path(cfg.__file__).resolve()),
        ("current_step09", SCRIPT_FILE.resolve()),
    ]
    for step, (manifest, sidecar) in UPSTREAM.items():
        sources.append((f"step{step:02d}_manifest", manifest.resolve()))
        sources.append((f"step{step:02d}_manifest_sha", sidecar.resolve()))

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


def write_failure(stage: str, exc: Exception, staging: Optional[Path]) -> Path:
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = FAILED_DIR / f"09_targets_FAILED_{stamp}.json"
    atomic_json(
        path,
        {
            "status": "FAIL",
            "step": STEP_NAME,
            "failed_at_utc": utc_now(),
            "stage": stage,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "staging_root": str(staging) if staging else None,
        },
    )
    return path


# =============================================================================
# VERIFY EXISTING
# =============================================================================

def verify_existing() -> int:
    print("=" * 128)
    print("STEP 09 — VERIFY EXISTING MULTI-HORIZON TARGET FREEZE")
    print("=" * 128)

    step00 = verify_step00()
    upstream = load_and_verify_upstream(step00)

    manifest_sha = verify_sidecar(MANIFEST_JSON, MANIFEST_SHA, "Step 09")
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))

    if manifest.get("step", {}).get("status") != "PASS":
        print("[FAIL] Stored Step-09 status is not PASS.")
        return 20

    for step in range(1, 9):
        field = f"step{step:02d}_manifest_sha256"
        if manifest.get(field) != upstream["sha"][step]:
            print(f"[FAIL] Step-09 chain mismatch: {field}")
            return 21

    if (
        manifest.get("configuration_fingerprint_sha256")
        != step00["fingerprint"]
    ):
        print("[FAIL] Step-09 configuration fingerprint mismatch.")
        return 22

    if (
        manifest.get("feature_interface_sha256")
        != upstream["feature_interface_sha256"]
    ):
        print("[FAIL] Step-09 feature-interface mismatch.")
        return 23

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
        print("=" * 128)
        print("STEP 09 VERIFICATION STATUS: FAIL")
        print("=" * 128)
        return 24

    print(f"[PASS] Configuration fingerprint : {step00['fingerprint']}")
    print(f"[PASS] Step-08 manifest          : {upstream['sha'][8]}")
    print(
        f"[PASS] 63-feature interface      : "
        f"{upstream['feature_interface_sha256']}"
    )
    print(f"[PASS] Step-09 manifest          : {manifest_sha}")
    print(f"[PASS] Final-test state          : {step00['state']}")
    print("=" * 128)
    print("STEP 09 VERIFICATION STATUS: PASS")
    print("=" * 128)
    return 0


# =============================================================================
# MAIN
# =============================================================================

def run() -> int:
    print("=" * 128)
    print(STEP_NAME)
    print("=" * 128)

    stage = "prerequisite_verification"
    staging: Optional[Path] = None

    try:
        print("[1/11] Verifying Step 00 and immutable Step 01 -> Step 08 chain...")
        step00 = verify_step00()
        upstream = load_and_verify_upstream(step00)
        print(f"       Configuration fingerprint: {step00['fingerprint']}")
        print(f"       Step-08 manifest          : {upstream['sha'][8]}")
        print(
            f"       63-feature interface      : "
            f"{upstream['feature_interface_sha256']}"
        )
        print(f"       Final-test state          : {step00['state']}")

        stage = "collision_guard"
        print("[2/11] Checking immutable Step-09 output target...")
        assert_fresh_target()
        cfg.create_directories()

        stage = "source_loading"
        print("[3/11] Loading master calendar, continuity policy and frozen sources...")
        policy = json.loads(STEP04_POLICY.read_text(encoding="utf-8"))
        breaks = load_break_dates(policy)
        master = load_master_calendar(breaks)
        ref = load_references(upstream["feature_columns"], master)
        lookup = load_price_lookup(master, breaks)

        print(f"       Master sessions      : {len(master):,}")
        print(f"       Reference rows       : {len(ref):,}")
        print(f"       Reference rows/stock : {ref.groupby('Ticker').size().iloc[0]:,}")
        print(
            "       Break dates          : "
            + (
                ", ".join(d.strftime("%Y-%m-%d") for d in breaks)
                if breaks
                else "NONE"
            )
        )
        print(f"       Horizons             : {FORECAST_HORIZONS}")

        stage = "reference_reconciliation"
        print("[4/11] Reconciling Step-08 close/segments to frozen Step-05 Model_Close...")
        ref_audit = reference_audit(ref, lookup)
        print(
            "       Price exact match     : "
            f"{bool(ref_audit.iloc[0]['all_reference_prices_match'])}"
        )
        print(
            "       Segment exact match   : "
            f"{bool(ref_audit.iloc[0]['all_continuity_ids_match'])}"
        )
        print(
            "       Maximum price diff    : "
            f"{ref_audit.iloc[0]['maximum_absolute_close_difference']:.3e}"
        )

        stage = "target_construction"
        print("[5/11] Constructing exact-master-session 1d/3d/5d targets...")
        panel, endpoints = construct_targets(ref, lookup)

        # Predictor immutability.
        if not np.array_equal(
            ref[upstream["feature_columns"]].to_numpy(dtype=float),
            panel[upstream["feature_columns"]].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise RuntimeError("63 predictor values changed during Step 09.")

        stage = "target_audits"
        print("[6/11] Running structural target/continuity/auxiliary audits...")
        cont = continuity_audit(panel)
        avail = availability_audit(panel, lookup)
        aux = auxiliary_audit(panel)
        bound = boundary_audit(panel)
        dictionary = target_dictionary()
        common, common_audit = common_panel(panel)

        target_names = set(dictionary["column_name"])
        overlap = sorted(target_names & set(upstream["feature_columns"]))
        if overlap:
            raise RuntimeError(f"Target/predictor column overlap: {overlap}")

        for row in cont.itertuples(index=False):
            print(
                f"       {row.forecast_horizon_sessions}d: "
                f"valid={row.valid_target_rows:,} | "
                f"missing={row.missing_target_rows:,} | "
                f"session-diff={row.session_difference_min}"
                f"..{row.session_difference_max} | "
                f"cross-segment={row.cross_segment_targets}"
            )
        print(f"       Common multi-task rows : {len(common):,}")
        print(
            f"       Common rows/stock      : "
            f"{common.groupby('Ticker').size().iloc[0]:,}"
        )
        print("       Direction class ratios : NOT REPORTED")

        stage = "staging"
        print("[7/11] Writing target panels to staging area...")
        staging = create_stage()

        target_pq = staging / TARGET_PANEL_PARQUET.name
        target_csv = staging / TARGET_PANEL_CSV.name
        common_pq = staging / COMMON_PANEL_PARQUET.name
        common_csv = staging / COMMON_PANEL_CSV.name

        write_pair(panel, target_pq, target_csv)
        write_pair(common, common_pq, common_csv)

        stage = "promotion"
        print("[8/11] Promoting Step-09 panels and writing audits...")
        if OUT_DIR.exists():
            raise RuntimeError("Canonical Step-09 output directory appeared unexpectedly.")
        os.replace(staging, OUT_DIR)
        staging = None

        write_csv(TARGET_DICTIONARY_CSV, dictionary)
        write_csv(CONTINUITY_AUDIT_CSV, cont)
        write_csv(AVAILABILITY_AUDIT_CSV, avail)
        write_csv(REFERENCE_AUDIT_CSV, ref_audit)
        write_csv(BOUNDARY_AUDIT_CSV, bound)
        write_csv(AUXILIARY_AUDIT_CSV, aux)
        write_csv(COMMON_AUDIT_CSV, common_audit)

        src_hashes = source_hashes()
        write_csv(SOURCE_HASH_CSV, src_hashes)

        hash_rows = []
        for scope, path in [
            ("annotated_target_panel_parquet", TARGET_PANEL_PARQUET),
            ("annotated_target_panel_csv", TARGET_PANEL_CSV),
            ("common_multitask_parquet", COMMON_PANEL_PARQUET),
            ("common_multitask_csv", COMMON_PANEL_CSV),
            ("target_dictionary", TARGET_DICTIONARY_CSV),
            ("continuity_audit", CONTINUITY_AUDIT_CSV),
            ("availability_audit", AVAILABILITY_AUDIT_CSV),
            ("reference_audit", REFERENCE_AUDIT_CSV),
            ("boundary_audit", BOUNDARY_AUDIT_CSV),
            ("auxiliary_audit", AUXILIARY_AUDIT_CSV),
            ("common_audit", COMMON_AUDIT_CSV),
            ("source_hashes", SOURCE_HASH_CSV),
        ]:
            hash_rows.append(
                {
                    "scope": scope,
                    "file_path": str(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
        write_csv(HASH_REGISTER_CSV, pd.DataFrame(hash_rows))

        stage = "freeze"
        print("[9/11] Freezing summary, source snapshot and manifest...")
        lock_after = sha256_file(Path(step00["lock_path"]))
        if lock_after != step00["lock_sha256"]:
            raise RuntimeError("Final-test lock changed during Step 09.")

        availability_summary = {
            f"{int(row.forecast_horizon_sessions)}d": {
                "valid_rows": int(row.valid_target_rows),
                "missing_rows": int(row.missing_target_rows),
            }
            for row in cont.itertuples(index=False)
        }

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
            "reference_rows": len(ref),
            "reference_rows_per_stock": int(ref.groupby("Ticker").size().iloc[0]),
            "master_market_sessions": len(master),
            "forecast_horizons_market_sessions": list(FORECAST_HORIZONS),
            "continuity_break_dates": [
                x.strftime("%Y-%m-%d") for x in breaks
            ],
            "canonical_target_price": "Step-05 Model_Close",
            "availability": availability_summary,
            "common_multitask_rows": len(common),
            "common_rows_per_stock": int(common.groupby("Ticker").size().iloc[0]),
            "directional_class_ratio_reported_here": False,
            "pipeline_state": {
                "chronological_split_assigned": False,
                "horizon_boundary_purging_applied": False,
                "scaling_applied": False,
                "regimes_constructed": False,
                "feature_selection_applied": False,
                "sequence_construction_performed": False,
                "model_selection_performed": False,
                "threshold_selection_performed": False,
                "heldout_class_distribution_inspected": False,
                "final_test_scoring_performed": False,
            },
            "final_test_lock": {
                "state": step00["state"],
                "lock_unchanged": True,
            },
        }
        atomic_json(SUMMARY_JSON, summary)

        shutil.copy2(SCRIPT_FILE, FROZEN_SCRIPT)

        atomic_text(
            SUMMARY_TXT,
            "\n".join(
                [
                    STEP_NAME,
                    "=" * 96,
                    "Status: PASS",
                    f"Reference rows: {len(ref)}",
                    f"Reference rows/stock: {ref.groupby('Ticker').size().iloc[0]}",
                    f"Master sessions: {len(master)}",
                    f"Horizons: {FORECAST_HORIZONS}",
                    f"Common multi-task rows: {len(common)}",
                    f"Common rows/stock: {common.groupby('Ticker').size().iloc[0]}",
                    "Directional class ratios reported: False",
                    f"Final-test state: {step00['state']}",
                ]
            )
            + "\n",
        )

        artifact_paths = [
            TARGET_PANEL_PARQUET,
            TARGET_PANEL_CSV,
            COMMON_PANEL_PARQUET,
            COMMON_PANEL_CSV,
            TARGET_DICTIONARY_CSV,
            CONTINUITY_AUDIT_CSV,
            AVAILABILITY_AUDIT_CSV,
            REFERENCE_AUDIT_CSV,
            BOUNDARY_AUDIT_CSV,
            AUXILIARY_AUDIT_CSV,
            COMMON_AUDIT_CSV,
            HASH_REGISTER_CSV,
            SOURCE_HASH_CSV,
            SUMMARY_JSON,
            FROZEN_SCRIPT,
            SUMMARY_TXT,
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
            "configuration_fingerprint_sha256": step00["fingerprint"],
            "step00_manifest_sha256": step00["manifest_sha256"],
            **{
                f"step{step:02d}_manifest_sha256": upstream["sha"][step]
                for step in range(1, 9)
            },
            "feature_interface_sha256": upstream["feature_interface_sha256"],
            "ordered_63_features": upstream["feature_columns"],
            "target_design": {
                "horizons_market_sessions": list(FORECAST_HORIZONS),
                "master_calendar": "Step-03 complete price-valid NIFTY calendar",
                "canonical_target_price": "Step-05 Model_Close",
                "same_ticker_required": True,
                "same_continuity_segment_required": True,
                "zero_return_class": 0,
                "unavailable_endpoint_remains_missing": True,
                "realized_volatility_ddof": 1,
                "directional_class_ratio_reported_here": False,
            },
            "availability": availability_summary,
            "common_multitask": {
                "rows": len(common),
                "rows_per_stock": int(common.groupby("Ticker").size().iloc[0]),
                "common_mask_equals_5d_availability": True,
            },
            "final_test_state": step00["state"],
            "artifacts": {
                p.name: artifact(p)
                for p in artifact_paths
            },
        }

        atomic_json(MANIFEST_JSON, manifest)
        manifest_sha = sha256_file(MANIFEST_JSON)
        atomic_text(
            MANIFEST_SHA,
            f"{manifest_sha}  {MANIFEST_JSON.name}\n",
        )

        print("[10/11] Validating frozen artifact set...")
        verify_sidecar(MANIFEST_JSON, MANIFEST_SHA, "Step 09")

        stage = "report"
        print("[11/11] Final Step-09 report...")
        print()
        print("-" * 128)
        print("MULTI-HORIZON TARGET PANEL")
        print("-" * 128)
        print(f"Reference rows                : {len(panel):,}")
        print(
            f"Reference rows/stock          : "
            f"{panel.groupby('Ticker').size().iloc[0]:,}"
        )
        print(f"Locked predictor count        : {len(upstream['feature_columns'])}")
        print(f"Auxiliary target count        : {len(AUXILIARY_TARGET_COLUMNS)}")
        print(f"Master price-valid sessions   : {len(master):,}")
        print(f"Forecast horizons             : {FORECAST_HORIZONS}")
        print("Canonical target price        : Step-05 Model_Close")

        print("-" * 128)
        print("TARGET AVAILABILITY")
        print("-" * 128)
        for row in cont.itertuples(index=False):
            print(
                f"{row.forecast_horizon_sessions}d: "
                f"valid={row.valid_target_rows:,} | "
                f"missing={row.missing_target_rows:,} | "
                f"session-diff={row.session_difference_min}"
                f"..{row.session_difference_max} | "
                f"cross-segment={row.cross_segment_targets}"
            )
        print(
            "5d realized-volatility valid : "
            f"{int(panel['realized_volatility_5d'].notna().sum()):,}"
        )
        print(f"Common multi-task rows        : {len(common):,}")
        print(
            f"Common rows/stock             : "
            f"{common.groupby('Ticker').size().iloc[0]:,}"
        )

        print("-" * 128)
        print("REFERENCE RECONCILIATION")
        print("-" * 128)
        print("Step-08 close == Step-05 Model_Close : True")
        print("Continuity IDs exact                 : True")
        print(
            "Maximum close difference             : "
            f"{ref_audit.iloc[0]['maximum_absolute_close_difference']:.3e}"
        )

        print("-" * 128)
        print("HELD-OUT-TEST DISCIPLINE")
        print("-" * 128)
        print("Directional class ratios reported    : False")
        print("Held-out label distribution inspected: False")
        print("Final test scored                     : False")

        print("-" * 128)
        print("PIPELINE STATE")
        print("-" * 128)
        print("Chronological split assigned : False")
        print("Split-boundary purge applied : False")
        print("Scaling applied              : False")
        print("Regimes constructed          : False")
        print("Sequences constructed        : False")
        print("Feature selection applied    : False")

        print("-" * 128)
        print("OUTPUT FILES")
        print("-" * 128)
        print(f"Target panel          : {TARGET_PANEL_PARQUET}")
        print(f"Common multi-task     : {COMMON_PANEL_PARQUET}")
        print(f"Target dictionary     : {TARGET_DICTIONARY_CSV}")
        print(f"Continuity audit      : {CONTINUITY_AUDIT_CSV}")
        print(f"Availability audit    : {AVAILABILITY_AUDIT_CSV}")
        print(f"Manifest              : {MANIFEST_JSON}")
        print(f"Manifest SHA-256      : {manifest_sha}")
        print(f"Final-test state      : {step00['state']}")
        print("=" * 128)
        print("PASS: Targets use exact future Step-03 master-market session IDs.")
        print("PASS: No target crosses a ticker or continuity-segment boundary.")
        print(
            "PASS: Direction labels are binary where exact endpoints exist; "
            "unavailable endpoints remain NA."
        )
        print(
            "PASS: Five-session realized volatility uses five consecutive "
            "future log returns with ddof=1."
        )
        print("PASS: The locked 63 predictor values were preserved exactly.")
        print("PASS: No held-out class distribution or model performance was inspected.")
        print("PASS: Final held-out test remains LOCKED_NOT_OPENED.")
        print("=" * 128)
        print("Do not run Step 10 yet. Paste the complete terminal output for audit.")

        return 0

    except Exception as exc:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

        archived = archive_partial()
        failure = write_failure(stage, exc, staging)

        print()
        print("=" * 128)
        print("STEP 09 FAILED — TARGETS NOT FROZEN")
        print("=" * 128)
        print(f"Stage      : {stage}")
        print(f"Error type : {type(exc).__name__}")
        print(f"Error      : {exc}")
        print(f"Diagnostic : {failure}")
        if archived:
            print(f"Archived partial artifacts: {archived}")
        print("=" * 128)
        print("Do not proceed to Step 10.")
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
        print("\nStep 09 interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print("=" * 128)
        print("STEP 09 UNHANDLED FAILURE")
        print("=" * 128)
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
