#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 14 — CLEAN V3 MODEL-READY SEQUENCE ARRAY MATERIALIZATION

Materializes the frozen Step-13 common-multitask sequence endpoints into:
    X_full       (n,60,63) float32
    X_stock      (n,60,47) float32
    X_market     (n,60,16) float32
    y_direction  (n,3)     int64     [1d,3d,5d]
    y_aux_raw    (n,10)    float32
    y_aux_scaled (n,10)    float32
    stock_id     (n,)      int64
    regime_id    (n,)      int64

For each fold, the 10 auxiliary targets are MinMax-scaled from TRAIN common
sequence endpoints only (clip=False), then the frozen transform is applied to
validation. A separate final-development auxiliary scaler is fitted on the
final-development common endpoints. Direction labels are never scaled.

No sequence membership is redefined. No feature selection, model fitting,
probability-threshold tuning, or final-test access occurs here.

Run:
    python step14_materialize_model_arrays_v3.py
Verify:
    python step14_materialize_model_arrays_v3.py --verify-only
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
import sklearn
from sklearn.preprocessing import MinMaxScaler

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent
if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError("Place this script inside <repository>/q1_revision/.")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import step00_config as cfg

STEP_NUMBER = 14
STEP_ID = "14"
STEP_NAME = "STEP 14 — MODEL-READY 60-SESSION ARRAY MATERIALIZATION"
SCHEMA_VERSION = "1.0"
LOOKBACK = 60
FOLDS = (1, 2, 3, 4, 5)
PARTITIONS = ("train", "validation")
STOCK_FEATURE_COUNT = 47
MARKET_FEATURE_COUNT = 16
TOTAL_FEATURE_COUNT = 63
DIRECTION_COLUMNS = ["direction_1d", "direction_3d", "direction_5d"]
AUX_COLUMNS = [
    "future_close_1d", "future_close_3d", "future_close_5d",
    "future_return_1d", "future_return_3d", "future_return_5d",
    "future_log_return_1d", "future_log_return_3d", "future_log_return_5d",
    "realized_volatility_5d",
]
FORBIDDEN_TEST_FILENAME = "10_final_test_locked_panel.parquet"

STEP01_ENV = cfg.ENVIRONMENT_DIR / "01_environment_audit.json"
UPSTREAM = {
    1: (cfg.MANIFEST_DIR / "01_environment_audit_manifest.json", cfg.MANIFEST_DIR / "01_environment_audit_manifest.sha256"),
    2: (cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.json", cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.sha256"),
    3: (cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.json", cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.sha256"),
    4: (cfg.MANIFEST_DIR / "04_session_policy_manifest.json", cfg.MANIFEST_DIR / "04_session_policy_manifest.sha256"),
    5: (cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.json", cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.sha256"),
    6: (cfg.MANIFEST_DIR / "06_stock_feature_manifest.json", cfg.MANIFEST_DIR / "06_stock_feature_manifest.sha256"),
    7: (cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.json", cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.sha256"),
    8: (cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.json", cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.sha256"),
    9: (cfg.MANIFEST_DIR / "09_target_manifest.json", cfg.MANIFEST_DIR / "09_target_manifest.sha256"),
    10: (cfg.MANIFEST_DIR / "10_chronological_split_manifest.json", cfg.MANIFEST_DIR / "10_chronological_split_manifest.sha256"),
    11: (cfg.MANIFEST_DIR / "11_preprocessing_manifest.json", cfg.MANIFEST_DIR / "11_preprocessing_manifest.sha256"),
    12: (cfg.MANIFEST_DIR / "12_volatility_regime_manifest.json", cfg.MANIFEST_DIR / "12_volatility_regime_manifest.sha256"),
    13: (cfg.MANIFEST_DIR / "13_sequence_index_manifest.json", cfg.MANIFEST_DIR / "13_sequence_index_manifest.sha256"),
}
STEP10_DEVELOPMENT = cfg.FEATURE_DATA_DIR / "step10_chronological_splits" / "development" / "10_development_target_panel.parquet"
STEP10_TEST_SEAL = cfg.MANIFEST_DIR / "10_final_test_seal.json"
STEP12_ROOT = cfg.FEATURE_DATA_DIR / "step12_volatility_regimes"
STEP12_FINAL = STEP12_ROOT / "final_development" / "12_final_development_scaled_regimes.parquet"
STEP13_ROOT = cfg.FEATURE_DATA_DIR / "step13_sequence_indices"
STEP13_FINAL = STEP13_ROOT / "final_development" / "13_final_development_sequence_index.parquet"

OUT_DIR = cfg.FEATURE_DATA_DIR / "step14_model_arrays"
FINAL_DIR = OUT_DIR / "final_development"
ARRAY_SUMMARY_CSV = cfg.AUDIT_DIR / "14_array_summary.csv"
AUX_SCALER_AUDIT_CSV = cfg.AUDIT_DIR / "14_auxiliary_scaler_audit.csv"
AUX_VALIDATION_RANGE_CSV = cfg.AUDIT_DIR / "14_auxiliary_validation_range_audit.csv"
AUX_INVERSE_AUDIT_CSV = cfg.AUDIT_DIR / "14_auxiliary_inverse_transform_audit.csv"
ARRAY_INTEGRITY_CSV = cfg.AUDIT_DIR / "14_array_integrity_audit.csv"
ENDPOINT_RECONCILIATION_CSV = cfg.AUDIT_DIR / "14_endpoint_reconciliation_audit.csv"
SOURCE_HASH_CSV = cfg.AUDIT_DIR / "14_source_file_hashes.csv"
HASH_REGISTER_CSV = cfg.AUDIT_DIR / "14_array_hashes.csv"
SUMMARY_JSON = cfg.AUDIT_DIR / "14_array_summary.json"
MANIFEST_JSON = cfg.MANIFEST_DIR / "14_model_array_manifest.json"
MANIFEST_SHA = cfg.MANIFEST_DIR / "14_model_array_manifest.sha256"
FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step14_materialize_model_arrays_frozen.py"
SUMMARY_TXT = cfg.REPORT_DIR / "14_model_array_summary.txt"
FAILED_DIR = cfg.AUDIT_DIR / "step14_failed_attempts"


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jdefault(x: Any):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return None if np.isnan(x) else float(x)
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, (pd.Timestamp, datetime)): return x.isoformat()
    if x is pd.NA: return None
    raise TypeError(type(x).__name__)


def atomic_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False, default=jdefault)
        f.write("\n")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, date_format="%Y-%m-%d", float_format="%.17g")
    os.replace(tmp, path)


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(cfg.EXPERIMENT_ROOT_DIR.resolve()))


def artifact(path: Path):
    return {"path": str(path), "relative_path": rel(path), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def pkgver(name: str):
    try: return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError: return None


def verify_sidecar(manifest: Path, sidecar: Path, label: str) -> str:
    if not manifest.exists() or not sidecar.exists():
        raise RuntimeError(f"{label} manifest/sidecar missing.")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    observed = sha256_file(manifest)
    if expected != observed:
        raise RuntimeError(f"{label} manifest SHA mismatch: expected={expected}, observed={observed}")
    return observed


def find_artifact(manifest: dict, filename: str) -> dict:
    hits = []
    for key in ("artifacts", "canonical_artifacts"):
        for rec in manifest.get(key, {}).values():
            if isinstance(rec, dict) and rec.get("path") and Path(rec["path"]).name == filename:
                hits.append(rec)
    if len(hits) != 1:
        raise RuntimeError(f"Expected one artifact named {filename}; found {len(hits)}.")
    return hits[0]


def verify_step00():
    cfg.validate_configuration()
    payload = cfg.build_configuration_payload()
    fp = cfg.canonical_json_sha256(payload)
    _, manifest_sha, status = cfg.verify_existing_freeze(
        configuration_payload=payload,
        configuration_fingerprint=fp,
    )
    if status != "VERIFIED_EXISTING":
        raise RuntimeError("Step-00 freeze is not VERIFIED_EXISTING.")
    if int(getattr(cfg, "LOOKBACK", LOOKBACK)) != LOOKBACK:
        raise RuntimeError("Frozen lookback is not 60.")
    lock_path = cfg.get_final_test_lock_path()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("state") != "LOCKED_NOT_OPENED" or lock.get("final_test_opened") is not False or lock.get("final_test_scored") is not False:
        raise RuntimeError("Global final-test lock is not pristine.")
    return {"fingerprint": fp, "step00_manifest_sha": manifest_sha, "lock_path": str(lock_path), "lock_sha": sha256_file(lock_path), "state": lock["state"]}


def verify_environment():
    audit = json.loads(STEP01_ENV.read_text(encoding="utf-8"))
    expected_python = audit.get("system", {}).get("python_executable")
    if expected_python and Path(expected_python).resolve() != Path(sys.executable).resolve():
        raise RuntimeError(f"Wrong Python environment. frozen={expected_python}, current={sys.executable}")
    drift = []
    for row in audit.get("packages", []):
        if row.get("required", False):
            cur = pkgver(row.get("distribution"))
            if cur != row.get("version"):
                drift.append((row.get("distribution"), row.get("version"), cur))
    if drift:
        raise RuntimeError("Required-package drift:\n" + "\n".join(f"{n}: frozen={a}, current={b}" for n, a, b in drift))


def step12_path(fold: int, partition: str) -> Path:
    return STEP12_ROOT / f"fold_{fold}" / f"12_fold_{fold}_{partition}_scaled_regimes.parquet"


def step13_path(fold: int, partition: str) -> Path:
    return STEP13_ROOT / f"fold_{fold}" / f"13_fold_{fold}_{partition}_sequence_index.parquet"


def verify_upstream(step00: dict):
    shas, manifests = {}, {}
    for step in range(1, 14):
        mp, sp = UPSTREAM[step]
        shas[step] = verify_sidecar(mp, sp, f"Step {step:02d}")
        manifests[step] = json.loads(mp.read_text(encoding="utf-8"))

    if manifests[1].get("status") != "PASS" or manifests[1].get("configuration_fingerprint_sha256") != step00["fingerprint"]:
        raise RuntimeError("Step-01 verification failed.")
    if manifests[2].get("step", {}).get("status") != "PASS":
        raise RuntimeError("Step-02 status is not PASS.")
    if manifests[2].get("step00", {}).get("configuration_fingerprint_sha256") != step00["fingerprint"]:
        raise RuntimeError("Step-02 fingerprint mismatch.")
    if manifests[2].get("step01", {}).get("step01_manifest_sha256") != shas[1]:
        raise RuntimeError("Step-02 -> Step-01 chain mismatch.")

    for step in range(3, 14):
        m = manifests[step]
        if m.get("step", {}).get("status") != "PASS":
            raise RuntimeError(f"Step-{step:02d} status is not PASS.")
        if m.get("configuration_fingerprint_sha256") != step00["fingerprint"]:
            raise RuntimeError(f"Step-{step:02d} fingerprint mismatch.")
        for prev in range(1, step):
            k = f"step{prev:02d}_manifest_sha256"
            if k in m and m[k] != shas[prev]:
                raise RuntimeError(f"Step-{step:02d} chain mismatch: {k}")

    verify_environment()

    # ------------------------------------------------------------------
    # CLEAN-V3 FEATURE INTERFACE
    # ------------------------------------------------------------------
    # Step 08 freezes the feature contract at TOP LEVEL:
    #   ordered_63_features
    #   feature_interface_sha256
    #
    # The canonical Step-08 construction is stock-first (47) followed by
    # NIFTY context (16). Do not require the obsolete nested
    # `feature_interface` object used by older scripts.
    all_features = manifests[8].get("ordered_63_features")
    interface_sha = manifests[8].get("feature_interface_sha256")

    if not isinstance(all_features, list) or len(all_features) != TOTAL_FEATURE_COUNT:
        raise RuntimeError(
            "Step-08 does not expose exactly 63 ordered predictors at top level."
        )

    if not isinstance(interface_sha, str) or not interface_sha:
        raise RuntimeError(
            "Step-08 does not expose a top-level feature_interface_sha256."
        )

    stock_features = list(all_features[:STOCK_FEATURE_COUNT])
    market_features = list(all_features[STOCK_FEATURE_COUNT:])

    if len(stock_features) != STOCK_FEATURE_COUNT:
        raise RuntimeError("Derived stock-feature block is not exactly 47 features.")

    if len(market_features) != MARKET_FEATURE_COUNT:
        raise RuntimeError("Derived NIFTY-feature block is not exactly 16 features.")

    # Step-07's frozen context features are explicitly NIFTY-prefixed.
    # This prevents a silent modality-order drift.
    if any(str(name).startswith("nifty_") for name in stock_features):
        raise RuntimeError(
            "Step-08 ordering drift: a NIFTY feature appears inside the first 47 predictors."
        )

    if not all(str(name).startswith("nifty_") for name in market_features):
        raise RuntimeError(
            "Step-08 ordering drift: the final 16 predictors are not all NIFTY context features."
        )

    if all_features != stock_features + market_features:
        raise RuntimeError("Step-08 63-feature modality ordering is invalid.")
    for step in (9, 10, 11, 12, 13):
        if manifests[step].get("ordered_63_features") != all_features or manifests[step].get("feature_interface_sha256") != interface_sha:
            raise RuntimeError(f"Step-{step:02d} feature-interface drift.")

    split_sha = manifests[13].get("split_design_sha256")
    if not split_sha:
        raise RuntimeError("Step-13 split-design SHA missing.")
    for step in (10, 11, 12, 13):
        if manifests[step].get("split_design_sha256") != split_sha:
            raise RuntimeError(f"Split-design SHA drift at Step {step:02d}.")

    dev_rec = find_artifact(manifests[10], STEP10_DEVELOPMENT.name)
    if Path(dev_rec["path"]).resolve() != STEP10_DEVELOPMENT.resolve() or sha256_file(STEP10_DEVELOPMENT) != dev_rec["sha256"]:
        raise RuntimeError("Step-10 development-panel identity failed.")
    dev_sha = dev_rec["sha256"]

    step12_hashes, step13_hashes = {}, {}
    for fold in FOLDS:
        for partition in PARTITIONS:
            p12 = step12_path(fold, partition)
            r12 = find_artifact(manifests[12], p12.name)
            if Path(r12["path"]).resolve() != p12.resolve() or sha256_file(p12) != r12["sha256"]:
                raise RuntimeError(f"Step-12 artifact mismatch: {p12.name}")
            step12_hashes[f"fold_{fold}_{partition}"] = r12["sha256"]

            p13 = step13_path(fold, partition)
            r13 = find_artifact(manifests[13], p13.name)
            if Path(r13["path"]).resolve() != p13.resolve() or sha256_file(p13) != r13["sha256"]:
                raise RuntimeError(f"Step-13 artifact mismatch: {p13.name}")
            step13_hashes[f"fold_{fold}_{partition}"] = r13["sha256"]

    r12f = find_artifact(manifests[12], STEP12_FINAL.name)
    r13f = find_artifact(manifests[13], STEP13_FINAL.name)
    final12_sha = sha256_file(STEP12_FINAL)
    final13_sha = sha256_file(STEP13_FINAL)
    if final12_sha != r12f["sha256"] or final13_sha != r13f["sha256"]:
        raise RuntimeError("Final-development Step-12/13 source hash mismatch.")

    expected_stock_id_map = {ticker: int(i) for i, ticker in enumerate(cfg.STOCK_TICKERS)}
    if manifests[13].get("stock_id_map") != expected_stock_id_map:
        raise RuntimeError("Step-13 stock_id_map differs from Step-00 ticker order.")

    seal = json.loads(STEP10_TEST_SEAL.read_text(encoding="utf-8"))
    if seal.get("status") != "SEALED_NOT_OPENED" or seal.get("global_final_test_lock_state") != "LOCKED_NOT_OPENED":
        raise RuntimeError("Step-10 test seal is not pristine.")

    return {
        "shas": shas, "manifests": manifests,
        "stock_features": stock_features, "market_features": market_features, "all_features": all_features,
        "feature_interface_sha": interface_sha,
        "feature_partition_policy": "first_47_stock_then_last_16_nifty",
        "split_design_sha": split_sha,
        "development_sha": dev_sha, "step12_hashes": step12_hashes, "step13_hashes": step13_hashes,
        "final12_sha": final12_sha, "final13_sha": final13_sha,
        "stock_id_map": expected_stock_id_map,
        "test_seal_status": seal["status"], "sealed_test_sha_metadata_only": seal.get("sha256"),
    }


def canonical_outputs():
    return (
        OUT_DIR, ARRAY_SUMMARY_CSV, AUX_SCALER_AUDIT_CSV, AUX_VALIDATION_RANGE_CSV,
        AUX_INVERSE_AUDIT_CSV, ARRAY_INTEGRITY_CSV, ENDPOINT_RECONCILIATION_CSV,
        SOURCE_HASH_CSV, HASH_REGISTER_CSV, SUMMARY_JSON, MANIFEST_JSON, MANIFEST_SHA,
        FROZEN_SCRIPT, SUMMARY_TXT,
    )


def assert_fresh():
    collisions = [p for p in canonical_outputs() if p.exists()]
    if collisions:
        raise RuntimeError("Step-14 outputs already exist; refusing overwrite.\n" + "\n".join(str(p) for p in collisions) + "\nUse --verify-only if already frozen.")


def create_stage():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = cfg.FEATURE_DATA_DIR / f"_step14_staging_{stamp}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    for fold in FOLDS: (root / f"fold_{fold}").mkdir()
    (root / "final_development").mkdir()
    return root


def archive_partial():
    if MANIFEST_JSON.exists(): return None
    existing = [p for p in canonical_outputs() if p.exists()]
    if not existing: return None
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    dest = FAILED_DIR / f"partial_artifacts_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    dest.mkdir(exist_ok=False)
    for p in existing:
        try: shutil.move(str(p), str(dest / p.name))
        except Exception: pass
    return dest


def normalize_dates(df: pd.DataFrame):
    out = df.copy()
    for c in ("Date", "reference_date", "sequence_start_date", "sequence_end_date", "target_date_1d", "target_date_3d", "target_date_5d"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce" if c.startswith("target_date_") or c == "sequence_start_date" else "raise").dt.normalize()
    return out


def load_development():
    df = normalize_dates(pd.read_parquet(STEP10_DEVELOPMENT, engine="pyarrow"))
    required = {"Ticker", "Date", "target_date_1d", "target_date_3d", "target_date_5d", *DIRECTION_COLUMNS, *AUX_COLUMNS}
    missing = required - set(df.columns)
    if missing: raise RuntimeError(f"Step-10 development targets missing: {sorted(missing)}")
    if df[["Ticker", "Date"]].duplicated().any(): raise RuntimeError("Duplicate Step-10 development keys.")
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def load_panel(path: Path, features: list[str], label: str):
    if path.name == FORBIDDEN_TEST_FILENAME: raise RuntimeError("Forbidden final-test parquet requested.")
    df = normalize_dates(pd.read_parquet(path, engine="pyarrow"))
    required = {"Ticker", "Date", "market_session_id", "continuity_segment_id", "regime_id", "regime_name", "regime_threshold_fit_scope", *features}
    missing = required - set(df.columns)
    if missing: raise RuntimeError(f"{label}: missing columns: {sorted(missing)}")
    if df[["Ticker", "Date"]].duplicated().any(): raise RuntimeError(f"{label}: duplicate keys.")
    if not df["regime_id"].isin([0, 1, 2]).all(): raise RuntimeError(f"{label}: invalid regime IDs.")
    if not np.isfinite(df[features].to_numpy(dtype=np.float64)).all(): raise RuntimeError(f"{label}: non-finite predictors.")
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def load_index(path: Path, label: str):
    df = normalize_dates(pd.read_parquet(path, engine="pyarrow"))
    required = {
        "fold", "partition", "ticker", "stock_id", "continuity_segment_id", "reference_date",
        "reference_market_session_id", "sequence_start_date", "sequence_end_date", "sequence_length",
        "sequence_history_available", "lookback_crosses_partition_start", "historical_rows_before_partition",
        "regime_id", "regime_name", "regime_threshold_fit_scope", "target_date_1d", "target_date_3d", "target_date_5d",
        "sequence_eligible_common_multitask",
    }
    missing = required - set(df.columns)
    if missing: raise RuntimeError(f"{label}: missing Step-13 columns: {sorted(missing)}")
    if df[["ticker", "reference_date"]].duplicated().any(): raise RuntimeError(f"{label}: duplicate endpoints.")
    common = df.loc[df["sequence_eligible_common_multitask"]].copy().sort_values(["reference_date", "ticker"]).reset_index(drop=True)
    if common.empty or not common["sequence_history_available"].all() or not common["sequence_length"].eq(LOOKBACK).all():
        raise RuntimeError(f"{label}: invalid common sequence subset.")
    return common


def attach_targets(index: pd.DataFrame, development: pd.DataFrame, panel: pd.DataFrame, label: str):
    tcols = ["Ticker", "Date", "target_date_1d", "target_date_3d", "target_date_5d", *DIRECTION_COLUMNS, *AUX_COLUMNS]
    targets = development[tcols].rename(columns={
        "Ticker": "ticker", "Date": "reference_date",
        "target_date_1d": "dev_target_date_1d", "target_date_3d": "dev_target_date_3d", "target_date_5d": "dev_target_date_5d",
    })
    out = index.merge(targets, on=["ticker", "reference_date"], how="left", validate="one_to_one")
    if out[DIRECTION_COLUMNS + AUX_COLUMNS].isna().any().any(): raise RuntimeError(f"{label}: missing endpoint targets.")
    for h in (1, 3, 5):
        if not pd.to_datetime(out[f"target_date_{h}d"]).equals(pd.to_datetime(out[f"dev_target_date_{h}d"])):
            raise RuntimeError(f"{label}: {h}d target-date mismatch.")
    for c in DIRECTION_COLUMNS:
        if not out[c].isin([0, 1]).all(): raise RuntimeError(f"{label}: non-binary {c}.")
    if not np.isfinite(out[AUX_COLUMNS].to_numpy(dtype=np.float64)).all(): raise RuntimeError(f"{label}: non-finite auxiliary target.")

    endpoint = panel[["Ticker", "Date", "regime_id", "continuity_segment_id", "market_session_id"]].rename(columns={
        "Ticker": "ticker", "Date": "reference_date", "regime_id": "p_regime_id",
        "continuity_segment_id": "p_continuity_segment_id", "market_session_id": "p_market_session_id",
    })
    out = out.merge(endpoint, on=["ticker", "reference_date"], how="left", validate="one_to_one")
    regime_ok = np.array_equal(out["regime_id"].to_numpy(np.int64), out["p_regime_id"].to_numpy(np.int64))
    cont_ok = np.array_equal(out["continuity_segment_id"].to_numpy(np.int64), out["p_continuity_segment_id"].to_numpy(np.int64))
    session_ok = np.array_equal(out["reference_market_session_id"].to_numpy(np.int64), out["p_market_session_id"].to_numpy(np.int64))
    if not (regime_ok and cont_ok and session_ok): raise RuntimeError(f"{label}: Step-12/13 endpoint reconciliation failed.")
    out["array_row_index"] = np.arange(len(out), dtype=np.int64)
    return out, {
        "scope": label, "rows": int(len(out)), "target_dates_exact_step10_to_step13": True,
        "regime_ids_exact_step12_to_step13": regime_ok, "continuity_ids_exact_step12_to_step13": cont_ok,
        "market_session_ids_exact_step12_to_step13": session_ok, "direction_labels_binary": True,
        "auxiliary_targets_finite": True,
    }


def build_history(train_panel: pd.DataFrame, validation_panel: Optional[pd.DataFrame], partition: str, features: list[str]):
    if partition in ("train", "final_development"):
        hdf = train_panel.copy()
    elif partition == "validation":
        hdf = pd.concat([train_panel, validation_panel], ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    else:
        raise RuntimeError(f"Unknown partition {partition}")
    if hdf[["Ticker", "Date"]].duplicated().any(): raise RuntimeError(f"{partition}: duplicate history keys.")
    lookup = {}
    for (ticker, segment), g in hdf.groupby(["Ticker", "continuity_segment_id"], sort=True):
        g = g.sort_values("Date").reset_index(drop=True)
        dates = list(g["Date"])
        lookup[(ticker, int(segment))] = {
            "dates": dates,
            "positions": {pd.Timestamp(d): i for i, d in enumerate(dates)},
            "matrix": g[features].to_numpy(dtype=np.float32),
            "session_ids": g["market_session_id"].to_numpy(dtype=np.int64),
        }
    return lookup


def materialize_x(endpoints: pd.DataFrame, history: dict):
    X = np.empty((len(endpoints), LOOKBACK, TOTAL_FEATURE_COUNT), dtype=np.float32)
    for row in endpoints.itertuples(index=False):
        key = (row.ticker, int(row.continuity_segment_id))
        if key not in history: raise RuntimeError(f"Missing history for {key}.")
        h = history[key]
        ref = pd.Timestamp(row.reference_date)
        if ref not in h["positions"]: raise RuntimeError(f"Reference not found: {row.ticker} {ref}")
        end = int(h["positions"][ref]); start = end - LOOKBACK + 1
        if start < 0: raise RuntimeError("Negative sequence start position.")
        if pd.Timestamp(h["dates"][start]) != pd.Timestamp(row.sequence_start_date): raise RuntimeError("Sequence-start mismatch with Step 13.")
        seq = h["matrix"][start:end+1]
        sess = h["session_ids"][start:end+1]
        if seq.shape != (LOOKBACK, TOTAL_FEATURE_COUNT): raise RuntimeError(f"Invalid sequence shape {seq.shape}.")
        if not np.all(np.diff(sess) == 1): raise RuntimeError("Non-consecutive master sessions in sequence.")
        if int(sess[-1]) != int(row.reference_market_session_id): raise RuntimeError("Sequence does not end at frozen endpoint.")
        if not np.isfinite(seq).all(): raise RuntimeError("Non-finite sequence values.")
        X[int(row.array_row_index)] = seq
    return X


def fit_aux(endpoints: pd.DataFrame, scope: str):
    raw = endpoints[AUX_COLUMNS].astype(np.float64)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0), copy=True, clip=False)
    scaled = scaler.fit_transform(raw)
    if scaler.n_features_in_ != len(AUX_COLUMNS): raise RuntimeError(f"{scope}: auxiliary scaler feature-count mismatch.")
    if hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) != AUX_COLUMNS: raise RuntimeError(f"{scope}: auxiliary target order mismatch.")
    if np.any(np.asarray(scaler.data_range_) == 0.0): raise RuntimeError(f"{scope}: constant auxiliary target detected.")
    if not np.allclose(scaled.min(0), 0.0, rtol=0.0, atol=1e-10) or not np.allclose(scaled.max(0), 1.0, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"{scope}: training auxiliary scale is not [0,1].")
    return scaler, scaled


def aux_inverse(scaler, endpoints, scaled, scope, fold, partition):
    raw = endpoints[AUX_COLUMNS].to_numpy(np.float64)
    back = scaler.inverse_transform(scaled)
    diff = np.abs(back - raw)
    norm_base = np.maximum.reduce([np.ones(raw.shape[1]), np.max(np.abs(raw), axis=0), np.asarray(scaler.data_range_, dtype=np.float64)])
    max_abs = float(diff.max()); max_norm = float((diff / norm_base.reshape(1, -1)).max())
    if max_norm > 1e-12: raise RuntimeError(f"{scope}: auxiliary inverse audit failed: {max_norm:.3e}")
    return {"scope": scope, "fold": np.nan if fold is None else fold, "partition": partition, "rows": int(len(endpoints)), "maximum_absolute_difference": max_abs, "maximum_normalized_difference": max_norm, "normalized_tolerance": 1e-12, "passed": True}


def scaler_rows(scaler, scope, fold, fit_rows):
    rows = []
    for i, target in enumerate(AUX_COLUMNS):
        rows.append({
            "scope": scope, "fold": np.nan if fold is None else fold, "target_number": i+1, "target": target, "fit_rows": int(fit_rows),
            "train_data_min": float(scaler.data_min_[i]), "train_data_max": float(scaler.data_max_[i]), "train_data_range": float(scaler.data_range_[i]),
            "scaler_scale": float(scaler.scale_[i]), "scaler_min": float(scaler.min_[i]), "feature_range_low": 0.0, "feature_range_high": 1.0,
            "clip": False, "validation_used_in_fit": False, "final_test_used_in_fit": False,
        })
    return rows


def validation_range_rows(fold, scaled):
    rows = []
    for i, target in enumerate(AUX_COLUMNS):
        v = scaled[:, i]; below = int((v < 0).sum()); above = int((v > 1).sum())
        rows.append({"fold": fold, "target": target, "validation_min": float(v.min()), "validation_max": float(v.max()), "below_zero_count": below, "above_one_count": above, "outside_0_1_count": below + above, "clip": False, "scaler_refit_on_validation": False})
    return rows


def build_arrays(endpoints, X_full, aux_scaled):
    return {
        "X_full": X_full,
        "X_stock": X_full[:, :, :STOCK_FEATURE_COUNT].copy(),
        "X_market": X_full[:, :, STOCK_FEATURE_COUNT:].copy(),
        "y_direction": endpoints[DIRECTION_COLUMNS].to_numpy(dtype=np.int64),
        "y_aux_raw": endpoints[AUX_COLUMNS].to_numpy(dtype=np.float32),
        "y_aux_scaled": aux_scaled.astype(np.float32),
        "stock_id": endpoints["stock_id"].to_numpy(dtype=np.int64),
        "regime_id": endpoints["regime_id"].to_numpy(dtype=np.int64),
    }


def audit_arrays(arrays, n, scope, fold, partition):
    shapes = {
        "X_full": (n, 60, 63), "X_stock": (n, 60, 47), "X_market": (n, 60, 16),
        "y_direction": (n, 3), "y_aux_raw": (n, 10), "y_aux_scaled": (n, 10), "stock_id": (n,), "regime_id": (n,),
    }
    dtypes = {"X_full": np.float32, "X_stock": np.float32, "X_market": np.float32, "y_direction": np.int64, "y_aux_raw": np.float32, "y_aux_scaled": np.float32, "stock_id": np.int64, "regime_id": np.int64}
    for name in shapes:
        if arrays[name].shape != shapes[name] or arrays[name].dtype != np.dtype(dtypes[name]):
            raise RuntimeError(f"{scope}: {name} shape/dtype mismatch: {arrays[name].shape}, {arrays[name].dtype}")
    stock_exact = np.array_equal(arrays["X_stock"], arrays["X_full"][:, :, :47])
    market_exact = np.array_equal(arrays["X_market"], arrays["X_full"][:, :, 47:])
    if not stock_exact or not market_exact: raise RuntimeError(f"{scope}: modality slice mismatch.")
    if not np.isin(arrays["y_direction"], [0, 1]).all(): raise RuntimeError(f"{scope}: non-binary direction label.")
    if not np.isin(arrays["regime_id"], [0, 1, 2]).all(): raise RuntimeError(f"{scope}: invalid regime ID.")
    if arrays["stock_id"].min() < 0 or arrays["stock_id"].max() >= len(cfg.STOCK_TICKERS): raise RuntimeError(f"{scope}: invalid stock ID.")
    for name in ("X_full", "X_stock", "X_market", "y_aux_raw", "y_aux_scaled"):
        if not np.isfinite(arrays[name]).all(): raise RuntimeError(f"{scope}: non-finite {name}.")
    return {"scope": scope, "fold": np.nan if fold is None else fold, "partition": partition, "rows": n, "x_full_shape": str(arrays["X_full"].shape), "x_stock_shape": str(arrays["X_stock"].shape), "x_market_shape": str(arrays["X_market"].shape), "y_direction_shape": str(arrays["y_direction"].shape), "y_aux_shape": str(arrays["y_aux_scaled"].shape), "stock_slice_exact": stock_exact, "market_slice_exact": market_exact, "direction_binary": True, "regime_ids_valid": True, "stock_ids_valid": True, "finite_arrays": True, "passed": True}


def save_npy(array, path, scope, fold, partition, name):
    np.save(path, array, allow_pickle=False)
    loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    if loaded.shape != array.shape or loaded.dtype != array.dtype or not np.array_equal(np.asarray(loaded), array, equal_nan=True):
        raise RuntimeError(f"{path.name}: NumPy roundtrip failed.")
    del loaded
    return {"scope": scope, "fold": fold, "partition": partition, "artifact_kind": "npy", "array_name": name, "file_name": path.name, "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def save_metadata(endpoints, path, scope, fold, partition):
    cols = [
        "array_row_index", "fold", "partition", "ticker", "stock_id", "continuity_segment_id", "reference_date", "reference_market_session_id",
        "sequence_start_date", "sequence_end_date", "lookback_crosses_partition_start", "historical_rows_before_partition", "regime_id", "regime_name",
        "regime_threshold_fit_scope", "target_date_1d", "target_date_3d", "target_date_5d", *DIRECTION_COLUMNS, *AUX_COLUMNS,
    ]
    missing = [c for c in cols if c not in endpoints.columns]
    if missing: raise RuntimeError(f"{scope}: metadata columns missing: {missing}")
    md = endpoints[cols].copy(); md.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    obs = pd.read_parquet(path, engine="pyarrow")
    if len(obs) != len(md) or not np.array_equal(obs["array_row_index"].to_numpy(np.int64), md["array_row_index"].to_numpy(np.int64)):
        raise RuntimeError(f"{scope}: metadata roundtrip failed.")
    return {"scope": scope, "fold": fold, "partition": partition, "artifact_kind": "metadata_parquet", "array_name": None, "file_name": path.name, "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def save_scaler(scaler, path, scope, fold, partition):
    joblib.dump(scaler, path, compress=3)
    loaded = joblib.load(path)
    if loaded.n_features_in_ != len(AUX_COLUMNS) or not np.array_equal(np.asarray(loaded.data_min_), np.asarray(scaler.data_min_)):
        raise RuntimeError(f"{scope}: persisted auxiliary scaler mismatch.")
    del loaded
    return {"scope": scope, "fold": fold, "partition": partition, "artifact_kind": "auxiliary_scaler_joblib", "array_name": None, "file_name": path.name, "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def materialize_partition(endpoints, train_panel, validation_panel, partition, features, aux_scaled, outdir, prefix, scope, fold):
    history = build_history(train_panel, validation_panel, partition, features)
    X = materialize_x(endpoints, history)
    arrays = build_arrays(endpoints, X, aux_scaled)
    integrity = audit_arrays(arrays, len(endpoints), scope, fold, partition)
    records = []
    for name, array in arrays.items():
        records.append(save_npy(array, outdir / f"{prefix}_{name}.npy", scope, fold, partition, name))
    records.append(save_metadata(endpoints, outdir / f"{prefix}_metadata.parquet", scope, fold, partition))
    total = sum(r["size_bytes"] for r in records)
    summary = {
        "scope": scope, "fold": np.nan if fold is None else fold, "partition": partition, "sequences": int(len(endpoints)), "lookback": 60,
        "features": 63, "stock_features": 47, "market_features": 16, "direction_targets": 3, "auxiliary_targets": 10,
        "cross_boundary_lookback_endpoints": int(endpoints["lookback_crosses_partition_start"].sum()),
        "auxiliary_scaled_values_outside_0_1": int((arrays["y_aux_scaled"] < 0).sum() + (arrays["y_aux_scaled"] > 1).sum()),
        "saved_bytes": int(total), "saved_gib": float(total / (1024 ** 3)),
    }
    del history, X, arrays; gc.collect()
    return summary, integrity, records


def source_hashes():
    sources = [("current_step00", Path(cfg.__file__).resolve()), ("current_step14", SCRIPT_FILE.resolve()), ("step10_development", STEP10_DEVELOPMENT.resolve()), ("step10_test_seal_metadata", STEP10_TEST_SEAL.resolve()), ("step12_final", STEP12_FINAL.resolve()), ("step13_final_index", STEP13_FINAL.resolve())]
    for fold in FOLDS:
        for partition in PARTITIONS:
            sources += [(f"step12_fold_{fold}_{partition}", step12_path(fold, partition).resolve()), (f"step13_fold_{fold}_{partition}", step13_path(fold, partition).resolve())]
    for step, (mp, sp) in UPSTREAM.items():
        sources += [(f"step{step:02d}_manifest", mp.resolve()), (f"step{step:02d}_manifest_sha", sp.resolve())]
    rows = []
    for role, path in sources:
        if path.name == FORBIDDEN_TEST_FILENAME: raise RuntimeError("Forbidden test parquet entered Step-14 source register.")
        if not path.exists(): raise RuntimeError(f"Missing Step-14 source: {path}")
        rows.append({"role": role, "file_name": path.name, "absolute_path": str(path), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return pd.DataFrame(rows)


def verify_existing():
    print("=" * 136); print("STEP 14 — VERIFY EXISTING MODEL-ARRAY FREEZE"); print("=" * 136)
    step00 = verify_step00(); upstream = verify_upstream(step00)
    manifest_sha = verify_sidecar(MANIFEST_JSON, MANIFEST_SHA, "Step 14")
    m = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    if m.get("step", {}).get("status") != "PASS": return 20
    if m.get("configuration_fingerprint_sha256") != step00["fingerprint"]: return 21
    for step in range(1, 14):
        if m.get(f"step{step:02d}_manifest_sha256") != upstream["shas"][step]: return 22
    if m.get("feature_interface_sha256") != upstream["feature_interface_sha"]: return 23
    if m.get("split_design_sha256") != upstream["split_design_sha"]: return 24
    failures = []
    for name, rec in m.get("artifacts", {}).items():
        p = Path(rec["path"])
        if not p.exists(): failures.append(f"missing: {name}")
        elif sha256_file(p) != rec["sha256"]: failures.append(f"hash mismatch: {name}")
    if failures:
        for x in failures: print(f"[FAIL] {x}")
        print("=" * 136); print("STEP 14 VERIFICATION STATUS: FAIL"); print("=" * 136); return 25
    if sha256_file(Path(step00["lock_path"])) != step00["lock_sha"]: return 26
    print(f"[PASS] Configuration fingerprint : {step00['fingerprint']}")
    print(f"[PASS] Step-13 manifest          : {upstream['shas'][13]}")
    print(f"[PASS] 63-feature interface      : {upstream['feature_interface_sha']}")
    print(f"[PASS] Split-design SHA-256      : {upstream['split_design_sha']}")
    print(f"[PASS] Step-14 manifest          : {manifest_sha}")
    print(f"[PASS] Step-10 test seal         : {upstream['test_seal_status']}")
    print(f"[PASS] Global final-test state   : {step00['state']}")
    print("=" * 136); print("STEP 14 VERIFICATION STATUS: PASS"); print("=" * 136)
    return 0


def run():
    print("=" * 136); print(STEP_NAME); print("=" * 136)
    stage = "verify"; staging = None
    try:
        print("[1/10] Verifying Step 00 and immutable Step 01 -> Step 13 chain...")
        step00 = verify_step00(); upstream = verify_upstream(step00)
        print(f"       Configuration fingerprint: {step00['fingerprint']}")
        print(f"       Step-13 manifest          : {upstream['shas'][13]}")
        print(f"       Development SHA-256       : {upstream['development_sha']}")
        print(f"       63-feature interface      : {upstream['feature_interface_sha']}")
        print(f"       Split-design SHA-256      : {upstream['split_design_sha']}")
        print(f"       Step-10 test seal         : {upstream['test_seal_status']}")
        print(f"       Global final-test state   : {step00['state']}")
        print("       Final-test parquet access : PROHIBITED")

        stage = "collision"; print("[2/10] Checking immutable Step-14 output target..."); assert_fresh(); cfg.create_directories()
        stage = "load"; print("[3/10] Loading frozen development targets and 47+16 feature interface...")
        development = load_development(); features = upstream["all_features"]
        print("       X_full                  : (n, 60, 63) float32")
        print("       X_stock / X_market      : (n,60,47) / (n,60,16)")
        print(f"       Auxiliary targets       : {len(AUX_COLUMNS)} | clip=False")

        stage = "staging"; staging = create_stage()
        summaries = []; integrity_rows = []; reconciliation_rows = []; scaler_audits = []; val_range_rows = []; inverse_rows = []; staged_records = []

        stage = "folds"; print("[4/10] Materializing five rolling-origin train/validation array pairs...")
        for fold in FOLDS:
            train_panel = load_panel(step12_path(fold, "train"), features, f"fold_{fold}_train_step12")
            val_panel = load_panel(step12_path(fold, "validation"), features, f"fold_{fold}_validation_step12")
            train_index = load_index(step13_path(fold, "train"), f"fold_{fold}_train_step13")
            val_index = load_index(step13_path(fold, "validation"), f"fold_{fold}_validation_step13")
            train_ep, rec1 = attach_targets(train_index, development, train_panel, f"fold_{fold}_train")
            val_ep, rec2 = attach_targets(val_index, development, val_panel, f"fold_{fold}_validation")
            reconciliation_rows += [rec1, rec2]

            scaler, train_aux = fit_aux(train_ep, f"fold_{fold}_train")
            val_aux = scaler.transform(val_ep[AUX_COLUMNS].astype(np.float64))
            if not np.isfinite(val_aux).all(): raise RuntimeError(f"Fold {fold}: non-finite validation auxiliary values.")
            scaler_audits += scaler_rows(scaler, "fold_training", fold, len(train_ep))
            val_range_rows += validation_range_rows(fold, val_aux)
            inverse_rows += [aux_inverse(scaler, train_ep, train_aux, f"fold_{fold}_train", fold, "train"), aux_inverse(scaler, val_ep, val_aux, f"fold_{fold}_validation", fold, "validation")]

            fdir = staging / f"fold_{fold}"
            staged_records.append(save_scaler(scaler, fdir / f"14_fold_{fold}_auxiliary_target_minmax_scaler.joblib", f"fold_{fold}_aux_scaler", fold, "train_fit"))
            s, a, r = materialize_partition(train_ep, train_panel, None, "train", features, train_aux, fdir, f"14_fold_{fold}_train", f"fold_{fold}_train", fold)
            summaries.append(s); integrity_rows.append(a); staged_records += r
            s, a, r = materialize_partition(val_ep, train_panel, val_panel, "validation", features, val_aux, fdir, f"14_fold_{fold}_validation", f"fold_{fold}_validation", fold)
            summaries.append(s); integrity_rows.append(a); staged_records += r
            outside = int((val_aux < 0).sum() + (val_aux > 1).sum())
            print(f"       Fold {fold}: train n={len(train_ep):>6,} | val n={len(val_ep):>5,} | val cross={int(val_ep['lookback_crosses_partition_start'].sum()):>5,} | aux outside[0,1]={outside:>6,}")
            del train_panel, val_panel, train_index, val_index, train_ep, val_ep, scaler, train_aux, val_aux; gc.collect()

        stage = "final_dev"; print("[5/10] Materializing final-development refit arrays...")
        final_panel = load_panel(STEP12_FINAL, features, "final_development_step12")
        final_index = load_index(STEP13_FINAL, "final_development_step13")
        final_ep, final_rec = attach_targets(final_index, development, final_panel, "final_development")
        reconciliation_rows.append(final_rec)
        final_scaler, final_aux = fit_aux(final_ep, "final_development")
        scaler_audits += scaler_rows(final_scaler, "final_development", None, len(final_ep))
        inverse_rows.append(aux_inverse(final_scaler, final_ep, final_aux, "final_development", None, "final_development"))
        fdir = staging / "final_development"
        staged_records.append(save_scaler(final_scaler, fdir / "14_final_development_auxiliary_target_minmax_scaler.joblib", "final_development_aux_scaler", None, "final_development_fit"))
        s, a, r = materialize_partition(final_ep, final_panel, None, "final_development", features, final_aux, fdir, "14_final_development", "final_development", None)
        summaries.append(s); integrity_rows.append(a); staged_records += r
        print(f"       Final-development n    : {len(final_ep):,}")
        print(f"       Final-development X    : ({len(final_ep):,}, 60, 63)")
        print("       Final-test arrays      : NOT CREATED")
        del final_panel, final_index, final_ep, final_scaler, final_aux; gc.collect()

        stage = "audits"; print("[6/10] Building array, auxiliary-scaling and reconciliation audits...")
        summary_df = pd.DataFrame(summaries); integrity_df = pd.DataFrame(integrity_rows); rec_df = pd.DataFrame(reconciliation_rows); scaler_df = pd.DataFrame(scaler_audits); val_df = pd.DataFrame(val_range_rows); inv_df = pd.DataFrame(inverse_rows)
        if len(summary_df) != 11 or len(integrity_df) != 11 or len(rec_df) != 11: raise RuntimeError("Expected exactly 11 materialized/reconciled partitions.")
        if not integrity_df["passed"].all() or not inv_df["passed"].all(): raise RuntimeError("Array or auxiliary inverse audit failed.")
        for c in ("target_dates_exact_step10_to_step13", "regime_ids_exact_step12_to_step13", "continuity_ids_exact_step12_to_step13", "market_session_ids_exact_step12_to_step13", "direction_labels_binary", "auxiliary_targets_finite"):
            if not rec_df[c].all(): raise RuntimeError(f"Endpoint reconciliation failed: {c}")
        print(f"       Materialized partitions  : {len(summary_df)}")
        print(f"       All array audits pass    : {bool(integrity_df['passed'].all())}")
        print(f"       Endpoint reconciliations : {len(rec_df)}")
        print(f"       Aux inverse checks       : {len(inv_df)}")

        stage = "promotion"; print("[7/10] Promoting model-ready arrays and writing audits...")
        if OUT_DIR.exists(): raise RuntimeError("Canonical Step-14 output directory appeared unexpectedly.")
        os.replace(staging, OUT_DIR); staging = None
        promoted = []
        for rec in staged_records:
            p = FINAL_DIR / rec["file_name"] if rec["fold"] is None else OUT_DIR / f"fold_{rec['fold']}" / rec["file_name"]
            if not p.exists() or sha256_file(p) != rec["sha256"]: raise RuntimeError(f"Promoted artifact mismatch: {p}")
            promoted.append({**rec, "file_path": str(p), "relative_path": rel(p)})

        write_csv(ARRAY_SUMMARY_CSV, summary_df); write_csv(AUX_SCALER_AUDIT_CSV, scaler_df); write_csv(AUX_VALIDATION_RANGE_CSV, val_df); write_csv(AUX_INVERSE_AUDIT_CSV, inv_df); write_csv(ARRAY_INTEGRITY_CSV, integrity_df); write_csv(ENDPOINT_RECONCILIATION_CSV, rec_df); write_csv(SOURCE_HASH_CSV, source_hashes())
        for p in (ARRAY_SUMMARY_CSV, AUX_SCALER_AUDIT_CSV, AUX_VALIDATION_RANGE_CSV, AUX_INVERSE_AUDIT_CSV, ARRAY_INTEGRITY_CSV, ENDPOINT_RECONCILIATION_CSV, SOURCE_HASH_CSV):
            promoted.append({"scope": p.stem, "fold": None, "partition": None, "artifact_kind": "audit", "array_name": None, "file_name": p.name, "size_bytes": int(p.stat().st_size), "sha256": sha256_file(p), "file_path": str(p), "relative_path": rel(p)})

        stage = "freeze"; print("[8/10] Freezing Step-14 summary, source snapshot and manifest...")
        if sha256_file(Path(step00["lock_path"])) != step00["lock_sha"]: raise RuntimeError("Global final-test lock changed during Step 14.")
        total_bytes = int(summary_df["saved_bytes"].sum()); val_outside = int(val_df["outside_0_1_count"].sum())
        summary = {
            "schema_version": SCHEMA_VERSION, "step": {"step_number": 14, "step_id": "14", "step_name": STEP_NAME, "status": "PASS"}, "generated_at_utc": utc_now(),
            "configuration_fingerprint_sha256": step00["fingerprint"], "step13_manifest_sha256": upstream["shas"][13], "feature_interface_sha256": upstream["feature_interface_sha"], "split_design_sha256": upstream["split_design_sha"],
            "array_interface": {"lookback_sessions": 60, "full_features": 63, "stock_features": 47, "market_features": 16, "direction_targets": 3, "auxiliary_targets": 10, "X_dtype": "float32", "direction_dtype": "int64", "auxiliary_dtype": "float32", "categorical_id_dtype": "int64"},
            "auxiliary_scaling_policy": {"scaler": "MinMaxScaler", "feature_range": [0.0, 1.0], "clip": False, "fold_fit_scope": "fold-training common sequence endpoints only", "validation_used_in_fit": False, "final_development_fit_scope": "final-development common sequence endpoints only", "final_test_used_in_fit": False, "directional_targets_scaled": False, "validation_values_outside_0_1_total": val_outside},
            "storage": {"total_saved_bytes": total_bytes, "total_saved_gib": total_bytes / (1024 ** 3)},
            "pipeline_state": {"rolling_fold_common_multitask_arrays_materialized": True, "rolling_fold_auxiliary_scalers_fitted": True, "final_development_refit_arrays_created": True, "final_development_auxiliary_scaler_fitted": True, "feature_selection_applied": False, "models_fitted": False, "probability_threshold_selection_performed": False, "final_test_arrays_created": False},
            "test_protection": {"step10_seal_status": upstream["test_seal_status"], "sealed_test_sha256_from_metadata_only": upstream["sealed_test_sha_metadata_only"], "sealed_test_parquet_opened_or_hashed_by_step14": False, "global_final_test_state": step00["state"], "global_lock_unchanged": True},
        }
        atomic_json(SUMMARY_JSON, summary); shutil.copy2(SCRIPT_FILE, FROZEN_SCRIPT)
        atomic_text(SUMMARY_TXT, "\n".join([STEP_NAME, "="*100, "Status: PASS", f"Step-13 manifest: {upstream['shas'][13]}", f"63-feature interface: {upstream['feature_interface_sha']}", f"Split-design SHA-256: {upstream['split_design_sha']}", "Fold common-multitask arrays: True", "Final-development arrays: True", "Auxiliary clipping: False", "Final-test parquet accessed: False", f"Global final-test state: {step00['state']}"]) + "\n")
        for p in (SUMMARY_JSON, FROZEN_SCRIPT, SUMMARY_TXT):
            promoted.append({"scope": p.stem, "fold": None, "partition": None, "artifact_kind": "freeze_metadata", "array_name": None, "file_name": p.name, "size_bytes": int(p.stat().st_size), "sha256": sha256_file(p), "file_path": str(p), "relative_path": rel(p)})
        write_csv(HASH_REGISTER_CSV, pd.DataFrame(promoted))
        artifact_paths = [Path(r["file_path"]) for r in promoted] + [HASH_REGISTER_CSV]
        manifest = {
            "schema_version": SCHEMA_VERSION, "step": {"step_number": 14, "step_id": "14", "step_name": STEP_NAME, "status": "PASS"}, "frozen_at_utc": utc_now(), "experiment_tag": cfg.EXPERIMENT_TAG,
            "configuration_fingerprint_sha256": step00["fingerprint"], "step00_manifest_sha256": step00["step00_manifest_sha"],
            **{f"step{s:02d}_manifest_sha256": upstream["shas"][s] for s in range(1, 14)},
            "feature_interface_sha256": upstream["feature_interface_sha"], "ordered_63_features": features, "stock_features": upstream["stock_features"], "market_features": upstream["market_features"], "split_design_sha256": upstream["split_design_sha"], "stock_id_map": upstream["stock_id_map"],
            "direction_target_order": DIRECTION_COLUMNS, "auxiliary_target_order": AUX_COLUMNS,
            "source_integrity": {"step10_development_panel_sha256": upstream["development_sha"], "step12_fold_panel_hashes": upstream["step12_hashes"], "step12_final_development_sha256": upstream["final12_sha"], "step13_fold_index_hashes": upstream["step13_hashes"], "step13_final_development_index_sha256": upstream["final13_sha"]},
            "array_interface": {"lookback_sessions": 60, "continuous_feature_count": 63, "stock_feature_count": 47, "market_feature_count": 16, "X_dtype": "float32", "direction_dtype": "int64", "auxiliary_dtype": "float32", "categorical_id_dtype": "int64", "common_multitask_only": True},
            "auxiliary_scaling_policy": {"scaler": "MinMaxScaler", "feature_range": [0.0, 1.0], "clip": False, "fold_fit_scope": "fold-training common sequence endpoints only", "validation_used_in_fit": False, "final_development_fit_scope": "final-development common sequence endpoints only", "final_test_used_in_fit": False, "directional_targets_scaled": False},
            "sequence_policy": {"indices_reused_from_step13": True, "sequence_membership_recomputed": False, "validation_causal_training_history_preserved": True, "same_ticker_required": True, "same_continuity_segment_required": True},
            "test_protection": {"step10_seal_status": upstream["test_seal_status"], "global_final_test_state": step00["state"], "sealed_test_parquet_opened_or_hashed_by_step14": False, "final_test_arrays_created": False, "test_targets_scaled": False, "test_performance_computed": False},
            "pipeline_state": {"rolling_fold_sequence_arrays_materialized": True, "rolling_fold_auxiliary_scalers_fitted": True, "final_development_refit_arrays_created": True, "final_development_auxiliary_scaler_fitted": True, "feature_selection_applied": False, "models_fitted": False, "probability_threshold_selection_performed": False},
            "artifacts": {p.name: artifact(p) for p in artifact_paths},
        }
        atomic_json(MANIFEST_JSON, manifest); manifest_sha = sha256_file(MANIFEST_JSON); atomic_text(MANIFEST_SHA, f"{manifest_sha}  {MANIFEST_JSON.name}\n")

        stage = "validate"; print("[9/10] Validating frozen Step-14 artifact set...")
        verify_sidecar(MANIFEST_JSON, MANIFEST_SHA, "Step 14")
        for rec in manifest["artifacts"].values():
            p = Path(rec["path"])
            if not p.exists() or sha256_file(p) != rec["sha256"]: raise RuntimeError(f"Frozen Step-14 artifact mismatch: {p}")
        if sha256_file(Path(step00["lock_path"])) != step00["lock_sha"]: raise RuntimeError("Global final-test lock changed during validation.")

        stage = "report"; print("[10/10] Final Step-14 report...")
        print(); print("-" * 136); print("ARRAY INTERFACE"); print("-" * 136)
        print("X_full                   : (n, 60, 63) float32")
        print("X_stock                  : (n, 60, 47) float32")
        print("X_market                 : (n, 60, 16) float32")
        print("y_direction              : (n, 3) int64")
        print("y_aux_raw / y_aux_scaled : (n, 10) float32")
        print("stock_id / regime_id     : (n,) int64")
        print("-" * 136); print("ROLLING-ORIGIN ARRAYS"); print("-" * 136)
        for row in summary_df.loc[summary_df["fold"].notna()].sort_values(["fold", "partition"]).itertuples(index=False):
            print(f"Fold {int(row.fold)} {row.partition:<10} | n={row.sequences:>6,} | X=({row.sequences:,},60,63) | cross={row.cross_boundary_lookback_endpoints:>5,} | aux outside[0,1]={row.auxiliary_scaled_values_outside_0_1:>6,} | saved={row.saved_gib:.3f} GiB")
        final_row = summary_df.loc[summary_df["partition"] == "final_development"].iloc[0]
        print("-" * 136); print("FINAL-DEVELOPMENT REFIT ARRAYS"); print("-" * 136)
        print(f"Sequences                  : {int(final_row['sequences']):,}")
        print(f"X_full                     : ({int(final_row['sequences']):,},60,63)")
        print("Auxiliary scaler           : FINAL-DEVELOPMENT ONLY")
        print("Final-test arrays           : NOT CREATED")
        print("-" * 136); print("INTEGRITY / STORAGE"); print("-" * 136)
        print(f"Materialized partitions    : {len(integrity_df)}")
        print(f"All array audits pass      : {bool(integrity_df['passed'].all())}")
        print(f"Stock modality exact       : {bool(integrity_df['stock_slice_exact'].all())}")
        print(f"Market modality exact      : {bool(integrity_df['market_slice_exact'].all())}")
        print(f"Endpoint reconciliations   : {len(rec_df)}")
        print(f"Aux inverse checks pass    : {bool(inv_df['passed'].all())}")
        print(f"Validation aux outside[0,1]: {val_outside:,} values retained")
        print(f"Total persisted size       : {total_bytes / (1024 ** 3):.3f} GiB")
        print("-" * 136); print("PIPELINE STATE"); print("-" * 136)
        print("Fold sequence arrays       : COMPLETE")
        print("Fold auxiliary scaling     : COMPLETE")
        print("Final-development arrays   : COMPLETE")
        print("Feature selection          : False")
        print("Models fitted              : False")
        print("Final-test parquet accessed: False")
        print(f"Global final-test state    : {step00['state']}")
        print("-" * 136); print("OUTPUT FILES"); print("-" * 136)
        print(f"Array root             : {OUT_DIR}")
        print(f"Array summary          : {ARRAY_SUMMARY_CSV}")
        print(f"Aux scaler audit       : {AUX_SCALER_AUDIT_CSV}")
        print(f"Aux validation audit   : {AUX_VALIDATION_RANGE_CSV}")
        print(f"Aux inverse audit      : {AUX_INVERSE_AUDIT_CSV}")
        print(f"Array integrity        : {ARRAY_INTEGRITY_CSV}")
        print(f"Endpoint reconciliation: {ENDPOINT_RECONCILIATION_CSV}")
        print(f"Manifest               : {MANIFEST_JSON}")
        print(f"Manifest SHA-256       : {manifest_sha}")
        print("=" * 136)
        print("PASS: Five train/validation common-multitask array pairs were materialized from frozen Step-13 indices.")
        print("PASS: 47-stock-feature and 16-market-feature tensors are exact slices of X_full.")
        print("PASS: Directional labels remain unscaled binary integer targets.")
        print("PASS: Auxiliary targets use fold-training-only scaling; validation is transformed without clipping/refitting.")
        print("PASS: Final-development arrays and auxiliary scaler were frozen from development data only.")
        print("PASS: The sealed final-test parquet was not opened, hashed, transformed, materialized, or inspected.")
        print("PASS: No feature selection, probability-threshold tuning, or forecasting model was performed.")
        print("=" * 136)
        print("Do not run the old Step-14 script. Do not run Step 15 yet. Paste the complete terminal output for audit.")
        return 0

    except Exception as exc:
        if staging is not None and staging.exists(): shutil.rmtree(staging, ignore_errors=True)
        archived = archive_partial(); FAILED_DIR.mkdir(parents=True, exist_ok=True)
        failure = FAILED_DIR / f"14_model_arrays_FAILED_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        atomic_json(failure, {"step": STEP_NAME, "status": "FAIL", "failed_at_utc": utc_now(), "stage": stage, "error_type": type(exc).__name__, "error_message": str(exc), "python_executable": sys.executable, "archived_partial_artifacts": None if archived is None else str(archived)})
        print(); print("=" * 136); print("STEP 14 FAILED — MODEL ARRAYS NOT FROZEN"); print("=" * 136)
        print(f"Stage      : {stage}"); print(f"Error type : {type(exc).__name__}"); print(f"Error      : {exc}"); print(f"Diagnostic : {failure}")
        if archived is not None: print(f"Archived partial artifacts: {archived}")
        print("=" * 136); print("Do not proceed to Step 15.")
        return 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verify-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    return verify_existing() if args.verify_only else run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStep 14 interrupted by user."); sys.exit(130)
    except Exception as exc:
        print(); print("=" * 136); print("STEP 14 UNHANDLED FAILURE"); print("=" * 136)
        print(f"{type(exc).__name__}: {exc}"); traceback.print_exc(); print("=" * 136); sys.exit(1)
