#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 36 — ONE-TIME FINAL-TEST MATERIALIZATION, INFERENCE AND SCORING

Run --prepare-only first. It verifies/fixes nothing on the held-out test and
freezes this Step-36 source before any test access.

Then --execute performs the one-time held-out evaluation under the frozen
Step-35 contract. Raw-test access is consumed once, materialized arrays are
frozen, and all later inference is resumable from those arrays without
reopening the raw test parquet.

No architecture promotion, best-seed selection, threshold retuning, or
post-test hyperparameter changes are allowed here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent
if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError(
        "Place this script inside <repository>/q1_revision/ before running it."
    )
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg

STEP_NAME = "STEP 36 — ONE-TIME FINAL-TEST MATERIALIZATION, INFERENCE AND SCORING"
SCHEMA_VERSION = "1.0"

EXPECTED_STEP14_MANIFEST_SHA = "b19b034bd31bb5d426e78c58f808d9e4c9cba16281e666984aae2c8bf6a150a0"
EXPECTED_STEP32_MANIFEST_SHA = "474b77ac670bbb7f61870372cdb98a6d7ac3ecaa4db8864ceb096fe9b26c39f0"
EXPECTED_STEP34_MANIFEST_SHA = "2224aa145e4bdbce2563586cd1a28742facd29638985f29fb4c6a34fa9bcaf48"
EXPECTED_STEP34_GLOBAL_FP = "1ede9c52aaadf66ea4e82f777c4c03fcccece419bff0867c2a434e3fb13828c0"
EXPECTED_STEP35_MANIFEST_SHA = "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7"
EXPECTED_STEP35_CONTRACT_FP = "44949863de7e82c3c303ca55beff8c59ae3206ea7eb542a9aab115ff88632bfc"

EXPECTED_OUTER_TEST_ROWS = 6680
EXPECTED_OUTER_TEST_DATES = 334
EXPECTED_COMMON_ROWS = 6480
EXPECTED_COMMON_DATES = 324
EXPECTED_STOCKS = 20
EXPECTED_LOOKBACK = 60
EXPECTED_FEATURES = 63
EXPECTED_STOCK_FEATURES = 47
EXPECTED_MARKET_FEATURES = 16
EXPECTED_ARCHITECTURES = 8
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_FITS = 40
EXPECTED_THRESHOLDS = 24
HORIZONS = (1, 3, 5)
DIRECTION_COLUMNS = ("direction_1d", "direction_3d", "direction_5d")

MODEL_ORDER = (
    "DLINEAR", "NLINEAR", "PATCHTST", "TIMESNET",
    "ITRANSFORMER", "FREDFORMER", "FRNET", "MLF",
)
EXPECTED_CONFIG = {
    "DLINEAR": "DLINEAR_B",
    "NLINEAR": "NLINEAR_B",
    "PATCHTST": "PATCHTST_B",
    "TIMESNET": "TIMESNET_B",
    "ITRANSFORMER": "ITRANSFORMER_A",
    "FREDFORMER": "FREDFORMER_B",
    "FRNET": "FRNET_A",
    "MLF": "MLF_A",
}
EXPECTED_PHYSICAL_MB = {
    "DLINEAR": 128,
    "NLINEAR": 128,
    "PATCHTST": 64,
    "TIMESNET": 64,
    "ITRANSFORMER": 128,
    "FREDFORMER": 16,
    "FRNET": 64,
    "MLF": 32,
}

EXPERIMENT_ROOT = cfg.MANIFEST_DIR.parent
STEP10_TEST_SEAL = cfg.MANIFEST_DIR / "10_final_test_seal.json"
STEP14_MANIFEST = cfg.MANIFEST_DIR / "14_model_array_manifest.json"
STEP32_MANIFEST = cfg.MANIFEST_DIR / "32_development_threshold_final_freeze_manifest.json"
STEP34_MANIFEST = cfg.MANIFEST_DIR / "34_final_development_refit_final_freeze_manifest.json"
STEP35_MANIFEST = cfg.MANIFEST_DIR / "35_final_test_evaluation_contract_freeze_manifest.json"
STEP35_THRESHOLDS = cfg.AUDIT_DIR / "35_final_test_threshold_contract.csv"
STEP34_REGISTER = cfg.AUDIT_DIR / "34_final_development_refit_register.csv"
STEP35_INFERENCE_PLAN = cfg.AUDIT_DIR / "35_final_test_inference_plan.csv"

FINAL_DEV_HISTORY = (
    cfg.FEATURE_DATA_DIR / "step12_volatility_regimes" /
    "final_development" / "12_final_development_scaled_regimes.parquet"
)
FINAL_DEV_REGIME_THRESHOLDS = (
    cfg.FEATURE_DATA_DIR / "step12_volatility_regimes" /
    "final_development" / "12_final_development_regime_thresholds.csv"
)

PREACCESS_FREEZE = cfg.MANIFEST_DIR / "36_final_test_execution_preaccess_freeze_manifest.json"
PREACCESS_FREEZE_SHA = cfg.MANIFEST_DIR / "36_final_test_execution_preaccess_freeze_manifest.sha256"
FROZEN_SOURCE = cfg.MANIFEST_DIR / "step36_one_time_final_test_materialization_inference_scoring_v1_frozen.py"
ACCESS_MARKER = cfg.MANIFEST_DIR / "36_final_test_access_consumption_state.json"

ARRAY_DIR = cfg.FEATURE_DATA_DIR / "step36_final_test_arrays"
ARRAY_MANIFEST = cfg.MANIFEST_DIR / "36_final_test_materialization_manifest.json"
PRED_ROOT = EXPERIMENT_ROOT / "final_test_evaluation" / "step36_predictions"
SEED_PRED_ROOT = PRED_ROOT / "seed_predictions"
ENSEMBLE_ROOT = PRED_ROOT / "ensemble_predictions"
METRICS_CSV = cfg.AUDIT_DIR / "36_final_test_metrics.csv"
CLASS_BALANCE_CSV = cfg.AUDIT_DIR / "36_final_test_class_balance.csv"
INFERENCE_REGISTER_CSV = cfg.AUDIT_DIR / "36_final_test_inference_register.csv"
ENSEMBLE_PREDICTIONS_PARQUET = EXPERIMENT_ROOT / "final_test_evaluation" / "36_final_test_ensemble_predictions.parquet"
ARTIFACT_REGISTER_CSV = cfg.AUDIT_DIR / "36_final_test_artifact_register.csv"
FINAL_MANIFEST = cfg.MANIFEST_DIR / "36_one_time_final_test_materialization_inference_scoring_manifest.json"
FINAL_MANIFEST_SHA = cfg.MANIFEST_DIR / "36_one_time_final_test_materialization_inference_scoring_manifest.sha256"
FAILURE_JSON = cfg.AUDIT_DIR / "36_final_test_execution_failure.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON missing:\n{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str,
        ).encode("utf-8")
    ).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def artifact_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "relative_path": str(path.resolve().relative_to(EXPERIMENT_ROOT.resolve())),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def require_equal(observed, expected, label: str) -> None:
    if observed != expected:
        raise RuntimeError(f"{label} mismatch.\nExpected: {expected}\nObserved: {observed}")


def require_file_sha(path: Path, expected_sha: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing:\n{path}")
    observed = sha256_file(path)
    if observed != expected_sha:
        raise RuntimeError(f"{label} SHA mismatch.\nExpected: {expected_sha}\nObserved: {observed}")


def contains_value(obj: Any, target: str) -> bool:
    if isinstance(obj, dict):
        return any(contains_value(k, target) or contains_value(v, target) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return any(contains_value(v, target) for v in obj)
    return str(obj) == str(target)


def find_unique_named_file(root: Path, filename: str) -> Path:
    matches = sorted(p for p in root.rglob(filename) if p.is_file())
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {filename}; found {len(matches)}:\n" +
            "\n".join(str(x) for x in matches)
        )
    return matches[0]


def write_csv_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def verify_upstream_without_test_access() -> dict:
    require_file_sha(STEP14_MANIFEST, EXPECTED_STEP14_MANIFEST_SHA, "Step-14 manifest")
    require_file_sha(STEP32_MANIFEST, EXPECTED_STEP32_MANIFEST_SHA, "Step-32 manifest")
    require_file_sha(STEP34_MANIFEST, EXPECTED_STEP34_MANIFEST_SHA, "Step-34 manifest")
    require_file_sha(STEP35_MANIFEST, EXPECTED_STEP35_MANIFEST_SHA, "Step-35 manifest")

    m14 = load_json(STEP14_MANIFEST)
    m34 = load_json(STEP34_MANIFEST)
    m35 = load_json(STEP35_MANIFEST)
    seal = load_json(STEP10_TEST_SEAL)

    require_equal(m34.get("global_step34_fingerprint"), EXPECTED_STEP34_GLOBAL_FP, "Step-34 global fingerprint")
    if not contains_value(m35, EXPECTED_STEP35_CONTRACT_FP):
        raise RuntimeError("Step-35 manifest does not contain the frozen contract fingerprint.")

    require_equal(seal.get("status"), "SEALED_NOT_OPENED", "Step-10 final-test seal status")
    require_equal(seal.get("global_final_test_lock_state"), "LOCKED_NOT_OPENED", "Step-10 global final-test state")
    require_equal(int(seal.get("reference_rows")), EXPECTED_OUTER_TEST_ROWS, "Step-10 outer test rows")
    require_equal(int(seal.get("unique_reference_dates")), EXPECTED_OUTER_TEST_DATES, "Step-10 outer test dates")
    require_equal(int(seal.get("common_multitask_structural_rows")), EXPECTED_COMMON_ROWS, "Step-10 common structural rows")
    require_equal(int(seal.get("stock_count")), EXPECTED_STOCKS, "Step-10 stock count")

    test_path = Path(seal["file_path"])
    if test_path.name != "10_final_test_locked_panel.parquet":
        raise RuntimeError("Unexpected sealed final-test filename.")
    if not test_path.exists():
        raise FileNotFoundError(f"Sealed final-test parquet missing:\n{test_path}")

    features = list(m14.get("ordered_63_features", []))
    if len(features) != EXPECTED_FEATURES or len(set(features)) != EXPECTED_FEATURES:
        raise RuntimeError("Step-14 ordered 63-feature interface is invalid.")

    stock_id_map = m14.get("stock_id_map")
    if not isinstance(stock_id_map, dict) or len(stock_id_map) != EXPECTED_STOCKS:
        raise RuntimeError("Step-14 stock_id_map is invalid.")

    scaler_path = find_unique_named_file(cfg.FEATURE_DATA_DIR, "11_final_development_minmax_scaler.joblib")
    scaler = joblib.load(scaler_path)
    if int(scaler.n_features_in_) != EXPECTED_FEATURES:
        raise RuntimeError("Final-development predictor scaler width drift.")
    if bool(getattr(scaler, "clip", False)) is not False:
        raise RuntimeError("Final-development predictor scaler unexpectedly clips.")
    if hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) != features:
        raise RuntimeError("Final-development scaler feature order drift.")
    del scaler

    if not FINAL_DEV_HISTORY.exists():
        raise FileNotFoundError(f"Final-development scaled regime history missing:\n{FINAL_DEV_HISTORY}")
    if not FINAL_DEV_REGIME_THRESHOLDS.exists():
        raise FileNotFoundError(f"Final-development regime-threshold table missing:\n{FINAL_DEV_REGIME_THRESHOLDS}")

    regime_thresholds = pd.read_csv(FINAL_DEV_REGIME_THRESHOLDS)
    if len(regime_thresholds) != EXPECTED_STOCKS:
        raise RuntimeError("Final-development regime threshold row count drift.")
    required_thr = {"ticker", "low_volatility_threshold_q33", "high_volatility_threshold_q67"}
    if not required_thr.issubset(regime_thresholds.columns):
        raise RuntimeError("Final-development regime threshold schema drift.")

    thresholds = pd.read_csv(STEP35_THRESHOLDS)
    required_t = {"model_id", "config_id", "horizon_days"}
    if not required_t.issubset(thresholds.columns) or len(thresholds) != EXPECTED_THRESHOLDS:
        raise RuntimeError("Step-35 threshold snapshot is incomplete.")
    threshold_col = "selected_threshold" if "selected_threshold" in thresholds.columns else "threshold" if "threshold" in thresholds.columns else None
    if threshold_col is None:
        raise RuntimeError("Step-35 threshold snapshot has no threshold column.")

    for model_id in MODEL_ORDER:
        sub = thresholds.loc[thresholds["model_id"] == model_id]
        if len(sub) != 3 or set(sub["horizon_days"].astype(int)) != set(HORIZONS):
            raise RuntimeError(f"{model_id}: frozen threshold rows drift.")
        if set(sub["config_id"].astype(str)) != {EXPECTED_CONFIG[model_id]}:
            raise RuntimeError(f"{model_id}: frozen threshold config drift.")

    fits = pd.read_csv(STEP34_REGISTER)
    required_f = {"model_id", "config_id", "seed", "physical_microbatch_size", "fit_manifest_path", "fit_manifest_sha256"}
    if not required_f.issubset(fits.columns) or len(fits) != EXPECTED_FITS:
        raise RuntimeError("Step-34 final fit register is incomplete.")

    for model_id in MODEL_ORDER:
        sub = fits.loc[fits["model_id"] == model_id]
        if len(sub) != 5 or set(sub["seed"].astype(int)) != set(EXPECTED_SEEDS):
            raise RuntimeError(f"{model_id}: final refit seed set drift.")
        if set(sub["config_id"].astype(str)) != {EXPECTED_CONFIG[model_id]}:
            raise RuntimeError(f"{model_id}: final refit config drift.")
        if set(sub["physical_microbatch_size"].astype(int)) != {EXPECTED_PHYSICAL_MB[model_id]}:
            raise RuntimeError(f"{model_id}: final refit microbatch drift.")
        for row in sub.itertuples(index=False):
            mp = Path(row.fit_manifest_path)
            require_file_sha(mp, str(row.fit_manifest_sha256), f"{model_id}/seed{int(row.seed)} fit manifest")
            fm = load_json(mp)
            cp_rec = fm.get("artifacts", {}).get("final_checkpoint.pt")
            if not cp_rec:
                raise RuntimeError(f"{model_id}/seed{int(row.seed)} checkpoint artifact missing.")
            cp = Path(cp_rec.get("path", mp.parent / "final_checkpoint.pt"))
            if not cp.exists():
                raise FileNotFoundError(f"Frozen checkpoint missing:\n{cp}")
            if cp_rec.get("sha256") and sha256_file(cp) != cp_rec["sha256"]:
                raise RuntimeError(f"{model_id}/seed{int(row.seed)} checkpoint SHA drift.")

    if STEP35_INFERENCE_PLAN.exists():
        plan = pd.read_csv(STEP35_INFERENCE_PLAN)
        if len(plan) not in (8, 40):
            raise RuntimeError(f"Step-35 inference plan row count drift: {len(plan)}")

    return {
        "step14_manifest": m14,
        "step34_manifest": m34,
        "step35_manifest": m35,
        "test_seal": seal,
        "features": features,
        "stock_id_map": {str(k): int(v) for k, v in stock_id_map.items()},
        "scaler_path": scaler_path,
        "threshold_col": threshold_col,
    }


def prepare_only() -> int:
    print("=" * 140)
    print(STEP_NAME)
    print("=" * 140)
    print("MODE                     : PREPARE-ONLY")
    print("Final-test parquet read  : PROHIBITED")
    print("Final-test parquet hash  : PROHIBITED")
    print("Final-test class balance : NOT INSPECTED")
    print("Final-test metrics       : NOT COMPUTED")
    print("-" * 140)

    upstream = verify_upstream_without_test_access()
    source_sha = sha256_file(SCRIPT_FILE)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "STEP36_PREACCESS_EXECUTION_SOURCE_FROZEN",
        "frozen_at_utc": utc_now(),
        "source_file": str(SCRIPT_FILE),
        "source_sha256": source_sha,
        "step14_manifest_sha256": EXPECTED_STEP14_MANIFEST_SHA,
        "step32_manifest_sha256": EXPECTED_STEP32_MANIFEST_SHA,
        "step34_manifest_sha256": EXPECTED_STEP34_MANIFEST_SHA,
        "step34_global_fingerprint": EXPECTED_STEP34_GLOBAL_FP,
        "step35_manifest_sha256": EXPECTED_STEP35_MANIFEST_SHA,
        "step35_contract_fingerprint": EXPECTED_STEP35_CONTRACT_FP,
        "execution_contract": {
            "outer_test_rows_expected": EXPECTED_OUTER_TEST_ROWS,
            "outer_test_dates_expected": EXPECTED_OUTER_TEST_DATES,
            "common_test_rows_expected": EXPECTED_COMMON_ROWS,
            "common_test_dates_expected": EXPECTED_COMMON_DATES,
            "stock_count": EXPECTED_STOCKS,
            "lookback": EXPECTED_LOOKBACK,
            "feature_count": EXPECTED_FEATURES,
            "horizons": list(HORIZONS),
            "architectures": list(MODEL_ORDER),
            "seeds": list(EXPECTED_SEEDS),
            "final_checkpoints": EXPECTED_FITS,
            "frozen_thresholds": EXPECTED_THRESHOLDS,
            "seed_probability_ensemble": "ARITHMETIC_MEAN_OF_FIVE_SEED_PROBABILITIES",
            "threshold_operator": ">=",
            "test_threshold_tuning": False,
            "best_seed_selection": False,
            "architecture_selection": False,
            "post_test_model_promotion": False,
            "raw_test_filesystem_open_count": 1,
            "raw_test_reopen_after_materialization": False,
            "inference_resumable_from_materialized_arrays": True,
        },
        "sealed_test_metadata_only": {
            "path": upstream["test_seal"]["file_path"],
            "expected_sha256": upstream["test_seal"]["sha256"],
            "seal_status": upstream["test_seal"]["status"],
            "global_preaccess_state": upstream["test_seal"]["global_final_test_lock_state"],
        },
        "final_development_transform_state": {
            "predictor_scaler_path": str(upstream["scaler_path"]),
            "predictor_scaler_sha256": sha256_file(upstream["scaler_path"]),
            "regime_threshold_path": str(FINAL_DEV_REGIME_THRESHOLDS),
            "regime_threshold_sha256": sha256_file(FINAL_DEV_REGIME_THRESHOLDS),
            "development_history_path": str(FINAL_DEV_HISTORY),
            "development_history_sha256": sha256_file(FINAL_DEV_HISTORY),
        },
        "protection": {
            "final_test_opened_during_prepare": False,
            "final_test_hashed_during_prepare": False,
            "final_test_scored_during_prepare": False,
        },
    }
    payload["step36_preaccess_fingerprint"] = canonical_hash(payload)

    if PREACCESS_FREEZE.exists():
        existing = load_json(PREACCESS_FREEZE)
        if existing.get("source_sha256") != source_sha:
            raise RuntimeError("A Step-36 pre-access freeze exists for a different source. Do not overwrite it.")
        print("Existing pre-access freeze : VERIFIED")
    else:
        shutil.copy2(SCRIPT_FILE, FROZEN_SOURCE)
        atomic_json(PREACCESS_FREEZE, payload)
        freeze_sha = sha256_file(PREACCESS_FREEZE)
        atomic_text(PREACCESS_FREEZE_SHA, f"{freeze_sha}  {PREACCESS_FREEZE.name}\n")

    freeze_sha = sha256_file(PREACCESS_FREEZE)
    sidecar = PREACCESS_FREEZE_SHA.read_text(encoding="utf-8").strip().split()[0]
    require_equal(freeze_sha, sidecar, "Step-36 pre-access freeze sidecar")
    require_equal(sha256_file(FROZEN_SOURCE), source_sha, "Frozen Step-36 source SHA")

    print("Upstream freezes           : VERIFIED")
    print(f"Step-35 contract FP        : {EXPECTED_STEP35_CONTRACT_FP}")
    print(f"Step-36 source SHA         : {source_sha}")
    print(f"Step-36 preaccess FP       : {load_json(PREACCESS_FREEZE)['step36_preaccess_fingerprint']}")
    print(f"Preaccess manifest SHA     : {freeze_sha}")
    print("Final-test parquet opened  : False")
    print("Final-test parquet hashed  : False")
    print("Final-test metrics computed: False")
    print("=" * 140)
    print("PREPARE-ONLY PASS — STEP-36 SOURCE/EXECUTION CONTRACT FROZEN")
    print("=" * 140)
    print("DO NOT RUN --execute UNTIL THIS PREPARE-ONLY OUTPUT HAS BEEN AUDITED.")
    return 0


def verify_preaccess_freeze() -> dict:
    if not PREACCESS_FREEZE.exists() or not PREACCESS_FREEZE_SHA.exists():
        raise RuntimeError("Step-36 pre-access freeze is missing. Run --prepare-only first.")
    expected = PREACCESS_FREEZE_SHA.read_text(encoding="utf-8").strip().split()[0]
    observed = sha256_file(PREACCESS_FREEZE)
    require_equal(observed, expected, "Step-36 pre-access freeze SHA sidecar")
    payload = load_json(PREACCESS_FREEZE)
    require_equal(payload.get("status"), "STEP36_PREACCESS_EXECUTION_SOURCE_FROZEN", "Step-36 pre-access status")
    require_equal(payload.get("source_sha256"), sha256_file(SCRIPT_FILE), "Current Step-36 source vs pre-access freeze")
    require_equal(sha256_file(FROZEN_SOURCE), payload["source_sha256"], "Frozen Step-36 source")
    return payload


def write_access_marker(state: str, **extra) -> None:
    prior = load_json(ACCESS_MARKER) if ACCESS_MARKER.exists() else {}
    payload = {
        **prior,
        "schema_version": SCHEMA_VERSION,
        "step": 36,
        "state": state,
        "updated_at_utc": utc_now(),
        "step35_manifest_sha256": EXPECTED_STEP35_MANIFEST_SHA,
        "step35_contract_fingerprint": EXPECTED_STEP35_CONTRACT_FP,
        **extra,
    }
    if "created_at_utc" not in payload:
        payload["created_at_utc"] = utc_now()
    atomic_json(ACCESS_MARKER, payload)


def marker_state() -> str | None:
    return None if not ACCESS_MARKER.exists() else str(load_json(ACCESS_MARKER).get("state"))


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    for h in HORIZONS:
        c = f"target_date_{h}d"
        if c in frame.columns:
            frame[c] = pd.to_datetime(frame[c], errors="coerce").dt.normalize()
    return frame


def load_and_parse_raw_test_exactly_once(test_path: Path, expected_sha: str) -> tuple[pd.DataFrame, str]:
    with test_path.open("rb") as handle:
        raw_bytes = handle.read()
    observed_sha = sha256_bytes(raw_bytes)
    require_equal(observed_sha, expected_sha, "Sealed final-test parquet SHA")
    frame = pd.read_parquet(io.BytesIO(raw_bytes), engine="pyarrow")
    del raw_bytes
    gc.collect()
    return frame, observed_sha


def apply_frozen_regimes(raw_test: pd.DataFrame) -> pd.DataFrame:
    import step12_volatility_regimes_v3 as s12
    thresholds = pd.read_csv(FINAL_DEV_REGIME_THRESHOLDS)
    out = s12.apply_threshold_state(raw_test.copy(), thresholds, fit_scope="final_development")
    out["regime_threshold_fit_scope"] = "final_development"
    return out


def transform_frozen_predictors(frame: pd.DataFrame, features: list[str], scaler_path: Path) -> pd.DataFrame:
    scaler = joblib.load(scaler_path)
    if int(scaler.n_features_in_) != len(features):
        raise RuntimeError("Final-development scaler width mismatch at execution.")
    if bool(getattr(scaler, "clip", False)) is not False:
        raise RuntimeError("Final-development scaler clip drift.")
    if hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) != features:
        raise RuntimeError("Final-development scaler feature-order drift.")
    transformed = scaler.transform(frame[features].astype(np.float64))
    if not np.isfinite(transformed).all():
        raise RuntimeError("Non-finite final-test scaled predictor detected.")
    out = frame.copy()
    out.loc[:, features] = transformed
    del scaler, transformed
    return out


def validate_opened_test(frame: pd.DataFrame, features: list[str]) -> None:
    required = {
        "Ticker", "Date", "market_session_id", "continuity_segment_id",
        "eligible_common_multitask", "direction_1d", "direction_3d", "direction_5d",
        "target_date_1d", "target_date_3d", "target_date_5d", *features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Final-test panel missing required columns:\n{missing}")
    require_equal(len(frame), EXPECTED_OUTER_TEST_ROWS, "Final-test rows")
    require_equal(frame["Date"].nunique(), EXPECTED_OUTER_TEST_DATES, "Final-test dates")
    require_equal(frame["Ticker"].nunique(), EXPECTED_STOCKS, "Final-test stocks")
    if frame[["Ticker", "Date"]].duplicated().any():
        raise RuntimeError("Duplicate final-test ticker-date keys.")
    if not (frame.groupby("Date")["Ticker"].nunique() == EXPECTED_STOCKS).all():
        raise RuntimeError("Final-test cross-section is incomplete.")
    if not np.isfinite(frame[features].to_numpy(np.float64)).all():
        raise RuntimeError("Final-test predictors contain non-finite values.")
    for c in DIRECTION_COLUMNS:
        vals = pd.to_numeric(frame[c], errors="raise")
        if not vals.isin([0, 1]).all():
            raise RuntimeError(f"Non-binary final-test direction label: {c}")


def build_common_endpoints(scaled_test: pd.DataFrame, stock_id_map: dict[str, int]) -> pd.DataFrame:
    flag = scaled_test["eligible_common_multitask"]
    if flag.dtype != bool:
        flag = flag.astype(bool)
    ep = scaled_test.loc[flag].copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)
    require_equal(len(ep), EXPECTED_COMMON_ROWS, "Common final-test endpoint rows")
    require_equal(ep["Date"].nunique(), EXPECTED_COMMON_DATES, "Common final-test endpoint dates")
    if not (ep.groupby("Date")["Ticker"].nunique() == EXPECTED_STOCKS).all():
        raise RuntimeError("Common final-test endpoints are not balanced by date.")
    unknown = sorted(set(ep["Ticker"]) - set(stock_id_map))
    if unknown:
        raise RuntimeError(f"Unknown test tickers in stock_id_map: {unknown}")
    ep["stock_id"] = ep["Ticker"].map(stock_id_map).astype(np.int64)
    for h in HORIZONS:
        td = pd.to_datetime(ep[f"target_date_{h}d"], errors="raise")
        rd = pd.to_datetime(ep["Date"], errors="raise")
        if (td <= rd).any():
            raise RuntimeError(f"{h}d contains a non-future target date.")
    return ep


def build_history_lookup(final_dev_scaled: pd.DataFrame, test_scaled: pd.DataFrame, features: list[str]) -> dict:
    keep = ["Ticker", "Date", "market_session_id", "continuity_segment_id", *features]
    for label, df in (("final-development", final_dev_scaled), ("final-test", test_scaled)):
        missing = sorted(set(keep) - set(df.columns))
        if missing:
            raise RuntimeError(f"{label} history missing columns: {missing}")
    history = pd.concat([final_dev_scaled[keep], test_scaled[keep]], ignore_index=True)
    history = history.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    if history[["Ticker", "Date"]].duplicated().any():
        raise RuntimeError("Combined development/test history has duplicate keys.")
    if not np.isfinite(history[features].to_numpy(np.float64)).all():
        raise RuntimeError("Combined history contains non-finite predictors.")
    lookup = {}
    for (ticker, segment), g in history.groupby(["Ticker", "continuity_segment_id"], sort=True):
        table = g.sort_values("Date").reset_index(drop=True)
        lookup[(str(ticker), int(segment))] = {
            "positions": {pd.Timestamp(d): int(i) for i, d in enumerate(table["Date"].tolist())},
            "values": table[features].to_numpy(dtype=np.float32),
            "sessions": table["market_session_id"].to_numpy(dtype=np.int64),
            "dates": pd.to_datetime(table["Date"]).to_numpy(),
        }
    return lookup


def materialize_sequences(endpoints: pd.DataFrame, history_lookup: dict, features: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    n = len(endpoints)
    X = np.empty((n, EXPECTED_LOOKBACK, EXPECTED_FEATURES), dtype=np.float32)
    md_rows = []
    for out_i, row in enumerate(endpoints.itertuples(index=False)):
        key = (str(row.Ticker), int(row.continuity_segment_id))
        if key not in history_lookup:
            raise RuntimeError(f"Missing history key: {key}")
        item = history_lookup[key]
        reference_date = pd.Timestamp(row.Date)
        if reference_date not in item["positions"]:
            raise RuntimeError(f"Endpoint absent from history: {key} {reference_date.date()}")
        end = int(item["positions"][reference_date])
        start = end - EXPECTED_LOOKBACK + 1
        if start < 0:
            raise RuntimeError(f"Insufficient causal history: {key} {reference_date.date()}")
        seq = item["values"][start:end + 1]
        sess = item["sessions"][start:end + 1]
        dates = item["dates"][start:end + 1]
        if seq.shape != (60, 63):
            raise RuntimeError("Final-test sequence shape drift.")
        if int(sess[-1]) - int(sess[0]) != 59:
            raise RuntimeError("Non-consecutive master-session final-test window.")
        if pd.Timestamp(dates[-1]) != reference_date:
            raise RuntimeError("Final-test sequence does not end at endpoint.")
        if (pd.to_datetime(dates) > reference_date).any():
            raise RuntimeError("Future predictor row entered final-test sequence.")
        X[out_i] = seq
        md_rows.append({
            "array_row_index": out_i,
            "ticker": str(row.Ticker),
            "stock_id": int(row.stock_id),
            "continuity_segment_id": int(row.continuity_segment_id),
            "reference_date": reference_date,
            "reference_market_session_id": int(row.market_session_id),
            "sequence_start_date": pd.Timestamp(dates[0]),
            "sequence_end_date": reference_date,
            "sequence_start_market_session_id": int(sess[0]),
            "sequence_end_market_session_id": int(sess[-1]),
            "regime_id": int(row.regime_id),
            "regime_name": str(row.regime_name),
            "regime_threshold_fit_scope": str(row.regime_threshold_fit_scope),
            "target_date_1d": pd.Timestamp(row.target_date_1d),
            "target_date_3d": pd.Timestamp(row.target_date_3d),
            "target_date_5d": pd.Timestamp(row.target_date_5d),
            "direction_1d": int(row.direction_1d),
            "direction_3d": int(row.direction_3d),
            "direction_5d": int(row.direction_5d),
        })
    if not np.isfinite(X).all():
        raise RuntimeError("Materialized final-test X contains non-finite values.")
    return X, pd.DataFrame(md_rows)


def save_array(path: Path, array: np.ndarray) -> None:
    np.save(path, array, allow_pickle=False)
    check = np.load(path, mmap_mode="r", allow_pickle=False)
    if check.shape != array.shape or check.dtype != array.dtype:
        raise RuntimeError(f"NumPy roundtrip drift: {path.name}")
    del check


def verify_materialized_arrays() -> dict:
    m = load_json(ARRAY_MANIFEST)
    require_equal(m.get("status"), "STEP36_FINAL_TEST_MATERIALIZATION_COMPLETE", "Step-36 materialization status")
    require_equal(int(m["raw_test"]["filesystem_content_open_count"]), 1, "Raw final-test filesystem content-open count")
    require_equal(int(m["evaluation_panel"]["rows"]), EXPECTED_COMMON_ROWS, "Materialized evaluation rows")
    for rec in m.get("artifacts", {}).values():
        p = Path(rec["path"])
        if not p.exists() or sha256_file(p) != rec["sha256"]:
            raise RuntimeError(f"Step-36 materialized artifact drift:\n{p}")
    return m


def materialize_final_test(upstream: dict) -> dict:
    if ARRAY_MANIFEST.exists():
        return verify_materialized_arrays()
    if marker_state() is not None:
        raise RuntimeError(
            "FINAL-TEST ACCESS MARKER EXISTS BUT MATERIALIZATION MANIFEST IS MISSING.\n"
            "The raw final-test may already have been opened. Automatic raw-test reopening is prohibited."
        )

    seal = upstream["test_seal"]
    raw_path = Path(seal["file_path"])
    expected_raw_sha = str(seal["sha256"])

    write_access_marker(
        "RAW_TEST_ACCESS_CONSUMPTION_STARTED",
        raw_test_path=str(raw_path),
        expected_raw_test_sha256=expected_raw_sha,
        raw_test_filesystem_open_count_planned=1,
        raw_test_filesystem_open_count_completed=0,
        final_test_scored=False,
    )

    print("Opening sealed final-test parquet ONCE...")
    raw_test, observed_raw_sha = load_and_parse_raw_test_exactly_once(raw_path, expected_raw_sha)
    write_access_marker(
        "RAW_TEST_OPENED_ONCE_IN_MEMORY",
        observed_raw_test_sha256=observed_raw_sha,
        raw_test_filesystem_open_count_completed=1,
        final_test_scored=False,
    )

    raw_test = normalize_dates(raw_test)
    validate_opened_test(raw_test, upstream["features"])
    test_regimes = apply_frozen_regimes(raw_test)
    test_scaled = transform_frozen_predictors(test_regimes, upstream["features"], upstream["scaler_path"])
    endpoints = build_common_endpoints(test_scaled, upstream["stock_id_map"])

    final_dev = normalize_dates(pd.read_parquet(FINAL_DEV_HISTORY, engine="pyarrow"))
    lookup = build_history_lookup(final_dev, test_scaled, upstream["features"])
    X_full, metadata = materialize_sequences(endpoints, lookup, upstream["features"])
    y_direction = metadata[list(DIRECTION_COLUMNS)].to_numpy(dtype=np.int64)
    stock_id = metadata["stock_id"].to_numpy(dtype=np.int64)
    regime_id = metadata["regime_id"].to_numpy(dtype=np.int64)
    X_stock = X_full[:, :, :47].copy()
    X_market = X_full[:, :, 47:].copy()

    require_equal(X_full.shape, (EXPECTED_COMMON_ROWS, 60, 63), "X_full shape")
    require_equal(X_stock.shape, (EXPECTED_COMMON_ROWS, 60, 47), "X_stock shape")
    require_equal(X_market.shape, (EXPECTED_COMMON_ROWS, 60, 16), "X_market shape")
    require_equal(y_direction.shape, (EXPECTED_COMMON_ROWS, 3), "y_direction shape")
    if not np.array_equal(X_stock, X_full[:, :, :47]) or not np.array_equal(X_market, X_full[:, :, 47:]):
        raise RuntimeError("Final-test modality slice mismatch.")
    if not np.isin(y_direction, [0, 1]).all():
        raise RuntimeError("Final-test y_direction is not binary.")
    if not np.isin(regime_id, [0, 1, 2]).all():
        raise RuntimeError("Final-test regime IDs are invalid.")

    staging = cfg.FEATURE_DATA_DIR / f"_step36_final_test_staging_{os.getpid()}_{int(time.time())}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        save_array(staging / "36_final_test_X_full.npy", X_full)
        save_array(staging / "36_final_test_X_stock.npy", X_stock)
        save_array(staging / "36_final_test_X_market.npy", X_market)
        save_array(staging / "36_final_test_y_direction.npy", y_direction)
        save_array(staging / "36_final_test_stock_id.npy", stock_id)
        save_array(staging / "36_final_test_regime_id.npy", regime_id)
        metadata.to_parquet(staging / "36_final_test_metadata.parquet", index=False, engine="pyarrow", compression="snappy")
        if len(pd.read_parquet(staging / "36_final_test_metadata.parquet", engine="pyarrow")) != EXPECTED_COMMON_ROWS:
            raise RuntimeError("Final-test metadata roundtrip row drift.")
        if ARRAY_DIR.exists():
            raise RuntimeError(f"Unexpected existing Step-36 array directory:\n{ARRAY_DIR}")
        os.replace(staging, ARRAY_DIR)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    artifacts = {p.name: artifact_record(p) for p in sorted(ARRAY_DIR.iterdir()) if p.is_file()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "STEP36_FINAL_TEST_MATERIALIZATION_COMPLETE",
        "created_at_utc": utc_now(),
        "step35_manifest_sha256": EXPECTED_STEP35_MANIFEST_SHA,
        "step35_contract_fingerprint": EXPECTED_STEP35_CONTRACT_FP,
        "raw_test": {
            "path": str(raw_path),
            "expected_sha256": expected_raw_sha,
            "observed_sha256": observed_raw_sha,
            "filesystem_content_open_count": 1,
            "parsed_from_same_in_memory_bytes_used_for_hash": True,
            "rows": int(len(raw_test)),
            "dates": int(raw_test["Date"].nunique()),
            "stocks": int(raw_test["Ticker"].nunique()),
        },
        "transformation": {
            "predictor_scaler_path": str(upstream["scaler_path"]),
            "predictor_scaler_sha256": sha256_file(upstream["scaler_path"]),
            "predictor_scaler_refit_on_test": False,
            "predictor_clipping": False,
            "regime_threshold_path": str(FINAL_DEV_REGIME_THRESHOLDS),
            "regime_threshold_sha256": sha256_file(FINAL_DEV_REGIME_THRESHOLDS),
            "regime_threshold_refit_on_test": False,
            "development_history_path": str(FINAL_DEV_HISTORY),
            "development_history_sha256": sha256_file(FINAL_DEV_HISTORY),
            "lookback": 60,
            "causal_development_history_allowed": True,
            "future_predictor_rows_allowed": False,
        },
        "evaluation_panel": {
            "rows": EXPECTED_COMMON_ROWS,
            "dates": EXPECTED_COMMON_DATES,
            "stocks": EXPECTED_STOCKS,
            "x_full_shape": [EXPECTED_COMMON_ROWS, 60, 63],
            "y_direction_shape": [EXPECTED_COMMON_ROWS, 3],
        },
        "artifacts": artifacts,
    }
    manifest["materialization_fingerprint"] = canonical_hash(manifest)
    atomic_json(ARRAY_MANIFEST, manifest)
    write_access_marker(
        "RAW_TEST_MATERIALIZED_AND_CLOSED",
        observed_raw_test_sha256=observed_raw_sha,
        raw_test_filesystem_open_count_completed=1,
        raw_test_reopen_prohibited=True,
        materialization_manifest_path=str(ARRAY_MANIFEST),
        materialization_manifest_sha256=sha256_file(ARRAY_MANIFEST),
        final_test_scored=False,
    )
    return verify_materialized_arrays()


def get_fit_row(model_id: str, seed: int) -> pd.Series:
    reg = pd.read_csv(STEP34_REGISTER)
    sub = reg.loc[(reg["model_id"] == model_id) & (reg["seed"].astype(int) == int(seed))]
    if len(sub) != 1:
        raise RuntimeError(f"Could not uniquely identify Step-34 fit for {model_id}/seed{seed}.")
    return sub.iloc[0]


def worker_output_dir(model_id: str, seed: int) -> Path:
    return SEED_PRED_ROOT / model_id / f"seed_{seed}"


def worker_manifest_path(model_id: str, seed: int) -> Path:
    return worker_output_dir(model_id, seed) / "inference_manifest.json"


def verify_worker_complete(model_id: str, seed: int) -> dict | None:
    mp = worker_manifest_path(model_id, seed)
    if not mp.exists():
        return None
    m = load_json(mp)
    if m.get("status") != "PASS":
        return None
    pp = Path(m["prediction_artifact"]["path"])
    if not pp.exists() or sha256_file(pp) != m["prediction_artifact"]["sha256"]:
        return None
    return m


def run_worker(model_id: str, seed: int) -> int:
    import torch
    verify_materialized_arrays()
    if model_id not in MODEL_ORDER or seed not in EXPECTED_SEEDS:
        raise RuntimeError("Unexpected worker model/seed.")

    fit_row = get_fit_row(model_id, seed)
    config_id = str(fit_row["config_id"])
    require_equal(config_id, EXPECTED_CONFIG[model_id], f"{model_id}/seed{seed} config")
    physical_mb = int(fit_row["physical_microbatch_size"])
    require_equal(physical_mb, EXPECTED_PHYSICAL_MB[model_id], f"{model_id}/seed{seed} physical microbatch")

    fit_manifest_path = Path(fit_row["fit_manifest_path"])
    require_file_sha(fit_manifest_path, str(fit_row["fit_manifest_sha256"]), f"{model_id}/seed{seed} fit manifest")
    fm = load_json(fit_manifest_path)
    cp_rec = fm["artifacts"]["final_checkpoint.pt"]
    checkpoint_path = Path(cp_rec.get("path", fit_manifest_path.parent / "final_checkpoint.pt"))
    if cp_rec.get("sha256"):
        require_file_sha(checkpoint_path, cp_rec["sha256"], f"{model_id}/seed{seed} checkpoint")

    import step26_stage1_runner_dryrun_v3 as s26
    s26.set_training_determinism(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen Step-36 neural inference.")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model, build_metadata, repository = s26.build_candidate_model(
        model_id=model_id,
        config_id=config_id,
        physical_microbatch=physical_mb,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    X = np.load(ARRAY_DIR / "36_final_test_X_full.npy", mmap_mode="r", allow_pickle=False)
    require_equal(X.shape, (EXPECTED_COMMON_ROWS, 60, 63), "Worker final-test X shape")
    logits_parts, prob_parts = [], []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, EXPECTED_COMMON_ROWS, physical_mb):
            end = min(start + physical_mb, EXPECTED_COMMON_ROWS)
            xb = torch.from_numpy(np.asarray(X[start:end], dtype=np.float32)).to(device=device, dtype=torch.float32, non_blocking=False)
            logits, _ = model(xb)
            if tuple(logits.shape) != (end - start, 3):
                raise RuntimeError(f"{model_id}/seed{seed}: inference logits shape drift.")
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError(f"{model_id}/seed{seed}: non-finite logits.")
            prob = torch.sigmoid(logits)
            logits_parts.append(logits.detach().cpu().to(torch.float32).numpy())
            prob_parts.append(prob.detach().cpu().to(torch.float32).numpy())
            del xb, logits, prob
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    logits_np = np.concatenate(logits_parts, axis=0)
    prob_np = np.concatenate(prob_parts, axis=0)
    require_equal(logits_np.shape, (EXPECTED_COMMON_ROWS, 3), "Worker logits array shape")
    require_equal(prob_np.shape, (EXPECTED_COMMON_ROWS, 3), "Worker probability array shape")
    if not np.isfinite(logits_np).all() or not np.isfinite(prob_np).all():
        raise RuntimeError("Non-finite worker inference output.")

    outdir = worker_output_dir(model_id, seed)
    if outdir.exists():
        existing = verify_worker_complete(model_id, seed)
        if existing is not None:
            return 0
        shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=False)
    pred_path = outdir / "predictions.npz"
    np.savez_compressed(
        pred_path,
        logits=logits_np.astype(np.float32),
        probabilities=prob_np.astype(np.float32),
        row_index=np.arange(EXPECTED_COMMON_ROWS, dtype=np.int64),
    )
    pred_rec = artifact_record(pred_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "created_at_utc": utc_now(),
        "model_id": model_id,
        "config_id": config_id,
        "seed": int(seed),
        "physical_microbatch_size": physical_mb,
        "n": EXPECTED_COMMON_ROWS,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "fit_manifest_path": str(fit_manifest_path),
        "fit_manifest_sha256": sha256_file(fit_manifest_path),
        "model_build_metadata": build_metadata,
        "source_repository_record": repository,
        "inference_shuffle": False,
        "gradient_tracking": False,
        "seed_selection": False,
        "threshold_applied_in_worker": False,
        "inference_seconds": float(seconds),
        "examples_per_second": float(EXPECTED_COMMON_ROWS / seconds),
        "peak_cuda_allocated_mib": peak_mib,
        "prediction_artifact": pred_rec,
    }
    atomic_json(outdir / "inference_manifest.json", manifest)
    print(json.dumps({"status": "PASS", "model_id": model_id, "seed": seed, "seconds": seconds, "peak_cuda_allocated_mib": peak_mib}, sort_keys=True))
    return 0


def execute_all_workers() -> pd.DataFrame:
    SEED_PRED_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_id in MODEL_ORDER:
        for seed in EXPECTED_SEEDS:
            m = verify_worker_complete(model_id, seed)
            if m is None:
                print(f"RUN  {model_id:14s} seed={seed}")
                env = os.environ.copy()
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                proc = subprocess.run(
                    [sys.executable, str(SCRIPT_FILE), "--worker", "--worker-model", model_id, "--worker-seed", str(seed)],
                    cwd=str(SCRIPT_DIR), env=env, text=True, capture_output=True,
                )
                if proc.stdout:
                    print(proc.stdout, end="")
                if proc.stderr:
                    print(proc.stderr, end="", file=sys.stderr)
                if proc.returncode != 0:
                    raise RuntimeError(f"Step-36 worker failed: {model_id}/seed{seed}")
                m = verify_worker_complete(model_id, seed)
                if m is None:
                    raise RuntimeError(f"Worker success but artifact verification failed: {model_id}/seed{seed}")
            else:
                print(f"SKIP {model_id:14s} seed={seed} COMPLETE+HASHED")
            rows.append({
                "model_id": model_id,
                "config_id": EXPECTED_CONFIG[model_id],
                "seed": seed,
                "status": "PASS",
                "physical_microbatch_size": int(m["physical_microbatch_size"]),
                "inference_seconds": float(m["inference_seconds"]),
                "examples_per_second": float(m["examples_per_second"]),
                "peak_cuda_allocated_mib": float(m["peak_cuda_allocated_mib"]),
                "prediction_path": m["prediction_artifact"]["path"],
                "prediction_sha256": m["prediction_artifact"]["sha256"],
            })
    df = pd.DataFrame(rows)
    require_equal(len(df), EXPECTED_FITS, "Inference register rows")
    write_csv_atomic(INFERENCE_REGISTER_CSV, df)
    return df


def frozen_threshold_map() -> dict[tuple[str, int], float]:
    df = pd.read_csv(STEP35_THRESHOLDS)
    col = "selected_threshold" if "selected_threshold" in df.columns else "threshold"
    result = {(str(r.model_id), int(r.horizon_days)): float(getattr(r, col)) for r in df.itertuples(index=False)}
    require_equal(len(result), EXPECTED_THRESHOLDS, "Frozen threshold map size")
    return result


def binary_diagnostics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "predicted_positive_rate": float(pred.mean()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def build_ensembles_and_score() -> tuple[pd.DataFrame, pd.DataFrame]:
    ENSEMBLE_ROOT.mkdir(parents=True, exist_ok=True)
    y = np.load(ARRAY_DIR / "36_final_test_y_direction.npy", allow_pickle=False)
    metadata = pd.read_parquet(ARRAY_DIR / "36_final_test_metadata.parquet", engine="pyarrow")
    require_equal(y.shape, (EXPECTED_COMMON_ROWS, 3), "Held-out label array shape")
    thresholds = frozen_threshold_map()

    class_df = pd.DataFrame([
        {
            "horizon_days": h,
            "n": EXPECTED_COMMON_ROWS,
            "positive_count": int(y[:, j].sum()),
            "negative_count": int(EXPECTED_COMMON_ROWS - y[:, j].sum()),
            "positive_rate": float(y[:, j].mean()),
            "used_for_development": False,
            "used_for_threshold_selection": False,
            "observed_only_after_step35_freeze": True,
        }
        for j, h in enumerate(HORIZONS)
    ])
    write_csv_atomic(CLASS_BALANCE_CSV, class_df)

    metric_rows, long_parts = [], []
    for model_id in MODEL_ORDER:
        seed_probs = []
        for seed in EXPECTED_SEEDS:
            m = verify_worker_complete(model_id, seed)
            if m is None:
                raise RuntimeError(f"Missing worker output: {model_id}/seed{seed}")
            data = np.load(m["prediction_artifact"]["path"], allow_pickle=False)
            probs = np.asarray(data["probabilities"], dtype=np.float64)
            row_index = np.asarray(data["row_index"], dtype=np.int64)
            require_equal(probs.shape, (EXPECTED_COMMON_ROWS, 3), "Seed probability shape")
            if not np.array_equal(row_index, np.arange(EXPECTED_COMMON_ROWS, dtype=np.int64)):
                raise RuntimeError("Seed row-index drift.")
            seed_probs.append(probs)
        ensemble = np.mean(np.stack(seed_probs, axis=0), axis=0)
        out_npz = ENSEMBLE_ROOT / f"{model_id}_five_seed_ensemble.npz"
        np.savez_compressed(out_npz, probabilities=ensemble.astype(np.float32), row_index=np.arange(EXPECTED_COMMON_ROWS, dtype=np.int64))

        part = metadata[[
            "array_row_index", "ticker", "stock_id", "continuity_segment_id",
            "reference_date", "reference_market_session_id", "regime_id", "regime_name",
            "target_date_1d", "target_date_3d", "target_date_5d",
            "direction_1d", "direction_3d", "direction_5d",
        ]].copy()
        part.insert(0, "model_id", model_id)
        part.insert(1, "config_id", EXPECTED_CONFIG[model_id])

        for j, h in enumerate(HORIZONS):
            t = float(thresholds[(model_id, h)])
            p = ensemble[:, j]
            pred = (p >= t).astype(np.int64)
            part[f"probability_{h}d"] = p.astype(np.float32)
            part[f"frozen_threshold_{h}d"] = t
            part[f"prediction_{h}d"] = pred
            metric_rows.append({
                "model_id": model_id,
                "config_id": EXPECTED_CONFIG[model_id],
                "horizon_days": h,
                "n": EXPECTED_COMMON_ROWS,
                "positive_rate": float(y[:, j].mean()),
                "threshold": t,
                "threshold_source": "FROZEN_DEVELOPMENT_OOF_STEP35",
                "threshold_operator": ">=",
                "five_seed_probability_ensemble": True,
                "seed_selection": False,
                "architecture_selection": False,
                "test_threshold_tuning": False,
                **binary_diagnostics(y[:, j], p, t),
            })
        long_parts.append(part)

    metrics = pd.DataFrame(metric_rows)
    require_equal(len(metrics), EXPECTED_THRESHOLDS, "Held-out metric rows")
    write_csv_atomic(METRICS_CSV, metrics)
    pd.concat(long_parts, ignore_index=True).to_parquet(
        ENSEMBLE_PREDICTIONS_PARQUET, index=False, engine="pyarrow", compression="snappy"
    )
    return metrics, class_df


def freeze_final_outputs(materialization: dict, inference_df: pd.DataFrame, metrics_df: pd.DataFrame, class_df: pd.DataFrame) -> dict:
    artifact_paths = [
        ARRAY_MANIFEST, INFERENCE_REGISTER_CSV, METRICS_CSV,
        CLASS_BALANCE_CSV, ENSEMBLE_PREDICTIONS_PARQUET,
    ]
    for model_id in MODEL_ORDER:
        artifact_paths.append(ENSEMBLE_ROOT / f"{model_id}_five_seed_ensemble.npz")
        for seed in EXPECTED_SEEDS:
            wm = worker_manifest_path(model_id, seed)
            artifact_paths.append(wm)
            artifact_paths.append(Path(load_json(wm)["prediction_artifact"]["path"]))

    rows = []
    for p in artifact_paths:
        rec = artifact_record(p)
        rows.append({"file_name": p.name, **rec})
    artifact_df = pd.DataFrame(rows).drop_duplicates(subset=["path"])
    write_csv_atomic(ARTIFACT_REGISTER_CSV, artifact_df)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "STEP36_ONE_TIME_FINAL_TEST_EVALUATION_COMPLETE",
        "completed_at_utc": utc_now(),
        "preaccess_freeze": {
            "path": str(PREACCESS_FREEZE),
            "sha256": sha256_file(PREACCESS_FREEZE),
            "source_sha256": sha256_file(SCRIPT_FILE),
        },
        "upstream": {
            "step14_manifest_sha256": EXPECTED_STEP14_MANIFEST_SHA,
            "step32_manifest_sha256": EXPECTED_STEP32_MANIFEST_SHA,
            "step34_manifest_sha256": EXPECTED_STEP34_MANIFEST_SHA,
            "step34_global_fingerprint": EXPECTED_STEP34_GLOBAL_FP,
            "step35_manifest_sha256": EXPECTED_STEP35_MANIFEST_SHA,
            "step35_contract_fingerprint": EXPECTED_STEP35_CONTRACT_FP,
        },
        "one_time_test_access": {
            "raw_test_path": materialization["raw_test"]["path"],
            "raw_test_sha256": materialization["raw_test"]["observed_sha256"],
            "raw_test_filesystem_content_open_count": 1,
            "raw_test_reopened_after_materialization": False,
            "class_balance_seen_before_step35_freeze": False,
            "metrics_computed_before_step35_freeze": False,
        },
        "materialization": {
            "rows": EXPECTED_COMMON_ROWS,
            "dates": EXPECTED_COMMON_DATES,
            "stocks": EXPECTED_STOCKS,
            "lookback": 60,
            "features": 63,
            "predictor_scaler_refit_on_test": False,
            "regime_threshold_refit_on_test": False,
            "causal_development_history_used": True,
        },
        "inference": {
            "architectures": EXPECTED_ARCHITECTURES,
            "seed_models": EXPECTED_FITS,
            "seeds": list(EXPECTED_SEEDS),
            "probability_ensemble": "ARITHMETIC_MEAN_OF_FIVE_FIXED_SEED_PROBABILITIES",
            "best_seed_selection": False,
            "architecture_selection": False,
            "post_test_model_promotion": False,
            "threshold_reoptimization": False,
            "verified_inference_rows": int(len(inference_df)),
        },
        "scoring": {
            "metric_rows": int(len(metrics_df)),
            "class_balance_rows": int(len(class_df)),
            "metrics": ["balanced_accuracy", "roc_auc", "mcc", "pr_auc", "brier_score", "sensitivity", "specificity"],
            "primary_inferential_testing_performed_here": False,
            "bootstrap_performed_here": False,
            "bh_fdr_performed_here": False,
            "next_step": "DEPENDENCE_AWARE_FINAL_TEST_INFERENCE_AND_FDR",
        },
        "artifacts": {
            row["relative_path"]: {"path": row["path"], "size_bytes": int(row["size_bytes"]), "sha256": row["sha256"]}
            for row in artifact_df.to_dict("records")
        },
    }
    manifest["global_step36_fingerprint"] = canonical_hash(manifest)
    atomic_json(FINAL_MANIFEST, manifest)
    final_sha = sha256_file(FINAL_MANIFEST)
    atomic_text(FINAL_MANIFEST_SHA, f"{final_sha}  {FINAL_MANIFEST.name}\n")
    write_access_marker(
        "FINAL_TEST_OPENED_MATERIALIZED_INFERRED_SCORED",
        raw_test_filesystem_open_count_completed=1,
        raw_test_reopen_prohibited=True,
        final_test_scored=True,
        final_manifest_path=str(FINAL_MANIFEST),
        final_manifest_sha256=final_sha,
        global_step36_fingerprint=manifest["global_step36_fingerprint"],
    )
    return load_json(FINAL_MANIFEST)


def verify_final_complete() -> dict:
    if not FINAL_MANIFEST.exists() or not FINAL_MANIFEST_SHA.exists():
        raise FileNotFoundError("Final Step-36 manifest/sidecar missing.")
    expected = FINAL_MANIFEST_SHA.read_text(encoding="utf-8").strip().split()[0]
    observed = sha256_file(FINAL_MANIFEST)
    require_equal(observed, expected, "Final Step-36 manifest sidecar")
    m = load_json(FINAL_MANIFEST)
    require_equal(m.get("status"), "STEP36_ONE_TIME_FINAL_TEST_EVALUATION_COMPLETE", "Final Step-36 status")
    return m


def execute() -> int:
    print("=" * 140)
    print(STEP_NAME)
    print("=" * 140)
    print("MODE                         : EXECUTE")
    print("Architecture selection       : PROHIBITED")
    print("Best-seed selection          : PROHIBITED")
    print("Threshold reoptimization     : PROHIBITED")
    print("Post-test model promotion    : PROHIBITED")
    print("-" * 140)

    if FINAL_MANIFEST.exists():
        m = verify_final_complete()
        print("Step 36 already complete and hash-valid.")
        print(f"Global Step-36 fingerprint: {m['global_step36_fingerprint']}")
        print(f"Final manifest SHA256     : {sha256_file(FINAL_MANIFEST)}")
        return 0

    verify_preaccess_freeze()
    upstream = verify_upstream_without_test_access()
    print("[1] PRE-ACCESS FREEZE / UPSTREAM CONTRACT: PASS")
    print("-" * 140)

    print("[2] ONE-TIME FINAL-TEST MATERIALIZATION")
    materialization = materialize_final_test(upstream)
    print(f"Materialized common rows : {materialization['evaluation_panel']['rows']:,}")
    print(f"Raw test open count      : {materialization['raw_test']['filesystem_content_open_count']}")
    print("PASS")
    print("-" * 140)

    print("[3] 40 FROZEN CHECKPOINT INFERENCES")
    inference_df = execute_all_workers()
    print(f"Verified inference rows  : {len(inference_df)}/40")
    print("PASS")
    print("-" * 140)

    print("[4] FIVE-SEED PROBABILITY ENSEMBLES + FROZEN-THRESHOLD SCORING")
    metrics_df, class_df = build_ensembles_and_score()
    print(f"Metric rows              : {len(metrics_df)}/24")
    print("Threshold tuning         : False")
    print("Architecture selection   : False")
    print("PASS")
    print("-" * 140)

    print("[5] FINAL STEP-36 FREEZE")
    final = freeze_final_outputs(materialization, inference_df, metrics_df, class_df)
    final = verify_final_complete()
    print("PASS")
    print("-" * 140)

    print("=" * 140)
    print("STEP 36 COMPLETE — ONE-TIME FINAL TEST CONSUMED AND SCORED")
    print("=" * 140)
    print(f"Final-test common rows        : {EXPECTED_COMMON_ROWS:,}")
    print(f"Final-test common dates       : {EXPECTED_COMMON_DATES:,}")
    print(f"Architectures                 : {EXPECTED_ARCHITECTURES}")
    print(f"Seed checkpoints              : {EXPECTED_FITS}")
    print(f"Frozen thresholds             : {EXPECTED_THRESHOLDS}")
    print("Best-seed selection           : NOT PERFORMED")
    print("Architecture promotion        : NOT PERFORMED")
    print("Threshold reoptimization      : NOT PERFORMED")
    print("Raw test filesystem open count: 1")
    print("Raw test reopen               : False")
    print("Bootstrap/FDR                 : NOT YET PERFORMED")
    print(f"Global Step-36 fingerprint   : {final['global_step36_fingerprint']}")
    print(f"Final manifest SHA256         : {sha256_file(FINAL_MANIFEST)}")
    print("=" * 140)
    print("NEXT GATE: dependence-aware final-test inference (moving-block bootstrap + BH-FDR).")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", type=str)
    parser.add_argument("--worker-seed", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.prepare_only:
            return prepare_only()
        if args.worker:
            if args.worker_model is None or args.worker_seed is None:
                raise RuntimeError("--worker requires --worker-model and --worker-seed.")
            return run_worker(args.worker_model, int(args.worker_seed))
        if args.execute:
            return execute()
        raise RuntimeError("No valid mode selected.")
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "failed_at_utc": utc_now(),
            "script": str(SCRIPT_FILE),
            "script_sha256": sha256_file(SCRIPT_FILE) if SCRIPT_FILE.exists() else None,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "access_marker_state": marker_state(),
            "array_manifest_exists": ARRAY_MANIFEST.exists(),
            "final_manifest_exists": FINAL_MANIFEST.exists(),
        }
        try:
            atomic_json(FAILURE_JSON, failure)
        except Exception:
            pass
        print("\n" + "=" * 140, file=sys.stderr)
        print("STEP 36 FAILURE", file=sys.stderr)
        print("=" * 140, file=sys.stderr)
        traceback.print_exc()
        print("=" * 140, file=sys.stderr)
        print("Do not delete the access marker, materialized arrays, or partial prediction artifacts.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
