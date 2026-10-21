#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F11 — independent frozen upstream provenance audit (read-only).

Allowed: --offline-preflight (on WSL), --self-test (portable E2 fixtures).

This audit checks non-test upstream files only: Step-14 frozen feature order and
stock IDs, the Step-11 fitted development scaler, the Step-12 frozen final-
development regime thresholds and scaled development history. The original
Step-35 contract and the frozen E2 review ZIP authenticate expected hashes.

DO NOT USE as a production launcher. This script cannot open the sealed test,
create an access journal, commit real outputs, grant amendment approval, or
authorize another raw-test read.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

# Fail before project discovery if a production-like action was requested.
DENIED_FLAGS = {"--run", "--execute", "--production", "--read-test", "--test-parquet",
                "--score-test", "--authorize", "--approve", "--resume", "--worker",
                "--commit-real", "--materialize", "--enable", "--final-test"}
if DENIED_FLAGS.intersection(sys.argv[1:]):
    raise SystemExit("DENIED: F11 only audits frozen NON-TEST upstream provenance")

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
MAN = EXP / "manifests"
FROZEN_WSL_EXP = Path("/home/harman/projects/AI-Based_Multi-Horizon_Directional_Forecasting_of_Indian_Equities/stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026")
ZIP = ROOT / "step36i_e2_source_review_bundle.zip"
F9 = ROOT / "step36i_f9_preloaded_production_assembly_REVIEW_ONLY.py"
F10 = ROOT / "step36i_f10_independent_assembly_review_READ_ONLY.py"
LEDGER = MAN / "36_final_test_access_consumption_state.json"
STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"
STEP14 = MAN / "14_model_array_manifest.json"

DIGESTS = {
    "zip": "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851",
    "step35": "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
    "f9": "8074cfe7cd1de9cf17609d4cef71aff27f910347da3ad86f3861a3cf7ad1373b",
    "f10": "d778fe150b3a06f85992b8077937a8f4509a9f2c741f0d913433443a0d07a686",
}


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def digest_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_file(path: Path, sha: str, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}: {path}")
    require(digest_of(path) == sha, f"Frozen SHA-256 mismatch: {label}")


def load_bundle() -> tuple[dict, dict, dict]:
    check_file(ZIP, DIGESTS["zip"], "E2 source ZIP")
    with zipfile.ZipFile(ZIP) as z:
        require(z.testzip() is None and len(z.namelist()) == 18 and
                len(set(z.namelist())) == 18, "E2 bundle changed")
        snap35 = json.loads(z.read("step35_contract.json"))
        snap14 = json.loads(z.read("model_interface.json"))
        preaccess = json.loads(z.read("step36_preaccess_freeze.json"))
    return snap35, snap14, preaccess


def verify_feature_map(snap35: dict, snap14: dict, live14: dict | None = None) -> tuple[list, dict]:
    expected = snap35["predictor_preprocessing_contract"]["ordered_features"]
    interface = snap14["ordered_63_features"]
    require(isinstance(expected, list) and len(expected) == 63 and
            len(set(expected)) == 63 and expected == interface,
            "63-feature order differs between Step35 and Step14 frozen interface")
    stock_map = snap14["stock_id_map"]
    require(isinstance(stock_map, dict) and len(stock_map) == 20 and
            set(int(v) for v in stock_map.values()) == set(range(20)),
            "Frozen stock_id_map is not a 20-stock bijection")
    if live14 is not None:
        require(live14.get("ordered_63_features") == expected,
                "LIVE Step14 feature ordering differs from signed reference")
        require({str(k): int(v) for k, v in live14.get("stock_id_map", {}).items()} ==
                {str(k): int(v) for k, v in stock_map.items()},
                "LIVE Step14 stock-ID mapping differs from signed reference")
    return expected, stock_map


def expected_non_test_paths(snap35: dict, preaccess: dict) -> dict:
    preproc = snap35["predictor_preprocessing_contract"]
    sequences = snap35["test_sequence_materialization_contract"]
    regimes = snap35["volatility_regime_contract"]
    state = preaccess["final_development_transform_state"]
    entries = {
        "development scaler": (preproc["scaler_path"], preproc["scaler_sha256"],
                               "11_final_development_minmax_scaler.joblib"),
        "development history": (sequences["final_development_history_panel"],
                                sequences["final_development_history_sha256"],
                                "12_final_development_scaled_regimes.parquet"),
        "regime thresholds": (regimes["threshold_artifact"],
                              regimes["threshold_artifact_sha256"],
                              "12_final_development_regime_thresholds.csv"),
    }
    aliases = {
        "development scaler": (state["predictor_scaler_path"], state["predictor_scaler_sha256"]),
        "development history": (state["development_history_path"], state["development_history_sha256"]),
        "regime thresholds": (state["regime_threshold_path"], state["regime_threshold_sha256"]),
    }
    for label, (raw, sha, filename) in entries.items():
        path = Path(raw)
        require(path.is_absolute() and path.name == filename and
                path.is_relative_to(FROZEN_WSL_EXP) and (raw, sha) == aliases[label],
                f"Expected {label} path/hash differs between frozen manifests")
    require(preproc["fit_on_final_test"] is False and
            preproc["refit_on_final_test"] is False and
            preproc["clip"] is False and
            preproc["transform_only"] is True,
            "Frozen scaler policy unexpectedly allows test fitting or clipping")
    require(sequences["lookback_sessions"] == 60 and
            sequences["same_continuity_segment_required"] is True and
            sequences["consecutive_market_session_ids_required"] is True,
            "Frozen causal lookback/continuity policy changed")
    return entries


def test_negative_controls(snap35: dict, snap14: dict, preaccess: dict) -> None:
    import copy
    feat = copy.deepcopy(snap14)
    feat["ordered_63_features"][0], feat["ordered_63_features"][1] = (
        feat["ordered_63_features"][1], feat["ordered_63_features"][0])
    try:
        verify_feature_map(snap35, feat)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Fail-open: swapped feature order accepted")
    IDs = copy.deepcopy(snap14)
    IDs["stock_id_map"][next(iter(IDs["stock_id_map"]))] = 999
    try:
        verify_feature_map(snap35, IDs)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Fail-open: modified stock-ID mapping accepted")
    mutated = copy.deepcopy(preaccess)
    mutated["final_development_transform_state"]["predictor_scaler_sha256"] = "0" * 64
    try:
        expected_non_test_paths(snap35, mutated)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Fail-open: scaler digest drift accepted")
    mutated = copy.deepcopy(preaccess)
    mutated["final_development_transform_state"]["development_history_path"] = (
        str(EXP / "feature_data/step10_chronological_splits/10_final_test_locked_panel.parquet"))
    try:
        expected_non_test_paths(snap35, mutated)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Fail-open: inappropriate test-path substitution accepted")


def audit(wsl: bool) -> None:
    print("=" * 108)
    print("STEP 36I-F11 — INDEPENDENT FROZEN UPSTREAM PROVENANCE AUDIT / NO TEST FILE ACCESS")
    print("=" * 108)
    print("[1] REVIEWED SOURCE FINGERPRINTS AND SIGNED NON-TEST INPUT CONTRACT")
    check_file(F9, DIGESTS["f9"], "F9 source")
    check_file(F10, DIGESTS["f10"], "F10 source")
    snap35, snap14, preaccess = load_bundle()
    features, stock_map = verify_feature_map(snap35, snap14)
    entries = expected_non_test_paths(snap35, preaccess)
    print("PASS — F9/F10 and E2 verified; 63 ordered features, 20 stock IDs, three frozen input digests")
    print("[2] FAIL-CLOSED NEGATIVE CONTROLS")
    test_negative_controls(snap35, snap14, preaccess)
    print("PASS — swapped features, stock-ID drift, scaler digest tamper and test-path substitution rejected")
    if not wsl:
        print("[3] PORTABLE MODE")
        print("PASS — frozen metadata verified; WSL development files were NOT accessed")
        print("STATUS: PASS PORTABLE SELF-TEST ONLY")
        return
    print("[3] LIVE FROZEN DEVELOPMENT ARTIFACT AUTHENTICATION — NON-TEST ONLY")
    require(EXP == FROZEN_WSL_EXP, "Unexpected WSL experiment directory")
    check_file(STEP35, DIGESTS["step35"], "Step35 original scientific contract")
    require(json.loads(STEP35.read_text("utf-8")) == snap35,
            "Live Step35 JSON differs from the independently archived E2 copy")
    m14_sha = snap35["upstream_freezes"]["step14_manifest_sha256"]
    check_file(STEP14, m14_sha, "Step14 original array manifest")
    live14 = json.loads(STEP14.read_text("utf-8"))
    verify_feature_map(snap35, snap14, live14)
    for label, (raw, sha, _) in entries.items():
        check_file(Path(raw), sha, label)
        print(f"PASS — {label}: frozen SHA-256 verified")
    print("[4] DEVELOPMENT-ONLY SCALER, REGIME AND HISTORY SEMANTICS")
    import joblib
    import numpy as np
    import pandas as pd
    scaler = joblib.load(entries["development scaler"][0])
    require(int(scaler.n_features_in_) == 63 and
            not bool(getattr(scaler, "clip", False)) and
            tuple(float(v) for v in scaler.feature_range) == (0.0, 1.0),
            "Final-development scaler width/range/clip changed")
    named = hasattr(scaler, "feature_names_in_")
    if named:
        require(list(scaler.feature_names_in_) == features,
                "Scaler feature_names_in_ order differs from Step35")
    for attribute in ("data_min_", "data_max_", "scale_", "min_"):
        values = np.asarray(getattr(scaler, attribute), dtype="float64")
        require(values.shape == (63,) and np.isfinite(values).all(),
                f"Invalid frozen scaler {attribute}")
    del scaler
    thr = pd.read_csv(entries["regime thresholds"][0])
    tcols = {"ticker", "low_volatility_threshold_q33", "high_volatility_threshold_q67"}
    require(tcols <= set(thr.columns) and len(thr) == 20 and
            not thr.ticker.duplicated().any() and set(thr.ticker) == set(stock_map),
            "Regime threshold rows/tickers differ from frozen stock universe")
    lo = thr["low_volatility_threshold_q33"].to_numpy(dtype="float64")
    hi = thr["high_volatility_threshold_q67"].to_numpy(dtype="float64")
    require(np.isfinite(lo).all() and np.isfinite(hi).all() and
            (lo <= hi).all(), "Invalid regime quantile ordering")
    # Only the frozen DEVELOPMENT HISTORY is read; NEVER scan or open a sealed test path.
    history = pd.read_parquet(entries["development history"][0],
                              columns=["Ticker", "Date", "market_session_id",
                                       "continuity_segment_id", *features])
    required = {"Ticker", "Date", "market_session_id", "continuity_segment_id", *features}
    require(required <= set(history.columns) and len(history) > 0,
            "Development history is missing frozen features/metadata")
    dates = pd.to_datetime(history["Date"], errors="raise")
    require(dates.notna().all() and dates.max() <= pd.Timestamp("2024-12-31") and
            set(history["Ticker"].astype(str)) == set(stock_map) and
            not history[["Ticker", "Date"]].duplicated().any(),
            "Frozen development history date/stock/duplicate integrity failed")
    require(np.isfinite(history[features].to_numpy(dtype="float64")).all(),
            "Frozen development history has non-finite predictors")
    print("PASS — 63-column scaler; fit-only semantics; 20 frozen regime thresholds")
    print(f"PASS — {len(history):,} development-history rows, last date {dates.max().date()}, 20 stocks")
    if not named:
        print("NOTE — scaler has no feature_names_in_; frozen fit artifact/order established by manifests")
    del history
    print("[5] ACCESS HISTORY AND PRODUCTION DENIAL")
    require(LEDGER.is_file() and not LEDGER.is_symlink(), "Historical access marker missing")
    before = digest_of(LEDGER)
    state = json.loads(LEDGER.read_text("utf-8"))
    require(int(state.get("raw_test_filesystem_open_count_completed", -1)) == 2 and
            state.get("final_test_scored") is False,
            "Historical final-test access state changed")
    require(digest_of(LEDGER) == before, "Historical access marker changed during F11")
    print("PASS — historical 2 reads/0 scorings; no additional access; ledger unchanged")
    print("=" * 108)
    print("STEP 36I-F11 PASS — FROZEN UPSTREAM PROVENANCE AUTHENTICATED, DEVELOPMENT DATA ONLY")
    print("Real raw test opened: FALSE | Real scoring: FALSE | GPU: FALSE")
    print("Amendment approved: FALSE | Third read authorized: FALSE | Production CLI: DENIED")
    print("NEXT: independent integration of approved loader/journal/atomic-commit; scientific sign-off")
    print("=" * 108)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--offline-preflight", action="store_true")
    args = parser.parse_args()
    audit(wsl=args.offline_preflight)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        print(f"STEP36I-F11 FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
