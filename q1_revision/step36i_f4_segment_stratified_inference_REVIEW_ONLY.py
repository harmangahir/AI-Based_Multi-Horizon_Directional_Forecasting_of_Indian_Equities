#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F4: segment-aware statistical engine, SYNTHETIC-ONLY REVIEW BUILD.

Usage in WSL:
  python step36i_f4_segment_stratified_inference_REVIEW_ONLY.py --offline-preflight

This module implements a pure/in-memory proposed statistical analysis, but its
CLI NEVER LOADS the held-out test, real labels/predictions, or GPU checkpoints.
Only synthetic labels/probabilities are generated during preflight. It never
writes into the experiment; no production entrypoint or new test read exists.

The Step35 scientific contract remains untouched. The proposed 5,300-endpoint
cohort and segment-stratified resampling are POST-FREEZE amendments requiring
formal methodological approval. This code does not authorize them.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

# Deny any production-like CLI option before importing any module or probing files.
_DENY = {"--execute", "--production", "--worker", "--authorize", "--approve",
         "--read-test", "--score-test", "--run", "--resume", "--commit-real",
         "--materialize", "--final-test", "--enable", "--test-parquet"}
if _DENY.intersection(sys.argv[1:]):
    raise SystemExit("DENIED: F4 offline synthetic statistics cannot access/score the final test")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
MAN = EXPERIMENT / "manifests"
AUD = EXPERIMENT / "audits"
BUNDLE = HERE / "step36i_e2_source_review_bundle.zip"
F1 = HERE / "step36i_f1_amended_production_runner_REVIEW_ONLY.py"
F2 = HERE / "step36i_f2_independent_source_review_READ_ONLY.py"
F3 = HERE / "step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py"
STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"
ACCESS = MAN / "36_final_test_access_consumption_state.json"
CANDIDATE = AUD / "step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv"
THRESHOLDS = AUD / "35_final_test_threshold_contract.csv"
E4 = AUD / "step36i_e4_synthetic_artifact_transaction_draft/36i_e4_synthetic_transaction_DRAFT.json"
H = {
    "E2": "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851",
    "F1": "e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556",
    "F2": "ec6400f0f356b7a9888c443890b333f6fa6c6d80f6d29d700625eba4335f00bc",
    "F3": "b7c28f536a57653f10cc74155b38941b7e2e86b508089e5a3378bf9ff97b7539",
    "STEP35": "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
    "COHORT": "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
    "THRESHOLDS": "79dc80e0e3b18427cebd77742ff2d10b87aebcdd9ea1e459fed30ac980d0e24f",
    "E4": "30fbd08cecba4e65eb427358c6a24c9d28b984bdf889afc83668a6c9952ab925",
}
MODELS = ("DLINEAR", "NLINEAR", "PATCHTST", "TIMESNET", "ITRANSFORMER",
          "FREDFORMER", "FRNET", "MLF")
HORIZONS = (1, 3, 5)
SEEDS = (42, 43, 44, 45, 46)
N_DATE, N_STOCK, N = 265, 20, 5300
SEG_SIZES = (48, 217)
BLOCKS = (10, 20, 40)
BOOT = 2000
PRIMARY_L = 20
RNG_SEED = 20261013
ALPHA = 0.05
EXPECTED_ROWS = 24


def require(cond: bool, message: str) -> None:
    if not bool(cond):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked(path: Path, wanted: str, desc: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"Missing or unsafe {desc}: {path}")
    require(file_sha(path) == wanted, f"SHA256 drift: {desc}")


def verify_contract(contract: dict) -> None:
    dep = contract["dependence_aware_inference_contract"]
    expected = {
        "FDR_alpha": ALPHA,
        "bootstrap_rng_seed": RNG_SEED,
        "replications_per_block_length": BOOT,
        "primary_block_length_sessions": PRIMARY_L,
        "sensitivity_block_lengths_sessions": [10, 20, 40],
        "primary_multiple_testing_family": "all 8 architectures x 3 horizons = 24 balanced-accuracy above-chance tests at block length 20",
    }
    for key, val in expected.items():
        require(dep.get(key) == val, f"Step35 statistical contract mismatch: {key}")
    require(dep.get("primary_null_hypothesis") == "balanced_accuracy <= 0.50",
            "Primary null drift")
    require(dep.get("primary_alternative") == "balanced_accuracy > 0.50",
            "Primary alternative drift")
    require(dep.get("multiple_testing_method") == "Benjamini-Hochberg FDR",
            "Multiplicity contract drift")
    require(dep.get("resampling_axis") == "ordered unique reference dates",
            "Resampling axis drift")
    require(dep.get("cross_sectional_dependence_preserved") is True,
            "Cross-section contract drift")


def key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["Ticker", "Date", "market_session_id", "continuity_segment_id"]
    require(all(c in frame for c in cols), "Candidate columns missing")
    k = frame.loc[:, cols].copy()
    k["Date"] = pd.to_datetime(k.Date, errors="raise").dt.normalize()
    for c in cols[2:]:
        k[c] = pd.to_numeric(k[c], errors="raise").astype("int64")
    require(not k.isna().any().any() and not k.duplicated().any(), "Duplicate or null key")
    require(len(k) == N and k.Date.nunique() == N_DATE and k.Ticker.nunique() == N_STOCK,
            "Cohort size drift")
    require(k.reset_index(drop=True).equals(k.sort_values(["Date", "Ticker"]).reset_index(drop=True)),
            "Physical row order drift")
    date_info = k.groupby("Date", sort=True).agg(
        n=("Ticker", "nunique"), seg=("continuity_segment_id", "first"),
        minseg=("continuity_segment_id", "min"), maxseg=("continuity_segment_id", "max"),
        minsession=("market_session_id", "min"), maxsession=("market_session_id", "max")
    ).reset_index()
    require(len(date_info) == N_DATE and date_info.n.eq(N_STOCK).all(),
            "Unbalanced date/ticker cross-section")
    require((date_info.minseg == date_info.maxseg).all() and
            (date_info.minsession == date_info.maxsession).all(),
            "Mixed segment/session within one date")
    require(np.array_equal(date_info.seg.to_numpy(), np.r_[[0]*48, [1]*217]),
            "Frozen segment allocation drift")
    for seg, data in date_info.groupby("seg"):
        require(np.all(np.diff(data.minsession.to_numpy()) == 1),
                f"Reference sessions nonconsecutive within segment {seg}")
    require(str(date_info.Date.iloc[0].date()) == "2025-01-01" and
            str(date_info.Date.iloc[47].date()) == "2025-03-07" and
            str(date_info.Date.iloc[48].date()) == "2025-08-01" and
            str(date_info.Date.iloc[-1].date()) == "2026-06-22",
            "Candidate endpoint date boundary mismatch")
    return k.reset_index(drop=True)


def threshold_map(frame: pd.DataFrame) -> dict[tuple[str, int], float]:
    require({"model_id", "horizon_days", "selected_threshold"}.issubset(frame.columns),
            "Threshold schema drift")
    require(len(frame) == EXPECTED_ROWS and
            not frame[["model_id", "horizon_days"]].duplicated().any(),
            "Threshold cardinality/duplicates")
    result = {(str(r.model_id), int(r.horizon_days)): float(r.selected_threshold)
              for r in frame.itertuples(index=False)}
    require(set(result) == {(m, h) for m in MODELS for h in HORIZONS},
            "Frozen threshold keys changed")
    require(all(np.isfinite(t) and 0 <= t <= 1 for t in result.values()),
            "Invalid frozen thresholds")
    return result


def circular_segment_indices(length: int, block: int, rng: np.random.Generator, reps: int) -> np.ndarray:
    """Generate exactly `length` reselected dates/repetition in one segment.

    Circular block starts are sampled uniformly within a segment and wrap only
    inside that segment; independent blocks are truncated at total segment size.
    """
    require(length > 0 and 1 <= block <= length and reps > 0, "Bad block specification")
    n_blocks = (length + block - 1) // block
    starts = rng.integers(0, length, size=(reps, n_blocks), endpoint=False)
    positions = (starts[:, :, None] + np.arange(block)[None, None, :]) % length
    return positions.reshape(reps, -1)[:, :length]


def stratified_session_indices(block: int, seed: int = RNG_SEED,
                               reps: int = BOOT) -> tuple[np.ndarray, np.ndarray]:
    """Resample same date indices for all 20 stocks, eight models and three horizons."""
    require(block in BLOCKS, "Unapproved block length")
    # A separately reinitialized stream for each block-length sensitivity;
    # Step35 freezes the RNG seed, while the per-segment application is proposed.
    rng = np.random.default_rng(seed)
    seg0 = circular_segment_indices(SEG_SIZES[0], block, rng, reps)
    seg1 = circular_segment_indices(SEG_SIZES[1], block, rng, reps) + SEG_SIZES[0]
    out = np.concatenate((seg0, seg1), axis=1).astype(np.int32, copy=False)
    require(out.shape == (reps, N_DATE), "Resampling matrix shape drift")
    require(np.all((out[:, :48] >= 0) & (out[:, :48] < 48)) and
            np.all((out[:, 48:] >= 48) & (out[:, 48:] < N_DATE)),
            "Circular block crossed a continuity boundary")
    return out, np.r_[np.zeros(48, dtype=np.int8), np.ones(217, dtype=np.int8)]


def _ba(tn: np.ndarray, fp: np.ndarray, fn: np.ndarray, tp: np.ndarray) -> np.ndarray:
    pos = tp + fn
    neg = tn + fp
    require(np.all(pos > 0) and np.all(neg > 0),
            "Undefined bootstrap BA: single-class resample; stop for protocol review")
    return 0.5*(tp/pos + tn/neg)


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    require(p.shape == (EXPECTED_ROWS,) and np.isfinite(p).all() and
            ((p >= 0) & (p <= 1)).all(), "BH family must have 24 finite p-values")
    order = np.argsort(p, kind="stable")
    s = p[order]
    q_sorted = np.minimum.accumulate((s*len(s)/np.arange(1, len(s)+1))[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[order] = np.minimum(1.0, q_sorted)
    require(np.isfinite(q).all() and (q >= p-1e-14).all() and
            np.all(np.diff(q[order]) >= -1e-14), "BH numerical check failed")
    return q


def evaluate_in_memory(keys: pd.DataFrame, y: np.ndarray,
                       ensembles: dict[str, np.ndarray],
                       thresholds: dict[tuple[str, int], float],
                       reps: int = BOOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure, proposed dependence-aware engine. Does not read any files.

    Only call on approved real arrays in a separately reviewed production source.
    For F4 CLI, inputs are synthetic fixture arrays generated below.
    """
    key_frame(keys)  # fail closed on ordering/cohort shifts
    yy = np.asarray(y)
    require(yy.shape == (N, 3) and np.isin(yy, [0, 1]).all(),
            "Invalid aligned binary outcomes")
    require(set(ensembles) == set(MODELS), "Missing/extra model ensembles")
    require(set(thresholds) == {(m, h) for m in MODELS for h in HORIZONS},
            "Frozen threshold cardinality changed")
    require(reps == BOOT, "Replicate count deviates from Step35")
    y3 = yy.reshape(N_DATE, N_STOCK, len(HORIZONS)).astype(bool)
    # One 20-stock cluster = one date. Counts per date enable fast, exact BA
    # computations without ever resampling 5,300 individual stock rows.
    confusion = np.empty((N_DATE, len(MODELS), len(HORIZONS), 4), dtype=np.int16)
    for mi, model in enumerate(MODELS):
        p = np.asarray(ensembles[model], dtype=np.float64)
        require(p.shape == (N, 3) and np.isfinite(p).all() and
                np.all((0 <= p) & (p <= 1)), f"Invalid ensemble probabilities: {model}")
        p3 = p.reshape(N_DATE, N_STOCK, 3)
        for hi, horizon in enumerate(HORIZONS):
            t = thresholds[(model, horizon)]
            require(np.isfinite(t) and 0 <= t <= 1, "Threshold range violation")
            truth = y3[:, :, hi]
            predicted = p3[:, :, hi] >= t  # frozen >= operator
            confusion[:, mi, hi, 0] = np.sum(~truth & ~predicted, axis=1)  # TN
            confusion[:, mi, hi, 1] = np.sum(~truth & predicted, axis=1)   # FP
            confusion[:, mi, hi, 2] = np.sum(truth & ~predicted, axis=1)   # FN
            confusion[:, mi, hi, 3] = np.sum(truth & predicted, axis=1)    # TP
    require((confusion.sum(axis=-1) == N_STOCK).all(), "Date-cluster count failure")
    observed_cnt = confusion.sum(axis=0, dtype=np.int64)
    obs = _ba(observed_cnt[:, :, 0], observed_cnt[:, :, 1],
              observed_cnt[:, :, 2], observed_cnt[:, :, 3])
    rows = []
    pairwise = []
    for block in BLOCKS:
        idx, segment = stratified_session_indices(block, RNG_SEED, BOOT)
        require(np.array_equal(segment, np.r_[[0]*48, [1]*217]), "Segment vector drift")
        bs = np.empty((BOOT, len(MODELS), len(HORIZONS)), dtype=np.float64)
        for mi, model in enumerate(MODELS):
            for hi, horizon in enumerate(HORIZONS):
                dt = confusion[:, mi, hi, :]
                # Shape: B x 265 x 4 -> B x 4. One original date's 20 stocks
                # enters the replicate together, repeated if its date repeats.
                c = dt[idx].sum(axis=1, dtype=np.int64)
                bs[:, mi, hi] = _ba(c[:, 0], c[:, 1], c[:, 2], c[:, 3])
                observed = float(obs[mi, hi])
                distribution = bs[:, mi, hi]
                # Frozen centered-bootstrap one-sided p-value under BA <= 0.5.
                delta = observed - 0.5
                pval = float((1 + np.count_nonzero((distribution - observed) >= delta)) / (BOOT + 1))
                lo, up = np.quantile(distribution, [0.025, 0.975])
                rows.append({"model_id": model, "horizon_days": horizon,
                             "block_length": block, "n": N,
                             "segment_0_sessions": 48, "segment_1_sessions": 217,
                             "bootstrap_replicates": BOOT, "observed_BA": observed,
                             "ci_lower_percentile_95": float(lo),
                             "ci_upper_percentile_95": float(up),
                             "p_one_sided_centered": pval,
                             "q_BH_primary_only": np.nan,
                             "primary_test": block == PRIMARY_L,
                             "protocol_status": "POST_FREEZE_AMENDMENT_NOT_APPROVED"})
        if block == PRIMARY_L:
            for hi, h in enumerate(HORIZONS):
                for mi in range(len(MODELS)):
                    for mj in range(mi+1, len(MODELS)):
                        diffs = bs[:, mi, hi] - bs[:, mj, hi]
                        low, high = np.quantile(diffs, [0.025, 0.975])
                        pairwise.append({"horizon_days": h, "model_a": MODELS[mi],
                                         "model_b": MODELS[mj],
                                         "BA_difference_a_minus_b": float(obs[mi, hi]-obs[mj, hi]),
                                         "ci_lower_percentile_95": float(low),
                                         "ci_upper_percentile_95": float(high),
                                         "role": "EXPLORATORY_NO_SELECTION_OR_FDR_PROMOTION"})
    results = pd.DataFrame(rows)
    require(len(results) == 72 and len(pairwise) == 84,
            "Primary/sensitivity or pairwise table cardinality changed")
    primary = results.block_length.eq(PRIMARY_L)
    require(int(primary.sum()) == EXPECTED_ROWS, "Incorrect primary family")
    results.loc[primary, "q_BH_primary_only"] = bh_adjust(
        results.loc[primary, "p_one_sided_centered"].to_numpy())
    require(results.loc[~primary, "q_BH_primary_only"].isna().all(),
            "Sensitivity tests improperly FDR-promoted")
    return results, pd.DataFrame(pairwise)


def assert_raises(call, term: str) -> None:
    try:
        call()
    except (RuntimeError, ValueError) as exc:
        require(term.lower() in str(exc).lower(), f"Wrong fail-closed cause: {exc}")
    else:
        raise RuntimeError(f"Negative control was accepted: {term}")


def build_synthetic_fixture(keys: pd.DataFrame, thresholds: dict) -> tuple[np.ndarray, dict]:
    """Generate purely fictional labels and five-seed probabilities."""
    rng = np.random.default_rng(7142026)
    y = rng.binomial(1, p=np.array([0.53, 0.54, 0.55]), size=(N, 3)).astype(np.int8)
    # Mild date-common shocks intentionally induce cross-stock dependence.
    common = rng.normal(0, 0.28, size=(N_DATE, 1, 3))
    seed_probs = {}
    for mi, mid in enumerate(MODELS):
        for seed in SEEDS:
            logit = (y.reshape(N_DATE, N_STOCK, 3)*2-1) * (0.1+mi*0.06)
            logit = logit + common + rng.normal(0, 1.3, size=(N_DATE, N_STOCK, 3))
            prob = 1.0/(1.0+np.exp(-logit))
            seed_probs[(mid, seed)] = np.asarray(prob.reshape(N, 3), dtype=np.float32)
    require(len(seed_probs) == 40, "Fixture seed counts incorrect")
    return y, seed_probs


def offline_fixture(bundle: dict[str, bytes]) -> None:
    keys = key_frame(pd.read_csv(__import__('io').BytesIO(bundle["candidate_keys_DRAFT.csv"])))
    thr = threshold_map(pd.read_csv(__import__('io').BytesIO(bundle["frozen_thresholds.csv"])))
    y, seed_probs = build_synthetic_fixture(keys, thr)
    # Actual reviewed F1 scoring implementation, imported as data-free module:
    spec = importlib.util.spec_from_file_location("f1_offline_score_module", str(F1))
    require(spec is not None and spec.loader is not None, "Cannot load F1 module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ensembles, score = module.in_memory_five_seed_scoring(seed_probs, y, thr)
    require(len(score) == EXPECTED_ROWS and len(ensembles) == 8, "F1 scoring shape drift")
    results, pairwise = evaluate_in_memory(keys, y, ensembles, thr)
    obs = results[results.block_length == PRIMARY_L].set_index(["model_id", "horizon_days"])
    for r in score.itertuples(index=False):
        recorded = float(obs.loc[(r.model_id, int(r.horizon_days)), "observed_BA"])
        require(abs(recorded - float(r.balanced_accuracy)) <= 1e-12,
                f"Observed BA discrepancy with independently audited F1 scorer: {r.model_id}/{r.horizon_days}")
    require(len(results) == 72 and len(pairwise) == 84, "Wrong result shape")
    require(results.groupby("block_length").size().to_dict() == {10:24,20:24,40:24},
            "Incorrect per-length testing")
    require(results[results.primary_test].q_BH_primary_only.notna().all(),
            "Missing primary BH correction")
    # Simple BH oracle (not a self-fulfilling expectation from the function).
    oracle = np.r_[np.full(12, .01), np.full(12, .80)]
    q = bh_adjust(oracle)
    require(np.allclose(q[:12], .02) and np.allclose(q[12:], .80),
            "BH FDR independent numeric oracle failed")
    # Prove segment isolation, reproducibility, and primary sensitivity distinction.
    for block in BLOCKS:
        idx, _ = stratified_session_indices(block)
        idx2, _ = stratified_session_indices(block)
        require(np.array_equal(idx, idx2), "Bootstrap not deterministic")
        require(np.all(idx[:, :48] < 48) and np.all(idx[:, 48:] >= 48),
                "Bootstrap crossed segment")
    require(not np.array_equal(stratified_session_indices(10)[0],
                               stratified_session_indices(20)[0]),
            "Sensitivity block length ineffective")
    print("PASS — fixed 48/217-session allocation, same-date 20-stock clusters, no cross-segment blocks")
    print("PASS — 2,000 deterministic replicates at L=10,20,40; primary L=20")
    print("PASS — 72 synthetic inference rows; 24 primary BH-adjusted q-values; 84 exploratory pairwise CIs")
    print("PASS — 24 exact BA comparisons to audited F1 synthetic scoring; BH numerical oracle")
    print("[4] FAIL-CLOSED NEGATIVE CONTROLS")
    assert_raises(lambda: key_frame(keys.iloc[:-1]), "Cohort size")
    shuffled = keys.iloc[np.r_[1, 0, np.arange(2, N)]].reset_index(drop=True)
    assert_raises(lambda: key_frame(shuffled), "Physical row order")
    corrupt_segment = keys.copy()
    corrupt_segment.loc[0, "continuity_segment_id"] = 1
    assert_raises(lambda: key_frame(corrupt_segment), "Mixed segment")
    broken_thresholds = {**thr, (MODELS[0], HORIZONS[0]): float('nan')}
    assert_raises(lambda: evaluate_in_memory(keys, y, ensembles, broken_thresholds),
                  "Threshold range")
    assert_raises(lambda: evaluate_in_memory(keys, y, {k:v for k,v in ensembles.items() if k!=MODELS[0]}, thr),
                  "Missing/extra")
    broken = y.copy(); broken[:, 0] = 1
    assert_raises(lambda: evaluate_in_memory(keys, broken, ensembles, thr), "single-class")
    assert_raises(lambda: bh_adjust(np.array([.05])), "24 finite")
    assert_raises(lambda: evaluate_in_memory(keys, y, ensembles, thr, reps=1999),
                  "Replicate count")
    print("PASS — bad cohort, misalignment, segment drift, thresholds, ensemble, single class and FDR controls rejected")


def no_real_gate() -> None:
    raise PermissionError("DENIED — F4 has no real-test loader or approved execution route")

open_test = no_real_gate
run_real_inference = no_real_gate
commit_real_results = no_real_gate


def preflight(real_project: bool) -> None:
    print("=" * 108)
    print("STEP 36I-F4 — POST-FREEZE SEGMENT-STRATIFIED BOOTSTRAP / FDR: OFFLINE REVIEW")
    print("=" * 108)
    print("[1] FROZEN F1/F2/F3/E2 SOURCES AND ORIGINAL STEP35 STATISTICAL CONTRACT")
    for label, path in (("F1", F1), ("F2", F2), ("F3", F3), ("E2", BUNDLE)):
        checked(path, H[label], label)
    with zipfile.ZipFile(BUNDLE, "r") as archive:
        require(archive.testzip() is None, "Damaged E2 archive")
        archive_keys = archive.namelist()
        require(len(archive_keys) == 18 and len(set(archive_keys)) == 18, "E2 inventory drift")
        b = {name: archive.read(name) for name in archive_keys}
    require(hashlib.sha256(b["candidate_keys_DRAFT.csv"]).hexdigest() == H["COHORT"],
            "Frozen candidate key SHA drift")
    require(hashlib.sha256(b["frozen_thresholds.csv"]).hexdigest() == H["THRESHOLDS"],
            "Frozen threshold SHA drift")
    require(hashlib.sha256(b["step35_contract.json"]).hexdigest() == H["STEP35"],
            "Step35 bundle SHA drift")
    verify_contract(json.loads(b["step35_contract.json"]))
    amendment = json.loads(b["amendment_review_DRAFT.json"])
    protocol = json.loads(b["protocol_deviation_DRAFT.json"])
    require(amendment.get("approval_granted") is False and
            amendment.get("third_raw_test_read_authorized") is False and
            protocol.get("status") == "DRAFT_NOT_APPROVED", "Review approval drift")
    print("PASS — frozen source fingerprints and prespecified BA > 0.5, B=2000, L=20, BH-family=24")
    if real_project:
        checked(STEP35, H["STEP35"], "WSL Step35")
        checked(CANDIDATE, H["COHORT"], "WSL candidate ledger")
        checked(THRESHOLDS, H["THRESHOLDS"], "WSL development thresholds")
        checked(E4, H["E4"], "WSL E4 synthetic transaction")
        require(ACCESS.is_file() and not ACCESS.is_symlink(), "Historical access marker missing")
        before = file_sha(ACCESS)
        marker = json.loads(ACCESS.read_text("utf-8"))
        require(marker.get("state") == "RAW_TEST_REOPENED_FOR_TECHNICAL_RECOVERY" and
                marker.get("raw_test_filesystem_open_count_completed") == 2 and
                marker.get("final_test_scored") is False, "Historical access state drift")
        require(marker.get("step35_manifest_sha256") == H["STEP35"],
                "Frozen marker/contract link drift")
        print("PASS — WSL frozen artifacts, two historical reads, zero scored, no new approval")
    print("[2] IN-MEMORY STATISTICAL INTERFACE / SYNTHETIC F1 ENSEMBLES")
    print("[3] STRATIFIED RESAMPLING, BH FAMILY, SENSITIVITY AND EXPLORATORY COMPARISONS")
    offline_fixture(b)
    for gated in (open_test, run_real_inference, commit_real_results):
        try:
            gated()
        except PermissionError:
            pass
        else:
            raise RuntimeError("Real-data execution gate unexpectedly open")
    if real_project:
        require(file_sha(ACCESS) == before, "Historical access marker mutated")
        print("PASS — WSL historical access marker unchanged")
    print("=" * 108)
    print("STEP 36I-F4: PASS — SYNTHETIC STATISTICAL INTEGRATION ONLY")
    print("Real test opened: FALSE | GPU: FALSE | Real scored: FALSE")
    print("Original Step35 changed: FALSE | Amendment approved: FALSE | Third read authorized: FALSE")
    print("Production CLI: DENIED | Next: independent statistical review and remaining access/transaction integration")
    print("=" * 108)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--offline-preflight", action="store_true")
    group.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    preflight(real_project=bool(args.offline_preflight))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STEP 36I-F4 FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
