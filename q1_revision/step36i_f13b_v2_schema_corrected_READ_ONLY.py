#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F13B v2 — Independently reconciled Step08/Step10 session identifiers.

USAGE: python step36i_f13b_v2_schema_corrected_READ_ONLY.py --offline-preflight

Does not read the sealed final-test parquet. Uses ONLY ticker/date/session/segment
columns from Step08 feature-panel Parquets, Step10 signed membership, and Step12
signed development history. Does not read a predictor value, target, regime,
checkpoint, test probability, or test metric. No files are written by this script.

Important: this establishes deterministic sequence-eligibility under the frozen
60-session same-ticker/same-segment rule. It is NOT scientific approval of the
post-freeze cohort or permission for a third sealed-test read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

# Deny active/production modes before opening anything.
DENIED = {"--execute", "--run", "--production", "--read-test", "--score-test",
          "--approve", "--authorize", "--commit", "--worker", "--resume",
          "--write", "--final-test", "--materialize", "--enable"}
if any(arg in DENIED for arg in sys.argv[1:]):
    raise SystemExit("DENIED: F13B is a pre-seal structural, read-only review")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
FEATURE = EXP / "feature_data"
MAN = EXP / "manifests"
AUD = EXP / "audits"

STEP08_FULL = FEATURE / "step08_merged_63_features/08_merged_63_feature_panel.parquet"
STEP08_COMPLETE = FEATURE / "step08_merged_63_features/08_merged_63_feature_panel_complete.parquet"
STEP08_MAN = MAN / "08_63_feature_panel_manifest.json"
STEP08_MAN_SIG = MAN / "08_63_feature_panel_manifest.sha256"
STEP06_WARMUP = AUD / "06_stock_feature_warmup_audit.csv"
CLEAN_POLICY = MAN / "04_cleaning_policy.json"
STEP10_LEDGER = FEATURE / "step10_chronological_splits/10_split_membership.parquet"
DEV_HISTORY = FEATURE / "step12_volatility_regimes/final_development/12_final_development_scaled_regimes.parquet"
CANDIDATE = AUD / "step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv"
STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"
ACCESS = MAN / "36_final_test_access_consumption_state.json"

SHA = {
    "membership": "82772196c92e88f1926a75a70c42f5f19119d7245d93b8364455fd28642ee121",
    "dev_history": "c166253e781031095c9c90fc864973e2b12e5fe2b8950d81e7fda0f9c2eba9bd",
    "candidate": "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
    "step35": "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
}
KEYS = ["Ticker", "Date", "market_session_id", "continuity_segment_id"]
STEP08_KEYS = ["Ticker", "Date", "session_number", "continuity_segment_id"]
IDENTITY_KEYS = ["Ticker", "Date", "continuity_segment_id"]
LOOKBACK = 60


def require(ok: bool, message: str):
    if not bool(ok):
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_file(path: Path):
    require(path.is_file() and not path.is_symlink(), f"Missing or symlinked input: {path}")


def checked(path: Path, expected: str, label: str):
    safe_file(path)
    require(sha256(path) == expected, f"SHA-256 differs for {label}: {path}")


def normalize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    require(set(KEYS).issubset(df.columns), f"{name} missing structural columns")
    x = df.loc[:, KEYS].copy()
    x["Ticker"] = x["Ticker"].astype(str)
    x["Date"] = pd.to_datetime(x["Date"], errors="raise").dt.normalize()
    for c in KEYS[2:]:
        x[c] = pd.to_numeric(x[c], errors="raise").astype("int64")
    require(not x.isna().any().any(), f"{name} null structural key")
    require(not x.duplicated(["Ticker", "Date"]).any(), f"{name} duplicate ticker/date")
    require(not x.duplicated(KEYS).any(), f"{name} duplicate structural key")
    return x.sort_values(["Date", "Ticker"], kind="stable").reset_index(drop=True)


def read_structural_parquet(path: Path, name: str) -> pd.DataFrame:
    safe_file(path)
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    require(set(KEYS).issubset(schema.names), f"{name} missing keys in Parquet schema")
    # Only key columns leave the on-disk parquet reader; no predictor/target values.
    table = pq.read_table(path, columns=KEYS, use_threads=False)
    require(table.column_names == KEYS, "Unexpected read columns")
    return normalize(table.to_pandas(), name)


def normalize_step08(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Retain Step08's *native* session_number; never relabel it blindly."""
    require(set(STEP08_KEYS).issubset(df.columns), f"{name} missing native structural columns")
    x = df.loc[:, STEP08_KEYS].copy()
    x["Ticker"] = x["Ticker"].astype(str)
    x["Date"] = pd.to_datetime(x["Date"], errors="raise").dt.normalize()
    for c in ("session_number", "continuity_segment_id"):
        x[c] = pd.to_numeric(x[c], errors="raise").astype("int64")
    require(not x.isna().any().any(), f"{name} null structural field")
    require(not x.duplicated(["Ticker", "Date"]).any(), f"{name} duplicate ticker/date")
    require(not x.duplicated(STEP08_KEYS).any(), f"{name} duplicate native structural key")
    return x.sort_values(["Date", "Ticker"], kind="stable").reset_index(drop=True)


def read_step08_structural(path: Path, name: str) -> pd.DataFrame:
    """Read only the four Step08 identity fields, never feature values."""
    safe_file(path)
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    require(set(STEP08_KEYS).issubset(schema.names), f"{name}: missing Step08 native keys")
    table = pq.read_table(path, columns=STEP08_KEYS, use_threads=False)
    require(table.column_names == STEP08_KEYS, f"{name}: unexpected projected columns")
    return normalize_step08(table.to_pandas(), name)


def reconcile_step08_step10(complete: pd.DataFrame, ledger: pd.DataFrame):
    """Authenticate the date/stock/segment join and independent session numbering.

    Step08 uses session_number, while Step10 uses market_session_id. Require
    a per-segment affine offset, never presume these column values equal.
    """
    require(len(complete) == len(ledger), "Step08/Step10 row count differs")
    require(complete[IDENTITY_KEYS].equals(ledger[IDENTITY_KEYS]),
            "Step08 and Step10 ticker/date/segment identities differ")
    offsets = {}
    for segment in sorted(complete["continuity_segment_id"].unique()):
        a = complete.loc[complete["continuity_segment_id"].eq(segment)]
        b = ledger.loc[ledger["continuity_segment_id"].eq(segment)]
        require(len(a) == len(b), f"segment {segment} crosswalk count drift")
        require(a[IDENTITY_KEYS].reset_index(drop=True).equals(
                b[IDENTITY_KEYS].reset_index(drop=True)),
                f"segment {segment} crosswalk identity mismatch")
        require((a.groupby("Date")["session_number"].nunique() == 1).all(),
                f"segment {segment} inconsistent Step08 IDs across stocks")
        require((b.groupby("Date")["market_session_id"].nunique() == 1).all(),
                f"segment {segment} inconsistent Step10 IDs across stocks")
        sa = a.drop_duplicates("Date").sort_values("Date")["session_number"].to_numpy(dtype=np.int64)
        sb = b.drop_duplicates("Date").sort_values("Date")["market_session_id"].to_numpy(dtype=np.int64)
        require(len(sa) == len(sb) and len(sa) > 0, "Empty/misaligned crosswalk dates")
        require(np.all(np.diff(sa) == 1) and np.all(np.diff(sb) == 1),
                f"segment {segment}: either native or membership session numbering has a gap")
        offset = sb - sa
        require(np.all(offset == offset[0]),
                f"segment {segment}: native/ledger session IDs lack a fixed mapping")
        offsets[int(segment)] = int(offset[0])
    require(set(offsets) == {0, 1}, "Unexpected Step08/Step10 continuity segment map")
    return offsets


def same_keys(a: pd.DataFrame, b: pd.DataFrame, name: str):
    require(a[KEYS].equals(b[KEYS]), f"Structural key mismatch: {name}")


def date_clusters(x: pd.DataFrame, name: str, session_column: str = "market_session_id"):
    require(x["Ticker"].nunique() == 20, f"{name}: not 20 stocks")
    counts = x.groupby("Date", sort=True)["Ticker"].nunique()
    require(len(counts) * 20 == len(x) and counts.eq(20).all(),
            f"{name}: unbalanced or missing stock-date clusters")
    require((x.groupby("Date")[session_column].nunique() == 1).all(),
            f"{name}: inconsistent session IDs")
    require((x.groupby("Date")["continuity_segment_id"].nunique() == 1).all(),
            f"{name}: inconsistent segment IDs")


def check_manifest_signature():
    safe_file(STEP08_MAN)
    safe_file(STEP08_MAN_SIG)
    contents = STEP08_MAN_SIG.read_text(encoding="utf-8").strip()
    match = re.search(r"\b[a-fA-F0-9]{64}\b", contents)
    require(match is not None, "Step08 manifest signature malformed")
    checked(STEP08_MAN, match.group(0).lower(), "Step08 frozen manifest")
    manifest = json.loads(STEP08_MAN.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict), "Step08 manifest is not a dictionary")
    return sha256(STEP08_MAN)


def verify_warmup(full: pd.DataFrame, complete: pd.DataFrame):
    # Independent of any post-freeze candidate key list.
    require(len(full) == 51640 and full["Date"].nunique() == 2582,
            "Step08 full panel 51,640/2,582 drift")
    require(len(complete) == 50320 and complete["Date"].nunique() == 2516,
            "Step08 complete panel 50,320/2,516 drift")
    date_clusters(full, "Step08 full", session_column="session_number")
    date_clusters(complete, "Step08 complete", session_column="session_number")
    joined = full.merge(complete.assign(_is_complete=1), on=STEP08_KEYS, how="left", validate="one_to_one")
    require(int(joined["_is_complete"].notna().sum()) == 50320,
            "Step08 complete is not the exact full-panel subset")
    missing = joined.loc[joined["_is_complete"].isna(), STEP08_KEYS]
    require(len(missing) == 1320 and missing["Date"].nunique() == 66,
            "Unexpected 66-session feature warm-up loss")
    info = {}
    for segment, fs in full.groupby("continuity_segment_id", sort=True):
        cs = complete[complete["continuity_segment_id"] == segment]
        require(len(cs) > 0, f"No complete features for segment {segment}")
        dates_f = fs.drop_duplicates("Date").sort_values("Date")
        dates_c = cs.drop_duplicates("Date").sort_values("Date")
        all_dates = list(dates_f["Date"])
        valid_dates = list(dates_c["Date"])
        require(len(all_dates) - len(valid_dates) == 33,
                f"Segment {segment} must have 33 discarded feature warm-up dates")
        require(all_dates[33:] == valid_dates,
                f"Segment {segment} has missing complete features after warm-up")
        sess = dates_c["session_number"].to_numpy(dtype=np.int64)
        require(np.all(np.diff(sess) == 1),
                f"Segment {segment} complete sessions have an internal discontinuity")
        info[int(segment)] = {"raw_sessions": len(all_dates), "complete_sessions": len(valid_dates),
                              "first_full": str(all_dates[0].date()),
                              "first_complete": str(valid_dates[0].date()),
                              "warmup_dates": 33}
    require(set(info) == {0, 1}, f"Unexpected continuity segments: {list(info)}")
    require(info[1]["first_full"] == "2025-03-19",
            "Second segment start drift")
    require(info[1]["first_complete"] == "2025-05-12",
            "Second segment first complete date drift")
    return info


def assess_sequences(history: pd.DataFrame, original: pd.DataFrame, lookback: int):
    """Independent date/segment/session eligibility, no source model functions."""
    groups = {}
    for (ticker, seg), g in history.groupby(["Ticker", "continuity_segment_id"], sort=True):
        g = g.sort_values(["Date", "market_session_id"]).reset_index(drop=True)
        require(not g.duplicated("Date").any(), "Duplicate history date")
        groups[(str(ticker), int(seg))] = g
    rows = []
    for row in original.itertuples(index=False):
        ticker = str(row.Ticker)
        segment = int(row.continuity_segment_id)
        ref = pd.Timestamp(row.Date)
        sess = int(row.market_session_id)
        key = (ticker, segment)
        require(key in groups, f"Missing history ticker/segment {key}")
        g = groups[key]
        # Independent lookup: exact endpoint key must occur exactly once.
        pos = np.flatnonzero((g["Date"].to_numpy() == ref.to_datetime64()) &
                             (g["market_session_id"].to_numpy() == sess))
        require(len(pos) == 1, f"Missing/ambiguous endpoint in history {ticker} {ref.date()}")
        end = int(pos[0])
        start = end - lookback + 1
        if start < 0:
            eligible = False
            cause = "fewer_than_60_same_segment_complete_rows"
        else:
            window = g.iloc[start:end+1]
            sid = window["market_session_id"].to_numpy(dtype=np.int64)
            dates = window["Date"].to_numpy()
            sequential = bool(np.all(np.diff(sid) == 1) and len(sid) == lookback)
            chronological = bool((dates[-1] == ref.to_datetime64()) and
                                 np.all(dates <= ref.to_datetime64()))
            require(sequential and chronological,
                    f"Unexpected nonconsecutive or future history {ticker} {ref.date()}")
            eligible = True
            cause = "eligible"
        rows.append((ticker, ref, sess, segment, end+1, eligible, cause))
    return pd.DataFrame(rows, columns=["Ticker", "Date", "market_session_id",
                                       "continuity_segment_id", "history_rows_available",
                                       "eligible", "cause"])


def synthetic_unit_test():
    # The schema fix must accept *different* but consistently mapped identifiers.
    native = pd.DataFrame(
        [(s, pd.Timestamp("2025-01-01") + pd.Timedelta(days=d), d + 100, 0)
         for d in range(5) for s in ("AAA", "BBB")], columns=STEP08_KEYS
    )
    # The native Step08 parser must preserve its own session identifier.
    require("market_session_id" not in normalize_step08(native, "synthetic native"),
            "Step08 native key improperly renamed")
    # Two-segment crosswalk with distinct per-segment offsets.
    native2 = pd.concat([native, native.assign(
        Date=lambda z: z["Date"] + pd.Timedelta(days=60),
        session_number=lambda z: z["session_number"] + 500,
        continuity_segment_id=1)], ignore_index=True)
    member2 = native2.rename(columns={"session_number": "market_session_id"}).copy()
    member2.loc[member2.continuity_segment_id.eq(0), "market_session_id"] += 250
    member2.loc[member2.continuity_segment_id.eq(1), "market_session_id"] += 700
    mapping = reconcile_step08_step10(normalize_step08(native2, "synthetic native"),
                                        normalize(member2, "synthetic membership"))
    require(mapping == {0: 250, 1: 700}, "Synthetic per-segment mapping incorrect")
    bad = normalize(member2, "synthetic membership")
    bad.loc[(bad.continuity_segment_id.eq(1)) &
            (bad.Date.eq(pd.Timestamp("2025-03-04"))), "market_session_id"] += 1
    try:
        reconcile_step08_step10(normalize_step08(native2, "synthetic native"), bad)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("FAIL OPEN: non-constant ID mapping accepted")

    # 2 independent ticker clusters, a new segment that starts with only one row.
    rows = []
    for name in ("AAA", "BBB"):
        for t in range(8):
            rows.append((name, pd.Timestamp("2025-01-01") + pd.Timedelta(days=t), t, 0))
        for t in range(6):
            rows.append((name, pd.Timestamp("2025-03-19") + pd.Timedelta(days=t), 100+t, 1))
    hist = pd.DataFrame(rows, columns=KEYS)
    orig = hist.loc[hist["continuity_segment_id"].eq(1)].copy()
    r = assess_sequences(hist, orig, lookback=4)
    assert len(r) == 12 and int(r.eligible.sum()) == 6
    assert set(r.loc[~r.eligible, "cause"]) == {"fewer_than_60_same_segment_complete_rows"}
    changed = hist.drop(hist[(hist.Ticker == "AAA") &
                             (hist.market_session_id == 103)].index).reset_index(drop=True)
    try:
        assess_sequences(changed, orig, lookback=4)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("FAIL OPEN: missing endpoint not rejected")
    # Missing interior session must not be mistaken for a consecutive window.
    interior = hist.drop(hist[(hist.Ticker == "AAA") &
                              (hist.market_session_id == 101)].index).reset_index(drop=True)
    try:
        assess_sequences(interior, orig.loc[orig.market_session_id != 101], lookback=4)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("FAIL OPEN: interior session gap accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    require(args.offline_preflight != args.self_test, "Choose exactly one review mode")
    synthetic_unit_test()
    if args.self_test:
        print("PASS — portable synthetic eligibility/negative controls; no WSL access")
        return

    print("="*102)
    print("STEP 36I-F13B v2 — SCHEMA-CORRECTED CAUSAL-HISTORY ELIGIBILITY PROOF")
    print("="*102)
    print("[1] FROZEN INPUT AUTHENTICATION AND RAW-TEST BOUNDARY")
    require(ROOT.name == "q1_revision", "Place this file directly inside q1_revision")
    checked(STEP10_LEDGER, SHA["membership"], "frozen membership")
    checked(DEV_HISTORY, SHA["dev_history"], "frozen final-development history")
    checked(CANDIDATE, SHA["candidate"], "frozen candidate keys")
    checked(STEP35, SHA["step35"], "frozen scientific contract")
    manifest_sha = check_manifest_signature()
    safe_file(CLEAN_POLICY)
    safe_file(STEP06_WARMUP)
    policy = json.loads(CLEAN_POLICY.read_text(encoding="utf-8"))
    p = policy.get("date_wide_anomaly_policy", {})
    require(p.get("valid_nifty_sessions_excluded_by_rule") == ["2025-03-18"],
            "Frozen anomalous-session exclusion differs")
    access_before = ACCESS.read_bytes()
    access = json.loads(access_before)
    require(access.get("raw_test_filesystem_open_count_completed") == 2 and
            access.get("final_test_scored") is False,
            "Unexpected historical raw-test access/scoring state")
    print("PASS — Step10, Step12, candidate and Step35 hashes; Step08 signed manifest")
    print("PASS — excluded 2025-03-18; access ledger 2 reads/0 scored; no sealed-test parquet")
    print(f"Step08 manifest SHA256: {manifest_sha}")

    print("[2] STEP08 FULL-VS-COMPLETE WARM-UP AND CONTINUITY RECONCILIATION")
    full = read_step08_structural(STEP08_FULL, "Step08 full")
    complete = read_step08_structural(STEP08_COMPLETE, "Step08 complete")
    warmup = verify_warmup(full, complete)
    for segment, v in warmup.items():
        print(f"PASS — segment {segment}: raw={v['raw_sessions']} dates, complete={v['complete_sessions']} "
              f"dates, first raw={v['first_full']}, first complete={v['first_complete']}, "
              f"33 warm-up dates")
    print("PASS — exactly 66 incomplete dates, all at segment starts; no later internal missingness")

    print("[3] DEVELOPMENT HISTORY + NON-LABEL TEST STRUCTURE + ORIGINAL ELIGIBILITY")
    dev = read_structural_parquet(DEV_HISTORY, "Step12 signed final-development history")
    require(len(dev) == 43640 and dev.Date.max() == pd.Timestamp("2024-12-31"),
            "Final development history size or end-date drift")
    import pyarrow.parquet as pq
    schema = pq.read_schema(STEP10_LEDGER)
    flag_name = "outer_final_test_common_multitask"
    require(set(KEYS + [flag_name]).issubset(schema.names),
            "Step10 frozen membership schema drift")
    led_tab = pq.read_table(STEP10_LEDGER, columns=KEYS+[flag_name], use_threads=False)
    raw_ledger = led_tab.to_pandas()
    require(not raw_ledger[flag_name].isna().any(), "Null test membership flag")
    ledger = normalize(raw_ledger, "Step10 frozen membership")
    require(len(ledger) == 50320, "Step10 ledger row count differs from Step08 complete")
    offsets = reconcile_step08_step10(complete, ledger)
    print("PASS — Step08 session_number / Step10 market_session_id date-key crosswalk:", offsets)
    # Recreate the flag after normalization using an exact structural merge.
    flags = raw_ledger[KEYS+[flag_name]].copy()
    flags["Date"] = pd.to_datetime(flags["Date"], errors="raise").dt.normalize()
    flags["Ticker"] = flags["Ticker"].astype(str)
    for c in KEYS[2:]:
        flags[c] = pd.to_numeric(flags[c], errors="raise").astype("int64")
    flags = flags.sort_values(["Date", "Ticker"], kind="stable").reset_index(drop=True)
    require(ledger[KEYS].equals(flags[KEYS]), "Membership flag ordering ambiguous")
    flag = flags[flag_name].astype(bool)
    original = ledger.loc[flag, KEYS].reset_index(drop=True)
    require(len(original) == 6480 and original.Date.nunique() == 324,
            "Frozen 6,480/324 original final-test endpoints differ")
    date_clusters(original, "Step10 original eligible cohort")
    dev_join = ledger.merge(dev.assign(_dev=1), on=KEYS, how="left", validate="one_to_one")
    require(dev_join["_dev"].notna().sum() == 43640,
            "Development panel is not exactly in Step10 history")
    test_struct = dev_join.loc[dev_join["_dev"].isna(), KEYS].copy()
    require(len(test_struct) == 6680 and test_struct.Date.nunique() == 334,
            "Non-label outer-test structural history should be 6,680/334")
    require(test_struct.Date.min() == pd.Timestamp("2025-01-01") and
            test_struct.Date.max() == pd.Timestamp("2026-06-30"),
            "Test structural history boundaries drift")
    date_clusters(test_struct, "outer-test structural-history rows")
    print("PASS — 43,640 signed development rows + 6,680 non-label test keys = 50,320 complete keys")
    print("PASS — exact ticker/date/segment equality: Step08 complete = Step10 membership")
    print("PASS — Step10 full structural membership = development/test union")

    print("[4] INDEPENDENT PER-ENDPOINT 60-SESSION CAUSAL WINDOW CHECK")
    outcome = assess_sequences(ledger, original, lookback=LOOKBACK)
    candidate = normalize(pd.read_csv(CANDIDATE), "signed candidate keys")
    date_clusters(candidate, "candidate cohort")
    require(len(candidate) == 5300 and candidate.Date.nunique() == 265,
            "Proposed 5,300/265 candidate size drift")
    kept = outcome.loc[outcome.eligible, KEYS].sort_values(["Date", "Ticker"]).reset_index(drop=True)
    same_keys(kept, candidate, "independently eligible vs proposed cohort")
    removed = outcome.loc[~outcome.eligible].copy()
    require(len(removed) == 1180 and removed.Date.nunique() == 59,
            "Failure count not 1,180 endpoints / 59 dates")
    require(removed.cause.eq("fewer_than_60_same_segment_complete_rows").all(),
            "Some exclusions were not caused exclusively by insufficient history")
    require(set(removed.continuity_segment_id) == {1}, "Exclusions extend to another segment")
    require((removed.Date.min(), removed.Date.max()) ==
            (pd.Timestamp("2025-05-12"), pd.Timestamp("2025-07-31")),
            "Excluded date range changed")
    require((removed.groupby("Ticker").size() == 59).all() and removed.Ticker.nunique() == 20,
            "All ticker exclusions must be 59 each")
    require(removed.history_rows_available.min() == 1 and
            removed.history_rows_available.max() == 59,
            "Excluded history positions are not 1 through 59")
    first_ok_seg1 = kept.loc[kept.continuity_segment_id.eq(1), "Date"].min()
    require(first_ok_seg1 == pd.Timestamp("2025-08-01"),
            "First 60-history segment-1 endpoint unexpectedly shifted")
    seg_dates = kept.groupby("continuity_segment_id")["Date"].nunique().to_dict()
    require(seg_dates == {0:48,1:217}, "Retained segment allocation changed")
    print("PASS — all 6,480 original endpoints assessed by same-ticker/same-segment history")
    print("PASS — precisely 1,180 fail because complete history has only 1..59 sessions")
    print("PASS — all 20 stocks lose the same 59 dates (2025-05-12 through 2025-07-31)")
    print("PASS — 5,300/265 independently eligible endpoints equal signed candidate keys exactly")
    print("PASS — first segment-1 valid 60-session window ends 2025-08-01")

    print("[5] ORIGINAL ARTIFACT/ACCESS IMMUTABILITY AND SCIENTIFIC BOUNDARIES")
    require(ACCESS.read_bytes() == access_before, "FAIL: historical access marker changed")
    print("PASS — original Step35 and historical access marker remain unchanged")
    print("OPEN — exclusion-window regime/covariate selection-bias assessment")
    print("OPEN — segment-stratified inferential validity / scientific amendment decision")
    print("OPEN — independent sign-off and explicit third raw-test read authorization")
    print("="*102)
    print("STEP 36I-F13B v2 PASS — SCHEMA-CORRECTED CAUSAL-HISTORY EXCLUSION PROOF")
    print("Sealed final-test parquet opened: FALSE | Test predictors/labels read: FALSE")
    print("GPU: FALSE | Test scored: FALSE | Original artifacts modified: FALSE")
    print("Amendment approved: FALSE | Third access authorized: FALSE | Production: DENIED")
    print("="*102)


if __name__ == "__main__":
    main()
