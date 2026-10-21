#!/usr/bin/env python3
"""Step 36I-D2b: offline-only, fail-closed amended-contract verification."""

from __future__ import annotations

import sys

# This draft has no production execution pathway.
DENIED = {"--execute", "--worker", "--prepare", "--materialize"}

if DENIED.intersection(sys.argv[1:]):
    raise SystemExit(
        "DENIED: Step 36I-D2b is an offline-only draft. "
        "No final-test access or production inference is authorized."
    )

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
EXP = (
    WORK.parent
    / "stock_q1_revision_project"
    / "experiments/q1_revision_v3_clean_2016_2026"
)
MAN = EXP / "manifests"
AUD = EXP / "audits"

SOURCE = WORK / (
    "step36_one_time_final_test_materialization_inference_scoring_v1.py"
)
LEDGER = (
    EXP / "feature_data/step10_chronological_splits"
    / "10_split_membership.parquet"
)
KEYS = (
    AUD / "step36i_protocol_review_draft"
    / "36i_candidate_sequence_eligible_keys_DRAFT.csv"
)
PROPOSAL = (
    AUD / "step36i_protocol_review_draft"
    / "36i_protocol_deviation_DRAFT.json"
)

STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"
ACCESS = MAN / "36_final_test_access_consumption_state.json"

OUT = AUD / "step36i_d2b_locked_recovery_contract_draft"

EXPECTED_HASHES = {
    SOURCE:
        "6de04ba7e167449101e2e16bab47ca4ce088f11284c4cbc93b61a6cfbb979f12",
    LEDGER:
        "82772196c92e88f1926a75a70c42f5f19119d7245d93b8364455fd28642ee121",
    KEYS:
        "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
    PROPOSAL:
        "ce89aebe2467fcb2003639aa00256d0670ce2ac2ac96fd140041dfd7a1895df8",
    STEP35:
        "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
    AUD / "step36i_b1_synthetic_preflight_draft"
    / "36i_b1_synthetic_sequences_DRAFT.json":
        "108db80959940d143a0c052e7116022bc24b4fbd118ce8b53a9facdbc39d1967",
    AUD / "step36i_b2_v2_synthetic_gpu_preflight_draft"
    / "36i_b2_v2_synthetic_gpu_results_DRAFT.json":
        "e0546ac4fdd25b3ce426903aae7752ac81f6d7ebaf10a24c33e946b31b55b70d",
    AUD / "step36i_c_synthetic_bootstrap_draft"
    / "36i_c_synthetic_bootstrap_DRAFT.json":
        "d939f1adb3ff882c88fb529b591c66f24d6fdb95e28bc13a2bad375903a21775",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_only():
    print("=" * 110)
    print("STEP 36I-D2b — LOCKED AMENDED-COHORT CONTRACT")
    print("=" * 110)

    print("\n[1] VERIFY IMMUTABLE SOURCE AND OFFLINE EVIDENCE")

    for path, expected in EXPECTED_HASHES.items():
        assert path.is_file(), f"Missing: {path}"
        assert sha(path) == expected, f"SHA mismatch: {path.name}"
        print("PASS:", path.name)

    if OUT.exists():
        raise RuntimeError(
            "D2b audit directory already exists. Refusing overwrite."
        )

    print("\n[2] VERIFY FINAL-TEST ACCESS STATE")

    marker = json.loads(ACCESS.read_text(encoding="utf-8"))

    assert (
        marker["state"]
        == "RAW_TEST_REOPENED_FOR_TECHNICAL_RECOVERY"
    )
    assert int(
        marker["raw_test_filesystem_open_count_completed"]
    ) == 2
    assert marker["final_test_scored"] is False

    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    assert proposal["status"] == "DRAFT_NOT_APPROVED"
    assert proposal["future_actions_authorized"] is False
    assert proposal["raw_test_additional_read_authorized"] is False

    print("PASS — two previous reads retained")
    print("PASS — no final-test scoring")
    print("PASS — additional raw-test access remains unauthorized")

    print("\n[3] VERIFY ORIGINAL SOURCE CONTRACT")

    tree = ast.parse(
        SOURCE.read_text(encoding="utf-8"),
        filename=SOURCE.name,
    )

    constants = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id in {
                        "EXPECTED_OUTER_TEST_ROWS",
                        "EXPECTED_OUTER_TEST_DATES",
                        "EXPECTED_COMMON_ROWS",
                        "EXPECTED_COMMON_DATES",
                        "EXPECTED_LOOKBACK",
                        "EXPECTED_FEATURES",
                        "EXPECTED_FITS",
                        "EXPECTED_THRESHOLDS",
                    }:
                        constants[target.id] = ast.literal_eval(
                            node.value
                        )

    expected_constants = {
        "EXPECTED_OUTER_TEST_ROWS": 6680,
        "EXPECTED_OUTER_TEST_DATES": 334,
        "EXPECTED_COMMON_ROWS": 6480,
        "EXPECTED_COMMON_DATES": 324,
        "EXPECTED_LOOKBACK": 60,
        "EXPECTED_FEATURES": 63,
        "EXPECTED_FITS": 40,
        "EXPECTED_THRESHOLDS": 24,
    }

    assert constants == expected_constants
    print("PASS — original frozen counts preserved")

    print("\n[4] RECONSTRUCT CORRECTED COHORT INDEPENDENTLY")

    columns = [
        "Ticker",
        "Date",
        "market_session_id",
        "continuity_segment_id",
        "outer_final_test_common_multitask",
    ]

    df = pd.read_parquet(
        LEDGER, columns=columns, engine="pyarrow"
    )
    df["Date"] = pd.to_datetime(
        df["Date"], errors="raise"
    ).dt.normalize()

    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    assert not df[["Ticker", "Date"]].duplicated().any()

    grouped = df.groupby("Ticker", sort=False)

    discontinuity = (
        df["continuity_segment_id"].ne(
            grouped["continuity_segment_id"].shift()
        )
        | df["market_session_id"].sub(
            grouped["market_session_id"].shift()
        ).ne(1)
    )

    df["run_id"] = (
        discontinuity.astype(np.int64)
        .groupby(df["Ticker"])
        .cumsum()
    )

    df["available_history"] = (
        df.groupby(["Ticker", "run_id"]).cumcount() + 1
    )

    original = df.loc[
        df["outer_final_test_common_multitask"].astype(bool)
    ].copy()

    candidate = original.loc[
        original["available_history"] >= 60
    ].copy()

    excluded = original.loc[
        original["available_history"] < 60
    ].copy()

    assert len(original) == 6480
    assert original["Date"].nunique() == 324

    assert len(candidate) == 5300
    assert candidate["Date"].nunique() == 265

    assert len(excluded) == 1180
    assert excluded["Date"].nunique() == 59

    assert candidate.groupby("Date")["Ticker"].nunique().eq(20).all()

    key_columns = [
        "Ticker",
        "Date",
        "market_session_id",
        "continuity_segment_id",
    ]

    frozen_keys = pd.read_csv(KEYS, parse_dates=["Date"])
    frozen_keys["Date"] = pd.to_datetime(
        frozen_keys["Date"], errors="raise"
    ).dt.normalize()

    actual_keys = candidate[key_columns].sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    expected_keys = frozen_keys[key_columns].sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        actual_keys,
        expected_keys,
        check_dtype=False,
    )

    periods = []

    for segment, part in candidate.groupby(
        "continuity_segment_id", sort=True
    ):
        periods.append({
            "segment": int(segment),
            "sessions": int(part["Date"].nunique()),
            "rows": int(len(part)),
            "start": str(part["Date"].min().date()),
            "end": str(part["Date"].max().date()),
        })

    assert [
        (p["sessions"], p["rows"])
        for p in periods
    ] == [(48, 960), (217, 4340)]

    print("PASS — original endpoints: 6480 across 324 sessions")
    print("PASS — corrected endpoints: 5300 across 265 sessions")
    print("PASS — 1180 endpoints excluded for insufficient history")
    print("PASS — candidate keys exactly match hashed DRAFT CSV")

    for period in periods:
        print("Verified period:", period)

    print("\n[5] FREEZE REVIEW REQUIREMENTS — WITHOUT APPROVAL")

    requirements = [
        "Original 6480/324 structural population remains provenance.",
        "Amended 5300/265 population uses frozen candidate keys.",
        "Every input sequence uses 60 same-segment causal sessions.",
        "Final-development scaler and regime thresholds remain frozen.",
        "All 40 fitted checkpoints and 24 thresholds are retained.",
        "Seed-wise physical batches and ordered inference are preserved.",
        "TimesNet batch-composition dependence is disclosed.",
        "Two historical raw-test reads remain in provenance.",
        "Any third raw-test read requires separate explicit authorization.",
        "Original Step-35 freeze is not retroactively rewritten.",
        "Amended outputs require separate names and manifests.",
        "Segment-aware bootstrap requires formal methodological review.",
        "Post-freeze changes are disclosed in the manuscript.",
    ]

    for i, requirement in enumerate(requirements, 1):
        print(f"{i:2d}. {requirement}")

    print("\n[6] WRITE DRAFT CONTRACT REPORT")

    OUT.mkdir(parents=True, exist_ok=False)

    report = {
        "status": "DRAFT_LOCKED_NOT_APPROVED",
        "stage": "36I-D2b",
        "function": "offline_recovery_contract_validation",
        "original_source_sha256": sha(SOURCE),
        "step35_manifest_sha256": sha(STEP35),
        "candidate_keys_sha256": sha(KEYS),
        "original_rows": 6480,
        "original_dates": 324,
        "proposed_rows": 5300,
        "proposed_dates": 265,
        "excluded_rows": 1180,
        "excluded_dates": 59,
        "verified_periods": periods,
        "expected_frozen_fits": 40,
        "expected_frozen_thresholds": 24,
        "raw_test_reads_recorded": 2,
        "raw_test_scored": False,
        "production_execution_enabled": False,
        "worker_execution_enabled": False,
        "sealed_test_read_enabled": False,
        "statistical_amendment_approved": False,
        "requirements_for_later_review": requirements,
        "frozen_original_source_modified": False,
        "frozen_step35_contract_modified": False,
    }

    report_path = OUT / "36i_d2b_locked_contract_DRAFT.json"

    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Contract draft:", report_path)
    print("Contract SHA256:", sha(report_path))

    print("\n" + "=" * 110)
    print("STEP 36I-D2b OFFLINE CONTRACT VERIFIED")
    print("Amended production runner implemented: FALSE")
    print("Amended statistical protocol approved: FALSE")
    print("Sealed final-test file opened: FALSE")
    print("Held-out inference performed: FALSE")
    print("Additional raw-test access authorized: FALSE")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only", action="store_true", required=True
    )
    parser.parse_args()
    verify_only()


if __name__ == "__main__":
    main()
