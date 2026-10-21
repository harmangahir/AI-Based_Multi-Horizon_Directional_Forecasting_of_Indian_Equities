#!/usr/bin/env python3
"""Step 36I-D2d: locked amended-cohort implementation scaffold.

OFFLINE SYNTHETIC PREFLIGHT ONLY.
No production materialization, sealed-test access, or checkpoint inference.
"""

from __future__ import annotations

import sys

# Deny production actions before loading any project modules.
DENIED = {
    "--execute", "--worker", "--prepare-only",
    "--materialize", "--production", "--resume"
}

if any(arg in DENIED for arg in sys.argv[1:]):
    raise SystemExit(
        "DENIED: amended production execution is not authorized."
    )

import argparse
import hashlib
import importlib.util
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

ORIGINAL_SOURCE = WORK / (
    "step36_one_time_final_test_materialization_inference_scoring_v1.py"
)
LEDGER = (
    EXP / "feature_data/step10_chronological_splits"
    / "10_split_membership.parquet"
)
MODEL_INTERFACE = MAN / "14_model_array_manifest.json"
STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"
ACCESS = MAN / "36_final_test_access_consumption_state.json"

KEYS = (
    AUD / "step36i_protocol_review_draft"
    / "36i_candidate_sequence_eligible_keys_DRAFT.csv"
)
CONTRACT = (
    AUD / "step36i_d2b_locked_recovery_contract_draft"
    / "36i_d2b_locked_contract_DRAFT.json"
)

REPORT_DIR = AUD / "step36i_d2d_amended_scaffold_draft"

EXPECTED = {
    ORIGINAL_SOURCE:
        "6de04ba7e167449101e2e16bab47ca4ce088f11284c4cbc93b61a6cfbb979f12",
    LEDGER:
        "82772196c92e88f1926a75a70c42f5f19119d7245d93b8364455fd28642ee121",
    KEYS:
        "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
    CONTRACT:
        "6bee0ef0048ef41b1122bffa65c7b46a7a4ffd0ffafe5ec29c6ca74bc5348923",
    STEP35:
        "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
    MODEL_INTERFACE:
        "b19b034bd31bb5d426e78c58f808d9e4c9cba16281e666984aae2c8bf6a150a0",
}

KEY_COLS = [
    "Ticker", "Date",
    "market_session_id", "continuity_segment_id"
]

PLANNED_OUTPUTS = {
    "array_directory":
        EXP / "feature_data/step36i_amended_final_test_arrays",
    "materialization_manifest":
        MAN / "36i_amended_final_test_materialization_manifest.json",
    "prediction_directory":
        EXP / "final_test_evaluation/step36i_amended_predictions",
    "evaluation_manifest":
        MAN / "36i_amended_final_test_evaluation_manifest.json",
    "metrics_csv":
        AUD / "36i_amended_final_test_metrics.csv",
    "access_journal":
        MAN / "36i_amended_final_test_access_journal.json",
    "failure_log":
        AUD / "36i_amended_final_test_execution_failure.json",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs():
    print("\n[1] IMMUTABLE SOURCE AND CONTRACT VERIFICATION")

    for path, expected_sha in EXPECTED.items():
        if not path.is_file():
            raise RuntimeError(f"Missing required artifact: {path}")

        if sha(path) != expected_sha:
            raise RuntimeError(f"SHA drift: {path.name}")

        print("PASS:", path.name)

    state = json.loads(ACCESS.read_text(encoding="utf-8"))

    assert state["state"] == (
        "RAW_TEST_REOPENED_FOR_TECHNICAL_RECOVERY"
    )
    assert int(
        state["raw_test_filesystem_open_count_completed"]
    ) == 2
    assert state["final_test_scored"] is False

    contract = json.loads(
        CONTRACT.read_text(encoding="utf-8")
    )
    assert contract["status"] == "DRAFT_LOCKED_NOT_APPROVED"
    assert contract["production_execution_enabled"] is False
    assert contract["proposed_rows"] == 5300

    print("PASS — two historical reads preserved")
    print("PASS — production access remains denied")

    return state


def check_output_isolation():
    print("\n[2] AMENDED OUTPUT PATH ISOLATION")

    for name, path in PLANNED_OUTPUTS.items():
        if path.exists():
            raise RuntimeError(
                f"Planned amended output already exists: {path}"
            )
        print("RESERVED:", name, path.name)

    if REPORT_DIR.exists():
        raise RuntimeError(
            "Existing D2d draft report directory; refusing overwrite."
        )

    print("PASS — no amended output paths occupied")


def load_original_sequence_builder():
    """Import frozen functions; do not run original main()."""
    spec = importlib.util.spec_from_file_location(
        "step36i_d2d_source_probe", ORIGINAL_SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_structural_panel():
    print("\n[3] STRUCTURAL PANEL AND FROZEN MODEL INTERFACE")

    columns = KEY_COLS + [
        "outer_final_test_common_multitask"
    ]

    panel = pd.read_parquet(
        LEDGER,
        columns=columns,
        engine="pyarrow"
    )

    panel["Date"] = pd.to_datetime(
        panel["Date"], errors="raise"
    ).dt.normalize()

    panel = panel.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    assert len(panel) == 50320
    assert not panel[["Ticker", "Date"]].duplicated().any()

    model_interface = json.loads(
        MODEL_INTERFACE.read_text(encoding="utf-8")
    )

    features = list(
        model_interface["ordered_63_features"]
    )
    stock_map = {
        str(k): int(v)
        for k, v in model_interface["stock_id_map"].items()
    }

    assert len(features) == 63
    assert len(set(features)) == 63
    assert len(stock_map) == 20
    assert set(stock_map) == set(panel["Ticker"].unique())

    print("PASS — structural panel:", len(panel))
    print("PASS — ordered features:", len(features))
    print("PASS — ticker IDs:", len(stock_map))

    return panel, features, stock_map


def select_amended_endpoints(
    panel, frozen_keys, stock_map
):
    """Original target eligibility followed by frozen sequence eligibility."""

    original = panel.loc[
        panel["outer_final_test_common_multitask"].astype(bool)
    ].copy()

    assert len(original) == 6480
    assert original["Date"].nunique() == 324

    if frozen_keys[KEY_COLS].isna().any().any():
        raise RuntimeError("Missing amended cohort keys.")

    if frozen_keys[KEY_COLS].duplicated().any():
        raise RuntimeError("Duplicate amended cohort keys.")

    if len(frozen_keys) != 5300:
        raise RuntimeError("Unexpected amended cohort size.")

    candidate = original.merge(
        frozen_keys[KEY_COLS],
        on=KEY_COLS,
        how="inner",
        validate="one_to_one",
    )

    candidate = candidate.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    if len(candidate) != 5300:
        raise RuntimeError(
            "Some frozen candidate keys do not match original eligibility."
        )

    if candidate["Date"].nunique() != 265:
        raise RuntimeError("Amended evaluation date count drift.")

    if not (
        candidate.groupby("Date")["Ticker"].nunique() == 20
    ).all():
        raise RuntimeError("Unbalanced amended panel.")

    candidate["stock_id"] = (
        candidate["Ticker"].map(stock_map)
        .astype(np.int64)
    )

    pd.testing.assert_frame_equal(
        candidate[KEY_COLS],
        frozen_keys[KEY_COLS]
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True),
        check_dtype=False,
    )

    return candidate


def attach_dummy_metadata(frame):
    """Synthetic metadata only; never real final-test labels."""

    frame = frame.copy()
    sessions = frame["market_session_id"].to_numpy(
        dtype=np.int64
    )

    frame["regime_id"] = sessions % 3
    frame["regime_name"] = frame["regime_id"].map({
        0: "low",
        1: "medium",
        2: "high",
    })
    frame["regime_threshold_fit_scope"] = "final_development"

    for h in (1, 3, 5):
        frame[f"target_date_{h}d"] = (
            frame["Date"] + pd.Timedelta(days=h + 1)
        )
        frame[f"direction_{h}d"] = (sessions + h) % 2

    return frame


def synthetic_materialization():
    state = verify_inputs()
    check_output_isolation()

    panel, features, stock_map = load_structural_panel()

    print("\n[4] VERIFY NEW AMENDED COHORT SELECTOR")

    keys = pd.read_csv(KEYS, parse_dates=["Date"])
    keys["Date"] = pd.to_datetime(
        keys["Date"], errors="raise"
    ).dt.normalize()

    chosen = select_amended_endpoints(
        panel, keys, stock_map
    )

    print("PASS — original 6480/324 preserved")
    print("PASS — proposed 5300/265 selected")

    print("\n[5] TEST FAIL-CLOSED COHORT SELECTION")

    try:
        select_amended_endpoints(
            panel, keys.iloc[:-1].copy(), stock_map
        )
    except RuntimeError:
        print("PASS — missing candidate key rejected")
    else:
        raise AssertionError(
            "Missing key was incorrectly accepted."
        )

    duplicate = pd.concat(
        [keys, keys.iloc[[0]]],
        ignore_index=True
    )

    try:
        select_amended_endpoints(
            panel, duplicate, stock_map
        )
    except RuntimeError:
        print("PASS — duplicate candidate key rejected")
    else:
        raise AssertionError(
            "Duplicate key was incorrectly accepted."
        )

    print("\n[6] SYNTHETIC PREDICTOR MATERIALIZATION")

    n = len(panel)
    sessions = panel["market_session_id"].to_numpy(
        dtype=np.float32
    )
    stock_ids = (
        panel["Ticker"].map(stock_map)
        .to_numpy(dtype=np.float32)
    )
    feature_ids = np.arange(63, dtype=np.float32)

    fake_features = (
        (sessions[:, None] % 97) * 0.001
        + stock_ids[:, None] * 0.01
        + feature_ids[None, :] * 0.0001
    ).astype(np.float32)

    synthetic = pd.concat(
        [
            panel,
            pd.DataFrame(
                fake_features,
                columns=features
            )
        ],
        axis=1
    )

    development = synthetic.loc[
        synthetic["Date"] < pd.Timestamp("2025-01-01")
    ].copy()

    test = synthetic.loc[
        synthetic["Date"] >= pd.Timestamp("2025-01-01")
    ].copy()

    assert len(development) == 43640
    assert len(test) == 6680

    test["stock_id"] = (
        test["Ticker"].map(stock_map).astype(np.int64)
    )

    chosen = select_amended_endpoints(
        test, keys, stock_map
    )
    chosen = attach_dummy_metadata(chosen)

    original_module = load_original_sequence_builder()

    history = original_module.build_history_lookup(
        development, test, features
    )

    X, metadata = original_module.materialize_sequences(
        chosen, history, features
    )

    assert X.shape == (5300, 60, 63)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    assert len(metadata) == 5300

    assert metadata.groupby(
        "reference_date"
    )["ticker"].nunique().eq(20).all()

    assert (
        metadata["sequence_end_market_session_id"].to_numpy(
            dtype=np.int64
        )
        - metadata["sequence_start_market_session_id"].to_numpy(
            dtype=np.int64
        )
        == 59
    ).all()

    print("PASS — amended selector + actual 60-session builder")
    print("PASS — synthetic tensor:", X.shape)
    print("PASS — 20-stock cross-sections remain balanced")

    print("\n[7] WRITE OFFLINE-ONLY DRAFT REPORT")

    REPORT_DIR.mkdir(parents=True, exist_ok=False)

    report = {
        "status": "PASS_OFFLINE_SYNTHETIC_ONLY",
        "stage": "36I-D2d",
        "source_sha256": sha(Path(__file__).resolve()),
        "original_step36_source_sha256": sha(ORIGINAL_SOURCE),
        "locked_contract_sha256": sha(CONTRACT),
        "candidate_keys_sha256": sha(KEYS),
        "original_structural_rows": 6480,
        "original_structural_dates": 324,
        "proposed_sequence_rows": 5300,
        "proposed_sequence_dates": 265,
        "synthetic_shape": list(X.shape),
        "missing_key_rejected": True,
        "duplicate_key_rejected": True,
        "historical_raw_test_reads": int(
            state["raw_test_filesystem_open_count_completed"]
        ),
        "planned_amended_outputs": {
            name: str(path)
            for name, path in PLANNED_OUTPUTS.items()
        },
        "production_runner_complete": False,
        "production_execution_enabled": False,
        "raw_test_read_enabled": False,
        "gpu_inference_executed_here": False,
        "real_labels_accessed": False,
        "scientific_amendment_approved": False,
        "additional_test_access_authorized": False,
    }

    report_path = (
        REPORT_DIR / "36i_d2d_amended_scaffold_DRAFT.json"
    )

    report_path.write_text(
        json.dumps(
            report, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )

    print("Report:", report_path)
    print("Report SHA256:", sha(report_path))

    print("\n" + "=" * 110)
    print("STEP 36I-D2d OFFLINE SCAFFOLD PREFLIGHT PASSED")
    print("Original Step-36 source modified: FALSE")
    print("Original Step-35 contract modified: FALSE")
    print("Sealed final-test parquet opened: FALSE")
    print("Held-out predictions generated: FALSE")
    print("Additional raw-test access authorized: FALSE")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline-preflight",
        action="store_true",
        required=True
    )
    parser.parse_args()
    synthetic_materialization()


if __name__ == "__main__":
    main()
