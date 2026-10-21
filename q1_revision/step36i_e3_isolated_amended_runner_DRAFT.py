#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-E3 — ISOLATED AMENDED EVALUATION RUNNER, REVIEW BUILD.

No production execution or raw held-out test access in this version.

The only CLI mode is --offline-preflight. It checks the original contracts,
reconstructs the post-freeze structural cohort (when the WSL ledger is present),
verifies the full frozen model/threshold inventory, tests fail-closed key
selection, and writes a new draft review report without overwriting evidence.

In-memory integration functions specify a future evaluation pipeline, but are
hard-gated against invocation pending independent methodological approval.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

# Fail closed before importing any project module or touching a data path.
PRODUCTION_AUTHORIZED = False
FORBIDDEN_FLAGS = {
    "--execute", "--worker", "--materialize", "--resume", "--approve",
    "--authorize", "--read-test", "--production", "--score-test",
}
if any(arg in FORBIDDEN_FLAGS for arg in sys.argv[1:]):
    raise SystemExit("DENIED: E3 is offline only. Third raw-test access is NOT authorized.")

import numpy as np
import pandas as pd

EXPECTED_BUNDLE_SHA = "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851"
EXPECTED_SOURCE_SHA = "6de04ba7e167449101e2e16bab47ca4ce088f11284c4cbc93b61a6cfbb979f12"
EXPECTED_STEP35_SHA = "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7"
EXPECTED_KEYS_SHA = "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65"
EXPECTED_MEMBERSHIP_SHA = "82772196c92e88f1926a75a70c42f5f19119d7245d93b8364455fd28642ee121"
EXPECTED_D2F_SHA = "197ad07632bfdcd5134ca959c132a896a75711986500d72d7713d681e56370ab"
EXPECTED_D2G_SHA = "f520919ef7cd0640a5807fcc61d9eee638fdf19a651b89523f1ef55607a52a6d"
EXPECTED_E1_SHA = "e6c50c7dbbca89f3bd0ad6dd42f61c14cac15355cda7ff2eacaa5612f170eca0"
EXPECTED_THRESH_SHA = "79dc80e0e3b18427cebd77742ff2d10b87aebcdd9ea1e459fed30ac980d0e24f"
EXPECTED_PLAN_SHA = "a2f8489933e8d2bbd11d6bab72361f7ad066dd37bc62e1db45bb0d20d87ce086"
EXPECTED_ORIGINAL_STEP36_SHA = "b5321fe514c7f9f34d6d675debbbca1fd292c58b2874b724987b3a819671b131"
MODEL_BATCH = {"DLINEAR": ("DLINEAR_B", 128), "NLINEAR": ("NLINEAR_B", 128),
               "PATCHTST": ("PATCHTST_B", 64), "TIMESNET": ("TIMESNET_B", 64),
               "ITRANSFORMER": ("ITRANSFORMER_A", 128), "FREDFORMER": ("FREDFORMER_B", 16),
               "FRNET": ("FRNET_A", 64), "MLF": ("MLF_A", 32)}
SEEDS = (42, 43, 44, 45, 46)
HORIZONS = (1, 3, 5)
KEY_COLUMNS = ["Ticker", "Date", "market_session_id", "continuity_segment_id"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def assert_sha(data: bytes, expected: str, label: str) -> None:
    if digest(data) != expected:
        raise RuntimeError(f"{label} SHA256 mismatch; stopping.")


def load_bundle(path: Path) -> dict[str, bytes]:
    if not path.is_file() or file_digest(path) != EXPECTED_BUNDLE_SHA:
        raise RuntimeError("Source-review bundle missing or SHA256 mismatch.")
    with zipfile.ZipFile(path, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Corrupted archive member.")
        names = archive.namelist()
        if len(names) != 18 or len(names) != len(set(names)):
            raise RuntimeError("Unexpected source-review archive entry count.")
        if any("/" in name or "\\" in name or ".." in name for name in names):
            raise RuntimeError("Unsafe/nonflat ZIP member names.")
        return {name: archive.read(name) for name in names}


def parse_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


def parse_json(raw: bytes) -> dict:
    return json.loads(raw.decode("utf-8"))


def verify_access_state(marker: dict, proposal: dict, review: dict) -> None:
    if (marker.get("state") != "RAW_TEST_REOPENED_FOR_TECHNICAL_RECOVERY"
            or int(marker.get("raw_test_filesystem_open_count_completed", -1)) != 2
            or marker.get("final_test_scored") is not False):
        raise RuntimeError("Unexpected historical raw-test access state.")
    if (proposal.get("status") != "DRAFT_NOT_APPROVED"
            or proposal.get("future_actions_authorized") is not False
            or proposal.get("raw_test_additional_read_authorized") is not False
            or review.get("approval_granted") is not False
            or review.get("third_raw_test_read_authorized") is not False):
        raise RuntimeError("Amendment or access status has changed; STOP.")
    if marker.get("observed_raw_test_sha256") != marker.get("expected_raw_test_sha256"):
        raise RuntimeError("Raw-test access provenance hash mismatch.")


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if not set(KEY_COLUMNS).issubset(frame.columns):
        raise RuntimeError("Missing cohort key columns.")
    keys = frame[KEY_COLUMNS].copy()
    keys["Date"] = pd.to_datetime(keys["Date"], errors="raise").dt.normalize()
    for name in ("market_session_id", "continuity_segment_id"):
        keys[name] = pd.to_numeric(keys[name], errors="raise").astype(np.int64)
    if keys.isna().any().any() or keys.duplicated().any():
        raise RuntimeError("Missing/duplicate cohort keys.")
    return keys.reset_index(drop=True)


def verify_key_cohort(keys: pd.DataFrame) -> dict:
    keys = normalize_keys(keys)
    if len(keys) != 5300 or keys["Date"].nunique() != 265:
        raise RuntimeError("Expected precisely 5300 rows and 265 sessions.")
    if not keys.equals(keys.sort_values(["Date", "Ticker"]).reset_index(drop=True)):
        raise RuntimeError("Candidate physical row order drift.")
    if not keys.groupby("Date")["Ticker"].nunique().eq(20).all():
        raise RuntimeError("Unbalanced 20-ticker cross-section.")
    if keys["Ticker"].nunique() != 20:
        raise RuntimeError("Unexpected equity universe.")
    periods = []
    for segment, part in keys.groupby("continuity_segment_id", sort=True):
        dates = part.drop_duplicates("Date").sort_values("Date")
        if not np.all(np.diff(dates["market_session_id"].to_numpy()) == 1):
            raise RuntimeError("Master-session discontinuity within segment.")
        periods.append({"segment": int(segment), "sessions": len(dates), "rows": len(part),
                        "start": str(dates["Date"].min().date()),
                        "end": str(dates["Date"].max().date())})
    if [(p["sessions"], p["rows"]) for p in periods] != [(48, 960), (217, 4340)]:
        raise RuntimeError("Expected 48+217 segment composition.")
    if [(p["start"], p["end"]) for p in periods] != [
        ("2025-01-01", "2025-03-07"), ("2025-08-01", "2026-06-22")
    ]:
        raise RuntimeError("Frozen candidate segment date interval drift.")
    return {"original_rows": 6480, "original_dates": 324,
            "corrected_rows": 5300, "corrected_dates": 265,
            "excluded_rows": 1180, "periods": periods}


def select_preloaded_endpoints(frame: pd.DataFrame, candidate_keys: pd.DataFrame) -> pd.DataFrame:
    """Pure amended selector. Does not read a parquet or infer targets.

    Requires caller-supplied in-memory rows already annotated with the original
    frozen Step-10 structural multitask flag. This function DOES NOT grant
    permission to obtain held-out rows.
    """
    if "eligible_common_multitask" not in frame.columns:
        raise RuntimeError("Frozen common-eligibility flag absent.")
    original = frame.loc[frame["eligible_common_multitask"].astype(bool)].copy()
    if len(original) != 6480 or original["Date"].nunique() != 324:
        raise RuntimeError("Original structural membership count mismatch.")
    if original[KEY_COLUMNS].isna().any().any() or original[KEY_COLUMNS].duplicated().any():
        raise RuntimeError("Invalid original structural keys.")
    frozen = normalize_keys(candidate_keys)
    verify_key_cohort(frozen)
    original["Date"] = pd.to_datetime(original["Date"]).dt.normalize()
    result = original.merge(frozen, how="inner", on=KEY_COLUMNS, validate="one_to_one")
    result = result.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    if len(result) != 5300:
        raise RuntimeError("Cohort whitelist is not a complete subset of original rows.")
    pd.testing.assert_frame_equal(normalize_keys(result), frozen, check_dtype=False)
    return result


def verify_frozen_model_plan(plan: pd.DataFrame, thresholds: pd.DataFrame,
                             d2f: dict, d2g: dict, step35: dict) -> dict:
    required = {"model_id", "config_id", "seed", "physical_inference_microbatch",
                "inference_shuffle", "gradient_tracking", "checkpoint_sha256"}
    if not required.issubset(plan.columns) or len(plan) != 40:
        raise RuntimeError("Frozen 40-row plan schema drift.")
    actual_pairs = {(str(m), int(s)) for m, s in zip(plan["model_id"], plan["seed"])}
    expected_pairs = {(m, s) for m in MODEL_BATCH for s in SEEDS}
    if actual_pairs != expected_pairs or plan[["model_id", "seed"]].duplicated().any():
        raise RuntimeError("Frozen model-seed inventory drift.")
    for rec in plan.itertuples(index=False):
        expected_cfg, batch = MODEL_BATCH[rec.model_id]
        if rec.config_id != expected_cfg or int(rec.physical_inference_microbatch) != batch:
            raise RuntimeError("Frozen architecture/microbatch configuration drift.")
        if str(rec.inference_shuffle).lower() not in ("false", "0") or str(rec.gradient_tracking).lower() not in ("false", "0"):
            raise RuntimeError("Shuffling/gradient tracking unexpectedly enabled.")
        if (str(rec.inference_row_order) != "reference_date_then_ticker"
                or str(rec.probability_transform) != "sigmoid(logit)"
                or str(rec.ensemble_rule) != "arithmetic_mean_of_seed_probabilities_42_43_44_45_46"
                or str(rec.seed_probability_retained).lower() not in ("true", "1")):
            raise RuntimeError("Frozen row order, probability or ensemble rule drift.")
    if len(thresholds) != 24 or thresholds[["model_id", "horizon_days"]].duplicated().any():
        raise RuntimeError("Frozen 24-threshold family drift.")
    if {(str(m), int(h)) for m, h in zip(thresholds["model_id"], thresholds["horizon_days"])} != {
        (m, h) for m in MODEL_BATCH for h in HORIZONS
    }:
        raise RuntimeError("Frozen threshold model/horizon pairs drift.")
    values = pd.to_numeric(thresholds["selected_threshold"], errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise RuntimeError("Invalid frozen decision thresholds.")
    if (d2f.get("checkpoints_tested") != 40 or len(d2f.get("synthetic_worker_results", [])) != 40
            or len(d2f.get("synthetic_ensembles", [])) != 8
            or d2g.get("seed_predictions_hash_verified") != 40
            or d2g.get("frozen_thresholds_used") != 24):
        raise RuntimeError("Offline 40-checkpoint/scoring evidence incomplete.")
    contract = step35["dependence_aware_inference_contract"]
    if (contract["primary_block_length_sessions"] != 20
            or contract["replications_per_block_length"] != 2000
            or contract["sensitivity_block_lengths_sessions"] != [10, 20, 40]
            or contract["primary_multiple_testing_family"] !=
            "all 8 architectures x 3 horizons = 24 balanced-accuracy above-chance tests at block length 20"):
        raise RuntimeError("Frozen bootstrap/FDR design drift.")
    if step35["final_test_evaluation_contract_fingerprint"] != "44949863de7e82c3c303ca55beff8c59ae3206ea7eb542a9aab115ff88632bfc":
        raise RuntimeError("Frozen Step-35 scientific contract fingerprint drift.")
    return {"verified_model_seed_pairs": 40, "verified_threshold_pairs": 24,
            "synthetic_checkpoint_runs": 40, "synthetic_five_seed_ensembles": 8,
            "primary_bootstrap_block": 20, "bootstrap_sensitivities": [10, 40],
            "bootstrap_replicates": 2000, "bh_fdr_family": 24}


def reject_unapproved_production(*args, **kwargs):
    """No further held-out access is supported by this review build."""
    raise PermissionError(
        "E3 is an UNAPPROVED post-freeze execution design. No sealed test open, "
        "checkpoint worker, real-data scoring or journal claim is enabled."
    )

# These explicit entry points are intentionally DISABLED pending an approved,
# independently reviewed post-freeze protocol amendment and a separate
# authorization decision about a third test read. Source alone is not approval.
materialize_sealed_test = reject_unapproved_production
run_real_checkpoint_workers = reject_unapproved_production
score_real_final_test = reject_unapproved_production


def review_planned_output_isolation(experiment_root: Path) -> list[dict]:
    """Metadata-only path checks; no directories created or test data read."""
    paths = {
        "array_dir": experiment_root / "feature_data/step36i_amended_final_test_arrays",
        "array_manifest": experiment_root / "manifests/36i_amended_final_test_materialization_manifest.json",
        "predictions": experiment_root / "final_test_evaluation/step36i_amended_predictions",
        "evaluation_manifest": experiment_root / "manifests/36i_amended_final_test_evaluation_manifest.json",
        "metrics": experiment_root / "audits/36i_amended_final_test_metrics.csv",
        "independent_access_journal": experiment_root / "manifests/36i_amended_final_test_access_journal.json",
        "failure_evidence": experiment_root / "audits/36i_amended_final_test_execution_failure.json",
    }
    result = []
    for name, path in paths.items():
        if path.exists():
            raise RuntimeError(f"Isolated planned output occupied; STOP: {path}")
        result.append({"name": name, "path": str(path), "vacant": True})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-preflight", action="store_true", required=True)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="The exact 18-file SHA256-verified E2 review ZIP (no raw test)")
    parser.add_argument("--project-root", type=Path,
                        help="Optional WSL project q1_revision directory; verifies local ledger and live marker")
    parser.add_argument("--report-dir", type=Path,
                        help="Optional NEW directory for an E3 synthetic-free review report")
    args = parser.parse_args()

    print("=" * 104)
    print("STEP 36I-E3 — OFFLINE ISOLATED AMENDED RUNNER REVIEW")
    print("=" * 104)
    print("[1] VERIFY E2 BUNDLE AND ORIGINAL SCIENTIFIC CONTRACT")
    bundle = load_bundle(args.bundle)
    for name, expected in [
        ("recovered_step36_source.py", EXPECTED_SOURCE_SHA),
        ("step35_contract.json", EXPECTED_STEP35_SHA),
        ("candidate_keys_DRAFT.csv", EXPECTED_KEYS_SHA),
        ("synthetic_inference_report.json", EXPECTED_D2F_SHA),
        ("synthetic_scoring_report.json", EXPECTED_D2G_SHA),
        ("amendment_review_DRAFT.json", EXPECTED_E1_SHA),
        ("frozen_thresholds.csv", EXPECTED_THRESH_SHA),
        ("frozen_inference_plan.csv", EXPECTED_PLAN_SHA),
    ]:
        assert_sha(bundle[name], expected, name)
        print("PASS:", name)

    marker = parse_json(bundle["access_history.json"])
    proposal = parse_json(bundle["protocol_deviation_DRAFT.json"])
    review = parse_json(bundle["amendment_review_DRAFT.json"])
    verify_access_state(marker, proposal, review)
    print("[2] ACCESS HISTORY: 2 completed reads, 0 real scorings; third read DENIED")

    keys = normalize_keys(parse_csv(bundle["candidate_keys_DRAFT.csv"]))
    cohort = verify_key_cohort(keys)
    print("[3] CORRECTED COHORT: 6480 original, 5300 proposed, 1180 excluded")
    for period in cohort["periods"]:
        print("PERIOD:", period)

    inventory = verify_frozen_model_plan(
        parse_csv(bundle["frozen_inference_plan.csv"]),
        parse_csv(bundle["frozen_thresholds.csv"]),
        parse_json(bundle["synthetic_inference_report.json"]),
        parse_json(bundle["synthetic_scoring_report.json"]),
        parse_json(bundle["step35_contract.json"]),
    )
    print("[4] FROZEN MODEL/THRESHOLD/BOOTSTRAP PLAN: PASS")

    # Fail-closed selectors; these are non-market fixtures and cannot reach the test.
    fixture = keys.copy()
    fixture["eligible_common_multitask"] = True
    fixture = pd.concat([fixture, pd.DataFrame([{**fixture.iloc[0].to_dict(),
                                    "Ticker": "SYNTHETIC_EXCLUDED",
                                    "eligible_common_multitask": False}])], ignore_index=True)
    # The fixture has only 5300 positive rows rather than the true 6480. We
    # intentionally test the precondition rejection here; never forge the true
    # original 1180 rows or claim this fixture is an end-to-end test.
    try:
        select_preloaded_endpoints(fixture, keys)
    except RuntimeError as exc:
        if "Original structural membership count mismatch" not in str(exc):
            raise
        print("[5] FAIL-CLOSED ORIGINAL-COHORT PRECONDITION: PASS")
    else:
        raise RuntimeError("Missing original 6480-row gate; STOP")
    try:
        materialize_sealed_test()
    except PermissionError:
        print("[6] RAW TEST PRODUCTION GATE: PASS (DENIED)")
    else:
        raise RuntimeError("Unexpected production access capability")

    local_checks = {"project_ledger_checked": False, "live_marker_checked": False}
    output_isolation = []
    if args.project_root:
        work = args.project_root.resolve()
        if work.name != "q1_revision":
            raise RuntimeError("Project root must point to q1_revision")
        exp = work.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
        ledger = exp / "feature_data/step10_chronological_splits/10_split_membership.parquet"
        for local_path, expected in [
            (work / "step36_one_time_final_test_materialization_inference_scoring_v1.py", EXPECTED_SOURCE_SHA),
            (exp / "manifests/35_final_test_evaluation_contract_freeze_manifest.json", EXPECTED_STEP35_SHA),
            (exp / "audits/35_final_test_inference_plan.csv", EXPECTED_PLAN_SHA),
            (exp / "audits/35_final_test_threshold_contract.csv", EXPECTED_THRESH_SHA),
        ]:
            if file_digest(local_path) != expected:
                raise RuntimeError(f"Local frozen input mismatch: {local_path.name}")
        print("[7] LOCAL SOURCE/STEP35/INFERENCE PLAN/THRESHOLD HASHES: PASS")
        if file_digest(ledger) != EXPECTED_MEMBERSHIP_SHA:
            raise RuntimeError("Frozen Step10 membership SHA drift")
        cols = KEY_COLUMNS + ["outer_final_test_common_multitask"]
        membership = pd.read_parquet(ledger, columns=cols, engine="pyarrow")
        membership["Date"] = pd.to_datetime(membership["Date"]).dt.normalize()
        membership = membership.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        g = membership.groupby("Ticker", sort=False)
        boundary = (membership["continuity_segment_id"].ne(g["continuity_segment_id"].shift()) |
                    membership["market_session_id"].sub(g["market_session_id"].shift()).ne(1))
        membership["run_id"] = boundary.astype(int).groupby(membership["Ticker"]).cumsum()
        membership["history_count"] = membership.groupby(["Ticker", "run_id"]).cumcount() + 1
        original = membership.loc[membership["outer_final_test_common_multitask"].astype(bool)].copy()
        eligible = original.loc[original["history_count"] >= 60, KEY_COLUMNS]
        eligible = eligible.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        if len(membership) != 50320 or len(original) != 6480 or len(eligible) != 5300:
            raise RuntimeError("Live ledger cohort mismatch")
        pd.testing.assert_frame_equal(normalize_keys(eligible), keys, check_dtype=False)
        # End-to-end verification of the independently authored pure selector
        # against the actual frozen structural ledger, not fabricated test data.
        membership["eligible_common_multitask"] = membership[
            "outer_final_test_common_multitask"
        ].astype(bool)
        selected = select_preloaded_endpoints(membership, keys)
        if len(selected) != 5300:
            raise RuntimeError("Independent selector did not select 5300 rows")
        # Adversarial controls must fail closed before any dataset access.
        for malicious in (keys.iloc[:-1].copy(),
                          pd.concat([keys, keys.iloc[[0]]], ignore_index=True)):
            try:
                select_preloaded_endpoints(membership, malicious)
            except (RuntimeError, AssertionError, ValueError):
                pass
            else:
                raise RuntimeError("Malformed candidate keys were accepted")
        local_checks["project_ledger_checked"] = True
        local_checks["selector_passed_with_true_structural_ledger"] = True
        print("[8] LIVE LEDGER + EXACT SELECTOR + NEGATIVE CONTROLS: PASS")
        live = parse_json((exp / "manifests/36_final_test_access_consumption_state.json").read_bytes())
        if live != marker:
            raise RuntimeError("Live access marker differs from E2 preserved snapshot; STOP")
        local_checks["live_marker_checked"] = True
        print("[9] LIVE ACCESS MARKER MATCHES E2 SNAPSHOT: PASS")
        output_isolation = review_planned_output_isolation(exp)
        print("[10] SEPARATE OUTPUT NAMESPACE: VACANT")

    report = {
        "status": "PASS_OFFLINE_ONLY_NOT_APPROVED", "stage": "36I-E3",
        "source_sha256": file_digest(Path(__file__)),
        "e2_bundle_sha256": file_digest(args.bundle),
        "original_step36_source_sha256": EXPECTED_SOURCE_SHA,
        "step35_contract_sha256": EXPECTED_STEP35_SHA,
        "candidate_keys_sha256": EXPECTED_KEYS_SHA,
        "cohort": cohort, "frozen_inference_contract": inventory,
        "local_verifications": local_checks, "planned_output_isolation": output_isolation,
        "historical_raw_test_reads": 2, "heldout_scoring_performed": False,
        "production_runner_complete": False,
        "production_access_enabled": PRODUCTION_AUTHORIZED,
        "postfreeze_amendment_approved": False,
        "third_raw_test_read_authorized": False,
        "synthetic_gpu_runs_performed_here": 0,
        "limitations": [
            "Only verification and pure cohort-selection interface are implemented in E3.",
            "No raw-test loader or production inference/scoring orchestration is enabled.",
            "Production materialization/worker persistence must be independently implemented and reviewed.",
            "Synthetic prior outputs do not demonstrate out-of-sample predictive skill.",
            "Post-freeze cohort correction changes population and inference; claims require separate approval.",
        ],
    }
    if args.report_dir:
        if args.report_dir.exists():
            raise RuntimeError("Report directory already exists; refusing overwrite.")
        args.report_dir.mkdir(parents=True, exist_ok=False)
        path = args.report_dir / "36i_e3_isolated_runner_review_DRAFT.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("DRAFT REPORT:", path)
        print("DRAFT SHA256:", file_digest(path))
    print("=" * 104)
    print("STEP 36I-E3 OFFLINE REVIEW COMPLETE — PRODUCTION DISABLED")
    print("Raw-test access: NONE | Final-test scoring: NONE | Formal approval: NONE")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
