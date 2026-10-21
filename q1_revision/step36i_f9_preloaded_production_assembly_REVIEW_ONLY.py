#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 36I-F9 — preloaded production ASSEMBLY, REVIEW ONLY.

Permitted CLI modes: --offline-preflight (read-only WSL), --self-test (portable).

This is a genuinely integrated IN-MEMORY orchestration function, not a live
production launcher. In the future, a separately reviewed/approved release must:
  (a) validate the signed protocol amendment and explicit third-read authority,
  (b) reserve and archive a real-data access journal BEFORE an exclusive open,
  (c) load the single sealed parquet via the verified one-descriptor byte path,
  (d) pass the *already decoded* DataFrame to assemble_from_preloaded(),
  (e) commit the resulting payloads atomically with independent hashes,
  (f) durably record success or irreversible failure, with no blind retry.

NO CLI or callable in this file can open the final-test dataset or commit real
outputs. The historical marker and original Step35 contract are immutable.
F1, F3, F4, and F7 review helpers MUST NOT be modified to bypass their guards.

The 5,300 row amended cohort and stratified resampling remain UNAPPROVED;
this review source is not authorization to access, infer, or score held-out data.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

_DENIED = {
    "--execute", "--production", "--worker", "--read-test", "--score-test",
    "--resume", "--authorize", "--approve", "--run", "--commit-real",
    "--materialize", "--final-test", "--enable", "--test-parquet",
}
if _DENIED.intersection(sys.argv[1:]):
    raise SystemExit("DENIED: F9 is a read-only production assembly review, not an execution grant")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
SOURCE_NAMES = {
    1: "step36i_f1_amended_production_runner_REVIEW_ONLY.py",
    2: "step36i_f2_independent_source_review_READ_ONLY.py",
    3: "step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py",
    4: "step36i_f4_segment_stratified_inference_REVIEW_ONLY.py",
    5: "step36i_f5_independent_statistical_audit_READ_ONLY.py",
    6: "step36i_f6_end_to_end_synthetic_transaction_REVIEW_ONLY.py",
    7: "step36i_f7_one_read_loader_journal_integration_REVIEW_ONLY.py",
    8: "step36i_f8_single_read_to_atomic_commit_INTEGRATION_REVIEW_ONLY.py",
}
SOURCE_HASHES = {
    1: "e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556",
    2: "ec6400f0f356b7a9888c443890b333f6fa6c6d80f6d29d700625eba4335f00bc",
    3: "b7c28f536a57653f10cc74155b38941b7e2e86b508089e5a3378bf9ff97b7539",
    4: "421cb3e8b83f96efaef1f3e0578e2cb7513c728aef36bda68cdc9216b07b4b10",
    5: "47a5ed8044493f4d09cbafa16fea5a62482d13b27856da945bd5d6e04f5e8ee3",
    6: "f075401a94c3bd2a971d4ade7c71c28207754d5c5810b40462e0b2bc88d62bad",
    7: "b01bd3ce80194a4981a3c518fcf2e7f46948819715a51319843fa50630a54685",
    8: "30197a482e9ccb94eab43ee4438681261fe91fa7609b668a03cf743c1a6d81ef",
}
BUNDLE = ROOT / "step36i_e2_source_review_bundle.zip"
BUNDLE_SHA = "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851"
STEP35 = EXP / "manifests/35_final_test_evaluation_contract_freeze_manifest.json"
STEP35_SHA = "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7"
LEDGER = EXP / "manifests/36_final_test_access_consumption_state.json"
AUD = EXP / "audits"
CHECKPOINT_PLAN = AUD / "35_final_test_inference_plan.csv"
THRESHOLD_PLAN = AUD / "35_final_test_threshold_contract.csv"
AMENDED_KEYS = AUD / "step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv"


def require(test: bool, msg: str):
    if not bool(test):
        raise RuntimeError(msg)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def check(path: Path, expected: str, name: str):
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe {name}: {path}")
    require(file_sha(path) == expected, f"SHA256 drift: {name}")


def load_local_helper(index: int):
    # Helpers are loaded only for in-memory assembly; NEVER call their main().
    name = f"step36i_f9_review_dep_{index}"
    spec = importlib.util.spec_from_file_location(name, ROOT / SOURCE_NAMES[index])
    require(spec is not None and spec.loader is not None, f"Import unavailable: F{index}")
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def validate_interfaces_static():
    required = {
        1: {"candidate", "plan_thresholds", "confirm_legacy_output_isolation",
            "read_only_access_check", "materialize_preloaded_from_frozen_helpers",
            "frozen_inference_from_preloaded_X", "in_memory_five_seed_scoring"},
        3: {"rename_dir_no_replace", "write_unique", "fsync_directory"},
        4: {"evaluate_in_memory", "stratified_session_indices", "bh_adjust"},
        7: {"OneAttemptJournal", "read_bytes_exactly_once_fixture"},
    }
    for idx, expected_names in required.items():
        tree = ast.parse((ROOT / SOURCE_NAMES[idx]).read_text("utf-8"))
        actual = {item.name for item in tree.body
                  if isinstance(item, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
        require(expected_names <= actual, f"F{idx} implementation interface drift: {expected_names-actual}")
    # Warn if future authors incorrectly try to expose this review build as a CLI.
    tree = ast.parse(Path(__file__).read_text("utf-8"))
    main_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == "assemble_from_preloaded"]
    require(len(main_calls) == 0, "Production assembly unexpectedly connected to CLI")


def _save_npy(array: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, array, allow_pickle=False)
    return b.getvalue()


def _save_npz(**arrays: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    return b.getvalue()


def _save_parquet(frame: pd.DataFrame) -> bytes:
    b = io.BytesIO()
    frame.to_parquet(b, index=False, engine="pyarrow")
    return b.getvalue()


def _save_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _manifest(payloads: dict[str, bytes], metadata: dict) -> dict:
    require(payloads and len(set(payloads)) == len(payloads), "Empty/duplicate artifact names")
    art = []
    for name in sorted(payloads):
        p = Path(name)
        require(name and not p.is_absolute() and ".." not in p.parts and
                all(x and x != "." for x in p.parts), "Unsafe artifact path")
        data = payloads[name]
        require(isinstance(data, bytes), f"Artifact must be bytes: {name}")
        art.append({"relative_path": name, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()})
    return {"schema": "STEP36I_F9_PRELOADED_ASSEMBLY_DRAFT_V1",
            "original_step35_sha256": STEP35_SHA,
            "protocol_amendment_at_assembly_time": "REQUIRES_SEPARATE_APPROVAL",
            "access_journal": "NOT_ATTACHED_BY_REVIEW_BUILD",
            "source_dataset_opened_by_f9": False,
            "artifact_count": len(art), "artifacts": art, "context": metadata}


def assemble_from_preloaded(preloaded_test: pd.DataFrame,
                            final_dev_scaled_history: pd.DataFrame,
                            upstream: dict) -> tuple[dict[str, bytes], dict]:
    """Full computational assembly on *already-authorized, in-memory* inputs.

    IMPORTANT: This function does not open data or commit outputs. It is not a
    production permission gate. Do not invoke on real test data unless a later,
    independent source review and formal protocol/access signoff has occurred.
    It uses exactly F1 materialization/inference/scoring and F4 statistics.
    """
    f1 = load_local_helper(1)
    f4 = load_local_helper(4)
    keys = f1.candidate(pd.read_csv(AMENDED_KEYS))
    plan = pd.read_csv(CHECKPOINT_PLAN)
    threshold_map = f1.plan_thresholds(plan, pd.read_csv(THRESHOLD_PLAN))
    require(set(upstream) >= {"features", "scaler_path", "stock_id_map"}, "Upstream contract missing")
    require(len(upstream["features"]) == 63, "Frozen feature order missing")

    # This helper internally validates the *original 6480 structural members*
    # BEFORE selecting the proposed 5300; no test-label-based selection.
    X, meta = f1.materialize_preloaded_from_frozen_helpers(
        preloaded_test, final_dev_scaled_history, keys, upstream)
    require(X.shape == (5300, 60, 63) and X.dtype == np.float32,
            "Expected 5300x60x63 float32 sequence array")
    labels = np.column_stack([meta[f"direction_{h}d"].to_numpy(np.int64)
                              for h in (1, 3, 5)]).astype(np.int8)
    require(labels.shape == (5300, 3) and np.isin(labels, [0, 1]).all(),
            "Invalid target alignment")
    meta_keys = pd.DataFrame({"Ticker": meta.ticker, "Date": meta.reference_date,
                              "market_session_id": meta.reference_market_session_id,
                              "continuity_segment_id": meta.continuity_segment_id})
    pd.testing.assert_frame_equal(f1.candidate(meta_keys), keys, check_dtype=False)

    seed_probs = {}
    inference_records = []
    payloads = {
        "arrays/X_5300x60x63.npy": _save_npy(X),
        "arrays/labels_5300x3.npy": _save_npy(labels),
        "arrays/metadata.parquet": _save_parquet(meta),
    }
    lookup = {(str(r.model_id), int(r.seed)): r for r in plan.itertuples(index=False)}
    require(len(lookup) == 40, "Inference plan missing checkpoint rows")
    for model in f1.MODEL:
        for seed in f1.SEEDS:
            key = (model, seed)
            require(key in lookup, f"Missing checkpoint: {key}")
            r = pd.Series(lookup[key]._asdict())
            logits, probs, stats = f1.frozen_inference_from_preloaded_X(X, r)
            require(logits.shape == (5300, 3) and probs.shape == (5300, 3),
                    "Inference output differs from materialized cohort")
            seed_probs[key] = probs
            payloads[f"seed_predictions/{model}_seed{seed}.npz"] = _save_npz(
                logits=logits, probabilities=probs)
            inference_records.append({"model_id": model, "seed": seed,
                                      "checkpoint_sha256": str(r.checkpoint_sha256),
                                      "physical_inference_microbatch": int(r.physical_inference_microbatch),
                                      "peak_cuda_allocated_MiB": float(stats["peak_cuda_allocated_MiB"])})
            # No shuffle, no seed-selection; each frozen checkpoint used once.
    require(len(seed_probs) == 40, "Expected 40 frozen inference outputs")
    ensembles, metric_df = f1.in_memory_five_seed_scoring(seed_probs, labels, threshold_map)
    require(len(ensembles) == 8 and len(metric_df) == 24, "Scoring cardinality mismatch")
    stats_df, pair_df = f4.evaluate_in_memory(keys, labels, ensembles, threshold_map)
    require(len(stats_df) == 72 and len(pair_df) == 84, "Statistical output cardinality drift")
    # Independent agreement between F1 point BA and F4 observed BA.
    observed = stats_df.loc[stats_df.block_length.eq(20),
                            ["model_id", "horizon_days", "observed_BA"]]
    merged = metric_df.merge(observed, on=["model_id", "horizon_days"],
                             how="inner", validate="one_to_one")
    require(len(merged) == 24 and
            np.allclose(merged.balanced_accuracy, merged.observed_BA, atol=1e-12),
            "F1/F4 primary balanced accuracy disagreement")

    longitudinal = []
    common_cols = ["ticker", "reference_date", "continuity_segment_id",
                   "reference_market_session_id"]
    for model in f1.MODEL:
        prob = ensembles[model].astype(np.float64)
        frame = meta.loc[:, common_cols].copy()
        frame.insert(0, "model_id", model)
        for j, horizon in enumerate(f1.HORIZONS):
            frame[f"target_{horizon}d"] = labels[:, j]
            frame[f"probability_{horizon}d"] = prob[:, j]
            frame[f"predicted_{horizon}d"] = (
                prob[:, j] >= threshold_map[(model, horizon)]).astype("int8")
        longitudinal.append(frame)
        payloads[f"ensemble_predictions/{model}_five_seed_mean.npz"] = _save_npz(
            probabilities=ensembles[model])
    long_df = pd.concat(longitudinal, ignore_index=True)
    require(len(long_df) == 42400 and long_df.model_id.nunique() == 8,
            "Long-form ensemble prediction rows changed")
    payloads.update({
        "evaluation/ensemble_predictions.parquet": _save_parquet(long_df),
        "evaluation/classification_metrics.csv": _save_csv(metric_df),
        "evaluation/segment_stratified_bootstrap.csv": _save_csv(stats_df),
        "evaluation/exploratory_pairwise_BA.csv": _save_csv(pair_df),
        "evaluation/inference_register.csv": _save_csv(pd.DataFrame(inference_records)),
    })
    manifest_metadata = {
        "model_families": list(f1.MODEL), "seeds": list(f1.SEEDS),
        "horizons": list(f1.HORIZONS),
        "eligible_endpoints": 5300, "eligible_dates": 265,
        "segment_date_counts": [48, 217], "stock_count": 20,
        "seed_checkpoint_outputs": 40, "five_seed_ensembles": 8,
        "metric_rows": 24, "bootstrap_rows": 72, "primary_hypotheses": 24,
        "pairwise_rows": 84, "prediction_rows": 42400,
        "frozen_inference_plan_sha256": f1.HASH["plan"],
        "frozen_thresholds_sha256": f1.HASH["thresholds"],
        "candidate_key_sha256": f1.HASH["cohort"],
        "protocol_deviation": "POST_FREEZE_COHORT_AND_RESAMPLING_AMENDMENT",
        "test_scoring_permission": "MUST_BE_SUPPLIED_EXTERNALLY_BY_APPROVED_RELEASE",
    }
    manifest = _manifest(payloads, manifest_metadata)
    payloads["artifact_manifest.json"] = (
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(len(payloads) == manifest["artifact_count"] + 1,
            "Artifact manifest count mismatch")
    return payloads, manifest


def blocked_live_entrypoints():
    raise PermissionError("DENIED — F9 cannot read or commit real final-test data")


open_sealed_test = blocked_live_entrypoints
commit_real_artifacts = blocked_live_entrypoints
execute_final_test = blocked_live_entrypoints


def inspect_ro(wsl: bool):
    print("=" * 106)
    print("STEP 36I-F9 — PRODUCTION COMPUTATION ASSEMBLY / READ-ONLY SOURCE AND ENVIRONMENT AUDIT")
    print("=" * 106)
    print("[1] F1–F8 IMMUTABLE SOURCES AND FROZEN MANUSCRIPT CONTRACT")
    for idx in SOURCE_NAMES:
        check(ROOT / SOURCE_NAMES[idx], SOURCE_HASHES[idx], f"Step36I-F{idx}")
    check(BUNDLE, BUNDLE_SHA, "E2 review bundle")
    validate_interfaces_static()
    print("PASS — eight frozen review sources, original source bundle and Python interfaces")
    print("[2] IN-MEMORY PRODUCTION COMPUTATIONAL BINDINGS")
    require(assemble_from_preloaded.__code__.co_argcount == 3,
            "Preloaded assembly API must require raw frame, history, upstream")
    require(set(assemble_from_preloaded.__code__.co_names) >=
            {"materialize_preloaded_from_frozen_helpers", "frozen_inference_from_preloaded_X",
             "in_memory_five_seed_scoring", "evaluate_in_memory"},
            "Mandatory materialization/model/scoring/statistical call missing")
    # Pure serialization, independent of any test-file existence.
    m = _manifest({"SYNTHETIC_ONLY_fixture.bin": b"test"}, {"synthetic_only": True})
    require(m["artifact_count"] == 1 and m["artifacts"][0]["bytes"] == 4,
            "Artifact manifest in-memory hash failed")
    require(m["artifacts"][0]["sha256"] == hashlib.sha256(b"test").hexdigest(),
            "Artifact manifest digest mismatch")
    for fn in (open_sealed_test, commit_real_artifacts, execute_final_test):
        try:
            fn()
        except PermissionError:
            continue
        raise RuntimeError("FAIL: review CLI/production entrypoint unexpectedly enabled")
    print("PASS — F1 materializer/40 checkpoint iterations/scorer/F4 engine bound; manifest serializer checked")
    print("[3] UNAPPROVED PROTOCOL AND REAL OUTPUT NAMESPACE")
    if wsl:
        check(STEP35, STEP35_SHA, "Original Step35 scientific contract")
        f1 = load_local_helper(1)
        state = f1.read_only_access_check()
        require(state.get("raw_test_filesystem_open_count_completed") == 2 and
                state.get("final_test_scored") is False, "Historic access history changed")
        keys = f1.candidate(pd.read_csv(AMENDED_KEYS))
        plan = pd.read_csv(CHECKPOINT_PLAN)
        thresholds = f1.plan_thresholds(plan, pd.read_csv(THRESHOLD_PLAN))
        require(len(keys) == 5300 and len(thresholds) == 24, "Candidate contract drift")
        f1.confirm_legacy_output_isolation()
        seal = json.loads((EXP / "manifests/10_final_test_seal.json").read_text("utf-8"))
        require(Path(seal["file_path"]).name == "10_final_test_locked_panel.parquet",
                "Frozen test-file identity drift")
        # We never call test_path.exists()/stat()/open(), and never parse the parquet.
        for p in (
            EXP / "feature_data/step12_volatility_regimes/final_development/12_final_development_scaled_regimes.parquet",
            EXP / "feature_data/step12_volatility_regimes/final_development/12_final_development_regime_thresholds.csv",
            ROOT / "step26_stage1_runner_dryrun_v3.py",
            ROOT / "step12_volatility_regimes_v3.py",
        ):
            require(p.is_file() and not p.is_symlink(), f"Missing future-run dependency: {p}")
        print("PASS — WSL source/data dependencies present; old ledger 2/0; amended namespace empty")
    else:
        print("PASS — portable-only mode did not access live WSL experiment")
    print("[4] REAL EXECUTION AND TRANSACTION ARE EXPLICITLY NOT AVAILABLE")
    print("REMAINING — formal cohort/statistical amendment decision and third-read authorization")
    print("REMAINING — independently reviewed real-data journal/one-read loader and atomic commit")
    print("REMAINING — independent F9 implementation review before any approved new launcher")
    print("=" * 106)
    print("STEP 36I-F9 PASS — READ-ONLY ASSEMBLY REVIEW; NOT A LIVE PRODUCTION LAUNCHER")
    print("Sealed raw test opened: FALSE | GPU executed: FALSE | Real test scored: FALSE")
    print("Amendment approved: FALSE | Additional read authorized: FALSE | Production CLI: DENIED")
    print("=" * 106)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--offline-preflight", action="store_true")
    args = parser.parse_args()
    inspect_ro(wsl=args.offline_preflight)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, AssertionError, OSError) as exc:
        print(f"STEP36I-F9 FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
