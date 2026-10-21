#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F6: integrated synthetic evaluation-to-transaction review.

ONLY SUPPORTED USE:
  python step36i_f6_end_to_end_synthetic_transaction_REVIEW_ONLY.py --offline-preflight
  python step36i_f6_end_to_end_synthetic_transaction_REVIEW_ONLY.py --self-test

Reads previously reviewed F1--F5 code as modules and the E2 frozen audit bundle.
Builds a full-sized but entirely synthetic panel of model probabilities and
labels, runs ACTUAL F1 five-seed aggregation and F4 segment-stratified inference,
and commits their CSV outputs using F3's journal and atomic/no-replace operations.
All writes take place under a disposable TemporaryDirectory outside the project.

NO HELD-OUT TEST FILE LOADER IS PROVIDED. No production CLI exists. This file
DOES NOT grant protocol approval, modify the historical two-read ledger, or
implement real-test transaction orchestration. A separate reviewed release and
explicit authorization would be needed for any further raw-test access.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid
import zipfile

DENIED = {"--execute", "--production", "--worker", "--read-test", "--score-test",
          "--resume", "--approve", "--authorize", "--commit-real", "--run",
          "--materialize", "--final-test", "--enable", "--test-parquet"}
if DENIED.intersection(sys.argv[1:]):
    raise SystemExit("DENIED: Step 36I-F6 is synthetic-only; real-test access is forbidden")

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
EXP = WORK.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
MAN = EXP / "manifests"
AUD = EXP / "audits"
E2 = WORK / "step36i_e2_source_review_bundle.zip"
F1 = WORK / "step36i_f1_amended_production_runner_REVIEW_ONLY.py"
F2 = WORK / "step36i_f2_independent_source_review_READ_ONLY.py"
F3 = WORK / "step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py"
F4 = WORK / "step36i_f4_segment_stratified_inference_REVIEW_ONLY.py"
F5 = WORK / "step36i_f5_independent_statistical_audit_READ_ONLY.py"
ACCESS = MAN / "36_final_test_access_consumption_state.json"
STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"
E4 = AUD / "step36i_e4_synthetic_artifact_transaction_draft/36i_e4_synthetic_transaction_DRAFT.json"
HASH = {
    "E2": "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851",
    "F1": "e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556",
    "F2": "ec6400f0f356b7a9888c443890b333f6fa6c6d80f6d29d700625eba4335f00bc",
    "F3": "b7c28f536a57653f10cc74155b38941b7e2e86b508089e5a3378bf9ff97b7539",
    "F4": "421cb3e8b83f96efaef1f3e0578e2cb7513c728aef36bda68cdc9216b07b4b10",
    "F5": "47a5ed8044493f4d09cbafa16fea5a62482d13b27856da945bd5d6e04f5e8ee3",
    "STEP35": "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
    "E4": "30fbd08cecba4e65eb427358c6a24c9d28b984bdf889afc83668a6c9952ab925",
    "COHORT": "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
    "THRESHOLDS": "79dc80e0e3b18427cebd77742ff2d10b87aebcdd9ea1e459fed30ac980d0e24f",
}


def need(condition: bool, why: str) -> None:
    if not bool(condition):
        raise RuntimeError(why)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked(path: Path, want: str, label: str) -> None:
    need(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}: {path}")
    need(sha(path) == want, f"SHA-256 mismatch for {label}")


def imported(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, f"Cannot import {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_frozen_bundle():
    for name, path in (("E2", E2), ("F1", F1), ("F2", F2),
                       ("F3", F3), ("F4", F4), ("F5", F5)):
        checked(path, HASH[name], name)
    with zipfile.ZipFile(E2) as z:
        need(z.testzip() is None, "E2 archive corrupted")
        names = z.namelist()
        need(len(names) == len(set(names)) == 18 and all("/" not in n and ".." not in n for n in names),
             "E2 bundle inventory drift")
        raw = {name: z.read(name) for name in names}
    for file, name in (("candidate_keys_DRAFT.csv", "COHORT"),
                       ("frozen_thresholds.csv", "THRESHOLDS"),
                       ("step35_contract.json", "STEP35")):
        need(digest(raw[file]) == HASH[name], f"E2 member {file} hash drift")
    amendment = json.loads(raw["amendment_review_DRAFT.json"])
    protocol = json.loads(raw["protocol_deviation_DRAFT.json"])
    need(amendment.get("approval_granted") is False and
         amendment.get("third_raw_test_read_authorized") is False and
         protocol.get("status") == "DRAFT_NOT_APPROVED", "Protocol approval gate drift")
    return raw


def audit_live_marker():
    checked(STEP35, HASH["STEP35"], "Step35 frozen scientific contract")
    checked(E4, HASH["E4"], "E4 synthetic transaction")
    need(ACCESS.is_file() and not ACCESS.is_symlink(), "Historical access state absent")
    before = sha(ACCESS)
    marker = json.loads(ACCESS.read_text("utf-8"))
    need(marker.get("state") == "RAW_TEST_REOPENED_FOR_TECHNICAL_RECOVERY" and
         marker.get("raw_test_filesystem_open_count_completed") == 2 and
         marker.get("final_test_scored") is False,
         "Historical test-state mismatch: require 2 prior reads, 0 scored")
    need(marker.get("step35_manifest_sha256") == HASH["STEP35"],
         "Historical marker/Step35 link drift")
    need(marker.get("expected_raw_test_sha256") ==
         "ab325ede85fbb6d7819e85472274b82d151d350f39a872a17747c6978b1dddae" ==
         marker.get("observed_raw_test_sha256"), "Historical test SHA identity drift")
    return before


def synthetic_full_outputs(bundle, f1, f4):
    keys = f4.key_frame(pd.read_csv(io.BytesIO(bundle["candidate_keys_DRAFT.csv"])))
    thresholds = f4.threshold_map(pd.read_csv(io.BytesIO(bundle["frozen_thresholds.csv"])))
    y, seed_probs = f4.build_synthetic_fixture(keys, thresholds)
    need(y.shape == (5300, 3) and len(seed_probs) == 40, "Synthetic fixture interface drift")
    ensembles, metrics = f1.in_memory_five_seed_scoring(seed_probs, y, thresholds)
    infer, pairs = f4.evaluate_in_memory(keys, y, ensembles, thresholds)
    need((len(ensembles), len(metrics), len(infer), len(pairs)) == (8, 24, 72, 84),
         "End-to-end result cardinality drift")
    need(infer.loc[infer.primary_test, "q_BH_primary_only"].notna().sum() == 24 and
         infer.loc[~infer.primary_test, "q_BH_primary_only"].isna().all(),
         "Primary 24-only BH correction drift")
    primary = infer[infer.block_length.eq(20)].set_index(["model_id", "horizon_days"])
    for row in metrics.itertuples(index=False):
        observed = float(primary.loc[(row.model_id, int(row.horizon_days)), "observed_BA"])
        need(abs(observed - float(row.balanced_accuracy)) <= 1e-12,
             f"F1/F4 BA inconsistency: {row.model_id}/{row.horizon_days}")
    prediction_frames = []
    for model in f4.MODELS:
        ep = np.asarray(ensembles[model], dtype=np.float32)
        df = keys.copy()
        df.insert(0, "model_id", model)
        for j, horizon in enumerate(f4.HORIZONS):
            df[f"SYNTHETIC_ONLY_prob_{horizon}d"] = ep[:, j]
        prediction_frames.append(df)
    preds = pd.concat(prediction_frames, ignore_index=True)
    need(len(preds) == 42400 and preds.model_id.value_counts().eq(5300).all(),
         "Long-form 42,400 row synthetic prediction shape changed")
    # Explicit synthetic data labeling in EVERY written file.
    metrics.insert(0, "synthetic_only", True)
    infer.insert(0, "synthetic_only", True)
    pairs.insert(0, "synthetic_only", True)
    preds.insert(0, "synthetic_only", True)
    payloads = {
        "SYNTHETIC_ONLY_ensemble_predictions.csv": preds.to_csv(index=False, float_format="%.17g").encode("utf-8"),
        "SYNTHETIC_ONLY_frozen_threshold_metrics.csv": metrics.to_csv(index=False, float_format="%.17g").encode("utf-8"),
        "SYNTHETIC_ONLY_segment_bootstrap_FDR.csv": infer.to_csv(index=False, float_format="%.17g").encode("utf-8"),
        "SYNTHETIC_ONLY_exploratory_pairwise.csv": pairs.to_csv(index=False, float_format="%.17g").encode("utf-8"),
    }
    counts = {"predictions": len(preds), "metrics": len(metrics),
              "bootstrap_rows": len(infer), "pairwise_rows": len(pairs),
              "models": len(ensembles), "synthetic_seed_outputs": len(seed_probs)}
    return payloads, counts


def atomic_synthetic_transaction(f3, root: Path, payloads: dict, counts: dict,
                                 name: str, inject_crash: bool):
    need(name.startswith("SYNTHETIC_ONLY_"), "Non-synthetic transaction target refused")
    need(root.is_dir() and not root.is_symlink(), "Unsafe synthetic root")
    # Never write any transaction fixture inside the live experiment.
    resolved = root.resolve()
    live = EXP.resolve()
    need(resolved != live and live not in resolved.parents,
         "Synthetic transaction must not touch the experiment")
    final = root / name
    need(not final.exists() and not final.is_symlink(), "No-replace guard: destination exists")
    staging = root / (".SYNTHETIC_ONLY_STAGE_" + uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    entries = []
    for fname, blob in sorted(payloads.items()):
        need(fname.startswith("SYNTHETIC_ONLY_") and "/" not in fname, "Unsafe synthetic file name")
        f3.write_unique(staging / fname, blob)
        entries.append({"file": fname, "bytes": len(blob), "sha256": digest(blob)})
    manifest = {
        "synthetic_only": True,
        "status": "SYNTHETIC_INTEGRATION_REVIEW_NOT_PRODUCTION",
        "protocol_amendment_approved": False,
        "third_raw_test_read_authorized": False,
        "real_test_scored": False,
        "code_hashes": {label: HASH[label] for label in ("F1", "F3", "F4", "F5", "STEP35")},
        "counts": counts,
        "artifacts": entries,
    }
    f3.write_unique(staging / "SYNTHETIC_ONLY_manifest.json", f3.canonical(manifest) + b"\n")
    f3.fsync_directory(staging)
    if inject_crash:
        raise RuntimeError("SYNTHETIC_INJECTED_CRASH_BEFORE_ATOMIC_COMMIT")
    f3.rename_dir_no_replace(staging, final)
    need(final.is_dir() and not staging.exists(), "Atomic directory commit failed")
    for entry in entries:
        p = final / entry["file"]
        need(p.is_file() and p.stat().st_size == entry["bytes"] and sha(p) == entry["sha256"],
             "Committed synthetic artifact rehash failed")
    need(json.loads((final / "SYNTHETIC_ONLY_manifest.json").read_text("utf-8")) == manifest,
         "Committed manifest differs from prepared manifest")
    return final


def inspect_semantic_outputs(final: Path, counts: dict) -> None:
    preds = pd.read_csv(final / "SYNTHETIC_ONLY_ensemble_predictions.csv")
    metrics = pd.read_csv(final / "SYNTHETIC_ONLY_frozen_threshold_metrics.csv")
    stat = pd.read_csv(final / "SYNTHETIC_ONLY_segment_bootstrap_FDR.csv")
    pair = pd.read_csv(final / "SYNTHETIC_ONLY_exploratory_pairwise.csv")
    for label, frame, n in (("predictions", preds, 42400), ("metrics", metrics, 24),
                            ("bootstrap_rows", stat, 72), ("pairwise_rows", pair, 84)):
        need(len(frame) == n == counts[label], f"Post-commit semantic rows mismatch: {label}")
        need("synthetic_only" in frame and frame.synthetic_only.astype(str).str.lower().eq("true").all(),
             f"Missing synthetic-only labeling: {label}")
    need(preds["model_id"].nunique() == 8 and preds.groupby("model_id").size().eq(5300).all(),
         "Committed ensemble model cluster mismatch")
    need(stat.loc[stat.primary_test.astype(str).str.lower().eq("true"),
                  "q_BH_primary_only"].notna().sum() == 24,
         "Committed primary q-values mismatch")
    need(stat.loc[~stat.primary_test.astype(str).str.lower().eq("true"),
                  "q_BH_primary_only"].isna().all(),
         "Committed sensitivity FDR must be absent")


def transaction_test(f3, payloads, counts):
    with tempfile.TemporaryDirectory(prefix="step36i_f6_SYNTHETIC_ONLY_") as dirname:
        root = Path(dirname)
        journal = f3.FixtureJournal(root / "SYNTHETIC_ONLY_access_journal.jsonl")
        journal.append("SYNTHETIC_INIT")
        journal.append("SYNTHETIC_READ_SLOT_RESERVED")
        journal.append("SYNTHETIC_READ_SIMULATED")
        try:
            journal.append("SYNTHETIC_READ_SLOT_RESERVED")
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Synthetic journal accepted duplicate read reservation")
        try:
            atomic_synthetic_transaction(f3, root, payloads, counts, "SYNTHETIC_ONLY_FAILED", True)
        except RuntimeError as exc:
            need("SYNTHETIC_INJECTED_CRASH_BEFORE_ATOMIC_COMMIT" in str(exc),
                 "Crash-injection exception unexpected")
        else:
            raise RuntimeError("Injected crash did not fail transaction")
        need(not (root / "SYNTHETIC_ONLY_FAILED").exists(),
             "Failed synthetic artifact transaction was published")
        committed = atomic_synthetic_transaction(
            f3, root, payloads, counts, "SYNTHETIC_ONLY_COMMITTED", False)
        manifest_before = sha(committed / "SYNTHETIC_ONLY_manifest.json")
        inspect_semantic_outputs(committed, counts)
        try:
            atomic_synthetic_transaction(f3, root, payloads, counts,
                                         "SYNTHETIC_ONLY_COMMITTED", False)
        except RuntimeError as exc:
            need("No-replace guard" in str(exc), "Unexpected replay rejection reason")
        else:
            raise RuntimeError("Committed transaction was overwritten")
        need(sha(committed / "SYNTHETIC_ONLY_manifest.json") == manifest_before,
             "Replay mutated committed manifest")
        journal.append("SYNTHETIC_COMMITTED")
        need(len(journal.read_events()) == 4, "Synthetic journal final state mismatch")
        raw = journal.path.read_bytes()
        journal.path.write_bytes(raw.replace(b"SYNTHETIC_READ_SIMULATED", b"SYNTHETIC_READ_TAMPERED"))
        try:
            journal.read_events()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Synthetic journal tampering not rejected")
        return sum(e["bytes"] for e in json.loads((committed / "SYNTHETIC_ONLY_manifest.json").read_text())["artifacts"])


def run(wsl: bool) -> None:
    print("=" * 110)
    print("STEP 36I-F6 — INTEGRATED EVALUATION AND ATOMIC ARTIFACT TRANSACTION (SYNTHETIC ONLY)")
    print("=" * 110)
    print("[1] F1–F5 SOURCE FINGERPRINTS, E2 CONTRACT AND UNAPPROVED ACCESS STATUS")
    bundle = audit_frozen_bundle()
    before = audit_live_marker() if wsl else None
    print("PASS — reviewed sources/Step35 identity intact; protocol unapproved; real access disabled")
    if wsl:
        # Enforces that all 7 proposed real-data output paths remain vacant.
        f1_probe = imported(F1, "f6_review_f1_namespace")
        f1_probe.confirm_legacy_output_isolation()
        print("PASS — historical marker says 2 reads, 0 scoring; 7 amended real output paths vacant")
    print("[2] ACTUAL F1 FIVE-SEED SCORER + F4 BOOTSTRAP/FDR ON SYNTHETIC 5300-ROW PANEL")
    f1 = imported(F1, "f6_integrated_f1")
    f3 = imported(F3, "f6_integrated_f3")
    f4 = imported(F4, "f6_integrated_f4")
    payloads, counts = synthetic_full_outputs(bundle, f1, f4)
    print("PASS — 40 seed arrays -> 8 ensembles -> 42,400 prediction rows + 24 metrics")
    print("PASS — 72 bootstrap/FDR rows, 24 BH primary q-values, 84 exploratory pairs")
    print("[3] F3 JOURNAL AND ATOMIC-NO-REPLACE COMMIT OF REAL-SHAPED SYNTHETIC OUTPUTS")
    nbytes = transaction_test(f3, payloads, counts)
    print(f"PASS — {nbytes:,} synthetic result bytes committed then independently rehashed in tempdir")
    print("PASS — committed CSV rows read back; duplicate reservation, precommit crash, replay, tamper rejected")
    print("[4] ORIGINAL ARTIFACT IMMUTABILITY / PRODUCTION ENTRYPOINT DENIAL")
    for forbidden in (f1.execute_production, f1.open_sealed_test,
                      f4.open_test, f4.run_real_inference):
        try:
            forbidden()
        except PermissionError:
            pass
        else:
            raise RuntimeError("A real execution gate was unexpectedly callable")
    if wsl:
        need(sha(ACCESS) == before, "Historical access marker changed during F6")
        checked(STEP35, HASH["STEP35"], "Post-F6 Step35")
        checked(E4, HASH["E4"], "Post-F6 E4")
        f1.confirm_legacy_output_isolation()
        print("PASS — access marker and frozen files unchanged; amended namespace untouched")
    else:
        print("PASS — no live experiment opened during portable synthetic self-test")
    print("=" * 110)
    print("STEP 36I-F6: PASS — FULL-SIZED SYNTHETIC IN-MEMORY SCORING -> STATS -> ATOMIC COMMIT")
    print("Real test read: FALSE | Real scoring: FALSE | GPU: FALSE | Production runner: DISABLED")
    print("Protocol amendment approved: FALSE | Third raw-test access authorized: FALSE")
    print("REMAINING: reviewed real loader + production durable journal/transaction; formal approval and explicit authorization")
    print("=" * 110)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--offline-preflight", action="store_true")
    g.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    run(wsl=bool(args.offline_preflight))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"F6 FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
