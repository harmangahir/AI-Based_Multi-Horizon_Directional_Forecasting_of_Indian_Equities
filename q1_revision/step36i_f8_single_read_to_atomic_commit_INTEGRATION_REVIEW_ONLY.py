#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F8 — synthetic source-byte -> scoring -> statistics -> commit integration.

ONLY modes: --self-test (portable), --offline-preflight (read-only WSL audit).

F8 is NOT a production launcher. It creates ONE synthetic NPZ source in a
TemporaryDirectory outside the experiment, reserves a durable synthetic access
journal before opening that fixture once, decodes its bytes in memory, runs the
reviewed F1 scorer and F4 segment bootstrap/FDR, and commits artifacts through
F3/F6's atomic no-replace transaction. F8 verifies byte-for-byte equality of
these output artifacts against F6's independent synthetic replay. The real
held-out parquet is never located, stat'ed, opened, or read.

Frozen Step35 is not modified; the proposed Step36I cohort remains unapproved;
a third real-test file read is NOT authorized. Production gates do not exist.
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
import zipfile

_DENIED = {"--execute", "--production", "--worker", "--read-test", "--score-test",
           "--resume", "--authorize", "--approve", "--run", "--commit-real",
           "--materialize", "--final-test", "--enable", "--test-parquet"}
if _DENIED.intersection(sys.argv[1:]):
    raise SystemExit("DENIED — Step36I-F8 cannot access or evaluate held-out data")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
E2 = ROOT / "step36i_e2_source_review_bundle.zip"
F1 = ROOT / "step36i_f1_amended_production_runner_REVIEW_ONLY.py"
F3 = ROOT / "step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py"
F4 = ROOT / "step36i_f4_segment_stratified_inference_REVIEW_ONLY.py"
F6 = ROOT / "step36i_f6_end_to_end_synthetic_transaction_REVIEW_ONLY.py"
F7 = ROOT / "step36i_f7_one_read_loader_journal_integration_REVIEW_ONLY.py"
HASHES = {
    "F1": "e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556",
    "F3": "b7c28f536a57653f10cc74155b38941b7e2e86b508089e5a3378bf9ff97b7539",
    "F4": "421cb3e8b83f96efaef1f3e0578e2cb7513c728aef36bda68cdc9216b07b4b10",
    "F6": "f075401a94c3bd2a971d4ade7c71c28207754d5c5810b40462e0b2bc88d62bad",
    "F7": "b01bd3ce80194a4981a3c518fcf2e7f46948819715a51319843fa50630a54685",
    "E2": "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851",
}


def require(check, msg):
    if not bool(check):
        raise RuntimeError(msg)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def disk_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def import_review(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    require(spec is not None and spec.loader is not None, f"Cannot import {alias}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses in F7 need the module to be registered before execution.
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def verify_sources():
    for name, path in (("F1", F1), ("F3", F3), ("F4", F4),
                       ("F6", F6), ("F7", F7), ("E2", E2)):
        require(path.is_file() and not path.is_symlink(), f"Missing/unsafe reviewed source: {name}")
        require(disk_sha(path) == HASHES[name], f"Reviewed SHA mismatch: {name}")
    with zipfile.ZipFile(E2, 'r') as z:
        require(z.testzip() is None and len(z.namelist()) == 18, "Frozen E2 bundle drift")
        archive = {name: z.read(name) for name in z.namelist()}
    protocol = json.loads(archive["protocol_deviation_DRAFT.json"])
    amend = json.loads(archive["amendment_review_DRAFT.json"])
    require(protocol["status"] == "DRAFT_NOT_APPROVED" and
            amend["approval_granted"] is False and
            amend["third_raw_test_read_authorized"] is False,
            "Protocol amendment unexpectedly changed")
    require(sha(archive["candidate_keys_DRAFT.csv"]) ==
            "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
            "Candidate key identity changed")
    return archive


def source_fixture(archive, f4) -> tuple[pd.DataFrame, dict, bytes]:
    keys = f4.key_frame(pd.read_csv(io.BytesIO(archive["candidate_keys_DRAFT.csv"])))
    threshold = f4.threshold_map(pd.read_csv(io.BytesIO(archive["frozen_thresholds.csv"])))
    y, probs = f4.build_synthetic_fixture(keys, threshold)
    tensor = np.stack([probs[(m, s)] for m in f4.MODELS for s in f4.SEEDS], axis=0)
    require(tensor.shape == (40, 5300, 3), "Synthetic fixture tensor drift")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, labels=y, forty_seed_probabilities=tensor)
    return keys, threshold, buffer.getvalue()


def results_from_decoded_bytes(verified_data: bytes, keys, thresholds, f1, f4):
    # The archived source is never parsed by pathname; verified bytes only.
    with np.load(io.BytesIO(verified_data), allow_pickle=False) as a:
        require(set(a.files) == {"labels", "forty_seed_probabilities"},
                "Synthetic decoded interface altered")
        y = np.asarray(a["labels"])
        all_probs = np.asarray(a["forty_seed_probabilities"])
    require(y.shape == (5300, 3) and np.isin(y, [0, 1]).all(), "Decoded labels invalid")
    require(all_probs.shape == (40, 5300, 3) and
            np.isfinite(all_probs).all() and
            np.all((all_probs >= 0) & (all_probs <= 1)),
            "Decoded probabilities invalid")
    seeds = {(m, seed): all_probs[k] for k, (m, seed) in enumerate(
        (m, s) for m in f4.MODELS for s in f4.SEEDS)}
    ensembles, metrics = f1.in_memory_five_seed_scoring(seeds, y, thresholds)
    stat, pairs = f4.evaluate_in_memory(keys, y, ensembles, thresholds)
    require((len(ensembles), len(metrics), len(stat), len(pairs)) == (8, 24, 72, 84),
            "Decoded pipeline cardinality mismatch")
    dates = pd.concat([
        keys.assign(model_id=m, **{f"SYNTHETIC_ONLY_prob_{h}d": ensembles[m][:, j]
                                    for j, h in enumerate(f4.HORIZONS)})
        for m in f4.MODELS
    ], ignore_index=True)
    # Identical physical column layout to reviewed F6 output, which inserts
    # model_id at the start of each model's frame.
    dates = dates[["model_id", *list(keys.columns),
                   *(f"SYNTHETIC_ONLY_prob_{h}d" for h in f4.HORIZONS)]]
    for df in (metrics, stat, pairs, dates):
        df.insert(0, "synthetic_only", True)
    payloads = {
        "SYNTHETIC_ONLY_ensemble_predictions.csv": dates.to_csv(index=False, float_format="%.17g").encode(),
        "SYNTHETIC_ONLY_frozen_threshold_metrics.csv": metrics.to_csv(index=False, float_format="%.17g").encode(),
        "SYNTHETIC_ONLY_segment_bootstrap_FDR.csv": stat.to_csv(index=False, float_format="%.17g").encode(),
        "SYNTHETIC_ONLY_exploratory_pairwise.csv": pairs.to_csv(index=False, float_format="%.17g").encode(),
    }
    counts = dict(predictions=42400, metrics=24, bootstrap_rows=72,
                  pairwise_rows=84, models=8, synthetic_seed_outputs=40)
    return payloads, counts


def denied_real_action():
    raise PermissionError("F8 has no real-test loader or production authorization")


def run(wsl=False):
    print("=" * 109)
    print("STEP 36I-F8 — SINGLE-READ -> FROZEN SCORING -> STATS -> ATOMIC COMMIT, SYNTHETIC ONLY")
    print("=" * 109)
    print("[1] IMMUTABLE REVIEWED SOURCES / FROZEN CONTRACT / LIVE ACCESS SAFETY")
    archive = verify_sources()
    f6 = import_review(F6, "step36i_f8_reviewed_f6")
    f1 = import_review(F1, "step36i_f8_reviewed_f1")
    f3 = import_review(F3, "step36i_f8_reviewed_f3")
    f4 = import_review(F4, "step36i_f8_reviewed_f4")
    f7 = import_review(F7, "step36i_f8_reviewed_f7")
    # F6 performs secondary verification of F2/F5, Step35, E4, and bundle.
    f6.audit_frozen_bundle()
    original_marker_hash = f6.audit_live_marker() if wsl else None
    if wsl:
        f1.confirm_legacy_output_isolation()
    print("PASS — F1–F7/E2 source contracts; unapproved protocol; live marker guarded" if wsl
          else "PASS — F1–F7/E2 source contracts; synthetic-only portable run")
    print("[2] ONE SYNTHETIC FILE DESCRIPTOR -> VERIFIED NPZ BYTES -> REVIEWED F1 AND F4")
    keys, thresholds, source_bytes = source_fixture(archive, f4)
    with tempfile.TemporaryDirectory(prefix="STEP36I_F8_SYNTHETIC_ONLY_") as directory:
        root = Path(directory)
        path = root / "SYNTHETIC_ONLY_seed_predictions.npz"
        path.write_bytes(source_bytes)
        journal = f7.OneAttemptJournal(root / "SYNTHETIC_ONLY_access_state.jsonl")
        opens = []
        def one_open(file, flags):
            opens.append(str(file))
            return os.open(file, flags)
        evidence = f7.read_bytes_exactly_once_fixture(path, sha(source_bytes), journal, opener=one_open)
        require(len(opens) == 1 and len(journal.records()) == 2, "Single-read reservation failed")
        require(evidence.sha256 == sha(source_bytes), "Single-read digest mismatch")
        payloads, counts = results_from_decoded_bytes(evidence.file_bytes, keys, thresholds, f1, f4)
        # Compare the fully decoded scoring outputs against F6's independently
        # replayed synthetic baseline. Equality requires the actual decoded
        # source bytes to be used, not a separate fabricated result array.
        baseline, baseline_counts = f6.synthetic_full_outputs(archive, f1, f4)
        require(counts == baseline_counts, "Synthetic result shape versus F6 oracle changed")
        for fname in sorted(payloads):
            require(payloads[fname] == baseline[fname], f"Decoded-to-score oracle mismatch: {fname}")
        print("PASS — durable synthetic reservation preceded exactly 1 fixture open; BytesIO decode only")
        print("PASS — 40 decoded seeds -> 8 ensembles -> 42,400 predictions; 24 metrics; 72 FDR; 84 pairs")
        print("[3] REVIEWED F3/F6 ATOMIC COMMIT OF THOSE DECODED RESULTS")
        output = f6.atomic_synthetic_transaction(
            f3, root, payloads, counts, "SYNTHETIC_ONLY_COMMITTED", inject_crash=False)
        f6.inspect_semantic_outputs(output, counts)
        manifest = json.loads((output / "SYNTHETIC_ONLY_manifest.json").read_text())
        total = sum(item["bytes"] for item in manifest["artifacts"])
        try:
            f6.atomic_synthetic_transaction(f3, root, payloads, counts,
                                             "SYNTHETIC_ONLY_COMMITTED", inject_crash=False)
        except RuntimeError as e:
            require("No-replace guard" in str(e), "Unexpected replay rejection")
        else:
            raise RuntimeError("Replay unexpectedly accepted")
        # The source read journal cannot be re-used even after artifact commit.
        try:
            f7.read_bytes_exactly_once_fixture(path, sha(source_bytes), journal, opener=one_open)
        except RuntimeError as e:
            require("no retry" in str(e).lower(), "Unexpected duplicate-read rejection")
        else:
            raise RuntimeError("Duplicate-read unexpectedly accepted")
        require(len(opens) == 1, "Replay bypassed read-once contract")
        require(len(journal.records()) == 2, "Synthetic access journal changed after commit")
        print(f"PASS — {total:,} synthetic output bytes atomically committed, reopened, and independently verified")
        print("PASS — duplicate source read and duplicate artifact commit both rejected")
    print("[4] RECHECK HISTORICAL ACCESS LEDGER / PRODUCTION GATES")
    for fn in (denied_real_action, f1.execute_production,
               f7.open_sealed_test_production, f7.atomic_live_commit):
        try:
            fn()
        except PermissionError:
            pass
        else:
            raise RuntimeError("Production gate unexpectedly enabled")
    if wsl:
        require(f6.sha(f6.ACCESS) == original_marker_hash,
                "Historical access marker modified")
        f6.checked(f6.STEP35, f6.HASH["STEP35"], "Step35 unchanged after F8")
        f1.confirm_legacy_output_isolation()
    print("PASS — real execution denied; frozen artifacts/output namespace and access history unchanged")
    print("=" * 109)
    print("STEP 36I-F8 PASS — COMPLETE SYNTHETIC BYTE-LEVEL INTEGRATION ONLY")
    print("Real raw test opened: FALSE | Real test scored: FALSE | GPU: FALSE")
    print("Amendment approved: FALSE | Third read authorized: FALSE | Production CLI: DENIED")
    print("REMAINING: implement/review real production launcher and journal; formal amendment and third-read approval")
    print("=" * 109)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--offline-preflight", action="store_true")
    args = parser.parse_args()
    run(wsl=bool(args.offline_preflight))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"F8 FAIL-CLOSED: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1)
