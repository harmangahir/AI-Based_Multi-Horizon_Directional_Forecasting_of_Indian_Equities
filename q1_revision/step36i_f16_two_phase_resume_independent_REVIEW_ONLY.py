#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F16 — independent two-phase materialization-to-inference handoff review.

OFFLINE ONLY: --self-test or --offline-preflight. Both use generated synthetic
arrays, outcomes, and simulated seed probabilities. NO real checkpoint inference,
sealed-test read, test scoring, authorization, persistent production journal, or
production resume is implemented or permitted.

This review tests a distinct failure boundary: phase A has atomically committed
materialized arrays, a mock phase B fails, and phase B can be restarted from
that exact *hashed* committed handoff without accessing an upstream source.
The frozen inference plan's 40 rows, physical batch composition/order, and
small incomplete final batches are tested using simulated workers. Passing
this script does NOT constitute sign-off on any future real release.
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

# Refuse execution-oriented options before data/code access.
_DENIED = {"--execute", "--read-test", "--production", "--resume", "--score-test",
           "--authorize", "--approve", "--worker", "--enable", "--commit-real",
           "--open-test", "--final-test", "--materialize-test", "--live"}
if _DENIED.intersection(sys.argv[1:]):
    raise SystemExit("DENIED — F16 can only test temporary synthetic fixtures")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
BUNDLE = ROOT / "step36i_e2_source_review_bundle.zip"
CANDIDATE = EXP / "audits/step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv"
LEDGER = EXP / "manifests/36_final_test_access_consumption_state.json"
FROZEN = {
    "F1": ("step36i_f1_amended_production_runner_REVIEW_ONLY.py", "e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556"),
    "F3": ("step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py", "b7c28f536a57653f10cc74155b38941b7e2e86b508089e5a3378bf9ff97b7539"),
    "F4": ("step36i_f4_segment_stratified_inference_REVIEW_ONLY.py", "421cb3e8b83f96efaef1f3e0578e2cb7513c728aef36bda68cdc9216b07b4b10"),
    "F9": ("step36i_f9_preloaded_production_assembly_REVIEW_ONLY.py", "8074cfe7cd1de9cf17609d4cef71aff27f910347da3ad86f3861a3cf7ad1373b"),
    "F11": ("step36i_f11_upstream_provenance_audit_READ_ONLY.py", "3405cb13a89d7c2403a66b5f6119b18a13c42f886d7f8d05aa80e94b09709ea6"),
    "F14": ("step36i_f14_approval_bound_production_integration_REVIEW_ONLY.py", "bc34722a3582812c0d5257581e04e320bd2e59eb67f4a037a4f473c4238f068f"),
    "F15": ("step36i_f15_durable_materialization_handoff_REVIEW_ONLY.py", "a2e2a05ef363f389c5208dceb4eef518227f050a8294d03326f962767fc8289c"),
}
BUNDLE_SHA = "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851"
CANDIDATE_SHA = "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65"
STEP35_SHA = "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7"
EXPECTED = {
    "arrays/sequence_X.npy", "arrays/directions_SYNTHETIC_ONLY.npy",
    "arrays/metadata_keys.csv",
}


def require(ok: bool, message: str):
    if not ok:
        raise RuntimeError(message)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for b in iter(lambda: fd.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def checked(path: Path, expected: str, label: str):
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe {label}: {path}")
    require(sha_file(path) == expected, f"SHA-256 drift: {label}")


def imports_checked():
    for label, (filename, wanted) in FROZEN.items():
        checked(ROOT / filename, wanted, label)
    checked(BUNDLE, BUNDLE_SHA, "E2 bundle")
    mods = {}
    for label in ("F1", "F3", "F4", "F15"):
        filename = FROZEN[label][0]
        spec = importlib.util.spec_from_file_location("step36i_f16_dep_" + label, ROOT / filename)
        require(spec is not None and spec.loader is not None, f"Cannot import {label}")
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        mods[label] = m
    return mods


def independent_read_handoff(dirpath: Path, keys_signed: pd.DataFrame, upstream: str):
    """Phase B reader: independently revalidate the F15 manifest and every file."""
    require(dirpath.is_dir() and not dirpath.is_symlink(), "Materialization dir missing/unsafe")
    manifest_path = dirpath / "materialization_manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "Missing safe manifest")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    require(manifest.get("schema") == "STEP36I_F15_DURABLE_ARRAY_HANDOFF_SYNTHETIC_ONLY_V1",
            "F15 handoff format mismatch")
    require(manifest.get("synthetic_only") is True and
            manifest.get("read_sealed_test") is False and
            manifest.get("compute_real_labels") is False and
            manifest.get("produced_gpu_inference") is False and
            manifest.get("third_raw_test_access_authorized") is False,
            "Not an authorized synthetic handoff")
    require(manifest.get("frozen_original_step35_sha256") == STEP35_SHA and
            manifest.get("study_owner_approval_sha256") ==
            "f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493" and
            manifest.get("approved_candidate_keys_sha256") == CANDIDATE_SHA and
            manifest.get("upstream_manifest_hash") == upstream, "Handoff provenance mismatch")
    require(manifest.get("shape_X") == [5300, 60, 63] and
            manifest.get("shape_y") == [5300, 3] and
            manifest.get("segments") == [48, 217], "Handoff dimension/segment mismatch")
    entries = manifest.get("artifacts")
    require(isinstance(entries, list) and len(entries) == 3, "Manifest file count drift")
    names = [x.get("relative_path") for x in entries]
    require(set(names) == EXPECTED and len(set(names)) == 3, "Unexpected manifest artifact names")
    actual_files = {x.relative_to(dirpath).as_posix() for x in dirpath.rglob("*") if x.is_file()}
    require(actual_files == EXPECTED | {"materialization_manifest.json"},
            "Unlisted or absent sealed handoff file")
    require(not any(x.is_symlink() for x in dirpath.rglob("*")), "Handoff symlink rejected")
    for item in entries:
        file = dirpath / item["relative_path"]
        require(file.is_file() and not file.is_symlink() and
                file.stat().st_size == item.get("bytes") and
                sha_file(file) == item.get("sha256"),
                f"Committed handoff byte mismatch: {item['relative_path']}")
    X = np.load(dirpath / "arrays/sequence_X.npy", allow_pickle=False, mmap_mode="r")
    y = np.load(dirpath / "arrays/directions_SYNTHETIC_ONLY.npy", allow_pickle=False)
    k = pd.read_csv(dirpath / "arrays/metadata_keys.csv")
    require(X.shape == (5300, 60, 63) and X.dtype == np.float32 and np.isfinite(X).all(),
            "Invalid committed sequence array")
    require(y.shape == (5300, 3) and y.dtype == np.int8 and np.isin(y, [0, 1]).all(),
            "Invalid committed synthetic label array")
    k["Date"] = pd.to_datetime(k["Date"]).dt.normalize()
    ks = keys_signed.reset_index(drop=True).copy()
    ks["Date"] = pd.to_datetime(ks["Date"]).dt.normalize()
    for c in ("market_session_id", "continuity_segment_id"):
        k[c] = k[c].astype(np.int64)
        ks[c] = ks[c].astype(np.int64)
    pd.testing.assert_frame_equal(k, ks, check_dtype=False)
    require(np.array_equal(np.asarray(X[:, 0, 0], dtype=np.int64), np.arange(5300)),
            "Synthetic physical sequence row identity/order changed")
    return X, y, k


def mock_worker(X, model: str, seed: int, physical: int, *, fail_after_first_batch=False):
    """Synthetic batch-sensitive worker. Never imports Torch or loads checkpoints."""
    n = X.shape[0]
    require(n == 5300 and physical > 0 and model, "Bad mock inference contract")
    probs = np.empty((n, 3), dtype=np.float32)
    bounds = []
    for index, start in enumerate(range(0, n, physical)):
        stop = min(n, start + physical)
        rowid = np.asarray(X[start:stop, 0, 0], dtype=np.float64)
        require(np.array_equal(rowid, np.arange(start, stop)), "Batch order drift")
        base = ((rowid.astype(np.int64) * (seed - 39) + len(model)) % 101 - 50) / 80.0
        for j in range(3):
            raw = base + 0.05 * j + 0.007 * (seed - 42)
            probs[start:stop, j] = (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
        bounds.append((start, stop))
        if fail_after_first_batch and index == 0:
            raise RuntimeError("SYNTHETIC_INFERENCE_INTERRUPTED_AFTER_MATERIALIZATION")
    require(bounds[0] == (0, min(physical, n)) and
            bounds[-1] == (n - (n % physical if n % physical else physical), n),
            "Partial physical batch composition changed")
    return probs, bounds


def assert_rejected(fn, msg: str):
    try:
        fn()
    except (RuntimeError, ValueError, AssertionError):
        return
    raise RuntimeError("Negative control accepted: " + msg)


def run(offline: bool):
    print("=" * 108)
    print("STEP 36I-F16 — INDEPENDENT TWO-PHASE DURABLE RESUME REVIEW, SYNTHETIC ONLY")
    print("=" * 108)
    print("[1] F1/F3/F4/F9/F11/F14/F15 SOURCE PROVENANCE AND ACCESS DENIAL")
    mods = imports_checked()
    f1, f3, f4, f15 = (mods[k] for k in ("F1", "F3", "F4", "F15"))
    # Reject all live entrypoints, including those defined in reviewed helpers.
    for fn in (f15.open_real_final_test, f15.commit_real_materialization,
               f15.resume_real_inference, f1.open_sealed_test,
               f1.commit_real_test_results, f1.execute_production):
        try:
            fn()
        except PermissionError:
            pass
        else:
            raise RuntimeError("Production operation unexpectedly callable")
    with zipfile.ZipFile(BUNDLE) as z:
        require(z.testzip() is None, "E2 archive CRC failure")
        k_bytes = z.read("candidate_keys_DRAFT.csv")
        t_bytes = z.read("frozen_thresholds.csv")
        plan_bytes = z.read("frozen_inference_plan.csv")
    require(hashlib.sha256(k_bytes).hexdigest() == CANDIDATE_SHA and
            hashlib.sha256(t_bytes).hexdigest() == f1.HASH["thresholds"] and
            hashlib.sha256(plan_bytes).hexdigest() == f1.HASH["plan"],
            "Signed E2 cohort/plan/threshold drift")
    keys = f15.synth_keys(pd.read_csv(io.BytesIO(k_bytes)))
    plan = pd.read_csv(io.BytesIO(plan_bytes))
    thresholds = f1.plan_thresholds(plan, pd.read_csv(io.BytesIO(t_bytes)))
    before = None
    if offline:
        before = f15.check_live_review_inputs()
        # Also check signed live plan but do not hash/read sealed test data.
        checked(f1.PLAN, f1.HASH["plan"], "live frozen inference plan")
        checked(f1.THRESHOLDS, f1.HASH["thresholds"], "live frozen thresholds")
        f1.confirm_legacy_output_isolation()
    print("PASS — frozen reviews and 40-row inference plan; live permission gates DENIED")

    print("[2] PHASE A: ATOMIC SEAL OF FULL-SIZED SYNTHETIC MATERIALIZATION")
    X = np.zeros((5300, 60, 63), dtype=np.float32)
    X[:, 0, 0] = np.arange(5300, dtype=np.float32)
    y = np.zeros((5300, 3), dtype=np.int8)
    y[::3, :] = 1
    upstream = hashlib.sha256(b"STEP36I_F16_SYNTHETIC_UPSTREAM_ID").hexdigest()
    contents = f15.package_arrays(X, y, keys, upstream_manifest_hash=upstream, synthetic_only=True)
    del X, y
    with tempfile.TemporaryDirectory(prefix="STEP36I_F16_SYNTHETIC_ONLY_") as tmp:
        root = Path(tmp)
        committed = f15.commit_fixture(contents, root, "SYNTHETIC_ONLY_MATERIALIZATION_SEALED", f3)
        require(committed.is_dir(), "Synthetic materialization was not committed")
        count = sum((committed / name).stat().st_size for name in EXPECTED)
        require(count > 80_000_000, "Physical materialization incomplete")
        assert_rejected(lambda: f15.commit_fixture(contents, root,
                        "SYNTHETIC_ONLY_MATERIALIZATION_SEALED", f3), "duplicate phase-A commit")
        print(f"PASS — {count:,} bytes durably sealed in temporary fixture; no-overwrite enforced")

        print("[3] PHASE B: MOCK GPU FAILURE THEN RESTART FROM COMMITTED ARRAYS")
        x, labels, key_read = independent_read_handoff(committed, keys, upstream)
        try:
            mock_worker(x, "TIMESNET", 42, f1.MODEL["TIMESNET"][1],
                        fail_after_first_batch=True)
        except RuntimeError as exc:
            require(str(exc) == "SYNTHETIC_INFERENCE_INTERRUPTED_AFTER_MATERIALIZATION",
                    "Wrong mock interruption cause")
        else:
            raise RuntimeError("Mock failure injection did not interrupt phase B")
        # No upstream dataset loader exists in this file; phase B only uses committed dir.
        x, labels, key_read = independent_read_handoff(committed, keys, upstream)
        lookup = {(str(r.model_id), int(r.seed)): r for r in plan.itertuples(index=False)}
        require(len(lookup) == 40, "Frozen checkpoint plan cardinality changed")
        predictions = {}
        tail_lengths = {}
        for mid in f1.MODEL:
            for seed in f1.SEEDS:
                r = lookup[(mid, seed)]
                batch = int(r.physical_inference_microbatch)
                require((str(r.config_id), batch) == f1.MODEL[mid], "Frozen inference batch drift")
                prob, bounds = mock_worker(x, mid, seed, batch)
                require(len(bounds) == (5300 + batch - 1) // batch,
                        "Physical batch count changed")
                tail_lengths[mid] = bounds[-1][1] - bounds[-1][0]
                predictions[(mid, seed)] = prob
        require(len(predictions) == 40 and tail_lengths["TIMESNET"] ==
                (5300 % f1.MODEL["TIMESNET"][1] or f1.MODEL["TIMESNET"][1]),
                "TimesNet tail batch composition changed")
        ensembles, metrics = f1.in_memory_five_seed_scoring(predictions, labels, thresholds)
        stats, pairs = f4.evaluate_in_memory(key_read, labels, ensembles, thresholds)
        require(len(ensembles) == 8 and len(metrics) == 24 and
                len(stats) == 72 and len(pairs) == 84,
                "Mock resume scoring/statistics cardinality drift")
        require(np.isfinite(metrics.balanced_accuracy.to_numpy()).all(),
                "Non-finite synthetic point estimates")
        print("PASS — injected mock inference failure; resumed only from sealed arrays")
        print("PASS — frozen 40 model-seed batch schedules, TimesNet final tail, 8 ensembles")
        print("PASS — 24 synthetic metric rows, 72 bootstrap rows, 84 exploratory comparisons")

        print("[4] NEGATIVE CONTROLS: PROVENANCE DRIFT, BIT FLIP AND ACCESS MARKER")
        assert_rejected(lambda: independent_read_handoff(committed, keys, "0" * 64),
                        "wrong upstream hash")
        bitpath = committed / "arrays/sequence_X.npy"
        with bitpath.open("r+b") as stream:  # Temporary synthetic fixture only
            stream.seek(170)
            old = stream.read(1)
            stream.seek(170)
            stream.write(bytes([old[0] ^ 0xff]))
        assert_rejected(lambda: independent_read_handoff(committed, keys, upstream),
                        "materialization tamper")
        print("PASS — changed provenance and corrupted committed sequence bytes rejected")

    if offline:
        require(LEDGER.read_bytes() == before, "Historical access record changed")
        f1.confirm_legacy_output_isolation()
        print("PASS — WSL historical ledger remains 2 reads/0 scored; outputs vacant")
    print("=" * 108)
    print("STEP 36I-F16 PASS — INDEPENDENT TWO-PHASE SYNTHETIC RESUME REVIEW ONLY")
    print("Real final test read: FALSE | Real inference: FALSE | Actual test scored: FALSE")
    print("Study-owner design: APPROVED | Independent signoff: PENDING")
    print("Third raw-test read authorized: FALSE | Production launcher: DISABLED")
    print("REMAINING — independent live release/source review, journal/commit wiring, separate access approval")
    print("=" * 108)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--self-test", action="store_true", help="Portable test using frozen E2 bundle")
    selection.add_argument("--offline-preflight", action="store_true", help="Read-only WSL + synthetic test")
    opts = parser.parse_args()
    run(offline=bool(opts.offline_preflight))


if __name__ == "__main__":
    main()
