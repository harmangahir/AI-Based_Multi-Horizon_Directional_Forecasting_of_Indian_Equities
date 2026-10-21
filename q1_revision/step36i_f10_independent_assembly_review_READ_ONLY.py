#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F10: independent, offline audit of F9's computation assembly.

USAGE: python step36i_f10_independent_assembly_review_READ_ONLY.py --offline-preflight

Access policy: never locates/opens the sealed final-test parquet. No real GPU,
checkpoint inference, amendment approval, or artifact commit. Reads only F1-F9
review scripts, the E2 review bundle, frozen non-test contracts, and live access
marker during WSL preflight. Runs F9's actual assembly orchestration on a fully
synthetic DataFrame, with a synthetic materializer/checkpoint-inference adapter.
The genuine F1 scoring and F4 statistical implementations ARE called. This is
NOT proof of real-data/real-checkpoint execution.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import zipfile

DENIED = {"--execute", "--production", "--worker", "--resume", "--read-test",
          "--score-test", "--test-parquet", "--authorize", "--approve",
          "--commit-real", "--enable", "--materialize", "--run"}
if DENIED.intersection(sys.argv[1:]):
    raise SystemExit("DENIED: F10 is an independent OFFLINE review only")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
F9_PATH = ROOT / "step36i_f9_preloaded_production_assembly_REVIEW_ONLY.py"
F9_SHA = "8074cfe7cd1de9cf17609d4cef71aff27f910347da3ad86f3861a3cf7ad1373b"
BUNDLE_PATH = ROOT / "step36i_e2_source_review_bundle.zip"
BUNDLE_SHA = "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851"


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked(path: Path, expected: str):
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe frozen source: {path}")
    require(sha256(path) == expected, f"Source SHA mismatch: {path.name}")


def import_checked(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import {path.name}")
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def assert_rejected(fn, context):
    try:
        fn()
    except (RuntimeError, PermissionError, ValueError, AssertionError, KeyError):
        return
    raise RuntimeError(f"Fail-open negative control: {context}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-preflight", action="store_true", required=True)
    args = parser.parse_args()
    print("=" * 109)
    print("STEP 36I-F10 — INDEPENDENT F9 ASSEMBLY REVIEW (SYNTHETIC / NO REAL-TEST ACCESS)")
    print("=" * 109)
    print("[1] F9 SOURCE IDENTITY, F1–F8 CONTRACT, AND LIVE ACCESS STATE")
    checked(F9_PATH, F9_SHA)
    checked(BUNDLE_PATH, BUNDLE_SHA)
    f9_source = F9_PATH.read_text("utf-8")
    f9_tree = ast.parse(f9_source)
    f9_defs = {n.name for n in f9_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    require({"assemble_from_preloaded", "_manifest", "inspect_ro", "blocked_live_entrypoints"} <= f9_defs,
            "F9 implementation interface differs from reviewed source")
    f9 = import_checked("step36i_f10_audited_f9", F9_PATH)
    for idx, name in f9.SOURCE_NAMES.items():
        checked(ROOT / name, f9.SOURCE_HASHES[idx])
    f9.validate_interfaces_static()
    for blocked in (f9.open_sealed_test, f9.commit_real_artifacts, f9.execute_final_test):
        assert_rejected(blocked, "F9 public execution gate")
    live_wsl = f9.STEP35.is_file()
    if live_wsl:
        checked(f9.STEP35, f9.STEP35_SHA)
        f1_original = f9.load_local_helper(1)
        state = f1_original.read_only_access_check()
        require(state.get("raw_test_filesystem_open_count_completed") == 2 and
                state.get("final_test_scored") is False, "Historical test ledger drift")
        f1_original.confirm_legacy_output_isolation()
    else:
        f1_original = f9.load_local_helper(1)
    f4_original = f9.load_local_helper(4)
    require(f1_original.HASH["step35"] == f9.STEP35_SHA, "Original scientific contract mismatch")
    print("PASS — F9 and F1–F8 SHA-256; Step35 boundary; production entrypoints blocked")
    print("PASS — historical ledger 2 reads/0 scoring, namespace vacant" if live_wsl else
          "PASS — portable review; live WSL access marker not mounted here")

    print("[2] FULL-SIZED SYNTHETIC F9 ASSEMBLY: REAL F1 SCORING AND F4 STATISTICS")
    with zipfile.ZipFile(BUNDLE_PATH) as z:
        fixture_files = {
            "candidate": pd.read_csv(io.BytesIO(z.read("candidate_keys_DRAFT.csv"))),
            "plan": pd.read_csv(io.BytesIO(z.read("frozen_inference_plan.csv"))),
            "thresholds": pd.read_csv(io.BytesIO(z.read("frozen_thresholds.csv"))),
        }
    keys = f1_original.candidate(fixture_files["candidate"])
    plan_thresholds = f1_original.plan_thresholds(fixture_files["plan"], fixture_files["thresholds"])
    require(len(keys) == 5300 and len(plan_thresholds) == 24, "Frozen fixture mismatch")
    n = len(keys)
    seq = np.zeros((n, 60, 63), dtype=np.float32)  # synthetic, no real market observations
    ii = np.arange(n, dtype=np.int64)
    md = pd.DataFrame({
        "ticker": keys.Ticker.to_numpy(),
        "reference_date": keys.Date.to_numpy(),
        "reference_market_session_id": keys.market_session_id.to_numpy(),
        "continuity_segment_id": keys.continuity_segment_id.to_numpy(),
        "array_row_index": ii,
        "direction_1d": ((ii // 20 + ii % 20) % 2).astype(np.int8),
        "direction_3d": ((ii // 20 + 3 * (ii % 20) // 2) % 2).astype(np.int8),
        "direction_5d": ((ii // 20 + 7 * (ii % 20) // 3) % 2).astype(np.int8),
    })
    calls = {"materializations": 0, "inference": [], "scorer": 0, "stats": 0,
             "csv": [], "tamper_meta": False}

    def simulated_materializer(raw, history, passed_keys, upstream):
        calls["materializations"] += 1
        pd.testing.assert_frame_equal(f1_original.candidate(passed_keys), keys, check_dtype=False)
        require(raw.get("synthetic_only") is True and history.get("synthetic_only") is True,
                "Synthetic boundary unexpectedly received a real input")
        if calls["tamper_meta"]:
            wrong = md.copy()
            wrong.iloc[[0, 1]] = wrong.iloc[[1, 0]].to_numpy()
            return seq, wrong
        return seq, md

    def simulated_checkpoint_inference(X, row):
        model, seed = str(row["model_id"]), int(row["seed"])
        calls["inference"].append((model, seed))
        require(X is seq and X.shape == (5300, 60, 63), "In-memory sequence identity drift")
        require((model, seed) not in calls["inference"][:-1], "Duplicate checkpoint execution")
        k = list(f1_original.MODEL).index(model)
        z = np.stack([
            0.23 * np.sin((ii + 3 * seed + 9 * k + 11 * horizon) / (13.0 + horizon))
            + 0.15 * np.cos((ii % 20 + 2 * k + seed) / (2.0 + horizon))
            for horizon in (1, 3, 5)
        ], axis=1).astype(np.float32)
        p = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        return z, p, {"peak_cuda_allocated_MiB": 0.0}

    def score_spy(seed_probs, y, thresholds):
        calls["scorer"] += 1
        return f1_original.in_memory_five_seed_scoring(seed_probs, y, thresholds)

    def stats_spy(k, y, ensembles, thresholds):
        calls["stats"] += 1
        return f4_original.evaluate_in_memory(k, y, ensembles, thresholds)

    isolated_f1 = types.SimpleNamespace(
        MODEL=f1_original.MODEL, SEEDS=f1_original.SEEDS, HORIZONS=f1_original.HORIZONS,
        HASH=f1_original.HASH, candidate=f1_original.candidate,
        plan_thresholds=f1_original.plan_thresholds,
        materialize_preloaded_from_frozen_helpers=simulated_materializer,
        frozen_inference_from_preloaded_X=simulated_checkpoint_inference,
        in_memory_five_seed_scoring=score_spy,
    )
    isolated_f4 = types.SimpleNamespace(evaluate_in_memory=stats_spy)
    original_helper = f9.load_local_helper
    original_csv = pd.read_csv
    original_parquet_saver = f9._save_parquet
    original_parquet = True
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        original_parquet = False

    def fake_csv(path, *a, **kw):
        calls["csv"].append(str(path))
        if str(path) == str(f9.AMENDED_KEYS):
            return fixture_files["candidate"].copy()
        if str(path) == str(f9.CHECKPOINT_PLAN):
            return fixture_files["plan"].copy()
        if str(path) == str(f9.THRESHOLD_PLAN):
            return fixture_files["thresholds"].copy()
        raise RuntimeError(f"F10 blocked unexpected CSV access: {path}")

    # Directly intercept F9's three known non-test contract reads ONLY.
    # Because pandas is a shared module this is restored in 'finally'.
    try:
        f9.load_local_helper = lambda idx: ({1: isolated_f1, 4: isolated_f4}[idx])
        pd.read_csv = fake_csv
        if not original_parquet:
            # Portable reviewer only: substitute a deterministic synthetic encoding.
            # A proper pyarrow Parquet serialization is required and used on WSL.
            f9._save_parquet = lambda frame: frame.to_csv(index=False).encode("utf-8")
        payloads, manifest = f9.assemble_from_preloaded(
            {"synthetic_only": True}, {"synthetic_only": True},
            {"features": [f"SYNTHETIC_F{i:02d}" for i in range(63)],
             "scaler_path": "SYNTHETIC_ONLY", "stock_id_map": {"synthetic": True}})
        require(calls["materializations"] == 1 and calls["scorer"] == 1 and calls["stats"] == 1,
                "End-to-end assembly components did not execute exactly once")
        expected_seeds = [(m, s) for m in f1_original.MODEL for s in f1_original.SEEDS]
        require(calls["inference"] == expected_seeds, "40 frozen model-seed call ordering drift")
        require(len(calls["csv"]) == 3 and len(set(calls["csv"])) == 3,
                "Input contract CSV access drift")
        require(manifest["artifact_count"] == len(payloads) - 1,
                "Artifact count mismatch")
        require(manifest["context"]["seed_checkpoint_outputs"] == 40 and
                manifest["context"]["prediction_rows"] == 42400 and
                manifest["context"]["bootstrap_rows"] == 72,
                "Audit manifest run configuration changed")
        require("artifact_manifest.json" in payloads and
                json.loads(payloads["artifact_manifest.json"]) == manifest,
                "Manifest payload does not match computed manifest")
        for entry in manifest["artifacts"]:
            data = payloads[entry["relative_path"]]
            require(len(data) == entry["bytes"] and
                    hashlib.sha256(data).hexdigest() == entry["sha256"],
                    f"Artifact byte-level integrity failure: {entry['relative_path']}")
        # Inspect actual generated CSVs without using pd.read_csv (blocked above).
        metric_frame = original_csv(io.BytesIO(payloads["evaluation/classification_metrics.csv"]))
        bootstrap_frame = original_csv(io.BytesIO(payloads["evaluation/segment_stratified_bootstrap.csv"]))
        pair_frame = original_csv(io.BytesIO(payloads["evaluation/exploratory_pairwise_BA.csv"]))
        register = original_csv(io.BytesIO(payloads["evaluation/inference_register.csv"]))
        require((len(metric_frame), len(bootstrap_frame), len(pair_frame), len(register)) ==
                (24, 72, 84, 40), "Persisted CSV cardinality mismatch")
        primary = bootstrap_frame.loc[bootstrap_frame["block_length"].eq(20)]
        require(len(primary) == 24 and primary.q_BH_primary_only.notna().all(),
                "24 primary FDR rows missing")
        require(bootstrap_frame.loc[~bootstrap_frame.block_length.eq(20),
                                   "q_BH_primary_only"].isna().all(),
                "Secondary sensitivity runs improperly adjusted")
        require(len([x for x in payloads if x.startswith("seed_predictions/")]) == 40 and
                len([x for x in payloads if x.startswith("ensemble_predictions/")]) == 8,
                "Per-checkpoint and ensemble payload counts changed")
        restored_labels = np.load(io.BytesIO(payloads["arrays/labels_5300x3.npy"]), allow_pickle=False)
        require(restored_labels.shape == (5300, 3) and np.array_equal(
                    restored_labels, md[["direction_1d", "direction_3d", "direction_5d"]].to_numpy()),
                "Persisted labels differ from independently constructed synthetic fixture")
    finally:
        # NO source file modifications and NO real-test file reads.
        pd.read_csv = original_csv
        f9.load_local_helper = original_helper
        f9._save_parquet = original_parquet_saver
    print("PASS — actual F9 assembly; 40 ordered synthetic checkpoint invocations; real F1 scorer + F4 bootstrap")
    print("PASS — 24 metrics, 72 bootstrap rows, 84 pairs, 40 register rows and hashed payload manifest")
    if not original_parquet:
        print("NOTE — portable environment lacks pyarrow; Parquet serializer was replaced in fixture ONLY")
    else:
        print("PASS — actual pyarrow Parquet serialization exercised in synthetic assembly")

    print("[3] INDEPENDENT NEGATIVE CONTROLS / ASSEMBLY BOUNDARIES")
    for unsafe in ("../escape.bin", "/absolute.bin", "a/../../escape", ""):
        assert_rejected(lambda name=unsafe: f9._manifest({name: b"fixture"}, {}),
                        f"unsafe output artifact path {unsafe!r}")
    assert_rejected(lambda: f9._manifest({"safe.bin": "not-bytes"}, {}), "nonbytes artifact")
    for denied in (f9.open_sealed_test, f9.commit_real_artifacts, f9.execute_final_test):
        assert_rejected(denied, "production entrypoint")
    print("PASS — path traversal, nonbytes payload, production-entrypoint bypass rejected")

    print("[4] REVIEW FINDINGS / UNIMPLEMENTED PRODUCTION PERMISSIONS")
    # These are explicit review findings, not implicit permissions to run.
    print("OPEN — F9 does not independently authenticate the supplied 63-feature order, scaler, stock map or development-history bytes")
    print("OPEN — real-data single-read journal and atomic output commit not connected to F9's assembly")
    print("OPEN — the post-freeze 5300-row/segmented-bootstrap amendment and third-read authority are NOT approved")
    print("=" * 109)
    print("STEP 36I-F10 PASS — SYNTHETIC ASSEMBLY REVIEW ONLY; PRODUCTION SIGNOFF BLOCKED")
    print("Sealed test opened: FALSE | GPU: FALSE | Real scoring: FALSE | Live ledger unchanged")
    print("Approved amendment: FALSE | Third read authorized: FALSE | Production launch: DENIED")
    print("=" * 109)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError, ValueError, KeyError, OSError, TypeError) as err:
        print(f"STEP 36I-F10 FAIL-CLOSED: {type(err).__name__}: {err}", file=sys.stderr)
        sys.exit(1)
