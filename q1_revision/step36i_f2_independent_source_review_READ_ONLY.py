#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F2: independent OFFLINE source/contract/functional review of F1.

Works from the F1 script and the signed E2 bundle in the SAME q1_revision
folder; reads NO Step10/test parquet or real market data. Does not perform
GPU inference, create output folders, alter any access state or authorize test
access. It is intentionally NOT a production launcher.

Run: PYTHONDONTWRITEBYTECODE=1 python step36i_f2_independent_source_review_READ_ONLY.py
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
F1 = BASE / "step36i_f1_amended_production_runner_REVIEW_ONLY.py"
E2 = BASE / "step36i_e2_source_review_bundle.zip"
F1_HASH = "e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556"
E2_HASH = "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851"
RECOVERED_HASH = "6de04ba7e167449101e2e16bab47ca4ce088f11284c4cbc93b61a6cfbb979f12"
COHORT_HASH = "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65"
PLAN_HASH = "a2f8489933e8d2bbd11d6bab72361f7ad066dd37bc62e1db45bb0d20d87ce086"
THRESH_HASH = "79dc80e0e3b18427cebd77742ff2d10b87aebcdd9ea1e459fed30ac980d0e24f"


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def literal(tree: ast.AST, variable: str):
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == variable for t in targets):
                return ast.literal_eval(node.value)
    raise RuntimeError(f"Missing or nonliteral frozen source variable {variable}")


def expected_failure(fn, signature):
    try:
        fn()
    except (RuntimeError, PermissionError, ValueError, AssertionError, KeyError) as ex:
        require(signature.lower() in str(ex).lower(), f"Unexpected fail-closed reason: {ex}")
        return
    raise RuntimeError("Negative control incorrectly accepted")


def main():
    print("=" * 106)
    print("STEP 36I-F2 — INDEPENDENT F1 SOURCE REVIEW (OFFLINE; NO TEST FILE ACCESS)")
    print("=" * 106)
    print("[1] SOURCE IDENTITY, SIGNED E2 BUNDLE, HISTORICAL CONTRACT")
    require(F1.is_file() and E2.is_file(), "F1 script or E2 source bundle missing")
    source = F1.read_bytes()
    require(digest(source) == F1_HASH, "F1 SHA-256 mismatch")
    require(digest(E2.read_bytes()) == E2_HASH, "E2 bundle SHA-256 mismatch")
    with zipfile.ZipFile(E2) as z:
        names = z.namelist()
        require(len(names) == 18 and len(set(names)) == 18 and z.testzip() is None,
                "Unexpected/corrupt E2 bundle")
        require(all(n and "/" not in n and "\\" not in n and n not in {".", ".."} for n in names),
                "Unsafe E2 archive paths")
        stored = {n: z.read(n) for n in names}
    for n, h in [
        ("recovered_step36_source.py", RECOVERED_HASH),
        ("candidate_keys_DRAFT.csv", COHORT_HASH),
        ("frozen_inference_plan.csv", PLAN_HASH),
        ("frozen_thresholds.csv", THRESH_HASH),
    ]:
        require(digest(stored[n]) == h, f"E2 {n} byte identity mismatch")
    import json
    p = json.loads(stored["protocol_deviation_DRAFT.json"])
    a = json.loads(stored["amendment_review_DRAFT.json"])
    require(p.get("status") == "DRAFT_NOT_APPROVED" and p.get("raw_test_additional_read_authorized") is False,
            "Protocol unexpectedly approved or authorization drift")
    require(a.get("approval_granted") is False and a.get("third_raw_test_read_authorized") is False,
            "Amendment approval / third-read drift")
    print("PASS — F1 source digest, E2 archive and 4 core frozen artifacts; amendment remains unapproved")

    print("[2] INDEPENDENT AST REVIEW OF FROZEN SPEC AND CLI GATES")
    f1_ast = ast.parse(source.decode("utf-8"), filename=F1.name)
    legacy_ast = ast.parse(stored["recovered_step36_source.py"].decode("utf-8"),
                           filename="recovered_step36_source.py")
    legacy_config = literal(legacy_ast, "EXPECTED_CONFIG")
    legacy_batches = literal(legacy_ast, "EXPECTED_PHYSICAL_MB")
    require(literal(legacy_ast, "EXPECTED_COMMON_ROWS") == 6480 and
            literal(legacy_ast, "EXPECTED_COMMON_DATES") == 324 and
            literal(legacy_ast, "EXPECTED_LOOKBACK") == 60 and
            literal(legacy_ast, "EXPECTED_FEATURES") == 63, "Original structural contract drift")
    require(literal(f1_ast, "SEEDS") == tuple(range(42, 47)) and
            literal(f1_ast, "HORIZONS") == (1, 3, 5), "F1 seeds/horizons drift")
    fnames = {n.name for n in f1_ast.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    necessary = {"candidate", "select_from_original_6480", "materialize_preloaded_from_frozen_helpers",
                 "frozen_inference_from_preloaded_X", "in_memory_five_seed_scoring",
                 "refuse_real_execution", "offline_preflight", "read_only_access_check"}
    require(necessary.issubset(fnames), "Missing F1 implementation functions")
    require("--offline-preflight" in source.decode() and
            "--execute" in source.decode() and
            "--production" in source.decode() and
            "--worker" in source.decode(), "Execution guard declarations missing")
    for target in ("open_sealed_test", "commit_real_test_results", "execute_production"):
        assigns = [n for n in f1_ast.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == target for t in n.targets)]
        require(len(assigns) == 1 and isinstance(assigns[0].value, ast.Name) and
                assigns[0].value.id == "refuse_real_execution", f"Gate alias is not deny-only: {target}")
    has_cuda_init = ("torch.empty(1, dtype=torch.float32, device=device)" in source.decode()
                     and source.decode().find("torch.empty(1, dtype=torch.float32, device=device)") <
                     source.decode().find("torch.cuda.reset_peak_memory_stats(device)"))
    require(has_cuda_init, "CUDA context is not initialized before resetting peak statistics")
    require("torch.inference_mode()" in source.decode(), "Frozen inference-mode requirement missing")
    require("for start in range(0, N, physical)" in source.decode(), "Physical batch loop drift")
    print("PASS — original 6480/324 interface; CUDA-order fix; guarded CLI; three denied production entrypoints")

    print("[3] INDEPENDENT IN-MEMORY SPEC AND SELECTOR NEGATIVE CONTROLS")
    spec = importlib.util.spec_from_file_location("review_f1_module", F1)
    require(spec is not None and spec.loader is not None, "Cannot inspect F1 implementation")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # module definitions only: NO F1.main() / files opened
    require(m.N == 5300 and m.ND == 265 and m.NS == 20 and m.L == 60 and m.F == 63,
            "F1 data interface drift")
    require({name: cfg for name, (cfg, batch) in m.MODEL.items()} == legacy_config,
            "F1 selected configurations disagree with recovered frozen source")
    require({name: batch for name, (cfg, batch) in m.MODEL.items()} == legacy_batches,
            "F1 physical batches disagree with recovered frozen source")
    candidates = m.candidate(pd.read_csv(io.BytesIO(stored["candidate_keys_DRAFT.csv"])))
    plan = pd.read_csv(io.BytesIO(stored["frozen_inference_plan.csv"]))
    threshold_frame = pd.read_csv(io.BytesIO(stored["frozen_thresholds.csv"]))
    thresholds = m.plan_thresholds(plan, threshold_frame)
    require(len(candidates) == 5300 and len(thresholds) == 24, "Frozen cohort/threshold dimensions")
    expected_failure(lambda: m.candidate(candidates.iloc[:-1].copy()), "cohort mismatch")
    dup = candidates.copy()
    dup.iloc[0] = dup.iloc[1]
    expected_failure(lambda: m.candidate(dup), "Duplicate/missing")
    drift = candidates.copy()
    drift.iloc[0, drift.columns.get_loc("continuity_segment_id")] = 7
    expected_failure(lambda: m.candidate(drift), "segment")
    # Fixture contains EXACT frozen candidate dates/keys and 59 artificial dates;
    # this does not read the raw test, and is NOT a real sequence validation.
    template = candidates[candidates.Date == candidates.Date.min()].copy()
    extra = []
    for k, date in enumerate(pd.date_range("2030-01-01", periods=59)):
        part = template.copy()
        part["Date"] = date
        part["market_session_id"] = 999900 + k
        part["continuity_segment_id"] = 91
        extra.append(part)
    original = pd.concat([candidates, *extra], ignore_index=True)
    original["eligible_common_multitask"] = True
    recovered = m.select_from_original_6480(original, candidates)
    pd.testing.assert_frame_equal(recovered[m.KEYS].reset_index(drop=True), candidates, check_dtype=False)
    expected_failure(lambda: m.select_from_original_6480(candidates.assign(
        eligible_common_multitask=True), candidates), "original step35 6480/324")
    print("PASS — frozen 40 model-seeds, 24 thresholds, 5300 ordered keys; selector and negative controls")

    print("[4] SCORING FUNCTION WITH NONTRIVIAL INDEPENDENT SYNTHETIC ORACLE")
    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
    index = np.arange(m.N, dtype=np.int64)
    y = np.column_stack([((index // 20 + (index % 20) * (j + 2) + j) % (j + 3) < 1).astype(np.int64)
                         for j in range(3)])
    require(np.all(y.sum(axis=0) > 0) and np.all(y.sum(axis=0) < m.N), "Synthetic oracle classes")
    seed_probs = {}
    for mi, mid in enumerate(m.MODEL):
        base = np.column_stack([
            0.08 + 0.84 * (((index * (13 + j) + mi * 19 + 31 * j) % 109) / 108.0)
            for j in range(3)])
        for si, seed in enumerate(m.SEEDS):
            seed_probs[(mid, seed)] = np.clip(base + (si - 2) * 0.023 + (si % 2) * 0.003, 0, 1)
    ensembles, metrics = m.in_memory_five_seed_scoring(seed_probs, y, thresholds)
    require(len(ensembles) == 8 and len(metrics) == 24, "Scoring output dimensions")
    for mi, mid in enumerate(m.MODEL):
        p_expected = np.mean(np.stack([seed_probs[(mid, s)] for s in m.SEEDS]), axis=0)
        require(np.allclose(ensembles[mid], p_expected.astype(np.float32), atol=1e-7),
                f"Five-seed arithmetic mean drift for {mid}")
        for j, horizon in enumerate(m.HORIZONS):
            rec = metrics.loc[(metrics.model_id == mid) & (metrics.horizon_days == horizon)]
            require(len(rec) == 1, f"Missing model/horizon score: {mid}/{horizon}")
            row = rec.iloc[0]
            tau = thresholds[(mid, horizon)]
            pred = (p_expected[:, j] >= tau).astype(np.int64)
            truth = y[:, j]
            tp = int(np.sum((pred == 1) & (truth == 1)))
            fp = int(np.sum((pred == 1) & (truth == 0)))
            tn = int(np.sum((pred == 0) & (truth == 0)))
            fn = int(np.sum((pred == 0) & (truth == 1)))
            ba = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
            brier = float(np.mean((p_expected[:, j] - truth) ** 2))
            require((int(row.tp), int(row.fp), int(row.tn), int(row.fn)) == (tp, fp, tn, fn),
                    f"Confusion counts drift: {mid}/{horizon}")
            require(abs(row.balanced_accuracy - ba) < 1e-12 and
                    abs(row.brier_score - brier) < 1e-12 and
                    abs(row.threshold - tau) < 1e-12,
                    f"Independent BA/Brier/threshold oracle mismatch: {mid}/{horizon}")
            require(abs(row.roc_auc - roc_auc_score(truth, p_expected[:, j])) < 1e-12 and
                    abs(row.pr_auc - average_precision_score(truth, p_expected[:, j])) < 1e-12 and
                    abs(row.mcc - matthews_corrcoef(truth, pred)) < 1e-12,
                    f"AUC/PR-AUC/MCC oracle mismatch: {mid}/{horizon}")
    bad = dict(seed_probs)
    bad.pop(("DLINEAR", 42))
    expected_failure(lambda: m.in_memory_five_seed_scoring(bad, y, thresholds), "Missing/extra")
    corrupted_y = y.copy()
    corrupted_y[0, 0] = 2
    expected_failure(lambda: m.in_memory_five_seed_scoring(seed_probs, corrupted_y, thresholds), "Invalid 5300x3")
    for gate in (m.open_sealed_test, m.commit_real_test_results, m.execute_production):
        expected_failure(lambda gate=gate: gate(), "review only")
    print("PASS — 8 five-seed ensemble averages; all 24 BA/AUC/MCC/PR-AUC/Brier oracles; negative controls")

    print("[5] REVIEW FINDINGS AND EXECUTION STATUS")
    print("REMAINING — independent authorized raw-file loader and durable access journal: NOT IMPLEMENTED")
    print("REMAINING — isolated atomic real-artifact commit, failure/resume policy: NOT IMPLEMENTED")
    print("REMAINING — real-data integration of segment-stratified bootstrap and BH-FDR: NOT IMPLEMENTED")
    print("REMAINING — end-to-end real checkpoint/sequence audit: NOT EXECUTED")
    print("REMAINING — formal post-freeze protocol review and third-read authorization: NOT GRANTED")
    print("=" * 106)
    print("STEP 36I-F2 PASS — OFFLINE SOURCE, SELECTOR AND SYNTHETIC SCORING AUDIT ONLY")
    print("Real test opened: FALSE | Real GPU inference: FALSE | Test scored: FALSE")
    print("Production CLI: DENIED | Third test read authorized: FALSE")
    print("=" * 106)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"STEP 36I-F2 FAIL-CLOSED: {type(ex).__name__}: {ex}", file=sys.stderr)
        raise SystemExit(1)
