#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F1 — isolated amended production *implementation*, REVIEW ONLY.

USAGE NOW: python step36i_f1_amended_production_runner_REVIEW_ONLY.py --offline-preflight

The current build is intentionally NOT a production executable. It implements
pure/preloaded data selection, checked causal sequence construction, frozen
checkpoint inference, five-seed ensemble scoring and isolated artifact writing
as independently auditable functions. Its sole CLI action is read-only offline
preflight. It cannot acquire real held-out data, access it a third time, execute
real checkpoint workers, or write actual held-out predictions. This prohibition
is intentional and cannot be lifted by a command-line argument.

Future approved production integration requires a SEPARATE reviewed source,
explicit protocol amendment, and separately documented authorization of a
third raw-test read. Never change the access marker, Step35 contract or
previous Step36 outputs to bypass that requirement.
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
import zipfile

# CLI DENIAL IS BEFORE IMPORTS/PROJECT DISCOVERY/ANY FILE ACCESS.
_REJECT = {"--execute", "--worker", "--resume", "--production", "--authorize",
           "--approve", "--read-test", "--score-test", "--materialize",
           "--run", "--final-test", "--enable"}
if _REJECT.intersection(sys.argv[1:]):
    raise SystemExit("DENIED — F1 review build cannot access, infer, or score the held-out test.")

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
EXP = WORK.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
MAN = EXP / "manifests"
AUD = EXP / "audits"
FEATURE = EXP / "feature_data"
BUNDLE = WORK / "step36i_e2_source_review_bundle.zip"

COHORT = AUD / "step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv"
STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"
PLAN = AUD / "35_final_test_inference_plan.csv"
THRESHOLDS = AUD / "35_final_test_threshold_contract.csv"
ACCESS = MAN / "36_final_test_access_consumption_state.json"
STEP36_RECOVERY = WORK / "step36_one_time_final_test_materialization_inference_scoring_v1.py"
LEDGER = FEATURE / "step10_chronological_splits/10_split_membership.parquet"
E4_REPORT = AUD / "step36i_e4_synthetic_artifact_transaction_draft/36i_e4_synthetic_transaction_DRAFT.json"
E3_REPORT = AUD / "step36i_e3_isolated_runner_review_draft/36i_e3_isolated_runner_review_DRAFT.json"

# Every future actual-output location is separate from the historical Step36
# namespace and from all synthetic validation artifacts.
ISOLATED = {
    "array_dir": FEATURE / "step36i_amended_final_test_arrays",
    "array_manifest": MAN / "36i_amended_final_test_materialization_manifest.json",
    "predictions": EXP / "final_test_evaluation/step36i_amended_predictions",
    "evaluation_manifest": MAN / "36i_amended_final_test_evaluation_manifest.json",
    "metrics": AUD / "36i_amended_final_test_metrics.csv",
    "independent_access_journal": MAN / "36i_amended_final_test_access_journal.json",
    "failure_evidence": AUD / "36i_amended_final_test_execution_failure.json",
}

HASH = {
    "bundle": "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851",
    "source": "6de04ba7e167449101e2e16bab47ca4ce088f11284c4cbc93b61a6cfbb979f12",
    "step35": "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
    "plan": "a2f8489933e8d2bbd11d6bab72361f7ad066dd37bc62e1db45bb0d20d87ce086",
    "thresholds": "79dc80e0e3b18427cebd77742ff2d10b87aebcdd9ea1e459fed30ac980d0e24f",
    "cohort": "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
    "membership": "82772196c92e88f1926a75a70c42f5f19119d7245d93b8364455fd28642ee121",
    "e3": "4027e53e49ef918a3f3a3bc933908d327e233b741d387088279359b392fe1094",
    "e4": "30fbd08cecba4e65eb427358c6a24c9d28b984bdf889afc83668a6c9952ab925",
}
MODEL = {
    "DLINEAR": ("DLINEAR_B", 128), "NLINEAR": ("NLINEAR_B", 128),
    "PATCHTST": ("PATCHTST_B", 64), "TIMESNET": ("TIMESNET_B", 64),
    "ITRANSFORMER": ("ITRANSFORMER_A", 128), "FREDFORMER": ("FREDFORMER_B", 16),
    "FRNET": ("FRNET_A", 64), "MLF": ("MLF_A", 32),
}
SEEDS = (42, 43, 44, 45, 46)
HORIZONS = (1, 3, 5)
KEYS = ["Ticker", "Date", "market_session_id", "continuity_segment_id"]
N, ND, NS, L, F = 5300, 265, 20, 60, 63


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def checked(path: Path, digest: str, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe {label}: {path}")
    require(sha(path) == digest, f"SHA mismatch for {label}: {path}")


def load_bundle(path: Path) -> dict[str, bytes]:
    checked(path, HASH["bundle"], "E2 source bundle")
    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()
        require(len(names) == 18 and len(set(names)) == 18, "E2 member count/duplicates")
        require(all(name and name not in (".", "..") and "/" not in name and "\\" not in name
                    for name in names), "Unsafe E2 member path")
        require(z.testzip() is None, "Damaged E2 bundle")
        return {name: z.read(name) for name in names}


def candidate(frame: pd.DataFrame) -> pd.DataFrame:
    require(set(KEYS).issubset(frame.columns), "Candidate key columns missing")
    out = frame[KEYS].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="raise").dt.normalize()
    for col in KEYS[2:]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("int64")
    require(not out.isna().any().any() and not out.duplicated().any(), "Duplicate/missing candidate keys")
    require(len(out) == N and out.Date.nunique() == ND and out.Ticker.nunique() == NS,
            "5,300/265/20 cohort mismatch")
    require(out.equals(out.sort_values(["Date", "Ticker"]).reset_index(drop=True)),
            "Frozen physical endpoint order changed")
    require(out.groupby("Date")["Ticker"].nunique().eq(NS).all(),
            "Candidate date clusters are not 20-stock balanced")
    ds = out.drop_duplicates("Date").sort_values("Date")
    require(ds["continuity_segment_id"].astype(int).tolist() == [0] * 48 + [1] * 217,
            "Amended 48/217 segment allocation changed")
    first = ds[ds.continuity_segment_id == 0]["Date"]
    second = ds[ds.continuity_segment_id == 1]["Date"]
    require((str(first.min().date()), str(first.max().date())) ==
            ("2025-01-01", "2025-03-07"), "Segment 0 dates drift")
    require((str(second.min().date()), str(second.max().date())) ==
            ("2025-08-01", "2026-06-22"), "Segment 1 dates drift")
    for seg, section in ds.groupby("continuity_segment_id", sort=True):
        require(np.all(np.diff(section.market_session_id.to_numpy(dtype=np.int64)) == 1),
                f"Nonconsecutive master session inside segment {seg}")
    return out.reset_index(drop=True)


def plan_thresholds(plan: pd.DataFrame, thresholds: pd.DataFrame) -> dict[tuple[str, int], float]:
    cols = {"model_id", "config_id", "seed", "physical_inference_microbatch",
            "checkpoint_path", "checkpoint_sha256", "inference_row_order",
            "inference_shuffle", "gradient_tracking", "probability_transform",
            "ensemble_rule"}
    require(cols.issubset(plan.columns) and len(plan) == 40, "40-row inference plan invalid")
    require(set((str(m), int(s)) for m, s in zip(plan.model_id, plan.seed)) ==
            {(m, s) for m in MODEL for s in SEEDS}, "Frozen model-seed pairs changed")
    for row in plan.itertuples(index=False):
        config, physical = MODEL[str(row.model_id)]
        require((str(row.config_id), int(row.physical_inference_microbatch)) ==
                (config, physical), "Model configuration or physical batch drift")
        require(str(row.inference_row_order) == "reference_date_then_ticker", "Row ordering changed")
        require(str(row.inference_shuffle).lower() in {"false", "0"} and
                str(row.gradient_tracking).lower() in {"false", "0"}, "Shuffle/gradient enabled")
        require(str(row.probability_transform) == "sigmoid(logit)" and
                str(row.ensemble_rule) == "arithmetic_mean_of_seed_probabilities_42_43_44_45_46",
                "Frozen probability/ensemble rule changed")
        cp = Path(str(row.checkpoint_path))
        require(cp.name == "final_checkpoint.pt" and len(str(row.checkpoint_sha256)) == 64,
                "Checkpoint inventory malformed")
    require(len(thresholds) == 24 and
            not thresholds[["model_id", "horizon_days"]].duplicated().any(),
            "24 threshold inventory invalid")
    require({(str(m), int(h)) for m, h in zip(thresholds.model_id, thresholds.horizon_days)} ==
            {(m, h) for m in MODEL for h in HORIZONS}, "Threshold pairs changed")
    t = {(str(row.model_id), int(row.horizon_days)): float(row.selected_threshold)
         for row in thresholds.itertuples(index=False)}
    require(all(np.isfinite(v) and 0 <= v <= 1 for v in t.values()),
            "Invalid frozen thresholds")
    return t


def confirm_legacy_output_isolation() -> None:
    # Do not create/touch the namespace. Existing outputs mean STOP.
    for name, path in ISOLATED.items():
        require(not path.exists() and not path.is_symlink(),
                f"Isolated production namespace occupied: {name} {path}")
    # A different original Step36 namespace is intentionally preserved.
    original = {
        FEATURE / "step36_final_test_arrays",
        MAN / "36_final_test_materialization_manifest.json",
        MAN / "36_one_time_final_test_materialization_inference_scoring_manifest.json",
        AUD / "36_final_test_metrics.csv",
    }
    require(not any(path in original for path in ISOLATED.values()),
            "Amended namespace intersects original Step36")


def read_only_access_check() -> dict:
    state = json.loads(ACCESS.read_text(encoding="utf-8"))
    require(state.get("state") == "RAW_TEST_REOPENED_FOR_TECHNICAL_RECOVERY",
            "Historical access state has changed")
    require(int(state.get("raw_test_filesystem_open_count_completed", -1)) == 2 and
            state.get("final_test_scored") is False,
            "Expected precisely 2 historical raw reads and 0 successful real scores")
    require(state.get("observed_raw_test_sha256") == state.get("expected_raw_test_sha256"),
            "Historical sealed hash provenance mismatch")
    return state


def inspect_e4(root: Path) -> dict:
    doc = json.loads((root / "36i_e4_synthetic_transaction_DRAFT.json").read_text("utf-8"))
    require(doc.get("status") == "PASS_SYNTHETIC_ARTIFACT_TRANSACTION_ONLY", "E4 status drift")
    require(doc.get("synthetic_observations") == N and
            doc.get("synthetic_sessions") == ND and
            doc.get("synthetic_ensembles") == 8 and
            doc.get("synthetic_metrics") == 24 and
            doc.get("synthetic_prediction_rows") == N * 8,
            "E4 synthetic dimensions drift")
    require(doc.get("production_runner_implemented") is False and
            doc.get("protocol_amendment_approved") is False and
            doc.get("third_raw_test_access_authorized") is False,
            "E4 approval status drift")
    items = doc["artifacts"]
    require(len(items) == 10 and len(set(it["relative_path"] for it in items)) == 10,
            "E4 artifact manifest incomplete")
    for item in items:
        rel = Path(item["relative_path"])
        require(not rel.is_absolute() and ".." not in rel.parts and
                rel.parts[0] in {"synthetic_ensembles", "SYNTHETIC_ONLY_metrics.csv",
                                 "SYNTHETIC_ONLY_ensemble_predictions.parquet"},
                "E4 artifact pathname unsafe")
        p = root / rel
        checked(p, item["sha256"], "E4 synthetic artifact")
        require(p.stat().st_size == item["bytes"], "E4 artifact size drift")
    return doc


def import_recovered_source_checked():
    """Safe to import AFTER source hash verified. This does not call its main()."""
    checked(STEP36_RECOVERY, HASH["source"], "Recovered Step36 source")
    spec = importlib.util.spec_from_file_location("step36i_f1_recovered_readonly", STEP36_RECOVERY)
    require(spec is not None and spec.loader is not None, "Cannot import recovered source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_from_original_6480(preloaded: pd.DataFrame, frozen_keys: pd.DataFrame) -> pd.DataFrame:
    """Pure in-memory selector: never touches a real-data file.

    Input must ALREADY have the frozen Step10 original structural flag attached,
    not inferred from test labels or test performance. Called only after the
    as-yet-unapproved data-acquisition gate in a future reviewed release.
    """
    require("eligible_common_multitask" in preloaded.columns,
            "Missing frozen structural flag")
    frame = preloaded.copy()
    frame["Date"] = pd.to_datetime(frame.Date).dt.normalize()
    original = frame.loc[frame.eligible_common_multitask.astype(bool)].copy()
    require(len(original) == 6480 and original.Date.nunique() == 324,
            "Original Step35 6480/324 cohort missing; STOP")
    require(not original[KEYS].duplicated().any(), "Original structural keys duplicated")
    keys = candidate(frozen_keys)
    selected = original.merge(keys, on=KEYS, how="inner", validate="one_to_one")
    selected = selected.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    require(len(selected) == N, "Candidate whitelist does not match 5300 original rows")
    selected_keys = candidate(selected)
    pd.testing.assert_frame_equal(selected_keys, keys, check_dtype=False)
    return selected


def materialize_preloaded_from_frozen_helpers(preloaded_raw: pd.DataFrame,
                                               final_dev_scaled: pd.DataFrame,
                                               keys: pd.DataFrame, upstream: dict):
    """Design implementation for causal sequence materialization; no file acquisition.

    Review only: there is NO route from the F1 command line to this function.
    The recovered frozen helper functions are used after source-hash validation.
    It checks original 6480 eligibility before applying candidate 5300 keys.
    """
    o = import_recovered_source_checked()
    df = o.normalize_dates(preloaded_raw)
    df = o.attach_frozen_final_test_membership(df)  # reads only Step10 membership ledger
    o.validate_opened_test(df, upstream["features"])  # original 6480 contract
    df = o.apply_frozen_regimes(df)  # final-development regime thresholds only
    scaled = o.transform_frozen_predictors(df, upstream["features"], upstream["scaler_path"])
    full_original = o.build_common_endpoints(scaled, upstream["stock_id_map"])
    ep = select_from_original_6480(full_original, keys)
    hist = o.build_history_lookup(o.normalize_dates(final_dev_scaled), scaled, upstream["features"])
    X, meta = o.materialize_sequences(ep, hist, upstream["features"])
    require(X.shape == (N, L, F) and np.isfinite(X).all(), "Materialization shape/finiteness")
    require(len(meta) == N and np.array_equal(meta.array_row_index.to_numpy(), np.arange(N)),
            "Metadata row ordering differs from selected cohort")
    require(np.isin(meta[[f"direction_{h}d" for h in HORIZONS]].to_numpy(), [0, 1]).all(),
            "Non-binary direction labels")
    normalized_md = pd.DataFrame({
        "Ticker": meta.ticker,
        "Date": meta.reference_date,
        "market_session_id": meta.reference_market_session_id,
        "continuity_segment_id": meta.continuity_segment_id,
    })
    pd.testing.assert_frame_equal(candidate(normalized_md), candidate(keys), check_dtype=False)
    return X, meta


def frozen_inference_from_preloaded_X(X: np.ndarray, plan_row: pd.Series) -> tuple[np.ndarray, np.ndarray, dict]:
    """Executable in-memory inference algorithm, NOT exposed by F1 CLI.

    Crucially initialize CUDA context BEFORE reset_peak_memory_stats. Physical
    batches and exact row order are retained (including the last partial batch),
    especially for TimesNet, whose FFT path can depend on batch composition.
    """
    require(X.shape == (N, L, F) and X.dtype == np.float32, "Frozen X interface mismatch")
    require(np.isfinite(X).all(), "Nonfinite predictors")
    import torch
    import step26_stage1_runner_dryrun_v3 as step26
    mid, seed = str(plan_row["model_id"]), int(plan_row["seed"])
    config, physical = MODEL[mid]
    require(str(plan_row["config_id"]) == config and
            int(plan_row["physical_inference_microbatch"]) == physical,
            "Config/batch changed for frozen inference")
    cp = Path(str(plan_row["checkpoint_path"]))
    checked(cp, str(plan_row["checkpoint_sha256"]), "Frozen checkpoint")
    require(torch.cuda.is_available(), "CUDA unavailable")
    step26.set_training_determinism(seed)
    device = torch.device("cuda:0")
    torch.empty(1, dtype=torch.float32, device=device)  # INITIALIZE CUDA CONTEXT
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    model, build_metadata, repository = step26.build_candidate_model(
        model_id=mid, config_id=config, physical_microbatch=physical, device=device,
    )
    checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    all_logits, all_prob = [], []
    with torch.inference_mode():
        for start in range(0, N, physical):
            stop = min(start + physical, N)
            xb = torch.from_numpy(np.ascontiguousarray(X[start:stop])).to(device)
            logits, _ = model(xb)
            require(tuple(logits.shape) == (stop - start, len(HORIZONS)),
                    f"{mid}/seed{seed}: logits shape drift")
            require(bool(torch.isfinite(logits).all().item()), "Nonfinite inference logits")
            probs = torch.sigmoid(logits)
            all_logits.append(logits.detach().cpu().float().numpy())
            all_prob.append(probs.detach().cpu().float().numpy())
    torch.cuda.synchronize(device)
    logits = np.concatenate(all_logits).astype(np.float32)
    probabilities = np.concatenate(all_prob).astype(np.float32)
    require(logits.shape == (N, 3) and probabilities.shape == (N, 3),
            "Frozen inference result shape drift")
    require(np.isfinite(logits).all() and np.isfinite(probabilities).all() and
            ((probabilities >= 0) & (probabilities <= 1)).all(),
            "Frozen inference numerical integrity")
    device_stats = {"peak_cuda_allocated_MiB": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
                    "model_id": mid, "seed": seed,
                    "checkpoint_sha256": str(plan_row["checkpoint_sha256"]),
                    "physical_inference_microbatch": physical,
                    "model_build_metadata": build_metadata,
                    "source_repository_record": repository}
    del model, checkpoint
    return logits, probabilities, device_stats


def in_memory_five_seed_scoring(seed_probs: dict[tuple[str, int], np.ndarray],
                                y: np.ndarray, thresholds: dict[tuple[str, int], float]) -> tuple[dict, pd.DataFrame]:
    """Pure scoring; requires 40 verified, aligned seed-probability matrices."""
    require(y.shape == (N, 3) and np.isin(y, [0, 1]).all(), "Invalid 5300x3 labels")
    require(set(seed_probs) == {(m, s) for m in MODEL for s in SEEDS},
            "Missing/extra five-seed prediction arrays")
    from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                                 roc_auc_score, average_precision_score,
                                 brier_score_loss, confusion_matrix)
    ensembles: dict[str, np.ndarray] = {}
    rows = []
    for model_id, (config, _) in MODEL.items():
        p5 = []
        for seed in SEEDS:
            p = np.asarray(seed_probs[(model_id, seed)], dtype=np.float64)
            require(p.shape == (N, 3) and np.isfinite(p).all() and
                    np.all((0 <= p) & (p <= 1)), "Seed probabilities invalid")
            p5.append(p)
        mean = np.mean(np.stack(p5, axis=0), axis=0)
        ensembles[model_id] = mean.astype(np.float32)
        for j, horizon in enumerate(HORIZONS):
            target = y[:, j].astype(np.int64)
            p = mean[:, j]
            threshold = thresholds[(model_id, horizon)]
            yh = (p >= threshold).astype(np.int64)
            tn, fp, fn, tp = confusion_matrix(target, yh, labels=[0, 1]).ravel().tolist()
            if len(np.unique(target)) < 2:
                fail(f"{horizon}d real-test labels single class: metrics undefined, STOP")
            rows.append({"model_id": model_id, "config_id": config,
                         "horizon_days": horizon, "n": N,
                         "threshold": threshold, "balanced_accuracy": float(balanced_accuracy_score(target, yh)),
                         "roc_auc": float(roc_auc_score(target, p)),
                         "mcc": float(matthews_corrcoef(target, yh)),
                         "pr_auc": float(average_precision_score(target, p)),
                         "brier_score": float(brier_score_loss(target, p)),
                         "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
                         "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
                         "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                         "threshold_source": "STEP35_FROZEN_DEV_OOF",
                         "best_seed_selection": False, "test_threshold_tuning": False})
    return ensembles, pd.DataFrame(rows)


def refuse_real_execution() -> None:
    raise PermissionError("Step 36I-F1 is REVIEW ONLY. No additional raw-test access is authorized.")


# Deliberately not connected to any CLI argument or public execution path.
# A later independently approved release needs a separately reviewed one-read
# loader, isolated atomic transaction, durable access journal, complete resume
# policy, and segment-stratified bootstrap/FDR integration BEFORE running.
open_sealed_test = refuse_real_execution
commit_real_test_results = refuse_real_execution
execute_production = refuse_real_execution


def offline_preflight(bundle_path: Path, project_root: Path) -> None:
    require(project_root.resolve() == WORK.resolve(), "Wrong project root / script location")
    print("=" * 109)
    print("STEP 36I-F1 — AMENDED PRODUCTION IMPLEMENTATION, READ-ONLY PREFLIGHT")
    print("=" * 109)
    print("[1] FROZEN E2 SOURCE, STEP35 AND ACCESS HISTORY")
    bundle = load_bundle(bundle_path)
    data_checks = {
        "recovered_step36_source.py": HASH["source"],
        "step35_contract.json": HASH["step35"],
        "candidate_keys_DRAFT.csv": HASH["cohort"],
        "frozen_inference_plan.csv": HASH["plan"],
        "frozen_thresholds.csv": HASH["thresholds"],
    }
    for fname, digest in data_checks.items():
        require(hashlib.sha256(bundle[fname]).hexdigest() == digest, f"Bundled {fname} hash mismatch")
    protocol_draft = json.loads(bundle["protocol_deviation_DRAFT.json"])
    amendment_draft = json.loads(bundle["amendment_review_DRAFT.json"])
    require(protocol_draft.get("status") == "DRAFT_NOT_APPROVED" and
            protocol_draft.get("raw_test_additional_read_authorized") is False and
            protocol_draft.get("future_actions_authorized") is False,
            "Protocol draft unexpectedly approved; stop for independent re-review")
    require(amendment_draft.get("approval_granted") is False and
            amendment_draft.get("third_raw_test_read_authorized") is False,
            "Amendment review/authorization drift")
    for path, digest, label in (
        (STEP36_RECOVERY, HASH["source"], "Recovered source"),
        (STEP35, HASH["step35"], "Step35 original contract"),
        (PLAN, HASH["plan"], "Frozen inference plan"),
        (THRESHOLDS, HASH["thresholds"], "Frozen thresholds"),
        (COHORT, HASH["cohort"], "Candidate keys"),
        (LEDGER, HASH["membership"], "Frozen Step10 membership"),
        (E3_REPORT, HASH["e3"], "E3 offline report"),
        (E4_REPORT, HASH["e4"], "E4 synthetic transaction"),
    ):
        checked(path, digest, label)
    hist = read_only_access_check()
    print("PASS — original Step35 contract intact; exactly 2 historical reads; real test unscored")

    print("[2] FROZEN AMENDED COHORT AND INFERENCE SPECIFICATION")
    keys = candidate(pd.read_csv(COHORT))
    threshold_map = plan_thresholds(pd.read_csv(PLAN), pd.read_csv(THRESHOLDS))
    require(len(threshold_map) == 24, "Frozen threshold mapping incomplete")
    # Rebuild original structural membership from immutable non-label ledger.
    original = pd.read_parquet(LEDGER, columns=KEYS+["outer_final_test_common_multitask"], engine="pyarrow")
    original["Date"] = pd.to_datetime(original["Date"]).dt.normalize()
    original = original.loc[original["outer_final_test_common_multitask"].astype(bool)].copy()
    original["eligible_common_multitask"] = True
    require(len(original) == 6480 and original.Date.nunique() == 324,
            "Original frozen 6480/324 membership changed")
    verified = select_from_original_6480(original, keys)
    require(len(verified) == N, "Frozen candidate fails structural ledger subset")
    print("PASS — 6480 original structural rows, 5300 causal candidate endpoints, 265 dates")
    print("PASS — 40 frozen model-seed specifications; 24 fixed thresholds")

    print("[3] E4 SYNTHETIC TRANSACTION AND OUTPUT ISOLATION")
    e4root = E4_REPORT.parent
    inspect_e4(e4root)
    confirm_legacy_output_isolation()
    print("PASS — 10 committed E4 artifacts rehashed and size-checked; amended output namespace vacant")

    print("[4] FAIL-CLOSED LOGIC AND RUNNER BOUNDARIES")
    probe = keys.copy()
    probe["eligible_common_multitask"] = True
    try:
        select_from_original_6480(probe, keys)
    except RuntimeError as exc:
        require("Original Step35 6480/324 cohort missing" in str(exc),
                f"Selector rejected fixture for an unexpected reason: {exc}")
    else:
        fail("FAIL: selector accepted invalid original-cohort fixture")
    for fn in (open_sealed_test, commit_real_test_results, execute_production):
        try:
            fn()
        except PermissionError:
            pass
        else:
            fail("FAIL: review-only production gate unexpectedly open")
    print("PASS — original cohort negative control rejected; all real-execution gates DENIED")

    print("=" * 109)
    print("STEP 36I-F1 READ-ONLY PREFLIGHT: PASS")
    print("Implementation scope: in-memory cohort/materialization, 40-model inference, ensemble scoring")
    print("Raw test opened: FALSE | Real GPU inference: FALSE | Real test scored: FALSE")
    print("Additional read authorized: FALSE | Protocol approved: FALSE | Production CLI: DISABLED")
    print("Next: independent F1 implementation review and completion of approved access/transaction design")
    print("=" * 109)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-preflight", action="store_true", required=True)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--project-root", type=Path, default=WORK)
    args = parser.parse_args()
    offline_preflight(args.bundle, args.project_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, KeyError, FileNotFoundError, AssertionError) as exc:
        print(f"F1 REVIEW FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
