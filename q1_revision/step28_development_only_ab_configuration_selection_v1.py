#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 28 — DEVELOPMENT-ONLY A/B CONFIGURATION SELECTION

Select exactly one frozen A/B configuration per architecture from the completed
Step-27I amendment-aware Stage-1 development corpus.

Frozen hierarchy, within each architecture:
1) higher mean 5-fold macro balanced accuracy
2) higher mean 5-fold macro ROC-AUC
3) higher mean 5-fold macro MCC
4) higher mean 5-fold macro PR-AUC
5) lower mean 5-fold macro Brier score
6) lower trainable parameter count
7) ascending lexical config_id

No threshold optimization, best-seed selection, cross-architecture winner
selection, or final-test access occurs in this step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import step00_config as cfg
import step27_stage1_execution_v3 as s27


STEP_NUMBER = "28"
STEP_NAME = "STEP 28 — DEVELOPMENT-ONLY A/B CONFIGURATION SELECTION"
SCHEMA_VERSION = "1.0"
SCRIPT = Path(__file__).resolve()

EXPECTED_CONFIGURATION_FINGERPRINT = (
    "9675cb42e5544d9a3cda235208b33cc4c27f7583531535505b42056aa86b04dd"
)
EXPECTED_STEP16_SELECTION_PROTOCOL_FINGERPRINT = (
    "ba87cc46c61d3d3ed0e4e82e4cc2d712f1b511ded94fe336a6adbb56b9a54978"
)
EXPECTED_STEP27I_STAGE1_FINGERPRINT = (
    "0118cd700127727dbb3a193e23b1da893155bee05e3cd0e98844c12844a25f39"
)
EXPECTED_STEP27I_MANIFEST_SHA256 = (
    "1065a5e2e45e61d3fe9a6526976edeb47ede9f067e03ef87cb550fbec3ab622b"
)
EXPECTED_STEP27I_SOURCE_SHA256 = (
    "fca13b2e965e9fc9d48cfc30eafd87e3ff2391dfacffb8f531f08f69d7cd5f10"
)
EXPECTED_STEP27F_AMENDMENT_SHA256 = (
    "04c30b59b8bef7327b44e1764240ead258edbba64f278e31ca29249141dfdce8"
)

EXPECTED_ROWS = 80
EXPECTED_MODELS = 8
EXPECTED_CONFIGS = 16
EXPECTED_FOLDS = [1, 2, 3, 4, 5]
EXPECTED_SEED = 42
SCREENING_THRESHOLD = 0.5

SELECTION_RULE = [
    {"priority": 1, "field": "mean_macro_balanced_accuracy", "direction": "higher", "source_column": "macro_balanced_accuracy"},
    {"priority": 2, "field": "mean_macro_roc_auc", "direction": "higher", "source_column": "macro_roc_auc"},
    {"priority": 3, "field": "mean_macro_mcc", "direction": "higher", "source_column": "macro_mcc"},
    {"priority": 4, "field": "mean_macro_pr_auc", "direction": "higher", "source_column": "macro_pr_auc"},
    {"priority": 5, "field": "mean_macro_brier_score", "direction": "lower", "source_column": "macro_brier_score"},
    {"priority": 6, "field": "trainable_parameter_count", "direction": "lower", "source_column": "trainable_parameter_count"},
    {"priority": 7, "field": "config_id", "direction": "ascending_lexical", "source_column": "config_id"},
]

STEP27I_FIT_REGISTER = cfg.AUDIT_DIR / "27i_stage1_amendment_aware_fit_register.csv"
STEP27I_COMPLETION_REGISTER = cfg.AUDIT_DIR / "27i_stage1_amendment_aware_completion_hash_register.csv"
STEP27I_FREEZE_JSON = cfg.MANIFEST_DIR / "27i_stage1_amendment_aware_execution_freeze.json"
STEP27I_MANIFEST_JSON = cfg.MANIFEST_DIR / "27i_stage1_amendment_aware_execution_manifest.json"
STEP27I_SOURCE_FREEZE_JSON = cfg.MANIFEST_DIR / "27i_stage1_amendment_aware_final_freeze_source_freeze.json"
STEP27I_FROZEN_SOURCE = cfg.MANIFEST_DIR / "step27i_stage1_amendment_aware_final_freeze_frozen.py"

SOURCE_FREEZE_JSON = cfg.MANIFEST_DIR / "28_development_only_ab_selection_source_freeze.json"
SOURCE_FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step28_development_only_ab_configuration_selection_frozen.py"
CONFIG_SUMMARY_CSV = cfg.AUDIT_DIR / "28_configuration_fivefold_summary.csv"
SELECTED_CONFIGS_CSV = cfg.AUDIT_DIR / "28_selected_configurations.csv"
SELECTION_DECISIONS_JSON = cfg.MANIFEST_DIR / "28_configuration_selection_decisions.json"
FREEZE_JSON = cfg.MANIFEST_DIR / "28_configuration_selection_freeze.json"
SUMMARY_TXT = cfg.REPORT_DIR / "28_configuration_selection_summary.txt"
MANIFEST_JSON = cfg.MANIFEST_DIR / "28_configuration_selection_manifest.json"
MANIFEST_SHA = cfg.MANIFEST_DIR / "28_configuration_selection_manifest.sha256"
OUTPUT_HASH_CSV = cfg.AUDIT_DIR / "28_configuration_selection_output_hashes.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(payload: Any) -> str:
    return cfg.canonical_json_sha256(payload)


def ensure_parent_dirs() -> None:
    for path in (
        SOURCE_FREEZE_JSON, SOURCE_FROZEN_SCRIPT, CONFIG_SUMMARY_CSV,
        SELECTED_CONFIGS_CSV, SELECTION_DECISIONS_JSON, FREEZE_JSON,
        SUMMARY_TXT, MANIFEST_JSON, MANIFEST_SHA, OUTPUT_HASH_CSV,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)


def find_step16_protocol_references() -> list[dict]:
    roots = [cfg.MANIFEST_DIR, cfg.AUDIT_DIR, cfg.REPORT_DIR]
    matches, seen = [], set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".txt", ".csv"}:
                continue
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if EXPECTED_STEP16_SELECTION_PROTOCOL_FINGERPRINT in txt:
                matches.append({
                    "path": str(rp),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                })
    if not matches:
        raise RuntimeError(
            "Could not locate a frozen artifact containing the expected "
            "Step-16 selection-protocol fingerprint."
        )
    return matches


def source_semantic() -> dict:
    return {
        "schema_version": 1,
        "status": "STEP28_DEVELOPMENT_ONLY_AB_SELECTION_SOURCE_FROZEN",
        "configuration_fingerprint_sha256": EXPECTED_CONFIGURATION_FINGERPRINT,
        "step28_source_sha256": sha256_file(SCRIPT),
        "upstream": {
            "step16_selection_protocol_fingerprint_sha256": EXPECTED_STEP16_SELECTION_PROTOCOL_FINGERPRINT,
            "step27i_stage1_execution_fingerprint_sha256": EXPECTED_STEP27I_STAGE1_FINGERPRINT,
            "step27i_manifest_sha256": EXPECTED_STEP27I_MANIFEST_SHA256,
            "step27i_source_sha256": EXPECTED_STEP27I_SOURCE_SHA256,
            "step27f_protocol_amendment_sha256": EXPECTED_STEP27F_AMENDMENT_SHA256,
        },
        "selection_scope": {
            "unit": "within_architecture_A_vs_B",
            "architectures": EXPECTED_MODELS,
            "candidate_configurations": EXPECTED_CONFIGS,
            "folds_per_configuration": len(EXPECTED_FOLDS),
            "seed": EXPECTED_SEED,
            "screening_threshold": SCREENING_THRESHOLD,
        },
        "selection_rule": SELECTION_RULE,
        "prohibitions": {
            "final_test_access": True,
            "threshold_optimization": True,
            "best_seed_selection": True,
            "cross_architecture_winner_selection": True,
            "post_hoc_metric_reordering": True,
        },
    }


def ensure_source_freeze() -> tuple[str, str]:
    ensure_parent_dirs()
    semantic = source_semantic()
    source_sha = semantic["step28_source_sha256"]
    fingerprint = canonical_hash(semantic)

    if not SOURCE_FREEZE_JSON.exists():
        if SOURCE_FROZEN_SCRIPT.exists():
            raise RuntimeError("Frozen Step-28 source exists without source-freeze JSON.")
        payload = {**semantic, "step28_source_fingerprint_sha256": fingerprint}
        s27.atomic_json(SOURCE_FREEZE_JSON, payload)
        shutil.copy2(SCRIPT, SOURCE_FROZEN_SCRIPT)

    payload = load_json(SOURCE_FREEZE_JSON)
    if payload.get("step28_source_sha256") != source_sha:
        raise RuntimeError("Current Step-28 source differs from frozen source.")
    if payload.get("step28_source_fingerprint_sha256") != fingerprint:
        raise RuntimeError("Step-28 source fingerprint drift.")

    recompute = dict(payload)
    stored = recompute.pop("step28_source_fingerprint_sha256", None)
    if canonical_hash(recompute) != stored:
        raise RuntimeError("Step-28 source-freeze semantic no longer recomputes.")
    if not SOURCE_FROZEN_SCRIPT.is_file():
        raise FileNotFoundError(SOURCE_FROZEN_SCRIPT)
    if sha256_file(SOURCE_FROZEN_SCRIPT) != source_sha:
        raise RuntimeError("Frozen Step-28 source copy drift.")
    return fingerprint, source_sha


def verify_step27i() -> dict:
    required = [
        STEP27I_FIT_REGISTER, STEP27I_COMPLETION_REGISTER, STEP27I_FREEZE_JSON,
        STEP27I_MANIFEST_JSON, STEP27I_SOURCE_FREEZE_JSON, STEP27I_FROZEN_SOURCE,
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("Required Step-27I artifacts missing:\n  - " + "\n  - ".join(map(str, missing)))

    observed_manifest_sha = sha256_file(STEP27I_MANIFEST_JSON)
    if observed_manifest_sha != EXPECTED_STEP27I_MANIFEST_SHA256:
        raise RuntimeError("Step-27I manifest SHA drift.")

    manifest = load_json(STEP27I_MANIFEST_JSON)
    if manifest.get("stage1_execution_amendment_aware_fingerprint_sha256") != EXPECTED_STEP27I_STAGE1_FINGERPRINT:
        raise RuntimeError("Step-27I Stage-1 fingerprint drift.")
    if int(manifest.get("completed_fits", -1)) != EXPECTED_ROWS:
        raise RuntimeError("Step-27I does not report 80 fits.")
    if bool(manifest.get("configuration_selection_performed")):
        raise RuntimeError("Step-27I unexpectedly reports prior config selection.")
    if bool(manifest.get("threshold_optimization_performed")):
        raise RuntimeError("Step-27I unexpectedly reports threshold optimization.")
    if bool(manifest.get("best_seed_selection_performed")):
        raise RuntimeError("Step-27I unexpectedly reports best-seed selection.")
    if bool(manifest.get("final_test_accessed")):
        raise RuntimeError("Step-27I unexpectedly reports final-test access.")

    freeze = load_json(STEP27I_FREEZE_JSON)
    if freeze.get("stage1_execution_amendment_aware_fingerprint_sha256") != EXPECTED_STEP27I_STAGE1_FINGERPRINT:
        raise RuntimeError("Step-27I freeze fingerprint drift.")

    amend = freeze["protocol"]["timesnet_b_protocol_amendment"]
    if amend["amendment_manifest_sha256"] != EXPECTED_STEP27F_AMENDMENT_SHA256:
        raise RuntimeError("TimesNet-B amendment provenance drift.")
    if int(amend["physical_microbatch_new"]) != 64:
        raise RuntimeError("TimesNet-B amended microbatch drift.")

    expected_fit_hash = freeze["fit_register"]["sha256"]
    observed_fit_hash = sha256_file(STEP27I_FIT_REGISTER)
    if observed_fit_hash != expected_fit_hash:
        raise RuntimeError("Step-27I fit-register hash drift.")
    if int(freeze["fit_register"]["rows"]) != EXPECTED_ROWS:
        raise RuntimeError("Step-27I fit-register row count drift.")

    upstream = s27.verify_upstream()
    if upstream["global_final_test_state"] != "LOCKED_NOT_OPENED":
        raise RuntimeError("Global final-test state is not LOCKED_NOT_OPENED.")
    if upstream["test_seal_status"] != "SEALED_NOT_OPENED":
        raise RuntimeError("Step-10 final-test seal is not SEALED_NOT_OPENED.")

    return {
        "manifest_sha256": observed_manifest_sha,
        "fit_register_sha256": observed_fit_hash,
        "upstream": upstream,
    }


REQUIRED_COLUMNS = [
    "model_id","config_id","fold","seed","physical_microbatch_size","learning_rate",
    "train_n","validation_n","epochs_run","best_epoch",
    "best_validation_mean_directional_bce","stopped_early",
    "trainable_parameter_count","total_parameter_count","training_seconds",
    "wall_seconds","peak_cuda_allocated_mib","macro_balanced_accuracy",
    "macro_roc_auc","macro_mcc","macro_pr_auc","macro_brier_score",
    "fit_manifest_path","fit_manifest_sha256","worker_log",
    "1d_balanced_accuracy","1d_roc_auc","1d_mcc","1d_pr_auc","1d_brier_score",
    "3d_balanced_accuracy","3d_roc_auc","3d_mcc","3d_pr_auc","3d_brier_score",
    "5d_balanced_accuracy","5d_roc_auc","5d_mcc","5d_pr_auc","5d_brier_score",
]


def read_and_verify_fit_register() -> pd.DataFrame:
    df = pd.read_csv(STEP27I_FIT_REGISTER)
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected 80 fit rows; observed {len(df)}.")
    if list(df.columns) != REQUIRED_COLUMNS:
        raise RuntimeError(
            "Step-27I fit-register schema drift.\n"
            f"Missing: {sorted(set(REQUIRED_COLUMNS)-set(df.columns))}\n"
            f"Extra: {sorted(set(df.columns)-set(REQUIRED_COLUMNS))}"
        )
    if df[["model_id","config_id","fold"]].duplicated().any():
        raise RuntimeError("Duplicate model/config/fold rows.")
    if set(df["model_id"]) != set(s27.MODEL_ORDER):
        raise RuntimeError("Architecture set drift.")
    if len(df["config_id"].unique()) != EXPECTED_CONFIGS:
        raise RuntimeError("Expected 16 unique candidate configurations.")
    if set(df["seed"].astype(int)) != {EXPECTED_SEED}:
        raise RuntimeError("Stage-1 screening seed is not uniformly 42.")

    for model_id in s27.MODEL_ORDER:
        sub = df[df["model_id"] == model_id]
        configs = sorted(sub["config_id"].unique().tolist())
        expected = {f"{model_id}_A", f"{model_id}_B"}
        if set(configs) != expected:
            raise RuntimeError(f"{model_id}: expected {sorted(expected)}, observed {configs}.")
        for config_id in configs:
            g = sub[sub["config_id"] == config_id]
            if sorted(g["fold"].astype(int).tolist()) != EXPECTED_FOLDS:
                raise RuntimeError(f"{model_id}/{config_id}: fold coverage drift.")
            if len(g["trainable_parameter_count"].astype(int).unique()) != 1:
                raise RuntimeError(f"{model_id}/{config_id}: parameter-count drift.")

    for col in ["macro_balanced_accuracy","macro_roc_auc","macro_mcc","macro_pr_auc","macro_brier_score"]:
        if not np.isfinite(df[col].astype(float).to_numpy()).all():
            raise RuntimeError(f"Non-finite values in {col}.")

    tb = df[(df["model_id"]=="TIMESNET") & (df["config_id"]=="TIMESNET_B")]
    if set(tb["physical_microbatch_size"].astype(int)) != {64}:
        raise RuntimeError("TIMESNET_B does not reflect amended microbatch=64.")
    return df


def aggregate_configurations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id in s27.MODEL_ORDER:
        model_df = df[df["model_id"] == model_id]
        for config_id in sorted(model_df["config_id"].unique()):
            g = model_df[model_df["config_id"] == config_id].sort_values("fold")
            rows.append({
                "model_id": model_id,
                "config_id": config_id,
                "fold_count": len(g),
                "seed": int(g["seed"].iloc[0]),
                "physical_microbatch_size": int(g["physical_microbatch_size"].iloc[0]),
                "learning_rate": float(g["learning_rate"].iloc[0]),
                "trainable_parameter_count": int(g["trainable_parameter_count"].iloc[0]),
                "total_parameter_count": int(g["total_parameter_count"].iloc[0]),
                "mean_macro_balanced_accuracy": float(g["macro_balanced_accuracy"].mean()),
                "std_macro_balanced_accuracy": float(g["macro_balanced_accuracy"].std(ddof=1)),
                "mean_macro_roc_auc": float(g["macro_roc_auc"].mean()),
                "std_macro_roc_auc": float(g["macro_roc_auc"].std(ddof=1)),
                "mean_macro_mcc": float(g["macro_mcc"].mean()),
                "std_macro_mcc": float(g["macro_mcc"].std(ddof=1)),
                "mean_macro_pr_auc": float(g["macro_pr_auc"].mean()),
                "std_macro_pr_auc": float(g["macro_pr_auc"].std(ddof=1)),
                "mean_macro_brier_score": float(g["macro_brier_score"].mean()),
                "std_macro_brier_score": float(g["macro_brier_score"].std(ddof=1)),
                "mean_best_validation_bce": float(g["best_validation_mean_directional_bce"].mean()),
                "mean_training_seconds": float(g["training_seconds"].mean()),
                "mean_peak_cuda_allocated_mib": float(g["peak_cuda_allocated_mib"].mean()),
            })
    out = pd.DataFrame(rows)
    if len(out) != EXPECTED_CONFIGS:
        raise RuntimeError("Aggregate summary is not 16 rows.")
    return out


def rank_two_configs(group: pd.DataFrame) -> pd.DataFrame:
    if len(group) != 2:
        raise RuntimeError("A/B ranking requires exactly two configs.")
    ranked = group.sort_values(
        by=[
            "mean_macro_balanced_accuracy","mean_macro_roc_auc","mean_macro_mcc",
            "mean_macro_pr_auc","mean_macro_brier_score",
            "trainable_parameter_count","config_id",
        ],
        ascending=[False,False,False,False,True,True,True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["architecture_rank"] = [1,2]
    return ranked


def decisive_criterion(winner: pd.Series, runner: pd.Series) -> str:
    checks = [
        ("mean_macro_balanced_accuracy","higher"),
        ("mean_macro_roc_auc","higher"),
        ("mean_macro_mcc","higher"),
        ("mean_macro_pr_auc","higher"),
        ("mean_macro_brier_score","lower"),
        ("trainable_parameter_count","lower"),
        ("config_id","lexical"),
    ]
    for field, direction in checks:
        a, b = winner[field], runner[field]
        if field == "config_id":
            if str(a) != str(b):
                return field
        elif float(a) != float(b):
            if direction == "higher" and float(a) > float(b):
                return field
            if direction == "lower" and float(a) < float(b):
                return field
            raise RuntimeError(f"Internal ordering contradiction at {field}.")
    raise RuntimeError("Configs indistinguishable after lexical config_id.")


def select_configurations(summary_df: pd.DataFrame):
    selected_rows, decisions, ranked_groups = [], [], []
    for model_id in s27.MODEL_ORDER:
        ranked = rank_two_configs(summary_df[summary_df["model_id"] == model_id].copy())
        ranked_groups.append(ranked)
        w, r = ranked.iloc[0], ranked.iloc[1]
        crit = decisive_criterion(w, r)
        selected_rows.append({
            "model_id": model_id,
            "selected_config_id": w["config_id"],
            "runner_up_config_id": r["config_id"],
            "decisive_criterion": crit,
            "selected_mean_macro_balanced_accuracy": w["mean_macro_balanced_accuracy"],
            "runner_up_mean_macro_balanced_accuracy": r["mean_macro_balanced_accuracy"],
            "balanced_accuracy_difference": w["mean_macro_balanced_accuracy"] - r["mean_macro_balanced_accuracy"],
            "selected_mean_macro_roc_auc": w["mean_macro_roc_auc"],
            "runner_up_mean_macro_roc_auc": r["mean_macro_roc_auc"],
            "selected_mean_macro_mcc": w["mean_macro_mcc"],
            "runner_up_mean_macro_mcc": r["mean_macro_mcc"],
            "selected_mean_macro_pr_auc": w["mean_macro_pr_auc"],
            "runner_up_mean_macro_pr_auc": r["mean_macro_pr_auc"],
            "selected_mean_macro_brier_score": w["mean_macro_brier_score"],
            "runner_up_mean_macro_brier_score": r["mean_macro_brier_score"],
            "selected_trainable_parameter_count": int(w["trainable_parameter_count"]),
            "runner_up_trainable_parameter_count": int(r["trainable_parameter_count"]),
        })
        decisions.append({
            "model_id": model_id,
            "selected_config_id": str(w["config_id"]),
            "runner_up_config_id": str(r["config_id"]),
            "decisive_criterion": crit,
            "selected_full_precision": {
                "mean_macro_balanced_accuracy": float(w["mean_macro_balanced_accuracy"]),
                "mean_macro_roc_auc": float(w["mean_macro_roc_auc"]),
                "mean_macro_mcc": float(w["mean_macro_mcc"]),
                "mean_macro_pr_auc": float(w["mean_macro_pr_auc"]),
                "mean_macro_brier_score": float(w["mean_macro_brier_score"]),
                "trainable_parameter_count": int(w["trainable_parameter_count"]),
            },
            "runner_up_full_precision": {
                "mean_macro_balanced_accuracy": float(r["mean_macro_balanced_accuracy"]),
                "mean_macro_roc_auc": float(r["mean_macro_roc_auc"]),
                "mean_macro_mcc": float(r["mean_macro_mcc"]),
                "mean_macro_pr_auc": float(r["mean_macro_pr_auc"]),
                "mean_macro_brier_score": float(r["mean_macro_brier_score"]),
                "trainable_parameter_count": int(r["trainable_parameter_count"]),
            },
        })
    selected_df = pd.DataFrame(selected_rows)
    ranked_df = pd.concat(ranked_groups, ignore_index=True)
    if len(selected_df) != EXPECTED_MODELS:
        raise RuntimeError("Expected 8 architecture selections.")
    return selected_df, decisions, ranked_df


def prepare_state() -> dict:
    # Source is frozen before reading performance.
    source_fp, source_sha = ensure_source_freeze()
    step16_refs = find_step16_protocol_references()
    step27i = verify_step27i()
    fit_df = read_and_verify_fit_register()
    summary_df = aggregate_configurations(fit_df)
    selected_df, decisions, ranked_df = select_configurations(summary_df)
    upstream_after = s27.verify_upstream()
    if upstream_after["global_final_test_state"] != "LOCKED_NOT_OPENED":
        raise RuntimeError("Final-test state changed during Step 28.")
    if upstream_after["test_seal_status"] != "SEALED_NOT_OPENED":
        raise RuntimeError("Final-test seal changed during Step 28.")
    return {
        "source_fp": source_fp, "source_sha": source_sha,
        "step16_refs": step16_refs, "step27i": step27i,
        "fit_df": fit_df, "summary_df": summary_df,
        "selected_df": selected_df, "decisions": decisions,
        "ranked_df": ranked_df, "upstream_after": upstream_after,
    }


def print_preview(state: dict) -> None:
    print("\n" + "=" * 140)
    print("DEVELOPMENT-ONLY SELECTION PREVIEW — NOT YET FROZEN")
    print("=" * 140)
    for _, r in state["selected_df"].iterrows():
        print(
            f"{r['model_id']:<14} "
            f"select={r['selected_config_id']:<18} "
            f"runner={r['runner_up_config_id']:<18} "
            f"criterion={r['decisive_criterion']:<32} "
            f"BA={r['selected_mean_macro_balanced_accuracy']:.9f} "
            f"vs {r['runner_up_mean_macro_balanced_accuracy']:.9f} "
            f"Δ={r['balanced_accuracy_difference']:+.9f}"
        )


def prepare_only() -> int:
    print("=" * 140)
    print(STEP_NAME)
    print("=" * 140)
    state = prepare_state()
    print(f"[PASS] Step-28 source SHA-256            : {state['source_sha']}")
    print(f"[PASS] Step-28 source fingerprint        : {state['source_fp']}")
    print(f"[PASS] Step-16 selection protocol fp     : {EXPECTED_STEP16_SELECTION_PROTOCOL_FINGERPRINT}")
    print(f"[PASS] Step-16 references found          : {len(state['step16_refs'])}")
    for ref in state["step16_refs"]:
        print(f"       - {ref['path']}")
        print(f"         SHA-256: {ref['sha256']}")
    print(f"[PASS] Step-27I Stage-1 fingerprint      : {EXPECTED_STEP27I_STAGE1_FINGERPRINT}")
    print(f"[PASS] Step-27I manifest SHA-256         : {state['step27i']['manifest_sha256']}")
    print(f"[PASS] Step-27I fit register SHA-256     : {state['step27i']['fit_register_sha256']}")
    print(f"[PASS] Verified fit rows                 : {len(state['fit_df'])}/80")
    print(f"[PASS] Aggregated candidate rows         : {len(state['summary_df'])}/16")
    print(f"[PASS] Architecture selections computed  : {len(state['selected_df'])}/8")
    print("\n[PASS] Frozen selection hierarchy:")
    for item in SELECTION_RULE:
        print(f"       {item['priority']}. {item['field']} ({item['direction']})")
    print_preview(state)
    print("\n" + "=" * 140)
    print("STEP 28 PREPARE STATUS: PASS")
    print("=" * 140)
    print("Selection outputs frozen          : False")
    print("Configuration selection committed : False")
    print("Threshold optimization            : NOT PERFORMED")
    print("Best-seed selection               : NOT PERFORMED")
    print("Final-test data opened             : False")
    print(f"Final-test state                   : {state['upstream_after']['global_final_test_state']}")
    print("=" * 140)
    print("NEXT: audit this preview, then run without --prepare-only to freeze the eight A/B selections.")
    return 0


def build_selection_semantic(state: dict) -> dict:
    records = []
    for _, r in state["selected_df"].iterrows():
        records.append({
            "model_id": str(r["model_id"]),
            "selected_config_id": str(r["selected_config_id"]),
            "runner_up_config_id": str(r["runner_up_config_id"]),
            "decisive_criterion": str(r["decisive_criterion"]),
            "selected_mean_macro_balanced_accuracy": float(r["selected_mean_macro_balanced_accuracy"]),
            "runner_up_mean_macro_balanced_accuracy": float(r["runner_up_mean_macro_balanced_accuracy"]),
            "balanced_accuracy_difference": float(r["balanced_accuracy_difference"]),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "DEVELOPMENT_ONLY_AB_CONFIGURATION_SELECTION_FROZEN",
        "configuration_fingerprint_sha256": EXPECTED_CONFIGURATION_FINGERPRINT,
        "upstream": {
            "step16_selection_protocol_fingerprint_sha256": EXPECTED_STEP16_SELECTION_PROTOCOL_FINGERPRINT,
            "step27i_stage1_execution_fingerprint_sha256": EXPECTED_STEP27I_STAGE1_FINGERPRINT,
            "step27i_manifest_sha256": EXPECTED_STEP27I_MANIFEST_SHA256,
            "step27i_fit_register_sha256": state["step27i"]["fit_register_sha256"],
            "step27f_protocol_amendment_sha256": EXPECTED_STEP27F_AMENDMENT_SHA256,
            "step28_source_fingerprint_sha256": state["source_fp"],
            "step28_source_sha256": state["source_sha"],
        },
        "selection_scope": {
            "data": "development_validation_folds_only",
            "architectures": 8, "candidate_configurations": 16,
            "folds_per_configuration": 5, "selected_configurations": 8,
            "seed": 42, "screening_threshold": 0.5,
        },
        "selection_rule": SELECTION_RULE,
        "selection_rule_numerics": {
            "aggregate": "arithmetic mean over five chronological folds",
            "sort": "deterministic lexicographic sort",
            "tie_tolerance": None,
            "full_precision_used": True,
        },
        "selected_configurations": records,
        "protection_state": {
            "configuration_selection_performed": True,
            "threshold_optimization_performed": False,
            "best_seed_selection_performed": False,
            "final_test_data_opened": False,
            "final_test_metric_computed": False,
            "global_final_test_state": state["upstream_after"]["global_final_test_state"],
        },
    }


def freeze_selection(state: dict) -> tuple[str, str]:
    if MANIFEST_JSON.exists():
        return verify_existing_selection()

    ranked_df = state["ranked_df"].copy()
    order = {m: i for i, m in enumerate(s27.MODEL_ORDER)}
    ranked_df["_model_order"] = ranked_df["model_id"].map(order)
    ranked_df = ranked_df.sort_values(["_model_order","architecture_rank"]).drop(columns="_model_order").reset_index(drop=True)

    s27.atomic_csv(CONFIG_SUMMARY_CSV, ranked_df)
    s27.atomic_csv(SELECTED_CONFIGS_CSV, state["selected_df"])
    s27.atomic_json(SELECTION_DECISIONS_JSON, {
        "schema_version": SCHEMA_VERSION,
        "step": STEP_NUMBER,
        "selection_protocol_fingerprint_sha256": EXPECTED_STEP16_SELECTION_PROTOCOL_FINGERPRINT,
        "stage1_execution_fingerprint_sha256": EXPECTED_STEP27I_STAGE1_FINGERPRINT,
        "selection_rule": SELECTION_RULE,
        "decisions": state["decisions"],
        "configuration_selection_performed": True,
        "threshold_optimization_performed": False,
        "best_seed_selection_performed": False,
        "final_test_accessed": False,
    })

    semantic = build_selection_semantic(state)
    selection_fp = canonical_hash(semantic)
    s27.atomic_json(FREEZE_JSON, {
        **semantic,
        "generated_at_utc": s27.utc_now(),
        "configuration_selection_fingerprint_sha256": selection_fp,
    })

    summary_lines = [
        STEP_NAME, "=" * 112, "Status: PASS",
        "Selection scope: DEVELOPMENT VALIDATION ONLY",
        "Stage-1 fits consumed: 80/80",
        "Architectures: 8", "Candidate configurations: 16",
        "Selected configurations: 8", "Folds per configuration: 5",
        "Screening seed: 42", "Screening threshold: 0.5", "",
        "Frozen hierarchy:",
        "1. Higher mean 5-fold macro balanced accuracy",
        "2. Higher mean 5-fold macro ROC-AUC",
        "3. Higher mean 5-fold macro MCC",
        "4. Higher mean 5-fold macro PR-AUC",
        "5. Lower mean 5-fold macro Brier score",
        "6. Lower trainable parameter count",
        "7. Ascending lexical config_id", "", "Selected configurations:",
    ]
    for _, r in state["selected_df"].iterrows():
        summary_lines.append(
            f"  {r['model_id']}: {r['selected_config_id']} "
            f"(runner-up {r['runner_up_config_id']}; decisive {r['decisive_criterion']})"
        )
    summary_lines += [
        "", "Threshold optimization performed: False",
        "Best-seed selection performed: False",
        "Final-test data opened: False",
        f"Global final-test state: {state['upstream_after']['global_final_test_state']}",
    ]
    s27.atomic_text(SUMMARY_TXT, "\n".join(summary_lines) + "\n")

    pre_outputs = [
        CONFIG_SUMMARY_CSV, SELECTED_CONFIGS_CSV, SELECTION_DECISIONS_JSON,
        FREEZE_JSON, SUMMARY_TXT, SOURCE_FREEZE_JSON, SOURCE_FROZEN_SCRIPT,
    ]
    artifacts = {
        p.name: {"path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in pre_outputs
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "step": {"step_number": STEP_NUMBER, "step_name": STEP_NAME, "status": "PASS"},
        "frozen_at_utc": s27.utc_now(),
        "configuration_fingerprint_sha256": EXPECTED_CONFIGURATION_FINGERPRINT,
        "step16_selection_protocol_fingerprint_sha256": EXPECTED_STEP16_SELECTION_PROTOCOL_FINGERPRINT,
        "stage1_execution_fingerprint_sha256": EXPECTED_STEP27I_STAGE1_FINGERPRINT,
        "step27i_manifest_sha256": EXPECTED_STEP27I_MANIFEST_SHA256,
        "step28_source_fingerprint_sha256": state["source_fp"],
        "configuration_selection_fingerprint_sha256": selection_fp,
        "selected_configurations": 8,
        "configuration_selection_performed": True,
        "threshold_optimization_performed": False,
        "best_seed_selection_performed": False,
        "final_test_accessed": False,
        "artifacts": artifacts,
    }
    s27.atomic_json(MANIFEST_JSON, manifest)
    manifest_sha = sha256_file(MANIFEST_JSON)
    s27.atomic_text(MANIFEST_SHA, f"{manifest_sha}  {MANIFEST_JSON.name}\n")

    hash_rows = []
    for p in (*pre_outputs, MANIFEST_JSON, MANIFEST_SHA):
        hash_rows.append({"path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    s27.atomic_csv(OUTPUT_HASH_CSV, pd.DataFrame(hash_rows))

    upstream = s27.verify_upstream()
    if upstream["global_final_test_state"] != "LOCKED_NOT_OPENED":
        raise RuntimeError("Final-test state changed during Step-28 freeze.")
    if upstream["test_seal_status"] != "SEALED_NOT_OPENED":
        raise RuntimeError("Final-test seal changed during Step-28 freeze.")
    return selection_fp, manifest_sha


def verify_existing_selection() -> tuple[str, str]:
    required = [
        CONFIG_SUMMARY_CSV, SELECTED_CONFIGS_CSV, SELECTION_DECISIONS_JSON,
        FREEZE_JSON, SUMMARY_TXT, SOURCE_FREEZE_JSON, SOURCE_FROZEN_SCRIPT,
        MANIFEST_JSON, MANIFEST_SHA, OUTPUT_HASH_CSV,
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("Existing Step-28 freeze incomplete:\n  - " + "\n  - ".join(map(str, missing)))

    freeze = load_json(FREEZE_JSON)
    if freeze.get("status") != "DEVELOPMENT_ONLY_AB_CONFIGURATION_SELECTION_FROZEN":
        raise RuntimeError("Step-28 freeze status drift.")
    fp = freeze.get("configuration_selection_fingerprint_sha256")
    semantic = dict(freeze)
    semantic.pop("generated_at_utc", None)
    semantic.pop("configuration_selection_fingerprint_sha256", None)
    if canonical_hash(semantic) != fp:
        raise RuntimeError("Step-28 selection fingerprint no longer recomputes.")

    selected = pd.read_csv(SELECTED_CONFIGS_CSV)
    if len(selected) != 8:
        raise RuntimeError("Selected-config table is not 8 rows.")

    manifest = load_json(MANIFEST_JSON)
    if not manifest.get("configuration_selection_performed"):
        raise RuntimeError("Manifest does not report selection performed.")
    if manifest.get("threshold_optimization_performed") or manifest.get("best_seed_selection_performed") or manifest.get("final_test_accessed"):
        raise RuntimeError("Manifest protection state drift.")

    manifest_sha = sha256_file(MANIFEST_JSON)
    if MANIFEST_SHA.read_text(encoding="utf-8").strip() != f"{manifest_sha}  {MANIFEST_JSON.name}":
        raise RuntimeError("Step-28 manifest SHA sidecar drift.")

    hashes = pd.read_csv(OUTPUT_HASH_CSV)
    for _, row in hashes.iterrows():
        p = Path(row["path"])
        if not p.is_file() or sha256_file(p) != str(row["sha256"]):
            raise RuntimeError(f"Step-28 output hash drift: {p}")

    upstream = s27.verify_upstream()
    if upstream["global_final_test_state"] != "LOCKED_NOT_OPENED":
        raise RuntimeError("Final-test state changed after Step-28.")
    return str(fp), manifest_sha


def final_selection() -> int:
    print("=" * 140)
    print(STEP_NAME)
    print("=" * 140)

    if MANIFEST_JSON.exists():
        fp, msha = verify_existing_selection()
        print("[PASS] Existing Step-28 selection freeze verified.")
        print(f"       Selection fingerprint : {fp}")
        print(f"       Manifest SHA-256      : {msha}")
        print("       Selected configs      : 8/8")
        print("       Threshold tuning      : NOT PERFORMED")
        print("       Best-seed selection   : NOT PERFORMED")
        print("       Final-test opened     : False")
        return 0

    state = prepare_state()
    print("[1/4] Frozen Step-16 selection protocol reference verified.")
    print("[2/4] Step-27I amendment-aware 80-fit corpus verified.")
    print("[3/4] Five-fold development metrics aggregated; final test remains sealed.")
    print("[4/4] Applying and freezing deterministic A/B selections...")
    fp, msha = freeze_selection(state)

    vfp, vmsha = verify_existing_selection()
    if vfp != fp or vmsha != msha:
        raise RuntimeError("Post-write Step-28 verification mismatch.")

    print_preview(state)
    print("\n" + "=" * 140)
    print("STEP 28 — DEVELOPMENT-ONLY A/B SELECTION COMPLETE")
    print("=" * 140)
    print("Architectures selected            : 8/8")
    print("Candidate configurations consumed : 16/16")
    print("Chronological folds per config    : 5")
    print("Primary criterion                 : mean macro balanced accuracy")
    print("Screening threshold               : 0.5 (UNCHANGED)")
    print("Threshold optimization            : NOT PERFORMED")
    print("Best-seed selection               : NOT PERFORMED")
    print("Final-test data opened             : False")
    print(f"Selection fingerprint             : {fp}")
    print(f"Step-28 manifest SHA-256          : {msha}")
    print(f"Global final-test state           : {state['upstream_after']['global_final_test_state']}")
    print("=" * 140)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    return prepare_only() if args.prepare_only else final_selection()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStep 28 interrupted. No training or final-test access occurs in this step.")
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print("\n" + "=" * 140)
        print("STEP 28 FAILED")
        print("=" * 140)
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        print("=" * 140)
        raise SystemExit(1)
