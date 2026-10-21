#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 31 — FIVE-SEED DEVELOPMENT OOF PROBABILITY AGGREGATION
============================================================

Purpose
-------
Construct the frozen development-only five-seed probability endpoint for
each selected architecture.

For each architecture × rolling-origin validation fold:

    seeds = {42, 43, 44, 45, 46}

    p_bar(i,h) = mean_s p_s(i,h)

where h ∈ {1d, 3d, 5d}.

Important
---------
This step averages PROBABILITIES, not logits.

It verifies:
    - Step-30 global Stage-2 freeze;
    - all 200 prediction files against the Step-30 frozen inventory;
    - row_index identity across the five seeds;
    - validation-target identity across seeds;
    - validation-target identity across architectures for each fold;
    - horizon order = [1, 3, 5];
    - probability/logit internal consistency;
    - binary labels and valid probability range;
    - no seed selection;
    - no threshold optimization;
    - no final-test use.

Outputs
-------
1. One five-seed probability artifact per architecture × fold.
2. One pooled development OOF artifact per architecture.
3. Seed × fold × horizon metrics.
4. Seed-pooled OOF metrics.
5. Mean/std robustness summary across the five fixed seeds.
6. Five-seed fold metrics.
7. Five-seed pooled OOF metrics.
8. Frozen artifact inventory.
9. Step-31 final freeze manifest.

Threshold 0.5 is used ONLY for reporting diagnostic BA/MCC.
No threshold is optimized here.

Final test remains LOCKED_NOT_OPENED.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent

ROOT = (
    HERE.parent
    / "stock_q1_revision_project"
    / "experiments"
    / "q1_revision_v3_clean_2016_2026"
).resolve()

AUDIT_DIR = ROOT / "audits"
MANIFEST_DIR = ROOT / "manifests"

STEP30_MANIFEST = (
    MANIFEST_DIR
    / "30_stage2_global_final_freeze_manifest.json"
)

OOF_ROOT = ROOT / "stage2_five_seed_oof"

SEED_FOLD_METRICS_CSV = (
    AUDIT_DIR
    / "31_seed_fold_horizon_metrics.csv"
)

SEED_POOLED_METRICS_CSV = (
    AUDIT_DIR
    / "31_seed_pooled_oof_metrics.csv"
)

SEED_ROBUSTNESS_SUMMARY_CSV = (
    AUDIT_DIR
    / "31_seed_robustness_mean_std.csv"
)

FIVE_SEED_FOLD_METRICS_CSV = (
    AUDIT_DIR
    / "31_five_seed_fold_metrics.csv"
)

FIVE_SEED_POOLED_METRICS_CSV = (
    AUDIT_DIR
    / "31_five_seed_pooled_oof_metrics.csv"
)

ARTIFACT_INVENTORY_CSV = (
    AUDIT_DIR
    / "31_five_seed_oof_artifact_inventory.csv"
)

ALIGNMENT_AUDIT_CSV = (
    AUDIT_DIR
    / "31_fold_alignment_audit.csv"
)

FINAL_MANIFEST = (
    MANIFEST_DIR
    / "31_stage2_five_seed_oof_final_freeze_manifest.json"
)

FROZEN_SOURCE = (
    MANIFEST_DIR
    / "step31_stage2_five_seed_oof_aggregation_v1_frozen.py"
)


# =============================================================================
# AUTHORITATIVE STEP-30 FREEZE
# =============================================================================

EXPECTED_STEP30_MANIFEST_SHA256 = (
    "8216ac6947a2c2c2741758f76d9c38f085f59ac4158f9fcdc58effb18c1ab420"
)

EXPECTED_STEP30_GLOBAL_FINGERPRINT = (
    "328c77bac04ad61dabb9fe50f9b481389a94b95709cefe3c00fc680e7c807bed"
)


# =============================================================================
# FROZEN STAGE-2 ENDPOINT CONTRACT
# =============================================================================

SELECTED = {
    "DLINEAR": "DLINEAR_B",
    "NLINEAR": "NLINEAR_B",
    "PATCHTST": "PATCHTST_B",
    "TIMESNET": "TIMESNET_B",
    "ITRANSFORMER": "ITRANSFORMER_A",
    "FREDFORMER": "FREDFORMER_B",
    "FRNET": "FRNET_A",
    "MLF": "MLF_A",
}

SEEDS = [42, 43, 44, 45, 46]
FOLDS = [1, 2, 3, 4, 5]
HORIZONS = [1, 3, 5]

SCREENING_THRESHOLD = 0.5

EXPECTED_LOGICAL_FITS = 200
EXPECTED_FITS_PER_MODEL = 25
EXPECTED_FITS_PER_MODEL_FOLD = 5

EXPECTED_FOLD_ROWS = {
    1: 4900,
    2: 4860,
    3: 4860,
    4: 4800,
    5: 4820,
}

EXPECTED_POOLED_ROWS = sum(EXPECTED_FOLD_ROWS.values())  # 24,240

METRIC_NAMES = [
    "balanced_accuracy",
    "roc_auc",
    "mcc",
    "pr_auc",
    "brier",
    "bce",
]


# =============================================================================
# HELPERS
# =============================================================================

def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if not path.is_file():
        fail(f"Missing file for hashing: {path}")

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()

    for a in arrays:
        x = np.ascontiguousarray(a)

        h.update(str(x.dtype).encode("utf-8"))
        h.update(str(tuple(x.shape)).encode("utf-8"))
        h.update(x.tobytes(order="C"))

    return h.hexdigest()


def canonical_json_sha256(obj) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"Missing JSON file: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        fail(f"Expected JSON object: {path}")

    return obj


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        fail(
            f"{label}: expected {expected!r}, got {actual!r}"
        )


def require_false(value, label: str) -> None:
    if value is not False:
        fail(f"{label}: expected False, got {value!r}")


def manifest_input_path(m: dict, key: str) -> Path:
    try:
        x = m["inputs"][key]
    except KeyError:
        fail(f"Manifest missing inputs.{key}")

    if isinstance(x, dict):
        x = x.get("path")

    if not x:
        fail(f"Manifest has no path for inputs.{key}")

    return Path(str(x)).expanduser().resolve()


def parse_horizon_order(value) -> list[int]:
    if not isinstance(value, (list, tuple)):
        fail(f"Invalid horizon_order: {value!r}")

    out = []

    for x in value:
        if isinstance(x, (int, np.integer)):
            out.append(int(x))
            continue

        matches = re.findall(r"\d+", str(x))

        if len(matches) != 1:
            fail(f"Cannot parse horizon entry: {x!r}")

        out.append(int(matches[0]))

    return out


def normalize_threshold_operator(value) -> str:
    s = str(value).strip().lower().replace(" ", "")

    ge = {
        ">=",
        "ge",
        "gte",
        "greater_equal",
        "greaterthanorequal",
        "greater_than_or_equal",
    }

    gt = {
        ">",
        "gt",
        "greater",
        "greaterthan",
        "greater_than",
    }

    if s in ge:
        return ">="

    if s in gt:
        return ">"

    fail(f"Unsupported threshold operator: {value!r}")


def hard_predictions(
    probabilities: np.ndarray,
    operator: str,
) -> np.ndarray:

    if operator == ">=":
        return (probabilities >= SCREENING_THRESHOLD).astype(np.int8)

    if operator == ">":
        return (probabilities > SCREENING_THRESHOLD).astype(np.int8)

    fail(f"Unsupported normalized operator: {operator}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold_operator: str,
) -> dict:

    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)

    if y.ndim != 1 or p.ndim != 1:
        fail("Metric inputs must be one-dimensional.")

    if len(y) != len(p):
        fail("Metric label/probability length mismatch.")

    if len(y) == 0:
        fail("Empty metric input.")

    values = set(np.unique(y).tolist())

    if not values.issubset({0, 1}):
        fail(f"Non-binary labels detected: {values}")

    if values != {0, 1}:
        fail(
            "Both classes are required for development OOF metrics; "
            f"found {values}"
        )

    if not np.all(np.isfinite(p)):
        fail("Non-finite probabilities detected.")

    if np.any(p < 0.0) or np.any(p > 1.0):
        fail("Probability outside [0,1].")

    pred = hard_predictions(p, threshold_operator)

    eps = np.finfo(np.float64).eps
    pc = np.clip(p, eps, 1.0 - eps)

    bce = -np.mean(
        y * np.log(pc)
        + (1 - y) * np.log(1.0 - pc)
    )

    return {
        "balanced_accuracy":
            float(balanced_accuracy_score(y, pred)),

        "roc_auc":
            float(roc_auc_score(y, p)),

        "mcc":
            float(matthews_corrcoef(y, pred)),

        "pr_auc":
            float(average_precision_score(y, p)),

        "brier":
            float(np.mean((p - y) ** 2)),

        "bce":
            float(bce),
    }


def add_metric_row(
    rows: list[dict],
    base: dict,
    metrics: dict,
) -> None:

    row = dict(base)
    row.update(metrics)
    rows.append(row)


def freeze_source() -> tuple[str, str]:
    src = Path(__file__).resolve()
    current_sha = sha256_file(src)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    if FROZEN_SOURCE.exists():

        frozen_sha = sha256_file(FROZEN_SOURCE)

        if frozen_sha != current_sha:
            fail(
                "Step-31 frozen source already exists but differs from "
                "the executable source.\n"
                f"Frozen : {frozen_sha}\n"
                f"Current: {current_sha}"
            )

    else:
        shutil.copy2(src, FROZEN_SOURCE)

    frozen_sha = sha256_file(FROZEN_SOURCE)

    require_equal(
        frozen_sha,
        current_sha,
        "Step31 frozen-source SHA256",
    )

    return current_sha, frozen_sha


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 140)
    print(
        "STEP 31 — DEVELOPMENT-ONLY FIVE-SEED "
        "OOF PROBABILITY AGGREGATION"
    )
    print("=" * 140)
    print(f"Experiment root : {ROOT}")
    print("Seeds           : 42,43,44,45,46")
    print("Horizons        : 1d,3d,5d")
    print("Aggregation     : arithmetic mean of PROBABILITIES")
    print("Seed selection  : NONE")
    print("Threshold tune  : NONE")
    print("Final test      : LOCKED_NOT_OPENED")
    print("-" * 140)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 0. Preserve an already-established final Step-31 freeze.
    # -------------------------------------------------------------------------

    if FINAL_MANIFEST.exists():
        print("Step-31 final freeze already exists:")
        print(FINAL_MANIFEST)
        print()
        print("No Step-31 artifacts were modified.")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # 1. Verify Step-30 authoritative freeze.
    # -------------------------------------------------------------------------

    step30_sha = sha256_file(STEP30_MANIFEST)

    require_equal(
        step30_sha,
        EXPECTED_STEP30_MANIFEST_SHA256,
        "Step30 freeze-manifest SHA256",
    )

    step30 = load_json(STEP30_MANIFEST)

    require_equal(
        step30.get("status"),
        "STAGE2_GLOBAL_ROBUSTNESS_FINAL_FREEZE_COMPLETE",
        "Step30 status",
    )

    require_equal(
        step30.get("global_stage2_freeze_fingerprint"),
        EXPECTED_STEP30_GLOBAL_FINGERPRINT,
        "Step30 global freeze fingerprint",
    )

    ps30 = step30.get("protection_state", {})

    require_false(
        ps30.get("best_seed_selection_performed"),
        "Step30 best_seed_selection_performed",
    )

    require_false(
        ps30.get("best_seed_promotion_performed"),
        "Step30 best_seed_promotion_performed",
    )

    require_false(
        ps30.get("threshold_optimization_performed"),
        "Step30 threshold_optimization_performed",
    )

    require_false(
        ps30.get("final_test_data_opened"),
        "Step30 final_test_data_opened",
    )

    require_false(
        ps30.get("final_test_metric_computed"),
        "Step30 final_test_metric_computed",
    )

    require_equal(
        ps30.get("final_test_state"),
        "LOCKED_NOT_OPENED",
        "Step30 final test state",
    )

    inv_meta = step30["outputs"]["logical_fit_inventory"]

    inventory_path = Path(
        inv_meta["path"]
    ).expanduser().resolve()

    inventory_sha = sha256_file(inventory_path)

    require_equal(
        inventory_sha,
        inv_meta["sha256"],
        "Step30 logical inventory SHA256",
    )

    inventory = pd.read_csv(inventory_path)

    require_equal(
        len(inventory),
        EXPECTED_LOGICAL_FITS,
        "Step30 logical inventory rows",
    )

    print("[1] STEP-30 GLOBAL FREEZE")
    print(f"Freeze manifest SHA256 : {step30_sha}")
    print(
        "Global fingerprint    : "
        f"{EXPECTED_STEP30_GLOBAL_FINGERPRINT}"
    )
    print(f"Logical inventory rows : {len(inventory)}/200")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 2. Freeze Step-31 source.
    # -------------------------------------------------------------------------

    source_sha, frozen_source_sha = freeze_source()

    print("[2] STEP-31 SOURCE FREEZE")
    print(f"Executable SHA256 : {source_sha}")
    print(f"Frozen SHA256     : {frozen_source_sha}")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 3. Clean only unfinished Step-31 namespace.
    # -------------------------------------------------------------------------

    if OOF_ROOT.exists():
        shutil.rmtree(OOF_ROOT)

    OOF_ROOT.mkdir(parents=True, exist_ok=True)

    for p in [
        SEED_FOLD_METRICS_CSV,
        SEED_POOLED_METRICS_CSV,
        SEED_ROBUSTNESS_SUMMARY_CSV,
        FIVE_SEED_FOLD_METRICS_CSV,
        FIVE_SEED_POOLED_METRICS_CSV,
        ARTIFACT_INVENTORY_CSV,
        ALIGNMENT_AUDIT_CSV,
    ]:
        if p.exists():
            p.unlink()

    # -------------------------------------------------------------------------
    # 4. Inventory structural validation.
    # -------------------------------------------------------------------------

    required_cols = {
        "model_id",
        "config_id",
        "seed",
        "fold",
        "fit_manifest_path",
        "fit_manifest_sha256",
        "validation_predictions_path",
        "validation_predictions_sha256",
    }

    missing = required_cols - set(inventory.columns)

    if missing:
        fail(
            "Step30 inventory missing columns: "
            f"{sorted(missing)}"
        )

    keys = ["model_id", "config_id", "seed", "fold"]

    if inventory.duplicated(keys).any():
        fail("Duplicate logical-fit keys in Step30 inventory.")

    require_equal(
        set(inventory["model_id"].astype(str)),
        set(SELECTED),
        "Architecture set",
    )

    for model_id, config_id in SELECTED.items():

        x = inventory.loc[
            inventory["model_id"].astype(str) == model_id
        ]

        require_equal(
            len(x),
            EXPECTED_FITS_PER_MODEL,
            f"{model_id}: inventory fit count",
        )

        require_equal(
            set(x["config_id"].astype(str)),
            {config_id},
            f"{model_id}: config",
        )

        require_equal(
            set(x["seed"].astype(int)),
            set(SEEDS),
            f"{model_id}: seeds",
        )

        require_equal(
            set(x["fold"].astype(int)),
            set(FOLDS),
            f"{model_id}: folds",
        )

    print("[3] LOGICAL INVENTORY CONTRACT")
    print("Architectures            : 8/8")
    print("Logical fits             : 200/200")
    print("Fits/architecture        : 25")
    print("Seeds/architecture       : 5")
    print("Folds/seed               : 5")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # Containers
    # -------------------------------------------------------------------------

    seed_fold_metric_rows = []
    seed_pooled_metric_rows = []

    five_seed_fold_metric_rows = []
    five_seed_pooled_metric_rows = []

    artifact_rows = []
    alignment_rows = []

    # Cross-architecture fold references.
    global_fold_reference = {}

    threshold_operator_global = None

    # -------------------------------------------------------------------------
    # 5. Model-wise five-seed aggregation.
    # -------------------------------------------------------------------------

    for model_id, config_id in SELECTED.items():

        print(
            f"[MODEL] {model_id:14s} "
            f"{config_id:22s}"
        )

        model_dir = OOF_ROOT / model_id / config_id
        model_dir.mkdir(parents=True, exist_ok=True)

        pooled_mean_prob_parts = []
        pooled_label_parts = []
        pooled_fold_parts = []
        pooled_row_index_parts = []

        per_seed_pooled_prob = {
            seed: [] for seed in SEEDS
        }

        per_seed_pooled_label = {
            seed: [] for seed in SEEDS
        }

        for fold in FOLDS:

            subset = inventory.loc[
                (inventory["model_id"].astype(str) == model_id)
                & (inventory["config_id"].astype(str) == config_id)
                & (inventory["fold"].astype(int) == fold)
            ].copy()

            subset = subset.sort_values("seed")

            require_equal(
                len(subset),
                EXPECTED_FITS_PER_MODEL_FOLD,
                f"{model_id}/fold{fold}: seed count",
            )

            require_equal(
                subset["seed"].astype(int).tolist(),
                SEEDS,
                f"{model_id}/fold{fold}: seed ordering",
            )

            fold_probabilities = []

            canonical_row_index = None
            canonical_labels = None
            canonical_y_path = None
            canonical_y_sha = None
            canonical_alignment_fp = None

            for r in subset.itertuples(index=False):

                seed = int(r.seed)

                context = (
                    f"{model_id}/{config_id}/"
                    f"seed={seed}/fold={fold}"
                )

                manifest_path = Path(
                    str(r.fit_manifest_path)
                ).expanduser().resolve()

                prediction_path = Path(
                    str(r.validation_predictions_path)
                ).expanduser().resolve()

                # -------------------------------------------------------------
                # Frozen file-hash verification.
                # -------------------------------------------------------------

                require_equal(
                    sha256_file(manifest_path),
                    str(r.fit_manifest_sha256),
                    f"{context}: fit manifest SHA256",
                )

                require_equal(
                    sha256_file(prediction_path),
                    str(r.validation_predictions_sha256),
                    f"{context}: prediction SHA256",
                )

                m = load_json(manifest_path)

                require_equal(
                    str(m.get("model_id")),
                    model_id,
                    f"{context}: model_id",
                )

                require_equal(
                    str(m.get("config_id")),
                    config_id,
                    f"{context}: config_id",
                )

                require_equal(
                    int(m.get("seed", -1)),
                    seed,
                    f"{context}: seed",
                )

                require_equal(
                    int(m.get("fold", -1)),
                    fold,
                    f"{context}: fold",
                )

                require_equal(
                    parse_horizon_order(
                        m.get("horizon_order")
                    ),
                    HORIZONS,
                    f"{context}: horizon_order",
                )

                require_equal(
                    bool(m.get("validation_shuffle")),
                    False,
                    f"{context}: validation_shuffle",
                )

                if abs(
                    float(m.get("threshold", -999.0))
                    - SCREENING_THRESHOLD
                ) > 1e-12:
                    fail(
                        f"{context}: screening threshold "
                        "is not 0.5"
                    )

                op = normalize_threshold_operator(
                    m.get("threshold_operator")
                )

                if threshold_operator_global is None:
                    threshold_operator_global = op
                else:
                    require_equal(
                        op,
                        threshold_operator_global,
                        f"{context}: threshold operator",
                    )

                pstate = m.get("protection_state", {})

                require_false(
                    pstate.get("final_test_data_opened"),
                    f"{context}: final_test_data_opened",
                )

                # -------------------------------------------------------------
                # Validation labels.
                # -------------------------------------------------------------

                y_path = manifest_input_path(
                    m,
                    "validation_y",
                )

                y_sha = sha256_file(y_path)

                y_full = np.load(
                    y_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )

                if y_full.ndim != 2:
                    fail(
                        f"{context}: validation_y must be "
                        f"2-D, got {y_full.shape}"
                    )

                if y_full.shape[1] != 3:
                    fail(
                        f"{context}: expected 3 target horizons, "
                        f"got {y_full.shape}"
                    )

                # -------------------------------------------------------------
                # Prediction artifact.
                # -------------------------------------------------------------

                with np.load(
                    prediction_path,
                    allow_pickle=False,
                ) as z:

                    required_npz_keys = {
                        "logits",
                        "probabilities",
                        "row_index",
                    }

                    require_equal(
                        set(z.files),
                        required_npz_keys,
                        f"{context}: prediction NPZ keys",
                    )

                    logits = np.asarray(
                        z["logits"],
                        dtype=np.float64,
                    )

                    probabilities = np.asarray(
                        z["probabilities"],
                        dtype=np.float64,
                    )

                    row_index = np.asarray(
                        z["row_index"],
                        dtype=np.int64,
                    )

                # -------------------------------------------------------------
                # Structural checks.
                # -------------------------------------------------------------

                expected_rows = EXPECTED_FOLD_ROWS[fold]

                require_equal(
                    probabilities.shape,
                    (expected_rows, 3),
                    f"{context}: probability shape",
                )

                require_equal(
                    logits.shape,
                    (expected_rows, 3),
                    f"{context}: logits shape",
                )

                require_equal(
                    row_index.shape,
                    (expected_rows,),
                    f"{context}: row_index shape",
                )

                if len(np.unique(row_index)) != len(row_index):
                    fail(
                        f"{context}: duplicate row_index "
                        "values detected."
                    )

                if row_index.min() < 0:
                    fail(
                        f"{context}: negative row_index."
                    )

                if row_index.max() >= y_full.shape[0]:
                    fail(
                        f"{context}: row_index outside "
                        "validation_y bounds."
                    )

                if not np.all(np.isfinite(probabilities)):
                    fail(
                        f"{context}: non-finite probability."
                    )

                if (
                    np.any(probabilities < 0.0)
                    or np.any(probabilities > 1.0)
                ):
                    fail(
                        f"{context}: probability outside [0,1]."
                    )

                implied_prob = sigmoid(logits)

                if not np.allclose(
                    implied_prob,
                    probabilities,
                    rtol=2e-6,
                    atol=2e-6,
                ):
                    max_diff = float(
                        np.max(
                            np.abs(
                                implied_prob
                                - probabilities
                            )
                        )
                    )

                    fail(
                        f"{context}: probabilities do not "
                        "match sigmoid(logits). "
                        f"max_abs_diff={max_diff}"
                    )

                labels = np.asarray(
                    y_full[row_index],
                    dtype=np.int8,
                )

                label_values = set(
                    np.unique(labels).tolist()
                )

                if not label_values.issubset({0, 1}):
                    fail(
                        f"{context}: non-binary labels "
                        f"{label_values}"
                    )

                alignment_fp = sha256_arrays(
                    row_index.astype(np.int64),
                    labels.astype(np.int8),
                )

                # -------------------------------------------------------------
                # Five seeds must reference exactly the same observations.
                # -------------------------------------------------------------

                if canonical_row_index is None:

                    canonical_row_index = row_index.copy()
                    canonical_labels = labels.copy()
                    canonical_y_path = y_path
                    canonical_y_sha = y_sha
                    canonical_alignment_fp = alignment_fp

                else:

                    require_equal(
                        y_path,
                        canonical_y_path,
                        f"{context}: validation_y path",
                    )

                    require_equal(
                        y_sha,
                        canonical_y_sha,
                        f"{context}: validation_y SHA256",
                    )

                    if not np.array_equal(
                        row_index,
                        canonical_row_index,
                    ):
                        fail(
                            f"{context}: row_index differs "
                            "across seeds."
                        )

                    if not np.array_equal(
                        labels,
                        canonical_labels,
                    ):
                        fail(
                            f"{context}: labels differ "
                            "across seeds."
                        )

                    require_equal(
                        alignment_fp,
                        canonical_alignment_fp,
                        f"{context}: alignment fingerprint",
                    )

                # -------------------------------------------------------------
                # Seed × fold × horizon reporting.
                # -------------------------------------------------------------

                for h_idx, horizon in enumerate(HORIZONS):

                    metrics = calculate_metrics(
                        labels[:, h_idx],
                        probabilities[:, h_idx],
                        threshold_operator_global,
                    )

                    add_metric_row(
                        seed_fold_metric_rows,
                        {
                            "model_id": model_id,
                            "config_id": config_id,
                            "seed": seed,
                            "fold": fold,
                            "horizon_days": horizon,
                            "n": len(labels),
                            "positive_rate":
                                float(
                                    labels[:, h_idx].mean()
                                ),
                            "screening_threshold":
                                SCREENING_THRESHOLD,
                            "threshold_operator":
                                threshold_operator_global,
                        },
                        metrics,
                    )

                fold_probabilities.append(
                    probabilities.astype(np.float32)
                )

                per_seed_pooled_prob[seed].append(
                    probabilities.astype(np.float64)
                )

                per_seed_pooled_label[seed].append(
                    labels.astype(np.int8)
                )

            # -----------------------------------------------------------------
            # Cross-architecture fold target/alignment identity.
            # -----------------------------------------------------------------

            if fold not in global_fold_reference:

                global_fold_reference[fold] = {
                    "row_index":
                        canonical_row_index.copy(),
                    "labels":
                        canonical_labels.copy(),
                    "validation_y_sha256":
                        canonical_y_sha,
                    "alignment_fingerprint":
                        canonical_alignment_fp,
                }

            else:

                ref = global_fold_reference[fold]

                if not np.array_equal(
                    canonical_row_index,
                    ref["row_index"],
                ):
                    fail(
                        f"{model_id}/fold{fold}: "
                        "cross-model row_index mismatch."
                    )

                if not np.array_equal(
                    canonical_labels,
                    ref["labels"],
                ):
                    fail(
                        f"{model_id}/fold{fold}: "
                        "cross-model target mismatch."
                    )

                require_equal(
                    canonical_y_sha,
                    ref["validation_y_sha256"],
                    f"{model_id}/fold{fold}: "
                    "cross-model validation_y SHA256",
                )

                require_equal(
                    canonical_alignment_fp,
                    ref["alignment_fingerprint"],
                    f"{model_id}/fold{fold}: "
                    "cross-model alignment fingerprint",
                )

            # -----------------------------------------------------------------
            # Five-seed arithmetic probability mean.
            # -----------------------------------------------------------------

            seed_stack = np.stack(
                fold_probabilities,
                axis=0,
            )

            require_equal(
                seed_stack.shape,
                (
                    len(SEEDS),
                    EXPECTED_FOLD_ROWS[fold],
                    3,
                ),
                f"{model_id}/fold{fold}: seed stack shape",
            )

            mean_probability = np.mean(
                seed_stack.astype(np.float64),
                axis=0,
                dtype=np.float64,
            )

            # Explicit arithmetic check.
            direct_mean = (
                seed_stack[0].astype(np.float64)
                + seed_stack[1].astype(np.float64)
                + seed_stack[2].astype(np.float64)
                + seed_stack[3].astype(np.float64)
                + seed_stack[4].astype(np.float64)
            ) / 5.0

            if not np.array_equal(
                mean_probability,
                direct_mean,
            ):
                if not np.allclose(
                    mean_probability,
                    direct_mean,
                    rtol=0.0,
                    atol=1e-15,
                ):
                    fail(
                        f"{model_id}/fold{fold}: "
                        "five-seed arithmetic mean mismatch."
                    )

            # -----------------------------------------------------------------
            # Save fold five-seed endpoint.
            # -----------------------------------------------------------------

            fold_dir = model_dir / f"fold_{fold:02d}"
            fold_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            fold_out = (
                fold_dir
                / "five_seed_validation_predictions.npz"
            )

            np.savez_compressed(
                fold_out,

                seeds=np.asarray(
                    SEEDS,
                    dtype=np.int16,
                ),

                horizons=np.asarray(
                    HORIZONS,
                    dtype=np.int16,
                ),

                row_index=canonical_row_index.astype(
                    np.int64
                ),

                labels=canonical_labels.astype(
                    np.int8
                ),

                probabilities_by_seed=seed_stack.astype(
                    np.float32
                ),

                probabilities_mean=mean_probability.astype(
                    np.float64
                ),
            )

            fold_out_sha = sha256_file(fold_out)

            # -----------------------------------------------------------------
            # Five-seed fold metrics at fixed screening threshold.
            # -----------------------------------------------------------------

            for h_idx, horizon in enumerate(HORIZONS):

                metrics = calculate_metrics(
                    canonical_labels[:, h_idx],
                    mean_probability[:, h_idx],
                    threshold_operator_global,
                )

                add_metric_row(
                    five_seed_fold_metric_rows,
                    {
                        "model_id": model_id,
                        "config_id": config_id,
                        "fold": fold,
                        "horizon_days": horizon,
                        "n": len(canonical_labels),
                        "positive_rate":
                            float(
                                canonical_labels[
                                    :, h_idx
                                ].mean()
                            ),
                        "seed_count": 5,
                        "aggregation":
                            "ARITHMETIC_MEAN_PROBABILITY",
                        "screening_threshold":
                            SCREENING_THRESHOLD,
                        "threshold_operator":
                            threshold_operator_global,
                    },
                    metrics,
                )

            alignment_rows.append(
                {
                    "model_id": model_id,
                    "config_id": config_id,
                    "fold": fold,
                    "rows": len(canonical_row_index),
                    "validation_y_path":
                        str(canonical_y_path),
                    "validation_y_sha256":
                        canonical_y_sha,
                    "alignment_fingerprint_sha256":
                        canonical_alignment_fp,
                    "five_seed_fold_artifact":
                        str(fold_out.resolve()),
                    "five_seed_fold_artifact_sha256":
                        fold_out_sha,
                    "alignment_status":
                        "VERIFIED_IDENTICAL_ACROSS_5_SEEDS_AND_MODELS",
                }
            )

            artifact_rows.append(
                {
                    "model_id": model_id,
                    "config_id": config_id,
                    "scope": "FOLD",
                    "fold": fold,
                    "rows": len(canonical_row_index),
                    "artifact_path":
                        str(fold_out.resolve()),
                    "artifact_sha256":
                        fold_out_sha,
                    "alignment_fingerprint_sha256":
                        canonical_alignment_fp,
                }
            )

            # Pool later.
            pooled_mean_prob_parts.append(
                mean_probability
            )

            pooled_label_parts.append(
                canonical_labels
            )

            pooled_fold_parts.append(
                np.full(
                    len(canonical_row_index),
                    fold,
                    dtype=np.int8,
                )
            )

            pooled_row_index_parts.append(
                canonical_row_index.astype(np.int64)
            )

            print(
                f"    PASS fold={fold} "
                f"rows={len(canonical_row_index)} "
                f"seeds=5"
            )

        # =====================================================================
        # Model-level pooled development OOF endpoint.
        # =====================================================================

        pooled_probability = np.concatenate(
            pooled_mean_prob_parts,
            axis=0,
        )

        pooled_labels = np.concatenate(
            pooled_label_parts,
            axis=0,
        )

        pooled_fold = np.concatenate(
            pooled_fold_parts,
            axis=0,
        )

        pooled_row_index = np.concatenate(
            pooled_row_index_parts,
            axis=0,
        )

        require_equal(
            pooled_probability.shape,
            (EXPECTED_POOLED_ROWS, 3),
            f"{model_id}: pooled probability shape",
        )

        require_equal(
            pooled_labels.shape,
            (EXPECTED_POOLED_ROWS, 3),
            f"{model_id}: pooled label shape",
        )

        pooled_out = (
            model_dir
            / "pooled_development_oof_predictions.npz"
        )

        np.savez_compressed(
            pooled_out,

            seeds=np.asarray(
                SEEDS,
                dtype=np.int16,
            ),

            horizons=np.asarray(
                HORIZONS,
                dtype=np.int16,
            ),

            fold=pooled_fold.astype(
                np.int8
            ),

            row_index=pooled_row_index.astype(
                np.int64
            ),

            labels=pooled_labels.astype(
                np.int8
            ),

            probabilities_mean=pooled_probability.astype(
                np.float64
            ),
        )

        pooled_out_sha = sha256_file(
            pooled_out
        )

        pooled_alignment_fp = sha256_arrays(
            pooled_fold.astype(np.int8),
            pooled_row_index.astype(np.int64),
            pooled_labels.astype(np.int8),
        )

        artifact_rows.append(
            {
                "model_id": model_id,
                "config_id": config_id,
                "scope": "POOLED_OOF",
                "fold": 0,
                "rows": EXPECTED_POOLED_ROWS,
                "artifact_path":
                    str(pooled_out.resolve()),
                "artifact_sha256":
                    pooled_out_sha,
                "alignment_fingerprint_sha256":
                    pooled_alignment_fp,
            }
        )

        # Five-seed pooled metrics.
        for h_idx, horizon in enumerate(HORIZONS):

            metrics = calculate_metrics(
                pooled_labels[:, h_idx],
                pooled_probability[:, h_idx],
                threshold_operator_global,
            )

            add_metric_row(
                five_seed_pooled_metric_rows,
                {
                    "model_id": model_id,
                    "config_id": config_id,
                    "horizon_days": horizon,
                    "n": EXPECTED_POOLED_ROWS,
                    "positive_rate":
                        float(
                            pooled_labels[
                                :, h_idx
                            ].mean()
                        ),
                    "fold_count": 5,
                    "seed_count": 5,
                    "aggregation":
                        "ARITHMETIC_MEAN_PROBABILITY",
                    "screening_threshold":
                        SCREENING_THRESHOLD,
                    "threshold_operator":
                        threshold_operator_global,
                },
                metrics,
            )

        # ---------------------------------------------------------------------
        # Seed-pooled OOF metrics:
        # each fixed seed evaluated across all five rolling-origin folds.
        # ---------------------------------------------------------------------

        for seed in SEEDS:

            p_seed = np.concatenate(
                per_seed_pooled_prob[seed],
                axis=0,
            )

            y_seed = np.concatenate(
                per_seed_pooled_label[seed],
                axis=0,
            )

            require_equal(
                p_seed.shape,
                (EXPECTED_POOLED_ROWS, 3),
                f"{model_id}/seed{seed}: pooled shape",
            )

            if not np.array_equal(
                y_seed,
                pooled_labels,
            ):
                fail(
                    f"{model_id}/seed{seed}: "
                    "pooled labels differ."
                )

            for h_idx, horizon in enumerate(HORIZONS):

                metrics = calculate_metrics(
                    y_seed[:, h_idx],
                    p_seed[:, h_idx],
                    threshold_operator_global,
                )

                add_metric_row(
                    seed_pooled_metric_rows,
                    {
                        "model_id": model_id,
                        "config_id": config_id,
                        "seed": seed,
                        "horizon_days": horizon,
                        "n": EXPECTED_POOLED_ROWS,
                        "fold_count": 5,
                        "screening_threshold":
                            SCREENING_THRESHOLD,
                        "threshold_operator":
                            threshold_operator_global,
                    },
                    metrics,
                )

        print(
            f"    POOLED PASS rows={EXPECTED_POOLED_ROWS}"
        )
        print("-" * 140)

    # -------------------------------------------------------------------------
    # 6. Cross-architecture pooled label identity.
    # -------------------------------------------------------------------------

    # At this point fold-wise equality has already been enforced.
    require_equal(
        len(global_fold_reference),
        5,
        "Global fold-reference count",
    )

    print("[4] FIVE-SEED ALIGNMENT")
    print("Architectures                      : 8/8")
    print("Folds/architecture                 : 5/5")
    print("Seeds/fold                         : 5/5")
    print("row_index identical across seeds   : YES")
    print("Targets identical across seeds     : YES")
    print("Targets identical across models    : YES")
    print("Probability = sigmoid(logit)        : VERIFIED")
    print("Arithmetic mean probabilities       : VERIFIED")
    print("Final test                          : NOT OPENED")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 7. Materialize metric tables.
    # -------------------------------------------------------------------------

    seed_fold_df = pd.DataFrame(
        seed_fold_metric_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "seed",
            "fold",
            "horizon_days",
        ]
    ).reset_index(drop=True)

    seed_pooled_df = pd.DataFrame(
        seed_pooled_metric_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "seed",
            "horizon_days",
        ]
    ).reset_index(drop=True)

    five_seed_fold_df = pd.DataFrame(
        five_seed_fold_metric_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "fold",
            "horizon_days",
        ]
    ).reset_index(drop=True)

    five_seed_pooled_df = pd.DataFrame(
        five_seed_pooled_metric_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "horizon_days",
        ]
    ).reset_index(drop=True)

    require_equal(
        len(seed_fold_df),
        8 * 5 * 5 * 3,
        "Seed-fold metric rows",
    )

    require_equal(
        len(seed_pooled_df),
        8 * 5 * 3,
        "Seed-pooled metric rows",
    )

    require_equal(
        len(five_seed_fold_df),
        8 * 5 * 3,
        "Five-seed fold metric rows",
    )

    require_equal(
        len(five_seed_pooled_df),
        8 * 3,
        "Five-seed pooled metric rows",
    )

    # -------------------------------------------------------------------------
    # Mean/std across the FIVE FIXED seeds.
    #
    # Unit being summarized:
    # pooled development OOF metric for each seed across all 5 folds.
    # -------------------------------------------------------------------------

    summary_rows = []

    grouped = seed_pooled_df.groupby(
        [
            "model_id",
            "config_id",
            "horizon_days",
        ],
        sort=True,
    )

    for (
        model_id,
        config_id,
        horizon,
    ), g in grouped:

        require_equal(
            len(g),
            5,
            (
                f"{model_id}/{config_id}/"
                f"{horizon}d seed summary count"
            ),
        )

        row = {
            "model_id": model_id,
            "config_id": config_id,
            "horizon_days": int(horizon),
            "seed_count": 5,
            "folds_per_seed": 5,
            "summary_unit":
                "POOLED_DEVELOPMENT_OOF_METRIC_PER_FIXED_SEED",
        }

        for metric in METRIC_NAMES:

            vals = g[metric].astype(float).to_numpy()

            row[f"{metric}_mean"] = float(
                np.mean(vals)
            )

            row[f"{metric}_std"] = float(
                np.std(vals, ddof=1)
            )

            row[f"{metric}_min"] = float(
                np.min(vals)
            )

            row[f"{metric}_max"] = float(
                np.max(vals)
            )

        summary_rows.append(row)

    seed_summary_df = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "horizon_days",
        ]
    ).reset_index(drop=True)

    require_equal(
        len(seed_summary_df),
        8 * 3,
        "Seed robustness summary rows",
    )

    alignment_df = pd.DataFrame(
        alignment_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "fold",
        ]
    ).reset_index(drop=True)

    artifact_df = pd.DataFrame(
        artifact_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "scope",
            "fold",
        ]
    ).reset_index(drop=True)

    require_equal(
        len(alignment_df),
        8 * 5,
        "Alignment audit rows",
    )

    require_equal(
        len(artifact_df),
        (8 * 5) + 8,
        "OOF artifact inventory rows",
    )

    # -------------------------------------------------------------------------
    # 8. Save audits.
    # -------------------------------------------------------------------------

    seed_fold_df.to_csv(
        SEED_FOLD_METRICS_CSV,
        index=False,
    )

    seed_pooled_df.to_csv(
        SEED_POOLED_METRICS_CSV,
        index=False,
    )

    seed_summary_df.to_csv(
        SEED_ROBUSTNESS_SUMMARY_CSV,
        index=False,
    )

    five_seed_fold_df.to_csv(
        FIVE_SEED_FOLD_METRICS_CSV,
        index=False,
    )

    five_seed_pooled_df.to_csv(
        FIVE_SEED_POOLED_METRICS_CSV,
        index=False,
    )

    artifact_df.to_csv(
        ARTIFACT_INVENTORY_CSV,
        index=False,
    )

    alignment_df.to_csv(
        ALIGNMENT_AUDIT_CSV,
        index=False,
    )

    output_files = {
        "seed_fold_horizon_metrics":
            SEED_FOLD_METRICS_CSV,

        "seed_pooled_oof_metrics":
            SEED_POOLED_METRICS_CSV,

        "seed_robustness_mean_std":
            SEED_ROBUSTNESS_SUMMARY_CSV,

        "five_seed_fold_metrics":
            FIVE_SEED_FOLD_METRICS_CSV,

        "five_seed_pooled_oof_metrics":
            FIVE_SEED_POOLED_METRICS_CSV,

        "oof_artifact_inventory":
            ARTIFACT_INVENTORY_CSV,

        "fold_alignment_audit":
            ALIGNMENT_AUDIT_CSV,
    }

    output_hashes = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in output_files.items()
    }

    # Include all 48 NPZ artifacts in the freeze fingerprint.
    npz_hash_records = (
        artifact_df[
            [
                "model_id",
                "config_id",
                "scope",
                "fold",
                "rows",
                "artifact_path",
                "artifact_sha256",
                "alignment_fingerprint_sha256",
            ]
        ]
        .to_dict(orient="records")
    )

    oof_content_fingerprint = canonical_json_sha256(
        npz_hash_records
    )

    print("[5] DEVELOPMENT OOF OUTPUTS")
    print(
        f"Seed×fold×horizon metric rows : "
        f"{len(seed_fold_df)}"
    )
    print(
        f"Seed-pooled metric rows        : "
        f"{len(seed_pooled_df)}"
    )
    print(
        f"Seed mean/std summary rows     : "
        f"{len(seed_summary_df)}"
    )
    print(
        f"Five-seed fold metric rows     : "
        f"{len(five_seed_fold_df)}"
    )
    print(
        f"Five-seed pooled metric rows   : "
        f"{len(five_seed_pooled_df)}"
    )
    print(
        f"Frozen OOF NPZ artifacts       : "
        f"{len(artifact_df)}"
    )
    print(
        f"OOF content fingerprint        : "
        f"{oof_content_fingerprint}"
    )
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 9. Compact five-seed pooled diagnostic report.
    #
    # IMPORTANT:
    # These are development OOF diagnostics at the fixed screening threshold
    # 0.5. They are NOT final-test results and do not select an architecture.
    # -------------------------------------------------------------------------

    print("[6] FIVE-SEED POOLED DEVELOPMENT OOF DIAGNOSTICS")
    print(
        "(Fixed threshold 0.5 for BA/MCC reporting; "
        "NO threshold optimization)"
    )
    print()

    for r in five_seed_pooled_df.itertuples(index=False):

        print(
            f"{r.model_id:14s} "
            f"{int(r.horizon_days)}d "
            f"BA={r.balanced_accuracy:.6f} "
            f"AUC={r.roc_auc:.6f} "
            f"MCC={r.mcc:.6f} "
            f"PR-AUC={r.pr_auc:.6f} "
            f"Brier={r.brier:.6f}"
        )

    print("-" * 140)

    # -------------------------------------------------------------------------
    # 10. Build final Step-31 freeze fingerprint.
    # -------------------------------------------------------------------------

    freeze_basis = {
        "step30_manifest_sha256":
            step30_sha,

        "step30_global_freeze_fingerprint":
            EXPECTED_STEP30_GLOBAL_FINGERPRINT,

        "step30_logical_inventory_sha256":
            inventory_sha,

        "step31_source_sha256":
            source_sha,

        "step31_frozen_source_sha256":
            frozen_source_sha,

        "architectures":
            SELECTED,

        "seeds":
            SEEDS,

        "folds":
            FOLDS,

        "horizons":
            HORIZONS,

        "aggregation":
            "ARITHMETIC_MEAN_OF_PROBABILITIES",

        "seed_selection":
            False,

        "screening_threshold":
            SCREENING_THRESHOLD,

        "threshold_optimization":
            False,

        "final_test_data_opened":
            False,

        "oof_content_fingerprint":
            oof_content_fingerprint,

        "audit_output_hashes":
            output_hashes,
    }

    global_step31_fingerprint = canonical_json_sha256(
        freeze_basis
    )

    # -------------------------------------------------------------------------
    # 11. Write final manifest LAST.
    # -------------------------------------------------------------------------

    final_manifest = {
        "status":
            "STEP31_FIVE_SEED_DEVELOPMENT_OOF_FINAL_FREEZE_COMPLETE",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "experiment_root":
            str(ROOT),

        "upstream_stage2_freeze": {
            "manifest_path":
                str(STEP30_MANIFEST),

            "manifest_sha256":
                step30_sha,

            "global_freeze_fingerprint":
                EXPECTED_STEP30_GLOBAL_FINGERPRINT,

            "logical_inventory_path":
                str(inventory_path),

            "logical_inventory_sha256":
                inventory_sha,
        },

        "endpoint_contract": {
            "architectures":
                SELECTED,

            "seed_set":
                SEEDS,

            "fold_set":
                FOLDS,

            "horizon_order":
                HORIZONS,

            "probability_endpoint":
                "ARITHMETIC_MEAN_OF_FIXED_FIVE_SEED_PROBABILITIES",

            "logit_averaging":
                False,

            "best_seed_selection":
                False,

            "best_seed_promotion":
                False,

            "screening_threshold_for_diagnostic_metrics":
                SCREENING_THRESHOLD,

            "threshold_operator":
                threshold_operator_global,

            "threshold_optimization":
                False,

            "pooled_development_rows_per_architecture":
                EXPECTED_POOLED_ROWS,
        },

        "metric_definitions": {
            "balanced_accuracy":
                "sklearn.metrics.balanced_accuracy_score "
                "at frozen screening threshold 0.5",

            "roc_auc":
                "sklearn.metrics.roc_auc_score",

            "mcc":
                "sklearn.metrics.matthews_corrcoef "
                "at frozen screening threshold 0.5",

            "pr_auc":
                "sklearn.metrics.average_precision_score",

            "brier":
                "mean((probability-label)^2)",

            "bce":
                "mean binary cross-entropy",
        },

        "reconciliation": {
            "logical_prediction_files_verified":
                200,

            "architectures":
                8,

            "folds_per_architecture":
                5,

            "seeds_per_fold":
                5,

            "five_seed_fold_artifacts":
                40,

            "pooled_oof_artifacts":
                8,

            "total_oof_npz_artifacts":
                48,

            "row_index_identity_across_seeds":
                True,

            "target_identity_across_seeds":
                True,

            "target_identity_across_architectures":
                True,

            "sigmoid_logit_probability_consistency":
                True,

            "five_seed_arithmetic_probability_mean_verified":
                True,

            "invalid_or_missing":
                0,
        },

        "protection_state": {
            "best_seed_selection_performed":
                False,

            "best_seed_promotion_performed":
                False,

            "threshold_optimization_performed":
                False,

            "final_test_data_opened":
                False,

            "final_test_metric_computed":
                False,

            "final_test_state":
                "LOCKED_NOT_OPENED",
        },

        "outputs": output_hashes,

        "oof_artifacts": {
            "root":
                str(OOF_ROOT.resolve()),

            "artifact_count":
                48,

            "content_fingerprint":
                oof_content_fingerprint,
        },

        "source_freeze": {
            "executable_source_sha256":
                source_sha,

            "frozen_source_path":
                str(FROZEN_SOURCE),

            "frozen_source_sha256":
                frozen_source_sha,
        },

        "global_step31_fingerprint":
            global_step31_fingerprint,

        "next_protocol_gate": {
            "allowed_next_step":
                "DEVELOPMENT_ONLY_THRESHOLD_OPTIMIZATION",

            "threshold_grid":
                {
                    "start": 0.30,
                    "stop": 0.70,
                    "step": 0.01,
                },

            "optimization_target":
                "POOLED_DEVELOPMENT_OOF_BALANCED_ACCURACY",

            "tie_breaking":
                [
                    "closest_to_0.50",
                    "smaller_threshold",
                ],

            "separate_threshold_per_architecture_and_horizon":
                True,

            "final_test_allowed_now":
                False,
        },
    }

    with FINAL_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_manifest,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    final_manifest_sha = sha256_file(
        FINAL_MANIFEST
    )

    # -------------------------------------------------------------------------
    # 12. Final report.
    # -------------------------------------------------------------------------

    print("[7] STEP-31 FINAL FREEZE")
    print(
        "Global Step-31 fingerprint : "
        f"{global_step31_fingerprint}"
    )
    print(
        "Freeze manifest SHA256     : "
        f"{final_manifest_sha}"
    )
    print()

    print("=" * 140)
    print(
        "STEP 31 COMPLETE — FIVE-SEED DEVELOPMENT "
        "OOF PROBABILITY ENDPOINT FROZEN"
    )
    print("=" * 140)

    print("Architectures aggregated          : 8/8")
    print("Rolling-origin folds              : 5/5")
    print("Fixed seeds/fold                  : 5/5")
    print("Logical prediction fits verified  : 200/200")
    print("Five-seed fold artifacts          : 40/40")
    print("Pooled development OOF artifacts  : 8/8")
    print(
        f"Development OOF rows/model        : "
        f"{EXPECTED_POOLED_ROWS}"
    )
    print("Probability aggregation           : ARITHMETIC MEAN")
    print("Logit averaging                   : NOT PERFORMED")
    print("Best-seed selection               : NOT PERFORMED")
    print("Best-seed promotion               : NOT PERFORMED")
    print("Threshold optimization            : NOT PERFORMED")
    print("Final-test data opened            : False")
    print("Final-test state                  : LOCKED_NOT_OPENED")

    print("-" * 140)

    print(
        "NEXT GATE: development-only threshold optimization "
        "on the frozen five-seed pooled OOF probabilities."
    )

    print(
        "Threshold grid remains 0.30–0.70 in 0.01 increments; "
        "do not open the final test."
    )

    print("=" * 140)


if __name__ == "__main__":
    main()