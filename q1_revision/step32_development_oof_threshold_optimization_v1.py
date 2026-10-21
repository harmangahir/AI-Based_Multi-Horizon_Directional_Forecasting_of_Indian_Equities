#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 32 — DEVELOPMENT-ONLY OOF THRESHOLD OPTIMIZATION
======================================================

Purpose
-------
Freeze one decision threshold for each:

    selected architecture × horizon

using ONLY the frozen five-seed pooled development OOF probabilities
created by authoritative Step-31A.

Threshold grid
--------------
    0.30, 0.31, ..., 0.70

Optimization target
-------------------
    pooled development OOF Balanced Accuracy

Tie-breaking
------------
    1. maximum Balanced Accuracy
    2. threshold closest to 0.50
    3. smaller threshold

Important
---------
- Five fixed seeds have already been averaged at probability level.
- No best-seed selection is performed.
- No architecture is selected/promoted.
- Thresholds are optimized separately for every architecture × horizon.
- Final-test data are NOT opened.
- Final-test metrics are NOT computed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef


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

STEP31A_MANIFEST = (
    MANIFEST_DIR
    / "31a_stage2_five_seed_oof_final_freeze_manifest.json"
)

THRESHOLD_LANDSCAPE_CSV = (
    AUDIT_DIR
    / "32_development_threshold_landscape.csv"
)

SELECTED_THRESHOLDS_CSV = (
    AUDIT_DIR
    / "32_selected_development_thresholds.csv"
)

FINAL_MANIFEST = (
    MANIFEST_DIR
    / "32_development_threshold_final_freeze_manifest.json"
)

FROZEN_SOURCE = (
    MANIFEST_DIR
    / "step32_development_oof_threshold_optimization_v1_frozen.py"
)


# =============================================================================
# AUTHORITATIVE UPSTREAM FREEZE
# =============================================================================

EXPECTED_STEP31A_MANIFEST_SHA256 = (
    "a6e8b2fb056399bf28f4b8a28e5ecdf7176d808e0e4950968a437a29ac5c6ee5"
)

EXPECTED_STEP31_GLOBAL_FINGERPRINT = (
    "1fc383fd4580819dc98b2212e1731fd651c4566d9a657fc06abcbca3406cbcd0"
)

EXPECTED_OOF_CONTENT_FINGERPRINT = (
    "b374658dbf32e1ac4615267b9694ce4b442b346218ce1fe6c3082be4260ac727"
)


# =============================================================================
# FROZEN THRESHOLD CONTRACT
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

EXPECTED_ROWS_PER_MODEL = 24240

THRESHOLD_INTEGERS = list(range(30, 71))
REFERENCE_THRESHOLD_INTEGER = 50

EXPECTED_ARCHITECTURE_HORIZON_ROWS = 8 * 3       # 24
EXPECTED_LANDSCAPE_ROWS = 8 * 3 * 41            # 984


# =============================================================================
# HELPERS
# =============================================================================

def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

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

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        obj = json.load(f)

    if not isinstance(obj, dict):
        fail(f"Expected JSON object: {path}")

    return obj


def require_equal(
    actual,
    expected,
    label: str,
) -> None:

    if actual != expected:

        fail(
            f"{label}: expected {expected!r}, "
            f"got {actual!r}"
        )


def require_false(
    value,
    label: str,
) -> None:

    if value is not False:

        fail(
            f"{label}: expected False, "
            f"got {value!r}"
        )


def freeze_source() -> tuple[str, str]:

    src = Path(__file__).resolve()

    current_sha = sha256_file(src)

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if FROZEN_SOURCE.exists():

        frozen_sha = sha256_file(
            FROZEN_SOURCE
        )

        if frozen_sha != current_sha:

            fail(
                "Step-32 frozen source already exists but "
                "differs from executable source.\n"
                f"Frozen : {frozen_sha}\n"
                f"Current: {current_sha}"
            )

    else:

        shutil.copy2(
            src,
            FROZEN_SOURCE,
        )

    frozen_sha = sha256_file(
        FROZEN_SOURCE
    )

    require_equal(
        frozen_sha,
        current_sha,
        "Step32 frozen-source SHA256",
    )

    return current_sha, frozen_sha


def threshold_confusion(
    y: np.ndarray,
    p: np.ndarray,
    threshold_integer: int,
) -> dict:

    threshold = threshold_integer / 100.0

    pred = (
        p >= threshold
    ).astype(np.int8)

    y = y.astype(np.int8)

    tp = int(
        np.sum(
            (y == 1)
            & (pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y == 0)
            & (pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y == 0)
            & (pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y == 1)
            & (pred == 0)
        )
    )

    positive_n = tp + fn
    negative_n = tn + fp

    if positive_n <= 0 or negative_n <= 0:
        fail(
            "Both classes are required for "
            "Balanced Accuracy."
        )

    # -------------------------------------------------------------
    # Exact tie-comparison score.
    #
    # BA = 1/2 * (TP/P + TN/N)
    #
    # For a fixed target vector P and N are constant, so:
    #
    # score_num = TP*N + TN*P
    #
    # Maximizing score_num is EXACTLY equivalent to maximizing BA.
    # This avoids floating-point ambiguity in threshold ties.
    # -------------------------------------------------------------

    ba_score_numerator = (
        tp * negative_n
        + tn * positive_n
    )

    ba_score_denominator = (
        2
        * positive_n
        * negative_n
    )

    balanced_accuracy = (
        ba_score_numerator
        / ba_score_denominator
    )

    sensitivity = (
        tp / positive_n
    )

    specificity = (
        tn / negative_n
    )

    predicted_positive_rate = float(
        pred.mean()
    )

    mcc = float(
        matthews_corrcoef(
            y,
            pred,
        )
    )

    return {
        "threshold":
            threshold,

        "threshold_integer":
            threshold_integer,

        "tp":
            tp,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "positive_n":
            positive_n,

        "negative_n":
            negative_n,

        "balanced_accuracy_score_numerator":
            int(ba_score_numerator),

        "balanced_accuracy_score_denominator":
            int(ba_score_denominator),

        "balanced_accuracy":
            float(balanced_accuracy),

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "mcc":
            mcc,

        "predicted_positive_rate":
            predicted_positive_rate,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 140)
    print(
        "STEP 32 — DEVELOPMENT-ONLY OOF "
        "THRESHOLD OPTIMIZATION"
    )
    print("=" * 140)

    print(
        f"Experiment root : {ROOT}"
    )

    print(
        "Input endpoint  : frozen five-seed "
        "pooled development OOF probabilities"
    )

    print(
        "Threshold grid  : 0.30–0.70 step 0.01"
    )

    print(
        "Objective       : pooled development "
        "OOF Balanced Accuracy"
    )

    print(
        "Tie-breaking    : closest to 0.50, "
        "then smaller threshold"
    )

    print(
        "Architecture selection : NONE"
    )

    print(
        "Best-seed selection    : NONE"
    )

    print(
        "Final test             : LOCKED_NOT_OPENED"
    )

    print("-" * 140)

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 0. Preserve established freeze.
    # -------------------------------------------------------------------------

    if FINAL_MANIFEST.exists():

        print(
            "Step-32 final freeze already exists:"
        )

        print(
            FINAL_MANIFEST
        )

        print()

        print(
            "No Step-32 artifacts were modified."
        )

        sys.exit(0)

    # -------------------------------------------------------------------------
    # 1. Verify authoritative Step-31A freeze.
    # -------------------------------------------------------------------------

    step31a_sha = sha256_file(
        STEP31A_MANIFEST
    )

    require_equal(
        step31a_sha,
        EXPECTED_STEP31A_MANIFEST_SHA256,
        "Step31A final-manifest SHA256",
    )

    step31 = load_json(
        STEP31A_MANIFEST
    )

    require_equal(
        step31.get("status"),
        "STEP31_FIVE_SEED_DEVELOPMENT_OOF_FINAL_FREEZE_COMPLETE",
        "Step31A status",
    )

    require_equal(
        step31.get("global_step31_fingerprint"),
        EXPECTED_STEP31_GLOBAL_FINGERPRINT,
        "Step31 global fingerprint",
    )

    require_equal(
        step31[
            "oof_artifacts"
        ][
            "content_fingerprint"
        ],
        EXPECTED_OOF_CONTENT_FINGERPRINT,
        "Step31 OOF content fingerprint",
    )

    protection = step31.get(
        "protection_state",
        {},
    )

    require_false(
        protection.get(
            "best_seed_selection_performed"
        ),
        "Step31 best_seed_selection_performed",
    )

    require_false(
        protection.get(
            "best_seed_promotion_performed"
        ),
        "Step31 best_seed_promotion_performed",
    )

    require_false(
        protection.get(
            "threshold_optimization_performed"
        ),
        "Step31 threshold_optimization_performed",
    )

    require_false(
        protection.get(
            "final_test_data_opened"
        ),
        "Step31 final_test_data_opened",
    )

    require_false(
        protection.get(
            "final_test_metric_computed"
        ),
        "Step31 final_test_metric_computed",
    )

    require_equal(
        protection.get(
            "final_test_state"
        ),
        "LOCKED_NOT_OPENED",
        "Step31 final-test state",
    )

    endpoint = step31[
        "endpoint_contract"
    ]

    require_equal(
        endpoint.get(
            "probability_endpoint"
        ),
        "ARITHMETIC_MEAN_OF_FIXED_FIVE_SEED_PROBABILITIES",
        "Step31 probability endpoint",
    )

    require_equal(
        endpoint.get(
            "seed_set"
        ),
        SEEDS,
        "Step31 seed set",
    )

    require_equal(
        endpoint.get(
            "fold_set"
        ),
        FOLDS,
        "Step31 fold set",
    )

    require_equal(
        endpoint.get(
            "horizon_order"
        ),
        HORIZONS,
        "Step31 horizon order",
    )

    require_equal(
        endpoint.get(
            "threshold_operator"
        ),
        ">=",
        "Step31 threshold operator",
    )

    require_equal(
        endpoint.get(
            "pooled_development_rows_per_architecture"
        ),
        EXPECTED_ROWS_PER_MODEL,
        "Step31 pooled rows/model",
    )

    print("[1] STEP-31A FIVE-SEED OOF FREEZE")
    print(
        f"Manifest SHA256      : {step31a_sha}"
    )
    print(
        "Global fingerprint   : "
        f"{EXPECTED_STEP31_GLOBAL_FINGERPRINT}"
    )
    print(
        "OOF fingerprint      : "
        f"{EXPECTED_OOF_CONTENT_FINGERPRINT}"
    )
    print(
        "Final test           : LOCKED_NOT_OPENED"
    )
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 2. Verify OOF artifact inventory.
    # -------------------------------------------------------------------------

    inventory_meta = step31[
        "outputs"
    ][
        "oof_artifact_inventory"
    ]

    inventory_path = Path(
        inventory_meta["path"]
    ).expanduser().resolve()

    require_equal(
        sha256_file(inventory_path),
        inventory_meta["sha256"],
        "Step31 OOF artifact inventory SHA256",
    )

    inventory = pd.read_csv(
        inventory_path
    )

    require_equal(
        len(inventory),
        48,
        "Step31 OOF artifact inventory rows",
    )

    pooled_inventory = inventory.loc[
        inventory["scope"].astype(str)
        == "POOLED_OOF"
    ].copy()

    require_equal(
        len(pooled_inventory),
        8,
        "Pooled OOF artifact count",
    )

    require_equal(
        set(
            pooled_inventory[
                "model_id"
            ].astype(str)
        ),
        set(SELECTED),
        "Pooled OOF architecture set",
    )

    print("[2] FROZEN OOF ARTIFACT INVENTORY")
    print(
        f"All OOF artifacts     : {len(inventory)}/48"
    )
    print(
        f"Pooled OOF artifacts  : {len(pooled_inventory)}/8"
    )
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 3. Freeze Step-32 source.
    # -------------------------------------------------------------------------

    source_sha, frozen_source_sha = (
        freeze_source()
    )

    print("[3] STEP-32 SOURCE FREEZE")
    print(
        f"Executable SHA256 : {source_sha}"
    )
    print(
        f"Frozen SHA256     : {frozen_source_sha}"
    )
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 4. Remove only unfinished Step-32 audit outputs.
    # -------------------------------------------------------------------------

    for p in [
        THRESHOLD_LANDSCAPE_CSV,
        SELECTED_THRESHOLDS_CSV,
    ]:

        if p.exists():
            p.unlink()

    landscape_rows = []
    selected_rows = []

    global_reference = None

    # -------------------------------------------------------------------------
    # 5. Threshold optimization.
    # -------------------------------------------------------------------------

    for model_id, config_id in SELECTED.items():

        x = pooled_inventory.loc[
            (
                pooled_inventory[
                    "model_id"
                ].astype(str)
                == model_id
            )
            &
            (
                pooled_inventory[
                    "config_id"
                ].astype(str)
                == config_id
            )
        ]

        require_equal(
            len(x),
            1,
            (
                f"{model_id}/{config_id}: "
                "pooled artifact count"
            ),
        )

        r = x.iloc[0]

        artifact_path = Path(
            str(
                r["artifact_path"]
            )
        ).expanduser().resolve()

        require_equal(
            sha256_file(
                artifact_path
            ),
            str(
                r["artifact_sha256"]
            ),
            (
                f"{model_id}/{config_id}: "
                "pooled artifact SHA256"
            ),
        )

        with np.load(
            artifact_path,
            allow_pickle=False,
        ) as z:

            required_keys = {
                "seeds",
                "horizons",
                "fold",
                "row_index",
                "labels",
                "probabilities_mean",
            }

            require_equal(
                set(z.files),
                required_keys,
                (
                    f"{model_id}/{config_id}: "
                    "pooled NPZ keys"
                ),
            )

            seeds = np.asarray(
                z["seeds"],
                dtype=np.int64,
            )

            horizons = np.asarray(
                z["horizons"],
                dtype=np.int64,
            )

            fold = np.asarray(
                z["fold"],
                dtype=np.int64,
            )

            row_index = np.asarray(
                z["row_index"],
                dtype=np.int64,
            )

            labels = np.asarray(
                z["labels"],
                dtype=np.int8,
            )

            probabilities = np.asarray(
                z["probabilities_mean"],
                dtype=np.float64,
            )

        require_equal(
            seeds.tolist(),
            SEEDS,
            f"{model_id}: seed vector",
        )

        require_equal(
            horizons.tolist(),
            HORIZONS,
            f"{model_id}: horizon vector",
        )

        require_equal(
            labels.shape,
            (EXPECTED_ROWS_PER_MODEL, 3),
            f"{model_id}: label shape",
        )

        require_equal(
            probabilities.shape,
            (EXPECTED_ROWS_PER_MODEL, 3),
            f"{model_id}: probability shape",
        )

        require_equal(
            fold.shape,
            (EXPECTED_ROWS_PER_MODEL,),
            f"{model_id}: fold shape",
        )

        require_equal(
            row_index.shape,
            (EXPECTED_ROWS_PER_MODEL,),
            f"{model_id}: row-index shape",
        )

        require_equal(
            set(
                np.unique(fold).tolist()
            ),
            set(FOLDS),
            f"{model_id}: pooled fold set",
        )

        if not np.all(
            np.isfinite(
                probabilities
            )
        ):
            fail(
                f"{model_id}: non-finite "
                "pooled probabilities."
            )

        if (
            np.any(
                probabilities < 0.0
            )
            or
            np.any(
                probabilities > 1.0
            )
        ):
            fail(
                f"{model_id}: probability "
                "outside [0,1]."
            )

        if not set(
            np.unique(
                labels
            ).tolist()
        ).issubset({0, 1}):

            fail(
                f"{model_id}: non-binary labels."
            )

        # -------------------------------------------------------------
        # Re-confirm that every architecture references the same pooled
        # OOF observations and targets.
        # -------------------------------------------------------------

        reference = {
            "fold":
                fold.copy(),

            "row_index":
                row_index.copy(),

            "labels":
                labels.copy(),
        }

        if global_reference is None:

            global_reference = reference

        else:

            if not np.array_equal(
                fold,
                global_reference["fold"],
            ):
                fail(
                    f"{model_id}: cross-model "
                    "fold ordering mismatch."
                )

            if not np.array_equal(
                row_index,
                global_reference["row_index"],
            ):
                fail(
                    f"{model_id}: cross-model "
                    "row_index ordering mismatch."
                )

            if not np.array_equal(
                labels,
                global_reference["labels"],
            ):
                fail(
                    f"{model_id}: cross-model "
                    "label mismatch."
                )

        print(
            f"[MODEL] {model_id:14s} "
            f"{config_id:20s}"
        )

        for h_idx, horizon in enumerate(
            HORIZONS
        ):

            y = labels[
                :,
                h_idx
            ]

            p = probabilities[
                :,
                h_idx
            ]

            threshold_results = []

            for threshold_integer in (
                THRESHOLD_INTEGERS
            ):

                m = threshold_confusion(
                    y,
                    p,
                    threshold_integer,
                )

                threshold_results.append(m)

                landscape_rows.append(
                    {
                        "model_id":
                            model_id,

                        "config_id":
                            config_id,

                        "horizon_days":
                            horizon,

                        "n":
                            len(y),

                        "positive_rate":
                            float(
                                y.mean()
                            ),

                        **m,
                    }
                )

            # ---------------------------------------------------------
            # Exact BA maximization.
            # ---------------------------------------------------------

            max_score_num = max(
                x[
                    "balanced_accuracy_score_numerator"
                ]
                for x in threshold_results
            )

            ties = [
                x
                for x in threshold_results
                if (
                    x[
                        "balanced_accuracy_score_numerator"
                    ]
                    == max_score_num
                )
            ]

            # Protocol tie break:
            # 1. closest to 0.50
            # 2. smaller threshold
            chosen = min(
                ties,
                key=lambda x: (
                    abs(
                        x[
                            "threshold_integer"
                        ]
                        - REFERENCE_THRESHOLD_INTEGER
                    ),
                    x[
                        "threshold_integer"
                    ],
                ),
            )

            fixed_050 = next(
                x
                for x in threshold_results
                if (
                    x[
                        "threshold_integer"
                    ]
                    == REFERENCE_THRESHOLD_INTEGER
                )
            )

            selected_rows.append(
                {
                    "model_id":
                        model_id,

                    "config_id":
                        config_id,

                    "horizon_days":
                        horizon,

                    "n":
                        len(y),

                    "positive_rate":
                        float(
                            y.mean()
                        ),

                    "threshold_grid_start":
                        0.30,

                    "threshold_grid_stop":
                        0.70,

                    "threshold_grid_step":
                        0.01,

                    "threshold_operator":
                        ">=",

                    "optimization_target":
                        "POOLED_DEVELOPMENT_OOF_BALANCED_ACCURACY",

                    "selected_threshold":
                        chosen[
                            "threshold"
                        ],

                    "selected_threshold_integer":
                        chosen[
                            "threshold_integer"
                        ],

                    "selected_balanced_accuracy":
                        chosen[
                            "balanced_accuracy"
                        ],

                    "selected_mcc":
                        chosen[
                            "mcc"
                        ],

                    "selected_sensitivity":
                        chosen[
                            "sensitivity"
                        ],

                    "selected_specificity":
                        chosen[
                            "specificity"
                        ],

                    "selected_predicted_positive_rate":
                        chosen[
                            "predicted_positive_rate"
                        ],

                    "fixed_050_balanced_accuracy":
                        fixed_050[
                            "balanced_accuracy"
                        ],

                    "fixed_050_mcc":
                        fixed_050[
                            "mcc"
                        ],

                    "balanced_accuracy_delta_vs_050":
                        (
                            chosen[
                                "balanced_accuracy"
                            ]
                            -
                            fixed_050[
                                "balanced_accuracy"
                            ]
                        ),

                    "maximum_ba_tie_count":
                        len(ties),

                    "tie_break_rule_1":
                        "CLOSEST_TO_0.50",

                    "tie_break_rule_2":
                        "SMALLER_THRESHOLD",

                    "best_seed_selection":
                        False,

                    "architecture_selection":
                        False,

                    "final_test_used":
                        False,
                }
            )

            print(
                f"    {horizon}d "
                f"threshold={chosen['threshold']:.2f} "
                f"BA={chosen['balanced_accuracy']:.6f} "
                f"BA@0.50={fixed_050['balanced_accuracy']:.6f} "
                f"delta={chosen['balanced_accuracy'] - fixed_050['balanced_accuracy']:+.6f} "
                f"ties={len(ties)}"
            )

        print("-" * 140)

    # -------------------------------------------------------------------------
    # 6. Build audit tables.
    # -------------------------------------------------------------------------

    landscape_df = pd.DataFrame(
        landscape_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "horizon_days",
            "threshold_integer",
        ]
    ).reset_index(
        drop=True
    )

    selected_df = pd.DataFrame(
        selected_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "horizon_days",
        ]
    ).reset_index(
        drop=True
    )

    require_equal(
        len(landscape_df),
        EXPECTED_LANDSCAPE_ROWS,
        "Threshold landscape rows",
    )

    require_equal(
        len(selected_df),
        EXPECTED_ARCHITECTURE_HORIZON_ROWS,
        "Selected threshold rows",
    )

    require_equal(
        set(
            selected_df[
                "selected_threshold_integer"
            ].astype(int)
        ).issubset(
            set(
                THRESHOLD_INTEGERS
            )
        ),
        True,
        "Selected threshold grid membership",
    )

    landscape_df.to_csv(
        THRESHOLD_LANDSCAPE_CSV,
        index=False,
    )

    selected_df.to_csv(
        SELECTED_THRESHOLDS_CSV,
        index=False,
    )

    landscape_sha = sha256_file(
        THRESHOLD_LANDSCAPE_CSV
    )

    selected_sha = sha256_file(
        SELECTED_THRESHOLDS_CSV
    )

    selected_records = (
        selected_df.to_dict(
            orient="records"
        )
    )

    threshold_content_fingerprint = (
        canonical_json_sha256(
            selected_records
        )
    )

    print("[4] THRESHOLD OPTIMIZATION OUTPUTS")
    print(
        f"Threshold landscape rows : "
        f"{len(landscape_df)}/984"
    )
    print(
        f"Selected thresholds      : "
        f"{len(selected_df)}/24"
    )
    print(
        f"Landscape SHA256         : "
        f"{landscape_sha}"
    )
    print(
        f"Selected SHA256          : "
        f"{selected_sha}"
    )
    print(
        "Threshold fingerprint    : "
        f"{threshold_content_fingerprint}"
    )
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 7. Compact frozen threshold report.
    # -------------------------------------------------------------------------

    print("[5] FROZEN DEVELOPMENT THRESHOLDS")
    print()

    for r in selected_df.itertuples(
        index=False
    ):

        print(
            f"{r.model_id:14s} "
            f"{int(r.horizon_days)}d "
            f"t*={r.selected_threshold:.2f} "
            f"BA={r.selected_balanced_accuracy:.6f} "
            f"BA@0.50={r.fixed_050_balanced_accuracy:.6f} "
            f"Δ={r.balanced_accuracy_delta_vs_050:+.6f}"
        )

    print("-" * 140)

    # -------------------------------------------------------------------------
    # 8. Final Step-32 freeze fingerprint.
    # -------------------------------------------------------------------------

    freeze_basis = {
        "step31a_manifest_sha256":
            step31a_sha,

        "step31_global_fingerprint":
            EXPECTED_STEP31_GLOBAL_FINGERPRINT,

        "step31_oof_content_fingerprint":
            EXPECTED_OOF_CONTENT_FINGERPRINT,

        "step32_source_sha256":
            source_sha,

        "step32_frozen_source_sha256":
            frozen_source_sha,

        "threshold_grid_integer":
            THRESHOLD_INTEGERS,

        "threshold_operator":
            ">=",

        "optimization_target":
            "POOLED_DEVELOPMENT_OOF_BALANCED_ACCURACY",

        "tie_breaking": [
            "CLOSEST_TO_0.50",
            "SMALLER_THRESHOLD",
        ],

        "separate_threshold_per_architecture_and_horizon":
            True,

        "architecture_selection_performed":
            False,

        "best_seed_selection_performed":
            False,

        "best_seed_promotion_performed":
            False,

        "final_test_data_opened":
            False,

        "threshold_landscape_sha256":
            landscape_sha,

        "selected_thresholds_sha256":
            selected_sha,

        "threshold_content_fingerprint":
            threshold_content_fingerprint,
    }

    global_step32_fingerprint = (
        canonical_json_sha256(
            freeze_basis
        )
    )

    # -------------------------------------------------------------------------
    # 9. Write final manifest LAST.
    # -------------------------------------------------------------------------

    final_manifest = {
        "status":
            "STEP32_DEVELOPMENT_THRESHOLD_FINAL_FREEZE_COMPLETE",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "experiment_root":
            str(ROOT),

        "upstream_step31a": {
            "manifest_path":
                str(
                    STEP31A_MANIFEST
                ),

            "manifest_sha256":
                step31a_sha,

            "global_step31_fingerprint":
                EXPECTED_STEP31_GLOBAL_FINGERPRINT,

            "oof_content_fingerprint":
                EXPECTED_OOF_CONTENT_FINGERPRINT,
        },

        "threshold_contract": {
            "architectures":
                SELECTED,

            "horizons":
                HORIZONS,

            "threshold_grid": {
                "start":
                    0.30,

                "stop":
                    0.70,

                "step":
                    0.01,

                "candidate_count":
                    41,
            },

            "threshold_operator":
                ">=",

            "optimization_target":
                "POOLED_DEVELOPMENT_OOF_BALANCED_ACCURACY",

            "tie_breaking": [
                "CLOSEST_TO_0.50",
                "SMALLER_THRESHOLD",
            ],

            "separate_threshold_per_architecture_and_horizon":
                True,

            "input_probability_endpoint":
                "ARITHMETIC_MEAN_OF_FIXED_FIVE_SEED_PROBABILITIES",
        },

        "reconciliation": {
            "architectures":
                8,

            "horizons":
                3,

            "threshold_candidates_per_endpoint":
                41,

            "threshold_landscape_rows":
                984,

            "selected_threshold_rows":
                24,

            "invalid_or_missing":
                0,
        },

        "protection_state": {
            "architecture_selection_performed":
                False,

            "best_seed_selection_performed":
                False,

            "best_seed_promotion_performed":
                False,

            "threshold_optimization_performed":
                True,

            "threshold_optimization_scope":
                "DEVELOPMENT_ONLY_POOLED_OOF",

            "final_test_data_opened":
                False,

            "final_test_metric_computed":
                False,

            "final_test_state":
                "LOCKED_NOT_OPENED",
        },

        "outputs": {
            "threshold_landscape": {
                "path":
                    str(
                        THRESHOLD_LANDSCAPE_CSV.resolve()
                    ),

                "rows":
                    984,

                "sha256":
                    landscape_sha,
            },

            "selected_thresholds": {
                "path":
                    str(
                        SELECTED_THRESHOLDS_CSV.resolve()
                    ),

                "rows":
                    24,

                "sha256":
                    selected_sha,

                "content_fingerprint":
                    threshold_content_fingerprint,
            },

            "frozen_source": {
                "path":
                    str(
                        FROZEN_SOURCE.resolve()
                    ),

                "sha256":
                    frozen_source_sha,
            },
        },

        "selected_thresholds":
            selected_records,

        "global_step32_fingerprint":
            global_step32_fingerprint,

        "next_protocol_gate": {
            "allowed_next_step":
                "FINAL_DEVELOPMENT_REFIT_PREPARATION",

            "selected_configurations_already_frozen":
                True,

            "seed_set_for_final_development_refit":
                SEEDS,

            "thresholds_frozen":
                True,

            "post_threshold_tuning_allowed":
                False,

            "final_test_allowed_now":
                False,

            "note":
                "Prepare final-development refits for the "
                "eight frozen selected configurations using "
                "fixed seeds 42–46. Do not open final-test "
                "data until all final-development artifacts "
                "and inference/evaluation contracts required "
                "before test opening are frozen.",
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
    # 10. Final report.
    # -------------------------------------------------------------------------

    print("[6] STEP-32 FINAL FREEZE")
    print(
        "Global Step-32 fingerprint : "
        f"{global_step32_fingerprint}"
    )
    print(
        "Freeze manifest SHA256     : "
        f"{final_manifest_sha}"
    )
    print()

    print("=" * 140)
    print(
        "STEP 32 COMPLETE — DEVELOPMENT-ONLY "
        "THRESHOLDS FROZEN"
    )
    print("=" * 140)

    print(
        "Architectures                  : 8/8"
    )

    print(
        "Horizons/architecture          : 3/3"
    )

    print(
        "Frozen thresholds              : 24/24"
    )

    print(
        "Threshold grid                 : 0.30–0.70 step 0.01"
    )

    print(
        "Optimization target            : pooled development OOF BA"
    )

    print(
        "Five-seed probability endpoint : FROZEN"
    )

    print(
        "Architecture selection         : NOT PERFORMED"
    )

    print(
        "Best-seed selection            : NOT PERFORMED"
    )

    print(
        "Best-seed promotion            : NOT PERFORMED"
    )

    print(
        "Threshold optimization         : DEVELOPMENT ONLY — COMPLETE"
    )

    print(
        "Final-test data opened         : False"
    )

    print(
        "Final-test state               : LOCKED_NOT_OPENED"
    )

    print("-" * 140)

    print(
        "NEXT GATE: final-development refit preparation."
    )

    print(
        "DO NOT open the final test yet."
    )

    print("=" * 140)


if __name__ == "__main__":
    main()