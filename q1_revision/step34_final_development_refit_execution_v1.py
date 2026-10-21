#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 34 — FINAL-DEVELOPMENT REFIT EXECUTION
============================================

Purpose
-------
Train the eight frozen selected configurations on the frozen Step-14
final-development arrays using the five fixed seeds 42..46.

Total refits:
    8 architectures × 5 fixed seeds = 40

Authoritative training-duration rule
------------------------------------
Step-33 freezes, for each architecture + seed:

    final_refit_epochs
      = median of the five same-seed rolling-origin best_epoch values.

Final-refit differences from rolling-fold fitting
--------------------------------------------------
Only the changes explicitly frozen by Step-33 are permitted:

    * data:
        Step-14 final-development arrays, n=42,360

    * seed:
        one of {42,43,44,45,46}

    * epoch permutation:
        SeedSequence([seed, 0, epoch])

    * training duration:
        frozen Step-33 scheduled epoch count

    * validation:
        NONE

    * early stopping:
        NONE

    * checkpoint:
        final scheduled epoch

Everything else is inherited from the frozen training engine:

    * official model implementations
    * selected configuration
    * learning rate
    * AdamW
    * weight decay = 1e-4
    * BCEWithLogitsLoss(reduction="mean")
    * logical batch = 256
    * selected physical microbatch
    * exact sample-weighted gradient accumulation
    * global gradient clipping norm 1.0
    * FP32
    * AMP=False
    * TF32=False
    * no scheduler

Protection
----------
NO final-test file is opened, created, hashed or evaluated here.

Resumability
------------
A refit is complete only when fit_manifest.json exists and every canonical
artifact registered by that manifest passes hash verification.

Incomplete or invalid refit directories are archived and rerun from scratch.

Recommended workflow
--------------------
Prepare/audit only:
    python step34_final_development_refit_execution_v1.py --prepare-only

One real refit:
    python step34_final_development_refit_execution_v1.py --max-new-fits 1

Continue:
    python step34_final_development_refit_execution_v1.py --max-new-fits N

All remaining:
    python step34_final_development_refit_execution_v1.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import step26_stage1_runner_dryrun_v3 as s26


# =============================================================================
# PATHS
# =============================================================================

SCRIPT = Path(__file__).resolve()
Q1_DIR = SCRIPT.parent

ROOT = (
    Q1_DIR.parent
    / "stock_q1_revision_project"
    / "experiments"
    / "q1_revision_v3_clean_2016_2026"
).resolve()

AUDIT_DIR = ROOT / "audits"
MANIFEST_DIR = ROOT / "manifests"

STEP14_MANIFEST = (
    MANIFEST_DIR
    / "14_model_array_manifest.json"
)

STEP30_MANIFEST = (
    MANIFEST_DIR
    / "30_stage2_global_final_freeze_manifest.json"
)

STEP32_MANIFEST = (
    MANIFEST_DIR
    / "32_development_threshold_final_freeze_manifest.json"
)

STEP33_MANIFEST = (
    MANIFEST_DIR
    / "33_final_development_refit_duration_contract.json"
)

STEP33_SCHEDULE = (
    AUDIT_DIR
    / "33_final_development_refit_epoch_schedule.csv"
)

STEP26_SOURCE = (
    Q1_DIR
    / "step26_stage1_runner_dryrun_v3.py"
)

STEP26_FROZEN_SOURCE = (
    MANIFEST_DIR
    / "step26_stage1_runner_dryrun_frozen.py"
)

FINAL_DEV_DIR = (
    ROOT
    / "feature_data"
    / "step14_model_arrays"
    / "final_development"
)

FINAL_X = (
    FINAL_DEV_DIR
    / "14_final_development_X_full.npy"
)

FINAL_Y = (
    FINAL_DEV_DIR
    / "14_final_development_y_direction.npy"
)

RUN_ROOT = (
    ROOT
    / "final_development_refits"
)

ARCHIVE_ROOT = (
    AUDIT_DIR
    / "step34_incomplete_refit_archives"
)

SOURCE_FREEZE_JSON = (
    MANIFEST_DIR
    / "34_final_development_refit_execution_source_freeze.json"
)

FROZEN_SOURCE = (
    MANIFEST_DIR
    / "step34_final_development_refit_execution_v1_frozen.py"
)

EXECUTION_PLAN_CSV = (
    AUDIT_DIR
    / "34_final_development_refit_execution_plan.csv"
)

PROTOCOL_REFERENCE_JSON = (
    MANIFEST_DIR
    / "34_final_development_refit_protocol_reference.json"
)

FIT_REGISTER_CSV = (
    AUDIT_DIR
    / "34_final_development_refit_register.csv"
)

ARTIFACT_REGISTER_CSV = (
    AUDIT_DIR
    / "34_final_development_refit_artifact_register.csv"
)

FINAL_FREEZE_MANIFEST = (
    MANIFEST_DIR
    / "34_final_development_refit_final_freeze_manifest.json"
)


# =============================================================================
# AUTHORITATIVE UPSTREAM IDENTIFIERS
# =============================================================================

EXPECTED_STEP30_MANIFEST_SHA256 = (
    "8216ac6947a2c2c2741758f76d9c38f085f59ac4158f9fcdc58effb18c1ab420"
)

EXPECTED_STEP30_GLOBAL_FINGERPRINT = (
    "328c77bac04ad61dabb9fe50f9b481389a94b95709cefe3c00fc680e7c807bed"
)

EXPECTED_STEP32_MANIFEST_SHA256 = (
    "474b77ac670bbb7f61870372cdb98a6d7ac3ecaa4db8864ceb096fe9b26c39f0"
)

EXPECTED_STEP32_GLOBAL_FINGERPRINT = (
    "bfc00e361e883fb2066e4254c67f73ac3235d2abafde32e3ad989b59e1f8e92d"
)

EXPECTED_STEP33_MANIFEST_SHA256 = (
    "d4905e058a409cba7e49ffe2410be70278103011c240b375a12c78219a91f18d"
)

EXPECTED_STEP33_GLOBAL_FINGERPRINT = (
    "18a23cce7380c1e16e2b1503bfa7c7ed09c99becee645e1cb42abcd19a68f1ee"
)

EXPECTED_STEP33_SCHEDULE_SHA256 = (
    "49ffe440d9d9187357731aa7841d781137463f1d79a4d0e966c6a6411753c827"
)

EXPECTED_STEP33_SCHEDULE_FINGERPRINT = (
    "2bbae72b6a8be938c7f6c1448cf57b956bd1e341204b38103143aff966360a3b"
)


# =============================================================================
# FROZEN FINAL-REFIT CONTRACT
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

MODEL_ORDER = [
    "DLINEAR",
    "NLINEAR",
    "PATCHTST",
    "TIMESNET",
    "ITRANSFORMER",
    "FREDFORMER",
    "FRNET",
    "MLF",
]

SEEDS = [42, 43, 44, 45, 46]

EXPECTED_FINAL_N = 42360
LOOKBACK = 60
FEATURES = 63
OUTPUTS = 3
HORIZONS = [1, 3, 5]

LOGICAL_BATCH = 256
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

EXPECTED_PHYSICAL = {
    "DLINEAR": 128,
    "NLINEAR": 128,
    "PATCHTST": 64,
    "TIMESNET": 64,
    "ITRANSFORMER": 128,
    "FREDFORMER": 16,
    "FRNET": 64,
    "MLF": 32,
}

FINAL_DEVELOPMENT_PARTITION_TOKEN = 0

EXPECTED_FITS = 40

CANONICAL_ARTIFACTS = (
    "final_checkpoint.pt",
    "epoch_history.csv",
    "fit_metrics.json",
)

RESULT_PREFIX = "__STEP34_RESULT__"


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    if not path.is_file():
        fail(f"Missing file: {path}")

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def canonical_hash(obj) -> str:

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
        fail(f"Missing JSON: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        fail(f"Expected JSON object: {path}")

    return obj


def atomic_json(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

        f.write("\n")

    os.replace(
        tmp,
        path,
    )


def atomic_csv(
    path: Path,
    frame: pd.DataFrame,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    frame.to_csv(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        path,
    )


def atomic_torch_save(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    torch.save(
        payload,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


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


def artifact_record(
    path: Path,
) -> dict:

    return {
        "path": str(
            path.resolve()
        ),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(
            path
        ),
    }


# =============================================================================
# FINAL-REFIT PATHS
# =============================================================================

def fit_dir(
    model_id: str,
    config_id: str,
    seed: int,
) -> Path:

    return (
        RUN_ROOT
        / model_id
        / config_id
        / f"seed_{seed}"
    )


def fit_manifest_path(
    model_id: str,
    config_id: str,
    seed: int,
) -> Path:

    return (
        fit_dir(
            model_id,
            config_id,
            seed,
        )
        / "fit_manifest.json"
    )


# =============================================================================
# UPSTREAM VERIFICATION
# =============================================================================

def verify_upstream() -> dict:

    require_equal(
        sha256_file(
            STEP30_MANIFEST
        ),
        EXPECTED_STEP30_MANIFEST_SHA256,
        "Step30 manifest SHA256",
    )

    step30 = load_json(
        STEP30_MANIFEST
    )

    require_equal(
        step30.get(
            "global_stage2_freeze_fingerprint"
        ),
        EXPECTED_STEP30_GLOBAL_FINGERPRINT,
        "Step30 global fingerprint",
    )

    require_equal(
        sha256_file(
            STEP32_MANIFEST
        ),
        EXPECTED_STEP32_MANIFEST_SHA256,
        "Step32 manifest SHA256",
    )

    step32 = load_json(
        STEP32_MANIFEST
    )

    require_equal(
        step32.get(
            "global_step32_fingerprint"
        ),
        EXPECTED_STEP32_GLOBAL_FINGERPRINT,
        "Step32 global fingerprint",
    )

    p32 = step32.get(
        "protection_state",
        {},
    )

    require_false(
        p32.get(
            "best_seed_selection_performed"
        ),
        "Step32 best_seed_selection",
    )

    require_false(
        p32.get(
            "best_seed_promotion_performed"
        ),
        "Step32 best_seed_promotion",
    )

    require_false(
        p32.get(
            "final_test_data_opened"
        ),
        "Step32 final_test access",
    )

    require_equal(
        sha256_file(
            STEP33_MANIFEST
        ),
        EXPECTED_STEP33_MANIFEST_SHA256,
        "Step33 manifest SHA256",
    )

    step33 = load_json(
        STEP33_MANIFEST
    )

    require_equal(
        step33.get(
            "status"
        ),
        "STEP33_FINAL_DEVELOPMENT_REFIT_DURATION_CONTRACT_FROZEN",
        "Step33 status",
    )

    require_equal(
        step33.get(
            "global_step33_fingerprint"
        ),
        EXPECTED_STEP33_GLOBAL_FINGERPRINT,
        "Step33 global fingerprint",
    )

    require_equal(
        sha256_file(
            STEP33_SCHEDULE
        ),
        EXPECTED_STEP33_SCHEDULE_SHA256,
        "Step33 schedule SHA256",
    )

    require_equal(
        step33[
            "outputs"
        ][
            "final_refit_epoch_schedule"
        ][
            "content_fingerprint"
        ],
        EXPECTED_STEP33_SCHEDULE_FINGERPRINT,
        "Step33 schedule fingerprint",
    )

    p33 = step33.get(
        "protection_state",
        {},
    )

    require_false(
        p33.get(
            "architecture_selection_performed"
        ),
        "Step33 architecture selection",
    )

    require_false(
        p33.get(
            "best_seed_selection_performed"
        ),
        "Step33 best-seed selection",
    )

    require_false(
        p33.get(
            "best_seed_promotion_performed"
        ),
        "Step33 best-seed promotion",
    )

    require_false(
        p33.get(
            "final_test_data_opened"
        ),
        "Step33 final-test access",
    )

    require_equal(
        p33.get(
            "final_test_state"
        ),
        "LOCKED_NOT_OPENED",
        "Step33 final-test state",
    )

    # -----------------------------------------------------------------
    # Step-26 executable must remain byte-identical to its frozen copy.
    # -----------------------------------------------------------------

    step26_sha = sha256_file(
        STEP26_SOURCE
    )

    step26_frozen_sha = sha256_file(
        STEP26_FROZEN_SOURCE
    )

    require_equal(
        step26_sha,
        step26_frozen_sha,
        "Step26 executable/frozen source SHA256",
    )

    # -----------------------------------------------------------------
    # Final-development arrays must match Step-33 frozen hashes.
    # -----------------------------------------------------------------

    require_equal(
        sha256_file(
            FINAL_X
        ),
        step33[
            "final_development_data"
        ][
            "x_full_sha256"
        ],
        "Final-development X SHA256",
    )

    require_equal(
        sha256_file(
            FINAL_Y
        ),
        step33[
            "final_development_data"
        ][
            "y_direction_sha256"
        ],
        "Final-development y SHA256",
    )

    x = np.load(
        FINAL_X,
        mmap_mode="r",
        allow_pickle=False,
    )

    y = np.load(
        FINAL_Y,
        mmap_mode="r",
        allow_pickle=False,
    )

    require_equal(
        x.shape,
        (
            EXPECTED_FINAL_N,
            LOOKBACK,
            FEATURES,
        ),
        "Final-development X shape",
    )

    require_equal(
        y.shape,
        (
            EXPECTED_FINAL_N,
            OUTPUTS,
        ),
        "Final-development y shape",
    )

    if not set(
        np.unique(y).tolist()
    ).issubset({0, 1}):

        fail(
            "Final-development labels are not binary."
        )

    del x, y

    # -----------------------------------------------------------------
    # Load logical robustness inventory as immutable protocol reference.
    # -----------------------------------------------------------------

    inventory_meta = (
        step30[
            "outputs"
        ][
            "logical_fit_inventory"
        ]
    )

    inventory_path = Path(
        inventory_meta[
            "path"
        ]
    ).expanduser().resolve()

    require_equal(
        sha256_file(
            inventory_path
        ),
        inventory_meta[
            "sha256"
        ],
        "Step30 logical inventory SHA256",
    )

    inventory = pd.read_csv(
        inventory_path
    )

    require_equal(
        len(inventory),
        200,
        "Step30 logical inventory rows",
    )

    return {
        "step30":
            step30,

        "step32":
            step32,

        "step33":
            step33,

        "step26_source_sha256":
            step26_sha,

        "logical_inventory":
            inventory,

        "logical_inventory_path":
            inventory_path,

        "final_x_sha256":
            sha256_file(FINAL_X),

        "final_y_sha256":
            sha256_file(FINAL_Y),
    }


# =============================================================================
# SOURCE FREEZE
# =============================================================================

def build_source_semantic(
    upstream: dict,
) -> dict:

    return {
        "status":
            "STEP34_FINAL_DEVELOPMENT_REFIT_EXECUTION_SOURCE_FROZEN",

        "step34_source_sha256":
            sha256_file(
                SCRIPT
            ),

        "upstream": {
            "step30_manifest_sha256":
                EXPECTED_STEP30_MANIFEST_SHA256,

            "step30_global_fingerprint":
                EXPECTED_STEP30_GLOBAL_FINGERPRINT,

            "step32_manifest_sha256":
                EXPECTED_STEP32_MANIFEST_SHA256,

            "step32_global_fingerprint":
                EXPECTED_STEP32_GLOBAL_FINGERPRINT,

            "step33_manifest_sha256":
                EXPECTED_STEP33_MANIFEST_SHA256,

            "step33_global_fingerprint":
                EXPECTED_STEP33_GLOBAL_FINGERPRINT,

            "step33_schedule_sha256":
                EXPECTED_STEP33_SCHEDULE_SHA256,

            "step33_schedule_fingerprint":
                EXPECTED_STEP33_SCHEDULE_FINGERPRINT,

            "step26_source_sha256":
                upstream[
                    "step26_source_sha256"
                ],
        },

        "execution_contract": {
            "selected_configurations":
                SELECTED,

            "seeds":
                SEEDS,

            "total_refits":
                EXPECTED_FITS,

            "final_development_n":
                EXPECTED_FINAL_N,

            "lookback":
                LOOKBACK,

            "features":
                FEATURES,

            "horizons":
                HORIZONS,

            "logical_batch":
                LOGICAL_BATCH,

            "weight_decay":
                WEIGHT_DECAY,

            "gradient_clip_norm":
                GRADIENT_CLIP_NORM,

            "duration_rule":
                "MEDIAN_OF_5_SAME_SEED_ROLLING_FOLD_BEST_EPOCHS",

            "shuffle_seed_contract":
                "SeedSequence([seed,0,epoch])",

            "validation":
                False,

            "early_stopping":
                False,

            "checkpoint":
                "FINAL_SCHEDULED_EPOCH",

            "final_test_access":
                False,
        },
    }


def ensure_source_freeze(
    upstream: dict,
) -> tuple[str, str]:

    semantic = build_source_semantic(
        upstream
    )

    source_fp = canonical_hash(
        semantic
    )

    payload = dict(
        semantic
    )

    payload[
        "source_fingerprint_sha256"
    ] = source_fp

    if SOURCE_FREEZE_JSON.exists():

        old = load_json(
            SOURCE_FREEZE_JSON
        )

        require_equal(
            old,
            payload,
            "Existing Step34 source freeze",
        )

    else:

        atomic_json(
            SOURCE_FREEZE_JSON,
            payload,
        )

    current_sha = sha256_file(
        SCRIPT
    )

    if FROZEN_SOURCE.exists():

        require_equal(
            sha256_file(
                FROZEN_SOURCE
            ),
            current_sha,
            "Step34 frozen source SHA256",
        )

    else:

        shutil.copy2(
            SCRIPT,
            FROZEN_SOURCE,
        )

    require_equal(
        sha256_file(
            FROZEN_SOURCE
        ),
        current_sha,
        "Step34 frozen/executable source",
    )

    return (
        current_sha,
        source_fp,
    )


# =============================================================================
# PROTOCOL REFERENCE FROM FROZEN ROBUSTNESS MANIFESTS
# =============================================================================

def build_protocol_reference(
    upstream: dict,
) -> dict:

    inventory = upstream[
        "logical_inventory"
    ]

    result = {}

    for model_id in MODEL_ORDER:

        config_id = SELECTED[
            model_id
        ]

        g = inventory.loc[
            (
                inventory[
                    "model_id"
                ].astype(str)
                == model_id
            )
            &
            (
                inventory[
                    "config_id"
                ].astype(str)
                == config_id
            )
        ].copy()

        require_equal(
            len(g),
            25,
            (
                f"{model_id}/{config_id}: "
                "frozen robustness manifests"
            ),
        )

        values = []

        for row in g.itertuples(
            index=False
        ):

            manifest_path = Path(
                str(
                    row.fit_manifest_path
                )
            ).expanduser().resolve()

            require_equal(
                sha256_file(
                    manifest_path
                ),
                str(
                    row.fit_manifest_sha256
                ),
                (
                    f"{model_id}/{config_id}: "
                    "frozen fit manifest SHA256"
                ),
            )

            m = load_json(
                manifest_path
            )

            values.append(
                {
                    "logical_batch_size":
                        int(
                            m[
                                "logical_batch_size"
                            ]
                        ),

                    "physical_microbatch_size":
                        int(
                            m[
                                "physical_microbatch_size"
                            ]
                        ),

                    "learning_rate":
                        float(
                            m[
                                "learning_rate"
                            ]
                        ),

                    "weight_decay":
                        float(
                            m[
                                "weight_decay"
                            ]
                        ),

                    "gradient_clip_norm":
                        float(
                            m[
                                "gradient_clip_norm"
                            ]
                        ),

                    "optimizer":
                        str(
                            m[
                                "optimizer"
                            ]
                        ),

                    "adamw_betas":
                        list(
                            m[
                                "adamw_betas"
                            ]
                        ),

                    "adamw_eps":
                        float(
                            m[
                                "adamw_eps"
                            ]
                        ),

                    "loss":
                        str(
                            m[
                                "loss"
                            ]
                        ),

                    "dtype":
                        str(
                            m[
                                "dtype"
                            ]
                        ),

                    "amp_enabled":
                        bool(
                            m[
                                "amp_enabled"
                            ]
                        ),

                    "tf32_enabled":
                        bool(
                            m[
                                "tf32_enabled"
                            ]
                        ),

                    "scheduler":
                        str(
                            m[
                                "scheduler"
                            ]
                        ),

                    "trainable_parameter_count":
                        int(
                            m[
                                "trainable_parameter_count"
                            ]
                        ),

                    "total_parameter_count":
                        int(
                            m[
                                "total_parameter_count"
                            ]
                        ),

                    "model_build_metadata_fingerprint":
                        canonical_hash(
                            m[
                                "model_build_metadata"
                            ]
                        ),

                    "source_repository_record_fingerprint":
                        canonical_hash(
                            m[
                                "source_repository_record"
                            ]
                        ),
                }
            )

        first = values[0]

        for v in values[1:]:

            require_equal(
                v,
                first,
                (
                    f"{model_id}/{config_id}: "
                    "training/build protocol drift "
                    "across frozen robustness fits"
                ),
            )

        require_equal(
            first[
                "logical_batch_size"
            ],
            LOGICAL_BATCH,
            f"{model_id}: logical batch",
        )

        require_equal(
            first[
                "physical_microbatch_size"
            ],
            EXPECTED_PHYSICAL[
                model_id
            ],
            f"{model_id}: physical microbatch",
        )

        require_equal(
            first[
                "weight_decay"
            ],
            WEIGHT_DECAY,
            f"{model_id}: weight decay",
        )

        require_equal(
            first[
                "gradient_clip_norm"
            ],
            GRADIENT_CLIP_NORM,
            f"{model_id}: gradient clip",
        )

        require_equal(
            first[
                "optimizer"
            ],
            "AdamW",
            f"{model_id}: optimizer",
        )

        require_equal(
            first[
                "amp_enabled"
            ],
            False,
            f"{model_id}: AMP",
        )

        require_equal(
            first[
                "tf32_enabled"
            ],
            False,
            f"{model_id}: TF32",
        )

        require_equal(
            first[
                "scheduler"
            ],
            "none",
            f"{model_id}: scheduler",
        )

        result[
            model_id
        ] = {
            "model_id":
                model_id,

            "config_id":
                config_id,

            **first,
        }

    return result


def ensure_protocol_reference(
    upstream: dict,
) -> tuple[dict, str]:

    reference = {
        "status":
            "STEP34_FINAL_REFIT_PROTOCOL_REFERENCE",

        "created_from":
            "200 frozen Step30 robustness logical-fit manifests",

        "performance_metrics_used":
            False,

        "models":
            build_protocol_reference(
                upstream
            ),
    }

    reference[
        "content_fingerprint_sha256"
    ] = canonical_hash(
        reference[
            "models"
        ]
    )

    if PROTOCOL_REFERENCE_JSON.exists():

        old = load_json(
            PROTOCOL_REFERENCE_JSON
        )

        # created timestamps are intentionally absent,
        # so the object must reproduce exactly.
        require_equal(
            old,
            reference,
            "Existing Step34 protocol reference",
        )

    else:

        atomic_json(
            PROTOCOL_REFERENCE_JSON,
            reference,
        )

    return (
        reference,
        sha256_file(
            PROTOCOL_REFERENCE_JSON
        ),
    )


# =============================================================================
# EXECUTION PLAN
# =============================================================================

def build_execution_plan() -> pd.DataFrame:

    schedule = pd.read_csv(
        STEP33_SCHEDULE
    )

    require_equal(
        len(schedule),
        EXPECTED_FITS,
        "Step33 schedule rows",
    )

    rows = []

    for model_id in MODEL_ORDER:

        config_id = SELECTED[
            model_id
        ]

        for seed in SEEDS:

            g = schedule.loc[
                (
                    schedule[
                        "model_id"
                    ].astype(str)
                    == model_id
                )
                &
                (
                    schedule[
                        "config_id"
                    ].astype(str)
                    == config_id
                )
                &
                (
                    schedule[
                        "seed"
                    ].astype(int)
                    == seed
                )
            ]

            require_equal(
                len(g),
                1,
                (
                    f"{model_id}/{config_id}/"
                    f"seed{seed}: schedule row"
                ),
            )

            r = g.iloc[0]

            require_equal(
                int(
                    r[
                        "final_development_n"
                    ]
                ),
                EXPECTED_FINAL_N,
                f"{model_id}/seed{seed}: final n",
            )

            require_equal(
                int(
                    r[
                        "logical_batch_size"
                    ]
                ),
                LOGICAL_BATCH,
                f"{model_id}/seed{seed}: logical batch",
            )

            require_equal(
                int(
                    r[
                        "physical_microbatch_size"
                    ]
                ),
                EXPECTED_PHYSICAL[
                    model_id
                ],
                f"{model_id}/seed{seed}: microbatch",
            )

            require_equal(
                str(
                    r[
                        "checkpoint_policy"
                    ]
                ),
                "FINAL_SCHEDULED_EPOCH",
                f"{model_id}/seed{seed}: checkpoint",
            )

            require_equal(
                str(
                    r[
                        "duration_rule"
                    ]
                ),
                "MEDIAN_OF_5_SAME_SEED_ROLLING_FOLD_BEST_EPOCHS",
                f"{model_id}/seed{seed}: duration rule",
            )

            require_equal(
                bool(
                    r[
                        "validation_during_final_refit"
                    ]
                ),
                False,
                f"{model_id}/seed{seed}: validation",
            )

            require_equal(
                bool(
                    r[
                        "early_stopping_during_final_refit"
                    ]
                ),
                False,
                f"{model_id}/seed{seed}: early stop",
            )

            epochs = int(
                r[
                    "final_refit_epochs"
                ]
            )

            if epochs < 1:
                fail(
                    f"{model_id}/seed{seed}: invalid "
                    f"scheduled epochs={epochs}"
                )

            rows.append(
                {
                    "model_id":
                        model_id,

                    "config_id":
                        config_id,

                    "seed":
                        seed,

                    "final_refit_epochs":
                        epochs,

                    "physical_microbatch_size":
                        EXPECTED_PHYSICAL[
                            model_id
                        ],

                    "logical_batch_size":
                        LOGICAL_BATCH,

                    "final_development_n":
                        EXPECTED_FINAL_N,

                    "fit_directory":
                        str(
                            fit_dir(
                                model_id,
                                config_id,
                                seed,
                            ).resolve()
                        ),

                    "final_test_used":
                        False,
                }
            )

    plan = pd.DataFrame(
        rows
    )

    require_equal(
        len(plan),
        EXPECTED_FITS,
        "Step34 execution plan rows",
    )

    return plan


def ensure_execution_plan() -> tuple[pd.DataFrame, str, str]:

    plan = build_execution_plan()

    if EXECUTION_PLAN_CSV.exists():

        old = pd.read_csv(
            EXECUTION_PLAN_CSV
        )

        pd.testing.assert_frame_equal(
            old,
            plan,
            check_dtype=False,
        )

    else:

        atomic_csv(
            EXECUTION_PLAN_CSV,
            plan,
        )

    plan_sha = sha256_file(
        EXECUTION_PLAN_CSV
    )

    plan_fp = canonical_hash(
        plan.to_dict(
            orient="records"
        )
    )

    return (
        plan,
        plan_sha,
        plan_fp,
    )


# =============================================================================
# RESUME / VALIDATION
# =============================================================================

def verify_completed_fit(
    model_id: str,
    config_id: str,
    seed: int,
    expected_epochs: int,
    source_fp: str,
    protocol_reference: dict,
    upstream: dict,
) -> dict | None:

    directory = fit_dir(
        model_id,
        config_id,
        seed,
    )

    manifest_path = (
        directory
        / "fit_manifest.json"
    )

    if not manifest_path.is_file():
        return None

    try:

        m = load_json(
            manifest_path
        )

        require_equal(
            m.get("status"),
            "FINAL_DEVELOPMENT_REFIT_COMPLETE",
            "Refit status",
        )

        require_equal(
            m.get("model_id"),
            model_id,
            "Refit model",
        )

        require_equal(
            m.get("config_id"),
            config_id,
            "Refit config",
        )

        require_equal(
            int(
                m.get(
                    "seed",
                    -1,
                )
            ),
            seed,
            "Refit seed",
        )

        require_equal(
            int(
                m.get(
                    "scheduled_epochs",
                    -1,
                )
            ),
            expected_epochs,
            "Refit scheduled epochs",
        )

        require_equal(
            int(
                m.get(
                    "epochs_run",
                    -1,
                )
            ),
            expected_epochs,
            "Refit epochs run",
        )

        require_equal(
            m.get(
                "checkpoint_policy"
            ),
            "FINAL_SCHEDULED_EPOCH",
            "Refit checkpoint policy",
        )

        require_equal(
            m.get(
                "validation_used"
            ),
            False,
            "Refit validation",
        )

        require_equal(
            m.get(
                "early_stopping_used"
            ),
            False,
            "Refit early stopping",
        )

        require_equal(
            m.get(
                "step34_execution_source_fingerprint_sha256"
            ),
            source_fp,
            "Step34 source fingerprint",
        )

        require_equal(
            m.get(
                "step33_global_fingerprint"
            ),
            EXPECTED_STEP33_GLOBAL_FINGERPRINT,
            "Step33 fingerprint in refit",
        )

        require_equal(
            m.get(
                "final_test_data_opened"
            ),
            False,
            "Refit test state",
        )

        ref = (
            protocol_reference[
                "models"
            ][
                model_id
            ]
        )

        require_equal(
            float(
                m[
                    "learning_rate"
                ]
            ),
            float(
                ref[
                    "learning_rate"
                ]
            ),
            "Refit LR",
        )

        require_equal(
            int(
                m[
                    "physical_microbatch_size"
                ]
            ),
            int(
                ref[
                    "physical_microbatch_size"
                ]
            ),
            "Refit physical microbatch",
        )

        inputs = m[
            "inputs"
        ]

        require_equal(
            inputs[
                "train_x"
            ][
                "sha256"
            ],
            upstream[
                "final_x_sha256"
            ],
            "Refit X SHA256",
        )

        require_equal(
            inputs[
                "train_y"
            ][
                "sha256"
            ],
            upstream[
                "final_y_sha256"
            ],
            "Refit y SHA256",
        )

        artifacts = m[
            "artifacts"
        ]

        for name in CANONICAL_ARTIFACTS:

            rec = artifacts[
                name
            ]

            p = Path(
                rec[
                    "path"
                ]
            )

            require_equal(
                sha256_file(
                    p
                ),
                rec[
                    "sha256"
                ],
                (
                    f"{model_id}/seed{seed}/"
                    f"{name} SHA256"
                ),
            )

            require_equal(
                int(
                    p.stat().st_size
                ),
                int(
                    rec[
                        "size_bytes"
                    ]
                ),
                (
                    f"{model_id}/seed{seed}/"
                    f"{name} size"
                ),
            )

        return m

    except Exception:

        return None


def archive_incomplete_fit(
    model_id: str,
    config_id: str,
    seed: int,
) -> None:

    directory = fit_dir(
        model_id,
        config_id,
        seed,
    )

    if not directory.exists():
        return

    if not any(
        directory.iterdir()
    ):
        directory.rmdir()
        return

    ARCHIVE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    target = (
        ARCHIVE_ROOT
        / (
            f"{model_id}__{config_id}"
            f"__seed_{seed}"
            f"__{stamp}"
        )
    )

    suffix = 1

    while target.exists():

        target = (
            ARCHIVE_ROOT
            / (
                f"{model_id}__{config_id}"
                f"__seed_{seed}"
                f"__{stamp}"
                f"__{suffix}"
            )
        )

        suffix += 1

    shutil.move(
        str(directory),
        str(target),
    )


# =============================================================================
# TRAINING
# =============================================================================

def epoch_permutation(
    n: int,
    seed: int,
    epoch: int,
) -> np.ndarray:

    if epoch < 1:
        fail(
            "Epoch must be 1-indexed."
        )

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                FINAL_DEVELOPMENT_PARTITION_TOKEN,
                int(epoch),
            ]
        )
    )

    p = rng.permutation(
        n
    )

    if p.shape != (n,):
        fail(
            "Final-development epoch permutation "
            "shape mismatch."
        )

    return p


def cpu_state_dict(
    model: torch.nn.Module,
) -> dict:

    return {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in model.state_dict().items()
    }


def worker_fit(
    model_id: str,
    config_id: str,
    seed: int,
) -> dict:

    started = time.perf_counter()

    upstream = verify_upstream()

    source_sha, source_fp = (
        ensure_source_freeze(
            upstream
        )
    )

    protocol_reference, protocol_reference_sha = (
        ensure_protocol_reference(
            upstream
        )
    )

    plan, plan_sha, plan_fp = (
        ensure_execution_plan()
    )

    g = plan.loc[
        (
            plan[
                "model_id"
            ].astype(str)
            == model_id
        )
        &
        (
            plan[
                "config_id"
            ].astype(str)
            == config_id
        )
        &
        (
            plan[
                "seed"
            ].astype(int)
            == seed
        )
    ]

    require_equal(
        len(g),
        1,
        "Worker execution-plan lookup",
    )

    plan_row = g.iloc[0]

    scheduled_epochs = int(
        plan_row[
            "final_refit_epochs"
        ]
    )

    physical_microbatch = int(
        plan_row[
            "physical_microbatch_size"
        ]
    )

    reference = (
        protocol_reference[
            "models"
        ][
            model_id
        ]
    )

    # -----------------------------------------------------------------
    # Load only frozen final-development arrays.
    # -----------------------------------------------------------------

    x_train = np.load(
        FINAL_X,
        mmap_mode="r",
        allow_pickle=False,
    )

    y_train = np.load(
        FINAL_Y,
        mmap_mode="r",
        allow_pickle=False,
    )

    require_equal(
        x_train.shape,
        (
            EXPECTED_FINAL_N,
            LOOKBACK,
            FEATURES,
        ),
        "Worker X shape",
    )

    require_equal(
        y_train.shape,
        (
            EXPECTED_FINAL_N,
            OUTPUTS,
        ),
        "Worker y shape",
    )

    # -----------------------------------------------------------------
    # Exact fixed-seed training determinism.
    # -----------------------------------------------------------------

    s26.set_training_determinism(
        int(seed)
    )

    device = torch.device(
        "cuda:0"
    )

    device_index = 0

    # -----------------------------------------------------------------
    # Frozen model construction.
    # -----------------------------------------------------------------

    model, build_metadata, repository = (
        s26.build_candidate_model(
            model_id=model_id,
            config_id=config_id,
            physical_microbatch=physical_microbatch,
            device=device,
        )
    )

    learning_rate = float(
        build_metadata[
            "learning_rate"
        ]
    )

    require_equal(
        learning_rate,
        float(
            reference[
                "learning_rate"
            ]
        ),
        "Runtime/frozen learning rate",
    )

    runtime_build_fp = canonical_hash(
        build_metadata
    )

    require_equal(
        runtime_build_fp,
        reference[
            "model_build_metadata_fingerprint"
        ],
        (
            f"{model_id}: runtime model-build "
            "metadata fingerprint"
        ),
    )

    runtime_repo_fp = canonical_hash(
        repository
    )

    require_equal(
        runtime_repo_fp,
        reference[
            "source_repository_record_fingerprint"
        ],
        (
            f"{model_id}: runtime repository "
            "record fingerprint"
        ),
    )

    criterion = nn.BCEWithLogitsLoss(
        reduction="mean"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=s26.BETAS,
        eps=s26.ADAMW_EPS,
        weight_decay=WEIGHT_DECAY,
    )

    trainable_parameters = int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    total_parameters = int(
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    require_equal(
        trainable_parameters,
        int(
            reference[
                "trainable_parameter_count"
            ]
        ),
        "Runtime trainable parameter count",
    )

    require_equal(
        total_parameters,
        int(
            reference[
                "total_parameter_count"
            ]
        ),
        "Runtime total parameter count",
    )

    directory = fit_dir(
        model_id,
        config_id,
        seed,
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        directory
        / "final_checkpoint.pt"
    )

    history_path = (
        directory
        / "epoch_history.csv"
    )

    metrics_path = (
        directory
        / "fit_metrics.json"
    )

    manifest_path = (
        directory
        / "fit_manifest.json"
    )

    for path in (
        checkpoint_path,
        history_path,
        metrics_path,
        manifest_path,
    ):

        if path.exists():
            path.unlink()

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        device_index
    )

    epoch_rows = []

    total_training_seconds = 0.0

    final_train_bce = None

    # -----------------------------------------------------------------
    # Fixed scheduled training — NO validation and NO early stopping.
    # -----------------------------------------------------------------

    for epoch in range(
        1,
        scheduled_epochs + 1,
    ):

        epoch_started = time.perf_counter()

        permutation = epoch_permutation(
            len(x_train),
            seed=seed,
            epoch=epoch,
        )

        train_started = time.perf_counter()

        sample_weighted_bce_sum = 0.0
        train_samples_seen = 0
        logical_updates = 0
        grad_norms = []

        for logical_start in range(
            0,
            len(permutation),
            LOGICAL_BATCH,
        ):

            logical_indices = permutation[
                logical_start:
                logical_start + LOGICAL_BATCH
            ]

            update = s26.train_logical_batch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                x_source=x_train,
                y_source=y_train,
                logical_indices=logical_indices,
                physical_microbatch=physical_microbatch,
                device=device,
            )

            logical_n = int(
                update[
                    "logical_batch_n"
                ]
            )

            sample_weighted_bce_sum += (
                float(
                    update[
                        "weighted_mean_bce"
                    ]
                )
                * logical_n
            )

            train_samples_seen += logical_n

            logical_updates += 1

            grad_norms.append(
                float(
                    update[
                        "preclip_gradient_norm"
                    ]
                )
            )

        torch.cuda.synchronize(
            device_index
        )

        training_seconds = (
            time.perf_counter()
            - train_started
        )

        total_training_seconds += (
            training_seconds
        )

        require_equal(
            train_samples_seen,
            len(x_train),
            (
                f"{model_id}/seed{seed}/epoch{epoch}: "
                "training coverage"
            ),
        )

        require_equal(
            logical_updates,
            math.ceil(
                len(x_train)
                / LOGICAL_BATCH
            ),
            (
                f"{model_id}/seed{seed}/epoch{epoch}: "
                "logical update count"
            ),
        )

        train_bce = (
            sample_weighted_bce_sum
            / train_samples_seen
        )

        if not math.isfinite(
            train_bce
        ):
            fail(
                "Non-finite final-development "
                "training BCE."
            )

        final_train_bce = float(
            train_bce
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_started
        )

        epoch_rows.append(
            {
                "epoch":
                    int(epoch),

                "train_mean_directional_bce":
                    float(train_bce),

                "learning_rate":
                    learning_rate,

                "mean_preclip_gradient_norm":
                    float(
                        np.mean(
                            grad_norms
                        )
                    ),

                "epoch_seconds":
                    float(
                        epoch_seconds
                    ),

                "training_seconds":
                    float(
                        training_seconds
                    ),

                "logical_optimizer_updates":
                    int(
                        logical_updates
                    ),

                "training_samples_seen":
                    int(
                        train_samples_seen
                    ),
            }
        )

        print(
            f"EPOCH "
            f"{model_id}/{config_id}/"
            f"seed{seed} "
            f"{epoch:03d}/"
            f"{scheduled_epochs:03d} "
            f"trainBCE={train_bce:.6f} "
            f"updates={logical_updates} "
            f"sec={epoch_seconds:.1f}",
            flush=True,
        )

    require_equal(
        len(epoch_rows),
        scheduled_epochs,
        "Worker scheduled epoch completion",
    )

    if final_train_bce is None:
        fail(
            "No final training BCE produced."
        )

    # -----------------------------------------------------------------
    # Final scheduled-epoch checkpoint.
    # -----------------------------------------------------------------

    checkpoint_payload = {
        "schema_version":
            "1.0",

        "status":
            "FINAL_DEVELOPMENT_REFIT_CHECKPOINT",

        "model_id":
            model_id,

        "config_id":
            config_id,

        "seed":
            int(seed),

        "epoch":
            int(
                scheduled_epochs
            ),

        "scheduled_epochs":
            int(
                scheduled_epochs
            ),

        "checkpoint_policy":
            "FINAL_SCHEDULED_EPOCH",

        "validation_used":
            False,

        "early_stopping_used":
            False,

        "final_train_mean_directional_bce":
            float(
                final_train_bce
            ),

        "model_state_dict":
            cpu_state_dict(
                model
            ),
    }

    atomic_torch_save(
        checkpoint_path,
        checkpoint_payload,
    )

    history_df = pd.DataFrame(
        epoch_rows
    )

    atomic_csv(
        history_path,
        history_df,
    )

    torch.cuda.synchronize(
        device_index
    )

    peak_allocated_bytes = int(
        torch.cuda.max_memory_allocated(
            device_index
        )
    )

    wall_seconds = (
        time.perf_counter()
        - started
    )

    fit_metrics = {
        "status":
            "FINAL_DEVELOPMENT_REFIT_COMPLETE",

        "model_id":
            model_id,

        "config_id":
            config_id,

        "seed":
            int(seed),

        "train_n":
            EXPECTED_FINAL_N,

        "scheduled_epochs":
            int(
                scheduled_epochs
            ),

        "epochs_run":
            int(
                scheduled_epochs
            ),

        "final_train_mean_directional_bce":
            float(
                final_train_bce
            ),

        "training_seconds":
            float(
                total_training_seconds
            ),

        "wall_seconds":
            float(
                wall_seconds
            ),

        "peak_cuda_allocated_bytes":
            int(
                peak_allocated_bytes
            ),

        "peak_cuda_allocated_mib":
            float(
                peak_allocated_bytes
                / (1024 ** 2)
            ),

        "logical_batch_size":
            LOGICAL_BATCH,

        "physical_microbatch_size":
            physical_microbatch,

        "learning_rate":
            learning_rate,

        "trainable_parameter_count":
            trainable_parameters,

        "total_parameter_count":
            total_parameters,

        "validation_metrics_computed":
            False,

        "final_test_metrics_computed":
            False,
    }

    atomic_json(
        metrics_path,
        fit_metrics,
    )

    artifacts = {
        "final_checkpoint.pt":
            artifact_record(
                checkpoint_path
            ),

        "epoch_history.csv":
            artifact_record(
                history_path
            ),

        "fit_metrics.json":
            artifact_record(
                metrics_path
            ),
    }

    manifest = {
        "schema_version":
            "1.0",

        "status":
            "FINAL_DEVELOPMENT_REFIT_COMPLETE",

        "completed_at_utc":
            utc_now(),

        "model_id":
            model_id,

        "config_id":
            config_id,

        "seed":
            int(seed),

        "stage":
            "FINAL_DEVELOPMENT_REFIT",

        "scheduled_epochs":
            int(
                scheduled_epochs
            ),

        "epochs_run":
            int(
                scheduled_epochs
            ),

        "duration_rule":
            "MEDIAN_OF_5_SAME_SEED_ROLLING_FOLD_BEST_EPOCHS",

        "checkpoint_policy":
            "FINAL_SCHEDULED_EPOCH",

        "validation_used":
            False,

        "early_stopping_used":
            False,

        "logical_batch_size":
            LOGICAL_BATCH,

        "physical_microbatch_size":
            physical_microbatch,

        "partial_logical_batch_rule":
            "exact sample-weighted micro_n/logical_n; no drop/pad/duplicate",

        "learning_rate":
            learning_rate,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "optimizer":
            "AdamW",

        "adamw_betas":
            list(
                s26.BETAS
            ),

        "adamw_eps":
            s26.ADAMW_EPS,

        "loss":
            "BCEWithLogitsLoss(reduction='mean')",

        "dtype":
            "float32",

        "amp_enabled":
            False,

        "tf32_enabled":
            False,

        "scheduler":
            "none",

        "train_shuffle":
            (
                "numpy.random.SeedSequence("
                "[seed,0,epoch])"
            ),

        "final_development_partition_token":
            FINAL_DEVELOPMENT_PARTITION_TOKEN,

        "horizon_order":
            HORIZONS,

        "trainable_parameter_count":
            trainable_parameters,

        "total_parameter_count":
            total_parameters,

        "model_build_metadata":
            build_metadata,

        "source_repository_record":
            repository,

        "inputs": {
            "train_x": {
                "path":
                    str(
                        FINAL_X.resolve()
                    ),

                "sha256":
                    upstream[
                        "final_x_sha256"
                    ],

                "shape":
                    [
                        EXPECTED_FINAL_N,
                        LOOKBACK,
                        FEATURES,
                    ],

                "dtype":
                    "float32",
            },

            "train_y": {
                "path":
                    str(
                        FINAL_Y.resolve()
                    ),

                "sha256":
                    upstream[
                        "final_y_sha256"
                    ],

                "shape":
                    [
                        EXPECTED_FINAL_N,
                        OUTPUTS,
                    ],

                "dtype":
                    "int64",
            },
        },

        "artifacts":
            artifacts,

        "step30_global_stage2_fingerprint":
            EXPECTED_STEP30_GLOBAL_FINGERPRINT,

        "step32_global_fingerprint":
            EXPECTED_STEP32_GLOBAL_FINGERPRINT,

        "step33_global_fingerprint":
            EXPECTED_STEP33_GLOBAL_FINGERPRINT,

        "step33_schedule_sha256":
            EXPECTED_STEP33_SCHEDULE_SHA256,

        "step34_execution_source_sha256":
            source_sha,

        "step34_execution_source_fingerprint_sha256":
            source_fp,

        "step34_execution_plan_sha256":
            plan_sha,

        "step34_execution_plan_fingerprint_sha256":
            plan_fp,

        "step34_protocol_reference_sha256":
            protocol_reference_sha,

        "step34_protocol_reference_fingerprint_sha256":
            protocol_reference[
                "content_fingerprint_sha256"
            ],

        "protection_state": {
            "development_training_used":
                True,

            "development_validation_used":
                False,

            "architecture_selection_performed":
                False,

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

        "final_test_data_opened":
            False,
    }

    # Manifest MUST be written last.
    atomic_json(
        manifest_path,
        manifest,
    )

    manifest_sha = sha256_file(
        manifest_path
    )

    result = {
        "status":
            "PASS",

        "model_id":
            model_id,

        "config_id":
            config_id,

        "seed":
            int(seed),

        "epochs":
            int(
                scheduled_epochs
            ),

        "final_train_bce":
            float(
                final_train_bce
            ),

        "peak_cuda_allocated_mib":
            float(
                peak_allocated_bytes
                / (1024 ** 2)
            ),

        "wall_seconds":
            float(
                wall_seconds
            ),

        "fit_manifest_path":
            str(
                manifest_path.resolve()
            ),

        "fit_manifest_sha256":
            manifest_sha,
    }

    del optimizer
    del criterion
    del model
    del x_train
    del y_train

    gc.collect()

    torch.cuda.empty_cache()

    return result


# =============================================================================
# WORKER ENTRY POINT
# =============================================================================

def worker_main(
    model_id: str,
    config_id: str,
    seed: int,
) -> int:

    try:

        result = worker_fit(
            model_id=model_id,
            config_id=config_id,
            seed=seed,
        )

        print(
            RESULT_PREFIX
            + json.dumps(
                result,
                sort_keys=True,
            ),
            flush=True,
        )

        return 0

    except Exception as exc:

        import traceback

        traceback.print_exc()

        failure = {
            "status":
                "FAIL",

            "model_id":
                model_id,

            "config_id":
                config_id,

            "seed":
                int(seed),

            "error":
                repr(
                    exc
                ),
        }

        print(
            RESULT_PREFIX
            + json.dumps(
                failure,
                sort_keys=True,
            ),
            flush=True,
        )

        return 1


# =============================================================================
# FINAL GLOBAL FREEZE
# =============================================================================

def freeze_if_complete(
    plan: pd.DataFrame,
    source_sha: str,
    source_fp: str,
    protocol_reference: dict,
    protocol_reference_sha: str,
    plan_sha: str,
    plan_fp: str,
    upstream: dict,
) -> bool:

    fit_rows = []
    artifact_rows = []

    for r in plan.itertuples(
        index=False
    ):

        m = verify_completed_fit(
            model_id=str(
                r.model_id
            ),
            config_id=str(
                r.config_id
            ),
            seed=int(
                r.seed
            ),
            expected_epochs=int(
                r.final_refit_epochs
            ),
            source_fp=source_fp,
            protocol_reference=protocol_reference,
            upstream=upstream,
        )

        if m is None:
            return False

        manifest_path = fit_manifest_path(
            str(
                r.model_id
            ),
            str(
                r.config_id
            ),
            int(
                r.seed
            ),
        )

        metrics = load_json(
            Path(
                m[
                    "artifacts"
                ][
                    "fit_metrics.json"
                ][
                    "path"
                ]
            )
        )

        fit_rows.append(
            {
                "model_id":
                    str(
                        r.model_id
                    ),

                "config_id":
                    str(
                        r.config_id
                    ),

                "seed":
                    int(
                        r.seed
                    ),

                "scheduled_epochs":
                    int(
                        r.final_refit_epochs
                    ),

                "epochs_run":
                    int(
                        m[
                            "epochs_run"
                        ]
                    ),

                "learning_rate":
                    float(
                        m[
                            "learning_rate"
                        ]
                    ),

                "physical_microbatch_size":
                    int(
                        m[
                            "physical_microbatch_size"
                        ]
                    ),

                "final_train_mean_directional_bce":
                    float(
                        metrics[
                            "final_train_mean_directional_bce"
                        ]
                    ),

                "training_seconds":
                    float(
                        metrics[
                            "training_seconds"
                        ]
                    ),

                "wall_seconds":
                    float(
                        metrics[
                            "wall_seconds"
                        ]
                    ),

                "peak_cuda_allocated_mib":
                    float(
                        metrics[
                            "peak_cuda_allocated_mib"
                        ]
                    ),

                "fit_manifest_path":
                    str(
                        manifest_path.resolve()
                    ),

                "fit_manifest_sha256":
                    sha256_file(
                        manifest_path
                    ),
            }
        )

        for artifact_name, rec in (
            m[
                "artifacts"
            ].items()
        ):

            artifact_rows.append(
                {
                    "model_id":
                        str(
                            r.model_id
                        ),

                    "config_id":
                        str(
                            r.config_id
                        ),

                    "seed":
                        int(
                            r.seed
                        ),

                    "artifact_name":
                        artifact_name,

                    "artifact_path":
                        rec[
                            "path"
                        ],

                    "artifact_size_bytes":
                        int(
                            rec[
                                "size_bytes"
                            ]
                        ),

                    "artifact_sha256":
                        rec[
                            "sha256"
                        ],
                }
            )

    fit_df = pd.DataFrame(
        fit_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "seed",
        ]
    ).reset_index(
        drop=True
    )

    artifact_df = pd.DataFrame(
        artifact_rows
    ).sort_values(
        [
            "model_id",
            "config_id",
            "seed",
            "artifact_name",
        ]
    ).reset_index(
        drop=True
    )

    require_equal(
        len(fit_df),
        40,
        "Final-refit register rows",
    )

    require_equal(
        len(artifact_df),
        40 * 3,
        "Final-refit artifact rows",
    )

    atomic_csv(
        FIT_REGISTER_CSV,
        fit_df,
    )

    atomic_csv(
        ARTIFACT_REGISTER_CSV,
        artifact_df,
    )

    fit_register_sha = sha256_file(
        FIT_REGISTER_CSV
    )

    artifact_register_sha = sha256_file(
        ARTIFACT_REGISTER_CSV
    )

    artifact_fp = canonical_hash(
        artifact_df.to_dict(
            orient="records"
        )
    )

    freeze_basis = {
        "step30_global_fingerprint":
            EXPECTED_STEP30_GLOBAL_FINGERPRINT,

        "step32_global_fingerprint":
            EXPECTED_STEP32_GLOBAL_FINGERPRINT,

        "step33_global_fingerprint":
            EXPECTED_STEP33_GLOBAL_FINGERPRINT,

        "step34_source_sha256":
            source_sha,

        "step34_source_fingerprint":
            source_fp,

        "execution_plan_sha256":
            plan_sha,

        "execution_plan_fingerprint":
            plan_fp,

        "protocol_reference_sha256":
            protocol_reference_sha,

        "protocol_reference_fingerprint":
            protocol_reference[
                "content_fingerprint_sha256"
            ],

        "final_development_x_sha256":
            upstream[
                "final_x_sha256"
            ],

        "final_development_y_sha256":
            upstream[
                "final_y_sha256"
            ],

        "fit_register_sha256":
            fit_register_sha,

        "artifact_register_sha256":
            artifact_register_sha,

        "artifact_content_fingerprint":
            artifact_fp,

        "final_test_data_opened":
            False,
    }

    global_fp = canonical_hash(
        freeze_basis
    )

    final_manifest = {
        "status":
            "STEP34_FINAL_DEVELOPMENT_REFITS_FINAL_FREEZE_COMPLETE",

        "created_utc":
            utc_now(),

        "experiment_root":
            str(
                ROOT
            ),

        "execution_scope": {
            "architectures":
                8,

            "selected_configurations":
                SELECTED,

            "fixed_seeds":
                SEEDS,

            "total_refits":
                40,

            "completed_refits":
                40,

            "final_development_n":
                EXPECTED_FINAL_N,

            "total_scheduled_epochs":
                int(
                    fit_df[
                        "scheduled_epochs"
                    ].sum()
                ),
        },

        "training_contract": {
            "duration_rule":
                "MEDIAN_OF_5_SAME_SEED_ROLLING_FOLD_BEST_EPOCHS",

            "validation_during_refit":
                False,

            "early_stopping_during_refit":
                False,

            "checkpoint_policy":
                "FINAL_SCHEDULED_EPOCH",

            "logical_batch_size":
                LOGICAL_BATCH,

            "physical_microbatch_by_model":
                EXPECTED_PHYSICAL,

            "optimizer":
                "AdamW",

            "weight_decay":
                WEIGHT_DECAY,

            "loss":
                "BCEWithLogitsLoss(reduction='mean')",

            "gradient_clip_norm":
                GRADIENT_CLIP_NORM,

            "shuffle_seed_contract":
                "numpy.random.SeedSequence([seed,0,epoch])",

            "probability_ensemble_for_final_inference":
                "ARITHMETIC_MEAN_ACROSS_FIXED_SEEDS_42_46",
        },

        "upstream": {
            "step30_manifest_sha256":
                EXPECTED_STEP30_MANIFEST_SHA256,

            "step30_global_fingerprint":
                EXPECTED_STEP30_GLOBAL_FINGERPRINT,

            "step32_manifest_sha256":
                EXPECTED_STEP32_MANIFEST_SHA256,

            "step32_global_fingerprint":
                EXPECTED_STEP32_GLOBAL_FINGERPRINT,

            "step33_manifest_sha256":
                EXPECTED_STEP33_MANIFEST_SHA256,

            "step33_global_fingerprint":
                EXPECTED_STEP33_GLOBAL_FINGERPRINT,

            "step33_schedule_sha256":
                EXPECTED_STEP33_SCHEDULE_SHA256,

            "step33_schedule_fingerprint":
                EXPECTED_STEP33_SCHEDULE_FINGERPRINT,
        },

        "outputs": {
            "fit_register": {
                "path":
                    str(
                        FIT_REGISTER_CSV.resolve()
                    ),

                "rows":
                    40,

                "sha256":
                    fit_register_sha,
            },

            "artifact_register": {
                "path":
                    str(
                        ARTIFACT_REGISTER_CSV.resolve()
                    ),

                "rows":
                    120,

                "sha256":
                    artifact_register_sha,

                "content_fingerprint":
                    artifact_fp,
            },

            "execution_plan": {
                "path":
                    str(
                        EXECUTION_PLAN_CSV.resolve()
                    ),

                "sha256":
                    plan_sha,

                "content_fingerprint":
                    plan_fp,
            },

            "protocol_reference": {
                "path":
                    str(
                        PROTOCOL_REFERENCE_JSON.resolve()
                    ),

                "sha256":
                    protocol_reference_sha,

                "content_fingerprint":
                    protocol_reference[
                        "content_fingerprint_sha256"
                    ],
            },

            "frozen_source": {
                "path":
                    str(
                        FROZEN_SOURCE.resolve()
                    ),

                "sha256":
                    source_sha,

                "source_fingerprint":
                    source_fp,
            },
        },

        "protection_state": {
            "architecture_selection_performed":
                False,

            "best_seed_selection_performed":
                False,

            "best_seed_promotion_performed":
                False,

            "threshold_reoptimization_performed":
                False,

            "final_test_data_opened":
                False,

            "final_test_metric_computed":
                False,

            "final_test_state":
                "LOCKED_NOT_OPENED",
        },

        "global_step34_fingerprint":
            global_fp,

        "next_protocol_gate": {
            "allowed_next_step":
                "FINAL_TEST_INFERENCE_CONTRACT_PREPARATION",

            "final_test_allowed_immediately":
                False,

            "note":
                "Freeze the final-test materialization, inference, "
                "five-seed probability aggregation, threshold application, "
                "metric and statistical-inference contracts before opening "
                "the sealed final test.",
        },
    }

    atomic_json(
        FINAL_FREEZE_MANIFEST,
        final_manifest,
    )

    final_sha = sha256_file(
        FINAL_FREEZE_MANIFEST
    )

    print("=" * 140)
    print(
        "STEP 34 COMPLETE — FINAL-DEVELOPMENT "
        "REFITS FROZEN"
    )
    print("=" * 140)

    print(
        "Completed final refits       : 40/40"
    )

    print(
        "Total scheduled epochs       : "
        f"{int(fit_df['scheduled_epochs'].sum())}"
    )

    print(
        "Final-development sequences  : 42,360"
    )

    print(
        "Validation during refit      : NONE"
    )

    print(
        "Early stopping during refit  : NONE"
    )

    print(
        "Best-seed selection          : NOT PERFORMED"
    )

    print(
        "Final-test data opened       : False"
    )

    print(
        "Final-test state             : LOCKED_NOT_OPENED"
    )

    print(
        f"Global Step-34 fingerprint   : {global_fp}"
    )

    print(
        f"Freeze manifest SHA256       : {final_sha}"
    )

    print("=" * 140)

    return True


# =============================================================================
# MASTER EXECUTION
# =============================================================================

def master_main(
    args,
) -> int:

    print("=" * 140)
    print(
        "STEP 34 — FINAL-DEVELOPMENT "
        "REFIT EXECUTION"
    )
    print("=" * 140)

    print(
        f"Experiment root       : {ROOT}"
    )

    print(
        "Final-development n   : 42,360"
    )

    print(
        "Architectures         : 8"
    )

    print(
        "Fixed seeds           : 42,43,44,45,46"
    )

    print(
        "Total refits          : 40"
    )

    print(
        "Validation            : NONE"
    )

    print(
        "Early stopping        : NONE"
    )

    print(
        "Checkpoint            : FINAL SCHEDULED EPOCH"
    )

    print(
        "Final test            : LOCKED_NOT_OPENED"
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

    RUN_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Upstream verification.
    # -----------------------------------------------------------------

    upstream = verify_upstream()

    print("[1] UPSTREAM FREEZES")
    print("Step30 robustness freeze : VERIFIED")
    print("Step32 thresholds        : VERIFIED")
    print("Step33 refit schedule    : VERIFIED")
    print("Step26 training engine   : EXECUTABLE == FROZEN")
    print("Final-test state         : LOCKED_NOT_OPENED")
    print("PASS")
    print("-" * 140)

    # -----------------------------------------------------------------
    # Freeze source.
    # -----------------------------------------------------------------

    source_sha, source_fp = (
        ensure_source_freeze(
            upstream
        )
    )

    print("[2] STEP-34 SOURCE FREEZE")
    print(
        f"Source SHA256      : {source_sha}"
    )
    print(
        f"Source fingerprint : {source_fp}"
    )
    print("PASS")
    print("-" * 140)

    # -----------------------------------------------------------------
    # Protocol reference.
    # -----------------------------------------------------------------

    (
        protocol_reference,
        protocol_reference_sha,
    ) = ensure_protocol_reference(
        upstream
    )

    print("[3] FROZEN TRAINING PROTOCOL REFERENCE")
    print(
        "Source manifests      : 200 frozen robustness fits"
    )
    print(
        "Architectures verified: 8/8"
    )
    print(
        "Performance metrics   : NOT USED"
    )
    print(
        f"Reference SHA256      : {protocol_reference_sha}"
    )
    print("PASS")
    print("-" * 140)

    # -----------------------------------------------------------------
    # Execution plan.
    # -----------------------------------------------------------------

    plan, plan_sha, plan_fp = (
        ensure_execution_plan()
    )

    total_epochs = int(
        plan[
            "final_refit_epochs"
        ].sum()
    )

    require_equal(
        total_epochs,
        65,
        "Total frozen scheduled epochs",
    )

    print("[4] FINAL-REFIT EXECUTION PLAN")
    print(
        f"Plan rows             : {len(plan)}/40"
    )
    print(
        f"Total scheduled epochs: {total_epochs}"
    )
    print(
        f"Plan SHA256           : {plan_sha}"
    )
    print(
        f"Plan fingerprint      : {plan_fp}"
    )
    print("PASS")
    print("-" * 140)

    for r in plan.itertuples(
        index=False
    ):

        print(
            f"{r.model_id:14s} "
            f"seed={int(r.seed)} "
            f"epochs={int(r.final_refit_epochs):2d} "
            f"mb={int(r.physical_microbatch_size):3d}"
        )

    print("-" * 140)

    if args.prepare_only:

        print(
            "PREPARE-ONLY PASS — NO MODEL TRAINING PERFORMED."
        )

        print(
            "Final-test data opened : False"
        )

        print(
            "Final-test state       : LOCKED_NOT_OPENED"
        )

        print("=" * 140)

        return 0

    # -----------------------------------------------------------------
    # Optional filtering.
    # -----------------------------------------------------------------

    run_plan = plan.copy()

    if args.model is not None:

        model_upper = args.model.upper()

        if model_upper not in SELECTED:
            fail(
                f"Unknown model: {args.model}"
            )

        run_plan = run_plan.loc[
            run_plan[
                "model_id"
            ].astype(str)
            == model_upper
        ]

    if args.seed is not None:

        require_equal(
            int(
                args.seed
            ) in SEEDS,
            True,
            "Requested seed membership",
        )

        run_plan = run_plan.loc[
            run_plan[
                "seed"
            ].astype(int)
            == int(
                args.seed
            )
        ]

    new_fits = 0
    skipped_fits = 0

    # -----------------------------------------------------------------
    # Run fits in isolated subprocesses.
    # -----------------------------------------------------------------

    for r in run_plan.itertuples(
        index=False
    ):

        model_id = str(
            r.model_id
        )

        config_id = str(
            r.config_id
        )

        seed = int(
            r.seed
        )

        epochs = int(
            r.final_refit_epochs
        )

        prior = verify_completed_fit(
            model_id=model_id,
            config_id=config_id,
            seed=seed,
            expected_epochs=epochs,
            source_fp=source_fp,
            protocol_reference=protocol_reference,
            upstream=upstream,
        )

        if prior is not None:

            skipped_fits += 1

            print(
                f"SKIP "
                f"{model_id:<14} "
                f"{config_id:<20} "
                f"seed={seed} "
                f"COMPLETE+HASHED "
                f"epochs={epochs}",
                flush=True,
            )

            continue

        if (
            args.max_new_fits is not None
            and new_fits >= args.max_new_fits
        ):

            print("-" * 140)

            print(
                "Maximum requested new-fit count reached."
            )

            break

        archive_incomplete_fit(
            model_id,
            config_id,
            seed,
        )

        print(
            f"RUN  "
            f"{model_id:<14} "
            f"{config_id:<20} "
            f"seed={seed} "
            f"epochs={epochs}",
            flush=True,
        )

        cmd = [
            sys.executable,
            "-u",
            str(
                SCRIPT
            ),
            "--worker",
            model_id,
            config_id,
            str(
                seed
            ),
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(
                Q1_DIR
            ),
        )

        if proc.returncode != 0:

            fail(
                f"Worker failed: "
                f"{model_id}/{config_id}/"
                f"seed{seed}"
            )

        verified = verify_completed_fit(
            model_id=model_id,
            config_id=config_id,
            seed=seed,
            expected_epochs=epochs,
            source_fp=source_fp,
            protocol_reference=protocol_reference,
            upstream=upstream,
        )

        if verified is None:

            fail(
                "Worker returned success but canonical "
                "fit verification failed."
            )

        metrics = load_json(
            Path(
                verified[
                    "artifacts"
                ][
                    "fit_metrics.json"
                ][
                    "path"
                ]
            )
        )

        print(
            f"PASS "
            f"{model_id:<14} "
            f"seed={seed} "
            f"epochs={epochs} "
            f"trainBCE="
            f"{metrics['final_train_mean_directional_bce']:.6f} "
            f"peak="
            f"{metrics['peak_cuda_allocated_mib']:.1f}MiB",
            flush=True,
        )

        new_fits += 1

    print("-" * 140)

    # -----------------------------------------------------------------
    # Reconcile full execution state.
    # -----------------------------------------------------------------

    completed = 0

    for r in plan.itertuples(
        index=False
    ):

        m = verify_completed_fit(
            model_id=str(
                r.model_id
            ),
            config_id=str(
                r.config_id
            ),
            seed=int(
                r.seed
            ),
            expected_epochs=int(
                r.final_refit_epochs
            ),
            source_fp=source_fp,
            protocol_reference=protocol_reference,
            upstream=upstream,
        )

        if m is not None:
            completed += 1

    print(
        f"New fits this invocation : {new_fits}"
    )

    print(
        f"Hash-valid skipped fits  : {skipped_fits}"
    )

    print(
        f"Total completed refits   : {completed}/40"
    )

    print(
        "Final-test data opened  : False"
    )

    print(
        "Final-test state        : LOCKED_NOT_OPENED"
    )

    print("-" * 140)

    if completed == 40:

        freeze_if_complete(
            plan=plan,
            source_sha=source_sha,
            source_fp=source_fp,
            protocol_reference=protocol_reference,
            protocol_reference_sha=protocol_reference_sha,
            plan_sha=plan_sha,
            plan_fp=plan_fp,
            upstream=upstream,
        )

    else:

        print(
            "STEP 34 remains resumable/incomplete."
        )

        print(
            "Final global freeze will be written only "
            "after all 40 refits are hash-valid."
        )

        print("=" * 140)

    return 0


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Freeze/verify Step-34 source, protocol reference "
            "and execution plan without training."
        ),
    )

    parser.add_argument(
        "--max-new-fits",
        type=int,
        default=None,
        help=(
            "Maximum number of new refits to execute "
            "during this invocation."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional architecture filter."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional fixed-seed filter."
        ),
    )

    parser.add_argument(
        "--worker",
        nargs=3,
        metavar=(
            "MODEL_ID",
            "CONFIG_ID",
            "SEED",
        ),
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    if args.worker is not None:

        model_id = str(
            args.worker[0]
        )

        config_id = str(
            args.worker[1]
        )

        seed = int(
            args.worker[2]
        )

        return worker_main(
            model_id=model_id,
            config_id=config_id,
            seed=seed,
        )

    return master_main(
        args
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )