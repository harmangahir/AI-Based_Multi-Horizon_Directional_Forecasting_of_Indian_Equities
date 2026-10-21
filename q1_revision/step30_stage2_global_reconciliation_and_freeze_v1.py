#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 30 — STAGE-2 GLOBAL RECONCILIATION AND FINAL ROBUSTNESS FREEZE
==================================================================

Purpose
-------
Formally reconcile and freeze the complete Stage-2 robustness experiment:

    8 architectures
    × 5 seeds (42–46)
    × 5 rolling-origin folds
    = 200 logical fits

Composition
-----------
    40 seed-42 fits : immutable Stage-1 hash-verified reuse
    160 new fits    : Stage-2 robustness fits, seeds 43–46

This script:
    - verifies the frozen Step-29A logical plan;
    - verifies the 40 Stage-1 seed-42 reuse registrations by SHA-256;
    - verifies all 160 Stage-2 fit manifests and canonical artifacts;
    - verifies selected configuration, seed, fold and batch contracts;
    - verifies no best-seed promotion;
    - verifies no threshold optimization;
    - verifies no final-test use;
    - builds an authoritative 200-row logical-fit inventory;
    - freezes the Stage-2 source and global inventory;
    - creates an immutable Stage-2 final-freeze manifest.

This script DOES NOT:
    - choose a best seed;
    - optimize thresholds;
    - aggregate five-seed probabilities;
    - refit on full development data;
    - open final-test data;
    - calculate final-test metrics.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


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

PLAN_PATH = AUDIT_DIR / "29a_robustness_logical_fit_plan.csv"
REUSE_PATH = AUDIT_DIR / "29a_seed42_reuse_register.csv"

OUT_LOGICAL_INVENTORY = AUDIT_DIR / "30_stage2_global_logical_fit_inventory.csv"
OUT_NEW_ARTIFACTS = AUDIT_DIR / "30_stage2_new_fit_artifact_inventory.csv"
OUT_FREEZE_MANIFEST = MANIFEST_DIR / "30_stage2_global_final_freeze_manifest.json"

FROZEN_SOURCE = (
    MANIFEST_DIR
    / "step30_stage2_global_reconciliation_and_freeze_v1_frozen.py"
)


# =============================================================================
# FROZEN PROTOCOL CONTRACT
# =============================================================================

EXPECTED_SELECTED = {
    "DLINEAR": "DLINEAR_B",
    "NLINEAR": "NLINEAR_B",
    "PATCHTST": "PATCHTST_B",
    "TIMESNET": "TIMESNET_B",
    "ITRANSFORMER": "ITRANSFORMER_A",
    "FREDFORMER": "FREDFORMER_B",
    "FRNET": "FRNET_A",
    "MLF": "MLF_A",
}

EXPECTED_MICROBATCH = {
    "DLINEAR": 128,
    "NLINEAR": 128,
    "PATCHTST": 64,
    "TIMESNET": 64,       # Step-27F amendment
    "ITRANSFORMER": 128,
    "FREDFORMER": 16,
    "FRNET": 64,
    "MLF": 32,
}

EXPECTED_SEEDS = [42, 43, 44, 45, 46]
EXPECTED_FOLDS = [1, 2, 3, 4, 5]

EXPECTED_LOGICAL_BATCH = 256
EXPECTED_SCREENING_THRESHOLD = 0.5

EXPECTED_LOGICAL_FITS = 200
EXPECTED_STAGE1_REUSES = 40
EXPECTED_STAGE2_NEW = 160

TIMESNET_B_AMENDMENT_SHA256 = (
    "04c30b59b8bef7327b44e1764240ead258edbba64f278e31ca29249141dfdce8"
)

CANONICAL_NEW_FILES = [
    "best_checkpoint.pt",
    "epoch_history.csv",
    "validation_predictions.npz",
    "fit_metrics.json",
    "fit_manifest.json",
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


def canonical_json_sha256(obj) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        fail("Boolean field is NaN.")

    s = str(x).strip().lower()

    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False

    fail(f"Cannot parse boolean value: {x!r}")


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{label}: missing required columns: {missing}")


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, got {actual!r}")


def load_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"Missing JSON file: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        fail(f"Expected JSON object: {path}")

    return obj


def protect_false(ps: dict, key: str, context: str) -> None:
    if key not in ps:
        fail(f"{context}: protection_state missing key {key!r}")
    if ps[key] is not False:
        fail(f"{context}: {key} must be False, got {ps[key]!r}")


def normalize_path(x) -> Path:
    return Path(str(x)).expanduser().resolve()


def freeze_source() -> tuple[str, str]:
    src = Path(__file__).resolve()
    src_sha = sha256_file(src)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    if FROZEN_SOURCE.exists():
        existing_sha = sha256_file(FROZEN_SOURCE)
        if existing_sha != src_sha:
            fail(
                "Frozen Step-30 source already exists but differs from the "
                "current source.\n"
                f"Existing : {existing_sha}\n"
                f"Current  : {src_sha}"
            )
    else:
        shutil.copy2(src, FROZEN_SOURCE)

    frozen_sha = sha256_file(FROZEN_SOURCE)

    if frozen_sha != src_sha:
        fail("Frozen Step-30 source SHA does not match executable source SHA.")

    return src_sha, frozen_sha


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 140)
    print("STEP 30 — STAGE-2 GLOBAL RECONCILIATION AND FINAL ROBUSTNESS FREEZE")
    print("=" * 140)
    print(f"Experiment root : {ROOT}")
    print("Final test      : LOCKED_NOT_OPENED")
    print("-" * 140)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 0. Refuse silent replacement of an existing final freeze.
    # -------------------------------------------------------------------------

    if OUT_FREEZE_MANIFEST.exists():
        print("A Step-30 final freeze manifest already exists:")
        print(OUT_FREEZE_MANIFEST)
        print()
        print("No files were modified.")
        print("Delete/replace a frozen manifest only after explicit protocol review.")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # 1. Freeze Step-30 source.
    # -------------------------------------------------------------------------

    source_sha, frozen_source_sha = freeze_source()

    print("[1] STEP-30 SOURCE FREEZE")
    print(f"Executable source SHA256 : {source_sha}")
    print(f"Frozen source SHA256     : {frozen_source_sha}")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 2. Load authoritative Step-29A registers.
    # -------------------------------------------------------------------------

    if not PLAN_PATH.is_file():
        fail(f"Missing logical fit plan: {PLAN_PATH}")

    if not REUSE_PATH.is_file():
        fail(f"Missing seed42 reuse register: {REUSE_PATH}")

    plan_sha = sha256_file(PLAN_PATH)
    reuse_sha = sha256_file(REUSE_PATH)

    plan = pd.read_csv(PLAN_PATH)
    reuse = pd.read_csv(REUSE_PATH)

    require_columns(
        plan,
        [
            "model_id",
            "config_id",
            "seed",
            "fold",
            "artifact_origin",
            "physical_microbatch_size",
            "logical_batch_size",
            "screening_threshold",
            "best_seed_promotion",
            "test_used",
        ],
        "Step-29A logical plan",
    )

    require_columns(
        reuse,
        [
            "model_id",
            "config_id",
            "fold",
            "seed",
            "reuse_status",
            "stage1_fit_manifest_path",
            "stage1_fit_manifest_sha256",
            "training_protocol_config_data_contract_sha256",
            "validation_predictions_path",
            "validation_predictions_sha256",
        ],
        "Step-29A seed42 reuse register",
    )

    require_equal(len(plan), EXPECTED_LOGICAL_FITS, "Logical plan row count")
    require_equal(len(reuse), EXPECTED_STAGE1_REUSES, "Seed42 reuse row count")

    print("[2] STEP-29A AUTHORITATIVE REGISTERS")
    print(f"Logical plan rows       : {len(plan)}/200")
    print(f"Seed42 reuse rows       : {len(reuse)}/40")
    print(f"Logical plan SHA256     : {plan_sha}")
    print(f"Seed42 register SHA256  : {reuse_sha}")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 3. Validate global plan structure.
    # -------------------------------------------------------------------------

    key_cols = ["model_id", "config_id", "seed", "fold"]

    if plan.duplicated(key_cols).any():
        dup = plan.loc[plan.duplicated(key_cols, keep=False), key_cols]
        fail(
            "Duplicate logical-fit keys detected:\n"
            + dup.to_string(index=False)
        )

    require_equal(
        set(plan["model_id"].astype(str)),
        set(EXPECTED_SELECTED),
        "Model set",
    )

    for model_id, config_id in EXPECTED_SELECTED.items():
        m = plan.loc[plan["model_id"] == model_id].copy()

        require_equal(len(m), 25, f"{model_id}: logical rows")
        require_equal(
            set(m["config_id"].astype(str)),
            {config_id},
            f"{model_id}: selected config",
        )
        require_equal(
            set(m["seed"].astype(int)),
            set(EXPECTED_SEEDS),
            f"{model_id}: seed set",
        )
        require_equal(
            set(m["fold"].astype(int)),
            set(EXPECTED_FOLDS),
            f"{model_id}: fold set",
        )

        combos = {
            (int(r.seed), int(r.fold))
            for r in m.itertuples(index=False)
        }

        expected_combos = {
            (seed, fold)
            for seed in EXPECTED_SEEDS
            for fold in EXPECTED_FOLDS
        }

        require_equal(
            combos,
            expected_combos,
            f"{model_id}: seed × fold Cartesian product",
        )

        expected_mb = EXPECTED_MICROBATCH[model_id]

        require_equal(
            set(m["physical_microbatch_size"].astype(int)),
            {expected_mb},
            f"{model_id}: physical microbatch",
        )

        require_equal(
            set(m["logical_batch_size"].astype(int)),
            {EXPECTED_LOGICAL_BATCH},
            f"{model_id}: logical batch",
        )

    for r in plan.itertuples(index=False):

        if abs(float(r.screening_threshold) - 0.5) > 1e-12:
            fail(
                f"{r.model_id}/{r.config_id}/seed{r.seed}/fold{r.fold}: "
                f"screening threshold is not 0.5"
            )

        if as_bool(r.best_seed_promotion):
            fail(
                f"{r.model_id}/{r.config_id}/seed{r.seed}/fold{r.fold}: "
                "best-seed promotion unexpectedly True"
            )

        if as_bool(r.test_used):
            fail(
                f"{r.model_id}/{r.config_id}/seed{r.seed}/fold{r.fold}: "
                "test_used unexpectedly True"
            )

        if int(r.seed) == 42:
            if str(r.artifact_origin) != "STAGE1_REUSE":
                fail(
                    f"Seed42 row is not STAGE1_REUSE: "
                    f"{r.model_id}/{r.config_id}/fold{r.fold}"
                )
        else:
            if str(r.artifact_origin) == "STAGE1_REUSE":
                fail(
                    f"Non-seed42 row incorrectly marked STAGE1_REUSE: "
                    f"{r.model_id}/{r.config_id}/seed{r.seed}/fold{r.fold}"
                )

    seed42_count = int((plan["seed"].astype(int) == 42).sum())
    new_count = int((plan["seed"].astype(int) != 42).sum())

    require_equal(seed42_count, EXPECTED_STAGE1_REUSES, "Seed42 plan rows")
    require_equal(new_count, EXPECTED_STAGE2_NEW, "New Stage2 plan rows")

    print("[3] GLOBAL LOGICAL PLAN CONTRACT")
    print("Architectures              : 8/8")
    print("Selected configs           : 8/8")
    print("Seeds                      : 42,43,44,45,46")
    print("Folds per seed             : 1,2,3,4,5")
    print("Logical fits               : 200/200")
    print("Seed42 Stage-1 reuse       : 40")
    print("New Stage-2 fits           : 160")
    print("Screening threshold        : 0.5")
    print("Best-seed promotion        : False")
    print("Test used                  : False")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 4. Validate 40 immutable Stage-1 seed42 reuses.
    # -------------------------------------------------------------------------

    if reuse.duplicated(["model_id", "config_id", "seed", "fold"]).any():
        fail("Duplicate rows in seed42 reuse register.")

    require_equal(
        set(reuse["seed"].astype(int)),
        {42},
        "Reuse register seed set",
    )

    stage1_rows = []

    for r in reuse.itertuples(index=False):

        model_id = str(r.model_id)
        config_id = str(r.config_id)
        seed = int(r.seed)
        fold = int(r.fold)

        context = f"{model_id}/{config_id}/seed={seed}/fold={fold}"

        if model_id not in EXPECTED_SELECTED:
            fail(f"{context}: unexpected model.")

        require_equal(
            config_id,
            EXPECTED_SELECTED[model_id],
            f"{context}: config",
        )

        require_equal(
            str(r.reuse_status),
            "HASH_VERIFIED_STAGE1_REUSE",
            f"{context}: reuse_status",
        )

        manifest_path = normalize_path(r.stage1_fit_manifest_path)
        pred_path = normalize_path(r.validation_predictions_path)

        actual_manifest_sha = sha256_file(manifest_path)
        actual_pred_sha = sha256_file(pred_path)

        require_equal(
            actual_manifest_sha,
            str(r.stage1_fit_manifest_sha256),
            f"{context}: Stage1 fit manifest SHA256",
        )

        require_equal(
            actual_pred_sha,
            str(r.validation_predictions_sha256),
            f"{context}: Stage1 validation predictions SHA256",
        )

        # Confirm this logical row exists in the 200-row plan.
        p = plan.loc[
            (plan["model_id"] == model_id)
            & (plan["config_id"] == config_id)
            & (plan["seed"].astype(int) == seed)
            & (plan["fold"].astype(int) == fold)
        ]

        require_equal(len(p), 1, f"{context}: matching plan row count")

        pr = p.iloc[0]

        stage1_rows.append(
            {
                "model_id": model_id,
                "config_id": config_id,
                "seed": seed,
                "fold": fold,
                "artifact_origin": "STAGE1_REUSE",
                "physical_microbatch_size": int(
                    pr["physical_microbatch_size"]
                ),
                "logical_batch_size": int(pr["logical_batch_size"]),
                "screening_threshold": float(
                    pr["screening_threshold"]
                ),
                "fit_manifest_path": str(manifest_path),
                "fit_manifest_sha256": actual_manifest_sha,
                "validation_predictions_path": str(pred_path),
                "validation_predictions_sha256": actual_pred_sha,
                "training_protocol_config_data_contract_sha256":
                    str(
                        r.training_protocol_config_data_contract_sha256
                    ),
                "best_seed_promotion": False,
                "threshold_optimization_performed": False,
                "final_test_data_opened": False,
                "final_test_metric_computed": False,
                "reconciliation_status": "HASH_VERIFIED_STAGE1_REUSE",
            }
        )

    require_equal(
        len(stage1_rows),
        EXPECTED_STAGE1_REUSES,
        "Verified Stage1 reuse count",
    )

    print("[4] IMMUTABLE SEED42 STAGE-1 REUSES")
    print(f"Hash-verified reuse fits : {len(stage1_rows)}/40")
    print("Manifest hashes          : VERIFIED")
    print("Prediction hashes        : VERIFIED")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 5. Validate all 160 new Stage-2 fits and canonical artifacts.
    # -------------------------------------------------------------------------

    stage2_rows = []
    artifact_rows = []

    new_plan = plan.loc[plan["seed"].astype(int) != 42].copy()

    for pr in new_plan.itertuples(index=False):

        model_id = str(pr.model_id)
        config_id = str(pr.config_id)
        seed = int(pr.seed)
        fold = int(pr.fold)

        context = f"{model_id}/{config_id}/seed={seed}/fold={fold}"

        fit_dir = (
            ROOT
            / "stage2_robustness_fits"
            / model_id
            / config_id
            / f"seed_{seed:02d}"
            / f"fold_{fold:02d}"
        )

        manifest_path = fit_dir / "fit_manifest.json"

        if not manifest_path.is_file():
            fail(f"{context}: missing fit_manifest.json at {manifest_path}")

        m = load_json(manifest_path)
        ps = m.get("protection_state")

        if not isinstance(ps, dict):
            fail(f"{context}: missing/invalid protection_state.")

        require_equal(
            m.get("status"),
            "STAGE2_ROBUSTNESS_FIT_COMPLETE",
            f"{context}: status",
        )

        require_equal(
            m.get("model_id"),
            model_id,
            f"{context}: model_id",
        )

        require_equal(
            m.get("config_id"),
            config_id,
            f"{context}: config_id",
        )

        require_equal(
            int(m.get("seed", -1)),
            seed,
            f"{context}: seed",
        )

        require_equal(
            int(m.get("robustness_seed", -1)),
            seed,
            f"{context}: robustness_seed",
        )

        require_equal(
            int(m.get("fold", -1)),
            fold,
            f"{context}: fold",
        )

        expected_mb = int(pr.physical_microbatch_size)

        require_equal(
            int(m.get("physical_microbatch_size", -1)),
            expected_mb,
            f"{context}: physical_microbatch_size",
        )

        require_equal(
            int(m.get("logical_batch_size", -1)),
            EXPECTED_LOGICAL_BATCH,
            f"{context}: logical_batch_size",
        )

        if abs(float(m.get("threshold", -999)) - 0.5) > 1e-12:
            fail(f"{context}: manifest threshold is not 0.5.")

        protect_false(
            ps,
            "best_seed_selection_performed",
            context,
        )
        protect_false(
            ps,
            "threshold_optimization_performed",
            context,
        )
        protect_false(
            ps,
            "final_test_data_opened",
            context,
        )
        protect_false(
            ps,
            "final_test_metric_computed",
            context,
        )

        # Special TimesNet-B amendment check.
        if model_id == "TIMESNET":
            require_equal(
                m.get("timesnet_b_protocol_amendment_sha256"),
                TIMESNET_B_AMENDMENT_SHA256,
                f"{context}: TimesNet-B amendment SHA256",
            )

        # Canonical Stage-2 artifact integrity.
        file_hashes = {}

        for filename in CANONICAL_NEW_FILES:
            p = fit_dir / filename
            if not p.is_file():
                fail(f"{context}: missing canonical artifact {filename}")
            file_hashes[filename] = sha256_file(p)

        manifest_sha = file_hashes["fit_manifest.json"]
        pred_sha = file_hashes["validation_predictions.npz"]

        stage2_rows.append(
            {
                "model_id": model_id,
                "config_id": config_id,
                "seed": seed,
                "fold": fold,
                "artifact_origin": str(pr.artifact_origin),
                "physical_microbatch_size": expected_mb,
                "logical_batch_size": EXPECTED_LOGICAL_BATCH,
                "screening_threshold": 0.5,
                "fit_manifest_path": str(manifest_path.resolve()),
                "fit_manifest_sha256": manifest_sha,
                "validation_predictions_path":
                    str(
                        (
                            fit_dir / "validation_predictions.npz"
                        ).resolve()
                    ),
                "validation_predictions_sha256": pred_sha,
                "training_protocol_config_data_contract_sha256":
                    m.get(
                        "training_protocol_config_data_contract_sha256"
                    ),
                "best_seed_promotion": False,
                "threshold_optimization_performed": False,
                "final_test_data_opened": False,
                "final_test_metric_computed": False,
                "reconciliation_status":
                    "HASH_VERIFIED_STAGE2_ROBUSTNESS_FIT",
            }
        )

        artifact_rows.append(
            {
                "model_id": model_id,
                "config_id": config_id,
                "seed": seed,
                "fold": fold,
                "fit_dir": str(fit_dir.resolve()),
                "best_checkpoint_sha256":
                    file_hashes["best_checkpoint.pt"],
                "epoch_history_sha256":
                    file_hashes["epoch_history.csv"],
                "validation_predictions_sha256":
                    file_hashes["validation_predictions.npz"],
                "fit_metrics_sha256":
                    file_hashes["fit_metrics.json"],
                "fit_manifest_sha256":
                    file_hashes["fit_manifest.json"],
            }
        )

    require_equal(
        len(stage2_rows),
        EXPECTED_STAGE2_NEW,
        "Verified Stage2 new-fit count",
    )

    print("[5] NEW STAGE-2 ROBUSTNESS FITS")
    print(f"Verified new fits         : {len(stage2_rows)}/160")
    print("Canonical artifacts/fit   : 5")
    print("Manifest protection state : VERIFIED")
    print("TimesNet amendment         : VERIFIED")
    print("Best-seed promotion        : NOT PERFORMED")
    print("Threshold optimization     : NOT PERFORMED")
    print("Final-test data opened     : False")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 6. Build authoritative 200-row logical inventory.
    # -------------------------------------------------------------------------

    logical_inventory = pd.DataFrame(stage1_rows + stage2_rows)

    logical_inventory = logical_inventory.sort_values(
        ["model_id", "config_id", "seed", "fold"]
    ).reset_index(drop=True)

    require_equal(
        len(logical_inventory),
        EXPECTED_LOGICAL_FITS,
        "Final logical inventory count",
    )

    if logical_inventory.duplicated(key_cols).any():
        fail("Duplicate keys in final logical inventory.")

    inv_keys = {
        (
            str(r.model_id),
            str(r.config_id),
            int(r.seed),
            int(r.fold),
        )
        for r in logical_inventory.itertuples(index=False)
    }

    plan_keys = {
        (
            str(r.model_id),
            str(r.config_id),
            int(r.seed),
            int(r.fold),
        )
        for r in plan.itertuples(index=False)
    }

    require_equal(
        inv_keys,
        plan_keys,
        "Final inventory key set versus Step29A plan",
    )

    logical_inventory.to_csv(
        OUT_LOGICAL_INVENTORY,
        index=False,
    )

    artifact_inventory = pd.DataFrame(artifact_rows).sort_values(
        ["model_id", "config_id", "seed", "fold"]
    ).reset_index(drop=True)

    require_equal(
        len(artifact_inventory),
        EXPECTED_STAGE2_NEW,
        "New-fit artifact inventory count",
    )

    artifact_inventory.to_csv(
        OUT_NEW_ARTIFACTS,
        index=False,
    )

    logical_inventory_sha = sha256_file(OUT_LOGICAL_INVENTORY)
    artifact_inventory_sha = sha256_file(OUT_NEW_ARTIFACTS)

    logical_records = logical_inventory.to_dict(orient="records")
    artifact_records = artifact_inventory.to_dict(orient="records")

    logical_fingerprint = canonical_json_sha256(logical_records)
    artifact_fingerprint = canonical_json_sha256(artifact_records)

    print("[6] AUTHORITATIVE STAGE-2 INVENTORIES")
    print(f"Logical inventory rows        : {len(logical_inventory)}/200")
    print(f"New artifact inventory rows   : {len(artifact_inventory)}/160")
    print(f"Logical inventory SHA256      : {logical_inventory_sha}")
    print(f"Artifact inventory SHA256     : {artifact_inventory_sha}")
    print(f"Logical content fingerprint   : {logical_fingerprint}")
    print(f"Artifact content fingerprint  : {artifact_fingerprint}")
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # 7. Architecture-level reconciliation summary.
    # -------------------------------------------------------------------------

    architecture_summary = {}

    for model_id, config_id in EXPECTED_SELECTED.items():

        x = logical_inventory.loc[
            logical_inventory["model_id"] == model_id
        ]

        require_equal(
            len(x),
            25,
            f"{model_id}: reconciled logical count",
        )

        origins = (
            x["artifact_origin"]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        architecture_summary[model_id] = {
            "config_id": config_id,
            "logical_fits": int(len(x)),
            "seed42_reused_fits":
                int((x["seed"].astype(int) == 42).sum()),
            "new_stage2_fits":
                int((x["seed"].astype(int) != 42).sum()),
            "physical_microbatch_size":
                int(EXPECTED_MICROBATCH[model_id]),
            "logical_batch_size":
                EXPECTED_LOGICAL_BATCH,
            "artifact_origins": origins,
        }

    # -------------------------------------------------------------------------
    # 8. Global freeze fingerprint.
    # -------------------------------------------------------------------------

    freeze_basis = {
        "step29a_logical_plan_sha256": plan_sha,
        "step29a_seed42_reuse_register_sha256": reuse_sha,
        "stage2_logical_inventory_sha256": logical_inventory_sha,
        "stage2_new_artifact_inventory_sha256": artifact_inventory_sha,
        "stage2_logical_content_fingerprint": logical_fingerprint,
        "stage2_artifact_content_fingerprint": artifact_fingerprint,
        "step30_source_sha256": source_sha,
        "step30_frozen_source_sha256": frozen_source_sha,
        "logical_fit_count": 200,
        "stage1_reuse_count": 40,
        "new_stage2_fit_count": 160,
        "models": EXPECTED_SELECTED,
        "seeds": EXPECTED_SEEDS,
        "folds": EXPECTED_FOLDS,
        "best_seed_promotion_performed": False,
        "threshold_optimization_performed": False,
        "final_test_data_opened": False,
        "final_test_metric_computed": False,
    }

    stage2_global_freeze_fingerprint = canonical_json_sha256(
        freeze_basis
    )

    # -------------------------------------------------------------------------
    # 9. Write final freeze manifest LAST.
    # -------------------------------------------------------------------------

    freeze_manifest = {
        "status": "STAGE2_GLOBAL_ROBUSTNESS_FINAL_FREEZE_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(ROOT),

        "protocol": {
            "architectures": 8,
            "selected_configs": EXPECTED_SELECTED,
            "seeds": EXPECTED_SEEDS,
            "folds": EXPECTED_FOLDS,
            "logical_fit_count": 200,
            "seed42_stage1_reuses": 40,
            "new_stage2_robustness_fits": 160,
            "screening_threshold": 0.5,
            "logical_batch_size": 256,
            "physical_microbatch_by_model": EXPECTED_MICROBATCH,
        },

        "reconciliation": {
            "logical_fits_verified": 200,
            "stage1_reuses_hash_verified": 40,
            "new_stage2_fits_hash_verified": 160,
            "invalid_or_missing": 0,
            "architecture_summary": architecture_summary,
        },

        "protection_state": {
            "best_seed_selection_performed": False,
            "best_seed_promotion_performed": False,
            "threshold_optimization_performed": False,
            "final_test_data_opened": False,
            "final_test_metric_computed": False,
            "final_test_state": "LOCKED_NOT_OPENED",
        },

        "timesnet_b_amendment": {
            "physical_microbatch_size": 64,
            "logical_batch_size": 256,
            "protocol_amendment_sha256":
                TIMESNET_B_AMENDMENT_SHA256,
            "verified": True,
        },

        "inputs": {
            "step29a_logical_fit_plan": {
                "path": str(PLAN_PATH),
                "sha256": plan_sha,
            },
            "step29a_seed42_reuse_register": {
                "path": str(REUSE_PATH),
                "sha256": reuse_sha,
            },
        },

        "outputs": {
            "logical_fit_inventory": {
                "path": str(OUT_LOGICAL_INVENTORY),
                "rows": 200,
                "sha256": logical_inventory_sha,
                "content_fingerprint": logical_fingerprint,
            },
            "new_fit_artifact_inventory": {
                "path": str(OUT_NEW_ARTIFACTS),
                "rows": 160,
                "sha256": artifact_inventory_sha,
                "content_fingerprint": artifact_fingerprint,
            },
            "frozen_source": {
                "path": str(FROZEN_SOURCE),
                "sha256": frozen_source_sha,
            },
        },

        "global_stage2_freeze_fingerprint":
            stage2_global_freeze_fingerprint,

        "next_protocol_gate": {
            "allowed_next_step":
                "DEVELOPMENT_ONLY_FIVE_SEED_OOF_PROBABILITY_AGGREGATION",
            "threshold_optimization_allowed_now": False,
            "final_test_allowed_now": False,
            "note":
                "Five-seed OOF probability aggregation must be "
                "constructed and verified before development-only "
                "threshold optimization.",
        },
    }

    with OUT_FREEZE_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(
            freeze_manifest,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    freeze_manifest_sha = sha256_file(OUT_FREEZE_MANIFEST)

    # -------------------------------------------------------------------------
    # 10. Final report.
    # -------------------------------------------------------------------------

    print("[7] FINAL STAGE-2 FREEZE")
    print(
        f"Global freeze fingerprint : "
        f"{stage2_global_freeze_fingerprint}"
    )
    print(
        f"Freeze manifest SHA256    : "
        f"{freeze_manifest_sha}"
    )
    print()

    print("=" * 140)
    print(
        "STEP 30 COMPLETE — STAGE-2 GLOBAL ROBUSTNESS "
        "FINAL FREEZE ESTABLISHED"
    )
    print("=" * 140)
    print("Logical robustness fits verified : 200/200")
    print("Seed42 Stage-1 reuses verified   : 40/40")
    print("New Stage-2 fits verified        : 160/160")
    print("Architectures frozen             : 8/8")
    print("Best-seed promotion              : NOT PERFORMED")
    print("Threshold optimization           : NOT PERFORMED")
    print("Final-test data opened           : False")
    print("Final-test state                 : LOCKED_NOT_OPENED")
    print("-" * 140)
    print(
        "NEXT GATE: development-only five-seed OOF "
        "probability aggregation."
    )
    print(
        "DO NOT optimize thresholds or open the final test yet."
    )
    print("=" * 140)


if __name__ == "__main__":
    main()