#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 31A — THRESHOLD-OPERATOR COMPATIBILITY RUNNER
===================================================

Purpose
-------
Execute the frozen Step-31 five-seed development OOF aggregation while
correcting ONLY the representation-level threshold-operator parser.

Observed frozen-manifest representation:
    "probability >= 0.50"

Frozen Step-31 v1 originally accepted only compact operator forms such
as ">=" and therefore failed before OOF aggregation began.

This compatibility runner:
    - preserves the failed Step-31 v1 source unchanged;
    - verifies its exact frozen SHA-256;
    - patches ONLY threshold-operator interpretation;
    - does not alter threshold value (0.50);
    - does not alter predictions, labels, seeds, folds or horizons;
    - does not optimize thresholds;
    - does not select/promote a seed;
    - does not open final-test data;
    - freezes this compatibility runner as the authoritative executable
      source for the successful Step-31 endpoint.

Final test remains LOCKED_NOT_OPENED.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import step31_stage2_five_seed_oof_aggregation_v1 as s31


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

MANIFEST_DIR = ROOT / "manifests"

ORIGINAL_STEP31_SOURCE = (
    HERE / "step31_stage2_five_seed_oof_aggregation_v1.py"
).resolve()

ORIGINAL_STEP31_FROZEN_SOURCE = (
    MANIFEST_DIR
    / "step31_stage2_five_seed_oof_aggregation_v1_frozen.py"
)

ORIGINAL_STEP31_FINAL_MANIFEST = (
    MANIFEST_DIR
    / "31_stage2_five_seed_oof_final_freeze_manifest.json"
)

STEP31A_FROZEN_SOURCE = (
    MANIFEST_DIR
    / "step31a_stage2_oof_threshold_operator_compatibility_runner_v1_frozen.py"
)

STEP31A_AMENDMENT_MANIFEST = (
    MANIFEST_DIR
    / "31a_threshold_operator_compatibility_amendment.json"
)

STEP31A_FINAL_MANIFEST = (
    MANIFEST_DIR
    / "31a_stage2_five_seed_oof_final_freeze_manifest.json"
)


# =============================================================================
# IMMUTABLE FAILED STEP-31 V1 RECORD
# =============================================================================

EXPECTED_STEP31_V1_SHA256 = (
    "63f4fbd844037c71feff4e9ea47ee35a2afed0f23e5a197befa5390b11d34599"
)

OBSERVED_FAILURE = (
    "Unsupported threshold operator: 'probability >= 0.50'"
)


# =============================================================================
# HELPERS
# =============================================================================

def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if not path.is_file():
        fail(f"Missing file: {path}")

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


# =============================================================================
# EXACT COMPATIBILITY PATCH
# =============================================================================

ORIGINAL_NORMALIZER = s31.normalize_threshold_operator


def compatibility_threshold_operator(value) -> str:
    """
    Interpret descriptive representations without changing semantics.

    Examples accepted:
        >=
        probability >= 0.50
        probability>=0.5
        probability >= .50

    Likewise supports the strict '>' representation if encountered.
    """

    raw = str(value).strip().lower()

    compact = re.sub(r"\s+", "", raw)

    # Keep original compact forms supported by Step-31 v1.
    try:
        return ORIGINAL_NORMALIZER(value)
    except RuntimeError:
        pass

    # Compatibility representation:
    # probability >= 0.50
    ge_forms = {
        "probability>=0.50",
        "probability>=0.5",
        "probability>=.50",
        "probability>=.5",
    }

    gt_forms = {
        "probability>0.50",
        "probability>0.5",
        "probability>.50",
        "probability>.5",
    }

    if compact in ge_forms:
        return ">="

    if compact in gt_forms:
        return ">"

    fail(
        "Unsupported threshold operator even after Step-31A "
        f"compatibility normalization: {value!r}"
    )


# =============================================================================
# STEP-31A SOURCE FREEZE
# =============================================================================

def step31a_freeze_source() -> tuple[str, str]:

    src = Path(__file__).resolve()

    source_sha = sha256_file(src)

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if STEP31A_FROZEN_SOURCE.exists():

        frozen_sha = sha256_file(
            STEP31A_FROZEN_SOURCE
        )

        if frozen_sha != source_sha:
            fail(
                "Step-31A frozen source exists but differs from "
                "the executable source.\n"
                f"Frozen : {frozen_sha}\n"
                f"Current: {source_sha}"
            )

    else:

        shutil.copy2(
            src,
            STEP31A_FROZEN_SOURCE,
        )

    frozen_sha = sha256_file(
        STEP31A_FROZEN_SOURCE
    )

    if frozen_sha != source_sha:
        fail(
            "Step-31A frozen-source SHA mismatch."
        )

    return source_sha, frozen_sha


# =============================================================================
# PREPARE COMPATIBILITY AMENDMENT
# =============================================================================

def prepare_amendment() -> str:

    original_sha = sha256_file(
        ORIGINAL_STEP31_SOURCE
    )

    frozen_original_sha = sha256_file(
        ORIGINAL_STEP31_FROZEN_SOURCE
    )

    if original_sha != EXPECTED_STEP31_V1_SHA256:
        fail(
            "Original Step-31 v1 executable source drift.\n"
            f"Expected: {EXPECTED_STEP31_V1_SHA256}\n"
            f"Actual  : {original_sha}"
        )

    if frozen_original_sha != EXPECTED_STEP31_V1_SHA256:
        fail(
            "Original Step-31 v1 frozen source drift.\n"
            f"Expected: {EXPECTED_STEP31_V1_SHA256}\n"
            f"Actual  : {frozen_original_sha}"
        )

    # The failed run must not already have established a final freeze.
    if ORIGINAL_STEP31_FINAL_MANIFEST.exists():
        fail(
            "Original Step-31 final manifest unexpectedly exists. "
            "Do not run compatibility correction until reviewed:\n"
            f"{ORIGINAL_STEP31_FINAL_MANIFEST}"
        )

    wrapper_sha = sha256_file(
        Path(__file__).resolve()
    )

    amendment = {
        "status":
            "STEP31A_THRESHOLD_OPERATOR_COMPATIBILITY_AMENDMENT",

        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "experiment_root":
            str(ROOT),

        "failed_step31_v1": {
            "source_path":
                str(ORIGINAL_STEP31_SOURCE),

            "source_sha256":
                original_sha,

            "frozen_source_path":
                str(ORIGINAL_STEP31_FROZEN_SOURCE),

            "frozen_source_sha256":
                frozen_original_sha,

            "observed_failure":
                OBSERVED_FAILURE,

            "final_freeze_established":
                False,
        },

        "compatibility_correction": {
            "scope":
                "THRESHOLD_OPERATOR_STRING_NORMALIZATION_ONLY",

            "observed_manifest_value":
                "probability >= 0.50",

            "normalized_semantics":
                ">=",

            "threshold_value_changed":
                False,

            "probabilities_changed":
                False,

            "labels_changed":
                False,

            "seed_set_changed":
                False,

            "fold_set_changed":
                False,

            "horizon_set_changed":
                False,

            "metric_definition_changed":
                False,

            "training_changed":
                False,
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

        "step31a_runner": {
            "path":
                str(Path(__file__).resolve()),

            "sha256":
                wrapper_sha,
        },
    }

    amendment["amendment_fingerprint_sha256"] = (
        canonical_json_sha256(amendment)
    )

    with STEP31A_AMENDMENT_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            amendment,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    return sha256_file(
        STEP31A_AMENDMENT_MANIFEST
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 140)
    print(
        "STEP 31A — THRESHOLD-OPERATOR COMPATIBILITY "
        "RUNNER FOR FIVE-SEED OOF AGGREGATION"
    )
    print("=" * 140)

    print(
        "Failed Step-31 v1 SHA256 : "
        f"{EXPECTED_STEP31_V1_SHA256}"
    )

    print(
        "Observed incompatibility : "
        "'probability >= 0.50' -> '>='"
    )

    print(
        "Scientific semantics     : UNCHANGED"
    )

    print(
        "Final test               : LOCKED_NOT_OPENED"
    )

    print("-" * 140)

    amendment_sha = prepare_amendment()

    print("[1] COMPATIBILITY AMENDMENT")
    print(
        f"Amendment manifest SHA256 : {amendment_sha}"
    )
    print(
        "Patch scope               : operator-string normalization only"
    )
    print(
        "Threshold value           : unchanged at 0.50"
    )
    print(
        "Predictions/labels        : unchanged"
    )
    print(
        "Seed/fold/horizon sets    : unchanged"
    )
    print("PASS")
    print("-" * 140)

    # -------------------------------------------------------------------------
    # Patch ONLY representation-level parser.
    # -------------------------------------------------------------------------

    s31.normalize_threshold_operator = (
        compatibility_threshold_operator
    )

    # -------------------------------------------------------------------------
    # Make successful Step-31 endpoint explicitly Step-31A authoritative.
    # -------------------------------------------------------------------------

    s31.FINAL_MANIFEST = (
        STEP31A_FINAL_MANIFEST
    )

    s31.FROZEN_SOURCE = (
        STEP31A_FROZEN_SOURCE
    )

    # Source freeze must refer to this compatibility runner, not failed v1.
    s31.freeze_source = (
        step31a_freeze_source
    )

    # -------------------------------------------------------------------------
    # Execute the otherwise unchanged frozen Step-31 algorithm.
    # -------------------------------------------------------------------------

    s31.main()

    if not STEP31A_FINAL_MANIFEST.is_file():
        fail(
            "Step-31A execution returned without creating "
            "the authoritative final freeze manifest."
        )

    final_sha = sha256_file(
        STEP31A_FINAL_MANIFEST
    )

    print()
    print("=" * 140)
    print(
        "STEP 31A COMPATIBILITY EXECUTION COMPLETE"
    )
    print("=" * 140)

    print(
        f"Compatibility amendment SHA256 : {amendment_sha}"
    )

    print(
        f"Authoritative final manifest    : {STEP31A_FINAL_MANIFEST}"
    )

    print(
        f"Authoritative manifest SHA256   : {final_sha}"
    )

    print(
        "Threshold optimization          : NOT PERFORMED"
    )

    print(
        "Best-seed promotion             : NOT PERFORMED"
    )

    print(
        "Final-test data opened          : False"
    )

    print(
        "Final-test state                : LOCKED_NOT_OPENED"
    )

    print("=" * 140)


if __name__ == "__main__":
    main()