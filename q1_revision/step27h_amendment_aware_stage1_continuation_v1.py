#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 27H — AMENDMENT-AWARE STAGE-1 CONTINUATION WRAPPER

Purpose
-------
Continue Stage-1 training for the remaining NON-TIMESNET model families while
correctly reconciling the already-frozen TIMESNET_B Step-27F amendment
(physical microbatch 2 -> 64).

Why this wrapper is needed
--------------------------
The immutable original Step-27 verifier expects the original TIMESNET_B
physical microbatch=2.  The five valid amended TIMESNET_B manifests now record
physical microbatch=64.  Running the original Step-27 parent directly would
therefore under-count those amended fits during global reconciliation.

Step 27H:
  * verifies Step-27F and Step-27G provenance;
  * verifies all five TIMESNET_B amended fit manifests before new training;
  * patches ONLY the in-memory execution-map and completed-fit verifier so the
    amended TIMESNET_B fits reconcile as valid;
  * keeps the original Step-27 worker launcher for all new non-TimesNet fits;
  * does not modify the original Step-27 source;
  * blocks the ordinary global Step-27 freeze; a later amendment-aware final
    freeze will be used after all 80 fits exist;
  * performs no configuration selection, threshold optimization, best-seed
    selection, or final-test evaluation.

Real training under this wrapper is intentionally restricted to:
  ITRANSFORMER, FREDFORMER, FRNET, MLF
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import step00_config as cfg
import step27_stage1_execution_v3 as s27
import step27g_timesnetb_mb64_compatibility_runner_v1 as s27g


STEP_NAME = "STEP 27H — AMENDMENT-AWARE STAGE-1 CONTINUATION WRAPPER"
SCRIPT = Path(__file__).resolve()

ALLOWED_MODELS = ("ITRANSFORMER", "FREDFORMER", "FRNET", "MLF")

EXPECTED_CONFIGURATION_FINGERPRINT = (
    "9675cb42e5544d9a3cda235208b33cc4c27f7583531535505b42056aa86b04dd"
)
EXPECTED_STEP27_SOURCE_SHA256 = (
    "83ac36dcc5e178051ac90865970f4062626a05f48eda4c0548b3a8b1799ec074"
)
EXPECTED_STEP27_SOURCE_FINGERPRINT = (
    "3978415d953a1561ebe345180529c97bb8e4a39a89219668008a25b3ad72b3d9"
)
EXPECTED_STEP27F_AMENDMENT_SHA256 = (
    "04c30b59b8bef7327b44e1764240ead258edbba64f278e31ca29249141dfdce8"
)
EXPECTED_STEP27G_SOURCE_SHA256 = (
    "566ee24f269d1ec049cb815258409527e47ab0a438adf4bd78f7731f6a4a318b"
)
EXPECTED_STEP27G_SOURCE_FINGERPRINT = (
    "b5efacae2f49f9110e82dd9a8a2ae08e7053621f5342b5cf8f42b78f15a9de66"
)

STEP27G_SOURCE_FREEZE = (
    cfg.MANIFEST_DIR
    / "27g_timesnetb_mb64_compatibility_runner_source_freeze.json"
)
STEP27G_FROZEN_SOURCE = (
    cfg.MANIFEST_DIR
    / "step27g_timesnetb_mb64_compatibility_runner_frozen.py"
)

SOURCE_FREEZE = (
    cfg.MANIFEST_DIR
    / "27h_amendment_aware_stage1_continuation_source_freeze.json"
)
FROZEN_SOURCE = (
    cfg.MANIFEST_DIR
    / "step27h_amendment_aware_stage1_continuation_frozen.py"
)


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


def verify_step27g_provenance() -> None:
    s27g.verify_step27f_amendment()

    if not STEP27G_SOURCE_FREEZE.is_file():
        raise FileNotFoundError(STEP27G_SOURCE_FREEZE)
    if not STEP27G_FROZEN_SOURCE.is_file():
        raise FileNotFoundError(STEP27G_FROZEN_SOURCE)

    payload = load_json(STEP27G_SOURCE_FREEZE)

    if (
        payload.get("step27g_source_sha256")
        != EXPECTED_STEP27G_SOURCE_SHA256
    ):
        raise RuntimeError("Step-27G source SHA drift.")

    if (
        payload.get("step27g_source_fingerprint_sha256")
        != EXPECTED_STEP27G_SOURCE_FINGERPRINT
    ):
        raise RuntimeError("Step-27G source fingerprint drift.")

    if sha256_file(STEP27G_FROZEN_SOURCE) != EXPECTED_STEP27G_SOURCE_SHA256:
        raise RuntimeError("Frozen Step-27G source copy drift.")

    amendment = payload.get("protocol_amendment", {})
    if (
        amendment.get("manifest_sha256")
        != EXPECTED_STEP27F_AMENDMENT_SHA256
    ):
        raise RuntimeError("Step-27G -> Step-27F amendment binding drift.")


def source_semantic() -> dict:
    return {
        "schema_version": 1,
        "status": "AMENDMENT_AWARE_STAGE1_CONTINUATION_SOURCE_FROZEN",
        "configuration_fingerprint_sha256":
            EXPECTED_CONFIGURATION_FINGERPRINT,
        "step27h_source_sha256": sha256_file(SCRIPT),
        "original_step27": {
            "source_sha256": EXPECTED_STEP27_SOURCE_SHA256,
            "source_fingerprint_sha256":
                EXPECTED_STEP27_SOURCE_FINGERPRINT,
        },
        "step27f_amendment_sha256":
            EXPECTED_STEP27F_AMENDMENT_SHA256,
        "step27g": {
            "source_sha256": EXPECTED_STEP27G_SOURCE_SHA256,
            "source_fingerprint_sha256":
                EXPECTED_STEP27G_SOURCE_FINGERPRINT,
        },
        "new_training_scope": list(ALLOWED_MODELS),
        "policy": {
            "timesnet_b_new_training_allowed": False,
            "original_step27_worker_used_for_new_non_timesnet_fits": True,
            "global_step27_freeze_allowed": False,
            "configuration_selection_performed": False,
            "threshold_optimization_performed": False,
            "best_seed_selection_performed": False,
            "final_test_allowed": False,
        },
    }


def ensure_source_freeze() -> tuple[str, str]:
    semantic = source_semantic()
    source_sha = semantic["step27h_source_sha256"]
    fingerprint = canonical_hash(semantic)

    if not SOURCE_FREEZE.exists():
        if FROZEN_SOURCE.exists():
            raise RuntimeError(
                "Step-27H frozen source exists without source-freeze JSON."
            )
        payload = {
            **semantic,
            "step27h_source_fingerprint_sha256": fingerprint,
        }
        s27.atomic_json(SOURCE_FREEZE, payload)
        shutil.copy2(SCRIPT, FROZEN_SOURCE)

    payload = load_json(SOURCE_FREEZE)

    if payload.get("step27h_source_sha256") != source_sha:
        raise RuntimeError("Current Step-27H source differs from frozen source.")

    if (
        payload.get("step27h_source_fingerprint_sha256")
        != fingerprint
    ):
        raise RuntimeError("Step-27H source fingerprint drift.")

    recompute = dict(payload)
    stored = recompute.pop("step27h_source_fingerprint_sha256", None)
    if canonical_hash(recompute) != stored:
        raise RuntimeError("Step-27H source freeze no longer recomputes.")

    if not FROZEN_SOURCE.is_file():
        raise FileNotFoundError(FROZEN_SOURCE)
    if sha256_file(FROZEN_SOURCE) != source_sha:
        raise RuntimeError("Frozen Step-27H source copy drift.")

    return fingerprint, source_sha


def install_reconciliation_patches() -> None:
    # Amendment-aware global reconciliation.
    s27.execution_maps = s27g.amended_execution_maps
    s27.verify_completed_fit = s27g.amended_verify_completed_fit

    # New fits for the permitted non-TimesNet families continue to use the
    # original immutable Step-27 worker launcher.  Do NOT replace
    # s27.run_worker_streaming with the Step-27G TIMESNET_B-only launcher.
    s27.freeze_complete_stage = blocked_global_freeze


def blocked_global_freeze(*args, **kwargs):
    raise RuntimeError(
        "All 80 fits may now exist, but the ordinary Step-27 global freeze is "
        "not amendment-aware. Stop here and use the later amendment-aware "
        "final-freeze step."
    )


def verify_all_timesnetb_fits() -> tuple[dict, str, str, dict, dict]:
    upstream = s27.verify_upstream()

    source_fp, source_sha = s27.ensure_source_freeze(upstream)

    if source_sha != EXPECTED_STEP27_SOURCE_SHA256:
        raise RuntimeError("Original Step-27 source SHA drift.")
    if source_fp != EXPECTED_STEP27_SOURCE_FINGERPRINT:
        raise RuntimeError("Original Step-27 source fingerprint drift.")

    input_fp, input_register = s27.ensure_input_register(
        upstream,
        source_fp,
    )
    input_map = s27.input_record_map(input_register)
    schedule_map, _batch_plan = s27g.amended_execution_maps(upstream)

    verified = []
    for fold in s27.FOLDS:
        fit = s27g.amended_verify_completed_fit(
            s27g.MODEL_ID,
            s27g.CONFIG_ID,
            int(fold),
            source_fp=source_fp,
            input_fp=input_fp,
            schedule_map=schedule_map,
            input_map=input_map,
        )
        if fit is None:
            raise RuntimeError(
                f"Amended TIMESNET_B fold {fold} does not verify."
            )
        verified.append(fit)

    return upstream, source_fp, input_fp, input_map, schedule_map


def prepare_only() -> int:
    print("=" * 136)
    print(STEP_NAME)
    print("=" * 136)

    verify_step27g_provenance()
    fp, source_sha = ensure_source_freeze()

    upstream, _sfp, _ifp, _imap, _smap = verify_all_timesnetb_fits()

    if upstream["global_final_test_state"] != "LOCKED_NOT_OPENED":
        raise RuntimeError("Global final-test state drift.")

    print("[PASS] Step-27F amendment verified.")
    print(f"[PASS] Step-27F amendment SHA-256 : {EXPECTED_STEP27F_AMENDMENT_SHA256}")
    print("[PASS] Step-27G provenance verified.")
    print(f"[PASS] Step-27G source fingerprint : {EXPECTED_STEP27G_SOURCE_FINGERPRINT}")
    print("[PASS] TIMESNET_B amended fits      : 5/5 VERIFIED")
    print(f"[PASS] Step-27H source SHA-256     : {source_sha}")
    print(f"[PASS] Step-27H source fingerprint : {fp}")
    print(f"[PASS] New-training scope           : {', '.join(ALLOWED_MODELS)}")
    print("[PASS] New real training started    : False")
    print("[PASS] Configuration selection      : NOT PERFORMED")
    print("[PASS] Threshold optimization       : NOT PERFORMED")
    print("[PASS] Best-seed selection          : NOT PERFORMED")
    print("[PASS] Final-test data opened       : False")
    print()
    print("=" * 136)
    print("STEP 27H PREPARE STATUS: PASS")
    print("=" * 136)
    print("Verified amended TIMESNET_B fits : 5/5")
    print("Global Step-27 freeze             : BLOCKED IN THIS WRAPPER")
    print("Final-test state                  : LOCKED_NOT_OPENED")
    print("=" * 136)
    print(
        "NEXT: audit this output. Then run one remaining model family "
        "when instructed."
    )
    return 0


def run_parent(
    model_id: str,
    config_id: str | None,
    fold: int | None,
    max_new_fits: int | None,
) -> int:
    if model_id not in ALLOWED_MODELS:
        raise RuntimeError(
            f"Step 27H new training is restricted to {ALLOWED_MODELS}; "
            f"received {model_id!r}."
        )

    verify_step27g_provenance()
    ensure_source_freeze()
    verify_all_timesnetb_fits()
    install_reconciliation_patches()

    args = SimpleNamespace(
        prepare_only=False,
        verify_only=False,
        worker=False,
        model_id=model_id,
        config_id=config_id,
        fold=fold,
        max_new_fits=max_new_fits,
    )

    print()
    print("=" * 136)
    print("STEP 27H AMENDMENT-AWARE CONTINUATION")
    print("=" * 136)
    print(f"New-training model               : {model_id}")
    print(f"Step-27F amendment SHA-256       : {EXPECTED_STEP27F_AMENDMENT_SHA256}")
    print("TIMESNET_B amended fits          : 5/5 VERIFIED")
    print("New-fit core worker              : IMMUTABLE ORIGINAL STEP 27")
    print("Global reconciliation            : AMENDMENT-AWARE")
    print("Configuration selection          : NOT PERFORMED")
    print("Threshold optimization           : NOT PERFORMED")
    print("Best-seed selection              : NOT PERFORMED")
    print("Final-test data opened           : False")
    print("=" * 136)
    print()

    return s27.master_main(args)


def parse_args():
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze/verify Step-27H without starting real training.",
    )
    parser.add_argument(
        "--model-id",
        choices=ALLOWED_MODELS,
        default=None,
        help="Required for real training.",
    )
    parser.add_argument(
        "--config-id",
        default=None,
        help="Optional exact frozen config ID within --model-id.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
    )
    parser.add_argument(
        "--max-new-fits",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_new_fits is not None and args.max_new_fits < 1:
        raise SystemExit("--max-new-fits must be >= 1")

    if args.prepare_only:
        if any(
            value is not None
            for value in (
                args.model_id,
                args.config_id,
                args.fold,
                args.max_new_fits,
            )
        ):
            raise SystemExit(
                "--prepare-only cannot be combined with training filters."
            )
        return prepare_only()

    if args.model_id is None:
        raise SystemExit(
            "Real training requires --model-id. "
            f"Choose one of: {', '.join(ALLOWED_MODELS)}"
        )

    return run_parent(
        model_id=args.model_id,
        config_id=args.config_id,
        fold=args.fold,
        max_new_fits=args.max_new_fits,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nStep 27H interrupted by user. "
            "Hash-valid completed fits remain resumable."
        )
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print()
        print("=" * 136)
        print("STEP 27H FAILED")
        print("=" * 136)
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        print("=" * 136)
        raise SystemExit(1)
