#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 27G — TIMESNET_B MB=64 AMENDMENT-AWARE COMPATIBILITY RUNNER

Purpose
-------
Execute ONLY the already-authorized TIMESNET_B Stage-1 fits under the frozen
Step-27F pre-performance protocol amendment:

    TIMESNET_B physical microbatch: 2 -> 64
    logical batch:                 256 unchanged

The original Step-27 source remains immutable and remains the core training
engine.  Step 27G wraps it narrowly by:
  * verifying the exact Step-27F amendment manifest;
  * freezing this compatibility-runner source before real data use;
  * substituting mb=64 ONLY for TIMESNET_B in the in-memory execution map;
  * launching worker subprocesses through this wrapper so the same amendment
    is installed inside each worker;
  * preserving the canonical Step-27 fit directory/artifact layout;
  * adding explicit Step-27F amendment provenance to each TIMESNET_B
    fit_manifest.json;
  * reusing all original Step-27 training, validation, checkpoint, metric,
    resumability, and protection logic.

Important methodological note
-----------------------------
This is NOT described as a pure execution optimization.  The frozen TimesNet
source selects dominant periods using FFT amplitudes averaged over the current
physical batch.  Therefore mb=64 is an explicit pre-performance protocol
amendment frozen in Step 27F.

Step 27G does NOT perform configuration selection, threshold optimization,
best-seed selection, or final-test evaluation.

Global 80-fit Step-27 freezing is deliberately blocked in this wrapper.
After all 80 fits exist, a later amendment-aware final-freeze step must
reconcile the original Step-27 provenance with Step 27F/27G.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

import step00_config as cfg
import step27_stage1_execution_v3 as s27


STEP_NAME = "STEP 27G — TIMESNET_B MB=64 AMENDMENT-AWARE COMPATIBILITY RUNNER"
SCRIPT = Path(__file__).resolve()

MODEL_ID = "TIMESNET"
CONFIG_ID = "TIMESNET_B"
OLD_MICROBATCH = 2
NEW_MICROBATCH = 64
LOGICAL_BATCH = 256
FULL_ACCUMULATION_STEPS = 4

EXPECTED_CONFIGURATION_FINGERPRINT = (
    "9675cb42e5544d9a3cda235208b33cc4c27f7583531535505b42056aa86b04dd"
)
EXPECTED_STEP27_SOURCE_SHA256 = (
    "83ac36dcc5e178051ac90865970f4062626a05f48eda4c0548b3a8b1799ec074"
)
EXPECTED_STEP27_SOURCE_FINGERPRINT = (
    "3978415d953a1561ebe345180529c97bb8e4a39a89219668008a25b3ad72b3d9"
)
EXPECTED_STEP27F_SOURCE_SHA256 = (
    "2527e2bba7a88deb265f116df3be385f87ca88fb482bdb2bad0299bb586f9e51"
)
EXPECTED_STEP27F_AMENDMENT_SHA256 = (
    "04c30b59b8bef7327b44e1764240ead258edbba64f278e31ca29249141dfdce8"
)

STEP27F_AMENDMENT = (
    cfg.MANIFEST_DIR / "27f_timesnetb_microbatch64_protocol_amendment.json"
)
STEP27_SOURCE_FREEZE = (
    cfg.MANIFEST_DIR / "27_stage1_execution_source_freeze.json"
)
STEP27_FROZEN_SCRIPT = (
    cfg.MANIFEST_DIR / "step27_stage1_execution_frozen.py"
)

SOURCE_FREEZE = (
    cfg.MANIFEST_DIR
    / "27g_timesnetb_mb64_compatibility_runner_source_freeze.json"
)
FROZEN_SOURCE_COPY = (
    cfg.MANIFEST_DIR
    / "step27g_timesnetb_mb64_compatibility_runner_frozen.py"
)

# Save original Step-27 callables before any monkey-patching.
ORIGINAL_EXECUTION_MAPS = s27.execution_maps
ORIGINAL_VERIFY_COMPLETED_FIT = s27.verify_completed_fit


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(payload: Any) -> str:
    return cfg.canonical_json_sha256(payload)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_step27f_amendment() -> dict:
    if not STEP27F_AMENDMENT.is_file():
        raise FileNotFoundError(STEP27F_AMENDMENT)

    observed_sha = sha256_file(STEP27F_AMENDMENT)
    if observed_sha != EXPECTED_STEP27F_AMENDMENT_SHA256:
        raise RuntimeError(
            "Step-27F amendment SHA mismatch.\n"
            f"Expected: {EXPECTED_STEP27F_AMENDMENT_SHA256}\n"
            f"Observed: {observed_sha}"
        )

    amendment = load_json(STEP27F_AMENDMENT)

    if amendment.get("step") != "27F":
        raise RuntimeError("Step-27F amendment step identifier drift.")
    if amendment.get("status") != "FROZEN_PRE_PERFORMANCE_PROTOCOL_AMENDMENT":
        raise RuntimeError("Step-27F amendment status drift.")

    scope = amendment.get("scope", {})
    if (
        scope.get("model_id") != MODEL_ID
        or scope.get("config_id") != CONFIG_ID
    ):
        raise RuntimeError("Step-27F amendment scope drift.")

    change = amendment.get("change", {})
    if int(change.get("physical_microbatch_old", -1)) != OLD_MICROBATCH:
        raise RuntimeError("Step-27F old microbatch drift.")
    if int(change.get("physical_microbatch_new", -1)) != NEW_MICROBATCH:
        raise RuntimeError("Step-27F new microbatch drift.")
    if int(change.get("logical_batch", -1)) != LOGICAL_BATCH:
        raise RuntimeError("Step-27F logical-batch drift.")

    guard = amendment.get("pre_performance_guard", {})
    if int(guard.get("completed_timesnetb_fit_manifests", -1)) != 0:
        raise RuntimeError("Step-27F was not frozen before TIMESNET_B fits.")
    if bool(guard.get("development_arrays_opened_by_step27f")):
        raise RuntimeError("Step-27F development-array guard drift.")
    if bool(guard.get("validation_performance_computed_by_step27f")):
        raise RuntimeError("Step-27F validation-performance guard drift.")
    if bool(guard.get("final_test_opened_or_hashed_by_step27f")):
        raise RuntimeError("Step-27F final-test guard drift.")

    source = amendment.get("source", {})
    if source.get("step27f_script_sha256") != EXPECTED_STEP27F_SOURCE_SHA256:
        raise RuntimeError("Step-27F source SHA provenance drift.")

    upstream = amendment.get("upstream", {})
    if (
        upstream.get("configuration_fingerprint")
        != EXPECTED_CONFIGURATION_FINGERPRINT
    ):
        raise RuntimeError("Step-27F configuration fingerprint drift.")
    if (
        upstream.get("step27_source_fingerprint")
        != EXPECTED_STEP27_SOURCE_FINGERPRINT
    ):
        raise RuntimeError("Step-27F Step-27 source fingerprint drift.")
    if upstream.get("step27_source_sha256") != EXPECTED_STEP27_SOURCE_SHA256:
        raise RuntimeError("Step-27F Step-27 source SHA drift.")

    if not STEP27_SOURCE_FREEZE.is_file() or not STEP27_FROZEN_SCRIPT.is_file():
        raise FileNotFoundError("Frozen original Step-27 source artifacts missing.")

    source_freeze = load_json(STEP27_SOURCE_FREEZE)
    if (
        source_freeze.get("stage1_execution_source_fingerprint_sha256")
        != EXPECTED_STEP27_SOURCE_FINGERPRINT
    ):
        raise RuntimeError("Original Step-27 source-freeze fingerprint drift.")
    if (
        source_freeze.get("step27_execution_source_sha256")
        != EXPECTED_STEP27_SOURCE_SHA256
    ):
        raise RuntimeError("Original Step-27 source-freeze SHA drift.")
    if sha256_file(STEP27_FROZEN_SCRIPT) != EXPECTED_STEP27_SOURCE_SHA256:
        raise RuntimeError("Frozen original Step-27 source copy drift.")

    return amendment


def build_source_semantic() -> dict:
    return {
        "schema_version": 1,
        "status": "TIMESNET_B_MB64_COMPATIBILITY_RUNNER_SOURCE_FROZEN",
        "configuration_fingerprint_sha256": EXPECTED_CONFIGURATION_FINGERPRINT,
        "step27g_source_sha256": sha256_file(SCRIPT),
        "original_step27": {
            "source_sha256": EXPECTED_STEP27_SOURCE_SHA256,
            "source_fingerprint_sha256": EXPECTED_STEP27_SOURCE_FINGERPRINT,
        },
        "protocol_amendment": {
            "step": "27F",
            "manifest_path": str(STEP27F_AMENDMENT.resolve()),
            "manifest_sha256": EXPECTED_STEP27F_AMENDMENT_SHA256,
            "step27f_source_sha256": EXPECTED_STEP27F_SOURCE_SHA256,
            "model_id": MODEL_ID,
            "config_id": CONFIG_ID,
            "physical_microbatch_old": OLD_MICROBATCH,
            "physical_microbatch_new": NEW_MICROBATCH,
            "logical_batch": LOGICAL_BATCH,
        },
        "scope": {
            "real_fit_scope": f"{MODEL_ID}/{CONFIG_ID} only",
            "folds": [1, 2, 3, 4, 5],
            "configuration_selection_performed": False,
            "threshold_optimization_performed": False,
            "best_seed_selection_performed": False,
            "global_step27_freeze_allowed": False,
            "final_test_allowed": False,
        },
    }


def ensure_source_freeze() -> tuple[str, str]:
    semantic = build_source_semantic()
    source_sha = semantic["step27g_source_sha256"]
    fingerprint = canonical_hash(semantic)

    if not SOURCE_FREEZE.exists():
        if FROZEN_SOURCE_COPY.exists():
            raise RuntimeError(
                "Step-27G frozen source copy exists without source-freeze JSON."
            )

        payload = {
            **semantic,
            "step27g_source_fingerprint_sha256": fingerprint,
        }
        # Deliberately omit wall-clock timestamps from the immutable semantic.
        s27.atomic_json(SOURCE_FREEZE, payload)
        shutil.copy2(SCRIPT, FROZEN_SOURCE_COPY)

    payload = load_json(SOURCE_FREEZE)

    if (
        payload.get("step27g_source_fingerprint_sha256")
        != fingerprint
    ):
        raise RuntimeError("Step-27G source fingerprint drift.")

    recompute = dict(payload)
    stored_fp = recompute.pop("step27g_source_fingerprint_sha256", None)
    if canonical_hash(recompute) != stored_fp:
        raise RuntimeError("Step-27G source freeze no longer recomputes.")

    if payload.get("step27g_source_sha256") != source_sha:
        raise RuntimeError("Current Step-27G source differs from frozen source.")

    if not FROZEN_SOURCE_COPY.is_file():
        raise FileNotFoundError(FROZEN_SOURCE_COPY)
    if sha256_file(FROZEN_SOURCE_COPY) != source_sha:
        raise RuntimeError("Frozen Step-27G source copy differs from current source.")

    return fingerprint, source_sha


def partition_sizes(n: int, physical: int) -> list[int]:
    if n < 1 or physical < 1:
        raise ValueError("Invalid partition dimensions.")
    out = []
    left = int(n)
    while left:
        take = min(left, physical)
        out.append(take)
        left -= take
    return out


def amended_execution_maps(upstream: dict) -> tuple[dict, pd.DataFrame]:
    schedule_map, batch_plan = ORIGINAL_EXECUTION_MAPS(upstream)

    key = (MODEL_ID, CONFIG_ID)
    if key not in schedule_map:
        raise RuntimeError("Frozen schedule is missing TIMESNET_B.")

    original_record = schedule_map[key]
    if int(original_record["physical_microbatch_size"]) != OLD_MICROBATCH:
        raise RuntimeError(
            "Original Step-24/25 TIMESNET_B microbatch is no longer 2."
        )

    schedule_map = copy.deepcopy(schedule_map)
    amended_record = copy.deepcopy(schedule_map[key])
    amended_record["physical_microbatch_size"] = NEW_MICROBATCH
    amended_record["gradient_accumulation_steps"] = FULL_ACCUMULATION_STEPS
    amended_record["step27f_protocol_amendment_sha256"] = (
        EXPECTED_STEP27F_AMENDMENT_SHA256
    )
    amended_record["physical_microbatch_provenance"] = (
        "Step-27F pre-performance protocol amendment"
    )
    schedule_map[key] = amended_record

    batch_plan = batch_plan.copy(deep=True)
    mask = (
        (batch_plan["model_id"] == MODEL_ID)
        & (batch_plan["config_id"] == CONFIG_ID)
    )
    if int(mask.sum()) != 5:
        raise RuntimeError("Expected five TIMESNET_B fold plans.")

    # Confirm all five original rows were mb=2 before in-memory amendment.
    original_values = set(
        batch_plan.loc[mask, "physical_microbatch_size"].astype(int).tolist()
    )
    if original_values != {OLD_MICROBATCH}:
        raise RuntimeError(
            f"Original TIMESNET_B fold-plan microbatch drift: {original_values}"
        )

    batch_plan.loc[mask, "physical_microbatch_size"] = NEW_MICROBATCH

    if "full_logical_accumulation_steps" in batch_plan.columns:
        batch_plan.loc[
            mask, "full_logical_accumulation_steps"
        ] = FULL_ACCUMULATION_STEPS

    # Recompute final partial-logical-batch partition metadata for audit fidelity.
    for idx in batch_plan.index[mask]:
        final_n = int(batch_plan.at[idx, "final_logical_batch_size"])
        parts = partition_sizes(final_n, NEW_MICROBATCH)

        if "final_logical_microbatch_count" in batch_plan.columns:
            batch_plan.at[idx, "final_logical_microbatch_count"] = len(parts)
        if "final_logical_partition_json" in batch_plan.columns:
            batch_plan.at[idx, "final_logical_partition_json"] = json.dumps(parts)
        if "final_loss_weight_sum" in batch_plan.columns:
            batch_plan.at[idx, "final_loss_weight_sum"] = sum(
                p / final_n for p in parts
            )

    return schedule_map, batch_plan


def amendment_manifest_record() -> dict:
    return {
        "step": "27F",
        "classification": (
            "pre-performance protocol amendment affecting TimesNet "
            "physical-batch-dependent FFT period selection"
        ),
        "manifest_path": str(STEP27F_AMENDMENT.resolve()),
        "manifest_sha256": EXPECTED_STEP27F_AMENDMENT_SHA256,
        "step27f_source_sha256": EXPECTED_STEP27F_SOURCE_SHA256,
        "physical_microbatch_old": OLD_MICROBATCH,
        "physical_microbatch_new": NEW_MICROBATCH,
        "logical_batch_unchanged": LOGICAL_BATCH,
        "timesnet_batch_averaged_fft_semantics_confirmed": True,
    }


def attach_amendment_to_fit_manifest(path: Path) -> str:
    payload = load_json(path)

    if payload.get("model_id") != MODEL_ID or payload.get("config_id") != CONFIG_ID:
        raise RuntimeError("Refusing to attach Step-27F amendment to non-TIMESNET_B fit.")
    if int(payload.get("physical_microbatch_size", -1)) != NEW_MICROBATCH:
        raise RuntimeError("TIMESNET_B fit manifest is not mb=64.")
    if int(payload.get("logical_batch_size", -1)) != LOGICAL_BATCH:
        raise RuntimeError("TIMESNET_B logical batch drift.")
    if int(payload.get("full_logical_accumulation_steps", -1)) != FULL_ACCUMULATION_STEPS:
        raise RuntimeError("TIMESNET_B accumulation-step drift.")

    payload["protocol_amendment"] = amendment_manifest_record()
    payload["physical_microbatch_provenance"] = (
        "Step-27F frozen pre-performance protocol amendment; "
        "original Step-24/25 TIMESNET_B value was 2"
    )
    payload["core_training_engine"] = (
        "immutable Step-27 worker_fit via Step-26 helpers"
    )

    # Rewrite atomically so the final completion record includes amendment provenance.
    s27.atomic_json(path, payload)
    return sha256_file(path)


def amended_verify_completed_fit(
    model_id: str,
    config_id: str,
    fold: int,
    *,
    source_fp: str,
    input_fp: str,
    schedule_map: dict,
    input_map: dict,
):
    payload = ORIGINAL_VERIFY_COMPLETED_FIT(
        model_id,
        config_id,
        fold,
        source_fp=source_fp,
        input_fp=input_fp,
        schedule_map=schedule_map,
        input_map=input_map,
    )
    if payload is None:
        return None

    if (model_id, config_id) != (MODEL_ID, CONFIG_ID):
        return payload

    amendment = payload.get("protocol_amendment", {})
    if amendment.get("step") != "27F":
        return None
    if amendment.get("manifest_sha256") != EXPECTED_STEP27F_AMENDMENT_SHA256:
        return None
    if int(amendment.get("physical_microbatch_old", -1)) != OLD_MICROBATCH:
        return None
    if int(amendment.get("physical_microbatch_new", -1)) != NEW_MICROBATCH:
        return None
    if int(amendment.get("logical_batch_unchanged", -1)) != LOGICAL_BATCH:
        return None
    if not bool(
        amendment.get("timesnet_batch_averaged_fft_semantics_confirmed")
    ):
        return None

    if int(payload.get("physical_microbatch_size", -1)) != NEW_MICROBATCH:
        return None
    if int(payload.get("full_logical_accumulation_steps", -1)) != FULL_ACCUMULATION_STEPS:
        return None

    return payload


def amended_worker_main(fold: int) -> int:
    try:
        verify_step27f_amendment()
        ensure_source_freeze()
        install_patches(for_worker=True)

        result = s27.worker_fit(
            model_id=MODEL_ID,
            config_id=CONFIG_ID,
            fold=int(fold),
        )

        manifest_path = Path(result["fit_manifest_path"])
        result["fit_manifest_sha256"] = attach_amendment_to_fit_manifest(
            manifest_path
        )
        result["step27f_protocol_amendment_sha256"] = (
            EXPECTED_STEP27F_AMENDMENT_SHA256
        )

        print(
            s27.RESULT_PREFIX
            + json.dumps(result, sort_keys=True, ensure_ascii=False),
            flush=True,
        )
        return 0

    except Exception as exc:
        failure = {
            "status": "FAIL",
            "model_id": MODEL_ID,
            "config_id": CONFIG_ID,
            "fold": int(fold),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "step27f_protocol_amendment_sha256":
                EXPECTED_STEP27F_AMENDMENT_SHA256,
            "configuration_selection_performed": False,
            "threshold_optimization_performed": False,
            "best_seed_selection_performed": False,
            "final_test_data_opened": False,
        }
        print(
            s27.RESULT_PREFIX
            + json.dumps(failure, sort_keys=True, ensure_ascii=False),
            flush=True,
        )
        return 1


def amended_run_worker_streaming(
    model_id: str,
    config_id: str,
    fold: int,
) -> dict:
    if (model_id, config_id) != (MODEL_ID, CONFIG_ID):
        raise RuntimeError(
            "Step 27G is authorized only for TIMESNET/TIMESNET_B."
        )

    directory = s27.fit_dir(model_id, config_id, fold)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = s27.worker_log_path(model_id, config_id, fold)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    command = [
        sys.executable,
        str(SCRIPT),
        "--worker",
        "--fold",
        str(int(fold)),
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    lines = []
    result_payloads = []

    assert process.stdout is not None

    for raw in process.stdout:
        line = raw.rstrip("\n")
        lines.append(line)

        if line.startswith(s27.RESULT_PREFIX):
            try:
                result_payloads.append(
                    json.loads(line[len(s27.RESULT_PREFIX):])
                )
            except Exception:
                pass
        else:
            print(f"        {line}", flush=True)

    return_code = process.wait()

    s27.atomic_text(log_path, "\n".join(lines) + "\n")

    if len(result_payloads) != 1:
        result = {
            "status": "FAIL",
            "model_id": model_id,
            "config_id": config_id,
            "fold": int(fold),
            "exception_type": "WorkerProtocolError",
            "exception_message": (
                "Expected exactly one Step-27G worker result marker; "
                f"found {len(result_payloads)}."
            ),
        }
    else:
        result = result_payloads[0]

    result["worker_return_code"] = int(return_code)
    result["worker_log"] = str(log_path.resolve())
    return result


def blocked_global_freeze(*args, **kwargs):
    raise RuntimeError(
        "Step 27G deliberately blocks the original global Step-27 freeze. "
        "Use the later amendment-aware final-freeze step after all 80 fits "
        "are complete."
    )


def install_patches(*, for_worker: bool) -> None:
    # Install only in memory.  Original Step-27 source remains unchanged.
    s27.execution_maps = amended_execution_maps
    s27.verify_completed_fit = amended_verify_completed_fit

    if not for_worker:
        s27.run_worker_streaming = amended_run_worker_streaming
        s27.freeze_complete_stage = blocked_global_freeze


def count_existing_timesnetb_manifests() -> int:
    root = s27.RUN_ROOT / MODEL_ID / CONFIG_ID
    if not root.exists():
        return 0
    return len(list(root.rglob("fit_manifest.json")))


def prepare_only() -> int:
    print("=" * 136)
    print(STEP_NAME)
    print("=" * 136)

    amendment = verify_step27f_amendment()

    upstream = s27.verify_upstream()
    if upstream["global_final_test_state"] != "LOCKED_NOT_OPENED":
        raise RuntimeError("Global final-test state drift.")

    fp, source_sha = ensure_source_freeze()

    print("[PASS] Original Step-27 source remains immutable.")
    print(f"[PASS] Step-27F amendment SHA-256 : {EXPECTED_STEP27F_AMENDMENT_SHA256}")
    print(f"[PASS] Step-27G source SHA-256     : {source_sha}")
    print(f"[PASS] Step-27G source fingerprint : {fp}")
    print(f"[PASS] Compatibility scope         : {MODEL_ID}/{CONFIG_ID}")
    print(f"[PASS] Physical microbatch         : {OLD_MICROBATCH} -> {NEW_MICROBATCH}")
    print(f"[PASS] Logical batch               : {LOGICAL_BATCH} (UNCHANGED)")
    print(f"[PASS] Existing TIMESNET_B manifests: {count_existing_timesnetb_manifests()}")
    print("[PASS] Development arrays opened   : False")
    print("[PASS] Validation performance read : False")
    print("[PASS] Final-test data opened      : False")
    print()
    print("=" * 136)
    print("STEP 27G PREPARE STATUS: PASS")
    print("=" * 136)
    print("Real TIMESNET_B training started : False")
    print("Configuration selection          : NOT PERFORMED")
    print("Threshold optimization           : NOT PERFORMED")
    print("Best-seed selection              : NOT PERFORMED")
    print("Global Step-27 freeze             : BLOCKED IN THIS WRAPPER")
    print("Final-test state                  : LOCKED_NOT_OPENED")
    print("=" * 136)
    print(
        "NEXT: audit this preparation output. "
        "Then run a one-fit canary with --max-new-fits 1 when instructed."
    )
    return 0


def run_parent(max_new_fits: int | None, fold: int | None) -> int:
    verify_step27f_amendment()
    ensure_source_freeze()
    install_patches(for_worker=False)

    args = SimpleNamespace(
        prepare_only=False,
        verify_only=False,
        worker=False,
        model_id=MODEL_ID,
        config_id=CONFIG_ID,
        fold=fold,
        max_new_fits=max_new_fits,
    )

    print()
    print("=" * 136)
    print("STEP 27G AMENDMENT BINDING")
    print("=" * 136)
    print(f"Target                           : {MODEL_ID}/{CONFIG_ID}")
    print(f"Step-27F amendment SHA-256       : {EXPECTED_STEP27F_AMENDMENT_SHA256}")
    print(f"Physical microbatch              : {OLD_MICROBATCH} -> {NEW_MICROBATCH}")
    print(f"Logical batch                    : {LOGICAL_BATCH} (UNCHANGED)")
    print("TimesNet batch-dependent FFT     : EXPLICITLY AMENDED")
    print("Core trainer                     : IMMUTABLE STEP 27")
    print("Configuration selection          : NOT PERFORMED")
    print("Final-test data opened           : False")
    print("=" * 136)
    print()

    return s27.master_main(args)


def parse_args():
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze and verify Step-27G without starting real training.",
    )
    parser.add_argument(
        "--max-new-fits",
        type=int,
        default=None,
        help="Run at most this many new TIMESNET_B folds.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Operationally restrict execution to one TIMESNET_B fold.",
    )
    # Internal worker mode.  Model/config are fixed constants by design.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_new_fits is not None and args.max_new_fits < 1:
        raise SystemExit("--max-new-fits must be >= 1")

    if args.worker:
        if args.fold is None:
            raise SystemExit("--worker requires --fold")
        if args.prepare_only or args.max_new_fits is not None:
            raise SystemExit("Invalid internal worker argument combination.")
        return amended_worker_main(args.fold)

    if args.prepare_only:
        if args.fold is not None or args.max_new_fits is not None:
            raise SystemExit(
                "--prepare-only cannot be combined with --fold or --max-new-fits"
            )
        return prepare_only()

    return run_parent(args.max_new_fits, args.fold)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nStep 27G interrupted by user. "
            "Hash-valid completed fits remain resumable."
        )
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print()
        print("=" * 136)
        print("STEP 27G FAILED")
        print("=" * 136)
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        print("=" * 136)
        raise SystemExit(1)
