#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 27E — TIMESNET_B PHYSICAL-MICROBATCH-64 FEASIBILITY PROBE

Purpose
-------
Test ONLY whether the already-frozen TIMESNET_B model can execute one complete
logical training update of 256 synthetic samples using physical microbatch 64.

This is an execution-feasibility diagnostic only:
  * no development arrays are discovered or opened;
  * no validation performance is computed;
  * no model/configuration selection is performed;
  * no final-test artifact is discovered, opened, or hashed;
  * no frozen Steps 00-27 are modified;
  * no compatibility amendment is created by this script.

If mb=64 does not fit, the frozen TIMESNET_B mb=2 execution remains untouched.
If mb=64 fits, a separate audited compatibility-amendment step would still be
required before any real TIMESNET_B fit may use it.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np

import step00_config as cfg
import step26_stage1_runner_dryrun_v3 as s26
import step27_stage1_execution_v3 as s27


MODEL_ID = "TIMESNET"
CONFIG_ID = "TIMESNET_B"

SEED = 42
LOGICAL_BATCH = 256
PROBE_MICROBATCH = 64
LOOKBACK = 60
FEATURES = 63
OUTPUTS = 3

EXPECTED_CONFIGURATION_FINGERPRINT = (
    "9675cb42e5544d9a3cda235208b33cc4c27f7583531535505b42056aa86b04dd"
)
EXPECTED_STEP26_FINGERPRINT = (
    "f9198f5799650beedeb42400edc79a9f572d66f9cfe44ade4aae65e059248d22"
)
EXPECTED_STEP27_SOURCE_FINGERPRINT = (
    "3978415d953a1561ebe345180529c97bb8e4a39a89219668008a25b3ad72b3d9"
)
EXPECTED_STEP27_SOURCE_SHA256 = (
    "83ac36dcc5e178051ac90865970f4062626a05f48eda4c0548b3a8b1799ec074"
)

STEP27_SOURCE_FREEZE = (
    cfg.MANIFEST_DIR / "27_stage1_execution_source_freeze.json"
)
STEP27_FROZEN_SCRIPT = (
    cfg.MANIFEST_DIR / "step27_stage1_execution_frozen.py"
)

FORBIDDEN_TEST_FILENAME = "10_final_test_locked_panel.parquet"


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    print("=" * 136)
    print("STEP 27E — TIMESNET_B SYNTHETIC MICRO-BATCH-64 FEASIBILITY PROBE")
    print("=" * 136)

    upstream = s27.verify_upstream()

    if (
        upstream["upstream25"]["step00"]["fingerprint"]
        != EXPECTED_CONFIGURATION_FINGERPRINT
    ):
        raise RuntimeError("Configuration fingerprint drift.")

    if upstream["global_final_test_state"] != "LOCKED_NOT_OPENED":
        raise RuntimeError("Final-test global state drift.")

    source_freeze = load_json(STEP27_SOURCE_FREEZE)

    if (
        source_freeze["stage1_execution_source_fingerprint_sha256"]
        != EXPECTED_STEP27_SOURCE_FINGERPRINT
    ):
        raise RuntimeError("Step-27 source fingerprint drift.")

    if (
        source_freeze["step27_execution_source_sha256"]
        != EXPECTED_STEP27_SOURCE_SHA256
    ):
        raise RuntimeError("Step-27 source SHA drift.")

    if not STEP27_FROZEN_SCRIPT.is_file():
        raise FileNotFoundError(STEP27_FROZEN_SCRIPT)

    if sha256_file(STEP27_FROZEN_SCRIPT) != EXPECTED_STEP27_SOURCE_SHA256:
        raise RuntimeError("Frozen Step-27 source copy drift.")

    if (
        upstream["step26_freeze"][
            "stage1_runner_dryrun_fingerprint_sha256"
        ]
        != EXPECTED_STEP26_FINGERPRINT
    ):
        raise RuntimeError("Step-26 runner fingerprint drift.")

    print("[PASS] Upstream Steps 00-27 source freezes verified.")
    print("[PASS] Step-10 test seal            : SEALED_NOT_OPENED")
    print("[PASS] Global final-test state      : LOCKED_NOT_OPENED")
    print("[PASS] Development arrays opened    : False")
    print("[PASS] Validation performance read  : False")
    print()
    print(
        f"Probe target: {MODEL_ID}/{CONFIG_ID}, "
        f"logical_batch={LOGICAL_BATCH}, physical_microbatch={PROBE_MICROBATCH}"
    )

    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    s26.set_training_determinism(SEED)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, build_metadata, repository = s26.build_candidate_model(
        model_id=MODEL_ID,
        config_id=CONFIG_ID,
        physical_microbatch=PROBE_MICROBATCH,
        device=device,
    )

    learning_rate = float(build_metadata["learning_rate"])

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=s26.BETAS,
        eps=s26.ADAMW_EPS,
        weight_decay=s27.WEIGHT_DECAY,
    )

    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, 27, 1, PROBE_MICROBATCH])
    )
    x = rng.standard_normal(
        (LOGICAL_BATCH, LOOKBACK, FEATURES)
    ).astype(np.float32)
    y = rng.integers(
        0,
        2,
        size=(LOGICAL_BATCH, OUTPUTS),
        dtype=np.int64,
    ).astype(np.float32)

    logical_indices = np.arange(LOGICAL_BATCH, dtype=np.int64)

    torch.cuda.synchronize()
    started = time.perf_counter()

    try:
        update = s26.train_logical_batch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            x_source=x,
            y_source=y,
            logical_indices=logical_indices,
            physical_microbatch=PROBE_MICROBATCH,
            device=device,
        )
        torch.cuda.synchronize()

    except torch.cuda.OutOfMemoryError as exc:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        print()
        print("=" * 136)
        print("STEP 27E RESULT: MICRO-BATCH 64 DOES NOT FIT")
        print("=" * 136)
        print(f"CUDA OOM: {exc}")
        print("Development arrays opened       : False")
        print("Validation performance computed : False")
        print("Compatibility amendment created : False")
        print("Final-test data opened          : False")
        print("=" * 136)
        return 4

    elapsed = time.perf_counter() - started

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)

    if int(update["logical_batch_n"]) != LOGICAL_BATCH:
        raise RuntimeError("Logical batch size drift in probe.")

    if not math.isfinite(float(update["weighted_mean_bce"])):
        raise RuntimeError("Non-finite synthetic BCE.")

    if not math.isfinite(float(update["preclip_gradient_norm"])):
        raise RuntimeError("Non-finite gradient norm.")

    s26.import_step24().assert_single_repository_clean(repository)

    print()
    print("=" * 136)
    print("STEP 27E RESULT: MICRO-BATCH 64 FITS ONE FULL LOGICAL UPDATE")
    print("=" * 136)
    print(f"Model/config                    : {MODEL_ID}/{CONFIG_ID}")
    print(f"Logical batch                   : {LOGICAL_BATCH}")
    print(f"Physical microbatch             : {PROBE_MICROBATCH}")
    print(f"Micro-steps in logical update   : {math.ceil(LOGICAL_BATCH / PROBE_MICROBATCH)}")
    print(f"Candidate learning rate         : {learning_rate:.12g}")
    print(f"Synthetic weighted BCE          : {float(update['weighted_mean_bce']):.8f}")
    print(f"Pre-clip gradient norm          : {float(update['preclip_gradient_norm']):.8f}")
    print(f"Full logical-update time        : {elapsed:.2f} s")
    print(f"Peak CUDA allocated             : {peak_allocated / (1024**2):.1f} MiB")
    print(f"Peak CUDA reserved              : {peak_reserved / (1024**2):.1f} MiB")
    print(f"Physical VRAM                   : {total_memory / (1024**2):.1f} MiB")
    print(f"Allocated fraction of VRAM      : {peak_allocated / total_memory:.4f}")
    print(f"Reserved fraction of VRAM       : {peak_reserved / total_memory:.4f}")
    print("Development arrays opened       : False")
    print("Validation performance computed : False")
    print("Compatibility amendment created : False")
    print("Final-test data opened          : False")
    print("=" * 136)
    print(
        "IMPORTANT: PASS here establishes feasibility only. "
        "Do not use micro-batch 64 for real TIMESNET_B training until a "
        "separate audited compatibility amendment is explicitly frozen."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStep 27E interrupted by user.")
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print()
        print("=" * 136)
        print("STEP 27E FAILED")
        print("=" * 136)
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        print("=" * 136)
        raise SystemExit(1)
