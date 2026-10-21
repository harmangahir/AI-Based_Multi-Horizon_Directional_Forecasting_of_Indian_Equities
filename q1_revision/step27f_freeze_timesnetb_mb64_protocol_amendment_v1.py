#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 27F — TIMESNET_B PRE-PERFORMANCE MICRO-BATCH PROTOCOL AMENDMENT

Purpose
-------
Freeze one transparent protocol amendment BEFORE any TIMESNET_B validation
performance has been observed:

    TIMESNET_B physical microbatch: 2 -> 64
    Logical batch:                 256 (unchanged)

Why this is a protocol amendment, not a pure execution optimization
-------------------------------------------------------------------
The frozen official TimesNet implementation selects dominant periods using
FFT amplitudes averaged across the current physical batch:

    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)

Therefore physical microbatch size can affect the model's forward computation.
This amendment is made solely because the originally frozen mb=2 path was
computationally infeasible in real execution, and BEFORE any completed
TIMESNET_B fit or validation metric exists.

This script:
  * verifies frozen upstream state;
  * verifies no completed TIMESNET_B fit exists;
  * verifies the frozen TimesNet batch-averaged FFT source semantics;
  * programmatically re-probes TIMESNET_B at mb=64 using synthetic data only;
  * requires the mb=64 probe to satisfy the frozen allocated-memory budget;
  * writes an immutable amendment manifest/report;
  * does NOT alter Steps 00-27;
  * does NOT open development arrays;
  * does NOT compute validation performance;
  * does NOT open/hash final-test data;
  * does NOT resume real Stage-1 training.

A separate compatibility runner must consume this amendment before any real
TIMESNET_B training is allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import step00_config as cfg
import step26_stage1_runner_dryrun_v3 as s26
import step27_stage1_execution_v3 as s27


MODEL_ID = "TIMESNET"
CONFIG_ID = "TIMESNET_B"

SEED = 42
LOGICAL_BATCH = 256
FROZEN_OLD_MICROBATCH = 2
AMENDED_MICROBATCH = 64
LOOKBACK = 60
FEATURES = 63
OUTPUTS = 3

# Frozen Step-24 allocated-memory safety budget: 75% of physical 8,150.6 MiB.
ALLOCATED_MEMORY_BUDGET_MIB = 6113.2

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

STEP27_SOURCE_FREEZE = cfg.MANIFEST_DIR / "27_stage1_execution_source_freeze.json"
STEP27_FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step27_stage1_execution_frozen.py"

TIMESNET_SOURCE = (
    cfg.EXPERIMENT_ROOT_DIR
    / "third_party_sources"
    / "step17_official_model_sources"
    / "Time-Series-Library"
    / "models"
    / "TimesNet.py"
)

RUN_ROOT = cfg.EXPERIMENT_ROOT_DIR / "stage1_fits"

AMENDMENT_MANIFEST = (
    cfg.MANIFEST_DIR / "27f_timesnetb_microbatch64_protocol_amendment.json"
)
AMENDMENT_REPORT = (
    cfg.REPORT_DIR / "27f_timesnetb_microbatch64_protocol_amendment.txt"
)
PROBE_AUDIT = (
    cfg.AUDIT_DIR / "27f_timesnetb_microbatch64_synthetic_probe.json"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return (
        json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def walk_nodes(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_nodes(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_nodes(v)


def looks_like_timesnetb_mb2_record(d: dict) -> bool:
    values = {str(v) for v in d.values() if isinstance(v, (str, int, float, bool))}
    has_model = MODEL_ID in values
    has_config = CONFIG_ID in values

    mb_keys = [
        k for k in d.keys()
        if isinstance(k, str)
        and "micro" in k.lower()
        and "batch" in k.lower()
    ]
    has_mb2 = any(
        isinstance(d.get(k), (int, float))
        and int(d[k]) == FROZEN_OLD_MICROBATCH
        for k in mb_keys
    )
    return has_model and has_config and has_mb2


def main() -> int:
    print("=" * 136)
    print("STEP 27F — TIMESNET_B PRE-PERFORMANCE MICRO-BATCH PROTOCOL AMENDMENT")
    print("=" * 136)

    for p in (AMENDMENT_MANIFEST, AMENDMENT_REPORT, PROBE_AUDIT):
        if p.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing Step-27F artifact: {p}"
            )

    upstream = s27.verify_upstream()

    if (
        upstream["upstream25"]["step00"]["fingerprint"]
        != EXPECTED_CONFIGURATION_FINGERPRINT
    ):
        raise RuntimeError("Configuration fingerprint drift.")

    if upstream["global_final_test_state"] != "LOCKED_NOT_OPENED":
        raise RuntimeError("Global final-test state drift.")

    if (
        upstream["step26_freeze"]["stage1_runner_dryrun_fingerprint_sha256"]
        != EXPECTED_STEP26_FINGERPRINT
    ):
        raise RuntimeError("Step-26 runner fingerprint drift.")

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

    if sha256_file(STEP27_FROZEN_SCRIPT) != EXPECTED_STEP27_SOURCE_SHA256:
        raise RuntimeError("Frozen Step-27 source copy drift.")

    # Confirm the original frozen execution metadata contains TIMESNET_B mb=2.
    matching_nodes = [
        d for d in walk_nodes(upstream)
        if isinstance(d, dict) and looks_like_timesnetb_mb2_record(d)
    ]
    if not matching_nodes:
        raise RuntimeError(
            "Could not verify frozen TIMESNET_B physical microbatch=2 "
            "inside the verified upstream execution metadata."
        )

    # No accepted TIMESNET_B validation result may exist before amendment.
    completed_timesnetb = []
    cfg_root = RUN_ROOT / MODEL_ID / CONFIG_ID
    if cfg_root.exists():
        completed_timesnetb = sorted(cfg_root.rglob("fit_manifest.json"))
    if completed_timesnetb:
        raise RuntimeError(
            "Protocol amendment forbidden: completed TIMESNET_B fit(s) already exist:\n"
            + "\n".join(str(p) for p in completed_timesnetb)
        )

    if not TIMESNET_SOURCE.is_file():
        raise FileNotFoundError(TIMESNET_SOURCE)

    timesnet_text = TIMESNET_SOURCE.read_text(encoding="utf-8")
    required_snippets = [
        "def FFT_for_Period",
        "torch.fft.rfft(x, dim=1)",
        "frequency_list = abs(xf).mean(0).mean(-1)",
    ]
    for snippet in required_snippets:
        if snippet not in timesnet_text:
            raise RuntimeError(
                f"Required TimesNet batch-dependent FFT source snippet missing: {snippet}"
            )

    source_sha = sha256_file(Path(__file__).resolve())
    timesnet_source_sha = sha256_file(TIMESNET_SOURCE)

    print("[PASS] Frozen upstream verified.")
    print(f"[PASS] Original TIMESNET_B physical microbatch : {FROZEN_OLD_MICROBATCH}")
    print("[PASS] Completed TIMESNET_B fits              : 0")
    print("[PASS] TimesNet batch-averaged FFT semantics  : CONFIRMED")
    print("[PASS] Development arrays opened              : False")
    print("[PASS] Validation performance read            : False")
    print("[PASS] Final-test data opened                 : False")
    print()
    print(
        f"Re-probing amended candidate with synthetic data only: "
        f"logical_batch={LOGICAL_BATCH}, physical_microbatch={AMENDED_MICROBATCH}"
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
        physical_microbatch=AMENDED_MICROBATCH,
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
        np.random.SeedSequence([SEED, 27, 6, AMENDED_MICROBATCH])
    )
    x = rng.standard_normal((LOGICAL_BATCH, LOOKBACK, FEATURES)).astype(np.float32)
    y = rng.integers(
        0, 2, size=(LOGICAL_BATCH, OUTPUTS), dtype=np.int64
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
            physical_microbatch=AMENDED_MICROBATCH,
            device=device,
        )
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"Amended microbatch {AMENDED_MICROBATCH} failed synthetic CUDA probe: {exc}"
        ) from exc

    elapsed = time.perf_counter() - started

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    physical_vram = int(torch.cuda.get_device_properties(device).total_memory)

    peak_allocated_mib = peak_allocated / (1024 ** 2)
    peak_reserved_mib = peak_reserved / (1024 ** 2)
    physical_vram_mib = physical_vram / (1024 ** 2)

    if int(update["logical_batch_n"]) != LOGICAL_BATCH:
        raise RuntimeError("Synthetic logical-batch accounting drift.")
    if not math.isfinite(float(update["weighted_mean_bce"])):
        raise RuntimeError("Non-finite synthetic BCE.")
    if not math.isfinite(float(update["preclip_gradient_norm"])):
        raise RuntimeError("Non-finite synthetic gradient norm.")
    if peak_allocated_mib > ALLOCATED_MEMORY_BUDGET_MIB:
        raise RuntimeError(
            f"Peak allocated memory {peak_allocated_mib:.1f} MiB exceeds "
            f"frozen budget {ALLOCATED_MEMORY_BUDGET_MIB:.1f} MiB."
        )

    s26.import_step24().assert_single_repository_clean(repository)

    probe = {
        "step": "27F",
        "purpose": "TIMESNET_B synthetic mb=64 amendment qualification probe",
        "model_id": MODEL_ID,
        "config_id": CONFIG_ID,
        "seed": SEED,
        "logical_batch": LOGICAL_BATCH,
        "physical_microbatch": AMENDED_MICROBATCH,
        "microsteps_per_full_logical_batch": LOGICAL_BATCH // AMENDED_MICROBATCH,
        "candidate_learning_rate": learning_rate,
        "synthetic_weighted_bce": float(update["weighted_mean_bce"]),
        "preclip_gradient_norm": float(update["preclip_gradient_norm"]),
        "logical_update_seconds": elapsed,
        "peak_cuda_allocated_mib": peak_allocated_mib,
        "peak_cuda_reserved_mib": peak_reserved_mib,
        "physical_vram_mib": physical_vram_mib,
        "allocated_memory_budget_mib": ALLOCATED_MEMORY_BUDGET_MIB,
        "allocated_budget_pass": True,
        "development_arrays_opened": False,
        "validation_performance_computed": False,
        "final_test_opened_or_hashed": False,
    }

    probe_bytes = canonical_json_bytes(probe)
    probe_sha = sha256_bytes(probe_bytes)

    amendment = {
        "step": "27F",
        "status": "FROZEN_PRE_PERFORMANCE_PROTOCOL_AMENDMENT",
        "scope": {
            "model_id": MODEL_ID,
            "config_id": CONFIG_ID,
        },
        "change": {
            "physical_microbatch_old": FROZEN_OLD_MICROBATCH,
            "physical_microbatch_new": AMENDED_MICROBATCH,
            "logical_batch": LOGICAL_BATCH,
        },
        "classification": (
            "protocol amendment affecting TimesNet forward semantics because "
            "dominant FFT periods are selected from physical-batch-averaged amplitudes"
        ),
        "reason": (
            "original mb=2 real execution was computationally infeasible; "
            "amendment frozen before any completed TIMESNET_B fit or validation metric"
        ),
        "unchanged": [
            "model/configuration",
            "learning_rate provenance",
            "optimizer",
            "weight_decay",
            "loss",
            "logical batch size",
            "gradient weighting",
            "gradient clipping",
            "seed",
            "folds",
            "early stopping",
            "checkpoint criterion",
            "metric definitions",
            "configuration-selection protocol",
            "threshold protocol",
            "final-test lock",
        ],
        "timesnet_batch_semantics": {
            "source_path": str(TIMESNET_SOURCE),
            "source_sha256": timesnet_source_sha,
            "rfft_over_time": True,
            "frequency_amplitude_mean_over_physical_batch": True,
            "verified_snippet": "frequency_list = abs(xf).mean(0).mean(-1)",
        },
        "pre_performance_guard": {
            "completed_timesnetb_fit_manifests": 0,
            "development_arrays_opened_by_step27f": False,
            "validation_performance_computed_by_step27f": False,
            "final_test_opened_or_hashed_by_step27f": False,
        },
        "qualification_probe": {
            "audit_path": str(PROBE_AUDIT),
            "audit_sha256": probe_sha,
            "physical_microbatch": AMENDED_MICROBATCH,
            "allocated_memory_budget_mib": ALLOCATED_MEMORY_BUDGET_MIB,
            "peak_cuda_allocated_mib": peak_allocated_mib,
            "logical_update_seconds": elapsed,
        },
        "upstream": {
            "configuration_fingerprint": EXPECTED_CONFIGURATION_FINGERPRINT,
            "step26_runner_fingerprint": EXPECTED_STEP26_FINGERPRINT,
            "step27_source_fingerprint": EXPECTED_STEP27_SOURCE_FINGERPRINT,
            "step27_source_sha256": EXPECTED_STEP27_SOURCE_SHA256,
            "original_mb2_metadata_records_found": len(matching_nodes),
        },
        "source": {
            "step27f_script_path": str(Path(__file__).resolve()),
            "step27f_script_sha256": source_sha,
        },
        "authorization": (
            "Real TIMESNET_B training may use physical_microbatch=64 only through "
            "a subsequent compatibility runner that verifies this amendment."
        ),
    }

    amendment_bytes = canonical_json_bytes(amendment)
    amendment_sha = sha256_bytes(amendment_bytes)

    report = f"""STEP 27F — TIMESNET_B MICRO-BATCH PROTOCOL AMENDMENT
================================================================
STATUS: FROZEN_PRE_PERFORMANCE_PROTOCOL_AMENDMENT

Scope
-----
Model/config                 : {MODEL_ID}/{CONFIG_ID}
Physical microbatch          : {FROZEN_OLD_MICROBATCH} -> {AMENDED_MICROBATCH}
Logical batch                : {LOGICAL_BATCH} (unchanged)

Methodological classification
-----------------------------
This is NOT treated as a pure execution-only optimization.
The frozen official TimesNet source uses physical-batch-averaged FFT
amplitudes to select dominant periods:
    frequency_list = abs(xf).mean(0).mean(-1)

Therefore changing physical microbatch can change TimesNet forward semantics.

Pre-performance guard
---------------------
Completed TIMESNET_B fits    : 0
Development arrays opened    : False
Validation performance read  : False
Final-test data opened       : False

Synthetic qualification probe
-----------------------------
Microsteps/logical batch     : {LOGICAL_BATCH // AMENDED_MICROBATCH}
Learning rate                : {learning_rate:.12g}
Weighted BCE                 : {float(update["weighted_mean_bce"]):.8f}
Pre-clip gradient norm       : {float(update["preclip_gradient_norm"]):.8f}
Logical-update time          : {elapsed:.2f} s
Peak CUDA allocated          : {peak_allocated_mib:.1f} MiB
Allocated-memory budget      : {ALLOCATED_MEMORY_BUDGET_MIB:.1f} MiB
Peak CUDA reserved           : {peak_reserved_mib:.1f} MiB
Physical VRAM                : {physical_vram_mib:.1f} MiB

Frozen provenance
-----------------
Configuration fingerprint    : {EXPECTED_CONFIGURATION_FINGERPRINT}
Step-26 runner fingerprint   : {EXPECTED_STEP26_FINGERPRINT}
Step-27 source fingerprint   : {EXPECTED_STEP27_SOURCE_FINGERPRINT}
Step-27F source SHA-256      : {source_sha}
Probe audit SHA-256          : {probe_sha}
Amendment manifest SHA-256   : {amendment_sha}

No real Stage-1 TIMESNET_B training was performed by Step 27F.
A subsequent compatibility runner must verify this amendment before use.
"""

    # Manifest last, after probe and report, so its presence is the completion marker.
    atomic_write_bytes(PROBE_AUDIT, probe_bytes)
    atomic_write_bytes(AMENDMENT_REPORT, report.encode("utf-8"))
    atomic_write_bytes(AMENDMENT_MANIFEST, amendment_bytes)

    print()
    print("=" * 136)
    print("STEP 27F STATUS: PASS — PRE-PERFORMANCE AMENDMENT FROZEN")
    print("=" * 136)
    print(f"TIMESNET_B physical microbatch : {FROZEN_OLD_MICROBATCH} -> {AMENDED_MICROBATCH}")
    print(f"Logical batch                  : {LOGICAL_BATCH} (UNCHANGED)")
    print(f"Synthetic update time          : {elapsed:.2f} s")
    print(f"Peak CUDA allocated            : {peak_allocated_mib:.1f} MiB")
    print(f"Allocated-memory budget        : {ALLOCATED_MEMORY_BUDGET_MIB:.1f} MiB")
    print("TimesNet batch-dependent FFT   : CONFIRMED")
    print("Completed TIMESNET_B fits      : 0")
    print("Validation performance read    : False")
    print("Final-test data opened         : False")
    print(f"Amendment manifest             : {AMENDMENT_MANIFEST}")
    print(f"Amendment manifest SHA-256     : {amendment_sha}")
    print("=" * 136)
    print(
        "STOP: do not run real TIMESNET_B yet. "
        "Next step is a separate compatibility runner that consumes this frozen amendment."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStep 27F interrupted by user. No amendment manifest should be trusted unless PASS was printed.")
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print()
        print("=" * 136)
        print("STEP 27F FAILED")
        print("=" * 136)
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        print("=" * 136)
        raise SystemExit(1)
