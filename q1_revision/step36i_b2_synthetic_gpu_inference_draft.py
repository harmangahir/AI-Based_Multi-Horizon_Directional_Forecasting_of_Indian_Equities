#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
EXP = (
    WORK.parent
    / "stock_q1_revision_project"
    / "experiments"
    / "q1_revision_v3_clean_2016_2026"
)
MAN = EXP / "manifests"
AUD = EXP / "audits"
DRAFT = AUD / "step36i_protocol_review_draft"
REPORT_DIR = AUD / "step36i_b2_synthetic_gpu_preflight_draft"

STEP36 = WORK / (
    "step36_one_time_final_test_materialization_inference_scoring_v1.py"
)
KEYS = DRAFT / "36i_candidate_sequence_eligible_keys_DRAFT.csv"
PROPOSAL = DRAFT / "36i_protocol_deviation_DRAFT.json"
B1_REPORT = (
    AUD / "step36i_b1_synthetic_preflight_draft"
    / "36i_b1_synthetic_sequences_DRAFT.json"
)

N = 5300
SEED = 42

MODELS = [
    "DLINEAR",
    "NLINEAR",
    "PATCHTST",
    "TIMESNET",
    "ITRANSFORMER",
    "FREDFORMER",
    "FRNET",
    "MLF",
]

BATCHES = {
    "DLINEAR": 128,
    "NLINEAR": 128,
    "PATCHTST": 64,
    "TIMESNET": 64,
    "ITRANSFORMER": 128,
    "FREDFORMER": 16,
    "FRNET": 64,
    "MLF": 32,
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_guard():
    expected = {
        STEP36:
            "6de04ba7e167449101e2e16bab47ca4ce088f11284c4cbc93b61a6cfbb979f12",
        KEYS:
            "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
        PROPOSAL:
            "ce89aebe2467fcb2003639aa00256d0670ce2ac2ac96fd140041dfd7a1895df8",
        B1_REPORT:
            "108db80959940d143a0c052e7116022bc24b4fbd118ce8b53a9facdbc39d1967",
        MAN / "35_final_test_evaluation_contract_freeze_manifest.json":
            "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
    }

    for path, digest in expected.items():
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"Missing or modified frozen input: {path}")

    marker = json.loads(
        (MAN / "36_final_test_access_consumption_state.json")
        .read_text(encoding="utf-8")
    )
    if (
        int(marker["raw_test_filesystem_open_count_completed"]) != 2
        or marker["final_test_scored"] is not False
    ):
        raise RuntimeError("Unexpected final-test access state. STOP.")

    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    if (
        proposal["status"] != "DRAFT_NOT_APPROVED"
        or proposal["future_actions_authorized"] is not False
    ):
        raise RuntimeError("Protocol draft state changed. STOP.")

def synthetic_batch(start, count):
    rows = np.arange(start, start + count, dtype=np.float32)
    times = np.arange(60, dtype=np.float32)
    feats = np.arange(63, dtype=np.float32)

    x = np.sin(
        rows[:, None, None] * 0.014
        + times[None, :, None] * 0.035
        + feats[None, None, :] * 0.077
    ).astype(np.float32)

    x += (rows[:, None, None] % 20) * np.float32(0.002)
    assert x.shape == (count, 60, 63)
    assert np.isfinite(x).all()
    return x

def load_step36_module():
    spec = importlib.util.spec_from_file_location(
        "step36i_b2_source_probe", STEP36
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def worker(model_id):
    verify_guard()

    import pandas as pd
    import torch
    import step26_stage1_runner_dryrun_v3 as s26

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    mod = load_step36_module()

    register = pd.read_csv(mod.STEP34_REGISTER)
    selected = register.loc[
        register["model_id"].eq(model_id)
        & register["seed"].astype(int).eq(SEED)
    ]

    if len(selected) != 1:
        raise RuntimeError(f"Non-unique frozen checkpoint: {model_id}")

    fit = selected.iloc[0]
    config_id = str(fit["config_id"])
    physical = int(fit["physical_microbatch_size"])

    if config_id != mod.EXPECTED_CONFIG[model_id]:
        raise RuntimeError("Frozen configuration mismatch.")
    if physical != BATCHES[model_id]:
        raise RuntimeError("Frozen physical microbatch mismatch.")

    fit_manifest = Path(fit["fit_manifest_path"])
    mod.require_file_sha(
        fit_manifest,
        str(fit["fit_manifest_sha256"]),
        "Frozen final-development fit manifest"
    )

    fm = mod.load_json(fit_manifest)
    cp_record = fm["artifacts"]["final_checkpoint.pt"]
    checkpoint = Path(
        cp_record.get(
            "path",
            fit_manifest.parent / "final_checkpoint.pt"
        )
    )
    mod.require_file_sha(
        checkpoint,
        cp_record["sha256"],
        "Frozen final-development checkpoint"
    )

    s26.set_training_determinism(SEED)
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model, _, _ = s26.build_candidate_model(
        model_id=model_id,
        config_id=config_id,
        physical_microbatch=physical,
        device=device,
    )

    weights = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(
        weights["model_state_dict"],
        strict=True,
    )
    model.eval()

    tail = N % physical
    if tail == 0:
        raise RuntimeError("Unexpected zero-tail test case.")

    cases = [
        ("full_batch_1", 0, physical),
        ("full_batch_2", physical, physical),
        ("final_partial_batch", N - tail, tail),
    ]

    times = {}
    torch.cuda.synchronize()

    with torch.no_grad():
        for label, start, count in cases:
            x = synthetic_batch(start, count)
            xb = torch.from_numpy(x).to(
                device=device,
                dtype=torch.float32,
                non_blocking=False,
            )

            t0 = time.perf_counter()
            logits, _ = model(xb)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            if tuple(logits.shape) != (count, 3):
                raise RuntimeError(
                    f"Incorrect logits shape: {tuple(logits.shape)}"
                )

            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError("Non-finite synthetic logits.")

            probabilities = torch.sigmoid(logits)
            if not bool(torch.isfinite(probabilities).all().item()):
                raise RuntimeError("Non-finite synthetic probabilities.")

            times[label] = round(elapsed, 6)
            del xb, logits, probabilities

    result = {
        "status": "PASS_SYNTHETIC_ONLY",
        "model_id": model_id,
        "config_id": config_id,
        "seed": SEED,
        "physical_microbatch": physical,
        "full_batch_cases": 2,
        "final_partial_batch_size": tail,
        "output_width": 3,
        "times_seconds": times,
        "peak_cuda_allocated_mib": round(
            torch.cuda.max_memory_allocated(device) / 1048576, 2
        ),
        "peak_cuda_reserved_mib": round(
            torch.cuda.max_memory_reserved(device) / 1048576, 2
        ),
        "checkpoint_sha256": sha(checkpoint),
        "heldout_data_accessed": False,
        "metrics_computed": False,
    }

    print("SYNTHETIC_GPU_RESULT=" + json.dumps(result))

def orchestrate():
    print("=" * 110)
    print("STEP 36I-B2 — EIGHT-ARCHITECTURE SYNTHETIC GPU PREFLIGHT")
    print("=" * 110)

    verify_guard()

    if REPORT_DIR.exists():
        raise RuntimeError("Existing draft report directory; refusing overwrite.")

    REPORT_DIR.mkdir(parents=True, exist_ok=False)

    results = []
    report_path = REPORT_DIR / "36i_b2_synthetic_gpu_results_DRAFT.json"

    for model_id in MODELS:
        physical = BATCHES[model_id]
        tail = N % physical

        print(
            f"\nTEST {model_id:14s} "
            f"seed={SEED} batch={physical} tail={tail}",
            flush=True,
        )

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                model_id,
            ],
            cwd=str(WORK),
            env=env,
            capture_output=True,
            text=True,
        )

        output = proc.stdout + proc.stderr
        (REPORT_DIR / f"{model_id}_worker_log.txt").write_text(
            output, encoding="utf-8"
        )

        if proc.returncode != 0:
            report_path.write_text(
                json.dumps({
                    "status": "FAILED_SYNTHETIC_PREFLIGHT",
                    "completed_models": results,
                    "failed_model": model_id,
                    "additional_test_access": False,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            print(output)
            raise RuntimeError(
                f"Synthetic GPU preflight failed for {model_id}. "
                "Do not rerun without reviewing the error."
            )

        result_lines = [
            line.split("=", 1)[1]
            for line in proc.stdout.splitlines()
            if line.startswith("SYNTHETIC_GPU_RESULT=")
        ]

        if len(result_lines) != 1:
            raise RuntimeError("Missing or ambiguous worker result.")

        result = json.loads(result_lines[0])
        assert result["status"] == "PASS_SYNTHETIC_ONLY"
        assert result["model_id"] == model_id
        assert result["physical_microbatch"] == physical
        assert result["final_partial_batch_size"] == tail

        results.append(result)

        print(
            f"PASS | tail={tail} | "
            f"peak allocated={result['peak_cuda_allocated_mib']} MiB | "
            f"peak reserved={result['peak_cuda_reserved_mib']} MiB",
            flush=True,
        )

    report = {
        "status": "PASS_SYNTHETIC_ONLY",
        "stage": "36I-B2",
        "source_sha256": sha(Path(__file__).resolve()),
        "recovered_step36_sha256": sha(STEP36),
        "candidate_keys_sha256": sha(KEYS),
        "sample_size_for_batch_planning": N,
        "model_families_verified": len(results),
        "seed_verified": SEED,
        "results": results,
        "all_40_checkpoints_tested": False,
        "real_final_test_read": False,
        "real_predictions_generated": False,
        "performance_metrics_computed": False,
        "recovery_execution_authorized": False,
    }

    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("STEP 36I-B2 SYNTHETIC GPU PREFLIGHT PASS")
    print("Model families:", len(results))
    print("Seed per family:", SEED)
    print("Report:", report_path)
    print("Report SHA256:", sha(report_path))
    print("Raw final-test parquet accessed: FALSE")
    print("Frozen source/checkpoints modified: FALSE")
    print("Held-out metrics computed: FALSE")
    print("Further test access authorized: FALSE")
    print("=" * 110)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODELS)
    args = parser.parse_args()

    if args.worker:
        worker(args.worker)
    else:
        orchestrate()

if __name__ == "__main__":
    main()
