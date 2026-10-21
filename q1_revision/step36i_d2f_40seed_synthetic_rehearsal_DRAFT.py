#!/usr/bin/env python3
"""Step 36I-D2f: synthetic-only, complete frozen inference rehearsal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
EXP = (
    WORK.parent
    / "stock_q1_revision_project"
    / "experiments/q1_revision_v3_clean_2016_2026"
)

MAN = EXP / "manifests"
AUD = EXP / "audits"

OUT = AUD / "step36i_d2f_full_synthetic_inference_draft"

KEYS = (
    AUD / "step36i_protocol_review_draft"
    / "36i_candidate_sequence_eligible_keys_DRAFT.csv"
)

D2E = (
    AUD / "step36i_d2e_inference_inventory_draft"
    / "36i_d2e_frozen_inference_inventory_DRAFT.json"
)

STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"

SOURCE = WORK / (
    "step36_one_time_final_test_materialization_inference_scoring_v1.py"
)

B2 = (
    AUD / "step36i_b2_v2_synthetic_gpu_preflight_draft"
    / "36i_b2_v2_synthetic_gpu_results_DRAFT.json"
)

ACCESS = MAN / "36_final_test_access_consumption_state.json"
PLAN = AUD / "35_final_test_inference_plan.csv"
THRESHOLDS = AUD / "35_final_test_threshold_contract.csv"

EXPECTED = {
    KEYS:
        "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
    D2E:
        "5880f8c4329c91b772e121d097462ddf8af8124eb4bf2910c27bed35d165861e",
    STEP35:
        "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
    SOURCE:
        "6de04ba7e167449101e2e16bab47ca4ce088f11284c4cbc93b61a6cfbb979f12",
    B2:
        "e0546ac4fdd25b3ce426903aae7752ac81f6d7ebaf10a24c33e946b31b55b70d",
}

MODEL = {
    "DLINEAR": ("DLINEAR_B", 128),
    "NLINEAR": ("NLINEAR_B", 128),
    "PATCHTST": ("PATCHTST_B", 64),
    "TIMESNET": ("TIMESNET_B", 64),
    "ITRANSFORMER": ("ITRANSFORMER_A", 128),
    "FREDFORMER": ("FREDFORMER_B", 16),
    "FRNET": ("FRNET_A", 64),
    "MLF": ("MLF_A", 32),
}

SEEDS = (42, 43, 44, 45, 46)
N = 5300


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def guard():
    for path, expected in EXPECTED.items():
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(
                f"Frozen source or evidence mismatch: {path}"
            )

    marker = json.loads(
        ACCESS.read_text(encoding="utf-8")
    )

    if (
        int(marker["raw_test_filesystem_open_count_completed"]) != 2
        or marker["final_test_scored"] is not False
    ):
        raise RuntimeError(
            "Final-test access state has changed. STOP."
        )

    inventory = json.loads(
        D2E.read_text(encoding="utf-8")
    )

    if (
        inventory["status"] != "PASS_INTEGRITY_ONLY"
        or len(inventory["records"]) != 40
        or inventory["third_raw_test_read_authorized"] is not False
    ):
        raise RuntimeError("Frozen inventory state mismatch.")

    plan = pd.read_csv(PLAN)

    if (
        len(plan) != 40
        or plan[["model_id", "seed"]].duplicated().any()
    ):
        raise RuntimeError("Frozen inference plan mismatch.")

    actual_pairs = {
        (str(m), int(s))
        for m, s in zip(plan["model_id"], plan["seed"])
    }

    expected_pairs = {
        (m, s) for m in MODEL for s in SEEDS
    }

    if actual_pairs != expected_pairs:
        raise RuntimeError("Frozen model-seed pairs mismatch.")

    if not plan["inference_shuffle"].astype(str).str.lower().isin(
        ["false", "0", "no"]
    ).all():
        raise RuntimeError("Frozen inference unexpectedly shuffles rows.")

    if not plan["gradient_tracking"].astype(str).str.lower().isin(
        ["false", "0", "no"]
    ).all():
        raise RuntimeError("Frozen inference enables gradient tracking.")

    if plan["inference_row_order"].nunique() != 1:
        raise RuntimeError("Frozen row-order contract inconsistent.")

    keys = pd.read_csv(KEYS, parse_dates=["Date"])

    if (
        len(keys) != N
        or keys["Date"].nunique() != 265
        or keys[["Ticker", "Date"]].duplicated().any()
    ):
        raise RuntimeError("Synthetic layout keys mismatch.")

    sorted_keys = keys.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    if not sorted_keys.equals(keys):
        raise RuntimeError("Candidate row order changed.")

    if not keys.groupby("Date")["Ticker"].nunique().eq(20).all():
        raise RuntimeError("Unbalanced candidate panel.")

    return inventory, plan


def fake_batch(start, stop):
    rows = np.arange(start, stop, dtype=np.float32)
    periods = np.arange(60, dtype=np.float32)
    feature_ids = np.arange(63, dtype=np.float32)

    x = np.sin(
        rows[:, None, None] * 0.014
        + periods[None, :, None] * 0.035
        + feature_ids[None, None, :] * 0.077
    ).astype(np.float32)

    x += (rows[:, None, None] % 20) * np.float32(0.002)

    return np.ascontiguousarray(x, dtype=np.float32)


def worker(model_id, seed):
    inventory, plan = guard()

    config, batch = MODEL[model_id]

    found = [
        record for record in inventory["records"]
        if record["model_id"] == model_id
        and int(record["seed"]) == seed
    ]

    if len(found) != 1:
        raise RuntimeError("Non-unique checkpoint inventory.")

    record = found[0]

    selected = plan.loc[
        plan["model_id"].eq(model_id)
        & plan["seed"].astype(int).eq(seed)
    ]

    if len(selected) != 1:
        raise RuntimeError("Non-unique inference plan entry.")

    selected = selected.iloc[0]

    if (
        record["config_id"] != config
        or record["physical_microbatch_size"] != batch
        or str(selected["config_id"]) != config
        or int(selected["physical_inference_microbatch"]) != batch
        or str(selected["checkpoint_sha256"])
            != record["checkpoint_sha256"]
        or str(selected["fit_manifest_sha256"])
            != record["fit_manifest_sha256"]
    ):
        raise RuntimeError("Frozen inference parameter mismatch.")

    checkpoint = Path(record["checkpoint_path"])

    if sha(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint SHA256 drift.")

    import torch
    import step26_stage1_runner_dryrun_v3 as s26

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    s26.set_training_determinism(seed)

    device = torch.device("cuda:0")

    # Explicit initialization, verified in Step 36I-B2b.
    torch.cuda.init()

    probe = torch.ones(
        (1,), dtype=torch.float32, device=device
    )
    torch.cuda.synchronize(device)
    del probe

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    network, _, _ = s26.build_candidate_model(
        model_id=model_id,
        config_id=config,
        physical_microbatch=batch,
        device=device,
    )

    weights = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    network.load_state_dict(
        weights["model_state_dict"],
        strict=True,
    )
    network.eval()

    predictions = []
    start_time = time.perf_counter()

    with torch.no_grad():
        for start in range(0, N, batch):
            stop = min(start + batch, N)

            x = torch.from_numpy(
                fake_batch(start, stop)
            ).to(device)

            logits, _ = network(x)

            if (
                tuple(logits.shape) != (stop - start, 3)
                or not bool(torch.isfinite(logits).all().item())
            ):
                raise RuntimeError(
                    f"Invalid synthetic logits: "
                    f"{model_id}/seed{seed}/row{start}"
                )

            probabilities = (
                torch.sigmoid(logits)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if (
                not np.isfinite(probabilities).all()
                or np.any(
                    (probabilities < 0) | (probabilities > 1)
                )
            ):
                raise RuntimeError(
                    "Invalid synthetic probabilities."
                )

            predictions.append(probabilities)

            del x, logits, probabilities

    torch.cuda.synchronize(device)

    seconds = round(
        time.perf_counter() - start_time, 3
    )

    array = np.concatenate(predictions, axis=0)

    if array.shape != (N, 3):
        raise RuntimeError("Synthetic row-count mismatch.")

    path = (
        OUT / "synthetic_seed_probabilities"
        / f"SYNTHETIC_ONLY_{model_id}_seed{seed}.npz"
    )

    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            synthetic_probability=array,
            synthetic_row_index=np.arange(
                N, dtype=np.int64
            ),
        )

    result = {
        "model_id": model_id,
        "config_id": config,
        "seed": seed,
        "physical_batch": batch,
        "final_partial_batch": N % batch,
        "synthetic_rows": N,
        "horizons": 3,
        "checkpoint_sha256": record["checkpoint_sha256"],
        "synthetic_prediction_path": str(path),
        "synthetic_prediction_sha256": sha(path),
        "inference_seconds": seconds,
        "peak_cuda_allocated_mib": round(
            torch.cuda.max_memory_allocated(device) / 1048576, 2
        ),
        "peak_cuda_reserved_mib": round(
            torch.cuda.max_memory_reserved(device) / 1048576, 2
        ),
        "real_test_opened": False,
    }

    print(
        "SYNTHETIC_WORKER_RESULT=" + json.dumps(result),
        flush=True,
    )


def orchestrate():
    inventory, plan = guard()

    if OUT.exists():
        raise RuntimeError(
            "Existing D2f output directory. Refusing overwrite."
        )

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "synthetic_seed_probabilities").mkdir()
    (OUT / "synthetic_ensembles").mkdir()
    (OUT / "worker_logs").mkdir()

    results = []

    print("=" * 110, flush=True)
    print(
        "STEP 36I-D2f — FULL 40-CHECKPOINT SYNTHETIC GPU REHEARSAL",
        flush=True,
    )
    print("=" * 110, flush=True)
    print("Synthetic evaluation observations:", N)
    print("Architectures:", len(MODEL))
    print("Seeds:", SEEDS)
    print("Physical row order: Date, Ticker")
    print("Raw sealed test access: PROHIBITED")

    for model_id, (_, batch) in MODEL.items():
        for seed in SEEDS:
            print(
                f"\nRUN {model_id:14s} "
                f"seed={seed} batch={batch} tail={N % batch}",
                flush=True,
            )

            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--synthetic-worker",
                    model_id,
                    str(seed),
                ],
                cwd=str(WORK),
                env=env,
                capture_output=True,
                text=True,
            )

            log_path = (
                OUT / "worker_logs"
                / f"{model_id}_seed{seed}_SYNTHETIC_log.txt"
            )

            with log_path.open("x", encoding="utf-8") as handle:
                handle.write(
                    process.stdout + process.stderr
                )

            matched = [
                line.split("=", 1)[1]
                for line in process.stdout.splitlines()
                if line.startswith("SYNTHETIC_WORKER_RESULT=")
            ]

            if process.returncode != 0 or len(matched) != 1:
                failure = {
                    "status": "FAILED_SYNTHETIC_ONLY",
                    "failed_model": model_id,
                    "failed_seed": seed,
                    "completed_runs": results,
                    "worker_log": str(log_path),
                    "heldout_access": False,
                }

                failure_path = (
                    OUT / "36i_d2f_failure_DRAFT.json"
                )

                with failure_path.open("x", encoding="utf-8") as handle:
                    json.dump(failure, handle, indent=2)

                print(
                    process.stdout + process.stderr,
                    flush=True,
                )

                raise RuntimeError(
                    f"Synthetic inference failed: "
                    f"{model_id}/seed{seed}"
                )

            result = json.loads(matched[0])

            if (
                result["model_id"] != model_id
                or int(result["seed"]) != seed
                or result["physical_batch"] != batch
                or result["synthetic_rows"] != N
            ):
                raise RuntimeError("Worker metadata mismatch.")

            results.append(result)

            print(
                f"PASS — {model_id:14s} "
                f"seed={seed} "
                f"runtime={result['inference_seconds']}s "
                f"peak={result['peak_cuda_reserved_mib']} MiB",
                flush=True,
            )

    if len(results) != 40:
        raise RuntimeError("Incorrect checkpoint inference count.")

    print("\n[2] FIVE-SEED SYNTHETIC PROBABILITY ENSEMBLES")

    thresholds = pd.read_csv(THRESHOLDS)

    threshold_column = (
        "selected_threshold"
        if "selected_threshold" in thresholds.columns
        else "threshold"
    )

    if (
        len(thresholds) != 24
        or len(
            thresholds[
                ["model_id", "horizon_days"]
            ].drop_duplicates()
        ) != 24
    ):
        raise RuntimeError("Frozen threshold family mismatch.")

    ensemble_records = []

    for model_id in MODEL:
        probability_arrays = []

        for seed in SEEDS:
            result = next(
                record for record in results
                if record["model_id"] == model_id
                and record["seed"] == seed
            )

            path = Path(result["synthetic_prediction_path"])

            if sha(path) != result["synthetic_prediction_sha256"]:
                raise RuntimeError("Synthetic artifact SHA drift.")

            with np.load(
                path, allow_pickle=False
            ) as archive:
                probabilities = archive[
                    "synthetic_probability"
                ]
                row_index = archive[
                    "synthetic_row_index"
                ]

            if (
                probabilities.shape != (N, 3)
                or not np.array_equal(
                    row_index, np.arange(N)
                )
            ):
                raise RuntimeError(
                    "Synthetic probability row-order mismatch."
                )

            probability_arrays.append(probabilities)

        ensemble = np.mean(
            np.stack(
                probability_arrays, axis=0
            ).astype(np.float64),
            axis=0,
        ).astype(np.float32)

        assert ensemble.shape == (N, 3)
        assert np.isfinite(ensemble).all()

        model_thresholds = thresholds.loc[
            thresholds["model_id"].eq(model_id)
        ]

        for horizon_index, horizon in enumerate((1, 3, 5)):
            selected = model_thresholds.loc[
                model_thresholds[
                    "horizon_days"
                ].astype(int).eq(horizon)
            ]

            if len(selected) != 1:
                raise RuntimeError(
                    "Missing or duplicate frozen threshold."
                )

            threshold = float(
                selected.iloc[0][threshold_column]
            )

            assert 0 <= threshold <= 1

            decisions = (
                ensemble[:, horizon_index] >= threshold
            )

            assert decisions.shape == (N,)
            assert decisions.dtype == np.bool_

        path = (
            OUT / "synthetic_ensembles"
            / f"SYNTHETIC_ONLY_{model_id}_five_seed_mean.npz"
        )

        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                synthetic_ensemble_probability=ensemble,
                synthetic_row_index=np.arange(
                    N, dtype=np.int64
                ),
            )

        ensemble_records.append({
            "model_id": model_id,
            "seed_count": 5,
            "synthetic_shape": [N, 3],
            "frozen_thresholds_exercised": 3,
            "synthetic_ensemble_sha256": sha(path),
        })

        print(
            f"PASS — {model_id:14s} "
            "five-seed mean and three frozen thresholds",
            flush=True,
        )

    print("\n[3] PRESERVE SYNTHETIC-ONLY AUDIT")

    report = {
        "status": "PASS_SYNTHETIC_ONLY",
        "stage": "36I-D2f",
        "purpose": (
            "Complete frozen 40-checkpoint GPU rehearsal "
            "using entirely fabricated predictors."
        ),
        "script_sha256": sha(
            Path(__file__).resolve()
        ),
        "d2e_inventory_sha256": sha(D2E),
        "candidate_keys_sha256": sha(KEYS),
        "step35_manifest_sha256": sha(STEP35),
        "architectures": 8,
        "seeds": list(SEEDS),
        "checkpoints_tested": 40,
        "synthetic_rows_per_checkpoint": N,
        "lookback": 60,
        "features": 63,
        "horizons": [1, 3, 5],
        "five_seed_mean_verified": True,
        "frozen_thresholds_exercised": 24,
        "synthetic_worker_results": results,
        "synthetic_ensembles": ensemble_records,
        "real_test_features_read": False,
        "real_test_labels_read": False,
        "real_test_inference": False,
        "real_final_test_metrics": False,
        "bootstrap_performed_here": False,
        "postfreeze_amendment_approved": False,
        "third_test_read_authorized": False,
    }

    report_path = (
        OUT / "36i_d2f_full_synthetic_rehearsal_DRAFT.json"
    )

    with report_path.open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print("\n" + "=" * 110)
    print("STEP 36I-D2f — SYNTHETIC GPU REHEARSAL PASS")
    print("Checkpoint inferences: 40/40")
    print("Five-seed ensembles: 8/8")
    print("Frozen thresholds exercised: 24/24")
    print("Report:", report_path)
    print("Report SHA256:", sha(report_path))
    print("Raw final-test opened: FALSE")
    print("Production final-test predictions: FALSE")
    print("Further test access authorized: FALSE")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--offline-synthetic",
        action="store_true",
    )

    mode.add_argument(
        "--synthetic-worker",
        nargs=2,
        metavar=("MODEL", "SEED"),
    )

    args = parser.parse_args()

    if args.offline_synthetic:
        orchestrate()
    else:
        model_id, seed_text = args.synthetic_worker

        if (
            model_id not in MODEL
            or seed_text not in tuple(
                str(seed) for seed in SEEDS
            )
        ):
            raise SystemExit(
                "Only the eight frozen models "
                "and five seeds are permitted."
            )

        worker(model_id, int(seed_text))


if __name__ == "__main__":
    main()
