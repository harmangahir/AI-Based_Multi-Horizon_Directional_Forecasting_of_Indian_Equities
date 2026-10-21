#!/usr/bin/env python3
"""Step 36I-D2g: offline synthetic scoring and bootstrap integration only."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
EXP = (
    WORK.parent / "stock_q1_revision_project"
    / "experiments/q1_revision_v3_clean_2016_2026"
)
AUD = EXP / "audits"
MAN = EXP / "manifests"

D2F = (
    AUD / "step36i_d2f_full_synthetic_inference_draft"
)
D2F_REPORT = D2F / "36i_d2f_full_synthetic_rehearsal_DRAFT.json"
D2E_REPORT = (
    AUD / "step36i_d2e_inference_inventory_draft"
    / "36i_d2e_frozen_inference_inventory_DRAFT.json"
)
KEYS = (
    AUD / "step36i_protocol_review_draft"
    / "36i_candidate_sequence_eligible_keys_DRAFT.csv"
)
THRESHOLDS = AUD / "35_final_test_threshold_contract.csv"
SOURCE = WORK / (
    "step36_one_time_final_test_materialization_inference_scoring_v1.py"
)
STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"
MARKER = MAN / "36_final_test_access_consumption_state.json"

OUT = AUD / "step36i_d2g_synthetic_scoring_draft"

EXPECTED = {
    D2F_REPORT:
        "197ad07632bfdcd5134ca959c132a896a75711986500d72d7713d681e56370ab",
    D2E_REPORT:
        "5880f8c4329c91b772e121d097462ddf8af8124eb4bf2910c27bed35d165861e",
    KEYS:
        "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
    SOURCE:
        "6de04ba7e167449101e2e16bab47ca4ce088f11284c4cbc93b61a6cfbb979f12",
    STEP35:
        "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
}

MODELS = (
    "DLINEAR", "NLINEAR", "PATCHTST", "TIMESNET",
    "ITRANSFORMER", "FREDFORMER", "FRNET", "MLF"
)
SEEDS = (42, 43, 44, 45, 46)
HORIZONS = (1, 3, 5)
N = 5300
DAYS = 265
B = 2000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def balanced_accuracy_from_counts(total):
    tp, tn, pos, neg = [
        total[..., i].astype(np.float64)
        for i in range(4)
    ]
    if np.any(pos <= 0) or np.any(neg <= 0):
        raise RuntimeError("Synthetic bootstrap class denominator zero.")
    return 0.5 * (tp / pos + tn / neg)


def bh_adjust(p):
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(
        (ranked * len(p) / ranks)[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def bootstrap_indices(L, rng, count):
    result = np.empty((count, DAYS), dtype=np.int64)
    for b in range(count):
        for offset, length in ((0, 48), (48, 217)):
            filled = 0
            while filled < length:
                start = int(rng.integers(length))
                take = min(L, length - filled)
                block = (
                    offset +
                    (start + np.arange(take, dtype=np.int64)) % length
                )
                result[b, offset + filled:offset + filled + take] = block
                filled += take
    return result


def main():
    print("=" * 105)
    print("STEP 36I-D2g — OFFLINE SYNTHETIC SCORING / BOOTSTRAP INTEGRATION")
    print("=" * 105)

    print("\n[1] VERIFY SCIENTIFIC FREEZES AND ACCESS STATE")

    for path, digest in EXPECTED.items():
        assert path.is_file(), f"Missing: {path}"
        assert sha(path) == digest, f"SHA mismatch: {path.name}"
        print("PASS:", path.name)

    marker = json.loads(MARKER.read_text(encoding="utf-8"))
    assert int(marker["raw_test_filesystem_open_count_completed"]) == 2
    assert marker["final_test_scored"] is False

    assert not OUT.exists(), "D2g output already exists. STOP."

    d2f = json.loads(D2F_REPORT.read_text(encoding="utf-8"))
    assert d2f["status"] == "PASS_SYNTHETIC_ONLY"
    assert d2f["checkpoints_tested"] == 40
    assert len(d2f["synthetic_worker_results"]) == 40
    assert len(d2f["synthetic_ensembles"]) == 8
    assert d2f["third_test_read_authorized"] is False

    inventory = json.loads(D2E_REPORT.read_text(encoding="utf-8"))
    assert sha(THRESHOLDS) == inventory["threshold_contract_sha256"]

    print("PASS — complete D2f synthetic inventory verified")
    print("PASS — no additional raw-test access")

    print("\n[2] VERIFY PHYSICAL EVALUATION ROW ORDER")

    keys = pd.read_csv(KEYS, parse_dates=["Date"])
    assert len(keys) == N
    assert keys["Date"].nunique() == DAYS
    assert keys.groupby("Date")["Ticker"].nunique().eq(20).all()
    assert keys.equals(
        keys.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    )

    date_frame = (
        keys[["Date", "continuity_segment_id"]]
        .drop_duplicates()
        .sort_values("Date")
        .reset_index(drop=True)
    )
    assert date_frame["continuity_segment_id"].tolist() == (
        [0] * 48 + [1] * 217
    )

    print("PASS — 265 dates, 20 stocks each, two continuity segments")

    print("\n[3] LOAD ACTUAL STEP-36 SCORING IMPLEMENTATION")

    spec = importlib.util.spec_from_file_location(
        "step36i_d2g_source_probe", SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.binary_diagnostics)
    print("PASS — original scoring function imported without execution")

    print("\n[4] INDEPENDENTLY RECONSTRUCT EIGHT SYNTHETIC ENSEMBLES")

    worker_records = {
        (r["model_id"], int(r["seed"])): r
        for r in d2f["synthetic_worker_results"]
    }
    assert len(worker_records) == 40

    ensemble_records = {
        r["model_id"]: r
        for r in d2f["synthetic_ensembles"]
    }
    assert set(ensemble_records) == set(MODELS)

    thresholds = pd.read_csv(THRESHOLDS)
    tcol = (
        "selected_threshold"
        if "selected_threshold" in thresholds.columns
        else "threshold"
    )
    assert len(thresholds) == 24
    assert not thresholds[["model_id", "horizon_days"]].duplicated().any()

    rng = np.random.default_rng(20261013)
    y = rng.integers(0, 2, size=(N, 3), dtype=np.int8)

    counts = np.zeros((DAYS, 24, 4), dtype=np.int32)
    checks = []

    for model_index, model_id in enumerate(MODELS):
        seeds = []

        for seed in SEEDS:
            rec = worker_records[(model_id, seed)]
            path = Path(rec["synthetic_prediction_path"])

            assert path.is_file()
            assert sha(path) == rec["synthetic_prediction_sha256"]

            with np.load(path, allow_pickle=False) as data:
                prob = data["synthetic_probability"].copy()
                row_index = data["synthetic_row_index"].copy()

            assert prob.shape == (N, 3)
            assert np.array_equal(row_index, np.arange(N))
            assert np.isfinite(prob).all()
            seeds.append(prob)

        ensemble = np.mean(
            np.stack(seeds).astype(np.float64),
            axis=0
        ).astype(np.float32)

        saved_path = (
            D2F / "synthetic_ensembles"
            / f"SYNTHETIC_ONLY_{model_id}_five_seed_mean.npz"
        )
        assert sha(saved_path) == (
            ensemble_records[model_id]["synthetic_ensemble_sha256"]
        )

        with np.load(saved_path, allow_pickle=False) as data:
            saved = data["synthetic_ensemble_probability"]
            saved_index = data["synthetic_row_index"]

        assert np.array_equal(ensemble, saved)
        assert np.array_equal(saved_index, np.arange(N))

        for j, horizon in enumerate(HORIZONS):
            test_id = model_index * 3 + j

            selected = thresholds.loc[
                thresholds["model_id"].eq(model_id)
                & thresholds["horizon_days"].astype(int).eq(horizon)
            ]
            assert len(selected) == 1
            threshold = float(selected.iloc[0][tcol])

            truth = y[:, j].astype(np.int64)
            probability = ensemble[:, j].astype(np.float64)
            predicted = (probability >= threshold).astype(np.int64)

            diag = module.binary_diagnostics(
                truth, probability, threshold
            )

            assert all(
                np.isfinite(diag[m])
                for m in (
                    "balanced_accuracy", "roc_auc", "mcc",
                    "pr_auc", "brier_score"
                )
            )

            truth_days = truth.reshape(DAYS, 20)
            pred_days = predicted.reshape(DAYS, 20)

            tp = ((truth_days == 1) & (pred_days == 1)).sum(axis=1)
            tn = ((truth_days == 0) & (pred_days == 0)).sum(axis=1)
            pos = (truth_days == 1).sum(axis=1)
            neg = (truth_days == 0).sum(axis=1)

            counts[:, test_id] = np.stack(
                [tp, tn, pos, neg], axis=1
            )

            assert sum(
                int(diag[name])
                for name in ("tp", "tn", "fp", "fn")
            ) == N

            check_ba = balanced_accuracy_from_counts(
                counts[:, test_id].sum(axis=0)
            )
            assert np.isclose(
                check_ba, diag["balanced_accuracy"], atol=1e-12
            )

            checks.append((model_id, horizon))

        print(
            f"PASS — {model_id:14s} "
            "5-seed ensemble verified; 3 scoring interfaces",
            flush=True
        )

    assert len(checks) == 24
    assert np.all(counts[:, :, 2] + counts[:, :, 3] == 20)

    print("\n[5] SEGMENT-STRATIFIED BOOTSTRAP ON SYNTHETIC SCORING COUNTS")

    observed = balanced_accuracy_from_counts(
        counts.sum(axis=0, dtype=np.int64)
    )
    assert observed.shape == (24,)

    bootstrap_summary = []

    for L in (10, 20, 40):
        boot = np.empty((B, 24), dtype=np.float64)

        for start in range(0, B, 100):
            stop = min(start + 100, B)
            index = bootstrap_indices(L, rng, stop - start)

            assert np.all(index[:, :48] < 48)
            assert np.all(index[:, 48:] >= 48)

            totals = counts[index].sum(axis=1, dtype=np.int64)
            boot[start:stop] = balanced_accuracy_from_counts(totals)

        assert np.isfinite(boot).all()

        low, high = np.quantile(
            boot, [0.025, 0.975], axis=0
        )
        assert np.all(low <= high)

        effect = observed - 0.5
        centered = boot - observed[None, :]
        p = (
            np.sum(centered >= effect[None, :], axis=0) + 1
        ) / (B + 1)

        assert len(p) == 24
        assert np.all((p > 0) & (p <= 1))

        if L == 20:
            q = bh_adjust(p)
            assert len(q) == 24
            assert np.all((q >= p - 1e-12) & (q <= 1))

        bootstrap_summary.append({
            "block_length": L,
            "replicates": B,
            "synthetic_tests": 24,
            "segment_allocation": [48, 217],
            "cross_segment_blocks": 0,
            "finite_statistics": True,
        })

        print(
            f"PASS — L={L}, 2000 replicates, "
            "24 tests, zero segment crossings",
            flush=True
        )

    assert np.allclose(
        bh_adjust([0.001, 0.01, 0.04, 0.2]),
        [0.004, 0.02, 0.05333333333333334, 0.2]
    )

    print("PASS — 24-test BH-FDR adjustment and numerical oracle")

    print("\n[6] WRITE INTEGRATION REPORT — SYNTHETIC ONLY")

    OUT.mkdir(parents=True, exist_ok=False)

    report = {
        "status": "PASS_SYNTHETIC_ONLY",
        "stage": "36I-D2g",
        "purpose": "Full synthetic prediction-to-statistics integration",
        "source_sha256": sha(Path(__file__).resolve()),
        "d2f_report_sha256": sha(D2F_REPORT),
        "step35_manifest_sha256": sha(STEP35),
        "candidate_keys_sha256": sha(KEYS),
        "frozen_threshold_contract_sha256": sha(THRESHOLDS),
        "seed_predictions_hash_verified": 40,
        "saved_ensemble_arrays_exactly_verified": 8,
        "frozen_thresholds_used": 24,
        "original_step36_scoring_functions_tested": True,
        "synthetic_observations": N,
        "synthetic_dates": DAYS,
        "segment_lengths": [48, 217],
        "bootstrap": bootstrap_summary,
        "primary_block_length": 20,
        "bh_fdr_tests": 24,
        "real_final_test_metrics_generated": False,
        "synthetic_results_are_manuscript_claims": False,
        "sealed_test_read": False,
        "historical_raw_test_reads": 2,
        "frozen_scientific_files_modified": False,
        "formal_amendment_approved": False,
        "additional_test_access_authorized": False,
    }

    path = OUT / "36i_d2g_synthetic_scoring_DRAFT.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("Report:", path)
    print("Report SHA256:", sha(path))

    print("\n" + "=" * 105)
    print("STEP 36I-D2g — SYNTHETIC SCORING INTEGRATION PASS")
    print("Seed predictions verified: 40/40")
    print("Five-seed ensembles verified: 8/8")
    print("Architecture-horizon scoring checks: 24/24")
    print("Bootstrap specifications verified: 10, 20, 40")
    print("Real test data accessed: FALSE")
    print("Production execution authorized: FALSE")
    print("=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline-synthetic", action="store_true", required=True
    )
    parser.parse_args()
    main()
