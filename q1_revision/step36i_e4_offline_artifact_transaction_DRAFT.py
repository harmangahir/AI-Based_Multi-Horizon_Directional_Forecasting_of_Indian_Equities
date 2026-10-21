#!/usr/bin/env python3
"""Step 36I-E4: test isolated, atomic artifact persistence on synthetic inputs.

No real final-test data loader, execution mode, or authorization exists here.
No GPU inference. This reuses the previously saved 40 SYNTHETIC_ONLY files.
"""
from __future__ import annotations

import sys

FORBIDDEN = {"--execute", "--worker", "--production", "--read-test", "--resume",
             "--materialize", "--authorize", "--score-test"}
if FORBIDDEN.intersection(sys.argv[1:]):
    raise SystemExit("DENIED: E4 supports SYNTHETIC_ONLY persistence, not real evaluation.")

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
EXP = WORK.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
MAN = EXP / "manifests"
AUD = EXP / "audits"
D2F_DIR = AUD / "step36i_d2f_full_synthetic_inference_draft"
D2F = D2F_DIR / "36i_d2f_full_synthetic_rehearsal_DRAFT.json"
D2G = AUD / "step36i_d2g_synthetic_scoring_draft/36i_d2g_synthetic_scoring_DRAFT.json"
E3 = AUD / "step36i_e3_isolated_runner_review_draft/36i_e3_isolated_runner_review_DRAFT.json"
KEYS = AUD / "step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv"
THRESHOLDS = AUD / "35_final_test_threshold_contract.csv"
STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"
ACCESS = MAN / "36_final_test_access_consumption_state.json"
SOURCE = WORK / "step36_one_time_final_test_materialization_inference_scoring_v1.py"
DEST = AUD / "step36i_e4_synthetic_artifact_transaction_draft"
MODELS = ("DLINEAR", "NLINEAR", "PATCHTST", "TIMESNET", "ITRANSFORMER", "FREDFORMER", "FRNET", "MLF")
SEEDS = (42,43,44,45,46)
HORIZONS = (1,3,5)
N = 5300
SHAS = {
    D2F: "197ad07632bfdcd5134ca959c132a896a75711986500d72d7713d681e56370ab",
    D2G: "f520919ef7cd0640a5807fcc61d9eee638fdf19a651b89523f1ef55607a52a6d",
    E3: "4027e53e49ef918a3f3a3bc933908d327e233b741d387088279359b392fe1094",
    KEYS: "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
    STEP35: "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
    SOURCE: "6de04ba7e167449101e2e16bab47ca4ce088f11284c4cbc93b61a6cfbb979f12",
}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def require(cond: bool, message: str):
    if not cond:
        raise RuntimeError(message)

def main():
    print("="*105)
    print("STEP 36I-E4 — OFFLINE SYNTHETIC ARTIFACT TRANSACTION PREFLIGHT")
    print("="*105)
    print("\n[1] IMMUTABILITY, ACCESS HISTORY AND OUTPUT NAMESPACE")
    for path, digest in SHAS.items():
        require(path.is_file() and sha(path)==digest, f"Missing or SHA mismatch: {path}")
        print("PASS:", path.name)
    require(THRESHOLDS.is_file(), "Frozen thresholds missing")
    state = json.loads(ACCESS.read_text(encoding="utf-8"))
    require(int(state.get("raw_test_filesystem_open_count_completed", -1)) == 2 and
            state.get("final_test_scored") is False, "Final-test access state changed")
    d2f = json.loads(D2F.read_text(encoding="utf-8"))
    d2g = json.loads(D2G.read_text(encoding="utf-8"))
    e3 = json.loads(E3.read_text(encoding="utf-8"))
    require(d2f["status"]=="PASS_SYNTHETIC_ONLY" and d2f["checkpoints_tested"]==40,
            "D2f test status unexpected")
    require(d2g["status"]=="PASS_SYNTHETIC_ONLY" and d2g["frozen_thresholds_used"]==24,
            "D2g test status unexpected")
    require(e3["status"]=="PASS_OFFLINE_ONLY_NOT_APPROVED" and
            e3["production_runner_complete"] is False and
            e3["third_raw_test_read_authorized"] is False,
            "E3 approval state changed")
    require(not DEST.exists(), "E4 final directory already exists; STOP")
    staging = AUD / f"_step36i_e4_synthetic_staging_{os.getpid()}"
    require(not staging.exists(), "E4 staging path exists; STOP")
    require(not (MAN/"36i_amended_final_test_materialization_manifest.json").exists(),
            "Production namespace occupied; STOP")
    print("PASS — 2 historical raw reads; 0 scored; production namespace untouched")

    print("\n[2] STRUCTURAL ORDER AND FROZEN THRESHOLDS")
    keys = pd.read_csv(KEYS, parse_dates=["Date"])
    require(len(keys)==N and keys["Date"].nunique()==265, "Cohort dimensions drift")
    require(keys.equals(keys.sort_values(["Date","Ticker"]).reset_index(drop=True)),
            "Physical row order drift")
    require(keys.groupby("Date")["Ticker"].nunique().eq(20).all(),
            "Unbalanced date clusters")
    segments = keys.drop_duplicates("Date")["continuity_segment_id"].astype(int).tolist()
    require(segments == [0]*48 + [1]*217, "Segment allocation drift")
    thresholds = pd.read_csv(THRESHOLDS)
    tcol = "selected_threshold" if "selected_threshold" in thresholds else "threshold"
    require(len(thresholds)==24 and
            not thresholds[["model_id","horizon_days"]].duplicated().any(),
            "Threshold count or duplicates drift")
    require({(str(r.model_id),int(r.horizon_days)) for r in thresholds.itertuples(index=False)} ==
            {(m,h) for m in MODELS for h in HORIZONS}, "Threshold identity drift")
    tm = {(str(r.model_id),int(r.horizon_days)): float(getattr(r, tcol))
          for r in thresholds.itertuples(index=False)}
    require(all(np.isfinite(t) and 0<=t<=1 for t in tm.values()), "Invalid threshold")
    print("PASS — 265 ordered dates, 20-stock clusters, 24 frozen thresholds")

    print("\n[3] REPLAY SAVED SYNTHETIC PREDICTIONS; NO GPU OR REAL LABELS")
    spec = importlib.util.spec_from_file_location("e4_scoring_probe", SOURCE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    records = {(r["model_id"],int(r["seed"])): r
               for r in d2f["synthetic_worker_results"]}
    require(len(records)==40, "Synthetic checkpoint report incomplete")
    saved_ensembles = {r["model_id"]:r for r in d2f["synthetic_ensembles"]}
    require(set(saved_ensembles)==set(MODELS), "Synthetic ensemble report incomplete")
    synthetic_y = np.random.default_rng(20261013).integers(0,2,size=(N,3),dtype=np.int8)
    stage_ensemble = staging/"synthetic_ensembles"
    stage_ensemble.mkdir(parents=True, exist_ok=False)
    metric_rows=[]
    long_parts=[]
    artifact_rows=[]
    for model_id in MODELS:
        seed_probs=[]
        for seed in SEEDS:
            rec=records[(model_id,seed)]
            path=Path(rec["synthetic_prediction_path"])
            require(path.is_file() and sha(path)==rec["synthetic_prediction_sha256"],
                    f"Synthetic checkpoint artifact hash changed: {model_id}/{seed}")
            with np.load(path,allow_pickle=False) as npz:
                p=np.asarray(npz["synthetic_probability"],dtype=np.float64)
                ri=np.asarray(npz["synthetic_row_index"],dtype=np.int64)
            require(p.shape==(N,3) and np.isfinite(p).all() and
                    np.array_equal(ri,np.arange(N,dtype=np.int64)),
                    "Saved synthetic input shape/order drift")
            seed_probs.append(p)
        ensemble=np.stack(seed_probs,axis=0).mean(axis=0)
        orig_path=D2F_DIR/"synthetic_ensembles"/f"SYNTHETIC_ONLY_{model_id}_five_seed_mean.npz"
        require(sha(orig_path)==saved_ensembles[model_id]["synthetic_ensemble_sha256"],
                "Saved synthetic ensemble hash drift")
        with np.load(orig_path,allow_pickle=False) as npz:
            prior=np.asarray(npz["synthetic_ensemble_probability"],dtype=np.float32)
        require(np.array_equal(ensemble.astype(np.float32),prior),
                "Five-seed mean mismatch")
        output_path=stage_ensemble/f"SYNTHETIC_ONLY_{model_id}_five_seed_mean.npz"
        with output_path.open("xb") as fh:
            np.savez_compressed(fh,synthetic_ensemble_probability=prior,
                                synthetic_row_index=np.arange(N,dtype=np.int64))
        part=keys[["Ticker","Date","continuity_segment_id"]].copy()
        part.insert(0,"model_id",model_id)
        part.insert(1,"synthetic_row_index",np.arange(N,dtype=np.int64))
        for j,h in enumerate(HORIZONS):
            threshold=tm[(model_id,h)]
            p=prior[:,j].astype(np.float64)
            y=synthetic_y[:,j].astype(np.int64)
            diag=original.binary_diagnostics(y,p,threshold)
            require(all(np.isfinite(diag[x]) for x in ("balanced_accuracy","roc_auc","mcc","pr_auc","brier_score")),
                    "Nonfinite synthetic scoring statistic")
            pred=(p>=threshold).astype(np.int8)
            part[f"synthetic_probability_{h}d"]=prior[:,j]
            part[f"synthetic_direction_{h}d"]=synthetic_y[:,j]
            part[f"frozen_threshold_{h}d"]=threshold
            part[f"synthetic_prediction_{h}d"]=pred
            metric_rows.append({"synthetic_only":True,"model_id":model_id,"horizon_days":h,
                                "n":N,"threshold":threshold,**diag})
        long_parts.append(part)
        artifact_rows.append({"relative_path":str(output_path.relative_to(staging)),
                              "sha256":sha(output_path),"bytes":output_path.stat().st_size})
        print(f"PASS — {model_id}: five-seed replay + 3 synthetic scores")
    metrics=pd.DataFrame(metric_rows)
    long_df=pd.concat(long_parts,ignore_index=True)
    require(len(metrics)==24 and len(long_df)==8*N,"Synthetic metrics/predictions row count drift")
    metrics_path=staging/"SYNTHETIC_ONLY_metrics.csv"
    long_path=staging/"SYNTHETIC_ONLY_ensemble_predictions.parquet"
    metrics.to_csv(metrics_path,index=False)
    long_df.to_parquet(long_path,index=False,engine="pyarrow",compression="snappy")
    reread=pd.read_parquet(long_path,engine="pyarrow")
    require(len(reread)==8*N and (reread.groupby("model_id").size()==N).all(),
            "Synthetic parquet roundtrip mismatch")
    for path in (metrics_path,long_path):
        artifact_rows.append({"relative_path":str(path.relative_to(staging)),
                              "sha256":sha(path),"bytes":path.stat().st_size})
    print("PASS — 8 synthetic ensemble NPZ, 24 metric rows, 42,400 long-form prediction rows")

    print("\n[4] COMMIT HASHED SYNTHETIC TRANSACTION WITHOUT OVERWRITE")
    report={
        "status":"PASS_SYNTHETIC_ARTIFACT_TRANSACTION_ONLY", "stage":"36I-E4",
        "source_sha256":sha(Path(__file__).resolve()),"e3_report_sha256":sha(E3),
        "d2f_report_sha256":sha(D2F),"d2g_report_sha256":sha(D2G),
        "candidate_keys_sha256":sha(KEYS),"step35_sha256":sha(STEP35),
        "frozen_thresholds_sha256":sha(THRESHOLDS),
        "synthetic_observations":N,"synthetic_sessions":265,
        "stocks_per_session":20,"synthetic_ensembles":8,"synthetic_metrics":24,
        "synthetic_prediction_rows":len(long_df),"historical_test_file_reads":2,
        "original_scientific_contract_modified":False,
        "model_training_or_gpu_inference_performed":False,
        "actual_final_test_predictors_accessed":False,"actual_final_test_labels_accessed":False,
        "production_runner_implemented":False,"protocol_amendment_approved":False,
        "third_raw_test_access_authorized":False,
        "limitations":["Only synthetic artifact persistence is demonstrated.",
                       "No real-data loader, sealed-test access, production checkpoint orchestration or final-test scoring is enabled.",
                       "A separate independently reviewed production implementation and access authorization remain required."],
        "artifacts":artifact_rows,
    }
    report_path=staging/"36i_e4_synthetic_transaction_DRAFT.json"
    with report_path.open("x",encoding="utf-8") as fh:
        json.dump(report,fh,indent=2,sort_keys=True)
        fh.write("\n")
    # The directory rename is the only publication action. Never replace a prior draft.
    require(not DEST.exists(),"E4 destination appeared during execution; STOP")
    os.replace(staging,DEST)
    report_final=DEST/report_path.name
    for rec in artifact_rows:
        final_artifact = DEST / rec["relative_path"]
        require(final_artifact.is_file() and sha(final_artifact) == rec["sha256"],
                f"Committed artifact hash mismatch: {rec['relative_path']}")
    require(sha(STEP35)==SHAS[STEP35] and sha(SOURCE)==SHAS[SOURCE],
            "Frozen scientific input changed unexpectedly")
    last_state=json.loads(ACCESS.read_text(encoding="utf-8"))
    require(last_state==state, "Raw-test marker changed during synthetic transaction")
    print("PASS — all committed artifacts independently rehashed")
    print("COMMITTED:", DEST)
    print("REPORT:",report_final)
    print("REPORT SHA256:",sha(report_final))
    print("="*105)
    print("STEP 36I-E4 SYNTHETIC TRANSACTION PASS — PRODUCTION STILL DISABLED")
    print("40 frozen seed outputs reused; GPU executed: FALSE")
    print("Sealed test opened: FALSE | Test scored: FALSE | Additional read authorized: FALSE")
    print("="*105)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--offline-commit", action="store_true", required=True)
    parser.parse_args()
    main()
