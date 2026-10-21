#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 29A: Stage-2 seed-scope compatibility robustness execution.

This compatibility runner preserves the frozen Step-29 robustness protocol while
correcting one implementation-order issue discovered by the first seed-43 canary:
the immutable Step-25/26/27 Stage-1 provenance checks must execute with their
frozen screening seed 42, whereas the subsequent Stage-2 model initialization
and epoch permutations must execute with the requested robustness seed 43..46.

No successful Stage-2 fit was produced by the failed Step-29 v1 canary.
Stage-1 artifacts remain read-only and the final test remains locked.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, subprocess, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import step00_config as cfg
import step26_stage1_runner_dryrun_v3 as s26
import step27_stage1_execution_v3 as s27
import step27g_timesnetb_mb64_compatibility_runner_v1 as s27g

STEP_NAME = "STEP 29A — STAGE-2 SEED-SCOPE COMPATIBILITY ROBUSTNESS EXECUTION"
SCRIPT = Path(__file__).resolve()
RESULT_PREFIX = "__STEP29A_RESULT__="

EXPECTED_CONFIGURATION_FINGERPRINT = "9675cb42e5544d9a3cda235208b33cc4c27f7583531535505b42056aa86b04dd"
EXPECTED_STEP16_PROTOCOL_FP = "ba87cc46c61d3d3ed0e4e82e4cc2d712f1b511ded94fe336a6adbb56b9a54978"
EXPECTED_STEP27_SOURCE_SHA = "83ac36dcc5e178051ac90865970f4062626a05f48eda4c0548b3a8b1799ec074"
EXPECTED_STEP27_SOURCE_FP = "3978415d953a1561ebe345180529c97bb8e4a39a89219668008a25b3ad72b3d9"
EXPECTED_STEP27_INPUT_FP = "920c2fee5761bf76b438039881d97976081572733b4d3c8a09bf6a9a33549c85"
EXPECTED_STEP27F_SHA = "04c30b59b8bef7327b44e1764240ead258edbba64f278e31ca29249141dfdce8"
EXPECTED_STEP27I_FP = "0118cd700127727dbb3a193e23b1da893155bee05e3cd0e98844c12844a25f39"
EXPECTED_STEP27I_MANIFEST_SHA = "1065a5e2e45e61d3fe9a6526976edeb47ede9f067e03ef87cb550fbec3ab622b"
EXPECTED_STEP28_FP = "254a70bb3b5462cfefbd460e7d6308e04fc7531f3f7d488ef7b26edbe062fc93"
EXPECTED_STEP28_MANIFEST_SHA = "1b441992f9bcd0faed0201bcd509d6d143a371919478b0490437e6da38dbc165"

EXPECTED_STEP29_V1_SOURCE_SHA = "48b2f5bb1f52932d14c98fb45c49153e373c1f3884310f6d917ea9e939a03e8b"
EXPECTED_STEP29_V1_SOURCE_FP = "f951f98ec9a695651bd8d222db0c9d554c297070113d03d3251f64902db7bc44"

SEEDS = (42,43,44,45,46)
NEW_SEEDS = (43,44,45,46)
FOLDS = (1,2,3,4,5)
SCREENING_THRESHOLD = 0.5
EXPECTED_LOGICAL = 200
EXPECTED_REUSE = 40
EXPECTED_NEW = 160

STEP16_FREEZE = cfg.MANIFEST_DIR / "16_stage1_model_registry_freeze.json"
STEP27I_FIT_REGISTER = cfg.AUDIT_DIR / "27i_stage1_amendment_aware_fit_register.csv"
STEP27I_FREEZE = cfg.MANIFEST_DIR / "27i_stage1_amendment_aware_execution_freeze.json"
STEP27I_MANIFEST = cfg.MANIFEST_DIR / "27i_stage1_amendment_aware_execution_manifest.json"
STEP28_SELECTED = cfg.AUDIT_DIR / "28_selected_configurations.csv"
STEP28_FREEZE = cfg.MANIFEST_DIR / "28_configuration_selection_freeze.json"
STEP28_MANIFEST = cfg.MANIFEST_DIR / "28_configuration_selection_manifest.json"

ROBUST_ROOT = s27.RUN_ROOT.parent / "stage2_robustness_fits"
ARCHIVE_ROOT = cfg.AUDIT_DIR / "step29a_incomplete_fit_archives"

# Frozen Step-29 v1 audit trail — verified, never overwritten.
STEP29_V1_WORKING_SCRIPT = SCRIPT.parent / "step29_selected_config_fixed_seed_robustness_v1.py"
STEP29_V1_SOURCE_FREEZE_JSON = cfg.MANIFEST_DIR / "29_robustness_execution_source_freeze.json"
STEP29_V1_FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step29_selected_config_fixed_seed_robustness_frozen.py"

# Distinct Step-29A provenance/output-control artifacts.
SOURCE_FREEZE_JSON = cfg.MANIFEST_DIR / "29a_robustness_seed_scope_compat_source_freeze.json"
SOURCE_FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step29a_stage2_seed_scope_compatibility_runner_frozen.py"
PLAN_CSV = cfg.AUDIT_DIR / "29a_robustness_logical_fit_plan.csv"
REUSE_CSV = cfg.AUDIT_DIR / "29a_seed42_reuse_register.csv"
PREP_JSON = cfg.MANIFEST_DIR / "29a_robustness_prepare_manifest.json"


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def canonical_hash(x: Any) -> str:
    return cfg.canonical_json_sha256(x)

def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def ensure_dirs():
    for p in (ROBUST_ROOT, ARCHIVE_ROOT, SOURCE_FREEZE_JSON.parent, PLAN_CSV.parent): p.mkdir(parents=True, exist_ok=True)


def source_semantic() -> dict:
    return {
        "schema_version":1,
        "status":"STEP29A_ROBUSTNESS_SEED_SCOPE_COMPAT_SOURCE_FROZEN",
        "configuration_fingerprint_sha256":EXPECTED_CONFIGURATION_FINGERPRINT,
        "step29a_source_sha256":sha256_file(SCRIPT),
        "upstream":{
            "step16_selection_protocol_fingerprint_sha256":EXPECTED_STEP16_PROTOCOL_FP,
            "step27_source_sha256":EXPECTED_STEP27_SOURCE_SHA,
            "step27_source_fingerprint_sha256":EXPECTED_STEP27_SOURCE_FP,
            "step27_development_input_fingerprint_sha256":EXPECTED_STEP27_INPUT_FP,
            "step27f_timesnetb_amendment_sha256":EXPECTED_STEP27F_SHA,
            "step27i_stage1_fingerprint_sha256":EXPECTED_STEP27I_FP,
            "step27i_manifest_sha256":EXPECTED_STEP27I_MANIFEST_SHA,
            "step28_selection_fingerprint_sha256":EXPECTED_STEP28_FP,
            "step28_manifest_sha256":EXPECTED_STEP28_MANIFEST_SHA,
            "step29_v1_source_sha256":EXPECTED_STEP29_V1_SOURCE_SHA,
            "step29_v1_source_fingerprint_sha256":EXPECTED_STEP29_V1_SOURCE_FP,
        },
        "robustness_protocol":{
            "selected_config_only":True,
            "folds":list(FOLDS),
            "seeds":list(SEEDS),
            "logical_fits":EXPECTED_LOGICAL,
            "seed42_reused_fits":EXPECTED_REUSE,
            "incremental_new_fits":EXPECTED_NEW,
            "screening_threshold":SCREENING_THRESHOLD,
            "best_seed_promotion":False,
            "test_used":False,
            "oof_probability_per_endpoint":"arithmetic mean across seeds 42,43,44,45,46 for same architecture/fold/endpoint",
        },
        "controlled_adaptations":{
            "run_seed_parameterized":True,
            "stage1_provenance_verification_seed":42,
            "robustness_training_seed_transition_point":"after original Step-27 verify_upstream/source/input checks, inside execution_maps",
            "seed_controls_global_rngs":True,
            "seed_controls_epoch_permutation":"SeedSequence([seed, fold, epoch])",
            "separate_stage2_output_namespace":True,
            "stage1_artifacts_overwritten":False,
            "timesnet_b_microbatch":64,
            "all_other_step27_training_semantics_preserved":True,
        },
        "protection":{"threshold_optimization_performed":False,"best_seed_selection_performed":False,"final_test_access":False},
    }


def ensure_source_freeze() -> tuple[str,str]:
    ensure_dirs(); sem=source_semantic(); sha=sem["step29a_source_sha256"]; fp=canonical_hash(sem)
    if not SOURCE_FREEZE_JSON.exists():
        if SOURCE_FROZEN_SCRIPT.exists():
            raise RuntimeError("Frozen Step-29A source exists without source-freeze JSON.")
        s27.atomic_json(SOURCE_FREEZE_JSON,{**sem,"step29a_source_fingerprint_sha256":fp})
        shutil.copy2(SCRIPT,SOURCE_FROZEN_SCRIPT)
    d=load_json(SOURCE_FREEZE_JSON)
    if d.get("step29a_source_sha256")!=sha:
        raise RuntimeError("Current Step-29A source differs from frozen source.")
    if d.get("step29a_source_fingerprint_sha256")!=fp:
        raise RuntimeError("Step-29A source fingerprint drift.")
    c=dict(d); stored=c.pop("step29a_source_fingerprint_sha256",None)
    if canonical_hash(c)!=stored:
        raise RuntimeError("Step-29A source-freeze semantic no longer recomputes.")
    if not SOURCE_FROZEN_SCRIPT.is_file() or sha256_file(SOURCE_FROZEN_SCRIPT)!=sha:
        raise RuntimeError("Frozen Step-29A source copy drift.")
    return fp,sha



def verify_step29_v1_failure_provenance():
    """Verify the frozen v1 implementation that failed before any real fit."""
    required = (
        STEP29_V1_WORKING_SCRIPT,
        STEP29_V1_SOURCE_FREEZE_JSON,
        STEP29_V1_FROZEN_SCRIPT,
    )
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(
            "Required frozen Step-29 v1 audit artifacts are missing:\n  - "
            + "\n  - ".join(str(p) for p in missing)
        )

    if sha256_file(STEP29_V1_WORKING_SCRIPT) != EXPECTED_STEP29_V1_SOURCE_SHA:
        raise RuntimeError("Step-29 v1 working source SHA drift.")
    if sha256_file(STEP29_V1_FROZEN_SCRIPT) != EXPECTED_STEP29_V1_SOURCE_SHA:
        raise RuntimeError("Step-29 v1 frozen source SHA drift.")

    d = load_json(STEP29_V1_SOURCE_FREEZE_JSON)
    if d.get("step29_source_sha256") != EXPECTED_STEP29_V1_SOURCE_SHA:
        raise RuntimeError("Step-29 v1 source-freeze SHA drift.")
    if d.get("step29_source_fingerprint_sha256") != EXPECTED_STEP29_V1_SOURCE_FP:
        raise RuntimeError("Step-29 v1 source fingerprint drift.")

    c = dict(d)
    stored = c.pop("step29_source_fingerprint_sha256", None)
    if canonical_hash(c) != stored:
        raise RuntimeError("Step-29 v1 source-freeze fingerprint no longer recomputes.")

    return {
        "source_sha256": EXPECTED_STEP29_V1_SOURCE_SHA,
        "source_fingerprint_sha256": EXPECTED_STEP29_V1_SOURCE_FP,
        "failure_classification":
            "PRE_TRAINING_STAGE1_SCOPE_GUARD; NO_SUCCESSFUL_STAGE2_FIT",
    }


def verify_step16():
    d=load_json(STEP16_FREEZE); r=d["selection_protocol"]["stage2_selected_config_seed_robustness"]
    if r.get("seeds")!=list(SEEDS): raise RuntimeError("Step-16 robustness seed-set drift.")
    if r.get("folds")!=list(FOLDS): raise RuntimeError("Step-16 robustness fold-set drift.")
    if r.get("selected_config_only") is not True: raise RuntimeError("Step-16 selected-config-only drift.")
    if r.get("best_seed_promotion") is not False: raise RuntimeError("Step-16 best-seed-promotion drift.")
    if r.get("test_used") is not False: raise RuntimeError("Step-16 robustness test-used drift.")
    if "exact training-protocol/config/data hashes match" not in str(r.get("screening_seed42_artifact_reuse_allowed","")):
        raise RuntimeError("Step-16 seed42-reuse contract drift.")
    if d["selection_protocol"]["stage1_configuration_selection"].get("screening_threshold")!=0.5: raise RuntimeError("Screening threshold drift.")
    return d


def verify_step27i():
    if sha256_file(STEP27I_MANIFEST)!=EXPECTED_STEP27I_MANIFEST_SHA: raise RuntimeError("Step-27I manifest SHA drift.")
    m=load_json(STEP27I_MANIFEST)
    if m.get("stage1_execution_amendment_aware_fingerprint_sha256")!=EXPECTED_STEP27I_FP: raise RuntimeError("Step-27I fingerprint drift.")
    if int(m.get("completed_fits",-1))!=80 or m.get("final_test_accessed") is not False: raise RuntimeError("Step-27I state drift.")
    f=load_json(STEP27I_FREEZE)
    a=f["protocol"]["timesnet_b_protocol_amendment"]
    if a.get("amendment_manifest_sha256")!=EXPECTED_STEP27F_SHA or int(a.get("physical_microbatch_new",-1))!=64: raise RuntimeError("TimesNet-B amendment drift.")
    if sha256_file(STEP27I_FIT_REGISTER)!=f["fit_register"]["sha256"]: raise RuntimeError("Step-27I fit-register hash drift.")
    return f


def verify_step28() -> pd.DataFrame:
    if sha256_file(STEP28_MANIFEST)!=EXPECTED_STEP28_MANIFEST_SHA: raise RuntimeError("Step-28 manifest SHA drift.")
    m=load_json(STEP28_MANIFEST)
    if m.get("configuration_selection_fingerprint_sha256")!=EXPECTED_STEP28_FP: raise RuntimeError("Step-28 fingerprint drift.")
    if m.get("configuration_selection_performed") is not True or m.get("threshold_optimization_performed") is not False or m.get("best_seed_selection_performed") is not False or m.get("final_test_accessed") is not False:
        raise RuntimeError("Step-28 protection-state drift.")
    if sha256_file(STEP28_SELECTED)!=m["artifacts"][STEP28_SELECTED.name]["sha256"]: raise RuntimeError("Step-28 selected-config CSV hash drift.")
    d=pd.read_csv(STEP28_SELECTED)
    if len(d)!=8 or d["model_id"].duplicated().any() or set(d["model_id"])!=set(s27.MODEL_ORDER): raise RuntimeError("Step-28 selected-config table drift.")
    return d


def verify_final_test_lock():
    u=s27.verify_upstream()
    if u["global_final_test_state"]!="LOCKED_NOT_OPENED" or u["test_seal_status"]!="SEALED_NOT_OPENED": raise RuntimeError("Final-test protection drift.")
    return u


def execution_state():
    u=s27.verify_upstream(); sfp,ssha=s27.ensure_source_freeze(u)
    if sfp!=EXPECTED_STEP27_SOURCE_FP or ssha!=EXPECTED_STEP27_SOURCE_SHA: raise RuntimeError("Original Step-27 source drift.")
    ifp,ireg=s27.ensure_input_register(u,sfp)
    if ifp!=EXPECTED_STEP27_INPUT_FP: raise RuntimeError("Development-input fingerprint drift.")
    imap=s27.input_record_map(ireg); smap,bplan=s27g.amended_execution_maps(u)
    if int(smap[("TIMESNET","TIMESNET_B")]["physical_microbatch_size"])!=64: raise RuntimeError("Amended TimesNet-B schedule drift.")
    return {"upstream":u,"source_fp":sfp,"source_sha":ssha,"input_fp":ifp,"input_map":imap,"schedule_map":smap,"batch_plan":bplan}


def selected_map(selected):
    return {str(r["model_id"]):str(r["selected_config_id"]) for _,r in selected.iterrows()}


def verify_seed42_reuse(selected,state):
    df=pd.read_csv(STEP27I_FIT_REGISTER); sel=selected_map(selected); rows=[]
    for mid in s27.MODEL_ORDER:
        cid=sel[mid]
        for fold in FOLDS:
            x=df[(df.model_id==mid)&(df.config_id==cid)&(df.fold.astype(int)==fold)&(df.seed.astype(int)==42)]
            if len(x)!=1: raise RuntimeError(f"Seed42 reuse row missing/duplicated: {mid}/{cid}/fold{fold}")
            r=x.iloc[0]; mp=Path(str(r.fit_manifest_path))
            if not mp.is_file() or sha256_file(mp)!=str(r.fit_manifest_sha256): raise RuntimeError(f"Seed42 manifest hash drift: {mid}/{cid}/fold{fold}")
            m=load_json(mp)
            if int(m.get("seed",-1))!=42 or m.get("model_id")!=mid or m.get("config_id")!=cid or int(m.get("fold",-1))!=fold: raise RuntimeError("Seed42 identity drift.")
            if m.get("configuration_fingerprint_sha256")!=EXPECTED_CONFIGURATION_FINGERPRINT or m.get("stage1_execution_source_fingerprint_sha256")!=state["source_fp"] or m.get("stage1_development_input_fingerprint_sha256")!=state["input_fp"]: raise RuntimeError("Seed42 provenance drift.")
            expmicro=int(state["schedule_map"][(mid,cid)]["physical_microbatch_size"])
            if int(m.get("physical_microbatch_size",-1))!=expmicro or float(m.get("threshold"))!=0.5: raise RuntimeError("Seed42 protocol drift.")
            if mid=="TIMESNET" and m.get("protocol_amendment",{}).get("manifest_sha256")!=EXPECTED_STEP27F_SHA: raise RuntimeError("TimesNet seed42 amendment provenance missing.")
            arts=m.get("artifacts",{})
            for key in ("best_checkpoint","epoch_history","validation_predictions","fit_metrics"):
                rec=arts.get(key); p=Path(rec["path"]) if rec else None
                if not rec or not p.is_file() or int(rec["size_bytes"])!=p.stat().st_size or rec["sha256"]!=sha256_file(p): raise RuntimeError(f"Seed42 artifact drift: {mid}/{cid}/fold{fold}/{key}")
            ins=m.get("inputs",{}); expected={"train_x":(fold,"train","X_full"),"train_y":(fold,"train","y_direction"),"validation_x":(fold,"validation","X_full"),"validation_y":(fold,"validation","y_direction")}
            for k,src_key in expected.items():
                rec=ins.get(k); src=state["input_map"][src_key]
                if not rec or rec["sha256"]!=src["sha256"] or Path(rec["path"]).resolve()!=Path(src["path"]).resolve(): raise RuntimeError(f"Seed42 input drift: {mid}/{cid}/fold{fold}/{k}")
            contract={
                "configuration_fingerprint_sha256":m["configuration_fingerprint_sha256"],
                "stage1_execution_source_fingerprint_sha256":m["stage1_execution_source_fingerprint_sha256"],
                "stage1_development_input_fingerprint_sha256":m["stage1_development_input_fingerprint_sha256"],
                "step24_preflight_fingerprint_sha256":m["step24_preflight_fingerprint_sha256"],
                "step25_engine_fingerprint_sha256":m["step25_engine_fingerprint_sha256"],
                "step26_runner_fingerprint_sha256":m["step26_runner_fingerprint_sha256"],
                "model_id":mid,"config_id":cid,"fold":fold,
                "logical_batch_size":int(m["logical_batch_size"]),"physical_microbatch_size":int(m["physical_microbatch_size"]),
                "learning_rate":float(m["learning_rate"]),"weight_decay":float(m["weight_decay"]),"gradient_clip_norm":float(m["gradient_clip_norm"]),
                "optimizer":m["optimizer"],"loss":m["loss"],"dtype":m["dtype"],"amp_enabled":bool(m["amp_enabled"]),"tf32_enabled":bool(m["tf32_enabled"]),"scheduler":m["scheduler"],
                "checkpoint_metric":m["checkpoint_metric"],"checkpoint_mode":m["checkpoint_mode"],"max_epochs":int(m["max_epochs"]),"minimum_epoch_before_stopping":int(m["minimum_epoch_before_stopping"]),"early_stopping_patience":int(m["early_stopping_patience"]),"threshold":float(m["threshold"]),
                "inputs":{k:{"sha256":ins[k]["sha256"],"size_bytes":int(ins[k]["size_bytes"])} for k in sorted(expected)}
            }
            rows.append({"model_id":mid,"config_id":cid,"fold":fold,"seed":42,"reuse_status":"HASH_VERIFIED_STAGE1_REUSE","stage1_fit_manifest_path":str(mp.resolve()),"stage1_fit_manifest_sha256":sha256_file(mp),"training_protocol_config_data_contract_sha256":canonical_hash(contract),"validation_predictions_path":arts["validation_predictions"]["path"],"validation_predictions_sha256":arts["validation_predictions"]["sha256"]})
    out=pd.DataFrame(rows)
    if len(out)!=40: raise RuntimeError("Seed42 reuse register is not 40 rows.")
    return out


def build_plan(selected,state):
    sel=selected_map(selected); rows=[]
    for mid in s27.MODEL_ORDER:
        cid=sel[mid]; sch=state["schedule_map"][(mid,cid)]
        for seed in SEEDS:
            for fold in FOLDS:
                rows.append({"model_id":mid,"config_id":cid,"seed":seed,"fold":fold,"artifact_origin":"STAGE1_REUSE" if seed==42 else "STAGE2_NEW_FIT","physical_microbatch_size":int(sch["physical_microbatch_size"]),"logical_batch_size":int(s27.LOGICAL_BATCH),"screening_threshold":0.5,"best_seed_promotion":False,"test_used":False})
    out=pd.DataFrame(rows)
    if len(out)!=200 or out[["model_id","config_id","seed","fold"]].duplicated().any() or int((out.seed==42).sum())!=40: raise RuntimeError("Robustness logical-plan drift.")
    return out


def fit_dir(mid,cid,seed,fold):
    return ROBUST_ROOT/mid/cid/f"seed_{seed:02d}"/f"fold_{fold:02d}"

def manifest_path(mid,cid,seed,fold): return fit_dir(mid,cid,seed,fold)/"fit_manifest.json"
def log_path(mid,cid,seed,fold): return fit_dir(mid,cid,seed,fold)/"orchestration_worker.log"


def verify_new_fit(mid,cid,seed,fold,source_fp,state):
    p=manifest_path(mid,cid,seed,fold)
    if not p.is_file(): return None
    try:
        m=load_json(p)
        if m.get("status")!="STAGE2_ROBUSTNESS_FIT_COMPLETE" or m.get("model_id")!=mid or m.get("config_id")!=cid or int(m.get("fold",-1))!=fold or int(m.get("seed",-1))!=seed: return None
        if m.get("stage2_robustness_execution_source_fingerprint_sha256")!=source_fp or m.get("stage2_configuration_selection_fingerprint_sha256")!=EXPECTED_STEP28_FP: return None
        if m.get("stage1_execution_source_fingerprint_sha256")!=state["source_fp"] or m.get("stage1_development_input_fingerprint_sha256")!=state["input_fp"]: return None
        if int(m.get("physical_microbatch_size",-1))!=int(state["schedule_map"][(mid,cid)]["physical_microbatch_size"]): return None
        if float(m.get("threshold"))!=0.5 or m.get("train_shuffle")!=f"SeedSequence([{seed}, fold, epoch]); candidate-independent": return None
        if mid=="TIMESNET" and m.get("timesnet_b_protocol_amendment_sha256")!=EXPECTED_STEP27F_SHA: return None
        ps=m.get("protection_state",{})
        if not bool(ps.get("configuration_selection_performed")) or bool(ps.get("threshold_optimization_performed")) or bool(ps.get("best_seed_selection_performed")) or bool(ps.get("final_test_data_opened")) or bool(ps.get("final_test_metric_computed")): return None
        for key,rec in m.get("artifacts",{}).items():
            if key not in ("best_checkpoint","epoch_history","validation_predictions","fit_metrics"): continue
            q=Path(rec["path"])
            if not q.is_file() or int(rec["size_bytes"])!=q.stat().st_size or rec["sha256"]!=sha256_file(q): return None
        metrics=load_json(Path(m["artifacts"]["fit_metrics"]["path"]))
        if int(metrics.get("seed",-1))!=seed or float(metrics.get("threshold"))!=0.5: return None
        return m
    except Exception:
        return None


def archive_invalid(mid,cid,seed,fold):
    d=fit_dir(mid,cid,seed,fold)
    if not d.exists(): return None
    target=ARCHIVE_ROOT/f"{mid}__{cid}__seed_{seed:02d}__fold_{fold:02d}__{stamp()}"
    target.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(d),str(target)); return target


def install_worker_contract(seed):
    """
    Two-phase seed scope.

    Phase 1 — immutable Stage-1 provenance verification:
        s27.SEED = 42
        s26.SEED = 42

    Phase 2 — Stage-2 robustness training:
        transition to requested seed only when worker_fit reaches
        execution_maps(), i.e. AFTER verify_upstream(), ensure_source_freeze(),
        and ensure_input_register() have all verified their frozen Stage-1
        provenance under seed 42.

    This preserves the immutable upstream checks instead of weakening or
    bypassing them.
    """
    if seed not in NEW_SEEDS:
        raise RuntimeError("Worker trains only seeds 43..46.")

    s27.SEED = 42
    s26.SEED = 42

    transition = {
        "transitioned": False,
        "from_seed": 42,
        "to_seed": int(seed),
    }

    def stage2_execution_maps(upstream):
        if transition["transitioned"]:
            raise RuntimeError("Stage-2 seed transition attempted more than once.")
        if int(s27.SEED) != 42 or int(s26.SEED) != 42:
            raise RuntimeError(
                "Stage-1 provenance checks were not entered with frozen seed 42."
            )

        schedule_map, batch_plan = s27g.amended_execution_maps(upstream)

        s27.SEED = int(seed)
        s26.SEED = int(seed)

        probe = s26.epoch_permutation(17, fold=1, epoch=1)
        expected = np.random.default_rng(
            np.random.SeedSequence([seed, 1, 1])
        ).permutation(17)
        if not np.array_equal(probe, expected):
            raise RuntimeError(
                "Requested robustness seed does not control epoch permutation."
            )

        transition["transitioned"] = True
        return schedule_map, batch_plan

    s27.execution_maps = stage2_execution_maps

    def redirected(mid, cid, fold):
        return fit_dir(mid, cid, seed, fold)

    s27.fit_dir = redirected
    return transition


def augment_manifest(mid,cid,seed,fold,source_fp,source_sha):
    p=manifest_path(mid,cid,seed,fold); m=load_json(p)
    if int(m.get("seed",-1))!=seed: raise RuntimeError("Original worker failed to record requested seed.")
    m["status"]="STAGE2_ROBUSTNESS_FIT_COMPLETE"
    m["stage2_robustness_execution_source_fingerprint_sha256"]=source_fp
    m["stage2_robustness_execution_source_sha256"]=source_sha
    m["stage2_configuration_selection_fingerprint_sha256"]=EXPECTED_STEP28_FP
    m["stage2_selection_manifest_sha256"]=EXPECTED_STEP28_MANIFEST_SHA
    m["stage2_robustness_protocol_fingerprint_sha256"]=EXPECTED_STEP16_PROTOCOL_FP
    m["step29_v1_failed_source_sha256"]=EXPECTED_STEP29_V1_SOURCE_SHA
    m["step29_v1_failed_source_fingerprint_sha256"]=EXPECTED_STEP29_V1_SOURCE_FP
    m["stage2_seed_scope_compatibility"]="Stage-1 provenance verified at frozen seed 42; training transitioned to requested robustness seed only after verify_upstream/source/input checks."
    m["robustness_seed"]=seed; m["seed_role"]="FIXED_ROBUSTNESS_SEED_NOT_SELECTED"; m["seed_set"]=list(SEEDS)
    m["train_shuffle"]=f"SeedSequence([{seed}, fold, epoch]); candidate-independent"
    if mid=="TIMESNET":
        m["timesnet_b_protocol_amendment_sha256"]=EXPECTED_STEP27F_SHA
        m["timesnet_batch_dependent_fft_semantics"]="Step-27F amendment remains binding; physical microbatch=64"
    ps=dict(m.get("protection_state",{})); ps.update({"configuration_selection_performed":True,"stage2_selected_config_robustness":True,"threshold_optimization_performed":False,"best_seed_selection_performed":False,"final_test_data_opened":False,"final_test_metric_computed":False}); m["protection_state"]=ps
    s27.atomic_json(p,m); return m


def worker(mid,cid,seed,fold):
    try:
        source_fp,source_sha=ensure_source_freeze(); selected=verify_step28(); sm=selected_map(selected)
        if sm.get(mid)!=cid: raise RuntimeError(f"Refusing non-selected config {mid}/{cid}; frozen selection is {sm.get(mid)}")
        if seed not in NEW_SEEDS or fold not in FOLDS: raise RuntimeError("Unexpected seed/fold.")
        transition=install_worker_contract(seed)
        result=s27.worker_fit(model_id=mid,config_id=cid,fold=fold)
        if not transition["transitioned"]:
            raise RuntimeError("Stage-2 seed transition did not occur.")
        if int(s27.SEED)!=seed or int(s26.SEED)!=seed:
            raise RuntimeError("Robustness training seed was not retained after transition.")
        augment_manifest(mid,cid,seed,fold,source_fp,source_sha)
        p=manifest_path(mid,cid,seed,fold); result=dict(result); result.update({"status":"COMPLETE","seed":seed,"fit_manifest_path":str(p.resolve()),"fit_manifest_sha256":sha256_file(p)})
        print(RESULT_PREFIX+json.dumps(result,sort_keys=True),flush=True); return 0
    except Exception as e:
        fail={"status":"FAIL","model_id":mid,"config_id":cid,"seed":seed,"fold":fold,"exception_type":type(e).__name__,"exception_message":str(e),"traceback":traceback.format_exc(),"threshold_optimization_performed":False,"best_seed_selection_performed":False,"final_test_data_opened":False}
        print(RESULT_PREFIX+json.dumps(fail,sort_keys=True),flush=True); return 1


def run_worker(mid,cid,seed,fold):
    d=fit_dir(mid,cid,seed,fold); d.mkdir(parents=True,exist_ok=True); lp=log_path(mid,cid,seed,fold)
    env=os.environ.copy(); env["PYTHONDONTWRITEBYTECODE"]="1"; env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    cmd=[sys.executable,str(SCRIPT),"--worker","--model-id",mid,"--config-id",cid,"--seed",str(seed),"--fold",str(fold)]
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env); lines=[]; payloads=[]
    assert p.stdout is not None
    for raw in p.stdout:
        line=raw.rstrip("\n"); lines.append(line)
        if line.startswith(RESULT_PREFIX):
            try: payloads.append(json.loads(line[len(RESULT_PREFIX):]))
            except Exception: pass
        else: print("        "+line,flush=True)
    rc=p.wait(); s27.atomic_text(lp,"\n".join(lines)+"\n")
    res=payloads[0] if len(payloads)==1 else {"status":"FAIL","exception_type":"WorkerProtocolError","exception_message":f"Expected one result marker, found {len(payloads)}"}
    res["worker_return_code"]=rc; res["worker_log"]=str(lp.resolve()); return res


def prepare_state():
    source_fp,source_sha=ensure_source_freeze(); v1=verify_step29_v1_failure_provenance(); verify_step16(); verify_step27i(); selected=verify_step28(); upstream=verify_final_test_lock(); state=execution_state(); reuse=verify_seed42_reuse(selected,state); plan=build_plan(selected,state)
    s27.atomic_csv(PLAN_CSV,plan); s27.atomic_csv(REUSE_CSV,reuse)
    sem={"schema_version":1,"status":"STEP29A_ROBUSTNESS_SEED_SCOPE_COMPAT_PREPARED","step29a_source_fingerprint_sha256":source_fp,"step29a_source_sha256":source_sha,"step29_v1_source_sha256":EXPECTED_STEP29_V1_SOURCE_SHA,"step29_v1_source_fingerprint_sha256":EXPECTED_STEP29_V1_SOURCE_FP,"seed_scope_policy":"verify immutable Stage-1 provenance at seed 42; transition to robustness seed inside execution_maps after verify_upstream/source/input","step16_selection_protocol_fingerprint_sha256":EXPECTED_STEP16_PROTOCOL_FP,"step27i_stage1_fingerprint_sha256":EXPECTED_STEP27I_FP,"step28_selection_fingerprint_sha256":EXPECTED_STEP28_FP,"selected_configurations":selected_map(selected),"seeds":list(SEEDS),"folds":list(FOLDS),"logical_fits":200,"seed42_reuse_fits":40,"new_fits_required":160,"plan_sha256":sha256_file(PLAN_CSV),"seed42_reuse_register_sha256":sha256_file(REUSE_CSV),"configuration_selection_performed":True,"threshold_optimization_performed":False,"best_seed_selection_performed":False,"final_test_data_opened":False,"global_final_test_state":upstream["global_final_test_state"]}
    fp=canonical_hash(sem); s27.atomic_json(PREP_JSON,{**sem,"prepared_at_utc":now(),"step29a_prepare_fingerprint_sha256":fp})
    return {"source_fp":source_fp,"source_sha":source_sha,"v1":v1,"selected":selected,"upstream":upstream,"state":state,"reuse":reuse,"plan":plan,"prepare_fp":fp}


def print_prepare(x):
    print("="*140); print(STEP_NAME); print("="*140)
    print(f"[PASS] Step-29A source SHA-256           : {x['source_sha']}")
    print(f"[PASS] Step-29A source fingerprint       : {x['source_fp']}")
    print(f"[PASS] Step-29 v1 failed source SHA      : {x['v1']['source_sha256']}")
    print("[PASS] Step-29 v1 failure class          : PRE-TRAINING SCOPE GUARD / NO FIT")
    print(f"[PASS] Step-16 robustness protocol fp   : {EXPECTED_STEP16_PROTOCOL_FP}")
    print(f"[PASS] Step-27I Stage-1 fingerprint     : {EXPECTED_STEP27I_FP}")
    print(f"[PASS] Step-28 selection fingerprint    : {EXPECTED_STEP28_FP}")
    print("\nFrozen selected configurations:")
    for _,r in x["selected"].iterrows(): print(f"       {r['model_id']:<14} -> {r['selected_config_id']}")
    print(f"\n[PASS] Robustness seeds                 : {list(SEEDS)}")
    print(f"[PASS] Chronological folds              : {list(FOLDS)}")
    print(f"[PASS] Logical robustness fits          : {len(x['plan'])}/200")
    print(f"[PASS] Seed-42 Stage-1 reuses verified  : {len(x['reuse'])}/40")
    print("[PASS] Incremental new fits required    : 160/160")
    print(f"[PASS] Stage-2 output root              : {ROBUST_ROOT}")
    print("[PASS] Stage-1 output namespace          : READ-ONLY / NOT OVERWRITTEN")
    print("[PASS] Seed-scope compatibility          : Stage-1 verify=42 -> Stage-2 train=requested seed")
    print("[PASS] Epoch shuffle                     : SeedSequence([seed, fold, epoch])")
    print("[PASS] TIMESNET_B microbatch             : 64 (Step-27F remains binding)")
    print("[PASS] Threshold optimization            : NOT PERFORMED")
    print("[PASS] Best-seed promotion               : NOT PERFORMED")
    print("[PASS] Final-test data opened             : False")
    print(f"[PASS] Global final-test state           : {x['upstream']['global_final_test_state']}")
    print("\n"+"="*140); print("STEP 29A PREPARE STATUS: PASS"); print("="*140)
    print("Real Stage-2 training started : False"); print("Seed42 logical fits           : 40 reused, hash-verified"); print("New seeds to train            : 43, 44, 45, 46"); print("New real fits required        : 160"); print("Final test                    : LOCKED_NOT_OPENED"); print("="*140)


def parent(model_id=None,seed=None,fold=None,max_new_fits=None):
    x=prepare_state(); sm=selected_map(x["selected"])
    if model_id is not None and model_id not in sm: raise RuntimeError("Unknown model-id.")
    if seed is not None and seed not in NEW_SEEDS: raise RuntimeError("Execution seed must be 43..46; seed42 is reused.")
    if fold is not None and fold not in FOLDS: raise RuntimeError("Fold must be 1..5.")
    rows=[]
    for mid in s27.MODEL_ORDER:
        if model_id and mid!=model_id: continue
        cid=sm[mid]
        for sd in NEW_SEEDS:
            if seed and sd!=seed: continue
            for fd in FOLDS:
                if fold and fd!=fold: continue
                rows.append((mid,cid,sd,fd))
    print("="*140); print(STEP_NAME); print("="*140); print(f"Requested new-fit rows : {len(rows)}"); print("Seed42 reuses          : 40/40 VERIFIED"); print("Final test             : LOCKED_NOT_OPENED"); print("-"*140)
    done=0; skipped=0
    for mid,cid,sd,fd in rows:
        m=verify_new_fit(mid,cid,sd,fd,x["source_fp"],x["state"])
        if m is not None:
            skipped+=1; print(f"SKIP {mid:<14} {cid:<20} seed={sd} fold={fd} COMPLETE+HASHED best={m['best_epoch']}"); continue
        if fit_dir(mid,cid,sd,fd).exists(): print(f"ARCHIVE incomplete/invalid prior fit -> {archive_invalid(mid,cid,sd,fd)}")
        mb=int(x["state"]["schedule_map"][(mid,cid)]["physical_microbatch_size"]); print(f"RUN  {mid:<14} {cid:<20} seed={sd} fold={fd} mb={mb}")
        r=run_worker(mid,cid,sd,fd)
        if r.get("status")!="COMPLETE" or int(r.get("worker_return_code",-1))!=0: raise RuntimeError(f"Worker failed {mid}/{cid}/seed{sd}/fold{fd}: {r.get('exception_type')} {r.get('exception_message')}")
        m=verify_new_fit(mid,cid,sd,fd,x["source_fp"],x["state"])
        if m is None: raise RuntimeError(f"Post-run verification failed {mid}/{cid}/seed{sd}/fold{fd}")
        done+=1; metrics=load_json(Path(m["artifacts"]["fit_metrics"]["path"])); peak=float(metrics["peak_cuda_allocated_bytes"])/(1024**2)
        print(f"PASS {mid:<14} {cid:<20} seed={sd} fold={fd} epochs={m['epochs_run']:<3} best={m['best_epoch']:<3} valBCE={float(m['best_validation_mean_directional_bce']):.6f} peak={peak:.1f}MiB")
        if max_new_fits is not None and done>=max_new_fits: break
    verified=0
    for mid in s27.MODEL_ORDER:
        cid=sm[mid]
        for sd in NEW_SEEDS:
            for fd in FOLDS:
                if verify_new_fit(mid,cid,sd,fd,x["source_fp"],x["state"]) is not None: verified+=1
    logical=40+verified
    print("-"*140); print("\n"+"="*140); print("STEP 29A INCOMPLETE — RESUMABLE, NOT GLOBALLY FROZEN" if logical<200 else "STEP 29A EXECUTION COMPLETE — FINAL ROBUSTNESS FREEZE STILL PENDING"); print("="*140)
    print(f"Logical robustness fits complete : {logical}/200"); print("Seed42 reused fits               : 40/40"); print(f"New Stage-2 fits verified        : {verified}/160"); print(f"Remaining new fits               : {160-verified}"); print(f"New fits completed this run      : {done}"); print(f"Hash-valid new fits skipped      : {skipped}"); print("Best-seed promotion              : NOT PERFORMED"); print("Threshold optimization           : NOT PERFORMED"); print("Final-test data opened           : False"); print("="*140); return 0


def main():
    p=argparse.ArgumentParser(description=STEP_NAME); p.add_argument("--prepare-only",action="store_true"); p.add_argument("--worker",action="store_true"); p.add_argument("--model-id"); p.add_argument("--config-id"); p.add_argument("--seed",type=int); p.add_argument("--fold",type=int); p.add_argument("--max-new-fits",type=int); a=p.parse_args()
    if a.worker:
        if None in (a.model_id,a.config_id,a.seed,a.fold): raise RuntimeError("Worker requires model-id, config-id, seed, fold.")
        return worker(a.model_id,a.config_id,a.seed,a.fold)
    if a.prepare_only:
        x=prepare_state(); print_prepare(x); return 0
    if a.config_id is not None: raise RuntimeError("--config-id is worker-only; parent always uses Step-28 selection.")
    return parent(a.model_id,a.seed,a.fold,a.max_new_fits)

if __name__=="__main__":
    try: raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStep 29A interrupted. Hash-valid robustness fits remain resumable; Stage-1 artifacts are untouched."); raise SystemExit(130)
    except SystemExit: raise
    except Exception as e:
        print("\n"+"="*140); print("STEP 29A FAILED"); print("="*140); print(f"{type(e).__name__}: {e}\n"); traceback.print_exc(); print("="*140); raise SystemExit(1)
