#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 23 — CLEAN V3 ISOLATED SYNTHETIC CPU/GPU FORWARD-BACKWARD SMOKE TESTS

First executable use of the eight frozen Stage-1 architecture implementations.
All 16 Step-16 A/B candidates are tested on synthetic data only, first on CPU
and then CUDA.  No optimizer step, model fitting, validation scoring, threshold
selection, development-array access, or final-test access occurs.

The clean-V3 Step-22 freeze already integrates the old 22B/22C/22D FRNet/MLF
compatibility rules.  Those legacy amendment files are deliberately not used.

Run:    python step23_synthetic_model_smoke_tests_v3.py
Verify: python step23_synthetic_model_smoke_tests_v3.py --verify-only
"""
from __future__ import annotations

import argparse, copy, hashlib, importlib, json, os, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
import pandas as pd

SCRIPT_FILE=Path(__file__).resolve(); SCRIPT_DIR=SCRIPT_FILE.parent
if SCRIPT_DIR.name!="q1_revision":
    raise RuntimeError("step23_synthetic_model_smoke_tests_v3.py must be inside <repository>/q1_revision/.")
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
import step00_config as cfg

STEP_NUMBER=23; STEP_ID="23"; STEP_NAME="STEP 23 — ISOLATED SYNTHETIC CPU/GPU FORWARD-BACKWARD SMOKE TESTS"; SCHEMA_VERSION="1.0"
MODELS=("DLINEAR","NLINEAR","PATCHTST","TIMESNET","ITRANSFORMER","FREDFORMER","FRNET","MLF")
EXPECTED_MODEL_COUNT=8; EXPECTED_CANDIDATE_COUNT=16; EXPECTED_REPOS=8; EXPECTED_IMPLEMENTATION_REPOS=7; EXPECTED_REFERENCE_REPOS=1
LOOKBACK=60; FEATURES=63; FORECAST_LEN=5; OUTPUTS=3; BATCH=2; SEED=230042; TIMEOUT=600
READOUT_IN=FORECAST_LEN*FEATURES; READOUT_PARAMS=READOUT_IN*OUTPUTS+OUTPUTS
RESULT_PREFIX="__STEP23_RESULT__"; FORBIDDEN_TEST_FILENAME="10_final_test_locked_panel.parquet"
EXPECTED_REPO={"DLINEAR":"LTSF_LINEAR","NLINEAR":"LTSF_LINEAR","PATCHTST":"PATCHTST","TIMESNET":"TIMESNET_TSLIB","ITRANSFORMER":"ITRANSFORMER","FREDFORMER":"FREDFORMER","FRNET":"FRNET","MLF":"MLF"}

ENV_AUDIT=cfg.ENVIRONMENT_DIR/"01_environment_audit.json"
UP={
1:(cfg.MANIFEST_DIR/"01_environment_audit_manifest.json",cfg.MANIFEST_DIR/"01_environment_audit_manifest.sha256"),
2:(cfg.MANIFEST_DIR/"02_market_data_acquisition_manifest.json",cfg.MANIFEST_DIR/"02_market_data_acquisition_manifest.sha256"),
3:(cfg.MANIFEST_DIR/"03_market_integrity_audit_manifest.json",cfg.MANIFEST_DIR/"03_market_integrity_audit_manifest.sha256"),
4:(cfg.MANIFEST_DIR/"04_session_policy_manifest.json",cfg.MANIFEST_DIR/"04_session_policy_manifest.sha256"),
5:(cfg.MANIFEST_DIR/"05_clean_model_panel_manifest.json",cfg.MANIFEST_DIR/"05_clean_model_panel_manifest.sha256"),
6:(cfg.MANIFEST_DIR/"06_stock_feature_manifest.json",cfg.MANIFEST_DIR/"06_stock_feature_manifest.sha256"),
7:(cfg.MANIFEST_DIR/"07_nifty_context_feature_manifest.json",cfg.MANIFEST_DIR/"07_nifty_context_feature_manifest.sha256"),
8:(cfg.MANIFEST_DIR/"08_63_feature_panel_manifest.json",cfg.MANIFEST_DIR/"08_63_feature_panel_manifest.sha256"),
9:(cfg.MANIFEST_DIR/"09_target_manifest.json",cfg.MANIFEST_DIR/"09_target_manifest.sha256"),
10:(cfg.MANIFEST_DIR/"10_chronological_split_manifest.json",cfg.MANIFEST_DIR/"10_chronological_split_manifest.sha256"),
11:(cfg.MANIFEST_DIR/"11_preprocessing_manifest.json",cfg.MANIFEST_DIR/"11_preprocessing_manifest.sha256"),
12:(cfg.MANIFEST_DIR/"12_volatility_regime_manifest.json",cfg.MANIFEST_DIR/"12_volatility_regime_manifest.sha256"),
13:(cfg.MANIFEST_DIR/"13_sequence_index_manifest.json",cfg.MANIFEST_DIR/"13_sequence_index_manifest.sha256"),
14:(cfg.MANIFEST_DIR/"14_model_array_manifest.json",cfg.MANIFEST_DIR/"14_model_array_manifest.sha256"),
15:(cfg.MANIFEST_DIR/"15_pre_model_integrity_manifest.json",cfg.MANIFEST_DIR/"15_pre_model_integrity_manifest.sha256"),
16:(cfg.MANIFEST_DIR/"16_stage1_model_registry_manifest.json",cfg.MANIFEST_DIR/"16_stage1_model_registry_manifest.sha256"),
17:(cfg.MANIFEST_DIR/"17_official_model_source_manifest.json",cfg.MANIFEST_DIR/"17_official_model_source_manifest.sha256"),
18:(cfg.MANIFEST_DIR/"18_source_api_manifest.json",cfg.MANIFEST_DIR/"18_source_api_manifest.sha256"),
19:(cfg.MANIFEST_DIR/"19_adapter_locus_manifest.json",cfg.MANIFEST_DIR/"19_adapter_locus_manifest.sha256"),
20:(cfg.MANIFEST_DIR/"20_transitive_backbone_manifest.json",cfg.MANIFEST_DIR/"20_transitive_backbone_manifest.sha256"),
21:(cfg.MANIFEST_DIR/"21_adapter_contract_manifest.json",cfg.MANIFEST_DIR/"21_adapter_contract_manifest.sha256"),
22:(cfg.MANIFEST_DIR/"22_source_config_translation_manifest.json",cfg.MANIFEST_DIR/"22_source_config_translation_manifest.sha256"),
}
F16=cfg.MANIFEST_DIR/"16_stage1_model_registry_freeze.json"; F17=cfg.MANIFEST_DIR/"17_official_model_source_freeze.json"; F18=cfg.MANIFEST_DIR/"18_source_api_freeze.json"; F19=cfg.MANIFEST_DIR/"19_adapter_locus_freeze.json"; F20=cfg.MANIFEST_DIR/"20_transitive_backbone_freeze.json"; F21=cfg.MANIFEST_DIR/"21_adapter_contract_freeze.json"; F22=cfg.MANIFEST_DIR/"22_source_config_translation_freeze.json"; TEST_SEAL=cfg.MANIFEST_DIR/"10_final_test_seal.json"

SIG_CSV=cfg.AUDIT_DIR/"23_candidate_executable_signature_audit.csv"; RESULTS_CSV=cfg.AUDIT_DIR/"23_synthetic_smoke_results.csv"; REPO_BEFORE=cfg.AUDIT_DIR/"23_repository_recheck_before.csv"; REPO_CPU=cfg.AUDIT_DIR/"23_repository_recheck_after_cpu.csv"; REPO_CUDA=cfg.AUDIT_DIR/"23_repository_recheck_after_cuda.csv"; DEP_CSV=cfg.AUDIT_DIR/"23_runtime_dependency_versions.csv"; SOURCE_HASH_CSV=cfg.AUDIT_DIR/"23_source_file_hashes.csv"; LOG_DIR=cfg.AUDIT_DIR/"23_worker_logs"
FREEZE=cfg.MANIFEST_DIR/"23_stage1_model_smoke_freeze.json"; MANIFEST=cfg.MANIFEST_DIR/"23_stage1_model_smoke_manifest.json"; MANIFEST_SHA=cfg.MANIFEST_DIR/"23_stage1_model_smoke_manifest.sha256"; HASH_CSV=cfg.AUDIT_DIR/"23_stage1_model_smoke_hashes.csv"; FROZEN_SCRIPT=cfg.MANIFEST_DIR/"step23_synthetic_model_smoke_tests_frozen.py"; SUMMARY=cfg.REPORT_DIR/"23_stage1_model_smoke_summary.txt"; FAILED_DIR=cfg.AUDIT_DIR/"step23_failed_attempts"

def now(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def load(path):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Required JSON missing:\n{p}")
    return json.loads(p.read_text(encoding="utf-8"))
def atomic_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+".tmp"); q.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8"); os.replace(q,p)
def atomic_text(path,text):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+".tmp"); q.write_text(text,encoding="utf-8"); os.replace(q,p)
def write_csv(path,df):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+".tmp"); df.to_csv(q,index=False,float_format="%.17g"); os.replace(q,p)
def rel(path): return str(Path(path).resolve().relative_to(cfg.EXPERIMENT_ROOT_DIR.resolve()))
def art(path):
    p=Path(path); return {"path":str(p),"relative_path":rel(p),"size_bytes":int(p.stat().st_size),"sha256":sha(p)}
def norm_runtime(v):
    if isinstance(v,dict): return {k:norm_runtime(x) for k,x in v.items()}
    if isinstance(v,list): return [norm_runtime(x) for x in v]
    return "__DEVICE__" if v=="__RUNTIME_DEVICE__" else v
def gitexe():
    p=shutil.which("git")
    if not p: raise RuntimeError("git executable not found on PATH.")
    return p
def git(args,cwd):
    e=os.environ.copy(); e["GIT_TERMINAL_PROMPT"]="0"; r=subprocess.run([gitexe(),*args],cwd=str(cwd),env=e,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=120)
    if r.returncode: raise RuntimeError(f"GIT FAILED: git {' '.join(args)}\nCWD:{cwd}\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}")
    return r.stdout.strip()
def norm_url(s):
    s=s.strip().rstrip("/"); return (s[:-4] if s.endswith(".git") else s).lower()
def verify_sidecar(m,s,label):
    m=Path(m); s=Path(s)
    if not m.exists() or not s.exists(): raise RuntimeError(f"{label} manifest/sidecar missing.")
    exp=s.read_text(encoding="utf-8").strip().split()[0]; obs=sha(m)
    if exp!=obs: raise RuntimeError(f"{label} manifest SHA mismatch: expected {exp}, observed {obs}")
    return obs
def find_artifact(manifest,name):
    hits=[]
    for key in ("artifacts","canonical_artifacts"):
        for r in manifest.get(key,{}).values():
            if isinstance(r,dict) and r.get("path") and Path(r["path"]).name==name: hits.append(r)
    if len(hits)!=1: raise RuntimeError(f"Expected one manifest artifact named {name}; found {len(hits)}")
    return hits[0]
def verify_fp(path,key):
    obj=load(path); stored=obj.get(key)
    if not stored: raise RuntimeError(f"{Path(path).name} lacks {key}")
    body=dict(obj); body.pop(key,None); rec=cfg.canonical_json_sha256(body)
    if rec!=stored: raise RuntimeError(f"{Path(path).name}: fingerprint mismatch")
    return obj,stored

def verify_step00():
    cfg.validate_configuration(); payload=cfg.build_configuration_payload(); fp=cfg.canonical_json_sha256(payload); _,msha,status=cfg.verify_existing_freeze(configuration_payload=payload,configuration_fingerprint=fp)
    if status!="VERIFIED_EXISTING": raise RuntimeError("Step-00 freeze is not VERIFIED_EXISTING.")
    lp=cfg.get_final_test_lock_path(); lock=load(lp)
    if lock.get("state")!="LOCKED_NOT_OPENED" or lock.get("final_test_opened") is not False or lock.get("final_test_scored") is not False: raise RuntimeError("Global final-test state is not pristine.")
    return {"fingerprint":fp,"step00_manifest_sha256":msha,"lock_path":str(lp),"lock_sha256":sha(lp),"state":lock["state"]}
def verify_environment():
    a=load(ENV_AUDIT); exp=a.get("system",{}).get("python_executable")
    if exp and Path(exp).resolve()!=Path(sys.executable).resolve(): raise RuntimeError(f"Wrong Python. Frozen={exp}; current={sys.executable}")
    return {"python_executable":exp,"pandas_version":pd.__version__}
def verify_upstream(s0):
    shas={}; mans={}
    for i in range(1,23): shas[i]=verify_sidecar(*UP[i],f"Step {i:02d}"); mans[i]=load(UP[i][0])
    if mans[1].get("status")!="PASS" or mans[1].get("configuration_fingerprint_sha256")!=s0["fingerprint"]: raise RuntimeError("Step-01 status/fingerprint mismatch.")
    if mans[2].get("step",{}).get("status")!="PASS": raise RuntimeError("Step-02 status is not PASS.")
    for i in range(3,23):
        m=mans[i]
        if m.get("step",{}).get("status")!="PASS" or m.get("configuration_fingerprint_sha256")!=s0["fingerprint"]: raise RuntimeError(f"Step-{i:02d} status/fingerprint mismatch.")
        for j in range(1,i):
            f=f"step{j:02d}_manifest_sha256"
            if f in m and m[f]!=shas[j]: raise RuntimeError(f"Step-{i:02d} chain mismatch: {f}")
    verify_environment()
    f16,p16=verify_fp(F16,"registry_freeze_fingerprint_sha256"); f17,p17=verify_fp(F17,"source_freeze_fingerprint_sha256"); f18,p18=verify_fp(F18,"source_api_fingerprint_sha256"); f19,p19=verify_fp(F19,"adapter_locus_fingerprint_sha256"); f20,p20=verify_fp(F20,"transitive_backbone_fingerprint_sha256"); f21,p21=verify_fp(F21,"adapter_contract_fingerprint_sha256"); f22,p22=verify_fp(F22,"source_config_translation_fingerprint_sha256")
    expected=((f16.get("status"),"STAGE1_MODEL_REGISTRY_AND_SELECTION_PROTOCOL_FROZEN",16),(f17.get("status"),"OFFICIAL_AUTHOR_MAINTAINED_MODEL_SOURCE_SNAPSHOTS_FROZEN",17),(f18.get("status"),"FROZEN_SOURCE_APIS_AUDITED_BEFORE_ADAPTER_IMPLEMENTATION",18),(f19.get("status"),"ADAPTER_LOCI_AUDITED_BEFORE_IMPLEMENTATION",19),(f20.get("status"),"TRANSITIVE_BACKBONE_STRUCTURE_AUDITED_BEFORE_ADAPTER_IMPLEMENTATION",20),(f21.get("status"),"STAGE1_DIRECTIONAL_ADAPTER_CONTRACTS_FROZEN",21),(f22.get("status"),"STAGE1_CANDIDATE_SOURCE_CONFIGURATION_TRANSLATION_FROZEN",22))
    for obs,exp,i in expected:
        if obs!=exp: raise RuntimeError(f"Step-{i:02d} freeze status mismatch: {obs}")
    fps=((16,"registry_freeze_fingerprint_sha256",p16),(17,"source_freeze_fingerprint_sha256",p17),(18,"source_api_fingerprint_sha256",p18),(19,"adapter_locus_fingerprint_sha256",p19),(20,"transitive_backbone_fingerprint_sha256",p20),(21,"adapter_contract_fingerprint_sha256",p21),(22,"source_config_translation_fingerprint_sha256",p22))
    for i,k,v in fps:
        if mans[i].get(k)!=v: raise RuntimeError(f"Step-{i:02d} manifest/freeze fingerprint mismatch.")
    for path,m in ((F16,mans[16]),(F17,mans[17]),(F18,mans[18]),(F19,mans[19]),(F20,mans[20]),(F21,mans[21]),(F22,mans[22])):
        r=find_artifact(m,Path(path).name)
        if Path(r["path"]).resolve()!=Path(path).resolve() or r["sha256"]!=sha(path): raise RuntimeError(f"Frozen artifact drift: {Path(path).name}")
    if f16.get("training_started") is not False: raise RuntimeError("Step-16 says Stage-1 training already started.")
    contracts=f21.get("contracts",[]); ids=[str(x["model_id"]) for x in contracts]
    if len(ids)!=8 or len(set(ids))!=8 or set(ids)!=set(MODELS): raise RuntimeError("Step-21 adapter-contract model set mismatch.")
    st=f22.get("pipeline_state",{})
    if st.get("source_config_translation_frozen") is not True: raise RuntimeError("Step-22 translation not frozen.")
    for k in ("task_adapters_implemented","models_instantiated","forward_pass_run","backward_pass_run","smoke_tests_run","models_fitted","validation_performance_inspected","feature_selection_applied","probability_threshold_selection_performed"):
        if k in st and st.get(k) is not False: raise RuntimeError(f"Step-22 pre-smoke state violation: {k}")
    cp=f22.get("compatibility_policy",{})
    if cp.get("legacy_22b_22c_22d_stages_required") is not False or cp.get("all_compatibility_rules_frozen_in_this_step_before_execution") is not True: raise RuntimeError("Step-22 clean-V3 compatibility policy mismatch.")
    if cp.get("candidate_search_space_changed") is not False or cp.get("performance_information_used") is not False or cp.get("source_code_modified") is not False: raise RuntimeError("Step-22 performance-blind compatibility policy violated.")
    rec22=f22.get("candidate_source_configurations",[])
    if len(rec22)!=16: raise RuntimeError("Step-22 candidate source-config count is not 16.")
    counts={}
    for r in rec22: counts[r["model_id"]]=counts.get(r["model_id"],0)+1
    if set(counts)!=set(MODELS) or any(counts[m]!=2 for m in MODELS): raise RuntimeError(f"Step-22 candidate model/count mismatch: {counts}")
    rec20=f20.get("model_records",[]); ids20=[str(x["model_id"]) for x in rec20]
    if len(ids20)!=8 or len(set(ids20))!=8 or set(ids20)!=set(MODELS): raise RuntimeError("Step-20 model-record set mismatch.")
    repos=f17.get("repositories",[])
    if len(repos)!=8 or sum(x.get("repository_role")=="implementation" for x in repos)!=7 or sum(x.get("repository_role")=="reference_redirect" for x in repos)!=1: raise RuntimeError("Step-17 repository inventory drift.")
    seal=load(TEST_SEAL)
    if seal.get("status")!="SEALED_NOT_OPENED" or seal.get("global_final_test_lock_state")!="LOCKED_NOT_OPENED": raise RuntimeError("Step-10 final-test seal invalid.")
    return {"manifest_sha":shas,"manifests":mans,"freeze16":f16,"freeze17":f17,"freeze18":f18,"freeze19":f19,"freeze20":f20,"freeze21":f21,"freeze22":f22,"step20_by_model":{x["model_id"]:x for x in rec20},"step21_by_model":{x["model_id"]:x for x in contracts},"fingerprint16":p16,"fingerprint17":p17,"fingerprint18":p18,"fingerprint19":p19,"fingerprint20":p20,"fingerprint21":p21,"fingerprint22":p22,"test_seal_status":seal["status"]}

def recheck_repos(f17):
    rmap={}; rows=[]
    for r in f17["repositories"]:
        rid=r["repo_id"]; root=Path(r["local_path"])
        if rid in rmap or not root.exists() or not (root/".git").exists(): raise RuntimeError(f"Frozen repo missing/duplicate: {rid} {root}")
        origin=git(["remote","get-url","origin"],root); commit=git(["rev-parse","HEAD"],root); tree=git(["rev-parse","HEAD^{tree}"],root); branch=git(["branch","--show-current"],root); status=git(["status","--porcelain","--untracked-files=all"],root)
        oh=norm_url(origin)==norm_url(r["repository_url"]); ch=commit==r["commit_sha"]; th=tree==r["git_tree_sha"]; bh=branch==r["branch"]; clean=status==""; rh=True; n=0
        for fr in r["required_file_hashes"]:
            p=root/fr["relative_path"]
            if not p.is_file() or sha(p)!=fr["sha256"]: rh=False
            else: n+=1
        passed=oh and ch and th and bh and clean and rh
        if not passed: raise RuntimeError(f"Frozen repository recheck failed: {rid}; origin={oh}, commit={ch}, tree={th}, branch={bh}, clean={clean}, hashes={rh}")
        rmap[rid]=r; rows.append({"repo_id":rid,"repository_role":r["repository_role"],"local_path":str(root),"origin_match":oh,"commit_match":ch,"tree_match":th,"branch_match":bh,"worktree_clean_including_untracked":clean,"required_files_checked":n,"required_file_hashes_match":rh,"passed":passed})
    df=pd.DataFrame(rows)
    if len(df)!=8 or not df["passed"].all(): raise RuntimeError("Repository recheck structural failure.")
    return rmap,df

def assert_repo_clean(record):
    s=git(["status","--porcelain","--untracked-files=all"],Path(record["local_path"]))
    if s!="": raise RuntimeError(f"Frozen repository became dirty: {record['repo_id']}\n{s}")

def outputs(): return (SIG_CSV,RESULTS_CSV,REPO_BEFORE,REPO_CPU,REPO_CUDA,DEP_CSV,SOURCE_HASH_CSV,FREEZE,MANIFEST,MANIFEST_SHA,HASH_CSV,FROZEN_SCRIPT,SUMMARY)
def assert_fresh():
    hits=[p for p in outputs() if p.exists()]
    if LOG_DIR.exists(): hits.append(LOG_DIR)
    if hits: raise RuntimeError("Step-23 canonical output(s) already exist; refusing overwrite.\n"+"\n".join(map(str,hits))+"\nUse --verify-only if Step 23 already passed.")
def archive_partial():
    # Preserve a genuinely complete canonical freeze, but archive an incomplete
    # manifest/sidecar pair if a rare late write failure occurs.
    if MANIFEST.exists() and MANIFEST_SHA.exists():
        try:
            if MANIFEST_SHA.read_text(encoding="utf-8").strip().split()[0]==sha(MANIFEST):
                return None
        except Exception:
            pass
    items=[p for p in outputs() if p.exists()]
    if not items and not LOG_DIR.exists(): return None
    FAILED_DIR.mkdir(parents=True,exist_ok=True); d=FAILED_DIR/("partial_artifacts_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")); d.mkdir(exist_ok=False)
    for p in items:
        try: shutil.move(str(p),str(d/p.name))
        except Exception: pass
    if LOG_DIR.exists():
        try: shutil.move(str(LOG_DIR),str(d/LOG_DIR.name))
        except Exception: pass
    return d

def audit_signatures(f22):
    rows=[]; groups={}
    for r in f22["candidate_source_configurations"]:
        src=norm_runtime(r["source_config"]); tr=norm_runtime(r["training_config"]); a=cfg.canonical_json_sha256({"configuration_mode":r["configuration_mode"],"source_config":src}); full=cfg.canonical_json_sha256({"configuration_mode":r["configuration_mode"],"source_config":src,"training_config":tr}); groups.setdefault(r["model_id"],[]).append((r["config_id"],full)); rows.append({"model_id":r["model_id"],"config_id":r["config_id"],"architecture_signature_sha256":a,"full_executable_training_signature_sha256":full})
    if len(rows)!=16 or set(groups)!=set(MODELS) or any(len(groups[m])!=2 for m in MODELS): raise RuntimeError("Candidate signature structure mismatch.")
    dup=[]
    for m,pairs in groups.items():
        sig=[x[1] for x in pairs]
        if len(sig)!=len(set(sig)): dup.append((m,pairs))
    if dup: raise RuntimeError("Duplicate complete executable/training contracts: "+str(dup))
    df=pd.DataFrame(rows); df["architecture_duplicate_within_model"]=df.duplicated(["model_id","architecture_signature_sha256"],keep=False); return df
def candidate_order(f22):
    pairs=[]
    for m in MODELS:
        rr=sorted([x for x in f22["candidate_source_configurations"] if x["model_id"]==m],key=lambda x:x["config_id"])
        if len(rr)!=2: raise RuntimeError(f"{m}: expected two candidates")
        pairs += [(m,x["config_id"]) for x in rr]
    if len(pairs)!=16: raise RuntimeError("Candidate order is not 16 entries.")
    return pairs
def step20_map(f20):
    d={x["model_id"]:x for x in f20["model_records"]}
    if set(d)!=set(MODELS): raise RuntimeError("Step-20 model set mismatch.")
    return d

def import_target(model_id,record,root):
    relp=Path(record["primary_source_file"]); cls=record["preferred_class"]
    if model_id=="PATCHTST" and relp.parts and relp.parts[0]=="PatchTST_supervised": imp=root/"PatchTST_supervised"; modrel=Path(*relp.parts[1:])
    else: imp=root; modrel=relp
    if modrel.suffix!=".py": raise RuntimeError(f"Expected Python source: {modrel}")
    modname=".".join(modrel.with_suffix("").parts); os.chdir(imp); sys.path.insert(0,str(imp)); mod=importlib.import_module(modname); mf=Path(getattr(mod,"__file__","")).resolve()
    if not mf.is_relative_to(imp.resolve()): raise RuntimeError(f"Import did not resolve to frozen local repository: {model_id}; {mf}")
    return getattr(mod,cls),imp,modname,cls,mf

def worker_execute(model_id,config_id,device_name):
    import torch
    import torch.nn as nn
    started=time.perf_counter(); sys.dont_write_bytecode=True; os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    s0=verify_step00(); up=verify_upstream(s0); rmap={r["repo_id"]:r for r in up["freeze17"]["repositories"]}; mmap=step20_map(up["freeze20"])
    hits=[r for r in up["freeze22"]["candidate_source_configurations"] if r["model_id"]==model_id and r["config_id"]==config_id]
    if len(hits)!=1: raise RuntimeError(f"Step-22 candidate lookup failure: {model_id}/{config_id}, count={len(hits)}")
    rec=hits[0]; rid=rec["repo_id"]
    if rid!=EXPECTED_REPO[model_id] or up["step21_by_model"][model_id]["repo_id"]!=rid: raise RuntimeError(f"{model_id}: repository mapping mismatch")
    rrec=rmap[rid]; root=Path(rrec["local_path"]); assert_repo_clean(rrec)
    if device_name=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA smoke requested but CUDA unavailable.")
        device=torch.device("cuda:0")
    elif device_name=="cpu": device=torch.device("cpu")
    else: raise ValueError(device_name)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"; torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    try: torch.backends.cuda.matmul.allow_tf32=False
    except Exception: pass
    try: torch.backends.cudnn.allow_tf32=False
    except Exception: pass
    if hasattr(torch,"set_float32_matmul_precision"): torch.set_float32_matmul_precision("highest")
    sc=copy.deepcopy(rec["source_config"])
    if sc.get("device")=="__RUNTIME_DEVICE__": sc["device"]=device
    cls,imp,modname,pclass,mfile=import_target(model_id,mmap[model_id],root); mode=rec["configuration_mode"]
    if mode=="direct_constructor_kwargs": backbone=cls(**sc)
    elif mode=="config_namespace": backbone=cls(SimpleNamespace(**sc))
    else: raise RuntimeError(f"Unknown configuration_mode: {mode}")
    class Adapter(nn.Module):
        def __init__(self,mid,bb):
            super().__init__(); self.model_id=mid; self.backbone=bb; self.readout=None if mid=="TIMESNET" else nn.Linear(READOUT_IN,OUTPUTS,bias=True)
        def forward(self,x):
            if self.model_id=="TIMESNET":
                mask=torch.ones((x.shape[0],x.shape[1]),dtype=x.dtype,device=x.device); logits=self.backbone(x,mask,None,None)
                if not isinstance(logits,torch.Tensor) or tuple(logits.shape)!=(x.shape[0],OUTPUTS): raise RuntimeError(f"TimesNet output contract mismatch: {type(logits)} {getattr(logits,'shape',None)}")
                return logits,None
            if self.model_id=="PATCHTST":
                forecast=self.backbone(x.transpose(1,2)); exp=(x.shape[0],FEATURES,FORECAST_LEN)
                if not isinstance(forecast,torch.Tensor) or tuple(forecast.shape)!=exp: raise RuntimeError(f"PatchTST raw shape mismatch: expected {exp}, got {getattr(forecast,'shape',None)}")
                forecast=forecast.transpose(1,2).contiguous()
            elif self.model_id=="ITRANSFORMER":
                forecast=self.backbone(x,None,None,None)
                if isinstance(forecast,(tuple,list)):
                    if not forecast: raise RuntimeError("iTransformer returned empty tuple/list")
                    forecast=forecast[0]
            elif self.model_id=="MLF":
                forecast=self.backbone(x,None,None,None)
                if not isinstance(forecast,(tuple,list)) or not forecast: raise RuntimeError("MLF output must be nonempty tuple/list")
                forecast=forecast[0]
            else: forecast=self.backbone(x)
            exp=(x.shape[0],FORECAST_LEN,FEATURES)
            if not isinstance(forecast,torch.Tensor) or tuple(forecast.shape)!=exp: raise RuntimeError(f"{self.model_id} forecast shape mismatch: expected {exp}, got {getattr(forecast,'shape',None)}")
            flat=forecast.reshape(forecast.shape[0],-1)
            if flat.shape[1]!=READOUT_IN: raise RuntimeError("Flattened forecast dimension drift")
            logits=self.readout(flat)
            if tuple(logits.shape)!=(x.shape[0],OUTPUTS): raise RuntimeError("Directional logit shape mismatch")
            return logits,forecast
    model=Adapter(model_id,backbone).to(device)
    for name,p in model.named_parameters():
        if p.device!=device: raise RuntimeError(f"Parameter device mismatch: {name}")
        if p.is_floating_point() and p.dtype!=torch.float32: raise RuntimeError(f"Non-FP32 parameter: {name} {p.dtype}")
    adapter_params=0 if model_id=="TIMESNET" else sum(p.numel() for p in model.readout.parameters())
    if model_id=="TIMESNET" and model.readout is not None: raise RuntimeError("TimesNet received external readout")
    if model_id!="TIMESNET" and adapter_params!=READOUT_PARAMS: raise RuntimeError(f"{model_id}: adapter params {adapter_params} != {READOUT_PARAMS}")
    g=torch.Generator(device="cpu"); g.manual_seed(SEED); xcpu=torch.randn((BATCH,LOOKBACK,FEATURES),generator=g,dtype=torch.float32); ycpu=torch.tensor([[0.,1.,0.],[1.,0.,1.]],dtype=torch.float32); x=xcpu.to(device); y=ycpu.to(device)
    model.eval(); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    with torch.no_grad(): la,fa=model(x)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    with torch.no_grad(): lb,fb=model(x)
    det_logits=bool(torch.equal(la,lb)); det_forecast=True
    if not det_logits: raise RuntimeError(f"{model_id}/{config_id}/{device_name}: deterministic logits repeat failed; maxdiff={(la-lb).abs().max().item():.6e}")
    if fa is not None or fb is not None:
        if fa is None or fb is None: raise RuntimeError("Forecast packaging changed across deterministic repeat")
        det_forecast=bool(torch.equal(fa,fb))
        if not det_forecast: raise RuntimeError(f"{model_id}/{config_id}/{device_name}: deterministic forecast repeat failed; maxdiff={(fa-fb).abs().max().item():.6e}")
    model.train(); model.zero_grad(set_to_none=True); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    if device.type=="cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    t0=time.perf_counter(); logits,forecast=model(x)
    if device.type=="cuda": torch.cuda.synchronize(device)
    fsec=time.perf_counter()-t0
    if logits.dtype!=torch.float32 or logits.device!=device or tuple(logits.shape)!=(BATCH,OUTPUTS) or not bool(torch.isfinite(logits).all().item()): raise RuntimeError("Final logits dtype/device/shape/finiteness failure")
    if forecast is not None and (forecast.dtype!=torch.float32 or forecast.device!=device or not bool(torch.isfinite(forecast).all().item())): raise RuntimeError("Forecast dtype/device/finiteness failure")
    loss=nn.BCEWithLogitsLoss(reduction="mean")(logits,y)
    if not bool(torch.isfinite(loss).item()): raise RuntimeError("Non-finite BCE loss")
    if device.type=="cuda": torch.cuda.synchronize(device)
    t0=time.perf_counter(); loss.backward()
    if device.type=="cuda": torch.cuda.synchronize(device)
    bsec=time.perf_counter()-t0
    total=trainable=gt=fg=nzg=bgt=rgt=bge=rge=0
    for name,p in model.named_parameters():
        total+=p.numel(); trainable+=p.numel() if p.requires_grad else 0
        if p.requires_grad and p.grad is not None:
            gt+=1
            if not bool(torch.isfinite(p.grad).all().item()): raise RuntimeError(f"Non-finite gradient: {name}")
            fg+=1; nz=int(torch.count_nonzero(p.grad).item())
            if nz>0:
                nzg+=1
                if name.startswith("backbone."): bgt+=1; bge+=nz
                elif name.startswith("readout."): rgt+=1; rge+=nz
    if gt==0 or fg!=gt or nzg==0 or bgt==0: raise RuntimeError("Gradient-flow smoke failed")
    if model_id!="TIMESNET" and rgt==0: raise RuntimeError("External readout received no non-zero gradient")
    if model_id=="TIMESNET" and rgt!=0: raise RuntimeError("TimesNet unexpectedly has external-readout gradient")
    peak=int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0; assert_repo_clean(rrec)
    return {"status":"PASS","model_id":model_id,"config_id":config_id,"repo_id":rid,"device":device_name,"python_executable":sys.executable,"torch_version":torch.__version__,"cuda_available":bool(torch.cuda.is_available()),"cuda_device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "","import_root":str(imp),"module_name":modname,"resolved_module_file":str(mfile),"preferred_class":pclass,"configuration_mode":mode,"input_shape":[BATCH,LOOKBACK,FEATURES],"forecast_shape":list(forecast.shape) if forecast is not None else None,"logit_shape":list(logits.shape),"input_dtype":str(x.dtype),"logit_dtype":str(logits.dtype),"loss":float(loss.detach().cpu().item()),"loss_reduction":"mean","total_parameters":int(total),"trainable_parameters":int(trainable),"adapter_parameter_count":int(adapter_params),"gradient_parameter_tensors":int(gt),"finite_gradient_tensors":int(fg),"nonzero_gradient_tensors":int(nzg),"backbone_nonzero_gradient_tensors":int(bgt),"backbone_nonzero_gradient_elements":int(bge),"readout_nonzero_gradient_tensors":int(rgt),"readout_nonzero_gradient_elements":int(rge),"deterministic_logits_exact":det_logits,"deterministic_forecast_exact":det_forecast,"deterministic_algorithms_enabled":bool(torch.are_deterministic_algorithms_enabled()),"amp_enabled":False,"float32_only_contract":True,"forward_seconds":float(fsec),"backward_seconds":float(bsec),"elapsed_seconds":float(time.perf_counter()-started),"peak_cuda_memory_bytes":peak,"repository_clean_after_worker":True,"synthetic_only":True,"optimizer_step_run":False,"model_fitted":False,"validation_data_opened":False,"final_test_data_opened":False}

def worker_main(model_id,config_id,device):
    try:
        r=worker_execute(model_id,config_id,device); print(RESULT_PREFIX+json.dumps(r,sort_keys=True,ensure_ascii=False)); return 0
    except Exception as e:
        r={"status":"FAIL","model_id":model_id,"config_id":config_id,"device":device,"exception_type":type(e).__name__,"exception_message":str(e),"missing_module":str(getattr(e,"name","")) if isinstance(e,ModuleNotFoundError) else "","traceback":traceback.format_exc(),"synthetic_only":True,"optimizer_step_run":False,"model_fitted":False,"validation_data_opened":False,"final_test_data_opened":False}; print(RESULT_PREFIX+json.dumps(r,sort_keys=True,ensure_ascii=False)); return 1

def parse_worker(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX): return json.loads(line[len(RESULT_PREFIX):])
    return None
def run_worker(model,config,device):
    env=os.environ.copy(); env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"; env["PYTHONDONTWRITEBYTECODE"]="1"; env["PYTHONHASHSEED"]=str(SEED); cmd=[sys.executable,str(SCRIPT_FILE),"--worker","--model-id",model,"--config-id",config,"--device",device]
    try:
        c=subprocess.run(cmd,cwd=str(SCRIPT_DIR),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=TIMEOUT); r=parse_worker(c.stdout)
        if r is None: r={"status":"FAIL","model_id":model,"config_id":config,"device":device,"exception_type":"WorkerResultMissing","exception_message":"Worker exited without Step-23 result sentinel.","missing_module":"","traceback":"","synthetic_only":True,"optimizer_step_run":False,"model_fitted":False,"validation_data_opened":False,"final_test_data_opened":False}
        return r,c.stdout,c.stderr,int(c.returncode)
    except subprocess.TimeoutExpired as e:
        return {"status":"FAIL","model_id":model,"config_id":config,"device":device,"exception_type":"TimeoutExpired","exception_message":f"Worker exceeded {TIMEOUT} seconds.","missing_module":"","traceback":"","synthetic_only":True,"optimizer_step_run":False,"model_fitted":False,"validation_data_opened":False,"final_test_data_opened":False}, e.stdout if isinstance(e.stdout,str) else "", e.stderr if isinstance(e.stderr,str) else "",124
def write_log(model,config,device,stdout,stderr,rc):
    LOG_DIR.mkdir(parents=True,exist_ok=True); p=LOG_DIR/f"{model}__{config}__{device}.log"; p.write_text(f"return_code={rc}\n\n===== STDOUT =====\n{stdout}\n===== STDERR =====\n{stderr}",encoding="utf-8"); return p

def deps():
    rows=[]
    for name in ("torch","numpy","pandas","scipy","einops","reformer-pytorch"):
        try: v=package_version(name); ok=True
        except PackageNotFoundError: v="NOT_INSTALLED"; ok=False
        rows.append({"distribution":name,"version":v,"installed":ok})
    return pd.DataFrame(rows)
def source_hashes():
    src=[("current_step00",Path(cfg.__file__).resolve()),("current_step23",SCRIPT_FILE),("step16_registry_freeze",F16),("step17_source_freeze",F17),("step18_source_api_freeze",F18),("step19_adapter_locus_freeze",F19),("step20_transitive_backbone_freeze",F20),("step21_adapter_contract_freeze",F21),("step22_source_config_translation_freeze",F22),("step10_final_test_seal_metadata",TEST_SEAL)]
    for i,(m,s) in UP.items(): src += [(f"step{i:02d}_manifest",m.resolve()),(f"step{i:02d}_manifest_sha",s.resolve())]
    rows=[]
    for role,p in src:
        if p.name==FORBIDDEN_TEST_FILENAME: raise RuntimeError("Final-test parquet entered Step-23 source list")
        if not p.exists(): raise RuntimeError(f"Missing Step-23 source: {p}")
        rows.append({"role":role,"file_name":p.name,"absolute_path":str(p),"size_bytes":int(p.stat().st_size),"sha256":sha(p)})
    return pd.DataFrame(rows)
def validate_results(df):
    if len(df)!=32 or not (df["status"]=="PASS").all() or (df["device"]=="cpu").sum()!=16 or (df["device"]=="cuda").sum()!=16: raise RuntimeError("Smoke result matrix/count failure")
    for c in ("deterministic_logits_exact","deterministic_forecast_exact","repository_clean_after_worker"):
        if not df[c].all(): raise RuntimeError(f"Smoke result flag failed: {c}")
    if df["optimizer_step_run"].any() or df["model_fitted"].any() or df["validation_data_opened"].any() or df["final_test_data_opened"].any(): raise RuntimeError("Protection flag violation in smoke results")
    for m in MODELS:
        for c in sorted(df.loc[df.model_id==m,"config_id"].unique()):
            p=df[(df.model_id==m)&(df.config_id==c)]
            if set(p.device)!={"cpu","cuda"}: raise RuntimeError(f"Incomplete CPU/CUDA pair: {m}/{c}")
            a=p[p.device=="cpu"].iloc[0]; b=p[p.device=="cuda"].iloc[0]
            for f in ("total_parameters","trainable_parameters","adapter_parameter_count"):
                if int(a[f])!=int(b[f]): raise RuntimeError(f"CPU/CUDA {f} mismatch: {m}/{c}")
            exp=0 if m=="TIMESNET" else READOUT_PARAMS
            if int(a.adapter_parameter_count)!=exp or int(b.adapter_parameter_count)!=exp: raise RuntimeError(f"Adapter param mismatch: {m}/{c}")

def verify_existing():
    print("="*136); print("STEP 23 — VERIFY EXISTING SYNTHETIC MODEL-SMOKE FREEZE"); print("="*136)
    s0=verify_step00(); up=verify_upstream(s0); msha=verify_sidecar(MANIFEST,MANIFEST_SHA,"Step 23"); m=load(MANIFEST)
    if m.get("step",{}).get("status")!="PASS" or m.get("configuration_fingerprint_sha256")!=s0["fingerprint"]: return 20
    for i in range(1,23):
        if m.get(f"step{i:02d}_manifest_sha256")!=up["manifest_sha"][i]: print(f"[FAIL] Step-{i:02d} chain mismatch"); return 21
    checks={"step16_registry_freeze_fingerprint_sha256":up["fingerprint16"],"step17_source_freeze_fingerprint_sha256":up["fingerprint17"],"step18_source_api_fingerprint_sha256":up["fingerprint18"],"step19_adapter_locus_fingerprint_sha256":up["fingerprint19"],"step20_transitive_backbone_fingerprint_sha256":up["fingerprint20"],"step21_adapter_contract_fingerprint_sha256":up["fingerprint21"],"step22_source_config_translation_fingerprint_sha256":up["fingerprint22"]}
    if any(m.get(k)!=v for k,v in checks.items()): print("[FAIL] Upstream fingerprint mismatch"); return 22
    for name,r in m.get("artifacts",{}).items():
        p=Path(r["path"])
        if not p.exists() or sha(p)!=r["sha256"]: print(f"[FAIL] artifact {name}"); return 23
    fr,fp=verify_fp(FREEZE,"stage1_model_smoke_fingerprint_sha256")
    if m.get("stage1_model_smoke_fingerprint_sha256")!=fp: return 24
    if fr.get("status")!="STAGE1_MODEL_SYNTHETIC_SMOKE_TESTS_FROZEN": return 24
    rs=fr.get("results_summary",{})
    if int(rs.get("records",-1))!=32 or int(rs.get("cpu_records",-1))!=16 or int(rs.get("cuda_records",-1))!=16 or int(rs.get("passed_records",-1))!=32: return 24
    if fr.get("final_test_protection",{}).get("global_final_test_state")!="LOCKED_NOT_OPENED": return 24
    if len([k for k in m.get("artifacts",{}) if k.startswith("worker_log::")])!=32: return 24
    df=pd.read_csv(RESULTS_CSV)
    if len(df)!=32 or not (df.status=="PASS").all() or (df.device=="cpu").sum()!=16 or (df.device=="cuda").sum()!=16: return 25
    rmap,rdf=recheck_repos(up["freeze17"])
    if sha(Path(s0["lock_path"]))!=s0["lock_sha256"]: return 26
    print(f"[PASS] Configuration fingerprint : {s0['fingerprint']}"); print(f"[PASS] Step-22 manifest          : {up['manifest_sha'][22]}"); print(f"[PASS] Step-21 adapter freeze    : {up['fingerprint21']}"); print(f"[PASS] Step-22 source-config freeze: {up['fingerprint22']}"); print(f"[PASS] Frozen repositories       : {len(rmap)}"); print(f"[PASS] Smoke records             : {len(df)}"); print("[PASS] CPU candidate passes      : 16"); print("[PASS] CUDA candidate passes     : 16"); print(f"[PASS] Step-23 smoke freeze      : {fp}"); print(f"[PASS] Step-23 manifest          : {msha}"); print(f"[PASS] Step-10 test seal         : {up['test_seal_status']}"); print(f"[PASS] Global final-test state   : {s0['state']}"); print("="*136); print("STEP 23 VERIFICATION STATUS: PASS"); print("="*136); return 0

def run_master():
    print("="*140); print(STEP_NAME); print("="*140); stage="verify"
    try:
        print("[1/11] Verifying Step 00 and immutable Step 01 -> Step 22 chain..."); s0=verify_step00(); up=verify_upstream(s0); print(f"        Configuration fingerprint : {s0['fingerprint']}"); print(f"        Step-22 manifest          : {up['manifest_sha'][22]}"); print(f"        Step-21 adapter freeze    : {up['fingerprint21']}"); print(f"        Step-22 source-config freeze: {up['fingerprint22']}"); print(f"        Step-10 test seal         : {up['test_seal_status']}"); print(f"        Global final-test state   : {s0['state']}"); print("        Legacy Step 22B/22C/22D  : NOT USED"); print("        Development/validation/final-test arrays: NOT OPENED")
        stage="collision_guard"; print("[2/11] Checking immutable Step-23 output target..."); assert_fresh(); cfg.create_directories()
        stage="repo_before"; print("[3/11] Rechecking all eight frozen Step-17 repositories..."); rmap,r0=recheck_repos(up["freeze17"]); write_csv(REPO_BEFORE,r0); print(f"        Frozen repositories        : {len(r0)}"); print(f"        Worktrees clean incl untracked: {bool(r0.worktree_clean_including_untracked.all())}")
        stage="signatures"; print("[4/11] Auditing all 16 frozen executable/training signatures before third-party imports..."); sdf=audit_signatures(up["freeze22"]); write_csv(SIG_CSV,sdf); archdup=int(sdf.architecture_duplicate_within_model.sum()); pairs=candidate_order(up["freeze22"]); print(f"        Candidate configurations    : {len(pairs)}"); print("        Duplicate complete contracts: 0"); print(f"        Architecture-duplicate rows : {archdup}"); print("        Third-party model imported  : False")
        stage="deps"; print("[5/11] Recording runtime dependency versions..."); ddf=deps(); write_csv(DEP_CSV,ddf)
        for row in ddf.itertuples(index=False): print(f"        {row.distribution:<18}: {row.version}")
        results=[]
        stage="cpu"; print("\n[6/11] PHASE 1 — CPU synthetic forward/backward for all 16 candidates..."); print("-"*140)
        for model,conf in pairs:
            print(f"RUN  {model:<14} {conf:<20} device=cpu"); res,out,err,rc=run_worker(model,conf,"cpu"); lp=write_log(model,conf,"cpu",out,err,rc); res["worker_log"]=str(lp); res["return_code"]=rc; results.append(res); write_csv(RESULTS_CSV,pd.DataFrame(results))
            if res.get("status")!="PASS" or rc!=0: print(f"FAIL {model}/{conf}: {res.get('exception_type')}: {res.get('exception_message')}\nWorker log: {lp}"); raise RuntimeError(f"CPU SYNTHETIC SMOKE FAILED: {model}/{conf}")
            print(f"PASS {model:<14} {conf:<20} params={res['trainable_parameters']:,} loss={res['loss']:.6f} deterministic={res['deterministic_logits_exact']}")
        stage="repo_cpu"; print("[7/11] Rechecking repositories after CPU phase..."); _,rcpu=recheck_repos(up["freeze17"]); write_csv(REPO_CPU,rcpu); print(f"        Worktrees clean incl untracked: {bool(rcpu.worktree_clean_including_untracked.all())}")
        stage="cuda"; print("\n[8/11] PHASE 2 — CUDA synthetic forward/backward for all 16 candidates..."); print("-"*140)
        for model,conf in pairs:
            print(f"RUN  {model:<14} {conf:<20} device=cuda"); res,out,err,rc=run_worker(model,conf,"cuda"); lp=write_log(model,conf,"cuda",out,err,rc); res["worker_log"]=str(lp); res["return_code"]=rc; results.append(res); write_csv(RESULTS_CSV,pd.DataFrame(results))
            if res.get("status")!="PASS" or rc!=0: print(f"FAIL {model}/{conf}: {res.get('exception_type')}: {res.get('exception_message')}\nWorker log: {lp}"); raise RuntimeError(f"CUDA SYNTHETIC SMOKE FAILED: {model}/{conf}")
            print(f"PASS {model:<14} {conf:<20} params={res['trainable_parameters']:,} loss={res['loss']:.6f} peak_cuda={res['peak_cuda_memory_bytes']/(1024**2):.1f} MiB deterministic={res['deterministic_logits_exact']}")
        stage="repo_cuda"; print("[9/11] Rechecking repositories after CUDA phase..."); _,rcuda=recheck_repos(up["freeze17"]); write_csv(REPO_CUDA,rcuda); print(f"        Worktrees clean incl untracked: {bool(rcuda.worktree_clean_including_untracked.all())}")
        stage="validate"; print("[10/11] Validating complete 32-record smoke matrix..."); rdf=pd.DataFrame(results); validate_results(rdf); write_csv(RESULTS_CSV,rdf)
        if sha(Path(s0["lock_path"]))!=s0["lock_sha256"]: raise RuntimeError("Global final-test lock changed during Step 23.")
        print(f"        Smoke records             : {len(rdf)}"); print(f"        Deterministic repeats     : {int(rdf.deterministic_logits_exact.sum())}/{len(rdf)}"); print(f"        Zero-backbone-grad cases  : {int((rdf.backbone_nonzero_gradient_tensors==0).sum())}")
        stage="freeze"; print("[11/11] Freezing successful Step-23 synthetic smoke evidence..."); write_csv(SOURCE_HASH_CSV,source_hashes())
        compact=[]
        keep=("model_id","config_id","device","status","repo_id","configuration_mode","input_shape","forecast_shape","logit_shape","input_dtype","logit_dtype","total_parameters","trainable_parameters","adapter_parameter_count","gradient_parameter_tensors","nonzero_gradient_tensors","backbone_nonzero_gradient_tensors","readout_nonzero_gradient_tensors","deterministic_logits_exact","deterministic_forecast_exact","repository_clean_after_worker","synthetic_only","optimizer_step_run","model_fitted","validation_data_opened","final_test_data_opened")
        for r in results: compact.append({k:r[k] for k in keep})
        freeze={"schema_version":SCHEMA_VERSION,"status":"STAGE1_MODEL_SYNTHETIC_SMOKE_TESTS_FROZEN","generated_at_utc":now(),"configuration_fingerprint_sha256":s0["fingerprint"],"step16_manifest_sha256":up["manifest_sha"][16],"step17_manifest_sha256":up["manifest_sha"][17],"step18_manifest_sha256":up["manifest_sha"][18],"step19_manifest_sha256":up["manifest_sha"][19],"step20_manifest_sha256":up["manifest_sha"][20],"step21_manifest_sha256":up["manifest_sha"][21],"step22_manifest_sha256":up["manifest_sha"][22],"step16_registry_freeze_fingerprint_sha256":up["fingerprint16"],"step17_source_freeze_fingerprint_sha256":up["fingerprint17"],"step18_source_api_fingerprint_sha256":up["fingerprint18"],"step19_adapter_locus_fingerprint_sha256":up["fingerprint19"],"step20_transitive_backbone_fingerprint_sha256":up["fingerprint20"],"step21_adapter_contract_fingerprint_sha256":up["fingerprint21"],"step22_source_config_translation_fingerprint_sha256":up["fingerprint22"],"smoke_policy":{"synthetic_batch_size":BATCH,"lookback_sessions":LOOKBACK,"input_features":FEATURES,"forecast_length":FORECAST_LEN,"directional_outputs":OUTPUTS,"directional_horizons":[1,3,5],"loss":"BCEWithLogitsLoss(reduction='mean')","seed":SEED,"devices":["cpu","cuda"],"worker_isolation":True,"worker_timeout_seconds":TIMEOUT,"cublas_workspace_config":":4096:8","python_dont_write_bytecode":True,"deterministic_algorithms":True,"deterministic_repeat_required":True,"amp":False,"dtype":"float32","tf32_disabled":True,"forecast_model_external_readout":"Linear(315,3,bias=True)","forecast_model_added_parameters":READOUT_PARAMS,"timesnet_external_readout":False,"timesnet_three_outputs_semantics":"independent 1d/3d/5d binary-direction logits","legacy_step22b_22c_22d_used":False,"optimizer_step_run":False,"performance_metrics_computed":False,"validation_data_used":False,"final_test_used":False},"candidate_distinctness":{"candidate_count":len(sdf),"duplicate_full_executable_training_contracts":0,"architecture_duplicate_records":archdup,"architecture_duplicates_are_not_automatically_invalid":True},"results_summary":{"records":len(rdf),"cpu_records":int((rdf.device=="cpu").sum()),"cuda_records":int((rdf.device=="cuda").sum()),"passed_records":int((rdf.status=="PASS").sum()),"stage1_models":int(rdf.model_id.nunique()),"candidate_configurations":16,"deterministic_repeat_passes":int(rdf.deterministic_logits_exact.sum()),"zero_backbone_gradient_records":int((rdf.backbone_nonzero_gradient_tensors==0).sum()),"repository_mutation_records":int((~rdf.repository_clean_after_worker).sum())},"smoke_records":compact,"pipeline_state":{"development_interface_frozen":True,"stage1_model_registry_frozen":True,"official_source_snapshots_frozen":True,"source_apis_audited":True,"adapter_loci_audited":True,"transitive_backbones_audited":True,"adapter_contracts_frozen":True,"source_config_translation_frozen":True,"task_adapters_implemented_for_smoke":True,"third_party_models_imported":True,"models_instantiated":True,"synthetic_forward_smoke_tests_run":True,"synthetic_backward_smoke_tests_run":True,"optimizer_steps_run":False,"models_fitted":False,"validation_performance_inspected":False,"feature_selection_applied":False,"probability_threshold_selection_performed":False},"final_test_protection":{"step10_seal_status":up["test_seal_status"],"final_test_parquet_opened_or_hashed_by_step23":False,"final_test_performance_computed":False,"global_final_test_state":s0["state"]}}
        fp=cfg.canonical_json_sha256(freeze); freeze["stage1_model_smoke_fingerprint_sha256"]=fp; atomic_json(FREEZE,freeze); shutil.copy2(SCRIPT_FILE,FROZEN_SCRIPT); atomic_text(SUMMARY,"\n".join([STEP_NAME,"="*100,"Status: PASS",f"Configuration fingerprint: {s0['fingerprint']}",f"Step-22 source-config fingerprint: {up['fingerprint22']}","Stage-1 models: 8","Candidate configurations: 16","CPU forward/backward: 16/16","CUDA forward/backward: 16/16",f"Deterministic repeats: {int(rdf.deterministic_logits_exact.sum())}/{len(rdf)}",f"Forecast adapter parameters: {READOUT_PARAMS}","Legacy 22B/22C/22D used: False","Optimizer steps: 0","Models fitted: False","Validation performance inspected: False","Final-test information used: False","Repository mutation: False",f"Step-10 final-test seal: {up['test_seal_status']}",f"Global final-test state: {s0['state']}"])+"\n")
        logs=sorted(LOG_DIR.glob("*.log"))
        if len(logs)!=32: raise RuntimeError(f"Expected 32 worker logs; found {len(logs)}")
        pre=(SIG_CSV,RESULTS_CSV,REPO_BEFORE,REPO_CPU,REPO_CUDA,DEP_CSV,SOURCE_HASH_CSV,FREEZE,FROZEN_SCRIPT,SUMMARY,*logs); write_csv(HASH_CSV,pd.DataFrame([{"file_name":p.name,"file_path":str(p),"relative_path":rel(p),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in pre])); arts={}
        for p in (*pre,HASH_CSV): arts[("worker_log::"+p.name) if p.parent.resolve()==LOG_DIR.resolve() else p.name]=art(p)
        man={"schema_version":SCHEMA_VERSION,"step":{"step_number":23,"step_id":"23","step_name":STEP_NAME,"status":"PASS"},"frozen_at_utc":now(),"experiment_tag":cfg.EXPERIMENT_TAG,"configuration_fingerprint_sha256":s0["fingerprint"],"step00_manifest_sha256":s0["step00_manifest_sha256"],**{f"step{i:02d}_manifest_sha256":up["manifest_sha"][i] for i in range(1,23)},"step16_registry_freeze_fingerprint_sha256":up["fingerprint16"],"step17_source_freeze_fingerprint_sha256":up["fingerprint17"],"step18_source_api_fingerprint_sha256":up["fingerprint18"],"step19_adapter_locus_fingerprint_sha256":up["fingerprint19"],"step20_transitive_backbone_fingerprint_sha256":up["fingerprint20"],"step21_adapter_contract_fingerprint_sha256":up["fingerprint21"],"step22_source_config_translation_fingerprint_sha256":up["fingerprint22"],"stage1_model_smoke_fingerprint_sha256":fp,"counts":{"stage1_models":8,"candidate_configurations":16,"smoke_records":32,"cpu_passes":16,"cuda_passes":16,"worker_logs":32},"pipeline_state":freeze["pipeline_state"],"test_protection":{"step10_seal_status":up["test_seal_status"],"global_final_test_state":s0["state"],"sealed_test_parquet_opened_or_hashed_by_step23":False,"final_test_performance_computed":False},"artifacts":arts}; atomic_json(MANIFEST,man); msha=sha(MANIFEST); atomic_text(MANIFEST_SHA,f"{msha}  {MANIFEST.name}\n")
        maxmem=float(rdf.loc[rdf.device=="cuda","peak_cuda_memory_bytes"].max()/(1024**2)); print("\n"+"-"*140); print("SMOKE TEST SUMMARY"); print("-"*140); print("Stage-1 models                    : 8"); print("Candidate configurations          : 16"); print("CPU forward/backward passes       : 16/16"); print("CUDA forward/backward passes      : 16/16"); print("Total smoke records               : 32"); print("Non-finite logits/loss/gradients  : 0"); print("Candidates with zero backbone grad: 0"); print("Complete executable duplicates    : 0"); print(f"Architecture-duplicate rows       : {archdup}"); print(f"Deterministic repeat passes       : {int(rdf.deterministic_logits_exact.sum())}/32"); print(f"Maximum CUDA allocation           : {maxmem:.1f} MiB"); print("-"*140); print("Legacy Steps 22B/22C/22D           : NOT USED"); print("Repository mutation                : False"); print("Optimizer steps                    : 0"); print("Models fitted                      : False"); print("Validation performance inspected   : False"); print("Final test accessed                : False"); print(f"Step-10 final-test seal            : {up['test_seal_status']}"); print(f"Global final-test state            : {s0['state']}"); print("-"*140); print(f"Smoke freeze           : {FREEZE}"); print(f"Smoke fingerprint      : {fp}"); print(f"Manifest               : {MANIFEST}"); print(f"Manifest SHA-256       : {msha}"); print("="*140); print("PASS: All 16 frozen Stage-1 A/B candidates completed isolated synthetic CPU and CUDA forward/backward smoke tests."); print("PASS: [B,60,63] -> [B,3], FP32, finite BCE loss, deterministic repeats, and finite non-zero backbone gradients were verified."); print("PASS: Seven forecasting models used exactly the frozen 948-parameter readout; TimesNet used its native three-output pathway."); print("PASS: No repository mutation, optimizer step, fitting, validation scoring, threshold selection, or final-test access occurred."); print("="*140); print("Do not begin Stage-1 fitting yet. Paste the complete Step-23 output for audit, then run --verify-only if instructed."); return 0
    except Exception as e:
        archived=archive_partial(); FAILED_DIR.mkdir(parents=True,exist_ok=True); p=FAILED_DIR/("23_synthetic_smoke_FAILED_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".json"); atomic_json(p,{"step":STEP_NAME,"status":"FAIL","failed_at_utc":now(),"stage":stage,"error_type":type(e).__name__,"error_message":str(e),"python_executable":sys.executable,"archived_partial_artifacts":None if archived is None else str(archived),"canonical_manifest_exists":MANIFEST.exists()}); print("\n"+"="*140); print("STEP 23 FAILED — SYNTHETIC MODEL-SMOKE EVIDENCE NOT FROZEN"); print("="*140); print(f"Stage      : {stage}"); print(f"Error type : {type(e).__name__}"); print(f"Error      : {e}"); print(f"Diagnostic : {p}");
        if archived is not None: print(f"Archived partial Step-23 outputs: {archived}")
        print("="*140); print("Do not proceed to Stage-1 model fitting."); return 30

def parse_args():
    p=argparse.ArgumentParser(description=STEP_NAME); p.add_argument("--verify-only",action="store_true"); p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS); p.add_argument("--model-id",choices=list(MODELS),default=None,help=argparse.SUPPRESS); p.add_argument("--config-id",default=None,help=argparse.SUPPRESS); p.add_argument("--device",choices=["cpu","cuda"],default=None,help=argparse.SUPPRESS); return p.parse_args()
def main():
    a=parse_args()
    if a.worker:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8"); os.environ["PYTHONDONTWRITEBYTECODE"]="1"; os.environ.setdefault("PYTHONHASHSEED",str(SEED)); sys.dont_write_bytecode=True
        if not a.model_id or not a.config_id or not a.device: raise SystemExit("Worker mode requires --model-id, --config-id, --device")
        return worker_main(a.model_id,a.config_id,a.device)
    if a.verify_only: return verify_existing()
    return run_master()
if __name__=="__main__":
    try: sys.exit(main())
    except KeyboardInterrupt: print("\nStep 23 interrupted by user."); sys.exit(130)
    except Exception as e: print("\n"+"="*140); print("STEP 23 UNHANDLED FAILURE"); print("="*140); print(f"{type(e).__name__}: {e}"); traceback.print_exc(); print("="*140); sys.exit(1)
