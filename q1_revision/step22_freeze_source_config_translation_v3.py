#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 22 — CLEAN V3 SOURCE-CONFIG TRANSLATION + PRE-EXECUTION COMPATIBILITY FREEZE

Consolidates the useful logic of legacy Step 22 / 22B / 22C / 22D into one
performance-blind stage BEFORE any third-party model import/instantiation.

Run:
    python step22_freeze_source_config_translation_v3.py
Verify:
    python step22_freeze_source_config_translation_v3.py --verify-only
"""
from __future__ import annotations

import argparse, ast, copy, hashlib, json, os, re, shutil, subprocess, sys, traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent
if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError("Place this script inside <repository>/q1_revision/.")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import step00_config as cfg

STEP_NUMBER=22
STEP_ID="22"
STEP_NAME="STEP 22 — FREEZE STAGE-1 CANDIDATE -> FROZEN-SOURCE CONFIGURATION TRANSLATION + COMPATIBILITY"
SCHEMA_VERSION="1.0"

MODELS=("DLINEAR","NLINEAR","PATCHTST","TIMESNET","ITRANSFORMER","FREDFORMER","FRNET","MLF")
ALIASES={f"SOTA_{m}":m for m in MODELS}
EXPECTED_MODEL_COUNT=8
EXPECTED_CANDIDATE_COUNT=16
EXPECTED_REPO_COUNT=8
LOOKBACK=60
FEATURE_COUNT=63
FORECAST_LENGTH=5
DIRECTION_OUTPUTS=3
HORIZONS=(1,3,5)
FORBIDDEN_TEST_FILENAME="10_final_test_locked_panel.parquet"

REPO_BY_MODEL={
    "DLINEAR":"LTSF_LINEAR","NLINEAR":"LTSF_LINEAR","PATCHTST":"PATCHTST",
    "TIMESNET":"TIMESNET_TSLIB","ITRANSFORMER":"ITRANSFORMER",
    "FREDFORMER":"FREDFORMER","FRNET":"FRNET","MLF":"MLF",
}
RUNNERS={
    "LTSF_LINEAR":["run_longExp.py","run.py"],
    "PATCHTST":["PatchTST_supervised/run_longExp.py","PatchTST_supervised/run.py"],
    "TIMESNET_TSLIB":["run.py"],
    "ITRANSFORMER":["run.py","run_longExp.py"],
    "FREDFORMER":["run_longExp.py","run.py"],
    "FRNET":["run_longExp.py","run.py"],
    "MLF":["main_MLF_longterm.py","main_MLF_longterm_seqlen96.py","main_MLF_shortterm_public.py","main_MLF_shortterm_fund.py"],
}

KEY_MAP={
"DLINEAR":{
 "moving_average_kernel":("source_assertion","official_hardcoded_kernel_size"),
 "individual":("source_config","individual"),
 "head_hidden":("adapter_inert","Step21_single_Linear_315x3_has_no_hidden_layer"),
 "dropout":("adapter_inert","Step21_postforecast_adapter_dropout_fixed_to_zero"),
 "learning_rate":("training","learning_rate")},
"NLINEAR":{
 "individual":("source_config","individual"),
 "head_hidden":("adapter_inert","Step21_single_Linear_315x3_has_no_hidden_layer"),
 "dropout":("adapter_inert","Step21_postforecast_adapter_dropout_fixed_to_zero"),
 "learning_rate":("training","learning_rate")},
"PATCHTST":{
 "patch_length":("constructor_kwarg","patch_len"),"stride":("constructor_kwarg","stride"),
 "d_model":("constructor_kwarg","d_model"),"n_heads":("constructor_kwarg","n_heads"),
 "encoder_layers":("constructor_kwarg","n_layers"),"d_ff":("constructor_kwarg","d_ff"),
 "dropout":("constructor_kwarg","dropout"),"learning_rate":("training","learning_rate")},
"TIMESNET":{
 "top_k_periods":("source_config","top_k"),"num_kernels":("source_config","num_kernels"),
 "d_model":("source_config","d_model"),"d_ff":("source_config","d_ff"),
 "encoder_layers":("source_config","e_layers"),"dropout":("source_config","dropout"),
 "learning_rate":("training","learning_rate")},
"ITRANSFORMER":{
 "d_model":("source_config","d_model"),"n_heads":("source_config","n_heads"),
 "encoder_layers":("source_config","e_layers"),"d_ff":("source_config","d_ff"),
 "dropout":("source_config","dropout"),"learning_rate":("training","learning_rate")},
"FREDFORMER":{
 "capacity":("metadata","capacity_label"),"d_model":("source_config","d_model"),
 "n_heads":("source_config","n_heads"),"encoder_layers":("source_config","e_layers"),
 "dropout":("source_config","dropout"),"learning_rate":("training","learning_rate"),
 "official_source_mapping_required":("metadata","official_source_mapping_required")},
"FRNET":{
 "capacity":("metadata","capacity_label"),"hidden_dim":("source_config","emb"),
 "dropout":("source_config","dropout"),"learning_rate":("training","learning_rate"),
 "official_source_mapping_required":("metadata","official_source_mapping_required")},
"MLF":{
 "period_lengths":("source_config","scal_all"),"d_model":("source_config","d_model"),
 "n_heads":("source_config","n_heads"),"encoder_layers":("source_config","e_layers"),
 "dropout":("source_config","dropout"),"learning_rate":("training","learning_rate"),
 "official_source_mapping_required":("metadata","official_source_mapping_required")},
}

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
}
F16=cfg.MANIFEST_DIR/"16_stage1_model_registry_freeze.json"
STEP16_HYPERPARAMETER_JSON=cfg.MANIFEST_DIR/"16_stage1_hyperparameter_candidates.json"
F17=cfg.MANIFEST_DIR/"17_official_model_source_freeze.json"
F18=cfg.MANIFEST_DIR/"18_source_api_freeze.json"
F19=cfg.MANIFEST_DIR/"19_adapter_locus_freeze.json"
F20=cfg.MANIFEST_DIR/"20_transitive_backbone_freeze.json"
F21=cfg.MANIFEST_DIR/"21_adapter_contract_freeze.json"
TEST_SEAL=cfg.MANIFEST_DIR/"10_final_test_seal.json"

OUT_CONFIG=cfg.AUDIT_DIR/"22_candidate_source_configurations.csv"
OUT_COVER=cfg.AUDIT_DIR/"22_source_config_coverage.csv"
OUT_KEYS=cfg.AUDIT_DIR/"22_candidate_key_translation_audit.csv"
OUT_PARSER=cfg.AUDIT_DIR/"22_repository_parser_defaults.csv"
OUT_REFS=cfg.AUDIT_DIR/"22_reachable_config_reference_audit.csv"
OUT_COMPAT=cfg.AUDIT_DIR/"22_source_compatibility_amendments.csv"
OUT_EVIDENCE=cfg.AUDIT_DIR/"22_source_compatibility_evidence.csv"
OUT_REPO=cfg.AUDIT_DIR/"22_repository_recheck.csv"
OUT_SOURCE_HASH=cfg.AUDIT_DIR/"22_source_file_hashes.csv"
FREEZE=cfg.MANIFEST_DIR/"22_source_config_translation_freeze.json"
MANIFEST=cfg.MANIFEST_DIR/"22_source_config_translation_manifest.json"
MANIFEST_SHA=cfg.MANIFEST_DIR/"22_source_config_translation_manifest.sha256"
OUT_HASHES=cfg.AUDIT_DIR/"22_source_config_translation_hashes.csv"
FROZEN_SCRIPT=cfg.MANIFEST_DIR/"step22_freeze_source_config_translation_frozen.py"
SUMMARY=cfg.REPORT_DIR/"22_source_config_translation_summary.txt"
FAILED_DIR=cfg.AUDIT_DIR/"step22_failed_attempts"

def now(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); t=path.with_name(path.name+".tmp")
    t.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8"); os.replace(t,path)
def atomic_text(path,text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); t=path.with_name(path.name+".tmp")
    t.write_text(text,encoding="utf-8"); os.replace(t,path)
def write_csv(path,df):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); t=path.with_name(path.name+".tmp")
    df.to_csv(t,index=False,float_format="%.17g"); os.replace(t,path)
def rel(path): return str(Path(path).resolve().relative_to(cfg.EXPERIMENT_ROOT_DIR.resolve()))
def art(path):
    path=Path(path); return {"path":str(path),"relative_path":rel(path),"size_bytes":path.stat().st_size,"sha256":sha(path)}
def safe(v):
    if isinstance(v,(str,int,float,bool)) or v is None:return v
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    return str(v)
def boolish(v,default=False):
    if v is None:return default
    if isinstance(v,bool):return v
    if isinstance(v,int):return bool(v)
    if isinstance(v,str):
        s=v.strip().lower()
        if s in {"1","true","yes","y"}:return True
        if s in {"0","false","no","n"}:return False
    return default
def gitexe():
    x=shutil.which("git")
    if not x:raise RuntimeError("git not found.")
    return x
def git(args,cwd):
    env=os.environ.copy(); env["GIT_TERMINAL_PROMPT"]="0"
    p=subprocess.run([gitexe(),*args],cwd=str(cwd),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=120)
    if p.returncode:raise RuntimeError(f"git {' '.join(args)} failed in {cwd}\n{p.stderr}")
    return p.stdout.strip()
def norm_url(s):
    s=s.strip().rstrip("/")
    if s.endswith(".git"):s=s[:-4]
    return s.lower()

def verify_sidecar(m,s,label):
    if not Path(m).exists() or not Path(s).exists():raise RuntimeError(f"{label} manifest/sidecar missing.")
    exp=Path(s).read_text(encoding="utf-8").strip().split()[0]; obs=sha(m)
    if exp!=obs:raise RuntimeError(f"{label} manifest SHA mismatch.")
    return obs
def find_artifact(manifest,name):
    hits=[]
    for k in ("artifacts","canonical_artifacts"):
        for r in manifest.get(k,{}).values():
            if isinstance(r,dict) and r.get("path") and Path(r["path"]).name==name:hits.append(r)
    if len(hits)!=1:raise RuntimeError(f"Expected one artifact {name}; found {len(hits)}.")
    return hits[0]
def verify_fp(path,key):
    d=json.loads(Path(path).read_text(encoding="utf-8")); stored=d.get(key)
    if not stored:raise RuntimeError(f"{Path(path).name} lacks {key}.")
    x=dict(d); x.pop(key,None)
    if cfg.canonical_json_sha256(x)!=stored:raise RuntimeError(f"{Path(path).name}: embedded fingerprint mismatch.")
    return d,stored

def verify_step00():
    cfg.validate_configuration(); payload=cfg.build_configuration_payload(); fp=cfg.canonical_json_sha256(payload)
    _,msha,status=cfg.verify_existing_freeze(configuration_payload=payload,configuration_fingerprint=fp)
    if status!="VERIFIED_EXISTING":raise RuntimeError("Step-00 not VERIFIED_EXISTING.")
    lockp=cfg.get_final_test_lock_path(); lock=json.loads(lockp.read_text(encoding="utf-8"))
    if lock.get("state")!="LOCKED_NOT_OPENED" or lock.get("final_test_opened") is not False or lock.get("final_test_scored") is not False:
        raise RuntimeError("Global final-test lock not pristine.")
    return {"fingerprint":fp,"step00_manifest_sha256":msha,"lock_path":str(lockp),"lock_sha256":sha(lockp),"state":lock["state"]}

def verify_environment():
    d=json.loads(ENV_AUDIT.read_text(encoding="utf-8")); exp=d.get("system",{}).get("python_executable")
    if exp and Path(exp).resolve()!=Path(sys.executable).resolve():raise RuntimeError(f"Wrong Python.\nFrozen={exp}\nCurrent={sys.executable}")
    return exp

def verify_upstream(s0):
    shas={}; mans={}
    for i in range(1,22):
        m,s=UP[i]; shas[i]=verify_sidecar(m,s,f"Step {i:02d}"); mans[i]=json.loads(m.read_text(encoding="utf-8"))
    if mans[1].get("status")!="PASS" or mans[1].get("configuration_fingerprint_sha256")!=s0["fingerprint"]:raise RuntimeError("Step-01 mismatch.")
    if mans[2].get("step",{}).get("status")!="PASS":raise RuntimeError("Step-02 not PASS.")
    for i in range(3,22):
        m=mans[i]
        if m.get("step",{}).get("status")!="PASS" or m.get("configuration_fingerprint_sha256")!=s0["fingerprint"]:
            raise RuntimeError(f"Step-{i:02d} status/config mismatch.")
        for j in range(1,i):
            key=f"step{j:02d}_manifest_sha256"
            if key in m and m[key]!=shas[j]:raise RuntimeError(f"Step-{i:02d} chain mismatch: {key}")
    verify_environment()
    f16,p16=verify_fp(F16,"registry_freeze_fingerprint_sha256")
    f17,p17=verify_fp(F17,"source_freeze_fingerprint_sha256")
    f18,p18=verify_fp(F18,"source_api_fingerprint_sha256")
    f19,p19=verify_fp(F19,"adapter_locus_fingerprint_sha256")
    f20,p20=verify_fp(F20,"transitive_backbone_fingerprint_sha256")
    f21,p21=verify_fp(F21,"adapter_contract_fingerprint_sha256")
    exp_status={
      16:"STAGE1_MODEL_REGISTRY_AND_SELECTION_PROTOCOL_FROZEN",
      17:"OFFICIAL_AUTHOR_MAINTAINED_MODEL_SOURCE_SNAPSHOTS_FROZEN",
      18:"FROZEN_SOURCE_APIS_AUDITED_BEFORE_ADAPTER_IMPLEMENTATION",
      19:"ADAPTER_LOCI_AUDITED_BEFORE_IMPLEMENTATION",
      20:"TRANSITIVE_BACKBONE_STRUCTURE_AUDITED_BEFORE_ADAPTER_IMPLEMENTATION",
      21:"STAGE1_DIRECTIONAL_ADAPTER_CONTRACTS_FROZEN",
    }
    freezes={16:f16,17:f17,18:f18,19:f19,20:f20,21:f21}
    fps={16:p16,17:p17,18:p18,19:p19,20:p20,21:p21}
    fields={16:"registry_freeze_fingerprint_sha256",17:"source_freeze_fingerprint_sha256",18:"source_api_fingerprint_sha256",19:"adapter_locus_fingerprint_sha256",20:"transitive_backbone_fingerprint_sha256",21:"adapter_contract_fingerprint_sha256"}
    paths={16:F16,17:F17,18:F18,19:F19,20:F20,21:F21}
    for i in range(16,22):
        if freezes[i].get("status")!=exp_status[i]:raise RuntimeError(f"Step-{i} freeze status mismatch.")
        if mans[i].get(fields[i])!=fps[i]:raise RuntimeError(f"Step-{i} manifest/freeze fingerprint mismatch.")
        r=find_artifact(mans[i],paths[i].name)
        if Path(r["path"]).resolve()!=paths[i].resolve() or r["sha256"]!=sha(paths[i]):raise RuntimeError(f"Step-{i} freeze artifact drift.")
    if f16.get("training_started") is not False:raise RuntimeError("Step-16 says training started.")
    st=f21.get("pipeline_state",{})
    if st.get("adapter_contracts_frozen") is not True:raise RuntimeError("Step-21 contracts not frozen.")
    for k in ("task_adapters_implemented","models_instantiated","smoke_tests_run","models_fitted","feature_selection_applied","probability_threshold_selection_performed"):
        if k in st and st[k] is not False:raise RuntimeError(f"Step-21 pre-execution state violation: {k}")
    contracts=f21.get("contracts",[])
    if len(contracts)!=8 or set(r["model_id"] for r in contracts)!=set(MODELS):raise RuntimeError("Step-21 contract model set mismatch.")
    rec20=f20.get("model_records",[])
    if len(rec20)!=8 or set(r["model_id"] for r in rec20)!=set(MODELS):raise RuntimeError("Step-20 model set mismatch.")
    repos=f17.get("repositories",[])
    if len(repos)!=8 or sum(r.get("repository_role")=="implementation" for r in repos)!=7 or sum(r.get("repository_role")=="reference_redirect" for r in repos)!=1:
        raise RuntimeError("Step-17 repository inventory drift.")
    seal=json.loads(TEST_SEAL.read_text(encoding="utf-8"))
    if seal.get("status")!="SEALED_NOT_OPENED" or seal.get("global_final_test_lock_state")!="LOCKED_NOT_OPENED":raise RuntimeError("Step-10 seal invalid.")
    return {"manifest_sha":shas,"manifests":mans,"freeze16":f16,"freeze17":f17,"freeze20":f20,"freeze21":f21,
            "p16":p16,"p17":p17,"p18":p18,"p19":p19,"p20":p20,"p21":p21,
            "step20_by_model":{r["model_id"]:r for r in rec20},"step21_by_model":{r["model_id"]:r for r in contracts},
            "test_seal_status":seal["status"],"sealed_test_sha256_from_metadata_only":seal.get("sha256")}

def recheck_repos(f17):
    rows=[]; mp={}
    for r in f17["repositories"]:
        rid=r["repo_id"]; root=Path(r["local_path"])
        if not (root/".git").exists():raise RuntimeError(f"{rid}: missing Git checkout.")
        origin=git(["remote","get-url","origin"],root); commit=git(["rev-parse","HEAD"],root); tree=git(["rev-parse","HEAD^{tree}"],root)
        branch=git(["branch","--show-current"],root); status=git(["status","--porcelain","--untracked-files=all"],root)
        ok=(norm_url(origin)==norm_url(r["repository_url"]) and commit==r["commit_sha"] and tree==r["git_tree_sha"] and branch==r["branch"] and status=="")
        checked=0
        for fr in r["required_file_hashes"]:
            p=root/fr["relative_path"]
            if not p.is_file() or sha(p)!=fr["sha256"]:ok=False
            else:checked+=1
        if not ok:raise RuntimeError(f"{rid}: frozen repository recheck failed.")
        mp[rid]=r; rows.append({"repo_id":rid,"repository_role":r["repository_role"],"local_path":str(root),
            "origin_match":True,"commit_match":True,"tree_match":True,"branch_match":True,
            "worktree_clean_including_untracked":True,"required_files_checked":checked,"required_file_hashes_match":True,"passed":True})
    if len(rows)!=8:raise RuntimeError("Repository count drift.")
    return mp,pd.DataFrame(rows)

def outputs():
    return (OUT_CONFIG,OUT_COVER,OUT_KEYS,OUT_PARSER,OUT_REFS,OUT_COMPAT,OUT_EVIDENCE,OUT_REPO,OUT_SOURCE_HASH,FREEZE,MANIFEST,MANIFEST_SHA,OUT_HASHES,FROZEN_SCRIPT,SUMMARY)
def assert_fresh():
    hit=[p for p in outputs() if p.exists()]
    if hit:raise RuntimeError("Step-22 outputs already exist; use --verify-only.\n"+"\n".join(map(str,hit)))
def archive_partial():
    if MANIFEST.exists():return None
    hit=[p for p in outputs() if p.exists()]
    if not hit:return None
    FAILED_DIR.mkdir(parents=True,exist_ok=True); d=FAILED_DIR/f"partial_artifacts_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"; d.mkdir()
    for p in hit:
        try:shutil.move(str(p),str(d/p.name))
        except Exception:pass
    return d

def normalize_id(x): return ALIASES.get(str(x),str(x))

def load_step16_hyperparameter_registry(step16_manifest):
    """
    Clean-V3 Step 16 stores the A/B candidate payload in the dedicated,
    immutable artifact 16_stage1_hyperparameter_candidates.json rather than
    duplicating it inside the Step-16 freeze JSON.
    """
    record=find_artifact(step16_manifest,STEP16_HYPERPARAMETER_JSON.name)
    artifact_path=Path(record["path"]).resolve()

    if artifact_path!=STEP16_HYPERPARAMETER_JSON.resolve():
        raise RuntimeError(
            "Step-16 hyperparameter artifact path drift.\n"
            f"Manifest: {artifact_path}\n"
            f"Expected: {STEP16_HYPERPARAMETER_JSON.resolve()}"
        )

    if not artifact_path.is_file():
        raise RuntimeError(
            f"Step-16 hyperparameter artifact missing: {artifact_path}"
        )

    observed_sha=sha(artifact_path)
    if observed_sha!=record["sha256"]:
        raise RuntimeError(
            "Step-16 hyperparameter artifact SHA mismatch.\n"
            f"Manifest: {record['sha256']}\n"
            f"Observed: {observed_sha}"
        )

    payload=json.loads(artifact_path.read_text(encoding="utf-8"))

    if payload.get("stage")!="stage1_architecture_screening":
        raise RuntimeError(
            "Unexpected Step-16 hyperparameter artifact stage."
        )

    raw=payload.get("model_candidates")
    if not isinstance(raw,dict):
        raise RuntimeError(
            "Step-16 hyperparameter artifact does not contain model_candidates."
        )

    return raw,observed_sha,artifact_path

def extract_candidates(step16_manifest):
    raw,artifact_sha,artifact_path=load_step16_hyperparameter_registry(
        step16_manifest
    )

    out={}; src={}
    for k,v in raw.items():
        m=normalize_id(k)
        if m in MODELS:
            if m in out:
                raise RuntimeError(
                    f"Duplicate normalized candidate family {m}."
                )
            out[m]=copy.deepcopy(v)
            src[m]=str(k)

    if set(out)!=set(MODELS):
        raise RuntimeError(
            "Step-16 Stage-1 candidate set mismatch.\n"
            f"Expected: {sorted(MODELS)}\n"
            f"Observed: {sorted(out)}"
        )

    total=0
    for m in MODELS:
        if len(out[m])!=2:
            raise RuntimeError(
                f"{m}: expected two candidates; found {len(out[m])}."
            )

        ids=[str(x.get("config_id","")) for x in out[m]]
        if any(not x for x in ids) or len(set(ids))!=2:
            raise RuntimeError(
                f"{m}: invalid or duplicate config IDs."
            )

        total+=len(out[m])

    if total!=EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATE_COUNT} Stage-1 candidates; "
            f"found {total}."
        )

    return out,src,artifact_sha,artifact_path

def locate_runner(root,rid):
    for r in RUNNERS[rid]:
        p=root/r
        if p.is_file():return p
    raise RuntimeError(f"No runner found for {rid}; tried {RUNNERS[rid]}")
def lit(node):
    try:return ast.literal_eval(node),True
    except Exception:return None,False
def chain(node):
    if isinstance(node,ast.Name):return [node.id]
    if isinstance(node,ast.Attribute):return chain(node.value)+[node.attr]
    return []
def callname(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):
        x=callname(node.value); return f"{x}.{node.attr}" if x else node.attr
    return ""
def arg_dest(call):
    for kw in call.keywords:
        if kw.arg=="dest":
            v,ok=lit(kw.value)
            if ok and isinstance(v,str):return v
    flags=[]
    for a in call.args:
        v,ok=lit(a)
        if ok and isinstance(v,str) and v.startswith("-"):flags.append(v)
    if not flags:return None
    long=[x for x in flags if x.startswith("--")]; s=max(long,key=len) if long else flags[0]
    return s.lstrip("-").replace("-","_")
def parser_defaults(path,rid):
    tree=ast.parse(path.read_text(encoding="utf-8",errors="replace")); defaults={}; rows=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!="add_argument":continue
        dest=arg_dest(node)
        if not dest:continue
        default=None; action=None; nargs=None; required=False; typ=""
        for kw in node.keywords:
            if kw.arg=="default":default=kw.value
            elif kw.arg=="action":
                v,ok=lit(kw.value); action=v if ok else None
            elif kw.arg=="nargs":
                v,ok=lit(kw.value); nargs=v if ok else None
            elif kw.arg=="required":
                v,ok=lit(kw.value); required=bool(v) if ok else False
            elif kw.arg=="type":typ=callname(kw.value)
        if default is not None:v,ok=lit(default)
        elif action=="store_true":v,ok=False,True
        elif action=="store_false":v,ok=True,True
        else:v,ok=None,False
        rows.append({"repo_id":rid,"runner_file":str(path),"line":int(getattr(node,"lineno",-1)),"destination":dest,
                     "default_resolved":ok,"default_value_json":json.dumps(safe(v),ensure_ascii=False) if ok else "",
                     "action":action,"nargs_json":json.dumps(safe(nargs),ensure_ascii=False) if nargs is not None else "",
                     "type_name":typ,"required":required})
        if ok:defaults[dest]=safe(v)
    return defaults,rows

class RefVisitor(ast.NodeVisitor):
    def __init__(self):self.reads=set(); self.writes=set()
    def visit_Attribute(self,node):
        c=chain(node); a=None
        if len(c)==2 and c[0] in {"configs","config","args"}:a=c[1]
        elif len(c)==3 and c[0]=="self" and c[1] in {"configs","config","args"}:a=c[2]
        if a:
            (self.writes if isinstance(node.ctx,ast.Store) else self.reads).add(a)
        self.generic_visit(node)
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=="getattr" and len(node.args)>=2:
            c=chain(node.args[0]); a,ok=lit(node.args[1])
            if c in [["configs"],["config"],["args"],["self","configs"],["self","config"],["self","args"]] and ok and isinstance(a,str):self.reads.add(a)
        self.generic_visit(node)
def reachable_classes(record):
    root=(record["primary_source_file"],record["preferred_class"]); adj={}
    for r in record["self_module_assignments"]:
        cf=str(r.get("resolved_target_file","")); cc=str(r.get("resolved_target_class",""))
        if cf and cc:adj.setdefault((r["source_file"],r["class_name"]),set()).add((cf,cc))
    q=deque([root]); seen=set()
    while q:
        x=q.popleft()
        if x in seen:continue
        seen.add(x); q.extend(y for y in adj.get(x,set()) if y not in seen)
    return seen
def inspect_refs(record,root):
    reads=set(); writes=set(); rows=[]; cache={}
    for rf,cn in sorted(reachable_classes(record)):
        p=root/rf
        if p not in cache:cache[p]=ast.parse(p.read_text(encoding="utf-8",errors="replace"))
        hits=[n for n in cache[p].body if isinstance(n,ast.ClassDef) and n.name==cn]
        if len(hits)!=1:raise RuntimeError(f"Reachable class not uniquely found: {rf}::{cn}")
        v=RefVisitor(); v.visit(hits[0]); reads|=v.reads; writes|=v.writes
        for a in sorted(v.reads|v.writes):rows.append({"model_id":record["model_id"],"source_file":rf,"class_name":cn,"config_attribute":a,"read":a in v.reads,"written_at_runtime":a in v.writes})
    return reads-writes,rows

def args_assignments(path,attr):
    tree=ast.parse(path.read_text(encoding="utf-8",errors="replace")); vals=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign):targets=n.targets; value=n.value
        elif isinstance(n,ast.AnnAssign):targets=[n.target]; value=n.value
        else:continue
        if value is None:continue
        v,ok=lit(value)
        if not ok:continue
        for t in targets:
            if chain(t)==["args",attr]:vals.append(safe(v))
    return vals

def frnet_evidence(root,runner):
    _,rows=parser_defaults(runner,"FRNET"); hits=[r for r in rows if r["destination"]=="period_list"]
    if len(hits)!=1:raise RuntimeError("Expected one FRNet period_list parser declaration.")
    rec=hits[0]
    if json.loads(rec["default_value_json"])!=1 or rec["nargs_json"]!=json.dumps("+") or rec["type_name"] not in {"int","builtins.int"}:
        raise RuntimeError(f"Unexpected FRNet period_list parser contract: {rec}")
    model=root/"models"/"FRNet.py"
    if not model.is_file():raise RuntimeError(f"Missing FRNet source: {model}")
    txt=model.read_text(encoding="utf-8",errors="replace")
    snippets={"period_index_zero":"self.period_list[0]","period_len":"len(self.period_list)","period_index_k":"self.period_list[k]",
              "global_switch":"if self.global_freq_pred:","rfft":"torch.fft.rfft","period_loop":"for k in range(len(self.period_list))"}
    present={k:v in txt for k,v in snippets.items()}
    if not all(present.values()):raise RuntimeError(f"FRNet source evidence missing: {present}")
    scripts=root/"scripts"/"FRNet"; pattern=re.compile(r"--global_freq_pred\s+([01])"); vals=[]
    for p in scripts.rglob("*.sh"):
        for m in pattern.finditer(p.read_text(encoding="utf-8",errors="replace")):vals.append(int(m.group(1)))
    if set(vals)!={0,1}:raise RuntimeError(f"FRNet scripts do not show both global_freq_pred states: {sorted(set(vals))}")
    return {"parser":rec,"source_presence":present,"switches":vals,"model_path":model}

def mlf_evidence(root):
    backbone=root/"layers"/"MLF_backbone.py"; runners=[root/"main_MLF_longterm.py",root/"main_MLF_longterm_seqlen96.py"]
    if not backbone.is_file() or any(not p.is_file() for p in runners):raise RuntimeError("Required MLF backbone/long-horizon runner missing.")
    if "activation_tag" not in backbone.read_text(encoding="utf-8",errors="replace"):raise RuntimeError("MLF backbone no longer reads activation_tag.")
    av=[]; fp=[]; ma=[]
    for p in runners:
        av+=args_assignments(p,"activation_tag"); fp+=args_assignments(p,"fixed_patch_num"); ma+=args_assignments(p,"MAP_alpha")
    false=sum(v is False for v in av); true=sum(v is True for v in av)
    if false<=0:raise RuntimeError("No activation_tag=False assignment in frozen MLF long-horizon runners.")
    fps=sorted({int(v) for v in fp if isinstance(v,int) and not isinstance(v,bool)})
    mas=sorted({int(v) for v in ma if isinstance(v,int) and not isinstance(v,bool)})
    return {"backbone":backbone,"runners":runners,"false":false,"true":true,
            "official_fixed_patch_num":fps[0] if len(fps)==1 else 64,"official_map_alpha":mas[0] if len(mas)==1 else 2,
            "fixed_values":fps,"alpha_values":mas}

def base_config(model,defaults):
    s=dict(defaults)
    s.update({"seq_len":60,"context_window":60,"pred_len":5,"enc_in":63,"dec_in":63,"c_out":63,"label_len":0,"num_class":3,
              "output_attention":False,"activation":"gelu","factor":int(defaults.get("factor",1)),
              "embed":defaults.get("embed","timeF"),"freq":defaults.get("freq","d")})
    if model in {"DLINEAR","NLINEAR"}:s["individual"]=boolish(s.get("individual"),False)
    elif model=="TIMESNET":s.update({"task_name":"classification","pred_len":0,"num_class":3,"d_layers":int(s.get("d_layers",1))})
    elif model=="ITRANSFORMER":s.update({"use_norm":True,"class_strategy":s.get("class_strategy","projection"),"output_attention":False})
    elif model=="FREDFORMER":
        s.update({"individual":False,"fc_dropout":float(s.get("fc_dropout",.05)),"head_dropout":float(s.get("head_dropout",0)),
                  "patch_len":int(s.get("patch_len",16)),"stride":int(s.get("stride",8)),"padding_patch":s.get("padding_patch","end"),
                  "revin":boolish(s.get("revin"),True),"affine":boolish(s.get("affine"),False),"subtract_last":boolish(s.get("subtract_last"),False),
                  "use_nys":int(s.get("use_nys",0)),"ablation":int(s.get("ablation",0)),"cf_dim":int(s.get("cf_dim",48)),
                  "cf_depth":int(s.get("cf_depth",2)),"cf_heads":int(s.get("cf_heads",6)),"cf_mlp":int(s.get("cf_mlp",128)),
                  "cf_head_dim":int(s.get("cf_head_dim",32)),"cf_drop":float(s.get("cf_drop",.2)),
                  "mlp_hidden":int(s.get("mlp_hidden",128)),"mlp_drop":float(s.get("mlp_drop",.3))})
    elif model=="FRNET":
        s.update({"kernel_size":int(s.get("kernel_size",25)),"revin":boolish(s.get("revin"),True),
                  "affine":boolish(s.get("affine"),False),"subtract_last":boolish(s.get("subtract_last"),False)})
    elif model=="MLF":
        s.update({"device":"__RUNTIME_DEVICE__","fc_dropout":float(s.get("fc_dropout",.05)),"head_dropout":float(s.get("head_dropout",0)),
                  "individual":boolish(s.get("individual"),False),"patch_len":int(s.get("patch_len",16)),"stride":int(s.get("stride",8)),
                  "padding_patch":s.get("padding_patch","end"),"revin":boolish(s.get("revin"),True),"affine":boolish(s.get("affine"),False),
                  "subtract_last":boolish(s.get("subtract_last"),False),"D_norm":boolish(s.get("D_norm"),False),
                  "revin_norm":boolish(s.get("revin_norm"),False),"bs":2,"seq_last":None})
    return s
def patchtst_base(d):
    return {"c_in":63,"context_window":60,"target_window":5,"patch_len":int(d.get("patch_len",16)),"stride":int(d.get("stride",8)),
            "n_layers":int(d.get("e_layers",3)),"d_model":int(d.get("d_model",128)),"n_heads":int(d.get("n_heads",8)),
            "d_ff":int(d.get("d_ff",256)),"norm":"BatchNorm","attn_dropout":0.0,"dropout":float(d.get("dropout",0)),
            "act":"gelu","res_attention":True,"pre_norm":False,"store_attn":False,"pe":"zeros","learn_pe":True,
            "fc_dropout":float(d.get("fc_dropout",.05)),"head_dropout":float(d.get("head_dropout",0)),
            "padding_patch":d.get("padding_patch","end"),"pretrain_head":False,"head_type":"flatten","individual":boolish(d.get("individual"),False),
            "revin":boolish(d.get("revin"),True),"affine":boolish(d.get("affine"),False),"subtract_last":boolish(d.get("subtract_last"),False)}

def finalize_frnet(s,config_id,e):
    s=dict(s); rows=[]
    orig=boolish(s.get("decomposition"),False); s["decomposition"]=False
    if orig is not False:rows.append({"model_id":"FRNET","config_id":config_id,"field":"decomposition","operation":"disable_short_history_incompatible_optional_decomposition_branch","original_value_json":json.dumps(orig),"effective_value_json":"false","numeric_value_changed":False,"candidate_search_dimension":False,"candidate_search_space_changed":False,"performance_information_used":False,"source_code_modified":False})
    p=s.get("period_list",1)
    if isinstance(p,int) and not isinstance(p,bool):
        periods=[int(p)]; rows.append({"model_id":"FRNET","config_id":config_id,"field":"period_list","operation":"scalar_to_one_element_list_type_normalization","original_value_json":json.dumps(p),"effective_value_json":json.dumps(periods),"numeric_value_changed":False,"candidate_search_dimension":False,"candidate_search_space_changed":False,"performance_information_used":False,"source_code_modified":False})
    elif isinstance(p,list):periods=[int(x) for x in p]
    else:raise RuntimeError(f"{config_id}: unsupported period_list {p!r}")
    if not periods or any(x<=0 or x>60 for x in periods):raise RuntimeError(f"{config_id}: invalid period_list {periods}")
    s["period_list"]=periods
    dom=int(60//periods[0]+1)*2+10; bins=31; orig_g=s.get("global_freq_pred")
    if orig_g is None:raise RuntimeError(f"{config_id}: global_freq_pred missing from frozen source defaults.")
    gb=boolish(orig_g,False)
    if dom>bins and gb:
        s["global_freq_pred"]=0
        rows.append({"model_id":"FRNET","config_id":config_id,"field":"global_freq_pred","operation":"disable_dimensionally_incompatible_optional_global_frequency_branch","original_value_json":json.dumps(safe(orig_g)),"effective_value_json":"0","numeric_value_changed":True,"candidate_search_dimension":False,"candidate_search_space_changed":False,"performance_information_used":False,"source_code_modified":False})
    else:s["global_freq_pred"]=int(gb)
    s["_clean_v3_frnet_dimension_audit"]={"seq_len":60,"period_list":periods,"dominance_freq":dom,"available_rfft_bins":bins,"global_branch_dimensionally_incompatible":dom>bins,
      "official_scripts_contain_global_freq_0":0 in e["switches"],"official_scripts_contain_global_freq_1":1 in e["switches"],
      "period_parser_default":json.loads(e["parser"]["default_value_json"]),"period_parser_nargs":json.loads(e["parser"]["nargs_json"]),
      "period_parser_type":e["parser"]["type_name"],"model_uses_period_list_as_sequence":all(e["source_presence"].values())}
    return s,rows

def finalize_mlf(s,config_id,e):
    s=dict(s); rows=[]
    if "scal_all" not in s:raise RuntimeError(f"{config_id}: MLF scal_all missing.")
    scales=[int(x) for x in s["scal_all"]]
    if not scales or any(x<=0 or x>60 for x in scales):raise RuntimeError(f"{config_id}: invalid MLF scales {scales}")
    official=int(e["official_fixed_patch_num"]); alpha=int(e["official_map_alpha"]); effective=min(official,min(scales))
    if effective!=official:rows.append({"model_id":"MLF","config_id":config_id,"field":"fixed_patch_num","operation":"cap_to_minimum_frozen_scale_to_prevent_zero_stride","original_value_json":json.dumps(official),"effective_value_json":json.dumps(effective),"numeric_value_changed":True,"candidate_search_dimension":False,"candidate_search_space_changed":False,"performance_information_used":False,"source_code_modified":False})
    ps=[]; eql=[]
    for scale in scales:
        stride=int(scale/effective)
        if stride<=0:raise RuntimeError(f"{config_id}: zero MLF stride.")
        plen=stride*alpha
        if plen<=0 or plen>scale:raise RuntimeError(f"{config_id}: invalid patch length {plen} at scale {scale}")
        ps.append([plen,stride]); eql.append(plen)
    s.update({"scal_all":scales,"MAP":True,"MAP_alpha":alpha,"fixed_patch_num":effective,"official_fixed_patch_num":official,
              "patchLen_stride_all":ps,"equal_patch_len":eql,"max_patch_len":max(eql),"patch_pad":True,"patch_squeeze":True,
              "squeeze_factor":[4 for _ in scales],"threshold_patch_num":7,"redundancy_scaling":True,"activation_tag":False})
    rows.append({"model_id":"MLF","config_id":config_id,"field":"activation_tag","operation":"complete_required_runtime_boolean_with_conservative_frozen_source_supported_false","original_value_json":json.dumps("MISSING_OR_NONAUTHORITATIVE_PARSER_STATE"),"effective_value_json":"false","numeric_value_changed":False,"candidate_search_dimension":False,"candidate_search_space_changed":False,"performance_information_used":False,"source_code_modified":False})
    s["_clean_v3_mlf_compatibility_audit"]={"scales_from_step16_candidate":scales,"official_fixed_patch_num":official,"effective_fixed_patch_num":effective,"map_alpha":alpha,
      "activation_tag":False,"activation_false_assignments_in_frozen_long_horizon_runners":e["false"],"activation_true_assignments_in_frozen_long_horizon_runners":e["true"],
      "exact_external_dataset_reproduction_claimed":False}
    return s,rows

def translate(model,cand,defaults,fr_e=None,mlf_e=None):
    config_id=str(cand["config_id"]); source=patchtst_base(defaults) if model=="PATCHTST" else base_config(model,defaults); mode="direct_constructor_kwargs" if model=="PATCHTST" else "config_namespace"
    train={}; keyrows=[]
    ckeys={k for k in cand if k!="config_id"}; unmapped=ckeys-set(KEY_MAP[model])
    if unmapped:raise RuntimeError(f"{model}/{config_id}: unmapped Step-16 candidate keys {sorted(unmapped)}")
    for k in sorted(ckeys):
        typ,dest=KEY_MAP[model][k]; val=cand[k]
        if typ in {"source_config","constructor_kwarg"}:source[dest]=safe(val)
        elif typ=="training":train[dest]=safe(val)
        elif typ=="source_assertion":
            if model=="DLINEAR" and k=="moving_average_kernel" and int(val)!=25:raise RuntimeError("DLinear moving_average_kernel != frozen hardcoded 25.")
        elif typ not in {"adapter_inert","metadata"}:raise RuntimeError(f"Unsupported mapping type {typ}")
        keyrows.append({"model_id":model,"config_id":config_id,"candidate_key":k,"candidate_value_json":json.dumps(safe(val),ensure_ascii=False),
                        "destination_type":typ,"destination":dest,"silently_ignored":False,"performance_information_used":False})
    compat=[]
    if model=="FRNET":source,compat=finalize_frnet(source,config_id,fr_e)
    if model=="MLF":source,compat=finalize_mlf(source,config_id,mlf_e)
    if model=="TIMESNET" and (source["task_name"]!="classification" or int(source["pred_len"])!=0 or int(source["num_class"])!=3):raise RuntimeError("TimesNet task contract drift.")
    if model=="PATCHTST" and (int(source["context_window"])!=60 or int(source["c_in"])!=63 or int(source["target_window"])!=5):raise RuntimeError("PatchTST dimensions drift.")
    if model not in {"TIMESNET","PATCHTST"} and int(source.get("pred_len",-1))!=5:raise RuntimeError(f"{model}: pred_len drift.")
    return {"configuration_mode":mode,"source_config":source,"training_config":train},keyrows,compat

def source_hashes(runners,extra):
    src=[("current_step00",Path(cfg.__file__).resolve()),("current_step22",SCRIPT_FILE),("step16_registry_freeze",F16),("step16_hyperparameter_candidates",STEP16_HYPERPARAMETER_JSON),("step17_source_freeze",F17),("step18_source_api_freeze",F18),("step19_adapter_locus_freeze",F19),("step20_transitive_backbone_freeze",F20),("step21_adapter_contract_freeze",F21),("step10_final_test_seal_metadata",TEST_SEAL)]
    for i,(m,s) in UP.items():src += [(f"step{i:02d}_manifest",m),(f"step{i:02d}_manifest_sha",s)]
    for rid,p in sorted(runners.items()):src.append((f"frozen_runner::{rid}",p))
    src += extra
    rows=[]
    for role,p in src:
        p=Path(p)
        if p.name==FORBIDDEN_TEST_FILENAME:raise RuntimeError("Final-test parquet entered source list.")
        if not p.exists():raise RuntimeError(f"Missing source: {p}")
        rows.append({"role":role,"file_name":p.name,"absolute_path":str(p),"size_bytes":p.stat().st_size,"sha256":sha(p)})
    return pd.DataFrame(rows)

def verify_existing():
    print("="*140); print("STEP 22 — VERIFY EXISTING SOURCE-CONFIG TRANSLATION FREEZE"); print("="*140)
    s0=verify_step00(); up=verify_upstream(s0); msha=verify_sidecar(MANIFEST,MANIFEST_SHA,"Step 22"); m=json.loads(MANIFEST.read_text(encoding="utf-8"))
    if m.get("step",{}).get("status")!="PASS" or m.get("configuration_fingerprint_sha256")!=s0["fingerprint"]:return 20
    for i in range(1,22):
        if m.get(f"step{i:02d}_manifest_sha256")!=up["manifest_sha"][i]:return 21
    hp_raw,hp_sha,hp_path=load_step16_hyperparameter_registry(up["manifests"][16])
    if m.get("step16_hyperparameter_candidates_sha256")!=hp_sha:return 22
    for field,val in (("step16_registry_freeze_fingerprint_sha256",up["p16"]),("step17_source_freeze_fingerprint_sha256",up["p17"]),("step18_source_api_fingerprint_sha256",up["p18"]),("step19_adapter_locus_fingerprint_sha256",up["p19"]),("step20_transitive_backbone_fingerprint_sha256",up["p20"]),("step21_adapter_contract_fingerprint_sha256",up["p21"])):
        if m.get(field)!=val:return 22
    for r in m.get("artifacts",{}).values():
        p=Path(r["path"])
        if not p.exists() or sha(p)!=r["sha256"]:return 23
    freeze,fp=verify_fp(FREEZE,"source_config_translation_fingerprint_sha256")
    if m.get("source_config_translation_fingerprint_sha256")!=fp:return 24
    repo_map,recheck=recheck_repos(up["freeze17"])
    if sha(Path(s0["lock_path"]))!=s0["lock_sha256"]:return 25
    print(f"[PASS] Configuration fingerprint : {s0['fingerprint']}")
    print(f"[PASS] Step-21 manifest          : {up['manifest_sha'][21]}")
    print(f"[PASS] Step-16 registry freeze   : {up['p16']}")
    print(f"[PASS] Step-16 hyperparameters  : {hp_sha}")
    print(f"[PASS] Step-17 source freeze     : {up['p17']}")
    print(f"[PASS] Step-18 source API freeze : {up['p18']}")
    print(f"[PASS] Step-19 locus freeze      : {up['p19']}")
    print(f"[PASS] Step-20 transitive freeze : {up['p20']}")
    print(f"[PASS] Step-21 adapter freeze    : {up['p21']}")
    print(f"[PASS] Frozen repositories       : {len(repo_map)}")
    print(f"[PASS] Candidate source configs  : {len(freeze['candidate_source_configurations'])}")
    print(f"[PASS] Compatibility records     : {len(freeze['compatibility_amendments'])}")
    print(f"[PASS] Step-22 source-config freeze: {fp}")
    print(f"[PASS] Step-22 manifest          : {msha}")
    print(f"[PASS] Step-10 test seal         : {up['test_seal_status']}")
    print(f"[PASS] Global final-test state   : {s0['state']}")
    print("="*140); print("STEP 22 VERIFICATION STATUS: PASS"); print("="*140)
    return 0

def run():
    stage="verify"
    try:
        print("="*140); print(STEP_NAME); print("="*140)
        print("[1/10] Verifying Step 00 and immutable Step 01 -> Step 21 chain...")
        s0=verify_step00(); up=verify_upstream(s0)
        print(f"        Configuration fingerprint: {s0['fingerprint']}")
        print(f"        Step-21 manifest          : {up['manifest_sha'][21]}")
        print(f"        Step-16 registry freeze   : {up['p16']}")
        print(f"        Step-17 source freeze     : {up['p17']}")
        print(f"        Step-18 source API freeze : {up['p18']}")
        print(f"        Step-19 locus freeze      : {up['p19']}")
        print(f"        Step-20 transitive freeze : {up['p20']}")
        print(f"        Step-21 adapter freeze    : {up['p21']}")
        print(f"        Step-10 test seal         : {up['test_seal_status']}")
        print(f"        Global final-test state   : {s0['state']}")
        print("        Third-party import/execute: PROHIBITED")
        print("        Validation/final-test use : PROHIBITED")

        stage="collision_guard"; print("[2/10] Checking immutable Step-22 output target..."); assert_fresh(); cfg.create_directories()
        stage="repository_recheck"; print("[3/10] Rechecking all eight frozen Step-17 repositories...")
        repo_map,recheck=recheck_repos(up["freeze17"])
        print(f"        Repositories rechecked      : {len(recheck)}")
        print(f"        Implementation repositories : {int((recheck.repository_role=='implementation').sum())}")
        print(f"        Reference repositories      : {int((recheck.repository_role=='reference_redirect').sum())}")
        print(f"        Worktrees clean incl untracked: {bool(recheck.worktree_clean_including_untracked.all())}")

        stage="candidate_extract"; print("[4/10] Loading the dedicated frozen Step-16 hyperparameter registry and extracting 16 A/B candidates...")
        cmap,source_keys,step16_hp_sha,step16_hp_path=extract_candidates(up["manifests"][16])
        print(f"        Hyperparameter registry: {step16_hp_path}")
        print(f"        Hyperparameter SHA-256 : {step16_hp_sha}")
        print(f"        Stage-1 model families : {len(cmap)}")
        print(f"        Candidate configs      : {sum(len(x) for x in cmap.values())}")

        stage="static_source_audit"; print("[5/10] Extracting parser defaults, reachable config refs, and compatibility evidence...")
        defaults_by_repo={}; parser_rows=[]; runner_paths={}
        for rid in RUNNERS:
            root=Path(repo_map[rid]["local_path"]); rp=locate_runner(root,rid); d,rows=parser_defaults(rp,rid)
            defaults_by_repo[rid]=d; parser_rows+=rows; runner_paths[rid]=rp
        required={}; ref_rows=[]
        for m in MODELS:
            rec=up["step20_by_model"][m]; rid=rec["repo_id"]
            if rid!=REPO_BY_MODEL[m]:raise RuntimeError(f"{m}: Step-20 repo mismatch.")
            req,rows=inspect_refs(rec,Path(repo_map[rid]["local_path"])); required[m]=req; ref_rows+=rows

        frroot=Path(repo_map["FRNET"]["local_path"]); fre=frnet_evidence(frroot,runner_paths["FRNET"])
        mlroot=Path(repo_map["MLF"]["local_path"]); mle=mlf_evidence(mlroot)
        evidence=[
          {"model_id":"FRNET","source_kind":"official_runner_parser","file_path":str(runner_paths["FRNET"]),"fact":"period_list parser declaration","value_json":json.dumps({"default":1,"nargs":"+","type":"int"}),"used_for_compatibility":True},
          {"model_id":"FRNET","source_kind":"official_model_source","file_path":str(fre["model_path"]),"fact":"period_list sequence/global-frequency structure","value_json":json.dumps(fre["source_presence"],sort_keys=True),"used_for_compatibility":True},
          {"model_id":"FRNET","source_kind":"official_repository_scripts","file_path":str(frroot/"scripts"/"FRNet"),"fact":"global_freq_pred switch states","value_json":json.dumps(sorted(set(fre["switches"]))),"used_for_compatibility":True},
          {"model_id":"MLF","source_kind":"official_model_source","file_path":str(mle["backbone"]),"fact":"activation_tag required","value_json":"true","used_for_compatibility":True},
          {"model_id":"MLF","source_kind":"official_long_horizon_runners","file_path":str(mlroot),"fact":"activation_tag assignments","value_json":json.dumps({"false":mle["false"],"true":mle["true"]},sort_keys=True),"used_for_compatibility":True},
          {"model_id":"MLF","source_kind":"official_long_horizon_runners","file_path":str(mlroot),"fact":"adaptive patching constants","value_json":json.dumps({"fixed_patch_num":mle["official_fixed_patch_num"],"MAP_alpha":mle["official_map_alpha"]},sort_keys=True),"used_for_compatibility":True},
        ]
        print(f"        Runner repositories audited : {len(runner_paths)}")
        print(f"        Reachable config refs rows   : {len(ref_rows)}")
        print("        FRNet compatibility evidence : CONFIRMED")
        print("        MLF compatibility evidence   : CONFIRMED")

        stage="translate"; print("[6/10] Translating all candidates with integrated pre-execution compatibility rules...")
        source_rows=[]; key_rows=[]; cover_rows=[]; compat_rows=[]; frozen=[]
        for m in MODELS:
            rid=REPO_BY_MODEL[m]; defs=defaults_by_repo[rid]
            for cand in cmap[m]:
                tr,kr,cr=translate(m,cand,defs,fre if m=="FRNET" else None,mle if m=="MLF" else None)
                key_rows+=kr; compat_rows+=cr; cid=str(cand["config_id"]); sc=tr["source_config"]; tc=tr["training_config"]
                req=set() if tr["configuration_mode"]=="direct_constructor_kwargs" else required[m]
                missing=sorted(req-set(sc))
                for a in sorted(req):cover_rows.append({"model_id":m,"config_id":cid,"config_attribute":a,"supplied":a in sc,"value_json":json.dumps(safe(sc.get(a)),ensure_ascii=False) if a in sc else ""})
                source_rows.append({"model_id":m,"step16_source_model_key":source_keys[m],"repo_id":rid,"config_id":cid,"configuration_mode":tr["configuration_mode"],
                    "required_initial_config_attributes":len(req),"missing_required_attributes":len(missing),"missing_attribute_names":";".join(missing),
                    "source_config_json":json.dumps(safe(sc),sort_keys=True,ensure_ascii=False),"training_config_json":json.dumps(safe(tc),sort_keys=True,ensure_ascii=False),
                    "runner_file":str(runner_paths[rid]),"model_imported":False,"model_instantiated":False})
                frozen.append({"model_id":m,"step16_source_model_key":source_keys[m],"repo_id":rid,"config_id":cid,"configuration_mode":tr["configuration_mode"],
                    "step16_candidate":copy.deepcopy(cand),"source_config":sc,"training_config":tc,"required_initial_config_attributes":sorted(req),"missing_required_attributes":missing})
        sdf=pd.DataFrame(source_rows); kdf=pd.DataFrame(key_rows); cdf=pd.DataFrame(cover_rows); compdf=pd.DataFrame(compat_rows); pdf=pd.DataFrame(parser_rows); rdf=pd.DataFrame(ref_rows); edf=pd.DataFrame(evidence)
        if len(sdf)!=16:raise RuntimeError("Expected 16 source configs.")
        if (sdf.missing_required_attributes>0).any():raise RuntimeError("Source config coverage failed:\n"+sdf.loc[sdf.missing_required_attributes>0,["model_id","config_id","missing_attribute_names"]].to_string(index=False))
        if not kdf.empty and kdf.silently_ignored.any():raise RuntimeError("Candidate key silently ignored.")
        for m in MODELS:
            if int((sdf.model_id==m).sum())!=2:raise RuntimeError(f"{m}: translated candidate count != 2.")
            for rec in frozen:
                if rec["model_id"]==m and up["step21_by_model"][m]["repo_id"]!=rec["repo_id"]:raise RuntimeError(f"{m}: Step21/22 repo mismatch.")
        print(f"        Source configs frozen       : {len(sdf)}")
        print(f"        Candidate fields audited    : {len(kdf)}")
        print("        Missing source attributes   : 0")
        print(f"        Compatibility records       : {len(compdf)}")
        print("        Validation performance used : False")
        print("        Final-test information used : False")

        stage="write"; print("[7/10] Writing audits and provenance...")
        for p,df in ((OUT_CONFIG,sdf),(OUT_COVER,cdf),(OUT_KEYS,kdf),(OUT_PARSER,pdf),(OUT_REFS,rdf),(OUT_COMPAT,compdf),(OUT_EVIDENCE,edf),(OUT_REPO,recheck)):
            write_csv(p,df)
        extra=[("frozen_frnet_model_source",fre["model_path"]),("frozen_mlf_backbone_source",mle["backbone"])]
        extra += [(f"frozen_mlf_long_runner_{i+1}",p) for i,p in enumerate(mle["runners"])]
        write_csv(OUT_SOURCE_HASH,source_hashes(runner_paths,extra))

        stage="freeze"; print("[8/10] Freezing the single effective Step-22 translation...")
        if sha(Path(s0["lock_path"]))!=s0["lock_sha256"]:raise RuntimeError("Global final-test lock changed.")
        freeze={
          "schema_version":SCHEMA_VERSION,"status":"STAGE1_CANDIDATE_SOURCE_CONFIGURATION_TRANSLATION_FROZEN","generated_at_utc":now(),
          "configuration_fingerprint_sha256":s0["fingerprint"],
          "step16_manifest_sha256":up["manifest_sha"][16],"step17_manifest_sha256":up["manifest_sha"][17],"step18_manifest_sha256":up["manifest_sha"][18],
          "step19_manifest_sha256":up["manifest_sha"][19],"step20_manifest_sha256":up["manifest_sha"][20],"step21_manifest_sha256":up["manifest_sha"][21],
          "step16_registry_freeze_fingerprint_sha256":up["p16"],"step16_hyperparameter_candidates_sha256":step16_hp_sha,"step17_source_freeze_fingerprint_sha256":up["p17"],
          "step18_source_api_fingerprint_sha256":up["p18"],"step19_adapter_locus_fingerprint_sha256":up["p19"],
          "step20_transitive_backbone_fingerprint_sha256":up["p20"],"step21_adapter_contract_fingerprint_sha256":up["p21"],
          "translation_policy":{"stage1_models":8,"candidate_configurations":16,"lookback_sessions":60,"input_features":63,"forecast_length_for_forecasting_models":5,
             "directional_outputs":3,"direction_horizons":[1,3,5],"timesnet_task":"classification","timesnet_pred_len":0,
             "candidate_field_silent_ignore_allowed":False,"validation_performance_used":False,"final_test_information_used":False,
             "third_party_source_imported":False,"third_party_source_executed":False,"model_instantiated":False,"model_trained":False},
          "compatibility_policy":{"legacy_22b_22c_22d_stages_required":False,"all_compatibility_rules_frozen_in_this_step_before_execution":True,
             "frnet":{"decomposition":False,"period_list_policy":"type-only scalar default -> list; numeric value unchanged",
                      "global_frequency_policy":"disable optional branch only when 60-session dimensions are impossible",
                      "dataset_specific_period_values_borrowed":False,"principal_period_frequency_backbone_retained":True},
             "mlf":{"scales_source":"exact Step-16 candidate period_lengths","new_period_values_introduced":False,
                    "fixed_patch_num_policy":"cap only if needed to prevent zero stride at frozen Step-16 scales",
                    "activation_tag":False,"activation_tag_is_candidate_search_dimension":False,
                    "activation_tag_exact_external_dataset_reproduction_claimed":False},
             "candidate_search_space_changed":False,"performance_information_used":False,"source_code_modified":False},
          "candidate_source_configurations":frozen,"compatibility_amendments":compat_rows,"source_compatibility_evidence":evidence,
          "pipeline_state":{"development_interface_frozen":True,"stage1_model_registry_frozen":True,"official_source_snapshots_frozen":True,
             "source_apis_audited":True,"adapter_loci_audited":True,"transitive_backbones_audited":True,"adapter_contracts_frozen":True,
             "source_config_translation_frozen":True,"task_adapters_implemented":False,"models_instantiated":False,"forward_pass_run":False,
             "backward_pass_run":False,"smoke_tests_run":False,"models_fitted":False,"validation_performance_inspected":False,
             "feature_selection_applied":False,"probability_threshold_selection_performed":False},
          "final_test_protection":{"step10_seal_status":up["test_seal_status"],"sealed_test_sha256_from_metadata_only":up["sealed_test_sha256_from_metadata_only"],
             "final_test_parquet_opened_or_hashed_by_step22":False,"final_test_performance_computed":False,"global_final_test_state":s0["state"]}}
        fp=cfg.canonical_json_sha256(freeze); freeze["source_config_translation_fingerprint_sha256"]=fp; atomic_json(FREEZE,freeze); shutil.copy2(SCRIPT_FILE,FROZEN_SCRIPT)
        atomic_text(SUMMARY,"\n".join([STEP_NAME,"="*100,"Status: PASS",f"Configuration fingerprint: {s0['fingerprint']}",f"Step-21 adapter fingerprint: {up['p21']}",
          f"Step-16 hyperparameter registry SHA-256: {step16_hp_sha}",f"Candidate source configs: {len(sdf)}",f"Compatibility records: {len(compdf)}","Legacy 22B/22C/22D stages required: False",
          "FRNet period_list normalization: integrated","FRNet short-history global-frequency rule: integrated","MLF activation_tag=False completion: integrated",
          "Validation performance used: False","Final-test information used: False","Third-party source imported/executed: False","Models fitted: False",
          f"Global final-test state: {s0['state']}"])+"\n")
        pre=(OUT_CONFIG,OUT_COVER,OUT_KEYS,OUT_PARSER,OUT_REFS,OUT_COMPAT,OUT_EVIDENCE,OUT_REPO,OUT_SOURCE_HASH,FREEZE,FROZEN_SCRIPT,SUMMARY)
        write_csv(OUT_HASHES,pd.DataFrame([{"file_name":p.name,"file_path":str(p),"relative_path":rel(p),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in pre]))
        manifest={"schema_version":SCHEMA_VERSION,"step":{"step_number":22,"step_id":"22","step_name":STEP_NAME,"status":"PASS"},"frozen_at_utc":now(),"experiment_tag":cfg.EXPERIMENT_TAG,
          "configuration_fingerprint_sha256":s0["fingerprint"],"step00_manifest_sha256":s0["step00_manifest_sha256"],
          **{f"step{i:02d}_manifest_sha256":up["manifest_sha"][i] for i in range(1,22)},
          "step16_registry_freeze_fingerprint_sha256":up["p16"],"step16_hyperparameter_candidates_sha256":step16_hp_sha,"step17_source_freeze_fingerprint_sha256":up["p17"],"step18_source_api_fingerprint_sha256":up["p18"],
          "step19_adapter_locus_fingerprint_sha256":up["p19"],"step20_transitive_backbone_fingerprint_sha256":up["p20"],"step21_adapter_contract_fingerprint_sha256":up["p21"],
          "source_config_translation_fingerprint_sha256":fp,
          "counts":{"stage1_models":8,"candidate_source_configurations":len(sdf),"candidate_key_translation_rows":len(kdf),"required_config_reference_rows":len(cdf),
                    "parser_default_rows":len(pdf),"reachable_config_reference_rows":len(rdf),"compatibility_amendments":len(compdf),"source_evidence_rows":len(edf)},
          "pipeline_state":freeze["pipeline_state"],"test_protection":{"step10_seal_status":up["test_seal_status"],"global_final_test_state":s0["state"],
                    "sealed_test_parquet_opened_or_hashed_by_step22":False,"final_test_performance_computed":False},
          "artifacts":{p.name:art(p) for p in (*pre,OUT_HASHES)}}
        atomic_json(MANIFEST,manifest); msha=sha(MANIFEST); atomic_text(MANIFEST_SHA,f"{msha}  {MANIFEST.name}\n")

        stage="validate"; print("[9/10] Validating frozen Step-22 outputs..."); verify_sidecar(MANIFEST,MANIFEST_SHA,"Step 22")
        for r in manifest["artifacts"].values():
            p=Path(r["path"])
            if not p.exists() or sha(p)!=r["sha256"]:raise RuntimeError(f"Step-22 artifact mismatch: {p}")
        if sha(Path(s0["lock_path"]))!=s0["lock_sha256"]:raise RuntimeError("Global final-test lock changed.")

        stage="report"; print("[10/10] Final Step-22 report...\n")
        print("-"*140); print("SOURCE-CONFIGURATION TRANSLATION"); print("-"*140)
        print(f"Stage-1 model families            : 8")
        print(f"Candidate configurations          : {len(sdf)}")
        print(f"Candidate fields audited          : {len(kdf)}")
        print("Missing required config attributes: 0")
        print(f"Parser-default audit rows          : {len(pdf)}")
        print(f"Reachable config-reference rows    : {len(rdf)}")
        print("-"*140); print("INTEGRATED SOURCE-COMPATIBILITY RULES"); print("-"*140)
        print("FRNet decomposition                : False for 60-session interface")
        print("FRNet period_list                  : parser scalar 1 -> [1] type-only")
        print("FRNet global-frequency branch      : disabled when dimensionally impossible")
        print("FRNet dataset-specific periods used: False")
        print("MLF scales                          : exact Step-16 candidate values")
        print("MLF fixed_patch_num                 : deterministic cap only if required")
        print("MLF activation_tag                  : False compatibility completion")
        print("MLF exact external reproduction     : False")
        print("Legacy separate Steps 22B/22C/22D  : NOT REQUIRED")
        print("-"*140); print("PROTECTION STATE"); print("-"*140)
        print("Candidate search space changed      : False")
        print("Validation performance inspected    : False")
        print("Final-test information used         : False")
        print("Third-party model imported          : False")
        print("Model instantiated                  : False")
        print("Forward/backward pass run           : False")
        print("Model fitted                        : False")
        print(f"Step-10 final-test seal             : {up['test_seal_status']}")
        print(f"Global final-test state             : {s0['state']}")
        print("-"*140); print("OUTPUTS"); print("-"*140)
        print(f"Translation freeze       : {FREEZE}")
        print(f"Translation fingerprint  : {fp}")
        print(f"Manifest                 : {MANIFEST}")
        print(f"Manifest SHA-256         : {msha}")
        print("="*140)
        print("PASS: All 16 frozen Step-16 candidates were explicitly translated.")
        print("PASS: FRNet period-list/global-frequency compatibility was frozen before model execution.")
        print("PASS: MLF adaptive-patching/activation_tag compatibility was frozen before model execution.")
        print("PASS: No candidate field was silently ignored; no validation/final-test information was used.")
        print("PASS: No third-party model was imported, instantiated, smoke-tested, or trained.")
        print("PASS: The sealed final-test parquet remained unopened.")
        print("="*140)
        print("Do not run legacy Step 22B/22C/22D. Paste the complete Step-22 output for audit.")
        return 0
    except Exception as exc:
        archived=archive_partial(); FAILED_DIR.mkdir(parents=True,exist_ok=True)
        fail=FAILED_DIR/f"22_source_config_translation_FAILED_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        atomic_json(fail,{"step":STEP_NAME,"status":"FAIL","failed_at_utc":now(),"stage":stage,"error_type":type(exc).__name__,"error_message":str(exc),
                          "python_executable":sys.executable,"archived_partial_artifacts":None if archived is None else str(archived),"canonical_manifest_exists":MANIFEST.exists()})
        print("\n"+"="*140); print("STEP 22 FAILED — SOURCE-CONFIG TRANSLATION NOT FROZEN"); print("="*140)
        print(f"Stage      : {stage}\nError type : {type(exc).__name__}\nError      : {exc}\nDiagnostic : {fail}")
        if archived:print(f"Archived partial outputs: {archived}")
        print("="*140); print("Do not proceed to model import, smoke tests, or training.")
        return 30

def main():
    p=argparse.ArgumentParser(description="Clean-V3 Step-22 source-config translation freeze.")
    p.add_argument("--verify-only",action="store_true")
    a=p.parse_args()
    return verify_existing() if a.verify_only else run()

if __name__=="__main__":
    try:sys.exit(main())
    except KeyboardInterrupt:print("\nStep 22 interrupted by user."); sys.exit(130)
    except Exception as exc:
        print("\n"+"="*140); print("STEP 22 UNHANDLED FAILURE"); print("="*140); print(f"{type(exc).__name__}: {exc}\n"); traceback.print_exc(); print("="*140); sys.exit(1)
