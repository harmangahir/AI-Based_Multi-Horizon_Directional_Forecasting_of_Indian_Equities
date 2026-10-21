#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 20 — CLEAN V3 TRANSITIVE FROZEN-BACKBONE / TERMINAL-LOCUS AUDIT

Static-only stage. Recursively traces local Python dependencies under the exact
Step-17 frozen repositories, inventories nested classes/modules/returns, and
records reachable terminal/head/projection CANDIDATES for the eight Step-16
Stage-1 architectures.

No third-party import/execution, repository mutation, model instantiation,
adapter implementation, forward/backward pass, training, feature selection,
threshold selection, or sealed final-test parquet access occurs here.

Run:
    python step20_trace_nested_backbones_v3.py

Verify:
    python step20_trace_nested_backbones_v3.py --verify-only
"""
from __future__ import annotations

import argparse, ast, hashlib, json, os, shutil, subprocess, sys, traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent
if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError("Script must be inside <repository>/q1_revision/.")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import step00_config as cfg

STEP_NUMBER = 20
STEP_ID = "20"
STEP_NAME = "STEP 20 — TRANSITIVE FROZEN-BACKBONE AND TERMINAL-LOCUS AUDIT"
SCHEMA_VERSION = "1.0"

STAGE1_MODEL_IDS = (
    "DLINEAR","NLINEAR","PATCHTST","TIMESNET",
    "ITRANSFORMER","FREDFORMER","FRNET","MLF",
)
EXPECTED_MODEL_COUNT = 8
EXPECTED_FROZEN_REPOSITORY_COUNT = 8
EXPECTED_IMPLEMENTATION_REPOSITORY_COUNT = 7
EXPECTED_REFERENCE_REPOSITORY_COUNT = 1
MAX_DEPENDENCY_DEPTH = 5
MAX_FILES_PER_MODEL = 120
MAX_TEXT_LENGTH = 700
FORBIDDEN_TEST_FILENAME = "10_final_test_locked_panel.parquet"
TERMINAL_ATTRIBUTE_TOKENS = (
    "head","projection","projector","classifier","classification",
    "linear","predict","prediction","output","out_layer","fc",
)
TERMINAL_CANDIDATE_POLICY = "heuristic_name_and_reachable_class_evidence_only"

STEP01_ENVIRONMENT_AUDIT = cfg.ENVIRONMENT_DIR / "01_environment_audit.json"
UPSTREAM = {
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
}
STEP16_FREEZE = cfg.MANIFEST_DIR/"16_stage1_model_registry_freeze.json"
STEP17_FREEZE = cfg.MANIFEST_DIR/"17_official_model_source_freeze.json"
STEP18_FREEZE = cfg.MANIFEST_DIR/"18_source_api_freeze.json"
STEP19_FREEZE = cfg.MANIFEST_DIR/"19_adapter_locus_freeze.json"
STEP10_TEST_SEAL = cfg.MANIFEST_DIR/"10_final_test_seal.json"

DEPENDENCY_GRAPH_CSV = cfg.AUDIT_DIR/"20_local_dependency_graph.csv"
DISCOVERED_FILE_CSV = cfg.AUDIT_DIR/"20_discovered_python_file_inventory.csv"
CLASS_INVENTORY_CSV = cfg.AUDIT_DIR/"20_transitive_class_inventory.csv"
MODULE_ASSIGNMENT_CSV = cfg.AUDIT_DIR/"20_self_module_assignment_audit.csv"
CLASS_FORWARD_CSV = cfg.AUDIT_DIR/"20_nested_forward_audit.csv"
TERMINAL_LOCUS_CSV = cfg.AUDIT_DIR/"20_nested_terminal_locus_audit.csv"
MODEL_RESOLUTION_CSV = cfg.AUDIT_DIR/"20_model_adapter_resolution_summary.csv"
REPOSITORY_RECHECK_CSV = cfg.AUDIT_DIR/"20_repository_recheck.csv"
SOURCE_HASH_CSV = cfg.AUDIT_DIR/"20_source_file_hashes.csv"
TRANSITIVE_FREEZE_JSON = cfg.MANIFEST_DIR/"20_transitive_backbone_freeze.json"
MANIFEST_JSON = cfg.MANIFEST_DIR/"20_transitive_backbone_manifest.json"
MANIFEST_SHA = cfg.MANIFEST_DIR/"20_transitive_backbone_manifest.sha256"
HASH_REGISTER_CSV = cfg.AUDIT_DIR/"20_transitive_backbone_hashes.csv"
FROZEN_SCRIPT = cfg.MANIFEST_DIR/"step20_trace_nested_backbones_frozen.py"
SUMMARY_TXT = cfg.REPORT_DIR/"20_transitive_backbone_summary.txt"
FAILED_DIR = cfg.AUDIT_DIR/"step20_failed_attempts"

def utc_now(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def atomic_json(path:Path,payload:Dict[str,Any]):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8",newline="\n") as f:
        json.dump(payload,f,indent=2,sort_keys=True,ensure_ascii=False); f.write("\n")
    os.replace(tmp,path)

def atomic_text(path:Path,text:str):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp")
    tmp.write_text(text,encoding="utf-8"); os.replace(tmp,path)

def write_csv(path:Path,df:pd.DataFrame):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp")
    df.to_csv(tmp,index=False,float_format="%.17g"); os.replace(tmp,path)

def rel(path:Path): return str(path.resolve().relative_to(cfg.EXPERIMENT_ROOT_DIR.resolve()))
def artifact(path:Path): return {"path":str(path),"relative_path":rel(path),"size_bytes":int(path.stat().st_size),"sha256":sha256_file(path)}
def git_exe():
    p=shutil.which("git")
    if not p: raise RuntimeError("git executable not found.")
    return p

def run_git(args:list[str],cwd:Path,timeout=120)->str:
    env=os.environ.copy(); env["GIT_TERMINAL_PROMPT"]="0"
    r=subprocess.run([git_exe(),*args],cwd=str(cwd),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False,timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"GIT FAILED: git {' '.join(args)}\nCWD={cwd}\nstdout={r.stdout}\nstderr={r.stderr}")
    return r.stdout.strip()

def norm_url(s:str):
    s=s.strip().rstrip("/")
    return s[:-4].lower() if s.endswith(".git") else s.lower()

def ast_text(node,maxlen=MAX_TEXT_LENGTH):
    if node is None:return "<none>"
    try:s=ast.unparse(node)
    except Exception:s=f"<{type(node).__name__}>"
    s=s.replace("\n"," ").strip()
    return s if len(s)<=maxlen else s[:maxlen-3]+"..."

def verify_step00():
    cfg.validate_configuration()
    payload=cfg.build_configuration_payload(); fp=cfg.canonical_json_sha256(payload)
    _,msha,status=cfg.verify_existing_freeze(configuration_payload=payload,configuration_fingerprint=fp)
    if status!="VERIFIED_EXISTING": raise RuntimeError("Step-00 freeze is not VERIFIED_EXISTING.")
    lock_path=cfg.get_final_test_lock_path()
    lock=json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("state")!="LOCKED_NOT_OPENED" or lock.get("final_test_opened") is not False or lock.get("final_test_scored") is not False:
        raise RuntimeError("Global final-test lock is not pristine.")
    return {"fingerprint":fp,"step00_manifest_sha256":msha,"lock_path":str(lock_path),"lock_sha256":sha256_file(lock_path),"state":lock["state"]}

def verify_environment():
    audit=json.loads(STEP01_ENVIRONMENT_AUDIT.read_text(encoding="utf-8"))
    expected=audit.get("system",{}).get("python_executable")
    if expected and Path(expected).resolve()!=Path(sys.executable).resolve():
        raise RuntimeError(f"Wrong Python.\nFrozen={expected}\nCurrent={sys.executable}")
    return {"python_executable":expected,"pandas_version":pd.__version__,"git_executable":git_exe()}

def verify_sidecar(m:Path,s:Path,label:str):
    if not m.exists() or not s.exists(): raise RuntimeError(f"{label} manifest/sidecar missing.")
    expected=s.read_text(encoding="utf-8").strip().split()[0]; observed=sha256_file(m)
    if expected!=observed: raise RuntimeError(f"{label} manifest SHA mismatch.")
    return observed

def find_artifact(manifest,filename):
    hits=[]
    for key in ("artifacts","canonical_artifacts"):
        for rec in manifest.get(key,{}).values():
            if isinstance(rec,dict) and rec.get("path") and Path(rec["path"]).name==filename: hits.append(rec)
    if len(hits)!=1: raise RuntimeError(f"Expected one artifact {filename}; found {len(hits)}.")
    return hits[0]

def verify_fp_file(path,field):
    d=json.loads(path.read_text(encoding="utf-8")); stored=d.get(field)
    if not stored: raise RuntimeError(f"{path.name} lacks {field}.")
    body=dict(d); body.pop(field,None)
    if cfg.canonical_json_sha256(body)!=stored: raise RuntimeError(f"{path.name}: fingerprint mismatch.")
    return d,stored

def verify_upstream(step00):
    shas={}; mans={}
    for step in range(1,20):
        m,s=UPSTREAM[step]; shas[step]=verify_sidecar(m,s,f"Step {step:02d}"); mans[step]=json.loads(m.read_text(encoding="utf-8"))
    if mans[1].get("status")!="PASS" or mans[1].get("configuration_fingerprint_sha256")!=step00["fingerprint"]:
        raise RuntimeError("Step-01 mismatch.")
    if mans[2].get("step",{}).get("status")!="PASS": raise RuntimeError("Step-02 status is not PASS.")
    for step in range(3,20):
        m=mans[step]
        if m.get("step",{}).get("status")!="PASS" or m.get("configuration_fingerprint_sha256")!=step00["fingerprint"]:
            raise RuntimeError(f"Step-{step:02d} status/fingerprint mismatch.")
        for prev in range(1,step):
            field=f"step{prev:02d}_manifest_sha256"
            if field in m and m[field]!=shas[prev]: raise RuntimeError(f"Step-{step:02d} chain mismatch: {field}")
    verify_environment()
    f16,p16=verify_fp_file(STEP16_FREEZE,"registry_freeze_fingerprint_sha256")
    f17,p17=verify_fp_file(STEP17_FREEZE,"source_freeze_fingerprint_sha256")
    f18,p18=verify_fp_file(STEP18_FREEZE,"source_api_fingerprint_sha256")
    f19,p19=verify_fp_file(STEP19_FREEZE,"adapter_locus_fingerprint_sha256")
    statuses=[
        (f16.get("status"),"STAGE1_MODEL_REGISTRY_AND_SELECTION_PROTOCOL_FROZEN"),
        (f17.get("status"),"OFFICIAL_AUTHOR_MAINTAINED_MODEL_SOURCE_SNAPSHOTS_FROZEN"),
        (f18.get("status"),"FROZEN_SOURCE_APIS_AUDITED_BEFORE_ADAPTER_IMPLEMENTATION"),
        (f19.get("status"),"ADAPTER_LOCI_AUDITED_BEFORE_IMPLEMENTATION"),
    ]
    if any(a!=b for a,b in statuses): raise RuntimeError("Step-16/19 freeze status mismatch.")
    for step,field,val,path in (
        (16,"registry_freeze_fingerprint_sha256",p16,STEP16_FREEZE),
        (17,"source_freeze_fingerprint_sha256",p17,STEP17_FREEZE),
        (18,"source_api_fingerprint_sha256",p18,STEP18_FREEZE),
        (19,"adapter_locus_fingerprint_sha256",p19,STEP19_FREEZE),
    ):
        if mans[step].get(field)!=val: raise RuntimeError(f"Step-{step} manifest/freeze fingerprint mismatch.")
        rec=find_artifact(mans[step],path.name)
        if Path(rec["path"]).resolve()!=path.resolve() or rec["sha256"]!=sha256_file(path): raise RuntimeError(f"Step-{step} freeze artifact drift.")
    if f16.get("training_started") is not False: raise RuntimeError("Training already started.")
    for key in ("adapter_contracts_finalized","task_adapters_implemented","smoke_tests_run","models_fitted","feature_selection_applied","probability_threshold_selection_performed"):
        if key in f19.get("pipeline_state",{}) and f19["pipeline_state"][key] is not False: raise RuntimeError(f"Step-19 state violation: {key}")
    api=f18.get("source_api_records",[]); loci=f19.get("model_records",[])
    api_ids=[str(x["model_id"]) for x in api]
    locus_ids=[str(x["model_id"]) for x in loci]

    # Step 18 intentionally serialized its source-API records in lexical
    # model_id order, whereas Step 19 preserved the frozen Stage-1 registry
    # order. Record order is therefore not a semantic contract here.
    # Require exact membership, uniqueness and count; downstream access is
    # keyed by model_id and Step 20 itself iterates STAGE1_MODEL_IDS.
    expected_ids=set(STAGE1_MODEL_IDS)

    if (
        len(api_ids)!=EXPECTED_MODEL_COUNT
        or len(set(api_ids))!=EXPECTED_MODEL_COUNT
        or set(api_ids)!=expected_ids
    ):
        raise RuntimeError(
            "Step-18 source-API records do not contain exactly the eight "
            "frozen Stage-1 model IDs."
        )

    if (
        len(locus_ids)!=EXPECTED_MODEL_COUNT
        or len(set(locus_ids))!=EXPECTED_MODEL_COUNT
        or set(locus_ids)!=expected_ids
    ):
        raise RuntimeError(
            "Step-19 adaptation-locus records do not contain exactly the eight "
            "frozen Stage-1 model IDs."
        )
    repos=f17.get("repositories",[])
    if len(repos)!=8 or sum(r.get("repository_role")=="implementation" for r in repos)!=7 or sum(r.get("repository_role")=="reference_redirect" for r in repos)!=1:
        raise RuntimeError("Step-17 repository inventory drift.")
    seal=json.loads(STEP10_TEST_SEAL.read_text(encoding="utf-8"))
    if seal.get("status")!="SEALED_NOT_OPENED" or seal.get("global_final_test_lock_state")!="LOCKED_NOT_OPENED":
        raise RuntimeError("Step-10 test seal invalid.")
    return {
        "manifest_sha":shas,"manifests":mans,
        "step17_freeze":f17,"step18_api_by_model":{x["model_id"]:x for x in api},
        "step19_locus_by_model":{x["model_id"]:x for x in loci},
        "p16":p16,"p17":p17,"p18":p18,"p19":p19,
        "test_seal_status":seal["status"],"sealed_test_sha256_from_metadata_only":seal.get("sha256"),
    }

def outputs():
    return (DEPENDENCY_GRAPH_CSV,DISCOVERED_FILE_CSV,CLASS_INVENTORY_CSV,MODULE_ASSIGNMENT_CSV,CLASS_FORWARD_CSV,TERMINAL_LOCUS_CSV,MODEL_RESOLUTION_CSV,REPOSITORY_RECHECK_CSV,SOURCE_HASH_CSV,TRANSITIVE_FREEZE_JSON,MANIFEST_JSON,MANIFEST_SHA,HASH_REGISTER_CSV,FROZEN_SCRIPT,SUMMARY_TXT)

def assert_fresh():
    hit=[p for p in outputs() if p.exists()]
    if hit: raise RuntimeError("Step-20 outputs already exist; use --verify-only.\n"+"\n".join(map(str,hit)))

def archive_partial():
    if MANIFEST_JSON.exists(): return None
    hit=[p for p in outputs() if p.exists()]
    if not hit:return None
    FAILED_DIR.mkdir(parents=True,exist_ok=True)
    d=FAILED_DIR/f"partial_artifacts_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    d.mkdir()
    for p in hit:
        try: shutil.move(str(p),str(d/p.name))
        except Exception: pass
    return d

def recheck_repositories(f17):
    rows=[]; repo_map={}
    for rec in f17["repositories"]:
        rid=rec["repo_id"]; root=Path(rec["local_path"])
        if not (root/".git").exists(): raise RuntimeError(f"{rid}: missing Git checkout.")
        origin=run_git(["remote","get-url","origin"],root)
        commit=run_git(["rev-parse","HEAD"],root); tree=run_git(["rev-parse","HEAD^{tree}"],root)
        branch=run_git(["branch","--show-current"],root); status=run_git(["status","--porcelain","--untracked-files=all"],root)
        origin_ok=norm_url(origin)==norm_url(rec["repository_url"]); commit_ok=commit==rec["commit_sha"]; tree_ok=tree==rec["git_tree_sha"]; branch_ok=branch==rec["branch"]; clean=status==""
        files_ok=True; checked=0
        for fr in rec["required_file_hashes"]:
            p=root/fr["relative_path"]
            if not p.is_file() or sha256_file(p)!=fr["sha256"]: files_ok=False
            else: checked+=1
        passed=origin_ok and commit_ok and tree_ok and branch_ok and clean and files_ok
        if not passed: raise RuntimeError(f"{rid}: frozen repository recheck failed.")
        repo_map[rid]=rec
        rows.append({"repo_id":rid,"repository_role":rec["repository_role"],"local_path":str(root),"origin_match":origin_ok,"commit_match":commit_ok,"tree_match":tree_ok,"branch_match":branch_ok,"worktree_clean_including_untracked":clean,"required_files_checked":checked,"required_file_hashes_match":files_ok,"passed":passed})
    if len(rows)!=8: raise RuntimeError("Frozen repository count drift.")
    return repo_map,pd.DataFrame(rows)

def module_candidates(root,current,module,level):
    parts=module.split(".") if module else []; out=[]
    if level>0:
        base=current.parent
        for _ in range(level-1): base=base.parent
        t=base.joinpath(*parts) if parts else base; out += [t.with_suffix(".py"),t/"__init__.py"]
    if parts:
        for t in (root.joinpath(*parts),current.parent.joinpath(*parts)): out += [t.with_suffix(".py"),t/"__init__.py"]
    seen=set(); good=[]
    for p in out:
        try:p=p.resolve(); p.relative_to(root.resolve())
        except Exception:continue
        if p in seen:continue
        seen.add(p)
        if p.is_file():good.append(p)
    return good

def resolve_module(root,current,module,level):
    c=module_candidates(root,current,module,level); return c[0] if c else None

def resolve_from(root,current,module,level,name):
    p=resolve_module(root,current,module,level)
    if name=="*":return p
    if p is not None and p.name=="__init__.py":
        q=p.parent/f"{name}.py"
        if q.is_file():return q.resolve()
        q=p.parent/name/"__init__.py"
        if q.is_file():return q.resolve()
    if not module and level>0:
        base=current.parent
        for _ in range(level-1):base=base.parent
        q=base/f"{name}.py"
        if q.is_file():return q.resolve()
    return p

def local_imports(root,source,tree):
    rows=[]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom):
            for a in node.names:
                p=resolve_from(root,source,node.module,int(node.level),a.name)
                if p: rows.append({"import_type":"from","module":node.module or "","level":int(node.level),"imported_name":a.name,"local_alias":a.asname or a.name,"target_file":p})
        elif isinstance(node,ast.Import):
            for a in node.names:
                p=resolve_module(root,source,a.name,0)
                if p: rows.append({"import_type":"import","module":a.name,"level":0,"imported_name":"","local_alias":a.asname or a.name.split(".")[0],"target_file":p})
    return rows

def crawl(model_id,root,primary):
    q=deque([(primary.resolve(),0)]); visited=set(); edges=[]; files=[]; inventory=[]
    while q:
        source,depth=q.popleft()
        if source in visited:continue
        visited.add(source); files.append(source)
        if len(files)>MAX_FILES_PER_MODEL:raise RuntimeError(f"{model_id}: dependency safety limit exceeded.")
        text=source.read_text(encoding="utf-8",errors="replace"); tree=ast.parse(text)
        inventory.append({"model_id":model_id,"depth_first_seen":depth,"relative_path":str(source.relative_to(root)),"size_bytes":int(source.stat().st_size),"sha256":sha256_file(source),"ast_parse_ok":True})
        if depth>=MAX_DEPENDENCY_DEPTH:continue
        for item in local_imports(root,source,tree):
            target=Path(item["target_file"]).resolve(); target.relative_to(root.resolve())
            edges.append({"model_id":model_id,"depth":depth+1,"source_file":str(source.relative_to(root)),"import_type":item["import_type"],"module":item["module"],"imported_name":item["imported_name"],"local_alias":item["local_alias"],"target_file":str(target.relative_to(root)),"target_sha256":sha256_file(target)})
            if target not in visited:q.append((target,depth+1))
    return edges,files,inventory

def top_classes(tree): return [n for n in tree.body if isinstance(n,ast.ClassDef)]
def get_method(cls,name):
    return next((n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name),None)
def methods(cls):return [n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
def dotted(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):
        left=dotted(node.value); return f"{left}.{node.attr}" if left else node.attr
    return ast_text(node,220)
def self_attr(node):
    if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=="self":return node.attr
    return None
def arg_text(a):
    s=a.arg
    if a.annotation is not None:
        try:s+=": "+ast.unparse(a.annotation)
        except Exception:pass
    return s
def signature(node):
    if node is None:return "<missing>"
    pos=list(node.args.posonlyargs)+list(node.args.args); defs=[None]*(len(pos)-len(node.args.defaults))+list(node.args.defaults); out=[]; po=len(node.args.posonlyargs)
    for i,(a,d) in enumerate(zip(pos,defs)):
        s=arg_text(a)
        if d is not None:s+="="+ast_text(d,150)
        out.append(s)
        if po and i==po-1:out.append("/")
    if node.args.vararg:out.append("*"+arg_text(node.args.vararg))
    elif node.args.kwonlyargs:out.append("*")
    for a,d in zip(node.args.kwonlyargs,node.args.kw_defaults):
        s=arg_text(a)
        if d is not None:s+="="+ast_text(d,150)
        out.append(s)
    if node.args.kwarg:out.append("**"+arg_text(node.args.kwarg))
    return "("+", ".join(out)+")"

def classes_in_file(path):
    return [n.name for n in ast.parse(path.read_text(encoding="utf-8",errors="replace")).body if isinstance(n,ast.ClassDef)]

def symbol_map(root,source,tree):
    m={c.name:{"target_file":source.resolve(),"target_symbol":c.name,"resolution":"same_file"} for c in top_classes(tree)}
    for item in local_imports(root,source,tree):
        target=Path(item["target_file"]).resolve()
        if item["imported_name"]=="*":
            for c in classes_in_file(target):m[c]={"target_file":target,"target_symbol":c,"resolution":"local_star_import_class"}
        elif item["local_alias"]:
            m[item["local_alias"]]={"target_file":target,"target_symbol":item["imported_name"] if item["import_type"]=="from" else None,"resolution":"local_import"}
    return m

def inspect_class(model_id,root,source,cls):
    init=get_method(cls,"__init__"); forward=get_method(cls,"forward"); meth=methods(cls); assignments=[]; terminals=set()
    if init:
        for n in ast.walk(init):
            target=value=None
            if isinstance(n,ast.Assign) and len(n.targets)==1:target,value=n.targets[0],n.value
            elif isinstance(n,ast.AnnAssign):target,value=n.target,n.value
            if target is None or value is None:continue
            a=self_attr(target)
            if a is None:continue
            ctor=dotted(value.func) if isinstance(value,ast.Call) else "<non-call>"
            terminal=any(tok in a.lower() for tok in TERMINAL_ATTRIBUTE_TOKENS)
            if terminal:terminals.add(a)
            assignments.append({"model_id":model_id,"source_file":str(source.relative_to(root)),"class_name":cls.name,"line":int(getattr(n,"lineno",-1)),"self_attribute":a,"constructor":ctor,"assignment_expression":ast_text(value),"likely_terminal_candidate":terminal,"resolved_target_file":"","resolved_target_class":"","resolution_method":""})
    returns=[]
    if forward:
        for n in ast.walk(forward):
            if isinstance(n,ast.Return):returns.append({"model_id":model_id,"source_file":str(source.relative_to(root)),"class_name":cls.name,"line":int(getattr(n,"lineno",-1)),"return_expression":ast_text(n.value)})
    inv={"model_id":model_id,"source_file":str(source.relative_to(root)),"source_sha256":sha256_file(source),"class_name":cls.name,"class_bases":";".join(ast_text(b,180) for b in cls.bases),"method_count":len(meth),"methods":";".join(x.name for x in meth),"constructor_signature":signature(init),"forward_signature":signature(forward),"has_forward":forward is not None,"terminal_candidate_count":len(terminals),"terminal_attribute_candidates":";".join(sorted(terminals)),"terminal_candidates_are_heuristic_only":True}
    return inv,assignments,returns

def resolve_assignments(root,files,rows):
    maps={}; class_index=set()
    for source in files:
        tree=ast.parse(source.read_text(encoding="utf-8",errors="replace")); maps[source.resolve()]=symbol_map(root,source,tree)
        for c in top_classes(tree):class_index.add((source.resolve(),c.name))
    out=[]
    for original in rows:
        row=dict(original); source=(root/row["source_file"]).resolve(); ctor=row["constructor"]
        if not ctor or ctor=="<non-call>":out.append(row);continue
        parts=ctor.split("."); res=maps.get(source,{}).get(parts[0])
        if res:
            target=Path(res["target_file"]).resolve(); symbol=res.get("target_symbol")
            if symbol is None and len(parts)>=2:symbol=parts[-1]
            if symbol and (target,symbol) in class_index:
                row["resolved_target_file"]=str(target.relative_to(root)); row["resolved_target_class"]=symbol; row["resolution_method"]=res["resolution"]
        out.append(row)
    return out

def find_root(class_df,primary,preferred):
    m=class_df[(class_df.source_file==primary)&(class_df.class_name==preferred)]
    if len(m)!=1:raise RuntimeError(f"Root class resolution failed for {primary}::{preferred}.")
    return m.iloc[0]

def reachable_candidates(model_id,root_file,root_class,assign_df,class_df):
    rows=[]; q=deque([(root_file,root_class,0,"<root>",f"{root_file}::{root_class}")]); seen=set()
    while q:
        source,cls,depth,via,path=q.popleft(); key=(source,cls)
        if key in seen:continue
        seen.add(key)
        m=class_df[(class_df.model_id==model_id)&(class_df.source_file==source)&(class_df.class_name==cls)]
        if len(m)!=1:continue
        terms=[x for x in str(m.iloc[0]["terminal_attribute_candidates"]).split(";") if x and x!="nan"]
        for t in terms:rows.append({"model_id":model_id,"depth":depth,"source_file":source,"class_name":cls,"reached_via_self_attribute":via,"reachability_path":path,"terminal_attribute_candidate":t,"candidate_policy":TERMINAL_CANDIDATE_POLICY,"executable_adapter_locus_proven":False})
        if depth>=4 or assign_df.empty:continue
        ch=assign_df[(assign_df.model_id==model_id)&(assign_df.source_file==source)&(assign_df.class_name==cls)&(assign_df.resolved_target_file!="")&(assign_df.resolved_target_class!="")]
        for r in ch.itertuples(index=False):
            q.append((r.resolved_target_file,r.resolved_target_class,depth+1,r.self_attribute,path+f" -> self.{r.self_attribute} -> {r.resolved_target_file}::{r.resolved_target_class}"))
    return rows

def classify(step19,terms):
    if step19["adapter_readiness"]=="native_classification_path_available":
        return "native_classification_path_confirmed_from_step19","Native classification path is frozen by Step 19; Step 21 must freeze its exact representation/head and three-logit output."
    if any(int(r["depth"])==0 for r in terms):
        return "root_terminal_candidates_identified","Root terminal candidates exist, but Step 21 must verify runtime tensor semantics before replacement/bypass."
    if any(int(r["depth"])>0 for r in terms):
        return "nested_terminal_candidates_identified","Nested terminal candidates are statically reachable; Step 21 must verify the exact runtime tensor path."
    return "no_terminal_candidate_identified","No terminal candidate resolved statically; do not implement by guesswork."

def source_hashes():
    src=[("current_step00",Path(cfg.__file__).resolve()),("current_step20",SCRIPT_FILE.resolve()),("step16_registry_freeze",STEP16_FREEZE.resolve()),("step17_source_freeze",STEP17_FREEZE.resolve()),("step18_source_api_freeze",STEP18_FREEZE.resolve()),("step19_adapter_locus_freeze",STEP19_FREEZE.resolve()),("step10_final_test_seal_metadata",STEP10_TEST_SEAL.resolve())]
    for step,(m,s) in UPSTREAM.items():src += [(f"step{step:02d}_manifest",m.resolve()),(f"step{step:02d}_manifest_sha",s.resolve())]
    rows=[]
    for role,p in src:
        if p.name==FORBIDDEN_TEST_FILENAME:raise RuntimeError("Final-test parquet entered Step-20 source list.")
        if not p.exists():raise RuntimeError(f"Missing Step-20 source: {p}")
        rows.append({"role":role,"file_name":p.name,"absolute_path":str(p),"size_bytes":int(p.stat().st_size),"sha256":sha256_file(p)})
    return pd.DataFrame(rows)

def verify_existing():
    print("="*136); print("STEP 20 — VERIFY EXISTING TRANSITIVE BACKBONE FREEZE"); print("="*136)
    s0=verify_step00(); up=verify_upstream(s0); msha=verify_sidecar(MANIFEST_JSON,MANIFEST_SHA,"Step 20"); m=json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    if m.get("step",{}).get("status")!="PASS" or m.get("configuration_fingerprint_sha256")!=s0["fingerprint"]:return 20
    for step in range(1,20):
        if m.get(f"step{step:02d}_manifest_sha256")!=up["manifest_sha"][step]:return 21
    for field,val in (("step16_registry_freeze_fingerprint_sha256",up["p16"]),("step17_source_freeze_fingerprint_sha256",up["p17"]),("step18_source_api_fingerprint_sha256",up["p18"]),("step19_adapter_locus_fingerprint_sha256",up["p19"])):
        if m.get(field)!=val:return 22
    for rec in m.get("artifacts",{}).values():
        p=Path(rec["path"])
        if not p.exists() or sha256_file(p)!=rec["sha256"]:return 23
    freeze,fp=verify_fp_file(TRANSITIVE_FREEZE_JSON,"transitive_backbone_fingerprint_sha256")
    if m.get("transitive_backbone_fingerprint_sha256")!=fp:return 24
    repo_map,recheck=recheck_repositories(up["step17_freeze"])
    if sha256_file(Path(s0["lock_path"]))!=s0["lock_sha256"]:return 25
    print(f"[PASS] Configuration fingerprint : {s0['fingerprint']}")
    print(f"[PASS] Step-19 manifest          : {up['manifest_sha'][19]}")
    print(f"[PASS] Step-16 registry freeze   : {up['p16']}")
    print(f"[PASS] Step-17 source freeze     : {up['p17']}")
    print(f"[PASS] Step-18 source API freeze : {up['p18']}")
    print(f"[PASS] Step-19 locus freeze      : {up['p19']}")
    print(f"[PASS] Frozen repositories       : {len(repo_map)}")
    print(f"[PASS] Models audited            : {len(freeze['model_records'])}")
    print(f"[PASS] Step-20 transitive freeze : {fp}")
    print(f"[PASS] Step-20 manifest          : {msha}")
    print(f"[PASS] Step-10 test seal         : {up['test_seal_status']}")
    print(f"[PASS] Global final-test state   : {s0['state']}")
    print("="*136); print("STEP 20 VERIFICATION STATUS: PASS"); print("="*136)
    return 0

def run():
    print("="*136); print(STEP_NAME); print("="*136)
    stage="verify"
    try:
        print("[1/9] Verifying Step 00 and immutable Step 01 -> Step 19 chain...")
        s0=verify_step00(); up=verify_upstream(s0)
        print(f"       Configuration fingerprint: {s0['fingerprint']}")
        print(f"       Step-19 manifest          : {up['manifest_sha'][19]}")
        print(f"       Step-16 registry freeze   : {up['p16']}")
        print(f"       Step-17 source freeze     : {up['p17']}")
        print(f"       Step-18 source API freeze : {up['p18']}")
        print(f"       Step-19 locus freeze      : {up['p19']}")
        print("       Stage-1 architectures     : 8")
        print("       Frozen repositories       : 8")
        print(f"       Step-10 test seal         : {up['test_seal_status']}")
        print(f"       Global final-test state   : {s0['state']}")
        print("       Third-party imports       : PROHIBITED")
        print("       Model instantiation       : PROHIBITED")
        print("       Model training            : PROHIBITED")
        print("       Final-test parquet access : PROHIBITED")

        stage="collision_guard"; print("[2/9] Checking immutable Step-20 output target..."); assert_fresh(); cfg.create_directories()
        stage="repository_recheck"; print("[3/9] Rechecking the eight frozen Step-17 repository snapshots...")
        repo_map,recheck=recheck_repositories(up["step17_freeze"])
        print(f"       Repositories rechecked      : {len(recheck)}")
        print(f"       Origins/commits/trees/branches exact: {bool(recheck.passed.all())}")
        print(f"       Worktrees clean incl. untracked: {bool(recheck.worktree_clean_including_untracked.all())}")
        print(f"       Required-file hashes exact  : {bool(recheck.required_file_hashes_match.all())}")

        stage="transitive_audit"; print("[4/9] Recursively auditing local Python architecture dependencies...")
        dep_rows=[]; file_rows=[]; class_rows=[]; assign_rows=[]; fwd_rows=[]; term_rows=[]; res_rows=[]
        for model_id in STAGE1_MODEL_IDS:
            api=up["step18_api_by_model"][model_id]; loc=up["step19_locus_by_model"][model_id]; rec=repo_map[api["repo_id"]]; root=Path(rec["local_path"]).resolve()
            primary_rel=api["primary_source_file"]; primary=(root/primary_rel).resolve()
            ph=sha256_file(primary)
            if ph!=api["source_sha256"] or ph!=loc["source_sha256"]:raise RuntimeError(f"{model_id}: primary source SHA drift.")
            deps,files,inv_files=crawl(model_id,root,primary)
            mclass=[]; massign=[]; mfwd=[]
            for source in files:
                tree=ast.parse(source.read_text(encoding="utf-8",errors="replace"))
                for cls in top_classes(tree):
                    inv,a,r=inspect_class(model_id,root,source,cls); mclass.append(inv); massign.extend(a); mfwd.extend(r)
            massign=resolve_assignments(root,files,massign)
            cdf=pd.DataFrame(mclass); adf=pd.DataFrame(massign,columns=["model_id","source_file","class_name","line","self_attribute","constructor","assignment_expression","likely_terminal_candidate","resolved_target_file","resolved_target_class","resolution_method"])
            rootrow=find_root(cdf,primary_rel,api["preferred_class"])
            if rootrow["constructor_signature"]!=api["constructor_signature"] or rootrow["forward_signature"]!=api["forward_signature"]:
                raise RuntimeError(f"{model_id}: Step-18 signature drift.")
            terms=reachable_candidates(model_id,primary_rel,api["preferred_class"],adf,cdf)
            resolution,recommend=classify(loc,terms)
            children=""
            if not adf.empty:
                ch=adf[(adf.source_file==primary_rel)&(adf.class_name==api["preferred_class"])&(adf.resolved_target_file!="")]
                children=";".join(sorted({f"{r.self_attribute}->{r.resolved_target_file}::{r.resolved_target_class}" for r in ch.itertuples(index=False)}))
            nested=";".join(sorted({f"d{r['depth']}:{r['source_file']}::{r['class_name']}.{r['terminal_attribute_candidate']}" for r in terms if int(r["depth"])>0}))
            res_rows.append({"model_id":model_id,"repo_id":api["repo_id"],"primary_source_file":primary_rel,"preferred_class":api["preferred_class"],"repository_commit_sha":rec["commit_sha"],"repository_git_tree_sha":rec["git_tree_sha"],"primary_source_sha256":ph,"dependency_files_discovered":len(files),"local_dependency_edges":len(deps),"classes_discovered":len(cdf),"root_terminal_attribute_candidates":str(rootrow["terminal_attribute_candidates"]),"resolved_root_children":children,"nested_terminal_candidates":nested,"reachable_terminal_candidate_count":len(terms),"step19_readiness":loc["adapter_readiness"],"step20_resolution":resolution,"recommended_next_action":recommend,"terminal_candidates_are_heuristic_only":True,"executable_adapter_contract_frozen":False,"adapter_implemented":False,"model_instantiated":False,"model_trained":False})
            dep_rows.extend(deps); file_rows.extend(inv_files); class_rows.extend(mclass); assign_rows.extend(massign); fwd_rows.extend(mfwd); term_rows.extend(terms)
            print(f"       {model_id:<14} | files={len(files):>3} | edges={len(deps):>3} | classes={len(cdf):>3} | candidates={len(terms):>3} | {resolution}")

        dep_df=pd.DataFrame(dep_rows,columns=["model_id","depth","source_file","import_type","module","imported_name","local_alias","target_file","target_sha256"])
        file_df=pd.DataFrame(file_rows,columns=["model_id","depth_first_seen","relative_path","size_bytes","sha256","ast_parse_ok"])
        class_df=pd.DataFrame(class_rows,columns=["model_id","source_file","source_sha256","class_name","class_bases","method_count","methods","constructor_signature","forward_signature","has_forward","terminal_candidate_count","terminal_attribute_candidates","terminal_candidates_are_heuristic_only"])
        assign_df=pd.DataFrame(assign_rows,columns=["model_id","source_file","class_name","line","self_attribute","constructor","assignment_expression","likely_terminal_candidate","resolved_target_file","resolved_target_class","resolution_method"])
        fwd_df=pd.DataFrame(fwd_rows,columns=["model_id","source_file","class_name","line","return_expression"])
        term_df=pd.DataFrame(term_rows,columns=["model_id","depth","source_file","class_name","reached_via_self_attribute","reachability_path","terminal_attribute_candidate","candidate_policy","executable_adapter_locus_proven"])
        res_df=pd.DataFrame(res_rows)

        stage="structural_validation"; print("[5/9] Validating transitive source/backbone evidence...")
        if len(res_df)!=8 or tuple(res_df.model_id.astype(str))!=STAGE1_MODEL_IDS:raise RuntimeError("Resolution table mismatch.")
        if (res_df.dependency_files_discovered<=0).any() or not file_df.ast_parse_ok.all():raise RuntimeError("Dependency/AST audit failed.")
        t=res_df[res_df.model_id=="TIMESNET"]
        if len(t)!=1 or t.iloc[0].step20_resolution!="native_classification_path_confirmed_from_step19":raise RuntimeError("TimesNet native classification pathway lost.")
        print(f"       Models audited                  : {len(res_df)}")
        print(f"       Discovered Python-file rows     : {len(file_df)}")
        print(f"       Local dependency edges          : {len(dep_df)}")
        print(f"       Classes inspected               : {len(class_df)}")
        print(f"       Reachable terminal candidates   : {len(term_df)}")
        print("       TimesNet native classification  : PRESERVED")
        print("       Terminal candidates proven executable: False")

        stage="write"; print("[6/9] Writing transitive audits and provenance...")
        for path,df in ((DEPENDENCY_GRAPH_CSV,dep_df),(DISCOVERED_FILE_CSV,file_df),(CLASS_INVENTORY_CSV,class_df),(MODULE_ASSIGNMENT_CSV,assign_df),(CLASS_FORWARD_CSV,fwd_df),(TERMINAL_LOCUS_CSV,term_df),(MODEL_RESOLUTION_CSV,res_df),(REPOSITORY_RECHECK_CSV,recheck),(SOURCE_HASH_CSV,source_hashes())):write_csv(path,df)

        stage="freeze"; print("[7/9] Freezing transitive backbone evidence before executable adapters...")
        if sha256_file(Path(s0["lock_path"]))!=s0["lock_sha256"]:raise RuntimeError("Global final-test lock changed.")
        model_records=[]
        for model_id in STAGE1_MODEL_IDS:
            rr=res_df[res_df.model_id==model_id].iloc[0]
            model_records.append({
                "model_id":model_id,"repo_id":rr.repo_id,"repository_commit_sha":rr.repository_commit_sha,"repository_git_tree_sha":rr.repository_git_tree_sha,
                "primary_source_file":rr.primary_source_file,"primary_source_sha256":rr.primary_source_sha256,"preferred_class":rr.preferred_class,
                "step19_readiness":rr.step19_readiness,"step20_resolution":rr.step20_resolution,"recommended_next_action":rr.recommended_next_action,
                "root_terminal_attribute_candidates":rr.root_terminal_attribute_candidates,"resolved_root_children":rr.resolved_root_children,
                "nested_terminal_candidates":rr.nested_terminal_candidates,"terminal_candidates_are_heuristic_only":True,
                "dependency_graph":dep_df[dep_df.model_id==model_id].to_dict("records"),
                "discovered_python_files":file_df[file_df.model_id==model_id].to_dict("records"),
                "class_inventory":class_df[class_df.model_id==model_id].to_dict("records"),
                "self_module_assignments":assign_df[assign_df.model_id==model_id].to_dict("records"),
                "forward_returns":fwd_df[fwd_df.model_id==model_id].to_dict("records"),
                "terminal_candidates":term_df[term_df.model_id==model_id].to_dict("records"),
            })
        freeze={
            "schema_version":SCHEMA_VERSION,"status":"TRANSITIVE_BACKBONE_STRUCTURE_AUDITED_BEFORE_ADAPTER_IMPLEMENTATION","generated_at_utc":utc_now(),
            "configuration_fingerprint_sha256":s0["fingerprint"],
            "step16_manifest_sha256":up["manifest_sha"][16],"step17_manifest_sha256":up["manifest_sha"][17],"step18_manifest_sha256":up["manifest_sha"][18],"step19_manifest_sha256":up["manifest_sha"][19],
            "step16_registry_freeze_fingerprint_sha256":up["p16"],"step17_source_freeze_fingerprint_sha256":up["p17"],"step18_source_api_fingerprint_sha256":up["p18"],"step19_adapter_locus_fingerprint_sha256":up["p19"],
            "model_records":model_records,
            "audit_policy":{"maximum_local_dependency_depth":MAX_DEPENDENCY_DEPTH,"maximum_files_per_model":MAX_FILES_PER_MODEL,"terminal_candidate_policy":TERMINAL_CANDIDATE_POLICY,"terminal_candidates_are_executable_contracts":False,"third_party_source_imported":False,"third_party_source_executed":False,"third_party_source_modified":False,"repository_fetch_or_pull":False,"model_instantiation":False,"model_training":False,"adapter_implementation":False,"smoke_tests":False},
            "adapter_contract_constraints":{"dataset_input":"[batch,60,63]","directional_output":"[batch,3] raw logits","directional_horizons":[1,3,5],"directional_loss":"BCEWithLogitsLoss(reduction='mean')","published_backbone_preservation_required":True,"stage1_comparator_stock_embedding_allowed":False,"stage1_comparator_regime_conditioning_allowed":False,"stage1_comparator_auxiliary_regression_allowed":False,"executable_adapter_contract_finalized_here":False},
            "pipeline_state":{"development_interface_frozen":True,"stage1_model_registry_frozen":True,"official_source_snapshots_frozen":True,"source_apis_audited":True,"root_adapter_loci_audited":True,"transitive_backbones_audited":True,"final_adapter_contracts_frozen":False,"task_adapters_implemented":False,"smoke_tests_run":False,"models_fitted":False,"feature_selection_applied":False,"probability_threshold_selection_performed":False},
            "final_test_protection":{"step10_seal_status":up["test_seal_status"],"sealed_test_sha256_from_metadata_only":up["sealed_test_sha256_from_metadata_only"],"final_test_parquet_opened_or_hashed_by_step20":False,"final_test_performance_computed":False,"global_final_test_state":s0["state"]},
        }
        fp=cfg.canonical_json_sha256(freeze); freeze["transitive_backbone_fingerprint_sha256"]=fp; atomic_json(TRANSITIVE_FREEZE_JSON,freeze); shutil.copy2(SCRIPT_FILE,FROZEN_SCRIPT)
        unresolved=res_df.loc[res_df.step20_resolution=="no_terminal_candidate_identified","model_id"].tolist()
        atomic_text(SUMMARY_TXT,"\n".join([STEP_NAME,"="*108,"Status: PASS",f"Configuration fingerprint: {s0['fingerprint']}",f"Step-19 locus fingerprint: {up['p19']}",f"Models audited: {len(res_df)}",f"Discovered Python-file rows: {len(file_df)}",f"Local dependency edges: {len(dep_df)}",f"Classes inspected: {len(class_df)}",f"Reachable terminal candidates: {len(term_df)}",f"Unresolved models: {unresolved}","Terminal candidates heuristic only: True","Task adapters implemented: False","Smoke tests run: False","Models fitted: False","Final-test parquet accessed: False",f"Global final-test state: {s0['state']}"])+"\n")
        pre=(DEPENDENCY_GRAPH_CSV,DISCOVERED_FILE_CSV,CLASS_INVENTORY_CSV,MODULE_ASSIGNMENT_CSV,CLASS_FORWARD_CSV,TERMINAL_LOCUS_CSV,MODEL_RESOLUTION_CSV,REPOSITORY_RECHECK_CSV,SOURCE_HASH_CSV,TRANSITIVE_FREEZE_JSON,FROZEN_SCRIPT,SUMMARY_TXT)
        write_csv(HASH_REGISTER_CSV,pd.DataFrame([{"file_name":p.name,"file_path":str(p),"relative_path":rel(p),"size_bytes":int(p.stat().st_size),"sha256":sha256_file(p)} for p in pre]))
        manifest={
            "schema_version":SCHEMA_VERSION,"step":{"step_number":20,"step_id":"20","step_name":STEP_NAME,"status":"PASS"},"frozen_at_utc":utc_now(),"experiment_tag":cfg.EXPERIMENT_TAG,
            "configuration_fingerprint_sha256":s0["fingerprint"],"step00_manifest_sha256":s0["step00_manifest_sha256"],
            **{f"step{i:02d}_manifest_sha256":up["manifest_sha"][i] for i in range(1,20)},
            "step16_registry_freeze_fingerprint_sha256":up["p16"],"step17_source_freeze_fingerprint_sha256":up["p17"],"step18_source_api_fingerprint_sha256":up["p18"],"step19_adapter_locus_fingerprint_sha256":up["p19"],"transitive_backbone_fingerprint_sha256":fp,
            "counts":{"models_audited":len(res_df),"repository_rechecks":len(recheck),"discovered_python_file_rows":len(file_df),"local_dependency_edges":len(dep_df),"classes_inspected":len(class_df),"self_module_assignments":len(assign_df),"forward_returns":len(fwd_df),"reachable_terminal_candidates":len(term_df),"unresolved_models":len(unresolved)},
            "pipeline_state":{"development_interface_frozen":True,"stage1_model_registry_frozen":True,"official_source_snapshots_frozen":True,"source_apis_audited":True,"root_adapter_loci_audited":True,"transitive_backbones_audited":True,"final_adapter_contracts_frozen":False,"task_adapters_implemented":False,"smoke_tests_run":False,"models_fitted":False,"feature_selection_applied":False,"probability_threshold_selection_performed":False},
            "test_protection":{"step10_seal_status":up["test_seal_status"],"global_final_test_state":s0["state"],"sealed_test_parquet_opened_or_hashed_by_step20":False,"final_test_performance_computed":False},
            "artifacts":{p.name:artifact(p) for p in (*pre,HASH_REGISTER_CSV)},
        }
        atomic_json(MANIFEST_JSON,manifest); msha=sha256_file(MANIFEST_JSON); atomic_text(MANIFEST_SHA,f"{msha}  {MANIFEST_JSON.name}\n")

        stage="validate"; print("[8/9] Validating frozen Step-20 outputs..."); verify_sidecar(MANIFEST_JSON,MANIFEST_SHA,"Step 20")
        for rec in manifest["artifacts"].values():
            p=Path(rec["path"])
            if not p.exists() or sha256_file(p)!=rec["sha256"]:raise RuntimeError(f"Frozen Step-20 artifact mismatch: {p}")
        if sha256_file(Path(s0["lock_path"]))!=s0["lock_sha256"]:raise RuntimeError("Global final-test lock changed.")

        stage="report"; print("[9/9] Final Step-20 report...\n")
        print("-"*136); print("TRANSITIVE BACKBONE RESOLUTION"); print("-"*136)
        for r in res_df.itertuples(index=False):
            print(r.model_id)
            print(f"  dependency files : {r.dependency_files_discovered}")
            print(f"  dependency edges : {r.local_dependency_edges}")
            print(f"  classes          : {r.classes_discovered}")
            print(f"  root candidates  : {r.root_terminal_attribute_candidates or '<none>'}")
            print(f"  root children    : {r.resolved_root_children or '<none resolved>'}")
            print(f"  nested candidates: {r.nested_terminal_candidates or '<none>'}")
            print(f"  Step-19 state    : {r.step19_readiness}")
            print(f"  Step-20 state    : {r.step20_resolution}\n")
        print("-"*136); print("STATIC AUDIT COUNTS"); print("-"*136)
        print(f"Models audited                     : {len(res_df)}")
        print(f"Discovered Python-file rows        : {len(file_df)}")
        print(f"Local dependency edges             : {len(dep_df)}")
        print(f"Classes inspected                  : {len(class_df)}")
        print(f"self.* module assignments          : {len(assign_df)}")
        print(f"Forward return expressions         : {len(fwd_df)}")
        print(f"Reachable terminal candidates      : {len(term_df)}")
        print("-"*136); print("UNRESOLVED MODELS"); print("-"*136)
        print(f"Count                              : {len(unresolved)}")
        print(f"Models                             : {unresolved if unresolved else 'None'}")
        print("-"*136); print("FREEZE STATE"); print("-"*136)
        print("Development interface              : FROZEN")
        print("Stage-1 registry                   : FROZEN")
        print("Official source snapshots          : FROZEN")
        print("Source APIs                        : AUDITED + FROZEN")
        print("Root adaptation loci               : AUDITED + FROZEN")
        print("Transitive/nested backbones        : AUDITED + FROZEN")
        print("Terminal candidates                : HEURISTIC ONLY")
        print("Executable adapter contracts       : NOT YET FROZEN")
        print("Task adapters                      : NOT IMPLEMENTED")
        print("Models instantiated                : False")
        print("Forward/backward smoke tests       : NOT RUN")
        print("Models fitted                      : False")
        print("Final test accessed                : False")
        print(f"Global final-test state            : {s0['state']}")
        print("-"*136); print("OUTPUT FILES"); print("-"*136)
        print(f"Dependency graph        : {DEPENDENCY_GRAPH_CSV}")
        print(f"Discovered files        : {DISCOVERED_FILE_CSV}")
        print(f"Class inventory         : {CLASS_INVENTORY_CSV}")
        print(f"Module assignments      : {MODULE_ASSIGNMENT_CSV}")
        print(f"Nested forward audit    : {CLASS_FORWARD_CSV}")
        print(f"Terminal candidates     : {TERMINAL_LOCUS_CSV}")
        print(f"Resolution summary      : {MODEL_RESOLUTION_CSV}")
        print(f"Transitive freeze       : {TRANSITIVE_FREEZE_JSON}")
        print(f"Transitive fingerprint  : {fp}")
        print(f"Manifest                : {MANIFEST_JSON}")
        print(f"Manifest SHA-256        : {msha}")
        print("="*136)
        print("PASS: Frozen local architecture dependencies were recursively traced without importing or executing third-party code.")
        print("PASS: All eight Step-17 repository snapshots were revalidated including origin, commit, tree, branch, untracked cleanliness and hashes.")
        print("PASS: Every discovered local Python file was AST parsed and explicitly SHA-256 inventoried.")
        print("PASS: Reachable root/nested terminal attributes remain static candidates only; no executable adapter contract was assumed.")
        print("PASS: No model was instantiated or trained and the sealed final-test parquet remained unopened.")
        print("="*136)
        print("Do not run the old Step-20 script. Do not implement/train adapters yet. Paste the complete terminal output for audit.")
        return 0

    except Exception as exc:
        archived=archive_partial(); FAILED_DIR.mkdir(parents=True,exist_ok=True)
        fail=FAILED_DIR/f"20_transitive_backbone_audit_FAILED_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        atomic_json(fail,{"step":STEP_NAME,"status":"FAIL","failed_at_utc":utc_now(),"stage":stage,"error_type":type(exc).__name__,"error_message":str(exc),"python_executable":sys.executable,"archived_partial_artifacts":None if archived is None else str(archived),"canonical_manifest_exists":MANIFEST_JSON.exists()})
        print("\n"+"="*136); print("STEP 20 FAILED — TRANSITIVE BACKBONE EVIDENCE NOT FROZEN"); print("="*136)
        print(f"Stage      : {stage}\nError type : {type(exc).__name__}\nError      : {exc}\nDiagnostic : {fail}")
        if archived:print(f"Archived partial Step-20 outputs: {archived}")
        print("="*136); print("Do not proceed to executable adapter implementation or model training.")
        return 30

def main():
    p=argparse.ArgumentParser(description="Clean-V3 transitive frozen-backbone audit.")
    p.add_argument("--verify-only",action="store_true")
    a=p.parse_args()
    return verify_existing() if a.verify_only else run()

if __name__=="__main__":
    try:sys.exit(main())
    except KeyboardInterrupt:print("\nStep 20 interrupted by user."); sys.exit(130)
    except Exception as exc:
        print("\n"+"="*136); print("STEP 20 UNHANDLED FAILURE"); print("="*136); print(f"{type(exc).__name__}: {exc}\n"); traceback.print_exc(); print("="*136); sys.exit(1)
