#!/usr/bin/env python3
"""Step 36I-F20R1: versioned owner-self-review provenance correction to F20.

REVIEW ONLY. The sole CLI is --offline-preflight. The original frozen F20 is unmodified. Importable functions expose
post-read computation but not a raw-test opener, access authorization, or live
launcher. DO NOT invoke these library functions on real test bytes until a
separately reviewed controller has received explicit third-read authorization.

Distinct stages for that future controller:
  authenticate_upstream()                    # NON-TEST reads only
  seal_preloaded_phase_a(raw_bytes, journal, phase_dir)  # bytes from a FUTURE authorized open
  inspect_phase_a(phase_pkg)                 # no raw file access
  compute_one_seed(phase_pkg, seed_dir, model, seed)      # GPU from sealed arrays
  finalize_from_saved_seeds(phase_pkg, seed_dir, target_dir)  # no raw file access

Current CLI NEVER calls any stage that materializes, scores, writes real
outputs, loads real raw-test bytes, or initializes CUDA.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
AUD = EXP / 'audits'
MAN = EXP / 'manifests'
ACCESS = MAN / '36_final_test_access_consumption_state.json'
STEP35 = MAN / '35_final_test_evaluation_contract_freeze_manifest.json'
APPROVAL = ROOT / 'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json'
CANDIDATE = AUD / 'step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv'
PLAN = AUD / '35_final_test_inference_plan.csv'
THRESHOLDS = AUD / '35_final_test_threshold_contract.csv'
ZIP = ROOT / 'step36i_e2_source_review_bundle.zip'
SOURCES = {
    'F1': ('step36i_f1_amended_production_runner_REVIEW_ONLY.py','e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556'),
    'F3': ('step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py','b7c28f536a57653f10cc74155b38941b7e2e86b508089e5a3378bf9ff97b7539'),
    'F4': ('step36i_f4_segment_stratified_inference_REVIEW_ONLY.py','421cb3e8b83f96efaef1f3e0578e2cb7513c728aef36bda68cdc9216b07b4b10'),
    'F11': ('step36i_f11_upstream_provenance_audit_READ_ONLY.py','3405cb13a89d7c2403a66b5f6119b18a13c42f886d7f8d05aa80e94b09709ea6'),
    'F17': ('step36i_f17_two_phase_release_implementation_REVIEW_ONLY.py','f865db97463535e093c9f4dcf4ff478e9ecbc50a0d6752961200f9aad98cf068'),
    'F18': ('step36i_f18_durable_two_phase_artifact_store_REVIEW_ONLY.py','c4706fb360233473078f5f4aac3865c9525535ef6cd4e225b258d1a728037f69'),
    'F19': ('step36i_f19_checkpoint_bound_resume_integration_REVIEW_ONLY.py','727d57ff80876a214166f71c407f2b9ecd4686b7e216a7d1649c692835d9213e'),
}
FROZEN = {
    'Step35': (STEP35,'6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7'),
    'Approval': (APPROVAL,'f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493'),
    'Candidates': (CANDIDATE,'4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65'),
    'Plan': (PLAN,'a2f8489933e8d2bbd11d6bab72361f7ad066dd37bc62e1db45bb0d20d87ce086'),
    'Thresholds': (THRESHOLDS,'79dc80e0e3b18427cebd77742ff2d10b87aebcdd9ea1e459fed30ac980d0e24f'),
    'E2': (ZIP,'9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851'),
}
TEST_SHA = 'ab325ede85fbb6d7819e85472274b82d151d350f39a872a17747c6978b1dddae'
BLOCKED = {'--execute','--production','--read-test','--phase-a','--phase-b','--resume',
           '--run','--authorize','--approve','--open-test','--commit-real','--gpu','--score-test'}
if BLOCKED.intersection(sys.argv[1:]):
    raise SystemExit('DENIED — F20 provides no live release command or raw-test authorization')


def require(test, message):
    if not bool(test):
        raise RuntimeError(message)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()


def check(path: Path, wanted: str, label: str):
    require(path.is_file() and not path.is_symlink(), f'Missing or unsafe {label}: {path}')
    require(sha_file(path) == wanted, f'Frozen digest mismatch: {label}')


def import_reviewed(tag: str):
    filename, digest = SOURCES[tag]
    path = ROOT / filename
    check(path,digest,tag)
    name = 'step36i_f20_verified_' + tag.lower()
    spec = importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'Verified import failed')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_historical_access():
    require(ACCESS.is_file() and not ACCESS.is_symlink(),'Access marker unavailable')
    snapshot = ACCESS.read_bytes()
    state = json.loads(snapshot)
    require(state.get('raw_test_filesystem_open_count_completed') == 2 and
            state.get('final_test_scored') is False, 'Historical read/scoring record differs')
    return snapshot


def authenticate_upstream():
    """Call before a future authorized read. No raw-test access occurs here."""
    for label,(path,digest) in FROZEN.items(): check(path,digest,label)
    for label,(filename,digest) in SOURCES.items(): check(ROOT/filename,digest,label)
    before = check_historical_access()
    approval = json.loads(APPROVAL.read_bytes())
    require(approval.get('record_status') ==
            'APPROVED_BY_STUDY_OWNER_FOR_IMPLEMENTATION_AND_INDEPENDENT_REVIEW',
            'Scientific design approval absent')
    require(approval.get('remaining_review_gates',{}).get('additional_raw_test_read_authorized') is False,
            'Study owner approval improperly represented as raw-read permit')
    f11=import_reviewed('F11')
    snap35,snap14,preaccess=f11.load_bundle()
    features,stock_map=f11.verify_feature_map(snap35,snap14,
                    json.loads(f11.STEP14.read_text('utf-8')))
    entries=f11.expected_non_test_paths(snap35,preaccess)
    for label,(path,digest,_) in entries.items(): check(Path(path),digest,label)
    require(check_historical_access()==before,'Access marker changed during upstream authentication')
    return {'features':features,'stock_id_map':stock_map,
            'scaler_path':entries['development scaler'][0],
            'history_path':entries['development history'][0],
            'regime_path':entries['regime thresholds'][0],
            'development_history_sha256':entries['development history'][1],
            'upstream_step35_sha256':FROZEN['Step35'][1]}


def seal_preloaded_phase_a(*, raw_verified_bytes:bytes, authorized_journal_digest:str,
                           output_parent:Path, upstream:dict):
    """FUTURE EXTERNAL CALLER ONLY: consumes bytes supplied by an APPROVED one-open loader.

    This function MUST NOT itself open raw-test paths. The future controller must
    independently validate third-read permission, reserve access durably BEFORE
    opening the file, and bind this journal digest to the exact verified bytes.
    """
    import pyarrow.parquet as pq
    require(isinstance(raw_verified_bytes,bytes) and
            hashlib.sha256(raw_verified_bytes).hexdigest()==TEST_SHA,
            'Preloaded sealed byte digest mismatch')
    require(isinstance(authorized_journal_digest,str) and
            len(authorized_journal_digest)==64 and all(c in '0123456789abcdef' for c in authorized_journal_digest),
            'Externally verified access-journal binding required')
    before = check_historical_access()
    f17=import_reviewed('F17'); f18=import_reviewed('F18')
    raw_df=pq.read_table(io.BytesIO(raw_verified_bytes)).to_pandas()
    history=pd.read_parquet(upstream['history_path'])
    payloads,_=f17.phase_a_from_preloaded(raw_df,history,upstream)
    method_sha=upstream.get('owner_method_record_sha256')
    require(isinstance(method_sha,str) and len(method_sha)==64 and
            all(c in '0123456789abcdef' for c in method_sha),
            'An authenticated owner self-review payload digest is mandatory')
    context={'step35_sha256':FROZEN['Step35'][1],
             'amendment_sha256':FROZEN['Approval'][1],
             'sealed_input_sha256':TEST_SHA,
             'externally_approved_access_journal_sha256':authorized_journal_digest,
             'upstream_development_history_sha256':upstream['development_history_sha256'],
             'amended_cohort_rows':5300,'amended_dates':265,
             'methodological_review_type':'SOLE_INVESTIGATOR_SELF_REVIEW_NOT_INDEPENDENT',
             'external_independent_review_performed':False,
             'owner_method_record_sha256':method_sha}
    require(check_historical_access()==before,'Historical access marker changed')
    pkg=f18.atomic_package(output_parent,'phase_a_materialization',payloads,'PHASE_A',context)
    return f18.phase_a_manifest_digest(pkg)


def inspect_phase_a(package: Path):
    """Verify BOTH F18 outer and F17 inner manifests before deserialization."""
    f1=import_reviewed('F1'); f18=import_reviewed('F18')
    manifest=f18.verify_package(package)
    require(manifest.get('phase')=='PHASE_A' and package.name=='phase_a_materialization',
            'Phase A package identity mismatch')
    c=manifest['context']
    require(c.get('sealed_input_sha256')==TEST_SHA and
            c.get('amended_cohort_rows')==5300 and c.get('amended_dates')==265 and
            isinstance(c.get('externally_approved_access_journal_sha256'),str),
            'Missing real-input or authorized-journal provenance')
    inner_path=package/'arrays/phase_a_manifest.json'
    inner=json.loads(inner_path.read_bytes())
    require(inner.get('phase')=='A_MATERIALIZATION' and
            inner.get('step35_sha256')==FROZEN['Step35'][1] and
            inner.get('amendment_approval_sha256')==FROZEN['Approval'][1],
            'Inner Phase A manifest mismatch')
    entries=inner.get('artifacts',[])
    require({e['relative_path'] for e in entries}=={
         'arrays/X_5300x60x63.npy','arrays/labels_5300x3.npy',
         'arrays/metadata.parquet','arrays/ordered_keys.csv'},
         'Inner Phase A payload inventory drift')
    for e in entries:
        p=package/e['relative_path']
        require(p.stat().st_size==e['bytes'] and sha_file(p)==e['sha256'],
                'Inner manifest hash/size mismatch')
    with (package/'arrays/X_5300x60x63.npy').open('rb') as f:
        X=np.load(f,allow_pickle=False)
    with (package/'arrays/labels_5300x3.npy').open('rb') as f:
        y=np.load(f,allow_pickle=False)
    meta=pd.read_parquet(package/'arrays/metadata.parquet')
    keys=f1.candidate(pd.read_csv(package/'arrays/ordered_keys.csv'))
    fixed=f1.candidate(pd.read_csv(CANDIDATE))
    pd.testing.assert_frame_equal(keys,fixed,check_dtype=False)
    require(X.shape==(5300,60,63) and X.dtype==np.float32 and np.isfinite(X).all(),
            'Invalid materialized X')
    require(y.shape==(5300,3) and np.isin(y,[0,1]).all(), 'Invalid materialized y')
    require(len(meta)==5300 and np.array_equal(meta.array_row_index.to_numpy(),np.arange(5300)),
            'Metadata rows/order mismatched')
    reference=pd.DataFrame({'Ticker':meta.ticker,'Date':meta.reference_date,
           'market_session_id':meta.reference_market_session_id,
           'continuity_segment_id':meta.continuity_segment_id})
    pd.testing.assert_frame_equal(f1.candidate(reference),keys,check_dtype=False)
    for j,h in enumerate((1,3,5)):
        require(np.array_equal(y[:,j],meta[f'direction_{h}d'].to_numpy(dtype=np.int8)),
                'Label array and metadata mismatch')
    return X,y,meta,keys,f18.phase_a_manifest_digest(package)


def frozen_plan():
    f1=import_reviewed('F1'); f19=import_reviewed('F19')
    check(PLAN,FROZEN['Plan'][1],'Plan'); check(THRESHOLDS,FROZEN['Thresholds'][1],'Thresholds')
    plan=pd.read_csv(PLAN);th=pd.read_csv(THRESHOLDS)
    rows=f19.trusted_plan(plan,th,f1)
    thresholds=f1.plan_thresholds(plan,th)
    return rows,thresholds


def compute_one_seed(*,phase_package:Path,seed_store:Path,model:str,seed:int):
    """Future Phase B GPU worker: hashes the committed Phase A before one seed."""
    X,_,_,_,a_sha=inspect_phase_a(phase_package)
    plan,_=frozen_plan()
    require((model,seed) in plan,'Model/seed absent from frozen inference plan')
    f1=import_reviewed('F1');f18=import_reviewed('F18');f19=import_reviewed('F19')
    return f19.infer_and_commit_one(X=X,phase_a_sha=a_sha,store=seed_store,
                plan_row=plan[(model,seed)],plan_rows=plan,f1=f1,f18=f18)


def finalize_from_saved_seeds(*,phase_package:Path,seed_store:Path,output_parent:Path):
    """Future ALL-SEEDS-COMPLETE aggregation; NEVER reads the source test path."""
    f1=import_reviewed('F1');f3=import_reviewed('F3');f4=import_reviewed('F4')
    f18=import_reviewed('F18');f19=import_reviewed('F19')
    _,y,meta,keys,a_sha=inspect_phase_a(phase_package)
    plan,thresholds=frozen_plan()
    done,missing=f19.strict_seed_inventory(seed_store,a_sha,plan,f18)
    require(len(done)==40 and not missing,'Cannot score incomplete seed inventory')
    seed_probs={(model,seed):f19.authenticate_seed_against_plan(
         seed_store/f'seed_{model}_{seed}',plan[(model,seed)],a_sha,f18)
         for model,seed in sorted(done)}
    ensemble,metrics=f1.in_memory_five_seed_scoring(seed_probs,y,thresholds)
    stats,pairs=f4.evaluate_in_memory(keys,y,ensemble,thresholds)
    require((len(metrics),len(stats),len(pairs))==(24,72,84),'Inference output count drift')
    primary=stats.loc[stats.block_length.eq(20),['model_id','horizon_days','observed_BA']]
    merged=metrics.merge(primary,on=['model_id','horizon_days'],validate='one_to_one')
    require(len(merged)==24 and np.allclose(merged.balanced_accuracy,merged.observed_BA,atol=1e-12),
            'Scorer and bootstrap observed BA disagree')
    frames=[]
    for model in f1.MODEL:
        p=ensemble[model]
        frame=meta[['ticker','reference_date','reference_market_session_id','continuity_segment_id']].copy()
        frame.insert(0,'model_id',model)
        for k,h in enumerate((1,3,5)):
            frame[f'target_{h}d']=y[:,k]
            frame[f'probability_{h}d']=p[:,k]
            frame[f'predicted_{h}d']=(p[:,k]>=thresholds[(model,h)]).astype('int8')
        frames.append(frame)
    long=pd.concat(frames,ignore_index=True)
    require(len(long)==42400,'Long-form ensemble rows drift')
    buf=io.BytesIO();long.to_parquet(buf,index=False,engine='pyarrow')
    artifacts={'ensemble_predictions.parquet':buf.getvalue(),
          'classification_metrics.csv':metrics.to_csv(index=False,lineterminator='\n').encode(),
          'bootstrap_fdr.csv':stats.to_csv(index=False,lineterminator='\n').encode(),
          'pairwise_exploratory.csv':pairs.to_csv(index=False,lineterminator='\n').encode(),
          'frozen_seed_plan.csv':pd.DataFrame([dict(plan[z]) for z in sorted(plan)]).to_csv(
             index=False,lineterminator='\n').encode()}
    ledger={'schema':'STEP36I_F20_FINAL_EVALUATION_REVIEW_CANDIDATE_V1',
            'step35_sha256':FROZEN['Step35'][1],
            'amendment_approval_sha256':FROZEN['Approval'][1],
            'phase_a_manifest_sha256':a_sha,'frozen_plan_sha256':FROZEN['Plan'][1],
            'thresholds_sha256':FROZEN['Thresholds'][1],
            'predictions':42400,'metrics':24,'statistics':72,'primary_bh_tests':24,
            'exploratory_pairs':84,
            'source_test_file_was_not_reopened_in_phase_b':True,
            'artifacts':[{'name':n,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
                         for n,b in sorted(artifacts.items())]}
    artifacts['evaluation_manifest.json']=(json.dumps(ledger,sort_keys=True,indent=2,
                                        allow_nan=False)+'\n').encode()
    require(output_parent.is_dir() and not output_parent.is_symlink(),'Output parent must preexist')
    require(not any(output_parent.glob('.F20_STAGING_*')),'Abandoned transaction: manual review')
    dst=output_parent/'completed_evaluation'
    require(not dst.exists() and not dst.is_symlink(),'Final evaluation already exists')
    stage=output_parent/('.F20_STAGING_'+uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    # On failure, retain staging evidence; do not automatically remove or retry.
    for name,b in sorted(artifacts.items()): f3.write_unique(stage/name,b)
    f3.fsync_directory(stage)
    f3.rename_dir_no_replace(stage,dst)
    for e in ledger['artifacts']:
        require(sha_file(dst/e['name'])==e['sha256'] and
                (dst/e['name']).stat().st_size==e['bytes'],'Postcommit file hash mismatch')
    return dst


def offline_preflight():
    print('='*104)
    print('STEP 36I-F20 — CONNECTED PHASE A/B SOURCE CANDIDATE, READ-ONLY RELEASE PREFLIGHT')
    print('='*104)
    print('[1] ORIGINAL SCIENTIFIC CONTRACT, AMENDMENT, FROZEN INPUTS, ACCESS HISTORY')
    prior=check_historical_access()
    upstream=authenticate_upstream()
    require(check_historical_access()==prior,'Historical access marker modified')
    print('PASS — F1/F3/F4/F11/F17/F18/F19, Step35, F13E, plan and non-test upstream verified')
    print('PASS — 63 frozen ordered features, 20 stock IDs; 2 historic reads, 0 completed scorings')
    print('[2] SOURCE-SHAPE REVIEW AND FUTURE COMPOSITION BOUNDARIES')
    tree=ast.parse(Path(__file__).read_text('utf-8'))
    functions={item.name:item for item in tree.body if isinstance(item,ast.FunctionDef)}
    needed={'authenticate_upstream','seal_preloaded_phase_a','inspect_phase_a',
            'frozen_plan','compute_one_seed','finalize_from_saved_seeds','offline_preflight'}
    require(needed <= set(functions),'Missing source-level production operations')
    call_funcs={n.func.id for n in ast.walk(functions['offline_preflight'])
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    require(not call_funcs.intersection({'seal_preloaded_phase_a','inspect_phase_a',
                             'compute_one_seed','finalize_from_saved_seeds'}),
            'Preflight unexpectedly connected to real computation')
    require('63'==str(len(upstream['features'])) and len(upstream['stock_id_map'])==20,
            'Feature/map contract changed')
    print('PASS — fully defined Phase A materialization, sealed-array reader, 40-seed resume and finalization APIs')
    print('[3] LIVE RELEASE SAFETY')
    require(check_historical_access()==prior,'Access marker drift after preflight')
    print('PASS — no sealed input opened, no Phase A commit, no GPU, no evaluation output created')
    print('OPEN — external audited single-read journal + third-read permission not connected')
    print('OPEN — independent F20 source / methodological signoff and approved output mapping')
    print('='*104)
    print('STEP 36I-F20 PREFLIGHT PASS — COMPUTATION WIRING ONLY; NO LIVE RELEASE PERMISSION')
    print('Real raw test opened: FALSE | GPU: FALSE | Real scored: FALSE | Production CLI: DENIED')
    print('='*104)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline-preflight',action='store_true',required=True)
    args=p.parse_args()
    require(args.offline_preflight,'Only offline review permitted')
    offline_preflight()


if __name__=='__main__':
    try: main()
    except (RuntimeError,ValueError,KeyError,AssertionError,OSError) as e:
        print('F20 FAIL-CLOSED:',type(e).__name__,str(e),file=sys.stderr)
        sys.exit(1)
