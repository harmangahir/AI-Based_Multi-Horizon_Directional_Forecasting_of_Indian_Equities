#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F17 — REAL-ALGORITHM TWO-PHASE RELEASE *CANDIDATE*, REVIEW ONLY.

This source separates the preloaded-data causal materializer (Phase A) from the
40-checkpoint inference/statistics pipeline (Phase B). These are source-level
implementations for independent review, NOT a live final-test launcher.

The actual raw-file loader, live reservation journal, real output commit and
production CLI are all explicitly ABSENT/FAIL-CLOSED.  The study-owner F13E
amendment approval is NOT an authorization to read the sealed test a third time.

Permitted CLI: --offline-preflight only.  It hashes frozen non-test material,
authenticates approval and implements no action that reads sealed test files.

Further work: external independently reviewed orchestrator must require a
separate third-read authorization and sign-off, atomically seal Phase A outputs,
then resume Phase B from independently verified sealed arrays. No blind retry.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from typing import Any

PROHIBITED = {'--execute','--production','--authorize','--approve','--read-test',
              '--run','--resume','--test-parquet','--gpu','--phase-a','--phase-b',
              '--open-test','--materialize','--commit-real','--score-test'}
if PROHIBITED.intersection(sys.argv[1:]):
    raise SystemExit('DENIED — F17 source candidate cannot execute a real test or commit real outputs')

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
APPROVAL_JSON = ROOT / 'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json'
APPROVAL_MD = ROOT / 'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.md'
ACCESS_MARKER = EXP / 'manifests/36_final_test_access_consumption_state.json'
STEP35 = EXP / 'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
CANDIDATE = EXP / 'audits/step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv'
PLAN = EXP / 'audits/35_final_test_inference_plan.csv'
THRESHOLDS = EXP / 'audits/35_final_test_threshold_contract.csv'
HISTORY = EXP / 'feature_data/step12_volatility_regimes/final_development/12_final_development_scaled_regimes.parquet'
FROZEN = {
 'F1': ('step36i_f1_amended_production_runner_REVIEW_ONLY.py','e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556'),
 'F3': ('step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py','b7c28f536a57653f10cc74155b38941b7e2e86b508089e5a3378bf9ff97b7539'),
 'F4': ('step36i_f4_segment_stratified_inference_REVIEW_ONLY.py','421cb3e8b83f96efaef1f3e0578e2cb7513c728aef36bda68cdc9216b07b4b10'),
 'F7': ('step36i_f7_one_read_loader_journal_integration_REVIEW_ONLY.py','b01bd3ce80194a4981a3c518fcf2e7f46948819715a51319843fa50630a54685'),
 'F9': ('step36i_f9_preloaded_production_assembly_REVIEW_ONLY.py','8074cfe7cd1de9cf17609d4cef71aff27f910347da3ad86f3861a3cf7ad1373b'),
 'F10': ('step36i_f10_independent_assembly_review_READ_ONLY.py','d778fe150b3a06f85992b8077937a8f4509a9f2c741f0d913433443a0d07a686'),
 'F11': ('step36i_f11_upstream_provenance_audit_READ_ONLY.py','3405cb13a89d7c2403a66b5f6119b18a13c42f886d7f8d05aa80e94b09709ea6'),
 'F14': ('step36i_f14_approval_bound_production_integration_REVIEW_ONLY.py','bc34722a3582812c0d5257581e04e320bd2e59eb67f4a037a4f473c4238f068f'),
 'F15': ('step36i_f15_durable_materialization_handoff_REVIEW_ONLY.py','a2e2a05ef363f389c5208dceb4eef518227f050a8294d03326f962767fc8289c'),
 'F16': ('step36i_f16_two_phase_resume_independent_REVIEW_ONLY.py','938d7e08774fcd7b61f3e4dc3fdd5865925ef8a234df6b3538c388784f1f7382'),
}
STEP35_SHA = '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7'
APPROVAL_JSON_SHA = 'f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493'
APPROVAL_MD_SHA = 'c53013b5ce6082356e1e2a3e3df137203eeb4ea4270510596255bcbb899f4d49'
CANDIDATE_SHA = '4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65'


def require(ok: bool, message: str) -> None:
    if not ok: raise RuntimeError(message)


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, digest: str, label: str) -> None:
    require(path.is_file() and not path.is_symlink(),f'Unsafe/missing {label}: {path}')
    require(sha(path)==digest,f'SHA-256 differs: {label}')


def frozen_import(tag: str):
    filename,digest=FROZEN[tag]
    path=ROOT/filename
    verify(path,digest,tag)
    name='step36i_f17_dependency_'+tag.lower()
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,f'Cannot import {tag}')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def approve_design_record() -> dict:
    verify(APPROVAL_JSON,APPROVAL_JSON_SHA,'F13E JSON')
    verify(APPROVAL_MD,APPROVAL_MD_SHA,'F13E Markdown')
    doc=json.loads(APPROVAL_JSON.read_text('utf-8'))
    require(doc['record_status']=='APPROVED_BY_STUDY_OWNER_FOR_IMPLEMENTATION_AND_INDEPENDENT_REVIEW',
            'Study owner scientific-design approval missing')
    require(doc['immutable_original_scientific_contract_sha256']==STEP35_SHA,
            'Approval not bound to Step35')
    c=doc['approved_amended_cohort']
    require((c['endpoints'],c['dates'],c['stocks'],c['segment_dates'])==(5300,265,20,[48,217]),
            'Amended cohort drift')
    require(doc['approved_candidate_keys_sha256']==CANDIDATE_SHA,
            'Approval/candidate mismatch')
    require(doc['third_raw_test_access_authorized'] is False if 'third_raw_test_access_authorized' in doc else True,
            'Approval is not a read permit')
    return doc


def _np_bytes(x: Any) -> bytes:
    import numpy as np
    b=io.BytesIO(); np.save(b,x,allow_pickle=False);return b.getvalue()


def _parquet_bytes(df: Any) -> bytes:
    b=io.BytesIO(); df.to_parquet(b,index=False,engine='pyarrow');return b.getvalue()


def _csv_bytes(df: Any) -> bytes:
    return df.to_csv(index=False,lineterminator='\n').encode('utf-8')


def _manifest(payloads: dict[str,bytes], context: dict, phase: str) -> dict:
    # This is a pure serializer. Final verification and no-overwrite COMMIT
    # must occur in the external approved release transaction.
    require(phase in ('A_MATERIALIZATION','B_INFERENCE'),'Unsupported phase')
    allowed_base = ('arrays/','inference/','evaluation/')
    artifacts=[]
    for relative,data in sorted(payloads.items()):
        p=Path(relative)
        require(not p.is_absolute() and '..' not in p.parts and
                not '\\' in relative and relative.startswith(allowed_base),
                f'Unsafe artifact name: {relative}')
        require(isinstance(data,bytes),'Nonbyte artifact')
        artifacts.append({'relative_path':relative,'bytes':len(data),
                          'sha256':hashlib.sha256(data).hexdigest()})
    return {'schema':'STEP36I_F17_TWO_PHASE_CANDIDATE_REVIEW_ONLY_V1',
            'phase':phase,'step35_sha256':STEP35_SHA,
            'amendment_approval_sha256':APPROVAL_JSON_SHA,
            'live_access_and_commit_authorized':False,
            'requires_independent_source_review':True,
            'context':context,'artifacts':artifacts}


def phase_a_from_preloaded(preloaded_test, signed_history, upstream) -> tuple[dict[str,bytes],dict]:
    """Actual causal algorithm on ALREADY decoded input; NO raw-test loader here.

    NOT callable from F17 CLI. Must first pass a future approved release's
    journal/authorization and live upstream integrity checks. Does not run GPU.
    Returns bytes for an independent durable atomic Phase A transaction.
    """
    import numpy as np
    import pandas as pd
    f1=frozen_import('F1')
    keys=f1.candidate(pd.read_csv(CANDIDATE))
    X,meta=f1.materialize_preloaded_from_frozen_helpers(
        preloaded_test,signed_history,keys,upstream)
    require(X.shape==(5300,60,63) and X.dtype==np.float32 and np.isfinite(X).all(),
            'Incorrect Phase A array')
    y=meta[[f'direction_{h}d' for h in (1,3,5)]].to_numpy(dtype=np.int8)
    require(y.shape==(5300,3) and np.isin(y,[0,1]).all(),'Incorrect Phase A labels')
    observed=pd.DataFrame({'Ticker':meta.ticker,'Date':meta.reference_date,
        'market_session_id':meta.reference_market_session_id,
        'continuity_segment_id':meta.continuity_segment_id})
    pd.testing.assert_frame_equal(f1.candidate(observed),keys,check_dtype=False)
    payloads={'arrays/X_5300x60x63.npy':_np_bytes(X),
              'arrays/labels_5300x3.npy':_np_bytes(y),
              'arrays/metadata.parquet':_parquet_bytes(meta),
              'arrays/ordered_keys.csv':_csv_bytes(keys)}
    manifest=_manifest(payloads,{'X_shape':[5300,60,63],'y_shape':[5300,3],
        'keys_sha256':CANDIDATE_SHA,'stock_count':20,'dates':265,
        'segments':[48,217]},'A_MATERIALIZATION')
    payloads['arrays/phase_a_manifest.json']=(json.dumps(manifest,sort_keys=True,
        indent=2,allow_nan=False)+'\n').encode('utf-8')
    return payloads,manifest


def phase_b_from_verified_arrays(X, y, meta, phase_a_manifest: dict, upstream: dict):
    """Real checkpoint inference ALGORITHM from externally VERIFIED sealed arrays.

    Does not read sealed source or alter access journal. Does not write output.
    The caller must independently verify the Phase A package and its SHA-256s
    before deserializing, and must commit each 40-seed result durably so GPU
    interruptions cannot silently promote incomplete outputs.
    """
    import numpy as np
    import pandas as pd
    f1=frozen_import('F1'); f4=frozen_import('F4'); f9=frozen_import('F9')
    require(phase_a_manifest.get('phase')=='A_MATERIALIZATION' and
            phase_a_manifest.get('step35_sha256')==STEP35_SHA and
            phase_a_manifest.get('amendment_approval_sha256')==APPROVAL_JSON_SHA,
            'Phase A handoff provenance absent')
    require(X.shape==(5300,60,63) and X.dtype==np.float32 and
            np.isfinite(X).all() and y.shape==(5300,3) and
            np.isin(y,[0,1]).all(),'Handoff arrays do not match interface')
    require(len(meta)==5300 and np.array_equal(meta.array_row_index.to_numpy(),
            np.arange(5300)),'Metadata order differs')
    keys=f1.candidate(pd.read_csv(CANDIDATE))
    present=pd.DataFrame({'Ticker':meta.ticker,'Date':meta.reference_date,
        'market_session_id':meta.reference_market_session_id,
        'continuity_segment_id':meta.continuity_segment_id})
    pd.testing.assert_frame_equal(f1.candidate(present),keys,check_dtype=False)
    plan=pd.read_csv(PLAN)
    thresholds=f1.plan_thresholds(plan,pd.read_csv(THRESHOLDS))
    index={(str(r.model_id),int(r.seed)):pd.Series(r._asdict())
           for r in plan.itertuples(index=False)}
    require(len(index)==40,'Frozen plan cardinality drift')
    logits_outputs={}; seed_probs={}; records=[]
    for model in f1.MODEL:
        for seed in f1.SEEDS:
            row=index[(model,seed)]
            logits,probs,metrics=f1.frozen_inference_from_preloaded_X(X,row)
            require(logits.shape==(5300,3) and probs.shape==(5300,3),
                    'Checkpoint output shape drift')
            seed_probs[(model,seed)]=probs
            logits_outputs[(model,seed)]=(logits,probs)
            records.append({'model_id':model,'seed':seed,
                'checkpoint_sha256':row['checkpoint_sha256'],
                'physical_microbatch':int(row['physical_inference_microbatch']),
                'peak_cuda_allocated_MiB':float(metrics['peak_cuda_allocated_MiB'])})
    ensemble,metrics=f1.in_memory_five_seed_scoring(seed_probs,y,thresholds)
    stats,pairs=f4.evaluate_in_memory(keys,y,ensemble,thresholds)
    require(len(metrics)==24 and len(stats)==72 and len(pairs)==84,
            'Scoring cardinality drift')
    primary=stats.loc[stats.block_length.eq(20),
                      ['model_id','horizon_days','observed_BA']]
    merged=metrics.merge(primary,on=['model_id','horizon_days'],validate='one_to_one')
    require(len(merged)==24 and np.allclose(merged.balanced_accuracy,
            merged.observed_BA,atol=1e-12), 'Score and bootstrap disagreement')
    payloads={}
    for (model,seed),(logits,probs) in logits_outputs.items():
        b=io.BytesIO();np.savez_compressed(b,logits=logits,probabilities=probs)
        payloads[f'inference/{model}_seed{seed}.npz']=b.getvalue()
    long_frames=[]
    for model in f1.MODEL:
        frame=meta[['ticker','reference_date','continuity_segment_id',
                    'reference_market_session_id']].copy()
        frame.insert(0,'model_id',model)
        prob=np.asarray(ensemble[model])
        for j,h in enumerate((1,3,5)):
            frame[f'target_{h}d']=y[:,j]
            frame[f'probability_{h}d']=prob[:,j]
            frame[f'predicted_{h}d']=(prob[:,j]>=thresholds[(model,h)]).astype(np.int8)
        long_frames.append(frame)
    long=pd.concat(long_frames,ignore_index=True)
    require(len(long)==42400,'Ensemble length drift')
    payloads.update({'evaluation/predictions.parquet':_parquet_bytes(long),
        'evaluation/classification_metrics.csv':_csv_bytes(metrics),
        'evaluation/bootstrap_fdr.csv':_csv_bytes(stats),
        'evaluation/pairwise_exploratory.csv':_csv_bytes(pairs),
        'evaluation/inference_registry.csv':_csv_bytes(pd.DataFrame(records))})
    manifest=_manifest(payloads,{'checkpoints':40,'ensembles':8,
        'metrics':24,'bootstrap_rows':72,'primary_tests':24,'pairs':84,
        'phase_a_manifest_sha256':hashlib.sha256((json.dumps(phase_a_manifest,
             sort_keys=True,indent=2,allow_nan=False)+'\n').encode()).hexdigest()},
        'B_INFERENCE')
    payloads['evaluation/phase_b_manifest.json']=(json.dumps(manifest,
        indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
    return payloads,manifest


def live_one_read_loader(*_args,**_kwargs):
    raise PermissionError('DENIED — independent journal + third-read authorization absent')


def live_phase_a_commit(*_args,**_kwargs):
    raise PermissionError('DENIED — atomic real package transaction not approved')


def live_phase_b_resume(*_args,**_kwargs):
    raise PermissionError('DENIED — production resume and per-checkpoint commit not implemented')


def live_production_cli(*_args,**_kwargs):
    raise PermissionError('DENIED — independent source/method signoff and raw-test access permit absent')


def audit(*, portable: bool) -> None:
    print('='*102)
    print('STEP 36I-F17 — TWO-PHASE PRODUCTION COMPUTATION SOURCE CANDIDATE: OFFLINE AUDIT')
    print('='*102)
    print('[1] SCIENTIFIC APPROVAL / FROZEN SOURCE AND CONTRACT')
    approval=approve_design_record()
    require(approval['record_status'].startswith('APPROVED_BY_STUDY_OWNER_'),
            'Study-owner design approval missing')
    for label,(name,expected) in FROZEN.items(): verify(ROOT/name,expected,label)
    if not portable:
        verify(STEP35,STEP35_SHA,'Step35')
        verify(CANDIDATE,CANDIDATE_SHA,'Amended keys')
    print('PASS — F1/F3/F4/F7/F9/F10/F11/F14/F15/F16 and F13E records verified')
    print('PASS — Step35/candidate verified' if not portable else 'PORTABLE — WSL Step35/candidate live checks deferred')
    print('[2] ACTUAL PHASE A / B COMPUTATIONAL API STRUCTURE')
    tree=ast.parse(Path(__file__).read_text('utf-8'))
    func={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    require({'phase_a_from_preloaded','phase_b_from_verified_arrays','live_one_read_loader',
             'live_phase_a_commit','live_phase_b_resume','live_production_cli'}<=set(func),
            'Required interfaces not present')
    phaseA=ast.unparse(func['phase_a_from_preloaded'])
    phaseB=ast.unparse(func['phase_b_from_verified_arrays'])
    require('materialize_preloaded_from_frozen_helpers' in phaseA and
            'frozen_inference_from_preloaded_X' not in phaseA,
            'Phase A must materialize without inference')
    for token in ('frozen_inference_from_preloaded_X','in_memory_five_seed_scoring',
                  'evaluate_in_memory','phase_a_manifest'):
        require(token in phaseB,f'Phase B missing binding: {token}')
    require(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and
                    n.func.id in {'phase_a_from_preloaded','phase_b_from_verified_arrays'}
                    for n in ast.walk(func['audit'])),
            'Audit unexpectedly executing production computation')
    print('PASS — Phase A materializer separate from Phase B frozen model/scorer/bootstrap')
    print('[3] EXPLICIT DISABLED REAL ACCESS, COMMIT AND RESUME GATES')
    for function in (live_one_read_loader,live_phase_a_commit,
                     live_phase_b_resume,live_production_cli):
        try: function()
        except PermissionError: continue
        raise RuntimeError('Real production entrypoint unexpectedly open')
    print('PASS — four real execution entrypoints rejected; no authorized launcher')
    print('[4] WSL NON-TEST INPUT / IMMUTABILITY REVIEW')
    if portable:
        print('PORTABLE ONLY — live source paths and marker not accessed')
    else:
        before=ACCESS_MARKER.read_bytes()
        state=json.loads(before)
        require(state.get('raw_test_filesystem_open_count_completed')==2 and
                state.get('final_test_scored') is False,'Historic state changed')
        # Do not open/test stat/hash the sealed test file, even incidentally.
        require(HISTORY.is_file() and PLAN.is_file() and THRESHOLDS.is_file(),
                'Frozen non-test inputs unavailable')
        require(ACCESS_MARKER.read_bytes()==before,'Live access marker changed')
        print('PASS — development input paths present; ledger remains 2 reads/0 scoring')
    print('[5] OPEN RELEASE ENGINEERING AND AUTHORIZATION ISSUES')
    print('OPEN — independent authentication of signed upstream bytes inside future live launcher')
    print('OPEN — durable third-read journal and single-descriptor sealed-test loader')
    print('OPEN — atomic Phase A materialization seal, fail-closed Phase B recovery and per-seed commit')
    print('OPEN — independent release-source/methodological signoff and separate third-read authorization')
    print('='*102)
    print('STEP 36I-F17 SOURCE PREFLIGHT: PASS — LIVE PRODUCTION NOT IMPLEMENTED/AUTHORIZED')
    print('Sealed test read: FALSE | GPU: FALSE | Real test scored: FALSE | Production CLI: DENIED')
    print('='*102)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    exclusive=parser.add_mutually_exclusive_group(required=True)
    exclusive.add_argument('--offline-preflight',action='store_true')
    exclusive.add_argument('--portable-self-test',action='store_true')
    args=parser.parse_args()
    audit(portable=args.portable_self_test)

if __name__=='__main__':
    try: main()
    except (OSError,RuntimeError,ValueError,KeyError,AssertionError) as exc:
        print(f'STEP36I-F17 FAIL-CLOSED: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
