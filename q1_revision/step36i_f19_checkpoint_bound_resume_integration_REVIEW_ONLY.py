#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F19 — checkpoint-bound per-seed recovery integration, REVIEW ONLY.

Purpose: close the F18 provenance gap before a finished seed is promoted/skipped:
its model, seed, physical inference batch and checkpoint digest MUST match the
40-entry frozen plan, and its Phase A package digest MUST match the manifest.

This module offers reusable library functions that use the F1 GPU worker and
F18 per-seed storage interfaces. Only the synthetic fixture self-test is wired
to its CLI. This is NOT a production launcher and does not provide the final-
test data loader, durable third-read journal, approved real output root, or
independent scientific/release sign-off. No third test access is authorized.

Use only --offline-preflight on the study WSL project or --portable-self-test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

PROHIBITED = {
    '--execute', '--production', '--phase-a', '--phase-b', '--authorize',
    '--read-test', '--open-test', '--run', '--resume', '--gpu', '--commit-real',
    '--worker', '--launch', '--enable', '--final-test', '--test-parquet',
}
if PROHIBITED.intersection(sys.argv[1:]):
    raise SystemExit('DENIED: F19 CLI has no real checkpoint/test execution command')

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
MAN = EXP / 'manifests'
AUD = EXP / 'audits'
ACCESS = MAN / '36_final_test_access_consumption_state.json'
STEP35 = MAN / '35_final_test_evaluation_contract_freeze_manifest.json'
APPROVAL = ROOT / 'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json'
PLAN = AUD / '35_final_test_inference_plan.csv'
THRESH = AUD / '35_final_test_threshold_contract.csv'
F1 = ROOT / 'step36i_f1_amended_production_runner_REVIEW_ONLY.py'
F17 = ROOT / 'step36i_f17_two_phase_release_implementation_REVIEW_ONLY.py'
F18 = ROOT / 'step36i_f18_durable_two_phase_artifact_store_REVIEW_ONLY.py'
F11 = ROOT / 'step36i_f11_upstream_provenance_audit_READ_ONLY.py'
EXPECTED_SHA = {
    'F1': (F1,'e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556'),
    'F17': (F17,'f865db97463535e093c9f4dcf4ff478e9ecbc50a0d6752961200f9aad98cf068'),
    'F18': (F18,'c4706fb360233473078f5f4aac3865c9525535ef6cd4e225b258d1a728037f69'),
    'F11': (F11,'3405cb13a89d7c2403a66b5f6119b18a13c42f886d7f8d05aa80e94b09709ea6'),
    'Step35': (STEP35,'6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7'),
    'Approval': (APPROVAL,'f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493'),
}
MODELS = ('DLINEAR','NLINEAR','PATCHTST','TIMESNET','ITRANSFORMER','FREDFORMER','FRNET','MLF')
SEEDS = (42,43,44,45,46)
BATCH = dict(DLINEAR=128,NLINEAR=128,PATCHTST=64,TIMESNET=64,
             ITRANSFORMER=128,FREDFORMER=16,FRNET=64,MLF=32)


def require(ok: bool, problem: str):
    if not ok: raise RuntimeError(problem)


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def authentic(path: Path, sha: str, label: str):
    require(path.is_file() and not path.is_symlink(), f'Missing/unsafe {label}')
    require(sha_file(path)==sha, f'Frozen digest mismatch: {label}')


def import_verified(tag: str, source: Path, sha: str):
    authentic(source,sha,tag)
    spec=importlib.util.spec_from_file_location('step36i_f19_import_'+tag,source)
    require(spec is not None and spec.loader is not None, f'Cannot import {tag}')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def trusted_plan(plan: pd.DataFrame, thresholds: pd.DataFrame, f1):
    """Reject plan, parameter or physical-batch drift; preserves physical order."""
    f1.plan_thresholds(plan,thresholds)
    pairs={(str(r.model_id),int(r.seed)):pd.Series(r._asdict()) for r in plan.itertuples(index=False)}
    require(len(pairs)==40 and set(pairs)=={(m,s) for m in MODELS for s in SEEDS},
            'Frozen model-seed identity changed')
    for (model,seed),row in pairs.items():
        require(int(row['physical_inference_microbatch'])==BATCH[model],
                'Physical inference batch drift')
        require(str(row['checkpoint_sha256']) and len(str(row['checkpoint_sha256']))==64,
                'Checkpoint SHA malformed')
    return pairs


def authenticate_seed_against_plan(pkg: Path, plan_row: pd.Series, phase_a_sha: str, f18):
    """F18 checks integrity; this function ALSO verifies every frozen plan field."""
    manifest=f18.verify_package(pkg)
    context=manifest['context']
    model=str(plan_row['model_id']); seed=int(plan_row['seed'])
    require(manifest['phase']=='SEED' and manifest['name']==f'seed_{model}_{seed}',
            'Saved package identity disagrees with plan')
    require(context.get('phase_a_package_sha256')==phase_a_sha,
            'Saved seed belongs to another materialization')
    require(context.get('checkpoint_sha256')==str(plan_row['checkpoint_sha256']),
            'Saved seed checkpoint hash differs from frozen plan')
    require(int(context.get('physical_inference_microbatch',-1))==BATCH[model],
            'Saved seed physical batch differs from frozen plan')
    require(context.get('model_id')==model and int(context.get('seed',-1))==seed,
            'Saved seed model/seed differs from frozen plan')
    return f18.read_seed(pkg.parent,model,seed,phase_a_sha)


def strict_seed_inventory(store: Path, phase_a_sha: str, plan_rows: dict, f18):
    done,missing=f18.resume_inventory(store,phase_a_sha)
    require(set(done)|set(missing)==set(plan_rows) and not (set(done)&set(missing)),
            'Plan/seed inventory mismatch')
    for model,seed in sorted(done):
        authenticate_seed_against_plan(store/f'seed_{model}_{seed}',
                                       plan_rows[(model,seed)],phase_a_sha,f18)
    return done,missing


def infer_and_commit_one(*, X: np.ndarray, phase_a_sha: str,
                         store: Path, plan_row: pd.Series, plan_rows: dict,
                         f1, f18, synthetic_worker=None):
    """Reusable ONE-seed compute/commit kernel; NOT linked to real-test CLI.

    The real F1 worker is the only allowed default; a substitute worker is
    accepted solely for a synthetic-only review fixture. The future release
    must independently authenticate the input arrays and access permissions
    before invoking this library function.
    """
    model=str(plan_row['model_id']);seed=int(plan_row['seed'])
    require((model,seed) in plan_rows and plan_row.equals(plan_rows[(model,seed)]),
            'Seed row does not match authoritative frozen plan')
    require(X.shape==(5300,60,63) and X.dtype==np.float32 and np.isfinite(X).all(),
            'Phase A X structure invalid')
    done,missing=strict_seed_inventory(store,phase_a_sha,plan_rows,f18)
    require((model,seed) in missing and (model,seed) not in done,
            'Completed seed cannot be executed/overwritten')
    require(f18.HEX.fullmatch(phase_a_sha) is not None,'Phase A SHA malformed')
    if synthetic_worker is not None:
        require(getattr(synthetic_worker,'synthetic_only_fixture',False) is True,
                'Custom worker must be explicitly synthetic-only')
        require(str(plan_row['checkpoint_sha256'])=='a'*64,
                'Synthetic worker cannot replace a real signed checkpoint')
        logits,probs,stats=synthetic_worker(X,plan_row)
    else:
        # The F1 worker checks the exact checkpoint hash and physical batch.
        # There is NO callable CLI route to this branch in F19.
        logits,probs,stats=f1.frozen_inference_from_preloaded_X(X,plan_row)
    require(isinstance(stats,dict) and
            str(stats.get('model_id'))==model and int(stats.get('seed',-1))==seed and
            str(stats.get('checkpoint_sha256'))==str(plan_row['checkpoint_sha256']) and
            int(stats.get('physical_inference_microbatch',-1))==BATCH[model],
            'Worker execution metadata not bound to frozen plan')
    f18.commit_seed_arrays(store,model,seed,logits,probs,
                           str(plan_row['checkpoint_sha256']),BATCH[model],phase_a_sha)
    require(np.array_equal(authenticate_seed_against_plan(
            store/f'seed_{model}_{seed}',plan_row,phase_a_sha,f18),probs),
            'Postcommit predictions differ from worker output')
    return model,seed


def verify_non_test_environment():
    for label,(path,expected) in EXPECTED_SHA.items(): authentic(path,expected,label)
    before=ACCESS.read_bytes()
    marker=json.loads(before)
    require(marker.get('raw_test_filesystem_open_count_completed')==2 and
            marker.get('final_test_scored') is False, 'Historical access ledger drift')
    approval=json.loads(APPROVAL.read_text())
    require(approval.get('record_status')=='APPROVED_BY_STUDY_OWNER_FOR_IMPLEMENTATION_AND_INDEPENDENT_REVIEW'
            and approval.get('remaining_review_gates',{}).get('additional_raw_test_read_authorized') is False,
            'Study-owner design decision cannot grant raw-test access')
    require(ACCESS.read_bytes()==before, 'Historical marker modified')
    print('PASS — F1/F17/F18/F11, Step35, F13E identities; historical 2/0, third read denied')


def synthetic_fixture_checks(f1,f18):
    """One-seed integration with same F18 storage functions, CPU/mock worker ONLY."""
    with tempfile.TemporaryDirectory(prefix='STEP36I_F19_SYNTHETIC_ONLY_') as td:
        root=Path(td)
        phase_root=root/'phase';phase_root.mkdir()
        seed_store=root/'seed';seed_store.mkdir()
        # F18's payload interface is bytes-based. No real dataset is opened.
        dummy={'arrays/X_5300x60x63.npy':b'SYNTHETIC_ONLY_X_REFERENCE',
               'arrays/labels_5300x3.npy':b'SYNTHETIC_ONLY_Y_REFERENCE',
               'arrays/metadata.parquet':b'SYNTHETIC_ONLY_METADATA',
               'arrays/ordered_keys.csv':b'SYNTHETIC_ONLY_KEYS',
               'arrays/phase_a_manifest.json':b'SYNTHETIC_ONLY_F17_MANIFEST'}
        provenance={'step35_sha256':EXPECTED_SHA['Step35'][1],
                    'amendment_sha256':EXPECTED_SHA['Approval'][1],
                    'synthetic_only':True}
        package=f18.atomic_package(phase_root,'phase_a_materialization',dummy,'PHASE_A',provenance)
        phase_a_sha=f18.phase_a_manifest_digest(package)
        # Build a synthetic 40-row frozen-schema inference plan. No disk/GPU.
        rows=[]
        for m in MODELS:
            config=f1.MODEL[m][0]
            for seed in SEEDS:
                rows.append(dict(model_id=m,config_id=config,seed=seed,
                    physical_inference_microbatch=BATCH[m],
                    checkpoint_path='/SYNTHETIC_ONLY/final_checkpoint.pt',
                    checkpoint_sha256='a'*64,
                    inference_row_order='reference_date_then_ticker',
                    inference_shuffle=False,gradient_tracking=False,
                    probability_transform='sigmoid(logit)',
                    ensemble_rule='arithmetic_mean_of_seed_probabilities_42_43_44_45_46'))
        plan=pd.DataFrame(rows)
        thresholds=pd.DataFrame([dict(model_id=m,horizon_days=h,selected_threshold=0.5)
                                 for m in MODELS for h in (1,3,5)])
        index=trusted_plan(plan,thresholds,f1)
        require(len(index)==40,'Synthetic plan incomplete')
        X=np.zeros((5300,60,63),dtype=np.float32)
        class CPUFixtureWorker:
            synthetic_only_fixture=True
            def __call__(self, X, row):
                m=str(row['model_id']);s=int(row['seed'])
                logits=np.full((5300,3), 0.02*(s-42)+0.14,dtype=np.float32)
                p=(1/(1+np.exp(-logits.astype(np.float64)))).astype(np.float32)
                metadata={'model_id':m,'seed':s,'checkpoint_sha256':'a'*64,
                          'physical_inference_microbatch':BATCH[m]}
                return logits,p,metadata
        worker=CPUFixtureWorker()
        require(infer_and_commit_one(X=X,phase_a_sha=phase_a_sha,store=seed_store,
                plan_row=index[('DLINEAR',42)],plan_rows=index,f1=f1,f18=f18,
                synthetic_worker=worker)==('DLINEAR',42),'One-seed commit failed')
        done,missing=strict_seed_inventory(seed_store,phase_a_sha,index,f18)
        require(len(done)==1 and len(missing)==39,'Resume status drift')
        print('PASS — genuine F18 one-seed commit, checkpoint-bound rehash, 1 complete/39 pending')
        def reject(fn,name):
            try: fn()
            except (RuntimeError,PermissionError): return
            raise RuntimeError('Negative control was incorrectly accepted: '+name)
        reject(lambda:infer_and_commit_one(X=X,phase_a_sha=phase_a_sha,store=seed_store,
               plan_row=index[('DLINEAR',42)],plan_rows=index,f1=f1,f18=f18,
               synthetic_worker=worker),'Duplicate inference')
        wrong=index[('DLINEAR',42)].copy()
        wrong['checkpoint_sha256']='b'*64
        reject(lambda:authenticate_seed_against_plan(seed_store/'seed_DLINEAR_42',wrong,
                                                      phase_a_sha,f18),'Checkpoint provenance drift')
        reject(lambda:strict_seed_inventory(seed_store,'b'*64,index,f18),'Phase A mismatch')
        wrong_meta=index[('DLINEAR',43)].copy()
        wrong_meta['physical_inference_microbatch']=16
        reject(lambda:infer_and_commit_one(X=X,phase_a_sha=phase_a_sha,store=seed_store,
               plan_row=wrong_meta,plan_rows=index,f1=f1,f18=f18,
               synthetic_worker=worker),'Wrong physical batch')
        print('PASS — wrong checkpoint, wrong Phase A, wrong batch, duplicate seed rejected')
        target=seed_store/'seed_DLINEAR_42'/'seed_outputs.npz'
        with target.open('ab') as fd: fd.write(b'BIT_FLIP')
        reject(lambda:strict_seed_inventory(seed_store,phase_a_sha,index,f18),'Corrupted seed')
        print('PASS — corrupted saved output blocks resume without hidden overwrite')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--offline-preflight',action='store_true')
    modes.add_argument('--portable-self-test',action='store_true')
    args=parser.parse_args()
    print('='*102)
    print('STEP 36I-F19 — CHECKPOINT-BOUND PHASE B PER-SEED INTEGRATION (REVIEW ONLY)')
    print('='*102)
    if args.offline_preflight:
        print('[1] REVIEWED SOURCE AND FROZEN WSL ACCESS BOUNDARY')
        verify_non_test_environment()
    else: print('[1] PORTABLE MODE — WSL non-test provenance deferred')
    f1=import_verified('F1',*EXPECTED_SHA['F1'])
    f18=import_verified('F18',*EXPECTED_SHA['F18'])
    print('[2] ACTUAL F18 PACKAGE / F1 PLAN AND WORKER CONTRACT, SYNTHETIC FIXTURE')
    synthetic_fixture_checks(f1,f18)
    print('[3] LIVE RELEASE AND PERMISSION GATES')
    if args.offline_preflight:
        require(ACCESS.read_bytes()==open_marker,'Access marker changed')
    print('NOT CONNECTED — third-read journal, real Phase A loader, production CLI or authorized GPU execution')
    print('PENDING — independent live-release source and scientific sign-off; separate third-read permission')
    print('='*102)
    print('STEP 36I-F19 PASS — ONE-SEED CHECKPOINT-BOUND RECOVERY INTEGRATION, SYNTHETIC ONLY')
    print('Real raw test opened: FALSE | Real GPU: FALSE | Real test scored: FALSE')
    print('Production entrypoint: DISABLED | Additional access authorized: FALSE')
    print('='*102)

if __name__=='__main__':
    try:
        # Also check the marker byte-for-byte after the synthetic fixture exercise.
        open_marker=ACCESS.read_bytes() if '--offline-preflight' in sys.argv else None
        main()
    except Exception as exc:
        print(f'F19 FAIL-CLOSED — {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
