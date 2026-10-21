#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F5: independently audit F4's PROPOSED segment bootstrap.

Usage (after copying beside F1/F2/F3/F4 and the E2 ZIP):
  PYTHONDONTWRITEBYTECODE=1 python step36i_f5_independent_statistical_audit_READ_ONLY.py --offline-preflight

READ-ONLY, synthetic-only. No sealed test data, no GPU, no production route,
no real output artifacts, no scientific-protocol approval or additional read.
The proposed segment-stratified analysis is a POST-FREEZE AMENDMENT.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd

_DENY = {"--execute", "--production", "--worker", "--authorize", "--approve",
         "--read-test", "--score-test", "--run", "--resume", "--commit-real",
         "--materialize", "--final-test", "--enable", "--test-parquet"}
if set(sys.argv[1:]) & _DENY:
    raise SystemExit("DENIED: F5 is an independent synthetic-only audit")

BASE = Path(__file__).resolve().parent
E2 = BASE / 'step36i_e2_source_review_bundle.zip'
F1 = BASE / 'step36i_f1_amended_production_runner_REVIEW_ONLY.py'
F2 = BASE / 'step36i_f2_independent_source_review_READ_ONLY.py'
F3 = BASE / 'step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py'
F4 = BASE / 'step36i_f4_segment_stratified_inference_REVIEW_ONLY.py'
EXP = BASE.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
STEP35 = EXP / 'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
ACCESS = EXP / 'manifests/36_final_test_access_consumption_state.json'
CANDIDATE = EXP / 'audits/step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv'
THRESHOLDS = EXP / 'audits/35_final_test_threshold_contract.csv'
E4_REPORT = EXP / 'audits/step36i_e4_synthetic_artifact_transaction_draft/36i_e4_synthetic_transaction_DRAFT.json'
HASH = {
    'E2': '9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851',
    'F1': 'e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556',
    'F2': 'ec6400f0f356b7a9888c443890b333f6fa6c6d80f6d29d700625eba4335f00bc',
    'F3': 'b7c28f536a57653f10cc74155b38941b7e2e86b508089e5a3378bf9ff97b7539',
    'F4': '421cb3e8b83f96efaef1f3e0578e2cb7513c728aef36bda68cdc9216b07b4b10',
    'STEP35': '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7',
    'COHORT': '4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65',
    'THRESHOLDS': '79dc80e0e3b18427cebd77742ff2d10b87aebcdd9ea1e459fed30ac980d0e24f',
    'E4': '30fbd08cecba4e65eb427358c6a24c9d28b984bdf889afc83668a6c9952ab925',
}
N_DATES, N_STOCKS, N, REPS = 265, 20, 5300, 2000
SEGMENTS = (48, 217)
BLOCKS = (10, 20, 40)
MODELS = ('DLINEAR','NLINEAR','PATCHTST','TIMESNET','ITRANSFORMER','FREDFORMER','FRNET','MLF')
HORIZONS = (1,3,5)


def need(condition, msg):
    if not bool(condition):
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def verified(path, expected, label):
    need(path.is_file() and not path.is_symlink(), f'Missing/unsafe {label}: {path}')
    need(sha(path) == expected, f'SHA-256 mismatch: {label}')


def load_read_only(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, f'Cannot import {name}')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def independent_indices(block):
    """Reference index generator: sequential explicit circular-block filling.

    Match the frozen stream of random block-start draws, but independently
    construct block positions in Python (not via F4's broadcasting/reshape).
    """
    rng = np.random.default_rng(20261013)
    res = np.empty((REPS, N_DATES), dtype=np.int64)
    offset = 0
    for seg_len in SEGMENTS:
        nb = (seg_len + block - 1)//block
        starts = rng.integers(0, seg_len, size=(REPS,nb))
        for r in range(REPS):
            vals = []
            for st in starts[r]:
                vals.extend((int(st)+j) % seg_len for j in range(block))
                if len(vals) >= seg_len:
                    break
            res[r,offset:offset+seg_len] = offset + np.asarray(vals[:seg_len])
        offset += seg_len
    return res


def independent_ba(y_true, pred):
    y_true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    pos = int(np.count_nonzero(y_true))
    neg = int(y_true.size-pos)
    need(pos>0 and neg>0, 'BA undefined for one-class sample')
    return 0.5*(float(np.count_nonzero(pred & y_true))/pos +
                float(np.count_nonzero((~pred) & (~y_true)))/neg)


def independent_bh(pvals):
    """Independent descending BH monotone adjustment, not calling F4."""
    pvals = np.asarray(pvals, dtype=float)
    need(pvals.shape == (24,) and np.isfinite(pvals).all(), 'Invalid independent BH input')
    order = np.argsort(pvals, kind='stable')
    q = np.zeros(24, dtype=float)
    ceiling = 1.0
    for rank in range(24, 0, -1):
        index = int(order[rank-1])
        ceiling = min(ceiling, 24.0*pvals[index]/rank)
        q[index] = ceiling
    return q


def reference_engine(keys, truth, preds, thresholds, actual_stats, actual_pairs):
    """Reconstruct all 72 tests and 84 pairwise CIs without calling F4 statistics.

    Uses replicate-by-date multiplicities and matrix multiplication, rather
    than F4's per-model indexed confusion-matrix summation.
    """
    need(len(keys) == N and truth.shape == (N,3), 'Invalid independent fixture')
    y = truth.reshape(N_DATES,N_STOCKS,3).astype(bool)
    daily = np.empty((N_DATES,8,3,4), dtype=np.int16)
    observed = np.empty((8,3), dtype=float)
    for mi, model in enumerate(MODELS):
        probs = preds[model].reshape(N_DATES,N_STOCKS,3)
        for hi,h in enumerate(HORIZONS):
            pred = probs[:,:,hi] >= thresholds[(model,h)]
            target = y[:,:,hi]
            observed[mi,hi] = independent_ba(target.ravel(),pred.ravel())
            daily[:,mi,hi,0] = np.count_nonzero((~target)&(~pred),axis=1)
            daily[:,mi,hi,1] = np.count_nonzero((~target)&(pred),axis=1)
            daily[:,mi,hi,2] = np.count_nonzero(target&(~pred),axis=1)
            daily[:,mi,hi,3] = np.count_nonzero(target&pred,axis=1)
    need(np.array_equal(daily.sum(-1),np.full((N_DATES,8,3),20)),
         'Daily cluster count invalid')
    bs_at20 = None
    for block in BLOCKS:
        indices = independent_indices(block)
        weights = np.zeros((REPS,N_DATES),dtype=np.float64)
        rr = np.repeat(np.arange(REPS),N_DATES)
        np.add.at(weights,(rr,indices.ravel()),1)
        need(np.all(weights.sum(1)==N_DATES),'Wrong bootstrap sample length')
        need(np.all(weights[:,:48].sum(1)==48) and
             np.all(weights[:,48:].sum(1)==217), 'Bootstrap segment weights drift')
        bs = np.empty((REPS,8,3),dtype=float)
        for mi,model in enumerate(MODELS):
            for hi,h in enumerate(HORIZONS):
                cc = weights @ daily[:,mi,hi,:].astype(float)
                pos = cc[:,2]+cc[:,3]
                neg = cc[:,0]+cc[:,1]
                need(np.all(pos>0) and np.all(neg>0), 'Undefined replicated BA')
                bs[:,mi,hi] = 0.5*(cc[:,3]/pos+cc[:,0]/neg)
                sub = actual_stats[(actual_stats.block_length==block)&
                                  (actual_stats.model_id==model)&
                                  (actual_stats.horizon_days==h)]
                need(len(sub)==1,f'Exactly one statistic expected for {model}/{h}/L{block}')
                r = sub.iloc[0]
                qlo,qhi = np.quantile(bs[:,mi,hi],[0.025,0.975])
                observed_ba = observed[mi,hi]
                pv = (1+np.count_nonzero(bs[:,mi,hi]-observed_ba >= observed_ba-.5))/(REPS+1)
                for col,want in [('observed_BA',observed_ba),
                                 ('ci_lower_percentile_95',qlo),
                                 ('ci_upper_percentile_95',qhi),
                                 ('p_one_sided_centered',pv)]:
                    need(np.isclose(float(r[col]),want,rtol=0,atol=1e-12),
                         f'Independent {col} discrepancy {model}/{h}/L{block}: {r[col]} vs {want}')
                need(bool(r['primary_test']) == (block==20),'Incorrect primary designation')
        if block==20:
            bs_at20 = bs
            subset = actual_stats[actual_stats.block_length==20]
            sorted_subset = subset.set_index(['model_id','horizon_days']).loc[
                [(m,h) for m in MODELS for h in HORIZONS]].reset_index()
            need(np.allclose(sorted_subset['q_BH_primary_only'].to_numpy(),
                             independent_bh(sorted_subset['p_one_sided_centered'].to_numpy()),
                             atol=1e-12,rtol=0), 'BH q-value oracle failure')
    need(bs_at20 is not None and len(actual_pairs)==84,'Pairwise family incorrect')
    for hi,h in enumerate(HORIZONS):
        for mi in range(8):
            for mj in range(mi+1,8):
                m0,m1 = MODELS[mi],MODELS[mj]
                subset = actual_pairs[(actual_pairs.horizon_days==h)&
                                      (actual_pairs.model_a==m0)&
                                      (actual_pairs.model_b==m1)]
                need(len(subset)==1,'Pairwise missing/duplicate')
                r = subset.iloc[0]
                lo,up = np.quantile(bs_at20[:,mi,hi]-bs_at20[:,mj,hi],[.025,.975])
                for col,want in [('BA_difference_a_minus_b',observed[mi,hi]-observed[mj,hi]),
                                 ('ci_lower_percentile_95',lo),
                                 ('ci_upper_percentile_95',up)]:
                    need(np.isclose(float(r[col]),want,rtol=0,atol=1e-12),
                         f'Pairwise {col} mismatch {m0}/{m1}/{h}')
    return observed


def boundary_oracles(f4, keys, thresholds):
    """Neutral and perfectly accurate synthetic edge cases; no real observations."""
    y = np.zeros((N_DATES,N_STOCKS,3),dtype=np.int8)
    y[:,10:,:] = 1
    y = y.reshape(N,3)
    for name in ('ALWAYS_UP_NULL','PERFECT_SEPARATION'):
        if name=='ALWAYS_UP_NULL':
            arr=np.full((N,3),.999,dtype=np.float64)
            exp_ba,exp_p,exp_diff = .5,1.0,0.0
        else:
            arr=np.where(y==1,.999,.001).astype(np.float64)
            exp_ba,exp_p,exp_diff = 1.0,1.0/(REPS+1),0.0
        ensemble={m:arr.copy() for m in MODELS}
        results,pairs=f4.evaluate_in_memory(keys,y,ensemble,thresholds)
        need(len(results)==72 and len(pairs)==84,'Boundary output sizes drift')
        need(np.allclose(results.observed_BA.to_numpy(),exp_ba,atol=1e-14),
             f'{name}: balanced accuracy oracle failed')
        need(np.allclose(results.p_one_sided_centered.to_numpy(),exp_p,atol=1e-14),
             f'{name}: one-sided p-value oracle failed')
        need(np.allclose(results.ci_lower_percentile_95,exp_ba) and
             np.allclose(results.ci_upper_percentile_95,exp_ba),
             f'{name}: percentile CI oracle failed')
        need(np.allclose(pairs.BA_difference_a_minus_b,exp_diff) and
             np.allclose(pairs.ci_lower_percentile_95,exp_diff) and
             np.allclose(pairs.ci_upper_percentile_95,exp_diff),
             f'{name}: pairwise CI oracle failed')
        need(np.allclose(results[results.primary_test].q_BH_primary_only,exp_p),
             f'{name}: FDR oracle failed')
    print('PASS — independent exact null (BA=.50,p=1) and perfect-classifier (BA=1,p=1/2001) oracles')


def check_cli_gate():
    proc = subprocess.run([sys.executable,str(F4),'--read-test'],
                          capture_output=True,text=True,timeout=30,cwd=str(BASE))
    need(proc.returncode != 0 and 'DENIED' in (proc.stdout+proc.stderr),
         'F4 prohibited CLI did not fail closed')
    src=F4.read_text(encoding='utf-8')
    tree=ast.parse(src,filename=str(F4))
    # Inspect reviewed callable to ensure it is not quietly wired to real-data IO.
    funcs={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    for req in ('stratified_session_indices','evaluate_in_memory','bh_adjust','preflight','no_real_gate'):
        need(req in funcs, f'F4 required function missing: {req}')
    eval_text=ast.get_source_segment(src,funcs['evaluate_in_memory'])
    for disallowed in ('read_parquet(', 'read_csv(', 'np.load(', 'torch.load(', '.open(', 'cuda(', 'open_test('):
        need(disallowed not in eval_text,
             f'F4 pure statistical interface has unexpected I/O/GPU: {disallowed}')
    print('PASS — F4 statistical interface remains in-memory; prohibited --read-test CLI rejected')


def run(wsl):
    print('='*106)
    print('STEP 36I-F5 — INDEPENDENT SEGMENT-STRATIFIED STATISTICAL AUDIT, SYNTHETIC ONLY')
    print('='*106)
    print('[1] SOURCE IDENTITY, FROZEN CONTRACT, HISTORICAL TEST ACCESS')
    for name,path in [('E2',E2),('F1',F1),('F2',F2),('F3',F3),('F4',F4)]:
        verified(path,HASH[name],name)
    with zipfile.ZipFile(E2,'r') as archive:
        need(archive.testzip() is None,'E2 archive integrity failure')
        needed=['candidate_keys_DRAFT.csv','frozen_thresholds.csv','step35_contract.json']
        raw={name:archive.read(name) for name in needed}
    for filename,identity in [('candidate_keys_DRAFT.csv','COHORT'),
                              ('frozen_thresholds.csv','THRESHOLDS'),
                              ('step35_contract.json','STEP35')]:
        need(hashlib.sha256(raw[filename]).hexdigest()==HASH[identity],
             f'E2 contract identity drift: {filename}')
    contract=json.loads(raw['step35_contract.json'])
    d=contract['dependence_aware_inference_contract']
    checks={'primary_block_length_sessions':20,'bootstrap_rng_seed':20261013,
            'replications_per_block_length':2000,
            'sensitivity_block_lengths_sessions':[10,20,40],
            'primary_null_hypothesis':'balanced_accuracy <= 0.50',
            'primary_alternative':'balanced_accuracy > 0.50'}
    for k,v in checks.items(): need(d.get(k)==v,f'Frozen Step35 statistical drift: {k}')
    before=None
    if wsl:
        for name,path in [('STEP35',STEP35),('COHORT',CANDIDATE),
                          ('THRESHOLDS',THRESHOLDS),('E4',E4_REPORT)]:
            verified(path,HASH[name],name)
        verified(ACCESS,sha(ACCESS),'Historical read ledger')
        before=sha(ACCESS)
        marker=json.loads(ACCESS.read_text(encoding='utf-8'))
        need(marker.get('raw_test_filesystem_open_count_completed')==2 and
             marker.get('final_test_scored') is False and
             marker.get('step35_manifest_sha256')==HASH['STEP35'],
             'Historical test-access state changed')
    print('PASS — F1–F4, E2 and Step35 immutable; two historical reads, zero scored' if wsl
          else 'PASS — F1–F4, E2 and Step35 frozen source signatures')
    f4=load_read_only(F4,'step36i_f4_synthetic_inference')
    f1=load_read_only(F1,'step36i_f1_synthetic_scoring')
    keys=f4.key_frame(pd.read_csv(io.BytesIO(raw['candidate_keys_DRAFT.csv'])))
    thresholds=f4.threshold_map(pd.read_csv(io.BytesIO(raw['frozen_thresholds.csv'])))
    need(len(keys)==5300 and len(thresholds)==24,'F5 fixture alignment drift')
    print('[2] INDEPENDENT CIRCULAR DATE-BLOCK REPLICATION ORACLE')
    for L in BLOCKS:
        idx,_=f4.stratified_session_indices(L)
        ref=independent_indices(L)
        need(idx.shape==(REPS,N_DATES) and np.array_equal(idx,ref),
             f'Independent RNG/segment-index mismatch for L={L}')
        need(np.all(ref[:,:48]<48) and np.all(ref[:,48:]>=48),
             f'Segment crossing for L={L}')
    print('PASS — 2,000 replicate session indices reproduced independently at L=10/20/40; no segment crossing')
    print('[3] INDEPENDENT 24-HYPOTHESIS / 72-ROW METRIC AND 84-PAIRWISE ORACLE')
    y,seeds=f4.build_synthetic_fixture(keys,thresholds)
    ensemble,f1_metrics=f1.in_memory_five_seed_scoring(seeds,y,thresholds)
    observed, pairwise=f4.evaluate_in_memory(keys,y,ensemble,thresholds)
    need(len(f1_metrics)==24 and len(observed)==72 and len(pairwise)==84,
         'F4/F1 cardinality mismatch')
    reference_engine(keys,y,ensemble,thresholds,observed,pairwise)
    print('PASS — all 72 BA/percentile-CI/centered-p oracles; 24 BH q-values; 84 pairwise CI/difference oracles')
    print('[4] NONTRIVIAL EDGE CASES, GATING, AND ACCESS-MARKER IMMUTABILITY')
    boundary_oracles(f4,keys,thresholds)
    check_cli_gate()
    if wsl:
        need(sha(ACCESS)==before,'Original access journal mutated by F5')
        print('PASS — historical access marker unchanged')
    print('='*106)
    print('STEP 36I-F5: PASS — INDEPENDENT SYNTHETIC STATISTICAL REVIEW ONLY')
    print('Test opened: FALSE | GPU: FALSE | Real scoring: FALSE | Amended cohort approved: FALSE')
    print('Third raw-test read authorized: FALSE | Production execution: DENIED')
    print('NEXT — review production access/transaction integration and formal protocol deviation')
    print('='*106)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test',action='store_true')
    group.add_argument('--offline-preflight',action='store_true')
    a=p.parse_args()
    run(wsl=a.offline_preflight)

if __name__=='__main__':
    try: main()
    except Exception as ex:
        print(f'STEP 36I-F5 FAIL-CLOSED: {type(ex).__name__}: {ex}',file=sys.stderr)
        raise SystemExit(1)
