#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F18: reusable two-phase immutable artifact store, REVIEW ONLY.

F17 provides algorithms for (A) converting already-loaded raw inputs to arrays
and (B) computing 40 frozen checkpoint outputs. F18 adds real reusable Python
functions to atomically seal Phase A and EACH individual checkpoint's output,
then rehash, validate, enumerate and resume strictly from those saved objects.

Only --offline-preflight or --portable-self-test is callable from this CLI.
Neither CLI command reads a final-test parquet, uses the GPU, or touches any
production output. Its exercise uses a temporary SYNTHETIC_ONLY directory.

No real access-control journal, release authority, GPU worker or live launcher
is implemented here. Calling library functions with real bytes requires a
SEPARATELY independently reviewed production orchestrator and an explicit
third-raw-read authorization before any sealed-test access. Do not use this
module itself as a production command.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
ACCESS = EXP / 'manifests/36_final_test_access_consumption_state.json'
STEP35 = EXP / 'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
APPROVAL = ROOT / 'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json'
F17 = ROOT / 'step36i_f17_two_phase_release_implementation_REVIEW_ONLY.py'
F11 = ROOT / 'step36i_f11_upstream_provenance_audit_READ_ONLY.py'
F15 = ROOT / 'step36i_f15_durable_materialization_handoff_REVIEW_ONLY.py'
F16 = ROOT / 'step36i_f16_two_phase_resume_independent_REVIEW_ONLY.py'
EXPECTED = {
    'F17': (F17, 'f865db97463535e093c9f4dcf4ff478e9ecbc50a0d6752961200f9aad98cf068'),
    'F11': (F11, '3405cb13a89d7c2403a66b5f6119b18a13c42f886d7f8d05aa80e94b09709ea6'),
    'F15': (F15, 'a2e2a05ef363f389c5208dceb4eef518227f050a8294d03326f962767fc8289c'),
    'F16': (F16, '938d7e08774fcd7b61f3e4dc3fdd5865925ef8a234df6b3538c388784f1f7382'),
    'approval': (APPROVAL, 'f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493'),
    'Step35': (STEP35, '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7')
}
MODELS = ('DLINEAR','NLINEAR','PATCHTST','TIMESNET','ITRANSFORMER','FREDFORMER','FRNET','MLF')
SEEDS = (42,43,44,45,46)
FROZEN_BATCH = {'DLINEAR':128,'NLINEAR':128,'PATCHTST':64,'TIMESNET':64,
                'ITRANSFORMER':128,'FREDFORMER':16,'FRNET':64,'MLF':32}
SHAPE_X = (5300,60,63)
SHAPE_Y = (5300,3)
EXPECTED_PHASE_A = {
    'arrays/X_5300x60x63.npy', 'arrays/labels_5300x3.npy',
    'arrays/metadata.parquet', 'arrays/ordered_keys.csv', 'arrays/phase_a_manifest.json',
}
HEX = re.compile(r'^[0-9a-f]{64}$')
DENY_ARGS = {'--read-test','--run','--execute','--production','--phase-a','--phase-b',
             '--authorize','--commit-real','--resume','--gpu','--open-test','--seal-real'}
if DENY_ARGS.intersection(sys.argv[1:]):
    raise SystemExit('DENIED: F18 does not have a live launcher or test-data permission')


def require(ok, why):
    if not ok: raise RuntimeError(why)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1<<20), b''): h.update(block)
    return h.hexdigest()


def verify_source(path: Path, expected: str, label: str):
    require(path.is_file() and not path.is_symlink(), f'Unsafe/missing {label}')
    require(sha_file(path)==expected, f'Immutable source digest drift: {label}')


def syncdir(p: Path):
    fd=os.open(p, os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
    try: os.fsync(fd)
    finally: os.close(fd)


def create_exclusive(p: Path, data: bytes):
    require(isinstance(data,bytes), 'Only byte payloads allowed')
    flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os,'O_NOFOLLOW',0)
    fd=os.open(p,flags,0o600)
    try:
        mv=memoryview(data)
        pos=0
        while pos<len(mv):
            n=os.write(fd,mv[pos:pos+(1<<20)])
            require(n>0,'Short write')
            pos+=n
        os.fsync(fd)
    finally: os.close(fd)


def safe_relative(raw: str) -> PurePosixPath:
    require(isinstance(raw,str) and raw and '\\' not in raw and not raw.startswith('/'),
            'Invalid relative artifact path')
    path=PurePosixPath(raw)
    require(all(part not in ('','..','.') for part in path.parts) and path.as_posix()==raw,
            'Path traversal/noncanonical component')
    require(len(path.parts)<=3 and all(part not in ('.','..') for part in path.parts),
            'Unsafe artifact path')
    return path


def rename_noreplace(src: Path, dst: Path):
    require(src.parent==dst.parent and not dst.exists() and not dst.is_symlink(),
            'No-overwrite condition failed')
    lib=ctypes.CDLL(None, use_errno=True)
    fn=getattr(lib,'renameat2',None)
    require(fn is not None, 'Linux renameat2 unavailable; abort')
    fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    fn.restype=ctypes.c_int
    rc=fn(-100,os.fsencode(src),-100,os.fsencode(dst),1)
    if rc!=0:
        e=ctypes.get_errno()
        raise OSError(e,os.strerror(e),str(dst))
    syncdir(dst.parent)


def ensure_clean_parent(base: Path):
    require(base.is_dir() and not base.is_symlink(),'Invalid transaction parent')
    require(not any(base.glob('.F18_STAGING_*')), 'Abandoned staging requires manual review')


def atomic_package(base: Path, name: str, payloads: dict[str,bytes],
                   phase: str, context: dict, inject_failure: bool=False) -> Path:
    """Publish one immutable directory with content manifest, never replace.

    Reusable library API; F18 CLI calls this ONLY on synthetic temporary fixtures.
    The production release must separately authorize caller/input/output root.
    """
    require(phase in ('PHASE_A','SEED'), 'Phase not recognized')
    require(re.fullmatch(r'(phase_a_materialization|seed_(?:DLINEAR|NLINEAR|PATCHTST|TIMESNET|ITRANSFORMER|FREDFORMER|FRNET|MLF)_(?:42|43|44|45|46))',name) is not None,
            'Invalid package name')
    ensure_clean_parent(base)
    require(not (base/name).exists() and not (base/name).is_symlink(),'Already committed; never overwrite')
    require(isinstance(context,dict) and context.get('step35_sha256')==EXPECTED['Step35'][1],
            'Original scientific contract binding absent')
    require(context.get('amendment_sha256')==EXPECTED['approval'][1],
            'Amendment binding absent')
    require(payloads and len(payloads)==len(set(payloads)),'Payload map invalid')
    if phase=='PHASE_A':
        require(name=='phase_a_materialization' and set(payloads)==EXPECTED_PHASE_A,
                'Phase A payload membership drift')
    else:
        require(set(payloads)=={'seed_outputs.npz'},'Seed payload membership drift')
        require(context.get('phase_a_package_sha256') and HEX.fullmatch(context['phase_a_package_sha256']),
                'No verified Phase A provenance')
    entries=[]
    for raw,b in sorted(payloads.items()):
        p=safe_relative(raw)
        require(isinstance(b,bytes),'Nonbyte payload')
        entries.append({'relative_path':p.as_posix(),'bytes':len(b),'sha256':digest(b)})
    manifest={'schema':'STEP36I_F18_IMMUTABLE_ARTIFACT_V1',
              'phase':phase,'name':name,'context':context,'artifacts':entries}
    stage=base/('.F18_STAGING_'+uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    try:
        made={stage}
        for record in entries:
            p=stage/record['relative_path']
            dirs=[]; v=p.parent
            while v not in made:
                dirs.append(v);v=v.parent
            for d in reversed(dirs):
                d.mkdir(mode=0o700)
                made.add(d)
            create_exclusive(p,payloads[record['relative_path']])
        create_exclusive(stage/'package_manifest.json',canonical(manifest))
        # Sync nested dirs bottom-up; no file is exposed until atomic rename.
        for d in sorted(made,key=lambda item:len(item.parts),reverse=True):
            syncdir(d)
        if inject_failure:
            raise RuntimeError('SYNTHETIC_INJECTED_PRECOMMIT_CRASH')
        rename_noreplace(stage,base/name)
    except Exception:
        # Retain incomplete stage as evidence; never silently auto-clean it.
        raise
    verified=verify_package(base/name)
    require(verified==manifest,'Postcommit manifest drift')
    return base/name


def verify_package(pkg: Path) -> dict:
    require(pkg.is_dir() and not pkg.is_symlink(),'Package missing or symlink')
    mp=pkg/'package_manifest.json'
    require(mp.is_file() and not mp.is_symlink(),'Manifest unsafe')
    manifest=json.loads(mp.read_bytes())
    require(manifest.get('schema')=='STEP36I_F18_IMMUTABLE_ARTIFACT_V1',
            'Wrong manifest schema')
    require(manifest.get('name')==pkg.name and manifest.get('phase') in ('PHASE_A','SEED'),
            'Wrong package identity')
    require(manifest.get('context',{}).get('step35_sha256')==EXPECTED['Step35'][1],
            'Scientific contract drift')
    require(manifest.get('context',{}).get('amendment_sha256')==EXPECTED['approval'][1],
            'Amendment drift')
    entries=manifest['artifacts']
    require(len(entries)>=1 and len({e['relative_path'] for e in entries})==len(entries),
            'Duplicate/empty inventory')
    expected_files={'package_manifest.json'}
    for e in entries:
        name=safe_relative(e['relative_path']).as_posix()
        expected_files.add(name)
        f=pkg/name
        require(f.is_file() and not f.is_symlink(),f'Unsafe/missing file: {name}')
        require(f.stat().st_size==e['bytes'] and sha_file(f)==e['sha256'],
                f'Hash/size drift: {name}')
    observed={str(f.relative_to(pkg).as_posix()) for f in pkg.rglob('*') if f.is_file()}
    require(observed==expected_files,'Unexpected or missing files in package')
    require(not any(x.is_symlink() for x in pkg.rglob('*')),'Symlink in package')
    return manifest


def phase_a_manifest_digest(phase_a_dir: Path) -> str:
    p=verify_package(phase_a_dir)
    require(p['phase']=='PHASE_A' and set(e['relative_path'] for e in p['artifacts'])==EXPECTED_PHASE_A,
            'Phase A package incomplete')
    return sha_file(phase_a_dir/'package_manifest.json')


def commit_seed_arrays(base:Path, model:str, seed:int, logits:np.ndarray, probabilities:np.ndarray,
                       checkpoint_sha256:str, physical_batch:int, phase_a_sha:str):
    require(model in MODELS and seed in SEEDS,'Unrecognized frozen model/seed')
    require(HEX.fullmatch(checkpoint_sha256) is not None and HEX.fullmatch(phase_a_sha) is not None,
            'Checkpoint or Phase A digest malformed')
    require(int(physical_batch)==FROZEN_BATCH[model],
            'Physical inference batch disagrees with F1 frozen contract')
    for x in (logits,probabilities):
        require(isinstance(x,np.ndarray) and x.shape==(5300,3) and
                x.dtype==np.float32 and np.isfinite(x).all(), 'Invalid seed arrays')
    require(((probabilities>=0)&(probabilities<=1)).all(),'Invalid seed probabilities')
    # Numerically verify probability is sigmoid(logit), independent of checkpoint runner.
    sigmoid=1.0/(1.0+np.exp(-np.clip(logits.astype(np.float64),-100.,100.)))
    require(np.allclose(sigmoid,probabilities,atol=1e-6,rtol=1e-6),
            'Logits and probabilities inconsistent')
    b=io.BytesIO();np.savez_compressed(b,logits=logits,probabilities=probabilities)
    context={'step35_sha256':EXPECTED['Step35'][1],
             'amendment_sha256':EXPECTED['approval'][1],
             'phase_a_package_sha256':phase_a_sha,
             'checkpoint_sha256':checkpoint_sha256,
             'physical_inference_microbatch':int(physical_batch),
             'model_id':model,'seed':int(seed),'n':5300,'horizons':[1,3,5]}
    return atomic_package(base,f'seed_{model}_{seed}',{'seed_outputs.npz':b.getvalue()},'SEED',context)


def read_seed(base:Path, model:str, seed:int, phase_a_sha:str) -> np.ndarray:
    pkg=base/f'seed_{model}_{seed}'
    m=verify_package(pkg)
    require(m['phase']=='SEED' and m['context']['model_id']==model and
            m['context']['seed']==seed and
            m['context']['phase_a_package_sha256']==phase_a_sha,
            'Seed identity/Phase A mismatch')
    require(set(e['relative_path'] for e in m['artifacts'])=={'seed_outputs.npz'},
            'Seed inventory drift')
    with np.load(io.BytesIO((pkg/'seed_outputs.npz').read_bytes()),allow_pickle=False) as npz:
        require(set(npz.files)=={'logits','probabilities'},'Seed NPZ keys drift')
        l=npz['logits'];p=npz['probabilities']
    require(l.shape==p.shape==(5300,3) and l.dtype==p.dtype==np.float32,
            'Seed matrix structure drift')
    require(np.isfinite(l).all() and np.isfinite(p).all() and
            ((p>=0)&(p<=1)).all(),'Seed matrix numerical drift')
    require(np.allclose(1.0/(1.0+np.exp(-np.clip(l.astype(np.float64),-100.,100.))),
                        p,atol=1e-6,rtol=1e-6), 'Seed numeric oracle failed')
    return p


def resume_inventory(base:Path, phase_a_sha:str):
    ensure_clean_parent(base)
    all_possible={(m,s) for m in MODELS for s in SEEDS}
    done=set()
    for p in base.iterdir():
        if p.name=='phase_a_materialization':
            continue
        require(p.is_dir() and not p.is_symlink(), 'Unexpected file in seed namespace')
        mm=re.fullmatch(r'seed_([A-Z]+)_(4[2-6])',p.name)
        require(mm is not None, 'Unknown seed package in namespace')
        key=(mm.group(1),int(mm.group(2)))
        require(key in all_possible and key not in done,'Invalid/duplicated seed package')
        read_seed(base,*key,phase_a_sha)
        done.add(key)
    missing=sorted(all_possible-done)
    return done,missing


def portable_synthetic_test():
    # Does not re-run GPU or bootstrap. Exercises REAL reusable storage API.
    with tempfile.TemporaryDirectory(prefix='STEP36I_F18_SYNTHETIC_ONLY_') as tmp:
        t=Path(tmp)
        pha=t/'phase_a';pha.mkdir()
        sample={'arrays/X_5300x60x63.npy':b'SYNTHETIC_ONLY_X',
                'arrays/labels_5300x3.npy':b'SYNTHETIC_ONLY_Y',
                'arrays/metadata.parquet':b'SYNTHETIC_ONLY_METADATA',
                'arrays/ordered_keys.csv':b'SYNTHETIC_ONLY_KEYS',
                'arrays/phase_a_manifest.json':b'SYNTHETIC_ONLY_F17_MANIFEST'}
        context={'step35_sha256':EXPECTED['Step35'][1],
                 'amendment_sha256':EXPECTED['approval'][1],
                 'synthetic_only':True}
        out=atomic_package(pha,'phase_a_materialization',sample,'PHASE_A',context)
        assert phase_a_manifest_digest(out)==sha_file(out/'package_manifest.json')
        try: atomic_package(pha,'phase_a_materialization',sample,'PHASE_A',context)
        except RuntimeError: pass
        else: raise RuntimeError('Duplicate Phase A commit accepted')
        print('PASS — Phase A atomic manifest, rehash and duplicate rejection')
        store=t/'seed_results';store.mkdir()
        n=np.tile(np.asarray([[0.2,-0.4,1.2]],dtype=np.float32),(5300,1))
        p=(1.0/(1.0+np.exp(-n.astype(np.float64)))).astype(np.float32)
        a=phase_a_manifest_digest(out)
        commit_seed_arrays(store,'DLINEAR',42,n,p,'a'*64,128,a)
        resumed=read_seed(store,'DLINEAR',42,a)
        require(np.array_equal(resumed,p),'Saved seed differs')
        done,missing=resume_inventory(store,a)
        require(done=={('DLINEAR',42)} and len(missing)==39,'Resume list drift')
        try: commit_seed_arrays(store,'DLINEAR',42,n,p,'a'*64,128,a)
        except RuntimeError: pass
        else: raise RuntimeError('Repeated seed commit accepted')
        try: commit_seed_arrays(store,'NLINEAR',43,n,p,'a'*64,16,a)
        except RuntimeError: pass
        else: raise RuntimeError('Incorrect frozen physical batch accepted')
        try: read_seed(store,'DLINEAR',42,'b'*64)
        except RuntimeError: pass
        else: raise RuntimeError('Wrong Phase A identity accepted')
        print('PASS — independent seed rehash, numerical oracle, 1 completed/39 pending, duplicate rejection')
        broken=store/'seed_DLINEAR_42'/'seed_outputs.npz'
        with broken.open('ab') as f: f.write(b'X')
        try: resume_inventory(store,a)
        except RuntimeError: pass
        else: raise RuntimeError('Corrupted saved seed accepted')
        print('PASS — modified committed output blocks resume')
    with tempfile.TemporaryDirectory(prefix='STEP36I_F18_SYNTHETIC_CRASH_') as tmp:
        b=Path(tmp)
        try: atomic_package(b,'phase_a_materialization',sample,'PHASE_A',context,True)
        except RuntimeError as exc:
            require('SYNTHETIC_INJECTED' in str(exc),'Wrong simulated crash')
        else: raise RuntimeError('Injected crash not triggered')
        require(not (b/'phase_a_materialization').exists(),'Crash published incomplete target')
        try: atomic_package(b,'phase_a_materialization',sample,'PHASE_A',context)
        except RuntimeError: pass
        else: raise RuntimeError('Unreviewed crash-state replay accepted')
        print('PASS — failed precommit staging blocks silent replay')


def wsl_readonly_checks():
    for label,(path,sha) in EXPECTED.items(): verify_source(path,sha,label)
    doc=json.loads(APPROVAL.read_text('utf-8'))
    require(doc['record_status']=='APPROVED_BY_STUDY_OWNER_FOR_IMPLEMENTATION_AND_INDEPENDENT_REVIEW',
            'Study-owner approval drift')
    require(doc['remaining_review_gates']['additional_raw_test_read_authorized'] is False,
            'F13E must not be reinterpreted as access permission')
    marker_bytes=ACCESS.read_bytes()
    marker=json.loads(marker_bytes)
    require(marker.get('raw_test_filesystem_open_count_completed')==2 and
            marker.get('final_test_scored') is False,'Historical access record drift')
    require(ACCESS.read_bytes()==marker_bytes,'Historical marker changed')
    print('PASS — F17/F11/F15/F16, Step35, F13E source hashes; ledger 2 reads/0 scored')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--offline-preflight',action='store_true')
    group.add_argument('--portable-self-test',action='store_true')
    args=parser.parse_args()
    print('='*104)
    print('STEP 36I-F18 — IMMUTABLE PHASE A + PER-SEED RESUME STORE (REVIEW ONLY)')
    print('='*104)
    if args.offline_preflight:
        print('[1] LIVE NON-TEST PROVENANCE AND ACCESS DENIAL')
        wsl_readonly_checks()
    else:
        print('[1] PORTABLE SOURCE TEST; LIVE PROVENANCE DEFERRED')
    print('[2] REAL REUSABLE IMMUTABLE STORAGE METHODS ON SMALL SYNTHETIC FIXTURES')
    portable_synthetic_test()
    if args.offline_preflight:
        require(json.loads(ACCESS.read_bytes())['raw_test_filesystem_open_count_completed']==2,
                'Historical read count mutated')
    print('[3] REMAINING RELEASE GATES')
    print('NOT IMPLEMENTED — authorized single-open real loader/journal/writer integration')
    print('NOT IMPLEMENTED — live checkpoint worker commits and approved Phase B orchestration')
    print('PENDING — independent source and methodological sign-off; separate third-read authorization')
    print('='*104)
    print('STEP 36I-F18 PASS — REVIEWABLE PHASE A AND PER-SEED IMMUTABLE STORAGE ONLY')
    print('Real raw test opened: FALSE | GPU: FALSE | Test scored: FALSE | Production CLI: DENIED')
    print('='*104)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'F18 FAIL-CLOSED — {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
