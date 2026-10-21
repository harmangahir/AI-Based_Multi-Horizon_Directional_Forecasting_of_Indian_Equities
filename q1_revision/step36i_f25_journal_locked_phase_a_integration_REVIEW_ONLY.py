#!/usr/bin/env python3
"""STEP 36I-F25 — source-bound journal-locked Phase A integration, REVIEW ONLY.

This verifies a real composition of the frozen F22 journal/source-reader, F24
strict JSONL codec and POSIX directory lock, and F18 atomic no-replace
persistence. Source bindings and F23's exact output map are verified.

The only exposed execution paths create TEMPORARY SYNTHETIC fixtures. Neither
this CLI nor its functions can open the sealed held-out test. It never writes to
an experiment output path, changes the historical access marker, runs GPUs, or
creates a third-read permit. The F20 real Phase A computational API is checked
as a dependency, NOT invoked with actual test bytes.

F25 is NOT independent methodological sign-off or a production release.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
HISTORY = EXP / 'manifests/36_final_test_access_consumption_state.json'
STEP35 = EXP / 'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
APPROVAL = ROOT / 'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json'
SOURCE = {
    'F1': ('step36i_f1_amended_production_runner_REVIEW_ONLY.py', 'e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556'),
    'F18': ('step36i_f18_durable_two_phase_artifact_store_REVIEW_ONLY.py', 'c4706fb360233473078f5f4aac3865c9525535ef6cd4e225b258d1a728037f69'),
    'F20': ('step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py', '937de4fb1dc7a58f6ecb8449ad55a515a5dcb28120adbeaa82a4e31c0af5d018'),
    'F22': ('step36i_f22_access_bound_phase_a_controller_REVIEW_ONLY.py', '7ae182b83a20d2f621ded09ceb7e5e696f1b9c41b0f814a0ffd05c4d79997b26'),
    'F23': ('step36i_f23_live_release_controller_CANDIDATE_REVIEW_ONLY.py', '11497c21fda381112ca65012540d3bc2bc7522030ba0bc577d9808c0c812d1b7'),
    'F24': ('step36i_f24_journal_codec_concurrency_REVIEW_ONLY.py', 'c91d8c8cf224d3f465960cb552dcfb908504a850b1b968c7d68de3b0cdb0c487'),
}
SHA35 = '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7'
SHA_APPROVAL = 'f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493'
SHA_MAPPING = 'bbd66ccc6de6232575329c8da5f7e0238814c1049290098e21b44aab36d586bb'
SEALED_SHA = 'ab325ede85fbb6d7819e85472274b82d151d350f39a872a17747c6978b1dddae'
DENIED = {'--read-test','--execute','--production','--live','--phase-a','--phase-b',
          '--resume','--authorize','--grant','--run','--gpu','--final-test','--commit-real'}
if DENIED.intersection(sys.argv[1:]):
    raise SystemExit('DENIED: F25 is a review-only synthetic test; real execution is unavailable')


def require(condition, msg):
    if not condition:
        raise RuntimeError(msg)


def sha_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(),f'Missing or unsafe input: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1<<20), b''):
            h.update(part)
    return h.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_sources():
    for label, (filename, expected) in SOURCE.items():
        path=ROOT / filename
        require(sha_file(path)==expected, f'{label} source SHA mismatch')
    return True


def load(name):
    filename=SOURCE[name][0]
    path=ROOT/filename
    alias='f25_verified_'+name.lower()
    spec=importlib.util.spec_from_file_location(alias,path)
    require(spec is not None and spec.loader is not None,f'Import failed: {name}')
    obj=importlib.util.module_from_spec(spec)
    sys.modules[alias]=obj
    spec.loader.exec_module(obj)
    return obj


def expect_reject(action, message):
    try:
        action()
    except (RuntimeError, PermissionError, BlockingIOError, OSError, ValueError):
        return
    raise RuntimeError('Negative control was ACCEPTED: '+message)


def verify_apis(mod, *, wsl_live:bool=False):
    f1,f18,f20,f22,f23,f24=(mod[n] for n in ('F1','F18','F20','F22','F23','F24'))
    require(not f23.LIVE_RELEASE_ENABLED, 'F23 live gate must remain disabled')
    require(bool(f23.mapping_digest()), 'F23 output mapping digest missing')
    if wsl_live:
        require(f23.mapping_digest()==SHA_MAPPING, 'WSL frozen output mapping digest changed')
    require({k:str(v) for k,v in f23.ISOLATED.items()} ==
            {k:str(v) for k,v in f1.ISOLATED.items()} and len(f23.ISOLATED)==7,
            'F23/F1 isolated output mapping drift')
    require({'raw_verified_bytes','authorized_journal_digest','output_parent','upstream'} <=
            set(inspect.signature(f20.seal_preloaded_phase_a).parameters),
            'Actual F20 Phase A consumer signature changed')
    require({'fixture','authority','journal','consumer'} <=
            set(inspect.signature(f22.fixture_single_read_and_consume).parameters),
            'F22 read-once adapter signature changed')
    require(all(callable(x) for x in (
        f24.cooperative_directory_lock, f24.strict_jsonl_journal,
        f18.atomic_package, f18.verify_package, f23.ReleaseRecords.inspect)),
        'Frozen journal, transaction or permit interface missing')
    return True


def strict_journal_check(*,f24, journal_path:Path, authority_hash:str, complete:bool):
    data=journal_path.read_bytes()
    rows=f24.strict_jsonl_journal(data,authority_sha256=authority_hash,
                                   require_complete=complete)
    require((len(rows)==2)==complete, 'Journal completion state mismatch')
    return sha_bytes(data)


def synthetic_locked_phase_a(*, fixture:Path, authority, f22, f24, f18,
                              package_parent:Path, inject_before_open:bool=False):
    """Real F22+F24+F18 integration, restricted to a synthetic temporary source.

    Strictly requires fixture and journal in the same synthetic temporary root.
    The lock is held through pre-read reservation, verified read, codec check,
    and atomic publication. F18 payloads are deliberately small mock files;
    they are NOT genuine 5300-row model arrays or a test-data materialization.
    """
    require(fixture.name.startswith('SYNTHETIC_ONLY_F25_') and
            fixture.parent == package_parent and
            package_parent.name.startswith('F25_SYNTHETIC_ONLY_') and
            package_parent.is_dir() and not package_parent.is_symlink(),
            'F25 fixture path cannot represent the real experiment')
    require(authority.source_sha256 != SEALED_SHA,'Synthetic fixture cannot use the sealed test hash')
    journal_path=package_parent/'SYNTHETIC_ONLY_F25_access_journal.json'
    journal=f22.ReservedReadJournal(journal_path,authority.digest())
    calls=[]

    def consumer(raw:bytes,journal_hash:str,provenance:dict):
        require(isinstance(raw,bytes) and sha_bytes(raw)==authority.source_sha256,
                'Reader/consumer digest binding mismatch')
        require(journal_hash==strict_journal_check(
            f24=f24,journal_path=journal_path,authority_hash=authority.digest(),complete=True),
            'Independent JSONL journal verification failed')
        require(journal_hash==provenance['read_journal_sha256'] and
                provenance['verified_source_bytes']==len(raw),
                'F22 provenance values do not match consumed bytes')
        calls.append(1)
        context={
            'step35_sha256': SHA35,
            'amendment_sha256': SHA_APPROVAL,
            'synthetic_only': True,
            'fixture_sha256': sha_bytes(raw),
            'authorization_sha256': authority.digest(),
            'journal_sha256': journal_hash,
            'real_test_data': False,
        }
        payloads={name:b'SYNTHETIC_ONLY_F25_PAYLOAD_'+name.encode()
                  for name in sorted(f18.EXPECTED_PHASE_A)}
        pkg=f18.atomic_package(package_parent,'phase_a_materialization',
                                payloads,'PHASE_A',context)
        manifest=f18.verify_package(pkg)
        require(manifest['context']==context and
                len(manifest['artifacts'])==5,'Atomic synthetic Phase A provenance drift')
        return pkg

    with f24.cooperative_directory_lock(package_parent):
        # Verify cooperative directory lock excludes a second FD during the
        # entire read/consume/commit transaction.
        fd=os.open(package_parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:
            expect_reject(lambda:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB),
                          'Concurrent Phase A controller admitted')
        finally:
            os.close(fd)
        try:
            result=f22.fixture_single_read_and_consume(
                fixture,authority,journal,consumer,
                inject_crash_before_open=inject_before_open)
        finally:
            require(journal_path.is_file(), 'Durable reservation missing after attempt')
            strict_journal_check(f24=f24,journal_path=journal_path,
                                 authority_hash=authority.digest(),
                                 complete=not inject_before_open)
    if inject_before_open:
        raise RuntimeError('Expected injected crash but reader returned')
    require(len(calls)==1, 'Synthetic consumer must run exactly once')
    require(result.is_dir(), 'Phase A package missing after completed synthetic transaction')
    return result, journal


def synthetic_tests():
    checked_sources()
    mod={name:load(name) for name in SOURCE}
    verify_apis(mod)
    f22,f24,f18=(mod[x] for x in ('F22','F24','F18'))
    with tempfile.TemporaryDirectory(prefix='F25_SYNTHETIC_ONLY_') as p:
        root=Path(p)
        payload=(b'SYNTHETIC_ONLY_F25_BYTES:' + bytes(range(256)))*6
        fixture=root/'SYNTHETIC_ONLY_F25_fixture.bin'
        fixture.write_bytes(payload)
        authority=f22.SyntheticAuthorization(
            source_sha256=sha_bytes(payload),source_basename=fixture.name,
            upstream_sha256=sha_bytes(b'SYNTHETIC_ONLY_UPSTREAM'),
            owner_approval_sha256=SHA_APPROVAL,step35_sha256=SHA35)
        pkg,journal=synthetic_locked_phase_a(
            fixture=fixture,authority=authority,f22=f22,f24=f24,f18=f18,
            package_parent=root)
        require(f18.verify_package(pkg)['phase']=='PHASE_A','Phase A seal invalid')
        require(strict_journal_check(f24=f24,journal_path=journal.path,
                  authority_hash=authority.digest(),complete=True)==journal.final_digest(),
                'F22/F24 journal digest mismatch')
        expect_reject(lambda:synthetic_locked_phase_a(
            fixture=fixture,authority=authority,f22=f22,f24=f24,f18=f18,
            package_parent=root), 'A second attempt at completed Phase A')
        expect_reject(lambda:f18.atomic_package(
            root,'phase_a_materialization',{'x':b'y'},'PHASE_A',{
            'step35_sha256':SHA35,'amendment_sha256':SHA_APPROVAL}),
            'Overwrite of committed Phase A')
        # One-bit committed artifact corruption must invalidate reuse.
        target=pkg/'arrays/ordered_keys.csv'
        original=target.read_bytes()
        target.write_bytes(original+b'X')
        expect_reject(lambda:f18.verify_package(pkg), 'Corrupt materialization reused')
    with tempfile.TemporaryDirectory(prefix='F25_SYNTHETIC_ONLY_') as p:
        root=Path(p)
        fixture=root/'SYNTHETIC_ONLY_F25_crash.bin'
        data=b'SYNTHETIC_ONLY_F25_CRASH'
        fixture.write_bytes(data)
        auth=f22.SyntheticAuthorization(
            source_sha256=sha_bytes(data),source_basename=fixture.name,
            upstream_sha256=sha_bytes(b'SYNTHETIC_ONLY_UPSTREAM'),
            owner_approval_sha256=SHA_APPROVAL,step35_sha256=SHA35)
        expect_reject(lambda:synthetic_locked_phase_a(
            fixture=fixture,authority=auth,f22=f22,f24=f24,f18=f18,
            package_parent=root, inject_before_open=True),
            'Injected crash after irreversible reservation')
        journal=root/'SYNTHETIC_ONLY_F25_access_journal.json'
        strict_journal_check(f24=f24,journal_path=journal,
                             authority_hash=auth.digest(),complete=False)
        require(not (root/'phase_a_materialization').exists(),
                'Crash unexpectedly committed output')
        expect_reject(lambda:synthetic_locked_phase_a(
            fixture=fixture,authority=auth,f22=f22,f24=f24,f18=f18,
            package_parent=root), 'Restart after reservation-only crash')
    # F23's live execution candidate must remain explicitly unreachable.
    expect_reject(lambda:mod['F23'].live_phase_a_candidate(
        source=ROOT/'DO_NOT_OPEN_REAL_TEST.parquet',
        records=mod['F23'].ReleaseRecords({},{}),
        independently_reviewed_source=False),
        'F23 production gate bypass')
    return True


def offline_preflight():
    print('='*106)
    print('STEP 36I-F25 — JOURNAL-LOCKED PHASE A INTEGRATION (REVIEW ONLY)')
    print('='*106)
    print('[1] REVIEWED F1/F18/F20/F22/F23/F24 SOURCES AND UPSTREAM BOUNDARIES')
    checked_sources()
    require(sha_file(STEP35)==SHA35 and sha_file(APPROVAL)==SHA_APPROVAL,
            'Step35 or study-owner amendment digest mismatch')
    mod={name:load(name) for name in SOURCE}
    verify_apis(mod, wsl_live=True)
    print('PASS — source SHA-256, source interfaces, frozen Step35 and owner amendment')
    print('PASS — F23/F1 original seven-path output map:',SHA_MAPPING)
    print('[2] F22 SOURCE -> F24 JSONL CODEC/EXCLUSIVE LOCK -> F18 ATOMIC COMMIT')
    synthetic_tests()
    print('PASS — same verified fixture bytes and journal consumed by atomic Phase A adapter')
    print('PASS — lock held across reservation/read/codec/commit; independent rehash')
    print('PASS — duplicate/replay, interruption, journal/asset corruption and concurrent lock rejected')
    print('[3] LIVE WSL ACCESS STATE AND OUTPUT IMMUTABILITY')
    require(HISTORY.is_file() and not HISTORY.is_symlink(),'Historical marker missing')
    original=HISTORY.read_bytes()
    state=json.loads(original)
    require(state.get('raw_test_filesystem_open_count_completed')==2 and
            state.get('final_test_scored') is False,'Unexpected historical access state')
    for name,path in mod['F23'].ISOLATED.items():
        require(not path.exists() and not path.is_symlink(),
                f'Blocked: previously occupied output destination {name}')
    require(HISTORY.read_bytes()==original,'Historical access marker modified')
    print('PASS — historical marker 2 reads/0 scored; 7 live paths vacant/untouched')
    print('[4] OPEN RELEASE ISSUES / NO FALSE APPROVAL')
    print('OPEN — wire separately authorized raw-test loader and F20 genuine data materializer under this lock')
    print('OPEN — independent source and methodological sign-off; real output path/format decision')
    print('OPEN — independently verified distinct third-read permit and safe final Phase B controller')
    print('='*106)
    print('STEP 36I-F25 PASS — SOURCE-BOUND TEMPORARY SYNTHETIC INTEGRATION ONLY')
    print('Real raw test opened: FALSE | GPU: FALSE | Test scored: FALSE')
    print('Third read authorized: FALSE | Production CLI: DENIED')
    print('='*106)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--offline-preflight',action='store_true')
    group.add_argument('--portable-self-test',action='store_true')
    a=parser.parse_args()
    if a.offline_preflight:
        offline_preflight()
    else:
        print('STEP 36I-F25 — PORTABLE SYNTHETIC LOCKED PHASE A INTEGRATION')
        synthetic_tests()
        print('PASS — real inputs and outputs remain unreachable')


if __name__=='__main__':
    try:
        main()
    except (RuntimeError,PermissionError,ValueError,OSError,KeyError,AssertionError) as e:
        print('STEP 36I-F25 FAIL CLOSED:',type(e).__name__,str(e),file=sys.stderr)
        raise SystemExit(1)
