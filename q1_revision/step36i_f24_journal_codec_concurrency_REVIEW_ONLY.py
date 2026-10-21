#!/usr/bin/env python3
"""STEP 36I-F24 -- source-bound journal format and concurrent-release safety contract.

REVIEW ONLY, NO REAL ACCESS. This module supplies a strict reader for the
existing F22 JSON-Lines journal (which has a .json filename under the F1 frozen
output mapping) and a cooperative POSIX directory-lock context manager.

The existing F23 production candidate does not yet call this module. The
module does NOT authorize or expose a real launcher, sealed-test read, GPU
inference, original access marker mutation, or live results commit.

Run --offline-preflight from q1_revision in WSL; it reads only source files,
manifest metadata and the historical access marker. Synthetic journal fixtures
are created exclusively in a temporary directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
HISTORY = EXP / 'manifests/36_final_test_access_consumption_state.json'
F23 = ROOT / 'step36i_f23_live_release_controller_CANDIDATE_REVIEW_ONLY.py'
F22 = ROOT / 'step36i_f22_access_bound_phase_a_controller_REVIEW_ONLY.py'
F20 = ROOT / 'step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py'
F1 = ROOT / 'step36i_f1_amended_production_runner_REVIEW_ONLY.py'
SHA = {
    'F23':'11497c21fda381112ca65012540d3bc2bc7522030ba0bc577d9808c0c812d1b7',
    'F22':'7ae182b83a20d2f621ded09ceb7e5e696f1b9c41b0f814a0ffd05c4d79997b26',
    'F20':'937de4fb1dc7a58f6ecb8449ad55a515a5dcb28120adbeaa82a4e31c0af5d018',
    'F1':'e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556',
}
FROZEN_MAP_SHA='bbd66ccc6de6232575329c8da5f7e0238814c1049290098e21b44aab36d586bb'
DENIED={'--execute','--production','--read-test','--live','--authorize','--phase-a','--phase-b','--resume','--gpu','--grant'}
if DENIED.intersection(sys.argv[1:]):
    raise SystemExit('DENIED -- F24 is offline review only; no third-test-read authority')


def require(ok:bool, reason:str)->None:
    if not ok:
        raise RuntimeError(reason)


def canonical(obj:dict)->bytes:
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def sha_bytes(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path:Path)->str:
    require(path.is_file() and not path.is_symlink(),f'Missing or unsafe read-only source {path}')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def verify_sources()->None:
    for name,path in [('F23',F23),('F22',F22),('F20',F20),('F1',F1)]:
        require(sha_file(path)==SHA[name],f'{name} source fingerprint mismatch')


def load_module(path:Path,alias:str):
    spec=importlib.util.spec_from_file_location(alias,path)
    require(spec is not None and spec.loader is not None, 'Module load failure')
    module=importlib.util.module_from_spec(spec)
    sys.modules[alias]=module
    spec.loader.exec_module(module)
    return module


def strict_jsonl_journal(data:bytes, *, authority_sha256:str, require_complete:bool=True)->list[dict]:
    """Independently validate F22's *JSON Lines* format, not a JSON document.

    No filesystem paths or test bytes are opened. This parser explicitly
    rejects extra events, truncation, content drift and forged chains.
    A hash chain detects accidental tampering; it is not a digital signature.
    """
    require(len(authority_sha256)==64 and
            all(ch in '0123456789abcdef' for ch in authority_sha256),
            'Bad expected authorization digest')
    require(data and data.endswith(b'\n'), 'Truncated/empty journal')
    lines=data.splitlines()
    require(len(lines) in ((2,) if require_complete else (1,2)),
            'Incorrect journal event count')
    prev='0'*64
    rows=[]
    allowed={'index','event','prev','authority_sha256','source_sha256','bytes','event_sha256'}
    event_names=('RESERVED_BEFORE_OPEN','SOURCE_BYTES_VERIFIED')
    for ix,line in enumerate(lines):
        require(line and len(line)<10000,'Empty/excessively long journal record')
        try:
            row=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError) as e:
            raise RuntimeError('Invalid JSON Lines record') from e
        require(type(row) is dict and set(row)==allowed,'Unexpected journal record keys')
        for name in ('prev','authority_sha256','event_sha256'):
            require(type(row[name]) is str and len(row[name])==64 and
                    all(ch in '0123456789abcdef' for ch in row[name]),
                    f'Invalid {name}')
        require(type(row['index']) is int and row['index']==ix and
                row['event']==event_names[ix] and row['prev']==prev and
                row['authority_sha256']==authority_sha256,
                'Journal event order / prior hash / authority mismatch')
        base={k:v for k,v in row.items() if k!='event_sha256'}
        require(row['event_sha256']==sha_bytes(canonical(base)),
                'Journal event content/hash mismatch')
        if ix==0:
            require(row['source_sha256'] is None and row['bytes'] is None,
                    'Reservation cannot claim source bytes')
        else:
            h=row['source_sha256']
            require(type(h) is str and len(h)==64 and
                    all(ch in '0123456789abcdef' for ch in h),
                    'Invalid sealed-source hash')
            require(type(row['bytes']) is int and row['bytes']>0,
                    'Invalid sealed-source byte count')
        rows.append(row)
        prev=row['event_sha256']
    return rows


@contextmanager
def cooperative_directory_lock(directory:Path):
    """POSIX cooperative lock; future controller must hold across reservation+seal.

    This makes no new live output pathname. It is not yet integrated in F23.
    Other code not participating in the flock is not prevented from writing.
    """
    require(directory.is_dir() and not directory.is_symlink(),
            'Unsafe journal parent directory')
    fd=os.open(directory,os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_CLOEXEC',0))
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield fd
    finally:
        # Releasing an unsuccessful acquisition is not needed; close the fd.
        try:
            fcntl.flock(fd,fcntl.LOCK_UN)
        finally:
            os.close(fd)


def expect_rejected(action,reason:str)->None:
    try:
        action()
    except (RuntimeError,ValueError,OSError,BlockingIOError):
        return
    raise RuntimeError('Negative control incorrectly accepted: '+reason)


def portable_fixture()->None:
    verify_sources()
    f22=load_module(F22,'step36i_f24_signed_f22')
    f23=load_module(F23,'step36i_f24_signed_f23')
    require(f23.LIVE_RELEASE_ENABLED is False,'F23 live gate was promoted')
    # Portable test: the absolute WSL paths do not exist in this container.
    # Verify the internal mapping only; exact frozen WSL digest is checked
    # separately in --offline-preflight.
    require(len(f23.ISOLATED)==7 and f23.mapping_digest(),
            'F23 output mapping invalid')
    require(f22.ReservedReadJournal.EVENTS==('RESERVED_BEFORE_OPEN','SOURCE_BYTES_VERIFIED'),
            'F22 journal event interface changed')
    with tempfile.TemporaryDirectory(prefix='F24_SYNTHETIC_ONLY_') as root:
        d=Path(root)
        authority=sha_bytes(b'SYNTHETIC-F24-ONLY-AUTHORITY')
        source=sha_bytes(b'SYNTHETIC-F24-ONLY-SOURCE')
        target=d/'SYNTHETIC_ONLY_access_journal.json'  # Frozen .json suffix, JSONL CONTENT
        journal=f22.ReservedReadJournal(target,authority)
        with cooperative_directory_lock(d):
            # A second independent descriptor cannot acquire an exclusive lock.
            contender=os.open(d,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
            try:
                expect_rejected(lambda:fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB),
                                'Concurrent release transaction lock')
            finally:
                os.close(contender)
            journal.reserve()
            reservation=target.read_bytes()
            strict_jsonl_journal(reservation,authority_sha256=authority,require_complete=False)
            expect_rejected(lambda:strict_jsonl_journal(reservation,authority_sha256=authority),
                            'Incomplete access journal accepted as verified read')
            journal.verified(source,17)
        data=target.read_bytes()
        independent=strict_jsonl_journal(data,authority_sha256=authority)
        require(independent==journal.entries() and
                sha_bytes(data)==journal.final_digest(),
                'F22 journal and independent codec disagree')
        # A JSON parser fails after second JSON line; consumers MUST use JSONL.
        expect_rejected(lambda:json.loads(data),'Two event JSONL wrongly accepted as JSON')
        expect_rejected(lambda:strict_jsonl_journal(data[:-1],authority_sha256=authority),
                        'Truncated journal')
        expect_rejected(lambda:strict_jsonl_journal(data,authority_sha256='0'*64),
                        'Wrong authority')
        expect_rejected(lambda:strict_jsonl_journal(data+data.splitlines()[-1]+b'\n',
                                                        authority_sha256=authority),
                        'Duplicate verified event')
        tampered=bytearray(data)
        at=tampered.index(b'RESERVED_BEFORE_OPEN')
        tampered[at]=ord('X')
        expect_rejected(lambda:strict_jsonl_journal(bytes(tampered),authority_sha256=authority),
                        'Tampered journal event')
        expect_rejected(journal.reserve,'Duplicate reservation')
        expect_rejected(lambda:journal.verified(source,17),'Duplicate source verification')
        # The original F1 journal output path is not renamed or created.
        require(target.suffix=='.json','Fixture journal name was changed')


def offline_preflight()->None:
    print('='*106)
    print('STEP 36I-F24 -- JSONL JOURNAL CODEC AND COOPERATIVE CONCURRENCY CONTRACT')
    print('='*106)
    print('[1] F23/F22/F20/F1 SOURCE HASHES AND FROZEN F1 OUTPUT MAPPING')
    verify_sources()
    f23=load_module(F23,'step36i_f24_source_f23')
    f1=load_module(F1,'step36i_f24_source_f1')
    require(f23.mapping_digest()==FROZEN_MAP_SHA and
            {k:str(v) for k,v in f23.ISOLATED.items()}==
            {k:str(v) for k,v in f1.ISOLATED.items()},
            'Frozen output mapping mismatch')
    require(f23.LIVE_RELEASE_ENABLED is False,'F23 real execution gate promoted')
    print('PASS -- four sources and 7 existing reserved output paths unchanged')
    print('[2] F22 TWO-EVENT JOURNAL FORMAT AND CONCURRENT RELEASE CONTRACT')
    portable_fixture()
    print('PASS -- independently validated F22 two-event JSON Lines, hash chain, and full/partial states')
    print('PASS -- exclusive POSIX directory-lock fixture and replay/tamper controls')
    print('FORMAT NOTE -- frozen .json path has application/x-ndjson content; plain json.load FAILS')
    print('[3] HISTORICAL ACCESS AND LIVE OUTPUT IMMUTABILITY')
    require(HISTORY.is_file() and not HISTORY.is_symlink(),'Historical access state missing')
    before=HISTORY.read_bytes()
    state=json.loads(before)
    require(state.get('raw_test_filesystem_open_count_completed')==2 and
            state.get('final_test_scored') is False,'Unexpected test access/scoring history')
    for name,path in f23.ISOLATED.items():
        require(not path.exists() and not path.is_symlink(),
                f'Live output already occupied: {name}')
    require(HISTORY.read_bytes()==before,'Historical ledger changed')
    print('PASS -- original marker 2 reads/0 scored; seven amended output paths vacant')
    print('[4] RELEASE STATUS -- EXPLICIT NON-IMPLEMENTED INTEGRATION')
    print('OPEN -- F23 must CALL this journal codec and hold a concurrency lock across Phase A')
    print('OPEN -- independently review and bind real Phase A, Phase B and output mapping')
    print('OPEN -- independent methodological signoff and a DISTINCT third-read permit')
    print('='*106)
    print('F24 PASS -- SOURCE-BOUND JSONL/CONCURRENCY CONTRACT, SYNTHETIC ONLY')
    print('Sealed test opened: FALSE | GPU: FALSE | Final scoring: FALSE')
    print('Production CLI: DENIED | Third read authorized: FALSE')
    print('='*106)


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--portable-self-test',action='store_true')
    g.add_argument('--offline-preflight',action='store_true')
    args=p.parse_args()
    if args.offline_preflight:
        offline_preflight()
    else:
        print('STEP 36I-F24 -- PORTABLE SYNTHETIC JOURNAL / LOCK TEST')
        portable_fixture()
        print('PASS -- source provenance, JSONL codec, locking and tamper/replay controls')
        print('Real raw test read: FALSE')

if __name__=='__main__':
    try:
        main()
    except (RuntimeError,PermissionError,OSError,ValueError,KeyError,AssertionError) as e:
        print('F24 FAIL CLOSED:',type(e).__name__,str(e),file=sys.stderr)
        raise SystemExit(1)
