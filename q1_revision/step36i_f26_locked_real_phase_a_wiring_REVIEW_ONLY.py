#!/usr/bin/env python3
"""STEP 36I-F26 — REAL Phase A *source wiring* under frozen gates, REVIEW ONLY.

This is NOT a production launcher, permission record, external methodological
approval, or authorization for another held-out-test read. ONLY the read-only
--offline-preflight CLI is accessible. All live operations are blocked with a
hard-coded False gate before any read/creation.

For independent review of a later, separately authorized source version, this
candidate documents a REAL (not synthetic stand-in) algorithm joining:
  F23 upstream authentication/third-read record validation/one-FD loader;
  F24 exclusive directory lock + strict JSONL event validator;
  F20 genuine Phase A materializer on the same verified bytes;
  F18 atomic Phase A package (called by F20), and no-overwrite JSON receipt.

The historical marker remains immutable and contains the first two reads. The
future third attempt, if separately authorized, needs a DISTINCT append-only
journal and external signed independent review. A hash-chained JSONL record is
not by itself a digital signature. Cooperative flock cannot stop uncooperative
writers; exclusive release ownership must be reviewed before production.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
SOURCES = {
    'F1': ('step36i_f1_amended_production_runner_REVIEW_ONLY.py','e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556'),
    'F18': ('step36i_f18_durable_two_phase_artifact_store_REVIEW_ONLY.py','c4706fb360233473078f5f4aac3865c9525535ef6cd4e225b258d1a728037f69'),
    'F20': ('step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py','937de4fb1dc7a58f6ecb8449ad55a515a5dcb28120adbeaa82a4e31c0af5d018'),
    'F22': ('step36i_f22_access_bound_phase_a_controller_REVIEW_ONLY.py','7ae182b83a20d2f621ded09ceb7e5e696f1b9c41b0f814a0ffd05c4d79997b26'),
    'F23': ('step36i_f23_live_release_controller_CANDIDATE_REVIEW_ONLY.py','11497c21fda381112ca65012540d3bc2bc7522030ba0bc577d9808c0c812d1b7'),
    'F24': ('step36i_f24_journal_codec_concurrency_REVIEW_ONLY.py','c91d8c8cf224d3f465960cb552dcfb908504a850b1b968c7d68de3b0cdb0c487'),
    'F25': ('step36i_f25_journal_locked_phase_a_integration_REVIEW_ONLY.py','e633d00ff1a9532d7c8eda17ccde86071bcee375faa522dcbd3f5a7ff19bf27e'),
}
SHA35='6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7'
SHA_APPROVAL='f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493'
SHA_MAP='bbd66ccc6de6232575329c8da5f7e0238814c1049290098e21b44aab36d586bb'
STEP35=EXP/'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
APPROVAL=ROOT/'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json'
HISTORY=EXP/'manifests/36_final_test_access_consumption_state.json'
LIVE_RELEASE_ENABLED=False     # Non-negotiable in this review file.
DENY={'--execute','--production','--read-test','--live','--phase-a','--phase-b',
      '--authorize','--grant','--gpu','--resume','--commit-real','--run','--approve'}
if DENY.intersection(sys.argv[1:]):
    raise SystemExit('DENIED: F26 is a source-review candidate, not a live release')


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def sha_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe source: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda:src.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def load(name: str):
    filename, expected=SOURCES[name]
    source=ROOT/filename
    require(sha_file(source)==expected,'Frozen source changed: '+name)
    spec=importlib.util.spec_from_file_location('f26_verified_'+name.lower(),source)
    require(spec is not None and spec.loader is not None,'Verified module unavailable: '+name)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def checked_contract():
    for key,(filename,wanted) in SOURCES.items():
        require(sha_file(ROOT/filename)==wanted,key+' source hash drift')
    require(sha_file(STEP35)==SHA35,'Step35 modified')
    require(sha_file(APPROVAL)==SHA_APPROVAL,'F13E owner record modified')
    before=HISTORY.read_bytes()
    state=json.loads(before)
    require(state.get('raw_test_filesystem_open_count_completed')==2 and
            state.get('final_test_scored') is False,'Unexpected historical access state')
    return before


def proposed_genuine_phase_a(*, source: Path, review_records,
                             signed_permit_verified_externally: bool=False):
    """Full source-level composition, deliberately NOT executable in this version.

    IMPORTANT: any later version enabling this function must independently
    authenticate records and the release source, resolve concurrent-writer
    ownership and upstream snapshot/TOCTOU limitations, and obtain a separate
    explicit authorization. Do not call this function on real data here.
    """
    if not LIVE_RELEASE_ENABLED or not signed_permit_verified_externally:
        raise PermissionError('LIVE PHASE A DENIED — independent sign-off and separate authorization absent')

    # Inaccessible candidate body below is for independent source review ONLY.
    f23=load('F23');f24=load('F24');f22=load('F22');f20=load('F20');f18=load('F18')
    with f24.cooperative_directory_lock(EXP/'manifests'):
        historical=f23.get_history()
        f23.check_new_output_paths_empty()
        upstream,bound=f23.pre_read_upstream_snapshot(f20)
        auth_sha=review_records.inspect(intended_source_path=source,
              upstream_digest=f23.digest_bytes(f23.canonical(bound)))
        require(f23.mapping_digest()==SHA_MAP,'Output mapping changed')
        f23.recheck_upstream(bound)
        require(HISTORY.read_bytes()==historical,'Historical access marker modified')

        paths=f23.ISOLATED
        target=paths['array_dir']
        require(not target.exists() and not target.is_symlink(),
                'Phase A destination already occupied')
        require(target.parent.is_dir() and not target.parent.is_symlink(),
                'Missing Phase A parent directory')

        # Creation before reservation consumes the one-attempt output namespace
        # upon subsequent failure. NEVER remove it or silently start again.
        target.mkdir(mode=0o700)
        f22.fsync_dir(target.parent)
        journal=f22.ReservedReadJournal(paths['independent_access_journal'],auth_sha)
        raw_bytes,journal_sha=f23.journalized_single_read(
            source,approved_source_sha256=f23.SHA['Test'],journal=journal,
            expected_history=historical)
        rows=f24.strict_jsonl_journal(journal.path.read_bytes(),
                     authority_sha256=auth_sha,require_complete=True)
        require(len(rows)==2 and journal_sha==sha_file(journal.path),
                'Completed journal invalid before Phase A materialization')
        f23.recheck_upstream(bound)
        package_sha=f20.seal_preloaded_phase_a(
            raw_verified_bytes=raw_bytes,authorized_journal_digest=journal_sha,
            output_parent=target,upstream=upstream)
        f23.recheck_upstream(bound)
        package=target/'phase_a_materialization'
        require(f20.inspect_phase_a(package)[-1]==package_sha,
                'Committed Phase A authentication failed')
        receipt={
          'stage':'F26_PROPOSED_PHASE_A_RELEASE',
          'step35_sha256':SHA35,'amendment_approval_sha256':SHA_APPROVAL,
          'historical_reads_before_attempt':2,'additional_read_attempts_reserved':1,
          'journal_format':'application/x-ndjson',
          'access_journal_sha256':journal_sha,
          'phase_a_package_sha256':package_sha,
          'frozen_output_mapping_sha256':SHA_MAP,
          'authorization_binding_sha256':auth_sha,
          'real_model_inference_performed':False,
          'held_out_metrics_computed':False,
        }
        # Receipt is a distinct immutable no-replace transaction. A failed
        # staging write is preserved as evidence and blocks automatic retry.
        # This still needs independent approval for actual output mapping.
        data=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
        stage=paths['array_manifest'].with_name('.F26_RECEIPT_STAGING')
        require(not stage.exists() and not stage.is_symlink(),
                'Abandoned receipt staging: manual review required')
        require(not paths['array_manifest'].exists(), 'Receipt already present')
        f18.create_exclusive(stage,data)
        f18.rename_noreplace(stage,paths['array_manifest'])
        require(sha_file(paths['array_manifest'])==hashlib.sha256(data).hexdigest(),
                'Receipt commit cannot be independently verified')
        require(HISTORY.read_bytes()==historical,'Original access marker changed')
        return receipt


def offline_preflight():
    print('='*100)
    print('STEP 36I-F26 — GENUINE PHASE A SOURCE-WIRING REVIEW (NO TEST ACCESS)')
    print('='*100)
    print('[1] SOURCE IDENTITIES, ORIGINAL CONTRACT, OWNER APPROVAL, TWO-READ HISTORY')
    original=checked_contract()
    print('PASS — F1/F18/F20/F22/F23/F24/F25, Step35 and owner records unchanged; historical 2/0')
    print('[2] API INTEGRATION AND EXPLICIT DENIAL OF LIVE PHASE A')
    f20,f23,f24,f22=(load(k) for k in ('F20','F23','F24','F22'))
    require(f23.mapping_digest()==SHA_MAP and len(f23.ISOLATED)==7,'Frozen seven-path mapping mismatch')
    require(all(not p.exists() and not p.is_symlink() for p in f23.ISOLATED.values()),
            'Release namespace already occupied')
    require({'raw_verified_bytes','authorized_journal_digest','output_parent','upstream'} <=
            set(inspect.signature(f20.seal_preloaded_phase_a).parameters),
            'F20 genuine materializer interface mismatch')
    require(callable(f23.journalized_single_read) and callable(f24.strict_jsonl_journal)
           and callable(f24.cooperative_directory_lock) and
           callable(f22.ReservedReadJournal),'Journal/lock/reader contracts absent')
    try:
        proposed_genuine_phase_a(source=ROOT/'NONEXISTENT_DO_NOT_OPEN_TEST.parquet',
                  review_records=None,signed_permit_verified_externally=False)
    except PermissionError:
        pass
    else:
        raise RuntimeError('Real Phase A permission bypassed')
    print('PASS — F20 genuine Phase A API, F23 one-FD read, F24 codec and lock linked in denied candidate')
    print('PASS — live execution rejected before any filesystem access')
    print('[3] SOURCE-LEVEL SEQUENCING REVIEW')
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and
            x.name=='proposed_genuine_phase_a')
    calls=[ast.unparse(x.func) for x in ast.walk(fn) if isinstance(x,ast.Call)]
    for label in ('f24.cooperative_directory_lock','review_records.inspect',
                  'f23.journalized_single_read','f24.strict_jsonl_journal',
                  'f20.seal_preloaded_phase_a','f20.inspect_phase_a'):
        require(label in calls,'Missing actual source call: '+label)
    print('PASS — authorization, source reader, validated journal, genuine materializer and seal verification present')
    print('[4] UNRESOLVED RELEASE BOUNDARIES')
    require(HISTORY.read_bytes()==original,'Access history changed')
    print('OPEN — independent review of actual release source and receipt/no-overwrite integration')
    print('OPEN — strong external record authentication, upstream TOCTOU and cross-process ownership')
    print('OPEN — approved Phase B controller and distinct third-read permit')
    print('='*100)
    print('F26 PASS — SOURCE-WIRING CONTRACT ONLY; NO LIVE IMPLEMENTATION/GRANT')
    print('Sealed test opened: FALSE | GPU: FALSE | Test scored: FALSE | Production CLI: DENIED')
    print('='*100)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline-preflight',action='store_true',required=True)
    args=p.parse_args()
    require(args.offline_preflight,'Only offline review permitted')
    offline_preflight()

if __name__=='__main__':
    try: main()
    except (RuntimeError,ValueError,KeyError,OSError,AssertionError) as e:
        print('F26 FAIL CLOSED:',type(e).__name__,str(e),file=sys.stderr)
        raise SystemExit(1)
