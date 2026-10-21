#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F29 — single-investigator accountable release CANDIDATE.

Production orchestration is implemented (Phase A -> per-seed Phase B -> finalization),
but NO real action is permitted without TWO DISTINCT owner-signed decisions
and an owner keyring pinned after out-of-band fingerprint confirmation.

F29 does NOT itself grant methodological self-signoff or a third raw-test read. Its release
operations are deliberately unavailable with the current project records.

Usage NOW: python step36i_f29_single_investigator_release_CANDIDATE.py --offline-preflight

FUTURE separately authorized interface (do NOT invoke before release review):
  --phase-a / --phase-b / --finalize --trusted-keys ... --keyring-sha256 ...
  --method-attestation ... --read-permit ...

Security boundary: cooperative flock plus private owner-only directories; no
protection against privileged or uncooperative processes under the same UID.
Third read is single-fd; a private durable snapshot of its bytes is retained
for explicitly reviewed recovery, never silently consumed or retried.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import uuid
from typing import Callable

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
SOURCE_FILES = {
    'F1':('step36i_f1_amended_production_runner_REVIEW_ONLY.py','e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556'),
    'F18':('step36i_f18_durable_two_phase_artifact_store_REVIEW_ONLY.py','c4706fb360233473078f5f4aac3865c9525535ef6cd4e225b258d1a728037f69'),
    'F20':('step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py','937de4fb1dc7a58f6ecb8449ad55a515a5dcb28120adbeaa82a4e31c0af5d018'),
    'F22':('step36i_f22_access_bound_phase_a_controller_REVIEW_ONLY.py','7ae182b83a20d2f621ded09ceb7e5e696f1b9c41b0f814a0ffd05c4d79997b26'),
    'F23':('step36i_f23_live_release_controller_CANDIDATE_REVIEW_ONLY.py','11497c21fda381112ca65012540d3bc2bc7522030ba0bc577d9808c0c812d1b7'),
    'F24':('step36i_f24_journal_codec_concurrency_REVIEW_ONLY.py','c91d8c8cf224d3f465960cb552dcfb908504a850b1b968c7d68de3b0cdb0c487'),
    'F26':('step36i_f26_locked_real_phase_a_wiring_REVIEW_ONLY.py','4814ea8e23f93333389946c2ddd0f412af813e4fa82b84e94bb79e539e3c01a9'),
}
SHA35='6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7'
OWNER_SHA='f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493'
MAPPING_SHA='bbd66ccc6de6232575329c8da5f7e0238814c1049290098e21b44aab36d586bb'
TEST_SHA='ab325ede85fbb6d7819e85472274b82d151d350f39a872a17747c6978b1dddae'
# Must be pinned only after the sole researcher confirms their own PUBLIC key
# fingerprint out-of-band and the updated source is audited.
# None means NO genuine live source access can occur through this candidate.
OWNER_PINNED_KEYRING_SHA256=None
HISTORY=EXP/'manifests/36_final_test_access_consumption_state.json'
STEP35=EXP/'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
APPROVAL=ROOT/'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path:Path)->str:
    require(path.is_file() and not path.is_symlink(),f'Missing/unsafe file {path}')
    h=hashlib.sha256()
    with path.open('rb') as inp:
        for chunk in iter(lambda:inp.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(record:object)->bytes:
    return json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def check(path:Path,want:str,name:str):
    require(sha_file(path)==want,f'Immutable {name} SHA-256 mismatch')


def reviewed_module(key):
    filename,expected=SOURCE_FILES[key]
    path=ROOT/filename
    check(path,expected,key)
    tag='f28_verified_'+key.lower()
    spec=importlib.util.spec_from_file_location(tag,path)
    require(spec is not None and spec.loader is not None,f'{key} import unavailable')
    mod=importlib.util.module_from_spec(spec)
    sys.modules[tag]=mod
    spec.loader.exec_module(mod)
    return mod


def historic_state()->bytes:
    require(HISTORY.is_file() and not HISTORY.is_symlink(),'Legacy access marker missing')
    raw=HISTORY.read_bytes()
    record=json.loads(raw)
    require(record.get('raw_test_filesystem_open_count_completed')==2 and
            record.get('final_test_scored') is False,'Historical marker drift')
    return raw


def authentic_core():
    for key,(filename,expected) in SOURCE_FILES.items():
        check(ROOT/filename,expected,key)
    check(STEP35,SHA35,'Step35')
    check(APPROVAL,OWNER_SHA,'F13E study owner approval')
    return historic_state()


def check_private_dir(directory:Path):
    s=directory.lstat()
    require(stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode),f'Unsafe directory {directory}')
    require(s.st_uid==os.geteuid(),f'Non-owned directory {directory}')
    require(not (stat.S_IMODE(s.st_mode)&0o022),f'Group/other writable directory {directory}')


def check_critical_environment():
    # Filesystem ownership is a defense in depth, NOT a replacement for genuine independent permission.
    for directory in (ROOT,EXP,EXP/'manifests',EXP/'feature_data',EXP/'audits'):
        check_private_dir(directory)
    require(EXP.stat().st_dev==(EXP/'manifests').stat().st_dev,
            'Unexpected filesystem change for journal location')


def verify_owner_envelope(path:Path, pub:bytes, expected_kind:str)->dict:
    """Verify an externally supplied, owner-signed envelope; never generate it."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    require(path.is_file() and not path.is_symlink(),f'Missing owner-signed {expected_kind}')
    obj=json.loads(path.read_bytes())
    require(set(obj)=={'payload','signature_b64'} and isinstance(obj['payload'],dict),
            'Invalid owner-signed envelope')
    require(obj['payload'].get('record_type')==expected_kind,
            'Wrong owner-signed record type')
    signature=base64.b64decode(obj['signature_b64'],validate=True)
    require(len(signature)==64,'Invalid Ed25519 signature length')
    try:
        Ed25519PublicKey.from_public_bytes(pub).verify(signature,canonical(obj['payload']))
    except InvalidSignature as exc:
        raise RuntimeError('Owner signature invalid') from exc
    return obj['payload']


class SingleInvestigatorReleaseRecords:
    """Self-review is explicitly NOT independent review.

    The separate read permit is a separate owner decision, even if signed with
    the same long-lived owner key. This preserves auditability, not independence.
    """
    def __init__(self,method:dict,permit:dict):
        self.method=method
        self.permit=permit

    def inspect(self,*,intended_source_path:Path,upstream_digest:str)->str:
        method,permit=self.method,self.permit
        require(method.get('record_type')=='SINGLE_INVESTIGATOR_METHOD_SELF_REVIEW' and
                method.get('decision')=='APPROVED_WITH_LIMITATIONS_FOR_AMENDED_EVALUATION' and
                method.get('review_type')=='SELF_REVIEW_NOT_INDEPENDENT' and
                method.get('limitations_acknowledged') is True and
                method.get('step35_sha256')==SHA35 and
                method.get('study_owner_approval_sha256')==OWNER_SHA and
                method.get('approved_candidate_keys_sha256')==
                    '4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65',
                'Signed method self-review insufficient or wrong')
        require(permit.get('record_type')=='SEPARATE_THIRD_RAW_TEST_READ_PERMIT' and
                permit.get('decision')=='EXPLICIT_SINGLE_ADDITIONAL_RAW_READ_AUTHORIZED' and
                permit.get('historical_reads_before_release')==2 and
                permit.get('test_sha256')==TEST_SHA and
                permit.get('method_record_sha256')==digest(canonical(method)) and
                permit.get('step35_sha256')==SHA35 and
                permit.get('study_owner_approval_sha256')==OWNER_SHA and
                permit.get('output_mapping_sha256')==MAPPING_SHA and
                permit.get('upstream_binding_sha256')==upstream_digest and
                permit.get('source_path')==str(intended_source_path) and
                permit.get('acknowledge_two_historical_reads') is True and
                permit.get('allow_durable_raw_snapshot_for_recovery') is True,
                'Signed separate read permit insufficient or wrong')
        require(intended_source_path.is_absolute() and
                not intended_source_path.is_symlink() and
                intended_source_path.parent.resolve()==intended_source_path.parent,
                'Noncanonical or unsafe signed source path')
        return digest(canonical({'method':method,'permit':permit}))


def validate_signatures(args, f23)->tuple[object,str,dict,dict]:
    """Authenticate TWO DISTINCT owner decisions from an out-of-band pinned key.

    Owner-signed methodological self-review is NOT independent methodological
    review. Without a genuinely pinned key and a separately signed THIRD READ
    permit, every live execution operation is denied.
    """
    require(OWNER_PINNED_KEYRING_SHA256 is not None,
            'DENIED: no owner keyring trust anchor pinned in this reviewed source')
    supplied=(args.trusted_keys,args.keyring_sha256,
              args.method_attestation,args.read_permit)
    require(all(supplied),'Owner keyring, self-review, and separate permit are required')
    require(args.keyring_sha256==OWNER_PINNED_KEYRING_SHA256,
            'CLI keyring SHA differs from the reviewed pinned owner trust anchor')
    keyring_path=Path(args.trusted_keys)
    check(keyring_path,OWNER_PINNED_KEYRING_SHA256,'pinned owner keyring')
    ring=json.loads(keyring_path.read_bytes())
    require(ring.get('schema')=='F29_SINGLE_INVESTIGATOR_TRUST_V1' and
            ring.get('review_model')=='SOLE_RESEARCHER_SELF_REVIEW_NOT_INDEPENDENT' and
            isinstance(ring.get('owner_identity'),str) and bool(ring['owner_identity']),
            'Owner keyring identity/policy incorrect')
    pub=base64.b64decode(ring['owner_public_key_b64'],validate=True)
    require(len(pub)==32,'Invalid pinned owner Ed25519 public key')
    method=verify_owner_envelope(Path(args.method_attestation),pub,
                                 'SINGLE_INVESTIGATOR_METHOD_SELF_REVIEW')
    permit=verify_owner_envelope(Path(args.read_permit),pub,
                                 'SEPARATE_THIRD_RAW_TEST_READ_PERMIT')
    identity=ring['owner_identity']
    require(method.get('owner_identity')==identity and
            permit.get('owner_identity')==identity,
            'Owner-signed record identity mismatch')
    source_hash=sha_file(Path(__file__))
    require(method.get('f29_source_sha256')==source_hash and
            permit.get('f29_source_sha256')==source_hash,
            'Signed decisions do not bind this precise release code')
    require(method.get('review_type')=='SELF_REVIEW_NOT_INDEPENDENT' and
            method.get('decision')=='APPROVED_WITH_LIMITATIONS_FOR_AMENDED_EVALUATION' and
            method.get('limitations_acknowledged') is True,
            'Owner methodological self-review limitations not acknowledged')
    require(permit.get('decision')=='EXPLICIT_SINGLE_ADDITIONAL_RAW_READ_AUTHORIZED' and
            permit.get('method_record_sha256')==digest(canonical(method)) and
            permit.get('historical_reads_before_release')==2 and
            permit.get('permit_scope')=='ONE_RAW_READ_WITH_DURABLE_PHASE_A_THEN_40_FROZEN_SEEDS' and
            permit.get('acknowledge_two_historical_reads') is True and
            permit.get('allow_durable_raw_snapshot_for_recovery') is True,
            'Separate third-read permission not sufficiently specific')
    # Separate authorization must occur AFTER the method decision. Neither
    # trusted timestamps nor genuine human independence can be inferred solely
    # from local signed records; this is only an accountable owner workflow.
    from datetime import datetime
    try:
        t_m=datetime.fromisoformat(method['decision_time_utc'].replace('Z','+00:00'))
        t_p=datetime.fromisoformat(permit['decision_time_utc'].replace('Z','+00:00'))
        require(t_m.tzinfo is not None and t_p.tzinfo is not None and t_p>t_m,
                'Third-read permit must be a later, separate owner decision')
    except (ValueError,KeyError) as exc:
        raise RuntimeError('Missing/invalid decision times') from exc
    auth=SingleInvestigatorReleaseRecords(method,permit)
    return auth,digest(canonical({'method':method,'permit':permit})),method,permit


def upstream_snapshot(f20, upstream:dict, base:Path):
    """Use one validated fd per development input; never use the original pathname for Phase A."""
    f18=reviewed_module('F18')
    require(base.is_dir() and not base.is_symlink(),'Upstream snapshot root missing')
    want={
        'history':(Path(upstream['history_path']),upstream['development_history_sha256']),
        'scaler':(Path(upstream['scaler_path']),None),
        'regime':(Path(upstream['regime_path']),None),
    }
    # Verify frozen digests from authenticated F11's independent ledger, not mutable metadata.
    f11=f20.import_reviewed('F11')
    snap35,snap14,preaccess=f11.load_bundle()
    entries=f11.expected_non_test_paths(snap35,preaccess)
    signed={'history':entries['development history'][1],
            'scaler':entries['development scaler'][1],
            'regime':entries['regime thresholds'][1]}
    result={}
    for alias,(src,_) in want.items():
        require(src.is_absolute() and not src.is_symlink(),'Untrusted original upstream path')
        require(src.parent.is_dir() and not src.parent.is_symlink(),'Untrusted upstream parent')
        fd=os.open(src,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_CLOEXEC',0))
        dest=base/('signed_'+alias+'.'+('parquet' if alias=='history' else ('joblib' if alias=='scaler' else 'csv')))
        try:
            st=os.fstat(fd)
            require(stat.S_ISREG(st.st_mode),'Upstream input not regular file')
            h=hashlib.sha256()
            with os.fdopen(os.dup(fd),'rb',closefd=True) as r:
                flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)
                out=os.open(dest,flags,0o600)
                try:
                    while True:
                        data=r.read(1<<20)
                        if not data:break
                        h.update(data)
                        view=memoryview(data)
                        while view:
                            n=os.write(out,view)
                            require(n>0,'Snapshot write interrupted')
                            view=view[n:]
                    os.fsync(out)
                finally:os.close(out)
            after=os.fstat(fd)
            require((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==
                    (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),
                    'Upstream source changed during snapshot')
            require(h.hexdigest()==signed[alias] and sha_file(dest)==signed[alias],
                    'Frozen upstream source mismatch')
        finally:os.close(fd)
        result[alias]=str(dest)
    # Snapshot paths are private, committed once and never replaced.
    fs=reviewed_module('F22')
    fs.fsync_dir(base)
    return {**upstream,'history_path':result['history'],'scaler_path':result['scaler'],
            'regime_path':result['regime'],'development_history_sha256':signed['history']},signed


def source_read_to_snapshot(source:Path, dest:Path, journal, *, expected_digest:str, legacy:bytes):
    """Exactly ONE open of original raw-test file; reserve journal first; persist bytes while reading."""
    require(source.is_absolute() and not source.is_symlink(),'Unsafe sealed source')
    require(not dest.exists() and not dest.is_symlink(),'Already exists: sealed snapshot')
    require(HISTORY.read_bytes()==legacy,'Legacy marker changed before access')
    journal.reserve()  # Must be durable BEFORE original raw file is opened
    fd=os.open(source,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_CLOEXEC',0))
    target=dest.with_name('.RAW_SNAPSHOT_STAGING')
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode),'Sealed source not regular')
        out=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
        h=hashlib.sha256()
        size=0
        try:
            while True:
                b=os.read(fd,1<<20)
                if not b:break
                h.update(b);size+=len(b)
                part=memoryview(b)
                while part:
                    n=os.write(out,part)
                    require(n>0,'Snapshot write stalled')
                    part=part[n:]
            os.fsync(out)
        finally:os.close(out)
        after=os.fstat(fd)
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and
                size==before.st_size and h.hexdigest()==expected_digest,
                'Sealed source changed or did not match expected bytes')
    finally:os.close(fd)
    f18=reviewed_module('F18')
    f18.rename_noreplace(target,dest)
    require(sha_file(dest)==expected_digest,'Durable test snapshot digest mismatch')
    journal.verified(expected_digest,size)
    require(HISTORY.read_bytes()==legacy,'Legacy marker changed during third read')
    return journal.final_digest()


def stage_receipt(dest:Path,payload:dict):
    """Write once; if failure retains staging. No overwrite and no automatic restart."""
    f18=reviewed_module('F18')
    require(not dest.exists() and not dest.is_symlink(),'Receipt already exists')
    staging=dest.with_name('.F29_RECEIPT_STAGING')
    require(not staging.exists() and not staging.is_symlink(),'Abandoned receipt staging: manual review')
    raw=(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    f18.create_exclusive(staging,raw)
    f18.rename_noreplace(staging,dest)
    require(sha_file(dest)==digest(raw),'Receipt post-commit mismatch')


def phase_a(args,auth,authority_sha:str,permit:dict):
    f20=reviewed_module('F20');f22=reviewed_module('F22')
    f23=reviewed_module('F23');f24=reviewed_module('F24')
    from pathlib import Path
    source=Path(permit['source_path'])
    require(str(source)==str(Path(args.source_path)),'CLI and signed source path differ')
    legacy=historic_state()
    upstream=f20.authenticate_upstream()  # non-test read only
    snap,bind=f23.pre_read_upstream_snapshot(f20)
    require(f23.digest_bytes(f23.canonical(bind))==permit.get('upstream_binding_sha256'),
            'Signed permit not bound to frozen upstream inputs')
    require(f23.mapping_digest()==MAPPING_SHA,'Frozen seven-path mapping differs')
    require(auth.inspect(intended_source_path=source,
                         upstream_digest=f23.digest_bytes(f23.canonical(bind)))==authority_sha,
            'Authenticated authority does not match bound release records')
    check_critical_environment()
    with f24.cooperative_directory_lock(EXP/'manifests'):
        require(historic_state()==legacy,'Access history changed under lock')
        f23.check_new_output_paths_empty()
        destination=f23.ISOLATED['array_dir']
        require(destination.parent.is_dir() and not destination.parent.is_symlink(),
                'Untrusted Phase A destination parent')
        destination.mkdir(mode=0o700)
        f22.fsync_dir(destination.parent)
        snaps=destination/'upstream_snapshots'
        snaps.mkdir(mode=0o700)
        fs_upstream,signed=upstream_snapshot(f20,upstream,snaps)
        # Existing F23 input verification must still agree with the snapshotted originals.
        f23.recheck_upstream(bind)
        journal_path=f23.ISOLATED['independent_access_journal']
        journal=f22.ReservedReadJournal(journal_path,authority_sha)
        durable_raw=destination/'raw_test_snapshot.sealed'
        journal_sha=source_read_to_snapshot(source,durable_raw,journal,
                                           expected_digest=TEST_SHA,legacy=legacy)
        rows=f24.strict_jsonl_journal(journal_path.read_bytes(),authority_sha256=authority_sha,
                                      require_complete=True)
        require(len(rows)==2 and sha_file(journal_path)==journal_sha,'Journal verification failed')
        # Deliberate second read is from the *same already-persisted authorized snapshot*,
        # NOT a second open of the original sealed raw-test file.
        # Recovery of snapshot after a crash requires distinct explicit review; no CLI supports it.
        raw_bytes=durable_raw.read_bytes()
        require(digest(raw_bytes)==TEST_SHA,'Saved snapshot changed before materialization')
        phase_digest=f20.seal_preloaded_phase_a(raw_verified_bytes=raw_bytes,
                   authorized_journal_digest=journal_sha,output_parent=destination,
                   upstream=fs_upstream)
        require(f20.inspect_phase_a(destination/'phase_a_materialization')[-1]==phase_digest,
                'Phase A package not independently verified')
        f23.recheck_upstream(bind)
        require(historic_state()==legacy,'Historical marker changed during materialization')
        stage_receipt(f23.ISOLATED['array_manifest'],{
             'schema':'STEP36I_F29_PHASE_A_RECEIPT_V1',
             'step35_sha256':SHA35,'owner_amendment_sha256':OWNER_SHA,
             'access_journal_format':'application/x-ndjson',
             'access_journal_sha256':journal_sha,
             'phase_a_manifest_sha256':phase_digest,
             'upstream_snapshot_sha256':signed,
             'raw_test_snapshot_sha256':TEST_SHA,
             'output_mapping_sha256':MAPPING_SHA,
             'authority_binding_sha256':authority_sha,
             'third_read_completed_under_separate_signed_permit':True,
             'historical_marker_left_unchanged':True,
             'real_scoring_performed':False})
    return phase_digest


def assert_phase_a_for_phase_b(authority_sha:str):
    f20=reviewed_module('F20');f23=reviewed_module('F23');f24=reviewed_module('F24')
    array_root=f23.ISOLATED['array_dir']
    receipt=f23.ISOLATED['array_manifest']
    require(receipt.is_file() and not receipt.is_symlink(),'No Phase A receipt')
    r=json.loads(receipt.read_bytes())
    require(r.get('schema')=='STEP36I_F29_PHASE_A_RECEIPT_V1' and
            r.get('authority_binding_sha256')==authority_sha and
            r.get('output_mapping_sha256')==MAPPING_SHA and
            r.get('raw_test_snapshot_sha256')==TEST_SHA,
            'Phase A receipt provenance mismatch')
    j=f23.ISOLATED['independent_access_journal']
    require(sha_file(j)==r['access_journal_sha256'],'Journal changed after Phase A')
    require(len(f24.strict_jsonl_journal(j.read_bytes(),authority_sha256=authority_sha,
                 require_complete=True))==2,'Malformed persisted read journal')
    package=array_root/'phase_a_materialization'
    require(f20.inspect_phase_a(package)[-1]==r['phase_a_manifest_sha256'],
            'Committed Phase A package mismatch')
    return package


def phase_b(args,authority_sha:str):
    f20=reviewed_module('F20');f23=reviewed_module('F23');f24=reviewed_module('F24')
    check_critical_environment()
    before=historic_state()
    with f24.cooperative_directory_lock(EXP/'manifests'):
        phase_pkg=assert_phase_a_for_phase_b(authority_sha)
        seeds=f23.ISOLATED['predictions']
        if not seeds.exists():
            seeds.mkdir(mode=0o700)
            reviewed_module('F22').fsync_dir(seeds.parent)
        else:
            check_private_dir(seeds)
        # This operation can be explicitly repeated after a GPU interruption;
        # completed committed seeds are verified and skipped. It NEVER opens raw test.
        plan,_=f20.frozen_plan()
        f19=f20.import_reviewed('F19');f18=f20.import_reviewed('F18')
        phase_digest=reviewed_module('F18').phase_a_manifest_digest(phase_pkg)
        done,missing=f19.strict_seed_inventory(seeds,phase_digest,plan,f18)
        print(f'Phase B: {len(done)} completed; {len(missing)} missing checkpoints')
        for model,seed in sorted(missing):
            print(f'Running frozen {model} seed {seed} — physical batch per signed plan',flush=True)
            f20.compute_one_seed(phase_package=phase_pkg,seed_store=seeds,model=model,seed=seed)
        complete,missing=f19.strict_seed_inventory(seeds,phase_digest,plan,f18)
        require(len(complete)==40 and not missing,'Phase B incomplete')
        require(historic_state()==before,'Legacy marker drift during Phase B')
    return True


def finalize(args,authority_sha:str):
    f20=reviewed_module('F20');f23=reviewed_module('F23');f24=reviewed_module('F24')
    f18=reviewed_module('F18')
    check_critical_environment()
    before=historic_state()
    with f24.cooperative_directory_lock(EXP/'manifests'):
        phase_pkg=assert_phase_a_for_phase_b(authority_sha)
        seeds=f23.ISOLATED['predictions']
        require(seeds.is_dir(),'No committed Phase B seed store')
        require(not f23.ISOLATED['evaluation_manifest'].exists() and
                not f23.ISOLATED['metrics'].exists(),
                'Final result manifest or metrics path pre-exists')
        target=f20.finalize_from_saved_seeds(phase_package=phase_pkg,
                 seed_store=seeds,output_parent=seeds)
        # Publish two pre-reserved standard paths after the complete atomic result directory.
        manifest=target/'evaluation_manifest.json'
        metrics=target/'classification_metrics.csv'
        require(manifest.is_file() and metrics.is_file(),'Final result artifacts missing')
        # Publish the EXACT frozen-format CSV at the reserved metrics path.
        # The final JSON is written LAST as the no-overwrite release receipt.
        # A failure between these commits leaves durable evidence and requires
        # manual review; there is no automatic overwrite or replay.
        expected_csv=metrics.read_bytes()
        require(expected_csv.startswith(b'model_id,') or
                b'model_id' in expected_csv.splitlines()[0],
                'Classification metrics file is not CSV')
        csv_dst=f23.ISOLATED['metrics']
        csv_stage=csv_dst.with_name('.F29_METRICS_STAGING')
        require(not csv_stage.exists() and not csv_stage.is_symlink(),
                'Previous metrics staging exists; manual review')
        f18.create_exclusive(csv_stage,expected_csv)
        f18.rename_noreplace(csv_stage,csv_dst)
        require(sha_file(csv_dst)==sha_file(metrics),'Published CSV differs from final artifact')
        content={
            'schema':'STEP36I_F29_FINAL_RESULT_RECEIPT_V1',
            'step35_sha256':SHA35,
            'owner_amendment_sha256':OWNER_SHA,
            'output_mapping_sha256':MAPPING_SHA,
            'phase_a_manifest_sha256':f18.phase_a_manifest_digest(phase_pkg),
            'evaluation_artifact_manifest_sha256':sha_file(manifest),
            'classification_metrics_csv_sha256':sha_file(csv_dst),
            'classification_metrics_csv_path':str(csv_dst),
            'completed_evaluation_dir':str(target),
            'primary_metrics_rows':24,
            'bootstrap_rows':72,
            'primary_bh_tests':24,
            'exploratory_pairwise_comparisons':84,
            'real_final_test_scored':True,
            'legacy_historical_marker_preserved':True,
        }
        stage_receipt(f23.ISOLATED['evaluation_manifest'],content)
        require(historic_state()==before,
                'Legacy historical access marker drift during finalization')
        return target


def portable_safety_test():
    """Unit-test authentic orchestration boundaries without any project or test data."""
    with tempfile.TemporaryDirectory(prefix='F29_SAFETY_ONLY_') as temp:
        parent=Path(temp)
        j=parent/'safe_journal.ndjson'
        f22=reviewed_module('F22');f24=reviewed_module('F24')
        auth=digest(b'F29_ONLY_MOCK_AUTH')
        journal=f22.ReservedReadJournal(j,auth)
        journal.reserve()
        partial=f24.strict_jsonl_journal(j.read_bytes(),authority_sha256=auth,require_complete=False)
        require(len(partial)==1,'Synthetic reservation not durable')
        journal.verified(digest(b'SYNTHETIC'),9)
        full=f24.strict_jsonl_journal(j.read_bytes(),authority_sha256=auth)
        require(len(full)==2,'Synthetic journal not valid')
        with f24.cooperative_directory_lock(parent):
            pass
        require(journal.final_digest()==sha_file(j),'Journal hash mismatch')
    return True


def solo_owner_signature_selftest():
    """Ephemeral in-memory key: test the solo-review policy, never authorize real access.

    The key is never saved, is not a trusted release key, and is never used
    outside this temporary synthetic fixture.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    from types import SimpleNamespace
    from datetime import datetime,timezone,timedelta
    global OWNER_PINNED_KEYRING_SHA256
    assert OWNER_PINNED_KEYRING_SHA256 is None, 'Fixture cannot run with release key pinned'
    key=Ed25519PrivateKey.generate()
    pub=key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,format=serialization.PublicFormat.Raw)
    identity='SYNTHETIC_TEST_OWNER_NOT_A_REAL_SIGNATORY'
    with tempfile.TemporaryDirectory(prefix='F29_SYNTHETIC_SIGNER_ONLY_') as d:
        temp=Path(d)
        ring=temp/'synthetic_public_keyring.json'
        ring.write_bytes(canonical({
            'schema':'F29_SINGLE_INVESTIGATOR_TRUST_V1',
            'review_model':'SOLE_RESEARCHER_SELF_REVIEW_NOT_INDEPENDENT',
            'owner_identity':identity,
            'owner_public_key_b64':base64.b64encode(pub).decode(),
        }))
        source=temp/'SYNTHETIC_NEVER_REAL_TEST.parquet'
        source.write_bytes(b'UNIT_TEST_ONLY')
        stamp=datetime(2026,10,15,9,0,tzinfo=timezone.utc)
        method={
            'record_type':'SINGLE_INVESTIGATOR_METHOD_SELF_REVIEW',
            'decision':'APPROVED_WITH_LIMITATIONS_FOR_AMENDED_EVALUATION',
            'review_type':'SELF_REVIEW_NOT_INDEPENDENT',
            'owner_identity':identity,
            'f29_source_sha256':sha_file(Path(__file__)),
            'decision_time_utc':stamp.isoformat(),
            'limitations_acknowledged':True,
            'step35_sha256':SHA35,
            'study_owner_approval_sha256':OWNER_SHA,
            'approved_candidate_keys_sha256':'4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65',
        }
        permit={
            'record_type':'SEPARATE_THIRD_RAW_TEST_READ_PERMIT',
            'decision':'EXPLICIT_SINGLE_ADDITIONAL_RAW_READ_AUTHORIZED',
            'owner_identity':identity,
            'f29_source_sha256':sha_file(Path(__file__)),
            'decision_time_utc':(stamp+timedelta(hours=1)).isoformat(),
            'historical_reads_before_release':2,
            'test_sha256':TEST_SHA,
            'method_record_sha256':digest(canonical(method)),
            'step35_sha256':SHA35,
            'study_owner_approval_sha256':OWNER_SHA,
            'output_mapping_sha256':MAPPING_SHA,
            'upstream_binding_sha256':'test-upstream-binding-00000000',
            'source_path':str(source),
            'acknowledge_two_historical_reads':True,
            'allow_durable_raw_snapshot_for_recovery':True,
            'permit_scope':'ONE_RAW_READ_WITH_DURABLE_PHASE_A_THEN_40_FROZEN_SEEDS',
        }
        mpath=temp/'method.json';ppath=temp/'read_permit.json'
        def write_signed(path,payload):
            path.write_bytes(canonical({
                'payload':payload,
                'signature_b64':base64.b64encode(key.sign(canonical(payload))).decode(),
            }))
        write_signed(mpath,method);write_signed(ppath,permit)
        args=SimpleNamespace(trusted_keys=str(ring),keyring_sha256=sha_file(ring),
                             method_attestation=str(mpath),read_permit=str(ppath))
        OWNER_PINNED_KEYRING_SHA256=args.keyring_sha256
        try:
            obj,sha,ma,pe=validate_signatures(args,None)
            require(obj.inspect(intended_source_path=source,
                                upstream_digest=permit['upstream_binding_sha256'])==sha,
                    'Valid signed solo-authority fixture failed to bind')
            # Tamper without resigning: cryptographic verification must fail.
            bad=dict(permit);bad['historical_reads_before_release']=3
            write_signed(ppath,bad)  # valid signature but invalid read history
            try:validate_signatures(args,None)
            except RuntimeError:pass
            else:raise RuntimeError('Wrong access count accepted')
            # Recreate exact permit but with timestamp BEFORE methodological decision.
            bad=dict(permit);bad['decision_time_utc']=(stamp-timedelta(hours=1)).isoformat()
            write_signed(ppath,bad)
            try:validate_signatures(args,None)
            except RuntimeError:pass
            else:raise RuntimeError('Premature permit accepted')
            # Corrupted signature must be rejected even if the payload is otherwise valid.
            write_signed(ppath,permit)
            altered=json.loads(ppath.read_bytes())
            altered['payload']['decision']='FORGED'
            ppath.write_bytes(canonical(altered))
            try:validate_signatures(args,None)
            except RuntimeError:pass
            else:raise RuntimeError('Forged payload accepted')
        finally:
            OWNER_PINNED_KEYRING_SHA256=None
    return True


def offline_preflight():
    print('='*102)
    print('STEP 36I-F29 — INTEGRATED TWO-PHASE RELEASE CANDIDATE (OFFLINE; NO TEST ACCESS)')
    print('='*102)
    core=authentic_core()
    f23=reviewed_module('F23');f20=reviewed_module('F20')
    require(f23.mapping_digest()==MAPPING_SHA and len(f23.ISOLATED)==7,
            'Seven frozen output paths differ')
    f20.authenticate_upstream() # signed development inputs only
    print('PASS — signed sources, Step35, F13E, 63-feature/20-stock upstream and output map')
    print('PASS — legacy historical access marker 2 reads, 0 scored')
    check_private = all(not p.exists() and not p.is_symlink() for p in f23.ISOLATED.values())
    require(check_private,'Amended output namespace not vacant')
    print('PASS — seven amended real output destinations vacant')
    portable_safety_test()
    print('PASS — F22 journal and F24 lock tested together on isolated fixture')
    solo_owner_signature_selftest()
    print('PASS — owner-signed self-review and DISTINCT later permit: positive/negative synthetic oracles')
    import inspect
    require({'raw_verified_bytes','authorized_journal_digest','output_parent','upstream'}<=
            set(inspect.signature(f20.seal_preloaded_phase_a).parameters),
            'Genuine F20 Phase A interface absent')
    require(callable(f20.compute_one_seed) and callable(f20.finalize_from_saved_seeds),
            'Phase B computation/finalization not connected')
    print('PASS — live Phase A/40-seed Phase B/finalization APIs wired in F29 source')
    require(historic_state()==core,'Historical marker changed during preflight')
    print('BLOCKED — owner public-key trust anchor not pinned; this is a release candidate')
    print('BLOCKED — separate owner-signed self-review/read permit not supplied')
    print('BLOCKED — independent review unavailable; owner self-review must be disclosed')
    print('BLOCKED — third read not authorized; no release CLI executed')
    print('='*102)
    print('F29 OFFLINE SOURCE PREFLIGHT PASS — FINAL RESULTS NOT YET GENERATED')
    print('Sealed final test read: FALSE | GPU: FALSE | Real scoring: FALSE')
    print('='*102)


@contextlib.contextmanager
def dependency_library_import_context():
    """Do not let F29's release argv be misread as a frozen HELPER's own CLI.

    This context is entered only AFTER successful external signature validation.
    Frozen helpers remain byte-identical; their own standalone --execute gates
    are not invoked. Independent release review must inspect this delegation.
    """
    previous=sys.argv[:]
    sys.argv=[str(Path(__file__)),'--offline-preflight']
    try:
        yield
    finally:
        sys.argv=previous


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    x=parser.add_mutually_exclusive_group(required=True)
    x.add_argument('--offline-preflight',action='store_true')
    x.add_argument('--phase-a',action='store_true')
    x.add_argument('--phase-b',action='store_true')
    x.add_argument('--finalize',action='store_true')
    parser.add_argument('--trusted-keys')
    parser.add_argument('--keyring-sha256')
    parser.add_argument('--method-attestation')
    parser.add_argument('--read-permit')
    parser.add_argument('--source-path')
    args=parser.parse_args()
    if args.offline_preflight:
        offline_preflight()
        return
    # Do not touch sealed input, GPU, output dirs or journal before BOTH owner signatures verify.
    authentic_core()
    f23=reviewed_module('F23')
    auth,authority_sha,method,permit=validate_signatures(args,f23)
    require(permit.get('source_path') and args.source_path==permit['source_path'],
            'Signed permit source does not match supplied input')
    with dependency_library_import_context():
        try:
            if args.phase_a:
                result=phase_a(args,auth,authority_sha,permit)
                print('Phase A committed and verified:',result)
            elif args.phase_b:
                phase_b(args,authority_sha)
                print('All 40 frozen seeds committed and verified')
            elif args.finalize:
                finalize(args,authority_sha)
                print('Final evaluation committed and verified')
        except Exception as exc:
            # Durable failure evidence after valid signatures; never overwrite
            # or silently retry. Retains phase-specific abandoned staging.
            evidence=f23.ISOLATED['failure_evidence']
            if not evidence.exists() and not evidence.is_symlink():
                try:
                    stage_receipt(evidence,{
                        'schema':'STEP36I_F29_FAILED_AUTHORIZED_ATTEMPT_V1',
                        'phase':('PHASE_A' if args.phase_a else
                                 ('PHASE_B' if args.phase_b else 'FINALIZE')),
                        'error_type':type(exc).__name__,
                        'error_message':str(exc)[:3000],
                        'source_sha256':sha_file(Path(__file__)),
                        'authorization_binding_sha256':authority_sha,
                        'historical_access_marker_sha256':sha_file(HISTORY),
                        'automatic_retry_authorized':False,
                    })
                except Exception as logging_error:
                    print('F29 FAILURE EVIDENCE COMMIT ALSO FAILED:',
                          str(logging_error),file=sys.stderr)
            raise


if __name__=='__main__':
    try:
        cli()
    except Exception as e:
        print('F29 FAIL-CLOSED:',type(e).__name__,str(e),file=sys.stderr)
        sys.exit(1)
