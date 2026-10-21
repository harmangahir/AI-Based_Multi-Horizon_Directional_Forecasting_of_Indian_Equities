#!/usr/bin/env python3
"""STEP 36I-F23 — permission-gated real Phase A release controller CANDIDATE.

REVIEW ONLY. Its CLI supports solely --offline-preflight and --portable-self-test.
It is NOT an approved live command and has NOT been independently audited.

Adds a real, *library-only* path from separately authorized and identified raw
bytes, through an irreversible one-attempt journal, into the reviewed F20 Phase A
materializer. The original historical marker is never modified. The F22 mock
fixture reader is NOT promoted to production.

Before the library-only function can ever be called, a separately approved
methodological sign-off record and a distinct, explicit third-read permit MUST
exist; those records do not exist/are not issued by this source. The records
must undergo external audit; JSON fields are procedural evidence, not secure
cryptographic signatures. No production CLI is exposed by this candidate.

The candidate has intentionally NOT been connected to a live finalization
launcher. Real result output mapping and concurrency/permissions need further
independent source review before any new access is considered.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Callable

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
STEP35 = EXP / 'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
HISTORY = EXP / 'manifests/36_final_test_access_consumption_state.json'
APPROVAL = ROOT / 'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json'
F20 = ROOT / 'step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py'
F22 = ROOT / 'step36i_f22_access_bound_phase_a_controller_REVIEW_ONLY.py'
F1 = ROOT / 'step36i_f1_amended_production_runner_REVIEW_ONLY.py'

SHA = {
    'F20': '937de4fb1dc7a58f6ecb8449ad55a515a5dcb28120adbeaa82a4e31c0af5d018',
    'F22': '7ae182b83a20d2f621ded09ceb7e5e696f1b9c41b0f814a0ffd05c4d79997b26',
    'F1': 'e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556',
    'Approval': 'f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493',
    'Step35': '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7',
    'Test': 'ab325ede85fbb6d7819e85472274b82d151d350f39a872a17747c6978b1dddae',
}

# This set is taken directly from F1, not a new/different results namespace.
ISOLATED = {
    'array_dir': EXP / 'feature_data/step36i_amended_final_test_arrays',
    'array_manifest': EXP / 'manifests/36i_amended_final_test_materialization_manifest.json',
    'predictions': EXP / 'final_test_evaluation/step36i_amended_predictions',
    'evaluation_manifest': EXP / 'manifests/36i_amended_final_test_evaluation_manifest.json',
    'metrics': EXP / 'audits/36i_amended_final_test_metrics.csv',
    'independent_access_journal': EXP / 'manifests/36i_amended_final_test_access_journal.json',
    'failure_evidence': EXP / 'audits/36i_amended_final_test_execution_failure.json',
}
LIVE_RELEASE_ENABLED = False  # Never promote this constant in this review-only source.

# F1's journal filename ends in .json. This candidate's journal is an ordered
# JSON Lines event log stored at that frozen filename. This format discrepancy
# requires explicit independent approval before live release.

DENY = {'--execute', '--production', '--read-test', '--live', '--phase-a',
        '--phase-b', '--resume', '--gpu', '--authorize', '--grant',
        '--commit-real', '--run', '--approve', '--final-test'}
if DENY & set(sys.argv[1:]):
    raise SystemExit('DENIED — F23 has no live production command or third-read authority')


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f'Unsafe or missing signed input: {path}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checked(path: Path, expected: str, name: str) -> None:
    need(digest_file(path) == expected, f'Frozen {name} SHA-256 mismatch')


def canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def get_history() -> bytes:
    need(HISTORY.is_file() and not HISTORY.is_symlink(), 'Historical access marker missing')
    snapshot = HISTORY.read_bytes()
    record = json.loads(snapshot)
    need(record.get('raw_test_filesystem_open_count_completed') == 2 and
         record.get('final_test_scored') is False, 'Unexpected historical test state')
    return snapshot


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def canonical_output_map() -> dict[str, str]:
    """Intentionally returns fixed, fully-qualified F1 output destinations."""
    return {name: str(path) for name, path in sorted(ISOLATED.items())}


def mapping_digest() -> str:
    return digest_bytes(canonical(canonical_output_map()))


def check_new_output_paths_empty() -> None:
    for name, path in ISOLATED.items():
        need(not path.exists() and not path.is_symlink(),
             f'Pre-existing amended output {name}: {path}; refuse overwrite/retry')
    # Do not make any directories in offline preflight.


@dataclass(frozen=True)
class ReleaseRecords:
    """Two externally supplied, separately approved records (not created here)."""
    method: dict
    permit: dict

    def inspect(self, *, intended_source_path: Path, upstream_digest: str) -> str:
        a, b = self.method, self.permit
        # Exact, deliberately distinct acceptance statements required.
        need(a.get('record_type') == 'INDEPENDENT_METHOD_REVIEW_SIGNOFF' and
             a.get('decision') == 'APPROVED_FOR_AMENDED_PROTOCOL' and
             bool(a.get('independent_reviewer_identity')) and
             a.get('step35_sha256') == SHA['Step35'] and
             a.get('study_owner_approval_sha256') == SHA['Approval'],
             'Independent protocol sign-off not demonstrated')
        need(b.get('record_type') == 'SEPARATE_THIRD_RAW_TEST_READ_PERMIT' and
             b.get('decision') == 'EXPLICIT_SINGLE_ADDITIONAL_RAW_READ_AUTHORIZED' and
             b.get('historical_reads_before_release') == 2 and
             b.get('test_sha256') == SHA['Test'] and
             b.get('method_record_sha256') == digest_bytes(canonical(a)) and
             b.get('step35_sha256') == SHA['Step35'] and
             b.get('study_owner_approval_sha256') == SHA['Approval'] and
             b.get('output_mapping_sha256') == mapping_digest() and
             b.get('upstream_binding_sha256') == upstream_digest and
             b.get('source_path') == str(intended_source_path) and
             bool(b.get('authorizing_person')),
             'Explicit third-read permit not demonstrated')
        need(intended_source_path.is_absolute() and not intended_source_path.is_symlink() and
             intended_source_path.parent.resolve() == intended_source_path.parent,
             'Unsafe or noncanonical approved input path')
        # A policy record's fields do NOT cryptographically authenticate its
        # signer's identity. Independent human sign-off still needs verification.
        return digest_bytes(canonical({'method': a, 'permit': b}))


def pre_read_upstream_snapshot(f20) -> tuple[dict, dict]:
    """Read only development artifacts and record their known signed digests."""
    upstream = f20.authenticate_upstream()
    paths = {
        'development_scaler': str(upstream['scaler_path']),
        'development_history': str(upstream['history_path']),
        'regime_thresholds': str(upstream['regime_path']),
    }
    for key, raw_path in paths.items():
        need(Path(raw_path).is_absolute() and not Path(raw_path).is_symlink(),
             f'Invalid upstream {key} path')
    fingerprints = {k: digest_file(Path(path)) for k, path in paths.items()}
    need(fingerprints['development_history'] == upstream['development_history_sha256'],
         'Development history does not match signed F20 provenance')
    bound = {'paths': paths, 'sha256': fingerprints,
             'feature_interface_sha256': SHA['Step35'],
             'approved_candidate_sha256': '4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65',
             'feature_count': len(upstream['features']),
             'stock_count': len(upstream['stock_id_map'])}
    need(bound['feature_count'] == 63 and bound['stock_count'] == 20,
         'Frozen feature or stock interface drift')
    return upstream, bound


def recheck_upstream(bound: dict) -> None:
    for key, raw_path in bound['paths'].items():
        need(digest_file(Path(raw_path)) == bound['sha256'][key],
             f'Upstream input changed during Phase A: {key}')


def journalized_single_read(source: Path, *, approved_source_sha256: str,
                            journal, expected_history: bytes) -> tuple[bytes, str]:
    """Single file-descriptor access AFTER irreversible on-disk reservation.

    Library function only. No caller in this script's CLI may execute it.
    Any failure consumes the journal reservation. Never auto-retry.
    """
    need(approved_source_sha256 == SHA['Test'], 'Not the independently frozen test digest')
    need(HISTORY.read_bytes() == expected_history, 'Historical marker changed before source read')
    need(source.is_absolute() and not source.is_symlink() and
         source.parent.resolve() == source.parent, 'Unsafe raw input path')
    journal.reserve()  # durable O_EXCL before raw source file is opened
    fd = os.open(source, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) |
                 getattr(os, 'O_CLOEXEC', 0))
    try:
        a = os.fstat(fd)
        need(stat.S_ISREG(a.st_mode), 'Sealed source is not a regular file')
        chunks = []
        h = hashlib.sha256()
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            h.update(block)
            chunks.append(block)
        b = os.fstat(fd)
        need((a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns) ==
             (b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns),
             'Sealed input changed during single read')
        data = b''.join(chunks)
        need(len(data) == a.st_size and h.hexdigest() == approved_source_sha256,
             'Sealed input bytes/hash mismatch after single read')
    finally:
        os.close(fd)
    journal.verified(h.hexdigest(), len(data))
    need(HISTORY.read_bytes() == expected_history,
         'Historical access marker changed during single read')
    return data, journal.final_digest()


def live_phase_a_candidate(*, source: Path, records: ReleaseRecords,
                           independently_reviewed_source: bool):
    """NOT PUBLICLY RELEASED; library-only code for independent source audit.

    Even external records cannot trigger this function unless an independently
    audited future version explicitly removes the deny gate below. That future
    change will require a *new* file/source SHA-256 and separate authorization.
    """
    raise PermissionError(
        'F23 LIVE RELEASE DENIED — candidate requires independent source review, '
        'verified methodological sign-off and distinct third-read authorization')


def proposed_phase_a_algorithm(*, source: Path, records: ReleaseRecords,
                               output_parent: Path, f20, f22):
    """Source-review-only algorithm, unexposed by any production CLI.

    The only consumer is F20.seal_preloaded_phase_a. This function must NOT
    be called with actual test data while the F23 release deny remains active.
    """
    before = get_history()
    upstream, bound = pre_read_upstream_snapshot(f20)
    auth_digest = records.inspect(intended_source_path=source,
                                  upstream_digest=digest_bytes(canonical(bound)))
    need(output_parent == ISOLATED['array_dir'], 'Array output parent mapping changed')
    need(not output_parent.exists() and not output_parent.is_symlink(),
         'Output parent already occupied; refuse replay')
    # A future independently audited version may provide an accessible CLI,
    # but THIS source never permits either real input access or filesystem
    # writes. These lines are intentionally an unreachable implementation
    # candidate for source review only.
    need(LIVE_RELEASE_ENABLED, 'DENIED — review candidate cannot create any live artifact')
    check_new_output_paths_empty()
    need(not ISOLATED['independent_access_journal'].exists(),
         'Access journal already exists: never restart or overwrite')
    need(output_parent.parent.is_dir() and not output_parent.parent.is_symlink(),
         'Missing/unsafe approved array parent')
    # The isolated output parent must be durably created before the reserved
    # third read. If anything then fails, retain evidence; do not auto-retry.
    output_parent.mkdir(mode=0o700)
    f22.fsync_dir(output_parent.parent)
    journal=f22.ReservedReadJournal(ISOLATED['independent_access_journal'],auth_digest)
    raw_bytes,journal_sha=journalized_single_read(
        source,approved_source_sha256=SHA['Test'],journal=journal,
        expected_history=before)
    recheck_upstream(bound)
    phase_a_sha=f20.seal_preloaded_phase_a(
        raw_verified_bytes=raw_bytes,
        authorized_journal_digest=journal_sha,
        output_parent=output_parent,
        upstream=upstream)
    recheck_upstream(bound)
    need(get_history()==before,'Original historical access marker modified')
    package=output_parent/'phase_a_materialization'
    need(f20.inspect_phase_a(package)[-1] == phase_a_sha,
         'Phase A durable manifest validation failed')
    # Future approved release must now write the separately reserved summary
    # manifest, verify output-path mapping, and audit the journal JSONL format.
    # No implicit GPU execution, no test-score computation and no retry here.
    return {'phase_a_package':str(package), 'phase_a_sha256':phase_a_sha,
            'access_journal_sha256':journal_sha,
            'authorization_binding_sha256':auth_digest,
            'output_mapping_sha256':mapping_digest()}


def portable_selftest() -> None:
    need(len(ISOLATED) == 7 and mapping_digest(), 'Bad frozen mapping')
    need('live_phase_a_candidate' in globals() and 'journalized_single_read' in globals(),
         'Missing review interfaces')
    f20 = module(F20, 'step36i_f23_f20')
    f22 = module(F22, 'step36i_f23_f22')
    need({'raw_verified_bytes', 'authorized_journal_digest', 'output_parent', 'upstream'} <=
         set(inspect.signature(f20.seal_preloaded_phase_a).parameters),
         'F20 Phase A consumer signature drift')
    need({'path', 'authority_digest'} <=
         set(inspect.signature(f22.ReservedReadJournal).parameters),
         'F22 durable journal signature drift')
    # Wrong/missing independent records are fail-closed before any access.
    bad = ReleaseRecords({}, {})
    try:
        bad.inspect(intended_source_path=ROOT / 'DO_NOT_OPEN_TEST.parquet', upstream_digest='0'*64)
    except RuntimeError:
        pass
    else:
        raise RuntimeError('Missing third-read records were accepted')
    for invoker in (
        lambda: live_phase_a_candidate(source=ROOT/'DO_NOT_OPEN_TEST.parquet',
                                       records=bad,independently_reviewed_source=False),
        lambda: proposed_phase_a_algorithm(source=ROOT/'DO_NOT_OPEN_TEST.parquet',
                      records=bad,output_parent=Path('/DO_NOT_USE'),f20=f20,f22=f22),
    ):
        try:
            invoker()
        except (RuntimeError, PermissionError):
            pass
        else:
            raise RuntimeError('Unexpected live release bypass')


def offline_preflight() -> None:
    print('=' * 106)
    print('STEP 36I-F23 — APPROVAL/JOURNAL TO F20 PHASE A CONTROLLER SOURCE CANDIDATE')
    print('=' * 106)
    print('[1] REVIEWED SOURCE INTEGRITY AND TWO HISTORICAL RAW READS')
    for key, path in [('F20',F20), ('F22',F22), ('F1',F1),
                      ('Step35',STEP35), ('Approval',APPROVAL)]:
        checked(path, SHA[key], key)
    before = get_history()
    print('PASS — F20/F22/F1, Step35 and F13E match frozen digests; marker 2/0')
    print('[2] PREVIOUSLY RESERVED OUTPUT MAPPING — NO CREATION')
    f1 = module(F1, 'step36i_f23_f1')
    need({k: str(v) for k, v in f1.ISOLATED.items()} ==
         {k: str(v) for k, v in ISOLATED.items()},
         'F1 output mapping not preserved')
    check_new_output_paths_empty()
    print('PASS — 7 exact F1 amended paths vacant; output-map binding fingerprint:', mapping_digest())
    print('[3] ACTUAL F20/F22 API BINDINGS AND NO-GRANT CONTROLS')
    portable_selftest()
    print('PASS — source API matched; separate independent signoff and third-read permit absent/denied')
    print('[4] FROZEN NON-TEST INPUT AUTHENTICATION AND DRIFT SNAPSHOT')
    f20=module(F20, 'step36i_f23_auth_f20')
    upstream,bound=pre_read_upstream_snapshot(f20)
    recheck_upstream(bound)
    need(get_history()==before, 'Original marker changed during offline preflight')
    print('PASS — 63 features/20 IDs, three development artifact hashes and upstream fingerprint:',
          digest_bytes(canonical(bound)))
    print('[5] EXPLICITLY UNRESOLVED RELEASE GATES')
    print('OPEN — independent source and methodological review, signed evidence and third-read decision')
    print('OPEN — production access/output mapping, concurrent journal protection and finalization wiring')
    print('OPEN — F22 .json filename contains JSON Lines data; review format compatibility')
    print('=' * 106)
    print('F23 PASS — REVIEW-ONLY CONTROLLER DESIGN; NO LIVE RAW-TEST ACCESS PATH EXPOSED')
    print('Real test opened: FALSE | Real scoring: FALSE | GPU: FALSE | Production CLI: DENIED')
    print('=' * 106)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--offline-preflight', action='store_true')
    group.add_argument('--portable-self-test', action='store_true')
    args=parser.parse_args()
    if args.offline_preflight:
        offline_preflight()
    else:
        print('F23 — PORTABLE CONTROLLER SOURCE NEGATIVE TEST')
        portable_selftest()
        print('PASS — no real source access possible')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, PermissionError, ValueError, KeyError, OSError, AssertionError) as exc:
        print('F23 FAIL CLOSED:', type(exc).__name__, str(exc), file=sys.stderr)
        raise SystemExit(1)
