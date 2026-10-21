#!/usr/bin/env python3
"""STEP 36I-F22 — authorization-bound Phase A controller *design* (REVIEW ONLY).

This module supplies a concrete, independently inspectable proposed boundary:
  verified non-test upstream -> immutable context digest -> durable single-attempt
  reservation -> exactly one byte read -> verified evidence -> F20 Phase A API.

F22's supported CLI is ONLY --offline-preflight or --portable-self-test.
It does NOT expose a live loader or accept authorization for a third raw-test
read. The production callback is replaced by a synthetic evidence consumer.
Before ANY real release, a separate implementation and independent signoff must
bind an explicit, separately approved access decision, the exact source path,
upstream snapshot, crash semantics, live journal path and F20 materializer.

Never reset the original historical access marker. It remains a record of two
previous opens. The future third attempt requires a DISTINCT append-only journal.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Callable

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
F20 = ROOT / 'step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py'
F21 = ROOT / 'step36i_f21_independent_live_release_gap_audit_READ_ONLY.py'
APPROVAL = ROOT / 'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json'
STEP35 = EXP / 'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
MARKER = EXP / 'manifests/36_final_test_access_consumption_state.json'

HASHES = {
    'F20': '937de4fb1dc7a58f6ecb8449ad55a515a5dcb28120adbeaa82a4e31c0af5d018',
    'F21': 'd668e8edfb3d2ac7e1ffb09c030826a4a509054abee28b9bf935708c73a42ab0',
    'APPROVAL': 'f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493',
    'STEP35': '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7',
}
SEALED_SHA = 'ab325ede85fbb6d7819e85472274b82d151d350f39a872a17747c6978b1dddae'

DENY_ARGS = {'--execute', '--production', '--read-test', '--phase-a', '--phase-b',
             '--resume', '--gpu', '--commit-real', '--authorize', '--grant',
             '--live', '--final-test', '--run', '--approve'}
if DENY_ARGS & set(sys.argv[1:]):
    raise SystemExit('DENIED — F22 has no authorized production entrypoint')


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f'Missing/unsafe artifact: {path}')
    h = hashlib.sha256()
    with path.open('rb') as inp:
        for piece in iter(lambda: inp.read(1 << 20), b''):
            h.update(piece)
    return h.hexdigest()


def assert_sha(path: Path, expected: str, label: str) -> None:
    require(file_sha(path) == expected, f'{label} SHA-256 mismatch')


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class SyntheticAuthorization:
    """Non-transferable fixture attestation. NEVER sufficient for real test."""
    source_sha256: str
    source_basename: str
    upstream_sha256: str
    owner_approval_sha256: str
    step35_sha256: str
    mock_authorization_id: str = 'SYNTHETIC_ONLY_NOT_LIVE_AUTHORIZATION'

    def digest(self) -> str:
        return hexdigest(canonical(self.__dict__))

    def validate_fixture(self, fixture: Path) -> None:
        require(self.mock_authorization_id == 'SYNTHETIC_ONLY_NOT_LIVE_AUTHORIZATION',
                'Fixture authority class cannot become live permission')
        require(fixture.name.startswith('SYNTHETIC_ONLY_') and fixture.name == self.source_basename,
                'Non-synthetic or mismatched source blocked')
        require(self.owner_approval_sha256 == HASHES['APPROVAL'] and
                self.step35_sha256 == HASHES['STEP35'], 'Frozen approval/contract mismatch')
        require(len(self.upstream_sha256) == 64 and len(self.source_sha256) == 64,
                'Malformed provenance hashes')
        require(self.source_sha256 != SEALED_SHA, 'Real sealed digest barred from F22 fixture')


class ReservedReadJournal:
    """One attempt, hash chained. Synthetic tempdir only in F22.

    Irreversible reservation occurs *before* opening the fixture. Any failure
    leaves the reservation behind; caller MUST NOT retry against that journal.
    """
    EVENTS = ('RESERVED_BEFORE_OPEN', 'SOURCE_BYTES_VERIFIED')

    def __init__(self, path: Path, authority_digest: str):
        self.path = path
        self.authority_digest = authority_digest
        require(len(authority_digest) == 64, 'Invalid binding digest')

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        require(self.path.is_file() and not self.path.is_symlink(), 'Unsafe journal')
        raw = self.path.read_bytes()
        require(raw and raw.endswith(b'\n'), 'Truncated journal')
        out = []
        prev = '0' * 64
        for index, line in enumerate(raw.splitlines()):
            require(index < 2, 'Excess journal events')
            record = json.loads(line)
            require(set(record) == {'index', 'event', 'prev', 'authority_sha256',
                                     'source_sha256', 'bytes', 'event_sha256'},
                    'Unexpected journal schema')
            fields = {k: v for k, v in record.items() if k != 'event_sha256'}
            require(record['index'] == index and
                    record['event'] == self.EVENTS[index] and
                    record['prev'] == prev and
                    record['authority_sha256'] == self.authority_digest and
                    record['event_sha256'] == hexdigest(canonical(fields)),
                    'Journal order, authorization, or hash-chain violation')
            if index == 0:
                require(record['source_sha256'] is None and record['bytes'] is None,
                        'Reservation cannot claim a completed read')
            else:
                require(isinstance(record['source_sha256'], str) and
                        len(record['source_sha256']) == 64 and
                        isinstance(record['bytes'], int) and record['bytes'] > 0,
                        'Malformed verification evidence')
            prev = record['event_sha256']
            out.append(record)
        return out

    def _append(self, event: str, source_hash: str | None = None,
                byte_size: int | None = None) -> None:
        prev = self.entries()
        ix = len(prev)
        require(ix < 2 and event == self.EVENTS[ix], 'Attempt consumed or event out of order')
        data = {'index': ix, 'event': event,
                'prev': prev[-1]['event_sha256'] if prev else '0' * 64,
                'authority_sha256': self.authority_digest,
                'source_sha256': source_hash, 'bytes': byte_size}
        record = {**data, 'event_sha256': hexdigest(canonical(data))}
        flags = os.O_WRONLY | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
        if ix == 0:
            require(not self.path.exists(), 'Duplicate reservation')
            flags |= os.O_EXCL
        else:
            flags = os.O_WRONLY | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            blob = canonical(record) + b'\n'
            pos = 0
            while pos < len(blob):
                n = os.write(fd, blob[pos:])
                require(n > 0, 'Journal write stalled')
                pos += n
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_dir(self.path.parent)
        require(len(self.entries()) == ix + 1, 'Journal not durable')

    def reserve(self) -> None:
        self._append('RESERVED_BEFORE_OPEN')

    def verified(self, source_hash: str, byte_size: int) -> None:
        self._append('SOURCE_BYTES_VERIFIED', source_hash, byte_size)

    def final_digest(self) -> str:
        require(len(self.entries()) == 2, 'Source has not been verified')
        return file_sha(self.path)


def fixture_single_read_and_consume(fixture: Path,
                                    authority: SyntheticAuthorization,
                                    journal: ReservedReadJournal,
                                    consumer: Callable[[bytes, str, dict], object],
                                    *, inject_crash_before_open: bool = False):
    """Only synthetic fixture. Consumer is the stand-in for F20 Phase A.

    Includes byte-to-journal binding. No preliminary source open/checksum occurs.
    Produces no final-test metrics. A production routine needs new source review.
    """
    require(fixture.parent == journal.path.parent, 'Fixture and journal parent mismatch')
    require(not fixture.is_symlink(), 'Symlink fixture forbidden')
    authority.validate_fixture(fixture)
    require(journal.authority_digest == authority.digest(), 'Journal authority drift')
    journal.reserve()
    if inject_crash_before_open:
        raise RuntimeError('INJECTED_CRASH_AFTER_DURABLE_RESERVATION')
    fd = os.open(fixture, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) |
                 getattr(os, 'O_CLOEXEC', 0))
    try:
        a = os.fstat(fd)
        require(stat.S_ISREG(a.st_mode), 'Source must be regular')
        h = hashlib.sha256()
        chunks = []
        while True:
            buf = os.read(fd, 1 << 20)
            if not buf:
                break
            chunks.append(buf)
            h.update(buf)
        b = os.fstat(fd)
        require((a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns) ==
                (b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns),
                'Source changed during single read')
        blob = b''.join(chunks)
        require(len(blob) == a.st_size and h.hexdigest() == authority.source_sha256,
                'Source byte count/digest differs from authority')
    finally:
        os.close(fd)
    journal.verified(h.hexdigest(), len(blob))
    provenance = {'read_journal_sha256': journal.final_digest(),
                  'authorization_binding_sha256': authority.digest(),
                  'sealed_source_sha256': h.hexdigest(),
                  'verified_source_bytes': len(blob)}
    require(provenance['read_journal_sha256'] == file_sha(journal.path),
            'Journal evidence drift before consumer')
    return consumer(blob, provenance['read_journal_sha256'], provenance)


def deny_real_release(*args, **kwargs):
    raise PermissionError('F22 is a review-only fixture controller; third raw-test read NOT authorized')


def check_immutable_wsL() -> None:
    for name, path in [('F20', F20), ('F21', F21), ('APPROVAL', APPROVAL), ('STEP35', STEP35)]:
        assert_sha(path, HASHES[name], name)
    old = MARKER.read_bytes()
    require(not MARKER.is_symlink(), 'Unsafe historical ledger')
    record = json.loads(old)
    require(record['raw_test_filesystem_open_count_completed'] == 2 and
            record['final_test_scored'] is False, 'Historical access record unexpectedly changed')
    approval = json.loads(APPROVAL.read_bytes())
    gate = approval['remaining_review_gates']
    require(approval['record_status'] == 'APPROVED_BY_STUDY_OWNER_FOR_IMPLEMENTATION_AND_INDEPENDENT_REVIEW',
            'Amended owner-approved design absent')
    require(gate['additional_raw_test_read_authorized'] is False and
            gate['independent_methodological_signoff'] is False,
            'Original owner approval erroneously elevated to access/methodological signoff')
    require(MARKER.read_bytes() == old, 'Historical access ledger changed')


def portable_test() -> None:
    with tempfile.TemporaryDirectory(prefix='STEP36I_F22_SYNTHETIC_ONLY_') as tmp:
        work = Path(tmp)
        data = (b'SYNTHETIC_ONLY_LOADER_INPUT' * 131) + bytes(range(256))
        source = work / 'SYNTHETIC_ONLY_input.bin'
        source.write_bytes(data)
        auth = SyntheticAuthorization(hexdigest(data), source.name,
                     'c' * 64, HASHES['APPROVAL'], HASHES['STEP35'])
        log = ReservedReadJournal(work / 'SYNTHETIC_ONLY_access_journal.jsonl', auth.digest())
        calls = []

        def phase_a_consumer(verified_bytes, journal_digest, provenance):
            require(verified_bytes == data and hexdigest(verified_bytes) == auth.source_sha256,
                    'Phase A was not provided exact authenticated bytes')
            require(journal_digest == file_sha(log.path) and
                    provenance['authorization_binding_sha256'] == auth.digest(),
                    'Phase A was not bound to completed journal')
            calls.append(1)
            return 'SYNTHETIC_PHASE_A_EVIDENCE_ACCEPTED'

        ans = fixture_single_read_and_consume(source, auth, log, phase_a_consumer)
        require(ans == 'SYNTHETIC_PHASE_A_EVIDENCE_ACCEPTED' and calls == [1] and
                len(log.entries()) == 2, 'End-to-end fixture contract failed')
        for bad in [lambda: fixture_single_read_and_consume(source, auth, log, phase_a_consumer),
                    lambda: deny_real_release()]:
            try:
                bad()
            except (RuntimeError, PermissionError):
                pass
            else:
                raise RuntimeError('Replay/production denial failed')
        # Crash after reservation irrevocably consumes a DIFFERENT fixture slot.
        crashlog = ReservedReadJournal(work / 'SYNTHETIC_ONLY_failed_journal.jsonl', auth.digest())
        try:
            fixture_single_read_and_consume(source, auth, crashlog, phase_a_consumer,
                                            inject_crash_before_open=True)
        except RuntimeError as exc:
            require('INJECTED_CRASH' in str(exc), 'Unexpected injected error')
        else:
            raise RuntimeError('Synthetic crash was not raised')
        require(len(crashlog.entries()) == 1, 'Crash reservation missing')
        try:
            fixture_single_read_and_consume(source, auth, crashlog, phase_a_consumer)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Consumed crash reservation was reused')
        # Different bytes with same claimed digest fail closed, after reservation.
        fake = source.with_name('SYNTHETIC_ONLY_modified.bin')
        fake.write_bytes(data + b'EXTRA')
        forged_auth = SyntheticAuthorization(auth.source_sha256, fake.name,
                            auth.upstream_sha256, auth.owner_approval_sha256, auth.step35_sha256)
        forged_log = ReservedReadJournal(work / 'SYNTHETIC_ONLY_modified_journal.jsonl', forged_auth.digest())
        try:
            fixture_single_read_and_consume(fake, forged_auth, forged_log, phase_a_consumer)
        except RuntimeError as exc:
            require('Source byte count/digest' in str(exc), 'Mismatch not rejected for expected reason')
        else:
            raise RuntimeError('Changed source bytes were accepted')
        require(len(forged_log.entries()) == 1, 'Bad digest attempt was not consumed')
        # Tampered journal event cannot be used in resume.
        with log.path.open('ab') as out:
            out.write(b'{}\n')
        try:
            log.entries()
        except (RuntimeError, ValueError, KeyError):
            pass
        else:
            raise RuntimeError('Tampered journal was accepted')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--portable-self-test', action='store_true')
    group.add_argument('--offline-preflight', action='store_true')
    args = p.parse_args()
    print('=' * 99)
    print('STEP 36I-F22 — AUTHORITY/INPUT/JOURNAL BINDING AT PHASE A (REVIEW ONLY)')
    print('=' * 99)
    if args.offline_preflight:
        print('[1] FROZEN F20/F21/STEP35/OWNER APPROVAL AND HISTORICAL ACCESS')
        check_immutable_wsL()
        print('PASS — source hashes and frozen records verified; original marker remains 2 reads / 0 scored')
    else:
        print('[1] PORTABLE SOURCE / FIXTURE TEST')
    print('[2] CONCRETE BOUND SINGLE-OPEN READER -> JOURNAL -> PHASE A CONSUMER')
    portable_test()
    print('PASS — one synthetic source, completed journal digest and bytes supplied to same consumer')
    print('PASS — replay, post-reservation crash, source mismatch and journal tamper rejected')
    print('[3] EXPLICIT REMAINING RELEASE BOUNDARIES')
    print('OPEN — real source path / signed access decision and independent signoff not provided')
    print('OPEN — actual F20 materializer and live output mapping not connected in F22 fixture')
    print('OPEN — independently reviewed live journal/concurrency architecture and release CLI')
    print('=' * 99)
    print('F22 PASS — REVIEW-ONLY AUTHORITY/JOURNAL BOUNDARY; PRODUCTION ACCESS DENIED')
    print('Real held-out data opened: FALSE | GPU: FALSE | Final-test scored: FALSE')
    print('Third raw-test read authorized: FALSE | Production CLI: DENIED')
    print('=' * 99)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, PermissionError, ValueError, KeyError, OSError, AssertionError) as exc:
        print('F22 FAIL-CLOSED:', type(exc).__name__, str(exc), file=sys.stderr)
        raise SystemExit(1)
