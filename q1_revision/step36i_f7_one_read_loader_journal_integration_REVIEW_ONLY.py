#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F7: single-file-read / durable-journal integration, REVIEW ONLY.

Only supported commands (NO SEALED TEST ACCESS):
  python step36i_f7_one_read_loader_journal_integration_REVIEW_ONLY.py --self-test
  python step36i_f7_one_read_loader_journal_integration_REVIEW_ONLY.py --offline-preflight

This implements and independently tests the exact byte-level single-read
boundary and fail-closed reservation logic against disposable, synthetic files.
The sealed-test loader is deliberately NOT wired to a production CLI. The
function `open_sealed_test_production` is permanently denied here.

A different, independently reviewed and approved production source must connect
(1) the durable journal and single raw-file open, (2) the frozen in-memory
Step36 materialization, (3) all 40 frozen model inferences, (4) F1/F4 evaluation,
and (5) an isolated atomic artifact commit. F7 cannot grant a third raw read.

The historical access marker remains two reads / zero scored; never rewrite it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import zipfile

_FORBID = {"--execute", "--production", "--worker", "--read-test",
           "--score-test", "--resume", "--authorize", "--approve", "--run",
           "--commit-real", "--materialize", "--final-test", "--enable"}
if _FORBID.intersection(sys.argv[1:]):
    raise SystemExit("DENIED — Step 36I-F7 is an offline review script, not an authorization")

BASE = Path(__file__).resolve().parent
EXP = BASE.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
E2 = BASE / "step36i_e2_source_review_bundle.zip"
STEP35 = EXP / "manifests/35_final_test_evaluation_contract_freeze_manifest.json"
MARKER = EXP / "manifests/36_final_test_access_consumption_state.json"
SOURCE_HASHES = {
    "step36i_f1_amended_production_runner_REVIEW_ONLY.py": "e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556",
    "step36i_f2_independent_source_review_READ_ONLY.py": "ec6400f0f356b7a9888c443890b333f6fa6c6d80f6d29d700625eba4335f00bc",
    "step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py": "b7c28f536a57653f10cc74155b38941b7e2e86b508089e5a3378bf9ff97b7539",
    "step36i_f4_segment_stratified_inference_REVIEW_ONLY.py": "421cb3e8b83f96efaef1f3e0578e2cb7513c728aef36bda68cdc9216b07b4b10",
    "step36i_f5_independent_statistical_audit_READ_ONLY.py": "47a5ed8044493f4d09cbafa16fea5a62482d13b27856da945bd5d6e04f5e8ee3",
    "step36i_f6_end_to_end_synthetic_transaction_REVIEW_ONLY.py": "f075401a94c3bd2a971d4ade7c71c28207754d5c5810b40462e0b2bc88d62bad",
    "step36i_e2_source_review_bundle.zip": "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851",
}
STEP35_SHA = "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7"
SEALED_SHA = "ab325ede85fbb6d7819e85472274b82d151d350f39a872a17747c6978b1dddae"
CANDIDATES_SHA = "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65"
FROZEN_PLAN_SHA = "a2f8489933e8d2bbd11d6bab72361f7ad066dd37bc62e1db45bb0d20d87ce086"


def require(cond, reason):
    if not bool(cond):
        raise RuntimeError(reason)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(path, wanted, label):
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe {label}")
    require(file_digest(path) == wanted, f"SHA drift: {label}")


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class SingleReadEvidence:
    file_bytes: bytes
    sha256: str
    file_size: int
    inode: int


class OneAttemptJournal:
    """One durable reservation, irreversible after any interruption.

    Designed for a single writer, and TESTED ONLY in synthetic tempdir. Never
    construct this on the production experiment until independent source review
    and access authorization. A final release needs explicit concurrency locks
    and external archiving to protect against privileged file replacement.
    """

    EVENTS = ("RESERVED_BEFORE_OPEN", "BYTES_READ_VERIFIED")

    def __init__(self, path: Path):
        self.path = path

    def records(self):
        if not self.path.exists():
            return []
        require(self.path.is_file() and not self.path.is_symlink(), "Unsafe journal")
        raw = self.path.read_bytes()
        require(raw.endswith(b"\n"), "Truncated journal")
        prev = "0" * 64
        records = []
        for ix, line in enumerate(raw.splitlines()):
            rec = json.loads(line)
            require(set(rec) == {"event", "index", "prev", "sha256"}, "Invalid journal keys")
            require(rec["index"] == ix and rec["event"] == self.EVENTS[ix], "Invalid event order")
            require(rec["prev"] == prev, "Invalid journal hash chain")
            fields = {"event": rec["event"], "index": ix, "prev": prev}
            require(rec["sha256"] == digest(canonical(fields)), "Tampered journal record")
            prev = rec["sha256"]
            records.append(rec)
        require(0 < len(records) <= len(self.EVENTS), "Journal length invalid")
        return records

    def reserve_before_open(self):
        # O_EXCL prevents replacing an existing prior attempt or concurrent reservation.
        require(not self.path.exists() and not self.path.is_symlink(), "Access already reserved; no retry")
        fields = {"event": self.EVENTS[0], "index": 0, "prev": "0" * 64}
        record = {**fields, "sha256": digest(canonical(fields))}
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            payload = canonical(record) + b"\n"
            written = 0
            while written < len(payload):
                n = os.write(fd, payload[written:])
                require(n > 0, "Journal write did not progress")
                written += n
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(self.path.parent)
        require(len(self.records()) == 1, "Reservation not durable")

    def record_verified(self):
        recs = self.records()
        require(len(recs) == 1, "Attempt already consumed or not reserved")
        fields = {"event": self.EVENTS[1], "index": 1, "prev": recs[0]["sha256"]}
        entry = {**fields, "sha256": digest(canonical(fields))}
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
        try:
            payload = canonical(entry) + b"\n"
            written = 0
            while written < len(payload):
                n = os.write(fd, payload[written:])
                require(n > 0, "Append did not progress")
                written += n
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(self.path.parent)
        require(len(self.records()) == 2, "Verified-read record not durable")


def read_bytes_exactly_once_fixture(path: Path, expected_hash: str, journal: OneAttemptJournal,
                                    opener=os.open, inject_crash_after_reservation=False) -> SingleReadEvidence:
    """Single verified file-descriptor read. SYNTHETIC FIXTURE USE ONLY.

    We do not read a file for a preliminary hash, nor call pandas on a path;
    after verified read, a future release would parse bytes via io.BytesIO.
    Reservation is durable before os.open. Once reserved, any crash/error is
    irreversible; never reattempt under this journal.
    """
    # This is not a public real-data loader. Synthetic basename requirement is
    # intentional and prevents inadvertent use with the sealed test parquet.
    require(path.name.startswith("SYNTHETIC_ONLY_"), "Only synthetic source names permitted in F7")
    require(path.parent == journal.path.parent, "Source/journal fixture directory mismatch")
    require(len(expected_hash) == 64, "Invalid expected digest")
    journal.reserve_before_open()
    if inject_crash_after_reservation:
        raise RuntimeError("SYNTHETIC_INJECTED_CRASH_AFTER_RESERVATION")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = opener(path, flags)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), "Source must be a regular file")
        content = bytearray()
        h = hashlib.sha256()
        while True:
            b = os.read(fd, 4 * 1024 * 1024)
            if not b:
                break
            content.extend(b)
            h.update(b)
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                "Source changed during read")
        require(len(content) == before.st_size, "Unexpected byte count")
        observed = h.hexdigest()
        require(observed == expected_hash, "Sealed-style digest mismatch, access consumed")
    finally:
        os.close(fd)
    journal.record_verified()
    return SingleReadEvidence(bytes(content), observed, len(content), int(before.st_ino))


def open_sealed_test_production(*args, **kwargs):
    raise PermissionError("F7 never opens sealed final test; no third-read authorization")


def atomic_live_commit(*args, **kwargs):
    raise PermissionError("F7 never writes amended real outputs; no approved transaction")


def assert_denied(call, expected, term):
    try:
        call()
    except expected as exc:
        require(term.lower() in str(exc).lower(), f"Wrong rejection reason: {exc}")
    else:
        raise RuntimeError(f"Failed negative control: {term}")


def synthetic_one_read_tests():
    """Prove one FD open, BytesIO parsing, no retry after crash/hash failure."""
    with tempfile.TemporaryDirectory(prefix="STEP36I_F7_SYNTHETIC_ONLY_") as td:
        root = Path(td)
        payload = b"PAR1-SYNTHETIC_ONLY_BYTE_FIXTURE-PAR1" * 2048
        fixture = root / "SYNTHETIC_ONLY_mock.parquet"
        fixture.write_bytes(payload)
        opens = []

        def counted_open(path, flags):
            opens.append((path.name, flags))
            return os.open(path, flags)

        journal = OneAttemptJournal(root / "SYNTHETIC_ONLY_attempt.jsonl")
        ev = read_bytes_exactly_once_fixture(fixture, digest(payload), journal, opener=counted_open)
        require(ev.sha256 == digest(payload) and ev.file_bytes == payload, "Single-read data mismatch")
        require(len(opens) == 1 and len(journal.records()) == 2, "Not exactly one fixture open")
        # Simulate the future in-memory Parquet parse: BytesIO only; no filesystem open.
        require(io.BytesIO(ev.file_bytes).read() == payload, "In-memory parse source mismatch")
        require(len(opens) == 1, "In-memory consumer caused additional file open")
        assert_denied(lambda: read_bytes_exactly_once_fixture(fixture, digest(payload), journal, opener=counted_open),
                      RuntimeError, "no retry")
        require(len(opens) == 1, "Retry reached filesystem")

        crashed = OneAttemptJournal(root / "SYNTHETIC_ONLY_crash.jsonl")
        assert_denied(lambda: read_bytes_exactly_once_fixture(fixture, digest(payload), crashed,
                      opener=counted_open, inject_crash_after_reservation=True), RuntimeError, "injected_crash")
        require(len(crashed.records()) == 1 and len(opens) == 1, "Crash used an extra open")
        assert_denied(lambda: read_bytes_exactly_once_fixture(fixture, digest(payload), crashed,
                      opener=counted_open), RuntimeError, "no retry")

        incorrect = OneAttemptJournal(root / "SYNTHETIC_ONLY_bad_hash.jsonl")
        assert_denied(lambda: read_bytes_exactly_once_fixture(fixture, "0"*64, incorrect, opener=counted_open),
                      RuntimeError, "digest mismatch")
        require(len(incorrect.records()) == 1 and len(opens) == 2,
                "Hash failure must consume exactly one new attempt")
        assert_denied(lambda: read_bytes_exactly_once_fixture(fixture, digest(payload), incorrect, opener=counted_open),
                      RuntimeError, "no retry")

        link = root / "SYNTHETIC_ONLY_symlink.parquet"
        link.symlink_to(fixture)
        symlink_journal = OneAttemptJournal(root / "SYNTHETIC_ONLY_symlink.jsonl")
        assert_denied(lambda: read_bytes_exactly_once_fixture(link, digest(payload), symlink_journal,
                      opener=counted_open), OSError, "symbolic link")
        require(len(symlink_journal.records()) == 1, "Symlink failure did not consume reservation")

        # Tamper an already committed synthetic journal record (test-only).
        raw = journal.path.read_bytes()
        journal.path.write_bytes(raw.replace(b"BYTES_READ_VERIFIED", b"BYTES_READ_TAMPERED"))
        assert_denied(journal.records, RuntimeError, "event order")
    return len(opens)


def verify_project_read_only():
    require(EXP.is_dir(), "WSL experiment root missing")
    for name, expected in SOURCE_HASHES.items():
        check_file(BASE / name, expected, name)
    check_file(STEP35, STEP35_SHA, "Step35 original scientific contract")
    require(MARKER.is_file() and not MARKER.is_symlink(), "Historical access marker absent/unsafe")
    marker_before = file_digest(MARKER)
    marker = json.loads(MARKER.read_text("utf-8"))
    require(marker.get("state") == "RAW_TEST_REOPENED_FOR_TECHNICAL_RECOVERY" and
            marker.get("raw_test_filesystem_open_count_completed") == 2 and
            marker.get("final_test_scored") is False, "History not 2 reads / 0 scored")
    require(marker.get("expected_raw_test_sha256") == SEALED_SHA ==
            marker.get("observed_raw_test_sha256"), "Historic sealed digest mismatch")
    require(marker.get("step35_manifest_sha256") == STEP35_SHA, "Historical marker/Step35 relation drift")
    with zipfile.ZipFile(E2, "r") as z:
        require(z.testzip() is None and len(z.namelist()) == 18, "E2 bundle integrity failure")
        member_sha = {name: digest(z.read(name)) for name in (
            "candidate_keys_DRAFT.csv", "frozen_inference_plan.csv", "step35_contract.json")}
        require(member_sha == {
            "candidate_keys_DRAFT.csv": CANDIDATES_SHA,
            "frozen_inference_plan.csv": FROZEN_PLAN_SHA,
            "step35_contract.json": STEP35_SHA}, "Frozen E2 member drift")
        amendment = json.loads(z.read("amendment_review_DRAFT.json"))
        protocol = json.loads(z.read("protocol_deviation_DRAFT.json"))
    require(amendment.get("approval_granted") is False and
            amendment.get("third_raw_test_read_authorized") is False and
            protocol.get("status") == "DRAFT_NOT_APPROVED", "Approval unexpectedly changed")
    # No stat/open/read of the sealed parquet. Only frozen metadata are checked.
    isolated = [
        EXP / "feature_data/step36i_amended_final_test_arrays",
        EXP / "manifests/36i_amended_final_test_materialization_manifest.json",
        EXP / "final_test_evaluation/step36i_amended_predictions",
        EXP / "manifests/36i_amended_final_test_evaluation_manifest.json",
        EXP / "audits/36i_amended_final_test_metrics.csv",
        EXP / "manifests/36i_amended_final_test_access_journal.json",
        EXP / "audits/36i_amended_final_test_execution_failure.json",
    ]
    require(all(not p.exists() and not p.is_symlink() for p in isolated),
            "Amended output namespace occupied")
    assert_denied(open_sealed_test_production, PermissionError, "no third-read")
    assert_denied(atomic_live_commit, PermissionError, "no approved transaction")
    require(file_digest(MARKER) == marker_before, "Historical access marker mutated")
    return marker_before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--offline-preflight", action="store_true")
    args = parser.parse_args()
    print("="*110)
    print("STEP 36I-F7 — ONE-READ LOADER + DURABLE RESERVATION BOUNDARY, OFFLINE REVIEW")
    print("="*110)
    if args.offline_preflight:
        print("[1] F1–F6 / E2 source hashes, original Step35, access and vacant output namespace")
        verify_project_read_only()
        print("PASS — six reviewed implementation files, bundle and contract; 2 historic reads, 0 scored")
    else:
        print("[1] Portable fixture mode; no live project paths accessed")
    print("[2] Single-open byte loader, SHA-256 match, in-memory consumer and no retry")
    nopen = synthetic_one_read_tests()
    print(f"PASS — synthetic successful fixture exactly once; all fixture opens including negative cases={nopen}")
    print("[3] Durable access reservation, crash, replay, symlink, digest-tamper and journal integrity")
    print("PASS — reservation precedes file open; failures irreversibly consume fixture slot")
    print("[4] Production CLI and scientific amendment gates")
    assert_denied(open_sealed_test_production, PermissionError, "no third-read")
    assert_denied(atomic_live_commit, PermissionError, "no approved transaction")
    print("PASS — protected real loader and production commit are uncallable in this review build")
    print("="*110)
    print("STEP 36I-F7 PASS — READ-ONCE ACCESS BOUNDARY TESTED AGAINST SYNTHETIC BYTES ONLY")
    print("Real test opened: FALSE | GPU: FALSE | Real scored: FALSE | Source/marker modified: FALSE")
    print("Production CLI: DENIED | Protocol approved: FALSE | Third read authorized: FALSE")
    print("REMAINING — reviewed production launcher, real transaction integration, amendment review/approval")
    print("="*110)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"F7 FAIL-CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
