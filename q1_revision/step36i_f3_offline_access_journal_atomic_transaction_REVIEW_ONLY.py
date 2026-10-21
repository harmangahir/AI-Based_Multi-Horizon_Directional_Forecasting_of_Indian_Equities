#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F3 — READ-ONLY production-safety design, offline transaction tests.

Required invocation:
    python step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py --offline-preflight

This script does NOT implement a real raw-test loader, authorize data access,
score labels, run a GPU, or write production artifacts.  It checks frozen
source/access history, tests journal/transaction mechanics only in a temporary
SYNTHETIC fixture directory, and denies all production commands.

A future *different*, independently reviewed, formally approved source release
must implement raw-file access, coordinate a durable journal, and commit real
artifacts. Nothing here authorizes the third read or modifies the existing
Step35/Step36 contracts or access marker.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid
import zipfile

# Deny executable paths before importing any project module or test-file API.
DENIED = {"--execute", "--production", "--worker", "--authorize",
          "--read-test", "--score-test", "--resume", "--commit-real",
          "--approve", "--enable", "--run", "--materialize", "--final-test"}
if DENIED.intersection(sys.argv[1:]):
    raise SystemExit("DENIED — F3 is a synthetic-only safety design; no real test access")

BASE = Path(__file__).resolve().parent
EXP = BASE.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
MAN = EXP / "manifests"
AUD = EXP / "audits"
FEATURE = EXP / "feature_data"
F1 = BASE / "step36i_f1_amended_production_runner_REVIEW_ONLY.py"
F2 = BASE / "step36i_f2_independent_source_review_READ_ONLY.py"
E2 = BASE / "step36i_e2_source_review_bundle.zip"
ACCESS = MAN / "36_final_test_access_consumption_state.json"
STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"

HASHES = {
    "F1": "e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556",
    "F2": "ec6400f0f356b7a9888c443890b333f6fa6c6d80f6d29d700625eba4335f00bc",
    "E2": "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851",
    "STEP35": "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7",
    "COHORT": "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
}
EXPECTED_TEST_SHA = "ab325ede85fbb6d7819e85472274b82d151d350f39a872a17747c6978b1dddae"

# Frozen paths are inspection-only. Do not create any of these locations.
ISOLATED = {
    "array_dir": FEATURE / "step36i_amended_final_test_arrays",
    "array_manifest": MAN / "36i_amended_final_test_materialization_manifest.json",
    "predictions": EXP / "final_test_evaluation/step36i_amended_predictions",
    "evaluation_manifest": MAN / "36i_amended_final_test_evaluation_manifest.json",
    "metrics": AUD / "36i_amended_final_test_metrics.csv",
    "independent_access_journal": MAN / "36i_amended_final_test_access_journal.json",
    "failure_evidence": AUD / "36i_amended_final_test_execution_failure.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_file(path: Path, wanted: str, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe {label}: {path}")
    require(sha_file(path) == wanted, f"SHA drift for {label}")


def deny_real_access(marker: dict, amendment: dict, review: dict) -> None:
    """Production permission model for the *current* unapproved state: always deny.

    This is *not* an authorization facility. Synthetic journal events below use
    different names and cannot be promoted into permission records.
    """
    require(marker.get("raw_test_filesystem_open_count_completed") == 2,
            "Historical read counter unexpected; stop")
    require(marker.get("final_test_scored") is False, "Test scoring state unexpected")
    require(amendment.get("approval_granted") is False, "Amendment approval status changed")
    require(amendment.get("third_raw_test_read_authorized") is False,
            "Third-read authorization status changed")
    require(review.get("status") == "DRAFT_NOT_APPROVED", "Protocol review drift")
    raise PermissionError("DENIED — no approved amendment, no new raw-test read")


def assert_raises(call, expected: type[Exception], contains: str) -> None:
    try:
        call()
    except expected as exc:
        require(contains.lower() in str(exc).lower(), f"Wrong exception reason: {exc}")
    else:
        raise RuntimeError(f"Negative control unexpectedly succeeded: {contains}")


def canonical(event: dict) -> bytes:
    return json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fsync_directory(directory: Path) -> None:
    fd = os.open(str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class FixtureJournal:
    """Hash-chained, append-only JSONL. STRICTLY synthetic tempdir use only.

    A future real implementation needs single-writer serialization/locking,
    tamper-evident external archival and an independently audited state machine.
    """

    EVENTS = ("SYNTHETIC_INIT", "SYNTHETIC_READ_SLOT_RESERVED",
              "SYNTHETIC_READ_SIMULATED", "SYNTHETIC_COMMITTED")

    def __init__(self, path: Path):
        self.path = path

    def read_events(self) -> list[dict]:
        if not self.path.exists():
            return []
        require(self.path.is_file() and not self.path.is_symlink(), "Unsafe synthetic journal")
        events: list[dict] = []
        prev = "0" * 64
        for idx, line in enumerate(self.path.read_bytes().splitlines()):
            rec = json.loads(line)
            require(set(rec) == {"index", "event", "prev", "sha256"}, "Journal schema drift")
            require(rec["index"] == idx and rec["prev"] == prev, "Journal chain/index broken")
            require(rec["event"] in self.EVENTS, "Unknown journal event")
            digest = sha_bytes(canonical({"index": idx, "event": rec["event"], "prev": prev}))
            require(rec["sha256"] == digest, "Journal hash tamper detected")
            prev = digest
            events.append(rec)
        require(self.path.read_bytes().endswith(b"\n") if events else True,
                "Truncated journal record")
        require([e["event"] for e in events] == list(self.EVENTS[:len(events)]),
                "Journal state-machine ordering invalid")
        return events

    def append(self, event: str) -> None:
        existing = self.read_events()
        require(len(existing) < len(self.EVENTS), "Synthetic journal already committed")
        require(event == self.EVENTS[len(existing)], "Duplicate/out-of-order read permit denied")
        previous = existing[-1]["sha256"] if existing else "0" * 64
        fields = {"index": len(existing), "event": event, "prev": previous}
        record = {**fields, "sha256": sha_bytes(canonical(fields))}
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        if not existing:
            flags |= os.O_EXCL
        fd = os.open(self.path, flags, 0o600)
        try:
            encoded = canonical(record) + b"\n"
            written = 0
            while written < len(encoded):
                written += os.write(fd, encoded[written:])
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(self.path.parent)
        self.read_events()


def rename_dir_no_replace(src: Path, dst: Path) -> None:
    """Linux renameat2(RENAME_NOREPLACE), fail closed if unavailable.

    Tested only in a TemporaryDirectory; NOT wired to any real-data commit.
    """
    require(src.is_dir() and not src.is_symlink(), "Unsafe staging directory")
    require(src.parent == dst.parent, "Staging and commit must share a filesystem parent")
    require(not dst.exists() and not dst.is_symlink(), "Existing target; overwrite prohibited")
    lib = ctypes.CDLL(None, use_errno=True)
    fn = getattr(lib, "renameat2", None)
    require(fn is not None, "renameat2 unavailable; fail closed")
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    AT_FDCWD, RENAME_NOREPLACE = -100, 1
    rc = fn(AT_FDCWD, os.fsencode(src), AT_FDCWD, os.fsencode(dst), RENAME_NOREPLACE)
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(dst))
    fsync_directory(dst.parent)


def write_unique(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        size = 0
        while size < len(data):
            size += os.write(fd, data[size:])
        os.fsync(fd)
    finally:
        os.close(fd)


def synthetic_commit(base: Path, target_name: str, inject_failure: bool = False) -> tuple[Path, Path]:
    """Emulate an atomic transaction using ONLY synthetic bytes under tmp."""
    require(target_name.startswith("SYNTHETIC_ONLY_"), "Only synthetic transaction names allowed")
    require(base.resolve() != EXP.resolve() and not str(base.resolve()).startswith(str(EXP.resolve()) + "/"),
            "Synthetic target must not reside in the live experiment")
    final = base / target_name
    stage = base / (".SYNTHETIC_ONLY_PARTIAL_" + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    data = {"SYNTHETIC_ONLY_predictions.csv": b"n,p\n1,0.25\n2,0.75\n",
            "SYNTHETIC_ONLY_metrics.json": b'{"synthetic_only":true,"n":2}\n'}
    items = []
    for name, payload in sorted(data.items()):
        path = stage / name
        write_unique(path, payload)
        items.append({"relative_path": name, "bytes": len(payload), "sha256": sha_bytes(payload)})
    manifest = {"synthetic_only": True, "transaction_schema": "F3_DRAFT_V1", "artifacts": items}
    write_unique(stage / "SYNTHETIC_ONLY_manifest.json", canonical(manifest) + b"\n")
    fsync_directory(stage)
    if inject_failure:
        raise RuntimeError("SYNTHETIC_INJECTED_FAILURE_BEFORE_COMMIT")
    rename_dir_no_replace(stage, final)
    require(final.is_dir() and not stage.exists(), "Atomic rename verification failed")
    for rec in manifest["artifacts"]:
        p = final / rec["relative_path"]
        require(p.stat().st_size == rec["bytes"] and sha_file(p) == rec["sha256"],
                "Post-commit fixture rehash failure")
    require(json.loads((final / "SYNTHETIC_ONLY_manifest.json").read_text()) == manifest,
            "Committed manifest differs")
    return stage, final


def offline_test_transaction() -> None:
    with tempfile.TemporaryDirectory(prefix="step36i_f3_SYNTHETIC_ONLY_") as scratch:
        t = Path(scratch)
        journal = FixtureJournal(t / "SYNTHETIC_ONLY_journal.jsonl")
        journal.append("SYNTHETIC_INIT")
        journal.append("SYNTHETIC_READ_SLOT_RESERVED")
        assert_raises(lambda: journal.append("SYNTHETIC_READ_SLOT_RESERVED"),
                      RuntimeError, "duplicate/out-of-order")
        journal.append("SYNTHETIC_READ_SIMULATED")
        assert_raises(lambda: journal.append("SYNTHETIC_READ_SLOT_RESERVED"),
                      RuntimeError, "duplicate/out-of-order")
        try:
            synthetic_commit(t, "SYNTHETIC_ONLY_FAILED", inject_failure=True)
        except RuntimeError as exc:
            require("SYNTHETIC_INJECTED_FAILURE_BEFORE_COMMIT" in str(exc),
                    "Unexpected injection failure")
        else:
            raise RuntimeError("Failure injection did not stop transaction")
        require(not (t / "SYNTHETIC_ONLY_FAILED").exists(),
                "Failed transaction unexpectedly published")
        require(len(list(t.glob(".SYNTHETIC_ONLY_PARTIAL_*"))) == 1,
                "Abandoned partial transaction fixture missing")
        # The abandoned partial is not promoted or resumed; this is deliberate.
        _, committed = synthetic_commit(t, "SYNTHETIC_ONLY_COMMITTED")
        manifest_before = sha_file(committed / "SYNTHETIC_ONLY_manifest.json")
        assert_raises(lambda: synthetic_commit(t, "SYNTHETIC_ONLY_COMMITTED"),
                      RuntimeError, "existing target")
        require(sha_file(committed / "SYNTHETIC_ONLY_manifest.json") == manifest_before,
                "Replay/overwrite mutated committed artifact")
        journal.append("SYNTHETIC_COMMITTED")
        assert_raises(lambda: journal.append("SYNTHETIC_COMMITTED"),
                      RuntimeError, "already committed")
        raw = journal.path.read_bytes()
        journal.path.write_bytes(raw.replace(b"SYNTHETIC_READ_SIMULATED", b"SYNTHETIC_READ_FALSIFIED"))
        assert_raises(lambda: journal.read_events(), RuntimeError, "unknown journal event")


def main() -> None:
    parser = argparse.ArgumentParser(description="F3 OFFLINE safety design only; no production mode")
    parser.add_argument("--offline-preflight", action="store_true", required=True)
    args = parser.parse_args()
    require(args.offline_preflight, "Offline-only mode required")
    require(BASE.is_dir() and EXP.is_dir(), "Expected WSL project layout missing")
    print("=" * 110)
    print("STEP 36I-F3 — OFFLINE ACCESS-CONTROL AND SYNTHETIC ATOMIC TRANSACTION DESIGN")
    print("=" * 110)
    print("[1] FROZEN F1/F2/E2 SOURCES AND ORIGINAL SCIENTIFIC CONTRACT")
    for label, path in [("F1", F1), ("F2", F2), ("E2", E2), ("STEP35", STEP35)]:
        checked_file(path, HASHES[label], label)
    print("PASS — F1, F2, E2 and Step35 SHA-256 identities match")
    print("[2] HISTORICAL ACCESS MARKER AND EXPLICIT UNAPPROVED STATUS")
    require(ACCESS.is_file() and not ACCESS.is_symlink(), "Access marker missing/unsafe")
    access_before = sha_file(ACCESS)
    marker = json.loads(ACCESS.read_text("utf-8"))
    require(marker.get("state") == "RAW_TEST_REOPENED_FOR_TECHNICAL_RECOVERY", "Access-state drift")
    require(marker.get("raw_test_filesystem_open_count_completed") == 2 and
            marker.get("final_test_scored") is False, "Raw-test read/scoring history mismatch")
    require(marker.get("expected_raw_test_sha256") == EXPECTED_TEST_SHA and
            marker.get("observed_raw_test_sha256") == EXPECTED_TEST_SHA,
            "Sealed test digest provenance changed")
    require(marker.get("step35_manifest_sha256") == HASHES["STEP35"], "Step35 marker link drift")
    with zipfile.ZipFile(E2) as bundle:
        require(bundle.testzip() is None, "E2 bundle damaged")
        review = json.loads(bundle.read("protocol_deviation_DRAFT.json"))
        amend = json.loads(bundle.read("amendment_review_DRAFT.json"))
        require(sha_bytes(bundle.read("candidate_keys_DRAFT.csv")) == HASHES["COHORT"],
                "Frozen candidate whitelist mismatch")
    assert_raises(lambda: deny_real_access(marker, amend, review),
                  PermissionError, "no approved amendment")
    for name, path in ISOLATED.items():
        require(not path.exists() and not path.is_symlink(),
                f"Real output namespace is not vacant ({name}): {path}")
    print("PASS — 2 historical reads, no scoring; approval/read gate denied; 7 real output paths untouched")
    print("[3] SYNTHETIC DURABLE JOURNAL STATE MACHINE AND NEGATIVE CONTROLS")
    print("[4] SYNTHETIC ATOMIC-NO-REPLACE COMMIT / CRASH / REPLAY CONTROLS")
    offline_test_transaction()
    print("PASS — synthetic journal enforces one reservation, order, hash chain, tamper and replay rejection")
    print("PASS — synthetic-only fsync, no-replace commit, failure injection, post-commit rehash")
    require(sha_file(ACCESS) == access_before, "Historical access marker mutated")
    print("[5] IMMUTABILITY AND IMPLEMENTATION BOUNDARIES")
    print("PASS — historical access marker unchanged; only ephemeral synthetic transaction exercised")
    print("REMAINING — production data loader/authorization/review not implemented")
    print("REMAINING — production durable journal + atomic artifact integration not implemented")
    print("REMAINING — segment-aware bootstrap/FDR real-data integration not implemented")
    print("=" * 110)
    print("STEP 36I-F3: PASS — OFFLINE ACCESS-JOURNAL/TRANSACTION DESIGN TEST ONLY")
    print("Raw test opened: FALSE | GPU: FALSE | Real test scored: FALSE")
    print("Amendment approved: FALSE | Third read authorized: FALSE | Production CLI: DENIED")
    print("=" * 110)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"STEP 36I-F3 FAIL-CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
