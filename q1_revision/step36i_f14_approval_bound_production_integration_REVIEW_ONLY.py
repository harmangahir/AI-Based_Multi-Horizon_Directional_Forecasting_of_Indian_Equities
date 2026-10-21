#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F14: approved protocol provenance + production integration boundary.

MODES: --self-test or --offline-preflight ONLY.

F14 links the study-owner F13E approval record (not a test-read permit),
F11's independently signed NON-TEST upstream contract, F9's preloaded
computational assembly, F7's single-read journal and F3's atomic no-replace
transaction primitives. It tests its byte-to-commit coordinator with ONLY
synthetic fixture bytes in a disposable TemporaryDirectory.

THIS IS NOT A REAL-TEST EXECUTABLE. There is no allowed production CLI path,
no third-read authority, and no real-dataset loader connected here.
Neither this file nor earlier frozen files should be amended to bypass guards.
Any later live release requires an independent source review, methodological
sign-off, a separate user authorization for a single additional test read,
and a new, explicitly approved production entrypoint.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile

PROHIBITED = {"--execute", "--production", "--worker", "--read-test",
              "--score-test", "--resume", "--authorize", "--approve", "--run",
              "--commit-real", "--final-test", "--enable", "--materialize",
              "--test-parquet"}
if PROHIBITED.intersection(sys.argv[1:]):
    raise SystemExit("DENIED: F14 approval-bound integration review never accesses real test")

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
MAN = EXP / "manifests"
APPROVAL_JSON = ROOT / "STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json"
APPROVAL_MD = ROOT / "STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.md"
SOURCE_FILES = {
    "F1": ("step36i_f1_amended_production_runner_REVIEW_ONLY.py", "e139455be2e788b2bb846f6e9de81a45e3a9bfca4260f0cea32e202c3adee556"),
    "F3": ("step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py", "b7c28f536a57653f10cc74155b38941b7e2e86b508089e5a3378bf9ff97b7539"),
    "F4": ("step36i_f4_segment_stratified_inference_REVIEW_ONLY.py", "421cb3e8b83f96efaef1f3e0578e2cb7513c728aef36bda68cdc9216b07b4b10"),
    "F7": ("step36i_f7_one_read_loader_journal_integration_REVIEW_ONLY.py", "b01bd3ce80194a4981a3c518fcf2e7f46948819715a51319843fa50630a54685"),
    "F9": ("step36i_f9_preloaded_production_assembly_REVIEW_ONLY.py", "8074cfe7cd1de9cf17609d4cef71aff27f910347da3ad86f3861a3cf7ad1373b"),
    "F10": ("step36i_f10_independent_assembly_review_READ_ONLY.py", "d778fe150b3a06f85992b8077937a8f4509a9f2c741f0d913433443a0d07a686"),
    "F11": ("step36i_f11_upstream_provenance_audit_READ_ONLY.py", "3405cb13a89d7c2403a66b5f6119b18a13c42f886d7f8d05aa80e94b09709ea6"),
}
APPROVAL_HASHES = {
    "json": "f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493",
    "md": "c53013b5ce6082356e1e2a3e3df137203eeb4ea4270510596255bcbb899f4d49",
}
STEP35_SHA = "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7"
LEDGER = MAN / "36_final_test_access_consumption_state.json"
STEP35 = MAN / "35_final_test_evaluation_contract_freeze_manifest.json"


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_file(path: Path, digest: str, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe {label}: {path}")
    require(sha_file(path) == digest, f"SHA-256 differs for {label}")


def import_checked(tag: str):
    path = ROOT / SOURCE_FILES[tag][0]
    name = f"step36i_f14_review_{tag.lower()}"
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Unable to import {tag}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def approved_design() -> dict:
    verify_file(APPROVAL_JSON, APPROVAL_HASHES["json"], "F13E JSON approval record")
    verify_file(APPROVAL_MD, APPROVAL_HASHES["md"], "F13E Markdown approval record")
    record = json.loads(APPROVAL_JSON.read_text("utf-8"))
    require(record["record_status"] == "APPROVED_BY_STUDY_OWNER_FOR_IMPLEMENTATION_AND_INDEPENDENT_REVIEW",
            "Expected study-owner amendment approval absent")
    require(record["immutable_original_scientific_contract_sha256"] == STEP35_SHA,
            "Step35 binding differs from approved protocol")
    cohort = record["approved_amended_cohort"]
    require((cohort["endpoints"], cohort["dates"], cohort["stocks"],
             cohort["segment_dates"]) == (5300, 265, 20, [48, 217]), "Approval cohort drift")
    require(record["approved_candidate_keys_sha256"] ==
            "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65",
            "Candidate cohort SHA drift")
    orig = record["original_cohort"]
    excl = record["excluded_cohort"]
    require((orig["endpoints"], orig["dates"], excl["endpoints"], excl["dates"]) ==
            (6480, 324, 1180, 59), "Original/amended delta changed")
    stats = record["approved_modified_statistics"]
    require(stats["primary_block_length_sessions"] == 20 and
            stats["sensitivity_block_lengths_sessions"] == [10, 40] and
            record["preserved"]["bootstrap_replicates"] == 2000 and
            record["preserved"]["primary_hypotheses"] == 24,
            "Scientific statistics no longer match approved F13E design")
    remain = record["remaining_review_gates"]
    require(remain["independent_methodological_signoff"] is False and
            remain["additional_raw_test_read_authorized"] is False and
            remain["real_final_test_scored"] is False,
            "F13E approval has been misrepresented as production permission")
    require(record["historical_raw_test_filesystem_reads"] == 2 and
            record["test_performance_scores_completed"] == 0,
            "Historical read/scoring account altered")
    return record


def verify_interface_contract():
    expected = {
        "F1": {"materialize_preloaded_from_frozen_helpers", "frozen_inference_from_preloaded_X", "in_memory_five_seed_scoring"},
        "F3": {"write_unique", "rename_dir_no_replace", "fsync_directory"},
        "F4": {"evaluate_in_memory"},
        "F7": {"OneAttemptJournal", "read_bytes_exactly_once_fixture"},
        "F9": {"assemble_from_preloaded", "_manifest"},
        "F11": {"verify_feature_map", "expected_non_test_paths"},
    }
    for tag, defs in expected.items():
        tree = ast.parse((ROOT / SOURCE_FILES[tag][0]).read_text("utf-8"))
        got = {n.name for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        require(defs <= got, f"F14 integration API drift for {tag}: {defs - got}")
    # No later release may treat the historical E2 DRAFT as itself approved.
    return True


def synthetic_byte_to_atomic_commit() -> None:
    """Integration of actual F7 and F3 primitives, with the F9 manifest builder.

    For production-like shape, payload content is written from the *verified*
    source bytes; no synthetic control may write into the experiment.
    """
    f7 = import_checked("F7")
    f3 = import_checked("F3")
    f9 = import_checked("F9")
    with tempfile.TemporaryDirectory(prefix="STEP36I_F14_SYNTHETIC_ONLY_") as td:
        root = Path(td)
        source = root / "SYNTHETIC_ONLY_input.bin"
        content = (b"STEP36I_F14_SYNTHETIC_ONLY_VERIFIED_BYTES" * 4096)
        source.write_bytes(content)
        wanted = hashlib.sha256(content).hexdigest()
        journal = f7.OneAttemptJournal(root / "SYNTHETIC_ONLY_one_read_journal.jsonl")
        opens = []
        def counting_open(path, flags):
            opens.append(str(path))
            return os.open(path, flags)
        ev = f7.read_bytes_exactly_once_fixture(source, wanted, journal, opener=counting_open)
        require(ev.file_bytes == content and len(opens) == 1 and len(journal.records()) == 2,
                "F7 read-once test failed")
        require(io.BytesIO(ev.file_bytes).getvalue() == content, "Verified bytes not used downstream")
        try:
            f7.read_bytes_exactly_once_fixture(source, wanted, journal, opener=counting_open)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Duplicate read succeeded")
        require(len(opens) == 1, "Duplicate read reached filesystem")

        payloads = {
            "synthetic_data/verified_bytes.bin": ev.file_bytes,
            "synthetic_data/test_summary.json": (json.dumps({
                "synthetic_only": True, "verified_source_sha256": wanted,
                "byte_count": len(ev.file_bytes)}, sort_keys=True) + "\n").encode(),
        }
        metadata = {"study_owner_approval_sha256": APPROVAL_HASHES["json"],
                    "independent_methodology_signed_off": False,
                    "third_read_authorized": False,
                    "synthetic_only": True}
        art_manifest = f9._manifest(payloads, metadata)
        payloads["artifact_manifest.json"] = (json.dumps(
            art_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        staging = root / ".SYNTHETIC_ONLY_STAGING"
        final = root / "SYNTHETIC_ONLY_COMMITTED"
        staging.mkdir(mode=0o700)
        for rel, data in sorted(payloads.items()):
            out = staging / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            f3.write_unique(out, data)
        f3.fsync_directory(staging / "synthetic_data")
        f3.fsync_directory(staging)
        f3.rename_dir_no_replace(staging, final)
        require(final.is_dir() and not staging.exists(), "Atomic rename not committed")
        for info in art_manifest["artifacts"]:
            out = final / info["relative_path"]
            require(out.stat().st_size == info["bytes"] and sha_file(out) == info["sha256"],
                    "Committed bytes differ from frozen manifest")
        require(json.loads((final / "artifact_manifest.json").read_text()) == art_manifest,
                "Manifest readback differs")
        require(len(art_manifest["artifacts"]) == 2, "Manifest artifact count changed")
        try:
            f3.rename_dir_no_replace(final, final)
        except Exception:
            pass
        else:
            raise RuntimeError("Duplicate commit not rejected")
        # Confirm that no local fixture includes an authentic test-dataset name.
        require(all("SYNTHETIC_ONLY" in p.name or p.name in
                    {"verified_bytes.bin", "test_summary.json", "artifact_manifest.json", "synthetic_data"}
                    for p in root.iterdir()), "Unexpected fixture artifact")
    return None


def prohibit_real_execution(*args, **kwargs):
    raise PermissionError("DENIED: F14 is review-only; no third test read or real output commit")


open_real_test = prohibit_real_execution
run_real_inference = prohibit_real_execution
commit_real_artifacts = prohibit_real_execution


def run(offline_preflight: bool) -> None:
    print("=" * 108)
    print("STEP 36I-F14 — APPROVAL-BOUND INTEGRATION, READ-ONLY / SYNTHETIC")
    print("=" * 108)
    print("[1] F13E STUDY-OWNER APPROVAL AND INDEPENDENT REVIEW BOUNDARIES")
    approved_design()
    print("PASS — both F13E records verified; 5300/265, 48/217, 2000 blocks, primary L20; NO access grant")
    print("[2] VERIFIED F1/F3/F4/F7/F9/F10/F11 SOURCES AND UPSTREAM APIS")
    for tag, (fn, sha) in SOURCE_FILES.items():
        verify_file(ROOT / fn, sha, tag)
    verify_interface_contract()
    f11 = import_checked("F11")
    snap35, snap14, pre = f11.load_bundle()
    f11.verify_feature_map(snap35, snap14)
    paths = f11.expected_non_test_paths(snap35, pre)
    require(len(paths) == 3, "Upstream input inventory changed")
    print("PASS — frozen source hashes/interfaces; 63-feature + 20-stock non-test input contract")
    print("[3] F7 READ-ONCE BYTES -> F9 MANIFEST -> F3 ATOMIC COMMIT, SYNTHETIC")
    synthetic_byte_to_atomic_commit()
    for fn in (open_real_test, run_real_inference, commit_real_artifacts):
        try:
            fn()
        except PermissionError:
            pass
        else:
            raise RuntimeError("Real-data boundary opened in review source")
    print("PASS — one synthetic source read; journal reservation; no-replace atomic commit; replay denied")
    if offline_preflight:
        print("[4] WSL APPROVAL RECORD / FROZEN DEVELOPMENT / HISTORICAL ACCESS IMMUTABILITY")
        verify_file(STEP35, STEP35_SHA, "original Step35 contract")
        require(json.loads(STEP35.read_text("utf-8")) == snap35,
                "Step35 live differs from E2 archived contract")
        before = LEDGER.read_bytes()
        state = json.loads(before)
        require(state.get("raw_test_filesystem_open_count_completed") == 2 and
                state.get("final_test_scored") is False, "Historical access state changed")
        for name, (path, digest, _) in paths.items():
            f11.check_file(Path(path), digest, name)
        f1 = import_checked("F1")
        f1.confirm_legacy_output_isolation()
        require(LEDGER.read_bytes() == before, "Historical access marker modified during preflight")
        print("PASS — three signed development artifacts verified; 2 historic reads/0 scoring; 7 output paths vacant")
    else:
        print("[4] PORTABLE SELF-TEST ONLY — no WSL experiment resources accessed")
    print("[5] REMAINING PRODUCTION GATES")
    print("OPEN — independent methodological sign-off; production release source review")
    print("OPEN — actual real-data one-read journal + isolated transaction launcher not enabled in this build")
    print("OPEN — explicitly authorized third raw-test read, as a separate future decision")
    print("=" * 108)
    print("STEP 36I-F14 PASS — APPROVAL-BOUND INTEGRATION REVIEW ONLY; PRODUCTION DENIED")
    print("Real test opened: FALSE | Test scored: FALSE | GPU: FALSE | Historical marker unchanged")
    print("Study-owner amended design: APPROVED | Independent sign-off: PENDING")
    print("Third read authorized: FALSE | Live production CLI: DENIED")
    print("=" * 108)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--self-test", action="store_true")
    options.add_argument("--offline-preflight", action="store_true")
    parsed = parser.parse_args()
    run(offline_preflight=parsed.offline_preflight)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError, AssertionError) as e:
        print(f"STEP 36I-F14 FAIL-CLOSED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
