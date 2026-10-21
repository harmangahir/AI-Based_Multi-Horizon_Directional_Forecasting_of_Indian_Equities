#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F15 — durability boundary between test materialization and GPU inference.

ONLY allowed modes: --self-test and --offline-preflight. Neither reads test data.
This is a review prototype for a future *separate* authorized production release.
It requires no real labels or model inference. No third access is authorized.

Why F15 exists: F9's all-in-one computation would lose its in-memory sequence
materialization if GPU inference failed. Here we test a durable, verified,
no-overwrite array handoff BEFORE any GPU work so a later run could resume from
an independently committed materialization rather than reopening the test.

Not implemented by this file: actual sealed test loader, authorization, GPU
release, or permission to persist real test labels. The eventual live source
MUST include independent approval and detailed security/access review.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile

DENIED = {"--execute", "--run", "--production", "--resume", "--read-test", "--score-test",
          "--real", "--authorize", "--approve", "--materialize-test", "--open-test",
          "--commit-real", "--final-test"}
if any(arg in DENIED for arg in sys.argv[1:]):
    raise SystemExit("DENIED: F15 never accesses or persists real held-out test data")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
APPROVAL = ROOT / "STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json"
APPROVAL_SHA = "f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493"
F14 = ROOT / "step36i_f14_approval_bound_production_integration_REVIEW_ONLY.py"
F14_SHA = "bc34722a3582812c0d5257581e04e320bd2e59eb67f4a037a4f473c4238f068f"
F3 = ROOT / "step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py"
F3_SHA = "b7c28f536a57653f10cc74155b38941b7e2e86b508089e5a3378bf9ff97b7539"
STEP35 = EXP / "manifests/35_final_test_evaluation_contract_freeze_manifest.json"
STEP35_SHA = "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7"
ACCESS = EXP / "manifests/36_final_test_access_consumption_state.json"
CANDIDATE = EXP / "audits/step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv"
CANDIDATE_SHA = "4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65"
TARGET_REVIEW_ONLY = EXP / "feature_data/step36i_amended_final_test_arrays"


def require(ok: bool, why: str):
    if not ok:
        raise RuntimeError(why)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(4 << 20), b""):
            sha.update(part)
    return sha.hexdigest()


def checked(path: Path, expected: str, label: str):
    require(path.is_file() and not path.is_symlink(), f"Missing or unsafe {label}: {path}")
    require(file_sha(path) == expected, f"SHA-256 mismatch for {label}")


def check_live_review_inputs():
    for path, sha, label in [(APPROVAL, APPROVAL_SHA, "study-owner approval"),
                             (F14, F14_SHA, "F14 approval-bound assembly"),
                             (F3, F3_SHA, "F3 atomic-commit primitive"),
                             (STEP35, STEP35_SHA, "original Step35 contract"),
                             (CANDIDATE, CANDIDATE_SHA, "candidate cohort")]:
        checked(path, sha, label)
    record = json.loads(APPROVAL.read_text())
    require(record["record_status"] == "APPROVED_BY_STUDY_OWNER_FOR_IMPLEMENTATION_AND_INDEPENDENT_REVIEW",
            "Study-owner amendment record differs")
    require(record["remaining_review_gates"]["independent_methodological_signoff"] is False and
            record["remaining_review_gates"]["additional_raw_test_read_authorized"] is False,
            "F15 must not infer access authorization")
    before = ACCESS.read_bytes()
    access = json.loads(before)
    require(access["raw_test_filesystem_open_count_completed"] == 2 and
            access["final_test_scored"] is False, "Historical raw-test access mismatch")
    require(not TARGET_REVIEW_ONLY.exists() and not TARGET_REVIEW_ONLY.is_symlink(),
            "Amended real materialization namespace not vacant")
    return before


def synth_keys(frame: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["Ticker", "Date", "market_session_id", "continuity_segment_id"]
    require(set(key_cols) <= set(frame.columns), "Signed candidate structural columns missing")
    out = frame[key_cols].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out = out.reset_index(drop=True)
    require(out.equals(out.sort_values(["Date", "Ticker"]).reset_index(drop=True)),
            "Frozen physical endpoint order changed; do not silently sort sequences")
    require(len(out) == 5300 and out.Date.nunique() == 265 and
            out.Ticker.nunique() == 20 and not out.duplicated().any(), "Cohort structural cardinality mismatch")
    require(out.groupby("Date").Ticker.nunique().eq(20).all(), "Unbalanced market-date clusters")
    seg = out.drop_duplicates("Date").continuity_segment_id.astype(int).tolist()
    require(seg == [0] * 48 + [1] * 217, "Cohort segment drift")
    return out


def bytes_npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def bytes_metadata_csv(frame: pd.DataFrame) -> bytes:
    """Stable no-pyarrow structural metadata serialization for isolated handoff."""
    csv_frame = frame.copy()
    csv_frame["Date"] = pd.to_datetime(csv_frame["Date"]).dt.strftime("%Y-%m-%d")
    return csv_frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def package_arrays(X: np.ndarray, y: np.ndarray, keys: pd.DataFrame,
                   *, upstream_manifest_hash: str, synthetic_only: bool) -> dict[str, bytes]:
    """Pure in-memory serializer, never a test-data reader."""
    # This review build has an unremovable synthetic-only constraint. Real
    # materialization must be handled by a separately audited release source.
    require(synthetic_only is True, "F15 only serializes synthetic fixtures")
    require(isinstance(X, np.ndarray) and X.shape == (5300, 60, 63) and
            X.dtype == np.float32 and np.isfinite(X).all(), "Invalid X array")
    require(isinstance(y, np.ndarray) and y.shape == (5300, 3) and
            y.dtype == np.int8 and np.isin(y, (0, 1)).all(), "Invalid synthetic labels")
    keys = synth_keys(keys)
    require(len(upstream_manifest_hash) == 64, "Upstream provenance hash absent")
    contents = {
        "arrays/sequence_X.npy": bytes_npy(X),
        "arrays/directions_SYNTHETIC_ONLY.npy": bytes_npy(y),
        "arrays/metadata_keys.csv": bytes_metadata_csv(keys),
    }
    items = [{"relative_path": name, "bytes": len(data), "sha256": digest(data)}
             for name, data in sorted(contents.items())]
    manifest = {
        "schema": "STEP36I_F15_DURABLE_ARRAY_HANDOFF_SYNTHETIC_ONLY_V1",
        "synthetic_only": True,
        "frozen_original_step35_sha256": STEP35_SHA,
        "study_owner_approval_sha256": APPROVAL_SHA,
        "approved_candidate_keys_sha256": CANDIDATE_SHA,
        "upstream_manifest_hash": upstream_manifest_hash,
        "shape_X": [5300, 60, 63], "shape_y": [5300, 3],
        "date_clusters": 265, "stocks_per_date": 20, "segments": [48, 217],
        "read_sealed_test": False, "compute_real_labels": False,
        "produced_gpu_inference": False, "third_raw_test_access_authorized": False,
        "artifacts": items,
    }
    contents["materialization_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return contents


def import_f3():
    spec = importlib.util.spec_from_file_location("step36i_f15_f3_checked", F3)
    require(spec is not None and spec.loader is not None, "F3 import unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_name(rel: str):
    path = PurePosixPath(rel)
    require(bool(rel) and not path.is_absolute() and ".." not in path.parts and
            all(p not in ("", ".") for p in path.parts) and
            not rel.startswith("/") and "\\" not in rel,
            f"Artifact traversal/invalid name rejected: {rel}")


def commit_fixture(payload: dict[str, bytes], root: Path, name: str, f3) -> Path:
    require(name.startswith("SYNTHETIC_ONLY_") and "/" not in name, "Unsafe fixture namespace")
    require(root.is_dir() and not root.is_symlink(), "Unsafe fixture root")
    staged = root / (".SYNTHETIC_ONLY_STAGE_" + name)
    target = root / name
    require(not staged.exists() and not target.exists(), "No-overwrite guard")
    staged.mkdir(mode=0o700)
    for rel, data in sorted(payload.items()):
        safe_name(rel)
        require(isinstance(data, bytes), "Nonbytes artifact")
        dest = staged / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        f3.write_unique(dest, data)
    f3.fsync_directory(staged / "arrays")
    f3.fsync_directory(staged)
    f3.rename_dir_no_replace(staged, target)
    return target


def validate_saved_handoff(committed: Path, expected_upstream: str):
    path = committed / "materialization_manifest.json"
    manifest = json.loads(path.read_text())
    require(manifest["synthetic_only"] is True and
            manifest["upstream_manifest_hash"] == expected_upstream and
            manifest["study_owner_approval_sha256"] == APPROVAL_SHA,
            "Materialization seal/provenance mismatch")
    require(len(manifest["artifacts"]) == 3, "Materialization artifact count changed")
    for item in manifest["artifacts"]:
        safe_name(item["relative_path"])
        f = committed / item["relative_path"]
        require(f.is_file() and not f.is_symlink() and
                f.stat().st_size == item["bytes"] and file_sha(f) == item["sha256"],
                "Committed materialization byte integrity failure")
    X = np.load(committed / "arrays/sequence_X.npy", allow_pickle=False, mmap_mode="r")
    y = np.load(committed / "arrays/directions_SYNTHETIC_ONLY.npy", allow_pickle=False)
    keys = pd.read_csv(committed / "arrays/metadata_keys.csv")
    require(X.shape == (5300, 60, 63) and X.dtype == np.float32 and
            y.shape == (5300, 3) and y.dtype == np.int8, "Read-back shape/dtype mismatch")
    synth_keys(keys)
    return manifest


def run_synthetic_handoff(keys: pd.DataFrame):
    # The only bytes written by this program are in a disposable TemporaryDirectory.
    # 5300x60x63 is full-sized; no actual market feature values or labels are used.
    X = np.zeros((5300, 60, 63), dtype=np.float32)
    y = np.zeros((5300, 3), dtype=np.int8)
    y[::3, :] = 1
    upstream = digest(b"STEP36I_F15_VERIFIED_SYNTHETIC_UPSTREAM_ONLY")
    payload = package_arrays(X, y, keys, upstream_manifest_hash=upstream, synthetic_only=True)
    require(len(payload) == 4, "Unexpected materialization files")
    f3 = import_f3()
    with tempfile.TemporaryDirectory(prefix="STEP36I_F15_SYNTHETIC_ONLY_") as tmp:
        root = Path(tmp)
        saved = commit_fixture(payload, root, "SYNTHETIC_ONLY_SEALED_MATERIALIZATION", f3)
        manifest = validate_saved_handoff(saved, upstream)
        try:
            commit_fixture(payload, root, "SYNTHETIC_ONLY_SEALED_MATERIALIZATION", f3)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Duplicate commit incorrectly allowed")
        # Negative control: a changed upstream identity cannot be loaded.
        try:
            validate_saved_handoff(saved, "0" * 64)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Changed upstream digest incorrectly accepted")
        try:
            safe_name("../escape.npy")
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Path traversal incorrectly accepted")
        return sum(item["bytes"] for item in manifest["artifacts"])


def never_execute_real(*args, **kwargs):
    raise PermissionError("DENIED: F15 is synthetic-only, no live release or additional raw-test access")


open_real_final_test = never_execute_real
commit_real_materialization = never_execute_real
resume_real_inference = never_execute_real


def main():
    p = argparse.ArgumentParser(description=__doc__)
    choices = p.add_mutually_exclusive_group(required=True)
    choices.add_argument("--self-test", action="store_true")
    choices.add_argument("--offline-preflight", action="store_true")
    opt = p.parse_args()
    print("=" * 107)
    print("STEP 36I-F15 — DURABLE PRE-INFERENCE MATERIALIZATION HANDOFF (SYNTHETIC ONLY)")
    print("=" * 107)
    print("[1] TEST/PRODUCTION DENIAL AND SEPARATE MATERIALIZATION CONTRACT")
    require(all(call is never_execute_real for call in
                (open_real_final_test, commit_real_materialization, resume_real_inference)),
            "Real production execution accidentally enabled")
    before = None
    if opt.offline_preflight:
        before = check_live_review_inputs()
        print("PASS — F13E/F14/F3/Step35/candidate frozen hashes; 2 reads/0 scored; real output vacant")
        keys = synth_keys(pd.read_csv(CANDIDATE))
    else:
        # Portable smoke test uses signed keys from E2 bundle, without WSL data.
        import zipfile
        bundle = ROOT / "step36i_e2_source_review_bundle.zip"
        checked(bundle, "9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851",
                "E2 source review bundle")
        with zipfile.ZipFile(bundle) as z:
            keys = synth_keys(pd.read_csv(io.BytesIO(z.read("candidate_keys_DRAFT.csv"))))
        checked(F3, F3_SHA, "F3 atomic primitive")
        print("PASS — signed E2 candidate keys, production entrypoints denied")
    print("[2] FULL-SIZED SYNTHETIC ARRAY PERSISTENCE AND INDEPENDENT READ-BACK")
    result = run_synthetic_handoff(keys)
    print(f"PASS — 5300x60x63 float32 sequences + 5300x3 synthetic labels + exact keys, {result:,} artifact bytes")
    print("[3] ATOMIC COMMIT, REHASH, DUPLICATE-REPLAY AND PROVENANCE NEGATIVE CONTROLS")
    print("PASS — durable temporary fixture commit, byte verification, no-replace, provenance and traversal denial")
    print("[4] LIVE ACCESS AND LEGACY IMMUTABILITY")
    if opt.offline_preflight:
        require(ACCESS.read_bytes() == before, "Historical test marker modified")
        require(not TARGET_REVIEW_ONLY.exists(), "Real output namespace changed")
        print("PASS — historical marker and seven-path real output namespace unchanged")
    print("=" * 107)
    print("STEP 36I-F15 PASS — SYNTHETIC DURABLE ARRAY HANDOFF ONLY")
    print("Raw final test opened: FALSE | Real GPU inference: FALSE | Real test scored: FALSE")
    print("Independent methodological review: PENDING | Third access authorized: FALSE | Production: DENIED")
    print("NEXT — independently review the two-phase production materialization/inference release")
    print("=" * 107)


if __name__ == "__main__":
    main()
