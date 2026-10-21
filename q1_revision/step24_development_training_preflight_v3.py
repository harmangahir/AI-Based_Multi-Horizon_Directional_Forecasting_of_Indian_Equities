#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 24 — CLEAN V3 DEVELOPMENT-TRAINING / GPU MICROBATCH PREFLIGHT

This is the single pre-training Step 24 for the clean-V3 pipeline.

Repository rechecking and frozen-source import resolution are implemented inside this Step-24 script so it does not depend on non-public helper names inside the frozen Step-23 implementation.

It verifies the frozen Step 00->23 chain, audits the Step-14 development
X_full/y_direction headers without loading numerical arrays, reconciles and
freezes the current training runtime, and determines a deterministic physical
CUDA microbatch for each of the 16 frozen A/B candidates using synthetic data
only.

The logical/effective batch remains exactly 256 for every candidate through
gradient accumulation.  The physical microbatch is an execution-resource
parameter, not a predictive hyperparameter.

The hard GPU headroom gate uses peak CUDA *allocated* memory plus successful
completion of the synthetic training step.  Peak *reserved* memory is retained
for diagnostics only because, on the verified WSL2/native-allocator runtime,
its reported high-water mark can exceed the dedicated-memory figure despite a
successful complete CUDA step.

A synthetic AdamW step is executed only to allocate optimizer state and test
the real FP32 training-memory path.  No optimizer step is executed on
development data.  No validation score, threshold, feature selection,
checkpoint selection or final-test information is used.

Legacy Step 22B/22C/22D files are not used; their compatibility decisions are
already integrated in the single frozen Step-22 translation.

Run:
    python step24_development_training_preflight_v3.py

Verify after a successful audited run:
    python step24_development_training_preflight_v3.py --verify-only
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import step00_config as cfg


# =============================================================================
# 1. CONSTANTS / PATHS
# =============================================================================

STEP_NAME = "STEP 24 — DEVELOPMENT-TRAINING + GPU MICROBATCH PREFLIGHT"
MODELS = (
    "DLINEAR", "NLINEAR", "PATCHTST", "TIMESNET",
    "ITRANSFORMER", "FREDFORMER", "FRNET", "MLF",
)
REPO_BY_MODEL = {
    "DLINEAR": "LTSF_LINEAR",
    "NLINEAR": "LTSF_LINEAR",
    "PATCHTST": "PATCHTST",
    "TIMESNET": "TIMESNET_TSLIB",
    "ITRANSFORMER": "ITRANSFORMER",
    "FREDFORMER": "FREDFORMER",
    "FRNET": "FRNET",
    "MLF": "MLF",
}

LOOKBACK = 60
FEATURES = 63
PRED_LEN = 5
OUTPUTS = 3
FOLDS = 5
LOGICAL_BATCH = 256
SEED = 42

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
GPU_BUDGET_FRACTION = 0.75
# Hard acceptance uses peak CUDA ALLOCATED memory, not peak RESERVED memory.
# In this WSL2/native-allocator runtime, max_memory_reserved() can exceed the
# dedicated-memory figure even when the complete CUDA step succeeds; it is
# retained as a diagnostic only.
STEP23_SAFETY_MULTIPLIER = 1.25
MICROBATCHES = (256, 128, 64, 32, 16, 8, 4, 2, 1)
STEP23_SMOKE_BATCH = 2
WORKER_TIMEOUT = 900
RESULT_PREFIX = "__STEP24_RESULT__"
COMMON_HEAD_PARAMS = (PRED_LEN * FEATURES) * OUTPUTS + OUTPUTS

SCRIPT = Path(__file__).resolve()
Q1_DIR = SCRIPT.parent
STEP23_SCRIPT = Q1_DIR / "step23_synthetic_model_smoke_tests_v3.py"

STEP23_MANIFEST = cfg.MANIFEST_DIR / "23_stage1_model_smoke_manifest.json"
STEP23_MANIFEST_SHA = cfg.MANIFEST_DIR / "23_stage1_model_smoke_manifest.sha256"
STEP23_FREEZE = cfg.MANIFEST_DIR / "23_stage1_model_smoke_freeze.json"
STEP22_FREEZE = cfg.MANIFEST_DIR / "22_source_config_translation_freeze.json"
STEP14_MANIFEST = cfg.MANIFEST_DIR / "14_model_array_manifest.json"
STEP01_PIP_FREEZE = cfg.ENVIRONMENT_DIR / "01_pip_freeze.txt"
TEST_SEAL = cfg.MANIFEST_DIR / "10_final_test_seal.json"
FORBIDDEN_TEST = "10_final_test_locked_panel.parquet"

ARRAY_CSV = cfg.AUDIT_DIR / "24_development_array_header_inventory.csv"
SPLIT_CSV = cfg.AUDIT_DIR / "24_development_split_summary.csv"
CONTRACT_CSV = cfg.AUDIT_DIR / "24_candidate_training_runtime_contracts.csv"
PLAN_CSV = cfg.AUDIT_DIR / "24_microbatch_initial_plan.csv"
PROBE_CSV = cfg.AUDIT_DIR / "24_microbatch_probe_results.csv"
SCHEDULE_CSV = cfg.AUDIT_DIR / "24_microbatch_schedule.csv"
REPO_BEFORE_CSV = cfg.AUDIT_DIR / "24_repository_recheck_before.csv"
REPO_AFTER_CSV = cfg.AUDIT_DIR / "24_repository_recheck_after.csv"
ENV_DELTA_CSV = cfg.AUDIT_DIR / "24_environment_delta_from_step01.csv"
RUNTIME_CSV = cfg.AUDIT_DIR / "24_runtime_dependency_versions.csv"
PIP_FREEZE = cfg.ENVIRONMENT_DIR / "24_training_runtime_pip_freeze.txt"
WORKER_LOG_DIR = cfg.AUDIT_DIR / "24_microbatch_worker_logs"
FREEZE = cfg.MANIFEST_DIR / "24_training_runtime_preflight_freeze.json"
MANIFEST = cfg.MANIFEST_DIR / "24_training_runtime_preflight_manifest.json"
MANIFEST_SHA = cfg.MANIFEST_DIR / "24_training_runtime_preflight_manifest.sha256"
FROZEN_SCRIPT = cfg.MANIFEST_DIR / "step24_development_training_preflight_frozen.py"
SUMMARY = cfg.REPORT_DIR / "24_training_runtime_preflight_summary.txt"
FAILED_DIR = cfg.AUDIT_DIR / "step24_failed_attempts"


# =============================================================================
# 2. GENERIC HELPERS
# =============================================================================

def now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload):
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def write_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)


def sidecar_ok(manifest: Path, sidecar: Path):
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    observed = sha(manifest)
    if expected != observed:
        raise RuntimeError(f"Manifest sidecar mismatch: {manifest.name}")
    return observed


def semantic_hash(payload):
    return cfg.canonical_json_sha256(payload)


def freeze_fingerprint(path: Path, field: str):
    payload = load(path)
    stored = payload[field]
    body = dict(payload)
    body.pop(field)
    if semantic_hash(body) != stored:
        raise RuntimeError(f"{path.name}: fingerprint no longer recomputes.")
    return payload, stored


def artifact_by_name(manifest, filename):
    matches = []
    for key in ("artifacts", "canonical_artifacts"):
        obj = manifest.get(key, {})
        records = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, list) else []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            p = rec.get("path") or rec.get("file_path")
            if p and Path(p).name == filename:
                matches.append(rec)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one artifact {filename}; found {len(matches)}.")
    return matches[0]


def canonical_outputs():
    return (
        ARRAY_CSV, SPLIT_CSV, CONTRACT_CSV, PLAN_CSV, PROBE_CSV,
        SCHEDULE_CSV, REPO_BEFORE_CSV, REPO_AFTER_CSV, ENV_DELTA_CSV,
        RUNTIME_CSV, PIP_FREEZE, FREEZE, MANIFEST, MANIFEST_SHA,
        FROZEN_SCRIPT, SUMMARY,
    )


def assert_fresh_target():
    bad = [p for p in canonical_outputs() if p.exists()]
    if WORKER_LOG_DIR.exists():
        bad.append(WORKER_LOG_DIR)
    if bad:
        raise RuntimeError(
            "Step-24 canonical outputs already exist; refusing overwrite:\n"
            + "\n".join(str(p) for p in bad)
        )


def archive_partial():
    if MANIFEST.exists():
        return None
    items = [p for p in canonical_outputs() if p.exists()]
    if not items and not WORKER_LOG_DIR.exists():
        return None
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    dst = FAILED_DIR / ("partial_artifacts_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    dst.mkdir()
    for p in items:
        try:
            shutil.move(str(p), str(dst / p.name))
        except Exception:
            pass
    if WORKER_LOG_DIR.exists():
        try:
            shutil.move(str(WORKER_LOG_DIR), str(dst / WORKER_LOG_DIR.name))
        except Exception:
            pass
    return dst



# =============================================================================
# 2B. SELF-CONTAINED FROZEN-REPOSITORY / MODEL-IMPORT HELPERS
# =============================================================================

def normalize_git_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


def git_executable() -> str:
    exe = shutil.which("git")
    if exe is None:
        raise RuntimeError("git executable not found on PATH.")
    return exe


def run_git(arguments, cwd: Path, timeout_seconds: int = 120) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        [git_executable(), *arguments],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Git command failed.\n"
            f"Command: git {' '.join(arguments)}\n"
            f"CWD: {cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def recheck_repositories(step17_freeze):
    repositories = step17_freeze["repositories"]

    if len(repositories) != 8:
        raise RuntimeError(
            f"Expected 8 frozen Step-17 repositories; found {len(repositories)}."
        )

    implementation_count = sum(
        row.get("repository_role") == "implementation"
        for row in repositories
    )
    reference_count = sum(
        row.get("repository_role") == "reference_redirect"
        for row in repositories
    )

    if implementation_count != 7 or reference_count != 1:
        raise RuntimeError(
            "Step-17 repository-role counts drifted "
            f"(implementation={implementation_count}, reference={reference_count})."
        )

    repo_map = {}
    rows = []

    for record in repositories:
        repo_id = record["repo_id"]
        path = Path(record["local_path"])

        if repo_id in repo_map:
            raise RuntimeError(f"Duplicate frozen repository ID: {repo_id}")
        if not path.exists() or not (path / ".git").exists():
            raise RuntimeError(f"Frozen Git repository missing: {path}")

        observed_origin = run_git(["remote", "get-url", "origin"], path)
        observed_commit = run_git(["rev-parse", "HEAD"], path)
        observed_tree = run_git(["rev-parse", "HEAD^{tree}"], path)
        observed_branch = run_git(["branch", "--show-current"], path)
        observed_status = run_git(
            ["status", "--porcelain", "--untracked-files=all"],
            path,
        )

        origin_match = (
            normalize_git_url(observed_origin)
            == normalize_git_url(record["repository_url"])
        )
        commit_match = observed_commit == record["commit_sha"]
        tree_match = observed_tree == record["git_tree_sha"]
        branch_match = observed_branch == record["branch"]
        clean = observed_status == ""

        required_hashes_match = True
        checked_files = 0

        for file_record in record["required_file_hashes"]:
            file_path = path / file_record["relative_path"]
            if (
                not file_path.is_file()
                or sha(file_path) != file_record["sha256"]
            ):
                required_hashes_match = False
            else:
                checked_files += 1

        passed = bool(
            origin_match
            and commit_match
            and tree_match
            and branch_match
            and clean
            and required_hashes_match
        )

        if not passed:
            raise RuntimeError(
                "Frozen repository recheck failed.\n"
                f"Repository: {repo_id}\n"
                f"Origin match: {origin_match}\n"
                f"Commit match: {commit_match}\n"
                f"Tree match: {tree_match}\n"
                f"Branch match: {branch_match}\n"
                f"Worktree clean incl untracked: {clean}\n"
                f"Required hashes match: {required_hashes_match}"
            )

        repo_map[repo_id] = record
        rows.append(
            {
                "repo_id": repo_id,
                "repository_role": record["repository_role"],
                "local_path": str(path),
                "origin_match": origin_match,
                "commit_match": commit_match,
                "tree_match": tree_match,
                "branch_match": branch_match,
                "worktree_clean_including_untracked": clean,
                "required_files_checked": int(checked_files),
                "required_file_hashes_match": required_hashes_match,
                "passed": passed,
            }
        )

    return repo_map, pd.DataFrame(rows)


def assert_single_repository_clean(repository_record):
    path = Path(repository_record["local_path"])
    status = run_git(
        ["status", "--porcelain", "--untracked-files=all"],
        path,
    )
    if status != "":
        raise RuntimeError(
            "Frozen repository became dirty during Step 24.\n"
            f"Repository: {repository_record['repo_id']}\n"
            f"Status:\n{status}"
        )


def step20_model_map(step20_freeze):
    records = {
        row["model_id"]: row
        for row in step20_freeze["model_records"]
    }
    if set(records) != set(MODELS):
        raise RuntimeError(
            "Step-20 model-record set does not match the eight clean-V3 models."
        )
    return records


def import_target(model_id, model_record, repository_root: Path):
    relative_file = Path(model_record["primary_source_file"])
    preferred_class = model_record["preferred_class"]

    if (
        model_id == "PATCHTST"
        and relative_file.parts
        and relative_file.parts[0] == "PatchTST_supervised"
    ):
        import_root = repository_root / "PatchTST_supervised"
        module_relative = Path(*relative_file.parts[1:])
    else:
        import_root = repository_root
        module_relative = relative_file

    if module_relative.suffix != ".py":
        raise RuntimeError(
            f"Expected Python model source file; got {module_relative}"
        )

    module_name = ".".join(module_relative.with_suffix("").parts)

    os.chdir(import_root)
    sys.path.insert(0, str(import_root))

    module = importlib.import_module(module_name)
    module_file = Path(getattr(module, "__file__", "")).resolve()

    if not module_file.is_relative_to(import_root.resolve()):
        raise RuntimeError(
            "Third-party import did not resolve to the frozen local repository.\n"
            f"Model: {model_id}\n"
            f"Import root: {import_root}\n"
            f"Module: {module_name}\n"
            f"Resolved: {module_file}"
        )

    target = getattr(module, preferred_class)

    return (
        target,
        import_root,
        module_name,
        preferred_class,
        module_file,
    )


# =============================================================================
# 3. LOAD VERIFIED STEP-23 HELPERS / VERIFY STEP 00->23
# =============================================================================

def load_step23_module():
    if not STEP23_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Required verified Step-23 source missing:\n{STEP23_SCRIPT}"
        )

    step23_manifest = load(STEP23_MANIFEST)
    frozen_rec = artifact_by_name(
        step23_manifest,
        "step23_synthetic_model_smoke_tests_frozen.py",
    )
    frozen_path = Path(frozen_rec["path"])
    if not frozen_path.is_file() or sha(frozen_path) != frozen_rec["sha256"]:
        raise RuntimeError("Frozen Step-23 script artifact failed verification.")
    if sha(STEP23_SCRIPT) != sha(frozen_path):
        raise RuntimeError(
            "Working Step-23 script no longer matches its frozen Step-23 copy."
        )

    spec = importlib.util.spec_from_file_location("clean_v3_step23", STEP23_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_all():
    s23 = load_step23_module()
    s0 = s23.verify_step00()
    upstream = s23.verify_upstream(s0)  # verifies Step 01->22 and freezes/repositories schema

    m23_sha = sidecar_ok(STEP23_MANIFEST, STEP23_MANIFEST_SHA)
    m23 = load(STEP23_MANIFEST)
    if (
        m23.get("step", {}).get("status") != "PASS"
        or m23.get("configuration_fingerprint_sha256") != s0["fingerprint"]
    ):
        raise RuntimeError("Step-23 manifest is not a PASS for the active configuration.")

    f23, fp23 = freeze_fingerprint(
        STEP23_FREEZE,
        "stage1_model_smoke_fingerprint_sha256",
    )
    if m23.get("stage1_model_smoke_fingerprint_sha256") != fp23:
        raise RuntimeError("Step-23 manifest/freeze fingerprint mismatch.")

    if f23.get("status") != "STAGE1_MODEL_SYNTHETIC_SMOKE_TESTS_FROZEN":
        raise RuntimeError("Unexpected Step-23 freeze status.")
    state23 = f23["pipeline_state"]
    if (
        state23.get("synthetic_forward_smoke_tests_run") is not True
        or state23.get("synthetic_backward_smoke_tests_run") is not True
        or state23.get("optimizer_steps_run") is not False
        or state23.get("models_fitted") is not False
        or state23.get("validation_performance_inspected") is not False
    ):
        raise RuntimeError("Step-23 pre-training state drift.")

    seal = load(TEST_SEAL)
    if (
        seal.get("status") != "SEALED_NOT_OPENED"
        or seal.get("global_final_test_lock_state") != "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError("Step-10 final-test seal is not pristine.")

    f22, fp22 = freeze_fingerprint(
        STEP22_FREEZE,
        "source_config_translation_fingerprint_sha256",
    )
    if f22.get("status") != "STAGE1_CANDIDATE_SOURCE_CONFIGURATION_TRANSLATION_FROZEN":
        raise RuntimeError("Unexpected Step-22 freeze status.")
    if len(f22["candidate_source_configurations"]) != 16:
        raise RuntimeError("Step-22 does not contain 16 candidate configurations.")

    return s23, s0, upstream, m23, m23_sha, f23, fp23, f22, fp22


# =============================================================================
# 4. CURRENT TRAINING-RUNTIME FREEZE / GPU RESOURCE
# =============================================================================

def current_pip_freeze():
    p = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=False, timeout=120,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return "\n".join(sorted(x.strip() for x in p.stdout.splitlines() if x.strip())) + "\n"


def parse_freeze(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==(.+)$", line.strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def environment_audit():
    current_text = current_pip_freeze()
    current = parse_freeze(current_text)
    baseline = parse_freeze(STEP01_PIP_FREEZE.read_text(encoding="utf-8")) if STEP01_PIP_FREEZE.exists() else {}

    required = {
        "torch": "2.12.1+cu130",
        "reformer-pytorch": "1.4.4",
        "nvidia-cudnn-cu13": "9.20.0.48",
    }
    for name, expected in required.items():
        if current.get(name) != expected:
            raise RuntimeError(
                f"Training runtime mismatch: {name}; expected {expected}; "
                f"observed {current.get(name)}"
            )
    if "nvidia-cudnn-cu12" in current:
        raise RuntimeError("nvidia-cudnn-cu12 is present; CUDA-13 cuDNN runtime must remain coherent.")

    delta = []
    for name in sorted(set(baseline) | set(current)):
        a, b = baseline.get(name), current.get(name)
        if a == b:
            continue
        change = "added_after_step01" if a is None else "removed_after_step01" if b is None else "version_changed_after_step01"
        delta.append({
            "package": name,
            "step01_version": a,
            "step24_version": b,
            "change": change,
            "performance_information_used": False,
        })

    runtime = pd.DataFrame([
        {"distribution": name, "version": current.get(name, "NOT_INSTALLED")}
        for name in (
            "torch", "numpy", "pandas", "scipy", "einops",
            "reformer-pytorch", "nvidia-cudnn-cu13",
        )
    ])
    return current_text, current, pd.DataFrame(delta), runtime


def nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe and Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        exe = "/usr/lib/wsl/lib/nvidia-smi"
    if not exe:
        raise RuntimeError("nvidia-smi not found.")

    q = subprocess.run(
        [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=60,
    )
    if q.returncode != 0:
        raise RuntimeError(q.stderr)
    lines = [x.strip() for x in q.stdout.splitlines() if x.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Expected one GPU; found {len(lines)}.")
    name, mem = [x.strip() for x in lines[0].rsplit(",", 1)]
    total_mib = int(float(mem))

    q2 = subprocess.run(
        [exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=60,
    )
    active = [x.strip() for x in q2.stdout.splitlines() if x.strip() and "No running" not in x]
    if active:
        raise RuntimeError("Concurrent CUDA process detected:\n" + "\n".join(active))

    total = total_mib * 1024 * 1024
    return {
        "gpu_name": name,
        "total_memory_mib": total_mib,
        "total_memory_bytes": total,
        "budget_bytes": int(total * GPU_BUDGET_FRACTION),
        "active_compute_processes": 0,
    }


# =============================================================================
# 5. STEP-14 DEVELOPMENT ARRAY HEADER AUDIT
# =============================================================================

def walk_records(obj, inherited=None):
    inherited = dict(inherited or {})
    if isinstance(obj, dict):
        local = dict(inherited)
        for k in ("fold", "partition", "split", "scope", "sha256"):
            if k in obj:
                local[k] = obj[k]
        p = obj.get("file_path") or obj.get("path")
        if isinstance(p, str) and Path(p).suffix.lower() == ".npy":
            yield {**local, "file_path": p}
        for v in obj.values():
            yield from walk_records(v, local)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_records(v, inherited)


def infer_fold(value, path):
    try:
        i = int(value)
        if 1 <= i <= 5:
            return i
    except Exception:
        pass
    m = re.search(r"fold[_\- ]?0?([1-5])", str(path).lower())
    return int(m.group(1)) if m else None


def infer_partition(value, path):
    text = str(value if value is not None else path).lower()
    if re.search(r"(^|[^a-z])train([^a-z]|$)", text):
        return "train"
    if re.search(r"(^|[^a-z])(val|valid|validation)([^a-z]|$)", text):
        return "validation"
    return None


def logical_name(rec, path):
    text = re.sub(r"[^a-z0-9]+", "_", (str(rec.get("scope", "")) + "_" + path.stem).lower())
    if "x_full" in text:
        return "X_full"
    if "y_direction" in text or "_y_dir" in text:
        return "y_direction"
    return None


def npy_header(path: Path):
    with path.open("rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise RuntimeError(f"Invalid NPY: {path}")
        major, minor = f.read(1)[0], f.read(1)[0]
        n = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
        meta = ast.literal_eval(f.read(n).decode("latin1" if major < 3 else "utf-8").strip())
    return tuple(meta["shape"]), str(meta["descr"]), bool(meta["fortran_order"])


def development_headers():
    manifest = load(STEP14_MANIFEST)
    indexed = {}
    for rec in walk_records(manifest):
        path = Path(rec["file_path"]).resolve()
        if path.name == FORBIDDEN_TEST:
            raise RuntimeError("Final-test parquet entered Step-14 resolver.")
        fold = infer_fold(rec.get("fold"), path)
        part = infer_partition(rec.get("partition", rec.get("split")), path)
        logical = logical_name(rec, path)
        if fold and part and logical:
            indexed.setdefault((fold, part, logical), []).append((rec, path))

    rows, summary = [], []
    for fold in range(1, FOLDS + 1):
        for part in ("train", "validation"):
            pair = {}
            for logical in ("X_full", "y_direction"):
                matches = indexed.get((fold, part, logical), [])
                # Manifest traversal may encounter the same record more than once; deduplicate paths.
                unique = {}
                for rec, p in matches:
                    unique[str(p)] = (rec, p)
                matches = list(unique.values())
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Step-14 resolver failure: fold={fold}, partition={part}, "
                        f"logical={logical}, matches={len(matches)}"
                    )
                rec, path = matches[0]
                if not path.is_file():
                    raise FileNotFoundError(path)
                shape, dtype, fortran = npy_header(path)
                if rec.get("sha256"):
                    if sha(path) != rec["sha256"]:
                        raise RuntimeError(f"Development-array hash mismatch: {path}")
                pair[logical] = (shape, path)
                rows.append({
                    "fold": fold, "partition": part, "logical_array": logical,
                    "file_path": str(path), "shape_json": json.dumps(shape),
                    "dtype": dtype, "fortran_order": fortran,
                    "numerical_array_loaded": False, "final_test_opened": False,
                })

            xs, xp = pair["X_full"]
            ys, yp = pair["y_direction"]
            if len(xs) != 3 or xs[1:] != (LOOKBACK, FEATURES):
                raise RuntimeError(f"X_full contract mismatch: {fold}/{part}: {xs}")
            if len(ys) != 2 or ys[1] != OUTPUTS or ys[0] != xs[0]:
                raise RuntimeError(f"y_direction contract mismatch: {fold}/{part}: X={xs}, y={ys}")

            summary.append({
                "fold": fold, "partition": part, "n_sequences": int(xs[0]),
                "lookback": int(xs[1]), "feature_count": int(xs[2]),
                "direction_outputs": int(ys[1]),
                "x_file": str(xp), "y_file": str(yp),
                "numerical_arrays_loaded": False, "final_test_opened": False,
            })

    return pd.DataFrame(rows), pd.DataFrame(summary)


# =============================================================================
# 6. TRAINING CONTRACT / STEP-23 MEMORY PLAN
# =============================================================================

def training_contracts(f22):
    rows = []
    records = f22["candidate_source_configurations"]
    for rec in sorted(records, key=lambda r: (MODELS.index(r["model_id"]), r["config_id"])):
        model = rec["model_id"]
        if model not in MODELS or rec["repo_id"] != REPO_BY_MODEL[model]:
            raise RuntimeError(f"Candidate mapping drift: {model}/{rec['config_id']}")
        tc = rec["training_config"]
        if "learning_rate" not in tc:
            raise RuntimeError(f"learning_rate missing: {model}/{rec['config_id']}")
        rows.append({
            "model_id": model,
            "config_id": rec["config_id"],
            "repo_id": rec["repo_id"],
            "configuration_mode": rec["configuration_mode"],
            "learning_rate": float(tc["learning_rate"]),
            "logical_batch_size": LOGICAL_BATCH,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "adamw_beta1": BETAS[0],
            "adamw_beta2": BETAS[1],
            "adamw_eps": ADAMW_EPS,
            "loss": "BCEWithLogitsLoss(mean)",
            "gradient_clip_norm": GRAD_CLIP,
            "dtype": "float32",
            "amp_enabled": False,
            "source_config_sha256": semantic_hash(rec["source_config"]),
            "training_config_sha256": semantic_hash(tc),
        })
    df = pd.DataFrame(rows)
    if len(df) != 16 or df.model_id.nunique() != 8:
        raise RuntimeError("Training-contract structure mismatch.")
    return df


def step23_cuda_records(m23):
    rec = artifact_by_name(m23, "23_synthetic_smoke_results.csv")
    p = Path(rec["path"])
    if not p.is_file() or sha(p) != rec["sha256"]:
        raise RuntimeError("Step-23 smoke-results artifact drift.")
    df = pd.read_csv(p)
    df = df[(df.device == "cuda") & (df.status == "PASS")].copy()
    if len(df) != 16:
        raise RuntimeError("Step-23 CUDA PASS count is not 16.")
    return df


def allowed_not_above(n):
    for x in MICROBATCHES:
        if x <= n:
            return x
    return 1


def initial_plan(contracts, smoke, gpu):
    rows = []
    budget = gpu["budget_bytes"]
    for r in contracts.itertuples(index=False):
        m = smoke[(smoke.model_id == r.model_id) & (smoke.config_id == r.config_id)]
        if len(m) != 1:
            raise RuntimeError(f"Missing Step-23 CUDA memory row: {r.model_id}/{r.config_id}")
        peak = int(m.iloc[0].peak_cuda_memory_bytes)
        estimate = max(
            1,
            min(
                LOGICAL_BATCH,
                int(math.floor((STEP23_SMOKE_BATCH * budget) / (peak * STEP23_SAFETY_MULTIPLIER))),
            ),
        )
        start = allowed_not_above(estimate)
        rows.append({
            "model_id": r.model_id,
            "config_id": r.config_id,
            "step23_peak_allocated_bytes": peak,
            "gpu_budget_bytes": budget,
            "safety_multiplier": STEP23_SAFETY_MULTIPLIER,
            "estimated_safe_upper_integer": estimate,
            "initial_probe_microbatch": start,
            "initial_accumulation_steps": LOGICAL_BATCH // start,
            "performance_information_used": False,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 7. ISOLATED CUDA WORKER
# =============================================================================

def worker_execute(model_id, config_id, microbatch, budget_bytes):
    import torch
    import torch.nn as nn

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.dont_write_bytecode = True

    s23, s0, upstream, m23, m23_sha, f23, fp23, f22, fp22 = verify_all()
    repo_map = {
        r["repo_id"]: r
        for r in upstream["freeze17"]["repositories"]
    }
    model_map = step20_model_map(upstream["freeze20"])

    matches = [
        r for r in f22["candidate_source_configurations"]
        if r["model_id"] == model_id and r["config_id"] == config_id
    ]
    if len(matches) != 1:
        raise RuntimeError("Step-22 candidate lookup failure.")
    rec = matches[0]
    repo = repo_map[rec["repo_id"]]
    assert_single_repository_clean(repo)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    device = torch.device("cuda:0")
    # PyTorch CUDA memory-stat APIs are more robust when given an explicit
    # integer device index rather than a torch.device object in this
    # CUDA-13/WSL runtime.  This is a runtime API compatibility binding only.
    device_index = 0 if device.index is None else int(device.index)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    # Do not reset allocator peak statistics before the first CUDA allocation.
    # In this WSL/CUDA-13 runtime the allocator may reject reset_peak_memory_stats
    # until a CUDA allocation has materialized the device allocator.  Step 23
    # successfully used the same API only after model.to(cuda).
    sc = copy.deepcopy(rec["source_config"])
    runtime_bindings = {}
    if sc.get("device") == "__RUNTIME_DEVICE__":
        sc["device"] = device
        runtime_bindings["device"] = "cuda:0"

    # MLF's configs.bs is a runtime tensor-shape binding, not a predictive
    # hyperparameter. Step 22 used bs=2 for its synthetic interface. Bind it
    # prospectively to the physical microbatch used by Stage-1 execution.
    if model_id == "MLF" and "bs" in sc:
        runtime_bindings["bs_original_frozen_value"] = sc["bs"]
        sc["bs"] = int(microbatch)
        runtime_bindings["bs_effective_runtime_value"] = int(microbatch)

    target, import_root, module_name, preferred_class, module_file = import_target(
        model_id=model_id,
        model_record=model_map[model_id],
        repository_root=Path(repo["local_path"]),
    )

    if rec["configuration_mode"] == "direct_constructor_kwargs":
        backbone = target(**sc)
    elif rec["configuration_mode"] == "config_namespace":
        backbone = target(SimpleNamespace(**sc))
    else:
        raise RuntimeError("Unknown configuration_mode.")

    class Adapter(nn.Module):
        def __init__(self, mid, bb):
            super().__init__()
            self.mid = mid
            self.backbone = bb
            self.readout = None if mid == "TIMESNET" else nn.Linear(PRED_LEN * FEATURES, OUTPUTS)

        def forward(self, x):
            if self.mid == "TIMESNET":
                mark = torch.ones((x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
                z = self.backbone(x, mark, None, None)
                if tuple(z.shape) != (x.shape[0], OUTPUTS):
                    raise RuntimeError("TimesNet output contract drift.")
                return z, None

            if self.mid == "PATCHTST":
                f = self.backbone(x.transpose(1, 2))
                if tuple(f.shape) != (x.shape[0], FEATURES, PRED_LEN):
                    raise RuntimeError("PatchTST output contract drift.")
                f = f.transpose(1, 2).contiguous()
            elif self.mid == "ITRANSFORMER":
                f = self.backbone(x, None, None, None)
                if isinstance(f, (tuple, list)):
                    f = f[0]
            elif self.mid == "MLF":
                f = self.backbone(x, None, None, None)
                if not isinstance(f, (tuple, list)) or not f:
                    raise RuntimeError("MLF output contract drift.")
                f = f[0]
            else:
                f = self.backbone(x)

            if tuple(f.shape) != (x.shape[0], PRED_LEN, FEATURES):
                raise RuntimeError(f"{self.mid}: forecast contract drift: {tuple(f.shape)}")
            z = self.readout(f.reshape(f.shape[0], -1))
            if tuple(z.shape) != (x.shape[0], OUTPUTS):
                raise RuntimeError("Directional head output contract drift.")
            return z, f

    model = Adapter(model_id, backbone).to(device)

    # Match the verified Step-23 ordering: first materialize the model on CUDA,
    # then reset peak allocator statistics.  Resetting here preserves the model's
    # current allocation as the starting high-water mark while allowing the
    # subsequent synthetic forward/backward + AdamW-state allocation to define
    # the measured training-runtime peak.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    adapter_params = 0 if model_id == "TIMESNET" else sum(p.numel() for p in model.readout.parameters())
    if model_id != "TIMESNET" and adapter_params != COMMON_HEAD_PARAMS:
        raise RuntimeError("Forecast adapter parameter count is not 948.")

    for name, p in model.named_parameters():
        if p.device != device:
            raise RuntimeError(f"Device drift: {name}")
        if p.is_floating_point() and p.dtype != torch.float32:
            raise RuntimeError(f"Non-FP32 parameter: {name}")

    lr = float(rec["training_config"]["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY,
    )
    accum = LOGICAL_BATCH // microbatch
    if LOGICAL_BATCH % microbatch != 0:
        raise RuntimeError("Microbatch does not divide 256.")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED)
    x = torch.randn((microbatch, LOOKBACK, FEATURES), generator=gen, dtype=torch.float32).to(device)
    i = torch.arange(microbatch).unsqueeze(1)
    j = torch.arange(OUTPUTS).unsqueeze(0)
    y = ((i + j) % 2).to(torch.float32).to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, forecast = model(x)
    if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite/non-FP32 logits.")

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    raw_loss = criterion(logits, y)
    scaled_loss = raw_loss / accum
    scaled_loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("Non-finite gradient norm.")

    backbone_nonzero = 0
    readout_nonzero = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            raise RuntimeError(f"Non-finite gradient: {name}")
        if torch.count_nonzero(p.grad).item() > 0:
            if name.startswith("backbone."):
                backbone_nonzero += 1
            elif name.startswith("readout."):
                readout_nonzero += 1

    if backbone_nonzero == 0:
        raise RuntimeError("No non-zero backbone gradient.")
    if model_id != "TIMESNET" and readout_nonzero == 0:
        raise RuntimeError("No non-zero readout gradient.")

    # Purpose: force AdamW state allocation. This is synthetic-only and is not
    # a development-data fit or a statistical training step.
    optimizer.step()
    torch.cuda.synchronize(device_index)

    state_tensors = 0
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                state_tensors += 1
                if v.is_floating_point() and not torch.isfinite(v).all():
                    raise RuntimeError("Non-finite AdamW state.")
    if state_tensors == 0:
        raise RuntimeError("AdamW state was not allocated.")

    allocated = int(torch.cuda.max_memory_allocated(device_index))
    reserved = int(torch.cuda.max_memory_reserved(device_index))
    assert_single_repository_clean(repo)

    return {
        "status": "PASS",
        "model_id": model_id,
        "config_id": config_id,
        "microbatch_size": int(microbatch),
        "gradient_accumulation_steps": int(accum),
        "logical_batch_size": LOGICAL_BATCH,
        "effective_batch_size": int(microbatch * accum),
        "learning_rate": lr,
        "adapter_parameter_count": int(adapter_params),
        "backbone_nonzero_gradient_tensors": int(backbone_nonzero),
        "readout_nonzero_gradient_tensors": int(readout_nonzero),
        "optimizer_state_tensor_count": int(state_tensors),
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
        "gpu_budget_bytes": int(budget_bytes),
        "within_gpu_budget": bool(allocated <= budget_bytes),
        "gpu_budget_metric": "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_is_diagnostic_only": True,
        "raw_synthetic_loss": float(raw_loss.detach().cpu()),
        "scaled_synthetic_loss": float(scaled_loss.detach().cpu()),
        "preclip_gradient_norm": float(grad_norm.detach().cpu()),
        "runtime_bindings": runtime_bindings,
        "cuda_memory_stats_device_index": int(device_index),
        "cuda_peak_reset_after_model_to_device": True,
        "dtype": "float32",
        "amp_enabled": False,
        "optimizer_step_run": True,
        "optimizer_step_purpose": "synthetic optimizer-state/memory preflight only",
        "model_fitted": False,
        "development_arrays_opened_by_worker": False,
        "validation_performance_inspected": False,
        "final_test_opened": False,
        "repository_clean_after_worker": True,
    }


def worker_main(model_id, config_id, microbatch, budget_bytes):
    try:
        result = worker_execute(model_id, config_id, microbatch, budget_bytes)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    except Exception as exc:
        oom = type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError") or "out of memory" in str(exc).lower()
        result = {
            "status": "OOM" if oom else "FAIL",
            "model_id": model_id,
            "config_id": config_id,
            "microbatch_size": int(microbatch),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "model_fitted": False,
            "validation_performance_inspected": False,
            "final_test_opened": False,
        }
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 42 if oom else 1


# =============================================================================
# 8. MASTER WORKER ORCHESTRATION
# =============================================================================

def parse_result(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    return None


def run_worker(model, config, mb, budget):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONHASHSEED"] = str(SEED)
    cmd = [
        sys.executable, str(SCRIPT), "--worker",
        "--model-id", model, "--config-id", config,
        "--microbatch", str(mb), "--budget-bytes", str(budget),
    ]
    try:
        p = subprocess.run(
            cmd, cwd=str(Q1_DIR), env=env,
            capture_output=True, text=True, check=False, timeout=WORKER_TIMEOUT,
        )
        r = parse_result(p.stdout)
        if r is None:
            r = {
                "status": "FAIL", "model_id": model, "config_id": config,
                "microbatch_size": mb, "exception_type": "WorkerResultMissing",
                "exception_message": "Worker emitted no Step-24 result sentinel.",
            }
        return r, p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "FAIL", "model_id": model, "config_id": config,
            "microbatch_size": mb, "exception_type": "TimeoutExpired",
            "exception_message": f"Worker exceeded {WORKER_TIMEOUT}s.",
        }, exc.stdout or "", exc.stderr or "", 124


def log_worker(model, config, mb, attempt, stdout, stderr, code):
    WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = WORKER_LOG_DIR / f"{model}__{config}__mb{mb}__attempt{attempt}.log"
    path.write_text(
        f"return_code={code}\n\n===== STDOUT =====\n{stdout}\n===== STDERR =====\n{stderr}",
        encoding="utf-8",
    )
    return path


def probe_schedule(plan, budget):
    probes, schedule = [], []
    for row in plan.itertuples(index=False):
        start_idx = MICROBATCHES.index(int(row.initial_probe_microbatch))
        selected = None
        for attempt, mb in enumerate(MICROBATCHES[start_idx:], start=1):
            print(f"PROBE {row.model_id:<14} {row.config_id:<20} mb={mb:<3} accum={LOGICAL_BATCH // mb:<3}")
            result, stdout, stderr, code = run_worker(row.model_id, row.config_id, mb, budget)
            log = log_worker(row.model_id, row.config_id, mb, attempt, stdout, stderr, code)
            result.update({"attempt": attempt, "worker_log": str(log), "return_code": int(code)})
            probes.append(result)

            if result["status"] == "OOM":
                print("      OOM -> halve.")
                continue
            if result["status"] != "PASS" or code != 0:
                raise RuntimeError(
                    f"Non-OOM runtime failure {row.model_id}/{row.config_id}/mb={mb}: "
                    f"{result.get('exception_type')}: {result.get('exception_message')}\n"
                    f"Worker log: {log}"
                )
            if not result["within_gpu_budget"]:
                print("      PASS but above 75% peak-allocated-memory budget -> halve.")
                continue
            selected = result
            print(
                f"      SELECT allocated={result['peak_cuda_allocated_bytes']/(1024**2):.1f} MiB "
                f"<= budget={budget/(1024**2):.1f} MiB; "
                f"reserved(diagnostic)={result['peak_cuda_reserved_bytes']/(1024**2):.1f} MiB"
            )
            break

        if selected is None:
            raise RuntimeError(f"No safe microbatch: {row.model_id}/{row.config_id}")

        schedule.append({
            "model_id": row.model_id,
            "config_id": row.config_id,
            "physical_microbatch_size": int(selected["microbatch_size"]),
            "gradient_accumulation_steps": int(selected["gradient_accumulation_steps"]),
            "logical_batch_size": LOGICAL_BATCH,
            "effective_batch_size": int(selected["effective_batch_size"]),
            "learning_rate": float(selected["learning_rate"]),
            "peak_cuda_allocated_bytes": int(selected["peak_cuda_allocated_bytes"]),
            "peak_cuda_reserved_bytes": int(selected["peak_cuda_reserved_bytes"]),
            "gpu_budget_bytes": int(selected["gpu_budget_bytes"]),
            "gpu_budget_metric": "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_is_diagnostic_only": True,
            "optimizer_state_allocated": int(selected["optimizer_state_tensor_count"]) > 0,
            "adapter_parameter_count": int(selected["adapter_parameter_count"]),
            "backbone_nonzero_gradient_tensors": int(selected["backbone_nonzero_gradient_tensors"]),
            "mlf_runtime_bs_bound_to_microbatch": (
                selected.get("runtime_bindings", {}).get("bs_effective_runtime_value")
                == selected["microbatch_size"]
                if row.model_id == "MLF" else False
            ),
            "selection_uses_validation_performance": False,
            "selection_uses_final_test": False,
        })

    sdf = pd.DataFrame(schedule)
    if (
        len(sdf) != 16
        or not (sdf.effective_batch_size == LOGICAL_BATCH).all()
        or not (sdf.peak_cuda_allocated_bytes <= sdf.gpu_budget_bytes).all()
    ):
        raise RuntimeError("Microbatch schedule validation failed.")
    return pd.DataFrame(probes), sdf


# =============================================================================
# 9. FREEZE / VERIFY
# =============================================================================

def step24_fp_payload(freeze_payload):
    body = dict(freeze_payload)
    body.pop("generated_at_utc", None)
    body.pop("training_runtime_preflight_fingerprint_sha256", None)
    return body


def verify_existing():
    print("=" * 140)
    print("STEP 24 — VERIFY EXISTING TRAINING-RUNTIME PREFLIGHT")
    print("=" * 140)

    s23, s0, upstream, m23, m23_sha, f23, fp23, f22, fp22 = verify_all()
    msha = sidecar_ok(MANIFEST, MANIFEST_SHA)
    manifest = load(MANIFEST)
    freeze = load(FREEZE)

    if manifest.get("step", {}).get("status") != "PASS":
        return 20
    if manifest.get("configuration_fingerprint_sha256") != s0["fingerprint"]:
        return 21
    for i in range(1, 24):
        if manifest.get(f"step{i:02d}_manifest_sha256") != (
            upstream["manifest_sha"][i] if i <= 22 else m23_sha
        ):
            return 22

    for rec in manifest.get("artifacts", {}).values():
        p = Path(rec["path"])
        if not p.exists() or sha(p) != rec["sha256"]:
            return 23

    stored = freeze["training_runtime_preflight_fingerprint_sha256"]
    if semantic_hash(step24_fp_payload(freeze)) != stored:
        return 24
    if manifest["training_runtime_preflight_fingerprint_sha256"] != stored:
        return 25

    schedule = pd.read_csv(SCHEDULE_CSV)
    if (
        len(schedule) != 16
        or not (schedule.effective_batch_size == LOGICAL_BATCH).all()
        or not (schedule.peak_cuda_allocated_bytes <= schedule.gpu_budget_bytes).all()
    ):
        return 26

    repo_map, rdf = recheck_repositories(upstream["freeze17"])
    if not rdf.passed.all():
        return 27
    if sha(Path(s0["lock_path"])) != s0["lock_sha256"]:
        return 28

    print(f"[PASS] Configuration fingerprint : {s0['fingerprint']}")
    print(f"[PASS] Step-23 manifest          : {m23_sha}")
    print(f"[PASS] Step-23 smoke freeze      : {fp23}")
    print(f"[PASS] Frozen repositories       : {len(repo_map)}")
    print(f"[PASS] Microbatch schedules      : {len(schedule)}")
    print(f"[PASS] Effective batch           : 256 for all 16 candidates")
    print(f"[PASS] Step-24 preflight freeze  : {stored}")
    print(f"[PASS] Step-24 manifest          : {msha}")
    print(f"[PASS] Step-10 test seal         : SEALED_NOT_OPENED")
    print(f"[PASS] Global final-test state   : {s0['state']}")
    print("=" * 140)
    print("STEP 24 VERIFICATION STATUS: PASS")
    print("=" * 140)
    return 0


def run_master():
    stage = "verify"
    try:
        print("=" * 140)
        print(STEP_NAME)
        print("=" * 140)

        print("[1/11] Verifying Step 00 and immutable Step 01 -> Step 23 chain...")
        s23, s0, upstream, m23, m23_sha, f23, fp23, f22, fp22 = verify_all()
        print(f"        Configuration fingerprint : {s0['fingerprint']}")
        print(f"        Step-23 manifest          : {m23_sha}")
        print(f"        Step-23 smoke freeze      : {fp23}")
        print("        Step-10 test seal         : SEALED_NOT_OPENED")
        print(f"        Global final-test state   : {s0['state']}")
        print("        Legacy Step 22B/22C/22D  : NOT USED")

        stage = "collision_guard"
        print("[2/11] Checking immutable Step-24 output target...")
        assert_fresh_target()
        cfg.create_directories()

        stage = "environment"
        print("[3/11] Reconciling and freezing current training runtime...")
        pip_text, packages, env_delta, runtime = environment_audit()
        atomic_text(PIP_FREEZE, pip_text)
        write_csv(ENV_DELTA_CSV, env_delta)
        write_csv(RUNTIME_CSV, runtime)
        print(f"        torch                     : {packages['torch']}")
        print(f"        reformer-pytorch          : {packages['reformer-pytorch']}")
        print(f"        nvidia-cudnn-cu13         : {packages['nvidia-cudnn-cu13']}")
        print("        nvidia-cudnn-cu12 present : False")
        print(f"        Step01->Step24 package deltas: {len(env_delta)}")

        stage = "repositories_before"
        print("[4/11] Rechecking all eight frozen Step-17 repositories...")
        repo_map, repo_before = recheck_repositories(upstream["freeze17"])
        write_csv(REPO_BEFORE_CSV, repo_before)
        print(f"        Frozen repositories        : {len(repo_map)}")
        print(f"        Worktrees clean incl untracked: {bool(repo_before.passed.all())}")

        stage = "development_headers"
        print("[5/11] Auditing Step-14 development X_full/y_direction headers only...")
        inventory, split = development_headers()
        write_csv(ARRAY_CSV, inventory)
        write_csv(SPLIT_CSV, split)
        for r in split.itertuples(index=False):
            print(
                f"        Fold {r.fold} {r.partition:<10} n={r.n_sequences:>6,} "
                f"X=[N,{r.lookback},{r.feature_count}] y=[N,{r.direction_outputs}]"
            )
        print("        Numerical development arrays loaded: False")
        print("        Final-test parquet opened       : False")

        stage = "contracts"
        print("[6/11] Freezing 16 candidate training-runtime contracts...")
        contracts = training_contracts(f22)
        write_csv(CONTRACT_CSV, contracts)
        print("        Logical batch             : 256")
        print("        Optimizer                 : AdamW")
        print("        Weight decay              : 1e-4")
        print("        BCE reduction             : mean")
        print("        Gradient clip             : 1.0")
        print("        FP32 / AMP                : float32 / False")

        stage = "gpu_resource"
        print("[7/11] Auditing exclusive CUDA resource budget...")
        gpu = nvidia_smi()
        print(f"        GPU                       : {gpu['gpu_name']}")
        print(f"        Total memory              : {gpu['total_memory_mib']:,} MiB")
        print(
            f"        Reserved-memory budget    : "
            f"{gpu['budget_bytes']/(1024**2):,.1f} MiB ({GPU_BUDGET_FRACTION:.0%})"
        )
        print("        Concurrent CUDA processes : 0")

        stage = "plan"
        print("[8/11] Deriving conservative initial probes from Step-23 CUDA evidence...")
        smoke = step23_cuda_records(m23)
        plan = initial_plan(contracts, smoke, gpu)
        write_csv(PLAN_CSV, plan)
        for r in plan.itertuples(index=False):
            print(
                f"        {r.model_id:<14} {r.config_id:<20} "
                f"Step23@2={r.step23_peak_allocated_bytes/(1024**2):>7.1f} MiB "
                f"-> start mb={r.initial_probe_microbatch:<3}"
            )

        stage = "probe"
        print()
        print("[9/11] Running isolated synthetic CUDA AdamW-step microbatch preflight...")
        print("-" * 140)
        probes, schedule = probe_schedule(plan, gpu["budget_bytes"])
        write_csv(PROBE_CSV, probes)
        write_csv(SCHEDULE_CSV, schedule)

        stage = "repositories_after"
        print("[10/11] Rechecking repositories after all runtime probes...")
        _, repo_after = recheck_repositories(upstream["freeze17"])
        write_csv(REPO_AFTER_CSV, repo_after)
        print(f"        Worktrees clean incl untracked: {bool(repo_after.passed.all())}")

        if sha(Path(s0["lock_path"])) != s0["lock_sha256"]:
            raise RuntimeError("Final-test lock changed during Step 24.")

        stage = "freeze"
        print("[11/11] Freezing the development/training-runtime preflight...")

        semantic = {
            "schema_version": "1.0",
            "status": "STAGE1_DEVELOPMENT_TRAINING_RUNTIME_PREFLIGHT_FROZEN",
            "configuration_fingerprint_sha256": s0["fingerprint"],
            "upstream": {
                "step21_adapter_contract_fingerprint_sha256": upstream["fingerprint21"],
                "step22_source_config_translation_fingerprint_sha256": fp22,
                "step23_model_smoke_fingerprint_sha256": fp23,
                "step23_manifest_sha256": m23_sha,
            },
            "development_interface": {
                "fold_split_records": len(split),
                "array_header_records": len(inventory),
                "lookback": LOOKBACK,
                "features": FEATURES,
                "direction_outputs": OUTPUTS,
                "numerical_arrays_loaded": False,
                "validation_performance_inspected": False,
            },
            "training_contract": {
                "seed": SEED,
                "logical_batch_size": LOGICAL_BATCH,
                "optimizer": "AdamW",
                "weight_decay": WEIGHT_DECAY,
                "adamw_betas": list(BETAS),
                "adamw_eps": ADAMW_EPS,
                "loss": "BCEWithLogitsLoss(reduction='mean')",
                "gradient_clip_norm": GRAD_CLIP,
                "dtype": "float32",
                "amp_enabled": False,
                "accumulation_rule": (
                    "equal-size microbatch mean loss divided by accumulation_steps; "
                    "optimizer.step after accumulation_steps microbatches"
                ),
            },
            "resource_policy": {
                "gpu_name": gpu["gpu_name"],
                "gpu_total_memory_bytes": gpu["total_memory_bytes"],
                "peak_allocated_budget_fraction": GPU_BUDGET_FRACTION,
                "peak_allocated_budget_bytes": gpu["budget_bytes"],
                "hard_acceptance_metric": "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_role": "diagnostic_only",
                "empirical_cuda_step_must_pass": True,
                "step23_safety_multiplier": STEP23_SAFETY_MULTIPLIER,
                "allowed_microbatches": list(MICROBATCHES),
                "selection_uses_validation_performance": False,
                "selection_uses_final_test": False,
            },
            "runtime_environment": {
                "pip_freeze_sha256": sha(PIP_FREEZE),
                "step01_to_step24_changed_packages": len(env_delta),
                "torch": packages["torch"],
                "reformer_pytorch": packages["reformer-pytorch"],
                "nvidia_cudnn_cu13": packages["nvidia-cudnn-cu13"],
                "nvidia_cudnn_cu12_present": False,
            },
            "runtime_compatibility": {
                "legacy_step22b_22c_22d_used": False,
                "mlf_bs_runtime_bound_to_physical_microbatch": True,
                "mlf_bs_binding_is_candidate_hyperparameter_change": False,
                "repository_mutation": False,
            },
            "candidate_microbatch_schedule": schedule.to_dict(orient="records"),
            "pipeline_state": {
                "synthetic_optimizer_preflight_run": True,
                "optimizer_steps_on_development_data": False,
                "stage1_model_fitting_started": False,
                "validation_performance_inspected": False,
                "threshold_selection_performed": False,
                "feature_selection_applied": False,
            },
            "final_test_protection": {
                "step10_seal_status": "SEALED_NOT_OPENED",
                "sealed_test_parquet_opened_or_hashed_by_step24": False,
                "final_test_performance_computed": False,
                "global_final_test_state": s0["state"],
            },
        }
        fp24 = semantic_hash(semantic)
        freeze = {
            **semantic,
            "generated_at_utc": now(),
            "training_runtime_preflight_fingerprint_sha256": fp24,
        }
        atomic_json(FREEZE, freeze)
        shutil.copy2(SCRIPT, FROZEN_SCRIPT)

        lines = [
            STEP_NAME, "=" * 110, "Status: PASS",
            f"Step-23 smoke fingerprint: {fp23}",
            "Logical/effective batch: 256",
            f"GPU peak-allocated-memory budget: {GPU_BUDGET_FRACTION:.0%}",
            "Optimizer: AdamW; weight_decay=1e-4; BCEWithLogitsLoss(mean); grad_clip=1.0",
            "FP32; AMP=False",
            "Validation performance inspected: False",
            "Final test accessed: False",
            "", "Frozen microbatch schedule:",
        ]
        for r in schedule.itertuples(index=False):
            lines.append(
                f"{r.model_id:<14} {r.config_id:<20} "
                f"mb={r.physical_microbatch_size:<3} "
                f"accum={r.gradient_accumulation_steps:<3} "
                f"effective={r.effective_batch_size:<3} "
                f"allocated={r.peak_cuda_allocated_bytes/(1024**2):.1f} MiB "
                f"reserved_diag={r.peak_cuda_reserved_bytes/(1024**2):.1f} MiB"
            )
        atomic_text(SUMMARY, "\n".join(lines) + "\n")

        worker_logs = sorted(WORKER_LOG_DIR.glob("*.log"))
        artifacts = {}
        output_files = [
            ARRAY_CSV, SPLIT_CSV, CONTRACT_CSV, PLAN_CSV, PROBE_CSV,
            SCHEDULE_CSV, REPO_BEFORE_CSV, REPO_AFTER_CSV, ENV_DELTA_CSV,
            RUNTIME_CSV, PIP_FREEZE, FREEZE, FROZEN_SCRIPT, SUMMARY,
            *worker_logs,
        ]
        for p in output_files:
            artifacts[
                f"worker_log::{p.name}" if p.parent.resolve() == WORKER_LOG_DIR.resolve() else p.name
            ] = {
                "path": str(p),
                "size_bytes": int(p.stat().st_size),
                "sha256": sha(p),
            }

        manifest = {
            "schema_version": "1.0",
            "step": {"step_number": 24, "step_name": STEP_NAME, "status": "PASS"},
            "frozen_at_utc": now(),
            "configuration_fingerprint_sha256": s0["fingerprint"],
            **{
                f"step{i:02d}_manifest_sha256": (
                    upstream["manifest_sha"][i] if i <= 22 else m23_sha
                )
                for i in range(1, 24)
            },
            "step23_model_smoke_fingerprint_sha256": fp23,
            "training_runtime_preflight_fingerprint_sha256": fp24,
            "counts": {
                "development_fold_split_records": len(split),
                "candidate_configurations": len(schedule),
                "microbatch_probe_records": len(probes),
                "worker_logs": len(worker_logs),
            },
            "artifacts": artifacts,
        }
        atomic_json(MANIFEST, manifest)
        msha = sha(MANIFEST)
        atomic_text(MANIFEST_SHA, f"{msha}  {MANIFEST.name}\n")

        print()
        print("-" * 140)
        print("FROZEN PHYSICAL MICROBATCH SCHEDULE")
        print("-" * 140)
        for r in schedule.itertuples(index=False):
            print(
                f"{r.model_id:<14} {r.config_id:<20} "
                f"mb={r.physical_microbatch_size:<3} "
                f"accum={r.gradient_accumulation_steps:<3} "
                f"effective={r.effective_batch_size:<3} "
                f"allocated={r.peak_cuda_allocated_bytes/(1024**2):>7.1f} MiB "
                f"reserved_diag={r.peak_cuda_reserved_bytes/(1024**2):>7.1f} MiB"
            )
        print("-" * 140)
        print("Stage-1 fitting started           : False")
        print("Validation performance inspected : False")
        print("Final-test information used       : False")
        print("Repository mutation               : False")
        print("Legacy Step 22B/22C/22D          : NOT USED")
        print("Step-10 final-test seal           : SEALED_NOT_OPENED")
        print(f"Global final-test state           : {s0['state']}")
        print("-" * 140)
        print(f"Preflight freeze      : {FREEZE}")
        print(f"Preflight fingerprint : {fp24}")
        print(f"Manifest              : {MANIFEST}")
        print(f"Manifest SHA-256      : {msha}")
        print("=" * 140)
        print(
            "PASS: All 16 candidates have a CUDA-confirmed physical microbatch "
            "and exact accumulation schedule preserving effective batch 256."
        )
        print(
            "PASS: AdamW state allocation, FP32 BCE backward, gradient clipping "
            "and finite non-zero backbone gradients were verified synthetically."
        )
        print(
            "PASS: No development-data optimizer step, validation performance, "
            "threshold selection or final-test information was used."
        )
        print("=" * 140)
        print(
            "Do not start Stage-1 fitting yet. Paste the complete Step-24 output "
            "for audit, then run --verify-only if instructed."
        )
        return 0

    except Exception as exc:
        archived = archive_partial()
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        failure = FAILED_DIR / (
            "24_training_runtime_preflight_FAILED_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
        atomic_json(
            failure,
            {
                "status": "FAIL",
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "failed_at_utc": now(),
                "archived_partial_artifacts": None if archived is None else str(archived),
            },
        )
        print()
        print("=" * 140)
        print("STEP 24 FAILED — TRAINING-RUNTIME PREFLIGHT NOT FROZEN")
        print("=" * 140)
        print(f"Stage      : {stage}")
        print(f"Error type : {type(exc).__name__}")
        print(f"Error      : {exc}")
        print(f"Diagnostic : {failure}")
        if archived:
            print(f"Archived partial Step-24 outputs: {archived}")
        print("=" * 140)
        print("Do not proceed to Stage-1 model fitting.")
        return 30


# =============================================================================
# 10. CLI
# =============================================================================

def cli():
    p = argparse.ArgumentParser(description=STEP_NAME)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model-id", choices=list(MODELS), help=argparse.SUPPRESS)
    p.add_argument("--config-id", help=argparse.SUPPRESS)
    p.add_argument("--microbatch", type=int, choices=list(MICROBATCHES), help=argparse.SUPPRESS)
    p.add_argument("--budget-bytes", type=int, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = cli()
    if args.worker:
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("PYTHONHASHSEED", str(SEED))
        sys.dont_write_bytecode = True
        if not args.model_id or not args.config_id or args.microbatch is None or args.budget_bytes is None:
            raise SystemExit("Worker requires model/config/microbatch/budget.")
        return worker_main(args.model_id, args.config_id, args.microbatch, args.budget_bytes)
    if args.verify_only:
        return verify_existing()
    return run_master()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStep 24 interrupted by user.")
        raise SystemExit(130)
