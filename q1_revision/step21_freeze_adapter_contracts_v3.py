#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 21 — CLEAN V3 FREEZE EXACT STAGE-1 DIRECTIONAL ADAPTER CONTRACTS

Purpose
-------
Freeze the executable task-adaptation contract for the eight frozen Stage-1
architecture families BEFORE any third-party model source is imported,
instantiated, or smoke-tested.

Frozen design
-------------
Seven forecast-style architectures:
    preserve the complete frozen forecasting path
    -> configure/retain a 5-session forecast output
    -> normalize only tensor orientation / documented return packaging
    -> flatten [B,5,63] -> [B,315]
    -> append ONE trainable Linear(315,3) directional readout
    -> raw [B,3] logits for 1d / 3d / 5d direction

TimesNet:
    use the frozen native classification pathway
    -> task_name='classification'
    -> num_class=3
    -> all-one [B,60] valid-timestep mask
    -> raw [B,3] logits
    -> no extra adapter head

All eight:
    input [B,60,63]
    BCEWithLogitsLoss(reduction='mean')
    no class weights
    Stage-1 evaluation threshold = 0.50
    no stock embedding
    no regime conditioning / FiLM
    no auxiliary regression
    no pretrained weights
    end-to-end training from random initialization
    frozen A/B candidate hyperparameters remain unchanged

This stage is a CONTRACT FREEZE ONLY.

It DOES NOT:
- import third-party repositories;
- execute third-party model source;
- instantiate a model;
- implement an adapter;
- run forward/backward passes;
- train a model;
- perform feature selection;
- tune probability thresholds;
- open or hash the sealed final-test parquet.

Step 22 will implement these exact contracts in isolated repository subprocesses
and run CPU/GPU forward/backward smoke tests.

Run
---
    python step21_freeze_adapter_contracts_v3.py

Verify
------
    python step21_freeze_adapter_contracts_v3.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# =============================================================================
# 0. SCRIPT / CONFIG
# =============================================================================

SCRIPT_FILE = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_FILE.parent

if SCRIPT_DIR.name != "q1_revision":
    raise RuntimeError(
        "step21_freeze_adapter_contracts_v3.py must be inside "
        "<repository>/q1_revision/."
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg


# =============================================================================
# 1. FROZEN DESIGN
# =============================================================================

STEP_NUMBER = 21
STEP_ID = "21"
STEP_NAME = "STEP 21 — FREEZE EXACT STAGE-1 DIRECTIONAL ADAPTER CONTRACTS"
SCHEMA_VERSION = "1.0"

STAGE1_MODEL_IDS = (
    "DLINEAR",
    "NLINEAR",
    "PATCHTST",
    "TIMESNET",
    "ITRANSFORMER",
    "FREDFORMER",
    "FRNET",
    "MLF",
)

EXPECTED_MODEL_COUNT = 8
EXPECTED_FROZEN_REPOSITORY_COUNT = 8
EXPECTED_IMPLEMENTATION_REPOSITORY_COUNT = 7
EXPECTED_REFERENCE_REPOSITORY_COUNT = 1

LOOKBACK = 60
FEATURE_COUNT = 63
HORIZONS = (1, 3, 5)
DIRECTION_OUTPUTS = 3
FORECAST_LENGTH = max(HORIZONS)
FORECAST_READOUT_INPUT_DIM = FORECAST_LENGTH * FEATURE_COUNT
FORECAST_ADAPTER_ADDED_PARAMS = (
    FORECAST_READOUT_INPUT_DIM * DIRECTION_OUTPUTS
    +
    DIRECTION_OUTPUTS
)

EXPECTED_REPO_BY_MODEL = {
    "DLINEAR": "LTSF_LINEAR",
    "NLINEAR": "LTSF_LINEAR",
    "PATCHTST": "PATCHTST",
    "TIMESNET": "TIMESNET_TSLIB",
    "ITRANSFORMER": "ITRANSFORMER",
    "FREDFORMER": "FREDFORMER",
    "FRNET": "FRNET",
    "MLF": "MLF",
}

ACCEPTABLE_STEP20_RESOLUTIONS = {
    "root_terminal_candidates_identified",
    "nested_terminal_candidates_identified",
    "native_classification_path_confirmed_from_step19",
}

FORBIDDEN_TEST_FILENAME = "10_final_test_locked_panel.parquet"


# =============================================================================
# 2. UPSTREAM PATHS
# =============================================================================

STEP01_ENVIRONMENT_AUDIT = (
    cfg.ENVIRONMENT_DIR / "01_environment_audit.json"
)

UPSTREAM = {
    1: (
        cfg.MANIFEST_DIR / "01_environment_audit_manifest.json",
        cfg.MANIFEST_DIR / "01_environment_audit_manifest.sha256",
    ),
    2: (
        cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.json",
        cfg.MANIFEST_DIR / "02_market_data_acquisition_manifest.sha256",
    ),
    3: (
        cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.json",
        cfg.MANIFEST_DIR / "03_market_integrity_audit_manifest.sha256",
    ),
    4: (
        cfg.MANIFEST_DIR / "04_session_policy_manifest.json",
        cfg.MANIFEST_DIR / "04_session_policy_manifest.sha256",
    ),
    5: (
        cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.json",
        cfg.MANIFEST_DIR / "05_clean_model_panel_manifest.sha256",
    ),
    6: (
        cfg.MANIFEST_DIR / "06_stock_feature_manifest.json",
        cfg.MANIFEST_DIR / "06_stock_feature_manifest.sha256",
    ),
    7: (
        cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.json",
        cfg.MANIFEST_DIR / "07_nifty_context_feature_manifest.sha256",
    ),
    8: (
        cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.json",
        cfg.MANIFEST_DIR / "08_63_feature_panel_manifest.sha256",
    ),
    9: (
        cfg.MANIFEST_DIR / "09_target_manifest.json",
        cfg.MANIFEST_DIR / "09_target_manifest.sha256",
    ),
    10: (
        cfg.MANIFEST_DIR / "10_chronological_split_manifest.json",
        cfg.MANIFEST_DIR / "10_chronological_split_manifest.sha256",
    ),
    11: (
        cfg.MANIFEST_DIR / "11_preprocessing_manifest.json",
        cfg.MANIFEST_DIR / "11_preprocessing_manifest.sha256",
    ),
    12: (
        cfg.MANIFEST_DIR / "12_volatility_regime_manifest.json",
        cfg.MANIFEST_DIR / "12_volatility_regime_manifest.sha256",
    ),
    13: (
        cfg.MANIFEST_DIR / "13_sequence_index_manifest.json",
        cfg.MANIFEST_DIR / "13_sequence_index_manifest.sha256",
    ),
    14: (
        cfg.MANIFEST_DIR / "14_model_array_manifest.json",
        cfg.MANIFEST_DIR / "14_model_array_manifest.sha256",
    ),
    15: (
        cfg.MANIFEST_DIR / "15_pre_model_integrity_manifest.json",
        cfg.MANIFEST_DIR / "15_pre_model_integrity_manifest.sha256",
    ),
    16: (
        cfg.MANIFEST_DIR / "16_stage1_model_registry_manifest.json",
        cfg.MANIFEST_DIR / "16_stage1_model_registry_manifest.sha256",
    ),
    17: (
        cfg.MANIFEST_DIR / "17_official_model_source_manifest.json",
        cfg.MANIFEST_DIR / "17_official_model_source_manifest.sha256",
    ),
    18: (
        cfg.MANIFEST_DIR / "18_source_api_manifest.json",
        cfg.MANIFEST_DIR / "18_source_api_manifest.sha256",
    ),
    19: (
        cfg.MANIFEST_DIR / "19_adapter_locus_manifest.json",
        cfg.MANIFEST_DIR / "19_adapter_locus_manifest.sha256",
    ),
    20: (
        cfg.MANIFEST_DIR / "20_transitive_backbone_manifest.json",
        cfg.MANIFEST_DIR / "20_transitive_backbone_manifest.sha256",
    ),
}

STEP16_FREEZE = (
    cfg.MANIFEST_DIR / "16_stage1_model_registry_freeze.json"
)
STEP17_FREEZE = (
    cfg.MANIFEST_DIR / "17_official_model_source_freeze.json"
)
STEP18_FREEZE = (
    cfg.MANIFEST_DIR / "18_source_api_freeze.json"
)
STEP19_FREEZE = (
    cfg.MANIFEST_DIR / "19_adapter_locus_freeze.json"
)
STEP20_FREEZE = (
    cfg.MANIFEST_DIR / "20_transitive_backbone_freeze.json"
)
STEP10_TEST_SEAL = (
    cfg.MANIFEST_DIR / "10_final_test_seal.json"
)


# =============================================================================
# 3. OUTPUT PATHS
# =============================================================================

ADAPTER_CONTRACT_CSV = (
    cfg.AUDIT_DIR / "21_stage1_adapter_contracts.csv"
)
REGISTRY_CLARIFICATION_CSV = (
    cfg.AUDIT_DIR / "21_registry_adapter_clarification.csv"
)
REPOSITORY_RECHECK_CSV = (
    cfg.AUDIT_DIR / "21_repository_recheck.csv"
)
SOURCE_HASH_CSV = (
    cfg.AUDIT_DIR / "21_source_file_hashes.csv"
)
ADAPTER_FREEZE_JSON = (
    cfg.MANIFEST_DIR / "21_adapter_contract_freeze.json"
)
MANIFEST_JSON = (
    cfg.MANIFEST_DIR / "21_adapter_contract_manifest.json"
)
MANIFEST_SHA = (
    cfg.MANIFEST_DIR / "21_adapter_contract_manifest.sha256"
)
HASH_REGISTER_CSV = (
    cfg.AUDIT_DIR / "21_adapter_contract_hashes.csv"
)
FROZEN_SCRIPT = (
    cfg.MANIFEST_DIR / "step21_freeze_adapter_contracts_frozen.py"
)
SUMMARY_TXT = (
    cfg.REPORT_DIR / "21_adapter_contract_summary.txt"
)
FAILED_DIR = (
    cfg.AUDIT_DIR / "step21_failed_attempts"
)


# =============================================================================
# 4. BASIC HELPERS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def atomic_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
    )
    os.replace(temporary, path)


def relative_to_experiment(path: Path) -> str:
    return str(
        path.resolve().relative_to(
            cfg.EXPERIMENT_ROOT_DIR.resolve()
        )
    )


def artifact_record(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": relative_to_experiment(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def git_executable() -> str:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("git executable was not found on PATH.")
    return executable


def run_git(
    arguments: list[str],
    cwd: Path,
    *,
    timeout_seconds: int = 120,
) -> str:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        [git_executable(), *arguments],
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "\nGIT COMMAND FAILED\n"
            f"Command: git {' '.join(arguments)}\n"
            f"CWD    : {cwd}\n"
            f"stdout :\n{result.stdout}\n"
            f"stderr :\n{result.stderr}"
        )
    return result.stdout.strip()


def normalize_git_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


# =============================================================================
# 5. STEP 00 / ENVIRONMENT / UPSTREAM CHAIN
# =============================================================================

def verify_step00() -> Dict[str, Any]:
    cfg.validate_configuration()
    payload = cfg.build_configuration_payload()
    fingerprint = cfg.canonical_json_sha256(payload)

    _, manifest_sha, status = cfg.verify_existing_freeze(
        configuration_payload=payload,
        configuration_fingerprint=fingerprint,
    )

    if status != "VERIFIED_EXISTING":
        raise RuntimeError("Step-00 freeze is not VERIFIED_EXISTING.")

    lock_path = cfg.get_final_test_lock_path()
    lock = json.loads(
        lock_path.read_text(encoding="utf-8")
    )

    if (
        lock.get("state") != "LOCKED_NOT_OPENED"
        or
        lock.get("final_test_opened") is not False
        or
        lock.get("final_test_scored") is not False
    ):
        raise RuntimeError(
            "Global final-test lock is not pristine."
        )

    return {
        "fingerprint": fingerprint,
        "step00_manifest_sha256": manifest_sha,
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "state": lock["state"],
    }


def verify_environment() -> Dict[str, Any]:
    if not STEP01_ENVIRONMENT_AUDIT.exists():
        raise RuntimeError(
            f"Step-01 environment audit missing: {STEP01_ENVIRONMENT_AUDIT}"
        )

    audit = json.loads(
        STEP01_ENVIRONMENT_AUDIT.read_text(encoding="utf-8")
    )

    expected_python = (
        audit.get("system", {}).get("python_executable")
    )

    if expected_python:
        if (
            Path(expected_python).resolve()
            !=
            Path(sys.executable).resolve()
        ):
            raise RuntimeError(
                "Current Python differs from frozen Step-01 environment.\n"
                f"Frozen : {expected_python}\n"
                f"Current: {sys.executable}"
            )

    return {
        "python_executable": expected_python,
        "pandas_version": pd.__version__,
        "git_executable": git_executable(),
    }


def verify_sidecar(
    manifest_path: Path,
    sidecar_path: Path,
    label: str,
) -> str:
    if not manifest_path.exists():
        raise RuntimeError(
            f"{label} manifest missing: {manifest_path}"
        )
    if not sidecar_path.exists():
        raise RuntimeError(
            f"{label} SHA sidecar missing: {sidecar_path}"
        )

    expected = (
        sidecar_path.read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )
    observed = sha256_file(manifest_path)

    if expected != observed:
        raise RuntimeError(
            f"{label} manifest SHA mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}"
        )
    return observed


def find_artifact(
    manifest: Dict[str, Any],
    filename: str,
) -> Dict[str, Any]:
    matches = []
    for key in ("artifacts", "canonical_artifacts"):
        for record in manifest.get(key, {}).values():
            if not isinstance(record, dict):
                continue
            path_text = record.get("path")
            if (
                path_text
                and
                Path(path_text).name == filename
            ):
                matches.append(record)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest artifact named "
            f"{filename}; found {len(matches)}."
        )
    return matches[0]


def verify_fingerprint_file(
    path: Path,
    fingerprint_field: str,
) -> tuple[Dict[str, Any], str]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    stored = payload.get(fingerprint_field)

    if not stored:
        raise RuntimeError(
            f"{path.name} lacks {fingerprint_field}."
        )

    body = dict(payload)
    body.pop(fingerprint_field, None)

    recomputed = cfg.canonical_json_sha256(body)
    if recomputed != stored:
        raise RuntimeError(
            f"{path.name}: fingerprint mismatch.\n"
            f"Stored    : {stored}\n"
            f"Recomputed: {recomputed}"
        )

    return payload, stored


def verify_upstream(
    step00: Dict[str, Any],
) -> Dict[str, Any]:
    shas: Dict[int, str] = {}
    manifests: Dict[int, Dict[str, Any]] = {}

    for step in range(1, 21):
        manifest_path, sidecar_path = UPSTREAM[step]
        shas[step] = verify_sidecar(
            manifest_path,
            sidecar_path,
            f"Step {step:02d}",
        )
        manifests[step] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

    if (
        manifests[1].get("status") != "PASS"
        or
        manifests[1].get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00["fingerprint"]
    ):
        raise RuntimeError("Step-01 status/fingerprint mismatch.")

    if (
        manifests[2]
        .get("step", {})
        .get("status")
        !=
        "PASS"
    ):
        raise RuntimeError("Step-02 status is not PASS.")

    for step in range(3, 21):
        manifest = manifests[step]

        if (
            manifest.get("step", {}).get("status") != "PASS"
            or
            manifest.get(
                "configuration_fingerprint_sha256"
            )
            !=
            step00["fingerprint"]
        ):
            raise RuntimeError(
                f"Step-{step:02d} status/fingerprint mismatch."
            )

        for previous in range(1, step):
            field = f"step{previous:02d}_manifest_sha256"
            if (
                field in manifest
                and
                manifest[field] != shas[previous]
            ):
                raise RuntimeError(
                    f"Step-{step:02d} chain mismatch: {field}"
                )

    environment = verify_environment()

    freeze16, fingerprint16 = verify_fingerprint_file(
        STEP16_FREEZE,
        "registry_freeze_fingerprint_sha256",
    )
    freeze17, fingerprint17 = verify_fingerprint_file(
        STEP17_FREEZE,
        "source_freeze_fingerprint_sha256",
    )
    freeze18, fingerprint18 = verify_fingerprint_file(
        STEP18_FREEZE,
        "source_api_fingerprint_sha256",
    )
    freeze19, fingerprint19 = verify_fingerprint_file(
        STEP19_FREEZE,
        "adapter_locus_fingerprint_sha256",
    )
    freeze20, fingerprint20 = verify_fingerprint_file(
        STEP20_FREEZE,
        "transitive_backbone_fingerprint_sha256",
    )

    expected_statuses = (
        (
            freeze16.get("status"),
            "STAGE1_MODEL_REGISTRY_AND_SELECTION_PROTOCOL_FROZEN",
            16,
        ),
        (
            freeze17.get("status"),
            "OFFICIAL_AUTHOR_MAINTAINED_MODEL_SOURCE_SNAPSHOTS_FROZEN",
            17,
        ),
        (
            freeze18.get("status"),
            "FROZEN_SOURCE_APIS_AUDITED_BEFORE_ADAPTER_IMPLEMENTATION",
            18,
        ),
        (
            freeze19.get("status"),
            "ADAPTER_LOCI_AUDITED_BEFORE_IMPLEMENTATION",
            19,
        ),
        (
            freeze20.get("status"),
            "TRANSITIVE_BACKBONE_STRUCTURE_AUDITED_BEFORE_ADAPTER_IMPLEMENTATION",
            20,
        ),
    )

    for observed, expected, step in expected_statuses:
        if observed != expected:
            raise RuntimeError(
                f"Step-{step:02d} freeze status mismatch."
            )

    fingerprint_checks = (
        (
            16,
            "registry_freeze_fingerprint_sha256",
            fingerprint16,
        ),
        (
            17,
            "source_freeze_fingerprint_sha256",
            fingerprint17,
        ),
        (
            18,
            "source_api_fingerprint_sha256",
            fingerprint18,
        ),
        (
            19,
            "adapter_locus_fingerprint_sha256",
            fingerprint19,
        ),
        (
            20,
            "transitive_backbone_fingerprint_sha256",
            fingerprint20,
        ),
    )

    for step, field, value in fingerprint_checks:
        if manifests[step].get(field) != value:
            raise RuntimeError(
                f"Step-{step:02d} manifest/freeze fingerprint mismatch."
            )

    for path, manifest in (
        (STEP16_FREEZE, manifests[16]),
        (STEP17_FREEZE, manifests[17]),
        (STEP18_FREEZE, manifests[18]),
        (STEP19_FREEZE, manifests[19]),
        (STEP20_FREEZE, manifests[20]),
    ):
        record = find_artifact(
            manifest,
            path.name,
        )
        if (
            Path(record["path"]).resolve() != path.resolve()
            or
            record["sha256"] != sha256_file(path)
        ):
            raise RuntimeError(
                f"Frozen artifact drift: {path.name}"
            )

    if freeze16.get("training_started") is not False:
        raise RuntimeError(
            "Step-16 says Stage-1 training already started."
        )

    state20 = freeze20.get("pipeline_state", {})
    for key in (
        "final_adapter_contracts_frozen",
        "task_adapters_implemented",
        "smoke_tests_run",
        "models_fitted",
        "feature_selection_applied",
        "probability_threshold_selection_performed",
    ):
        if key in state20 and state20.get(key) is not False:
            raise RuntimeError(
                f"Step-20 pre-adapter state violation: {key}"
            )

    records20 = freeze20.get("model_records", [])
    ids20 = [str(row["model_id"]) for row in records20]

    if (
        len(ids20) != EXPECTED_MODEL_COUNT
        or
        len(set(ids20)) != EXPECTED_MODEL_COUNT
        or
        set(ids20) != set(STAGE1_MODEL_IDS)
    ):
        raise RuntimeError(
            "Step-20 model records do not contain exactly "
            "the eight frozen Stage-1 model IDs."
        )

    step20_by_model = {
        row["model_id"]: row
        for row in records20
    }

    repositories = freeze17.get("repositories", [])
    if (
        len(repositories) != EXPECTED_FROZEN_REPOSITORY_COUNT
        or
        sum(
            row.get("repository_role") == "implementation"
            for row in repositories
        )
        !=
        EXPECTED_IMPLEMENTATION_REPOSITORY_COUNT
        or
        sum(
            row.get("repository_role") == "reference_redirect"
            for row in repositories
        )
        !=
        EXPECTED_REFERENCE_REPOSITORY_COUNT
    ):
        raise RuntimeError(
            "Step-17 frozen repository inventory drift."
        )

    if not STEP10_TEST_SEAL.exists():
        raise RuntimeError(
            "Step-10 final-test seal metadata missing."
        )

    seal = json.loads(
        STEP10_TEST_SEAL.read_text(encoding="utf-8")
    )

    if (
        seal.get("status") != "SEALED_NOT_OPENED"
        or
        seal.get("global_final_test_lock_state")
        !=
        "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError(
            "Step-10 final-test seal is invalid."
        )

    return {
        "manifest_sha": shas,
        "manifests": manifests,
        "environment": environment,
        "freeze16": freeze16,
        "freeze17": freeze17,
        "freeze18": freeze18,
        "freeze19": freeze19,
        "freeze20": freeze20,
        "step20_by_model": step20_by_model,
        "fingerprint16": fingerprint16,
        "fingerprint17": fingerprint17,
        "fingerprint18": fingerprint18,
        "fingerprint19": fingerprint19,
        "fingerprint20": fingerprint20,
        "test_seal_status": seal["status"],
        "sealed_test_sha256_from_metadata_only": seal.get("sha256"),
    }


# =============================================================================
# 6. RECHECK FROZEN REPOSITORIES
# =============================================================================

def recheck_repositories(
    step17_freeze: Dict[str, Any],
) -> tuple[
    Dict[str, Dict[str, Any]],
    pd.DataFrame,
]:
    repository_map = {}
    rows = []

    for record in step17_freeze["repositories"]:
        repo_id = record["repo_id"]
        path = Path(record["local_path"])

        if repo_id in repository_map:
            raise RuntimeError(
                f"Duplicate frozen repository ID: {repo_id}"
            )

        if not path.exists() or not (path / ".git").exists():
            raise RuntimeError(
                f"Frozen Git repository missing: {path}"
            )

        observed_origin = run_git(
            ["remote", "get-url", "origin"],
            cwd=path,
        )
        observed_commit = run_git(
            ["rev-parse", "HEAD"],
            cwd=path,
        )
        observed_tree = run_git(
            ["rev-parse", "HEAD^{tree}"],
            cwd=path,
        )
        observed_branch = run_git(
            ["branch", "--show-current"],
            cwd=path,
        )
        observed_status = run_git(
            [
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            cwd=path,
        )

        origin_match = (
            normalize_git_url(observed_origin)
            ==
            normalize_git_url(record["repository_url"])
        )
        commit_match = (
            observed_commit == record["commit_sha"]
        )
        tree_match = (
            observed_tree == record["git_tree_sha"]
        )
        branch_match = (
            observed_branch == record["branch"]
        )
        clean = observed_status == ""

        required_hashes_match = True
        checked_files = 0

        for file_record in record["required_file_hashes"]:
            file_path = (
                path / file_record["relative_path"]
            )
            if (
                not file_path.is_file()
                or
                sha256_file(file_path)
                !=
                file_record["sha256"]
            ):
                required_hashes_match = False
            else:
                checked_files += 1

        passed = bool(
            origin_match
            and
            commit_match
            and
            tree_match
            and
            branch_match
            and
            clean
            and
            required_hashes_match
        )

        if not passed:
            raise RuntimeError(
                "\nFROZEN REPOSITORY RECHECK FAILED\n"
                f"Repository : {repo_id}\n"
                f"Origin     : {origin_match}\n"
                f"Commit     : {commit_match}\n"
                f"Tree       : {tree_match}\n"
                f"Branch     : {branch_match}\n"
                f"Clean      : {clean}\n"
                f"File hashes: {required_hashes_match}"
            )

        repository_map[repo_id] = record
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
                "required_file_hashes_match": (
                    required_hashes_match
                ),
                "passed": passed,
            }
        )

    result = pd.DataFrame(rows)

    if (
        len(result) != EXPECTED_FROZEN_REPOSITORY_COUNT
        or
        not result["passed"].all()
    ):
        raise RuntimeError(
            "Frozen repository recheck failed."
        )

    return repository_map, result


# =============================================================================
# 7. IMMUTABILITY / FAILURE HANDLING
# =============================================================================

def canonical_outputs() -> tuple[Path, ...]:
    return (
        ADAPTER_CONTRACT_CSV,
        REGISTRY_CLARIFICATION_CSV,
        REPOSITORY_RECHECK_CSV,
        SOURCE_HASH_CSV,
        ADAPTER_FREEZE_JSON,
        MANIFEST_JSON,
        MANIFEST_SHA,
        HASH_REGISTER_CSV,
        FROZEN_SCRIPT,
        SUMMARY_TXT,
    )


def assert_fresh_target() -> None:
    collisions = [
        path
        for path in canonical_outputs()
        if path.exists()
    ]

    if collisions:
        raise RuntimeError(
            "Step-21 canonical output(s) already exist. "
            "Refusing to overwrite.\n"
            +
            "\n".join(str(path) for path in collisions)
            +
            "\nUse --verify-only if Step 21 already passed."
        )


def archive_partial() -> Optional[Path]:
    if MANIFEST_JSON.exists():
        return None

    existing = [
        path
        for path in canonical_outputs()
        if path.exists()
    ]

    if not existing:
        return None

    FAILED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )

    destination = (
        FAILED_DIR / f"partial_artifacts_{stamp}"
    )
    destination.mkdir(exist_ok=False)

    for path in existing:
        try:
            shutil.move(
                str(path),
                str(destination / path.name),
            )
        except Exception:
            pass

    return destination


# =============================================================================
# 8. BUILD EXACT ADAPTER CONTRACTS
# =============================================================================

def common_contract_fields() -> Dict[str, Any]:
    return {
        "lookback_sessions": LOOKBACK,
        "input_features": FEATURE_COUNT,
        "direction_outputs": DIRECTION_OUTPUTS,
        "direction_horizons": "1;3;5",
        "final_logits_shape": "[B,3]",
        "loss": "BCEWithLogitsLoss(reduction='mean')",
        "loss_horizon_weighting": (
            "unweighted; mean reduction across batch and three horizons"
        ),
        "class_weights": False,
        "stage1_probability_threshold": 0.50,
        "probability_transform_for_metrics": "sigmoid",
        "pretrained_weights": False,
        "backbone_trainable": True,
        "stock_embedding": False,
        "regime_conditioning": False,
        "regime_embedding": False,
        "regime_film": False,
        "auxiliary_regression": False,
        "amp": False,
        "dtype": "float32",
        "source_repository_modified": False,
        "candidate_hyperparameters_changed": False,
        "candidate_count_changed": False,
        "seed_policy_changed": False,
        "selection_protocol_changed": False,
        "validation_results_used_to_define_contract": False,
        "final_test_used_to_define_contract": False,
        "execution_isolation_target": (
            "one frozen implementation repository per Python subprocess"
        ),
        "classification_description": (
            "architecture-adapted multi-horizon directional classification baseline"
        ),
    }


def forecast_contract(
    *,
    model_id: str,
    repo_id: str,
    input_contract: str,
    official_call_contract: str,
    official_return_contract: str,
    forecast_tensor_selection: str,
    orientation_normalization: str,
    special_notes: str,
) -> Dict[str, Any]:
    row = {
        "model_id": model_id,
        "repo_id": repo_id,
        "adapter_type": (
            "complete_frozen_forecast_path_plus_common_postforecast_linear_readout"
        ),
        "native_classification_path": False,
        "input_contract": input_contract,
        "official_call_contract": official_call_contract,
        "official_return_contract": official_return_contract,
        "forecast_tensor_selection": forecast_tensor_selection,
        "forecast_length_sessions": FORECAST_LENGTH,
        "normalized_forecast_shape": "[B,5,63]",
        "orientation_normalization": orientation_normalization,
        "flatten_order": (
            "time-major then feature; reshape contiguous [B,5,63] -> [B,315]"
        ),
        "flattened_shape": "[B,315]",
        "adapter_head": "torch.nn.Linear(315,3,bias=True)",
        "adapter_head_input_features": (
            FORECAST_READOUT_INPUT_DIM
        ),
        "adapter_added_trainable_parameters": (
            FORECAST_ADAPTER_ADDED_PARAMS
        ),
        "activation_after_adapter_head": "none",
        "adapter_dropout": 0.0,
        "special_notes": special_notes,
    }
    row.update(common_contract_fields())
    return row


def build_adapter_contracts() -> list[Dict[str, Any]]:
    contracts = [
        forecast_contract(
            model_id="DLINEAR",
            repo_id="LTSF_LINEAR",
            input_contract="[B,60,63], no transpose",
            official_call_contract=(
                "frozen Model.forward(x) with pred_len=5"
            ),
            official_return_contract="[B,5,63] tensor",
            forecast_tensor_selection="returned tensor",
            orientation_normalization="none",
            special_notes=(
                "Preserve moving-average decomposition and frozen "
                "Linear_Seasonal/Linear_Trend forecasting path. The common "
                "directional readout is strictly post-forecast."
            ),
        ),
        forecast_contract(
            model_id="NLINEAR",
            repo_id="LTSF_LINEAR",
            input_contract="[B,60,63], no transpose",
            official_call_contract=(
                "frozen Model.forward(x) with pred_len=5"
            ),
            official_return_contract="[B,5,63] tensor",
            forecast_tensor_selection="returned tensor",
            orientation_normalization="none",
            special_notes=(
                "Preserve frozen last-value normalization and temporal Linear "
                "forecast mapping. The common readout is strictly post-forecast."
            ),
        ),
        forecast_contract(
            model_id="PATCHTST",
            repo_id="PATCHTST",
            input_contract=(
                "dataset [B,60,63] -> frozen backbone input [B,63,60]"
            ),
            official_call_contract=(
                "frozen PatchTST_backbone.forward(x.transpose(1,2)); "
                "target_window=5"
            ),
            official_return_contract="[B,63,5] tensor",
            forecast_tensor_selection="returned tensor",
            orientation_normalization=(
                "transpose [B,63,5] -> contiguous [B,5,63]"
            ),
            special_notes=(
                "Preserve frozen patching, channel-independent encoder, "
                "configured RevIN behavior and official Flatten_Head forecast "
                "path. Do not replace the internal forecast head."
            ),
        ),
        {
            "model_id": "TIMESNET",
            "repo_id": "TIMESNET_TSLIB",
            "adapter_type": "frozen_native_classification_path",
            "native_classification_path": True,
            "input_contract": "[B,60,63]",
            "official_call_contract": (
                "task_name='classification'; num_class=3; "
                "forward(x_enc=x, x_mark_enc=ones[B,60], "
                "x_dec=None, x_mark_dec=None)"
            ),
            "official_return_contract": "[B,3] raw logits",
            "forecast_tensor_selection": "not applicable",
            "forecast_length_sessions": 0,
            "normalized_forecast_shape": "not applicable",
            "orientation_normalization": "none",
            "flatten_order": "native TimesNet classification pathway",
            "flattened_shape": "not applicable",
            "adapter_head": (
                "none; use frozen TimesNet native classification projection"
            ),
            "adapter_head_input_features": 0,
            "adapter_added_trainable_parameters": 0,
            "activation_after_adapter_head": "none",
            "adapter_dropout": (
                "frozen TimesNet classification dropout from each Step-16 config"
            ),
            "special_notes": (
                "The three native output logits are interpreted as independent "
                "1d/3d/5d binary-direction logits and trained with BCEWithLogitsLoss, "
                "not softmax/cross-entropy. x_mark_enc is an all-one valid-timestep "
                "mask because every Step-14 common sequence has exactly 60 observed "
                "sessions."
            ),
            **common_contract_fields(),
        },
        forecast_contract(
            model_id="ITRANSFORMER",
            repo_id="ITRANSFORMER",
            input_contract="[B,60,63], no transpose",
            official_call_contract=(
                "frozen Model.forward(x, None, None, None) with pred_len=5"
            ),
            official_return_contract=(
                "[B,5,63] forecast tensor; if frozen output_attention=True "
                "returns (forecast, attention), select forecast element only"
            ),
            forecast_tensor_selection=(
                "forecast tensor; tuple first element only when source "
                "output_attention packaging is enabled"
            ),
            orientation_normalization="none",
            special_notes=(
                "Preserve frozen DataEmbedding_inverted, encoder and projector "
                "forecast path. The common readout is strictly post-forecast."
            ),
        ),
        forecast_contract(
            model_id="FREDFORMER",
            repo_id="FREDFORMER",
            input_contract=(
                "[B,60,63], no external transpose; frozen wrapper handles layout"
            ),
            official_call_contract=(
                "frozen Model.forward(x) with pred_len=5"
            ),
            official_return_contract="[B,5,63] tensor",
            forecast_tensor_selection="returned tensor",
            orientation_normalization="none",
            special_notes=(
                "Preserve complete frozen Fredformer wrapper/backbone, paired "
                "frequency branches, nested heads and inverse-frequency "
                "reconstruction. No nested head is replaced."
            ),
        ),
        forecast_contract(
            model_id="FRNET",
            repo_id="FRNET",
            input_contract="[B,60,63], no external transpose",
            official_call_contract=(
                "frozen Model.forward(x) with pred_len=5"
            ),
            official_return_contract="[B,5,63] tensor",
            forecast_tensor_selection="returned tensor",
            orientation_normalization="none",
            special_notes=(
                "Preserve frozen FRNet decomposition/frequency-rotation, "
                "trend pathway and linear_projection forecast output. Do not "
                "replace freq_Linear, fuse_linear or attention projections."
            ),
        ),
        forecast_contract(
            model_id="MLF",
            repo_id="MLF",
            input_contract="[B,60,63]",
            official_call_contract=(
                "frozen Model.forward(x, None, None, None); "
                "all MLF architecture/configuration fields come unchanged "
                "from the selected Step-16 A/B candidate"
            ),
            official_return_contract=(
                "tuple(forecast, scale_all_rec, scale_all_patch); "
                "forecast expected [B,5,63]"
            ),
            forecast_tensor_selection=(
                "first returned forecast tensor only"
            ),
            orientation_normalization="none expected; Step-22 must verify",
            special_notes=(
                "Do not introduce new period/scal_all values in Step 21. "
                "Use each frozen Step-16 candidate's MLF settings unchanged. "
                "Scale reconstruction outputs are ignored by the directional "
                "loss and no extra reconstruction loss is added."
            ),
        ),
    ]

    return contracts


# =============================================================================
# 9. CONTRACT VALIDATION
# =============================================================================

def validate_contracts(
    contracts: list[Dict[str, Any]],
    upstream: Dict[str, Any],
) -> None:
    if len(contracts) != EXPECTED_MODEL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_MODEL_COUNT} contracts; "
            f"found {len(contracts)}."
        )

    ids = [row["model_id"] for row in contracts]

    if (
        len(set(ids)) != EXPECTED_MODEL_COUNT
        or
        set(ids) != set(STAGE1_MODEL_IDS)
    ):
        raise RuntimeError(
            "Adapter contract model set differs from frozen Stage-1 set."
        )

    if tuple(ids) != STAGE1_MODEL_IDS:
        raise RuntimeError(
            "Adapter contracts must be emitted in frozen Stage-1 order."
        )

    step20 = upstream["step20_by_model"]

    native_models = []

    for row in contracts:
        model_id = row["model_id"]
        step20_record = step20[model_id]

        if (
            row["repo_id"]
            !=
            EXPECTED_REPO_BY_MODEL[model_id]
        ):
            raise RuntimeError(
                f"{model_id}: contract repository ID drift."
            )

        if (
            step20_record["repo_id"]
            !=
            row["repo_id"]
        ):
            raise RuntimeError(
                f"{model_id}: Step-20/Step-21 repository mismatch."
            )

        resolution = step20_record["step20_resolution"]

        if resolution not in ACCEPTABLE_STEP20_RESOLUTIONS:
            raise RuntimeError(
                f"{model_id}: unresolved Step-20 state: {resolution}"
            )

        is_native = bool(
            row["native_classification_path"]
        )

        if is_native:
            native_models.append(model_id)

            if (
                resolution
                !=
                "native_classification_path_confirmed_from_step19"
            ):
                raise RuntimeError(
                    f"{model_id}: native classification contract "
                    "not supported by Step 20."
                )

            if row["adapter_head"] != (
                "none; use frozen TimesNet native classification projection"
            ):
                raise RuntimeError(
                    "TimesNet must not receive an extra adapter head."
                )

        else:
            if row["adapter_head"] != (
                "torch.nn.Linear(315,3,bias=True)"
            ):
                raise RuntimeError(
                    f"{model_id}: forecast adapter head drift."
                )

            if int(
                row["adapter_head_input_features"]
            ) != FORECAST_READOUT_INPUT_DIM:
                raise RuntimeError(
                    f"{model_id}: forecast-readout input dimension drift."
                )

            if int(
                row["adapter_added_trainable_parameters"]
            ) != FORECAST_ADAPTER_ADDED_PARAMS:
                raise RuntimeError(
                    f"{model_id}: forecast-adapter parameter count drift."
                )

            if int(
                row["forecast_length_sessions"]
            ) != FORECAST_LENGTH:
                raise RuntimeError(
                    f"{model_id}: forecast length is not 5."
                )

            if row["normalized_forecast_shape"] != "[B,5,63]":
                raise RuntimeError(
                    f"{model_id}: normalized forecast shape drift."
                )

        for forbidden_field in (
            "stock_embedding",
            "regime_conditioning",
            "regime_embedding",
            "regime_film",
            "auxiliary_regression",
            "pretrained_weights",
        ):
            if bool(row[forbidden_field]):
                raise RuntimeError(
                    f"{model_id}: forbidden Stage-1 field enabled: "
                    f"{forbidden_field}"
                )

        if row["loss"] != "BCEWithLogitsLoss(reduction='mean')":
            raise RuntimeError(
                f"{model_id}: loss contract drift."
            )

        if float(
            row["stage1_probability_threshold"]
        ) != 0.50:
            raise RuntimeError(
                f"{model_id}: Stage-1 threshold must remain 0.50."
            )

        if row["dtype"] != "float32" or bool(row["amp"]):
            raise RuntimeError(
                f"{model_id}: FP32 / AMP contract drift."
            )

    if native_models != ["TIMESNET"]:
        raise RuntimeError(
            "Exactly TIMESNET must use native classification. "
            f"Observed: {native_models}"
        )

    if (
        FORECAST_ADAPTER_ADDED_PARAMS
        !=
        948
    ):
        raise RuntimeError(
            "Linear(315,3) parameter-count arithmetic drift."
        )


# =============================================================================
# 10. REGISTRY CLARIFICATION
# =============================================================================

def build_registry_clarification(
    contracts: list[Dict[str, Any]],
    upstream: Dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for contract in contracts:
        model_id = contract["model_id"]
        step20_record = upstream[
            "step20_by_model"
        ][model_id]

        rows.append(
            {
                "model_id": model_id,
                "step20_repo_id": step20_record["repo_id"],
                "step20_resolution": (
                    step20_record["step20_resolution"]
                ),
                "step21_adapter_type": (
                    contract["adapter_type"]
                ),
                "native_classification_path": bool(
                    contract["native_classification_path"]
                ),
                "forecast_readout_added_parameters": int(
                    contract[
                        "adapter_added_trainable_parameters"
                    ]
                ),
                "candidate_count_changed": False,
                "candidate_hyperparameters_changed": False,
                "seed_policy_changed": False,
                "selection_protocol_changed": False,
                "validation_metrics_used": False,
                "final_test_information_used": False,
                "clarification": (
                    "Step 16 already froze architecture identities, A/B "
                    "candidate hyperparameters, Stage-1 training and selection "
                    "protocol. Steps 17-20 exposed exact frozen source/API/"
                    "nested-structure evidence without model execution. Step 21 "
                    "only freezes the deterministic task-interface contract; "
                    "it does not alter any frozen candidate hyperparameter."
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 11. SOURCE HASH REGISTER
# =============================================================================

def source_hashes() -> pd.DataFrame:
    sources = [
        ("current_step00", Path(cfg.__file__).resolve()),
        ("current_step21", SCRIPT_FILE.resolve()),
        ("step16_registry_freeze", STEP16_FREEZE.resolve()),
        ("step17_source_freeze", STEP17_FREEZE.resolve()),
        ("step18_source_api_freeze", STEP18_FREEZE.resolve()),
        ("step19_adapter_locus_freeze", STEP19_FREEZE.resolve()),
        ("step20_transitive_backbone_freeze", STEP20_FREEZE.resolve()),
        (
            "step10_final_test_seal_metadata",
            STEP10_TEST_SEAL.resolve(),
        ),
    ]

    for step, (
        manifest_path,
        sidecar_path,
    ) in UPSTREAM.items():
        sources.extend(
            [
                (
                    f"step{step:02d}_manifest",
                    manifest_path.resolve(),
                ),
                (
                    f"step{step:02d}_manifest_sha",
                    sidecar_path.resolve(),
                ),
            ]
        )

    rows = []

    for role, path in sources:
        if path.name == FORBIDDEN_TEST_FILENAME:
            raise RuntimeError(
                "Final-test parquet entered Step-21 source list."
            )

        if not path.exists():
            raise RuntimeError(
                f"Required Step-21 source missing: {path}"
            )

        rows.append(
            {
                "role": role,
                "file_name": path.name,
                "absolute_path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 12. VERIFY EXISTING STEP 21
# =============================================================================

def verify_existing() -> int:
    print("=" * 136)
    print("STEP 21 — VERIFY EXISTING ADAPTER-CONTRACT FREEZE")
    print("=" * 136)

    step00 = verify_step00()
    upstream = verify_upstream(step00)

    manifest_sha = verify_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 21",
    )

    manifest = json.loads(
        MANIFEST_JSON.read_text(encoding="utf-8")
    )

    if (
        manifest.get("step", {}).get("status") != "PASS"
        or
        manifest.get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00["fingerprint"]
    ):
        print("[FAIL] Stored Step-21 status/fingerprint mismatch.")
        return 20

    for step in range(1, 21):
        field = f"step{step:02d}_manifest_sha256"
        if (
            manifest.get(field)
            !=
            upstream["manifest_sha"][step]
        ):
            print(
                f"[FAIL] Upstream chain mismatch: {field}"
            )
            return 21

    expected_fingerprints = {
        "step16_registry_freeze_fingerprint_sha256": (
            upstream["fingerprint16"]
        ),
        "step17_source_freeze_fingerprint_sha256": (
            upstream["fingerprint17"]
        ),
        "step18_source_api_fingerprint_sha256": (
            upstream["fingerprint18"]
        ),
        "step19_adapter_locus_fingerprint_sha256": (
            upstream["fingerprint19"]
        ),
        "step20_transitive_backbone_fingerprint_sha256": (
            upstream["fingerprint20"]
        ),
    }

    for field, expected in expected_fingerprints.items():
        if manifest.get(field) != expected:
            print(
                f"[FAIL] Upstream fingerprint mismatch: {field}"
            )
            return 22

    failures = []

    for name, record in manifest.get(
        "artifacts",
        {},
    ).items():
        path = Path(record["path"])

        if not path.exists():
            failures.append(f"missing: {name}")
            continue

        if sha256_file(path) != record["sha256"]:
            failures.append(f"hash mismatch: {name}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        print("=" * 136)
        print("STEP 21 VERIFICATION STATUS: FAIL")
        print("=" * 136)
        return 23

    freeze, contract_fingerprint = (
        verify_fingerprint_file(
            ADAPTER_FREEZE_JSON,
            "adapter_contract_fingerprint_sha256",
        )
    )

    if (
        manifest.get(
            "adapter_contract_fingerprint_sha256"
        )
        !=
        contract_fingerprint
    ):
        print(
            "[FAIL] Manifest/adapter-contract fingerprint mismatch."
        )
        return 24

    repository_map, repository_recheck = (
        recheck_repositories(
            upstream["freeze17"]
        )
    )

    if not repository_recheck["passed"].all():
        print(
            "[FAIL] Frozen repository recheck failed."
        )
        return 25

    if (
        sha256_file(
            Path(step00["lock_path"])
        )
        !=
        step00["lock_sha256"]
    ):
        print(
            "[FAIL] Global final-test lock changed."
        )
        return 26

    print(
        "[PASS] Configuration fingerprint : "
        f"{step00['fingerprint']}"
    )
    print(
        "[PASS] Step-20 manifest          : "
        f"{upstream['manifest_sha'][20]}"
    )
    print(
        "[PASS] Step-16 registry freeze   : "
        f"{upstream['fingerprint16']}"
    )
    print(
        "[PASS] Step-17 source freeze     : "
        f"{upstream['fingerprint17']}"
    )
    print(
        "[PASS] Step-18 source API freeze : "
        f"{upstream['fingerprint18']}"
    )
    print(
        "[PASS] Step-19 locus freeze      : "
        f"{upstream['fingerprint19']}"
    )
    print(
        "[PASS] Step-20 transitive freeze : "
        f"{upstream['fingerprint20']}"
    )
    print(
        "[PASS] Frozen repositories       : "
        f"{len(repository_map)}"
    )
    print(
        "[PASS] Adapter contracts         : "
        f"{len(freeze['contracts'])}"
    )
    print(
        "[PASS] Step-21 adapter freeze    : "
        f"{contract_fingerprint}"
    )
    print(
        "[PASS] Step-21 manifest          : "
        f"{manifest_sha}"
    )
    print(
        "[PASS] Step-10 test seal         : "
        f"{upstream['test_seal_status']}"
    )
    print(
        "[PASS] Global final-test state   : "
        f"{step00['state']}"
    )
    print("=" * 136)
    print("STEP 21 VERIFICATION STATUS: PASS")
    print("=" * 136)

    return 0


# =============================================================================
# 13. MAIN RUN
# =============================================================================

def run() -> int:
    print("=" * 136)
    print(STEP_NAME)
    print("=" * 136)

    stage = "verify"

    try:
        print(
            "[1/9] Verifying Step 00 and immutable Step 01 -> Step 20 chain..."
        )

        step00 = verify_step00()
        upstream = verify_upstream(step00)

        print(
            "       Configuration fingerprint: "
            f"{step00['fingerprint']}"
        )
        print(
            "       Step-20 manifest          : "
            f"{upstream['manifest_sha'][20]}"
        )
        print(
            "       Step-16 registry freeze   : "
            f"{upstream['fingerprint16']}"
        )
        print(
            "       Step-17 source freeze     : "
            f"{upstream['fingerprint17']}"
        )
        print(
            "       Step-18 source API freeze : "
            f"{upstream['fingerprint18']}"
        )
        print(
            "       Step-19 locus freeze      : "
            f"{upstream['fingerprint19']}"
        )
        print(
            "       Step-20 transitive freeze : "
            f"{upstream['fingerprint20']}"
        )
        print(
            "       Stage-1 architectures     : "
            f"{EXPECTED_MODEL_COUNT}"
        )
        print(
            "       Step-10 test seal         : "
            f"{upstream['test_seal_status']}"
        )
        print(
            "       Global final-test state   : "
            f"{step00['state']}"
        )
        print(
            "       Third-party imports       : PROHIBITED"
        )
        print(
            "       Model instantiation       : PROHIBITED"
        )
        print(
            "       Smoke tests               : PROHIBITED"
        )
        print(
            "       Model training            : PROHIBITED"
        )
        print(
            "       Final-test parquet access : PROHIBITED"
        )

        stage = "collision_guard"

        print(
            "[2/9] Checking immutable Step-21 output target..."
        )

        assert_fresh_target()
        cfg.create_directories()

        stage = "repository_recheck"

        print(
            "[3/9] Rechecking the eight frozen Step-17 repository snapshots..."
        )

        (
            repository_map,
            repository_recheck,
        ) = recheck_repositories(
            upstream["freeze17"]
        )

        print(
            "       Repositories rechecked      : "
            f"{len(repository_recheck)}"
        )
        print(
            "       Origins/commits/trees/branches exact: "
            f"{bool(repository_recheck['passed'].all())}"
        )
        print(
            "       Worktrees clean incl. untracked: "
            f"{bool(repository_recheck['worktree_clean_including_untracked'].all())}"
        )
        print(
            "       Required-file hashes exact  : "
            f"{bool(repository_recheck['required_file_hashes_match'].all())}"
        )

        stage = "contract_build"

        print(
            "[4/9] Building exact performance-blind Stage-1 adapter contracts..."
        )

        contracts = build_adapter_contracts()

        validate_contracts(
            contracts=contracts,
            upstream=upstream,
        )

        contracts_df = pd.DataFrame(
            contracts
        )

        clarification_df = (
            build_registry_clarification(
                contracts=contracts,
                upstream=upstream,
            )
        )

        for row in contracts:
            if row["native_classification_path"]:
                print(
                    f"       {row['model_id']:<14} | "
                    "native TimesNet classification -> [B,3] logits"
                )
            else:
                print(
                    f"       {row['model_id']:<14} | "
                    "[B,5,63] -> flatten [B,315] -> Linear(315,3)"
                )

        stage = "contract_validation"

        print(
            "[5/9] Validating fairness, no-tuning and Step-20 compatibility..."
        )

        forecast_count = int(
            (
                contracts_df[
                    "native_classification_path"
                ]
                ==
                False
            ).sum()
        )
        native_count = int(
            (
                contracts_df[
                    "native_classification_path"
                ]
                ==
                True
            ).sum()
        )

        if forecast_count != 7 or native_count != 1:
            raise RuntimeError(
                "Expected 7 forecast-readout contracts and "
                "1 native-classification contract."
            )

        print(
            "       Forecast-readout adapters     : "
            f"{forecast_count}"
        )
        print(
            "       Native classification adapters: "
            f"{native_count}"
        )
        print(
            "       Common forecast adapter params : "
            f"{FORECAST_ADAPTER_ADDED_PARAMS}"
        )
        print(
            "       Validation metrics consulted   : False"
        )
        print(
            "       Final-test information used    : False"
        )
        print(
            "       Stage-16 candidate configs changed: False"
        )

        stage = "write"

        print(
            "[6/9] Writing adapter-contract audits and provenance..."
        )

        write_csv(
            ADAPTER_CONTRACT_CSV,
            contracts_df,
        )
        write_csv(
            REGISTRY_CLARIFICATION_CSV,
            clarification_df,
        )
        write_csv(
            REPOSITORY_RECHECK_CSV,
            repository_recheck,
        )
        write_csv(
            SOURCE_HASH_CSV,
            source_hashes(),
        )

        stage = "freeze"

        print(
            "[7/9] Freezing executable contracts before any model import/instantiation..."
        )

        if (
            sha256_file(
                Path(step00["lock_path"])
            )
            !=
            step00["lock_sha256"]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step 21."
            )

        freeze_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "STAGE1_DIRECTIONAL_ADAPTER_CONTRACTS_FROZEN"
            ),
            "generated_at_utc": utc_now(),
            "configuration_fingerprint_sha256": (
                step00["fingerprint"]
            ),
            "step16_manifest_sha256": (
                upstream["manifest_sha"][16]
            ),
            "step17_manifest_sha256": (
                upstream["manifest_sha"][17]
            ),
            "step18_manifest_sha256": (
                upstream["manifest_sha"][18]
            ),
            "step19_manifest_sha256": (
                upstream["manifest_sha"][19]
            ),
            "step20_manifest_sha256": (
                upstream["manifest_sha"][20]
            ),
            "step16_registry_freeze_fingerprint_sha256": (
                upstream["fingerprint16"]
            ),
            "step17_source_freeze_fingerprint_sha256": (
                upstream["fingerprint17"]
            ),
            "step18_source_api_fingerprint_sha256": (
                upstream["fingerprint18"]
            ),
            "step19_adapter_locus_fingerprint_sha256": (
                upstream["fingerprint19"]
            ),
            "step20_transitive_backbone_fingerprint_sha256": (
                upstream["fingerprint20"]
            ),
            "common_contract": {
                "dataset_input_shape": "[B,60,63]",
                "direction_horizons_sessions": list(HORIZONS),
                "direction_output_shape": "[B,3] raw logits",
                "direction_loss": (
                    "BCEWithLogitsLoss(reduction='mean')"
                ),
                "class_weights": False,
                "stage1_probability_threshold": 0.50,
                "probability_transform_for_metrics": "sigmoid",
                "forecast_style_models": {
                    "count": 7,
                    "forecast_length_sessions": (
                        FORECAST_LENGTH
                    ),
                    "normalized_forecast_shape": "[B,5,63]",
                    "flattened_shape": "[B,315]",
                    "common_readout": (
                        "torch.nn.Linear(315,3,bias=True)"
                    ),
                    "added_trainable_parameters": (
                        FORECAST_ADAPTER_ADDED_PARAMS
                    ),
                    "post_head_activation": "none",
                },
                "native_classification_exception": (
                    "TIMESNET uses its frozen native classification "
                    "path with num_class=3 and no extra adapter head."
                ),
                "training": (
                    "end-to-end from random initialization under "
                    "the already frozen Step-16 training protocol"
                ),
                "pretrained_weights": False,
                "stock_embedding": False,
                "regime_conditioning": False,
                "regime_film": False,
                "auxiliary_regression": False,
                "dtype": "float32",
                "amp": False,
                "source_repository_modification": False,
                "repository_execution_isolation": (
                    "one frozen implementation repository per "
                    "Python subprocess"
                ),
            },
            "contracts": contracts,
            "registry_clarification": {
                "performance_observed_before_contract_freeze": False,
                "validation_metrics_used": False,
                "final_test_used": False,
                "candidate_counts_changed": False,
                "candidate_hyperparameters_changed": False,
                "seed_policy_changed": False,
                "selection_protocol_changed": False,
                "reason": (
                    "Steps 17-20 resolved exact frozen source/API/"
                    "architecture evidence without model performance. "
                    "Step 21 converts the already frozen Stage-1 "
                    "task-adaptation intent into a deterministic executable "
                    "interface before any model import or execution."
                ),
            },
            "pipeline_state": {
                "development_interface_frozen": True,
                "stage1_model_registry_frozen": True,
                "official_source_snapshots_frozen": True,
                "source_apis_audited": True,
                "adapter_loci_audited": True,
                "transitive_backbones_audited": True,
                "adapter_contracts_frozen": True,
                "task_adapters_implemented": False,
                "models_instantiated": False,
                "smoke_tests_run": False,
                "models_fitted": False,
                "feature_selection_applied": False,
                "probability_threshold_selection_performed": False,
            },
            "final_test_protection": {
                "step10_seal_status": (
                    upstream["test_seal_status"]
                ),
                "sealed_test_sha256_from_metadata_only": (
                    upstream[
                        "sealed_test_sha256_from_metadata_only"
                    ]
                ),
                "final_test_parquet_opened_or_hashed_by_step21": False,
                "final_test_performance_computed": False,
                "global_final_test_state": step00["state"],
            },
        }

        contract_fingerprint = (
            cfg.canonical_json_sha256(
                freeze_payload
            )
        )

        freeze_payload[
            "adapter_contract_fingerprint_sha256"
        ] = contract_fingerprint

        atomic_json(
            ADAPTER_FREEZE_JSON,
            freeze_payload,
        )

        shutil.copy2(
            SCRIPT_FILE,
            FROZEN_SCRIPT,
        )

        atomic_text(
            SUMMARY_TXT,
            "\n".join(
                [
                    STEP_NAME,
                    "=" * 108,
                    "Status: PASS",
                    (
                        "Configuration fingerprint: "
                        f"{step00['fingerprint']}"
                    ),
                    (
                        "Step-20 transitive fingerprint: "
                        f"{upstream['fingerprint20']}"
                    ),
                    "Stage-1 model contracts: 8",
                    "Forecast-output adapters: 7",
                    "Native classification adapters: 1 (TIMESNET)",
                    (
                        "Forecast adapter: "
                        "Linear(315,3,bias=True), added params=948"
                    ),
                    "Validation metrics used: False",
                    "Final-test information used: False",
                    "Third-party model imported: False",
                    "Model instantiated: False",
                    "Adapter implemented: False",
                    "Smoke tests run: False",
                    "Models fitted: False",
                    "Final-test parquet accessed: False",
                    (
                        "Global final-test state: "
                        f"{step00['state']}"
                    ),
                ]
            )
            +
            "\n",
        )

        output_paths_pre_manifest = (
            ADAPTER_CONTRACT_CSV,
            REGISTRY_CLARIFICATION_CSV,
            REPOSITORY_RECHECK_CSV,
            SOURCE_HASH_CSV,
            ADAPTER_FREEZE_JSON,
            FROZEN_SCRIPT,
            SUMMARY_TXT,
        )

        write_csv(
            HASH_REGISTER_CSV,
            pd.DataFrame(
                [
                    {
                        "file_name": path.name,
                        "file_path": str(path),
                        "relative_path": (
                            relative_to_experiment(path)
                        ),
                        "size_bytes": int(
                            path.stat().st_size
                        ),
                        "sha256": sha256_file(path),
                    }
                    for path in output_paths_pre_manifest
                ]
            ),
        )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "step": {
                "step_number": STEP_NUMBER,
                "step_id": STEP_ID,
                "step_name": STEP_NAME,
                "status": "PASS",
            },
            "frozen_at_utc": utc_now(),
            "experiment_tag": cfg.EXPERIMENT_TAG,
            "configuration_fingerprint_sha256": (
                step00["fingerprint"]
            ),
            "step00_manifest_sha256": (
                step00["step00_manifest_sha256"]
            ),
            **{
                f"step{step:02d}_manifest_sha256": (
                    upstream["manifest_sha"][step]
                )
                for step in range(1, 21)
            },
            "step16_registry_freeze_fingerprint_sha256": (
                upstream["fingerprint16"]
            ),
            "step17_source_freeze_fingerprint_sha256": (
                upstream["fingerprint17"]
            ),
            "step18_source_api_fingerprint_sha256": (
                upstream["fingerprint18"]
            ),
            "step19_adapter_locus_fingerprint_sha256": (
                upstream["fingerprint19"]
            ),
            "step20_transitive_backbone_fingerprint_sha256": (
                upstream["fingerprint20"]
            ),
            "adapter_contract_fingerprint_sha256": (
                contract_fingerprint
            ),
            "counts": {
                "stage1_models": int(
                    len(contracts_df)
                ),
                "forecast_output_adapters": forecast_count,
                "native_classification_adapters": native_count,
                "forecast_adapter_added_params_each": (
                    FORECAST_ADAPTER_ADDED_PARAMS
                ),
            },
            "pipeline_state": {
                "development_interface_frozen": True,
                "stage1_model_registry_frozen": True,
                "official_source_snapshots_frozen": True,
                "source_apis_audited": True,
                "adapter_loci_audited": True,
                "transitive_backbones_audited": True,
                "adapter_contracts_frozen": True,
                "task_adapters_implemented": False,
                "models_instantiated": False,
                "smoke_tests_run": False,
                "models_fitted": False,
                "feature_selection_applied": False,
                "probability_threshold_selection_performed": False,
            },
            "test_protection": {
                "step10_seal_status": (
                    upstream["test_seal_status"]
                ),
                "global_final_test_state": step00["state"],
                "sealed_test_parquet_opened_or_hashed_by_step21": False,
                "final_test_performance_computed": False,
            },
            "artifacts": {
                path.name: artifact_record(path)
                for path in (
                    *output_paths_pre_manifest,
                    HASH_REGISTER_CSV,
                )
            },
        }

        atomic_json(
            MANIFEST_JSON,
            manifest,
        )

        manifest_sha = sha256_file(
            MANIFEST_JSON
        )

        atomic_text(
            MANIFEST_SHA,
            f"{manifest_sha}  {MANIFEST_JSON.name}\n",
        )

        stage = "validate"

        print(
            "[8/9] Validating frozen Step-21 outputs..."
        )

        verify_sidecar(
            MANIFEST_JSON,
            MANIFEST_SHA,
            "Step 21",
        )

        stored_manifest = json.loads(
            MANIFEST_JSON.read_text(encoding="utf-8")
        )

        for record in stored_manifest[
            "artifacts"
        ].values():
            path = Path(record["path"])
            if (
                not path.exists()
                or
                sha256_file(path) != record["sha256"]
            ):
                raise RuntimeError(
                    f"Frozen Step-21 artifact mismatch: {path}"
                )

        if (
            sha256_file(
                Path(step00["lock_path"])
            )
            !=
            step00["lock_sha256"]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step-21 validation."
            )

        stage = "report"

        print(
            "[9/9] Final Step-21 report..."
        )

        print()
        print("-" * 136)
        print("FROZEN ADAPTER CONTRACTS")
        print("-" * 136)

        for row in contracts_df.itertuples(index=False):
            print(
                f"{row.model_id:<14} | {row.adapter_type}"
            )
            print(
                f"  input     : {row.input_contract}"
            )
            print(
                f"  call      : {row.official_call_contract}"
            )
            print(
                f"  return    : {row.official_return_contract}"
            )
            print(
                f"  normalize : {row.orientation_normalization}"
            )
            print(
                f"  logits    : {row.final_logits_shape}"
            )
            print(
                f"  head      : {row.adapter_head}"
            )

        print("-" * 136)
        print("COMMON FAIRNESS / TRAINING CONSTRAINTS")
        print("-" * 136)
        print(
            "Input history                    : 60 sessions"
        )
        print(
            "Predictors                       : identical 63-feature tensor"
        )
        print(
            "Forecast representation          : 5 sessions for 7 forecast models"
        )
        print(
            "Directional outputs              : 3 raw logits (1d / 3d / 5d)"
        )
        print(
            "Forecast adapter                 : Linear(315,3,bias=True)"
        )
        print(
            "Added adapter parameters         : 948 per forecast model"
        )
        print(
            "Loss                             : BCEWithLogitsLoss(mean)"
        )
        print(
            "Stage-1 threshold                : 0.50"
        )
        print(
            "Stock embedding                  : prohibited"
        )
        print(
            "Regime conditioning / FiLM       : prohibited"
        )
        print(
            "Auxiliary regression             : prohibited"
        )
        print(
            "Pretrained weights               : prohibited"
        )
        print(
            "Backbone repository modification : prohibited"
        )
        print(
            "Backbone training                : end-to-end"
        )
        print(
            "dtype / AMP                      : float32 / disabled"
        )

        print("-" * 136)
        print("REGISTRY / PERFORMANCE-BLIND CLARIFICATION")
        print("-" * 136)
        print(
            "Candidate counts changed         : False"
        )
        print(
            "Candidate hyperparameters changed: False"
        )
        print(
            "Seed policy changed              : False"
        )
        print(
            "Selection protocol changed       : False"
        )
        print(
            "Validation metrics used          : False"
        )
        print(
            "Final-test information used      : False"
        )

        print("-" * 136)
        print("FREEZE STATE")
        print("-" * 136)
        print(
            "Development interface            : FROZEN"
        )
        print(
            "Stage-1 registry                 : FROZEN"
        )
        print(
            "Official source snapshots        : FROZEN"
        )
        print(
            "Source APIs                      : AUDITED + FROZEN"
        )
        print(
            "Architecture loci                : AUDITED + FROZEN"
        )
        print(
            "Transitive backbones             : AUDITED + FROZEN"
        )
        print(
            "Executable adapter contracts     : FROZEN"
        )
        print(
            "Task adapters                    : NOT YET IMPLEMENTED"
        )
        print(
            "Models instantiated              : False"
        )
        print(
            "Smoke tests                      : NOT RUN"
        )
        print(
            "Models fitted                    : False"
        )
        print(
            "Final test accessed              : False"
        )
        print(
            "Global final-test state          : "
            f"{step00['state']}"
        )

        print("-" * 136)
        print("OUTPUT FILES")
        print("-" * 136)
        print(
            f"Adapter contracts       : {ADAPTER_CONTRACT_CSV}"
        )
        print(
            f"Registry clarification  : {REGISTRY_CLARIFICATION_CSV}"
        )
        print(
            f"Repository recheck      : {REPOSITORY_RECHECK_CSV}"
        )
        print(
            f"Adapter freeze          : {ADAPTER_FREEZE_JSON}"
        )
        print(
            f"Adapter fingerprint     : {contract_fingerprint}"
        )
        print(
            f"Manifest                : {MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256        : {manifest_sha}"
        )

        print("=" * 136)
        print(
            "PASS: Exact Stage-1 task-adapter contracts were frozen "
            "before third-party model execution."
        )
        print(
            "PASS: Seven forecast-style models preserve their complete "
            "frozen five-session forecast path and receive the same "
            "948-parameter post-forecast directional readout."
        )
        print(
            "PASS: TimesNet uses its frozen native classification path "
            "with exactly three raw logits and no extra adapter head."
        )
        print(
            "PASS: Step-16 A/B candidate hyperparameters, seed policy "
            "and selection protocol remain unchanged."
        )
        print(
            "PASS: No validation performance or final-test information "
            "was used to define these contracts."
        )
        print(
            "PASS: No third-party model was imported, instantiated, "
            "smoke-tested or trained."
        )
        print(
            "PASS: The sealed final-test parquet remained unopened."
        )
        print("=" * 136)
        print(
            "Do not implement adapters yet. Paste the complete terminal "
            "output for audit, then run --verify-only if instructed."
        )

        return 0

    except Exception as exc:
        archived = archive_partial()

        FAILED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        stamp = (
            datetime.now(timezone.utc)
            .strftime("%Y%m%dT%H%M%SZ")
        )

        failure_path = (
            FAILED_DIR
            /
            f"21_adapter_contract_freeze_FAILED_{stamp}.json"
        )

        atomic_json(
            failure_path,
            {
                "step": STEP_NAME,
                "status": "FAIL",
                "failed_at_utc": utc_now(),
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "python_executable": sys.executable,
                "archived_partial_artifacts": (
                    None
                    if archived is None
                    else str(archived)
                ),
                "canonical_manifest_exists": (
                    MANIFEST_JSON.exists()
                ),
            },
        )

        print()
        print("=" * 136)
        print(
            "STEP 21 FAILED — ADAPTER CONTRACTS NOT FROZEN"
        )
        print("=" * 136)
        print(f"Stage      : {stage}")
        print(f"Error type : {type(exc).__name__}")
        print(f"Error      : {exc}")
        print(f"Diagnostic : {failure_path}")

        if archived is not None:
            print(
                f"Archived partial Step-21 outputs: {archived}"
            )

        print("=" * 136)
        print(
            "Do not proceed to adapter implementation, smoke tests, "
            "or model training."
        )

        return 30


# =============================================================================
# 14. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean-V3 freeze of exact Stage-1 directional adapter contracts."
        )
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an existing frozen Step-21 adapter-contract freeze "
            "without importing/executing third-party model code."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        return verify_existing()
    return run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStep 21 interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print()
        print("=" * 136)
        print("STEP 21 UNHANDLED FAILURE")
        print("=" * 136)
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        print("=" * 136)
        sys.exit(1)
