#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 19 — CLEAN V3 FROZEN ARCHITECTURE ADAPTATION-LOCUS AUDIT

Purpose
-------
Inspect the exact Step-17/18 frozen model implementations deeply enough to
identify WHERE a later directional-classification adapter may attach, without
importing or executing third-party model code.

For each of the eight frozen Stage-1 architectures this stage records:

- exact frozen repository / source-file identity;
- preferred source class and exact Step-18 API signatures;
- __init__ self-attribute assignments;
- likely terminal head/projection CANDIDATES;
- forward() assignment/data-flow statements;
- forward() calls rooted at self.*;
- forward() return expressions;
- tensor shape-changing operations;
- task-name / classification / forecast dispatch conditions;
- helper-method signatures;
- Step-18 configuration references;
- conservative adapter-readiness classification.

This is a STATIC architecture-locus audit only.  Heuristic terminal candidates
are evidence for Step 20; they are NOT an executable adapter contract.

This stage DOES NOT:
- import third-party source modules;
- execute third-party source code;
- edit frozen repositories;
- fetch or pull repositories;
- instantiate models;
- finalize executable adapter contracts;
- implement task adapters;
- run forward/backward smoke tests;
- train models;
- perform feature selection;
- estimate probability thresholds;
- open or hash the sealed final-test parquet.

Run
---
    python step19_audit_adapter_loci_v3.py

Verify
------
    python step19_audit_adapter_loci_v3.py --verify-only
"""

from __future__ import annotations

import argparse
import ast
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
        "step19_audit_adapter_loci_v3.py must be inside "
        "<repository>/q1_revision/."
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg


# =============================================================================
# 1. FROZEN DESIGN
# =============================================================================

STEP_NUMBER = 19
STEP_ID = "19"
STEP_NAME = "STEP 19 — FROZEN ARCHITECTURE ADAPTATION-LOCUS AUDIT"
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
TOTAL_FEATURE_COUNT = 63
DIRECTION_OUTPUTS = 3

MAX_AST_TEXT_LENGTH = 700

FORBIDDEN_TEST_FILENAME = "10_final_test_locked_panel.parquet"

SHAPE_OPERATION_NAMES = {
    "permute",
    "transpose",
    "reshape",
    "view",
    "flatten",
    "squeeze",
    "unsqueeze",
    "repeat",
    "expand",
    "contiguous",
    "rearrange",
}

TERMINAL_ATTRIBUTE_TOKENS = (
    "head",
    "projection",
    "project",
    "predict",
    "prediction",
    "classifier",
    "classification",
    "fc",
    "out",
    "output",
    "decoder",
)

TASK_SIGNAL_TOKENS = (
    "task_name",
    "classification",
    "forecast",
    "long_term_forecast",
    "short_term_forecast",
    "imputation",
    "anomaly",
)


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

STEP10_TEST_SEAL = (
    cfg.MANIFEST_DIR / "10_final_test_seal.json"
)


# =============================================================================
# 3. OUTPUT PATHS
# =============================================================================

MODEL_LOCUS_SUMMARY_CSV = (
    cfg.AUDIT_DIR / "19_model_adaptation_locus_summary.csv"
)

INIT_ASSIGNMENT_AUDIT_CSV = (
    cfg.AUDIT_DIR / "19_init_self_assignment_audit.csv"
)

FORWARD_FLOW_AUDIT_CSV = (
    cfg.AUDIT_DIR / "19_forward_dataflow_audit.csv"
)

FORWARD_SELF_CALL_AUDIT_CSV = (
    cfg.AUDIT_DIR / "19_forward_self_call_audit.csv"
)

RETURN_AUDIT_CSV = (
    cfg.AUDIT_DIR / "19_forward_return_audit.csv"
)

SHAPE_OPERATION_AUDIT_CSV = (
    cfg.AUDIT_DIR / "19_shape_operation_audit.csv"
)

TASK_DISPATCH_AUDIT_CSV = (
    cfg.AUDIT_DIR / "19_task_dispatch_audit.csv"
)

METHOD_SIGNATURE_AUDIT_CSV = (
    cfg.AUDIT_DIR / "19_method_signature_audit.csv"
)

REPOSITORY_RECHECK_CSV = (
    cfg.AUDIT_DIR / "19_repository_recheck.csv"
)

SOURCE_HASH_CSV = (
    cfg.AUDIT_DIR / "19_source_file_hashes.csv"
)

ADAPTER_LOCUS_FREEZE_JSON = (
    cfg.MANIFEST_DIR / "19_adapter_locus_freeze.json"
)

MANIFEST_JSON = (
    cfg.MANIFEST_DIR / "19_adapter_locus_manifest.json"
)

MANIFEST_SHA = (
    cfg.MANIFEST_DIR / "19_adapter_locus_manifest.sha256"
)

HASH_REGISTER_CSV = (
    cfg.AUDIT_DIR / "19_adapter_locus_hashes.csv"
)

FROZEN_SCRIPT = (
    cfg.MANIFEST_DIR / "step19_audit_adapter_loci_frozen.py"
)

SUMMARY_TXT = (
    cfg.REPORT_DIR / "19_adapter_locus_summary.txt"
)

FAILED_DIR = (
    cfg.AUDIT_DIR / "step19_failed_attempts"
)


# =============================================================================
# 4. GENERIC HELPERS
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

    temporary.write_text(
        text,
        encoding="utf-8",
    )

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
        raise RuntimeError(
            "git executable was not found on PATH."
        )

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
        [
            git_executable(),
            *arguments,
        ],
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

    manifest_path, manifest_sha, status = cfg.verify_existing_freeze(
        configuration_payload=payload,
        configuration_fingerprint=fingerprint,
    )

    if status != "VERIFIED_EXISTING":
        raise RuntimeError(
            "Step-00 freeze is not VERIFIED_EXISTING."
        )

    lock_path = cfg.get_final_test_lock_path()

    lock = json.loads(
        lock_path.read_text(
            encoding="utf-8"
        )
    )

    if lock.get("state") != "LOCKED_NOT_OPENED":
        raise RuntimeError(
            "Global final-test state is not LOCKED_NOT_OPENED."
        )

    if lock.get("final_test_opened") is not False:
        raise RuntimeError(
            "global final_test_opened != False."
        )

    if lock.get("final_test_scored") is not False:
        raise RuntimeError(
            "global final_test_scored != False."
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
        STEP01_ENVIRONMENT_AUDIT.read_text(
            encoding="utf-8"
        )
    )

    expected_python = (
        audit
        .get("system", {})
        .get("python_executable")
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
        sidecar_path.read_text(
            encoding="utf-8"
        )
        .strip()
        .split()[0]
    )

    observed = sha256_file(manifest_path)

    if observed != expected:
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

    for key in (
        "artifacts",
        "canonical_artifacts",
    ):
        for record in manifest.get(
            key,
            {},
        ).values():
            if not isinstance(record, dict):
                continue

            path_text = record.get("path")

            if (
                path_text
                and
                Path(path_text).name
                ==
                filename
            ):
                matches.append(record)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest artifact named "
            f"{filename}; found {len(matches)}."
        )

    return matches[0]


def verify_upstream(
    step00: Dict[str, Any],
) -> Dict[str, Any]:
    shas: Dict[int, str] = {}
    manifests: Dict[int, Dict[str, Any]] = {}

    for step in range(1, 19):
        manifest_path, sidecar_path = UPSTREAM[step]

        shas[step] = verify_sidecar(
            manifest_path,
            sidecar_path,
            f"Step {step:02d}",
        )

        manifests[step] = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

    if manifests[1].get("status") != "PASS":
        raise RuntimeError(
            "Step-01 status is not PASS."
        )

    if (
        manifests[1].get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00["fingerprint"]
    ):
        raise RuntimeError(
            "Step-01 fingerprint mismatch."
        )

    if (
        manifests[2]
        .get("step", {})
        .get("status")
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Step-02 status is not PASS."
        )

    for step in range(3, 19):
        manifest = manifests[step]

        if (
            manifest
            .get("step", {})
            .get("status")
            !=
            "PASS"
        ):
            raise RuntimeError(
                f"Step-{step:02d} status is not PASS."
            )

        if (
            manifest.get(
                "configuration_fingerprint_sha256"
            )
            !=
            step00["fingerprint"]
        ):
            raise RuntimeError(
                f"Step-{step:02d} fingerprint mismatch."
            )

        for previous in range(1, step):
            field = (
                f"step{previous:02d}_manifest_sha256"
            )

            if (
                field in manifest
                and
                manifest[field]
                !=
                shas[previous]
            ):
                raise RuntimeError(
                    f"Step-{step:02d} chain mismatch: {field}"
                )

    environment = verify_environment()

    step16_manifest = manifests[16]
    step17_manifest = manifests[17]
    step18_manifest = manifests[18]

    # -------------------------------------------------------------------------
    # Step 16 registry freeze.
    # -------------------------------------------------------------------------
    if not STEP16_FREEZE.exists():
        raise RuntimeError(
            "Step-16 registry freeze missing."
        )

    freeze16 = json.loads(
        STEP16_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    payload16 = dict(freeze16)
    registry_fingerprint = payload16.pop(
        "registry_freeze_fingerprint_sha256",
        None,
    )

    if not registry_fingerprint:
        raise RuntimeError(
            "Step-16 registry freeze lacks fingerprint."
        )

    if (
        cfg.canonical_json_sha256(payload16)
        !=
        registry_fingerprint
    ):
        raise RuntimeError(
            "Step-16 registry freeze fingerprint mismatch."
        )

    if (
        step16_manifest.get(
            "registry_freeze_fingerprint_sha256"
        )
        !=
        registry_fingerprint
    ):
        raise RuntimeError(
            "Step-16 manifest/freeze fingerprint mismatch."
        )

    if (
        freeze16.get("status")
        !=
        "STAGE1_MODEL_REGISTRY_AND_SELECTION_PROTOCOL_FROZEN"
    ):
        raise RuntimeError(
            "Step-16 registry is not formally frozen."
        )

    if freeze16.get("training_started") is not False:
        raise RuntimeError(
            "Step-16 says training has started."
        )

    # -------------------------------------------------------------------------
    # Step 17 source freeze.
    # -------------------------------------------------------------------------
    if not STEP17_FREEZE.exists():
        raise RuntimeError(
            "Step-17 source freeze missing."
        )

    freeze17 = json.loads(
        STEP17_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    payload17 = dict(freeze17)
    source_fingerprint = payload17.pop(
        "source_freeze_fingerprint_sha256",
        None,
    )

    if not source_fingerprint:
        raise RuntimeError(
            "Step-17 source freeze lacks fingerprint."
        )

    if (
        cfg.canonical_json_sha256(payload17)
        !=
        source_fingerprint
    ):
        raise RuntimeError(
            "Step-17 source freeze fingerprint mismatch."
        )

    if (
        step17_manifest.get(
            "source_freeze_fingerprint_sha256"
        )
        !=
        source_fingerprint
    ):
        raise RuntimeError(
            "Step-17 manifest/freeze fingerprint mismatch."
        )

    if (
        freeze17.get("status")
        !=
        "OFFICIAL_AUTHOR_MAINTAINED_MODEL_SOURCE_SNAPSHOTS_FROZEN"
    ):
        raise RuntimeError(
            "Step-17 source snapshots are not formally frozen."
        )

    if (
        freeze17
        .get("adapter_state", {})
        .get("models_trained")
        is not False
    ):
        raise RuntimeError(
            "Step-17 says models have been trained."
        )

    # -------------------------------------------------------------------------
    # Step 18 source-API freeze.
    # -------------------------------------------------------------------------
    if not STEP18_FREEZE.exists():
        raise RuntimeError(
            "Step-18 source-API freeze missing."
        )

    freeze18 = json.loads(
        STEP18_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    payload18 = dict(freeze18)
    api_fingerprint = payload18.pop(
        "source_api_fingerprint_sha256",
        None,
    )

    if not api_fingerprint:
        raise RuntimeError(
            "Step-18 source-API freeze lacks fingerprint."
        )

    if (
        cfg.canonical_json_sha256(payload18)
        !=
        api_fingerprint
    ):
        raise RuntimeError(
            "Step-18 source-API fingerprint mismatch."
        )

    if (
        step18_manifest.get(
            "source_api_fingerprint_sha256"
        )
        !=
        api_fingerprint
    ):
        raise RuntimeError(
            "Step-18 manifest/API fingerprint mismatch."
        )

    if (
        freeze18.get("status")
        !=
        "FROZEN_SOURCE_APIS_AUDITED_BEFORE_ADAPTER_IMPLEMENTATION"
    ):
        raise RuntimeError(
            "Step-18 source APIs are not formally frozen."
        )

    state18 = freeze18.get(
        "pipeline_state",
        {},
    )

    required_true = (
        "development_interface_frozen",
        "stage1_model_registry_frozen",
        "official_source_snapshots_frozen",
        "source_apis_audited",
    )

    for key in required_true:
        if state18.get(key) is not True:
            raise RuntimeError(
                f"Step-18 prerequisite not frozen: {key}"
            )

    required_false = (
        "task_adapter_contracts_finalized",
        "task_adapters_implemented",
        "task_adapter_smoke_tests_run",
        "models_fitted",
        "feature_selection_applied",
        "probability_threshold_selection_performed",
    )

    for key in required_false:
        if state18.get(key) is not False:
            raise RuntimeError(
                f"Step-18 pipeline-state violation: {key}"
            )

    source_api_records = freeze18.get(
        "source_api_records",
        [],
    )

    if len(source_api_records) != EXPECTED_MODEL_COUNT:
        raise RuntimeError(
            "Unexpected Step-18 source-API record count."
        )

    source_api_by_model = {
        row["model_id"]: row
        for row in source_api_records
    }

    if set(source_api_by_model) != set(
        STAGE1_MODEL_IDS
    ):
        raise RuntimeError(
            "Step-18 source-API records do not cover exactly the eight Stage-1 models."
        )

    # Exact freeze artifacts must still match their producing manifests.
    for path, manifest in (
        (
            STEP16_FREEZE,
            step16_manifest,
        ),
        (
            STEP17_FREEZE,
            step17_manifest,
        ),
        (
            STEP18_FREEZE,
            step18_manifest,
        ),
    ):
        record = find_artifact(
            manifest,
            path.name,
        )

        if (
            Path(record["path"]).resolve()
            !=
            path.resolve()
        ):
            raise RuntimeError(
                f"Canonical freeze path mismatch: {path.name}"
            )

        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(
                f"Frozen artifact SHA mismatch: {path.name}"
            )

    # Step-17 repository inventory.
    repositories = freeze17.get(
        "repositories",
        [],
    )

    if len(repositories) != EXPECTED_FROZEN_REPOSITORY_COUNT:
        raise RuntimeError(
            "Unexpected Step-17 frozen repository count."
        )

    implementation_count = sum(
        1
        for row in repositories
        if row.get("repository_role")
        ==
        "implementation"
    )

    reference_count = sum(
        1
        for row in repositories
        if row.get("repository_role")
        ==
        "reference_redirect"
    )

    if implementation_count != EXPECTED_IMPLEMENTATION_REPOSITORY_COUNT:
        raise RuntimeError(
            "Step-17 implementation-repository count drift."
        )

    if reference_count != EXPECTED_REFERENCE_REPOSITORY_COUNT:
        raise RuntimeError(
            "Step-17 reference-repository count drift."
        )

    # Test seal metadata only.
    if not STEP10_TEST_SEAL.exists():
        raise RuntimeError(
            "Step-10 final-test seal metadata missing."
        )

    seal = json.loads(
        STEP10_TEST_SEAL.read_text(
            encoding="utf-8"
        )
    )

    if seal.get("status") != "SEALED_NOT_OPENED":
        raise RuntimeError(
            "Step-10 final-test seal is not SEALED_NOT_OPENED."
        )

    if (
        seal.get("global_final_test_lock_state")
        !=
        "LOCKED_NOT_OPENED"
    ):
        raise RuntimeError(
            "Step-10 seal does not preserve LOCKED_NOT_OPENED."
        )

    return {
        "manifest_sha": shas,
        "manifests": manifests,
        "environment": environment,
        "step16_freeze": freeze16,
        "step16_registry_fingerprint_sha256": (
            registry_fingerprint
        ),
        "step17_freeze": freeze17,
        "step17_source_fingerprint_sha256": (
            source_fingerprint
        ),
        "step18_freeze": freeze18,
        "step18_api_fingerprint_sha256": (
            api_fingerprint
        ),
        "step18_source_api_by_model": (
            source_api_by_model
        ),
        "test_seal_status": seal["status"],
        "sealed_test_sha256_from_metadata_only": (
            seal.get("sha256")
        ),
    }


# =============================================================================
# 6. IMMUTABILITY / FAILURE HANDLING
# =============================================================================

def canonical_outputs() -> tuple[Path, ...]:
    return (
        MODEL_LOCUS_SUMMARY_CSV,
        INIT_ASSIGNMENT_AUDIT_CSV,
        FORWARD_FLOW_AUDIT_CSV,
        FORWARD_SELF_CALL_AUDIT_CSV,
        RETURN_AUDIT_CSV,
        SHAPE_OPERATION_AUDIT_CSV,
        TASK_DISPATCH_AUDIT_CSV,
        METHOD_SIGNATURE_AUDIT_CSV,
        REPOSITORY_RECHECK_CSV,
        SOURCE_HASH_CSV,
        ADAPTER_LOCUS_FREEZE_JSON,
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
            "Step-19 canonical output(s) already exist. "
            "Refusing to overwrite.\n"
            +
            "\n".join(
                str(path)
                for path in collisions
            )
            +
            "\nUse --verify-only if Step 19 already passed."
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
        FAILED_DIR
        /
        f"partial_artifacts_{stamp}"
    )

    destination.mkdir(
        exist_ok=False
    )

    for path in existing:
        try:
            shutil.move(
                str(path),
                str(
                    destination
                    /
                    path.name
                ),
            )

        except Exception:
            pass

    return destination


# =============================================================================
# 7. FROZEN REPOSITORY RECHECK
# =============================================================================

def recheck_repositories(
    step17_freeze: Dict[str, Any],
) -> tuple[
    Dict[str, Dict[str, Any]],
    pd.DataFrame,
]:
    repositories = step17_freeze[
        "repositories"
    ]

    if len(repositories) != EXPECTED_FROZEN_REPOSITORY_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FROZEN_REPOSITORY_COUNT} frozen repositories; "
            f"found {len(repositories)}."
        )

    repository_map: Dict[
        str,
        Dict[str, Any],
    ] = {}

    rows = []

    for record in repositories:
        repo_id = record["repo_id"]
        path = Path(
            record["local_path"]
        )

        if repo_id in repository_map:
            raise RuntimeError(
                f"Duplicate frozen repository ID: {repo_id}"
            )

        if not path.exists():
            raise RuntimeError(
                f"Frozen repository missing:\n{path}"
            )

        if not (
            path
            /
            ".git"
        ).exists():
            raise RuntimeError(
                f"Frozen source is not a Git repository:\n{path}"
            )

        observed_origin = run_git(
            [
                "remote",
                "get-url",
                "origin",
            ],
            cwd=path,
        )

        observed_commit = run_git(
            [
                "rev-parse",
                "HEAD",
            ],
            cwd=path,
        )

        observed_tree = run_git(
            [
                "rev-parse",
                "HEAD^{tree}",
            ],
            cwd=path,
        )

        observed_branch = run_git(
            [
                "branch",
                "--show-current",
            ],
            cwd=path,
        )

        status = run_git(
            [
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            cwd=path,
        )

        origin_match = (
            normalize_git_url(
                observed_origin
            )
            ==
            normalize_git_url(
                record["repository_url"]
            )
        )

        commit_match = (
            observed_commit
            ==
            record["commit_sha"]
        )

        tree_match = (
            observed_tree
            ==
            record["git_tree_sha"]
        )

        branch_match = (
            observed_branch
            ==
            record["branch"]
        )

        clean = (
            status
            ==
            ""
        )

        files_match = True
        checked_files = 0

        for file_record in record[
            "required_file_hashes"
        ]:
            file_path = (
                path
                /
                file_record[
                    "relative_path"
                ]
            )

            if (
                not file_path.exists()
                or
                not file_path.is_file()
            ):
                files_match = False
                continue

            checked_files += 1

            if (
                sha256_file(
                    file_path
                )
                !=
                file_record[
                    "sha256"
                ]
            ):
                files_match = False

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
            files_match
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
                f"Files      : {files_match}"
            )

        repository_map[
            repo_id
        ] = record

        rows.append(
            {
                "repo_id": repo_id,
                "repository_role": (
                    record[
                        "repository_role"
                    ]
                ),
                "local_path": str(path),
                "origin_match": origin_match,
                "commit_match": commit_match,
                "tree_match": tree_match,
                "branch_match": branch_match,
                "worktree_clean_including_untracked": clean,
                "required_files_checked": int(
                    checked_files
                ),
                "required_file_hashes_match": (
                    files_match
                ),
                "passed": passed,
            }
        )

    implementation_count = sum(
        1
        for row in rows
        if row["repository_role"]
        ==
        "implementation"
    )

    reference_count = sum(
        1
        for row in rows
        if row["repository_role"]
        ==
        "reference_redirect"
    )

    if implementation_count != EXPECTED_IMPLEMENTATION_REPOSITORY_COUNT:
        raise RuntimeError(
            "Frozen implementation-repository count drift."
        )

    if reference_count != EXPECTED_REFERENCE_REPOSITORY_COUNT:
        raise RuntimeError(
            "Frozen reference-repository count drift."
        )

    result = pd.DataFrame(rows)

    if not result[
        "passed"
    ].all():
        raise RuntimeError(
            "One or more frozen repository rechecks failed."
        )

    return (
        repository_map,
        result,
    )


# =============================================================================
# 8. AST HELPERS
# =============================================================================

def ast_text(
    node,
    maximum_length: int = MAX_AST_TEXT_LENGTH,
) -> str:
    if node is None:
        return "<none>"

    try:
        text = ast.unparse(
            node
        )

    except Exception:
        text = (
            f"<{type(node).__name__}>"
        )

    text = (
        text
        .replace(
            "\n",
            " ",
        )
        .strip()
    )

    if len(text) > maximum_length:
        text = (
            text[
                : maximum_length - 3
            ]
            +
            "..."
        )

    return text


def find_top_level_class(
    tree: ast.Module,
    class_name: str,
) -> ast.ClassDef:
    matches = [
        node
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
        and
        node.name
        ==
        class_name
    ]

    if len(matches) != 1:
        classes = [
            node.name
            for node in tree.body
            if isinstance(
                node,
                ast.ClassDef,
            )
        ]

        raise RuntimeError(
            "\nSOURCE CLASS RESOLUTION FAILED\n"
            f"Expected: {class_name}\n"
            f"Found   : {classes}"
        )

    return matches[0]


def find_method(
    class_node: ast.ClassDef,
    method_name: str,
):
    for node in class_node.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            if node.name == method_name:
                return node

    return None


def class_methods(
    class_node: ast.ClassDef,
) -> list:
    return [
        node
        for node in class_node.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
    ]


def argument_text(
    argument: ast.arg,
) -> str:
    text = argument.arg

    if argument.annotation is not None:
        try:
            text += (
                ": "
                +
                ast.unparse(
                    argument.annotation
                )
            )

        except Exception:
            pass

    return text


def function_signature(
    node,
) -> str:
    if node is None:
        return "<missing>"

    positional = (
        list(
            node.args.posonlyargs
        )
        +
        list(
            node.args.args
        )
    )

    defaults = (
        [None]
        *
        (
            len(positional)
            -
            len(
                node.args.defaults
            )
        )
        +
        list(
            node.args.defaults
        )
    )

    pieces = []
    posonly_count = len(
        node.args.posonlyargs
    )

    for index, (
        argument,
        default,
    ) in enumerate(
        zip(
            positional,
            defaults,
        )
    ):
        piece = argument_text(
            argument
        )

        if default is not None:
            piece += (
                "="
                +
                ast_text(
                    default,
                    150,
                )
            )

        pieces.append(
            piece
        )

        if (
            posonly_count
            and
            index
            ==
            posonly_count
            -
            1
        ):
            pieces.append("/")

    if node.args.vararg is not None:
        pieces.append(
            "*"
            +
            argument_text(
                node.args.vararg
            )
        )

    elif node.args.kwonlyargs:
        pieces.append("*")

    for (
        argument,
        default,
    ) in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
    ):
        piece = argument_text(
            argument
        )

        if default is not None:
            piece += (
                "="
                +
                ast_text(
                    default,
                    150,
                )
            )

        pieces.append(
            piece
        )

    if node.args.kwarg is not None:
        pieces.append(
            "**"
            +
            argument_text(
                node.args.kwarg
            )
        )

    return (
        "("
        +
        ", ".join(
            pieces
        )
        +
        ")"
    )


def dotted_name(
    node,
) -> str:
    if isinstance(
        node,
        ast.Name,
    ):
        return node.id

    if isinstance(
        node,
        ast.Attribute,
    ):
        left = dotted_name(
            node.value
        )

        if left:
            return (
                left
                +
                "."
                +
                node.attr
            )

        return node.attr

    return ast_text(
        node,
        200,
    )


def self_attribute_name(
    target,
) -> Optional[str]:
    if isinstance(
        target,
        ast.Attribute,
    ):
        if isinstance(
            target.value,
            ast.Name,
        ):
            if target.value.id == "self":
                return target.attr

    return None


# =============================================================================
# 9. STATIC ARCHITECTURE EXTRACTION
# =============================================================================

def extract_init_assignments(
    model_id: str,
    init_method,
) -> list[dict]:
    if init_method is None:
        return []

    rows = []

    for node in ast.walk(
        init_method
    ):
        target = None
        value = None

        if isinstance(
            node,
            ast.Assign,
        ):
            if len(
                node.targets
            ) == 1:
                target = (
                    node.targets[0]
                )
                value = node.value

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            target = node.target
            value = node.value

        if (
            target is None
            or
            value is None
        ):
            continue

        attribute = self_attribute_name(
            target
        )

        if attribute is None:
            continue

        if isinstance(
            value,
            ast.Call,
        ):
            constructor = dotted_name(
                value.func
            )

        else:
            constructor = "<non-call>"

        lower_attribute = attribute.lower()

        likely_terminal = any(
            token
            in
            lower_attribute
            for token in TERMINAL_ATTRIBUTE_TOKENS
        )

        rows.append(
            {
                "model_id": model_id,
                "line": int(
                    getattr(
                        node,
                        "lineno",
                        -1,
                    )
                ),
                "self_attribute": attribute,
                "constructor_or_source": constructor,
                "assignment_expression": ast_text(
                    value
                ),
                "likely_terminal_head_or_projection_candidate": (
                    likely_terminal
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            row["line"],
            row["self_attribute"],
        ),
    )


def extract_forward_dataflow(
    model_id: str,
    forward_method,
) -> list[dict]:
    rows = []

    for node in ast.walk(
        forward_method
    ):
        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
            ),
        ):
            if isinstance(
                node,
                ast.Assign,
            ):
                targets = (
                    ";".join(
                        ast_text(
                            target,
                            180,
                        )
                        for target
                        in node.targets
                    )
                )
                value = node.value

            elif isinstance(
                node,
                ast.AnnAssign,
            ):
                targets = ast_text(
                    node.target,
                    180,
                )
                value = node.value

            else:
                targets = ast_text(
                    node.target,
                    180,
                )
                value = node.value

            rows.append(
                {
                    "model_id": model_id,
                    "line": int(
                        getattr(
                            node,
                            "lineno",
                            -1,
                        )
                    ),
                    "targets": targets,
                    "expression": ast_text(
                        value
                    ),
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            row["line"],
            row["targets"],
        ),
    )


def extract_forward_self_calls(
    model_id: str,
    forward_method,
) -> list[dict]:
    rows = []

    for node in ast.walk(
        forward_method
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        call_name = dotted_name(
            node.func
        )

        if not call_name.startswith(
            "self."
        ):
            continue

        rows.append(
            {
                "model_id": model_id,
                "line": int(
                    getattr(
                        node,
                        "lineno",
                        -1,
                    )
                ),
                "self_call": call_name,
                "call_expression": ast_text(
                    node
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            row["line"],
            row["self_call"],
        ),
    )


def extract_returns(
    model_id: str,
    forward_method,
) -> list[dict]:
    rows = []

    for node in ast.walk(
        forward_method
    ):
        if isinstance(
            node,
            ast.Return,
        ):
            rows.append(
                {
                    "model_id": model_id,
                    "line": int(
                        getattr(
                            node,
                            "lineno",
                            -1,
                        )
                    ),
                    "return_expression": ast_text(
                        node.value
                    ),
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            row["line"],
        ),
    )


def extract_shape_operations(
    model_id: str,
    methods: list,
) -> list[dict]:
    rows = []

    for method in methods:
        for node in ast.walk(
            method
        ):
            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            operation = None

            if isinstance(
                node.func,
                ast.Attribute,
            ):
                if (
                    node.func.attr
                    in
                    SHAPE_OPERATION_NAMES
                ):
                    operation = (
                        node.func.attr
                    )

            elif isinstance(
                node.func,
                ast.Name,
            ):
                if (
                    node.func.id
                    in
                    SHAPE_OPERATION_NAMES
                ):
                    operation = (
                        node.func.id
                    )

            if operation is None:
                continue

            rows.append(
                {
                    "model_id": model_id,
                    "method": method.name,
                    "line": int(
                        getattr(
                            node,
                            "lineno",
                            -1,
                        )
                    ),
                    "operation": operation,
                    "expression": ast_text(
                        node
                    ),
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            row["method"],
            row["line"],
            row["operation"],
        ),
    )


def extract_task_dispatch(
    model_id: str,
    methods: list,
) -> list[dict]:
    rows = []

    for method in methods:
        for node in ast.walk(
            method
        ):
            if not isinstance(
                node,
                (
                    ast.If,
                    ast.IfExp,
                ),
            ):
                continue

            text = ast_text(
                node.test
            )

            lower = text.lower()

            if not any(
                token
                in
                lower
                for token
                in TASK_SIGNAL_TOKENS
            ):
                continue

            rows.append(
                {
                    "model_id": model_id,
                    "method": method.name,
                    "line": int(
                        getattr(
                            node,
                            "lineno",
                            -1,
                        )
                    ),
                    "condition": text,
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            row["method"],
            row["line"],
        ),
    )


def extract_method_signatures(
    model_id: str,
    methods: list,
) -> list[dict]:
    return sorted(
        [
            {
                "model_id": model_id,
                "method": method.name,
                "line": int(
                    getattr(
                        method,
                        "lineno",
                        -1,
                    )
                ),
                "signature": function_signature(
                    method
                ),
            }
            for method in methods
        ],
        key=lambda row: (
            row["line"],
            row["method"],
        ),
    )


def classify_adapter_readiness(
    method_names: list[str],
    task_dispatch_rows: list[dict],
    terminal_candidates: list[str],
) -> tuple[str, str]:
    method_names_lower = {
        name.lower()
        for name in method_names
    }

    if (
        "classification"
        in
        method_names_lower
    ):
        return (
            "native_classification_path_available",
            (
                "Inspect the frozen native classification representation/head "
                "first. Step 20 may adapt only the terminal task output to three "
                "independent directional logits while preserving the frozen backbone."
            ),
        )

    classification_dispatch = any(
        "classification"
        in row["condition"].lower()
        for row in task_dispatch_rows
    )

    if classification_dispatch:
        return (
            "native_classification_dispatch_available",
            (
                "A frozen classification dispatch condition exists. Step 20 must "
                "trace that exact pathway before deciding whether to reuse its "
                "representation/head or replace only the terminal output."
            ),
        )

    if (
        "forecast"
        in
        method_names_lower
        or
        any(
            "forecast"
            in row["condition"].lower()
            for row in task_dispatch_rows
        )
    ):
        return (
            "forecast_path_available",
            (
                "Preserve the frozen temporal/frequency backbone and identify the "
                "exact terminal forecasting projection. Step 20 may replace or "
                "bypass only the verified terminal projection."
            ),
        )

    if terminal_candidates:
        return (
            "generic_forward_with_terminal_candidates",
            (
                "Generic forward pathway with heuristic terminal candidates. "
                "Step 20 must confirm the exact tensor locus from the frozen "
                "data-flow evidence before implementing any replacement head."
            ),
        )

    return (
        "generic_forward_requires_manual_locus_confirmation",
        (
            "No native classification/forecast method and no terminal candidate "
            "was established automatically. Step 20 must not implement this "
            "adapter until the recorded forward/return evidence is reviewed."
        ),
    )


# =============================================================================
# 10. INSPECT ONE MODEL
# =============================================================================

def frozen_required_file_hash(
    repository_record: Dict[str, Any],
    relative_path: str,
) -> str:
    matches = [
        row
        for row in repository_record[
            "required_file_hashes"
        ]
        if row[
            "relative_path"
        ]
        ==
        relative_path
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Step-17 frozen file record for "
            f"{repository_record['repo_id']}::{relative_path}; "
            f"found {len(matches)}."
        )

    return str(
        matches[0][
            "sha256"
        ]
    )


def inspect_model(
    api_record: Dict[str, Any],
    repository_record: Dict[str, Any],
) -> tuple[
    dict,
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
]:
    model_id = api_record[
        "model_id"
    ]

    source_file = api_record[
        "primary_source_file"
    ]

    preferred_class = api_record[
        "preferred_class"
    ]

    repository_path = Path(
        repository_record[
            "local_path"
        ]
    )

    path = (
        repository_path
        /
        source_file
    )

    if not path.exists():
        raise RuntimeError(
            f"Primary source missing:\n{path}"
        )

    observed_source_hash = sha256_file(
        path
    )

    expected_step18_hash = api_record[
        "source_sha256"
    ]

    expected_step17_hash = (
        frozen_required_file_hash(
            repository_record,
            source_file,
        )
    )

    if observed_source_hash != expected_step18_hash:
        raise RuntimeError(
            "\nSTEP-18 PRIMARY SOURCE HASH MISMATCH\n"
            f"Model    : {model_id}\n"
            f"Expected : {expected_step18_hash}\n"
            f"Observed : {observed_source_hash}"
        )

    if observed_source_hash != expected_step17_hash:
        raise RuntimeError(
            "\nSTEP-17 PRIMARY SOURCE HASH MISMATCH\n"
            f"Model    : {model_id}\n"
            f"Expected : {expected_step17_hash}\n"
            f"Observed : {observed_source_hash}"
        )

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    try:
        tree = ast.parse(
            text
        )

    except SyntaxError as exc:
        raise RuntimeError(
            f"AST parse failed for {path}: {exc}"
        ) from exc

    class_node = find_top_level_class(
        tree=tree,
        class_name=preferred_class,
    )

    init_method = find_method(
        class_node,
        "__init__",
    )

    forward_method = find_method(
        class_node,
        "forward",
    )

    if forward_method is None:
        raise RuntimeError(
            f"{model_id}: preferred class has no forward() method."
        )

    observed_constructor_signature = (
        function_signature(
            init_method
        )
    )

    observed_forward_signature = (
        function_signature(
            forward_method
        )
    )

    if (
        observed_constructor_signature
        !=
        api_record[
            "constructor_signature"
        ]
    ):
        raise RuntimeError(
            "\nSTEP-18 CONSTRUCTOR SIGNATURE DRIFT\n"
            f"Model    : {model_id}\n"
            f"Frozen   : {api_record['constructor_signature']}\n"
            f"Observed : {observed_constructor_signature}"
        )

    if (
        observed_forward_signature
        !=
        api_record[
            "forward_signature"
        ]
    ):
        raise RuntimeError(
            "\nSTEP-18 FORWARD SIGNATURE DRIFT\n"
            f"Model    : {model_id}\n"
            f"Frozen   : {api_record['forward_signature']}\n"
            f"Observed : {observed_forward_signature}"
        )

    methods = class_methods(
        class_node
    )

    method_names = [
        method.name
        for method in methods
    ]

    init_assignments = (
        extract_init_assignments(
            model_id,
            init_method,
        )
    )

    forward_flow = (
        extract_forward_dataflow(
            model_id,
            forward_method,
        )
    )

    forward_self_calls = (
        extract_forward_self_calls(
            model_id,
            forward_method,
        )
    )

    returns = extract_returns(
        model_id,
        forward_method,
    )

    if not returns:
        raise RuntimeError(
            f"{model_id}: no return expression found inside forward()."
        )

    shape_operations = (
        extract_shape_operations(
            model_id,
            methods,
        )
    )

    task_dispatch = (
        extract_task_dispatch(
            model_id,
            methods,
        )
    )

    method_signatures = (
        extract_method_signatures(
            model_id,
            methods,
        )
    )

    terminal_candidates = sorted(
        {
            row[
                "self_attribute"
            ]
            for row
            in init_assignments
            if bool(
                row[
                    "likely_terminal_head_or_projection_candidate"
                ]
            )
        }
    )

    forward_self_call_names = sorted(
        {
            row[
                "self_call"
            ]
            for row
            in forward_self_calls
        }
    )

    (
        readiness,
        recommendation,
    ) = classify_adapter_readiness(
        method_names=method_names,
        task_dispatch_rows=task_dispatch,
        terminal_candidates=terminal_candidates,
    )

    step18_config_references = sorted(
        str(value)
        for value in api_record.get(
            "config_references",
            []
        )
    )

    summary = {
        "model_id": model_id,
        "repo_id": api_record[
            "repo_id"
        ],
        "reference_id": api_record[
            "reference_id"
        ],
        "repository_commit_sha": (
            repository_record[
                "commit_sha"
            ]
        ),
        "repository_git_tree_sha": (
            repository_record[
                "git_tree_sha"
            ]
        ),
        "primary_source_file": source_file,
        "source_sha256": (
            observed_source_hash
        ),
        "source_hash_matches_step17_and_step18": True,
        "preferred_class": preferred_class,
        "constructor_signature": (
            observed_constructor_signature
        ),
        "forward_signature": (
            observed_forward_signature
        ),
        "signatures_match_step18_freeze": True,
        "class_methods": (
            ";".join(
                method_names
            )
        ),
        "step18_config_reference_count": int(
            len(
                step18_config_references
            )
        ),
        "step18_config_references": (
            ";".join(
                step18_config_references
            )
        ),
        "init_self_assignment_count": int(
            len(
                init_assignments
            )
        ),
        "likely_terminal_attribute_candidates": (
            ";".join(
                terminal_candidates
            )
        ),
        "forward_self_calls": (
            ";".join(
                forward_self_call_names
            )
        ),
        "forward_assignment_count": int(
            len(
                forward_flow
            )
        ),
        "return_expression_count": int(
            len(
                returns
            )
        ),
        "shape_operation_count": int(
            len(
                shape_operations
            )
        ),
        "task_dispatch_condition_count": int(
            len(
                task_dispatch
            )
        ),
        "adapter_readiness": readiness,
        "recommended_next_adapter_action": (
            recommendation
        ),
        "terminal_candidates_are_heuristic_only": True,
        "third_party_source_imported": False,
        "third_party_source_executed": False,
    }

    return (
        summary,
        init_assignments,
        forward_flow,
        forward_self_calls,
        returns,
        shape_operations,
        task_dispatch,
        method_signatures,
    )


# =============================================================================
# 11. SOURCE HASH REGISTER
# =============================================================================

def source_hashes() -> pd.DataFrame:
    sources = [
        (
            "current_step00",
            Path(cfg.__file__).resolve(),
        ),
        (
            "current_step19",
            SCRIPT_FILE.resolve(),
        ),
        (
            "step16_registry_freeze",
            STEP16_FREEZE.resolve(),
        ),
        (
            "step17_source_freeze",
            STEP17_FREEZE.resolve(),
        ),
        (
            "step18_source_api_freeze",
            STEP18_FREEZE.resolve(),
        ),
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
                "Final-test parquet entered Step-19 source list."
            )

        if not path.exists():
            raise RuntimeError(
                f"Required Step-19 source missing: {path}"
            )

        rows.append(
            {
                "role": role,
                "file_name": path.name,
                "absolute_path": str(path),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(
                    path
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 12. VERIFY EXISTING STEP 19
# =============================================================================

def verify_existing() -> int:
    print("=" * 136)
    print("STEP 19 — VERIFY EXISTING ADAPTATION-LOCUS FREEZE")
    print("=" * 136)

    step00 = verify_step00()
    upstream = verify_upstream(
        step00
    )

    manifest_sha = verify_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 19",
    )

    manifest = json.loads(
        MANIFEST_JSON.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest
        .get("step", {})
        .get("status")
        !=
        "PASS"
    ):
        print(
            "[FAIL] Stored Step-19 status is not PASS."
        )
        return 20

    if (
        manifest.get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00["fingerprint"]
    ):
        print(
            "[FAIL] Configuration fingerprint mismatch."
        )
        return 21

    for step in range(1, 19):
        field = (
            f"step{step:02d}_manifest_sha256"
        )

        if (
            manifest.get(field)
            !=
            upstream["manifest_sha"][step]
        ):
            print(
                f"[FAIL] Upstream chain mismatch: {field}"
            )
            return 22

    if (
        manifest.get(
            "step16_registry_freeze_fingerprint_sha256"
        )
        !=
        upstream[
            "step16_registry_fingerprint_sha256"
        ]
    ):
        print(
            "[FAIL] Step-16 registry fingerprint mismatch."
        )
        return 23

    if (
        manifest.get(
            "step17_source_freeze_fingerprint_sha256"
        )
        !=
        upstream[
            "step17_source_fingerprint_sha256"
        ]
    ):
        print(
            "[FAIL] Step-17 source fingerprint mismatch."
        )
        return 24

    if (
        manifest.get(
            "step18_source_api_fingerprint_sha256"
        )
        !=
        upstream[
            "step18_api_fingerprint_sha256"
        ]
    ):
        print(
            "[FAIL] Step-18 source-API fingerprint mismatch."
        )
        return 25

    failures = []

    for name, record in manifest.get(
        "artifacts",
        {},
    ).items():
        path = Path(
            record["path"]
        )

        if not path.exists():
            failures.append(
                f"missing: {name}"
            )
            continue

        if sha256_file(path) != record["sha256"]:
            failures.append(
                f"hash mismatch: {name}"
            )

    if failures:
        for failure in failures:
            print(
                f"[FAIL] {failure}"
            )

        print("=" * 136)
        print("STEP 19 VERIFICATION STATUS: FAIL")
        print("=" * 136)
        return 26

    freeze = json.loads(
        ADAPTER_LOCUS_FREEZE_JSON.read_text(
            encoding="utf-8"
        )
    )

    stored_fingerprint = freeze.get(
        "adapter_locus_fingerprint_sha256"
    )

    payload = dict(freeze)
    payload.pop(
        "adapter_locus_fingerprint_sha256",
        None,
    )

    recomputed = cfg.canonical_json_sha256(
        payload
    )

    if stored_fingerprint != recomputed:
        print(
            "[FAIL] Adapter-locus fingerprint mismatch."
        )
        return 27

    if (
        manifest.get(
            "adapter_locus_fingerprint_sha256"
        )
        !=
        stored_fingerprint
    ):
        print(
            "[FAIL] Manifest/adapter-locus fingerprint mismatch."
        )
        return 28

    repository_map, repository_recheck = (
        recheck_repositories(
            upstream[
                "step17_freeze"
            ]
        )
    )

    if not repository_recheck[
        "passed"
    ].all():
        print(
            "[FAIL] Frozen repository recheck failed."
        )
        return 29

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
        return 30

    print(
        "[PASS] Configuration fingerprint : "
        f"{step00['fingerprint']}"
    )
    print(
        "[PASS] Step-18 manifest          : "
        f"{upstream['manifest_sha'][18]}"
    )
    print(
        "[PASS] Step-16 registry freeze   : "
        f"{upstream['step16_registry_fingerprint_sha256']}"
    )
    print(
        "[PASS] Step-17 source freeze     : "
        f"{upstream['step17_source_fingerprint_sha256']}"
    )
    print(
        "[PASS] Step-18 source API freeze : "
        f"{upstream['step18_api_fingerprint_sha256']}"
    )
    print(
        "[PASS] Frozen repositories       : "
        f"{len(repository_map)}"
    )
    print(
        "[PASS] Model loci audited        : "
        f"{len(freeze['model_records'])}"
    )
    print(
        "[PASS] Step-19 locus freeze      : "
        f"{stored_fingerprint}"
    )
    print(
        "[PASS] Step-19 manifest          : "
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
    print("STEP 19 VERIFICATION STATUS: PASS")
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
            "[1/9] Verifying Step 00 and immutable Step 01 -> Step 18 chain..."
        )

        step00 = verify_step00()
        upstream = verify_upstream(
            step00
        )

        print(
            "       Configuration fingerprint: "
            f"{step00['fingerprint']}"
        )
        print(
            "       Step-18 manifest          : "
            f"{upstream['manifest_sha'][18]}"
        )
        print(
            "       Step-16 registry freeze   : "
            f"{upstream['step16_registry_fingerprint_sha256']}"
        )
        print(
            "       Step-17 source freeze     : "
            f"{upstream['step17_source_fingerprint_sha256']}"
        )
        print(
            "       Step-18 source API freeze : "
            f"{upstream['step18_api_fingerprint_sha256']}"
        )
        print(
            "       Stage-1 architectures     : "
            f"{EXPECTED_MODEL_COUNT}"
        )
        print(
            "       Frozen repositories       : "
            f"{EXPECTED_FROZEN_REPOSITORY_COUNT}"
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
            "       Model training            : PROHIBITED"
        )
        print(
            "       Final-test parquet access : PROHIBITED"
        )

        stage = "collision_guard"

        print(
            "[2/9] Checking immutable Step-19 output target..."
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
            upstream[
                "step17_freeze"
            ]
        )

        print(
            "       Repositories rechecked      : "
            f"{len(repository_recheck)}"
        )
        print(
            "       Origins/commits/trees match : "
            f"{bool(repository_recheck['passed'].all())}"
        )
        print(
            "       Branches match              : "
            f"{bool(repository_recheck['branch_match'].all())}"
        )
        print(
            "       Worktrees clean incl. untracked: "
            f"{bool(repository_recheck['worktree_clean_including_untracked'].all())}"
        )
        print(
            "       Required-file hashes match  : "
            f"{bool(repository_recheck['required_file_hashes_match'].all())}"
        )

        stage = "locus_inspection"

        print(
            "[4/9] Auditing frozen architecture adaptation loci..."
        )

        source_api_by_model = upstream[
            "step18_source_api_by_model"
        ]

        summary_rows = []
        init_rows = []
        forward_flow_rows = []
        forward_self_call_rows = []
        return_rows = []
        shape_rows = []
        task_dispatch_rows = []
        method_signature_rows = []

        for model_id in STAGE1_MODEL_IDS:
            api_record = source_api_by_model[
                model_id
            ]

            repo_id = api_record[
                "repo_id"
            ]

            if repo_id not in repository_map:
                raise RuntimeError(
                    f"Frozen repository missing for {repo_id}."
                )

            (
                summary,
                model_init_rows,
                model_forward_flow_rows,
                model_forward_self_call_rows,
                model_return_rows,
                model_shape_rows,
                model_task_dispatch_rows,
                model_method_signature_rows,
            ) = inspect_model(
                api_record=api_record,
                repository_record=(
                    repository_map[
                        repo_id
                    ]
                ),
            )

            summary_rows.append(summary)
            init_rows.extend(model_init_rows)
            forward_flow_rows.extend(
                model_forward_flow_rows
            )
            forward_self_call_rows.extend(
                model_forward_self_call_rows
            )
            return_rows.extend(
                model_return_rows
            )
            shape_rows.extend(
                model_shape_rows
            )
            task_dispatch_rows.extend(
                model_task_dispatch_rows
            )
            method_signature_rows.extend(
                model_method_signature_rows
            )

            terminal = (
                summary[
                    "likely_terminal_attribute_candidates"
                ]
                or
                "<none>"
            )

            print(
                f"       {model_id:<14} | "
                f"readiness={summary['adapter_readiness']:<44} | "
                f"terminal={terminal}"
            )
            print(
                f"           forward {summary['forward_signature']}"
            )

        summary_df = pd.DataFrame(
            summary_rows
        )

        init_df = pd.DataFrame(
            init_rows,
            columns=[
                "model_id",
                "line",
                "self_attribute",
                "constructor_or_source",
                "assignment_expression",
                "likely_terminal_head_or_projection_candidate",
            ],
        )

        forward_flow_df = pd.DataFrame(
            forward_flow_rows,
            columns=[
                "model_id",
                "line",
                "targets",
                "expression",
            ],
        )

        forward_self_call_df = pd.DataFrame(
            forward_self_call_rows,
            columns=[
                "model_id",
                "line",
                "self_call",
                "call_expression",
            ],
        )

        return_df = pd.DataFrame(
            return_rows,
            columns=[
                "model_id",
                "line",
                "return_expression",
            ],
        )

        shape_df = pd.DataFrame(
            shape_rows,
            columns=[
                "model_id",
                "method",
                "line",
                "operation",
                "expression",
            ],
        )

        task_dispatch_df = pd.DataFrame(
            task_dispatch_rows,
            columns=[
                "model_id",
                "method",
                "line",
                "condition",
            ],
        )

        method_signature_df = pd.DataFrame(
            method_signature_rows,
            columns=[
                "model_id",
                "method",
                "line",
                "signature",
            ],
        )

        stage = "structural_validation"

        print(
            "[5/9] Validating static adaptation-locus evidence..."
        )

        if len(summary_df) != EXPECTED_MODEL_COUNT:
            raise RuntimeError(
                "Model locus summary count mismatch."
            )

        if tuple(
            summary_df[
                "model_id"
            ].astype(str)
        ) != STAGE1_MODEL_IDS:
            raise RuntimeError(
                "Model locus summary order differs from frozen Stage-1 order."
            )

        if summary_df[
            "source_sha256"
        ].isna().any():
            raise RuntimeError(
                "Missing primary-source SHA-256."
            )

        if not summary_df[
            "source_hash_matches_step17_and_step18"
        ].all():
            raise RuntimeError(
                "One or more primary sources fail Step-17/18 hash reconciliation."
            )

        if not summary_df[
            "signatures_match_step18_freeze"
        ].all():
            raise RuntimeError(
                "One or more API signatures differ from Step 18."
            )

        if (
            summary_df[
                "return_expression_count"
            ]
            <=
            0
        ).any():
            bad = (
                summary_df.loc[
                    summary_df[
                        "return_expression_count"
                    ]
                    <=
                    0,
                    "model_id",
                ]
                .tolist()
            )

            raise RuntimeError(
                f"No forward return expression found for: {bad}"
            )

        if not repository_recheck[
            "passed"
        ].all():
            raise RuntimeError(
                "Frozen repository recheck failed."
            )

        print(
            "       Models audited                 : "
            f"{len(summary_df)}"
        )
        print(
            "       Step-17/18 source hashes exact : "
            f"{bool(summary_df['source_hash_matches_step17_and_step18'].all())}"
        )
        print(
            "       API signatures equal Step 18   : "
            f"{bool(summary_df['signatures_match_step18_freeze'].all())}"
        )
        print(
            "       forward() returns present       : "
            f"{bool((summary_df['return_expression_count'] > 0).all())}"
        )
        print(
            "       Terminal candidates are heuristic only: True"
        )
        print(
            "       Third-party source imported    : False"
        )
        print(
            "       Third-party source executed    : False"
        )

        stage = "write"

        print(
            "[6/9] Writing adaptation-locus audits and provenance..."
        )

        write_csv(
            MODEL_LOCUS_SUMMARY_CSV,
            summary_df,
        )

        write_csv(
            INIT_ASSIGNMENT_AUDIT_CSV,
            init_df,
        )

        write_csv(
            FORWARD_FLOW_AUDIT_CSV,
            forward_flow_df,
        )

        write_csv(
            FORWARD_SELF_CALL_AUDIT_CSV,
            forward_self_call_df,
        )

        write_csv(
            RETURN_AUDIT_CSV,
            return_df,
        )

        write_csv(
            SHAPE_OPERATION_AUDIT_CSV,
            shape_df,
        )

        write_csv(
            TASK_DISPATCH_AUDIT_CSV,
            task_dispatch_df,
        )

        write_csv(
            METHOD_SIGNATURE_AUDIT_CSV,
            method_signature_df,
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
            "[7/9] Freezing adaptation-locus evidence before adapter implementation..."
        )

        if (
            sha256_file(
                Path(
                    step00["lock_path"]
                )
            )
            !=
            step00["lock_sha256"]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step 19."
            )

        model_records = []

        for model_id in STAGE1_MODEL_IDS:
            row = (
                summary_df.loc[
                    summary_df[
                        "model_id"
                    ]
                    ==
                    model_id
                ]
                .iloc[0]
            )

            model_init = (
                init_df.loc[
                    init_df[
                        "model_id"
                    ]
                    ==
                    model_id
                ]
                .to_dict(
                    orient="records"
                )
            )

            model_forward = (
                forward_flow_df.loc[
                    forward_flow_df[
                        "model_id"
                    ]
                    ==
                    model_id
                ]
                .to_dict(
                    orient="records"
                )
            )

            model_calls = (
                forward_self_call_df.loc[
                    forward_self_call_df[
                        "model_id"
                    ]
                    ==
                    model_id
                ]
                .to_dict(
                    orient="records"
                )
            )

            model_returns = (
                return_df.loc[
                    return_df[
                        "model_id"
                    ]
                    ==
                    model_id
                ]
                .to_dict(
                    orient="records"
                )
            )

            model_shapes = (
                shape_df.loc[
                    shape_df[
                        "model_id"
                    ]
                    ==
                    model_id
                ]
                .to_dict(
                    orient="records"
                )
            )

            model_dispatch = (
                task_dispatch_df.loc[
                    task_dispatch_df[
                        "model_id"
                    ]
                    ==
                    model_id
                ]
                .to_dict(
                    orient="records"
                )
            )

            model_methods = (
                method_signature_df.loc[
                    method_signature_df[
                        "model_id"
                    ]
                    ==
                    model_id
                ]
                .to_dict(
                    orient="records"
                )
            )

            model_records.append(
                {
                    "model_id": model_id,
                    "repo_id": row[
                        "repo_id"
                    ],
                    "reference_id": row[
                        "reference_id"
                    ],
                    "repository_commit_sha": row[
                        "repository_commit_sha"
                    ],
                    "repository_git_tree_sha": row[
                        "repository_git_tree_sha"
                    ],
                    "primary_source_file": row[
                        "primary_source_file"
                    ],
                    "source_sha256": row[
                        "source_sha256"
                    ],
                    "preferred_class": row[
                        "preferred_class"
                    ],
                    "constructor_signature": row[
                        "constructor_signature"
                    ],
                    "forward_signature": row[
                        "forward_signature"
                    ],
                    "step18_config_references": (
                        [
                            value
                            for value in str(
                                row[
                                    "step18_config_references"
                                ]
                            ).split(";")
                            if value
                        ]
                    ),
                    "likely_terminal_attribute_candidates": (
                        [
                            value
                            for value in str(
                                row[
                                    "likely_terminal_attribute_candidates"
                                ]
                            ).split(";")
                            if value
                        ]
                    ),
                    "terminal_candidates_are_heuristic_only": True,
                    "forward_self_calls": (
                        [
                            value
                            for value in str(
                                row[
                                    "forward_self_calls"
                                ]
                            ).split(";")
                            if value
                        ]
                    ),
                    "adapter_readiness": row[
                        "adapter_readiness"
                    ],
                    "recommended_next_adapter_action": row[
                        "recommended_next_adapter_action"
                    ],
                    "init_self_assignments": model_init,
                    "forward_dataflow": model_forward,
                    "forward_self_calls_detail": (
                        model_calls
                    ),
                    "forward_returns": model_returns,
                    "shape_operations": model_shapes,
                    "task_dispatch_conditions": (
                        model_dispatch
                    ),
                    "method_signatures": model_methods,
                }
            )

        freeze_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "ADAPTER_LOCI_AUDITED_BEFORE_IMPLEMENTATION"
            ),
            "generated_at_utc": utc_now(),
            "configuration_fingerprint_sha256": (
                step00["fingerprint"]
            ),
            "step16_manifest_sha256": (
                upstream[
                    "manifest_sha"
                ][16]
            ),
            "step17_manifest_sha256": (
                upstream[
                    "manifest_sha"
                ][17]
            ),
            "step18_manifest_sha256": (
                upstream[
                    "manifest_sha"
                ][18]
            ),
            "step16_registry_freeze_fingerprint_sha256": (
                upstream[
                    "step16_registry_fingerprint_sha256"
                ]
            ),
            "step17_source_freeze_fingerprint_sha256": (
                upstream[
                    "step17_source_fingerprint_sha256"
                ]
            ),
            "step18_source_api_fingerprint_sha256": (
                upstream[
                    "step18_api_fingerprint_sha256"
                ]
            ),
            "model_records": model_records,
            "adapter_contract_constraints": {
                "dataset_input": (
                    "[batch,60,63]"
                ),
                "directional_output": (
                    "[batch,3] raw logits"
                ),
                "directional_horizons": [
                    1,
                    3,
                    5,
                ],
                "directional_loss": (
                    "BCEWithLogitsLoss(reduction='mean')"
                ),
                "published_backbone_preservation_required": True,
                "published_terminal_task_head_may_be_replaced_only_after_locus_confirmation": True,
                "published_backbone_may_not_be_silently_rewritten": True,
                "stage1_comparator_stock_embedding_allowed": False,
                "stage1_comparator_regime_conditioning_allowed": False,
                "stage1_comparator_auxiliary_regression_allowed": False,
                "output_probability": (
                    "sigmoid(raw logits)"
                ),
                "reported_description": (
                    "architecture-adapted directional classification baseline"
                ),
                "terminal_attribute_candidates_are_heuristic_only": True,
                "adapter_implementation_frozen": False,
                "executable_adapter_contract_finalized": False,
            },
            "pipeline_state": {
                "development_interface_frozen": True,
                "stage1_model_registry_frozen": True,
                "official_source_snapshots_frozen": True,
                "source_apis_audited": True,
                "adapter_loci_audited": True,
                "adapter_contracts_finalized": False,
                "task_adapters_implemented": False,
                "smoke_tests_run": False,
                "models_fitted": False,
                "feature_selection_applied": False,
                "probability_threshold_selection_performed": False,
            },
            "final_test_protection": {
                "step10_seal_status": (
                    upstream[
                        "test_seal_status"
                    ]
                ),
                "sealed_test_sha256_from_metadata_only": (
                    upstream[
                        "sealed_test_sha256_from_metadata_only"
                    ]
                ),
                "final_test_parquet_opened_or_hashed_by_step19": False,
                "final_test_performance_computed": False,
                "global_final_test_state": (
                    step00["state"]
                ),
            },
        }

        locus_fingerprint = (
            cfg.canonical_json_sha256(
                freeze_payload
            )
        )

        freeze_payload[
            "adapter_locus_fingerprint_sha256"
        ] = locus_fingerprint

        atomic_json(
            ADAPTER_LOCUS_FREEZE_JSON,
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
                        "Step-16 registry fingerprint: "
                        f"{upstream['step16_registry_fingerprint_sha256']}"
                    ),
                    (
                        "Step-17 source fingerprint: "
                        f"{upstream['step17_source_fingerprint_sha256']}"
                    ),
                    (
                        "Step-18 source-API fingerprint: "
                        f"{upstream['step18_api_fingerprint_sha256']}"
                    ),
                    (
                        "Frozen repositories rechecked: "
                        f"{len(repository_recheck)}"
                    ),
                    (
                        "Model adaptation loci audited: "
                        f"{len(summary_df)}"
                    ),
                    (
                        "__init__ self assignments: "
                        f"{len(init_df)}"
                    ),
                    (
                        "forward data-flow assignments: "
                        f"{len(forward_flow_df)}"
                    ),
                    (
                        "forward self calls: "
                        f"{len(forward_self_call_df)}"
                    ),
                    (
                        "forward returns: "
                        f"{len(return_df)}"
                    ),
                    (
                        "shape operations: "
                        f"{len(shape_df)}"
                    ),
                    (
                        "task-dispatch conditions: "
                        f"{len(task_dispatch_df)}"
                    ),
                    "Terminal candidates heuristic only: True",
                    "Third-party source imported: False",
                    "Third-party source executed: False",
                    "Executable adapter contracts finalized: False",
                    "Task adapters implemented: False",
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
            MODEL_LOCUS_SUMMARY_CSV,
            INIT_ASSIGNMENT_AUDIT_CSV,
            FORWARD_FLOW_AUDIT_CSV,
            FORWARD_SELF_CALL_AUDIT_CSV,
            RETURN_AUDIT_CSV,
            SHAPE_OPERATION_AUDIT_CSV,
            TASK_DISPATCH_AUDIT_CSV,
            METHOD_SIGNATURE_AUDIT_CSV,
            REPOSITORY_RECHECK_CSV,
            SOURCE_HASH_CSV,
            ADAPTER_LOCUS_FREEZE_JSON,
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
                            relative_to_experiment(
                                path
                            )
                        ),
                        "size_bytes": int(
                            path.stat().st_size
                        ),
                        "sha256": sha256_file(
                            path
                        ),
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
            "experiment_tag": (
                cfg.EXPERIMENT_TAG
            ),
            "configuration_fingerprint_sha256": (
                step00["fingerprint"]
            ),
            "step00_manifest_sha256": (
                step00[
                    "step00_manifest_sha256"
                ]
            ),
            **{
                f"step{step:02d}_manifest_sha256": (
                    upstream[
                        "manifest_sha"
                    ][step]
                )
                for step in range(1, 19)
            },
            "step16_registry_freeze_fingerprint_sha256": (
                upstream[
                    "step16_registry_fingerprint_sha256"
                ]
            ),
            "step17_source_freeze_fingerprint_sha256": (
                upstream[
                    "step17_source_fingerprint_sha256"
                ]
            ),
            "step18_source_api_fingerprint_sha256": (
                upstream[
                    "step18_api_fingerprint_sha256"
                ]
            ),
            "adapter_locus_fingerprint_sha256": (
                locus_fingerprint
            ),
            "counts": {
                "repositories_rechecked": int(
                    len(
                        repository_recheck
                    )
                ),
                "models_audited": int(
                    len(
                        summary_df
                    )
                ),
                "init_assignments": int(
                    len(
                        init_df
                    )
                ),
                "forward_assignments": int(
                    len(
                        forward_flow_df
                    )
                ),
                "forward_self_calls": int(
                    len(
                        forward_self_call_df
                    )
                ),
                "return_expressions": int(
                    len(
                        return_df
                    )
                ),
                "shape_operations": int(
                    len(
                        shape_df
                    )
                ),
                "task_dispatch_conditions": int(
                    len(
                        task_dispatch_df
                    )
                ),
                "method_signatures": int(
                    len(
                        method_signature_df
                    )
                ),
            },
            "pipeline_state": {
                "development_interface_frozen": True,
                "stage1_model_registry_frozen": True,
                "official_source_snapshots_frozen": True,
                "source_apis_audited": True,
                "adapter_loci_audited": True,
                "adapter_contracts_frozen": False,
                "task_adapters_implemented": False,
                "smoke_tests_run": False,
                "models_fitted": False,
                "feature_selection_applied": False,
                "probability_threshold_selection_performed": False,
            },
            "test_protection": {
                "step10_seal_status": (
                    upstream[
                        "test_seal_status"
                    ]
                ),
                "global_final_test_state": (
                    step00["state"]
                ),
                "sealed_test_parquet_opened_or_hashed_by_step19": False,
                "final_test_performance_computed": False,
            },
            "artifacts": {
                path.name: artifact_record(
                    path
                )
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
            "[8/9] Validating frozen Step-19 outputs..."
        )

        verify_sidecar(
            MANIFEST_JSON,
            MANIFEST_SHA,
            "Step 19",
        )

        stored_manifest = json.loads(
            MANIFEST_JSON.read_text(
                encoding="utf-8"
            )
        )

        for record in stored_manifest[
            "artifacts"
        ].values():
            path = Path(
                record["path"]
            )

            if not path.exists():
                raise RuntimeError(
                    f"Frozen Step-19 artifact missing: {path}"
                )

            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(
                    f"Frozen Step-19 artifact mismatch: {path}"
                )

        if (
            sha256_file(
                Path(
                    step00["lock_path"]
                )
            )
            !=
            step00["lock_sha256"]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step-19 validation."
            )

        stage = "report"

        print(
            "[9/9] Final Step-19 report..."
        )

        print()
        print("-" * 136)
        print("FROZEN REPOSITORY RECHECK")
        print("-" * 136)
        print(
            "Repositories rechecked              : "
            f"{len(repository_recheck)}"
        )
        print(
            "Origins/commits/trees/branches exact: "
            f"{bool(repository_recheck['passed'].all())}"
        )
        print(
            "Worktrees clean incl. untracked     : "
            f"{bool(repository_recheck['worktree_clean_including_untracked'].all())}"
        )
        print(
            "Required-file hashes exact          : "
            f"{bool(repository_recheck['required_file_hashes_match'].all())}"
        )

        print("-" * 136)
        print("ARCHITECTURE ADAPTATION LOCI")
        print("-" * 136)

        for row in summary_df.itertuples(
            index=False
        ):
            terminal = (
                row.likely_terminal_attribute_candidates
                if row.likely_terminal_attribute_candidates
                else
                "<none automatically identified>"
            )

            calls = (
                row.forward_self_calls
                if row.forward_self_calls
                else
                "<none>"
            )

            print(
                f"{row.model_id}"
            )
            print(
                f"  class             : {row.preferred_class}"
            )
            print(
                f"  methods           : {row.class_methods}"
            )
            print(
                f"  terminal candidates: {terminal}"
            )
            print(
                f"  forward self calls : {calls}"
            )
            print(
                f"  forward assigns    : {row.forward_assignment_count}"
            )
            print(
                f"  return exprs       : {row.return_expression_count}"
            )
            print(
                f"  shape operations   : {row.shape_operation_count}"
            )
            print(
                f"  task conditions    : {row.task_dispatch_condition_count}"
            )
            print(
                f"  readiness          : {row.adapter_readiness}"
            )

            model_returns = (
                return_df.loc[
                    return_df[
                        "model_id"
                    ]
                    ==
                    row.model_id
                ]
                .sort_values(
                    "line"
                )
            )

            for return_row in model_returns.itertuples(
                index=False
            ):
                print(
                    f"  return L{return_row.line:<4}: "
                    f"{return_row.return_expression}"
                )

            model_dispatch = (
                task_dispatch_df.loc[
                    task_dispatch_df[
                        "model_id"
                    ]
                    ==
                    row.model_id
                ]
                .sort_values(
                    [
                        "method",
                        "line",
                    ]
                )
            )

            for dispatch_row in model_dispatch.itertuples(
                index=False
            ):
                print(
                    f"  dispatch "
                    f"{dispatch_row.method}:L{dispatch_row.line}: "
                    f"{dispatch_row.condition}"
                )

            print()

        print("-" * 136)
        print("STATIC AUDIT COUNTS")
        print("-" * 136)
        print(
            f"Models audited                  : {len(summary_df)}"
        )
        print(
            f"__init__ self assignments       : {len(init_df)}"
        )
        print(
            f"forward data-flow assignments   : {len(forward_flow_df)}"
        )
        print(
            f"forward self.* calls            : {len(forward_self_call_df)}"
        )
        print(
            f"forward return expressions      : {len(return_df)}"
        )
        print(
            f"shape-changing operations       : {len(shape_df)}"
        )
        print(
            f"task-dispatch conditions        : {len(task_dispatch_df)}"
        )
        print(
            f"class method signatures         : {len(method_signature_df)}"
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
            "Source APIs                      : FROZEN / AUDITED"
        )
        print(
            "Architecture adaptation loci     : AUDITED + FROZEN"
        )
        print(
            "Terminal candidates              : HEURISTIC ONLY"
        )
        print(
            "Executable adapter contracts     : NOT YET FROZEN"
        )
        print(
            "Task adapters                    : NOT IMPLEMENTED"
        )
        print(
            "Forward/backward smoke tests     : NOT RUN"
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
            f"Model locus summary      : {MODEL_LOCUS_SUMMARY_CSV}"
        )
        print(
            f"Init assignments         : {INIT_ASSIGNMENT_AUDIT_CSV}"
        )
        print(
            f"Forward dataflow         : {FORWARD_FLOW_AUDIT_CSV}"
        )
        print(
            f"Forward self calls       : {FORWARD_SELF_CALL_AUDIT_CSV}"
        )
        print(
            f"Return audit             : {RETURN_AUDIT_CSV}"
        )
        print(
            f"Shape-operation audit    : {SHAPE_OPERATION_AUDIT_CSV}"
        )
        print(
            f"Task dispatch audit      : {TASK_DISPATCH_AUDIT_CSV}"
        )
        print(
            f"Method signatures        : {METHOD_SIGNATURE_AUDIT_CSV}"
        )
        print(
            f"Adapter-locus freeze     : {ADAPTER_LOCUS_FREEZE_JSON}"
        )
        print(
            f"Adapter-locus fingerprint: {locus_fingerprint}"
        )
        print(
            f"Manifest                 : {MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256         : {manifest_sha}"
        )

        print("=" * 136)
        print(
            "PASS: Frozen architecture internals were inspected without "
            "importing or executing third-party code."
        )
        print(
            "PASS: Step-17 source hashes and Step-18 constructor/forward "
            "signatures were reconciled exactly before locus analysis."
        )
        print(
            "PASS: Terminal candidates, forward dataflow, returns, shape "
            "operations and task dispatch were frozen as static evidence."
        )
        print(
            "PASS: Terminal candidates remain explicitly heuristic; no adapter "
            "implementation decision was silently treated as proven."
        )
        print(
            "PASS: No model was instantiated or trained and the sealed final-test "
            "parquet remained unopened."
        )
        print("=" * 136)
        print(
            "Do not run the old Step-19 script. "
            "Do not implement/train adapters yet. Paste the complete terminal output for audit."
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
            f"19_adapter_locus_audit_FAILED_{stamp}.json"
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
                "python_executable": (
                    sys.executable
                ),
                "archived_partial_artifacts": (
                    None
                    if archived is None
                    else str(
                        archived
                    )
                ),
                "canonical_manifest_exists": (
                    MANIFEST_JSON.exists()
                ),
            },
        )

        print()
        print("=" * 136)
        print("STEP 19 FAILED — ADAPTATION LOCI NOT FROZEN")
        print("=" * 136)
        print(
            f"Stage      : {stage}"
        )
        print(
            f"Error type : {type(exc).__name__}"
        )
        print(
            f"Error      : {exc}"
        )
        print(
            f"Diagnostic : {failure_path}"
        )

        if archived is not None:
            print(
                f"Archived partial Step-19 outputs: {archived}"
            )

        print("=" * 136)
        print(
            "Do not proceed to executable adapter implementation or model training."
        )

        return 30


# =============================================================================
# 14. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean-V3 frozen architecture adaptation-locus audit."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an existing frozen Step-19 adaptation-locus audit "
            "without importing/executing third-party code."
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
        sys.exit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "\nStep 19 interrupted by user."
        )
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 136)
        print("STEP 19 UNHANDLED FAILURE")
        print("=" * 136)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        traceback.print_exc()
        print("=" * 136)
        sys.exit(1)
