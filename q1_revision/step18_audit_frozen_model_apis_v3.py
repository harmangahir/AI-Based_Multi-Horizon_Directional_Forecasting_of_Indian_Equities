#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 18 — CLEAN V3 STATIC AUDIT OF FROZEN MODEL SOURCE APIS

Purpose
-------
Mechanically inspect the exact upstream source snapshots frozen in Step 17
BEFORE any local task-adapter implementation or model training.

For each of the eight Stage-1 architectures this stage records, without
importing or executing third-party model code:

- frozen repository commit and Git tree;
- frozen primary-source-file SHA-256;
- preferred architecture class;
- class bases;
- constructor signature;
- forward signature;
- all directly declared class methods;
- config/configs/args attribute references;
- top-level imports in the primary source file;
- native task-pathway signals (classification/forecast/imputation/anomaly);
- a preliminary, non-executable adapter route.

This is intentionally a STATIC source-API audit.

Step 18 DOES NOT:
- import third-party source code;
- execute third-party source code;
- install repository dependencies;
- alter frozen third-party repositories;
- finalize executable adapter contracts;
- implement task adapters;
- run forward/backward smoke tests;
- train any model;
- perform feature selection;
- estimate a probability threshold;
- open or hash the sealed final-test parquet.

The executable adapter contract + CPU/GPU forward/backward smoke-test gate is
left to the next stage, after these static API facts have been frozen.

Run
---
    python step18_audit_frozen_model_apis_v3.py

Verify
------
    python step18_audit_frozen_model_apis_v3.py --verify-only
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
        "step18_audit_frozen_model_apis_v3.py must be inside "
        "<repository>/q1_revision/."
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg


# =============================================================================
# 1. FROZEN DESIGN
# =============================================================================

STEP_NUMBER = 18
STEP_ID = "18"
STEP_NAME = "STEP 18 — STATIC AUDIT OF FROZEN MODEL SOURCE APIS"
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
}

STEP16_FREEZE = (
    cfg.MANIFEST_DIR / "16_stage1_model_registry_freeze.json"
)

STEP16_MODEL_REGISTRY = (
    cfg.AUDIT_DIR / "16_stage1_model_registry.csv"
)

STEP17_FREEZE = (
    cfg.MANIFEST_DIR / "17_official_model_source_freeze.json"
)

STEP17_MODEL_SOURCE_MAP = (
    cfg.AUDIT_DIR / "17_model_to_official_source_map.csv"
)

STEP10_TEST_SEAL = (
    cfg.MANIFEST_DIR / "10_final_test_seal.json"
)


# =============================================================================
# 3. OUTPUT PATHS
# =============================================================================

MODEL_API_AUDIT_CSV = (
    cfg.AUDIT_DIR / "18_model_api_audit.csv"
)

CLASS_METHOD_AUDIT_CSV = (
    cfg.AUDIT_DIR / "18_class_method_audit.csv"
)

CONFIG_REFERENCE_AUDIT_CSV = (
    cfg.AUDIT_DIR / "18_config_reference_audit.csv"
)

IMPORT_AUDIT_CSV = (
    cfg.AUDIT_DIR / "18_primary_source_import_audit.csv"
)

REPOSITORY_RECHECK_CSV = (
    cfg.AUDIT_DIR / "18_repository_freeze_recheck.csv"
)

ADAPTER_PREREQUISITE_CSV = (
    cfg.AUDIT_DIR / "18_adapter_prerequisite_audit.csv"
)

SOURCE_HASH_CSV = (
    cfg.AUDIT_DIR / "18_source_file_hashes.csv"
)

SOURCE_API_FREEZE_JSON = (
    cfg.MANIFEST_DIR / "18_source_api_freeze.json"
)

MANIFEST_JSON = (
    cfg.MANIFEST_DIR / "18_source_api_manifest.json"
)

MANIFEST_SHA = (
    cfg.MANIFEST_DIR / "18_source_api_manifest.sha256"
)

HASH_REGISTER_CSV = (
    cfg.AUDIT_DIR / "18_source_api_hashes.csv"
)

FROZEN_SCRIPT = (
    cfg.MANIFEST_DIR / "step18_audit_frozen_model_apis_frozen.py"
)

SUMMARY_TXT = (
    cfg.REPORT_DIR / "18_source_api_summary.txt"
)

FAILED_DIR = (
    cfg.AUDIT_DIR / "step18_failed_attempts"
)


# =============================================================================
# 4. PREFERRED SOURCE CLASSES
# =============================================================================

# These are source-level class identities only.  Step 18 does NOT instantiate
# them.  If a frozen upstream snapshot changes class names, the audit fails
# rather than silently selecting another architecture.
PREFERRED_CLASSES = {
    "DLINEAR": "Model",
    "NLINEAR": "Model",
    "PATCHTST": "PatchTST_backbone",
    "TIMESNET": "Model",
    "ITRANSFORMER": "Model",
    "FREDFORMER": "Model",
    "FRNET": "Model",
    "MLF": "Model",
}


# =============================================================================
# 5. PRELIMINARY NON-EXECUTABLE ADAPTER ROUTES
# =============================================================================

PRELIMINARY_ADAPTER_ROUTES = {
    "DLINEAR": (
        "Preserve the frozen DLinear moving-average decomposition and temporal "
        "linear backbone; replace only the terminal forecasting output with a "
        "three-logit 1d/3d/5d directional head."
    ),
    "NLINEAR": (
        "Preserve the frozen NLinear last-value normalization and temporal "
        "linear transformation; adapt the terminal output to three directional logits."
    ),
    "PATCHTST": (
        "Preserve the frozen supervised PatchTST patching, channel-independent "
        "Transformer backbone and representation flow; adapt only the terminal "
        "task head to three directional logits."
    ),
    "TIMESNET": (
        "Inspect and preferentially reuse the frozen source's native "
        "classification pathway when compatible with the common 60x63 input; "
        "otherwise retain TimesBlock temporal 2D-variation features and expose "
        "three independent directional logits."
    ),
    "ITRANSFORMER": (
        "Preserve the frozen inverted variate-token embedding and Transformer "
        "encoder; adapt the terminal representation/projection to three "
        "directional logits."
    ),
    "FREDFORMER": (
        "Preserve the frozen frequency-debiased Fredformer backbone and adapt "
        "only the terminal forecasting output to three directional logits."
    ),
    "FRNET": (
        "Preserve the frozen frequency-rotation FRNet backbone and adapt only "
        "the terminal forecasting output to three directional logits."
    ),
    "MLF": (
        "Preserve the frozen multi-period representation/integration mechanism; "
        "the later executable adapter may expose nested 20/40/60-session views "
        "derived causally from the same 60-session input, then emit three "
        "directional logits."
    ),
}


# =============================================================================
# 6. GENERIC HELPERS
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

    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
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


# =============================================================================
# 7. STEP 00 / ENVIRONMENT / UPSTREAM CHAIN
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

    for step in range(1, 18):
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

    for step in range(3, 18):
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

    stored_registry_fingerprint = payload16.pop(
        "registry_freeze_fingerprint_sha256",
        None,
    )

    if not stored_registry_fingerprint:
        raise RuntimeError(
            "Step-16 registry freeze lacks its fingerprint."
        )

    if (
        cfg.canonical_json_sha256(payload16)
        !=
        stored_registry_fingerprint
    ):
        raise RuntimeError(
            "Step-16 registry freeze fingerprint mismatch."
        )

    if (
        step16_manifest.get(
            "registry_freeze_fingerprint_sha256"
        )
        !=
        stored_registry_fingerprint
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

    stored_source_fingerprint = payload17.pop(
        "source_freeze_fingerprint_sha256",
        None,
    )

    if not stored_source_fingerprint:
        raise RuntimeError(
            "Step-17 source freeze lacks its fingerprint."
        )

    if (
        cfg.canonical_json_sha256(payload17)
        !=
        stored_source_fingerprint
    ):
        raise RuntimeError(
            "Step-17 source freeze fingerprint mismatch."
        )

    if (
        step17_manifest.get(
            "source_freeze_fingerprint_sha256"
        )
        !=
        stored_source_fingerprint
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
            "Step-17 official/author-maintained sources are not formally frozen."
        )

    if (
        freeze17
        .get("adapter_state", {})
        .get("models_trained")
        is not False
    ):
        raise RuntimeError(
            "Step-17 freeze says models have been trained."
        )

    if (
        freeze17
        .get("adapter_state", {})
        .get("task_adapters_implemented")
        is not False
    ):
        raise RuntimeError(
            "Step-17 freeze says task adapters are already implemented."
        )

    # Verify key Step-16/17 artifacts against their manifests.
    for path, manifest in (
        (
            STEP16_MODEL_REGISTRY,
            step16_manifest,
        ),
        (
            STEP17_MODEL_SOURCE_MAP,
            step17_manifest,
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
                f"Canonical artifact path mismatch: {path.name}"
            )

        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(
                f"Artifact SHA mismatch: {path.name}"
            )

    registry = pd.read_csv(
        STEP16_MODEL_REGISTRY
    )

    if tuple(
        registry["model_id"].astype(str)
    ) != STAGE1_MODEL_IDS:
        raise RuntimeError(
            "Step-16 registry does not contain the exact clean-V3 Stage-1 model IDs."
        )

    source_map_csv = pd.read_csv(
        STEP17_MODEL_SOURCE_MAP
    )

    if tuple(
        source_map_csv["model_id"].astype(str)
    ) != STAGE1_MODEL_IDS:
        raise RuntimeError(
            "Step-17 model-source CSV does not match the Step-16 model order."
        )

    frozen_source_map_ids = tuple(
        row["model_id"]
        for row in freeze17["model_source_map"]
    )

    if frozen_source_map_ids != STAGE1_MODEL_IDS:
        raise RuntimeError(
            "Step-17 frozen model-source map does not match Step 16."
        )

    repositories = freeze17.get(
        "repositories",
        []
    )

    if len(repositories) != EXPECTED_FROZEN_REPOSITORY_COUNT:
        raise RuntimeError(
            "Unexpected frozen repository count in Step 17."
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
            "Unexpected implementation-repository count in Step 17."
        )

    if reference_count != EXPECTED_REFERENCE_REPOSITORY_COUNT:
        raise RuntimeError(
            "Unexpected reference-repository count in Step 17."
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
            stored_registry_fingerprint
        ),
        "step17_freeze": freeze17,
        "step17_source_fingerprint_sha256": (
            stored_source_fingerprint
        ),
        "test_seal_status": seal["status"],
        "sealed_test_sha256_from_metadata_only": (
            seal.get("sha256")
        ),
    }


# =============================================================================
# 8. IMMUTABILITY / FAILURE HANDLING
# =============================================================================

def canonical_outputs() -> tuple[Path, ...]:
    return (
        MODEL_API_AUDIT_CSV,
        CLASS_METHOD_AUDIT_CSV,
        CONFIG_REFERENCE_AUDIT_CSV,
        IMPORT_AUDIT_CSV,
        REPOSITORY_RECHECK_CSV,
        ADAPTER_PREREQUISITE_CSV,
        SOURCE_HASH_CSV,
        SOURCE_API_FREEZE_JSON,
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
            "Step-18 canonical output(s) already exist. "
            "Refusing to overwrite.\n"
            +
            "\n".join(
                str(path)
                for path in collisions
            )
            +
            "\nUse --verify-only if Step 18 already passed."
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
# 9. AST HELPERS
# =============================================================================

def argument_text(
    argument: ast.arg,
) -> str:
    text = argument.arg

    if argument.annotation is not None:
        try:
            text += ": " + ast.unparse(
                argument.annotation
            )

        except Exception:
            pass

    return text


def function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> str:
    if node is None:
        return "<missing>"

    arguments = node.args
    parts = []

    positional = (
        list(arguments.posonlyargs)
        +
        list(arguments.args)
    )

    defaults = (
        [None]
        *
        (
            len(positional)
            -
            len(arguments.defaults)
        )
        +
        list(arguments.defaults)
    )

    posonly_count = len(
        arguments.posonlyargs
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
        text = argument_text(argument)

        if default is not None:
            try:
                text += "=" + ast.unparse(default)

            except Exception:
                text += "=<expr>"

        parts.append(text)

        if (
            posonly_count
            and
            index
            ==
            posonly_count
            -
            1
        ):
            parts.append("/")

    if arguments.vararg is not None:
        parts.append(
            "*"
            +
            argument_text(
                arguments.vararg
            )
        )

    elif arguments.kwonlyargs:
        parts.append("*")

    for (
        argument,
        default,
    ) in zip(
        arguments.kwonlyargs,
        arguments.kw_defaults,
    ):
        text = argument_text(argument)

        if default is not None:
            try:
                text += "=" + ast.unparse(default)

            except Exception:
                text += "=<expr>"

        parts.append(text)

    if arguments.kwarg is not None:
        parts.append(
            "**"
            +
            argument_text(
                arguments.kwarg
            )
        )

    return (
        "("
        +
        ", ".join(parts)
        +
        ")"
    )


class ConfigReferenceVisitor(
    ast.NodeVisitor
):
    def __init__(self) -> None:
        self.references: set[str] = set()

    def visit_Attribute(
        self,
        node: ast.Attribute,
    ) -> None:
        # Direct names: configs.foo / config.foo / args.foo
        if isinstance(
            node.value,
            ast.Name,
        ):
            if node.value.id in {
                "configs",
                "config",
                "args",
            }:
                self.references.add(
                    f"{node.value.id}.{node.attr}"
                )

        # Common self-held config objects: self.configs.foo etc.
        if isinstance(
            node.value,
            ast.Attribute,
        ):
            parent = node.value

            if (
                isinstance(
                    parent.value,
                    ast.Name,
                )
                and
                parent.value.id
                ==
                "self"
                and
                parent.attr
                in
                {
                    "configs",
                    "config",
                    "args",
                }
            ):
                self.references.add(
                    f"self.{parent.attr}.{node.attr}"
                )

        self.generic_visit(node)


class StringLiteralVisitor(
    ast.NodeVisitor
):
    def __init__(self) -> None:
        self.values: list[str] = []

    def visit_Constant(
        self,
        node: ast.Constant,
    ) -> None:
        if isinstance(node.value, str):
            self.values.append(
                node.value
            )


def extract_imports(
    tree: ast.Module,
    model_id: str,
    source_file: str,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                rows.append(
                    {
                        "model_id": model_id,
                        "source_file": source_file,
                        "import_type": "import",
                        "module": alias.name,
                        "names": "",
                        "level": 0,
                    }
                )

        elif isinstance(node, ast.ImportFrom):
            rows.append(
                {
                    "model_id": model_id,
                    "source_file": source_file,
                    "import_type": "from",
                    "module": (
                        node.module
                        or
                        ""
                    ),
                    "names": ";".join(
                        alias.name
                        for alias in node.names
                    ),
                    "level": int(node.level),
                }
            )

    return rows


def top_level_classes(
    tree: ast.Module,
) -> list[ast.ClassDef]:
    return [
        node
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
    ]


def get_method(
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


def get_class_methods(
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


# =============================================================================
# 10. RECHECK STEP-17 FROZEN REPOSITORIES
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
                f"Frozen repository is no longer a Git checkout:\n{path}"
            )

        observed_origin = run_git(
            [
                "remote",
                "get-url",
                "origin",
            ],
            cwd=path,
        )

        expected_origin = record[
            "repository_url"
        ]

        def normalize_git_url(value: str) -> str:
            normalized = value.strip().rstrip("/")

            if normalized.endswith(".git"):
                normalized = normalized[:-4]

            return normalized.lower()

        origin_match = (
            normalize_git_url(
                observed_origin
            )
            ==
            normalize_git_url(
                expected_origin
            )
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

        required_hashes_match = True
        checked_required_files = 0

        for file_record in record[
            "required_file_hashes"
        ]:
            source_path = (
                path
                /
                file_record[
                    "relative_path"
                ]
            )

            if (
                not source_path.exists()
                or
                not source_path.is_file()
            ):
                required_hashes_match = False
                continue

            checked_required_files += 1

            observed_hash = sha256_file(
                source_path
            )

            if (
                observed_hash
                !=
                file_record["sha256"]
            ):
                required_hashes_match = False

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

        rows.append(
            {
                "repo_id": repo_id,
                "repository_role": (
                    record["repository_role"]
                ),
                "local_path": str(path),
                "origin_match": origin_match,
                "expected_commit_sha": (
                    record["commit_sha"]
                ),
                "observed_commit_sha": (
                    observed_commit
                ),
                "commit_match": commit_match,
                "expected_git_tree_sha": (
                    record["git_tree_sha"]
                ),
                "observed_git_tree_sha": (
                    observed_tree
                ),
                "tree_match": tree_match,
                "expected_branch": (
                    record["branch"]
                ),
                "observed_branch": (
                    observed_branch
                ),
                "branch_match": branch_match,
                "worktree_clean_including_untracked": clean,
                "required_files_checked": int(
                    checked_required_files
                ),
                "required_file_hashes_match": (
                    required_hashes_match
                ),
                "passed": passed,
            }
        )

        repository_map[repo_id] = record

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

    return (
        repository_map,
        pd.DataFrame(rows),
    )


# =============================================================================
# 11. SOURCE-FILE API INSPECTION
# =============================================================================

def frozen_required_file_record(
    repository_record: Dict[str, Any],
    relative_path: str,
) -> Dict[str, Any]:
    matches = [
        row
        for row in repository_record[
            "required_file_hashes"
        ]
        if row["relative_path"]
        ==
        relative_path
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one frozen required-file record for "
            f"{repository_record['repo_id']}::{relative_path}; "
            f"found {len(matches)}."
        )

    return matches[0]


def inspect_model_source(
    model_mapping: Dict[str, Any],
    repository_record: Dict[str, Any],
) -> tuple[
    Dict[str, Any],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    Dict[str, Any],
]:
    model_id = model_mapping[
        "model_id"
    ]

    source_file = model_mapping[
        "primary_source_file"
    ]

    preferred_class = PREFERRED_CLASSES[
        model_id
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
            f"Primary source missing for {model_id}:\n{path}"
        )

    frozen_file_record = (
        frozen_required_file_record(
            repository_record,
            source_file,
        )
    )

    source_hash = sha256_file(
        path
    )

    if (
        source_hash
        !=
        frozen_file_record["sha256"]
    ):
        raise RuntimeError(
            "\nPRIMARY SOURCE HASH DIFFERS FROM STEP-17 FREEZE\n"
            f"Model   : {model_id}\n"
            f"Source  : {source_file}\n"
            f"Frozen  : {frozen_file_record['sha256']}\n"
            f"Observed: {source_hash}"
        )

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    try:
        tree = ast.parse(text)

    except SyntaxError as exc:
        raise RuntimeError(
            f"AST parse failed for {path}: {exc}"
        ) from exc

    classes = top_level_classes(
        tree
    )

    class_names = [
        node.name
        for node in classes
    ]

    matching = [
        node
        for node in classes
        if node.name
        ==
        preferred_class
    ]

    if len(matching) != 1:
        raise RuntimeError(
            "\nPREFERRED SOURCE CLASS NOT FOUND\n"
            f"Model          : {model_id}\n"
            f"Source         : {source_file}\n"
            f"Expected class : {preferred_class}\n"
            f"Classes found  : {class_names}"
        )

    class_node = matching[0]

    init_node = get_method(
        class_node,
        "__init__",
    )

    forward_node = get_method(
        class_node,
        "forward",
    )

    if forward_node is None:
        raise RuntimeError(
            f"{model_id}: preferred class has no forward() method."
        )

    methods = get_class_methods(
        class_node
    )

    method_names = [
        node.name
        for node in methods
    ]

    # -------------------------------------------------------------------------
    # Config references.
    # -------------------------------------------------------------------------
    config_visitor = (
        ConfigReferenceVisitor()
    )

    config_visitor.visit(
        class_node
    )

    config_references = sorted(
        config_visitor.references
    )

    config_rows = [
        {
            "model_id": model_id,
            "repo_id": (
                model_mapping[
                    "repo_id"
                ]
            ),
            "source_file": source_file,
            "preferred_class": preferred_class,
            "config_reference": reference,
        }
        for reference in config_references
    ]

    # -------------------------------------------------------------------------
    # Native task signals from method names / class string literals.
    # -------------------------------------------------------------------------
    literal_visitor = (
        StringLiteralVisitor()
    )

    literal_visitor.visit(
        class_node
    )

    combined_literals = (
        "\n".join(
            literal_visitor.values
        )
        .lower()
    )

    has_classification_method = (
        "classification"
        in
        method_names
    )

    has_forecast_method = any(
        name
        in
        method_names
        for name in (
            "forecast",
            "long_forecast",
            "short_forecast",
        )
    )

    has_imputation_method = (
        "imputation"
        in
        method_names
    )

    has_anomaly_method = any(
        "anomaly"
        in
        method_name.lower()
        for method_name in method_names
    )

    mentions_classification = (
        "classification"
        in
        combined_literals
        or
        "classify"
        in
        combined_literals
    )

    mentions_forecast = (
        "forecast"
        in
        combined_literals
        or
        "prediction"
        in
        combined_literals
    )

    mentions_imputation = (
        "imputation"
        in
        combined_literals
    )

    mentions_anomaly = (
        "anomaly"
        in
        combined_literals
    )

    class_bases = []

    for base in class_node.bases:
        try:
            class_bases.append(
                ast.unparse(base)
            )

        except Exception:
            class_bases.append(
                "<unparsed>"
            )

    method_rows = []

    for method in methods:
        method_rows.append(
            {
                "model_id": model_id,
                "repo_id": (
                    model_mapping[
                        "repo_id"
                    ]
                ),
                "source_file": source_file,
                "class_name": preferred_class,
                "method_name": method.name,
                "signature": (
                    function_signature(
                        method
                    )
                ),
                "is_async": isinstance(
                    method,
                    ast.AsyncFunctionDef,
                ),
            }
        )

    import_rows = extract_imports(
        tree=tree,
        model_id=model_id,
        source_file=source_file,
    )

    api_record = {
        "model_id": model_id,
        "repo_id": (
            model_mapping[
                "repo_id"
            ]
        ),
        "reference_id": (
            model_mapping[
                "reference_id"
            ]
        ),
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
        "source_sha256": source_hash,
        "source_hash_matches_step17_freeze": True,
        "source_line_count": int(
            len(
                text.splitlines()
            )
        ),
        "top_level_classes": ";".join(
            class_names
        ),
        "preferred_class": preferred_class,
        "class_bases": ";".join(
            class_bases
        ),
        "constructor_signature": (
            function_signature(
                init_node
            )
        ),
        "forward_signature": (
            function_signature(
                forward_node
            )
        ),
        "class_method_count": int(
            len(methods)
        ),
        "class_methods": ";".join(
            method_names
        ),
        "config_reference_count": int(
            len(config_references)
        ),
        "import_count": int(
            len(import_rows)
        ),
        "has_classification_method": (
            has_classification_method
        ),
        "has_forecast_method": (
            has_forecast_method
        ),
        "has_imputation_method": (
            has_imputation_method
        ),
        "has_anomaly_method": (
            has_anomaly_method
        ),
        "mentions_classification": (
            mentions_classification
        ),
        "mentions_forecast": (
            mentions_forecast
        ),
        "mentions_imputation": (
            mentions_imputation
        ),
        "mentions_anomaly": (
            mentions_anomaly
        ),
        "third_party_source_imported": False,
        "third_party_source_executed": False,
    }

    prerequisite = {
        "model_id": model_id,
        "preferred_class_found": True,
        "forward_method_found": True,
        "source_ast_valid": True,
        "source_commit_frozen": True,
        "source_file_hash_matches_step17_freeze": True,
        "native_classification_signal": bool(
            has_classification_method
            or
            mentions_classification
        ),
        "native_forecasting_signal": bool(
            has_forecast_method
            or
            mentions_forecast
        ),
        "preliminary_adapter_route": (
            PRELIMINARY_ADAPTER_ROUTES[
                model_id
            ]
        ),
        "common_input_contract_target": (
            "[batch,60,63]"
        ),
        "common_output_contract_target": (
            "[batch,3] logits"
        ),
        "adapter_contract_finalized": False,
        "adapter_implemented": False,
        "smoke_test_run": False,
    }

    return (
        api_record,
        method_rows,
        config_rows,
        import_rows,
        prerequisite,
    )


# =============================================================================
# 12. SOURCE HASH REGISTER
# =============================================================================

def source_hashes() -> pd.DataFrame:
    sources = [
        (
            "current_step00",
            Path(cfg.__file__).resolve(),
        ),
        (
            "current_step18",
            SCRIPT_FILE.resolve(),
        ),
        (
            "step16_registry_freeze",
            STEP16_FREEZE.resolve(),
        ),
        (
            "step16_model_registry",
            STEP16_MODEL_REGISTRY.resolve(),
        ),
        (
            "step17_source_freeze",
            STEP17_FREEZE.resolve(),
        ),
        (
            "step17_model_source_map",
            STEP17_MODEL_SOURCE_MAP.resolve(),
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
                "Final-test parquet entered Step-18 source list."
            )

        if not path.exists():
            raise RuntimeError(
                f"Required Step-18 source missing: {path}"
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

    return pd.DataFrame(rows)


# =============================================================================
# 13. VERIFY EXISTING STEP 18
# =============================================================================

def verify_existing() -> int:
    print("=" * 136)
    print("STEP 18 — VERIFY EXISTING STATIC SOURCE-API FREEZE")
    print("=" * 136)

    step00 = verify_step00()
    upstream = verify_upstream(
        step00
    )

    manifest_sha = verify_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 18",
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
            "[FAIL] Stored Step-18 status is not PASS."
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

    for step in range(1, 18):
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
        print("STEP 18 VERIFICATION STATUS: FAIL")
        print("=" * 136)
        return 25

    freeze = json.loads(
        SOURCE_API_FREEZE_JSON.read_text(
            encoding="utf-8"
        )
    )

    stored_fingerprint = freeze.get(
        "source_api_fingerprint_sha256"
    )

    payload = dict(freeze)
    payload.pop(
        "source_api_fingerprint_sha256",
        None,
    )

    recomputed = cfg.canonical_json_sha256(
        payload
    )

    if stored_fingerprint != recomputed:
        print(
            "[FAIL] Source-API fingerprint mismatch."
        )
        return 26

    if (
        manifest.get(
            "source_api_fingerprint_sha256"
        )
        !=
        stored_fingerprint
    ):
        print(
            "[FAIL] Manifest/source-API fingerprint mismatch."
        )
        return 27

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
        return 28

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
        return 29

    print(
        "[PASS] Configuration fingerprint : "
        f"{step00['fingerprint']}"
    )
    print(
        "[PASS] Step-17 manifest          : "
        f"{upstream['manifest_sha'][17]}"
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
        "[PASS] Frozen repositories       : "
        f"{len(repository_map)}"
    )
    print(
        "[PASS] Model APIs audited        : "
        f"{len(freeze['source_api_records'])}"
    )
    print(
        "[PASS] Step-18 source API freeze : "
        f"{stored_fingerprint}"
    )
    print(
        "[PASS] Step-18 manifest          : "
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
    print("STEP 18 VERIFICATION STATUS: PASS")
    print("=" * 136)

    return 0


# =============================================================================
# 14. MAIN RUN
# =============================================================================

def run() -> int:
    print("=" * 136)
    print(STEP_NAME)
    print("=" * 136)

    stage = "verify"

    try:
        print(
            "[1/9] Verifying Step 00 and immutable Step 01 -> Step 17 chain..."
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
            "       Step-17 manifest          : "
            f"{upstream['manifest_sha'][17]}"
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
            "[2/9] Checking immutable Step-18 output target..."
        )

        assert_fresh_target()
        cfg.create_directories()

        stage = "repository_recheck"

        print(
            "[3/9] Rechecking all frozen Step-17 repository snapshots..."
        )

        repository_map, repository_recheck = (
            recheck_repositories(
                upstream[
                    "step17_freeze"
                ]
            )
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
            "       Worktrees clean             : "
            f"{bool(repository_recheck['worktree_clean_including_untracked'].all())}"
        )
        print(
            "       Required-file hashes match  : "
            f"{bool(repository_recheck['required_file_hashes_match'].all())}"
        )

        stage = "api_inspection"

        print(
            "[4/9] Statically inspecting the eight frozen primary model APIs..."
        )

        model_source_map = (
            upstream[
                "step17_freeze"
            ][
                "model_source_map"
            ]
        )

        mapped_ids = tuple(
            row["model_id"]
            for row in model_source_map
        )

        if mapped_ids != STAGE1_MODEL_IDS:
            raise RuntimeError(
                "Step-17 frozen model-source map order differs from Step 16."
            )

        if set(PREFERRED_CLASSES) != set(
            STAGE1_MODEL_IDS
        ):
            raise RuntimeError(
                "Preferred-class map does not cover exactly eight Stage-1 models."
            )

        if set(
            PRELIMINARY_ADAPTER_ROUTES
        ) != set(
            STAGE1_MODEL_IDS
        ):
            raise RuntimeError(
                "Preliminary adapter-route map does not cover exactly eight Stage-1 models."
            )

        api_rows = []
        method_rows = []
        config_rows = []
        import_rows = []
        prerequisite_rows = []

        for mapping in model_source_map:
            repo_id = mapping["repo_id"]

            if repo_id not in repository_map:
                raise RuntimeError(
                    f"No frozen repository record for {repo_id}."
                )

            (
                api_record,
                model_method_rows,
                model_config_rows,
                model_import_rows,
                prerequisite,
            ) = inspect_model_source(
                model_mapping=mapping,
                repository_record=(
                    repository_map[
                        repo_id
                    ]
                ),
            )

            api_rows.append(
                api_record
            )

            method_rows.extend(
                model_method_rows
            )

            config_rows.extend(
                model_config_rows
            )

            import_rows.extend(
                model_import_rows
            )

            prerequisite_rows.append(
                prerequisite
            )

            print(
                f"       {api_record['model_id']:<14} | "
                f"class={api_record['preferred_class']:<20} | "
                f"methods={api_record['class_method_count']:>2} | "
                f"config_refs={api_record['config_reference_count']:>2} | "
                f"imports={api_record['import_count']:>2}"
            )
            print(
                f"           __init__ {api_record['constructor_signature']}"
            )
            print(
                f"           forward  {api_record['forward_signature']}"
            )

        model_api_audit = pd.DataFrame(
            api_rows
        )

        class_method_audit = pd.DataFrame(
            method_rows
        )

        config_reference_audit = pd.DataFrame(
            config_rows,
            columns=[
                "model_id",
                "repo_id",
                "source_file",
                "preferred_class",
                "config_reference",
            ],
        )

        import_audit = pd.DataFrame(
            import_rows,
            columns=[
                "model_id",
                "source_file",
                "import_type",
                "module",
                "names",
                "level",
            ],
        )

        adapter_prerequisite = pd.DataFrame(
            prerequisite_rows
        )

        stage = "structural_validation"

        print(
            "[5/9] Validating API/source integrity and adapter prerequisites..."
        )

        if len(model_api_audit) != EXPECTED_MODEL_COUNT:
            raise RuntimeError(
                "Expected exactly eight model API records."
            )

        if tuple(
            model_api_audit["model_id"].astype(str)
        ) != STAGE1_MODEL_IDS:
            raise RuntimeError(
                "Model API audit order differs from frozen Stage-1 model order."
            )

        if (
            model_api_audit[
                "forward_signature"
            ]
            ==
            "<missing>"
        ).any():
            raise RuntimeError(
                "One or more preferred classes lack forward()."
            )

        if not model_api_audit[
            "source_hash_matches_step17_freeze"
        ].all():
            raise RuntimeError(
                "One or more primary sources differ from Step-17 frozen hashes."
            )

        for column in (
            "preferred_class_found",
            "forward_method_found",
            "source_ast_valid",
            "source_commit_frozen",
            "source_file_hash_matches_step17_freeze",
        ):
            if not adapter_prerequisite[
                column
            ].all():
                raise RuntimeError(
                    f"Adapter prerequisite failure: {column}"
                )

        if not repository_recheck[
            "passed"
        ].all():
            raise RuntimeError(
                "Repository recheck failed."
            )

        print(
            "       Model APIs audited          : "
            f"{len(model_api_audit)}"
        )
        print(
            "       Preferred classes found     : "
            f"{bool(adapter_prerequisite['preferred_class_found'].all())}"
        )
        print(
            "       forward() methods found     : "
            f"{bool(adapter_prerequisite['forward_method_found'].all())}"
        )
        print(
            "       Primary source hashes exact : "
            f"{bool(model_api_audit['source_hash_matches_step17_freeze'].all())}"
        )
        print(
            "       Third-party source imported : False"
        )
        print(
            "       Third-party source executed : False"
        )

        stage = "write"

        print(
            "[6/9] Writing static API audits and source provenance..."
        )

        write_csv(
            MODEL_API_AUDIT_CSV,
            model_api_audit,
        )

        write_csv(
            CLASS_METHOD_AUDIT_CSV,
            class_method_audit,
        )

        write_csv(
            CONFIG_REFERENCE_AUDIT_CSV,
            config_reference_audit,
        )

        write_csv(
            IMPORT_AUDIT_CSV,
            import_audit,
        )

        write_csv(
            REPOSITORY_RECHECK_CSV,
            repository_recheck,
        )

        write_csv(
            ADAPTER_PREREQUISITE_CSV,
            adapter_prerequisite,
        )

        write_csv(
            SOURCE_HASH_CSV,
            source_hashes(),
        )

        stage = "freeze"

        print(
            "[7/9] Freezing the static source-API audit before adapter implementation..."
        )

        if (
            sha256_file(
                Path(step00["lock_path"])
            )
            !=
            step00["lock_sha256"]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step 18."
            )

        api_payload_records = []

        for row in (
            model_api_audit
            .sort_values(
                "model_id"
            )
            .itertuples(
                index=False
            )
        ):
            model_configs = (
                config_reference_audit.loc[
                    config_reference_audit[
                        "model_id"
                    ]
                    ==
                    row.model_id,
                    "config_reference",
                ]
                .tolist()
            )

            model_methods = (
                class_method_audit.loc[
                    class_method_audit[
                        "model_id"
                    ]
                    ==
                    row.model_id,
                    [
                        "method_name",
                        "signature",
                    ],
                ]
                .sort_values(
                    "method_name"
                )
                .to_dict(
                    orient="records"
                )
            )

            model_imports = (
                import_audit.loc[
                    import_audit[
                        "model_id"
                    ]
                    ==
                    row.model_id,
                    [
                        "import_type",
                        "module",
                        "names",
                        "level",
                    ],
                ]
                .to_dict(
                    orient="records"
                )
            )

            prerequisite = (
                adapter_prerequisite.loc[
                    adapter_prerequisite[
                        "model_id"
                    ]
                    ==
                    row.model_id
                ]
                .iloc[0]
                .to_dict()
            )

            api_payload_records.append(
                {
                    "model_id": row.model_id,
                    "repo_id": row.repo_id,
                    "reference_id": row.reference_id,
                    "repository_commit_sha": (
                        row.repository_commit_sha
                    ),
                    "repository_git_tree_sha": (
                        row.repository_git_tree_sha
                    ),
                    "primary_source_file": (
                        row.primary_source_file
                    ),
                    "source_sha256": (
                        row.source_sha256
                    ),
                    "preferred_class": (
                        row.preferred_class
                    ),
                    "class_bases": (
                        row.class_bases
                    ),
                    "constructor_signature": (
                        row.constructor_signature
                    ),
                    "forward_signature": (
                        row.forward_signature
                    ),
                    "methods": model_methods,
                    "config_references": (
                        model_configs
                    ),
                    "imports": model_imports,
                    "task_signals": {
                        "classification": bool(
                            row.has_classification_method
                            or
                            row.mentions_classification
                        ),
                        "forecast": bool(
                            row.has_forecast_method
                            or
                            row.mentions_forecast
                        ),
                        "imputation": bool(
                            row.has_imputation_method
                            or
                            row.mentions_imputation
                        ),
                        "anomaly": bool(
                            row.has_anomaly_method
                            or
                            row.mentions_anomaly
                        ),
                    },
                    "adapter_prerequisite": (
                        prerequisite
                    ),
                }
            )

        freeze_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "FROZEN_SOURCE_APIS_AUDITED_BEFORE_ADAPTER_IMPLEMENTATION"
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
            "source_api_records": (
                api_payload_records
            ),
            "static_adapter_policy_target": {
                "input_shape": (
                    "[batch,60,63]"
                ),
                "output_shape": (
                    "[batch,3] logits"
                ),
                "three_outputs_mean": (
                    "independent 1d/3d/5d binary-direction logits"
                ),
                "loss_target": (
                    "BCEWithLogitsLoss(reduction='mean')"
                ),
                "stage1_comparator_stock_embedding_allowed": False,
                "stage1_comparator_regime_conditioning_allowed": False,
                "stage1_comparator_auxiliary_regression_allowed": False,
                "published_backbone_preservation_required": True,
                "task_adaptation_reporting_required": (
                    "Each comparator must be described as an architecture-adapted "
                    "directional baseline rather than an unmodified forecasting implementation."
                ),
                "executable_adapter_contract_finalized_here": False,
                "task_adapter_implementation_performed_here": False,
                "smoke_tests_performed_here": False,
            },
            "pipeline_state": {
                "development_interface_frozen": True,
                "stage1_model_registry_frozen": True,
                "official_source_snapshots_frozen": True,
                "source_apis_audited": True,
                "task_adapter_contracts_finalized": False,
                "task_adapters_implemented": False,
                "task_adapter_smoke_tests_run": False,
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
                "final_test_parquet_opened_or_hashed_by_step18": False,
                "final_test_performance_computed": False,
                "global_final_test_state": (
                    step00["state"]
                ),
            },
        }

        api_fingerprint = (
            cfg.canonical_json_sha256(
                freeze_payload
            )
        )

        freeze_payload[
            "source_api_fingerprint_sha256"
        ] = api_fingerprint

        atomic_json(
            SOURCE_API_FREEZE_JSON,
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
                        "Frozen repositories rechecked: "
                        f"{len(repository_recheck)}"
                    ),
                    (
                        "Model APIs audited: "
                        f"{len(model_api_audit)}"
                    ),
                    (
                        "Preferred classes found: "
                        f"{bool(adapter_prerequisite['preferred_class_found'].all())}"
                    ),
                    (
                        "Forward methods found: "
                        f"{bool(adapter_prerequisite['forward_method_found'].all())}"
                    ),
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
            MODEL_API_AUDIT_CSV,
            CLASS_METHOD_AUDIT_CSV,
            CONFIG_REFERENCE_AUDIT_CSV,
            IMPORT_AUDIT_CSV,
            REPOSITORY_RECHECK_CSV,
            ADAPTER_PREREQUISITE_CSV,
            SOURCE_HASH_CSV,
            SOURCE_API_FREEZE_JSON,
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
            "experiment_tag": cfg.EXPERIMENT_TAG,
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
                    ][
                        step
                    ]
                )
                for step in range(1, 18)
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
            "source_api_fingerprint_sha256": (
                api_fingerprint
            ),
            "counts": {
                "repositories_rechecked": int(
                    len(repository_recheck)
                ),
                "models_audited": int(
                    len(model_api_audit)
                ),
                "preferred_class_methods": int(
                    len(class_method_audit)
                ),
                "config_references": int(
                    len(config_reference_audit)
                ),
                "primary_source_imports": int(
                    len(import_audit)
                ),
            },
            "pipeline_state": {
                "development_interface_frozen": True,
                "stage1_model_registry_frozen": True,
                "official_source_snapshots_frozen": True,
                "source_apis_audited": True,
                "adapter_contracts_frozen": False,
                "task_adapters_implemented": False,
                "task_adapter_smoke_tests_run": False,
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
                "sealed_test_parquet_opened_or_hashed_by_step18": False,
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
            "[8/9] Validating frozen Step-18 outputs..."
        )

        verify_sidecar(
            MANIFEST_JSON,
            MANIFEST_SHA,
            "Step 18",
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
                    f"Frozen Step-18 artifact missing: {path}"
                )

            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(
                    f"Frozen Step-18 artifact mismatch: {path}"
                )

        if (
            sha256_file(
                Path(step00["lock_path"])
            )
            !=
            step00["lock_sha256"]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step-18 validation."
            )

        stage = "report"

        print(
            "[9/9] Final Step-18 report..."
        )

        print()
        print("-" * 136)
        print("FROZEN REPOSITORY RECHECK")
        print("-" * 136)
        print(
            "Repositories rechecked             : "
            f"{len(repository_recheck)}"
        )
        print(
            "All origins match                  : "
            f"{bool(repository_recheck['origin_match'].all())}"
        )
        print(
            "All commits match                  : "
            f"{bool(repository_recheck['commit_match'].all())}"
        )
        print(
            "All Git trees match                : "
            f"{bool(repository_recheck['tree_match'].all())}"
        )
        print(
            "All branches match                 : "
            f"{bool(repository_recheck['branch_match'].all())}"
        )
        print(
            "All worktrees clean incl. untracked: "
            f"{bool(repository_recheck['worktree_clean_including_untracked'].all())}"
        )
        print(
            "All required-file hashes match     : "
            f"{bool(repository_recheck['required_file_hashes_match'].all())}"
        )

        print("-" * 136)
        print("MODEL SOURCE API AUDIT")
        print("-" * 136)

        for row in model_api_audit.itertuples(
            index=False
        ):
            native_classification = bool(
                row.has_classification_method
                or
                row.mentions_classification
            )

            native_forecast = bool(
                row.has_forecast_method
                or
                row.mentions_forecast
            )

            print(
                f"{row.model_id:<14} | "
                f"class={row.preferred_class:<20} | "
                f"methods={row.class_method_count:>2} | "
                f"config_refs={row.config_reference_count:>2} | "
                f"classification={str(native_classification):<5} | "
                f"forecast={str(native_forecast):<5}"
            )
            print(
                f"    __init__ {row.constructor_signature}"
            )
            print(
                f"    forward  {row.forward_signature}"
            )

        print("-" * 136)
        print("ADAPTER PREREQUISITES")
        print("-" * 136)
        print(
            "Models audited                      : "
            f"{len(adapter_prerequisite)}"
        )
        print(
            "Preferred classes found             : "
            f"{bool(adapter_prerequisite['preferred_class_found'].all())}"
        )
        print(
            "forward() methods found             : "
            f"{bool(adapter_prerequisite['forward_method_found'].all())}"
        )
        print(
            "AST source validity                 : "
            f"{bool(adapter_prerequisite['source_ast_valid'].all())}"
        )
        print(
            "Primary hashes equal Step-17 freeze : "
            f"{bool(adapter_prerequisite['source_file_hash_matches_step17_freeze'].all())}"
        )

        print("-" * 136)
        print("PIPELINE STATE")
        print("-" * 136)
        print(
            "Development interface               : FROZEN"
        )
        print(
            "Stage-1 registry                    : FROZEN"
        )
        print(
            "Official/author source snapshots    : FROZEN"
        )
        print(
            "Static frozen-source APIs           : AUDITED + FROZEN"
        )
        print(
            "Executable adapter contracts        : NOT YET FROZEN"
        )
        print(
            "Task adapters                       : NOT IMPLEMENTED"
        )
        print(
            "Forward/backward smoke tests        : NOT RUN"
        )
        print(
            "Models fitted                       : False"
        )
        print(
            "Final test accessed                 : False"
        )
        print(
            "Global final-test state             : "
            f"{step00['state']}"
        )

        print("-" * 136)
        print("OUTPUT FILES")
        print("-" * 136)
        print(
            f"Model API audit        : {MODEL_API_AUDIT_CSV}"
        )
        print(
            f"Class methods          : {CLASS_METHOD_AUDIT_CSV}"
        )
        print(
            f"Config references      : {CONFIG_REFERENCE_AUDIT_CSV}"
        )
        print(
            f"Import audit           : {IMPORT_AUDIT_CSV}"
        )
        print(
            f"Repository recheck     : {REPOSITORY_RECHECK_CSV}"
        )
        print(
            f"Adapter prerequisites  : {ADAPTER_PREREQUISITE_CSV}"
        )
        print(
            f"Source API freeze      : {SOURCE_API_FREEZE_JSON}"
        )
        print(
            f"Source API fingerprint : {api_fingerprint}"
        )
        print(
            f"Manifest               : {MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256       : {manifest_sha}"
        )

        print("=" * 136)
        print(
            "PASS: All eight frozen repository snapshots remain "
            "origin/commit/tree/branch/hash consistent with Step 17."
        )
        print(
            "PASS: Preferred architecture classes and forward() interfaces "
            "were identified for all eight Stage-1 comparators."
        )
        print(
            "PASS: Constructor signatures, class methods, configuration references, "
            "imports and native task signals were recorded statically."
        )
        print(
            "PASS: No third-party source was imported or executed, and no task "
            "adapter or model was trained."
        )
        print(
            "PASS: The sealed final-test parquet remained unopened."
        )
        print("=" * 136)
        print(
            "Do not run the old Step-18 script. "
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
            f"18_source_api_audit_FAILED_{stamp}.json"
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
        print("STEP 18 FAILED — FROZEN SOURCE APIS NOT FROZEN")
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
                f"Archived partial Step-18 outputs: {archived}"
            )

        print("=" * 136)
        print(
            "Do not proceed to adapter implementation or model training."
        )

        return 30


# =============================================================================
# 15. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean-V3 static audit of frozen model source APIs."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an existing frozen Step-18 source-API audit "
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
            "\nStep 18 interrupted by user."
        )
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 136)
        print("STEP 18 UNHANDLED FAILURE")
        print("=" * 136)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        traceback.print_exc()
        print("=" * 136)
        sys.exit(1)
