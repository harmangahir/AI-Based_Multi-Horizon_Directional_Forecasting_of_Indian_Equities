#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 17 — CLEAN V3 OFFICIAL/AUTHOR-MAINTAINED MODEL-SOURCE SNAPSHOT FREEZE

Purpose
-------
Freeze the exact upstream source-code snapshots that will be inspected when
implementing the eight Stage-1 task adapters frozen in Step 16.

This stage is SOURCE-PROVENANCE ONLY.

It:
- verifies the immutable Step 00 -> Step 16 chain;
- verifies the Step-16 registry freeze fingerprint;
- verifies the exact eight Stage-1 model IDs from the frozen registry CSV;
- acquires the public upstream repositories by shallow clone on first use;
- NEVER fetches/pulls an existing repository;
- records repository origin, branch, HEAD commit, Git tree and required-file
  SHA-256 hashes;
- parses selected Python files with ast.parse() without importing/executing them;
- freezes an explicit model -> upstream-source mapping;
- records the special TimesNet provenance:
    thuml/TimesNet = authors' reference repository;
    thuml/Time-Series-Library = authors' maintained complete implementation;
- preserves the final-test seal.

It DOES NOT:
- import third-party model code;
- execute third-party model code;
- install repository requirements;
- implement task adapters;
- run forward/gradient smoke tests;
- train any model;
- perform feature selection;
- estimate a probability threshold;
- open or hash the sealed final-test parquet.

First run
---------
    python step17_freeze_official_model_sources_v3.py

Verify existing freeze
----------------------
    python step17_freeze_official_model_sources_v3.py --verify-only
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata as importlib_metadata
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
        "step17_freeze_official_model_sources_v3.py must be inside "
        "<repository>/q1_revision/."
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg


# =============================================================================
# 1. FROZEN DESIGN
# =============================================================================

STEP_NUMBER = 17
STEP_ID = "17"
STEP_NAME = "STEP 17 — OFFICIAL/AUTHOR-MAINTAINED MODEL-SOURCE SNAPSHOT FREEZE"
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

EXPECTED_STAGE1_MODEL_COUNT = 8

# Seven implementation repositories serve eight models because DLinear/NLinear
# share one repository.  The eighth cloned repository is the TimesNet authors'
# lightweight reference repository, whose README points users to the complete
# implementation in Time-Series-Library.
EXPECTED_IMPLEMENTATION_REPOSITORY_COUNT = 7
EXPECTED_REFERENCE_REPOSITORY_COUNT = 1
EXPECTED_CLONED_REPOSITORY_COUNT = 8
EXPECTED_MODEL_SOURCE_MAP_COUNT = 8

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
}

STEP16_FREEZE = (
    cfg.MANIFEST_DIR / "16_stage1_model_registry_freeze.json"
)

STEP16_MODEL_REGISTRY = (
    cfg.AUDIT_DIR / "16_stage1_model_registry.csv"
)

STEP16_REFERENCE_REGISTER = (
    cfg.AUDIT_DIR / "16_stage1_reference_register.csv"
)

STEP10_TEST_SEAL = (
    cfg.MANIFEST_DIR / "10_final_test_seal.json"
)


# =============================================================================
# 3. SOURCE / OUTPUT PATHS
# =============================================================================

SOURCE_ROOT = (
    cfg.EXPERIMENT_ROOT_DIR
    / "third_party_sources"
    / "step17_official_model_sources"
)

REPOSITORY_AUDIT_CSV = (
    cfg.AUDIT_DIR / "17_official_repository_audit.csv"
)

SOURCE_FILE_AUDIT_CSV = (
    cfg.AUDIT_DIR / "17_official_source_file_audit.csv"
)

PYTHON_SYMBOL_AUDIT_CSV = (
    cfg.AUDIT_DIR / "17_python_source_symbol_audit.csv"
)

MODEL_SOURCE_MAP_CSV = (
    cfg.AUDIT_DIR / "17_model_to_official_source_map.csv"
)

SOURCE_HASH_CSV = (
    cfg.AUDIT_DIR / "17_source_file_hashes.csv"
)

SOURCE_FREEZE_JSON = (
    cfg.MANIFEST_DIR / "17_official_model_source_freeze.json"
)

MANIFEST_JSON = (
    cfg.MANIFEST_DIR / "17_official_model_source_manifest.json"
)

MANIFEST_SHA = (
    cfg.MANIFEST_DIR / "17_official_model_source_manifest.sha256"
)

HASH_REGISTER_CSV = (
    cfg.AUDIT_DIR / "17_official_model_source_hashes.csv"
)

FROZEN_SCRIPT = (
    cfg.MANIFEST_DIR / "step17_freeze_official_model_sources_frozen.py"
)

SUMMARY_TXT = (
    cfg.REPORT_DIR / "17_official_model_source_summary.txt"
)

FAILED_DIR = (
    cfg.AUDIT_DIR / "step17_failed_attempts"
)


# =============================================================================
# 4. UPSTREAM REPOSITORY SPECIFICATION
# =============================================================================

REPOSITORIES = [
    {
        "repo_id": "LTSF_LINEAR",
        "repository_role": "implementation",
        "repository_name": "cure-lab/LTSF-Linear",
        "url": "https://github.com/cure-lab/LTSF-Linear.git",
        "local_directory": "LTSF-Linear",
        "required_files": [
            "README.md",
            "models/DLinear.py",
            "models/NLinear.py",
        ],
        "primary_python_files": [
            "models/DLinear.py",
            "models/NLinear.py",
        ],
        "registered_models": [
            "DLINEAR",
            "NLINEAR",
        ],
    },
    {
        "repo_id": "PATCHTST",
        "repository_role": "implementation",
        "repository_name": "yuqinie98/PatchTST",
        "url": "https://github.com/yuqinie98/PatchTST.git",
        "local_directory": "PatchTST",
        "required_files": [
            "README.md",
            "PatchTST_supervised/models/PatchTST.py",
            "PatchTST_supervised/layers/PatchTST_backbone.py",
            "PatchTST_supervised/layers/PatchTST_layers.py",
            "PatchTST_supervised/layers/RevIN.py",
        ],
        "primary_python_files": [
            "PatchTST_supervised/models/PatchTST.py",
            "PatchTST_supervised/layers/PatchTST_backbone.py",
        ],
        "registered_models": [
            "PATCHTST",
        ],
    },
    {
        "repo_id": "TIMESNET_REFERENCE",
        "repository_role": "reference_redirect",
        "repository_name": "thuml/TimesNet",
        "url": "https://github.com/thuml/TimesNet.git",
        "local_directory": "TimesNet-reference",
        "required_files": [
            "README.md",
        ],
        "primary_python_files": [],
        "registered_models": [],
    },
    {
        "repo_id": "TIMESNET_TSLIB",
        "repository_role": "implementation",
        "repository_name": "thuml/Time-Series-Library",
        "url": "https://github.com/thuml/Time-Series-Library.git",
        "local_directory": "Time-Series-Library",
        "required_files": [
            "README.md",
            "models/TimesNet.py",
            "layers/Conv_Blocks.py",
            "layers/Embed.py",
            "scripts/classification/TimesNet.sh",
        ],
        "primary_python_files": [
            "models/TimesNet.py",
            "layers/Conv_Blocks.py",
            "layers/Embed.py",
        ],
        "registered_models": [
            "TIMESNET",
        ],
    },
    {
        "repo_id": "ITRANSFORMER",
        "repository_role": "implementation",
        "repository_name": "thuml/iTransformer",
        "url": "https://github.com/thuml/iTransformer.git",
        "local_directory": "iTransformer",
        "required_files": [
            "README.md",
            "model/iTransformer.py",
            "layers/Embed.py",
            "layers/Transformer_EncDec.py",
            "layers/SelfAttention_Family.py",
        ],
        "primary_python_files": [
            "model/iTransformer.py",
            "layers/Embed.py",
            "layers/Transformer_EncDec.py",
            "layers/SelfAttention_Family.py",
        ],
        "registered_models": [
            "ITRANSFORMER",
        ],
    },
    {
        "repo_id": "FREDFORMER",
        "repository_role": "implementation",
        "repository_name": "chenzRG/Fredformer",
        "url": "https://github.com/chenzRG/Fredformer.git",
        "local_directory": "Fredformer",
        "required_files": [
            "README.md",
            "models/Fredformer.py",
        ],
        "primary_python_files": [
            "models/Fredformer.py",
        ],
        "registered_models": [
            "FREDFORMER",
        ],
    },
    {
        "repo_id": "FRNET",
        "repository_role": "implementation",
        "repository_name": "SiriZhang45/FRNet",
        "url": "https://github.com/SiriZhang45/FRNet.git",
        "local_directory": "FRNet",
        "required_files": [
            "README.md",
            "models/FRNet.py",
        ],
        "primary_python_files": [
            "models/FRNet.py",
        ],
        "registered_models": [
            "FRNET",
        ],
    },
    {
        "repo_id": "MLF",
        "repository_role": "implementation",
        "repository_name": "Meteor-Stars/MLF",
        "url": "https://github.com/Meteor-Stars/MLF.git",
        "local_directory": "MLF",
        "required_files": [
            "README.md",
            "models/MLF.py",
        ],
        "primary_python_files": [
            "models/MLF.py",
        ],
        "registered_models": [
            "MLF",
        ],
    },
]


MODEL_SOURCE_MAP = [
    {
        "model_id": "DLINEAR",
        "reference_id": "LTSF_LINEAR_2023",
        "repo_id": "LTSF_LINEAR",
        "primary_source_file": "models/DLinear.py",
        "source_role": "upstream implementation source",
        "adapter_status_after_step17": "not_implemented",
    },
    {
        "model_id": "NLINEAR",
        "reference_id": "LTSF_LINEAR_2023",
        "repo_id": "LTSF_LINEAR",
        "primary_source_file": "models/NLinear.py",
        "source_role": "upstream implementation source",
        "adapter_status_after_step17": "not_implemented",
    },
    {
        "model_id": "PATCHTST",
        "reference_id": "PATCHTST_2023",
        "repo_id": "PATCHTST",
        "primary_source_file": (
            "PatchTST_supervised/layers/PatchTST_backbone.py"
        ),
        "source_role": "official supervised PatchTST backbone",
        "adapter_status_after_step17": "not_implemented",
    },
    {
        "model_id": "TIMESNET",
        "reference_id": "TIMESNET_2023",
        "repo_id": "TIMESNET_TSLIB",
        "primary_source_file": "models/TimesNet.py",
        "source_role": (
            "authors' maintained complete TimesNet implementation; "
            "the dedicated thuml/TimesNet reference repository is frozen separately"
        ),
        "adapter_status_after_step17": "not_implemented",
    },
    {
        "model_id": "ITRANSFORMER",
        "reference_id": "ITRANSFORMER_2024",
        "repo_id": "ITRANSFORMER",
        "primary_source_file": "model/iTransformer.py",
        "source_role": "official dedicated iTransformer implementation",
        "adapter_status_after_step17": "not_implemented",
    },
    {
        "model_id": "FREDFORMER",
        "reference_id": "FREDFORMER_2024",
        "repo_id": "FREDFORMER",
        "primary_source_file": "models/Fredformer.py",
        "source_role": "official KDD 2024 implementation source",
        "adapter_status_after_step17": "not_implemented",
    },
    {
        "model_id": "FRNET",
        "reference_id": "FRNET_2024",
        "repo_id": "FRNET",
        "primary_source_file": "models/FRNet.py",
        "source_role": "upstream FRNet implementation source",
        "adapter_status_after_step17": "not_implemented",
    },
    {
        "model_id": "MLF",
        "reference_id": "MLF_2025",
        "repo_id": "MLF",
        "primary_source_file": "models/MLF.py",
        "source_role": "upstream finance-specific MLF implementation source",
        "adapter_status_after_step17": "not_implemented",
    },
]


# =============================================================================
# 5. GENERIC HELPERS
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


def package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)

    except importlib_metadata.PackageNotFoundError:
        return None


def normalize_git_url(url: str) -> str:
    value = url.strip().rstrip("/")

    if value.endswith(".git"):
        value = value[:-4]

    return value.lower()


def git_executable() -> str:
    executable = shutil.which("git")

    if executable is None:
        raise RuntimeError(
            "git executable was not found on PATH."
        )

    return executable


def run_git(
    arguments: list[str],
    cwd: Optional[Path] = None,
    *,
    timeout_seconds: int = 600,
) -> str:
    command = [
        git_executable(),
        *arguments,
    ]

    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"

    result = subprocess.run(
        command,
        cwd=(
            str(cwd)
            if cwd is not None
            else None
        ),
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
            f"Command : {' '.join(command)}\n"
            f"CWD     : {cwd}\n"
            f"stdout  :\n{result.stdout}\n"
            f"stderr  :\n{result.stderr}"
        )

    return result.stdout.strip()


# =============================================================================
# 6. STEP 00 / ENVIRONMENT / UPSTREAM CHAIN
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
        "git_version": run_git(["--version"]),
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

            if path_text and Path(path_text).name == filename:
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

    for step in range(1, 17):
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

    for step in range(3, 17):
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

    if not STEP16_FREEZE.exists():
        raise RuntimeError(
            "Step-16 registry freeze is missing."
        )

    freeze16 = json.loads(
        STEP16_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    freeze_payload = dict(freeze16)

    stored_registry_fingerprint = (
        freeze_payload.pop(
            "registry_freeze_fingerprint_sha256",
            None,
        )
    )

    if not stored_registry_fingerprint:
        raise RuntimeError(
            "Step-16 registry freeze lacks its fingerprint."
        )

    recomputed_registry_fingerprint = (
        cfg.canonical_json_sha256(
            freeze_payload
        )
    )

    if (
        recomputed_registry_fingerprint
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
            "Step-16 freeze says training has started."
        )

    if freeze16.get("final_test_accessed") is not False:
        raise RuntimeError(
            "Step-16 freeze says final test was accessed."
        )

    frozen_architectures = tuple(
        freeze16
        .get("stage1_design", {})
        .get("architectures", [])
    )

    if frozen_architectures != STAGE1_MODEL_IDS:
        raise RuntimeError(
            "Step-16 architecture ordering differs from clean-V3 Stage-1 design."
        )

    # Verify the registry/reference CSVs as exact Step-16 artifacts.
    for path in (
        STEP16_MODEL_REGISTRY,
        STEP16_REFERENCE_REGISTER,
    ):
        record = find_artifact(
            step16_manifest,
            path.name,
        )

        if Path(record["path"]).resolve() != path.resolve():
            raise RuntimeError(
                f"Step-16 canonical artifact path mismatch: {path.name}"
            )

        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(
                f"Step-16 artifact SHA mismatch: {path.name}"
            )

    registry = pd.read_csv(
        STEP16_MODEL_REGISTRY
    )

    references = pd.read_csv(
        STEP16_REFERENCE_REGISTER
    )

    if tuple(
        registry["model_id"].astype(str)
    ) != STAGE1_MODEL_IDS:
        raise RuntimeError(
            "Step-16 registry CSV does not contain the exact eight Stage-1 IDs."
        )

    if len(registry) != EXPECTED_STAGE1_MODEL_COUNT:
        raise RuntimeError(
            "Unexpected Step-16 registry row count."
        )

    if registry["model_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate Step-16 model IDs."
        )

    if references["reference_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate Step-16 reference IDs."
        )

    reference_ids = set(
        references["reference_id"].astype(str)
    )

    if not set(
        registry["reference_id"].astype(str)
    ).issubset(
        reference_ids
    ):
        raise RuntimeError(
            "Step-16 registry references are not covered by reference register."
        )

    # Test seal metadata only.
    if not STEP10_TEST_SEAL.exists():
        raise RuntimeError(
            "Step-10 final-test seal metadata is missing."
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
        "step16_registry": registry,
        "step16_references": references,
        "test_seal_status": seal["status"],
        "sealed_test_sha256_from_metadata_only": (
            seal.get("sha256")
        ),
    }


# =============================================================================
# 7. SOURCE-SPEC / STEP-16 COVERAGE
# =============================================================================

def verify_source_specification(
    upstream: Dict[str, Any],
) -> None:
    if len(REPOSITORIES) != EXPECTED_CLONED_REPOSITORY_COUNT:
        raise RuntimeError(
            "Unexpected repository-specification count."
        )

    implementation_repositories = [
        row
        for row in REPOSITORIES
        if row["repository_role"] == "implementation"
    ]

    reference_repositories = [
        row
        for row in REPOSITORIES
        if row["repository_role"] == "reference_redirect"
    ]

    if (
        len(implementation_repositories)
        !=
        EXPECTED_IMPLEMENTATION_REPOSITORY_COUNT
    ):
        raise RuntimeError(
            "Unexpected implementation-repository count."
        )

    if (
        len(reference_repositories)
        !=
        EXPECTED_REFERENCE_REPOSITORY_COUNT
    ):
        raise RuntimeError(
            "Unexpected reference-repository count."
        )

    repo_ids = [
        row["repo_id"]
        for row in REPOSITORIES
    ]

    if len(set(repo_ids)) != len(repo_ids):
        raise RuntimeError(
            "Duplicate repository IDs."
        )

    mapped_models = [
        row["model_id"]
        for row in MODEL_SOURCE_MAP
    ]

    if tuple(mapped_models) != STAGE1_MODEL_IDS:
        raise RuntimeError(
            "Step-17 model-source map does not match Step-16 model order."
        )

    if len(MODEL_SOURCE_MAP) != EXPECTED_MODEL_SOURCE_MAP_COUNT:
        raise RuntimeError(
            "Unexpected model-source mapping count."
        )

    registry = upstream["step16_registry"]

    expected_reference_by_model = dict(
        zip(
            registry["model_id"].astype(str),
            registry["reference_id"].astype(str),
        )
    )

    for row in MODEL_SOURCE_MAP:
        model_id = row["model_id"]

        if (
            expected_reference_by_model[model_id]
            !=
            row["reference_id"]
        ):
            raise RuntimeError(
                f"{model_id}: Step-16/17 reference-ID mismatch."
            )

        repository = next(
            (
                spec
                for spec in REPOSITORIES
                if spec["repo_id"] == row["repo_id"]
            ),
            None,
        )

        if repository is None:
            raise RuntimeError(
                f"{model_id}: mapped repository does not exist."
            )

        if (
            model_id
            not in
            repository["registered_models"]
        ):
            raise RuntimeError(
                f"{model_id}: repository does not declare model coverage."
            )

        if (
            row["primary_source_file"]
            not in
            repository["required_files"]
        ):
            raise RuntimeError(
                f"{model_id}: primary source file is not a required frozen file."
            )

    # Explicitly reconcile the one deliberate source-location exception:
    # Step-16 cites the dedicated TimesNet repo; its maintained implementation
    # is frozen from Time-Series-Library.
    references = upstream["step16_references"].copy()

    timesnet_reference = references.loc[
        references["reference_id"].astype(str)
        ==
        "TIMESNET_2023"
    ]

    if len(timesnet_reference) != 1:
        raise RuntimeError(
            "Step-16 TimesNet reference record is not unique."
        )

    timesnet_url = str(
        timesnet_reference.iloc[0]["official_code"]
    )

    if (
        normalize_git_url(timesnet_url)
        !=
        normalize_git_url(
            "https://github.com/thuml/TimesNet.git"
        )
    ):
        raise RuntimeError(
            "Step-16 TimesNet reference URL drift."
        )


# =============================================================================
# 8. IMMUTABILITY / FAILURE HANDLING
# =============================================================================

def canonical_outputs() -> tuple[Path, ...]:
    return (
        REPOSITORY_AUDIT_CSV,
        SOURCE_FILE_AUDIT_CSV,
        PYTHON_SYMBOL_AUDIT_CSV,
        MODEL_SOURCE_MAP_CSV,
        SOURCE_HASH_CSV,
        SOURCE_FREEZE_JSON,
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
            "Step-17 canonical output(s) already exist. "
            "Refusing to overwrite.\n"
            +
            "\n".join(
                str(path)
                for path in collisions
            )
            +
            "\nUse --verify-only if Step 17 already passed."
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
# 9. REPOSITORY ACQUISITION / AUDIT
# =============================================================================

def clone_or_validate_repository(
    spec: Dict[str, Any],
) -> tuple[Path, str]:
    target = (
        SOURCE_ROOT
        /
        spec["local_directory"]
    )

    if target.exists():
        if not (
            target
            /
            ".git"
        ).exists():
            raise RuntimeError(
                "\nSOURCE DIRECTORY EXISTS BUT IS NOT A GIT REPOSITORY\n"
                f"{target}"
            )

        return (
            target,
            "existing_snapshot_no_fetch_or_pull",
        )

    temporary = (
        SOURCE_ROOT
        /
        f".clone_{spec['repo_id']}_{os.getpid()}"
    )

    if temporary.exists():
        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )

    print(
        f"       Cloning {spec['repository_name']} ..."
    )

    try:
        run_git(
            [
                "clone",
                "--depth",
                "1",
                "--no-tags",
                "--single-branch",
                spec["url"],
                str(temporary),
            ],
            timeout_seconds=1200,
        )

        if target.exists():
            raise RuntimeError(
                f"Repository target appeared during clone: {target}"
            )

        os.replace(
            temporary,
            target,
        )

    except Exception:
        if temporary.exists():
            shutil.rmtree(
                temporary,
                ignore_errors=True,
            )

        raise

    return (
        target,
        "cloned_this_run",
    )


def inspect_python_file(
    path: Path,
) -> Dict[str, Any]:
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    try:
        tree = ast.parse(text)
        parse_ok = True

        class_names = [
            node.name
            for node in tree.body
            if isinstance(
                node,
                ast.ClassDef,
            )
        ]

        function_names = [
            node.name
            for node in tree.body
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
        ]

    except SyntaxError:
        parse_ok = False
        class_names = []
        function_names = []

    return {
        "parse_ok": parse_ok,
        "class_names": class_names,
        "function_names": function_names,
        "line_count": len(
            text.splitlines()
        ),
    }


def audit_repository(
    spec: Dict[str, Any],
    repository_path: Path,
    acquisition_status: str,
) -> tuple[
    Dict[str, Any],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
]:
    origin = run_git(
        [
            "remote",
            "get-url",
            "origin",
        ],
        cwd=repository_path,
    )

    if (
        normalize_git_url(origin)
        !=
        normalize_git_url(
            spec["url"]
        )
    ):
        raise RuntimeError(
            "\nGIT ORIGIN MISMATCH\n"
            f"Repository: {spec['repo_id']}\n"
            f"Expected  : {spec['url']}\n"
            f"Observed  : {origin}"
        )

    commit_sha = run_git(
        [
            "rev-parse",
            "HEAD",
        ],
        cwd=repository_path,
    )

    tree_sha = run_git(
        [
            "rev-parse",
            "HEAD^{tree}",
        ],
        cwd=repository_path,
    )

    branch = run_git(
        [
            "branch",
            "--show-current",
        ],
        cwd=repository_path,
    )

    commit_date = run_git(
        [
            "show",
            "-s",
            "--format=%cI",
            "HEAD",
        ],
        cwd=repository_path,
    )

    commit_subject = run_git(
        [
            "show",
            "-s",
            "--format=%s",
            "HEAD",
        ],
        cwd=repository_path,
    )

    is_shallow = (
        run_git(
            [
                "rev-parse",
                "--is-shallow-repository",
            ],
            cwd=repository_path,
        )
        .strip()
        .lower()
        ==
        "true"
    )

    status = run_git(
        [
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=repository_path,
    )

    if status:
        raise RuntimeError(
            "\nTHIRD-PARTY SOURCE WORKTREE IS NOT CLEAN\n"
            f"Repository: {spec['repo_id']}\n"
            f"{status}"
        )

    tracked_files = run_git(
        [
            "ls-files",
        ],
        cwd=repository_path,
    ).splitlines()

    if not tracked_files:
        raise RuntimeError(
            f"{spec['repo_id']}: tracked-file inventory is empty."
        )

    file_rows: list[Dict[str, Any]] = []
    symbol_rows: list[Dict[str, Any]] = []

    for relative_path in spec["required_files"]:
        path = repository_path / relative_path

        if not path.exists() or not path.is_file():
            raise RuntimeError(
                "\nREQUIRED UPSTREAM SOURCE FILE MISSING\n"
                f"Repository: {spec['repo_id']}\n"
                f"Missing   : {relative_path}"
            )

        file_rows.append(
            {
                "repo_id": spec["repo_id"],
                "repository_name": spec["repository_name"],
                "repository_role": spec["repository_role"],
                "relative_path": relative_path,
                "required": True,
                "exists": True,
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(path),
            }
        )

    for relative_path in spec["primary_python_files"]:
        path = repository_path / relative_path

        source_info = inspect_python_file(
            path
        )

        if not source_info["parse_ok"]:
            raise RuntimeError(
                "\nPYTHON SOURCE AST PARSE FAILED\n"
                f"{path}"
            )

        symbol_rows.append(
            {
                "repo_id": spec["repo_id"],
                "relative_path": relative_path,
                "parse_ok": True,
                "line_count": int(
                    source_info["line_count"]
                ),
                "top_level_classes": ";".join(
                    source_info["class_names"]
                ),
                "top_level_functions": ";".join(
                    source_info["function_names"]
                ),
                "source_imported": False,
                "source_executed": False,
            }
        )

    license_candidates = sorted(
        {
            *repository_path.glob("LICENSE*"),
            *repository_path.glob("COPYING*"),
        }
    )

    license_files = [
        path.name
        for path in license_candidates
        if path.is_file()
    ]

    repository_record = {
        "repo_id": spec["repo_id"],
        "repository_role": spec["repository_role"],
        "repository_name": spec["repository_name"],
        "repository_url": spec["url"],
        "local_path": str(repository_path),
        "acquisition_status": acquisition_status,
        "origin_verified": True,
        "branch": branch,
        "commit_sha": commit_sha,
        "git_tree_sha": tree_sha,
        "commit_date": commit_date,
        "commit_subject": commit_subject,
        "shallow_repository": is_shallow,
        "tracked_file_count": int(
            len(tracked_files)
        ),
        "worktree_clean_including_untracked": True,
        "required_files_present": True,
        "primary_python_files": int(
            len(
                spec["primary_python_files"]
            )
        ),
        "license_files": ";".join(
            license_files
        ),
        "registered_models": ";".join(
            spec["registered_models"]
        ),
        "third_party_code_imported": False,
        "third_party_code_executed": False,
        "repository_fetched_or_pulled_by_step17": False,
    }

    return (
        repository_record,
        file_rows,
        symbol_rows,
    )


# =============================================================================
# 10. FREEZE PAYLOAD
# =============================================================================

def build_frozen_repository_payload(
    repository_audit: pd.DataFrame,
    source_file_audit: pd.DataFrame,
) -> list[Dict[str, Any]]:
    frozen_repositories = []

    for row in (
        repository_audit
        .sort_values("repo_id")
        .itertuples(index=False)
    ):
        repo_files = (
            source_file_audit.loc[
                source_file_audit["repo_id"]
                ==
                row.repo_id
            ]
            .sort_values("relative_path")
        )

        frozen_repositories.append(
            {
                "repo_id": row.repo_id,
                "repository_role": row.repository_role,
                "repository_name": row.repository_name,
                "repository_url": row.repository_url,
                "local_path": row.local_path,
                "branch": row.branch,
                "commit_sha": row.commit_sha,
                "git_tree_sha": row.git_tree_sha,
                "commit_date": row.commit_date,
                "commit_subject": row.commit_subject,
                "shallow_repository": bool(
                    row.shallow_repository
                ),
                "tracked_file_count": int(
                    row.tracked_file_count
                ),
                "license_files": row.license_files,
                "required_file_hashes": [
                    {
                        "relative_path": (
                            file_row.relative_path
                        ),
                        "size_bytes": int(
                            file_row.size_bytes
                        ),
                        "sha256": file_row.sha256,
                    }
                    for file_row
                    in repo_files.itertuples(
                        index=False
                    )
                ],
            }
        )

    return frozen_repositories


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
            "current_step17",
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
            "step16_reference_register",
            STEP16_REFERENCE_REGISTER.resolve(),
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
                "Final-test parquet entered Step-17 source list."
            )

        if not path.exists():
            raise RuntimeError(
                f"Required Step-17 source missing: {path}"
            )

        rows.append(
            {
                "role": role,
                "file_name": path.name,
                "absolute_path": str(path),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(path),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 12. VERIFY FROZEN REPOSITORIES
# =============================================================================

def verify_repository_against_freeze(
    frozen_repository: Dict[str, Any],
) -> None:
    repository_path = Path(
        frozen_repository["local_path"]
    )

    if not repository_path.exists():
        raise RuntimeError(
            f"Frozen repository missing: {repository_path}"
        )

    if not (
        repository_path
        /
        ".git"
    ).exists():
        raise RuntimeError(
            f"Frozen source is not a Git repository: {repository_path}"
        )

    observed_origin = run_git(
        [
            "remote",
            "get-url",
            "origin",
        ],
        cwd=repository_path,
    )

    if (
        normalize_git_url(observed_origin)
        !=
        normalize_git_url(
            frozen_repository["repository_url"]
        )
    ):
        raise RuntimeError(
            f"Frozen repository origin drift: {frozen_repository['repo_id']}"
        )

    observed_commit = run_git(
        [
            "rev-parse",
            "HEAD",
        ],
        cwd=repository_path,
    )

    observed_tree = run_git(
        [
            "rev-parse",
            "HEAD^{tree}",
        ],
        cwd=repository_path,
    )

    observed_branch = run_git(
        [
            "branch",
            "--show-current",
        ],
        cwd=repository_path,
    )

    if observed_commit != frozen_repository["commit_sha"]:
        raise RuntimeError(
            f"Frozen repository commit drift: {frozen_repository['repo_id']}"
        )

    if observed_tree != frozen_repository["git_tree_sha"]:
        raise RuntimeError(
            f"Frozen repository tree drift: {frozen_repository['repo_id']}"
        )

    if observed_branch != frozen_repository["branch"]:
        raise RuntimeError(
            f"Frozen repository branch drift: {frozen_repository['repo_id']}"
        )

    status = run_git(
        [
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=repository_path,
    )

    if status:
        raise RuntimeError(
            f"Frozen repository worktree is dirty: "
            f"{frozen_repository['repo_id']}\n{status}"
        )

    for file_record in frozen_repository[
        "required_file_hashes"
    ]:
        path = (
            repository_path
            /
            file_record["relative_path"]
        )

        if not path.exists() or not path.is_file():
            raise RuntimeError(
                f"Frozen required source file missing: {path}"
            )

        if sha256_file(path) != file_record["sha256"]:
            raise RuntimeError(
                f"Frozen source-file SHA drift: {path}"
            )


# =============================================================================
# 13. VERIFY EXISTING STEP 17
# =============================================================================

def verify_existing() -> int:
    print("=" * 136)
    print("STEP 17 — VERIFY EXISTING OFFICIAL MODEL-SOURCE FREEZE")
    print("=" * 136)

    step00 = verify_step00()
    upstream = verify_upstream(
        step00
    )

    manifest_sha = verify_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 17",
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
            "[FAIL] Stored Step-17 status is not PASS."
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

    for step in range(1, 17):
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
        print("STEP 17 VERIFICATION STATUS: FAIL")
        print("=" * 136)
        return 24

    freeze = json.loads(
        SOURCE_FREEZE_JSON.read_text(
            encoding="utf-8"
        )
    )

    stored_fingerprint = freeze.get(
        "source_freeze_fingerprint_sha256"
    )

    payload = dict(freeze)
    payload.pop(
        "source_freeze_fingerprint_sha256",
        None,
    )

    recomputed = cfg.canonical_json_sha256(
        payload
    )

    if stored_fingerprint != recomputed:
        print(
            "[FAIL] Source-freeze fingerprint mismatch."
        )
        return 25

    if (
        manifest.get(
            "source_freeze_fingerprint_sha256"
        )
        !=
        stored_fingerprint
    ):
        print(
            "[FAIL] Manifest/source-freeze fingerprint mismatch."
        )
        return 26

    for repository in freeze["repositories"]:
        verify_repository_against_freeze(
            repository
        )

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
        return 27

    print(
        "[PASS] Configuration fingerprint : "
        f"{step00['fingerprint']}"
    )
    print(
        "[PASS] Step-16 manifest          : "
        f"{upstream['manifest_sha'][16]}"
    )
    print(
        "[PASS] Step-16 registry freeze   : "
        f"{upstream['step16_registry_fingerprint_sha256']}"
    )
    print(
        "[PASS] Repositories frozen       : "
        f"{len(freeze['repositories'])}"
    )
    print(
        "[PASS] Model-source mappings     : "
        f"{len(freeze['model_source_map'])}"
    )
    print(
        "[PASS] Step-17 source freeze     : "
        f"{stored_fingerprint}"
    )
    print(
        "[PASS] Step-17 manifest          : "
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
    print("STEP 17 VERIFICATION STATUS: PASS")
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
            "[1/9] Verifying Step 00 and immutable Step 01 -> Step 16 chain..."
        )

        step00 = verify_step00()
        upstream = verify_upstream(
            step00
        )

        verify_source_specification(
            upstream
        )

        print(
            "       Configuration fingerprint: "
            f"{step00['fingerprint']}"
        )
        print(
            "       Step-16 manifest          : "
            f"{upstream['manifest_sha'][16]}"
        )
        print(
            "       Step-16 registry freeze   : "
            f"{upstream['step16_registry_fingerprint_sha256']}"
        )
        print(
            "       Stage-1 architectures     : "
            f"{EXPECTED_STAGE1_MODEL_COUNT}"
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
            "       Third-party code import   : PROHIBITED"
        )
        print(
            "       Model training            : PROHIBITED"
        )
        print(
            "       Final-test parquet access : PROHIBITED"
        )

        stage = "collision_guard"

        print(
            "[2/9] Checking immutable Step-17 output target..."
        )

        assert_fresh_target()
        cfg.create_directories()

        SOURCE_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        stage = "repository_acquisition"

        print(
            "[3/9] Acquiring/auditing upstream repository snapshots..."
        )

        repository_rows = []
        source_file_rows = []
        symbol_rows = []

        for spec in REPOSITORIES:
            repository_path, acquisition_status = (
                clone_or_validate_repository(
                    spec
                )
            )

            (
                repository_record,
                repository_files,
                repository_symbols,
            ) = audit_repository(
                spec,
                repository_path,
                acquisition_status,
            )

            repository_rows.append(
                repository_record
            )

            source_file_rows.extend(
                repository_files
            )

            symbol_rows.extend(
                repository_symbols
            )

            print(
                f"       {spec['repo_id']:<20} | "
                f"{acquisition_status:<34} | "
                f"commit={repository_record['commit_sha'][:12]} | "
                f"tree={repository_record['git_tree_sha'][:12]}"
            )

        repository_audit = pd.DataFrame(
            repository_rows
        )

        source_file_audit = pd.DataFrame(
            source_file_rows
        )

        symbol_audit = pd.DataFrame(
            symbol_rows
        )

        model_source_map = pd.DataFrame(
            MODEL_SOURCE_MAP
        )

        stage = "structural_validation"

        print(
            "[4/9] Validating repository, source-file and model-source coverage..."
        )

        if len(repository_audit) != EXPECTED_CLONED_REPOSITORY_COUNT:
            raise RuntimeError(
                "Repository audit count mismatch."
            )

        if int(
            (
                repository_audit["repository_role"]
                ==
                "implementation"
            ).sum()
        ) != EXPECTED_IMPLEMENTATION_REPOSITORY_COUNT:
            raise RuntimeError(
                "Implementation-repository audit count mismatch."
            )

        if int(
            (
                repository_audit["repository_role"]
                ==
                "reference_redirect"
            ).sum()
        ) != EXPECTED_REFERENCE_REPOSITORY_COUNT:
            raise RuntimeError(
                "Reference-repository audit count mismatch."
            )

        for column in (
            "origin_verified",
            "worktree_clean_including_untracked",
            "required_files_present",
        ):
            if not repository_audit[column].all():
                raise RuntimeError(
                    f"Repository audit failure: {column}"
                )

        if not source_file_audit["exists"].all():
            raise RuntimeError(
                "Required source-file audit contains missing files."
            )

        if not symbol_audit.empty:
            if not symbol_audit["parse_ok"].all():
                raise RuntimeError(
                    "One or more primary Python files failed AST parsing."
                )

        if len(model_source_map) != EXPECTED_MODEL_SOURCE_MAP_COUNT:
            raise RuntimeError(
                "Model-source mapping count mismatch."
            )

        if tuple(
            model_source_map["model_id"].astype(str)
        ) != STAGE1_MODEL_IDS:
            raise RuntimeError(
                "Model-source mapping order differs from Step 16."
            )

        print(
            "       Repositories audited       : "
            f"{len(repository_audit)}"
        )
        print(
            "       Implementation repositories: "
            f"{int((repository_audit['repository_role'] == 'implementation').sum())}"
        )
        print(
            "       Reference repositories     : "
            f"{int((repository_audit['repository_role'] == 'reference_redirect').sum())}"
        )
        print(
            "       Required files             : "
            f"{len(source_file_audit)}"
        )
        print(
            "       Python files AST-parsed    : "
            f"{len(symbol_audit)}"
        )
        print(
            "       Model-source mappings      : "
            f"{len(model_source_map)}"
        )

        stage = "write_audits"

        print(
            "[5/9] Writing source-provenance audits..."
        )

        write_csv(
            REPOSITORY_AUDIT_CSV,
            repository_audit,
        )

        write_csv(
            SOURCE_FILE_AUDIT_CSV,
            source_file_audit,
        )

        write_csv(
            PYTHON_SYMBOL_AUDIT_CSV,
            symbol_audit,
        )

        write_csv(
            MODEL_SOURCE_MAP_CSV,
            model_source_map,
        )

        write_csv(
            SOURCE_HASH_CSV,
            source_hashes(),
        )

        stage = "freeze"

        print(
            "[6/9] Freezing exact repository/source snapshots..."
        )

        if (
            sha256_file(
                Path(step00["lock_path"])
            )
            !=
            step00["lock_sha256"]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step 17."
            )

        frozen_repositories = (
            build_frozen_repository_payload(
                repository_audit,
                source_file_audit,
            )
        )

        freeze_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "OFFICIAL_AUTHOR_MAINTAINED_MODEL_SOURCE_SNAPSHOTS_FROZEN"
            ),
            "generated_at_utc": utc_now(),
            "configuration_fingerprint_sha256": (
                step00["fingerprint"]
            ),
            "step16_manifest_sha256": (
                upstream["manifest_sha"][16]
            ),
            "step16_registry_freeze_fingerprint_sha256": (
                upstream[
                    "step16_registry_fingerprint_sha256"
                ]
            ),
            "source_policy": {
                "cloned_repository_count": (
                    EXPECTED_CLONED_REPOSITORY_COUNT
                ),
                "implementation_repository_count": (
                    EXPECTED_IMPLEMENTATION_REPOSITORY_COUNT
                ),
                "reference_repository_count": (
                    EXPECTED_REFERENCE_REPOSITORY_COUNT
                ),
                "mapped_stage1_models": (
                    EXPECTED_MODEL_SOURCE_MAP_COUNT
                ),
                "clone_policy": (
                    "shallow single-branch clone on first acquisition"
                ),
                "existing_snapshot_update_policy": (
                    "never fetch or pull"
                ),
                "worktree_policy": (
                    "must be clean including untracked files at freeze"
                ),
                "third_party_source_imported_in_step17": False,
                "third_party_source_executed_in_step17": False,
                "third_party_requirements_installed": False,
                "current_python_environment_modified": False,
                "timesnet_provenance": (
                    "thuml/TimesNet reference repository frozen; "
                    "complete maintained implementation frozen from "
                    "thuml/Time-Series-Library"
                ),
            },
            "repositories": frozen_repositories,
            "model_source_map": MODEL_SOURCE_MAP,
            "adapter_state": {
                "task_adapters_implemented": False,
                "task_adapter_contracts_frozen": False,
                "task_adapters_smoke_tested": False,
                "models_trained": False,
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
                "final_test_parquet_opened_or_hashed_by_step17": False,
                "final_test_performance_computed": False,
                "global_final_test_state": (
                    step00["state"]
                ),
            },
        }

        freeze_fingerprint = (
            cfg.canonical_json_sha256(
                freeze_payload
            )
        )

        freeze_payload[
            "source_freeze_fingerprint_sha256"
        ] = freeze_fingerprint

        atomic_json(
            SOURCE_FREEZE_JSON,
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
                        "Step-16 manifest SHA-256: "
                        f"{upstream['manifest_sha'][16]}"
                    ),
                    (
                        "Step-16 registry fingerprint: "
                        f"{upstream['step16_registry_fingerprint_sha256']}"
                    ),
                    (
                        "Repositories frozen: "
                        f"{EXPECTED_CLONED_REPOSITORY_COUNT}"
                    ),
                    (
                        "Implementation repositories: "
                        f"{EXPECTED_IMPLEMENTATION_REPOSITORY_COUNT}"
                    ),
                    (
                        "Reference repositories: "
                        f"{EXPECTED_REFERENCE_REPOSITORY_COUNT}"
                    ),
                    (
                        "Stage-1 model-source mappings: "
                        f"{EXPECTED_MODEL_SOURCE_MAP_COUNT}"
                    ),
                    "Third-party code imported: False",
                    "Third-party code executed: False",
                    "Repository requirements installed: False",
                    "Task adapters implemented: False",
                    "Models trained: False",
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
            REPOSITORY_AUDIT_CSV,
            SOURCE_FILE_AUDIT_CSV,
            PYTHON_SYMBOL_AUDIT_CSV,
            MODEL_SOURCE_MAP_CSV,
            SOURCE_HASH_CSV,
            SOURCE_FREEZE_JSON,
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
                for step in range(1, 17)
            },
            "step16_registry_freeze_fingerprint_sha256": (
                upstream[
                    "step16_registry_fingerprint_sha256"
                ]
            ),
            "source_freeze_fingerprint_sha256": (
                freeze_fingerprint
            ),
            "counts": {
                "cloned_repositories": (
                    EXPECTED_CLONED_REPOSITORY_COUNT
                ),
                "implementation_repositories": (
                    EXPECTED_IMPLEMENTATION_REPOSITORY_COUNT
                ),
                "reference_repositories": (
                    EXPECTED_REFERENCE_REPOSITORY_COUNT
                ),
                "mapped_stage1_models": (
                    EXPECTED_MODEL_SOURCE_MAP_COUNT
                ),
                "required_source_files": int(
                    len(source_file_audit)
                ),
                "primary_python_files_parsed": int(
                    len(symbol_audit)
                ),
            },
            "pipeline_state": {
                "development_arrays_frozen": True,
                "pre_model_integrity_complete": True,
                "stage1_model_registry_frozen": True,
                "official_model_sources_frozen": True,
                "task_adapter_contracts_frozen": False,
                "task_adapters_implemented": False,
                "task_adapters_smoke_tested": False,
                "models_fitted": False,
                "feature_selection_applied": False,
                "probability_threshold_selection_performed": False,
            },
            "test_protection": {
                "step10_seal_status": (
                    upstream["test_seal_status"]
                ),
                "global_final_test_state": (
                    step00["state"]
                ),
                "sealed_test_parquet_opened_or_hashed_by_step17": False,
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
            "[7/9] Validating frozen Step-17 artifacts and repositories..."
        )

        verify_sidecar(
            MANIFEST_JSON,
            MANIFEST_SHA,
            "Step 17",
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
                    f"Frozen Step-17 artifact missing: {path}"
                )

            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(
                    f"Frozen Step-17 artifact mismatch: {path}"
                )

        for repository in frozen_repositories:
            verify_repository_against_freeze(
                repository
            )

        if (
            sha256_file(
                Path(step00["lock_path"])
            )
            !=
            step00["lock_sha256"]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step-17 validation."
            )

        stage = "report"

        print(
            "[8/9] Building final source-provenance report..."
        )

        print()
        print("-" * 136)
        print("FROZEN UPSTREAM REPOSITORIES")
        print("-" * 136)

        for row in (
            repository_audit
            .sort_values(
                [
                    "repository_role",
                    "repo_id",
                ]
            )
            .itertuples(index=False)
        ):
            print(
                f"{row.repo_id:<20} | "
                f"{row.repository_role:<18} | "
                f"{row.acquisition_status:<34} | "
                f"commit={row.commit_sha[:12]} | "
                f"tree={row.git_tree_sha[:12]}"
            )

        print("-" * 136)
        print("MODEL -> SOURCE MAP")
        print("-" * 136)

        for row in model_source_map.itertuples(
            index=False
        ):
            print(
                f"{row.model_id:<16} | "
                f"{row.repo_id:<18} | "
                f"{row.primary_source_file}"
            )

        print("-" * 136)
        print("SOURCE INTEGRITY")
        print("-" * 136)
        print(
            "Repositories frozen            : "
            f"{len(repository_audit)}"
        )
        print(
            "Implementation repositories   : "
            f"{int((repository_audit['repository_role'] == 'implementation').sum())}"
        )
        print(
            "Reference repositories        : "
            f"{int((repository_audit['repository_role'] == 'reference_redirect').sum())}"
        )
        print(
            "Required files present        : "
            f"{bool(source_file_audit['exists'].all())}"
        )
        print(
            "Primary Python AST parses     : "
            f"{bool(symbol_audit['parse_ok'].all()) if not symbol_audit.empty else True}"
        )
        print(
            "Repository worktrees clean    : "
            f"{bool(repository_audit['worktree_clean_including_untracked'].all())}"
        )
        print(
            "Third-party code imported     : False"
        )
        print(
            "Third-party code executed     : False"
        )
        print(
            "Requirements installed        : False"
        )
        print(
            "Existing repositories updated : False"
        )

        print("-" * 136)
        print("PIPELINE STATE")
        print("-" * 136)
        print(
            "Development interface         : FROZEN"
        )
        print(
            "Stage-1 registry              : FROZEN"
        )
        print(
            "Upstream source snapshots     : FROZEN"
        )
        print(
            "Task-adapter contracts        : NOT YET FROZEN"
        )
        print(
            "Task adapters                 : NOT YET IMPLEMENTED"
        )
        print(
            "Adapter smoke tests           : NOT YET RUN"
        )
        print(
            "Models fitted                 : False"
        )
        print(
            "Final-test accessed           : False"
        )
        print(
            "Global final-test state       : "
            f"{step00['state']}"
        )

        print("-" * 136)
        print("OUTPUT FILES")
        print("-" * 136)
        print(
            f"Source root            : {SOURCE_ROOT}"
        )
        print(
            f"Repository audit       : {REPOSITORY_AUDIT_CSV}"
        )
        print(
            f"Source-file audit      : {SOURCE_FILE_AUDIT_CSV}"
        )
        print(
            f"Python-symbol audit    : {PYTHON_SYMBOL_AUDIT_CSV}"
        )
        print(
            f"Model-source map       : {MODEL_SOURCE_MAP_CSV}"
        )
        print(
            f"Source freeze          : {SOURCE_FREEZE_JSON}"
        )
        print(
            f"Source fingerprint     : {freeze_fingerprint}"
        )
        print(
            f"Manifest               : {MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256       : {manifest_sha}"
        )

        print(
            "[9/9] Step-17 source freeze complete."
        )
        print("=" * 136)
        print(
            "PASS: The exact upstream source snapshots for all eight Stage-1 "
            "architectures were frozen before adapter implementation or training."
        )
        print(
            "PASS: TimesNet provenance explicitly freezes both the authors' "
            "reference repository and the maintained Time-Series-Library implementation."
        )
        print(
            "PASS: Repository origins, commits, Git trees and required-file "
            "SHA-256 hashes were recorded."
        )
        print(
            "PASS: Selected Python source files were AST-parsed without importing "
            "or executing third-party model code."
        )
        print(
            "PASS: No repository requirements were installed and no existing "
            "snapshot was fetched or pulled."
        )
        print(
            "PASS: No model was trained and the sealed final-test parquet remained unopened."
        )
        print("=" * 136)
        print(
            "Do not run the old Step-17 script. "
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
            f"17_model_source_freeze_FAILED_{stamp}.json"
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
                "source_root_preserved_for_retry": str(
                    SOURCE_ROOT
                ),
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
        print("STEP 17 FAILED — OFFICIAL MODEL SOURCES NOT FROZEN")
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
        print(
            f"Source root: {SOURCE_ROOT}"
        )

        if archived is not None:
            print(
                f"Archived partial Step-17 outputs: {archived}"
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
            "Clean-V3 official/author-maintained model-source snapshot freeze."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an existing frozen Step-17 source snapshot without fetching/pulling."
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
            "\nStep 17 interrupted by user."
        )
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 136)
        print("STEP 17 UNHANDLED FAILURE")
        print("=" * 136)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        traceback.print_exc()
        print("=" * 136)
        sys.exit(1)
