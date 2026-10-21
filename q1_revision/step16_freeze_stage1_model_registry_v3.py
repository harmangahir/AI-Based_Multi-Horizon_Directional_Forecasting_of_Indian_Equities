#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 16 — CLEAN V3 STAGE-1 MODEL REGISTRY AND SELECTION-PROTOCOL FREEZE

Purpose
-------
Freeze the exact pretraining model-comparison protocol BEFORE the first
candidate model is fitted.

The clean-V3 Stage-1 design is deliberately narrow and immutable:

    8 sequence architectures
    x 2 predeclared configurations (A/B)
    x 5 rolling-origin folds
    x screening seed 42
    = 80 Stage-1 screening fits

Architectures
-------------
1. DLinear
2. NLinear
3. PatchTST
4. TimesNet
5. iTransformer
6. Fredformer
7. FRNet
8. MLF

This stage also freezes:
- the shared neural optimization protocol;
- the A/B candidate hyperparameters;
- five-fold configuration aggregation;
- the fixed 0.50 screening threshold;
- the complete tie-break hierarchy;
- the selected-config robustness policy using seeds 42..46;
- the development-OOF threshold grid 0.30..0.70 by 0.01;
- final-development refit policy;
- complexity-reporting requirements;
- statistical-inference policy;
- four prespecified downstream ablations.

Important
---------
The four ablations are frozen as a DOWNSTREAM plan only. They are NOT Stage-1
registry entries, do not increase the 16 Stage-1 configurations, and are not
trained here.

Likewise, horizon-specific classical modelling remains available from the
Step-13/15 contract but is NOT silently inserted into this locked Stage-1
neural architecture screen.

NO MODEL IS FITTED.
NO FEATURE SELECTION IS PERFORMED.
NO PROBABILITY THRESHOLD IS ESTIMATED.
THE FINAL-TEST PARQUET IS NOT OPENED OR HASHED.

Run
---
    python step16_freeze_stage1_model_registry_v3.py

Verify
------
    python step16_freeze_stage1_model_registry_v3.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import shutil
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
        "step16_freeze_stage1_model_registry_v3.py must be inside "
        "<repository>/q1_revision/."
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step00_config as cfg


# =============================================================================
# 1. FROZEN STAGE-1 DESIGN
# =============================================================================

STEP_NUMBER = 16
STEP_ID = "16"
STEP_NAME = "STEP 16 — STAGE-1 MODEL REGISTRY AND SELECTION-PROTOCOL FREEZE"
SCHEMA_VERSION = "1.0"

FOLDS = (1, 2, 3, 4, 5)
HORIZONS = (1, 3, 5)

SCREENING_SEED = 42
ROBUSTNESS_SEEDS = (42, 43, 44, 45, 46)

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

EXPECTED_STAGE1_MODELS = 8
CONFIGS_PER_MODEL = 2
EXPECTED_STAGE1_CONFIGS = 16
EXPECTED_STAGE1_SCREENING_FITS = 80

PRIMARY_METRIC = "balanced_accuracy"
SECONDARY_METRIC = "roc_auc"
TERTIARY_METRIC = "mcc"
QUATERNARY_METRIC = "pr_auc"
FIFTH_METRIC = "brier_score"

SCREENING_THRESHOLD = 0.50

OOF_THRESHOLD_GRID_START = 0.30
OOF_THRESHOLD_GRID_END = 0.70
OOF_THRESHOLD_GRID_STEP = 0.01

LOOKBACK = 60
TOTAL_FEATURE_COUNT = 63
STOCK_FEATURE_COUNT = 47
MARKET_FEATURE_COUNT = 16
DIRECTION_OUTPUTS = 3
AUXILIARY_OUTPUTS = 10

BATCH_SIZE = 256
MAX_EPOCHS = 100
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 12
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
AMP_ENABLED = False
DTYPE = "float32"
DIRECTION_LOSS = "BCEWithLogitsLoss(reduction='mean')"
CLASS_WEIGHTING = "none"

EXPECTED_ABLATIONS = 4


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
}

STEP15_FREEZE = (
    cfg.MANIFEST_DIR / "15_pre_model_freeze.json"
)

STEP15_MODEL_INPUT_CONTRACT = (
    cfg.AUDIT_DIR / "15_model_input_contract.csv"
)

STEP10_TEST_SEAL = (
    cfg.MANIFEST_DIR / "10_final_test_seal.json"
)


# =============================================================================
# 3. OUTPUT PATHS
# =============================================================================

MODEL_REGISTRY_CSV = (
    cfg.AUDIT_DIR / "16_stage1_model_registry.csv"
)

REFERENCE_REGISTER_CSV = (
    cfg.AUDIT_DIR / "16_stage1_reference_register.csv"
)

HYPERPARAMETER_JSON = (
    cfg.MANIFEST_DIR / "16_stage1_hyperparameter_candidates.json"
)

TRAINING_PROTOCOL_JSON = (
    cfg.MANIFEST_DIR / "16_stage1_training_protocol.json"
)

SELECTION_PROTOCOL_JSON = (
    cfg.MANIFEST_DIR / "16_stage1_selection_protocol.json"
)

ABLATION_PLAN_JSON = (
    cfg.MANIFEST_DIR / "16_prespecified_downstream_ablation_plan.json"
)

RUN_BUDGET_CSV = (
    cfg.AUDIT_DIR / "16_stage1_run_budget.csv"
)

DEPENDENCY_AUDIT_CSV = (
    cfg.AUDIT_DIR / "16_model_dependency_audit.csv"
)

SOURCE_HASH_CSV = (
    cfg.AUDIT_DIR / "16_source_file_hashes.csv"
)

REGISTRY_FREEZE_JSON = (
    cfg.MANIFEST_DIR / "16_stage1_model_registry_freeze.json"
)

MANIFEST_JSON = (
    cfg.MANIFEST_DIR / "16_stage1_model_registry_manifest.json"
)

MANIFEST_SHA = (
    cfg.MANIFEST_DIR / "16_stage1_model_registry_manifest.sha256"
)

HASH_REGISTER_CSV = (
    cfg.AUDIT_DIR / "16_stage1_model_registry_hashes.csv"
)

FROZEN_SCRIPT = (
    cfg.MANIFEST_DIR / "step16_freeze_stage1_model_registry_frozen.py"
)

SUMMARY_TXT = (
    cfg.REPORT_DIR / "16_stage1_model_registry_summary.txt"
)

FAILED_DIR = (
    cfg.AUDIT_DIR / "step16_failed_attempts"
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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    raise TypeError(type(value).__name__)


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
            default=json_default,
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

    os.replace(
        temporary,
        path,
    )


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


def relative_to_experiment(
    path: Path,
) -> str:
    return str(
        path.resolve().relative_to(
            cfg.EXPERIMENT_ROOT_DIR.resolve()
        )
    )


def artifact_record(
    path: Path,
) -> Dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": relative_to_experiment(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def package_version(
    package_name: str,
) -> Optional[str]:
    try:
        return importlib_metadata.version(package_name)

    except importlib_metadata.PackageNotFoundError:
        return None


# =============================================================================
# 5. STEP 00 / ENVIRONMENT
# =============================================================================

def verify_step00() -> Dict[str, Any]:
    cfg.validate_configuration()

    payload = cfg.build_configuration_payload()

    fingerprint = cfg.canonical_json_sha256(
        payload
    )

    manifest_path, manifest_sha, status = cfg.verify_existing_freeze(
        configuration_payload=payload,
        configuration_fingerprint=fingerprint,
    )

    if status != "VERIFIED_EXISTING":
        raise RuntimeError(
            "Step-00 freeze is not VERIFIED_EXISTING."
        )

    configured_lookback = int(
        getattr(
            cfg,
            "LOOKBACK",
            LOOKBACK,
        )
    )

    if configured_lookback != LOOKBACK:
        raise RuntimeError(
            f"Configured lookback is {configured_lookback}; expected {LOOKBACK}."
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

    drift = []

    for row in audit.get(
        "packages",
        [],
    ):
        if not row.get(
            "required",
            False,
        ):
            continue

        distribution = row.get(
            "distribution"
        )

        frozen_version = row.get(
            "version"
        )

        current_version = package_version(
            distribution
        )

        if current_version != frozen_version:
            drift.append(
                (
                    distribution,
                    frozen_version,
                    current_version,
                )
            )

    if drift:
        raise RuntimeError(
            "Required-package drift detected:\n"
            +
            "\n".join(
                f"{name}: frozen={frozen!r}, current={current!r}"
                for name, frozen, current in drift
            )
        )

    return {
        "python_executable": expected_python,
    }


# =============================================================================
# 6. UPSTREAM CHAIN / STEP 15 FREEZE
# =============================================================================

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

    observed = sha256_file(
        manifest_path
    )

    if observed != expected:
        raise RuntimeError(
            f"{label} manifest SHA mismatch.\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}"
        )

    return observed


def verify_upstream(
    step00: Dict[str, Any],
) -> Dict[str, Any]:
    shas: Dict[int, str] = {}
    manifests: Dict[int, Dict[str, Any]] = {}

    for step in range(
        1,
        16,
    ):
        manifest_path, sidecar_path = (
            UPSTREAM[
                step
            ]
        )

        shas[
            step
        ] = verify_sidecar(
            manifest_path,
            sidecar_path,
            f"Step {step:02d}",
        )

        manifests[
            step
        ] = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

    # Step 01.
    if manifests[
        1
    ].get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Step-01 status is not PASS."
        )

    if (
        manifests[
            1
        ].get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00[
            "fingerprint"
        ]
    ):
        raise RuntimeError(
            "Step-01 fingerprint mismatch."
        )

    # Step 02.
    if (
        manifests[
            2
        ]
        .get(
            "step",
            {},
        )
        .get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Step-02 status is not PASS."
        )

    # Step 03 -> 15.
    for step in range(
        3,
        16,
    ):
        manifest = manifests[
            step
        ]

        if (
            manifest
            .get(
                "step",
                {},
            )
            .get(
                "status"
            )
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
            step00[
                "fingerprint"
            ]
        ):
            raise RuntimeError(
                f"Step-{step:02d} fingerprint mismatch."
            )

        for previous in range(
            1,
            step,
        ):
            field = (
                f"step{previous:02d}_manifest_sha256"
            )

            if (
                field in manifest
                and
                manifest[
                    field
                ]
                !=
                shas[
                    previous
                ]
            ):
                raise RuntimeError(
                    f"Step-{step:02d} chain mismatch: {field}"
                )

    environment = verify_environment()

    step15 = manifests[
        15
    ]

    if not STEP15_FREEZE.exists():
        raise RuntimeError(
            "Step-15 pre-model freeze is missing."
        )

    freeze15 = json.loads(
        STEP15_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    freeze15_sha = sha256_file(
        STEP15_FREEZE
    )

    if (
        step15.get(
            "pre_model_freeze_sha256"
        )
        !=
        freeze15_sha
    ):
        raise RuntimeError(
            "Step-15 pre-model freeze SHA mismatch."
        )

    if (
        freeze15.get(
            "status"
        )
        !=
        "DEVELOPMENT_MODEL_INTERFACE_FROZEN_BEFORE_MODELLING"
    ):
        raise RuntimeError(
            "Step-15 development interface is not formally frozen."
        )

    if freeze15.get(
        "modelling_started"
    ) is not False:
        raise RuntimeError(
            "Step-15 freeze says modelling has already started."
        )

    if freeze15.get(
        "model_registry_frozen"
    ) is not False:
        raise RuntimeError(
            "Step-15 freeze unexpectedly says model registry is already frozen."
        )

    state15 = step15.get(
        "pipeline_state",
        {},
    )

    required_true = (
        "development_arrays_frozen",
        "final_development_arrays_frozen",
        "pre_model_integrity_complete",
    )

    for key in required_true:
        if state15.get(
            key
        ) is not True:
            raise RuntimeError(
                f"Step-15 pipeline state is incomplete: {key}"
            )

    required_false = (
        "model_registry_frozen",
        "feature_selection_applied",
        "models_fitted",
        "probability_threshold_selection_performed",
    )

    for key in required_false:
        if state15.get(
            key
        ) is not False:
            raise RuntimeError(
                f"Step-15 pipeline state violation: {key}"
            )

    # Confirm the Step-15 model-input contract has the neural/common boundary.
    if not STEP15_MODEL_INPUT_CONTRACT.exists():
        raise RuntimeError(
            "Step-15 model-input contract CSV is missing."
        )

    contract = pd.read_csv(
        STEP15_MODEL_INPUT_CONTRACT
    )

    if contract.empty:
        raise RuntimeError(
            "Step-15 model-input contract is empty."
        )

    required_contracts = {
        "common_multitask_neural",
        "horizon_specific_1d",
        "horizon_specific_3d",
        "horizon_specific_5d",
    }

    if not required_contracts.issubset(
        set(
            contract[
                "dataset_contract"
            ].astype(
                str
            )
        )
    ):
        raise RuntimeError(
            "Step-15 model-input contract is incomplete."
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

    if seal.get(
        "status"
    ) != "SEALED_NOT_OPENED":
        raise RuntimeError(
            "Step-10 final-test seal is not SEALED_NOT_OPENED."
        )

    if (
        seal.get(
            "global_final_test_lock_state"
        )
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
        "step15_freeze_sha256": freeze15_sha,
        "step15_freeze": freeze15,
        "test_seal_status": seal[
            "status"
        ],
        "sealed_test_sha256_from_metadata_only": (
            seal.get(
                "sha256"
            )
        ),
    }


# =============================================================================
# 7. IMMUTABILITY / FAILURE HANDLING
# =============================================================================

def canonical_outputs() -> tuple[Path, ...]:
    return (
        MODEL_REGISTRY_CSV,
        REFERENCE_REGISTER_CSV,
        HYPERPARAMETER_JSON,
        TRAINING_PROTOCOL_JSON,
        SELECTION_PROTOCOL_JSON,
        ABLATION_PLAN_JSON,
        RUN_BUDGET_CSV,
        DEPENDENCY_AUDIT_CSV,
        SOURCE_HASH_CSV,
        REGISTRY_FREEZE_JSON,
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
            "Step-16 canonical output(s) already exist. "
            "Refusing to overwrite.\n"
            +
            "\n".join(
                str(
                    path
                )
                for path in collisions
            )
            +
            "\nUse --verify-only if Step 16 already passed."
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
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
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
                str(
                    path
                ),
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
# 8. DEPENDENCY AUDIT
# =============================================================================

def build_dependency_audit() -> pd.DataFrame:
    dependencies = (
        ("numpy", True),
        ("pandas", True),
        ("scikit-learn", True),
        ("torch", True),
        ("scipy", True),
        ("joblib", True),
    )

    rows = []

    for package, required in dependencies:
        version = package_version(
            package
        )

        installed = (
            version
            is not None
        )

        rows.append(
            {
                "package": package,
                "required_for_stage1": required,
                "installed": installed,
                "version": version,
            }
        )

        if required and not installed:
            raise RuntimeError(
                f"Required Stage-1 package is missing: {package}"
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 9. REFERENCE REGISTER
# =============================================================================

def build_reference_register() -> list[dict]:
    """
    Bibliographic metadata is carried forward from the uploaded legacy Step-16
    registry. Step 17 will independently audit official-source implementations
    and task-adapter mappings before training.
    """

    return [
        {
            "reference_id": "LTSF_LINEAR_2023",
            "models": "DLinear; NLinear",
            "title": "Are Transformers Effective for Time Series Forecasting?",
            "venue": "AAAI Conference on Artificial Intelligence",
            "year": 2023,
            "doi": "10.1609/aaai.v37i9.26317",
            "official_code": "https://github.com/cure-lab/LTSF-Linear",
            "step17_source_mapping_required": True,
        },
        {
            "reference_id": "PATCHTST_2023",
            "models": "PatchTST",
            "title": "A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers",
            "venue": "ICLR",
            "year": 2023,
            "doi": "",
            "official_code": "https://github.com/yuqinie98/PatchTST",
            "step17_source_mapping_required": True,
        },
        {
            "reference_id": "TIMESNET_2023",
            "models": "TimesNet",
            "title": "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis",
            "venue": "ICLR",
            "year": 2023,
            "doi": "",
            "official_code": "https://github.com/thuml/TimesNet",
            "step17_source_mapping_required": True,
        },
        {
            "reference_id": "ITRANSFORMER_2024",
            "models": "iTransformer",
            "title": "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
            "venue": "ICLR",
            "year": 2024,
            "doi": "",
            "official_code": "https://github.com/thuml/iTransformer",
            "step17_source_mapping_required": True,
        },
        {
            "reference_id": "FREDFORMER_2024",
            "models": "Fredformer",
            "title": "Fredformer: Frequency Debiased Transformer for Time Series Forecasting",
            "venue": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
            "year": 2024,
            "doi": "10.1145/3637528.3671928",
            "official_code": "https://github.com/chenzRG/Fredformer",
            "step17_source_mapping_required": True,
        },
        {
            "reference_id": "FRNET_2024",
            "models": "FRNet",
            "title": "FRNet: Frequency-based Rotation Network for Long-term Time Series Forecasting",
            "venue": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
            "year": 2024,
            "doi": "10.1145/3637528.3671713",
            "official_code": "https://github.com/SiriZhang45/FRNet",
            "step17_source_mapping_required": True,
        },
        {
            "reference_id": "MLF_2025",
            "models": "MLF",
            "title": "Multi-period Learning for Financial Time Series Forecasting",
            "venue": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
            "year": 2025,
            "doi": "10.1145/3690624.3709422",
            "official_code": "https://github.com/Meteor-Stars/MLF",
            "step17_source_mapping_required": True,
        },
    ]


# =============================================================================
# 10. STAGE-1 MODEL REGISTRY
# =============================================================================

def build_model_registry() -> list[dict]:
    common = {
        "family": "modern_time_series",
        "stage": "stage1_architecture_screening",
        "role": "comparative_sequence_architecture",
        "input_view": "common_multitask_sequence_60x63",
        "target_view": "joint_direction_1d_3d_5d",
        "trainable": True,
        "stochastic": True,
        "screening_seed": SCREENING_SEED,
        "robustness_seeds": list(
            ROBUSTNESS_SEEDS
        ),
        "candidate_count": CONFIGS_PER_MODEL,
        "uses_stock_id": False,
        "uses_regime_id": False,
        "uses_auxiliary_regression": False,
        "feature_selection_in_stage1": False,
    }

    return [
        {
            **common,
            "model_id": "DLINEAR",
            "display_name": "DLinear Adapter",
            "reference_id": "LTSF_LINEAR_2023",
            "adaptation": (
                "Published DLinear decomposition backbone; continuous forecast "
                "head replaced by a three-logit directional head."
            ),
        },
        {
            **common,
            "model_id": "NLINEAR",
            "display_name": "NLinear Adapter",
            "reference_id": "LTSF_LINEAR_2023",
            "adaptation": (
                "Published NLinear normalization principle retained; forecast "
                "head replaced by a three-logit directional head."
            ),
        },
        {
            **common,
            "model_id": "PATCHTST",
            "display_name": "PatchTST Adapter",
            "reference_id": "PATCHTST_2023",
            "adaptation": (
                "Channel-independent patching/Transformer backbone retained; "
                "forecast head replaced by a three-logit directional head."
            ),
        },
        {
            **common,
            "model_id": "TIMESNET",
            "display_name": "TimesNet Adapter",
            "reference_id": "TIMESNET_2023",
            "adaptation": (
                "Temporal 2D-variation backbone retained; output adapted to "
                "joint 1d/3d/5d directional logits."
            ),
        },
        {
            **common,
            "model_id": "ITRANSFORMER",
            "display_name": "iTransformer Adapter",
            "reference_id": "ITRANSFORMER_2024",
            "adaptation": (
                "Inverted variate-token Transformer retained; forecast projection "
                "replaced by joint three-horizon directional classification."
            ),
        },
        {
            **common,
            "model_id": "FREDFORMER",
            "display_name": "Fredformer Adapter",
            "reference_id": "FREDFORMER_2024",
            "adaptation": (
                "Frequency-debiased Transformer backbone retained; terminal "
                "forecasting head adapted to three directional logits."
            ),
        },
        {
            **common,
            "model_id": "FRNET",
            "display_name": "FRNet Adapter",
            "reference_id": "FRNET_2024",
            "adaptation": (
                "Frequency-rotation backbone retained; forecasting output "
                "adapted to joint three-horizon direction."
            ),
        },
        {
            **common,
            "model_id": "MLF",
            "display_name": "Multi-period Learning Framework Adapter",
            "reference_id": "MLF_2025",
            "adaptation": (
                "Multi-period mechanism retained; the 60-session history is "
                "exposed as nested 20/40/60-session views and the output is "
                "adapted to joint three-horizon direction."
            ),
        },
    ]


# =============================================================================
# 11. HYPERPARAMETER CANDIDATES
# =============================================================================

def build_hyperparameter_candidates() -> Dict[str, list[dict]]:
    return {
        "DLINEAR": [
            {
                "config_id": "DLINEAR_A",
                "moving_average_kernel": 25,
                "individual": False,
                "head_hidden": 128,
                "dropout": 0.10,
                "learning_rate": 1e-3,
            },
            {
                "config_id": "DLINEAR_B",
                "moving_average_kernel": 25,
                "individual": True,
                "head_hidden": 128,
                "dropout": 0.10,
                "learning_rate": 3e-4,
            },
        ],
        "NLINEAR": [
            {
                "config_id": "NLINEAR_A",
                "individual": False,
                "head_hidden": 128,
                "dropout": 0.10,
                "learning_rate": 1e-3,
            },
            {
                "config_id": "NLINEAR_B",
                "individual": True,
                "head_hidden": 128,
                "dropout": 0.10,
                "learning_rate": 3e-4,
            },
        ],
        "PATCHTST": [
            {
                "config_id": "PATCHTST_A",
                "patch_length": 12,
                "stride": 6,
                "d_model": 128,
                "n_heads": 4,
                "encoder_layers": 2,
                "d_ff": 256,
                "dropout": 0.10,
                "learning_rate": 3e-4,
            },
            {
                "config_id": "PATCHTST_B",
                "patch_length": 16,
                "stride": 8,
                "d_model": 128,
                "n_heads": 8,
                "encoder_layers": 3,
                "d_ff": 512,
                "dropout": 0.10,
                "learning_rate": 3e-4,
            },
        ],
        "TIMESNET": [
            {
                "config_id": "TIMESNET_A",
                "top_k_periods": 3,
                "num_kernels": 4,
                "d_model": 64,
                "d_ff": 128,
                "encoder_layers": 2,
                "dropout": 0.10,
                "learning_rate": 3e-4,
            },
            {
                "config_id": "TIMESNET_B",
                "top_k_periods": 5,
                "num_kernels": 6,
                "d_model": 128,
                "d_ff": 256,
                "encoder_layers": 2,
                "dropout": 0.10,
                "learning_rate": 3e-4,
            },
        ],
        "ITRANSFORMER": [
            {
                "config_id": "ITRANSFORMER_A",
                "d_model": 128,
                "n_heads": 4,
                "encoder_layers": 2,
                "d_ff": 256,
                "dropout": 0.10,
                "learning_rate": 3e-4,
            },
            {
                "config_id": "ITRANSFORMER_B",
                "d_model": 256,
                "n_heads": 8,
                "encoder_layers": 3,
                "d_ff": 512,
                "dropout": 0.10,
                "learning_rate": 3e-4,
            },
        ],
        "FREDFORMER": [
            {
                "config_id": "FREDFORMER_A",
                "capacity": "small",
                "d_model": 128,
                "n_heads": 4,
                "encoder_layers": 2,
                "dropout": 0.10,
                "learning_rate": 3e-4,
                "official_source_mapping_required": True,
            },
            {
                "config_id": "FREDFORMER_B",
                "capacity": "medium",
                "d_model": 256,
                "n_heads": 8,
                "encoder_layers": 3,
                "dropout": 0.10,
                "learning_rate": 3e-4,
                "official_source_mapping_required": True,
            },
        ],
        "FRNET": [
            {
                "config_id": "FRNET_A",
                "capacity": "small",
                "hidden_dim": 128,
                "dropout": 0.10,
                "learning_rate": 3e-4,
                "official_source_mapping_required": True,
            },
            {
                "config_id": "FRNET_B",
                "capacity": "medium",
                "hidden_dim": 256,
                "dropout": 0.10,
                "learning_rate": 3e-4,
                "official_source_mapping_required": True,
            },
        ],
        "MLF": [
            {
                "config_id": "MLF_A",
                "period_lengths": [20, 40, 60],
                "d_model": 128,
                "n_heads": 4,
                "encoder_layers": 2,
                "dropout": 0.10,
                "learning_rate": 3e-4,
                "official_source_mapping_required": True,
            },
            {
                "config_id": "MLF_B",
                "period_lengths": [20, 40, 60],
                "d_model": 256,
                "n_heads": 8,
                "encoder_layers": 3,
                "dropout": 0.10,
                "learning_rate": 3e-4,
                "official_source_mapping_required": True,
            },
        ],
    }


# =============================================================================
# 12. SHARED TRAINING PROTOCOL
# =============================================================================

def build_training_protocol() -> Dict[str, Any]:
    return {
        "stage": "stage1_architecture_screening",
        "data_contract": {
            "source": "Step-14 common-multitask arrays",
            "folds": list(
                FOLDS
            ),
            "lookback_sessions": LOOKBACK,
            "continuous_predictors": TOTAL_FEATURE_COUNT,
            "stock_predictors": STOCK_FEATURE_COUNT,
            "market_predictors": MARKET_FEATURE_COUNT,
            "direction_outputs": DIRECTION_OUTPUTS,
            "uses_stock_id": False,
            "uses_regime_id": False,
            "uses_auxiliary_regression": False,
            "feature_selection": False,
            "class_weighting": CLASS_WEIGHTING,
        },
        "optimization": {
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "minimum_epoch_before_stopping": MIN_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "direction_loss": DIRECTION_LOSS,
            "direction_loss_horizon_aggregation": (
                "unweighted arithmetic mean across 1d, 3d and 5d BCE"
            ),
            "mixed_precision_amp": AMP_ENABLED,
            "numeric_dtype": DTYPE,
            "checkpoint_metric": "validation_mean_directional_bce",
            "checkpoint_mode": "min",
            "checkpoint_tie_break": (
                "earlier epoch if validation_mean_directional_bce is exactly tied"
            ),
        },
        "screening": {
            "seed": SCREENING_SEED,
            "probability_threshold": SCREENING_THRESHOLD,
            "threshold_tuned_during_stage1": False,
            "fits_per_configuration": len(
                FOLDS
            ),
            "validation_fold_predictions_retained": True,
            "validation_metadata_retained": True,
        },
        "reproducibility": {
            "best_seed_selection": False,
            "screening_seed_is_not_a_promoted_best_seed": True,
            "all_random_generators_seeded_from_run_seed": True,
            "deterministic_algorithms_required_where_supported": True,
            "training_shuffle": True,
            "validation_shuffle": False,
        },
    }


# =============================================================================
# 13. SELECTION / ROBUSTNESS / OOF PROTOCOL
# =============================================================================

def threshold_grid() -> list[float]:
    values = []

    value = OOF_THRESHOLD_GRID_START

    while (
        value
        <=
        OOF_THRESHOLD_GRID_END
        +
        1e-12
    ):
        values.append(
            round(
                value,
                2,
            )
        )

        value += OOF_THRESHOLD_GRID_STEP

    if len(
        values
    ) != 41:
        raise RuntimeError(
            "OOF threshold grid must contain 41 values."
        )

    if values[
        0
    ] != 0.30 or values[
        -1
    ] != 0.70:
        raise RuntimeError(
            "OOF threshold grid endpoints are invalid."
        )

    return values


def build_selection_protocol() -> Dict[str, Any]:
    return {
        "stage1_configuration_selection": {
            "selection_scope": (
                "select exactly one A/B configuration independently within each "
                "of the eight registered architectures"
            ),
            "screening_seed": SCREENING_SEED,
            "screening_threshold": SCREENING_THRESHOLD,
            "folds": list(
                FOLDS
            ),
            "primary_score": (
                "unweighted mean across five validation folds of the "
                "macro-horizon balanced accuracy, where each fold's macro score "
                "is the arithmetic mean of 1d/3d/5d balanced accuracy at threshold 0.50"
            ),
            "tie_break_order": [
                "higher unweighted five-fold mean macro-horizon ROC-AUC",
                "higher unweighted five-fold mean macro-horizon MCC",
                "higher unweighted five-fold mean macro-horizon PR-AUC",
                "lower unweighted five-fold mean macro-horizon Brier score",
                "lower trainable parameter count",
                "lexicographically smaller config_id",
            ],
            "no_fold_weighting_by_sample_count": True,
            "no_horizon_weighting": True,
            "best_seed_selection": False,
            "test_used": False,
        },
        "stage2_selected_config_seed_robustness": {
            "selected_config_only": True,
            "architectures": list(
                STAGE1_MODEL_IDS
            ),
            "seeds": list(
                ROBUSTNESS_SEEDS
            ),
            "folds": list(
                FOLDS
            ),
            "best_seed_promotion": False,
            "report": (
                "retain fold-wise and seed-wise metrics; report arithmetic "
                "mean and standard deviation across the five fixed seeds"
            ),
            "oof_probability_per_endpoint": (
                "arithmetic mean of probabilities from seeds 42,43,44,45,46 "
                "for the same architecture, fold and endpoint"
            ),
            "screening_seed42_artifact_reuse_allowed": (
                "yes, but only for the selected configuration if the exact "
                "training-protocol/config/data hashes match; reuse is computational, "
                "not statistical selection"
            ),
            "test_used": False,
        },
        "stage3_development_oof_threshold_freeze": {
            "data_source": (
                "pooled rolling-origin validation predictions from development only"
            ),
            "input_probability": (
                "fixed-five-seed mean probability for each validation endpoint"
            ),
            "threshold_grid": threshold_grid(),
            "criterion": (
                "maximum pooled development OOF balanced accuracy"
            ),
            "separate_threshold_per_architecture_and_horizon": True,
            "tie_break_order": [
                "threshold closest to 0.50",
                "smaller threshold if still tied",
            ],
            "test_used": False,
        },
        "final_development_refit": {
            "source": (
                "Step-14 final-development common-multitask arrays"
            ),
            "configuration": (
                "architecture-specific A/B configuration frozen from Stage 1"
            ),
            "seeds": list(
                ROBUSTNESS_SEEDS
            ),
            "probability_ensemble": (
                "arithmetic mean across the five fixed seed models"
            ),
            "threshold": (
                "architecture-and-horizon threshold frozen from development OOF; "
                "no final-test threshold tuning"
            ),
            "hyperparameter_changes_after_oof": False,
        },
        "final_test_policy": {
            "access": (
                "one-time only after Stage-1 config choices, robustness seeds, "
                "OOF thresholds, downstream ablation definitions, implementation "
                "mappings and all final-development refits are frozen"
            ),
            "post_test_model_promotion": False,
            "post_test_hyperparameter_changes": False,
            "post_test_threshold_changes": False,
            "post_test_feature_changes": False,
            "post_test_seed_selection": False,
            "test_class_balance_used_during_development": False,
        },
        "reporting": {
            "primary_metrics": [
                PRIMARY_METRIC,
                SECONDARY_METRIC,
                TERTIARY_METRIC,
                QUATERNARY_METRIC,
            ],
            "calibration_metrics": [
                FIFTH_METRIC,
                "ece",
            ],
            "complexity_metrics": [
                "trainable_parameters",
                "training_seconds",
                "inference_seconds",
                "peak_gpu_memory_mb",
            ],
        },
        "statistical_inference": {
            "resampling_unit": "market_session_cluster",
            "bootstrap": "circular_moving_block_bootstrap",
            "replicates": 2000,
            "primary_block_length_sessions": 20,
            "sensitivity_block_lengths_sessions": [
                10,
                20,
                40,
            ],
            "multiple_testing": "Benjamini-Hochberg FDR",
            "test_information_used_to_choose_inference_settings": False,
        },
    }


# =============================================================================
# 14. PRESPECIFIED DOWNSTREAM ABLATION PLAN
# =============================================================================

def build_ablation_plan() -> Dict[str, Any]:
    ablations = [
        {
            "ablation_id": "NO_REGIME_FILM",
            "display_name": "No Regime FiLM",
            "change": (
                "disable train-defined regime FiLM conditioning only"
            ),
        },
        {
            "ablation_id": "NO_STOCK_EMBEDDING",
            "display_name": "No Stock Embedding",
            "change": (
                "disable stock embedding only"
            ),
        },
        {
            "ablation_id": "NO_STOCK_AND_REGIME_CONTEXT",
            "display_name": "No Stock + Regime Context",
            "change": (
                "disable stock embedding and regime conditioning"
            ),
        },
        {
            "ablation_id": "NO_AUXILIARY_REGRESSION",
            "display_name": "No Auxiliary Regression",
            "change": (
                "remove auxiliary regression head/loss while preserving the "
                "directional classification backbone"
            ),
        },
    ]

    return {
        "status": "PRESPECIFIED_BEFORE_ANY_TRAINING",
        "ablation_count": len(
            ablations
        ),
        "ablations": ablations,
        "stage1_registry_entries": False,
        "stage1_configuration_count_effect": 0,
        "training_now": False,
        "inheritance_policy": (
            "Ablations inherit the later-frozen selected context-enabled "
            "iTransformer/proposed configuration; only the named component is changed."
        ),
        "post_hoc_ablation_addition_after_test": False,
    }


# =============================================================================
# 15. VALIDATE REGISTRY
# =============================================================================

def validate_registry(
    registry: list[dict],
    candidates: Dict[str, list[dict]],
    references: list[dict],
    ablation_plan: Dict[str, Any],
) -> None:
    if len(
        registry
    ) != EXPECTED_STAGE1_MODELS:
        raise RuntimeError(
            f"Expected {EXPECTED_STAGE1_MODELS} Stage-1 models; "
            f"found {len(registry)}."
        )

    model_ids = [
        row[
            "model_id"
        ]
        for row in registry
    ]

    if tuple(
        model_ids
    ) != STAGE1_MODEL_IDS:
        raise RuntimeError(
            "Stage-1 model ordering/identity differs from frozen design."
        )

    if len(
        set(
            model_ids
        )
    ) != EXPECTED_STAGE1_MODELS:
        raise RuntimeError(
            "Duplicate Stage-1 model IDs."
        )

    reference_ids = {
        row[
            "reference_id"
        ]
        for row in references
    }

    total_configs = 0
    config_ids = []

    for row in registry:
        model_id = row[
            "model_id"
        ]

        if row.get(
            "candidate_count"
        ) != CONFIGS_PER_MODEL:
            raise RuntimeError(
                f"{model_id}: candidate_count must equal 2."
            )

        if row.get(
            "screening_seed"
        ) != SCREENING_SEED:
            raise RuntimeError(
                f"{model_id}: screening seed drift."
            )

        if tuple(
            row.get(
                "robustness_seeds",
                [],
            )
        ) != ROBUSTNESS_SEEDS:
            raise RuntimeError(
                f"{model_id}: robustness seed drift."
            )

        if row.get(
            "reference_id"
        ) not in reference_ids:
            raise RuntimeError(
                f"{model_id}: unknown reference ID."
            )

        if model_id not in candidates:
            raise RuntimeError(
                f"{model_id}: no hyperparameter candidates."
            )

        model_candidates = candidates[
            model_id
        ]

        if len(
            model_candidates
        ) != CONFIGS_PER_MODEL:
            raise RuntimeError(
                f"{model_id}: expected exactly two candidate configurations."
            )

        total_configs += len(
            model_candidates
        )

        local_ids = [
            candidate[
                "config_id"
            ]
            for candidate in model_candidates
        ]

        if len(
            set(
                local_ids
            )
        ) != CONFIGS_PER_MODEL:
            raise RuntimeError(
                f"{model_id}: duplicate config IDs."
            )

        config_ids.extend(
            local_ids
        )

    if total_configs != EXPECTED_STAGE1_CONFIGS:
        raise RuntimeError(
            f"Expected {EXPECTED_STAGE1_CONFIGS} total Stage-1 configs; "
            f"found {total_configs}."
        )

    if len(
        set(
            config_ids
        )
    ) != EXPECTED_STAGE1_CONFIGS:
        raise RuntimeError(
            "Stage-1 config IDs are not globally unique."
        )

    if ablation_plan.get(
        "ablation_count"
    ) != EXPECTED_ABLATIONS:
        raise RuntimeError(
            "Prespecified ablation count is not four."
        )

    if ablation_plan.get(
        "stage1_registry_entries"
    ) is not False:
        raise RuntimeError(
            "Ablations must not be counted as Stage-1 registry entries."
        )


# =============================================================================
# 16. RUN BUDGET
# =============================================================================

def build_run_budget(
    registry: list[dict],
) -> pd.DataFrame:
    rows = []

    for row in registry:
        candidate_configs = int(
            row[
                "candidate_count"
            ]
        )

        screening_fits = (
            candidate_configs
            *
            len(
                FOLDS
            )
        )

        robustness_logical_fits = (
            len(
                FOLDS
            )
            *
            len(
                ROBUSTNESS_SEEDS
            )
        )

        # One seed-42 fit per fold for the selected A/B config already exists
        # from screening and may be reused only after config selection if exact
        # protocol/data/config hashes match.
        seed42_reusable = len(
            FOLDS
        )

        incremental_robustness_fits = (
            robustness_logical_fits
            -
            seed42_reusable
        )

        final_development_refit_fits = len(
            ROBUSTNESS_SEEDS
        )

        rows.append(
            {
                "model_id": row[
                    "model_id"
                ],
                "candidate_configs": (
                    candidate_configs
                ),
                "folds": len(
                    FOLDS
                ),
                "stage1_screening_seed_count": 1,
                "stage1_screening_fits": (
                    screening_fits
                ),
                "selected_config_robustness_seed_count": len(
                    ROBUSTNESS_SEEDS
                ),
                "selected_config_robustness_logical_fits": (
                    robustness_logical_fits
                ),
                "screening_seed42_fits_reusable_for_robustness": (
                    seed42_reusable
                ),
                "incremental_robustness_fits_if_reused": (
                    incremental_robustness_fits
                ),
                "final_development_refit_seed_models": (
                    final_development_refit_fits
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if int(
        result[
            "stage1_screening_fits"
        ].sum()
    ) != EXPECTED_STAGE1_SCREENING_FITS:
        raise RuntimeError(
            "Stage-1 screening budget must equal 80 fits."
        )

    if int(
        result[
            "selected_config_robustness_logical_fits"
        ].sum()
    ) != 200:
        raise RuntimeError(
            "Selected-config robustness budget must equal 200 logical fits."
        )

    return result


# =============================================================================
# 17. SOURCE HASH REGISTER
# =============================================================================

def source_hashes() -> pd.DataFrame:
    sources = [
        (
            "current_step00",
            Path(
                cfg.__file__
            ).resolve(),
        ),
        (
            "current_step16",
            SCRIPT_FILE.resolve(),
        ),
        (
            "step15_pre_model_freeze",
            STEP15_FREEZE.resolve(),
        ),
        (
            "step15_model_input_contract",
            STEP15_MODEL_INPUT_CONTRACT.resolve(),
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
        if not path.exists():
            raise RuntimeError(
                f"Required Step-16 source missing: {path}"
            )

        # Reading/hash verification of the Step-10 seal JSON is allowed.
        # No final-test parquet is part of this source list.
        if path.name == "10_final_test_locked_panel.parquet":
            raise RuntimeError(
                "Final-test parquet entered Step-16 source list."
            )

        rows.append(
            {
                "role": role,
                "file_name": path.name,
                "absolute_path": str(
                    path
                ),
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
# 18. VERIFY EXISTING STEP 16
# =============================================================================

def verify_existing() -> int:
    print("=" * 136)
    print("STEP 16 — VERIFY EXISTING STAGE-1 REGISTRY FREEZE")
    print("=" * 136)

    step00 = verify_step00()

    upstream = verify_upstream(
        step00
    )

    manifest_sha = verify_sidecar(
        MANIFEST_JSON,
        MANIFEST_SHA,
        "Step 16",
    )

    manifest = json.loads(
        MANIFEST_JSON.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest
        .get(
            "step",
            {},
        )
        .get(
            "status"
        )
        !=
        "PASS"
    ):
        print(
            "[FAIL] Stored Step-16 status is not PASS."
        )
        return 20

    if (
        manifest.get(
            "configuration_fingerprint_sha256"
        )
        !=
        step00[
            "fingerprint"
        ]
    ):
        print(
            "[FAIL] Configuration fingerprint mismatch."
        )
        return 21

    for step in range(
        1,
        16,
    ):
        field = (
            f"step{step:02d}_manifest_sha256"
        )

        if (
            manifest.get(
                field
            )
            !=
            upstream[
                "manifest_sha"
            ][
                step
            ]
        ):
            print(
                f"[FAIL] Upstream chain mismatch: {field}"
            )
            return 22

    failures = []

    for name, record in manifest.get(
        "artifacts",
        {},
    ).items():
        path = Path(
            record[
                "path"
            ]
        )

        if not path.exists():
            failures.append(
                f"missing: {name}"
            )
            continue

        if sha256_file(
            path
        ) != record[
            "sha256"
        ]:
            failures.append(
                f"hash mismatch: {name}"
            )

    if failures:
        for failure in failures:
            print(
                f"[FAIL] {failure}"
            )

        print("=" * 136)
        print("STEP 16 VERIFICATION STATUS: FAIL")
        print("=" * 136)
        return 23

    freeze = json.loads(
        REGISTRY_FREEZE_JSON.read_text(
            encoding="utf-8"
        )
    )

    if (
        freeze.get(
            "status"
        )
        !=
        "STAGE1_MODEL_REGISTRY_AND_SELECTION_PROTOCOL_FROZEN"
    ):
        print(
            "[FAIL] Registry freeze status is invalid."
        )
        return 24

    if freeze.get(
        "training_started"
    ) is not False:
        print(
            "[FAIL] Freeze says training has started."
        )
        return 25

    if (
        sha256_file(
            Path(
                step00[
                    "lock_path"
                ]
            )
        )
        !=
        step00[
            "lock_sha256"
        ]
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
        "[PASS] Step-15 manifest          : "
        f"{upstream['manifest_sha'][15]}"
    )
    print(
        "[PASS] Step-15 pre-model freeze  : "
        f"{upstream['step15_freeze_sha256']}"
    )
    print(
        "[PASS] Stage-1 models/configs    : "
        f"{EXPECTED_STAGE1_MODELS} / {EXPECTED_STAGE1_CONFIGS}"
    )
    print(
        "[PASS] Stage-1 screening fits    : "
        f"{EXPECTED_STAGE1_SCREENING_FITS}"
    )
    print(
        "[PASS] Step-16 manifest          : "
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
    print("STEP 16 VERIFICATION STATUS: PASS")
    print("=" * 136)

    return 0


# =============================================================================
# 19. MAIN RUN
# =============================================================================

def run() -> int:
    print("=" * 136)
    print(STEP_NAME)
    print("=" * 136)

    stage = "verify"

    try:
        print(
            "[1/9] Verifying Step 00 and immutable Step 01 -> Step 15 chain..."
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
            "       Step-15 manifest          : "
            f"{upstream['manifest_sha'][15]}"
        )
        print(
            "       Step-15 pre-model freeze  : "
            f"{upstream['step15_freeze_sha256']}"
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
            "       Model training            : PROHIBITED"
        )
        print(
            "       Final-test parquet access : PROHIBITED"
        )

        stage = "collision_guard"

        print(
            "[2/9] Checking immutable Step-16 output target..."
        )

        assert_fresh_target()

        cfg.create_directories()

        stage = "registry_build"

        print(
            "[3/9] Building exact eight-architecture Stage-1 registry..."
        )

        dependency_audit = build_dependency_audit()
        references = build_reference_register()
        registry = build_model_registry()
        candidates = build_hyperparameter_candidates()
        training_protocol = build_training_protocol()
        selection_protocol = build_selection_protocol()
        ablation_plan = build_ablation_plan()

        validate_registry(
            registry,
            candidates,
            references,
            ablation_plan,
        )

        print(
            "       Registered architectures : "
            f"{len(registry)}"
        )
        print(
            "       Configurations/model      : "
            f"{CONFIGS_PER_MODEL}"
        )
        print(
            "       Total Stage-1 configs     : "
            f"{sum(len(v) for v in candidates.values())}"
        )
        print(
            "       Prespecified ablations    : "
            f"{ablation_plan['ablation_count']} (downstream; not Stage 1)"
        )

        stage = "protocol_validation"

        print(
            "[4/9] Validating screening, robustness and OOF-selection protocol..."
        )

        grid = selection_protocol[
            "stage3_development_oof_threshold_freeze"
        ][
            "threshold_grid"
        ]

        if grid != threshold_grid():
            raise RuntimeError(
                "Threshold grid is not deterministic."
            )

        if (
            training_protocol[
                "screening"
            ][
                "seed"
            ]
            !=
            SCREENING_SEED
        ):
            raise RuntimeError(
                "Screening seed drift."
            )

        if (
            training_protocol[
                "optimization"
            ][
                "batch_size"
            ]
            !=
            BATCH_SIZE
        ):
            raise RuntimeError(
                "Batch-size drift."
            )

        if (
            training_protocol[
                "optimization"
            ][
                "mixed_precision_amp"
            ]
            is not False
        ):
            raise RuntimeError(
                "AMP must remain disabled in frozen Stage-1 design."
            )

        print(
            "       Screening seed          : "
            f"{SCREENING_SEED}"
        )
        print(
            "       Robustness seeds        : "
            f"{list(ROBUSTNESS_SEEDS)}"
        )
        print(
            "       Screening threshold     : "
            f"{SCREENING_THRESHOLD:.2f}"
        )
        print(
            "       OOF threshold grid      : "
            f"{grid[0]:.2f}..{grid[-1]:.2f} by {OOF_THRESHOLD_GRID_STEP:.2f}"
        )
        print(
            "       Best-seed promotion     : PROHIBITED"
        )

        stage = "budget"

        print(
            "[5/9] Freezing computational run budget..."
        )

        run_budget = build_run_budget(
            registry
        )

        screening_fits = int(
            run_budget[
                "stage1_screening_fits"
            ].sum()
        )

        robustness_logical = int(
            run_budget[
                "selected_config_robustness_logical_fits"
            ].sum()
        )

        robustness_incremental = int(
            run_budget[
                "incremental_robustness_fits_if_reused"
            ].sum()
        )

        final_refits = int(
            run_budget[
                "final_development_refit_seed_models"
            ].sum()
        )

        print(
            "       Stage-1 screening fits        : "
            f"{screening_fits}"
        )
        print(
            "       Robustness logical fits       : "
            f"{robustness_logical}"
        )
        print(
            "       Robustness incremental if reuse: "
            f"{robustness_incremental}"
        )
        print(
            "       Final-development seed refits : "
            f"{final_refits}"
        )

        stage = "write"

        print(
            "[6/9] Writing registry, hyperparameters, protocols and provenance..."
        )

        registry_df = pd.DataFrame(
            registry
        )

        references_df = pd.DataFrame(
            references
        )

        write_csv(
            MODEL_REGISTRY_CSV,
            registry_df,
        )

        write_csv(
            REFERENCE_REGISTER_CSV,
            references_df,
        )

        write_csv(
            RUN_BUDGET_CSV,
            run_budget,
        )

        write_csv(
            DEPENDENCY_AUDIT_CSV,
            dependency_audit,
        )

        atomic_json(
            HYPERPARAMETER_JSON,
            {
                "schema_version": SCHEMA_VERSION,
                "stage": "stage1_architecture_screening",
                "model_candidates": candidates,
            },
        )

        atomic_json(
            TRAINING_PROTOCOL_JSON,
            training_protocol,
        )

        atomic_json(
            SELECTION_PROTOCOL_JSON,
            selection_protocol,
        )

        atomic_json(
            ABLATION_PLAN_JSON,
            ablation_plan,
        )

        write_csv(
            SOURCE_HASH_CSV,
            source_hashes(),
        )

        stage = "freeze"

        print(
            "[7/9] Freezing Stage-1 model registry before training..."
        )

        if (
            sha256_file(
                Path(
                    step00[
                        "lock_path"
                    ]
                )
            )
            !=
            step00[
                "lock_sha256"
            ]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step 16."
            )

        freeze_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "STAGE1_MODEL_REGISTRY_AND_SELECTION_PROTOCOL_FROZEN"
            ),
            "frozen_at_utc": utc_now(),
            "configuration_fingerprint_sha256": (
                step00[
                    "fingerprint"
                ]
            ),
            "step15_manifest_sha256": (
                upstream[
                    "manifest_sha"
                ][
                    15
                ]
            ),
            "step15_pre_model_freeze_sha256": (
                upstream[
                    "step15_freeze_sha256"
                ]
            ),
            "stage1_design": {
                "architectures": list(
                    STAGE1_MODEL_IDS
                ),
                "architecture_count": (
                    EXPECTED_STAGE1_MODELS
                ),
                "configs_per_architecture": (
                    CONFIGS_PER_MODEL
                ),
                "total_configurations": (
                    EXPECTED_STAGE1_CONFIGS
                ),
                "rolling_origin_folds": list(
                    FOLDS
                ),
                "screening_seed": (
                    SCREENING_SEED
                ),
                "screening_fits": (
                    EXPECTED_STAGE1_SCREENING_FITS
                ),
                "input": "Step-14 common-multitask 60x63 arrays",
                "feature_selection": False,
                "stock_id_used": False,
                "regime_id_used": False,
                "auxiliary_regression_used": False,
            },
            "training_protocol": (
                training_protocol
            ),
            "selection_protocol": (
                selection_protocol
            ),
            "ablation_plan": (
                ablation_plan
            ),
            "run_budget": {
                "stage1_screening_fits": (
                    screening_fits
                ),
                "selected_config_robustness_logical_fits": (
                    robustness_logical
                ),
                "incremental_robustness_fits_if_seed42_reused": (
                    robustness_incremental
                ),
                "final_development_refit_seed_models_if_all_8_refit": (
                    final_refits
                ),
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
                "final_test_parquet_opened_or_hashed_by_step16": False,
                "final_test_metrics_computed": False,
                "global_final_test_state": (
                    step00[
                        "state"
                    ]
                ),
            },
            "training_started": False,
            "feature_selection_applied": False,
            "threshold_estimation_performed": False,
            "final_test_accessed": False,
        }

        freeze_fingerprint = cfg.canonical_json_sha256(
            freeze_payload
        )

        freeze_payload[
            "registry_freeze_fingerprint_sha256"
        ] = freeze_fingerprint

        atomic_json(
            REGISTRY_FREEZE_JSON,
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
                        "Step-15 manifest SHA-256: "
                        f"{upstream['manifest_sha'][15]}"
                    ),
                    (
                        "Step-15 pre-model freeze SHA-256: "
                        f"{upstream['step15_freeze_sha256']}"
                    ),
                    (
                        "Stage-1 architectures/configurations: "
                        f"{EXPECTED_STAGE1_MODELS}/{EXPECTED_STAGE1_CONFIGS}"
                    ),
                    (
                        "Stage-1 screening fits: "
                        f"{EXPECTED_STAGE1_SCREENING_FITS}"
                    ),
                    (
                        "Robustness seeds: "
                        f"{list(ROBUSTNESS_SEEDS)}"
                    ),
                    "Best-seed selection: False",
                    "Screening threshold: 0.50",
                    "OOF threshold grid: 0.30..0.70 by 0.01",
                    "Models fitted: False",
                    "Feature selection: False",
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
            MODEL_REGISTRY_CSV,
            REFERENCE_REGISTER_CSV,
            HYPERPARAMETER_JSON,
            TRAINING_PROTOCOL_JSON,
            SELECTION_PROTOCOL_JSON,
            ABLATION_PLAN_JSON,
            RUN_BUDGET_CSV,
            DEPENDENCY_AUDIT_CSV,
            SOURCE_HASH_CSV,
            REGISTRY_FREEZE_JSON,
            FROZEN_SCRIPT,
            SUMMARY_TXT,
        )

        write_csv(
            HASH_REGISTER_CSV,
            pd.DataFrame(
                [
                    {
                        "file_name": path.name,
                        "file_path": str(
                            path
                        ),
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
                step00[
                    "fingerprint"
                ]
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
                for step in range(
                    1,
                    16,
                )
            },
            "step15_pre_model_freeze_sha256": (
                upstream[
                    "step15_freeze_sha256"
                ]
            ),
            "registry_freeze_fingerprint_sha256": (
                freeze_fingerprint
            ),
            "stage1_registry": {
                "models": (
                    EXPECTED_STAGE1_MODELS
                ),
                "configurations": (
                    EXPECTED_STAGE1_CONFIGS
                ),
                "screening_fits": (
                    EXPECTED_STAGE1_SCREENING_FITS
                ),
                "screening_seed": (
                    SCREENING_SEED
                ),
                "robustness_seeds": list(
                    ROBUSTNESS_SEEDS
                ),
            },
            "selection": {
                "primary_metric": PRIMARY_METRIC,
                "secondary_metric": SECONDARY_METRIC,
                "tertiary_metric": TERTIARY_METRIC,
                "quaternary_metric": QUATERNARY_METRIC,
                "fifth_metric": FIFTH_METRIC,
                "screening_threshold": (
                    SCREENING_THRESHOLD
                ),
                "best_seed_selection": False,
            },
            "pipeline_state": {
                "development_arrays_frozen": True,
                "pre_model_integrity_complete": True,
                "stage1_model_registry_frozen": True,
                "stage1_hyperparameter_candidates_frozen": True,
                "stage1_training_protocol_frozen": True,
                "stage1_selection_protocol_frozen": True,
                "downstream_ablation_plan_prespecified": True,
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
                    step00[
                        "state"
                    ]
                ),
                "sealed_test_parquet_opened_or_hashed_by_step16": False,
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
            "[8/9] Validating frozen Step-16 outputs..."
        )

        verify_sidecar(
            MANIFEST_JSON,
            MANIFEST_SHA,
            "Step 16",
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
                record[
                    "path"
                ]
            )

            if not path.exists():
                raise RuntimeError(
                    f"Frozen Step-16 artifact missing: {path}"
                )

            if sha256_file(
                path
            ) != record[
                "sha256"
            ]:
                raise RuntimeError(
                    f"Frozen Step-16 artifact mismatch: {path}"
                )

        if (
            sha256_file(
                Path(
                    step00[
                        "lock_path"
                    ]
                )
            )
            !=
            step00[
                "lock_sha256"
            ]
        ):
            raise RuntimeError(
                "Global final-test lock changed during Step-16 validation."
            )

        stage = "report"

        print(
            "[9/9] Final Step-16 report..."
        )

        print()
        print("-" * 136)
        print("STAGE-1 MODEL REGISTRY")
        print("-" * 136)

        for row in registry:
            print(
                f"{row['model_id']:<16} | "
                f"configs=2 | "
                f"reference={row['reference_id']}"
            )

        print("-" * 136)
        print("STAGE-1 CONFIGURATION SCREENING")
        print("-" * 136)
        print(
            "Architectures                 : "
            f"{EXPECTED_STAGE1_MODELS}"
        )
        print(
            "Configurations                : "
            f"{EXPECTED_STAGE1_CONFIGS}"
        )
        print(
            "Rolling-origin folds          : "
            f"{len(FOLDS)}"
        )
        print(
            "Screening seed                : "
            f"{SCREENING_SEED}"
        )
        print(
            "Planned screening fits        : "
            f"{screening_fits}"
        )
        print(
            "Screening threshold           : "
            f"{SCREENING_THRESHOLD:.2f}"
        )
        print(
            "Primary selection metric      : "
            "mean 5-fold macro-horizon balanced accuracy"
        )

        print("-" * 136)
        print("SHARED NEURAL TRAINING")
        print("-" * 136)
        print(
            "Optimizer                      : AdamW"
        )
        print(
            f"Weight decay                   : {WEIGHT_DECAY}"
        )
        print(
            f"Logical batch size             : {BATCH_SIZE}"
        )
        print(
            f"Max / min epochs               : {MAX_EPOCHS} / {MIN_EPOCHS}"
        )
        print(
            f"Early-stopping patience        : {EARLY_STOPPING_PATIENCE}"
        )
        print(
            "Checkpoint metric             : minimum validation mean directional BCE"
        )
        print(
            f"Gradient clip norm             : {GRADIENT_CLIP_NORM}"
        )
        print(
            f"Numeric dtype / AMP            : {DTYPE} / {AMP_ENABLED}"
        )
        print(
            f"Class weighting                : {CLASS_WEIGHTING}"
        )

        print("-" * 136)
        print("ROBUSTNESS / THRESHOLD POLICY")
        print("-" * 136)
        print(
            "Robustness seeds               : "
            f"{list(ROBUSTNESS_SEEDS)}"
        )
        print(
            "Selected-config robustness     : "
            f"{robustness_logical} logical fits"
        )
        print(
            "Best-seed selection            : PROHIBITED"
        )
        print(
            "OOF threshold grid             : 0.30 .. 0.70 by 0.01"
        )
        print(
            "Threshold source               : development rolling-origin OOF only"
        )
        print(
            "Final-test threshold tuning    : PROHIBITED"
        )

        print("-" * 136)
        print("DOWNSTREAM ABLATION PLAN")
        print("-" * 136)
        print(
            "Prespecified ablations         : "
            f"{EXPECTED_ABLATIONS}"
        )
        print(
            "Counted as Stage-1 models      : False"
        )
        print(
            "Trained in Step 16             : False"
        )

        print("-" * 136)
        print("FREEZE STATE")
        print("-" * 136)
        print(
            "Development interface          : FROZEN"
        )
        print(
            "Stage-1 registry               : FROZEN"
        )
        print(
            "A/B hyperparameters            : FROZEN"
        )
        print(
            "Training protocol              : FROZEN"
        )
        print(
            "Selection/tie-break protocol   : FROZEN"
        )
        print(
            "Robustness seed policy         : FROZEN"
        )
        print(
            "OOF threshold rule             : FROZEN"
        )
        print(
            "Models fitted                  : False"
        )
        print(
            "Feature selection              : False"
        )
        print(
            "Final-test accessed            : False"
        )
        print(
            "Global final-test state        : "
            f"{step00['state']}"
        )

        print("-" * 136)
        print("OUTPUT FILES")
        print("-" * 136)
        print(
            f"Model registry          : {MODEL_REGISTRY_CSV}"
        )
        print(
            f"Reference register      : {REFERENCE_REGISTER_CSV}"
        )
        print(
            f"Hyperparameter registry : {HYPERPARAMETER_JSON}"
        )
        print(
            f"Training protocol       : {TRAINING_PROTOCOL_JSON}"
        )
        print(
            f"Selection protocol      : {SELECTION_PROTOCOL_JSON}"
        )
        print(
            f"Ablation plan           : {ABLATION_PLAN_JSON}"
        )
        print(
            f"Run budget              : {RUN_BUDGET_CSV}"
        )
        print(
            f"Registry freeze         : {REGISTRY_FREEZE_JSON}"
        )
        print(
            f"Registry fingerprint    : {freeze_fingerprint}"
        )
        print(
            f"Manifest                : {MANIFEST_JSON}"
        )
        print(
            f"Manifest SHA-256        : {manifest_sha}"
        )

        print("=" * 136)
        print(
            "PASS: Exactly eight Stage-1 sequence architectures and sixteen A/B "
            "configurations were frozen before training."
        )
        print(
            "PASS: Stage-1 screening is exactly 16 configurations x 5 folds "
            "at seed 42 = 80 fits."
        )
        print(
            "PASS: Shared optimizer, loss, early stopping, checkpointing, "
            "batch size, FP32/AMP and gradient-clipping rules are frozen."
        )
        print(
            "PASS: Five-seed selected-config robustness and development-only "
            "OOF threshold selection are frozen with no best-seed promotion."
        )
        print(
            "PASS: Four downstream ablations were prespecified without "
            "inflating the Stage-1 registry."
        )
        print(
            "PASS: No model, feature selector or probability threshold was fitted."
        )
        print(
            "PASS: The sealed final-test parquet was not opened or hashed."
        )
        print("=" * 136)
        print(
            "Do not run the old Step-16 script. "
            "Do not start training yet. Paste the complete terminal output for audit."
        )

        return 0

    except Exception as exc:
        archived = archive_partial()

        FAILED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        stamp = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y%m%dT%H%M%SZ"
            )
        )

        failure_path = (
            FAILED_DIR
            /
            f"16_stage1_registry_FAILED_{stamp}.json"
        )

        atomic_json(
            failure_path,
            {
                "step": STEP_NAME,
                "status": "FAIL",
                "failed_at_utc": utc_now(),
                "stage": stage,
                "error_type": type(
                    exc
                ).__name__,
                "error_message": str(
                    exc
                ),
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
        print("STEP 16 FAILED — STAGE-1 REGISTRY NOT FROZEN")
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
                f"Archived partial artifacts: {archived}"
            )

        print("=" * 136)
        print(
            "Do not start model training."
        )

        return 30


# =============================================================================
# 20. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean-V3 Stage-1 model registry and selection-protocol freeze."
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify an existing frozen Step-16 stage without recomputation."
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
            "\nStep 16 interrupted by user."
        )
        sys.exit(
            130
        )

    except Exception as exc:
        print()
        print("=" * 136)
        print("STEP 16 UNHANDLED FAILURE")
        print("=" * 136)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        traceback.print_exc()
        print("=" * 136)
        sys.exit(
            1
        )
