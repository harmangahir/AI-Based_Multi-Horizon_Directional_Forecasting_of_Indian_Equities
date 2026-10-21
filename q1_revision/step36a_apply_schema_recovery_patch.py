#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import py_compile
import re
import shutil
import sys

import pandas as pd

import step00_config as cfg

SOURCE = Path("step36_one_time_final_test_materialization_inference_scoring_v1.py").resolve()
PREACCESS_FREEZE = cfg.MANIFEST_DIR / "36_final_test_execution_preaccess_freeze_manifest.json"
PREACCESS_FREEZE_SHA = cfg.MANIFEST_DIR / "36_final_test_execution_preaccess_freeze_manifest.sha256"
ORIGINAL_FROZEN_SOURCE = cfg.MANIFEST_DIR / "step36_one_time_final_test_materialization_inference_scoring_v1_frozen.py"
ACCESS_MARKER = cfg.MANIFEST_DIR / "36_final_test_access_consumption_state.json"
MEMBERSHIP = cfg.FEATURE_DATA_DIR / "step10_chronological_splits" / "10_split_membership.parquet"
EXPECTED_MEMBERSHIP_SHA = "82772196c92e88f1926a75a70c42f5f19119d7245d93b8364455fd28642ee121"
EXPECTED_STEP35_FP = "44949863de7e82c3c303ca55beff8c59ae3206ea7eb542a9aab115ff88632bfc"
EXPECTED_STEP35_MANIFEST_SHA = "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7"
AMENDMENT = cfg.MANIFEST_DIR / "36a_schema_recovery_compatibility_amendment.json"
RECOVERY_FROZEN_SOURCE = cfg.MANIFEST_DIR / "step36_one_time_final_test_materialization_inference_scoring_v1_recovery_frozen.py"
BACKUP = Path("step36_one_time_final_test_materialization_inference_scoring_v1.pre_step36a_recovery.py").resolve()


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def canonical_hash(payload: dict) -> str:
    work = dict(payload)
    work.pop("recovery_amendment_fingerprint", None)
    return hashlib.sha256(
        json.dumps(work, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {n}")
    return text.replace(old, new, 1)


def regex_replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    new_text, n = re.subn(pattern, lambda m: replacement, text, flags=re.S)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {n}")
    return new_text


print("=" * 140)
print("STEP 36A — FREEZE SCHEMA-ONLY TECHNICAL RECOVERY AMENDMENT")
print("=" * 140)
print("Raw final-test parquet opened by this script : False")
print("Raw final-test parquet hashed by this script : False")
print("Final-test metrics computed                  : False")
print("Scientific model/seed/threshold changes      : False")
print("-" * 140)

for p in [SOURCE, PREACCESS_FREEZE, PREACCESS_FREEZE_SHA, ORIGINAL_FROZEN_SOURCE, ACCESS_MARKER, MEMBERSHIP]:
    if not p.exists():
        raise FileNotFoundError(f"Required recovery input missing:\n{p}")

if AMENDMENT.exists() or RECOVERY_FROZEN_SOURCE.exists():
    raise RuntimeError(
        "Step-36A recovery amendment/frozen source already exists. "
        "Do not overwrite an existing recovery freeze."
    )

pre = load_json(PREACCESS_FREEZE)
expected_sidecar = PREACCESS_FREEZE_SHA.read_text(encoding="utf-8").strip().split()[0]
if sha256_file(PREACCESS_FREEZE) != expected_sidecar:
    raise RuntimeError("Original Step-36 pre-access manifest SHA sidecar mismatch.")

original_source_sha = str(pre["source_sha256"])
current_source_sha = sha256_file(SOURCE)
if current_source_sha != original_source_sha:
    raise RuntimeError(
        "Current Step-36 source no longer matches the pre-access-frozen source.\n"
        f"Frozen : {original_source_sha}\nCurrent: {current_source_sha}"
    )
if sha256_file(ORIGINAL_FROZEN_SOURCE) != original_source_sha:
    raise RuntimeError("Original frozen Step-36 source SHA mismatch.")

marker = load_json(ACCESS_MARKER)
required_marker = {
    "state": "RAW_TEST_OPENED_ONCE_IN_MEMORY",
    "raw_test_filesystem_open_count_completed": 1,
    "final_test_scored": False,
    "step35_contract_fingerprint": EXPECTED_STEP35_FP,
    "step35_manifest_sha256": EXPECTED_STEP35_MANIFEST_SHA,
}
for key, expected in required_marker.items():
    observed = marker.get(key)
    if observed != expected:
        raise RuntimeError(
            f"Access-marker recovery precondition failed for {key}: "
            f"expected {expected!r}, observed {observed!r}"
        )
if marker.get("expected_raw_test_sha256") != marker.get("observed_raw_test_sha256"):
    raise RuntimeError("First-open test SHA did not match the pre-sealed SHA.")

membership_sha = sha256_file(MEMBERSHIP)
if membership_sha != EXPECTED_MEMBERSHIP_SHA:
    raise RuntimeError(
        "Step-10 structural membership SHA mismatch.\n"
        f"Expected: {EXPECTED_MEMBERSHIP_SHA}\nObserved: {membership_sha}"
    )

m = pd.read_parquet(MEMBERSHIP, engine="pyarrow")
required_membership = [
    "Ticker", "Date",
    "outer_final_test_eligible_1d",
    "outer_final_test_eligible_3d",
    "outer_final_test_eligible_5d",
    "outer_final_test_common_multitask",
]
missing = [c for c in required_membership if c not in m.columns]
if missing:
    raise RuntimeError(f"Membership ledger missing required columns: {missing}")
counts = {
    "outer_final_test_eligible_1d": 6640,
    "outer_final_test_eligible_3d": 6560,
    "outer_final_test_eligible_5d": 6480,
    "outer_final_test_common_multitask": 6480,
}
for c, expected in counts.items():
    observed = int(m[c].astype(bool).sum())
    if observed != expected:
        raise RuntimeError(f"{c} count drift: expected {expected}, observed {observed}")
if not (
    m["outer_final_test_eligible_5d"].astype(bool).to_numpy()
    == m["outer_final_test_common_multitask"].astype(bool).to_numpy()
).all():
    raise RuntimeError("Frozen final-test 5d/common membership mismatch.")

text = SOURCE.read_text(encoding="utf-8")
shutil.copy2(SOURCE, BACKUP)

# -----------------------------------------------------------------------------
# A. Add recovery constants.
# -----------------------------------------------------------------------------
old = 'ACCESS_MARKER = cfg.MANIFEST_DIR / "36_final_test_access_consumption_state.json"\n'
new = '''ACCESS_MARKER = cfg.MANIFEST_DIR / "36_final_test_access_consumption_state.json"\n\n# Step-36A technical-recovery provenance. The first authorized raw-test\n# open stopped before scoring because the sealed parquet did not embed the\n# convenience common-multitask flag. Recovery uses the already-frozen\n# Step-10 structural membership ledger and permits exactly one logged\n# technical reopen, with no scientific decision changes.\nSTEP10_MEMBERSHIP = (\n    cfg.FEATURE_DATA_DIR / "step10_chronological_splits" / "10_split_membership.parquet"\n)\nEXPECTED_STEP10_MEMBERSHIP_SHA = (\n    "82772196c92e88f1926a75a70c42f5f19119d7245d93b8364455fd28642ee121"\n)\nRECOVERY_AMENDMENT = (\n    cfg.MANIFEST_DIR / "36a_schema_recovery_compatibility_amendment.json"\n)\nRECOVERY_FROZEN_SOURCE = (\n    cfg.MANIFEST_DIR / "step36_one_time_final_test_materialization_inference_scoring_v1_recovery_frozen.py"\n)\n'''
text = replace_once(text, old, new, "recovery constants")

# -----------------------------------------------------------------------------
# B. Preserve the original pre-access freeze while authorizing only the exact
#    patched recovery source through the Step-36A amendment.
# -----------------------------------------------------------------------------
pattern = r'def verify_preaccess_freeze\(\) -> dict:\n.*?\n\ndef write_access_marker'
replacement = '''def verify_preaccess_freeze() -> dict:\n    if not PREACCESS_FREEZE.exists() or not PREACCESS_FREEZE_SHA.exists():\n        raise RuntimeError("Step-36 pre-access freeze is missing. Run --prepare-only first.")\n\n    expected = PREACCESS_FREEZE_SHA.read_text(encoding="utf-8").strip().split()[0]\n    observed = sha256_file(PREACCESS_FREEZE)\n    require_equal(observed, expected, "Step-36 pre-access freeze SHA sidecar")\n\n    payload = load_json(PREACCESS_FREEZE)\n    require_equal(\n        payload.get("status"),\n        "STEP36_PREACCESS_EXECUTION_SOURCE_FROZEN",\n        "Step-36 pre-access status",\n    )\n    require_equal(\n        sha256_file(FROZEN_SOURCE),\n        payload["source_sha256"],\n        "Original frozen Step-36 source",\n    )\n\n    current_source_sha = sha256_file(SCRIPT_FILE)\n    if current_source_sha == payload["source_sha256"]:\n        return payload\n\n    if not RECOVERY_AMENDMENT.exists() or not RECOVERY_FROZEN_SOURCE.exists():\n        raise RuntimeError(\n            "Current Step-36 source differs from the original pre-access freeze, "\n            "and no frozen Step-36A recovery amendment is present."\n        )\n\n    amendment = load_json(RECOVERY_AMENDMENT)\n    require_equal(\n        amendment.get("status"),\n        "STEP36A_SCHEMA_RECOVERY_COMPATIBILITY_FROZEN",\n        "Step-36A recovery status",\n    )\n    require_equal(\n        amendment.get("original_preaccess_source_sha256"),\n        payload["source_sha256"],\n        "Step-36A original-source reference",\n    )\n    require_equal(\n        amendment.get("recovery_source_sha256"),\n        current_source_sha,\n        "Step-36A recovery-source reference",\n    )\n    require_equal(\n        sha256_file(RECOVERY_FROZEN_SOURCE),\n        current_source_sha,\n        "Frozen Step-36A recovery source",\n    )\n    require_equal(\n        amendment.get("step10_membership_sha256"),\n        EXPECTED_STEP10_MEMBERSHIP_SHA,\n        "Step-36A membership-ledger reference",\n    )\n    return payload\n\n\ndef write_access_marker'''
text = regex_replace_once(text, pattern, replacement, "verify_preaccess_freeze")

# -----------------------------------------------------------------------------
# C. Attach the exact pre-frozen Step-10 outer-final-test membership flags.
# -----------------------------------------------------------------------------
anchor = 'def load_and_parse_raw_test_exactly_once(test_path: Path, expected_sha: str) -> tuple[pd.DataFrame, str]:\n'
attach = '''def attach_frozen_final_test_membership(frame: pd.DataFrame) -> pd.DataFrame:\n    require_file_sha(\n        STEP10_MEMBERSHIP,\n        EXPECTED_STEP10_MEMBERSHIP_SHA,\n        "Step-10 structural membership ledger",\n    )\n    membership = pd.read_parquet(STEP10_MEMBERSHIP, engine="pyarrow")\n    membership["Date"] = pd.to_datetime(\n        membership["Date"], errors="raise"\n    ).dt.normalize()\n\n    source_columns = [\n        "Ticker",\n        "Date",\n        "outer_final_test_eligible_1d",\n        "outer_final_test_eligible_3d",\n        "outer_final_test_eligible_5d",\n        "outer_final_test_common_multitask",\n    ]\n    missing = sorted(set(source_columns) - set(membership.columns))\n    if missing:\n        raise RuntimeError(\n            f"Step-10 membership ledger missing recovery columns: {missing}"\n        )\n\n    flags = membership[source_columns].copy()\n    if flags[["Ticker", "Date"]].duplicated().any():\n        raise RuntimeError("Duplicate ticker-date keys in Step-10 membership ledger.")\n\n    flags = flags.rename(columns={\n        "outer_final_test_eligible_1d": "eligible_1d",\n        "outer_final_test_eligible_3d": "eligible_3d",\n        "outer_final_test_eligible_5d": "eligible_5d",\n        "outer_final_test_common_multitask": "eligible_common_multitask",\n    })\n\n    out = frame.merge(\n        flags,\n        on=["Ticker", "Date"],\n        how="left",\n        validate="one_to_one",\n    )\n\n    eligibility_columns = [\n        "eligible_1d",\n        "eligible_3d",\n        "eligible_5d",\n        "eligible_common_multitask",\n    ]\n    if out[eligibility_columns].isna().any().any():\n        raise RuntimeError(\n            "Some sealed-test ticker-date rows have no Step-10 frozen membership record."\n        )\n    for c in eligibility_columns:\n        out[c] = out[c].astype(bool)\n\n    expected_counts = {\n        "eligible_1d": 6640,\n        "eligible_3d": 6560,\n        "eligible_5d": 6480,\n        "eligible_common_multitask": 6480,\n    }\n    for c, expected in expected_counts.items():\n        require_equal(int(out[c].sum()), expected, f"Frozen final-test {c} rows")\n\n    conjunction = out["eligible_1d"] & out["eligible_3d"] & out["eligible_5d"]\n    if not np.array_equal(\n        conjunction.to_numpy(),\n        out["eligible_common_multitask"].to_numpy(),\n    ):\n        raise RuntimeError(\n            "Step-10 final-test common-multitask flag is not the frozen "\n            "1d/3d/5d intersection."\n        )\n    if not np.array_equal(\n        out["eligible_5d"].to_numpy(),\n        out["eligible_common_multitask"].to_numpy(),\n    ):\n        raise RuntimeError(\n            "Frozen final-test 5d eligibility no longer equals common-multitask eligibility."\n        )\n\n    return out\n\n\n''' + anchor
text = replace_once(text, anchor, attach, "frozen membership adapter")

# -----------------------------------------------------------------------------
# C2. Validate labels/target dates only on the pre-frozen horizon-eligible
#     rows. The outer sealed panel intentionally contains terminal reference
#     rows that are structurally ineligible for one or more future horizons.
# -----------------------------------------------------------------------------
pattern = r'def validate_opened_test\(frame: pd\.DataFrame, features: list\[str\]\) -> None:\n.*?\n\ndef build_common_endpoints'
replacement = '''def validate_opened_test(frame: pd.DataFrame, features: list[str]) -> None:
    required = {
        "Ticker", "Date", "market_session_id", "continuity_segment_id",
        "eligible_1d", "eligible_3d", "eligible_5d",
        "eligible_common_multitask",
        "direction_1d", "direction_3d", "direction_5d",
        "target_date_1d", "target_date_3d", "target_date_5d", *features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Final-test panel missing required columns:\\n{missing}")

    require_equal(len(frame), EXPECTED_OUTER_TEST_ROWS, "Final-test rows")
    require_equal(frame["Date"].nunique(), EXPECTED_OUTER_TEST_DATES, "Final-test dates")
    require_equal(frame["Ticker"].nunique(), EXPECTED_STOCKS, "Final-test stocks")
    if frame[["Ticker", "Date"]].duplicated().any():
        raise RuntimeError("Duplicate final-test ticker-date keys.")
    if not (frame.groupby("Date")["Ticker"].nunique() == EXPECTED_STOCKS).all():
        raise RuntimeError("Final-test cross-section is incomplete.")
    if not np.isfinite(frame[features].to_numpy(np.float64)).all():
        raise RuntimeError("Final-test predictors contain non-finite values.")

    for h in HORIZONS:
        flag = frame[f"eligible_{h}d"].astype(bool)
        labels = pd.to_numeric(frame.loc[flag, f"direction_{h}d"], errors="raise")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise RuntimeError(
                f"Eligible final-test rows contain invalid {h}d direction labels."
            )
        target_dates = pd.to_datetime(
            frame.loc[flag, f"target_date_{h}d"], errors="coerce"
        )
        if target_dates.isna().any():
            raise RuntimeError(
                f"Eligible final-test rows contain missing {h}d target dates."
            )
        reference_dates = pd.to_datetime(frame.loc[flag, "Date"], errors="raise")
        if not (target_dates.to_numpy() > reference_dates.to_numpy()).all():
            raise RuntimeError(
                f"Eligible final-test rows contain non-future {h}d target dates."
            )

    common = frame["eligible_common_multitask"].astype(bool)
    if int(common.sum()) != EXPECTED_COMMON_ROWS:
        raise RuntimeError("Frozen common-multitask row count drift.")


def build_common_endpoints'''
text = regex_replace_once(text, pattern, replacement, "eligibility-aware test validation")

# -----------------------------------------------------------------------------
# D. Replace materialization with one controlled technical recovery reopen.
# -----------------------------------------------------------------------------
pattern = r'def verify_materialized_arrays\(\) -> dict:\n.*?\n\ndef get_fit_row'
replacement = '''def verify_materialized_arrays() -> dict:\n    m = load_json(ARRAY_MANIFEST)\n    require_equal(\n        m.get("status"),\n        "STEP36_FINAL_TEST_MATERIALIZATION_COMPLETE",\n        "Step-36 materialization status",\n    )\n    require_equal(\n        int(m["raw_test"]["filesystem_content_open_count"]),\n        2,\n        "Raw final-test filesystem content-open count",\n    )\n    require_equal(\n        bool(m["raw_test"].get("technical_recovery_reopen")),\n        True,\n        "Step-36 technical-recovery reopen flag",\n    )\n    require_equal(\n        int(m["evaluation_panel"]["rows"]),\n        EXPECTED_COMMON_ROWS,\n        "Materialized evaluation rows",\n    )\n    for rec in m.get("artifacts", {}).values():\n        p = Path(rec["path"])\n        if not p.exists() or sha256_file(p) != rec["sha256"]:\n            raise RuntimeError(f"Step-36 materialized artifact drift:\\n{p}")\n    return m\n\n\ndef materialize_final_test(upstream: dict) -> dict:\n    if ARRAY_MANIFEST.exists():\n        return verify_materialized_arrays()\n\n    seal = upstream["test_seal"]\n    raw_path = Path(seal["file_path"])\n    expected_raw_sha = str(seal["sha256"])\n\n    if not RECOVERY_AMENDMENT.exists():\n        raise RuntimeError("Step-36A recovery amendment is missing.")\n    amendment = load_json(RECOVERY_AMENDMENT)\n    require_equal(\n        amendment.get("status"),\n        "STEP36A_SCHEMA_RECOVERY_COMPATIBILITY_FROZEN",\n        "Step-36A recovery amendment status",\n    )\n\n    if not ACCESS_MARKER.exists():\n        raise RuntimeError(\n            "Step-36A recovery requires the preserved first-access marker."\n        )\n    marker = load_json(ACCESS_MARKER)\n    require_equal(\n        marker.get("state"),\n        "RAW_TEST_OPENED_ONCE_IN_MEMORY",\n        "Step-36 recovery access state",\n    )\n    require_equal(\n        int(marker.get("raw_test_filesystem_open_count_completed", -1)),\n        1,\n        "Step-36 first raw-test open count",\n    )\n    require_equal(\n        bool(marker.get("final_test_scored")),\n        False,\n        "Step-36 pre-recovery scored flag",\n    )\n    require_equal(\n        marker.get("expected_raw_test_sha256"),\n        expected_raw_sha,\n        "Step-36 expected raw-test SHA",\n    )\n    require_equal(\n        marker.get("observed_raw_test_sha256"),\n        expected_raw_sha,\n        "Step-36 first-open raw-test SHA",\n    )\n\n    write_access_marker(\n        "TECHNICAL_RECOVERY_REOPEN_STARTED",\n        raw_test_filesystem_open_count_planned=2,\n        raw_test_filesystem_open_count_completed=1,\n        technical_recovery_reason=(\n            "sealed parquet omitted convenience eligible_common_multitask column; "\n            "use frozen Step-10 membership ledger"\n        ),\n        recovery_amendment_path=str(RECOVERY_AMENDMENT),\n        recovery_amendment_sha256=sha256_file(RECOVERY_AMENDMENT),\n        final_test_scored=False,\n    )\n\n    print(\n        "Reopening sealed final-test parquet ONCE for logged technical "\n        "schema recovery (second filesystem content open; no prior scoring)..."\n    )\n    raw_test, observed_raw_sha = load_and_parse_raw_test_exactly_once(\n        raw_path, expected_raw_sha\n    )\n    write_access_marker(\n        "RAW_TEST_REOPENED_FOR_TECHNICAL_RECOVERY",\n        observed_raw_test_sha256=observed_raw_sha,\n        raw_test_filesystem_open_count_completed=2,\n        technical_recovery_reopen_count=1,\n        final_test_scored=False,\n    )\n\n    raw_test = normalize_dates(raw_test)\n    raw_test = attach_frozen_final_test_membership(raw_test)\n    validate_opened_test(raw_test, upstream["features"])\n\n    test_regimes = apply_frozen_regimes(raw_test)\n    test_scaled = transform_frozen_predictors(\n        test_regimes, upstream["features"], upstream["scaler_path"]\n    )\n    endpoints = build_common_endpoints(test_scaled, upstream["stock_id_map"])\n\n    final_dev = normalize_dates(\n        pd.read_parquet(FINAL_DEV_HISTORY, engine="pyarrow")\n    )\n    lookup = build_history_lookup(final_dev, test_scaled, upstream["features"])\n    X_full, metadata = materialize_sequences(\n        endpoints, lookup, upstream["features"]\n    )\n    y_direction = metadata[list(DIRECTION_COLUMNS)].to_numpy(dtype=np.int64)\n    stock_id = metadata["stock_id"].to_numpy(dtype=np.int64)\n    regime_id = metadata["regime_id"].to_numpy(dtype=np.int64)\n    X_stock = X_full[:, :, :47].copy()\n    X_market = X_full[:, :, 47:].copy()\n\n    require_equal(X_full.shape, (EXPECTED_COMMON_ROWS, 60, 63), "X_full shape")\n    require_equal(X_stock.shape, (EXPECTED_COMMON_ROWS, 60, 47), "X_stock shape")\n    require_equal(X_market.shape, (EXPECTED_COMMON_ROWS, 60, 16), "X_market shape")\n    require_equal(y_direction.shape, (EXPECTED_COMMON_ROWS, 3), "y_direction shape")\n    if not np.array_equal(X_stock, X_full[:, :, :47]) or not np.array_equal(\n        X_market, X_full[:, :, 47:]\n    ):\n        raise RuntimeError("Final-test modality slice mismatch.")\n    if not np.isin(y_direction, [0, 1]).all():\n        raise RuntimeError("Final-test y_direction is not binary.")\n    if not np.isin(regime_id, [0, 1, 2]).all():\n        raise RuntimeError("Final-test regime IDs are invalid.")\n\n    staging = cfg.FEATURE_DATA_DIR / (\n        f"_step36_final_test_staging_{os.getpid()}_{int(time.time())}"\n    )\n    staging.mkdir(parents=True, exist_ok=False)\n    try:\n        save_array(staging / "36_final_test_X_full.npy", X_full)\n        save_array(staging / "36_final_test_X_stock.npy", X_stock)\n        save_array(staging / "36_final_test_X_market.npy", X_market)\n        save_array(staging / "36_final_test_y_direction.npy", y_direction)\n        save_array(staging / "36_final_test_stock_id.npy", stock_id)\n        save_array(staging / "36_final_test_regime_id.npy", regime_id)\n        metadata.to_parquet(\n            staging / "36_final_test_metadata.parquet",\n            index=False,\n            engine="pyarrow",\n            compression="snappy",\n        )\n        if len(\n            pd.read_parquet(\n                staging / "36_final_test_metadata.parquet", engine="pyarrow"\n            )\n        ) != EXPECTED_COMMON_ROWS:\n            raise RuntimeError("Final-test metadata roundtrip row drift.")\n        if ARRAY_DIR.exists():\n            raise RuntimeError(\n                f"Unexpected existing Step-36 array directory:\\n{ARRAY_DIR}"\n            )\n        os.replace(staging, ARRAY_DIR)\n    except Exception:\n        if staging.exists():\n            shutil.rmtree(staging, ignore_errors=True)\n        raise\n\n    artifacts = {\n        p.name: artifact_record(p)\n        for p in sorted(ARRAY_DIR.iterdir())\n        if p.is_file()\n    }\n    manifest = {\n        "schema_version": SCHEMA_VERSION,\n        "status": "STEP36_FINAL_TEST_MATERIALIZATION_COMPLETE",\n        "created_at_utc": utc_now(),\n        "step35_manifest_sha256": EXPECTED_STEP35_MANIFEST_SHA,\n        "step35_contract_fingerprint": EXPECTED_STEP35_CONTRACT_FP,\n        "step36a_recovery_amendment": {\n            "path": str(RECOVERY_AMENDMENT),\n            "sha256": sha256_file(RECOVERY_AMENDMENT),\n            "reason": (\n                "schema-only recovery using frozen Step-10 membership ledger; "\n                "no metrics existed before amendment"\n            ),\n        },\n        "raw_test": {\n            "path": str(raw_path),\n            "expected_sha256": expected_raw_sha,\n            "observed_sha256": observed_raw_sha,\n            "filesystem_content_open_count": 2,\n            "initial_authorized_open_count": 1,\n            "technical_recovery_reopen_count": 1,\n            "technical_recovery_reopen": True,\n            "parsed_from_same_in_memory_bytes_used_for_hash_on_each_open": True,\n            "rows": int(len(raw_test)),\n            "dates": int(raw_test["Date"].nunique()),\n            "stocks": int(raw_test["Ticker"].nunique()),\n        },\n        "structural_membership": {\n            "path": str(STEP10_MEMBERSHIP),\n            "sha256": sha256_file(STEP10_MEMBERSHIP),\n            "source_column": "outer_final_test_common_multitask",\n            "attached_column": "eligible_common_multitask",\n            "common_rows": int(raw_test["eligible_common_multitask"].sum()),\n            "membership_refit_or_recomputed_from_test_labels": False,\n        },\n        "transformation": {\n            "predictor_scaler_path": str(upstream["scaler_path"]),\n            "predictor_scaler_sha256": sha256_file(upstream["scaler_path"]),\n            "predictor_scaler_refit_on_test": False,\n            "predictor_clipping": False,\n            "regime_threshold_path": str(FINAL_DEV_REGIME_THRESHOLDS),\n            "regime_threshold_sha256": sha256_file(FINAL_DEV_REGIME_THRESHOLDS),\n            "regime_threshold_refit_on_test": False,\n            "development_history_path": str(FINAL_DEV_HISTORY),\n            "development_history_sha256": sha256_file(FINAL_DEV_HISTORY),\n            "lookback": 60,\n            "causal_development_history_allowed": True,\n            "future_predictor_rows_allowed": False,\n        },\n        "evaluation_panel": {\n            "rows": EXPECTED_COMMON_ROWS,\n            "dates": EXPECTED_COMMON_DATES,\n            "stocks": EXPECTED_STOCKS,\n            "x_full_shape": [EXPECTED_COMMON_ROWS, 60, 63],\n            "y_direction_shape": [EXPECTED_COMMON_ROWS, 3],\n        },\n        "artifacts": artifacts,\n    }\n    manifest["materialization_fingerprint"] = canonical_hash(manifest)\n    atomic_json(ARRAY_MANIFEST, manifest)\n\n    write_access_marker(\n        "RAW_TEST_MATERIALIZED_AND_CLOSED_AFTER_TECHNICAL_RECOVERY",\n        observed_raw_test_sha256=observed_raw_sha,\n        raw_test_filesystem_open_count_completed=2,\n        technical_recovery_reopen_count=1,\n        raw_test_reopen_prohibited=True,\n        materialization_manifest_path=str(ARRAY_MANIFEST),\n        materialization_manifest_sha256=sha256_file(ARRAY_MANIFEST),\n        final_test_scored=False,\n    )\n    return verify_materialized_arrays()\n\n\ndef get_fit_row'''
text = regex_replace_once(text, pattern, replacement, "recovery materialization")

# -----------------------------------------------------------------------------
# E. Final provenance must report two filesystem content opens transparently:
#    one initial authorized open + one technical recovery reopen before scoring.
# -----------------------------------------------------------------------------
old = '''            "raw_test_filesystem_content_open_count": 1,\n            "raw_test_reopened_after_materialization": False,'''
new = '''            "raw_test_filesystem_content_open_count": 2,\n            "initial_authorized_open_count": 1,\n            "technical_recovery_reopen_count": 1,\n            "technical_recovery_reopen_before_scoring": True,\n            "raw_test_reopened_after_materialization": False,'''
text = replace_once(text, old, new, "final manifest open-count provenance")

old = '''        raw_test_filesystem_open_count_completed=1,\n        raw_test_reopen_prohibited=True,\n        final_test_scored=True,'''
new = '''        raw_test_filesystem_open_count_completed=2,\n        technical_recovery_reopen_count=1,\n        raw_test_reopen_prohibited=True,\n        final_test_scored=True,'''
text = replace_once(text, old, new, "final access-marker provenance")

# Write patched source, compile it, freeze exact recovery source.
SOURCE.write_text(text, encoding="utf-8")
try:
    py_compile.compile(str(SOURCE), doraise=True)
except Exception:
    shutil.copy2(BACKUP, SOURCE)
    raise

recovery_source_sha = sha256_file(SOURCE)
shutil.copy2(SOURCE, RECOVERY_FROZEN_SOURCE)
if sha256_file(RECOVERY_FROZEN_SOURCE) != recovery_source_sha:
    raise RuntimeError("Recovery frozen-source copy SHA mismatch.")

payload = {
    "schema_version": "1.0",
    "status": "STEP36A_SCHEMA_RECOVERY_COMPATIBILITY_FROZEN",
    "created_at_utc": utc_now(),
    "reason": (
        "The first authorized Step-36 raw-test open stopped before scoring because "
        "the sealed parquet did not embed the convenience eligible_common_multitask "
        "column. The exact eligibility decision already existed in the pre-frozen "
        "Step-10 structural membership ledger."
    ),
    "scientific_protocol_changed": False,
    "model_config_changed": False,
    "seed_set_changed": False,
    "thresholds_changed": False,
    "preprocessing_fit_changed": False,
    "regime_thresholds_changed": False,
    "test_metrics_seen_before_amendment": False,
    "test_class_balance_used_for_amendment": False,
    "original_preaccess_manifest_path": str(PREACCESS_FREEZE),
    "original_preaccess_manifest_sha256": sha256_file(PREACCESS_FREEZE),
    "original_preaccess_source_sha256": original_source_sha,
    "recovery_source_path": str(SOURCE),
    "recovery_source_sha256": recovery_source_sha,
    "recovery_frozen_source_path": str(RECOVERY_FROZEN_SOURCE),
    "step10_membership_path": str(MEMBERSHIP),
    "step10_membership_sha256": membership_sha,
    "membership_source_columns": required_membership[2:],
    "frozen_membership_counts": counts,
    "first_access_marker_before_amendment": marker,
    "recovery_permission": {
        "additional_raw_test_filesystem_content_opens_allowed": 1,
        "maximum_total_raw_test_filesystem_content_opens": 2,
        "purpose": "technical schema recovery and deterministic materialization only",
        "must_verify_same_presealed_sha": True,
        "must_join_only_pre_frozen_step10_membership": True,
        "model_or_threshold_adaptation_allowed": False,
        "post_test_promotion_allowed": False,
    },
}
payload["recovery_amendment_fingerprint"] = canonical_hash(payload)
atomic_json(AMENDMENT, payload)

print("Original Step-36 source SHA :", original_source_sha)
print("Recovery Step-36 source SHA :", recovery_source_sha)
print("Membership SHA              :", membership_sha)
print("Recovery amendment          :", AMENDMENT)
print("Recovery amendment SHA      :", sha256_file(AMENDMENT))
print("Recovery amendment FP       :", payload["recovery_amendment_fingerprint"])
print("Backup source               :", BACKUP)
print("Recovery frozen source      :", RECOVERY_FROZEN_SOURCE)
print("Syntax check                : PASS")
print("Raw final-test reopened     : False")
print("Final-test metrics computed : False")
print("=" * 140)
print("STEP 36A PASS — SCHEMA-ONLY RECOVERY AMENDMENT FROZEN")
print("DO NOT RUN STEP 36 --execute UNTIL THIS OUTPUT HAS BEEN AUDITED.")
print("=" * 140)
