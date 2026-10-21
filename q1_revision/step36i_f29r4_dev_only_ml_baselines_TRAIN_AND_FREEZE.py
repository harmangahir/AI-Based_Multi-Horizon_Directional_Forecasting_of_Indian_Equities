#!/usr/bin/env python3
"""F29R4 — development-only, post-freeze supplementary ML baselines.

Train 3 conventional classifiers x 3 horizons with FIXED hyperparameters and
fixed 0.5 decisions. Uses only frozen Step14 final-development sequences and
Step12 final-development metadata/labels; does not open raw test data or Phase A.

This makes NO cross-validation / out-of-sample performance claims. The Step14
predictor scaler was fitted across final development, so CV on its scaled
outputs would be contaminated. This supplementary study instead uses fixed
settings, full-development final fits, and a single later held-out evaluation.

All nine models frozen together in an atomic new directory; no overwrites.
Original Step35 tests, 40 checkpoints, and 24 thresholds are never modified.

Run: python step36i_f29r4_dev_only_ml_baselines_TRAIN_AND_FREEZE.py --train-and-freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
MAN = EXP / 'manifests'
STEP35 = MAN / '35_final_test_evaluation_contract_freeze_manifest.json'
STEP14 = MAN / '14_model_array_manifest.json'
ACCESS = MAN / '36_final_test_access_consumption_state.json'
SHA35 = '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7'
X_SHA = 'c6857902355ffdf4dd4feb5c10f079f27e9b66a8c3ba5c749c959be7caee1778'
OUT = EXP / 'feature_data' / 'step36i_f29r4_devonly_ml_baselines_v1'
HORIZONS = (1, 3, 5)
MODELS = ('LOGREG', 'RANDOM_FOREST', 'XGBOOST')
EXPECTED_N = 42360
EXPECTED_FEATURES = 63
DEVELOPMENT_END = pd.Timestamp('2024-12-31')
RESERVED = [
    EXP / 'feature_data/step36i_amended_final_test_arrays',
    MAN / '36i_amended_final_test_materialization_manifest.json',
    EXP / 'final_test_evaluation/step36i_amended_predictions',
    MAN / '36i_amended_final_test_evaluation_manifest.json',
    EXP / 'audits/36i_amended_final_test_metrics.csv',
    MAN / '36i_amended_final_test_access_journal.json',
    EXP / 'audits/36i_amended_final_test_execution_failure.json',
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def guard_file(path, expected, label):
    require(path.is_file() and not path.is_symlink(), f'Missing/unsafe {label}: {path}')
    require(sha(path) == expected, f'{label} SHA-256 mismatch')


def check_private_outputs_vacant():
    for item in RESERVED:
        require(not item.exists() and not item.is_symlink(), f'Reserved live-test artifact occupied: {item}')


def locate_frozen_data():
    guard_file(STEP35, SHA35, 'Step35 contract')
    d35 = json.loads(STEP35.read_bytes())
    d14 = json.loads(STEP14.read_bytes())
    feat35 = d35['predictor_preprocessing_contract']['ordered_features']
    feat14 = d14['ordered_63_features']
    require(isinstance(feat35, list) and len(feat35) == 63 and len(set(feat35)) == 63,
            'Invalid frozen ordered features')
    require(feat35 == feat14, 'Step14 and Step35 ordered features differ')
    require(d14['array_interface']['lookback_sessions'] == 60 and
            d14['array_interface']['common_multitask_only'] is True,
            'Step14 lookback / multitask interface changed')
    history_obj = d35['test_sequence_materialization_contract']
    hp = Path(history_obj['final_development_history_panel'])
    require(hp.name == '12_final_development_scaled_regimes.parquet' and
            hp.is_relative_to(EXP) and hp.is_file() and not hp.is_symlink(),
            'Development history file not frozen to expected location')
    guard_file(hp, history_obj['final_development_history_sha256'], 'Step12 development history')
    xrel = 'feature_data/step14_model_arrays/final_development/14_final_development_X_full.npy'
    artifact = d14['artifacts'].get(Path(xrel).name)
    require(isinstance(artifact, dict) and artifact.get('relative_path') == xrel,
            'Frozen Step14 final-development sequence registry missing or drifted')
    require(artifact.get('sha256') == X_SHA, 'Step14 manifest X SHA drifted')
    xp = EXP / xrel
    guard_file(xp, X_SHA, 'Step14 final-development X_full')
    return feat35, hp, xp, d14


def row_keys_exact(panel_matrix: np.ndarray, array_last: np.ndarray):
    """Exact last-timestep match gives sample keys without guessing Step14 row order.

    Dataset is only 43,640 x 63, so the non-cryptographic vector -> row map is
    subsequently checked with exact equality, unique matches, and eligibility.
    """
    require(panel_matrix.dtype == np.float32 and array_last.dtype == np.float32,
            'Expected float32 panel and array')
    index = {}
    for i, r in enumerate(panel_matrix):
        if not np.isfinite(r).all():
            continue
        b = r.tobytes()
        if b in index:
            # Duplicate feature fingerprints are ambiguous rather than guessed.
            index[b] = None
        else:
            index[b] = i
    matched = np.empty(array_last.shape[0], dtype=np.int64)
    for i in range(array_last.shape[0]):
        row = np.asarray(array_last[i], dtype=np.float32)
        k = index.get(row.tobytes(), -1)
        require(k is not None and k != -1,
                f'Could not uniquely match sequence {i} to the frozen development panel; STOP')
        require(np.array_equal(row, panel_matrix[k]), 'Last timestep differs from frozen panel')
        matched[i] = k
    require(len(set(matched.tolist())) == len(matched), 'Step14 sequence endpoint duplicated')
    return matched


def cohort_and_predictors(features, hp, xp, stock_id_map):
    # Deliberately access ONLY the two verified final-development files.
    print('[2] Read frozen non-test Step14 sequences and Step12 development labels', flush=True)
    X = np.load(xp, mmap_mode='r', allow_pickle=False)
    require(X.shape == (EXPECTED_N, 60, EXPECTED_FEATURES) and X.dtype == np.float32,
            f'Frozen Step14 X shape/dtype changed: {X.shape}/{X.dtype}')
    last = np.asarray(X[:, -1, :], dtype=np.float32)
    require(np.isfinite(last).all(), 'Nonfinite frozen development endpoint features')
    columns = list(dict.fromkeys(features + [
        'Ticker', 'Date', 'market_session_id', 'session_number',
        'continuity_segment_id', 'eligible_common_multitask',
        'preprocessing_partition', 'preprocessing_fold',
    ] + [f'direction_{h}d' for h in HORIZONS] +
        [f'target_date_{h}d' for h in HORIZONS] +
        [f'target_continuity_segment_id_{h}d' for h in HORIZONS]))
    panel = pd.read_parquet(hp, columns=columns)
    require(len(panel) == 43640, f'Expected 43640 development-only rows, got {len(panel)}')
    panel['Date'] = pd.to_datetime(panel['Date'], errors='raise').dt.normalize()
    require(panel.Date.max() <= DEVELOPMENT_END and panel.Date.notna().all(),
            'Development history includes future or missing reference date')
    require(panel.Ticker.nunique() == 20 and set(panel.Ticker) == set(stock_id_map),
            'Frozen 20-stock universe differs')
    frame = panel[features].to_numpy(dtype=np.float32)
    matched = row_keys_exact(frame, last)
    meta = panel.iloc[matched].reset_index(drop=True)
    require(meta['eligible_common_multitask'].fillna(False).astype(bool).all(),
            'Step14 endpoints include ineligible multi-task rows')
    require(meta['Date'].nunique() == EXPECTED_N // 20 and
            meta.groupby('Date').Ticker.nunique().eq(20).all(),
            'Not 20 stocks per final-development endpoint date')
    require(meta['Ticker'].value_counts().eq(EXPECTED_N // 20).all(),
            'Unbalanced training cohort')
    require(not meta[['Ticker','Date']].duplicated().any(), 'Duplicate dev endpoint')
    for h in HORIZONS:
        labels = pd.to_numeric(meta[f'direction_{h}d'], errors='raise')
        require(labels.notna().all() and labels.isin((0,1)).all(),
                f'Invalid {h}d direction labels')
        td = pd.to_datetime(meta[f'target_date_{h}d'], errors='raise').dt.normalize()
        require(td.notna().all() and td.le(DEVELOPMENT_END).all() and
                td.gt(meta.Date).all(),
                f'{h}d target matures outside final development or at/before reference')
        require((pd.to_numeric(meta[f'target_continuity_segment_id_{h}d'], errors='raise') ==
                 pd.to_numeric(meta['continuity_segment_id'], errors='raise')).all(),
                f'{h}d labels cross a continuity segment')
    # Entire training input derives from the exact last row of each frozen
    # 60x63 sequence. Includes lagged/rolling indicators; does not add future data.
    y = np.column_stack([meta[f'direction_{h}d'].to_numpy(dtype=np.int8)
                         for h in HORIZONS])
    return last, y, meta


def make_estimator(name, y, seed=42):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    if name == 'LOGREG':
        return Pipeline([
            ('standardize', StandardScaler()),
            ('classifier', LogisticRegression(C=1.0, solver='lbfgs',
               max_iter=2000, class_weight='balanced', random_state=seed))])
    if name == 'RANDOM_FOREST':
        return RandomForestClassifier(n_estimators=300, max_depth=12,
            min_samples_leaf=20, max_features='sqrt', bootstrap=True,
            class_weight='balanced_subsample', n_jobs=4, random_state=seed)
    if name == 'XGBOOST':
        from xgboost import XGBClassifier
        n_pos = int(y.sum())
        require(n_pos > 0 and n_pos < len(y), 'XGBoost train labels single-class')
        return XGBClassifier(n_estimators=350, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=15, reg_lambda=10, tree_method='hist',
            objective='binary:logistic', eval_metric='logloss', n_jobs=4,
            random_state=seed, scale_pos_weight=(len(y)-n_pos)/n_pos)
    raise RuntimeError(f'Unknown baseline {name}')


def train_all():
    import sklearn
    import joblib
    import xgboost
    import pyarrow
    import importlib.metadata as md

    os.umask(0o077)
    print('=' * 98, flush=True)
    print('F29R4 — ACTUAL DEVELOPMENT-ONLY TRAINING AND IMMUTABLE BASELINE FREEZE', flush=True)
    print('=' * 98, flush=True)
    print('NO sealed test access. Post-freeze supplementary comparators. Fixed hyperparameters.', flush=True)
    require(not OUT.exists() and not OUT.is_symlink(),
            'Baseline freeze output already exists; NEVER overwrite')
    check_private_outputs_vacant()
    raw_access = ACCESS.read_bytes()
    access_doc = json.loads(raw_access)
    require(access_doc.get('raw_test_filesystem_open_count_completed') == 2 and
            access_doc.get('final_test_scored') is False,
            'Historical test marker incompatible with protected non-test training')
    print('[1] Verify frozen Step35, Step14, Step12 digests', flush=True)
    feat, hp, xp, d14 = locate_frozen_data()
    source_hash = sha(Path(__file__).resolve())
    XX, YY, meta = cohort_and_predictors(feat, hp, xp, d14['stock_id_map'])
    print(f'PASS — exact Step14 cohort {len(XX)} endpoints / {meta.Date.nunique()} dates / {len(feat)} features', flush=True)
    print('PASS — all 1d/3d/5d targets mature within 2016–2024 development', flush=True)
    print('PASS — 60-session frozen sequence last-timestep match; 63 causal indicators', flush=True)

    # Leave the globally reserved test namespace untouched. Publish all 9
    # baselines atomically only when every fit and integrity check passes.
    staging = OUT.parent / ('.step36i_f29r4_staging_' + uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    try:
        (staging / 'models').mkdir(mode=0o700)
        entries = []
        for model in MODELS:
            for j, h in enumerate(HORIZONS):
                labels = YY[:, j]
                require(set(labels.tolist()) == {0,1}, f'One class in {model} {h}d training')
                print(f'[3] FIT {model} | {h}d | N={len(labels)} | positives={int(labels.sum())}', flush=True)
                tic = time.monotonic()
                clf = make_estimator(model, labels)
                clf.fit(XX, labels)
                require(list(map(int, clf.classes_)) == [0,1],
                        f'Unexpected {model} {h}d class mapping')
                # Training-data sanity only; NEVER interpret as predictive accuracy.
                pred = np.asarray(clf.predict_proba(XX[:25])[:, 1])
                require(pred.shape == (25,) and np.isfinite(pred).all() and
                        np.logical_and(pred >= 0, pred <= 1).all(),
                        f'Nonfinite {model} {h}d sanity probabilities')
                name = f'{model}_{h}d.joblib'
                out = staging / 'models' / name
                joblib.dump(clf, out, compress=3)
                entries.append({
                    'model_id': model, 'horizon_days': h,
                    'relative_path': f'models/{name}', 'sha256': sha(out),
                    'size_bytes': out.stat().st_size,
                    'decision_threshold': 0.5,
                    'positive_class': 1, 'n_train': int(len(labels)),
                    'positive_train': int(labels.sum()),
                    'elapsed_seconds': round(time.monotonic()-tic, 2),
                })
                print(f'PASS — {name} | frozen SHA256={entries[-1]["sha256"]}', flush=True)
        require(len(entries) == 9, 'Nine baseline models not produced')
        key_columns = ['Ticker','Date','market_session_id','continuity_segment_id']
        keys = meta[key_columns].copy()
        keys['Date'] = keys.Date.dt.strftime('%Y-%m-%d')
        keys_data = keys.to_csv(index=False, lineterminator='\n').encode()
        (staging / 'training_endpoint_keys.csv').write_bytes(keys_data)
        manifest = {
            'schema': 'STEP36I_F29R4_DEV_ONLY_POST_FREEZE_BASELINES_V1',
            'scientific_role': 'SUPPLEMENTARY_POST_FREEZE_EXPLORATORY_NO_PRIMARY_HYPOTHESIS_MODIFICATION',
            'trained_on': 'frozen Step14 final-development 60x63 endpoint last snapshot',
            'input_representation': '63 values of most recent reference-day session from frozen 60-session sequence',
            'information_cutoff': 'reference-session close; all target dates before or on 2024-12-31',
            'training_scaling': 'Step11 final-development MinMax frozen; LOGREG extra StandardScaler fitted on dev only',
            'parameter_policy': 'fixed a priori in signed source; no hyperparameter selection or cross-validation claimed',
            'cv_caution': 'Final-development scaler was fitted on all development; no leakage-free nested CV claim',
            'seed': 42,
            'decision_rule': 'probability >= 0.5 predicts up; fixed before final test',
            'step35_sha256': SHA35,
            'step14_manifest_sha256': sha(STEP14),
            'step12_history_sha256': sha(hp),
            'step14_final_dev_X_full_sha256': X_SHA,
            'source_sha256': source_hash,
            'ordered_63_features': feat,
            'stock_id_map': d14['stock_id_map'],
            'rows': int(len(XX)), 'trading_sessions': int(meta.Date.nunique()),
            'model_artifacts': entries,
            'training_keys': {'relative_path': 'training_endpoint_keys.csv',
                              'sha256': hashlib.sha256(keys_data).hexdigest(),
                              'size_bytes': len(keys_data)},
            'package_versions': {k:md.version(k) for k in ('numpy','pandas','scikit-learn','xgboost','joblib','pyarrow')},
            'python_version': platform.python_version(),
            'historical_raw_test_reads_before_and_after': 2,
            'test_data_opened': False,
            'real_final_test_scored': False,
            'original_40_checkpoints_modified': False,
            'original_24_hypotheses_modified': False,
        }
        manifest_data = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
        (staging / 'manifest.json').write_bytes(manifest_data)
        # Independent file-hash verification in staging BEFORE atomic publish.
        for entry in entries:
            path = staging / entry['relative_path']
            require(sha(path) == entry['sha256'] and path.stat().st_size == entry['size_bytes'],
                    f'Baseline artifact changed before publish: {path}')
        require(sha(staging/'training_endpoint_keys.csv') == manifest['training_keys']['sha256'],
                'Development-only key inventory changed')
        require(ACCESS.read_bytes() == raw_access, 'Historical access marker changed during training')
        check_private_outputs_vacant()
        # Publication by atomic rename; never replace an existing output.
        require(not OUT.exists() and not OUT.is_symlink(), 'Baseline output race/occupied')
        staging.rename(OUT)
        print('=' * 98, flush=True)
        print('F29R4 PASS — NINE REAL DEVELOPMENT-TRAINED ML BASELINES FROZEN', flush=True)
        print('Freeze directory:', OUT, flush=True)
        print('Manifest SHA256:', hashlib.sha256(manifest_data).hexdigest(), flush=True)
        print('Development endpoints:', len(XX), '| model files:', len(entries), flush=True)
        print('Raw final test opened: FALSE | Raw test scored: FALSE | Third read authorized: FALSE', flush=True)
        print('F29R3 controller UNCHANGED; integration into supplementary finalizer still pending.', flush=True)
        print('=' * 98, flush=True)
    except BaseException:
        # We deliberately retain failed staging for diagnosis rather than write
        # anything to a reserved raw-test or held-out results namespace.
        print('TRAIN/FREEZE FAILED — no baseline freeze published.', file=sys.stderr, flush=True)
        print('Uncommitted work directory (contains DEV-only artifacts):', staging,
              file=sys.stderr, flush=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-and-freeze', action='store_true')
    args = p.parse_args()
    require(args.train_and_freeze, 'Only explicit --train-and-freeze is available')
    train_all()

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'F29R4 STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(2)
