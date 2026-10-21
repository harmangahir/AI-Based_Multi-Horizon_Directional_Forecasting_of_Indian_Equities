#!/usr/bin/env python3
"""F29R5: verified, immutable F29R4 development-only ML baselines on sealed Phase A.

This library NEVER reads raw held-out source paths. It consumes only a previously
independently authorized and sealed 5300x60x63 Phase A array, immutable trained
model files, and the already verified segment-stratified bootstrap indices.
This script's only CLI is --offline-preflight, which reads DEV-ONLY model files
for SHA authentication, but performs NO held-out scoring or GPU inference.

All analyses are POST-FREEZE EXPLORATORY; original 24 primary tests unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
BASE_DIR = EXP / 'feature_data/step36i_f29r4_devonly_ml_baselines_v1'
MANIFEST_SHA = 'de89cb9d2c17d374aea3870f66a5633d3a2b1a8d86a6c37310d31dc79d3be09f'
TRAIN_SOURCE_SHA = '8f5b93f56a365f9fa424e647030cc23d59fb1f59812bfdddbb0fd8a2c9800994'
STEP14_X_SHA = 'c6857902355ffdf4dd4feb5c10f079f27e9b66a8c3ba5c749c959be7caee1778'
STEP35_SHA = '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7'
MODEL_SHAS = {
 ('LOGREG',1): 'b26be5a7aa434fafbfb3e5695652c23e4b1123e018638199dee4cbe24fbe47ef',
 ('LOGREG',3): '3540d709d46a9f72d1f4ac05c0eab19d85117a42c0248508358f7e7ea9507908',
 ('LOGREG',5): 'f3355bcd4c5a8c6ae10ab6fe427eb81d26af8f7ca15cf06f3592003e4df4efaa',
 ('RANDOM_FOREST',1): '4da8e56d7437d109025226e789dd8f8c2cd17b5d652774787fa7ea9312ca4f62',
 ('RANDOM_FOREST',3): '140689223635a814aaa2b02b0ea08fc5cee9d1590f19d610660980cfcee1f787',
 ('RANDOM_FOREST',5): '0b152d166a3d543fc4c4c8c6618ef954c6af2d9b9cf673e9187f5d47278262ad',
 ('XGBOOST',1): '36b3a674ac170dac3ead41b399873bc8841b9d597292de37196638730833f672',
 ('XGBOOST',3): 'decf5bb0abf9c13f45a5f9409533b464a227ae8dee4134008bfa1c0b0692ea1c',
 ('XGBOOST',5): '9735bbe6bc2626a011b68b92b68d2ee1681f1f115384cad17320ec3b9a026bca',
}
HORIZONS = (1,3,5)
NN_MODELS = ('DLINEAR','NLINEAR','PATCHTST','TIMESNET',
             'ITRANSFORMER','FREDFORMER','FRNET','MLF')
N, ND, NS = 5300, 265, 20
B = 2000


def require(flag, msg):
    if not bool(flag): raise RuntimeError(msg)


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as inp:
        for part in iter(lambda: inp.read(4<<20), b''): h.update(part)
    return h.hexdigest()


def safe_regular(path):
    s=path.lstat()
    require(stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode) and
            s.st_uid==os.geteuid() and not (stat.S_IMODE(s.st_mode)&0o022),
            f'Unsafe or non-owned baseline artifact: {path}')


def require_safe_dir(path):
    s=path.lstat()
    require(stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode) and
            s.st_uid==os.geteuid() and not (stat.S_IMODE(s.st_mode)&0o022),
            f'Unsafe baseline directory: {path}')


def authenticate_frozen_baselines(*, ordered_features, inspect_only=True):
    """Authenticate F29R4 manifest + 9 models before deserializing joblib.

    Uses a literal signed-manifest SHA copied from the completed WSL run;
    failure rather than discovery if an artifact changes. No sealed test access.
    """
    require_safe_dir(BASE_DIR)
    require_safe_dir(BASE_DIR/'models')
    manifest_path=BASE_DIR/'manifest.json'
    safe_regular(manifest_path)
    require(sha_file(manifest_path)==MANIFEST_SHA,'F29R4 freeze manifest SHA mismatch')
    manifest=json.loads(manifest_path.read_bytes())
    require(manifest.get('schema')=='STEP36I_F29R4_DEV_ONLY_POST_FREEZE_BASELINES_V1' and
            manifest.get('step35_sha256')==STEP35_SHA and
            manifest.get('source_sha256')==TRAIN_SOURCE_SHA and
            manifest.get('rows')==42360 and manifest.get('trading_sessions')==2118 and
            manifest.get('test_data_opened') is False and
            manifest.get('real_final_test_scored') is False and
            manifest.get('original_40_checkpoints_modified') is False and
            manifest.get('original_24_hypotheses_modified') is False and
            manifest.get('seed')==42 and
            manifest.get('decision_rule')=='probability >= 0.5 predicts up; fixed before final test' and
            manifest.get('input_representation')==
              '63 values of most recent reference-day session from frozen 60-session sequence',
            'Frozen F29R4 scientific or development-only training contract changed')
    require(list(ordered_features)==manifest['ordered_63_features'] and
            len(ordered_features)==63 and len(set(ordered_features))==63,
            'Final-test ordered features differ from F29R4 trained feature order')
    require(len(manifest['model_artifacts'])==9,'Expected nine ML model entries')
    entries={}
    for entry in manifest['model_artifacts']:
        model=entry['model_id']; horizon=int(entry['horizon_days']); k=(model,horizon)
        require(k in MODEL_SHAS and k not in entries,'Duplicate or unexpected ML model')
        relative=f'models/{model}_{horizon}d.joblib'
        require(entry.get('relative_path')==relative and
                entry.get('sha256')==MODEL_SHAS[k] and
                entry.get('n_train')==42360 and entry.get('positive_class')==1 and
                entry.get('decision_threshold')==0.5,'Frozen ML artifact metadata drift')
        path=BASE_DIR/'models'/f'{model}_{horizon}d.joblib'
        safe_regular(path)
        require(path.stat().st_size==entry['size_bytes'] and
                sha_file(path)==MODEL_SHAS[k],f'F29R4 classifier tampering: {k}')
        entries[k]=path
    require(set(entries)==set(MODEL_SHAS),'Not all 9 ML classifiers verified')
    kentry=manifest['training_keys']
    require(kentry['relative_path']=='training_endpoint_keys.csv','Development cohort keys path drift')
    kp=BASE_DIR/'training_endpoint_keys.csv'
    safe_regular(kp)
    require(kp.stat().st_size==kentry['size_bytes'] and
            sha_file(kp)==kentry['sha256'],'Development cohort key audit changed')
    # Inspect_only avoids unpickling in audit-only CLI.
    return entries, manifest


def date_confusion(y,hat):
    yt=y.reshape(ND,NS).astype(bool); ph=hat.reshape(ND,NS).astype(bool)
    return np.stack([np.sum(~yt&~ph,axis=1),np.sum(~yt&ph,axis=1),
                     np.sum(yt&~ph,axis=1),np.sum(yt&ph,axis=1)],axis=1).astype(np.int64)


def calc_boot_ba(counts):
    require(counts.ndim==2 and counts.shape[1]==4,'Invalid confusion count rows')
    plus=counts[:,2]+counts[:,3]; minus=counts[:,0]+counts[:,1]
    require(np.all(plus>0) and np.all(minus>0),'Degenerate paired bootstrap BA')
    return .5*(counts[:,3]/plus+counts[:,0]/minus)


def paired_ci(model_counts, reference_counts, index):
    left=calc_boot_ba(model_counts[index].sum(axis=1))
    right=calc_boot_ba(reference_counts[index].sum(axis=1))
    values=left-right
    return tuple(float(x) for x in np.quantile(values,[.025,.975]))


def score_models(*, X,y,meta,keys, ensembles,thresholds,
                 ordered_features, frozen_scaler, bootstrap_indices,
                 diagnostics_module):
    """Return new ML artifacts ONLY; called after 40-seed primary scoring completed.

    Do not call with raw test bytes; supply verified Phase A arrays/labels.
    """
    import joblib
    require(isinstance(X,np.ndarray) and X.shape==(N,60,63) and
            X.dtype==np.float32 and np.isfinite(X).all(),'ML final input must be verified 5300x60x63')
    require(np.shape(y)==(N,3) and np.isin(y,[0,1]).all(),'Invalid frozen label matrix')
    require(len(meta)==N and len(keys)==N,'ML target and metadata membership mismatch')
    bi=np.asarray(bootstrap_indices)
    require(bi.shape==(B,ND) and np.issubdtype(bi.dtype,np.integer) and
            np.all((bi[:,:48]>=0)&(bi[:,:48]<48)) and
            np.all((bi[:,48:]>=48)&(bi[:,48:]<265)),
            'ML paired bootstrap must match 2000 segment-stratified date samples')
    require(set(ensembles)==set(NN_MODELS) and
            set(thresholds)=={(m,h) for m in NN_MODELS for h in HORIZONS},
            'Changed frozen neural registry or thresholds')
    require(np.array_equal(pd.to_datetime(meta.reference_date).to_numpy(dtype='datetime64[ns]'),
                           pd.to_datetime(keys.Date).to_numpy(dtype='datetime64[ns]')) and
            np.array_equal(meta.ticker.astype(str).to_numpy(),keys.Ticker.astype(str).to_numpy()) and
            np.array_equal(meta.continuity_segment_id.to_numpy(),keys.continuity_segment_id.to_numpy()),
            'F29R4 ML scoring cohort differs from frozen Phase A keys')
    require(np.array_equal(meta.reference_market_session_id.to_numpy(),keys.market_session_id.to_numpy()),
            'F29R4 ML scoring session ID differs from Phase A keys')
    entries, manifest=authenticate_frozen_baselines(ordered_features=ordered_features)
    last=np.asarray(X[:,-1,:],dtype=np.float32)
    require(np.isfinite(last).all(),'Nonfinite ML inference input')
    p_ml={}; hats={}; metrics=[]; pair_persistence=[]; pair_network=[]
    metadata=meta[['ticker','reference_date','reference_market_session_id','continuity_segment_id']].copy()
    for j,h in enumerate(HORIZONS):
        metadata[f'target_{h}d']=y[:,j]
    tickers=meta.ticker.astype(str).to_numpy()
    regimes=meta.regime_name.astype(str).str.lower().to_numpy()
    segments=meta.continuity_segment_id.astype(int).to_numpy()
    quarters=pd.to_datetime(meta.reference_date).dt.to_period('Q').astype(str).to_numpy()
    require(set(regimes)<= {'low','normal','high'} and set(segments)=={0,1},
            'ML diagnostic segment/regime registration differs from sealed metadata')
    # Use F29R3's own scaler-driven, causal day-t return-sign persistence.
    persistence=(diagnostics_module._frozen_scaler_return_raw(X,list(ordered_features),frozen_scaler)>0).astype(np.int8)
    counts_p={h:date_confusion(y[:,j],persistence) for j,h in enumerate(HORIZONS)}
    group_artifacts={g:[] for g in ('stock','regime','segment','calendar_quarter')}
    cal=[]
    for model in ('LOGREG','RANDOM_FOREST','XGBOOST'):
        for j,h in enumerate(HORIZONS):
            estimator=joblib.load(entries[(model,h)])  # SHAs verified immediately above
            require(int(getattr(estimator,'n_features_in_',-1))==63 and
                    list(map(int,estimator.classes_))==[0,1],
                    f'Frozen {model} {h}d estimator feature/class schema drift')
            scores=np.asarray(estimator.predict_proba(last),dtype=np.float64)
            require(scores.shape==(N,2) and np.isfinite(scores).all() and
                    np.all((scores>=0)&(scores<=1)) and
                    np.allclose(scores.sum(axis=1),1,rtol=0,atol=1e-6),
                    f'Invalid frozen {model} {h}d ML probabilities')
            prob=scores[:,1]
            hat=(prob>=0.5).astype(np.int8)
            p_ml[(model,h)]=prob
            hats[(model,h)]=hat
            metadata[f'probability_{model}_{h}d']=prob
            metadata[f'predicted_{model}_{h}d']=hat
            truth=np.asarray(y[:,j],dtype=np.int8)
            overall=diagnostics_module.group_metrics(truth,prob,hat)
            metrics.append({'model_id':model,'horizon_days':h,
               'comparison_role':'SUPPLEMENTARY_POST_FREEZE_FIXED_HYPERPARAMETERS',
               'decision_threshold':0.5,'training_rows':42360,
               **overall})
            cm=date_confusion(truth,hat)
            lo,hi=paired_ci(cm,counts_p[h],bi)
            observed=overall['balanced_accuracy']-diagnostics_module.group_metrics(
                         truth,persistence.astype(float),persistence)['balanced_accuracy']
            pair_persistence.append({'model_id':model,'horizon_days':h,
                'reference':'lagged_return_sign_persistence',
                'observed_BA_difference':float(observed),
                'ci_95_low':lo,'ci_95_high':hi,
                'block_length':20,'replicates':B,'exploratory_unadjusted':True})
            for neural in NN_MODELS:
                neural_p=np.asarray(ensembles[neural],dtype=np.float64)[:,j]
                neural_hat=(neural_p>=float(thresholds[(neural,h)])).astype(np.int8)
                nc=date_confusion(truth,neural_hat)
                lo,hi=paired_ci(nc,cm,bi)
                neural_ba=diagnostics_module.group_metrics(truth,neural_p,neural_hat)['balanced_accuracy']
                pair_network.append({'neural_model':neural,'baseline_model':model,
                    'horizon_days':h,'observed_BA_neural_minus_ML':
                    float(neural_ba-overall['balanced_accuracy']),
                    'ci_95_low':lo,'ci_95_high':hi,'block_length':20,
                    'replicates':B,'exploratory_unadjusted':True})
            filters={'stock':tickers,'regime':regimes,'segment':segments.astype(str),
                     'calendar_quarter':quarters}
            for g,vals in filters.items():
                for name in sorted(set(vals)):
                    mask=(vals==name)
                    group_artifacts[g].append({'model_id':model,'horizon_days':h,
                        'group':name,'exploratory':True,
                        **diagnostics_module.group_metrics(truth[mask],prob[mask],hat[mask])})
            idx=np.minimum(9,np.floor(prob*10).astype(int))
            for b in range(10):
                mask=(idx==b)
                cal.append({'model_id':model,'horizon_days':h,
                    'bin_index':b,'n':int(mask.sum()),
                    'mean_probability':float(prob[mask].mean()) if mask.any() else np.nan,
                    'observed_positive_fraction':float(truth[mask].mean()) if mask.any() else np.nan})
            del estimator
    require(len(metrics)==9 and len(pair_persistence)==9 and len(pair_network)==72 and
            len(group_artifacts['stock'])==180 and len(group_artifacts['regime'])==27 and
            len(group_artifacts['segment'])==18 and len(cal)==90,
            'ML supplementary cardinality mismatch')
    def csv(rows):
        return pd.DataFrame(rows).to_csv(index=False,lineterminator='\n',na_rep='').encode()
    buf=io.BytesIO()
    metadata.to_parquet(buf,index=False,engine='pyarrow')
    artifacts={
      'ml_devonly_predictions.parquet':buf.getvalue(),
      'ml_devonly_classification_metrics.csv':csv(metrics),
      'ml_devonly_vs_persistence_paired_block_ci.csv':csv(pair_persistence),
      'ml_devonly_neural_vs_ml_paired_block_ci.csv':csv(pair_network),
      'ml_devonly_stock_level.csv':csv(group_artifacts['stock']),
      'ml_devonly_regime_level.csv':csv(group_artifacts['regime']),
      'ml_devonly_segment_level.csv':csv(group_artifacts['segment']),
      'ml_devonly_calendar_quarter.csv':csv(group_artifacts['calendar_quarter']),
      'ml_devonly_calibration_10_bins.csv':csv(cal),
    }
    evidence={
      'schema':'STEP36I_F29R5_VERIFIED_DEVONLY_ML_SUPPLEMENT_V1',
      'scope':'POST_FREEZE_EXPLORATORY_NO_FROZEN_PRIMARY_HYPOTHESES_CHANGED',
      'trained_baseline_count':9,'frozen_model_manifest_sha256':MANIFEST_SHA,
      'training_source_sha256':TRAIN_SOURCE_SHA,
      'training_examples':42360,'input_shape':[N,60,63],
      'feature_snapshot':'X[:,-1,:] before any held-out outcome use',
      'decision_threshold':0.5,
      'trained_models':['LOGREG','RANDOM_FOREST','XGBOOST'],
      'horizons':[1,3,5],
      'primary_BH_family_modified':False,
      'paired_bootstrap':'2000 common date samples, 48+217 stratified segments, L=20; exploratory unadjusted percentile CIs',
      'neural_vs_ML_pair_count':72,'ML_vs_persistence_pair_count':9,
      'hyperparameters':'fixed prior to held-out scoring; fitted on full 2016-2024 development only',
      'no_heldout_based_selection':True,
      'no_cross_validation_performance_claim':True,
      'no_economic_return_claim':True,
      'files':[{'name':name,'sha256':hashlib.sha256(contents).hexdigest(),
                'bytes':len(contents)} for name,contents in sorted(artifacts.items())],
    }
    artifacts['ml_devonly_methods_manifest.json']=(json.dumps(evidence,sort_keys=True,indent=2,
                                                         allow_nan=False)+'\n').encode()
    return artifacts, pd.DataFrame(metrics), pd.DataFrame(pair_network)


def development_only_deserialization_smoke(entries):
    """Load genuine nine frozen models on authenticated NON-TEST Step14 dev input.

    The sample is for file/runtime compatibility ONLY. It is *in sample* and
    must not be reported as predictive accuracy or cross-validation.
    """
    import joblib
    xp=EXP/'feature_data/step14_model_arrays/final_development/14_final_development_X_full.npy'
    safe_regular(xp)
    require(sha_file(xp)==STEP14_X_SHA, 'Frozen development-only X data SHA drift')
    stored=np.load(xp,mmap_mode='r',allow_pickle=False)
    require(stored.shape==(42360,60,63) and stored.dtype==np.float32,
            'Development array interface incompatible with F29R4')
    sample=np.asarray(stored[:25,-1,:],dtype=np.float32)
    require(np.isfinite(sample).all(), 'Invalid development-only inference sample')
    for key,path in sorted(entries.items()):
        model=joblib.load(path)  # SHA verified before unpickling
        require(int(getattr(model,'n_features_in_',-1))==63 and
                list(map(int,model.classes_))==[0,1],
                f'Deserialized {key} model has incorrect feature/class interface')
        probability=np.asarray(model.predict_proba(sample),dtype=np.float64)
        require(probability.shape==(25,2) and np.isfinite(probability).all() and
                np.all((probability>=0)&(probability<=1)) and
                np.allclose(probability.sum(axis=1),1,atol=1e-6,rtol=0),
                f'Deserialized {key} model cannot predict on development input')
        del model
    print('PASS — all 9 genuine frozen models deserialize and predict on verified Step14 DEV sample')


def offline_preflight():
    import json
    source=EXP/'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
    require(source.is_file() and sha_file(source)==STEP35_SHA,'Frozen Step35 contract drift')
    contract=json.loads(source.read_bytes())
    ordered=contract['predictor_preprocessing_contract']['ordered_features']
    entries, _=authenticate_frozen_baselines(ordered_features=ordered)
    require(len(entries)==9,'Incomplete ML baseline registry')
    development_only_deserialization_smoke(entries)
    marker=EXP/'manifests/36_final_test_access_consumption_state.json'
    original=marker.read_bytes()
    state=json.loads(original)
    require(state.get('raw_test_filesystem_open_count_completed')==2 and
            state.get('final_test_scored') is False,'Historical access changed')
    require(marker.read_bytes()==original,'Historical access marker mutated')
    print('PASS — F29R4 immutable nine models and 42,360 training endpoints verified')
    print('PASS — frozen 63-feature schema and Step35 contract unchanged')
    print('PASS — historical marker 2 reads, 0 scored; raw test never opened')
    print('NOTICE — no held-out evaluation, no trained baseline cross-validation claim')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline-preflight',action='store_true',required=True)
    parser.parse_args()
    offline_preflight()

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'F29R5 ML STOP: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(2)
