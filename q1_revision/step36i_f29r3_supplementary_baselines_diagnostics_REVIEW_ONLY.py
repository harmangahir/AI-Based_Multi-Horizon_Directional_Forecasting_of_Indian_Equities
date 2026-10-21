#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I F29-R3: additional, non-selective descriptive analyses.

No data-path access, GPU, training, parameter tuning, or test acquisition.
Importable pure computation from authenticated Phase A arrays and frozen 40-seed
ensemble outputs. Its CLI is synthetic-only. Observed held-out numbers may
NEVER be used to change architectures, thresholds, baselines, or hypotheses.

PRE-SPECIFIED by Step35: per-stock and regime diagnostics, fixed 10-bin
calibration, the fixed temporal/group metadata. SUPPLEMENTARY AMENDMENT:
constant-up/down and causal day-t return-sign continuation baselines; the
paired BA comparison to lag-sign continuation is descriptive, not new primary
significance evidence and receives NO FDR promotion. No transaction costs or
investment returns are estimated: signed direction outcomes are insufficient.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

MODELS = ('DLINEAR', 'NLINEAR', 'PATCHTST', 'TIMESNET',
          'ITRANSFORMER', 'FREDFORMER', 'FRNET', 'MLF')
HORIZONS = (1, 3, 5)
N = 5300
N_DATES = 265
N_STOCKS = 20
FEATURES = 63
LOOKBACK = 60
BOOT = 2000
PRIMARY_L = 20
BIN_EDGES = tuple(float(x)/10 for x in range(11))
BASELINES = ('always_up', 'always_down', 'lagged_return_sign_persistence')


def require(test: Any, text: str) -> None:
    if not bool(test):
        raise RuntimeError(text)


def _as_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator='\n', na_rep='').encode('utf-8')


def _counts(truth: np.ndarray, prediction: np.ndarray) -> tuple[int,int,int,int]:
    a = np.asarray(truth, dtype=bool)
    b = np.asarray(prediction, dtype=bool)
    return (int(np.count_nonzero(~a & ~b)), int(np.count_nonzero(~a & b)),
            int(np.count_nonzero(a & ~b)), int(np.count_nonzero(a & b)))


def group_metrics(y: np.ndarray, proba: np.ndarray, prediction: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(proba, dtype=np.float64)
    hat = np.asarray(prediction, dtype=np.int8)
    require(y.ndim == 1 and p.shape == y.shape and hat.shape == y.shape and len(y)>0,
            'Invalid diagnostic group vectors')
    require(np.isin(y,[0,1]).all() and np.isin(hat,[0,1]).all()
            and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),
            'Invalid diagnostic probability or label')
    tn, fp, fn, tp = _counts(y,hat)
    has_both = bool((tp+fn)>0 and (tn+fp)>0)
    return dict(n=len(y), n_positive=int(y.sum()),
                observed_positive_rate=float(y.mean()),
                predicted_positive_rate=float(hat.mean()),
                balanced_accuracy=(float(0.5*(tp/(tp+fn)+tn/(tn+fp))) if has_both else np.nan),
                roc_auc=(float(roc_auc_score(y,p)) if has_both else np.nan),
                mcc=(float(matthews_corrcoef(y,hat)) if has_both else np.nan),
                brier_score=float(np.mean((p-y)**2)),
                tp=tp,tn=tn,fp=fp,fn=fn,
                single_class_group=not has_both)


def _bootstrap_ba(c: np.ndarray) -> np.ndarray:
    """Input Bx4 count array [TN,FP,FN,TP]; BA undefined if one class absent."""
    den1=c[:,3]+c[:,2]; den0=c[:,0]+c[:,1]
    require(np.all(den1>0) and np.all(den0>0),
            'Degenerate paired block replicate; scientific review required')
    return 0.5*(c[:,3]/den1+c[:,0]/den0)


def _frozen_scaler_return_raw(X, features, scaler):
    require(len(features)==FEATURES and len(set(features))==FEATURES,
            'Frozen feature list mismatch')
    require(features[5]=='return_1d' and features.index('return_1d')==5,
            'Causal return feature index drift')
    require(getattr(scaler, 'n_features_in_', -1)==FEATURES,
            'Frozen scaler width mismatch')
    if hasattr(scaler,'feature_names_in_'):
        require(list(scaler.feature_names_in_)==list(features),
                'Frozen scaler feature order differs')
    require(hasattr(scaler,'min_') and hasattr(scaler,'scale_'),
            'Expected fitted MinMaxScaler offset and scale')
    # Compare in the stored float32 feature domain: an exactly zero
    # unscaled return is represented as float32(min_[5]), not float64(min_[5]).
    # Avoid accidental >0 from the quantization of a zero return.
    offset=float(np.float32(scaler.min_[5])); scale=float(scaler.scale_[5])
    require(np.isfinite(offset) and np.isfinite(scale) and scale>0,
            'Bad frozen return_1d scaler coefficients')
    raw=(np.asarray(X[:,-1,5],dtype=np.float64)-offset)/scale
    require(raw.shape==(N,) and np.isfinite(raw).all(),
            'Invalid causally reconstructed day-t return')
    return raw


def evaluate_supplementary(*, X, y, meta, keys, ensembles, thresholds,
                           ordered_features, frozen_scaler,
                           paired_bootstrap_date_indices):
    """Use only authenticated F20 Phase A arrays and frozen F1/F4 outputs.

    Returns supplemental artifact bytes and an evidence inventory. Never
    changes frozen primary metrics, 24 hypotheses, checkpoints or thresholds.
    """
    X=np.asarray(X)
    y=np.asarray(y)
    require(X.shape==(N,LOOKBACK,FEATURES) and X.dtype==np.float32 and np.isfinite(X).all(),
            'Finalized Phase A X does not match signed contract')
    require(y.shape==(N,3) and np.isin(y,[0,1]).all(),
            'Finalized binary labels invalid')
    require(isinstance(meta,pd.DataFrame) and len(meta)==N and
            {'ticker','reference_date','regime_id','regime_name',
             'regime_threshold_fit_scope','continuity_segment_id',
             'reference_market_session_id','array_row_index'}.issubset(meta.columns),
            'Missing signed Phase A metadata for diagnostics')
    require(isinstance(keys,pd.DataFrame) and len(keys)==N and
            {'Ticker','Date','continuity_segment_id','market_session_id'}.issubset(keys.columns),
            'Ordered cohort keys absent')
    require(np.array_equal(meta.array_row_index.to_numpy(),np.arange(N)),
            'Phase A row-index drift')
    require(np.array_equal(meta.ticker.astype(str).to_numpy(),
                           keys.Ticker.astype(str).to_numpy()), 'Ticker order mismatch')
    require(np.array_equal(pd.to_datetime(meta.reference_date).to_numpy(dtype='datetime64[ns]'),
                           pd.to_datetime(keys.Date).to_numpy(dtype='datetime64[ns]')),
            'Date key order mismatch')
    require(np.array_equal(meta.continuity_segment_id.to_numpy(),
                           keys.continuity_segment_id.to_numpy()),
            'Continuity segment key order mismatch')
    require(np.array_equal(meta.reference_market_session_id.to_numpy(),
                           keys.market_session_id.to_numpy()),
            'Market session key order mismatch')
    require((meta.regime_threshold_fit_scope=='final_development').all(),
            'Regime thresholds not trained on development only')
    regime=meta.regime_id.to_numpy(dtype=np.int64)
    require(np.isin(regime,(0,1,2)).all(), 'Unknown regime labels')
    expected_regime_names={0:'low',1:'medium',2:'high'}
    require(all(str(n).lower()==expected_regime_names[int(i)]
                for i,n in zip(regime,meta.regime_name)), 'Regime name/id contradiction')
    segment=meta.continuity_segment_id.to_numpy(dtype=np.int64)
    require(np.isin(segment,(0,1)).all(), 'Non-frozen segment')
    dates=pd.to_datetime(meta.reference_date)
    dates_unique=dates.drop_duplicates().to_numpy(dtype='datetime64[ns]')
    require(len(dates_unique)==N_DATES and
            np.array_equal(segment.reshape(N_DATES,N_STOCKS)[:,0],
                           np.r_[np.zeros(48,dtype=np.int64),np.ones(217,dtype=np.int64)]),
            'Segment dates drift')
    require(np.array_equal(dates.to_numpy(dtype='datetime64[ns]').reshape(N_DATES,N_STOCKS),
                           np.repeat(dates_unique[:,None],N_STOCKS,axis=1)),
            'Not 20 stocks per date in date-major order')
    require(meta.ticker.nunique()==20, 'Not 20 prespecified stocks')
    require(set(ensembles)==set(MODELS) and
            set(thresholds)=={(m,h) for m in MODELS for h in HORIZONS},
            'Model/threshold registry changed')
    bi=np.asarray(paired_bootstrap_date_indices)
    require(bi.shape==(BOOT,N_DATES) and np.issubdtype(bi.dtype,np.integer),
            'Wrong 2000x265 paired-bootstrap date index')
    require(np.all((bi[:,:48]>=0)&(bi[:,:48]<48)) and
            np.all((bi[:,48:]>=48)&(bi[:,48:]<265)),
            'Bootstrap crosses segment')

    causal_return=_frozen_scaler_return_raw(X, list(ordered_features), frozen_scaler)
    baseline_hats={
        'always_up':np.ones(N,dtype=np.int8),
        'always_down':np.zeros(N,dtype=np.int8),
        'lagged_return_sign_persistence':(causal_return>0.0).astype(np.int8),
    }
    group_frames={'stock':[], 'regime':[], 'segment':[], 'calendar_quarter':[]}
    baseline_rows=[]; calibration_rows=[]; pair_rows=[]; summary_rows=[]
    ticker=meta.ticker.astype(str).to_numpy()
    quarter=dates.dt.to_period('Q').astype(str).to_numpy()
    groups={
        'stock':[(name,ticker==name) for name in sorted(set(ticker))],
        'regime':[(str(n),regime==n) for n in (0,1,2)],
        'segment':[(str(n),segment==n) for n in (0,1)],
        'calendar_quarter':[(name,quarter==name) for name in sorted(set(quarter))],
    }
    for group_type,blocks in groups.items():
        require(all(np.any(mask) for _,mask in blocks),
                f'Empty {group_type} subgroup should be disclosed explicitly')
    # the same 20-equity date cluster feeds both model and baseline in every block replicate
    for h_index,h in enumerate(HORIZONS):
        truth=y[:,h_index]
        persistence_hat=baseline_hats['lagged_return_sign_persistence']
        persistence_c = np.array([_counts(truth[i*N_STOCKS:(i+1)*N_STOCKS],
                          persistence_hat[i*N_STOCKS:(i+1)*N_STOCKS])
                          for i in range(N_DATES)], dtype=np.int64)
        persistence_bs=_bootstrap_ba(persistence_c[bi].sum(axis=1))
        persistence_obs=group_metrics(truth,persistence_hat.astype(float),persistence_hat)['balanced_accuracy']
        for base,hat in baseline_hats.items():
            baseline_rows.append({'baseline':base,'horizon_days':h,
                  'baseline_role':'POST_FREEZE_SUPPLEMENTARY_DESCRIPTIVE_NO_SELECTION',
                  'causal_information':'day_t_lagged_return_only' if base=='lagged_return_sign_persistence' else 'none',
                  'threshold':'return_1d > 0' if base=='lagged_return_sign_persistence' else 'constant',
                  **group_metrics(truth,hat.astype(float),hat)})
        for model in MODELS:
            p=np.asarray(ensembles[model],dtype=np.float64)
            require(p.shape==(N,3) and np.isfinite(p).all() and
                    ((p>=0)&(p<=1)).all(),f'Invalid ensemble {model}')
            th=float(thresholds[(model,h)])
            require(np.isfinite(th) and 0<=th<=1,'Frozen threshold invalid')
            prob=p[:,h_index]
            prediction=(prob>=th).astype(np.int8)
            overall=group_metrics(truth,prob,prediction)
            summary_rows.append({'model_id':model,'horizon_days':h,**overall})
            for group_type,blocks in groups.items():
                for name,mask in blocks:
                    group_frames[group_type].append({'model_id':model,
                         'horizon_days':h,'group':name,
                         'descriptive_only':True,
                         'frozen_threshold':th,
                         **group_metrics(truth[mask],prob[mask],prediction[mask])})
            bin_index=np.minimum(9,np.floor(10*prob).astype(int))
            for b in range(10):
                mask=(bin_index==b)
                count=int(mask.sum())
                calibration_rows.append({
                     'model_id':model,'horizon_days':h,'bin_index':b,
                     'bin_left_inclusive':BIN_EDGES[b],
                     'bin_right':BIN_EDGES[b+1],
                     'last_bin_right_inclusive':b==9,
                     'bin_count':count,
                     'mean_predicted_probability':float(prob[mask].mean()) if count else np.nan,
                     'observed_positive_fraction':float(truth[mask].mean()) if count else np.nan})
            model_c=np.array([_counts(truth[i*N_STOCKS:(i+1)*N_STOCKS],
                           prediction[i*N_STOCKS:(i+1)*N_STOCKS])
                           for i in range(N_DATES)],dtype=np.int64)
            model_bs=_bootstrap_ba(model_c[bi].sum(axis=1))
            delta=model_bs-persistence_bs
            lo,high=np.quantile(delta,(0.025,0.975))
            pair_rows.append({'model_id':model,'horizon_days':h,
                  'comparison_baseline':'lagged_return_sign_persistence',
                  'BA_difference_model_minus_persistence':float(overall['balanced_accuracy']-persistence_obs),
                  'paired_segment_stratified_block_length':PRIMARY_L,
                  'paired_bootstrap_replicates':BOOT,
                  'bootstrap_ci_lower_percentile_95':float(lo),
                  'bootstrap_ci_upper_percentile_95':float(high),
                  'exploratory_no_primary_fdr_or_model_selection':True})
    require(len(baseline_rows)==9 and len(summary_rows)==24 and len(pair_rows)==24 and
            len(calibration_rows)==240 and len(group_frames['stock'])==480 and
            len(group_frames['regime'])==72 and len(group_frames['segment'])==48 and
            len(group_frames['calendar_quarter'])==len(groups['calendar_quarter'])*24,
            'Supplementary diagnostic table cardinality drift')
    artifacts={
        'supplementary_baselines.csv':_as_csv(pd.DataFrame(baseline_rows)),
        'supplementary_model_vs_persistence_paired_block_ci.csv':_as_csv(pd.DataFrame(pair_rows)),
        'supplementary_stock_level.csv':_as_csv(pd.DataFrame(group_frames['stock'])),
        'supplementary_regime_level.csv':_as_csv(pd.DataFrame(group_frames['regime'])),
        'supplementary_segment_level.csv':_as_csv(pd.DataFrame(group_frames['segment'])),
        'supplementary_calendar_quarter.csv':_as_csv(pd.DataFrame(group_frames['calendar_quarter'])),
        'supplementary_calibration_10_fixed_bins.csv':_as_csv(pd.DataFrame(calibration_rows)),
    }
    evidence={
        'schema':'STEP36I_F29R3_SUPPLEMENTARY_ANALYSIS_V1',
        'analysis_status':'POST_FREEZE_SUPPLEMENTARY_EXPLORATORY; NO PRIMARY HYPOTHESES MODIFIED',
        'baseline_definitions':{
          'always_up':'constant binary up prediction, irrespective of horizon',
          'always_down':'constant binary down prediction, irrespective of horizon',
          'lagged_return_sign_persistence':'predict each future horizon with sign(return_1d at reference day t); zero treated as down; return reconstructed from frozen development-only MinMaxScaler, not future labels'},
        'frozen_primary_model_hypotheses':24,
        'paired_block_ci':'2000 common date-cluster replicates, segment-stratified, L=20; descriptive unadjusted percentile CI, no formal p/q or promotion to primary family',
        'single_class_subgroup_policy':'undefined BA/AUC/MCC => empty CSV cells, flag single_class_group=true',
        'regime_threshold_fit_scope':'final_development',
        'calibration':'10 bins [0,.1),...,[.9,1]; no recalibration',
        'economic_return_or_transaction_cost_analysis':'NOT_AVAILABLE_FROM_DIRECTIONAL_LABELS; NOT PERFORMED',
        'development_trained_conventional_ML_baselines':'NOT_IMPLEMENTED; no ex-post fitting or test-guided tuning',
        'leakage_guard':'no features after reference day t; no future labels used to form baseline',
        'summary_check':summary_rows,
        'artifacts':[{'file':name,'sha256':hashlib.sha256(data).hexdigest(),
                      'bytes':len(data)} for name,data in sorted(artifacts.items())],
    }
    # Verify internal group means and BA against full cohort (single-class groups may be undefined)
    assert len(evidence['summary_check'])==24
    artifacts['supplementary_methods_manifest.json']=(json.dumps(evidence,sort_keys=True,
                        indent=2,allow_nan=False)+'\n').encode('utf-8')
    return artifacts, pd.DataFrame(summary_rows)


def synthetic_selftest():
    from types import SimpleNamespace
    rng=np.random.default_rng(29411)
    X=np.zeros((N,LOOKBACK,FEATURES),dtype=np.float32)
    # mocked signed development-only scaler has return feature scale=2, min=0
    X[:,-1,5]=np.where(np.arange(N)%4==0,-0.8,1.2).astype(np.float32)
    sc=SimpleNamespace(n_features_in_=FEATURES,min_=np.zeros(63,dtype=float),
         scale_=np.r_[[1.0]*5,[2.0],[1.0]*57])
    features=[f'f{i}' for i in range(FEATURES)];features[5]='return_1d'
    base=pd.bdate_range('2025-01-01',periods=265)
    dates=np.repeat(base.to_numpy(),N_STOCKS)
    tick=np.tile([f'STOCK{i:02}' for i in range(20)],N_DATES)
    seg=np.repeat(np.r_[np.zeros(48,dtype=int),np.ones(217,dtype=int)],20)
    reg=np.arange(N)%3
    meta=pd.DataFrame({'array_row_index':np.arange(N),'ticker':tick,
       'reference_date':dates,'reference_market_session_id':np.repeat(np.arange(265),20),
       'continuity_segment_id':seg,'regime_id':reg,
       'regime_name':[('low','medium','high')[a] for a in reg],
       'regime_threshold_fit_scope':['final_development']*N})
    keys=pd.DataFrame({'Ticker':tick,'Date':dates,
       'market_session_id':meta.reference_market_session_id,
       'continuity_segment_id':seg})
    y=rng.integers(0,2,size=(N,3),dtype=np.int8)
    ensembles={m:rng.random(size=(N,3)) for m in MODELS}
    thresholds={(m,h):0.5 for m in MODELS for h in HORIZONS}
    # 2000 fixed, segment-preserving synthetic date samples
    bi=np.tile(np.arange(265,dtype=np.int32),(2000,1))
    artifacts,summ=evaluate_supplementary(X=X,y=y,meta=meta,keys=keys,
        ensembles=ensembles,thresholds=thresholds,ordered_features=features,
        frozen_scaler=sc,paired_bootstrap_date_indices=bi)
    require(len(artifacts)==8 and len(summ)==24,'Supplementary synthetic output drift')
    baseline=pd.read_csv(io.BytesIO(artifacts['supplementary_baselines.csv']))
    require(len(baseline)==9 and baseline.query("baseline == 'always_up'").balanced_accuracy.eq(.5).all(),
            'Constant-up BA oracle failed')
    require(baseline.query("baseline == 'always_down'").balanced_accuracy.eq(.5).all(),
            'Constant-down BA oracle failed')
    cal=pd.read_csv(io.BytesIO(artifacts['supplementary_calibration_10_fixed_bins.csv']))
    require(cal.groupby(['model_id','horizon_days']).bin_count.sum().eq(N).all(),
            'Calibration bins do not partition test')
    # Label permutation must leave baseline predictions unchanged
    ret=_frozen_scaler_return_raw(X,features,sc)
    require((ret>0).sum()==N*3//4,'Inverse scaler causal return reconstruction oracle failed')
    # Exact signed zero must NOT be classified up after float32 quantization.
    raw_scaled=X[:,-1,5].copy()
    X[:,-1,5]=np.float32(0.)
    neutral=_frozen_scaler_return_raw(X,features,sc)
    require(np.all(neutral==0),'Zero-return quantization oracle failed')
    X[:,-1,5]=raw_scaled
    return 'PASS — stock/regime/segment/quarter/calibration, persistence and paired CI fixture'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--offline-selftest',action='store_true')
    x=p.parse_args()
    require(x.offline_selftest,'Only --offline-selftest is supported')
    print(synthetic_selftest())
    print('No sealed test opened; no real artifacts created; no production permission')

if __name__=='__main__':
    main()
