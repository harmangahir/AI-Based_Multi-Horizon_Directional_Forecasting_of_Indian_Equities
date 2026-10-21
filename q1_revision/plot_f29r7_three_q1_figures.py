#!/usr/bin/env python3
"""F29R7: THREE journal figures, drawn solely from existing frozen artifacts.

Figures
-------
01 Probability reliability diagrams: 10 committed calibration bins (neural/ML).
02 Bootstrap sensitivity forest: all 24 models x L=10/20/40, NO resampling.
03 Cohort attrition + continuity: signed Step35 counts and authenticated Phase A dates.

Inputs: an immutable, previously published F29R7 completed_evaluation package;
        the frozen Step35 contract; Phase A metadata + its manifest.
Outputs: PNG (600 dpi default), vector PDF, and SHA-256 provenance JSON.

No sealed raw-test opening, inference, model fitting, label regeneration,
bootstrap recomputation, or recalibration. No example results are substituted.

Run from q1_revision:
    python plot_f29r7_three_q1_figures.py --verify-only
    python plot_f29r7_three_q1_figures.py --dpi 600 --formats png pdf
    python plot_f29r7_three_q1_figures.py --plots calibration bootstrap --dpi 600

Required: numpy, pandas, matplotlib, pyarrow (for Phase A metadata.parquet)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RECEIPT_SHA256 = '637598ea722cf6083a3797d6de467ad91c6d3287fb2819edb92e90ff850af628'
EVAL_MANIFEST_SHA256 = '47e7820dafe9ec8f951f8daa45ce77aae37599d872bc631a4657f77c96903512'
STEP35_SHA256 = '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7'
HORIZONS = (1, 3, 5)
NEURAL = ('DLINEAR', 'NLINEAR', 'PATCHTST', 'TIMESNET',
          'ITRANSFORMER', 'FREDFORMER', 'FRNET', 'MLF')
ML_MODELS = ('LOGREG', 'RANDOM_FOREST', 'XGBOOST')
DISPLAY = {'DLINEAR': 'DLinear', 'NLINEAR': 'NLinear', 'PATCHTST': 'PatchTST',
           'TIMESNET': 'TimesNet', 'ITRANSFORMER': 'iTransformer',
           'FREDFORMER': 'Fredformer', 'FRNET': 'FRNet', 'MLF': 'MLF',
           'LOGREG': 'Logistic Regression', 'RANDOM_FOREST': 'Random Forest',
           'XGBOOST': 'XGBoost'}
# These ORIGINAL counts are protocol constants; verified through the signed
# Step35 SHA, not estimated from the held-out outcomes.
REGISTERED_ENDPOINTS = 6480
REGISTERED_SESSIONS = 324


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Missing or unsafe file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_unique(frame, cols, n, tag):
    require(len(frame) == n and not frame.duplicated(cols).any(),
            f'{tag}: expected {n} unique rows keyed by {cols}, got {len(frame)}')


def numbers(frame, columns, tag):
    for col in columns:
        require(col in frame, f'{tag}: missing column {col}')
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        require(np.isfinite(frame[col].to_numpy(dtype=float)).all(),
                f'{tag}: non-finite value in {col}')


def read_calibration(path, neural):
    df = pd.read_csv(path)
    tag = path.name
    if neural:
        cols = {'model_id','horizon_days','bin_index','bin_count',
                'mean_predicted_probability','observed_positive_fraction',
                'bin_left_inclusive','bin_right'}
        require(cols.issubset(df.columns), f'{tag}: missing {cols-set(df.columns)}')
        df = df.rename(columns={'bin_count':'n_bin',
                                'mean_predicted_probability':'mean_probability',
                                'bin_left_inclusive':'bin_left'})
    else:
        cols = {'model_id','horizon_days','bin_index','n',
                'mean_probability','observed_positive_fraction'}
        require(cols.issubset(df.columns), f'{tag}: missing {cols-set(df.columns)}')
        df = df.rename(columns={'n':'n_bin'})
    check_unique(df, ['model_id','horizon_days','bin_index'],
                 240 if neural else 90, tag)
    numbers(df, ('horizon_days','bin_index','n_bin'), tag)
    require((df.n_bin >= 0).all() and np.equal(df.n_bin, df.n_bin.astype(int)).all(),
            f'{tag}: non-integer or negative bin counts')
    models = set(NEURAL if neural else ML_MODELS)
    require(set(df.model_id)==models and set(df.horizon_days)==set(HORIZONS),
            f'{tag}: unexpected models or horizons')
    require(set(df.bin_index)==set(range(10)), f'{tag}: unexpected bin indices')
    expected_keys = pd.MultiIndex.from_product([models,HORIZONS,range(10)])
    found = pd.MultiIndex.from_frame(df[['model_id','horizon_days','bin_index']])
    require(set(found)==set(expected_keys), f'{tag}: incomplete calibration register')
    sums = df.groupby(['model_id','horizon_days']).n_bin.sum()
    require((sums==5300).all(), f'{tag}: calibration bins do not sum to 5300')
    for key in ('mean_probability','observed_positive_fraction'):
        df[key] = pd.to_numeric(df[key], errors='coerce')
        expected = df.n_bin > 0
        require(df.loc[expected,key].notna().all(), f'{tag}: missing occupied-bin {key}')
        require(df.loc[~expected,key].isna().all(), f'{tag}: nonempty metric for zero-count bin')
        require(df.loc[expected,key].between(0,1).all(), f'{tag}: {key} out of [0,1]')
    valid = df[df.n_bin > 0]
    require(np.all(valid.mean_probability >= valid.bin_index/10-1e-7) and
            np.all(valid.mean_probability <= (valid.bin_index+1)/10+1e-7),
            f'{tag}: calibration probabilities incompatible with 10 fixed bins')
    if neural:
        numbers(df, ('bin_left','bin_right'),tag)
        require(np.allclose(df.bin_left,df.bin_index/10,atol=1e-10) and
                np.allclose(df.bin_right,(df.bin_index+1)/10,atol=1e-10),
                f'{tag}: bin boundary definitions drifted')
    return df.sort_values(['horizon_days','model_id','bin_index']).reset_index(drop=True)


def authenticate_results(exp):
    require(exp.is_dir(), f'Experiment root missing: {exp}')
    receipt = exp/'manifests/36i_amended_final_test_evaluation_manifest.json'
    contract = exp/'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
    result = exp/'final_test_evaluation/step36i_amended_predictions/completed_evaluation'
    manifest_path = result/'evaluation_manifest.json'
    require(sha256(receipt)==RECEIPT_SHA256,'F29R7 published receipt SHA mismatch')
    pub=json.loads(receipt.read_bytes())
    require(pub.get('final_evaluation_manifest_sha256')==EVAL_MANIFEST_SHA256,
            'Published receipt does not bind expected result manifest')
    require(sha256(manifest_path)==EVAL_MANIFEST_SHA256,
            'F29R6 committed evaluation manifest SHA mismatch')
    require(sha256(contract)==STEP35_SHA256,
            'Original Step35 frozen experiment contract SHA mismatch')
    ledger=json.loads(manifest_path.read_bytes())
    require(ledger.get('schema')=='STEP36I_F29R6_FINAL_EVALUATION_REGIME_LABEL_RECOVERY_V1' and
            ledger.get('regime_label_correction')=='ID_1_NORMAL_NOT_MEDIUM',
            'Unexpected F29R6 evaluation identity')
    for key,n in [('metrics',24),('statistics',72),('primary_bh_tests',24),
                  ('ml_baseline_model_count',9)]:
        require(ledger.get(key)==n, f'Manifest count mismatch: {key}')
    entries=ledger.get('artifacts',[])
    require(len(entries)==23 and len(set(e['name'] for e in entries))==23,
            'Expected exactly 23 unique committed result artifacts')
    expected={e['name'] for e in entries}|{'evaluation_manifest.json'}
    require({p.name for p in result.iterdir()}==expected,
            'Unexpected evaluation output entries')
    hashes={}
    for e in entries:
        name=e['name']
        require(Path(name).name==name and name not in ('.','..'),f'Unsafe artifact path {name}')
        f=result/name
        require(f.stat().st_size==int(e['bytes']),f'Artifact size mismatch: {name}')
        actual=sha256(f)
        require(actual==e['sha256'],f'Artifact hash mismatch: {name}')
        hashes[name]=actual
    print('PASS: F29R7 receipt + F29R6 manifest + all 23 immutable artifacts')
    print('PASS: original frozen Step35 contract hash (registered 6480 / 324)')

    neural=pd.read_csv(result/'classification_metrics.csv')
    ml=pd.read_csv(result/'ml_devonly_classification_metrics.csv')
    boot=pd.read_csv(result/'bootstrap_fdr.csv')
    check_unique(neural,['model_id','horizon_days'],24,'neural metrics')
    check_unique(ml,['model_id','horizon_days'],9,'ML metrics')
    check_unique(boot,['model_id','horizon_days','block_length'],72,'bootstrap')
    require(set(neural.model_id)==set(NEURAL) and set(ml.model_id)==set(ML_MODELS)
            and set(neural.horizon_days)==set(HORIZONS), 'Model/horizon inventory mismatch')
    numbers(neural,('horizon_days','n','tp','tn','fp','fn','balanced_accuracy'),
            'neural metrics')
    numbers(ml,('horizon_days','balanced_accuracy'),'ML metrics')
    numbers(boot,('block_length','observed_BA','ci_lower_percentile_95',
                  'ci_upper_percentile_95'),'bootstrap')
    require((neural.n==5300).all() and
            (neural[['tp','tn','fp','fn']].sum(axis=1)==5300).all(),
            'Neural cohort and confusion counts mismatch')
    require(set(boot.block_length)=={10,20,40} and
            set(boot.model_id)==set(NEURAL), 'Bootstrap lengths/model set mismatch')
    primary=boot.query('block_length == 20').merge(
        neural[['model_id','horizon_days','balanced_accuracy']],
        on=['model_id','horizon_days'],validate='one_to_one')
    require(np.allclose(primary.observed_BA,primary.balanced_accuracy,atol=1e-9),
            'Primary bootstrap BA mismatches classification metrics')
    require((boot.ci_lower_percentile_95<=boot.ci_upper_percentile_95).all() and
            boot.ci_lower_percentile_95.between(0,1).all() and
            boot.ci_upper_percentile_95.between(0,1).all(),
            'Invalid frozen bootstrap confidence limits')
    neural_cal=read_calibration(result/'supplementary_calibration_10_fixed_bins.csv',True)
    ml_cal=read_calibration(result/'ml_devonly_calibration_10_bins.csv',False)
    # Cross-validate all 11 model/horizon class prevalences directly against
    # independently recorded classification confusion counts, not outcomes.
    truth={(int(r.horizon_days)):float((r.tp+r.fn)/r.n)
           for r in neural.itertuples() if r.model_id=='DLINEAR'}
    for frame in (neural_cal,ml_cal):
        for (model,h),g in frame.groupby(['model_id','horizon_days']):
            obs=float(np.sum(g.n_bin*g.observed_positive_fraction.fillna(0))/5300)
            require(abs(obs-truth[int(h)])<1e-9,
                    f'Calibration bin prevalence disagrees with frozen labels: {model}/{h}')
    print('PASS: neural 240 and ML 90 authentic calibration-bin records')
    print('PASS: 72 frozen bootstrap rows (L = 10, 20, 40)')

    # Read ONLY Phase A metadata, not the raw sealed final-test dataset.
    phase=exp/'feature_data/step36i_amended_final_test_arrays/phase_a_materialization'
    inner_path=phase/'arrays/phase_a_manifest.json'
    inner=json.loads(inner_path.read_bytes())
    require(inner.get('phase')=='A_MATERIALIZATION' and
            inner.get('step35_sha256')==STEP35_SHA256,
            'Phase A identity / frozen Step35 association mismatch')
    registered={e.get('relative_path'):e for e in inner.get('artifacts',[])}
    rel='arrays/metadata.parquet'
    require(rel in registered, 'Phase A metadata missing from authenticated manifest')
    meta_path=phase/rel
    entry=registered[rel]
    require(meta_path.stat().st_size==int(entry['bytes']) and
            sha256(meta_path)==entry['sha256'], 'Phase A metadata SHA / size mismatch')
    meta=pd.read_parquet(meta_path,columns=['reference_date','ticker','continuity_segment_id'])
    require(len(meta)==5300 and meta.ticker.nunique()==20,
            'Phase A stock-endpoint population mismatch')
    meta['reference_date']=pd.to_datetime(meta.reference_date)
    require(meta.reference_date.notna().all(), 'Invalid Phase A dates')
    require(not meta.duplicated(['ticker','reference_date']).any(),
            'Duplicated Phase A ticker-date endpoints')
    daily=meta.groupby('reference_date').agg(n=('ticker','size'),
                            distinct_segments=('continuity_segment_id','nunique'),
                            segment=('continuity_segment_id','first')).sort_index()
    require(len(daily)==265 and (daily.n==20).all() and
            (daily.distinct_segments==1).all(), 'Daily cross-sectional membership drift')
    require(set(daily.segment)=={0,1} and
            not np.any(np.diff(daily.segment.to_numpy(dtype=int))<0),
            'Unexpected segment IDs/order')
    summary=daily.groupby('segment').agg(sessions=('n','size'),
                        first=('n',lambda s:s.index.min()),
                        last=('n',lambda s:s.index.max()))
    require(list(summary.sessions)==[48,217] and
            summary.loc[0,'first'].strftime('%Y-%m-%d')=='2025-01-01' and
            summary.loc[0,'last'].strftime('%Y-%m-%d')=='2025-03-07' and
            summary.loc[1,'first'].strftime('%Y-%m-%d')=='2025-08-01' and
            summary.loc[1,'last'].strftime('%Y-%m-%d')=='2026-06-22',
            'Phase A continuity-cohort details disagree with frozen amendment')
    gap=(summary.loc[1,'first']-summary.loc[0,'last']).days
    require(gap==147 and REGISTERED_ENDPOINTS-5300==1180 and
            REGISTERED_SESSIONS-265==59,
            'Frozen attrition and discontinuity arithmetic drift')
    print('PASS: authenticated Phase A metadata: 20 x 265 = 5300; 48 + 217 sessions')
    print('PASS: 147-calendar-day date separation; excludes 1180 registered endpoints')
    context={
        'public_receipt_sha256':RECEIPT_SHA256,
        'evaluation_manifest_sha256':EVAL_MANIFEST_SHA256,
        'frozen_step35_contract_sha256':STEP35_SHA256,
        'phase_a_metadata_sha256':sha256(meta_path),
        'committed_artifact_sha256':hashes,
        'original_registered_counts_from_frozen_protocol':{
            'endpoints':REGISTERED_ENDPOINTS,'sessions':REGISTERED_SESSIONS},
        'held_out_authenticated_counts':{
            'endpoints':len(meta),'sessions':len(daily),
            'segment_0_sessions':48,'segment_1_sessions':217,
            'calendar_day_date_separation':gap},
        'calibration_definition':'10 fixed equal-width probability bins; no retuning',
        'sensitivity_definition':'Stored L=10,20,40 percentile CIs; primary inference remains L=20',
    }
    return {'neural':neural,'ml':ml,'bootstrap':boot,'neural_cal':neural_cal,
            'ml_cal':ml_cal,'daily':daily,'summary':summary},context


def styling(font):
    plt.rcParams.update({
       'font.family':'DejaVu Sans',
       'font.size':font,
       'axes.titlesize':font+1.5,
       'axes.titleweight':'semibold',
       'axes.labelsize':font,
       'xtick.labelsize':font-.5,
       'ytick.labelsize':font-.5,
       'legend.fontsize':font-.7,
       'axes.spines.top':False,
       'axes.spines.right':False,
       'axes.grid':False,
       'pdf.fonttype':42,
       'ps.fonttype':42,
       'figure.facecolor':'white',
       'savefig.facecolor':'white',
    })


def fig_calibration(data,width,font,leaders,ml_model):
    fig,axes=plt.subplots(1,3,figsize=(width,4.5),sharex=True,sharey=True)
    series=((data['neural_cal'],'neural'),(data['ml_cal'],'ml'))
    for ax,h,nmodel in zip(axes,HORIZONS,leaders):
        for df,kind in series:
            selected=nmodel if kind=='neural' else ml_model
            g=df[(df.model_id==selected)&(df.horizon_days==h)]
            require(len(g)==10,f'Calibration: missing {selected}/{h} bins')
            valid=g[g.n_bin>0]
            x=valid.mean_probability.to_numpy(dtype=float)
            y=valid.observed_positive_fraction.to_numpy(dtype=float)
            c='#2377ae' if kind=='neural' else '#d5832b'
            marker='o' if kind=='neural' else 's'
            label=DISPLAY[selected] + (' (descriptive horizon leader)' if kind=='neural' else '')
            # No artificial interpolation: connecting segments only between
            # CONSECUTIVE occupied fixed bins (skip empty bins).
            yfull=g.observed_positive_fraction.to_numpy(dtype=float)
            xfull=g.mean_probability.to_numpy(dtype=float)
            ax.plot(xfull,yfull,color=c,linewidth=1.2,alpha=.65,zorder=2)
            ax.scatter(x,y,s=36,marker=marker,color=c,edgecolor='white',
                       linewidth=.5,zorder=3,label=label)
        ax.plot([0,1],[0,1],ls='--',lw=1.1,color='#606060',zorder=1,
                label='Perfect calibration')
        ax.set(xlim=(-.025,1.025),ylim=(-.025,1.025),
               xlabel='Mean predicted up probability')
        ax.set_title(f'{h}-session horizon')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(color='#d3d8dd',alpha=.45,lw=.6)
        ax.set_xticks(np.arange(0,1.01,.2))
        ax.set_yticks(np.arange(0,1.01,.2))
        if h==1: ax.set_ylabel('Observed up fraction in bin')
    # Legend from individual panels differs by horizon; generic, stable keys.
    from matplotlib.lines import Line2D
    legend=[Line2D([0],[0],color='#606060',ls='--',label='Perfect calibration'),
            Line2D([0],[0],marker='o',ls='-',color='#2377ae',
                   label='Neural (selected descriptively)'),
            Line2D([0],[0],marker='s',ls='-',color='#d5832b',
                   label=DISPLAY[ml_model]+' (exploratory)')]
    fig.suptitle('Held-out 10-bin probability reliability',y=.985,weight='semibold')
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.925),
               frameon=False,ncol=3)
    fig.subplots_adjust(left=.07,right=.985,top=.79,bottom=.20,wspace=.24)
    fig.text(.5,.055,'Points are occupied original bins only. Empty bins omitted; gaps not interpolated. '
             'No post-test recalibration or new inference.',ha='center',fontsize=font-1.2)
    return fig


def fig_bootstrap(data,width,font):
    boot=data['bootstrap']
    fig,axes=plt.subplots(1,3,figsize=(width,6.6),sharex=True,sharey=True)
    colors={10:'#98a5ad',20:'#176ba0',40:'#d58a3c'}
    styles={10:'o',20:'D',40:'s'}
    shift={10:-.22,20:0.,40:.22}
    for ax,h in zip(axes,HORIZONS):
        frame=boot.loc[boot.horizon_days==h].set_index(['model_id','block_length'])
        for i,model in enumerate(NEURAL):
            for L in (10,20,40):
                row=frame.loc[(model,L)]
                point=float(row.observed_BA)*100
                lo=float(row.ci_lower_percentile_95)*100
                hi=float(row.ci_upper_percentile_95)*100
                require(lo<=hi, f'Invalid bootstrap CI for {model}/{h}/{L}')
                y=i+shift[L]
                ax.plot([lo,hi],[y,y],lw=1.6,color=colors[L],solid_capstyle='round')
                ax.plot([lo,lo],[y-.055,y+.055],lw=1,color=colors[L])
                ax.plot([hi,hi],[y-.055,y+.055],lw=1,color=colors[L])
                ax.scatter([point],[y],marker=styles[L],s=24,
                           color=colors[L],edgecolor='white',linewidth=.35,zorder=4)
        ax.axvline(50,color='#303030',ls='--',lw=1)
        ax.set_title(f'{h}-session horizon',pad=12)
        ax.set_xlabel('Balanced accuracy (%)')
        ax.set_yticks(range(len(NEURAL)),[DISPLAY[m] for m in NEURAL])
        ax.set_ylim(len(NEURAL)-.5,-.5)
        ax.grid(axis='x',color='#d6d6d6',alpha=.5,lw=.6)
    from matplotlib.lines import Line2D
    legend=[Line2D([0],[0],color=colors[L],marker=styles[L],lw=1.5,
                   markersize=5,label=f'L={L}'+(' (primary)' if L==20 else ' (sensitivity)'))
            for L in (10,20,40)]
    fig.legend(handles=legend,ncol=3,loc='upper center',bbox_to_anchor=(.5,.955),
               frameon=False)
    fig.suptitle('Dependence-aware bootstrap: frozen block-length sensitivity',y=.995)
    fig.subplots_adjust(left=.115,right=.985,top=.84,bottom=.17,wspace=.12)
    fig.text(.5,.055,'Stored 95% percentile intervals, B=2,000 per L. '
             'Only L=20 belongs to the 24 original primary BH-FDR tests; no interval selection.',
             fontsize=font-1.35,ha='center')
    return fig


def fig_attrition(data,width,font):
    smry=data['summary']
    retained=5300
    excluded=REGISTERED_ENDPOINTS-retained
    fig=plt.figure(figsize=(width,7.1))
    grid=fig.add_gridspec(2,1,height_ratios=[1,1.5],hspace=.55,
                         left=.08,right=.98,top=.89,bottom=.13)
    ax=fig.add_subplot(grid[0])
    blue='#226ba3'; gray='#c7cdd2'; amber='#d58a3c'
    ax.barh(0,retained,height=.48,color=blue,label='Retained (eligible)')
    ax.barh(0,excluded,height=.48,left=retained,color=gray,
            edgecolor='#777777',linewidth=.4,label='Excluded by eligibility')
    ax.text(retained/2,0,f'Retained: {retained:,}\n({retained/REGISTERED_ENDPOINTS:.1%})',
            color='white',ha='center',va='center',fontsize=font+.8,fontweight='semibold')
    ax.text(retained+excluded/2,0,f'Excluded: {excluded:,}\n({excluded/REGISTERED_ENDPOINTS:.1%})',
            color='#303030',ha='center',va='center',fontsize=font-.2)
    ax.set_xlim(0,REGISTERED_ENDPOINTS)
    ax.set_ylim(-.62,.75)
    ax.set_yticks([])
    ax.set_xlabel('Originally registered target-eligible stock-session endpoints')
    ax.set_title(f'A  |  Cohort eligibility: {REGISTERED_ENDPOINTS:,} planned '
                 f'({REGISTERED_SESSIONS} sessions) to {retained:,} evaluated (265 sessions)',loc='left')
    ax.grid(axis='x',color='#d8dce0',alpha=.6,lw=.6)
    ax.set_axisbelow(True)

    ax2=fig.add_subplot(grid[1])
    s0a=smry.loc[0,'first']; s0b=smry.loc[0,'last']
    s1a=smry.loc[1,'first']; s1b=smry.loc[1,'last']
    conv=mdates.date2num
    start=conv(s0a); a=conv(s0b); b=conv(s1a); end=conv(s1b)
    y0=.34; height=.24
    ax2.broken_barh([(start,a-start)],(y0,height),facecolors=blue,edgecolors='none')
    ax2.broken_barh([(b,end-b)],(y0,height),facecolors='#257f6b',edgecolors='none')
    ax2.axvspan(a,b,ymin=.22,ymax=.64,facecolor='#f4eee4',alpha=.95,
                hatch='///',edgecolor='#b0a69a',linewidth=.3)
    ax2.text((start+a)/2,.78,'Segment 0: 48 sessions\n960 stock endpoints',
             ha='center',va='center',color=blue,fontweight='semibold',fontsize=font-.5)
    ax2.text((b+end)/2,.78,'Segment 1: 217 sessions\n4,340 stock endpoints',
             ha='center',va='center',color='#176954',fontweight='semibold',fontsize=font-.5)
    ax2.text((a+b)/2,.08,'147-day date separation\nNo evaluated endpoints',
             ha='center',va='center',fontsize=font-.5,color='#665343')
    ax2.set_xlim(conv(pd.Timestamp('2024-12-15')),conv(pd.Timestamp('2026-07-10')))
    ax2.set_ylim(-.1,1.05)
    ax2.set_yticks([])
    ax2.set_title('B  |  Actual authenticated Phase A evaluation dates',loc='left')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.tick_params(axis='x',rotation=25)
    ax2.grid(axis='x',color='#c7c7c7',alpha=.45,lw=.6)
    ax2.set_axisbelow(True)
    fig.suptitle('Amended held-out cohort: eligibility attrition and temporal coverage',y=.98)
    fig.text(.5,.038,'Original 6,480/324 counts are from the SHA-verified frozen Step-35 contract; '
             'retained dates and segments are taken from SHA-verified Phase A metadata.',
             ha='center',fontsize=font-1.5)
    return fig


PLOTS={'calibration':fig_calibration,'bootstrap':fig_bootstrap,'cohort':fig_attrition}
FILENAMES={'calibration':'fig16_probability_reliability',
           'bootstrap':'fig17_bootstrap_block_sensitivity',
           'cohort':'fig18_cohort_attrition_continuity'}


def parse_args():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment-root',type=Path,default=here.parent/
            'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026')
    parser.add_argument('--output-dir',type=Path,default=here/'F29R7_Three_Q1_Figures')
    parser.add_argument('--plots',nargs='+',choices=tuple(PLOTS)+('all',),default=['all'])
    parser.add_argument('--formats',nargs='+',choices=('png','pdf'),default=['png','pdf'])
    parser.add_argument('--dpi',type=int,default=600,
                        help='Raster PNG DPI. PDF stays vector; dpi is passed to raster artists if any.')
    parser.add_argument('--width',type=float,default=14.5,
                        help='Figure width in inches; PDF width uses this physical size.')
    parser.add_argument('--font-size',type=float,default=10.)
    parser.add_argument('--neural-by-horizon',nargs=3,
                        default=['DLINEAR','PATCHTST','ITRANSFORMER'],
                        metavar=('1D','3D','5D'),
                        help='Three neural model names in horizon order; exploratory fixed selection')
    parser.add_argument('--ml-model',choices=ML_MODELS,default='RANDOM_FOREST')
    parser.add_argument('--verify-only',action='store_true',
                        help='Authenticate real inputs; output NOTHING')
    parser.add_argument('--overwrite',action='store_true',
                        help='Explicitly allow overwriting generated figures')
    return parser.parse_args()


def main():
    args=parse_args()
    require(150<=args.dpi<=1200,'DPI outside 150..1200')
    require(11<=args.width<=23 and 8<=args.font_size<=13,
            'Width must be 11..23 in; font size 8..13 pt')
    require(all(m in NEURAL for m in args.neural_by_horizon),
            'Unknown neural model; available: '+', '.join(NEURAL))
    exp=args.experiment_root.expanduser().resolve()
    out=args.output_dir.expanduser().resolve()
    require(exp != out and exp not in out.parents,
            'Never write images inside the immutable experiment root')
    selected=list(PLOTS) if 'all' in args.plots else list(dict.fromkeys(args.plots))
    print('F29R7 VERIFIED SCIENTIFIC FIGURE PRODUCTION')
    print('Input:',exp)
    print('Output:',out)
    print('Figures:',selected)
    data,context=authenticate_results(exp)
    if args.verify_only:
        print('VERIFY-ONLY PASS — no PDF/PNG output or experiment modifications')
        return
    targets=[out/f'{FILENAMES[name]}.{fmt}' for name in selected for fmt in dict.fromkeys(args.formats)]
    report=out/'F29R7_THREE_FIGURES_PROVENANCE.json'
    if not args.overwrite:
        require(not report.exists() and all(not p.exists() for p in targets),
                'Figure files already exist: choose a new --output-dir or pass --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    styling(args.font_size)
    output_hashes={}
    for name in selected:
        if name=='calibration':
            fig=PLOTS[name](data,args.width,args.font_size,
                            tuple(args.neural_by_horizon),args.ml_model)
        else:
            fig=PLOTS[name](data,args.width,args.font_size)
        try:
            for fmt in dict.fromkeys(args.formats):
                dest=out/f'{FILENAMES[name]}.{fmt}'
                fig.savefig(dest,format=fmt,dpi=args.dpi,
                            facecolor='white',bbox_inches='tight',pad_inches=.13)
                output_hashes[dest.name]=sha256(dest)
                print(f'PASS {dest.name}: SHA-256 {output_hashes[dest.name]}')
        finally:
            plt.close(fig)
    report_data={
        'schema':'F29R7_Q1_FIGURES_CALIB_BOOTSTRAP_COHORT_V1',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'sources':context,
        'python_script_sha256':sha256(Path(__file__).resolve()),
        'versions':{'python':sys.version.split()[0],'numpy':np.__version__,
                    'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
        'calibration_neural_models_by_horizon':dict(zip(map(str,HORIZONS),args.neural_by_horizon)),
        'calibration_exploratory_ml_model':args.ml_model,
        'selected_figures':selected,'output_formats':args.formats,
        'png_dpi':args.dpi,'pdf_is_vector':True,
        'generated_figure_sha256':output_hashes,
        'original_hypothesis_registry_unchanged':True,
        'raw_sealed_test_opened':False,
        'training_or_inference_repeated':False,
        'bootstrap_recomputed':False,
        'probabilities_recalibrated':False,
        'note':'Selected neural leaders identified descriptively; these plots are exploratory diagnostics.'
    }
    report.write_text(json.dumps(report_data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('SUCCESS — three-source figure pipeline; PNG/PDF and provenance saved; no reanalysis')


if __name__=='__main__':
    try:
        main()
    except (RuntimeError,ValueError,KeyError,AssertionError,OSError,ImportError) as e:
        print(f'F29R7 PLOTTING FAIL-CLOSED: {type(e).__name__}: {e}',file=sys.stderr)
        raise SystemExit(2)
