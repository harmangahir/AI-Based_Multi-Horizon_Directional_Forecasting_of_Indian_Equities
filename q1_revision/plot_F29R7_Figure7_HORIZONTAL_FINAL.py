#!/usr/bin/env python3
"""F29R7: THREE journal figures, drawn solely from existing frozen artifacts.

Figures
-------
01 Probability reliability diagrams: 10 committed bins with exact n per bin (neural/ML).
02 Bootstrap sensitivity forest: all 24 models x L=10/20/40, NO resampling.
03 Cohort attrition + continuity: signed Step35 counts and authenticated Phase A dates.

Inputs: an immutable, previously published F29R7 completed_evaluation package;
        the frozen Step35 contract; Phase A metadata + its manifest.
Outputs: PNG (600 dpi default), vector PDF, and SHA-256 provenance JSON.

No sealed raw-test opening, inference, model fitting, label regeneration,
bootstrap recomputation, or recalibration. No example results are substituted.

Run from q1_revision:
    python plot_f29r7_three_q1_figures_Q1_CORRECTED.py --verify-only
    python plot_f29r7_three_q1_figures_Q1_CORRECTED.py --dpi 600 --formats png pdf
    python plot_f29r7_three_q1_figures_Q1_CORRECTED.py --plots cohort \
        --formats pdf png --dpi 600 --width 8.2 --font-size 11 \
        --output-dir F29R7_Figure2_Q1_Refined
    python plot_f29r7_three_q1_figures_Q1_CORRECTED.py --plots calibration bootstrap --dpi 600

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
# These ORIGINAL counts are protocol constants; verified through the frozen
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


def fig_calibration(data, width, font, leaders, ml_model):
    """Figure 13: authenticated 10-bin reliability AND exact occupied-bin counts.

    The data are the already-committed calibration aggregates. Show exactly
    the recorded n_bin value for every fixed-width bin (zero denotes empty).
    No probabilities, outcomes, bin edges, models, or test metrics are changed.
    No line joins non-consecutive occupied bins; no recalibration is performed.
    """
    neural_cal, ml_cal = data['neural_cal'], data['ml_cal']
    total_n = 5300
    blue, orange, gray = '#176b9a', '#c57b31', '#555f69'
    grid_color = '#e0e5e9'

    # Horizontal panels; each has a reliability curve above two small tables:
    # bins 0-50% and 50-100%. Splitting into 5-bin halves makes exact
    # integer counts (including 5,300) legible at journal-page dimensions.
    fig = plt.figure(figsize=(width, width * .68))
    grids = fig.add_gridspec(
        3, 3, height_ratios=[4.7, 1.08, 1.08],
        left=.087, right=.985, bottom=.115, top=.780,
        wspace=.24, hspace=.31,
    )
    reports = {}

    def draw_counts_table(ax, bins, counts_neural, counts_ml, neural_name):
        """Exact five-bin count table, no aggregation, interpolation or rounding."""
        ax.set_axis_off()
        # A small heading distinguishes counts from percentages in plot.
        labels = [f'{10*b}-{10*(b+1)}' for b in bins]
        table = ax.table(
            cellText=[
                ['Neural'] + [f'{int(counts_neural[b]):,}' for b in bins],
                ['RF'] + [f'{int(counts_ml[b]):,}' for b in bins],
            ],
            colLabels=['Bin (%)'] + labels,
            colWidths=[.18] + [.164] * 5,
            cellLoc='center', colLoc='center',
            loc='center', bbox=[0.0, 0.0, 1.0, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font - 2.0)
        # Each cell carries an exact integer; 0 explicitly marks an empty bin.
        # Only use subtle tints to distinguish the two model families.
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#d4dbe0')
            cell.set_linewidth(.45)
            if r == 0:
                cell.set_facecolor('#eef1f4')
                cell.get_text().set_color('#3d4751')
                cell.get_text().set_fontweight('semibold')
                cell.get_text().set_fontsize(font - 2.85)
            elif c == 0:
                cell.set_facecolor('white')
                cell.get_text().set_color(blue if r == 1 else orange)
                cell.get_text().set_fontweight('semibold')
                cell.get_text().set_fontsize(font - 1.8)
            else:
                cell.set_facecolor('#eaf3f9' if r == 1 else '#fff1e4')
                cell.get_text().set_color('#263744')
                if cell.get_text().get_text() == '0':
                    cell.get_text().set_color('#7c8791')
        return table

    for j, (horizon, neural_model) in enumerate(zip(HORIZONS, leaders)):
        ax = fig.add_subplot(grids[0, j])
        counts = {}
        for frame, model, color, marker in (
            (neural_cal, neural_model, blue, 'o'),
            (ml_cal, ml_model, orange, 's'),
        ):
            g = frame.loc[(frame.model_id == model) &
                          (frame.horizon_days == horizon)].sort_values('bin_index')
            require(len(g) == 10 and np.array_equal(
                g.bin_index.to_numpy(dtype=int), np.arange(10)
            ), f'Calibration missing ten fixed bins: {model}/{horizon}')
            n = g.n_bin.to_numpy(dtype=int)
            require(np.all(n >= 0) and n.sum() == total_n,
                    f'Invalid original bin counts: {model}/{horizon}')
            occupied = n > 0
            x = g.mean_probability.to_numpy(dtype=float)
            y = g.observed_positive_fraction.to_numpy(dtype=float)
            require(np.isfinite(x[occupied]).all() and
                    np.isfinite(y[occupied]).all() and
                    np.isnan(x[~occupied]).all() and
                    np.isnan(y[~occupied]).all(),
                    f'Original empty/occupied bin data corrupted: {model}/{horizon}')
            require(np.all((0 <= x[occupied]) & (x[occupied] <= 1)) and
                    np.all((0 <= y[occupied]) & (y[occupied] <= 1)),
                    f'Invalid reliability values: {model}/{horizon}')
            # Draw only between genuinely ADJACENT occupied original bins.
            for k in range(9):
                if occupied[k] and occupied[k+1]:
                    ax.plot([x[k], x[k+1]], [y[k], y[k+1]],
                            color=color, lw=1.5, alpha=.8, zorder=3)
            ax.scatter(x[occupied], y[occupied], color=color,
                       marker=marker, s=53, edgecolor='white',
                       linewidth=.65, zorder=4)
            counts[model] = n
        ax.plot([0, 1], [0, 1], color=gray, ls=(0, (4, 3)),
                lw=1.15, zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, .2))
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(color=grid_color, lw=.55, alpha=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', length=0, pad=4, labelsize=font-1)
        ax.set_xlabel('Mean predicted upward probability',
                      fontsize=font-1.1, labelpad=6)
        if j == 0:
            ax.set_ylabel('Observed upward fraction',
                          fontsize=font-1.1, labelpad=6)
        else:
            ax.tick_params(axis='y', labelleft=False)
        ax.set_title(
            f'{chr(65+j)}  |  {horizon}-session: {DISPLAY[neural_model]}',
            fontsize=font+.1, pad=9,
        )

        draw_counts_table(fig.add_subplot(grids[1, j]),
                          range(0, 5), counts[neural_model],
                          counts[ml_model], neural_model)
        draw_counts_table(fig.add_subplot(grids[2, j]),
                          range(5, 10), counts[neural_model],
                          counts[ml_model], neural_model)
        reports[str(horizon)] = {
            model: [int(v) for v in n] for model, n in counts.items()
        }

    from matplotlib.lines import Line2D
    legend = [
        Line2D([], [], marker='o', color=blue, markersize=6,
               label='Neural (descriptive horizon leader)'),
        Line2D([], [], marker='s', color=orange, markersize=6,
               label=f'{DISPLAY[ml_model]} (exploratory)'),
        Line2D([], [], color=gray, ls=(0, (4, 3)),
               label='Perfect calibration'),
    ]
    fig.suptitle('Held-out probability reliability with exact calibration-bin counts',
                 x=.5, y=.966, fontsize=font+1.45, fontweight='semibold')
    fig.legend(handles=legend, loc='upper center',
               bbox_to_anchor=(.5, .886), frameon=False, ncol=3,
               fontsize=font-1.05, columnspacing=1.4, handletextpad=.4)
    fig.text(.5, .080,
             'Tables: actual observations in each fixed probability bin (% ranges); '
             'each model has 5,300 endpoints per horizon.',
             ha='center', fontsize=font-1.10, color='#4f5964')
    fig.text(.5, .044,
             '0 = empty bin. Curve segments link only adjacent occupied bins. '
             'No interpolation, recalibration, or new inference.',
             ha='center', fontsize=font-1.15, color='#4f5964')
    # Store count source records on figure for optional provenance reporting.
    fig.f29r7_count_records = reports
    return fig


def fig_bootstrap(data, width, font):
    """FIGURE 7 — HORIZONTAL (1 row x 3 horizons), archival bootstrap only.

    Displays exactly 72 previously committed model/horizon/block-length rows.
    No resampling, prediction, checkpoint access, or test reinterpretation.
    """
    boot = data['bootstrap']
    required = {'model_id', 'horizon_days', 'block_length',
                'observed_BA', 'ci_lower_percentile_95',
                'ci_upper_percentile_95'}
    require(required.issubset(boot.columns),
            f'Bootstrap columns missing: {required - set(boot.columns)}')
    check_unique(boot, ['model_id', 'horizon_days', 'block_length'],
                 72, 'committed bootstrap rows')
    require(set(boot.model_id) == set(NEURAL) and
            set(boot.horizon_days) == set(HORIZONS) and
            set(boot.block_length) == {10, 20, 40},
            'Bootstrap model/horizon/block-length inventory differs')

    for (_, _), group in boot.groupby(['model_id', 'horizon_days']):
        require(np.ptp(group.observed_BA.to_numpy(dtype=float)) <= 1e-9,
                'Observed BA is inconsistent between block lengths')
    lows = boot.ci_lower_percentile_95.to_numpy(dtype=float) * 100
    highs = boot.ci_upper_percentile_95.to_numpy(dtype=float) * 100
    require(np.isfinite(lows).all() and np.isfinite(highs).all() and
            np.all(lows <= highs), 'Invalid stored confidence intervals')
    # Same x limits in all three panels, without clipping the stored CIs.
    xmin = min(44., 2 * np.floor((min(50., lows.min()) - .5) / 2))
    xmax = max(56., 2 * np.ceil((max(50., highs.max()) + .5) / 2))
    tick_step = 2 if xmax - xmin <= 16 else 4

    # REQUIRED LANDSCAPE LAYOUT: one row, three columns.
    fig, axes = plt.subplots(1, 3, figsize=(width, width * .43), sharey=True)
    fig.subplots_adjust(left=.148, right=.985, top=.755,
                        bottom=.188, wspace=.16)
    colors = {10: '#89969f', 20: '#176b9a', 40: '#c27632'}
    markers = {10: 'o', 20: 'D', 40: 's'}
    shifts = {10: -.21, 20: 0, 40: .21}

    for panel, (ax, horizon) in enumerate(zip(axes, HORIZONS)):
        rows = boot.loc[boot.horizon_days == horizon].set_index(
            ['model_id', 'block_length'])
        ax.set_facecolor('white')
        ax.set_axisbelow(True)
        for i, model in enumerate(NEURAL):
            ax.axhline(i, color='#e8ebee', lw=.55, zorder=0)
            for L in (10, 20, 40):
                r = rows.loc[(model, L)]
                point = float(r.observed_BA) * 100
                lo = float(r.ci_lower_percentile_95) * 100
                hi = float(r.ci_upper_percentile_95) * 100
                y = i + shifts[L]
                primary = L == 20
                ax.plot((lo, hi), (y, y), color=colors[L],
                        lw=2.0 if primary else 1.35,
                        solid_capstyle='round', zorder=3 if primary else 2)
                cap = .074
                ax.plot((lo, lo), (y-cap, y+cap), color=colors[L],
                        lw=1.1 if primary else .85, zorder=3)
                ax.plot((hi, hi), (y-cap, y+cap), color=colors[L],
                        lw=1.1 if primary else .85, zorder=3)
                ax.scatter(point, y, marker=markers[L], color=colors[L],
                           s=36 if primary else 24, edgecolor='white',
                           linewidth=.45, zorder=5 if primary else 4)
        ax.axvline(50, color='#3d4247', ls=(0, (4, 3)), lw=1.1, zorder=1)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(len(NEURAL)-.53, -.52)
        ax.set_yticks(range(len(NEURAL)))
        ax.set_yticklabels([DISPLAY[m] for m in NEURAL])
        ax.set_xticks(np.arange(xmin, xmax+.1, tick_step))
        ax.tick_params(axis='x', length=0, labelsize=font-1.0, pad=5)
        ax.tick_params(axis='y', length=0, labelsize=font-.9, pad=6)
        if panel > 0:
            ax.tick_params(axis='y', left=False, labelleft=False)
        ax.grid(axis='x', color='#d7dce1', lw=.6, alpha=.85)
        ax.set_title(f'{chr(65+panel)}  |  {horizon}-session',
                     loc='left', pad=9, fontsize=font+.1, fontweight='semibold')
        ax.set_xlabel('Balanced accuracy (%)', labelpad=7, fontsize=font-.45)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#a5adb4')
        ax.spines['bottom'].set_linewidth(.6)

    from matplotlib.lines import Line2D
    legend = [Line2D([], [], color=colors[L], marker=markers[L],
                     lw=2. if L==20 else 1.4,
                     markeredgecolor='white', markersize=6.,
                     label=f'L = {L}'+(' (primary)' if L==20 else ' (sensitivity)'))
              for L in (10,20,40)]
    legend.append(Line2D([], [], color='#3d4247', ls=(0, (4, 3)),
                         lw=1.1, label='BA = 50%'))
    fig.suptitle('Dependence-aware bootstrap: confidence-interval sensitivity',
                 x=.5, y=.969, fontsize=font+1.45, fontweight='semibold')
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .889),
               ncol=4, frameon=False, fontsize=font-1.1,
               handlelength=1.5, handletextpad=.35, columnspacing=1.15)
    fig.text(.5, .065,
             'Stored 95% percentile CIs (B = 2,000); L = 20 is primary; '
             'L = 10 and 40 are sensitivity checks only.',
             ha='center', fontsize=font-1.2, color='#525a62')
    return fig


def fig_attrition(data, width, font):
    """Figure 2: publication-oriented cohort attrition and date coverage.

    Values and endpoint dates come ONLY from authenticate_results(), which
    validates the committed frozen contract and Phase A metadata. The figure
    does not read test labels, regenerate models or change eligibility.
    """
    smry = data['summary']
    retained = 5300
    excluded = REGISTERED_ENDPOINTS - retained
    require(REGISTERED_ENDPOINTS == 6480 and REGISTERED_SESSIONS == 324,
            'Original registered cohort constants differ')
    require(retained == 5300 and excluded == 1180,
            'Frozen attrition arithmetic differs')
    require(int(smry.loc[0, 'sessions']) == 48 and
            int(smry.loc[1, 'sessions']) == 217,
            'Frozen continuity-segment session counts differ')

    blue = '#256b9e'
    gray = '#c5cbd1'
    green = '#23806a'
    neutral = '#4b5563'

    # Drawing at final journal width (e.g., 8.2 inches) retains readable
    # typography. Unlike merely increasing DPI, fonts are sized in points.
    fig = plt.figure(figsize=(width, width * .63))
    grid = fig.add_gridspec(2, 1,
                           height_ratios=[1.03, 1.55],
                           left=.095, right=.975,
                           top=.815, bottom=.145,
                           hspace=.72)

    # Panel A: arithmetic is taken from the original contract and amended
    # evaluation metadata, and is deliberately not inferred from outcomes.
    ax = fig.add_subplot(grid[0])
    ax.barh(0, retained, height=.48, color=blue, linewidth=0)
    ax.barh(0, excluded, height=.48, left=retained,
            color=gray, edgecolor='#7a8790', linewidth=.5)
    ax.text(retained / 2, 0,
            f'Retained\n{retained:,} ({retained / REGISTERED_ENDPOINTS:.1%})',
            color='white', ha='center', va='center',
            fontsize=font-.1, fontweight='semibold')
    ax.text(retained + excluded / 2, 0,
            f'Excluded\n{excluded:,} ({excluded / REGISTERED_ENDPOINTS:.1%})',
            color='#24323d', ha='center', va='center',
            fontsize=font-1.15, fontweight='semibold')
    ax.set_xlim(0, REGISTERED_ENDPOINTS)
    ax.set_ylim(-.66, .72)
    ax.set_yticks([])
    ax.set_xticks([0, 1500, 3000, 4500, 6000])
    ax.set_xlabel('Originally registered target-eligible stock-session endpoints',
                  labelpad=6)
    ax.set_title('A  |  Eligibility attrition: 6,480 planned to 5,300 evaluated',
                 loc='left', pad=13, fontsize=font+.5)
    ax.grid(axis='x', color='#d6dce1', alpha=.75, lw=.65)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', length=0)

    # Panel B: two disconnected evaluation intervals, not a continuous
    # eighteen-month backtest. The source dates are taken from authenticated
    # sequence-eligibility metadata, not manually specified plot labels.
    ax2 = fig.add_subplot(grid[1])
    s0a = pd.Timestamp(smry.loc[0, 'first'])
    s0b = pd.Timestamp(smry.loc[0, 'last'])
    s1a = pd.Timestamp(smry.loc[1, 'first'])
    s1b = pd.Timestamp(smry.loc[1, 'last'])
    conv = mdates.date2num
    start, a, b, end = (conv(d) for d in (s0a, s0b, s1a, s1b))
    require((s1a - s0b).days == 147, 'Date separation differs from frozen cohort')

    y0, band_h = .37, .22
    # Add one day to give the last scored date a visible inclusive interval.
    ax2.broken_barh([(start, a-start+1)], (y0, band_h),
                    facecolors=blue, edgecolors='none', zorder=3)
    ax2.broken_barh([(b, end-b+1)], (y0, band_h),
                    facecolors=green, edgecolors='none', zorder=3)
    ax2.axvspan(a+1, b, ymin=.26, ymax=.64,
                facecolor='#f5efe5', hatch='///',
                edgecolor='#c5b7a4', linewidth=.35, zorder=1)

    ax2.text((start+a)/2, .85,
             'Segment 0\n48 sessions / 960 endpoints',
             ha='center', va='center', color=blue,
             fontweight='semibold', fontsize=font-.7)
    ax2.text((b+end)/2, .85,
             'Segment 1\n217 sessions / 4,340 endpoints',
             ha='center', va='center', color=green,
             fontweight='semibold', fontsize=font-.7)
    ax2.text((a+b)/2, .065,
             '147-day date separation\nNo scored endpoints between',
             ha='center', va='center', fontsize=font-1.15,
             color='#665343')

    ax2.set_xlim(conv(pd.Timestamp('2024-12-15')),
                 conv(pd.Timestamp('2026-07-10')))
    ax2.set_ylim(-.08, 1.06)
    ax2.set_yticks([])
    ax2.set_title('B  |  Observed held-out evaluation dates',
                  loc='left', pad=12, fontsize=font+.5)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax2.tick_params(axis='x', length=0, pad=5)
    ax2.grid(axis='x', color='#d6dce1', alpha=.65, lw=.6)
    ax2.set_axisbelow(True)
    ax2.spines['left'].set_visible(False)

    fig.suptitle('Held-out cohort: eligibility attrition and temporal coverage',
                 x=.5, y=.965, fontsize=font+2, fontweight='semibold')
    fig.text(.5, .044,
             'The intervals show scored reference dates only; the separation is not interpolated.',
             ha='center', color=neutral, fontsize=font-1.35)
    return fig


PLOTS={'calibration':fig_calibration,'bootstrap':fig_bootstrap,'cohort':fig_attrition}
FILENAMES={'calibration':'Figure13_Probability_Reliability_HORIZONTAL',
           'bootstrap':'fig17_bootstrap_block_sensitivity_HORIZONTAL',
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
    parser.add_argument('--width',type=float,default=11.5,
                        help='Figure width in inches; PDF width uses this physical size.')
    parser.add_argument('--font-size',type=float,default=12.)
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
    require(7<=args.width<=23 and 8<=args.font_size<=13,
            'Width must be 7..23 in; font size 8..13 pt')
    require(all(m in NEURAL for m in args.neural_by_horizon),
            'Unknown neural model; available: '+', '.join(NEURAL))
    exp=args.experiment_root.expanduser().resolve()
    out=args.output_dir.expanduser().resolve()
    require(exp != out and exp not in out.parents,
            'Never write images inside the immutable experiment root')
    selected=list(PLOTS) if 'all' in args.plots else list(dict.fromkeys(args.plots))
    print('F29R7 VERIFIED FIGURES — CALIBRATION / BOOTSTRAP / COHORT; HORIZONTAL PANELS')
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
    calibration_counts={}
    for name in selected:
        if name=='calibration':
            fig=PLOTS[name](data,args.width,args.font_size,
                            tuple(args.neural_by_horizon),args.ml_model)
            calibration_counts=fig.f29r7_count_records
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
        'png_dpi':args.dpi,'pdf_is_vector':True,'panel_layout':'1_row_by_3_columns',
        'generated_figure_sha256':output_hashes,
        'displayed_original_calibration_counts_by_bin_0_to_9':calibration_counts,
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
