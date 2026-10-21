#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F13C — Non-label calendar coverage and statistical design risk review.

This auditor uses only Step10 identity/eligibility columns and the signed candidate
identity CSV. It reads neither sealed final-test parquet nor feature columns,
labels, regimes, model outputs, or performance metrics. Scientific approval is NOT
conferred by a PASS. File system writes: none.

Use: PYTHONDONTWRITEBYTECODE=1 python step36i_f13c_temporal_coverage_review_READ_ONLY.py --offline-preflight
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

DENIED = {'--execute', '--production', '--read-test', '--score-test', '--approve',
          '--authorize', '--write', '--commit', '--run', '--resume', '--final-test'}
if any(v in DENIED for v in sys.argv[1:]):
    raise SystemExit('DENIED: non-label calendar-coverage review only')

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
LEDGER = EXP / 'feature_data/step10_chronological_splits/10_split_membership.parquet'
CANDIDATE = EXP / 'audits/step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv'
STEP35 = EXP / 'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
ACCESS = EXP / 'manifests/36_final_test_access_consumption_state.json'
HASHES = {
    LEDGER: '82772196c92e88f1926a75a70c42f5f19119d7245d93b8364455fd28642ee121',
    CANDIDATE: '4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65',
    STEP35: '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7',
}
KEYS = ['Ticker', 'Date', 'market_session_id', 'continuity_segment_id']
FLAG = 'outer_final_test_common_multitask'
MODELS = 8
STOCKS = 20


def must(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)


def hash_path(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize(df, label):
    must(set(KEYS).issubset(df.columns), f'{label}: missing structural keys')
    x = df[KEYS].copy()
    x['Ticker'] = x['Ticker'].astype(str)
    x['Date'] = pd.to_datetime(x['Date'], errors='raise').dt.normalize()
    for name in KEYS[2:]:
        x[name] = pd.to_numeric(x[name], errors='raise').astype('int64')
    must(not x.isnull().any().any(), f'{label}: null structural value')
    must(not x.duplicated(['Ticker','Date']).any(), f'{label}: duplicated stock/date')
    return x.sort_values(['Date','Ticker'], kind='stable').reset_index(drop=True)


def balanced(x, label):
    counts = x.groupby('Date')['Ticker'].nunique()
    must(len(x) == len(counts) * STOCKS and counts.eq(STOCKS).all(),
         f'{label}: unexpected date-stock panel composition')
    must(x['Ticker'].nunique() == STOCKS, f'{label}: stock universe drift')
    must(x.groupby('Date')['continuity_segment_id'].nunique().eq(1).all(),
         f'{label}: mixed segments on date')
    must(x.groupby('Date')['market_session_id'].nunique().eq(1).all(),
         f'{label}: mixed sessions on date')


def date_level(x, status):
    d = x[['Date', 'market_session_id', 'continuity_segment_id']].drop_duplicates()
    return d.assign(status=status).sort_values('Date').reset_index(drop=True)


def negative_tests(kept, orig):
    # A candidate outside the frozen original keys must be rejected.
    a = kept.iloc[:1].copy()
    a.loc[a.index[0], 'Date'] = pd.Timestamp('2040-01-01')
    must(not pd.MultiIndex.from_frame(a[KEYS]).isin(pd.MultiIndex.from_frame(orig[KEYS])).all(),
         'Negative control: forged membership was not detected')
    # Drift in segment on the same date must be visible to key-level comparison.
    b = kept.iloc[:1].copy()
    b.loc[b.index[0], 'continuity_segment_id'] += 1
    must(not pd.MultiIndex.from_frame(b[KEYS]).isin(pd.MultiIndex.from_frame(orig[KEYS])).all(),
         'Negative control: altered continuity segment was not detected')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline-preflight', action='store_true')
    args = parser.parse_args()
    must(args.offline_preflight, 'Only --offline-preflight is supported')
    print('='*100)
    print('STEP 36I-F13C — SIGNED NON-LABEL TEMPORAL-COVERAGE AND INFERENCE-RISK REVIEW')
    print('='*100)
    print('[1] FROZEN SOURCES AND ACCESS BOUNDARY')
    must(ROOT.name == 'q1_revision', 'Place script in the q1_revision directory')
    for path, digest in HASHES.items():
        must(path.is_file() and not path.is_symlink(), f'Missing/unsafe: {path}')
        must(hash_path(path) == digest, f'Frozen digest mismatch: {path}')
    must(ACCESS.is_file() and not ACCESS.is_symlink(), 'Missing access ledger')
    state_bytes = ACCESS.read_bytes()
    state = json.loads(state_bytes)
    must(state.get('raw_test_filesystem_open_count_completed') == 2 and
         state.get('final_test_scored') is False, 'Access state drift')
    print('PASS — frozen membership/candidate/Step35 hashes; historical access 2 reads/0 scoring')

    print('[2] EXACT DATE-CLUSTER AND SET DIFFERENCE AUDIT')
    schema = pq.read_schema(LEDGER)
    must(set(KEYS+[FLAG]).issubset(schema.names), 'Membership Parquet missing expected columns')
    src = pq.read_table(LEDGER, columns=KEYS+[FLAG], use_threads=False).to_pandas()
    must(not src[FLAG].isna().any(), 'Null membership flags')
    original = normalize(src.loc[src[FLAG].eq(True), KEYS], 'Frozen original cohort')
    kept = normalize(pd.read_csv(CANDIDATE), 'Signed amended candidate')
    balanced(original, 'original')
    balanced(kept, 'candidate')
    must(len(original) == 6480 and original.Date.nunique() == 324, 'Original size drift')
    must(len(kept) == 5300 and kept.Date.nunique() == 265, 'Candidate size drift')
    joined = original.merge(kept.assign(_retained=True), how='left', on=KEYS, validate='one_to_one', indicator=True)
    must(len(joined) == 6480 and joined['_merge'].ne('right_only').all(), 'Original merge abnormal')
    removed = normalize(joined.loc[joined['_merge'].eq('left_only'), KEYS], 'Excluded endpoints')
    must(len(removed) == 1180 and removed.Date.nunique() == 59, 'Excluded count drift')
    balanced(removed, 'excluded')
    # Confirm that all candidate keys appear exactly once in original, not just a subset with extras.
    inverse = kept.merge(original, how='left', on=KEYS, validate='one_to_one', indicator=True)
    must(inverse['_merge'].eq('both').all(), 'Candidate contains unknown original endpoint')
    must(removed.Date.min() == pd.Timestamp('2025-05-12') and
         removed.Date.max() == pd.Timestamp('2025-07-31'), 'Excluded dates drift')
    negative_tests(kept, original)
    print('PASS — 6,480 original / 5,300 retained / 1,180 excluded; exactly 20 stocks per date')
    print('PASS — candidate subset, key/segment negative controls, and excluded interval verified')

    print('[3] NON-LABEL CALENDAR COVERAGE OF THE AMENDED COHORT')
    d_orig = date_level(original, 'original')
    d_keep = date_level(kept, 'retained')
    d_drop = date_level(removed, 'excluded')
    must(len(d_orig) == 324 and len(d_keep) == 265 and len(d_drop) == 59, 'Date-level drift')
    must(set(d_keep.Date).isdisjoint(set(d_drop.Date)), 'Date not fully retained or excluded')
    must(set(d_orig.Date) == set(d_keep.Date).union(set(d_drop.Date)), 'Date membership mismatch')
    segments = d_keep.groupby('continuity_segment_id').size().to_dict()
    must(segments == {0: 48, 1: 217}, f'Segment-size drift: {segments}')
    df = pd.DataFrame({'Date': d_orig.Date})
    df['month'] = df.Date.dt.to_period('M').astype(str)
    df['year'] = df.Date.dt.year
    df['quarter'] = df.Date.dt.to_period('Q').astype(str)
    df['retained'] = df.Date.isin(d_keep.Date)
    df['excluded'] = ~df.retained
    result = df.groupby('month', sort=True).agg(original=('Date', 'size'),
         retained=('retained','sum'), excluded=('excluded','sum')).reset_index()
    result['coverage_pct'] = (100.0 * result.retained/result.original).round(1)
    print('MONTH     ORIGINAL  RETAINED  EXCLUDED  COVERAGE_PCT')
    for r in result.itertuples(index=False):
        print(f'{r.month:<10}{r.original:>8}{r.retained:>10}{r.excluded:>10}{r.coverage_pct:>14.1f}')
    q = df.groupby('quarter', sort=True).agg(original=('Date','size'),
         retained=('retained','sum'), excluded=('excluded','sum')).reset_index()
    print('QUARTERLY COVERAGE (DATES):')
    for r in q.itertuples(index=False):
        print(f'  {r.quarter}: original={r.original}, retained={r.retained}, excluded={r.excluded}')
    print(f'PASS — 265/324 dates retained = {265/324:.2%}; excluded fraction={59/324:.2%}')
    print('[4] PRESPECIFIED BOOTSTRAP FEASIBILITY FLAGS — STRUCTURAL ONLY')
    for segment, n_dates in sorted(segments.items()):
        print(f'  Segment {segment}: {n_dates} dates; block lengths as % of segment: '
              + ', '.join(f'L{L}={L/n_dates:.1%}' for L in (10,20,40)))
    print('REVIEW REQUIRED — L=40 covers 83.3% of the 48-day short segment;')
    print('                  sensitivity CI/p-value stability requires scrutiny, not automatic rejection.')
    print('REVIEW REQUIRED — BH adjustment across 24 dependent tests requires explicit assumptions;')
    print('                  dependency-robust sensitivity may be justified but must be labeled post-freeze.')
    print('[5] CLAIM AND PERMISSION BOUNDARIES')
    must(ACCESS.read_bytes() == state_bytes, 'Historical access marker changed')
    print('PASS — original manifest, candidate and access marker unchanged')
    print('NOT ESTABLISHED — covariate/regime representativeness or comparative market difficulty;')
    print('                  this date-only audit intentionally does NOT access any test predictors/labels.')
    print('NOT APPROVED — post-freeze cohort/statistical amendment or third raw-test read')
    print('='*100)
    print('STEP 36I-F13C PASS — NON-LABEL TEMPORAL COVERAGE REVIEW ONLY')
    print('Sealed test opened: FALSE | Test feature values/labels used: FALSE | GPU: FALSE')
    print('Amendment approved: FALSE | Third read authorized: FALSE | Production: DENIED')
    print('='*100)

if __name__ == '__main__':
    main()
