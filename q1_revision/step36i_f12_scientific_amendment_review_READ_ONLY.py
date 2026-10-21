#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 36I-F12 — Independent methodological amendment review (read-only).

This is NOT scientific approval or an executable production launcher.
Permitted: --offline-preflight (WSL) or --self-test (portable review bundle).
Reads ONLY the frozen, non-label Step10 membership columns, frozen metadata,
review scripts, candidate key CSV, and historical access marker/logs.
Never reads the sealed test parquet, model outputs, test labels, or metrics.
Never writes to the experimental directory or consumes another access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
from collections import Counter

DENIED = {
    '--run', '--execute', '--production', '--read-test', '--test-parquet',
    '--score-test', '--authorize', '--approve', '--commit-real', '--resume',
    '--worker', '--enable', '--materialize', '--final-test', '--write',
}
if any(flag in DENIED for flag in sys.argv[1:]):
    raise SystemExit('DENIED: F12 is a READ-ONLY methodological review, not approval or production')

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
FROZEN_EXP = Path('/home/harman/projects/AI-Based_Multi-Horizon_Directional_Forecasting_of_Indian_Equities/stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026')
E2 = ROOT / 'step36i_e2_source_review_bundle.zip'
LIVE_STEP35 = EXP / 'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
LIVE_LEDGER = EXP / 'feature_data/step10_chronological_splits/10_split_membership.parquet'
LIVE_CANDIDATE = EXP / 'audits/step36i_protocol_review_draft/36i_candidate_sequence_eligible_keys_DRAFT.csv'
ACCESS = EXP / 'manifests/36_final_test_access_consumption_state.json'

HASH = {
    'E2': '9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851',
    'Step35': '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7',
    'membership': '82772196c92e88f1926a75a70c42f5f19119d7245d93b8364455fd28642ee121',
    'candidate': '4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65',
    'F11': '3405cb13a89d7c2403a66b5f6119b18a13c42f886d7f8d05aa80e94b09709ea6',
}
KEYS = ['Ticker', 'Date', 'market_session_id', 'continuity_segment_id']
N_ORIGINAL, D_ORIGINAL, N_CANDIDATE, D_CANDIDATE, N_STOCKS = 6480, 324, 5300, 265, 20


def check(cond, message):
    if not cond:
        raise RuntimeError(message)


def sha256(p: Path):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(2 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(p: Path, expected: str, label: str):
    check(p.is_file() and not p.is_symlink(), f'{label} missing or symbolic link: {p}')
    check(sha256(p) == expected, f'{label} SHA-256 drift: {p}')


def expected_records(z):
    z_names = z.namelist()
    check(len(z_names) == 18 and len(set(z_names)) == 18 and z.testzip() is None,
          'E2 bundle integrity or member inventory differs')
    for n in z_names:
        check('/' not in n and '\\' not in n and n not in ('.', '..'),
              'Unsafe E2 member name')
    step35 = json.loads(z.read('step35_contract.json'))
    deviation = json.loads(z.read('protocol_deviation_DRAFT.json'))
    amend = json.loads(z.read('amendment_review_DRAFT.json'))
    access = json.loads(z.read('access_history.json'))
    candidate_bytes = z.read('candidate_keys_DRAFT.csv')
    check(hashlib.sha256(candidate_bytes).hexdigest() == HASH['candidate'],
          'Frozen candidate CSV in E2 differs from expected checksum')
    check(deviation.get('status') == 'DRAFT_NOT_APPROVED'
          and amend.get('approval_granted') is False
          and amend.get('third_raw_test_read_authorized') is False,
          'E2 material records an unexpected approval')
    check(access.get('raw_test_filesystem_open_count_completed') == 2
          and access.get('final_test_scored') is False,
          'Historical E2 test-access state changed')
    return step35, deviation, amend, access, candidate_bytes


def check_statistical_contract(step35):
    stat = step35['dependence_aware_inference_contract']
    check(stat['replications_per_block_length'] == 2000
          and stat['primary_block_length_sessions'] == 20
          and stat['sensitivity_block_lengths_sessions'] == [10,20,40]
          and stat['multiple_testing_method'] == 'Benjamini-Hochberg FDR',
          'Original statistical freeze diverges')
    check(stat['primary_alternative'] == 'balanced_accuracy > 0.50'
          and stat['FDR_alpha'] == 0.05
          and len(step35['probability_threshold_contract']['thresholds']) == 24,
          'Original primary hypothesis/thresholds diverge')
    check(len(step35['predictor_preprocessing_contract']['ordered_features']) == 63,
          'Original feature interface changed')
    # Inspect documented amended design; do not silently label it prespecified.
    return stat


def normalize_keys(df):
    import pandas as pd
    check(set(KEYS).issubset(df.columns), 'Missing structural columns')
    out = df[KEYS].copy()
    out['Date'] = pd.to_datetime(out['Date'], errors='raise').dt.normalize()
    out['Ticker'] = out['Ticker'].astype(str)
    for c in KEYS[2:]:
        out[c] = pd.to_numeric(out[c], errors='raise').astype('int64')
    check(not out.isna().any().any() and not out.duplicated().any(),
          'Null or duplicate structural key')
    return out


def date_cluster_invariants(frame, rows, dates, label):
    check(len(frame) == rows and frame['Date'].nunique() == dates
          and frame['Ticker'].nunique() == N_STOCKS,
          f'{label} dimension mismatch: rows={len(frame)}, dates={frame.Date.nunique()}')
    check(frame.groupby('Date')['Ticker'].nunique().eq(N_STOCKS).all(),
          f'{label} unbalanced stocks per date')
    for d, grp in frame.groupby('Date'):
        check(grp['market_session_id'].nunique() == 1
              and grp['continuity_segment_id'].nunique() == 1,
              f'{label} inconsistent session or continuity segment on {d.date()}')


def structurally_review(original, candidate):
    import pandas as pd
    original = normalize_keys(original)
    candidate = normalize_keys(candidate)
    date_cluster_invariants(original, N_ORIGINAL, D_ORIGINAL, 'Original')
    date_cluster_invariants(candidate, N_CANDIDATE, D_CANDIDATE, 'Candidate')
    join = original.merge(candidate.assign(_kept=1), on=KEYS, how='left', validate='one_to_one')
    check(int(join['_kept'].fillna(0).sum()) == N_CANDIDATE,
          'Candidate not exactly a subset of frozen original keys')
    keep = join.loc[join['_kept'].eq(1), KEYS].copy()
    removed = join.loc[join['_kept'].isna(), KEYS].copy()
    check(len(removed) == 1180 and removed.Date.nunique() == 59,
          'Expected 1180/59 structural exclusions not reproduced')
    date_cluster_invariants(removed, 1180, 59, 'Removed')
    check(set(keep.Date.unique()).isdisjoint(set(removed.Date.unique())),
          'Partial-date exclusions detected')
    check(removed['continuity_segment_id'].unique().tolist() == [1],
          'Exclusions unexpectedly extend beyond the second segment')
    # These are documented technical-recovery boundaries, never tuned to labels.
    check((removed.Date.min().strftime('%Y-%m-%d'), removed.Date.max().strftime('%Y-%m-%d'))
          == ('2025-05-12','2025-07-31'), 'Excluded date boundaries drift')
    by_seg = keep.groupby('continuity_segment_id')['Date'].nunique().to_dict()
    check(by_seg == {0:48,1:217}, f'48/217 segment allocation changed: {by_seg}')
    for seg, group in keep.groupby('continuity_segment_id'):
        sess = sorted(group.drop_duplicates('Date')['market_session_id'].astype(int))
        check(all(b-a==1 for a,b in zip(sess,sess[1:])),
              f'Non-consecutive kept reference sessions in segment {seg}')
    # Negative-control: exclude one key in an otherwise full 20-stock date.
    trial = candidate.iloc[:-1].copy()
    try:
        date_cluster_invariants(trial, N_CANDIDATE, D_CANDIDATE, 'Negative control')
    except RuntimeError:
        pass
    else:
        raise RuntimeError('FAIL OPEN: dropped ticker-date accepted by cohort validator')
    return by_seg, removed


def review(portable: bool):
    print('=' * 106)
    print('STEP 36I-F12 — INDEPENDENT SCIENTIFIC PROTOCOL AMENDMENT REVIEW (NON-LABEL ONLY)')
    print('=' * 106)
    print('[1] ORIGINAL FROZEN CONTRACT AND E2 POST-FREEZE DEVIATION RECORD')
    verify(E2, HASH['E2'], 'Frozen E2 ZIP')
    with zipfile.ZipFile(E2) as z:
        step35, deviation, amend, snapshot_access, cb = expected_records(z)
    check_statistical_contract(step35)
    check(deviation.get('original_evaluation_rows') == N_ORIGINAL
          and deviation.get('proposed_evaluation_rows') == N_CANDIDATE
          and deviation.get('excluded_rows') == 1180
          and deviation.get('original_evaluation_dates') == D_ORIGINAL
          and deviation.get('proposed_evaluation_dates') == D_CANDIDATE,
          'Preexisting deviation ledger counts drifted')
    check(deviation['original_step35_sha256'] == HASH['Step35']
          and amend['scientific_contract_original_sha256'] == HASH['Step35'],
          'Post-freeze draft does not reference the original Step35 digest')
    print('PASS — Step35 unchanged; hash-verified E2 deviation remains explicitly NOT APPROVED')
    if portable:
        import pandas as pd
        cand = normalize_keys(pd.read_csv(io.BytesIO(cb)))
        date_cluster_invariants(cand, N_CANDIDATE, D_CANDIDATE, 'Portable candidate')
        check(cand.groupby('continuity_segment_id')['Date'].nunique().to_dict() == {0:48,1:217},
              'Portable candidate segment allocation changed')
        print('PASS — portable candidate 5300/265/20 with 48/217 segments (no WSL files opened)')
        print('STATUS: PASS — PORTABLE METADATA CHECK ONLY; WSL MEMBERSHIP REVIEW NOT PERFORMED')
        return
    print('[2] READ-ONLY WSL ORIGINAL-COHORT VERSUS CANDIDATE KEY AUDIT')
    check(EXP == FROZEN_EXP, 'Unexpected experiment root')
    verify(ROOT / 'step36i_f11_upstream_provenance_audit_READ_ONLY.py', HASH['F11'], 'F11 source')
    verify(LIVE_STEP35, HASH['Step35'], 'Live Step35')
    verify(LIVE_LEDGER, HASH['membership'], 'Frozen NON-LABEL Step10 membership')
    verify(LIVE_CANDIDATE, HASH['candidate'], 'Candidate key CSV')
    check(json.loads(LIVE_STEP35.read_text()) == step35, 'Live scientific contract differs from E2 copy')
    import pandas as pd
    # EXPLICIT 5 columns ONLY; no prices, returns, targets, outcomes, or sealed test parquet.
    raw = pd.read_parquet(LIVE_LEDGER,
                          columns=KEYS+['outer_final_test_common_multitask'], engine='pyarrow')
    flag = raw['outer_final_test_common_multitask'].astype(bool)
    original = raw.loc[flag, KEYS].copy()
    candidate = pd.read_csv(LIVE_CANDIDATE)
    by_seg, removed = structurally_review(original, candidate)
    print(f"PASS — exactly 6480/324 original vs 5300/265 proposed; excluded 1180/59 = {1180/6480*100:.4f}%")
    print('PASS — removed dates: 2025-05-12 through 2025-07-31, all 20 stocks, segment 1')
    print(f"PASS — retained segments {by_seg}; candidate is an exact frozen-membership subset")
    print('[3] ORIGINAL STATISTICS, ACCESS HISTORY AND PROPOSED REPORTING STATUS')
    hist = json.loads(ACCESS.read_text(encoding='utf-8'))
    check(hist.get('raw_test_filesystem_open_count_completed') == 2
          and hist.get('final_test_scored') is False
          and hist.get('state') == 'RAW_TEST_REOPENED_FOR_TECHNICAL_RECOVERY',
          'Access marker changed or an unauthorized scoring occurred')
    check(hist.get('expected_raw_test_sha256') == snapshot_access.get('expected_raw_test_sha256')
          and hist.get('observed_raw_test_sha256') == snapshot_access.get('expected_raw_test_sha256'),
          'Sealed test provenance differs from frozen historical record')
    print('PASS — original bootstrap B=2000, L=20 (+10/40 sensitivities), BA>.50, BH-FDR family 24')
    print('PASS — proposed 48/217 segment-stratified design is explicitly POST-FREEZE, not original Step35')
    print('PASS — historical access: 2 reads; no held-out performance metrics; no access modification')
    print('[4] METHODOLOGICAL DECISION GATES — NO AUTOMATIC APPROVAL')
    print('OPEN — independently confirm 1180 losses follow ONLY from 60-session causal-history eligibility')
    print('OPEN — document excluded-window covariate/regime coverage and resulting selection-bias limitations')
    print('OPEN — review validity of 48-session stratum with L=40 sensitivity and 24 dependent tests')
    print('OPEN — independent methodological sign-off, amended claim status and third-read decision')
    print('=' * 106)
    print('STEP 36I-F12 STRUCTURAL REVIEW: PASS — NOT SCIENTIFIC APPROVAL OR REAL-TEST AUTHORIZATION')
    print('Sealed test read: FALSE | Real labels/metrics read: FALSE | Original files modified: FALSE')
    print('Protocol approved: FALSE | Third read authorized: FALSE | Production enabled: FALSE')
    print('=' * 106)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--offline-preflight', action='store_true')
    group.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    review(portable=a.self_test)


if __name__ == '__main__':
    main()
