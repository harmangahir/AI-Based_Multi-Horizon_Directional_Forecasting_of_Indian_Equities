#!/usr/bin/env python3
"""Step 36I-F21: independent F20 live-release gap audit. No test data access.

Inspects only source text, the amendment decision, Step35 manifest hash, and
historical raw-test access ledger. Never imports F20, calls GPU functions,
materializes arrays, opens the sealed test file, or modifies outputs.

--portable-self-test : source-only static review (can be run outside WSL)
--offline-preflight  : source review plus live WSL immutable boundaries

A PASS here means the SOURCE REVIEW found expected contracts and explicit OPEN
release gates; it does NOT approve or enable production execution.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

EXPECTED_F20 = '937de4fb1dc7a58f6ecb8449ad55a515a5dcb28120adbeaa82a4e31c0af5d018'
EXPECTED_F13E = 'f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493'
EXPECTED_STEP35 = '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7'

ROOT = Path(__file__).resolve().parent
F20 = ROOT / 'step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py'
APPROVAL = ROOT / 'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json'
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
STEP35 = EXP / 'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
ACCESS = EXP / 'manifests/36_final_test_access_consumption_state.json'


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(path):
    require(path.is_file() and not path.is_symlink(), f'Missing or unsafe file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def call_names(node):
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            x = n.func
            if isinstance(x, ast.Name):
                names.add(x.id)
            elif isinstance(x, ast.Attribute):
                if isinstance(x.value, ast.Name):
                    names.add(f'{x.value.id}.{x.attr}')
                names.add(x.attr)
    return names


def has_string(node, token):
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str)
               and token in n.value for n in ast.walk(node))


def review_source():
    require(sha(F20) == EXPECTED_F20, 'F20 source SHA-256 mismatch')
    tree = ast.parse(F20.read_text(encoding='utf-8'), filename=F20.name)
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required = {
        'check_historical_access', 'authenticate_upstream', 'seal_preloaded_phase_a',
        'inspect_phase_a', 'frozen_plan', 'compute_one_seed',
        'finalize_from_saved_seeds', 'offline_preflight', 'main'
    }
    require(required <= funcs.keys(), f'Missing F20 APIs: {sorted(required - funcs.keys())}')
    print('PASS — exact F20 source digest and nine named computational/control interfaces')

    a = funcs['seal_preloaded_phase_a']
    ac = call_names(a)
    require('BytesIO' in ac and 'read_table' in ac,
            'Phase A source must decode verified bytes in memory')
    require('sha256' in ac and 'phase_a_from_preloaded' in ac and
            'atomic_package' in ac and 'phase_a_manifest_digest' in ac,
            'Phase A hashing/materialization/commit sequence not evident')
    require('TEST_SHA' in ast.unparse(a), 'Frozen sealed-input digest not referenced')
    require('authorized_journal_digest' in ast.unparse(a), 'Journal provenance argument absent')
    print('PASS — Phase A requires input digest and accepts external journal provenance')

    inspect = funcs['inspect_phase_a']
    ic = call_names(inspect)
    require('verify_package' in ic and 'sha_file' in ic and 'load' in ic,
            'Phase A package verification/read-back checks absent')
    b = funcs['compute_one_seed']
    bc = call_names(b)
    require('inspect_phase_a' in bc and 'frozen_plan' in bc and
            'infer_and_commit_one' in bc,
            'Phase B no-raw-input seed recovery binding absent')
    finish = funcs['finalize_from_saved_seeds']
    fc = call_names(finish)
    need = {'strict_seed_inventory', 'authenticate_seed_against_plan',
            'in_memory_five_seed_scoring', 'evaluate_in_memory',
            'write_unique', 'rename_dir_no_replace'}
    require(need <= fc, f'Missing finalization hooks: {sorted(need-fc)}')
    for function in (b, finish):
        contents = ast.unparse(function)
        require('raw_verified_bytes' not in contents and
                'read_table' not in call_names(function),
                'Phase B unexpectedly uses preloaded raw bytes')
    print('PASS — sealed Phase A reader, frozen seed commits and post-seed finalization hooks')

    main_calls = call_names(funcs['main'])
    require(main_calls.isdisjoint({'seal_preloaded_phase_a',
                 'compute_one_seed', 'finalize_from_saved_seeds', 'inspect_phase_a'}),
            'Production functions callable from existing CLI')
    require('offline_preflight' in main_calls and
            has_string(funcs['main'], '--offline-preflight'),
            'Offline CLI contract absent')
    require(has_string(tree, '--read-test') and has_string(tree, '--production'),
            'Explicit production CLI denial absent')
    print('PASS — F20 CLI remains offline-only; protected real execution paths not connected')

    # These are real independent findings, not a pass that claims a release exists.
    require('authenticate_upstream' not in ac,
            'F20 call graph changed: re-evaluate upstream authorization placement')
    require('OneAttemptJournal' not in ac and 'record_verified' not in ac,
            'F20 access journal newly connected: requires new review')
    require('authorized_journal_digest' in ast.unparse(a) and
            'len(authorized_journal_digest)' in ast.unparse(a),
            'Journal argument contract changed')
    print('OPEN — Phase A takes externally supplied journal digest; it does not verify live durable journal/permission')
    print('OPEN — upstream verification occurs outside Phase A; release must guard input-binding and TOCTOU')
    print('OPEN — no approved sealed-test single-open loader or restartable production controller')
    print('OPEN — independently approved real output mapping and completed methodological sign-off absent')


def review_live_immutability():
    require(sha(APPROVAL) == EXPECTED_F13E, 'Study-owner approval SHA changed')
    require(sha(STEP35) == EXPECTED_STEP35, 'Step35 SHA changed')
    approval = json.loads(APPROVAL.read_bytes())
    require(approval.get('record_status') ==
            'APPROVED_BY_STUDY_OWNER_FOR_IMPLEMENTATION_AND_INDEPENDENT_REVIEW',
            'Study-owner amendment approval state incorrect')
    gates = approval.get('remaining_review_gates', {})
    require(gates.get('additional_raw_test_read_authorized') is False,
            'Unexpected test access authorization in scientific approval')
    require(gates.get('independent_methodological_signoff') is False,
            'Scientific approval unexpectedly elevated to methodological sign-off')
    original = ACCESS.read_bytes()
    state = json.loads(original)
    require(state.get('raw_test_filesystem_open_count_completed') == 2 and
            state.get('final_test_scored') is False,
            'Raw-test historical state differs from two reads / zero scores')
    require(ACCESS.read_bytes() == original, 'Access marker modified during review')
    print('PASS — Step35 and F13E hash-verified; historical ledger 2 reads / 0 scored')
    print('PASS — post-freeze design approved for implementation/review, NOT third read')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--portable-self-test', action='store_true')
    g.add_argument('--offline-preflight', action='store_true')
    args = p.parse_args()
    print('=' * 104)
    print('STEP 36I-F21 — INDEPENDENT F20 SOURCE / LIVE-RELEASE GAP REVIEW — NO TEST ACCESS')
    print('=' * 104)
    print('[1] SOURCE IDENTITY, REAL API FLOW AND NO-RAW-INPUT PHASE B')
    review_source()
    if args.offline_preflight:
        print('[2] LIVE IMMUTABLE SCIENTIFIC / HISTORICAL ACCESS STATE')
        review_live_immutability()
    else:
        print('[2] PORTABLE SELF-TEST: WSL-only immutability checks deferred')
    print('[3] RELEASE DECISION')
    print('F21 SOURCE REVIEW: PASS WITH EXPLICIT OPEN RELEASE BLOCKERS')
    print('No live source loader/journal, production CLI, or raw-test authorization granted')
    print('Real test opened: FALSE | GPU: FALSE | Test scored: FALSE | Files changed: FALSE')
    print('=' * 104)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, AssertionError, ValueError, KeyError, OSError, SyntaxError) as e:
        print('F21 REVIEW FAILED:', type(e).__name__, str(e), file=sys.stderr)
        sys.exit(1)
