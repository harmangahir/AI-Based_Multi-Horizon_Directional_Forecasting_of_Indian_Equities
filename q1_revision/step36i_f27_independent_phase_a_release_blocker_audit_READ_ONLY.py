#!/usr/bin/env python3
"""STEP 36I-F27 — independent read-only F26 release-blocker audit.

Does NOT open test Parquet, import production components, write to experiment,
issue access permission, or execute model inference. The purpose is to distinguish
source-interface completeness from production release readiness.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / 'stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026'
SOURCES = {
    'F26': ('step36i_f26_locked_real_phase_a_wiring_REVIEW_ONLY.py',
            '4814ea8e23f93333389946c2ddd0f412af813e4fa82b84e94bb79e539e3c01a9'),
    'F25': ('step36i_f25_journal_locked_phase_a_integration_REVIEW_ONLY.py',
            'e633d00ff1a9532d7c8eda17ccde86071bcee375faa522dcbd3f5a7ff19bf27e'),
    'F24': ('step36i_f24_journal_codec_concurrency_REVIEW_ONLY.py',
            'c91d8c8cf224d3f465960cb552dcfb908504a850b1b968c7d68de3b0cdb0c487'),
    'F23': ('step36i_f23_live_release_controller_CANDIDATE_REVIEW_ONLY.py',
            '11497c21fda381112ca65012540d3bc2bc7522030ba0bc577d9808c0c812d1b7'),
    'F22': ('step36i_f22_access_bound_phase_a_controller_REVIEW_ONLY.py',
            '7ae182b83a20d2f621ded09ceb7e5e696f1b9c41b0f814a0ffd05c4d79997b26'),
    'F20': ('step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py',
            '937de4fb1dc7a58f6ecb8449ad55a515a5dcb28120adbeaa82a4e31c0af5d018'),
    'F18': ('step36i_f18_durable_two_phase_artifact_store_REVIEW_ONLY.py',
            'c4706fb360233473078f5f4aac3865c9525535ef6cd4e225b258d1a728037f69'),
}
STEP35 = EXP/'manifests/35_final_test_evaluation_contract_freeze_manifest.json'
APPROVAL = ROOT/'STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json'
MARKER = EXP/'manifests/36_final_test_access_consumption_state.json'
SHA35 = '6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7'
SHA_APPROVAL = 'f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493'


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha256_file(path: Path):
    require(path.is_file() and not path.is_symlink(), f'Missing or unsafe: {path}')
    h = hashlib.sha256()
    with path.open('rb') as fd:
        for block in iter(lambda: fd.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def tree(name):
    return ast.parse((ROOT/SOURCES[name][0]).read_text('utf-8'))


def get_function(t, name):
    arr = [x for x in t.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name == name]
    require(len(arr)==1, f'Missing/ambiguous function: {name}')
    return arr[0]


def calls_in_order(node):
    return [(x.lineno, ast.unparse(x.func)) for x in ast.walk(node) if isinstance(x, ast.Call)]


def has_call(node, qual):
    return any(x == qual for _,x in calls_in_order(node))


def call_line(node, qual):
    lines = [n for n,f in calls_in_order(node) if f==qual]
    require(len(lines) >= 1, f'Expected call missing: {qual}')
    return min(lines)


def audit(portable=False):
    print('='*100)
    print('STEP 36I-F27 — INDEPENDENT F26 PHASE A RELEASE-BLOCKER AUDIT (READ-ONLY)')
    print('='*100)
    print('[1] REVIEWED SOURCE/SCIENTIFIC CONTRACT IDENTITY')
    for tag,(filename,expected) in SOURCES.items():
        require(sha256_file(ROOT/filename)==expected, f'{tag} digest changed')
    require(sha256_file(APPROVAL)==SHA_APPROVAL, 'F13E owner approval changed')
    marker_before = None
    if not portable:
        require(sha256_file(STEP35)==SHA35, 'Step35 changed')
        marker_before = MARKER.read_bytes()
        marker = json.loads(marker_before)
        require(marker.get('raw_test_filesystem_open_count_completed')==2 and
                marker.get('final_test_scored') is False, 'Unexpected historical access state')
        print('PASS — F18/F20/F22/F23/F24/F25/F26, Step35, F13E verified; historical reads=2, scored=0')
    else:
        print('PASS — F18/F20/F22/F23/F24/F25/F26 and F13E verified; WSL state check deferred')

    f26=tree('F26'); f23=tree('F23'); f20=tree('F20'); f24=tree('F24')
    print('[2] LIVE GATE AND TRUE COMPOSITION ORDER')
    assignments = [node for node in f26.body if isinstance(node,ast.Assign)]
    gates = [node for node in assignments if any(isinstance(x,ast.Name) and x.id=='LIVE_RELEASE_ENABLED'
             for x in node.targets)]
    require(len(gates)==1 and isinstance(gates[0].value,ast.Constant) and gates[0].value.value is False,
            'F26 release gate not hard-disabled')
    fn=get_function(f26,'proposed_genuine_phase_a')
    require(isinstance(fn.body[1],ast.If) or isinstance(fn.body[0],ast.If),
            'Release function must check its gate at entry')
    required = ['f24.cooperative_directory_lock','review_records.inspect',
       'f22.ReservedReadJournal','f23.journalized_single_read',
       'f24.strict_jsonl_journal','f20.seal_preloaded_phase_a',
       'f20.inspect_phase_a','f18.rename_noreplace']
    sequence = [call_line(fn,s) for s in required]
    require(sequence == sorted(sequence),f'Unexpected call order: {list(zip(required,sequence))}')
    locks = [n for n in ast.walk(fn) if isinstance(n,ast.With)]
    require(any(any(ast.unparse(item.context_expr).startswith('f24.cooperative_directory_lock(')
                    for item in n.items) and
                all(has_call(n,k) for k in ('f23.journalized_single_read','f20.seal_preloaded_phase_a',
                                            'f18.rename_noreplace')) for n in locks),
            'Read/seal/receipt operations not under same cooperative lock')
    print('PASS — real Phase A disabled; release source orders review, reservation, single-read, journal, seal, receipt')
    print('PASS — cooperative lock lexically encloses read, Phase A seal and receipt transaction')

    print('[3] INDEPENDENT RELEASE BLOCKERS (NOT CLEARED BY THE PREFLIGHT)')
    # ReleaseRecords.inspect is defined on a class in F23; no externally verified signature authority is bound.
    classes = [x for x in f23.body if isinstance(x,ast.ClassDef) and x.name=='ReleaseRecords']
    require(len(classes)==1,'ReleaseRecords definition not found')
    inspection = next((x for x in classes[0].body if isinstance(x,ast.FunctionDef) and x.name=='inspect'),None)
    require(inspection is not None, 'ReleaseRecords.inspect missing')
    require(not any('signature' in n or 'certificate' in n for _,n in calls_in_order(inspection)),
            'External signature verifier unexpectedly added; independent audit needs update')
    print('BLOCKER A — F23 approval/permit fields are self-asserted JSON; independently authenticated signoff/permit missing')

    snap = get_function(f23,'recheck_upstream')
    need_names = [n for _,n in calls_in_order(snap)]
    require('digest_file' in need_names,'Upstream path-based rehash absent')
    materializer = get_function(f20,'seal_preloaded_phase_a')
    require(has_call(materializer,'pd.read_parquet'), 'Expected development-history load missing')
    print('BLOCKER B — development-history file is reopened by pathname after hashing (input-binding/TOCTOU not resolved)')

    require(call_line(fn,'f23.journalized_single_read') < call_line(fn,'f20.seal_preloaded_phase_a'),
            'Single-read/materialization order unexpectedly changed')
    require(not any(x.startswith('durable_raw') or x.startswith('seal_raw_bytes') for _,x in calls_in_order(fn)),
            'Durable raw-byte spool appears added; reviewer must reassess crash policy')
    print('BLOCKER C — after third read, parsing/materialization failure can strand the one-use attempt before arrays seal')

    require(has_call(fn,'f24.cooperative_directory_lock') and
            has_call(fn,'f20.inspect_phase_a'),'Required Phase A integrity checks missing')
    require(has_call(fn,'f23.check_new_output_paths_empty'),
            'F26 does not revalidate all seven output destinations')
    print('PASS — F26 rechecks all seven reserved output paths before Phase A directory creation')
    f24lock=get_function(f24,'cooperative_directory_lock')
    require(any('flock' in call for _,call in calls_in_order(f24lock)),
            'F24 lock implementation changed unexpectedly')
    print('BLOCKER D — cooperative flock alone cannot exclude uncooperative writers; ownership/security review pending')

    require(not any(isinstance(x,ast.FunctionDef) and x.name in
                    ('production_cli','run_live_phase_b','authorize_third_read') for x in f26.body),
            'New production path detected; review needs revision')
    print('BLOCKER E — independently approved Phase B/live release and explicit third-read authority still absent')

    print('[4] IMMUTABILITY AND RELEASE DECISION')
    if not portable:
        require(MARKER.read_bytes()==marker_before, 'Historical access marker modified')
        print('PASS — historical marker unchanged; no raw-test file accessed; no outputs created')
    else:
        print('PASS — static source-only review; WSL access marker not inspected')
    print('F27 SOURCE AUDIT: PASS WITH FIVE RELEASE BLOCKERS — PRODUCTION SIGNOFF DENIED')
    print('Scientific amendment: STUDY-OWNER APPROVED | Third read: NOT AUTHORIZED | Real scoring: NOT DONE')
    print('='*100)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--offline-preflight', action='store_true')
    group.add_argument('--portable-self-test', action='store_true')
    args=parser.parse_args()
    try:
        audit(portable=args.portable_self_test)
    except (RuntimeError,ValueError,TypeError,OSError,KeyError,AssertionError) as e:
        print(f'F27 FAIL CLOSED: {type(e).__name__}: {e}',file=sys.stderr)
        raise SystemExit(1)
