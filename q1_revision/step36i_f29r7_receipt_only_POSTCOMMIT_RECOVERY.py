#!/usr/bin/env python3
"""Post-commit, receipt-only publication of immutable F29R6 held-out evaluation.

Does NOT reopen sealed raw-test file, rerun metrics/bootstrap, or perform GPU inference.
Requires authenticated signed F29R5 source/owner decisions, committed Phase A,
all forty frozen seed packages and the exact F29R6 committed result manifest.
Original F29R6 source and historical failure evidence remain unchanged.
"""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import importlib.util
import json
import os
import stat
import sys

ROOT = Path(__file__).resolve().parent
OWNER = Path.home() / '.config/indian_equities_f29_owner'
EXPECTED = {
    'step36i_f29r5_frozen_ml_extended_signature_gated_CANDIDATE.py': 'bc073d6bf565ddbc06fa87007e897eaea16f1cce75cc8940d2b7a0f9a33be220',
    'step36i_f29r6_finalization_only_RECOVERY.py': '7b69b58af0968c0bc5084c4fc2b3b61f495f949057d41996c786dcb69b5c0e06',
    'step36i_f29r6_finalizer_CORRECTED.py': 'a1df40c21a24c586034a88971ffd5aef655a32b056df5e8d550a8d78aaeee65b',
    'step36i_f29r6_regime_diagnostics_CORRECTED.py': '8ff304f9d8bbba3b10dfac81e40146404ec6805c298c672d2a4039c6027dc680',
    'step36i_f29r6_ml_scorer_CORRECTED.py': '9108556bfb345d6089adaa9da5cac7c8b300d55c69e6536847d1978a67ac7a11',
}
METHOD_SHA='d17e1b503ca43d659fb4e292dbc90a1c2f11780dd686ee7b41e258dc8410ec3a'
PERMIT_SHA='d91b818d7d4f3d4e840702d69da92421beb2d141f4b6b300524e84e46027d456'
RESULT_SHA='47e7820dafe9ec8f951f8daa45ce77aae37599d872bc631a4657f77c96903512'
FAILURE_SHA='a0b90fac1e16fc557b31a967cfd25238aa9c443c6fc528b352a9797306cb13c0'


def require(cond, why):
    if not cond:
        raise RuntimeError(why)


def sha(path):
    require(path.is_file() and not path.is_symlink(), f'Missing/unsafe: {path}')
    h=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def check(path, wanted):
    require(sha(path)==wanted, f'SHA mismatch: {path}')


def import_file(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None, 'Source import unavailable')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def record_count(path):
    with path.open(newline='') as f:
        return sum(1 for _ in csv.DictReader(f))


def run():
    require(ROOT.is_dir() and ROOT.name=='q1_revision', 'Wrong working source root')
    for name,expected in EXPECTED.items():
        check(ROOT/name,expected)
    method=OWNER/'signed_decisions/F29R5_method_self_review.json'
    permit=OWNER/'signed_decisions/F29R5_third_raw_test_read_permit.json'
    check(method,METHOD_SHA)
    check(permit,PERMIT_SHA)
    c=import_file(ROOT/'step36i_f29r5_frozen_ml_extended_signature_gated_CANDIDATE.py',
                  'f29r7_parent_immutable')
    c.authentic_core()
    c.inspect_owner_public_keyring_only()
    c.check_critical_environment()
    history=c.historic_state()
    args=SimpleNamespace(
        trusted_keys=str(OWNER/'owner_public_keyring.json'),
        keyring_sha256=c.OWNER_PINNED_KEYRING_SHA256,
        method_attestation=str(method),read_permit=str(permit))
    with c.dependency_library_import_context():
        f23=c.reviewed_module('F23')
        f20=c.reviewed_module('F20')
        f24=c.reviewed_module('F24')
        authority,auth_sha,_,permit_body=c.validate_signatures(args,f23)
        _,bound=f23.pre_read_upstream_snapshot(f20)
        upstream_sha=f23.digest_bytes(f23.canonical(bound))
        require(upstream_sha==permit_body['upstream_binding_sha256'],
                'Frozen development upstream binding drift')
        require(authority.inspect(intended_source_path=Path(permit_body['source_path']),
                                  upstream_digest=upstream_sha)==auth_sha,
                'Signed owner authorization mismatch')
        require(f23.mapping_digest()==c.MAPPING_SHA,'Frozen output mapping drift')
        with f24.cooperative_directory_lock(c.EXP/'manifests'):
            phase=c.assert_phase_a_for_phase_b(auth_sha)
            f18=f20.import_reviewed('F18')
            f19=f20.import_reviewed('F19')
            plan,_=f20.frozen_plan()
            phase_sha=f18.phase_a_manifest_digest(phase)
            seeds=f23.ISOLATED['predictions']
            require(seeds.is_dir() and not seeds.is_symlink(),'Seed store missing/unsafe')
            expected_seeds={f'seed_{model}_{seed}' for model,seed in plan}
            result=seeds/'completed_evaluation'
            require(result.is_dir() and not result.is_symlink(),
                    'Existing completed evaluation unavailable')
            actual_dirs={p.name for p in seeds.iterdir() if p.is_dir()}
            require(actual_dirs==expected_seeds|{'completed_evaluation'},
                    'Unexpected directory in seed namespace')
            require(not any(seeds.glob('.F20_STAGING_*')),
                    'Finalization staging exists; manual review')
            for model,seed in sorted(plan):
                f19.authenticate_seed_against_plan(
                    seeds/f'seed_{model}_{seed}',plan[(model,seed)],phase_sha,f18)
            print('PASS — signed owner authorization, Phase A and 40 original seeds',flush=True)
            manifest=result/'evaluation_manifest.json'
            check(manifest,RESULT_SHA)
            ledger=json.loads(manifest.read_bytes())
            require(ledger.get('schema')=='STEP36I_F29R6_FINAL_EVALUATION_REGIME_LABEL_RECOVERY_V1'
                    and ledger.get('regime_label_correction')=='ID_1_NORMAL_NOT_MEDIUM'
                    and ledger.get('phase_a_manifest_sha256')==phase_sha,
                    'F29R6 committed manifest scientific provenance mismatch')
            wanted={'predictions':42400,'metrics':24,'statistics':72,
                    'primary_bh_tests':24,'exploratory_pairs':84,
                    'ml_baseline_model_count':9,'ml_vs_persistence_pairs':9,
                    'neural_vs_ml_exploratory_pairs':72,
                    'supplementary_artifact_count':8,
                    'ml_supplementary_artifact_count':10}
            require(all(ledger.get(k)==v for k,v in wanted.items()),
                    'Committed scientific inventory unexpected')
            artifacts=ledger.get('artifacts')
            require(isinstance(artifacts,list) and len(artifacts)==23,
                    'Expected precisely 23 committed result artifacts')
            names=[e['name'] for e in artifacts]
            require(len(set(names))==23,'Duplicate result artifact names')
            real_names={p.name for p in result.iterdir() if p.is_file()}
            require(real_names==set(names)|{'evaluation_manifest.json'},
                    'Completed directory inventory differs from manifest')
            for item in artifacts:
                rel=Path(item['name'])
                require(not rel.is_absolute() and len(rel.parts)==1 and '..' not in rel.parts,
                        'Unsafe result artifact name')
                path=result/rel
                require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],
                        f'Result artifact changed: {rel}')
            metrics=result/'classification_metrics.csv'
            require(record_count(metrics)==24,'Original primary metrics row count differs')
            require(record_count(result/'bootstrap_fdr.csv')==72,'Bootstrap rows differ')
            require(record_count(result/'ml_devonly_classification_metrics.csv')==9,
                    'ML metrics rows differ')
            require(record_count(result/'ml_devonly_neural_vs_ml_paired_block_ci.csv')==72,
                    'Neural vs ML pair rows differ')
            failure=f23.ISOLATED['failure_evidence']
            check(failure,FAILURE_SHA)
            dst=f23.ISOLATED['metrics']
            receipt_dst=f23.ISOLATED['evaluation_manifest']
            require(not dst.exists() and not dst.is_symlink() and
                    not receipt_dst.exists() and not receipt_dst.is_symlink(),
                    'External publication already present; STOP; never overwrite')
            metrics_stage=dst.with_name('.F29R7_METRICS_STAGING')
            receipt_stage=receipt_dst.with_name('.F29_RECEIPT_STAGING')
            require(not metrics_stage.exists() and not metrics_stage.is_symlink() and
                    not receipt_stage.exists() and not receipt_stage.is_symlink(),
                    'Previous staging exists; manual review')
            require(c.historic_state()==history,'Legacy history drift before publication')
            print('PASS — 23/23 immutable committed results, primary/ML row counts and preserved failure evidence',flush=True)
            # No model computations; publish byte-identical metrics before final receipt.
            raw=metrics.read_bytes()
            f18.create_exclusive(metrics_stage,raw)
            f18.rename_noreplace(metrics_stage,dst)
            require(sha(dst)==sha(metrics),'Metrics external copy differs from committed file')
            receipt={
                'schema':'STEP36I_F29R6_FINALIZATION_ONLY_REGIME_CORRECTION_RECEIPT_V1',
                'publication_recovery':'F29R7_RECEIPT_ONLY_AFTER_F29R6_POSTCOMMIT_KEY_TYPO',
                'finalization_recovery_code_sha256':EXPECTED['step36i_f29r6_finalization_only_RECOVERY.py'],
                'original_f29r5_signed_controller_sha256':EXPECTED['step36i_f29r5_frozen_ml_extended_signature_gated_CANDIDATE.py'],
                'method_self_review_record_sha256':METHOD_SHA,
                'third_read_permit_record_sha256':PERMIT_SHA,
                'third_read_permit_scope':'already_consumed; NO ADDITIONAL RAW TEST ACCESS',
                'original_phase_b_failure_evidence_sha256':FAILURE_SHA,
                'corrected_supplementary_source_sha256':EXPECTED['step36i_f29r6_regime_diagnostics_CORRECTED.py'],
                'corrected_ml_source_sha256':EXPECTED['step36i_f29r6_ml_scorer_CORRECTED.py'],
                'corrected_finalizer_source_sha256':EXPECTED['step36i_f29r6_finalizer_CORRECTED.py'],
                'correction':'regime_id=1 is Normal, not Medium in frozen Step12 and Phase A',
                'phase_a_manifest_sha256':phase_sha,
                'final_evaluation_manifest_sha256':RESULT_SHA,
                'final_metrics_csv_sha256':sha(dst),
                'completed_evaluation_dir':str(result),
                'verified_immutable_result_artifacts':23,
                'original_confirmatory_BH_tests':24,
                'supplementary_ML_baselines':9,
                'previous_40_checkpoints_reused_without_reinference':True,
                'heldout_scoring_repeated_during_receipt_recovery':False,
                'sealed_test_reopened_during_recovery':False,
                'historical_raw_access_marker_preserved':True,
                'independent_reviewer':False,
            }
            c.stage_receipt(receipt_dst,receipt)
            require(c.historic_state()==history,'Historical marker drift after receipt')
            print('F29R7 RECEIPT-ONLY POSTCOMMIT RECOVERY PASS',flush=True)
            print('Committed evaluation:',result)
            print('Published metrics:',dst)
            print('Published receipt:',receipt_dst)
            print('Receipt SHA256:',sha(receipt_dst))
            print('Raw test reopened: FALSE | GPU inference: FALSE | Bootstrap rerun: FALSE')

if __name__=='__main__':
    try:
        run()
    except Exception as exc:
        print(f'F29R7 FAIL-CLOSED: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(1)
