#!/usr/bin/env python3
"""F29R6 — finalization-only recovery for frozen Step12 regime ID 1 = 'Normal'.

Derivative software only. No raw sealed-test opening, no checkpoint training or
neural inference. Requires authentic signed F29R5 parent release records, verified
committed Phase A, 40 immutable seeds, and a new explicit CLI acknowledgement.
Preserves original F29R5 and F29R3 source, signatures and earlier failure evidence.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OWNER = Path.home()/'.config/indian_equities_f29_owner'
PARENT_SOURCE = ROOT/'step36i_f29r5_frozen_ml_extended_signature_gated_CANDIDATE.py'
METHOD = OWNER/'signed_decisions/F29R5_method_self_review.json'
PERMIT = OWNER/'signed_decisions/F29R5_third_raw_test_read_permit.json'
KEYRING = OWNER/'owner_public_keyring.json'
SUP_FIXED = ROOT/'step36i_f29r6_regime_diagnostics_CORRECTED.py'
ML_FIXED = ROOT/'step36i_f29r6_ml_scorer_CORRECTED.py'
F20_FIXED = ROOT/'step36i_f29r6_finalizer_CORRECTED.py'
PARENT_SHA = 'bc073d6bf565ddbc06fa87007e897eaea16f1cce75cc8940d2b7a0f9a33be220'
METHOD_SHA = 'd17e1b503ca43d659fb4e292dbc90a1c2f11780dd686ee7b41e258dc8410ec3a'
PERMIT_SHA = 'd91b818d7d4f3d4e840702d69da92421beb2d141f4b6b300524e84e46027d456'
SUP_SHA = '8ff304f9d8bbba3b10dfac81e40146404ec6805c298c672d2a4039c6027dc680'
ML_SHA = '9108556bfb345d6089adaa9da5cac7c8b300d55c69e6536847d1978a67ac7a11'
F20_SHA = 'a1df40c21a24c586034a88971ffd5aef655a32b056df5e8d550a8d78aaeee65b'
ORIGINAL = {
    'step36i_f29r3_supplementary_baselines_diagnostics_REVIEW_ONLY.py': '136e6607292e0e2849f84cecc5b48cb7f7ce00ab5f3daff2af2fbe1d29256311',
    'step36i_f29r5_ml_baseline_scoring_REVIEW_ONLY.py': '45758d962d7d16e9e500e88be1560ca97a5298a3cc3f247d7d8a03a6a15ee49f',
    'step36i_f20r3_frozen_ml_finalizer_CANDIDATE.py': 'a9b9d44470a7cba6aeb54a4a734b6ebf7b52a3194c9b6f645bae38ad297dfc5d',
}


def need(flag, why):
    if not flag:
        raise RuntimeError(why)


def sha(path):
    need(path.is_file() and not path.is_symlink(), f'Missing or unsafe file: {path}')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def check(path, wanted, tag):
    need(sha(path)==wanted, f'Immutable {tag} hash mismatch')


def imported(path, tag):
    spec=importlib.util.spec_from_file_location(tag,path)
    need(spec is not None and spec.loader is not None, f'Unavailable module: {tag}')
    mod=importlib.util.module_from_spec(spec)
    sys.modules[tag]=mod
    spec.loader.exec_module(mod)
    return mod


def setup():
    check(PARENT_SOURCE,PARENT_SHA,'signed F29R5 source')
    check(METHOD,METHOD_SHA,'signed method record')
    check(PERMIT,PERMIT_SHA,'signed third-read permit')
    for name,digest in ORIGINAL.items():
        check(ROOT/name,digest,'unchanged original '+name)
    for path,digest in ((SUP_FIXED,SUP_SHA),(ML_FIXED,ML_SHA),(F20_FIXED,F20_SHA)):
        check(path,digest,'corrected derivative '+path.name)
    c=imported(PARENT_SOURCE,'f29r6_parent_verified')
    c.authentic_core()
    c.inspect_owner_public_keyring_only()
    c.check_critical_environment()
    old_history=c.historic_state()
    with c.dependency_library_import_context():
        f23=c.reviewed_module('F23')
        f20old=c.reviewed_module('F20')
        f24=c.reviewed_module('F24')
        args=SimpleNamespace(trusted_keys=str(KEYRING),
            keyring_sha256=c.OWNER_PINNED_KEYRING_SHA256,
            method_attestation=str(METHOD),read_permit=str(PERMIT))
        auth,authority_sha,method,permit=c.validate_signatures(args,f23)
        upstream,binding=f23.pre_read_upstream_snapshot(f20old)
        upstream_sha=f23.digest_bytes(f23.canonical(binding))
        need(permit['upstream_binding_sha256']==upstream_sha,
             'Changed development-only inputs')
        need(auth.inspect(intended_source_path=Path(permit['source_path']),
                          upstream_digest=upstream_sha)==authority_sha,
             'Signed original study owner authority mismatch')
        need(f23.mapping_digest()==c.MAPPING_SHA,'Output map drift')
        package=c.assert_phase_a_for_phase_b(authority_sha)
        seeds=f23.ISOLATED['predictions']
        need(seeds.is_dir() and not seeds.is_symlink(),'Missing committed seed store')
        f19=f20old.import_reviewed('F19');f18=f20old.import_reviewed('F18')
        plan,_=f20old.frozen_plan()
        phase_sha=f18.phase_a_manifest_digest(package)
        done,missing=f19.strict_seed_inventory(seeds,phase_sha,plan,f18)
        need(len(done)==40 and not missing,'Not all 40 authentic frozen seeds committed')
        import pandas as pd
        metadata=pd.read_parquet(package/'arrays/metadata.parquet',
                                 columns=['regime_id','regime_name'])
        need(len(metadata)==5300,'Amended cohort drift')
        names={0:'low',1:'normal',2:'high'}
        observed=metadata.regime_name.astype(str).str.lower().tolist()
        ids=metadata.regime_id.tolist()
        need(all(names.get(int(k))==n for k,n in zip(ids,observed)),
             'Unexpected regime semantics; recovery not applicable')
        need({int(k) for k in ids}=={0,1,2},'Incomplete regime ID set')
        print('PASS — 5,300 Phase A records: ID 0 Low, ID 1 Normal, ID 2 High')
        print('PASS — all 40 original frozen seed packages independently authenticated')
        need(c.historic_state()==old_history,'Historical raw-access marker drift')
        need(not (seeds/'completed_evaluation').exists(),
             'Published evaluation already present; refuse overwrite')
        need(not any(seeds.glob('.F20_STAGING_*')),
             'Abandoned finalization staging present; manual review required')
        need(not f23.ISOLATED['evaluation_manifest'].exists() and
             not f23.ISOLATED['metrics'].exists(),
             'Reserved output already occupied; refuse overwrite')
        previous_failure=f23.ISOLATED['failure_evidence']
        need(previous_failure.is_file(), 'Original Phase B failure evidence missing')
        failure=json.loads(previous_failure.read_bytes())
        need(failure.get('phase')=='PHASE_B' and
             failure.get('authorization_binding_sha256')==authority_sha,
             'Historical import failure evidence changed')
        failure_sha=sha(previous_failure)
        fixed=imported(F20_FIXED,'f29r6_corrected_finalizer')
        need(fixed.SUPPLEMENTARY_SHA==SUP_SHA and
             fixed.ML_SUPPLEMENTARY_SHA==ML_SHA,
             'Corrected supplementary hashes not bound to derivative finalizer')
        print('PASS — prior Phase B failure evidence preserved, SHA',failure_sha)
        print('PASS — amended finalization outputs vacant; no abandoned staging')
        return c,f23,f24,f18,fixed,seeds,package,authority_sha,old_history,failure_sha


def finalization():
    c,f23,f24,f18,fixed,seeds,package,authority_sha,old_history,failure_sha=setup()
    with c.dependency_library_import_context():
        with f24.cooperative_directory_lock(c.EXP/'manifests'):
            # No original test access: load only authenticated Phase A and 40 seed files.
            c.assert_phase_a_for_phase_b(authority_sha)
            target=fixed.finalize_from_saved_seeds(
                phase_package=package,seed_store=seeds,output_parent=seeds)
            manifest=target/'evaluation_manifest.json'
            metrics=target/'classification_metrics.csv'
            expected=tuple(
                ['supplementary_baselines.csv','supplementary_stock_level.csv',
                 'supplementary_regime_level.csv','supplementary_segment_level.csv',
                 'supplementary_calendar_quarter.csv','supplementary_calibration_10_fixed_bins.csv',
                 'supplementary_model_vs_persistence_paired_block_ci.csv',
                 'supplementary_methods_manifest.json'] +
                ['ml_devonly_predictions.parquet','ml_devonly_classification_metrics.csv',
                 'ml_devonly_vs_persistence_paired_block_ci.csv',
                 'ml_devonly_neural_vs_ml_paired_block_ci.csv',
                 'ml_devonly_stock_level.csv','ml_devonly_regime_level.csv',
                 'ml_devonly_segment_level.csv','ml_devonly_calendar_quarter.csv',
                 'ml_devonly_calibration_10_bins.csv','ml_devonly_methods_manifest.json']
            )
            need(manifest.is_file() and metrics.is_file() and
                 all((target/n).is_file() for n in expected),
                 'Corrected finalization missing mandatory artifacts')
            ledger=json.loads(manifest.read_bytes())
            need(ledger['schema']=='STEP36I_F29R6_FINAL_EVALUATION_REGIME_LABEL_RECOVERY_V1' and
                 ledger['primary_bh_tests']==24 and ledger['metrics']==24 and
                 ledger['ml_baseline_model_count']==9 and
                 ledger['neural_vs_ML_exploratory_pairs']==72,
                 'Recovered result manifest scientific inventory drift')
            csv=metrics.read_bytes()
            need(b'model_id' in csv.splitlines()[0], 'Invalid final metrics CSV')
            csv_dst=f23.ISOLATED['metrics']
            stage=csv_dst.with_name('.F29R6_METRICS_STAGING')
            need(not stage.exists() and not stage.is_symlink(),'Metrics staging already exists')
            f18.create_exclusive(stage,csv)
            f18.rename_noreplace(stage,csv_dst)
            need(sha(csv_dst)==sha(metrics),'Published metrics SHA mismatch')
            receipt={
                'schema':'STEP36I_F29R6_FINALIZATION_ONLY_REGIME_CORRECTION_RECEIPT_V1',
                'finalization_recovery_code_sha256':sha(Path(__file__)),
                'original_f29r5_signed_controller_sha256':PARENT_SHA,
                'method_self_review_record_sha256':METHOD_SHA,
                'third_read_permit_record_sha256':PERMIT_SHA,
                'third_read_permit_scope':'already_consumed; NO ADDITIONAL RAW TEST ACCESS',
                'original_phase_b_failure_evidence_sha256':failure_sha,
                'corrected_supplementary_source_sha256':SUP_SHA,
                'corrected_ml_source_sha256':ML_SHA,
                'corrected_finalizer_source_sha256':F20_SHA,
                'correction':'regime_id=1 is Normal, not Medium in frozen Step12 and Phase A',
                'phase_a_manifest_sha256':f18.phase_a_manifest_digest(package),
                'final_evaluation_manifest_sha256':sha(manifest),
                'final_metrics_csv_sha256':sha(csv_dst),
                'completed_evaluation_dir':str(target),
                'original_confirmatory_BH_tests':24,
                'supplementary_ML_baselines':9,
                'previous_40_checkpoints_reused_without_reinference':True,
                'sealed_test_reopened_during_recovery':False,
                'historical_raw_access_marker_preserved':True,
                'independent_reviewer':False,
            }
            c.stage_receipt(f23.ISOLATED['evaluation_manifest'],receipt)
            need(c.historic_state()==old_history,'Historical test ledger changed')
            print('F29R6 FINALIZATION-ONLY RECOVERY PASS:',target)
            print('Published metrics:',csv_dst)
            print('Receipt SHA:',sha(f23.ISOLATED['evaluation_manifest']))
            print('Original sealed test reread: FALSE')


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--offline-preflight',action='store_true')
    g.add_argument('--finalize',action='store_true')
    p.add_argument('--i-authorize-finalization-only-correction',action='store_true')
    a=p.parse_args()
    if a.offline_preflight:
        need(not a.i_authorize_finalization_only_correction,
             'Do not pass execution authorization during preflight')
        print('F29R6 — corrected regime-label finalization-only source review')
        setup()
        print('F29R6 OFFLINE PREFLIGHT PASS — NO FINAL SCORING OR RAW TEST ACCESS')
    else:
        need(a.i_authorize_finalization_only_correction,
             'Explicit finalization-only recovery acknowledgement required')
        finalization()

if __name__=='__main__':
    try:
        cli()
    except Exception as exc:
        print('F29R6 FAIL-CLOSED:',type(exc).__name__,str(exc),file=sys.stderr)
        raise SystemExit(1)
