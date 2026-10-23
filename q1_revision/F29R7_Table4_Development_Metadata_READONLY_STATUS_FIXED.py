#!/usr/bin/env python3
"""F29R7 Table 4 metadata-only extraction and LaTeX authoring.

READ SCOPE, strictly:
  audits/33_final_development_refit_epoch_schedule.csv
  final_development_refits/<MODEL>/<CONFIG>/seed_<42..46>/fit_manifest.json

DOES NOT READ: raw data, held-out/test outcomes, predictions, trained model
weights, calibration/bootstraps, or any other experiment file.
Does not mutate the experiment or run training/inference.

Usage:
  python F29R7_Table4_Development_Metadata_READONLY.py --exp ../stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026 --verify-only
  python F29R7_Table4_Development_Metadata_READONLY.py --exp ../stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026 --out F29R7_Table4_Metadata_Q1

The output directory MUST NOT EXIST and MUST BE OUTSIDE the experiment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

CONFIG = [
    ('DLINEAR', 'DLINEAR_B', 'DLinear', 'Decomposition/linear temporal mapping'),
    ('NLINEAR', 'NLINEAR_B', 'NLinear', 'Last-value-normalized linear mapping'),
    ('PATCHTST', 'PATCHTST_B', 'PatchTST', 'Patch-based temporal tokens'),
    ('TIMESNET', 'TIMESNET_B', 'TimesNet', 'Period-aware temporal representation'),
    ('ITRANSFORMER', 'ITRANSFORMER_A', 'iTransformer', 'Variate-as-token attention'),
    ('FREDFORMER', 'FREDFORMER_B', 'Fredformer', 'Spectral/frequency debiasing'),
    ('FRNET', 'FRNET_A', 'FRNet', 'Frequency-based rotation'),
    ('MLF', 'MLF_A', 'MLF', 'Multi-period financial representation'),
]
SEEDS = range(42, 47)
SCHEDULE = Path('audits/33_final_development_refit_epoch_schedule.csv')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def strictly_named_file(root, relative):
    """Read exactly named development metadata, never follow external symlinks."""
    p = root / relative
    require(p.exists() and p.is_file(), f'Missing required development metadata: {relative}')
    require(not p.is_symlink(), f'Symlink disallowed: {relative}')
    require(p.resolve().is_relative_to(root), f'Escaping experiment root: {relative}')
    return p


def int_field(obj, key, description, minimum=0):
    v = obj.get(key)
    require(type(v) is int and v >= minimum, f'{description}: invalid {key}={v!r}')
    return v


def float_field(obj, key, description):
    v = obj.get(key)
    require(type(v) in (int, float) and math.isfinite(float(v)) and float(v) >= 0,
            f'{description}: missing/invalid {key}={v!r}')
    return float(v)


def fmt_scalar(v):
    # This produces ordinary LaTeX scientific notation, not an implied estimate.
    require(math.isfinite(v), 'Non-finite numeric value')
    if v == 0:
        return '0'
    if abs(v) < 0.001 or abs(v) >= 1000:
        return f'{v:.7g}'
    return f'{v:.7g}'


def tex_number(v):
    """Convert a verified float to LaTeX math, with no value modification."""
    txt = fmt_scalar(v)
    if 'e' in txt:
        coefficient, exponent = txt.split('e')
        return rf'${coefficient}\times 10^{{{int(exponent)}}}$'
    return rf'${txt}$'


def latex_escape(s):
    return s.replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def read_rows(root):
    inventory = []
    file_hashes = {}
    spath = strictly_named_file(root, SCHEDULE)
    file_hashes[str(SCHEDULE)] = sha256(spath)
    with spath.open(newline='', encoding='utf-8-sig') as f:
        schedule_rows = list(csv.DictReader(f))
    require(len(schedule_rows) == 40, f'Expected exactly 40 schedule rows; got {len(schedule_rows)}')
    schedule = {}
    for sr in schedule_rows:
        key = (sr.get('model_id'), sr.get('config_id'), sr.get('seed'))
        require(key not in schedule, f'Duplicate schedule entry: {key}')
        schedule[key] = sr

    by_model = {}
    for model, config, display, mechanism in CONFIG:
        recs = []
        for seed in SEEDS:
            rel = Path('final_development_refits') / model / config / f'seed_{seed}' / 'fit_manifest.json'
            p = strictly_named_file(root, rel)
            file_hashes[str(rel)] = sha256(p)
            with p.open(encoding='utf-8') as f:
                m = json.load(f)
            desc = f'{model}/{config}/seed_{seed}'
            for key, expected in [('model_id', model), ('config_id', config), ('seed', seed)]:
                require(m.get(key) == expected, f'{desc}: {key} mismatch, got {m.get(key)!r}')
            for key in ('validation_used', 'early_stopping_used', 'final_test_data_opened'):
                require(m.get(key) is False, f'{desc}: {key} is not explicitly False')
            # The authenticated F29R7 final-refit manifests use the exact,
            # stage-specific completed status below.  Keep a closed allowlist:
            # never accept arbitrary or partial statuses as successful.
            completed_statuses = (
                'FINAL_DEVELOPMENT_REFIT_COMPLETE',
                'complete', 'completed', 'success', 'succeeded',
            )
            require(m.get('status') in completed_statuses,
                    f'{desc}: status not completed: {m.get("status")!r}')
            count = int_field(m, 'final_development_n', desc, 1) if 'final_development_n' in m else None
            # Some frozen manifests record N in inputs rather than a top-level key.
            # Do not infer N when it is absent. It is described elsewhere in the paper.
            if count is not None:
                require(count == 42360, f'{desc}: unexpected final-development n={count}')
            sched = int_field(m, 'scheduled_epochs', desc, 1)
            actually_run = int_field(m, 'epochs_run', desc, 1)
            require(sched == actually_run, f'{desc}: scheduled_epochs != epochs_run')
            logical = int_field(m, 'logical_batch_size', desc, 1)
            micro = int_field(m, 'physical_microbatch_size', desc, 1)
            require(logical == 256 and micro <= logical, f'{desc}: batch rule inconsistent')
            trainable = int_field(m, 'trainable_parameter_count', desc, 1)
            total = int_field(m, 'total_parameter_count', desc, 1)
            require(trainable <= total, f'{desc}: trainable count exceeds total')
            lr = float_field(m, 'learning_rate', desc)
            wd = float_field(m, 'weight_decay', desc)
            require(lr > 0, f'{desc}: learning rate must be positive')
            s_key = (model, config, str(seed))
            require(s_key in schedule, f'{desc}: missing schedule CSV record')
            sr = schedule[s_key]
            require(str(sr.get('final_refit_epochs', '')).strip() == str(sched),
                    f'{desc}: schedule CSV differs: {sr.get("final_refit_epochs")} != {sched}')
            fold_epochs = [int(x.strip()) for x in str(sr.get('fold_best_epochs', '')).split(',') if x.strip()]
            require(len(fold_epochs) == 5 and all(v >= 1 for v in fold_epochs),
                    f'{desc}: invalid five-fold epoch list')
            sorted_folds = sorted(fold_epochs)
            require(sorted_folds[2] == sched, f'{desc}: median five-fold best epoch mismatch')
            require(str(sr.get('duration_rule', '')).strip() ==
                    'MEDIAN_OF_5_SAME_SEED_ROLLING_FOLD_BEST_EPOCHS',
                    f'{desc}: unexpected refit duration rule')
            for key in ('validation_during_final_refit', 'early_stopping_during_final_refit', 'final_test_used'):
                require(str(sr.get(key, '')).strip().lower() == 'false',
                        f'{desc}: schedule {key} not false')
            rec = dict(model=model, config=config, display=display, mechanism=mechanism,
                       seed=seed, epochs=sched, fold_best_epochs=fold_epochs,
                       logical_batch=logical, physical_microbatch=micro,
                       trainable_parameters=trainable, total_parameters=total,
                       learning_rate=lr, weight_decay=wd,
                       final_development_n=count, manifest_sha256=file_hashes[str(rel)])
            recs.append(rec)
            inventory.append(rec)
        # All five seed refits of one selected architecture should use the
        # same configuration and optimizer hyperparameters. Fail closed if not.
        for key in ('logical_batch', 'physical_microbatch', 'trainable_parameters',
                    'total_parameters', 'learning_rate', 'weight_decay'):
            values = [r[key] for r in recs]
            require(all(x == values[0] for x in values),
                    f'{model}: {key} not identical across five seed manifests: {values}')
        by_model[model] = dict(config=config, display=display, mechanism=mechanism,
                               logical_batch=recs[0]['logical_batch'],
                               physical_microbatch=recs[0]['physical_microbatch'],
                               trainable_parameters=recs[0]['trainable_parameters'],
                               total_parameters=recs[0]['total_parameters'],
                               learning_rate=recs[0]['learning_rate'],
                               weight_decay=recs[0]['weight_decay'],
                               epochs=[r['epochs'] for r in recs])
    require(len(inventory) == 40 and len(schedule) == 40, 'Incomplete 40-refit manifest set')
    require(sum(v == 1 for r in inventory for v in [r['epochs']]) == 27,
            'Expected 27 of 40 refits to use one epoch')
    return inventory, by_model, file_hashes


def make_latex(by_model):
    l = []
    l += [r'\begin{table}[htbp]', r'\centering',
          r'\caption{Audited development-selected neural configurations and final-refit metadata. Panel A reports recorded trainable PyTorch parameters and optimizer settings; Panel B reports the five frozen seed-specific refit durations and the architectural mechanisms. The logical batch size was 256 in every refit. Entries were extracted from 40 development-only final-refit manifests and reconciled against the frozen epoch-schedule register; no held-out performance was used.}',
          r'\label{tab:architectures}', r'\footnotesize',
          r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{1.12}',
          r'\vspace{2pt}',
          r'\textit{Panel A. Recorded model size and optimizer settings}\\[3pt]',
          r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l l r r r c@{}}',
          r'\toprule',
          r'\textbf{Family} & \textbf{Config.} & \shortstack{\textbf{Trainable}\\\textbf{parameters}} & \shortstack{\textbf{Learning}\\\textbf{rate}} & \shortstack{\textbf{Weight}\\\textbf{decay}} & \shortstack{\textbf{Batch}\\\textbf{logical/physical}} \\',
          r'\midrule']
    for model, config, display, mechanism in CONFIG:
        v = by_model[model]
        l.append(f"{latex_escape(display)} & \\texttt{{{latex_escape(config)}}} & {v['trainable_parameters']:,} & {tex_number(v['learning_rate'])} & {tex_number(v['weight_decay'])} & {v['logical_batch']}/{v['physical_microbatch']} \\\\")
    l += [r'\bottomrule', r'\end{tabular*}', r'\medskip',
          r'\textit{Panel B. Final-refit epochs and representative mechanisms}\\[3pt]',
          r'\begin{tabularx}{\linewidth}{@{}l l X@{}}', r'\toprule',
          r'\textbf{Family} & \shortstack{\textbf{Refit epochs}\\\textbf{seeds 42--46}} & \textbf{Representative mechanism} \\',
          r'\midrule']
    for model, config, display, mechanism in CONFIG:
        v = by_model[model]
        epoch_text = ', '.join(str(e) for e in v['epochs'])
        l.append(f'{latex_escape(display)} & {epoch_text} & {latex_escape(mechanism)} \\\\')
    l += [r'\bottomrule', r'\end{tabularx}', r'\vspace{2pt}',
          r'\par\raggedright\footnotesize\emph{Note.} Trainable parameter counts are the recorded PyTorch-registered parameter counts; they are not a compute-budget equivalence metric. The MLF implementation contained positional embeddings outside the standard PyTorch registration hierarchy, so its count does not include unregistered parameters. LR and weight decay are the recorded final-refit optimizer settings. The number of epochs for each seed is the median of five corresponding development-fold best-validation epochs; final refits used neither new validation nor early stopping.',
          r'\end{table}']
    return '\n'.join(l) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exp', type=Path, required=True, help='Frozen F29R7 experiment root')
    p.add_argument('--out', type=Path, default=Path('F29R7_Table4_Metadata_Q1'),
                   help='FRESH output folder outside experiment; pre-existing paths are rejected')
    p.add_argument('--verify-only', action='store_true', help='Read manifests and display summary; write nothing')
    args = p.parse_args()
    root = args.exp.expanduser().resolve(strict=True)
    require(root.is_dir(), 'Experiment root must be a directory')
    out = args.out.expanduser().absolute()
    if not args.verify_only:
        require(not out.resolve().is_relative_to(root), 'Refusing output within frozen experiment')
        require(not out.exists(), 'Existing output folder; choose a fresh --out (no overwrite)')
    inventory, by_model, hashes = read_rows(root)
    print('F29R7 — TABLE 4 — DEVELOPMENT-ONLY METADATA')
    print('Read files:', len(hashes), '= 40 final-refit manifests + 1 development schedule')
    print('Seed-specific refits:', len(inventory), '| Models:', len(by_model))
    print('No test data, inference, training, bootstrap or checkpoint loads.')
    print('Model        Params(trainable/total)  LR        WD       batch       epochs (42–46)')
    for model, v in by_model.items():
        print(f"{v['display']:<13}{v['trainable_parameters']:>9,}/{v['total_parameters']:<10,} "
              f"{fmt_scalar(v['learning_rate']):<9} {fmt_scalar(v['weight_decay']):<8} "
              f"{v['logical_batch']}/{v['physical_microbatch']:<7} {v['epochs']}")
    if args.verify_only:
        print('[PASS] verify-only; no output files written')
        return
    tex = make_latex(by_model)
    out.mkdir(parents=False, exist_ok=False)
    table_path = out / 'F29R7_Table4_Replacement.tex'
    table_path.write_text(tex, encoding='utf-8')
    allcsv = out / 'F29R7_Table4_All40_Metadata.csv'
    fields = ['model', 'config', 'seed', 'epochs', 'fold_best_epochs',
              'logical_batch', 'physical_microbatch', 'trainable_parameters',
              'total_parameters', 'learning_rate', 'weight_decay',
              'final_development_n', 'manifest_sha256']
    with allcsv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in inventory:
            w.writerow({k: (','.join(map(str, r[k])) if k == 'fold_best_epochs' else r[k]) for k in fields})
    audit = dict(title='F29R7 development-only Table 4 metadata extraction',
                 immutable_source=True, test_accessed=False, files_read_sha256=hashes,
                 all_40_records_checked=True,
                 single_epoch_refits=27,
                 generated_table_sha256=sha256(table_path),
                 generated_csv_sha256=sha256(allcsv),
                 model_summary=by_model)
    (out / 'F29R7_Table4_Provenance.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('[PASS] New Table 4 written:', table_path)
    print('[PASS] CSV:', allcsv)
    print('[PASS] Provenance:', out / 'F29R7_Table4_Provenance.json')
    print('No sealed experiment files modified.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'FAIL CLOSED: {type(exc).__name__}: {exc}') from exc
