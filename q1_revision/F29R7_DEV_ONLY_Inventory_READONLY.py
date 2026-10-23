#!/usr/bin/env python3
"""Read-only, development-metadata inventory for F29R7.

No test data, model prediction files, or experiment checkpoints are read.
Only two explicitly named development audit CSVs under EXP/audits are inspected.
Outputs text to stdout, and never writes to the experiment folder.
"""
import argparse
import csv
from pathlib import Path

FILES = [
    '33_final_development_refit_epoch_schedule.csv',
    '33_refit_duration_source_best_epochs.csv',
]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp', type=Path, required=True, help='Experiment root directory')
    args=parser.parse_args()
    root=args.exp.expanduser().resolve()
    audits=(root / 'audits').resolve()
    if not root.is_dir() or not audits.is_dir() or audits.parent != root:
        parser.error('Experiment root and its audits/ directory must exist')
    print('F29R7 development-only metadata inventory')
    print('Experiment:', root)
    print('Scope: only the two named CSV files in audits/; no held-out test reads.')
    for filename in FILES:
        path=(audits / filename).resolve()
        print('\nFILE:', path)
        if path.parent != audits or not path.is_file():
            print('  [MISSING]');continue
        with path.open('r', encoding='utf-8-sig', newline='') as stream:
            reader=csv.DictReader(stream)
            cols=reader.fieldnames or []
            rows=list(reader)
        print('  Row count:', len(rows))
        print('  Columns:', ', '.join(cols))
        print('  First 2 records (development metadata only):')
        for i,row in enumerate(rows[:2],1):
            print('   ',i, ' | '.join(f'{k}={row[k]}' for k in cols))
    print('\nRead-only inventory completed. No original files were changed.')

if __name__=='__main__':
    main()
