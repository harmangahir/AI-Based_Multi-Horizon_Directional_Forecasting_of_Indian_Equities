#!/usr/bin/env python3
"""F29R7 R2: read-only schema discovery in HASHED DEVELOPMENT fit manifests.

Safety boundaries:
  * Opens only audits/33_refit_duration_source_best_epochs.csv and the
    fit_manifest.json files named there, inside stage1_fits or
    stage2_robustness_fits.
  * Verifies each opened manifest's SHA-256 before parsing JSON.
  * Never scans test folders, loads checkpoints, recalculates predictions,
    trains models, reads thresholds, or modifies archived source files.
  * Prints metadata candidates, not inferred experimental values.

This is DISCOVERY, not a finished hyperparameter or chronological-fold audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {
    'DLINEAR': 'DLINEAR_B', 'NLINEAR': 'NLINEAR_B',
    'PATCHTST': 'PATCHTST_B', 'TIMESNET': 'TIMESNET_B',
    'ITRANSFORMER': 'ITRANSFORMER_A', 'FREDFORMER': 'FREDFORMER_B',
    'FRNET': 'FRNET_A', 'MLF': 'MLF_A',
}
SEEDS = set(range(42, 47))
FOLDS = set(range(1, 6))
ACCEPTED_ROOTS = ('stage1_fits', 'stage2_robustness_fits')
# Only expose scientific metadata candidates: never print test/path/checkpoint keys.
SCIENTIFIC = re.compile(
    r'(learning[_-]?rate|(?:^|\.)lr$|weight[_-]?decay|batch[_-]?size|microbatch|'
    r'parameters?|n[_-]?params|param[_-]?count|dropout|hidden|d_model|n_heads|'
    r'epochs?|patience|train[_-]?(?:start|end)|val(?:idation)?[_-]?(?:start|end)|'
    r'purge|embargo|split[_-]?date|fold[_-]?(?:start|end))', re.I
)
SUPPRESS = re.compile(
    r'(sha256|hash|path|file|checkpoint|device|test|target|label|predict|'
    r'held[_-]?out|result|score|auc|accuracy|loss|error|metric|token|url|api|key)', re.I
)


def stop(s: str) -> None:
    raise RuntimeError('FAIL CLOSED: ' + s)


def hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def discover(node, prefix='', depth=0):
    if depth > 10:
        return
    if isinstance(node, dict):
        for key, value in node.items():
            trail = f'{prefix}.{key}' if prefix else str(key)
            yield from discover(value, trail, depth + 1)
    elif isinstance(node, list):
        # Fixed-size metadata lists only; avoid iterating training arrays.
        if len(node) <= 12:
            for i, v in enumerate(node):
                yield from discover(v, f'{prefix}[{i}]', depth + 1)
    elif isinstance(node, (str, float, int, bool)) or node is None:
        key = prefix.split('.')[-1]
        if SCIENTIFIC.search(key) and not SUPPRESS.search(prefix):
            sval = str(node)
            if len(sval) <= 120 and '/' not in sval and '\\' not in sval:
                yield prefix, node


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exp', required=True, help='Path to the unchanged F29R7 experiment')
    ap.add_argument('--output-dir', help='Optional NEW directory for derived JSON; must not exist')
    args = ap.parse_args()

    root = Path(args.exp).expanduser().resolve(strict=True)
    csv_file = root / 'audits' / '33_refit_duration_source_best_epochs.csv'
    if not csv_file.is_file():
        stop(f'Development CSV missing: {csv_file}')
    out = Path(args.output_dir).expanduser().absolute() if args.output_dir else None
    if out and out.exists():
        stop(f'Output directory already exists: {out}')
    if out and (out == root or root in out.parents):
        stop('Output directory must be outside the frozen experiment')

    with csv_file.open(newline='', encoding='utf-8-sig') as fh:
        data = list(csv.DictReader(fh))
    if len(data) != 200:
        stop(f'Expected 200 development history rows, found {len(data)}')
    seen = set()
    records = []
    for r in data:
        model = r.get('model_id', '').upper()
        conf = r.get('config_id', '')
        if model not in EXPECTED or conf != EXPECTED[model]:
            stop(f'Unknown model/config pair: {model}/{conf}')
        seed, fold = int(r['seed']), int(r['fold'])
        if seed not in SEEDS or fold not in FOLDS or (model, seed, fold) in seen:
            stop(f'Unexpected or duplicated model/seed/fold: {model}/{seed}/{fold}')
        seen.add((model, seed, fold))
        raw_path = Path(r['fit_manifest_path']).expanduser()
        candidate = (root / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
        if candidate.name != 'fit_manifest.json':
            stop(f'Invalid manifest basename: {candidate.name}')
        if not any(candidate.is_relative_to(root / x) for x in ACCEPTED_ROOTS):
            stop(f'Non-development path in manifest register: {candidate}')
        if not candidate.is_file():
            stop(f'Missing registered fit manifest: {candidate}')
        expected_sha = r['fit_manifest_sha256'].strip().lower()
        if not re.fullmatch(r'[a-f0-9]{64}', expected_sha):
            stop('Malformed registered manifest SHA-256')
        if hash_file(candidate) != expected_sha:
            stop(f'Development manifest hash mismatch: {candidate}')
        # All referenced JSON manifests are authenticated before summarizing.
        with candidate.open(encoding='utf-8') as fp:
            doc = json.load(fp)
        if not isinstance(doc, dict):
            stop('Expected JSON object in fit manifest')
        hits = dict(discover(doc))
        records.append({'model': model, 'configuration': conf, 'seed': seed,
                        'fold': fold, 'keys': sorted(doc.keys()), 'candidates': hits})

    if len(seen) != 8 * 5 * 5:
        stop(f'Incomplete model/seed/fold register: {len(seen)}')
    by_model = defaultdict(list)
    for r in records:
        by_model[r['model']].append(r)

    print('=' * 96)
    print('F29R7 — AUTHENTICATED DEVELOPMENT MANIFEST DISCOVERY (NO TEST ACCESS)')
    print('=' * 96)
    print(f'Authenticated and read {len(records)}/200 registered development fit manifests.')
    print('Only development-stage fit_manifest.json files accessed. No held-out files read.')
    print('Scientific field values below are MANIFEST CANDIDATES, not yet reconciled freeze specifications.')
    for model in EXPECTED:
        group = by_model[model]
        exemplar = next(x for x in group if x['seed'] == 42 and x['fold'] == 1)
        union_keys = set().union(*(set(x['candidates']) for x in group))
        print(f'\n{model} ({EXPECTED[model]}) — 25 authenticated manifests')
        print(f'  Top-level JSON keys: {", ".join(exemplar["keys"][:30]) or "(none)"}')
        print(f'  Scientific field names across 25 manifests: {", ".join(sorted(union_keys)[:35]) or "NOT FOUND"}')
        if exemplar['candidates']:
            print('  Example (seed 42, fold 1):')
            for key, value in list(sorted(exemplar['candidates'].items()))[:18]:
                print(f'    {key}: {value}')
    all_fields = sorted({key for r in records for key in r['candidates']})
    print('\nTOTAL distinct candidate metadata fields:', len(all_fields))
    print('Exact chronology, hyperparameters and parameter counts require separate key-specific verification.')
    print('Development-frozen thresholds are NOT read or inferred in this step.')
    if out:
        out.mkdir(parents=False, exist_ok=False)
        fname = out / 'F29R7_DEV_Manifest_Schema_Candidates.json'
        fname.write_text(json.dumps({'scope': 'development-only', 'manifest_hashes_verified': len(records),
                                    'fields': all_fields, 'records': records}, indent=2), encoding='utf-8')
        print('New derived JSON:', fname)
    print('[PASS] Discovery completed. Frozen experiment files unchanged.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(2)
