#!/usr/bin/env python3
"""F29R7 reproducibility report from two allowlisted DEVELOPMENT CSVs.

Does not import trained models, examine held-out results, read test files,
retrain models, choose new thresholds, or modify any frozen source artifact.
Only creates NEW derived reports inside a fresh --output-dir after validation.

Usage:
 python F29R7_STEP_R1_DEV_REFIT_REPRO_READONLY.py \
   --exp ../stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026

Optional: --verify-dev-manifest-hashes validates referenced DEVELOPMENT-only
fit_manifest.json file hashes (reads in stage1_fits and stage2_robustness_fits).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_MODELS = {
    "DLINEAR": "DLINEAR_B",
    "NLINEAR": "NLINEAR_B",
    "PATCHTST": "PATCHTST_B",
    "TIMESNET": "TIMESNET_B",
    "ITRANSFORMER": "ITRANSFORMER_A",
    "FREDFORMER": "FREDFORMER_B",
    "FRNET": "FRNET_A",
    "MLF": "MLF_A",
}
SEEDS = (42, 43, 44, 45, 46)
FOLDS = (1, 2, 3, 4, 5)
SOURCE_1 = "33_final_development_refit_epoch_schedule.csv"
SOURCE_2 = "33_refit_duration_source_best_epochs.csv"


def fail(msg: str) -> None:
    raise RuntimeError("FAIL CLOSED: " + msg)


def sha256_file(p: Path) -> str:
    hasher = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_csv(path: Path, columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"Missing allowlisted development source: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        present = set(reader.fieldnames or [])
        if not columns.issubset(present):
            fail(f"Missing required columns in {path.name}: {sorted(columns - present)}")
        records = list(reader)
    return records


def int_field(row: dict[str, str], key: str, location: str) -> int:
    try:
        return int(row[key])
    except (ValueError, TypeError, KeyError) as exc:
        fail(f"Invalid {key} in {location}: {exc}")
        raise AssertionError from exc


def is_false(v: str) -> bool:
    return str(v).strip().lower() in {"false", "0"}


def label(m: str) -> str:
    return {
        "DLINEAR": "DLinear", "NLINEAR": "NLinear", "PATCHTST": "PatchTST",
        "TIMESNET": "TimesNet", "ITRANSFORMER": "iTransformer",
        "FREDFORMER": "Fredformer", "FRNET": "FRNet", "MLF": "MLF",
    }[m]


def latex_table(summary: list[dict], one_epoch_fits: int, one_epoch_refits: int) -> str:
    out = [
        "% Derived exclusively from development-only CSVs; DO NOT interpret as test evidence.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Authenticated development-selected epoch schedules for the eight neural families. "
        "Each architecture has 25 development-fit histories (five expanding folds and five seeds) "
        "and five final-development refits. Final durations are same-seed medians of the five "
        "development-fold best epochs; no final-refit validation or early stopping was performed.}",
        "\\label{tab:repro_refit_metadata}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llccccc}",
        "\\hline",
        "Architecture & Configuration & Best at epoch 1 & Median best epoch & "
        "Refit epochs (seeds 42--46) & Median refit & One-epoch refits \\\\ ",
        "\\hline",
    ]
    for r in summary:
        seq = ", ".join(str(x) for x in r["refit_epochs_by_seed"])
        out.append(f'{r["architecture"]} & \\texttt{{{r["config_id"]}}} & '
                   f'{r["development_best_at_epoch_1"]}/25 & '
                   f'{r["median_development_best_epoch"]:g} & '
                   f'{seq} & {r["median_final_refit_epoch"]:g} & '
                   f'{r["one_epoch_refits"]}/5 \\\\')
    out.extend([
        "\\hline",
        f"Overall & -- & {one_epoch_fits}/200 & -- & -- & -- & {one_epoch_refits}/40 \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, type=Path,
                        help="Root of frozen experiment; only audits/<two named csvs> read")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("F29R7_DEV_Refit_Metadata_Q1_v1"),
                        help="Fresh directory for derived output, not inside --exp (default: %(default)s)")
    parser.add_argument("--verify-dev-manifest-hashes", action="store_true",
                        help="Also verify SHA256 of 200 referenced development-only fit manifests")
    args = parser.parse_args()

    exp = args.exp.expanduser().resolve(strict=True)
    out = args.output_dir.expanduser().resolve()
    if exp == out or exp in out.parents:
        fail("Output must not be inside the sealed experiment directory")
    if out.exists():
        fail(f"Output directory exists: {out}; choose a new --output-dir")

    audits = exp / "audits"
    p1, p2 = audits / SOURCE_1, audits / SOURCE_2
    schedules = read_csv(p1, {
        "model_id", "config_id", "seed", "fold_best_epochs", "final_refit_epochs",
        "duration_rule", "validation_during_final_refit", "early_stopping_during_final_refit",
        "final_development_n", "logical_batch_size", "physical_microbatch_size",
        "best_seed_selection", "final_test_used",
    })
    histories = read_csv(p2, {
        "model_id", "config_id", "seed", "fold", "best_epoch", "epochs_run",
        "fit_manifest_path", "fit_manifest_sha256",
    })
    if len(schedules) != 40 or len(histories) != 200:
        fail(f"Unexpected record counts: schedules={len(schedules)}, histories={len(histories)}")

    sources_by_key: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
    hist_epoch1 = Counter()
    source_paths_verified = 0
    for i, rec in enumerate(histories, 2):
        location = f"{p2.name}:{i}"
        m, cfg = rec["model_id"], rec["config_id"]
        if m not in EXPECTED_MODELS or cfg != EXPECTED_MODELS[m]:
            fail(f"Unexpected model or config in {location}: {m}/{cfg}")
        seed, fold = int_field(rec, "seed", location), int_field(rec, "fold", location)
        best, executed = int_field(rec, "best_epoch", location), int_field(rec, "epochs_run", location)
        if seed not in SEEDS or fold not in FOLDS or best < 1 or executed < best:
            fail(f"Invalid seed/fold/best/executed epoch in {location}")
        if fold in sources_by_key[(m, seed)]:
            fail(f"Duplicate development fold for {m}, seed={seed}, fold={fold}")
        sources_by_key[(m, seed)][fold] = best
        hist_epoch1[m] += int(best == 1)
        manifest = Path(rec["fit_manifest_path"]).expanduser().resolve()
        allowed = [(exp / sub).resolve() for sub in ("stage1_fits", "stage2_robustness_fits")]
        if not any(directory in manifest.parents for directory in allowed) or manifest.name != "fit_manifest.json":
            fail(f"Non-development manifest path in {location}: {manifest}")
        if args.verify_dev_manifest_hashes:
            if not manifest.is_file():
                fail(f"Missing development fit manifest: {manifest}")
            if sha256_file(manifest) != rec["fit_manifest_sha256"].lower():
                fail(f"Development fit-manifest SHA mismatch: {manifest}")
            source_paths_verified += 1

    expected_keys = {(m, s) for m in EXPECTED_MODELS for s in SEEDS}
    if set(sources_by_key) != expected_keys:
        fail("Development-fold model/seed combinations do not match 8 x 5")
    if any(set(v) != set(FOLDS) for v in sources_by_key.values()):
        fail("One or more development histories lacks five folds")

    sched_by_key = {}
    per_model = defaultdict(list)
    for i, rec in enumerate(schedules, 2):
        location = f"{p1.name}:{i}"
        m, cfg = rec["model_id"], rec["config_id"]
        if m not in EXPECTED_MODELS or cfg != EXPECTED_MODELS[m]:
            fail(f"Unexpected model/config in {location}: {m}/{cfg}")
        seed = int_field(rec, "seed", location)
        if seed not in SEEDS or (m, seed) in sched_by_key:
            fail(f"Missing or repeated seed in {location}: {seed}")
        sched_by_key[(m, seed)] = rec
        listed = [int(v.strip()) for v in rec["fold_best_epochs"].split(",")]
        actual = [sources_by_key[(m, seed)][fold] for fold in FOLDS]
        if listed != actual:
            fail(f"Fold-best epoch mismatch for {m}, seed={seed}: {listed} != {actual}")
        expected_epoch = int(statistics.median(actual))
        observed_epoch = int_field(rec, "final_refit_epochs", location)
        if expected_epoch != observed_epoch:
            fail(f"Final duration rule mismatch for {m}, seed={seed}")
        if rec["duration_rule"] != "MEDIAN_OF_5_SAME_SEED_ROLLING_FOLD_BEST_EPOCHS":
            fail(f"Unrecognized final-refit epoch rule in {location}")
        for field in ("validation_during_final_refit", "early_stopping_during_final_refit",
                      "best_seed_selection", "final_test_used"):
            if not is_false(rec[field]):
                fail(f"Expected false for {field} in {location}")
        if (int_field(rec, "final_development_n", location) != 42360
            or int_field(rec, "logical_batch_size", location) != 256
            or int_field(rec, "physical_microbatch_size", location) != 128):
            fail(f"Unexpected sample/batch settings in {location}")
        per_model[m].append((seed, observed_epoch))
    if set(sched_by_key) != expected_keys:
        fail("Final-refit model/seed coverage is incomplete")

    summary = []
    for m, cfg in EXPECTED_MODELS.items():
        refit_pairs = sorted(per_model[m])
        refit = [epoch for _, epoch in refit_pairs]
        bests = [sources_by_key[(m, s)][f] for s in SEEDS for f in FOLDS]
        summary.append({
            "architecture": label(m), "model_id": m, "config_id": cfg,
            "development_fit_count": len(bests),
            "development_best_at_epoch_1": hist_epoch1[m],
            "median_development_best_epoch": statistics.median(bests),
            "refit_epochs_by_seed": refit,
            "min_final_refit_epoch": min(refit),
            "median_final_refit_epoch": statistics.median(refit),
            "max_final_refit_epoch": max(refit),
            "one_epoch_refits": sum(x == 1 for x in refit),
        })
    total_best1 = sum(x["development_best_at_epoch_1"] for x in summary)
    total_refit1 = sum(x["one_epoch_refits"] for x in summary)
    if (total_best1, total_refit1) != (121, 27):
        fail(f"Does not reconcile with frozen manuscript totals: best1={total_best1} (121 expected), one_refit={total_refit1} (27 expected)")

    combined = {
        "scope": "development-only, two frozen audit CSVs",
        "experimental_results_modified": False,
        "final_test_used": False,
        "csv_sha256": {SOURCE_1: sha256_file(p1), SOURCE_2: sha256_file(p2)},
        "checked_rows": {SOURCE_1: len(schedules), SOURCE_2: len(histories)},
        "checked_dev_manifests": source_paths_verified,
        "same_seed_median_rule_verified": True,
        "final_refits_no_validation_or_early_stopping": True,
        "development_best_epoch_1": total_best1,
        "one_epoch_final_refits": total_refit1,
        "logical_batch_size": 256,
        "physical_microbatch_size": 128,
        "final_development_endpoints": 42360,
        "models": summary,
    }

    # Create only new derived artifacts, after completing all read-only checks.
    out.mkdir(parents=True, exist_ok=False)
    with (out / "F29R7_Development_Refit_Reproducibility.csv").open("x", newline="", encoding="utf-8") as fh:
        fields = ["architecture", "model_id", "config_id", "development_fit_count",
                  "development_best_at_epoch_1", "median_development_best_epoch",
                  "refit_epochs_by_seed", "min_final_refit_epoch", "median_final_refit_epoch",
                  "max_final_refit_epoch", "one_epoch_refits"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({**row, "refit_epochs_by_seed": ",".join(map(str, row["refit_epochs_by_seed"]))})
    (out / "F29R7_Development_Refit_Reproducibility.tex").write_text(
        latex_table(summary, total_best1, total_refit1), encoding="utf-8")
    (out / "F29R7_Development_Refit_Reproducibility.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8")

    print("=" * 102)
    print("F29R7 DEVELOPMENT-ONLY REFIT REPRODUCIBILITY — VERIFIED")
    print("=" * 102)
    print(f"Source rows: schedules={len(schedules)}, histories={len(histories)}")
    print(f"Per-model: 25 development histories, 5 final refits; seeds {SEEDS}; folds {FOLDS}")
    print(f"Same-seed median-of-five-folds: PASS for all {len(schedules)} refits")
    print(f"Best epoch 1 in development: {total_best1}/200")
    print(f"One-epoch final refits: {total_refit1}/40")
    print(f"Final development n=42,360; batch logical=256; microbatch=128")
    print(f"Development manifest hashes verified: {source_paths_verified}/200" if args.verify_dev_manifest_hashes else
          "Development manifest hashes: not rechecked (use --verify-dev-manifest-hashes if needed)")
    print("-")
    for s in summary:
        epochs = ",".join(map(str, s["refit_epochs_by_seed"]))
        print(f'{s["architecture"]:13s} {s["config_id"]:16s} best1={s["development_best_at_epoch_1"]:2d}/25  '
              f'refit=[{epochs}]  one-epoch={s["one_epoch_refits"]}/5')
    print("-")
    print(f"[PASS] All checks. New derived files: {out}")
    for p in out.iterdir():
        print(f"       {p.name}")
    print("No test data accessed. Original experiment files unchanged.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
