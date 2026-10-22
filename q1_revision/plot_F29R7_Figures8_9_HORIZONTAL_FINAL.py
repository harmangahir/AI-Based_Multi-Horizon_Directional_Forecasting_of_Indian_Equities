#!/usr/bin/env python3
"""Publication-scale F29R7 Figures 8 (BA) and 9 (ROC-AUC).

Reads ONLY previously committed aggregate classification-metric CSVs after
SHA-256 verification of the immutable final-evaluation receipt and manifest.
Does NOT read raw-test arrays, probabilities, checkpoints, labels, calibration
bins, or execute training, inference, scoring, or bootstrap resampling.

Usage from q1_revision (Harman_GPU):
    python plot_f29r7_figures_8_9_Q1_CORRECTED.py --verify-only
    python plot_f29r7_figures_8_9_Q1_CORRECTED.py \\
        --figures both --formats pdf png --dpi 600 \\
        --output-dir F29R7_Figures_8_9_Q1_Refined

Both vector PDFs are set at approximately 7.4 x 8.9 inches. Keep LaTeX placement near
full-page size; 600-dpi output does not compensate for downscaled labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

RECEIPT_SHA256 = "637598ea722cf6083a3797d6de467ad91c6d3287fb2819edb92e90ff850af628"
EVAL_MANIFEST_SHA256 = "47e7820dafe9ec8f951f8daa45ce77aae37599d872bc631a4657f77c96903512"
STEP35_SHA256 = "6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7"

HORIZONS = (1, 3, 5)
NEURAL = ("DLINEAR", "NLINEAR", "PATCHTST", "TIMESNET",
          "ITRANSFORMER", "FREDFORMER", "FRNET", "MLF")
ML_MODELS = ("LOGREG", "RANDOM_FOREST", "XGBOOST")
ALL_MODELS = NEURAL + ML_MODELS
DISPLAY = {
    "DLINEAR": "DLinear", "NLINEAR": "NLinear", "PATCHTST": "PatchTST",
    "TIMESNET": "TimesNet", "ITRANSFORMER": "iTransformer",
    "FREDFORMER": "Fredformer", "FRNET": "FRNet", "MLF": "MLF",
    "LOGREG": "Logistic regression", "RANDOM_FOREST": "Random Forest",
    "XGBOOST": "XGBoost",
}
FILENAMES = {
    "ba": "fig18_all_models_BA_HORIZONTAL",
    "auc": "fig19_all_models_ROCAUC_HORIZONTAL",
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing or unsafe file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load_authenticated_metrics(exp):
    require(exp.is_dir(), f"Experiment directory not found: {exp}")
    receipt_path = exp / "manifests/36i_amended_final_test_evaluation_manifest.json"
    contract_path = exp / "manifests/35_final_test_evaluation_contract_freeze_manifest.json"
    results = (exp / "final_test_evaluation/step36i_amended_predictions"
                  / "completed_evaluation")
    manifest_path = results / "evaluation_manifest.json"

    require(sha256(receipt_path) == RECEIPT_SHA256,
            "Frozen public receipt SHA-256 mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("final_evaluation_manifest_sha256") == EVAL_MANIFEST_SHA256,
            "Receipt does not bind the expected completed-evaluation manifest")
    require(sha256(manifest_path) == EVAL_MANIFEST_SHA256,
            "Evaluation manifest SHA-256 mismatch")
    require(sha256(contract_path) == STEP35_SHA256,
            "Original evaluation contract SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "STEP36I_F29R6_FINAL_EVALUATION_REGIME_LABEL_RECOVERY_V1",
            "Unexpected source evaluation identity")
    require(manifest.get("regime_label_correction") == "ID_1_NORMAL_NOT_MEDIUM",
            "Unexpected regime-label identity")
    for key, expected in (("metrics", 24), ("statistics", 72),
                          ("primary_bh_tests", 24), ("ml_baseline_model_count", 9)):
        require(manifest.get(key) == expected, f"Source manifest count mismatch: {key}")

    entries = manifest.get("artifacts", [])
    require(len(entries) == 23 and len({e["name"] for e in entries}) == 23,
            "Expected exactly 23 committed evaluation artifacts")
    require({p.name for p in results.iterdir()}
            == {e["name"] for e in entries} | {"evaluation_manifest.json"},
            "Completed-evaluation artifact inventory changed")
    hashes = {}
    for entry in entries:
        filename = entry["name"]
        require(Path(filename).name == filename and filename not in (".", ".."),
                f"Unsafe manifest artifact name: {filename}")
        p = results / filename
        require(p.stat().st_size == int(entry["bytes"]),
                f"Committed source size mismatch: {filename}")
        digest = sha256(p)
        require(digest == entry["sha256"],
                f"Committed source SHA-256 mismatch: {filename}")
        hashes[filename] = digest

    def read_metrics(filename, expected_models, expected_rows):
        frame = pd.read_csv(results / filename)
        require(len(frame) == expected_rows, f"Wrong row count in {filename}")
        require({"model_id", "horizon_days", "balanced_accuracy"}.issubset(frame.columns),
                f"Missing mandatory metric columns in {filename}: {frame.columns.tolist()}")
        auc_columns = [k for k in ("roc_auc", "roc_auc_score", "auc_roc") if k in frame.columns]
        require(len(auc_columns) == 1,
                f"Expected one ROC-AUC column in {filename}; found {auc_columns}; available {frame.columns.tolist()}")
        frame = frame.rename(columns={auc_columns[0]: "roc_auc"})
        require(not frame.duplicated(["model_id", "horizon_days"]).any(),
                f"Duplicate model/horizon in {filename}")
        require(set(frame.model_id) == set(expected_models)
                and set(frame.horizon_days) == set(HORIZONS),
                f"Incomplete model/horizon coverage in {filename}")
        for key in ("balanced_accuracy", "roc_auc"):
            frame[key] = pd.to_numeric(frame[key], errors="raise")
            require(np.isfinite(frame[key]).all() and frame[key].between(0, 1).all(),
                    f"Nonfinite or invalid {key} in {filename}")
        if "n" in frame.columns:
            require((pd.to_numeric(frame["n"], errors="raise") == 5300).all(),
                    f"Different held-out sample size in {filename}")
        return frame[["model_id", "horizon_days", "balanced_accuracy", "roc_auc"]].copy()

    neural = read_metrics("classification_metrics.csv", NEURAL, 24)
    ml = read_metrics("ml_devonly_classification_metrics.csv", ML_MODELS, 9)
    combined = pd.concat([neural, ml], ignore_index=True)
    require(len(combined) == 33 and not combined.duplicated(["model_id", "horizon_days"]).any(),
            "Expected 33 unique common-cohort model/horizon metrics")

    # Manuscript values are rounding checks, not data sources. Fail closed if
    # frozen aggregate values disagree with the already-reviewed manuscript.
    def anchor(model, horizon, key, published):
        rows = combined[(combined.model_id == model)
                        & (combined.horizon_days == horizon)]
        require(len(rows) == 1, f"Missing anchor {model}/{horizon}")
        recorded = float(rows.iloc[0][key])
        require(abs(recorded - published) < 0.000015,
                f"Frozen value for {model}/{horizon}/{key} conflicts with manuscript: {recorded} vs {published}")
    anchor("DLINEAR", 1, "balanced_accuracy", 0.49950)
    anchor("PATCHTST", 3, "balanced_accuracy", 0.50118)
    anchor("ITRANSFORMER", 5, "balanced_accuracy", 0.50975)
    anchor("ITRANSFORMER", 5, "roc_auc", 0.50092)
    anchor("RANDOM_FOREST", 1, "balanced_accuracy", 0.51087)
    anchor("RANDOM_FOREST", 3, "balanced_accuracy", 0.51258)
    anchor("RANDOM_FOREST", 3, "roc_auc", 0.51954)
    anchor("RANDOM_FOREST", 5, "balanced_accuracy", 0.50233)
    print("PASS: Receipt, original contract, manifest and all 23 source artifact hashes")
    print("PASS: 24 frozen neural + 9 exploratory ML metrics, same 5,300 endpoints")
    print("PASS: BA/ROC-AUC manuscript-anchor reconciliation")
    context = {
        "receipt_sha256": RECEIPT_SHA256,
        "completed_evaluation_manifest_sha256": EVAL_MANIFEST_SHA256,
        "original_contract_sha256": STEP35_SHA256,
        "aggregate_neural_metrics_sha256": hashes["classification_metrics.csv"],
        "aggregate_ml_metrics_sha256": hashes["ml_devonly_classification_metrics.csv"],
        "all_23_evaluation_artifacts_sha256_verified": True,
        "evaluation_stock_session_endpoints_per_horizon": 5300,
    }
    return combined, context


def style(font):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": font,
        "axes.titlesize": font + 1.3, "axes.titleweight": "semibold",
        "axes.labelsize": font, "xtick.labelsize": font - 0.3,
        "ytick.labelsize": font - 0.15, "legend.fontsize": font - 0.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def plot_metric(df, which, width, font):
    """HORIZONTAL Figures 8 and 9: one row x three forecasting horizons.

    Plot only already authenticated 33 aggregate model/horizon values.
    Same fixed x axis, model ordering, and 50% reference in both figures.
    """
    require(which in ('ba', 'auc'), 'Unsupported metric')
    column = 'balanced_accuracy' if which == 'ba' else 'roc_auc'
    metric = 'Balanced accuracy' if which == 'ba' else 'ROC-AUC'
    require(len(df) == 33 and not df.duplicated(['model_id','horizon_days']).any(),
            'Expected 33 distinct aggregate records')
    # Wide, three-panel landscape arrangement; avoid portrait 3x1 figure.
    fig, axes = plt.subplots(1, 3, figsize=(width, width*.46), sharey=True)
    fig.subplots_adjust(left=.17, right=.99, top=.765, bottom=.18, wspace=.18)
    blue, orange = '#226d9c', '#c77832'
    left, right = 45.8, 55.0  # common scale in all 6 subplots (BA + AUC)
    ticks = [46, 48, 50, 52, 54]
    y = np.arange(len(ALL_MODELS))
    for panel, (ax, horizon) in enumerate(zip(axes, HORIZONS)):
        rows = df.loc[df.horizon_days == horizon].set_index('model_id')
        require(len(rows) == 11 and set(rows.index) == set(ALL_MODELS),
                f'Incomplete data for horizon {horizon}')
        ax.set_facecolor('white')
        ax.set_axisbelow(True)
        ax.axhspan(7.55, 10.53, color='#f4f5f6', zorder=0)
        ax.axhline(7.5, color='#c9cfd4', lw=.7, zorder=1)
        for i, model in enumerate(ALL_MODELS):
            value = float(rows.loc[model, column])*100.
            require(left+.2 <= value <= right-1.0,
                    f'{model}/{horizon}/{column} outside shared plotting scale')
            is_ml = i >= len(NEURAL)
            c = orange if is_ml else blue
            ax.scatter(value, i, color=c, marker='D' if is_ml else 'o',
                       s=40 if is_ml else 34, linewidth=.45,
                       edgecolor='white', zorder=4)
            # Place numerical values immediately to the right of their marker
            # (within shared axes) rather than outside a neighboring panel.
            ax.annotate(f'{value:.2f}', xy=(value,i), xytext=(5,0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=font-1.9, color='#343a40', clip_on=True)
        ax.axvline(50, color='#36393b', ls=(0, (4, 3)), lw=1.1, zorder=2)
        ax.set_xlim(left,right)
        ax.set_ylim(10.6,-.6)
        ax.set_xticks(ticks)
        ax.set_yticks(y)
        ax.set_yticklabels([DISPLAY[m] for m in ALL_MODELS])
        ax.tick_params(axis='y', length=0, labelsize=font-.9, pad=7)
        if panel > 0:
            ax.tick_params(axis='y', left=False, labelleft=False)
        ax.tick_params(axis='x', length=0, labelsize=font-.9)
        ax.grid(axis='x', color='#dfe3e7', lw=.65, alpha=.9)
        ax.set_title(f'{chr(65+panel)}  |  {horizon}-session',
                     loc='left', pad=10, fontsize=font+.1)
        ax.set_xlabel(f'{metric} (%)', labelpad=8, fontsize=font-.55)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#b7bec4')

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color='none', marker='o', markerfacecolor=blue,
               markeredgecolor='white', markersize=6.5, label='Neural ensembles (8)'),
        Line2D([], [], color='none', marker='D', markerfacecolor=orange,
               markeredgecolor='white', markersize=6.5, label='Exploratory ML (3)'),
        Line2D([], [], color='#36393b', ls=(0,(4,3)),
               lw=1.1, label='50% reference')]
    fig.suptitle(f'Held-out {metric.lower()} across all 11 model families',
                 y=.967, fontsize=font+1.5, fontweight='semibold')
    fig.legend(handles=handles, ncol=3, frameon=False,
               loc='upper center', bbox_to_anchor=(.5,.89),
               fontsize=font-1., columnspacing=1.4, handletextpad=.4)
    fig.text(.5, .062,
             'Same 5,300 endpoints per horizon; 24 neural BA tests are primary; '
             'ML metrics are exploratory.',
             ha='center', fontsize=font-1.1, color='#4b5563')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--experiment-root", type=Path,
                        default=here.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026")
    parser.add_argument("--output-dir", type=Path,
                        default=here / "F29R7_Figures_8_9_Q1_Refined")
    parser.add_argument("--figures", nargs="+", choices=("ba", "auc", "both"),
                        default=["both"])
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png"),
                        default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--width", type=float, default=11.5)
    parser.add_argument("--font-size", type=float, default=12.0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    require(150 <= args.dpi <= 1200, "DPI must be 150--1200")
    require(10.5 <= args.width <= 14.5, "Width must be 10.5--14.5 inches")
    require(10.0 <= args.font_size <= 13.0, "Font size must be 10.0--13.0 pt")
    exp = args.experiment_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    require(out != exp and not out.is_relative_to(exp),
            "Refusing to write inside the immutable experiment directory")
    selected = list(FILENAMES) if "both" in args.figures else list(dict.fromkeys(args.figures))
    print("F29R7 FIGURES 8 AND 9 — HORIZONTAL 1x3 — VERIFIED AGGREGATES")
    print("Experiment:", exp)
    print("Output:", out)
    df, context = load_authenticated_metrics(exp)
    if args.verify_only:
        print("VERIFY-ONLY PASS: no figures written, no experiment files modified")
        return
    targets = [out / f"{FILENAMES[k]}.{ext}"
               for k in selected for ext in dict.fromkeys(args.formats)]
    provenance = out / "F29R7_Figures_8_9_Provenance.json"
    require(not any(t.exists() for t in targets) and not provenance.exists(),
            "Output exists: select a new --output-dir (no overwriting)")
    out.mkdir(parents=True, exist_ok=True)
    style(args.font_size)
    generated = {}
    for which in selected:
        fig = plot_metric(df, which, args.width, args.font_size)
        try:
            for ext in dict.fromkeys(args.formats):
                target = out / f"{FILENAMES[which]}.{ext}"
                fig.savefig(target, dpi=args.dpi, format=ext,
                            facecolor="white", bbox_inches="tight", pad_inches=0.13)
                generated[target.name] = sha256(target)
                print(f"PASS {target.name} | SHA-256 {generated[target.name]}")
        finally:
            plt.close(fig)
    info = {
        "schema": "F29R7_Q1_11_MODELS_3_HORIZONS_BA_AUC_READONLY_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authenticated_input": context,
        "generator_sha256": sha256(Path(__file__).resolve()),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "figures": selected, "formats": args.formats, "raster_dpi": args.dpi,
        "vector_pdf": True, "panel_layout": "1_row_by_3_columns", "figure_sha256": generated,
        "training_performed": False, "inference_performed": False,
        "bootstrap_recomputed": False, "thresholds_changed": False,
        "sealed_raw_test_opened": False,
        "previously_committed_held_out_aggregate_metrics_read": True,
        "comparators_are_exploratory": True,
    }
    provenance.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUCCESS: Two journal figures generated from frozen aggregate metrics only")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError, ImportError) as e:
        print(f"F29R7 FAIL-CLOSED: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(2)
