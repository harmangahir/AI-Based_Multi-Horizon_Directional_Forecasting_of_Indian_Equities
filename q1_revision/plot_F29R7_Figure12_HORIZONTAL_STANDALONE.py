#!/usr/bin/env python3
"""Generate manuscript Figure 12 ONLY from signed, frozen aggregate results.

Three horizontal panels (1, 3, 5 sessions). Source values are read from the
SHA-256-authenticated final aggregate classification_metrics.csv. This file
NEVER reads raw test inputs, labels, checkpoints or model predictions, and
NEVER trains or resamples anything. It writes to a separate output directory.

Usage (from repository q1_revision):
  python plot_F29R7_Figure12_HORIZONTAL_STANDALONE.py --verify-only
  python plot_F29R7_Figure12_HORIZONTAL_STANDALONE.py --dpi 600
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

RECEIPT_SHA256 = "637598ea722cf6083a3797d6de467ad91c6d3287fb2819edb92e90ff850af628"
EVAL_MANIFEST_SHA256 = "47e7820dafe9ec8f951f8daa45ce77aae37599d872bc631a4657f77c96903512"
MODELS = ("DLINEAR", "NLINEAR", "PATCHTST", "TIMESNET", "ITRANSFORMER", "FREDFORMER", "FRNET", "MLF")
NAMES = ("DLinear", "NLinear", "PatchTST", "TimesNet", "iTransformer", "Fredformer", "FRNet", "MLF")
HORIZONS = (1, 3, 5)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_authenticated_rates(exp):
    """Validate immutable receipt, signed aggregate manifest, and all 23 files."""
    exp = exp.resolve()
    require(exp.is_dir(), f"Experiment directory does not exist: {exp}")
    receipt = exp / "manifests/36i_amended_final_test_evaluation_manifest.json"
    results = exp / "final_test_evaluation/step36i_amended_predictions/completed_evaluation"
    manifest = results / "evaluation_manifest.json"

    require(sha256(receipt) == RECEIPT_SHA256, "F29R7 public receipt fingerprint mismatch")
    rec = json.loads(receipt.read_text(encoding="utf-8"))
    require(rec.get("final_evaluation_manifest_sha256") == EVAL_MANIFEST_SHA256,
            "Receipt does not bind expected evaluation manifest")
    require(sha256(manifest) == EVAL_MANIFEST_SHA256,
            "Final evaluation manifest fingerprint mismatch")
    ledger = json.loads(manifest.read_text(encoding="utf-8"))
    require(ledger.get("schema") == "STEP36I_F29R6_FINAL_EVALUATION_REGIME_LABEL_RECOVERY_V1",
            "Unexpected frozen evaluation schema")
    entries = ledger.get("artifacts", [])
    require(len(entries) == 23 and len({e["name"] for e in entries}) == 23,
            "Expected exactly 23 unique frozen aggregate artifacts")
    require({p.name for p in results.iterdir()} == {e["name"] for e in entries} | {"evaluation_manifest.json"},
            "Unexpected aggregate result directory content")
    for entry in entries:
        name = entry["name"]
        require(Path(name).name == name and name not in (".", ".."), f"Unsafe artifact name: {name}")
        path = results / name
        require(path.stat().st_size == int(entry["bytes"]), f"Artifact size mismatch: {name}")
        require(sha256(path) == entry["sha256"], f"Artifact hash mismatch: {name}")
    print("[PASS] Authenticated signed F29R7 receipt, manifest and all 23 aggregate artifacts")

    frame = pd.read_csv(results / "classification_metrics.csv")
    needed = {"model_id", "horizon_days", "n", "tp", "tn", "fp", "fn",
              "predicted_positive_rate", "observed_positive_rate"}
    require(needed.issubset(frame.columns),
            f"Required columns missing from frozen metrics: {sorted(needed - set(frame.columns))}")
    require(len(frame) == 24 and not frame.duplicated(["model_id", "horizon_days"]).any(),
            "Expected 24 unique neural architecture/horizon rows")
    require(set(frame.model_id) == set(MODELS) and set(frame.horizon_days) == set(HORIZONS),
            "Unexpected neural architecture / horizon inventory")
    columns = ["n", "tp", "tn", "fp", "fn", "predicted_positive_rate", "observed_positive_rate"]
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        require(np.isfinite(frame[col].to_numpy(dtype=float)).all(), f"Nonfinite: {col}")
    require((frame.n == 5300).all() and
            (frame[["tp", "tn", "fp", "fn"]].sum(axis=1) == 5300).all(),
            "Endpoint count or confusion matrix mismatch")
    expected_pred = (frame.tp + frame.fp) / frame.n
    expected_obs = (frame.tp + frame.fn) / frame.n
    require(np.allclose(frame.predicted_positive_rate, expected_pred, atol=5e-7, rtol=0),
            "Predicted-up rate disagrees with the frozen confusion matrix")
    require(np.allclose(frame.observed_positive_rate, expected_obs, atol=5e-7, rtol=0),
            "Observed-up rate disagrees with the frozen confusion matrix")
    require(frame.predicted_positive_rate.between(0, 1).all() and
            frame.observed_positive_rate.between(0, 1).all(), "Invalid rate values")
    for h in HORIZONS:
        sub = frame.loc[frame.horizon_days == h]
        require(np.max(sub.observed_positive_rate) - np.min(sub.observed_positive_rate) < 5e-7,
                f"Observed class prevalence varies across models at horizon {h}")
    print("[PASS] All 24 model/horizon rates match the frozen confusion-matrix counts")
    return frame


def build_figure(frame, width=11.7, font=13.0):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": font,
        "axes.titlesize": font, "axes.labelsize": font - .3,
        "xtick.labelsize": font - .8, "ytick.labelsize": font - .6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(width, 4.7), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    yy = np.arange(8)[::-1]
    blue = "#17699b"
    for j, (ax, horizon) in enumerate(zip(axes, HORIZONS)):
        sub = frame.loc[frame.horizon_days == horizon].set_index("model_id").loc[list(MODELS)]
        predicted = sub.predicted_positive_rate.to_numpy(dtype=float) * 100
        observed = float(sub.observed_positive_rate.iloc[0] * 100)

        for y, rate in zip(yy, predicted):
            ax.plot([0, rate], [y, y], color="#c7d3db", linewidth=1.8,
                    solid_capstyle="round", zorder=1)
            ax.scatter(rate, y, s=50, color=blue, edgecolor="white", linewidth=.65, zorder=3)
            ax.text(105.1, y, f"{rate:.1f}%", ha="left", va="center",
                    fontsize=font - 1.2, color="#263341")
        ax.axvline(observed, linestyle="--", linewidth=1.2, color="#33383d", zorder=2)
        ax.set_xlim(-2.5, 126)
        ax.set_ylim(-.6, 7.6)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", alpha=.18, linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("Predicted upward (%)", labelpad=7)
        ax.set_title(f"{chr(65 + j)}  |  {horizon}-session\nObserved upward: {observed:.1f}%",
                     loc="left", pad=12, fontweight="semibold")
        ax.set_yticks(yy)
        if j == 0:
            ax.set_yticklabels(NAMES)
            ax.tick_params(axis="y", length=0, pad=6)
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)

    fig.suptitle("Directional prediction imbalance across eight neural ensembles",
                 x=.5, y=.975, fontsize=font + 2, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color=blue, lw=1.3, markersize=5,
               label="Frozen model-predicted upward rate"),
        Line2D([0], [0], linestyle="--", color="#33383d", lw=1.3,
               label="Observed upward rate"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(.5, .905), fontsize=font - 1.1, frameon=False)
    fig.subplots_adjust(left=.125, right=.992, top=.66, bottom=.21, wspace=.12)
    fig.text(.5, .065,
             "Same 5,300 stock-session endpoints per horizon; class-prediction rates are descriptive.",
             ha="center", fontsize=font - 1.3, color="#515d67")
    return fig


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path,
                   default=here.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026")
    p.add_argument("--output-dir", type=Path, default=here / "F29R7_Figure12_HORIZONTAL_Q1")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--width", type=float, default=11.7)
    p.add_argument("--font-size", type=float, default=13.0)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    require(150 <= a.dpi <= 1200 and 10 <= a.width <= 16 and 11 <= a.font_size <= 16,
            "Use dpi 150..1200, width 10..16 inches, font size 11..16 points")
    exp = a.experiment_root.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    require(exp != out and exp not in out.parents,
            "Refusing to write inside immutable experiment directory")
    frame = read_authenticated_rates(exp)
    if a.verify_only:
        print("VERIFY-ONLY PASS — no figures written or experiment files changed")
        return
    out.mkdir(parents=True, exist_ok=True)
    basename = "Figure12_Directional_Prediction_Imbalance_HORIZONTAL"
    paths = [out / f"{basename}.pdf", out / f"{basename}.png"]
    report = out / "Figure12_SHA256_Provenance.json"
    if not a.overwrite:
        require(not report.exists() and all(not p.exists() for p in paths),
                "Output already exists: choose a fresh --output-dir (recommended) or --overwrite")
    fig = build_figure(frame, a.width, a.font_size)
    try:
        for f in paths:
            fig.savefig(f, dpi=a.dpi, bbox_inches="tight", pad_inches=.10,
                        facecolor="white")
            print(f"[PASS] {f.name}: SHA-256 {sha256(f)}")
    finally:
        plt.close(fig)
    payload = {
        "schema": "F29R7_FIGURE12_PREDICTION_IMBALANCE_HORIZ_1x3_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "signed_receipt_sha256": RECEIPT_SHA256,
        "evaluation_manifest_sha256": EVAL_MANIFEST_SHA256,
        "source_metrics_sha256": sha256(exp / "final_test_evaluation/step36i_amended_predictions/completed_evaluation/classification_metrics.csv"),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "output_sha256": {f.name: sha256(f) for f in paths},
        "layout": "1 row x 3 columns (1/3/5 sessions)",
        "source_rows": 24,
        "no_raw_test_access_or_new_inference": True,
        "no_training_or_bootstrap": True,
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUCCESS: Figure 12 generated from authenticated aggregate metrics only")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError, ImportError) as e:
        print(f"F29R7 FIGURE12 FAIL-CLOSED: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(2)
