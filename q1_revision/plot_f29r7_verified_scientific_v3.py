#!/usr/bin/env python3
"""Reproducible scientific plots from the IMMUTABLE Step-36I F29R7 results.

No result values are embedded in this program. Every plotted statistic is read
from the authenticated committed evaluation CSVs. The script does not open the
sealed raw final test, access seed predictions, fit models, or run bootstraps.

Dependency requirements: Python >= 3.9, pandas, numpy, matplotlib.

Example (from q1_revision):
    python plot_f29r7_verified_scientific.py --plots all --formats png pdf

Figure filenames 01--04 match those in Revision.tex. Figures 05--07 are
additional exploratory visualizations and should be disclosed as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd

MODEL_ORDER = (
    "DLINEAR", "NLINEAR", "PATCHTST", "TIMESNET",
    "ITRANSFORMER", "FREDFORMER", "FRNET", "MLF",
)
HORIZONS = (1, 3, 5)
MODEL_NAMES = {
    "DLINEAR": "DLinear", "NLINEAR": "NLinear", "PATCHTST": "PatchTST",
    "TIMESNET": "TimesNet", "ITRANSFORMER": "iTransformer",
    "FREDFORMER": "Fredformer", "FRNET": "FRNet", "MLF": "MLF",
    "RANDOM_FOREST": "Random Forest",
}
RECEIPT_SHA256 = "637598ea722cf6083a3797d6de467ad91c6d3287fb2819edb92e90ff850af628"
EVALUATION_SHA256 = "47e7820dafe9ec8f951f8daa45ce77aae37599d872bc631a4657f77c96903512"
REQUIRED_CSVS = {
    "classification_metrics.csv": {
        "model_id", "horizon_days", "n", "balanced_accuracy", "roc_auc",
        "tp", "tn", "fp", "fn",
    },
    "bootstrap_fdr.csv": {
        "model_id", "horizon_days", "block_length", "observed_BA",
        "ci_lower_percentile_95", "ci_upper_percentile_95", "q_BH_primary_only",
    },
    "ml_devonly_classification_metrics.csv": {
        "model_id", "horizon_days", "balanced_accuracy", "roc_auc",
    },
    "ml_devonly_neural_vs_ml_paired_block_ci.csv": {
        "neural_model", "baseline_model", "horizon_days",
        "observed_BA_neural_minus_ML", "ci_95_low", "ci_95_high",
    },
    "supplementary_regime_level.csv": {
        "model_id", "horizon_days", "group", "n", "balanced_accuracy",
    },
    "ml_devonly_regime_level.csv": {
        "model_id", "horizon_days", "group", "n", "balanced_accuracy",
    },
    "supplementary_calendar_quarter.csv": {
        "model_id", "horizon_days", "group", "n", "balanced_accuracy",
    },
    "ml_devonly_calendar_quarter.csv": {
        "model_id", "horizon_days", "group", "n", "balanced_accuracy",
    },
    "supplementary_stock_level.csv": {
        "model_id", "horizon_days", "group", "n", "balanced_accuracy",
    },
    "ml_devonly_stock_level.csv": {
        "model_id", "horizon_days", "group", "n", "balanced_accuracy",
    },
}
FIGURES = (
    "fig01_neural_BA_heatmap",
    "fig02_best_neural_vs_rf",
    "fig03_regime_5d",
    "fig04_quarter_5d",
    "fig05_primary_bootstrap_CI",
    "fig06_prediction_rate_diagnostics",
    "fig07_stock_level_heatmap",
)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"Missing or unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_finite_numeric(frame: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    for col in cols:
        require(col in frame.columns, f"{name}: missing column {col}")
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        require(bool(np.isfinite(frame[col].to_numpy(dtype=float)).all()),
                f"{name}: {col} contains missing/invalid numbers")


def equal_pairs(frame: pd.DataFrame, keys: list[str], expected: int, name: str) -> None:
    require(len(frame) == expected, f"{name}: expected {expected} rows, found {len(frame)}")
    require(not frame.duplicated(subset=keys).any(),
            f"{name}: duplicate keys {keys}")


def read_verified_results(exp: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    receipt = exp / "manifests/36i_amended_final_test_evaluation_manifest.json"
    evaluation = exp / (
        "final_test_evaluation/step36i_amended_predictions/completed_evaluation"
    )
    manifest_path = evaluation / "evaluation_manifest.json"
    require(sha256_file(receipt) == RECEIPT_SHA256,
            "F29R7 public receipt SHA-256 mismatch; refusing to plot")
    receipt_payload = json.loads(receipt.read_bytes())
    require(receipt_payload.get("final_evaluation_manifest_sha256") == EVALUATION_SHA256,
            "Public receipt does not reference the expected F29R6 result manifest")
    require(sha256_file(manifest_path) == EVALUATION_SHA256,
            "Committed evaluation-manifest SHA-256 mismatch; refusing to plot")
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest.get("schema") ==
            "STEP36I_F29R6_FINAL_EVALUATION_REGIME_LABEL_RECOVERY_V1" and
            manifest.get("regime_label_correction") == "ID_1_NORMAL_NOT_MEDIUM",
            "Unexpected evaluation schema or regime correction")
    counts = {
        "metrics": 24, "statistics": 72, "primary_bh_tests": 24,
        "ml_baseline_model_count": 9, "neural_vs_ml_exploratory_pairs": 72,
    }
    require(all(manifest.get(k) == v for k, v in counts.items()),
            "Unexpected verified-result scientific inventory")
    entries = manifest.get("artifacts")
    require(isinstance(entries, list) and len(entries) == 23,
            "Expected 23 immutable result artifacts")
    names = [e["name"] for e in entries]
    require(len(set(names)) == 23 and set(REQUIRED_CSVS).issubset(set(names)),
            "Results manifest does not contain unique required CSV files")
    # Authenticate all 23 committed artifacts, including parquets that we will not open.
    for item in entries:
        name = item["name"]
        require(Path(name).name == name and name not in (".", ".."),
                f"Unsafe result artifact name: {name}")
        file_path = evaluation / name
        require(file_path.stat().st_size == int(item["bytes"]),
                f"Artifact byte size changed: {name}")
        require(sha256_file(file_path) == item["sha256"],
                f"Artifact SHA-256 changed: {name}")
    require({p.name for p in evaluation.iterdir() if p.is_file()}
            == set(names) | {"evaluation_manifest.json"},
            "Unexpected committed evaluation-file inventory")
    data: dict[str, pd.DataFrame] = {}
    for filename, required_cols in REQUIRED_CSVS.items():
        frame = pd.read_csv(evaluation / filename)
        require(required_cols.issubset(set(frame.columns)),
                f"{filename}: unexpected columns; missing {required_cols-set(frame.columns)}")
        data[filename] = frame
    neural = data["classification_metrics.csv"]
    ml = data["ml_devonly_classification_metrics.csv"]
    boot = data["bootstrap_fdr.csv"]
    equal_pairs(neural, ["model_id", "horizon_days"], 24, "Neural")
    equal_pairs(ml, ["model_id", "horizon_days"], 9, "ML")
    equal_pairs(boot, ["model_id", "horizon_days", "block_length"], 72, "Bootstrap")
    require(set(neural.model_id) == set(MODEL_ORDER) and
            set(ml.model_id) == {"LOGREG", "RANDOM_FOREST", "XGBOOST"} and
            set(neural.horizon_days) == set(HORIZONS) and
            set(ml.horizon_days) == set(HORIZONS), "Unexpected model or horizon interface")
    safe_finite_numeric(neural,
                        ("balanced_accuracy", "roc_auc", "n", "tp", "tn", "fp", "fn"),
                        "Neural metrics")
    safe_finite_numeric(ml, ("balanced_accuracy", "roc_auc"), "ML metrics")
    safe_finite_numeric(boot,
                        ("observed_BA", "ci_lower_percentile_95",
                         "ci_upper_percentile_95"), "Bootstrap")
    require((neural.n == 5300).all(), "Primary metrics sample count differs from 5,300")
    require(((neural[["tp", "tn", "fp", "fn"]] >= 0).all().all()) and
            (neural[["tp", "tn", "fp", "fn"]].sum(axis=1) == neural.n).all(),
            "Neural confusion-matrix counts are inconsistent")
    # Exact observed/predicted up-rate from VERIFIED full-cohort confusion counts.
    neural["observed_positive_rate"] = (neural["tp"] + neural["fn"]) / neural["n"]
    neural["predicted_positive_rate"] = (neural["tp"] + neural["fp"]) / neural["n"]
    require(set(boot.block_length) == {10, 20, 40},
            "Expected the frozen bootstrap lengths 10, 20 and 40")
    primary = boot.loc[boot.block_length == 20].copy()
    require(len(primary) == 24, "Expected exactly 24 primary-block results")
    safe_finite_numeric(primary, ("q_BH_primary_only",), "Primary-block bootstrap")
    require(primary.q_BH_primary_only.between(0, 1).all(),
            "Primary adjusted q-values outside [0,1]")
    match = primary.merge(neural[["model_id", "horizon_days", "balanced_accuracy"]],
                          on=["model_id", "horizon_days"], validate="one_to_one")
    require(np.allclose(match.observed_BA, match.balanced_accuracy, rtol=0, atol=1e-10),
            "Disagreement between primary BA and reported classification BA")
    for filename in (
        "supplementary_regime_level.csv", "ml_devonly_regime_level.csv",
        "supplementary_calendar_quarter.csv", "ml_devonly_calendar_quarter.csv",
        "supplementary_stock_level.csv", "ml_devonly_stock_level.csv",
    ):
        frame = data[filename]
        require(not frame.duplicated(["model_id", "horizon_days", "group"]).any(),
                f"Duplicated subgroup records: {filename}")
        safe_finite_numeric(frame, ("n", "balanced_accuracy"), filename)
        require(frame.n.gt(0).all(), f"Empty group: {filename}")
    information = {
        "public_receipt_sha256": RECEIPT_SHA256,
        "committed_evaluation_manifest_sha256": EVALUATION_SHA256,
        "experiment_root": str(exp),
        "evaluation_dir": str(evaluation),
        "verified_artifact_sha256": {item["name"]: item["sha256"] for item in entries},
    }
    return data, information


def configure_style(size: float, minimal: bool = False) -> None:
    plt.rcParams.update({
        "figure.dpi": 115,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.size": size,
        "axes.titlesize": size + 2,
        "axes.titleweight": "semibold",
        "axes.labelsize": size,
        "xtick.labelsize": size - 0.3,
        "ytick.labelsize": size - 0.3,
        "legend.fontsize": size - 1,
        "figure.constrained_layout.use": False,
        "pdf.fonttype": 42,  # TrueType outlines/text, suited to vector production
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": minimal,
        "grid.alpha": 0.14,
        "axes.axisbelow": True,
    })


def footnote(fig: plt.Figure, text: str, size: float, y: float = 0.016, width: int = 130) -> None:
    wrapped = textwrap.fill(text, width=width)
    fig.text(0.012, y, wrapped, ha="left", va="bottom", fontsize=size - 2.1, linespacing=1.18)


def finalize_layout(fig: plt.Figure, bottom: float = 0.12, top: float = 0.94, left: float = 0.075, right: float = 0.985) -> None:
    fig.tight_layout(rect=(left, bottom, right, top), pad=0.8)


def fig01(data: dict, width: float, font: float):
    frame = data["classification_metrics.csv"]
    pv = frame.pivot(index="model_id", columns="horizon_days", values="balanced_accuracy")
    values = pv.loc[list(MODEL_ORDER), list(HORIZONS)].to_numpy(dtype=float) * 100
    require(np.isfinite(values).all(), "Missing model/horizon BA in heatmap")
    maxdev = max(float(np.abs(values - 50).max()), 0.5)
    fig, ax = plt.subplots(figsize=(width, width * 0.69))
    im = ax.imshow(values, cmap="RdBu", norm=TwoSlopeNorm(
        vmin=50 - maxdev, vcenter=50, vmax=50 + maxdev), aspect="auto")
    ax.set_xticks(range(3), ["1 session", "3 sessions", "5 sessions"])
    ax.set_yticks(range(8), [MODEL_NAMES[m] for m in MODEL_ORDER])
    ax.set_title("Five-seed ensemble: held-out balanced accuracy", pad=17)
    for j, h in enumerate(HORIZONS):
        best_idx = int(np.argmax(values[:, j]))
        for i in range(8):
            ax.text(j, i, f"{values[i,j]:.2f}%", ha="center", va="center",
                    fontsize=font - 0.3, fontweight="bold" if i == best_idx else "normal")
    cb = fig.colorbar(im, ax=ax, pad=0.035, shrink=0.87)
    cb.set_label("Balanced accuracy (%)")
    footnote(fig, "Identical n=5,300 endpoints per cell. Bold = descriptive horizon leader; no BH-FDR discovery.", font, y=0.018, width=110)
    finalize_layout(fig, bottom=0.14, top=0.93)
    return fig, {"best_neural_per_horizon": best_neural(data)}


def best_neural(data: dict) -> dict[int, str]:
    neural = data["classification_metrics.csv"]
    winners: dict[int, str] = {}
    for h in HORIZONS:
        sub = neural.loc[neural.horizon_days == h]
        winner = sub.sort_values(["balanced_accuracy", "model_id"],
                                 ascending=[False, True]).iloc[0]
        winners[h] = str(winner.model_id)
    return winners


def fig02(data: dict, width: float, font: float):
    winners = best_neural(data)
    neural = data["classification_metrics.csv"]
    ml = data["ml_devonly_classification_metrics.csv"]
    paired = data["ml_devonly_neural_vs_ml_paired_block_ci.csv"]
    b_neural = [float(neural.loc[(neural.model_id == winners[h]) &
                                 (neural.horizon_days == h), "balanced_accuracy"].iloc[0])
                for h in HORIZONS]
    b_rf = [float(ml.loc[(ml.model_id == "RANDOM_FOREST") &
                             (ml.horizon_days == h), "balanced_accuracy"].iloc[0])
            for h in HORIZONS]
    for j, h in enumerate(HORIZONS):
        row = paired.loc[(paired.neural_model == winners[h]) &
                         (paired.baseline_model == "RANDOM_FOREST") &
                         (paired.horizon_days == h)]
        require(len(row) == 1, f"Missing paired neural-versus-RF comparison: {winners[h]}, {h}")
        diff = float(row.iloc[0].observed_BA_neural_minus_ML)
        require(abs(diff - (b_neural[j] - b_rf[j])) < 1e-9,
                f"Paired delta and observed BA disagree for {h}d")
    fig, ax = plt.subplots(figsize=(width, width * 0.60))
    x = np.arange(3)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    y_neural = np.array(b_neural)*100
    y_rf = np.array(b_rf)*100
    # Dot plots, not truncated bars: magnify small differences transparently.
    for j in range(3):
        ax.plot([x[j]-0.13, x[j]+0.13], [y_neural[j], y_rf[j]],
                linestyle="-", linewidth=1.2, alpha=0.55, color="0.5", zorder=1)
    ax.scatter(x-0.13, y_neural, s=94, marker="o", color=cycle[0],
               label="Best observed neural (exploratory)", zorder=3)
    ax.scatter(x+0.13, y_rf, s=87, marker="s", color=cycle[1],
               label="Random Forest (supplementary)", zorder=3)
    ax.axhline(50, ls="--", lw=1.1, label="BA = 50% reference", color="0.42")
    for xx, yy in ((x-0.13,y_neural),(x+0.13,y_rf)):
        for xi, yi in zip(xx, yy):
            ax.annotate(f"{yi:.2f}%", (xi,yi), xytext=(0,8),
                        textcoords="offset points", ha="center", fontsize=font-.8)
    ax.set_xticks(x, [f"{h} session" + ("s" if h > 1 else "") + f"\n{MODEL_NAMES[winners[h]]}" for h in HORIZONS])
    ax.tick_params(axis="x", pad=6)
    ax.set_ylabel("Balanced accuracy (%)")
    observed = b_neural + b_rf
    ax.set_ylim(min(48, min(observed)*100 - 1), max(53.5, max(observed)*100 + 1.5))
    ax.set_title("Observed horizon leaders and conventional Random Forest", pad=15)
    ax.legend(loc="upper right", frameon=False)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(axis="y", alpha=0.15)
    footnote(fig, "Test-selected neural leaders are exploratory. Narrow y-axis highlights differences; no pairwise superiority established.", font, y=0.018, width=118)
    finalize_layout(fig, bottom=0.17, top=0.92)
    return fig, {"best_neural_per_horizon": winners}


def map_regime(group: str) -> str:
    mapping = {"0": "Low", "1": "Normal", "2": "High",
               "low": "Low", "normal": "Normal", "high": "High"}
    key = str(group).strip().lower()
    require(key in mapping, f"Unknown frozen volatility regime: {group}")
    return mapping[key]


def regime_frame(data: dict, filename: str, model: str, horizon: int) -> pd.DataFrame:
    frame = data[filename]
    sub = frame.loc[(frame.model_id == model) &
                    (frame.horizon_days == horizon)].copy()
    sub["regime"] = sub.group.map(map_regime)
    equal_pairs(sub, ["regime"], 3, filename + ":" + model)
    require(set(sub.regime) == {"Low", "Normal", "High"},
            f"Missing canonical regime in {filename}")
    return sub.set_index("regime").loc[["Low", "Normal", "High"]]


def fig03(data: dict, width: float, font: float):
    n = regime_frame(data, "supplementary_regime_level.csv", "ITRANSFORMER", 5)
    rf = regime_frame(data, "ml_devonly_regime_level.csv", "RANDOM_FOREST", 5)
    require(np.array_equal(n.n.to_numpy(dtype=int), rf.n.to_numpy(dtype=int)),
            "Mismatch in regime sample counts between models")
    fig, ax = plt.subplots(figsize=(width, width * 0.55))
    x = np.arange(3)
    y_n = n.balanced_accuracy.to_numpy(dtype=float)*100
    y_rf = rf.balanced_accuracy.to_numpy(dtype=float)*100
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j in range(3):
        ax.plot([x[j]-.12,x[j]+.12], [y_n[j], y_rf[j]],
                linestyle="-", linewidth=1.15, alpha=.55, color="0.5", zorder=1)
    ax.scatter(x-.12, y_n, s=102, marker="o", color=cycle[0],
               label="5-session iTransformer", zorder=3)
    ax.scatter(x+.12, y_rf, s=95, marker="s", color=cycle[1],
               label="5-session Random Forest", zorder=3)
    ax.axhline(50, linestyle="--", linewidth=1.1, color="0.42")
    ax.set_xticks(x, [f"{name}\nn={int(n.loc[name, 'n']):,}" for name in n.index])
    ax.tick_params(axis="x", pad=6)
    for xx, yy in ((x-.12,y_n),(x+.12,y_rf)):
        for xi, yi in zip(xx, yy):
            ax.annotate(f"{yi:.2f}%", (xi,yi), xytext=(0,8),
                        textcoords="offset points", ha="center", fontsize=font-.8)
    ax.set_ylim(max(0, min(y_n.min(),y_rf.min())-5),
                min(100, max(y_n.max(),y_rf.max())+6))
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Five-session performance varies across frozen volatility regimes", pad=14)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.15)
    footnote(fig, "Low / Normal / High labels are original (ID 0 / 1 / 2). Descriptive subgroups, no interaction hypothesis test.", font, y=0.018, width=115)
    finalize_layout(fig, bottom=0.19, top=0.92)
    return fig, {}


def fig04(data: dict, width: float, font: float):
    n = data["supplementary_calendar_quarter.csv"]
    rf = data["ml_devonly_calendar_quarter.csv"]
    n = n.loc[(n.model_id == "ITRANSFORMER") & (n.horizon_days == 5)].copy()
    rf = rf.loc[(rf.model_id == "RANDOM_FOREST") & (rf.horizon_days == 5)].copy()
    equal_pairs(n, ["group"], len(n), "Neural calendar quarters")
    equal_pairs(rf, ["group"], len(rf), "ML calendar quarters")
    require(set(n.group) == set(rf.group), "Different quarters across neural and ML")
    periods = pd.period_range(min(n.group), max(n.group), freq="Q")
    names = [str(p) for p in periods]
    n_vals = n.set_index("group").balanced_accuracy.reindex(names)*100
    r_vals = rf.set_index("group").balanced_accuracy.reindex(names)*100
    require(n_vals.notna().sum() == len(n) and r_vals.notna().sum() == len(rf),
            "Unable to align quarter records")
    missing = [name for name in names if pd.isna(n_vals.loc[name])]
    fig, ax = plt.subplots(figsize=(width, width * 0.57))
    x = np.arange(len(names))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.scatter(x-0.09, n_vals, marker="o", s=72, color=cycle[0],
               label="5-session iTransformer", zorder=3)
    ax.scatter(x+0.09, r_vals, marker="s", s=67, color=cycle[1],
               label="5-session Random Forest", zorder=3)
    ax.axhline(50, ls="--", lw=1.1, color="0.42")
    ax.set_xticks(x, [f"{p.year} Q{p.quarter}" + ("\nno observations" if str(p) in missing else "") for p in periods])
    ax.tick_params(axis="x", pad=6)
    ax.set_xlim(-0.4, len(names)-0.6)
    observed = pd.concat([n_vals, r_vals]).dropna().to_numpy(dtype=float)
    ax.set_ylim(max(0, float(observed.min())-5), min(100, float(observed.max())+5))
    ax.set_title("Five-session performance by observed calendar quarter", pad=15)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.grid(axis="y", alpha=0.15)
    ax.legend(loc="upper right", frameon=False)
    footnote(fig, "Quarterly points deliberately unconnected; Q2 2025 and July 2025 have no evaluated endpoints.", font, y=0.018, width=110)
    finalize_layout(fig, bottom=0.17, top=0.92)
    return fig, {"unsampled_calendar_quarters": missing}


def fig05(data: dict, width: float, font: float):
    stats = data["bootstrap_fdr.csv"]
    primary = stats.loc[stats.block_length == 20].copy()
    ordered = []
    for h in HORIZONS:
        sub = primary.set_index("model_id").loc[list(MODEL_ORDER)]
        ordered.append(sub.loc[sub.horizon_days == h])
    fig, axes = plt.subplots(1, 3, figsize=(width * 1.5, width * 0.86), sharex=True, sharey=True)
    smallest = float(primary.ci_lower_percentile_95.min())*100
    largest = float(primary.ci_upper_percentile_95.max())*100
    width_delta = max(1, (largest-smallest)*0.04)
    for ax, h, sub in zip(axes, HORIZONS, ordered):
        y = np.arange(len(MODEL_ORDER))[::-1]
        center = sub.observed_BA.to_numpy(dtype=float)*100
        lo = sub.ci_lower_percentile_95.to_numpy(dtype=float)*100
        hi = sub.ci_upper_percentile_95.to_numpy(dtype=float)*100
        require(np.all(lo <= center) and np.all(center <= hi),
                f"Bootstrap CI inconsistent with BA, horizon {h}")
        ax.errorbar(center, y, xerr=np.array([center-lo, hi-center]),
                    fmt="o", markersize=4, lw=1.25, capsize=2.4)
        ax.axvline(50, ls="--", lw=1, color="0.4")
        ax.set_title(f"{h}-session")
        ax.set_yticks(y, [MODEL_NAMES[m] for m in MODEL_ORDER])
        ax.set_xlim(smallest-width_delta, largest+width_delta)
        ax.set_xlabel("Balanced accuracy (%)")
        ax.grid(axis="x", alpha=0.15)
    num_disc = int((primary.q_BH_primary_only < 0.05).sum())
    fig.suptitle("Dependence-aware uncertainty: all 24 frozen neural hypotheses")
    footnote(fig, f"Segment-stratified 20-session moving-block bootstrap, 2,000 replicates. BH-FDR discoveries: {num_disc}/24.", font, y=0.018, width=145)
    finalize_layout(fig, bottom=0.12, top=0.93, left=0.06, right=0.99)
    return fig, {"primary_BH_discoveries_at_0_05": num_disc}


def fig06(data: dict, width: float, font: float):
    frame = data["classification_metrics.csv"]
    fig, axes = plt.subplots(1, 3, figsize=(width*1.70, width*.62), sharex=True, sharey=True)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    label_offsets = {
        1: {
            "DLINEAR": (4, 4), "NLINEAR": (4, -2), "PATCHTST": (4, 4),
            "TIMESNET": (4, -2), "ITRANSFORMER": (4, 4), "FREDFORMER": (4, 4),
            "FRNET": (4, 4), "MLF": (4, 4),
        },
        3: {
            "DLINEAR": (4, 4), "NLINEAR": (4, 1), "PATCHTST": (4, -6),
            "TIMESNET": (4, -1), "ITRANSFORMER": (4, -1), "FREDFORMER": (4, 7),
            "FRNET": (4, -2), "MLF": (4, 4),
        },
        5: {
            "DLINEAR": (4, 4), "NLINEAR": (4, 4), "PATCHTST": (4, 4),
            "TIMESNET": (4, 4), "ITRANSFORMER": (4, 4), "FREDFORMER": (4, 4),
            "FRNET": (4, 4), "MLF": (4, -7),
        },
    }
    for ax, h in zip(axes, HORIZONS):
        sub = frame.loc[frame.horizon_days == h].set_index("model_id").loc[list(MODEL_ORDER)]
        x = sub.observed_positive_rate.to_numpy(dtype=float)*100
        y = sub.predicted_positive_rate.to_numpy(dtype=float)*100
        for i, (model, xi, yi) in enumerate(zip(MODEL_ORDER, x, y)):
            ax.scatter(xi, yi, s=34, color=cycle[i % len(cycle)], zorder=3)
            dx, dy = label_offsets.get(h, {}).get(model, (4, 4))
            ax.annotate(
                MODEL_NAMES[model],
                (xi, yi),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=font-3.0,
            )
        ax.plot([0,100], [0,100], linestyle="--", linewidth=.9, color="0.52")
        ax.set_title(f"{h}-session")
        ax.set_xlabel("Realized upward outcomes (%)")
        ax.set_xlim(30, 70)
        ax.set_ylim(-6, 107)
        ax.grid(alpha=0.10)
    axes[0].set_ylabel("Model-predicted upward outcomes (%)")
    fig.suptitle("Directional prediction imbalance in the frozen five-seed ensembles")
    footnote(fig, "Each point is one complete held-out neural model--horizon result (n=5,300). Dashed line: predicted = observed rate.", font, y=0.018, width=145)
    finalize_layout(fig, bottom=0.15, top=0.92, left=0.07, right=0.99)
    return fig, {}


def fig07(data: dict, width: float, font: float):
    winners = best_neural(data)
    panel_names = []
    series = []
    nref = None
    for h in HORIZONS:
        for model, filename in ((winners[h], "supplementary_stock_level.csv"),
                                ("RANDOM_FOREST", "ml_devonly_stock_level.csv")):
            frame = data[filename]
            sub = frame.loc[(frame.model_id == model) & (frame.horizon_days == h)].copy()
            equal_pairs(sub, ["group"], 20, f"Stock subgroup: {model}, {h}d")
            require((sub.n == 265).all(), "Stock has unexpected sample count")
            if nref is None:
                nref = tuple(sorted(sub.group))
            require(tuple(sorted(sub.group)) == nref,
                    "Stock inventory inconsistent across models")
            panel_names.append(f"{h}d\n{MODEL_NAMES[model]}")
            series.append(sub.set_index("group").balanced_accuracy.reindex(nref)*100)
    mat = np.stack([s.to_numpy(dtype=float) for s in series], axis=1)
    require(np.isfinite(mat).all(), "Non-finite stock-level BA")
    maxdev = max(float(np.abs(mat-50).max()), .5)
    fig, ax = plt.subplots(figsize=(width*1.23, width*.98))
    norm = TwoSlopeNorm(vmin=50-maxdev, vcenter=50, vmax=50+maxdev)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", norm=norm)
    ax.set_xticks(range(6), panel_names)
    ax.tick_params(axis="x", pad=7)
    ax.set_yticks(range(20), [str(s).replace(".NS", "") for s in nref])
    ax.set_title("Stock-level heterogeneity: selected neural leaders versus Random Forest", pad=15)
    cmap = plt.get_cmap("RdBu")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            rgba = cmap(norm(mat[i, j]))
            luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            tcolor = "black" if luminance > 0.55 else "white"
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=font-3.5, color=tcolor)
    cb = fig.colorbar(im, ax=ax, shrink=0.83)
    cb.set_label("Balanced accuracy (%)")
    footnote(fig, "Each stock has 265 endpoints. Neural leaders selected post-test; exploratory descriptive heatmap only.", font, y=0.018, width=118)
    finalize_layout(fig, bottom=0.12, top=0.93, left=0.10, right=0.99)
    return fig, {"best_neural_per_horizon": winners}


FUNCTIONS = dict(zip(FIGURES, (fig01, fig02, fig03, fig04, fig05, fig06, fig07)))


def output_safe(out: Path, exp: Path) -> None:
    exp = exp.resolve()
    out = out.resolve()
    require(out != exp and exp not in out.parents,
            "Output must be outside the immutable experiment directory")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    default_exp = base.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026"
    parser = argparse.ArgumentParser(
        description="Authentic F29R7 result figures (CSV-only; cryptographic checks; no invented data)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment-root", type=Path, default=default_exp,
                        help="Full F29R7 experiment root; must contain the signed publication receipt")
    parser.add_argument("--output-dir", type=Path,
                        default=base / "Revision_F29R7_Assets_REGENERATED",
                        help="Figures and provenance go here; must be OUTSIDE the experiment root")
    parser.add_argument("--plots", nargs="+", choices=("main", "all") + FIGURES,
                        default=["main"],
                        help="main=four Revision.tex figures; all=main and three additional diagnostics")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"),
                        default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution; PDF remains vector")
    parser.add_argument("--width", type=float, default=8.6, help="Base figure width in inches")
    parser.add_argument("--font-size", type=float, default=10.5)
    parser.add_argument("--minimal-grid", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly permit replacement of existing generated plot files")
    parser.add_argument("--verify-only", action="store_true",
                        help="Authenticate result artifacts and validate CSVs, but do not write files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.dpi >= 150 and args.dpi <= 1200, "--dpi must be in [150,1200]")
    require(5 <= args.width <= 18 and 8 <= args.font_size <= 16,
            "Unexpected width or font-size")
    exp = args.experiment_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    output_safe(out, exp)
    choose = list(FIGURES[:4]) if "main" in args.plots else (
             list(FIGURES) if "all" in args.plots else list(dict.fromkeys(args.plots)))
    print("=" * 88)
    print("F29R7 VERIFIED SCIENTIFIC FIGURE PRODUCTION — READ-ONLY INPUTS")
    print("Experiment:", exp)
    print("Output:", out)
    print("Selected figures:", ", ".join(choose))
    print("=" * 88)
    data, provenance = read_verified_results(exp)
    print("PASS: SHA-256 matched F29R7 public receipt")
    print("PASS: SHA-256 matched F29R6 committed manifest")
    print("PASS: all 23 committed artifacts verified against manifest")
    print("PASS: metric, model, horizon and subgroup schemas verified")
    configure_style(args.font_size, args.minimal_grid)
    if args.verify_only:
        print("VERIFY-ONLY PASS — no plots or files written")
        return
    planned = [out / f"{name}.{fmt}" for name in choose for fmt in args.formats]
    planned += [out / "F29R7_FIGURE_PROVENANCE.json"]
    if not args.overwrite:
        require(not any(p.exists() or p.is_symlink() for p in planned),
                "Output files already exist. Select an empty --output-dir or use --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str,str] = {}
    figure_context: dict[str, dict] = {}
    for name in choose:
        fig, notes = FUNCTIONS[name](data, args.width, args.font_size)
        figure_context[name] = notes
        for ext in args.formats:
            target = out / f"{name}.{ext}"
            save_kwargs = {"format": ext, "facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.10}
            if ext == "png":
                save_kwargs["dpi"] = args.dpi
            try:
                fig.savefig(target, **save_kwargs)
            except Exception:
                plt.close(fig)
                raise
            output_hashes[target.name] = sha256_file(target)
            print(f"SAVED: {target} (SHA-256 {output_hashes[target.name]})")
        plt.close(fig)
    report = {
        "schema": "F29R7_VERIFIED_FIGURE_PROVENANCE_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": provenance,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "plot_names": choose,
        "formats": args.formats,
        "png_dpi": args.dpi,
        "width_in": args.width,
        "font_size_pt": args.font_size,
        "figure_annotations": figure_context,
        "output_sha256": output_hashes,
        "data_policy": "All empirical values derived from 23 authenticated F29R7 artifacts; no test reopen/retraining/bootstrapping",
    }
    with (out / "F29R7_FIGURE_PROVENANCE.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"FIGURE PRODUCTION PASS: {len(output_hashes)} figure files, "+
          "provenance JSON; no experiment inputs modified")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, AssertionError, OSError) as exc:
        print(f"F29R7 PLOTTING FAIL-CLOSED: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        sys.exit(1)
