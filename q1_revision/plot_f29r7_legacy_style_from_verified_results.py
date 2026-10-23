#!/usr/bin/env python3
"""Exploratory, source-authenticated figure styles for the F29R7 revised NSE study.

This extends (does not replace) plot_f29r7_verified_scientific_v4.py. It
imports that program's SHA-256-verifying reader. It never opens sealed raw-test
files, refits a model, generates predictions, tunes a threshold, or recomputes
bootstrap intervals. All plotted statistics come from existing authenticated
F29R7 CSVs. The seven scientific layouts reinterpret VISUAL STYLES used in an
older manuscript; the old experiment's numbers and study arms are not reused.

Run from q1_revision after placing this and the original v4 plotting source
beside each other:
    python plot_f29r7_legacy_style_from_verified_results_FIG14_Q1_CORRECTED.py \
        --plots figure14 --formats pdf png --dpi 600 \
        --font-size 12 --output-dir F29R7_Figure14_HORIZONTAL_Q1_FINAL

    ``figure14`` is a publication-oriented alias for the updated legacy
    ``sup04_disjoint_segment_stability`` plot, and writes a Figure14-named file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    import plot_f29r7_verified_scientific_v4 as verified
except ImportError as exc:
    raise SystemExit(
        "FAIL CLOSED: put plot_f29r7_verified_scientific_v4.py in the same folder "
        "as this extension; the original receipt-authentication logic is required."
    ) from exc

HORIZONS = (1, 3, 5)
NEURAL_ORDER = tuple(verified.MODEL_ORDER)
ML_ORDER = ("LOGREG", "RANDOM_FOREST", "XGBOOST")
ALL_MODELS = NEURAL_ORDER + ML_ORDER
LABELS = dict(verified.MODEL_NAMES)
LABELS.update({"LOGREG": "Logistic Reg.", "XGBOOST": "XGBoost"})
FOCUS = ("DLINEAR", "PATCHTST", "ITRANSFORMER", "RANDOM_FOREST")
PALETTE = {
    "DLINEAR": "#245AA8", "PATCHTST": "#248780", "ITRANSFORMER": "#B04B75",
    "RANDOM_FOREST": "#D47F30", "OTHER_NEURAL": "#4774B2", "OTHER_ML": "#D47F30",
}
FIGURES = (
    "sup01_all_neural_ml_BA_bars",
    "sup02_all_neural_ml_ROCAUC_bars",
    "sup03_regimes_by_horizon",
    "sup04_disjoint_segment_stability",
    "sup05_stock_level_BA_boxplots",
    "sup06_heldout_up_class_balance",
    "sup07_heldout_regime_composition",
)


def check(ok: bool, msg: str):
    if not ok:
        raise RuntimeError(msg)


def stat(data: dict[str, pd.DataFrame], model: str, horizon: int, field: str) -> float:
    file = ("classification_metrics.csv" if model in NEURAL_ORDER
            else "ml_devonly_classification_metrics.csv")
    f = data[file]
    s = f.loc[f.model_id.eq(model) & f.horizon_days.eq(horizon)]
    check(len(s) == 1, f"Missing/duplicate frozen metric: {model} {horizon}d")
    v = float(s.iloc[0][field])
    check(np.isfinite(v), f"Non-finite metric: {model} {horizon}d {field}")
    return v


def group_metric(data: dict, suffix: str, model: str, horizon: int) -> pd.DataFrame:
    fname = ("supplementary_" if model in NEURAL_ORDER else "ml_devonly_") + suffix + ".csv"
    f = data[fname]
    s = f.loc[f.model_id.eq(model) & f.horizon_days.eq(horizon)].copy()
    check(not s.empty and not s.group.duplicated().any(), f"Invalid subgroup {fname} {model} {horizon}d")
    return s


def clean_style(font: float):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": font,
        "axes.labelsize": font + 0.2, "axes.titlesize": font + 1,
        "axes.titleweight": "semibold", "xtick.labelsize": font - 1,
        "ytick.labelsize": font - 1, "legend.fontsize": font - 1.1,
        "figure.constrained_layout.use": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#A9AFB9",
        "grid.alpha": 0.20, "axes.axisbelow": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_bars(data: dict, field: str, name: str, font: float):
    data_matrix = np.array([[stat(data, m, h, field) * 100 for m in ALL_MODELS]
                            for h in HORIZONS])
    check(np.isfinite(data_matrix).all(), f"Invalid {field} plot matrix")
    # Descriptive observed scores only; the ROC-AUC panel does not invent CIs.
    lo = max(0, float(data_matrix.min()) - 2.0)
    hi = min(100, float(data_matrix.max()) + 2.4)
    lo = min(lo, 49.1)  # Always include the 50% reference.
    hi = max(hi, 50.9)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 8.0), sharex=True, sharey=True)
    y = np.arange(len(ALL_MODELS))
    colors = [PALETTE["OTHER_NEURAL"] if m in NEURAL_ORDER else PALETTE["OTHER_ML"]
              for m in ALL_MODELS]
    for ax, h, vals in zip(axes, HORIZONS, data_matrix):
        ax.barh(y, vals - lo, left=lo, height=0.7, color=colors, alpha=0.92)
        ax.set_yticks(y, [LABELS[m] for m in ALL_MODELS])
        ax.invert_yaxis()
        ax.set_xlim(lo, hi)
        ax.axvline(50, ls="--", color="#3D4148", lw=1.15)
        ax.set_title(f"{h}-session horizon", pad=10)
        ax.set_xlabel(name + " (%)")
        ax.grid(axis="x")
        for j, v in enumerate(vals):
            ax.text(v + 0.09, j, f"{v:.2f}", ha="left", va="center", fontsize=font-2.1)
    fig.suptitle(f"Held-out {name.lower()}: all 11 model families", fontsize=font+3, y=0.96)
    fig.legend([
        Line2D([],[],marker="s",markersize=9,linestyle="none",color=PALETTE["OTHER_NEURAL"]),
        Line2D([],[],marker="s",markersize=9,linestyle="none",color=PALETTE["OTHER_ML"]),
    ], ["8 frozen neural ensembles", "3 development-only ML comparators"],
        ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.53, 0.925))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.105, top=0.83, wspace=0.15)
    return fig, {"field": field, "num_neural": 8, "num_exploratory_ML": 3,
                 "CIs_shown": False, "values_loaded_from_authenticated_CSVs": True}


def plot_ba(data: dict, font: float):
    return panel_bars(data, "balanced_accuracy", "Balanced accuracy", font)


def plot_auc(data: dict, font: float):
    return panel_bars(data, "roc_auc", "ROC-AUC", font)


def plot_regimes(data: dict, font: float):
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.9), sharex=True, sharey=True)
    regime_order = ["Low", "Normal", "High"]
    for ax, horizon in zip(axes, HORIZONS):
        for model in FOCUS:
            subset = group_metric(data, "regime_level", model, horizon)
            subset["regime"] = subset.group.map(verified.map_regime)
            check(len(subset) == 3 and set(subset.regime) == set(regime_order),
                  f"Regime membership mismatch: {model} {horizon}d")
            vals = subset.set_index("regime").loc[regime_order].balanced_accuracy.to_numpy(dtype=float)*100
            ax.plot(regime_order, vals, marker="o", ms=5.6, lw=1.7,
                    color=PALETTE[model], label=LABELS[model])
        ax.axhline(50, color="#4D525B", linestyle="--", lw=1)
        ax.set_title(f"{horizon}-session horizon")
        ax.grid(axis="y")
        ax.set_ylim(37, 62)
    axes[0].set_ylabel("Balanced accuracy (%)")
    fig.suptitle("Frozen volatility regimes: descriptive model contrast", y=0.98, fontsize=font+3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=4,
               bbox_to_anchor=(0.50, 0.91))
    fig.subplots_adjust(left=.065,right=.98,bottom=.15,top=.77,wspace=.12)
    return fig, {"interpretation": "exploratory, no preregistered interaction",
                 "regime_order": regime_order, "model_selection": "fixed displayed subset, not selected within regimes"}


def segment_index(value: object) -> int:
    text = str(value).strip().lower()
    remap = {"0": 0, "1": 1, "0.0": 0, "1.0": 1,
             "segment_0": 0, "segment_1": 1, "segment 0":0,"segment 1":1}
    check(text in remap, f"Unrecognized canonical continuity segment '{value}'")
    return remap[text]


def plot_segments(data: dict, font: float):
    """FIGURE 14: disjoint-cohort BA, 1 row x 3 horizon panels.

    Plot only previously authenticated segment CSV summary values. Each of the
    four displayed fixed models contributes exactly two subgroup points per
    horizon. Temporal segments remain UNCONNECTED (147-calendar-day separation).
    No group-wise model selection, re-estimation, confidence intervals, or
    significance tests are introduced by this descriptive visualization.
    """
    segments = {
        0: {"sessions": 48, "n": 960, "marker": "o", "color": "#246d9a"},
        1: {"sessions": 217, "n": 4340, "marker": "s", "color": "#c67637"},
    }
    rates = {}
    all_percentages = [50.0]
    provenance_rows = {}

    # Fail closed before plotting if any required authenticated subgroup is
    # absent, duplicated, non-finite, outside [0, 1], or has wrong cohort n.
    for horizon in HORIZONS:
        rates[horizon] = {}
        provenance_rows[str(horizon)] = {}
        for model in FOCUS:
            sub = group_metric(data, "segment_level", model, horizon)
            sub["seg"] = sub.group.map(segment_index)
            check(len(sub) == 2 and set(sub.seg) == {0, 1},
                  f"Missing/duplicated segment {model}, {horizon} session")
            sub = sub.set_index("seg").loc[[0, 1]]
            n = pd.to_numeric(sub.n, errors="raise").to_numpy(dtype=float)
            scores = pd.to_numeric(sub.balanced_accuracy, errors="raise").to_numpy(dtype=float)
            check(np.isfinite(n).all() and np.isfinite(scores).all(),
                  f"Non-finite segment values {model}, {horizon} session")
            check(np.allclose(n, [960, 4340], rtol=0, atol=0),
                  f"Wrong frozen segment sizes for {model}, {horizon} session: {n}")
            check(np.all((scores >= 0) & (scores <= 1)),
                  f"Invalid balanced accuracy for {model}, {horizon} session")
            vals = scores * 100.0
            rates[horizon][model] = vals
            all_percentages.extend(vals.tolist())
            provenance_rows[str(horizon)][model] = {
                "segment_0_ba": float(scores[0]),
                "segment_1_ba": float(scores[1]),
                "segment_0_endpoints": 960,
                "segment_1_endpoints": 4340,
            }

    # Identical x-axis limits protect comparison between the three horizons.
    low = max(0.0, 2 * np.floor((min(all_percentages) - 3.0) / 2))
    high = min(100.0, 2 * np.ceil((max(all_percentages) + 5.0) / 2))
    step = 2 if high - low <= 24 else (5 if high - low <= 45 else 10)
    ticks = np.arange(np.ceil(low / step) * step, high + 0.01, step)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.35),
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=.137, right=.985, bottom=.235,
                        top=.755, wspace=.15)
    row = np.arange(len(FOCUS))
    dy = {0: -.175, 1: .175}

    for panel_idx, (ax, horizon) in enumerate(zip(axes, HORIZONS)):
        # RF is an exploratory baseline using a different input interface.
        ax.axhspan(2.55, 3.45, facecolor="#f0f3f5", zorder=0)
        ax.set_axisbelow(True)
        for yy in row:
            ax.axhline(yy, color="#e3e8ec", lw=.65, zorder=.5)

        for model_idx, model in enumerate(FOCUS):
            two_vals = rates[horizon][model]
            for segment_id, val in enumerate(two_vals):
                meta = segments[segment_id]
                y = model_idx + dy[segment_id]
                ax.scatter(val, y, s=53, marker=meta["marker"],
                           color=meta["color"], edgecolor="white",
                           linewidth=.65, zorder=4)
                # Exact descriptive percentages, from the committed CSV only.
                ax.annotate(f"{val:.2f}", (val, y), xytext=(5, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=font-1.9, color=meta["color"], zorder=5)

        ax.axvline(50, color="#3d4248", lw=1.15,
                   linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlim(low, high)
        ax.set_ylim(3.55, -.57)
        ax.set_xticks(ticks)
        ax.set_yticks(row)
        ax.set_yticklabels([LABELS[m] for m in FOCUS])
        ax.tick_params(axis="y", length=0, pad=6, labelsize=font-.2)
        ax.tick_params(axis="x", length=0, labelsize=font-.8)
        if panel_idx:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.grid(axis="x", color="#c9d0d7", alpha=.55, lw=.65)
        ax.set_title(f"{chr(65+panel_idx)}  |  {horizon}-session horizon",
                     loc="left", fontsize=font+.5, pad=11)
        ax.set_xlabel("Balanced accuracy (%)", labelpad=9,
                      fontsize=font-.1)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#aeb6be")

    legend = [
        Line2D([], [], ls="none", marker="o", color=segments[0]["color"],
               markersize=7, label="Segment 0 (48 sessions; n = 960)"),
        Line2D([], [], ls="none", marker="s", color=segments[1]["color"],
               markersize=7, label="Segment 1 (217 sessions; n = 4,340)"),
        Line2D([], [], color="#3d4248", linestyle=(0, (4, 3)), lw=1.15,
               label="BA = 50%"),
    ]
    fig.suptitle("Held-out balanced accuracy across two discontinuous evaluation periods",
                 fontsize=font+2.1, weight="semibold", y=.975)
    fig.legend(handles=legend, ncol=3, frameon=False, loc="upper center",
               bbox_to_anchor=(.51, .891), fontsize=font-.9,
               columnspacing=1.25, handletextpad=.4)
    fig.text(.5, .112,
             "Segment 0: 1 Jan–7 Mar 2025. Segment 1: 1 Aug 2025–22 Jun 2026.",
             ha="center", va="center", fontsize=font-.8, color="#4d5660")
    fig.text(.5, .071,
             "147-calendar-day separation; no connecting lines. "
             "Random Forest is exploratory; no subgroup significance tests.",
             ha="center", va="center", fontsize=font-1.05, color="#4d5660")

    return fig, {
        "figure": "Figure 14: disjoint-continuity-segment descriptive BA",
        "segment_sessions": [48, 217],
        "segment_endpoints": [960, 4340],
        "calendar_day_date_separation": 147,
        "connecting_lines": False,
        "horizon_order": list(HORIZONS),
        "model_order": list(FOCUS),
        "score_units": "percent on plot; proportion in provenance rows",
        "plot_values": provenance_rows,
        "exploratory": True,
        "no_new_inference": True,
    }


def plot_stock_boxplots(data: dict, font: float):
    fig, axes = plt.subplots(1,3,figsize=(14.0,5.6),sharey=True)
    groups = []
    for ax,h in zip(axes,HORIZONS):
        stocks = []
        for model in FOCUS:
            subset=group_metric(data,"stock_level",model,h)
            check(len(subset)==20 and (subset.n == 265).all(),
                  f"Invalid 20-stock membership {model} {h}d")
            group = set(subset.group.astype(str))
            if not groups:
                groups.append(group)
            check(group == groups[0],"Different stock universes across frozen evaluation")
            stocks.append(subset.balanced_accuracy.to_numpy(dtype=float)*100)
        bp=ax.boxplot(stocks,patch_artist=True,widths=.54,showfliers=True,
                      medianprops={"color":"#1D2430","linewidth":1.4},
                      flierprops={"marker":"o","markersize":3,"markerfacecolor":"white"})
        for item,m in zip(bp['boxes'],FOCUS):
            item.set_facecolor(PALETTE[m]);item.set_alpha(.35)
            item.set_edgecolor(PALETTE[m]);item.set_linewidth(1.0)
        ax.set_xticks(np.arange(1,5),["DLinear","PatchTST","iTransf.","RF"])
        ax.tick_params(axis="x",pad=6)
        ax.axhline(50,color="#444950",ls="--",lw=1)
        ax.set_ylim(33,72)
        ax.set_title(f"{h}-session horizon")
        ax.grid(axis="y")
    axes[0].set_ylabel("Stock-level balanced accuracy (%)")
    fig.suptitle("Cross-stock distributions: 20 NSE equities per model",y=.96,fontsize=font+3)
    fig.subplots_adjust(left=.075,right=.985,bottom=.17,top=.85,wspace=.14)
    return fig, {"stocks_per_model":20,"outlier_definition":"Tukey 1.5 IQR",
                 "interpretation":"descriptive cross-section; no asset selection"}


def plot_up_balance(data: dict, font: float):
    frame=data["classification_metrics.csv"]
    sizes=[]
    shares=[]
    for h in HORIZONS:
        sub=frame.loc[frame.horizon_days.eq(h)]
        check(len(sub)==8 and sub.n.eq(5300).all(),"Unexpected horizon sample size")
        shares_h = (sub.tp + sub.fn)/sub.n
        check(float(shares_h.max() - shares_h.min()) < 1e-12,
              "Different true-label prevalence across common neural model endpoints")
        shares.append(float(shares_h.iloc[0])*100)
        sizes.append(int(sub.n.iloc[0]))
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    x=np.arange(3)
    ax.bar(x,shares,color="#3273B1",width=.54,label="Observed upward")
    ax.bar(x,100-np.array(shares),bottom=shares,color="#B2B7BD",width=.54,label="Observed non-upward")
    for i,p in enumerate(shares):
        ax.text(i,p/2,f"{p:.1f}%",ha="center",va="center",fontweight="bold",color="white")
        ax.text(i,p+(100-p)/2,f"{100-p:.1f}%",ha="center",va="center",color="#22262A")
    ax.set_ylim(0,108)
    ax.set_xticks(x,[f"{h}-session\n(n={n:,})" for h,n in zip(HORIZONS,sizes)])
    ax.set_ylabel("Observed class distribution (%)")
    ax.set_title("Held-out class composition by forecast horizon",pad=14)
    ax.legend(loc="upper right",frameon=False)
    ax.grid(axis="y")
    fig.subplots_adjust(left=.13,right=.97,bottom=.17,top=.90)
    return fig, {"ONLY_TEST_COHORT": True,"training_validation_composition":"not available in authenticated F29R7 result CSVs"}


def plot_regime_composition(data: dict, font: float):
    frame=group_metric(data,"regime_level","DLINEAR",1)
    frame["regime"]=frame.group.map(verified.map_regime)
    check(set(frame.regime)=={"Low","Normal","High"} and len(frame)==3,
          "Frozen regime ID/name mismatch")
    groups=frame.set_index("regime").loc[["Low","Normal","High"]]
    counts=groups.n.to_numpy(dtype=int)
    check(counts.sum()==5300 and np.array_equal(counts,[2524,1577,1199]),
          "Frozen held-out regime membership differs from amended Phase A")
    fig,ax=plt.subplots(figsize=(9.3,4.8))
    offset=0
    cols=["#4D91BA","#A7AEB8","#D88B50"]
    for i,(name,count,c) in enumerate(zip(["Low","Normal","High"],counts,cols)):
        pct=100*count/5300
        ax.barh([0],[pct],left=offset,height=.5,color=c,label=f"{name} ({count:,})")
        ax.text(offset+pct/2,0,f"{name}\n{pct:.1f}%",ha="center",va="center",
                fontsize=font,weight="semibold",color="#142235")
        offset+=pct
    ax.set_xlim(0,100)
    ax.set_yticks([])
    ax.set_xlabel("Share of frozen held-out endpoints (%)")
    ax.set_title("Actual held-out volatility-regime composition",pad=18)
    ax.legend(frameon=False,loc="lower center",bbox_to_anchor=(.5,-.32),ncol=3)
    fig.subplots_adjust(left=.09,right=.97,bottom=.31,top=.89)
    return fig, {"cohort": "5,300 amended held-out endpoints only",
                 "NOT_training_validation_distribution":True,"regime_counts":counts.tolist()}


FUNCTIONS={
    FIGURES[0]:plot_ba,FIGURES[1]:plot_auc,FIGURES[2]:plot_regimes,
    FIGURES[3]:plot_segments,FIGURES[4]:plot_stock_boxplots,
    FIGURES[5]:plot_up_balance,FIGURES[6]:plot_regime_composition,
    "figure14":plot_segments,
}


def sha(path:Path):
    return verified.sha256_file(path)


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description="Authenticated F29R7 exploratory old-style layouts; no old-study data")
    parser.add_argument("--experiment-root", type=Path,
        default=root.parent / "stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026")
    parser.add_argument("--output-dir", type=Path,
        default=root/"F29R7_Style_Recreations_Supplementary")
    parser.add_argument("--plots", nargs="+",default=["all"],choices=("all", "figure14")+FIGURES,
                        help="Specify figure14 for the revised manuscript Figure 14, or a legacy supXX name")
    parser.add_argument("--formats",nargs="+",default=["png","pdf"],choices=("png","pdf"))
    parser.add_argument("--dpi",type=int,default=600)
    parser.add_argument("--font-size",type=float,default=10)
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    check(150<=args.dpi<=1200,"DPI outside 150-1200")
    check(8<=args.font_size<=14,"Unsupported font size")
    experiment=args.experiment_root.expanduser().resolve()
    output=args.output_dir.expanduser().resolve()
    verified.output_safe(output,experiment)
    chosen=list(FIGURES) if "all" in args.plots else list(dict.fromkeys(args.plots))
    print("F29R7 — AUTHENTICATED REVISION FIGURES (SUPPLEMENTARY)")
    print("Experiment:",experiment)
    print("Output:",output)
    data,verified_provenance=verified.read_verified_results(experiment)
    # The v4 reader authenticates all 23 files but imports only the 10 CSVs
    # needed for its original figures. Import the two already-authenticated
    # segment summaries without recalculating anything.
    if "sup04_disjoint_segment_stability" in chosen or "figure14" in chosen:
        for filename in ("supplementary_segment_level.csv", "ml_devonly_segment_level.csv"):
            manifest_hash = verified_provenance["verified_artifact_sha256"].get(filename)
            check(bool(manifest_hash),f"Segment summary not in signed artifact inventory: {filename}")
            segment_path=Path(verified_provenance["evaluation_dir"])/filename
            check(sha(segment_path)==manifest_hash,f"Authenticated segment source changed: {filename}")
            frame=pd.read_csv(segment_path)
            check({"model_id","horizon_days","group","n","balanced_accuracy"}.issubset(frame.columns),
                  f"Unexpected segment-level schema in {filename}")
            data[filename]=frame
    print("PASS: F29R7 signed receipt, committed evaluation manifest, 23 SHA-256 verified results")
    clean_style(args.font_size)
    if args.verify_only:
        print("VERIFY ONLY PASS; no figures or files written")
        return
    output_names={"figure14": "Figure14_Disjoint_Segment_Stability_HORIZONTAL"}
    planned=[output/f"{output_names.get(n,n)}.{fmt}" for n in chosen for fmt in args.formats]
    planned.append(output/"F29R7_RECREATION_PROVENANCE.json")
    if not args.overwrite:
        check(all(not p.exists() and not p.is_symlink() for p in planned),
              "Existing outputs; use a fresh output directory or explicit --overwrite")
    output.mkdir(parents=True,exist_ok=True)
    files={};annotations={}
    for plot in chosen:
        fig,notes=FUNCTIONS[plot](data,args.font_size)
        try:
            for ext in args.formats:
                target=output/f"{output_names.get(plot,plot)}.{ext}"
                fig.savefig(target,format=ext,facecolor="white",bbox_inches="tight",pad_inches=.12,
                            **({"dpi":args.dpi} if ext=="png" else {}))
                files[target.name]=sha(target)
                print("SAVED",target.name,"SHA",files[target.name])
        finally:
            plt.close(fig)
        annotations[plot]=notes
    p={
        "schema":"F29R7_POSTCOMMIT_AUTHENTICATED_STYLE_RECREATION_V1",
        "utc":datetime.now(timezone.utc).isoformat(),
        "receipt_sha256":verified.RECEIPT_SHA256,
        "evaluation_manifest_sha256":verified.EVALUATION_SHA256,
        "authenticated_input_artifacts":verified_provenance,
        "this_script_sha256":sha(Path(__file__).resolve()),
        "validation_reader_sha256":sha(Path(verified.__file__).resolve()),
        "matplotlib_version":matplotlib.__version__,
        "pandas_version":pd.__version__,"numpy_version":np.__version__,
        "dpi_png":args.dpi,"figures":chosen,"plot_annotations":annotations,"output_sha256":files,
        "figure14_layout":"three horizontal horizon panels; two disjoint groups; no connecting lines",
        "explicitly_excluded":"historical 2280-row old study, outdated arms, economic backtests, train/val compositions",
        "access":"postcommit CSVs only; zero raw test, training, inference, bootstrap recomputation"
    }
    (output/"F29R7_RECREATION_PROVENANCE.json").write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
    print(f"PASS: {len(chosen)} figure types; {len(files)} PNG/PDF outputs, authenticated provenance")

if __name__=="__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, AssertionError, OSError) as exc:
        print(f"FAIL CLOSED: {type(exc).__name__}: {exc}",file=sys.stderr)
        sys.exit(1)
