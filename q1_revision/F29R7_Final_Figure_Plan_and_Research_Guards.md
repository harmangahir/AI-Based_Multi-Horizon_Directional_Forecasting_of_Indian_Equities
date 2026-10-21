# Final scientific figure plan — authenticated F29R7 revised Indian-equities study

## Scientific boundary

Study: 20 NSE stocks, horizons of 1/3/5 observed sessions, 5,300 common held-out stock-session endpoints over **265 distinct sessions in two separate continuity intervals**, eight five-seed neural architectures and nine development-trained horizon-specific ML classifiers. The **24 primary neural tests** had no BH-FDR discovery at 0.05. The 2025Q2 and July 2025 observations are absent from the amended test cohort. The committed result set comprises 23 SHA-256-authenticated output artifacts. All new plots are exploratory and **must not** overwrite the confirmatory inference registry.

The plotting extension imports the *existing v4 F29R7 authenticator* and refuses to plot if the publication receipt, evaluation manifest, or any one of the 23 committed artifacts fails its original cryptographic verification. No raw sealed test reads, model training, neural inference, threshold retuning, or bootstrap recomputation occur.

## A. Recommended main-manuscript display

Based on the current 16-page revised PDF, the manuscript currently has **seven figures** (visible on PDF pages 7–12). Keep the informative core and move redundant diagnostics to supplementary material if target journal page constraints permit.

| Main figure | Current revised PDF | Input / status | Text location | Reason |
|---|---|---|---|---|
| Figure 1. Neural 8×3 BA heatmap with cell values | Current Fig. 1, PDF p. 7 | `classification_metrics.csv`; authenticated | §6.1, after Table 4 | Complete primary model reporting |
| Figure 2. Best observed neural vs Random Forest | Current Fig. 2, PDF p. 7 | Neural + ML classification metrics; exploratory comparison | §6.1, after Fig. 1 | A transparent descriptive comparator, **not** superiority |
| Figure 3. All 24 primary dependence-aware BA intervals | Current Fig. 3, PDF p. 8 | Primary L=20 rows of `bootstrap_fdr.csv` | §6.2, after Table 5 | Central statistical evidence, 0/24 BH-FDR |
| Figure 4. Five-day volatility regime contrast | Current Fig. 4, PDF p. 10 | Frozen neural/ML regime CSVs; exploratory | §6.4, after Table 8 | Conditional instability with correct Low/Normal/High names |
| Figure 5. Quarterly stability with absent 2025Q2 | Current Fig. 7, PDF p. 12 | Frozen quarter-level CSVs; exploratory | §6.5, after Table 10 | Shows the 147-day gap and instability |

If the target journal permits seven figures, retaining the current seven in main is also defensible; the table above is an **editorial suggestion**, not a change to the frozen experimental results. The existing PDF's Fig. 5 predicted-class bias and Fig. 6 stock-level heatmap can be supplementary S-figures instead. Renumber only after the final layout is chosen and update all `\\ref` labels consistently.

## B. Newly reproducible visual styles inspired by older manuscript

The new `plot_f29r7_legacy_style_from_verified_results.py` **does not load the old 2,280-row rejected experiment or its numbers**. Filenames are deliberately `sup01_...` etc. These are newly derived descriptive figures, not replicas of an unsupported old hypothesis or model arm.

| New supplement | Visual style / figure | Real F29R7 inputs | Scientific guard | Suggested section |
|---|---|---|---|---|
| S1 | BA horizontal-bar small multiples, **all eight neural and three ML** models, 1/3/5 sessions | `classification_metrics.csv`, `ml_devonly_classification_metrics.csv` | No ML superiority claim; no invented CIs | §6.1–6.3 |
| S2 | ROC-AUC horizontal-bar small multiples, all 11 model families | Same two verified metrics files | **Point estimates only**: no ROC-AUC bootstrap CIs were registered | §6.1–6.3 |
| S3 | Three regime-by-horizon line panels, fixed DLinear/PatchTST/iTransformer/Random Forest display subset | `supplementary_regime_level.csv`, `ml_devonly_regime_level.csv` | Separate panels; exploratory, not regime-conditioned optimization | §6.4 |
| S4 | Two nonadjacent continuity-segment point panels per horizon; **no connecting line across the gap** | `supplementary_segment_level.csv`, `ml_devonly_segment_level.csv` | 960/4,340 stock endpoints; 147-calendar-day separation | §6.5 |
| S5 | Twenty-stock BA boxplots, four displayed models across three horizons | `supplementary_stock_level.csv`, `ml_devonly_stock_level.csv` | Post-hoc stock heterogeneity only; 265 endpoints/stock; no stock selection | §6.5 |
| S6 | Observed up/down label balance for **held-out cohort only**, across horizons | Verified primary confusion counts from `classification_metrics.csv` | Not labelled train/validation/test composition | §5.1 or §6.1 |
| S7 | Frozen held-out Low/Normal/High composition | `supplementary_regime_level.csv`, authenticated Phase A membership totals | Not labelled train/validation/test regime distribution | §5.1, §6.4 |

## C. Older visual styles that **cannot** be relabelled as current-study figures

1. Original training/validation/testing up-class composition **by split**: current authenticated final-evaluation summaries contain held-out labels only. Recreate training and validation only after a separate *read-only* inspection of appropriately frozen development labels and split definitions; do not invent percentages.
2. Full regime-construction time series with train-derived `q33/q67` lines: needs validated historical causal volatility/regime features and threshold manifests. Current F29R7 regime **subgroup scores** cannot substitute for those series.
3. Original four branches ('full-feature classical', 'selected-feature classical', 'simple neural', 'full proposed iTransformer'): not the present eight-neural-plus-three-ML design. The extension uses authentic present models instead.
4. Original ROC-AUC bootstrap intervals: only BA primary bootstrap intervals are present in the known committed `bootstrap_fdr.csv`. The extension plots AUC point estimates and **no fabricated intervals**.
5. Original four chronological performance blocks: the revised protocol records two disjoint continuity segments, not four prospectively defined test blocks. The extension uses the two real segments, **without crossing the gap**.
6. Transaction-cost / signed-return / portfolio-utility graphs: **not evaluated under revised F29R7 protocol**. Do not insert old signed-return plots. Predictive BA does not imply trading profitability.

## D. Precise locations in the attached 16-page revised PDF

| Discussion point | Revised PDF location | New figure references |
|---|---|---|
| Reporting all neural models / comparisons | §6.1–6.3, PDF pp. 6–9 | S1, S2 |
| Continuity amendment and cohort | §5.1, PDF pp. 4–5 | S6, S7 (held-out only) |
| Volatility results and direction bias | §6.4, PDF pp. 9–10 | S3 |
| Distinct continuity intervals | §6.5, PDF pp. 10–11 | S4 |
| Cross-stock dispersion | §6.5, PDF p. 11 | S5 |

These are **supplementary** figures: put them in a separate `F29R7_Style_Supplementary.tex` and cite them from main text only after you have actually generated their verified PDF files.

## E. WSL reproducibility and controls

Place the existing `plot_f29r7_verified_scientific_v4.py` and new extension `.py` files together in `q1_revision`. Run:

```bash
cd ~/projects/AI-Based_Multi-Horizon_Directional_Forecasting_of_Indian_Equities/q1_revision
python plot_f29r7_legacy_style_from_verified_results.py --verify-only
python plot_f29r7_legacy_style_from_verified_results.py \
  --plots all --formats png pdf --dpi 600
```

Plots are written **outside the immutable experiment directory** to `q1_revision/F29R7_Style_Recreations_Supplementary/`, plus `F29R7_RECREATION_PROVENANCE.json` containing hashes of sources and outputs. An incomplete file, inconsistent schema, modified authenticated artifact, or existing figure filename fails closed unless `--overwrite` is explicitly requested. **Do not use `--overwrite` on the first production run.**

The Python source has been syntax-checked and figure-layout functions tested using temporary private *non-research* fixtures. The program **has not been executed against your laptop-only F29R7 CSV files**; production validation occurs on WSL. No fictional experimental figure is supplied.

## F. Manuscript quality checks independent of graphics

The revised 16-page PDF currently includes an incomplete article title ending in “Evaluatio” (p. 1), author-affiliation numbering requiring review, question-mark/unresolved citations in Related Work (pp. 1–2), a partially rendered mathematical BA formula in §5.3, and an apparent old-code-availability statement referring to the rejected experiment in the back matter (p. 14). These are substantive Q1 submission risks and should be corrected separately; better figures alone will not resolve them.
