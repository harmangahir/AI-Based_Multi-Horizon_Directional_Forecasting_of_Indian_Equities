# STEP 36I-F29R5 — Frozen development-only ML baseline integration

**Status:** versioned code candidate; real-held-out run NOT authorized. F29R4 training was completed by the study owner in WSL; this package does not retrain or access raw test. A WSL offline preflight and source review are still required. None of these tools or outputs constitute independent sign-off.

## Immutable input boundary

- F29R4 freeze directory: `feature_data/step36i_f29r4_devonly_ml_baselines_v1` relative to the frozen experiment.
- Signed-by-code expected manifest SHA-256: `de89cb9d2c17d374aea3870f66a5633d3a2b1a8d86a6c37310d31dc79d3be09f` (reported in the genuine development-only training output).
- Nine model hashes are independently pinned in the F29R5 scoring module to the values reported during the training freeze.
- F29R4 used the **last reference-date snapshot** `X[:, -1, :]` from each 60-session frozen sequence (63 features). The new scorer uses exactly that same representation from **already verified** amended Phase A, with the same frozen scaler/feature ordering.
- Fixed probability threshold 0.5; no test-set-based tuning, selection or hyperparameter modification.

## New scientific outputs (in addition to F29R3)

The F20R3 atomic evaluation finalizer adds **ten** `ml_devonly_*` artifacts alongside the prior outputs:

1. `ml_devonly_predictions.parquet` — all 5,300 endpoint probabilities and decisions for the 9 ML model/horizon pairs (scored **only after** separately authorized Phase A and all frozen neural seed outputs are verified).
2. `ml_devonly_classification_metrics.csv` — nine model/horizon metrics.
3. `ml_devonly_vs_persistence_paired_block_ci.csv` — nine paired 95% percentile intervals.
4. `ml_devonly_neural_vs_ml_paired_block_ci.csv` — 72 neural-versus-ML paired 95% percentile intervals, by horizon.
5. `ml_devonly_stock_level.csv` — 180 descriptive stock/model/horizon rows.
6. `ml_devonly_regime_level.csv` — 27 regime/model/horizon rows.
7. `ml_devonly_segment_level.csv` — 18 segment/model/horizon rows.
8. `ml_devonly_calendar_quarter.csv` — descriptive quarter breakdown.
9. `ml_devonly_calibration_10_bins.csv` — 90 fixed-bin calibration records.
10. `ml_devonly_methods_manifest.json` — scientific limitations, hashes and inventory.

All new intervals use the **same 2,000 paired segment-stratified date-cluster resamples** from the frozen F4 evaluator (primary block length 20; 48+217 dates). They are exploratory unadjusted percentile intervals, not new primary BH-FDR claims.

## Unchanged

- 20 NSE equities, NIFTY features, 63-feature order, 60-session input and 1d/3d/5d targets.
- Original 40 neural checkpoints and 24 neural thresholds.
- Original 24 primary BA tests, 72 primary/sensitivity bootstrap records and 84 exploratory neural pair comparisons.
- F29R3 previous naive baseline, regime, segment, quarter, stock and calibration outputs.
- Amended cohort: 5,300 endpoints / 265 dates / 48+217 continuity segments.
- Original Step35 freeze, F13E study-owner amendment approval, historical record of two failed raw-test reads.

## Limitations & release gates

- Baselines are **post-freeze supplementary** and cannot be represented as pre-registered confirmatory competitors.
- The original failed raw-test accesses may have exposed test labels to the process even though no metrics were computed. This limits claims of perfect test blindness and must be reported.
- A last-snapshot classifier is **not sequence-equivalent** to a 60-session neural model; both share the same 63 causal indicators and panel/horizons, but differ in sequence capacity.
- F29R4 full-development scaling is suitable for its final-development fit and later hold-out evaluation, **not** evidence of leak-free cross-validation. No train/validation model-selection claims are made.
- Calibration probability metrics are descriptive; fixed 0.5 ML decisions are not identical to validation-selected neural thresholds.
- No independently conducted code or statistical review. Single-investigator methodological self-review must be described accurately.
- Signed method approval and a distinct, explicit third raw-test read permit have **not** been created.
- The F29R5 candidate retains authorization checks; offline preflight neither opens raw test nor runs GPU inference. Phase A, Phase B and finalization must remain blocked until final source review and permit.
- No economic-return or transaction-cost P&L claim.

## Verification performed before distribution

- Three sources compiled; controller/F20/ML-source hash bindings reconciled.
- F29R5 scorer ran on a **synthetic** 5,300-row sample with mock nine model probability APIs; obtained 9 ML metrics, 9 paired persistence intervals, 72 neural comparisons and ten artifact names. Synthetic parquet output was mocked because pyarrow is not installed in the assistant's local container; user WSL already uses pyarrow. This does **not** constitute end-to-end production inference validation.
- Actual nine F29R4 model bytes are only on the user's WSL machine; their independent SHA verification is deferred to WSL offline preflight.
