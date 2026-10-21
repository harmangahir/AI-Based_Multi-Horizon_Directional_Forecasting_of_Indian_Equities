# Step 36I–F29R3 — Baselines, diagnostics and scientific disclosure

**Status: VERSIONED RELEASE CANDIDATE; NO HELD-OUT TEST ACCESS OR SIGNED EXECUTION PERMIT.**

## Research motivation

The rejected paper and frozen Step35 protocol required credible comparisons and more mature empirical diagnostics. F29R3 adds supplementary comparison/diagnostic reports to the **same finalization transaction**, after the exact frozen ensemble predictions are authenticated. It does not overwrite original Step35, the F13E owner-approved amendment, F29R2, F20R1, the **40 checkpoint files**, the **24 development-selected thresholds**, or the **24 primary balanced-accuracy hypotheses**.

### Scientific analyses now implemented

| Analysis | Definition | Generated artifact | Status |
|---|---|---|---|
| Three naive baselines × 3 horizons | Always-up, always-down, sign of observed day-t 1-day stock return carried forward to the 1/3/5-day outcome | `supplementary_baselines.csv` (9 rows) | **Post-freeze supplementary**, no model selection |
| Model vs. persistence | Eight original architectures × three horizons, paired common-date/stock, segment-stratified circular blocks, B=2,000 L=20. Delta BA and percentile intervals, **no formal p-values/FDR promotion** | `supplementary_model_vs_persistence_paired_block_ci.csv` (24 rows) | Exploratory |
| Each NSE ticker | Frozen model and horizon; n, positives, BA, ROC-AUC, MCC and Brier | `supplementary_stock_level.csv` (480 rows) | **Already specified** by frozen Step35 as descriptive analysis; MCC additional descriptive metric |
| Volatility regimes | Regime IDs 0/1/2, 20-stock frozen, final-development thresholds, no test refit | `supplementary_regime_level.csv` (72 rows) | Already specified by frozen Step35 |
| Continuity segments | Separate 48-date and 217-date segments | `supplementary_segment_level.csv` (48 rows) | Amended cohort diagnostics |
| Calendar quarters | Retained reference dates only; explicitly absent 2025Q2 | `supplementary_calendar_quarter.csv` (24 rows per observed quarter) | Supplementary descriptive |
| 10-bin fixed calibration | Probability bins [0,0.1), ...,[0.9,1] ; fixed ensemble probabilities, no recalibration | `supplementary_calibration_10_fixed_bins.csv` (240 rows) | Already specified by frozen Step35 |
| Methodology evidence | Definitions, 24-model cross-check, source hashes and explicit nonclaims | `supplementary_methods_manifest.json` | Manifest-bound |

**Return-sign baseline:** The existing frozen 60 × 63 predictor tensor contains `return_1d` at feature index 5. F29R3 obtains the sign at the **reference day's final observation** by reversing only that feature's frozen development-only MinMax scaling, with exact-zero checked at float32 precision. It predicts `1` if the return is positive, else `0`. No future target, test-label-derived threshold, re-estimated scaler or test-era fitted parameter enters the baseline. This is a simple continuation heuristic, **not** an additional trained model and **not** a prespecified primary hypothesis. The zero sign convention is predeclared.

**Pairing:** The 24 comparisons to persistence reuse the frozen F4 date-cluster bootstrap index generator at L=20, preserving all 20 stocks on each selected session and the 48/217 continuity segments. The 95% intervals are *descriptive, unadjusted* intervals, not confirmatory superiority or multiple-testing-controlled findings. The original bootstrap/FDR tables and 84 model-vs-model exploratory intervals stay unchanged.

**Single-class subgroup rule:** For a subgroup with only one class, balanced accuracy, ROC-AUC and MCC are undefined: results are blank/NaN with `single_class_group=True`, not silently assigned 0.5 or 0.0. Brier and prevalence remain computable.

**Output integrity:** F20R2 includes all eight additional artifacts inside `completed_evaluation/`, atomically committed with the original result artifacts. The existing F29R3 reserved final receipt includes the supplementary-methods manifest SHA-256 and does not publish partial results. F29R3 also rechecks all 24 observed model BA values against the independent original F1 scorer before committing.

### Important limits and SCIE Q1 disclosure

1. **No conventional development-trained Logistic Regression, Random Forest, Gradient Boosting, or other classical ML models are added.** These three naive baselines improve the benchmark but **do not fully resolve** the prior rejection's criticism of insufficient conventional/recent comparative baselines. Further classical baselines would require an explicit development-only trained/frozen artifact and transparent supplementary scope; do not claim them here.
2. **No transaction-cost, Sharpe ratio, profit or trading simulation is implemented.** The preserved Phase A interface contains binary future direction labels, not the forward magnitude of price returns needed for valid net-profit comparisons. These claims remain unavailable; report as not evaluated.
3. No observed held-out result is known or reported. Synthetic verification establishes interface behavior, not predictive efficacy. Results could be null or unfavorable.
4. The post-freeze technical-cohort amendment still removes **1,180 originally frozen endpoints**; only 5,300 retained stock–date rows on 265 dates are evaluated. The entire 2025 Q2 evaluation coverage is missing; this and the 48/217 segment imbalance are material limitations.
5. Study-owner self-review is not independent expert review. Signing-key authentication proves origin, not methodological soundness.
6. This code must not be considered production-authorized or scientifically final until the source-bound methodological self-review and a distinct explicit third-read permit are granted **after** independent technical verification of the release path. No new test access is authorized by this package.

### Source boundaries

- Supplementary pure computation: `step36i_f29r3_supplementary_baselines_diagnostics_REVIEW_ONLY.py`.
- Versioned finalization: `step36i_f20r2_supplementary_finalization_CANDIDATE.py` (copy of F20R1 with supplementary verified source and artifact inventory; original F20R1 untouched).
- Versioned release controller: `step36i_f29r3_scientifically_extended_signature_gated_CANDIDATE.py` (copy of F29R2, new F20R2 and supplementary source hashes, final artifact receipt; F29R2 untouched).
- Python AST/compilation, source SHA-256 binding, frozen-primary-row count and isolated synthetic diagnostic/naive-baseline tests passed in preparation. **Live WSL source preflight is outstanding.**

### Action policy

Run only the **single `--offline-preflight`** for F29R3, after source SHA checks. Do **not** execute `--phase-a`, `--phase-b` or `--finalize`, issue an attestation/permit, publish results, or open the sealed source until the final reviewed release and separately explicit permission. No key material or passwords are included in this package.
