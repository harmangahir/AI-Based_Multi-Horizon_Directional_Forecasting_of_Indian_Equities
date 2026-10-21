# STEP 36I-F12 — Post-freeze scientific amendment: independent-review brief

**Status:** REVIEW DRAFT — NOT APPROVED  
**Date:** 13 October 2026  
**Experiment:** `q1_revision_v3_clean_2016_2026`  
**Original scientific contract SHA-256:** `6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7`

## 1. Decision to be reviewed

Determine whether the original common multitask *target-eligible* evaluation cohort should be restricted, without performance-based selection, to the subset satisfying the already-required 60-session same-continuity-segment causal predictor-history interface. The amendment is **not** part of the original pretest Step 35 protocol; it must not be labeled an unchanged confirmatory design.

## 2. Original versus proposed contract

| Item | Original Step 35 | Proposed post-freeze modification |
|---|---|---|
| Target-eligible test endpoints | 6,480 | 5,300 sequence-eligible endpoints |
| Reference dates | 324 | 265 |
| Stocks per date | 20 | 20 |
| Excluded | N/A | 1,180 endpoints (59 dates; 18.21%) |
| Retained segments | Originally not segmented for resampling | 48 dates + 217 dates |
| Input length / feature width | 60 sessions / 63 predictors | Unchanged |
| Model checkpoints | 40 (8 families × 5 seeds) | Unchanged |
| Frozen thresholds | 24 | Unchanged |
| Primary BA null | BA ≤ 0.5; alternative BA > 0.5 | Unchanged |
| Resampling | Circular date-block bootstrap, B=2,000, L=20 | **Segment-stratified** circular date-block bootstrap, B=2,000, L=20 |
| Sensitivities | L=10,40 | Segmented L=10,40 |
| Multiple testing | BH-FDR, one family of 24 tests | Same declared family and procedure; changed resampling population |

Candidate keys SHA-256: `4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65`. Frozen Step-10 structural membership ledger SHA-256: `82772196c92e88f1926a75a70c42f5f19119d7245d93b8364455fd28642ee121`.

## 3. Technical root cause and evidence

A 2025-03-18 date was excluded by the existing market-integrity cleaning process because of anomalous collective stock OHLCV/volume patterns. This produced a continuity-segment restart on 2025-03-19 and a stock-indicator warm-up period. The original target-eligibility condition was insufficient to guarantee the full 60-observation **same-segment** model-input history at every otherwise eligible endpoint. A second Step 36 recovery attempt failed in `materialize_sequences` on `ASIANPAINT.NS`, 2025-05-12, before inference or scoring.

Prior offline analyses identified 59 dates (2025-05-12 through 2025-07-31) lacking valid 60-session same-segment causal input history under the frozen feature construction, totaling 1,180 ticker-dates. The proposed cohort retains 2025-01-01 through 2025-03-07 (48 dates) and 2025-08-01 through 2026-06-22 (217 dates). F12 checks these against **non-label** frozen membership; it does not access held-out labels. An additional methodological reviewer must independently validate the causal-history exclusion mechanism, not merely the key counts.

## 4. Required safeguards before scientific sign-off

- Demonstrate the exclusion is mechanically implied by the frozen data-integrity, warm-up and 60-session same-segment rules, independent of held-out outcomes; preserve the complete 6,480-to-5,300 key reconciliation.
- Discuss how deleting 59 *target-eligible dates* may change market-regime and time-period representativeness. Provide a non-outcome coverage analysis where supported by previously frozen covariates; do not impute or speculate about omitted test performance.
- Review the segment-stratified bootstrap assumptions. The first stratum contains only 48 dates; L=40 yields limited independent information and needs cautious sensitivity interpretation. Keep 20-stock common-date clusters together and prevent blocks crossing the gap.
- Verify one-sided centered bootstrap inference and BH-FDR scope under temporal/cross-sectional dependence. Do not claim that BH alone solves every dependence or selection concern.
- Clearly distinguish original prespecified hypotheses from **post-freeze amended** evaluation. Do not call this an unchanged one-shot held-out test or retroactively prespecified cohort.
- Retain all eight architectures, all five seeds, all 24 thresholds, and all results including null findings. Any newly added naive/conventional comparator should be transparently presented as supplemental or exploratory if not frozen before held-out exposure.
- Independently review production loader/journal/transaction integration before any further data access.

## 5. Mandatory held-out access disclosure

The sealed test parquet was opened and read twice during unsuccessful implementation/recovery attempts. Both attempts parsed or validated held-out records/labels as part of execution, but no held-out performance metric or model prediction was completed. The access ledger currently records **2 completed reads, zero completed scorings**. There is **no authorization for another read**, and no permission to reuse the old Step 36 script as an ungated launcher.

## 6. Decision record (to be completed by authorized reviewer)

**Scientific decision:** [ ] Approve amendment  [ ] Revise and resubmit  [ ] Reject amendment  
**Evidence reviewed:** __________________________  
**Cohort representativeness and inference limitations accepted?** __________________________  
**Classification of empirical claims (confirmatory amended / exploratory / other):** __________________________  
**Reviewer and date:** __________________________

**Important:** Scientific approval of the amendment, if subsequently granted, does **not** itself authorize opening the test a third time. Third-read authorization requires a separate explicit decision and a reviewed, fail-closed production launcher.

## 7. Interim manuscript wording (not an empirical result)

> After the original final-test protocol was frozen, a technical materialization audit identified a mismatch between target eligibility and the frozen 60-session same-continuity-segment predictor-history requirement. The original 6,480 target-eligible stock-date observations spanned 324 dates; a mechanically determined subset of 5,300 observations on 265 dates was sequence eligible under the unchanged input definition. This 18.21% reduction and the associated segment-stratified date-block bootstrap constitute a post-freeze evaluation amendment, not a prespecified population change. All trained checkpoints, development-derived thresholds and model specifications were retained. Two unsuccessful held-out-file accesses occurred before model inference and metric computation; their provenance was preserved. [Insert formal amendment approval status and final-test results only after independent validation.]

No ranking, significance claim, or held-out result is established in this document.
