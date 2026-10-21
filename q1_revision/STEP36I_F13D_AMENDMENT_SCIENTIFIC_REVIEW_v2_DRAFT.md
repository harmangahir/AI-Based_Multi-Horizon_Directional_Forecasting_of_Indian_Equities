# Step 36I-F13D — Scientific review of post-freeze final-test amendment

**Status:** REVIEW DRAFT — NOT A FORMAL APPROVAL  
**Prepared:** 14 October 2026  
**Experiment:** `q1_revision_v3_clean_2016_2026`  
**Original Step 35 contract SHA-256:** `6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7`  
**Signed proposed cohort SHA-256:** `4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65`  
**Evaluation access state:** 2 historical raw-test file reads; no model prediction or performance metric completed; additional access not authorized.

## 1. Decision under review

A frozen Step 35 target-eligible cohort of 6,480 stock–date endpoints across 324 dates (20 equities per date) did not guarantee construction of each model's required 60-session same-ticker/same-continuity-segment causal history. The proposed, *post-freeze* amendment restricts the evaluation to the mechanically sequence-eligible subset: **5,300 endpoints on 265 dates**. The original Step 35 contract, trained checkpoints, 24 development-frozen thresholds and original access ledger remain unmodified.

This is not a retrospective claim that the original cohort had 5,300 rows or that the revised resampling was part of the original Step 35 freeze.

## 2. Independent structural evidence now available

| Audit | Result | Scope |
|---|---|---|
| F12 | 6,480 original vs 5,300 candidate; 1,180 (18.2099%) excluded | Frozen non-label membership |
| F13A | Step 06/08 files and warm-up audits identified; cleaning rule excludes 2025-03-18 | Frozen source metadata |
| F13B v2 | Step 08 `session_number` reconciled to Step 10 `market_session_id`; exactly 60-session eligibility independently reconstructed | Structural keys only |
| F13B v2 | Exactly 1,180 original endpoints lack 60 complete same-segment historical observations; 5,300 valid endpoints match signed candidate keys | Structural keys only |
| F13C | 265 of 324 original evaluation dates retained; removed dates precisely 2025-05-12–2025-07-31 | Non-label calendar coverage |
| F11 | Ordered 63 features, 20 stock IDs, scaler, regime thresholds and 43,640 development-history rows authenticated | Development artifacts only |
| F4/F5/F6/F8/F10 | Synthetic scoring, segment-aware bootstrap, BH-FDR and transaction interfaces passed their reported offline checks | Synthetic only; no real test inference |

The Step 08 feature-panel warm-up spans 33 sessions at each of the two segment starts, with no later internal feature incompleteness. Segment 1 begins 2025-03-19. The first complete segment-1 feature date is 2025-05-12; the first endpoint with 60 complete segment-1 history observations is 2025-08-01. These findings are consistent with the frozen causal sequence requirement.

**Interpretation:** There is strong evidence that all 1,180 exclusions are mechanically caused by the existing sequence-history rule, rather than filtering on realized model performance. This does **not** demonstrate representativeness of the retained dates or show how any model would have performed during the excluded window.

## 3. Temporal selection and reporting limitations

- The amended cohort loses all **36** originally eligible dates in **2025 Q2** (0% retained); **2025 Q3** retains 41/64 dates, with 23 excluded in July. The contiguous excluded original-evaluation interval is **12 May–31 July 2025**, 59 dates, or 1,180 stock-date endpoints.
- **April 2025 was already absent from the original Step 35 evaluation cohort** and must not be counted as a new amendment loss.
- Retained blocks are **48 dates** (2025-01-01–2025-03-07) and **217 dates** (2025-08-01–2026-06-22). Both include all 20 stocks each session.
- The primary estimand is directional skill **on these 265 retained dates**, not performance over the entire nominal January 2025–June 2026 market history and not performance during the excluded interval.
- The structural audits have **not** established that retained and omitted dates are comparable in volatility regimes, predictor distributions, market shocks, class balance, or intrinsic forecasting difficulty. Do not claim comparability without separate authorized evidence. No performance-weighted missingness correction is defensible from these audits.

## 4. Statistical review of the amended segment-stratified procedure

**Frozen elements maintained:** 24 architecture × horizon tests; BA null ≤0.50 vs one-sided alternative >0.50; B=2,000 replicates; L=20 primary, L=10 and 40 sensitivity; same development-frozen thresholds; no best-seed promotion; Benjamini–Hochberg FDR on the 24 primary BA tests only. The original evaluation used date-level circular moving blocks.

**Post-freeze modification:** Resampling preserves exactly 48 and 217 sessions in two strata. Date resamples keep the 20-stock cross-section together and never cross a continuity boundary. This is a proposed repair for discontinuous predictor histories, not a prespecified original Step 35 analysis.

**Inferential limitations to disclose and monitor:**

1. **Short-stratum dependence:** L=20 is 41.7% of the 48-date segment and L=40 is 83.3%. Circular resampling is mechanically well defined; it does not by itself establish accurate finite-sample coverage or p-value validity under time-varying, strongly dependent observations. Treat L=40 as a fragile diagnostic, not a decisive robustness confirmation.
2. **Bootstrap assumptions:** Resampling within each segment implicitly relies on within-segment dependence regularity and usable effective information; these assumptions have not been established by synthetic tests. After the authorized run, inspect stability of intervals and conclusions across L values without choosing the most favorable block length.
3. **Multiple testing:** BH across 24 tests is mathematically implemented and frozen, but control under arbitrary dependence among architecture/horizon p-values is not guaranteed. State the assumption and consider the **Benjamini–Yekutieli (BY) adjustment as a separately labeled, post-freeze dependence-robust multiplicity sensitivity** *if primary p-values are valid*. Do not replace or reinterpret the frozen primary BH procedure based on observed significance.
4. **One-sided testing:** The centered-bootstrap p-value is a test of BA above 0.50 for the amended date population, not evidence of economic profitability, trading capacity, or statistically significant head-to-head model superiority. The 84 paired model-difference intervals are exploratory unless a separately controlled family is defined.
5. **Test-access history:** Two unsuccessful raw-test reads parsed/validated records, including labels, before model inference/scoring. They must be disclosed. No new thresholds, checkpoints, model choices, hypotheses, or significance definitions may be selected using test outcomes.

**Scientific judgement:** This amendment is a potentially defensible methodological repair with important coverage and inferential limitations. Its mathematical feasibility and synthetic unit tests are verified; the empirical coverage properties of the proposed bootstrap are not proven. A favorable real-test result must not retroactively establish the validity of the amendment.

## 5. Recommended manuscript claim status

The primary hypotheses and trained models were frozen before real-test scoring, **but** the analyzed test cohort and segment-stratified resampling were amended after two unsuccessful file reads. Describe findings as an **amended held-out evaluation with prespecified model/threshold/hypothesis family**, not an unchanged, one-shot, fully prespecified confirmatory experiment. Qualify confirmatory language accordingly and preserve all unfavorable and null results.

Suggested methods disclosure (adapt only after approval):

> Following the final-test protocol freeze, an implementation failure exposed a mismatch between target eligibility and the fixed 60-session same-segment predictor-history requirement. An independent, label-free reconstruction showed that 1,180 of the original 6,480 eligible stock–date endpoints lacked valid causal input histories. We therefore proposed an explicitly post-freeze restriction to 5,300 endpoints spanning 265 sessions; all 40 frozen checkpoints, 24 development-derived decision thresholds and 24 primary hypotheses were retained. To avoid resampling across the resulting continuity break, the proposed uncertainty analysis preserves 48- and 217-session strata within date-block bootstrap replicates. The amendment removes all originally eligible 2025-Q2 dates and part of 2025-Q3, limiting temporal representativeness. Two unsuccessful accesses to the held-out file occurred before model inference or scoring; no held-out performance statistics were computed during those attempts. [Add authorized review decision and actual results here only after completion.]

## 6. Mandatory decisions before production

**A. Scientific amendment (requires human/authorized methodological sign-off):**
- [ ] Approve the fixed 5,300-row, 265-date sequence-eligible population and the exact excluded-date ledger as a post-freeze amendment.
- [ ] Approve the fixed 48/217 segment-stratified L=20 bootstrap for amended primary inference, retain L=10/40 as sensitivity, and document short-segment limitations.
- [ ] Preserve BH-FDR as the frozen primary method; decide *before scoring* whether an explicitly secondary BY sensitivity will be reported.
- [ ] Accept the loss of full Q2-2025 coverage and commit to a restricted-population claim.
- [ ] Require disclosure of two unsuccessful held-out file reads and changed cohort/inference scope.

**B. Engineering (separate review):**
- [ ] Complete and independently review a real production launcher authenticating F11 inputs before any access.
- [ ] Connect the F7 single-read reservation mechanism, the F9 computation assembly and the F3 atomic artifact commit into one fail-closed program.
- [ ] Define one-attempt crash/failure handling and verify preserved historical access state and no-overwrite output paths.

**C. Access approval (separate, explicit authorization):**
- [ ] Authorize exactly one additional raw-test read only after A and B are satisfied. Scientific amendment approval **does not imply** raw-test access approval.

Reviewer: __________________________  Date: __________________________
Decision: [ ] Approve amendment  [ ] Request changes  [ ] Reject
Reasons / limitations: ________________________________________________________________

**Current state:** Amendment NOT approved; third raw-test access NOT authorized; real final-test scoring NOT completed.
