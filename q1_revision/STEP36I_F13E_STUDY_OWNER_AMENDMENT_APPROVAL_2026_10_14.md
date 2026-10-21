# STEP 36I-F13E — Study-owner amendment decision

**Decision (14 October 2026): APPROVED BY STUDY OWNER for amended-design implementation and subsequent independent review.**

**Evidence:** In this conversation, the user answered **“yes”** to the explicit request for formal approval of the Step 36I-F13D amended scientific design, including its documented limitations. This is a **study-owner authorization** only, not a separately completed independent methodological review.

**Basis:** `STEP36I_F13D_AMENDMENT_SCIENTIFIC_REVIEW_v2_DRAFT.md` (SHA-256 `73fc6c9d5b52dde45ae31fe475b88bb90713459b6b7eb4ed79fbc5a807d58a29`). The original Step 35 contract is **unchanged**.

## Approved scope

- Fixed cohort: **5,300 stock-date endpoints on 265 sessions**, 20 equities/date; original frozen target cohort **6,480** endpoints on 324 dates.
- Exclude **1,180 endpoints/59 dates (12 May--31 July 2025)** solely due to the independently verified 60-complete-session same-ticker/same-segment causal-history requirement.
- Maintain the **40 frozen model-seed checkpoints**, **24 development-frozen thresholds**, **24 primary hypotheses** and primary metric **balanced accuracy**.
- Amended segment-stratified circular date-block bootstrap: **48 + 217** retained sessions, **B=2,000**, primary **L=20**, sensitivity **L=10 and L=40**, with all 20 stocks on a sampled date kept together and no crossing of continuity boundaries.
- Primary BH-FDR correction on the **24** BA tests; pairwise intervals remain exploratory. The optional secondary BY sensitivity has **not** been separately finalized and may not displace the primary BH method.
- Disclose post-freeze amendment, excluded temporal coverage (all original-evaluation dates in 2025 Q2), short first segment, fragile L=40 sensitivity, statistical dependence assumptions, and **two unsuccessful historical raw-test reads**.
- Restrict final claims to the **265 retained dates** and report all results, including null results.

## Not approved by this decision

- [ ] Independent methodological sign-off (still required).
- [ ] Real-data production launcher, access journal or atomic transaction as an approved implementation (still required).
- [ ] A third raw-test read or any final-test execution (**explicitly NOT authorized**).
- [ ] Revision to the original Step 35 scientific contract, checkpoints, thresholds or primary hypotheses.

**Next stage:** finish and independently audit production integration; separately document independent scientific review and explicit one-read authorization, if granted. No live test data may be opened at this stage.

**Machine-readable companion:** `STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json`.
