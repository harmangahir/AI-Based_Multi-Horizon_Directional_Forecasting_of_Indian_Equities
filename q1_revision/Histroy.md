# CONTINUATION HANDOFF — Indian Equities Multi-Horizon Directional Forecasting
**Snapshot:** 14 October 2026; latest verified step F19; F20 prepared but not yet run.  
**Status:** Original Step35 scientific freeze intact. Owner-approved *post-freeze* cohort amendment, independent methodological sign-off pending. Two historical raw-test reads; **zero successful held-out scorings**. No third read authorized.

## 1. Research objective and manuscript

**Working title:** *Artificial Intelligence-Based Multi-Horizon Directional Forecasting of Indian Equities: A Leakage-Controlled and Dependence-Aware Evaluation.*  
**Aim:** Compare modern architectures for 1-, 3-, and 5-session up/down forecasting across a fixed panel of Indian equities, with strict chronological preprocessing, development-only model and threshold choices, multi-seed ensembles, and inference that accounts for time/cross-stock dependence. Claims must follow real held-out evidence, even if results are null or modest.  
**Submission context:** Springer *International Journal of Computational Intelligence Systems* has been used as the manuscript template/target; verify **SCIE JCR Q1** status independently if that ranking is a hard requirement, since indexing schemes differ. Previous manuscript was desk-rejected for immature experimental analysis and insufficient comparators/recent references. A Springer `sn-jnl` LaTeX pre-results manuscript and BibTeX files were generated in earlier turns. Do not put invented held-out results into them.

**Universe (20 NSE symbols):** ASIANPAINT.NS, AXISBANK.NS, BAJFINANCE.NS, BHARTIARTL.NS, HDFCBANK.NS, HINDUNILVR.NS, ICICIBANK.NS, INFY.NS, ITC.NS, KOTAKBANK.NS, LT.NS, MARUTI.NS, NTPC.NS, POWERGRID.NS, RELIANCE.NS, SBIN.NS, SUNPHARMA.NS, TCS.NS, TITAN.NS, ULTRACEMCO.NS. Market reference: `^NSEI` (NIFTY 50).

## 2. Execution environment and exact paths

- Linux/WSL2 Ubuntu, Conda environment: `Harman_GPU`.
- Source directory:
  `~/projects/AI-Based_Multi-Horizon_Directional_Forecasting_of_Indian_Equities/q1_revision`
- Experiment directory:
  `~/projects/AI-Based_Multi-Horizon_Directional_Forecasting_of_Indian_Equities/stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026`
- GPU workstation reported previously: RTX 5070 8 GB, 32 GB RAM; CUDA/PyTorch environment already validated in prior stages.
- **Workflow preference:** One script/command per step; user pastes full WSL output; audit that output before issuing the next step. Never assume a script executed just because it was linked or its SHA-256 matched.

## 3. Frozen scientific and experimental decisions

| Component | Frozen design |
|---|---|
| Market-data window | 2016-01-01 through 2026-06-30 |
| Development period | Through 2024-12-31 |
| Held-out period | 2025-01-01 onward through eligible June 2026 endpoints |
| Instruments | 20 NSE stocks plus NIFTY `^NSEI` context |
| Input | 60 prior/current causal sessions, 63 ordered features (47 stock + 16 index) |
| Targets | Three directional labels, 1/3/5 future sessions |
| Preprocessing | Fit MinMax scaling on training/development only; no clipping; stock-specific regime q33/q67 from training data only |
| CV/development | Five expanding chronological folds (2020–2024), horizon-aware boundaries |
| Model families | DLinear, NLinear, PatchTST, TimesNet, iTransformer, Fredformer, FRNet, MLF |
| Selected configs | `DLINEAR_B`, `NLINEAR_B`, `PATCHTST_B`, `TIMESNET_B`, `ITRANSFORMER_A`, `FREDFORMER_B`, `FRNET_A`, `MLF_A` |
| Final seeds | 42, 43, 44, 45, 46 per selected architecture: 40 frozen model-seed checkpoints |
| Thresholds | 24 development-only OOF thresholds; no test tuning |
| Original Step35 target cohort | 6,480 endpoints, 324 dates (20 stocks/date) |
| Main metric/inference | Balanced accuracy (BA), one-sided `BA > 0.50`, B=2,000 session-block bootstrap, primary L=20, sensitivity L=10 and L=40, BH-FDR over 24 primary architecture–horizon tests |
| Inference behavior | Preserve sorted endpoint order, same physical batches and incomplete tail. TimesNet FFT-based inference can change with physical batch composition. |

**Frozen provenance values** (do not rewrite frozen inputs):

- Step35 contract SHA256: `6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7`
- Step35 experiment fingerprint: `44949863de7e82c3c303ca55beff8c59ae3206ea7eb542a9aab115ff88632bfc`
- Step10 membership ledger SHA256: `82772196c92e88f1926a75a70c42f5f19119d7245d93b8364455fd28642ee121`
- Step36 original source SHA256: `b5321fe514c7f9f34d6d675debbbca1fd292c58b2874b724987b3a819671b131`
- Recovery-amended Step36 source SHA256: `6de04ba7e167449101e2e16bab47ca4ce088f11284c4cbc93b61a6cfbb979f12`
- Step36A recovery amendment SHA256: `1e563468c21dd0f806aa33dc54552a4ce476fc7f231e8aea0be611c72288428c`
- Sealed raw test Parquet SHA256 (recorded in audit; **do not rehash by opening it now**): `ab325ede85fbb6d7819e85472274b82d151d350f39a872a17747c6978b1dddae`
- Signed Step08 manifest SHA256: `dd4e850de15f752425c6bf3df895010f089f5ad32459cb97046744871b5680a8`

## 4. Two prior held-out accesses: critical safety boundary

1. Original Step36 first real raw-test file read: test file opened, hashed and parsed; failed **before materialization and before predictions/scoring** because `eligible_common_multitask` was not persisted in the raw Step10 test panel.
2. Step36A recovery attempted another raw-test read, derived structural eligibility, but failed within actual `materialize_sequences` at `('ASIANPAINT.NS', 1) 2025-05-12` due to insufficient causal history.
3. **No real final-test predictions, final scoring, or performance metrics were produced.** The correct wording is *two prior raw-test reads, no completed final-test performance scoring*—NOT “test unopened” or “raw test never inspected.”
4. Live historical access marker: `manifests/36_final_test_access_consumption_state.json`; `raw_test_filesystem_open_count_completed=2`, `final_test_scored=false`. It must not be reset, rewritten, or silently bypassed.
5. Production output namespace has repeatedly been audited as vacant: seven protected real-output paths untouched. **No third read is authorized**; source/preflight PASS does not change that.

## 5. Why Step36 failed and scientific post-freeze amendment

- Frozen Step04 cleaning dropped `2025-03-18` from the complete aligned stock panel under the pre-established market-wide rule: remove a market date only when at least 90% of stocks simultaneously have zero volume and identical OHLC.
- The continuity split creates a second segment beginning `2025-03-19`.
- Step06 rolling feature construction requires 33-session warm-up in each continuity segment. Step08 full and complete date counts independently verified:
  - Segment 0: 2,268 raw dates vs 2,235 complete; 33 warm-up dates; earliest complete `2016-02-19`.
  - Segment 1: 314 raw dates vs 281 complete; 33 warm-up dates; earliest complete `2025-05-12`.
  - Exactly 66 incomplete dates are the two 33-date segment starts, with **no later internal missingness**.
- A valid 60-session model sequence must use continuous, same-ticker, same-segment **complete** observations ending at its endpoint. For every stock, the 59 dates from `2025-05-12` through `2025-07-31` have only 1–59 available complete history sessions, so are ineligible.
- The first eligible second-segment endpoint is `2025-08-01`.
- Frozen Step10 structural final-test membership: 6,480 rows / 324 dates. Independently reconstructing causal eligibility yields exactly 5,300 rows / 265 dates and excludes 1,180 rows / 59 dates (18.2099%), all 20 stocks on each removed date.
- Retained subperiods: segment 0 has 48 dates / 960 rows (`2025-01-01` through `2025-03-07`); segment 1 has 217 dates / 4,340 rows (`2025-08-01` through `2026-06-22`).
- Signed candidate keys file is `36i_candidate_sequence_eligible_keys_DRAFT.csv` (see E2 bundle/audit area); candidate membership SHA256 `4c9364df9e761b81d0776cdf927a88f25d785178640cb3ffe380ad93673a9d65`.
- This changes the original Step35 cohort and the resampling design **after freeze**, and must be declared a post-freeze methodological amendment, not falsely presented as unchanged prespecification. It does not license further tuning of model architectures, thresholds or checkpoints.

**Owner-approved amended scientific design (F13E, 14 Oct 2026):** Retain 5,300 exact endpoints; use segment-stratified circular moving-block bootstrap with B=2,000, fixed allocation of 48 dates from segment 0 and 217 dates from segment 1, same-date 20-stock clusters, blocks never crossing the gap, primary L=20, L=10/40 sensitivities, and BH-FDR on 24 primary BA hypotheses. The 40 checkpoints and 24 thresholds are unchanged. The amendment was **approved by the user/study owner for implementation and further review**, not independently signed off, and third read **not** approved.

**Coverage limitations from F13C:** 265/324 dates retained = 81.79%. All 36 original 2025Q2 reference dates absent from amended cohort, 23 additional dates excluded in 2025Q3; January/February/early-March and August 2025–June 2026 remain. April 2025 was already absent from the **original** 324-date membership, not removed by the new amendment. L=40 spans 83.3% of 48-date first segment, so short-stratum sensitivity CIs deserve particular caution; L=20 is 41.7%. BH under 24 dependent tests requires explicit inferential assumptions and caution. Date-only audit does **not** establish covariate/regime representativeness, market difficulty or performance on excluded dates. Optional BY sensitivity, if later implemented, must be marked post-freeze/exploratory.

**F13E verified approval files:**
- `STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.md`, SHA256 `c53013b5ce6082356e1e2a3e3df137203eeb4ea4270510596255bcbb899f4d49`
- `STEP36I_F13E_STUDY_OWNER_AMENDMENT_APPROVAL_2026_10_14.json`, SHA256 `f8ab9fc4979d7a39b7c2d2759de132f74513232cc7f4583b306de71d5eda1493`
- Both have been independently verified `OK` in WSL.
- Methodological review draft: `STEP36I_F13D_AMENDMENT_SCIENTIFIC_REVIEW_v2_DRAFT.md`; earlier `STEP36I_F12_PROTOCOL_AMENDMENT_INDEPENDENT_REVIEW_DRAFT.md`.

## 6. Signed non-test artifacts, directory names, and schema facts

Experiment-relative frozen files:

- `feature_data/step06_stock_features/06_stock_features_47.parquet`
- `feature_data/step06_stock_features/06_stock_features_47_complete.parquet`
- `feature_data/step08_merged_63_features/08_merged_63_feature_panel.parquet`
- `feature_data/step08_merged_63_features/08_merged_63_feature_panel_complete.parquet`
- `feature_data/step10_chronological_splits/10_split_membership.parquet`
- `feature_data/step12_volatility_regimes/final_development/12_final_development_scaled_regimes.parquet`
- `manifests/06_stock_feature_manifest.json`; `manifests/08_63_feature_panel_manifest.json`
- `audits/06_stock_feature_warmup_audit.csv`
- `audits/08_63_feature_segment_summary.csv`
- `audits/08_complete_key_reconciliation.csv`
- `manifests/04_cleaning_policy.json`
- `manifests/36_final_test_access_consumption_state.json` (protected)
- `manifests/35_final_test_evaluation_contract_freeze_manifest.json` (protected; precise local placement to be verified rather than inferred)
- Actual Python source file: `step36_one_time_final_test_materialization_inference_scoring_v1.py` (original/recovered frozen Step36 source). Never run directly on raw test without separate clearance.

**Schema gotcha documented by F13B v1 failure:** Step08 full/complete Parquet files have structural `session_number`, `Date`, `Ticker`, `continuity_segment_id`—**no** `market_session_id`. Step10 ledger has `market_session_id`, `Date`, `Ticker`, `continuity_segment_id`; Step12 development has both session identifiers. Schema-corrected F13B v2 independently validated the date-key crosswalk `{0:0, 1:1}` and exact Step08/Step10 complete key equality. Do not blindly rename IDs.

Development-history facts from F11/F13B v2: 43,640 signed development rows through `2024-12-31`, 20 stocks, 63-column scaler. Step10 non-label test structure has 6,680 complete keys (not the 6,480 original **target-eligible** endpoints). `43,640 + 6,680 = 50,320` complete ticker-date keys; verified exact Step08 complete/Step10 match.

## 7. Completed Step36I recovery chronology (latest output status)

Each PASS below is restricted to its described scope. **Most stages use synthetic data and never constitute live scoring authorization.**

### Earlier steps 36I-A through 36I-E5

- **36I-A**: Drafted causal subset and protocol-deviation record; status DRAFT_NOT_APPROVED.
- **36I-B1**: Recovered Step36 actual materializer passed synthetic fixture with `X=(5300,60,63)`; negative May 12 example rejected.
- **36I-B2**: Initial synthetic GPU test failed because `torch.cuda.reset_peak_memory_stats` occurred before CUDA initialization. Corrected B2 version passed all 8 model families with an explicit allocation first.
- **36I-C**: Segment-stratified synthetic bootstrap and BH-family checked at L=10/20/40, B=2,000.
- **36I-D1–D2g**: Audited hardcoded original 6480/324 counts, model interface, separate namespace; verified all 40 checkpoint hashes and 24 thresholds; 40 synthetic GPU/checkpoint inferences; 8 ensembles; numeric scoring oracle.
- **36I-E1/E2**: Drafted disclosure and source review bundle (`step36i_e2_source_review_bundle.zip`) and verified two historical failure logs.
- **36I-E3**: `step36i_e3_isolated_amended_runner_DRAFT.py` offline preflight PASS; no raw-test authorization.
- **36I-E4**: `step36i_e4_offline_artifact_transaction_DRAFT.py` synthetic atomic transaction PASS, 8 NPZ ensembles, 24 metric rows, 42,400 long-form prediction rows.
- **36I-E5/E5B**: Manifest inspection and independent verification of all 10 synthetic artifacts and sizes PASS. E4 report `36i_e4_synthetic_transaction_DRAFT.json` SHA256 `30fbd08cecba4e65eb427358c6a24c9d28b984bdf889afc83668a6c9952ab925`, location experiment-relative `audits/step36i_e4_synthetic_artifact_transaction_draft/`. Other E4 files: `SYNTHETIC_ONLY_metrics.csv`, `SYNTHETIC_ONLY_ensemble_predictions.parquet`, 8 NPZs under `synthetic_ensembles/`.

### F1 through F19 — exact local script filenames and verified outcomes

| Stage | Exact filename in `q1_revision` | Verified result |
|---|---|---|
| F1 | `step36i_f1_amended_production_runner_REVIEW_ONLY.py` | PASS; structural 6480/324 to 5300/265, 40 model–seeds and 24 frozen thresholds; real CLI disabled |
| F2 | `step36i_f2_independent_source_review_READ_ONLY.py` | PASS; AST/source and synthetic independent F1 scoring oracle |
| F3 | `step36i_f3_offline_access_journal_atomic_transaction_REVIEW_ONLY.py` | PASS; synthetic durable hash-chained journal, reservation/no-overwrite, crash/replay/tamper rejection |
| F4 | `step36i_f4_segment_stratified_inference_REVIEW_ONLY.py` | PASS; 72 bootstrap/FDR rows, 24 BH q-values, 84 exploratory pairs on synthetic panel |
| F5 | `step36i_f5_independent_statistical_audit_READ_ONLY.py` | PASS; independent B=2000 blocks L10/20/40, exact null/perfect classifier oracles |
| F6 | `step36i_f6_end_to_end_synthetic_transaction_REVIEW_ONLY.py` | PASS; F1 scorer+F4 stats+F3 commit; 4,363,233 synthetic bytes |
| F7 | `step36i_f7_one_read_loader_journal_integration_REVIEW_ONLY.py` | PASS; synthetic single-file-descriptor access, prior durable reservation, no silent retries |
| F8 | `step36i_f8_single_read_to_atomic_commit_INTEGRATION_REVIEW_ONLY.py` | PASS; 1 synthetic file read→40 seeds→scoring/stats→atomic commit; source SHA confirmed after initial file-not-found mistake |
| F9 | `step36i_f9_preloaded_production_assembly_REVIEW_ONLY.py` | PASS; in-memory preloaded materialization, inference, scorer, stats, manifest interfaces; no live launcher |
| F10 | `step36i_f10_independent_assembly_review_READ_ONLY.py` | PASS; independent synthetic F9 40 invocations, real F1/F4 computations, Parquet serialization |
| F11 | `step36i_f11_upstream_provenance_audit_READ_ONLY.py` | PASS; independent signed 63-feature order, 20 IDs, scaler, regimes and 43,640 development rows |
| F12 | `step36i_f12_scientific_amendment_review_READ_ONLY.py` | PASS structural original/candidate comparison; explicitly NOT scientific approval |
| F13A | one-off terminal structural discovery command | PASS; Step06/08 files found, Step04 excluded date verified |
| F13B v1 | `step36i_f13b_causal_history_eligibility_READ_ONLY.py` | **FAILED auditor schema assumption**: Step08 missing `market_session_id`; did not alter files |
| F13B v2 | `step36i_f13b_v2_schema_corrected_READ_ONLY.py` | PASS; independent causal proof: 1180 fail 60-session history, 5300 pass |
| F13C | `step36i_f13c_temporal_coverage_review_READ_ONLY.py` | PASS; missing 2025Q2 & partial Q3 coverage, short-stratum L40 caveat |
| F13D/E | amendment review/approval documents above | F13E study-owner decision YES; both files hash-verified; independent sign-off still pending |
| F14 | `step36i_f14_approval_bound_production_integration_REVIEW_ONLY.py` | PASS; approval-bound read-only synthetic source/journal/commit, production disabled |
| F15 | `step36i_f15_durable_materialization_handoff_REVIEW_ONLY.py` | PASS; saved 5300×60×63 float32 synthetic X + 5300×3 labels + exact keys, 80,306,173 bytes |
| F16 | `step36i_f16_two_phase_resume_independent_REVIEW_ONLY.py` | PASS; injected GPU mock failure; resumed from sealed synthetic arrays, preserved 40 schedules/TimesNet tail |
| F17 | `step36i_f17_two_phase_release_implementation_REVIEW_ONLY.py` | PASS source-only Phase A/B separation; **live loader, journal, commits and production launcher NOT implemented** |
| F18 | `step36i_f18_durable_two_phase_artifact_store_REVIEW_ONLY.py` | PASS; reusable immutable Phase A and per-seed store, 1 completed/39 pending in synthetic test |
| F19 | `step36i_f19_checkpoint_bound_resume_integration_REVIEW_ONLY.py` | **LATEST VERIFIED PASS**; binds saved seed to frozen checkpoint hash, Phase A provenance, physical batch and 1/39 inventory; wrong files/corruptions rejected. No real GPU. |
| F20 | `step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py` | **PREPARED, NO USER RUN OUTPUT RECEIVED YET**; review-only source candidate, production disabled |

**F19 last exact output:**

> `STEP 36I-F19 PASS — ONE-SEED CHECKPOINT-BOUND RECOVERY INTEGRATION, SYNTHETIC ONLY`  
> `Real raw test opened: FALSE | Real GPU: FALSE | Real test scored: FALSE`  
> `Production entrypoint: DISABLED | Additional access authorized: FALSE`

**F19 incidental shell error:** Initial `sha256sum -c` had two trailing spaces after `.py`, interpreted as part of filename; corrected direct `sha256sum` returned expected hash `727d57ff80876a214166f71c407f2b9ecd4686b7e216a7d1649c692835d9213e`; then F19 preflight passed. Not an implementation failure.

## 8. Current outstanding engineering and scientific blockers

1. **F20 has not yet been run locally.** Its offline code integration needs hash verification, WSL preflight, and independent source audit; this is not yet a production release.
2. **Real single-open loader and durable access accounting:** Complete independently reviewed implementation that records an authorized third-read reservation *before* any possible raw-file open; verify one file descriptor and SHA; fail closed on crashed/duplicate attempts. Do **not** implement a bypass or reset the 2-read historical marker.
3. **Authenticated Phase A live seal:** Verify all signed upstream bytes in the real launcher, materialize only exact approved 5,300 endpoint keys (60×63), bind valid stock/date/segment/order/labels and provenance, atomically commit and rehash the durable package before GPU inference.
4. **Recoverable Phase B live GPU:** Load Phase A without reopening test file; preserve all 40 frozen checkpoint/seed hashes and model-specific exact physical batches (TimesNet tail); atomically commit per-seed results, authenticate them at resume, and never silently overwrite or retry raw test access. Finalize eight ensembles, 24 metrics, bootstrap/BH-FDR and an immutable output manifest after all seeds complete.
5. **Independent engineering and methodological release sign-off:** Review real launcher, source, authorization gate, failure/resume semantics, exact cohort, short-segment bootstrap limitations, BH dependence and manuscript reporting. Study-owner F13E approval is **not** the same thing as independent sign-off.
6. **Explicit separate third raw-test read authorization:** Must be obtained **after** those audits and before live execution. No general statement such as “proceed” or PASS on an offline preflight should be construed as this authorization.
7. **Empirical test results and manuscript:** Not yet produced. Do not report BA, AUC, MCC, PR-AUC, Brier, q-values, model rankings or predictive edge on the held-out set until verified real results exist. After release, present 24 architecture×horizon results, uncertainties, q-values, limited temporal coverage, exclusion mechanism, two prior failed raw-data reads, and cautious null/modest interpretations. Consider simple baselines and covariate/regime coverage only under appropriate pre-specified vs exploratory disclosure.

## 9. Next action — **one command only**, F20 offline preflight

The user has **not** supplied an F20 result. First ensure this exact script has been downloaded into the `q1_revision` folder:

`step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py`

**Expected F20 script SHA256** from the last assistant message:  
`937de4fb1dc7a58f6ecb8449ad55a515a5dcb28120adbeaa82a4e31c0af5d018`

Execute in WSL:

```bash
cd ~/projects/AI-Based_Multi-Horizon_Directional_Forecasting_of_Indian_Equities/q1_revision
set -o pipefail
printf '%s\n' \
'937de4fb1dc7a58f6ecb8449ad55a515a5dcb28120adbeaa82a4e31c0af5d018  step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py' \
| sha256sum -c - && \
PYTHONDONTWRITEBYTECODE=1 python \
step36i_f20_two_phase_production_wiring_REVIEW_ONLY.py \
--offline-preflight \
2>&1 | tee step36i_f20_offline_preflight_output.txt
```

**Note:** If the hash fails or the file is absent, stop and verify/copy the correct file, then rerun only after hash verification. If PASS, inspect the exact reported interfaces and remaining blockers; do **not** infer independent source approval or turn on live production. This command does not authorize a third read or real scoring.

## 10. Reminders for the next chat

- This continuation pertains to **Indian equities directional forecasting**. The separate *Stock Price Prediction SLR (ACME)* pipeline (Step16U, 485-paper evidence corpus) briefly appeared in a pasted log during this conversation; it is a different project and must not be mixed into this experiment.
- An earlier historical 2021–2026 model run, with small baseline scores, is **not** the clean-v3 2016–2026 held-out experiment and must not be reported as its final test results.
- Prior assistant generated a Springer LaTeX pre-results manuscript package. It is a DRAFT until real-data metrics, discussion and claims are audited. The user asked about Overleaf compilation timeouts separately; that issue does not change Python experiment state.
- Do not create new scientific claims from synthetic results. Synthetic tests establish code integrity and recovery behavior, not forecasting performance.
- Prioritize actual release engineering and scientific defensibility rather than indefinite additional isolated synthetic test scripts.
