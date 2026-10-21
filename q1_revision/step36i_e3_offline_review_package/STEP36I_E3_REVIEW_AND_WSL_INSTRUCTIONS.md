# Step 36I-E3 — Isolated amended evaluation runner (offline-only review build)

## Evidence reviewed

The 18-member `step36i_e2_source_review_bundle.zip` passed SHA256 verification (`9a425d223d2c4c301aa8fb6df69c58b2f2047e2d719c8eee5bbf3b713b690851`). Original frozen Step-35: `6f715055fa30759fd9aa9b31655759bbd4dca25398f1d6d2cd39be5bf3bed1d7`; recovered Step-36 source: `6de04ba7e167449101e2e16bab47ca4ce088f11284c4cbc93b61a6cfbb979f12`.

Two prior raw-test opens were recorded; no held-out model metrics were computed. Both unsuccessful historical execution logs are referenced by immutable hashes in the deviation draft. Original Step-35 specifies 6,480 target-eligible stock-date rows (324 sessions); the independently reconstructed proposed cohort is 5,300 sequence-eligible rows (265 sessions), excluding 1,180 (18.21%) due to the 60-session continuity requirement. The two proposed periods are 48 and 217 sessions, with 20 stocks per date.

The bundle documents successful **synthetic-only** 40-checkpoint inference and 24-threshold scoring checks. Synthetic inference never establishes real held-out performance.

## What this E3 artifact implements

- SHA256 verification of the E2 source review bundle and primary Step-35/Step-36/candidate-key and synthetic reports.
- Fail-closed review of the historical two-read access marker and unapproved post-freeze protocol state.
- Balanced 5,300-row / 265-session cohort verification and cross-segment continuity checks.
- Pure `select_preloaded_endpoints` key-whitelist operation, requiring original 6,480-row structural eligibility, with strict duplicate/missing-key rejection.
- Verification of 40 frozen model/seed plan entries, exact architecture configurations, physical batches, no shuffle/gradients, 24 fixed thresholds, and the unchanged Step-35 bootstrap/FDR contract.
- Optional independent reconstruction of the candidate key set from the live **nonsealed** Step-10 membership ledger, with positive and adversarial selector checks.
- Isolated names for future amended output paths, checked for existing artifacts without creating or overwriting them.
- A separate draft JSON audit report in a new directory, created only when `--report-dir` is supplied.
- Hard denial of `--execute`, `--worker`, `--materialize`, and other production modes.

**This is not yet an approved or complete production evaluation runner.** It does not contain a raw sealed-test loader, implement 40-checkpoint real-data inference/persistence, or compute held-out metrics. These functions explicitly raise `PermissionError`. E3 is a reviewed implementation scaffold, not a completed protocol amendment.

## One WSL action to run next

Download `step36i_e3_isolated_amended_runner_DRAFT.py` into the existing `q1_revision` directory. Leave your source ZIP in that directory. In the `Harman_GPU` environment, run:

```bash
cd ~/projects/AI-Based_Multi-Horizon_Directional_Forecasting_of_Indian_Equities/q1_revision
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 python step36i_e3_isolated_amended_runner_DRAFT.py \
  --offline-preflight \
  --bundle step36i_e2_source_review_bundle.zip \
  --project-root "$PWD" \
  --report-dir ../stock_q1_revision_project/experiments/q1_revision_v3_clean_2016_2026/audits/step36i_e3_isolated_runner_review_draft \
  2>&1 | tee step36i_e3_isolated_runner_review_output.txt
```

If that report directory already exists, the command will stop to preserve evidence. Do **not** delete it or overwrite files; share the status. Do **not** run the original Step 36 `--execute`.

## Review checkpoints before any real scoring

1. Independently review the post-freeze structural cohort exclusion and assess selection/representativeness bias. Retain the 6,480-row original plan and 5,300-row corrected plan as distinct provenance.
2. Independently review the segment-stratified circular moving-block bootstrap, particularly L=40 in a 48-session segment, and define confirmatory versus exploratory reporting.
3. Complete and independently test a separate, explicitly authorized production implementation; preserve original failure evidence, historical access journal, hashes, ordered physical batches (notably TimesNet), and avoid all original Step-36 fixed output paths.
4. Make a **separate, explicit scientific decision** about whether the post-freeze amended evaluation and a third physical raw-test read are acceptable. This draft grants neither.
5. After an authorized actual evaluation, validate class balance, 24 held-out BA/AUC/other results, bootstrap p-values, FDR q-values and manuscript tables. No such real-data results exist as of this review.
