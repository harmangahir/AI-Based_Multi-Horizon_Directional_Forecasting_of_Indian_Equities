# F29R6 finalization-only recovery: canonical volatility regime labels

## Verified failure and repair scope

The original Phase A contained 5,300 observations and canonical regime_id/name pairs:
- 0 / Low: 2,524 observations
- 1 / Normal: 1,577 observations
- 2 / High: 1,199 observations

Frozen F29R3 diagnostics erroneously required 1/medium. F29R5's supplementary ML scorer also required medium. These are diagnostic label checks only, not revised input data, targets, checkpoints, model predictions, thresholds or primary tests.

### Minimal derivative changes
- `step36i_f29r6_regime_diagnostics_CORRECTED.py`: map `1:'normal'`, update synthetic fixture label.
- `step36i_f29r6_ml_scorer_CORRECTED.py`: accept `{'low','normal','high'}`.
- `step36i_f29r6_finalizer_CORRECTED.py`: pin the new two derivative SHA256 values and filenames; identify the recovered source/provenance in the inner evaluation manifest. The primary 24-test statistical evaluation logic is unchanged.
- `step36i_f29r6_finalization_only_RECOVERY.py`: a separate post-failure controller. Offline preflight authenticates original signed parent controller, signed records, committed Phase A, all 40 frozen seed artifacts and corrected files; it does NOT open sealed raw test or create outputs. Real `--finalize` (NOT in preflight) requires explicit additional command-line acknowledgment; it runs only on committed arrays/predictions, produces corrected supplementary ML results and publishes standard metrics and a new recovery provenance receipt. The previously signed original controller is never changed or used with an in-place patch.

## Immutable boundaries

Original F29R5 source SHA256: `bc073d6bf565ddbc06fa87007e897eaea16f1cce75cc8940d2b7a0f9a33be220`.
Signed methodological self-review SHA256: `d17e1b503ca43d659fb4e292dbc90a1c2f11780dd686ee7b41e258dc8410ec3a`.
Signed third-read permit SHA256: `d91b818d7d4f3d4e840702d69da92421beb2d141f4b6b300524e84e46027d456`.
Phase A manifest SHA256: `ec87bf367565add93d29fac9b884bcb665db1fd6731ad0318057113439146833`.
Original F29R5 source, F29R3 diagnostic module, F29R5 ML module, F20R3 finalizer and frozen tests are not modified. Historical Phase B import failure evidence must not be removed. The original raw test must never be reread by F29R6.

## Execution sequence

First: install ZIP in `q1_revision`, verify hashes, and execute ONLY

`python step36i_f29r6_finalization_only_RECOVERY.py --offline-preflight`

After inspecting the WSL result, proceed separately to the authorized finalization-only command. Do not invoke original F29R5 `--finalize`, do not delete original failure evidence, and do not retrain or rerun any neural seeds. Do not attempt to retry any failed F29R6 finalization without inspection of staging files and release outputs.

## Limitations

This is a code-level correction made after materialization and 40-seed inference. It must be disclosed as a post-execution diagnostic software correction. Independent review is unavailable; the methodological review remains a single-investigator self-review. No raw test is opened by the new controller; nevertheless finalization will first compute the genuine held-out summary statistics when independently authorized by the owner. Any unfavorable results must be reported honestly.
