# Step 36I–F29-R2 — Source-Corrected Single-Investigator Release Candidate

**Prepared:** 2026-10-15. **Status:** SIGNATURE-GATED CANDIDATE; NOT AUTHORIZED TO READ THE FINAL TEST.

## Why a versioned correction was necessary

F29-R1 passed `--offline-preflight`, but two important paths were not exercised by that preflight:

1. On a production-style CLI (`--phase-a`/`--phase-b`), it imported frozen F23 *before* entering the argv-isolation context. F23's import-time guard explicitly refuses those flags, so F29-R1 would fail before it could authenticate the method/read permits.
2. Frozen F20's new Phase A package context said `independent_signoff_required: True`, inconsistent with the accurately disclosed single-investigator methodological self-review established for F29.

**F29-R2** fixes the import order and binds the Phase A package/receipt to the SHA-256 of the real, independently cryptographically authenticated *owner self-review payload*. **F20-R1** is a versioned copy of frozen F20 with provenance correction. Original F20, F28, F29 and F29-R1 files stay unchanged.

## Exactly what changed

- Adds `step36i_f20r1_owner_review_provenance_REVIEW_ONLY.py`, requiring the authenticated method-payload SHA and writing `methodological_review_type = SOLE_INVESTIGATOR_SELF_REVIEW_NOT_INDEPENDENT`, `external_independent_review_performed = false`, and the method digest into the Phase A package manifest.
- Adds `step36i_f29r2_source_corrected_signature_gated_CANDIDATE.py` with source-hash binding to F20-R1, provenance-verification in Phase A receipts, and correct frozen-helper argv isolation before importing F23 or F20 under a production-style CLI.
- Adds a separate `--i-authorize-irreversible-read` required for Phase A **in addition to** a legitimate owner-signed scientific self-review AND a later, distinct owner-signed third-read permit. No permit or self-review signature is generated, inferred, or included.
- Keeps the owner's pinned public keyring fingerprint `2de1349896cf0012c5a88ff7fad09de27bd1ac2131d99bace396ce659819f4b7`.
- Adds a real WSL preflight check for owner-only release directories and all seven reserved output parents **without** opening test data.
- The main controller contains signature-gated live API code (no blanket `LIVE_RELEASE_ENABLED=False`), **but the mandatory signed attestations and separate permit remain absent**. Possessing the code or running `--offline-preflight` confers no authorization to read the sealed raw test.

## Remaining release boundaries and known limitations

- There is no independent human methodological review; this is explicitly **single-investigator self-review**. Cryptographic signatures authenticate the owner key, not scientific independence or quality.
- Original F17 inner manifests still carry historical `requires_independent_source_review` review-only metadata. Do not cite these flags as completed independent review; the outer versioned F20-R1 package and receipts describe the actual self-review truthfully.
- The complete live call path has **not** been executed in WSL, and no production results exist. Portable isolated journal/lock/signature fixtures and Python compilation passed locally; WSL real-input-free preflight is pending.
- The inherited frozen recovery helper retains a constant-path use of regime thresholds; this leaves a residual same-user/upstream path-binding assumption. Private-directory controls and pre/post rehash checks mitigate but do not eliminate an adversarial same-user or privileged process.
- A fault between reservation, one authorized raw read, durable raw snapshot commit and materialization can require manual recovery; retrying the sealed original source is never automatic. The authorized raw snapshot is confidential held-out data and must be specifically accepted under a retention policy.
- Baseline adequacy, forecasting diagnostics, evaluation exclusions, bootstrap assumptions and manuscript claims must be examined scientifically before signing the method attestation. Historical reads are **2**, completed held-out model scorings **0**, and the authorized new raw-test read **0**.
- The F29R2 owner-method attestation must be bound to the **final SHA-256 of the actual executable release source**. Do not sign against F29-R1, then use F29-R2; no source file may be edited after signing.

## Permitted user action now

1. Save the two new Python files in your existing `q1_revision` directory, without replacing earlier scripts.
2. Verify both SHA-256 values and run only the F29-R2 `--offline-preflight`.
3. Paste the full output. We will address *actual* failures and then complete a human-readable scientific self-review and accountable signing decision, separately from the later third-read authorization.

**Do not run `--phase-a`, `--phase-b` or `--finalize` and do not create a third-read permit yet.**
