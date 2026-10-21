# F29R7 post-commit figure style extensions

This package is **source code and LaTeX only**. It cannot contain the actual new plot PDFs because the authenticated evaluation CSVs remain in the user's WSL environment. The plotting code intentionally fails rather than substituting example/model-generated results.

## Package contents

- `plot_f29r7_verified_scientific_v4.py`: the unchanged original SHA-256-verified F29R7 CSV reader, included as a dependency.
- `plot_f29r7_legacy_style_from_verified_results.py`: supplementary figure styles, sourced from the **current F29R7 data only**.
- `F29R7_Final_Figure_Plan_and_Research_Guards.md`: main vs supplementary figure blueprint and unsupported old-figure disclosures.
- `F29R7_Style_Supplementary.tex`: **standalone supplementary PDF figure source**, to compile after generating all seven new PDFs.
- `main_F29R7_WITH_SUPPLEMENT_POINTERS_DRAFT.tex`: optional revised main-file derivative that inserts prose references to S1–S7. It retains the seven PDF figure references from the previous accepted manuscript draft; do not replace your latest hand-edited main source without diffing changes.

## First WSL action — verify only

Save and unzip the package in `q1_revision`. Keep the two `.py` files in this directory. Then:

```bash
cd ~/projects/AI-Based_Multi-Horizon_Directional_Forecasting_of_Indian_Equities/q1_revision
python plot_f29r7_legacy_style_from_verified_results.py --verify-only
```

Expected: `PASS: F29R7 signed receipt, committed evaluation manifest, 23 SHA-256 verified results` and `VERIFY ONLY PASS; no figures or files written`. If it fails, do **not** bypass verification; provide complete error output.

## Second WSL action — actual figures

Only after preflight passes:

```bash
python plot_f29r7_legacy_style_from_verified_results.py \
    --plots all --formats png pdf --dpi 600
```

Expected outputs: 14 files (7 `.png`, 7 `.pdf`) in `F29R7_Style_Recreations_Supplementary/`, plus a provenance JSON with all output hashes. Do not append `--overwrite` on the first run. For new revisions choose a fresh `--output-dir`.

## Manuscript

Keep `F29R7_Style_Supplementary.tex` alongside the generated `F29R7_Style_Recreations_Supplementary/` folder and compile with `pdflatex` twice. It is a **separate supplementary manuscript**, not an instruction to duplicate 7 figures into the existing main manuscript. Captions distinguish exploratory analyses and uncomputed ROC-AUC uncertainty.

The optional `main_F29R7_WITH_SUPPLEMENT_POINTERS_DRAFT.tex` inserts reference text at §§5.1, 6.1–6.4, and 6.5. It is derived from the preceding complete 7-PDF figure source (`main_F29R7_7_PDF_figures.tex`); if you have edited the latest main file since that version, merge only the supplementary-reference sentences instead of overwriting those edits.

## Failure policy

Neither plotting source opens the sealed raw test, reimports seed checkpoints, reruns neural inference, refits any model, changes classification thresholds, or recalculates bootstrap samples. The signed publication receipt, committed F29R6 manifest and all 23 artifacts must authenticate. Unsupported rejected-paper analysis and trading profits are not reproduced.
