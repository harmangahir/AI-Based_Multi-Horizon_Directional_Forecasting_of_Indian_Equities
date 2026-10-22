F29R7 — FINAL LANDSCAPE HORIZONTAL 1 x 3 FIGURE PLOTS
======================================================
Includes two full original-script based Python generators; both retain
SHA-256 authentication of the already frozen and committed aggregates.
No training, model inference, labels, test arrays, or bootstrap resampling.

Copy BOTH *.py files into your q1_revision WSL working directory.

Figure 7 — Bootstrap (all 72 frozen model/horizon/block rows):
  python plot_F29R7_Figure7_HORIZONTAL_FINAL.py --plots bootstrap \
      --formats pdf png --width 11.5 --font-size 12 --dpi 600 \
      --output-dir F29R7_Figure7_HORIZONTAL_FINAL

Output: F29R7_Figure7_HORIZONTAL_FINAL/
  fig17_bootstrap_block_sensitivity_HORIZONTAL.pdf
  fig17_bootstrap_block_sensitivity_HORIZONTAL.png

Figures 8 and 9 — BA and AUC (33 frozen aggregate metric rows):
  python plot_F29R7_Figures8_9_HORIZONTAL_FINAL.py --figures both \
      --formats pdf png --width 11.5 --font-size 12 --dpi 600 \
      --output-dir F29R7_Figures8_9_HORIZONTAL_FINAL

Outputs: F29R7_Figures8_9_HORIZONTAL_FINAL/
  fig18_all_models_BA_HORIZONTAL.pdf
  fig18_all_models_BA_HORIZONTAL.png
  fig19_all_models_ROCAUC_HORIZONTAL.pdf
  fig19_all_models_ROCAUC_HORIZONTAL.png

Both scripts also generate provenance JSON SHA-256 reports.

IMPORTANT: Old F29R7_Figure7_Q1_Refined contains the previous PORTRAIT PDF.
Do NOT reopen that folder. View the NEW files ending in _HORIZONTAL.pdf.

As these figures are 11.5-inch vector-wide, scale them to the final article
text width with appropriate figure placement. Inspect actual font size after
LaTeX compilation. If a single-column journal makes the horizontal type too
small, request a full-width landscape or two-column span as journal permits.

Figure placement: use existing figure labels/captions when replacing PDF filenames.
No scientific results are changed by this graphics-only update.
