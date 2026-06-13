---
title: Selected Sibling LRT Diagnostic 2026-06-12
type: source
status: reviewed
updated: 2026-06-12
sources:
  - benchmarks/diagnostics/calibration/selected_sibling_lrt_diagnostic.py
  - tests/validation/87_test_selected_sibling_lrt_diagnostic.py
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/29_selected_sibling_lrt_diagnostic_20260612/selected_sibling_lrt_node_panel.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/29_selected_sibling_lrt_diagnostic_20260612/selected_sibling_lrt_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/29_selected_sibling_lrt_diagnostic_20260612/selected_sibling_lrt_run_summary.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/29_selected_sibling_lrt_diagnostic_20260612/selected_sibling_lrt_vs_projected_wald.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/29_selected_sibling_lrt_diagnostic_20260612/selected_sibling_lrt_support_boxes.png
tags:
  - source
  - calibration
  - sibling
  - diagnostics
  - diffusion
---

# Selected Sibling LRT Diagnostic 2026-06-12

## Summary

This diagnostic adds a Bernoulli likelihood-ratio-style sibling deviance beside
the existing projected-Wald sibling statistic for whole-space fixed-diffusion
KL trees. It is diagnostic-only: the nominal fixed-pair chi-square tail in the
output is not a selected-tree calibration rule.

## Key Points

- The diagnostic was run on the Julia combined GO matrix for three fixed
  diffusion regimes: `k=15,t=3`, `k=15,t=5`, and fragmented `k=30,t=3`, all
  with `30` diffusion components.
- The readable regimes reproduced the earlier cluster counts: `54` clusters
  for `k=15,t=3` and `45` clusters for `k=15,t=5`. The fragmented diagnostic
  regime produced `619` clusters and singleton fraction `0.941842`.
- Median Bernoulli deviance per changed feature was nearly unchanged across
  regimes: `2.001041` for `k=15,t=3`, `1.970693` for `k=15,t=5`, and
  `2.010217` for `k=30,t=3`.
- Median projected-Wald selected ratio changed sharply: `1.625297` for
  `k=15,t=3`, `1.534204` for `k=15,t=5`, and `18.603097` for `k=30,t=3`.
- The fragmented `k=30,t=3` regime had `671` selected-nonnull-like sibling
  rows and only `31` strict-null-like rows, while `k=15,t=3` had `474`
  selected-nonnull-like rows and `228` strict-null-like rows.
- In selected-nonnull-like rows, sibling rejection was `0.056962` for
  `k=15,t=3`, `0.033333` for `k=15,t=5`, and `0.794337` for `k=30,t=3`.
- Spearman coupling between Bernoulli deviance per changed feature and
  projected-Wald selected ratio was modest in all runs: about `0.284`,
  `0.383`, and `0.257` respectively.

## Evidence

- `benchmarks/diagnostics/calibration/selected_sibling_lrt_diagnostic.py`
  implements the Bernoulli sibling deviance, fixed-diffusion rerun, node-panel
  extraction, summary table, and plots.
- `tests/validation/87_test_selected_sibling_lrt_diagnostic.py` verifies the
  deviance calculation, support-label extraction from edge-path annotations,
  and grid parsing.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/29_selected_sibling_lrt_diagnostic_20260612/selected_sibling_lrt_summary.csv`
  records the run-level summary.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/29_selected_sibling_lrt_diagnostic_20260612/selected_sibling_lrt_node_panel.csv`
  records `702` sibling rows per diffusion regime.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/29_selected_sibling_lrt_diagnostic_20260612/selected_sibling_lrt_vs_projected_wald.png`
  visualizes the separation between projected-Wald amplification and Bernoulli
  deviance.

## Links

- [[null-edge-sibling-calibration-enhancement-plan]]
- [[mixed-null-signal-geometry-validation-20260606]]
- [[adaptive-cosine-kak-benchmark-probe-20260605]]
- [[selected-pca-projected-wald-validation]]
