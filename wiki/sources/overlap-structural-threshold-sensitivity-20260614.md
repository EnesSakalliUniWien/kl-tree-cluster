---
title: Overlap Structural Threshold Sensitivity 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_structural_threshold_sensitivity.py
  - tests/validation/calibration/overlap/104_test_overlap_structural_threshold_sensitivity.py
  - raw/assets/benchmark-results/overlap_structural_sibling_20260614/threshold_sensitivity/overlap_structural_threshold_sensitivity.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_20260614/threshold_sensitivity/overlap_structural_threshold_recommendations.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/overlap_structural_sibling_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/threshold_sensitivity/overlap_structural_threshold_recommendations.csv
tags:
  - source
  - diagnostics
  - overlap
  - thresholds
  - traversal
---

# Overlap Structural Threshold Sensitivity 2026-06-14

## Summary

`overlap_structural_threshold_sensitivity.py` is a diagnostic-only post-run
analyzer for [[overlap-structural-sibling-panel-20260614]]. It sweeps
homogeneity gain, heterogeneity gain, subspace consensus, and sibling
p-value thresholds to show which quantities control traversal false splits
and signal retention in binary overlap cases.

## Key Points

- The analyzer reads `overlap_structural_sibling_rows.csv` and writes
  `overlap_structural_threshold_sensitivity.csv`,
  `overlap_structural_threshold_recommendations.csv`, and `manifest.json`.
- It separates traversal outcomes into structural homogeneous accepts,
  same-subspace heterogeneous warnings, unrelated-subspace signals, and
  weak/mixed rejections.
- The recommendation table now reports case-replicate stability fields,
  including null case-replicates with false accepts, truth-aligned signal
  case-replicates retained, truth-misaligned signal case-replicates accepted,
  and `threshold_stability_status`.
- The focused four-case run generated `2304` case-level sensitivity rows and
  `288` grid-level recommendation rows.
- The dominant threshold is `homogeneity_gain_min`, not sibling p-value. In
  this run, homogeneity thresholds below `0.01` leave null false accepts, while
  `0.03` blocks all truth-aligned signal. Around `0.02`, the diagnostic grid
  has zero null false accepts, keeps all five truth-aligned signal accepts,
  and keeps zero truth-misaligned signal accepts.
- A three-replicate follow-up generated `172` structural rows, `6912`
  case-level sensitivity rows, and `288` recommendation rows. This weakens the
  single-replicate `0.02` story: at homogeneity threshold `0.02`, null false
  accepts and truth-misaligned accepts stay at zero, but only `12/17`
  truth-aligned signal accepts are retained and only `6/8` truth-aligned signal
  case-replicates are retained. The stability status is therefore
  `threshold_unstable_signal_loss`, not a production candidate.
- At homogeneity threshold `0.01`, the three-replicate run retains more
  truth-aligned signal (`15/17`) and still blocks null false accepts, but keeps
  two truth-misaligned signal accepts. At `0.005`, it retains all truth-aligned
  signal but leaves null false accepts in three null case-replicates.
- Sibling p thresholds `0.001`, `0.01`, and `0.05` produce the same best
  diagnostic rows because accepted split p-values are already tiny. This is
  direct evidence that alpha tuning alone is not the main traversal fix for
  these overlap failures.
- Subspace consensus thresholds `0.15` and `0.25` preserve the best candidates;
  stricter thresholds `0.35` and `0.50` reduce truth-aligned signal retention.
- Heterogeneity thresholds mainly control warning rows. Thresholds near
  `0.01` or `0.02` remove spurious same-subspace heterogeneity warnings in
  this focused run.
- The sweep is evidence for a traversal-aware structural threshold layer, not
  for production promotion. The first three-replicate run shows no fully stable
  threshold in the tested grid: low homogeneity thresholds leave null false
  accepts, while stricter thresholds lose truth-aligned signal.

## Evidence

- `tests/validation/calibration/overlap/104_test_overlap_structural_threshold_sensitivity.py`
  verifies false-null blocking, truth-aligned signal retention,
  heterogeneous-warning separation, recommendation scoring, and output files.
- Verification passed:
  `pytest tests/validation/calibration/overlap/104_test_overlap_structural_threshold_sensitivity.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_structural_threshold_sensitivity.py tests/validation/calibration/overlap/104_test_overlap_structural_threshold_sensitivity.py`.

## Links

- [[overlap-structural-sibling-panel-20260614]]
- [[selected-family-traversal-panel-20260614]]
- [[open-mathematical-questions]]
