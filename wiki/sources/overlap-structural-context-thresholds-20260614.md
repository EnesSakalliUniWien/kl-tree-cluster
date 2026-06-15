---
title: Overlap Structural Context Thresholds 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - benchmarks/diagnostics/calibration/overlap_structural_context_thresholds.py
  - tests/validation/105_test_overlap_structural_context_thresholds.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_thresholds/overlap_structural_context_threshold_sensitivity.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_thresholds/overlap_structural_context_threshold_summary.csv
tags:
  - source
  - diagnostics
  - overlap
  - thresholds
  - traversal
---

# Overlap Structural Context Thresholds 2026-06-14

## Summary

`overlap_structural_context_thresholds.py` bins overlap structural traversal
rows by depth, parent size, barycentric balance, and subspace consensus, then
evaluates homogeneity-gain, subspace-consensus, and sibling-p thresholds inside
each context. It is diagnostic-only and does not change production traversal.

## Key Points

- The context analyzer writes
  `overlap_structural_context_threshold_sensitivity.csv`,
  `overlap_structural_context_threshold_summary.csv`, and `manifest.json`.
- The three-replicate overlap run produced `864` context sensitivity rows and
  `12` context summary rows.
- Depth is informative. `deep_3_plus` is a diagnostic candidate with
  homogeneity threshold `0.005`, subspace threshold `0.15`, zero null structural
  accepts, and `7/7` truth-aligned signal accepts retained. `shallow_1_2`
  needs a stricter threshold around `0.015` to avoid null accepts, but then
  retains only `8/10` truth-aligned signal accepts.
- Parent size is also informative. `large_parent_ge300` at the best diagnostic
  row keeps truth-misaligned signal, while `medium_parent_150_299` has signal
  loss. Small parents in this focused run have no truth-aligned signal support.
- Barycentric balance separates some instability but does not solve it:
  balanced contexts retain `13/15` truth-aligned signal accepts at the best
  row, while moderate-balance contexts retain only `1/2`.
- Subspace-consensus bins show that both `high_consensus_ge0.5` and
  `low_consensus_lt0.25` can be diagnostic candidates in this small run, but
  the low-consensus candidate has only two truth-aligned signal accepts and
  needs broader validation before any interpretation.
- Sibling p threshold remains uninformative inside the best context rows; the
  selected rows choose `0.001`, but earlier sweeps show thresholds up to
  `0.05` do not explain the overlap failure.
- The method implication is a traversal-context structural rule, not one
  global homogeneity-gain cutoff: deep/internal contexts can be more
  permissive, while shallow/root or large-parent contexts need separate guards
  or multi-scale warning output.

## Evidence

- `tests/validation/105_test_overlap_structural_context_thresholds.py` verifies
  context binning, context-specific threshold summaries, and output writing.
- Verification passed:
  `pytest tests/validation/105_test_overlap_structural_context_thresholds.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_structural_context_thresholds.py tests/validation/105_test_overlap_structural_context_thresholds.py`.

## Links

- [[overlap-structural-sibling-panel-20260614]]
- [[overlap-structural-threshold-sensitivity-20260614]]
- [[open-mathematical-questions]]
