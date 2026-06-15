---
title: Overlap Weak Zone Separability 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - benchmarks/diagnostics/calibration/overlap_weak_zone_separability.py
  - tests/validation/108_test_overlap_weak_zone_separability.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/weak_zone_separability/overlap_weak_zone_metric_separability.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/weak_zone_separability/overlap_weak_zone_threshold_scan.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/weak_zone_separability/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - thresholds
---

# Overlap Weak Zone Separability 2026-06-14

## Summary

`overlap_weak_zone_separability.py` analyzes only the
`unstable_weak_homogeneity_zone` from
[[overlap-structural-decision-zones-20260614]]. It asks whether any scalar
structural metric can separate weak truth-aligned signal rows from selected-null
or truth-misaligned rows. The result is diagnostic-only: scalar thresholds can
triage the weak zone, but they do not solve it.

## Key Points

- The runner writes `overlap_weak_zone_metric_separability.csv`,
  `overlap_weak_zone_threshold_scan.csv`, and `manifest.json`.
- The unstable weak zone has `5` truth-aligned signal rows and `43` combined
  negative rows (`23` selected-null plus `20` truth-misaligned signal).
- Against selected null alone, `homogeneity_gain_min` is nearly separable
  (`AUC = 0.991304`) and a zero-null threshold retains `4/5` weak truth-aligned
  rows.
- Against truth-misaligned signal alone, context homogeneity margin has high
  rank separation (`AUC = 0.900`), but its zero-negative threshold retains
  `0/5` weak truth-aligned rows. `homogeneity_gain_min` retains only `1/5`.
- Against the combined negative family, the best rank metric is
  `context_homogeneity_margin` (`AUC = 0.939535`), but its zero-negative
  threshold retains `0/5` weak truth-aligned rows.
- The best zero-negative retention against the combined negative family comes
  from `continuous_homogeneity_threshold <= 0.008264365`, retaining only `2/5`
  weak truth-aligned rows. `homogeneity_gain_min >= 0.01485809` retains `1/5`.
- Therefore no tested scalar structural threshold can safely recover all weak
  truth-aligned rows while blocking selected-null and truth-misaligned rows.
  The weak zone is a selected-family mixture, not a threshold-tuning problem.
- Method implication: stable same-subspace accepts may be reported as stable
  internal regions in this focused run, but weak-homogeneity rows need either
  explicit unstable multi-scale reporting or a selected-family null law that
  conditions on the selected weak family.

## Evidence

- `tests/validation/108_test_overlap_weak_zone_separability.py` verifies
  pairwise rank-AUC with ties, partial zero-negative threshold retention, and
  output writing.
- Verification passed:
  `pytest tests/validation/108_test_overlap_weak_zone_separability.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_weak_zone_separability.py tests/validation/108_test_overlap_weak_zone_separability.py`.

## Links

- [[overlap-structural-decision-zones-20260614]]
- [[overlap-structural-continuous-rule-20260614]]
- [[overlap-structural-threshold-sensitivity-20260614]]
- [[open-mathematical-questions]]
