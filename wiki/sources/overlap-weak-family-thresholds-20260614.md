---
title: Overlap Weak Family Thresholds 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_weak_family_thresholds.py
  - tests/validation/calibration/overlap/109_test_overlap_weak_family_thresholds.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/weak_family_thresholds/overlap_weak_family_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/weak_family_thresholds/overlap_weak_family_metric_separability.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/weak_family_thresholds/overlap_weak_family_threshold_scan.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/weak_family_thresholds/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - thresholds
---

# Overlap Weak Family Thresholds 2026-06-14

## Summary

`overlap_weak_family_thresholds.py` aggregates
`unstable_weak_homogeneity_zone` rows by selected traversal family:
`case_id`, `data_role`, and `replicate`. It tests whether family-level
statistics, such as selected-family minimum p-value or maximum homogeneity
gain, separate weak truth-aligned signal families from selected-null and
truth-misaligned families. It is diagnostic-only.

## Key Points

- The runner writes `overlap_weak_family_rows.csv`,
  `overlap_weak_family_metric_separability.csv`,
  `overlap_weak_family_threshold_scan.csv`, and `manifest.json`.
- The three-replicate overlap run produces `19` weak-family rows:
  `9` `null_like_family`, `5` `signal_truth_aligned_family`, and `5`
  `signal_truth_misaligned_family`.
- Family grouping must include `data_role`. Grouping only by case and
  replicate incorrectly mixes selected-null and signal rows from different
  generated datasets.
- Against selected-null families alone, selected-family p-value is perfectly
  separating in this focused run: `min_sibling_p_value` and
  `neg_log10_min_sibling_p_value` both have `AUC = 1.0`, with zero-null
  retention `5/5`.
- That apparent success does not survive the signal-side selected-family
  mixture. Against selected-null plus truth-misaligned families, the same
  p-value family metrics have `AUC = 0.871429` but zero-negative retention
  `0/5`, because truth-misaligned signal families can have even more extreme
  selected-family p-values.
- The best zero-negative family-level retention against the combined negative
  family is still only `2/5`, achieved by
  `min_continuous_homogeneity_threshold <= 0.008966457` or `max_depth >= 2`.
  `max_homogeneity_gain_min >= 0.01485809` retains only `1/5`.
- Therefore family aggregation confirms the earlier row-wise conclusion:
  alpha-like p-value thresholds separate selected-null from weak signal, but
  not truth-aligned weak signal from truth-misaligned selected-family signal.
  The unresolved object is a selected-family mixture law or an explicit
  unstable-zone reporting rule.

## Evidence

- `tests/validation/calibration/overlap/109_test_overlap_weak_family_thresholds.py` verifies
  family grouping by `(case_id, data_role, replicate)`, positive-family
  classification when any weak truth-aligned row is present, zero-negative
  family threshold scans, and output writing.
- Verification passed:
  `pytest tests/validation/calibration/overlap/109_test_overlap_weak_family_thresholds.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_weak_family_thresholds.py tests/validation/calibration/overlap/109_test_overlap_weak_family_thresholds.py`.

## Links

- [[overlap-structural-decision-zones-20260614]]
- [[overlap-weak-zone-separability-20260614]]
- [[overlap-structural-continuous-rule-20260614]]
- [[open-mathematical-questions]]
