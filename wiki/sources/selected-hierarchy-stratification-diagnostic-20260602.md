---
title: Selected Hierarchy Stratification Diagnostic 2026-06-02
type: source
status: reviewed
updated: 2026-06-02
sources:
  - benchmarks/diagnostics/calibration/selected_hierarchy_stratification_diagnostic.py
  - raw/assets/benchmark-results/selected_hierarchy_stratification_20260602_500/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_stratification_20260602_500/case_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_stratification_20260602_500/strata_by_parent_size.csv
  - raw/assets/benchmark-results/selected_hierarchy_stratification_20260602_500/strata_by_depth.csv
tags:
  - source
  - calibration
  - selection
  - stratification
---

# Selected Hierarchy Stratification Diagnostic 2026-06-02

## Summary

This diagnostic describes how selected-hierarchy null scale varies across
parent depth and parent-size strata. It rebuilds the hierarchy inside each null
replicate and keeps all selected, edge-path-open sibling records. The output is
descriptive only: depth and parent-size strata describe heterogeneity and must
not be used as calibration fallback or borrowing rules.

The regenerated 2026-06-02 table also records the selected-ratio law
\(R=W/(a\nu)\), raw statistic quantiles, reference expectations, and the
standard projected-Wald rejection rate inside each stratum. These columns
characterize the conditional selected-null phenomenon; they are not production
calibration estimates.

## Key Points

- The run used 500 null replicates for `gauss_null_large`,
  `gauss_clear_medium`, `binary_low_noise_4c`, and `cat_clear_3cat_4c`.
- Selected-record rates differed strongly by case: about `0.88` to `0.90` for
  the two Gaussian cases, `0.10` for the binary low-noise case, and `0.40` for
  the categorical clear case.
- Parent-size strata are informative. Small selected parent nodes often have
  higher \(c\) than root-like selected nodes.
- The selected-ratio law is not close to the unselected projected-Wald
  reference. Among reliable parent-size rows with at least 100 matching
  simulations, \(R\) has q95 values from about `38` to `103`.
- The standard projected-Wald reference rejects almost all selected-null
  records in most reliable strata. The recorded rejection rate at
  `SIBLING_ALPHA = 0.01` is `1.0` in most rows and never below about `0.70`
  among the reliable rows.
- Reliable parent-size rows with at least 100 matching simulations are:

```text
case/projection/parent-size stratum          c-hat
binary_low_noise_4c k=1 root_0.75_1           18.2
binary_low_noise_4c k=1 small_0_0.25          67.5
binary_low_noise_4c k=2 root_0.75_1           23.5
binary_low_noise_4c k=2 small_0_0.25          50.7
cat_clear_3cat_4c k=1 small_0_0.25            56.6
cat_clear_3cat_4c k=2 large_0.5_0.75          48.6
cat_clear_3cat_4c k=2 medium_0.25_0.5         60.7
cat_clear_3cat_4c k=2 root_0.75_1             34.3
cat_clear_3cat_4c k=2 small_0_0.25            52.4
gauss_clear_medium k=1 small_0_0.25           31.0
gauss_clear_medium k=2 large_0.5_0.75         36.3
gauss_clear_medium k=2 medium_0.25_0.5        35.5
gauss_clear_medium k=2 root_0.75_1            35.5
gauss_clear_medium k=2 small_0_0.25           28.2
gauss_null_large k=1 small_0_0.25             64.6
gauss_null_large k=2 large_0.5_0.75           57.5
gauss_null_large k=2 medium_0.25_0.5          64.2
gauss_null_large k=2 root_0.75_1              47.7
gauss_null_large k=2 small_0_0.25             57.9
```

- Depth strata are highly confounded with parent size. Deep strata are often
  small-parent strata, and exact depth matching can make support sparse even
  when selected records exist abundantly at nearby depths.
- The selected-hierarchy scale remains in the tens across most supported
  strata. This supports the geometric explanation that same-data hierarchy
  selection creates selected high-contrast null records.

## Evidence

- `benchmarks/diagnostics/calibration/selected_hierarchy_stratification_diagnostic.py`
  implements the diagnostic-only grouping by parent depth and parent-size bin.
- `raw/assets/benchmark-results/selected_hierarchy_stratification_20260602_500/case_summary.csv`
  records selected-record rates and edge rejection rates by case.
- `raw/assets/benchmark-results/selected_hierarchy_stratification_20260602_500/strata_by_parent_size.csv`
  records parent-size-bin summaries, including \(n\), feature dimension,
  projection dimension, \(R\)-law quantiles, statistic quantiles, reference
  expectations, and raw projected-Wald rejection rates.
- `raw/assets/benchmark-results/selected_hierarchy_stratification_20260602_500/strata_by_depth.csv`
  records the same selected-law summaries by exact depth.

## Links

- [[selected-hierarchy-null-support-contract]]
- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-null-audit-20260601]]
- [[selected-hierarchy-external-calibration-contract-20260602]]
- [[open-mathematical-questions]]
