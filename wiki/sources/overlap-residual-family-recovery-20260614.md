---
title: Overlap Residual Family Recovery 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - benchmarks/diagnostics/calibration/overlap_residual_family_recovery.py
  - tests/validation/114_test_overlap_residual_family_recovery.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/residual_family_recovery/overlap_residual_family_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/residual_family_recovery/overlap_residual_family_metric_separability.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/residual_family_recovery/overlap_residual_family_threshold_scan.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/residual_family_recovery/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - selected-family
---

# Overlap Residual Family Recovery 2026-06-14

## Summary

`overlap_residual_family_recovery.py` analyzes only rows left in
`weak_unstable_multiscale_zone` after the continuous structural rule and
fragment-risk guard. It aggregates those rows by selected family
`(case_id, data_role, replicate)` and tests whether non-oracle family metrics
can separate residual truth-recovery families from selected-null and
non-recovery families. Oracle truth labels are used only for evaluation.

## Key Points

- The runner writes `overlap_residual_family_rows.csv`,
  `overlap_residual_family_metric_separability.csv`,
  `overlap_residual_family_threshold_scan.csv`, and `manifest.json`.
- Residual selected-family roles are: `8` selected-null families, `5`
  truth-recovery families, and `3` non-recovery signal families.
- Against selected-null families alone, sibling p-value extremeness separates
  residual recovery families: `residual_min_sibling_p_value` and
  `residual_neg_log10_min_sibling_p_value` both have `AUC = 1.0`, with a
  zero-null threshold retaining `5/5` recovery families.
- Against residual non-recovery signal families, p-value extremeness fails:
  the best p-value direction has `AUC = 0.666667` and zero-negative retention
  `0/5`.
- Against selected-null plus non-recovery families, p-value extremeness remains
  high-rank (`AUC = 0.909091`) but has zero-negative retention `0/5`.
- The best residual non-recovery separator is structural, not p-value based:
  `residual_min_fragment_risk_proxy_score` has `AUC = 0.866667` for recovery
  versus non-recovery, but its zero-negative threshold retains only `3/5`
  recovery families.
- `residual_max_homogeneity_gain_min` has `AUC = 0.836364` against the full
  null-or-nonrecovery set and zero-negative retention `1/5`. Thus it is useful
  context, not a production threshold.
- Method implication: after fragment blocking, selected-family p-value
  extremeness can distinguish residual recovery from selected null, but not
  from non-recovery selected signal. The remaining law must condition on
  structural recovery, not just selected-family extremeness.

## Evidence

- `tests/validation/114_test_overlap_residual_family_recovery.py` verifies
  residual family aggregation after fragment blocking, metric summary/AUC
  output, and file writing.
- Verification passed:
  `pytest tests/validation/114_test_overlap_residual_family_recovery.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_residual_family_recovery.py tests/validation/114_test_overlap_residual_family_recovery.py`.

## Links

- [[overlap-diagnostic-traversal-policy-20260614]]
- [[overlap-weak-family-thresholds-20260614]]
- [[overlap-fragment-risk-guard-20260614]]
- [[open-mathematical-questions]]
