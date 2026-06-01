---
title: Selected Hierarchy Null Audit 2026-06-01
type: source
status: reviewed
updated: 2026-06-01
sources:
  - benchmarks/diagnostics/calibration/selected_hierarchy_null_audit.py
  - raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601/selected_hierarchy_null_audit_summary.csv
tags:
  - source
  - calibration
  - selection
  - null
---

# Selected Hierarchy Null Audit 2026-06-01

## Summary

This audit estimates the same-data selected-hierarchy sibling null by
regenerating null feature matrices, rebuilding the hierarchy inside every
replicate, rerunning the edge gate, and collecting selected focal sibling
statistics. It keeps cross-fitting out of the candidate method and directly
targets the selected-hierarchy inference problem.

## Key Points

- The audit is diagnostic-only and does not install a production calibration
  fallback.
- The implemented null generator supports Bernoulli and categorical feature
  contracts. Continuous selected-hierarchy null simulation remains unsupported
  until a validated covariance generator exists.
- In the four representative cases, selected-hierarchy inflation estimates are
  much larger than fixed-subspace or local-edge-selection diagnostics:
  approximately `58.5` for `gauss_null_large`, `28.8` for
  `binary_low_noise_4c`, `50.5` for `cat_clear_3cat_4c`, and `30.6` for
  `gauss_clear_medium`.
- The selected-hierarchy mean-scaled chi-square p-value blocks the null case
  (`gauss_null_large`) and still rejects the three signal examples.
- This supports the interpretation that the large correction is caused by
  same-data hierarchy selection, not by the projected-Wald kernel alone.

## Evidence

- `benchmarks/diagnostics/calibration/selected_hierarchy_null_audit.py`
  implements the selected same-data hierarchy simulation and context matching.
- `raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601/selected_hierarchy_null_audit_summary.csv`
  records observed target statistics, selected-hierarchy null counts,
  selected-hierarchy \(c\), mean-scaled p-values, empirical tail p-values, and
  blocking flags.
- `raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601/manifest.json`
  records the case names, seed, replicate count, target mode, and context
  matching rule.

## Links

- [[edge-selection-null-audit-20260601]]
- [[feature-split-selection-audit-20260601]]
- [[oracle-gate-path-diagnostic]]
- [[open-mathematical-questions]]
