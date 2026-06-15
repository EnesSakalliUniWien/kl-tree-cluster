---
title: Regularized Wald Statistic Panel 2026-06-13
type: source
status: reviewed
updated: 2026-06-13
sources:
  - benchmarks/diagnostics/calibration/regularized_wald_statistic_panel.py
  - tests/validation/97_test_regularized_wald_statistic_panel.py
  - benchmarks/diagnostics/calibration/differential_statistic_validity_panel.py
  - benchmarks/diagnostics/calibration/production_admissibility_contract.py
tags:
  - source
  - diagnostics
  - calibration
  - statistics
---

# Regularized Wald Statistic Panel 2026-06-13

## Summary

`regularized_wald_statistic_panel.py` compares plug-in, Jeffreys-smoothed,
Dirichlet-smoothed, and root-shrunk sibling projected-Wald statistics under
selected-edge null runs. It tests whether smoothing removes the Fisher/Wald
boundary instability detected by the differential statistic-validity panel.

## Key Points

- The panel is diagnostic-only and does not change production KL-TE statistics.
- Fixed-tree rows are evaluated before selected-tree rows because a statistic
  that fails fixed-tree null calibration cannot be rescued by selected-tree
  conditioning.
- On a five-replicate `binary_2clusters` fixed-tree smoke, plug-in rows had
  boundary-unstable fractions `0.972` in df `0`--`1` and `0.273` in df
  `1`--`2`; Jeffreys, Dirichlet, and root-shrink variants reduced boundary
  instability to `0.0`.
- The smoothed fixed-tree variants still failed the chi-square tail check:
  tail rates ranged from `0.500` to `0.933`, so every fixed-tree component
  remained `regularized_tail_misaligned` or `regularized_boundary_unstable`.
- On the selected-tree smoke, smoothing again removed boundary instability,
  but tail rates remained `0.969`--`1.000`; selected-tree evidence therefore
  remains fail-closed.
- The production-admissibility summary for both smokes was
  `fail_closed_undefined`.

## Evidence

- `tests/validation/97_test_regularized_wald_statistic_panel.py` verifies
  Bernoulli smoothing, categorical smoothing, boundary-stable variant
  statistics, summary status decisions, and output writing.
- The fixed-tree smoke output records that smoothing repairs the boundary
  diagnostic but not the fixed-tree chi-square tail law.

## Links

- [[differential-statistic-validity-panel-20260613]]
- [[statistic-distribution-shape-panel-20260613]]
- [[projected-wald-statistic]]
- [[production-admissibility-contract-20260613]]
- [[open-mathematical-questions]]
