---
title: Null Law Decomposition Panel 2026-06-13
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/statistics/null_law_decomposition_panel.py
  - tests/validation/calibration/statistics/98_test_null_law_decomposition_panel.py
  - benchmarks/diagnostics/calibration/traversal/production_admissibility_contract.py
tags:
  - source
  - diagnostics
  - calibration
  - statistics
---

# Null Law Decomposition Panel 2026-06-13

## Summary

`null_law_decomposition_panel.py` isolates the sibling projected-Wald null law
under fixed topology by comparing three projection sources: the current
same-sample adaptive projection, an independent projection learned from the
tree-construction null sample, and an independent random orthonormal
projection. It tests whether the chi-square failure is caused by the quadratic
operator itself or by data-dependent projection selection.

## Key Points

- The panel is diagnostic-only and does not change production Tree-Break Selection
  calibration.
- The maintained `binary_2clusters` fixed-topology smoke with `20` replicates
  shows that same-sample adaptive projection has severely inflated tails:
  `0.916` in df `0`--`1` and `0.745` in df `1`--`2`.
- The independent tree-sample projection is near nominal in the same topology:
  tail rates are `0.053` in df `0`--`1` and `0.076` in df `1`--`2`.
- The random fixed orthonormal projection is also near nominal: tail rates are
  `0.063` in df `0`--`1` and `0.036` in df `1`--`2`.
- Projection row orthonormality error is numerical noise only, and the
  operator Satterthwaite scale is `1.0` with df equal to the projection
  dimension. The implemented quadratic form is therefore algebraically the
  fixed-subspace chi-square form when the projection is fixed.
- The production-admissibility summary remains `fail_closed_undefined` because
  the adaptive same-sample projection components are fail-closed. The fixed
  projection controls are diagnostic candidates, not production promotion.

## Evidence

- `tests/validation/calibration/statistics/98_test_null_law_decomposition_panel.py` verifies
  projection-operator weights, adaptive-tail fail-closed status, fixed
  projection diagnostic status, and output writing.
- The `20`-replicate smoke output records that the null-law failure is caused
  by learning the projection from the same null sample being tested, not by a
  nonorthonormal projection matrix or wrong quadratic weights.

## Links

- [[projected-wald-statistic]]
- [[regularized-wald-statistic-panel-20260613]]
- [[differential-statistic-validity-panel-20260613]]
- [[production-admissibility-contract-20260613]]
- [[open-mathematical-questions]]
