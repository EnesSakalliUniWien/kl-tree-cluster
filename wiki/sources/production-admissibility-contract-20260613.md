---
title: Production Admissibility Contract 2026-06-13
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/traversal/production_admissibility_contract.py
  - tests/validation/calibration/traversal/91_test_production_admissibility_contract.py
  - benchmarks/diagnostics/calibration/edge/selected_edge_sibling_null_equation.py
  - benchmarks/diagnostics/calibration/statistics/statistic_distribution_shape_panel.py
  - benchmarks/diagnostics/calibration/statistics/covariance_laplacian_panel.py
  - benchmarks/diagnostics/calibration/edge/selected_edge_sibling_postrun_analysis.py
  - benchmarks/diagnostics/calibration/statistics/differential_statistic_validity_panel.py
  - benchmarks/diagnostics/calibration/statistics/regularized_wald_statistic_panel.py
  - benchmarks/diagnostics/calibration/statistics/null_law_decomposition_panel.py
  - wiki/analyses/null-edge-sibling-calibration-enhancement-plan.md
tags:
  - source
  - diagnostics
  - calibration
  - production
---

# Production Admissibility Contract 2026-06-13

## Summary

The final roadmap phase now has an explicit diagnostic contract for aggregating
component evidence. `production_admissibility_contract.py` combines required
component statuses into `production_admissible`, `diagnostic_only`, or
`fail_closed_undefined` decisions.

## Key Points

- Every required component must be production-ready before a contract can be
  `production_admissible`.
- Fail-closed statuses such as `undefined_external_not_admissible`,
  `undefined_sparse_context`, `above_nominal_tolerance`, or
  `external_selected_tail_fail_closed` force `fail_closed_undefined`.
- Diagnostic statuses such as `diagnostic_only_guard` or
  `external_selected_tail_candidate_descriptive` keep the whole contract
  diagnostic-only.
- The selected edge+sibling equation statuses are deliberately classified:
  `conditional_empirical_p_value` and `supported_context` are diagnostic-only,
  while `insufficient_null_support` and `undefined_no_matched_null_context`
  fail closed.
- Statistic distribution-shape statuses are deliberately conservative:
  `chi_square_shape_candidate` is diagnostic-only, while tail or skew mismatch
  statuses such as `skew_exceeds_df_reference` fail closed.
- Covariance Laplacian statuses are diagnostic-only descriptors of covariance
  graph geometry and do not make a component production-ready.
- `selected_edge_sibling_postrun_analysis.py` now writes component rows and a
  production summary directly from enriched sibling artifacts, so benchmark
  evidence has an executable promotion/fail-closed decision path.
- Differential statistic-validity statuses are classified conservatively:
  `fixed_subspace_candidate` is diagnostic-only, while Fisher boundary,
  whitening, projection, selection, and residual tail-mismatch statuses fail
  closed.
- Regularized Wald statuses are also conservative:
  `regularized_fixed_tree_candidate` is diagnostic-only, while boundary,
  insufficient-row, selected-tree-not-yet-calibrated, and tail-misaligned
  statuses fail closed.
- Null-law decomposition statuses are conservative:
  `null_law_fixed_projection_candidate` is diagnostic-only, while adaptive
  same-sample tail inflation, fixed-projection tail mismatch, and insufficient
  rows fail closed.
- Unknown component statuses are rejected, so new diagnostic statuses must be
  added deliberately before they can enter a production decision summary.

## Evidence

- `benchmarks/diagnostics/calibration/traversal/production_admissibility_contract.py`
  defines the dataframe API, CLI, normalized component output, summary output,
  and manifest.
- `tests/validation/calibration/traversal/91_test_production_admissibility_contract.py` verifies
  production-admissible, diagnostic-only, fail-closed, unknown-status, and
  output-writing cases, including the selected edge+sibling equation statuses.
- `tests/validation/calibration/edge/95_test_selected_edge_sibling_postrun_analysis.py` verifies
  that post-run distribution-shape mismatch produces a fail-closed production
  summary through this contract.
- `tests/validation/calibration/statistics/96_test_differential_statistic_validity_panel.py` verifies
  that differential statistic-validity outputs feed the same fail-closed
  production-admissibility contract.
- `tests/validation/calibration/statistics/97_test_regularized_wald_statistic_panel.py` verifies that
  regularized Wald outputs feed the same production-admissibility contract.
- `tests/validation/calibration/statistics/98_test_null_law_decomposition_panel.py` verifies that
  null-law decomposition outputs feed the same production-admissibility
  contract.
- [[null-edge-sibling-calibration-enhancement-plan]] calls for explicit
  production-admissible fail-closed statuses after edge, sibling, and traversal
  diagnostics are separated.

## Links

- [[null-edge-sibling-calibration-enhancement-plan]]
- [[edge-null-calibration-panel-20260613]]
- [[sibling-null-calibration-panel-20260613]]
- [[traversal-guard-validation-panel-20260613]]
- [[differential-statistic-validity-panel-20260613]]
- [[regularized-wald-statistic-panel-20260613]]
- [[null-law-decomposition-panel-20260613]]
