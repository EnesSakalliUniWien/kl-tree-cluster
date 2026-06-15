---
title: Selected Edge Sibling Null Equation 2026-06-13
type: source
status: reviewed
updated: 2026-06-13
sources:
  - benchmarks/diagnostics/calibration/selected_edge_sibling_null_equation.py
  - tests/validation/92_test_selected_edge_sibling_null_equation.py
  - benchmarks/validation/selected_edge_type1_geometry.py
  - benchmarks/diagnostics/math_trace/barycentric_action.py
  - raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/selected_edge_geometry_siblings.csv
  - wiki/sources/sibling-null-calibration-panel-20260613.md
tags:
  - source
  - diagnostics
  - calibration
  - equation
---

# Selected Edge Sibling Null Equation 2026-06-13

## Summary

The barycentric edge/sibling relationship is now represented as an executable
conditional empirical-null equation. The diagnostic estimates

\[
\Pr(T_u^{\mathrm{sib}} \ge t \mid f_u, k_u, A_u^{\mathrm{edge}},
b_u, \mathrm{edge\ path\ open})
\]

from matched null rows, where \(f_u\) is feature family, \(k_u\) is sibling
projection dimension, \(A_u^{\mathrm{edge}}\) is an edge-action bin, and
\(b_u\) is barycentric balance.

## Key Points

- `selected_edge_sibling_null_equation.py` computes barycentric variables:
  left weight, balance, log leverage, sampling variance scale, edge action,
  edge-action bin, and balance bin.
- Matched null support is exact over feature family, projection dimension,
  edge-action bin, barycentric-balance bin, and edge-path-open status.
- A finite-sample add-one empirical tail probability is emitted only when the
  matched null context has at least the configured support count.
- Missing or sparse matched contexts return fail-closed statuses such as
  `undefined_no_matched_null_context` or `insufficient_null_support`.
- This is an equation-level diagnostic, not a default production p-value
  replacement.
- The 2026-06-04 stored selected-edge pilot sibling export is not sufficient
  to evaluate this equation directly: it has finite adjusted sibling p-values
  but no finite `sibling_raw_stat` or `sibling_raw_p` values. Future production
  runs must export raw sibling statistics for every equation candidate row.
- The selected-edge geometry exporter now emits raw sibling statistics and raw
  p-values before calibration adjustment, including raw-only rows when
  decomposition fails before adjusted sibling annotations are available.
- On live method-proof examples, the equation produced finite conditional
  empirical p-values where exact matched strict-null support existed and
  withheld p-values for sparse or unmatched contexts. This confirms fail-closed
  behavior, not production calibration.

## Evidence

- `benchmarks/diagnostics/calibration/selected_edge_sibling_null_equation.py`
  defines the equation variables, matched-context summaries, per-record
  conditional p-values, CSV runner, and manifest.
- `tests/validation/92_test_selected_edge_sibling_null_equation.py` verifies
  barycentric variable construction, supported empirical p-values, sparse
  support fail-closed behavior, and output writing.
- `benchmarks/diagnostics/math_trace/barycentric_action.py` records the exact
  edge/sibling barycentric identity that motivates the conditioning variables.

## Links

- [[sibling-null-calibration-panel-20260613]]
- [[edge-null-calibration-panel-20260613]]
- [[production-admissibility-contract-20260613]]
- [[open-mathematical-questions]]
