---
title: Selected Edge Sibling Postrun Analysis 2026-06-13
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/edge/selected_edge_sibling_postrun_analysis.py
  - tests/validation/calibration/edge/95_test_selected_edge_sibling_postrun_analysis.py
  - benchmarks/validation/statistics/selected_edge_type1_geometry.py
  - benchmarks/diagnostics/calibration/statistics/statistic_distribution_shape_panel.py
  - benchmarks/diagnostics/calibration/edge/selected_edge_sibling_null_equation.py
  - benchmarks/diagnostics/calibration/traversal/production_admissibility_contract.py
tags:
  - source
  - diagnostics
  - calibration
  - distribution
---

# Selected Edge Sibling Postrun Analysis 2026-06-13

## Summary

`selected_edge_sibling_postrun_analysis.py` consumes enriched
`selected_edge_geometry_siblings.csv` artifacts and writes the downstream
analyses needed for the selected-edge sibling question: distribution-shape rows
and summaries, conditional selected edge+sibling equation rows, contexts, and
summaries, plus a conservative production-admissibility summary.

## Key Points

- The analyzer requires raw sibling statistics, raw edge p-values, sample
  sizes, projection dimension, feature family, and edge-path-open metadata.
- It preserves the production boundary: all outputs are diagnostic evidence
  unless a separate admissibility contract promotes a component.
- A local smoke run over `binary_2clusters`, `selected_tree`, two replicates,
  and `edge_alpha=0.001`, `sibling_alpha=0.01` produced `196` edge rows,
  `98` sibling rows, and `2` final rows.
- In that smoke run, current chi-square sibling tails were still inflated:
  both df bins had current tail rate `1.0` at alpha `0.05`.
- The covariance-inferred alternate reference reduced but did not solve the
  tails: alternate tail rates were about `0.158` for df `0`--`1` and `0.864`
  for df `1`--`2`.
- The selected edge+sibling equation emitted `95` supported
  `conditional_empirical_p_value` rows with median p-value `0.6` and withheld
  p-values for `3` rows as `insufficient_null_support`.
- The same smoke run now produces an explicit production-admissibility summary
  with `7` required components: `3` fail-closed blockers and `4`
  diagnostic-only components. The final decision is `fail_closed_undefined`.
- The blockers are the two `skew_exceeds_df_reference` distribution-shape bins
  and the selected edge+sibling equation `insufficient_null_support` status.

## Evidence

- `tests/validation/calibration/edge/95_test_selected_edge_sibling_postrun_analysis.py` builds a
  real enriched sibling artifact through `run_selected_edge_replicate`, checks
  raw-statistic distribution records, checks selected edge+sibling equation
  records, and verifies all expected post-run output files are written.
- `benchmarks/diagnostics/calibration/edge/selected_edge_sibling_postrun_analysis.py`
  writes `sibling_distribution_shape_rows.csv`,
  `sibling_distribution_shape_summary.csv`,
  `selected_edge_sibling_equation_rows.csv`,
  `selected_edge_sibling_equation_contexts.csv`,
  `selected_edge_sibling_equation_summary.csv`,
  `production_admissibility_components.csv`,
  `production_admissibility_summary.csv`, and `manifest.json`.

## Links

- [[statistic-distribution-shape-panel-20260613]]
- [[selected-edge-sibling-null-equation-20260613]]
- [[covariance-laplacian-panel-20260613]]
- [[production-admissibility-contract-20260613]]
