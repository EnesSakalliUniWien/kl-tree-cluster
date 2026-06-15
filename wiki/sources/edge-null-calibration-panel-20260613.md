---
title: Edge Null Calibration Panel 2026-06-13
type: source
status: reviewed
updated: 2026-06-13
sources:
  - benchmarks/diagnostics/calibration/edge_null_calibration_panel.py
  - tests/validation/88_test_edge_null_calibration_panel.py
  - wiki/analyses/null-edge-sibling-calibration-enhancement-plan.md
tags:
  - source
  - diagnostics
  - calibration
  - edge
---

# Edge Null Calibration Panel 2026-06-13

## Summary

The first null-edge calibration roadmap phase now has an executable diagnostic
contract. `edge_null_calibration_panel.py` scores precomputed child-parent
edge-test rows separately for `fixed_tree_null`, `selected_tree_null`, and
`selected_tree_signal` roles.

## Key Points

- The diagnostic normalizes edge rows, computes raw and BH edge rejection
  indicators at a supplied alpha, records `-log10(p)`, and preserves common
  case, edge, feature-family, and projection metadata when present.
- Summary rows are role-specific. Fixed-tree and selected-tree null roles are
  checked against nominal alpha with an explicit tolerance; selected-tree signal
  rows are reported as descriptive retention, not null calibration.
- Unknown context roles are rejected instead of pooled. This prevents the
  post-selection bias question from collapsing fixed-tree null, selected-tree
  null, and selected signal evidence into one misleading rate.
- The output manifest states that the panel is diagnostic-only and does not
  install an edge calibration rule.

## Evidence

- `benchmarks/diagnostics/calibration/edge_null_calibration_panel.py` defines
  the dataframe API, CSV runner, manifest, CLI, and role contract.
- `tests/validation/88_test_edge_null_calibration_panel.py` verifies nominal
  fixed-tree null behavior, inflated selected-tree null behavior, descriptive
  selected signal retention, unknown-role rejection, and output writing.
- [[null-edge-sibling-calibration-enhancement-plan]] names the edge-null panel
  as the first roadmap phase before sibling-null and traversal-geometry
  validation.

## Links

- [[null-edge-sibling-calibration-enhancement-plan]]
- [[open-mathematical-questions]]
- [[projected-wald-statistic]]
