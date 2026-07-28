---
title: Differential Statistic Validity Panel 2026-06-13
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/statistics/differential_statistic_validity_panel.py
  - tests/validation/calibration/statistics/96_test_differential_statistic_validity_panel.py
  - benchmarks/diagnostics/calibration/traversal/production_admissibility_contract.py
  - benchmarks/validation/statistics/selected_edge_type1_geometry.py
tags:
  - source
  - diagnostics
  - calibration
  - statistics
  - geometry
---

# Differential Statistic Validity Panel 2026-06-13

## Summary

`differential_statistic_validity_panel.py` adds a diagnostic-only validity
layer for projected-Wald sibling statistics. It checks Fisher/Wald boundary
geometry, fixed-projection finite-difference sensitivity, recomputed-projection
sensitivity, eigengap stability, and nonsmooth Hamming tree-selection
stability before any production promotion.

## Key Points

- The panel reconstructs the same whitened sibling contrast and parent PCA
  projection used by the existing projected-Wald sibling test.
- Bernoulli and categorical V1 rows report Fisher variance floors and condition
  numbers. Zero-variance pooled coordinates become
  `wald_metric_boundary_unstable`.
- Projection diagnostics report eigengap at selected sibling dimension,
  projection-instability score, fixed-projection sensitivity, and
  recomputed-projection sensitivity under small spectral covariance
  perturbations.
- Hamming selected-tree derivatives are deliberately marked nonsmooth. The
  panel reports `nonsmooth_hamming_selection` and topology stability under
  discrete perturbations instead of pretending a smooth tree derivative exists.
- A one-replicate `binary_2clusters` smoke over `fixed_tree` and
  `selected_tree` produced `98` rows. The production summary had `4` required
  components, all fail-closed, with final decision `fail_closed_undefined`.
- In that smoke, `81` rows were `wald_metric_boundary_unstable`, `11` were
  `fixed_subspace_candidate`, and `6` were
  `nonsmooth_selection_geometry`. The boundary instability dominated the group
  summaries in both fixed-tree and selected-tree modes.
- The next diagnostic step is the regularized Wald panel. It shows that
  Jeffreys, Dirichlet, and root-shrink smoothing remove the boundary
  instability in the tested fixed-tree rows, but do not repair chi-square tail
  alignment.

## Evidence

- `tests/validation/calibration/statistics/96_test_differential_statistic_validity_panel.py` verifies
  Fisher boundary detection, categorical boundary detection,
  finite-difference agreement with the analytic projected-quadratic derivative,
  small-eigengap projection instability, fixed-subspace diagnostic-only
  contract behavior, and output writing.
- The public CLI writes `differential_statistic_validity_rows.csv`,
  `differential_statistic_validity_summary.csv`,
  `production_admissibility_components.csv`,
  `production_admissibility_summary.csv`, and `manifest.json`.

## Links

- [[projected-wald-statistic]]
- [[statistic-distribution-shape-panel-20260613]]
- [[covariance-laplacian-panel-20260613]]
- [[production-admissibility-contract-20260613]]
- [[regularized-wald-statistic-panel-20260613]]
- [[open-mathematical-questions]]
