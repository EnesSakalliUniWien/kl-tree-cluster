---
title: Recursive P-Value Geometry 2026-06-06
type: source
status: reviewed
updated: 2026-06-06
sources:
  - benchmarks/diagnostics/path_b/recursive_pvalue_geometry.py
  - tests/validation/80_test_recursive_pvalue_geometry.py
  - benchmarks/results/diagnostics/recursive_pvalue_geometry_full_20260606/recursive_pvalue_geometry_summary.csv
  - benchmarks/results/diagnostics/recursive_pvalue_geometry_full_20260606/recursive_pvalue_geometry_edges.csv
  - benchmarks/results/diagnostics/recursive_pvalue_geometry_full_20260606/recursive_pvalue_geometry_nodes.csv
  - benchmarks/results/diagnostics/recursive_pvalue_geometry_full_20260606/recursive_pvalue_geometry_case_status.csv
  - benchmarks/results/diagnostics/recursive_pvalue_geometry_full_20260606/recursive_pvalue_geometry_report.md
  - benchmarks/results/diagnostics/recursive_pvalue_geometry_full_20260606/recursive_pvalue_geometry_six_panel.png
  - benchmarks/results/diagnostics/recursive_pvalue_geometry_full_20260606/manifest.json
tags:
  - source
  - diagnostics
  - geometry
  - calibration
  - p-values
---

# Recursive P-Value Geometry 2026-06-06

## Summary

This diagnostic tests whether edge p-values, sibling p-values, alpha margins,
selected PCA subspaces, eigenvalue spectra, and chi-square tail sensitivity
form a coherent recursive coordinate field on the selected tree. It is
diagnostic only: it records geometry after tree, traversal path, projection
dimension, and PCA basis selection, and does not promote a production
calibration or traversal rule.

## Key Points

- The full TBS-only diagnostic completed `110` ok cases and skipped `10` cases.
  Skips were the expected strict calibration-support or dense continuous
  covariance memory-contract failures.
- The edge and sibling p-value fields are connected but not equivalent. Raw
  edge-vs-parent-sibling evidence has Spearman correlation `0.392907`, while
  the gate-visible corrected version has correlation `0.309670`.
- Sibling p-values have recursive continuity. Raw parent-vs-child sibling
  evidence has Spearman correlation `0.443386`; traversal-aligned corrected
  sibling evidence has correlation `0.538674`.
- The dominant missed law is an edge/sibling asymmetry. Raw edge p-values pass
  the edge alpha margin on `44150 / 52738` edges (`0.8372`), while raw sibling
  p-values pass sibling alpha on only `537 / 6534` sibling contexts (`0.0822`).
  Corrected edge p-values pass on `13068 / 14078` Tree-BH-tested edges
  (`0.9283`), while corrected sibling p-values pass on `516 / 6534` contexts
  (`0.0790`).
- Selected-subspace rotation tracks the edge-minus-parent-sibling evidence gap
  only moderately. Raw and corrected correlations are `0.254220` and
  `0.255522`, respectively. This supports using principal-angle subspace
  geometry as a traversal diagnostic or stratifier, not as a standalone
  calibration equation.
- Eigenvalue gap alone is not the missing chi-square law in the full suite:
  selected eigenvalue gap vs sibling chi-square tail sensitivity has global
  Spearman correlation `0.027132`, despite stronger effects in small probes.
- Stored edge and sibling p-values are mechanically consistent with the
  chi-square survival function for their stored statistic and degrees of
  freedom: median absolute log10 residual is `0.000000` for both. This only
  checks arithmetic consistency, not selected-tree or selected-PCA validity.
- By depth, sibling evidence is strong at the root and first level, then the
  median sibling alpha margin falls below threshold from depth `2` onward,
  while edge evidence can remain very large. The recursive manifold therefore
  looks like a strong edge field passing through sparse sibling-support
  neighborhoods, not a single smooth p-value diagonal.
- Subspace alignment is computed with principal angles, so it is invariant to
  eigenvector sign flips. Individual eigenvectors remain unstable when
  eigenvalue gaps are small; selected subspaces are the safer coordinate
  object.

## Evidence

- `benchmarks/diagnostics/path_b/recursive_pvalue_geometry.py` implements the
  edge/node panel construction, raw and corrected p-value transforms, alpha
  margins, chi-square tail sensitivity, chi-square residual checks, eigenvalue
  geometry, and sign-invariant principal-subspace alignment.
- `tests/validation/80_test_recursive_pvalue_geometry.py` verifies p-value
  transforms, chi-square residual arithmetic, sign-invariant subspace
  alignment, eigenvalue-gap reporting, and synthetic panel construction.
- `benchmarks/results/diagnostics/recursive_pvalue_geometry_full_20260606/recursive_pvalue_geometry_summary.csv`
  records the global and per-case correlation and residual summaries.
- `benchmarks/results/diagnostics/recursive_pvalue_geometry_full_20260606/recursive_pvalue_geometry_edges.csv`
  and `benchmarks/results/diagnostics/recursive_pvalue_geometry_full_20260606/recursive_pvalue_geometry_nodes.csv`
  record the row-level edge and node geometry panels.
- `benchmarks/results/diagnostics/recursive_pvalue_geometry_full_20260606/recursive_pvalue_geometry_six_panel.png`
  visualizes the raw p-value field, corrected gate-visible field, recursive
  sibling continuity, subspace-rotation gap, and eigen-gap tail-sensitivity
  relationships.

## Links

- [[phase1-path-b-foundation-20260606]]
- [[path-conditioned-barycentric-action-diagnostics-20260606]]
- [[barycentric-action-equation-diagnostic-20260606]]
- [[selected-tail-law-q5-validation-20260604]]
- [[null-edge-sibling-calibration-enhancement-plan]]
