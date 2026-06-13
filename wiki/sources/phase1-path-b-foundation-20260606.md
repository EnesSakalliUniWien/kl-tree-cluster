---
title: Phase 1 Path B Foundation 2026-06-06
type: source
status: reviewed
updated: 2026-06-06
sources:
  - benchmarks/diagnostics/path_b/phase1_path_b_foundation.py
  - tests/validation/79_test_phase1_path_b_foundation.py
  - benchmarks/results/diagnostics/phase1_path_b_foundation_full_20260606/phase1_path_b_benchmark_comparison.csv
  - benchmarks/results/diagnostics/phase1_path_b_foundation_full_20260606/phase1_path_b_summary.csv
  - benchmarks/results/diagnostics/phase1_path_b_foundation_full_20260606/phase1_cluster_count_distribution.csv
  - benchmarks/results/diagnostics/phase1_path_b_foundation_full_20260606/q5_geometry_covariate_predictive_gain.csv
  - benchmarks/results/diagnostics/phase1_path_b_foundation_full_20260606/phase1_path_b_report.md
  - benchmarks/results/diagnostics/phase1_path_b_foundation_full_20260606/manifest.json
tags:
  - source
  - diagnostics
  - benchmarks
  - path-b
  - calibration
---

# Phase 1 Path B Foundation 2026-06-06

## Summary

This diagnostic implements the Phase 1 Path B foundation sweep. It exposes
`spectral_minimum_dimension` and `passthrough` as KL benchmark parameters,
runs the full KL-only benchmark over `k_min in {0,1,2,3}` crossed with
pass-through on/off, and reruns the Q5 selected-tail diagnostic to measure
geometry covariate gain. The outputs are diagnostic only and do not promote a
production traversal or calibration rule.

## Key Points

- The full sweep wrote `960` KL rows: `120` cases times `4` `k_min` settings
  times `2` pass-through settings.
- The selected full-suite optimum by penalized mean ARI is
  `k_min=1, passthrough=True`.
- For `k_min=1, passthrough=True`, penalized mean ARI is `0.728301`, ok-row
  mean ARI is `0.794510`, skip rate is `0.083333`, and exact cluster-count
  rate on ok rows is `0.736364`.
- For `k_min=1, passthrough=False`, penalized mean ARI falls to `0.674496`,
  ok-row mean ARI falls to `0.735813`, and exact cluster-count rate falls to
  `0.654545`.
- `k_min=2` has higher ok-row mean ARI (`0.824578` with pass-through), but it
  skips more rows (`0.233333` skip rate), so its penalized mean ARI is lower
  (`0.632176`).
- `k_min=3` increases support failures further, with skip rate `0.383333`.
- `k_min=0` is not usable under the current projected-Wald path: it skips
  `117` of `120` rows per pass-through setting because zero-dimensional
  spectral contexts do not support nonzero projected-Wald contrasts.
- Pass-through helps at `k_min=1` by lowering under-split rate from `0.300000`
  to `0.218182` with no increase in over-split rate (`0.045455` in both
  settings).
- The Q5 geometry covariate gain panel identifies
  `q5_barycentric_edge_spectral` as the best model by residual-tail error
  reduction versus the no-spectral baseline. Median residual-tail absolute
  error improves by `0.260780`, median tail AUC improves by `0.004612`, and
  median held-out R-squared improves by `0.452426`.
- The result supports Path B's immediate direction: prefer `k_min=1` over the
  current `k_min=2` for the full benchmark when skips are penalized, preserve
  pass-through for now, and keep barycentric edge/spectral geometry as
  diagnostic selected-tail context rather than production calibration.

## Evidence

- `benchmarks/diagnostics/path_b/phase1_path_b_foundation.py` contains the
  resumable Phase 1 Path B runner and summary logic.
- `tests/validation/79_test_phase1_path_b_foundation.py` verifies the grid,
  summary, cluster-count distribution, and Q5 predictive-gain calculations.
- `benchmarks/results/diagnostics/phase1_path_b_foundation_full_20260606/phase1_path_b_summary.csv`
  records the full benchmark ablation summary.
- `benchmarks/results/diagnostics/phase1_path_b_foundation_full_20260606/q5_geometry_covariate_predictive_gain.csv`
  records the Q5 geometry covariate gain table.

## Links

- [[path-conditioned-barycentric-action-diagnostics-20260606]]
- [[mp-projection-dimension-behavior-sweeps-20260605]]
- [[selected-tail-law-q5-validation-20260604]]
- [[null-edge-sibling-calibration-enhancement-plan]]
