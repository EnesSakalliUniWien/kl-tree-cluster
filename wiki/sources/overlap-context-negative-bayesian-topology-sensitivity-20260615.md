---
title: Overlap Context-Negative Bayesian Topology Sensitivity 2026-06-15
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_context_negative_bayesian_topology_sensitivity.py
  - tests/validation/calibration/overlap/133_test_overlap_context_negative_bayesian_topology_sensitivity.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_bayesian_topology_sensitivity/overlap_context_negative_bayesian_topology_sensitivity_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_bayesian_topology_sensitivity/overlap_context_negative_bayesian_topology_sensitivity_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_bayesian_topology_sensitivity/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - topology
  - bayesian
  - sensitivity
---

# Overlap Context-Negative Bayesian Topology Sensitivity 2026-06-15

## Summary

`overlap_context_negative_bayesian_topology_sensitivity.py` audits whether the
Bayesian topology-law result is robust to component ablations and context
penalty weights. It tests whether the focused truth row is carried by
structural topology components rather than by selected-family evidence or the
soft context term.

## Key Points

- The runner writes profile-level sensitivity rows, a one-row summary, and
  `manifest.json`.
- It evaluates `13` component profiles across context penalty weights
  `0`, `25`, `50`, `75`, and `100`.
- `60/65` profile-weight combinations top-rank and separate the focused truth
  row from all negatives.
- All `5/5` `topology_only` profiles separate the truth row, with minimum
  margin `4.632369`.
- All `5/5` `outgoing_topology_only` profiles separate the truth row, with
  minimum margin `1.198668`.
- `selected_context_only` separates in `0/5` weights. It ranks the truth row
  third at context weight `50`, with margin `-0.897055`.
- Single-component topology profiles also separate in `10` cases:
  outgoing balance alone and outgoing edge-norm balance alone separate at all
  tested context weights.
- The diagnostic status is
  `bayesian_topology_structural_signal_robust_not_selected_context`.

## Evidence

- `tests/validation/calibration/overlap/133_test_overlap_context_negative_bayesian_topology_sensitivity.py`
  verifies that topology-only and outgoing-topology-only profiles separate,
  while selected-family plus context stays fail-closed, and that the runner
  writes outputs.
- Verification passed:
  `pytest tests/validation/calibration/overlap/133_test_overlap_context_negative_bayesian_topology_sensitivity.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_context_negative_bayesian_topology_sensitivity.py tests/validation/calibration/overlap/133_test_overlap_context_negative_bayesian_topology_sensitivity.py`.

## Links

- [[overlap-context-negative-bayesian-topology-law-20260615]]
- [[overlap-context-negative-topology-transfer-20260615]]
- [[overlap-context-negative-topology-conditioning-20260615]]
- [[open-mathematical-questions]]
