---
title: Overlap Internal-Node Likelihood Sensitivity 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - benchmarks/diagnostics/calibration/overlap_internal_node_likelihood_sensitivity.py
  - tests/validation/123_test_overlap_internal_node_likelihood_sensitivity.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_likelihood_sensitivity/overlap_internal_node_likelihood_sensitivity.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_likelihood_sensitivity/overlap_internal_node_likelihood_sensitivity_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_likelihood_sensitivity/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - bayesian
---

# Overlap Internal-Node Likelihood Sensitivity 2026-06-15

## Summary

`overlap_internal_node_likelihood_sensitivity.py` sweeps a small predeclared
threshold grid around the row-level overlap internal-node likelihood probe. It
checks whether the useful behavior is stable or a single lucky cutoff. It is
diagnostic-only and does not promote a production threshold.

## Key Points

- The runner writes `overlap_internal_node_likelihood_sensitivity.csv`,
  `overlap_internal_node_likelihood_sensitivity_summary.csv`, and
  `manifest.json`.
- The grid varies context margin floor, soft subspace floor, size-balance
  floor, edge-norm-balance floor, and fragment-risk ceiling while keeping the
  selected-family p-value Bayes-factor floor fixed at `3.0`.
- The focused overlap run evaluates `324` grid points.
- `216/324` grid points have zero selected-null, diffuse/wrong, or
  fragment-like candidates.
- `24/324` grid points retain at least `4/5` truth-recovery internal nodes
  with zero negative candidates.
- The default row-level rule is `zero_negative_recovery_retaining`.
- Context margin is the decisive threshold. When the context margin floor is
  relaxed to `-0.002`, all `108/108` grid points leak negative rows. At floors
  `0.0` and `0.002`, leakage is `0/108`; the maximum truth-recovery retention
  is `4/5` at `0.0` and `3/5` at `0.002`.
- Method implication: context margin should remain a hard conditional
  neighborhood gate in the next Bayesian traversal law. The softer overlap
  degrees of freedom should be subspace, balance, and fragment-risk tolerance,
  not negative context margin.

## Evidence

- `tests/validation/123_test_overlap_internal_node_likelihood_sensitivity.py`
  verifies zero-negative versus leakage classification, default-rule summary,
  and output writing.
- Verification passed:
  `pytest tests/validation/123_test_overlap_internal_node_likelihood_sensitivity.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_internal_node_likelihood_sensitivity.py tests/validation/123_test_overlap_internal_node_likelihood_sensitivity.py`.

## Links

- [[overlap-internal-node-bayesian-likelihood-probe-20260615]]
- [[overlap-bayesian-neighborhood-component-audit-20260615]]
- [[open-mathematical-questions]]
