---
title: Overlap Conditional Bayesian Traversal Law 2026-06-15
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_conditional_bayesian_traversal_law.py
  - tests/validation/calibration/overlap/120_test_overlap_conditional_bayesian_traversal_law.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_bayesian_traversal_law/overlap_conditional_bayesian_traversal_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_bayesian_traversal_law/overlap_conditional_bayesian_traversal_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_bayesian_traversal_law/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - bayesian
---

# Overlap Conditional Bayesian Traversal Law 2026-06-15

## Summary

`overlap_conditional_bayesian_traversal_law.py` is a non-permutation
diagnostic that converts residual selected-family evidence into transparent
posterior-style log odds for a coherent structural split. It is a model
specification check, not a production rule.

## Key Points

- The diagnostic writes `overlap_conditional_bayesian_traversal_rows.csv`,
  `overlap_conditional_bayesian_traversal_summary.csv`, and `manifest.json`.
- The selected-family p-value contributes through a lower-bound log Bayes
  factor against a point null. The law then subtracts a selection-context
  penalty for family size, parent size, and imbalance.
- The neighborhood term adds homogeneity gain, continuous context margin,
  subspace consensus, size and edge-norm balance, and balanced-recovery proxy
  evidence, then subtracts fragment-risk penalty.
- A family is a coherent candidate only when posterior odds are high and the
  neighborhood log Bayes factor is at least `2.0`; p-value extremeness alone is
  explicitly fail-closed as either
  `p_value_extreme_neighborhood_insufficient` or
  `p_value_extreme_structurally_incoherent`.
- In the focused overlap residual run, selected-null families have high
  posterior medians under the p-value bridge, but `0/8` pass the strong
  neighborhood requirement. Non-recovery families also have `0/3` coherent
  candidates.
- Only `1/5` truth-recovery residual families passes as a coherent candidate;
  the other `4/5` stay unstable because their neighborhood evidence is
  insufficient or structurally incoherent. This is conservative and
  diagnostic-only.
- Method implication: a Bayesian formulation helps by making the missing
  conditional law explicit, but the likelihood cannot be dominated by sibling
  p-value evidence. The structural neighborhood likelihood must be strong
  enough to beat selected-null and non-recovery selected-family alternatives.

## Evidence

- `tests/validation/calibration/overlap/120_test_overlap_conditional_bayesian_traversal_law.py`
  verifies monotonic p-value Bayes-factor behavior, the structural-neighborhood
  requirement, and output writing.
- Verification passed:
  `pytest tests/validation/calibration/overlap/120_test_overlap_conditional_bayesian_traversal_law.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_conditional_bayesian_traversal_law.py tests/validation/calibration/overlap/120_test_overlap_conditional_bayesian_traversal_law.py`.

## Links

- [[overlap-selected-family-law-requirements-20260614]]
- [[overlap-residual-family-recovery-20260614]]
- [[overlap-threshold-stability-contract-20260614]]
- [[open-mathematical-questions]]
