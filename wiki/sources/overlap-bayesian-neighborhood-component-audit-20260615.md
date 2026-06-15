---
title: Overlap Bayesian Neighborhood Component Audit 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - benchmarks/diagnostics/calibration/overlap_bayesian_neighborhood_component_audit.py
  - tests/validation/121_test_overlap_bayesian_neighborhood_component_audit.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/bayesian_neighborhood_component_audit/overlap_bayesian_neighborhood_component_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/bayesian_neighborhood_component_audit/overlap_bayesian_neighborhood_component_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/bayesian_neighborhood_component_audit/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - bayesian
---

# Overlap Bayesian Neighborhood Component Audit 2026-06-15

## Summary

`overlap_bayesian_neighborhood_component_audit.py` inspects the limiting terms
inside the non-permutation conditional Bayesian traversal diagnostic. It shows
which structural neighborhood likelihood components block coherent-split
promotion after selected-family p-value evidence is already strong.

## Key Points

- The runner writes `overlap_bayesian_neighborhood_component_rows.csv`,
  `overlap_bayesian_neighborhood_component_summary.csv`, and `manifest.json`.
- It consumes posterior-style component rows and does not refit weights,
  introduce permutations, or promote thresholds.
- A strong-neighborhood candidate still requires neighborhood log Bayes factor
  at least `2.0`.
- In the focused residual run, selected-null families have `0/8` coherent
  candidates: `5/8` have p-value evidence without strong neighborhood support,
  and `3/8` are structurally incoherent. The limiting component is
  `context_margin_log_bayes_factor` for all selected-null families.
- Non-recovery signal families have `0/3` coherent candidates: `2/3` have
  p-value evidence without strong neighborhood support, and `1/3` is
  structurally incoherent. Their limiting components are context margin
  (`2/3`) and subspace (`1/3`).
- Truth-recovery families have `1/5` coherent candidates. Among the `4/5`
  blocked recovery families, two are limited by
  `balanced_recovery_log_bayes_factor`, one by context margin, and one by
  subspace.
- Method implication: the Bayesian law should keep p-value evidence separate
  from structural neighborhood likelihood. The next likelihood term must better
  model true overlap recovery where current balanced-recovery and subspace
  proxies are too conservative, while preserving the context-margin block that
  catches selected-null families.

## Evidence

- `tests/validation/121_test_overlap_bayesian_neighborhood_component_audit.py`
  verifies limiting-component detection, p-value-without-neighborhood status
  counts, and output writing.
- Verification passed:
  `pytest tests/validation/121_test_overlap_bayesian_neighborhood_component_audit.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_bayesian_neighborhood_component_audit.py tests/validation/121_test_overlap_bayesian_neighborhood_component_audit.py`.

## Links

- [[overlap-conditional-bayesian-traversal-law-20260615]]
- [[overlap-selected-family-law-requirements-20260614]]
- [[open-mathematical-questions]]
