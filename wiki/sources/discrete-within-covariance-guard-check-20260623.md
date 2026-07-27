---
title: Discrete Within-Covariance Guard Check 2026-06-23
type: source
status: reviewed
updated: 2026-06-24
sources:
  - benchmarks/diagnostics/calibration/discrete_covariance_guard.py
  - raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/manifest.json
  - raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/node_covariance_comparison.csv
  - raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/case_context_summary.csv
  - raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/all_suite_roots/manifest.json
  - raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/all_suite_roots/node_covariance_comparison.csv
  - raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/all_suite_roots/case_context_summary.csv
tags:
  - source
  - benchmarks
  - bernoulli
  - categorical
  - covariance
  - validation
---

# Discrete Within-Covariance Guard Check 2026-06-23

## Summary

This diagnostic checks whether the Gaussian guarded within-child covariance
idea should also be applied to Bernoulli and categorical sibling tests. It keeps
the selected tree and branch-time multiplier fixed, then compares the current
pooled null covariance with a Jeffreys-smoothed within-child analogue and a
`25%` pooled-covariance guard.

## Key Points

- The current Bernoulli and categorical formulas are not parent-total empirical
  Gaussian covariance formulas. Bernoulli uses the pooled two-sample null
  variance `(1/n_L + 1/n_R) p_bar (1 - p_bar)`. Categorical one-hot blocks use
  `(1/n_L + 1/n_R) (diag(p_bar) - p_bar p_bar^T)` in the drop-last simplex
  chart. The same branch-time multiplier can be applied to both.
- The within-child analogue was tested as
  `p_L(1-p_L)/n_L + p_R(1-p_R)/n_R` for Bernoulli and
  `Sigma(p_L)/n_L + Sigma(p_R)/n_R` for categorical, with Jeffreys smoothing;
  the guarded variant adds `0.25` of the pooled covariance.
- On representative all-node diagnostics, within-child covariance opens more
  selected null sibling nodes. Bernoulli strict-null nodes open at `18/722`
  under Jeffreys within covariance versus `4/722` under current pooled
  covariance. Categorical strict-null nodes open at `72/994` versus `12/994`.
  The synthetic categorical null is the clearest warning: current pooled
  covariance opens `0/119` selected sibling nodes at alpha `0.01`, Jeffreys
  within covariance opens `14/119`, and the `25%` pooled guard still opens
  `7/119`.
- Root-only all-suite diagnostics show that most binary/categorical roots are
  already significant under the current pooled covariance. Within-child
  covariance mostly strengthens already-open roots, but it also changes
  threshold decisions in a way that is not uniformly beneficial. It rescues
  some weak two-cluster roots, but closes the method-proof categorical
  Dirichlet-multinomial roots where the selected split has a tiny child
  (`1` or `2` samples).
- The discrete conclusion differs from the Gaussian conclusion. For Gaussian
  mean shifts, parent-total empirical covariance contains the split direction
  as a between-child covariance term. For Bernoulli/categorical two-sample
  tests, pooled probability covariance is the ordinary null MLE. A
  within-child discrete replacement is not a safe direct port without a
  stronger selected-null calibration and boundary guard.

## Evidence

- `benchmarks/diagnostics/calibration/discrete_covariance_guard.py` defines the replay.
- `raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/manifest.json`
  records generation metadata for the representative selected-node panel,
  including `generated_at = 2026-06-24T20:10:31+02:00`.
- `raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/node_covariance_comparison.csv`
  records `2697` representative selected-node comparisons.
- `raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/case_context_summary.csv`
  aggregates the representative panel by case and truth context.
- `raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/all_suite_roots/manifest.json`
  records generation metadata for the all-suite root replay, including
  `generated_at = 2026-06-24T20:10:31+02:00`.
- `raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/all_suite_roots/node_covariance_comparison.csv`
  records root-only comparisons for all `71` binary/categorical suite cases.
- `raw/assets/benchmark-results/discrete_covariance_guard_check_20260623/all_suite_roots/case_context_summary.csv`
  summarizes the all-suite root replay.

## Links

- [[gaussian-within-covariance-replay-20260623]]
- [[gaussian-inner-node-branch-time-debug-20260623]]
- [[full-benchmark-run-20260623]]
- [[open-mathematical-questions]]
