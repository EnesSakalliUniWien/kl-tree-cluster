---
title: Guarded Continuous Covariance Implementation 2026-06-23
type: source
status: reviewed
updated: 2026-06-23
sources:
  - tree_break_selection/tree/distributions.py
  - benchmarks/shared/runners/dispatch.py
  - benchmarks/shared/runners/method_registry.py
  - raw/assets/benchmark-results/continuous_guarded_covariance_validation_20260623/focused_guarded_covariance_results.csv
  - raw/assets/benchmark-results/continuous_guarded_covariance_validation_20260623/manifest.json
  - raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/manifest.json
tags:
  - source
  - benchmarks
  - gaussian
  - covariance
  - validation
---

# Guarded Continuous Covariance Implementation 2026-06-23

## Summary

The implemented opt-in candidate adds guarded within-child covariance for
continuous Gaussian Wald contrasts while keeping Bernoulli and categorical
pooled null covariance unchanged. The benchmark method
`tbs_continuous_guarded_within_covariance` now runs through the normal TBS
benchmark path for all feature families. Continuous feature-space rows receive
guarded Gaussian covariance plus fixed-coordinate BH sibling gating, while
non-continuous rows retain default TBS sibling gating and discrete covariance.

## Key Points

- The public covariance policy is `continuous_covariance_policy`, with default
  `parent_total` and opt-in `guarded_within_child`. The child-mass guard is
  `continuous_covariance_min_child_leaf_count`, default `8`.
- Under `guarded_within_child`, a binary parent uses pooled immediate-child
  residual covariance only when both child subtrees pass the mass guard:
  `((n_L - 1) Sigma_L + (n_R - 1) Sigma_R) / (n_L + n_R - 2)`. Otherwise the
  implementation falls back to parent-total covariance.
- The covariance override is continuous-only. For Bernoulli and categorical
  feature spaces, the resolver returns no continuous covariance block, so the
  existing pooled Bernoulli and multinomial null covariance formulas remain
  active.
- Benchmark dispatch resolves `continuous_sibling_gate_method` only when
  `feature_space.has_continuous_blocks` is true. Non-continuous rows therefore
  run instead of being skipped, and use the default `projected_wald_inflation`
  sibling gate.
- Focused validation shows the continuous candidate repairs
  `gauss_clear_medium_continuous` from `K=1`, ARI `0.0` to `K=4`, ARI `1.0`,
  repairs `gauss_moderate_3c_continuous` from `K=1`, ARI `0.0` to `K=3`, ARI
  `1.0`, keeps `gauss_null_large_continuous` closed at `K=1`, ARI `1.0`, and
  keeps `dim_consolidated_4c_24f_continuous` at `K=4`, ARI `1.0`.
- The final method id uses `fixed_coordinate_bh` as the active sibling gate.
  A projected-only guarded candidate produced strong fixed-coordinate evidence
  on the dimensional Gaussian root but projected the newly whitened contrast
  away and under-split to one cluster.
- The method id has no benchmark-level continuous-feature-space skip guard.
  The focused smoke panel runs binary and categorical cases and keeps their
  results unchanged relative to default TBS.

## Evidence

- `tree_break_selection/tree/distributions.py` defines the covariance policy
  constants, validation, and policy-aware node covariance resolver.
- `benchmarks/shared/runners/method_registry.py` defines the opt-in
  `tbs_continuous_guarded_within_covariance` method id.
- `benchmarks/shared/runners/dispatch.py` resolves fixed-coordinate sibling
  gating only for continuous feature-space rows.
- `raw/assets/benchmark-results/continuous_guarded_covariance_validation_20260623/focused_guarded_covariance_results.csv`
  records the focused continuous, binary, and categorical smoke results.
- `raw/assets/benchmark-results/continuous_guarded_covariance_validation_20260623/manifest.json`
  records the method list and focused validation note.
- `raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/manifest.json`
  records the no-guard adaptive big benchmark configuration.

## Links

- [[gaussian-within-covariance-replay-20260623]]
- [[discrete-within-covariance-guard-check-20260623]]
- [[gaussian-inner-node-branch-time-debug-20260623]]
