---
title: Guarded Continuous Covariance Big Benchmark 2026-06-23
type: source
status: reviewed
updated: 2026-06-24
sources:
  - raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/full_benchmark_comparison.csv
  - raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/tbs_vs_guarded_case_comparison.csv
  - raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/paired_delta_by_feature_representation.csv
  - raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/summary_by_category.csv
  - raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/failure_report.md
  - raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/manifest.json
tags:
  - source
  - benchmarks
  - gaussian
  - covariance
  - validation
---

# Guarded Continuous Covariance Big Benchmark 2026-06-23

## Summary

The full 121-case benchmark compared default `tbs` with the opt-in
`tbs_continuous_guarded_within_covariance` candidate after removing the
benchmark-level continuous-feature-space skip guard. The guarded candidate runs
for all benchmark feature families, but its guarded covariance and
fixed-coordinate BH sibling gate activate only for continuous feature-space
rows; binary, categorical, graph/SBM, median-binary, and quantile-one-hot rows
use the ordinary TBS sibling gate and covariance paths.

## Key Points

- The run used `TBS_CASE_SUITE=full`,
  `TBS_METHODS=tbs,tbs_continuous_guarded_within_covariance`,
  `TBS_ENABLE_PLOTS=0`, `TBS_RUN_RELATIONSHIP_ANALYSIS=0`, and `TBS_N_JOBS=1`.
- Default `tbs` produced `98` ok rows and `23` skip rows. The guarded
  candidate produced `100` ok rows and `21` skip rows; the old explicit
  continuous-feature-space skip reason is absent.
- On the `98` cases where both methods returned ok rows, the guarded candidate
  had no ARI regressions, four improvements, and `94` ties.
- The improvements were continuous Gaussian or Gaussian-outlier cases:
  `gauss_clear_medium_continuous` and `gauss_moderate_3c_continuous` moved from
  ARI `0.0` to `1.0`; `gauss_single_outlier_4c_continuous` moved from ARI
  `0.489047` to `0.991577`; `gauss_outlier_cluster_4c_continuous` moved from
  ARI `0.461191` to `0.951660`.
- Paired feature-representation deltas are exactly zero for binary,
  categorical-one-hot, graph-adjacency, median-binary, and quantile-one-hot
  cases. Improvements are confined to continuous feature-space rows.
- The guarded candidate also ran on two continuous cases where default `tbs`
  skipped because empirical-null inflation calibration support was unavailable:
  `gauss_extreme_noise_highd_continuous` recovered ARI `1.0`, while
  `cont_lowrank_pggn_shrinkage` still under-split with ARI `0.0`.
- The candidate still under-splits `dim_consolidated_4c_72f_continuous`,
  `dim_diffuse_6c_136f_continuous`, `mp_spike_above_bbp_continuous`,
  `cont_lowrank_pggn_shrinkage`, and `phylo_brownian_null_16taxa`. Guarded
  continuous covariance is therefore a diagnostic improvement, not a complete
  production calibration.

## Evidence

- `raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/full_benchmark_comparison.csv`
  records all 242 method-case rows.
- `raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/tbs_vs_guarded_case_comparison.csv`
  records per-case paired comparisons.
- `raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/paired_delta_by_feature_representation.csv`
  records paired deltas by feature representation.
- `raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/summary_by_category.csv`
  records category-level ok counts and mean ARI values.
- `raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/failure_report.md`
  records the benchmark runner failure diagnosis with
  `Generated at: 2026-06-24T20:20:27+02:00`.
- `raw/assets/benchmark-results/continuous_guarded_covariance_adaptive_big_20260623/manifest.json`
  records the command, suite, method list, and derived artifact paths.

## Links

- [[guarded-continuous-covariance-implementation-20260623]]
- [[gaussian-within-covariance-replay-20260623]]
- [[discrete-within-covariance-guard-check-20260623]]
