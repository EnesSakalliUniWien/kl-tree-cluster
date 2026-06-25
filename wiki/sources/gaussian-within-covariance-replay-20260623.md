---
title: Gaussian Within-Covariance Replay 2026-06-23
type: source
status: reviewed
updated: 2026-06-24
sources:
  - raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/manifest.json
  - raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/node_covariance_variant_statistics.csv
  - raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/pipeline_covariance_variant_summary.csv
  - raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/pipeline_covariance_variant_by_sibling_gate.csv
  - raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/pipeline_guarded_within_covariance_focused.csv
  - raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/pipeline_guarded_full_within_active_time.csv
  - raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/pipeline_guarded_full_within_default_empirical_active_time.csv
  - raw/assets/benchmark-results/continuous_guarded_covariance_validation_20260623/focused_guarded_covariance_results.csv
tags:
  - source
  - benchmarks
  - gaussian
  - covariance
  - validation
---

# Gaussian Within-Covariance Replay 2026-06-23

## Summary

The within-covariance replay tests the hypothesis that continuous Gaussian
mean-shift failures come from using parent-total covariance as null covariance.
It compares parent-total covariance, unguarded pooled within-child covariance,
guarded/mixed within covariance, and guarded full within covariance while
separating branch-time active and no-time settings.

## Key Points

- Node-level statistics support the covariance hypothesis. With branch-time
  still active, replacing parent-total covariance by pooled within-child
  covariance changes the audited Gaussian projected p-values from marginal or
  closed values to extremely small values. For example, the clear Gaussian root
  `N118` projected p-value changes from about `2.36e-02` under parent-total
  covariance to `2.21e-71` under shrinkage-`0.05` within covariance.
- Unguarded within-child covariance is not a valid global replacement. It
  makes almost every edge significant, over-splits signal and null cases to
  singleton-like partitions, and can make the default empirical-null sibling
  inflation layer fail because no strict-null calibration records remain.
- Partial guarded mixtures (`rho=0.25` or `0.5`) improve p-values but do not
  overcome branch-time closure when branch time remains active. They recover
  the two Gaussian signal cases only in no-time mode, which parent-total
  covariance also does.
- Guarded full within-child covariance is the first tested variant that keeps
  branch-time active and fixes the focused Gaussian failures. Under
  fixed-coordinate BH, `guard_full_within_min8_active_time` recovers
  `gauss_clear_medium_continuous` (`K=4`, ARI `1.0`),
  `gauss_moderate_3c_continuous` (`K=3`, ARI `1.0`), keeps
  `gauss_null_large_continuous` closed (`K=1`, ARI `1.0`), and preserves
  `dim_consolidated_4c_24f_continuous` (`K=4`, ARI `1.0`).
- The same guarded full within-child covariance also works with the default
  `projected_wald_inflation` sibling layer on the four focused cases:
  `gauss_clear_medium_continuous` recovers `K=4`, `gauss_moderate_3c_continuous`
  recovers `K=3`, the Gaussian null remains `K=1`, and the consolidated
  24-feature dimensional Gaussian remains `K=4`.
- Later implementation validation refined that replay conclusion. In the
  current code path, guarded contrast covariance with the projected sibling
  layer can project away the newly whitened root contrast on
  `dim_consolidated_4c_24f_continuous`; the final opt-in method therefore uses
  fixed-coordinate BH as the active sibling gate and is documented in
  [[guarded-continuous-covariance-implementation-20260623]].
- The guarded rule tested here uses within-child covariance only at nodes whose
  immediate children have enough mass; tiny edges keep parent-total covariance.
  This prevents the null and leaf-level over-splitting seen under unguarded
  within covariance.

## Evidence

- `raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/manifest.json`
  records static provenance for the CSV bundle with
  `provenance_timestamp = 2026-06-24T20:27:01+02:00`; the original generation
  timestamp was not recorded in the CSV artifacts.
- `raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/node_covariance_variant_statistics.csv`
  records node-level parent-total versus within-child covariance statistics.
- `raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/pipeline_covariance_variant_summary.csv`
  records the first pipeline replay where unguarded within covariance exposed
  empirical-null support failure.
- `raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/pipeline_covariance_variant_by_sibling_gate.csv`
  records the unguarded within-covariance sibling-gate sweep showing severe
  over-splitting.
- `raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/pipeline_guarded_within_covariance_focused.csv`
  records the focused guarded partial-mixture sweep.
- `raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/pipeline_guarded_full_within_active_time.csv`
  records guarded full within-child covariance with fixed-coordinate BH and
  active branch time.
- `raw/assets/benchmark-results/gaussian_within_covariance_test_20260623/pipeline_guarded_full_within_default_empirical_active_time.csv`
  records guarded full within-child covariance with the default empirical-null
  sibling layer and active branch time.

## Links

- [[gaussian-inner-node-branch-time-debug-20260623]]
- [[full-benchmark-run-20260623]]
- [[projected-wald-statistic]]
