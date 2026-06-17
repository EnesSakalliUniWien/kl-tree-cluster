---
title: Legacy Internal Spectral Comparison Panel 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/legacy_internal_spectral_comparison_panel.py
  - tests/validation/151_test_legacy_internal_spectral_comparison_panel.py
  - tests/test_methodology_fixes.py
  - tests/pipeline/51_test_dispatch_contract.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/runners/dispatch.py
  - benchmarks/shared/runners/kl_runner.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/node_spectral_task.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/tree_estimator.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/marchenko_pastur.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_internal_spectral_comparison_panel/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_internal_spectral_comparison_panel/legacy_internal_spectral_comparison_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_internal_spectral_comparison_panel/legacy_internal_spectral_comparison_pairwise.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_internal_spectral_comparison_panel/legacy_internal_spectral_comparison_summary.csv
tags:
  - source
  - diagnostics
  - spectral
  - marchenko-pastur
  - benchmark
---

# Legacy Internal Spectral Comparison Panel 2026-06-16

## Summary

`legacy_internal_spectral_comparison_panel.py` compares the current leaf-only
MP spectral path against an opt-in reconstruction of the commit-era behavior
that appends descendant internal node distributions to each local spectral
matrix. The copied behavior is exposed as
`include_internal_barycenters=True` in the spectral estimator and as benchmark
method id `kl_legacy_internal_spectral_diagnostic`.

The one-replicate overlap benchmark shows that the legacy rows strongly change
the spectral diagnostics but do not change the resulting partitions on the
completed rows. This supports treating internal barycenters as a diagnostic
spectral perturbation, not as a production rescue rule.

## Key Points

- The legacy diagnostic mode adds descendant internal distributions as extra
  node-local spectral rows, then uses the augmented row count for the MP
  threshold while retaining the leaf count as `effective_independent_rows`.
- Targeted tests pin the row-count contract: a root with three leaves and one
  descendant internal node has `mp_threshold_rows=3` in current mode and
  `mp_threshold_rows=4` in legacy mode, with `effective_independent_rows=3` in
  both modes.
- The small benchmark ran `12` method rows and `6` paired comparisons over
  `overlap_part_4c_small`, `overlap_mod_4c_small`, and
  `overlap_heavy_4c_small_feat` for selected-null and signal roles.
- Two paired comparisons skipped in both variants because the strict sibling
  inflation model had no supported strict-null records. Four paired comparisons
  completed.
- Among completed rows, current and legacy partitions were identical:
  partition ARI was `1.0`, mean delta ARI was `0.0`, and mean delta cluster
  count was `0.0` for both selected-null and signal groups.
- The legacy diagnostic substantially increased spectral counts. Mean legacy
  minus current raw MP signal count was `191.0` for selected-null rows and
  `190.5` for signal rows. Mean MP threshold row count increased by `3347.0`
  for selected-null rows and `5158.0` for signal rows.
- Therefore the old internal-node row mechanism is visible in the spectral
  algebra but, on this compact overlapping benchmark, it does not explain or
  fix clustering fragmentation by itself.

## Evidence

- `node_spectral_task.py`, `tree_estimator.py`, and `marchenko_pastur.py`
  implement the opt-in internal-barycenter spectral task and worker path.
- `method_registry.py`, `dispatch.py`, and `kl_runner.py` expose the
  diagnostic as `kl_legacy_internal_spectral_diagnostic` for standard
  benchmark dispatch.
- `151_test_legacy_internal_spectral_comparison_panel.py` verifies the
  pairwise comparison and output-writing contract.
- `test_methodology_fixes.py` verifies that the copied internal-row mode
  changes MP threshold rows without changing effective independent rows.
- The output manifest records the run parameters: suite `binary`, one
  replicate, base seed `20260613`, edge alpha `0.001`, sibling alpha `0.01`,
  and the three overlap cases listed above.

## Links

- [[local-marchenko-pastur-rule]]
- [[selected-neighborhood-measurability-law]]
- [[spectral-vs-bandwidth-tradeoff-panel-20260616]]
