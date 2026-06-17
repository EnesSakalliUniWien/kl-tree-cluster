---
title: Legacy c2ef9a69 Root Tail Overlap Comparison 2026-06-17
type: source
status: reviewed
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/legacy_c2ef9a69_method_comparison_panel.py
  - benchmarks/shared/runners/legacy_commit_runner.py
  - kl_clustering_analysis/legacy_methods/commit_c2ef9a69/kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/pair_testing/sibling_null_prior_interpolation/adaptive_kernel_bandwidths.py
  - kl_clustering_analysis/legacy_methods/commit_c2ef9a69/kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/pair_testing/sibling_null_prior_interpolation/kernel_interpolation.py
  - kl_clustering_analysis/legacy_methods/commit_c2ef9a69/kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/pair_testing/sibling_null_prior_interpolation/child_prior_estimation.py
  - kl_clustering_analysis/legacy_methods/commit_c2ef9a69/kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/pair_testing/sibling_null_prior_interpolation/sibling_null_prior_interpolation.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_root_tail_overlap_comparison_20260617/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_root_tail_overlap_comparison_20260617/legacy_c2ef9a69_method_comparison_pairwise.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_root_tail_overlap_comparison_20260617/legacy_c2ef9a69_method_comparison_summary.csv
tags:
  - source
  - diagnostics
  - legacy
  - root
  - kernel
---

# Legacy c2ef9a69 Root Tail Overlap Comparison 2026-06-17

## Summary

This diagnostic reruns the full legacy `c2ef9a69` package on the seven overlap
case names used by the selected-root spectral-tail work:
`overlap_extreme_4c`, `overlap_heavy_4c_small_feat`,
`overlap_mod_4c_small`, `overlap_mod_6c_med`, `overlap_part_4c_small`,
`overlap_unbal_4c_small`, and `overlap_unbal_6c_med`, in both selected-null
and signal roles.

The old method completes more rows than the current strict method and improves
one signal row, but it also creates two selected-null false splits. This
supports the narrower conclusion that the old kernel/neighborhood layer is
useful evidence, not a valid direct replacement for the current fail-closed
root law.

## Key Points

- The run writes `14` paired rows and a two-row summary under
  `legacy_c2ef9a69_root_tail_overlap_comparison_20260617`.
- In selected-null rows, only `2/7` rows complete in both methods because the
  current strict path skips five rows. Among the completed pairs, the old
  method regresses `overlap_mod_4c_small` from one current cluster with ARI
  `1.0` to three legacy clusters with ARI `0.0`.
- Across all selected-null rows, the old method creates two false splits:
  `overlap_mod_4c_small` with three clusters and `overlap_mod_6c_med` with two
  clusters.
- In signal rows, `5/7` rows complete in both methods. The old method improves
  `overlap_unbal_4c_small` from three current clusters and ARI `0.387667` to
  four legacy clusters and ARI `0.471217`.
- The hardest signal row, `overlap_heavy_4c_small_feat`, remains a warning:
  the current method skips, while the old method returns one cluster with ARI
  `0.0`.
- The old sibling-null interpolation uses adaptive tree-neighborhood kernels:
  `tau_b` from stopping-edge distance, `tau_t` from nearest stable tree
  distances, `tau_s` from nearest signal tree distances, and `h_k` from the
  spread of stable log neighborhood scale. Stable-neighbor p-values are
  averaged with tree and log-scale kernels, then multiplied by a nearby-signal
  suppression factor and clipped to `[0, 1]`.
- The old smoother therefore estimates a tree-local selected-neighborhood
  prior. It is not a selected-root spectral-tail law and it does not condition
  on root tie rank, action edge, measured topology, or \(H_u\).

## Evidence

- `legacy_c2ef9a69_method_comparison_panel.py` runs current `kl` and full
  `kl_legacy_c2ef9a69` through the same benchmark dispatch on the seven root
  overlap cases.
- `legacy_c2ef9a69_method_comparison_summary.csv` reports selected-null
  `legacy_false_split_count = 2`, selected-null mean paired ARI delta
  `-0.5`, signal `legacy_improvement_count = 1`, and signal mean paired ARI
  delta `0.01671`.
- The legacy interpolation code computes adaptive bandwidths in
  `adaptive_kernel_bandwidths.py`, tree/log-scale stable-neighbor weights and
  signal suppression in `kernel_interpolation.py`, and writes the per-pair
  minimum child estimate back into `sibling_null_prior_from_edge_pvalue` in
  `sibling_null_prior_interpolation.py`.

## Links

- [[legacy-c2ef9a69-method-comparison-panel-20260616]]
- [[legacy-c2ef9a69-method-package-20260616]]
- [[root-selected-spectral-tail-law-with-legacy-overlay-20260617]]
- [[root-conditional-kernel-spectral-law]]
