---
title: Graphtools Adaptive K NNLS Focus Benchmark 2026-06-30
type: source
status: reviewed
updated: 2026-06-30
sources:
  - reports/graphtools_adaptive_k_20260630/focus_diffusion_nnls_adaptive_k_panel.csv
  - reports/graphtools_adaptive_k_20260630/focus_diffusion_nnls_adaptive_k_metrics_compact.csv
  - reports/graphtools_adaptive_k_20260630/focus_graphtools_neighbor_metadata.csv
  - reports/graphtools_adaptive_k_20260630/focus_adaptive_k_deltas_vs_nnls_baselines.csv
  - benchmarks/shared/runners/tbs_diffusion_runner.py
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/runners/dispatch.py
  - benchmarks/README.md
tags:
  - source
  - benchmarks
  - diffusion
  - graphtools
  - adaptive-k
  - nnls
---

# Graphtools Adaptive K NNLS Focus Benchmark 2026-06-30

## Summary

The optional GPL graphtools diffusion path now includes
`tbs_diffusion_graphtools_adaptive_nnls`, a fixed-topology NNLS branch-time
method with an adaptive kNN support profile. The registered profile is
`fragmentation_guard`: it starts from the requested K, keeps that K when the
graph is connected or has only substantial disconnected components, and only
increases K along the configured grid when the graph shows small-fragment
behavior.

## Key Points

- The focused panel covers the broad quality regressions
  `cat_overlap_3cat_4c`, `overlap_unbal_4c_small`,
  `overlap_mod_4c_small`, and `dim_consolidated_4c_24f`, plus the
  fragmentation-sensitive cases `cat_highd_3cat_500feat`,
  `gauss_overlap_3c_small`, and `gauss_overlap_8c_highd`.
- Compared methods are `tbs_diffusion_adaptive`,
  `tbs_diffusion_adaptive_nnls`, `tbs_diffusion_graphtools`,
  `tbs_diffusion_graphtools_nnls`, and
  `tbs_diffusion_graphtools_adaptive_nnls`.
- On all seven focused cases, `fragmentation_guard` selected K `10`, so the
  adaptive-K graphtools NNLS metrics match fixed-K graphtools NNLS on this
  panel. Metadata records connected status for the broad and low-dimensional
  cases, and stable-component status for `cat_highd_3cat_500feat` and
  `gauss_overlap_8c_highd`.
- Versus pydiffmap NNLS, adaptive-K graphtools NNLS improves ARI on
  `cat_overlap_3cat_4c` (`+0.031426`), `overlap_unbal_4c_small`
  (`+0.056588`), `overlap_mod_4c_small` (`+0.049250`),
  `cat_highd_3cat_500feat` (`+0.901540`), and `gauss_overlap_8c_highd`
  (`+0.065700`).
- It remains worse than pydiffmap NNLS on `dim_consolidated_4c_24f`
  (`-0.126283` ARI, found `9` clusters versus true `4`) and
  `gauss_overlap_3c_small` (`-0.182030` ARI, found `8` clusters versus true
  `3`).
- Independent metrics reinforce the split: `cat_highd_3cat_500feat` and
  `gauss_overlap_8c_highd` move to the expected effective cluster count and
  higher silhouette/Calinski-Harabasz scores, while
  `dim_consolidated_4c_24f` and `gauss_overlap_3c_small` remain
  over-fragmented with weaker internal scores than pydiffmap NNLS.
- The method still infers topology by average linkage on graphtools diffusion
  distances. NNLS optimizes fixed-topology branch lengths only; it does not
  infer a new tree topology.

## Evidence

- `reports/graphtools_adaptive_k_20260630/focus_diffusion_nnls_adaptive_k_panel.csv`
  records all focused method-case rows.
- `reports/graphtools_adaptive_k_20260630/focus_diffusion_nnls_adaptive_k_metrics_compact.csv`
  records supervised and independent cluster-quality metrics.
- `reports/graphtools_adaptive_k_20260630/focus_graphtools_neighbor_metadata.csv`
  records selected K, graph component counts, component sizes, and graphtools
  kernel sparsity.
- `reports/graphtools_adaptive_k_20260630/focus_adaptive_k_deltas_vs_nnls_baselines.csv`
  records deltas versus fixed graphtools NNLS and pydiffmap NNLS.
- `benchmarks/shared/runners/tbs_diffusion_runner.py` implements the
  adaptive-K resolver and graphtools metadata.
- `benchmarks/shared/runners/method_registry.py` registers
  `tbs_diffusion_graphtools_adaptive_nnls`.
- `benchmarks/shared/runners/dispatch.py` forwards the adaptive neighbor
  profile and grid through the benchmark method contract.

## Links

- [[full-graphtools-nnls-benchmark-run-20260630]]
- [[full-adaptive-nnls-benchmark-run-20260628]]
- [[adaptive-nnls-regression-skip-analysis-20260628]]
