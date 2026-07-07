---
title: Graphtools Adaptive K Tree Inference Focus Benchmark 2026-06-30
type: source
status: reviewed
updated: 2026-06-30
sources:
  - reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_panel.csv
  - reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_metrics_compact.csv
  - reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_method_summary.csv
  - reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_deltas_vs_average.csv
  - reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_case_best_summary.csv
  - tree_break_selection/tree/io.py
  - benchmarks/shared/runners/tbs_runner.py
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
  - tree-inference
  - nnls
---

# Graphtools Adaptive K Tree Inference Focus Benchmark 2026-06-30

## Summary

The adaptive-K graphtools NNLS method was tested across all applicable
distance-tree inference methods currently exposed by the benchmark runner:
average linkage, complete linkage, weighted linkage, single linkage, centroid
linkage, median linkage, Ward linkage, and MAD-rooted neighbor-joining. IQ-TREE
3 is installed locally, but it infers trees from raw encoded feature alignments
rather than from a graphtools diffusion distance, so it was not included in
this graphtools-distance topology panel.

Centroid and median linkage can emit nonmonotone merge heights. The benchmark
therefore constructs their topology with placeholder branch lengths only when
fixed-topology NNLS is active, then lets NNLS fit branch lengths before
branch-time statistics are used.

## Key Points

- The panel covers the same seven cases as the adaptive-K focus benchmark:
  `cat_overlap_3cat_4c`, `overlap_unbal_4c_small`,
  `overlap_mod_4c_small`, `dim_consolidated_4c_24f`,
  `cat_highd_3cat_500feat`, `gauss_overlap_3c_small`, and
  `gauss_overlap_8c_highd`.
- All `56` method-case rows completed with status `ok`.
- Mean ARI by topology was weighted linkage `0.831689`, centroid linkage
  `0.771060`, Ward linkage `0.769584`, average linkage `0.764584`,
  neighbor joining `0.752394`, median linkage `0.719972`, complete linkage
  `0.711700`, and single linkage `0.530392`.
- Average linkage won by ARI on `3` of `7` cases:
  `overlap_unbal_4c_small`, `cat_highd_3cat_500feat`, and
  `gauss_overlap_8c_highd`.
- Centroid linkage was best on `cat_overlap_3cat_4c` with ARI `0.936204`,
  found `4` clusters, and macro F1 `0.974990`.
- Ward linkage was best on `overlap_mod_4c_small` with ARI `0.921729`, found
  `4` clusters, and the strongest independent metrics for that case.
- Neighbor joining repaired `dim_consolidated_4c_24f`: ARI `0.686774`, found
  `4` clusters, effective cluster count `3.945513`, and silhouette `0.074451`.
  Average linkage remained over-fragmented on that case with ARI `0.403520`
  and found `9` clusters.
- Weighted, single, centroid, and neighbor joining repaired
  `gauss_overlap_3c_small` to ARI/NMI `1.0` with `3` clusters, while average
  linkage found `8` clusters.
- Single linkage is not a safe global replacement: it collapsed
  `overlap_unbal_4c_small`, `overlap_mod_4c_small`, and
  `dim_consolidated_4c_24f` to one cluster.
- Weighted linkage, Ward linkage, and neighbor joining all reduce mean
  cluster-count absolute error to `0.571429`, but neighbor joining is much
  slower on large rows; tree build time reached about `94` seconds on
  `gauss_overlap_8c_highd`.

## Evidence

- `reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_panel.csv`
  records the complete focused benchmark.
- `reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_metrics_compact.csv`
  records supervised and independent cluster-quality metrics.
- `reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_method_summary.csv`
  records mean metrics by topology method.
- `reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_deltas_vs_average.csv`
  records per-case deltas against average linkage.
- `reports/graphtools_adaptive_k_tree_inference_20260630/focus_tree_inference_case_best_summary.csv`
  records per-case best ARI and best internal-metric topology.
- `tree_break_selection/tree/io.py` provides the topology-only linkage
  constructor used for nonmonotone centroid/median linkage when NNLS will
  replace placeholder branch lengths.
- `benchmarks/shared/runners/tbs_runner.py` uses that fallback only for
  nonmonotone linkage heights under fixed-topology NNLS.
- `benchmarks/shared/runners/tbs_diffusion_runner.py` passes tree-builder,
  rooting, and linkage parameters through the graphtools diffusion runner.
- `benchmarks/shared/runners/method_registry.py` registers the complete,
  weighted, single, centroid, median, Ward, and neighbor-joining adaptive-K
  graphtools NNLS variants.

## Links

- [[graphtools-adaptive-k-tree-consensus-focus-benchmark-20260630]]
- [[graphtools-adaptive-k-nnls-focus-benchmark-20260630]]
- [[full-graphtools-nnls-benchmark-run-20260630]]
- [[adaptive-nnls-regression-skip-analysis-20260628]]
