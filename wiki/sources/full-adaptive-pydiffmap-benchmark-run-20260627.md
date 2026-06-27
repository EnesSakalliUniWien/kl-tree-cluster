---
title: Full Adaptive pydiffmap Benchmark Run 2026-06-27
type: source
status: reviewed
updated: 2026-06-27
sources:
  - benchmarks/results/run_20260627_0000_full_adaptive_pydiffmap_plots/full_benchmark_comparison.csv
  - benchmarks/results/run_20260627_0000_full_adaptive_pydiffmap_plots/full_benchmark_report.pdf
  - benchmarks/results/run_20260627_0000_full_adaptive_pydiffmap_plots/benchmark_relationship_report.md
  - benchmarks/results/run_20260627_0000_full_adaptive_pydiffmap_plots/benchmark_relationship_plots.pdf
  - benchmarks/results/run_20260627_0000_full_adaptive_pydiffmap_plots/failure_report.md
  - benchmarks/shared/runners/method_registry.py
  - benchmarks/shared/runners/tbs_diffusion_runner.py
tags:
  - source
  - benchmarks
  - diffusion
  - plots
---

# Full Adaptive pydiffmap Benchmark Run 2026-06-27

## Summary

The full 121-case benchmark was rerun with the default baseline methods but
with `tbs_diffusion_adaptive` replacing the Hamming nearest-neighbor diffusion
method. The adaptive row is displayed as `TBS (Adaptive pydiffmap Diffusion)`
and records `diffusion_method=adaptive_pydiffmap_diffusion` plus
`branch_length_optimization_method=linkage_ultrametric`.

## Key Points

- The run completed all `121` cases and wrote `121` per-case PDFs plus a
  `574`-page merged report under
  `benchmarks/results/run_20260627_0000_full_adaptive_pydiffmap_plots/`.
- Mean ARI over ok rows ranked methods as `kmeans` `0.863921`, `spectral`
  `0.833765`, `leiden` `0.816252`, `louvain` `0.812457`, `tbs` `0.775146`,
  `hdbscan` `0.600511`, `dbscan` `0.572938`,
  `tbs_diffusion_adaptive` `0.531196`, and `optics` `0.524159`.
- `tbs_diffusion_adaptive` produced `106` ok rows and `15` skips. Skip reasons
  were mostly small-neighborhood pydiffmap `kth(=6) out of bounds` failures and
  strict sibling-inflation calibration support failures.
- The adaptive pydiffmap runner fixes the quantized Gaussian overlap issue seen
  with `TBS (Hamming NN Diffusion)`: `gauss_overlap_3c_small_q3`, `q4`, and
  `q5` all return exactly `3` clusters with ARI `1.0000`.
- On `phylo_large`, adaptive pydiffmap returns ARI `0.9770` with `43` clusters
  for `phylo_large_32taxa`, and ARI `0.5612` with `61` clusters for
  `phylo_large_64taxa`. The 32-taxon row is strong; the 64-taxon row remains
  below spectral, OPTICS, and HDBSCAN, which all return ARI `1.0000`.
- Adaptive pydiffmap improves over plain TBS on several phylogenetic,
  categorical-overlap, and moderate-overlap rows, but it also has severe
  under-split rows where plain TBS is exact, including clear categorical,
  several binary moderate/hard rows, and selected method-proof rows.

## Evidence

- `benchmarks/results/run_20260627_0000_full_adaptive_pydiffmap_plots/full_benchmark_comparison.csv`
  records the full result table.
- `benchmarks/results/run_20260627_0000_full_adaptive_pydiffmap_plots/full_benchmark_report.pdf`
  is the merged plotted report with UMAP and radial-tree pages.
- `benchmarks/results/run_20260627_0000_full_adaptive_pydiffmap_plots/benchmark_relationship_report.md`
  and
  `benchmarks/results/run_20260627_0000_full_adaptive_pydiffmap_plots/benchmark_relationship_plots.pdf`
  record the relationship analysis.
- `benchmarks/results/run_20260627_0000_full_adaptive_pydiffmap_plots/failure_report.md`
  records the failure diagnosis.
- `benchmarks/shared/runners/method_registry.py` identifies
  `tbs_diffusion_adaptive` as `TBS (Adaptive pydiffmap Diffusion)`.
- `benchmarks/shared/runners/tbs_diffusion_runner.py` implements the adaptive
  pydiffmap distance and forwards branch-length optimization settings.

## Links

- [[full-radial-plots-benchmark-run-20260626]]
- [[categorical-adaptive-diffusion-focus-audit-20260626]]
- [[benchmark-pipeline-contract]]
