---
title: Full Radial Plots Benchmark Run 2026-06-26
type: source
status: reviewed
updated: 2026-06-26
sources:
  - benchmarks/results/run_20260626_1506_full_radial_plots_layoutfix/full_benchmark_comparison.csv
  - benchmarks/results/run_20260626_1506_full_radial_plots_layoutfix/full_benchmark_report.pdf
  - benchmarks/results/run_20260626_1506_full_radial_plots_layoutfix/benchmark_relationship_report.md
  - benchmarks/results/run_20260626_1506_full_radial_plots_layoutfix/benchmark_relationship_plots.pdf
  - benchmarks/results/run_20260626_1506_full_radial_plots_layoutfix/failure_report.md
  - benchmarks/full/run.py
  - benchmarks/shared/plots/export.py
  - pyproject.toml
tags:
  - source
  - benchmarks
  - plots
  - umap
---

# Full Radial Plots Benchmark Run 2026-06-26

## Summary

The full benchmark was rerun on 2026-06-26 with plots enabled, benchmark tree
plots defaulting to radial layout, and UMAP embeddings cached under
`benchmarks/results/.embedding_cache`. The final layout-fixed run is
`benchmarks/results/run_20260626_1506_full_radial_plots_layoutfix/`.

## Key Points

- The run completed all `121` full-suite cases and wrote `121` per-case PDFs,
  a `577`-page merged `full_benchmark_report.pdf`, a relationship report, a
  relationship-plot PDF, and a failure report.
- `pygraphviz` is now part of the project visualization extras so NetworkX can
  use Graphviz layouts. Benchmark tree exports default to radial layout through
  `TBS_BENCHMARK_TREE_LAYOUT=radial`, with rectangular layout still available
  as an override.
- Dense tree pages suppress the cluster legend, which fixed the clipped legend
  seen in the first radial `phylo_large_64taxa` render. The final
  `phylo_large_64taxa` tree page renders as a readable radial tree, and its
  per-case PDF has four pages including UMAP comparison pages.
- UMAP plots are present in the final case PDFs. The embedding cache contains
  `121` `.npy` files, matching the number of benchmark cases and avoiding
  repeated embedding computation on plotted reruns.
- Mean ARI over ok rows ranked methods as `kmeans` `0.863921`, `spectral`
  `0.833765`, `leiden` `0.816252`, `louvain` `0.812457`, `tbs` `0.775146`,
  `hdbscan` `0.600511`, `dbscan` `0.572938`, `optics` `0.524159`, and
  `tbs_diffusion` `0.497085`.
- The quantized Gaussian overlap UMAP pages show three separated islands, but
  `tbs_diffusion`, now displayed as `TBS (Hamming NN Diffusion)`, returns two
  clusters on `gauss_overlap_3c_small_q4` and `q5` (`ARI=0.5698` in both),
  while plain `tbs` stays high (`ARI=1.0000` and `0.9801`). This is distinct
  from the separate `tbs_diffusion_adaptive` runner, now displayed as
  `TBS (Adaptive pydiffmap Diffusion)`.
- A focused extraction under
  `benchmarks/results/focus_gauss_overlap_q4_q5_20260626/` records the q4/q5
  matrices, assignments, cluster crosstabs, UMAP pages, node stats, and live
  p-value traces. It shows the q4/q5 `tbs_diffusion` failure is a two-island
  merge: true class `0` is isolated exactly, while true classes `1` and `2`
  are held together as one `200`-sample boundary. The decisive sibling
  corrected p-values on that `200`-sample branch are `0.01220272` for q4 and
  `0.01091002` for q5, just above the default `sibling_alpha=0.01`; the edge
  gates are open, so the stop is a sibling-gate boundary rather than missing
  edge signal. The extraction manifest and summary now also record
  `diffusion_method=hamming_nn_diffusion` and
  `branch_length_optimization_method=linkage_ultrametric`.
- `phylo_large_32taxa` and `phylo_large_64taxa` still skip plain `tbs` for
  calibration support, while `tbs_diffusion` returns `34` and `80` clusters
  with ARI `0.6849` and `0.4531`. The radial tree plot confirms the large
  phylogenetic topology is plotted, not dropped.

## Evidence

- `benchmarks/results/run_20260626_1506_full_radial_plots_layoutfix/full_benchmark_comparison.csv`
  records the complete result table for all `121` cases and `9` methods.
- `benchmarks/results/run_20260626_1506_full_radial_plots_layoutfix/full_benchmark_report.pdf`
  is the merged plotted report with the cover, manifest, case PDFs, and
  relationship plots.
- `benchmarks/results/run_20260626_1506_full_radial_plots_layoutfix/benchmark_relationship_report.md`
  and
  `benchmarks/results/run_20260626_1506_full_radial_plots_layoutfix/benchmark_relationship_plots.pdf`
  record the relationship analysis.
- `benchmarks/results/run_20260626_1506_full_radial_plots_layoutfix/failure_report.md`
  records the benchmark failure diagnosis.
- `benchmarks/full/run.py` sets a default embedding cache directory for plotted
  full runs when `TBS_EMBEDDING_CACHE_DIR` is unset.
- `benchmarks/shared/plots/export.py` sets benchmark tree plots to radial
  layout by default and disables dense-tree legends to avoid clipping.
- `pyproject.toml` lists `pygraphviz` in the visualization extras.
- `benchmarks/results/focus_gauss_overlap_q4_q5_20260626/key_pvalue_rows.csv`
  records the compact root and boundary p-value rows for q4/q5.
- `benchmarks/results/focus_gauss_overlap_q4_q5_20260626/cluster_crosstabs.csv`
  records the exact true-label composition of the TBS and `tbs_diffusion`
  cluster assignments.
- `benchmarks/shared/runners/method_registry.py` names `tbs_diffusion` as
  `TBS (Hamming NN Diffusion)` and `tbs_diffusion_adaptive` as
  `TBS (Adaptive pydiffmap Diffusion)`.
- `benchmarks/shared/util/method_execution.py` records
  `branch_length_optimization_method` on TBS-family benchmark rows, defaulting
  to `linkage_ultrametric`.

## Links

- [[full-benchmark-run-20260623]]
- [[categorical-adaptive-diffusion-focus-audit-20260626]]
- [[benchmark-pipeline-contract]]
