---
title: Goncalves Pancreas Progenitor Benchmark Run 2026-06-24
type: source
status: reviewed
updated: 2026-06-24
sources:
  - applications/scrna/goncalves_benchmark.py
  - applications/scrna/plots/pancreas_all_method_umap_clusters.py
  - applications/scrna/plots/pancreas_radial_trees_ggtree.R
  - applications/scrna/plots/pancreas_umap_tree_combo_ggtree.R
  - wiki/questions/pancreas-progenitor-dataset-selection-20260624.md
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/summary.md
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/method_metrics.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/edge_gate_distance_time_model_analysis.md
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_umap_tree_highlighting_audit.csv
tags:
  - benchmark
  - pancreas
  - scrna
  - progenitor
---

# Goncalves Pancreas Progenitor Benchmark Run 2026-06-24

## Summary

The Goncalves fetal pancreas benchmark was downloaded and executed after
network access was enabled. The UCSC `exprMatrix.tsv.gz` is processed/scaled
expression rather than raw counts, so the script preserves it as
`layers["input_expression"]`, uses metadata count fields for QC summaries,
selects high-variance genes, runs a conventional Scanpy PCA/neighbors/UMAP
workflow, and reuses the existing pancreas benchmark TBS/classical method
comparison.

## Key Points

- `applications/scrna/goncalves_benchmark.py` targets the UCSC
  fetal-pancreas expression matrix and metadata for the Goncalves et al. human
  fetal pancreas dataset.
- The downloaded local input paths are
  `raw/inbox/goncalves_human_pancreas_dev/fetal-pancreas/exprMatrix.tsv.gz`
  and `raw/inbox/goncalves_human_pancreas_dev/fetal-pancreas/meta.tsv`.
- The output directory defaults to
  `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/`.
- The benchmark uses all `1,465` cells and `8` `population` labels:
  `trunk`, `mesenchyme`, `proliferating`, `tip`, `blood`, `unknown`,
  `endocrine`, and `neurons`.
- The UCSC matrix contains negative non-integer values, so it is treated as
  processed/scaled expression, not count-like data.
- The TBS rows retain the repaired settings: sibling alpha `0.01`, edge alpha
  `0.001`, local adaptive projected-Wald dimensions at 90% contrast energy,
  topology-only versus recomputed-NNLS branch-time rows, and raw-linkage
  diagnostics.
- Louvain is the best classical row by V-measure: `12` clusters, purity
  `0.6846`, dominant-label recall `0.5092`, V-measure `0.4431`, and ARI
  `0.3264`.
- Adaptive-diffusion TBS gives `24` clusters with purity `0.6546`,
  dominant-label recall `0.5399`, V-measure `0.4100`, and ARI `0.2058`.
- Standardized-distance TBS topology gives `40` clusters and lower recall
  (`0.3857`), while standardized recomputed-NNLS and raw-linkage branch-time
  rows collapse to one cluster on this processed matrix.
- The UMAP/tree combo audit finds `114/114` colored TBS clusters are exact
  clades across the six TBS rows.

## Evidence

- `ruff check applications/scrna/goncalves_benchmark.py
  applications/scrna/plots/pancreas_all_method_umap_clusters.py` passed.
- `python -m py_compile applications/scrna/goncalves_benchmark.py
  applications/scrna/plots/pancreas_all_method_umap_clusters.py` passed.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/edge_gate_distance_time_model_analysis.md`
  records the topology-vs-time interpretation and branch-length summary with
  `Generated at: 2026-06-24T19:57:27+02:00`.
- `Rscript -e "parse('applications/scrna/plots/pancreas_radial_trees_ggtree.R');
  parse('applications/scrna/plots/pancreas_umap_tree_combo_ggtree.R')"` passed.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/summary.md`
  records the data shape, label counts, processed-expression handling, and
  method metrics.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/all_methods_umap_clusters_all_colored.png`
  shows every method cluster colored on the UMAP.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.png`
  shows each TBS UMAP beside its full radial tree.

## Links

- [[pancreas-progenitor-dataset-selection-20260624]]
- [[pancreas-scrna-clustering-benchmark-20260623]]
