---
title: Julia allGO New GO-IC Tree Summary Plots 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - data/feature_matrices/feature_matrix_julia_allGO_new.tsv
  - scripts/analysis/go_ic_tree_summary_plots.py
  - scripts/analysis/split_go_ic_results_by_method.py
  - raw/assets/benchmark-results/julia_allGO_new_go_ic_tree_plots_20260618/raw_adaptive_kak/matrix_kak_probe_summary.csv
  - raw/assets/benchmark-results/julia_allGO_new_go_ic_tree_plots_20260618/adaptive_diffusion_kak/matrix_kak_diffusion_probe_summary.csv
  - raw/assets/benchmark-results/julia_allGO_new_go_ic_tree_plots_20260618/whole_adaptive_diffusion/cluster_assignments.csv
  - raw/assets/benchmark-results/julia_allGO_new_go_ic_tree_plots_20260618/allgo_new_quality_aware_go_ic_plots/README.md
  - raw/assets/benchmark-results/julia_allGO_new_go_ic_tree_plots_20260618/allgo_new_quality_aware_go_ic_plots/allgo_new_quality_aware_go_ic_tree_ranking.csv
  - raw/assets/benchmark-results/julia_allGO_new_go_ic_tree_plots_20260618/allgo_new_quality_aware_go_ic_plots/allgo_new_quality_aware_go_ic_all_tree_pages.pdf
  - raw/assets/benchmark-results/julia_allGO_new_go_ic_tree_plots_20260618/allgo_new_quality_aware_go_ic_by_method/README.md
  - raw/assets/benchmark-results/julia_allGO_new_go_ic_tree_plots_20260618/allgo_new_quality_aware_go_ic_by_method/allgo_new_quality_aware_go_ic_method_summary.csv
tags:
  - source
  - julia
  - allgo
  - go-ic
  - diffusion
  - plots
---

# Julia allGO New GO-IC Tree Summary Plots 2026-06-18

## Summary

The allGO-new matrix was scored across `31` tree assignments: raw cosine
subspace trees, adaptive-diffusion trees in cosine subspaces, the legacy c2ef
TF-IDF cosine components `2-5` split, and a full-matrix adaptive diffusion
tree. The summary script computes GO annotation Bernoulli information
criteria, GO-enrichment coherence, within-cluster TF-IDF cosine quality, full
allGO UMAP panels, tree-subspace embeddings, and tree leaf-order panels.

The display ranking is quality-aware before applying GO-IC. This is necessary
because raw GO-BIC can rank near-singleton overfit trees first. The raw GO-IC
ordering is preserved in `raw_go_ic_rank` for audit, while `display_rank` first
uses a quality tier and then GO-BIC within the tier.

## Key Points

- The final plot deck contains `31` pages, one for each scored tree.
- Method-separated result folders contain independent PDFs, PNG pages, and CSVs
  for `legacy_c2ef`, `whole_adaptive_diffusion`,
  `adaptive_diffusion_cosine_subspace`, and `raw_cosine_subspace`.
- The previous internal diagnostic name `kak` is not a separate method in the
  reader-facing outputs; those trees are cosine eigenspace/subspace diagnostics.
- Each page includes the full allGO UMAP colored by that tree, the tree
  subspace embedding, the tree leaf order, largest-cluster sizes, and cluster
  quality fields.
- The top quality-aware tree is the full-matrix adaptive diffusion tree:
  `20` clusters, `15/20` coherent clusters, no singleton clusters, GO-BIC
  active per gene `1475.898994`, and largest cluster fraction `0.568106`.
- The second-ranked tree is the legacy c2ef TF-IDF components `2-5` split:
  `19` clusters, `11/19` coherent clusters, one singleton gene, GO-BIC active
  per gene `1504.867882`, and largest cluster fraction `0.215947`.
- The top adaptive-diffusion cosine-subspace tree is
  `adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_02_05`: `40` clusters,
  `21/40` coherent clusters, GO-BIC active per gene `1687.966840`, and largest
  cluster fraction `0.232558`.
- The top raw cosine-subspace tree is
  `raw_cosine_subspace__tfidf__adaptive_modes_16_19`: `76` clusters, `23/76`
  coherent clusters, GO-BIC active per gene `2005.867846`, and largest cluster
  fraction `0.088040`.
- The raw GO-IC-only ranks put near-singleton trees first. For example, the
  top raw GO-IC tree has `592` clusters and a singleton-gene fraction near
  `0.985`, so it is demoted by the quality tier.
- The full adaptive diffusion runner produced `cluster_assignments.csv`,
  `cluster_sizes.csv`, `data_with_clusters.tsv`, and `embedding_coordinates.tsv`
  before exiting with status `134` during its own final plotting/summary path.
  The unified GO-IC plotting script used the completed assignments and
  regenerated the relevant tree, UMAP, and subspace panels.

## Evidence

- `allgo_new_quality_aware_go_ic_tree_ranking.csv` stores the display rank,
  raw GO-IC rank, GO-IC fields, cluster-count fields, coherence fields, and
  within-cluster TF-IDF cosine fields for every tree.
- `allgo_new_quality_aware_go_ic_by_method/*/*_tree_ranking.csv` stores
  method-isolated rankings with `method_rank`, `display_rank`, and
  `raw_go_ic_rank`.
- `allgo_new_quality_aware_go_ic_by_method/*/*_tree_pages.pdf` stores one
  method family per PDF, with a cover sentence defining the method and values.
- `allgo_new_quality_aware_go_ic_all_tree_pages.pdf` stores all `31` tree pages
  in quality-aware display order.
- `tree_pages/*.png` stores one PNG per tree page with filenames prefixed by
  display rank.
- `allgo_new_quality_aware_go_ic_top_trees.png` shows the top `12` trees by
  quality tier and GO annotation information criterion.
- `allgo_new_quality_aware_go_ic_quality_scatter_all_trees.png` and
  `allgo_new_quality_aware_go_ic_quality_scatter_plausible_trees.png` show
  GO-IC versus coherent-cluster fraction, with color indicating weighted
  within-cluster TF-IDF cosine.
- Legacy compatibility aliases are still present for earlier generic filenames.
- The old `go_ic_tree_ranking.csv` stores the display rank, raw GO-IC rank, GO-IC
  fields, cluster-count fields, coherence fields, and within-cluster TF-IDF
  cosine fields for every tree.

## Links

- [[julia-allgo-new-c2ef-cosine-subspace-validation-20260618]]
- [[ad-hoc-sibling-gate-selected-null-diffusion-comparison-20260617]]
