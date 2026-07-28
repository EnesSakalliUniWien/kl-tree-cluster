---
title: Julia Tree Estimator Run 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - raw/inbox/julia-tree-estimator-run-20260614.md
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/summary.csv
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/manifest.json
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/umap_plots/julia_tree_estimators_umap_top_clusters.png
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/umap_plots/julia_tree_estimators_umap_cluster_size_classes.png
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/umap_plots/julia_tree_estimators_umap_top_clusters_clear.png
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/umap_plots/julia_tree_estimators_umap_cluster_size_classes_clear.png
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/umap_plots/julia_full_umap_reference_endotypes.png
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/umap_plots/julia_full_umap_method_cluster_sizes.png
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/umap_plots/julia_full_umap_interactive_methods.html
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/umap_plots/julia_full_umap_interactive_methods_standalone.html
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/clustering_diagnostics/cluster_size_summary.csv
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/clustering_diagnostics/reference_recovery_summary.csv
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/clustering_diagnostics/feature_coherence_summary.csv
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/clustering_diagnostics/clustering_diagnostic_review_panel.png
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/alpha_audit/posthoc_alpha_sensitivity_summary.csv
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/alpha_audit/kl_posthoc_alpha_sensitivity_heatmaps.png
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/alpha_audit/kl_neighbor_joining_posthoc_alpha_sensitivity_heatmaps.png
  - raw/assets/benchmark-results/julia_tree_estimators_20260614/analysis_debug_report.md
  - applications/endotypes/reports/cluster_diagnostics_panel.py
  - data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_nmi_ordered_20260610/kak_signal_adaptive_reference_labels.csv
tags:
  - source
  - julia
  - tree
  - benchmark
---

# Julia Tree Estimator Run 2026-06-14

## Summary

The combined Julia GO binary matrix was run through the baseline Hamming
average-linkage TBS tree, MAD-rooted neighbor joining, and a bounded IQ-TREE 3
fast likelihood tree rooted by MAD. All three methods remain highly fragmented
on the full `703 x 14766` matrix. Neighbor joining reduces fragmentation most
and gives the best reference ARI, but the reference NMI remains near the same
range as the baseline. The run supports the current interpretation: changing
the tree estimator/root orientation helps topology shape but does not solve the
selected traversal null-law problem.

## Key Points

- Baseline `tbs` produced `670` clusters, reference ARI `0.001812`, and
  reference NMI `0.633958` over `262` matched Julia-reference genes.
- `kl_neighbor_joining` produced `494` clusters, reference ARI `0.021596`, and
  reference NMI `0.616780`.
- `tbs_iqtree3_fast` used IQ-TREE 3.1.2 with `JC2 --fast`, produced `566`
  clusters, reference ARI `0.015612`, and reference NMI `0.629320`.
- The IQ-TREE fast treefile was rooted by minimum ancestor deviation before
  the normal TBS gate/traversal layer ran.
- UMAP overlays show the same fragmentation visually: baseline TBS is almost
  entirely singleton clusters, neighbor joining creates the largest visible
  multi-gene islands, and IQ-TREE fast/MAD creates many small `2`--`5` gene
  groups but no large coherent endotype-scale regions.
- The high-contrast UMAP variants are the preferred visual evidence because
  the first categorical panel reused many similar colors across hundreds of
  tiny clusters. The clear panels highlight only the largest non-singleton
  clusters and encode cluster-size classes separately.
- The clustering-diagnostic panel separates fragmentation, reference recovery,
  UMAP compactness, and active-feature Jaccard coherence. It shows that
  neighbor joining groups substantially more genes, but its grouped clusters
  have weaker active-feature overlap than the tiny baseline TBS clusters.
- The full UMAP exports show all `703` genes. Static full views include
  reference-endotype colors and per-method cluster-size classes; the
  interactive HTML views expose gene, cluster ID, cluster size, and reference
  label on hover. The standalone interactive HTML is preferred for local review
  because the first HTML export references Plotly from the CDN.
- The post-hoc alpha audit shows that edge alpha is mostly saturated on this
  matrix, while sibling alpha controls the fragmentation/scatter tradeoff.
  Stricter sibling alpha merges more genes in the neighbor-joining tree and
  improves ARI modestly, but those merged clusters have larger UMAP radii, so
  alpha tuning alone does not recover UMAP-local endotype basins.
- The debug report validates row counts, assignment sizes, and coordinate
  joins, then records two limitations: the original CDN-backed interactive HTML
  can fail under local browser restrictions, and the alpha audit retresholds
  saved q-values rather than rerunning the full gate pipeline at every alpha.
- The result is a topology/rooting diagnostic, not production calibration
  evidence. It does not remove same-data tree selection, selected root, or
  selected pass-through family effects.

## Evidence

- `raw/assets/benchmark-results/julia_tree_estimators_20260614/summary.csv`
  stores the method rows, timings, cluster counts, and reference scores.
- `raw/assets/benchmark-results/julia_tree_estimators_20260614/*_assignments.csv`
  stores per-gene cluster assignments for the three completed methods.
- `raw/assets/benchmark-results/julia_tree_estimators_20260614/julia_iqtree_fast.treefile`
  stores the IQ-TREE fast Newick tree used before MAD rooting.
- `raw/assets/benchmark-results/julia_tree_estimators_20260614/umap_plots/`
  stores the UMAP coordinates, original overview panels, high-contrast
  top-cluster panels, per-method top-15 cluster pages, and cluster-size
  fragmentation panels, plus full static and interactive UMAP views.
- `raw/assets/benchmark-results/julia_tree_estimators_20260614/clustering_diagnostics/`
  stores reusable clustering analytics: size bins, effective cluster counts,
  reference recovery and fragmentation, method-cluster reference purity, UMAP
  compactness, feature coherence, top non-singleton clusters, and the combined
  review panel.
- `raw/assets/benchmark-results/julia_tree_estimators_20260614/alpha_audit/`
  stores default node-level gate annotations and a post-hoc alpha sensitivity
  replay over edge and sibling thresholds. The replay retresholds saved q-value
  columns and should be treated as an alpha-sensitivity audit, not a fresh
  production calibration run.
- `raw/assets/benchmark-results/julia_tree_estimators_20260614/analysis_debug_report.md`
  records the analysis-debugging checks and resulting limitations.

## Links

- [[phylogenetic-tree-builders-20260614]]
- [[adaptive-cosine-kak-benchmark-probe-20260605]]
- [[selected-root-selected-family-traversal-literature-20260614]]
- [[open-mathematical-questions]]
