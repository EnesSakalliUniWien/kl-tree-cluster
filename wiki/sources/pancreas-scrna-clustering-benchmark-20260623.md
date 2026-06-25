---
title: Pancreas scRNA Clustering Benchmark 2026-06-23
type: source
status: reviewed
updated: 2026-06-24
sources:
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/summary.md
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/method_metrics.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/celltype_fragmentation_by_method.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/cluster_composition_by_method.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/edge_gate_distance_time_model_analysis.md
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_tree_branch_length_summary.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_branch_time_sensitivity.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_branch_time_sensitivity.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_cluster_radial_tree_ggtree_outputs.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_cluster_radial_tree_highlighting_audit.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.pdf
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_cluster_radial_tree_combo_all_clusters_ggtree.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_cluster_radial_tree_combo_all_clusters_ggtree.pdf
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_cluster_radial_tree_combo_ggtree.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_cluster_radial_tree_combo_ggtree.pdf
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_cluster_radial_tree_combo_ggtree_outputs.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_tree_highlighting_audit.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/all_methods_umap_clusters_all_colored.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/all_methods_umap_clusters_all_colored.pdf
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/all_methods_umap_clusters_all_colored_summary.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_readable_umap_clusters_all_colored.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_readable_umap_clusters_all_colored.pdf
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_readable_umap_clusters_all_colored_summary.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_readable_umap_clusters_ge50.png
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_readable_umap_clusters_ge50_summary.csv
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_failure_diagnostic.md
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/manifest.json
  - tree_break_selection/tree/optimized_branch_lengths.py
  - tree_break_selection/hierarchy_analysis/statistics/contrast_covariance.py
  - scripts/pancreas_scrna_cluster_benchmark.py
  - scripts/plot_pancreas_tbs_cluster_radial_trees_ggtree.R
  - scripts/plot_pancreas_tbs_umap_tree_combo_ggtree.R
  - scripts/plot_pancreas_all_method_umap_clusters.py
  - scripts/plot_pancreas_tbs_readable_umap_clusters.py
  - tests/tree/test_optimized_branch_lengths.py
  - tests/statistics/46_test_continuous_covariance_numerical_psd.py
tags:
  - source
  - benchmark
  - scrna
  - pancreas
  - clustering
---

# Pancreas scRNA Clustering Benchmark 2026-06-23

## Summary

This benchmark downloads the public Scanpy human pancreas AnnData object,
recomputes a classical Scanpy latent-space workflow, and compares clustering
methods on a deterministic stratified subset of clear cell-type labels. The
benchmark is post-count: it does not run FASTQ-to-count generation, scDblFinder,
or ambient-RNA correction because the input is a prepackaged AnnData object
without channel-level empty-droplet inputs.

## Key Points

- The downloaded object has `14,693` cells, `2,448` genes, `4` batches, and
  `24` cell-type labels.
- The classical workflow recomputes scaling, `30`-component PCA, a
  `15`-neighbor graph, UMAP, and Leiden clustering.
- The benchmark subset is capped at `2,500` cells and excludes ambiguous labels:
  `not applicable`, `dropped`, `co-expression`, `unclear`, `unclassified`,
  `unclassified endocrine`, and `MHC class II`.
- ARI is not the main diagnostic for this hierarchy. The current report
  emphasizes split/merge metrics: weighted cluster purity, dominant-cluster
  recall per cell type, overcluster ratio, homogeneity, completeness, and
  V-measure.
- The original guarded continuous TBS path collapsed to one cluster because the
  sibling gate saw strong sibling difference while the edge gate stayed closed
  under a too-small local projection. A later full-rank edge-floor workaround was
  rejected because it coupled edge math to a fixed-coordinate sibling diagnostic.
- The repaired run makes both production gates projected/adaptive. The stored
  MP/floor dimension remains `2`, while each edge or sibling projected-Wald
  contrast uses the shortest local PCA prefix explaining `90%` of that
  contrast's projected energy. The benchmark metadata records
  `adaptive_projection_dimension_energy_fraction = 0.9`.
- The rerun separates topology from time. Adaptive-diffusion TBS gives `43`
  clusters, weighted purity `0.9288`, dominant-cluster recall `0.4840`, split
  error `0.5160`, and V-measure `0.6620`, slightly ahead of Leiden/Louvain by
  V-measure on this subset. Topology-only standardized-PCA linkage TBS gives
  `45` clusters, purity `0.7928`, recall `0.5800`, split error `0.4200`, and
  V-measure `0.6020`. The explicitly labeled raw-linkage branch-time diagnostic
  collapses too aggressively to `9` clusters with purity `0.3772`, recall
  `0.9932`, and V-measure `0.3124`.
- A native fixed-topology NNLS branch-length optimizer was added and rerun as
  the branch-time linkage-tree method. It recomputes every edge length on the
  selected topology by fitting non-negative additive path lengths to continuous
  PCA pairwise distances. On the standardized-PCA topology, recomputed-NNLS
  branch time gives the same final clustering as the topology-only row: `45`
  clusters, purity `0.7928`, dominant-cluster recall `0.5800`, and V-measure
  `0.6020`. On adaptive diffusion topology, recomputed-NNLS branch time gives
  the same final clustering as the adaptive-diffusion topology-only row: `43`
  clusters, purity `0.9288`, recall `0.4840`, and V-measure `0.6620`.
- The recomputed-NNLS result isolates the failure: raw linkage branch lengths
  are not a valid stochastic edge-time model for this benchmark. Raw linkage
  heights are now an explicitly allowed diagnostic/negative-control path only;
  production-facing linkage branch-time requires recomputed fixed-topology
  lengths.
- The two NNLS branch-length fits used `50,000` sampled leaf pairs. The
  standardized-PCA topology fit has residual RMSE `0.3018`, residual MAE
  `0.2017`, and optimizer time `3.2815` seconds. The adaptive-diffusion topology
  fit has residual RMSE `0.3075`, residual MAE `0.2013`, and optimizer time
  `3.3597` seconds.
- The adaptive-diffusion NNLS row originally exposed a numerical covariance
  issue: a tiny negative continuous covariance eigenvalue, accepted as
  numerical PSD before branch-time scaling, could become Cholesky-negative after
  scaling. The contrast covariance path now symmetrizes and jitters continuous
  full covariance blocks after scaling; the regression test covers this case.
- The non-ARI conclusion is that the adaptive projected repair reduces the
  previous extreme over-fragmentation, but the split/merge balance is still
  topology-dependent. Adaptive diffusion is the strongest TBS setting in this
  rerun; branch-time variance scaling is not a production fix because it trades
  splits for large merges.
- The benchmark now writes `method_umap_clusters.png`, a subset UMAP colored by
  curated labels and each method's cluster assignment, including the corrected
  TBS labels. A corrected readable comparison,
  `all_methods_umap_clusters_all_colored.png`, colors every assigned cluster
  for every method and only restricts labels to clusters with at least `50`
  cells.
- The benchmark also writes `method_split_merge_diagnostic.png`,
  `celltype_fragmentation_by_method.csv`, `cluster_composition_by_method.csv`,
  TBS dendrograms, per-edge tree diagnostics, and branch-length histograms.
- Whole-tree circular `ggtree` cluster plots were regenerated for all six TBS
  topologies. Each radial plot contains `2,500` tips, `2,499` internal nodes,
  and `4,998` edges, with final cluster subtrees colored and shared ancestors
  grey. PNG and PDF versions are recorded in
  `tbs_cluster_radial_tree_ggtree_outputs.csv`.
- A readable TBS UMAP was regenerated with every final cluster colored. Only
  clusters of at least `50` cells are labeled, so unlabeled small clusters
  remain visible as assigned clusters rather than grey fragments. Topology-only
  and recomputed-NNLS branch-time TBS have `11` labeled large clusters,
  raw-linkage branch-time diagnostic TBS has `3`, and adaptive diffusion TBS has
  `12`.
- The combined UMAP-plus-tree `ggtree` panel now places each method's UMAP and
  full radial tree side by side in
  `tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.png`. Every final TBS
  cluster is colored on the UMAP and on its terminal tree subtree; only clusters
  of at least `50` cells are labeled, and shared tree ancestors remain grey. The
  UMAP panels use tight global limits, larger points, no axis chrome, and a more
  balanced UMAP/tree width allocation so cluster structure is readable. The
  radial tree panels use a display-only sqrt transform with a `99%` cap and
  small positive floor so long NNLS edges do not compress the whole tree; raw
  edge lengths remain in the `tree_edges.csv` files. The older
  `tbs_umap_cluster_radial_tree_combo_all_clusters_ggtree.png` and
  `tbs_umap_cluster_radial_tree_combo_ggtree.png` paths are overwritten as
  aliases to the same scaled all-cluster figure. Direct clade audits confirmed
  `226/226` plotted clusters as exact tree clades under the `L#` leaf mapping.
- A supervised fixed-topology branch-time sensitivity scan does not make length
  scaling a production rule, but it shows the tradeoff clearly. On adaptive
  diffusion topology, `linear_scale_2` gives the best scanned V-measure,
  `0.6700`, with `36` clusters, purity `0.9284`, and dominant-cluster recall
  `0.4992`. On standardized-PCA topology, the scanned length transforms did not
  improve over the unscaled adaptive-projected topology row.

## Evidence

- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/summary.md`
  records the dataset, QC caveats, benchmark design, and result table.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/method_metrics.csv`
  stores the machine-readable metric table.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/celltype_fragmentation_by_method.csv`
  stores per-cell-type fragmentation metrics, including effective cluster count
  and dominant-cluster recall.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/cluster_composition_by_method.csv`
  stores per-cluster purity and dominant-label composition.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/edge_gate_distance_time_model_analysis.md`
  records the topology-vs-time interpretation, adaptive diffusion metadata, and
  branch-length summary with `Generated at: 2026-06-24T20:07:07+02:00`.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_tree_branch_length_summary.csv`
  records the six TBS tree summaries, `spectral_minimum_dimension = 2`,
  `adaptive_projection_dimension_energy_fraction = 0.9`, recomputed-NNLS
  residual diagnostics, and the raw-linkage diagnostic rows.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_branch_time_sensitivity.csv`
  records the fixed-topology branch-time transform scan over Tree-BH and
  traversal decisions.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_branch_time_sensitivity.png`
  plots the branch-time transform purity/recall tradeoff.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_cluster_radial_tree_ggtree_outputs.csv`
  records the whole-tree circular `ggtree` cluster PNG/PDF outputs for
  topology-only, recomputed-NNLS branch-time, raw-linkage branch-time
  diagnostic, adaptive-diffusion topology, adaptive-diffusion recomputed-NNLS
  branch-time, and adaptive-diffusion raw-linkage branch-time diagnostic TBS
  trees.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_cluster_radial_tree_highlighting_audit.csv`
  and
  `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_tree_highlighting_audit.csv`
  verify that every colored tree cluster is an exact clade.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.png`
  records the combined side-by-side UMAP and full radial tree figure with every
  final cluster colored on the UMAP and scaled for readability.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/all_methods_umap_clusters_all_colored.png`
  records the corrected all-method UMAP comparison. Every assigned cluster is
  colored for the curated labels, all six TBS rows, Leiden, Louvain, K-means,
  spectral clustering, and HDBSCAN; only labels are thresholded by cluster
  size.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_readable_umap_clusters_all_colored.png`
  records the standalone TBS UMAP comparison with the same all-cluster color
  policy. The legacy `tbs_readable_umap_clusters_ge50.png` path is overwritten
  as a compatibility alias to this all-colored rendering.
- `scripts/plot_pancreas_tbs_cluster_radial_trees_ggtree.R` is the reproducible
  R plotting script that converts TBS edge tables into `ape::phylo` objects and
  renders final clusters on whole radial trees with `ggtree`.
- `scripts/plot_pancreas_tbs_umap_tree_combo_ggtree.R` renders the combined
  UMAP-plus-tree panel with all final clusters colored and labels restricted to
  larger clusters.
- `scripts/plot_pancreas_all_method_umap_clusters.py` renders the corrected
  all-method UMAP comparison with every assigned cluster colored.
- `scripts/plot_pancreas_tbs_readable_umap_clusters.py` renders the TBS UMAP
  cluster panel that colors every final cluster and labels only larger clusters.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/tbs_failure_diagnostic.md`
  records the original root traversal p-values, child composition, and
  edge-projection sensitivity behind the corrected TBS run; its original
  generation timestamp was not recorded, and a provenance timestamp was added
  at `2026-06-24T20:22:42+02:00`.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/method_umap_clusters.png`
  visualizes the subset UMAP by curated labels and benchmark cluster
  assignments.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/method_split_merge_diagnostic.png`
  visualizes merge control versus split control with overcluster ratio.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/manifest.json`
  stores parameters, package versions, provenance, and runtime metadata.
- `scripts/pancreas_scrna_cluster_benchmark.py` is the reproducible runner.
- `tree_break_selection/tree/optimized_branch_lengths.py` implements the native
  fixed-topology NNLS branch-length fit used in the two new benchmark rows.
- `tests/tree/test_optimized_branch_lengths.py` and
  `tests/statistics/46_test_continuous_covariance_numerical_psd.py` cover the
  branch-length optimizer and the continuous covariance scaling repair.

## Links

- [[project-overview]]
- [[current-adaptive-diffusion-subspace-tree-pipeline]]
