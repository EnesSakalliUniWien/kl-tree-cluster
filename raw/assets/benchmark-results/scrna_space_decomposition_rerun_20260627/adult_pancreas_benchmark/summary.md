# Pancreas scRNA clustering benchmark

Generated at: 2026-06-27T21:27:32+02:00

## Data

- Source: Scanpy pancreas AnnData tutorial object downloaded from `https://www.dropbox.com/s/qj1jlm9w10wmt0u/pancreas.h5ad?dl=1`.
- Local raw object: `/Users/berksakalli/Projects/kl-te-cluster/raw/inbox/pancreas.h5ad`.
- Shape: 14693 cells x 2448 genes.
- Batches: 4.
- Cell-type labels: 24.

Top labels:

- alpha: 4214
- beta: 3354
- ductal: 1804
- acinar: 1368
- not applicable: 1154
- delta: 917
- gamma: 571
- endothelial: 289
- activated_stellate: 284
- dropped: 178
- quiescent_stellate: 173
- mesenchymal: 80

## Classical pipeline

The run recomputed a conventional Scanpy latent-space workflow on the downloaded
AnnData object: scaling, PCA (30 components), 15-neighbor graph, UMAP, and
Leiden clustering at resolution 1.0. The input object is a prepackaged post-count
AnnData object, so FASTQ-to-count generation was not performed.

QC notes:

- Raw snapshot present: True.
- Mitochondrial genes detected by `MT-` prefix: 0.
- Doublet status: not_run_prepackaged_post_count_object; R/Bioconductor scDblFinder was not available in this Python-only benchmark run.
- Ambient RNA status: not_run_no_empty_droplet_channel_inputs_in_prepackaged_h5ad.

## Benchmark design

The benchmark uses the first 30 recomputed PCs and a deterministic
stratified subset capped at 2500 cells, excluding ambiguous labels:
MHC class II, co-expression, dropped, not applicable, unclassified, unclassified endocrine, unclear. Metrics compare cluster assignments to
the curated `celltype` labels.

TBS is run at sibling alpha `0.01` and edge alpha `0.001`. Both production gates
use projected-Wald tests with a repaired local adaptive dimension rule. The
stored MP/floor dimension remains `2`, but each edge or sibling contrast can use
the shortest local PCA prefix explaining `90%` of that contrast's projected
energy.

The TBS variants separate topology from time:

- topology-only average linkage on standardized PCA distance;
- recomputed fixed-topology native NNLS branch lengths fitted to squared
  standardized continuous distances, used with normalized branch-time variance;
- explicitly labeled raw-linkage branch-time diagnostics, retained only as a
  negative-control sensitivity check;
- adaptive diffusion topology using a variable-bandwidth diffusion distance;
- adaptive diffusion with the same recomputed-NNLS versus raw-linkage
  diagnostic branch-time split.

Adaptive diffusion metadata: `{"backend": "pydiffmap", "bandwidth_type": "-1/(d+2)", "diffusion_time": 3, "epsilon": 0.14566057964753984, "epsilon_method": "median", "k_neighbors_requested": 15, "metric": "euclidean", "n_components": 30, "neighbor_search_k": 15}`.

## Results

| method                                                                                                 | status   |   significance_level |   edge_alpha |   n_clusters |   overcluster_ratio |   weighted_cluster_purity |   weighted_label_dominant_cluster_recall |   merge_error_rate |   split_error_rate |   weighted_effective_clusters_per_label |   homogeneity |   completeness |   v_measure |    nmi |    ari |   silhouette |   elapsed_sec | skip_reason   |
|:-------------------------------------------------------------------------------------------------------|:---------|---------------------:|-------------:|-------------:|--------------------:|--------------------------:|-----------------------------------------:|-------------------:|-------------------:|----------------------------------------:|--------------:|---------------:|------------:|-------:|-------:|-------------:|--------------:|:--------------|
| TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001 | ok       |               0.0100 |       0.0010 |           40 |              2.6667 |                    0.9288 |                                   0.4836 |             0.0712 |             0.5164 |                                  4.4881 |        0.8697 |         0.5396 |      0.6660 | 0.6660 | 0.4152 |       0.2303 |        7.2384 |               |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001                           | ok       |               0.0100 |       0.0010 |           48 |              3.2000 |                    0.9292 |                                   0.4836 |             0.0708 |             0.5164 |                                  5.5682 |        0.8712 |         0.5168 |      0.6488 | 0.6488 | 0.3886 |       0.1911 |        7.4824 |               |
| TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001        | ok       |               0.0100 |       0.0010 |           48 |              3.2000 |                    0.9292 |                                   0.4836 |             0.0708 |             0.5164 |                                  5.5682 |        0.8712 |         0.5168 |      0.6488 | 0.6488 | 0.3886 |       0.1911 |       10.3923 |               |
| Leiden                                                                                                 | ok       |             nan      |     nan      |           18 |              1.2000 |                    0.8632 |                                   0.4692 |             0.1368 |             0.5308 |                                  3.8781 |        0.7925 |         0.5388 |      0.6415 | 0.6415 | 0.3605 |       0.2535 |        0.2943 |               |
| Louvain                                                                                                | ok       |             nan      |     nan      |           18 |              1.2000 |                    0.8612 |                                   0.4688 |             0.1388 |             0.5312 |                                  3.9001 |        0.7906 |         0.5370 |      0.6396 | 0.6396 | 0.3594 |       0.2553 |        0.4730 |               |
| Spectral true K                                                                                        | ok       |             nan      |     nan      |           15 |              1.0000 |                    0.8172 |                                   0.6352 |             0.1828 |             0.3648 |                                  2.9888 |        0.6989 |         0.5582 |      0.6207 | 0.6207 | 0.4486 |       0.2671 |        0.0867 |               |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                                              | ok       |               0.0100 |       0.0010 |           45 |              3.0000 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |                                  3.5375 |        0.6968 |         0.5299 |      0.6020 | 0.6020 | 0.3657 |       0.1039 |        8.4165 |               |
| TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001                           | ok       |               0.0100 |       0.0010 |           45 |              3.0000 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |                                  3.5375 |        0.6968 |         0.5299 |      0.6020 | 0.6020 | 0.3657 |       0.1039 |       10.8321 |               |
| K-means true K                                                                                         | ok       |             nan      |     nan      |           15 |              1.0000 |                    0.7796 |                                   0.6332 |             0.2204 |             0.3668 |                                  2.9505 |        0.6474 |         0.5403 |      0.5890 | 0.5890 | 0.4347 |       0.2983 |        0.0781 |               |
| HDBSCAN                                                                                                | ok       |             nan      |     nan      |           14 |              0.9333 |                    0.5980 |                                   0.6320 |             0.4020 |             0.3680 |                                  2.9134 |        0.4563 |         0.4515 |      0.4539 | 0.4539 | 0.2406 |       0.3263 |        0.1585 |               |
| TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001                    | ok       |               0.0100 |       0.0010 |            9 |              0.6000 |                    0.3772 |                                   0.9932 |             0.6228 |             0.0068 |                                  1.0371 |        0.1885 |         0.9111 |      0.3124 | 0.3124 | 0.0787 |       0.4459 |        7.7235 |               |

## TBS tree and branch lengths

| method                                                                                                 |   edge_count |   branch_length_count |   branch_length_mean |   branch_length_median |   branch_length_min |   branch_length_max | edge_branch_length_variance_policy   |   spectral_minimum_dimension |   adaptive_projection_dimension_energy_fraction | tree_builder   | branch_length_optimization_method   |   branch_length_optimization_residual_rmse |   branch_length_optimization_residual_mae |   branch_length_optimization_n_pairs_used |   branch_length_optimization_elapsed_sec |
|:-------------------------------------------------------------------------------------------------------|-------------:|----------------------:|---------------------:|-----------------------:|--------------------:|--------------------:|:-------------------------------------|-----------------------------:|------------------------------------------------:|:---------------|:------------------------------------|-------------------------------------------:|------------------------------------------:|------------------------------------------:|-----------------------------------------:|
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                                              |         4998 |                  4998 |               0.0545 |                 0.0466 |              0.0002 |              0.7815 | none                                 |                            2 |                                          0.9000 | linkage        | linkage_ultrametric                 |                                   nan      |                                  nan      |                                  nan      |                                 nan      |
| TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001                           |         4998 |                  4998 |               0.1750 |                 0.0815 |              0.0000 |              8.1387 | normalized_branch_length             |                            2 |                                          0.9000 | linkage        | fixed_topology_nnls                 |                                     0.3050 |                                    0.2043 |                                50000.0000 |                                   2.8513 |
| TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001                    |         4998 |                  4998 |               0.0545 |                 0.0466 |              0.0002 |              0.7815 | normalized_branch_length             |                            2 |                                          0.9000 | linkage        | linkage_ultrametric                 |                                   nan      |                                  nan      |                                  nan      |                                 nan      |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001                           |         4998 |                  4998 |               0.0150 |                 0.0080 |              0.0001 |              0.6265 | none                                 |                            2 |                                          0.9000 | linkage        | linkage_ultrametric                 |                                   nan      |                                  nan      |                                  nan      |                                 nan      |
| TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001        |         4998 |                  4998 |               0.1865 |                 0.0812 |              0.0000 |             13.0880 | normalized_branch_length             |                            2 |                                          0.9000 | linkage        | fixed_topology_nnls                 |                                     0.3084 |                                    0.2022 |                                50000.0000 |                                   2.9092 |
| TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001 |         4998 |                  4998 |               0.0150 |                 0.0080 |              0.0001 |              0.6265 | normalized_branch_length             |                            2 |                                          0.9000 | linkage        | linkage_ultrametric                 |                                   nan      |                                  nan      |                                  nan      |                                 nan      |

## Branch-time sensitivity

The length-transform search is a supervised diagnostic on fixed TBS topologies:
it rescales edge Wald statistics, reapplies Tree-BH, traverses the same
hierarchy, and scores split/merge behavior against curated cell types. It is not
used as a production clustering rule.

| method                                                                       | length_model      |   edge_open_count |   n_clusters |   weighted_cluster_purity |   weighted_label_dominant_cluster_recall |   merge_error_rate |   split_error_rate |   v_measure |    ari |
|:-----------------------------------------------------------------------------|:------------------|------------------:|-------------:|--------------------------:|-----------------------------------------:|-------------------:|-------------------:|------------:|-------:|
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | linear_scale_2    |                83 |           35 |                    0.9284 |                                   0.4988 |             0.0716 |             0.5012 |      0.6707 | 0.4175 |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | linear_scale_1    |               102 |           40 |                    0.9288 |                                   0.4836 |             0.0712 |             0.5164 |      0.6660 | 0.4152 |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | linear_scale_0p5  |               122 |           42 |                    0.9288 |                                   0.4836 |             0.0712 |             0.5164 |      0.6627 | 0.4129 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | none              |               214 |           45 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |      0.6020 | 0.3657 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | linear_scale_0p25 |               205 |           45 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |      0.6020 | 0.3657 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | linear_scale_0p5  |               188 |           45 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |      0.6020 | 0.3657 |

## Files

- `method_metrics.csv`: benchmark metrics.
- `method_assignments.csv`: per-cell benchmark labels.
- `celltype_fragmentation_by_method.csv`: per-cell-type split diagnostics.
- `cluster_composition_by_method.csv`: per-cluster merge/purity diagnostics.
- `tbs_branch_time_sensitivity.csv`: fixed-topology branch-time transform
  sensitivity scored with split/merge metrics.
- `tbs_branch_time_sensitivity.png`: branch-time sensitivity plot.
- `edge_gate_distance_time_model_analysis.md`: distance-vs-time model notes,
  adaptive diffusion metadata, and branch-length summary.
- `tbs_tree_branch_length_summary.csv`: per-TBS tree branch-length summaries.
- `*_tree_edges.csv`: per-edge TBS tree diagnostics with edge-gate p-values.
- `*_traversal_trace.csv`: live TBS traversal diagnostics with sibling-gate
  and final-boundary decisions.
- `*_full_edge_traversal_trace.csv`: edge-reachable traversal diagnostics,
  independent of sibling-gate stops.
- `*_tree_dendrogram.png`: truncated TBS dendrogram plots.
- `*_branch_lengths.png`: TBS branch-length histograms.
- `benchmark_subset_cells.csv`: subset metadata.
- `benchmark_subset_pca.csv`: PCA features used for benchmarking.
- `pancreas_classical_pipeline.h5ad`: processed AnnData checkpoint.
- `classical_umap_overview.png`: UMAP colored by batch and cell type.
- `method_umap_clusters.png`: subset UMAP colored by curated labels and method
  cluster assignments.
- `method_ari_barplot.png`: ARI comparison.
- `method_split_merge_diagnostic.png`: purity-vs-fragmentation diagnostic.
- `manifest.json`: parameters, package versions, and provenance.
