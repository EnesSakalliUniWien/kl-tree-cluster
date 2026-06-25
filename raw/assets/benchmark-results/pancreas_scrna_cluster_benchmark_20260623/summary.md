# Pancreas scRNA clustering benchmark

Generated at: 2026-06-24T20:07:07+02:00

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

Adaptive diffusion metadata: `{"backend": "pydiffmap", "bandwidth_type": "-1/(d+2)", "diffusion_time": 3, "epsilon": 0.1485654790161756, "epsilon_method": "median", "k_neighbors_requested": 15, "metric": "euclidean", "n_components": 30, "neighbor_search_k": 15}`.

## Results

| method                                                                                                 | status   |   significance_level |   edge_alpha |   n_clusters |   overcluster_ratio |   weighted_cluster_purity |   weighted_label_dominant_cluster_recall |   merge_error_rate |   split_error_rate |   weighted_effective_clusters_per_label |   homogeneity |   completeness |   v_measure |    nmi |    ari |   silhouette |   elapsed_sec |   skip_reason |
|:-------------------------------------------------------------------------------------------------------|:---------|---------------------:|-------------:|-------------:|--------------------:|--------------------------:|-----------------------------------------:|-------------------:|-------------------:|----------------------------------------:|--------------:|---------------:|------------:|-------:|-------:|-------------:|--------------:|--------------:|
| TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001 | ok       |               0.0100 |       0.0010 |           41 |              2.7333 |                    0.9288 |                                   0.4840 |             0.0712 |             0.5160 |                                  4.4989 |        0.8697 |         0.5387 |      0.6653 | 0.6653 | 0.4151 |       0.2304 |        7.2762 |           nan |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001                           | ok       |               0.0100 |       0.0010 |           43 |              2.8667 |                    0.9288 |                                   0.4840 |             0.0712 |             0.5160 |                                  4.6610 |        0.8697 |         0.5344 |      0.6620 | 0.6620 | 0.4127 |       0.2200 |        7.2530 |           nan |
| TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001        | ok       |               0.0100 |       0.0010 |           43 |              2.8667 |                    0.9288 |                                   0.4840 |             0.0712 |             0.5160 |                                  4.6610 |        0.8697 |         0.5344 |      0.6620 | 0.6620 | 0.4127 |       0.2200 |       10.3837 |           nan |
| Leiden                                                                                                 | ok       |             nan      |     nan      |           18 |              1.2000 |                    0.8632 |                                   0.4692 |             0.1368 |             0.5308 |                                  3.8781 |        0.7925 |         0.5388 |      0.6415 | 0.6415 | 0.3605 |       0.2535 |        0.3168 |           nan |
| Louvain                                                                                                | ok       |             nan      |     nan      |           18 |              1.2000 |                    0.8612 |                                   0.4688 |             0.1388 |             0.5312 |                                  3.9001 |        0.7906 |         0.5370 |      0.6396 | 0.6396 | 0.3594 |       0.2553 |        0.3913 |           nan |
| Spectral true K                                                                                        | ok       |             nan      |     nan      |           15 |              1.0000 |                    0.8172 |                                   0.6352 |             0.1828 |             0.3648 |                                  2.9888 |        0.6989 |         0.5582 |      0.6207 | 0.6207 | 0.4486 |       0.2671 |        0.2844 |           nan |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                                              | ok       |               0.0100 |       0.0010 |           45 |              3.0000 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |                                  3.5375 |        0.6968 |         0.5299 |      0.6020 | 0.6020 | 0.3657 |       0.1039 |        8.2862 |           nan |
| TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001                           | ok       |               0.0100 |       0.0010 |           45 |              3.0000 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |                                  3.5375 |        0.6968 |         0.5299 |      0.6020 | 0.6020 | 0.3657 |       0.1039 |       10.9440 |           nan |
| K-means true K                                                                                         | ok       |             nan      |     nan      |           15 |              1.0000 |                    0.7796 |                                   0.6332 |             0.2204 |             0.3668 |                                  2.9505 |        0.6474 |         0.5403 |      0.5890 | 0.5890 | 0.4347 |       0.2983 |        0.0885 |           nan |
| HDBSCAN                                                                                                | ok       |             nan      |     nan      |           14 |              0.9333 |                    0.5980 |                                   0.6320 |             0.4020 |             0.3680 |                                  2.9134 |        0.4563 |         0.4515 |      0.4539 | 0.4539 | 0.2406 |       0.3263 |        0.1799 |           nan |
| TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001                    | ok       |               0.0100 |       0.0010 |            9 |              0.6000 |                    0.3772 |                                   0.9932 |             0.6228 |             0.0068 |                                  1.0371 |        0.1885 |         0.9111 |      0.3124 | 0.3124 | 0.0787 |       0.4459 |        7.3264 |           nan |

## TBS tree and branch lengths

|   adaptive_projection_dimension_energy_fraction |   branch_length_count |   branch_length_max |   branch_length_mean |   branch_length_median |   branch_length_min |   branch_length_optimization_elapsed_sec | branch_length_optimization_method   |   branch_length_optimization_n_pairs_used |   branch_length_optimization_residual_mae |   branch_length_optimization_residual_rmse | edge_branch_length_variance_policy   |   edge_count | method                                                                                                 |   spectral_minimum_dimension | tree_builder   |
|------------------------------------------------:|----------------------:|--------------------:|---------------------:|-----------------------:|--------------------:|-----------------------------------------:|:------------------------------------|------------------------------------------:|------------------------------------------:|-------------------------------------------:|:-------------------------------------|-------------:|:-------------------------------------------------------------------------------------------------------|-----------------------------:|:---------------|
|                                          0.9000 |                  4998 |              0.7815 |               0.0545 |                 0.0466 |              0.0002 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | none                                 |         4998 | TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                                              |                            2 | linkage        |
|                                          0.9000 |                  4998 |              8.1859 |               0.1759 |                 0.0808 |              0.0000 |                                   3.2049 | fixed_topology_nnls                 |                                50000.0000 |                                    0.2017 |                                     0.3018 | normalized_branch_length             |         4998 | TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001                           |                            2 | linkage        |
|                                          0.9000 |                  4998 |              0.7815 |               0.0545 |                 0.0466 |              0.0002 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | normalized_branch_length             |         4998 | TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001                    |                            2 | linkage        |
|                                          0.9000 |                  4998 |              0.6321 |               0.0151 |                 0.0080 |              0.0001 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | none                                 |         4998 | TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001                           |                            2 | linkage        |
|                                          0.9000 |                  4998 |             13.1820 |               0.1865 |                 0.0804 |              0.0000 |                                   3.3028 | fixed_topology_nnls                 |                                50000.0000 |                                    0.2013 |                                     0.3075 | normalized_branch_length             |         4998 | TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001        |                            2 | linkage        |
|                                          0.9000 |                  4998 |              0.6321 |               0.0151 |                 0.0080 |              0.0001 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | normalized_branch_length             |         4998 | TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001 |                            2 | linkage        |

## Branch-time sensitivity

The length-transform search is a supervised diagnostic on fixed TBS topologies:
it rescales edge Wald statistics, reapplies Tree-BH, traverses the same
hierarchy, and scores split/merge behavior against curated cell types. It is not
used as a production clustering rule.

| method                                                                       | length_model      |   edge_open_count |   n_clusters |   weighted_cluster_purity |   weighted_label_dominant_cluster_recall |   merge_error_rate |   split_error_rate |   v_measure |    ari |
|:-----------------------------------------------------------------------------|:------------------|------------------:|-------------:|--------------------------:|-----------------------------------------:|-------------------:|-------------------:|------------:|-------:|
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | linear_scale_2    |                82 |           36 |                    0.9284 |                                   0.4992 |             0.0716 |             0.5008 |      0.6700 | 0.4173 |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | linear_scale_1    |               104 |           41 |                    0.9288 |                                   0.4840 |             0.0712 |             0.5160 |      0.6653 | 0.4151 |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | none              |               142 |           43 |                    0.9288 |                                   0.4840 |             0.0712 |             0.5160 |      0.6620 | 0.4127 |
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
