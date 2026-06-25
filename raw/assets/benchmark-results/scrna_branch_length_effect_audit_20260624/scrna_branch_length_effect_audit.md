# scRNA Branch-Length Effect Audit

Generated at: 2026-06-24T20:48:26+02:00

## Summary

The rerun confirms that branch lengths are not needed to place the cells on the
tree: topology is inferred first from a distance matrix, then branch lengths are
stored or refit. Branch lengths change clustering only when the edge gate is run
with `edge_branch_length_variance_policy="normalized_branch_length"`.

The stable interpretation is:

- Adaptive-diffusion topology is the safest current tree for the Goncalves
  progenitor review. Its topology-only, NNLS branch-time, and raw-linkage
  branch-time rows give the same `24` fetal TBS clusters in the rerun.
- Adult pancreas also keeps the same adaptive-diffusion topology-only and NNLS
  clusters (`43` clusters); raw-linkage branch-time changes that row only mildly
  (`41` clusters).
- Standardized-PCA linkage is branch-time sensitive. Adult raw-linkage
  branch-time collapses from `45` to `9` clusters. Goncalves standardized-PCA
  branch-time, including the NNLS row, collapses from `40` to `1` cluster.

So the wrong conclusion would be "branch lengths define the correct clusters."
The evidence says the tree topology and the edge/sibling gates define the
current clusters; branch lengths are a variance-scaling sensitivity dial unless
we can justify a real stochastic-time model.

## How The Tree Is Inferred

1. The benchmark builds a PCA matrix from the current scRNA workflow.
2. For standardized-PCA topology, it computes pairwise Euclidean distances in
   PCA space and runs average-linkage hierarchical clustering.
3. For adaptive-diffusion topology, it builds a variable-bandwidth diffusion
   distance from the PCA kNN graph (`k=15`, diffusion time `3`) and runs the
   same average-linkage tree builder.
4. Linkage heights become diagnostic branch lengths. In NNLS rows, the topology
   is held fixed and non-negative least squares refits edge lengths to better
   approximate continuous pairwise distances.
5. TBS traverses that fixed tree using child-parent edge gates and sibling
   gates. Branch lengths affect the edge gate only under the explicit
   normalized branch-time variance policy.

## Rerun Manifests

- adult_pancreas: elapsed 85.97822145800092 seconds, seed 0, max cells 2500
- goncalves_fetal: elapsed 40.24420908399952 seconds, seed 0, max cells 2500

## Branch-Length Effect Table

| dataset_label            | geometry           | variant                 |   n_clusters |   v_measure_vs_celltype |   weighted_cluster_purity |   weighted_label_dominant_cluster_recall |   delta_n_clusters_vs_topology |   assignment_ari_vs_topology |   edge_open_count |   branch_length_median |   branch_length_max | branch_length_optimization_method   |
|:-------------------------|:-------------------|:------------------------|-------------:|------------------------:|--------------------------:|-----------------------------------------:|-------------------------------:|-----------------------------:|------------------:|-----------------------:|--------------------:|:------------------------------------|
| Adult pancreas           | pca_linkage        | topology_only           |           45 |                  0.6020 |                    0.7928 |                                   0.5800 |                         0.0000 |                       1.0000 |               214 |                 0.0466 |              0.7815 | linkage_ultrametric                 |
| Adult pancreas           | pca_linkage        | branch_time_nnls        |           45 |                  0.6020 |                    0.7928 |                                   0.5800 |                         0.0000 |                       1.0000 |               160 |                 0.0808 |              8.1859 | fixed_topology_nnls                 |
| Adult pancreas           | pca_linkage        | branch_time_raw_linkage |            9 |                  0.3124 |                    0.3772 |                                   0.9932 |                       -36.0000 |                       0.0530 |               136 |                 0.0466 |              0.7815 | linkage_ultrametric                 |
| Adult pancreas           | adaptive_diffusion | topology_only           |           43 |                  0.6620 |                    0.9288 |                                   0.4840 |                         0.0000 |                       1.0000 |               142 |                 0.0080 |              0.6321 | linkage_ultrametric                 |
| Adult pancreas           | adaptive_diffusion | branch_time_nnls        |           43 |                  0.6620 |                    0.9288 |                                   0.4840 |                         0.0000 |                       1.0000 |               129 |                 0.0804 |             13.1820 | fixed_topology_nnls                 |
| Adult pancreas           | adaptive_diffusion | branch_time_raw_linkage |           41 |                  0.6653 |                    0.9288 |                                   0.4840 |                        -2.0000 |                       0.9970 |               104 |                 0.0080 |              0.6321 | linkage_ultrametric                 |
| Goncalves fetal pancreas | pca_linkage        | topology_only           |           40 |                  0.3887 |                    0.6628 |                                   0.3857 |                         0.0000 |                       1.0000 |               150 |                 0.1723 |              0.8175 | linkage_ultrametric                 |
| Goncalves fetal pancreas | pca_linkage        | branch_time_nnls        |            1 |                  0.0000 |                    0.2962 |                                   1.0000 |                       -39.0000 |                       0.0000 |                 0 |                 0.1181 |              3.3702 | fixed_topology_nnls                 |
| Goncalves fetal pancreas | pca_linkage        | branch_time_raw_linkage |            1 |                  0.0000 |                    0.2962 |                                   1.0000 |                       -39.0000 |                       0.0000 |                49 |                 0.1723 |              0.8175 | linkage_ultrametric                 |
| Goncalves fetal pancreas | adaptive_diffusion | topology_only           |           24 |                  0.4100 |                    0.6546 |                                   0.5399 |                         0.0000 |                       1.0000 |                98 |                 0.0423 |              0.7539 | linkage_ultrametric                 |
| Goncalves fetal pancreas | adaptive_diffusion | branch_time_nnls        |           24 |                  0.4100 |                    0.6546 |                                   0.5399 |                         0.0000 |                       1.0000 |                91 |                 0.1287 |              2.4400 | fixed_topology_nnls                 |
| Goncalves fetal pancreas | adaptive_diffusion | branch_time_raw_linkage |           24 |                  0.4100 |                    0.6546 |                                   0.5399 |                         0.0000 |                       1.0000 |                90 |                 0.0423 |              0.7539 | linkage_ultrametric                 |

## Branch-Time Sensitivity Scan

| dataset_label            | geometry           | length_model      |   edge_open_count |   n_clusters |   weighted_cluster_purity |   weighted_label_dominant_cluster_recall |   v_measure |   median_variance_multiplier |   max_variance_multiplier |
|:-------------------------|:-------------------|:------------------|------------------:|-------------:|--------------------------:|-----------------------------------------:|------------:|-----------------------------:|--------------------------:|
| Adult pancreas           | adaptive_diffusion | linear_scale_2    |                82 |           36 |                    0.9284 |                                   0.4992 |      0.6700 |                       2.0579 |                   84.7940 |
| Adult pancreas           | adaptive_diffusion | linear_scale_1    |               104 |           41 |                    0.9288 |                                   0.4840 |      0.6653 |                       1.5289 |                   42.8970 |
| Adult pancreas           | adaptive_diffusion | none              |               142 |           43 |                    0.9288 |                                   0.4840 |      0.6620 |                       1.0000 |                    1.0000 |
| Adult pancreas           | pca_linkage        | none              |               214 |           45 |                    0.7928 |                                   0.5800 |      0.6020 |                       1.0000 |                    1.0000 |
| Adult pancreas           | pca_linkage        | linear_scale_0p25 |               205 |           45 |                    0.7928 |                                   0.5800 |      0.6020 |                       1.2138 |                    4.5827 |
| Adult pancreas           | pca_linkage        | linear_scale_0p5  |               188 |           45 |                    0.7928 |                                   0.5800 |      0.6020 |                       1.4275 |                    8.1653 |
| Goncalves fetal pancreas | adaptive_diffusion | linear_scale_2    |                76 |           23 |                    0.6546 |                                   0.5399 |      0.4103 |                       2.5061 |                   27.8366 |
| Goncalves fetal pancreas | adaptive_diffusion | quadratic_scale_1 |                68 |           23 |                    0.6546 |                                   0.5399 |      0.4103 |                       1.5671 |                  181.0507 |
| Goncalves fetal pancreas | adaptive_diffusion | none              |                98 |           24 |                    0.6546 |                                   0.5399 |      0.4100 |                       1.0000 |                    1.0000 |
| Goncalves fetal pancreas | pca_linkage        | linear_scale_0p25 |               116 |           35 |                    0.6628 |                                   0.4464 |      0.3974 |                       1.2018 |                    1.9573 |
| Goncalves fetal pancreas | pca_linkage        | linear_scale_0p5  |                93 |           35 |                    0.6628 |                                   0.4464 |      0.3974 |                       1.4035 |                    2.9147 |
| Goncalves fetal pancreas | pca_linkage        | quadratic_scale_1 |                90 |           35 |                    0.6628 |                                   0.4464 |      0.3974 |                       1.6513 |                   15.6637 |

## Output Plots

- `branch_length_cluster_effects.png`
- `branch_length_assignment_similarity_heatmap.png`
- `branch_time_sensitivity_effects.png`

## Source Tables

- Adult pancreas: `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/`
- Goncalves fetal pancreas:
  `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/`
