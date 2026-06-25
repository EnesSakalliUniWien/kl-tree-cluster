# Edge gate distance/time model analysis

Generated at: 2026-06-24T19:57:27+02:00

## Interpretation

- Linkage and adaptive diffusion distances are used here as topology models:
  they choose which leaves/subtrees merge and assign dendrogram branch lengths.
- The default edge-gate null is sampling covariance only. It records branch
  lengths for diagnostics but does not treat them as elapsed stochastic time.
- The `normalized_branch_length` policy is the explicit branch-time variance
  model: it multiplies child-parent contrast variance by
  `1 + branch_length / mean_branch_length`.
- Production-facing linkage branch-time rows recompute all edge lengths on the
  fixed topology with native NNLS before applying that policy. Raw linkage
  ultrametric heights are kept only in explicitly labeled diagnostic rows.
- The TBS rows use projected-Wald sibling gates with local adaptive projected
  dimensions. For each edge or sibling contrast, the test starts from the local
  PCA basis and keeps the shortest prefix explaining 90% of that contrast's
  projected energy; the MP/floor dimension remains diagnostic metadata.
- The branch-time sensitivity table below is supervised failure analysis. It
  reuses a fixed topology and sibling gate, rescales edge Wald statistics by
  candidate time multipliers, reruns Tree-BH and traversal, then scores the
  resulting labels. These rows should not be interpreted as an unsupervised
  production rule.

## Adaptive diffusion topology

```json
{
  "backend": "pydiffmap",
  "bandwidth_type": "-1/(d+2)",
  "diffusion_time": 3,
  "epsilon": 0.4445929731234955,
  "epsilon_method": "median",
  "k_neighbors_requested": 15,
  "metric": "euclidean",
  "n_components": 30,
  "neighbor_search_k": 15
}
```

## Branch length summaries

|   adaptive_projection_dimension_energy_fraction |   branch_length_count |   branch_length_max |   branch_length_mean |   branch_length_median |   branch_length_min |   branch_length_optimization_elapsed_sec | branch_length_optimization_method   |   branch_length_optimization_n_pairs_used |   branch_length_optimization_residual_mae |   branch_length_optimization_residual_rmse | edge_branch_length_variance_policy   |   edge_count | method                                                                                                 |   spectral_minimum_dimension | tree_builder   |
|------------------------------------------------:|----------------------:|--------------------:|---------------------:|-----------------------:|--------------------:|-----------------------------------------:|:------------------------------------|------------------------------------------:|------------------------------------------:|-------------------------------------------:|:-------------------------------------|-------------:|:-------------------------------------------------------------------------------------------------------|-----------------------------:|:---------------|
|                                          0.9000 |                  2928 |              0.8175 |               0.2135 |                 0.1723 |              0.0012 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | none                                 |         2928 | TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                                              |                            2 | linkage        |
|                                          0.9000 |                  2928 |              3.3702 |               0.2255 |                 0.1181 |              0.0000 |                                   2.1033 | fixed_topology_nnls                 |                                50000.0000 |                                    0.2493 |                                     0.3303 | normalized_branch_length             |         2928 | TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001                           |                            2 | linkage        |
|                                          0.9000 |                  2928 |              0.8175 |               0.2135 |                 0.1723 |              0.0012 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | normalized_branch_length             |         2928 | TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001                    |                            2 | linkage        |
|                                          0.9000 |                  2928 |              0.7539 |               0.0562 |                 0.0423 |              0.0005 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | none                                 |         2928 | TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001                           |                            2 | linkage        |
|                                          0.9000 |                  2928 |              2.4400 |               0.2168 |                 0.1287 |              0.0000 |                                   1.8491 | fixed_topology_nnls                 |                                50000.0000 |                                    0.2536 |                                     0.3347 | normalized_branch_length             |         2928 | TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001        |                            2 | linkage        |
|                                          0.9000 |                  2928 |              0.7539 |               0.0562 |                 0.0423 |              0.0005 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | normalized_branch_length             |         2928 | TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001 |                            2 | linkage        |

## Branch-time sensitivity, top rows per topology

| method                                                                       | length_model      |   edge_open_count |   n_clusters |   weighted_cluster_purity |   weighted_label_dominant_cluster_recall |   merge_error_rate |   split_error_rate |   v_measure |    ari |
|:-----------------------------------------------------------------------------|:------------------|------------------:|-------------:|--------------------------:|-----------------------------------------:|-------------------:|-------------------:|------------:|-------:|
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | linear_scale_2    |                76 |           23 |                    0.6546 |                                   0.5399 |             0.3454 |             0.4601 |      0.4103 | 0.2061 |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | quadratic_scale_1 |                68 |           23 |                    0.6546 |                                   0.5399 |             0.3454 |             0.4601 |      0.4103 | 0.2061 |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | none              |                98 |           24 |                    0.6546 |                                   0.5399 |             0.3454 |             0.4601 |      0.4100 | 0.2058 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | linear_scale_0p25 |               116 |           35 |                    0.6628 |                                   0.4464 |             0.3372 |             0.5536 |      0.3974 | 0.2504 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | linear_scale_0p5  |                93 |           35 |                    0.6628 |                                   0.4464 |             0.3372 |             0.5536 |      0.3974 | 0.2504 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | quadratic_scale_1 |                90 |           35 |                    0.6628 |                                   0.4464 |             0.3372 |             0.5536 |      0.3974 | 0.2504 |
