# Edge gate distance/time model analysis

Generated at: 2026-06-27T21:27:32+02:00

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
  "epsilon": 0.14566057964753984,
  "epsilon_method": "median",
  "k_neighbors_requested": 15,
  "metric": "euclidean",
  "n_components": 30,
  "neighbor_search_k": 15
}
```

## Branch length summaries

| method                                                                                                 |   edge_count |   branch_length_count |   branch_length_mean |   branch_length_median |   branch_length_min |   branch_length_max | edge_branch_length_variance_policy   |   spectral_minimum_dimension |   adaptive_projection_dimension_energy_fraction | tree_builder   | branch_length_optimization_method   |   branch_length_optimization_residual_rmse |   branch_length_optimization_residual_mae |   branch_length_optimization_n_pairs_used |   branch_length_optimization_elapsed_sec |
|:-------------------------------------------------------------------------------------------------------|-------------:|----------------------:|---------------------:|-----------------------:|--------------------:|--------------------:|:-------------------------------------|-----------------------------:|------------------------------------------------:|:---------------|:------------------------------------|-------------------------------------------:|------------------------------------------:|------------------------------------------:|-----------------------------------------:|
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                                              |         4998 |                  4998 |               0.0545 |                 0.0466 |              0.0002 |              0.7815 | none                                 |                            2 |                                          0.9000 | linkage        | linkage_ultrametric                 |                                   nan      |                                  nan      |                                  nan      |                                 nan      |
| TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001                           |         4998 |                  4998 |               0.1750 |                 0.0815 |              0.0000 |              8.1387 | normalized_branch_length             |                            2 |                                          0.9000 | linkage        | fixed_topology_nnls                 |                                     0.3050 |                                    0.2043 |                                50000.0000 |                                   2.8513 |
| TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001                    |         4998 |                  4998 |               0.0545 |                 0.0466 |              0.0002 |              0.7815 | normalized_branch_length             |                            2 |                                          0.9000 | linkage        | linkage_ultrametric                 |                                   nan      |                                  nan      |                                  nan      |                                 nan      |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001                           |         4998 |                  4998 |               0.0150 |                 0.0080 |              0.0001 |              0.6265 | none                                 |                            2 |                                          0.9000 | linkage        | linkage_ultrametric                 |                                   nan      |                                  nan      |                                  nan      |                                 nan      |
| TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001        |         4998 |                  4998 |               0.1865 |                 0.0812 |              0.0000 |             13.0880 | normalized_branch_length             |                            2 |                                          0.9000 | linkage        | fixed_topology_nnls                 |                                     0.3084 |                                    0.2022 |                                50000.0000 |                                   2.9092 |
| TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001 |         4998 |                  4998 |               0.0150 |                 0.0080 |              0.0001 |              0.6265 | normalized_branch_length             |                            2 |                                          0.9000 | linkage        | linkage_ultrametric                 |                                   nan      |                                  nan      |                                  nan      |                                 nan      |

## Branch-time sensitivity, top rows per topology

| method                                                                       | length_model      |   edge_open_count |   n_clusters |   weighted_cluster_purity |   weighted_label_dominant_cluster_recall |   merge_error_rate |   split_error_rate |   v_measure |    ari |
|:-----------------------------------------------------------------------------|:------------------|------------------:|-------------:|--------------------------:|-----------------------------------------:|-------------------:|-------------------:|------------:|-------:|
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | linear_scale_2    |                83 |           35 |                    0.9284 |                                   0.4988 |             0.0716 |             0.5012 |      0.6707 | 0.4175 |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | linear_scale_1    |               102 |           40 |                    0.9288 |                                   0.4836 |             0.0712 |             0.5164 |      0.6660 | 0.4152 |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | linear_scale_0p5  |               122 |           42 |                    0.9288 |                                   0.4836 |             0.0712 |             0.5164 |      0.6627 | 0.4129 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | none              |               214 |           45 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |      0.6020 | 0.3657 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | linear_scale_0p25 |               205 |           45 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |      0.6020 | 0.3657 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | linear_scale_0p5  |               188 |           45 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |      0.6020 | 0.3657 |
