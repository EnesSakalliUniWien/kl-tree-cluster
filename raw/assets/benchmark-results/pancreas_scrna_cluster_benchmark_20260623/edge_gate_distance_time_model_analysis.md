# Edge gate distance/time model analysis

Generated at: 2026-06-24T20:07:07+02:00

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
  "epsilon": 0.1485654790161756,
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
|                                          0.9000 |                  4998 |              0.7815 |               0.0545 |                 0.0466 |              0.0002 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | none                                 |         4998 | TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                                              |                            2 | linkage        |
|                                          0.9000 |                  4998 |              8.1859 |               0.1759 |                 0.0808 |              0.0000 |                                   3.2049 | fixed_topology_nnls                 |                                50000.0000 |                                    0.2017 |                                     0.3018 | normalized_branch_length             |         4998 | TBS branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001                           |                            2 | linkage        |
|                                          0.9000 |                  4998 |              0.7815 |               0.0545 |                 0.0466 |              0.0002 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | normalized_branch_length             |         4998 | TBS raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001                    |                            2 | linkage        |
|                                          0.9000 |                  4998 |              0.6321 |               0.0151 |                 0.0080 |              0.0001 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | none                                 |         4998 | TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001                           |                            2 | linkage        |
|                                          0.9000 |                  4998 |             13.1820 |               0.1865 |                 0.0804 |              0.0000 |                                   3.3028 | fixed_topology_nnls                 |                                50000.0000 |                                    0.2013 |                                     0.3075 | normalized_branch_length             |         4998 | TBS adaptive diffusion branch-time recomputed-NNLS projected adaptive-k90 alpha=0.01 edge=0.001        |                            2 | linkage        |
|                                          0.9000 |                  4998 |              0.6321 |               0.0151 |                 0.0080 |              0.0001 |                                 nan      | linkage_ultrametric                 |                                  nan      |                                  nan      |                                   nan      | normalized_branch_length             |         4998 | TBS adaptive diffusion raw-linkage branch-time diagnostic projected adaptive-k90 alpha=0.01 edge=0.001 |                            2 | linkage        |

## Branch-time sensitivity, top rows per topology

| method                                                                       | length_model      |   edge_open_count |   n_clusters |   weighted_cluster_purity |   weighted_label_dominant_cluster_recall |   merge_error_rate |   split_error_rate |   v_measure |    ari |
|:-----------------------------------------------------------------------------|:------------------|------------------:|-------------:|--------------------------:|-----------------------------------------:|-------------------:|-------------------:|------------:|-------:|
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | linear_scale_2    |                82 |           36 |                    0.9284 |                                   0.4992 |             0.0716 |             0.5008 |      0.6700 | 0.4173 |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | linear_scale_1    |               104 |           41 |                    0.9288 |                                   0.4840 |             0.0712 |             0.5160 |      0.6653 | 0.4151 |
| TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 | none              |               142 |           43 |                    0.9288 |                                   0.4840 |             0.0712 |             0.5160 |      0.6620 | 0.4127 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | none              |               214 |           45 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |      0.6020 | 0.3657 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | linear_scale_0p25 |               205 |           45 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |      0.6020 | 0.3657 |
| TBS topology projected adaptive-k90 alpha=0.01 edge=0.001                    | linear_scale_0p5  |               188 |           45 |                    0.7928 |                                   0.5800 |             0.2072 |             0.4200 |      0.6020 | 0.3657 |
