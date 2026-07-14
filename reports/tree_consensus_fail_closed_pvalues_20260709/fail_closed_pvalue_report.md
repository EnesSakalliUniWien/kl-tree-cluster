# Fail-Closed Under-Split P-Value Analysis

## Scope

This report analyzes the 14 real multi-cluster fail-closed selections from the full adaptive-K graphtools NNLS tree-consensus gate. These are the cases where the full gate returned `skip_no_valid_topology` because every successful topology cell produced one cluster.

Inputs used:

- Full gate run: `benchmarks/results/run_20260706_170815Z_full_graphtools_tree_consensus`.
- Focused p-value rerun: `reports/tree_consensus_fail_closed_pvalues_20260709`.
- External fallback audit source: `benchmarks/results/run_20260630_202253Z_full_performance_grid_default/full_benchmark_comparison.csv`.

Gate thresholds in the focused rerun: `edge_alpha=0.001`, `sibling_alpha=0.01`. Reported edge p-values are BH-corrected child-parent p-values. Reported active sibling p-values are `Sibling_Divergence_P_Value_Corrected`, because that is what controls `Sibling_BH_Different` in this frozen setup. Sparse/dense sibling p-values are diagnostic evidence columns; they do not by themselves open the live split gate in this run.

## Executive Conclusion

The 14 cases split into two failure modes:

- **9 edge-gate failures**: no successful topology has a corrected edge p-value below `0.001` anywhere relevant; the split prerequisites close before sibling testing can matter.
- **5 sibling-gate failures after edge support**: edge traversal opens, sometimes very strongly, but no full-edge node has `sibling_gate_open=True`; the live traversal visits only the root and returns `boundary`.

This is not a topology selector problem in these cases. The selector never gets a valid multi-cluster candidate to rank. It is also not fixed by falling back to the weighted topology cell inside the same adaptive graphtools NNLS grid: weighted returns one cluster for all 14 cases where it runs.

## Loss Taxonomy

|   case | case_id                             |   truth_k |   ok |   skip | loss                                |   root_edge_open |   sig_edges_all |   edge_continue_nodes |   sibling_open_nodes |   min_root_edge_bh |   min_any_edge_bh |   min_sibling_corr |   max_ari |
|-------:|:------------------------------------|----------:|-----:|-------:|:------------------------------------|-----------------:|----------------:|----------------------:|---------------------:|-------------------:|------------------:|-------------------:|----------:|
|     13 | gauss_clear_medium_continuous       |         4 |    7 |      1 | edge_gate_closed_global             |                0 |               0 |                     0 |                    0 |           0.421    |         0.421     |                    |         0 |
|     14 | gauss_moderate_3c_continuous        |         3 |    7 |      1 | edge_gate_closed_global             |                0 |               0 |                     0 |                    0 |           0.621    |         0.621     |                    |         0 |
|     23 | dim_consolidated_4c_24f_continuous  |         4 |    7 |      1 | edge_gate_closed_global             |                0 |               0 |                     0 |                    0 |           0.0807   |         0.0807    |                    |         0 |
|     24 | dim_consolidated_4c_72f_continuous  |         4 |    7 |      1 | edge_gate_closed_global             |                0 |               0 |                     0 |                    0 |           0.174    |         0.174     |                    |         0 |
|     25 | dim_diffuse_6c_136f_continuous      |         6 |    7 |      1 | edge_gate_closed_global             |                0 |               0 |                     0 |                    0 |           0.0119   |         0.0119    |                    |         0 |
|     29 | gauss_single_outlier_4c_continuous  |         5 |    7 |      1 | edge_gate_closed_global             |                0 |               0 |                     0 |                    0 |           0.456    |         0.456     |                    |         0 |
|     30 | gauss_outlier_cluster_4c_continuous |         5 |    7 |      1 | edge_gate_closed_global             |                0 |               0 |                     0 |                    0 |           0.637    |         0.637     |                    |         0 |
|     60 | sbm_moderate                        |         3 |    8 |      0 | sibling_gate_closed_after_edge_open |                3 |              26 |                    15 |                    0 |           1.21e-10 |         2.13e-13  |             0.0222 |         0 |
|     61 | sbm_hard                            |         3 |    8 |      0 | sibling_gate_closed_after_edge_open |                3 |               6 |                     3 |                    0 |           4.85e-09 |         4.85e-09  |             0.0126 |         0 |
|     88 | overlap_heavy_8c_large_feat         |         8 |    7 |      1 | sibling_gate_closed_after_edge_open |                5 |            3930 |                  1970 |                    0 |           1.09e-37 |         3.21e-167 |             0.0995 |         0 |
|     89 | overlap_extreme_4c                  |         4 |    8 |      0 | sibling_gate_closed_after_edge_open |                4 |             631 |                   317 |                    0 |           1.22e-13 |         2.75e-52  |             0.194  |         0 |
|    114 | mp_spike_above_bbp_continuous       |         2 |    7 |      1 | edge_gate_closed_global             |                0 |               0 |                     0 |                    0 |           0.568    |         0.568     |                    |         0 |
|    117 | cont_lowrank_pggn_shrinkage         |         4 |    7 |      1 | sibling_gate_closed_after_edge_open |                5 |              65 |                    35 |                    0 |           8.83e-16 |         4.61e-70  |             0.0819 |         0 |
|    119 | phylo_brownian_null_16taxa          |        16 |    7 |      1 | edge_gate_closed_global             |                0 |               0 |                     0 |                    0 |           0.0335   |         0.0335    |                    |         0 |

## Edge-Gate Failures

These cases lose before the sibling split decision. Across successful topology cells, the root edge gate is closed in every cell and the all-node audit has zero significant edge tests at `edge_alpha=0.001`.

| case_id                             |   truth_k |   min_root_edge_bh |   median_root_edge_bh |   min_any_edge_bh |   sig_edges_all |   root_sibling_skipped |   min_sparse_diag |   min_dense_diag |
|:------------------------------------|----------:|-------------------:|----------------------:|------------------:|----------------:|-----------------------:|------------------:|-----------------:|
| gauss_clear_medium_continuous       |         4 |             0.421  |                0.532  |            0.421  |               0 |                      7 |          0.00615  |         0.16     |
| gauss_moderate_3c_continuous        |         3 |             0.621  |                0.621  |            0.621  |               0 |                      7 |          0.0658   |         0.205    |
| dim_consolidated_4c_24f_continuous  |         4 |             0.0807 |                0.838  |            0.0807 |               0 |                      7 |          3.68e-06 |         0.000117 |
| dim_consolidated_4c_72f_continuous  |         4 |             0.174  |                0.516  |            0.174  |               0 |                      7 |          0.000208 |         0.0688   |
| dim_diffuse_6c_136f_continuous      |         6 |             0.0119 |                0.86   |            0.0119 |               0 |                      7 |          0.00561  |         0.142    |
| gauss_single_outlier_4c_continuous  |         5 |             0.456  |                0.456  |            0.456  |               0 |                      7 |          2.15e-08 |         9.22e-05 |
| gauss_outlier_cluster_4c_continuous |         5 |             0.637  |                0.637  |            0.637  |               0 |                      7 |          0.0111   |         0.00231  |
| mp_spike_above_bbp_continuous       |         2 |             0.568  |                0.568  |            0.568  |               0 |                      7 |          0.0355   |         1        |
| phylo_brownian_null_16taxa          |        16 |             0.0335 |                0.0335 |            0.0335 |               0 |                      7 |          0.000138 |         0.00244  |

Interpretation:

- `gauss_clear_medium_continuous`, `gauss_moderate_3c_continuous`, `dim_consolidated_4c_24f_continuous`, `dim_consolidated_4c_72f_continuous`, `dim_diffuse_6c_136f_continuous`, `gauss_single_outlier_4c_continuous`, `gauss_outlier_cluster_4c_continuous`, `mp_spike_above_bbp_continuous`, and `phylo_brownian_null_16taxa` are edge-support failures under the current diffusion/tree/NNLS geometry.
- The closest case is `dim_diffuse_6c_136f_continuous` with `min_any_edge_bh=0.0119`, still above the `0.001` edge threshold by roughly one order of magnitude.
- Some diagnostic sparse/dense p-values are small in outlier or phylogenetic cases, but those diagnostics do not open the edge prerequisite and therefore cannot rescue the split.

## Sibling-Gate Failures After Edge Support

These cases have corrected edge support in at least one topology. Full-edge traversal can walk below the root, but no node has an active sibling-divergence decision below `sibling_alpha=0.01`; passthrough also has no descendant split to target.

| case_id                     |   truth_k |   root_edge_open |   sig_edges_all |   edge_continue_nodes |   max_edge_depth |   min_sibling_corr |   sib_corr_sig_nodes |   sibling_open_nodes |   min_sparse_diag |   min_dense_diag |
|:----------------------------|----------:|-----------------:|----------------:|----------------------:|-----------------:|-------------------:|---------------------:|---------------------:|------------------:|-----------------:|
| sbm_moderate                |         3 |                3 |              26 |                    15 |                6 |             0.0222 |                    0 |                    0 |          2.57e-20 |         3.81e-25 |
| sbm_hard                    |         3 |                3 |               6 |                     3 |                1 |             0.0126 |                    0 |                    0 |          4.72e-20 |         1.36e-10 |
| overlap_heavy_8c_large_feat |         8 |                5 |            3930 |                  1970 |               31 |             0.0995 |                    0 |                    0 |          0.000574 |         6.6e-46  |
| overlap_extreme_4c          |         4 |                4 |             631 |                   317 |               13 |             0.194  |                    0 |                    0 |          4.3e-06  |         3.57e-33 |
| cont_lowrank_pggn_shrinkage |         4 |                5 |              65 |                    35 |                7 |             0.0819 |                    0 |                    0 |          0.016    |         0.00852  |

Interpretation:

- `sbm_hard` is the closest sibling-gate miss: the best corrected sibling p-value is `0.0126`, just above `0.01`. This suggests threshold sensitivity, but changing it would be a scientific decision with type-I implications.
- `sbm_moderate` has strong edge evidence (`min_any_edge_bh=2.13e-13`) but best corrected sibling p-value `0.0222`, so it fails at sibling support rather than topology construction.
- `overlap_heavy_8c_large_feat`, `overlap_extreme_4c`, and `cont_lowrank_pggn_shrinkage` show very strong edge and dense/sparse diagnostic p-values, but the active sibling correction remains far above `0.01` (`0.0995`, `0.194`, `0.0819` respectively). These are not near misses under the active sibling gate.
- The important scientific point: edge support is not enough. The current live rule requires sibling evidence at the candidate split, and none of these topologies produce it.

## Skipped Topology Cells

There were 11 skipped cells in the 14-case diagnostic subset:

| skipped topology | count | affected cases | reason |
|:-----------------|------:|:---------------|:-------|
| neighbor_joining | 10 | `gauss_clear_medium_continuous`, `gauss_moderate_3c_continuous`, `dim_consolidated_4c_24f_continuous`, `dim_consolidated_4c_72f_continuous`, `dim_diffuse_6c_136f_continuous`, `gauss_single_outlier_4c_continuous`, `gauss_outlier_cluster_4c_continuous`, `mp_spike_above_bbp_continuous`, `cont_lowrank_pggn_shrinkage`, `phylo_brownian_null_16taxa` | MAD rooting requires positive distances between leaf pairs. |
| ward | 1 | `overlap_heavy_8c_large_feat` | Sibling inflation calibration has no strict-null or stopped-edge empirical-null calibration records with positive weight; selected non-null positive-weight rows are intentionally rejected as empirical-null support. |

Interpretation:

- Ten skips are neighbor-joining/MAD-rooting precondition failures: `MAD rooting requires positive distances between leaf pairs`. This is a geometry/precondition issue, not a scientific split decision.
- One skip is the ward cell for `overlap_heavy_8c_large_feat`: the sibling inflation calibration refuses to use selected non-null positive-weight rows as empirical-null support. That is the intended fail-closed support contract.
- These skip reasons are not licensing failures. Licensing remains a packaging/dependency question for optional graphtools usage: local package metadata reports `graphtools 2.1.0` and `tasklogger 1.2.0` as GPLv2. The observed skipped cells here are runtime support/precondition failures.

## Fallback Audit

This audit is external-metric based and is not part of the label-free selector. It is included only to inform production fallback policy.

| case_id                             | weighted_status   |   weighted_k |   weighted_ari | tbs_status   |   tbs_k |   tbs_ari |   tbs_nmi |   tbs_f1 | best_default   |   best_k |   best_ari |   best_nmi |   best_f1 |
|:------------------------------------|:------------------|-------------:|---------------:|:-------------|--------:|----------:|----------:|---------:|:---------------|---------:|-----------:|-----------:|----------:|
| cont_lowrank_pggn_shrinkage         | ok                |            1 |              0 | skip         |       0 |           |           |          | leiden         |        5 |    0.0363  |     0.0899 |     0.447 |
| dim_consolidated_4c_24f_continuous  | ok                |            1 |              0 | ok           |       4 |   1       |     1     |  1       | tbs            |        4 |    1       |     1      |     1     |
| dim_consolidated_4c_72f_continuous  | ok                |            1 |              0 | ok           |       1 |   0       |     0     |  0.1     | leiden         |        4 |    1       |     1      |     1     |
| dim_diffuse_6c_136f_continuous      | ok                |            1 |              0 | ok           |       1 |   0       |     0     |  0.0476  | spectral       |        6 |    0.0827  |     0.135  |     0.321 |
| gauss_clear_medium_continuous       | ok                |            1 |              0 | ok           |       1 |   0       |     0     |  0.1     | leiden         |        4 |    1       |     1      |     1     |
| gauss_moderate_3c_continuous        | ok                |            1 |              0 | ok           |       1 |   0       |     0     |  0.167   | leiden         |        3 |    1       |     1      |     1     |
| gauss_outlier_cluster_4c_continuous | ok                |            1 |              0 | ok           |       2 |   0.461   |     0.634 |  0.26    | leiden         |        5 |    1       |     1      |     1     |
| gauss_single_outlier_4c_continuous  | ok                |            1 |              0 | ok           |       2 |   0.489   |     0.657 |  0.266   | kmeans         |        5 |    1       |     1      |     1     |
| mp_spike_above_bbp_continuous       | ok                |            1 |              0 | ok           |       1 |   0       |     0     |  0.333   | spectral       |        2 |    0.00537 |     0.01   |     0.557 |
| overlap_extreme_4c                  | ok                |            1 |              0 | skip         |       0 |           |           |          | kmeans         |        4 |    0.00818 |     0.0146 |     0.308 |
| overlap_heavy_8c_large_feat         | ok                |            1 |              0 | skip         |       0 |           |           |          | kmeans         |        8 |    0.092   |     0.128  |     0.313 |
| phylo_brownian_null_16taxa          | ok                |            1 |              0 | ok           |       1 |   0       |     0     |  0.00735 | leiden         |       16 |    1       |     1      |     1     |
| sbm_hard                            | ok                |            1 |              0 | ok           |       4 |  -0.00856 |     0.021 |  0.36    | louvain        |        7 |    0.0261  |     0.0549 |     0.472 |
| sbm_moderate                        | ok                |            1 |              0 | ok           |       8 |   0.0995  |     0.182 |  0.62    | spectral       |        3 |    0.271   |     0.218  |     0.649 |

Fallback implications:

- **Fallback to adaptive weighted topology is not useful** for these 14 cases. Weighted is either one cluster with ARI `0` or not the selected rescue; it shares the same gate failure.
- **Fallback to current `tbs` helps only a subset**. It perfectly recovers `dim_consolidated_4c_24f_continuous` and partially recovers the two outlier Gaussian cases plus `sbm_moderate`, but it still returns one cluster or skips several cases.
- **Other default methods show recoverable signal in several cases**, especially Gaussian and phylogenetic synthetic cases. That means these are not all impossible datasets; the adaptive graphtools NNLS TBS gate is specifically failing to validate a split under its current edge/sibling evidence rules.

## Production Recommendation

For production promotion of `tbs_diffusion_graphtools_adaptive_nnls`, treat these 14 as unsupported by the adaptive graphtools NNLS gate unless a deliberate fallback policy is added.

Recommended fallback order for now:

1. If the adaptive graphtools NNLS consensus selector returns a valid multi-cluster topology, use it.
2. If it returns `skip_no_valid_topology`, report `unsupported_by_adaptive_graphtools_nnls` with the loss bucket when available.
3. Optionally expose an explicit user-selected fallback to current `tbs`, but do not silently promote it as equivalent. It improves only some fail-closed cases and has its own support failures.
4. Do not fallback to the weighted topology cell inside the same grid for these cases; the evidence says it does not rescue them.

The next scientific development target is not another tie-breaker in the consensus selector. It is the split-evidence layer: edge calibration for continuous/diffusion geometries, and sibling calibration for graph/overlap/low-rank cases where edge evidence is strong but sibling divergence does not validate.

## Generated Artifacts

- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_loss_taxonomy.csv`
- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_traversal_trace.csv`
- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_traversal_counters.csv`
- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_fallback_external_audit.csv`
- `reports/tree_consensus_fail_closed_pvalues_20260709/fail_closed_fallback_compact.csv`
