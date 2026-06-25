# scRNA Distributional Action Audit

Generated at: 2026-06-24T20:48:26+02:00

## Summary

The check confirms the concern: descendant leaf count is only sample mass. It is
not enough to describe how much a tree edge or internal node changes the
distribution. The audit adds a mass-weighted movement score

`subtree_distributional_action = child_leaf_count * mean_standardized_PCA_delta^2`

where the delta is the child barycenter minus the parent barycenter in the saved
benchmark PCA space. This is the between-subtree contribution that was missing
from the branch-length-only discussion.

Branch length and distributional action are related but not interchangeable.
Top-action edges do not perfectly overlap the largest subtrees or the longest
branches, so the report/plots should include distributional action when we
interpret which internal nodes matter.

## Method

1. Read `benchmark_subset_pca.csv` and each canonical TBS `*_tree_edges.csv`.
2. Reconstruct every node barycenter bottom-up from `L0`, `L1`, ... leaf rows.
3. Standardize each PCA coordinate over the benchmark subset.
4. For every edge, compute child-parent displacement, child mass, branch length,
   edge-test metadata, and action scores.
5. Join Goncalves progenitor node annotations when the edge child or parent is a
   labeled progenitor analysis node.

The audit does not rerun clustering and does not change TBS. It is a diagnostic
over current benchmark artifacts.

## Adaptive-Diffusion Topology Summary

| dataset         | dataset_label            | geometry           | variant       | method                                                                       |   n_edges |   n_internal_child_edges |   edge_open_count |   split_filter_filtered_count |   split_filter_pass_count |   median_child_leaf_count |   median_standardized_delta_norm |   median_subtree_distributional_action |   max_subtree_distributional_action |   spearman_action_vs_leaf_count |   spearman_action_vs_branch_length |   spearman_action_vs_edge_statistic_tested |   spearman_internal_action_vs_leaf_count |   spearman_internal_action_vs_branch_length |   top10_action_leaf_count_overlap |   top10_action_branch_length_overlap | top_action_edge   |   top_action_child_leaf_count |   top_action_standardized_delta_norm |   top_action_value |   top_action_branch_length |
|:----------------|:-------------------------|:-------------------|:--------------|:-----------------------------------------------------------------------------|----------:|-------------------------:|------------------:|------------------------------:|--------------------------:|--------------------------:|---------------------------------:|---------------------------------------:|------------------------------------:|--------------------------------:|-----------------------------------:|-------------------------------------------:|-----------------------------------------:|--------------------------------------------:|----------------------------------:|-------------------------------------:|:------------------|------------------------------:|-------------------------------------:|-------------------:|---------------------------:|
| adult_pancreas  | Adult pancreas           | adaptive_diffusion | topology_only | TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 |      4998 |                     2498 |               142 |                             0 |                      4998 |                    1.0000 |                           1.1181 |                                 0.0699 |                             79.6280 |                          0.0760 |                             0.3672 |                                     0.4482 |                                   0.1641 |                                      0.4283 |                                 0 |                                    3 | N4982->N4943      |                            56 |                               6.5313 |            79.6280 |                     0.2649 |
| goncalves_fetal | Goncalves fetal pancreas | adaptive_diffusion | topology_only | TBS adaptive diffusion topology projected adaptive-k90 alpha=0.01 edge=0.001 |      2928 |                     1463 |                98 |                             0 |                      2928 |                    1.0000 |                           2.1548 |                                 0.2687 |                             34.0326 |                         -0.0334 |                             0.2097 |                                     0.5468 |                                   0.0541 |                                      0.1732 |                                 0 |                                    4 | N2927->N2926      |                            63 |                               4.0257 |            34.0326 |                     0.0328 |

## PCA-Linkage Topology Summary

| dataset         | dataset_label            | geometry    | variant       | method                                                    |   n_edges |   n_internal_child_edges |   edge_open_count |   split_filter_filtered_count |   split_filter_pass_count |   median_child_leaf_count |   median_standardized_delta_norm |   median_subtree_distributional_action |   max_subtree_distributional_action |   spearman_action_vs_leaf_count |   spearman_action_vs_branch_length |   spearman_action_vs_edge_statistic_tested |   spearman_internal_action_vs_leaf_count |   spearman_internal_action_vs_branch_length |   top10_action_leaf_count_overlap |   top10_action_branch_length_overlap | top_action_edge   |   top_action_child_leaf_count |   top_action_standardized_delta_norm |   top_action_value |   top_action_branch_length |
|:----------------|:-------------------------|:------------|:--------------|:----------------------------------------------------------|----------:|-------------------------:|------------------:|------------------------------:|--------------------------:|--------------------------:|---------------------------------:|---------------------------------------:|------------------------------------:|--------------------------------:|-----------------------------------:|-------------------------------------------:|-----------------------------------------:|--------------------------------------------:|----------------------------------:|-------------------------------------:|:------------------|------------------------------:|-------------------------------------:|-------------------:|---------------------------:|
| adult_pancreas  | Adult pancreas           | pca_linkage | topology_only | TBS topology projected adaptive-k90 alpha=0.01 edge=0.001 |      4998 |                     2498 |               214 |                             0 |                      4998 |                    1.0000 |                           0.9253 |                                 0.0497 |                             80.7675 |                          0.1997 |                             0.2649 |                                     0.5096 |                                   0.1814 |                                      0.4939 |                                 0 |                                    1 | N4996->N4991      |                            51 |                               6.8928 |            80.7675 |                     0.1480 |
| goncalves_fetal | Goncalves fetal pancreas | pca_linkage | topology_only | TBS topology projected adaptive-k90 alpha=0.01 edge=0.001 |      2928 |                     1463 |               150 |                             0 |                      2928 |                    1.0000 |                           1.9973 |                                 0.2077 |                             31.9647 |                          0.0497 |                             0.2246 |                                     0.4942 |                                  -0.0655 |                                      0.3429 |                                 0 |                                    0 | N2927->N2914      |                            87 |                               3.3200 |            31.9647 |                     0.1769 |

## Top Adult Adaptive-Diffusion Internal Edges By Action

| parent   | child   |   child_leaf_count |   standardized_delta_norm |   subtree_distributional_action |   branch_length |   edge_test_statistic | edge_significant   | child_progenitor_interpretation   | child_top_populations   |
|:---------|:--------|-------------------:|--------------------------:|--------------------------------:|----------------:|----------------------:|:-------------------|:----------------------------------|:------------------------|
| N4982    | N4943   |                 56 |                    6.5313 |                         79.6280 |          0.2649 |            29887.4285 | True               |                                   |                         |
| N4998    | N4978   |                  8 |                   17.1360 |                         78.3043 |          0.6321 |            13657.1346 | True               |                                   |                         |
| N4997    | N4981   |                 11 |                   14.4747 |                         76.8225 |          0.3993 |            26016.5204 | True               |                                   |                         |
| N4966    | N4929   |                209 |                    3.2940 |                         75.5893 |          0.1132 |            20287.8839 | True               |                                   |                         |
| N4990    | N4970   |                 34 |                    7.8679 |                         70.1579 |          0.2162 |             9668.5726 | True               |                                   |                         |
| N4979    | N4939   |                 81 |                    5.0638 |                         69.2322 |          0.2140 |            13404.7835 | True               |                                   |                         |
| N4986    | N4960   |                 33 |                    7.5673 |                         62.9910 |          0.2808 |             9706.7003 | True               |                                   |                         |
| N4968    | N4954   |                158 |                    3.3206 |                         58.0712 |          0.0982 |            31786.5215 | True               |                                   |                         |
| N4994    | N4959   |                 20 |                    9.1091 |                         55.3166 |          0.3759 |             3661.7225 | True               |                                   |                         |
| N4977    | N4908   |                 88 |                    4.2846 |                         53.8488 |          0.2499 |             5244.4152 | True               |                                   |                         |

## Top Goncalves Adaptive-Diffusion Internal Edges By Action

| parent   | child   |   child_leaf_count |   standardized_delta_norm |   subtree_distributional_action |   branch_length |   edge_test_statistic | edge_significant   | child_progenitor_interpretation                | child_top_populations                                                      |
|:---------|:--------|-------------------:|--------------------------:|--------------------------------:|----------------:|----------------------:|:-------------------|:-----------------------------------------------|:---------------------------------------------------------------------------|
| N2927    | N2926   |                 63 |                    4.0257 |                         34.0326 |          0.0328 |             3699.6606 | True               | non-progenitor or broad mixed grouping         | unknown:63                                                                 |
| N2921    | N2895   |                 36 |                    4.7674 |                         27.2736 |          0.3495 |             1956.7645 | True               | non-progenitor or broad mixed grouping         | blood:36                                                                   |
| N2925    | N2910   |                 47 |                    4.0618 |                         25.8469 |          0.2584 |             1541.6289 | True               | mixed fetal progenitor-state grouping          | proliferating:40; mesenchyme:6; trunk:1                                    |
| N2914    | N2896   |                 97 |                    2.8125 |                         25.5765 |          0.1690 |             2003.4598 | True               | mixed grouping with partial progenitor content | proliferating:69; mesenchyme:22; trunk:4; unknown:2                        |
| N2924    | N2912   |                 45 |                    4.1016 |                         25.2344 |          0.2273 |             1475.4313 | True               | mixed grouping with partial progenitor content | blood:16; trunk:14; proliferating:12; unknown:1; endocrine:1; mesenchyme:1 |
| N2920    | N2898   |                 42 |                    4.1362 |                         23.9517 |          0.3207 |             1853.1577 | True               | broad mixed progenitor-rich neighborhood       | trunk:31; mesenchyme:5; proliferating:4; blood:2                           |
| N2916    | N2888   |                 73 |                    3.0776 |                         23.0478 |          0.2616 |             1370.6345 | True               | mixed grouping with partial progenitor content | proliferating:47; mesenchyme:17; trunk:3; neurons:3; tip:2; endocrine:1    |
| N2928    | N2827   |                 14 |                    6.8499 |                         21.8966 |          0.7539 |             1208.4111 | True               | non-progenitor or broad mixed grouping         | neurons:14                                                                 |
| N2922    | N2851   |                 31 |                    4.5947 |                         21.8148 |          0.4736 |             1055.9108 | True               | tip-progenitor-enriched grouping               | tip:31                                                                     |
| N2919    | N2917   |                 44 |                    3.8392 |                         21.6180 |          0.0687 |             1371.8495 | True               | mixed grouping with partial progenitor content | trunk:17; proliferating:12; blood:9; tip:5; mesenchyme:1                   |

## Interpretation

- `leaf_count` is already present and useful, but it only measures how many
  cells descend from a node.
- `standardized_delta_norm` measures how far the child distribution moves away
  from the parent distribution per cell.
- `subtree_distributional_action` combines both. A node can be important by
  being large, by moving strongly, or by doing both.
- NNLS branch lengths still fit additive path distances. They do not currently
  optimize or display this mass-weighted distributional-action quantity.
- Edge projected-Wald statistics include sample-size effects and covariance
  normalization, so they are closer to the desired notion than branch length,
  but they are threshold/test statistics rather than a direct visual
  contribution score.

## Output Files

- `scrna_distributional_action_edges.csv`
- `scrna_distributional_action_method_summary.csv`
- `distributional_action_vs_branch_length.png`
- `top_internal_distributional_action_edges.png`
- `distributional_action_vs_edge_statistic.png`
