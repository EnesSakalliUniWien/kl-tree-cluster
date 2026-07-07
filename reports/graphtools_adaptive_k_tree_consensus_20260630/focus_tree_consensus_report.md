# Graphtools Adaptive-K NNLS Tree Consensus Report

Date: 2026-06-30

## Scope

This report compares the eight accessible graphtools adaptive-K NNLS topology strategies on the seven focus cases from the tree-inference panel. The final selector is label-free: it uses internal cluster validity, a small parsimony term, a dominant-cluster guard, and partition agreement only as a tie-break. ARI, NMI, macro F1, and purity are external audit metrics and are not used for selection.

## Final Frozen Selector

For each case, candidate topologies are average, complete, weighted, single, centroid, median, Ward, and MAD-rooted neighbor joining over the same graphtools adaptive-K diffusion distance; branch lengths are then refit by fixed-topology NNLS. Invalid one-cluster/internal-metric rows are fail-closed. The score is mean rank of silhouette (high), Calinski-Harabasz (high), and Davies-Bouldin (low), plus 0.25 times the within-case rank of cluster count (low), plus a 3.0 penalty when the largest cluster fraction is at least 0.45. Ties use mean pairwise partition agreement, the internal score, cluster-count rank, then a deterministic method priority that prefers weighted over equivalent single/centroid/neighbor-joining partitions.

## Selected Topologies

| case_id                 | selected_tree_inference   |   selected_found_clusters |   selected_mean_partition_agreement |   selected_ari |   selected_nmi |   selected_macro_f1 |   delta_ari_vs_average | external_best_tree_inference   |   external_best_ari |
|:------------------------|:--------------------------|--------------------------:|------------------------------------:|---------------:|---------------:|--------------------:|-----------------------:|:-------------------------------|--------------------:|
| cat_overlap_3cat_4c     | complete                  |                         4 |                               0.857 |          0.924 |          0.918 |               0.970 |                  0.028 | centroid                       |               0.936 |
| overlap_unbal_4c_small  | average                   |                         4 |                               0.303 |          0.659 |          0.611 |               0.860 |                  0.000 | average                        |               0.659 |
| overlap_mod_4c_small    | ward                      |                         4 |                               0.708 |          0.922 |          0.894 |               0.970 |                  0.020 | ward                           |               0.922 |
| dim_consolidated_4c_24f | neighbor_joining          |                         4 |                               0.426 |          0.687 |          0.682 |               0.876 |                  0.283 | neighbor_joining               |               0.687 |
| cat_highd_3cat_500feat  | weighted                  |                         4 |                               1.000 |          1.000 |          1.000 |               1.000 |                  0.000 | average                        |               1.000 |
| gauss_overlap_3c_small  | weighted                  |                         3 |                               0.768 |          1.000 |          1.000 |               1.000 |                  0.508 | weighted                       |               1.000 |
| gauss_overlap_8c_highd  | weighted                  |                         8 |                               1.000 |          1.000 |          1.000 |               1.000 |                  0.000 | average                        |               1.000 |

## Aggregate Audit

| selector                                                |   cases |   mean_selected_ari |   median_selected_ari |   mean_selected_nmi |   mean_selected_macro_f1 |   mean_delta_ari_vs_average |   mean_delta_nmi_vs_average |   mean_delta_macro_f1_vs_average |   mean_delta_ari_vs_weighted |   mean_delta_nmi_vs_weighted |   mean_delta_macro_f1_vs_weighted |   mean_delta_ari_vs_external_best |   mean_delta_nmi_vs_external_best |   mean_delta_macro_f1_vs_external_best |   external_best_mean_ari |   average_mean_ari |   weighted_mean_ari |
|:--------------------------------------------------------|--------:|--------------------:|----------------------:|--------------------:|-------------------------:|----------------------------:|----------------------------:|---------------------------------:|-----------------------------:|-----------------------------:|----------------------------------:|----------------------------------:|----------------------------------:|---------------------------------------:|-------------------------:|-------------------:|--------------------:|
| sharp_label_free_internal_fit_parsimony_dominance_guard |       7 |               0.885 |                 0.924 |               0.872 |                    0.954 |                       0.120 |                       0.060 |                            0.007 |                        0.053 |                        0.028 |                             0.017 |                            -0.002 |                            -0.002 |                                 -0.001 |                    0.886 |              0.765 |               0.832 |

## Tree-Method Summary

| tree_inference   |   selection_count |   mean_ari |   median_ari |   mean_nmi |   mean_macro_f1 |   mean_silhouette_score |   mean_davies_bouldin_index |   mean_cluster_count_abs_error |   mean_partition_agreement |   valid_internal_case_count |   mean_sharp_score |
|:-----------------|------------------:|-----------:|-------------:|-----------:|----------------:|------------------------:|----------------------------:|-------------------------------:|---------------------------:|----------------------------:|-------------------:|
| weighted         |                 3 |      0.832 |        0.849 |      0.844 |           0.936 |                   0.125 |                       3.297 |                          0.571 |                      0.712 |                           7 |              5.720 |
| ward             |                 1 |      0.770 |        0.901 |      0.817 |           0.932 |                   0.097 |                       3.338 |                          0.571 |                      0.693 |                           7 |              5.542 |
| average          |                 1 |      0.765 |        0.896 |      0.812 |           0.946 |                   0.096 |                       3.286 |                          1.429 |                      0.701 |                           7 |              5.625 |
| neighbor_joining |                 1 |      0.752 |        0.709 |      0.810 |           0.792 |                   0.129 |                       2.993 |                          0.571 |                      0.657 |                           7 |              5.250 |
| complete         |                 1 |      0.712 |        0.667 |      0.788 |           0.905 |                   0.092 |                       3.489 |                          2.143 |                      0.666 |                           7 |              6.583 |
| centroid         |                 0 |      0.771 |        0.936 |      0.779 |           0.849 |                   0.142 |                       2.680 |                          1.286 |                      0.730 |                           6 |              4.833 |
| median           |                 0 |      0.720 |        0.875 |      0.739 |           0.830 |                   0.123 |                       2.730 |                          1.429 |                      0.695 |                           6 |              6.771 |
| single           |                 0 |      0.530 |        0.713 |      0.551 |           0.569 |                   0.187 |                       2.378 |                          1.429 |                      0.542 |                           4 |              4.771 |

## Recursive Analysis Notes

- The initial broad Borda rule was rejected because it rewarded high effective cluster count and very small largest-cluster fraction, which can favor fragmentation; this specifically selected average on `dim_consolidated_4c_24f` despite weaker internal separation than neighbor joining.
- The final rule treats cluster count as a small parsimony cost and treats dominant clusters as a collapse/under-split warning. It does not reward fragmentation.
- `gauss_overlap_3c_small` is the clearest topology failure of average linkage: average fragments into eight clusters, while weighted/single/centroid/neighbor joining recover the same three-cluster partition. The final tie rule selects weighted because it is linkage-native and much faster than neighbor joining when the partition is equivalent.
- `dim_consolidated_4c_24f` favors neighbor joining by label-free internal fit: best silhouette and Calinski-Harabasz, acceptable balance, and four clusters without using the known true count.
- `cat_overlap_3cat_4c` shows why internal fit alone is insufficient: single and neighbor joining have strong silhouette/CH but a dominant half-sample cluster. The dominant-cluster guard moves selection to complete linkage, near the external best centroid result.
- `cat_highd_3cat_500feat` and `gauss_overlap_8c_highd` have near-perfect method agreement; topology selection is effectively irrelevant there.

## Consensus

- A single global tree inference method is not supported by the focus panel. Weighted has the strongest global mean ARI among fixed tree methods, but no single topology dominates all cases.
- The scientifically defensible sharp version is a predeclared label-free topology-selection layer over candidate tree builders/linkages, followed by NNLS branch-time fitting on the chosen topology.
- Average linkage should remain a baseline, not the default final topology rule for adaptive diffusion NNLS. It is repaired by the selector on the small Gaussian-overlap and dimensional-consolidation cases.
- Promotion from experimental to default should require the same frozen selector rerun on the full benchmark grid; the focus panel is strong enough to motivate the method, not enough to claim production superiority.

## Output Files

- `focus_tree_method_label_assignments.csv`
- `focus_tree_method_run_status.csv`
- `focus_tree_method_pairwise_agreement.csv`
- `focus_tree_method_stability.csv`
- `focus_tree_label_free_consensus_rankings.csv`
- `focus_tree_label_free_consensus_selection.csv`
- `focus_tree_consensus_method_summary.csv`
- `focus_tree_consensus_selection_summary.csv`
- `focus_tree_case_agreement_summary.csv`
