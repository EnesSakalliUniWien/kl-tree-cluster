# allGO New Quality-Aware GO-IC Tree Summary

Input: `data/feature_matrices/feature_matrix_julia_allGO_new.tsv`
Trees scored: `31`

Primary display ordering: quality tier, then GO annotation Bernoulli BIC with active within-cluster GO parameters.
The raw GO-IC-only order is preserved in `raw_go_ic_rank` because near-singleton trees can overfit GO labels.
Quality tiers: `quality_plausible` requires 5-150 clusters, largest cluster <= 75%, singleton-gene fraction <= 5%, and coherent-cluster fraction >= 25%; `broad_coherent` is a small-cluster coherent fallback; `degenerate` includes one-cluster, giant-cluster, or singleton-dominated trees.
Lower `go_bic_active` is better within a quality tier.

## Top Trees

 display_rank  raw_go_ic_rank                                                       method_run_id                      method_family quality_tier_label  go_bic_active_per_gene  n_clusters  coherent_cluster_count  coherent_cluster_fraction  weighted_mean_within_tfidf_cosine  singleton_fraction  largest_cluster_fraction
            1              15                                            whole_adaptive_diffusion           whole_adaptive_diffusion  quality_plausible             1475.898994          20                      15                   0.750000                           0.143322            0.000000                  0.568106
            2              17                                legacy_c2ef__tfidf__components_02_05                        legacy_c2ef  quality_plausible             1504.867882          19                      11                   0.578947                           0.134206            0.052632                  0.215947
            3              18     adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_02_05 adaptive_diffusion_cosine_subspace  quality_plausible             1687.966840          40                      21                   0.525000                           0.149686            0.125000                  0.232558
            4              20 adaptive_diffusion_cosine_subspace__binary__adaptive_common_mode_01 adaptive_diffusion_cosine_subspace  quality_plausible             1738.172304          12                       7                   0.583333                           0.069059            0.000000                  0.290698
            5              23    adaptive_diffusion_cosine_subspace__binary__adaptive_modes_22_40 adaptive_diffusion_cosine_subspace  quality_plausible             1804.886338          56                      35                   0.625000                           0.144869            0.142857                  0.056478
            6              24    adaptive_diffusion_cosine_subspace__binary__adaptive_modes_12_21 adaptive_diffusion_cosine_subspace  quality_plausible             1873.029319          46                      28                   0.608696                           0.138813            0.108696                  0.126246
            7              25     adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_53_80 adaptive_diffusion_cosine_subspace  quality_plausible             1910.228748          22                      17                   0.772727                           0.077208            0.000000                  0.372093
            8              26     adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_16_19 adaptive_diffusion_cosine_subspace  quality_plausible             1911.302510          69                      24                   0.347826                           0.113835            0.246377                  0.112957
            9              28    adaptive_diffusion_cosine_subspace__binary__adaptive_modes_56_80 adaptive_diffusion_cosine_subspace  quality_plausible             1978.942590          60                      24                   0.400000                           0.110029            0.200000                  0.141196
           10              29                    raw_cosine_subspace__tfidf__adaptive_modes_16_19                raw_cosine_subspace  quality_plausible             2005.867846          76                      23                   0.302632                           0.111223            0.197368                  0.088040
           11              30                    raw_cosine_subspace__tfidf__adaptive_modes_53_80                raw_cosine_subspace  quality_plausible             2029.284359         133                      71                   0.533835                           0.212193            0.030075                  0.019934
           12              31    adaptive_diffusion_cosine_subspace__binary__adaptive_modes_41_55 adaptive_diffusion_cosine_subspace  quality_plausible             2047.079987          65                      28                   0.430769                           0.105871            0.169231                  0.079734

## Outputs

- `allgo_new_quality_aware_go_ic_tree_ranking.csv`
- `allgo_new_quality_aware_go_ic_top_trees.png`
- `allgo_new_quality_aware_go_ic_quality_scatter_all_trees.png`
- `allgo_new_quality_aware_go_ic_quality_scatter_plausible_trees.png`
- `allgo_new_quality_aware_go_ic_all_tree_pages.pdf`
- `tree_pages/*.png`

Compatibility aliases are also written with the earlier generic names:
`go_ic_tree_ranking.csv`, `go_ic_top_tree_ranking.png`,
`go_ic_quality_scatter.png`, and `all_tree_pages_ordered_by_go_ic.pdf`.

