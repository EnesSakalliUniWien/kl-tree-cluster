# raw cosine subspace

Method family: `raw_cosine_subspace`
Method description: Raw cosine eigenspace component-block tree: the PosetTree gate is run directly on cosine eigenspace blocks without adaptive diffusion smoothing.
Trees in this folder: `15`

This folder is method-isolated. It contains no trees, CSV rows, or pages from other method families.
PDF cover page defines the method and the plotted values.

## Files

- `allgo_new_quality_aware_go_ic_raw_cosine_subspace_tree_ranking.csv`
- `allgo_new_quality_aware_go_ic_raw_cosine_subspace_cluster_coherence_long.csv`
- `allgo_new_quality_aware_go_ic_raw_cosine_subspace_tfidf_quality_long.csv`
- `allgo_new_quality_aware_go_ic_raw_cosine_subspace_tree_pages.pdf`
- `allgo_new_quality_aware_go_ic_raw_cosine_subspace_top_trees.png`
- `allgo_new_quality_aware_go_ic_raw_cosine_subspace_quality_scatter.png`
- `tree_pages/*.png`

## Best Tree In This Method

 method_rank  display_rank  raw_go_ic_rank                                               run_id quality_tier_label  go_bic_active_per_gene  n_clusters  coherent_cluster_count  coherent_cluster_fraction  singleton_gene_fraction  largest_cluster_fraction  weighted_mean_within_tfidf_cosine
           1            10              29     raw_cosine_subspace__tfidf__adaptive_modes_16_19  quality_plausible             2005.867846          76                      23                   0.302632                 0.024917                  0.088040                           0.111223
           2            11              30     raw_cosine_subspace__tfidf__adaptive_modes_53_80  quality_plausible             2029.284359         133                      71                   0.533835                 0.006645                  0.019934                           0.212193
           3            13              14 raw_cosine_subspace__binary__adaptive_common_mode_01     broad_coherent             1447.097567           3                       2                   0.666667                 0.000000                  0.548173                           0.061682
           4            14               7    raw_cosine_subspace__binary__adaptive_modes_22_40      quality_mixed              460.595675         399                      30                   0.075188                 0.476744                  0.013289                           0.311172
           5            15               8    raw_cosine_subspace__binary__adaptive_modes_12_21      quality_mixed              532.764523         362                      26                   0.071823                 0.438538                  0.021595                           0.265306
           6            16               9     raw_cosine_subspace__tfidf__adaptive_modes_02_05      quality_mixed              834.689449         256                      19                   0.074219                 0.294020                  0.058140                           0.180550
           7            17              10     raw_cosine_subspace__tfidf__adaptive_modes_06_10      quality_mixed              994.993167         233                      20                   0.085837                 0.250831                  0.064784                           0.188812
           8            19              13     raw_cosine_subspace__tfidf__adaptive_modes_20_31      quality_mixed             1446.871572         203                      55                   0.270936                 0.112957                  0.021595                           0.247971

