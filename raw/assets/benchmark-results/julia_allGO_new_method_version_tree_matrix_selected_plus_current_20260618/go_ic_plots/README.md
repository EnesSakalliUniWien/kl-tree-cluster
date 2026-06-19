# allGO New Quality-Aware GO-IC Tree Summary

Input: `data/feature_matrices/feature_matrix_julia_allGO_new.tsv`
Trees scored: `13`

Primary display ordering: quality tier, then GO annotation Bernoulli BIC with active within-cluster GO parameters.
The raw GO-IC-only order is preserved in `raw_go_ic_rank` because near-singleton trees can overfit GO labels.
Quality tiers: `quality_plausible` requires 5-150 clusters, largest cluster <= 75%, singleton-gene fraction <= 5%, and coherent-cluster fraction >= 25%; `broad_coherent` is a small-cluster coherent fallback; `degenerate` includes one-cluster, giant-cluster, or singleton-dominated trees.
Lower `go_bic_active` is better within a quality tier.

## Top Trees

 display_rank  raw_go_ic_rank                                                                method_run_id                                   method_family quality_tier_label  go_bic_active_per_gene  n_clusters  coherent_cluster_count  coherent_cluster_fraction  weighted_mean_within_tfidf_cosine  singleton_fraction  largest_cluster_fraction
            1               3                                            current__whole_adaptive_diffusion               current__whole_adaptive_diffusion  quality_plausible             1475.898994          20                      15                   0.750000                           0.143322            0.000000                  0.568106
            2               4                legacy_c2ef__raw_cosine_subspace__tfidf__adaptive_modes_02_05                legacy_c2ef__raw_cosine_subspace  quality_plausible             1504.867882          19                      11                   0.578947                           0.134206            0.052632                  0.215947
            3               5 legacy_c2ef__adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_16_19 legacy_c2ef__adaptive_diffusion_cosine_subspace  quality_plausible             1639.592297          10                       8                   0.800000                           0.069628            0.000000                  0.601329
            4               6 legacy_c2ef__adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_02_05 legacy_c2ef__adaptive_diffusion_cosine_subspace  quality_plausible             1681.038397          32                      20                   0.625000                           0.149144            0.031250                  0.232558
            5               7     current__adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_02_05     current__adaptive_diffusion_cosine_subspace  quality_plausible             1687.966840          40                      21                   0.525000                           0.149686            0.125000                  0.232558
            6               8     current__adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_53_80     current__adaptive_diffusion_cosine_subspace  quality_plausible             1910.228748          22                      17                   0.772727                           0.077208            0.000000                  0.372093
            7               9     current__adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_16_19     current__adaptive_diffusion_cosine_subspace  quality_plausible             1911.302510          69                      24                   0.347826                           0.113835            0.246377                  0.112957
            8              10 legacy_c2ef__adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_53_80 legacy_c2ef__adaptive_diffusion_cosine_subspace  quality_plausible             1962.715359          67                      37                   0.552239                           0.117427            0.179104                  0.061462
            9              11                legacy_c2ef__raw_cosine_subspace__tfidf__adaptive_modes_16_19                legacy_c2ef__raw_cosine_subspace  quality_plausible             1968.363193          27                      15                   0.555556                           0.083782            0.037037                  0.136213
           10              12                    current__raw_cosine_subspace__tfidf__adaptive_modes_16_19                    current__raw_cosine_subspace  quality_plausible             2005.867846          76                      23                   0.302632                           0.111223            0.197368                  0.088040
           11              13                    current__raw_cosine_subspace__tfidf__adaptive_modes_53_80                    current__raw_cosine_subspace  quality_plausible             2029.284359         133                      71                   0.533835                           0.212193            0.030075                  0.019934
           12               1                    current__raw_cosine_subspace__tfidf__adaptive_modes_02_05                    current__raw_cosine_subspace      quality_mixed              834.689449         256                      19                   0.074219                           0.180550            0.691406                  0.058140
           13               2                legacy_c2ef__raw_cosine_subspace__tfidf__adaptive_modes_53_80                legacy_c2ef__raw_cosine_subspace         degenerate             1438.657296           1                       0                   0.000000                           0.036854            0.000000                  1.000000

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

