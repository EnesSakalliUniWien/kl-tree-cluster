# Cosine Subspace Split Validation

Reference split: TF-IDF cosine components 2-5 tree with the project split method.

Genes: `602`
Non-empty deduplicated GO terms: `6368`
Reference clusters: `19`

## Biological Coherence

Clusters passing the coherence rule: `11` of `19`.
Non-singleton/non-tiny clusters (size >= 3): `15`.

Coherence rule: cluster size >= 3, at least 3 GO terms with FDR q < 0.05,
top GO-term q < 0.05, and top prevalence delta >= 0.25.

## Perturbation Stability

             kind  fraction  successful_runs  median_ari  median_nmi  median_largest_cluster_fraction  median_n_clusters
feature_subsample       0.6                4    0.534627    0.750884                         0.294020               21.5
feature_subsample       0.8                6    0.661477    0.809298                         0.289867               15.5
   gene_subsample       0.8                8    0.577732    0.791049                         0.289419               16.5

## Most Stable Reference Clusters

 reference_cluster_id  reference_cluster_size  mean_best_jaccard  median_best_jaccard
                    2                      36           0.917306             0.956443
                    1                      25           0.895284             0.951190
                    0                      27           0.846166             0.869748
                   18                      44           0.825558             0.889990
                    3                       1           0.757481             1.000000
                   15                       2           0.754385             1.000000
                    4                      52           0.703402             0.752049
                   17                     130           0.606169             0.681757
                   16                      39           0.602491             0.575980
                   11                     109           0.544135             0.538205

Runtime seconds: `732.78`
