# allGO Method-Version x Tree-Geometry Matrix

Input: `data/feature_matrices/feature_matrix_julia_allGO_new.tsv`
Rows x columns: `602 x 6368`

Axes:
- `method_version`: `legacy_c2ef` or `current` gate/decomposition layer.
- `tree_geometry`: `whole_adaptive_diffusion`, `raw_cosine_subspace`, or `adaptive_diffusion_cosine_subspace`.

`raw_cosine_subspace` is the reader-facing name for the internal KAK/cosine eigenspace diagnostic.

Status counts:
{'ok': 12}

Compact ok rows:
method_version                      tree_geometry weighting           block_name  n_clusters  largest_cluster_fraction  singleton_gene_fraction
   legacy_c2ef                raw_cosine_subspace     tfidf adaptive_modes_02_05          19                  0.215947                 0.001661
       current                raw_cosine_subspace     tfidf adaptive_modes_02_05         256                  0.058140                 0.294020
   legacy_c2ef adaptive_diffusion_cosine_subspace     tfidf adaptive_modes_02_05          32                  0.232558                 0.001661
       current adaptive_diffusion_cosine_subspace     tfidf adaptive_modes_02_05          40                  0.232558                 0.008306
   legacy_c2ef                raw_cosine_subspace     tfidf adaptive_modes_16_19          27                  0.136213                 0.001661
       current                raw_cosine_subspace     tfidf adaptive_modes_16_19          76                  0.088040                 0.024917
   legacy_c2ef adaptive_diffusion_cosine_subspace     tfidf adaptive_modes_16_19          10                  0.601329                 0.000000
       current adaptive_diffusion_cosine_subspace     tfidf adaptive_modes_16_19          69                  0.112957                 0.028239
   legacy_c2ef                raw_cosine_subspace     tfidf adaptive_modes_53_80           1                  1.000000                 0.000000
       current                raw_cosine_subspace     tfidf adaptive_modes_53_80         133                  0.019934                 0.006645
   legacy_c2ef adaptive_diffusion_cosine_subspace     tfidf adaptive_modes_53_80          67                  0.061462                 0.019934
       current adaptive_diffusion_cosine_subspace     tfidf adaptive_modes_53_80          22                  0.372093                 0.000000
