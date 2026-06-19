# allGO Method-Version x Tree-Geometry Matrix

Input: `data/feature_matrices/feature_matrix_julia_allGO_new.tsv`
Rows x columns: `602 x 6368`

Axes:
- `method_version`: `legacy_c2ef` or `current` gate/decomposition layer.
- `tree_geometry`: `whole_adaptive_diffusion`, `raw_cosine_subspace`, or `adaptive_diffusion_cosine_subspace`.

`raw_cosine_subspace` is the reader-facing name for the internal KAK/cosine eigenspace diagnostic.

Status counts:
{'ok': 1}

Compact ok rows:
method_version            tree_geometry weighting   block_name  n_clusters  largest_cluster_fraction  singleton_gene_fraction
       current whole_adaptive_diffusion     whole whole_matrix          20                  0.568106                      0.0
