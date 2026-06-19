# allGO Selected Method-Version x Tree-Geometry Matrix

Rows combine selected TF-IDF cosine-subspace examples with the current full-matrix adaptive diffusion row.

Included method families:
- `current__adaptive_diffusion_cosine_subspace`
- `current__raw_cosine_subspace`
- `current__whole_adaptive_diffusion`
- `legacy_c2ef__adaptive_diffusion_cosine_subspace`
- `legacy_c2ef__raw_cosine_subspace`

Note: `legacy_c2ef__whole_adaptive_diffusion` was attempted separately and interrupted because the c2ef legacy sibling-null interpolation repeatedly calls NetworkX shortest paths on the 602-leaf adaptive tree. The legacy adaptive example included here is `legacy_c2ef__adaptive_diffusion_cosine_subspace`, which uses adaptive diffusion inside the cosine-decomposed subspace.

Source summaries:
- `raw/assets/benchmark-results/julia_allGO_new_method_version_tree_matrix_tfidf_selected_20260618/method_tree_matrix_summary.csv`
- `raw/assets/benchmark-results/julia_allGO_new_method_version_tree_matrix_current_whole_20260618/method_tree_matrix_summary.csv`
