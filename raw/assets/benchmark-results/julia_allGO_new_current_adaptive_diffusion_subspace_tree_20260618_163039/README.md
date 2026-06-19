# Current Adaptive Diffusion Subspace Tree Experiment

Input: `data/feature_matrices/feature_matrix_julia_allGO_new.tsv`
Method version: `current`
Tree geometry: `adaptive_diffusion_cosine_subspace`
Experiment directory: `raw/assets/benchmark-results/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163039`

Directory layout:
- `rankings/`: experiment-level ranking, spectrum, block metadata, and long quality tables.
- `plots/`: experiment-level summary plots.
- `subspaces/<weighting>/<block_name>/`: one folder per subspace with assignments, tree, quality CSVs, coordinates, and axis term-loading plots.

Ranking:
- `display_rank` sorts by quality tier first, then lower GO-BIC active.
- `raw_go_ic_rank` preserves the raw GO-IC order for audit.
- `go_bic_active_per_gene` is lower-is-better only within comparable quality tiers.

Axis term loadings:
- `axis_term_loadings_all.csv` stores every GO term loading for every cosine mode in the subspace.
- `axis_top_terms.csv` stores the top positive, negative, and absolute GO-term loadings per axis.
- Positive and negative signs are orientation-dependent; the absolute loading is the stable strength score.

Ranking CSV: `raw/assets/benchmark-results/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163039/rankings/current_adaptive_diffusion_subspace_tree_ranking.csv`

Top ranked subspace:
- `tfidf / adaptive_modes_02_05`
- clusters: `40`
- quality tier: `quality_plausible`
- GO-BIC active/gene: `1687.966840`
- coherent clusters: `21/40`
