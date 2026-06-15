# c2ef Cosine Subspace Method Notes 2026-06-15

Source branch: `codex/c2ef-cosine-subspace-method`

Source commit: `c2ef9a69e0888168950bdee4a41ae8ab9996e32f`

The c2ef branch added two standalone utilities:

- `scripts/cosine_subspace_tree_sweep.py`
- `scripts/validate_cosine_subspace_split.py`

The useful method elements were:

- fixed sample-sample cosine eigen-bands:
  `common_mode_01`, `variation_02_05`, `variation_06_15`,
  `variation_16_35`, `variation_36_80`, `broad_variation_02_35`,
  `broad_variation_02_80`, and `all_modes_01_80`;
- binary and TF-IDF weighted cosine operators;
- one tree/decomposition run per fixed band;
- cluster-size, top-term, spectrum, and plot outputs;
- biological coherence checks using feature enrichment, within-cluster TF-IDF
  cosine, perturbation ARI, and cluster-level best Jaccard.

The current branch should not restore those scripts as production methods. The
portable part is a diagnostic comparator: keep the fixed bands and coherence
checks, but run them through the current decomposition stack and mark the
outputs diagnostic-only.
