---
title: Julia allGO New Method-Version Tree Matrix 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - data/feature_matrices/feature_matrix_julia_allGO_new.tsv
  - applications/endotypes/run_allgo_method_version_tree_matrix.py
  - applications/endotypes/go_ic_tree_summary_plots.py
  - applications/endotypes/split_go_ic_results_by_method.py
  - raw/assets/benchmark-results/julia_allGO_new_method_version_tree_matrix_selected_plus_current_20260618/method_tree_matrix_summary.csv
  - raw/assets/benchmark-results/julia_allGO_new_method_version_tree_matrix_selected_plus_current_20260618/go_ic_plots/allgo_new_quality_aware_go_ic_tree_ranking.csv
  - raw/assets/benchmark-results/julia_allGO_new_method_version_tree_matrix_selected_plus_current_20260618/go_ic_by_method/allgo_new_quality_aware_go_ic_method_summary.csv
tags:
  - source
  - julia
  - allgo
  - legacy
  - current
  - diffusion
  - cosine-subspace
---

# Julia allGO New Method-Version Tree Matrix 2026-06-18

## Summary

The selected allGO-new crossed run separates method version from tree geometry.
It runs `legacy_c2ef` and `current` gates on selected TF-IDF cosine-subspace
trees, with both raw cosine-subspace geometry and adaptive diffusion inside the
cosine subspace. It also includes the current full-matrix adaptive diffusion
tree for comparison.

The attempted `legacy_c2ef__whole_adaptive_diffusion` run was interrupted
because the c2ef legacy sibling-null interpolation repeatedly calls NetworkX
shortest paths on the 602-leaf adaptive tree. The completed legacy adaptive
examples are therefore the adaptive-diffusion cosine-subspace rows.

## Key Points

- Completed families are `legacy_c2ef__raw_cosine_subspace`,
  `legacy_c2ef__adaptive_diffusion_cosine_subspace`,
  `current__raw_cosine_subspace`,
  `current__adaptive_diffusion_cosine_subspace`, and
  `current__whole_adaptive_diffusion`.
- The strongest completed row remains `current__whole_adaptive_diffusion`:
  GO-BIC active per gene `1475.898994`, `20` clusters, and `15/20` coherent
  clusters.
- The best legacy raw cosine-subspace row is
  `legacy_c2ef__raw_cosine_subspace__tfidf__adaptive_modes_02_05`:
  GO-BIC active per gene `1504.867882`, `19` clusters, and `11/19` coherent
  clusters.
- The best legacy adaptive-diffusion cosine-subspace row is
  `legacy_c2ef__adaptive_diffusion_cosine_subspace__tfidf__adaptive_modes_16_19`:
  GO-BIC active per gene `1639.592297`, `10` clusters, and `8/10` coherent
  clusters, but with largest-cluster fraction `0.601329`.
- Raw GO-IC alone still over-ranks overfragmented rows, for example
  `current__raw_cosine_subspace__tfidf__adaptive_modes_02_05` has raw GO-IC
  rank `1` but is quality-mixed because it has `256` clusters and high
  singleton-gene fraction.

## Evidence

- `method_tree_matrix_summary.csv` stores the completed method-version by
  tree-geometry assignments and cluster-size summaries.
- `go_ic_plots/allgo_new_quality_aware_go_ic_tree_ranking.csv` stores the
  quality-aware GO-IC ranking over the selected crossed examples.
- `go_ic_by_method/` stores method-separated PDFs and CSVs for the completed
  families.

## Links

- [[julia-allgo-new-go-ic-tree-summary-plots-20260618]]
- [[julia-allgo-new-c2ef-cosine-subspace-validation-20260618]]
