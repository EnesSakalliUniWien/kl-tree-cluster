---
title: allGO New Interactome Current Adaptive Diffusion Subspace Tree 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - applications/endotypes/run_current_adaptive_diffusion_subspace_tree_experiment.py
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/README.md
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/ARTIFACT_INDEX.md
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/experiment_config.json
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/rankings/current_adaptive_diffusion_subspace_tree_ranking.csv
  - results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237/allgo_new_interactome_quality_aware_go_ic_by_method/current__adaptive_diffusion_cosine_subspace/allgo_new_interactome_quality_aware_go_ic_current__adaptive_diffusion_cosine_subspace_tree_pages.pdf
tags:
  - source
  - allgo
  - interactome
  - current
  - adaptive-diffusion
---

# allGO New Interactome Current Adaptive Diffusion Subspace Tree 2026-06-18

## Summary

The current adaptive-diffusion cosine-subspace tree pipeline was rerun on
`/Users/berksakalli/Downloads/feature_matrix_allGO_new_interactome.tsv`. The
experiment root is
`results/analyses/allgo_new_interactome_current_adaptive_diffusion_subspace_tree_20260618_175237`.
The matrix has `339` genes by `5873` binary GO terms, with no zero rows or zero
columns.

## Key Points

- The run created the full connected artifact structure described in
  [[current-adaptive-diffusion-subspace-tree-pipeline]].
- The adaptive spectral split produced `12` subspaces: `6` binary and `6`
  TF-IDF.
- Current TBS completed for `7` subspaces and failed closed for `5` subspaces.
  Failed subspaces still have coordinates, diffusion metadata, tree artifacts,
  axis-term outputs, and explicit `*_failure_status.csv` files.
- The top specificity-aware subspace is `tfidf / adaptive_modes_06_11`: `13`
  clusters, `12/13` coherent clusters, specificity score `0.753376`, and
  GO-BIC active per gene `1796.282946`.
- The next completed subspaces are `binary / adaptive_modes_13_22`,
  `tfidf / adaptive_modes_02_05`, and `binary / adaptive_modes_02_05`.
- The method-separated workflow PDF has `14` pages: two pages for each of the
  `7` completed subspaces. The axis-term PDF has `85` pages, and the combined
  axis-term PDF has `7` pages.
- The failed-gate subspaces are `binary / adaptive_common_mode_01`,
  `binary / adaptive_modes_42_80`, `tfidf / adaptive_common_mode_01`,
  `tfidf / adaptive_modes_12_23`, and `tfidf / adaptive_modes_24_45`.

## Evidence

- `rankings/current_adaptive_diffusion_subspace_tree_ranking.csv` stores the
  specificity-aware display rank, old display rank, raw GO-IC rank, GO-BIC
  fields, specificity fields, cluster-quality fields, and artifact paths.
- `ARTIFACT_INDEX.md`, `artifact_index.csv`, `subspace_plot_index.csv`, and
  `connected_results_manifest.json` connect the ranked rows to subspace
  directories, CSVs, trees, plots, and PDFs.
- `allgo_new_interactome_quality_aware_go_ic_by_method/current__adaptive_diffusion_cosine_subspace/`
  stores the method-separated PDF and per-page PNGs.

## Links

- [[current-adaptive-diffusion-subspace-tree-pipeline]]
- [[julia-allgo-new-current-adaptive-diffusion-subspace-tree-20260618]]
