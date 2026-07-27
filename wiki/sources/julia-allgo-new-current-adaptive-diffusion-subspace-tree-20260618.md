---
title: Julia allGO New Current Adaptive Diffusion Subspace Tree 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - data/feature_matrices/feature_matrix_julia_allGO_new.tsv
  - applications/endotypes/run_current_adaptive_diffusion_subspace_tree_experiment.py
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/README.md
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/experiment_config.json
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/rankings/current_adaptive_diffusion_subspace_tree_ranking.csv
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/rankings/current_adaptive_diffusion_subspace_tree_subspace_blocks.csv
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/rankings/current_adaptive_diffusion_subspace_tree_spectrum.csv
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/ARTIFACT_INDEX.md
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/connected_results_manifest.json
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/current_adaptive_diffusion_subspace_tree_axis_terms_by_subspace.pdf
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/current_adaptive_diffusion_subspace_tree_axis_terms_combined_by_subspace.pdf
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/current_adaptive_diffusion_subspace_tree_finished_subspace_cluster_report.pdf
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/allgo_new_quality_aware_go_ic_by_method/current__adaptive_diffusion_cosine_subspace/allgo_new_quality_aware_go_ic_current__adaptive_diffusion_cosine_subspace_tree_pages.pdf
  - results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811/allgo_new_quality_aware_go_ic_plots/allgo_new_quality_aware_go_ic_all_tree_pages.pdf
tags:
  - source
  - julia
  - allgo
  - current
  - adaptive-diffusion
  - cosine-subspace
  - go-ic
---

# Julia allGO New Current Adaptive Diffusion Subspace Tree 2026-06-18

## Summary

The current-method allGO-new run evaluates adaptive diffusion trees inside
cosine eigenspace subspaces. The experiment root is
`results/analyses/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163811`.
It uses both `binary` and `tfidf` weightings, writes one directory per subspace
under `subspaces/<weighting>/<block_name>/`, and adds per-axis plus combined
GO-term loading tables and plots for every cosine mode in each subspace.

## Key Points

- The experiment scored `15` subspaces: `14` completed with current TBS cluster
  assignments and `1` failed closed at the current gate layer.
- The preferred display rank is now specificity-aware: quality tier first, then
  cluster specificity score, specific-cluster fraction, weighted specificity
  delta, and GO-BIC active per gene. The top specificity-aware subspace is
  `binary / adaptive_modes_12_21`: `38` clusters, `27/38` specific/coherent
  clusters, specificity score `0.646011`, and GO-BIC active per gene
  `1868.260854`.
- The earlier quality-tiered GO-IC display rank is preserved as
  `old_display_rank`. Under that older order, `tfidf / adaptive_modes_02_05`
  was first; under the specificity-aware order it is rank `4`.
- Raw GO-IC alone still over-ranks degenerate trees. For example,
  `tfidf / adaptive_modes_32_52` has raw GO-IC rank `1` but `592` clusters and
  singleton-gene fraction `0.968439`, so it is demoted to the degenerate tier.
- The failed subspace is `binary / adaptive_modes_02_06`. The adaptive
  diffusion tree and axis-term outputs were written, but current TBS did not
  produce cluster assignments because the sibling inflation model had no valid
  strict-null or stopped-edge empirical-null calibration records.
- Per-axis term-loading outputs are stored in each subspace directory as
  `*_axis_term_loadings_all.csv`, `*_axis_top_terms.csv`, and
  `*_axis_terms__mode_XX.png`. Combined per-subspace heatmaps are stored as
  `*_axis_terms_combined.png`. The PDF
  `current_adaptive_diffusion_subspace_tree_axis_terms_by_subspace.pdf`
  contains the ranked subspace axis-term pages, while
  `current_adaptive_diffusion_subspace_tree_axis_terms_combined_by_subspace.pdf`
  contains one combined term-loading heatmap per subspace.
- Each eigenband also has a term-annotated subspace-coordinate embedding:
  `*_subspace_embedding_terms_only.png` for all `15` bands, plus
  `*_subspace_embedding_clusters_annotated_terms.png` and
  `*_adaptive_diffusion_embedding_clusters_annotated_terms.png` for the `14`
  bands with current TBS cluster assignments.
- The workflow-style PDFs were recreated under
  `allgo_new_quality_aware_go_ic_by_method/current__adaptive_diffusion_cosine_subspace/`
  and `allgo_new_quality_aware_go_ic_plots/`. The method PDF has `15` pages,
  one page per eigenband, with tree/subtree cluster strip, subspace embedding,
  adaptive-diffusion embedding where available, and combined GO-term loadings.
  It was then regenerated as a readable `30`-page vector-text workflow PDF with
  two pages per eigenband: a tree/subtree-strip page and a subspace/term page.

## Interpretation

The best current adaptive-diffusion subspace tree is again the TF-IDF
components `02-05` subspace, but the current adaptive tree creates `40`
clusters rather than the legacy c2ef `19`-cluster split. The ranking remains
quality-tiered because GO-IC rewards rare-term overfragmentation in high-mode
TF-IDF subspaces. The axis-term loadings make the subspace interpretation
auditable: mode `02` in the top TF-IDF block is dominated by transcription and
DNA-binding terms, while mode `04` separates extracellular/endoplasmic lumen
and granule terms from TNF/NF-kappaB/apoptotic signaling terms.

## Evidence

- `rankings/current_adaptive_diffusion_subspace_tree_ranking.csv` stores
  display rank, raw GO-IC rank, quality tier, GO-BIC fields, cluster counts,
  coherent-cluster fields, and paths back to subspace-specific artifacts.
- `subspaces/<weighting>/<block_name>/` stores the subspace coordinates,
  diffusion metadata, linkage matrix, cluster assignments when current TBS
  completed, failure status when it did not, cluster quality tables, and
  per-axis GO-term loading outputs.
- `ARTIFACT_INDEX.md`, `artifact_index.csv`, `subspace_plot_index.csv`, and
  `connected_results_manifest.json` connect each ranked row to its subspace
  directory, CSVs, trees, plain and term-annotated embedding plots, combined
  term plot, and PDFs.
- `plots/current_adaptive_diffusion_subspace_tree_top_ranked_subspaces.png`
  summarizes the ranked subspaces.

## Links

- [[julia-allgo-new-feature-matrix-quality-20260618]]
- [[julia-allgo-new-method-version-tree-matrix-20260618]]
- [[julia-allgo-new-go-ic-tree-summary-plots-20260618]]
