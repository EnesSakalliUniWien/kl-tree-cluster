---
title: Paper Figure Map 2026-06-29
type: analysis
status: draft
updated: 2026-06-29
sources:
  - manuscript/sections/method/overview.tex
  - manuscript/sections/experiments/section.tex
  - manuscript/figures/pipeline_overview_single_page.tex
  - manuscript/figures/pipeline_overview_modules
  - reports/toytrees/handdrawn_3d_example
  - reports/toytrees/right_block_example
  - raw/assets/benchmark-results
  - raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/plot_manifest.csv
  - raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/plot_manifest.csv
  - raw/assets/benchmark-results/mnist_tbs_analysis_20260624/manifest.json
  - raw/assets/benchmark-results/mnist_tbs_analysis_20260624_plotly/manifest.json
  - raw/assets/benchmark-results/scrna_selected_adaptive_diffusion_nnls_manifest.json
  - raw/assets/benchmark-results/scrna_selected_adaptive_diffusion_nnls_matched_manifest.json
  - applications/scrna/plot_pipeline.py
  - applications/endotypes/plots/go_ic_tree_summary_plots.py
  - applications/endotypes/pipelines/run_current_adaptive_diffusion_subspace_tree_experiment.py
  - wiki/analyses/scrna-plot-pipeline-audit-20260624.md
  - wiki/sources/pancreas-scrna-clustering-benchmark-20260623.md
  - wiki/sources/goncalves-pancreas-progenitor-benchmark-prep-20260624.md
  - wiki/sources/scrna-space-decomposition-rerun-20260627.md
tags:
  - manuscript
  - figures
  - benchmark
  - plotting
---

# Paper Figure Map 2026-06-29

## Summary

The current manuscript has one visual figure wired into LaTeX: the TBS pipeline
overview in `manuscript/sections/method/overview.tex`, assembled from TikZ
modules under `manuscript/figures/`. The `experiments` section is still a
prospective evaluation plan, so the benchmark and report figures are generated
evidence pools rather than locked manuscript figures.

Across `manuscript/figures`, `reports/`, and
`raw/assets/benchmark-results`, there are `685` generated or hand-authored
visual artifacts with extensions `.tex`, `.png`, `.pdf`, or `.html`. The
practical map is therefore bundle-level: canonical manuscript figure,
explanatory toy figures, active scRNA review figures, GO/subspace figure
decks, MNIST exploratory figures, and older diagnostic/legacy figure pools.

## Details

### Status legend

- `Manuscript-wired`: included by the current LaTeX manuscript.
- `Candidate`: plausible paper or supplement material, but not currently
  included in `manuscript/main.tex`.
- `Evidence pool`: generated review or diagnostic figures supporting project
  interpretation.
- `Legacy/diagnostic`: historical or negative-control figures; keep as raw
  evidence but do not promote without a new locked result contract.

### Current manuscript figure

| Figure surface | Status | Files | Role |
| --- | --- | ---: | --- |
| `manuscript/figures/pipeline_overview_single_page.tex` and modules | Manuscript-wired | `12` `.tex` files | Figure in `overview.tex`: method pipeline from input matrix through hierarchy, node distributions, child-parent test, sibling test, and final cluster cut. |
| `manuscript/figures/essential_tikz_figures.tex` | Candidate/unused | Included in the `12` `.tex` count | Holds older standalone TikZ sketches for pipeline, edge/sibling principles, sibling calibration, and traversal. Not currently included by `main.tex`. |

### Generated figure bundles

| Bundle | Count | Status | Main use |
| --- | ---: | --- | --- |
| `reports/toytrees/handdrawn_3d_example/` | `4` PNG | Candidate | Simple 3D point, inner-node, edge-gate, and distance-matrix-step explanatory panels. |
| `reports/toytrees/right_block_example/` | `8` PNG | Candidate | Right-block toy tree, projected 3D points, distance/covariance/sibling-test callouts, and heatmap variants. |
| `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/` | `120` PNG/PDF | Evidence pool | Adult pancreas scRNA benchmark plots. The plot manifest marks `9` canonical, `51` derivative, `6` alias, and `54` orphaned previous-run plots. |
| `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/` | `76` PNG/PDF | Evidence pool | Goncalves fetal pancreas benchmark and progenitor review plots. The plot manifest covers `74` files; the two additional visual files are `goncalves_progenitor_coherence_score.*`. |
| `raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/` | `40` PNG | Evidence pool | Latest adult/Goncalves rerun plus invariant/equivariant PCA-space diagnostic plots. |
| `raw/assets/benchmark-results/scrna_branch_length_effect_audit_20260624/` | `3` PNG | Evidence pool | Branch-length assignment similarity, cluster effects, and branch-time sensitivity effects. |
| `raw/assets/benchmark-results/scrna_distributional_action_audit_20260624/` | `3` PNG | Evidence pool | Distributional-action versus branch length/statistic plus top internal action edges. |
| `raw/assets/benchmark-results/scrna_selected_adaptive_diffusion_nnls_*` root files | `14` PNG/PDF | Evidence pool | Selected adaptive-diffusion NNLS report pages, matched adult/Goncalves pages, UMAP/tree figures, and fit summaries. |
| `raw/assets/benchmark-results/julia_allGO_new_current_adaptive_diffusion_subspace_tree_20260618_163039/` | `204` PNG/PDF | Evidence pool | Current adaptive-diffusion GO subspace tree run: top-ranked subspaces, per-subspace embeddings/dendrograms, axis term plots, and combined axis-term PDF. |
| `raw/assets/benchmark-results/julia_allGO_new_go_ic_tree_plots_20260618/` | `117` PNG/PDF | Evidence pool | GO-IC quality-aware tree pages, rankings, scatter plots, contact sheets, and ordered PDFs. |
| `raw/assets/benchmark-results/julia_allGO_new_method_version_tree_matrix_selected_plus_current_20260618/` | `48` PNG/PDF | Evidence pool | Method-version tree matrix comparison plots split by current adaptive diffusion and raw cosine subspace methods. |
| `raw/assets/benchmark-results/julia_allGO_new_feature_matrix_quality_20260618/` | `1` PDF | Evidence pool | Feature-matrix quality report. |
| `raw/assets/benchmark-results/julia_allGO_new_c2ef_validate_cosine_subspace_split_legacy_20260618/` | `4` PNG | Legacy/diagnostic | Legacy cosine-subspace split validation plots. |
| `raw/assets/benchmark-results/julia_selected_family_20260614/` | `2` PNG | Legacy/diagnostic | Selected-family clustering diagnostic and multiscale UMAP overlay. |
| `raw/assets/benchmark-results/julia_tree_estimators_20260614/` | `14` PNG/HTML | Legacy/diagnostic | Older Julia tree-estimator UMAPs, alpha sensitivity heatmaps, and interactive HTML views. |
| `raw/assets/benchmark-results/conditional_topology_law_20260615/` | `1` PNG | Legacy/diagnostic | Conditional topology-law UMAP overlay. |
| `raw/assets/benchmark-results/mnist_tbs_analysis_20260624/` | `2` PNG/PDF | Evidence pool | Static MNIST TBS summary and one-plot-per-page PDF. |
| `raw/assets/benchmark-results/mnist_tbs_analysis_20260624_plotly/` | `12` HTML | Evidence pool | Interactive MNIST UMAP, 3D UMAP, inspector, alpha heatmap, composition, baseline, and tree views. |

### Canonical review surfaces

| Question | Preferred artifact |
| --- | --- |
| What figure is in the current manuscript? | `manuscript/sections/method/overview.tex` includes `figures/pipeline_overview_single_page`. |
| What adult pancreas plot should be reviewed first? | `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/all_methods_umap_clusters_all_colored.png` and `tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.png`. |
| What Goncalves plot should be reviewed first? | `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/all_methods_umap_clusters_all_colored.png`, `tbs_umap_cluster_radial_tree_combo_scaled_umap_ggtree.png`, and `goncalves_tbs_progenitor_umap_tree_pages_ggtree.pdf`. |
| What latest scRNA rerun plots should be checked? | `raw/assets/benchmark-results/scrna_space_decomposition_rerun_20260627/space_decomposition/*/cell_space_decomposition.png` and `method_space_summary.png`, plus the benchmark UMAP/metric plots in the two rerun benchmark folders. |
| What GO/subspace plots should be checked first? | `current_adaptive_diffusion_subspace_tree_top_ranked_subspaces.png`, `current_adaptive_diffusion_subspace_tree_axis_terms_by_subspace.pdf`, and the GO-IC ordered tree-page PDFs. |
| What toy explanatory plots are available? | The handdrawn 3D bundle and right-block bundle under `reports/toytrees/`. |
| What MNIST figures are available? | `mnist_tbs_analysis_summary.png`, `mnist_tbs_analysis_one_plot_per_page.pdf`, and the Plotly HTML index/interactive pages. |

### Generator map

- `applications/scrna/plot_pipeline.py` is the coherent adult/Goncalves scRNA
  plot manifest and orchestration layer.
- `applications/scrna/pancreas_benchmark.py` and
  `applications/scrna/goncalves_benchmark.py` produce the benchmark
  metric, UMAP, dendrogram, branch-length, and assignment surfaces.
- `applications/scrna/plots/pancreas_all_method_umap_clusters.py`,
  `applications/scrna/plots/pancreas_readable_umap_clusters.py`,
  `applications/scrna/plots/pancreas_radial_trees_ggtree.R`,
  `applications/scrna/plots/pancreas_cluster_radial_trees_ggtree.R`, and
  `applications/scrna/plots/pancreas_umap_tree_combo_ggtree.R` produce the main scRNA
  review figures.
- `applications/scrna/analysis/analyze_goncalves_progenitors.py` and
  `applications/scrna/plots/goncalves_progenitor_trees_ggtree.R` produce the Goncalves
  progenitor-specific figures.
- `applications/scrna/analyze_space_decomposition.py` produces the space
  decomposition plots in the 2026-06-27 rerun.
- `applications/scrna/analysis/audit_branch_length_effects.py`,
  `applications/scrna/analysis/audit_distributional_action.py`,
  `applications/scrna/plots/selected_nnls_report.py`,
  `applications/scrna/plots/selected_nnls_fit_summary.py`, and
  `applications/scrna/plots/selected_nnls_matched_report.py` produce the scRNA audit
  and selected-NNLS report figures.
- `applications/endotypes/pipelines/run_current_adaptive_diffusion_subspace_tree_experiment.py`
  and `applications/endotypes/plots/go_ic_tree_summary_plots.py` produce the GO/subspace
  figure decks.
- `applications/mnist/plot_report.py` and
  `applications/mnist/plot_interactive.py` produce the static and
  interactive MNIST figure bundles.
- `manuscript/tools/figures/plot_handdrawn_3d_edge_gate.py`,
  `manuscript/tools/figures/plot_handdrawn_distance_matrix_steps.py`,
  `manuscript/tools/figures/plot_right_block_toytree.py`,
  `manuscript/tools/figures/plot_right_block_3d_points.py`,
  `manuscript/tools/figures/add_right_block_sibling_test.py`,
  `manuscript/tools/figures/add_right_block_covariance_callout.py`, and
  `manuscript/tools/figures/add_right_block_covariance_heatmaps.py` produce the toy-tree
  explanatory report figures.

## Evidence

- `manuscript/sections/method/overview.tex` includes the only current
  manuscript figure environment and loads `figures/pipeline_overview_single_page`.
- `manuscript/sections/experiments/section.tex` states that the results
  section is still prospective and requires locked scripts, manifests, seeds,
  and output files before reported figures enter the paper.
- A local inventory over `manuscript/figures`, `reports`, and
  `raw/assets/benchmark-results` found `685` visual artifacts: `12` manuscript
  `.tex` figure files, `12` toy-tree PNGs, and `661` result/report artifacts
  under `raw/assets/benchmark-results`.
- `raw/assets/benchmark-results/pancreas_scrna_cluster_benchmark_20260623/plot_manifest.csv`
  maps all `120` adult pancreas PNG/PDF files to role, stage, generator,
  canonical status, alias target, and intended question.
- `raw/assets/benchmark-results/goncalves_fetal_pancreas_progenitor_benchmark_20260624/plot_manifest.csv`
  maps `74` Goncalves PNG/PDF files; the unmanifested visual pair is
  `goncalves_progenitor_coherence_score.{png,pdf}`.
- `raw/assets/benchmark-results/mnist_tbs_analysis_20260624/manifest.json`,
  `raw/assets/benchmark-results/mnist_tbs_analysis_20260624_plotly/manifest.json`,
  `raw/assets/benchmark-results/scrna_selected_adaptive_diffusion_nnls_manifest.json`,
  and
  `raw/assets/benchmark-results/scrna_selected_adaptive_diffusion_nnls_matched_manifest.json`
  record figure/report artifacts and provenance for those bundles.

## Links

- [[scrna-plot-pipeline-audit-20260624]]
- [[pancreas-scrna-clustering-benchmark-20260623]]
- [[goncalves-pancreas-progenitor-benchmark-prep-20260624]]
- [[goncalves-tbs-progenitor-analysis-20260624]]
- [[scrna-space-decomposition-rerun-20260627]]
- [[current-adaptive-diffusion-subspace-tree-pipeline]]
- [[julia-allgo-new-current-adaptive-diffusion-subspace-tree-20260618]]
- [[julia-allgo-new-go-ic-tree-summary-plots-20260618]]
- [[julia-tree-estimator-run-20260614]]

## Open Questions

- Which generated evidence-pool figures should become locked manuscript or
  supplement figures once the experiments section is replaced?
- Should the toy-tree explanatory figures be promoted into the method section,
  or should the current TikZ pipeline overview remain the only method visual?
- Should legacy/diagnostic figure bundles be moved out of top-level benchmark
  result directories, or should readers continue filtering them through
  manifests and this map?
