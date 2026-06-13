---
title: Adaptive Cosine KAK Benchmark Probe 2026-06-05
type: source
status: reviewed
updated: 2026-06-12
sources:
  - benchmarks/diagnostics/spectral/adaptive_cosine_kak_benchmark_probe.py
  - benchmarks/diagnostics/spectral/adaptive_cosine_kak_matrix_probe.py
  - benchmarks/diagnostics/spectral/adaptive_cosine_kak_diffusion_matrix_probe.py
  - benchmarks/diagnostics/spectral/kak_lens_alpha_sweep.py
  - benchmarks/diagnostics/spectral/kak_lens_feature_axis_clustering.py
  - benchmarks/diagnostics/spectral/kak_feature_subspace_clustering.py
  - benchmarks/diagnostics/spectral/tree_strategy_semantic_panel.py
  - benchmarks/diagnostics/spectral/covariance_axis_stability.py
  - benchmarks/cloud/aws_kak_lens_linkage_alpha_sweep.py
  - benchmarks/cloud/aws_tree_strategy_semantic_panel.py
  - kl_clustering_analysis/plot/cluster_tree_visualization.py
  - tests/validation/82_test_aws_kak_lens_linkage_alpha_sweep.py
  - tests/validation/83_test_covariance_axis_stability.py
  - tests/validation/84_test_kak_lens_feature_axis_clustering.py
  - tests/validation/85_test_tree_strategy_semantic_panel.py
  - tests/validation/86_test_kak_feature_subspace_clustering.py
  - tests/visualization/71_test_cluster_tree_visualization.py
  - scripts/kak_signal_adaptive_umap_tree_page.py
  - scripts/analysis/run_feature_matrix_with_umap.py
  - benchmarks/results/diagnostics/adaptive_cosine_kak_probe_method_proof_cal_default_20260605/kak_benchmark_probe_rows.csv
  - benchmarks/results/diagnostics/adaptive_cosine_kak_probe_method_proof_cal_enforced_20260605/kak_benchmark_probe_rows.csv
  - benchmarks/results/diagnostics/adaptive_cosine_kak_probe_full_gate_bundle_20260605/kak_benchmark_probe_rows.csv
  - benchmarks/results/diagnostics/adaptive_cosine_kak_probe_full_cal_enforced_20260605/kak_benchmark_probe_rows.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_spectral_blocks_gate_bundle_20260605/matrix_kak_probe_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_spectral_blocks_support_enforced_20260605/matrix_kak_probe_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_20260606/kak_signal_adaptive_block_geometry_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_20260606/kak_signal_adaptive_internal_tree_geometry.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_20260606/kak_signal_adaptive_umap_tree_page.png
  - data/reference/adg6375_File_S7_endotypes_julia.txt
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_nmi_ordered_20260610/kak_signal_adaptive_reference_labels.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_nmi_ordered_20260610/kak_signal_adaptive_reference_nmi_metrics.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_nmi_ordered_20260610/kak_signal_adaptive_reference_nmi_ranking.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_nmi_ordered_20260610/kak_signal_adaptive_umap_tree_page.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/09_full_data_method_reference_comparison_20260610/all_method_and_kak_reference_comparison.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/09_full_data_method_reference_comparison_20260610/selected_method_reference_comparison.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/09_full_data_method_reference_comparison_20260610/method_vs_kak_reference_nmi_ari_scatter.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/09_full_data_method_reference_comparison_20260610/selected_method_reference_comparison_bars.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/10_full_data_kak_separated_diffusion_20260610/matrix_kak_diffusion_probe_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/10_full_data_kak_separated_diffusion_20260610/kak_diffusion_reference_nmi_metrics.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/10_full_data_kak_separated_diffusion_20260610/separated_diffusion_vs_existing_reference_comparison.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/10_full_data_kak_separated_diffusion_20260610/separated_diffusion_vs_existing_ari_nmi_scatter.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/11_full_data_kak_separated_adaptive_diffusion_20260610/matrix_kak_diffusion_probe_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/11_full_data_kak_separated_adaptive_diffusion_20260610/adaptive_vs_fixed_separated_and_existing_reference_comparison.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/11_full_data_kak_separated_adaptive_diffusion_20260610/adaptive_vs_fixed_separated_and_existing_ari_nmi_scatter.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/12_context_quality_tree_comparison_20260610/context_method_quality_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/12_context_quality_tree_comparison_20260610/context_quality_report.md
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/12_context_quality_tree_comparison_20260610/context_quality_scatter.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/13_subspace_lens_vs_main_adaptive_contexts_20260610/subspace_lens_vs_main_context_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/13_subspace_lens_vs_main_adaptive_contexts_20260610/subspace_lens_vs_main_context_report.md
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/13_subspace_lens_vs_main_adaptive_contexts_20260610/subspace_lens_vs_main_context_scatter.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/14_main_context_subspace_lens_plot_list_20260610/plot_list.md
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/14_main_context_subspace_lens_plot_list_20260610/plot_list.html
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/15_kak_lens_alpha_sweep_20260610/kak_lens_alpha_sweep_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/15_kak_lens_alpha_sweep_20260610/kak_lens_alpha_sweep_report.md
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/15_kak_lens_alpha_sweep_20260610/alpha_sweep_fragmentation_vs_purity.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/15_kak_lens_alpha_sweep_20260610/alpha_sweep_top_lens_scores.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/16_kak_lens_linkage_smoke_20260611/kak_lens_alpha_sweep_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/17_aws_kak_lens_linkage_alpha_sweep_smoke_20260611/merged/kak_lens_linkage_alpha_sweep_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/17_aws_kak_lens_linkage_alpha_sweep_smoke_20260611/merged/aws_kak_lens_linkage_alpha_sweep_manifest.json
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/18_kak_lens_linkage_complete_smoke_20260611/kak_lens_alpha_sweep_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/19_covariance_axis_stability_20260611/covariance_axis_stability_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/19_covariance_axis_stability_20260611/covariance_axis_stability_replicates.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/19_covariance_axis_stability_20260611/covariance_axis_stability_histograms.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/20_kak_lens_feature_axis_clustering_20260612/kak_lens_feature_axis_clustering_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/20_kak_lens_feature_axis_clustering_20260612/raw_kak__binary__adaptive_modes_10_15/lens_feature_axis_scores.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/20_kak_lens_feature_axis_clustering_20260612/raw_kak__tfidf__adaptive_modes_14_30/lens_feature_axis_scores.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/21_kak_lens_feature_axis_clustering_debug_20260612/kak_lens_feature_axis_clustering_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/21_kak_lens_feature_axis_clustering_debug_20260612/kak_lens_feature_axis_clustering_report.md
  - benchmarks/results/diagnostics/tree_strategy_semantic_panel_smoke_20260612/tree_strategy_semantic_panel.csv
  - benchmarks/results/diagnostics/tree_strategy_semantic_panel_smoke_20260612/tree_strategy_semantic_panel_report.md
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/22_aws_kak_lens_linkage_alpha_sweep_full_20260612/merged/kak_lens_linkage_alpha_sweep_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/22_aws_kak_lens_linkage_alpha_sweep_full_20260612/merged/aws_kak_lens_linkage_alpha_sweep_manifest.json
  - benchmarks/results/diagnostics/tree_strategy_semantic_panel_aws_full_kak_20260612/tree_strategy_semantic_panel.csv
  - benchmarks/results/diagnostics/tree_strategy_semantic_panel_aws_full_kak_20260612/tree_strategy_semantic_panel_report.md
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/23_kak_binary_feature_subspace_clustering_20260612/kak_feature_subspace_clustering_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/25_julia_classical_umap_clustering_20260612/julia_classical_umap_clustering_summary.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/25_julia_classical_umap_clustering_20260612/julia_classical_umap_clustering_panel.png
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/25_julia_classical_umap_clustering_20260612/julia_reference_endotypes_on_umap.png
  - benchmarks/results/diagnostics/adaptive_cosine_kak_probe_full_gate_bundle_20260605/kak_probe_full_block_logic_scatter.png
tags:
  - source
  - benchmarks
  - spectral
  - kak
  - diagnostics
---

# Adaptive Cosine KAK Benchmark Probe 2026-06-05

## Summary

A current-compatible adaptive cosine/KAK diagnostic was added and run against
the method-proof and full benchmark suites. The runner mirrors the historical
spectral-block idea by selecting tree topology in cosine eigenspaces, but it
uses the current production gate-bundle path: populate feature-space-aware node
divergences, run the gate annotation pipeline, then decompose from the
`GateAnnotationBundle`. A matrix-level counterpart reran the historical
combined Julia GO feature matrix and compared default versus support-threshold
enforced calibration. The historical `kak_signal_adaptive_umap_tree_page.py`
was restored as a current-compatible diagnostic page generator that reads the
matrix-probe outputs, recomputes cosine/KAK block coordinates, and writes
sample-level plus internal-tree geometry tables.
The restored page generator now also supports Julia reference endotypes as an
external diagnostic ordering signal: matched genes are labeled, block
assignments are scored by NMI/ARI, and displayed block pages can be sorted by
descending reference NMI.
A follow-up comparison reran the historical full-matrix methods on the same
Julia matrix and rescored KAK plus non-KAK assignments against the same saved
reference-label subset.
The KAK/cosine lens alpha runner now also supports comparing average, complete,
and Ward-Euclidean lens trees, and an AWS Batch wrapper shards the larger
`lens x linkage x alpha` grid.

## Key Points

- The method-proof run produced `42` spectral-block rows: `37` ok rows and `5`
  fail-closed rows. The failures were the dense continuous covariance memory
  contract (`4` rows) and lack of strict-null/stopped-edge empirical-null
  sibling calibration support (`1` row).
- Enforcing internal support thresholds on the method-proof run reduced ok rows
  from `37` to `28`; best-case mean ARI fell from `0.669402` to `0.629684`.
  The additional fail-closed rows were `undefined_sparse_context` calibration
  decisions.
- The full benchmark probe produced `418` rows over `120` cases: `369` ok
  rows, `46` failed-gate rows, and `3` failed-spectrum rows.
- Enforcing internal support thresholds on the full benchmark kept the same
  `418` spectral rows but changed the split between row states to `248` ok,
  `167` failed-gate, and `3` failed-spectrum. Cases with at least one ok block
  fell from `110` to `96`, and mean best-case ARI fell from `0.857673` to
  `0.674170`.
- The best block was an adaptive non-common spectral block in `100` of `110`
  cases with at least one ok row. Common-mode blocks were best in only `10`
  cases and had much lower all-row mean ARI (`0.158977`) than adaptive decay
  blocks (`0.421132`).
- The most frequent best intervals were `2-5` (`31` cases, mean ARI
  `0.860424`), `2-6` (`29` cases, mean ARI `0.960209`), `1-4` (`16` cases,
  mean ARI `0.962170`), and `2-8` (`12` cases, mean ARI `0.992832`).
- The method-proof heatmap shows the diagnostic logic clearly: high-energy
  common-mode axes usually collapse to one cluster, while early non-common
  blocks recover the binary barycentric, selected-support, deep-traversal, and
  some phylogenetic structure.
- High block energy is not a reliable block-selection rule. The full-suite
  scatter plot shows both high-energy common-mode failures and lower-energy
  early-block successes.
- The full-data combined Julia GO matrix (`703 x 14766`) reproduced the
  historical eigenspectrum segmentation but did not produce admissible
  production clusters. With support enforcement off, `11` of `15` binary/TF-IDF
  blocks were ok but severely over-split (`101` to `650` clusters for `703`
  genes). With support thresholds enforced, all `15` blocks failed closed:
  early/default-ok blocks became `undefined_sparse_context`, while tail blocks
  lacked strict-null or stopped-edge empirical-null support.
- The restored visualization/internal-geometry pass wrote `11` ok block pages
  and `7722` internal merge rows for the default full-data matrix run. It is
  diagnostic-only: it does not run clustering and does not promote selected
  KAK/cosine bases to production-calibrated tests.
- The geometric failure mode is visible in the restored pages. Non-common
  binary/TF-IDF KAK blocks have median angles to the leading block axis around
  `63.9` to `81.4` degrees and median independent-radius fractions around
  `0.897` to `0.989`, while producing hundreds of tiny clusters. Common-mode
  blocks have the expected one-dimensional radius law but also over-split.
- Internal merge diagnostics show median sibling separation to parent-radius
  ratios around `1.72` to `1.96` for non-common blocks. Many median parent
  dominant-cluster fractions are `1.0`, indicating that the current tree/gate
  path often turns local pure fragments into many assignments instead of
  validating a stable selected-tree decomposition.
- Future KAK/cosine tree visualizations should order displayed trees or block
  pages by cluster-level NMI when reference labels are available, so the most
  informative decompositions are inspected first. This is a diagnostic
  presentation rule only; it does not change tree construction, gate testing,
  traversal, or calibration.
- The NMI-ordered Julia rerun matched `262` of `703` matrix genes to `20`
  observed Julia endotypes. The highest-NMI block was
  `binary__adaptive_modes_16_19` with NMI `0.632591`, ARI `0.002297`, and
  `650` clusters. The next block,
  `binary__adaptive_modes_20_30`, had NMI `0.628401`, ARI `0.005370`, and
  `603` clusters.
- The Julia reference scores confirm that NMI ordering is useful for visual
  inspection but not for production selection. The top-NMI blocks still split
  the `703` genes into hundreds of fragments, so they expose geometry that
  overlaps with reference labels without validating a stable selected-tree
  clustering rule.
- The same-label comparison against historical full-matrix methods shows why
  NMI alone is misleading. The current KL gate has NMI `0.633958`, similar to
  the top KAK block, but it creates `670` clusters and has ARI `0.001812`.
  The top-NMI KAK block creates `650` clusters and has ARI `0.002297`.
- By ARI, fixed diffusion with gates is the best recreated comparator on the
  Julia labels: `54` clusters, NMI `0.441072`, and ARI `0.062392`. The
  paper-faithful cosine complete `K=20` baseline is close: NMI `0.336014`,
  ARI `0.058671`, and `20` clusters. These lower-NMI methods are better
  clusterings because they avoid the extreme singleton-heavy fragmentation.
- The best KAK block by ARI is `binary__adaptive_modes_10_15`, with `101`
  clusters, NMI `0.497740`, and ARI `0.045578`; it is closer to the diffusion
  comparators but still below fixed diffusion and paper cosine complete by
  adjusted overlap.
- A separated-space diffusion follow-up ran a k-NN Gaussian diffusion operator
  inside each adaptive cosine/KAK block, then ran the normal KL-TE gates on the
  original Julia GO feature matrix. It produced `12` ok rows and `3`
  calibration-support fail-closed rows. The best row was
  `binary__adaptive_modes_02_05` with `153` clusters, singleton fraction
  `0.568627`, NMI `0.498510`, and ARI `0.085175`.
- Separated-space diffusion improved the best Julia-reference ARI versus the
  recreated fixed-diffusion full-matrix comparator (`0.085175` versus
  `0.062392`) and the best raw KAK block (`0.045578`), but it still
  fragments heavily. The result supports using early KAK/cosine spaces as a
  promising geometry source for traversal diagnostics or a future calibrated
  merge rule, not as a standalone production clustering rule.
- The adaptive separated-space diffusion rerun used the pydiffmap
  variable-bandwidth kernel inside each KAK/cosine block. It also produced
  `12` ok rows and `3` calibration-support fail-closed rows. Its best row was
  again `binary__adaptive_modes_02_05`, with `130` clusters, singleton
  fraction `0.261538`, NMI `0.516169`, and ARI `0.062483`.
- Adaptive separated-space diffusion reduced singleton fragmentation in the
  best early binary block relative to fixed separated-space diffusion
  (`0.261538` versus `0.568627`) but did not improve ARI (`0.062483` versus
  `0.085175`). The tradeoff supports treating adaptive bandwidth as a
  smoothing diagnostic or candidate merge-precondition, not the current best
  inference rule.
- A context-quality reranking intentionally ignored the external Julia ARI and
  measured internal context utility: non-singleton coverage, medium-size
  context count, gene-weighted within-context GO Jaccard, and Fisher/BH
  enrichment fraction over non-singleton contexts. Under that lens, full
  adaptive diffusion and full fixed diffusion are stronger broad context trees
  than the separated KAK trees. Full adaptive diffusion produced `63` clusters,
  `54` non-singleton contexts, `40` medium-size contexts, gene-weighted
  Jaccard `0.143262`, and enrichment fraction `0.925926`.
- Separated KAK/cosine spaces remain useful as fine-context diagnostics rather
  than as the main context tree. The adaptive TF-IDF `14-30` separated-space
  tree had the highest gene-weighted Jaccard among tested candidates
  (`0.163748`) and enrichment fraction `0.819820`, but it created `220`
  clusters and put `15.5%` of genes into singleton contexts.
- This result supports using cosine/KAK spectral decomposition as a benchmark
  geometry probe. It does not make raw block selection production-admissible
  without selected-tree calibration and explicit support handling.
- A subspace-lens analysis fixed full adaptive diffusion as the main context
  tree and then measured how KAK/cosine blocks refine those contexts. The best
  refinement row was raw KAK `tfidf__adaptive_modes_14_30` with `311` clusters,
  lens purity to main context `0.813656`, main-context purity to lens
  `0.241821`, `4` strongly split main contexts, and refinement score
  `1.020799`.
- Lower-fragmentation lenses are more interpretable as overlays: separated
  fixed diffusion `tfidf__adaptive_modes_14_30` had `154` clusters, singleton
  fraction `0.054054`, and refinement score `0.933389`; raw KAK
  `binary__adaptive_modes_10_15` had `101` clusters, singleton fraction
  `0.022760`, and refinement score `0.896661`.
- The main repeated refinement target is full-adaptive main context `60`
  (`223` genes). Many lenses split it into dozens of subcontexts, which supports
  using KAK/cosine blocks as conditional diagnostic lenses over the main context
  tree instead of as independent replacement trees.
- A consolidated plot list was generated for the current interpretation: full
  adaptive diffusion is the main clustering, and KAK/cosine decompositions are
  diagnostic subspace lenses. The list includes the main embedding, main tree,
  context-quality plots, lens-refinement scatter, raw geometry overview, and
  top lens panels.
- The current decomposition should be described as cosine spectral or
  KAK-inspired, not as a formal Cartan decomposition. A production Cartan claim
  would require an explicit transformation group, compact/isotropy component,
  abelian invariant axis, invariant coordinates, and selected-subspace
  calibration law.
- A covariance-axis stability diagnostic now tests the centered dual covariance
  sample PCA axis, keeping genes aligned while resampling feature columns and
  aligning replicate axes by sign and Procrustes. On the Julia matrix with `30`
  replicates and `80%` feature subsampling, both binary and TF-IDF covariance
  axes are highly stable: median axis-1 absolute cosine is `0.999692` for
  binary and `0.999702` for TF-IDF; the top-6 subspace median canonical
  correlation is `0.998592` for binary and `0.998271` for TF-IDF. This supports
  a stable covariance-axis diagnostic, not a formal Cartan invariant-axis
  claim.
- A focused alpha sweep over six useful KAK/cosine lenses produced `54` rows:
  `48` ok and `6` fail-closed gate rows. Full adaptive diffusion was held fixed
  as the main context tree; only the lens edge and sibling alphas were swept.
- The sweep shows that sibling alpha is the main lens-resolution knob. Larger
  sibling alpha increases cluster count and lens purity to the main context, but
  also increases singleton fragmentation and lowers main-context-to-lens purity.
  Edge alpha is mostly secondary in the ok rows.
- Raw KAK TF-IDF `14-30` remains the strongest high-resolution lens, but it is
  visibly over-resolving the main context tree. Its best tradeoff in this sweep
  was `edge_alpha=0.0003`, `sibling_alpha=0.003`, with `244` clusters,
  singleton-gene fraction `0.129445`, lens-to-main purity `0.752489`, and
  diagnostic lens score `1.003044`.
- Lower-fragmentation overlays are better for practical inspection. Raw KAK
  binary `10-15` is best at `sibling_alpha=0.01` with `101` clusters and
  singleton-gene fraction `0.022760`; separated fixed TF-IDF `14-30` is best at
  `sibling_alpha=0.003` with `135` clusters and singleton-gene fraction
  `0.042674`; adaptive binary `02-05` is best at `sibling_alpha=0.01` with
  `130` clusters and singleton-gene fraction `0.048364`.
- The six fail-closed rows occurred at `edge_alpha=0.003` for separated fixed
  binary `02-05` and separated adaptive TF-IDF `06-09`. Both failures report no
  strict-null or stopped-edge empirical-null calibration support, so loosening
  edge alpha can break support even when the visible fragmentation metric looks
  attractive.
- Ward-Euclidean and complete linkage were added as comparison trees for the
  KAK/cosine lens coordinates. This is not Ward over Hamming distance; Ward is
  only used on Euclidean observations, while average and complete use Euclidean
  condensed distances.
- A direct raw KAK binary `10-15` smoke at `edge_alpha=0.001` and
  `sibling_alpha=0.01` shows the risk: average linkage produced `101` clusters,
  singleton-gene fraction `0.022760`, lens-to-main purity `0.512091`, and
  score `0.877331`; Ward-Euclidean produced `638` clusters, singleton-gene
  fraction `0.846373`, lens-to-main purity `0.971550`, and score `0.190178`.
  Ward therefore over-fragments this lens at the current alpha pair.
- After adding complete linkage, the same raw KAK binary `10-15` smoke shows
  complete also over-fragments this lens: `638` clusters, singleton-gene
  fraction `0.846373`, lens-to-main purity `0.970128`, and score `0.199089`.
  Complete is therefore useful to include in the large diagnostic grid, but the
  first local evidence still favors average linkage for this lens.
- Raw KAK/cosine lens axes now have an exact feature-loading map through the
  dual SVD relation \(q_j = Z^T u_j / \sqrt{\lambda_j}\), where \(Z\) is the
  row-normalized weighted feature matrix. This connects the common axis and
  selected variant lens axes back to GO features before clustering. Separated
  diffusion lenses are excluded from this exact map because their coordinates
  are nonlinear overlays.
- The first feature-axis clustering run mapped raw KAK binary `10-15` and raw
  KAK TF-IDF `14-30` to features, then clustered with the usual KL-TE gate
  path. Binary `10-15` reproduced the useful `101`-cluster lens with singleton
  fraction `0.022760`; TF-IDF `14-30` produced `311` clusters and singleton
  fraction `0.220484`. For both lenses, the top common-to-variant feature
  bridge was positive regulation of DNA-templated transcription
  (`GO:0045893`).
- A debug rerun added explicit reconstruction checks for the raw lens
  feature-axis map. The binary `10-15` lens had block-coordinate reconstruction
  max error `2.55e-15`, full feature-axis reconstruction max error `6.55e-15`,
  and feature-axis orthogonality max error `6.51e-15`. The TF-IDF `14-30` lens
  had corresponding errors `1.91e-15`, `3.33e-15`, and `4.51e-15`. This
  rules out a normalization or dual-axis implementation bug at numerical
  precision; the remaining fragmentation is geometric/traversal behavior, not
  a failed raw KAK feature map.
- The tree-strategy semantic panel joins the requested interpretation columns:
  tree strategy, lens family, linkage, alpha, cluster count, singleton
  fraction, medium contexts, GO Jaccard, enrichment fraction, main-context
  refinement, axis-feature bridge, edge/sibling p-value continuity, and
  radius/angle/action. The local smoke wrote `59` rows: `2` main-context trees,
  `37` diagnostic-lens rows, `19` fragmentation-lens rows, and `1` coarse
  baseline row.
- The panel confirms the working interpretation: full adaptive/fixed diffusion
  are main context trees, raw/separated KAK/cosine blocks are diagnostic lenses,
  and high-resolution TF-IDF or Ward/complete rows often become fragmentation
  lenses. The top feature bridge for exact raw KAK feature-axis rows remains
  positive regulation of DNA-templated transcription (`GO:0045893`), while the
  global p-value continuity summary is
  `edge_parent=0.309670; recursive_sibling=0.538674; raw_recursive_sibling=0.443386`.
- The AWS tree-strategy semantic-panel wrapper was added as a thin precomputed
  result joiner. It requires a repository-shaped S3 result bundle because the
  AWS Docker image excludes `benchmarks/results`. Local AWS submission was not
  completed because `aws sts get-caller-identity` could not reach the
  `us-east-1.signin.aws.amazon.com` endpoint from this environment.
- The main rectangular tree plotter was corrected to route edges as
  dendrogram-style elbows instead of straight parent-child diagonals. Existing
  saved PNGs still show the old straight-line rendering until regenerated; the
  fix is visual only and does not change clustering, traversal, p-values, or
  assignments. In the Plotly 3D subembedding, white lines are Plotly scene
  grid/axis lines, not inferred tree edges.
- The full AWS KAK/cosine lens linkage alpha sweep completed after retrying
  shard `0005`, whose first failure was an ECR pull timeout rather than a
  method failure. The merged output contains the full `450` candidate rows over
  six lenses, average/complete/Ward-Euclidean linkage, and a `5 x 5`
  edge/sibling alpha grid. Row states were `330` ok and `120` fail-closed gate
  rows.
- The full sweep confirms average linkage as the strongest diagnostic lens
  tree for the current KAK/cosine overlays. Raw KAK TF-IDF `14-30` average
  reached the highest score (`1.003044`) at `edge_alpha=0.0001` or `0.0003`
  with `sibling_alpha=0.003`, producing `244` clusters and singleton fraction
  `0.129445`. Raw KAK binary `10-15` average was the most practical
  lower-fragmentation raw overlay (`108` clusters, singleton fraction
  `0.025605`, score `0.896091`) at `edge_alpha=0.0001`,
  `sibling_alpha=0.001`.
- Complete and Ward-Euclidean are not uniformly bad but remain secondary
  diagnostic linkages. Complete can be useful for separated fixed binary
  diffusion (`106` clusters, singleton fraction `0.035562`, score `0.853812`),
  and Ward can be useful for separated fixed binary diffusion (`87` clusters,
  singleton fraction `0.022760`, score `0.835188`). Raw TF-IDF Ward is the
  clearest bad case, with a best score of only `0.110441` and singleton
  fraction `0.866287`.
- The full semantic panel regenerated from the merged AWS sweep wrote `339`
  rows: `2` main context trees, `149` diagnostic lenses, `185` fragmentation
  lenses, `2` fine lenses, and `1` coarse baseline. The panel preserves the
  working rule: full adaptive diffusion remains the main context tree, while
  KAK/cosine decompositions are diagnostic subspace lenses for geometry,
  feature-axis bridges, and traversal fragmentation.
- A feature-side KAK subspace diagnostic was added to correct the previously
  gene-side-only interpretation. It treats matrix columns as leaves, embeds
  binary GO features in each raw KAK block as `Q_B sqrt(Lambda_B)`, and then
  attempts the normal KL-TE gate path on those feature coordinates using an
  explicit continuous feature-space contract. On the Julia binary matrix,
  all `8` adaptive blocks wrote feature coordinates but failed closed in the
  gate layer (`zero-dimensional spectral context` or no positive active PCA
  direction). This shows that the current implementation now exposes the
  requested feature-side geometry, but production feature-side edge/sibling
  equations are still missing.
- A classical UMAP-only clustering panel was generated for the Julia full-data
  UMAP coordinates as a pre-lens baseline. Complete linkage with `K=20` gave
  the highest reference ARI in this diagnostic (`0.145744`, NMI `0.403709`);
  Ward (`0.129359`, NMI `0.407948`), average (`0.133501`, NMI `0.402036`),
  GMM (`0.132820`, NMI `0.404571`), and k-means (`0.121165`, NMI
  `0.398813`) were similar. The result is useful as a visual manifold
  diagnostic only: it clusters the stored two-dimensional UMAP plane and is not
  a calibrated tree-context test.

## Evidence

- `benchmarks/diagnostics/spectral/adaptive_cosine_kak_benchmark_probe.py`
  contains the current-compatible diagnostic runner.
- `benchmarks/diagnostics/spectral/adaptive_cosine_kak_matrix_probe.py`
  contains the feature-matrix counterpart.
- `benchmarks/results/diagnostics/adaptive_cosine_kak_probe_method_proof_cal_default_20260605/kak_benchmark_probe_rows.csv`
  and `benchmarks/results/diagnostics/adaptive_cosine_kak_probe_method_proof_cal_enforced_20260605/kak_benchmark_probe_rows.csv`
  contain method-proof default and support-enforced per-block rows.
- `benchmarks/results/diagnostics/adaptive_cosine_kak_probe_full_gate_bundle_20260605/kak_benchmark_probe_rows.csv`
  and `benchmarks/results/diagnostics/adaptive_cosine_kak_probe_full_cal_enforced_20260605/kak_benchmark_probe_rows.csv`
  contain full-suite default and support-enforced per-block rows.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_spectral_blocks_gate_bundle_20260605/matrix_kak_probe_summary.csv`
  and `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_spectral_blocks_support_enforced_20260605/matrix_kak_probe_summary.csv`
  contain the combined Julia GO matrix default and support-enforced summaries.
- `scripts/kak_signal_adaptive_umap_tree_page.py` restores the visualization
  layer against current matrix-probe outputs and adds per-sample and per-merge
  geometry diagnostics.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_20260606/kak_signal_adaptive_block_geometry_summary.csv`
  summarizes block-level radius, angle, invariant-axis, and internal-tree
  geometry.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_20260606/kak_signal_adaptive_internal_tree_geometry.csv`
  contains one row per internal linkage merge, including split balance,
  sibling separation ratios, sibling angles, common-axis gaps, and assigned
  cluster concentration.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_20260606/kak_signal_adaptive_umap_tree_page.png`
  is the consolidated restored visual diagnostic page.
- `data/reference/adg6375_File_S7_endotypes_julia.txt` supplies the Julia
  endotype labels used for external diagnostic NMI/ARI scoring.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_nmi_ordered_20260610/kak_signal_adaptive_reference_labels.csv`
  contains the matched reference labels after symbol-to-Entrez mapping.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_nmi_ordered_20260610/kak_signal_adaptive_reference_nmi_metrics.csv`
  contains per-block reference NMI/ARI scores and display-order ranks.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_nmi_ordered_20260610/kak_signal_adaptive_reference_nmi_ranking.png`
  is the NMI ranking plot for the Julia rerun.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_nmi_ordered_20260610/kak_signal_adaptive_umap_tree_page.png`
  is the consolidated Julia diagnostic page sorted by reference NMI.
- `scripts/analysis/run_feature_matrix_with_umap.py` was updated to use the
  current gate-annotation bundle path before recreating the historical
  full-matrix method outputs.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/09_full_data_method_reference_comparison_20260610/all_method_and_kak_reference_comparison.csv`
  rescored historical method assignments and all ok KAK blocks against the
  same `262` matched Julia-reference genes.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/09_full_data_method_reference_comparison_20260610/selected_method_reference_comparison.csv`
  contains the compact comparison table for historical methods plus the
  top-NMI and top-ARI KAK blocks.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/09_full_data_method_reference_comparison_20260610/method_vs_kak_reference_nmi_ari_scatter.png`
  plots NMI against ARI and labels cluster counts, making the fragmentation
  effect visible.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/09_full_data_method_reference_comparison_20260610/selected_method_reference_comparison_bars.png`
  compares ARI, NMI, and log cluster count for the selected rows.
- `benchmarks/diagnostics/spectral/adaptive_cosine_kak_diffusion_matrix_probe.py`
  contains the separated-space diffusion diagnostic.
- `benchmarks/diagnostics/spectral/kak_lens_alpha_sweep.py` contains the
  focused alpha sweep for selected KAK/cosine diagnostic lenses against the
  fixed full adaptive diffusion main context tree.
- `benchmarks/diagnostics/spectral/kak_lens_feature_axis_clustering.py`
  contains the exact raw KAK/cosine lens-axis feature-loading map and clustering
  diagnostic.
- `benchmarks/diagnostics/spectral/kak_feature_subspace_clustering.py`
  contains the feature-side eigenvariant subspace clustering diagnostic, where
  features rather than genes are tree leaves.
- `tests/validation/84_test_kak_lens_feature_axis_clustering.py` checks that
  recovered feature axes reconstruct raw cosine sample coordinates and that
  common/variant feature connection scores are well-formed.
- `tests/validation/86_test_kak_feature_subspace_clustering.py` checks the
  feature-side `Q_B sqrt(Lambda_B)` coordinate construction and active-feature
  selection.
- `benchmarks/diagnostics/spectral/covariance_axis_stability.py` contains the
  covariance/PCA axis stability diagnostic using feature resampling,
  sign-aligned axis comparison, and Procrustes top-subspace alignment.
- `tests/validation/83_test_covariance_axis_stability.py` checks sign
  invariance, Procrustes rotation invariance, and a low-rank synthetic
  covariance-axis stability case.
- `benchmarks/cloud/aws_kak_lens_linkage_alpha_sweep.py` contains the AWS
  Batch shard/merge wrapper for the KAK/cosine lens linkage alpha sweep.
- `tests/validation/82_test_aws_kak_lens_linkage_alpha_sweep.py` checks shard
  index resolution, lens/linkage group distribution, and merge behavior.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/10_full_data_kak_separated_diffusion_20260610/matrix_kak_diffusion_probe_summary.csv`
  contains the per-block separated-space diffusion outcomes.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/10_full_data_kak_separated_diffusion_20260610/separated_diffusion_vs_existing_reference_comparison.csv`
  compares separated-space diffusion with the recreated full-matrix methods and
  raw KAK block assignments on the same Julia-reference subset.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/10_full_data_kak_separated_diffusion_20260610/separated_diffusion_vs_existing_ari_nmi_scatter.png`
  plots the same comparison in NMI/ARI space with cluster-count-scaled points.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/11_full_data_kak_separated_adaptive_diffusion_20260610/matrix_kak_diffusion_probe_summary.csv`
  contains the adaptive separated-space diffusion outcomes.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/11_full_data_kak_separated_adaptive_diffusion_20260610/adaptive_vs_fixed_separated_and_existing_reference_comparison.csv`
  compares adaptive separated-space diffusion with fixed separated-space
  diffusion, raw KAK block assignments, and the recreated full-matrix methods.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/11_full_data_kak_separated_adaptive_diffusion_20260610/adaptive_vs_fixed_separated_and_existing_ari_nmi_scatter.png`
  plots the fixed/adaptive separated-space comparison against the Julia
  reference labels.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/12_context_quality_tree_comparison_20260610/context_method_quality_summary.csv`
  reranks candidate trees by internal context-quality metrics instead of
  external ARI.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/12_context_quality_tree_comparison_20260610/context_quality_report.md`
  contains per-method top contexts and top enriched GO terms.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/12_context_quality_tree_comparison_20260610/context_quality_scatter.png`
  plots within-context GO Jaccard against enrichment fraction.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/13_subspace_lens_vs_main_adaptive_contexts_20260610/subspace_lens_vs_main_context_summary.csv`
  compares KAK/cosine lens assignments against the full adaptive diffusion main
  context labels.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/13_subspace_lens_vs_main_adaptive_contexts_20260610/subspace_lens_vs_main_context_report.md`
  records the top refinement lenses and the main contexts most strongly split
  by those lenses.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/13_subspace_lens_vs_main_adaptive_contexts_20260610/subspace_lens_vs_main_context_scatter.png`
  plots lens purity against main-context purity.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/14_main_context_subspace_lens_plot_list_20260610/plot_list.md`
  and `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/14_main_context_subspace_lens_plot_list_20260610/plot_list.html`
  list and embed the current main-context and subspace-lens plots.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/15_kak_lens_alpha_sweep_20260610/kak_lens_alpha_sweep_summary.csv`
  contains the six-lens edge/sibling alpha sweep rows.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/15_kak_lens_alpha_sweep_20260610/kak_lens_alpha_sweep_report.md`
  summarizes the top alpha settings and best setting per lens.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/15_kak_lens_alpha_sweep_20260610/alpha_sweep_fragmentation_vs_purity.png`
  plots lens cluster count against lens purity to the main adaptive context.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/15_kak_lens_alpha_sweep_20260610/alpha_sweep_top_lens_scores.png`
  plots the top diagnostic alpha settings by lens score.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/16_kak_lens_linkage_smoke_20260611/kak_lens_alpha_sweep_summary.csv`
  contains the direct average-versus-Ward smoke comparison for raw KAK binary
  `10-15`.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/17_aws_kak_lens_linkage_alpha_sweep_smoke_20260611/merged/kak_lens_linkage_alpha_sweep_summary.csv`
  contains the local two-shard wrapper smoke merged output.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/17_aws_kak_lens_linkage_alpha_sweep_smoke_20260611/merged/aws_kak_lens_linkage_alpha_sweep_manifest.json`
  records the local shard/merge smoke contract.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/18_kak_lens_linkage_complete_smoke_20260611/kak_lens_alpha_sweep_summary.csv`
  contains the average, complete, and Ward smoke comparison for raw KAK binary
  `10-15`.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/19_covariance_axis_stability_20260611/covariance_axis_stability_summary.csv`
  records the Julia binary/TF-IDF covariance-axis stability summary.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/19_covariance_axis_stability_20260611/covariance_axis_stability_replicates.csv`
  contains one row per feature-resampling replicate.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/19_covariance_axis_stability_20260611/covariance_axis_stability_histograms.png`
  plots the sign-aligned axis stability and Procrustes residual distributions.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/20_kak_lens_feature_axis_clustering_20260612/kak_lens_feature_axis_clustering_summary.csv`
  summarizes the raw lens feature-axis clustering run.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/20_kak_lens_feature_axis_clustering_20260612/raw_kak__binary__adaptive_modes_10_15/lens_feature_axis_scores.csv`
  maps the raw KAK binary `10-15` common and variant axes back to GO features.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/20_kak_lens_feature_axis_clustering_20260612/raw_kak__tfidf__adaptive_modes_14_30/lens_feature_axis_scores.csv`
  maps the raw KAK TF-IDF `14-30` common and variant axes back to GO features.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/21_kak_lens_feature_axis_clustering_debug_20260612/kak_lens_feature_axis_clustering_summary.csv`
  records the feature-axis reconstruction and orthogonality debug metrics for
  the raw KAK binary `10-15` and TF-IDF `14-30` lens mappings.
- `benchmarks/results/diagnostics/tree_strategy_semantic_panel_smoke_20260612/tree_strategy_semantic_panel.csv`
  contains the joined tree-strategy semantic interpretation panel.
- `benchmarks/results/diagnostics/tree_strategy_semantic_panel_smoke_20260612/tree_strategy_semantic_panel_report.md`
  summarizes the semantic-role counts and top refinement rows.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/22_aws_kak_lens_linkage_alpha_sweep_full_20260612/merged/kak_lens_linkage_alpha_sweep_summary.csv`
  contains the completed AWS full KAK/cosine lens linkage alpha sweep.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/22_aws_kak_lens_linkage_alpha_sweep_full_20260612/merged/aws_kak_lens_linkage_alpha_sweep_manifest.json`
  records the completed full AWS shard/merge contract.
- `benchmarks/results/diagnostics/tree_strategy_semantic_panel_aws_full_kak_20260612/tree_strategy_semantic_panel.csv`
  contains the semantic panel regenerated from the full AWS alpha sweep.
- `benchmarks/results/diagnostics/tree_strategy_semantic_panel_aws_full_kak_20260612/tree_strategy_semantic_panel_report.md`
  summarizes the semantic-role counts for the completed full AWS panel.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/23_kak_binary_feature_subspace_clustering_20260612/kak_feature_subspace_clustering_summary.csv`
  records the binary-only Julia feature-side KAK subspace diagnostic; all
  blocks failed closed in the current gate layer after writing feature
  coordinates.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/25_julia_classical_umap_clustering_20260612/julia_classical_umap_clustering_summary.csv`
  records k-means, Ward, average, complete, GMM, and DBSCAN clustering on the
  stored Julia UMAP coordinates.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/25_julia_classical_umap_clustering_20260612/julia_classical_umap_clustering_panel.png`
  shows the classical-clustering UMAP panel.
- `benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/25_julia_classical_umap_clustering_20260612/julia_reference_endotypes_on_umap.png`
  shows the sparse Julia reference endotype labels on the same UMAP plane.
- `benchmarks/results/diagnostics/adaptive_cosine_kak_probe_full_gate_bundle_20260605/kak_probe_full_block_logic_scatter.png`
  visualizes early-block success versus common-mode and energy-based failure
  modes.

## Links

- [[historical-kak-spectral-pipelines-20260605]]
- [[full-benchmark-run-20260605]]
- [[mp-projection-dimension-behavior-sweeps-20260605]]
- [[selected-pca-projected-wald-validation]]
