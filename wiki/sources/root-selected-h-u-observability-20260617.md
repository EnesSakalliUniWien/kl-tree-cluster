---
title: Root Selected H U Observability 2026-06-17
type: source
status: draft
updated: 2026-06-17
sources:
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/node_spectral_result.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/spectral_decomposition_result.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/tree_estimator.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/marchenko_pastur.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/child_parent_divergence/child_parent_divergence_annotation/spectral_context.py
  - benchmarks/diagnostics/calibration/root_selected_region_margins.py
  - benchmarks/diagnostics/calibration/root_tie_rank_conditioned_coherent_topology_join.py
  - benchmarks/diagnostics/calibration/root_selected_h_u_observability_panel.py
  - tests/validation/55_test_root_selected_region_margins.py
  - tests/validation/168_test_root_tie_rank_conditioned_coherent_topology_join.py
  - tests/validation/175_test_root_selected_h_u_observability_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_region_margins_overlap_case_family/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_region_margins_overlap_case_family/root_selected_region_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_topology_join_mild_accumulated/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_topology_join_mild_accumulated/conditioned_coherent_joined_feasibility_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_h_u_observability_mild_accumulated/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_h_u_observability_mild_accumulated/root_selected_h_u_observability_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_h_u_observability_mild_accumulated/root_selected_h_u_observability_summary.csv
tags:
  - source
  - diagnostics
  - root
  - spectral
  - population-law
---

# Root Selected H U Observability 2026-06-17

## Summary

`root_selected_h_u_observability_panel.py` audits whether the current
selected-root artifacts contain enough information to estimate the local
null-whitened spectral population law \(H_u\). It reads the accumulated
root topology/importance feasibility rows and the population-law requirement
rows, then checks for the fields needed to move from identity MP to a
deformed MP edge: a root eigenvalue spectrum, an active feature count, MP
threshold rows, and a captured or computable deformed edge.

The follow-up implementation now carries full root covariance eigenvalues and
active feature counts separately from the projected-Wald PCA eigenvalues. After
regenerating the overlap root summary, topology join, root-tail, population
requirement, and \(H_u\) observability panels, the mild accumulated run writes
`7` rows and one summary with `h_u_estimable_target_count = 3`,
`missing_spectrum_target_count = 0`, and
`missing_feature_count_target_count = 0`. Two roots already have exact
selected-root tail support and therefore do not need \(H_u\) for current
production inference. Two roots still lack support before \(H_u\) estimation
can be useful. The remaining three roots now have the captured bulk inputs and
need the deformed MP edge calculation itself:
`overlap_mod_4c_small`, `overlap_mod_6c_med`, and `overlap_part_4c_small`.

## Key Points

- The panel is diagnostic-only and does not produce p-values.
- Existing target rows now contain `root_mp_threshold_rows`,
  `root_active_feature_count`, `root_full_eigenvalue_count`, and
  `root_full_component_eigenvalues_json`. The projected-Wald PCA eigenvalues
  remain separately recorded as `root_projected_eigenvalues_json`.
- `overlap_mod_4c_small`, `overlap_mod_6c_med`, and
  `overlap_part_4c_small` remain fail-closed, but their status is now
  `h_u_estimation_inputs_available_deformed_edge_missing`.
- `overlap_heavy_4c_small_feat` and `overlap_unbal_4c_small` are marked
  `support_missing_before_h_u_estimation`, because the support law itself is
  not yet populated.
- `overlap_extreme_4c` and `overlap_unbal_6c_med` are marked
  `exact_tail_support_available_h_u_optional`, because production inference
  can defer to the existing exact selected-tail panel.
- The old-commit comparison remains visible through
  `legacy_comparison_interpretation`; `overlap_mod_4c_small` is still labeled
  `legacy_full_method_leaks_selected_null_root_risk`.

## Evidence

The required deformed-MP object is not a scalar top-eigenvalue correction. A
local edge \(b(H_u,\gamma)\) requires the null-whitened root bulk spectrum and
the local aspect ratio \(\gamma=d_u/m_u\). The previous root-tail rows exposed
only

\[
S_{\mathrm{root}}
=
\log(\lambda/\lambda_{\mathrm{MP}})
\]

through the top ratio. The updated path captures full root eigenvalue vectors
from the spectral worker before PCA truncation and records the active feature
count used in the MP aspect ratio. Therefore the next implementable step has
moved from capture to computing the deformed MP edge before rerunning the
selected-root tail panel with \(b(H_u,\gamma)\).

The summary row records:

- `target_count = 7`
- `h_u_estimable_target_count = 3`
- `missing_spectrum_target_count = 0`
- `missing_feature_count_target_count = 0`
- `support_missing_target_count = 2`
- `exact_tail_support_target_count = 2`

## Links

- [[root-selected-population-law-requirement-20260617]]
- [[root-selected-deformed-mp-edge-20260617]]
- [[root-selected-spectral-tail-nearest-support-20260617]]
- [[root-selected-action-dominance-tail-20260617]]
- [[local-marchenko-pastur-rule]]
