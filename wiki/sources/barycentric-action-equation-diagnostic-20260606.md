---
title: Barycentric Action Equation Diagnostic 2026-06-06
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/selected/tail/selected_tail_law_q5_validation.py
  - raw/assets/benchmark-results/selected_hierarchy_topology_refinement_input_20260603_300/selected_geometry_records.csv
  - benchmarks/results/diagnostics/barycentric_action_equation_diagnostic_20260606/barycentric_action_equation_summary.csv
  - benchmarks/results/diagnostics/barycentric_action_equation_diagnostic_20260606/barycentric_action_equation_validation.csv
  - benchmarks/results/diagnostics/barycentric_action_equation_diagnostic_20260606/kak_internal_traversal_fragmentation_summary.csv
  - benchmarks/results/diagnostics/barycentric_action_equation_diagnostic_20260606/kak_internal_traversal_fragmentation_validation.csv
  - benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/08_full_data_adaptive_kak_signal_umap_tree_page_20260606/kak_signal_adaptive_internal_tree_geometry.csv
tags:
  - source
  - calibration
  - barycentric
  - traversal
  - diagnostics
---

# Barycentric Action Equation Diagnostic 2026-06-06

## Summary

This diagnostic tests whether the overlooked geometric law should be promoted
from the local barycentric identity to a broader calibration equation. The
tested candidate is a path-conditioned barycentric action law: edge action,
child balance, barycentric leverage, sampling scale, selected spectral
geometry, and selected-subspace angle. The result separates selected-tail
calibration from traversal-fragmentation explanation.

## Key Points

- The branch-length columns in
  `selected_hierarchy_topology_refinement_input_20260603_300/selected_geometry_records.csv`
  do not satisfy the exact barycentric child-parent length identity, so they
  are not the right equation-system variables.
- For selected-tail calibration, the compact existing
  `q5_barycentric_edge_spectral` law remains the best tested candidate. Its
  median residual-tail exceedance absolute error is `0.002386`, versus
  `0.007755` for `q5_edge_spectral_only`.
- Adding selected-subspace angle to the barycentric selected-tail law improves
  replicate-modulo \(R^2\) and some case-holdout absolute-error summaries, but
  it fails feature-family transfer badly. Its median residual-tail exceedance
  absolute error is about `0.114`, so it is not a production calibration
  improvement.
- On the restored KAK internal geometry table, radius-plus-angle variables do
  explain traversal fragmentation. A leave-one-block-out classifier for
  current local pure fragments improves from median AUC `0.835439` with
  size/balance only to `0.892810` with radius, angle, independent fraction, and
  common-axis gap.
- The actionable split is therefore: use the compact barycentric edge/spectral
  equation as the current selected-tail calibration candidate, and use
  radius/angle/invariant-axis variables as traversal diagnostics and possible
  future stratifiers, not as a direct tail-law replacement.

## Evidence

- `benchmarks/diagnostics/calibration/selected/tail/selected_tail_law_q5_validation.py`
  supplies the selected-tail validation helpers reused by this diagnostic.
- `benchmarks/results/diagnostics/barycentric_action_equation_diagnostic_20260606/barycentric_action_equation_summary.csv`
  compares the baseline edge/spectral, compact barycentric edge/spectral, and
  angle-augmented candidate laws.
- `benchmarks/results/diagnostics/barycentric_action_equation_diagnostic_20260606/barycentric_action_equation_validation.csv`
  records the fold-level held-out metrics by split strategy.
- `benchmarks/results/diagnostics/barycentric_action_equation_diagnostic_20260606/kak_internal_traversal_fragmentation_summary.csv`
  records the leave-one-block-out KAK traversal-fragment association summary.
- `benchmarks/results/diagnostics/barycentric_action_equation_diagnostic_20260606/kak_internal_traversal_fragmentation_validation.csv`
  records block-level AUCs for the traversal-fragment diagnostic.

## Links

- [[barycentric-method-literature-request-20260604]]
- [[adaptive-cosine-kak-benchmark-probe-20260605]]
- [[open-mathematical-questions]]
- [[top-down-traversal]]
