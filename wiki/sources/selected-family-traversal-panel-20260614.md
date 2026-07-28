---
title: Selected Family Traversal Panel 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/selected/family/selected_family_traversal_panel.py
  - tree_break_selection/plot/multiscale_umap.py
  - tree_break_selection/hierarchy_analysis/tree_decomposition.py
  - tests/validation/calibration/selected/family/102_test_selected_family_traversal_panel.py
  - benchmarks/diagnostics/calibration/sibling/gates/fixed_sibling_gate_profile_validation.py
tags:
  - source
  - diagnostics
  - traversal
  - selection
---

# Selected Family Traversal Panel 2026-06-14

## Summary

`selected_family_traversal_panel.py` is a diagnostic-only runner that compares
baseline TBS traversal with fixed-coordinate selected-root and selected-family
profiles, then emits multi-scale node, region, and sample outputs instead of
only one flat clustering.

## Key Points

- The panel treats `fixed_coordinate_global_passthrough_refined_v1` as the
  primary V1 selected-family diagnostic candidate, not as a production default.
- The default method grid includes baseline projected-Wald traversal,
  `fixed_coordinate_guarded_v1`, `fixed_coordinate_selective_root_v1`,
  `fixed_coordinate_selective_passthrough_v1`, and
  `fixed_coordinate_global_passthrough_refined_v1`.
- Fixed-profile rows surface the existing non-cross-fit fixed coordinate BH
  path; the panel does not add cross-fitting or reintroduce adaptive parent
  PCA into the sibling statistic.
- The runner writes `selected_family_traversal_rows.csv`,
  `selected_family_guard_rows.csv`, `multiscale_node_decisions.csv`,
  `multiscale_regions.csv`, `multiscale_gene_assignments.csv`,
  `production_admissibility_components.csv`,
  `production_admissibility_summary.csv`, and `manifest.json`.
- `TreeDecomposition.decompose_tree()` now returns a `traversal_trace`, making
  split, pass-through, and boundary decisions inspectable without duplicating
  traversal logic outside the production path.
- Multi-scale node decisions distinguish `stable_boundary`,
  `selected_root_blocked`, `selected_family_blocked`,
  `unstable_passthrough_zone`, `accepted_internal_split`, and
  `leaf_fragment`.
- The output role label for benchmark null data is `selected_null` so default
  CSV readers do not silently parse the role as missing.
- `multiscale_gene_assignments.csv` uses the decomposition sample labels, not
  raw tree leaf node ids, so the rows can join directly to UMAP coordinate
  tables.
- `tree_break_selection/plot/multiscale_umap.py` joins multi-scale assignments
  with existing UMAP coordinates and renders stable regions as the primary
  color layer with pass-through or guard zones as an overlay.
- The panel now writes checkpoint CSVs per case-role-method-replicate and can
  resume with `--resume-from-checkpoints`; long rows can be skipped and
  recorded with `--per-row-timeout-seconds`.
- A one-replicate full-suite supported smoke over
  `fixed_coordinate_global_passthrough_refined_v1` wrote `101` completed rows
  out of `106` supported binary/direct-categorical rows; `5` rows timed out at
  the `60` second per-row budget. Binary completed rows had `4` null false
  splits, all in overlap-template cases. Direct categorical nulls stayed
  closed in completed rows, but categorical signal mean ARI was `0.717742`.
  The production summaries therefore fail closed.
- Production-admissibility output remains conservative: baseline rows are
  reference-only, fixed-profile transfer candidates remain diagnostic-only
  unless all required production contract components pass, and missing
  null/signal coverage fails closed.

## Evidence

- `tests/validation/calibration/selected/family/102_test_selected_family_traversal_panel.py` covers method
  validation, node decision classification, conservative production summaries,
  CSV-stable null role output, checkpoint resume, row timeout recording, UMAP
  overlay rendering, and a tiny binary output run.
- `tests/localization/35_test_gates_traversal.py` continues to cover
  pass-through traversal behavior through `TreeDecomposition`.
- `tests/validation/calibration/sibling/gates/101_test_fixed_sibling_gate_profile_validation.py` covers
  the fixed-profile evidence path reused by this panel.

## Links

- [[fixed-sibling-gate-profile-validation-20260613]]
- [[data-independent-sibling-gate-traversal-panel-20260613]]
- [[selected-root-pass-through-null-fixture-20260614]]
- [[selected-root-selected-family-traversal-literature-20260614]]
- [[open-mathematical-questions]]
