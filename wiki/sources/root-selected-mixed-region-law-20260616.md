---
title: Root Selected Mixed Region Law 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/root_selected_mixed_region_law.py
  - tests/validation/155_test_root_selected_mixed_region_law.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_mixed_region_law_overlap_case_family/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_mixed_region_law_overlap_case_family/root_selected_mixed_region_law_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_mixed_region_law_overlap_case_family/root_selected_mixed_region_law_relationships.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_mixed_region_law_overlap_case_family/root_selected_mixed_region_law_summary.csv
tags:
  - source
  - diagnostics
  - root
  - selected-region
  - tie-cell
---

# Root Selected Mixed Region Law 2026-06-16

## Summary

`root_selected_mixed_region_law.py` joins root selected-region margins,
rank-aware tie-cell burden, and optional topology-frontier bandwidth evidence
into a single diagnostic table. It formalizes the root event as
\[
\mathcal E_{\mathrm{root}}
=
\mathcal E_{\mathrm{margin}}
\cap
\mathcal E_{\mathrm{tie}}
\cap
\mathcal E_{\mathrm{rank}},
\]
where the rank term records the deterministic lexicographic position of the
selected pair inside each tied minimum merge set. The panel does not compute a
calibrated p-value and does not promote traversal.

## Key Points

- The overlap run writes `7` row records, `19` relationship records, and one
  summary row under `root_selected_mixed_region_law_overlap_case_family`.
- All `7/7` roots are classified as `discrete_tie_rank_region`.
- All `7/7` roots have `root_calibration_status =
  blocked_until_discrete_tie_rank_null_calibrated`.
- Six of seven cases have `root_bandwidth_locality_status =
  bandwidth_reopens_without_root_law_support`; `overlap_unbal_6c_med` is the
  only case where the joined root frontier rows do not reopen at the reference
  bandwidth.
- The summary row reports median root selected ratio `597.314947`, median
  selected tie-rank fraction `0.809091`, `6` bandwidth-reopen cases, and `0`
  hybrid strict-support cases.
- Relationship rows reproduce the earlier separation: raw tie burden is not a
  monotone rescue rule, while selected tie-rank fraction and edge/spectral root
  variables align more strongly with the root selected ratio on this small
  diagnostic slice.
- The strongest descriptive covariates in this seven-case slice are
  `root_edge_path_radial_distance`, `root_edge_path_statistic_margin`, and
  `root_rank_fraction_edge_margin_product`, each with Spearman `1.0` against
  the root selected ratio. This remains descriptive because the selected root
  null law is not calibrated.

## Evidence

- The implementation reads `root_selected_region_summary.csv`,
  `root_selected_tie_cell_burden_rows.csv`, and optionally
  `selected_neighborhood_topology_frontier_rows.csv`.
- The validation test covers discrete-only, smooth-only, and mixed root
  classifications; bandwidth reopening without calibrated root support;
  relationship output; summary output; and file writing.
- The generated manifest records the exact three input surfaces and output
  files for the seven-case overlap artifact.

## Links

- [[root-selected-region-overlap-case-family-20260616]]
- [[root-selected-tie-cell-burden-20260616]]
- [[selected-neighborhood-topology-frontier-diagnostic-20260616]]
- [[selected-neighborhood-measurability-law]]
