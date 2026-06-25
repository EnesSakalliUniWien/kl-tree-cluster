---
title: Root Selected Region Overlap Case Family 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/root_selected_region_margins.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_region_margins_overlap_case_family
tags:
  - source
  - diagnostics
  - root
  - selected-region
  - topology
---

# Root Selected Region Overlap Case Family 2026-06-16

## Summary

The existing root selected-region margin diagnostic was run on the seven
overlap cases used by the selected-neighborhood candidate audits. All seven
binary overlap roots require discrete tie-cell geometry. This means a smooth
root outgoing-balance threshold is not the missing law; the missing law is a
discrete selected-region/tie-cell conditioning object for the root.

## Key Points

- The run covers `overlap_extreme_4c`, `overlap_heavy_4c_small_feat`,
  `overlap_mod_4c_small`, `overlap_mod_6c_med`, `overlap_part_4c_small`,
  `overlap_unbal_4c_small`, and `overlap_unbal_6c_med`.
- All `7/7` cases have `root_selected_region_law_status =
  discrete_tie_cell_geometry_required`.
- The diagnostic writes `7` root summary rows, `3,493` merge-margin rows, and
  `9` relationship rows. The regenerated merge-margin table also records
  selected tie-rank coordinates for deterministic tie-breaking. Runtime was
  about `57.20` seconds.
- Root-child construction margins are essentially zero in all seven cases:
  minimum margins are numerical zero and median margins are zero. The
  root-child tied-minimum counts range from `254` to `420` construction merges.
- Root selected ratios remain large, from `47.831161` on
  `overlap_extreme_4c` to `1768.512366` on `overlap_unbal_6c_med`, but those
  ratios sit inside tie-heavy discrete selected regions.
- Relationship summaries show strong monotone association between the root
  sibling selected ratio and edge/spectral quantities on this seven-case
  diagnostic slice: `root_edge_path_radial_distance`, `root_edge_path_statistic_margin`,
  and `root_edge_extra_parent_projection_energy` have Spearman correlations
  near `1`. This is descriptive evidence only, not calibration.
- The follow-up [[root-selected-tie-cell-burden-20260616]] quantifies the
  discrete tie-cell burden directly and shows that tie burden is not a monotone
  rescue rule; it is a conditioning coordinate for a future discrete root law.

## Evidence

- `root_selected_region_margins.py` replays average-linkage root-child
  construction inequalities and records root sibling, edge, and spectral
  quantities.
- The overlap case-family output manifest records the seven selected cases,
  output paths, and elapsed runtime.
- The summary table provides the case-level root selected-region law status
  used by the topology-frontier join.

## Links

- [[root-selected-region-margins-20260603]]
- [[root-selected-tie-cell-burden-20260616]]
- [[selected-neighborhood-topology-frontier-diagnostic-20260616]]
- [[selected-neighborhood-measurability-law]]
