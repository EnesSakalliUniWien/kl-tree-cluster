---
title: Root Selected Tie Cell Burden 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/root_selected_tie_cell_burden.py
  - tests/validation/154_test_root_selected_tie_cell_burden.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_tie_cell_burden_overlap_case_family/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_tie_cell_burden_overlap_case_family/root_selected_tie_cell_burden_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_tie_cell_burden_overlap_case_family/root_selected_tie_cell_burden_relationships.csv
tags:
  - source
  - diagnostics
  - root
  - selected-region
  - tie-cell
---

# Root Selected Tie Cell Burden 2026-06-16

## Summary

`root_selected_tie_cell_burden.py` quantifies the discrete root selected-region
blocker exposed by the overlap root-margin replay. It computes a tie-cell
selection burden by summing \(\log m_t\) across root-child construction steps,
where \(m_t\) is the number of tied minimum merge choices at step \(t\). This
is the diagnostic analogue of a tie-breaking conditioning term, not a
calibrated p-value.

## Key Points

- The seven-case overlap run writes `7` burden rows and `12` relationship rows
  under `root_selected_tie_cell_burden_overlap_case_family`.
- Every case has `root_tie_cell_status =
  discrete_tie_burden_observed_diagnostic_only`.
- Tie-heavy construction is pervasive: the root tie-step fraction ranges from
  `0.603679` to `0.718876`; log tie burden ranges from `548.428653` to
  `959.876888`; geometric mean tie multiplicity ranges from `3.027261` to
  `6.518937`; maximum tie multiplicity ranges from `27` to `93`.
- The diagnostic separates the discrete root law status from a smooth margin
  law. A large tie burden means the root event is a discrete selected-region
  and tie-breaking problem.
- Tie burden is not a direct explanation for the largest root selected ratios.
  On this seven-case slice, `root_tie_step_fraction` has Spearman `-0.964286`
  with the log root sibling selected ratio, and mean log tie multiplicity has
  Spearman `-0.857143`. The larger selected ratios still align more with the
  edge/spectral action variables recorded in the root-margin diagnostic.
- The rank-aware replay adds deterministic tie-breaking coordinates:
  `selected_tie_rank_lexicographic`, `selected_tie_rank_fraction`, and
  `selected_tie_log_rank`. Median selected tie-rank fraction has Spearman
  `0.928571` with the log root sibling selected ratio, and mean selected
  tie-rank fraction has Spearman `0.785714`. This suggests the selected
  position inside the tied set is more relevant than raw tie multiplicity.
- Therefore the next mathematical object is not a monotone penalty for more
  ties. It is a conditional discrete selected-region law that includes
  tie-cell membership, deterministic tie-breaking, and the edge/spectral root
  statistic.

## Evidence

- `root_selected_tie_cell_burden.py` reads the root selected-region summary and
  merge-margin tables, computes case-level tie burden, and summarizes
  relationships to the root sibling selected ratio.
- The validation test covers log-multiplicity burden, selected tie-rank
  burden, side asymmetry, relationship rows, and output writing.
- The output manifest records the overlap root selected-region inputs and the
  generated burden/relationship outputs.

## Links

- [[root-selected-region-overlap-case-family-20260616]]
- [[selected-neighborhood-topology-frontier-diagnostic-20260616]]
- [[selected-neighborhood-measurability-law]]
