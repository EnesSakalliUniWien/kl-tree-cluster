---
title: Selected Neighborhood Topology Frontier Diagnostic 2026-06-16
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/selected/neighborhood/selected_neighborhood_topology_frontier.py
  - tests/validation/calibration/selected/neighborhood/153_test_selected_neighborhood_topology_frontier.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_region_margins_overlap_case_family
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_topology_frontier_overlap_expanded_candidates
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_refined_candidate_audit_smoke/topology_frontier
tags:
  - source
  - diagnostics
  - topology
  - neighborhood
  - bandwidth
---

# Selected Neighborhood Topology Frontier Diagnostic 2026-06-16

## Summary

`selected_neighborhood_topology_frontier.py` evaluates the next proposed
selected-neighborhood law on the same candidate rows as the measurability
audit. It compares the current direct sibling decision, the older bandwidth
interpolation proxy at reference `tau_s = 20`, root structural outgoing
balance, non-root structural balance product, and strict spectral-support
availability. It can also join case-level root selected-region margin evidence
from `root_selected_region_summary.csv`. The output is diagnostic-only.

## Key Points

- The full expanded overlap run writes `69,860` row annotations, `4` summary
  rows, and `56` threshold-sweep rows under
  `selected_neighborhood_topology_frontier_overlap_expanded_candidates`.
- The corrected bandwidth column separates broad tau-significant rows from
  direct-positive reopen rows. At `tau_s = 20`, direct-positive reopen counts
  are `579` selected-null rows and `388` signal rows for each compared method
  profile, matching the earlier bandwidth tradeoff conclusion.
- The non-direct row set remains small: `21` selected-null root rows for the
  conditional-topology profile, `50` selected-null rows for the refined global
  profile, `10` signal root rows for the conditional-topology profile, and
  `23` signal rows for the refined global profile.
- Root structural outgoing balance is not a rescue law. With a root outgoing
  balance floor of `0.30`, `19/21` selected-null root rows pass in each method
  profile, while only `6/10` and `6/11` signal root rows pass. With floor
  `0.40`, `12/21` selected-null rows still pass, versus only `2/10` and
  `2/11` signal rows.
- After joining the seven-case root selected-region margin table, all root
  non-direct rows have `root_selected_law_status =
  discrete_tie_cell_geometry_required`: `21` selected-null root rows in each
  method profile, `10` signal root rows in the conditional-topology profile,
  and `11` signal root rows in the refined global profile. The root blocker is
  therefore no longer just missing evidence; it is a discrete tie-cell
  selected-region law that has not been derived.
- Lowering the non-root balance-product floor is also unsafe as a direct
  recovery rule on this panel. At floor `0.12`, the refined global profile
  passes `15/29` selected-null non-root rows but only `3/12` signal non-root
  rows. At floor `0.20`, it passes `2/29` selected-null rows and `1/12`
  signal rows. At the current floor `0.22`, no non-root row passes.
- Strict hybrid support remains zero on the full expanded overlap panel. The
  blocking statuses are effective-support weakness, discrete tie-cell root
  selected-region geometry, and non-root topology below the current floor. No
  non-direct row has enough measured topology plus strict spectral support to
  be promoted.
- The real-row smoke on `overlap_unbal_4c_small` signal replicate `0` writes
  `399` rows. It has `2` non-direct rows: one root row and one non-root row.
  The root row passes only below a `0.25` outgoing-balance floor and is labeled
  `discrete_tie_cell_geometry_required`; the non-root row is near low
  balance-product thresholds but fails the current `0.22` floor. Strict hybrid
  support is `0`.

## Evidence

- `benchmarks/diagnostics/calibration/selected/neighborhood/selected_neighborhood_topology_frontier.py`
  implements row annotations, summary tables, threshold sweeps, and manifests.
- `tests/validation/calibration/selected/neighborhood/153_test_selected_neighborhood_topology_frontier.py`
  checks root proxy separation from admissible root law, non-root frontier
  behavior, bandwidth direct-positive reopening, and output writing.
- The full overlap and smoke output manifests record the input measurability
  tables, joined root selected-region summary, thresholds, strict spectral
  requirement, and output paths.

## Links

- [[selected-neighborhood-measurability-law]]
- [[selected-neighborhood-bottleneck-law]]
- [[selected-neighborhood-measurability-law-diagnostic-20260616]]
- [[selected-neighborhood-pvalue-interpolation-comparison-20260616]]
- [[root-selected-region-overlap-case-family-20260616]]
