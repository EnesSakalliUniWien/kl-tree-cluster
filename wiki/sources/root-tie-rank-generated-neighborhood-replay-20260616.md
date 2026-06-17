---
title: Root Tie Rank Generated Neighborhood Replay 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/root_tie_rank_generated_neighborhood_replay.py
  - benchmarks/diagnostics/calibration/root_tie_rank_null_proposal_frontier.py
  - benchmarks/diagnostics/calibration/root_selected_mixed_region_law.py
  - benchmarks/diagnostics/calibration/root_tie_rank_neighborhood_join_audit.py
  - benchmarks/diagnostics/calibration/root_tie_rank_coupling_equation_panel.py
  - tests/validation/163_test_root_tie_rank_generated_neighborhood_replay.py
  - tests/validation/158_test_root_tie_rank_null_proposal_frontier.py
  - tests/validation/155_test_root_selected_mixed_region_law.py
  - tests/validation/162_test_root_tie_rank_neighborhood_join_audit.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_generated_neighborhood_replay_all_generated_smoke/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_generated_neighborhood_replay_all_generated_smoke/root_tie_rank_generated_neighborhood_replay_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_null_proposal_frontier_with_generated_replay/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_null_proposal_frontier_with_generated_replay/root_tie_rank_null_proposal_combined_feasibility_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_coupling_equation_after_generated_replay/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_coupling_equation_after_generated_replay/root_tie_rank_coupling_equation_summary.csv
tags:
  - source
  - diagnostics
  - root
  - bandwidth
  - replay
---

# Root Tie Rank Generated Neighborhood Replay 2026-06-16

## Summary

`root_tie_rank_generated_neighborhood_replay.py` closes the operational gap
identified by the root tie-rank neighborhood join audit. It replays generated
proposal matrices through the selected-family traversal, selected-neighborhood
distribution, hold-out p-value interpolation, measurability-law, and
topology-frontier builders. The replay shows that generated proposal bandwidth
evidence was missing because the topology-frontier replay had not been run, not
because generated matrices were absent.

After replay, the proposal frontier can be rebuilt with generated
topology-frontier rows joined before feasibility. The resulting coupling panel
has measured-neighborhood evidence for every generated family. This changes the
diagnostic conclusion from "bandwidth unmeasured" to "measured neighborhood
exists, but hard overlap roots still require stronger selected
spectral-action/tie-rank coupling and a calibrated discrete root law."

## Key Points

- The all-generated replay writes `10` run rows and `9,990` rows each for node
  decisions, selected-neighborhood distribution, p-value interpolation,
  measurability, and topology frontier.
- All `10` generated proposal matrices replay successfully under
  `fixed_coordinate_conditional_topology_diagnostic_v1`.
- The topology-frontier replay contains `4,990` direct-measurable rows and
  `5,000` nonroot non-direct rows. After the root aggregation was refined to
  count direct-measurable depth-0 rows, every generated proposal case has one
  measured root-frontier row.
- Joining generated topology-frontier rows into the proposal frontier produces
  measured root bandwidth bands for all generated cases. Three generated roots
  reopen at reference bandwidth: the coupled proposal for
  `overlap_mod_4c_small`, and the two-block proposal for both replayed base
  cases. The other seven generated roots are measured no-reopen rows.
- The rebuilt proposal frontier still has no observed target-stratum hits. Iid
  rows remain the only calibration-candidate support, while the richer proposal
  families remain diagnostic-only.
- The updated coupling panel has
  `generated_neighborhood_measured_count = 7/7` for every proposal family.
  Measured-neighborhood coupling dominance is `1/7` for column-beta, `2/7` for
  coupled edge-spectral, `0/7` for iid, `2/7` for sparse block spike, and `2/7`
  for two-block tilt.
- The hard roots remain unresolved. For `overlap_mod_4c_small` and
  `overlap_mod_6c_med`, the best coupled/two-block generated rows still have
  measured-neighborhood coupling ratios below one, so the missing object is not
  just bandwidth replay. It is the selected spectral-action/tie-rank law plus
  external support for the discrete root stratum.

## Evidence

- `root_tie_rank_generated_neighborhood_replay.py` implements generated matrix
  replay and writes run, node, distribution, interpolation, measurability, and
  topology-frontier tables.
- `root_tie_rank_null_proposal_frontier.py` now accepts optional
  topology-frontier rows when building generated proposal mixed-law rows.
- `root_selected_mixed_region_law.py` and
  `root_tie_rank_neighborhood_join_audit.py` now count direct-measurable root
  rows as root-frontier evidence when `parent_id` is empty or depth is zero.
- `163_test_root_tie_rank_generated_neighborhood_replay.py` validates generated
  row filtering, replay table composition, topology-frontier output, and runner
  artifacts.
- Focused validation ran with `17` tests passing across the proposal frontier,
  mixed root law, neighborhood join audit, and generated replay modules.

## Links

- [[root-tie-rank-neighborhood-join-audit-20260616]]
- [[root-tie-rank-coupling-equation-panel-20260616]]
- [[root-tie-rank-null-proposal-frontier-20260616]]
- [[selected-neighborhood-measurability-law]]
