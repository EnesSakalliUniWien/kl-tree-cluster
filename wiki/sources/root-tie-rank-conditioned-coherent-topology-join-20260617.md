---
title: Root Tie Rank Conditioned Coherent Topology Join 2026-06-17
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/tie_rank/root_tie_rank_conditioned_coherent_topology_join.py
  - benchmarks/diagnostics/calibration/root/tie_rank/root_tie_rank_generated_neighborhood_replay.py
  - benchmarks/diagnostics/calibration/root/tie_rank/root_tie_rank_selected_spectral_generator_target_panel.py
  - tests/validation/calibration/root/tie_rank/168_test_root_tie_rank_conditioned_coherent_topology_join.py
  - tests/validation/calibration/root/tie_rank/166_test_root_tie_rank_selected_spectral_generator_target_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_generated_neighborhood_replay_conditioned_coherent_capped_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_conditioned_coherent_topology_join_after_replay
tags:
  - source
  - diagnostics
  - root
  - spectral
  - topology
---

# Root Tie Rank Conditioned Coherent Topology Join 2026-06-17

## Summary

`root_tie_rank_generated_neighborhood_replay.py` was run on the capped
`conditioned_coherent_rank_one_spike_proposal` matrices, then
`root_tie_rank_conditioned_coherent_topology_join.py` joined the replayed
root-frontier evidence back into the selected spectral generator target panel.
The join also preserves the matched-target contract:
`conditioning_target_case_id` rows can support only their matched observed
target.

The topology replay completes for all seven conditioned coherent matrices and
produces measured root-frontier rows, but none reopens at the reference
bandwidth. After the join, the conditioned coherent family remains
diagnostic-only and reaches the same two easier targets as before. The five
hard roots still require spectral lift and external selected-null support.

## Key Points

- The generated-neighborhood replay writes `7` run rows and `8,393` rows each
  for node decisions, selected-neighborhood distribution, p-value
  interpolation, measurability, and topology frontier.
- All seven conditioned coherent replay rows have `run_status = ok`.
- The joined feasibility table contains `24` rows: the previous generated
  replay feasibility rows plus seven conditioned coherent rows.
- Every conditioned coherent row has `root_frontier_row_count = 1`,
  `root_bandwidth_reopen_count = 0`, `root_bandwidth_reopen_band =
  bandwidth_no_root_reopen`, and
  `root_bandwidth_locality_status = bandwidth_does_not_reopen_root_rows`.
- The joined spectral target panel writes `42` target rows and `6` family
  summaries. With matched-target filtering, each observed target sees exactly
  one conditioned coherent generated row.
- `conditioned_coherent_rank_one_spike_proposal` covers `7/7` high
  action-edge/tie measured targets and reaches `2/7` spectral targets, with
  median required spectral-lift multiplier `1.805514` and maximum `2.952730`.
- The coupled and two-block families remain at `0/7` spectral reach after the
  join, with median required lift `2.493244` and `2.595615` respectively.
- The result closes the requested topology replay for the new coherent family,
  but it does not close the selected-root spectral tail law.

## Evidence

- `root_tie_rank_conditioned_coherent_topology_join.py` attaches root-frontier
  counts, root bandwidth reopen counts, bandwidth bands, and locality status
  from topology replay rows, appends the conditioned coherent rows to the
  existing feasibility table, and reruns the spectral generator target panel.
- `root_tie_rank_selected_spectral_generator_target_panel.py` now enforces
  `conditioning_target_case_id` when present, so target-conditioned generators
  cannot be borrowed across observed targets.
- `168_test_root_tie_rank_conditioned_coherent_topology_join.py` validates
  missing-root-frontier handling, root-reopen eligibility, and runner outputs.
- `166_test_root_tie_rank_selected_spectral_generator_target_panel.py`
  validates matched-target filtering for conditioned generator rows.

## Links

- [[root-tie-rank-spectral-lift-parameter-sweep-20260616]]
- [[root-tie-rank-generated-neighborhood-replay-20260616]]
- [[root-tie-rank-selected-spectral-generator-targets-20260616]]
- [[selected-neighborhood-measurability-law]]
