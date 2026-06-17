---
title: Root Tie Rank Neighborhood Join Audit 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/root_tie_rank_neighborhood_join_audit.py
  - tests/validation/162_test_root_tie_rank_neighborhood_join_audit.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_neighborhood_join_audit_two_case_smoke/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_neighborhood_join_audit_two_case_smoke/root_tie_rank_neighborhood_join_audit_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_neighborhood_join_audit_two_case_smoke/root_tie_rank_neighborhood_join_audit_summary.csv
tags:
  - source
  - diagnostics
  - root
  - bandwidth
  - proposals
---

# Root Tie Rank Neighborhood Join Audit 2026-06-16

## Summary

`root_tie_rank_neighborhood_join_audit.py` audits why generated root proposal
rows have `bandwidth_reopen_missing` after the coupling-equation panel. It
joins root tie-rank feasibility rows, generated mixed-law rows, available
selected-neighborhood topology-frontier rows, and generated proposal matrix
paths. The output identifies whether bandwidth is already measured, topology
frontier rows exist but were not joined, generated matrices exist but still
need selected-neighborhood replay, or proposal matrices are missing.

## Key Points

- The two-case smoke writes `17` audit rows and `6` proposal-family summaries.
- All `7` observed target roots have measured bandwidth status and matching
  observed root topology-frontier rows.
- All `10` generated proposal roots have `bandwidth_reopen_missing` because
  their mixed-law rows were built with `topology_frontier_rows=None`.
- All `10` generated proposal matrices exist under the proposal frontier output
  directory, so the missing evidence is not matrix regeneration.
- Every generated proposal family is summarized as
  `generated_topology_frontier_replay_needed`.
- The row-level next action for every generated proposal root is
  `run_generated_measurability_and_topology_frontier_replay`.

## Evidence

- `root_tie_rank_neighborhood_join_audit.py` implements row-level join status,
  generated matrix checks, proposal-family summaries, and manifest output.
- `162_test_root_tie_rank_neighborhood_join_audit.py` verifies measured
  bandwidth, generated-matrix replay-needed, topology-frontier join-needed,
  missing-matrix, summary, and output-writing behavior.
- The smoke manifest records `rows = 17` and `summary = 6`.

## Links

- [[root-tie-rank-coupling-equation-panel-20260616]]
- [[root-tie-rank-null-proposal-frontier-20260616]]
- [[selected-neighborhood-topology-frontier-diagnostic-20260616]]
- [[selected-neighborhood-measurability-law]]
