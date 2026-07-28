---
title: Root Tie Rank Proposal Gap Panel 2026-06-16
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/tie_rank/root_tie_rank_proposal_gap_panel.py
  - tests/validation/calibration/root/tie_rank/159_test_root_tie_rank_proposal_gap_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_proposal_gap_panel_two_case_smoke
tags:
  - source
  - diagnostics
  - root
  - calibration
  - proposals
---

# Root Tie Rank Proposal Gap Panel 2026-06-16

## Summary

`root_tie_rank_proposal_gap_panel.py` reads the root tie-rank proposal
frontier feasibility table and compares every observed target root stratum
against the best generated row from each proposal family. It reports which
coordinate fails: tie-rank band, edge-margin band, spectral-ratio band,
bandwidth-reopen band, or selected-ratio tail.

## Key Points

- The two-case smoke writes `35` target-by-family best-gap rows and `5`
  proposal-family summary rows.
- No proposal family has an exact observed target-stratum hit.
- `two_block_tilt_proposal` and `coupled_edge_spectral_proposal` exceed every
  target root selected ratio and match edge-margin bands for `4/7` targets,
  but both are summarized as `separable_action_and_spectral_no_joint_match`
  because their best rows lack the required spectral-ratio bands.
- `iid_marginal_bernoulli`, `column_beta_bernoulli`, and
  `sparse_block_spike_proposal` are summarized as
  `spectral_without_edge_action`: they can match spectral bands for `4/7`,
  `4/7`, and `5/7` targets respectively, but do not match the high edge/action
  event.
- Every best-gap row has `generated_bandwidth_unmeasured`, because the
  proposal frontier does not yet replay the topology-frontier bandwidth
  diagnostic for generated roots.
- For `overlap_mod_4c_small` and `overlap_mod_6c_med`, the best dense/coupled
  rows have `action_edge_without_spectral`; the best sparse rows have
  `spectral_without_edge_action`.
- This makes the next mathematical target explicit: the missing root law is a
  selected spectral-action coupling, not an additive dense-action plus
  sparse-spike mechanism.

## Evidence

- `root_tie_rank_proposal_gap_panel.py` implements target-vs-proposal gap
  scoring, coordinate-pattern labels, family summaries, and manifest output.
- `159_test_root_tie_rank_proposal_gap_panel.py` verifies exact-hit,
  action-without-spectral, spectral-without-action, summary, and output
  behavior.
- The smoke manifest records `rows = 35` and `summary = 5`.

## Links

- [[root-tie-rank-null-proposal-frontier-20260616]]
- [[root-tie-rank-selected-null-simulation-pilot-20260616]]
- [[root-tie-rank-calibration-feasibility-20260616]]
- [[selected-neighborhood-measurability-law]]
