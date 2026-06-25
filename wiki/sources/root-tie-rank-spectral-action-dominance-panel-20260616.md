---
title: Root Tie Rank Spectral Action Dominance Panel 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/root_tie_rank_spectral_action_dominance_panel.py
  - tests/validation/160_test_root_tie_rank_spectral_action_dominance_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_spectral_action_dominance_panel_two_case_smoke
tags:
  - source
  - diagnostics
  - root
  - spectral
  - proposals
---

# Root Tie Rank Spectral Action Dominance Panel 2026-06-16

## Summary

`root_tie_rank_spectral_action_dominance_panel.py` strengthens the proposal
gap analysis from coarse bands to continuous coordinates. For every observed
target root and proposal family, it asks whether the best generated row
dominates the target in selected-ratio action, edge margin, spectral ratio,
and tie-rank fraction.

## Key Points

- The two-case smoke writes `35` target-by-family dominance rows and `5`
  proposal-family summaries.
- No proposal family has full continuous dominance over any observed target.
- No proposal family has spectral-action dominance over any observed target.
- `two_block_tilt_proposal` and `coupled_edge_spectral_proposal` dominate
  selected-ratio action and edge margin for `7/7` targets, but have
  `0/7` spectral dominance; both are summarized as
  `action_edge_dominance_spectral_deficit`.
- `iid_marginal_bernoulli`, `column_beta_bernoulli`, and
  `sparse_block_spike_proposal` have spectral dominance for `2/7` targets but
  no selected-ratio or edge dominance; they are summarized as
  `spectral_dominance_action_edge_deficit`.
- Every best row has unmeasured generated bandwidth, so bandwidth still needs
  a generated-root topology-frontier replay before it can be part of a
  calibrated conditioning stratum.
- For `overlap_mod_4c_small`, the best coupled proposal has zero selected-ratio
  and edge deficit but spectral log deficit about `0.913584`. For
  `overlap_mod_6c_med`, the best coupled proposal has spectral log deficit
  about `1.282088`.
- This confirms that the observed root event is not reproduced by independent
  action and spectral perturbations. The missing object is a selected
  spectral-action coupling.

## Evidence

- `root_tie_rank_spectral_action_dominance_panel.py` implements continuous
  dominance deficits, family summaries, and manifest output.
- `160_test_root_tie_rank_spectral_action_dominance_panel.py` verifies full
  dominance, action-edge spectral deficit, spectral action-edge deficit,
  summary, and output behavior.
- The smoke manifest records `rows = 35` and `summary = 5`.

## Links

- [[root-tie-rank-proposal-gap-panel-20260616]]
- [[root-tie-rank-null-proposal-frontier-20260616]]
- [[root-tie-rank-calibration-feasibility-20260616]]
- [[selected-neighborhood-measurability-law]]
