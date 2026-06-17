---
title: Root Tie Rank Selected Spectral Generator Targets 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/root_tie_rank_selected_spectral_generator_target_panel.py
  - tests/validation/166_test_root_tie_rank_selected_spectral_generator_target_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_selected_spectral_generator_targets_after_generated_replay/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_selected_spectral_generator_targets_after_generated_replay/root_tie_rank_selected_spectral_generator_target_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_selected_spectral_generator_targets_after_generated_replay/root_tie_rank_selected_spectral_generator_target_summary.csv
tags:
  - source
  - diagnostics
  - root
  - spectral
  - generator
---

# Root Tie Rank Selected Spectral Generator Targets 2026-06-16

## Summary

`root_tie_rank_selected_spectral_generator_target_panel.py` converts the
selected spectral-excess residual into proposal-family generator targets. For
each observed root and proposal family, it checks whether the family produces a
measured high action-edge/tie root, whether that root reaches the observed
selected spectral excess, and how much spectral lift is still required.

The generated-neighborhood replay result writes `35` target-by-family rows and
`5` family summaries. Only the `coupled_edge_spectral_proposal` and
`two_block_tilt_proposal` families cover all seven observed targets in the
measured high action-edge/tie stratum, but both remain diagnostic-only and
need spectral lift for every target. No selected-null candidate family covers
the conditioning stratum.

## Key Points

- The panel is diagnostic-only and does not turn spectral lift into a p-value
  or traversal rescue rule.
- `coupled_edge_spectral_proposal` covers `7/7` targets in the conditioning
  stratum, with zero calibration-supported targets and zero spectral reaches.
  Its median required spectral-lift multiplier is `2.493244`; its maximum is
  `4.296772`.
- `two_block_tilt_proposal` also covers `7/7` targets, with zero
  calibration-supported targets and zero spectral reaches. Its median required
  spectral-lift multiplier is `2.595615`; its maximum is `4.473194`.
- `iid_marginal_bernoulli`, `column_beta_bernoulli`, and
  `sparse_block_spike_proposal` cover `0/7` targets in the measured high
  action-edge/tie stratum.
- The selected-null calibration support target remains unchanged: `99`
  selected-null roots per stratum for alpha-resolution only, or `1584` per
  stratum for the stated tail-precision target. Across seven observed strata,
  that is `693` and `11088` roots respectively.

## Evidence

- `root_tie_rank_selected_spectral_generator_target_panel.py` implements the
  target-by-family conditioning, support-role separation, spectral lift, and
  family summaries.
- `166_test_root_tie_rank_selected_spectral_generator_target_panel.py`
  validates required spectral lift, calibration reach, missing conditioning
  strata, summary totals, and runner output.
- The manifest records `rows = 35` and `summary = 5` for the generated-replay
  input artifact.

## Links

- [[root-tie-rank-selected-spectral-excess-panel-20260616]]
- [[root-tie-rank-measured-coupling-residual-panel-20260616]]
- [[root-tie-rank-generated-neighborhood-replay-20260616]]
- [[selected-neighborhood-measurability-law]]
