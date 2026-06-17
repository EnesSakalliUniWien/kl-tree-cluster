---
title: Root Tie Rank Measured Coupling Residual Panel 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/root_tie_rank_measured_coupling_residual_panel.py
  - tests/validation/164_test_root_tie_rank_measured_coupling_residual_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_measured_coupling_residual_after_generated_replay/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_measured_coupling_residual_after_generated_replay/root_tie_rank_measured_coupling_residual_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_measured_coupling_residual_after_generated_replay/root_tie_rank_measured_coupling_residual_summary.csv
tags:
  - source
  - diagnostics
  - root
  - coupling
  - spectral
---

# Root Tie Rank Measured Coupling Residual Panel 2026-06-16

## Summary

`root_tie_rank_measured_coupling_residual_panel.py` converts the
post-generated-replay coupling table into a target-level residual diagnosis.
For each observed overlap root, it selects the best measured-neighborhood
generated proposal row and decomposes the remaining gap into tie-rank,
action-edge bottleneck, spectral-excess, and bandwidth-conditioning residuals.

The result is unambiguous on the two-case proposal frontier: the coupled
edge-spectral proposal is the best measured proposal for all seven observed
targets. It already satisfies the action-edge bottleneck for the unresolved
targets. The remaining hard and partial targets are blocked by spectral-excess
residuals, not bandwidth coverage or action-edge mass.

## Key Points

- The panel writes `7` target-level residual rows and `3` summary rows.
- For all seven observed targets, the best measured proposal family is
  `coupled_edge_spectral_proposal`.
- Two easier targets,
  `overlap_extreme_4c` and `overlap_heavy_4c_small_feat`, are reached by
  measured-neighborhood coupling. Both are diagnostic proposal hits, not
  calibration support.
- Four targets are hard measured-coupling residuals with spectral-excess as the
  dominant missing axis: `overlap_mod_4c_small`, `overlap_mod_6c_med`,
  `overlap_part_4c_small`, and `overlap_unbal_6c_med`.
- One target, `overlap_unbal_4c_small`, is a partial measured-coupling residual
  with best measured-neighborhood coupling ratio `0.671395`; its dominant
  residual axis is also spectral excess.
- The action-edge bottleneck relative deficit is `0.0` in every summary row.
  Therefore increasing selected action or edge margin alone is not the next
  mathematical move for the hard roots.
- The recommended next mathematical step for all unresolved targets is
  `derive_selected_spectral_excess_given_high_action_edge_tie_rank`.

## Evidence

- `root_tie_rank_measured_coupling_residual_panel.py` implements target-level
  best proposal selection, normalized residual axes, resolution status labels,
  and next mathematical step labels.
- `164_test_root_tie_rank_measured_coupling_residual_panel.py` validates
  diagnostic reach, spectral residual localization, action-edge residual
  localization, summary labels, and output writing.
- The output manifest records `rows = 7` and `summary = 3`.

## Links

- [[root-tie-rank-generated-neighborhood-replay-20260616]]
- [[root-tie-rank-coupling-equation-panel-20260616]]
- [[root-tie-rank-proposal-gap-panel-20260616]]
- [[selected-neighborhood-measurability-law]]
