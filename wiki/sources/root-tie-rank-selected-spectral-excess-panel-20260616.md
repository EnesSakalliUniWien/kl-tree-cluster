---
title: Root Tie Rank Selected Spectral Excess Panel 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/root_tie_rank_selected_spectral_excess_panel.py
  - tests/validation/165_test_root_tie_rank_selected_spectral_excess_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_selected_spectral_excess_after_generated_replay
tags:
  - source
  - diagnostics
  - root
  - spectral
  - coupling
---

# Root Tie Rank Selected Spectral Excess Panel 2026-06-16

## Summary

`root_tie_rank_selected_spectral_excess_panel.py` tests the narrowed law left
by the measured coupling residual panel. It conditions generated roots on
measured neighborhood evidence plus high tie-weighted action-edge bottleneck,
then asks whether the remaining selected spectral excess reaches each observed
root.

The result shows that the measured coupling product can be reached without
matching the target spectral excess itself. Under the current two-case proposal
frontier, every observed target has four diagnostic eligible generated roots,
zero selected-null calibration support rows, and the same best spectral row:
the `overlap_mod_6c_med__coupled_edge_spectral_proposal_r0000` root. No target
has spectral-excess reach under the high-action-edge/tie conditioning set.

## Key Points

- The panel writes `7` target-level rows and `2` summary rows.
- Every target has `eligible_generated_count = 4` under measured neighborhood,
  high action-edge, and tie-fraction floor `0.70`.
- Every target has `eligible_calibration_support_count = 0`, so the high
  action-edge/tie measured stratum has no selected-null support in this
  artifact.
- The best spectral row for every target is
  `overlap_mod_6c_med__coupled_edge_spectral_proposal_r0000`, with selected
  spectral excess log `0.424898`.
- Two easier roots are partial spectral residuals, not spectral reaches:
  `overlap_extreme_4c` and `overlap_heavy_4c_small_feat`. Their required
  spectral-excess multipliers are about `1.228` and `1.015`.
- Five roots are hard spectral residuals:
  `overlap_mod_4c_small`, `overlap_mod_6c_med`, `overlap_part_4c_small`,
  `overlap_unbal_4c_small`, and `overlap_unbal_6c_med`. Their median spectral
  log-ratio is `0.305111`, with median required spectral-excess multiplier
  `2.631889`.
- The next mathematical step is no longer a generic coupling rule. It is an
  external selected spectral-excess law for high action-edge/tie measured
  roots, or a generator that produces selected-null roots in that stratum.

## Evidence

- `root_tie_rank_selected_spectral_excess_panel.py` implements target-level
  conditioning on measured bandwidth, tie-weighted action-edge bottleneck, and
  spectral-excess reach.
- `165_test_root_tie_rank_selected_spectral_excess_panel.py` validates
  diagnostic spectral reach, selected-null tail counting, hard residuals,
  missing conditioning support, summaries, and runner output.
- The manifest records `rows = 7` and `summary = 2`.

## Links

- [[root-tie-rank-measured-coupling-residual-panel-20260616]]
- [[root-tie-rank-generated-neighborhood-replay-20260616]]
- [[root-tie-rank-coupling-equation-panel-20260616]]
- [[selected-neighborhood-measurability-law]]
