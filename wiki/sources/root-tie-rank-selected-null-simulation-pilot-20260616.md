---
title: Root Tie Rank Selected Null Simulation Pilot 2026-06-16
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/tie_rank/root_tie_rank_selected_null_simulation_pilot.py
  - tests/validation/calibration/root/tie_rank/157_test_root_tie_rank_selected_null_simulation_pilot.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_selected_null_simulation_pilot_overlap_case_family
tags:
  - source
  - diagnostics
  - root
  - calibration
  - selected-null
---

# Root Tie Rank Selected Null Simulation Pilot 2026-06-16

## Summary

`root_tie_rank_selected_null_simulation_pilot.py` is the first executable
selected-null generator for the discrete root tie-rank calibration path. It
builds iid Bernoulli null matrices matched to each binary overlap case's
sample count, feature count, and expected sparse-template marginal feature
rate, sends those matrices through the existing root selected-region replay,
and recomputes target-stratum support against the observed mixed root-law
rows.

## Key Points

- The one-replicate overlap pilot attempted `7` selected-null roots and
  succeeded on all `7`; there were `0` failures.
- The run writes `7` root rows, `3493` merge-margin rows, `7` tie rows, `7`
  selected-null mixed rows, `14` combined feasibility rows, `10` combined
  feasibility strata, and `7` target-support rows.
- The generated selected-null roots occupy `3` conditioning strata, but none
  of those strata match the seven observed target strata.
- Therefore `target_strata_with_null_support_count = 0`,
  `target_strata_alpha_resolution_ready_count = 0`, and
  `target_strata_tail_precision_ready_count = 0`.
- The observed target strata still require `693` selected-null roots for
  alpha-resolution only and `11088` for the `0.25` relative tail precision
  target.
- The combined feasibility table now has `7` selected-null calibration-support
  rows, but only in null-generated strata. Its status is
  `calibration_null_support_observed_below_alpha_resolution`, while the target
  support status remains `no_observed_target_stratum_support_yet`.
- The generated null root selected ratios are much smaller than the observed
  overlap roots: they range from about `0.075806` to `11.038714`, while the
  observed target roots range from `47.831161` to `1768.512366`.

## Evidence

- The implementation uses `preloaded` generated null matrices so the canonical
  TBS context builder and root selected-region extractor remain unchanged.
- The unit test validates the Bernoulli null marginal-rate calculation,
  target-stratum support counting, simulation-summary statuses, and output
  writing.
- The output manifest records `elapsed_seconds` about `62.20`, the seven
  successful root rows, and all generated CSV paths.

## Links

- [[root-tie-rank-calibration-feasibility-20260616]]
- [[root-selected-mixed-region-law-20260616]]
- [[selected-neighborhood-measurability-law]]
