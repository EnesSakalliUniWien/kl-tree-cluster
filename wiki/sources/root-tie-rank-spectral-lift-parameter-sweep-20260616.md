---
title: Root Tie Rank Spectral Lift Parameter Sweep 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/root_tie_rank_spectral_lift_parameter_sweep.py
  - tests/validation/167_test_root_tie_rank_spectral_lift_parameter_sweep.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_spectral_lift_parameter_sweep_overlap_mod6_highlift_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_spectral_lift_parameter_sweep_overlap_mod6_moderate_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_spectral_lift_parameter_sweep_overlap_mod6_coherent_grid_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_spectral_lift_parameter_sweep_overlap_mod6_conditioned_coherent_capped_smoke
tags:
  - source
  - diagnostics
  - root
  - spectral
  - generator
---

# Root Tie Rank Spectral Lift Parameter Sweep 2026-06-16

## Summary

`root_tie_rank_spectral_lift_parameter_sweep.py` tests whether the existing
coupled dense-plus-spike generator knobs can produce the selected spectral
excess needed by the observed overlap roots while preserving high
action-edge/tie root metrics. The sweep is a pre-replay screen: passing rows
would still need generated-neighborhood replay before entering measured
bandwidth calibration.

Two one-setting smokes were run on `overlap_mod_6c_med`, the base case that
previous panels selected as the best generated spectral row. Both settings
cover all seven observed targets in action-edge/tie root metrics, but neither
reaches any observed target's spectral excess. Increasing both dense and sparse
amplitudes does not monotonically improve spectral excess; the stronger setting
has higher action/edge but lower selected spectral-excess log.

The diagnostic was then extended with
`coherent_rank_one_spike_proposal`, a rank-one binary population spike aligned
across active features. A six-setting `overlap_mod_6c_med` smoke shows that
coherent MP-mode construction improves the root spectral screen relative to
the coupled dense-plus-spike proposal: the best coherent setting reaches `2/7`
targets and reduces the median required spectral-lift multiplier to `1.739915`.
It still leaves five hard roots unresolved, so it is a candidate direction for
generated-neighborhood replay and calibrated selected-null design, not a
production rescue rule.

The diagnostic now also includes
`conditioned_coherent_rank_one_spike_proposal`, which maps each target's
selected tie-rank/action-edge geometry into a concentrated coherent spike and
then permits the generated row to support only its matched target. The capped
conditioning map improves over the uncapped amplitude ramp but does not beat
the best unconditional coherent grid point: it reaches `2/7` targets, with
median residual spectral lift `1.805514` and maximum `2.952730`.

## Key Points

- The high-lift setting
  `coupled_edge_spectral_proposal__td0_550__sf0_350__sd0_850` generates root
  selected ratio `46300.244363`, tie fraction `1.0`, and edge margin
  `92586.673035`, but selected eigenvalue over MP upper bound only
  `1.277092`. Its best generated spectral-excess log is `0.244586`; it reaches
  `0/7` targets and needs median spectral-lift multiplier `2.985886`, maximum
  `5.145775`.
- The moderate setting
  `coupled_edge_spectral_proposal__td0_450__sf0_200__sd0_650` generates root
  selected ratio `38542.151427`, tie fraction `0.75`, and edge margin
  `77070.487187`, with selected eigenvalue over MP upper bound `1.402801`.
  Its best generated spectral-excess log is `0.338471`; it reaches `0/7`
  targets and needs median spectral-lift multiplier `2.718312`, maximum
  `4.684647`.
- The stronger setting is worse spectrally than the moderate setting despite
  larger action and edge coordinates. Thus simply increasing perturbation
  amplitude appears to saturate action/edge without creating the needed
  MP-supported spectral mode.
- The coherent rank-one spike grid
  `coherent_rank_one_spike_proposal__sf{0.200,0.350,0.500}__sd{0.650,0.850}`
  produces six successful generated roots and no failures. All six settings
  cover all seven targets in action-edge/tie conditioning. Five settings reach
  `2/7` targets, and the strongest half-feature setting reaches `1/7`.
- The best coherent setting is
  `coherent_rank_one_spike_proposal__sf0_200__sd0_650`. It has selected
  eigenvalue over MP upper bound `2.191630`, best generated spectral-excess log
  `0.784646`, median required spectral-lift multiplier `1.739915`, and maximum
  `2.998511`.
- Compared with the best coupled smoke, the coherent setting improves target
  reach from `0/7` to `2/7` and lowers median residual lift from `2.718312` to
  `1.739915`. This supports the literature-derived MP-spike assumption that
  mode coherence matters more than raw perturbation amplitude.
- The conditioned coherent setting family derives
  `proposal_spike_feature_fraction = 0.50 - 0.30 g` and
  `proposal_spike_delta = 0.45 + 0.20 g`, with \(g\) the geometric mean of the
  target's normalized action-edge bottleneck and selected tie-rank fraction.
  The target spectral excess is not used in this map.
- The capped conditioned coherent smoke writes seven matched target settings,
  seven generated rows, seven target rows, and no failures. It reaches the same
  two easier targets as the unconditional coherent grid. The hard-target
  residuals remain: median required spectral lift `1.805514`, maximum
  `2.952730`.
- Conditioning is therefore useful as a geometry localization and no-borrowing
  audit, but it is not yet the missing law. The remaining object must add
  selected-neighborhood/topology replay or a sharper selected-root spectral
  tail, not just target-conditioned spike concentration.
- A larger 30-setting sweep was started and then intentionally stopped because
  root-margin replay was too slow for this iteration. Its partial generated
  matrix directory was removed; only completed smoke outputs are cited.

## Evidence

- `root_tie_rank_spectral_lift_parameter_sweep.py` implements setting-grid
  generation, mixed-law root rows, target-by-setting root-metric screening,
  setting summaries, the diagnostic coherent rank-one spike proposal, and the
  matched-target conditioned coherent spike proposal.
- `167_test_root_tie_rank_spectral_lift_parameter_sweep.py` validates
  spectral-reach candidate labeling, required spectral lift, missing
  conditioning, setting summaries, coherent-spike setting generation,
  coherent-spike matrix metadata, target-conditioned spike geometry, matched
  target filtering, and output writing.
- The high-lift manifest records `generated_rows = 1`, `target_rows = 7`,
  `summary = 1`, and `failures = 0`.
- The moderate manifest records the same row counts and no failures.
- The coherent grid manifest records `generated_rows = 6`, `target_rows = 42`,
  `summary = 6`, and `failures = 0`.
- The capped conditioned coherent manifest records `generated_rows = 7`,
  `target_rows = 7`, `summary = 7`, and `failures = 0`.

## Links

- [[root-tie-rank-selected-spectral-generator-targets-20260616]]
- [[root-tie-rank-selected-spectral-excess-panel-20260616]]
- [[selected-neighborhood-measurability-law]]
