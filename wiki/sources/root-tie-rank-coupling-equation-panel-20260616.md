---
title: Root Tie Rank Coupling Equation Panel 2026-06-16
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/tie_rank/root_tie_rank_coupling_equation_panel.py
  - tests/validation/calibration/root/tie_rank/161_test_root_tie_rank_coupling_equation_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_coupling_equation_panel_two_case_smoke
tags:
  - source
  - diagnostics
  - root
  - spectral
  - proposals
---

# Root Tie Rank Coupling Equation Panel 2026-06-16

## Summary

`root_tie_rank_coupling_equation_panel.py` turns the selected
spectral-action gap into explicit coupled coordinates. For each observed
overlap root and proposal family, it computes action
\(A=\log(1+\text{selected ratio})\), edge action
\(E=\log(1+\max\{\text{edge margin},0\})\), spectral excess
\(S=\max\{\log(\lambda/\lambda_{\mathrm{MP}}),0\}\), and tie-rank fraction
\(T\). The main diagnostic scalar is
\[
C_{\min}=T\min(A,E)S.
\]
It also computes a measured-neighborhood version that multiplies by a
bandwidth-measurability indicator, so generated rows with missing bandwidth
frontier evidence cannot be treated as supported neighborhood coupling.

## Key Points

- The two-case smoke writes `35` target-by-family coupling rows and `5`
  proposal-family summaries.
- No proposal family reaches measured-neighborhood coupling on any target:
  `measured_neighborhood_coupling_dominance_count` is `0/7` for every family.
- Every generated proposal row has unmeasured neighborhood bandwidth:
  `generated_neighborhood_measured_count` is `0/7` for every family.
- Pure bottleneck coupling \(C_{\min}\) reaches some easier observed roots but
  not the hard overlap roots. `column_beta_bernoulli`,
  `coupled_edge_spectral_proposal`, `sparse_block_spike_proposal`, and
  `two_block_tilt_proposal` have bottleneck-coupling dominance on `1/7`,
  `2/7`, `2/7`, and `2/7` targets respectively.
- `iid_marginal_bernoulli` reaches no bottleneck-coupling target and remains
  summarized as `spectral_excess_action_edge_deficit`.
- Dense/coupled proposals remain action-edge heavy but spectrally weak on the
  hard roots. For `overlap_mod_4c_small`, the coupled proposal has
  bottleneck-coupling ratio about `0.475`; for `overlap_mod_6c_med`, about
  `0.330`.
- The summary status for all non-iid families is
  `coupling_reached_but_bandwidth_unmeasured`, not production support. This
  means the pure coupling scalar is useful as a diagnostic coordinate, but the
  selected-neighborhood measurability factor remains missing.

## Evidence

- `root_tie_rank_coupling_equation_panel.py` implements coupled root
  coordinates, measured-neighborhood coupling, proposal-family summaries, and
  manifest output.
- `161_test_root_tie_rank_coupling_equation_panel.py` verifies full measured
  coupling, dense action-edge/spectral-low failure, sparse
  spectral/action-edge-low failure, summary labels, and output writing.
- The smoke manifest records `rows = 35` and `summary = 5`.

## Links

- [[root-tie-rank-spectral-action-dominance-panel-20260616]]
- [[root-tie-rank-proposal-gap-panel-20260616]]
- [[root-tie-rank-null-proposal-frontier-20260616]]
- [[selected-neighborhood-measurability-law]]
