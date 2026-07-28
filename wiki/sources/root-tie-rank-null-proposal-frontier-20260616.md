---
title: Root Tie Rank Null Proposal Frontier 2026-06-16
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/tie_rank/root_tie_rank_null_proposal_frontier.py
  - tests/validation/calibration/root/tie_rank/158_test_root_tie_rank_null_proposal_frontier.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_null_proposal_frontier_two_case_smoke
tags:
  - source
  - diagnostics
  - root
  - calibration
  - proposals
---

# Root Tie Rank Null Proposal Frontier 2026-06-16

## Summary

`root_tie_rank_null_proposal_frontier.py` extends the root tie-rank
calibration path with explicitly labeled proposal families. The diagnostic
keeps iid Bernoulli rows as calibration-candidate support, while
column-heterogeneous Bernoulli and two-block tilted rows are labeled as
diagnostic proposals that can test stratum reachability but do not count as
selected-null calibration support.

## Key Points

- The implementation writes proposal root rows, merge margins, tie rows,
  mixed-law rows, combined feasibility rows, target support, target frontier
  rows, proposal summaries, failures, and a manifest.
- Validation pins the semantic boundary: enriched proposal rows can hit target
  strata, but they do not increase `stratum_calibration_null_support_count`
  unless their `calibration_role` is selected-null support.
- The two-case smoke targeted `overlap_mod_4c_small` and `overlap_mod_6c_med`
  with one replicate each for `iid_marginal_bernoulli`,
  `column_beta_bernoulli`, `two_block_tilt_proposal`,
  `sparse_block_spike_proposal`, and `coupled_edge_spectral_proposal`.
- The final run generated `10` root rows, `4990` merge-margin rows, `14`
  combined feasibility strata, `35` target-frontier rows, and `0` failures in
  about `89.01` seconds.
- None of the five proposal families hit any of the seven observed target
  strata. Iid and column-beta rows stayed at root selected ratios around
  `3.44` to `11.04`, far below the observed target roots.
- The two-block tilted proposal produced very large root selected ratios
  (`7368.65` and `28387.42`) and high edge margins, but still missed the
  target strata because the spectral-ratio bands were only
  `spectral_ratio_le_1` and `spectral_ratio_1_2`.
- The sparse block-spike proposal reached `spectral_ratio_gt_4` for
  `overlap_mod_4c_small`, but with low edge margin and a root selected ratio
  only `5.97`. Thus sparse spectral structure can move the MP coordinate
  without producing the observed high-action root event.
- The coupled edge-spectral proposal combined dense action and sparse
  spectral perturbation, but behaved like the dense tilt: selected ratios were
  about `6987.92` and `30497.79`, edge margins were high, and spectral-ratio
  bands remained `spectral_ratio_1_2`. The simple superposition therefore does
  not reproduce the observed high-action plus high-spectral-ratio root event.
- Generated proposal rows now mark unjoined topology-frontier bandwidth as
  `bandwidth_reopen_missing`, not `bandwidth_no_root_reopen`. This prevents
  unmeasured bandwidth evidence from being mistaken for measured no-reopen
  evidence.
- Therefore matching any single coordinate is not enough. The missing root law
  must jointly condition selected tie rank, edge margin, spectral ratio, and
  measured bandwidth-reopen behavior.

## Evidence

- `root_tie_rank_null_proposal_frontier.py` implements the three proposal
  families, proposal/calibration role labels, target-frontier table, and
  proposal summary.
- `158_test_root_tie_rank_null_proposal_frontier.py` verifies matrix
  generation, target-stratum hit counting, calibration-vs-diagnostic proposal
  separation, and output writing.
- The two-case smoke manifest records `root_rows = 10`, `failures = 0`,
  `target_frontier = 35`, and `proposal_summary = 5`.
- The mixed-row output shows that the two-block proposal can create extreme
  selected ratios without creating the observed spectral-ratio coordinate, and
  that the sparse block-spike proposal can create a high spectral-ratio band
  without creating the observed edge-margin or selected-ratio coordinate.
- The coupled proposal rows show that adding a sparse spike to dense action is
  still insufficient; the observed root event appears to require a
  selection-coupled spectral action rather than an additive perturbation.

## Links

- [[root-tie-rank-selected-null-simulation-pilot-20260616]]
- [[root-tie-rank-calibration-feasibility-20260616]]
- [[root-selected-mixed-region-law-20260616]]
- [[selected-neighborhood-measurability-law]]
