---
title: Root Tie Rank Target Conditioned Importance Frontier 2026-06-17
type: source
status: draft
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/tie_rank/root_tie_rank_target_conditioned_importance_frontier.py
  - tests/validation/calibration/root/tie_rank/170_test_root_tie_rank_target_conditioned_importance_frontier.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_target_conditioned_importance_frontier_narrow_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_target_conditioned_importance_frontier_coupled_narrow_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_target_conditioned_importance_frontier_unbalanced_mod_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_target_conditioned_importance_frontier_unbalanced_mod_boundary_probe
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_target_conditioned_importance_frontier_unbalanced_mod_delta126_probe
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_target_conditioned_importance_frontier_unbalanced_mod_rejection_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_target_conditioned_importance_frontier_unbalanced_mod_boundary_fine_probe
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_target_conditioned_importance_frontier_two_factor_mod_smoke
tags:
  - source
  - diagnostics
  - root
  - spectral
  - importance-sampling
---

# Root Tie Rank Target Conditioned Importance Frontier 2026-06-17

## Summary

`root_tie_rank_target_conditioned_importance_frontier.py` adds a
target-conditioned importance frontier for the remaining selected-root spectral
tail problem. It reads unsupported observed targets, generates
likelihood-ratio-weighted external-null proposal rows with
`conditioning_target_case_id`, and reports whether each candidate reaches the
same pre-topology \(T,A,E\) stratum as its target before paying for
generated-neighborhood topology replay.

The first narrow smokes show that target conditioning alone is not enough for
the current two-block/coupled Bernoulli tilt families. For
`overlap_heavy_4c_small_feat` and `overlap_mod_4c_small`, both pure two-block
and coupled narrow settings produce zero pre-topology stratum hits. This
confirms that the remaining blocker is the proposal shape: the current tilted
families jump between low/low and high/high action-edge bands instead of
landing in the mixed low/mid and mid/high strata required by the observed
roots.

An unbalanced two-block external proposal was added to test whether unequal
latent block sizes decouple selected-ratio action from edge action. It improves
localization near the `overlap_mod_4c_small` target but still produces zero
pre-topology hits: candidates alternate between low/low and high/high
action-edge bands. The closest rows have the right tie band and high edge, but
their selected-ratio action remains just above the mid-band boundary.

The frontier now also has an accepted-stratum rejection mode. It repeatedly
samples from the likelihood-ratio proposal but retains only candidates whose
pre-topology \(T,A,E\) key matches the target before generated-neighborhood
topology replay. A micro smoke on `overlap_mod_4c_small` with unbalanced
two-block deltas `0.126` and `0.129`, block fraction `0.55`, and `10`
attempts per setting retained `0` rows. This is negative evidence: simply
rejecting to the observed mixed action-edge stratum is too sparse with the
current proposal shape.

A correlated two-factor external proposal was then added to test a more
specific signal-theoretic hypothesis. The proposal separates a primary root
contrast from a partially correlated residual factor:

\[
P_Q(X_{ij}=1)
=
\operatorname{clip}\left[
p_0
+\delta_1 z_i s_j
+\delta_2 r_i h_j
\right],
\]

where \(z_i\) is the primary two-block root factor, \(r_i\) is an independent
or \(z_i\)-correlated residual factor, \(s_j\) is a dense sign pattern, and
\(h_j\) is sparse. The likelihood-ratio weight remains
\(\log(dP_0/dQ)\). This tests whether parent-space residual energy can raise
edge action without always forcing the selected sibling action into the high
band. The first compact smoke still reports zero pre-topology support for
`overlap_mod_4c_small`.

## Key Points

- The diagnostic preserves external-null semantics by using the existing
  likelihood-ratio proposal generators and labeling rows as
  `external_selected_null`/`external_null_support`.
- Generated rows carry `conditioning_target_case_id`, so later root-tail
  joins can prevent cross-target borrowing.
- The pre-topology stratum key intentionally uses only root component, tie band,
  selected-ratio action band, and edge-action band. \(S_{\mathrm{root}}\) and
  measured bandwidth \(B\) are excluded at this stage.
- The pure two-block narrow smoke over two unsupported targets writes `4`
  generated rows and reports `pre_topology_supported_target_count = 0`.
- The coupled narrow smoke over the same targets also writes `4` generated rows
  and reports `pre_topology_supported_target_count = 0`.
- The unbalanced two-block smoke over `overlap_mod_4c_small` writes `8`
  generated rows and reports `pre_topology_supported_target_count = 0`.
- A boundary probe with deltas `0.127`, `0.128`, `0.129`, block fractions
  `0.55`, `0.60`, and two replicates per setting writes `12` generated rows
  and still reports no pre-topology hit.
- A final micro-probe at delta `0.126`, block fraction `0.55`, and six seeds
  also reports no pre-topology hit. The best near-miss has tie mid and
  high/high action-edge, while other rows fall to low/low.
- The accepted-stratum rejection smoke on `overlap_mod_4c_small` tries `20`
  proposal candidates and retains `0`, so no generated rows are available for
  topology replay in that run.
- A fine unbalanced boundary probe over deltas `0.123`--`0.126`, block
  fractions `0.50`, `0.55`, and `0.60`, and one replicate per setting writes
  `12` generated rows and still reports no pre-topology support.
- The correlated two-factor smoke over deltas `0.118`, `0.122`, spike deltas
  `0.08`, `0.14`, and factor correlations `0.25`, `0.50` writes `8`
  generated rows and reports no pre-topology support. Rows still fall into
  low/low or high/high action-edge bands, often with high tie rank.
- Because no pre-topology hit exists, these rows were not replayed through
  topology; they cannot improve same-stratum \(T,A,E,B,H_u\) support.

## Evidence

- The pure two-block narrow smoke tests `two_block_delta` values `0.135` and
  `0.145`. `overlap_heavy_4c_small_feat` jumps from low/low to high/high
  action-edge bands, while `overlap_mod_4c_small` stays high/high instead of
  hitting mid/high.
- The coupled narrow smoke tests `two_block_delta` values `0.125` and `0.135`
  with spike fraction `0.08` and spike delta `0.20`. It also reports no
  pre-topology target-stratum hits.
- The unbalanced proposal records
  `target_iid_bernoulli_over_unbalanced_tilted_proposal` likelihood-ratio
  weights, so the negative result is still on the same external-null
  importance path rather than a diagnostic-only unweighted sample.
- The accepted-stratum rejection mode is mathematically a support-search tool,
  not a production p-value rule. Within a fixed accepted stratum, the rejection
  normalizer is target/setting-specific and must be handled before using
  accepted rows for weighted spectral tails.
- The correlated two-factor result argues that the missing law is not simply
  extra parent spectral energy. The selected root construction still couples
  tie-rank, selected-ratio action, and edge action sharply enough that a
  low-dimensional proposal surface does not populate the mixed stratum.
- `170_test_root_tie_rank_target_conditioned_importance_frontier.py` verifies
  that pre-topology keys ignore spectral tail and bandwidth, and that
  target-conditioned candidates are not borrowed across targets. It also
  verifies the correlated two-factor likelihood-ratio metadata.

## Links

- [[root-selected-importance-tail-support-20260617]]
- [[selected-neighborhood-measurability-law]]
