---
title: Root Selected Importance Tail Support 2026-06-17
type: source
status: draft
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_tie_rank_null_proposal_frontier.py
  - benchmarks/diagnostics/calibration/root_tie_rank_generated_neighborhood_replay.py
  - benchmarks/diagnostics/calibration/root_tie_rank_conditioned_coherent_topology_join.py
  - benchmarks/diagnostics/calibration/root_selected_spectral_tail_law_panel.py
  - tests/validation/168_test_root_tie_rank_conditioned_coherent_topology_join.py
  - tests/validation/158_test_root_tie_rank_null_proposal_frontier.py
  - tests/validation/169_test_root_selected_spectral_tail_law_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_frontier_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_replay_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_topology_join_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_spectral_tail_law_importance_external_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_frontier_mild_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_replay_mild_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_topology_join_mild_accumulated
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_spectral_tail_law_importance_external_mild_accumulated
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_frontier_twoblock013_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_frontier_twoblock012_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_frontier_mild_hu_replay_v3_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_replay_mild_hu_replay_v3_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_topology_join_mild_hu_replay_v3_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_spectral_tail_law_importance_external_mild_hu_replay_v3_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_spectral_tail_law_deformed_hu_mild_replay_v3_smoke
tags:
  - source
  - diagnostics
  - root
  - spectral
  - importance-sampling
---

# Root Selected Importance Tail Support 2026-06-17

## Summary

The root selected spectral-tail path now has an explicit importance-weighted
external-null support channel. The proposal frontier accepts tilted binary
proposal families whose rows are labeled `external_selected_null` only when
they carry a likelihood-ratio weight
\[
\log w(X)=\log P_0(X)-\log Q(X),
\]
where \(P_0\) is the matched iid Bernoulli null and \(Q\) is the tilted
proposal used to reach rare selected-root geometry. The root-tail panel then
uses an effective-sample-size weighted conservative tail estimate instead of
treating tilted proposal rows as ordinary empirical null rows.

After generated-neighborhood/topology replay and joining measured bandwidth
evidence back into the root-tail input, the first importance smoke gives
same-stratum external support for `1/7` observed roots. The supported row is
`overlap_unbal_6c_med`, with one non-exceeding importance row, effective
sample size `1.0`, and conservative p-value `0.5`. The other six roots still
fail closed. This is the first evidence that the selected-root spectral tail
law can be populated in the observed high tie/action/topology strata, but it
is not precision-ready.

A second milder importance smoke improves the supported count to `2/7` by
adding support for `overlap_extreme_4c`. Additional scalar two-block tilts at
`0.13` and `0.12` show that blind one-parameter tilting is not enough for the
remaining roots: proposal rows jump between low/low and high/high
action-edge bands, while the unsupported observed roots require low/mid,
mid/high, or mid/mid action-edge bands with `bandwidth_root_reopen_observed`.
The next generator should therefore be target-conditioned in \(T,A,E,B\), not
another global scalar sweep.

The refreshed \(H_u\)-replay version of the mild importance path preserves full
root spectra and active feature counts for `14/14` external-null rows, replays
their topology, and reruns the root-tail panel. The identity-MP tail support
status remains `2/7`: `overlap_extreme_4c` and `overlap_unbal_6c_med` are
calibrated by conservative weighted p-values, while the other five roots fail
closed for missing same-stratum support. The refreshed deformed-MP support
diagnostic shows why this is not a spectral rescue: all external-null support
rows have zero deformed spectral excess.

The deformed \(S_{H_u}\) tail run uses the same support contract but replaces
the active tail variable with `s_root_deformed_excess_log` from the deformed-MP
edge panel. Its result is also `2/7` calibrated and `5/7` fail-closed. Thus the
remaining blocker is not the edge formula alone; it is the absence of
same-stratum selected-null or external roots with nonzero deformed spectral
excess.

## Key Points

- `importance_two_block_external_null` and `importance_coupled_external_null`
  are supported proposal families in the root tie-rank proposal frontier.
- Importance proposal rows are labeled with `data_role =
  external_selected_null` and `calibration_role = external_null_support`.
- The generator records `target_null_log_probability`,
  `proposal_log_probability`, `importance_log_weight`, and
  `importance_law_status`.
- `root_selected_spectral_tail_law_panel.py` now computes the selected-root
  spectral tail from same-stratum support using either direct plus-one
  empirical counts or an importance-weighted effective sample size:
  \[
  \hat p_{\mathrm{ESS}}
  =
  \frac{1+n_{\mathrm{eff}}\hat p_w}{1+n_{\mathrm{eff}}},
  \qquad
  n_{\mathrm{eff}}
  =
  \frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
  \]
- The regression test covers the weighted case with two external rows, one
  exceedance, and unequal weights.
- The importance frontier smoke generated `14` external rows from
  `importance_two_block_external_null` and `importance_coupled_external_null`
  with likelihood-ratio metadata.
- Generated-neighborhood/topology replay completed on those importance rows,
  giving measured root-frontier bandwidth status before root-tail calibration.
- Joining the importance rows to the existing conditioned-coherent feasibility
  table and rerunning the root-tail panel gives `calibrated_tail_count = 1`
  and `fail_closed_missing_support_count = 6`.
- The supported target is `overlap_unbal_6c_med`; it has
  `selected_null_support_count = 1`,
  `selected_null_importance_effective_sample_size = 1.0`,
  `selected_null_exceedance_count = 0`, and
  `conservative_spectral_tail_p_value = 0.5`.
- The result remains diagnostic-only because one effective support row cannot
  supply tail precision or alpha-resolution for a production claim.
- The milder accumulated run gives `calibrated_tail_count = 2` and
  `fail_closed_missing_support_count = 5`. It adds `overlap_extreme_4c`, with
  one non-exceeding weighted support row and conservative p-value `0.5`;
  `overlap_unbal_6c_med` has six support rows but ESS remains `1.0`, showing
  severe importance-weight degeneracy.
- The refreshed mild \(H_u\)-replay run keeps `calibrated_tail_count = 2` and
  `fail_closed_missing_support_count = 5`. `overlap_extreme_4c` has one
  weighted support row with p-value `1.0`; `overlap_unbal_6c_med` has six
  support rows, no exceedance, ESS `1.0`, and p-value `0.5`.
- The refreshed joined table has `14/14` external-null rows with full spectra,
  active feature counts, and measured bandwidth/topology status.
- Support-side deformed-edge diagnostics compute for all `14` external-null
  rows, but both importance families have median and maximum
  `s_root_deformed_excess_log = 0`.
- The deformed \(S_{H_u}\) tail panel keeps `calibrated_tail_count = 2`,
  `fail_closed_missing_support_count = 5`, and
  `spectral_tail_variable = deformed_mp_s_h_u` for all seven targets.
- The roots still fail closed under \(S_{H_u}\) for the same support reason:
  `overlap_heavy_4c_small_feat`, `overlap_mod_4c_small`,
  `overlap_mod_6c_med`, `overlap_part_4c_small`, and
  `overlap_unbal_4c_small` have no selected-null/external support in the same
  \(T,A,E,B,H_u\) stratum.
- Scalar two-block probes at `0.13` and `0.12` were completed but not replayed
  because their pre-replay \(T,A,E\) bands did not match the remaining missing
  target strata.
- Rerunning the legacy-overlay root-tail panel without the new importance rows
  still gives `calibrated_tail_count = 0` and
  `fail_closed_missing_support_count = 7`.
- A 20x iid selected-null probe was started but interrupted because the
  average-linkage replay was too slow for interactive iteration. It should not
  be treated as evidence because no completed manifest was written.

## Evidence

- `root_tie_rank_null_proposal_frontier.py` computes Bernoulli
  log-likelihood-ratio metadata for tilted external-null proposals.
- `root_tie_rank_null_proposal_frontier.py` also preserves root spectral-bulk
  metadata through combined feasibility rows; the regression test prevents
  duplicate-column suffixing from dropping canonical \(H_u\) fields.
- `root_selected_spectral_tail_law_panel.py` leaves
  \(S_{\mathrm{root}}\) as the tail variable and uses weighted p-values only
  when same-stratum support exists.
- `169_test_root_selected_spectral_tail_law_panel.py` verifies direct
  empirical support and importance-weighted external support.
- `root_tie_rank_conditioned_coherent_topology_join.py` now ignores observed
  target rows when a combined feasibility table is passed as generated input,
  so observed root targets are not overwritten during external-support joins.
- The regenerated importance root-tail summary still reports
  `selected_root_spectral_tail_support_missing` globally because six of seven
  roots remain unsupported, but one same-stratum weighted external tail is now
  available.

## Links

- [[root-selected-spectral-tail-law-with-legacy-overlay-20260617]]
- [[root-tie-rank-null-proposal-frontier-20260616]]
- [[selected-neighborhood-measurability-law]]
