---
title: Root Selected Deformed MP Edge 2026-06-17
type: source
status: draft
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/selected/root_selected_mixed_region_law.py
  - benchmarks/diagnostics/calibration/root/selected/root_selected_deformed_mp_edge_panel.py
  - benchmarks/diagnostics/calibration/root/tie_rank/root_tie_rank_null_proposal_frontier.py
  - tests/validation/calibration/root/selected/155_test_root_selected_mixed_region_law.py
  - tests/validation/calibration/root/tie_rank/158_test_root_tie_rank_null_proposal_frontier.py
  - tests/validation/calibration/root/selected/176_test_root_selected_deformed_mp_edge_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_deformed_mp_edge_mild_accumulated
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_topology_join_mild_accumulated
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_frontier_mild_hu_replay_v3_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_importance_external_null_topology_join_mild_hu_replay_v3_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_deformed_mp_edge_mild_hu_replay_v3_smoke
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_spectral_tail_law_deformed_hu_mild_replay_v3_smoke
tags:
  - source
  - diagnostics
  - root
  - spectral
  - population-law
---

# Root Selected Deformed MP Edge 2026-06-17

## Summary

`root_selected_deformed_mp_edge_panel.py` computes a diagnostic plug-in
deformed Marchenko--Pastur edge for observed selected roots with captured
bulk spectra. It uses the Silverstein--Choi inverse map

\[
z(v)
=
-\frac1v
+\gamma\int \frac{t}{1+t v}\,dH_u(t),
\]

and solves the right-edge condition

\[
\frac{1}{v^2}
=
\gamma\int\frac{t^2}{(1+t v)^2}\,dH_u(t)
\]

on the interval \((-1/t_{\max},0)\). The empirical \(H_u\) input is the
captured root spectrum after removing the top `root_raw_mp_signal_count`
identity-MP spikes.

The mild accumulated run computes plug-in deformed edges for `3/7` observed
roots: `overlap_mod_4c_small`, `overlap_mod_6c_med`, and
`overlap_part_4c_small`. The existing accumulated artifact remains
diagnostic-only because generated and selected-null support rows in that CSV do
not yet carry full root spectra or active feature counts. The row schema has
now been extended so newly generated mixed-law/support rows preserve the full
root spectrum, active feature count, projected spectrum, and identity MP edge.
The support-side replay has now been run on the mild importance external-null
smoke after fixing the mixed-law and proposal-frontier metadata contracts. The
refreshed joined table contains full spectra, active feature counts, and
measured topology status for `14/14` external-null rows. The deformed-edge
support output computes support-side \(b(H_u,\gamma)\) for all `14` of those
external rows, but every external row remains below its own deformed edge, so
the support-side deformed spectral excess is `0`.

`root_selected_spectral_tail_law_panel.py` can now consume the target and
support deformed-edge CSVs and use \(S_{H_u}\) as the active tail variable. On
the refreshed mild replay, the deformed-tail panel keeps `2/7` calibrated
roots and `5/7` fail-closed roots. The hard roots remain blocked by missing
same-stratum support, not by missing \(H_u\) measurements.

## Key Points

- `deformed_edge_computed_count = 3`.
- `exact_tail_h_u_optional_count = 2`.
- `support_missing_count = 2`.
- `missing_input_count = 0` for observed targets.
- The median deformed-edge multiplier among computed roots is `1.159819`; the
  maximum is `1.309283`.
- The median deformed spectral excess among computed roots is `0.617050`; the
  maximum is `0.864845`.
- `overlap_mod_4c_small` moves from identity excess `0.765314` to deformed
  excess `0.617050`.
- `overlap_mod_6c_med` moves from identity excess `0.885616` to deformed
  excess `0.616136`.
- `overlap_part_4c_small` moves from identity excess `0.903995` to deformed
  excess `0.864845`.
- The old mild accumulated joined feasibility table still has `0`
  full-spectrum captures for calibration support rows, but the refreshed
  `root_tie_rank_importance_external_null_topology_join_mild_hu_replay_v3_smoke`
  table has `14/14` external-null rows with full spectra, active feature
  counts, and measured bandwidth/topology status.
- `root_selected_mixed_region_law.py` now propagates
  `root_active_feature_count`, `root_full_eigenvalue_count`,
  `root_full_component_eigenvalues_json`, `root_projected_eigenvalues_json`,
  and `root_mp_upper_bound`; `155_test_root_selected_mixed_region_law.py`
  verifies those fields survive the mixed-law layer.
- `root_tie_rank_null_proposal_frontier.py` now preserves the same fields
  through combined feasibility rows without `_x`/`_y` suffixing;
  `158_test_root_tie_rank_null_proposal_frontier.py` covers that contract.
- The refreshed deformed support rows compute `14/14` external-null deformed
  edges. Both `importance_two_block_external_null` and
  `importance_coupled_external_null` have median and maximum
  `s_root_deformed_excess_log = 0`.
- The deformed-tail run sets `spectral_tail_variable = deformed_mp_s_h_u` for
  all seven observed roots.
- Deformed target excesses are `0.617050` for `overlap_mod_4c_small`,
  `0.616136` for `overlap_mod_6c_med`, `0.864845` for
  `overlap_part_4c_small`, `0.175224` for `overlap_unbal_4c_small`, and
  `0.695722` for `overlap_unbal_6c_med`; `overlap_extreme_4c` and
  `overlap_heavy_4c_small_feat` have deformed excess `0`.
- The deformed-tail support status remains `calibrated_tail_count = 2` and
  `fail_closed_missing_support_count = 5`.

## Evidence

For identity \(H=\delta_1\), the panel test verifies that the edge calculation
returns \((1+\sqrt{\gamma})^2\). For each observed root with computable
inputs, the panel records

\[
S_{H_u}
=
\max\left\{
\log\left(\frac{\lambda_{\mathrm{root}}}{b(H_u,\gamma)}\right),
0
\right\}.
\]

The production status remains
`fail_closed_deformed_mp_edge_diagnostic_only` for computed observed roots.
The observed and support-side deformed excess values are now joined into the
selected-root tail law. The next mathematical/software step is therefore a
selected-null or external generator that occupies the missing
\(T,A,E,B,H_u\) strata with nonzero calibrated spectral excess.

## Links

- [[root-selected-h-u-observability-20260617]]
- [[root-selected-population-law-requirement-20260617]]
- [[local-marchenko-pastur-rule]]
- [[selected-geometry-mp-integral-literature-20260602]]
