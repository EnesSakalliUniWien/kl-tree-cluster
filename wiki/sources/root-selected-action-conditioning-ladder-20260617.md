---
title: Root Selected Action Conditioning Ladder 2026-06-17
type: source
status: draft
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_selected_action_conditioning_ladder_panel.py
  - tests/validation/172_test_root_selected_action_conditioning_ladder_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_action_conditioning_ladder_mild_accumulated
tags:
  - source
  - diagnostics
  - root
  - action
  - support
---

# Root Selected Action Conditioning Ladder 2026-06-17

## Summary

`root_selected_action_conditioning_ladder_panel.py` quantifies the
selected-ratio action gap found by the nearest-support panel. It compares exact
\(T,A,E,B,H_u\) root-tail support with relaxed diagnostic levels that remove
\(A\), then remove \(B\), or remove \(E\). Exact support remains the only
calibration path; relaxed support is explicitly labeled
`relaxed_conditioning_not_calibration`.

The mild accumulated run preserves the root-tail result: exact support exists
for `2/7` observed roots. Relaxing only \(A\) while keeping \(T,E,B,H_u\)
restores support for three additional unsupported roots:
`overlap_mod_4c_small`, `overlap_mod_6c_med`, and
`overlap_part_4c_small`. The remaining two unsupported roots,
`overlap_heavy_4c_small_feat` and `overlap_unbal_4c_small`, still have no
support after relaxing \(A\) and \(B\); support appears only when \(E\) is
also relaxed. Thus the missing selected-root law separates into an
action-only gap for three roots and an action-plus-edge gap for two roots.

## Key Points

- The panel writes `28` rows: four conditioning levels for each of seven
  observed roots.
- The summary reports `exact_supported_target_count = 2`,
  `action_relaxed_supported_target_count = 5`, and
  `action_only_gap_target_count = 3`.
- The exact supported roots are unchanged: `overlap_extreme_4c` and
  `overlap_unbal_6c_med`.
- For `overlap_mod_4c_small`, exact support is missing but
  `relax_A_keep_T_E_B_H` has support count `2`; the old full commit overlay is
  still `legacy_full_method_leaks_selected_null_root_risk`.
- For `overlap_mod_6c_med` and `overlap_part_4c_small`,
  `relax_A_keep_T_E_B_H` also has support count `2`.
- For `overlap_heavy_4c_small_feat` and `overlap_unbal_4c_small`, support
  remains missing under `relax_A_keep_T_E_B_H` and
  `relax_A_B_keep_T_E_H`, but appears under `relax_A_E_keep_T_B_H`.
- No relaxed row emits a production p-value. Unsupported exact rows remain
  `fail_closed_action_ladder_diagnostic_only`.

## Evidence

The ladder levels are:

\[
L_0=(T,A,E,B,H_u),
\]

\[
L_1=(T,E,B,H_u)\quad\text{with }A\text{ relaxed},
\]

\[
L_2=(T,E,H_u)\quad\text{with }A,B\text{ relaxed},
\]

\[
L_3=(T,B,H_u)\quad\text{with }A,E\text{ relaxed}.
\]

The run shows that \(L_1\) is populated for three previously unsupported hard
roots, so the minimal missing coordinate for those roots is \(A\). For the two
remaining roots, \(L_1\) and \(L_2\) are empty while \(L_3\) is populated, so
edge action \(E\) is also part of the conditioning blocker.

## Links

- [[root-selected-spectral-tail-nearest-support-20260617]]
- [[root-selected-importance-tail-support-20260617]]
- [[root-tie-rank-target-conditioned-importance-frontier-20260617]]
- [[selected-neighborhood-measurability-law]]
