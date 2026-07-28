---
title: Root Selected Deformed Tail Support Gap 2026-06-17
type: source
status: draft
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/selected/root_selected_deformed_tail_support_gap_panel.py
  - tests/validation/calibration/root/selected/177_test_root_selected_deformed_tail_support_gap_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_deformed_tail_support_gap_mild_replay_v3_smoke
tags:
  - source
  - diagnostics
  - root
  - spectral
  - selected-tail
---

# Root Selected Deformed Tail Support Gap 2026-06-17

## Summary

`root_selected_deformed_tail_support_gap_panel.py` localizes the missing law
after the selected-root tail variable has been switched from identity
\(S_{\mathrm{root}}\) to deformed
\[
S_{H_u}
= \max\left\{\log\frac{\lambda_{\mathrm{root}}}{b(H_u,\gamma)},0\right\}.
\]
It reads the refreshed \(T,A,E,B,H_u\) joined feasibility table, the deformed
tail panel, and the support-side deformed MP edge rows. It then reports whether
same-stratum support exists and how much \(S_{H_u}\) lift the nearest support
row would need to reach the observed target.

The mild replay writes seven target rows. Exact support remains available for
`2/7` roots, while `5/7` roots fail closed because same-stratum deformed-tail
support is missing. Every target has some nearest support row with deformed
edge computed, but the nearest support \(S_{H_u}\) is `0` for all targets.
Thus the remaining law is not another MP-edge calculation; it must generate
selected-null or external roots in the same \(T,A,E,B,H_u\) stratum with
nonzero deformed spectral excess.

## Key Points

- `exact_support_target_count = 2`.
- `fail_closed_target_count = 5`.
- `missing_same_stratum_target_count = 5`.
- `same_stratum_nonexceeding_target_count = 0`.
- `nearest_deformed_support_available_target_count = 7`.
- `max_required_s_h_u_gap = 0.864845`.
- `max_required_s_h_u_lift_multiplier = 2.374638`.
- The two exact-support roots are `overlap_extreme_4c` and
  `overlap_unbal_6c_med`.
- The five fail-closed roots are `overlap_heavy_4c_small_feat`,
  `overlap_mod_4c_small`, `overlap_mod_6c_med`,
  `overlap_part_4c_small`, and `overlap_unbal_4c_small`.
- Dominant conditioning gap is selected-ratio action for four of the five
  fail-closed roots and edge action for `overlap_heavy_4c_small_feat`.
- Required deformed-excess lift multipliers are `1.853452` for
  `overlap_mod_4c_small`, `1.851760` for `overlap_mod_6c_med`,
  `2.374638` for `overlap_part_4c_small`, and `1.191513` for
  `overlap_unbal_4c_small`; `overlap_heavy_4c_small_feat` has target
  \(S_{H_u}=0\) but still lacks same-stratum support.

## Evidence

The panel preserves the production rule:

\[
\Pr_0\!\left(S_{H_u}^{\mathrm{null}}\ge S_{H_u}^{\mathrm{obs}}
\mid R,T,A,E,B,H_u\right)
\]

is only estimated when support exists in the same conditioning stratum. Nearest
support remains diagnostic. The output therefore converts the current failure
into a generator/law requirement:

```text
generate_same_T_A_E_B_Hu_roots_with_nonzero_deformed_spectral_excess
```

for every fail-closed target.

## Links

- [[root-selected-deformed-mp-edge-20260617]]
- [[root-selected-importance-tail-support-20260617]]
- [[root-selected-spectral-tail-law-with-legacy-overlay-20260617]]
- [[root-tie-rank-target-conditioned-importance-frontier-20260617]]
