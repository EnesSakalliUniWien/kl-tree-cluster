---
title: Root Selected Conditional Tilt Feasibility 2026-06-17
type: source
status: draft
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_selected_conditional_tilt_feasibility_panel.py
  - tests/validation/179_test_root_selected_conditional_tilt_feasibility_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_conditional_tilt_feasibility_mild_replay_v3_smoke/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_conditional_tilt_feasibility_mild_replay_v3_smoke/root_selected_conditional_tilt_feasibility_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_conditional_tilt_feasibility_mild_replay_v3_smoke/root_selected_conditional_tilt_feasibility_summary.csv
tags:
  - source
  - diagnostics
  - root
  - spectral
  - selected-tail
---

# Root Selected Conditional Tilt Feasibility 2026-06-17

## Summary

`root_selected_conditional_tilt_feasibility_panel.py` checks whether the
current external-null support can be reweighted into the selected-root law
target. It is the finite-support version of the conditional exponential tilt
\[
\frac{dQ_\theta}{dP_0}(x)
\propto
\exp\{\theta_TT(x)+\theta_AA(x)+\theta_EE(x)+\theta_SS_{H_u}(x)\},
\]
with hard conditioning on the selected-root event, bandwidth/topology status
\(B\), and measured local spectral population law \(H_u\).

The mathematical check is convex-hull feasibility. If the observed target
moment vector lies outside the convex hull of the available support moment
vectors, no finite exponential tilt over the current support can match it.
Such rows must remain fail-closed.

## Key Points

- The mild replay writes `19` feasibility rows over seven observed roots.
- Full \((T,A,E,S_{H_u})\) feasibility is `0/7` under the same \(B,H_u\)
  support pool.
- Required-axis feasibility is `1/5` for roots that still need a tilt.
- The only required-axis feasible root is `overlap_heavy_4c_small_feat`,
  because its target \(S_{H_u}=0\) and the `E,S_Hu` target lies on the current
  support hull.
- The four `A,S_Hu` roots with nonzero deformed spectral excess remain outside
  the support hull:
  `overlap_mod_4c_small`, `overlap_mod_6c_med`,
  `overlap_part_4c_small`, and `overlap_unbal_4c_small`.
- Their required-axis residuals are dominated by \(S_{H_u}\):
  `0.617050`, `0.616136`, `0.864845`, and `0.175224`.

This proves that a simple reweighting or entropy tilt of the existing
external-null rows is insufficient for the hard selected-root cases. The next
generator must produce actual selected-root support with nonzero
\(S_{H_u}\) while staying in the observed \(T,A,E,B,H_u\) geometry.

## Evidence

- `root_selected_conditional_tilt_feasibility_summary.csv` reports
  `full_phi_feasible_count = 0`, `required_axes_feasible_count = 1`,
  `full_phi_outside_hull_count = 7`, `required_axes_outside_hull_count = 4`,
  and `summary_status = conditional_tilt_support_hull_incomplete`.
- `root_selected_conditional_tilt_feasibility_rows.csv` shows the four
  nonzero-\(S_{H_u}\) required-axis rows have residual vectors whose
  \(S_{H_u}\) component equals the target deformed spectral excess.
- `179_test_root_selected_conditional_tilt_feasibility_panel.py` verifies that
  nonzero target \(S_{H_u}\) outside the support range fails closed, while a
  zero-\(S_{H_u}\) target can be matched on the support boundary.

## Links

- [[root-selected-deformed-external-law-target-20260617]]
- [[root-selected-deformed-tail-support-gap-20260617]]
- [[root-selected-deformed-mp-edge-20260617]]
- [[root-selected-importance-tail-support-20260617]]
