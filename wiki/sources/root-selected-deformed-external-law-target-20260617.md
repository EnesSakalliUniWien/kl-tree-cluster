---
title: Root Selected Deformed External Law Target 2026-06-17
type: source
status: draft
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_selected_deformed_external_law_target_panel.py
  - tests/validation/178_test_root_selected_deformed_external_law_target_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_deformed_external_law_target_mild_replay_v3_smoke
tags:
  - source
  - diagnostics
  - root
  - spectral
  - selected-tail
---

# Root Selected Deformed External Law Target 2026-06-17

## Summary

`root_selected_deformed_external_law_target_panel.py` turns the deformed
selected-root support gaps into explicit external-law targets. It does not
sample new roots, assign production p-values, or rescue failed roots. It
records the conditional law that must be generated before the deformed
selected-root spectral tail can be calibrated.

The proposed family is a selected-root conditional exponential tilt,
\[
\frac{dQ_\theta}{dP_0}(x)
\propto
\exp\{\theta_TT(x)+\theta_AA(x)+\theta_EE(x)+\theta_SS_{H_u}(x)\},
\]
with hard conditioning on the selected-root event, bandwidth/topology status
\(B\), and local deformed MP bulk \(H_u\). The sufficient statistics are
\(\phi_{\mathrm{root}}=(T,A,E,S_{H_u})\), where \(T\) is selected tie rank,
\(A\) is selected-ratio action, \(E\) is edge action, and \(S_{H_u}\) is the
deformed root spectral excess.

## Key Points

- The mild replay has `7` observed target roots.
- `2/7` roots already have same-stratum tail support.
- `5/7` roots require an external selected-root law before calibration.
- Four missing roots require a joint `A,S_Hu` tilt.
- One missing root requires a joint `E,S_Hu` tilt.
- The maximum required deformed spectral lift is `2.374638`, on
  `overlap_part_4c_small`.
- The panel records exact target moments, but it does not fabricate signed
  nearest-support coordinates for \(T,A,E\). The upstream gap panel stores
  those as absolute nearest-support gaps, so the law target preserves them as
  gaps and only records the nearest support \(S_{H_u}\) value.

This narrows the root problem. The remaining blocker is not the Marchenko--
Pastur edge itself, because deformed \(H_u\) edges are now computed for both
target and support-side rows. The blocker is missing selected-root support in
the same root event geometry. The next inference step is to generate or fit a
conditional external law that occupies the observed \(T,A,E,B,H_u\) stratum
while producing nonzero \(S_{H_u}\).

## Evidence

- `root_selected_deformed_external_law_target_summary.csv` reports
  `target_count = 7`, `external_law_required_count = 5`,
  `existing_support_count = 2`, `selected_ratio_axis_count = 4`,
  `edge_axis_count = 1`, and `summary_status =
  external_law_targets_required`.
- `root_selected_deformed_external_law_target_rows.csv` marks
  `overlap_mod_4c_small`, `overlap_mod_6c_med`,
  `overlap_part_4c_small`, and `overlap_unbal_4c_small` as requiring
  `A,S_Hu`.
- The same row file marks `overlap_heavy_4c_small_feat` as requiring
  `E,S_Hu`.
- `178_test_root_selected_deformed_external_law_target_panel.py` verifies
  that nearest \(T,A,E\) support coordinates are not reconstructed from
  absolute gaps.

## Links

- [[root-selected-deformed-tail-support-gap-20260617]]
- [[root-selected-deformed-mp-edge-20260617]]
- [[root-selected-importance-tail-support-20260617]]
- [[root-tie-rank-conditioned-coherent-topology-join-20260617]]
