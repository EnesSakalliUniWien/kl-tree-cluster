---
title: Root Selected Spectral Tail Law With Legacy Overlay 2026-06-17
type: source
status: reviewed
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_selected_spectral_tail_law_panel.py
  - tests/validation/169_test_root_selected_spectral_tail_law_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_spectral_tail_law_with_legacy_overlay
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_method_comparison_panel
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_internal_spectral_comparison_panel
tags:
  - source
  - diagnostics
  - root
  - spectral
  - legacy
---

# Root Selected Spectral Tail Law With Legacy Overlay 2026-06-17

## Summary

`root_selected_spectral_tail_law_panel.py` expresses the selected-root
spectral-tail inference target directly. For each observed root, it records
\(S_{\mathrm{root}}=\log(\lambda/\lambda_{\mathrm{MP}})\), selected
tie-rank \(T\), selected-ratio action \(A\), edge-margin action \(E\),
measured bandwidth/topology status \(B\), and local population-law status
\(H_u\). It then searches for selected-null or external-null support in the
same coarse root stratum over the root event, \(T,A,E,B,H_u\), leaving
\(S_{\mathrm{root}}\) as the tail variable.

The panel also joins the full legacy `c2ef9a69` method comparison and the
legacy internal-barycenter spectral comparison. This shows that old-method
behavior does not close the root-tail law: the full legacy method leaks one
selected-null false split on an overlap root, and the internal-barycenter
spectral diagnostic changes MP counts without providing calibrated
selected-root spectral-tail support.

## Key Points

- The run writes `7` root-tail rows and one summary row under
  `root_selected_spectral_tail_law_with_legacy_overlay`.
- All `7/7` observed roots have
  `fail_closed_selected_root_spectral_tail_support_missing`.
- `selected_null_support_count = 0` for every observed root, so no conservative
  spectral-tail p-value is emitted.
- The summary records `calibrated_tail_count = 0` and
  `fail_closed_missing_support_count = 7`.
- The full legacy method overlay records one selected-null false split:
  `overlap_mod_4c_small` has `legacy_full_selected_null_legacy_false_split =
  True`.
- The legacy internal-barycenter overlay records MP count changes on compact
  overlap rows, for example selected-null
  `overlap_mod_4c_small` has delta raw MP signal count `179`, and signal
  `overlap_mod_4c_small` has delta `218`. These changes remain diagnostic
  because no selected-null spectral-tail support exists in the root stratum.
- The local population-law status is explicitly
  `identity_mp_assumed_deformed_mp_unestimated`, so \(H_u\) remains an open
  layer rather than a validated deformed-MP correction.

## Evidence

- `root_selected_spectral_tail_law_panel.py` builds the stratum key from the
  root selected event, \(T,A,E,B,H_u\), deliberately excluding
  \(S_{\mathrm{root}}\) from conditioning so it remains the tail variable.
- `169_test_root_selected_spectral_tail_law_panel.py` validates fail-closed
  behavior when support is missing, conservative plus-one empirical tail
  p-values when selected-null support exists, and legacy overlay fields.
- The manifest records the joined feasibility input from
  [[root-tie-rank-conditioned-coherent-topology-join-20260617]] and both legacy
  pairwise comparison files.

## Links

- [[root-tie-rank-conditioned-coherent-topology-join-20260617]]
- [[legacy-c2ef9a69-method-comparison-panel-20260616]]
- [[legacy-internal-spectral-comparison-panel-20260616]]
- [[local-marchenko-pastur-rule]]
- [[selected-neighborhood-measurability-law]]
