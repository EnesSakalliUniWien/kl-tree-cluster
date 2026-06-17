---
title: Spectral Transport Threshold Calibration Panel 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/spectral_transport_threshold_calibration_panel.py
  - tests/validation/149_test_spectral_transport_threshold_calibration_panel.py
  - kl_clustering_analysis/hierarchy_analysis/decomposition/gates/spectral_transport.py
  - tests/localization/35_test_gates_traversal.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_threshold_calibration_panel/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_threshold_calibration_panel/spectral_transport_threshold_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_threshold_calibration_panel/spectral_transport_threshold_pairwise.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_threshold_calibration_panel/spectral_transport_threshold_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_threshold_calibration_panel_strict_mp_low_grid/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_threshold_calibration_panel_strict_mp_low_grid/spectral_transport_threshold_summary.csv
tags:
  - source
  - diagnostics
  - traversal
  - spectral
  - threshold
---

# Spectral Transport Threshold Calibration Panel 2026-06-16

## Summary

`spectral_transport_threshold_calibration_panel.py` tests whether the spectral
transport pass-through guard has an operating point that preserves signal rows
while reducing selected-null false splitting. After correcting
`require_mp_blocks=True` to mean strict measured MP support, the default
threshold grid passes this targeted promotion condition on the three overlap
selected-family cases.

## Key Points

- The strict support correction changes the traversal interpretation:
  unmeasured no-MP paths no longer count as spectral pass-through support when
  `spectral_transport_require_mp_blocks=True`. With
  `require_mp_blocks=False`, floor-only paths remain neutral.
- The canonical threshold panel covers `overlap_part_4c_small`,
  `overlap_mod_4c_small`, and `overlap_heavy_4c_small_feat` under
  selected-null and signal roles, comparing the refined pass-through baseline
  to spectral thresholds `0.75`, `1.0`, `1.2`, and `1.5`.
- All four default thresholds are `threshold_candidate`: selected-null false
  splits drop from `1` baseline row to `0` candidate rows, and signal
  `min_delta_ari` and `mean_delta_ari` are both `0.0`.
- The extended strict low-grid run at thresholds `0.0`, `0.01`, `0.05`,
  `0.1`, `0.25`, `0.5`, `0.75`, `1.0`, `1.2`, and `1.5` also marks every
  threshold as `threshold_candidate`.
- The changed row is `overlap_mod_4c_small` selected-null: the refined
  baseline returns `7` clusters with ARI `0.0`, while the strict spectral
  support guard blocks one pass-through node and returns `1` cluster with ARI
  `1.0`.
- This threshold panel is still diagnostic calibration evidence. It supports
  targeted traversal promotion for the spectral pass-through guard, but it does
  not by itself establish broad production calibration or confidence over all
  suites.

## Evidence

- `spectral_transport_threshold_calibration_panel.py` writes row, pairwise,
  summary, and manifest outputs for baseline-vs-threshold comparisons.
- `149_test_spectral_transport_threshold_calibration_panel.py` covers summary
  status separation and output-writing behavior.
- `35_test_gates_traversal.py` now distinguishes strict MP-required support
  from the non-required neutral floor-only policy.
- The canonical output directory records `30` row records, `24` pairwise
  records, and `4` threshold summaries; all summaries are
  `threshold_candidate`.
- The strict low-grid output records the same conclusion across `10`
  thresholds.

## Links

- [[spectral-transport-passthrough-guard-20260616]]
- [[spectral-transport-overlap-dispatch-panel-20260616]]
- [[spectral-transport-promotion-gate-20260616]]
- [[selected-neighborhood-measurability-law]]
