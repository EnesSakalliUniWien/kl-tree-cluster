---
title: Spectral Transport Promotion Gate 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/spectral_transport_promotion_gate.py
  - tests/validation/148_test_spectral_transport_promotion_gate.py
  - kl_clustering_analysis/hierarchy_analysis/decomposition/gates/spectral_transport.py
  - benchmarks/diagnostics/calibration/spectral_transport_threshold_calibration_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate/spectral_transport_promotion_components.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate/spectral_transport_promotion_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel/spectral_transport_overlap_dispatch_pairwise.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case_current/selected_family_traversal_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_threshold_calibration_panel/spectral_transport_threshold_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_overlap_dispatch_panel_promoted/spectral_transport_overlap_dispatch_pairwise.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_overlap_three_case_promoted/selected_family_traversal_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted/spectral_transport_promotion_components.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted/spectral_transport_promotion_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_family_traversal_spectral_transport_promoted_replicates/selected_family_traversal_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted_replicates/spectral_transport_promotion_components.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted_replicates/spectral_transport_promotion_summary.csv
tags:
  - source
  - diagnostics
  - traversal
  - spectral
  - promotion
---

# Spectral Transport Promotion Gate 2026-06-16

## Summary

`spectral_transport_promotion_gate.py` turns the current overlap evidence into
an explicit traversal-promotion decision. The gate requires three components:
standard-dispatch signal retention, selected-family signal retention, and
selected-null false-split reduction. After correcting strict MP-required
support so unmeasured no-MP paths no longer support pass-through, the promoted
entry point is `kl_spectral_transport_passthrough` /
`fixed_coordinate_spectral_transport_passthrough_v1`. The one-replicate gate
passes, but the 50-replicate selected-family gate fails signal retention. The
current default promotion decision is therefore `diagnostic_only_not_promoted`.

## Key Points

- The standard-dispatch component compares
  `kl_spectral_transport_passthrough` against
  `kl_global_passthrough_refined_diagnostic`. It passes: `3` paired rows are
  ok, minimum ARI delta is `0.0`, mean ARI delta is `0.0`, and mean partition
  ARI between methods is `1.0`.
- The one-replicate selected-family signal component passes, but the
  50-replicate component fails: `4/150` paired signal rows regress, minimum
  signal delta ARI is `-0.822005`, and mean signal delta ARI is `-0.006734`.
- The selected-family null component now passes. The refined baseline has `1`
  selected-null false-split row and the spectral candidate has `0`, so
  false-split reduction is `1` against the required reduction of `1`.
- The changed selected-null row is `overlap_mod_4c_small`: the spectral profile
  blocks one pass-through node and returns `1` cluster instead of `7`.
- The one-replicate summary decision is `promotion_admissible`, with `3`
  required components passed and `0` failed.
- The promotion gate defaults now point at the promoted standard-dispatch and
  50-replicate selected-family output directories, so a default rerun evaluates
  the stronger replicate evidence for the promoted method/profile pair.
- The default 50-replicate summary decision is `diagnostic_only_not_promoted`,
  blocked by `selected_family_signal_retention`.

## Evidence

- `spectral_transport_promotion_gate.py` reads the current standard-dispatch
  pairwise panel and selected-family traversal rows, evaluates required
  promotion components, and writes component and summary CSVs.
- `148_test_spectral_transport_promotion_gate.py` covers the passing case, the
  null-false-split failure case, and the output-writing contract.
- The threshold panel records that default thresholds `0.75`, `1.0`, `1.2`,
  and `1.5` all satisfy the same selected-family threshold-candidate
  condition.
- The gate output under
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted/`
  records `3` required components passed and `0` required components failed.
- The replicate gate output under
  `raw/assets/benchmark-results/specific_small_method_benchmark_20260615/spectral_transport_promotion_gate_promoted_replicates/`
  records `2` required components passed and `1` required component failed.

## Links

- [[spectral-transport-overlap-dispatch-panel-20260616]]
- [[spectral-transport-threshold-calibration-panel-20260616]]
- [[spectral-transport-promoted-replicate-panel-20260616]]
- [[spectral-transport-passthrough-guard-20260616]]
- [[selected-neighborhood-measurability-law]]
