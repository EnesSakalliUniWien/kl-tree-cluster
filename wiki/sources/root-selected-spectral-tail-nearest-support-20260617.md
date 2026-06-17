---
title: Root Selected Spectral Tail Nearest Support 2026-06-17
type: source
status: draft
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_selected_spectral_tail_nearest_support_panel.py
  - tests/validation/171_test_root_selected_spectral_tail_nearest_support_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_spectral_tail_nearest_support_mild_accumulated/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_spectral_tail_nearest_support_mild_accumulated/root_selected_spectral_tail_nearest_support_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_spectral_tail_nearest_support_mild_accumulated/root_selected_spectral_tail_nearest_support_summary.csv
tags:
  - source
  - diagnostics
  - root
  - spectral
  - support
---

# Root Selected Spectral Tail Nearest Support 2026-06-17

## Summary

`root_selected_spectral_tail_nearest_support_panel.py` localizes the remaining
selected-root spectral-tail support gap without promoting nearest neighbors to
calibration support. It reads the accumulated importance/topology feasibility
rows and the root-tail panel rows, then computes the nearest selected-null or
external-null support row for each observed root in the conditioning
coordinates \(T,A,E,B,H_u\). Exact same-stratum support remains the only route
to a conservative empirical spectral-tail p-value.

The mild accumulated run has exact selected-root tail support for `2/7`
targets and nearest support rows for all seven. The five unsupported targets
remain `fail_closed_nearest_support_only`. For the hard overlap roots, the
dominant nearest-support gap is usually selected-ratio action, while bandwidth
topology already matches. This refines the next mathematical task: the missing
law must condition the selected root construction and selected-ratio action
more sharply, not merely replay topology or add another bandwidth status.

## Key Points

- The panel writes `7` nearest-support rows and one summary row under
  `root_selected_spectral_tail_nearest_support_mild_accumulated`.
- `exact_support_target_count = 2`, matching the accumulated root-tail panel:
  `overlap_extreme_4c` and `overlap_unbal_6c_med` have exact support.
- `nearest_support_available_count = 7`, but `fail_closed_nearest_only_count =
  5`, because nearest support is diagnostic localization only.
- The unsupported hard roots have nearest calibration support with matching
  bandwidth/topology status, so the remaining gap is not generated topology
  replay.
- For `overlap_mod_4c_small`, the nearest support row is an
  `importance_two_block_external_null` row with bandwidth match but root-tail
  stratum mismatch; the dominant gap is selected-ratio action, and the old full
  commit overlay is still labeled
  `legacy_full_method_leaks_selected_null_root_risk`.
- For `overlap_mod_6c_med`, `overlap_part_4c_small`, and
  `overlap_unbal_4c_small`, the nearest-support diagnostic also identifies
  selected-ratio action as the dominant conditioning gap.
- `overlap_heavy_4c_small_feat` is different: the dominant nearest gap is edge
  action, with legacy internal MP-count changes but no calibrated root-tail
  support.

## Evidence

- The panel computes distance only over conditioning coordinates
  \(T,A,E,B,H_u\), leaving \(S_{\mathrm{root}}\) as a tail coordinate and
  reporting its nearest-support gap separately.
- The production status is
  `exact_support_available_defer_to_tail_panel` only when the existing
  root-tail panel already reports exact support. Otherwise it is
  `fail_closed_nearest_support_only`.
- The test file verifies that nearest support remains diagnostic when exact
  support is missing, and that exact support is counted separately from nearest
  support.

## Links

- [[root-selected-importance-tail-support-20260617]]
- [[root-tie-rank-target-conditioned-importance-frontier-20260617]]
- [[selected-neighborhood-measurability-law]]
