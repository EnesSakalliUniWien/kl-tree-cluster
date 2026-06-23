---
title: Legacy c2ef9a69 Edge Alpha Comparison 2026-06-17
type: source
status: reviewed
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/legacy_c2ef9a69_edge_alpha_comparison_panel.py
  - tests/validation/190_test_legacy_c2ef9a69_edge_alpha_comparison_panel.py
  - benchmarks/diagnostics/calibration/legacy_c2ef9a69_method_comparison_panel.py
  - benchmarks/shared/runners/legacy_commit_runner.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_edge_alpha_overlap_comparison_20260617/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_edge_alpha_overlap_comparison_20260617/legacy_c2ef9a69_edge_alpha_comparison_pairwise.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_edge_alpha_overlap_comparison_20260617/legacy_c2ef9a69_edge_alpha_comparison_alpha_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_edge_alpha_overlap_comparison_20260617/legacy_c2ef9a69_edge_alpha_comparison_tradeoff_summary.csv
tags:
  - source
  - diagnostics
  - legacy
  - edge-alpha
---

# Legacy c2ef9a69 Edge Alpha Comparison 2026-06-17

## Summary

This diagnostic runs current `tbs` and the full copied legacy
`tbs_legacy_c2ef9a69` method over the same seven overlap/root-tail cases used
by the selected-root work, both selected-null and signal roles, one replicate,
and the edge-alpha grid `0.0001, 0.0003, 0.001, 0.003, 0.01`.

The result rejects the shortcut that legacy's signal gain is explained by a
clean edge-alpha choice. Every tested edge alpha has one legacy signal gain and
at least one legacy extra selected-null false split. The panel therefore keeps
legacy as a power-source witness and safety-warning comparator, not as a
calibrated production rule.

## Key Points

- The run writes `140` method rows, `70` pairwise rows, `10` alpha-role summary
  rows, and `5` alpha tradeoff rows under
  `legacy_c2ef9a69_edge_alpha_overlap_comparison_20260617`.
- At every tested edge alpha, `signal_legacy_gain_count = 1`. The signal gain is
  consistently `overlap_unbal_4c_small`, where current has three clusters and
  ARI `0.387667`, while legacy has four clusters and ARI `0.471217`.
- At every tested edge alpha, selected-null legacy has extra false splits while
  current has zero selected-null false splits in the same grid.
- Selected-null extra legacy false splits increase with alpha:
  `1` at `0.0001`, `1` at `0.0003`, `2` at `0.001`, `3` at `0.003`, and
  `3` at `0.01`.
- The strictest alpha `0.0001` still leaks selected null on
  `overlap_mod_6c_med`, while retaining the same legacy signal gain. Lowering
  edge alpha therefore does not create an admissible legacy rule on this panel.
- The default alpha `0.001` reproduces the earlier root-tail comparison pattern:
  one signal gain and two selected-null extra false splits.

## Evidence

- `legacy_c2ef9a69_edge_alpha_comparison_tradeoff_summary.csv` labels all five
  edge alpha rows `legacy_power_not_admissible_extra_null_leak`.
- `legacy_c2ef9a69_edge_alpha_comparison_pairwise.csv` records the selected-null
  leak cases and the signal gain case separately, preventing a summary that
  collapses them into a single "legacy better" statement.
- `190_test_legacy_c2ef9a69_edge_alpha_comparison_panel.py` verifies the
  no-shortcut contract: a row with both legacy signal gain and selected-null
  extra false split must be classified as
  `legacy_power_not_admissible_extra_null_leak`.

## Links

- [[legacy-c2ef9a69-root-tail-overlap-comparison-20260617]]
- [[legacy-c2ef9a69-method-package-20260616]]
- [[root-conditional-kernel-spectral-law]]
- [[spectral-vs-bandwidth-tradeoff-panel-20260616]]
