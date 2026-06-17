---
title: Root Selected Validity Replay Panel 2026-06-17
type: source
status: draft
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_selected_validity_replay_panel.py
  - tests/validation/185_test_root_selected_validity_replay_panel.py
  - raw/assets/failure-fixtures/selected-root-pass-through-null-20260614/fixed_sibling_gate_profile_validation_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_validity_replay_profile_fixture_join/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_validity_replay_profile_fixture_join/root_selected_validity_replay_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_validity_replay_profile_fixture_join/root_selected_validity_replay_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_validity_replay_overlap_seven_signal_v1/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_validity_replay_overlap_seven_signal_v1/fixed_sibling_gate_profile_validation_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_validity_replay_overlap_seven_signal_v1/fixed_sibling_gate_profile_validation_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_validity_replay_overlap_seven_signal_v1/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_validity_replay_overlap_seven_signal_v1/root_selected_validity_replay_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_validity_replay_overlap_seven_signal_v1/root_selected_validity_replay_summary.csv
tags:
  - source
  - diagnostics
  - root
  - topology
  - selected-tail
---

# Root Selected Validity Replay Panel 2026-06-17

## Summary

`root_selected_validity_replay_panel.py` separates two questions that were
previously easy to conflate:

1. whether the selected root bifurcation is a stable/coherent object; and
2. whether the spectral tail is calibrated after conditioning on that root.

The panel joins selected-root tail rows to root replay evidence from
feature-subsample stability, selected-root permutation, or explicit topology
family replay rows. Its final usability verdict is a conjunction: a root is
usable only when replay supports the selected bifurcation and the root spectral
tail has selected-null support.

## Key Points

- The root-validity output is `root_validity_status`.
- The tail-calibration output remains `root_tail_inference_status`.
- The combined method-facing output is `selected_root_usability_status`.
- A calibrated tail inside an unstable or unmeasured root event fails closed.
- A replay-supported root with missing spectral-tail support also fails closed.
- Explicit topology-family rows can count plausible alternative roots through
  `plausible_alternative_root_count`.
- The profile-fixture join writes `7` rows: `1/7` fails root validity from the
  existing `overlap_mod_4c_small` stability fixture, `6/7` remain unmeasured,
  and `0/7` are usable selected roots.
- The same join has `2/7` tail-calibrated roots, but both still fail closed
  because root validity is unmeasured for those targets.
- The all-seven signal-role replay under
  `fixed_coordinate_selective_root_v1` gives complete target coverage:
  `2/7` roots are validity-supported, `5/7` fail root stability, and no root
  is unmeasured.
- Only `overlap_unbal_6c_med` is both root-validity supported and
  selected-tail calibrated. `overlap_part_4c_small` is root-validity supported
  but still tail-support missing. `overlap_extreme_4c` is tail-calibrated
  but fails root-validity replay.

This implements the sharper root-selection uncertainty distinction:
calibration inside \(G_{\hat r}\) is not evidence that \(\hat r\) is the right
first bifurcation.

## Evidence

- `185_test_root_selected_validity_replay_panel.py` verifies that a valid root
  with selected-tail support is usable.
- The tests verify that an unstable root fails closed even if the tail row is
  calibrated.
- The tests verify that a valid root still fails closed when selected-root tail
  support is missing.
- The tests verify selected-root permutation failure, missing replay evidence,
  explicit topology-family alternative counting, and runner output writing.
- `root_selected_validity_replay_summary.csv` records
  `root_validity_failed_count = 1`,
  `root_validity_unmeasured_count = 6`, and
  `usable_selected_root_count = 0`.
- `root_selected_validity_replay_rows.csv` marks `overlap_mod_4c_small` as
  `root_validity_failed_feature_subsample_replay` with method action
  `fail_closed_root_unstable_under_topology_replay`.
- `root_validity_replay_overlap_seven_signal_v1` records signal-role root
  stability and selected-root permutation evidence for all seven overlap
  root-tail targets. Mean root-stability ARI is below the `0.24` guard
  threshold for `overlap_extreme_4c`, `overlap_heavy_4c_small_feat`,
  `overlap_mod_4c_small`, `overlap_mod_6c_med`, and
  `overlap_unbal_4c_small`.
- `root_selected_validity_replay_overlap_seven_signal_v1` records
  `root_validity_supported_count = 2`, `root_validity_failed_count = 5`,
  `root_validity_unmeasured_count = 0`, `tail_calibrated_count = 2`, and
  `usable_selected_root_count = 1`.
- The usable selected-root row is `overlap_unbal_6c_med` with method action
  `use_selected_root_tail_with_validity_annotation`.

## Links

- [[root-selected-binary-resolution-20260617]]
- [[root-selected-kernel-spectral-tail-law-20260617]]
- [[root-conditional-kernel-spectral-law]]
- [[selected-neighborhood-measurability-law]]
