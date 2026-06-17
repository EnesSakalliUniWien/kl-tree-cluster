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

## Links

- [[root-selected-binary-resolution-20260617]]
- [[root-selected-kernel-spectral-tail-law-20260617]]
- [[root-conditional-kernel-spectral-law]]
- [[selected-neighborhood-measurability-law]]
