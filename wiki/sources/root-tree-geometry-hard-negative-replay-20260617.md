---
title: Root Tree Geometry Hard Negative Replay 2026-06-17
type: source
status: draft
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/center/root_tree_geometry_hard_negative_replay_panel.py
  - tests/validation/calibration/root/center/186_test_root_tree_geometry_hard_negative_replay_panel.py
  - tests/pipeline/51_test_dispatch_contract.py
  - benchmarks/shared/runners/tbs_runner.py
  - benchmarks/shared/runners/dispatch.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tree_geometry_hard_negative_overlap_extreme_4c_v1
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tree_geometry_hard_negative_overlap_extreme_4c_legacy_c2ef9a69_v1
tags:
  - source
  - diagnostics
  - root
  - topology
  - hard-negative
---

# Root Tree Geometry Hard Negative Replay 2026-06-17

## Summary

`root_tree_geometry_hard_negative_replay_panel.py` replays the warning case
`overlap_extreme_4c` across alternative selected-tree geometries. The panel
uses the same signal seed and `fixed_coordinate_selective_root_v1` guard
settings as the prior all-seven root-validity artifact, but varies tree
construction and distance:

- linkage/Hamming/average,
- linkage/Hamming/complete,
- linkage/Jaccard/average,
- linkage/Rogers-Tanimoto/average,
- neighbor joining/Hamming/MAD root,
- neighbor joining/Jaccard/MAD root.

The diagnostic treats `overlap_extreme_4c` as a hard negative for root-tail
rescue. A geometry leaks only if the selected root becomes validity-supported
and the root split is open. Missing selected-root permutation evidence remains
fail-closed.

## Key Points

- The shared TBS runner now accepts explicit
  `root_stability_tree_distance_metric`,
  `root_stability_tree_linkage_method`,
  `root_selective_permutation_guard_tree_distance_metric`, and
  `root_selective_permutation_guard_tree_linkage_method`.
- The dispatcher forwards those root replay geometry parameters from benchmark
  parameter dictionaries.
- The `overlap_extreme_4c` replay ran `6` geometries with no skips.
- `0/6` geometries are root-validity supported.
- `0/6` geometries leak the hard-negative control.
- All `6/6` geometries are classified as
  `hard_negative_control_blocked_root_unstable`.
- The panel now also removes the privileged root direction and scans every
  undirected edge bipartition in the selected tree. This rootless geometry check
  finds `0/6` truth-aligned current geometries; the best unrooted edge-cut ARI
  is only `0.004682`.
- Root-stability mean ARI ranges from `-0.002052` to `0.012785`, far below the
  `0.24` guard threshold.
- Selected-root permutation p-values, when measured, range from `0.10` to
  `0.23`, above the `0.01` guard threshold.
- Some geometries still fragment, for example Hamming average linkage finds
  `6` clusters and Rogers-Tanimoto average linkage finds `6` clusters, but the
  root partition truth ARI remains near zero.
- The same benchmark was rerun with the full copied old commit method
  `tbs_legacy_c2ef9a69`. The legacy runner supports only linkage trees, so both
  neighbor-joining geometries are explicit skips.
- On the four linkage geometries, the old method finds one cluster for
  Hamming average, Hamming complete, and Jaccard average linkage. It fragments
  on Rogers-Tanimoto average linkage with `5` found clusters and ARI
  `-0.002225`, while the root partition truth ARI is still only `0.000663`.
- The same rootless edge scan on the legacy geometries also finds `0/4`
  truth-aligned linkage geometries; the best unrooted edge-cut ARI is again
  only `0.004682`. The two neighbor-joining geometries remain explicit legacy
  skips.
- The legacy result therefore does not rescue `overlap_extreme_4c`; it shows
  the old method can either fail closed by under-splitting or fragment without
  a valid-root guard.

The interpretation is sharper than root-object invalidity alone. Removing the
root direction does not reveal a good coarse bipartition in this warning case.
The selected first bifurcation is unstable, and the underlying selected tree
geometry also lacks a truth-aligned undirected edge split under the tested
distance/linkage/rooting families.

## Evidence

- `186_test_root_tree_geometry_hard_negative_replay_panel.py` checks geometry
  parsing, fail-closed unstable-root classification, leak classification, and
  summary failure when any geometry leaks.
- `51_test_dispatch_contract.py` verifies that root replay distance/linkage
  parameters are forwarded through the shared TBS dispatcher.
- `root_tree_geometry_hard_negative_replay_summary.csv` records
  `row_count = 6`, `ok_geometry_count = 6`,
  `root_validity_supported_count = 0`, `hard_negative_leak_count = 0`, and
  `rootless_truth_aligned_geometry_count = 0`,
  `max_best_unrooted_edge_cut_truth_ari = 0.004682`, and
  `summary_status = hard_negative_control_supported`.
- `root_tree_geometry_hard_negative_replay_rows.csv` records the per-geometry
  root partitions, root-stability means, selected-root permutation p-values,
  found cluster counts, ARI, best unrooted edge-cut ARI, rootless geometry
  status, and hard-negative method action.
- `root_tree_geometry_hard_negative_overlap_extreme_4c_legacy_c2ef9a69_v1`
  records `row_count = 6`, `ok_geometry_count = 4`,
  `skipped_geometry_count = 2`, `hard_negative_leak_count = 0`, and
  `legacy_invalid_root_fragmentation_count = 1`, while its rootless edge scan
  also records `rootless_truth_aligned_geometry_count = 0`.

## Links

- [[root-selected-validity-replay-panel-20260617]]
- [[root-selected-kernel-spectral-tail-law-20260617]]
- [[root-conditional-kernel-spectral-law]]
- [[selected-neighborhood-measurability-law]]
