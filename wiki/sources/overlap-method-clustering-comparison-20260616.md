---
title: Overlap Method Clustering Comparison 2026-06-16
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_method_clustering_comparison.py
  - tests/validation/calibration/overlap/142_test_overlap_method_clustering_comparison.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_method_clustering_comparison
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_method_clustering_comparison_binary_suite
tags:
  - source
  - diagnostics
  - clustering
  - overlap
  - benchmarks
---

# Overlap Method Clustering Comparison 2026-06-16

## Summary

`overlap_method_clustering_comparison.py` pairs checkpoint assignment files from
`fixed_coordinate_conditional_topology_diagnostic_v1` and
`fixed_coordinate_global_passthrough_refined_v1` on shared
`case_id,data_role,replicate` keys. It reports per-method fragmentation and
label-invariant method-to-method partition agreement, then writes row,
pairwise, summary, and manifest outputs.

## Key Points

- The expanded overlap run covers `140` assignment files and `70` paired
  method comparisons across seven overlap cases and selected-null/signal roles.
- The broader binary-suite overlap-only run uses `--case-prefix overlap_` and
  covers `168` assignment files and `84` paired method comparisons across
  fourteen overlap cases.
- In both runs, the conditional-topology profile is never less fragmented than
  the refined global pass-through profile. Expanded overlap has `26/70`
  `left_more_fragmented` pairs and `44/70` same-fragment-count pairs; the
  binary-suite overlap slice has `25/84` `left_more_fragmented` pairs and
  `59/84` same-fragment-count pairs.
- The fragmentation difference concentrates in selected-null rows. In expanded
  overlap, selected-null deltas average `+1.7143` clusters for the conditional
  profile, with run-row ARI `0.1714` versus `0.4857` for the refined profile.
  In the binary-suite overlap slice, selected-null deltas average `+1.3333`
  clusters, with run-row ARI `0.4286` versus `0.7857`.
- Signal rows are much closer. Expanded overlap signal pairs have mean
  method-to-method partition ARI `0.9192` and mean run-row ARI delta
  `-0.0051`; binary-suite signal pairs have mean partition ARI `0.9517` and
  mean run-row ARI delta `+0.0004`.
- This supports the current merged method contract: the refined selected-family
  pass-through guard is doing real work against selected-null over-splitting,
  while signal recovery differences are localized and should be studied through
  topology/bandwidth bottleneck diagnostics rather than by removing the guard
  globally.

## Evidence

- `benchmarks/diagnostics/calibration/overlap/overlap_method_clustering_comparison.py`
  implements the paired comparison.
- `tests/validation/calibration/overlap/142_test_overlap_method_clustering_comparison.py` validates
  null truth metrics, signal no-truth behavior, paired method deltas, and output
  file creation.
- The two result manifests cite the exact checkpoint roots, method names,
  optional case filter, assignment file counts, paired run counts, and output
  paths.

## Links

- [[selected-neighborhood-bottleneck-law]]
- [[selected-neighborhood-distribution-panel-20260615]]
