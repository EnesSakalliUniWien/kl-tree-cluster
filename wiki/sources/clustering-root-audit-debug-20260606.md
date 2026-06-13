---
title: Clustering Root Audit Debug 2026-06-06
type: source
status: reviewed
updated: 2026-06-07
sources:
  - benchmarks/diagnostics/failure/debug_trace.py
  - benchmarks/shared/relationship_analysis.py
  - tests/pipeline/62_test_debug_trace_contract.py
  - tests/pipeline/61_test_benchmark_relationship_analysis.py
  - benchmarks/results/diagnostics/clustering_debug_20260606_corrected_root_audit/clustering_debug_report.md
  - benchmarks/results/diagnostics/clustering_debug_20260606_corrected_root_audit/failure_report.md
  - benchmarks/results/diagnostics/clustering_debug_20260606_corrected_root_audit/benchmark_relationship_augmented_rows.csv
  - benchmarks/results/run_20260605_224254Z_full_big/benchmark_relationship_report.md
  - benchmarks/results/run_20260605_224254Z_full_big/failure_report.md
  - benchmarks/results/run_20260605_224254Z_full_big/full_benchmark_report.pdf
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_kmin2_20260606/mixed_null_signal_labeled_nodes.csv
tags:
  - source
  - benchmarks
  - diagnostics
  - clustering
---

# Clustering Root Audit Debug 2026-06-06

## Summary

The full benchmark root-failure diagnosis had a software interpretation bug:
the exported tree audit stores the root sibling test on the root row, but the
failure and relationship analyzers were reading the root's child rows to decide
whether the root split was rejected. The corrected analyzers now read the root
row directly. On 2026-06-07 the canonical full benchmark derived reports were
regenerated with the corrected analyzer.

## Key Points

- Corrected KL root outcome over `92` ok rows: `79` root-accepted rows have
  mean ARI `0.901264`, while `13` root-rejected rows have mean ARI `0.307692`.
- The low-ARI KL rows are not all root-rejection cases. Nine low-ARI rows are
  root rejections, but four low-ARI rows have accepted root splits and then
  stall below the root: `dim_consolidated_4c_72f`,
  `dim_consolidated_4c_272f`, `dim_diffuse_6c_536f`, and `sbm_moderate`.
- In those four post-root stall rows, deeper sibling split rate is `0.0` even
  though median deeper edge action remains `2.0`. This points to a recursive
  sibling calibration/traversal failure, not a root-edge failure.
- The skip profile is unchanged: `92` KL rows are ok, while `28` fail closed,
  mostly due to missing strict-null or stopped-edge sibling calibration
  support; two dense continuous cases hit explicit covariance contracts.
- The aligned `k_min=2` mixed null/signal panel remains the right companion
  diagnostic for the canonical full benchmark. It shows that edge/KAK geometry
  predicts sibling truth labels, but the production traversal still lacks a
  selected-tree sibling calibration rule that can convert deeper edge action
  into admissible recursive splits.
- The canonical files
  `benchmarks/results/run_20260605_224254Z_full_big/benchmark_relationship_report.md`,
  `benchmarks/results/run_20260605_224254Z_full_big/failure_report.md`, and
  `benchmarks/results/run_20260605_224254Z_full_big/full_benchmark_report.pdf`
  now contain the corrected root-row interpretation.

## Evidence

- `benchmarks/diagnostics/failure/debug_trace.py` now reads root sibling
  p-values and decisions from the root row.
- `benchmarks/shared/relationship_analysis.py` now uses the same root-row
  interpretation for `audit_root_split_rejected` and
  `audit_root_sibling_p`.
- `tests/pipeline/62_test_debug_trace_contract.py` and
  `tests/pipeline/61_test_benchmark_relationship_analysis.py` cover the
  corrected root-row audit contract.
- `benchmarks/results/diagnostics/clustering_debug_20260606_corrected_root_audit/clustering_debug_report.md`
  contains the corrected status, root outcome, low-ARI, and post-root-stall
  summaries.

## Links

- [[full-benchmark-run-20260606]]
- [[mixed-null-signal-geometry-validation-20260606]]
- [[recursive-pvalue-geometry-20260606]]
- [[top-down-traversal]]
