---
title: Manual Guarded Benchmark Run Direct 2026-06-17
type: source
status: reviewed
updated: 2026-06-18
sources:
  - raw/assets/benchmark-results/manual_guarded_benchmark_run_direct_20260617/manifest.json
  - raw/assets/benchmark-results/manual_guarded_benchmark_run_direct_20260617/rows.csv
  - raw/assets/benchmark-results/manual_guarded_benchmark_run_direct_20260617/summary_by_method.csv
  - raw/assets/benchmark-results/manual_guarded_benchmark_run_direct_20260617/summary_by_case.csv
  - tests/pipeline/51_test_dispatch_contract.py
tags:
  - source
  - benchmarks
  - legacy
  - guarded
---

# Manual Guarded Benchmark Run Direct 2026-06-17

## Summary

This user-run direct-dispatch smoke compares current TBS, legacy
`c2ef9a69`, internal-filter variants, bandwidth-context TBS, and rescued
legacy behavior over six small target cases: two binary replicates, two clear
categorical replicates, and two `overlap_extreme_4c` replicates.

The run is useful as a narrow method-triage panel, not as a production
admissibility result. On completed rows, `tbs_internal_filter_branch_length_v1`
has the strongest mean ARI and exact-cluster-count profile. The hard overlap
case mainly confirms the strict calibration-support story: guarded variants
skip when sibling empirical-null support is missing, while legacy-style
methods return low-ARI splits or one-cluster outputs without solving the
support problem.

## Key Points

- The manifest records six methods and six cases, producing `36` rows under
  `manual_guarded_benchmark_run_direct_20260617`.
- `tbs_internal_filter_branch_length_v1` records `4` ok rows, `2` skips,
  `3` exact-K rows, and mean ARI `0.926991` on completed rows.
- `tbs_internal_filter_v1` records `5` ok rows, `1` skip, `3` exact-K rows,
  and mean ARI `0.741593`. Its second hard-overlap replicate does not skip;
  it returns one cluster with ARI `0.0`.
- `kl_current` and `tbs_bandwidth_context_v1` are identical on this run:
  `4` ok rows, `2` skips, `1` exact-K row, and mean ARI `0.791362`.
- `tbs_legacy_c2ef9a69` completes all six rows with `2` exact-K rows and mean
  ARI `0.611973`, but on `overlap_extreme_4c` it returns five clusters with
  ARI `0.003107`.
- `tbs_rescued_legacy_v1` completes all six rows with no exact-K rows and mean
  ARI `0.422755`; it under-splits the first binary replicate and both hard
  overlap replicates to one cluster.
- The hard overlap rows show the important safety distinction: current,
  bandwidth-context, and branch-length internal-filter variants skip both hard
  overlap replicates with the strict empirical-null calibration-support error,
  while legacy and rescued variants complete with near-zero ARI. Plain
  `tbs_internal_filter_v1` skips the first hard-overlap replicate but returns
  a one-cluster ARI `0.0` result on the second.

## Evidence

- `summary_by_method.csv` records the method-level ok, skip, exact-K, ARI,
  found-cluster, and purity aggregates.
- `rows.csv` records the calibration-support skip reason: no strict-null or
  stopped-edge empirical-null calibration records with positive weight and
  `599` selected non-null positive-weight records for the guarded
  hard-overlap skips.
- `rows.csv` also separates fail-closed skips from completed low-information
  outputs: `tbs_internal_filter_v1`, `tbs_legacy_c2ef9a69`, and
  `tbs_rescued_legacy_v1` all return one-cluster or near-zero-ARI completed
  rows on at least one hard-overlap replicate.
- The direct runner was used because the gate wrapper currently enforces a
  stricter TBS stage-timing contract than all of these skip-safe or ok paths
  expose.

## Links

- [[legacy-c2ef9a69-method-package-20260616]]
- [[legacy-c2ef9a69-edge-alpha-comparison-20260617]]
- [[root-tree-geometry-hard-negative-replay-20260617]]
- [[selected-neighborhood-internal-spectral-flow-conditional-energy-20260617]]
