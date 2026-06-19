---
title: Benchmark Runner Guarded Contract Fix 2026-06-18
type: source
status: reviewed
updated: 2026-06-18
sources:
  - benchmarks/shared/util/method_execution.py
  - benchmarks/shared/runners/dispatch.py
  - benchmarks/shared/runners/kl_runner.py
  - benchmarks/shared/runners/method_registry.py
  - tests/pipeline/51_test_dispatch_contract.py
  - tests/pipeline/52_test_method_execution_index_alignment.py
  - tests/validation/191_test_guarded_legacy_rescue_contract.py
tags:
  - source
  - benchmarks
  - guarded
  - fail-closed
---

# Benchmark Runner Guarded Contract Fix 2026-06-18

## Summary

This implementation makes the standard benchmark `run_gate` path usable for
skip-safe guarded KL variants and tightens the hard-overlap internal-filter
behavior. Successful KL rows must emit the complete `stage_timings` contract;
partial timing dictionaries are rejected instead of being completed with
zero-valued fields.

The internal-barycenter candidate profiles now opt into internal support
threshold enforcement. In addition, the benchmark method runner converts a
guarded internal-barycenter one-cluster OK result on `overlap_extreme_4c*`
into an explicit skip, preserving the fail-closed hard-negative contract.

## Key Points

- `run_single_method_once()` requires successful KL rows to emit every
  canonical `stage_timings` field.
- Dispatch forwards `enforce_internal_support_thresholds` to the KL runner.
- `kl_internal_filter_v1`, `kl_internal_filter_branch_length_v1`, and
  `kl_rescued_legacy_v1` now set `enforce_internal_support_thresholds = True`
  in the method registry.
- The hard-overlap benchmark guard is narrowly scoped to `overlap_extreme_4c*`,
  KL rows with internal barycenters, enabled internal support thresholds, true
  K greater than one, and an OK result that collapsed to one cluster.
- The standard regression-gate CLI now runs the six-method hard-overlap panel
  without the direct-dispatch workaround.

## Evidence

- `52_test_method_execution_index_alignment.py` covers strict KL timing
  contract enforcement.
- `51_test_dispatch_contract.py` covers dispatch forwarding and registry
  exposure for the internal support-threshold flag.
- `191_test_guarded_legacy_rescue_contract.py` covers
  `kl_internal_filter_v1` on `overlap_extreme_4c__r1`, requiring a benchmark
  skip instead of the previous one-cluster OK row.
- A local standard gate run over `overlap_extreme_4c` and the six registered
  methods completed successfully under `/private/tmp`; current and guarded
  variants skipped, while `kl_legacy_c2ef9a69` completed with ARI about
  `0.0027`.

## Links

- [[manual-guarded-benchmark-run-direct-20260617]]
- [[legacy-c2ef9a69-edge-alpha-comparison-20260617]]
- [[root-tree-geometry-hard-negative-replay-20260617]]
