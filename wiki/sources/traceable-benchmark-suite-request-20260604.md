---
title: Traceable Benchmark Suite Request 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - raw/inbox/traceable-benchmark-suite-request-20260604.txt
tags:
  - source
  - benchmarks
  - calibration
  - traceability
---

# Traceable Benchmark Suite Request 2026-06-04

## Summary

This request argues that the next Tree-Break Selection benchmark program should be a
traceable mathematical benchmark suite rather than a larger collection of ARI
tables. Each case should target a specific mathematical contract: selected-tail
calibration, internal support, MP/projection, categorical or continuous
covariance, traversal FDR, pass-through, phylogenetic null behavior, or
real-data evidence.

## Key Points

- Benchmark cases should record `benchmark_role`, `mathematical_target`,
  expected failure modes, required trace columns, admissibility rules, and
  success criteria.
- The first proposed method-proof cases cover barycentric balance and
  leverage, BBP/spiked MP behavior, high-cardinality categorical simplex
  boundaries, Dirichlet-multinomial overdispersion, continuous `p >> n`
  low-rank covariance, fail-closed support, phylogenetic Brownian nulls, and
  pass-through replacement.
- A benchmark run should emit locked trace artifacts: `manifest.json`,
  `case_summary.csv`, `node_decision_trace.csv`,
  `selected_sibling_records.csv`, and `failure_attribution.csv`.
- The requested math inference tool is deterministic attribution, not symbolic
  proof: it maps trace rows to the first failed mathematical layer in a
  predeclared hierarchy.
- Metamorphic checks are proposed for sample and feature permutation,
  categorical relabeling, noise-feature duplication, proportional null-sample
  addition, continuous rotations, left/right child swaps, and zero-signal
  collapse.

## Evidence

- `raw/inbox/traceable-benchmark-suite-request-20260604.txt` is the ingested
  benchmark-suite design request.
- `benchmarks/shared/cases/method_proof.py` implements the first method-proof
  case registry with trace metadata.
- `benchmarks/diagnostics/math_trace/infer_benchmark_math.py` implements the
  first deterministic trace-to-math inference entrypoint.

## Links

- [[benchmark-pipeline-contract]]
- [[selected-hierarchy-null-support-contract]]
- [[local-marchenko-pastur-rule]]
- [[traceable-mathematical-benchmark-suite]]
