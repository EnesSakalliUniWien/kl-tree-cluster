---
title: Benchmark Pipeline Contract
type: analysis
status: reviewed
updated: 2026-06-04
sources:
  - benchmarks/README.md
  - benchmarks/shared/README.md
  - benchmarks/full/run.py
  - benchmarks/shared/pipeline.py
  - benchmarks/shared/util/case_inputs.py
  - benchmarks/shared/util/method_execution.py
  - benchmarks/shared/util/case_execution.py
  - benchmarks/shared/runners/kl_diffusion_runner.py
  - tests/integration/60_test_benchmark_methods_smoke.py
tags:
  - benchmark
  - pipeline
  - contract
---

# Benchmark Pipeline Contract

## Summary

The benchmark system has one active execution path with two layers:
`benchmarks/full/run.py` is the suite/report orchestrator, while
`benchmarks/shared/pipeline.py` and its utility modules are the reusable
case-and-method execution pipeline. Case generation, distance construction,
method dispatch, result rows, and report assembly should remain separate
pipeline stages.

## Details

The active full-benchmark order is:

1. `benchmarks/full/run.py` selects the case suite, methods, plotting policy,
   output directory, and resume state.
2. `run_case_with_optional_isolation()` optionally isolates one generated case
   in a subprocess when UMAP plotting is enabled.
3. `benchmark_cluster_algorithm()` runs the shared in-process pipeline for that
   one case.
4. `run_single_case()` generates case data once, then loops over selected
   method IDs and parameter sets.
5. `prepare_case_inputs()` validates the generated matrix contract and returns
   a typed `PreparedCaseInputs` object with shared distances only for declared
   precomputed KL tree distances or distance-matrix methods.
6. `run_single_method_once()` applies method-specific parameter resolution,
   records the KL tree-distance source, dispatches the method, computes
   metrics, and builds the canonical result row.
7. `run_clustering_result()` routes the method ID to the registered runner and
   normalizes ok/skip results.
8. `full/run.py` stamps the full-run case index, appends the CSV, runs
   relationship/failure diagnostics, and merges report PDFs.

The strict distance contract is:

- Case recipes do not carry generated data.
- `generate_case_data()` is the only case recipe to generated-data boundary.
- A precomputed KL tree distance is valid only when metadata sets
  `requires_precomputed_kl_distance=True` and provides
  `precomputed_distance_condensed`.
- Otherwise, KL-family feature-space methods compute their tree distance inside
  the method execution stage from the selected run parameters.
- Case recipe geometry belongs in `benchmarks/shared/cases/geometry.py`, so
  runner large-case decisions and report manifests use the same shape rules.
- `kl_diffusion` is a Hamming-diffusion method for binary or one-hot matrices.
  Continuous `FeatureSpace` inputs are not scored by that method; they are
  reported as explicit skip rows instead of producing misleading Hamming-on-float
  results. Continuous KL tree construction belongs to the precomputed Euclidean
  KL path or to a separately declared diffusion method with an explicit
  continuous metric.

The creation audit on 2026-06-04 found no generator-level geometry or metadata
corruption across the 110 default cases: case recipe geometry matched generated
metadata, forwarded continuous variants preserved baseline labels, categorical
and binary representations had binary values, and all precomputed-distance
cases declared their distance metric. The important interpretation caveats are
suite weighting and method support, not corrupted generated data. The full-run
console summary now labels mean ARI as ok-row-only and prints method status
counts so unsupported case/method combinations do not disappear from the
headline.

This means the benchmark folder should be read as domains, not as equivalent
entry points:

- `full/`: canonical suite/report orchestration.
- `shared/`: reusable pipeline, case definitions, generators, runners, result
  records, plotting helpers, and metrics.
- `diagnostics/`: oracle, calibration, spectral, failure, and post-run
  diagnostics that inspect failures and mathematical assumptions.
- `experiments/`: targeted standalone studies outside the default full suite,
  such as branch-length, multi-split, MNIST, and UMAP dataset experiments.
- `smoke/` and `regression/`: fast development checks, separate from both the
  canonical full-suite report runner and standalone experiments.

## Evidence

- `benchmarks/full/run.py` chooses suites, resumes CSV output, invokes one-case
  shared pipeline runs, and assembles PDFs.
- `benchmarks/shared/pipeline.py` coordinates shared case iteration,
  one-case execution, metrics, and plots.
- `benchmarks/shared/util/case_inputs.py` validates generated matrices and
  resolves shared distance objects.
- `benchmarks/shared/util/method_execution.py` records whether the KL tree
  distance came from a feature metric or a declared precomputed source.
- `benchmarks/shared/runners/kl_diffusion_runner.py` enforces the binary/one-hot
  Hamming diffusion input contract.
- `tests/integration/60_test_benchmark_methods_smoke.py` checks that continuous
  cases are skipped by `kl_diffusion` with an explicit contract reason.
- `benchmarks/shared/README.md` documents this execution order and the
  canonical distance contract.

## Links

- [[dimensional-gaussian-representation-diagnostic]]
- [[oracle-gate-path-diagnostic]]
- [[local-marchenko-pastur-rule]]
- [[open-mathematical-questions]]

## Open Questions

- Should old generated files under `benchmarks/results/` be promoted to
  `raw/assets/` when cited, then deleted locally to keep Finder views clean?
