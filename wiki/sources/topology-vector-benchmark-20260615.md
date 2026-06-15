---
title: Topology Vector Benchmark 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - benchmarks/shared/runners/dispatch.py
  - tests/pipeline/51_test_dispatch_contract.py
  - raw/assets/benchmark-results/topology_vector_benchmark_20260615/regression_gate_kl_profile/regression_gate_kl_profile_comparison.csv
  - raw/assets/benchmark-results/topology_vector_benchmark_20260615/regression_gate_kl_profile/regression_gate_kl_profile_summary.csv
  - raw/assets/benchmark-results/topology_vector_benchmark_20260615/regression_gate_kl_profile/regression_gate_kl_profile_by_case.csv
  - raw/assets/benchmark-results/topology_vector_benchmark_20260615/regression_gate_kl_profile/regression_gate_kl_profile_case_deltas.csv
  - raw/assets/benchmark-results/topology_vector_benchmark_20260615/regression_gate_kl_profile/manifest.json
  - raw/assets/benchmark-results/topology_vector_benchmark_20260615/topology_vector_benchmark_summary.csv
  - raw/assets/benchmark-results/topology_vector_benchmark_20260615/topology_vector_benchmark_report.md
  - raw/assets/benchmark-results/topology_vector_benchmark_20260615/manifest.json
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_topology_law/overlap_conditional_topology_law_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_topology_law/overlap_conditional_topology_law_benchmark_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/conditional_topology_law/manifest.json
  - raw/assets/benchmark-results/conditional_topology_law_20260615/regression_gate/regression_gate_comparison.csv
  - raw/assets/benchmark-results/conditional_topology_law_20260615/regression_gate/regression_gate_metadata.json
tags:
  - source
  - benchmark
  - diagnostics
  - overlap
  - traversal
  - topology
---

# Topology Vector Benchmark 2026-06-15

## Summary

The topology-vector benchmark runs the standard regression-gate cases with two
KL variants and summarizes the already generated context-negative topology-law
diagnostics. A small dispatcher fix forwards KL gate-profile parameters from
benchmark param grids into the KL runner, allowing
`fixed_coordinate_global_passthrough_refined_v1` to be benchmarked as a normal
KL parameter variant.

## Key Points

- The regression gate covers `17` historically sensitive cases, including
  categorical, phylogenetic, SBM, Gaussian, and overlap cases.
- Default projected-Wald KL has `11/17` ok rows and `6/17` skips. Its ok-only
  mean ARI is `0.600384`, but skip-as-zero mean ARI is `0.388484`.
- `fixed_coordinate_global_passthrough_refined_v1` has `17/17` ok rows and no
  skips. Its mean ARI is `0.445742`, median ARI is `0.467063`, and exact-K
  count is `5/17`.
- The refined profile improves skip-as-zero mean ARI by `0.057258` and exact-K
  count by `2`, but it is worse on ok-only mean ARI and under-splits several
  easy non-overlap cases, including `gauss_extreme_noise_3c` and
  `sbm_moderate`.
- The topology-vector diagnostics remain stronger than the clustering profile
  benchmark. The context-negative topology law top-ranks the focused truth row
  with posterior log-odds margin `4.954869`.
- The sensitivity audit reports `60/65` separating profile-weight
  combinations; topology-only and outgoing-topology-only separate at all
  tested context weights, while selected-family plus context separates at
  `0/5`.
- The conditional topology-law panel turns the vector into a directed
  root/internal/pass-through/leaf diagnostic. The focused truth row remains
  rank `1`, with conditional log-odds margin `4.694486`, but the support
  status is `support_insufficient_fail_closed` because the internal incidence
  stratum has only one truth-recovery row.
- The benchmark-facing `kl_conditional_topology_diagnostic` method id runs the
  17-case regression gate with `17/17` ok rows, mean ARI `0.460293`, median
  ARI `0.480000`, and exact-K count `4/17`. This confirms registry/dispatch
  integration but does not promote the law to production.
- Interpretation: the topology vector is supported as a diagnostic
  conditioning object, especially outgoing balance and outgoing edge-norm
  balance conditioned by the incoming/family event. The refined traversal
  profile is not supported as a broad production clustering default by this
  benchmark.

## Evidence

- `tests/pipeline/51_test_dispatch_contract.py` verifies that the benchmark
  dispatcher forwards `sibling_gate_profile`, root-stability settings,
  selected-root/selected-family guard settings, and `passthrough` to the KL
  runner.
- The benchmark artifacts are stored under
  `raw/assets/benchmark-results/topology_vector_benchmark_20260615/`.

## Links

- [[overlap-context-negative-bayesian-topology-law-20260615]]
- [[overlap-context-negative-bayesian-topology-sensitivity-20260615]]
- [[overlap-conditional-topology-law-panel-20260615]]
- [[overlap-context-negative-topology-transfer-20260615]]
- [[fixed-sibling-gate-profile-validation-20260613]]
- [[open-mathematical-questions]]
