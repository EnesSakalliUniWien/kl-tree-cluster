---
title: Alpha Grid Full Benchmark 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - benchmarks/validation/alpha_grid_search.py
  - benchmarks/cloud/aws_alpha_grid_search.py
  - raw/assets/benchmark-results/alpha_grid_full_20260604/aws_alpha_grid_manifest.json
  - raw/assets/benchmark-results/alpha_grid_full_20260604/alpha_grid_summary.csv
  - raw/assets/benchmark-results/alpha_grid_full_20260604/alpha_grid_results.csv
tags:
  - source
  - aws
  - benchmark
  - validation
---

# Alpha Grid Full Benchmark 2026-06-04

## Summary

This AWS Batch diagnostic swept `25` edge/sibling alpha pairs over the full
`110`-case KL benchmark suite. The run used five edge alpha values
`0.0001`, `0.0003`, `0.001`, `0.003`, and `0.01`, crossed with five sibling
alpha values `0.001`, `0.003`, `0.01`, `0.03`, and `0.1`. The merged output
contains `2,750` result rows: `2,055` completed rows and `695` strict skip
rows.

The current default pair, edge alpha `0.001` and sibling alpha `0.01`, had the
best mean ARI in this grid. It is benchmark evidence for the current defaults,
not a proof of selected-tree Type-I error control.

## Key Points

- Best mean ARI: edge alpha `0.001`, sibling alpha `0.01`, with mean ARI
  `0.893811`, median ARI `1.0`, `82` completed rows, `28` skips, `62`
  exact-cluster-count rows, `8` under-splits, and `12` over-splits.
- Best exact-cluster-count hit rate among completed rows: edge alpha `0.0001`,
  sibling alpha `0.01`, with `68` exact rows, mean ARI `0.875341`, `87`
  completed rows, `23` skips, `12` under-splits, and `7` over-splits.
- The closest lower-edge-alpha alternative, edge alpha `0.0003` and sibling
  alpha `0.01`, had mean ARI `0.889121`, `66` exact rows, `85` completed
  rows, `25` skips, `9` under-splits, and `10` over-splits.
- Across this grid, sibling alpha `0.01` is the strongest compromise. Lower
  sibling alpha values increase under-splitting, while sibling alpha `0.03`
  and `0.1` increase over-splitting and cluster-count error.
- Increasing edge alpha reduces completed rows because strict calibration
  support failures become more common. This is a production contract outcome,
  not a hidden neutral fallback.
- Skip reasons are dominated by strict sibling-inflation support failures:
  no strict-null or stopped-edge positive-weight empirical-null records, with
  selected non-null records rejected as invalid calibration support. One
  continuous high-dimensional case hits the dense empirical-Gaussian covariance
  limit at raw dimension `20000`.
- AWS execution used `25` Fargate shards and one merge job. The observed
  compute time was about `1.85` Fargate task-hours, with estimated Fargate
  compute cost about `$0.43` before small S3, ECR, and CloudWatch storage
  charges.

## Evidence

- `raw/assets/benchmark-results/alpha_grid_full_20260604/aws_alpha_grid_manifest.json`
  records the run as `benchmarks.cloud.aws_alpha_grid_search`, with `25`
  alpha pairs, `25` shards, suite `full`, build commit
  `bb37e8e12d413a9b466f0be1890a5ed16fd562b5`, branch `dev`, and dirty
  worktree state `true`.
- `raw/assets/benchmark-results/alpha_grid_full_20260604/alpha_grid_summary.csv`
  records one summary row per alpha pair.
- `raw/assets/benchmark-results/alpha_grid_full_20260604/alpha_grid_results.csv`
  records one result row per alpha pair and benchmark case.
- `benchmarks/validation/alpha_grid_search.py` defines the local alpha-grid
  runner and records that the output is diagnostic benchmark evidence, not a
  selected-tree Type-I error proof.
- `benchmarks/cloud/aws_alpha_grid_search.py` defines the AWS shard and merge
  wrapper used for this run.

## Links

- [[open-mathematical-questions]]
- [[benchmark-pipeline-contract]]
- [[selected-hierarchy-null-support-contract]]
