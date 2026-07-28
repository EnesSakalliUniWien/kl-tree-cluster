---
title: Traversal Sibling FDR Smoke 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - benchmarks/validation/statistics/traversal_sibling_fdr_null.py
  - benchmarks/cloud/aws_traversal_sibling_fdr_null.py
  - tests/validation/statistics/67_test_traversal_sibling_fdr_null.py
  - tests/validation/68_test_aws_traversal_sibling_fdr_null.py
  - raw/assets/benchmark-results/traversal_sibling_fdr_smoke_20260604/synthetic/traversal_sibling_fdr_summary.csv
  - raw/assets/benchmark-results/traversal_sibling_fdr_smoke_20260604/binary/traversal_sibling_fdr_summary.csv
tags:
  - source
  - fdr
  - calibration
  - diagnostics
---

# Traversal Sibling FDR Smoke 2026-06-04

## Summary

This smoke diagnostic separates algorithmic traversal-aligned sibling BH from
projected-Wald calibration, selected-tree effects, and empirical-inflation
support. It is not a final calibration run and does not change production FDR,
alpha defaults, or sibling inflation behavior.

The diagnostic has four layers:

- `synthetic_valid_p`: valid uniform null sibling p-values on an always-open
  three-level traversal tree.
- `fixed_tree_wald`: raw sibling projected-Wald p-values on an independent
  fixed null tree.
- `selected_tree_wald`: raw sibling projected-Wald p-values after rebuilding
  the null tree from the same data.
- `selected_tree_inflated`: active empirical-null inflation after rebuilding
  the null tree from the same data.

## Key Points

- In the synthetic valid-p smoke with `200` replicates and sibling alpha
  `0.01`, mean FDP is `0.03`, giving outcome `algorithmic_fdr_failure`. This
  does not prove the production method fails, but it shows that repeatedly
  applying BH across traversal depths is not automatically a global sibling
  FDR guarantee even when p-values are valid.
- In the binary `binary_2clusters` smoke with `20` replicates, fixed-tree raw
  projected-Wald has mean FDP `0.25`, classified as
  `fixed_tree_calibration_failure`.
- The same binary case under selected-tree raw projected-Wald has mean FDP
  `0.95`, classified as `edge_selected_family_failure`.
- The same binary case with active empirical-null inflation has `18` support
  failures and only `2` valid decision rows, classified as
  `inflation_support_failure`. The diagnostic explicitly avoids treating the
  two valid rows as proof of control.

## Evidence

- `benchmarks/validation/statistics/traversal_sibling_fdr_null.py` defines the layered
  diagnostic runner and writes simulation, summary, and manifest files.
- `benchmarks/cloud/aws_traversal_sibling_fdr_null.py` defines the AWS
  shard/merge wrapper for larger runs.
- `tests/validation/statistics/67_test_traversal_sibling_fdr_null.py` verifies the layer
  contract, FDR summary logic, strict support-failure classification, and
  local output writing.
- `tests/validation/68_test_aws_traversal_sibling_fdr_null.py` verifies
  replicate sharding and merge validation.
- `raw/assets/benchmark-results/traversal_sibling_fdr_smoke_20260604/`
  contains the copied smoke summaries and simulation rows.

## Links

- [[open-mathematical-questions]]
- [[selected-hierarchy-null-support-contract]]
- [[benchmark-pipeline-contract]]
