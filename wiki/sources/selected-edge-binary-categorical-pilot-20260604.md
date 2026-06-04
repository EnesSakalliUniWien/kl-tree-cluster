---
title: Selected Edge Binary Categorical Pilot 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - benchmarks/validation/selected_edge_type1_geometry.py
  - benchmarks/cloud/aws_selected_edge_type1_geometry.py
  - benchmarks/diagnostics/analysis/selected_edge_geometry_analysis.py
  - raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/aws_selected_edge_geometry_manifest.json
  - raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/selected_edge_geometry_edges.csv
  - raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/selected_edge_geometry_siblings.csv
  - raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/selected_edge_geometry_final.csv
  - raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/selected_edge_geometry_models.csv
tags:
  - source
  - aws
  - validation
  - selected-inference
  - categorical
---

# Selected Edge Binary Categorical Pilot 2026-06-04

## Summary

This AWS Batch pilot extends the selected-edge Type-I geometry diagnostic from
binary global-null matrices to direct categorical multinomial global-null
matrices. It keeps continuous and precomputed-distance contexts unsupported
because their selected-edge null generators and covariance contracts have not
yet been validated.

The run used `40` replicates, four benchmark case shapes, two edge alphas
(`0.0001` and `0.001`), sibling alpha `0.01`, fixed-tree and same-data
selected-tree modes, and four AWS shards. The merged output contains
`136,320` edge rows, `41,435` sibling rows, and `640` final rows.

The AWS array job was `624b47a0-79d7-4be1-abd4-fac0222646e3`; the merge job
was `69e3a6d4-f9a1-495a-a717-ec95afb1cc18`. The observed task runtime was
about `790` task-seconds at `4` vCPU and `16` GB memory, or roughly `$0.05`
of Fargate compute before small storage and logging charges.

## Key Points

- Same-data selected-tree mode again rejects almost every Tree-BH-tested
  frontier edge. Tested-frontier rejection rates are about `0.996` to `0.999`
  across the two binary and two categorical null cases.
- Across all edge rows, selected-tree rejection rates are about `0.803` to
  `0.907` for binary cases and about `0.773` to `0.902` for categorical cases.
- Fixed-tree final decompositions return one cluster for both binary cases and
  for `cat_clear_3cat_4c`. In `cat_highcard_20cat_4c`, fixed-tree final rows
  have a `0.025` false-split rate at both tested edge alphas.
- High-cardinality categorical fixed-tree frontier behavior is already
  inflated: tested-frontier rejection rates are about `0.778` at edge alpha
  `0.0001` and `0.923` at edge alpha `0.001`. This means high-cardinality
  categorical calibration is not explained only by same-data tree selection.
- Selected-tree rows mostly fail explicitly at strict sibling-inflation support
  when only selected non-null calibration records are available. The failure is
  not converted into a neutral p-value or fallback calibration.
- The descriptive analysis again finds that edge action and edge statistic
  margin almost perfectly separate selected-tree openings. This supports
  selected-region geometry as a diagnostic direction, not a production law.

## Evidence

- `raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/aws_selected_edge_geometry_manifest.json`
  records suite `full`, four case names, `40` replicates, four shards, two
  edge alpha values, and runner
  `benchmarks.cloud.aws_selected_edge_type1_geometry`.
- `raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/selected_edge_geometry_edges.csv`
  records edge p-values, Tree-BH status, and selected-region geometry
  covariates.
- `raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/selected_edge_geometry_final.csv`
  records final decomposition status and strict support failures.
- `raw/assets/benchmark-results/selected-edge-type1-binary-categorical-pilot-20260604/merged/selected_edge_geometry_models.csv`
  records descriptive candidate-law outputs.

## Links

- [[selected-edge-type1-geometry-pilot-20260604]]
- [[edge-selection-null-audit-20260601]]
- [[selected-hierarchy-selection-geometry]]
- [[open-mathematical-questions]]
