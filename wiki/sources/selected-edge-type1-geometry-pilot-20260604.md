---
title: Selected Edge Type-I Geometry Pilot 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - benchmarks/validation/selected_edge_type1_geometry.py
  - benchmarks/cloud/aws_selected_edge_type1_geometry.py
  - benchmarks/diagnostics/analysis/selected_edge_geometry_analysis.py
  - raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/aws_selected_edge_geometry_manifest.json
  - raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/selected_edge_geometry_edges.csv
  - raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/selected_edge_geometry_siblings.csv
  - raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/selected_edge_geometry_final.csv
  - raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/selected_edge_geometry_models.csv
tags:
  - source
  - aws
  - validation
  - selected-inference
---

# Selected Edge Type-I Geometry Pilot 2026-06-04

## Summary

This AWS Batch pilot checks the same-data selected-edge hypothesis under
binary global-null data. It compares a fixed-tree control, where the hierarchy
is built from one null matrix and tested on an independent null matrix, with a
selected-tree mode, where the same null matrix builds the hierarchy and
supplies the child-parent edge tests.

The run used `40` replicates, two binary benchmark cases, two edge alphas
(`0.0001` and `0.001`), sibling alpha `0.01`, and four AWS shards. The merged
diagnostic contains `40,960` edge rows, `12,231` sibling rows, and `320` final
rows.

The AWS array job was `0050e601-3a53-4554-8d93-713b4dea3dd1`; the merge job
was `1c27b59e-7e3f-4f0c-ad20-2edd7064baca`. The observed task runtime was
about `251` task-seconds at the stack shape `4` vCPU and `16` GB memory,
corresponding to roughly `$0.02` of Fargate compute before small S3, ECR, and
CloudWatch charges.

## Key Points

- Same-data selected-tree edge testing is strongly anti-conservative in this
  pilot. Among Tree-BH-tested frontier edges, selected-tree rejection rates are
  about `0.996` to `0.998`; across all edge rows, rejection rates are about
  `0.803` to `0.907`.
- Fixed-tree testing is much less extreme but not a complete proof baseline in
  this runner. Across all fixed-tree edge rows, rejection rates are about
  `0.004` to `0.020`; among the smaller tested frontier, rejection rates range
  from about `0.148` to `0.543`.
- Fixed-tree final decompositions return one cluster in all `160` fixed-tree
  final rows. Selected-tree final rows produce no false split rows because most
  selected-tree decompositions fail explicitly at strict sibling-inflation
  support rather than silently accepting selected non-null calibration records.
- The selected-tree failures are the expected strict contract failure:
  `Cannot fit sibling inflation model: no strict-null or stopped-edge
  empirical-null calibration records with positive weight`. The runner reports
  selected non-null positive-weight records as invalid calibration support.
- Diagnostic model fits show that edge action and edge statistic margin almost
  perfectly separate selected-tree from fixed-tree edge openings in this pilot.
  This is descriptive evidence about the selected region, not a production
  calibration law.
- The runner now keeps full-space `edge_z_norm` and
  `edge_projection_energy_ratio` as `NaN`, because this diagnostic materializes
  only the projected-Wald statistic and projected norm.

## Evidence

- `raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/aws_selected_edge_geometry_manifest.json`
  records `40` replicates, four shards, suite `binary`, cases
  `binary_2clusters` and `binary_low_noise_4c`, modes `fixed_tree` and
  `selected_tree`, and runner
  `benchmarks.cloud.aws_selected_edge_type1_geometry`.
- `raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/selected_edge_geometry_edges.csv`
  records child-parent edge rows and selected-region geometry covariates.
- `raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/selected_edge_geometry_siblings.csv`
  records sibling-level rows when decomposition reaches sibling annotation.
- `raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/selected_edge_geometry_final.csv`
  records final per-replicate status and strict calibration-support failures.
- `raw/assets/benchmark-results/selected-edge-type1-pilot-20260604/merged/selected_edge_geometry_models.csv`
  records descriptive candidate-law search outputs.

## Links

- [[edge-selection-null-audit-20260601]]
- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-null-support-contract]]
- [[open-mathematical-questions]]
