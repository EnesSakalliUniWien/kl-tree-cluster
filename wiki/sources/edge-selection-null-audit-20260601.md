---
title: Edge Selection Null Audit 2026-06-01
type: source
status: reviewed
updated: 2026-06-01
sources:
  - raw/assets/benchmark-results/edge_selection_null_audit_20260601/manifest.json
  - raw/assets/benchmark-results/edge_selection_null_audit_20260601/edge_selection_null_summary.csv
  - raw/assets/benchmark-results/edge_selection_null_audit_20260601/edge_selection_null_replicate_summary.csv
  - raw/assets/benchmark-results/edge_selection_null_audit_20260601/edge_selection_null_edge_level.csv
tags:
  - source
  - calibration
  - edge-gate
  - selection
---

# Edge Selection Null Audit 2026-06-01

## Summary

This audit compares child-parent edge-gate behavior under pure Bernoulli null
data when the hierarchy is selected from the same data versus when the
hierarchy is fixed and feature columns are permuted. It isolates the upstream
selection effect that removes internal empirical-null support for sibling
inflation.

## Key Points

- The audit uses `EDGE_ALPHA = 0.001`, Hamming tree distance, and average
  linkage.
- In the in-sample mode, the hierarchy and edge tests use the same null data.
  Across `null64x32`, `null128x64`, and `null200x80`, the mean child-parent
  edge rejection rate is about `0.99`.
- In the fixed-tree permutation mode, the topology is held fixed while feature
  columns are permuted. Median rejection rate is `0.0` across all three
  scenarios.
- The result shows that the edge gate is calibrated as a fixed-tree test much
  more than as a post-tree-selection test. Data-selected hierarchy construction
  creates child-parent contrasts even under a global null.
- This explains why sibling calibration can have positive-weight records but no
  admissible strict-null or stopped-edge empirical-null records.

## Evidence

- `raw/assets/benchmark-results/edge_selection_null_audit_20260601/edge_selection_null_summary.csv`
  records the aggregate rejection rates.
- `raw/assets/benchmark-results/edge_selection_null_audit_20260601/edge_selection_null_replicate_summary.csv`
  records replicate-level rates.
- `raw/assets/benchmark-results/edge_selection_null_audit_20260601/edge_selection_null_edge_level.csv`
  records edge-level p-values and rejection flags.
- `raw/assets/benchmark-results/edge_selection_null_audit_20260601/manifest.json`
  records the seed, scenarios, tree metric, linkage method, and purpose.

## Links

- [[oracle-gate-path-diagnostic]]
- [[open-mathematical-questions]]
