---
title: Feature Split Selection Audit 2026-06-01
type: source
status: reviewed
updated: 2026-06-01
sources:
  - benchmarks/diagnostics/calibration/sample_split_selection_audit.py
  - raw/assets/benchmark-results/sample_split_selection_audit_20260601/manifest.json
  - raw/assets/benchmark-results/sample_split_selection_audit_20260601/sample_split_selection_audit_summary.csv
tags:
  - source
  - calibration
  - selection
  - crossfit
---

# Feature Split Selection Audit 2026-06-01

## Summary

This audit tests whether cross-fitting hierarchy selection and gate testing
restores edge calibration and sibling empirical-null support. Because Tree-Break Selection
uses a sample-leaf hierarchy, literal sample splitting is not yet a valid
diagnostic without an explicit assignment model for held-out samples. The
implemented cross-fit regime therefore splits feature blocks: one feature
block builds the tree, and the held-out feature block is used for node
distributions and gate tests on the same sample leaves.

## Key Points

- `gauss_null_large` fails in the in-sample full-feature regime: every tested
  edge is significant, no supported sibling calibration records exist, and the
  strict sibling calibration contract skips the run.
- The same null case under feature-split cross-fit tests only two root edges,
  rejects none, has 199 supported calibration records, and returns one cluster.
- The binary, categorical, and Gaussian signal examples retain perfect or
  high ARI under feature-split cross-fit while gaining many supported
  calibration records.
- Fixed-tree feature permutation closes the tested edges and collapses to one
  cluster in the signal examples, confirming that the held-out feature block
  carries the signal in the cross-fit runs.
- The result supports the diagnosis that the missing support problem is a
  selected-hierarchy inference problem, not a projected-Wald kernel failure.

## Evidence

- `benchmarks/diagnostics/calibration/sample_split_selection_audit.py`
  implements the diagnostic and explicitly rejects literal sample splitting
  until a held-out-sample assignment model is defined.
- `raw/assets/benchmark-results/sample_split_selection_audit_20260601/sample_split_selection_audit_summary.csv`
  records edge rejection rates, sibling support counts, ARI, status, and
  runtime for each case/regime.
- `raw/assets/benchmark-results/sample_split_selection_audit_20260601/manifest.json`
  records the seed, case names, split axis, selection fraction, and alphas.

## Links

- [[edge-selection-null-audit-20260601]]
- [[oracle-gate-path-diagnostic]]
- [[open-mathematical-questions]]
