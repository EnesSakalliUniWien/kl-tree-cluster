---
title: Selected Hierarchy Null Audit 2026-06-01
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/selected/hierarchy/selected_hierarchy_null_audit.py
  - raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601/selected_hierarchy_null_audit_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601_richer_root/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601_richer_root/selected_hierarchy_null_audit_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601_richer_nonroot/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601_richer_nonroot/selected_hierarchy_null_audit_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_precision_20260601_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_precision_20260601_root_strict_500/selected_hierarchy_null_audit_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_precision_20260601_nonroot_strict_500/selected_hierarchy_null_audit_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_precision_20260601_nonroot_relaxed_parent_500/selected_hierarchy_null_audit_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_precision_20260601_nonroot_relaxed_projection_500/selected_hierarchy_null_audit_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_precision_20260601_nonroot_relaxed_any_500/selected_hierarchy_null_audit_summary.csv
tags:
  - source
  - calibration
  - selection
  - null
---

# Selected Hierarchy Null Audit 2026-06-01

## Summary

This audit estimates the same-data selected-hierarchy sibling null by
regenerating null feature matrices, rebuilding the hierarchy inside every
replicate, rerunning the edge gate, and collecting selected focal sibling
statistics. It keeps cross-fitting out of the candidate method and directly
targets the selected-hierarchy inference problem.

## Key Points

- The audit is diagnostic-only and does not install a production calibration
  fallback.
- Missing matched selected-hierarchy records are an explicit unsupported
  descriptive status. They are not converted into \(c\), a p-value, or a
  blocking decision.
- The implemented null generator supports Bernoulli and categorical feature
  contracts. Continuous selected-hierarchy null simulation remains unsupported
  until a validated covariance generator and audit-owned continuous
  tree-distance contract exist.
- In the four representative cases, selected-hierarchy inflation estimates are
  much larger than fixed-subspace or local-edge-selection diagnostics:
  approximately `58.5` for `gauss_null_large`, `28.8` for
  `binary_low_noise_4c`, `50.5` for `cat_clear_3cat_4c`, and `30.6` for
  `gauss_clear_medium`.
- The selected-hierarchy mean-scaled chi-square p-value blocks the null case
  (`gauss_null_large`) and still rejects the three signal examples.
- The richer 100-replicate root rerun uses edge-path-open selected records and
  `projection_parent_size_depth` matching. It again blocks the null case and
  leaves all three signal examples significant, with \(c\) estimates
  approximately `51.0`, `37.2`, `18.0`, and `24.9`.
- The richer rerun now records Monte Carlo precision diagnostics. For matched
  rows, simulation-level relative standard errors for \(c\) are about `3%` to
  `10%`, so the broad scale of \(c\) in the tens is stable enough as a
  descriptive phenomenon. Empirical-tail resolution is much coarser:
  approximately `0.011` to `0.023` for root rows and `0.026` for the matched
  non-root null row. That is not enough to support a production tail decision
  at `SIBLING_ALPHA = 0.01`.
- The richer 100-replicate non-root strongest rerun gives matched support for
  the Gaussian null and Gaussian signal targets, with \(c\) estimates
  approximately `71.8` and `36.7`. The binary and categorical non-root targets
  have zero matched selected-null records under projection, parent-size, and
  depth matching. They are recorded as
  `unsupported_no_matched_selected_hierarchy_records`, not as negative
  calibration decisions.
- The 500-replicate descriptive precision run used a fixed target for
  description, not production calibration: relative simulation SE for \(c\)
  near or below `5%`, and matching-simulation tail resolution near or below
  `0.01`. Root strict-context rows met the tail-resolution target and had
  \(c\) estimates in the tens. Relative \(c\) SE ranged from `1.6%` to `5.2%`;
  the binary root row was just above the descriptive target.
- In the 500-replicate non-root strict-context run, Gaussian rows had matched
  support and \(c\) estimates in the tens. The binary row still had zero
  matched simulations, and the categorical row had only five matched
  simulations. This shows that exact non-root depth matching is too sparse for
  those two examples at this replicate count.
- The non-root context-relaxation ladder showed where support is lost. Dropping
  depth while keeping projection and parent-size gave binary/categorical
  matching-simulation counts `25` and `195`. Matching only projection gave
  counts `206` and `304`. Matching only feature family gave counts `246` and
  `313`. Across relaxed contexts, \(c\) remained in the tens. These relaxed
  contexts describe the support geometry; they are not calibration fallback
  rules.
- This supports the interpretation that the large correction is caused by
  same-data hierarchy selection, not by the projected-Wald kernel alone.

## Evidence

- `benchmarks/diagnostics/calibration/selected/hierarchy/selected_hierarchy_null_audit.py`
  implements the selected same-data hierarchy simulation and context matching.
- `raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601/selected_hierarchy_null_audit_summary.csv`
  records observed target statistics, selected-hierarchy null counts,
  selected-hierarchy \(c\), mean-scaled p-values, empirical tail p-values, and
  blocking flags.
- `raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601/manifest.json`
  records the case names, seed, replicate count, target mode, and context
  matching rule.
- `raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601_richer_root/selected_hierarchy_null_audit_summary.csv`
  records the 100-replicate root rerun with edge-path-open records and
  projection, parent-size, and depth matching, including record-level and
  simulation-level precision fields.
- `raw/assets/benchmark-results/selected_hierarchy_null_audit_20260601_richer_nonroot/selected_hierarchy_null_audit_summary.csv`
  records the 100-replicate non-root strongest rerun and the zero-match
  support warnings for the binary and categorical non-root rows, including
  explicit unsupported status fields rather than fallback decisions.
- `raw/assets/benchmark-results/selected_hierarchy_precision_20260601_summary.csv`
  aggregates the 500-replicate strict-context and context-relaxation ladder
  runs.

## Links

- [[edge-selection-null-audit-20260601]]
- [[feature-split-selection-audit-20260601]]
- [[oracle-gate-path-diagnostic]]
- [[open-mathematical-questions]]
