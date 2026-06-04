---
title: Sibling Null Prior Interpolation Audit 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - benchmarks/diagnostics/calibration/sibling_null_prior_interpolation_audit.py
  - tests/validation/60_test_sibling_null_prior_interpolation_audit.py
  - raw/assets/benchmark-results/sibling_null_prior_interpolation_audit_20260604/manifest.json
  - raw/assets/benchmark-results/sibling_null_prior_interpolation_audit_20260604/case_summary.csv
  - raw/assets/benchmark-results/sibling_null_prior_interpolation_audit_20260604/record_interpolation_audit.csv
tags:
  - source
  - calibration
  - diagnostics
  - selection
---

# Sibling Null Prior Interpolation Audit 2026-06-04

## Summary

This diagnostic reconstructs the old tree-neighborhood sibling-null prior
score from current explicit child-parent edge columns. It compares that score
against the active strict internal calibration support contract. The diagnostic
does not install interpolated priors as production calibration and records its
role as `diagnostic_interpolated_sibling_null_prior_not_calibration`.

The four-case run covers `binary_perfect_4c`, `cat_highcard_20cat_4c`,
`overlap_heavy_4c_small_feat`, and `phylo_large_32taxa`.

## Key Points

- All four cases had `no_strict_internal_support`: every current positive
  calibration-weight record was selected non-null under edge evidence, not
  strict null-like or stopped-edge supported.
- The diagnostic interpolation still assigns positive weights to selected
  non-null records: `3/6` in `binary_perfect_4c`, `196/199` in
  `cat_highcard_20cat_4c`, `499/499` in `overlap_heavy_4c_small_feat`, and
  `288/319` in `phylo_large_32taxa`.
- Candidate interpolated scalar scales are descriptive only:
  `19.26` for `binary_perfect_4c`, `1.00` for `cat_highcard_20cat_4c`,
  `18.68` for `overlap_heavy_4c_small_feat`, and `37.14` for
  `phylo_large_32taxa`.
- The old score therefore diagnoses why the previous method could avoid a hard
  support failure, but also why it was mathematically inadmissible as
  empirical-null support: it borrows selected non-null records.
- The diagnostic uses direct tested-edge BH p-values for tested child edges and
  requires an explicit stopped ancestor for untested or ancestor-blocked child
  edges. Missing stopped-ancestor support stays unsupported; no neutral prior
  is inserted.
- Dimensionless stable edges are counted separately instead of receiving an
  invented structural dimension. For `binary_perfect_4c`, eight stable
  dimensionless edges were present.

## Evidence

- `benchmarks/diagnostics/calibration/sibling_null_prior_interpolation_audit.py`
  implements the explicit edge-column diagnostic, writes `case_summary.csv`,
  `record_interpolation_audit.csv`, and a manifest, and marks production method
  changes as false.
- `tests/validation/60_test_sibling_null_prior_interpolation_audit.py` verifies
  direct tested-edge priors, stopped-ancestor interpolation, unsupported
  missing-ancestor states, and output writing.
- `raw/assets/benchmark-results/sibling_null_prior_interpolation_audit_20260604/case_summary.csv`
  records the four representative case summaries.
- `raw/assets/benchmark-results/sibling_null_prior_interpolation_audit_20260604/record_interpolation_audit.csv`
  records the per-sibling-pair priors, strict support flags, and selected
  non-null flags.

## Links

- [[open-mathematical-questions]]
- [[selected-hierarchy-null-support-contract]]
- [[internal-vs-selected-hierarchy-inflation-20260603]]
