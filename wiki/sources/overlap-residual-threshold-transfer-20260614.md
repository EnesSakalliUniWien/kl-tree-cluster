---
title: Overlap Residual Threshold Transfer 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_residual_threshold_transfer.py
  - tests/validation/calibration/overlap/117_test_overlap_residual_threshold_transfer.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/residual_threshold_transfer/overlap_residual_threshold_transfer_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/residual_threshold_transfer/overlap_residual_threshold_transfer_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/residual_threshold_transfer/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - thresholds
---

# Overlap Residual Threshold Transfer 2026-06-14

## Summary

`overlap_residual_threshold_transfer.py` tests whether the residual
selected-family thresholds transfer under leave-one-case and leave-one-replicate
splits. It recomputes max-negative thresholds on the training families and
evaluates retention and leakage on held-out families. It is diagnostic-only.

## Key Points

- The runner writes `overlap_residual_threshold_transfer_rows.csv`,
  `overlap_residual_threshold_transfer_summary.csv`, and `manifest.json`.
- All three residual threshold families show transfer leakage in both
  leave-one-case and leave-one-replicate summaries.
- The selected-family null-evidence threshold retains held-out recovery
  families (`5/5`) but leaks held-out negatives: in both split kinds it selects
  `1/8` selected-null families and `1/3` non-recovery families.
- The non-recovery structural threshold retains `3/5` held-out recovery
  families but leaks heavily into selected-null families: `7/8` selected-null
  under leave-one-case and `5/8` under leave-one-replicate. It also selects
  `1/3` held-out non-recovery families in both split kinds.
- The strict all-negative structural threshold retains only `2/5` held-out
  recovery families. It still leaks selected-null families under transfer:
  `2/8` under leave-one-case and `1/8` under leave-one-replicate.
- Method implication: the focused max-negative thresholds are explanatory
  cutpoints, not transferable calibration constants. They identify which
  threshold classes matter, but production needs a selected-family law or a
  predeclared conservative rule validated outside the same focused panel.

## Evidence

- `tests/validation/calibration/overlap/117_test_overlap_residual_threshold_transfer.py` verifies
  leave-one split construction, held-out retention/leakage summaries, and
  output writing.
- Verification passed:
  `pytest tests/validation/calibration/overlap/117_test_overlap_residual_threshold_transfer.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_residual_threshold_transfer.py tests/validation/calibration/overlap/117_test_overlap_residual_threshold_transfer.py`.

## Links

- [[overlap-residual-recovery-eligibility-20260614]]
- [[overlap-threshold-hierarchy-20260614]]
- [[open-mathematical-questions]]
