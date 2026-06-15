---
title: Overlap Residual Recovery Eligibility 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - benchmarks/diagnostics/calibration/overlap_residual_recovery_eligibility.py
  - tests/validation/115_test_overlap_residual_recovery_eligibility.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/residual_recovery_eligibility/overlap_residual_recovery_eligibility_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/residual_recovery_eligibility/overlap_residual_recovery_eligibility_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/residual_recovery_eligibility/overlap_residual_recovery_thresholds.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/residual_recovery_eligibility/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - selected-family
---

# Overlap Residual Recovery Eligibility 2026-06-14

## Summary

`overlap_residual_recovery_eligibility.py` converts residual selected-family
metrics into explicit diagnostic gates. It separates selected-family null
evidence from structural recovery evidence, then reports which residual
families pass null-only, non-recovery-controlled structural, or strict
all-negative structural thresholds. It is diagnostic-only and does not promote
production behavior.

## Key Points

- The runner writes `overlap_residual_recovery_eligibility_rows.csv`,
  `overlap_residual_recovery_eligibility_summary.csv`,
  `overlap_residual_recovery_thresholds.csv`, and `manifest.json`.
- The null-evidence gate uses
  `residual_neg_log10_min_sibling_p_value` above the largest selected-null
  value. The displayed threshold is `8.454637`, with a strict nextafter value
  in the CSV. It retains `5/5` truth-recovery families and selects `0/8`
  selected-null families.
- The non-recovery structural gate uses
  `residual_min_fragment_risk_proxy_score` above the largest non-recovery
  signal value. The displayed threshold is `0.747520`, with a strict nextafter
  value in the CSV. It retains `3/5` truth-recovery families and selects `0/3`
  non-recovery families.
- The strict all-negative structural gate uses the same structural metric
  against selected-null plus non-recovery families, with threshold `1.076065`.
  It retains only `2/5` truth-recovery families and selects `0/11` negatives.
- Eligibility statuses in the focused run are:
  `10` `residual_no_selected_family_null_evidence`, `3`
  `residual_null_evidence_only_unresolved`, `1`
  `residual_structural_recovery_candidate_nonrecovery_controlled`, and `2`
  `residual_strict_structural_recovery_candidate`.
- One non-recovery signal family has selected-family null evidence but fails
  structural recovery. This is the concrete failure of p-value-only promotion.
- Method implication: a future selected-family recovery law needs two layers:
  a selected-family null-evidence law and a structural recovery condition. The
  structural condition cannot be replaced by p-value extremeness.

## Evidence

- `tests/validation/115_test_overlap_residual_recovery_eligibility.py` verifies
  gate-status assignment, threshold retention/leakage summaries, and output
  writing.
- Verification passed:
  `pytest tests/validation/115_test_overlap_residual_recovery_eligibility.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_residual_recovery_eligibility.py tests/validation/115_test_overlap_residual_recovery_eligibility.py`.

## Links

- [[overlap-residual-family-recovery-20260614]]
- [[overlap-diagnostic-traversal-policy-20260614]]
- [[overlap-fragment-risk-guard-20260614]]
- [[open-mathematical-questions]]
