---
title: Overlap Threshold Hierarchy 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_threshold_hierarchy.py
  - tests/validation/calibration/overlap/116_test_overlap_threshold_hierarchy.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/threshold_hierarchy/overlap_threshold_hierarchy.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/threshold_hierarchy/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - thresholds
---

# Overlap Threshold Hierarchy 2026-06-14

## Summary

`overlap_threshold_hierarchy.py` composes the overlap traversal diagnostics
into one ordered threshold table. It is diagnostic-only: the table explains
which thresholds act at which stage and what each threshold retains or blocks,
but it does not define a production calibration rule.

## Key Points

- The runner writes `overlap_threshold_hierarchy.csv` and `manifest.json`.
- Stage 1, continuous structural stable accept, uses nonnegative continuous
  context margin plus same-subspace support. It retains `12/17`
  truth-aligned accepted rows and selects `0/43` selected-null or
  truth-misaligned accepted rows.
- Stage 2, fragment-risk blocking, uses
  `fragment_risk_proxy_score >= 1.252729`. It blocks `8/9` fragment-like weak
  rows and `0/5` truth-recovery rows.
- Stage 3, residual selected-family null evidence, uses
  `residual_neg_log10_min_sibling_p_value >= 8.454637`. It retains `5/5`
  truth-recovery families and selects `0/8` selected-null families.
- Stage 4, residual non-recovery structural evidence, uses
  `residual_min_fragment_risk_proxy_score >= 0.747520`. It retains `3/5`
  truth-recovery families and selects `0/3` non-recovery families.
- Stage 5, strict residual all-negative structural evidence, uses
  `residual_min_fragment_risk_proxy_score >= 1.076065`. It retains only `2/5`
  truth-recovery families and selects `0/11` selected-null or non-recovery
  families.
- Stage 6 records the unresolved selected-family mixture: `2/5` truth-recovery
  families and `1/3` non-recovery families have selected-family null evidence
  without enough structural recovery evidence.
- Method implication: the threshold hierarchy is now segmented. Structural
  margins define stable regions, fragment-risk thresholds block one-sided
  fragments, p-value thresholds provide null evidence, and residual structural
  thresholds are the remaining power/control tradeoff.

## Evidence

- `tests/validation/calibration/overlap/116_test_overlap_threshold_hierarchy.py` verifies ordered
  stage synthesis, fragment-threshold sourcing from the guard scan, and output
  writing.
- Verification passed:
  `pytest tests/validation/calibration/overlap/116_test_overlap_threshold_hierarchy.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_threshold_hierarchy.py tests/validation/calibration/overlap/116_test_overlap_threshold_hierarchy.py`.

## Links

- [[overlap-structural-decision-zones-20260614]]
- [[overlap-diagnostic-traversal-policy-20260614]]
- [[overlap-residual-recovery-eligibility-20260614]]
- [[open-mathematical-questions]]
