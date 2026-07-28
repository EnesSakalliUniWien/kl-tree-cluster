---
title: Overlap Internal-Node Transfer Gap Audit 2026-06-15
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_internal_node_transfer_gap_audit.py
  - tests/validation/calibration/overlap/125_test_overlap_internal_node_transfer_gap_audit.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_transfer_gap_audit/overlap_internal_node_transfer_gap_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_transfer_gap_audit/overlap_internal_node_transfer_gap_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_transfer_gap_audit/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - bayesian
---

# Overlap Internal-Node Transfer Gap Audit 2026-06-15

## Summary

`overlap_internal_node_transfer_gap_audit.py` separates the remaining
row-level overlap recovery miss from threshold tuning. It classifies
truth-recovery rows by Bayesian evidence, local context margin, soft subspace
support, size balance, edge-norm balance, and fragment-risk components, then
uses transfer rows to check whether relaxing context would leak.

## Key Points

- The runner writes `overlap_internal_node_transfer_gap_rows.csv`,
  `overlap_internal_node_transfer_gap_summary.csv`, and `manifest.json`.
- In the focused overlap panel, the default internal-node likelihood recovers
  `4/5` truth-recovery rows and selects `0/28` negative rows.
- The single missed truth-recovery row is
  `overlap_unbal_4c_small`, replicate `1`, node `N797`.
- That missed row is not blocked by subspace, balance, edge-norm, fragment
  risk, or Bayes-factor evidence. Its only blocking component is
  `local_context_margin`, with context margin `-0.005752`.
- Transfer evidence rules out simply lowering the local context gate:
  relaxed-context rules have `24` leakage evaluations, while nonnegative
  context rules have `0` leakage evaluations.
- The resulting diagnostic status is
  `context_exception_requires_higher_order_conditional_law`.

## Method Implication

The remaining miss is a conditional/Bayesian-law problem, not an alpha or
permutation-threshold problem. The local rule should keep nonnegative context
margin as a hard selected-neighborhood gate. To recover the context-negative
but structurally coherent truth row, the next method needs a higher-order law
that conditions on ancestor, selected-family, or neighborhood evidence rather
than relaxing the local context margin globally.

## Evidence

- `tests/validation/calibration/overlap/125_test_overlap_internal_node_transfer_gap_audit.py`
  verifies context-negative soft-supported truth classification, leakage-aware
  summary status, and output writing.
- Verification passed:
  `pytest tests/validation/calibration/overlap/125_test_overlap_internal_node_transfer_gap_audit.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_internal_node_transfer_gap_audit.py tests/validation/calibration/overlap/125_test_overlap_internal_node_transfer_gap_audit.py`.

## Links

- [[overlap-internal-node-likelihood-transfer-20260615]]
- [[overlap-internal-node-bayesian-likelihood-probe-20260615]]
- [[open-mathematical-questions]]
