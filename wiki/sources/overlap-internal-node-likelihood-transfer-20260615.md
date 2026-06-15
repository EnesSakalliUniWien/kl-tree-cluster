---
title: Overlap Internal-Node Likelihood Transfer 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - benchmarks/diagnostics/calibration/overlap_internal_node_likelihood_transfer.py
  - tests/validation/124_test_overlap_internal_node_likelihood_transfer.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_likelihood_transfer/overlap_internal_node_likelihood_transfer_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_likelihood_transfer/overlap_internal_node_likelihood_transfer_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_likelihood_transfer/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - bayesian
---

# Overlap Internal-Node Likelihood Transfer 2026-06-15

## Summary

`overlap_internal_node_likelihood_transfer.py` tests whether row-level
internal-node likelihood rules selected on training splits transfer to held-out
cases or replicates. It is diagnostic-only and does not promote a production
threshold.

## Key Points

- The runner writes `overlap_internal_node_likelihood_transfer_rows.csv`,
  `overlap_internal_node_likelihood_transfer_summary.csv`, and `manifest.json`.
- On each leave-one split, it selects rules from the predeclared sensitivity
  grid that are zero-negative and recovery-retaining on the training rows, then
  evaluates those selected rules on the held-out rows.
- Leave-one-case transfer is clean in the focused panel: `72` selected rule
  evaluations, `0` leakage evaluations, maximum held-out recovery retention
  `1.0`, and median held-out retention `0.75`.
- Leave-one-replicate transfer is mixed: `96` selected rule evaluations,
  `24` leakage evaluations, and maximum held-out recovery retention
  `0.666667`.
- All `24` replicate-level leakage evaluations come from rules with
  `context_margin_floor = -0.002`.
- Restricting to nonnegative context-margin rules gives zero leakage in both
  split kinds: `72/72` zero-leakage leave-one-case evaluations and `72/72`
  zero-leakage leave-one-replicate evaluations. Recovery retention is still
  limited at the replicate level, with maximum `0.666667` and median `0.5`.
- Method implication: nonnegative context margin transfers as a hard
  conditional gate in this focused panel, but the overlap internal-node
  likelihood is not production-ready because replicate-level recovery retention
  remains incomplete.

## Evidence

- `tests/validation/124_test_overlap_internal_node_likelihood_transfer.py`
  verifies that relaxed context-margin rules can leak on held-out rows, that
  transfer summaries report leakage, and that outputs are written.
- Verification passed:
  `pytest tests/validation/124_test_overlap_internal_node_likelihood_transfer.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_internal_node_likelihood_transfer.py tests/validation/124_test_overlap_internal_node_likelihood_transfer.py`.

## Links

- [[overlap-internal-node-likelihood-sensitivity-20260615]]
- [[overlap-internal-node-bayesian-likelihood-probe-20260615]]
- [[open-mathematical-questions]]
