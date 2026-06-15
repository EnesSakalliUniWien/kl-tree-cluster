---
title: Overlap Context-Negative Topology Transfer 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - benchmarks/diagnostics/calibration/overlap_context_negative_topology_transfer.py
  - tests/validation/131_test_overlap_context_negative_topology_transfer.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_topology_transfer/overlap_context_negative_topology_transfer_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_topology_transfer/overlap_context_negative_topology_transfer_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_topology_transfer/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - topology
  - transfer
---

# Overlap Context-Negative Topology Transfer 2026-06-15

## Summary

`overlap_context_negative_topology_transfer.py` tests whether the
single-positive topology separator found inside `context_negative_emergent`
rows transfers across held-out cases or held-out replicates. It learns
zero-negative thresholds on training folds and applies them to held-out folds,
so focused full-sample max-negative cutpoints are not mistaken for a validated
traversal law.

## Key Points

- The runner writes transfer rows, a transfer summary, and `manifest.json`.
- The audited metrics include `balance_product`, outgoing and size balance,
  edge-norm balance, and outgoing balance multiplied by edge or subspace
  evidence.
- In leave-one-case validation, `balance_product` has training separators in
  the three folds where the single truth row remains in training, but none of
  those folds contain held-out truth. The fold holding out the truth row has no
  training truth support, so the rule is not evaluable there.
- The leave-one-case `balance_product` status is
  `transfer_unvalidated_no_truth_holdout_support`, with zero held-out negative
  leakage among the folds that have a training rule.
- In leave-one-replicate validation, `balance_product` has the same
  unvalidated status and zero leakage. Several neighboring metrics leak one
  held-out negative row when split by replicate, including outgoing balance,
  edge-norm balance, outgoing edge-norm balance, and size balance.
- `outgoing_balance_edge_subspace_product` does not produce a training
  separator in either split family.
- The transfer audit therefore preserves the qualitative direction of the
  candidate, but rejects promotion: the focused dataset has only one positive
  context-negative emergent truth row, so no held-out positive fold can both
  have train support and validate retention.

## Evidence

- `tests/validation/131_test_overlap_context_negative_topology_transfer.py`
  verifies successful held-out validation when two truth cases exist, the
  missing-support status for a single truth case, and output writing.
- Verification passed:
  `pytest tests/validation/131_test_overlap_context_negative_topology_transfer.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_context_negative_topology_transfer.py tests/validation/131_test_overlap_context_negative_topology_transfer.py`.

## Links

- [[overlap-context-negative-topology-conditioning-20260615]]
- [[overlap-context-negative-edge-conditioning-20260615]]
- [[overlap-bayesian-incidence-mode-law-20260615]]
- [[open-mathematical-questions]]
