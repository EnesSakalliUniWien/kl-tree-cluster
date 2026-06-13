---
title: Selected Tail Parent Size Balance Stability 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - benchmarks/diagnostics/calibration/selected_tail_parent_size_balance_stability.py
  - tests/validation/73_test_selected_tail_parent_size_balance_stability.py
  - raw/assets/benchmark-results/selected_tail_parent_size_balance_stability_20260604/parent_size_balance_contexts.csv
  - raw/assets/benchmark-results/selected_tail_parent_size_balance_stability_20260604/parent_size_balance_parent_folds.csv
  - raw/assets/benchmark-results/selected_tail_parent_size_balance_stability_20260604/parent_size_balance_summary.csv
  - raw/assets/benchmark-results/selected_tail_parent_size_balance_stability_20260604/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_topology_refinement_input_20260603_300/selected_geometry_records.csv
tags:
  - source
  - calibration
  - selected-tail
  - barycentric
---

# Selected Tail Parent Size Balance Stability 2026-06-04

## Summary

This source records the balance-aware parent-size stability diagnostic for
selected-tail laws. The diagnostic groups row-level selected-geometry records
by source family, feature family, sibling projection dimension, edge-action
bin, and predeclared barycentric-balance bin. It then tests leave-one-parent-
size-bin tail transfer and reports simulation-level \( \hat c \) precision.

The run uses the 300-replicate row-level selected-geometry table because the
1000-replicate selected-tail output currently stores aggregate context rows,
not the row-level simulation records needed for \( \hat c \) precision and
parent-size holdout.

## Key Points

- The diagnostic evaluated `56` balance-scoped selected-tail contexts.
- `3` contexts are `parent_size_balance_external_candidate`: all are
  `gaussian_blobs`/Bernoulli, sibling projection dimension `2`,
  `edge_action_ge8`, with balance bins `balance_0.1_0.25`,
  `balance_0.25_0.4`, and `balance_0.4_0.5`.
- The three candidate contexts pass the `499` matching-simulation,
  `499` matched-record, `5%` relative \( \hat c \) simulation-SE,
  parent-size heldout absolute-error `0.005`, and heldout SE `0.002`
  gates on the available row-level table.
- `2` support-rich contexts are `undefined_parent_size_holdout` because they
  only cover one parent-size bin, so parent-size transfer cannot be tested.
- `51` contexts are `undefined_support_failure`.
- Categorical high-edge projection-2 contexts do have valid parent-size folds,
  but they fail support and/or parent-size tail stability under the current
  thresholds.

## Evidence

- `parent_size_balance_contexts.csv` is the context-level decision table.
- `parent_size_balance_parent_folds.csv` records the leave-one-parent-size-bin
  folds.
- `parent_size_balance_summary.csv` records decision counts.
- `tests/validation/73_test_selected_tail_parent_size_balance_stability.py`
  verifies predeclared balance bins, child-size validation, candidate
  promotion when all gates pass, support failure behavior, and artifact
  writing.

## Links

- [[open-mathematical-questions]]
- [[selected-tail-promotion-gate-20260604]]
- [[barycentric-method-literature-request-20260604]]
