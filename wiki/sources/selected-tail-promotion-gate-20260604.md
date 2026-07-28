---
title: Selected Tail Promotion Gate 2026-06-04
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/selected/tail/selected_tail_promotion_gate.py
  - tests/validation/calibration/selected/tail/72_test_selected_tail_promotion_gate.py
  - raw/assets/benchmark-results/selected_tail_promotion_gate_20260604/selected_tail_promotion_contexts.csv
  - raw/assets/benchmark-results/selected_tail_promotion_gate_20260604/selected_tail_q5_promotion_gate.csv
  - raw/assets/benchmark-results/selected_tail_promotion_gate_20260604/selected_tail_promotion_summary.csv
  - raw/assets/benchmark-results/selected_tail_promotion_gate_20260604/manifest.json
  - raw/assets/benchmark-results/selected_tail_equation_cloud_run_20260604_1000/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_tail_law_q5_validation_20260604/q5_selected_tail_law_validation.csv
tags:
  - source
  - calibration
  - diagnostics
  - selected-tail
---

# Selected Tail Promotion Gate 2026-06-04

## Summary

This source records the strict Q1/Q5/Q7/Q8 external-calibration promotion gate.
The gate combines the 2026-06-04 1000-replicate selected-tail context table
with the barycentric Q5 selected-tail validation. It writes explicit context
decisions: `external_admissible`, `external_diagnostic_only`, or
`undefined_support_failure`.

No context is promoted to production external calibration in this run.

## Key Points

- The promotion gate evaluated `79` selected-tail contexts from
  `selected_tail_equation_cloud_run_20260604_1000`.
- `69` contexts are `undefined_support_failure` because they do not meet the
  selected-tail support contract.
- `10` contexts are `external_diagnostic_only`.
- `7` contexts satisfy the existing context-level selected-tail law, but they
  are still not promotable because the global Q5 gate fails
  leave-one-parent-size-bin transfer and the strict relative \( \hat c \)
  precision field is not available in the selected-tail context table.
- The Q5 barycentric edge/spectral model passes replicate, case, and
  feature-family holdouts under the `0.005` residual-tail absolute-error gate,
  but fails leave-one-parent-size-bin holdout with residual-tail absolute error
  `0.250107`.
- The result keeps production behavior fail-closed for unsupported contexts
  and diagnostic-only for currently support-rich selected-tail contexts.

## Evidence

- `selected_tail_promotion_contexts.csv` is the row-level promotion decision
  table.
- `selected_tail_q5_promotion_gate.csv` records the Q5 split-level gate.
- `selected_tail_promotion_summary.csv` records the decision counts.
- `tests/validation/calibration/selected/tail/72_test_selected_tail_promotion_gate.py` verifies the gate
  blocks promotion under Q5 parent-size transfer failure and can only promote
  rows when all gates pass.

## Links

- [[open-mathematical-questions]]
- [[selected-tail-law-q5-validation-20260604]]
- [[open-question-full-diagnostic-contract-20260604]]
