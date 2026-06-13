---
title: Selected Tail Promotion Gate Debug 2026-06-05
type: source
status: reviewed
updated: 2026-06-05
sources:
  - benchmarks/results/diagnostics/selected_tail_promotion_gate_debug_20260605/promotion_gate_sensitivity_summary.csv
  - benchmarks/results/diagnostics/selected_tail_promotion_gate_debug_20260605/predicate_blocker_counts.csv
  - benchmarks/results/diagnostics/selected_tail_promotion_gate_debug_20260605/context_law_candidates_blocked_by_global_gates.csv
  - benchmarks/results/diagnostics/selected_tail_promotion_gate_debug_20260605/parent_size_balance_external_candidates.csv
tags:
  - source
  - diagnostics
  - calibration
  - questions
---

# Selected Tail Promotion Gate Debug 2026-06-05

## Summary

A focused debug pass examined why the strict selected-tail external promotion
gate produced zero `external_admissible` contexts. The pass decomposed the
gate predicates and ran sensitivity variants that force only Q5, only c-hat
precision, both, or neither.

## Key Points

- The zero `external_admissible` result is expected under the current strict
  gate, not a boolean parsing or support-count bug.
- Baseline predicate counts over `79` contexts are: `10` support-contract
  passes, `7` context-level tail-law admissible rows, `8` tail-precision
  passes, `17` absolute-tail-error passes, `0` Q5 global passes, and `0`
  c-hat precision passes.
- Six rows satisfy the context-level support, admissibility, precision, and
  absolute-error predicates, but they are blocked by the global Q5 parent-size
  transfer failure and missing c-hat precision metadata.
- Allowing missing c-hat precision alone still gives zero
  `external_admissible` contexts because the Q5 global gate fails.
- Forcing Q5 pass alone still gives zero `external_admissible` contexts
  because the selected-tail table lacks
  `selected_hierarchy_c_hat_relative_simulation_se`.
- Forcing both Q5 pass and missing c-hat allowance would produce six
  `external_admissible` rows; this is a sensitivity check only, not calibration
  evidence.
- The later parent-size/balance diagnostic contains three narrower
  Gaussian/Bernoulli projection-2 high-edge contexts with c-hat precision and
  parent-size holdout passing, but that schema is not yet wired into the
  selected-tail promotion gate.

## Evidence

- `promotion_gate_sensitivity_summary.csv` records the baseline and forced
  sensitivity decisions.
- `predicate_blocker_counts.csv` records predicate-level pass counts.
- `context_law_candidates_blocked_by_global_gates.csv` records the six
  context-law candidates blocked by Q5 and c-hat metadata.
- `parent_size_balance_external_candidates.csv` records the three narrower
  balance-conditioned diagnostic candidates.

## Links

- [[selected-tail-promotion-gate-20260604]]
- [[selected-tail-parent-size-balance-stability-20260604]]
- [[open-mathematical-questions]]
