---
title: Internal Calibration Q9 Q10 Q11 Debug 2026-06-05
type: source
status: reviewed
updated: 2026-06-05
sources:
  - benchmarks/results/diagnostics/internal_calibration_q9_q10_q11_debug_20260605/q9_q11_internal_support_debug.csv
  - benchmarks/results/diagnostics/internal_calibration_q9_q10_q11_debug_20260605/q10_weight_rule_debug.csv
  - benchmarks/results/diagnostics/internal_calibration_q9_q10_q11_debug_20260605/q9_q10_q11_root_cause_summary.csv
  - benchmarks/diagnostics/calibration/internal_support_threshold_validation.py
  - benchmarks/diagnostics/calibration/sibling_null_weight_rule_validation.py
  - wiki/sources/mixed-internal-calibration-sweeps-20260605.md
  - tests/validation/76_test_internal_support_threshold_validation.py
  - tests/validation/69_test_sibling_null_weight_rule_validation.py
tags:
  - source
  - diagnostics
  - calibration
  - questions
---

# Internal Calibration Q9 Q10 Q11 Debug 2026-06-05

## Summary

A focused debug pass inspected why Q9, Q10, and Q11 remain mathematically open
despite the new `CalibrationDecision` fail-closed statuses. The pass traced the
decision API, the sibling-test adjustment wrapper, and the Q10 weight-rule
diagnostic outputs.

## Key Points

- Q9/Q11 are implemented as support reporting plus opt-in threshold
  enforcement. `CalibrationDecision` reports support counts, effective sample
  size, max weight share, leave-one-record sensitivity, threshold values, and
  failure reasons.
- With `enforce_support_thresholds=True`, sparse internal contexts return
  `undefined_sparse_context`. The default production path does not enable this
  because the threshold values are not yet validated as method constants.
- The sibling-test adjustment wrapper originally called
  `decide_empirical_null_calibration` with default arguments and exposed no
  `enforce_support_thresholds` or custom `support_thresholds` parameter. The
  2026-06-05 follow-up threads those parameters through the sibling adjustment,
  sibling annotation, gate annotation, and `TreeDecomposition` path while
  keeping defaults off.
- Q10 has a predeclared rule grid, but the current real selected-geometry input
  has `support_labels_unavailable` for every rule summary, so it cannot
  validate a replacement weight rule.
- A 2026-06-05 implementation follow-up adds a dedicated internal
  support-threshold validation entrypoint and extends the Q10 weight-rule
  diagnostic to report labeled support and selected-nonnull weight leakage when
  `is_null_like` and `is_edge_blocked` columns are supplied.
- The 2026-06-05 mixed sweeps now provide labeled method-proof, binary, and
  categorical panels. They make Q10 leakage measurable and show near-zero
  leakage for the current product-BH rule, but they do not validate Q9/Q11
  thresholds as global method constants because categorical null false splits
  remain above nominal alpha among admissible contexts.
- The current product-BH rule has effective sample size `72.63`, max weight
  share `0.0250`, and weighted selected-ratio mean `7.37`; min-BH and
  geometric-mean rules increase effective sample size to `174.70`, but without
  support labels this is descriptive only.

## Evidence

- `q9_q11_internal_support_debug.csv` records the implemented surfaces and
  remaining blockers.
- `q10_weight_rule_debug.csv` annotates the Q10 rule grid with the validation
  blocker.
- `q9_q10_q11_root_cause_summary.csv` records root cause and next required
  diagnostic for Q9, Q10, and Q11.
- `internal_support_threshold_validation.py` scores permissive, current, and
  strict threshold profiles on mixed null/signal context panels and reports
  null false-split and signal-retention rates only when outcome labels are
  present.
- `sibling_null_weight_rule_validation.py` now emits selected-nonnull leakage
  shares and support-weight shares when true internal support labels are
  available.
- [[mixed-internal-calibration-sweeps-20260605]] records the labeled sweep
  outputs and interpretation.
- `tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/inflation_adjusted_sibling_tests.py`,
  `tree_break_selection/hierarchy_analysis/statistics/sibling_divergence/inflated_projected_wald_annotation/pipeline.py`,
  `tree_break_selection/hierarchy_analysis/decomposition/gates/orchestrator.py`,
  and `tree_break_selection/hierarchy_analysis/tree_decomposition.py` now
  carry the opt-in support-threshold parameters through the pipeline.

## Links

- [[recursive-method-followups-20260604]]
- [[selected-tail-promotion-gate-debug-20260605]]
- [[open-mathematical-questions]]
