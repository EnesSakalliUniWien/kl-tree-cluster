---
title: Traversal Guard Validation Panel 2026-06-13
type: source
status: reviewed
updated: 2026-06-13
sources:
  - benchmarks/diagnostics/calibration/traversal_guard_validation_panel.py
  - tests/validation/90_test_traversal_guard_validation_panel.py
  - benchmarks/diagnostics/math_trace/path_conditioned_barycentric_action.py
  - wiki/analyses/null-edge-sibling-calibration-enhancement-plan.md
tags:
  - source
  - diagnostics
  - traversal
  - calibration
---

# Traversal Guard Validation Panel 2026-06-13

## Summary

The traversal-geometry roadmap phase now has a reusable validation panel.
`traversal_guard_validation_panel.py` evaluates high-action angular-shell guard
thresholds on precomputed traversal rows labeled as `pure_fragment`,
`mixed_signal`, or `true_signal`.

## Key Points

- The panel uses the same action-budget, angle-to-leading-axis, and independent
  radius-fraction variables introduced by the path-conditioned barycentric
  action diagnostics.
- Guard summaries report pure-fragment precision, pure-fragment flag rate,
  signal-context flag rate, and cost-weighted utility.
- A guard can be marked only as `guard_validation_candidate`, not as a
  production traversal rule. Low precision, excessive signal blocking, or too
  few flagged rows keep the guard `diagnostic_only_guard`.
- This keeps traversal geometry separate from selected-tail p-value
  calibration.

## Evidence

- `benchmarks/diagnostics/calibration/traversal_guard_validation_panel.py`
  defines the dataframe API, CLI, row output, summary output, and manifest.
- `tests/validation/90_test_traversal_guard_validation_panel.py` verifies a
  high-precision diagnostic candidate, a low-precision diagnostic-only guard,
  and output writing.
- `benchmarks/diagnostics/math_trace/path_conditioned_barycentric_action.py`
  remains the source of the underlying action/angle diagnostic coordinates.

## Links

- [[null-edge-sibling-calibration-enhancement-plan]]
- [[sibling-null-calibration-panel-20260613]]
- [[open-mathematical-questions]]
