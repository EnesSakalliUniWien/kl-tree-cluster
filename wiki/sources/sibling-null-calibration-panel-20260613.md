---
title: Sibling Null Calibration Panel 2026-06-13
type: source
status: reviewed
updated: 2026-06-13
sources:
  - benchmarks/diagnostics/calibration/sibling_null_calibration_panel.py
  - tests/validation/89_test_sibling_null_calibration_panel.py
  - wiki/analyses/null-edge-sibling-calibration-enhancement-plan.md
tags:
  - source
  - diagnostics
  - calibration
  - sibling
---

# Sibling Null Calibration Panel 2026-06-13

## Summary

The sibling-null calibration roadmap phase now has an executable diagnostic
contract. `sibling_null_calibration_panel.py` scores precomputed sibling-test
rows separately for `strict_null`, `stopped_edge_null`,
`selected_nonnull_only`, and `external_selected_tail_context` roles.

## Key Points

- Strict-null and stopped-edge null rows are summarized against nominal alpha
  with an explicit tolerance.
- Selected-nonnull-only rows are treated as descriptive signal-retention rows,
  not null-calibration evidence.
- External selected-tail rows remain fail-closed unless the input explicitly
  marks an external rule as admissible; even then the status is descriptive and
  not a production promotion.
- Unknown sibling context roles are rejected to prevent pooled post-selection
  summaries from hiding the role that created the observed p-value behavior.

## Evidence

- `benchmarks/diagnostics/calibration/sibling_null_calibration_panel.py`
  defines the dataframe API, CLI, row output, summary output, and manifest.
- `tests/validation/89_test_sibling_null_calibration_panel.py` verifies
  nominal strict-null behavior, selected stopped-edge inflation, selected
  nonnull descriptive status, external fail-closed status, and output writing.
- [[null-edge-sibling-calibration-enhancement-plan]] names sibling-null support
  and external selected-tail calibration as the second roadmap phase.

## Links

- [[null-edge-sibling-calibration-enhancement-plan]]
- [[edge-null-calibration-panel-20260613]]
- [[open-mathematical-questions]]
