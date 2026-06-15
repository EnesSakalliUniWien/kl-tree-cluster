---
title: Overlap Diagnostic Traversal Policy 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - benchmarks/diagnostics/calibration/overlap_diagnostic_traversal_policy.py
  - tests/validation/113_test_overlap_diagnostic_traversal_policy.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/diagnostic_traversal_policy/overlap_diagnostic_traversal_policy_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/diagnostic_traversal_policy/overlap_diagnostic_traversal_policy_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/diagnostic_traversal_policy/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - multiscale
---

# Overlap Diagnostic Traversal Policy 2026-06-14

## Summary

`overlap_diagnostic_traversal_policy.py` composes the continuous structural
decision zones with the fragment-risk guard scan. It is diagnostic-only and
does not change production traversal. The policy layer makes the current
method direction explicit: stable same-subspace rows are stable-region accepts,
high fragment-risk weak-zone rows are guard-blocked, and the remaining weak
rows stay unstable/multi-scale pending a selected-family recovery law.

## Key Points

- The runner writes `overlap_diagnostic_traversal_policy_rows.csv`,
  `overlap_diagnostic_traversal_policy_summary.csv`, and `manifest.json`.
- The focused run used `fragment_risk_proxy_score >= 1.252728536810977`, the
  exact best candidate threshold from [[overlap-fragment-risk-guard-20260614]].
- The resulting diagnostic action counts are:
  `stable_region_accept = 12`, `weak_fragment_guard_blocked = 15`,
  `weak_unstable_multiscale_zone = 33`, and `nonaccepted_or_leaf = 112`.
- `stable_region_accept` remains clean: `12` truth-aligned signal rows, no
  selected-null rows, and no truth-misaligned rows.
- `weak_fragment_guard_blocked` contains `6` selected-null rows, `8` of `9`
  fragment-like rows, `1` diffuse/wrong row, and no truth-recovery rows. This
  is the useful part of the fragment-risk guard.
- `weak_unstable_multiscale_zone` still contains `17` selected-null rows,
  `5` truth-recovery rows, `1` fragment-like row, `10` diffuse/wrong rows,
  and `11` truth-misaligned signal rows. This is the unresolved selected-family
  mixture and cannot be promoted by the current evidence.
- Median minimum continuous context margin is positive in
  `stable_region_accept` (`0.017777`), more negative in
  `weak_fragment_guard_blocked` (`-0.017644`), and still negative in
  `weak_unstable_multiscale_zone` (`-0.007969`).

## Evidence

- `tests/validation/113_test_overlap_diagnostic_traversal_policy.py` verifies
  stable accept, fragment-guard blocked, unstable weak-zone, nonaccepted action
  assignment; summary guard-role counts; and output writing.
- Verification passed:
  `pytest tests/validation/113_test_overlap_diagnostic_traversal_policy.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_diagnostic_traversal_policy.py tests/validation/113_test_overlap_diagnostic_traversal_policy.py`.

## Links

- [[overlap-structural-decision-zones-20260614]]
- [[overlap-fragment-risk-guard-20260614]]
- [[overlap-recovery-proxy-separability-20260614]]
- [[open-mathematical-questions]]
