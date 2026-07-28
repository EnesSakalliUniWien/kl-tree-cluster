---
title: Overlap Threshold Stability Contract 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_threshold_stability_contract.py
  - tests/validation/calibration/overlap/118_test_overlap_threshold_stability_contract.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/threshold_stability_contract/overlap_threshold_stability_contract_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/threshold_stability_contract/overlap_threshold_stability_contract_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/threshold_stability_contract/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - thresholds
---

# Overlap Threshold Stability Contract 2026-06-14

## Summary

`overlap_threshold_stability_contract.py` classifies the overlap threshold
hierarchy after transfer testing. It separates stable reporting candidates,
diagnostic-only guards, non-transferable focused cutpoints, and stages that
require a selected-family law. It is diagnostic-only and does not change
production behavior.

## Key Points

- The runner writes `overlap_threshold_stability_contract_rows.csv`,
  `overlap_threshold_stability_contract_summary.csv`, and `manifest.json`.
- The continuous structural stable-accept stage is classified as
  `stable_reporting_candidate`: report stable regions only, not full traversal
  calibration.
- The weak fragment-risk block is classified as
  `diagnostic_only_guard_candidate`: useful focused guard experiment, but not
  transfer-established.
- The three residual selected-family cutpoints are all classified as
  `nontransferable_focused_cutpoint`:
  residual null evidence, residual non-recovery structural evidence, and
  strict all-negative structural evidence.
- The residual null-evidence-only mixture is classified as
  `selected_family_law_required`.
- The summary has `stage_count = 6`, `stable_reporting_count = 1`,
  `diagnostic_only_count = 1`, `nontransferable_count = 3`, and
  `law_required_count = 4`.
- Final promotion status is `fail_closed_selected_family_law_required`.
- Method implication: current thresholds are useful for reporting and
  diagnosis, but not for production promotion. Production needs a
  selected-family structural recovery law or a separate predeclared validation
  path.

## Evidence

- `tests/validation/calibration/overlap/118_test_overlap_threshold_stability_contract.py` verifies
  stage classification, selected-family-law blocking, final fail-closed status,
  and output writing.
- Verification passed:
  `pytest tests/validation/calibration/overlap/118_test_overlap_threshold_stability_contract.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_threshold_stability_contract.py tests/validation/calibration/overlap/118_test_overlap_threshold_stability_contract.py`.

## Links

- [[overlap-threshold-hierarchy-20260614]]
- [[overlap-residual-threshold-transfer-20260614]]
- [[open-mathematical-questions]]
