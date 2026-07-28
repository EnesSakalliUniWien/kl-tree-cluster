---
title: Selected Hierarchy External Calibration Contract 2026-06-02
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/selected/hierarchy/selected_hierarchy_external_calibration_contract.py
  - tests/validation/calibration/selected/hierarchy/52_test_selected_hierarchy_external_calibration_contract.py
  - raw/assets/benchmark-results/selected_hierarchy_external_contract_20260602_500/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_external_contract_20260602_500/case_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_external_contract_20260602_500/external_calibration_contract.csv
tags:
  - source
  - calibration
  - selection
  - validation
---

# Selected Hierarchy External Calibration Contract 2026-06-02

## Summary

This diagnostic defines a production-admissibility contract for an external
selected-hierarchy calibration object and tests whether scalar mean inflation is
distributionally adequate. It is diagnostic-only. Rows that fail admissibility
define no external production calibration estimate.

The conditioning scope is
`same_data_selected_hierarchy_edge_path_open`: null replicates rebuild the
hierarchy, rerun edge tests, and keep selected sibling records whose
child-parent edge path is open. The stratum variables are case, feature family,
sample count \(n\), feature dimension \(p\), sibling projection dimension \(k\),
and parent-size bin.

## Key Points

- The default production tail-resolution contract at
  `SIBLING_ALPHA = 0.01` requires at least `499` independent matching
  simulations and `499` matched selected records. This is the rule
  \(1/(m+1)\le 0.2\alpha\).
- No 500-replicate row is production-admissible. The largest independent
  matching-simulation count is `470`, so every row fails at least the
  matching-simulation tail-resolution requirement.
- Several rows have many selected records and low relative simulation SE for
  \(c\), but still fail because production tail calibration needs independent
  matching simulations, not only many selected records from fewer simulations.
- Among rows with at least 100 matching simulations, scalar-\(c\) p-values have
  rejection rate `0.0` at `SIBLING_ALPHA = 0.01`. Scalar mean scaling is
  therefore conservative in this run.
- Scalar-\(c\) p-values are not distribution-calibrated in the reliable rows:
  the KS p-values against Uniform(0,1) are tiny for the supported rows. The
  selected-ratio law is not a scaled chi-square law after mean rescaling.
- The estimator-family decision is therefore:

```text
production external estimator:
  undefined_not_production_admissible for all rows

descriptive shape:
  reliable rows indicate full selected-ratio tail-law modeling, not scalar-c
  mean scaling, if an external production model is ever pursued.
```

## Evidence

- `benchmarks/diagnostics/calibration/selected/hierarchy/selected_hierarchy_external_calibration_contract.py`
  implements the admissibility contract and scalar-vs-tail summary.
- `tests/validation/calibration/selected/hierarchy/52_test_selected_hierarchy_external_calibration_contract.py`
  verifies under-supported rows fail closed, scalar-\(c\) can be accepted when
  the synthetic shape is compatible, and heavy selected tails require a tail
  law.
- `raw/assets/benchmark-results/selected_hierarchy_external_contract_20260602_500/external_calibration_contract.csv`
  records per-stratum admissibility, failure reasons, scalar-\(c\) p-value
  diagnostics, and estimator-family decisions.
- `raw/assets/benchmark-results/selected_hierarchy_external_contract_20260602_500/manifest.json`
  records thresholds, run seed, cases, and output paths.

## Links

- [[selected-hierarchy-null-support-contract]]
- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-stratification-diagnostic-20260602]]
- [[open-mathematical-questions]]
