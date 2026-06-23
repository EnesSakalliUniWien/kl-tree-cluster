---
title: Mixed Internal Calibration Sweeps 2026-06-05
type: source
status: reviewed
updated: 2026-06-05
sources:
  - benchmarks/diagnostics/calibration/mixed_internal_calibration_sweep.py
  - benchmarks/results/diagnostics/mixed_internal_calibration_sweep_method_proof_20260605/mixed_q10_weight_rule_summary.csv
  - benchmarks/results/diagnostics/mixed_internal_calibration_sweep_method_proof_20260605/mixed_internal_support_threshold_summary.csv
  - benchmarks/results/diagnostics/mixed_internal_calibration_sweep_binary_20260605/mixed_q10_weight_rule_summary.csv
  - benchmarks/results/diagnostics/mixed_internal_calibration_sweep_binary_20260605/mixed_internal_support_threshold_summary.csv
  - benchmarks/results/diagnostics/mixed_internal_calibration_sweep_categorical_20260605/mixed_q10_weight_rule_summary.csv
  - benchmarks/results/diagnostics/mixed_internal_calibration_sweep_categorical_20260605/mixed_internal_support_threshold_summary.csv
tags:
  - source
  - diagnostics
  - calibration
  - validation
---

# Mixed Internal Calibration Sweeps 2026-06-05

## Summary

A mixed null/signal diagnostic runner generated labeled internal-calibration
panels for method-proof, binary, and categorical benchmark suites. The sweep
collects sibling records from production TBS trees, attaches internal support
labels (`is_null_like`, `is_edge_blocked`), derives conservative true
null/signal context labels from benchmark cluster labels, and runs the Q9/Q10
diagnostics over the resulting panels.

## Key Points

- The method-proof sweep ran `1000` case-replicates, producing `163100` Q10
  sibling records and `35760` threshold contexts.
- The binary sweep ran `840` case-replicates, producing `225639` Q10 sibling
  records and `69265` threshold contexts.
- The categorical sweep ran `290` case-replicates, producing `58861` Q10
  sibling records and `20083` threshold contexts.
- Q10 selected-nonnull weight leakage is measurable and very small for the
  current product-BH rule: about `4.73e-10` in method-proof, `1.82e-9` in
  binary, and `4.21e-9` in categorical. The hard null-indicator rule has zero
  selected-nonnull leakage by construction, but this is still diagnostic and
  not a production replacement.
- Current-default Q9/Q11 thresholds are not validated as global method
  constants. Binary current-default null false-split rate among admissible
  contexts is `0.00008`, but categorical is `0.029816`, above the nominal
  sibling alpha `0.01`; method-proof has zero null false splits but only
  `0.16343` signal retention among admissible signal contexts.
- Stricter thresholds reduce admissible contexts and do not fix categorical
  null false splits in this panel: categorical strict null false-split rate is
  `0.035465` with signal retention `0.288210`.

## Evidence

- `mixed_q10_sibling_records.csv` files contain row-level sibling records with
  edge p-values, selected-ratio values, internal support labels, and true
  null/signal context labels.
- `mixed_q10_weight_rule_summary.csv` files report weight concentration and
  selected-nonnull leakage for each predeclared Q10 rule.
- `mixed_internal_support_contexts.csv` files contain focal calibration
  support metrics and adjusted split outcomes.
- `mixed_internal_support_threshold_summary.csv` files compare permissive,
  current-default, and strict threshold profiles.

## Links

- [[internal-calibration-q9-q10-q11-debug-20260605]]
- [[open-mathematical-questions]]
