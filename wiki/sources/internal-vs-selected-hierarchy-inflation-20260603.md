---
title: Internal vs Selected Hierarchy Inflation 2026-06-03
type: source
status: reviewed
updated: 2026-06-03
sources:
  - benchmarks/diagnostics/calibration/internal_vs_selected_hierarchy_inflation.py
  - tests/validation/56_test_internal_vs_selected_hierarchy_inflation.py
  - raw/assets/benchmark-results/internal_vs_selected_hierarchy_inflation_20260603/manifest.json
  - raw/assets/benchmark-results/internal_vs_selected_hierarchy_inflation_20260603/internal_vs_selected_hierarchy_inflation.csv
tags:
  - source
  - calibration
  - selection
  - diagnostics
---

# Internal vs Selected Hierarchy Inflation 2026-06-03

## Summary

This diagnostic compares the active internal empirical-null inflation estimate
with a regenerated same-data selected-hierarchy null for the same observed root
sibling target. It is diagnostic-only: selected-hierarchy estimates remain
descriptive evidence and are not installed as production fallback calibration.

The run used `25` selected-hierarchy null replicates for
`gauss_null_large`, `dim_diffuse_6c_136f`, `binary_low_noise_4c`, and
`cat_highcard_20cat_4c`, with target mode `root` and context match
`projection`.

## Key Points

- `dim_diffuse_6c_136f` has internal empirical-null support and close
  agreement between internal and selected-hierarchy scale:
  \(c_{\mathrm{internal}}=85.27\), \(c_{\mathrm{sel}}=86.23\), ratio `0.989`.
  The required scale to block at `SIBLING_ALPHA = 0.01` is `82.21`.
- `gauss_null_large` has no internal strict/stopped empirical-null support,
  but the selected-hierarchy diagnostic has matched records with
  \(c_{\mathrm{sel}}=58.21\). This remains descriptive because internal
  production support is absent.
- `cat_highcard_20cat_4c` also has no internal support. Its selected-hierarchy
  descriptive scale is \(c_{\mathrm{sel}}=110.46\), below the scale `144.32`
  required to block the observed root target by scalar mean scaling.
- `binary_low_noise_4c` has both supports, but internal scale `18.83` is below
  selected-hierarchy scale `30.56`; both are far below the observed target's
  required blocking scale `218.73`, so the signal remains rejected.
- The result supports the current mathematical interpretation: selected
  hierarchy can explain large scale in diffuse Gaussian root contexts, but
  selected-hierarchy estimates must stay separate from production calibration
  until admissibility and tail-law requirements are satisfied.

## Evidence

- `benchmarks/diagnostics/calibration/internal_vs_selected_hierarchy_inflation.py`
  implements the comparison between internal support, selected-hierarchy
  support, required blocking scale, and cross-scale ratios.
- `tests/validation/56_test_internal_vs_selected_hierarchy_inflation.py`
  checks the projected-Wald blocking-scale calculation and a smoke run that
  writes explicit support statuses.
- `raw/assets/benchmark-results/internal_vs_selected_hierarchy_inflation_20260603/internal_vs_selected_hierarchy_inflation.csv`
  records the four-case diagnostic table.
- `raw/assets/benchmark-results/internal_vs_selected_hierarchy_inflation_20260603/manifest.json`
  records the seed, replicate count, target mode, context match, and
  diagnostic-only note.

## Links

- [[selected-hierarchy-null-support-contract]]
- [[selected-hierarchy-external-calibration-contract-20260602]]
- [[selected-hierarchy-selection-geometry]]
- [[open-mathematical-questions]]
