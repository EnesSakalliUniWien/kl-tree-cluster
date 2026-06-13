---
title: Selected Tail Law Q5 Validation 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - benchmarks/diagnostics/calibration/selected_tail_law_q5_validation.py
  - tests/validation/68_test_selected_tail_law_q5_validation.py
  - raw/assets/benchmark-results/selected_tail_law_q5_validation_20260604/manifest.json
  - raw/assets/benchmark-results/selected_tail_law_q5_validation_20260604/q5_selected_tail_law_validation.csv
  - raw/assets/benchmark-results/selected_tail_law_q5_validation_20260604/q5_selected_tail_law_summary.csv
tags:
  - source
  - calibration
  - selection
  - diagnostics
---

# Selected Tail Law Q5 Validation 2026-06-04

## Summary

This diagnostic fits predeclared selected-tail law candidates for
\(\log R\), where \(R\) is the selected sibling statistic divided by its
unselected reference expectation. The tested Q5 predictors are edge severity,
parent size, feature-parent aspect ratio, sibling projection dimension, feature
family, parent-size bin, barycentric leverage, and spectral geometry. The input
is the existing 300-replicate row-level selected geometry table from the
topology-refinement study.

The diagnostic evaluates out-of-fold mean prediction, top-tail ranking, and a
residual-tail threshold at \(\alpha=0.01\). It is explicitly
`diagnostic_q5_selected_tail_law_not_calibration`; it does not create a
production external calibration path.

## Key Points

- The run evaluates `128,528` selected sibling records from four cases:
  two Gaussian/blob Bernoulli-feature cases and two categorical multinomial
  cases.
- The rerun adds derived barycentric predictors:
  `barycentric_balance`, `log_barycentric_leverage`, and
  `log_sampling_variance_scale`, computed from left and right child sample
  sizes, plus `log_feature_parent_aspect_ratio`.
- Under replicate-modulo holdout, the barycentric full law is strong:
  holdout \(R^2 = 0.344\), top-tail AUC `0.975771`, and residual-tail
  exceedance `0.010013` at target alpha `0.01`.
- Under leave-one-case-out, the barycentric edge-plus-spectral model gives the
  best absolute tail result among the tested stable candidates:
  holdout \(R^2 = 0.342\), AUC `0.999128`, and residual-tail exceedance
  `0.013795`.
- Across all four split strategies, the barycentric edge-plus-spectral model
  has the smallest median residual-tail absolute error: `0.002386`, compared
  with `0.007755` for the non-barycentric edge-plus-spectral model.
- The full all-variable laws remain unstable under parent-size and
  feature-family transfer. The barycentric full model has median holdout
  \(R^2 = -0.650\), median top-tail AUC `0.865626`, and median residual-tail
  absolute error `0.333493`.
- The result answers Q5 more sharply in diagnostic form: barycentric leverage
  helps the low-dimensional edge/spectral tail model, but large all-variable
  linear laws still do not validate a production selected-tail calibration
  rule.

## Evidence

- `benchmarks/diagnostics/calibration/selected_tail_law_q5_validation.py`
  implements the predeclared Q5 models, held-out validation splits, and
  residual-tail exceedance checks.
- `tests/validation/68_test_selected_tail_law_q5_validation.py` verifies
  predictor construction, holdout outputs, summary outputs, and file writing.
- `raw/assets/benchmark-results/selected_tail_law_q5_validation_20260604/q5_selected_tail_law_validation.csv`
  records per-model validation metrics for replicate, case, feature-family,
  and parent-size-bin holdouts.
- `raw/assets/benchmark-results/selected_tail_law_q5_validation_20260604/q5_selected_tail_law_summary.csv`
  records compact per-model summaries.
- `raw/assets/benchmark-results/selected_tail_law_q5_validation_20260604/manifest.json`
  records the input table, alpha, tail quantile, fold count, and diagnostic
  interpretation.

## Links

- [[selected-hierarchy-geometric-law-map]]
- [[selected-hierarchy-null-support-contract]]
- [[open-mathematical-questions]]
