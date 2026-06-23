---
title: Mixed Null Signal Geometry Validation 2026-06-06
type: source
status: reviewed
updated: 2026-06-06
sources:
  - benchmarks/diagnostics/path_b/mixed_null_signal_geometry_validation.py
  - tests/validation/81_test_mixed_null_signal_geometry_validation.py
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/mixed_null_signal_labeled_nodes.csv
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/mixed_null_signal_edges.csv
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/mixed_null_signal_model_validation.csv
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/mixed_null_signal_model_summary.csv
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/mixed_null_signal_calibration_summary.csv
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/mixed_null_signal_case_status.csv
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/mixed_null_signal_geometry_report.md
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/mixed_null_signal_geometry_summary.png
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/manifest.json
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_method_proof_20260606/mixed_null_signal_model_summary.csv
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_method_proof_20260606/mixed_null_signal_geometry_report.md
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_method_proof_20260606/manifest.json
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606_sklearn_surface/mixed_null_signal_model_summary.csv
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606_sklearn_surface/mixed_null_signal_model_validation.csv
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606_sklearn_surface/README.md
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_method_proof_20260606_sklearn_surface/mixed_null_signal_model_summary.csv
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_method_proof_20260606_sklearn_surface/mixed_null_signal_model_validation.csv
  - benchmarks/results/diagnostics/mixed_null_signal_geometry_method_proof_20260606_sklearn_surface/README.md
tags:
  - source
  - diagnostics
  - calibration
  - geometry
  - p-values
---

# Mixed Null Signal Geometry Validation 2026-06-06

## Summary

This diagnostic labels selected sibling contexts using benchmark ground truth
and validates whether recursive p-value geometry improves held-out separation
of true null sibling neighborhoods from true signal sibling neighborhoods. It
keeps the result diagnostic-only and does not install a production calibration
or traversal rule.

## Key Points

- The full run completed `110` ok TBS cases and `10` expected skips. Skips were
  the same strict calibration-support or dense continuous covariance contract
  failures seen in the recursive p-value geometry run.
- The full labeled panel contains `26369` binary sibling contexts:
  `22599` `null_context`, `2366` `mixed_context`, and `1404`
  `signal_context` rows.
- Active sibling splitting is conservative on truth-null contexts in this
  labeled full panel: corrected split rejection is `0.002611` and raw-p
  rejection is `0.003230` for `null_context` rows.
- Active sibling splitting is weak on truth-signal contexts: corrected split
  rejection is only `0.141026` and raw-p rejection is `0.143162` for
  `signal_context` rows. This points to missed signal/power and support
  scarcity, not broad null over-rejection in the successful full-suite rows.
- `mixed_context` rows are riskier than clean null rows, with corrected split
  rejection `0.109467` and raw-p rejection `0.111158`.
- The row-aligned KAK-style radius/angle/action terms are now computed inside
  the labeled sibling panel using the root selected PCA frame in null-whitened
  tangent coordinates and joined directly on `case_id,node_id`.
- A follow-up scikit-learn surface evaluation over the same cached labeled
  panel adds a regularized logistic surface and a histogram-gradient nonlinear
  surface. The `case_hash_modulo_5` split now uses a stable CRC32 fold id
  instead of Python's process-randomized `hash()`.
- Under the stable-fold scikit-learn re-evaluation, the best full-panel
  held-out signal-vs-null model is
  `sklearn_hist_gradient_edge_sibling_kak_surface`, with median AUC
  `0.985084`. It improves by `0.089310` over the `chi_square_only` median AUC
  `0.895775`.
- The regularized logistic continuous surface is also strong on the full panel:
  `sklearn_logistic_edge_sibling_kak_surface` has median AUC `0.966521`.
- The original row-aligned `kak_radius_angle_action` linear diagnostic remains
  strong but is no longer the top model after adding scikit-learn surfaces; its
  stable-fold full-panel median AUC is `0.948342`.
- The KAK columns are populated for nearly all successful full-panel rows:
  `geometry_parent_radius`, `geometry_independent_radius_fraction`, and
  `action_budget_proxy` are non-null on `26111` of `26369` rows; angular terms
  are non-null on `26003` rows.
- On the stable-fold method-proof re-evaluation, the histogram-gradient and
  logistic scikit-learn surfaces lead with median AUCs `0.858480` and
  `0.842905`, respectively. The stable-fold `chi_square_only` baseline is
  `0.594557`.
- `recursive_pvalue_geometry` remains useful but is no longer the strongest
  model in this row-aligned panel: its stable-fold full-panel median AUC is
  `0.895871`, essentially tied with `chi_square_only`.
- In the same stable-fold full-panel summary, `barycentric_edge_spectral` has
  median AUC `0.854484`, `selected_tail_baseline` has `0.852901`, and
  `context_only` has `0.852085`.
- The diagnostic avoids truth leakage: truth labels are used only for response
  labels and calibration readout, not as predictors in the recursive geometry
  model.
- These KAK terms remain diagnostic-only covariates. They show traversal and
  signal/null separability in the selected tree, but they do not by themselves
  define a production selected-tail calibration law.

## Evidence

- `benchmarks/diagnostics/path_b/mixed_null_signal_geometry_validation.py`
  implements truth-context labeling, geometry joins, predeclared covariate
  models, held-out splits, and calibration summaries.
- `tests/validation/81_test_mixed_null_signal_geometry_validation.py` verifies
  null/signal/mixed labeling, panel construction, model evaluation, and
  row-aligned KAK sibling-node geometry construction.
- `benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/mixed_null_signal_model_summary.csv`
  records the original full-suite row-aligned model comparison.
- `benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606_sklearn_surface/mixed_null_signal_model_summary.csv`
  records the stable-fold scikit-learn continuous-surface re-evaluation.
- `benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/mixed_null_signal_calibration_summary.csv`
  records split and raw-p rejection rates by truth-context label.
- `benchmarks/results/diagnostics/mixed_null_signal_geometry_full_20260606/mixed_null_signal_geometry_summary.png`
  visualizes the model AUC comparison and truth-context rejection rates.

## Links

- [[recursive-pvalue-geometry-20260606]]
- [[phase1-path-b-foundation-20260606]]
- [[path-conditioned-barycentric-action-diagnostics-20260606]]
- [[barycentric-action-equation-diagnostic-20260606]]
- [[selected-tail-law-q5-validation-20260604]]
- [[null-edge-sibling-calibration-enhancement-plan]]
