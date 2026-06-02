---
title: Selected Ratio Tail Law Diagnostic 2026-06-02
type: source
status: reviewed
updated: 2026-06-02
sources:
  - benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py
  - tests/validation/53_test_selected_hierarchy_geometry_covariates.py
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/case_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/geometry_summary_by_case.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/selected_ratio_tail_law.csv
tags:
  - source
  - calibration
  - selection
  - tail-law
---

# Selected Ratio Tail Law Diagnostic 2026-06-02

## Summary

This diagnostic tests whether a selected-ratio tail law can be estimated
inside explicit selected-hierarchy contexts. The object is
\[
R_u=\frac{W_u}{a_u\nu_u},
\]
evaluated only after same-data hierarchy construction, child-parent edge
opening, and focal sibling selection. The context is `source_family`,
`feature_family`, `parent_size_bin`, `sibling_projection_dimension`, and
`edge_action_bin`.

The result is diagnostic-only. It does not define a production external
calibration model, scalar inflation fallback, context-borrowing rule, or
application default. No context in the 200-replicate broad run is
production-admissible under the predeclared support contract.

## Key Points

- The production tail-law contract uses \(\alpha_{\mathrm{sib}}=0.01\) and
  requires at least `499` independent matching simulations, at least `499`
  matched selected records, and held-out exceedance standard error at most
  `0.002`.
- The broad run used `200` replicates over nine requested cases. Eight cases
  completed. `sbm_moderate` was skipped because the selected-hierarchy null
  generator does not own the precomputed KL tree-distance contract.
- The selected-ratio tail table contains `104` contexts. `39` contexts have
  descriptive held-out tail-law folds; `65` have no valid tail-law folds.
- `0` of `104` contexts are production-admissible. Every context fails the
  independent matching-simulation requirement, and most also fail matched
  record count and held-out tail standard-error requirements.
- High-support small-node, high-edge-action contexts often have held-out
  exceedance near the target `0.01`. Ten descriptive contexts have absolute
  exceedance error at most `0.001` and held-out standard error at most
  `0.002`.
- Sparse root or low-edge-action contexts are unstable. The median absolute
  held-out exceedance error among descriptive contexts is about `0.0044`, and
  the worst sparse context has absolute error about `0.323`.
- The case-level selected-ratio scale remains large and family-dependent:
  mean \(R\) is about `30.9` for `gauss_clear_medium`, `62.9` for
  `gauss_null_large`, `109.3` for `dim_diffuse_6c_136f`, `210.1` for
  `phylo_dna_8taxa_med_mut`, and `579.8` for
  `cat_highd_3cat_500feat`.
- The result sharpens the mathematical target. A high global tail-ranking AUC
  is not enough; a production external law would need admissible within-context
  support and calibrated absolute tail probabilities.

## Evidence

- `benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py`
  implements the descriptive selected-ratio tail-law table and explicit
  production-admissibility checks.
- `tests/validation/53_test_selected_hierarchy_geometry_covariates.py`
  validates context support reporting, held-out exceedance estimation,
  admissibility failures, and output creation.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/manifest.json`
  records the run seed, cases, context columns, edge-action bins, alpha, and
  production support thresholds.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/case_summary.csv`
  records case completion status, selected-record counts, and the explicit SBM
  skip reason.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/geometry_summary_by_case.csv`
  records selected-ratio and angular/eigenvalue summaries by completed case.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/selected_ratio_tail_law.csv`
  records context-level support, held-out exceedance rates, admissibility
  failures, and diagnostic status.

## Links

- [[selected-hierarchy-null-support-contract]]
- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-geometric-law-map]]
- [[selected-hierarchy-geometry-covariates-20260602]]
- [[open-mathematical-questions]]
