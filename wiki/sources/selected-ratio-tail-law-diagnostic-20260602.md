---
title: Selected Ratio Tail Law Diagnostic 2026-06-02
type: source
status: reviewed
updated: 2026-06-03
sources:
  - benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py
  - benchmarks/diagnostics/calibration/selected_tail_admissibility_domain.py
  - tests/validation/53_test_selected_hierarchy_geometry_covariates.py
  - tests/validation/57_test_selected_tail_admissibility_domain.py
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/case_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/geometry_summary_by_case.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_focused_300/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_focused_300/case_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_focused_300/geometry_summary_by_case.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_focused_300/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_admissibility_boundary_20260603_500/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_hierarchy_tail_law_binary_boundary_20260603_600/selected_ratio_tail_law.csv
  - raw/assets/benchmark-results/selected_tail_admissibility_domain_20260603/context_admissibility_domain.csv
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

A focused 300-replicate follow-up over multi-case source families shows that
admissibility is reachable but narrow. Two `gaussian_blobs` contexts pass the
support contract; categorical contexts approach the support threshold; binary
template contexts remain fragmented.

A 500-replicate boundary expansion over comparable Gaussian and categorical
source families shows that categorical support is reachable under the same
exact context rule. A 600-replicate binary expansion shows that binary
projection-1 support is also reachable. The current admissible domain is
narrow selected-tail contexts with small parent size and high edge action, not
a general external calibration law.

## Key Points

- The production tail-law contract uses \(\alpha_{\mathrm{sib}}=0.01\) and
  requires at least `499` independent matching simulations, at least `499`
  matched selected records, and held-out exceedance standard error at most
  `0.002`.
- The independent simulation unit is explicitly recorded as
  `selected_hierarchy_simulation_id`, the pair of case id and replicate index.
  This prevents source-family contexts from collapsing independent
  case-replicates that share the same numeric `replicate_index`.
- The broad run used `200` replicates over nine requested cases. Eight cases
  completed. `sbm_moderate` was skipped because the selected-hierarchy null
  generator does not own the precomputed KL tree-distance contract.
- The selected-ratio tail table contains `104` contexts. `39` contexts have
  descriptive held-out tail-law folds; `65` have no valid tail-law folds.
- `0` of `104` contexts are production-admissible. Every context fails the
  independent matching-simulation requirement, and most also fail matched
  record count and held-out tail standard-error requirements.
- After counting independent case-replicates explicitly, the largest context
  support is `376` matching simulations for `gaussian_blobs`, still below the
  `499` production threshold.
- High-support small-node, high-edge-action contexts often have held-out
  exceedance near the target `0.01`. Ten descriptive contexts have absolute
  exceedance error at most `0.001` and held-out standard error at most
  `0.002`.
- Sparse root or low-edge-action contexts are unstable. The median absolute
  held-out exceedance error among descriptive contexts is about `0.0052`, and
  the worst sparse context has absolute error about `0.323`.
- The case-level selected-ratio scale remains large and family-dependent:
  mean \(R\) is about `30.9` for `gauss_clear_medium`, `62.9` for
  `gauss_null_large`, `109.3` for `dim_diffuse_6c_136f`, `210.1` for
  `phylo_dna_8taxa_med_mut`, and `579.8` for
  `cat_highd_3cat_500feat`.
- The result sharpens the mathematical target. A high global tail-ranking AUC
  is not enough; a production external law would need admissible within-context
  support and calibrated absolute tail probabilities.
- The focused 300-replicate run used `gauss_null_large`,
  `gauss_clear_medium`, `binary_low_noise_4c`,
  `overlap_heavy_4c_med_feat`, `cat_highcard_20cat_4c`, and
  `cat_highd_3cat_500feat`. It produced `72` tail-law contexts, of which `28`
  had descriptive held-out folds.
- In the focused run, exactly two contexts are production-admissible:
  `gaussian_blobs`, `bernoulli`, `small_0_0.25`, `edge_action_ge8`, with
  sibling projection dimension `1` and `2`. Their independent matching
  simulations are `564` and `563`, held-out exceedance rates are about
  `0.00998` and `0.01000`, and held-out standard errors are about `0.00048`
  and `0.00083`.
- The focused categorical source family remains just below the support
  threshold. Its strongest context has `476` independent simulations,
  exceedance rate about `0.01009`, and standard error about `0.00045`, but it
  still fails the `499` independent-simulation rule.
- The focused binary-template family remains support-fragmented. Its strongest
  context has `296` independent simulations, below the production threshold.
- The focused result changes the open question from "can any context become
  admissible?" to "which exact contexts are admissible, and how should
  non-admissible categorical, root, medium-parent, and binary contexts be
  handled without borrowing or fallback?"
- The 500-replicate boundary expansion used `gauss_null_large`,
  `gauss_clear_medium`, `cat_highcard_20cat_4c`, and
  `cat_highd_3cat_500feat`. Four contexts are production-admissible:
  `gaussian_blobs` and `categorical_multinomial`, both in `small_0_0.25`,
  `edge_action_ge8`, with sibling projection dimension `1` and `2`.
- The admissible categorical contexts have `776` and `689` independent
  matching simulations, held-out exceedance near `0.01`, and held-out standard
  errors below `0.001`. This overturns the earlier "categorical remains just
  below support" statement for the targeted boundary panel.
- The 600-replicate binary boundary expansion used `binary_low_noise_4c` and
  `overlap_heavy_4c_med_feat`. One `binary_template` context is
  production-admissible: `small_0_0.25`, `edge_action_ge8`, with sibling
  projection dimension `1`. The corresponding projection-2 context remains
  support-limited at `445/499` simulations.
- Root, medium-parent, large-parent, lower-edge-action, binary projection-2,
  continuous, and precomputed-distance contexts remain outside the current
  admissible production domain.

## Evidence

- `benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py`
  implements the descriptive selected-ratio tail-law table and explicit
  production-admissibility checks.
- `tests/validation/53_test_selected_hierarchy_geometry_covariates.py`
  validates context support reporting, held-out exceedance estimation,
  admissibility failures, and output creation.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/manifest.json`
  records the run seed, cases, independent simulation id column, context
  columns, edge-action bins, alpha, and production support thresholds.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/case_summary.csv`
  records case completion status, selected-record counts, and the explicit SBM
  skip reason.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/geometry_summary_by_case.csv`
  records selected-ratio and angular/eigenvalue summaries by completed case.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_broad_200/selected_ratio_tail_law.csv`
  records context-level support, held-out exceedance rates, admissibility
  failures, and diagnostic status.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_focused_300/manifest.json`
  records the focused 300-replicate multi-case source-family run.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_20260602_focused_300/selected_ratio_tail_law.csv`
  records the first production-admissible diagnostic contexts and the remaining
  support-fragmented contexts.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_admissibility_boundary_20260603_500/selected_ratio_tail_law.csv`
  records the boundary expansion with admissible categorical and Gaussian
  small-parent, high-edge-action contexts.
- `raw/assets/benchmark-results/selected_hierarchy_tail_law_binary_boundary_20260603_600/selected_ratio_tail_law.csv`
  records the binary boundary expansion with one admissible binary projection-1
  small-parent, high-edge-action context.
- `raw/assets/benchmark-results/selected_tail_admissibility_domain_20260603/context_admissibility_domain.csv`
  combines broad, focused, and boundary selected-tail contexts into one
  admissibility-domain table.

## Links

- [[selected-hierarchy-null-support-contract]]
- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-geometric-law-map]]
- [[selected-hierarchy-geometry-covariates-20260602]]
- [[selected-tail-admissibility-domain-20260603]]
- [[open-mathematical-questions]]
