---
title: Selected Hierarchy Geometry Covariates 2026-06-02
type: source
status: reviewed
updated: 2026-06-02
sources:
  - benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py
  - tests/validation/53_test_selected_hierarchy_geometry_covariates.py
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/case_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/geometry_summary_by_case.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/covariate_relationships.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/covariate_block_models.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/candidate_equations.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_holdout_100/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_holdout_100/candidate_equation_holdout.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_broad_200/manifest.json
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_broad_200/case_summary.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_broad_200/geometry_summary_by_case.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_broad_200/candidate_equations.csv
  - raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_broad_200/candidate_equation_holdout.csv
tags:
  - source
  - calibration
  - geometry
  - selection
---

# Selected Hierarchy Geometry Covariates 2026-06-02

## Summary

This diagnostic builds row-level tree, edge-selection, eigenvalue, and angular
geometry for selected sibling records under regenerated same-data null
hierarchies, then writes compact relationship and summary tables by default.
It is descriptive evidence for the selected-hierarchy external null question.
It does not define a production external calibration model, scalar inflation
fallback, or borrowing rule.

The 100-replicate run used `gauss_null_large`, `gauss_clear_medium`,
`binary_low_noise_4c`, and `cat_clear_3cat_4c`. All four cases completed and
the compact summaries aggregate `28,545` selected geometry records. The
relationship and block-model tables use `28,544` finite log-ratio pairs because
one selected record has \(R=0\), for which \(\log R\) is undefined.

A follow-up 100-replicate run with the same cases and seed writes
`candidate_equation_holdout.csv`. It trains each candidate equation on held-out
replicate folds and leave-one-case-out folds, then scores held-out selected
records. This is a transfer diagnostic only; it does not define an external
calibration law.

A broader 200-replicate run adds diffuse dimensional Gaussian, high-cardinality
categorical, high-dimensional categorical, heavy-overlap binary, phylogenetic
DNA, and an SBM boundary case. Eight cases completed; `sbm_moderate` was
reported as an explicit skip because the current selected-hierarchy null
generator does not own the precomputed KL tree-distance contract. The broad run
adds leave-one-source-family-out scoring.

## Key Points

- The row-level mathematical object is
  \[
  R_u=\frac{W_u}{a_u\nu_u}
  \]
  together with the null-whitened sibling contrast \(z_u\), selected PCA basis
  \(V_k\), and the angular mass
  \[
  \cos^2\theta_u=\frac{\lVert V_k^\top z_u\rVert^2}{\lVert z_u\rVert^2}.
  \]
- Selected-ratio scale remains large in all four representative cases:
  mean \(R\) is about `30.9` for `gauss_clear_medium`, `54.9` for
  `cat_clear_3cat_4c`, `57.5` for `binary_low_noise_4c`, and `62.9` for
  `gauss_null_large`.
- The selected sibling contrast is usually highly aligned with the selected
  spectral subspace. Mean selected-subspace \(\cos^2\theta\) is about `0.83`
  to `0.92` by case, and the q95 is `1.0` in all four cases.
- Univariate relationship diagnostics identify edge-selection strength as the
  dominant recorded correlate of \(\log R\). The Spearman correlation between
  `negative_log10_min_child_edge_bh_p_value` and log selected-ratio is about
  `0.712`.
- Spectral variables are secondary but visible: selected eigenvalue mass has
  Spearman correlation about `-0.436`, selected eigenvalue over the MP upper
  bound about `0.399`, and parent test projection dimension about `0.302`.
- Tree geometry is weaker in this run. Parent-size fraction has Spearman
  correlation about `-0.193`; branch-length asymmetry about `0.179`; branch
  length sum about `-0.147`.
- In-sample descriptive block models explain limited variance: edge-selection
  \(R^2 \approx 0.052\), spectral \(R^2 \approx 0.079\), angular
  \(R^2 \approx 0.037\), tree geometry \(R^2 \approx 0.024\), and
  spectrum-plus-angle \(R^2 \approx 0.121\). These are descriptive summaries,
  not calibration models.
- Candidate equation scoring separates mean behavior from upper-tail behavior.
  For mean \(\log R\), the full descriptive candidate has the largest
  in-sample \(R^2\), about `0.188`. For the top-10% tail, the full descriptive
  candidate has AUC about `0.969`, and the smaller edge-plus-spectral equation
  is nearly the same at about `0.969`. This suggests the upper tail may be
  governed mainly by edge action plus spectral mode variables, while angular
  and sampling variables add mean-shape information.
- The current best compact tail candidate is
  \[
  \log R_u
  \sim
  A_u+
  \log(\lambda_{k,u}/\lambda_{+,u})+
  m_{k,u}+
  r_{\mathrm{eff},u},
  \]
  where \(A_u\) is edge-selection action, \(\lambda_{k,u}/\lambda_{+,u}\) is
  selected-eigenvalue excess over the MP edge, \(m_{k,u}\) is selected
  eigenvalue mass, and \(r_{\mathrm{eff},u}\) is effective spectral rank.
- Held-out replicate folds preserve the same tail-ranking result: the compact
  edge-plus-spectral equation has top-10% tail AUC about `0.969` and holdout
  \(R^2 \approx 0.104\); the full descriptive equation has AUC about `0.969`
  and holdout \(R^2 \approx 0.181\).
- Leave-one-case-out transfer is weaker but still informative. The compact
  edge-plus-spectral equation has AUC about `0.922` and holdout
  \(R^2 \approx 0.198\), while the full descriptive equation has AUC about
  `0.877` and holdout \(R^2 \approx 0.184\). Sampling/energy-only candidates
  degrade more strongly, so the compact edge-plus-spectral equation is the
  most stable current candidate family.
- In the broader 200-replicate panel, selected-ratio scale varies much more
  strongly by family. Mean \(R\) is about `30.9` for `gauss_clear_medium`,
  `62.9` for `gauss_null_large`, `89.4` for `cat_highcard_20cat_4c`,
  `109.3` for `dim_diffuse_6c_136f`, `160.5` for
  `overlap_heavy_4c_med_feat`, `210.1` for `phylo_dna_8taxa_med_mut`, and
  `579.8` for `cat_highd_3cat_500feat`.
- Broad-panel holdout separates tail ranking from mean-scale prediction.
  Replicate-fold holdout gives the full descriptive equation \(R^2\approx
  0.429\) and AUC about `0.9996`; the selected-energy candidate has the best
  replicate-fold tail AUC, about `0.9999`. Leave-one-source-family-out gives
  the full descriptive equation the best \(R^2\), about `0.458`, but its tail
  AUC drops to about `0.944`. Simpler edge/spectral equations keep tail AUCs
  near `0.998`--`0.999` but fit absolute scale worse.
- The broader run therefore does not validate a production equation. It says
  edge-selection severity is a very strong tail-ranking coordinate, while
  absolute selected-ratio calibration depends on family/context scale and
  needs a proper selected-tail law.
- The diagnostic can optionally write the full row table. The compact evidence
  keeps only summaries by default; the block model uses one canonical
  edge-strength coordinate,
  `negative_log10_min_child_edge_bh_p_value`, to avoid duplicate min/max/raw/BH
  aliases in the model design.

## Evidence

- `benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py`
  implements the diagnostic-only row builder, relationship summaries, and
  block-model summaries.
- `tests/validation/53_test_selected_hierarchy_geometry_covariates.py`
  validates relationship summaries, block-model status reporting, case
  summaries, output writing, and missing-predictor errors.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/manifest.json`
  records seed `20260602`, 100 replicates, the four case names, and the
  diagnostic-only role.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/geometry_summary_by_case.csv`
  records selected-ratio and angular/eigenvalue summaries by case.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/covariate_relationships.csv`
  records descriptive univariate Pearson/Spearman relationships.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/covariate_block_models.csv`
  records in-sample descriptive block-model summaries.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_100/candidate_equations.csv`
  records nested candidate equations for mean log-ratio fit and top-tail
  discrimination.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_holdout_100/candidate_equation_holdout.csv`
  records held-out replicate and leave-one-case-out transfer scores for the
  same candidate equations.
- `raw/assets/benchmark-results/selected_hierarchy_geometry_covariates_20260602_broad_200/`
  records the broad 200-replicate panel with source-family holdout and the SBM
  precomputed-distance boundary skip.

## Links

- [[selected-hierarchy-selection-geometry]]
- [[selected-hierarchy-null-support-contract]]
- [[selected-hierarchy-external-calibration-contract-20260602]]
- [[selected-hierarchy-stratification-diagnostic-20260602]]
- [[open-mathematical-questions]]
